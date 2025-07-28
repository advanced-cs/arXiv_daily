# 计算机视觉 cs.CV

- **最新发布 118 篇**

- **更新 62 篇**

## 最新发布

#### [new 001] Video Self-Distillation for Single-Image Encoders: A Step Toward Physically Plausible Perception
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决单图像编码器缺乏时序和三维空间信息的问题。通过使用视频自蒸馏方法，利用单帧预测下一帧表示，引入时空先验，提升图像理解性能。预训练后在ADE20K上mIoU提升至36.4，适用于物理可信感知模型。**

- **链接: [http://arxiv.org/pdf/2507.19272v1](http://arxiv.org/pdf/2507.19272v1)**

> **作者:** Marcel Simon; Tae-Ho Kim; Seul-Ki Yeom
>
> **备注:** 4 pages, 2 figures, 2 tables
>
> **摘要:** Self-supervised image encoders such as DINO have recently gained significant interest for learning robust visual features without labels. However, most SSL methods train on static images and miss the temporal cues inherent in videos. We introduce a video-distilled single-image encoder trained to predict the next-frame representation from the current frame. This simple objective injects 3D spatial and temporal priors without optical flow or tracking. When pre-training on a single 2-hour video, our approach raises the mean Intersection-over-Union (mIoU) on ADE20K from 35.0 (DoRA) to 36.4 while remaining a drop-in replacement for image-only pipelines. Our results highlight video self-distillation as a lightweight route to geometry-aware perception an essential ingredient for physically plausible world models and Physical AI.
>
---
#### [new 002] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D数据生成任务，旨在解决大规模3D数据稀缺的问题。作者提出了一种可扩展的2D到3D数据提升方法，通过整合深度估计、相机标定和尺度标定，将单视角2D图像转换为包含点云、深度图等的3D表示，降低了3D数据采集成本，并推动空间智能的发展。**

- **链接: [http://arxiv.org/pdf/2507.18678v1](http://arxiv.org/pdf/2507.18678v1)**

> **作者:** Xingyu Miao; Haoran Duan; Quanhao Qian; Jiuniu Wang; Yang Long; Ling Shao; Deli Zhao; Ran Xu; Gongjie Zhang
>
> **备注:** ICCV 2025 (Highlight)
>
> **摘要:** Spatial intelligence is emerging as a transformative frontier in AI, yet it remains constrained by the scarcity of large-scale 3D datasets. Unlike the abundant 2D imagery, acquiring 3D data typically requires specialized sensors and laborious annotation. In this work, we present a scalable pipeline that converts single-view images into comprehensive, scale- and appearance-realistic 3D representations - including point clouds, camera poses, depth maps, and pseudo-RGBD - via integrated depth estimation, camera calibration, and scale calibration. Our method bridges the gap between the vast repository of imagery and the increasing demand for spatial scene understanding. By automatically generating authentic, scale-aware 3D data from images, we significantly reduce data collection costs and open new avenues for advancing spatial intelligence. We release two generated spatial datasets, i.e., COCO-3D and Objects365-v2-3D, and demonstrate through extensive experiments that our generated data can benefit various 3D tasks, ranging from fundamental perception to MLLM-based reasoning. These results validate our pipeline as an effective solution for developing AI systems capable of perceiving, understanding, and interacting with physical environments.
>
---
#### [new 003] Diffusion-FS: Multimodal Free-Space Prediction via Diffusion for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的可行驶区域预测任务，旨在解决现有方法难以准确估计可导航通道的问题。论文提出Diffusion-FS，通过扩散模型结合轮廓点去噪（ContourDiff），实现基于单目图像的自由空间走廊预测，提升了预测的准确性与结构化程度。**

- **链接: [http://arxiv.org/pdf/2507.18763v1](http://arxiv.org/pdf/2507.18763v1)**

> **作者:** Keshav Gupta; Tejas S. Stanley; Pranjal Paul; Arun K. Singh; K. Madhava Krishna
>
> **备注:** 8 pages, 7 figures, IROS 2025
>
> **摘要:** Drivable Free-space prediction is a fundamental and crucial problem in autonomous driving. Recent works have addressed the problem by representing the entire non-obstacle road regions as the free-space. In contrast our aim is to estimate the driving corridors that are a navigable subset of the entire road region. Unfortunately, existing corridor estimation methods directly assume a BEV-centric representation, which is hard to obtain. In contrast, we frame drivable free-space corridor prediction as a pure image perception task, using only monocular camera input. However such a formulation poses several challenges as one doesn't have the corresponding data for such free-space corridor segments in the image. Consequently, we develop a novel self-supervised approach for free-space sample generation by leveraging future ego trajectories and front-view camera images, making the process of visual corridor estimation dependent on the ego trajectory. We then employ a diffusion process to model the distribution of such segments in the image. However, the existing binary mask-based representation for a segment poses many limitations. Therefore, we introduce ContourDiff, a specialized diffusion-based architecture that denoises over contour points rather than relying on binary mask representations, enabling structured and interpretable free-space predictions. We evaluate our approach qualitatively and quantitatively on both nuScenes and CARLA, demonstrating its effectiveness in accurately predicting safe multimodal navigable corridors in the image.
>
---
#### [new 004] PTCMIL: Multiple Instance Learning via Prompt Token Clustering for Whole Slide Image Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决全切片图像（WSI）分析中多实例学习（MIL）的复杂性和异质性问题。作者提出PTCMIL方法，通过引入可学习提示标记和端到端的聚类与预测框架，提升WSI表示学习效果，实现更高效的分类和生存分析。**

- **链接: [http://arxiv.org/pdf/2507.18848v1](http://arxiv.org/pdf/2507.18848v1)**

> **作者:** Beidi Zhao; SangMook Kim; Hao Chen; Chen Zhou; Zu-hua Gao; Gang Wang; Xiaoxiao Li
>
> **摘要:** Multiple Instance Learning (MIL) has advanced WSI analysis but struggles with the complexity and heterogeneity of WSIs. Existing MIL methods face challenges in aggregating diverse patch information into robust WSI representations. While ViTs and clustering-based approaches show promise, they are computationally intensive and fail to capture task-specific and slide-specific variability. To address these limitations, we propose PTCMIL, a novel Prompt Token Clustering-based ViT for MIL aggregation. By introducing learnable prompt tokens into the ViT backbone, PTCMIL unifies clustering and prediction tasks in an end-to-end manner. It dynamically aligns clustering with downstream tasks, using projection-based clustering tailored to each WSI, reducing complexity while preserving patch heterogeneity. Through token merging and prototype-based pooling, PTCMIL efficiently captures task-relevant patterns. Extensive experiments on eight datasets demonstrate its superior performance in classification and survival analysis tasks, outperforming state-of-the-art methods. Systematic ablation studies confirm its robustness and strong interpretability. The code is released at https://github.com/ubc-tea/PTCMIL.
>
---
#### [new 005] RemoteReasoner: Towards Unifying Geospatial Reasoning Workflow
- **分类: cs.CV**

- **简介: 该论文属于遥感图像处理任务，旨在解决现有方法依赖监督微调、缺乏推理自主性的问题。作者提出了RemoteReasoner，一个基于多模态大语言模型的统一推理框架，通过强化学习训练，实现多粒度输出和多样任务推理，提升了遥感图像的空间上下文理解和用户意图解析能力。**

- **链接: [http://arxiv.org/pdf/2507.19280v1](http://arxiv.org/pdf/2507.19280v1)**

> **作者:** Liang Yao; Fan Liu; Hongbo Lu; Chuanyi Zhang; Rui Min; Shengxiang Xu; Shimin Di; Pai Peng
>
> **摘要:** Remote sensing imagery presents vast, inherently unstructured spatial data, demanding sophisticated reasoning to interpret complex user intents and contextual relationships beyond simple recognition tasks. In this paper, we aim to construct an Earth observation workflow to handle complex queries by reasoning about spatial context and user intent. As a reasoning workflow, it should be somewhat autonomous, where predefined ground-truth reasoning paths do not constrain the learning process. Furthermore, its architecture ought to be unified yet flexible, enabling the model to perform diverse reasoning tasks with distinct output formats through a single forward pass. Existing remote sensing approaches fail to address these requirements, as they rely on supervised fine-tuning paradigms that constrain the autonomy of reasoning. To this end, we propose RemoteReasoner, a flexible and robust workflow for remote sensing reasoning tasks. The design of RemoteReasoner integrates a multi-modal large language model (MLLM) for interpreting user instructions and localizing targets, together with task adaptation strategies that enable multi-granularity output generation. In contrast to existing methods, our framework is trained with reinforcement learning (RL) to endow the MLLM sufficient autonomy for precise reasoning. At the inference stage, our adaptation strategies enable diverse output formats at inference time without requiring task-specific decoders or further fine-tuning. Preliminary experiments demonstrated that RemoteReasoner achieves remarkable performance across multi-granularity reasoning tasks, including region-level and pixel-level. Additionally, our framework enables novel capabilities such as the contour extraction task beyond the reach of existing reasoning pipelines.
>
---
#### [new 006] Multi-Task Dense Prediction Fine-Tuning with Mixture of Fine-Grained Experts
- **分类: cs.CV**

- **简介: 该论文属于多任务学习（MTL）中的密集预测任务，旨在解决多任务学习中共享表示与任务特化之间的平衡问题。论文提出了FGMoE架构，包含任务内专家、共享专家和全局专家，结合微调方法，在减少参数的同时提升性能。实验表明其在NYUD-v2和PASCAL-Context数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.19077v1](http://arxiv.org/pdf/2507.19077v1)**

> **作者:** Yangyang Xu; Xi Ye; Duo Su
>
> **备注:** Accepted to ACM Multimedia 2025 (MM'25)
>
> **摘要:** Multi-task learning (MTL) for dense prediction has shown promising results but still faces challenges in balancing shared representations with task-specific specialization. In this paper, we introduce a novel Fine-Grained Mixture of Experts (FGMoE) architecture that explores MoE-based MTL models through a combination of three key innovations and fine-tuning. First, we propose intra-task experts that partition along intermediate hidden dimensions of MLPs, enabling finer decomposition of task information while maintaining parameter efficiency. Second, we introduce shared experts that consolidate common information across different contexts of the same task, reducing redundancy, and allowing routing experts to focus on unique aspects. Third, we design a global expert that facilitates adaptive knowledge transfer across tasks based on both input feature and task requirements, promoting beneficial information sharing while preventing harmful interference. In addition, we use the fine-tuning approach to improve parameter efficiency only by training the parameters of the decoder. Extensive experimental results show that the proposed FGMoE uses fewer parameters and significantly outperforms current MoE-based competitive MTL models on two dense prediction datasets (\textit{i.e.,} NYUD-v2, PASCAL-Context) in various metrics.
>
---
#### [new 007] PRE-MAP: Personalized Reinforced Eye-tracking Multimodal LLM for High-Resolution Multi-Attribute Point Prediction
- **分类: cs.CV**

- **简介: 该论文属于视觉注意力预测任务，旨在解决现有模型忽视主观认知差异、低分辨率预测及多点预测不准的问题。作者提出了PRE-MAP模型与大规模数据集SPA-ADV，并引入C-GRPO优化策略，以提升多属性个性化眼动预测的准确性。**

- **链接: [http://arxiv.org/pdf/2507.19213v1](http://arxiv.org/pdf/2507.19213v1)**

> **作者:** Hanbing Wu; Ping Jiang; Anyang Su; Chenxu Zhao; Tianyu Fu; Minghui Wu; Beiping Tan; Huiying Li
>
> **摘要:** Visual selective attention, driven by individual preferences, regulates human prioritization of visual stimuli by bridging subjective cognitive mechanisms with objective visual elements, thereby steering the semantic interpretation and hierarchical processing of dynamic visual scenes. However, existing models and datasets predominantly neglect the influence of subjective cognitive diversity on fixation behavior. Conventional saliency prediction models, typically employing segmentation approaches, rely on low-resolution imagery to generate saliency heatmaps, subsequently upscaled to native resolutions, which limiting their capacity to capture personalized attention patterns. Furthermore, MLLMs are constrained by factors such as hallucinations, making it very costly to strictly adhere to the expected format in tasks involving multiple point predictions, and achieving precise point positioning is challenging. To address these limitations, we present Subjective Personalized Attention for Advertisement Videos, namely SPA-ADV, a large-scale multimodal dataset capturing gaze behaviors from over 4,500 participants varying in age and gender with 486 videos. Furthermore, we propose PRE-MAP, a novel eye-tracking saliency model that characterizes Personalized visual disparities through Reinforcement learning-optimized Eye-tracking, built upon MLLMs and guided by Multi-Attribute user profiles to predict Points. To ensure MLLMs produce prediction points that are both format-correct and spatially accurate, we introduce Consistency Group Relative Policy Optimization (C-GRPO), inspired by the variability in eye movement points and Multi-Attribute profiles. Extensive experiments on SPA-ADV and other benchmarks demonstrate the effectiveness of our approach. The code and dataset are available at \href{https://github.com/mininglamp-MLLM/PRE-MAP}{this URL}.
>
---
#### [new 008] LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences
- **分类: cs.CV; cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于图像描述生成任务，旨在解决当前评估方法缺乏统一标准、社会偏见考量及用户偏好整合的问题。作者构建了LOTUS排行榜，全面评估描述质量、风险及社会偏见，并引入用户偏好导向的评价标准。**

- **链接: [http://arxiv.org/pdf/2507.19362v1](http://arxiv.org/pdf/2507.19362v1)**

> **作者:** Yusuke Hirota; Boyi Li; Ryo Hachiuma; Yueh-Hua Wu; Boris Ivanovic; Yuta Nakashima; Marco Pavone; Yejin Choi; Yu-Chiang Frank Wang; Chao-Han Huck Yang
>
> **备注:** Accepted to ACL 2025. Leaderboard: huggingface.co/spaces/nvidia/lotus-vlm-bias-leaderboard
>
> **摘要:** Large Vision-Language Models (LVLMs) have transformed image captioning, shifting from concise captions to detailed descriptions. We introduce LOTUS, a leaderboard for evaluating detailed captions, addressing three main gaps in existing evaluations: lack of standardized criteria, bias-aware assessments, and user preference considerations. LOTUS comprehensively evaluates various aspects, including caption quality (e.g., alignment, descriptiveness), risks (\eg, hallucination), and societal biases (e.g., gender bias) while enabling preference-oriented evaluations by tailoring criteria to diverse user preferences. Our analysis of recent LVLMs reveals no single model excels across all criteria, while correlations emerge between caption detail and bias risks. Preference-oriented evaluations demonstrate that optimal model selection depends on user priorities.
>
---
#### [new 009] Revisiting DETR for Small Object Detection via Noise-Resilient Query Optimization
- **分类: cs.CV**

- **简介: 该论文属于小目标检测任务，旨在解决Transformer检测器在特征金字塔网络中对噪声敏感及查询质量低的问题。论文提出了一种噪声鲁棒查询优化方法（NRQO），包括噪声容忍的特征金字塔网络（NT-FPN）和基于相似性的区域建议网络（PS-RPN），以提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.19059v1](http://arxiv.org/pdf/2507.19059v1)**

> **作者:** Xiaocheng Fang; Jieyi Cai; Huanyu Liu; Wenxiu Cai; Yishu Liu; Bingzhi Chen
>
> **备注:** 2025 IEEE International Conference on Multimedia and Expo (ICME)
>
> **摘要:** Despite advancements in Transformer-based detectors for small object detection (SOD), recent studies show that these detectors still face challenges due to inherent noise sensitivity in feature pyramid networks (FPN) and diminished query quality in existing label assignment strategies. In this paper, we propose a novel Noise-Resilient Query Optimization (NRQO) paradigm, which innovatively incorporates the Noise-Tolerance Feature Pyramid Network (NT-FPN) and the Pairwise-Similarity Region Proposal Network (PS-RPN). Specifically, NT-FPN mitigates noise during feature fusion in FPN by preserving spatial and semantic information integrity. Unlike existing label assignment strategies, PS-RPN generates a sufficient number of high-quality positive queries by enhancing anchor-ground truth matching through position and shape similarities, without the need for additional hyperparameters. Extensive experiments on multiple benchmarks consistently demonstrate the superiority of NRQO over state-of-the-art baselines.
>
---
#### [new 010] Preserving Topological and Geometric Embeddings for Point Cloud Recovery
- **分类: cs.CV**

- **简介: 该论文属于点云恢复任务，旨在解决现有方法难以有效结合拓扑与几何特征的问题。作者提出了TopGeoFormer架构，通过拓扑嵌入、InterTwining注意力机制及几何与拓扑损失函数，在采样与重建阶段同时优化结构与细节，提升了点云恢复效果。**

- **链接: [http://arxiv.org/pdf/2507.19121v1](http://arxiv.org/pdf/2507.19121v1)**

> **作者:** Kaiyue Zhou; Zelong Tan; Hongxiao Wang; Ya-li Li; Shengjin Wang
>
> **摘要:** Recovering point clouds involves the sequential process of sampling and restoration, yet existing methods struggle to effectively leverage both topological and geometric attributes. To address this, we propose an end-to-end architecture named \textbf{TopGeoFormer}, which maintains these critical features throughout the sampling and restoration phases. First, we revisit traditional feature extraction techniques to yield topological embedding using a continuous mapping of relative relationships between neighboring points, and integrate it in both phases for preserving the structure of the original space. Second, we propose the \textbf{InterTwining Attention} to fully merge topological and geometric embeddings, which queries shape with local awareness in both phases to form a learnable shape context facilitated with point-wise, point-shape-wise, and intra-shape features. Third, we introduce a full geometry loss and a topological constraint loss to optimize the embeddings in both Euclidean and topological spaces. The geometry loss uses inconsistent matching between coarse-to-fine generations and targets for reconstructing better geometric details, and the constraint loss limits embedding variances for better approximation of the topological space. In experiments, we comprehensively analyze the circumstances using the conventional and learning-based sampling/upsampling algorithms. The quantitative and qualitative results demonstrate that our method significantly outperforms existing sampling and recovery methods.
>
---
#### [new 011] EA-ViT: Efficient Adaptation for Elastic Vision Transformer
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决视觉Transformer（ViT）在不同资源约束下部署需重复训练的问题。提出了EA-ViT框架，通过嵌套弹性架构和课程学习策略，实现单次训练生成多规模模型，并设计轻量路由器选择合适子模型，提升部署效率与适应性。**

- **链接: [http://arxiv.org/pdf/2507.19360v1](http://arxiv.org/pdf/2507.19360v1)**

> **作者:** Chen Zhu; Wangbo Zhao; Huiwen Zhang; Samir Khaki; Yuhao Zhou; Weidong Tang; Shuo Wang; Zhihang Yuan; Yuzhang Shang; Xiaojiang Peng; Kai Wang; Dawei Yang
>
> **备注:** Published as a conference paper at ICCV 2025
>
> **摘要:** Vision Transformers (ViTs) have emerged as a foundational model in computer vision, excelling in generalization and adaptation to downstream tasks. However, deploying ViTs to support diverse resource constraints typically requires retraining multiple, size-specific ViTs, which is both time-consuming and energy-intensive. To address this issue, we propose an efficient ViT adaptation framework that enables a single adaptation process to generate multiple models of varying sizes for deployment on platforms with various resource constraints. Our approach comprises two stages. In the first stage, we enhance a pre-trained ViT with a nested elastic architecture that enables structural flexibility across MLP expansion ratio, number of attention heads, embedding dimension, and network depth. To preserve pre-trained knowledge and ensure stable adaptation, we adopt a curriculum-based training strategy that progressively increases elasticity. In the second stage, we design a lightweight router to select submodels according to computational budgets and downstream task demands. Initialized with Pareto-optimal configurations derived via a customized NSGA-II algorithm, the router is then jointly optimized with the backbone. Extensive experiments on multiple benchmarks demonstrate the effectiveness and versatility of EA-ViT. The code is available at https://github.com/zcxcf/EA-ViT.
>
---
#### [new 012] PatchTraj: Dynamic Patch Representation Learning for Time-Frequency Trajectory Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于行人轨迹预测任务，旨在解决现有方法在建模人类运动动态时局部细节与长程时空依赖建模不足、且缺乏时频交互的问题。论文提出PatchTraj，通过动态分块机制融合时域与频域表示，利用自适应嵌入与跨模态注意力提升轨迹预测性能。**

- **链接: [http://arxiv.org/pdf/2507.19119v1](http://arxiv.org/pdf/2507.19119v1)**

> **作者:** Yanghong Liu; Xingping Dong; Ming Li; Weixing Zhang; Yidong Lou
>
> **摘要:** Pedestrian trajectory prediction is crucial for autonomous driving and robotics. While existing point-based and grid-based methods expose two key limitations: insufficiently modeling human motion dynamics, as they fail to balance local motion details with long-range spatiotemporal dependencies, and the time representation lacks interaction with the frequency domain in modeling trajectory sequences. To address these challenges, we propose PatchTraj, a dynamic patch-based trajectory prediction framework that unifies time-domain and frequency-domain representations. Specifically, we decompose the trajectory into raw time sequences and frequency components, employing dynamic patch partitioning for multi-scale trajectory segmentation to capture hierarchical motion patterns. Each patch is processed by an adaptive embedding layer with scale-aware feature extraction, followed by hierarchical feature aggregation to model both fine-grained and long-range dependencies. The outputs of two branches interact via cross-modal attention, enabling complementary fusion of temporal and spectral cues. Finally, a Transformer encoder-decoder integrates both modalities to autoregressively predict future trajectories. Extensive experiments on ETH-UCY, SDD, NBA, and JRDB datasets demonstrate that our method achieves state-of-the-art performance with high efficiency.
>
---
#### [new 013] MMBench-GUI: Hierarchical Multi-Platform Evaluation Framework for GUI Agents
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于GUI自动化评估任务，旨在解决跨平台GUI代理评估与效率问题。作者提出了MMBench-GUI基准框架，包含四个评估层级，并设计了EQA指标衡量效率。研究发现视觉定位与模块化框架对任务成功至关重要，强调长期推理与高效策略的必要性。**

- **链接: [http://arxiv.org/pdf/2507.19478v1](http://arxiv.org/pdf/2507.19478v1)**

> **作者:** Xuehui Wang; Zhenyu Wu; JingJing Xie; Zichen Ding; Bowen Yang; Zehao Li; Zhaoyang Liu; Qingyun Li; Xuan Dong; Zhe Chen; Weiyun Wang; Xiangyu Zhao; Jixuan Chen; Haodong Duan; Tianbao Xie; Chenyu Yang; Shiqian Su; Yue Yu; Yuan Huang; Yiqian Liu; Xiao Zhang; Yanting Zhang; Xiangyu Yue; Weijie Su; Xizhou Zhu; Wei Shen; Jifeng Dai; Wenhai Wang
>
> **备注:** in progress
>
> **摘要:** We introduce MMBench-GUI, a hierarchical benchmark for evaluating GUI automation agents across Windows, macOS, Linux, iOS, Android, and Web platforms. It comprises four levels: GUI Content Understanding, Element Grounding, Task Automation, and Task Collaboration, covering essential skills for GUI agents. In addition, we propose a novel Efficiency-Quality Area (EQA) metric to assess GUI agent execution efficiency in online automation scenarios. Through MMBench-GUI, we identify accurate visual grounding as a critical determinant of overall task success, emphasizing the substantial benefits of modular frameworks that integrate specialized grounding modules. Furthermore, to achieve reliable GUI automation, an agent requires strong task planning and cross-platform generalization abilities, with long-context memory, a broad action space, and long-term reasoning playing a critical role. More important, task efficiency remains a critically underexplored dimension, and all models suffer from substantial inefficiencies, with excessive redundant steps even when tasks are ultimately completed. The integration of precise localization, effective planning, and early stopping strategies is indispensable to enable truly efficient and scalable GUI automation. Our benchmark code, evaluation data, and running environment will be publicly available at https://github.com/open-compass/MMBench-GUI.
>
---
#### [new 014] KuiSCIMA v2.0: Improved Baselines, Calibration, and Cross-Notation Generalization for Historical Chinese Music Notations in Jiang Kui's Baishidaoren Gequ
- **分类: cs.CV; cs.DL; cs.SD; eess.AS**

- **简介: 该论文属于光学音乐识别（OMR）任务，旨在解决历史中文乐谱识别中数据稀缺、类别不平衡的问题。论文改进了基线模型，提升了识别准确率与模型校准，并扩展了数据集以增强跨版本泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.18741v1](http://arxiv.org/pdf/2507.18741v1)**

> **作者:** Tristan Repolusk; Eduardo Veas
>
> **备注:** International Conference on Document Analysis and Recognition. This preprint has not undergone any post-submission improvements or corrections. The Version of Record of this contribution is published in "19th International Conference on Document Analysis and Recognition (ICDAR 2025), Wuhan, China, September 16-21, 2025, Proceedings", and is available online at the External DOI field below
>
> **摘要:** Optical Music Recognition (OMR) for historical Chinese musical notations, such as suzipu and l\"ul\"upu, presents unique challenges due to high class imbalance and limited training data. This paper introduces significant advancements in OMR for Jiang Kui's influential collection Baishidaoren Gequ from 1202. In this work, we develop and evaluate a character recognition model for scarce imbalanced data. We improve upon previous baselines by reducing the Character Error Rate (CER) from 10.4% to 7.1% for suzipu, despite working with 77 highly imbalanced classes, and achieve a remarkable CER of 0.9% for l\"ul\"upu. Our models outperform human transcribers, with an average human CER of 15.9% and a best-case CER of 7.6%. We employ temperature scaling to achieve a well-calibrated model with an Expected Calibration Error (ECE) below 0.0162. Using a leave-one-edition-out cross-validation approach, we ensure robust performance across five historical editions. Additionally, we extend the KuiSCIMA dataset to include all 109 pieces from Baishidaoren Gequ, encompassing suzipu, l\"ul\"upu, and jianzipu notations. Our findings advance the digitization and accessibility of historical Chinese music, promoting cultural diversity in OMR and expanding its applicability to underrepresented music traditions.
>
---
#### [new 015] ShrinkBox: Backdoor Attack on Object Detection to Disrupt Collision Avoidance in Machine Learning-based Advanced Driver Assistance Systems
- **分类: cs.CV**

- **简介: 论文研究在基于机器学习的高级驾驶辅助系统（ML-ADAS）中，针对目标检测的后门攻击。任务是对象检测与距离估计。要解决的问题是系统在碰撞避免中的安全性漏洞。工作提出了ShrinkBox攻击方法，通过缩小边界框破坏距离估计，实验证明其在YOLOv9m上攻击成功率高，严重影响系统安全性。**

- **链接: [http://arxiv.org/pdf/2507.18656v1](http://arxiv.org/pdf/2507.18656v1)**

> **作者:** Muhammad Zaeem Shahzad; Muhammad Abdullah Hanif; Bassem Ouni; Muhammad Shafique
>
> **备注:** 8 pages, 8 figures, 1 table
>
> **摘要:** Advanced Driver Assistance Systems (ADAS) significantly enhance road safety by detecting potential collisions and alerting drivers. However, their reliance on expensive sensor technologies such as LiDAR and radar limits accessibility, particularly in low- and middle-income countries. Machine learning-based ADAS (ML-ADAS), leveraging deep neural networks (DNNs) with only standard camera input, offers a cost-effective alternative. Critical to ML-ADAS is the collision avoidance feature, which requires the ability to detect objects and estimate their distances accurately. This is achieved with specialized DNNs like YOLO, which provides real-time object detection, and a lightweight, detection-wise distance estimation approach that relies on key features extracted from the detections like bounding box dimensions and size. However, the robustness of these systems is undermined by security vulnerabilities in object detectors. In this paper, we introduce ShrinkBox, a novel backdoor attack targeting object detection in collision avoidance ML-ADAS. Unlike existing attacks that manipulate object class labels or presence, ShrinkBox subtly shrinks ground truth bounding boxes. This attack remains undetected in dataset inspections and standard benchmarks while severely disrupting downstream distance estimation. We demonstrate that ShrinkBox can be realized in the YOLOv9m object detector at an Attack Success Rate (ASR) of 96%, with only a 4% poisoning ratio in the training instances of the KITTI dataset. Furthermore, given the low error targets introduced in our relaxed poisoning strategy, we find that ShrinkBox increases the Mean Absolute Error (MAE) in downstream distance estimation by more than 3x on poisoned samples, potentially resulting in delays or prevention of collision warnings altogether.
>
---
#### [new 016] DASH: 4D Hash Encoding with Self-Supervised Decomposition for Real-Time Dynamic Scene Rendering
- **分类: cs.CV**

- **简介: 该论文属于动态场景重建任务，旨在解决现有方法因低秩假设导致的特征重叠和渲染质量差问题。论文提出DASH框架，采用4D哈希编码与自监督分解，实现动态与静态成分分离，提升渲染质量与速度，达到实时264 FPS。**

- **链接: [http://arxiv.org/pdf/2507.19141v1](http://arxiv.org/pdf/2507.19141v1)**

> **作者:** Jie Chen; Zhangchi Hu; Peixi Wu; Huyue Zhu; Hebei Li; Xiaoyan Sun
>
> **摘要:** Dynamic scene reconstruction is a long-term challenge in 3D vision. Existing plane-based methods in dynamic Gaussian splatting suffer from an unsuitable low-rank assumption, causing feature overlap and poor rendering quality. Although 4D hash encoding provides an explicit representation without low-rank constraints, directly applying it to the entire dynamic scene leads to substantial hash collisions and redundancy. To address these challenges, we present DASH, a real-time dynamic scene rendering framework that employs 4D hash encoding coupled with self-supervised decomposition. Our approach begins with a self-supervised decomposition mechanism that separates dynamic and static components without manual annotations or precomputed masks. Next, we introduce a multiresolution 4D hash encoder for dynamic elements, providing an explicit representation that avoids the low-rank assumption. Finally, we present a spatio-temporal smoothness regularization strategy to mitigate unstable deformation artifacts. Experiments on real-world datasets demonstrate that DASH achieves state-of-the-art dynamic rendering performance, exhibiting enhanced visual quality at real-time speeds of 264 FPS on a single 4090 GPU. Code: https://github.com/chenj02/DASH.
>
---
#### [new 017] Cross-Subject Mind Decoding from Inaccurate Representations
- **分类: cs.CV**

- **简介: 该论文属于脑信号解码任务，旨在解决跨受试者解码fMRI信号为刺激图像时因个体差异导致的误差累积问题。作者提出了一种双向自编码框架，结合偏置调制、语义优化和视觉一致性模块，提升了跨受试者的解码准确性和适应性。**

- **链接: [http://arxiv.org/pdf/2507.19071v1](http://arxiv.org/pdf/2507.19071v1)**

> **作者:** Yangyang Xu; Bangzhen Liu; Wenqi Shao; Yong Du; Shengfeng He; Tingting Zhu
>
> **摘要:** Decoding stimulus images from fMRI signals has advanced with pre-trained generative models. However, existing methods struggle with cross-subject mappings due to cognitive variability and subject-specific differences. This challenge arises from sequential errors, where unidirectional mappings generate partially inaccurate representations that, when fed into diffusion models, accumulate errors and degrade reconstruction fidelity. To address this, we propose the Bidirectional Autoencoder Intertwining framework for accurate decoded representation prediction. Our approach unifies multiple subjects through a Subject Bias Modulation Module while leveraging bidirectional mapping to better capture data distributions for precise representation prediction. To further enhance fidelity when decoding representations into stimulus images, we introduce a Semantic Refinement Module to improve semantic representations and a Visual Coherence Module to mitigate the effects of inaccurate visual representations. Integrated with ControlNet and Stable Diffusion, our method outperforms state-of-the-art approaches on benchmark datasets in both qualitative and quantitative evaluations. Moreover, our framework exhibits strong adaptability to new subjects with minimal training samples.
>
---
#### [new 018] MedIQA: A Scalable Foundation Model for Prompt-Driven Medical Image Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像质量评估任务，旨在解决现有方法在多模态、多场景下泛化能力不足的问题。作者提出了MedIQA模型，构建了大规模多模态数据集，引入显著切片评估模块和自动提示策略，提升了图像质量评估的准确性和适用范围。**

- **链接: [http://arxiv.org/pdf/2507.19004v1](http://arxiv.org/pdf/2507.19004v1)**

> **作者:** Siyi Xun; Yue Sun; Jingkun Chen; Zitong Yu; Tong Tong; Xiaohong Liu; Mingxiang Wu; Tao Tan
>
> **备注:** We note that the version after peer review of this paper has been provisionally accepted by The 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2025)
>
> **摘要:** Rapid advances in medical imaging technology underscore the critical need for precise and automated image quality assessment (IQA) to ensure diagnostic accuracy. Existing medical IQA methods, however, struggle to generalize across diverse modalities and clinical scenarios. In response, we introduce MedIQA, the first comprehensive foundation model for medical IQA, designed to handle variability in image dimensions, modalities, anatomical regions, and types. We developed a large-scale multi-modality dataset with plentiful manually annotated quality scores to support this. Our model integrates a salient slice assessment module to focus on diagnostically relevant regions feature retrieval and employs an automatic prompt strategy that aligns upstream physical parameter pre-training with downstream expert annotation fine-tuning. Extensive experiments demonstrate that MedIQA significantly outperforms baselines in multiple downstream tasks, establishing a scalable framework for medical IQA and advancing diagnostic workflows and clinical decision-making.
>
---
#### [new 019] SAR-TEXT: A Large-Scale SAR Image-Text Dataset Built with SAR-Narrator and Progressive Transfer Learning
- **分类: cs.CV**

- **简介: 该论文属于遥感图像与自然语言处理的交叉任务，旨在解决缺乏大规模高质量SAR图像-文本数据的问题。作者构建了SAR-Text数据集（超13万对），并提出SAR-Narrator框架生成文本描述，通过多阶段渐进式迁移学习策略训练模型。实验验证了其在图像检索、描述生成和视觉问答任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.18743v1](http://arxiv.org/pdf/2507.18743v1)**

> **作者:** Xinjun Cheng; Yiguo He; Junjie Zhu; Chunping Qiu; Jun Wang; Qiangjuan Huang; Ke Yang
>
> **备注:** IEEE Submission
>
> **摘要:** Vision Language Models (VLMs) have achieved remarkable breakthroughs in the field of remote sensing in recent years. Synthetic Aperture Radar (SAR) imagery, with its all-weather capability, is essential in remote sensing, yet the lack of large-scale, high-quality SAR image-text datasets hinders its semantic understanding. In this paper, we construct SAR-Text, a large-scale and high-quality dataset consisting of over 130,000 SAR image-text pairs. To construct the SAR-Text dataset, we design the SAR-Narrator framework, which generates textual descriptions for SAR images through a multi-stage progressive transfer learning strategy. To verify the effectiveness of the SAR-TEXT dataset, we conduct experiments on three typical vision-language tasks: image-text retrieval, image captioning, and visual question answering (VQA). Specifically, we construct three representative models on SAR-TEXT: SAR-RS-CLIP, SAR-RS-CoCa, and SAR-GPT. SAR-RS-CLIP achieves notable improvements in retrieval performance, boosting average recall by 16.43% and 10.54% on the OSdataset-512 and HRSID test sets, respectively. In the captioning task, SAR-RS-CoCa achieves BLEU-4, SPICE, and CIDEr scores exceeding those of the original CoCa model by more than 8x, 4x, and 10x, respectively. In the VQA task, SAR-GPT outperforms baseline and single-stage models on multiple SAR-VQA datasets, demonstrating stronger semantic understanding and reasoning ability, as further confirmed by qualitative results. It is worth noting that, as a flexible captioning tool, SAR-Narrator can be readily adopted by the community to construct larger-scale SAR image-text datasets.
>
---
#### [new 020] Perspective from a Higher Dimension: Can 3D Geometric Priors Help Visual Floorplan Localization?
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决视觉定位与平面图之间的模态与几何差异问题。通过引入3D几何先验，利用多视角约束和场景表面重建，提升定位准确性。方法无需额外标注，采用自监督对比学习，有效桥接模态差异，显著提高定位效果。**

- **链接: [http://arxiv.org/pdf/2507.18881v1](http://arxiv.org/pdf/2507.18881v1)**

> **作者:** Bolei Chen; Jiaxu Kang; Haonan Yang; Ping Zhong; Jianxin Wang
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Since a building's floorplans are easily accessible, consistent over time, and inherently robust to changes in visual appearance, self-localization within the floorplan has attracted researchers' interest. However, since floorplans are minimalist representations of a building's structure, modal and geometric differences between visual perceptions and floorplans pose challenges to this task. While existing methods cleverly utilize 2D geometric features and pose filters to achieve promising performance, they fail to address the localization errors caused by frequent visual changes and view occlusions due to variously shaped 3D objects. To tackle these issues, this paper views the 2D Floorplan Localization (FLoc) problem from a higher dimension by injecting 3D geometric priors into the visual FLoc algorithm. For the 3D geometric prior modeling, we first model geometrically aware view invariance using multi-view constraints, i.e., leveraging imaging geometric principles to provide matching constraints between multiple images that see the same points. Then, we further model the view-scene aligned geometric priors, enhancing the cross-modal geometry-color correspondences by associating the scene's surface reconstruction with the RGB frames of the sequence. Both 3D priors are modeled through self-supervised contrastive learning, thus no additional geometric or semantic annotations are required. These 3D priors summarized in extensive realistic scenes bridge the modal gap while improving localization success without increasing the computational burden on the FLoc algorithm. Sufficient comparative studies demonstrate that our method significantly outperforms state-of-the-art methods and substantially boosts the FLoc accuracy. All data and code will be released after the anonymous review.
>
---
#### [new 021] Learned Image Compression with Hierarchical Progressive Context Modeling
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于图像压缩任务，旨在提升压缩性能与计算效率。针对现有方法难以有效利用长距离依赖和多阶段上下文信息的问题，论文提出了分层渐进上下文模型（HPCM），通过多尺度分层编码和渐进式上下文融合机制，更高效地获取和利用上下文信息，取得了先进的率失真性能。**

- **链接: [http://arxiv.org/pdf/2507.19125v1](http://arxiv.org/pdf/2507.19125v1)**

> **作者:** Yuqi Li; Haotian Zhang; Li Li; Dong Liu
>
> **备注:** 17 pages, ICCV 2025
>
> **摘要:** Context modeling is essential in learned image compression for accurately estimating the distribution of latents. While recent advanced methods have expanded context modeling capacity, they still struggle to efficiently exploit long-range dependency and diverse context information across different coding steps. In this paper, we introduce a novel Hierarchical Progressive Context Model (HPCM) for more efficient context information acquisition. Specifically, HPCM employs a hierarchical coding schedule to sequentially model the contextual dependencies among latents at multiple scales, which enables more efficient long-range context modeling. Furthermore, we propose a progressive context fusion mechanism that incorporates contextual information from previous coding steps into the current step, effectively exploiting diverse contextual information. Experimental results demonstrate that our method achieves state-of-the-art rate-distortion performance and strikes a better balance between compression performance and computational complexity. The code is available at https://github.com/lyq133/LIC-HPCM.
>
---
#### [new 022] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective
- **分类: cs.CV**

- **简介: 论文提出MixA-Q，一种混合精度激活量化框架，用于提升视觉Transformer推理效率。通过利用层内激活稀疏性，在不同窗口分配不同比特精度，优化性能与效率的平衡。方法兼容量化感知训练和后训练量化，实验表明其在无精度损失下显著提升计算速度，并减少量化误差影响。**

- **链接: [http://arxiv.org/pdf/2507.19131v1](http://arxiv.org/pdf/2507.19131v1)**

> **作者:** Weitian Wang; Rai Shubham; Cecilia De La Parra; Akash Kumar
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** In this paper, we propose MixA-Q, a mixed-precision activation quantization framework that leverages intra-layer activation sparsity (a concept widely explored in activation pruning methods) for efficient inference of quantized window-based vision transformers. For a given uniform-bit quantization configuration, MixA-Q separates the batched window computations within Swin blocks and assigns a lower bit width to the activations of less important windows, improving the trade-off between model performance and efficiency. We introduce a Two-Branch Swin Block that processes activations separately in high- and low-bit precision, enabling seamless integration of our method with most quantization-aware training (QAT) and post-training quantization (PTQ) methods, or with simple modifications. Our experimental evaluations over the COCO dataset demonstrate that MixA-Q achieves a training-free 1.35x computational speedup without accuracy loss in PTQ configuration. With QAT, MixA-Q achieves a lossless 1.25x speedup and a 1.53x speedup with only a 1% mAP drop by incorporating activation pruning. Notably, by reducing the quantization error in important regions, our sparsity-aware quantization adaptation improves the mAP of the quantized W4A4 model (with both weights and activations in 4-bit precision) by 0.7%, reducing quantization degradation by 24%.
>
---
#### [new 023] RealDeal: Enhancing Realism and Details in Brain Image Generation via Image-to-Image Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像生成任务，旨在解决现有脑部MRI图像生成模型输出过于平滑、缺乏真实细节的问题。作者提出了RealDeal方法，通过图像到图像扩散模型增强图像真实感与细节，并引入新指标评估生成效果。**

- **链接: [http://arxiv.org/pdf/2507.18830v1](http://arxiv.org/pdf/2507.18830v1)**

> **作者:** Shen Zhu; Yinzhu Jin; Tyler Spears; Ifrah Zawar; P. Thomas Fletcher
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** We propose image-to-image diffusion models that are designed to enhance the realism and details of generated brain images by introducing sharp edges, fine textures, subtle anatomical features, and imaging noise. Generative models have been widely adopted in the biomedical domain, especially in image generation applications. Latent diffusion models achieve state-of-the-art results in generating brain MRIs. However, due to latent compression, generated images from these models are overly smooth, lacking fine anatomical structures and scan acquisition noise that are typically seen in real images. This work formulates the realism enhancing and detail adding process as image-to-image diffusion models, which refines the quality of LDM-generated images. We employ commonly used metrics like FID and LPIPS for image realism assessment. Furthermore, we introduce new metrics to demonstrate the realism of images generated by RealDeal in terms of image noise distribution, sharpness, and texture.
>
---
#### [new 024] Gaussian Set Surface Reconstruction through Per-Gaussian Optimization
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决3D高斯点分布不均、几何重建不准确的问题。作者提出GSSR方法，通过像素级和高斯级法线一致性、多视角光度一致性优化高斯分布，并引入正则化损失和周期性重初始化，实现更精确的几何重建与场景编辑。**

- **链接: [http://arxiv.org/pdf/2507.18923v1](http://arxiv.org/pdf/2507.18923v1)**

> **作者:** Zhentao Huang; Di Wu; Zhenbang He; Minglun Gong
>
> **摘要:** 3D Gaussian Splatting (3DGS) effectively synthesizes novel views through its flexible representation, yet fails to accurately reconstruct scene geometry. While modern variants like PGSR introduce additional losses to ensure proper depth and normal maps through Gaussian fusion, they still neglect individual placement optimization. This results in unevenly distributed Gaussians that deviate from the latent surface, complicating both reconstruction refinement and scene editing. Motivated by pioneering work on Point Set Surfaces, we propose Gaussian Set Surface Reconstruction (GSSR), a method designed to distribute Gaussians evenly along the latent surface while aligning their dominant normals with the surface normal. GSSR enforces fine-grained geometric alignment through a combination of pixel-level and Gaussian-level single-view normal consistency and multi-view photometric consistency, optimizing both local and global perspectives. To further refine the representation, we introduce an opacity regularization loss to eliminate redundant Gaussians and apply periodic depth- and normal-guided Gaussian reinitialization for a cleaner, more uniform spatial distribution. Our reconstruction results demonstrate significantly improved geometric precision in Gaussian placement, enabling intuitive scene editing and efficient generation of novel Gaussian-based 3D environments. Extensive experiments validate GSSR's effectiveness, showing enhanced geometric accuracy while preserving high-quality rendering performance.
>
---
#### [new 025] HQ-SMem: Video Segmentation and Tracking Using Memory Efficient Object Embedding With Selective Update and Self-Supervised Distillation Feedback
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割（VOS）任务，旨在解决现有模型在长视频中分割精度低、拓扑变化适应差、跟踪漂移和内存效率低的问题。论文提出了HQ-SMem方法，通过引入高质量掩码优化、动态智能内存机制和动态外观模型更新，提升了分割与跟踪性能。**

- **链接: [http://arxiv.org/pdf/2507.18921v1](http://arxiv.org/pdf/2507.18921v1)**

> **作者:** Elham Soltani Kazemi; Imad Eddine Toubal; Gani Rahmon; Jaired Collins; K. Palaniappan
>
> **备注:** submit/6651762
>
> **摘要:** Video Object Segmentation (VOS) is foundational to numerous computer vision applications, including surveillance, autonomous driving, robotics and generative video editing. However, existing VOS models often struggle with precise mask delineation, deformable objects, topologically transforming objects, tracking drift and long video sequences. In this paper, we introduce HQ-SMem, for High Quality video segmentation and tracking using Smart Memory, a novel method that enhances the performance of VOS base models by addressing these limitations. Our approach incorporates three key innovations: (i) leveraging SAM with High-Quality masks (SAM-HQ) alongside appearance-based candidate-selection to refine coarse segmentation masks, resulting in improved object boundaries; (ii) implementing a dynamic smart memory mechanism that selectively stores relevant key frames while discarding redundant ones, thereby optimizing memory usage and processing efficiency for long-term videos; and (iii) dynamically updating the appearance model to effectively handle complex topological object variations and reduce drift throughout the video. These contributions mitigate several limitations of existing VOS models including, coarse segmentations that mix-in background pixels, fixed memory update schedules, brittleness to drift and occlusions, and prompt ambiguity issues associated with SAM. Extensive experiments conducted on multiple public datasets and state-of-the-art base trackers demonstrate that our method consistently ranks among the top two on VOTS and VOTSt 2024 datasets. Moreover, HQ-SMem sets new benchmarks on Long Video Dataset and LVOS, showcasing its effectiveness in challenging scenarios characterized by complex multi-object dynamics over extended temporal durations.
>
---
#### [new 026] ABCD: Automatic Blood Cell Detection via Attention-Guided Improved YOLOX
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决传统手动血细胞检测效率低、易出错的问题。作者提出了一种基于改进YOLOX的自动血细胞检测方法ABCD，引入CBAM增强特征提取，使用ASFF优化特征融合，并采用CIOU损失加速收敛。实验表明该方法在BCCD数据集上表现优异，具备高效实时检测能力。**

- **链接: [http://arxiv.org/pdf/2507.19296v1](http://arxiv.org/pdf/2507.19296v1)**

> **作者:** Ahmed Endris Hasen; Yang Shangming; Chiagoziem C. Ukwuoma; Biniyam Gashaw; Abel Zenebe Yutra
>
> **摘要:** Detection of blood cells in microscopic images has become a major focus of medical image analysis, playing a crucial role in gaining valuable insights into a patient's health. Manual blood cell checks for disease detection are known to be time-consuming, inefficient, and error-prone. To address these limitations, analyzing blood cells using deep learning-based object detectors can be regarded as a feasible solution. In this study, we propose automatic blood cell detection method (ABCD) based on an improved version of YOLOX, an object detector, for detecting various types of blood cells, including white blood cells, red blood cells, and platelets. Firstly, we introduce the Convolutional Block Attention Module (CBAM) into the network's backbone to enhance the efficiency of feature extraction. Furthermore, we introduce the Adaptively Spatial Feature Fusion (ASFF) into the network's neck, which optimizes the fusion of different features extracted from various stages of the network. Finally, to speed up the model's convergence, we substitute the Intersection over Union (IOU) loss function with the Complete Intersection over Union (CIOU) loss function. The experimental results demonstrate that the proposed method is more effective than other existing methods for BCCD dataset. Compared to the baseline algorithm, our method ABCD achieved 95.49 % mAP@0.5 and 86.89 % mAP@0.5-0.9, which are 2.8% and 23.41% higher, respectively, and increased the detection speed by 2.9%, making it highly efficient for real-time applications.
>
---
#### [new 027] Balancing Conservatism and Aggressiveness: Prototype-Affinity Hybrid Network for Few-Shot Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务中的小样本分割（Few-Shot Segmentation, FSS）问题，旨在通过少量标注样本提升模型对未见类别的分割效果。作者提出PAHNet方法，结合原型学习与亲和力学习的优势，通过特征增强和注意力校准模块，平衡保守与激进预测，提升分割准确性。实验表明该方法在多个数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.19140v1](http://arxiv.org/pdf/2507.19140v1)**

> **作者:** Tianyu Zou; Shengwu Xiong; Ruilin Yao; Yi Rong
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** This paper studies the few-shot segmentation (FSS) task, which aims to segment objects belonging to unseen categories in a query image by learning a model on a small number of well-annotated support samples. Our analysis of two mainstream FSS paradigms reveals that the predictions made by prototype learning methods are usually conservative, while those of affinity learning methods tend to be more aggressive. This observation motivates us to balance the conservative and aggressive information captured by these two types of FSS frameworks so as to improve the segmentation performance. To achieve this, we propose a **P**rototype-**A**ffinity **H**ybrid **Net**work (PAHNet), which introduces a Prototype-guided Feature Enhancement (PFE) module and an Attention Score Calibration (ASC) module in each attention block of an affinity learning model (called affinity learner). These two modules utilize the predictions generated by a pre-trained prototype learning model (called prototype predictor) to enhance the foreground information in support and query image representations and suppress the mismatched foreground-background (FG-BG) relationships between them, respectively. In this way, the aggressiveness of the affinity learner can be effectively mitigated, thereby eventually increasing the segmentation accuracy of our PAHNet method. Experimental results show that PAHNet outperforms most recently proposed methods across 1-shot and 5-shot settings on both PASCAL-5$^i$ and COCO-20$^i$ datasets, suggesting its effectiveness. The code is available at: [GitHub - tianyu-zou/PAHNet: Balancing Conservatism and Aggressiveness: Prototype-Affinity Hybrid Network for Few-Shot Segmentation (ICCV'25)](https://github.com/tianyu-zou/PAHNet)
>
---
#### [new 028] Fast Learning of Non-Cooperative Spacecraft 3D Models through Primitive Initialization
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于3D建模与计算机视觉任务，旨在解决从单目图像快速学习非合作航天器高精度3D模型的问题。论文提出了一种基于CNN的原始初始化器，结合3D高斯泼溅技术，减少了训练所需迭代次数和图像数量，并能在姿态估计不精确的情况下仍保持高性能，提升了空间应用的可行性。**

- **链接: [http://arxiv.org/pdf/2507.19459v1](http://arxiv.org/pdf/2507.19459v1)**

> **作者:** Pol Francesch Huc; Emily Bates; Simone D'Amico
>
> **摘要:** The advent of novel view synthesis techniques such as NeRF and 3D Gaussian Splatting (3DGS) has enabled learning precise 3D models only from posed monocular images. Although these methods are attractive, they hold two major limitations that prevent their use in space applications: they require poses during training, and have high computational cost at training and inference. To address these limitations, this work contributes: (1) a Convolutional Neural Network (CNN) based primitive initializer for 3DGS using monocular images; (2) a pipeline capable of training with noisy or implicit pose estimates; and (3) and analysis of initialization variants that reduce the training cost of precise 3D models. A CNN takes a single image as input and outputs a coarse 3D model represented as an assembly of primitives, along with the target's pose relative to the camera. This assembly of primitives is then used to initialize 3DGS, significantly reducing the number of training iterations and input images needed -- by at least an order of magnitude. For additional flexibility, the CNN component has multiple variants with different pose estimation techniques. This work performs a comparison between these variants, evaluating their effectiveness for downstream 3DGS training under noisy or implicit pose estimates. The results demonstrate that even with imperfect pose supervision, the pipeline is able to learn high-fidelity 3D representations, opening the door for the use of novel view synthesis in space applications.
>
---
#### [new 029] A New One-Shot Federated Learning Framework for Medical Imaging Classification with Feature-Guided Rectified Flow and Knowledge Distillation
- **分类: cs.CV; cs.DC**

- **简介: 该论文属于医学图像分类任务，旨在解决医疗数据隐私保护和非独立同分布（non-IID）下的一次通信联邦学习效率问题。作者提出了一种新的联邦学习框架，结合特征引导的修正流模型和双层知识蒸馏方法，在保护隐私的同时提升模型性能。实验表明该方法在多个医学图像数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.19045v1](http://arxiv.org/pdf/2507.19045v1)**

> **作者:** Yufei Ma; Hanwen Zhang; Qiya Yang; Guibo Luo; Yuesheng Zhu
>
> **备注:** Accepted at ECAI 2025
>
> **摘要:** In multi-center scenarios, One-Shot Federated Learning (OSFL) has attracted increasing attention due to its low communication overhead, requiring only a single round of transmission. However, existing generative model-based OSFL methods suffer from low training efficiency and potential privacy leakage in the healthcare domain. Additionally, achieving convergence within a single round of model aggregation is challenging under non-Independent and Identically Distributed (non-IID) data. To address these challenges, in this paper a modified OSFL framework is proposed, in which a new Feature-Guided Rectified Flow Model (FG-RF) and Dual-Layer Knowledge Distillation (DLKD) aggregation method are developed. FG-RF on the client side accelerates generative modeling in medical imaging scenarios while preserving privacy by synthesizing feature-level images rather than pixel-level images. To handle non-IID distributions, DLKD enables the global student model to simultaneously mimic the output logits and align the intermediate-layer features of client-side teacher models during aggregation. Experimental results on three non-IID medical imaging datasets show that our new framework and method outperform multi-round federated learning approaches, achieving up to 21.73% improvement, and exceeds the baseline FedISCA by an average of 21.75%. Furthermore, our experiments demonstrate that feature-level synthetic images significantly reduce privacy leakage risks compared to pixel-level synthetic images.
>
---
#### [new 030] ScenePainter: Semantically Consistent Perpetual 3D Scene Generation with Concept Relation Alignment
- **分类: cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决长期生成中语义漂移的问题。通过构建层次化场景概念图，对齐场景先验与理解，实现语义一致的连续视角生成，提升生成序列的连贯性和多样性。**

- **链接: [http://arxiv.org/pdf/2507.19058v1](http://arxiv.org/pdf/2507.19058v1)**

> **作者:** Chong Xia; Shengjun Zhang; Fangfu Liu; Chang Liu; Khodchaphun Hirunyaratsameewong; Yueqi Duan
>
> **摘要:** Perpetual 3D scene generation aims to produce long-range and coherent 3D view sequences, which is applicable for long-term video synthesis and 3D scene reconstruction. Existing methods follow a "navigate-and-imagine" fashion and rely on outpainting for successive view expansion. However, the generated view sequences suffer from semantic drift issue derived from the accumulated deviation of the outpainting module. To tackle this challenge, we propose ScenePainter, a new framework for semantically consistent 3D scene generation, which aligns the outpainter's scene-specific prior with the comprehension of the current scene. To be specific, we introduce a hierarchical graph structure dubbed SceneConceptGraph to construct relations among multi-level scene concepts, which directs the outpainter for consistent novel views and can be dynamically refined to enhance diversity. Extensive experiments demonstrate that our framework overcomes the semantic drift issue and generates more consistent and immersive 3D view sequences. Project Page: https://xiac20.github.io/ScenePainter/.
>
---
#### [new 031] SP-Mamba: Spatial-Perception State Space Model for Unsupervised Medical Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决无监督医学异常检测问题。针对传统CNN和Transformer模型的局限性，作者提出SP-Mamba模型，结合窗口滑动学习和空间信息建模，提升检测性能。实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.19076v1](http://arxiv.org/pdf/2507.19076v1)**

> **作者:** Rui Pan; Ruiying Lu
>
> **备注:** 11 pages
>
> **摘要:** Radiography imaging protocols target on specific anatomical regions, resulting in highly consistent images with recurrent structural patterns across patients. Recent advances in medical anomaly detection have demonstrated the effectiveness of CNN- and transformer-based approaches. However, CNNs exhibit limitations in capturing long-range dependencies, while transformers suffer from quadratic computational complexity. In contrast, Mamba-based models, leveraging superior long-range modeling, structural feature extraction, and linear computational efficiency, have emerged as a promising alternative. To capitalize on the inherent structural regularity of medical images, this study introduces SP-Mamba, a spatial-perception Mamba framework for unsupervised medical anomaly detection. The window-sliding prototype learning and Circular-Hilbert scanning-based Mamba are introduced to better exploit consistent anatomical patterns and leverage spatial information for medical anomaly detection. Furthermore, we excavate the concentration and contrast characteristics of anomaly maps for improving anomaly detection. Extensive experiments on three diverse medical anomaly detection benchmarks confirm the proposed method's state-of-the-art performance, validating its efficacy and robustness. The code is available at https://github.com/Ray-RuiPan/SP-Mamba.
>
---
#### [new 032] Dual Path Learning -- learning from noise and context for medical image denoising
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像去噪任务，旨在解决成像设备引入的噪声影响图像质量与临床诊断的问题。论文提出了一种双路径学习模型（DPL），融合噪声特征与图像上下文信息，提升去噪效果。实验表明其在多种成像模态与噪声类型中均表现优异，相较UNet提升3.35% PSNR。**

- **链接: [http://arxiv.org/pdf/2507.19035v1](http://arxiv.org/pdf/2507.19035v1)**

> **作者:** Jitindra Fartiyal; Pedro Freire; Yasmeen Whayeb; James S. Wolffsohn; Sergei K. Turitsyn; Sergei G. Sokolov
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Medical imaging plays a critical role in modern healthcare, enabling clinicians to accurately diagnose diseases and develop effective treatment plans. However, noise, often introduced by imaging devices, can degrade image quality, leading to misinterpretation and compromised clinical outcomes. Existing denoising approaches typically rely either on noise characteristics or on contextual information from the image. Moreover, they are commonly developed and evaluated for a single imaging modality and noise type. Motivated by Geng et.al CNCL, which integrates both noise and context, this study introduces a Dual-Pathway Learning (DPL) model architecture that effectively denoises medical images by leveraging both sources of information and fusing them to generate the final output. DPL is evaluated across multiple imaging modalities and various types of noise, demonstrating its robustness and generalizability. DPL improves PSNR by 3.35% compared to the baseline UNet when evaluated on Gaussian noise and trained across all modalities. The code is available at 10.5281/zenodo.15836053.
>
---
#### [new 033] Gen-AI Police Sketches with Stable Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在通过AI自动生成嫌疑人素描。论文使用Stable Diffusion模型构建三种方法，结合CLIP模型和LoRA微调技术，提升文本与图像的对齐效果。研究发现，仅使用Stable Diffusion的模型在结构相似性和图像质量上表现最佳。**

- **链接: [http://arxiv.org/pdf/2507.18667v1](http://arxiv.org/pdf/2507.18667v1)**

> **作者:** Nicholas Fidalgo; Aaron Contreras; Katherine Harvey; Johnny Ni
>
> **摘要:** This project investigates the use of multimodal AI-driven approaches to automate and enhance suspect sketching. Three pipelines were developed and evaluated: (1) baseline image-to-image Stable Diffusion model, (2) same model integrated with a pre-trained CLIP model for text-image alignment, and (3) novel approach incorporating LoRA fine-tuning of the CLIP model, applied to self-attention and cross-attention layers, and integrated with Stable Diffusion. An ablation study confirmed that fine-tuning both self- and cross-attention layers yielded the best alignment between text descriptions and sketches. Performance testing revealed that Model 1 achieved the highest structural similarity (SSIM) of 0.72 and a peak signal-to-noise ratio (PSNR) of 25 dB, outperforming Model 2 and Model 3. Iterative refinement enhanced perceptual similarity (LPIPS), with Model 3 showing improvement over Model 2 but still trailing Model 1. Qualitatively, sketches generated by Model 1 demonstrated the clearest facial features, highlighting its robustness as a baseline despite its simplicity.
>
---
#### [new 034] Transferable and Undefendable Point Cloud Attacks via Medial Axis Transform
- **分类: cs.CV**

- **简介: 该论文属于3D点云对抗攻击任务，旨在解决现有攻击方法在跨模型迁移性和对抗防御机制方面的不足。作者提出MAT-Adv，通过扰动点云的中轴变换（MAT）表示，生成具有结构级对抗特性的点云，提升攻击的可迁移性和逃避防御能力。实验表明该方法优于现有技术。**

- **链接: [http://arxiv.org/pdf/2507.18870v1](http://arxiv.org/pdf/2507.18870v1)**

> **作者:** Keke Tang; Yuze Gao; Weilong Peng; Xiaofei Wang; Meie Fang; Peican Zhu
>
> **摘要:** Studying adversarial attacks on point clouds is essential for evaluating and improving the robustness of 3D deep learning models. However, most existing attack methods are developed under ideal white-box settings and often suffer from limited transferability to unseen models and insufficient robustness against common defense mechanisms. In this paper, we propose MAT-Adv, a novel adversarial attack framework that enhances both transferability and undefendability by explicitly perturbing the medial axis transform (MAT) representations, in order to induce inherent adversarialness in the resulting point clouds. Specifically, we employ an autoencoder to project input point clouds into compact MAT representations that capture the intrinsic geometric structure of point clouds. By perturbing these intrinsic representations, MAT-Adv introduces structural-level adversarial characteristics that remain effective across diverse models and defense strategies. To mitigate overfitting and prevent perturbation collapse, we incorporate a dropout strategy into the optimization of MAT perturbations, further improving transferability and undefendability. Extensive experiments demonstrate that MAT-Adv significantly outperforms existing state-of-the-art methods in both transferability and undefendability. Codes will be made public upon paper acceptance.
>
---
#### [new 035] Dealing with Segmentation Errors in Needle Reconstruction for MRI-Guided Brachytherapy
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决MRI引导下近距离放疗中针重建的分割误差问题。作者改进了后处理技术，以提升自动针重建的准确性，实验显示其方法有效降低了定位误差并消除了误检。**

- **链接: [http://arxiv.org/pdf/2507.18895v1](http://arxiv.org/pdf/2507.18895v1)**

> **作者:** Vangelis Kostoulas; Arthur Guijt; Ellen M. Kerkhof; Bradley R. Pieters; Peter A. N. Bosman; Tanja Alderliesten
>
> **备注:** Published in: Proc. SPIE Medical Imaging 2025, Vol. 13408, 1340826
>
> **摘要:** Brachytherapy involves bringing a radioactive source near tumor tissue using implanted needles. Image-guided brachytherapy planning requires amongst others, the reconstruction of the needles. Manually annotating these needles on patient images can be a challenging and time-consuming task for medical professionals. For automatic needle reconstruction, a two-stage pipeline is commonly adopted, comprising a segmentation stage followed by a post-processing stage. While deep learning models are effective for segmentation, their results often contain errors. No currently existing post-processing technique is robust to all possible segmentation errors. We therefore propose adaptations to existing post-processing techniques mainly aimed at dealing with segmentation errors and thereby improving the reconstruction accuracy. Experiments on a prostate cancer dataset, based on MRI scans annotated by medical professionals, demonstrate that our proposed adaptations can help to effectively manage segmentation errors, with the best adapted post-processing technique achieving median needle-tip and needle-bottom point localization errors of $1.07$ (IQR $\pm 1.04$) mm and $0.43$ (IQR $\pm 0.46$) mm, respectively, and median shaft error of $0.75$ (IQR $\pm 0.69$) mm with 0 false positive and 0 false negative needles on a test set of 261 needles.
>
---
#### [new 036] DINO-SLAM: DINO-informed RGB-D SLAM for Neural Implicit and Explicit Representations
- **分类: cs.CV**

- **简介: 该论文属于SLAM任务，旨在提升RGB-D SLAM中的神经隐式（NeRF）与显式表示（3DGS）的场景重建效果。通过引入基于DINO特征的场景结构编码器（SSE），生成增强的EDINO特征，以更好捕捉场景结构。在此基础上，构建了两种结合EDINO的NeRF和3DGS SLAM框架，实验证明其在多个数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.19474v1](http://arxiv.org/pdf/2507.19474v1)**

> **作者:** Ziren Gong; Xiaohan Li; Fabio Tosi; Youmin Zhang; Stefano Mattoccia; Jun Wu; Matteo Poggi
>
> **摘要:** This paper presents DINO-SLAM, a DINO-informed design strategy to enhance neural implicit (Neural Radiance Field -- NeRF) and explicit representations (3D Gaussian Splatting -- 3DGS) in SLAM systems through more comprehensive scene representations. Purposely, we rely on a Scene Structure Encoder (SSE) that enriches DINO features into Enhanced DINO ones (EDINO) to capture hierarchical scene elements and their structural relationships. Building upon it, we propose two foundational paradigms for NeRF and 3DGS SLAM systems integrating EDINO features. Our DINO-informed pipelines achieve superior performance on the Replica, ScanNet, and TUM compared to state-of-the-art methods.
>
---
#### [new 037] DEFNet: Multitasks-based Deep Evidential Fusion Network for Blind Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于盲图像质量评估（BIQA）任务，旨在解决现有方法在多任务融合与不确定性估计方面的不足。作者提出DEFNet，通过多任务优化和可信赖的信息融合策略，结合局部-全局特征与不确定性估计，提升图像质量评估的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.19418v1](http://arxiv.org/pdf/2507.19418v1)**

> **作者:** Yiwei Lou; Yuanpeng He; Rongchao Zhang; Yongzhi Cao; Hanpin Wang; Yu Huang
>
> **摘要:** Blind image quality assessment (BIQA) methods often incorporate auxiliary tasks to improve performance. However, existing approaches face limitations due to insufficient integration and a lack of flexible uncertainty estimation, leading to suboptimal performance. To address these challenges, we propose a multitasks-based Deep Evidential Fusion Network (DEFNet) for BIQA, which performs multitask optimization with the assistance of scene and distortion type classification tasks. To achieve a more robust and reliable representation, we design a novel trustworthy information fusion strategy. It first combines diverse features and patterns across sub-regions to enhance information richness, and then performs local-global information fusion by balancing fine-grained details with coarse-grained context. Moreover, DEFNet exploits advanced uncertainty estimation technique inspired by evidential learning with the help of normal-inverse gamma distribution mixture. Extensive experiments on both synthetic and authentic distortion datasets demonstrate the effectiveness and robustness of the proposed framework. Additional evaluation and analysis are carried out to highlight its strong generalization capability and adaptability to previously unseen scenarios.
>
---
#### [new 038] VisHall3D: Monocular Semantic Scene Completion from Reconstructing the Visible Regions to Hallucinating the Invisible Regions
- **分类: cs.CV**

- **简介: 该论文属于单目语义场景补全任务，旨在解决现有方法中特征纠缠和几何不一致的问题。论文提出VisHall3D框架，分两阶段完成可见区域重建和不可见区域推理，通过VisFrontierNet和OcclusionMAE模块提升重建质量，取得了优异性能。**

- **链接: [http://arxiv.org/pdf/2507.19188v1](http://arxiv.org/pdf/2507.19188v1)**

> **作者:** Haoang Lu; Yuanqi Su; Xiaoning Zhang; Longjun Gao; Yu Xue; Le Wang
>
> **摘要:** This paper introduces VisHall3D, a novel two-stage framework for monocular semantic scene completion that aims to address the issues of feature entanglement and geometric inconsistency prevalent in existing methods. VisHall3D decomposes the scene completion task into two stages: reconstructing the visible regions (vision) and inferring the invisible regions (hallucination). In the first stage, VisFrontierNet, a visibility-aware projection module, is introduced to accurately trace the visual frontier while preserving fine-grained details. In the second stage, OcclusionMAE, a hallucination network, is employed to generate plausible geometries for the invisible regions using a noise injection mechanism. By decoupling scene completion into these two distinct stages, VisHall3D effectively mitigates feature entanglement and geometric inconsistency, leading to significantly improved reconstruction quality. The effectiveness of VisHall3D is validated through extensive experiments on two challenging benchmarks: SemanticKITTI and SSCBench-KITTI-360. VisHall3D achieves state-of-the-art performance, outperforming previous methods by a significant margin and paves the way for more accurate and reliable scene understanding in autonomous driving and other applications.
>
---
#### [new 039] SIDE: Sparse Information Disentanglement for Explainable Artificial Intelligence
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于可解释人工智能任务，旨在解决深度神经网络在计算机视觉中缺乏透明性的问题。论文提出SIDE方法，通过稀疏信息解耦和修剪机制，显著减少原型解释规模，提升模型可解释性。**

- **链接: [http://arxiv.org/pdf/2507.19321v1](http://arxiv.org/pdf/2507.19321v1)**

> **作者:** Viktar Dubovik; Łukasz Struski; Jacek Tabor; Dawid Rymarczyk
>
> **摘要:** Understanding the decisions made by deep neural networks is essential in high-stakes domains such as medical imaging and autonomous driving. Yet, these models often lack transparency, particularly in computer vision. Prototypical-parts-based neural networks have emerged as a promising solution by offering concept-level explanations. However, most are limited to fine-grained classification tasks, with few exceptions such as InfoDisent. InfoDisent extends prototypical models to large-scale datasets like ImageNet, but produces complex explanations. We introduce Sparse Information Disentanglement for Explainability (SIDE), a novel method that improves the interpretability of prototypical parts through a dedicated training and pruning scheme that enforces sparsity. Combined with sigmoid activations in place of softmax, this approach allows SIDE to associate each class with only a small set of relevant prototypes. Extensive experiments show that SIDE matches the accuracy of existing methods while reducing explanation size by over $90\%$, substantially enhancing the understandability of prototype-based explanations.
>
---
#### [new 040] Back to the Features: DINO as a Foundation for Video World Models
- **分类: cs.CV**

- **简介: 该论文属于视频预测任务，旨在解决视频世界模型的构建问题。作者提出DINO-world，基于DINOv2的预训练图像编码器，训练未来帧预测模型，学习多种场景的时间动态，实现了优越的视频预测性能，并支持基于动作的规划模拟。**

- **链接: [http://arxiv.org/pdf/2507.19468v1](http://arxiv.org/pdf/2507.19468v1)**

> **作者:** Federico Baldassarre; Marc Szafraniec; Basile Terver; Vasil Khalidov; Francisco Massa; Yann LeCun; Patrick Labatut; Maximilian Seitzer; Piotr Bojanowski
>
> **摘要:** We present DINO-world, a powerful generalist video world model trained to predict future frames in the latent space of DINOv2. By leveraging a pre-trained image encoder and training a future predictor on a large-scale uncurated video dataset, DINO-world learns the temporal dynamics of diverse scenes, from driving and indoor scenes to simulated environments. We show that DINO-world outperforms previous models on a variety of video prediction benchmarks, e.g. segmentation and depth forecasting, and demonstrates strong understanding of intuitive physics. Furthermore, we show that it is possible to fine-tune the predictor on observation-action trajectories. The resulting action-conditioned world model can be used for planning by simulating candidate trajectories in latent space.
>
---
#### [new 041] VGS-ATD: Robust Distributed Learning for Multi-Label Medical Image Classification Under Heterogeneous and Imbalanced Conditions
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于医疗图像分类任务，旨在解决分布式学习中数据异构、不平衡及灾难性遗忘问题。作者提出了VGS-ATD框架，实验证明其在多标签、多节点环境下具有高准确性、强扩展性和低计算成本，优于集中式和现有分布式方法。**

- **链接: [http://arxiv.org/pdf/2507.18657v1](http://arxiv.org/pdf/2507.18657v1)**

> **作者:** Zehui Zhao; Laith Alzubaidi; Haider A. Alwzwazy; Jinglan Zhang; Yuantong Gu
>
> **备注:** 15 pages, 8 figures, 6 tables
>
> **摘要:** In recent years, advanced deep learning architectures have shown strong performance in medical imaging tasks. However, the traditional centralized learning paradigm poses serious privacy risks as all data is collected and trained on a single server. To mitigate this challenge, decentralized approaches such as federated learning and swarm learning have emerged, allowing model training on local nodes while sharing only model weights. While these methods enhance privacy, they struggle with heterogeneous and imbalanced data and suffer from inefficiencies due to frequent communication and the aggregation of weights. More critically, the dynamic and complex nature of clinical environments demands scalable AI systems capable of continuously learning from diverse modalities and multilabels. Yet, both centralized and decentralized models are prone to catastrophic forgetting during system expansion, often requiring full model retraining to incorporate new data. To address these limitations, we propose VGS-ATD, a novel distributed learning framework. To validate VGS-ATD, we evaluate it in experiments spanning 30 datasets and 80 independent labels across distributed nodes, VGS-ATD achieved an overall accuracy of 92.7%, outperforming centralized learning (84.9%) and swarm learning (72.99%), while federated learning failed under these conditions due to high requirements on computational resources. VGS-ATD also demonstrated strong scalability, with only a 1% drop in accuracy on existing nodes after expansion, compared to a 20% drop in centralized learning, highlighting its resilience to catastrophic forgetting. Additionally, it reduced computational costs by up to 50% relative to both centralized and swarm learning, confirming its superior efficiency and scalability.
>
---
#### [new 042] Eyes Will Shut: A Vision-Based Next GPS Location Prediction Model by Reinforcement Learning from Visual Map Feed Back
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于轨迹预测任务，旨在解决基于地图的下一个位置预测问题。现有方法缺乏人类般的地图推理能力。作者提出VLMLocPredictor，结合视觉语言模型与强化学习，通过视觉地图反馈进行自我改进，实现更准确的预测。**

- **链接: [http://arxiv.org/pdf/2507.18661v1](http://arxiv.org/pdf/2507.18661v1)**

> **作者:** Ruixing Zhang; Yang Zhang; Tongyu Zhu; Leilei Sun; Weifeng Lv
>
> **摘要:** Next Location Prediction is a fundamental task in the study of human mobility, with wide-ranging applications in transportation planning, urban governance, and epidemic forecasting. In practice, when humans attempt to predict the next location in a trajectory, they often visualize the trajectory on a map and reason based on road connectivity and movement trends. However, the vast majority of existing next-location prediction models do not reason over maps \textbf{in the way that humans do}. Fortunately, the recent development of Vision-Language Models (VLMs) has demonstrated strong capabilities in visual perception and even visual reasoning. This opens up a new possibility: by rendering both the road network and trajectory onto an image and leveraging the reasoning abilities of VLMs, we can enable models to perform trajectory inference in a human-like manner. To explore this idea, we first propose a method called Vision-Guided Location Search (VGLS), which evaluates whether a general-purpose VLM is capable of trajectory-based reasoning without modifying any of its internal parameters. Based on insights from the VGLS results, we further propose our main approach: VLMLocPredictor, which is composed of two stages: In the first stage, we design two Supervised Fine-Tuning (SFT) tasks that help the VLM understand road network and trajectory structures and acquire basic reasoning ability on such visual inputs. In the second stage, we introduce Reinforcement Learning from Visual Map Feedback, enabling the model to self-improve its next-location prediction ability through interaction with the environment. Experiments conducted on datasets from four different cities show that our method achieves state-of-the-art (SOTA) performance and exhibits superior cross-city generalization compared to other LLM-based approaches.
>
---
#### [new 043] MedSymmFlow: Bridging Generative Modeling and Classification in Medical Imaging through Symmetrical Flow Matching
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决分类准确性、不确定性估计及生成能力的统一建模问题。作者提出MedSymmFlow，结合生成与判别模型优势，通过对称流匹配实现分类、生成与不确定性量化一体化框架，提升诊断可靠性与临床适用性。**

- **链接: [http://arxiv.org/pdf/2507.19098v1](http://arxiv.org/pdf/2507.19098v1)**

> **作者:** Francisco Caetano; Lemar Abdi; Christiaan Viviers; Amaan Valiuddin; Fons van der Sommen
>
> **备注:** DGM4MICCAI 2025
>
> **摘要:** Reliable medical image classification requires accurate predictions and well-calibrated uncertainty estimates, especially in high-stakes clinical settings. This work presents MedSymmFlow, a generative-discriminative hybrid model built on Symmetrical Flow Matching, designed to unify classification, generation, and uncertainty quantification in medical imaging. MedSymmFlow leverages a latent-space formulation that scales to high-resolution inputs and introduces a semantic mask conditioning mechanism to enhance diagnostic relevance. Unlike standard discriminative models, it naturally estimates uncertainty through its generative sampling process. The model is evaluated on four MedMNIST datasets, covering a range of modalities and pathologies. The results show that MedSymmFlow matches or exceeds the performance of established baselines in classification accuracy and AUC, while also delivering reliable uncertainty estimates validated by performance improvements under selective prediction.
>
---
#### [new 044] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality
- **分类: cs.CV**

- **简介: 该论文提出SimMLM框架，用于解决多模态学习中模态缺失的问题。属于多模态医学图像分割和分类任务。其核心工作是设计动态模态专家架构与MoFe排序损失，使模型在不同模态输入下保持高精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.19264v1](http://arxiv.org/pdf/2507.19264v1)**

> **作者:** Sijie Li; Chen Chen; Jungong Han
>
> **摘要:** In this paper, we propose SimMLM, a simple yet powerful framework for multimodal learning with missing modalities. Unlike existing approaches that rely on sophisticated network architectures or complex data imputation techniques, SimMLM provides a generic and effective solution that can adapt to various missing modality scenarios with improved accuracy and robustness. Specifically, SimMLM consists of a generic Dynamic Mixture of Modality Experts (DMoME) architecture, featuring a dynamic, learnable gating mechanism that automatically adjusts each modality's contribution in both full and partial modality settings. A key innovation of SimMLM is the proposed More vs. Fewer (MoFe) ranking loss, which ensures that task accuracy improves or remains stable as more modalities are made available. This aligns the model with an intuitive principle: removing one or more modalities should not increase accuracy. We validate SimMLM on multimodal medical image segmentation (BraTS 2018) and multimodal classification (UPMC Food-101, avMNIST) tasks, where it consistently surpasses competitive methods, demonstrating superior accuracy, interpretability, robustness, and reliability across both complete and missing modality scenarios at test time.
>
---
#### [new 045] Efficient Lines Detection for Robot Soccer
- **分类: cs.CV; cs.RO**

- **简介: 论文属于机器人视觉任务，旨在解决足球场上线条检测问题。为实现机器人自定位，作者改进ELSED算法，加入RGB颜色过渡分类步骤，并采用PSO优化阈值，提升检测效率与准确率，适用于低功耗平台实时应用。**

- **链接: [http://arxiv.org/pdf/2507.19469v1](http://arxiv.org/pdf/2507.19469v1)**

> **作者:** João G. Melo; João P. Mafaldo; Edna Barros
>
> **备注:** 12 pages, 8 figures, RoboCup Symposium 2025
>
> **摘要:** Self-localization is essential in robot soccer, where accurate detection of visual field features, such as lines and boundaries, is critical for reliable pose estimation. This paper presents a lightweight and efficient method for detecting soccer field lines using the ELSED algorithm, extended with a classification step that analyzes RGB color transitions to identify lines belonging to the field. We introduce a pipeline based on Particle Swarm Optimization (PSO) for threshold calibration to optimize detection performance, requiring only a small number of annotated samples. Our approach achieves accuracy comparable to a state-of-the-art deep learning model while offering higher processing speed, making it well-suited for real-time applications on low-power robotic platforms.
>
---
#### [new 046] CircuitProbe: Dissecting Spatiotemporal Visual Semantics with Circuit Tracing
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型的可解释性分析任务，旨在探究模型对时空视觉语义的理解机制。论文提出CircuitProbe框架，通过三个电路分析发现视觉语义高度集中在特定对象标记，并揭示模型中层到深层对时空语义的特化处理。**

- **链接: [http://arxiv.org/pdf/2507.19420v1](http://arxiv.org/pdf/2507.19420v1)**

> **作者:** Yiming Zhang; Chengzhang Yu; Zhuokai Zhao; Kun Wang; Qiankun Li; Zihan Chen; Yang Liu; Zenghui Ding; Yining Sun
>
> **摘要:** The processing mechanisms underlying language and image understanding in large vision-language models (LVLMs) have been extensively studied. However, the internal reasoning mechanisms of LVLMs for spatiotemporal understanding remain poorly understood. In this work, we introduce a systematic, circuit-based framework designed to investigate how spatiotemporal visual semantics are represented and processed within these LVLMs. Specifically, our framework comprises three circuits: visual auditing circuit, semantic tracing circuit, and attention flow circuit. Through the lens of these circuits, we discover that visual semantics are highly localized to specific object tokens--removing these tokens can degrade model performance by up to 92.6%. Furthermore, we identify that interpretable concepts of objects and actions emerge and become progressively refined in the middle-to-late layers of LVLMs. In contrary to the current works that solely focus on objects in one image, we reveal that the middle-to-late layers of LVLMs exhibit specialized functional localization for spatiotemporal semantics. Our findings offer significant mechanistic insights into spatiotemporal semantics analysis of LVLMs, laying a foundation for designing more robust and interpretable models.
>
---
#### [new 047] Tell Me What You See: An Iterative Deep Learning Framework for Image Captioning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像描述生成任务，旨在解决如何让模型准确理解图像并生成自然语言描述的问题。论文系统地迭代开发了多个模型，从基础的CNN-LSTM到带注意力机制的模型，最终提出的Nexus模型结合EfficientNetV2B3和动态注意力机制，在MS COCO数据集上取得优异表现，验证了注意力机制的重要性。**

- **链接: [http://arxiv.org/pdf/2507.18788v1](http://arxiv.org/pdf/2507.18788v1)**

> **作者:** Hitesh Kumar Gupta
>
> **备注:** 16 pages, 12 total figures (including a 7-figure appendix), 4 tables
>
> **摘要:** Image captioning, a task at the confluence of computer vision and natural language processing, requires a sophisticated understanding of both visual scenes and linguistic structure. While modern approaches are dominated by large-scale Transformer architectures, this paper documents a systematic, iterative development of foundational image captioning models, progressing from a simple CNN-LSTM encoder-decoder to a competitive attention-based system. We present a series of five models, beginning with Genesis and concluding with Nexus, an advanced model featuring an EfficientNetV2B3 backbone and a dynamic attention mechanism. Our experiments chart the impact of architectural enhancements and demonstrate a key finding within the classic CNN-LSTM paradigm: merely upgrading the visual backbone without a corresponding attention mechanism can degrade performance, as the single-vector bottleneck cannot transmit the richer visual detail. This insight validates the architectural shift to attention. Trained on the MS COCO 2017 dataset, our final model, Nexus, achieves a BLEU-4 score of 31.4, surpassing several foundational benchmarks and validating our iterative design process. This work provides a clear, replicable blueprint for understanding the core architectural principles that underpin modern vision-language tasks.
>
---
#### [new 048] Quantum-Cognitive Tunnelling Neural Networks for Military-Civilian Vehicle Classification and Sentiment Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人工智能与量子计算交叉任务，旨在解决军事与民用图像分类及情感分析问题。作者采用基于量子隧穿概率的神经网络模型，探索其在战场环境下的多模态应用，提升无人机作战中的人机协同推理能力。**

- **链接: [http://arxiv.org/pdf/2507.18645v1](http://arxiv.org/pdf/2507.18645v1)**

> **作者:** Milan Maksimovic; Anna Bohdanets; Immaculate Motsi-Omoijiade; Guido Governatori; Ivan S. Maksymov
>
> **摘要:** Prior work has demonstrated that incorporating well-known quantum tunnelling (QT) probability into neural network models effectively captures important nuances of human perception, particularly in the recognition of ambiguous objects and sentiment analysis. In this paper, we employ novel QT-based neural networks and assess their effectiveness in distinguishing customised CIFAR-format images of military and civilian vehicles, as well as sentiment, using a proprietary military-specific vocabulary. We suggest that QT-based models can enhance multimodal AI applications in battlefield scenarios, particularly within human-operated drone warfare contexts, imbuing AI with certain traits of human reasoning.
>
---
#### [new 049] Modality Agnostic Efficient Long Range Encoder
- **分类: cs.CV**

- **简介: 论文提出了一种通用、高效的长程编码器MAELRE，适用于多模态任务。它旨在解决长上下文处理中计算和内存复杂度高的问题，通过结合令牌合并与注意力近似方法，优化单设备上的长序列处理，兼顾准确性与效率。**

- **链接: [http://arxiv.org/pdf/2507.19409v1](http://arxiv.org/pdf/2507.19409v1)**

> **作者:** Toufiq Parag; Ahmed Elgammal
>
> **摘要:** The long-context capability of recent large transformer models can be surmised to rely on techniques such as attention/model parallelism, as well as hardware-level optimizations. While these strategies allow input lengths to scale to millions of tokens, they do not fundamentally mitigate the quadratic computational and memory complexity of the core attention mechanism. In this paper, we address the challenge of long-context processing on a single device using generic implementations by reducing the quadratic memory footprint and inference cost. Existing approaches to extend the context length for generic single device implementations -- such as token merging and modified attentions -- are often modality specific and attain a suboptimal tradeoff between accuracy and efficiency. To overcome these limitations, we propose MAELRE (Modality Agnostic Efficient Long Range Encoder), a unified and efficient transformer architecture designed for long-range encoding across diverse modalities. MAELRE integrates token merging with attention approximation, progressively merging tokens at different stages of internal computational blocks. It employs a lightweight attention approximation when the number of tokens is large, and switches to standard dot-product attention as the sequence becomes shorter through successive aggregation. We demonstrate that MAELRE achieves superior accuracy while reducing computational cost compared to existing long-context models on classification tasks spanning multiple modalities, including text, time series, audio, and vision.
>
---
#### [new 050] Advancing Vision-based Human Action Recognition: Exploring Vision-Language CLIP Model for Generalisation in Domain-Independent Tasks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言模型在动作识别中的应用任务，旨在解决传统模型在复杂动作中泛化能力不足的问题。论文评估了CLIP模型在不同遮蔽策略下的表现，提出通过添加类别特定噪声来增强模型对关键特征的关注，从而提高分类准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.18675v1](http://arxiv.org/pdf/2507.18675v1)**

> **作者:** Sanyam Jain; Marsha Mariya Kappan; Vijeta Sharma
>
> **摘要:** Human action recognition plays a critical role in healthcare and medicine, supporting applications such as patient behavior monitoring, fall detection, surgical robot supervision, and procedural skill assessment. While traditional models like CNNs and RNNs have achieved moderate success, they often struggle to generalize across diverse and complex actions. Recent advancements in vision-language models, especially the transformer-based CLIP model, offer promising capabilities for generalizing action recognition from video data. In this work, we evaluate CLIP on the UCF-101 dataset and systematically analyze its performance under three masking strategies: (1) percentage-based and shape-based black masking at 10%, 30%, and 50%, (2) feature-specific masking to suppress bias-inducing elements, and (3) isolation masking that retains only class-specific regions. Our results reveal that CLIP exhibits inconsistent behavior and frequent misclassifications, particularly when essential visual cues are obscured. To overcome these limitations, we propose incorporating class-specific noise, learned via a custom loss function, to reinforce attention to class-defining features. This enhancement improves classification accuracy and model confidence while reducing bias. We conclude with a discussion on the challenges of applying such models in clinical domains and outline directions for future work to improve generalizability across domain-independent healthcare scenarios.
>
---
#### [new 051] Adapt, But Don't Forget: Fine-Tuning and Contrastive Routing for Lane Detection under Distribution Shift
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于车道检测任务，旨在解决分布偏移下的灾难性遗忘问题。通过分阶段训练与分支微调，结合对比学习动态路由，实现参数高效适应，保持模型在多分布下的高性能。**

- **链接: [http://arxiv.org/pdf/2507.18653v1](http://arxiv.org/pdf/2507.18653v1)**

> **作者:** Mohammed Abdul Hafeez Khan; Parth Ganeriwala; Sarah M. Lehman; Siddhartha Bhattacharyya; Amy Alvarez; Natasha Neogi
>
> **备注:** Accepted to ICCV 2025, 2COOOL Workshop. Total 14 pages, 5 tables, and 4 figures
>
> **摘要:** Lane detection models are often evaluated in a closed-world setting, where training and testing occur on the same dataset. We observe that, even within the same domain, cross-dataset distribution shifts can cause severe catastrophic forgetting during fine-tuning. To address this, we first train a base model on a source distribution and then adapt it to each new target distribution by creating separate branches, fine-tuning only selected components while keeping the original source branch fixed. Based on a component-wise analysis, we identify effective fine-tuning strategies for target distributions that enable parameter-efficient adaptation. At inference time, we propose using a supervised contrastive learning model to identify the input distribution and dynamically route it to the corresponding branch. Our framework achieves near-optimal F1-scores while using significantly fewer parameters than training separate models for each distribution.
>
---
#### [new 052] PerioDet: Large-Scale Panoramic Radiograph Benchmark for Clinical-Oriented Apical Periodontitis Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决根尖周炎自动检测缺乏大规模标注数据的问题。作者构建了首个大规模全景X光基准数据集PerioXrays，并提出结合背景去噪注意力和IoU动态校准机制的PerioDet模型，显著提升了检测性能，验证了其临床辅助诊断潜力。**

- **链接: [http://arxiv.org/pdf/2507.18958v1](http://arxiv.org/pdf/2507.18958v1)**

> **作者:** Xiaocheng Fang; Jieyi Cai; Huanyu Liu; Chengju Zhou; Minhua Lu; Bingzhi Chen
>
> **备注:** MICCAI 2025(Early Accept)
>
> **摘要:** Apical periodontitis is a prevalent oral pathology that presents significant public health challenges. Despite advances in automated diagnostic systems across various medical fields, the development of Computer-Aided Diagnosis (CAD) applications for apical periodontitis is still constrained by the lack of a large-scale, high-quality annotated dataset. To address this issue, we release a large-scale panoramic radiograph benchmark called "PerioXrays", comprising 3,673 images and 5,662 meticulously annotated instances of apical periodontitis. To the best of our knowledge, this is the first benchmark dataset for automated apical periodontitis diagnosis. This paper further proposes a clinical-oriented apical periodontitis detection (PerioDet) paradigm, which jointly incorporates Background-Denoising Attention (BDA) and IoU-Dynamic Calibration (IDC) mechanisms to address the challenges posed by background noise and small targets in automated detection. Extensive experiments on the PerioXrays dataset demonstrate the superiority of PerioDet in advancing automated apical periodontitis detection. Additionally, a well-designed human-computer collaborative experiment underscores the clinical applicability of our method as an auxiliary diagnostic tool for professional dentists.
>
---
#### [new 053] Querying Autonomous Vehicle Point Clouds: Enhanced by 3D Object Counting with CounterNet
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于自动驾驶任务，旨在解决点云数据中准确物体计数的问题。现有检测模型在3D点云中计数易出错，影响查询结果。论文提出CounterNet，通过热力图检测物体中心，提升计数精度，并引入特征图划分和动态模型选择策略，优化复杂场景下的性能。**

- **链接: [http://arxiv.org/pdf/2507.19209v1](http://arxiv.org/pdf/2507.19209v1)**

> **作者:** Xiaoyu Zhang; Zhifeng Bao; Hai Dong; Ziwei Wang; Jiajun Liu
>
> **摘要:** Autonomous vehicles generate massive volumes of point cloud data, yet only a subset is relevant for specific tasks such as collision detection, traffic analysis, or congestion monitoring. Effectively querying this data is essential to enable targeted analytics. In this work, we formalize point cloud querying by defining three core query types: RETRIEVAL, COUNT, and AGGREGATION, each aligned with distinct analytical scenarios. All these queries rely heavily on accurate object counts to produce meaningful results, making precise object counting a critical component of query execution. Prior work has focused on indexing techniques for 2D video data, assuming detection models provide accurate counting information. However, when applied to 3D point cloud data, state-of-the-art detection models often fail to generate reliable object counts, leading to substantial errors in query results. To address this limitation, we propose CounterNet, a heatmap-based network designed for accurate object counting in large-scale point cloud data. Rather than focusing on accurate object localization, CounterNet detects object presence by finding object centers to improve counting accuracy. We further enhance its performance with a feature map partitioning strategy using overlapping regions, enabling better handling of both small and large objects in complex traffic scenes. To adapt to varying frame characteristics, we introduce a per-frame dynamic model selection strategy that selects the most effective configuration for each input. Evaluations on three real-world autonomous vehicle datasets show that CounterNet improves counting accuracy by 5% to 20% across object categories, resulting in more reliable query outcomes across all supported query types.
>
---
#### [new 054] PDT: Point Distribution Transformation with Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于3D几何处理任务，旨在解决如何从无序点云中提取并转换为语义上有意义的点分布问题。作者提出了PDT框架，结合扩散模型与新架构及学习策略，实现点云分布的语义化结构转换。**

- **链接: [http://arxiv.org/pdf/2507.18939v1](http://arxiv.org/pdf/2507.18939v1)**

> **作者:** Jionghao Wang; Cheng Lin; Yuan Liu; Rui Xu; Zhiyang Dou; Xiao-Xiao Long; Hao-Xiang Guo; Taku Komura; Wenping Wang; Xin Li
>
> **备注:** Project page: https://shanemankiw.github.io/PDT/
>
> **摘要:** Point-based representations have consistently played a vital role in geometric data structures. Most point cloud learning and processing methods typically leverage the unordered and unconstrained nature to represent the underlying geometry of 3D shapes. However, how to extract meaningful structural information from unstructured point cloud distributions and transform them into semantically meaningful point distributions remains an under-explored problem. We present PDT, a novel framework for point distribution transformation with diffusion models. Given a set of input points, PDT learns to transform the point set from its original geometric distribution into a target distribution that is semantically meaningful. Our method utilizes diffusion models with novel architecture and learning strategy, which effectively correlates the source and the target distribution through a denoising process. Through extensive experiments, we show that our method successfully transforms input point clouds into various forms of structured outputs - ranging from surface-aligned keypoints, and inner sparse joints to continuous feature lines. The results showcase our framework's ability to capture both geometric and semantic features, offering a powerful tool for various 3D geometry processing tasks where structured point distributions are desired. Code will be available at this link: https://github.com/shanemankiw/PDT.
>
---
#### [new 055] LISA: A Layer-wise Integration and Suppression Approach for Hallucination Mitigation in Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型任务，旨在解决图像描述生成中的物体幻觉问题。作者提出了LISA方法，通过层次集成与抑制策略，优化模型深层的注意力与语义融合，降低幻觉生成。该方法可插拔，适用于如Qwen2.5-VL等模型，并在多个基准上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2507.19110v1](http://arxiv.org/pdf/2507.19110v1)**

> **作者:** Zhihui Guo; Xin Man; Hui Xu; Jie Shao
>
> **摘要:** Multimodal Large Language Models (MLLMs) excel in vision-language tasks such as image captioning but remain prone to object hallucinations, where they describe objects that do not appear in the image. To mitigate this, we propose \textbf{LISA}, a \textbf{L}ayer-wise \textbf{I}ntegration and \textbf{S}uppression \textbf{A}pproach that enhances generation consistency through hierarchical modulation and multi-layer fusion. LISA leverages the functional hierarchy within MLLMs, where shallow layers provide visual grounding, middle layers encode semantics, and deep layers tend to amplify spurious signals. First, zone-specific spectral modulation stabilizes attention by suppressing over-amplified activations in deeper layers while preserving alignment cues in earlier layers. Second, token-level logits from selected layers are fused via anchor-based routing, with token-wise anchor selection and soft logit fusion enabling adaptive integration during decoding. LISA is fully \textbf{plug-and-play} and can be seamlessly integrated into existing MLLMs, including Qwen2.5-VL. Experiments on multiple benchmarks show that LISA reduces hallucinations by up to 53.6\% in $\mathrm{CHAIR}_I$ and improves POPE F1 by 4.5\%, demonstrating strong generalization across models and tasks.
>
---
#### [new 056] EffiComm: Bandwidth Efficient Multi Agent Communication
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶中车际通信任务，旨在解决传输原始感知数据导致的通信负载过高问题。论文提出EffiComm框架，通过选择性传输和自适应网格减少策略，降低传输数据量，同时保持高精度3D目标检测，实现了高效可扩展的通信。**

- **链接: [http://arxiv.org/pdf/2507.19354v1](http://arxiv.org/pdf/2507.19354v1)**

> **作者:** Melih Yazgan; Allen Xavier Arasan; J. Marius Zöllner
>
> **备注:** Accepted for publication at ITSC 2025
>
> **摘要:** Collaborative perception allows connected vehicles to exchange sensor information and overcome each vehicle's blind spots. Yet transmitting raw point clouds or full feature maps overwhelms Vehicle-to-Vehicle (V2V) communications, causing latency and scalability problems. We introduce EffiComm, an end-to-end framework that transmits less than 40% of the data required by prior art while maintaining state-of-the-art 3D object detection accuracy. EffiComm operates on Bird's-Eye-View (BEV) feature maps from any modality and applies a two-stage reduction pipeline: (1) Selective Transmission (ST) prunes low-utility regions with a confidence mask; (2) Adaptive Grid Reduction (AGR) uses a Graph Neural Network (GNN) to assign vehicle-specific keep ratios according to role and network load. The remaining features are fused with a soft-gated Mixture-of-Experts (MoE) attention layer, offering greater capacity and specialization for effective feature integration. On the OPV2V benchmark, EffiComm reaches 0.84 mAP@0.7 while sending only an average of approximately 1.5 MB per frame, outperforming previous methods on the accuracy-per-bit curve. These results highlight the value of adaptive, learned communication for scalable Vehicle-to-Everything (V2X) perception.
>
---
#### [new 057] GPSMamba: A Global Phase and Spectral Prompt-guided Mamba for Infrared Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于红外图像超分辨率任务，旨在解决红外图像低对比度、稀疏纹理及长程建模困难的问题。作者提出GPSMamba，通过引入语义-频率融合提示和热谱注意力与相位一致性损失，克服Mamba模型的1D因果扫描局限，提升全局结构与细节恢复效果。**

- **链接: [http://arxiv.org/pdf/2507.18998v1](http://arxiv.org/pdf/2507.18998v1)**

> **作者:** Yongsong Huang; Tomo Miyazaki; Xiaofeng Liu; Shinichiro Omachi
>
> **备注:** This manuscript is under review, and copyright will be transferred without notice
>
> **摘要:** Infrared Image Super-Resolution (IRSR) is challenged by the low contrast and sparse textures of infrared data, requiring robust long-range modeling to maintain global coherence. While State-Space Models like Mamba offer proficiency in modeling long-range dependencies for this task, their inherent 1D causal scanning mechanism fragments the global context of 2D images, hindering fine-detail restoration. To address this, we propose Global Phase and Spectral Prompt-guided Mamba (GPSMamba), a framework that synergizes architectural guidance with non-causal supervision. First, our Adaptive Semantic-Frequency State Space Module (ASF-SSM) injects a fused semantic-frequency prompt directly into the Mamba block, integrating non-local context to guide reconstruction. Then, a novel Thermal-Spectral Attention and Phase Consistency Loss provides explicit, non-causal supervision to enforce global structural and spectral fidelity. By combining these two innovations, our work presents a systematic strategy to mitigate the limitations of causal modeling. Extensive experiments demonstrate that GPSMamba achieves state-of-the-art performance, validating our approach as a powerful new paradigm for infrared image restoration. Code is available at https://github.com/yongsongH/GPSMamba.
>
---
#### [new 058] A Self-training Framework for Semi-supervised Pulmonary Vessel Segmentation and Its Application in COPD
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决慢性阻塞性肺疾病（COPD）患者肺血管分割精度低的问题。作者提出了一种半监督自训练框架Semi2，通过教师-学生模型迭代生成可靠伪标签，提升小血管分割效果。实验表明该方法在125例COPD患者的CT图像上取得了90.3%的分割精度，较原有方法提升2.3%。**

- **链接: [http://arxiv.org/pdf/2507.19074v1](http://arxiv.org/pdf/2507.19074v1)**

> **作者:** Shuiqing Zhao; Meihuan Wang; Jiaxuan Xu; Jie Feng; Wei Qian; Rongchang Chen; Zhenyu Liang; Shouliang Qi; Yanan Wu
>
> **摘要:** Background: It is fundamental for accurate segmentation and quantification of the pulmonary vessel, particularly smaller vessels, from computed tomography (CT) images in chronic obstructive pulmonary disease (COPD) patients. Objective: The aim of this study was to segment the pulmonary vasculature using a semi-supervised method. Methods: In this study, a self-training framework is proposed by leveraging a teacher-student model for the segmentation of pulmonary vessels. First, the high-quality annotations are acquired in the in-house data by an interactive way. Then, the model is trained in the semi-supervised way. A fully supervised model is trained on a small set of labeled CT images, yielding the teacher model. Following this, the teacher model is used to generate pseudo-labels for the unlabeled CT images, from which reliable ones are selected based on a certain strategy. The training of the student model involves these reliable pseudo-labels. This training process is iteratively repeated until an optimal performance is achieved. Results: Extensive experiments are performed on non-enhanced CT scans of 125 COPD patients. Quantitative and qualitative analyses demonstrate that the proposed method, Semi2, significantly improves the precision of vessel segmentation by 2.3%, achieving a precision of 90.3%. Further, quantitative analysis is conducted in the pulmonary vessel of COPD, providing insights into the differences in the pulmonary vessel across different severity of the disease. Conclusion: The proposed method can not only improve the performance of pulmonary vascular segmentation, but can also be applied in COPD analysis. The code will be made available at https://github.com/wuyanan513/semi-supervised-learning-for-vessel-segmentation.
>
---
#### [new 059] XAI-Guided Analysis of Residual Networks for Interpretable Pneumonia Detection in Paediatric Chest X-rays
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决儿童肺炎的自动诊断问题。作者基于ResNet-50构建模型，引入BayesGrad-CAM提升模型可解释性，实现高准确率分类并提供可视化依据，推动临床AI应用。**

- **链接: [http://arxiv.org/pdf/2507.18647v1](http://arxiv.org/pdf/2507.18647v1)**

> **作者:** Rayyan Ridwan
>
> **备注:** 13 pages, 14 figures
>
> **摘要:** Pneumonia remains one of the leading causes of death among children worldwide, underscoring a critical need for fast and accurate diagnostic tools. In this paper, we propose an interpretable deep learning model on Residual Networks (ResNets) for automatically diagnosing paediatric pneumonia on chest X-rays. We enhance interpretability through Bayesian Gradient-weighted Class Activation Mapping (BayesGrad-CAM), which quantifies uncertainty in visual explanations, and which offers spatial locations accountable for the decision-making process of the model. Our ResNet-50 model, trained on a large paediatric chest X-rays dataset, achieves high classification accuracy (95.94%), AUC-ROC (98.91%), and Cohen's Kappa (0.913), accompanied by clinically meaningful visual explanations. Our findings demonstrate that high performance and interpretability are not only achievable but critical for clinical AI deployment.
>
---
#### [new 060] Reconstruct or Generate: Exploring the Spectrum of Generative Modeling for Cardiac MRI
- **分类: cs.CV**

- **简介: 论文探讨了生成模型在心脏MRI中的重建与生成任务，分析扩散模型与自回归模型在不同掩码比例下的表现，发现扩散模型在无条件生成中感知质量高但易幻觉，而自回归模型在掩码下表现更稳定。**

- **链接: [http://arxiv.org/pdf/2507.19186v1](http://arxiv.org/pdf/2507.19186v1)**

> **作者:** Niklas Bubeck; Yundi Zhang; Suprosanna Shit; Daniel Rueckert; Jiazhen Pan
>
> **摘要:** In medical imaging, generative models are increasingly relied upon for two distinct but equally critical tasks: reconstruction, where the goal is to restore medical imaging (usually inverse problems like inpainting or superresolution), and generation, where synthetic data is created to augment datasets or carry out counterfactual analysis. Despite shared architecture and learning frameworks, they prioritize different goals: generation seeks high perceptual quality and diversity, while reconstruction focuses on data fidelity and faithfulness. In this work, we introduce a "generative model zoo" and systematically analyze how modern latent diffusion models and autoregressive models navigate the reconstruction-generation spectrum. We benchmark a suite of generative models across representative cardiac medical imaging tasks, focusing on image inpainting with varying masking ratios and sampling strategies, as well as unconditional image generation. Our findings show that diffusion models offer superior perceptual quality for unconditional generation but tend to hallucinate as masking ratios increase, whereas autoregressive models maintain stable perceptual performance across masking levels, albeit with generally lower fidelity.
>
---
#### [new 061] HeartUnloadNet: A Weakly-Supervised Cycle-Consistent Graph Network for Predicting Unloaded Cardiac Geometry from Diastolic States
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像处理与计算任务，旨在解决从心脏舒张状态预测无负荷心脏几何形状的问题。传统方法依赖计算昂贵的有限元求解器，而作者提出了HeartUnloadNet，一种基于图注意力和循环一致性的深度学习模型，实现快速且准确的预测，具备临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.18677v1](http://arxiv.org/pdf/2507.18677v1)**

> **作者:** Siyu Mu; Wei Xuan Chan; Choon Hwai Yap
>
> **备注:** Codes are available at https://github.com/SiyuMU/Loaded2UnNet
>
> **摘要:** The unloaded cardiac geometry (i.e., the state of the heart devoid of luminal pressure) serves as a valuable zero-stress and zero-strain reference and is critical for personalized biomechanical modeling of cardiac function, to understand both healthy and diseased physiology and to predict the effects of cardiac interventions. However, estimating the unloaded geometry from clinical images remains a challenging task. Traditional approaches rely on inverse finite element (FE) solvers that require iterative optimization and are computationally expensive. In this work, we introduce HeartUnloadNet, a deep learning framework that predicts the unloaded left ventricular (LV) shape directly from the end diastolic (ED) mesh while explicitly incorporating biophysical priors. The network accepts a mesh of arbitrary size along with physiological parameters such as ED pressure, myocardial stiffness scale, and fiber helix orientation, and outputs the corresponding unloaded mesh. It adopts a graph attention architecture and employs a cycle-consistency strategy to enable bidirectional (loading and unloading) prediction, allowing for partial self-supervision that improves accuracy and reduces the need for large training datasets. Trained and tested on 20,700 FE simulations across diverse LV geometries and physiological conditions, HeartUnloadNet achieves sub-millimeter accuracy, with an average DSC of 0.986 and HD of 0.083 cm, while reducing inference time to just 0.02 seconds per case, over 10^5 times faster and significantly more accurate than traditional inverse FE solvers. Ablation studies confirm the effectiveness of the architecture. Notably, the cycle-consistent design enables the model to maintain a DSC of 97% even with as few as 200 training samples. This work thus presents a scalable and accurate surrogate for inverse FE solvers, supporting real-time clinical applications in the future.
>
---
#### [new 062] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning
- **分类: cs.CV**

- **简介: 该论文属于虚拟角色手势生成任务，旨在解决手势与语音语义不一致的问题。作者提出SemGes模型，通过语义连贯性和相关性学习，实现基于语音、文本语义和说话人身份的自然手势生成，提升了手势的真实性和语义一致性。**

- **链接: [http://arxiv.org/pdf/2507.19359v1](http://arxiv.org/pdf/2507.19359v1)**

> **作者:** Lanmiao Liu; Esam Ghaleb; Aslı Özyürek; Zerrin Yumak
>
> **备注:** Accepted to IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Creating a virtual avatar with semantically coherent gestures that are aligned with speech is a challenging task. Existing gesture generation research mainly focused on generating rhythmic beat gestures, neglecting the semantic context of the gestures. In this paper, we propose a novel approach for semantic grounding in co-speech gesture generation that integrates semantic information at both fine-grained and global levels. Our approach starts with learning the motion prior through a vector-quantized variational autoencoder. Built on this model, a second-stage module is applied to automatically generate gestures from speech, text-based semantics and speaker identity that ensures consistency between the semantic relevance of generated gestures and co-occurring speech semantics through semantic coherence and relevance modules. Experimental results demonstrate that our approach enhances the realism and coherence of semantic gestures. Extensive experiments and user studies show that our method outperforms state-of-the-art approaches across two benchmarks in co-speech gesture generation in both objective and subjective metrics. The qualitative results of our model, code, dataset and pre-trained models can be viewed at https://semgesture.github.io/.
>
---
#### [new 063] A Survey of Multimodal Hallucination Evaluation and Detection
- **分类: cs.CV**

- **简介: 该论文属于多模态生成任务，旨在解决多模态大语言模型中的幻觉问题。论文提出了基于忠实性和事实性的幻觉分类体系，综述了图像到文本和文本到图像生成任务中的幻觉评估基准和检测方法，并指出了现有方法的局限性和未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.19024v1](http://arxiv.org/pdf/2507.19024v1)**

> **作者:** Zhiyuan Chen; Yuecong Min; Jie Zhang; Bei Yan; Jiahao Wang; Xiaozhen Wang; Shiguang Shan
>
> **备注:** 33 pages, 5 figures
>
> **摘要:** Multi-modal Large Language Models (MLLMs) have emerged as a powerful paradigm for integrating visual and textual information, supporting a wide range of multi-modal tasks. However, these models often suffer from hallucination, producing content that appears plausible but contradicts the input content or established world knowledge. This survey offers an in-depth review of hallucination evaluation benchmarks and detection methods across Image-to-Text (I2T) and Text-to-image (T2I) generation tasks. Specifically, we first propose a taxonomy of hallucination based on faithfulness and factuality, incorporating the common types of hallucinations observed in practice. Then we provide an overview of existing hallucination evaluation benchmarks for both T2I and I2T tasks, highlighting their construction process, evaluation objectives, and employed metrics. Furthermore, we summarize recent advances in hallucination detection methods, which aims to identify hallucinated content at the instance level and serve as a practical complement of benchmark-based evaluation. Finally, we highlight key limitations in current benchmarks and detection methods, and outline potential directions for future research.
>
---
#### [new 064] MGHFT: Multi-Granularity Hierarchical Fusion Transformer for Cross-Modal Sticker Emotion Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨模态贴纸情感识别任务，旨在解决贴纸情感理解中多视角信息融合的难题。作者提出了一种多粒度层次融合Transformer（MGHFT），结合多模态大语言模型与视觉Transformer，通过分层融合策略和注意力机制，将文本信息多阶段注入视觉特征提取过程，提升了贴纸情感识别的准确性与细粒度识别能力。**

- **链接: [http://arxiv.org/pdf/2507.18929v1](http://arxiv.org/pdf/2507.18929v1)**

> **作者:** Jian Chen; Yuxuan Hu; Haifeng Lu; Wei Wang; Min Yang; Chengming Li; Xiping Hu
>
> **备注:** Accepted by ACMMM2025
>
> **摘要:** Although pre-trained visual models with text have demonstrated strong capabilities in visual feature extraction, sticker emotion understanding remains challenging due to its reliance on multi-view information, such as background knowledge and stylistic cues. To address this, we propose a novel multi-granularity hierarchical fusion transformer (MGHFT), with a multi-view sticker interpreter based on Multimodal Large Language Models. Specifically, inspired by the human ability to interpret sticker emotions from multiple views, we first use Multimodal Large Language Models to interpret stickers by providing rich textual context via multi-view descriptions. Then, we design a hierarchical fusion strategy to fuse the textual context into visual understanding, which builds upon a pyramid visual transformer to extract both global and local sticker features at multiple stages. Through contrastive learning and attention mechanisms, textual features are injected at different stages of the visual backbone, enhancing the fusion of global- and local-granularity visual semantics with textual guidance. Finally, we introduce a text-guided fusion attention mechanism to effectively integrate the overall multimodal features, enhancing semantic understanding. Extensive experiments on 2 public sticker emotion datasets demonstrate that MGHFT significantly outperforms existing sticker emotion recognition approaches, achieving higher accuracy and more fine-grained emotion recognition. Compared to the best pre-trained visual models, our MGHFT also obtains an obvious improvement, 5.4% on F1 and 4.0% on accuracy. The code is released at https://github.com/cccccj-03/MGHFT_ACMMM2025.
>
---
#### [new 065] SaLF: Sparse Local Fields for Multi-Sensor Rendering in Real-Time
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶传感器模拟任务，旨在解决现有方法（如NeRF和3DGS）在训练渲染速度、多传感器支持及表示与渲染耦合方面的问题。论文提出SaLF，一种支持光栅化与光线追踪的稀疏局部场表示方法，实现高效、高质量的多传感器实时渲染。**

- **链接: [http://arxiv.org/pdf/2507.18713v1](http://arxiv.org/pdf/2507.18713v1)**

> **作者:** Yun Chen; Matthew Haines; Jingkang Wang; Krzysztof Baron-Lis; Sivabalan Manivasagam; Ze Yang; Raquel Urtasun
>
> **摘要:** High-fidelity sensor simulation of light-based sensors such as cameras and LiDARs is critical for safe and accurate autonomy testing. Neural radiance field (NeRF)-based methods that reconstruct sensor observations via ray-casting of implicit representations have demonstrated accurate simulation of driving scenes, but are slow to train and render, hampering scale. 3D Gaussian Splatting (3DGS) has demonstrated faster training and rendering times through rasterization, but is primarily restricted to pinhole camera sensors, preventing usage for realistic multi-sensor autonomy evaluation. Moreover, both NeRF and 3DGS couple the representation with the rendering procedure (implicit networks for ray-based evaluation, particles for rasterization), preventing interoperability, which is key for general usage. In this work, we present Sparse Local Fields (SaLF), a novel volumetric representation that supports rasterization and raytracing. SaLF represents volumes as a sparse set of 3D voxel primitives, where each voxel is a local implicit field. SaLF has fast training (<30 min) and rendering capabilities (50+ FPS for camera and 600+ FPS LiDAR), has adaptive pruning and densification to easily handle large scenes, and can support non-pinhole cameras and spinning LiDARs. We demonstrate that SaLF has similar realism as existing self-driving sensor simulation methods while improving efficiency and enhancing capabilities, enabling more scalable simulation. https://waabi.ai/salf/
>
---
#### [new 066] NerT-CA: Efficient Dynamic Reconstruction from Sparse-view X-ray Coronary Angiography
- **分类: cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决稀疏视角X射线冠状动脉造影的动态三维重建问题。论文提出NerT-CA方法，结合张量场与神经场表示，实现快速准确的4D重建，克服传统方法耗时和现有神经方法训练慢的问题。**

- **链接: [http://arxiv.org/pdf/2507.19328v1](http://arxiv.org/pdf/2507.19328v1)**

> **作者:** Kirsten W. H. Maas; Danny Ruijters; Nicola Pezzotti; Anna Vilanova
>
> **摘要:** Three-dimensional (3D) and dynamic 3D+time (4D) reconstruction of coronary arteries from X-ray coronary angiography (CA) has the potential to improve clinical procedures. However, there are multiple challenges to be addressed, most notably, blood-vessel structure sparsity, poor background and blood vessel distinction, sparse-views, and intra-scan motion. State-of-the-art reconstruction approaches rely on time-consuming manual or error-prone automatic segmentations, limiting clinical usability. Recently, approaches based on Neural Radiance Fields (NeRF) have shown promise for automatic reconstructions in the sparse-view setting. However, they suffer from long training times due to their dependence on MLP-based representations. We propose NerT-CA, a hybrid approach of Neural and Tensorial representations for accelerated 4D reconstructions with sparse-view CA. Building on top of the previous NeRF-based work, we model the CA scene as a decomposition of low-rank and sparse components, utilizing fast tensorial fields for low-rank static reconstruction and neural fields for dynamic sparse reconstruction. Our approach outperforms previous works in both training time and reconstruction accuracy, yielding reasonable reconstructions from as few as three angiogram views. We validate our approach quantitatively and qualitatively on representative 4D phantom datasets.
>
---
#### [new 067] Cross Spatial Temporal Fusion Attention for Remote Sensing Object Detection via Image Feature Matching
- **分类: cs.CV**

- **简介: 论文属于遥感目标检测任务，旨在解决多模态遥感图像间几何和辐射差异导致的跨模态特征匹配难题。作者提出了一种跨时空融合注意力机制（CSTF），通过融合尺度不变关键点与多区域信息，提升了特征匹配效果，并在HRSC2016和DOTA数据集上取得了最优性能。**

- **链接: [http://arxiv.org/pdf/2507.19118v1](http://arxiv.org/pdf/2507.19118v1)**

> **作者:** Abu Sadat Mohammad Salehin Amit; Xiaoli Zhang; Md Masum Billa Shagar; Zhaojun Liu; Xiongfei Li; Fanlong Meng
>
> **摘要:** Effectively describing features for cross-modal remote sensing image matching remains a challenging task due to the significant geometric and radiometric differences between multimodal images. Existing methods primarily extract features at the fully connected layer but often fail to capture cross-modal similarities effectively. We propose a Cross Spatial Temporal Fusion (CSTF) mechanism that enhances feature representation by integrating scale-invariant keypoints detected independently in both reference and query images. Our approach improves feature matching in two ways: First, by creating correspondence maps that leverage information from multiple image regions simultaneously, and second, by reformulating the similarity matching process as a classification task using SoftMax and Fully Convolutional Network (FCN) layers. This dual approach enables CSTF to maintain sensitivity to distinctive local features while incorporating broader contextual information, resulting in robust matching across diverse remote sensing modalities. To demonstrate the practical utility of improved feature matching, we evaluate CSTF on object detection tasks using the HRSC2016 and DOTA benchmark datasets. Our method achieves state-of-theart performance with an average mAP of 90.99% on HRSC2016 and 90.86% on DOTA, outperforming existing models. The CSTF model maintains computational efficiency with an inference speed of 12.5 FPS. These results validate that our approach to crossmodal feature matching directly enhances downstream remote sensing applications such as object detection.
>
---
#### [new 068] Features extraction for image identification using computer vision
- **分类: cs.CV**

- **简介: 该论文属于图像识别任务，旨在比较不同特征提取方法在计算机视觉中的性能。论文重点分析了Vision Transformers（ViTs）的结构和优势，并对比了GANs、深度特征模型、传统方法（如SIFT、SURF、ORB）以及对比与非对比特征模型。通过实验评估了各方法的优劣及适用场景。**

- **链接: [http://arxiv.org/pdf/2507.18650v1](http://arxiv.org/pdf/2507.18650v1)**

> **作者:** Venant Niyonkuru; Sylla Sekou; Jimmy Jackson Sinzinkayo
>
> **摘要:** This study examines various feature extraction techniques in computer vision, the primary focus of which is on Vision Transformers (ViTs) and other approaches such as Generative Adversarial Networks (GANs), deep feature models, traditional approaches (SIFT, SURF, ORB), and non-contrastive and contrastive feature models. Emphasizing ViTs, the report summarizes their architecture, including patch embedding, positional encoding, and multi-head self-attention mechanisms with which they overperform conventional convolutional neural networks (CNNs). Experimental results determine the merits and limitations of both methods and their utilitarian applications in advancing computer vision.
>
---
#### [new 069] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction for Autonomous Driving with Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的三维占据空间重建任务，旨在解决依赖激光雷达标注数据限制扩展性的问题。论文提出GS-Occ3D，一种基于高斯投影的纯视觉占据重建框架，通过八叉树优化显式占据表示，实现高效、可扩展的几何重建。**

- **链接: [http://arxiv.org/pdf/2507.19451v1](http://arxiv.org/pdf/2507.19451v1)**

> **作者:** Baijun Ye; Minghui Qin; Saining Zhang; Moonjun Gong; Shaoting Zhu; Zebang Shen; Luan Zhang; Lu Zhang; Hao Zhao; Hang Zhao
>
> **备注:** ICCV 2025. Project Page: https://gs-occ3d.github.io/
>
> **摘要:** Occupancy is crucial for autonomous driving, providing essential geometric priors for perception and planning. However, existing methods predominantly rely on LiDAR-based occupancy annotations, which limits scalability and prevents leveraging vast amounts of potential crowdsourced data for auto-labeling. To address this, we propose GS-Occ3D, a scalable vision-only framework that directly reconstructs occupancy. Vision-only occupancy reconstruction poses significant challenges due to sparse viewpoints, dynamic scene elements, severe occlusions, and long-horizon motion. Existing vision-based methods primarily rely on mesh representation, which suffer from incomplete geometry and additional post-processing, limiting scalability. To overcome these issues, GS-Occ3D optimizes an explicit occupancy representation using an Octree-based Gaussian Surfel formulation, ensuring efficiency and scalability. Additionally, we decompose scenes into static background, ground, and dynamic objects, enabling tailored modeling strategies: (1) Ground is explicitly reconstructed as a dominant structural element, significantly improving large-area consistency; (2) Dynamic vehicles are separately modeled to better capture motion-related occupancy patterns. Extensive experiments on the Waymo dataset demonstrate that GS-Occ3D achieves state-of-the-art geometry reconstruction results. By curating vision-only binary occupancy labels from diverse urban scenes, we show their effectiveness for downstream occupancy models on Occ3D-Waymo and superior zero-shot generalization on Occ3D-nuScenes. It highlights the potential of large-scale vision-based occupancy reconstruction as a new paradigm for autonomous driving perception. Project Page: https://gs-occ3d.github.io/
>
---
#### [new 070] Structure Matters: Revisiting Boundary Refinement in Video Object Segmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于视频目标分割任务，旨在解决遮挡场景下目标交互和特征相似导致的分割难题。作者提出OASIS方法，融合Canny边缘先验和对象特征，通过结构精炼模块提升边界特征表达，并引入不确定性估计优化遮挡区域处理，实现高效精准的视频目标分割。**

- **链接: [http://arxiv.org/pdf/2507.18944v1](http://arxiv.org/pdf/2507.18944v1)**

> **作者:** Guanyi Qin; Ziyue Wang; Daiyun Shen; Haofeng Liu; Hantao Zhou; Junde Wu; Runze Hu; Yueming Jin
>
> **摘要:** Given an object mask, Semi-supervised Video Object Segmentation (SVOS) technique aims to track and segment the object across video frames, serving as a fundamental task in computer vision. Although recent memory-based methods demonstrate potential, they often struggle with scenes involving occlusion, particularly in handling object interactions and high feature similarity. To address these issues and meet the real-time processing requirements of downstream applications, in this paper, we propose a novel bOundary Amendment video object Segmentation method with Inherent Structure refinement, hereby named OASIS. Specifically, a lightweight structure refinement module is proposed to enhance segmentation accuracy. With the fusion of rough edge priors captured by the Canny filter and stored object features, the module can generate an object-level structure map and refine the representations by highlighting boundary features. Evidential learning for uncertainty estimation is introduced to further address challenges in occluded regions. The proposed method, OASIS, maintains an efficient design, yet extensive experiments on challenging benchmarks demonstrate its superior performance and competitive inference speed compared to other state-of-the-art methods, i.e., achieving the F values of 91.6 (vs. 89.7 on DAVIS-17 validation set) and G values of 86.6 (vs. 86.2 on YouTubeVOS 2019 validation set) while maintaining a competitive speed of 48 FPS on DAVIS.
>
---
#### [new 071] Synthetic-to-Real Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文属于合成到真实伪装目标检测任务（S2R-COD），旨在解决因真实数据稀缺导致的伪装目标检测模型性能下降问题。作者提出CSRDA框架，通过伪标签、一致性正则化和循环学习机制，将合成数据知识迁移到真实数据中，提升模型在真实场景下的表现。**

- **链接: [http://arxiv.org/pdf/2507.18911v1](http://arxiv.org/pdf/2507.18911v1)**

> **作者:** Zhihao Luo; Luojun Lin; Zheng Lin
>
> **摘要:** Due to the high cost of collection and labeling, there are relatively few datasets for camouflaged object detection (COD). In particular, for certain specialized categories, the available image dataset is insufficiently populated. Synthetic datasets can be utilized to alleviate the problem of limited data to some extent. However, directly training with synthetic datasets compared to real datasets can lead to a degradation in model performance. To tackle this problem, in this work, we investigate a new task, namely Syn-to-Real Camouflaged Object Detection (S2R-COD). In order to improve the model performance in real world scenarios, a set of annotated synthetic camouflaged images and a limited number of unannotated real images must be utilized. We propose the Cycling Syn-to-Real Domain Adaptation Framework (CSRDA), a method based on the student-teacher model. Specially, CSRDA propagates class information from the labeled source domain to the unlabeled target domain through pseudo labeling combined with consistency regularization. Considering that narrowing the intra-domain gap can improve the quality of pseudo labeling, CSRDA utilizes a recurrent learning framework to build an evolving real domain for bridging the source and target domain. Extensive experiments demonstrate the effectiveness of our framework, mitigating the problem of limited data and handcraft annotations in COD. Our code is publicly available at: https://github.com/Muscape/S2R-COD
>
---
#### [new 072] Continual Learning-Based Unified Model for Unpaired Image Restoration Tasks
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决不同天气条件下（如雾、雪、雨）图像恢复的统一模型问题。现有方法多针对单一天气，而实际应用如自动驾驶需要统一模型。论文提出一种基于持续学习的统一框架，包含选择性核融合层、弹性权重巩固和循环对比损失，并采用无配对图像恢复方法减少数据依赖，显著提升了恢复效果。**

- **链接: [http://arxiv.org/pdf/2507.19184v1](http://arxiv.org/pdf/2507.19184v1)**

> **作者:** Kotha Kartheek; Lingamaneni Gnanesh Chowdary; Snehasis Mukherjee
>
> **备注:** Under Review
>
> **摘要:** Restoration of images contaminated by different adverse weather conditions such as fog, snow, and rain is a challenging task due to the varying nature of the weather conditions. Most of the existing methods focus on any one particular weather conditions. However, for applications such as autonomous driving, a unified model is necessary to perform restoration of corrupted images due to different weather conditions. We propose a continual learning approach to propose a unified framework for image restoration. The proposed framework integrates three key innovations: (1) Selective Kernel Fusion layers that dynamically combine global and local features for robust adaptive feature selection; (2) Elastic Weight Consolidation (EWC) to enable continual learning and mitigate catastrophic forgetting across multiple restoration tasks; and (3) a novel Cycle-Contrastive Loss that enhances feature discrimination while preserving semantic consistency during domain translation. Further, we propose an unpaired image restoration approach to reduce the dependance of the proposed approach on the training data. Extensive experiments on standard benchmark datasets for dehazing, desnowing and deraining tasks demonstrate significant improvements in PSNR, SSIM, and perceptual quality over the state-of-the-art.
>
---
#### [new 073] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于图像溯源任务，旨在解决生成图像来源追踪问题。现有方法在准确性和计算效率上存在不足，为此，论文提出AEDR方法，通过双重建损失比结合图像同质性指标，在无需训练的情况下提升溯源准确性并大幅降低计算开销。**

- **链接: [http://arxiv.org/pdf/2507.18988v1](http://arxiv.org/pdf/2507.18988v1)**

> **作者:** Chao Wang; Kejiang Chen; Zijin Yang; Yaofei Wang; Weiming Zhang
>
> **摘要:** The rapid advancement of image-generation technologies has made it possible for anyone to create photorealistic images using generative models, raising significant security concerns. To mitigate malicious use, tracing the origin of such images is essential. Reconstruction-based attribution methods offer a promising solution, but they often suffer from reduced accuracy and high computational costs when applied to state-of-the-art (SOTA) models. To address these challenges, we propose AEDR (AutoEncoder Double-Reconstruction), a novel training-free attribution method designed for generative models with continuous autoencoders. Unlike existing reconstruction-based approaches that rely on the value of a single reconstruction loss, AEDR performs two consecutive reconstructions using the model's autoencoder, and adopts the ratio of these two reconstruction losses as the attribution signal. This signal is further calibrated using the image homogeneity metric to improve accuracy, which inherently cancels out absolute biases caused by image complexity, with autoencoder-based reconstruction ensuring superior computational efficiency. Experiments on eight top latent diffusion models show that AEDR achieves 25.5% higher attribution accuracy than existing reconstruction-based methods, while requiring only 1% of the computational time.
>
---
#### [new 074] Learning Efficient and Generalizable Human Representation with Human Gaussian Model
- **分类: cs.CV**

- **简介: 该论文属于三维人体建模与动画生成任务，旨在从视频中建模可动画驱动的人体化身。现有方法未能充分利用多帧间的高斯关联，导致表征不够高效和泛化。论文提出Human Gaussian Graph，通过高斯节点与网格顶点双层结构，建模帧间关系并传递信息，提升了动画生成的效率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.18758v1](http://arxiv.org/pdf/2507.18758v1)**

> **作者:** Yifan Liu; Shengjun Zhang; Chensheng Dai; Yang Chen; Hao Liu; Chen Li; Yueqi Duan
>
> **摘要:** Modeling animatable human avatars from videos is a long-standing and challenging problem. While conventional methods require per-instance optimization, recent feed-forward methods have been proposed to generate 3D Gaussians with a learnable network. However, these methods predict Gaussians for each frame independently, without fully capturing the relations of Gaussians from different timestamps. To address this, we propose Human Gaussian Graph to model the connection between predicted Gaussians and human SMPL mesh, so that we can leverage information from all frames to recover an animatable human representation. Specifically, the Human Gaussian Graph contains dual layers where Gaussians are the first layer nodes and mesh vertices serve as the second layer nodes. Based on this structure, we further propose the intra-node operation to aggregate various Gaussians connected to one mesh vertex, and inter-node operation to support message passing among mesh node neighbors. Experimental results on novel view synthesis and novel pose animation demonstrate the efficiency and generalization of our method.
>
---
#### [new 075] WiSE-OD: Benchmarking Robustness in Infrared Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于红外目标检测任务，旨在解决红外图像检测中因数据不足和模态差异导致的模型鲁棒性问题。作者构建了两个新的跨模态分布外基准LLVIP-C和FLIR-C，并提出WiSE-OD方法，通过权重空间集成提升模型在分布偏移下的性能，无需额外训练或推理成本。**

- **链接: [http://arxiv.org/pdf/2507.18925v1](http://arxiv.org/pdf/2507.18925v1)**

> **作者:** Heitor R. Medeiros; Atif Belal; Masih Aminbeidokhti; Eric Granger; Marco Pedersoli
>
> **备注:** 8 pages, conference
>
> **摘要:** Object detection (OD) in infrared (IR) imagery is critical for low-light and nighttime applications. However, the scarcity of large-scale IR datasets forces models to rely on weights pre-trained on RGB images. While fine-tuning on IR improves accuracy, it often compromises robustness under distribution shifts due to the inherent modality gap between RGB and IR. To address this, we introduce LLVIP-C and FLIR-C, two cross-modality out-of-distribution (OOD) benchmarks built by applying corruption to standard IR datasets. Additionally, to fully leverage the complementary knowledge from RGB and infrared trained models, we propose WiSE-OD, a weight-space ensembling method with two variants: WiSE-OD$_{ZS}$, which combines RGB zero-shot and IR fine-tuned weights, and WiSE-OD$_{LP}$, which blends zero-shot and linear probing. Evaluated across three RGB-pretrained detectors and two robust baselines, WiSE-OD improves both cross-modality and corruption robustness without any additional training or inference cost.
>
---
#### [new 076] Joint Holistic and Lesion Controllable Mammogram Synthesis via Gated Conditional Diffusion Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像生成任务，旨在解决 mammogram 合成中病变特征不明显、多样性不足的问题。作者提出 Gated Conditional Diffusion Model（GCDM），通过结合病变区域软掩码和动态门控条件分支，实现对病变和整体图像的协同生成，提升合成 mammogram 的真实性和病变可控性。**

- **链接: [http://arxiv.org/pdf/2507.19201v1](http://arxiv.org/pdf/2507.19201v1)**

> **作者:** Xin Li; Kaixiang Yang; Qiang Li; Zhiwei Wang
>
> **备注:** Accepted, ACM Multimedia 2025, 10 pages, 5 figures
>
> **摘要:** Mammography is the most commonly used imaging modality for breast cancer screening, driving an increasing demand for deep-learning techniques to support large-scale analysis. However, the development of accurate and robust methods is often limited by insufficient data availability and a lack of diversity in lesion characteristics. While generative models offer a promising solution for data synthesis, current approaches often fail to adequately emphasize lesion-specific features and their relationships with surrounding tissues. In this paper, we propose Gated Conditional Diffusion Model (GCDM), a novel framework designed to jointly synthesize holistic mammogram images and localized lesions. GCDM is built upon a latent denoising diffusion framework, where the noised latent image is concatenated with a soft mask embedding that represents breast, lesion, and their transitional regions, ensuring anatomical coherence between them during the denoising process. To further emphasize lesion-specific features, GCDM incorporates a gated conditioning branch that guides the denoising process by dynamically selecting and fusing the most relevant radiomic and geometric properties of lesions, effectively capturing their interplay. Experimental results demonstrate that GCDM achieves precise control over small lesion areas while enhancing the realism and diversity of synthesized mammograms. These advancements position GCDM as a promising tool for clinical applications in mammogram synthesis. Our code is available at https://github.com/lixinHUST/Gated-Conditional-Diffusion-Model/
>
---
#### [new 077] YOLO for Knowledge Extraction from Vehicle Images: A Baseline Study
- **分类: cs.CV**

- **简介: 该论文属于知识抽取任务，旨在从车辆图像中准确识别车辆属性（如品牌、形状和颜色）。论文评估了三种YOLO模型在真实复杂数据集上的表现，采用多视角推理提升效果，并提供了高效的基线模型用于实际场景中的车辆信息提取。**

- **链接: [http://arxiv.org/pdf/2507.18966v1](http://arxiv.org/pdf/2507.18966v1)**

> **作者:** Saraa Al-Saddik; Manna Elizabeth Philip; Ali Haidar
>
> **摘要:** Accurate identification of vehicle attributes such as make, colour, and shape is critical for law enforcement and intelligence applications. This study evaluates the effectiveness of three state-of-the-art deep learning approaches YOLO-v11, YOLO-World, and YOLO-Classification on a real-world vehicle image dataset. This dataset was collected under challenging and unconstrained conditions by NSW Police Highway Patrol Vehicles. A multi-view inference (MVI) approach was deployed to enhance the performance of the models' predictions. To conduct the analyses, datasets with 100,000 plus images were created for each of the three metadata prediction tasks, specifically make, shape and colour. The models were tested on a separate dataset with 29,937 images belonging to 1809 number plates. Different sets of experiments have been investigated by varying the models sizes. A classification accuracy of 93.70%, 82.86%, 85.19%, and 94.86% was achieved with the best performing make, shape, colour, and colour-binary models respectively. It was concluded that there is a need to use MVI to get usable models within such complex real-world datasets. Our findings indicated that the object detection models YOLO-v11 and YOLO-World outperformed classification-only models in make and shape extraction. Moreover, smaller YOLO variants perform comparably to larger counterparts, offering substantial efficiency benefits for real-time predictions. This work provides a robust baseline for extracting vehicle metadata in real-world scenarios. Such models can be used in filtering and sorting user queries, minimising the time required to search large vehicle images datasets.
>
---
#### [new 078] BEV-LLM: Leveraging Multimodal BEV Maps for Scene Captioning in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶场景描述任务，旨在提升系统透明性和人机交互。论文提出BEV-LLM模型，融合多模态数据生成环境描述，提升BLEU评分，并发布新数据集评估多样场景。**

- **链接: [http://arxiv.org/pdf/2507.19370v1](http://arxiv.org/pdf/2507.19370v1)**

> **作者:** Felix Brandstaetter; Erik Schuetz; Katharina Winter; Fabian Flohr
>
> **摘要:** Autonomous driving technology has the potential to transform transportation, but its wide adoption depends on the development of interpretable and transparent decision-making systems. Scene captioning, which generates natural language descriptions of the driving environment, plays a crucial role in enhancing transparency, safety, and human-AI interaction. We introduce BEV-LLM, a lightweight model for 3D captioning of autonomous driving scenes. BEV-LLM leverages BEVFusion to combine 3D LiDAR point clouds and multi-view images, incorporating a novel absolute positional encoding for view-specific scene descriptions. Despite using a small 1B parameter base model, BEV-LLM achieves competitive performance on the nuCaption dataset, surpassing state-of-the-art by up to 5\% in BLEU scores. Additionally, we release two new datasets - nuView (focused on environmental conditions and viewpoints) and GroundView (focused on object grounding) - to better assess scene captioning across diverse driving scenarios and address gaps in current benchmarks, along with initial benchmarking results demonstrating their effectiveness.
>
---
#### [new 079] Livatar-1: Real-Time Talking Heads Generation with Tailored Flow Matching
- **分类: cs.CV**

- **简介: 论文提出Livatar-1，属于音频驱动的说话人视频生成任务，旨在解决唇形同步精度低和长时间姿态漂移问题。作者采用流匹配框架并结合系统优化，实现高质量实时生成，达到每秒141帧的吞吐量和0.17秒端到端延迟。**

- **链接: [http://arxiv.org/pdf/2507.18649v1](http://arxiv.org/pdf/2507.18649v1)**

> **作者:** Haiyang Liu; Xiaolin Hong; Xuancheng Yang; Yudi Ruan; Xiang Lian; Michael Lingelbach; Hongwei Yi; Wei Li
>
> **备注:** Technical Report
>
> **摘要:** We present Livatar, a real-time audio-driven talking heads videos generation framework. Existing baselines suffer from limited lip-sync accuracy and long-term pose drift. We address these limitations with a flow matching based framework. Coupled with system optimizations, Livatar achieves competitive lip-sync quality with a 8.50 LipSync Confidence on the HDTF dataset, and reaches a throughput of 141 FPS with an end-to-end latency of 0.17s on a single A10 GPU. This makes high-fidelity avatars accessible to broader applications. Our project is available at https://www.hedra.com/ with with examples at https://h-liu1997.github.io/Livatar-1/
>
---
#### [new 080] Learned Single-Pixel Fluorescence Microscopy
- **分类: cs.CV; cs.AI; physics.optics**

- **简介: 该论文属于图像重建任务，旨在解决单像素荧光显微成像中重建速度慢、质量低的问题。作者通过自监督训练一个编码器-解码器网络，学习优化测量矩阵与重建过程，显著提升了重建速度与图像质量，并实现了多光谱成像。**

- **链接: [http://arxiv.org/pdf/2507.18740v1](http://arxiv.org/pdf/2507.18740v1)**

> **作者:** Serban C. Tudosie; Valerio Gandolfi; Shivaprasad Varakkoth; Andrea Farina; Cosimo D'Andrea; Simon Arridge
>
> **备注:** 10 pages, 6 figures, 1 table
>
> **摘要:** Single-pixel imaging has emerged as a key technique in fluorescence microscopy, where fast acquisition and reconstruction are crucial. In this context, images are reconstructed from linearly compressed measurements. In practice, total variation minimisation is still used to reconstruct the image from noisy measurements of the inner product between orthogonal sampling pattern vectors and the original image data. However, data can be leveraged to learn the measurement vectors and the reconstruction process, thereby enhancing compression, reconstruction quality, and speed. We train an autoencoder through self-supervision to learn an encoder (or measurement matrix) and a decoder. We then test it on physically acquired multispectral and intensity data. During acquisition, the learned encoder becomes part of the physical device. Our approach can enhance single-pixel imaging in fluorescence microscopy by reducing reconstruction time by two orders of magnitude, achieving superior image quality, and enabling multispectral reconstructions. Ultimately, learned single-pixel fluorescence microscopy could advance diagnosis and biological research, providing multispectral imaging at a fraction of the cost.
>
---
#### [new 081] Deepfake Detection Via Facial Feature Extraction and Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度伪造检测任务，旨在解决AI生成视频真假难辨的问题。通过提取面部关键点，利用RNN、ANN和CNN模型识别面部动作中的细微不一致。实验表明该方法在多个模型中表现良好，准确率最高达96%，且无需直接处理图像，参数更少。**

- **链接: [http://arxiv.org/pdf/2507.18815v1](http://arxiv.org/pdf/2507.18815v1)**

> **作者:** Benjamin Carter; Nathan Dilla; Micheal Callahan; Atuhaire Ambala
>
> **备注:** Keywords: deepfake, facial recognition, feature extraction, artificial intelligence, recurrent neural network, convolutional neural network, artificial neural network
>
> **摘要:** The rise of deepfake technology brings forth new questions about the authenticity of various forms of media found online today. Videos and images generated by artificial intelligence (AI) have become increasingly more difficult to differentiate from genuine media, resulting in the need for new models to detect artificially-generated media. While many models have attempted to solve this, most focus on direct image processing, adapting a convolutional neural network (CNN) or a recurrent neural network (RNN) that directly interacts with the video image data. This paper introduces an approach of using solely facial landmarks for deepfake detection. Using a dataset consisting of both deepfake and genuine videos of human faces, this paper describes an approach for extracting facial landmarks for deepfake detection, focusing on identifying subtle inconsistencies in facial movements instead of raw image processing. Experimental results demonstrated that this feature extraction technique is effective in various neural network models, with the same facial landmarks tested on three neural network models, with promising performance metrics indicating its potential for real-world applications. The findings discussed in this paper include RNN and artificial neural network (ANN) models with accuracy between 96% and 93%, respectively, with a CNN model hovering around 78%. This research challenges the assumption that raw image processing is necessary to identify deepfake videos by presenting a facial feature extraction approach compatible with various neural network models while requiring fewer parameters.
>
---
#### [new 082] Phoneme-Level Visual Speech Recognition via Point-Visual Fusion and Language Model Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于视觉语音识别（V-ASR）任务，旨在仅通过视觉信息识别语音。为解决发音相似导致的识别错误和数据需求大问题，作者提出PV-ASR方法，结合视觉与面部特征，分两阶段识别音素并重建单词，提升准确率。**

- **链接: [http://arxiv.org/pdf/2507.18863v1](http://arxiv.org/pdf/2507.18863v1)**

> **作者:** Matthew Kit Khinn Teng; Haibo Zhang; Takeshi Saitoh
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Visual Automatic Speech Recognition (V-ASR) is a challenging task that involves interpreting spoken language solely from visual information, such as lip movements and facial expressions. This task is notably challenging due to the absence of auditory cues and the visual ambiguity of phonemes that exhibit similar visemes-distinct sounds that appear identical in lip motions. Existing methods often aim to predict words or characters directly from visual cues, but they commonly suffer from high error rates due to viseme ambiguity and require large amounts of pre-training data. We propose a novel phoneme-based two-stage framework that fuses visual and landmark motion features, followed by an LLM model for word reconstruction to address these challenges. Stage 1 consists of V-ASR, which outputs the predicted phonemes, thereby reducing training complexity. Meanwhile, the facial landmark features address speaker-specific facial characteristics. Stage 2 comprises an encoder-decoder LLM model, NLLB, that reconstructs the output phonemes back to words. Besides using a large visual dataset for deep learning fine-tuning, our PV-ASR method demonstrates superior performance by achieving 17.4% WER on the LRS2 and 21.0% WER on the LRS3 dataset.
>
---
#### [new 083] BridgeNet: A Unified Multimodal Framework for Bridging 2D and 3D Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于工业异常检测任务，旨在解决2D与3D模态信息融合不足、3D深度异常检测困难的问题。论文提出BridgeNet框架，通过解耦深度与外观信息，实现统一的异常生成，并设计多尺度异常生成模块，提升RGB与深度数据的异常检测性能。**

- **链接: [http://arxiv.org/pdf/2507.19253v1](http://arxiv.org/pdf/2507.19253v1)**

> **作者:** An Xiang; Zixuan Huang; Xitong Gao; Kejiang Ye; Cheng-zhong Xu
>
> **摘要:** Industrial anomaly detection for 2D objects has gained significant attention and achieved progress in anomaly detection (AD) methods. However, identifying 3D depth anomalies using only 2D information is insufficient. Despite explicitly fusing depth information into RGB images or using point cloud backbone networks to extract depth features, both approaches struggle to adequately represent 3D information in multimodal scenarios due to the disparities among different modal information. Additionally, due to the scarcity of abnormal samples in industrial data, especially in multimodal scenarios, it is necessary to perform anomaly generation to simulate real-world abnormal samples. Therefore, we propose a novel unified multimodal anomaly detection framework to address these issues. Our contributions consist of 3 key aspects. (1) We extract visible depth information from 3D point cloud data simply and use 2D RGB images to represent appearance, which disentangles depth and appearance to support unified anomaly generation. (2) Benefiting from the flexible input representation, the proposed Multi-Scale Gaussian Anomaly Generator and Unified Texture Anomaly Generator can generate richer anomalies in RGB and depth. (3) All modules share parameters for both RGB and depth data, effectively bridging 2D and 3D anomaly detection. Subsequent modules can directly leverage features from both modalities without complex fusion. Experiments show our method outperforms state-of-the-art (SOTA) on MVTec-3D AD and Eyecandies datasets. Code available at: https://github.com/Xantastic/BridgeNet
>
---
#### [new 084] Patch Pruning Strategy Based on Robust Statistical Measures of Attention Weight Diversity in Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决视觉Transformer中多头自注意力机制计算复杂度高的问题。通过提出一种基于注意力权重多样性的块剪枝策略，评估并去除冗余块，提升计算效率，同时保持分类准确率。方法适用于训练和推理阶段，并引入鲁棒统计量和重叠块嵌入以优化性能。**

- **链接: [http://arxiv.org/pdf/2507.19175v1](http://arxiv.org/pdf/2507.19175v1)**

> **作者:** Yuki Igaue; Hiroaki Aizawa
>
> **摘要:** Multi-head self-attention is a distinctive feature extraction mechanism of vision transformers that computes pairwise relationships among all input patches, contributing significantly to their high performance. However, it is known to incur a quadratic computational complexity with respect to the number of patches. One promising approach to address this issue is patch pruning, which improves computational efficiency by identifying and removing redundant patches. In this work, we propose a patch pruning strategy that evaluates the importance of each patch based on the variance of attention weights across multiple attention heads. This approach is inspired by the design of multi-head self-attention, which aims to capture diverse attention patterns across different subspaces of feature representations. The proposed method can be easily applied during both training and inference, and achieves improved throughput while maintaining classification accuracy in scenarios such as fine-tuning with pre-trained models. In addition, we also found that using robust statistical measures, such as the median absolute deviation in place of variance, to assess patch importance can similarly lead to strong performance. Furthermore, by introducing overlapping patch embeddings, our method achieves better performance with comparable throughput to conventional approaches that utilize all patches.
>
---
#### [new 085] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception
- **分类: cs.CV**

- **简介: 论文提出CoopTrack，用于协同顺序感知任务，解决多车协作下的3D多目标跟踪问题。其核心是端到端学习框架，通过稀疏实例级特征传输，实现跨智能体的特征提取、关联与聚合，提升了感知性能。**

- **链接: [http://arxiv.org/pdf/2507.19239v1](http://arxiv.org/pdf/2507.19239v1)**

> **作者:** Jiaru Zhong; Jiahao Wang; Jiahui Xu; Xiaofan Li; Zaiqing Nie; Haibao Yu
>
> **备注:** Accepted by ICCV 2025 (Highlight)
>
> **摘要:** Cooperative perception aims to address the inherent limitations of single-vehicle autonomous driving systems through information exchange among multiple agents. Previous research has primarily focused on single-frame perception tasks. However, the more challenging cooperative sequential perception tasks, such as cooperative 3D multi-object tracking, have not been thoroughly investigated. Therefore, we propose CoopTrack, a fully instance-level end-to-end framework for cooperative tracking, featuring learnable instance association, which fundamentally differs from existing approaches. CoopTrack transmits sparse instance-level features that significantly enhance perception capabilities while maintaining low transmission costs. Furthermore, the framework comprises two key components: Multi-Dimensional Feature Extraction, and Cross-Agent Association and Aggregation, which collectively enable comprehensive instance representation with semantic and motion features, and adaptive cross-agent association and fusion based on a feature graph. Experiments on both the V2X-Seq and Griffin datasets demonstrate that CoopTrack achieves excellent performance. Specifically, it attains state-of-the-art results on V2X-Seq, with 39.0\% mAP and 32.8\% AMOTA. The project is available at https://github.com/zhongjiaru/CoopTrack.
>
---
#### [new 086] Underwater Waste Detection Using Deep Learning A Performance Comparison of YOLOv7 to 10 and Faster RCNN
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解决水下垃圾检测问题。为应对水下污染，研究者比较了YOLOv7至YOLOv10及Faster R-CNN在水下垃圾识别中的性能。实验表明，YOLOv8以80.9%的mAP效果最佳，因其改进的架构和自监督学习，适用于复杂水下环境。**

- **链接: [http://arxiv.org/pdf/2507.18967v1](http://arxiv.org/pdf/2507.18967v1)**

> **作者:** UMMPK Nawarathne; HMNS Kumari; HMLS Kumari
>
> **备注:** 7 pages, 11 figures, to be published in International Journal of Research in Computing (IJRC)
>
> **摘要:** Underwater pollution is one of today's most significant environmental concerns, with vast volumes of garbage found in seas, rivers, and landscapes around the world. Accurate detection of these waste materials is crucial for successful waste management, environmental monitoring, and mitigation strategies. In this study, we investigated the performance of five cutting-edge object recognition algorithms, namely YOLO (You Only Look Once) models, including YOLOv7, YOLOv8, YOLOv9, YOLOv10, and Faster Region-Convolutional Neural Network (R-CNN), to identify which model was most effective at recognizing materials in underwater situations. The models were thoroughly trained and tested on a large dataset containing fifteen different classes under diverse conditions, such as low visibility and variable depths. From the above-mentioned models, YOLOv8 outperformed the others, with a mean Average Precision (mAP) of 80.9%, indicating a significant performance. This increased performance is attributed to YOLOv8's architecture, which incorporates advanced features such as improved anchor-free mechanisms and self-supervised learning, allowing for more precise and efficient recognition of items in a variety of settings. These findings highlight the YOLOv8 model's potential as an effective tool in the global fight against pollution, improving both the detection capabilities and scalability of underwater cleanup operations.
>
---
#### [new 087] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis
- **分类: cs.CV**

- **简介: 该论文属于点云分析任务，旨在解决点云数据因噪声和不完整导致的分析效果下降问题。作者提出了一种统一的点级提示方法（UPP），通过设计校正提示器和补全提示器，提升点云质量和下游任务的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.18997v1](http://arxiv.org/pdf/2507.18997v1)**

> **作者:** Zixiang Ai; Zhenyu Cui; Yuxin Peng; Jiahuan Zhou
>
> **备注:** Accepted by ICCV 2025 as a Poster
>
> **摘要:** Pre-trained point cloud analysis models have shown promising advancements in various downstream tasks, yet their effectiveness is typically suffering from low-quality point cloud (i.e., noise and incompleteness), which is a common issue in real scenarios due to casual object occlusions and unsatisfactory data collected by 3D sensors. To this end, existing methods focus on enhancing point cloud quality by developing dedicated denoising and completion models. However, due to the isolation between the point cloud enhancement and downstream tasks, these methods fail to work in various real-world domains. In addition, the conflicting objectives between denoising and completing tasks further limit the ensemble paradigm to preserve critical geometric features. To tackle the above challenges, we propose a unified point-level prompting method that reformulates point cloud denoising and completion as a prompting mechanism, enabling robust analysis in a parameter-efficient manner. We start by introducing a Rectification Prompter to adapt to noisy points through the predicted rectification vector prompts, effectively filtering noise while preserving intricate geometric features essential for accurate analysis. Sequentially, we further incorporate a Completion Prompter to generate auxiliary point prompts based on the rectified point clouds, facilitating their robustness and adaptability. Finally, a Shape-Aware Unit module is exploited to efficiently unify and capture the filtered geometric features for the downstream point cloud analysis.Extensive experiments on four datasets demonstrate the superiority and robustness of our method when handling noisy and incomplete point cloud data against existing state-of-the-art methods. Our code is released at https://github.com/zhoujiahuan1991/ICCV2025-UPP.
>
---
#### [new 088] Probing Multimodal Fusion in the Brain: The Dominance of Audiovisual Streams in Naturalistic Encoding
- **分类: cs.CV**

- **简介: 该论文属于计算神经科学任务，旨在预测大脑对自然多模态刺激的反应。研究者使用先进视觉（X-CLIP）和听觉（Whisper）特征提取模型，构建脑活动编码模型，并在分布内和分布外数据上进行测试。发现模型复杂度与泛化能力存在权衡，简单线性模型在分布外数据表现更优，且语言特征未提升预测效果。**

- **链接: [http://arxiv.org/pdf/2507.19052v1](http://arxiv.org/pdf/2507.19052v1)**

> **作者:** Hamid Abdollahi; Amir Hossein Mansouri Majoumerd; Amir Hossein Bagheri Baboukani; Amir Abolfazl Suratgar; Mohammad Bagher Menhaj
>
> **摘要:** Predicting brain activity in response to naturalistic, multimodal stimuli is a key challenge in computational neuroscience. While encoding models are becoming more powerful, their ability to generalize to truly novel contexts remains a critical, often untested, question. In this work, we developed brain encoding models using state-of-the-art visual (X-CLIP) and auditory (Whisper) feature extractors and rigorously evaluated them on both in-distribution (ID) and diverse out-of-distribution (OOD) data. Our results reveal a fundamental trade-off between model complexity and generalization: a higher-capacity attention-based model excelled on ID data, but a simpler linear model was more robust, outperforming a competitive baseline by 18\% on the OOD set. Intriguingly, we found that linguistic features did not improve predictive accuracy, suggesting that for familiar languages, neural encoding may be dominated by the continuous visual and auditory streams over redundant textual information. Spatially, our approach showed marked performance gains in the auditory cortex, underscoring the benefit of high-fidelity speech representations. Collectively, our findings demonstrate that rigorous OOD testing is essential for building robust neuro-AI models and provides nuanced insights into how model architecture, stimulus characteristics, and sensory hierarchies shape the neural encoding of our rich, multimodal world.
>
---
#### [new 089] OVFact: Measuring and Improving Open-Vocabulary Factuality for Long Caption Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决长描述生成中的事实性问题。作者提出OVFact方法，无需人工标注即可评估长描述的事实性，结合视觉定位和工具验证，提升模型生成的准确性和描述性，并通过过滤训练数据有效提高性能。**

- **链接: [http://arxiv.org/pdf/2507.19262v1](http://arxiv.org/pdf/2507.19262v1)**

> **作者:** Monika Wysoczańska; Shyamal Buch; Anurag Arnab; Cordelia Schmid
>
> **摘要:** Large vision-language models (VLMs) often struggle to generate long and factual captions. However, traditional measures for hallucination and factuality are not well suited for evaluating longer, more diverse captions and in settings where ground-truth human-annotated captions are unavailable. We introduce OV-Fact, a novel method for measuring caption factuality of long captions that leverages open-vocabulary visual grounding and tool-based verification without depending on human annotations. Our method improves agreement with human judgments and captures both caption descriptiveness (recall) and factual precision in the same metric. Furthermore, unlike previous metrics, our reference-free method design enables new applications towards factuality-based data filtering. We observe models trained on an OVFact-filtered (2.5-5x less) subset of a large-scale, noisy (VLM-generated) pretraining set meaningfully improve factuality precision without sacrificing caption descriptiveness across a range of downstream long caption benchmarks.
>
---
#### [new 090] Unstable Prompts, Unreliable Segmentations: A Challenge for Longitudinal Lesion Analysis
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决纵向病灶分析中自动化工具时间一致性差的问题。研究评估了ULS23分割模型在基线和随访CT扫描中的表现，发现其因跨扫描配准误差导致分割质量下降，并影响病灶对应关系。作者通过人为位移实验验证模型对病灶中心位置的依赖性，指出单时间点模型用于纵向分析的根本局限，提出需转向端到端的时间分析模型。**

- **链接: [http://arxiv.org/pdf/2507.19230v1](http://arxiv.org/pdf/2507.19230v1)**

> **作者:** Niels Rocholl; Ewoud Smit; Mathias Prokop; Alessa Hering
>
> **摘要:** Longitudinal lesion analysis is crucial for oncological care, yet automated tools often struggle with temporal consistency. While universal lesion segmentation models have advanced, they are typically designed for single time points. This paper investigates the performance of the ULS23 segmentation model in a longitudinal context. Using a public clinical dataset of baseline and follow-up CT scans, we evaluated the model's ability to segment and track lesions over time. We identified two critical, interconnected failure modes: a sharp degradation in segmentation quality in follow-up cases due to inter-scan registration errors, and a subsequent breakdown of the lesion correspondence process. To systematically probe this vulnerability, we conducted a controlled experiment where we artificially displaced the input volume relative to the true lesion center. Our results demonstrate that the model's performance is highly dependent on its assumption of a centered lesion; segmentation accuracy collapses when the lesion is sufficiently displaced. These findings reveal a fundamental limitation of applying single-timepoint models to longitudinal data. We conclude that robust oncological tracking requires a paradigm shift away from cascading single-purpose tools towards integrated, end-to-end models inherently designed for temporal analysis.
>
---
#### [new 091] Flow Stochastic Segmentation Networks
- **分类: cs.CV; cs.AI; stat.ML**

- **简介: 该论文属于医学图像分割任务，旨在解决现有模型在估计像素协方差时受限于低秩参数化的问题。作者提出了Flow-SSN，结合离散与连续流模型，能高效估计高秩协方差，并在医学图像分割中达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2507.18838v1](http://arxiv.org/pdf/2507.18838v1)**

> **作者:** Fabio De Sousa Ribeiro; Omar Todd; Charles Jones; Avinash Kori; Raghav Mehta; Ben Glocker
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** We introduce the Flow Stochastic Segmentation Network (Flow-SSN), a generative segmentation model family featuring discrete-time autoregressive and modern continuous-time flow variants. We prove fundamental limitations of the low-rank parameterisation of previous methods and show that Flow-SSNs can estimate arbitrarily high-rank pixel-wise covariances without assuming the rank or storing the distributional parameters. Flow-SSNs are also more efficient to sample from than standard diffusion-based segmentation models, thanks to most of the model capacity being allocated to learning the base distribution of the flow, constituting an expressive prior. We apply Flow-SSNs to challenging medical imaging benchmarks and achieve state-of-the-art results. Code available: https://github.com/biomedia-mira/flow-ssn.
>
---
#### [new 092] Multistream Network for LiDAR and Camera-based 3D Object Detection in Outdoor Scenes
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D目标检测任务，旨在解决户外场景中融合LiDAR与RGB数据提升检测精度的问题。论文提出了一种多流网络（MuStD），通过三个分支提取并融合多模态特征，最终在KITTI数据集上取得了优异性能。**

- **链接: [http://arxiv.org/pdf/2507.19304v1](http://arxiv.org/pdf/2507.19304v1)**

> **作者:** Muhammad Ibrahim; Naveed Akhtar; Haitian Wang; Saeed Anwar; Ajmal Mian
>
> **备注:** This paper has been accepted by IEEE/RSJ IROS 2025 for oral presentation on 19 Oct. 2025
>
> **摘要:** Fusion of LiDAR and RGB data has the potential to enhance outdoor 3D object detection accuracy. To address real-world challenges in outdoor 3D object detection, fusion of LiDAR and RGB input has started gaining traction. However, effective integration of these modalities for precise object detection task still remains a largely open problem. To address that, we propose a MultiStream Detection (MuStD) network, that meticulously extracts task-relevant information from both data modalities. The network follows a three-stream structure. Its LiDAR-PillarNet stream extracts sparse 2D pillar features from the LiDAR input while the LiDAR-Height Compression stream computes Bird's-Eye View features. An additional 3D Multimodal stream combines RGB and LiDAR features using UV mapping and polar coordinate indexing. Eventually, the features containing comprehensive spatial, textural and geometric information are carefully fused and fed to a detection head for 3D object detection. Our extensive evaluation on the challenging KITTI Object Detection Benchmark using public testing server at https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=d162ec699d6992040e34314d19ab7f5c217075e0 establishes the efficacy of our method by achieving new state-of-the-art or highly competitive results in different categories while remaining among the most efficient methods. Our code will be released through MuStD GitHub repository at https://github.com/IbrahimUWA/MuStD.git
>
---
#### [new 093] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars
- **分类: cs.CV**

- **简介: 该论文属于3D头像生成任务，旨在解决头发与面部表示难以分离、缺乏灵活性的问题。作者提出了HairCUP模型，通过合成无头发数据，分别学习面部与头发的潜在空间，实现3D头像的可控组合与交换。**

- **链接: [http://arxiv.org/pdf/2507.19481v1](http://arxiv.org/pdf/2507.19481v1)**

> **作者:** Byungjun Kim; Shunsuke Saito; Giljoo Nam; Tomas Simon; Jason Saragih; Hanbyul Joo; Junxuan Li
>
> **备注:** ICCV 2025. Project Page: https://bjkim95.github.io/haircup/
>
> **摘要:** We present a universal prior model for 3D head avatars with explicit hair compositionality. Existing approaches to build generalizable priors for 3D head avatars often adopt a holistic modeling approach, treating the face and hair as an inseparable entity. This overlooks the inherent compositionality of the human head, making it difficult for the model to naturally disentangle face and hair representations, especially when the dataset is limited. Furthermore, such holistic models struggle to support applications like 3D face and hairstyle swapping in a flexible and controllable manner. To address these challenges, we introduce a prior model that explicitly accounts for the compositionality of face and hair, learning their latent spaces separately. A key enabler of this approach is our synthetic hairless data creation pipeline, which removes hair from studio-captured datasets using estimated hairless geometry and texture derived from a diffusion prior. By leveraging a paired dataset of hair and hairless captures, we train disentangled prior models for face and hair, incorporating compositionality as an inductive bias to facilitate effective separation. Our model's inherent compositionality enables seamless transfer of face and hair components between avatars while preserving identity. Additionally, we demonstrate that our model can be fine-tuned in a few-shot manner using monocular captures to create high-fidelity, hair-compositional 3D head avatars for unseen subjects. These capabilities highlight the practical applicability of our approach in real-world scenarios, paving the way for flexible and expressive 3D avatar generation.
>
---
#### [new 094] Part Segmentation of Human Meshes via Multi-View Human Parsing
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于三维人体网格语义分割任务，旨在解决仅基于几何信息对人体网格进行细粒度部分分割的问题。作者构建了伪标签数据集，采用多视角分割与反投影生成标注，并提出一种高效采样策略与基于PointTransformer的几何分割方法，实现了无需纹理的人体网格解析。**

- **链接: [http://arxiv.org/pdf/2507.18655v1](http://arxiv.org/pdf/2507.18655v1)**

> **作者:** James Dickens; Kamyar Hamad
>
> **摘要:** Recent advances in point cloud deep learning have led to models that achieve high per-part labeling accuracy on large-scale point clouds, using only the raw geometry of unordered point sets. In parallel, the field of human parsing focuses on predicting body part and clothing/accessory labels from images. This work aims to bridge these two domains by enabling per-vertex semantic segmentation of large-scale human meshes. To achieve this, a pseudo-ground truth labeling pipeline is developed for the Thuman2.1 dataset: meshes are first aligned to a canonical pose, segmented from multiple viewpoints, and the resulting point-level labels are then backprojected onto the original mesh to produce per-point pseudo ground truth annotations. Subsequently, a novel, memory-efficient sampling strategy is introduced, a windowed iterative farthest point sampling (FPS) with space-filling curve-based serialization to effectively downsample the point clouds. This is followed by a purely geometric segmentation using PointTransformer, enabling semantic parsing of human meshes without relying on texture information. Experimental results confirm the effectiveness and accuracy of the proposed approach.
>
---
#### [new 095] Fuzzy Theory in Computer Vision: A Review
- **分类: cs.CV**

- **简介: 该论文属于综述任务，旨在探讨模糊逻辑在计算机视觉中的应用。它要解决图像数据中的不确定性、噪声和模糊性问题。论文总结了模糊聚类、模糊推理系统等技术，并讨论了其在图像分割、目标识别等任务中的应用，以及与深度学习的结合趋势。**

- **链接: [http://arxiv.org/pdf/2507.18660v1](http://arxiv.org/pdf/2507.18660v1)**

> **作者:** Adilet Yerkin; Ayan Igali; Elnara Kadyrgali; Maksat Shagyrov; Malika Ziyada; Muragul Muratbekova; Pakizar Shamoi
>
> **备注:** Submitted to Journal of Intelligent and Fuzzy Systems for consideration (8 pages, 6 figures, 1 table)
>
> **摘要:** Computer vision applications are omnipresent nowadays. The current paper explores the use of fuzzy logic in computer vision, stressing its role in handling uncertainty, noise, and imprecision in image data. Fuzzy logic is able to model gradual transitions and human-like reasoning and provides a promising approach to computer vision. Fuzzy approaches offer a way to improve object recognition, image segmentation, and feature extraction by providing more adaptable and interpretable solutions compared to traditional methods. We discuss key fuzzy techniques, including fuzzy clustering, fuzzy inference systems, type-2 fuzzy sets, and fuzzy rule-based decision-making. The paper also discusses various applications, including medical imaging, autonomous systems, and industrial inspection. Additionally, we explore the integration of fuzzy logic with deep learning models such as convolutional neural networks (CNNs) to enhance performance in complex vision tasks. Finally, we examine emerging trends such as hybrid fuzzy-deep learning models and explainable AI.
>
---
#### [new 096] Closing the Modality Gap for Mixed Modality Search
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于混合模态搜索任务，旨在解决对比视觉-语言模型（如CLIP）在跨模态检索中存在的模态差距问题。作者提出GR-CLIP方法，有效缩小模态差距，显著提升检索性能。**

- **链接: [http://arxiv.org/pdf/2507.19054v1](http://arxiv.org/pdf/2507.19054v1)**

> **作者:** Binxu Li; Yuhui Zhang; Xiaohan Wang; Weixin Liang; Ludwig Schmidt; Serena Yeung-Levy
>
> **备注:** Project page: https://yuhui-zh15.github.io/MixedModalitySearch/
>
> **摘要:** Mixed modality search -- retrieving information across a heterogeneous corpus composed of images, texts, and multimodal documents -- is an important yet underexplored real-world application. In this work, we investigate how contrastive vision-language models, such as CLIP, perform on the mixed modality search task. Our analysis reveals a critical limitation: these models exhibit a pronounced modality gap in the embedding space, where image and text embeddings form distinct clusters, leading to intra-modal ranking bias and inter-modal fusion failure. To address this issue, we propose GR-CLIP, a lightweight post-hoc calibration method that removes the modality gap in CLIP's embedding space. Evaluated on MixBench -- the first benchmark specifically designed for mixed modality search -- GR-CLIP improves NDCG@10 by up to 26 percentage points over CLIP, surpasses recent vision-language generative embedding models by 4 percentage points, while using 75x less compute.
>
---
#### [new 097] PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups
- **分类: cs.CV**

- **简介: 该论文属于多角色群体交互生成任务，旨在解决群体规模扩大时生成真实、可控交互动作的难题。现有方法依赖共享提示，限制了细节控制。论文提出PINO，通过分解为成对交互并使用预训练模型，实现任意规模群体的长时、可定制动作生成，并引入物理惩罚优化噪声，提升真实感与合理性。**

- **链接: [http://arxiv.org/pdf/2507.19292v1](http://arxiv.org/pdf/2507.19292v1)**

> **作者:** Sakuya Ota; Qing Yu; Kent Fujiwara; Satoshi Ikehata; Ikuro Sato
>
> **备注:** Accepted to ICCV 2025, Project page: https://sinc865.github.io/pino/
>
> **摘要:** Generating realistic group interactions involving multiple characters remains challenging due to increasing complexity as group size expands. While existing conditional diffusion models incrementally generate motions by conditioning on previously generated characters, they rely on single shared prompts, limiting nuanced control and leading to overly simplified interactions. In this paper, we introduce Person-Interaction Noise Optimization (PINO), a novel, training-free framework designed for generating realistic and customizable interactions among groups of arbitrary size. PINO decomposes complex group interactions into semantically relevant pairwise interactions, and leverages pretrained two-person interaction diffusion models to incrementally compose group interactions. To ensure physical plausibility and avoid common artifacts such as overlapping or penetration between characters, PINO employs physics-based penalties during noise optimization. This approach allows precise user control over character orientation, speed, and spatial relationships without additional training. Comprehensive evaluations demonstrate that PINO generates visually realistic, physically coherent, and adaptable multi-person interactions suitable for diverse animation, gaming, and robotics applications.
>
---
#### [new 098] Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene
- **分类: cs.CV**

- **简介: 该论文属于多智能体行为生成任务，旨在解决生成多角色与场景互动的动态三维场景问题。论文提出一种基于事件驱动的框架，利用大语言模型解析文本上下文，分解任务为可操作子问题，生成大规模、多样化的多角色行为，并提供相关基准测试。**

- **链接: [http://arxiv.org/pdf/2507.19232v1](http://arxiv.org/pdf/2507.19232v1)**

> **作者:** Donggeun Lim; Jinseok Bae; Inwoo Hwang; Seungmin Lee; Hwanhee Lee; Young Min Kim
>
> **备注:** 16 pages, project page: https://rms0329.github.io/Event-Driven-Storytelling/
>
> **摘要:** In this work, we propose a framework that creates a lively virtual dynamic scene with contextual motions of multiple humans. Generating multi-human contextual motion requires holistic reasoning over dynamic relationships among human-human and human-scene interactions. We adapt the power of a large language model (LLM) to digest the contextual complexity within textual input and convert the task into tangible subproblems such that we can generate multi-agent behavior beyond the scale that was not considered before. Specifically, our event generator formulates the temporal progression of a dynamic scene into a sequence of small events. Each event calls for a well-defined motion involving relevant characters and objects. Next, we synthesize the motions of characters at positions sampled based on spatial guidance. We employ a high-level module to deliver scalable yet comprehensive context, translating events into relative descriptions that enable the retrieval of precise coordinates. As the first to address this problem at scale and with diversity, we offer a benchmark to assess diverse aspects of contextual reasoning. Benchmark results and user studies show that our framework effectively captures scene context with high scalability. The code and benchmark, along with result videos, are available at our project page: https://rms0329.github.io/Event-Driven-Storytelling/.
>
---
#### [new 099] Negation-Aware Test-Time Adaptation for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决模型对否定语义理解不足的问题。现有方法依赖大量否定数据进行微调，成本高。作者提出NEAT方法，在推理阶段调整分布相关参数，缓解否定与肯定间的概念偏移，提升否定理解效果。**

- **链接: [http://arxiv.org/pdf/2507.19064v1](http://arxiv.org/pdf/2507.19064v1)**

> **作者:** Haochen Han; Alex Jinpeng Wang; Fangming Liu
>
> **备注:** This paper will be submitted to the IEEE for possible publication
>
> **摘要:** In this paper, we study a practical but less-touched problem in Vision-Language Models (VLMs), \ie, negation understanding. Specifically, many real-world applications require models to explicitly identify what is false or non-existent, \eg, radiologists may search for images that exclude specific conditions. Despite the impressive transferability of VLMs through large-scale training, they suffer from a critical limitation that fails to handle negation. To address this challenge, existing methods attribute its root cause to the scarcity of negation training data and propose to fine-tune VLMs on massive data containing explicit negation. Undoubtedly, such data-centric solutions demand substantial data and computational resources, limiting their sustainable widespread adoption. To tackle negation in a low-carbon manner, we empirically observe that the key obstacle lies in the dual-concept shifts between the affirmation and negation distributions. Therefore, we propose a Negation-Aware Test-Time Adaptation (NEAT) method to efficiently adjust distribution-related parameters during inference. In brief, NEAT can reduce distribution shift in consistent semantics while eliminating false distributional consistency in unrelated semantics. Extensive experiments on the various negation understanding tasks verify the effectiveness of the proposed method. The code is available at https://github.com/hhc1997/NEAT.
>
---
#### [new 100] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment
- **分类: cs.CV**

- **简介: 该论文属于图像生成评价任务，旨在解决现有评估模型无法准确反映人类审美偏好、低估高质量图像的问题。作者提出了ICT评分和HP评分模型，提升文本-图像对齐与美学评估的准确性，优化图像生成效果。**

- **链接: [http://arxiv.org/pdf/2507.19002v1](http://arxiv.org/pdf/2507.19002v1)**

> **作者:** Ying Ba; Tianyu Zhang; Yalong Bai; Wenyi Mo; Tao Liang; Bing Su; Ji-Rong Wen
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Contemporary image generation systems have achieved high fidelity and superior aesthetic quality beyond basic text-image alignment. However, existing evaluation frameworks have failed to evolve in parallel. This study reveals that human preference reward models fine-tuned based on CLIP and BLIP architectures have inherent flaws: they inappropriately assign low scores to images with rich details and high aesthetic value, creating a significant discrepancy with actual human aesthetic preferences. To address this issue, we design a novel evaluation score, ICT (Image-Contained-Text) score, that achieves and surpasses the objectives of text-image alignment by assessing the degree to which images represent textual content. Building upon this foundation, we further train an HP (High-Preference) score model using solely the image modality to enhance image aesthetics and detail quality while maintaining text-image alignment. Experiments demonstrate that the proposed evaluation model improves scoring accuracy by over 10\% compared to existing methods, and achieves significant results in optimizing state-of-the-art text-to-image models. This research provides theoretical and empirical support for evolving image generation technology toward higher-order human aesthetic preferences. Code is available at https://github.com/BarretBa/ICTHP.
>
---
#### [new 101] CXR-CML: Improved zero-shot classification of long-tailed multi-label diseases in Chest X-Rays
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决胸片图像中长尾多标签疾病分类的类别不平衡问题。作者提出CXR-CML方法，通过在潜在空间中对类别进行高斯混合聚类与学生t-分布优化，并引入度量损失，提升了零样本分类性能，尤其改善了罕见类别的识别效果。**

- **链接: [http://arxiv.org/pdf/2507.19398v1](http://arxiv.org/pdf/2507.19398v1)**

> **作者:** Rajesh Madhipati; Sheethal Bhat; Lukas Buess; Andreas Maier
>
> **摘要:** Chest radiography (CXR) plays a crucial role in the diagnosis of various diseases. However, the inherent class imbalance in the distribution of clinical findings presents a significant challenge for current self-supervised deep learning models. These models often fail to accurately classify long-tailed classes. Current Vision-Language models such as Contrastive Language Image Pre-training (CLIP) models effectively model the manifold distribution of the latent space, enabling high zero-shot classification accuracies. Although CLIP performs well on most of the primary classes in the dataset, our work reveals that its effectiveness decreases significantly for classes with a long-tailed distribution. Our approach employs a class-weighting mechanism that directly aligns with the distribution of classes within the latent space. This method ensures a substantial improvement in overall classification performance, with particular emphasis on enhancing the recognition and accuracy of rarely observed classes. We accomplish this by applying Gaussian Mixture Model (GMM) clustering to the latent space. The subsequent clusters are further refined by Student t-distribution, followed by a metric loss that utilizes the altered embeddings. Our approach facilitates stable and adaptive clustering of the features. This results in a notable average improvement of 7\% points in zero-shot AUC scores across 40 classes in the MIMIC-CXR-JPG dataset from previous SOTA models.
>
---
#### [new 102] SAM2-Aug: Prior knowledge-based Augmentation for Target Volume Auto-Segmentation in Adaptive Radiation Therapy Using Segment Anything Model 2
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像分割任务，旨在解决自适应放疗中肿瘤分割准确率低、依赖人工的问题。作者基于SAM2模型，引入先验图像和增强提示策略，提升分割精度和泛化能力，实验验证于多个数据集，效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.19282v1](http://arxiv.org/pdf/2507.19282v1)**

> **作者:** Guoping Xu; Yan Dai; Hengrui Zhao; Ying Zhang; Jie Deng; Weiguo Lu; You Zhang
>
> **备注:** 26 pages, 10 figures
>
> **摘要:** Purpose: Accurate tumor segmentation is vital for adaptive radiation therapy (ART) but remains time-consuming and user-dependent. Segment Anything Model 2 (SAM2) shows promise for prompt-based segmentation but struggles with tumor accuracy. We propose prior knowledge-based augmentation strategies to enhance SAM2 for ART. Methods: Two strategies were introduced to improve SAM2: (1) using prior MR images and annotations as contextual inputs, and (2) improving prompt robustness via random bounding box expansion and mask erosion/dilation. The resulting model, SAM2-Aug, was fine-tuned and tested on the One-Seq-Liver dataset (115 MRIs from 31 liver cancer patients), and evaluated without retraining on Mix-Seq-Abdomen (88 MRIs, 28 patients) and Mix-Seq-Brain (86 MRIs, 37 patients). Results: SAM2-Aug outperformed convolutional, transformer-based, and prompt-driven models across all datasets, achieving Dice scores of 0.86(liver), 0.89(abdomen), and 0.90(brain). It demonstrated strong generalization across tumor types and imaging sequences, with improved performance in boundary-sensitive metrics. Conclusions: Incorporating prior images and enhancing prompt diversity significantly boosts segmentation accuracy and generalizability. SAM2-Aug offers a robust, efficient solution for tumor segmentation in ART. Code and models will be released at https://github.com/apple1986/SAM2-Aug.
>
---
#### [new 103] How good are humans at detecting AI-generated images? Learnings from an experiment
- **分类: cs.HC; cs.AI; cs.CV**

- **简介: 该论文属于图像真实性识别任务，旨在研究人类区分AI生成图像与真实图像的能力。通过分析12,500名参与者对287,000张图像的判断，发现人类整体准确率仅为62%，尤其在风景类图像上表现较差。研究强调了开发透明工具以应对AI图像引发的信息风险。**

- **链接: [http://arxiv.org/pdf/2507.18640v1](http://arxiv.org/pdf/2507.18640v1)**

> **作者:** Thomas Roca; Anthony Cintron Roman; Jehú Torres Vega; Marcelo Duarte; Pengce Wang; Kevin White; Amit Misra; Juan Lavista Ferres
>
> **摘要:** As AI-powered image generation improves, a key question is how well human beings can differentiate between "real" and AI-generated or modified images. Using data collected from the online game "Real or Not Quiz.", this study investigates how effectively people can distinguish AI-generated images from real ones. Participants viewed a randomized set of real and AI-generated images, aiming to identify their authenticity. Analysis of approximately 287,000 image evaluations by over 12,500 global participants revealed an overall success rate of only 62\%, indicating a modest ability, slightly above chance. Participants were most accurate with human portraits but struggled significantly with natural and urban landscapes. These results highlight the inherent challenge humans face in distinguishing AI-generated visual content, particularly images without obvious artifacts or stylistic cues. This study stresses the need for transparency tools, such as watermarks and robust AI detection tools to mitigate the risks of misinformation arising from AI-generated content
>
---
#### [new 104] Mining Contextualized Visual Associations from Images for Creativity Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言理解任务，旨在解决现有模型缺乏抽象关联表达的问题。作者提出一种从图像中挖掘上下文视觉关联的方法，并构建了包含170万创意描述的MSCOCO数据集。通过生成具抽象性的描述并提升创意领域零样本检索效果，推动创造力理解。**

- **链接: [http://arxiv.org/pdf/2507.18915v1](http://arxiv.org/pdf/2507.18915v1)**

> **作者:** Ananya Sahu; Amith Ananthram; Kathleen McKeown
>
> **摘要:** Understanding another person's creative output requires a shared language of association. However, when training vision-language models such as CLIP, we rely on web-scraped datasets containing short, predominantly literal, alt-text. In this work, we introduce a method for mining contextualized associations for salient visual elements in an image that can scale to any unlabeled dataset. Given an image, we can use these mined associations to generate high quality creative captions at increasing degrees of abstraction. With our method, we produce a new dataset of visual associations and 1.7m creative captions for the images in MSCOCO. Human evaluation confirms that these captions remain visually grounded while exhibiting recognizably increasing abstraction. Moreover, fine-tuning a visual encoder on this dataset yields meaningful improvements in zero-shot image-text retrieval in two creative domains: poetry and metaphor visualization. We release our dataset, our generation code and our models for use by the broader community.
>
---
#### [new 105] Face2VoiceSync: Lightweight Face-Voice Consistency for Text-Driven Talking Face Generation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于文本驱动的说话人脸生成任务，旨在解决语音与人脸外貌不匹配的问题。论文提出了Face2VoiceSync框架，实现人脸与语音的一致性生成，并在生成质量、多样性、可控性及训练效率方面取得进展。**

- **链接: [http://arxiv.org/pdf/2507.19225v1](http://arxiv.org/pdf/2507.19225v1)**

> **作者:** Fang Kang; Yin Cao; Haoyu Chen
>
> **摘要:** Recent studies in speech-driven talking face generation achieve promising results, but their reliance on fixed-driven speech limits further applications (e.g., face-voice mismatch). Thus, we extend the task to a more challenging setting: given a face image and text to speak, generating both talking face animation and its corresponding speeches. Accordingly, we propose a novel framework, Face2VoiceSync, with several novel contributions: 1) Voice-Face Alignment, ensuring generated voices match facial appearance; 2) Diversity \& Manipulation, enabling generated voice control over paralinguistic features space; 3) Efficient Training, using a lightweight VAE to bridge visual and audio large-pretrained models, with significantly fewer trainable parameters than existing methods; 4) New Evaluation Metric, fairly assessing the diversity and identity consistency. Experiments show Face2VoiceSync achieves both visual and audio state-of-the-art performances on a single 40GB GPU.
>
---
#### [new 106] RealisVSR: Detail-enhanced Diffusion for Real-World 4K Video Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于视频超分辨率（VSR）任务，旨在解决真实世界4K视频中高频细节恢复不足、时间动态建模不一致以及缺乏高质量评估数据的问题。论文提出RealisVSR模型，包含一致性保留控制网络、高频修复扩散损失，并构建首个4K VSR公开基准RealisVideo-4K，显著提升细节增强效果。**

- **链接: [http://arxiv.org/pdf/2507.19138v1](http://arxiv.org/pdf/2507.19138v1)**

> **作者:** Weisong Zhao; Jingkai Zhou; Xiangyu Zhu; Weihua Chen; Xiao-Yu Zhang; Zhen Lei; Fan Wang
>
> **摘要:** Video Super-Resolution (VSR) has achieved significant progress through diffusion models, effectively addressing the over-smoothing issues inherent in GAN-based methods. Despite recent advances, three critical challenges persist in VSR community: 1) Inconsistent modeling of temporal dynamics in foundational models; 2) limited high-frequency detail recovery under complex real-world degradations; and 3) insufficient evaluation of detail enhancement and 4K super-resolution, as current methods primarily rely on 720P datasets with inadequate details. To address these challenges, we propose RealisVSR, a high-frequency detail-enhanced video diffusion model with three core innovations: 1) Consistency Preserved ControlNet (CPC) architecture integrated with the Wan2.1 video diffusion to model the smooth and complex motions and suppress artifacts; 2) High-Frequency Rectified Diffusion Loss (HR-Loss) combining wavelet decomposition and HOG feature constraints for texture restoration; 3) RealisVideo-4K, the first public 4K VSR benchmark containing 1,000 high-definition video-text pairs. Leveraging the advanced spatio-temporal guidance of Wan2.1, our method requires only 5-25% of the training data volume compared to existing approaches. Extensive experiments on VSR benchmarks (REDS, SPMCS, UDM10, YouTube-HQ, VideoLQ, RealisVideo-720P) demonstrate our superiority, particularly in ultra-high-resolution scenarios.
>
---
#### [new 107] Diffusion Models for Solving Inverse Problems via Posterior Sampling with Piecewise Guidance
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图像恢复任务，旨在解决图像修复和超分辨率等逆问题。提出了一种基于扩散模型的分段引导框架，通过在不同去噪阶段采用不同近似方法，提升计算效率并保持精度。方法无需针对特定任务重新训练，能灵活适应多种逆问题，并考虑测量噪声的影响。实验表明其在推理速度和重建质量上均表现优异。**

- **链接: [http://arxiv.org/pdf/2507.18654v1](http://arxiv.org/pdf/2507.18654v1)**

> **作者:** Saeed Mohseni-Sehdeh; Walid Saad; Kei Sakaguchi; Tao Yu
>
> **摘要:** Diffusion models are powerful tools for sampling from high-dimensional distributions by progressively transforming pure noise into structured data through a denoising process. When equipped with a guidance mechanism, these models can also generate samples from conditional distributions. In this paper, a novel diffusion-based framework is introduced for solving inverse problems using a piecewise guidance scheme. The guidance term is defined as a piecewise function of the diffusion timestep, facilitating the use of different approximations during high-noise and low-noise phases. This design is shown to effectively balance computational efficiency with the accuracy of the guidance term. Unlike task-specific approaches that require retraining for each problem, the proposed method is problem-agnostic and readily adaptable to a variety of inverse problems. Additionally, it explicitly incorporates measurement noise into the reconstruction process. The effectiveness of the proposed framework is demonstrated through extensive experiments on image restoration tasks, specifically image inpainting and super-resolution. Using a class conditional diffusion model for recovery, compared to the \pgdm baseline, the proposed framework achieves a reduction in inference time of \(25\%\) for inpainting with both random and center masks, and \(23\%\) and \(24\%\) for \(4\times\) and \(8\times\) super-resolution tasks, respectively, while incurring only negligible loss in PSNR and SSIM.
>
---
#### [new 108] PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于车载驾驶员生理监测任务，旨在解决远程生理测量在真实驾驶场景中缺乏全面数据集的问题。论文构建了PhysDrive数据集，包含多模态传感器数据及生理信号标注，覆盖多种驾驶条件，并提供了基准评估与开源代码。**

- **链接: [http://arxiv.org/pdf/2507.19172v1](http://arxiv.org/pdf/2507.19172v1)**

> **作者:** Jiyao Wang; Xiao Yang; Qingyong Hu; Jiankai Tang; Can Liu; Dengbo He; Yuntao Wang; Yingcong Chen; Kaishun Wu
>
> **备注:** It is the initial version, not the final version
>
> **摘要:** Robust and unobtrusive in-vehicle physiological monitoring is crucial for ensuring driving safety and user experience. While remote physiological measurement (RPM) offers a promising non-invasive solution, its translation to real-world driving scenarios is critically constrained by the scarcity of comprehensive datasets. Existing resources are often limited in scale, modality diversity, the breadth of biometric annotations, and the range of captured conditions, thereby omitting inherent real-world challenges in driving. Here, we present PhysDrive, the first large-scale multimodal dataset for contactless in-vehicle physiological sensing with dedicated consideration on various modality settings and driving factors. PhysDrive collects data from 48 drivers, including synchronized RGB, near-infrared camera, and raw mmWave radar data, accompanied with six synchronized ground truths (ECG, BVP, Respiration, HR, RR, and SpO2). It covers a wide spectrum of naturalistic driving conditions, including driver motions, dynamic natural light, vehicle types, and road conditions. We extensively evaluate both signal-processing and deep-learning methods on PhysDrive, establishing a comprehensive benchmark across all modalities, and release full open-source code with compatibility for mainstream public toolboxes. We envision PhysDrive will serve as a foundational resource and accelerate research on multimodal driver monitoring and smart-cockpit systems.
>
---
#### [new 109] Perpetua: Multi-Hypothesis Persistence Modeling for Semi-Static Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于环境建模任务，旨在解决动态环境中半静态特征的状态预测问题。作者提出了Perpetua方法，通过贝叶斯框架结合“持续”和“出现”滤波器链，实现对环境特征多假设建模，能适应时间变化并预测未来状态，提升了准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.18808v1](http://arxiv.org/pdf/2507.18808v1)**

> **作者:** Miguel Saavedra-Ruiz; Samer B. Nashed; Charlie Gauthier; Liam Paull
>
> **备注:** Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) Code available at https://github.com/montrealrobotics/perpetua-code. Webpage and additional videos at https://montrealrobotics.ca/perpetua/
>
> **摘要:** Many robotic systems require extended deployments in complex, dynamic environments. In such deployments, parts of the environment may change between subsequent robot observations. Most robotic mapping or environment modeling algorithms are incapable of representing dynamic features in a way that enables predicting their future state. Instead, they opt to filter certain state observations, either by removing them or some form of weighted averaging. This paper introduces Perpetua, a method for modeling the dynamics of semi-static features. Perpetua is able to: incorporate prior knowledge about the dynamics of the feature if it exists, track multiple hypotheses, and adapt over time to enable predicting of future feature states. Specifically, we chain together mixtures of "persistence" and "emergence" filters to model the probability that features will disappear or reappear in a formal Bayesian framework. The approach is an efficient, scalable, general, and robust method for estimating the states of features in an environment, both in the present as well as at arbitrary future times. Through experiments on simulated and real-world data, we find that Perpetua yields better accuracy than similar approaches while also being online adaptable and robust to missing observations.
>
---
#### [new 110] OS-MAP: How Far Can Computer-Using Agents Go in Breadth and Depth?
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于计算机使用代理任务，旨在解决现有基准未能充分评估代理在多样化任务中的自动化水平与泛化能力的问题。作者构建了OS-MAP基准，包含416个真实任务，从自动化层级和需求层次两个维度评估代理能力，以推动代理研究与实际应用的发展。**

- **链接: [http://arxiv.org/pdf/2507.19132v1](http://arxiv.org/pdf/2507.19132v1)**

> **作者:** Xuetian Chen; Yinghao Chen; Xinfeng Yuan; Zhuo Peng; Lu Chen; Yuekeng Li; Zhoujia Zhang; Yingqian Huang; Leyan Huang; Jiaqing Liang; Tianbao Xie; Zhiyong Wu; Qiushi Sun; Biqing Qi; Bowen Zhou
>
> **备注:** Work in progress
>
> **摘要:** Computer-using agents have shown strong potential to boost human productivity and enable new application forms across platforms. While recent advances have led to usable applications, existing benchmarks fail to account for the internal task heterogeneity and the corresponding agent capabilities, as well as their alignment with actual user demands-hindering both targeted capability development and the reliable transition of research progress into practical deployment. To bridge the gap, we present OS-MAP, a benchmark for daily computer-using automation that organizes its 416 realistic tasks across 15 applications along two key dimensions: a five-level taxonomy of automation and a generalization scope derived from a real-world user demand hierarchy. To enable fine-grained analysis of required capabilities and alignment with real-world scenarios, OS-MAP evaluates agents along two dimensions: automation level across a five-level taxonomy, and generalization scope across a demand hierarchy. This design captures varying levels of required agent autonomy and generalization, forming a performance-generalization evaluation matrix for structured and comprehensive assessment. Experiments show that even State-of-the-Art agents with VLM backbones struggle with higher-level tasks involving perception, reasoning, and coordination-highlighting the need for a deeper understanding of current strengths and limitations to drive the future progress in computer-using agents research and deployment. All code, environments, baselines, and data are publicly available at https://github.com/OS-Copilot/OS-Map.
>
---
#### [new 111] WACA-UNet: Weakness-Aware Channel Attention for Static IR Drop Prediction in Integrated Circuit Design
- **分类: cs.LG; cs.AI; cs.CV; B.7.2; I.5.1; I.2.10; I.5.4**

- **简介: 该论文属于集成电路设计中的电源完整性分析任务，旨在解决静态IR drop预测问题。传统方法计算成本高，而现有学习方法未考虑输入特征的重要性差异。作者提出WACA机制，通过注意力网络增强弱特征、抑制强特征，提升预测精度。在ICCAD-2023数据集上表现优于竞赛冠军模型。**

- **链接: [http://arxiv.org/pdf/2507.19197v1](http://arxiv.org/pdf/2507.19197v1)**

> **作者:** Youngmin Seo; Yunhyeong Kwon; Younghun Park; HwiRyong Kim; Seungho Eum; Jinha Kim; Taigon Song; Juho Kim; Unsang Park
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Accurate spatial prediction of power integrity issues, such as IR drop, is critical for reliable VLSI design. However, traditional simulation-based solvers are computationally expensive and difficult to scale. We address this challenge by reformulating IR drop estimation as a pixel-wise regression task on heterogeneous multi-channel physical maps derived from circuit layouts. Prior learning-based methods treat all input layers (e.g., metal, via, and current maps) equally, ignoring their varying importance to prediction accuracy. To tackle this, we propose a novel Weakness-Aware Channel Attention (WACA) mechanism, which recursively enhances weak feature channels while suppressing over-dominant ones through a two-stage gating strategy. Integrated into a ConvNeXtV2-based attention U-Net, our approach enables adaptive and balanced feature representation. On the public ICCAD-2023 benchmark, our method outperforms the ICCAD-2023 contest winner by reducing mean absolute error by 61.1% and improving F1-score by 71.0%. These results demonstrate that channel-wise heterogeneity is a key inductive bias in physical layout analysis for VLSI.
>
---
#### [new 112] PGKET: A Photonic Gaussian Kernel Enhanced Transformer
- **分类: quant-ph; cs.CV**

- **简介: 该论文属于机器学习与光子计算交叉任务，旨在解决传统自注意力机制处理长序列效率低的问题。作者提出了PGKET模型，基于光子高斯核增强自注意力机制（PGKSAM），利用光子干涉与叠加技术并行处理输入，提升了模型性能与计算效率。实验表明其在多分类任务上优于现有模型。**

- **链接: [http://arxiv.org/pdf/2507.19041v1](http://arxiv.org/pdf/2507.19041v1)**

> **作者:** Ren-Xin Zhao
>
> **摘要:** Self-Attention Mechanisms (SAMs) enhance model performance by extracting key information but are inefficient when dealing with long sequences. To this end, a photonic Gaussian Kernel Enhanced Transformer (PGKET) is proposed, based on the Photonic Gaussian Kernel Self-Attention Mechanism (PGKSAM). The PGKSAM calculates the Photonic Gaussian Kernel Self-Attention Score (PGKSAS) using photon interferometry and superposition to process multiple inputs in parallel. Experimental results show that PGKET outperforms some state-of-the-art transformers in multi-classification tasks on MedMNIST v2 and CIFAR-10, and is expected to improve performance in complex tasks and accelerate the convergence of Photonic Computing (PC) and machine learning.
>
---
#### [new 113] Fine-Grained Traffic Inference from Road to Lane via Spatio-Temporal Graph Node Generation
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于细粒度交通推断任务（FRTI），旨在通过有限的道路数据生成更详细的车道级交通信息，以解决车道级数据获取困难的问题。作者提出了RoadDiff框架，结合道路-车道相关性与扩散模型，有效推断车道交通状态，提升交通管理精度。**

- **链接: [http://arxiv.org/pdf/2507.19089v1](http://arxiv.org/pdf/2507.19089v1)**

> **作者:** Shuhao Li; Weidong Yang; Yue Cui; Xiaoxing Liu; Lingkai Meng; Lipeng Ma; Fan Zhang
>
> **摘要:** Fine-grained traffic management and prediction are fundamental to key applications such as autonomous driving, lane change guidance, and traffic signal control. However, obtaining lane-level traffic data has become a critical bottleneck for data-driven models due to limitations in the types and number of sensors and issues with the accuracy of tracking algorithms. To address this, we propose the Fine-grained Road Traffic Inference (FRTI) task, which aims to generate more detailed lane-level traffic information using limited road data, providing a more energy-efficient and cost-effective solution for precise traffic management. This task is abstracted as the first scene of the spatio-temporal graph node generation problem. We designed a two-stage framework--RoadDiff--to solve the FRTI task. solve the FRTI task. This framework leverages the Road-Lane Correlation Autoencoder-Decoder and the Lane Diffusion Module to fully utilize the limited spatio-temporal dependencies and distribution relationships of road data to accurately infer fine-grained lane traffic states. Based on existing research, we designed several baseline models with the potential to solve the FRTI task and conducted extensive experiments on six datasets representing different road conditions to validate the effectiveness of the RoadDiff model in addressing the FRTI task. The relevant datasets and code are available at https://github.com/ShuhaoLii/RoadDiff.
>
---
#### [new 114] Enhancing Diabetic Retinopathy Classification Accuracy through Dual Attention Mechanism in Deep Learning
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决糖尿病视网膜病变（DR）分类中数据分布不均导致模型泛化能力差的问题。作者在深度学习模型中引入全局注意力块（GAB）和类别注意力块（CAB），结合多个预训练网络进行实验，提升了分类性能，并在两个公开数据集上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.19199v1](http://arxiv.org/pdf/2507.19199v1)**

> **作者:** Abdul Hannan; Zahid Mahmood; Rizwan Qureshi; Hazrat Ali
>
> **备注:** submitted to Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization
>
> **摘要:** Automatic classification of Diabetic Retinopathy (DR) can assist ophthalmologists in devising personalized treatment plans, making it a critical component of clinical practice. However, imbalanced data distribution in the dataset becomes a bottleneck in the generalization of deep learning models trained for DR classification. In this work, we combine global attention block (GAB) and category attention block (CAB) into the deep learning model, thus effectively overcoming the imbalanced data distribution problem in DR classification. Our proposed approach is based on an attention mechanism-based deep learning model that employs three pre-trained networks, namely, MobileNetV3-small, Efficientnet-b0, and DenseNet-169 as the backbone architecture. We evaluate the proposed method on two publicly available datasets of retinal fundoscopy images for DR. Experimental results show that on the APTOS dataset, the DenseNet-169 yielded 83.20% mean accuracy, followed by the MobileNetV3-small and EfficientNet-b0, which yielded 82% and 80% accuracies, respectively. On the EYEPACS dataset, the EfficientNet-b0 yielded a mean accuracy of 80%, while the DenseNet-169 and MobileNetV3-small yielded 75.43% and 76.68% accuracies, respectively. In addition, we also compute the F1-score of 82.0%, precision of 82.1%, sensitivity of 83.0%, specificity of 95.5%, and a kappa score of 88.2% for the experiments. Moreover, in our work, the MobileNetV3-small has 1.6 million parameters on the APTOS dataset and 0.90 million parameters on the EYEPACS dataset, which is comparatively less than other methods. The proposed approach achieves competitive performance that is at par with recently reported works on DR classification.
>
---
#### [new 115] Relaxed Total Generalized Variation Regularized Piecewise Smooth Mumford-Shah Model for Triangulated Surface Segmentation
- **分类: cs.CG; cs.CV**

- **简介: 该论文属于三维网格分割任务，旨在解决三角网格模型的分片平滑分割问题。现有方法多采用全变差正则化，追求最短边界。本文提出基于松弛广义全变差（rTGV）正则化的新型Mumford-Shah模型，能更好刻画几何结构高阶不连续性，实现更优边界分割。实验表明该方法在分割质量和效率方面均具竞争力。**

- **链接: [http://arxiv.org/pdf/2507.19284v1](http://arxiv.org/pdf/2507.19284v1)**

> **作者:** Huayan Zhang; Shanqiang Wang; Xiaochao Wang
>
> **摘要:** The Mumford-Shah (MS) model is an important technique for mesh segmentation. Many existing researches focus on piecewise constant MS mesh segmentation model with total variation regularization, which pursue the shortest length of boundaries. Different from previous efforts, in this article, we propose a novel piecewise smooth MS mesh segmentation model by utilizing the relaxed total generalized variation regularization (rTGV). The new model assumes that the feature function of a mesh can be approximated by the sum of piecewise constant function and asmooth function, and the rTGV regularization is able to characterize the high order discontinuity of the geometric structure. The newly introduced method is effective in segmenting meshes with irregular structures and getting the better boundaries rather than the shortest boundaries. We solve the new model by alternating minimization and alternating direction method of multipliers (ADMM). Our algorithm is discussed from several aspects, and comparisons with several state-of-art methods. Experimental results show that our method can yield competitive results when compared to other approaches. In addition, our results compare favorably to those of the several state-of-art techniques when evaluated on the Princeton Segmentation Benchmark. Furthermore, the quantitative errors and computational costs confirm the robustness and efficiency of the proposed method.
>
---
#### [new 116] Extreme Cardiac MRI Analysis under Respiratory Motion: Results of the CMRxMotion Challenge
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决呼吸运动伪影影响心脏MRI分析的问题。论文组织了CMRxMotion挑战赛，发布了含运动伪影的CMR数据集，并评估了22种算法在图像质量分类和心肌分割中的表现，探讨了运动伪影对临床生物标志物的影响。**

- **链接: [http://arxiv.org/pdf/2507.19165v1](http://arxiv.org/pdf/2507.19165v1)**

> **作者:** Kang Wang; Chen Qin; Zhang Shi; Haoran Wang; Xiwen Zhang; Chen Chen; Cheng Ouyang; Chengliang Dai; Yuanhan Mo; Chenchen Dai; Xutong Kuang; Ruizhe Li; Xin Chen; Xiuzheng Yue; Song Tian; Alejandro Mora-Rubio; Kumaradevan Punithakumar; Shizhan Gong; Qi Dou; Sina Amirrajab; Yasmina Al Khalil; Cian M. Scannell; Lexiaozi Fan; Huili Yang; Xiaowu Sun; Rob van der Geest; Tewodros Weldebirhan Arega; Fabrice Meriaudeau; Caner Özer; Amin Ranem; John Kalkhof; İlkay Öksüz; Anirban Mukhopadhyay; Abdul Qayyum; Moona Mazher; Steven A Niederer; Carles Garcia-Cabrera; Eric Arazo; Michal K. Grzeszczyk; Szymon Płotka; Wanqin Ma; Xiaomeng Li; Rongjun Ge; Yongqing Kou; Xinrong Chen; He Wang; Chengyan Wang; Wenjia Bai; Shuo Wang
>
> **摘要:** Deep learning models have achieved state-of-the-art performance in automated Cardiac Magnetic Resonance (CMR) analysis. However, the efficacy of these models is highly dependent on the availability of high-quality, artifact-free images. In clinical practice, CMR acquisitions are frequently degraded by respiratory motion, yet the robustness of deep learning models against such artifacts remains an underexplored problem. To promote research in this domain, we organized the MICCAI CMRxMotion challenge. We curated and publicly released a dataset of 320 CMR cine series from 40 healthy volunteers who performed specific breathing protocols to induce a controlled spectrum of motion artifacts. The challenge comprised two tasks: 1) automated image quality assessment to classify images based on motion severity, and 2) robust myocardial segmentation in the presence of motion artifacts. A total of 22 algorithms were submitted and evaluated on the two designated tasks. This paper presents a comprehensive overview of the challenge design and dataset, reports the evaluation results for the top-performing methods, and further investigates the impact of motion artifacts on five clinically relevant biomarkers. All resources and code are publicly available at: https://github.com/CMRxMotion
>
---
#### [new 117] Concept Probing: Where to Find Human-Defined Concepts (Extended Version)
- **分类: cs.LG; cs.AI; cs.CV; cs.NE**

- **简介: 该论文属于可解释性机器学习任务，旨在解决如何在神经网络中找到适合探测人类定义概念的层次。通过分析各层表示的信息性和规律性，自动识别关键层，并进行大量实验验证。**

- **链接: [http://arxiv.org/pdf/2507.18681v1](http://arxiv.org/pdf/2507.18681v1)**

> **作者:** Manuel de Sousa Ribeiro; Afonso Leote; João Leite
>
> **备注:** Extended version of the paper published in Proceedings of the International Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)
>
> **摘要:** Concept probing has recently gained popularity as a way for humans to peek into what is encoded within artificial neural networks. In concept probing, additional classifiers are trained to map the internal representations of a model into human-defined concepts of interest. However, the performance of these probes is highly dependent on the internal representations they probe from, making identifying the appropriate layer to probe an essential task. In this paper, we propose a method to automatically identify which layer's representations in a neural network model should be considered when probing for a given human-defined concept of interest, based on how informative and regular the representations are with respect to the concept. We validate our findings through an exhaustive empirical analysis over different neural network models and datasets.
>
---
#### [new 118] Generating real-time detailed ground visualisations from sparse aerial point clouds
- **分类: cs.GR; cs.CV; I.3.2; I.4.10**

- **简介: 该论文属于3D可视化与计算机图形学任务，旨在解决从稀疏航拍点云生成高质量实时地面可视化内容的问题。现有方法依赖大量人工，成本高且效率低。论文提出一种自动化流程，可放大真实世界扫描数据，并实现高质量实时3D渲染，适用于训练、模拟、游戏和可视化应用。**

- **链接: [http://arxiv.org/pdf/2507.18664v1](http://arxiv.org/pdf/2507.18664v1)**

> **作者:** Aidan Murray; Eddie Waite; Caleb Ross; Scarlet Mitchell; Alexander Bradley; Joanna Jamrozy; Kenny Mitchell
>
> **备注:** CVMP Short Paper. 1 page, 3 figures, CVMP 2022: The 19th ACM SIGGRAPH European Conference on Visual Media Production, London. This work was supported by the European Union's Horizon 2020 research and innovation programme under Grant 101017779
>
> **摘要:** Building realistic wide scale outdoor 3D content with sufficient visual quality to observe at walking eye level or from driven vehicles is often carried out by large teams of artists skilled in modelling, texturing, material shading and lighting, which typically leads to both prohibitive costs and reduced accuracy honoring the variety of real world ground truth landscapes. In our proposed method, we define a process to automatically amplify real-world scanned data and render real-time in animated 3D to explore at close range with high quality for training, simulation, video game and visualisation applications.
>
---
## 更新

#### [replaced 001] Motion Synthesis with Sparse and Flexible Keyjoint Control
- **分类: cs.GR; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.15557v2](http://arxiv.org/pdf/2503.15557v2)**

> **作者:** Inwoo Hwang; Jinseok Bae; Donggeun Lim; Young Min Kim
>
> **备注:** Accepted to ICCV 2025. Project Page: http://inwoohwang.me/SFControl
>
> **摘要:** Creating expressive character animations is labor-intensive, requiring intricate manual adjustment of animators across space and time. Previous works on controllable motion generation often rely on a predefined set of dense spatio-temporal specifications (e.g., dense pelvis trajectories with exact per-frame timing), limiting practicality for animators. To process high-level intent and intuitive control in diverse scenarios, we propose a practical controllable motions synthesis framework that respects sparse and flexible keyjoint signals. Our approach employs a decomposed diffusion-based motion synthesis framework that first synthesizes keyjoint movements from sparse input control signals and then synthesizes full-body motion based on the completed keyjoint trajectories. The low-dimensional keyjoint movements can easily adapt to various control signal types, such as end-effector position for diverse goal-driven motion synthesis, or incorporate functional constraints on a subset of keyjoints. Additionally, we introduce a time-agnostic control formulation, eliminating the need for frame-specific timing annotations and enhancing control flexibility. Then, the shared second stage can synthesize a natural whole-body motion that precisely satisfies the task requirement from dense keyjoint movements. We demonstrate the effectiveness of sparse and flexible keyjoint control through comprehensive experiments on diverse datasets and scenarios.
>
---
#### [replaced 002] EmbodiedOcc++: Boosting Embodied 3D Occupancy Prediction with Plane Regularization and Uncertainty Sampler
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09540v2](http://arxiv.org/pdf/2504.09540v2)**

> **作者:** Hao Wang; Xiaobao Wei; Xiaoan Zhang; Jianing Li; Chengyu Bai; Ying Li; Ming Lu; Wenzhao Zheng; Shanghang Zhang
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Online 3D occupancy prediction provides a comprehensive spatial understanding of embodied environments. While the innovative EmbodiedOcc framework utilizes 3D semantic Gaussians for progressive indoor occupancy prediction, it overlooks the geometric characteristics of indoor environments, which are primarily characterized by planar structures. This paper introduces EmbodiedOcc++, enhancing the original framework with two key innovations: a Geometry-guided Refinement Module (GRM) that constrains Gaussian updates through plane regularization, along with a Semantic-aware Uncertainty Sampler (SUS) that enables more effective updates in overlapping regions between consecutive frames. GRM regularizes the position update to align with surface normals. It determines the adaptive regularization weight using curvature-based and depth-based constraints, allowing semantic Gaussians to align accurately with planar surfaces while adapting in complex regions. To effectively improve geometric consistency from different views, SUS adaptively selects proper Gaussians to update. Comprehensive experiments on the EmbodiedOcc-ScanNet benchmark demonstrate that EmbodiedOcc++ achieves state-of-the-art performance across different settings. Our method demonstrates improved edge accuracy and retains more geometric details while ensuring computational efficiency, which is essential for online embodied perception. The code will be released at: https://github.com/PKUHaoWang/EmbodiedOcc2.
>
---
#### [replaced 003] Multimodal Recurrent Ensembles for Predicting Brain Responses to Naturalistic Movies (Algonauts 2025)
- **分类: q-bio.NC; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.17897v2](http://arxiv.org/pdf/2507.17897v2)**

> **作者:** Semih Eren; Deniz Kucukahmetler; Nico Scherf
>
> **备注:** 8 pages, 2 figures, 1 table. Invited report, CCN 2025 Algonauts Project session (3rd-place team). Code: https://github.com/erensemih/Algonauts2025_ModalityRNN
>
> **摘要:** Accurately predicting distributed cortical responses to naturalistic stimuli requires models that integrate visual, auditory and semantic information over time. We present a hierarchical multimodal recurrent ensemble that maps pretrained video, audio, and language embeddings to fMRI time series recorded while four subjects watched almost 80 hours of movies provided by the Algonauts 2025 challenge. Modality-specific bidirectional RNNs encode temporal dynamics; their hidden states are fused and passed to a second recurrent layer, and lightweight subject-specific heads output responses for 1000 cortical parcels. Training relies on a composite MSE-correlation loss and a curriculum that gradually shifts emphasis from early sensory to late association regions. Averaging 100 model variants further boosts robustness. The resulting system ranked third on the competition leaderboard, achieving an overall Pearson r = 0.2094 and the highest single-parcel peak score (mean r = 0.63) among all participants, with particularly strong gains for the most challenging subject (Subject 5). The approach establishes a simple, extensible baseline for future multimodal brain-encoding benchmarks.
>
---
#### [replaced 004] Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit and the Empirical Findings
- **分类: cs.CV; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.19628v2](http://arxiv.org/pdf/2411.19628v2)**

> **作者:** Qiong Wu; Wenhao Lin; Yiyi Zhou; Weihao Ye; Zhanpeng Zen; Xiaoshuai Sun; Rongrong Ji
>
> **摘要:** The excessive use of visual tokens in existing Multimoal Large Language Models (MLLMs) often exhibits obvious redundancy and brings in prohibitively expensive computation. To gain insights into this problem, we first conduct extensive empirical studies on the attention behaviors of MLLMs, and summarize three main inference stages in MLLMs: (i) Early fusion between tokens is first accomplished quickly. (ii) Intra-modality modeling then comes to play. (iii) Multimodal reasoning} resumes and lasts until the end of inference. In particular, we reveal that visual tokens will stop contributing to reasoning when the text tokens receive enough image information, yielding obvious visual redundancy. Based on these generalized observations, we propose a simple yet effective method to improve the efficiency of MLLMs, termed dynamic visual-token exit (DyVTE). DyVTE uses lightweight hyper-networks to perceive the text token status and decide the removal of all visual tokens after a certain layer, thereby addressing the observed visual redundancy. To validate VTE, we apply it to a set of MLLMs, including LLaVA, VILA, Eagle and InternVL, and conduct extensive experiments on a bunch of benchmarks. The experiment results not only show the effectiveness of our VTE in improving MLLMs' efficiency, but also yield the general modeling patterns of MLLMs, well facilitating the in-depth understanding of MLLMs. Our code is released at https://github.com/DoubtedSteam/DyVTE.
>
---
#### [replaced 005] Enhancing Frequency for Single Image Super-Resolution with Learnable Separable Kernels
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.04555v2](http://arxiv.org/pdf/2506.04555v2)**

> **作者:** Heng Tian
>
> **摘要:** Existing approaches often enhance the performance of single-image super-resolution (SISR) methods by incorporating auxiliary structures, such as specialized loss functions, to indirectly boost the quality of low-resolution images. In this paper, we propose a plug-and-play module called Learnable Separable Kernels (LSKs), which are formally rank-one matrices designed to directly enhance image frequency components. We begin by explaining why LSKs are particularly suitable for SISR tasks from a frequency perspective. Baseline methods incorporating LSKs demonstrate a significant reduction of over 60\% in both the number of parameters and computational requirements. This reduction is achieved through the decomposition of LSKs into orthogonal and mergeable one-dimensional kernels. Additionally, we perform an interpretable analysis of the feature maps generated by LSKs. Visualization results reveal the capability of LSKs to enhance image frequency components effectively. Extensive experiments show that incorporating LSKs not only reduces the number of parameters and computational load but also improves overall model performance. Moreover, these experiments demonstrate that models utilizing LSKs exhibit superior performance, particularly as the upscaling factor increases.
>
---
#### [replaced 006] Tell Me What to Track: Infusing Robust Language Guidance for Enhanced Referring Multi-Object Tracking
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.12561v3](http://arxiv.org/pdf/2412.12561v3)**

> **作者:** Wenjun Huang; Yang Ni; Hanning Chen; Yirui He; Ian Bryant; Yezi Liu; Mohsen Imani
>
> **摘要:** Referring multi-object tracking (RMOT) is an emerging cross-modal task that aims to localize an arbitrary number of targets based on a language expression and continuously track them in a video. This intricate task involves reasoning on multi-modal data and precise target localization with temporal association. However, prior studies overlook the imbalanced data distribution between newborn targets and existing targets due to the nature of the task. In addition, they only indirectly fuse multi-modal features, struggling to deliver clear guidance on newborn target detection. To solve the above issues, we conduct a collaborative matching strategy to alleviate the impact of the imbalance, boosting the ability to detect newborn targets while maintaining tracking performance. In the encoder, we integrate and enhance the cross-modal and multi-scale fusion, overcoming the bottlenecks in previous work, where limited multi-modal information is shared and interacted between feature maps. In the decoder, we also develop a referring-infused adaptation that provides explicit referring guidance through the query tokens. The experiments showcase the superior performance of our model (+3.42%) compared to prior works, demonstrating the effectiveness of our designs.
>
---
#### [replaced 007] BGM: Background Mixup for X-ray Prohibited Items Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.00460v3](http://arxiv.org/pdf/2412.00460v3)**

> **作者:** Weizhe Liu; Renshuai Tao; Hongguang Zhu; Yunda Sun; Yao Zhao; Yunchao Wei
>
> **摘要:** Current data-driven approaches for X-ray prohibited items detection remain under-explored, particularly in the design of effective data augmentations. Existing natural image augmentations for reflected light imaging neglect the data characteristics of X-ray security images. Moreover, prior X-ray augmentation methods have predominantly focused on foreground prohibited items, overlooking informative background cues. In this paper, we propose Background Mixup (BGM), a background-based augmentation technique tailored for X-ray security imaging domain. Unlike conventional methods, BGM is founded on an in-depth analysis of physical properties including: 1) X-ray Transmission Imagery: Transmitted X-ray pixels represent composite information from multiple materials along the imaging path. 2) Material-based Pseudo-coloring: Pseudo-coloring in X-ray images correlates directly with material properties, aiding in material distinction. Building upon the above insights, BGM mixes background patches across regions on both 1) texture structure and 2) material variation, to benefit models from complicated background cues. This enhances the model's capability to handle domain-specific challenges such as occlusion-induced discriminative imbalance. Importantly, BGM is orthogonal and fully compatible with existing foreground-focused augmentation techniques, enabling joint use to further enhance detection performance. Extensive experiments on multiple X-ray security benchmarks show that BGM consistently surpasses strong baselines, without additional annotations or significant training overhead. This work pioneers the exploration of background-aware augmentation in X-ray prohibited items detection and provides a lightweight, plug-and-play solution with broad applicability.
>
---
#### [replaced 008] Preserve Anything: Controllable Image Synthesis with Object Preservation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22531v2](http://arxiv.org/pdf/2506.22531v2)**

> **作者:** Prasen Kumar Sharma; Neeraj Matiyali; Siddharth Srivastava; Gaurav Sharma
>
> **备注:** Accepted at ICCV 2025 (main conference)
>
> **摘要:** We introduce \textit{Preserve Anything}, a novel method for controlled image synthesis that addresses key limitations in object preservation and semantic consistency in text-to-image (T2I) generation. Existing approaches often fail (i) to preserve multiple objects with fidelity, (ii) maintain semantic alignment with prompts, or (iii) provide explicit control over scene composition. To overcome these challenges, the proposed method employs an N-channel ControlNet that integrates (i) object preservation with size and placement agnosticism, color and detail retention, and artifact elimination, (ii) high-resolution, semantically consistent backgrounds with accurate shadows, lighting, and prompt adherence, and (iii) explicit user control over background layouts and lighting conditions. Key components of our framework include object preservation and background guidance modules, enforcing lighting consistency and a high-frequency overlay module to retain fine details while mitigating unwanted artifacts. We introduce a benchmark dataset consisting of 240K natural images filtered for aesthetic quality and 18K 3D-rendered synthetic images with metadata such as lighting, camera angles, and object relationships. This dataset addresses the deficiencies of existing benchmarks and allows a complete evaluation. Empirical results demonstrate that our method achieves state-of-the-art performance, significantly improving feature-space fidelity (FID 15.26) and semantic alignment (CLIP-S 32.85) while maintaining competitive aesthetic quality. We also conducted a user study to demonstrate the efficacy of the proposed work on unseen benchmark and observed a remarkable improvement of $\sim25\%$, $\sim19\%$, $\sim13\%$, and $\sim14\%$ in terms of prompt alignment, photorealism, the presence of AI artifacts, and natural aesthetics over existing works.
>
---
#### [replaced 009] Multispectral Demosaicing via Dual Cameras
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.22026v3](http://arxiv.org/pdf/2503.22026v3)**

> **作者:** SaiKiran Tedla; Junyong Lee; Beixuan Yang; Mahmoud Afifi; Michael S. Brown
>
> **备注:** https://ms-demosaic.github.io/
>
> **摘要:** Multispectral (MS) images capture detailed scene information across a wide range of spectral bands, making them invaluable for applications requiring rich spectral data. Integrating MS imaging into multi camera devices, such as smartphones, has the potential to enhance both spectral applications and RGB image quality. A critical step in processing MS data is demosaicing, which reconstructs color information from the mosaic MS images captured by the camera. This paper proposes a method for MS image demosaicing specifically designed for dual-camera setups where both RGB and MS cameras capture the same scene. Our approach leverages co-captured RGB images, which typically have higher spatial fidelity, to guide the demosaicing of lower-fidelity MS images. We introduce the Dual-camera RGB-MS Dataset - a large collection of paired RGB and MS mosaiced images with ground-truth demosaiced outputs - that enables training and evaluation of our method. Experimental results demonstrate that our method achieves state-of-the-art accuracy compared to existing techniques.
>
---
#### [replaced 010] Long-Form Answers to Visual Questions from Blind and Low Vision People
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.06303v2](http://arxiv.org/pdf/2408.06303v2)**

> **作者:** Mina Huh; Fangyuan Xu; Yi-Hao Peng; Chongyan Chen; Hansika Murugu; Danna Gurari; Eunsol Choi; Amy Pavel
>
> **备注:** COLM 2024 Oral Spotlight
>
> **摘要:** Vision language models can now generate long-form answers to questions about images - long-form visual question answers (LFVQA). We contribute VizWiz-LF, a dataset of long-form answers to visual questions posed by blind and low vision (BLV) users. VizWiz-LF contains 4.2k long-form answers to 600 visual questions, collected from human expert describers and six VQA models. We develop and annotate functional roles of sentences of LFVQA and demonstrate that long-form answers contain information beyond the question answer such as explanations and suggestions. We further conduct automatic and human evaluations with BLV and sighted people to evaluate long-form answers. BLV people perceive both human-written and generated long-form answers to be plausible, but generated answers often hallucinate incorrect visual details, especially for unanswerable visual questions (e.g., blurry or irrelevant images). To reduce hallucinations, we evaluate the ability of VQA models to abstain from answering unanswerable questions across multiple prompting strategies.
>
---
#### [replaced 011] MagicDrive3D: Controllable 3D Generation for Any-View Rendering in Street Scenes
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.14475v4](http://arxiv.org/pdf/2405.14475v4)**

> **作者:** Ruiyuan Gao; Kai Chen; Zhihao Li; Lanqing Hong; Zhenguo Li; Qiang Xu
>
> **备注:** Project Page: https://flymin.github.io/magicdrive3d
>
> **摘要:** Controllable generative models for images and videos have seen significant success, yet 3D scene generation, especially in unbounded scenarios like autonomous driving, remains underdeveloped. Existing methods lack flexible controllability and often rely on dense view data collection in controlled environments, limiting their generalizability across common datasets (e.g., nuScenes). In this paper, we introduce MagicDrive3D, a novel framework for controllable 3D street scene generation that combines video-based view synthesis with 3D representation (3DGS) generation. It supports multi-condition control, including road maps, 3D objects, and text descriptions. Unlike previous approaches that require 3D representation before training, MagicDrive3D first trains a multi-view video generation model to synthesize diverse street views. This method utilizes routinely collected autonomous driving data, reducing data acquisition challenges and enriching 3D scene generation. In the 3DGS generation step, we introduce Fault-Tolerant Gaussian Splatting to address minor errors and use monocular depth for better initialization, alongside appearance modeling to manage exposure discrepancies across viewpoints. Experiments show that MagicDrive3D generates diverse, high-quality 3D driving scenes, supports any-view rendering, and enhances downstream tasks like BEV segmentation, demonstrating its potential for autonomous driving simulation and beyond.
>
---
#### [replaced 012] Framework of a multiscale data-driven DT of the musculoskeletal system
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11821v2](http://arxiv.org/pdf/2506.11821v2)**

> **作者:** Martina Paccini; Simone Cammarasana; Giuseppe Patanè
>
> **摘要:** Musculoskeletal disorders (MSDs) are a leading cause of disability worldwide, requiring advanced diagnostic and therapeutic tools for personalised assessment and treatment. Effective management of MSDs involves the interaction of heterogeneous data sources, making the Digital Twin (DT) paradigm a valuable option. This paper introduces the Musculoskeletal Digital Twin (MS-DT), a novel framework that integrates multiscale biomechanical data with computational modelling to create a detailed, patient-specific representation of the musculoskeletal system. By combining motion capture, ultrasound imaging, electromyography, and medical imaging, the MS-DT enables the analysis of spinal kinematics, posture, and muscle function. An interactive visualisation platform provides clinicians and researchers with an intuitive interface for exploring biomechanical parameters and tracking patient-specific changes. Results demonstrate the effectiveness of MS-DT in extracting precise kinematic and dynamic tissue features, offering a comprehensive tool for monitoring spine biomechanics and rehabilitation. This framework provides high-fidelity modelling and real-time visualization to improve patient-specific diagnosis and intervention planning.
>
---
#### [replaced 013] RoCo-Sim: Enhancing Roadside Collaborative Perception through Foreground Simulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10410v2](http://arxiv.org/pdf/2503.10410v2)**

> **作者:** Yuwen Du; Anning Hu; Zichen Chao; Yifan Lu; Junhao Ge; Genjia Liu; Weitao Wu; Lanjun Wang; Siheng Chen
>
> **摘要:** Roadside Collaborative Perception refers to a system where multiple roadside units collaborate to pool their perceptual data, assisting vehicles in enhancing their environmental awareness. Existing roadside perception methods concentrate on model design but overlook data issues like calibration errors, sparse information, and multi-view consistency, leading to poor performance on recent published datasets. To significantly enhance roadside collaborative perception and address critical data issues, we present the first simulation framework RoCo-Sim for road-side collaborative perception. RoCo-Sim is capable of generating diverse, multi-view consistent simulated roadside data through dynamic foreground editing and full-scene style transfer of a single image. RoCo-Sim consists of four components: (1) Camera Extrinsic Optimization ensures accurate 3D to 2D projection for roadside cameras; (2) A novel Multi-View Occlusion-Aware Sampler (MOAS) determines the placement of diverse digital assets within 3D space; (3) DepthSAM innovatively models foreground-background relationships from single-frame fixed-view images, ensuring multi-view consistency of foreground; and (4) Scalable Post-Processing Toolkit generates more realistic and enriched scenes through style transfer and other enhancements. RoCo-Sim significantly improves roadside 3D object detection, outperforming SOTA methods by 83.74 on Rcooper-Intersection and 83.12 on TUMTraf-V2X for AP70. RoCo-Sim fills a critical gap in roadside perception simulation. Code and pre-trained models will be released soon: https://github.com/duyuwen-duen/RoCo-Sim
>
---
#### [replaced 014] A Study of Anatomical Priors for Deep Learning-Based Segmentation of Pheochromocytoma in Abdominal CT
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15193v2](http://arxiv.org/pdf/2507.15193v2)**

> **作者:** Tanjin Taher Toma; Tejas Sudharshan Mathai; Bikash Santra; Pritam Mukherjee; Jianfei Liu; Wesley Jong; Darwish Alabyad; Vivek Batheja; Abhishek Jha; Mayank Patel; Darko Pucar; Jayadira del Rivero; Karel Pacak; Ronald M. Summers
>
> **摘要:** Accurate segmentation of pheochromocytoma (PCC) in abdominal CT scans is essential for tumor burden estimation, prognosis, and treatment planning. It may also help infer genetic clusters, reducing reliance on expensive testing. This study systematically evaluates anatomical priors to identify configurations that improve deep learning-based PCC segmentation. We employed the nnU-Net framework to evaluate eleven annotation strategies for accurate 3D segmentation of pheochromocytoma, introducing a set of novel multi-class schemes based on organ-specific anatomical priors. These priors were derived from adjacent organs commonly surrounding adrenal tumors (e.g., liver, spleen, kidney, aorta, adrenal gland, and pancreas), and were compared against a broad body-region prior used in previous work. The framework was trained and tested on 105 contrast-enhanced CT scans from 91 patients at the NIH Clinical Center. Performance was measured using Dice Similarity Coefficient (DSC), Normalized Surface Distance (NSD), and instance-wise F1 score. Among all strategies, the Tumor + Kidney + Aorta (TKA) annotation achieved the highest segmentation accuracy, significantly outperforming the previously used Tumor + Body (TB) annotation across DSC (p = 0.0097), NSD (p = 0.0110), and F1 score (25.84% improvement at an IoU threshold of 0.5), measured on a 70-30 train-test split. The TKA model also showed superior tumor burden quantification (R^2 = 0.968) and strong segmentation across all genetic subtypes. In five-fold cross-validation, TKA consistently outperformed TB across IoU thresholds (0.1 to 0.5), reinforcing its robustness and generalizability. These findings highlight the value of incorporating relevant anatomical context into deep learning models to achieve precise PCC segmentation, offering a valuable tool to support clinical assessment and longitudinal disease monitoring in PCC patients.
>
---
#### [replaced 015] AI Flow: Perspectives, Scenarios, and Approaches
- **分类: cs.AI; cs.CL; cs.CV; cs.DC; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.12479v3](http://arxiv.org/pdf/2506.12479v3)**

> **作者:** Hongjun An; Wenhan Hu; Sida Huang; Siqi Huang; Ruanjun Li; Yuanzhi Liang; Jiawei Shao; Yiliang Song; Zihan Wang; Cheng Yuan; Chi Zhang; Hongyuan Zhang; Wenhao Zhuang; Xuelong Li
>
> **备注:** Authors are with Institute of Artificial Intelligence (TeleAI), China Telecom, China. Author names are listed alphabetically by surname. This work was conducted at TeleAI, facilitated by Dr. Jiawei Shao (e-mail: shaojw2@chinatelecom.cn) under the leadership of Prof. Xuelong Li. The corresponding author is Prof. Xuelong Li (e-mail: xuelong li@ieee.org), the CTO and Chief Scientist of China Telecom
>
> **摘要:** Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems.
>
---
#### [replaced 016] Vid2Coach: Transforming How-To Videos into Task Assistants
- **分类: cs.HC; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00717v2](http://arxiv.org/pdf/2506.00717v2)**

> **作者:** Mina Huh; Zihui Xue; Ujjaini Das; Kumar Ashutosh; Kristen Grauman; Amy Pavel
>
> **备注:** Accepted to UIST 2025 Project website: https://minahuh.com/Vid2Coach/
>
> **摘要:** People use videos to learn new recipes, exercises, and crafts. Such videos remain difficult for blind and low vision (BLV) people to follow as they rely on visual comparison. Our observations of visual rehabilitation therapists (VRTs) guiding BLV people to follow how-to videos revealed that VRTs provide both proactive and responsive support including detailed descriptions, non-visual workarounds, and progress feedback. We propose Vid2Coach, a system that transforms how-to videos into wearable camera-based assistants that provide accessible instructions and mixed-initiative feedback. From the video, Vid2Coach generates accessible instructions by augmenting narrated instructions with demonstration details and completion criteria for each step. It then uses retrieval-augmented-generation to extract relevant non-visual workarounds from BLV-specific resources. Vid2Coach then monitors user progress with a camera embedded in commercial smart glasses to provide context-aware instructions, proactive feedback, and answers to user questions. BLV participants (N=8) using Vid2Coach completed cooking tasks with 58.5\% fewer errors than when using their typical workflow and wanted to use Vid2Coach in their daily lives. Vid2Coach demonstrates an opportunity for AI visual assistance that strengthens rather than replaces non-visual expertise.
>
---
#### [replaced 017] Stella Nera: A Differentiable Maddness-Based Hardware Accelerator for Efficient Approximate Matrix Multiplication
- **分类: cs.AR; cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2311.10207v2](http://arxiv.org/pdf/2311.10207v2)**

> **作者:** Jannis Schönleber; Lukas Cavigelli; Matteo Perotti; Luca Benini; Renzo Andri
>
> **备注:** Accepted as full paper at IEEE Computer Society Annual Symposium on VLSI (ISVLSI) 2025
>
> **摘要:** Artificial intelligence has surged in recent years, with advancements in machine learning rapidly impacting nearly every area of life. However, the growing complexity of these models has far outpaced advancements in available hardware accelerators, leading to significant computational and energy demands, primarily due to matrix multiplications, which dominate the compute workload. Maddness (i.e., Multiply-ADDitioN-lESS) presents a hash-based version of product quantization, which renders matrix multiplications into lookups and additions, eliminating the need for multipliers entirely. We present Stella Nera, the first Maddness-based accelerator achieving an energy efficiency of 161 TOp/s/W@0.55V, 25x better than conventional MatMul accelerators due to its small components and reduced computational complexity. We further enhance Maddness with a differentiable approximation, allowing for gradient-based fine-tuning and achieving an end-to-end performance of 92.5% Top-1 accuracy on CIFAR-10.
>
---
#### [replaced 018] Style-Adaptive Detection Transformer for Single-Source Domain Generalized Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20498v2](http://arxiv.org/pdf/2504.20498v2)**

> **作者:** Jianhong Han; Yupei Wang; Liang Chen
>
> **备注:** Manuscript submitted to IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** Single-source domain generalization (SDG) in object detection aims to develop a detector using only source domain data that generalizes well to unseen target domains. Existing methods are primarily CNN-based and improve robustness through data augmentation combined with feature alignment. However, these methods are limited, as augmentation is only effective when the synthetic distribution approximates that of unseen domains, thus failing to ensure generalization across diverse scenarios. While DEtection TRansformer (DETR) has shown strong generalization in domain adaptation due to global context modeling, its potential for SDG remains underexplored. To this end, we propose Style-Adaptive DEtection TRansformer (SA-DETR), a DETR-based detector tailored for SDG. SA-DETR introduces an online domain style adapter that projects the style representation of unseen domains into the source domain via a dynamic memory bank. This bank self-organizes into diverse style prototypes and is continuously updated under a test-time adaptation framework, enabling effective style rectification. Additionally, we design an object-aware contrastive learning module to promote extraction of domain-invariant features. By applying gating masks that constrain contrastive learning in both spatial and semantic dimensions, this module facilitates instance-level cross-domain contrast and enhances generalization. Extensive experiments across five distinct weather scenarios demonstrate that SA-DETR consistently outperforms existing methods in both detection accuracy and domain generalization capability.
>
---
#### [replaced 019] Information Extraction from Unstructured data using Augmented-AI and Computer Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.09880v2](http://arxiv.org/pdf/2312.09880v2)**

> **作者:** Aditya Parikh
>
> **摘要:** Information extraction (IE) from unstructured documents remains a critical challenge in data processing pipelines. Traditional optical character recognition (OCR) methods and conventional parsing engines demonstrate limited effectiveness when processing large-scale document datasets. This paper presents a comprehensive framework for information extraction that combines Augmented Intelligence (A2I) with computer vision and natural language processing techniques. Our approach addresses the limitations of conventional methods by leveraging deep learning architectures for object detection, particularly for tabular data extraction, and integrating cloud-based services for scalable document processing. The proposed methodology demonstrates improved accuracy and efficiency in extracting structured information from diverse document formats including PDFs, images, and scanned documents. Experimental validation shows significant improvements over traditional OCR-based approaches, particularly in handling complex document layouts and multi-modal content extraction.
>
---
#### [replaced 020] MLRU++: Multiscale Lightweight Residual UNETR++ with Attention for Efficient 3D Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16122v3](http://arxiv.org/pdf/2507.16122v3)**

> **作者:** Nand Kumar Yadav; Rodrigue Rizk; William CW Chen; KC Santosh
>
> **摘要:** Accurate and efficient medical image segmentation is crucial but challenging due to anatomical variability and high computational demands on volumetric data. Recent hybrid CNN-Transformer architectures achieve state-of-the-art results but add significant complexity. In this paper, we propose MLRU++, a Multiscale Lightweight Residual UNETR++ architecture designed to balance segmentation accuracy and computational efficiency. It introduces two key innovations: a Lightweight Channel and Bottleneck Attention Module (LCBAM) that enhances contextual feature encoding with minimal overhead, and a Multiscale Bottleneck Block (M2B) in the decoder that captures fine-grained details via multi-resolution feature aggregation. Experiments on four publicly available benchmark datasets (Synapse, BTCV, ACDC, and Decathlon Lung) demonstrate that MLRU++ achieves state-of-the-art performance, with average Dice scores of 87.57% (Synapse), 93.00% (ACDC), and 81.12% (Lung). Compared to existing leading models, MLRU++ improves Dice scores by 5.38% and 2.12% on Synapse and ACDC, respectively, while significantly reducing parameter count and computational cost. Ablation studies evaluating LCBAM and M2B further confirm the effectiveness of the proposed architectural components. Results suggest that MLRU++ offers a practical and high-performing solution for 3D medical image segmentation tasks. Source code is available at: https://github.com/1027865/MLRUPP
>
---
#### [replaced 021] High Performance Space Debris Tracking in Complex Skylight Backgrounds with a Large-Scale Dataset
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02614v4](http://arxiv.org/pdf/2506.02614v4)**

> **作者:** Guohang Zhuang; Weixi Song; Jinyang Huang; Chenwei Yang; Wanli OuYang; Yan Lu
>
> **摘要:** With the rapid development of space exploration, space debris has attracted more attention due to its potential extreme threat, leading to the need for real-time and accurate debris tracking. However, existing methods are mainly based on traditional signal processing, which cannot effectively process the complex background and dense space debris. In this paper, we propose a deep learning-based Space Debris Tracking Network~(SDT-Net) to achieve highly accurate debris tracking. SDT-Net effectively represents the feature of debris, enhancing the efficiency and stability of end-to-end model learning. To train and evaluate this model effectively, we also produce a large-scale dataset Space Debris Tracking Dataset (SDTD) by a novel observation-based data simulation scheme. SDTD contains 18,040 video sequences with a total of 62,562 frames and covers 250,000 synthetic space debris. Extensive experiments validate the effectiveness of our model and the challenging of our dataset. Furthermore, we test our model on real data from the Antarctic Station, achieving a MOTA score of 73.2%, which demonstrates its strong transferability to real-world scenarios. Our dataset and code will be released soon.
>
---
#### [replaced 022] Do Existing Testing Tools Really Uncover Gender Bias in Text-to-Image Models?
- **分类: cs.CV; cs.SE**

- **链接: [http://arxiv.org/pdf/2501.15775v2](http://arxiv.org/pdf/2501.15775v2)**

> **作者:** Yunbo Lyu; Zhou Yang; Yuqing Niu; Jing Jiang; David Lo
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Text-to-Image (T2I) models have recently gained significant attention due to their ability to generate high-quality images and are consequently used in a wide range of applications. However, there are concerns about the gender bias of these models. Previous studies have shown that T2I models can perpetuate or even amplify gender stereotypes when provided with neutral text prompts. Researchers have proposed automated gender bias uncovering detectors for T2I models, but a crucial gap exists: no existing work comprehensively compares the various detectors and understands how the gender bias detected by them deviates from the actual situation. This study addresses this gap by validating previous gender bias detectors using a manually labeled dataset and comparing how the bias identified by various detectors deviates from the actual bias in T2I models, as verified by manual confirmation. We create a dataset consisting of 6,000 images generated from three cutting-edge T2I models: Stable Diffusion XL, Stable Diffusion 3, and Dreamlike Photoreal 2.0. During the human-labeling process, we find that all three T2I models generate a portion (12.48% on average) of low-quality images (e.g., generate images with no face present), where human annotators cannot determine the gender of the person. Our analysis reveals that all three T2I models show a preference for generating male images, with SDXL being the most biased. Additionally, images generated using prompts containing professional descriptions (e.g., lawyer or doctor) show the most bias. We evaluate seven gender bias detectors and find that none fully capture the actual level of bias in T2I models, with some detectors overestimating bias by up to 26.95%. We further investigate the causes of inaccurate estimations, highlighting the limitations of detectors in dealing with low-quality images. Based on our findings, we propose an enhanced detector...
>
---
#### [replaced 023] Feed-Forward SceneDINO for Unsupervised Semantic Scene Completion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06230v2](http://arxiv.org/pdf/2507.06230v2)**

> **作者:** Aleksandar Jevtić; Christoph Reich; Felix Wimbauer; Oliver Hahn; Christian Rupprecht; Stefan Roth; Daniel Cremers
>
> **备注:** ICCV 2025. Christoph Reich and Aleksandar Jevti\'c - both authors contributed equally. Code: https://github.com/tum-vision/scenedino Project page: https://visinf.github.io/scenedino
>
> **摘要:** Semantic scene completion (SSC) aims to infer both the 3D geometry and semantics of a scene from single images. In contrast to prior work on SSC that heavily relies on expensive ground-truth annotations, we approach SSC in an unsupervised setting. Our novel method, SceneDINO, adapts techniques from self-supervised representation learning and 2D unsupervised scene understanding to SSC. Our training exclusively utilizes multi-view consistency self-supervision without any form of semantic or geometric ground truth. Given a single input image, SceneDINO infers the 3D geometry and expressive 3D DINO features in a feed-forward manner. Through a novel 3D feature distillation approach, we obtain unsupervised 3D semantics. In both 3D and 2D unsupervised scene understanding, SceneDINO reaches state-of-the-art segmentation accuracy. Linear probing our 3D features matches the segmentation accuracy of a current supervised SSC approach. Additionally, we showcase the domain generalization and multi-view consistency of SceneDINO, taking the first steps towards a strong foundation for single image 3D scene understanding.
>
---
#### [replaced 024] Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04790v3](http://arxiv.org/pdf/2507.04790v3)**

> **作者:** Giwon Lee; Wooseong Jeong; Daehee Park; Jaewoo Jeong; Kuk-Jin Yoon
>
> **备注:** Accepted at ICCV 2025 (Highlight)
>
> **摘要:** Motion planning is a crucial component of autonomous robot driving. While various trajectory datasets exist, effectively utilizing them for a target domain remains challenging due to differences in agent interactions and environmental characteristics. Conventional approaches, such as domain adaptation or ensemble learning, leverage multiple source datasets but suffer from domain imbalance, catastrophic forgetting, and high computational costs. To address these challenges, we propose Interaction-Merged Motion Planning (IMMP), a novel approach that leverages parameter checkpoints trained on different domains during adaptation to the target domain. IMMP follows a two-step process: pre-merging to capture agent behaviors and interactions, sufficiently extracting diverse information from the source domain, followed by merging to construct an adaptable model that efficiently transfers diverse interactions to the target domain. Our method is evaluated on various planning benchmarks and models, demonstrating superior performance compared to conventional approaches.
>
---
#### [replaced 025] TARS: Traffic-Aware Radar Scene Flow Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10210v2](http://arxiv.org/pdf/2503.10210v2)**

> **作者:** Jialong Wu; Marco Braun; Dominic Spata; Matthias Rottmann
>
> **摘要:** Scene flow provides crucial motion information for autonomous driving. Recent LiDAR scene flow models utilize the rigid-motion assumption at the instance level, assuming objects are rigid bodies. However, these instance-level methods are not suitable for sparse radar point clouds. In this work, we present a novel Traffic-Aware Radar Scene-Flow (TARS) estimation method, which utilizes motion rigidity at the traffic level. To address the challenges in radar scene flow, we perform object detection and scene flow jointly and boost the latter. We incorporate the feature map from the object detector, trained with detection losses, to make radar scene flow aware of the environment and road users. From this, we construct a Traffic Vector Field (TVF) in the feature space to achieve holistic traffic-level scene understanding in our scene flow branch. When estimating the scene flow, we consider both point-level motion cues from point neighbors and traffic-level consistency of rigid motion within the space. TARS outperforms the state of the art on a proprietary dataset and the View-of-Delft dataset, improving the benchmarks by 23% and 15%, respectively.
>
---
#### [replaced 026] GVCCS: A Dataset for Contrail Identification and Tracking on Visible Whole Sky Camera Sequences
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.18330v2](http://arxiv.org/pdf/2507.18330v2)**

> **作者:** Gabriel Jarry; Ramon Dalmau; Philippe Very; Franck Ballerini; Stefania-Denisa Bocu
>
> **摘要:** Aviation's climate impact includes not only CO2 emissions but also significant non-CO2 effects, especially from contrails. These ice clouds can alter Earth's radiative balance, potentially rivaling the warming effect of aviation CO2. Physics-based models provide useful estimates of contrail formation and climate impact, but their accuracy depends heavily on the quality of atmospheric input data and on assumptions used to represent complex processes like ice particle formation and humidity-driven persistence. Observational data from remote sensors, such as satellites and ground cameras, could be used to validate and calibrate these models. However, existing datasets don't explore all aspect of contrail dynamics and formation: they typically lack temporal tracking, and do not attribute contrails to their source flights. To address these limitations, we present the Ground Visible Camera Contrail Sequences (GVCCS), a new open data set of contrails recorded with a ground-based all-sky camera in the visible range. Each contrail is individually labeled and tracked over time, allowing a detailed analysis of its lifecycle. The dataset contains 122 video sequences (24,228 frames) and includes flight identifiers for contrails that form above the camera. As reference, we also propose a unified deep learning framework for contrail analysis using a panoptic segmentation model that performs semantic segmentation (contrail pixel identification), instance segmentation (individual contrail separation), and temporal tracking in a single architecture. By providing high-quality, temporally resolved annotations and a benchmark for model evaluation, our work supports improved contrail monitoring and will facilitate better calibration of physical models. This sets the groundwork for more accurate climate impact understanding and assessments.
>
---
#### [replaced 027] Towards Generalized Range-View LiDAR Segmentation in Adverse Weather
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08979v3](http://arxiv.org/pdf/2506.08979v3)**

> **作者:** Longyu Yang; Lu Zhang; Jun Liu; Yap-Peng Tan; Heng Tao Shen; Xiaofeng Zhu; Ping Hu
>
> **摘要:** LiDAR segmentation has emerged as an important task to enrich scene perception and understanding. Range-view-based methods have gained popularity due to their high computational efficiency and compatibility with real-time deployment. However, their generalized performance under adverse weather conditions remains underexplored, limiting their reliability in real-world environments. In this work, we identify and analyze the unique challenges that affect the generalization of range-view LiDAR segmentation in severe weather. To address these challenges, we propose a modular and lightweight framework that enhances robustness without altering the core architecture of existing models. Our method reformulates the initial stem block of standard range-view networks into two branches to process geometric attributes and reflectance intensity separately. Specifically, a Geometric Abnormality Suppression (GAS) module reduces the influence of weather-induced spatial noise, and a Reflectance Distortion Calibration (RDC) module corrects reflectance distortions through memory-guided adaptive instance normalization. The processed features are then fused and passed to the original segmentation pipeline. Extensive experiments on different benchmarks and baseline models demonstrate that our approach significantly improves generalization to adverse weather with minimal inference overhead, offering a practical and effective solution for real-world LiDAR segmentation.
>
---
#### [replaced 028] ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.18262v2](http://arxiv.org/pdf/2507.18262v2)**

> **作者:** Chenyu Su; Weiwei Shang; Chen Qian; Fei Zhang; Shuang Cong
>
> **备注:** 12 pages,9 figures
>
> **摘要:** Semantics-driven 3D spatial constraints align highlevel semantic representations with low-level action spaces, facilitating the unification of task understanding and execution in robotic manipulation. The synergistic reasoning of Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs) enables cross-modal 3D spatial constraint construction. Nevertheless, existing methods have three key limitations: (1) coarse semantic granularity in constraint modeling, (2) lack of real-time closed-loop planning, (3) compromised robustness in semantically diverse environments. To address these challenges, we propose ReSem3D, a unified manipulation framework for semantically diverse environments, leveraging the synergy between VFMs and MLLMs to achieve fine-grained visual grounding and dynamically constructs hierarchical 3D spatial constraints for real-time manipulation. Specifically, the framework is driven by hierarchical recursive reasoning in MLLMs, which interact with VFMs to automatically construct 3D spatial constraints from natural language instructions and RGB-D observations in two stages: part-level extraction and region-level refinement. Subsequently, these constraints are encoded as real-time optimization objectives in joint space, enabling reactive behavior to dynamic disturbances. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSem3D performs diverse manipulation tasks under zero-shot conditions, exhibiting strong adaptability and generalization. Code and videos are available at https://github.com/scy-v/ReSem3D and https://resem3d.github.io.
>
---
#### [replaced 029] All in One: Visual-Description-Guided Unified Point Cloud Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05211v2](http://arxiv.org/pdf/2507.05211v2)**

> **作者:** Zongyan Han; Mohamed El Amine Boudjoghra; Jiahua Dong; Jinhong Wang; Rao Muhammad Anwer
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Unified segmentation of 3D point clouds is crucial for scene understanding, but is hindered by its sparse structure, limited annotations, and the challenge of distinguishing fine-grained object classes in complex environments. Existing methods often struggle to capture rich semantic and contextual information due to limited supervision and a lack of diverse multimodal cues, leading to suboptimal differentiation of classes and instances. To address these challenges, we propose VDG-Uni3DSeg, a novel framework that integrates pre-trained vision-language models (e.g., CLIP) and large language models (LLMs) to enhance 3D segmentation. By leveraging LLM-generated textual descriptions and reference images from the internet, our method incorporates rich multimodal cues, facilitating fine-grained class and instance separation. We further design a Semantic-Visual Contrastive Loss to align point features with multimodal queries and a Spatial Enhanced Module to model scene-wide relationships efficiently. Operating within a closed-set paradigm that utilizes multimodal knowledge generated offline, VDG-Uni3DSeg achieves state-of-the-art results in semantic, instance, and panoptic segmentation, offering a scalable and practical solution for 3D understanding. Our code is available at https://github.com/Hanzy1996/VDG-Uni3DSeg.
>
---
#### [replaced 030] Level-Set Parameters: Novel Representation for 3D Shape Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.13502v2](http://arxiv.org/pdf/2412.13502v2)**

> **作者:** Huan Lei; Hongdong Li; Andreas Geiger; Anthony Dick
>
> **摘要:** 3D shape analysis has been largely focused on traditional 3D representations of point clouds and meshes, but the discrete nature of these data makes the analysis susceptible to variations in input resolutions. Recent development of neural fields brings in level-set parameters from signed distance functions as a novel, continuous, and numerical representation of 3D shapes, where the shape surfaces are defined as zero-level-sets of those functions. This motivates us to extend shape analysis from the traditional 3D data to these novel parameter data. Since the level-set parameters are not Euclidean like point clouds, we establish correlations across different shapes by formulating them as a pseudo-normal distribution, and learn the distribution prior from the respective dataset. To further explore the level-set parameters with shape transformations, we propose to condition a subset of these parameters on rotations and translations, and generate them with a hypernetwork. This simplifies the pose-related shape analysis compared to using traditional data. We demonstrate the promise of the novel representations through applications in shape classification (arbitrary poses), retrieval, and 6D object pose estimation.
>
---
#### [replaced 031] FedVSR: Towards Model-Agnostic Federated Learning in Video Super-Resolution
- **分类: cs.CV; cs.DC**

- **链接: [http://arxiv.org/pdf/2503.13745v2](http://arxiv.org/pdf/2503.13745v2)**

> **作者:** Ali Mollaahmadi Dehaghi; Hossein KhademSohi; Reza Razavi; Steve Drew; Mohammad Moshirpour
>
> **备注:** This version includes an updated abstract and introduction for improved clarity and context. We also added the LPIPS metric to our evaluation results to provide a more comprehensive assessment of perceptual quality
>
> **摘要:** Video super-resolution aims to enhance low-resolution videos by leveraging both spatial and temporal information. While deep learning has led to impressive progress, it typically requires centralized data, which raises privacy concerns. Federated learning offers a privacy-friendly solution, but general FL frameworks often struggle with low-level vision tasks, resulting in blurry, low-quality outputs. To address this, we introduce FedVSR, the first FL framework specifically designed for VSR. It is model-agnostic and stateless, and introduces a lightweight loss function based on the DWT to better preserve high-frequency details during local training. Additionally, a loss-aware aggregation strategy combines both DWT-based and task-specific losses to guide global updates effectively. Extensive experiments across multiple VSR models and datasets demonstrate that FedVSR consistently outperforms existing FL methods, achieving up to 0.82 dB higher PSNR, 0.0327 higher SSIM, and 0.0251 lower LPIPS. These results underscore FedVSR's ability to bridge the gap between privacy and performance, setting a new benchmark for federated learning in low-level vision tasks. The code is available at: https://github.com/alimd94/FedVSR
>
---
#### [replaced 032] MaskControl: Spatio-Temporal Control for Masked Motion Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.10780v3](http://arxiv.org/pdf/2410.10780v3)**

> **作者:** Ekkasit Pinyoanuntapong; Muhammad Usama Saleem; Korrawe Karunratanakul; Pu Wang; Hongfei Xue; Chen Chen; Chuan Guo; Junli Cao; Jian Ren; Sergey Tulyakov
>
> **备注:** Camera Ready Version. ICCV2025 (Oral). Change name from ControlMM to MaskControl. project page https://exitudio.github.io/ControlMM-page
>
> **摘要:** Recent advances in motion diffusion models have enabled spatially controllable text-to-motion generation. However, these models struggle to achieve high-precision control while maintaining high-quality motion generation. To address these challenges, we propose MaskControl, the first approach to introduce controllability to the generative masked motion model. Our approach introduces two key innovations. First, \textit{Logits Regularizer} implicitly perturbs logits at training time to align the distribution of motion tokens with the controlled joint positions, while regularizing the categorical token prediction to ensure high-fidelity generation. Second, \textit{Logit Optimization} explicitly optimizes the predicted logits during inference time, directly reshaping the token distribution that forces the generated motion to accurately align with the controlled joint positions. Moreover, we introduce \textit{Differentiable Expectation Sampling (DES)} to combat the non-differential distribution sampling process encountered by logits regularizer and optimization. Extensive experiments demonstrate that MaskControl outperforms state-of-the-art methods, achieving superior motion quality (FID decreases by ~77\%) and higher control precision (average error 0.91 vs. 1.08). Additionally, MaskControl enables diverse applications, including any-joint-any-frame control, body-part timeline control, and zero-shot objective control. Video visualization can be found at https://www.ekkasit.com/ControlMM-page/
>
---
#### [replaced 033] Benchmarking of Deep Learning Methods for Generic MRI Multi-Organ Abdominal Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17971v2](http://arxiv.org/pdf/2507.17971v2)**

> **作者:** Deepa Krishnaswamy; Cosmin Ciausu; Steve Pieper; Ron Kikinis; Benjamin Billot; Andrey Fedorov
>
> **摘要:** Recent advances in deep learning have led to robust automated tools for segmentation of abdominal computed tomography (CT). Meanwhile, segmentation of magnetic resonance imaging (MRI) is substantially more challenging due to the inherent signal variability and the increased effort required for annotating training datasets. Hence, existing approaches are trained on limited sets of MRI sequences, which might limit their generalizability. To characterize the landscape of MRI abdominal segmentation tools, we present here a comprehensive benchmarking of the three state-of-the-art and open-source models: MRSegmentator, MRISegmentator-Abdomen, and TotalSegmentator MRI. Since these models are trained using labor-intensive manual annotation cycles, we also introduce and evaluate ABDSynth, a SynthSeg-based model purely trained on widely available CT segmentations (no real images). More generally, we assess accuracy and generalizability by leveraging three public datasets (not seen by any of the evaluated methods during their training), which span all major manufacturers, five MRI sequences, as well as a variety of subject conditions, voxel resolutions, and fields-of-view. Our results reveal that MRSegmentator achieves the best performance and is most generalizable. In contrast, ABDSynth yields slightly less accurate results, but its relaxed requirements in training data make it an alternative when the annotation budget is limited. The evaluation code and datasets are given for future benchmarking at https://github.com/deepakri201/AbdoBench, along with inference code and weights for ABDSynth.
>
---
#### [replaced 034] $S^2M^2$: Scalable Stereo Matching Model for Reliable Depth Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13229v2](http://arxiv.org/pdf/2507.13229v2)**

> **作者:** Junhong Min; Youngpil Jeon; Jimin Kim; Minyong Choi
>
> **备注:** 8 pages, 5 figures, ICCV accepted paper
>
> **摘要:** The pursuit of a generalizable stereo matching model, capable of performing across varying resolutions and disparity ranges without dataset-specific fine-tuning, has revealed a fundamental trade-off. Iterative local search methods achieve high scores on constrained benchmarks, but their core mechanism inherently limits the global consistency required for true generalization. On the other hand, global matching architectures, while theoretically more robust, have been historically rendered infeasible by prohibitive computational and memory costs. We resolve this dilemma with $S^2M^2$: a global matching architecture that achieves both state-of-the-art accuracy and high efficiency without relying on cost volume filtering or deep refinement stacks. Our design integrates a multi-resolution transformer for robust long-range correspondence, trained with a novel loss function that concentrates probability on feasible matches. This approach enables a more robust joint estimation of disparity, occlusion, and confidence. $S^2M^2$ establishes a new state of the art on the Middlebury v3 and ETH3D benchmarks, significantly outperforming prior methods across most metrics while reconstructing high-quality details with competitive efficiency.
>
---
#### [replaced 035] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.13807v4](http://arxiv.org/pdf/2411.13807v4)**

> **作者:** Ruiyuan Gao; Kai Chen; Bo Xiao; Lanqing Hong; Zhenguo Li; Qiang Xu
>
> **备注:** ICCV 2025 camera-ready version, Project Website: https://flymin.github.io/magicdrive-v2/
>
> **摘要:** The rapid advancement of diffusion models has greatly improved video synthesis, especially in controllable video generation, which is vital for applications like autonomous driving. Although DiT with 3D VAE has become a standard framework for video generation, it introduces challenges in controllable driving video generation, especially for geometry control, rendering existing control methods ineffective. To address these issues, we propose MagicDrive-V2, a novel approach that integrates the MVDiT block and spatial-temporal conditional encoding to enable multi-view video generation and precise geometric control. Additionally, we introduce an efficient method for obtaining contextual descriptions for videos to support diverse textual control, along with a progressive training strategy using mixed video data to enhance training efficiency and generalizability. Consequently, MagicDrive-V2 enables multi-view driving video synthesis with $3.3\times$ resolution and $4\times$ frame count (compared to current SOTA), rich contextual control, and geometric controls. Extensive experiments demonstrate MagicDrive-V2's ability, unlocking broader applications in autonomous driving.
>
---
#### [replaced 036] FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Vision Language Models
- **分类: cs.CV; cs.AI; 68T45, 68T50; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2501.01986v2](http://arxiv.org/pdf/2501.01986v2)**

> **作者:** Tianyu Fu; Tengxuan Liu; Qinghao Han; Guohao Dai; Shengen Yan; Huazhong Yang; Xuefei Ning; Yu Wang
>
> **备注:** ICCV 2025
>
> **摘要:** The increasing demand to process long and high-resolution videos significantly burdens Large Vision-Language Models (LVLMs) due to the enormous number of visual tokens. Existing token reduction methods primarily prune tokens based on importance metrics, such as cumulative attention scores. However, even important tokens may exhibit high redundancy caused by similarity among adjacent video frames and repetitive visual elements. To address this limitation, we propose FrameFusion, a novel token reduction approach integrating similarity-based merging with importance-based pruning. We conduct a thorough study on token similarity characteristics, revealing three key insights: (1) spatially corresponding visual tokens between adjacent frames have higher cosine similarities compared to other token pairs; (2) high token similarities prominently decrease in deeper model layers; and (3) token similarity rankings are highly consistent across different layers. Guided by these observations, FrameFusion computes token similarities exclusively between corresponding visual tokens from adjacent frames, applies token merging at initial successive layers followed by pruning in deeper layers, and adopts a cascaded merging strategy to further enhance efficiency. We evaluate FrameFusion comprehensively across six diverse LVLMs, ranging from 2B to 72B parameters, using five video benchmarks encompassing video retrieval, question-answering, and spatial-temporal understanding tasks. Experiments show that FrameFusion reduces visual tokens by 70%, achieving 1.6-3.6x end-to-end speedups, with an average performance impact of less than 3%. Our code is available at: https://github.com/thu-nics/FrameFusion.
>
---
#### [replaced 037] Registration beyond Points: General Affine Subspace Alignment via Geodesic Distance on Grassmann Manifold
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17998v2](http://arxiv.org/pdf/2507.17998v2)**

> **作者:** Jaeho Shin; Hyeonjae Gil; Junwoo Jang; Maani Ghaffari; Ayoung Kim
>
> **摘要:** Affine Grassmannian has been favored for expressing proximity between lines and planes due to its theoretical exactness in measuring distances among features. Despite this advantage, the existing method can only measure the proximity without yielding the distance as an explicit function of rigid body transformation. Thus, an optimizable distance function on the manifold has remained underdeveloped, stifling its application in registration problems. This paper is the first to explicitly derive an optimizable cost function between two Grassmannian features with respect to rigid body transformation ($\mathbf{R}$ and $\mathbf{t}$). Specifically, we present a rigorous mathematical proof demonstrating that the bases of high-dimensional linear subspaces can serve as an explicit representation of the cost. Finally, we propose an optimizable cost function based on the transformed bases that can be applied to the registration problem of any affine subspace. Compared to vector parameter-based approaches, our method is able to find a globally optimal solution by directly minimizing the geodesic distance which is agnostic to representation ambiguity. The resulting cost function and its extension to the inlier-set maximizing Branch-and-Bound (BnB) solver have been demonstrated to improve the convergence of existing solutions or outperform them in various computer vision tasks. The code is available on https://github.com/joomeok/GrassmannRegistration.
>
---
#### [replaced 038] Tuned Reverse Distillation: Enhancing Multimodal Industrial Anomaly Detection with Crossmodal Tuners
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.08949v3](http://arxiv.org/pdf/2412.08949v3)**

> **作者:** Xinyue Liu; Jianyuan Wang; Biao Leng; Shuo Zhang
>
> **摘要:** Knowledge distillation (KD) has been widely studied in unsupervised image Anomaly Detection (AD), but its application to unsupervised multimodal AD remains underexplored. Existing KD-based methods for multimodal AD that use fused multimodal features to obtain teacher representations face challenges. Anomalies that only exist in one modality may not be effectively captured in the fused teacher features, leading to detection failures. Besides, these methods do not fully leverage the rich intra- and inter-modality information that are critical for effective anomaly detection. In this paper, we propose Tuned Reverse Distillation (TRD) based on Multi-branch design to realize Multimodal Industrial AD. By assigning independent branches to each modality, our method enables finer detection of anomalies within each modality. Furthermore, we enhance the interaction between modalities during the distillation process by designing two Crossmodal Tuners including Crossmodal Filter and Amplifier. With the idea of crossmodal mapping, the student network is allowed to better learn normal features while anomalies in all modalities are ensured to be effectively detected. Experimental verifications on multimodal AD datasets demonstrate that our method achieves state-of-the-art performance in multimodal anomaly detection and localization. Code is available at https://github.com/hito2448/TRD.
>
---
#### [replaced 039] ObjectRelator: Enabling Cross-View Object Relation Understanding Across Ego-Centric and Exo-Centric Perspectives
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.19083v2](http://arxiv.org/pdf/2411.19083v2)**

> **作者:** Yuqian Fu; Runze Wang; Bin Ren; Guolei Sun; Biao Gong; Yanwei Fu; Danda Pani Paudel; Xuanjing Huang; Luc Van Gool
>
> **备注:** Accepted by ICCV25 (Highlight)
>
> **摘要:** Bridging the gap between ego-centric and exo-centric views has been a long-standing question in computer vision. In this paper, we focus on the emerging Ego-Exo object correspondence task, which aims to understand object relations across ego-exo perspectives through segmentation. While numerous segmentation models have been proposed, most operate on a single image (view), making them impractical for cross-view scenarios. PSALM, a recently proposed segmentation method, stands out as a notable exception with its demonstrated zero-shot ability on this task. However, due to the drastic viewpoint change between ego and exo, PSALM fails to accurately locate and segment objects, especially in complex backgrounds or when object appearances change significantly. To address these issues, we propose ObjectRelator, a novel approach featuring two key modules: Multimodal Condition Fusion (MCFuse) and SSL-based Cross-View Object Alignment (XObjAlign). MCFuse introduces language as an additional cue, integrating both visual masks and textual descriptions to improve object localization and prevent incorrect associations. XObjAlign enforces cross-view consistency through self-supervised alignment, enhancing robustness to object appearance variations. Extensive experiments demonstrate ObjectRelator's effectiveness on the large-scale Ego-Exo4D benchmark and HANDAL-X (an adapted dataset for cross-view segmentation) with state-of-the-art performance. Code is made available at: http://yuqianfu.com/ObjectRelator.
>
---
#### [replaced 040] MEDTalk: Multimodal Controlled 3D Facial Animation with Dynamic Emotions by Disentangled Embedding
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.06071v3](http://arxiv.org/pdf/2507.06071v3)**

> **作者:** Chang Liu; Ye Pan; Chenyang Ding; Susanto Rahardja; Xiaokang Yang
>
> **摘要:** Audio-driven emotional 3D facial animation aims to generate synchronized lip movements and vivid facial expressions. However, most existing approaches focus on static and predefined emotion labels, limiting their diversity and naturalness. To address these challenges, we propose MEDTalk, a novel framework for fine-grained and dynamic emotional talking head generation. Our approach first disentangles content and emotion embedding spaces from motion sequences using a carefully designed cross-reconstruction process, enabling independent control over lip movements and facial expressions. Beyond conventional audio-driven lip synchronization, we integrate audio and speech text, predicting frame-wise intensity variations and dynamically adjusting static emotion features to generate realistic emotional expressions. Furthermore, to enhance control and personalization, we incorporate multimodal inputs-including text descriptions and reference expression images-to guide the generation of user-specified facial expressions. With MetaHuman as the priority, our generated results can be conveniently integrated into the industrial production pipeline. The code is available at: https://github.com/SJTU-Lucy/MEDTalk.
>
---
#### [replaced 041] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18192v2](http://arxiv.org/pdf/2507.18192v2)**

> **作者:** Minghao Fu; Guo-Hua Wang; Xiaohao Chen; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** Accepted by ICCV 2025. The code is publicly available at https://github.com/AIDC-AI/TeEFusion
>
> **摘要:** Recent advances in text-to-image synthesis largely benefit from sophisticated sampling strategies and classifier-free guidance (CFG) to ensure high-quality generation. However, CFG's reliance on two forward passes, especially when combined with intricate sampling algorithms, results in prohibitively high inference costs. To address this, we introduce TeEFusion (Text Embeddings Fusion), a novel and efficient distillation method that directly incorporates the guidance magnitude into the text embeddings and distills the teacher model's complex sampling strategy. By simply fusing conditional and unconditional text embeddings using linear operations, TeEFusion reconstructs the desired guidance without adding extra parameters, simultaneously enabling the student model to learn from the teacher's output produced via its sophisticated sampling approach. Extensive experiments on state-of-the-art models such as SD3 demonstrate that our method allows the student to closely mimic the teacher's performance with a far simpler and more efficient sampling strategy. Consequently, the student model achieves inference speeds up to 6$\times$ faster than the teacher model, while maintaining image quality at levels comparable to those obtained through the teacher's complex sampling approach. The code is publicly available at https://github.com/AIDC-AI/TeEFusion.
>
---
#### [replaced 042] Verbalized Representation Learning for Interpretable Few-Shot Generalization
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.18651v2](http://arxiv.org/pdf/2411.18651v2)**

> **作者:** Cheng-Fu Yang; Da Yin; Wenbo Hu; Nanyun Peng; Bolei Zhou; Kai-Wei Chang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Humans recognize objects after observing only a few examples, a remarkable capability enabled by their inherent language understanding of the real-world environment. Developing verbalized and interpretable representation can significantly improve model generalization in low-data settings. In this work, we propose Verbalized Representation Learning (VRL), a novel approach for automatically extracting human-interpretable features for object recognition using few-shot data. Our method uniquely captures inter-class differences and intra-class commonalities in the form of natural language by employing a Vision-Language Model (VLM) to identify key discriminative features between different classes and shared characteristics within the same class. These verbalized features are then mapped to numeric vectors through the VLM. The resulting feature vectors can be further utilized to train and infer with downstream classifiers. Experimental results show that, at the same model scale, VRL achieves a 24% absolute improvement over prior state-of-the-art methods while using 95% less data and a smaller mode. Furthermore, compared to human-labeled attributes, the features learned by VRL exhibit a 20% absolute gain when used for downstream classification tasks. Code is available at: https://github.com/joeyy5588/VRL/tree/main.
>
---
#### [replaced 043] Geometric Origins of Bias in Deep Neural Networks: A Human Visual System Perspective
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11809v4](http://arxiv.org/pdf/2502.11809v4)**

> **作者:** Yanbiao Ma; Bowei Liu; Andi Zhang
>
> **摘要:** Bias formation in deep neural networks (DNNs) remains a critical yet poorly understood challenge, influencing both fairness and reliability in artificial intelligence systems. Inspired by the human visual system, which decouples object manifolds through hierarchical processing to achieve object recognition, we propose a geometric analysis framework linking the geometric complexity of class-specific perceptual manifolds in DNNs to model bias. Our findings reveal that differences in geometric complexity can lead to varying recognition capabilities across categories, introducing biases. To support this analysis, we present the Perceptual-Manifold-Geometry library, designed for calculating the geometric properties of perceptual manifolds. The toolkit has been downloaded and installed over 4,500 times. This work provides a novel geometric perspective on bias formation in modern learning systems and lays a theoretical foundation for developing more equitable and robust artificial intelligence.
>
---
#### [replaced 044] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04963v3](http://arxiv.org/pdf/2505.04963v3)**

> **作者:** Onkar Susladkar; Gayatri Deshmukh; Yalcin Tur; Gorkhem Durak; Ulas Bagci
>
> **备注:** Accepted in ICCV 2025
>
> **摘要:** Synthesizing medical images remains challenging due to limited annotated pathological data, modality domain gaps, and the complexity of representing diffuse pathologies such as liver cirrhosis. Existing methods often struggle to maintain anatomical fidelity while accurately modeling pathological features, frequently relying on priors derived from natural images or inefficient multi-step sampling. In this work, we introduce ViCTr (Vital Consistency Transfer), a novel two-stage framework that combines a rectified flow trajectory with a Tweedie-corrected diffusion process to achieve high-fidelity, pathology-aware image synthesis. First, we pretrain ViCTr on the ATLAS-8k dataset using Elastic Weight Consolidation (EWC) to preserve critical anatomical structures. We then fine-tune the model adversarially with Low-Rank Adaptation (LoRA) modules for precise control over pathology severity. By reformulating Tweedie's formula within a linear trajectory framework, ViCTr supports one-step sampling, reducing inference from 50 steps to just 4, without sacrificing anatomical realism. We evaluate ViCTr on BTCV (CT), AMOS (MRI), and CirrMRI600+ (cirrhosis) datasets. Results demonstrate state-of-the-art performance, achieving a Medical Frechet Inception Distance (MFID) of 17.01 for cirrhosis synthesis 28% lower than existing approaches and improving nnUNet segmentation by +3.8% mDSC when used for data augmentation. Radiologist reviews indicate that ViCTr-generated liver cirrhosis MRIs are clinically indistinguishable from real scans. To our knowledge, ViCTr is the first method to provide fine-grained, pathology-aware MRI synthesis with graded severity control, closing a critical gap in AI-driven medical imaging research.
>
---
#### [replaced 045] Improving Multislice Electron Ptychography with a Generative Prior
- **分类: eess.IV; cond-mat.mtrl-sci; cs.CV; physics.optics**

- **链接: [http://arxiv.org/pdf/2507.17800v2](http://arxiv.org/pdf/2507.17800v2)**

> **作者:** Christian K. Belardi; Chia-Hao Lee; Yingheng Wang; Justin Lovelace; Kilian Q. Weinberger; David A. Muller; Carla P. Gomes
>
> **备注:** 16 pages, 10 figures, 5 tables
>
> **摘要:** Multislice electron ptychography (MEP) is an inverse imaging technique that computationally reconstructs the highest-resolution images of atomic crystal structures from diffraction patterns. Available algorithms often solve this inverse problem iteratively but are both time consuming and produce suboptimal solutions due to their ill-posed nature. We develop MEP-Diffusion, a diffusion model trained on a large database of crystal structures specifically for MEP to augment existing iterative solvers. MEP-Diffusion is easily integrated as a generative prior into existing reconstruction methods via Diffusion Posterior Sampling (DPS). We find that this hybrid approach greatly enhances the quality of the reconstructed 3D volumes, achieving a 90.50% improvement in SSIM over existing methods.
>
---
#### [replaced 046] Learning Multi-frame and Monocular Prior for Estimating Geometry in Dynamic Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01737v2](http://arxiv.org/pdf/2505.01737v2)**

> **作者:** Seong Hyeon Park; Jinwoo Shin
>
> **备注:** This paper was supported by RLWRD
>
> **摘要:** In monocular videos that capture dynamic scenes, estimating the 3D geometry of video contents has been a fundamental challenge in computer vision. Specifically, the task is significantly challenged by the object motion, where existing models are limited to predict only partial attributes of the dynamic scenes, such as depth or pointmaps spanning only over a pair of frames. Since these attributes are inherently noisy under multiple frames, test-time global optimizations are often employed to fully recover the geometry, which is liable to failure and incurs heavy inference costs. To address the challenge, we present a new model, coined MMP, to estimate the geometry in a feed-forward manner, which produces a dynamic pointmap representation that evolves over multiple frames. Specifically, based on the recent Siamese architecture, we introduce a new trajectory encoding module to project point-wise dynamics on the representation for each frame, which can provide significantly improved expressiveness for dynamic scenes. In our experiments, we find MMP can achieve state-of-the-art quality in feed-forward pointmap prediction, e.g., 15.1% enhancement in the regression error.
>
---
#### [replaced 047] Label Anything: Multi-Class Few-Shot Semantic Segmentation with Visual Prompts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.02075v3](http://arxiv.org/pdf/2407.02075v3)**

> **作者:** Pasquale De Marinis; Nicola Fanelli; Raffaele Scaringi; Emanuele Colonna; Giuseppe Fiameni; Gennaro Vessio; Giovanna Castellano
>
> **备注:** ECAI 2025 - 28th European Conference on Artificial Intelligence
>
> **摘要:** Few-shot semantic segmentation aims to segment objects from previously unseen classes using only a limited number of labeled examples. In this paper, we introduce Label Anything, a novel transformer-based architecture designed for multi-prompt, multi-way few-shot semantic segmentation. Our approach leverages diverse visual prompts -- points, bounding boxes, and masks -- to create a highly flexible and generalizable framework that significantly reduces annotation burden while maintaining high accuracy. Label Anything makes three key contributions: ($\textit{i}$) we introduce a new task formulation that relaxes conventional few-shot segmentation constraints by supporting various types of prompts, multi-class classification, and enabling multiple prompts within a single image; ($\textit{ii}$) we propose a novel architecture based on transformers and attention mechanisms; and ($\textit{iii}$) we design a versatile training procedure allowing our model to operate seamlessly across different $N$-way $K$-shot and prompt-type configurations with a single trained model. Our extensive experimental evaluation on the widely used COCO-$20^i$ benchmark demonstrates that Label Anything achieves state-of-the-art performance among existing multi-way few-shot segmentation methods, while significantly outperforming leading single-class models when evaluated in multi-class settings. Code and trained models are available at https://github.com/pasqualedem/LabelAnything.
>
---
#### [replaced 048] Towards Robust and Controllable Text-to-Motion via Masked Autoregressive Diffusion
- **分类: cs.CV; cs.MM; I.3.8**

- **链接: [http://arxiv.org/pdf/2505.11013v2](http://arxiv.org/pdf/2505.11013v2)**

> **作者:** Zongye Zhang; Bohan Kong; Qingjie Liu; Yunhong Wang
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Generating 3D human motion from text descriptions remains challenging due to the diverse and complex nature of human motion. While existing methods excel within the training distribution, they often struggle with out-of-distribution motions, limiting their applicability in real-world scenarios. Existing VQVAE-based methods often fail to represent novel motions faithfully using discrete tokens, which hampers their ability to generalize beyond seen data. Meanwhile, diffusion-based methods operating on continuous representations often lack fine-grained control over individual frames. To address these challenges, we propose a robust motion generation framework MoMADiff, which combines masked modeling with diffusion processes to generate motion using frame-level continuous representations. Our model supports flexible user-provided keyframe specification, enabling precise control over both spatial and temporal aspects of motion synthesis. MoMADiff demonstrates strong generalization capability on novel text-to-motion datasets with sparse keyframes as motion prompts. Extensive experiments on two held-out datasets and two standard benchmarks show that our method consistently outperforms state-of-the-art models in motion quality, instruction fidelity, and keyframe adherence. The code is available at: https://github.com/zzysteve/MoMADiff
>
---
#### [replaced 049] FBSDiff: Plug-and-Play Frequency Band Substitution of Diffusion Features for Highly Controllable Text-Driven Image Translation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.00998v5](http://arxiv.org/pdf/2408.00998v5)**

> **作者:** Xiang Gao; Jiaying Liu
>
> **备注:** Accepted conference paper of ACM MM 2024
>
> **摘要:** Large-scale text-to-image diffusion models have been a revolutionary milestone in the evolution of generative AI and multimodal technology, allowing wonderful image generation with natural-language text prompt. However, the issue of lacking controllability of such models restricts their practical applicability for real-life content creation. Thus, attention has been focused on leveraging a reference image to control text-to-image synthesis, which is also regarded as manipulating (or editing) a reference image as per a text prompt, namely, text-driven image-to-image translation. This paper contributes a novel, concise, and efficient approach that adapts pre-trained large-scale text-to-image (T2I) diffusion model to the image-to-image (I2I) paradigm in a plug-and-play manner, realizing high-quality and versatile text-driven I2I translation without any model training, model fine-tuning, or online optimization process. To guide T2I generation with a reference image, we propose to decompose diverse guiding factors with different frequency bands of diffusion features in the DCT spectral space, and accordingly devise a novel frequency band substitution layer which realizes dynamic control of the reference image to the T2I generation result in a plug-and-play manner. We demonstrate that our method allows flexible control over both guiding factor and guiding intensity of the reference image simply by tuning the type and bandwidth of the substituted frequency band, respectively. Extensive qualitative and quantitative experiments verify superiority of our approach over related methods in I2I translation visual quality, versatility, and controllability. The code is publicly available at: https://github.com/XiangGao1102/FBSDiff.
>
---
#### [replaced 050] Unraveling the geometry of visual relational reasoning
- **分类: q-bio.NC; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17382v2](http://arxiv.org/pdf/2502.17382v2)**

> **作者:** Jiaqi Shang; Gabriel Kreiman; Haim Sompolinsky
>
> **备注:** 27 pages, 7 figures, 8 SI figures, 2 SI tables
>
> **摘要:** Humans readily generalize abstract relations, such as recognizing "constant" in shape or color, whereas neural networks struggle, limiting their flexible reasoning. To investigate mechanisms underlying such generalization, we introduce SimplifiedRPM, a novel benchmark for systematically evaluating abstract relational reasoning, addressing limitations in prior datasets. In parallel, we conduct human experiments to quantify relational difficulty, enabling direct model-human comparisons. Testing four models, ResNet-50, Vision Transformer, Wild Relation Network, and Scattering Compositional Learner (SCL), we find that SCL generalizes best and most closely aligns with human behavior. Using a geometric approach, we identify key representation properties that accurately predict generalization and uncover a fundamental trade-off between signal and dimensionality: novel relations compress into training-induced subspaces. Layer-wise analysis reveals where relational structure emerges, highlights bottlenecks, and generates concrete hypotheses about abstract reasoning in the brain. Motivated by these insights, we propose SNRloss, a novel objective explicitly balancing representation geometry. Our results establish a geometric foundation for relational reasoning, paving the way for more human-like visual reasoning in AI and opening promising avenues for extending geometric analysis to broader cognitive tasks.
>
---
#### [replaced 051] HumorDB: Can AI understand graphical humor?
- **分类: cs.CV; cs.AI; I.5.4**

- **链接: [http://arxiv.org/pdf/2406.13564v2](http://arxiv.org/pdf/2406.13564v2)**

> **作者:** Veedant Jain; Gabriel Kreiman; Felipe dos Santos Alves Feitosa
>
> **备注:** 10 main figures, 4 additional appendix figures
>
> **摘要:** Despite significant advancements in image segmentation and object detection, understanding complex scenes remains a significant challenge. Here, we focus on graphical humor as a paradigmatic example of image interpretation that requires elucidating the interaction of different scene elements in the context of prior cognitive knowledge. This paper introduces \textbf{HumorDB}, a novel, controlled, and carefully curated dataset designed to evaluate and advance visual humor understanding by AI systems. The dataset comprises diverse images spanning photos, cartoons, sketches, and AI-generated content, including minimally contrastive pairs where subtle edits differentiate between humorous and non-humorous versions. We evaluate humans, state-of-the-art vision models, and large vision-language models on three tasks: binary humor classification, funniness rating prediction, and pairwise humor comparison. The results reveal a gap between current AI systems and human-level humor understanding. While pretrained vision-language models perform better than vision-only models, they still struggle with abstract sketches and subtle humor cues. Analysis of attention maps shows that even when models correctly classify humorous images, they often fail to focus on the precise regions that make the image funny. Preliminary mechanistic interpretability studies and evaluation of model explanations provide initial insights into how different architectures process humor. Our results identify promising trends and current limitations, suggesting that an effective understanding of visual humor requires sophisticated architectures capable of detecting subtle contextual features and bridging the gap between visual perception and abstract reasoning. All the code and data are available here: \href{https://github.com/kreimanlab/HumorDB}{https://github.com/kreimanlab/HumorDB}
>
---
#### [replaced 052] Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2505.15265v2](http://arxiv.org/pdf/2505.15265v2)**

> **作者:** Zihao Pan; Yu Tong; Weibin Wu; Jingyi Wang; Lifeng Chen; Zhe Zhao; Jiajia Wei; Yitong Qiao; Zibin Zheng
>
> **备注:** The paper needs major revisions, so it is being withdrawn
>
> **摘要:** Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research.
>
---
#### [replaced 053] VIBE: Video-Input Brain Encoder for fMRI Response Modeling
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17958v2](http://arxiv.org/pdf/2507.17958v2)**

> **作者:** Daniel Carlström Schad; Shrey Dixit; Janis Keck; Viktor Studenyak; Aleksandr Shpilevoi; Andrej Bicanski
>
> **摘要:** We present VIBE, a two-stage Transformer that fuses multi-modal video, audio, and text features to predict fMRI activity. Representations from open-source models (Qwen2.5, BEATs, Whisper, SlowFast, V-JEPA) are merged by a modality-fusion transformer and temporally decoded by a prediction transformer with rotary embeddings. Trained on 65 hours of movie data from the CNeuroMod dataset and ensembled across 20 seeds, VIBE attains mean parcel-wise Pearson correlations of 0.3225 on in-distribution Friends S07 and 0.2125 on six out-of-distribution films. An earlier iteration of the same architecture obtained 0.3198 and 0.2096, respectively, winning Phase-1 and placing second overall in the Algonauts 2025 Challenge.
>
---
#### [replaced 054] SE-VLN: A Self-Evolving Vision-Language Navigation Framework Based on Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13152v2](http://arxiv.org/pdf/2507.13152v2)**

> **作者:** Xiangyu Dong; Haoran Zhao; Jiang Gao; Haozhou Li; Xiaoguang Ma; Yaoming Zhou; Fuhai Chen; Juan Liu
>
> **摘要:** Recent advances in vision-language navigation (VLN) were mainly attributed to emerging large language models (LLMs). These methods exhibited excellent generalization capabilities in instruction understanding and task reasoning. However, they were constrained by the fixed knowledge bases and reasoning abilities of LLMs, preventing fully incorporating experiential knowledge and thus resulting in a lack of efficient evolutionary capacity. To address this, we drew inspiration from the evolution capabilities of natural agents, and proposed a self-evolving VLN framework (SE-VLN) to endow VLN agents with the ability to continuously evolve during testing. To the best of our knowledge, it was the first time that an multimodal LLM-powered self-evolving VLN framework was proposed. Specifically, SE-VLN comprised three core modules, i.e., a hierarchical memory module to transfer successful and failure cases into reusable knowledge, a retrieval-augmented thought-based reasoning module to retrieve experience and enable multi-step decision-making, and a reflection module to realize continual evolution. Comprehensive tests illustrated that the SE-VLN achieved navigation success rates of 57% and 35.2% in unseen environments, representing absolute performance improvements of 23.9% and 15.0% over current state-of-the-art methods on R2R and REVERSE datasets, respectively. Moreover, the SE-VLN showed performance improvement with increasing experience repository, elucidating its great potential as a self-evolving agent framework for VLN.
>
---
#### [replaced 055] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.06547v2](http://arxiv.org/pdf/2507.06547v2)**

> **作者:** Yonghyun Park; Chieh-Hsin Lai; Satoshi Hayakawa; Yuhta Takida; Naoki Murata; Wei-Hsiang Liao; Woosung Choi; Kin Wai Cheuk; Junghyun Koo; Yuki Mitsufuji
>
> **备注:** Preprint
>
> **摘要:** While diffusion models excel at image generation, their growing adoption raises critical concerns around copyright issues and model transparency. Existing attribution methods identify training examples influencing an entire image, but fall short in isolating contributions to specific elements, such as styles or objects, that matter most to stakeholders. To bridge this gap, we introduce \emph{concept-level attribution} via a novel method called \emph{Concept-TRAK}. Concept-TRAK extends influence functions with two key innovations: (1) a reformulated diffusion training loss based on diffusion posterior sampling, enabling robust, sample-specific attribution; and (2) a concept-aware reward function that emphasizes semantic relevance. We evaluate Concept-TRAK on the AbC benchmark, showing substantial improvements over prior methods. Through diverse case studies--ranging from identifying IP-protected and unsafe content to analyzing prompt engineering and compositional learning--we demonstrate how concept-level attribution yields actionable insights for responsible generative AI development and governance.
>
---
#### [replaced 056] A Multimodal Seq2Seq Transformer for Predicting Brain Responses to Naturalistic Stimuli
- **分类: cs.CV; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2507.18104v2](http://arxiv.org/pdf/2507.18104v2)**

> **作者:** Qianyi He; Yuan Chang Leong
>
> **摘要:** The Algonauts 2025 Challenge called on the community to develop encoding models that predict whole-brain fMRI responses to naturalistic multimodal movies. In this submission, we propose a sequence-to-sequence Transformer that autoregressively predicts fMRI activity from visual, auditory, and language inputs. Stimulus features were extracted using pretrained models including VideoMAE, HuBERT, Qwen, and BridgeTower. The decoder integrates information from prior brain states and current stimuli via dual cross-attention mechanisms that attend to both perceptual information extracted from the stimulus as well as narrative information provided by high-level summaries of the content. One core innovation of our approach is the use of sequences of multimodal context to predict sequences of brain activity, enabling the model to capture long-range temporal structure in both stimuli and neural responses. Another is the combination of a shared encoder with partial subject-specific decoder, which leverages common representational structure across subjects while accounting for individual variability. Our model achieves strong performance on both in-distribution and out-of-distribution data, demonstrating the effectiveness of temporally-aware, multimodal sequence modeling for brain activity prediction. The code is available at https://github.com/Angelneer926/Algonauts_challenge.
>
---
#### [replaced 057] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16289v2](http://arxiv.org/pdf/2503.16289v2)**

> **作者:** Inwoo Hwang; Bing Zhou; Young Min Kim; Jian Wang; Chuan Guo
>
> **备注:** Accepted to ICCV 2025. Project page: http://inwoohwang.me/SceneMI
>
> **摘要:** Modeling human-scene interactions (HSI) is essential for understanding and simulating everyday human behaviors. Recent approaches utilizing generative modeling have made progress in this domain; however, they are limited in controllability and flexibility for real-world applications. To address these challenges, we propose reformulating the HSI modeling problem as Scene-aware Motion In-betweening - a more tractable and practical task. We introduce SceneMI, a framework that supports several practical applications, including keyframe-guided character animation in 3D scenes and enhancing the motion quality of imperfect HSI data. SceneMI employs dual scene descriptors to comprehensively encode global and local scene context. Furthermore, our framework leverages the inherent denoising nature of diffusion models to generalize on noisy keyframes. Experimental results demonstrate SceneMI's effectiveness in scene-aware keyframe in-betweening and generalization to the real-world GIMO dataset, where motions and scenes are acquired by noisy IMU sensors and smartphones. We further showcase SceneMI's applicability in HSI reconstruction from monocular videos.
>
---
#### [replaced 058] GIE-Bench: Towards Grounded Evaluation for Text-Guided Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11493v3](http://arxiv.org/pdf/2505.11493v3)**

> **作者:** Yusu Qian; Jiasen Lu; Tsu-Jui Fu; Xinze Wang; Chen Chen; Yinfei Yang; Wenze Hu; Zhe Gan
>
> **备注:** Project page: https://sueqian6.github.io/GIE-Bench-web/
>
> **摘要:** Editing images using natural language instructions has become a natural and expressive way to modify visual content; yet, evaluating the performance of such models remains challenging. Existing evaluation approaches often rely on image-text similarity metrics like CLIP, which lack precision. In this work, we introduce a new benchmark designed to evaluate text-guided image editing models in a more grounded manner, along two critical dimensions: (i) functional correctness, assessed via automatically generated multiple-choice questions that verify whether the intended change was successfully applied; and (ii) image content preservation, which ensures that non-targeted regions of the image remain visually consistent using an object-aware masking technique and preservation scoring. The benchmark includes over 1000 high-quality editing examples across 20 diverse content categories, each annotated with detailed editing instructions, evaluation questions, and spatial object masks. We conduct a large-scale study comparing GPT-Image-1, the latest flagship in the text-guided image editing space, against several state-of-the-art editing models, and validate our automatic metrics against human ratings. Results show that GPT-Image-1 leads in instruction-following accuracy, but often over-modifies irrelevant image regions, highlighting a key trade-off in the current model behavior. GIE-Bench provides a scalable, reproducible framework for advancing more accurate evaluation of text-guided image editing.
>
---
#### [replaced 059] Semi-autonomous Prosthesis Control Using Minimal Depth Information and Vibrotactile Feedback
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2210.00541v2](http://arxiv.org/pdf/2210.00541v2)**

> **作者:** Miguel Nobre Castro; Strahinja Dosen
>
> **摘要:** Semi-autonomous prosthesis controllers based on computer vision improve performance while reducing cognitive effort. However, controllers relying on full-depth data face challenges in being deployed as embedded prosthesis controllers due to the computational demands of processing point clouds. To address this, the present study proposes a method to reconstruct the shape of various daily objects from minimal depth data. This is achieved using four concurrent laser scanner lines instead of a full point cloud. These lines represent the partial contours of an object's cross-section, enabling its dimensions and orientation to be reconstructed using simple geometry. A control prototype was implemented using a depth sensor with four laser scanners. Vibrotactile feedback was also designed to help users to correctly aim the sensor at target objects. Ten able-bodied volunteers used a prosthesis equipped with the novel controller to grasp ten objects of varying shapes, sizes, and orientations. For comparison, they also tested an existing benchmark controller that used full-depth information. The results showed that the novel controller handled all objects and, while performance improved with training, it remained slightly below that of the benchmark. This marks an important step towards a compact vision-based system for embedded depth sensing in prosthesis grasping.
>
---
#### [replaced 060] AgMMU: A Comprehensive Agricultural Multimodal Understanding Benchmark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10568v2](http://arxiv.org/pdf/2504.10568v2)**

> **作者:** Aruna Gauba; Irene Pi; Yunze Man; Ziqi Pang; Vikram S. Adve; Yu-Xiong Wang
>
> **备注:** Project Website: https://agmmu.github.io/ Huggingface: https://huggingface.co/datasets/AgMMU/AgMMU_v1/
>
> **摘要:** We present AgMMU, a challenging real-world benchmark for evaluating and advancing vision-language models (VLMs) in the knowledge-intensive domain of agriculture. Unlike prior datasets that rely on crowdsourced prompts, AgMMU is distilled from 116,231 authentic dialogues between everyday growers and USDA-authorized Cooperative Extension experts. Through a three-stage pipeline: automated knowledge extraction, QA generation, and human verification, we construct (i) AgMMU, an evaluation set of 746 multiple-choice questions (MCQs) and 746 open-ended questions (OEQs), and (ii) AgBase, a development corpus of 57,079 multimodal facts covering five high-stakes agricultural topics: insect identification, species identification, disease categorization, symptom description, and management instruction. Benchmarking 12 leading VLMs reveals pronounced gaps in fine-grained perception and factual grounding. Open-sourced models trail after proprietary ones by a wide margin. Simple fine-tuning on AgBase boosts open-sourced model performance on challenging OEQs for up to 11.6% on average, narrowing this gap and also motivating future research to propose better strategies in knowledge extraction and distillation from AgBase. We hope AgMMU stimulates research on domain-specific knowledge integration and trustworthy decision support in agriculture AI development.
>
---
#### [replaced 061] Latent Space Analysis for Melanoma Prevention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18414v2](http://arxiv.org/pdf/2506.18414v2)**

> **作者:** Ciro Listone; Aniello Murano
>
> **备注:** The proposed approach presents some technical imperfections and needs to be refined with further examinations
>
> **摘要:** Melanoma represents a critical health risk due to its aggressive progression and high mortality, underscoring the need for early, interpretable diagnostic tools. While deep learning has advanced in skin lesion classification, most existing models provide only binary outputs, offering limited clinical insight. This work introduces a novel approach that extends beyond classification, enabling interpretable risk modelling through a Conditional Variational Autoencoder. The proposed method learns a structured latent space that captures semantic relationships among lesions, allowing for a nuanced, continuous assessment of morphological differences. An SVM is also trained on this representation effectively differentiating between benign nevi and melanomas, demonstrating strong and consistent performance. More importantly, the learned latent space supports visual and geometric interpretation of malignancy, with the spatial proximity of a lesion to known melanomas serving as a meaningful indicator of risk. This approach bridges predictive performance with clinical applicability, fostering early detection, highlighting ambiguous cases, and enhancing trust in AI-assisted diagnosis through transparent and interpretable decision-making.
>
---
#### [replaced 062] RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22800v3](http://arxiv.org/pdf/2506.22800v3)**

> **作者:** Sicong Du; Jiarun Liu; Qifeng Chen; Hao-Xiang Chen; Tai-Jiang Mu; Sheng Yang
>
> **摘要:** A single-pass driving clip frequently results in incomplete scanning of the road structure, making reconstructed scene expanding a critical requirement for sensor simulators to effectively regress driving actions. Although contemporary 3D Gaussian Splatting (3DGS) techniques achieve remarkable reconstruction quality, their direct extension through the integration of diffusion priors often introduces cumulative physical inconsistencies and compromises training efficiency. To address these limitations, we present RGE-GS, a novel expansive reconstruction framework that synergizes diffusion-based generation with reward-guided Gaussian integration. The RGE-GS framework incorporates two key innovations: First, we propose a reward network that learns to identify and prioritize consistently generated patterns prior to reconstruction phases, thereby enabling selective retention of diffusion outputs for spatial stability. Second, during the reconstruction process, we devise a differentiated training strategy that automatically adjust Gaussian optimization progress according to scene converge metrics, which achieving better convergence than baseline methods. Extensive evaluations of publicly available datasets demonstrate that RGE-GS achieves state-of-the-art performance in reconstruction quality. Our source-code will be made publicly available at https://github.com/CN-ADLab/RGE-GS.
>
---
