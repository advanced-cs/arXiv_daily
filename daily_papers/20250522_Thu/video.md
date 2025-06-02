# 计算机视觉 cs.CV

- **最新发布 142 篇**

- **更新 81 篇**

## 最新发布

#### [new 001] Uncovering Cultural Representation Disparities in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLMs）的文化偏见问题，通过在Country211数据集上测试不同提示策略下的国家识别任务，揭示模型因训练数据分布和规模导致的跨文化表现差异，指出其数据偏见来源。**

- **链接: [http://arxiv.org/pdf/2505.14729v1](http://arxiv.org/pdf/2505.14729v1)**

> **作者:** Ram Mohan Rao Kadiyala; Siddhant Gupta; Jebish Purbey; Srishti Yadav; Alejandro Salamanca; Desmond Elliott
>
> **备注:** 26 pages, 36 figures
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated impressive capabilities across a range of tasks, yet concerns about their potential biases exist. This work investigates the extent to which prominent VLMs exhibit cultural biases by evaluating their performance on an image-based country identification task at a country level. Utilizing the geographically diverse Country211 dataset, we probe several large vision language models (VLMs) under various prompting strategies: open-ended questions, multiple-choice questions (MCQs) including challenging setups like multilingual and adversarial settings. Our analysis aims to uncover disparities in model accuracy across different countries and question formats, providing insights into how training data distribution and evaluation methodologies might influence cultural biases in VLMs. The findings highlight significant variations in performance, suggesting that while VLMs possess considerable visual understanding, they inherit biases from their pre-training data and scale that impact their ability to generalize uniformly across diverse global contexts.
>
---
#### [new 002] RUSplatting: Robust 3D Gaussian Splatting for Sparse-View Underwater Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出RUSplatting方法，解决水下稀疏视角场景重建中的光衰减、低可见度和视角不一致问题。通过分通道RGB学习恢复颜色，帧插值与自适应加权提升视图一致性，设计抗噪保边损失函数，并发布Submerged3D数据集。实验显示较SOTA方法PSNR提升1.9dB，提升海洋应用的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.15737v1](http://arxiv.org/pdf/2505.15737v1)**

> **作者:** Zhuodong Jiang; Haoran Wang; Guoxi Huang; Brett Seymour; Nantheera Anantrasirichai
>
> **备注:** 10 pages, 3 figures. Submitted to BMVC 2025
>
> **摘要:** Reconstructing high-fidelity underwater scenes remains a challenging task due to light absorption, scattering, and limited visibility inherent in aquatic environments. This paper presents an enhanced Gaussian Splatting-based framework that improves both the visual quality and geometric accuracy of deep underwater rendering. We propose decoupled learning for RGB channels, guided by the physics of underwater attenuation, to enable more accurate colour restoration. To address sparse-view limitations and improve view consistency, we introduce a frame interpolation strategy with a novel adaptive weighting scheme. Additionally, we introduce a new loss function aimed at reducing noise while preserving edges, which is essential for deep-sea content. We also release a newly collected dataset, Submerged3D, captured specifically in deep-sea environments. Experimental results demonstrate that our framework consistently outperforms state-of-the-art methods with PSNR gains up to 1.90dB, delivering superior perceptual quality and robustness, and offering promising directions for marine robotics and underwater visual analytics.
>
---
#### [new 003] Convolutional Long Short-Term Memory Neural Networks Based Numerical Simulation of Flow Field
- **分类: cs.CV**

- **简介: 该论文针对传统计算流体动力学（CFD）收敛慢、精度依赖性强的问题，提出改进的ConvLSTM模型结合残差网络和注意力机制，用于流场预测。通过动态网格技术与UDF模拟圆柱绕流，构建尾流区多时态数据集，提升时空特征提取效率，减少参数与训练时间。**

- **链接: [http://arxiv.org/pdf/2505.15533v1](http://arxiv.org/pdf/2505.15533v1)**

> **作者:** Chang Liu
>
> **备注:** ICIC 2025 accepted
>
> **摘要:** Computational Fluid Dynamics (CFD) is the main approach to analyzing flow field. However, the convergence and accuracy depend largely on mathematical models of flow, numerical methods, and time consumption. Deep learning-based analysis of flow filed provides an alternative. For the task of flow field prediction, an improved Convolutional Long Short-Term Memory (Con-vLSTM) Neural Network is proposed as the baseline network in consideration of the temporal and spatial characteristics of flow field. Combining dynamic mesh technology and User-Defined Function (UDF), numerical simulations of flow around a circular cylinder were conducted. Flow field snapshots were used to sample data from the cylinder's wake region at different time instants, constructing a flow field dataset with sufficient volume and rich flow state var-iations. Residual networks and attention mechanisms are combined with the standard ConvLSTM model. Compared with the standard ConvLSTM model, the results demonstrate that the improved ConvLSTM model can extract more temporal and spatial features while having fewer parameters and shorter train-ing time.
>
---
#### [new 004] seg_3D_by_PC2D: Multi-View Projection for Domain Generalization and Adaptation in 3D Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D语义分割任务，针对领域泛化（DG）和无监督领域自适应（UDA）中模型跨数据集性能下降的问题。提出多视角投影框架seg_3D_by_PC2D，将激光雷达点云对齐并渲染为多视角2D图像（PC2D），训练2D分割模型，推理时通过多视角预测融合生成3D标签。实验在nuScenes和SemanticKITTI数据集上取得UDA最优、DG近最优效果，尤其提升大静态物体分割精度。**

- **链接: [http://arxiv.org/pdf/2505.15545v1](http://arxiv.org/pdf/2505.15545v1)**

> **作者:** Andrew Caunes; Thierry Chateau; Vincent Fremont
>
> **摘要:** 3D semantic segmentation plays a pivotal role in autonomous driving and road infrastructure analysis, yet state-of-the-art 3D models are prone to severe domain shift when deployed across different datasets. We propose a novel multi-view projection framework that excels in both domain generalization (DG) and unsupervised domain adaptation (UDA). Our approach first aligns Lidar scans into coherent 3D scenes and renders them from multiple virtual camera poses to create a large-scale synthetic 2D dataset (PC2D). We then use it to train a 2D segmentation model in-domain. During inference, the model processes hundreds of views per scene; the resulting logits are back-projected to 3D with an occlusion-aware voting scheme to generate final point-wise labels. Our framework is modular and enables extensive exploration of key design parameters, such as view generation optimization (VGO), visualization modality optimization (MODO), and 2D model choice. We evaluate on the nuScenes and SemanticKITTI datasets under both the DG and UDA settings. We achieve state-of-the-art results in UDA and close to state-of-the-art in DG, with particularly large gains on large, static classes. Our code and dataset generation tools will be publicly available at https://github.com/andrewcaunes/ia4markings
>
---
#### [new 005] Spectral-Aware Global Fusion for RGB-Thermal Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于RGB-热成像语义分割任务，解决多模态特征融合的模态差异问题。提出SGFNet，通过光谱分解将特征分为低频场景上下文与高频细节，显式建模高频交互以增强融合效果，提升复杂环境下的分割性能。**

- **链接: [http://arxiv.org/pdf/2505.15491v1](http://arxiv.org/pdf/2505.15491v1)**

> **作者:** Ce Zhang; Zifu Wan; Simon Stepputtis; Katia Sycara; Yaqi Xie
>
> **备注:** Accepted by ICIP 2025
>
> **摘要:** Semantic segmentation relying solely on RGB data often struggles in challenging conditions such as low illumination and obscured views, limiting its reliability in critical applications like autonomous driving. To address this, integrating additional thermal radiation data with RGB images demonstrates enhanced performance and robustness. However, how to effectively reconcile the modality discrepancies and fuse the RGB and thermal features remains a well-known challenge. In this work, we address this challenge from a novel spectral perspective. We observe that the multi-modal features can be categorized into two spectral components: low-frequency features that provide broad scene context, including color variations and smooth areas, and high-frequency features that capture modality-specific details such as edges and textures. Inspired by this, we propose the Spectral-aware Global Fusion Network (SGFNet) to effectively enhance and fuse the multi-modal features by explicitly modeling the interactions between the high-frequency, modality-specific features. Our experimental results demonstrate that SGFNet outperforms the state-of-the-art methods on the MFNet and PST900 datasets.
>
---
#### [new 006] Contrastive Learning-Enhanced Trajectory Matching for Small-Scale Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文属于小规模数据集蒸馏任务，旨在解决现有轨迹匹配方法在极少量样本下无法保留语义的问题。提出结合对比学习的增强轨迹匹配方法，通过最大化实例级特征区分生成更丰富多样的合成数据，提升模型在小数据下的性能与图像质量。**

- **链接: [http://arxiv.org/pdf/2505.15267v1](http://arxiv.org/pdf/2505.15267v1)**

> **作者:** Wenmin Li; Shunsuke Sakai; Tatsuhito Hasegawa
>
> **备注:** Under review
>
> **摘要:** Deploying machine learning models in resource-constrained environments, such as edge devices or rapid prototyping scenarios, increasingly demands distillation of large datasets into significantly smaller yet informative synthetic datasets. Current dataset distillation techniques, particularly Trajectory Matching methods, optimize synthetic data so that the model's training trajectory on synthetic samples mirrors that on real data. While demonstrating efficacy on medium-scale synthetic datasets, these methods fail to adequately preserve semantic richness under extreme sample scarcity. To address this limitation, we propose a novel dataset distillation method integrating contrastive learning during image synthesis. By explicitly maximizing instance-level feature discrimination, our approach produces more informative and diverse synthetic samples, even when dataset sizes are significantly constrained. Experimental results demonstrate that incorporating contrastive learning substantially enhances the performance of models trained on very small-scale synthetic datasets. This integration not only guides more effective feature representation but also significantly improves the visual fidelity of the synthesized images. Experimental results demonstrate that our method achieves notable performance improvements over existing distillation techniques, especially in scenarios with extremely limited synthetic data.
>
---
#### [new 007] Visual Perturbation and Adaptive Hard Negative Contrastive Learning for Compositional Reasoning in Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言模型（VLMs）的组合推理（CR）任务，解决现有方法忽视图像负样本及统一处理负样本导致视觉编码器训练不足的问题。提出AHNPL方法，将文本负样本转化为视觉扰动负样本，并采用多模态硬负样本损失和动态边际损失，提升模型对复杂CR任务的区分与对齐能力。**

- **链接: [http://arxiv.org/pdf/2505.15576v1](http://arxiv.org/pdf/2505.15576v1)**

> **作者:** Xin Huang; Ruibin Li; Tong Jia; Wei Zheng; Ya Wang
>
> **备注:** Accepted at the International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** Vision-Language Models (VLMs) are essential for multimodal tasks, especially compositional reasoning (CR) tasks, which require distinguishing fine-grained semantic differences between visual and textual embeddings. However, existing methods primarily fine-tune the model by generating text-based hard negative samples, neglecting the importance of image-based negative samples, which results in insufficient training of the visual encoder and ultimately impacts the overall performance of the model. Moreover, negative samples are typically treated uniformly, without considering their difficulty levels, and the alignment of positive samples is insufficient, which leads to challenges in aligning difficult sample pairs. To address these issues, we propose Adaptive Hard Negative Perturbation Learning (AHNPL). AHNPL translates text-based hard negatives into the visual domain to generate semantically disturbed image-based negatives for training the model, thereby enhancing its overall performance. AHNPL also introduces a contrastive learning approach using a multimodal hard negative loss to improve the model's discrimination of hard negatives within each modality and a dynamic margin loss that adjusts the contrastive margin according to sample difficulty to enhance the distinction of challenging sample pairs. Experiments on three public datasets demonstrate that our method effectively boosts VLMs' performance on complex CR tasks. The source code is available at https://github.com/nynu-BDAI/AHNPL.
>
---
#### [new 008] Programmatic Video Prediction Using Large Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ProgGen方法，利用大视觉语言模型通过神经符号程序生成进行视频预测。任务是根据输入帧预测未来视觉结果，解决真实场景（如自动驾驶）中动态建模与可解释生成难题。工作包括通过程序估计视频状态、预测动态过渡并渲染图像，在PhyWorld和Cart Pole环境超越现有方法，支持反事实推理与可解释生成。**

- **链接: [http://arxiv.org/pdf/2505.14948v1](http://arxiv.org/pdf/2505.14948v1)**

> **作者:** Hao Tang; Kevin Ellis; Suhas Lohit; Michael J. Jones; Moitreya Chatterjee
>
> **摘要:** The task of estimating the world model describing the dynamics of a real world process assumes immense importance for anticipating and preparing for future outcomes. For applications such as video surveillance, robotics applications, autonomous driving, etc. this objective entails synthesizing plausible visual futures, given a few frames of a video to set the visual context. Towards this end, we propose ProgGen, which undertakes the task of video frame prediction by representing the dynamics of the video using a set of neuro-symbolic, human-interpretable set of states (one per frame) by leveraging the inductive biases of Large (Vision) Language Models (LLM/VLM). In particular, ProgGen utilizes LLM/VLM to synthesize programs: (i) to estimate the states of the video, given the visual context (i.e. the frames); (ii) to predict the states corresponding to future time steps by estimating the transition dynamics; (iii) to render the predicted states as visual RGB-frames. Empirical evaluations reveal that our proposed method outperforms competing techniques at the task of video frame prediction in two challenging environments: (i) PhyWorld (ii) Cart Pole. Additionally, ProgGen permits counter-factual reasoning and interpretable video generation attesting to its effectiveness and generalizability for video generation tasks.
>
---
#### [new 009] FastCar: Cache Attentive Replay for Fast Auto-Regressive Video Generation on the Edge
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于边缘设备上的自回归视频生成加速任务。针对视频生成解码阶段因大量token导致的高延迟问题（尤其MLP模块冗余计算），提出FastCar框架：通过Temporal Attention Score（TAS）判定是否复用前帧缓存的MLP输出，并设计基于FPGA的动态资源调度硬件加速器，实现2.1倍解码加速与更高能效，缓解长视频生成的漂移问题。**

- **链接: [http://arxiv.org/pdf/2505.14709v1](http://arxiv.org/pdf/2505.14709v1)**

> **作者:** Xuan Shen; Weize Ma; Yufa Zhou; Enhao Tang; Yanyue Xie; Zhengang Li; Yifan Gong; Quanyi Wang; Henghui Ding; Yiwei Wang; Yanzhi Wang; Pu Zhao; Jun Lin; Jiuxiang Gu
>
> **备注:** Preprint Version
>
> **摘要:** Auto-regressive (AR) models, initially successful in language generation, have recently shown promise in visual generation tasks due to their superior sampling efficiency. Unlike image generation, video generation requires a substantially larger number of tokens to produce coherent temporal frames, resulting in significant overhead during the decoding phase. Our key observations are: (i) MLP modules in the decode phase dominate the inference latency, and (ii) there exists high temporal redundancy in MLP outputs of adjacent frames. In this paper, we propose the \textbf{FastCar} framework to accelerate the decode phase for the AR video generation by exploring the temporal redundancy. The Temporal Attention Score (TAS) is proposed to determine whether to apply the replay strategy (\textit{i.e.}, reusing cached MLP outputs from the previous frame to reduce redundant computations) with detailed theoretical analysis and justification. Also, we develop a hardware accelerator on FPGA with Dynamic Resource Scheduling (DRS) based on TAS to enable better resource utilization and faster inference. Experimental results demonstrate the effectiveness of our method, which outperforms traditional sparse attention approaches with more than 2.1x decoding speedup and higher energy efficiency on the edge. Furthermore, by combining FastCar and sparse attention, FastCar can boost the performance of sparse attention with alleviated drifting, demonstrating our unique advantages for high-resolution and long-duration video generation. Code: https://github.com/shawnricecake/fast-car
>
---
#### [new 010] iPad: Iterative Proposal-centric End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于端到端自动驾驶规划任务。针对现有方法直接生成计划导致效率低和规划意识不足的问题，提出iPad框架：以候选未来计划为中心，通过ProFormer迭代优化BEV特征与规划，并引入轻量级映射和预测辅助任务，实现高效、高性能的自动驾驶，达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.15111v1](http://arxiv.org/pdf/2505.15111v1)**

> **作者:** Ke Guo; Haochen Liu; Xiaojun Wu; Jia Pan; Chen Lv
>
> **摘要:** End-to-end (E2E) autonomous driving systems offer a promising alternative to traditional modular pipelines by reducing information loss and error accumulation, with significant potential to enhance both mobility and safety. However, most existing E2E approaches directly generate plans based on dense bird's-eye view (BEV) grid features, leading to inefficiency and limited planning awareness. To address these limitations, we propose iterative Proposal-centric autonomous driving (iPad), a novel framework that places proposals - a set of candidate future plans - at the center of feature extraction and auxiliary tasks. Central to iPad is ProFormer, a BEV encoder that iteratively refines proposals and their associated features through proposal-anchored attention, effectively fusing multi-view image data. Additionally, we introduce two lightweight, proposal-centric auxiliary tasks - mapping and prediction - that improve planning quality with minimal computational overhead. Extensive experiments on the NAVSIM and CARLA Bench2Drive benchmarks demonstrate that iPad achieves state-of-the-art performance while being significantly more efficient than prior leading methods.
>
---
#### [new 011] Visual Question Answering on Multiple Remote Sensing Image Modalities
- **分类: cs.CV**

- **简介: 该论文提出多模态遥感视觉问答任务，解决单一图像模态信息不足导致场景理解困难的问题。构建了含RGB、多光谱及雷达三种模态的TAMMI数据集，并设计MM-RSVQA模型融合多模态与文本信息，实现65.56%的问答准确率。**

- **链接: [http://arxiv.org/pdf/2505.15401v1](http://arxiv.org/pdf/2505.15401v1)**

> **作者:** Hichem Boussaid; Lucrezia Tosato; Flora Weissgerber; Camille Kurtz; Laurent Wendling; Sylvain Lobry
>
> **备注:** EARTHVISION 2025 8 pages, 1 page of supplementary material, 4 figures
>
> **摘要:** The extraction of visual features is an essential step in Visual Question Answering (VQA). Building a good visual representation of the analyzed scene is indeed one of the essential keys for the system to be able to correctly understand the latter in order to answer complex questions. In many fields such as remote sensing, the visual feature extraction step could benefit significantly from leveraging different image modalities carrying complementary spectral, spatial and contextual information. In this work, we propose to add multiple image modalities to VQA in the particular context of remote sensing, leading to a novel task for the computer vision community. To this end, we introduce a new VQA dataset, named TAMMI (Text and Multi-Modal Imagery) with diverse questions on scenes described by three different modalities (very high resolution RGB, multi-spectral imaging data and synthetic aperture radar). Thanks to an automated pipeline, this dataset can be easily extended according to experimental needs. We also propose the MM-RSVQA (Multi-modal Multi-resolution Remote Sensing Visual Question Answering) model, based on VisualBERT, a vision-language transformer, to effectively combine the multiple image modalities and text through a trainable fusion process. A preliminary experimental study shows promising results of our methodology on this challenging dataset, with an accuracy of 65.56% on the targeted VQA task. This pioneering work paves the way for the community to a new multi-modal multi-resolution VQA task that can be applied in other imaging domains (such as medical imaging) where multi-modality can enrich the visual representation of a scene. The dataset and code are available at https://tammi.sylvainlobry.com/.
>
---
#### [new 012] MSVIT: Improving Spiking Vision Transformer Using Multi-scale Attention Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于脉冲神经网络与视觉Transformer结合的任务，旨在解决SNN-Transformer架构在多尺度特征提取上的性能瓶颈。提出MSVIT模型，通过多尺度脉冲注意力（MSSA）增强特征表达能力，并验证其在多个数据集上的优越性，成为SNN-Transformer领域的最新方法。**

- **链接: [http://arxiv.org/pdf/2505.14719v1](http://arxiv.org/pdf/2505.14719v1)**

> **作者:** Wei Hua; Chenlin Zhou; Jibin Wu; Yansong Chua; Yangyang Shu
>
> **摘要:** The combination of Spiking Neural Networks(SNNs) with Vision Transformer architectures has attracted significant attention due to the great potential for energy-efficient and high-performance computing paradigms. However, a substantial performance gap still exists between SNN-based and ANN-based transformer architectures. While existing methods propose spiking self-attention mechanisms that are successfully combined with SNNs, the overall architectures proposed by these methods suffer from a bottleneck in effectively extracting features from different image scales. In this paper, we address this issue and propose MSVIT, a novel spike-driven Transformer architecture, which firstly uses multi-scale spiking attention (MSSA) to enrich the capability of spiking attention blocks. We validate our approach across various main data sets. The experimental results show that MSVIT outperforms existing SNN-based models, positioning itself as a state-of-the-art solution among SNN-transformer architectures. The codes are available at https://github.com/Nanhu-AI-Lab/MSViT.
>
---
#### [new 013] VET-DINO: Learning Anatomical Understanding Through Multi-View Distillation in Veterinary Imaging
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出VET-DINO框架，针对兽医影像中标签数据稀缺问题，利用多视角真实影像进行自监督学习，通过学习视角不变的解剖结构和3D空间理解，提升模型性能。在500万张犬类X光片上验证，其效果优于传统合成增强方法，建立医疗影像新范式。**

- **链接: [http://arxiv.org/pdf/2505.15248v1](http://arxiv.org/pdf/2505.15248v1)**

> **作者:** Andre Dourson; Kylie Taylor; Xiaoli Qiao; Michael Fitzke
>
> **摘要:** Self-supervised learning has emerged as a powerful paradigm for training deep neural networks, particularly in medical imaging where labeled data is scarce. While current approaches typically rely on synthetic augmentations of single images, we propose VET-DINO, a framework that leverages a unique characteristic of medical imaging: the availability of multiple standardized views from the same study. Using a series of clinical veterinary radiographs from the same patient study, we enable models to learn view-invariant anatomical structures and develop an implied 3D understanding from 2D projections. We demonstrate our approach on a dataset of 5 million veterinary radiographs from 668,000 canine studies. Through extensive experimentation, including view synthesis and downstream task performance, we show that learning from real multi-view pairs leads to superior anatomical understanding compared to purely synthetic augmentations. VET-DINO achieves state-of-the-art performance on various veterinary imaging tasks. Our work establishes a new paradigm for self-supervised learning in medical imaging that leverages domain-specific properties rather than merely adapting natural image techniques.
>
---
#### [new 014] VP Lab: a PEFT-Enabled Visual Prompting Laboratory for Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文聚焦语义分割任务，针对预训练模型在特定领域视觉适配不足的问题，提出VP Lab框架。通过集成E-PEFT参数高效微调技术，结合视觉提示方法，仅用5张标注图像使SAM模型mIoU提升50%，实现交互式快速部署。**

- **链接: [http://arxiv.org/pdf/2505.15592v1](http://arxiv.org/pdf/2505.15592v1)**

> **作者:** Niccolo Avogaro; Thomas Frick; Yagmur G. Cinar; Daniel Caraballo; Cezary Skura; Filip M. Janicki; Piotr Kluska; Brown Ebouky; Nicola Farronato; Florian Scheidegger; Cristiano Malossi; Konrad Schindler; Andrea Bartezzaghi; Roy Assaf; Mattia Rigotti
>
> **摘要:** Large-scale pretrained vision backbones have transformed computer vision by providing powerful feature extractors that enable various downstream tasks, including training-free approaches like visual prompting for semantic segmentation. Despite their success in generic scenarios, these models often fall short when applied to specialized technical domains where the visual features differ significantly from their training distribution. To bridge this gap, we introduce VP Lab, a comprehensive iterative framework that enhances visual prompting for robust segmentation model development. At the core of VP Lab lies E-PEFT, a novel ensemble of parameter-efficient fine-tuning techniques specifically designed to adapt our visual prompting pipeline to specific domains in a manner that is both parameter- and data-efficient. Our approach not only surpasses the state-of-the-art in parameter-efficient fine-tuning for the Segment Anything Model (SAM), but also facilitates an interactive, near-real-time loop, allowing users to observe progressively improving results as they experiment within the framework. By integrating E-PEFT with visual prompting, we demonstrate a remarkable 50\% increase in semantic segmentation mIoU performance across various technical datasets using only 5 validated images, establishing a new paradigm for fast, efficient, and interactive model deployment in new, challenging domains. This work comes in the form of a demonstration.
>
---
#### [new 015] CineTechBench: A Benchmark for Cinematographic Technique Understanding and Generation
- **分类: cs.CV**

- **简介: 该论文提出CineTechBench基准，填补电影技术理解和生成评估的数据空白。针对现有模型因缺乏标注数据难以掌握电影拍摄技巧的问题，构建专家标注数据集（含7个维度、600+图像及120+视频），设计问答与生成任务，评估15+语言模型和5+视频生成模型，揭示其能力局限并指明优化方向。**

- **链接: [http://arxiv.org/pdf/2505.15145v1](http://arxiv.org/pdf/2505.15145v1)**

> **作者:** Xinran Wang; Songyu Xu; Xiangxuan Shan; Yuxuan Zhang; Muxi Diao; Xueyan Duan; Yanhua Huang; Kongming Liang; Zhanyu Ma
>
> **备注:** Under review
>
> **摘要:** Cinematography is a cornerstone of film production and appreciation, shaping mood, emotion, and narrative through visual elements such as camera movement, shot composition, and lighting. Despite recent progress in multimodal large language models (MLLMs) and video generation models, the capacity of current models to grasp and reproduce cinematographic techniques remains largely uncharted, hindered by the scarcity of expert-annotated data. To bridge this gap, we present CineTechBench, a pioneering benchmark founded on precise, manual annotation by seasoned cinematography experts across key cinematography dimensions. Our benchmark covers seven essential aspects-shot scale, shot angle, composition, camera movement, lighting, color, and focal length-and includes over 600 annotated movie images and 120 movie clips with clear cinematographic techniques. For the understanding task, we design question answer pairs and annotated descriptions to assess MLLMs' ability to interpret and explain cinematographic techniques. For the generation task, we assess advanced video generation models on their capacity to reconstruct cinema-quality camera movements given conditions such as textual prompts or keyframes. We conduct a large-scale evaluation on 15+ MLLMs and 5+ video generation models. Our results offer insights into the limitations of current models and future directions for cinematography understanding and generation in automatically film production and appreciation. The code and benchmark can be accessed at https://github.com/PRIS-CV/CineTechBench.
>
---
#### [new 016] DiffProb: Data Pruning for Face Recognition
- **分类: cs.CV**

- **简介: 该论文提出DiffProb——首个面向人脸识别的数据裁剪方法，旨在解决大规模数据集的高计算/存储成本及隐私问题。通过分析样本预测概率，去除冗余样本并清理错误标签，实现在CASIA-WebFace上裁剪50%数据同时保持甚至提升识别精度，适用于多种网络架构和损失函数，降低训练资源需求。**

- **链接: [http://arxiv.org/pdf/2505.15272v1](http://arxiv.org/pdf/2505.15272v1)**

> **作者:** Eduarda Caldeira; Jan Niklas Kolf; Naser Damer; Fadi Boutros
>
> **备注:** Accepted at IEEE International Conference on Automatic Face and Gesture Recognition (FG) 2025
>
> **摘要:** Face recognition models have made substantial progress due to advances in deep learning and the availability of large-scale datasets. However, reliance on massive annotated datasets introduces challenges related to training computational cost and data storage, as well as potential privacy concerns regarding managing large face datasets. This paper presents DiffProb, the first data pruning approach for the application of face recognition. DiffProb assesses the prediction probabilities of training samples within each identity and prunes the ones with identical or close prediction probability values, as they are likely reinforcing the same decision boundaries, and thus contribute minimally with new information. We further enhance this process with an auxiliary cleaning mechanism to eliminate mislabeled and label-flipped samples, boosting data quality with minimal loss. Extensive experiments on CASIA-WebFace with different pruning ratios and multiple benchmarks, including LFW, CFP-FP, and IJB-C, demonstrate that DiffProb can prune up to 50% of the dataset while maintaining or even, in some settings, improving the verification accuracies. Additionally, we demonstrate DiffProb's robustness across different architectures and loss functions. Our method significantly reduces training cost and data volume, enabling efficient face recognition training and reducing the reliance on massive datasets and their demanding management.
>
---
#### [new 017] Prompt Tuning Vision Language Models with Margin Regularizer for Few-Shot Learning under Distribution Shifts
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究分布偏移下视觉语言模型的小样本学习任务。针对目标数据与预训练数据分布差异大导致的过拟合和泛化不足问题，提出PromptMargin方法：结合选择性数据增强扩充样本，采用多模态边缘正则化提升类别区分，实验证明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15506v1](http://arxiv.org/pdf/2505.15506v1)**

> **作者:** Debarshi Brahma; Anuska Roy; Soma Biswas
>
> **备注:** Published in TMLR (2025)
>
> **摘要:** Recently, Vision-Language foundation models like CLIP and ALIGN, which are pre-trained on large-scale data have shown remarkable zero-shot generalization to diverse datasets with different classes and even domains. In this work, we take a step further and analyze whether these models can be adapted to target datasets having very different distributions and classes compared to what these models have been trained on, using only a few labeled examples from the target dataset. In such scenarios, finetuning large pretrained models is challenging due to problems of overfitting as well as loss of generalization, and has not been well explored in prior literature. Since, the pre-training data of such models are unavailable, it is difficult to comprehend the performance on various downstream datasets. First, we try to answer the question: Given a target dataset with a few labelled examples, can we estimate whether further fine-tuning can enhance the performance compared to zero-shot evaluation? by analyzing the common vision-language embedding space. Based on the analysis, we propose a novel prompt-tuning method, PromptMargin for adapting such large-scale VLMs directly on the few target samples. PromptMargin effectively tunes the text as well as visual prompts for this task, and has two main modules: 1) Firstly, we use a selective augmentation strategy to complement the few training samples in each task; 2) Additionally, to ensure robust training in the presence of unfamiliar class names, we increase the inter-class margin for improved class discrimination using a novel Multimodal Margin Regularizer. Extensive experiments and analysis across fifteen target benchmark datasets, with varying degrees of distribution shifts from natural images, shows the effectiveness of the proposed framework over the existing state-of-the-art approaches applied to this setting. github.com/debarshigit/PromptMargin.
>
---
#### [new 018] Beyond Modality Collapse: Representations Blending for Multimodal Dataset Distillation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态数据集蒸馏（MDD）任务，旨在解决现有方法因模态崩溃导致的跨模态表示分布失衡问题。提出RepBlend框架，通过表示融合缓解过强的跨模态监督，并采用对称投影轨迹匹配平衡模态优化，提升蒸馏数据的多样性和跨模态对齐效果。**

- **链接: [http://arxiv.org/pdf/2505.14705v1](http://arxiv.org/pdf/2505.14705v1)**

> **作者:** Xin Zhang; Ziruo Zhang; Jiawei Du; Zuozhu Liu; Joey Tianyi Zhou
>
> **摘要:** Multimodal Dataset Distillation (MDD) seeks to condense large-scale image-text datasets into compact surrogates while retaining their effectiveness for cross-modal learning. Despite recent progress, existing MDD approaches often suffer from \textit{\textbf{Modality Collapse}}, characterized by over-concentrated intra-modal representations and enlarged distributional gap across modalities. In this paper, at the first time, we identify this issue as stemming from a fundamental conflict between the over-compression behavior inherent in dataset distillation and the cross-modal supervision imposed by contrastive objectives. To alleviate modality collapse, we introduce \textbf{RepBlend}, a novel MDD framework that weakens overdominant cross-modal supervision via representation blending, thereby significantly enhancing intra-modal diversity. Additionally, we observe that current MDD methods impose asymmetric supervision across modalities, resulting in biased optimization. To address this, we propose symmetric projection trajectory matching, which synchronizes the optimization dynamics using modality-specific projection heads, thereby promoting balanced supervision and enhancing cross-modal alignment. Experiments on Flickr-30K and MS-COCO show that RepBlend consistently outperforms prior state-of-the-art MDD methods, achieving significant gains in retrieval performance (e.g., +9.4 IR@10, +6.3 TR@10 under the 100-pair setting) and offering up to 6.7$\times$ distillation speedup.
>
---
#### [new 019] CEBSNet: Change-Excited and Background-Suppressed Network with Temporal Dependency Modeling for Bitemporal Change Detection
- **分类: cs.CV**

- **简介: 该论文提出CEBSNet用于双时相变化检测任务，解决现有方法忽视时间依赖及忽略细微变化的问题。通过Channel Swap Module建模时间依赖、减少噪声；Feature Excitation and Suppression Module捕捉显著与细微变化；并设计Pyramid-Aware Spatial-Channel Attention提升多尺度变化检测能力。**

- **链接: [http://arxiv.org/pdf/2505.15322v1](http://arxiv.org/pdf/2505.15322v1)**

> **作者:** Qi'ao Xu; Yan Xing; Jiali Hu; Yunan Jia; Rui Huang
>
> **摘要:** Change detection, a critical task in remote sensing and computer vision, aims to identify pixel-level differences between image pairs captured at the same geographic area but different times. It faces numerous challenges such as illumination variation, seasonal changes, background interference, and shooting angles, especially with a large time gap between images. While current methods have advanced, they often overlook temporal dependencies and overemphasize prominent changes while ignoring subtle but equally important changes. To address these limitations, we introduce \textbf{CEBSNet}, a novel change-excited and background-suppressed network with temporal dependency modeling for change detection. During the feature extraction, we utilize a simple Channel Swap Module (CSM) to model temporal dependency, reducing differences and noise. The Feature Excitation and Suppression Module (FESM) is developed to capture both obvious and subtle changes, maintaining the integrity of change regions. Additionally, we design a Pyramid-Aware Spatial-Channel Attention module (PASCA) to enhance the ability to detect change regions at different sizes and focus on critical regions. We conduct extensive experiments on three common street view datasets and two remote sensing datasets, and our method achieves the state-of-the-art performance.
>
---
#### [new 020] Unified Cross-Modal Attention-Mixer Based Structural-Functional Connectomics Fusion for Neuropsychiatric Disorder Diagnosis
- **分类: cs.CV**

- **简介: 该论文提出ConneX方法，针对神经精神疾病诊断中结构与功能连接组数据融合不足的问题，通过模态专用GNN提取特征，结合跨模态注意力与MLP-Mixer网络融合特征，提升诊断性能。实验显示其在临床数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2505.15139v1](http://arxiv.org/pdf/2505.15139v1)**

> **作者:** Badhan Mazumder; Lei Wu; Vince D. Calhoun; Dong Hye Ye
>
> **备注:** Accepted at 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) 2025
>
> **摘要:** Gaining insights into the structural and functional mechanisms of the brain has been a longstanding focus in neuroscience research, particularly in the context of understanding and treating neuropsychiatric disorders such as Schizophrenia (SZ). Nevertheless, most of the traditional multimodal deep learning approaches fail to fully leverage the complementary characteristics of structural and functional connectomics data to enhance diagnostic performance. To address this issue, we proposed ConneX, a multimodal fusion method that integrates cross-attention mechanism and multilayer perceptron (MLP)-Mixer for refined feature fusion. Modality-specific backbone graph neural networks (GNNs) were firstly employed to obtain feature representation for each modality. A unified cross-modal attention network was then introduced to fuse these embeddings by capturing intra- and inter-modal interactions, while MLP-Mixer layers refined global and local features, leveraging higher-order dependencies for end-to-end classification with a multi-head joint loss. Extensive evaluations demonstrated improved performance on two distinct clinical datasets, highlighting the robustness of our proposed framework.
>
---
#### [new 021] EVA: Expressive Virtual Avatars from Multi-view Videos
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于虚拟人类形象生成与控制任务，旨在解决现有方法无法独立控制面部表情、身体动作的问题。提出EVA框架，通过两层模型（几何模板层和3D高斯外观层）及分离优化算法，实现多视角视频驱动下高保真、可独立控制的全身虚拟形象，提升渲染质量和表达能力。**

- **链接: [http://arxiv.org/pdf/2505.15385v1](http://arxiv.org/pdf/2505.15385v1)**

> **作者:** Hendrik Junkawitsch; Guoxing Sun; Heming Zhu; Christian Theobalt; Marc Habermann
>
> **备注:** Accepted at SIGGRAPH 2025 Conference Track, Project page: https://vcai.mpi-inf.mpg.de/projects/EVA/
>
> **摘要:** With recent advancements in neural rendering and motion capture algorithms, remarkable progress has been made in photorealistic human avatar modeling, unlocking immense potential for applications in virtual reality, augmented reality, remote communication, and industries such as gaming, film, and medicine. However, existing methods fail to provide complete, faithful, and expressive control over human avatars due to their entangled representation of facial expressions and body movements. In this work, we introduce Expressive Virtual Avatars (EVA), an actor-specific, fully controllable, and expressive human avatar framework that achieves high-fidelity, lifelike renderings in real time while enabling independent control of facial expressions, body movements, and hand gestures. Specifically, our approach designs the human avatar as a two-layer model: an expressive template geometry layer and a 3D Gaussian appearance layer. First, we present an expressive template tracking algorithm that leverages coarse-to-fine optimization to accurately recover body motions, facial expressions, and non-rigid deformation parameters from multi-view videos. Next, we propose a novel decoupled 3D Gaussian appearance model designed to effectively disentangle body and facial appearance. Unlike unified Gaussian estimation approaches, our method employs two specialized and independent modules to model the body and face separately. Experimental results demonstrate that EVA surpasses state-of-the-art methods in terms of rendering quality and expressiveness, validating its effectiveness in creating full-body avatars. This work represents a significant advancement towards fully drivable digital human models, enabling the creation of lifelike digital avatars that faithfully replicate human geometry and appearance.
>
---
#### [new 022] The Devil is in Fine-tuning and Long-tailed Problems:A New Benchmark for Scene Text Detection
- **分类: cs.CV**

- **简介: 该论文属于场景文本检测任务，旨在解决现实场景中模型性能下降问题。针对微调导致的领域泛化能力不足和长尾分布下复杂文本检测困难，提出联合数据集学习（JDL）缓解微调差距，构建长尾基准（LTB）并引入自监督方法MAEDet作为基线。**

- **链接: [http://arxiv.org/pdf/2505.15649v1](http://arxiv.org/pdf/2505.15649v1)**

> **作者:** Tianjiao Cao; Jiahao Lyu; Weichao Zeng; Weimin Mu; Yu Zhou
>
> **备注:** Accepted by IJCAI2025
>
> **摘要:** Scene text detection has seen the emergence of high-performing methods that excel on academic benchmarks. However, these detectors often fail to replicate such success in real-world scenarios. We uncover two key factors contributing to this discrepancy through extensive experiments. First, a \textit{Fine-tuning Gap}, where models leverage \textit{Dataset-Specific Optimization} (DSO) paradigm for one domain at the cost of reduced effectiveness in others, leads to inflated performances on academic benchmarks. Second, the suboptimal performance in practical settings is primarily attributed to the long-tailed distribution of texts, where detectors struggle with rare and complex categories as artistic or overlapped text. Given that the DSO paradigm might undermine the generalization ability of models, we advocate for a \textit{Joint-Dataset Learning} (JDL) protocol to alleviate the Fine-tuning Gap. Additionally, an error analysis is conducted to identify three major categories and 13 subcategories of challenges in long-tailed scene text, upon which we propose a Long-Tailed Benchmark (LTB). LTB facilitates a comprehensive evaluation of ability to handle a diverse range of long-tailed challenges. We further introduce MAEDet, a self-supervised learning-based method, as a strong baseline for LTB. The code is available at https://github.com/pd162/LTB.
>
---
#### [new 023] DeepKD: A Deeply Decoupled and Denoised Knowledge Distillation Trainer
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于知识蒸馏任务，旨在解决目标类与非目标类知识流冲突及低置信度暗知识噪声问题。提出DeepKD框架，通过双层解耦（独立动量更新器优化不同梯度）和动态top-k去噪（过滤低置信logits）提升知识转移效果，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.15133v1](http://arxiv.org/pdf/2505.15133v1)**

> **作者:** Haiduo Huang; Jiangcheng Song; Yadong Zhang; Pengju Ren
>
> **摘要:** Recent advances in knowledge distillation have emphasized the importance of decoupling different knowledge components. While existing methods utilize momentum mechanisms to separate task-oriented and distillation gradients, they overlook the inherent conflict between target-class and non-target-class knowledge flows. Furthermore, low-confidence dark knowledge in non-target classes introduces noisy signals that hinder effective knowledge transfer. To address these limitations, we propose DeepKD, a novel training framework that integrates dual-level decoupling with adaptive denoising. First, through theoretical analysis of gradient signal-to-noise ratio (GSNR) characteristics in task-oriented and non-task-oriented knowledge distillation, we design independent momentum updaters for each component to prevent mutual interference. We observe that the optimal momentum coefficients for task-oriented gradient (TOG), target-class gradient (TCG), and non-target-class gradient (NCG) should be positively related to their GSNR. Second, we introduce a dynamic top-k mask (DTM) mechanism that gradually increases K from a small initial value to incorporate more non-target classes as training progresses, following curriculum learning principles. The DTM jointly filters low-confidence logits from both teacher and student models, effectively purifying dark knowledge during early training. Extensive experiments on CIFAR-100, ImageNet, and MS-COCO demonstrate DeepKD's effectiveness. Our code is available at https://github.com/haiduo/DeepKD.
>
---
#### [new 024] My Face Is Mine, Not Yours: Facial Protection Against Diffusion Model Face Swapping
- **分类: cs.CV**

- **简介: 该论文属于主动防御任务，针对扩散模型换脸技术的滥用问题，提出基于对抗攻击的防护方法。现有方法多针对传统生成模型且依赖特定架构，防护效果有限。本文提出区域特定扰动策略，无需知道模型细节，通过预置对抗噪声阻止面部图像被扩散模型篡改，解决全局扰动与模型多样性的挑战。**

- **链接: [http://arxiv.org/pdf/2505.15336v1](http://arxiv.org/pdf/2505.15336v1)**

> **作者:** Hon Ming Yam; Zhongliang Guo; Chun Pong Lau
>
> **摘要:** The proliferation of diffusion-based deepfake technologies poses significant risks for unauthorized and unethical facial image manipulation. While traditional countermeasures have primarily focused on passive detection methods, this paper introduces a novel proactive defense strategy through adversarial attacks that preemptively protect facial images from being exploited by diffusion-based deepfake systems. Existing adversarial protection methods predominantly target conventional generative architectures (GANs, AEs, VAEs) and fail to address the unique challenges presented by diffusion models, which have become the predominant framework for high-quality facial deepfakes. Current diffusion-specific adversarial approaches are limited by their reliance on specific model architectures and weights, rendering them ineffective against the diverse landscape of diffusion-based deepfake implementations. Additionally, they typically employ global perturbation strategies that inadequately address the region-specific nature of facial manipulation in deepfakes.
>
---
#### [new 025] CrypticBio: A Large Multimodal Dataset for Visually Confusing Biodiversity
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CrypticBio，首个大型多模态数据集，解决视觉相似物种分类难题。针对现有数据集规模小、单分类且缺乏多模态的问题，该研究整合了67K物种的166M图像及地理、时间等信息，提供开源工具与基准测试，证明地理上下文对AI模型的重要性。**

- **链接: [http://arxiv.org/pdf/2505.14707v1](http://arxiv.org/pdf/2505.14707v1)**

> **作者:** Georgiana Manolache; Gerard Schouten; Joaquin Vanschoren
>
> **备注:** We present CrypticBio, the largest publicly available multimodal dataset of visually confusing species, specifically curated to support the development of AI models for biodiversity identification using images, language and spatiotemporal data
>
> **摘要:** We present CrypticBio, the largest publicly available multimodal dataset of visually confusing species, specifically curated to support the development of AI models in the context of biodiversity applications. Visually confusing or cryptic species are groups of two or more taxa that are nearly indistinguishable based on visual characteristics alone. While much existing work addresses taxonomic identification in a broad sense, datasets that directly address the morphological confusion of cryptic species are small, manually curated, and target only a single taxon. Thus, the challenge of identifying such subtle differences in a wide range of taxa remains unaddressed. Curated from real-world trends in species misidentification among community annotators of iNaturalist, CrypticBio contains 52K unique cryptic groups spanning 67K species, represented in 166 million images. Rich research-grade image annotations--including scientific, multicultural, and multilingual species terminology, hierarchical taxonomy, spatiotemporal context, and associated cryptic groups--address multimodal AI in biodiversity research. For easy dataset curation, we provide an open-source pipeline CrypticBio-Curate. The multimodal nature of the dataset beyond vision-language arises from the integration of geographical and temporal data as complementary cues to identifying cryptic species. To highlight the importance of the dataset, we benchmark a suite of state-of-the-art foundation models across CrypticBio subsets of common, unseen, endangered, and invasive species, and demonstrate the substantial impact of geographical context on vision-language zero-shot learning for cryptic species. By introducing CrypticBio, we aim to catalyze progress toward real-world-ready biodiversity AI models capable of handling the nuanced challenges of species ambiguity.
>
---
#### [new 026] Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，解决领域依赖和实时性问题。提出Flashback方法，分离线构建伪场景记忆（LLM生成）与在线相似度匹配两阶段，实现零样本、实时检测，性能显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15205v1](http://arxiv.org/pdf/2505.15205v1)**

> **作者:** Hyogun Lee; Haksub Kim; Ig-Jae Kim; Yonghun Choi
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Video Anomaly Detection (VAD) automatically identifies anomalous events from video, mitigating the need for human operators in large-scale surveillance deployments. However, three fundamental obstacles hinder real-world adoption: domain dependency and real-time constraints -- requiring near-instantaneous processing of incoming video. To this end, we propose Flashback, a zero-shot and real-time video anomaly detection paradigm. Inspired by the human cognitive mechanism of instantly judging anomalies and reasoning in current scenes based on past experience, Flashback operates in two stages: Recall and Respond. In the offline recall stage, an off-the-shelf LLM builds a pseudo-scene memory of both normal and anomalous captions without any reliance on real anomaly data. In the online respond stage, incoming video segments are embedded and matched against this memory via similarity search. By eliminating all LLM calls at inference time, Flashback delivers real-time VAD even on a consumer-grade GPU. On two large datasets from real-world surveillance scenarios, UCF-Crime and XD-Violence, we achieve 87.3 AUC (+7.0 pp) and 75.1 AP (+13.1 pp), respectively, outperforming prior zero-shot VAD methods by large margins.
>
---
#### [new 027] Geometrically Regularized Transfer Learning with On-Manifold and Off-Manifold Perturbation
- **分类: cs.CV**

- **简介: 该论文属于迁移学习任务，旨在解决领域偏移导致的源域与目标域数据流形差异问题。提出MAADA框架，通过分解对抗扰动为流形内/外成分，结合几何对齐损失，提升跨域泛化与模型鲁棒性，实验显示其在无监督和少样本场景下表现最优。**

- **链接: [http://arxiv.org/pdf/2505.15191v1](http://arxiv.org/pdf/2505.15191v1)**

> **作者:** Hana Satou; Alan Mitkiy; F Monkey
>
> **摘要:** Transfer learning under domain shift remains a fundamental challenge due to the divergence between source and target data manifolds. In this paper, we propose MAADA (Manifold-Aware Adversarial Data Augmentation), a novel framework that decomposes adversarial perturbations into on-manifold and off-manifold components to simultaneously capture semantic variation and model brittleness. We theoretically demonstrate that enforcing on-manifold consistency reduces hypothesis complexity and improves generalization, while off-manifold regularization smooths decision boundaries in low-density regions. Moreover, we introduce a geometry-aware alignment loss that minimizes geodesic discrepancy between source and target manifolds. Experiments on DomainNet, VisDA, and Office-Home show that MAADA consistently outperforms existing adversarial and adaptation methods in both unsupervised and few-shot settings, demonstrating superior structural robustness and cross-domain generalization.
>
---
#### [new 028] Open-Set Semi-Supervised Learning for Long-Tailed Medical Datasets
- **分类: cs.CV**

- **简介: 该论文提出一种针对长尾医学数据的开放集半监督学习方法，解决类别不平衡及未见类别识别问题。通过特征正则化与分类器归一化技术，提升模型在闭集和开集场景下的分类性能，并在多个医学数据集上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2505.14846v1](http://arxiv.org/pdf/2505.14846v1)**

> **作者:** Daniya Najiha A. Kareem; Jean Lahoud; Mustansar Fiaz; Amandeep Kumar; Hisham Cholakkal
>
> **摘要:** Many practical medical imaging scenarios include categories that are under-represented but still crucial. The relevance of image recognition models to real-world applications lies in their ability to generalize to these rare classes as well as unseen classes. Real-world generalization requires taking into account the various complexities that can be encountered in the real-world. First, training data is highly imbalanced, which may lead to model exhibiting bias toward the more frequently represented classes. Moreover, real-world data may contain unseen classes that need to be identified, and model performance is affected by the data scarcity. While medical image recognition has been extensively addressed in the literature, current methods do not take into account all the intricacies in the real-world scenarios. To this end, we propose an open-set learning method for highly imbalanced medical datasets using a semi-supervised approach. Understanding the adverse impact of long-tail distribution at the inherent model characteristics, we implement a regularization strategy at the feature level complemented by a classifier normalization technique. We conduct extensive experiments on the publicly available datasets, ISIC2018, ISIC2019, and TissueMNIST with various numbers of labelled samples. Our analysis shows that addressing the impact of long-tail data in classification significantly improves the overall performance of the network in terms of closed-set and open-set accuracies on all datasets. Our code and trained models will be made publicly available at https://github.com/Daniyanaj/OpenLTR.
>
---
#### [new 029] Harnessing Caption Detailness for Data-Efficient Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成（T2I）任务，旨在解决现有方法依赖简单指标（如标题长度）选择训练数据导致效果不佳的问题。提出基于图像覆盖度（ICR）和物体描述详细度（AOD）的新指标，通过实验表明，仅用20%高指标数据训练的模型性能优于全量数据或长度选择方法，凸显了详细度评估在数据选择中的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.15172v1](http://arxiv.org/pdf/2505.15172v1)**

> **作者:** Xinran Wang; Muxi Diao; Yuanzhi Liu; Chunyu Wang; Kongming Liang; Zhanyu Ma; Jun Guo
>
> **摘要:** Training text-to-image (T2I) models with detailed captions can significantly improve their generation quality. Existing methods often rely on simplistic metrics like caption length to represent the detailness of the caption in the T2I training set. In this paper, we propose a new metric to estimate caption detailness based on two aspects: image coverage rate (ICR), which evaluates whether the caption covers all regions/objects in the image, and average object detailness (AOD), which quantifies the detailness of each object's description. Through experiments on the COCO dataset using ShareGPT4V captions, we demonstrate that T2I models trained on high-ICR and -AOD captions achieve superior performance on DPG and other benchmarks. Notably, our metric enables more effective data selection-training on only 20% of full data surpasses both full-dataset training and length-based selection method, improving alignment and reconstruction ability. These findings highlight the critical role of detail-aware metrics over length-based heuristics in caption selection for T2I tasks.
>
---
#### [new 030] Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于对抗鲁棒性研究任务，旨在发现导致大型视觉语言模型（LVLMs）失效的敏感语义概念。通过结合LLM和文生图模型，提出语义进化框架：随机初始化语义经LLM交叉/变异生成图像描述，转化为视觉输入测试LVLMs，根据模型表现反馈迭代优化，最终定位易引发错误的敏感概念（如天气、材质等）。实验验证方法有效性并揭示LVLMs盲点。**

- **链接: [http://arxiv.org/pdf/2505.15265v1](http://arxiv.org/pdf/2505.15265v1)**

> **作者:** Zihao Pan; Yu Tong; Weibin Wu; Jingyi Wang; Lifeng Chen; Zhe Zhao; Jiajia Wei; Yitong Qiao; Zibin Zheng
>
> **摘要:** Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research.
>
---
#### [new 031] DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成加速任务，旨在解决扩散模型（DiTs）因高计算成本（尤其注意力机制占80%延迟）导致生成速度慢的问题。提出DraftAttention框架，通过低分辨率注意力图引导动态稀疏注意力计算，在GPU上优化执行效率，实现1.75倍加速且保持生成质量。**

- **链接: [http://arxiv.org/pdf/2505.14708v1](http://arxiv.org/pdf/2505.14708v1)**

> **作者:** Xuan Shen; Chenxia Han; Yufa Zhou; Yanyue Xie; Yifan Gong; Quanyi Wang; Yiwei Wang; Yanzhi Wang; Pu Zhao; Jiuxiang Gu
>
> **备注:** Preprint Version
>
> **摘要:** Diffusion transformer-based video generation models (DiTs) have recently attracted widespread attention for their excellent generation quality. However, their computational cost remains a major bottleneck-attention alone accounts for over 80% of total latency, and generating just 8 seconds of 720p video takes tens of minutes-posing serious challenges to practical application and scalability. To address this, we propose the DraftAttention, a training-free framework for the acceleration of video diffusion transformers with dynamic sparse attention on GPUs. We apply down-sampling to each feature map across frames in the compressed latent space, enabling a higher-level receptive field over the latent composed of hundreds of thousands of tokens. The low-resolution draft attention map, derived from draft query and key, exposes redundancy both spatially within each feature map and temporally across frames. We reorder the query, key, and value based on the draft attention map to guide the sparse attention computation in full resolution, and subsequently restore their original order after the attention computation. This reordering enables structured sparsity that aligns with hardware-optimized execution. Our theoretical analysis demonstrates that the low-resolution draft attention closely approximates the full attention, providing reliable guidance for constructing accurate sparse attention. Experimental results show that our method outperforms existing sparse attention approaches in video generation quality and achieves up to 1.75x end-to-end speedup on GPUs. Code: https://github.com/shawnricecake/draft-attention
>
---
#### [new 032] Oral Imaging for Malocclusion Issues Assessments: OMNI Dataset, Deep Learning Baselines and Benchmarking
- **分类: cs.CV**

- **简介: 该论文属于牙齿错颌自动诊断任务，旨在解决缺乏大规模标注数据集限制正畸自动化诊断的问题。研究团队构建了含4166张多视角图像的OMNI数据集，并采用CNN、Transformer、GNN等方法验证其有效性，为领域提供新基准与公开资源。**

- **链接: [http://arxiv.org/pdf/2505.15637v1](http://arxiv.org/pdf/2505.15637v1)**

> **作者:** Pujun Xue; Junyi Ge; Xiaotong Jiang; Siyang Song; Zijian Wu; Yupeng Huo; Weicheng Xie; Linlin Shen; Xiaoqin Zhou; Xiaofeng Liu; Min Gu
>
> **摘要:** Malocclusion is a major challenge in orthodontics, and its complex presentation and diverse clinical manifestations make accurate localization and diagnosis particularly important. Currently, one of the major shortcomings facing the field of dental image analysis is the lack of large-scale, accurately labeled datasets dedicated to malocclusion issues, which limits the development of automated diagnostics in the field of dentistry and leads to a lack of diagnostic accuracy and efficiency in clinical practice. Therefore, in this study, we propose the Oral and Maxillofacial Natural Images (OMNI) dataset, a novel and comprehensive dental image dataset aimed at advancing the study of analyzing dental images for issues of malocclusion. Specifically, the dataset contains 4166 multi-view images with 384 participants in data collection and annotated by professional dentists. In addition, we performed a comprehensive validation of the created OMNI dataset, including three CNN-based methods, two Transformer-based methods, and one GNN-based method, and conducted automated diagnostic experiments for malocclusion issues. The experimental results show that the OMNI dataset can facilitate the automated diagnosis research of malocclusion issues and provide a new benchmark for the research in this field. Our OMNI dataset and baseline code are publicly available at https://github.com/RoundFaceJ/OMNI.
>
---
#### [new 033] Clapper: Compact Learning and Video Representation in VLMs
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，针对视觉语言模型（VLMs）在处理长短视频时的性能瓶颈，提出Clapper方法。其通过slow-fast策略与TimePerceiver模块，平衡时空信息压缩与细节保留，在压缩视觉token达13倍（每帧61 token）下，仍保持高准确率，适用于长短视频理解。**

- **链接: [http://arxiv.org/pdf/2505.15529v1](http://arxiv.org/pdf/2505.15529v1)**

> **作者:** Lingyu Kong; Hongzhi Zhang; Jingyuan Zhang; Jianzhao Huang; Kunze Li; Qi Wang; Fuzheng Zhang
>
> **摘要:** Current vision-language models (VLMs) have demonstrated remarkable capabilities across diverse video understanding applications. Designing VLMs for video inputs requires effectively modeling the temporal dimension (i.e. capturing dependencies across frames) and balancing the processing of short and long videos. Specifically, short videos demand preservation of fine-grained details, whereas long videos require strategic compression of visual information to handle extensive temporal contexts efficiently. However, our empirical analysis reveals a critical limitation: most existing VLMs suffer severe performance degradation in long video understanding tasks when compressing visual tokens below a quarter of their original visual tokens. To enable more effective modeling of both short and long video inputs, we propose Clapper, a method that utilizes a slow-fast strategy for video representation and introduces a novel module named TimePerceiver for efficient temporal-spatial encoding within existing VLM backbones. By using our method, we achieves 13x compression of visual tokens per frame (averaging 61 tokens/frame) without compromising QA accuracy. In our experiments, Clapper achieves 62.0% on VideoMME, 69.8% on MLVU, and 67.4% on TempCompass, all with fewer than 6,000 visual tokens per video. The code will be publicly available on the homepage.
>
---
#### [new 034] Bridging Sign and Spoken Languages: Pseudo Gloss Generation for Sign Language Translation
- **分类: cs.CV**

- **简介: 该论文提出无需口语文本标注的手语翻译方法，解决依赖专家标注成本高的问题。通过LLM生成伪口语文本并优化其与手语视频的对齐，结合弱监督和CTC损失，三阶段训练缩小模态差距，提升翻译效果。**

- **链接: [http://arxiv.org/pdf/2505.15438v1](http://arxiv.org/pdf/2505.15438v1)**

> **作者:** Jianyuan Guo; Peike Li; Trevor Cohn
>
> **备注:** Technical report, 21 pages
>
> **摘要:** Sign Language Translation (SLT) aims to map sign language videos to spoken language text. A common approach relies on gloss annotations as an intermediate representation, decomposing SLT into two sub-tasks: video-to-gloss recognition and gloss-to-text translation. While effective, this paradigm depends on expert-annotated gloss labels, which are costly and rarely available in existing datasets, limiting its scalability. To address this challenge, we propose a gloss-free pseudo gloss generation framework that eliminates the need for human-annotated glosses while preserving the structured intermediate representation. Specifically, we prompt a Large Language Model (LLM) with a few example text-gloss pairs using in-context learning to produce draft sign glosses from spoken language text. To enhance the correspondence between LLM-generated pseudo glosses and the sign sequences in video, we correct the ordering in the pseudo glosses for better alignment via a weakly supervised learning process. This reordering facilitates the incorporation of auxiliary alignment objectives, and allows for the use of efficient supervision via a Connectionist Temporal Classification (CTC) loss. We train our SLT mode, which consists of a vision encoder and a translator, through a three-stage pipeline, which progressively narrows the modality gap between sign language and spoken language. Despite its simplicity, our approach outperforms previous state-of-the-art gloss-free frameworks on two SLT benchmarks and achieves competitive results compared to gloss-based methods.
>
---
#### [new 035] UWSAM: Segment Anything Model Guided Underwater Instance Segmentation and A Large-scale Benchmark Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于水下实例分割任务，旨在解决Segment Anything Model（SAM）在水下场景性能不足及计算开销大的问题。工作包括：1）构建含10,048张标注图像的UIIS10K数据集；2）提出UWSAM模型，通过Mask GAT知识蒸馏压缩SAM视觉主干，并设计自动提示生成模块EUPG，提升水下实例定位与分割效率。**

- **链接: [http://arxiv.org/pdf/2505.15581v1](http://arxiv.org/pdf/2505.15581v1)**

> **作者:** Hua Li; Shijie Lian; Zhiyuan Li; Runmin Cong; Sam Kwong
>
> **摘要:** With recent breakthroughs in large-scale modeling, the Segment Anything Model (SAM) has demonstrated significant potential in a variety of visual applications. However, due to the lack of underwater domain expertise, SAM and its variants face performance limitations in end-to-end underwater instance segmentation tasks, while their higher computational requirements further hinder their application in underwater scenarios. To address this challenge, we propose a large-scale underwater instance segmentation dataset, UIIS10K, which includes 10,048 images with pixel-level annotations for 10 categories. Then, we introduce UWSAM, an efficient model designed for automatic and accurate segmentation of underwater instances. UWSAM efficiently distills knowledge from the SAM ViT-Huge image encoder into the smaller ViT-Small image encoder via the Mask GAT-based Underwater Knowledge Distillation (MG-UKD) method for effective visual representation learning. Furthermore, we design an End-to-end Underwater Prompt Generator (EUPG) for UWSAM, which automatically generates underwater prompts instead of explicitly providing foreground points or boxes as prompts, thus enabling the network to locate underwater instances accurately for efficient segmentation. Comprehensive experimental results show that our model is effective, achieving significant performance improvements over state-of-the-art methods on multiple underwater instance datasets. Datasets and codes are available at https://github.com/LiamLian0727/UIIS10K.
>
---
#### [new 036] Lossless Token Merging Even Without Fine-Tuning in Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于视觉Transformer（ViT）模型压缩任务，旨在解决现有token压缩方法信息损失严重且需大量训练的问题。提出自适应token合并(ATM)方法，通过动态调整层间相似度阈值和优化合并策略，在无需微调情况下实现无损压缩，减少30%计算量且保持精度。**

- **链接: [http://arxiv.org/pdf/2505.15160v1](http://arxiv.org/pdf/2505.15160v1)**

> **作者:** Jaeyeon Lee; Dong-Wan Choi
>
> **备注:** Under Review
>
> **摘要:** Although Vision Transformers (ViTs) have become the standard architecture in computer vision, their massive sizes lead to significant computational overhead. Token compression techniques have attracted considerable attention to address this issue, but they often suffer from severe information loss, requiring extensive additional training to achieve practical performance. In this paper, we propose Adaptive Token Merging (ATM), a novel method that ensures lossless token merging, eliminating the need for fine-tuning while maintaining competitive performance. ATM adaptively reduces tokens across layers and batches by carefully adjusting layer-specific similarity thresholds, thereby preventing the undesirable merging of less similar tokens with respect to each layer. Furthermore, ATM introduces a novel token matching technique that considers not only similarity but also merging sizes, particularly for the final layers, to minimize the information loss incurred from each merging operation. We empirically validate our method across a wide range of pretrained models, demonstrating that ATM not only outperforms all existing training-free methods but also surpasses most training-intensive approaches, even without additional training. Remarkably, training-free ATM achieves over a 30% reduction in FLOPs for the DeiT-T and DeiT-S models without any drop in their original accuracy.
>
---
#### [new 037] Towards Zero-Shot Differential Morphing Attack Detection with Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文研究零样本差异形变攻击检测任务，利用多模态大语言模型（如ChatGPT-4o和Gemini）提升生物识别系统中攻击检测的准确性和可解释性。通过设计Chain-of-Thought提示工程优化模型推理，首次在真实 passport 数据上验证，对比分析两模型性能，发现ChatGPT-4o检测更优但存在较高拒答率，Gemini解释更稳定。**

- **链接: [http://arxiv.org/pdf/2505.15332v1](http://arxiv.org/pdf/2505.15332v1)**

> **作者:** Ria Shekhawat; Hailin Li; Raghavendra Ramachandra; Sushma Venkatesh
>
> **备注:** Accepted at IEEE International Conference on Automatic Face and Gesture Recognition (FG 2025)
>
> **摘要:** Leveraging the power of multimodal large language models (LLMs) offers a promising approach to enhancing the accuracy and interpretability of morphing attack detection (MAD), especially in real-world biometric applications. This work introduces the use of LLMs for differential morphing attack detection (D-MAD). To the best of our knowledge, this is the first study to employ multimodal LLMs to D-MAD using real biometric data. To effectively utilize these models, we design Chain-of-Thought (CoT)-based prompts to reduce failure-to-answer rates and enhance the reasoning behind decisions. Our contributions include: (1) the first application of multimodal LLMs for D-MAD using real data subjects, (2) CoT-based prompt engineering to improve response reliability and explainability, (3) comprehensive qualitative and quantitative benchmarking of LLM performance using data from 54 individuals captured in passport enrollment scenarios, and (4) comparative analysis of two multimodal LLMs: ChatGPT-4o and Gemini providing insights into their morphing attack detection accuracy and decision transparency. Experimental results show that ChatGPT-4o outperforms Gemini in detection accuracy, especially against GAN-based morphs, though both models struggle under challenging conditions. While Gemini offers more consistent explanations, ChatGPT-4o is more resilient but prone to a higher failure-to-answer rate.
>
---
#### [new 038] GS2E: Gaussian Splatting is an Effective Data Generator for Event Stream Generation
- **分类: cs.CV**

- **简介: 该论文提出GS2E数据集，通过3D高斯散射重建真实场景并结合物理模拟生成事件流，解决现有数据集视角单一、几何不一致及硬件依赖问题，提升事件视觉任务（如3D重建）的泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.15287v1](http://arxiv.org/pdf/2505.15287v1)**

> **作者:** Yuchen Li; Chaoran Feng; Zhenyu Tang; Kaiyuan Deng; Wangbo Yu; Yonghong Tian; Li Yuan
>
> **备注:** 21 pages, 7 figures. More details at http://intothemild.github.io/GS2E.github.io
>
> **摘要:** We introduce GS2E (Gaussian Splatting to Event), a large-scale synthetic event dataset for high-fidelity event vision tasks, captured from real-world sparse multi-view RGB images. Existing event datasets are often synthesized from dense RGB videos, which typically lack viewpoint diversity and geometric consistency, or depend on expensive, difficult-to-scale hardware setups. GS2E overcomes these limitations by first reconstructing photorealistic static scenes using 3D Gaussian Splatting, and subsequently employing a novel, physically-informed event simulation pipeline. This pipeline generally integrates adaptive trajectory interpolation with physically-consistent event contrast threshold modeling. Such an approach yields temporally dense and geometrically consistent event streams under diverse motion and lighting conditions, while ensuring strong alignment with underlying scene structures. Experimental results on event-based 3D reconstruction demonstrate GS2E's superior generalization capabilities and its practical value as a benchmark for advancing event vision research.
>
---
#### [new 039] LENS: Multi-level Evaluation of Multimodal Reasoning with Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出LENS基准，解决现有多模态模型评估无法有效测试低级感知与高级推理协同的问题。构建含3.4K现代图像及6万+问题的多层级数据集（感知、理解、推理），覆盖12种日常场景，评估15+前沿模型，显示其推理任务准确率均低于60%。**

- **链接: [http://arxiv.org/pdf/2505.15616v1](http://arxiv.org/pdf/2505.15616v1)**

> **作者:** Ruilin Yao; Bo Zhang; Jirui Huang; Xinwei Long; Yifang Zhang; Tianyu Zou; Yufei Wu; Shichao Su; Yifan Xu; Wenxi Zeng; Zhaoyu Yang; Guoyou Li; Shilan Zhang; Zichan Li; Yaxiong Chen; Shengwu Xiong; Peng Xu; Jiajun Zhang; Bowen Zhou; David Clifton; Luc Van Gool
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved significant advances in integrating visual and linguistic information, yet their ability to reason about complex and real-world scenarios remains limited. The existing benchmarks are usually constructed in the task-oriented manner without guarantee that different task samples come from the same data distribution, thus they often fall short in evaluating the synergistic effects of lower-level perceptual capabilities on higher-order reasoning. To lift this limitation, we contribute Lens, a multi-level benchmark with 3.4K contemporary images and 60K+ human-authored questions covering eight tasks and 12 daily scenarios, forming three progressive task tiers, i.e., perception, understanding, and reasoning. One feature is that each image is equipped with rich annotations for all tasks. Thus, this dataset intrinsically supports to evaluate MLLMs to handle image-invariable prompts, from basic perception to compositional reasoning. In addition, our images are manully collected from the social media, in which 53% were published later than Jan. 2025. We evaluate 15+ frontier MLLMs such as Qwen2.5-VL-72B, InternVL3-78B, GPT-4o and two reasoning models QVQ-72B-preview and Kimi-VL. These models are released later than Dec. 2024, and none of them achieve an accuracy greater than 60% in the reasoning tasks. Project page: https://github.com/Lens4MLLMs/lens. ICCV 2025 workshop page: https://lens4mllms.github.io/mars2-workshop-iccv2025/
>
---
#### [new 040] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态思维链（MCoT）机制，旨在解析其提升视觉语言模型性能的原理。针对现有MCoT文本型与交错型方法，提出"视觉思维"概念，分析其四种表达形式对推理效果的影响，并揭示其作为图像与深层模型间的中介作用，为MCoT优化提供理论依据。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15510v1](http://arxiv.org/pdf/2505.15510v1)**

> **作者:** Zihui Cheng; Qiguang Chen; Xiao Xu; Jiaqi Wang; Weiyun Wang; Hao Fei; Yidong Wang; Alex Jinpeng Wang; Zhi Chen; Wanxiang Che; Libo Qin
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved significant success in multimodal tasks, with multimodal chain-of-thought (MCoT) further enhancing performance and interpretability. Recent MCoT methods fall into two categories: (i) Textual-MCoT (T-MCoT), which takes multimodal input and produces textual output; and (ii) Interleaved-MCoT (I-MCoT), which generates interleaved image-text outputs. Despite advances in both approaches, the mechanisms driving these improvements are not fully understood. To fill this gap, we first reveal that MCoT boosts LVLMs by incorporating visual thoughts, which convey image information to the reasoning process regardless of the MCoT format, depending only on clarity and conciseness of expression. Furthermore, to explore visual thoughts systematically, we define four distinct forms of visual thought expressions and analyze them comprehensively. Our findings demonstrate that these forms differ in clarity and conciseness, yielding varying levels of MCoT improvement. Additionally, we explore the internal nature of visual thoughts, finding that visual thoughts serve as intermediaries between the input image and reasoning to deeper transformer layers, enabling more advanced visual information transmission. We hope that the visual thoughts can inspire further breakthroughs for future MCoT research.
>
---
#### [new 041] Intentional Gesture: Deliver Your Intentions with Gestures for Speech
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文属于手势生成任务，旨在解决现有方法依赖语言线索但忽视沟通意图导致语义浅薄的问题。提出Intentional-Gesture框架：构建含意图标注的InG数据集，并设计意图感知的Motion Tokenizer，将高阶沟通意图融入动作表示，实现语义与时间同步的手势生成，达新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.15197v1](http://arxiv.org/pdf/2505.15197v1)**

> **作者:** Pinxin Liu; Haiyang Liu; Luchuan Song; Chenliang Xu
>
> **摘要:** When humans speak, gestures help convey communicative intentions, such as adding emphasis or describing concepts. However, current co-speech gesture generation methods rely solely on superficial linguistic cues (\textit{e.g.} speech audio or text transcripts), neglecting to understand and leverage the communicative intention that underpins human gestures. This results in outputs that are rhythmically synchronized with speech but are semantically shallow. To address this gap, we introduce \textbf{Intentional-Gesture}, a novel framework that casts gesture generation as an intention-reasoning task grounded in high-level communicative functions. % First, we curate the \textbf{InG} dataset by augmenting BEAT-2 with gesture-intention annotations (\textit{i.e.}, text sentences summarizing intentions), which are automatically annotated using large vision-language models. Next, we introduce the \textbf{Intentional Gesture Motion Tokenizer} to leverage these intention annotations. It injects high-level communicative functions (\textit{e.g.}, intentions) into tokenized motion representations to enable intention-aware gesture synthesis that are both temporally aligned and semantically meaningful, achieving new state-of-the-art performance on the BEAT-2 benchmark. Our framework offers a modular foundation for expressive gesture generation in digital humans and embodied AI. Project Page: https://andypinxinliu.github.io/Intentional-Gesture
>
---
#### [new 042] PlantDreamer: Achieving Realistic 3D Plant Models with Diffusion-Guided Gaussian Splatting
- **分类: cs.CV; cs.GR; I.2.10; I.3.0; I.4.5**

- **简介: 该论文提出PlantDreamer，解决复杂3D植物生成难题。通过深度ControlNet、低秩适配及高斯筛选算法，提升纹理与几何精度，支持合成生成及现实点云优化。实验显示其优于现有方法，促进3D植物分析与数据升级。**

- **链接: [http://arxiv.org/pdf/2505.15528v1](http://arxiv.org/pdf/2505.15528v1)**

> **作者:** Zane K J Hartley; Lewis A G Stuart; Andrew P French; Michael P Pound
>
> **备注:** 13 pages, 5 figures, 4 tables
>
> **摘要:** Recent years have seen substantial improvements in the ability to generate synthetic 3D objects using AI. However, generating complex 3D objects, such as plants, remains a considerable challenge. Current generative 3D models struggle with plant generation compared to general objects, limiting their usability in plant analysis tools, which require fine detail and accurate geometry. We introduce PlantDreamer, a novel approach to 3D synthetic plant generation, which can achieve greater levels of realism for complex plant geometry and textures than available text-to-3D models. To achieve this, our new generation pipeline leverages a depth ControlNet, fine-tuned Low-Rank Adaptation and an adaptable Gaussian culling algorithm, which directly improve textural realism and geometric integrity of generated 3D plant models. Additionally, PlantDreamer enables both purely synthetic plant generation, by leveraging L-System-generated meshes, and the enhancement of real-world plant point clouds by converting them into 3D Gaussian Splats. We evaluate our approach by comparing its outputs with state-of-the-art text-to-3D models, demonstrating that PlantDreamer outperforms existing methods in producing high-fidelity synthetic plants. Our results indicate that our approach not only advances synthetic plant generation, but also facilitates the upgrading of legacy point cloud datasets, making it a valuable tool for 3D phenotyping applications.
>
---
#### [new 043] Enhancing Shape Perception and Segmentation Consistency for Industrial Image Inspection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对工业图像检测中传统语义分割模型在复杂环境下固定组件边界感知不足、分割一致性差及计算效率低的问题，提出Shape-Aware Efficient Network（SPENet）。通过分离边界与主体信息提取、引入Variable Boundary Domain（VBD）处理模糊边界及新指标CMSE，提升分割一致性和效率，实现高精度与实时性，CMSE较最优模型降低超50%。**

- **链接: [http://arxiv.org/pdf/2505.14718v1](http://arxiv.org/pdf/2505.14718v1)**

> **作者:** Guoxuan Mao; Ting Cao; Ziyang Li; Yuan Dong
>
> **摘要:** Semantic segmentation stands as a pivotal research focus in computer vision. In the context of industrial image inspection, conventional semantic segmentation models fail to maintain the segmentation consistency of fixed components across varying contextual environments due to a lack of perception of object contours. Given the real-time constraints and limited computing capability of industrial image detection machines, it is also necessary to create efficient models to reduce computational complexity. In this work, a Shape-Aware Efficient Network (SPENet) is proposed, which focuses on the shapes of objects to achieve excellent segmentation consistency by separately supervising the extraction of boundary and body information from images. In SPENet, a novel method is introduced for describing fuzzy boundaries to better adapt to real-world scenarios named Variable Boundary Domain (VBD). Additionally, a new metric, Consistency Mean Square Error(CMSE), is proposed to measure segmentation consistency for fixed components. Our approach attains the best segmentation accuracy and competitive speed on our dataset, showcasing significant advantages in CMSE among numerous state-of-the-art real-time segmentation networks, achieving a reduction of over 50% compared to the previously top-performing models.
>
---
#### [new 044] LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval
- **分类: cs.CV**

- **简介: 该论文属于在线视频理解任务，旨在解决现有视频LLMs在实时应用中内存占用高、响应慢的问题。提出无训练框架LiveVLM，通过构建流式KV缓存实时处理视频流，压缩视觉信息并保留长期细节，减少冗余，实现快速响应，提升处理效率与速度。**

- **链接: [http://arxiv.org/pdf/2505.15269v1](http://arxiv.org/pdf/2505.15269v1)**

> **作者:** Zhenyu Ning; Guangda Liu; Qihao Jin; Wenchao Ding; Minyi Guo; Jieru Zhao
>
> **摘要:** Recent developments in Video Large Language Models (Video LLMs) have enabled models to process long video sequences and demonstrate remarkable performance. Nonetheless, studies predominantly focus on offline video question answering, neglecting memory usage and response speed that are essential in various real-world applications, such as Deepseek services, autonomous driving, and robotics. To mitigate these challenges, we propose $\textbf{LiveVLM}$, a training-free framework specifically designed for streaming, online video understanding and real-time interaction. Unlike existing works that process videos only after one question is posed, LiveVLM constructs an innovative streaming-oriented KV cache to process video streams in real-time, retain long-term video details and eliminate redundant KVs, ensuring prompt responses to user queries. For continuous video streams, LiveVLM generates and compresses video key-value tensors (video KVs) to reserve visual information while improving memory efficiency. Furthermore, when a new question is proposed, LiveVLM incorporates an online question-answering process that efficiently fetches both short-term and long-term visual information, while minimizing interference from redundant context. Extensive experiments demonstrate that LiveVLM enables the foundation LLaVA-OneVision model to process 44$\times$ number of frames on the same device, and achieves up to 5$\times$ speedup in response speed compared with SoTA online methods at an input of 256 frames, while maintaining the same or better model performance.
>
---
#### [new 045] CAD: A General Multimodal Framework for Video Deepfake Detection via Cross-Modal Alignment and Distillation
- **分类: cs.CV**

- **简介: 该论文提出CAD框架，用于视频深度伪造检测。针对现有方法忽视模态特定特征与语义不一致互补性的问题，通过跨模态对齐识别语义同步异常（如口型不同步），及跨模态蒸馏融合特征并保留模态特定痕迹（如音频失真），有效整合多模态信息，显著提升检测性能。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15233v1](http://arxiv.org/pdf/2505.15233v1)**

> **作者:** Yuxuan Du; Zhendong Wang; Yuhao Luo; Caiyong Piao; Zhiyuan Yan; Hao Li; Li Yuan
>
> **摘要:** The rapid emergence of multimodal deepfakes (visual and auditory content are manipulated in concert) undermines the reliability of existing detectors that rely solely on modality-specific artifacts or cross-modal inconsistencies. In this work, we first demonstrate that modality-specific forensic traces (e.g., face-swap artifacts or spectral distortions) and modality-shared semantic misalignments (e.g., lip-speech asynchrony) offer complementary evidence, and that neglecting either aspect limits detection performance. Existing approaches either naively fuse modality-specific features without reconciling their conflicting characteristics or focus predominantly on semantic misalignment at the expense of modality-specific fine-grained artifact cues. To address these shortcomings, we propose a general multimodal framework for video deepfake detection via Cross-Modal Alignment and Distillation (CAD). CAD comprises two core components: 1) Cross-modal alignment that identifies inconsistencies in high-level semantic synchronization (e.g., lip-speech mismatches); 2) Cross-modal distillation that mitigates feature conflicts during fusion while preserving modality-specific forensic traces (e.g., spectral distortions in synthetic audio). Extensive experiments on both multimodal and unimodal (e.g., image-only/video-only)deepfake benchmarks demonstrate that CAD significantly outperforms previous methods, validating the necessity of harmonious integration of multimodal complementary information.
>
---
#### [new 046] Detection of Underwater Multi-Targets Based on Self-Supervised Learning and Deformable Path Aggregation Feature Pyramid Network
- **分类: cs.CV**

- **简介: 该论文针对水下环境低对比度、目标遮挡与密集分布导致检测精度低的问题，提出基于自监督学习（SimSiam）预训练和改进的可变形路径聚合特征金字塔网络。通过融合可变形/空洞卷积扩大感受野，并采用EIoU损失函数优化边界框回归，提升水下多目标检测的鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2505.15518v1](http://arxiv.org/pdf/2505.15518v1)**

> **作者:** Chang Liu
>
> **备注:** ICIC 2025 accepted
>
> **摘要:** To overcome the constraints of the underwater environment and improve the accuracy and robustness of underwater target detection models, this paper develops a specialized dataset for underwater target detection and proposes an efficient algorithm for underwater multi-target detection. A self-supervised learning based on the SimSiam structure is employed for the pre-training of underwater target detection network. To address the problems of low detection accuracy caused by low contrast, mutual occlusion and dense distribution of underwater targets in underwater object detection, a detection model suitable for underwater target detection is proposed by introducing deformable convolution and dilated convolution. The proposed detection model can obtain more effective information by increasing the receptive field. In addition, the regression loss function EIoU is introduced, which improves model performance by separately calculating the width and height losses of the predicted box. Experiment results show that the accuracy of the underwater target detection has been improved by the proposed detector.
>
---
#### [new 047] On the Robustness of Medical Vision-Language Models: Are they Truly Generalizable?
- **分类: cs.CV**

- **简介: 该论文属于医疗视觉语言模型（MVLMs）的鲁棒性研究，旨在解决其在真实世界噪声/损坏影像中的性能退化问题。团队构建了MediMeta-C基准测试框架，并提出RobustMedCLIP方法，通过微调提升模型抗干扰能力，实验显示现有模型鲁棒性不足，而新方法有效平衡了跨模态泛化与抗噪能力。**

- **链接: [http://arxiv.org/pdf/2505.15425v1](http://arxiv.org/pdf/2505.15425v1)**

> **作者:** Raza Imam; Rufael Marew; Mohammad Yaqub
>
> **备注:** Dataset and Code is available at https://github.com/BioMedIA-MBZUAI/RobustMedCLIP Accepted at: Medical Image Understanding and Analysis (MIUA) 2025
>
> **摘要:** Medical Vision-Language Models (MVLMs) have achieved par excellence generalization in medical image analysis, yet their performance under noisy, corrupted conditions remains largely untested. Clinical imaging is inherently susceptible to acquisition artifacts and noise; however, existing evaluations predominantly assess generally clean datasets, overlooking robustness -- i.e., the model's ability to perform under real-world distortions. To address this gap, we first introduce MediMeta-C, a corruption benchmark that systematically applies several perturbations across multiple medical imaging datasets. Combined with MedMNIST-C, this establishes a comprehensive robustness evaluation framework for MVLMs. We further propose RobustMedCLIP, a visual encoder adaptation of a pretrained MVLM that incorporates few-shot tuning to enhance resilience against corruptions. Through extensive experiments, we benchmark 5 major MVLMs across 5 medical imaging modalities, revealing that existing models exhibit severe degradation under corruption and struggle with domain-modality tradeoffs. Our findings highlight the necessity of diverse training and robust adaptation strategies, demonstrating that efficient low-rank adaptation when paired with few-shot tuning, improves robustness while preserving generalization across modalities.
>
---
#### [new 048] FRN: Fractal-Based Recursive Spectral Reconstruction Network
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出FRN网络，用于从RGB图像重建高光谱图像。针对传统方法直接整合全光谱信息效果有限的问题，其采用分形递归框架，逐步预测波段，利用邻近光谱信息及低秩特性，并设计带感知模型减少干扰，实验显示优于现有方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15439v1](http://arxiv.org/pdf/2505.15439v1)**

> **作者:** Ge Meng; Zhongnan Cai; Ruizhe Chen; Jingyan Tu; Yingying Wang; Yue Huang; Xinghao Ding
>
> **摘要:** Generating hyperspectral images (HSIs) from RGB images through spectral reconstruction can significantly reduce the cost of HSI acquisition. In this paper, we propose a Fractal-Based Recursive Spectral Reconstruction Network (FRN), which differs from existing paradigms that attempt to directly integrate the full-spectrum information from the R, G, and B channels in a one-shot manner. Instead, it treats spectral reconstruction as a progressive process, predicting from broad to narrow bands or employing a coarse-to-fine approach for predicting the next wavelength. Inspired by fractals in mathematics, FRN establishes a novel spectral reconstruction paradigm by recursively invoking an atomic reconstruction module. In each invocation, only the spectral information from neighboring bands is used to provide clues for the generation of the image at the next wavelength, which follows the low-rank property of spectral data. Moreover, we design a band-aware state space model that employs a pixel-differentiated scanning strategy at different stages of the generation process, further suppressing interference from low-correlation regions caused by reflectance differences. Through extensive experimentation across different datasets, FRN achieves superior reconstruction performance compared to state-of-the-art methods in both quantitative and qualitative evaluations.
>
---
#### [new 049] MultiMAE Meets Earth Observation: Pre-training Multi-modal Multi-task Masked Autoencoders for Earth Observation Tasks
- **分类: cs.CV**

- **简介: 该论文属于地球观测任务，旨在解决多模态预训练模型在下游任务数据结构变化时的迁移能力不足问题。提出基于MultiMAE的多模态多任务预训练策略，通过联合重建光谱、高程、分割等数据，提升模型在分类与分割任务中的灵活性和性能，无需针对不同模态单独训练。**

- **链接: [http://arxiv.org/pdf/2505.14951v1](http://arxiv.org/pdf/2505.14951v1)**

> **作者:** Jose Sosa; Danila Rukhovich; Anis Kacem; Djamila Aouada
>
> **摘要:** Multi-modal data in Earth Observation (EO) presents a huge opportunity for improving transfer learning capabilities when pre-training deep learning models. Unlike prior work that often overlooks multi-modal EO data, recent methods have started to include it, resulting in more effective pre-training strategies. However, existing approaches commonly face challenges in effectively transferring learning to downstream tasks where the structure of available data differs from that used during pre-training. This paper addresses this limitation by exploring a more flexible multi-modal, multi-task pre-training strategy for EO data. Specifically, we adopt a Multi-modal Multi-task Masked Autoencoder (MultiMAE) that we pre-train by reconstructing diverse input modalities, including spectral, elevation, and segmentation data. The pre-trained model demonstrates robust transfer learning capabilities, outperforming state-of-the-art methods on various EO datasets for classification and segmentation tasks. Our approach exhibits significant flexibility, handling diverse input configurations without requiring modality-specific pre-trained models. Code will be available at: https://github.com/josesosajs/multimae-meets-eo.
>
---
#### [new 050] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition
- **分类: cs.CV**

- **简介: 该论文提出InstructSAM，一种无需训练的框架，用于指令驱动的遥感目标识别。针对现有方法依赖显式类别提示、难以处理复杂查询的问题，提出InstructCDS任务（计数、检测、分割）及EarthInstruct基准数据集，并结合视觉语言模型与SAM2，通过二进制整数规划分配标签，提升效率与精度。**

- **链接: [http://arxiv.org/pdf/2505.15818v1](http://arxiv.org/pdf/2505.15818v1)**

> **作者:** Yijie Zheng; Weijie Wu; Qingyun Li; Xuehui Wang; Xu Zhou; Aiai Ren; Jun Shen; Long Zhao; Guoqing Li; Xue Yang
>
> **摘要:** Language-Guided object recognition in remote sensing imagery is crucial for large-scale mapping and automated data annotation. However, existing open-vocabulary and visual grounding methods rely on explicit category cues, limiting their ability to handle complex or implicit queries that require advanced reasoning. To address this issue, we introduce a new suite of tasks, including Instruction-Oriented Object Counting, Detection, and Segmentation (InstructCDS), covering open-vocabulary, open-ended, and open-subclass scenarios. We further present EarthInstruct, the first InstructCDS benchmark for earth observation. It is constructed from two diverse remote sensing datasets with varying spatial resolutions and annotation rules across 20 categories, necessitating models to interpret dataset-specific instructions. Given the scarcity of semantically rich labeled data in remote sensing, we propose InstructSAM, a training-free framework for instruction-driven object recognition. InstructSAM leverages large vision-language models to interpret user instructions and estimate object counts, employs SAM2 for mask proposal, and formulates mask-label assignment as a binary integer programming problem. By integrating semantic similarity with counting constraints, InstructSAM efficiently assigns categories to predicted masks without relying on confidence thresholds. Experiments demonstrate that InstructSAM matches or surpasses specialized baselines across multiple tasks while maintaining near-constant inference time regardless of object count, reducing output tokens by 89% and overall runtime by over 32% compared to direct generation approaches. We believe the contributions of the proposed tasks, benchmark, and effective approach will advance future research in developing versatile object recognition systems.
>
---
#### [new 051] TinyDrive: Multiscale Visual Question Answering with Selective Token Routing for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出TinyDrive，一种轻量化视觉语言模型，用于自动驾驶场景的多视角视觉问答。针对传统模型计算资源消耗大的问题，其通过多尺度视觉编码器和双层优先机制（令牌动态选择、序列评分筛选）优化效率，在 DriveLM 数据集上实现参数减少但提升11.1% BLEU-4和35.4% METEOR的最优效果。**

- **链接: [http://arxiv.org/pdf/2505.15564v1](http://arxiv.org/pdf/2505.15564v1)**

> **作者:** Hossein Hassani; Soodeh Nikan; Abdallah Shami
>
> **摘要:** Vision Language Models (VLMs) employed for visual question-answering (VQA) in autonomous driving often require substantial computational resources that pose a challenge for their deployment in resource-constrained vehicles. To address this challenge, we introduce TinyDrive, a lightweight yet effective VLM for multi-view VQA in driving scenarios. Our model comprises two key components including a multiscale vision encoder and a dual-level prioritization mechanism for tokens and sequences. The multiscale encoder facilitates the processing of multi-view images at diverse resolutions through scale injection and cross-scale gating to generate enhanced visual representations. At the token level, we design a token routing mechanism that dynamically selects and process the most informative tokens based on learned importance scores. At the sequence level, we propose integrating normalized loss, uncertainty estimates, and a diversity metric to formulate sequence scores that rank and preserve samples within a sequence priority buffer. Samples with higher scores are more frequently selected for training. TinyDrive is first evaluated on our custom-curated VQA dataset, and it is subsequently tested on the public DriveLM benchmark, where it achieves state-of-the-art language understanding performance. Notably, it achieves relative improvements of 11.1% and 35.4% in BLEU-4 and METEOR scores, respectively, despite having a significantly smaller parameter count.
>
---
#### [new 052] Mouse Lockbox Dataset: Behavior Recognition for Mice Solving Lockboxes
- **分类: cs.CV**

- **简介: 该论文提出Mouse Lockbox数据集，用于小鼠解决机械谜题的行为识别。针对现有数据集在复杂个体行为分析上的不足，收集超110小时三视角视频，提供13%人工标注的帧级动作标签，并展示基于姿态追踪的分类框架面临的挑战，推动自动化行为分析发展，数据集已公开。**

- **链接: [http://arxiv.org/pdf/2505.15408v1](http://arxiv.org/pdf/2505.15408v1)**

> **作者:** Patrik Reiske; Marcus N. Boon; Niek Andresen; Sole Traverso; Katharina Hohlbaum; Lars Lewejohann; Christa Thöne-Reineke; Olaf Hellwich; Henning Sprekeler
>
> **摘要:** Machine learning and computer vision methods have a major impact on the study of natural animal behavior, as they enable the (semi-)automatic analysis of vast amounts of video data. Mice are the standard mammalian model system in most research fields, but the datasets available today to refine such methods focus either on simple or social behaviors. In this work, we present a video dataset of individual mice solving complex mechanical puzzles, so-called lockboxes. The more than 110 hours of total playtime show their behavior recorded from three different perspectives. As a benchmark for frame-level action classification methods, we provide human-annotated labels for all videos of two different mice, that equal 13% of our dataset. Our keypoint (pose) tracking-based action classification framework illustrates the challenges of automated labeling of fine-grained behaviors, such as the manipulation of objects. We hope that our work will help accelerate the advancement of automated action and behavior classification in the computational neuroscience community. Our dataset is publicly available at https://doi.org/10.14279/depositonce-23850
>
---
#### [new 053] HAMF: A Hybrid Attention-Mamba Framework for Joint Scene Context Understanding and Future Motion Representation Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶运动预测任务，旨在解决现有方法因场景特征编码信息降级导致的预测精度不足问题。提出HAMF框架，通过融合注意力机制与Mamba模块，联合建模场景上下文与未来轨迹特征，实现精准多样的轨迹预测。**

- **链接: [http://arxiv.org/pdf/2505.15703v1](http://arxiv.org/pdf/2505.15703v1)**

> **作者:** Xiaodong Mei; Sheng Wang; Jie Cheng; Yingbing Chen; Dan Xu
>
> **备注:** In submission
>
> **摘要:** Motion forecasting represents a critical challenge in autonomous driving systems, requiring accurate prediction of surrounding agents' future trajectories. While existing approaches predict future motion states with the extracted scene context feature from historical agent trajectories and road layouts, they suffer from the information degradation during the scene feature encoding. To address the limitation, we propose HAMF, a novel motion forecasting framework that learns future motion representations with the scene context encoding jointly, to coherently combine the scene understanding and future motion state prediction. We first embed the observed agent states and map information into 1D token sequences, together with the target multi-modal future motion features as a set of learnable tokens. Then we design a unified Attention-based encoder, which synergistically combines self-attention and cross-attention mechanisms to model the scene context information and aggregate future motion features jointly. Complementing the encoder, we implement the Mamba module in the decoding stage to further preserve the consistency and correlations among the learned future motion representations, to generate the accurate and diverse final trajectories. Extensive experiments on Argoverse 2 benchmark demonstrate that our hybrid Attention-Mamba model achieves state-of-the-art motion forecasting performance with the simple and lightweight architecture.
>
---
#### [new 054] Seeing the Trees for the Forest: Rethinking Weakly-Supervised Medical Visual Grounding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于弱监督医学视觉定位任务，针对现有模型因注意力机制低效和全局token无法表征局部病灶特征导致文本-病灶关联失效的问题，提出DAP方法：利用VLM可解释性地图增强病灶区域特征、抑制背景干扰，无需像素标注使准确率提升20.74%。**

- **链接: [http://arxiv.org/pdf/2505.15123v1](http://arxiv.org/pdf/2505.15123v1)**

> **作者:** Ta Duc Huy; Duy Anh Huynh; Yutong Xie; Yuankai Qi; Qi Chen; Phi Le Nguyen; Sen Kim Tran; Son Lam Phung; Anton van den Hengel; Zhibin Liao; Minh-Son To; Johan W. Verjans; Vu Minh Hieu Phan
>
> **备注:** Under Review
>
> **摘要:** Visual grounding (VG) is the capability to identify the specific regions in an image associated with a particular text description. In medical imaging, VG enhances interpretability by highlighting relevant pathological features corresponding to textual descriptions, improving model transparency and trustworthiness for wider adoption of deep learning models in clinical practice. Current models struggle to associate textual descriptions with disease regions due to inefficient attention mechanisms and a lack of fine-grained token representations. In this paper, we empirically demonstrate two key observations. First, current VLMs assign high norms to background tokens, diverting the model's attention from regions of disease. Second, the global tokens used for cross-modal learning are not representative of local disease tokens. This hampers identifying correlations between the text and disease tokens. To address this, we introduce simple, yet effective Disease-Aware Prompting (DAP) process, which uses the explainability map of a VLM to identify the appropriate image features. This simple strategy amplifies disease-relevant regions while suppressing background interference. Without any additional pixel-level annotations, DAP improves visual grounding accuracy by 20.74% compared to state-of-the-art methods across three major chest X-ray datasets.
>
---
#### [new 055] Benchmarking Graph Neural Networks for Document Layout Analysis in Public Affairs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对数字PDF文档中异构元素与元数据不精确导致的布局分析难题，通过构建k近邻图/全连接图，结合预训练文本-视觉特征，测试单模态/多模态GNN模型，验证GraphSAGE在双分支k近邻图配置下效果最佳，证明局部布局关系与多模态融合的重要性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.14699v1](http://arxiv.org/pdf/2505.14699v1)**

> **作者:** Miguel Lopez-Duran; Julian Fierrez; Aythami Morales; Ruben Tolosana; Oscar Delgado-Mohatar; Alvaro Ortigosa
>
> **备注:** 15 pages, 2 figures, preprint presented in The Fifth ICDAR International Workshop on Machine Learning
>
> **摘要:** The automatic analysis of document layouts in digital-born PDF documents remains a challenging problem due to the heterogeneous arrangement of textual and nontextual elements and the imprecision of the textual metadata in the Portable Document Format. In this work, we benchmark Graph Neural Network (GNN) architectures for the task of fine-grained layout classification of text blocks from digital native documents. We introduce two graph construction structures: a k-closest-neighbor graph and a fully connected graph, and generate node features via pre-trained text and vision models, thus avoiding manual feature engineering. Three experimental frameworks are evaluated: single-modality (text or visual), concatenated multimodal, and dual-branch multimodal. We evaluated four foundational GNN models and compared them with the baseline. Our experiments are specifically conducted on a rich dataset of public affairs documents that includes more than 20 sources (e.g., regional and national-level official gazettes), 37K PDF documents, with 441K pages in total. Our results demonstrate that GraphSAGE operating on the k-closest-neighbor graph in a dual-branch configuration achieves the highest per-class and overall accuracy, outperforming the baseline in some sources. These findings confirm the importance of local layout relationships and multimodal fusion exploited through GNNs for the analysis of native digital document layouts.
>
---
#### [new 056] Exploring The Visual Feature Space for Multimodal Neural Decoding
- **分类: cs.CV**

- **简介: 该论文属于多模态神经解码任务，旨在解决现有方法对脑信号视觉细节（如物体、属性、关系）重建粗略的问题。工作包括分析多模态大模型的视觉特征空间，提出零样本脑解码方法及多粒度评估基准MG-BrainDub，提升解码精度。**

- **链接: [http://arxiv.org/pdf/2505.15755v1](http://arxiv.org/pdf/2505.15755v1)**

> **作者:** Weihao Xia; Cengiz Oztireli
>
> **备注:** Project: https://weihaox.github.io/VINDEX
>
> **摘要:** The intrication of brain signals drives research that leverages multimodal AI to align brain modalities with visual and textual data for explainable descriptions. However, most existing studies are limited to coarse interpretations, lacking essential details on object descriptions, locations, attributes, and their relationships. This leads to imprecise and ambiguous reconstructions when using such cues for visual decoding. To address this, we analyze different choices of vision feature spaces from pre-trained visual components within Multimodal Large Language Models (MLLMs) and introduce a zero-shot multimodal brain decoding method that interacts with these models to decode across multiple levels of granularities. % To assess a model's ability to decode fine details from brain signals, we propose the Multi-Granularity Brain Detail Understanding Benchmark (MG-BrainDub). This benchmark includes two key tasks: detailed descriptions and salient question-answering, with metrics highlighting key visual elements like objects, attributes, and relationships. Our approach enhances neural decoding precision and supports more accurate neuro-decoding applications. Code will be available at https://github.com/weihaox/VINDEX.
>
---
#### [new 057] SNAP: A Benchmark for Testing the Effects of Capture Conditions on Fundamental Vision Tasks
- **分类: cs.CV**

- **简介: 该论文提出SNAP基准，研究相机参数（快门、ISO、光圈）和光照等捕获条件对图像分类、目标检测及视觉问答任务的影响。针对现有模型对图像形成环境泛化能力不足的问题，分析数据集偏差，构建包含可控光照与密集采样参数的基准，测试多模型性能并建立人类基线，揭示模型易受捕获条件干扰且表现低于人类。**

- **链接: [http://arxiv.org/pdf/2505.15628v1](http://arxiv.org/pdf/2505.15628v1)**

> **作者:** Iuliia Kotseruba; John K. Tsotsos
>
> **摘要:** Generalization of deep-learning-based (DL) computer vision algorithms to various image perturbations is hard to establish and remains an active area of research. The majority of past analyses focused on the images already captured, whereas effects of the image formation pipeline and environment are less studied. In this paper, we address this issue by analyzing the impact of capture conditions, such as camera parameters and lighting, on DL model performance on 3 vision tasks -- image classification, object detection, and visual question answering (VQA). To this end, we assess capture bias in common vision datasets and create a new benchmark, SNAP (for $\textbf{S}$hutter speed, ISO se$\textbf{N}$sitivity, and $\textbf{AP}$erture), consisting of images of objects taken under controlled lighting conditions and with densely sampled camera settings. We then evaluate a large number of DL vision models and show the effects of capture conditions on each selected vision task. Lastly, we conduct an experiment to establish a human baseline for the VQA task. Our results show that computer vision datasets are significantly biased, the models trained on this data do not reach human accuracy even on the well-exposed images, and are susceptible to both major exposure changes and minute variations of camera settings. Code and data can be found at https://github.com/ykotseruba/SNAP
>
---
#### [new 058] ALN-P3: Unified Language Alignment for Perception, Prediction, and Planning in Autonomous Driving
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ALN-P3框架，解决自动驾驶中视觉系统与语言模型难以兼顾驾驶性能与语言推理的问题。通过感知、预测、规划三阶段的跨模态对齐机制，在训练阶段融合视觉与语言模块，提升决策与推理能力，实验显示其效果最优。**

- **链接: [http://arxiv.org/pdf/2505.15158v1](http://arxiv.org/pdf/2505.15158v1)**

> **作者:** Yunsheng Ma; Burhaneddin Yaman; Xin Ye; Mahmut Yurt; Jingru Luo; Abhirup Mallik; Ziran Wang; Liu Ren
>
> **备注:** 10 pages
>
> **摘要:** Recent advances have explored integrating large language models (LLMs) into end-to-end autonomous driving systems to enhance generalization and interpretability. However, most existing approaches are limited to either driving performance or vision-language reasoning, making it difficult to achieve both simultaneously. In this paper, we propose ALN-P3, a unified co-distillation framework that introduces cross-modal alignment between "fast" vision-based autonomous driving systems and "slow" language-driven reasoning modules. ALN-P3 incorporates three novel alignment mechanisms: Perception Alignment (P1A), Prediction Alignment (P2A), and Planning Alignment (P3A), which explicitly align visual tokens with corresponding linguistic outputs across the full perception, prediction, and planning stack. All alignment modules are applied only during training and incur no additional costs during inference. Extensive experiments on four challenging benchmarks-nuScenes, Nu-X, TOD3Cap, and nuScenes QA-demonstrate that ALN-P3 significantly improves both driving decisions and language reasoning, achieving state-of-the-art results.
>
---
#### [new 059] Constructing a 3D Town from a Single Image
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于单图像3D场景生成任务，旨在解决现有方法生成全场景时几何不一致、布局失真和网格质量低的问题。提出3DTown框架，通过区域分解与预训练模型生成、结合空间感知的3D插值技术，实现无需训练的高保真3D城镇生成，提升结构连贯性和细节质量。**

- **链接: [http://arxiv.org/pdf/2505.15765v1](http://arxiv.org/pdf/2505.15765v1)**

> **作者:** Kaizhi Zheng; Ruijian Zhang; Jing Gu; Jie Yang; Xin Eric Wang
>
> **摘要:** Acquiring detailed 3D scenes typically demands costly equipment, multi-view data, or labor-intensive modeling. Therefore, a lightweight alternative, generating complex 3D scenes from a single top-down image, plays an essential role in real-world applications. While recent 3D generative models have achieved remarkable results at the object level, their extension to full-scene generation often leads to inconsistent geometry, layout hallucinations, and low-quality meshes. In this work, we introduce 3DTown, a training-free framework designed to synthesize realistic and coherent 3D scenes from a single top-down view. Our method is grounded in two principles: region-based generation to improve image-to-3D alignment and resolution, and spatial-aware 3D inpainting to ensure global scene coherence and high-quality geometry generation. Specifically, we decompose the input image into overlapping regions and generate each using a pretrained 3D object generator, followed by a masked rectified flow inpainting process that fills in missing geometry while maintaining structural continuity. This modular design allows us to overcome resolution bottlenecks and preserve spatial structure without requiring 3D supervision or fine-tuning. Extensive experiments across diverse scenes show that 3DTown outperforms state-of-the-art baselines, including Trellis, Hunyuan3D-2, and TripoSG, in terms of geometry quality, spatial coherence, and texture fidelity. Our results demonstrate that high-quality 3D town generation is achievable from a single image using a principled, training-free approach.
>
---
#### [new 060] Zero-Shot Gaze-based Volumetric Medical Image Segmentation
- **分类: cs.CV; cs.AI; I.2.1**

- **简介: 该论文提出用眼动追踪替代手动提示进行三维医学图像分割，解决现有交互式模型（如SAM-2/MedSAM-2）依赖耗时的人工标注问题。通过测试合成/真实眼动数据，验证了眼动提示在效率上的优势及精度的轻微 trade-off，证明其可作为交互式分割的补充输入模态。**

- **链接: [http://arxiv.org/pdf/2505.15256v1](http://arxiv.org/pdf/2505.15256v1)**

> **作者:** Tatyana Shmykova; Leila Khaertdinova; Ilya Pershin
>
> **备注:** Accepted to MMFM-BIOMED Workshop @ CVPR 2025
>
> **摘要:** Accurate segmentation of anatomical structures in volumetric medical images is crucial for clinical applications, including disease monitoring and cancer treatment planning. Contemporary interactive segmentation models, such as Segment Anything Model 2 (SAM-2) and its medical variant (MedSAM-2), rely on manually provided prompts like bounding boxes and mouse clicks. In this study, we introduce eye gaze as a novel informational modality for interactive segmentation, marking the application of eye-tracking for 3D medical image segmentation. We evaluate the performance of using gaze-based prompts with SAM-2 and MedSAM-2 using both synthetic and real gaze data. Compared to bounding boxes, gaze-based prompts offer a time-efficient interaction approach with slightly lower segmentation quality. Our findings highlight the potential of using gaze as a complementary input modality for interactive 3D medical image segmentation.
>
---
#### [new 061] RAZER: Robust Accelerated Zero-Shot 3D Open-Vocabulary Panoptic Reconstruction with Spatio-Temporal Aggregation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D开放词汇全景重建任务，解决现有系统无法实时构建开放词汇语义地图及保持语义一致性的难题。提出零样本框架RAZER，融合GPU加速几何重建与视觉语言模型，通过实例级语义嵌入与空间索引关联，实现实时3D语义更新与自然语言交互，支持未知对象推理及分割检测。**

- **链接: [http://arxiv.org/pdf/2505.15373v1](http://arxiv.org/pdf/2505.15373v1)**

> **作者:** Naman Patel; Prashanth Krishnamurthy; Farshad Khorrami
>
> **摘要:** Mapping and understanding complex 3D environments is fundamental to how autonomous systems perceive and interact with the physical world, requiring both precise geometric reconstruction and rich semantic comprehension. While existing 3D semantic mapping systems excel at reconstructing and identifying predefined object instances, they lack the flexibility to efficiently build semantic maps with open-vocabulary during online operation. Although recent vision-language models have enabled open-vocabulary object recognition in 2D images, they haven't yet bridged the gap to 3D spatial understanding. The critical challenge lies in developing a training-free unified system that can simultaneously construct accurate 3D maps while maintaining semantic consistency and supporting natural language interactions in real time. In this paper, we develop a zero-shot framework that seamlessly integrates GPU-accelerated geometric reconstruction with open-vocabulary vision-language models through online instance-level semantic embedding fusion, guided by hierarchical object association with spatial indexing. Our training-free system achieves superior performance through incremental processing and unified geometric-semantic updates, while robustly handling 2D segmentation inconsistencies. The proposed general-purpose 3D scene understanding framework can be used for various tasks including zero-shot 3D instance retrieval, segmentation, and object detection to reason about previously unseen objects and interpret natural language queries. The project page is available at https://razer-3d.github.io.
>
---
#### [new 062] Beyond Linearity: Squeeze-and-Recalibrate Blocks for Few-Shot Whole Slide Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于少样本学习在病理全切片图像分类的任务，针对过拟合及特征学习偏差问题，提出Squeeze-and-Recalibrate（SR）块替代MIL模型线性层。通过低秩矩阵压缩参数并抑制冗余特征，结合随机冻结矩阵保留几何结构，理论证明其可逼近任意线性映射，实验显示参数更少且性能更优。**

- **链接: [http://arxiv.org/pdf/2505.15504v1](http://arxiv.org/pdf/2505.15504v1)**

> **作者:** Conghao Xiong; Zhengrui Guo; Zhe Xu; Yifei Zhang; Raymond Kai-Yu Tong; Si Yong Yeo; Hao Chen; Joseph J. Y. Sung; Irwin King
>
> **摘要:** Deep learning has advanced computational pathology but expert annotations remain scarce. Few-shot learning mitigates annotation burdens yet suffers from overfitting and discriminative feature mischaracterization. In addition, the current few-shot multiple instance learning (MIL) approaches leverage pretrained vision-language models to alleviate these issues, but at the cost of complex preprocessing and high computational cost. We propose a Squeeze-and-Recalibrate (SR) block, a drop-in replacement for linear layers in MIL models to address these challenges. The SR block comprises two core components: a pair of low-rank trainable matrices (squeeze pathway, SP) that reduces parameter count and imposes a bottleneck to prevent spurious feature learning, and a frozen random recalibration matrix that preserves geometric structure, diversifies feature directions, and redefines the optimization objective for the SP. We provide theoretical guarantees that the SR block can approximate any linear mapping to arbitrary precision, thereby ensuring that the performance of a standard MIL model serves as a lower bound for its SR-enhanced counterpart. Extensive experiments demonstrate that our SR-MIL models consistently outperform prior methods while requiring significantly fewer parameters and no architectural changes.
>
---
#### [new 063] Leveraging Generative AI Models to Explore Human Identity
- **分类: cs.CV; cs.AI**

- **简介: 该论文通过扩散模型生成人脸图像，研究人类身份形成与外部因素的关系。任务为探索人类身份流动性，解决身份如何受外部影响的问题。工作包括：建立生成过程与身份形成的对应关系，实验验证外部输入变化对生成图像的影响，间接证明身份依赖外部因素，并创作视频艺术《Fluidity of Human Identity》展现该概念。**

- **链接: [http://arxiv.org/pdf/2505.14843v1](http://arxiv.org/pdf/2505.14843v1)**

> **作者:** Yunha Yeo; Daeho Um
>
> **备注:** Accepted to ISEA 2025
>
> **摘要:** This paper attempts to explore human identity by utilizing neural networks in an indirect manner. For this exploration, we adopt diffusion models, state-of-the-art AI generative models trained to create human face images. By relating the generated human face to human identity, we establish a correspondence between the face image generation process of the diffusion model and the process of human identity formation. Through experiments with the diffusion model, we observe that changes in its external input result in significant changes in the generated face image. Based on the correspondence, we indirectly confirm the dependence of human identity on external factors in the process of human identity formation. Furthermore, we introduce \textit{Fluidity of Human Identity}, a video artwork that expresses the fluid nature of human identity affected by varying external factors. The video is available at https://www.behance.net/gallery/219958453/Fluidity-of-Human-Identity?.
>
---
#### [new 064] GT^2-GS: Geometry-aware Texture Transfer for Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D纹理迁移任务，旨在解决现有方法忽视几何信息导致纹理转移质量低的问题。提出GT²-GS框架，通过几何感知纹理增强模块扩充特征、几何一致损失函数优化纹理与场景的匹配，及交替迭代的几何保留策略，在保持几何完整性的同时提升纹理转移效果。**

- **链接: [http://arxiv.org/pdf/2505.15208v1](http://arxiv.org/pdf/2505.15208v1)**

> **作者:** Wenjie Liu; Zhongliang Liu; Junwei Shu; Changbo Wang; Yang Li
>
> **备注:** 15 pages, 16 figures
>
> **摘要:** Transferring 2D textures to 3D modalities is of great significance for improving the efficiency of multimedia content creation. Existing approaches have rarely focused on transferring image textures onto 3D representations. 3D style transfer methods are capable of transferring abstract artistic styles to 3D scenes. However, these methods often overlook the geometric information of the scene, which makes it challenging to achieve high-quality 3D texture transfer results. In this paper, we present GT^2-GS, a geometry-aware texture transfer framework for gaussian splitting. From the perspective of matching texture features with geometric information in rendered views, we identify the issue of insufficient texture features and propose a geometry-aware texture augmentation module to expand the texture feature set. Moreover, a geometry-consistent texture loss is proposed to optimize texture features into the scene representation. This loss function incorporates both camera pose and 3D geometric information of the scene, enabling controllable texture-oriented appearance editing. Finally, a geometry preservation strategy is introduced. By alternating between the texture transfer and geometry correction stages over multiple iterations, this strategy achieves a balance between learning texture features and preserving geometric integrity. Extensive experiments demonstrate the effectiveness and controllability of our method. Through geometric awareness, our approach achieves texture transfer results that better align with human visual perception. Our homepage is available at https://vpx-ecnu.github.io/GT2-GS-website.
>
---
#### [new 065] FaceCrafter: Identity-Conditional Diffusion with Disentangled Control over Facial Pose, Expression, and Emotion
- **分类: cs.CV**

- **简介: 该论文属于条件人脸生成任务，旨在解决现有方法难以精准控制面部姿势、表情和情绪等问题。提出FaceCrafter模型，通过在扩散模型中嵌入轻量控制模块，实现对非身份属性的独立操控，同时采用正交训练策略分离身份与控制信号，提升生成可控性和多样性，实验显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15313v1](http://arxiv.org/pdf/2505.15313v1)**

> **作者:** Kazuaki Mishima; Antoni Bigata Casademunt; Stavros Petridis; Maja Pantic; Kenji Suzuki
>
> **备注:** 9 pages(excluding references), 3 figures, 5 tables
>
> **摘要:** Human facial images encode a rich spectrum of information, encompassing both stable identity-related traits and mutable attributes such as pose, expression, and emotion. While recent advances in image generation have enabled high-quality identity-conditional face synthesis, precise control over non-identity attributes remains challenging, and disentangling identity from these mutable factors is particularly difficult. To address these limitations, we propose a novel identity-conditional diffusion model that introduces two lightweight control modules designed to independently manipulate facial pose, expression, and emotion without compromising identity preservation. These modules are embedded within the cross-attention layers of the base diffusion model, enabling precise attribute control with minimal parameter overhead. Furthermore, our tailored training strategy, which leverages cross-attention between the identity feature and each non-identity control feature, encourages identity features to remain orthogonal to control signals, enhancing controllability and diversity. Quantitative and qualitative evaluations, along with perceptual user studies, demonstrate that our method surpasses existing approaches in terms of control accuracy over pose, expression, and emotion, while also improving generative diversity under identity-only conditioning.
>
---
#### [new 066] Parameter-Efficient Fine-Tuning of Multispectral Foundation Models for Hyperspectral Image Classification
- **分类: cs.CV**

- **简介: 该论文针对多光谱基础模型在高光谱图像分类中的参数效率问题，提出基于SpectralGPT的微调框架，采用LoRA、KronA等PEFT方法并改进KronA+，实现高性能且参数仅0.056%，存储增加小，在五数据集验证有效。**

- **链接: [http://arxiv.org/pdf/2505.15334v1](http://arxiv.org/pdf/2505.15334v1)**

> **作者:** Bernardin Ligan; Khalide Jbilou; Fahd Kalloubi; Ahmed Ratnani
>
> **备注:** 33 pages, 14 figures
>
> **摘要:** Foundation models have achieved great success across diverse domains, including remote sensing (RS), thanks to their versatility and strong generalization abilities. However, most RS foundation models are designed for multispectral data, while hyperspectral imagery (HSI) - with its hundreds of spectral bands - remains less explored. Fine-tuning such models for downstream tasks is also challenging, often demanding considerable memory and storage. In this paper, we propose an efficient framework to fine-tune SpectralGPT, a multispectral foundation model, for hyperspectral image classification (HSIC). We explore several Parameter-Efficient Fine-Tuning (PEFT) methods, including Low-Rank Adaptation (LoRA), Kronecker-based adaptation (KronA), Low-Rank Kronecker (LoKr), and the recent LoRA+, which uses distinct learning rates for low-rank adapters scaled by a factor lambda. Inspired by LoRA+, we introduce KronA+, which applies a similar mechanism to the Kronecker matrices. We evaluate our approach on five datasets from different sensors, showing competitive performance with state-of-the-art HSI models. Our full fine-tuning (FFT) setup for SpectralGPT even outperforms a dedicated hyperspectral foundation model on some datasets while requiring only a quarter of the training epochs. Under the same number of epochs, KronA+ reaches similar performance with far fewer trainable parameters - just 0.056 percent - and adds only approximately 0.2 megabytes of storage, making it the most effective PEFT method tested.
>
---
#### [new 067] MORALISE: A Structured Benchmark for Moral Alignment in Visual Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.CY; cs.MM**

- **简介: 该论文提出MORALISE基准，评估视觉语言模型的道德对齐问题。针对现有方法依赖文本或AI生成数据导致的偏差，其基于Turiel理论构建13类道德主题，收集2481个真实图像文本对，设计道德判断与归因任务，测试19个模型，揭示当前技术的道德局限。**

- **链接: [http://arxiv.org/pdf/2505.14728v1](http://arxiv.org/pdf/2505.14728v1)**

> **作者:** Xiao Lin; Zhining Liu; Ze Yang; Gaotang Li; Ruizhong Qiu; Shuke Wang; Hui Liu; Haotian Li; Sumit Keswani; Vishwa Pardeshi; Huijun Zhao; Wei Fan; Hanghang Tong
>
> **备注:** 21 pages, 11 figures, 7 tables
>
> **摘要:** Warning: This paper contains examples of harmful language and images. Reader discretion is advised. Recently, vision-language models have demonstrated increasing influence in morally sensitive domains such as autonomous driving and medical analysis, owing to their powerful multimodal reasoning capabilities. As these models are deployed in high-stakes real-world applications, it is of paramount importance to ensure that their outputs align with human moral values and remain within moral boundaries. However, existing work on moral alignment either focuses solely on textual modalities or relies heavily on AI-generated images, leading to distributional biases and reduced realism. To overcome these limitations, we introduce MORALISE, a comprehensive benchmark for evaluating the moral alignment of vision-language models (VLMs) using diverse, expert-verified real-world data. We begin by proposing a comprehensive taxonomy of 13 moral topics grounded in Turiel's Domain Theory, spanning the personal, interpersonal, and societal moral domains encountered in everyday life. Built on this framework, we manually curate 2,481 high-quality image-text pairs, each annotated with two fine-grained labels: (1) topic annotation, identifying the violated moral topic(s), and (2) modality annotation, indicating whether the violation arises from the image or the text. For evaluation, we encompass two tasks, \textit{moral judgment} and \textit{moral norm attribution}, to assess models' awareness of moral violations and their reasoning ability on morally salient content. Extensive experiments on 19 popular open- and closed-source VLMs show that MORALISE poses a significant challenge, revealing persistent moral limitations in current state-of-the-art models. The full benchmark is publicly available at https://huggingface.co/datasets/Ze1025/MORALISE.
>
---
#### [new 068] GAMA: Geometry-Aware Manifold Alignment via Structured Adversarial Perturbations for Robust Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于领域自适应任务，针对源域与目标域流形差异大的问题，提出GAMA框架。通过几何引导的结构化对抗扰动，结合切空间探索与流形约束优化，实现显式流形对齐，提升跨域语义一致性与鲁棒性，理论和实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2505.15194v1](http://arxiv.org/pdf/2505.15194v1)**

> **作者:** Hana Satou; F Monkey
>
> **摘要:** Domain adaptation remains a challenge when there is significant manifold discrepancy between source and target domains. Although recent methods leverage manifold-aware adversarial perturbations to perform data augmentation, they often neglect precise manifold alignment and systematic exploration of structured perturbations. To address this, we propose GAMA (Geometry-Aware Manifold Alignment), a structured framework that achieves explicit manifold alignment via adversarial perturbation guided by geometric information. GAMA systematically employs tangent space exploration and manifold-constrained adversarial optimization, simultaneously enhancing semantic consistency, robustness to off-manifold deviations, and cross-domain alignment. Theoretical analysis shows that GAMA tightens the generalization bound via structured regularization and explicit alignment. Empirical results on DomainNet, VisDA, and Office-Home demonstrate that GAMA consistently outperforms existing adversarial and adaptation methods in both unsupervised and few-shot settings, exhibiting superior robustness, generalization, and manifold alignment capability.
>
---
#### [new 069] Discovering Pathology Rationale and Token Allocation for Efficient Multimodal Pathology Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态病理推理任务，针对现有方法推理能力不足及计算成本高的问题，提出双分支强化学习框架：一枝学习病理推理依据提升诊断能力，另一枝动态分配图像处理资源优化效率，在病理问答、癌症分型等任务中提升性能并降低70%推理成本。**

- **链接: [http://arxiv.org/pdf/2505.15687v1](http://arxiv.org/pdf/2505.15687v1)**

> **作者:** Zhe Xu; Cheng Jin; Yihui Wang; Ziyi Liu; Hao Chen
>
> **摘要:** Multimodal pathological image understanding has garnered widespread interest due to its potential to improve diagnostic accuracy and enable personalized treatment through integrated visual and textual data. However, existing methods exhibit limited reasoning capabilities, which hamper their ability to handle complex diagnostic scenarios. Additionally, the enormous size of pathological images leads to severe computational burdens, further restricting their practical deployment. To address these limitations, we introduce a novel bilateral reinforcement learning framework comprising two synergistic branches. One reinforcement branch enhances the reasoning capability by enabling the model to learn task-specific decision processes, i.e., pathology rationales, directly from labels without explicit reasoning supervision. While the other branch dynamically allocates a tailored number of tokens to different images based on both their visual content and task context, thereby optimizing computational efficiency. We apply our method to various pathological tasks such as visual question answering, cancer subtyping, and lesion detection. Extensive experiments show an average +41.7 absolute performance improvement with 70.3% lower inference costs over the base models, achieving both reasoning accuracy and computational efficiency.
>
---
#### [new 070] Objective Bicycle Occlusion Level Classification using a Deformable Parts-Based Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出基于可变形部件模型的自行车遮挡分级方法，解决现有主观评估无法客观量化遮挡程度的问题。通过定制图像检测管道，客观分析自行车部件可见性，建立遮挡水平新基准，提升自动驾驶中骑行者检测算法的评估精度。**

- **链接: [http://arxiv.org/pdf/2505.15358v1](http://arxiv.org/pdf/2505.15358v1)**

> **作者:** Angelique Mangubat; Shane Gilroy
>
> **摘要:** Road safety is a critical challenge, particularly for cyclists, who are among the most vulnerable road users. This study aims to enhance road safety by proposing a novel benchmark for bicycle occlusion level classification using advanced computer vision techniques. Utilizing a parts-based detection model, images are annotated and processed through a custom image detection pipeline. A novel method of bicycle occlusion level is proposed to objectively quantify the visibility and occlusion level of bicycle semantic parts. The findings indicate that the model robustly quantifies the visibility and occlusion level of bicycles, a significant improvement over the subjective methods used by the current state of the art. Widespread use of the proposed methodology will facilitate the accurate performance reporting of cyclist detection algorithms for occluded cyclists, informing the development of more robust vulnerable road user detection methods for autonomous vehicles.
>
---
#### [new 071] Pura: An Efficient Privacy-Preserving Solution for Face Recognition
- **分类: cs.CV; cs.CR**

- **简介: 论文提出Pura，解决隐私保护与效率问题。采用阈值Paillier加密系统构建非交互式架构，设计安全计算协议并引入并行机制，实现加密数据上高效人脸识别，隐私完全保护且速度提升16倍。**

- **链接: [http://arxiv.org/pdf/2505.15476v1](http://arxiv.org/pdf/2505.15476v1)**

> **作者:** Guotao Xu; Bowen Zhao; Yang Xiao; Yantao Zhong; Liang Zhai; Qingqi Pei
>
> **摘要:** Face recognition is an effective technology for identifying a target person by facial images. However, sensitive facial images raises privacy concerns. Although privacy-preserving face recognition is one of potential solutions, this solution neither fully addresses the privacy concerns nor is efficient enough. To this end, we propose an efficient privacy-preserving solution for face recognition, named Pura, which sufficiently protects facial privacy and supports face recognition over encrypted data efficiently. Specifically, we propose a privacy-preserving and non-interactive architecture for face recognition through the threshold Paillier cryptosystem. Additionally, we carefully design a suite of underlying secure computing protocols to enable efficient operations of face recognition over encrypted data directly. Furthermore, we introduce a parallel computing mechanism to enhance the performance of the proposed secure computing protocols. Privacy analysis demonstrates that Pura fully safeguards personal facial privacy. Experimental evaluations demonstrate that Pura achieves recognition speeds up to 16 times faster than the state-of-the-art.
>
---
#### [new 072] A Taxonomy of Structure from Motion Methods
- **分类: cs.CV**

- **简介: 该论文属于结构光运动（SfM）方法分类任务，系统梳理现有技术，按方法侧重于结构或运动分为三类，分析其理论条件，提出分类框架以明确开放问题及未来方向。**

- **链接: [http://arxiv.org/pdf/2505.15814v1](http://arxiv.org/pdf/2505.15814v1)**

> **作者:** Federica Arrigoni
>
> **摘要:** Structure from Motion (SfM) refers to the problem of recovering both structure (i.e., 3D coordinates of points in the scene) and motion (i.e., camera matrices) starting from point correspondences in multiple images. It has attracted significant attention over the years, counting practical reconstruction pipelines as well as theoretical results. This paper is conceived as a conceptual review of SfM methods, which are grouped into three main categories, according to which part of the problem - between motion and structure - they focus on. The proposed taxonomy brings a new perspective on existing SfM approaches as well as insights into open problems and possible future research directions. Particular emphasis is given on identifying the theoretical conditions that make SfM well posed, which depend on the problem formulation that is being considered.
>
---
#### [new 073] Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉模型优化任务，旨在解决Vision Transformer计算与资源消耗高的问题。提出基于预训练模型的两阶段方法提取MoE专家子网络：先聚类MLP层的激活模式，再提取对应子网络。实验证明可减少36%计算量和32%模型大小，微调后恢复98%原性能。**

- **链接: [http://arxiv.org/pdf/2505.15414v1](http://arxiv.org/pdf/2505.15414v1)**

> **作者:** Uranik Berisha; Jens Mehnert; Alexandru Paul Condurache
>
> **备注:** Accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025
>
> **摘要:** Vision Transformers have emerged as the state-of-the-art models in various Computer Vision tasks, but their high computational and resource demands pose significant challenges. While Mixture-of-Experts (MoE) can make these models more efficient, they often require costly retraining or even training from scratch. Recent developments aim to reduce these computational costs by leveraging pretrained networks. These have been shown to produce sparse activation patterns in the Multi-Layer Perceptrons (MLPs) of the encoder blocks, allowing for conditional activation of only relevant subnetworks for each sample. Building on this idea, we propose a new method to construct MoE variants from pretrained models. Our approach extracts expert subnetworks from the model's MLP layers post-training in two phases. First, we cluster output activations to identify distinct activation patterns. In the second phase, we use these clusters to extract the corresponding subnetworks responsible for producing them. On ImageNet-1k recognition tasks, we demonstrate that these extracted experts can perform surprisingly well out of the box and require only minimal fine-tuning to regain 98% of the original performance, all while reducing MACs and model size, by up to 36% and 32% respectively.
>
---
#### [new 074] STAR-R1: Spacial TrAnsformation Reasoning by Reinforcing Multimodal LLMs
- **分类: cs.CV**

- **简介: 论文针对多模态大模型在空间推理（TVR任务）中的跨视角推理缺陷，提出STAR-R1框架，整合单阶段强化学习与精细奖励机制，通过奖励部分正确推理、抑制冗余枚举，提升探索效率与推理精度，在11项指标超传统方法23%。**

- **链接: [http://arxiv.org/pdf/2505.15804v1](http://arxiv.org/pdf/2505.15804v1)**

> **作者:** Zongzhao Li; Zongyang Ma; Mingze Li; Songyou Li; Yu Rong; Tingyang Xu; Ziqi Zhang; Deli Zhao; Wenbing Huang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse tasks, yet they lag significantly behind humans in spatial reasoning. We investigate this gap through Transformation-Driven Visual Reasoning (TVR), a challenging task requiring identification of object transformations across images under varying viewpoints. While traditional Supervised Fine-Tuning (SFT) fails to generate coherent reasoning paths in cross-view settings, sparse-reward Reinforcement Learning (RL) suffers from inefficient exploration and slow convergence. To address these limitations, we propose STAR-R1, a novel framework that integrates a single-stage RL paradigm with a fine-grained reward mechanism tailored for TVR. Specifically, STAR-R1 rewards partial correctness while penalizing excessive enumeration and passive inaction, enabling efficient exploration and precise reasoning. Comprehensive evaluations demonstrate that STAR-R1 achieves state-of-the-art performance across all 11 metrics, outperforming SFT by 23% in cross-view scenarios. Further analysis reveals STAR-R1's anthropomorphic behavior and highlights its unique ability to compare all objects for improving spatial reasoning. Our work provides critical insights in advancing the research of MLLMs and reasoning models. The codes, model weights, and data will be publicly available at https://github.com/zongzhao23/STAR-R1.
>
---
#### [new 075] Seeing Through Deception: Uncovering Misleading Creator Intent in Multimodal News with Vision-Language Models
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于多模态虚假信息检测任务，旨在解决现有视觉语言模型（VLMs）难以识别误导性创作者意图的问题。提出通过模拟新闻创作过程构建含12000样本的DeceptionDecoded数据集，评估14种模型在检测误导意图、归因来源及推理创作者意图上的表现，发现其依赖表面线索，强调需开发意图感知模型以提升深度推理能力。**

- **链接: [http://arxiv.org/pdf/2505.15489v1](http://arxiv.org/pdf/2505.15489v1)**

> **作者:** Jiaying Wu; Fanxiao Li; Min-Yen Kan; Bryan Hooi
>
> **摘要:** The real-world impact of misinformation stems from the underlying misleading narratives that creators seek to convey. As such, interpreting misleading creator intent is essential for multimodal misinformation detection (MMD) systems aimed at effective information governance. In this paper, we introduce an automated framework that simulates real-world multimodal news creation by explicitly modeling creator intent through two components: the desired influence and the execution plan. Using this framework, we construct DeceptionDecoded, a large-scale benchmark comprising 12,000 image-caption pairs aligned with trustworthy reference articles. The dataset captures both misleading and non-misleading intents and spans manipulations across visual and textual modalities. We conduct a comprehensive evaluation of 14 state-of-the-art vision-language models (VLMs) on three intent-centric tasks: (1) misleading intent detection, (2) misleading source attribution, and (3) creator desire inference. Despite recent advances, we observe that current VLMs fall short in recognizing misleading intent, often relying on spurious cues such as superficial cross-modal consistency, stylistic signals, and heuristic authenticity hints. Our findings highlight the pressing need for intent-aware modeling in MMD and open new directions for developing systems capable of deeper reasoning about multimodal misinformation.
>
---
#### [new 076] Stronger ViTs With Octic Equivariance
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于计算机视觉模型优化任务，旨在提升Vision Transformer（ViT）的计算效率与性能。通过引入八面体群（反射及90度旋转）的等变性作为归纳偏置，提出octic-ViT架构，使用等变层设计，并在监督/自监督学习（如DeiT-III、DINOv2）中验证。结果表明，在ImageNet-1K上，octic-ViT-H较原模型FLOPs减少40%，同时提升分类与分割精度。**

- **链接: [http://arxiv.org/pdf/2505.15441v1](http://arxiv.org/pdf/2505.15441v1)**

> **作者:** David Nordström; Johan Edstedt; Fredrik Kahl; Georg Bökman
>
> **摘要:** Recent efforts at scaling computer vision models have established Vision Transformers (ViTs) as the leading architecture. ViTs incorporate weight sharing over image patches as an important inductive bias. In this work, we show that ViTs benefit from incorporating equivariance under the octic group, i.e., reflections and 90-degree rotations, as a further inductive bias. We develop new architectures, octic ViTs, that use octic-equivariant layers and put them to the test on both supervised and self-supervised learning. Through extensive experiments on DeiT-III and DINOv2 training on ImageNet-1K, we show that octic ViTs yield more computationally efficient networks while also improving performance. In particular, we achieve approximately 40% reduction in FLOPs for ViT-H while simultaneously improving both classification and segmentation results.
>
---
#### [new 077] Better Safe Than Sorry? Overreaction Problem of Vision Language Models in Visual Emergency Recognition
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究视觉语言模型（VLM）在紧急场景识别中的可靠性，旨在解决其过度反应问题。构建VERI数据集（200张对比图像），评估14种VLM模型，发现其虽能有效识别真实紧急情况（70-100%成功率），但误报率达31-96%，主要因上下文过度解读。结果表明模型规模无法缓解此问题，需改进安全评估方法。**

- **链接: [http://arxiv.org/pdf/2505.15367v1](http://arxiv.org/pdf/2505.15367v1)**

> **作者:** Dasol Choi; Seunghyun Lee; Youngsook Song
>
> **备注:** 13 pages
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated impressive capabilities in understanding visual content, but their reliability in safety-critical contexts remains under-explored. We introduce VERI (Visual Emergency Recognition Dataset), a carefully designed diagnostic benchmark of 200 images (100 contrastive pairs). Each emergency scene is matched with a visually similar but safe counterpart through multi-stage human verification and iterative refinement. Using a two-stage protocol - risk identification and emergency response - we evaluate 14 VLMs (2B-124B parameters) across medical emergencies, accidents, and natural disasters. Our analysis reveals a systematic overreaction problem: models excel at identifying real emergencies (70-100 percent success rate) but suffer from an alarming rate of false alarms, misidentifying 31-96 percent of safe situations as dangerous, with 10 scenarios failed by all models regardless of scale. This "better-safe-than-sorry" bias manifests primarily through contextual overinterpretation (88-93 percent of errors), challenging VLMs' reliability for safety applications. These findings highlight persistent limitations that are not resolved by increasing model scale, motivating targeted approaches for improving contextual safety assessment in visually misleading scenarios.
>
---
#### [new 078] Multimodal Conditional Information Bottleneck for Generalizable AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 论文提出InfoFD框架，解决CLIP-based检测方法因特征冗余和文本提示多样性导致的泛化不足问题。通过文本引导的条件信息瓶颈（TGCIB）与动态文本正交化（DTO），优化多模态特征，利用文本与图像相似度偏差提升AI生成图像检测的泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.15217v1](http://arxiv.org/pdf/2505.15217v1)**

> **作者:** Haotian Qin; Dongliang Chang; Yueying Gao; Bingyao Yu; Lei Chen; Zhanyu Ma
>
> **备注:** 24 pages, 16 figures
>
> **摘要:** Although existing CLIP-based methods for detecting AI-generated images have achieved promising results, they are still limited by severe feature redundancy, which hinders their generalization ability. To address this issue, incorporating an information bottleneck network into the task presents a straightforward solution. However, relying solely on image-corresponding prompts results in suboptimal performance due to the inherent diversity of prompts. In this paper, we propose a multimodal conditional bottleneck network to reduce feature redundancy while enhancing the discriminative power of features extracted by CLIP, thereby improving the model's generalization ability. We begin with a semantic analysis experiment, where we observe that arbitrary text features exhibit lower cosine similarity with real image features than with fake image features in the CLIP feature space, a phenomenon we refer to as "bias". Therefore, we introduce InfoFD, a text-guided AI-generated image detection framework. InfoFD consists of two key components: the Text-Guided Conditional Information Bottleneck (TGCIB) and Dynamic Text Orthogonalization (DTO). TGCIB improves the generalizability of learned representations by conditioning on both text and class modalities. DTO dynamically updates weighted text features, preserving semantic information while leveraging the global "bias". Our model achieves exceptional generalization performance on the GenImage dataset and latest generative models. Our code is available at https://github.com/Ant0ny44/InfoFD.
>
---
#### [new 079] ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频时间定位任务，解决现有方法依赖高成本标注或启发式策略导致帧选择效率低的问题。提出ViaRL框架，采用强化学习与迭代放大策略，通过下游模型准确率作为奖励优化帧选择，提升性能且无需昂贵标注。**

- **链接: [http://arxiv.org/pdf/2505.15447v1](http://arxiv.org/pdf/2505.15447v1)**

> **作者:** Ziqiang Xu; Qi Dai; Tian Xie; Yifan Yang; Kai Qiu; DongDong Chen; Zuxuan Wu; Chong Luo
>
> **摘要:** Video understanding is inherently intention-driven-humans naturally focus on relevant frames based on their goals. Recent advancements in multimodal large language models (MLLMs) have enabled flexible query-driven reasoning; however, video-based frameworks like Video Chain-of-Thought lack direct training signals to effectively identify relevant frames. Current approaches often rely on heuristic methods or pseudo-label supervised annotations, which are both costly and limited in scalability across diverse scenarios. To overcome these challenges, we introduce ViaRL, the first framework to leverage rule-based reinforcement learning (RL) for optimizing frame selection in intention-driven video understanding. An iterated amplification strategy is adopted to perform alternating cyclic training in the video CoT system, where each component undergoes iterative cycles of refinement to improve its capabilities. ViaRL utilizes the answer accuracy of a downstream model as a reward signal to train a frame selector through trial-and-error, eliminating the need for expensive annotations while closely aligning with human-like learning processes. Comprehensive experiments across multiple benchmarks, including VideoMME, LVBench, and MLVU, demonstrate that ViaRL consistently delivers superior temporal grounding performance and robust generalization across diverse video understanding tasks, highlighting its effectiveness and scalability. Notably, ViaRL achieves a nearly 15\% improvement on Needle QA, a subset of MLVU, which is required to search a specific needle within a long video and regarded as one of the most suitable benchmarks for evaluating temporal grounding.
>
---
#### [new 080] BadSR: Stealthy Label Backdoor Attacks on Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出BadSR方法，针对图像超分辨率模型的标签后门攻击，解决现有攻击忽视中毒高分辨率图像隐蔽性的问题。通过特征空间对齐约束修改范围，并设计对抗触发器及遗传算法优化样本选择，提升攻击隐蔽性和成功率。**

- **链接: [http://arxiv.org/pdf/2505.15308v1](http://arxiv.org/pdf/2505.15308v1)**

> **作者:** Ji Guo; Xiaolei Wen; Wenbo Jiang; Cheng Huang; Jinjin Li; Hongwei Li
>
> **摘要:** With the widespread application of super-resolution (SR) in various fields, researchers have begun to investigate its security. Previous studies have demonstrated that SR models can also be subjected to backdoor attacks through data poisoning, affecting downstream tasks. A backdoor SR model generates an attacker-predefined target image when given a triggered image while producing a normal high-resolution (HR) output for clean images. However, prior backdoor attacks on SR models have primarily focused on the stealthiness of poisoned low-resolution (LR) images while ignoring the stealthiness of poisoned HR images, making it easy for users to detect anomalous data. To address this problem, we propose BadSR, which improves the stealthiness of poisoned HR images. The key idea of BadSR is to approximate the clean HR image and the pre-defined target image in the feature space while ensuring that modifications to the clean HR image remain within a constrained range. The poisoned HR images generated by BadSR can be integrated with existing triggers. To further improve the effectiveness of BadSR, we design an adversarially optimized trigger and a backdoor gradient-driven poisoned sample selection method based on a genetic algorithm. The experimental results show that BadSR achieves a high attack success rate in various models and data sets, significantly affecting downstream tasks.
>
---
#### [new 081] Leveraging the Powerful Attention of a Pre-trained Diffusion Model for Exemplar-based Image Colorization
- **分类: cs.CV**

- **简介: 该论文属于基于示例的图像着色任务，旨在通过参考图精准匹配输入灰度图的语义区域以实现高质量上色。针对传统方法语义对齐不足的问题，提出双注意力引导颜色迁移（分别计算灰度与参考图注意力图提升语义对齐）和分类器自由着色增强（融合两种输出优化颜色质量），基于预训练扩散模型注意力机制无需微调，实验显示效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15812v1](http://arxiv.org/pdf/2505.15812v1)**

> **作者:** Satoshi Kosugi
>
> **备注:** Accepted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)
>
> **摘要:** Exemplar-based image colorization aims to colorize a grayscale image using a reference color image, ensuring that reference colors are applied to corresponding input regions based on their semantic similarity. To achieve accurate semantic matching between regions, we leverage the self-attention module of a pre-trained diffusion model, which is trained on a large dataset and exhibits powerful attention capabilities. To harness this power, we propose a novel, fine-tuning-free approach based on a pre-trained diffusion model, making two key contributions. First, we introduce dual attention-guided color transfer. We utilize the self-attention module to compute an attention map between the input and reference images, effectively capturing semantic correspondences. The color features from the reference image is then transferred to the semantically matching regions of the input image, guided by this attention map, and finally, the grayscale features are replaced with the corresponding color features. Notably, we utilize dual attention to calculate attention maps separately for the grayscale and color images, achieving more precise semantic alignment. Second, we propose classifier-free colorization guidance, which enhances the transferred colors by combining color-transferred and non-color-transferred outputs. This process improves the quality of colorization. Our experimental results demonstrate that our method outperforms existing techniques in terms of image quality and fidelity to the reference. Specifically, we use 335 input-reference pairs from previous research, achieving an FID of 95.27 (image quality) and an SI-FID of 5.51 (fidelity to the reference). Our source code is available at https://github.com/satoshi-kosugi/powerful-attention.
>
---
#### [new 082] Colors Matter: AI-Driven Exploration of Human Feature Colors
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人体特征颜色分类任务，解决复杂光照下精准识别肤色、发色、虹膜及静脉色调的挑战。提出多阶段AI框架，结合X-means聚类、Delta E算法及HSV/LAB颜色空间分析，通过区域分割与色彩匹配实现80%分类准确率，支持美妆科技与个性化应用。**

- **链接: [http://arxiv.org/pdf/2505.14931v1](http://arxiv.org/pdf/2505.14931v1)**

> **作者:** Rama Alyoubi; Taif Alharbi; Albatul Alghamdi; Yara Alshehri; Elham Alghamdi
>
> **摘要:** This study presents a robust framework that leverages advanced imaging techniques and machine learning for feature extraction and classification of key human attributes-namely skin tone, hair color, iris color, and vein-based undertones. The system employs a multi-stage pipeline involving face detection, region segmentation, and dominant color extraction to isolate and analyze these features. Techniques such as X-means clustering, alongside perceptually uniform distance metrics like Delta E (CIEDE2000), are applied within both LAB and HSV color spaces to enhance the accuracy of color differentiation. For classification, the dominant tones of the skin, hair, and iris are extracted and matched to a custom tone scale, while vein analysis from wrist images enables undertone classification into "Warm" or "Cool" based on LAB differences. Each module uses targeted segmentation and color space transformations to ensure perceptual precision. The system achieves up to 80% accuracy in tone classification using the Delta E-HSV method with Gaussian blur, demonstrating reliable performance across varied lighting and image conditions. This work highlights the potential of AI-powered color analysis and feature extraction for delivering inclusive, precise, and nuanced classification, supporting applications in beauty technology, digital personalization, and visual analytics.
>
---
#### [new 083] Leveraging Foundation Models for Multimodal Graph-Based Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于多模态动作识别任务，旨在解决精细双手操作动作的识别挑战。提出动态多模态图框架，整合VideoMAE（视觉）和BERT（文本），通过节点（帧/物体/文本）与动态边（空间/时间/语义关系）建模，并采用图注意力网络优化语义推理，提升动作识别的准确性和泛化性。**

- **链接: [http://arxiv.org/pdf/2505.15192v1](http://arxiv.org/pdf/2505.15192v1)**

> **作者:** Fatemeh Ziaeetabar; Florentin Wörgötter
>
> **摘要:** Foundation models have ushered in a new era for multimodal video understanding by enabling the extraction of rich spatiotemporal and semantic representations. In this work, we introduce a novel graph-based framework that integrates a vision-language foundation, leveraging VideoMAE for dynamic visual encoding and BERT for contextual textual embedding, to address the challenge of recognizing fine-grained bimanual manipulation actions. Departing from conventional static graph architectures, our approach constructs an adaptive multimodal graph where nodes represent frames, objects, and textual annotations, and edges encode spatial, temporal, and semantic relationships. These graph structures evolve dynamically based on learned interactions, allowing for flexible and context-aware reasoning. A task-specific attention mechanism within a Graph Attention Network further enhances this reasoning by modulating edge importance based on action semantics. Through extensive evaluations on diverse benchmark datasets, we demonstrate that our method consistently outperforms state-of-the-art baselines, underscoring the strength of combining foundation models with dynamic graph-based reasoning for robust and generalizable action recognition.
>
---
#### [new 084] gen2seg: Generative Models Enable Generalizable Instance Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于实例分割任务，旨在解决现有模型难以泛化到未见过对象类型和风格的问题。研究通过微调生成模型（如Stable Diffusion和MAE），仅在有限类别（家具、汽车）上使用实例着色损失，使其具备零样本泛化能力，能准确分割新类型/风格对象，性能接近强监督模型SAM，且更优处理细结构。结果表明生成模型的内在分组机制可跨领域迁移。**

- **链接: [http://arxiv.org/pdf/2505.15263v1](http://arxiv.org/pdf/2505.15263v1)**

> **作者:** Om Khangaonkar; Hamed Pirsiavash
>
> **备注:** Website: https://reachomk.github.io/gen2seg/
>
> **摘要:** By pretraining to synthesize coherent images from perturbed inputs, generative models inherently learn to understand object boundaries and scene compositions. How can we repurpose these generative representations for general-purpose perceptual organization? We finetune Stable Diffusion and MAE (encoder+decoder) for category-agnostic instance segmentation using our instance coloring loss exclusively on a narrow set of object types (indoor furnishings and cars). Surprisingly, our models exhibit strong zero-shot generalization, accurately segmenting objects of types and styles unseen in finetuning (and in many cases, MAE's ImageNet-1K pretraining too). Our best-performing models closely approach the heavily supervised SAM when evaluated on unseen object types and styles, and outperform it when segmenting fine structures and ambiguous boundaries. In contrast, existing promptable segmentation architectures or discriminatively pretrained models fail to generalize. This suggests that generative models learn an inherent grouping mechanism that transfers across categories and domains, even without internet-scale pretraining. Code, pretrained models, and demos are available on our website.
>
---
#### [new 085] R3GS: Gaussian Splatting for Robust Reconstruction and Relocalization in Unconstrained Image Collections
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出R3GS框架，用于无约束图像集合的鲁棒三维重建与重定位。解决瞬态物体干扰、天空区域误差及光照变化下的重定位问题。方法结合CNN全局特征与哈希网格局部特征，预测高斯属性；引入轻量检测网络生成可见性图抑制瞬态物体，采用天空球面表示减少重建误差，并设计光照鲁棒重定位模块。提升渲染效率与保真度，降低存储需求。**

- **链接: [http://arxiv.org/pdf/2505.15294v1](http://arxiv.org/pdf/2505.15294v1)**

> **作者:** Xu yan; Zhaohui Wang; Rong Wei; Jingbo Yu; Dong Li; Xiangde Liu
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** We propose R3GS, a robust reconstruction and relocalization framework tailored for unconstrained datasets. Our method uses a hybrid representation during training. Each anchor combines a global feature from a convolutional neural network (CNN) with a local feature encoded by the multiresolution hash grids [2]. Subsequently, several shallow multi-layer perceptrons (MLPs) predict the attributes of each Gaussians, including color, opacity, and covariance. To mitigate the adverse effects of transient objects on the reconstruction process, we ffne-tune a lightweight human detection network. Once ffne-tuned, this network generates a visibility map that efffciently generalizes to other transient objects (such as posters, banners, and cars) with minimal need for further adaptation. Additionally, to address the challenges posed by sky regions in outdoor scenes, we propose an effective sky-handling technique that incorporates a depth prior as a constraint. This allows the inffnitely distant sky to be represented on the surface of a large-radius sky sphere, signiffcantly reducing ffoaters caused by errors in sky reconstruction. Furthermore, we introduce a novel relocalization method that remains robust to changes in lighting conditions while estimating the camera pose of a given image within the reconstructed 3DGS scene. As a result, R3GS significantly enhances rendering ffdelity, improves both training and rendering efffciency, and reduces storage requirements. Our method achieves state-of-the-art performance compared to baseline methods on in-the-wild datasets. The code will be made open-source following the acceptance of the paper.
>
---
#### [new 086] VARD: Efficient and Dense Fine-Tuning for Diffusion Models with Value-based RL
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于扩散模型微调任务，解决现有RL方法在稳定高效优化及非可微奖励支持上的不足。提出VARD方法，通过学习预测中间状态奖励期望的价值函数，并结合KL正则化提供全程密集监督，在保持预训练模型特性的同时提升生成质量和训练效率。**

- **链接: [http://arxiv.org/pdf/2505.15791v1](http://arxiv.org/pdf/2505.15791v1)**

> **作者:** Fengyuan Dai; Zifeng Zhuang; Yufei Huang; Siteng Huang; Bangyan Liao; Donglin Wang; Fajie Yuan
>
> **备注:** Under review
>
> **摘要:** Diffusion models have emerged as powerful generative tools across various domains, yet tailoring pre-trained models to exhibit specific desirable properties remains challenging. While reinforcement learning (RL) offers a promising solution,current methods struggle to simultaneously achieve stable, efficient fine-tuning and support non-differentiable rewards. Furthermore, their reliance on sparse rewards provides inadequate supervision during intermediate steps, often resulting in suboptimal generation quality. To address these limitations, dense and differentiable signals are required throughout the diffusion process. Hence, we propose VAlue-based Reinforced Diffusion (VARD): a novel approach that first learns a value function predicting expection of rewards from intermediate states, and subsequently uses this value function with KL regularization to provide dense supervision throughout the generation process. Our method maintains proximity to the pretrained model while enabling effective and stable training via backpropagation. Experimental results demonstrate that our approach facilitates better trajectory guidance, improves training efficiency and extends the applicability of RL to diffusion models optimized for complex, non-differentiable reward functions.
>
---
#### [new 087] The P$^3$ dataset: Pixels, Points and Polygons for Multimodal Building Vectorization
- **分类: cs.CV**

- **简介: 该论文提出P³数据集，用于多模态建筑矢量化任务。针对现有数据集侧重图像缺乏3D信息的问题，整合航拍LiDAR点云、高分辨率影像和建筑矢量数据，包含百亿级高精度点云及25cm分辨率图像。实验表明LiDAR有效提升多边形预测，融合多模态进一步提高精度和几何质量，并公开数据及预训练模型。**

- **链接: [http://arxiv.org/pdf/2505.15379v1](http://arxiv.org/pdf/2505.15379v1)**

> **作者:** Raphael Sulzer; Liuyun Duan; Nicolas Girard; Florent Lafarge
>
> **摘要:** We present the P$^3$ dataset, a large-scale multimodal benchmark for building vectorization, constructed from aerial LiDAR point clouds, high-resolution aerial imagery, and vectorized 2D building outlines, collected across three continents. The dataset contains over 10 billion LiDAR points with decimeter-level accuracy and RGB images at a ground sampling distance of 25 centimeter. While many existing datasets primarily focus on the image modality, P$^3$ offers a complementary perspective by also incorporating dense 3D information. We demonstrate that LiDAR point clouds serve as a robust modality for predicting building polygons, both in hybrid and end-to-end learning frameworks. Moreover, fusing aerial LiDAR and imagery further improves accuracy and geometric quality of predicted polygons. The P$^3$ dataset is publicly available, along with code and pretrained weights of three state-of-the-art models for building polygon prediction at https://github.com/raphaelsulzer/PixelsPointsPolygons .
>
---
#### [new 088] AuxDet: Auxiliary Metadata Matters for Omni-Domain Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文针对跨领域红外小目标检测任务，提出AuxDet模型，解决现有方法因忽略元数据（如光谱、传感器等）导致的泛化性差问题。通过融合文本元数据与视觉特征的多模态框架，结合MLP融合与轻量级增强模块，提升复杂场景下的检测鲁棒性与精度。**

- **链接: [http://arxiv.org/pdf/2505.15184v1](http://arxiv.org/pdf/2505.15184v1)**

> **作者:** Yangting Shi; Renjie He; Le Hui; Xiang Li; Jian Yang; Ming-Ming Cheng; Yimian Dai
>
> **摘要:** Omni-domain infrared small target detection (IRSTD) poses formidable challenges, as a single model must seamlessly adapt to diverse imaging systems, varying resolutions, and multiple spectral bands simultaneously. Current approaches predominantly rely on visual-only modeling paradigms that not only struggle with complex background interference and inherently scarce target features, but also exhibit limited generalization capabilities across complex omni-scene environments where significant domain shifts and appearance variations occur. In this work, we reveal a critical oversight in existing paradigms: the neglect of readily available auxiliary metadata describing imaging parameters and acquisition conditions, such as spectral bands, sensor platforms, resolution, and observation perspectives. To address this limitation, we propose the Auxiliary Metadata Driven Infrared Small Target Detector (AuxDet), a novel multi-modal framework that fundamentally reimagines the IRSTD paradigm by incorporating textual metadata for scene-aware optimization. Through a high-dimensional fusion module based on multi-layer perceptrons (MLPs), AuxDet dynamically integrates metadata semantics with visual features, guiding adaptive representation learning for each individual sample. Additionally, we design a lightweight prior-initialized enhancement module using 1D convolutional blocks to further refine fused features and recover fine-grained target cues. Extensive experiments on the challenging WideIRSTD-Full benchmark demonstrate that AuxDet consistently outperforms state-of-the-art methods, validating the critical role of auxiliary information in improving robustness and accuracy in omni-domain IRSTD tasks. Code is available at https://github.com/GrokCV/AuxDet.
>
---
#### [new 089] From Pixels to Images: Deep Learning Advances in Remote Sensing Image Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于遥感图像语义分割任务，解决传统方法处理RSIs效率与精度不足的问题。通过将深度学习方法分为像素、块、图像级等四阶段，分析特征提取与学习策略的演变，评估近40种技术性能，总结进展并指明挑战。**

- **链接: [http://arxiv.org/pdf/2505.15147v1](http://arxiv.org/pdf/2505.15147v1)**

> **作者:** Quanwei Liu; Tao Huang; Yanni Dong; Jiaqi Yang; Wei Xiang
>
> **备注:** 38 pages, 14 figures, 10 tables
>
> **摘要:** Remote sensing images (RSIs) capture both natural and human-induced changes on the Earth's surface, serving as essential data for environmental monitoring, urban planning, and resource management. Semantic segmentation (SS) of RSIs enables the fine-grained interpretation of surface features, making it a critical task in remote sensing analysis. With the increasing diversity and volume of RSIs collected by sensors on various platforms, traditional processing methods struggle to maintain efficiency and accuracy. In response, deep learning (DL) has emerged as a transformative approach, enabling substantial advances in remote sensing image semantic segmentation (RSISS) by automating feature extraction and improving segmentation accuracy across diverse modalities. This paper revisits the evolution of DL-based RSISS by categorizing existing approaches into four stages: the early pixel-based methods, the prevailing patch-based and tile-based techniques, and the emerging image-based strategies enabled by foundation models. We analyze these developments from the perspective of feature extraction and learning strategies, revealing the field's progression from pixel-level to tile-level and from unimodal to multimodal segmentation. Furthermore, we conduct a comprehensive evaluation of nearly 40 advanced techniques on a unified dataset to quantitatively characterize their performance and applicability. This review offers a holistic view of DL-based SS for RS, highlighting key advancements, comparative insights, and open challenges to guide future research.
>
---
#### [new 090] Streamline Without Sacrifice -- Squeeze out Computation Redundancy in LMM
- **分类: cs.CV**

- **简介: 该论文属于多模态模型优化任务，旨在解决视觉token计算冗余问题。提出ProxyV方法，通过代理token减少视觉计算，提升效率且不降性能，可结合token削减方法进一步优化。**

- **链接: [http://arxiv.org/pdf/2505.15816v1](http://arxiv.org/pdf/2505.15816v1)**

> **作者:** Penghao Wu; Lewei Lu; Ziwei Liu
>
> **备注:** ICML 2025
>
> **摘要:** Large multimodal models excel in multimodal tasks but face significant computational challenges due to excessive computation on visual tokens. Unlike token reduction methods that focus on token-level redundancy, we identify and study the computation-level redundancy on vision tokens to ensure no information loss. Our key insight is that vision tokens from the pretrained vision encoder do not necessarily require all the heavy operations (e.g., self-attention, FFNs) in decoder-only LMMs and could be processed more lightly with proper designs. We designed a series of experiments to discover and progressively squeeze out the vision-related computation redundancy. Based on our findings, we propose ProxyV, a novel approach that utilizes proxy vision tokens to alleviate the computational burden on original vision tokens. ProxyV enhances efficiency without compromising performance and can even yield notable performance gains in scenarios with more moderate efficiency improvements. Furthermore, the flexibility of ProxyV is demonstrated through its combination with token reduction methods to boost efficiency further. The code will be made public at this https://github.com/penghao-wu/ProxyV URL.
>
---
#### [new 091] Continuous Representation Methods, Theories, and Applications: An Overview and Perspectives
- **分类: cs.CV**

- **简介: 该论文综述连续表示方法，解决传统离散框架在数据表示与重建中的分辨率固定、跨模态适应性差等问题。系统分析了方法设计（如基函数、神经隐式表示）、理论基础（误差、收敛性）及应用（图像修复、遥感等），并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2505.15222v1](http://arxiv.org/pdf/2505.15222v1)**

> **作者:** Yisi Luo; Xile Zhao; Deyu Meng
>
> **摘要:** Recently, continuous representation methods emerge as novel paradigms that characterize the intrinsic structures of real-world data through function representations that map positional coordinates to their corresponding values in the continuous space. As compared with the traditional discrete framework, the continuous framework demonstrates inherent superiority for data representation and reconstruction (e.g., image restoration, novel view synthesis, and waveform inversion) by offering inherent advantages including resolution flexibility, cross-modal adaptability, inherent smoothness, and parameter efficiency. In this review, we systematically examine recent advancements in continuous representation frameworks, focusing on three aspects: (i) Continuous representation method designs such as basis function representation, statistical modeling, tensor function decomposition, and implicit neural representation; (ii) Theoretical foundations of continuous representations such as approximation error analysis, convergence property, and implicit regularization; (iii) Real-world applications of continuous representations derived from computer vision, graphics, bioinformatics, and remote sensing. Furthermore, we outline future directions and perspectives to inspire exploration and deepen insights to facilitate continuous representation methods, theories, and applications. All referenced works are summarized in our open-source repository: https://github.com/YisiLuo/Continuous-Representation-Zoo.
>
---
#### [new 092] FragFake: A Dataset for Fine-Grained Detection of Edited Images with Vision Language Models
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文提出FragFake数据集及基于视觉语言模型（VLM）的检测方法，解决现有编辑图像检测缺乏定位能力、依赖高成本标注及数据不足的问题。通过自动化生成 pipeline 创建高质量编辑图像数据集，并首次利用微调VLM实现编辑分类与定位，实验显示其性能优于预训练模型，为多模态内容真实性研究提供新范式。**

- **链接: [http://arxiv.org/pdf/2505.15644v1](http://arxiv.org/pdf/2505.15644v1)**

> **作者:** Zhen Sun; Ziyi Zhang; Zeren Luo; Zeyang Sha; Tianshuo Cong; Zheng Li; Shiwen Cui; Weiqiang Wang; Jiaheng Wei; Xinlei He; Qi Li; Qian Wang
>
> **备注:** 14pages,15 figures
>
> **摘要:** Fine-grained edited image detection of localized edits in images is crucial for assessing content authenticity, especially given that modern diffusion models and image editing methods can produce highly realistic manipulations. However, this domain faces three challenges: (1) Binary classifiers yield only a global real-or-fake label without providing localization; (2) Traditional computer vision methods often rely on costly pixel-level annotations; and (3) No large-scale, high-quality dataset exists for modern image-editing detection techniques. To address these gaps, we develop an automated data-generation pipeline to create FragFake, the first dedicated benchmark dataset for edited image detection, which includes high-quality images from diverse editing models and a wide variety of edited objects. Based on FragFake, we utilize Vision Language Models (VLMs) for the first time in the task of edited image classification and edited region localization. Experimental results show that fine-tuned VLMs achieve higher average Object Precision across all datasets, significantly outperforming pretrained models. We further conduct ablation and transferability analyses to evaluate the detectors across various configurations and editing scenarios. To the best of our knowledge, this work is the first to reformulate localized image edit detection as a vision-language understanding task, establishing a new paradigm for the field. We anticipate that this work will establish a solid foundation to facilitate and inspire subsequent research endeavors in the domain of multimodal content authenticity.
>
---
#### [new 093] Enhancing Monte Carlo Dropout Performance for Uncertainty Quantification
- **分类: cs.CV; cs.AI**

- **简介: 论文提出结合灰狼优化、贝叶斯优化等算法及不确定性感知损失函数改进蒙特卡罗 dropout（MCD），解决其不确定性校准不足的问题，通过多网络实验验证，提升分类与不确定性估计精度2-3%，增强安全关键领域模型可信度。**

- **链接: [http://arxiv.org/pdf/2505.15671v1](http://arxiv.org/pdf/2505.15671v1)**

> **作者:** Hamzeh Asgharnezhad; Afshar Shamsi; Roohallah Alizadehsani; Arash Mohammadi; Hamid Alinejad-Rokny
>
> **备注:** 22 pages, 5 tables, 7 figures
>
> **摘要:** Knowing the uncertainty associated with the output of a deep neural network is of paramount importance in making trustworthy decisions, particularly in high-stakes fields like medical diagnosis and autonomous systems. Monte Carlo Dropout (MCD) is a widely used method for uncertainty quantification, as it can be easily integrated into various deep architectures. However, conventional MCD often struggles with providing well-calibrated uncertainty estimates. To address this, we introduce innovative frameworks that enhances MCD by integrating different search solutions namely Grey Wolf Optimizer (GWO), Bayesian Optimization (BO), and Particle Swarm Optimization (PSO) as well as an uncertainty-aware loss function, thereby improving the reliability of uncertainty quantification. We conduct comprehensive experiments using different backbones, namely DenseNet121, ResNet50, and VGG16, on various datasets, including Cats vs. Dogs, Myocarditis, Wisconsin, and a synthetic dataset (Circles). Our proposed algorithm outperforms the MCD baseline by 2-3% on average in terms of both conventional accuracy and uncertainty accuracy while achieving significantly better calibration. These results highlight the potential of our approach to enhance the trustworthiness of deep learning models in safety-critical applications.
>
---
#### [new 094] MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models
- **分类: cs.CV**

- **简介: 该论文提出MonoSplat，旨在提升3D高斯散射的泛化性，解决现有方法在新场景中处理陌生内容效果差的问题。通过结合单目深度模型的先验知识，设计特征适配器与高斯预测模块，将单目特征转化为多视角表示并融合生成精准高斯体素，在保持高效计算的同时提升重建质量和泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.15185v1](http://arxiv.org/pdf/2505.15185v1)**

> **作者:** Yifan Liu; Keyu Fan; Weihao Yu; Chenxin Li; Hao Lu; Yixuan Yuan
>
> **摘要:** Recent advances in generalizable 3D Gaussian Splatting have demonstrated promising results in real-time high-fidelity rendering without per-scene optimization, yet existing approaches still struggle to handle unfamiliar visual content during inference on novel scenes due to limited generalizability. To address this challenge, we introduce MonoSplat, a novel framework that leverages rich visual priors from pre-trained monocular depth foundation models for robust Gaussian reconstruction. Our approach consists of two key components: a Mono-Multi Feature Adapter that transforms monocular features into multi-view representations, coupled with an Integrated Gaussian Prediction module that effectively fuses both feature types for precise Gaussian generation. Through the Adapter's lightweight attention mechanism, features are seamlessly aligned and aggregated across views while preserving valuable monocular priors, enabling the Prediction module to generate Gaussian primitives with accurate geometry and appearance. Through extensive experiments on diverse real-world datasets, we convincingly demonstrate that MonoSplat achieves superior reconstruction quality and generalization capability compared to existing methods while maintaining computational efficiency with minimal trainable parameters. Codes are available at https://github.com/CUHK-AIM-Group/MonoSplat.
>
---
#### [new 095] Comprehensive Evaluation and Analysis for NSFW Concept Erasure in Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像扩散模型中NSFW（非工作安全）概念擦除评估任务，旨在解决现有方法缺乏系统性效果评估的问题。通过开发全流程工具包，首次系统研究不同场景下擦除方法的性能，分析其机制与实证结果，提供应用指导，推动扩散模型内容安全研究。**

- **链接: [http://arxiv.org/pdf/2505.15450v1](http://arxiv.org/pdf/2505.15450v1)**

> **作者:** Die Chen; Zhiwen Li; Cen Chen; Yuexiang Xie; Xiaodan Li; Jinyan Ye; Yingda Chen; Yaliang Li
>
> **摘要:** Text-to-image diffusion models have gained widespread application across various domains, demonstrating remarkable creative potential. However, the strong generalization capabilities of diffusion models can inadvertently lead to the generation of not-safe-for-work (NSFW) content, posing significant risks to their safe deployment. While several concept erasure methods have been proposed to mitigate the issue associated with NSFW content, a comprehensive evaluation of their effectiveness across various scenarios remains absent. To bridge this gap, we introduce a full-pipeline toolkit specifically designed for concept erasure and conduct the first systematic study of NSFW concept erasure methods. By examining the interplay between the underlying mechanisms and empirical observations, we provide in-depth insights and practical guidance for the effective application of concept erasure methods in various real-world scenarios, with the aim of advancing the understanding of content safety in diffusion models and establishing a solid foundation for future research and development in this critical area.
>
---
#### [new 096] SoftHGNN: Soft Hypergraph Neural Networks for General Visual Recognition
- **分类: cs.CV**

- **简介: 该论文属于视觉识别任务，旨在解决传统自注意力方法无法有效捕捉图像高阶语义关联及计算冗余的问题。提出SoftHGNN模型，通过动态软超边（连续参与权重替代二值分配）建模视觉语义连续性，结合可学习原型与稀疏超边选择机制，高效聚合高阶信息，提升场景理解能力。**

- **链接: [http://arxiv.org/pdf/2505.15325v1](http://arxiv.org/pdf/2505.15325v1)**

> **作者:** Mengqi Lei; Yihong Wu; Siqi Li; Xinhu Zheng; Juan Wang; Yue Gao; Shaoyi Du
>
> **摘要:** Visual recognition relies on understanding both the semantics of image tokens and the complex interactions among them. Mainstream self-attention methods, while effective at modeling global pair-wise relations, fail to capture high-order associations inherent in real-world scenes and often suffer from redundant computation. Hypergraphs extend conventional graphs by modeling high-order interactions and offer a promising framework for addressing these limitations. However, existing hypergraph neural networks typically rely on static and hard hyperedge assignments, leading to excessive and redundant hyperedges with hard binary vertex memberships that overlook the continuity of visual semantics. To overcome these issues, we present Soft Hypergraph Neural Networks (SoftHGNNs), which extend the methodology of hypergraph computation, to make it truly efficient and versatile in visual recognition tasks. Our framework introduces the concept of soft hyperedges, where each vertex is associated with hyperedges via continuous participation weights rather than hard binary assignments. This dynamic and differentiable association is achieved by using the learnable hyperedge prototype. Through similarity measurements between token features and the prototype, the model generates semantically rich soft hyperedges. SoftHGNN then aggregates messages over soft hyperedges to capture high-order semantics. To further enhance efficiency when scaling up the number of soft hyperedges, we incorporate a sparse hyperedge selection mechanism that activates only the top-k important hyperedges, along with a load-balancing regularizer to ensure balanced hyperedge utilization. Experimental results across three tasks on five datasets demonstrate that SoftHGNN efficiently captures high-order associations in visual scenes, achieving significant performance improvements.
>
---
#### [new 097] Interspatial Attention for Efficient 4D Human Video Generation
- **分类: cs.CV**

- **简介: 该论文属于4D人类视频生成任务，旨在解决现有方法生成视频质量低、动作不连贯或身份特征丢失的问题。提出新型跨空间注意力（ISA）机制，结合扩散 Transformer模型与定制视频自编码器，实现高质量、可控的4D视频合成，提升动作一致性与身份保真度。**

- **链接: [http://arxiv.org/pdf/2505.15800v1](http://arxiv.org/pdf/2505.15800v1)**

> **作者:** Ruizhi Shao; Yinghao Xu; Yujun Shen; Ceyuan Yang; Yang Zheng; Changan Chen; Yebin Liu; Gordon Wetzstein
>
> **备注:** Project page: https://dsaurus.github.io/isa4d/
>
> **摘要:** Generating photorealistic videos of digital humans in a controllable manner is crucial for a plethora of applications. Existing approaches either build on methods that employ template-based 3D representations or emerging video generation models but suffer from poor quality or limited consistency and identity preservation when generating individual or multiple digital humans. In this paper, we introduce a new interspatial attention (ISA) mechanism as a scalable building block for modern diffusion transformer (DiT)--based video generation models. ISA is a new type of cross attention that uses relative positional encodings tailored for the generation of human videos. Leveraging a custom-developed video variation autoencoder, we train a latent ISA-based diffusion model on a large corpus of video data. Our model achieves state-of-the-art performance for 4D human video synthesis, demonstrating remarkable motion consistency and identity preservation while providing precise control of the camera and body poses. Our code and model are publicly released at https://dsaurus.github.io/isa4d/.
>
---
#### [new 098] IA-T2I: Internet-Augmented Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出IA-T2I框架，解决文本到图像生成中因提示知识不确定（如新电影元素）导致生成效果差的问题。通过主动检索参考图、分层图像选择及自修正机制增强模型，构建Img-Ref-T2I数据集验证，实验显示优于基线30%。**

- **链接: [http://arxiv.org/pdf/2505.15779v1](http://arxiv.org/pdf/2505.15779v1)**

> **作者:** Chuanhao Li; Jianwen Sun; Yukang Feng; Mingliang Zhai; Yifan Chang; Kaipeng Zhang
>
> **备注:** 12 pages, 7 figures, a framework that integrates reference images from the Internet into T2I/TI2I models
>
> **摘要:** Current text-to-image (T2I) generation models achieve promising results, but they fail on the scenarios where the knowledge implied in the text prompt is uncertain. For example, a T2I model released in February would struggle to generate a suitable poster for a movie premiering in April, because the character designs and styles are uncertain to the model. To solve this problem, we propose an Internet-Augmented text-to-image generation (IA-T2I) framework to compel T2I models clear about such uncertain knowledge by providing them with reference images. Specifically, an active retrieval module is designed to determine whether a reference image is needed based on the given text prompt; a hierarchical image selection module is introduced to find the most suitable image returned by an image search engine to enhance the T2I model; a self-reflection mechanism is presented to continuously evaluate and refine the generated image to ensure faithful alignment with the text prompt. To evaluate the proposed framework's performance, we collect a dataset named Img-Ref-T2I, where text prompts include three types of uncertain knowledge: (1) known but rare. (2) unknown. (3) ambiguous. Moreover, we carefully craft a complex prompt to guide GPT-4o in making preference evaluation, which has been shown to have an evaluation accuracy similar to that of human preference evaluation. Experimental results demonstrate the effectiveness of our framework, outperforming GPT-4o by about 30% in human evaluation.
>
---
#### [new 099] KGAlign: Joint Semantic-Structural Knowledge Encoding for Multimodal Fake News Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出KGAlign框架，针对多模态假新闻检测中图像局部细节忽略及知识图谱未充分利用的问题，融合视觉（Bottom-up Attention、CLIP）、文本（RoBERTa）和知识图谱实体关系，通过自适应实体选择与Transformer分类，实现知识驱动的跨模态推理，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2505.14714v1](http://arxiv.org/pdf/2505.14714v1)**

> **作者:** Tuan-Vinh La; Minh-Hieu Nguyen; Minh-Son Dao
>
> **摘要:** Fake news detection remains a challenging problem due to the complex interplay between textual misinformation, manipulated images, and external knowledge reasoning. While existing approaches have achieved notable results in verifying veracity and cross-modal consistency, two key challenges persist: (1) Existing methods often consider only the global image context while neglecting local object-level details, and (2) they fail to incorporate external knowledge and entity relationships for deeper semantic understanding. To address these challenges, we propose a novel multi-modal fake news detection framework that integrates visual, textual, and knowledge-based representations. Our approach leverages bottom-up attention to capture fine-grained object details, CLIP for global image semantics, and RoBERTa for context-aware text encoding. We further enhance knowledge utilization by retrieving and adaptively selecting relevant entities from a knowledge graph. The fused multi-modal features are processed through a Transformer-based classifier to predict news veracity. Experimental results demonstrate that our model outperforms recent approaches, showcasing the effectiveness of neighbor selection mechanism and multi-modal fusion for fake news detection. Our proposal introduces a new paradigm: knowledge-grounded multimodal reasoning. By integrating explicit entity-level selection and NLI-guided filtering, we shift fake news detection from feature fusion to semantically grounded verification. For reproducibility and further research, the source code is publicly at \href{https://github.com/latuanvinh1998/KGAlign}{github.com/latuanvinh1998/KGAlign}.
>
---
#### [new 100] TimeCausality: Evaluating the Causal Ability in Time Dimension for Vision Language Models
- **分类: cs.CV; I.4.9**

- **简介: 该论文提出TimeCausality基准，评估视觉语言模型（VLM）在时间因果推理（如物体随时间不可逆变化）的能力。发现开源VLMs与闭源模型（如GPT-4o）在此任务上均表现显著不足，凸显需加强该方向的模型开发与评估。**

- **链接: [http://arxiv.org/pdf/2505.15435v1](http://arxiv.org/pdf/2505.15435v1)**

> **作者:** Zeqing Wang; Shiyuan Zhang; Chengpei Tang; Keze Wang
>
> **备注:** 17 pages, 6 figures, 3 tables
>
> **摘要:** Reasoning about temporal causality, particularly irreversible transformations of objects governed by real-world knowledge (e.g., fruit decay and human aging), is a fundamental aspect of human visual understanding. Unlike temporal perception based on simple event sequences, this form of reasoning requires a deeper comprehension of how object states change over time. Although the current powerful Vision-Language Models (VLMs) have demonstrated impressive performance on a wide range of downstream tasks, their capacity to reason about temporal causality remains underexplored. To address this gap, we introduce \textbf{TimeCausality}, a novel benchmark specifically designed to evaluate the causal reasoning ability of VLMs in the temporal dimension. Based on our TimeCausality, we find that while the current SOTA open-source VLMs have achieved performance levels comparable to closed-source models like GPT-4o on various standard visual question answering tasks, they fall significantly behind on our benchmark compared with their closed-source competitors. Furthermore, even GPT-4o exhibits a marked drop in performance on TimeCausality compared to its results on other tasks. These findings underscore the critical need to incorporate temporal causality into the evaluation and development of VLMs, and they highlight an important challenge for the open-source VLM community moving forward. Code and Data are available at \href{https://github.com/Zeqing-Wang/TimeCausality }{TimeCausality}.
>
---
#### [new 101] Exploring Generalized Gait Recognition: Reducing Redundancy and Noise within Indoor and Outdoor Datasets
- **分类: cs.CV**

- **简介: 该论文属于跨领域步态识别任务，旨在解决领域偏移导致的优化冲突及冗余噪声问题。提出解纠缠三元组损失隔离监督信号，并采用数据蒸馏过滤低效样本，提升跨域识别性能，同时保持源领域精度。**

- **链接: [http://arxiv.org/pdf/2505.15176v1](http://arxiv.org/pdf/2505.15176v1)**

> **作者:** Qian Zhou; Xianda Guo; Jilong Wang; Chuanfu Shen; Zhongyuan Wang; Hua Zou; Qin Zou; Chao Liang; Chen Long; Gang Wu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Generalized gait recognition, which aims to achieve robust performance across diverse domains, remains a challenging problem due to severe domain shifts in viewpoints, appearances, and environments. While mixed-dataset training is widely used to enhance generalization, it introduces new obstacles including inter-dataset optimization conflicts and redundant or noisy samples, both of which hinder effective representation learning. To address these challenges, we propose a unified framework that systematically improves cross-domain gait recognition. First, we design a disentangled triplet loss that isolates supervision signals across datasets, mitigating gradient conflicts during optimization. Second, we introduce a targeted dataset distillation strategy that filters out the least informative 20\% of training samples based on feature redundancy and prediction uncertainty, enhancing data efficiency. Extensive experiments on CASIA-B, OU-MVLP, Gait3D, and GREW demonstrate that our method significantly improves cross-dataset recognition for both GaitBase and DeepGaitV2 backbones, without sacrificing source-domain accuracy. Code will be released at https://github.com/li1er3/Generalized_Gait.
>
---
#### [new 102] Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL
- **分类: cs.CV**

- **简介: 该论文提出Chain-of-Focus方法，针对视觉语言模型（VLMs）在多模态推理中无法自适应聚焦关键图像区域的问题。通过构建MM-CoF数据集，采用监督微调与强化学习两阶段训练，使模型能动态搜索并缩放关键区域，提升视觉推理能力，在V*基准测试中超越现有模型5%。**

- **链接: [http://arxiv.org/pdf/2505.15436v1](http://arxiv.org/pdf/2505.15436v1)**

> **作者:** Xintong Zhang; Zhi Gao; Bofei Zhang; Pengxiang Li; Xiaowen Zhang; Yang Liu; Tao Yuan; Yuwei Wu; Yunde Jia; Song-Chun Zhu; Qing Li
>
> **摘要:** Vision language models (VLMs) have achieved impressive performance across a variety of computer vision tasks. However, the multimodal reasoning capability has not been fully explored in existing models. In this paper, we propose a Chain-of-Focus (CoF) method that allows VLMs to perform adaptive focusing and zooming in on key image regions based on obtained visual cues and the given questions, achieving efficient multimodal reasoning. To enable this CoF capability, we present a two-stage training pipeline, including supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct the MM-CoF dataset, comprising 3K samples derived from a visual agent designed to adaptively identify key regions to solve visual tasks with different image resolutions and questions. We use MM-CoF to fine-tune the Qwen2.5-VL model for cold start. In the RL stage, we leverage the outcome accuracies and formats as rewards to update the Qwen2.5-VL model, enabling further refining the search and reasoning strategy of models without human priors. Our model achieves significant improvements on multiple benchmarks. On the V* benchmark that requires strong visual reasoning capability, our model outperforms existing VLMs by 5% among 8 image resolutions ranging from 224 to 4K, demonstrating the effectiveness of the proposed CoF method and facilitating the more efficient deployment of VLMs in practical applications.
>
---
#### [new 103] Data Augmentation and Resolution Enhancement using GANs and Diffusion Models for Tree Segmentation
- **分类: cs.CV; cs.AI; cs.LG; 68T07 (Primary), 68U10, 68T45 (Secondary); I.4.8; I.2.10; I.5.4**

- **简介: 该论文针对低分辨率遥感图像中树木分割精度低及标注数据不足的问题，提出结合GANs（如Real-ESRGAN）和扩散模型（如Stable Diffusion）的数据增强与分辨率提升 pipeline。通过生成结构一致的合成样本统一图像尺度，提升分割模型跨场景鲁棒性，实验显示IoU提升超50%。任务为遥感图像树木分割，解决低分辨率与数据稀缺矛盾。**

- **链接: [http://arxiv.org/pdf/2505.15077v1](http://arxiv.org/pdf/2505.15077v1)**

> **作者:** Alessandro dos Santos Ferreira; Ana Paula Marques Ramos; José Marcato Junior; Wesley Nunes Gonçalves
>
> **备注:** 18 pages, 13 figures
>
> **摘要:** Urban forests play a key role in enhancing environmental quality and supporting biodiversity in cities. Mapping and monitoring these green spaces are crucial for urban planning and conservation, yet accurately detecting trees is challenging due to complex landscapes and the variability in image resolution caused by different satellite sensors or UAV flight altitudes. While deep learning architectures have shown promise in addressing these challenges, their effectiveness remains strongly dependent on the availability of large and manually labeled datasets, which are often expensive and difficult to obtain in sufficient quantity. In this work, we propose a novel pipeline that integrates domain adaptation with GANs and Diffusion models to enhance the quality of low-resolution aerial images. Our proposed pipeline enhances low-resolution imagery while preserving semantic content, enabling effective tree segmentation without requiring large volumes of manually annotated data. Leveraging models such as pix2pix, Real-ESRGAN, Latent Diffusion, and Stable Diffusion, we generate realistic and structurally consistent synthetic samples that expand the training dataset and unify scale across domains. This approach not only improves the robustness of segmentation models across different acquisition conditions but also provides a scalable and replicable solution for remote sensing scenarios with scarce annotation resources. Experimental results demonstrated an improvement of over 50% in IoU for low-resolution images, highlighting the effectiveness of our method compared to traditional pipelines.
>
---
#### [new 104] Multispectral Detection Transformer with Infrared-Centric Sensor Fusion
- **分类: cs.CV**

- **简介: 该论文提出IC-Fusion方法，属多光谱目标检测任务。针对RGB与红外模态互补性未充分挖掘的问题，设计以红外为中心的轻量化融合架构：通过多尺度特征蒸馏增强RGB语义，结合三阶段跨模态交互模块（含通道shuffle与大核门控），提升复杂环境下的检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.15137v1](http://arxiv.org/pdf/2505.15137v1)**

> **作者:** Seongmin Hwang; Daeyoung Han; Moongu Jeon
>
> **备注:** Under Review
>
> **摘要:** Multispectral object detection aims to leverage complementary information from visible (RGB) and infrared (IR) modalities to enable robust performance under diverse environmental conditions. In this letter, we propose IC-Fusion, a multispectral object detector that effectively fuses visible and infrared features through a lightweight and modalityaware design. Motivated by wavelet analysis and empirical observations, we find that IR images contain structurally rich high-frequency information critical for object localization, while RGB images provide complementary semantic context. To exploit this, we adopt a compact RGB backbone and design a novel fusion module comprising a Multi-Scale Feature Distillation (MSFD) block to enhance RGB features and a three-stage fusion block with Cross-Modal Channel Shuffle Gate (CCSG) and Cross-Modal Large Kernel Gate (CLKG) to facilitate effective cross-modal interaction. Experiments on the FLIR and LLVIP benchmarks demonstrate the effectiveness and efficiency of our IR-centric fusion strategy. Our code is available at https://github.com/smin-hwang/IC-Fusion.
>
---
#### [new 105] MMaDA: Multimodal Large Diffusion Language Models
- **分类: cs.CV**

- **简介: 该论文提出多模态扩散模型MMaDA，解决跨模态任务（文本推理、多模态理解、文生图）的性能与统一性问题。通过统一扩散架构、混合长链推理微调及UniGRPO强化学习算法，提升跨任务表现，实验显示其超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.15809v1](http://arxiv.org/pdf/2505.15809v1)**

> **作者:** Ling Yang; Ye Tian; Bowen Li; Xinchen Zhang; Ke Shen; Yunhai Tong; Mengdi Wang
>
> **备注:** Project: https://github.com/Gen-Verse/MMaDA
>
> **摘要:** We introduce MMaDA, a novel class of multimodal diffusion foundation models designed to achieve superior performance across diverse domains such as textual reasoning, multimodal understanding, and text-to-image generation. The approach is distinguished by three key innovations: (i) MMaDA adopts a unified diffusion architecture with a shared probabilistic formulation and a modality-agnostic design, eliminating the need for modality-specific components. This architecture ensures seamless integration and processing across different data types. (ii) We implement a mixed long chain-of-thought (CoT) fine-tuning strategy that curates a unified CoT format across modalities. By aligning reasoning processes between textual and visual domains, this strategy facilitates cold-start training for the final reinforcement learning (RL) stage, thereby enhancing the model's ability to handle complex tasks from the outset. (iii) We propose UniGRPO, a unified policy-gradient-based RL algorithm specifically tailored for diffusion foundation models. Utilizing diversified reward modeling, UniGRPO unifies post-training across both reasoning and generation tasks, ensuring consistent performance improvements. Experimental results demonstrate that MMaDA-8B exhibits strong generalization capabilities as a unified multimodal foundation model. It surpasses powerful models like LLaMA-3-7B and Qwen2-7B in textual reasoning, outperforms Show-o and SEED-X in multimodal understanding, and excels over SDXL and Janus in text-to-image generation. These achievements highlight MMaDA's effectiveness in bridging the gap between pretraining and post-training within unified diffusion architectures, providing a comprehensive framework for future research and development. We open-source our code and trained models at: https://github.com/Gen-Verse/MMaDA
>
---
#### [new 106] GAMA++: Disentangled Geometric Alignment with Adaptive Contrastive Perturbation for Reliable Domain Transfer
- **分类: cs.CV**

- **简介: 该论文属于领域自适应任务，针对现有方法在几何对齐中存在维度解纠缠不足和固定扰动方案僵化的问题，提出GAMA++框架。通过潜在空间解纠缠分离任务相关与无关维度，并设计自适应对比扰动策略，结合跨域一致性损失，提升跨域对齐精度和鲁棒性，实现迁移学习的语义几何优化。**

- **链接: [http://arxiv.org/pdf/2505.15241v1](http://arxiv.org/pdf/2505.15241v1)**

> **作者:** Kim Yun; Hana Satou; F Monkey
>
> **摘要:** Despite progress in geometry-aware domain adaptation, current methods such as GAMA still suffer from two unresolved issues: (1) insufficient disentanglement of task-relevant and task-irrelevant manifold dimensions, and (2) rigid perturbation schemes that ignore per-class alignment asymmetries. To address this, we propose GAMA++, a novel framework that introduces (i) latent space disentanglement to isolate label-consistent manifold directions from nuisance factors, and (ii) an adaptive contrastive perturbation strategy that tailors both on- and off-manifold exploration to class-specific manifold curvature and alignment discrepancy. We further propose a cross-domain contrastive consistency loss that encourages local semantic clusters to align while preserving intra-domain diversity. Our method achieves state-of-the-art results on DomainNet, Office-Home, and VisDA benchmarks under both standard and few-shot settings, with notable improvements in class-level alignment fidelity and boundary robustness. GAMA++ sets a new standard for semantic geometry alignment in transfer learning.
>
---
#### [new 107] DC-Scene: Data-Centric Learning for 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，针对数据质量差和计算效率低的挑战，提出DC-Scene框架。通过CLIP驱动的DIQ滤波器与课程调度策略，筛选高质量样本并逐步扩展训练集，提升性能同时减少训练成本。在ScanRefer和Nr3D数据集上，其top-75%子集达86.1 CIDEr，优于全数据集结果且成本降2/3。**

- **链接: [http://arxiv.org/pdf/2505.15232v1](http://arxiv.org/pdf/2505.15232v1)**

> **作者:** Ting Huang; Zeyu Zhang; Ruicheng Zhang; Yang Zhao
>
> **摘要:** 3D scene understanding plays a fundamental role in vision applications such as robotics, autonomous driving, and augmented reality. However, advancing learning-based 3D scene understanding remains challenging due to two key limitations: (1) the large scale and complexity of 3D scenes lead to higher computational costs and slower training compared to 2D counterparts; and (2) high-quality annotated 3D datasets are significantly scarcer than those available for 2D vision. These challenges underscore the need for more efficient learning paradigms. In this work, we propose DC-Scene, a data-centric framework tailored for 3D scene understanding, which emphasizes enhancing data quality and training efficiency. Specifically, we introduce a CLIP-driven dual-indicator quality (DIQ) filter, combining vision-language alignment scores with caption-loss perplexity, along with a curriculum scheduler that progressively expands the training pool from the top 25% to 75% of scene-caption pairs. This strategy filters out noisy samples and significantly reduces dependence on large-scale labeled 3D data. Extensive experiments on ScanRefer and Nr3D demonstrate that DC-Scene achieves state-of-the-art performance (86.1 CIDEr with the top-75% subset vs. 85.4 with the full dataset) while reducing training cost by approximately two-thirds, confirming that a compact set of high-quality samples can outperform exhaustive training. Code will be available at https://github.com/AIGeeksGroup/DC-Scene.
>
---
#### [new 108] Expanding Zero-Shot Object Counting with Rich Prompts
- **分类: cs.CV**

- **简介: 该论文属于零样本目标计数任务，旨在解决现有模型对未见类别计数不准确的问题。提出RichCount框架，通过两阶段训练增强文本特征与图像的对齐：用前馈网络优化文本编码，再结合计数任务提升模型泛化能力，实现开放场景下未见类别的精准计数，达 state-of-the-art 表现。**

- **链接: [http://arxiv.org/pdf/2505.15398v1](http://arxiv.org/pdf/2505.15398v1)**

> **作者:** Huilin Zhu; Senyao Li; Jingling Yuan; Zhengwei Yang; Yu Guo; Wenxuan Liu; Xian Zhong; Shengfeng He
>
> **摘要:** Expanding pre-trained zero-shot counting models to handle unseen categories requires more than simply adding new prompts, as this approach does not achieve the necessary alignment between text and visual features for accurate counting. We introduce RichCount, the first framework to address these limitations, employing a two-stage training strategy that enhances text encoding and strengthens the model's association with objects in images. RichCount improves zero-shot counting for unseen categories through two key objectives: (1) enriching text features with a feed-forward network and adapter trained on text-image similarity, thereby creating robust, aligned representations; and (2) applying this refined encoder to counting tasks, enabling effective generalization across diverse prompts and complex images. In this manner, RichCount goes beyond simple prompt expansion to establish meaningful feature alignment that supports accurate counting across novel categories. Extensive experiments on three benchmark datasets demonstrate the effectiveness of RichCount, achieving state-of-the-art performance in zero-shot counting and significantly enhancing generalization to unseen categories in open-world scenarios.
>
---
#### [new 109] AvatarShield: Visual Reinforcement Learning for Human-Centric Video Forgery Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AvatarShield，首个基于MLLM的可解释框架，用于检测人类为中心的伪造视频。针对现有方法泛化性差、依赖高成本标注及对真实感视频鲁棒性不足的问题，其采用GRPO优化算法，设计精度与时间补偿奖励机制，并结合高低层次双编码器架构，同时构建FakeHumanVid数据集。实验显示其检测效果显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15173v1](http://arxiv.org/pdf/2505.15173v1)**

> **作者:** Zhipei Xu; Xuanyu Zhang; Xing Zhou; Jian Zhang
>
> **摘要:** The rapid advancement of Artificial Intelligence Generated Content (AIGC) technologies, particularly in video generation, has led to unprecedented creative capabilities but also increased threats to information integrity, identity security, and public trust. Existing detection methods, while effective in general scenarios, lack robust solutions for human-centric videos, which pose greater risks due to their realism and potential for legal and ethical misuse. Moreover, current detection approaches often suffer from poor generalization, limited scalability, and reliance on labor-intensive supervised fine-tuning. To address these challenges, we propose AvatarShield, the first interpretable MLLM-based framework for detecting human-centric fake videos, enhanced via Group Relative Policy Optimization (GRPO). Through our carefully designed accuracy detection reward and temporal compensation reward, it effectively avoids the use of high-cost text annotation data, enabling precise temporal modeling and forgery detection. Meanwhile, we design a dual-encoder architecture, combining high-level semantic reasoning and low-level artifact amplification to guide MLLMs in effective forgery detection. We further collect FakeHumanVid, a large-scale human-centric video benchmark that includes synthesis methods guided by pose, audio, and text inputs, enabling rigorous evaluation of detection methods in real-world scenes. Extensive experiments show that AvatarShield significantly outperforms existing approaches in both in-domain and cross-domain detection, setting a new standard for human-centric video forensics.
>
---
#### [new 110] Predicting Neo-Adjuvant Chemotherapy Response in Triple-Negative Breast Cancer Using Pre-Treatment Histopathologic Images
- **分类: q-bio.QM; cs.CV; eess.IV**

- **简介: 该论文属于医学图像分析任务，旨在通过治疗前H&E染色病理图像预测三阴性乳腺癌（TNBC）患者新辅助化疗（NACT）反应，解决现有预测方法准确性不足的问题。研究构建深度学习模型，在五折验证中取得82%准确率，并结合免疫组化数据发现预测关键区域与PD-L1表达、CD8+T细胞浸润等标志物相关，为优化治疗策略和发现新生物标志物提供依据。**

- **链接: [http://arxiv.org/pdf/2505.14730v1](http://arxiv.org/pdf/2505.14730v1)**

> **作者:** Hikmat Khan; Ziyu Su; Huina Zhang; Yihong Wang; Bohan Ning; Shi Wei; Hua Guo; Zaibo Li; Muhammad Khalid Khan Niazi
>
> **摘要:** Triple-negative breast cancer (TNBC) is an aggressive subtype defined by the lack of estrogen receptor (ER), progesterone receptor (PR), and human epidermal growth factor receptor 2 (HER2) expression, resulting in limited targeted treatment options. Neoadjuvant chemotherapy (NACT) is the standard treatment for early-stage TNBC, with pathologic complete response (pCR) serving as a key prognostic marker; however, only 40-50% of patients with TNBC achieve pCR. Accurate prediction of NACT response is crucial to optimize therapy, avoid ineffective treatments, and improve patient outcomes. In this study, we developed a deep learning model to predict NACT response using pre-treatment hematoxylin and eosin (H&E)-stained biopsy images. Our model achieved promising results in five-fold cross-validation (accuracy: 82%, AUC: 0.86, F1-score: 0.84, sensitivity: 0.85, specificity: 0.81, precision: 0.80). Analysis of model attention maps in conjunction with multiplexed immunohistochemistry (mIHC) data revealed that regions of high predictive importance consistently colocalized with tumor areas showing elevated PD-L1 expression, CD8+ T-cell infiltration, and CD163+ macrophage density - all established biomarkers of treatment response. Our findings indicate that incorporating IHC-derived immune profiling data could substantially improve model interpretability and predictive performance. Furthermore, this approach may accelerate the discovery of novel histopathological biomarkers for NACT and advance the development of personalized treatment strategies for TNBC patients.
>
---
#### [new 111] Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于多模态大模型跨语言一致性评估任务，旨在解决模型在多语言环境下知识表达与视觉理解不一致的问题。提出KnowRecall（15语言视觉问答测文化知识）和VisRecall（9语言无图描述测视觉记忆）两个基准，实验显示当前模型跨语言一致性仍不足。**

- **链接: [http://arxiv.org/pdf/2505.15075v1](http://arxiv.org/pdf/2505.15075v1)**

> **作者:** Hao Wang; Pinzhi Huang; Jihan Yang; Saining Xie; Daisuke Kawahara
>
> **备注:** https://github.com/nlp-waseda/traveling-across-languages
>
> **摘要:** The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models.
>
---
#### [new 112] Kernel PCA for Out-of-Distribution Detection: Non-Linear Kernel Selections and Approximations
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于分布外检测任务，旨在通过非线性子空间区分分布内/外数据。提出基于KPCA的框架，利用Cosine-Gaussian核构建判别性子空间，并通过重建误差进行检测；针对核选择与大规模计算问题，设计了结合数据置信度的高效核近似方法，提升检测效果与效率。**

- **链接: [http://arxiv.org/pdf/2505.15284v1](http://arxiv.org/pdf/2505.15284v1)**

> **作者:** Kun Fang; Qinghua Tao; Mingzhen He; Kexin Lv; Runze Yang; Haibo Hu; Xiaolin Huang; Jie Yang; Longbin Cao
>
> **备注:** This study is an extension of its conference version published in NeurIPS'24, see https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2543511e5f4d4764857f9ad833a977d-Abstract-Conference.html
>
> **摘要:** Out-of-Distribution (OoD) detection is vital for the reliability of deep neural networks, the key of which lies in effectively characterizing the disparities between OoD and In-Distribution (InD) data. In this work, such disparities are exploited through a fresh perspective of non-linear feature subspace. That is, a discriminative non-linear subspace is learned from InD features to capture representative patterns of InD, while informative patterns of OoD features cannot be well captured in such a subspace due to their different distribution. Grounded on this perspective, we exploit the deviations of InD and OoD features in such a non-linear subspace for effective OoD detection. To be specific, we leverage the framework of Kernel Principal Component Analysis (KPCA) to attain the discriminative non-linear subspace and deploy the reconstruction error on such subspace to distinguish InD and OoD data. Two challenges emerge: (i) the learning of an effective non-linear subspace, i.e., the selection of kernel function in KPCA, and (ii) the computation of the kernel matrix with large-scale InD data. For the former, we reveal two vital non-linear patterns that closely relate to the InD-OoD disparity, leading to the establishment of a Cosine-Gaussian kernel for constructing the subspace. For the latter, we introduce two techniques to approximate the Cosine-Gaussian kernel with significantly cheap computations. In particular, our approximation is further tailored by incorporating the InD data confidence, which is demonstrated to promote the learning of discriminative subspaces for OoD data. Our study presents new insights into the non-linear feature subspace for OoD detection and contributes practical explorations on the associated kernel design and efficient computations, yielding a KPCA detection method with distinctively improved efficacy and efficiency.
>
---
#### [new 113] Explainable embeddings with Distance Explainer
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; 68T99; I.2.m**

- **简介: 该论文提出Distance Explainer方法，解决嵌入空间可解释性问题，通过遮罩和距离排序解释数据间相似性/差异性，评估于跨模态任务，有效识别特征贡献并分析参数影响。**

- **链接: [http://arxiv.org/pdf/2505.15516v1](http://arxiv.org/pdf/2505.15516v1)**

> **作者:** Christiaan Meijer; E. G. Patrick Bos
>
> **备注:** 33 pages, 19 figures. Submitted to JMLR. Method implementation: https://research-software-directory.org/software/distance-explainer
>
> **摘要:** While eXplainable AI (XAI) has advanced significantly, few methods address interpretability in embedded vector spaces where dimensions represent complex abstractions. We introduce Distance Explainer, a novel method for generating local, post-hoc explanations of embedded spaces in machine learning models. Our approach adapts saliency-based techniques from RISE to explain the distance between two embedded data points by assigning attribution values through selective masking and distance-ranked mask filtering. We evaluate Distance Explainer on cross-modal embeddings (image-image and image-caption pairs) using established XAI metrics including Faithfulness, Sensitivity/Robustness, and Randomization. Experiments with ImageNet and CLIP models demonstrate that our method effectively identifies features contributing to similarity or dissimilarity between embedded data points while maintaining high robustness and consistency. We also explore how parameter tuning, particularly mask quantity and selection strategy, affects explanation quality. This work addresses a critical gap in XAI research and enhances transparency and trustworthiness in deep learning applications utilizing embedded spaces.
>
---
#### [new 114] Pathobiological Dictionary Defining Pathomics and Texture Features: Addressing Understandable AI Issues in Personalized Liver Cancer; Dictionary Version LCP1.0
- **分类: physics.comp-ph; cs.CV; F.2.2; I.2.7**

- **简介: 该论文提出LCP1.0框架，通过整合Pathomics和Radiomics特征建立临床可解释的肝癌病理字典，解决AI医疗诊断的可解释性问题。研究提取333个影像特征，结合Variable Threshold与SVM筛选出20个关键特征（如核/细胞质特征），经专家验证，实现AI输出与临床诊断的衔接，提升模型透明度和可信度。**

- **链接: [http://arxiv.org/pdf/2505.14926v1](http://arxiv.org/pdf/2505.14926v1)**

> **作者:** Mohammad R. Salmanpour; Seyed Mohammad Piri; Somayeh Sadat Mehrnia; Ahmad Shariftabrizi; Masume Allahmoradi; Venkata SK. Manem; Arman Rahmim; Ilker Hacihaliloglu
>
> **备注:** 29 pages, 4 figures and 1 table
>
> **摘要:** Artificial intelligence (AI) holds strong potential for medical diagnostics, yet its clinical adoption is limited by a lack of interpretability and generalizability. This study introduces the Pathobiological Dictionary for Liver Cancer (LCP1.0), a practical framework designed to translate complex Pathomics and Radiomics Features (PF and RF) into clinically meaningful insights aligned with existing diagnostic workflows. QuPath and PyRadiomics, standardized according to IBSI guidelines, were used to extract 333 imaging features from hepatocellular carcinoma (HCC) tissue samples, including 240 PF-based-cell detection/intensity, 74 RF-based texture, and 19 RF-based first-order features. Expert-defined ROIs from the public dataset excluded artifact-prone areas, and features were aggregated at the case level. Their relevance to the WHO grading system was assessed using multiple classifiers linked with feature selectors. The resulting dictionary was validated by 8 experts in oncology and pathology. In collaboration with 10 domain experts, we developed a Pathobiological dictionary of imaging features such as PFs and RF. In our study, the Variable Threshold feature selection algorithm combined with the SVM model achieved the highest accuracy (0.80, P-value less than 0.05), selecting 20 key features, primarily clinical and pathomics traits such as Centroid, Cell Nucleus, and Cytoplasmic characteristics. These features, particularly nuclear and cytoplasmic, were strongly associated with tumor grading and prognosis, reflecting atypia indicators like pleomorphism, hyperchromasia, and cellular orientation.The LCP1.0 provides a clinically validated bridge between AI outputs and expert interpretation, enhancing model transparency and usability. Aligning AI-derived features with clinical semantics supports the development of interpretable, trustworthy diagnostic tools for liver cancer pathology.
>
---
#### [new 115] LOD1 3D City Model from LiDAR: The Impact of Segmentation Accuracy on Quality of Urban 3D Modeling and Morphology Extraction
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文研究基于LiDAR数据构建LOD1建筑3D模型及其形态特征提取任务，探讨分割精度对模型质量的影响。通过比较U-Net等四类深度学习模型进行建筑轮廓分割，结合统计方法估算建筑高度，发现分割准确度显著影响3D模型质量和建筑形态参数（如面积、外墙面积）的估计精度，U-Net3+结合90th百分位法最优。**

- **链接: [http://arxiv.org/pdf/2505.14747v1](http://arxiv.org/pdf/2505.14747v1)**

> **作者:** Fatemeh Chajaei; Hossein Bagheri
>
> **摘要:** Three-dimensional reconstruction of buildings, particularly at Level of Detail 1 (LOD1), plays a crucial role in various applications such as urban planning, urban environmental studies, and designing optimized transportation networks. This study focuses on assessing the potential of LiDAR data for accurate 3D building reconstruction at LOD1 and extracting morphological features from these models. Four deep semantic segmentation models, U-Net, Attention U-Net, U-Net3+, and DeepLabV3+, were used, applying transfer learning to extract building footprints from LiDAR data. The results showed that U-Net3+ and Attention U-Net outperformed the others, achieving IoU scores of 0.833 and 0.814, respectively. Various statistical measures, including maximum, range, mode, median, and the 90th percentile, were used to estimate building heights, resulting in the generation of 3D models at LOD1. As the main contribution of the research, the impact of segmentation accuracy on the quality of 3D building modeling and the accuracy of morphological features like building area and external wall surface area was investigated. The results showed that the accuracy of building identification (segmentation performance) significantly affects the 3D model quality and the estimation of morphological features, depending on the height calculation method. Overall, the UNet3+ method, utilizing the 90th percentile and median measures, leads to accurate height estimation of buildings and the extraction of morphological features.
>
---
#### [new 116] TransMedSeg: A Transferable Semantic Framework for Semi-Supervised Medical Image Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于半监督医学图像分割任务，针对现有方法忽视跨领域及模态语义关系的问题，提出TransMedSeg框架，通过Transferable Semantic Augmentation模块实现跨域分布匹配与域内结构保持，结合轻量记忆模块和隐式语义转换优化特征表示，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14753v1](http://arxiv.org/pdf/2505.14753v1)**

> **作者:** Mengzhu Wang; Jiao Li; Shanshan Wang; Long Lan; Huibin Tan; Liang Yang; Guoli Yang
>
> **摘要:** Semi-supervised learning (SSL) has achieved significant progress in medical image segmentation (SSMIS) through effective utilization of limited labeled data. While current SSL methods for medical images predominantly rely on consistency regularization and pseudo-labeling, they often overlook transferable semantic relationships across different clinical domains and imaging modalities. To address this, we propose TransMedSeg, a novel transferable semantic framework for semi-supervised medical image segmentation. Our approach introduces a Transferable Semantic Augmentation (TSA) module, which implicitly enhances feature representations by aligning domain-invariant semantics through cross-domain distribution matching and intra-domain structural preservation. Specifically, TransMedSeg constructs a unified feature space where teacher network features are adaptively augmented towards student network semantics via a lightweight memory module, enabling implicit semantic transformation without explicit data generation. Interestingly, this augmentation is implicitly realized through an expected transferable cross-entropy loss computed over the augmented teacher distribution. An upper bound of the expected loss is theoretically derived and minimized during training, incurring negligible computational overhead. Extensive experiments on medical image datasets demonstrate that TransMedSeg outperforms existing semi-supervised methods, establishing a new direction for transferable representation learning in medical image analysis.
>
---
#### [new 117] Deep Learning Enabled Segmentation, Classification and Risk Assessment of Cervical Cancer
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出多分辨率融合卷积网络及多任务学习方法，解决宫颈癌细胞图像分辨率差异与模型参数效率问题，实现高精度分割（IoU 0.83）与分类（90%），并利用概率模型预测病变风险，用于早期检测与预后。**

- **链接: [http://arxiv.org/pdf/2505.15505v1](http://arxiv.org/pdf/2505.15505v1)**

> **作者:** Abdul Samad Shaik; Shashaank Mattur Aswatha; Rahul Jashvantbhai Pandya
>
> **备注:** 11 pages, 10 figures
>
> **摘要:** Cervical cancer, the fourth leading cause of cancer in women globally, requires early detection through Pap smear tests to identify precancerous changes and prevent disease progression. In this study, we performed a focused analysis by segmenting the cellular boundaries and drawing bounding boxes to isolate the cancer cells. A novel Deep Learning (DL) architecture, the ``Multi-Resolution Fusion Deep Convolutional Network", was proposed to effectively handle images with varying resolutions and aspect ratios, with its efficacy showcased using the SIPaKMeD dataset. The performance of this DL model was observed to be similar to the state-of-the-art models, with accuracy variations of a mere 2\% to 3\%, achieved using just 1.7 million learnable parameters, which is approximately 85 times less than the VGG-19 model. Furthermore, we introduced a multi-task learning technique that simultaneously performs segmentation and classification tasks and begets an Intersection over Union score of 0.83 and a classification accuracy of 90\%. The final stage of the workflow employs a probabilistic approach for risk assessment, extracting feature vectors to predict the likelihood of normal cells progressing to malignant states, which can be utilized for the prognosis of cervical cancer.
>
---
#### [new 118] X-GRM: Large Gaussian Reconstruction Model for Sparse-view X-rays to Computed Tomography
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于CT图像重建任务，旨在解决现有方法模型容量小、体积表示僵化及训练数据不足的问题。提出X-GRM模型，采用可扩展Transformer编码多视角X光输入，设计Voxel-based Gaussian Splatting表示，并构建15K规模数据集，提升稀疏视图CT重建质量。**

- **链接: [http://arxiv.org/pdf/2505.15235v1](http://arxiv.org/pdf/2505.15235v1)**

> **作者:** Yifan Liu; Wuyang Li; Weihao Yu; Chenxin Li; Alexandre Alahi; Max Meng; Yixuan Yuan
>
> **摘要:** Computed Tomography serves as an indispensable tool in clinical workflows, providing non-invasive visualization of internal anatomical structures. Existing CT reconstruction works are limited to small-capacity model architecture, inflexible volume representation, and small-scale training data. In this paper, we present X-GRM (X-ray Gaussian Reconstruction Model), a large feedforward model for reconstructing 3D CT from sparse-view 2D X-ray projections. X-GRM employs a scalable transformer-based architecture to encode an arbitrary number of sparse X-ray inputs, where tokens from different views are integrated efficiently. Then, tokens are decoded into a new volume representation, named Voxel-based Gaussian Splatting (VoxGS), which enables efficient CT volume extraction and differentiable X-ray rendering. To support the training of X-GRM, we collect ReconX-15K, a large-scale CT reconstruction dataset containing around 15,000 CT/X-ray pairs across diverse organs, including the chest, abdomen, pelvis, and tooth etc. This combination of a high-capacity model, flexible volume representation, and large-scale training data empowers our model to produce high-quality reconstructions from various testing inputs, including in-domain and out-domain X-ray projections. Project Page: https://github.com/CUHK-AIM-Group/X-GRM.
>
---
#### [new 119] MedBLIP: Fine-tuning BLIP for Medical Image Captioning
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医疗影像描述生成任务，旨在解决现有视觉语言模型在医学领域生成描述不精准的问题。研究通过在ROCO数据集上微调BLIP模型，对比不同版本模型及微调策略，发现领域微调显著提升效果，解码器单独微调训练时间减少5%，但全模型微调最优，强调针对性适配对医疗应用的重要性。**

- **链接: [http://arxiv.org/pdf/2505.14726v1](http://arxiv.org/pdf/2505.14726v1)**

> **作者:** Manshi Limbu; Diwita Banerjee
>
> **摘要:** Medical image captioning is a challenging task that requires generating clinically accurate and semantically meaningful descriptions of radiology images. While recent vision-language models (VLMs) such as BLIP, BLIP2, Gemini and ViT-GPT2 show strong performance on natural image datasets, they often produce generic or imprecise captions when applied to specialized medical domains. In this project, we explore the effectiveness of fine-tuning the BLIP model on the ROCO dataset for improved radiology captioning. We compare the fine-tuned BLIP against its zero-shot version, BLIP-2 base, BLIP-2 Instruct and a ViT-GPT2 transformer baseline. Our results demonstrate that domain-specific fine-tuning on BLIP significantly improves performance across both quantitative and qualitative evaluation metrics. We also visualize decoder cross-attention maps to assess interpretability and conduct an ablation study to evaluate the contributions of encoder-only and decoder-only fine-tuning. Our findings highlight the importance of targeted adaptation for medical applications and suggest that decoder-only fine-tuning (encoder-frozen) offers a strong performance baseline with 5% lower training time than full fine-tuning, while full model fine-tuning still yields the best results overall.
>
---
#### [new 120] AsynFusion: Towards Asynchronous Latent Consistency Models for Decoupled Whole-Body Audio-Driven Avatars
- **分类: cs.SD; cs.AI; cs.CV; eess.AS; 68T10**

- **简介: 该论文提出AsynFusion框架，解决现有语音驱动虚拟形象面部与手势生成不协调的问题。任务为生成全身自然动画，通过双分支DiT架构并行生成表情与手势，引入协同同步模块促进模态交互，采用异步采样降低计算量，实现实时高质量同步动画。**

- **链接: [http://arxiv.org/pdf/2505.15058v1](http://arxiv.org/pdf/2505.15058v1)**

> **作者:** Tianbao Zhang; Jian Zhao; Yuer Li; Zheng Zhu; Ping Hu; Zhaoxin Fan; Wenjun Wu; Xuelong Li
>
> **备注:** 11pages, conference
>
> **摘要:** Whole-body audio-driven avatar pose and expression generation is a critical task for creating lifelike digital humans and enhancing the capabilities of interactive virtual agents, with wide-ranging applications in virtual reality, digital entertainment, and remote communication. Existing approaches often generate audio-driven facial expressions and gestures independently, which introduces a significant limitation: the lack of seamless coordination between facial and gestural elements, resulting in less natural and cohesive animations. To address this limitation, we propose AsynFusion, a novel framework that leverages diffusion transformers to achieve harmonious expression and gesture synthesis. The proposed method is built upon a dual-branch DiT architecture, which enables the parallel generation of facial expressions and gestures. Within the model, we introduce a Cooperative Synchronization Module to facilitate bidirectional feature interaction between the two modalities, and an Asynchronous LCM Sampling strategy to reduce computational overhead while maintaining high-quality outputs. Extensive experiments demonstrate that AsynFusion achieves state-of-the-art performance in generating real-time, synchronized whole-body animations, consistently outperforming existing methods in both quantitative and qualitative evaluations.
>
---
#### [new 121] Exploring In-Image Machine Translation with Real-World Background
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于跨模态机器翻译任务，针对现有模型在复杂真实背景图像翻译效果差的问题，构建了含真实背景字幕的IIMT数据集，并提出DebackX模型：通过分离图像背景与文字、直接翻译文字图像再融合，提升翻译质量和视觉效果。**

- **链接: [http://arxiv.org/pdf/2505.15282v1](http://arxiv.org/pdf/2505.15282v1)**

> **作者:** Yanzhi Tian; Zeming Liu; Zhengyang Liu; Yuhang Guo
>
> **备注:** Accepted to ACL 2025 Findings. Code available at https://github.com/BITHLP/DebackX
>
> **摘要:** In-Image Machine Translation (IIMT) aims to translate texts within images from one language to another. Previous research on IIMT was primarily conducted on simplified scenarios such as images of one-line text with black font in white backgrounds, which is far from reality and impractical for applications in the real world. To make IIMT research practically valuable, it is essential to consider a complex scenario where the text backgrounds are derived from real-world images. To facilitate research of complex scenario IIMT, we design an IIMT dataset that includes subtitle text with real-world background. However previous IIMT models perform inadequately in complex scenarios. To address the issue, we propose the DebackX model, which separates the background and text-image from the source image, performs translation on text-image directly, and fuses the translated text-image with the background, to generate the target image. Experimental results show that our model achieves improvements in both translation quality and visual effect.
>
---
#### [new 122] A Comprehensive Review of Techniques, Algorithms, Advancements, Challenges, and Clinical Applications of Multi-modal Medical Image Fusion for Improved Diagnosis
- **分类: eess.IV; cs.CV**

- **简介: 该论文综述多模态医学图像融合(MMIF)技术，整合X光、MRI等多源影像提升诊断精度。对比传统方法(像素/特征/决策级融合)与深度学习、Transformer等新算法，分析临床应用(如肿瘤、神经疾病)及挑战(数据异构、隐私、计算复杂度)，提出可解释AI、联邦学习等未来方向。**

- **链接: [http://arxiv.org/pdf/2505.14715v1](http://arxiv.org/pdf/2505.14715v1)**

> **作者:** Muhammad Zubair; Muzammil Hussai; Mousa Ahmad Al-Bashrawi; Malika Bendechache; Muhammad Owais
>
> **备注:** computerized medical imaging and graphics Journal submission
>
> **摘要:** Multi-modal medical image fusion (MMIF) is increasingly recognized as an essential technique for enhancing diagnostic precision and facilitating effective clinical decision-making within computer-aided diagnosis systems. MMIF combines data from X-ray, MRI, CT, PET, SPECT, and ultrasound to create detailed, clinically useful images of patient anatomy and pathology. These integrated representations significantly advance diagnostic accuracy, lesion detection, and segmentation. This comprehensive review meticulously surveys the evolution, methodologies, algorithms, current advancements, and clinical applications of MMIF. We present a critical comparative analysis of traditional fusion approaches, including pixel-, feature-, and decision-level methods, and delves into recent advancements driven by deep learning, generative models, and transformer-based architectures. A critical comparative analysis is presented between these conventional methods and contemporary techniques, highlighting differences in robustness, computational efficiency, and interpretability. The article addresses extensive clinical applications across oncology, neurology, and cardiology, demonstrating MMIF's vital role in precision medicine through improved patient-specific therapeutic outcomes. Moreover, the review thoroughly investigates the persistent challenges affecting MMIF's broad adoption, including issues related to data privacy, heterogeneity, computational complexity, interpretability of AI-driven algorithms, and integration within clinical workflows. It also identifies significant future research avenues, such as the integration of explainable AI, adoption of privacy-preserving federated learning frameworks, development of real-time fusion systems, and standardization efforts for regulatory compliance.
>
---
#### [new 123] Scaling Diffusion Transformers Efficiently via $μ$P
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究扩散Transformer的高效扩展问题，提出将μP方法推广至其架构，解决大规模模型超参数调优成本高的问题。通过理论证明和实验验证，μP使DiT等模型实现快速收敛（如DiT-XL-2加速2.9倍），并显著降低调参成本（仅需5.5%-3%），有效提升文本到图像生成性能。**

- **链接: [http://arxiv.org/pdf/2505.15270v1](http://arxiv.org/pdf/2505.15270v1)**

> **作者:** Chenyu Zheng; Xinyu Zhang; Rongzhen Wang; Wei Huang; Zhi Tian; Weilin Huang; Jun Zhu; Chongxuan Li
>
> **备注:** 35 pages, 10 figures, 15 tables
>
> **摘要:** Diffusion Transformers have emerged as the foundation for vision generative models, but their scalability is limited by the high cost of hyperparameter (HP) tuning at large scales. Recently, Maximal Update Parametrization ($\mu$P) was proposed for vanilla Transformers, which enables stable HP transfer from small to large language models, and dramatically reduces tuning costs. However, it remains unclear whether $\mu$P of vanilla Transformers extends to diffusion Transformers, which differ architecturally and objectively. In this work, we generalize standard $\mu$P to diffusion Transformers and validate its effectiveness through large-scale experiments. First, we rigorously prove that $\mu$P of mainstream diffusion Transformers, including DiT, U-ViT, PixArt-$\alpha$, and MMDiT, aligns with that of the vanilla Transformer, enabling the direct application of existing $\mu$P methodologies. Leveraging this result, we systematically demonstrate that DiT-$\mu$P enjoys robust HP transferability. Notably, DiT-XL-2-$\mu$P with transferred learning rate achieves 2.9 times faster convergence than the original DiT-XL-2. Finally, we validate the effectiveness of $\mu$P on text-to-image generation by scaling PixArt-$\alpha$ from 0.04B to 0.61B and MMDiT from 0.18B to 18B. In both cases, models under $\mu$P outperform their respective baselines while requiring small tuning cost, only 5.5% of one training run for PixArt-$\alpha$ and 3% of consumption by human experts for MMDiT-18B. These results establish $\mu$P as a principled and efficient framework for scaling diffusion Transformers.
>
---
#### [new 124] Reconsider the Template Mesh in Deep Learning-based Mesh Reconstruction
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于深度学习驱动的网格重建任务，旨在解决传统方法依赖固定模板导致的解剖结构差异忽视及精度不足问题。提出自适应模板网络ATMRN，通过从输入图像生成个体化模板再变形，提升重建精度。在OASIS脑MR数据集上实现0.267mm平均对称表面距离，方法具通用性。**

- **链接: [http://arxiv.org/pdf/2505.15285v1](http://arxiv.org/pdf/2505.15285v1)**

> **作者:** Fengting Zhang; Boxu Liang; Qinghao Liu; Min Liu; Xiang Chen; Yaonan Wang
>
> **摘要:** Mesh reconstruction is a cornerstone process across various applications, including in-silico trials, digital twins, surgical planning, and navigation. Recent advancements in deep learning have notably enhanced mesh reconstruction speeds. Yet, traditional methods predominantly rely on deforming a standardised template mesh for individual subjects, which overlooks the unique anatomical variations between them, and may compromise the fidelity of the reconstructions. In this paper, we propose an adaptive-template-based mesh reconstruction network (ATMRN), which generates adaptive templates from the given images for the subsequent deformation, moving beyond the constraints of a singular, fixed template. Our approach, validated on cortical magnetic resonance (MR) images from the OASIS dataset, sets a new benchmark in voxel-to-cortex mesh reconstruction, achieving an average symmetric surface distance of 0.267mm across four cortical structures. Our proposed method is generic and can be easily transferred to other image modalities and anatomical structures.
>
---
#### [new 125] Beyond Classification: Evaluating Diffusion Denoised Smoothing for Security-Utility Trade off
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究模型鲁棒性，评估扩散去噪平滑（DDS）在非分类任务中的安全-性能权衡。通过测试三个数据集、四任务及三种攻击算法，发现高噪声显著降解干净数据性能（达57%），低噪声防护不足，并提出针对扩散过程的新型攻击，揭示安全与性能的平衡难题。**

- **链接: [http://arxiv.org/pdf/2505.15594v1](http://arxiv.org/pdf/2505.15594v1)**

> **作者:** Yury Belousov; Brian Pulfer; Vitaliy Kinakh; Slava Voloshynovskiy
>
> **备注:** Paper accepted at the 33rd European Signal Processing Conference (EUSIPCO 2025)
>
> **摘要:** While foundation models demonstrate impressive performance across various tasks, they remain vulnerable to adversarial inputs. Current research explores various approaches to enhance model robustness, with Diffusion Denoised Smoothing emerging as a particularly promising technique. This method employs a pretrained diffusion model to preprocess inputs before model inference. Yet, its effectiveness remains largely unexplored beyond classification. We aim to address this gap by analyzing three datasets with four distinct downstream tasks under three different adversarial attack algorithms. Our findings reveal that while foundation models maintain resilience against conventional transformations, applying high-noise diffusion denoising to clean images without any distortions significantly degrades performance by as high as 57%. Low-noise diffusion settings preserve performance but fail to provide adequate protection across all attack types. Moreover, we introduce a novel attack strategy specifically targeting the diffusion process itself, capable of circumventing defenses in the low-noise regime. Our results suggest that the trade-off between adversarial robustness and performance remains a challenge to be addressed.
>
---
#### [new 126] GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文聚焦GUI视觉定位任务，针对R1-Zero训练中输入设计（长推理降效）、输出评估（奖励漏洞致定位偏差）及策略更新（过拟合易样本）问题，提出快速回答模板、奖励尺寸约束及难度感知优化方法，实现90.3%的ScreenSpot准确率，创同类模型最优。**

- **链接: [http://arxiv.org/pdf/2505.15810v1](http://arxiv.org/pdf/2505.15810v1)**

> **作者:** Yuqi Zhou; Sunhao Dai; Shuai Wang; Kaiwen Zhou; Qinqlin Jia; Junxu
>
> **摘要:** Recent Graphical User Interface (GUI) agents replicate the R1-Zero paradigm, coupling online Reinforcement Learning (RL) with explicit chain-of-thought reasoning prior to object grounding and thereby achieving substantial performance gains. In this paper, we first conduct extensive analysis experiments of three key components of that training pipeline: input design, output evaluation, and policy update-each revealing distinct challenges arising from blindly applying general-purpose RL without adapting to GUI grounding tasks. Input design: Current templates encourage the model to generate chain-of-thought reasoning, but longer chains unexpectedly lead to worse grounding performance. Output evaluation: Reward functions based on hit signals or box area allow models to exploit box size, leading to reward hacking and poor localization quality. Policy update: Online RL tends to overfit easy examples due to biases in length and sample difficulty, leading to under-optimization on harder cases. To address these issues, we propose three targeted solutions. First, we adopt a Fast Thinking Template that encourages direct answer generation, reducing excessive reasoning during training. Second, we incorporate a box size constraint into the reward function to mitigate reward hacking. Third, we revise the RL objective by adjusting length normalization and adding a difficulty-aware scaling factor, enabling better optimization on hard samples. Our GUI-G1-3B, trained on 17K public samples with Qwen2.5-VL-3B-Instruct, achieves 90.3% accuracy on ScreenSpot and 37.1% on ScreenSpot-Pro. This surpasses all prior models of similar size and even outperforms the larger UI-TARS-7B, establishing a new state-of-the-art in GUI agent grounding. The project repository is available at https://github.com/Yuqi-Zhou/GUI-G1.
>
---
#### [new 127] Physics-Guided Multi-View Graph Neural Network for Schizophrenia Classification via Structural-Functional Coupling
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于精神分裂症分类任务，旨在解决传统方法仅依赖结构连接（SC）忽视结构-功能连接（SC-FC）耦合的问题。提出物理引导的多视角图神经网络，通过神经振荡模型生成功能连接，并融合SC-FC进行分类，提升SZ诊断性能。**

- **链接: [http://arxiv.org/pdf/2505.15135v1](http://arxiv.org/pdf/2505.15135v1)**

> **作者:** Badhan Mazumder; Ayush Kanyal; Lei Wu; Vince D. Calhoun; Dong Hye Ye
>
> **备注:** Accepted and presented at the 7th International Workshop on PRedictive Intelligence in MEdicine (Held in Conjunction with MICCAI 2024)
>
> **摘要:** Clinical studies reveal disruptions in brain structural connectivity (SC) and functional connectivity (FC) in neuropsychiatric disorders such as schizophrenia (SZ). Traditional approaches might rely solely on SC due to limited functional data availability, hindering comprehension of cognitive and behavioral impairments in individuals with SZ by neglecting the intricate SC-FC interrelationship. To tackle the challenge, we propose a novel physics-guided deep learning framework that leverages a neural oscillation model to describe the dynamics of a collection of interconnected neural oscillators, which operate via nerve fibers dispersed across the brain's structure. Our proposed framework utilizes SC to simultaneously generate FC by learning SC-FC coupling from a system dynamics perspective. Additionally, it employs a novel multi-view graph neural network (GNN) with a joint loss to perform correlation-based SC-FC fusion and classification of individuals with SZ. Experiments conducted on a clinical dataset exhibited improved performance, demonstrating the robustness of our proposed approach.
>
---
#### [new 128] Model-Independent Machine Learning Approach for Nanometric Axial Localization and Tracking
- **分类: eess.IV; astro-ph.IM; cs.CV; cs.LG; physics.ins-det**

- **简介: 论文提出无需预设模型的深度学习方法，通过卷积神经网络分析双焦面图像实现纳米级轴向定位与追踪，解决光学显微镜高精度轴向定位难题，精度达40纳米（超传统方法6倍），适用于暗物质检测、医疗成像等多领域。**

- **链接: [http://arxiv.org/pdf/2505.14754v1](http://arxiv.org/pdf/2505.14754v1)**

> **作者:** Andrey Alexandrov; Giovanni Acampora; Giovanni De Lellis; Antonia Di Crescenzo; Chiara Errico; Daria Morozova; Valeri Tioukov; Autilia Vittiello
>
> **备注:** 11 pages, 4 figures, 1 table
>
> **摘要:** Accurately tracking particles and determining their position along the optical axis is a major challenge in optical microscopy, especially when extremely high precision is needed. In this study, we introduce a deep learning approach using convolutional neural networks (CNNs) that can determine axial positions from dual-focal plane images without relying on predefined models. Our method achieves an axial localization accuracy of 40 nanometers - six times better than traditional single-focal plane techniques. The model's simple design and strong performance make it suitable for a wide range of uses, including dark matter detection, proton therapy for cancer, and radiation protection in space. It also shows promise in fields like biological imaging, materials science, and environmental monitoring. This work highlights how machine learning can turn complex image data into reliable, precise information, offering a flexible and powerful tool for many scientific applications.
>
---
#### [new 129] Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning
- **分类: cs.LG; cs.AI; cs.CV; cs.IR; 20-XX, 08A02; F.4.1; I.2**

- **简介: 该论文提出一种新型方向非交换单oidal结构，用于多维组合嵌入，解决现有模型无法统一处理多维结构化数据的问题。通过定义各轴独立的结合性运算符并确保全局交换律，该框架兼容序列模型与Transformer等经典方法，支持多维递归操作，为结构化位置编码、图像嵌入等提供理论基础。**

- **链接: [http://arxiv.org/pdf/2505.15507v1](http://arxiv.org/pdf/2505.15507v1)**

> **作者:** Mahesh Godavarti
>
> **备注:** 11 pages submitted to NeurIPS 2025
>
> **摘要:** We introduce a new algebraic structure for multi-dimensional compositional embeddings, built on directional non-commutative monoidal operators. The core contribution of this work is this novel framework, which exhibits appealing theoretical properties (associativity along each dimension and an interchange law ensuring global consistency) while remaining compatible with modern machine learning architectures. Our construction defines a distinct composition operator circ_i for each axis i, ensuring associative combination along each axis without imposing global commutativity. Importantly, all axis-specific operators commute with one another, enforcing a global interchange law that enables consistent crossaxis compositions. This is, to our knowledge, the first approach that provides a common foundation that generalizes classical sequence-modeling paradigms (e.g., structured state-space models (SSMs) and transformer self-attention) to a unified multi-dimensional framework. For example, specific one-dimensional instances of our framework can recover the familiar affine transformation algebra, vanilla self-attention, and the SSM-style recurrence. The higher-dimensional generalizations naturally support recursive, structure-aware operations in embedding spaces. We outline several potential applications unlocked by this structure-including structured positional encodings in Transformers, directional image embeddings, and symbolic modeling of sequences or grids-indicating that it could inform future deep learning model designs. We formally establish the algebraic properties of our framework and discuss efficient implementations. Finally, as our focus is theoretical, we include no experiments here and defer empirical validation to future work, which we plan to undertake.
>
---
#### [new 130] Aneumo: A Large-Scale Multimodal Aneurysm Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文构建了Aneumo数据集，解决传统CFD计算成本高、难以大规模应用的问题。基于427例真实动脉瘤几何，生成10,660个3D变形模型并进行CFD模拟，获得8万余组血流动力学数据，包含分割掩码支持多模态分析，并建立深度学习基准，推动动脉瘤破裂风险评估的AI研究与临床应用。**

- **链接: [http://arxiv.org/pdf/2505.14717v1](http://arxiv.org/pdf/2505.14717v1)**

> **作者:** Xigui Li; Yuanye Zhou; Feiyang Xiao; Xin Guo; Chen Jiang; Tan Pan; Xingmeng Zhang; Cenyu Liu; Zeyun Miao; Jianchao Ge; Xiansheng Wang; Qimeng Wang; Yichi Zhang; Wenbo Zhang; Fengping Zhu; Limei Han; Yuan Qi; Chensen Lin; Yuan Cheng
>
> **摘要:** Intracranial aneurysms (IAs) are serious cerebrovascular lesions found in approximately 5\% of the general population. Their rupture may lead to high mortality. Current methods for assessing IA risk focus on morphological and patient-specific factors, but the hemodynamic influences on IA development and rupture remain unclear. While accurate for hemodynamic studies, conventional computational fluid dynamics (CFD) methods are computationally intensive, hindering their deployment in large-scale or real-time clinical applications. To address this challenge, we curated a large-scale, high-fidelity aneurysm CFD dataset to facilitate the development of efficient machine learning algorithms for such applications. Based on 427 real aneurysm geometries, we synthesized 10,660 3D shapes via controlled deformation to simulate aneurysm evolution. The authenticity of these synthetic shapes was confirmed by neurosurgeons. CFD computations were performed on each shape under eight steady-state mass flow conditions, generating a total of 85,280 blood flow dynamics data covering key parameters. Furthermore, the dataset includes segmentation masks, which can support tasks that use images, point clouds or other multimodal data as input. Additionally, we introduced a benchmark for estimating flow parameters to assess current modeling methods. This dataset aims to advance aneurysm research and promote data-driven approaches in biofluids, biomedical engineering, and clinical risk assessment. The code and dataset are available at: https://github.com/Xigui-Li/Aneumo.
>
---
#### [new 131] AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出AgentThink框架，解决视觉语言模型在自动驾驶中的幻觉、推理低效及现实验证不足问题。通过构建自动驾驶工具库生成结构化数据、两阶段训练（SFT+GRPO）提升工具调用能力，及多工具评估协议，显著提升推理准确性和一致性。**

- **链接: [http://arxiv.org/pdf/2505.15298v1](http://arxiv.org/pdf/2505.15298v1)**

> **作者:** Kangan Qian; Sicong Jiang; Yang Zhong; Ziang Luo; Zilin Huang; Tianze Zhu; Kun Jiang; Mengmeng Yang; Zheng Fu; Jinyu Miao; Yining Shi; He Zhe Lim; Li Liu; Tianbao Zhou; Hongyi Wang; Huang Yu; Yifei Hu; Guang Li; Guang Chen; Hao Ye; Lijun Sun; Diange Yang
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Vision-Language Models (VLMs) show promise for autonomous driving, yet their struggle with hallucinations, inefficient reasoning, and limited real-world validation hinders accurate perception and robust step-by-step reasoning. To overcome this, we introduce \textbf{AgentThink}, a pioneering unified framework that, for the first time, integrates Chain-of-Thought (CoT) reasoning with dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink's core innovations include: \textbf{(i) Structured Data Generation}, by establishing an autonomous driving tool library to automatically construct structured, self-verified reasoning data explicitly incorporating tool usage for diverse driving scenarios; \textbf{(ii) A Two-stage Training Pipeline}, employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to equip VLMs with the capability for autonomous tool invocation; and \textbf{(iii) Agent-style Tool-Usage Evaluation}, introducing a novel multi-tool assessment protocol to rigorously evaluate the model's tool invocation and utilization. Experiments on the DriveLMM-o1 benchmark demonstrate AgentThink significantly boosts overall reasoning scores by \textbf{53.91\%} and enhances answer accuracy by \textbf{33.54\%}, while markedly improving reasoning quality and consistency. Furthermore, ablation studies and robust zero-shot/few-shot generalization experiments across various benchmarks underscore its powerful capabilities. These findings highlight a promising trajectory for developing trustworthy and tool-aware autonomous driving models.
>
---
#### [new 132] Non-rigid Motion Correction for MRI Reconstruction via Coarse-To-Fine Diffusion Models
- **分类: eess.IV; cs.CV**

- **简介: 该论文属MRI运动校正与重建任务，解决动态成像中非刚性运动导致的伪影问题。提出粗到细扩散模型框架，通过交替优化联合重建图像并校正运动，先恢复低频信息再细化运动估计，适用于高倍率欠采样及多种扫描条件，在真实心脏MRI和模拟数据中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.15057v1](http://arxiv.org/pdf/2505.15057v1)**

> **作者:** Frederic Wang; Jonathan I. Tamir
>
> **备注:** ICIP 2025
>
> **摘要:** Magnetic Resonance Imaging (MRI) is highly susceptible to motion artifacts due to the extended acquisition times required for k-space sampling. These artifacts can compromise diagnostic utility, particularly for dynamic imaging. We propose a novel alternating minimization framework that leverages a bespoke diffusion model to jointly reconstruct and correct non-rigid motion-corrupted k-space data. The diffusion model uses a coarse-to-fine denoising strategy to capture large overall motion and reconstruct the lower frequencies of the image first, providing a better inductive bias for motion estimation than that of standard diffusion models. We demonstrate the performance of our approach on both real-world cine cardiac MRI datasets and complex simulated rigid and non-rigid deformations, even when each motion state is undersampled by a factor of 64x. Additionally, our method is agnostic to sampling patterns, anatomical variations, and MRI scanning protocols, as long as some low frequency components are sampled during each motion state.
>
---
#### [new 133] Lung Nodule-SSM: Self-Supervised Lung Nodule Detection and Classification in Thoracic CT Images
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文提出LungNodule-SSM方法，针对肺结节检测与分类任务，解决医学影像标注数据不足的问题。通过自监督学习使用DINOv2模型在无标注CT图像预训练，再微调Transformer模型实现病灶检测与分类，在LUNA16数据集上达到98.37%准确率。**

- **链接: [http://arxiv.org/pdf/2505.15120v1](http://arxiv.org/pdf/2505.15120v1)**

> **作者:** Muniba Noreen; Furqan Shaukat
>
> **摘要:** Lung cancer remains among the deadliest types of cancer in recent decades, and early lung nodule detection is crucial for improving patient outcomes. The limited availability of annotated medical imaging data remains a bottleneck in developing accurate computer-aided diagnosis (CAD) systems. Self-supervised learning can help leverage large amounts of unlabeled data to develop more robust CAD systems. With the recent advent of transformer-based architecture and their ability to generalize to unseen tasks, there has been an effort within the healthcare community to adapt them to various medical downstream tasks. Thus, we propose a novel "LungNodule-SSM" method, which utilizes selfsupervised learning with DINOv2 as a backbone to enhance lung nodule detection and classification without annotated data. Our methodology has two stages: firstly, the DINOv2 model is pre-trained on unlabeled CT scans to learn robust feature representations, then secondly, these features are fine-tuned using transformer-based architectures for lesionlevel detection and accurate lung nodule diagnosis. The proposed method has been evaluated on the challenging LUNA 16 dataset, consisting of 888 CT scans, and compared with SOTA methods. Our experimental results show the superiority of our proposed method with an accuracy of 98.37%, explaining its effectiveness in lung nodule detection. The source code, datasets, and pre-processed data can be accessed using the link:https://github.com/EMeRALDsNRPU/Lung-Nodule-SSM-Self-Supervised-Lung-Nodule-Detection-and-Classification/tree/main
>
---
#### [new 134] ComBAT Harmonization for diffusion MRI: Challenges and Best Practices
- **分类: stat.AP; cs.CV; cs.LG; physics.med-ph**

- **简介: 该论文属于扩散MRI数据标准化方法研究，旨在解决ComBAT算法因假设不符导致谐波失效的问题。通过分析其数学假设、实验评估人口规模、年龄分布等影响，并提出五项优化建议，提升数据一致性和可重复性，助力开放科学与临床应用。**

- **链接: [http://arxiv.org/pdf/2505.14722v1](http://arxiv.org/pdf/2505.14722v1)**

> **作者:** Pierre-Marc Jodoin; Manon Edde; Gabriel Girard; Félix Dumais; Guillaume Theaud; Matthieu Dumont; Jean-Christophe Houde; Yoan David; Maxime Descoteaux
>
> **摘要:** Over the years, ComBAT has become the standard method for harmonizing MRI-derived measurements, with its ability to compensate for site-related additive and multiplicative biases while preserving biological variability. However, ComBAT relies on a set of assumptions that, when violated, can result in flawed harmonization. In this paper, we thoroughly review ComBAT's mathematical foundation, outlining these assumptions, and exploring their implications for the demographic composition necessary for optimal results. Through a series of experiments involving a slightly modified version of ComBAT called Pairwise-ComBAT tailored for normative modeling applications, we assess the impact of various population characteristics, including population size, age distribution, the absence of certain covariates, and the magnitude of additive and multiplicative factors. Based on these experiments, we present five essential recommendations that should be carefully considered to enhance consistency and supporting reproducibility, two essential factors for open science, collaborative research, and real-life clinical deployment.
>
---
#### [new 135] SAMA-UNet: Enhancing Medical Image Segmentation with Self-Adaptive Mamba-Like Attention and Causal-Resonance Learning
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医疗图像分割任务，旨在解决现有模型在计算效率、局部全局特征平衡及SSMs与医学图像适配性上的问题。提出SAMA-UNet架构，整合自适应注意力模块（SAMA）优化多尺度特征选择，及因果共振多尺度模块（CR-MSM）增强编解码器信息流动，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2505.15234v1](http://arxiv.org/pdf/2505.15234v1)**

> **作者:** Saqib Qamar; Mohd Fazil; Parvez Ahmad; Ghulam Muhammad
>
> **摘要:** Medical image segmentation plays an important role in various clinical applications, but existing models often struggle with the computational inefficiencies and challenges posed by complex medical data. State Space Sequence Models (SSMs) have demonstrated promise in modeling long-range dependencies with linear computational complexity, yet their application in medical image segmentation remains hindered by incompatibilities with image tokens and autoregressive assumptions. Moreover, it is difficult to achieve a balance in capturing both local fine-grained information and global semantic dependencies. To address these challenges, we introduce SAMA-UNet, a novel architecture for medical image segmentation. A key innovation is the Self-Adaptive Mamba-like Aggregated Attention (SAMA) block, which integrates contextual self-attention with dynamic weight modulation to prioritise the most relevant features based on local and global contexts. This approach reduces computational complexity and improves the representation of complex image features across multiple scales. We also suggest the Causal-Resonance Multi-Scale Module (CR-MSM), which enhances the flow of information between the encoder and decoder by using causal resonance learning. This mechanism allows the model to automatically adjust feature resolution and causal dependencies across scales, leading to better semantic alignment between the low-level and high-level features in U-shaped architectures. Experiments on MRI, CT, and endoscopy images show that SAMA-UNet performs better in segmentation accuracy than current methods using CNN, Transformer, and Mamba. The implementation is publicly available at GitHub.
>
---
#### [new 136] Are Vision-Language Models Safe in the Wild? A Meme-Based Benchmark Study
- **分类: cs.CL; cs.CR; cs.CV**

- **简介: 该论文属于视觉语言模型（VLM）安全评估任务，旨在解决其在真实 meme 图像场景下的潜在风险。研究构建含5万余实例的MemeSafetyBench基准数据集，评估多款VLM在单/多轮交互中的安全性，发现 meme 显著提升有害输出概率，凸显需加强生态化安全机制。**

- **链接: [http://arxiv.org/pdf/2505.15389v1](http://arxiv.org/pdf/2505.15389v1)**

> **作者:** DongGeon Lee; Joonwon Jang; Jihae Jeong; Hwanjo Yu
>
> **摘要:** Rapid deployment of vision-language models (VLMs) magnifies safety risks, yet most evaluations rely on artificial images. This study asks: How safe are current VLMs when confronted with meme images that ordinary users share? To investigate this question, we introduce MemeSafetyBench, a 50,430-instance benchmark pairing real meme images with both harmful and benign instructions. Using a comprehensive safety taxonomy and LLM-based instruction generation, we assess multiple VLMs across single and multi-turn interactions. We investigate how real-world memes influence harmful outputs, the mitigating effects of conversational context, and the relationship between model scale and safety metrics. Our findings demonstrate that VLMs show greater vulnerability to meme-based harmful prompts than to synthetic or typographic images. Memes significantly increase harmful responses and decrease refusals compared to text-only inputs. Though multi-turn interactions provide partial mitigation, elevated vulnerability persists. These results highlight the need for ecologically valid evaluations and stronger safety mechanisms.
>
---
#### [new 137] Fooling the LVLM Judges: Visual Biases in LVLM-Based Evaluation
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究大型视觉语言模型（LVLM）在图文评估中的视觉偏差问题。任务为探究对抗性视觉篡改是否会导致LVLM误判高分。通过构建多领域基准FRAME，发现LVLM易受图像诱导偏差影响，组合偏差放大效果，且提示策略无法有效缓解，凸显现有模型脆弱性，呼吁开发更鲁棒的评估系统。**

- **链接: [http://arxiv.org/pdf/2505.15249v1](http://arxiv.org/pdf/2505.15249v1)**

> **作者:** Yerin Hwang; Dongryeol Lee; Kyungmin Min; Taegwan Kang; Yong-il Kim; Kyomin Jung
>
> **备注:** (21pgs, 12 Tables, 9 Figures)
>
> **摘要:** Recently, large vision-language models (LVLMs) have emerged as the preferred tools for judging text-image alignment, yet their robustness along the visual modality remains underexplored. This work is the first study to address a key research question: Can adversarial visual manipulations systematically fool LVLM judges into assigning unfairly inflated scores? We define potential image induced biases within the context of T2I evaluation and examine how these biases affect the evaluations of LVLM judges. Moreover, we introduce a novel, fine-grained, multi-domain meta-evaluation benchmark named FRAME, which is deliberately constructed to exhibit diverse score distributions. By introducing the defined biases into the benchmark, we reveal that all tested LVLM judges exhibit vulnerability across all domains, consistently inflating scores for manipulated images. Further analysis reveals that combining multiple biases amplifies their effects, and pairwise evaluations are similarly susceptible. Moreover, we observe that visual biases persist under prompt-based mitigation strategies, highlighting the vulnerability of current LVLM evaluation systems and underscoring the urgent need for more robust LVLM judges.
>
---
#### [new 138] A Hybrid Quantum Classical Pipeline for X Ray Based Fracture Diagnosis
- **分类: eess.IV; cs.CV; cs.ET; cs.IT; cs.LG; math.IT**

- **简介: 论文提出混合量子经典管道用于X光骨折诊断，解决传统方法耗时低效及机器学习依赖大量数据的问题。通过PCA降维提取8特征，结合4量子比特幅度编码增强特征，形成16维向量后分类，实现99%准确率，特征提取时间减少82%。**

- **链接: [http://arxiv.org/pdf/2505.14716v1](http://arxiv.org/pdf/2505.14716v1)**

> **作者:** Sahil Tomar; Rajeshwar Tripathi; Sandeep Kumar
>
> **备注:** 8 pages
>
> **摘要:** Bone fractures are a leading cause of morbidity and disability worldwide, imposing significant clinical and economic burdens on healthcare systems. Traditional X ray interpretation is time consuming and error prone, while existing machine learning and deep learning solutions often demand extensive feature engineering, large, annotated datasets, and high computational resources. To address these challenges, a distributed hybrid quantum classical pipeline is proposed that first applies Principal Component Analysis (PCA) for dimensionality reduction and then leverages a 4 qubit quantum amplitude encoding circuit for feature enrichment. By fusing eight PCA derived features with eight quantum enhanced features into a 16 dimensional vector and then classifying with different machine learning models achieving 99% accuracy using a public multi region X ray dataset on par with state of the art transfer learning models while reducing feature extraction time by 82%.
>
---
#### [new 139] Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉-语言-动作(VLA)模型的跨任务泛化能力。针对现有模型难以迁移到新任务的问题，提出AGNOSTOS基准测试（含23个未见过任务）及X-ICM方法：通过LLM利用已见任务示例预测动作序列，并采用动力学引导采样策略选择相关演示，显著提升零样本泛化性能。**

- **链接: [http://arxiv.org/pdf/2505.15660v1](http://arxiv.org/pdf/2505.15660v1)**

> **作者:** Jiaming Zhou; Ke Ye; Jiayi Liu; Teli Ma; Zifang Wang; Ronghe Qiu; Kun-Yu Lin; Zhilin Zhao; Junwei Liang
>
> **备注:** Project Page: https://jiaming-zhou.github.io/AGNOSTOS
>
> **摘要:** The generalization capabilities of vision-language-action (VLA) models to unseen tasks are crucial to achieving general-purpose robotic manipulation in open-world settings. However, the cross-task generalization capabilities of existing VLA models remain significantly underexplored. To address this gap, we introduce AGNOSTOS, a novel simulation benchmark designed to rigorously evaluate cross-task zero-shot generalization in manipulation. AGNOSTOS comprises 23 unseen manipulation tasks for testing, distinct from common training task distributions, and incorporates two levels of generalization difficulty to assess robustness. Our systematic evaluation reveals that current VLA models, despite being trained on diverse datasets, struggle to generalize effectively to these unseen tasks. To overcome this limitation, we propose Cross-Task In-Context Manipulation (X-ICM), a method that conditions large language models (LLMs) on in-context demonstrations from seen tasks to predict action sequences for unseen tasks. Additionally, we introduce a dynamics-guided sample selection strategy that identifies relevant demonstrations by capturing cross-task dynamics. On AGNOSTOS, X-ICM significantly improves cross-task zero-shot generalization performance over leading VLAs. We believe AGNOSTOS and X-ICM will serve as valuable tools for advancing general-purpose robotic manipulation.
>
---
#### [new 140] Super-Resolution Optical Coherence Tomography Using Diffusion Model-Based Plug-and-Play Priors
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于扩散模型的PnP-DM框架，用于OCT超分辨率重建任务。旨在从稀疏采样角膜图像中生成高质量图像，解决传统方法结构模糊和噪声问题。方法结合扩散先验与MCMC采样，构建深度学习训练集，实验显示优于2D-UNet，提升临床高速OCT成像质量。**

- **链接: [http://arxiv.org/pdf/2505.14916v1](http://arxiv.org/pdf/2505.14916v1)**

> **作者:** Yaning Wang; Jinglun Yu; Wenhan Guo; Yu Sun; Jin U. Kang
>
> **摘要:** We propose an OCT super-resolution framework based on a plug-and-play diffusion model (PnP-DM) to reconstruct high-quality images from sparse measurements (OCT B-mode corneal images). Our method formulates reconstruction as an inverse problem, combining a diffusion prior with Markov chain Monte Carlo sampling for efficient posterior inference. We collect high-speed under-sampled B-mode corneal images and apply a deep learning-based up-sampling pipeline to build realistic training pairs. Evaluations on in vivo and ex vivo fish-eye corneal models show that PnP-DM outperforms conventional 2D-UNet baselines, producing sharper structures and better noise suppression. This approach advances high-fidelity OCT imaging in high-speed acquisition for clinical applications.
>
---
#### [new 141] UPTor: Unified 3D Human Pose Dynamics and Trajectory Prediction for Human-Robot Interaction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UPTor方法，解决人机交互中3D人体姿态与运动轨迹联合预测问题。通过运动转换技术，结合图注意力网络和非自回归transformer，首次统一预测全局坐标下的姿态与轨迹，同时构建DARKO导航数据集，实验证明其准确且实时。**

- **链接: [http://arxiv.org/pdf/2505.14866v1](http://arxiv.org/pdf/2505.14866v1)**

> **作者:** Nisarga Nilavadi; Andrey Rudenko; Timm Linder
>
> **备注:** Project page: https://nisarganc.github.io/UPTor-page/
>
> **摘要:** We introduce a unified approach to forecast the dynamics of human keypoints along with the motion trajectory based on a short sequence of input poses. While many studies address either full-body pose prediction or motion trajectory prediction, only a few attempt to merge them. We propose a motion transformation technique to simultaneously predict full-body pose and trajectory key-points in a global coordinate frame. We utilize an off-the-shelf 3D human pose estimation module, a graph attention network to encode the skeleton structure, and a compact, non-autoregressive transformer suitable for real-time motion prediction for human-robot interaction and human-aware navigation. We introduce a human navigation dataset ``DARKO'' with specific focus on navigational activities that are relevant for human-aware mobile robot navigation. We perform extensive evaluation on Human3.6M, CMU-Mocap, and our DARKO dataset. In comparison to prior work, we show that our approach is compact, real-time, and accurate in predicting human navigation motion across all datasets. Result animations, our dataset, and code will be available at https://nisarganc.github.io/UPTor-page/
>
---
#### [new 142] A Methodology to Evaluate Strategies Predicting Rankings on Unseen Domains
- **分类: cs.PF; cs.CV**

- **简介: 该论文提出一种评估策略的方法，用于预测实体（如算法）在未知领域排名表现，解决无需重新评估即可迁移已有领域测试结果的核心问题。通过leave-one-domain-out框架验证30种策略，分析40种无监督背景分割方法在53个视频域的排序预测，旨在优化跨领域性能预测任务。**

- **链接: [http://arxiv.org/pdf/2505.15595v1](http://arxiv.org/pdf/2505.15595v1)**

> **作者:** Sébastien Piérard; Adrien Deliège; Anaïs Halin; Marc Van Droogenbroeck
>
> **摘要:** Frequently, multiple entities (methods, algorithms, procedures, solutions, etc.) can be developed for a common task and applied across various domains that differ in the distribution of scenarios encountered. For example, in computer vision, the input data provided to image analysis methods depend on the type of sensor used, its location, and the scene content. However, a crucial difficulty remains: can we predict which entities will perform best in a new domain based on assessments on known domains, without having to carry out new and costly evaluations? This paper presents an original methodology to address this question, in a leave-one-domain-out fashion, for various application-specific preferences. We illustrate its use with 30 strategies to predict the rankings of 40 entities (unsupervised background subtraction methods) on 53 domains (videos).
>
---
## 更新

#### [replaced 001] Denoising Score Distillation: From Noisy Diffusion Pretraining to One-Step High-Quality Generation
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07578v2](http://arxiv.org/pdf/2503.07578v2)**

> **作者:** Tianyu Chen; Yasi Zhang; Zhendong Wang; Ying Nian Wu; Oscar Leong; Mingyuan Zhou
>
> **备注:** First Author and Second Author contributed equally to this work. The last two authors equally advised this work
>
> **摘要:** Diffusion models have achieved remarkable success in generating high-resolution, realistic images across diverse natural distributions. However, their performance heavily relies on high-quality training data, making it challenging to learn meaningful distributions from corrupted samples. This limitation restricts their applicability in scientific domains where clean data is scarce or costly to obtain. In this work, we introduce denoising score distillation (DSD), a surprisingly effective and novel approach for training high-quality generative models from low-quality data. DSD first pretrains a diffusion model exclusively on noisy, corrupted samples and then distills it into a one-step generator capable of producing refined, clean outputs. While score distillation is traditionally viewed as a method to accelerate diffusion models, we show that it can also significantly enhance sample quality, particularly when starting from a degraded teacher model. Across varying noise levels and datasets, DSD consistently improves generative performancewe summarize our empirical evidence in Fig. 1. Furthermore, we provide theoretical insights showing that, in a linear model setting, DSD identifies the eigenspace of the clean data distributions covariance matrix, implicitly regularizing the generator. This perspective reframes score distillation as not only a tool for efficiency but also a mechanism for improving generative models, particularly in low-quality data settings.
>
---
#### [replaced 002] MagicTailor: Component-Controllable Personalization in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.13370v3](http://arxiv.org/pdf/2410.13370v3)**

> **作者:** Donghao Zhou; Jiancheng Huang; Jinbin Bai; Jiaze Wang; Hao Chen; Guangyong Chen; Xiaowei Hu; Pheng-Ann Heng
>
> **备注:** Accepted by IJCAI2025 (Project page: https://correr-zhou.github.io/MagicTailor)
>
> **摘要:** Text-to-image diffusion models can generate high-quality images but lack fine-grained control of visual concepts, limiting their creativity. Thus, we introduce component-controllable personalization, a new task that enables users to customize and reconfigure individual components within concepts. This task faces two challenges: semantic pollution, where undesired elements disrupt the target concept, and semantic imbalance, which causes disproportionate learning of the target concept and component. To address these, we design MagicTailor, a framework that uses Dynamic Masked Degradation to adaptively perturb unwanted visual semantics and Dual-Stream Balancing for more balanced learning of desired visual semantics. The experimental results show that MagicTailor achieves superior performance in this task and enables more personalized and creative image generation.
>
---
#### [replaced 003] FaVoR: Features via Voxel Rendering for Camera Relocalization
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.07571v4](http://arxiv.org/pdf/2409.07571v4)**

> **作者:** Vincenzo Polizzi; Marco Cannici; Davide Scaramuzza; Jonathan Kelly
>
> **备注:** In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Tucson, Arizona, US, Feb 28-Mar 4, 2025
>
> **摘要:** Camera relocalization methods range from dense image alignment to direct camera pose regression from a query image. Among these, sparse feature matching stands out as an efficient, versatile, and generally lightweight approach with numerous applications. However, feature-based methods often struggle with significant viewpoint and appearance changes, leading to matching failures and inaccurate pose estimates. To overcome this limitation, we propose a novel approach that leverages a globally sparse yet locally dense 3D representation of 2D features. By tracking and triangulating landmarks over a sequence of frames, we construct a sparse voxel map optimized to render image patch descriptors observed during tracking. Given an initial pose estimate, we first synthesize descriptors from the voxels using volumetric rendering and then perform feature matching to estimate the camera pose. This methodology enables the generation of descriptors for unseen views, enhancing robustness to view changes. We extensively evaluate our method on the 7-Scenes and Cambridge Landmarks datasets. Our results show that our method significantly outperforms existing state-of-the-art feature representation techniques in indoor environments, achieving up to a 39% improvement in median translation error. Additionally, our approach yields comparable results to other methods for outdoor scenarios while maintaining lower memory and computational costs.
>
---
#### [replaced 004] SPRMamba: Surgical Phase Recognition for Endoscopic Submucosal Dissection with Mamba
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.12108v2](http://arxiv.org/pdf/2409.12108v2)**

> **作者:** Xiangning Zhang; Qingwei Zhang; Jinnan Chen; Chengfeng Zhou; Yaqi Wang; Zhengjie Zhang; Xiaobo Li; Dahong Qian
>
> **摘要:** Endoscopic Submucosal Dissection (ESD) is a minimally invasive procedure initially developed for early gastric cancer treatment and has expanded to address diverse gastrointestinal lesions. While computer-assisted surgery (CAS) systems enhance ESD precision and safety, their efficacy hinges on accurate real-time surgical phase recognition, a task complicated by ESD's inherent complexity, including heterogeneous lesion characteristics and dynamic tissue interactions. Existing video-based phase recognition algorithms, constrained by inefficient temporal context modeling, exhibit limited performance in capturing fine-grained phase transitions and long-range dependencies. To overcome these limitations, we propose SPRMamba, a novel framework integrating a Mamba-based architecture with a Scaled Residual TranMamba (SRTM) block to synergize long-term temporal modeling and localized detail extraction. SPRMamba further introduces the Hierarchical Sampling Strategy to optimize computational efficiency, enabling real-time processing critical for clinical deployment. Evaluated on the ESD385 dataset and the cholecystectomy benchmark Cholec80, SPRMamba achieves state-of-the-art performance (87.64% accuracy on ESD385, +1.0% over prior methods), demonstrating robust generalizability across surgical workflows. This advancement bridges the gap between computational efficiency and temporal sensitivity, offering a transformative tool for intraoperative guidance and skill assessment in ESD surgery. The code is accessible at https://github.com/Zxnyyyyy/SPRMamba.
>
---
#### [replaced 005] Augmenting Chest X-ray Datasets with Non-Expert Annotations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2309.02244v3](http://arxiv.org/pdf/2309.02244v3)**

> **作者:** Veronika Cheplygina; Cathrine Damgaard; Trine Naja Eriksen; Dovile Juodelyte; Amelia Jiménez-Sánchez
>
> **备注:** Medical Image Understanding and Analysis Conference - MIUA 2025
>
> **摘要:** The advancement of machine learning algorithms in medical image analysis requires the expansion of training datasets. A popular and cost-effective approach is automated annotation extraction from free-text medical reports, primarily due to the high costs associated with expert clinicians annotating medical images, such as chest X-rays. However, it has been shown that the resulting datasets are susceptible to biases and shortcuts. Another strategy to increase the size of a dataset is crowdsourcing, a widely adopted practice in general computer vision with some success in medical image analysis. In a similar vein to crowdsourcing, we enhance two publicly available chest X-ray datasets by incorporating non-expert annotations. However, instead of using diagnostic labels, we annotate shortcuts in the form of tubes. We collect 3.5k chest drain annotations for NIH-CXR14, and 1k annotations for four different tube types in PadChest, and create the Non-Expert Annotations of Tubes in X-rays (NEATX) dataset. We train a chest drain detector with the non-expert annotations that generalizes well to expert labels. Moreover, we compare our annotations to those provided by experts and show "moderate" to "almost perfect" agreement. Finally, we present a pathology agreement study to raise awareness about the quality of ground truth annotations. We make our dataset available on Zenodo at https://zenodo.org/records/14944064 and our code available at https://github.com/purrlab/chestxr-label-reliability.
>
---
#### [replaced 006] SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12448v2](http://arxiv.org/pdf/2505.12448v2)**

> **作者:** Yang Liu; Ming Ma; Xiaomin Yu; Pengxiang Ding; Han Zhao; Mingyang Sun; Siteng Huang; Donglin Wang
>
> **摘要:** Despite impressive advancements in Visual-Language Models (VLMs) for multi-modal tasks, their reliance on RGB inputs limits precise spatial understanding. Existing methods for integrating spatial cues, such as point clouds or depth, either require specialized sensors or fail to effectively exploit depth information for higher-order reasoning. To this end, we propose a novel Spatial Sense and Reasoning method, dubbed SSR, a novel framework that transforms raw depth data into structured, interpretable textual rationales. These textual rationales serve as meaningful intermediate representations to significantly enhance spatial reasoning capabilities. Additionally, we leverage knowledge distillation to compress the generated rationales into compact latent embeddings, which facilitate resource-efficient and plug-and-play integration into existing VLMs without retraining. To enable comprehensive evaluation, we introduce a new dataset named SSR-CoT, a million-scale visual-language reasoning dataset enriched with intermediate spatial reasoning annotations, and present SSRBench, a comprehensive multi-task benchmark. Extensive experiments on multiple benchmarks demonstrate SSR substantially improves depth utilization and enhances spatial reasoning, thereby advancing VLMs toward more human-like multi-modal understanding. Our project page is at https://yliu-cs.github.io/SSR.
>
---
#### [replaced 007] An Information Theory-inspired Strategy for Automatic Network Pruning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2108.08532v4](http://arxiv.org/pdf/2108.08532v4)**

> **作者:** Xiawu Zheng; Yuexiao Ma; Teng Xi; Gang Zhang; Errui Ding; Yuchao Li; Jie Chen; Yonghong Tian; Rongrong Ji
>
> **备注:** Accepted by IJCV
>
> **摘要:** Despite superior performance on many computer vision tasks, deep convolution neural networks are well known to be compressed on devices that have resource constraints. Most existing network pruning methods require laborious human efforts and prohibitive computation resources, especially when the constraints are changed. This practically limits the application of model compression when the model needs to be deployed on a wide range of devices. Besides, existing methods are still challenged by the missing theoretical guidance. In this paper we propose an information theory-inspired strategy for automatic model compression. The principle behind our method is the information bottleneck theory, i.e., the hidden representation should compress information with each other. We thus introduce the normalized Hilbert-Schmidt Independence Criterion (nHSIC) on network activations as a stable and generalized indicator of layer importance. When a certain resource constraint is given, we integrate the HSIC indicator with the constraint to transform the architecture search problem into a linear programming problem with quadratic constraints. Such a problem is easily solved by a convex optimization method with a few seconds. We also provide a rigorous proof to reveal that optimizing the normalized HSIC simultaneously minimizes the mutual information between different layers. Without any search process, our method achieves better compression tradeoffs comparing to the state-of-the-art compression algorithms. For instance, with ResNet-50, we achieve a 45.3%-FLOPs reduction, with a 75.75 top-1 accuracy on ImageNet. Codes are avaliable at https://github.com/MAC-AutoML/ITPruner/tree/master.
>
---
#### [replaced 008] Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04046v2](http://arxiv.org/pdf/2505.04046v2)**

> **作者:** Xuyang Wang; Siyuan Duan; Qizhi Li; Guiduo Duan; Yuan Sun; Dezhong Peng
>
> **备注:** 11 pages, 11 figures, accepted by IJCAI 2025
>
> **摘要:** Trustworthy multi-view learning has attracted extensive attention because evidence learning can provide reliable uncertainty estimation to enhance the credibility of multi-view predictions. Existing trusted multi-view learning methods implicitly assume that multi-view data is secure. However, in safety-sensitive applications such as autonomous driving and security monitoring, multi-view data often faces threats from adversarial perturbations, thereby deceiving or disrupting multi-view models. This inevitably leads to the adversarial unreliability problem (AUP) in trusted multi-view learning. To overcome this tricky problem, we propose a novel multi-view learning framework, namely Reliable Disentanglement Multi-view Learning (RDML). Specifically, we first propose evidential disentanglement learning to decompose each view into clean and adversarial parts under the guidance of corresponding evidences, which is extracted by a pretrained evidence extractor. Then, we employ the feature recalibration module to mitigate the negative impact of adversarial perturbations and extract potential informative features from them. Finally, to further ignore the irreparable adversarial interferences, a view-level evidential attention mechanism is designed. Extensive experiments on multi-view classification tasks with adversarial attacks show that RDML outperforms the state-of-the-art methods by a relatively large margin. Our code is available at https://github.com/Willy1005/2025-IJCAI-RDML.
>
---
#### [replaced 009] PixelWorld: Towards Perceiving Everything as Pixels
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19339v2](http://arxiv.org/pdf/2501.19339v2)**

> **作者:** Zhiheng Lyu; Xueguang Ma; Wenhu Chen
>
> **摘要:** Recent agentic language models increasingly need to interact directly with real-world environments containing intertwined visual and textual information through raw camera pixels, rather than relying on separate image and tokenized text processing, underscoring the necessity of a unified perception paradigm. To close this gap, we explore this idea through Perceive Everything as Pixels (PEAP) and release PixelWorld, a benchmark that renders natural-language, tabular, mathematical and diagrammatic inputs into a single pixel space. Experiments show that PEAP attains competitive accuracy on semantic-understanding tasks, indicating that a vision transformer can capture global textual semantics without explicit tokens. In contrast, reasoning-intensive benchmarks (math and code) exhibit sharp performance drops; however, Chain-of-Thought prompting partially mitigates this gap, hinting that explicit reasoning traces compensate for the missing token structure. We also find that when visual and textual information are closely integrated, representing everything as pixels reduces preprocessing complexity and avoids misalignment issues that often arise in separate pipelines. PixelWorld therefore serves as a practical benchmark for evaluating unified vision-language models and supports broader exploration of PEAP across diverse tasks.
>
---
#### [replaced 010] FG-CLIP: Fine-Grained Visual and Textual Alignment
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.05071v3](http://arxiv.org/pdf/2505.05071v3)**

> **作者:** Chunyu Xie; Bin Wang; Fanjing Kong; Jincheng Li; Dawei Liang; Gengshen Zhang; Dawei Leng; Yuhui Yin
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) excels in multimodal tasks such as image-text retrieval and zero-shot classification but struggles with fine-grained understanding due to its focus on coarse-grained short captions. To address this, we propose Fine-Grained CLIP (FG-CLIP), which enhances fine-grained understanding through three key innovations. First, we leverage large multimodal models to generate 1.6 billion long caption-image pairs for capturing global-level semantic details. Second, a high-quality dataset is constructed with 12 million images and 40 million region-specific bounding boxes aligned with detailed captions to ensure precise, context-rich representations. Third, 10 million hard fine-grained negative samples are incorporated to improve the model's ability to distinguish subtle semantic differences. We construct a comprehensive dataset, termed FineHARD, by integrating high-quality region-specific annotations with hard fine-grained negative samples. Corresponding training methods are meticulously designed for these data. Extensive experiments demonstrate that FG-CLIP outperforms the original CLIP and other state-of-the-art methods across various downstream tasks, including fine-grained understanding, open-vocabulary object detection, image-text retrieval, and general multimodal benchmarks. These results highlight FG-CLIP's effectiveness in capturing fine-grained image details and improving overall model performance. The data, code, and models are available at https://github.com/360CVGroup/FG-CLIP.
>
---
#### [replaced 011] Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.00059v2](http://arxiv.org/pdf/2503.00059v2)**

> **作者:** Rui Hu; Delai Qiu; Shuyu Wei; Jiaming Zhang; Yining Wang; Shengping Liu; Jitao Sang
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Omnimodal Large Language Models (OLLMs) have shown significant progress in integrating vision and text, but still struggle with integrating vision and audio, often exhibiting suboptimal performance when processing audio queries compared to text queries. This disparity is primarily due to insufficient alignment between vision and audio modalities during training, leading to inadequate attention to visual information when using audio queries. To mitigate this issue, we propose a Self-Knowledge Distillation (Self-KD) training method where the vision-text component of the OLLM serves as the teacher and the vision-audio component as the student. This enables the model to process audio in a manner analogous to its text processing. Our experimental results demonstrate that Self-KD is an effective method for enhancing the vision-audio capabilities of OLLMs by learning from the vision-text components, which subsequently improves the interaction between audio and images and results in improved performance on multimodal tasks.
>
---
#### [replaced 012] Generalizing Medical Image Representations via Quaternion Wavelet Networks
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.10224v5](http://arxiv.org/pdf/2310.10224v5)**

> **作者:** Luigi Sigillo; Eleonora Grassucci; Aurelio Uncini; Danilo Comminiello
>
> **备注:** Paper accepted to Neurocomputing Journal
>
> **摘要:** Neural network generalizability is becoming a broad research field due to the increasing availability of datasets from different sources and for various tasks. This issue is even wider when processing medical data, where a lack of methodological standards causes large variations being provided by different imaging centers or acquired with various devices and cofactors. To overcome these limitations, we introduce a novel, generalizable, data- and task-agnostic framework able to extract salient features from medical images. The proposed quaternion wavelet network (QUAVE) can be easily integrated with any pre-existing medical image analysis or synthesis task, and it can be involved with real, quaternion, or hypercomplex-valued models, generalizing their adoption to single-channel data. QUAVE first extracts different sub-bands through the quaternion wavelet transform, resulting in both low-frequency/approximation bands and high-frequency/fine-grained features. Then, it weighs the most representative set of sub-bands to be involved as input to any other neural model for image processing, replacing standard data samples. We conduct an extensive experimental evaluation comprising different datasets, diverse image analysis, and synthesis tasks including reconstruction, segmentation, and modality translation. We also evaluate QUAVE in combination with both real and quaternion-valued models. Results demonstrate the effectiveness and the generalizability of the proposed framework that improves network performance while being flexible to be adopted in manifold scenarios and robust to domain shifts. The full code is available at: https://github.com/ispamm/QWT.
>
---
#### [replaced 013] Dress-1-to-3: Single Image to Simulation-Ready 3D Outfit with Diffusion Prior and Differentiable Physics
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03449v2](http://arxiv.org/pdf/2502.03449v2)**

> **作者:** Xuan Li; Chang Yu; Wenxin Du; Ying Jiang; Tianyi Xie; Yunuo Chen; Yin Yang; Chenfanfu Jiang
>
> **备注:** Project page: https://dress-1-to-3.github.io/
>
> **摘要:** Recent advances in large models have significantly advanced image-to-3D reconstruction. However, the generated models are often fused into a single piece, limiting their applicability in downstream tasks. This paper focuses on 3D garment generation, a key area for applications like virtual try-on with dynamic garment animations, which require garments to be separable and simulation-ready. We introduce Dress-1-to-3, a novel pipeline that reconstructs physics-plausible, simulation-ready separated garments with sewing patterns and humans from an in-the-wild image. Starting with the image, our approach combines a pre-trained image-to-sewing pattern generation model for creating coarse sewing patterns with a pre-trained multi-view diffusion model to produce multi-view images. The sewing pattern is further refined using a differentiable garment simulator based on the generated multi-view images. Versatile experiments demonstrate that our optimization approach substantially enhances the geometric alignment of the reconstructed 3D garments and humans with the input image. Furthermore, by integrating a texture generation module and a human motion generation module, we produce customized physics-plausible and realistic dynamic garment demonstrations. Project page: https://dress-1-to-3.github.io/
>
---
#### [replaced 014] Selective Structured State Space for Multispectral-fused Small Target Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14043v2](http://arxiv.org/pdf/2505.14043v2)**

> **作者:** Qianqian Zhang; WeiJun Wang; Yunxing Liu; Li Zhou; Hao Zhao; Junshe An; Zihan Wang
>
> **备注:** This work was submitted to CVPR 2025, but was rejected after being reviewed by 7 reviewers. After revision, it is currently under review
>
> **摘要:** Target detection in high-resolution remote sensing imagery faces challenges due to the low recognition accuracy of small targets and high computational costs. The computational complexity of the Transformer architecture increases quadratically with image resolution, while Convolutional Neural Networks (CNN) architectures are forced to stack deeper convolutional layers to expand their receptive fields, leading to an explosive growth in computational demands. To address these computational constraints, we leverage Mamba's linear complexity for efficiency. However, Mamba's performance declines for small targets, primarily because small targets occupy a limited area in the image and have limited semantic information. Accurate identification of these small targets necessitates not only Mamba's global attention capabilities but also the precise capture of fine local details. To this end, we enhance Mamba by developing the Enhanced Small Target Detection (ESTD) module and the Convolutional Attention Residual Gate (CARG) module. The ESTD module bolsters local attention to capture fine-grained details, while the CARG module, built upon Mamba, emphasizes spatial and channel-wise information, collectively improving the model's ability to capture distinctive representations of small targets. Additionally, to highlight the semantic representation of small targets, we design a Mask Enhanced Pixel-level Fusion (MEPF) module for multispectral fusion, which enhances target features by effectively fusing visible and infrared multimodal information.
>
---
#### [replaced 015] Opt-In Art: Learning Art Styles Only from Few Examples
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.00176v3](http://arxiv.org/pdf/2412.00176v3)**

> **作者:** Hui Ren; Joanna Materzynska; Rohit Gandikota; David Bau; Antonio Torralba
>
> **摘要:** We explore whether pre-training on datasets with paintings is necessary for a model to learn an artistic style with only a few examples. To investigate this, we train a text-to-image model exclusively on photographs, without access to any painting-related content. We show that it is possible to adapt a model that is trained without paintings to an artistic style, given only few examples. User studies and automatic evaluations confirm that our model (post-adaptation) performs on par with state-of-the-art models trained on massive datasets that contain artistic content like paintings, drawings or illustrations. Finally, using data attribution techniques, we analyze how both artistic and non-artistic datasets contribute to generating artistic-style images. Surprisingly, our findings suggest that high-quality artistic outputs can be achieved without prior exposure to artistic data, indicating that artistic style generation can occur in a controlled, opt-in manner using only a limited, carefully selected set of training examples.
>
---
#### [replaced 016] Local Clustering for Lung Cancer Image Classification via Sparse Solution Technique
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.08800v2](http://arxiv.org/pdf/2407.08800v2)**

> **作者:** Jackson Hamel; Ming-Jun Lai; Zhaiming Shen; Ye Tian
>
> **摘要:** In this work, we propose to use a local clustering approach based on the sparse solution technique to study the medical image, especially the lung cancer image classification task. We view images as the vertices in a weighted graph and the similarity between a pair of images as the edges in the graph. The vertices within the same cluster can be assumed to share similar features and properties, thus making the applications of graph clustering techniques very useful for image classification. Recently, the approach based on the sparse solutions of linear systems for graph clustering has been found to identify clusters more efficiently than traditional clustering methods such as spectral clustering. We propose to use the two newly developed local clustering methods based on sparse solution of linear system for image classification. In addition, we employ a box spline-based tight-wavelet-framelet method to clean these images and help build a better adjacency matrix before clustering. The performance of our methods is shown to be very effective in classifying images. Our approach is significantly more efficient and either favorable or equally effective compared with other state-of-the-art approaches. Finally, we shall make a remark by pointing out two image deformation methods to build up more artificial image data to increase the number of labeled images.
>
---
#### [replaced 017] Boosting Few-Shot Open-Set Object Detection via Prompt Learning and Robust Decision Boundary
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.18443v3](http://arxiv.org/pdf/2406.18443v3)**

> **作者:** Zhaowei Wu; Binyi Su; Qichuan Geng; Hua Zhang; Zhong Zhou
>
> **备注:** Accepted to IJCAI 2025
>
> **摘要:** Few-shot Open-set Object Detection (FOOD) poses a challenge in many open-world scenarios. It aims to train an open-set detector to detect known objects while rejecting unknowns with scarce training samples. Existing FOOD methods are subject to limited visual information, and often exhibit an ambiguous decision boundary between known and unknown classes. To address these limitations, we propose the first prompt-based few-shot open-set object detection framework, which exploits additional textual information and delves into constructing a robust decision boundary for unknown rejection. Specifically, as no available training data for unknown classes, we select pseudo-unknown samples with Attribution-Gradient based Pseudo-unknown Mining (AGPM), which leverages the discrepancy in attribution gradients to quantify uncertainty. Subsequently, we propose Conditional Evidence Decoupling (CED) to decouple and extract distinct knowledge from selected pseudo-unknown samples by eliminating opposing evidence. This optimization process can enhance the discrimination between known and unknown classes. To further regularize the model and form a robust decision boundary for unknown rejection, we introduce Abnormal Distribution Calibration (ADC) to calibrate the output probability distribution of local abnormal features in pseudo-unknown samples. Our method achieves superior performance over previous state-of-the-art approaches, improving the average recall of unknown class by 7.24% across all shots in VOC10-5-5 dataset settings and 1.38% in VOC-COCO dataset settings. Our source code is available at https://gitee.com/VR_NAVE/ced-food.
>
---
#### [replaced 018] SANER: Annotation-free Societal Attribute Neutralizer for Debiasing CLIP
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.10202v4](http://arxiv.org/pdf/2408.10202v4)**

> **作者:** Yusuke Hirota; Min-Hung Chen; Chien-Yi Wang; Yuta Nakashima; Yu-Chiang Frank Wang; Ryo Hachiuma
>
> **备注:** ICLR 2025
>
> **摘要:** Large-scale vision-language models, such as CLIP, are known to contain societal bias regarding protected attributes (e.g., gender, age). This paper aims to address the problems of societal bias in CLIP. Although previous studies have proposed to debias societal bias through adversarial learning or test-time projecting, our comprehensive study of these works identifies two critical limitations: 1) loss of attribute information when it is explicitly disclosed in the input and 2) use of the attribute annotations during debiasing process. To mitigate societal bias in CLIP and overcome these limitations simultaneously, we introduce a simple-yet-effective debiasing method called SANER (societal attribute neutralizer) that eliminates attribute information from CLIP text features only of attribute-neutral descriptions. Experimental results show that SANER, which does not require attribute annotations and preserves original information for attribute-specific descriptions, demonstrates superior debiasing ability than the existing methods.
>
---
#### [replaced 019] ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20930v2](http://arxiv.org/pdf/2504.20930v2)**

> **作者:** Ziqing Fan; Cheng Liang; Chaoyi Wu; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **摘要:** Recent advances in reasoning-enhanced large language models (LLMs) and multimodal LLMs (MLLMs) have significantly improved performance in complex tasks, yet medical AI models often overlook the structured reasoning processes inherent in clinical practice. In this work, we present ChestX-Reasoner, a radiology diagnosis MLLM designed to leverage process supervision mined directly from clinical reports, reflecting the step-by-step reasoning followed by radiologists. We construct a large dataset by extracting and refining reasoning chains from routine radiology reports. Our two-stage training framework combines supervised fine-tuning and reinforcement learning guided by process rewards to better align model reasoning with clinical standards. We introduce RadRBench-CXR, a comprehensive benchmark featuring 59K visual question answering samples with 301K clinically validated reasoning steps, and propose RadRScore, a metric evaluating reasoning factuality, completeness, and effectiveness. ChestX-Reasoner outperforms existing medical and general-domain MLLMs in both diagnostic accuracy and reasoning ability, achieving 16%, 5.9%, and 18% improvements in reasoning ability compared to the best medical MLLM, the best general MLLM, and its base model, respectively, as well as 3.3%, 24%, and 27% improvements in outcome accuracy. All resources are open-sourced to facilitate further research in medical reasoning MLLMs.
>
---
#### [replaced 020] DINOv2-powered Few-Shot Semantic Segmentation: A Unified Framework via Cross-Model Distillation and 4D Correlation Mining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15669v2](http://arxiv.org/pdf/2504.15669v2)**

> **作者:** Wei Zhuo; Zhiyue Tang; Wufeng Xue; Hao Ding; Linlin Shen
>
> **摘要:** Few-shot semantic segmentation has gained increasing interest due to its generalization capability, i.e., segmenting pixels of novel classes requiring only a few annotated images. Prior work has focused on meta-learning for support-query matching, with extensive development in both prototype-based and aggregation-based methods. To address data scarcity, recent approaches have turned to foundation models to enhance representation transferability for novel class segmentation. Among them, a hybrid dual-modal framework including both DINOv2 and SAM has garnered attention due to their complementary capabilities. We wonder "can we build a unified model with knowledge from both foundation models?" To this end, we propose FS-DINO, with only DINOv2's encoder and a lightweight segmenter. The segmenter features a bottleneck adapter, a meta-visual prompt generator based on dense similarities and semantic embeddings, and a decoder. Through coarse-to-fine cross-model distillation, we effectively integrate SAM's knowledge into our lightweight segmenter, which can be further enhanced by 4D correlation mining on support-query pairs. Extensive experiments on COCO-20i, PASCAL-5i, and FSS-1000 demonstrate the effectiveness and superiority of our method.
>
---
#### [replaced 021] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06141v3](http://arxiv.org/pdf/2412.06141v3)**

> **作者:** Kangyu Zhu; Peng Xia; Yun Li; Hongtu Zhu; Sheng Wang; Huaxiu Yao
>
> **备注:** ICML 2025
>
> **摘要:** The advancement of Large Vision-Language Models (LVLMs) has propelled their application in the medical field. However, Medical LVLMs (Med-LVLMs) encounter factuality challenges due to modality misalignment, where the models prioritize textual knowledge over visual input, leading to hallucinations that contradict information in medical images. Previous attempts to enhance modality alignment in Med-LVLMs through preference optimization have inadequately mitigated clinical relevance in preference data, making these samples easily distinguishable and reducing alignment effectiveness. To address this challenge, we propose MMedPO, a novel multimodal medical preference optimization approach that considers the clinical relevance of preference samples to enhance Med-LVLM alignment. MMedPO curates multimodal preference data by introducing two types of dispreference: (1) plausible hallucinations injected through target Med-LVLMs or GPT-4o to produce medically inaccurate responses, and (2) lesion region neglect achieved through local lesion-noising, disrupting visual understanding of critical areas. We then calculate clinical relevance for each sample based on scores from multiple Med-LLMs and visual tools, and integrate these scores into the preference optimization process as weights, enabling effective alignment. Our experiments demonstrate that MMedPO significantly enhances factual accuracy in Med-LVLMs, achieving substantial improvements over existing preference optimization methods by averaging 14.2% and 51.7% across the Med-VQA and report generation tasks. Our code are available in https://github.com/aiming-lab/MMedPO.
>
---
#### [replaced 022] Teaching Metric Distance to Autoregressive Multimodal Foundational Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02379v2](http://arxiv.org/pdf/2503.02379v2)**

> **作者:** Jiwan Chung; Saejin Kim; Yongrae Jo; Jaewoo Park; Dongjun Min; Youngjae Yu
>
> **摘要:** As large language models expand beyond natural language to domains such as mathematics, multimodal understanding, and embodied agents, tokens increasingly reflect metric relationships rather than purely linguistic meaning. We introduce DIST2Loss, a distance-aware framework designed to train autoregressive discrete models by leveraging predefined distance relationships among output tokens. At its core, DIST2Loss transforms continuous exponential family distributions derived from inherent distance metrics into discrete, categorical optimization targets compatible with the models' architectures. This approach enables the models to learn and preserve meaningful distance relationships during token generation while maintaining compatibility with existing architectures. Empirical evaluations show consistent performance gains in diverse multimodal applications, including visual grounding, robotic manipulation, generative reward modeling, and image generation using vector-quantized features. These improvements are most notable in low-data regimes, demonstrating DIST2Loss's strength under resource constraints.
>
---
#### [replaced 023] Towards Real-world Debiasing: Rethinking Evaluation, Challenge, and Solution
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.15240v4](http://arxiv.org/pdf/2405.15240v4)**

> **作者:** Peng Kuang; Zhibo Wang; Zhixuan Chu; Jingyi Wang; Kui Ren
>
> **备注:** 9 pages of main paper, 17 pages of appendix
>
> **摘要:** Spurious correlations in training data significantly hinder the generalization capability of machine learning models when faced with distribution shifts, leading to the proposition of numberous debiasing methods. However, it remains to be asked: \textit{Do existing benchmarks for debiasing really represent biases in the real world?} Recent works attempt to address such concerns by sampling from real-world data (instead of synthesizing) according to some predefined biased distributions to ensure the realism of individual samples. However, the realism of the biased distribution is more critical yet challenging and underexplored due to the complexity of real-world bias distributions. To tackle the problem, we propose a fine-grained framework for analyzing biased distributions, based on which we empirically and theoretically identify key characteristics of biased distributions in the real world that are poorly represented by existing benchmarks. Towards applicable debiasing in the real world, we further introduce two novel real-world-inspired biases to bridge this gap and build a systematic evaluation framework for real-world debiasing, RDBench\footnote{RDBench: Code to be released. Preliminary version in supplementary material for anonimized review.}. Furthermore, focusing on the practical setting of debiasing w/o bias label, we find real-world biases pose a novel \textit{Sparse bias capturing} challenge to the existing paradigm. We propose a simple yet effective approach named Debias in Destruction (DiD), to address the challenge, whose effectiveness is validated with extensive experiments on 8 datasets of various biased distributions.
>
---
#### [replaced 024] Video-GPT via Next Clip Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12489v2](http://arxiv.org/pdf/2505.12489v2)**

> **作者:** Shaobin Zhuang; Zhipeng Huang; Ying Zhang; Fangyikang Wang; Canmiao Fu; Binxin Yang; Chong Sun; Chen Li; Yali Wang
>
> **备注:** 22 pages, 12 figures, 18 tables
>
> **摘要:** GPT has shown its remarkable success in natural language processing. However, the language sequence is not sufficient to describe spatial-temporal details in the visual world. Alternatively, the video sequence is good at capturing such details. Motivated by this fact, we propose a concise Video-GPT in this paper by treating video as new language for visual world modeling. By analogy to next token prediction in GPT, we introduce a novel next clip diffusion paradigm for pretraining Video-GPT. Different from the previous works, this distinct paradigm allows Video-GPT to tackle both short-term generation and long-term prediction, by autoregressively denoising the noisy clip according to the clean clips in the history. Extensive experiments show our Video-GPT achieves the state-of-the-art performance on video prediction, which is the key factor towards world modeling (Physics-IQ Benchmark: Video-GPT 34.97 vs. Kling 23.64 vs. Wan 20.89). Moreover, it can be well adapted on 6 mainstream video tasks in both video generation and understanding, showing its great generalization capacity in downstream. The project page is at https://zhuangshaobin.github.io/Video-GPT.github.io/.
>
---
#### [replaced 025] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14846v2](http://arxiv.org/pdf/2502.14846v2)**

> **作者:** Yue Yang; Ajay Patel; Matt Deitke; Tanmay Gupta; Luca Weihs; Andrew Head; Mark Yatskar; Chris Callison-Burch; Ranjay Krishna; Aniruddha Kembhavi; Christopher Clark
>
> **备注:** Published in ACL 2025, project page: https://yueyang1996.github.io/cosyn/
>
> **摘要:** Reasoning about images with rich text, such as charts and documents, is a critical application of vision-language models (VLMs). However, VLMs often struggle in these domains due to the scarcity of diverse text-rich vision-language data. To address this challenge, we present CoSyn, a framework that leverages the coding capabilities of text-only large language models (LLMs) to automatically create synthetic text-rich multimodal data. Given input text describing a target domain (e.g., "nutrition fact labels"), CoSyn prompts an LLM to generate code (Python, HTML, LaTeX, etc.) for rendering synthetic images. With the underlying code as textual representations of the synthetic images, CoSyn can generate high-quality instruction-tuning data, again relying on a text-only LLM. Using CoSyn, we constructed a dataset comprising 400K images and 2.7M rows of vision-language instruction-tuning data. Comprehensive experiments on seven benchmarks demonstrate that models trained on our synthetic data achieve state-of-the-art performance among competitive open-source models, including Llama 3.2, and surpass proprietary models such as GPT-4V and Gemini 1.5 Flash. Furthermore, CoSyn can produce synthetic pointing data, enabling VLMs to ground information within input images, showcasing its potential for developing multimodal agents capable of acting in real-world environments.
>
---
#### [replaced 026] SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network
- **分类: cs.NE; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.06488v4](http://arxiv.org/pdf/2310.06488v4)**

> **作者:** Changze Lv; Tianlong Li; Wenhao Liu; Yufei Gu; Jianhan Xu; Cenyuan Zhang; Muling Wu; Xiaoqing Zheng; Xuanjing Huang
>
> **摘要:** Spiking Neural Networks (SNNs) have emerged as a promising alternative to conventional Artificial Neural Networks (ANNs), demonstrating comparable performance in both visual and linguistic tasks while offering the advantage of improved energy efficiency. Despite these advancements, the integration of linguistic and visual features into a unified representation through spike trains poses a significant challenge, and the application of SNNs to multimodal scenarios remains largely unexplored. This paper presents SpikeCLIP, a novel framework designed to bridge the modality gap in spike-based computation. Our approach employs a two-step recipe: an ``alignment pre-training'' to align features across modalities, followed by a ``dual-loss fine-tuning'' to refine the model's performance. Extensive experiments reveal that SNNs achieve results on par with ANNs while substantially reducing energy consumption across various datasets commonly used for multimodal model evaluation. Furthermore, SpikeCLIP maintains robust image classification capabilities, even when dealing with classes that fall outside predefined categories. This study marks a significant advancement in the development of energy-efficient and biologically plausible multimodal learning systems. Our code is available at https://github.com/Lvchangze/SpikeCLIP.
>
---
#### [replaced 027] Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16375v2](http://arxiv.org/pdf/2411.16375v2)**

> **作者:** Kaifeng Gao; Jiaxin Shi; Hanwang Zhang; Chunping Wang; Jun Xiao; Long Chen
>
> **备注:** Accepted by ICML 2025. Code is available: https://github.com/Dawn-LX/CausalCache-VDM
>
> **摘要:** With the advance of diffusion models, today's video generation has achieved impressive quality. To extend the generation length and facilitate real-world applications, a majority of video diffusion models (VDMs) generate videos in an autoregressive manner, i.e., generating subsequent clips conditioned on the last frame(s) of the previous clip. However, existing autoregressive VDMs are highly inefficient and redundant: The model must re-compute all the conditional frames that are overlapped between adjacent clips. This issue is exacerbated when the conditional frames are extended autoregressively to provide the model with long-term context. In such cases, the computational demands increase significantly (i.e., with a quadratic complexity w.r.t. the autoregression step). In this paper, we propose Ca2-VDM, an efficient autoregressive VDM with Causal generation and Cache sharing. For causal generation, it introduces unidirectional feature computation, which ensures that the cache of conditional frames can be precomputed in previous autoregression steps and reused in every subsequent step, eliminating redundant computations. For cache sharing, it shares the cache across all denoising steps to avoid the huge cache storage cost. Extensive experiments demonstrated that our Ca2-VDM achieves state-of-the-art quantitative and qualitative video generation results and significantly improves the generation speed. Code is available: https://github.com/Dawn-LX/CausalCache-VDM
>
---
#### [replaced 028] Learning Task-preferred Inference Routes for Gradient De-conflict in Multi-output DNNs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2305.19844v2](http://arxiv.org/pdf/2305.19844v2)**

> **作者:** Yi Sun; Xin Xu; Jian Li; Xiaochang Hu; Yifei Shi; Ling-Li Zeng
>
> **备注:** 15 pages
>
> **摘要:** Multi-output deep neural networks(MONs) contain multiple task branches, and these tasks usually share partial network filters that lead to the entanglement of different task inference routes. Due to the inconsistent optimization objectives, the task gradients used for training MONs will interfere with each other on the shared routes, which will decrease the overall model performance. To address this issue, we propose a novel gradient de-conflict algorithm named DR-MGF(Dynamic Routes and Meta-weighted Gradient Fusion) in this work. Different from existing de-conflict methods, DR-MGF achieves gradient de-conflict in MONs by learning task-preferred inference routes. The proposed method is motivated by our experimental findings: the shared filters are not equally important to different tasks. By designing the learnable task-specific importance variables, DR-MGF evaluates the importance of filters for different tasks. Through making the dominances of tasks over filters be proportional to the task-specific importance of filters, DR-MGF can effectively reduce the inter-task interference. The task-specific importance variables ultimately determine task-preferred inference routes at the end of training iterations. Extensive experimental results on CIFAR, ImageNet, and NYUv2 illustrate that DR-MGF outperforms the existing de-conflict methods both in prediction accuracy and convergence speed of MONs. Furthermore, DR-MGF can be extended to general MONs without modifying the overall network structures.
>
---
#### [replaced 029] Reconstructing People, Places, and Cameras
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.17806v2](http://arxiv.org/pdf/2412.17806v2)**

> **作者:** Lea Müller; Hongsuk Choi; Anthony Zhang; Brent Yi; Jitendra Malik; Angjoo Kanazawa
>
> **备注:** Project website: muelea.github.io/hsfm
>
> **摘要:** We present "Humans and Structure from Motion" (HSfM), a method for jointly reconstructing multiple human meshes, scene point clouds, and camera parameters in a metric world coordinate system from a sparse set of uncalibrated multi-view images featuring people. Our approach combines data-driven scene reconstruction with the traditional Structure-from-Motion (SfM) framework to achieve more accurate scene reconstruction and camera estimation, while simultaneously recovering human meshes. In contrast to existing scene reconstruction and SfM methods that lack metric scale information, our method estimates approximate metric scale by leveraging a human statistical model. Furthermore, it reconstructs multiple human meshes within the same world coordinate system alongside the scene point cloud, effectively capturing spatial relationships among individuals and their positions in the environment. We initialize the reconstruction of humans, scenes, and cameras using robust foundational models and jointly optimize these elements. This joint optimization synergistically improves the accuracy of each component. We compare our method to existing approaches on two challenging benchmarks, EgoHumans and EgoExo4D, demonstrating significant improvements in human localization accuracy within the world coordinate frame (reducing error from 3.51m to 1.04m in EgoHumans and from 2.9m to 0.56m in EgoExo4D). Notably, our results show that incorporating human data into the SfM pipeline improves camera pose estimation (e.g., increasing RRA@15 by 20.3% on EgoHumans). Additionally, qualitative results show that our approach improves overall scene reconstruction quality. Our code is available at: https://github.com/hongsukchoi/HSfM_RELEASE
>
---
#### [replaced 030] M3TR: A Generalist Model for Real-World HD Map Completion
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.10316v4](http://arxiv.org/pdf/2411.10316v4)**

> **作者:** Fabian Immel; Richard Fehler; Frank Bieder; Jan-Hendrik Pauls; Christoph Stiller
>
> **摘要:** Autonomous vehicles rely on HD maps for their operation, but offline HD maps eventually become outdated. For this reason, online HD map construction methods use live sensor data to infer map information instead. Research on real map changes shows that oftentimes entire parts of an HD map remain unchanged and can be used as a prior. We therefore introduce M3TR (Multi-Masking Map Transformer), a generalist approach for HD map completion both with and without offline HD map priors. As a necessary foundation, we address shortcomings in ground truth labels for Argoverse 2 and nuScenes and propose the first comprehensive benchmark for HD map completion. Unlike existing models that specialize in a single kind of map change, which is unrealistic for deployment, our Generalist model handles all kinds of changes, matching the effectiveness of Expert models. With our map masking as augmentation regime, we can even achieve a +1.4 mAP improvement without a prior. Finally, by fully utilizing prior HD map elements and optimizing query designs, M3TR outperforms existing methods by +4.3 mAP while being the first real-world deployable model for offline HD map priors. Code is available at https://github.com/immel-f/m3tr
>
---
#### [replaced 031] Efficient Partitioning Vision Transformer on Edge Devices for Distributed Inference
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.11650v2](http://arxiv.org/pdf/2410.11650v2)**

> **作者:** Xiang Liu; Yijun Song; Xia Li; Yifei Sun; Huiying Lan; Zemin Liu; Linshan Jiang; Jialin Li
>
> **备注:** 11 pages, 7 figures, 4 tables
>
> **摘要:** Deep learning models are increasingly utilized on resource-constrained edge devices for real-time data analytics. Recently, Vision Transformer and their variants have shown exceptional performance in various computer vision tasks. However, their substantial computational requirements and low inference latency create significant challenges for deploying such models on resource-constrained edge devices. To address this issue, we propose a novel framework, ED-ViT, which is designed to efficiently split and execute complex Vision Transformers across multiple edge devices. Our approach involves partitioning Vision Transformer models into several sub-models, while each dedicated to handling a specific subset of data classes. To further reduce computational overhead and inference latency, we introduce a class-wise pruning technique that decreases the size of each sub-model. Through extensive experiments conducted on five datasets using three model architectures and actual implementation on edge devices, we demonstrate that our method significantly cuts down inference latency on edge devices and achieves a reduction in model size by up to 28.9 times and 34.1 times, respectively, while maintaining test accuracy comparable to the original Vision Transformer. Additionally, we compare ED-ViT with two state-of-the-art methods that deploy CNN and SNN models on edge devices, evaluating metrics such as accuracy, inference time, and overall model size. Our comprehensive evaluation underscores the effectiveness of the proposed ED-ViT framework.
>
---
#### [replaced 032] TongUI: Building Generalized GUI Agents by Learning from Multimodal Web Tutorials
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12679v2](http://arxiv.org/pdf/2504.12679v2)**

> **作者:** Bofei Zhang; Zirui Shang; Zhi Gao; Wang Zhang; Rui Xie; Xiaojian Ma; Tao Yuan; Xinxiao Wu; Song-Chun Zhu; Qing Li
>
> **摘要:** Building Graphical User Interface (GUI) agents is a promising research direction, which simulates human interaction with computers or mobile phones to perform diverse GUI tasks. However, a major challenge in developing generalized GUI agents is the lack of sufficient trajectory data across various operating systems and applications, mainly due to the high cost of manual annotations. In this paper, we propose the TongUI framework that builds generalized GUI agents by learning from rich multimodal web tutorials. Concretely, we crawl and process online GUI tutorials (such as videos and articles) into GUI agent trajectory data, through which we produce the GUI-Net dataset containing 143K trajectory data across five operating systems and more than 200 applications. We develop the TongUI agent by fine-tuning Qwen2.5-VL-3B/7B models on GUI-Net, which show remarkable performance improvements on commonly used grounding and navigation benchmarks, outperforming baseline agents about 10\% on multiple benchmarks, showing the effectiveness of the GUI-Net dataset and underscoring the significance of our TongUI framework. We will fully open-source the code, the GUI-Net dataset, and the trained models soon.
>
---
#### [replaced 033] DisCoPatch: Batch Statistics Are All You Need For OOD Detection, But Only If You Can Trust Them
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.08005v2](http://arxiv.org/pdf/2501.08005v2)**

> **作者:** Francisco Caetano; Christiaan Viviers; Luis A. Zavala-Mondragón; Peter H. N. de With; Fons van der Sommen
>
> **摘要:** Out-of-distribution (OOD) detection holds significant importance across many applications. While semantic and domain-shift OOD problems are well-studied, this work focuses on covariate shifts - subtle variations in the data distribution that can degrade machine learning performance. We hypothesize that detecting these subtle shifts can improve our understanding of in-distribution boundaries, ultimately improving OOD detection. In adversarial discriminators trained with Batch Normalization (BN), real and adversarial samples form distinct domains with unique batch statistics - a property we exploit for OOD detection. We introduce DisCoPatch, an unsupervised Adversarial Variational Autoencoder (VAE) framework that harnesses this mechanism. During inference, batches consist of patches from the same image, ensuring a consistent data distribution that allows the model to rely on batch statistics. DisCoPatch uses the VAE's suboptimal outputs (generated and reconstructed) as negative samples to train the discriminator, thereby improving its ability to delineate the boundary between in-distribution samples and covariate shifts. By tightening this boundary, DisCoPatch achieves state-of-the-art results in public OOD detection benchmarks. The proposed model not only excels in detecting covariate shifts, achieving 95.5% AUROC on ImageNet-1K(-C) but also outperforms all prior methods on public Near-OOD (95.0%) benchmarks. With a compact model size of 25MB, it achieves high OOD detection performance at notably lower latency than existing methods, making it an efficient and practical solution for real-world OOD detection applications. The code will be made publicly available
>
---
#### [replaced 034] How far can we go with ImageNet for Text-to-Image generation?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.21318v2](http://arxiv.org/pdf/2502.21318v2)**

> **作者:** L. Degeorge; A. Ghosh; N. Dufour; D. Picard; V. Kalogeiton
>
> **摘要:** Recent text-to-image generation models have achieved remarkable results by training on billion-scale datasets, following a `bigger is better' paradigm that prioritizes data quantity over availability (closed vs open source) and reproducibility (data decay vs established collections). We challenge this established paradigm by demonstrating that one can match or outperform models trained on massive web-scraped collections, using only ImageNet enhanced with well-designed text and image augmentations. With this much simpler setup, we achieve a +1% overall score over SD-XL on GenEval and +0.5% on DPGBench while using just 1/10th the parameters and 1/1000th the training images. This opens the way for more reproducible research as ImageNet is a widely available dataset and our standardized training setup does not require massive compute resources.
>
---
#### [replaced 035] A re-calibration method for object detection with multi-modal alignment bias in autonomous driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.16848v2](http://arxiv.org/pdf/2405.16848v2)**

> **作者:** Zhihang Song; Dingyi Yao; Ruibo MIng; Lihui Peng; Jianming Hu; Danya Yao; Yi Zhang
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Multi-modal object detection in autonomous driving has achieved great breakthroughs due to the usage of fusing complementary information from different sensors. The calibration in fusion between sensors such as LiDAR and camera is always supposed to be precise in previous work. However, in reality, calibration matrices are fixed when the vehicles leave the factory, but vibration, bumps, and data lags may cause calibration bias. As the research on the calibration influence on fusion detection performance is relatively few, flexible calibration dependency multi-sensor detection method has always been attractive. In this paper, we conducted experiments on SOTA detection method EPNet++ and proved slight bias on calibration can reduce the performance seriously. We also proposed a re-calibration model based on semantic segmentation which can be combined with a detection algorithm to improve the performance and robustness of multi-modal calibration bias.
>
---
#### [replaced 036] Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04788v2](http://arxiv.org/pdf/2505.04788v2)**

> **作者:** Bangyan Liao; Zhenjun Zhao; Haoang Li; Yi Zhou; Yingping Zeng; Hao Li; Peidong Liu
>
> **备注:** Accepted to CVPR 2025 as Award Candidate & Oral Presentation. The first two authors contributed equally to this work. Code: https://github.com/WU-CVGL/GlobustVP
>
> **摘要:** Determining the vanishing points (VPs) in a Manhattan world, as a fundamental task in many 3D vision applications, consists of jointly inferring the line-VP association and locating each VP. Existing methods are, however, either sub-optimal solvers or pursuing global optimality at a significant cost of computing time. In contrast to prior works, we introduce convex relaxation techniques to solve this task for the first time. Specifically, we employ a "soft" association scheme, realized via a truncated multi-selection error, that allows for joint estimation of VPs' locations and line-VP associations. This approach leads to a primal problem that can be reformulated into a quadratically constrained quadratic programming (QCQP) problem, which is then relaxed into a convex semidefinite programming (SDP) problem. To solve this SDP problem efficiently, we present a globally optimal outlier-robust iterative solver (called GlobustVP), which independently searches for one VP and its associated lines in each iteration, treating other lines as outliers. After each independent update of all VPs, the mutual orthogonality between the three VPs in a Manhattan world is reinforced via local refinement. Extensive experiments on both synthetic and real-world data demonstrate that GlobustVP achieves a favorable balance between efficiency, robustness, and global optimality compared to previous works. The code is publicly available at https://github.com/WU-CVGL/GlobustVP.
>
---
#### [replaced 037] Faster Video Diffusion with Trainable Sparse Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13389v2](http://arxiv.org/pdf/2505.13389v2)**

> **作者:** Peiyuan Zhang; Haofeng Huang; Yongqi Chen; Will Lin; Zhengzhong Liu; Ion Stoica; Eric Xing; Hao Zhang
>
> **摘要:** Scaling video diffusion transformers (DiTs) is limited by their quadratic 3D attention, even though most of the attention mass concentrates on a small subset of positions. We turn this observation into VSA, a trainable, hardware-efficient sparse attention that replaces full attention at \emph{both} training and inference. In VSA, a lightweight coarse stage pools tokens into tiles and identifies high-weight \emph{critical tokens}; a fine stage computes token-level attention only inside those tiles subjecting to block computing layout to ensure hard efficiency. This leads to a single differentiable kernel that trains end-to-end, requires no post-hoc profiling, and sustains 85\% of FlashAttention3 MFU. We perform a large sweep of ablation studies and scaling-law experiments by pretraining DiTs from 60M to 1.4B parameters. VSA reaches a Pareto point that cuts training FLOPS by 2.53$\times$ with no drop in diffusion loss. Retrofitting the open-source Wan-2.1 model speeds up attention time by 6$\times$ and lowers end-to-end generation time from 31s to 18s with comparable quality. These results establish trainable sparse attention as a practical alternative to full attention and a key enabler for further scaling of video diffusion models.
>
---
#### [replaced 038] VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14096v2](http://arxiv.org/pdf/2504.14096v2)**

> **作者:** Yogesh Kulkarni; Pooyan Fazli
>
> **摘要:** Video-language models (Video-LLMs) excel at understanding video content but struggle with spatial relationships, temporal ordering, and cross-frame continuity. To address these limitations, we introduce VideoPASTA (Preference Alignment with Spatio-Temporal-Cross Frame Adversaries), a framework that enhances Video-LLMs through targeted preference optimization. VideoPASTA trains models to distinguish accurate video representations from carefully crafted adversarial examples that deliberately violate spatial, temporal, or cross-frame relationships. With only 7,020 preference pairs and Direct Preference Optimization, VideoPASTA enables models to learn robust representations that capture fine-grained spatial details and long-range temporal dynamics. Experiments demonstrate that VideoPASTA is model agnostic and significantly improves performance, for example, achieving gains of up to 3.8% on LongVideoBench, 4.1% on VideoMME, and 4.0% on MVBench, when applied to various state-of-the-art Video-LLMs. These results demonstrate that targeted alignment, rather than massive pretraining or architectural modifications, effectively addresses core video-language challenges. Notably, VideoPASTA achieves these improvements without any human annotation or captioning, relying solely on 32-frame sampling. This efficiency makes our approach a scalable plug-and-play solution that seamlessly integrates with existing models while preserving their original capabilities.
>
---
#### [replaced 039] Transductive One-Shot Learning Meet Subspace Decomposition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00348v2](http://arxiv.org/pdf/2504.00348v2)**

> **作者:** Kyle Stein; Andrew A. Mahyari; Guillermo Francia III; Eman El-Sheikh
>
> **摘要:** One-shot learning focuses on adapting pretrained models to recognize newly introduced and unseen classes based on a single labeled image. While variations of few-shot and zero-shot learning exist, one-shot learning remains a challenging yet crucial problem due to its ability to generalize knowledge to unseen classes from just one human-annotated image. In this paper, we introduce a transductive one-shot learning approach that employs subspace decomposition to utilize the information from labeled images in the support set and unlabeled images in the query set. These images are decomposed into a linear combination of latent variables representing primitives captured by smaller subspaces. By representing images in the query set as linear combinations of these latent primitives, we can propagate the label from a single image in the support set to query images that share similar combinations of primitives. Through a comprehensive quantitative analysis across various neural network feature extractors and datasets, we demonstrate that our approach can effectively generalize to novel classes from just one labeled image.
>
---
#### [replaced 040] CAV-MAE Sync: Improving Contrastive Audio-Visual Mask Autoencoders via Fine-Grained Alignment
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.01237v2](http://arxiv.org/pdf/2505.01237v2)**

> **作者:** Edson Araujo; Andrew Rouditchenko; Yuan Gong; Saurabhchand Bhati; Samuel Thomas; Brian Kingsbury; Leonid Karlinsky; Rogerio Feris; James R. Glass; Hilde Kuehne
>
> **备注:** To be published at CVPR 2025, code available at https://github.com/edsonroteia/cav-mae-sync
>
> **摘要:** Recent advances in audio-visual learning have shown promising results in learning representations across modalities. However, most approaches rely on global audio representations that fail to capture fine-grained temporal correspondences with visual frames. Additionally, existing methods often struggle with conflicting optimization objectives when trying to jointly learn reconstruction and cross-modal alignment. In this work, we propose CAV-MAE Sync as a simple yet effective extension of the original CAV-MAE framework for self-supervised audio-visual learning. We address three key challenges: First, we tackle the granularity mismatch between modalities by treating audio as a temporal sequence aligned with video frames, rather than using global representations. Second, we resolve conflicting optimization goals by separating contrastive and reconstruction objectives through dedicated global tokens. Third, we improve spatial localization by introducing learnable register tokens that reduce semantic load on patch tokens. We evaluate the proposed approach on AudioSet, VGG Sound, and the ADE20K Sound dataset on zero-shot retrieval, classification and localization tasks demonstrating state-of-the-art performance and outperforming more complex architectures.
>
---
#### [replaced 041] SeMv-3D: Towards Concurrency of Semantic and Multi-view Consistency in General Text-to-3D Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.07658v2](http://arxiv.org/pdf/2410.07658v2)**

> **作者:** Xiao Cai; Pengpeng Zeng; Lianli Gao; Sitong Su; Heng Tao Shen; Jingkuan Song
>
> **摘要:** General Text-to-3D (GT23D) generation is crucial for creating diverse 3D content across objects and scenes, yet it faces two key challenges: 1) ensuring semantic consistency between input text and generated 3D models, and 2) maintaining multi-view consistency across different perspectives within 3D. Existing approaches typically address only one of these challenges, often leading to suboptimal results in semantic fidelity and structural coherence. To overcome these limitations, we propose SeMv-3D, a novel framework that jointly enhances semantic alignment and multi-view consistency in GT23D generation. At its core, we introduce Triplane Prior Learning (TPL), which effectively learns triplane priors by capturing spatial correspondences across three orthogonal planes using a dedicated Orthogonal Attention mechanism, thereby ensuring geometric consistency across viewpoints. Additionally, we present Prior-based Semantic Aligning in Triplanes (SAT), which enables consistent any-view synthesis by leveraging attention-based feature alignment to reinforce the correspondence between textual semantics and triplane representations. Extensive experiments demonstrate that our method sets a new state-of-the-art in multi-view consistency, while maintaining competitive performance in semantic consistency compared to methods focused solely on semantic alignment. These results emphasize the remarkable ability of our approach to effectively balance and excel in both dimensions, establishing a new benchmark in the field.
>
---
#### [replaced 042] UncertainSAM: Fast and Efficient Uncertainty Quantification of the Segment Anything Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05049v3](http://arxiv.org/pdf/2505.05049v3)**

> **作者:** Timo Kaiser; Thomas Norrenbrock; Bodo Rosenhahn
>
> **备注:** Accepted to ICML'25
>
> **摘要:** The introduction of the Segment Anything Model (SAM) has paved the way for numerous semantic segmentation applications. For several tasks, quantifying the uncertainty of SAM is of particular interest. However, the ambiguous nature of the class-agnostic foundation model SAM challenges current uncertainty quantification (UQ) approaches. This paper presents a theoretically motivated uncertainty quantification model based on a Bayesian entropy formulation jointly respecting aleatoric, epistemic, and the newly introduced task uncertainty. We use this formulation to train USAM, a lightweight post-hoc UQ method. Our model traces the root of uncertainty back to under-parameterised models, insufficient prompts or image ambiguities. Our proposed deterministic USAM demonstrates superior predictive capabilities on the SA-V, MOSE, ADE20k, DAVIS, and COCO datasets, offering a computationally cheap and easy-to-use UQ alternative that can support user-prompting, enhance semi-supervised pipelines, or balance the tradeoff between accuracy and cost efficiency.
>
---
#### [replaced 043] MiniDrive: More Efficient Vision-Language Models with Multi-Level 2D Features as Text Tokens for Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.07267v5](http://arxiv.org/pdf/2409.07267v5)**

> **作者:** Enming Zhang; Xingyuan Dai; Min Huang; Yisheng Lv; Qinghai Miao
>
> **摘要:** Vision-language models (VLMs) serve as general-purpose end-to-end models in autonomous driving, performing subtasks such as prediction, planning, and perception through question-and-answer interactions. However, most existing methods rely on computationally expensive visual encoders and large language models (LLMs), making them difficult to deploy in real-world scenarios and real-time applications. Meanwhile, most existing VLMs lack the ability to process multiple images, making it difficult to adapt to multi-camera perception in autonomous driving. To address these issues, we propose a novel framework called MiniDrive, which incorporates our proposed Feature Engineering Mixture of Experts (FE-MoE) module and Dynamic Instruction Adapter (DI-Adapter). The FE-MoE effectively maps 2D features into visual token embeddings before being input into the language model. The DI-Adapter enables the visual token embeddings to dynamically change with the instruction text embeddings, resolving the issue of static visual token embeddings for the same image in previous approaches. Compared to previous works, MiniDrive achieves state-of-the-art performance in terms of parameter size, floating point operations, and response efficiency, with the smallest version containing only 83M parameters.
>
---
#### [replaced 044] Diversity-Driven View Subset Selection for Indoor Novel View Synthesis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.07098v2](http://arxiv.org/pdf/2409.07098v2)**

> **作者:** Zehao Wang; Han Zhou; Matthew B. Blaschko; Tinne Tuytelaars; Minye Wu
>
> **备注:** 12 pages, TMLR 2025
>
> **摘要:** Novel view synthesis of indoor scenes can be achieved by capturing a monocular video sequence of the environment. However, redundant information caused by artificial movements in the input video data reduces the efficiency of scene modeling. To address this, we formulate the problem as a combinatorial optimization task for view subset selection. In this work, we propose a novel subset selection framework that integrates a comprehensive diversity-based measurement with well-designed utility functions. We provide a theoretical analysis of these utility functions and validate their effectiveness through extensive experiments. Furthermore, we introduce IndoorTraj, a novel dataset designed for indoor novel view synthesis, featuring complex and extended trajectories that simulate intricate human behaviors. Experiments on IndoorTraj show that our framework consistently outperforms baseline strategies while using only 5-20% of the data, highlighting its remarkable efficiency and effectiveness. The code is available at: https://github.com/zehao-wang/IndoorTraj
>
---
#### [replaced 045] MambaFlow: A Mamba-Centric Architecture for End-to-End Optical Flow Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07046v2](http://arxiv.org/pdf/2503.07046v2)**

> **作者:** Juntian Du; Yuan Sun; Zhihu Zhou; Pinyi Chen; Runzhe Zhang; Keji Mao
>
> **摘要:** Recently, the Mamba architecture has demonstrated significant successes in various computer vision tasks, such as classification and segmentation. However, its application to optical flow estimation remains unexplored. In this paper, we introduce MambaFlow, a novel framework designed to leverage the high accuracy and efficiency of the Mamba architecture for capturing locally correlated features while preserving global information in end-to-end optical flow estimation. To our knowledge, MambaFlow is the first architecture centered around the Mamba design tailored specifically for optical flow estimation. It comprises two key components: (1) PolyMamba, which optimizes feature representation; and (2) PulseMamba, which facilitates efficient flow information dissemination. Our extensive experiments demonstrate that MambaFlow achieves remarkable results. On the Sintel benchmark, MambaFlow records an endpoint error (EPE) of 1.43 and an inference speed of 0.113 seconds, surpassing the state-of-the-art methods including GMFlow (with 18.9% lower EPE and 18.1% faster inference), SeparableFlow (5% lower EPE and 50.5% faster), CRAFT (1.11% lower EPE and 76.5% faster), and DIP (0.7% lower EPE and 77.2% faster)-demonstrating stronger potential for real-world deployment on resource-constrained devices. The source code will be made publicly available upon acceptance of the paper.
>
---
#### [replaced 046] EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.07293v2](http://arxiv.org/pdf/2412.07293v2)**

> **作者:** Toshiya Yura; Ashkan Mirzaei; Igor Gilitschenski
>
> **摘要:** We introduce a method for using event camera data in novel view synthesis via Gaussian Splatting. Event cameras offer exceptional temporal resolution and a high dynamic range. Leveraging these capabilities allows us to effectively address the novel view synthesis challenge in the presence of fast camera motion. For initialization of the optimization process, our approach uses prior knowledge encoded in an event-to-video model. We also use spline interpolation for obtaining high quality poses along the event camera trajectory. This enhances the reconstruction quality from fast-moving cameras while overcoming the computational limitations traditionally associated with event-based Neural Radiance Field (NeRF) methods. Our experimental evaluation demonstrates that our results achieve higher visual fidelity and better performance than existing event-based NeRF approaches while being an order of magnitude faster to render.
>
---
#### [replaced 047] Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach
- **分类: eess.AS; cs.CV; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14336v2](http://arxiv.org/pdf/2505.14336v2)**

> **作者:** Umberto Cappellazzo; Minsu Kim; Stavros Petridis; Daniele Falavigna; Alessio Brutti
>
> **备注:** Interspeech 2025
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) enhances robustness in noisy environments by integrating visual cues. While recent advances integrate Large Language Models (LLMs) into AVSR, their high computational cost hinders deployment in resource-constrained settings. To address this, we propose Llama-SMoP, an efficient Multimodal LLM that employs a Sparse Mixture of Projectors (SMoP) module to scale model capacity without increasing inference costs. By incorporating sparsely-gated mixture-of-experts (MoE) projectors, Llama-SMoP enables the use of smaller LLMs while maintaining strong performance. We explore three SMoP configurations and show that Llama-SMoP DEDR (Disjoint-Experts, Disjoint-Routers), which uses modality-specific routers and experts, achieves superior performance on ASR, VSR, and AVSR tasks. Ablation studies confirm its effectiveness in expert activation, scalability, and noise robustness.
>
---
#### [replaced 048] Retrospective Learning from Interactions
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.13852v2](http://arxiv.org/pdf/2410.13852v2)**

> **作者:** Zizhao Chen; Mustafa Omer Gul; Yiwei Chen; Gloria Geng; Anne Wu; Yoav Artzi
>
> **摘要:** Multi-turn interactions between large language models (LLMs) and users naturally include implicit feedback signals. If an LLM responds in an unexpected way to an instruction, the user is likely to signal it by rephrasing the request, expressing frustration, or pivoting to an alternative task. Such signals are task-independent and occupy a relatively constrained subspace of language, allowing the LLM to identify them even if it fails on the actual task. We introduce ReSpect, a method to learn from such signals in past interactions via retrospection without additional annotations. We deploy ReSpect in a new multimodal interaction scenario, where humans instruct a multimodal LLM to solve an abstract reasoning task with a combinatorial solution space. Through thousands of interactions with humans, we show how ReSpect gradually improves task completion rate from 31% to 82%, all without any external annotation.
>
---
#### [replaced 049] Accelerating Diffusion-based Super-Resolution with Dynamic Time-Spatial Sampling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12048v2](http://arxiv.org/pdf/2505.12048v2)**

> **作者:** Rui Qin; Qijie Wang; Ming Sun; Haowei Zhu; Chao Zhou; Bin Wang
>
> **摘要:** Diffusion models have gained attention for their success in modeling complex distributions, achieving impressive perceptual quality in SR tasks. However, existing diffusion-based SR methods often suffer from high computational costs, requiring numerous iterative steps for training and inference. Existing acceleration techniques, such as distillation and solver optimization, are generally task-agnostic and do not fully leverage the specific characteristics of low-level tasks like super-resolution (SR). In this study, we analyze the frequency- and spatial-domain properties of diffusion-based SR methods, revealing key insights into the temporal and spatial dependencies of high-frequency signal recovery. Specifically, high-frequency details benefit from concentrated optimization during early and late diffusion iterations, while spatially textured regions demand adaptive denoising strategies. Building on these observations, we propose the Time-Spatial-aware Sampling strategy (TSS) for the acceleration of Diffusion SR without any extra training cost. TSS combines Time Dynamic Sampling (TDS), which allocates more iterations to refining textures, and Spatial Dynamic Sampling (SDS), which dynamically adjusts strategies based on image content. Extensive evaluations across multiple benchmarks demonstrate that TSS achieves state-of-the-art (SOTA) performance with significantly fewer iterations, improving MUSIQ scores by 0.2 - 3.0 and outperforming the current acceleration methods with only half the number of steps.
>
---
#### [replaced 050] Enhanced Textual Feature Extraction for Visual Question Answering: A Simple Convolutional Approach
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.00479v3](http://arxiv.org/pdf/2405.00479v3)**

> **作者:** Zhilin Zhang; Fangyu Wu
>
> **备注:** To be published in 2025 6th International Conference on Computer Vision and Computational Intelligence (CVCI 2025)
>
> **摘要:** Visual Question Answering (VQA) has emerged as a highly engaging field in recent years, with increasing research focused on enhancing VQA accuracy through advanced models such as Transformers. Despite this growing interest, limited work has examined the comparative effectiveness of textual encoders in VQA, particularly considering model complexity and computational efficiency. In this work, we conduct a comprehensive comparison between complex textual models that leverage long-range dependencies and simpler models focusing on local textual features within a well-established VQA framework. Our findings reveal that employing complex textual encoders is not always the optimal approach for the VQA-v2 dataset. Motivated by this insight, we propose ConvGRU, a model that incorporates convolutional layers to improve text feature representation without substantially increasing model complexity. Tested on the VQA-v2 dataset, ConvGRU demonstrates a modest yet consistent improvement over baselines for question types such as Number and Count, which highlights the potential of lightweight architectures for VQA tasks, especially when computational resources are limited.
>
---
#### [replaced 051] VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12081v2](http://arxiv.org/pdf/2505.12081v2)**

> **作者:** Yuqi Liu; Tianyuan Qu; Zhisheng Zhong; Bohao Peng; Shu Liu; Bei Yu; Jiaya Jia
>
> **摘要:** Large vision-language models exhibit inherent capabilities to handle diverse visual perception tasks. In this paper, we introduce VisionReasoner, a unified framework capable of reasoning and solving multiple visual perception tasks within a shared model. Specifically, by designing novel multi-object cognitive learning strategies and systematic task reformulation, VisionReasoner enhances its reasoning capabilities to analyze visual inputs, and addresses diverse perception tasks in a unified framework. The model generates a structured reasoning process before delivering the desired outputs responding to user queries. To rigorously assess unified visual perception capabilities, we evaluate VisionReasoner on ten diverse tasks spanning three critical domains: detection, segmentation, and counting. Experimental results show that VisionReasoner achieves superior performance as a unified model, outperforming Qwen2.5VL by relative margins of 29.1% on COCO (detection), 22.1% on ReasonSeg (segmentation), and 15.3% on CountBench (counting).
>
---
#### [replaced 052] P3P: Pseudo-3D Pre-training for Scaling 3D Voxel-based Masked Autoencoders
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.10007v3](http://arxiv.org/pdf/2408.10007v3)**

> **作者:** Xuechao Chen; Ying Chen; Jialin Li; Qiang Nie; Hanqiu Deng; Yong Liu; Qixing Huang; Yang Li
>
> **备注:** Under review. Pre-print
>
> **摘要:** 3D pre-training is crucial to 3D perception tasks. Nevertheless, limited by the difficulties in collecting clean and complete 3D data, 3D pre-training has persistently faced data scaling challenges. In this work, we introduce a novel self-supervised pre-training framework that incorporates millions of images into 3D pre-training corpora by leveraging a large depth estimation model. New pre-training corpora encounter new challenges in representation ability and embedding efficiency of models. Previous pre-training methods rely on farthest point sampling and k-nearest neighbors to embed a fixed number of 3D tokens. However, these approaches prove inadequate when it comes to embedding millions of samples that feature a diverse range of point numbers, spanning from 1,000 to 100,000. In contrast, we propose a tokenizer with linear-time complexity, which enables the efficient embedding of a flexible number of tokens. Accordingly, a new 3D reconstruction target is proposed to cooperate with our 3D tokenizer. Our method achieves state-of-the-art performance in 3D classification, few-shot learning, and 3D segmentation. Code is available at https://github.com/XuechaoChen/P3P-MAE.
>
---
#### [replaced 053] MLEP: Multi-granularity Local Entropy Patterns for Universal AI-generated Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13726v2](http://arxiv.org/pdf/2504.13726v2)**

> **作者:** Lin Yuan; Xiaowan Li; Yan Zhang; Jiawei Zhang; Hongbo Li; Xinbo Gao
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Advancements in image generation technologies have raised significant concerns about their potential misuse, such as producing misinformation and deepfakes. Therefore, there is an urgent need for effective methods to detect AI-generated images (AIGI). Despite progress in AIGI detection, achieving reliable performance across diverse generation models and scenes remains challenging due to the lack of source-invariant features and limited generalization capabilities in existing methods. In this work, we explore the potential of using image entropy as a cue for AIGI detection and propose Multi-granularity Local Entropy Patterns (MLEP), a set of entropy feature maps computed across shuffled small patches over multiple image scaled. MLEP comprehensively captures pixel relationships across dimensions and scales while significantly disrupting image semantics, reducing potential content bias. Leveraging MLEP, a robust CNN-based classifier for AIGI detection can be trained. Extensive experiments conducted in an open-world scenario, evaluating images synthesized by 32 distinct generative models, demonstrate significant improvements over state-of-the-art methods in both accuracy and generalization.
>
---
#### [replaced 054] PlaySlot: Learning Inverse Latent Dynamics for Controllable Object-Centric Video Prediction and Planning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.07600v2](http://arxiv.org/pdf/2502.07600v2)**

> **作者:** Angel Villar-Corrales; Sven Behnke
>
> **备注:** ICML 2025
>
> **摘要:** Predicting future scene representations is a crucial task for enabling robots to understand and interact with the environment. However, most existing methods rely on videos and simulations with precise action annotations, limiting their ability to leverage the large amount of available unlabeled video data. To address this challenge, we propose PlaySlot, an object-centric video prediction model that infers object representations and latent actions from unlabeled video sequences. It then uses these representations to forecast future object states and video frames. PlaySlot allows the generation of multiple possible futures conditioned on latent actions, which can be inferred from video dynamics, provided by a user, or generated by a learned action policy, thus enabling versatile and interpretable world modeling. Our results show that PlaySlot outperforms both stochastic and object-centric baselines for video prediction across different environments. Furthermore, we show that our inferred latent actions can be used to learn robot behaviors sample-efficiently from unlabeled video demonstrations. Videos and code are available on https://play-slot.github.io/PlaySlot/.
>
---
#### [replaced 055] MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark
- **分类: cs.IR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11651v2](http://arxiv.org/pdf/2505.11651v2)**

> **作者:** Radek Osmulski; Gabriel de Souza P. Moreira; Ronay Ak; Mengyao Xu; Benedikt Schifferer; Even Oldridge
>
> **摘要:** Document retrieval is an important task for search and Retrieval-Augmented Generation (RAG) applications. Large Language Models (LLMs) have contributed to improving the accuracy of text-based document retrieval. However, documents with complex layout and visual elements like tables, charts and infographics are not perfectly represented in textual format. Recently, image-based document retrieval pipelines have become popular, which use visual large language models (VLMs) to retrieve relevant page images given a query. Current evaluation benchmarks on visual document retrieval are limited, as they primarily focus only English language, rely on synthetically generated questions and offer a small corpus size. Therefore, we introduce MIRACL-VISION, a multilingual visual document retrieval evaluation benchmark. MIRACL-VISION covers 18 languages, and is an extension of the MIRACL dataset, a popular benchmark to evaluate text-based multilingual retrieval pipelines. MIRACL was built using a human-intensive annotation process to generate high-quality questions. In order to reduce MIRACL-VISION corpus size to make evaluation more compute friendly while keeping the datasets challenging, we have designed a method for eliminating the "easy" negatives from the corpus. We conducted extensive experiments comparing MIRACL-VISION with other benchmarks, using popular public text and image models. We observe a gap in state-of-the-art VLM-based embedding models on multilingual capabilities, with up to 59.7% lower retrieval accuracy than a text-based retrieval models. Even for the English language, the visual models retrieval accuracy is 12.1% lower compared to text-based models. MIRACL-VISION is a challenging, representative, multilingual evaluation benchmark for visual retrieval pipelines and will help the community build robust models for document retrieval.
>
---
#### [replaced 056] The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01081v2](http://arxiv.org/pdf/2502.01081v2)**

> **作者:** Vernon Y. H. Toh; Yew Ken Chia; Deepanway Ghosal; Soujanya Poria
>
> **摘要:** The releases of OpenAI's o-[n] series, such as o1, o3, and o4-mini, mark a significant paradigm shift in Large Language Models towards advanced reasoning capabilities. Notably, models like o3 have demonstrated strong performance on benchmarks like the Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI). However, this benchmark is limited to symbolic patterns, whereas humans often perceive and reason about multimodal scenarios involving both vision and language data. Thus, there is an urgent need to investigate advanced reasoning capabilities in multimodal tasks. To this end, we track the evolution of the GPT-[n] and o-[n] series models (including o1, o3, and o4-mini) on challenging multimodal puzzles from PuzzleVQA and AlgoPuzzleVQA, which demand fine-grained visual perception. Our results reveal that o-[n] series, particularly later iterations like o3 and o4-mini, significantly outperform the GPT-[n] series and show strong scalability in multimodal reasoning. Nonetheless, despite these substantial advancements and the superior capabilities demonstrated by the o-[n] series, our findings highlight that even these leading models face persistent challenges. Difficulties are particularly evident in tasks requiring precise visual perception, robust compositional reasoning across multiple visual attributes, and solving complex algorithmic or highly combinatorial puzzles, indicating critical areas for future AGI development. We plan to continuously track new models in the series and update our results in this paper accordingly. All resources used in this evaluation are openly available at https://github.com/declare-lab/LLM-PuzzleTest.
>
---
#### [replaced 057] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction for Multimodal Retrieval
- **分类: cs.CV; cs.AI; cs.IR; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.08334v3](http://arxiv.org/pdf/2411.08334v3)**

> **作者:** Yeong-Joon Ju; Ho-Joong Kim; Seong-Whan Lee
>
> **备注:** Accepted to ACL 2025 (Findings)
>
> **摘要:** Recent multimodal retrieval methods have endowed text-based retrievers with multimodal capabilities by utilizing pre-training strategies for visual-text alignment. They often directly fuse the two modalities for cross-reference during the alignment to understand multimodal queries. However, existing methods often overlook crucial visual information due to a text-dominant issue, which overly depends on text-driven signals. In this paper, we introduce MIRe, a retrieval framework that achieves modality interaction without fusing textual features during the alignment. Our method allows the textual query to attend to visual embeddings while not feeding text-driven signals back into the visual representations. Additionally, we construct a pre-training dataset for multimodal query retrieval by transforming concise question-answer pairs into extended passages. Our experiments demonstrate that our pre-training strategy significantly enhances the understanding of multimodal queries, resulting in strong performance across four multimodal retrieval benchmarks under zero-shot settings. Moreover, our ablation studies and analyses explicitly verify the effectiveness of our framework in mitigating the text-dominant issue. Our code is publicly available: https://github.com/yeongjoonJu/MIRe
>
---
#### [replaced 058] DD-Ranking: Rethinking the Evaluation of Dataset Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13300v2](http://arxiv.org/pdf/2505.13300v2)**

> **作者:** Zekai Li; Xinhao Zhong; Samir Khaki; Zhiyuan Liang; Yuhao Zhou; Mingjia Shi; Ziqiao Wang; Xuanlei Zhao; Wangbo Zhao; Ziheng Qin; Mengxuan Wu; Pengfei Zhou; Haonan Wang; David Junhao Zhang; Jia-Wei Liu; Shaobo Wang; Dai Liu; Linfeng Zhang; Guang Li; Kun Wang; Zheng Zhu; Zhiheng Ma; Joey Tianyi Zhou; Jiancheng Lv; Yaochu Jin; Peihao Wang; Kaipeng Zhang; Lingjuan Lyu; Yiran Huang; Zeynep Akata; Zhiwei Deng; Xindi Wu; George Cazenavette; Yuzhang Shang; Justin Cui; Jindong Gu; Qian Zheng; Hao Ye; Shuo Wang; Xiaobo Wang; Yan Yan; Angela Yao; Mike Zheng Shou; Tianlong Chen; Hakan Bilen; Baharan Mirzasoleiman; Manolis Kellis; Konstantinos N. Plataniotis; Zhangyang Wang; Bo Zhao; Yang You; Kai Wang
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** In recent years, dataset distillation has provided a reliable solution for data compression, where models trained on the resulting smaller synthetic datasets achieve performance comparable to those trained on the original datasets. To further improve the performance of synthetic datasets, various training pipelines and optimization objectives have been proposed, greatly advancing the field of dataset distillation. Recent decoupled dataset distillation methods introduce soft labels and stronger data augmentation during the post-evaluation phase and scale dataset distillation up to larger datasets (e.g., ImageNet-1K). However, this raises a question: Is accuracy still a reliable metric to fairly evaluate dataset distillation methods? Our empirical findings suggest that the performance improvements of these methods often stem from additional techniques rather than the inherent quality of the images themselves, with even randomly sampled images achieving superior results. Such misaligned evaluation settings severely hinder the development of DD. Therefore, we propose DD-Ranking, a unified evaluation framework, along with new general evaluation metrics to uncover the true performance improvements achieved by different methods. By refocusing on the actual information enhancement of distilled datasets, DD-Ranking provides a more comprehensive and fair evaluation standard for future research advancements.
>
---
#### [replaced 059] Symmetry-Robust 3D Orientation Estimation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02101v3](http://arxiv.org/pdf/2410.02101v3)**

> **作者:** Christopher Scarvelis; David Benhaim; Paul Zhang
>
> **备注:** ICML 2025
>
> **摘要:** Orientation estimation is a fundamental task in 3D shape analysis which consists of estimating a shape's orientation axes: its side-, up-, and front-axes. Using this data, one can rotate a shape into canonical orientation, where its orientation axes are aligned with the coordinate axes. Developing an orientation algorithm that reliably estimates complete orientations of general shapes remains an open problem. We introduce a two-stage orientation pipeline that achieves state of the art performance on up-axis estimation and further demonstrate its efficacy on full-orientation estimation, where one seeks all three orientation axes. Unlike previous work, we train and evaluate our method on all of Shapenet rather than a subset of classes. We motivate our engineering contributions by theory describing fundamental obstacles to orientation estimation for rotationally-symmetric shapes, and show how our method avoids these obstacles.
>
---
#### [replaced 060] A Survey of Pathology Foundation Model: Progress and Future Directions
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.04045v2](http://arxiv.org/pdf/2504.04045v2)**

> **作者:** Conghao Xiong; Hao Chen; Joseph J. Y. Sung
>
> **备注:** Accepted to IJCAI 2025 Survey Track, 10 Pages
>
> **摘要:** Computational pathology, which involves analyzing whole slide images for automated cancer diagnosis, relies on multiple instance learning, where performance depends heavily on the feature extractor and aggregator. Recent Pathology Foundation Models (PFMs), pretrained on large-scale histopathology data, have significantly enhanced both the extractor and aggregator, but they lack a systematic analysis framework. In this survey, we present a hierarchical taxonomy organizing PFMs through a top-down philosophy applicable to foundation model analysis in any domain: model scope, model pretraining, and model design. Additionally, we systematically categorize PFM evaluation tasks into slide-level, patch-level, multimodal, and biological tasks, providing comprehensive benchmarking criteria. Our analysis identifies critical challenges in both PFM development (pathology-specific methodology, end-to-end pretraining, data-model scalability) and utilization (effective adaptation, model maintenance), paving the way for future directions in this promising field. Resources referenced in this survey are available at https://github.com/BearCleverProud/AwesomeWSI.
>
---
#### [replaced 061] SpaceR: Reinforcing MLLMs in Video Spatial Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.01805v2](http://arxiv.org/pdf/2504.01805v2)**

> **作者:** Kun Ouyang; Yuanxin Liu; Haoning Wu; Yi Liu; Hao Zhou; Jie Zhou; Fandong Meng; Xu Sun
>
> **摘要:** Video spatial reasoning, which involves inferring the underlying spatial structure from observed video frames, poses a significant challenge for existing Multimodal Large Language Models (MLLMs). This limitation stems primarily from 1) the absence of high-quality datasets for this task, and 2) the lack of effective training strategies to develop spatial reasoning capabilities. Motivated by the success of Reinforcement Learning with Verifiable Reward (RLVR) in unlocking LLM reasoning abilities, this work aims to improve MLLMs in video spatial reasoning through the RLVR paradigm. To this end, we introduce the $\textbf{SpaceR}$ framework. First, we present $\textbf{SpaceR-151k}$, a dataset with 91k questions spanning diverse spatial reasoning scenarios with verifiable answers, and 60k samples for maintaining general multimodal understanding. Second, we propose $\textbf{Spatially-Guided RLVR (SG-RLVR)}$, a novel reinforcement learning approach that extends Group Relative Policy Optimization (GRPO) with a novel map imagination mechanism, which encourages the model to infer spatial layouts in the thinking process, thereby facilitating more effective spatial reasoning. Extensive experiments demonstrate that SpaceR achieves state-of-the-art performance on spatial reasoning benchmarks (e.g., VSI-Bench, STI-Bench, and SPAR-Bench), while maintaining competitive results on video understanding benchmarks (e.g., Video-MME, TempCompass, and LongVideoBench). Remarkably, SpaceR surpasses the advanced GPT-4o by 11.6\% accuracy on VSI-Bench and is on par with the leading proprietary model Gemini-2.0-Flash, highlighting the effectiveness of our SpaceR-151k dataset and SG-RLVR in reinforcing spatial reasoning ability of MLLMs. Code, model, and dataset are available at https://github.com/OuyangKun10/SpaceR.
>
---
#### [replaced 062] Aggregation Schemes for Single-Vector WSI Representation Learning in Digital Pathology
- **分类: eess.IV; cs.AI; cs.CV; cs.IR; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2501.17822v2](http://arxiv.org/pdf/2501.17822v2)**

> **作者:** Sobhan Hemati; Ghazal Alabtah; Saghir Alfasly; H. R. Tizhoosh
>
> **摘要:** A crucial step to efficiently integrate Whole Slide Images (WSIs) in computational pathology is assigning a single high-quality feature vector, i.e., one embedding, to each WSI. With the existence of many pre-trained deep neural networks and the emergence of foundation models, extracting embeddings for sub-images (i.e., tiles or patches) is straightforward. However, for WSIs, given their high resolution and gigapixel nature, inputting them into existing GPUs as a single image is not feasible. As a result, WSIs are usually split into many patches. Feeding each patch to a pre-trained model, each WSI can then be represented by a set of patches, hence, a set of embeddings. Hence, in such a setup, WSI representation learning reduces to set representation learning where for each WSI we have access to a set of patch embeddings. To obtain a single embedding from a set of patch embeddings for each WSI, multiple set-based learning schemes have been proposed in the literature. In this paper, we evaluate the WSI search performance of multiple recently developed aggregation techniques (mainly set representation learning techniques) including simple average or max pooling operations, Deep Sets, Memory networks, Focal attention, Gaussian Mixture Model (GMM) Fisher Vector, and deep sparse and binary Fisher Vector on four different primary sites including bladder, breast, kidney, and Colon from TCGA. Further, we benchmark the search performance of these methods against the median of minimum distances of patch embeddings, a non-aggregating approach used for WSI retrieval.
>
---
#### [replaced 063] DLO-Splatting: Tracking Deformable Linear Objects Using 3D Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.08644v2](http://arxiv.org/pdf/2505.08644v2)**

> **作者:** Holly Dinkel; Marcel Büsching; Alberta Longhini; Brian Coltin; Trey Smith; Danica Kragic; Mårten Björkman; Timothy Bretl
>
> **备注:** 5 pages, 2 figures, presented at the 2025 5th Workshop: Reflections on Representations and Manipulating Deformable Objects at the IEEE International Conference on Robotics and Automation. RMDO workshop (https://deformable-workshop.github.io/icra2025/). Video (https://www.youtube.com/watch?v=CG4WDWumGXA). Poster (https://hollydinkel.github.io/assets/pdf/ICRA2025RMDO_poster.pdf)
>
> **摘要:** This work presents DLO-Splatting, an algorithm for estimating the 3D shape of Deformable Linear Objects (DLOs) from multi-view RGB images and gripper state information through prediction-update filtering. The DLO-Splatting algorithm uses a position-based dynamics model with shape smoothness and rigidity dampening corrections to predict the object shape. Optimization with a 3D Gaussian Splatting-based rendering loss iteratively renders and refines the prediction to align it with the visual observations in the update step. Initial experiments demonstrate promising results in a knot tying scenario, which is challenging for existing vision-only methods.
>
---
#### [replaced 064] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.12332v2](http://arxiv.org/pdf/2505.12332v2)**

> **作者:** Qianyue Hu; Junyan Wu; Wei Lu; Xiangyang Luo
>
> **摘要:** Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning.
>
---
#### [replaced 065] Volumetrically Consistent 3D Gaussian Rasterization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03378v3](http://arxiv.org/pdf/2412.03378v3)**

> **作者:** Chinmay Talegaonkar; Yash Belhe; Ravi Ramamoorthi; Nicholas Antipa
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS) has enabled photorealistic view synthesis at high inference speeds. However, its splatting-based rendering model makes several approximations to the rendering equation, reducing physical accuracy. We show that the core approximations in splatting are unnecessary, even within a rasterizer; We instead volumetrically integrate 3D Gaussians directly to compute the transmittance across them analytically. We use this analytic transmittance to derive more physically-accurate alpha values than 3DGS, which can directly be used within their framework. The result is a method that more closely follows the volume rendering equation (similar to ray-tracing) while enjoying the speed benefits of rasterization. Our method represents opaque surfaces with higher accuracy and fewer points than 3DGS. This enables it to outperform 3DGS for view synthesis (measured in SSIM and LPIPS). Being volumetrically consistent also enables our method to work out of the box for tomography. We match the state-of-the-art 3DGS-based tomography method with fewer points. Our code is publicly available at: https://github.com/chinmay0301ucsd/Vol3DGS
>
---
#### [replaced 066] RGBX-DiffusionDet: A Framework for Multi-Modal RGB-X Object Detection Using DiffusionDet
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02586v2](http://arxiv.org/pdf/2505.02586v2)**

> **作者:** Eliraz Orfaig; Inna Stainvas; Igal Bilik
>
> **摘要:** This work introduces RGBX-DiffusionDet, an object detection framework extending the DiffusionDet model to fuse the heterogeneous 2D data (X) with RGB imagery via an adaptive multimodal encoder. To enable cross-modal interaction, we design the dynamic channel reduction within a convolutional block attention module (DCR-CBAM), which facilitates cross-talk between subnetworks by dynamically highlighting salient channel features. Furthermore, the dynamic multi-level aggregation block (DMLAB) is proposed to refine spatial feature representations through adaptive multiscale fusion. Finally, novel regularization losses that enforce channel saliency and spatial selectivity are introduced, leading to compact and discriminative feature embeddings. Extensive experiments using RGB-Depth (KITTI), a novel annotated RGB-Polarimetric dataset, and RGB-Infrared (M$^3$FD) benchmark dataset were conducted. We demonstrate consistent superiority of the proposed approach over the baseline RGB-only DiffusionDet. The modular architecture maintains the original decoding complexity, ensuring efficiency. These results establish the proposed RGBX-DiffusionDet as a flexible multimodal object detection approach, providing new insights into integrating diverse 2D sensing modalities into diffusion-based detection pipelines.
>
---
#### [replaced 067] OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.18041v5](http://arxiv.org/pdf/2502.18041v5)**

> **作者:** Yunpeng Gao; Chenhui Li; Zhongrui You; Junli Liu; Zhen Li; Pengan Chen; Qizhi Chen; Zhonghan Tang; Liansheng Wang; Penghui Yang; Yiwen Tang; Yuhang Tang; Shuai Liang; Songyi Zhu; Ziqin Xiong; Yifei Su; Xinyi Ye; Jianan Li; Yan Ding; Dong Wang; Zhigang Wang; Bin Zhao; Xuelong Li
>
> **摘要:** Vision-Language Navigation (VLN) aims to guide agents by leveraging language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising various rendering engines, a versatile toolchain, and a large-scale benchmark for aerial VLN. Firstly, we integrate diverse rendering engines and advanced techniques for environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations during flight. For benchmarking, extensive experiments and analyses are conducted, evaluating several recent VLN methods and showcasing the superiority of our OpenFly platform and agent. The toolchain, dataset, and codes will be open-sourced.
>
---
#### [replaced 068] Unlocking the Power of SAM 2 for Few-Shot Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14100v2](http://arxiv.org/pdf/2505.14100v2)**

> **作者:** Qianxiong Xu; Lanyun Zhu; Xuanyi Liu; Guosheng Lin; Cheng Long; Ziyue Li; Rui Zhao
>
> **备注:** This paper is accepted by ICML'25
>
> **摘要:** Few-Shot Segmentation (FSS) aims to learn class-agnostic segmentation on few classes to segment arbitrary classes, but at the risk of overfitting. To address this, some methods use the well-learned knowledge of foundation models (e.g., SAM) to simplify the learning process. Recently, SAM 2 has extended SAM by supporting video segmentation, whose class-agnostic matching ability is useful to FSS. A simple idea is to encode support foreground (FG) features as memory, with which query FG features are matched and fused. Unfortunately, the FG objects in different frames of SAM 2's video data are always the same identity, while those in FSS are different identities, i.e., the matching step is incompatible. Therefore, we design Pseudo Prompt Generator to encode pseudo query memory, matching with query features in a compatible way. However, the memories can never be as accurate as the real ones, i.e., they are likely to contain incomplete query FG, and some unexpected query background (BG) features, leading to wrong segmentation. Hence, we further design Iterative Memory Refinement to fuse more query FG features into the memory, and devise a Support-Calibrated Memory Attention to suppress the unexpected query BG features in memory. Extensive experiments have been conducted on PASCAL-5$^i$ and COCO-20$^i$ to validate the effectiveness of our design, e.g., the 1-shot mIoU can be 4.2% better than the best baseline.
>
---
#### [replaced 069] Enrich the content of the image Using Context-Aware Copy Paste
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.08151v2](http://arxiv.org/pdf/2407.08151v2)**

> **作者:** Qiushi Guo
>
> **摘要:** Data augmentation remains a widely utilized technique in deep learning, particularly in tasks such as image classification, semantic segmentation, and object detection. Among them, Copy-Paste is a simple yet effective method and gain great attention recently. However, existing Copy-Paste often overlook contextual relevance between source and target images, resulting in inconsistencies in generated outputs. To address this challenge, we propose a context-aware approach that integrates Bidirectional Latent Information Propagation (BLIP) for content extraction from source images. By matching extracted content information with category information, our method ensures cohesive integration of target objects using Segment Anything Model (SAM) and You Only Look Once (YOLO). This approach eliminates the need for manual annotation, offering an automated and user-friendly solution. Experimental evaluations across diverse datasets demonstrate the effectiveness of our method in enhancing data diversity and generating high-quality pseudo-images across various computer vision tasks.
>
---
#### [replaced 070] Learning Cross-Spectral Point Features with Task-Oriented Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12593v2](http://arxiv.org/pdf/2505.12593v2)**

> **作者:** Mia Thomas; Trevor Ablett; Jonathan Kelly
>
> **备注:** In Proceedings of the {IEEE} International Conference on Robotics and Automation {(ICRA'25)} Thermal Infrared in Robotics (TIRO) Workshop, Atlanta, Georgia, USA, May 19, 2025
>
> **摘要:** Unmanned aerial vehicles (UAVs) enable operations in remote and hazardous environments, yet the visible-spectrum, camera-based navigation systems often relied upon by UAVs struggle in low-visibility conditions. Thermal cameras, which capture long-wave infrared radiation, are able to function effectively in darkness and smoke, where visible-light cameras fail. This work explores learned cross-spectral (thermal-visible) point features as a means to integrate thermal imagery into established camera-based navigation systems. Existing methods typically train a feature network's detection and description outputs directly, which often focuses training on image regions where thermal and visible-spectrum images exhibit similar appearance. Aiming to more fully utilize the available data, we propose a method to train the feature network on the tasks of matching and registration. We run our feature network on thermal-visible image pairs, then feed the network response into a differentiable registration pipeline. Losses are applied to the matching and registration estimates of this pipeline. Our selected model, trained on the task of matching, achieves a registration error (corner error) below 10 pixels for more than 75% of estimates on the MultiPoint dataset. We further demonstrate that our model can also be used with a classical pipeline for matching and registration.
>
---
#### [replaced 071] BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12620v2](http://arxiv.org/pdf/2505.12620v2)**

> **作者:** Haiquan Wen; Yiwei He; Zhenglin Huang; Tianxiao Li; Zihan Yu; Xingru Huang; Lu Qi; Baoyuan Wu; Xiangtai Li; Guangliang Cheng
>
> **摘要:** Advances in AI generative models facilitate super-realistic video synthesis, amplifying misinformation risks via social media and eroding trust in digital content. Several research works have explored new deepfake detection methods on AI-generated images to alleviate these risks. However, with the fast development of video generation models, such as Sora and WanX, there is currently a lack of large-scale, high-quality AI-generated video datasets for forgery detection. In addition, existing detection approaches predominantly treat the task as binary classification, lacking explainability in model decision-making and failing to provide actionable insights or guidance for the public. To address these challenges, we propose \textbf{GenBuster-200K}, a large-scale AI-generated video dataset featuring 200K high-resolution video clips, diverse latest generative techniques, and real-world scenes. We further introduce \textbf{BusterX}, a novel AI-generated video detection and explanation framework leveraging multimodal large language model (MLLM) and reinforcement learning for authenticity determination and explainable rationale. To our knowledge, GenBuster-200K is the {\it \textbf{first}} large-scale, high-quality AI-generated video dataset that incorporates the latest generative techniques for real-world scenarios. BusterX is the {\it \textbf{first}} framework to integrate MLLM with reinforcement learning for explainable AI-generated video detection. Extensive comparisons with state-of-the-art methods and ablation studies validate the effectiveness and generalizability of BusterX. The code, models, and datasets will be released.
>
---
#### [replaced 072] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.12030v4](http://arxiv.org/pdf/2406.12030v4)**

> **作者:** Yongting Zhang; Lu Chen; Guodong Zheng; Yifeng Gao; Rui Zheng; Jinlan Fu; Zhenfei Yin; Senjie Jin; Yu Qiao; Xuanjing Huang; Feng Zhao; Tao Gui; Jing Shao
>
> **摘要:** The emergence of Vision Language Models (VLMs) has brought unprecedented advances in understanding multimodal information. The combination of textual and visual semantics in VLMs is highly complex and diverse, making the safety alignment of these models challenging. Furthermore, due to the limited study on the safety alignment of VLMs, there is a lack of large-scale, high-quality datasets. To address these limitations, we propose a Safety Preference Alignment dataset for Vision Language Models named SPA-VL. In terms of breadth, SPA-VL covers 6 harmfulness domains, 13 categories, and 53 subcategories, and contains 100,788 samples of the quadruple (question, image, chosen response, rejected response). In terms of depth, the responses are collected from 12 open-source (e.g., QwenVL) and closed-source (e.g., Gemini) VLMs to ensure diversity. The construction of preference data is fully automated, and the experimental results indicate that models trained with alignment techniques on the SPA-VL dataset exhibit substantial improvements in harmlessness and helpfulness while maintaining core capabilities. SPA-VL, as a large-scale, high-quality, and diverse dataset, represents a significant milestone in ensuring that VLMs achieve both harmlessness and helpfulness.
>
---
#### [replaced 073] Sparc3D: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14521v2](http://arxiv.org/pdf/2505.14521v2)**

> **作者:** Zhihao Li; Yufei Wang; Heliang Zheng; Yihao Luo; Bihan Wen
>
> **备注:** Homepage: https://lizhihao6.github.io/Sparc3D
>
> **摘要:** High-fidelity 3D object synthesis remains significantly more challenging than 2D image generation due to the unstructured nature of mesh data and the cubic complexity of dense volumetric grids. Existing two-stage pipelines-compressing meshes with a VAE (using either 2D or 3D supervision), followed by latent diffusion sampling-often suffer from severe detail loss caused by inefficient representations and modality mismatches introduced in VAE. We introduce Sparc3D, a unified framework that combines a sparse deformable marching cubes representation Sparcubes with a novel encoder Sparconv-VAE. Sparcubes converts raw meshes into high-resolution ($1024^3$) surfaces with arbitrary topology by scattering signed distance and deformation fields onto a sparse cube, allowing differentiable optimization. Sparconv-VAE is the first modality-consistent variational autoencoder built entirely upon sparse convolutional networks, enabling efficient and near-lossless 3D reconstruction suitable for high-resolution generative modeling through latent diffusion. Sparc3D achieves state-of-the-art reconstruction fidelity on challenging inputs, including open surfaces, disconnected components, and intricate geometry. It preserves fine-grained shape details, reduces training and inference cost, and integrates naturally with latent diffusion models for scalable, high-resolution 3D generation.
>
---
#### [replaced 074] AnyCharV: Bootstrap Controllable Character Video Generation with Fine-to-Coarse Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.08189v2](http://arxiv.org/pdf/2502.08189v2)**

> **作者:** Zhao Wang; Hao Wen; Lingting Zhu; Chenming Shang; Yujiu Yang; Qi Dou
>
> **备注:** 18 pages, 10 figures, 4 tables
>
> **摘要:** Character video generation is a significant real-world application focused on producing high-quality videos featuring specific characters. Recent advancements have introduced various control signals to animate static characters, successfully enhancing control over the generation process. However, these methods often lack flexibility, limiting their applicability and making it challenging for users to synthesize a source character into a desired target scene. To address this issue, we propose a novel framework, AnyCharV, that flexibly generates character videos using arbitrary source characters and target scenes, guided by pose information. Our approach involves a two-stage training process. In the first stage, we develop a base model capable of integrating the source character with the target scene using pose guidance. The second stage further bootstraps controllable generation through a self-boosting mechanism, where we use the generated video in the first stage and replace the fine mask with the coarse one, enabling training outcomes with better preservation of character details. Extensive experimental results demonstrate the superiority of our method compared with previous state-of-the-art methods.
>
---
#### [replaced 075] Unsupervised Detection of Distribution Shift in Inverse Problems using Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11482v2](http://arxiv.org/pdf/2505.11482v2)**

> **作者:** Shirin Shoushtari; Edward P. Chandler; M. Salman Asif; Ulugbek S. Kamilov
>
> **摘要:** Diffusion models are widely used as priors in imaging inverse problems. However, their performance often degrades under distribution shifts between the training and test-time images. Existing methods for identifying and quantifying distribution shifts typically require access to clean test images, which are almost never available while solving inverse problems (at test time). We propose a fully unsupervised metric for estimating distribution shifts using only indirect (corrupted) measurements and score functions from diffusion models trained on different datasets. We theoretically show that this metric estimates the KL divergence between the training and test image distributions. Empirically, we show that our score-based metric, using only corrupted measurements, closely approximates the KL divergence computed from clean images. Motivated by this result, we show that aligning the out-of-distribution score with the in-distribution score -- using only corrupted measurements -- reduces the KL divergence and leads to improved reconstruction quality across multiple inverse problems.
>
---
#### [replaced 076] HV-BEV: Decoupling Horizontal and Vertical Feature Sampling for Multi-View 3D Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.18884v3](http://arxiv.org/pdf/2412.18884v3)**

> **作者:** Di Wu; Feng Yang; Benlian Xu; Pan Liao; Wenhui Zhao; Dingwen Zhang
>
> **备注:** 13 pages, 7 figures, submitted to T-ITS
>
> **摘要:** The application of vision-based multi-view environmental perception system has been increasingly recognized in autonomous driving technology, especially the BEV-based models. Current state-of-the-art solutions primarily encode image features from each camera view into the BEV space through explicit or implicit depth prediction. However, these methods often overlook the structured correlations among different parts of objects in 3D space and the fact that different categories of objects often occupy distinct local height ranges. For example, trucks appear at higher elevations, whereas traffic cones are near the ground. In this work, we propose a novel approach that decouples feature sampling in the \textbf{BEV} grid queries paradigm into \textbf{H}orizontal feature aggregation and \textbf{V}ertical adaptive height-aware reference point sampling (HV-BEV), aiming to improve both the aggregation of objects' complete information and awareness of diverse objects' height distribution. Specifically, a set of relevant neighboring points is dynamically constructed for each 3D reference point on the ground-aligned horizontal plane, enhancing the association of the same instance across different BEV grids, especially when the instance spans multiple image views around the vehicle. Additionally, instead of relying on uniform sampling within a fixed height range, we introduce a height-aware module that incorporates historical information, enabling the reference points to adaptively focus on the varying heights at which objects appear in different scenes. Extensive experiments validate the effectiveness of our proposed method, demonstrating its superior performance over the baseline across the nuScenes dataset. Moreover, our best-performing model achieves a remarkable 50.5\% mAP and 59.8\% NDS on the nuScenes testing set. The code is available at https://github.com/Uddd821/HV-BEV.
>
---
#### [replaced 077] Localizing Before Answering: A Hallucination Evaluation Benchmark for Grounded Medical Multimodal LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00744v3](http://arxiv.org/pdf/2505.00744v3)**

> **作者:** Dung Nguyen; Minh Khoi Ho; Huy Ta; Thanh Tam Nguyen; Qi Chen; Kumar Rav; Quy Duong Dang; Satwik Ramchandre; Son Lam Phung; Zhibin Liao; Minh-Son To; Johan Verjans; Phi Le Nguyen; Vu Minh Hieu Phan
>
> **备注:** Accepted at Joint Conference on Artificial Intelligence (IJCAI) 2025
>
> **摘要:** Medical Large Multi-modal Models (LMMs) have demonstrated remarkable capabilities in medical data interpretation. However, these models frequently generate hallucinations contradicting source evidence, particularly due to inadequate localization reasoning. This work reveals a critical limitation in current medical LMMs: instead of analyzing relevant pathological regions, they often rely on linguistic patterns or attend to irrelevant image areas when responding to disease-related queries. To address this, we introduce HEAL-MedVQA (Hallucination Evaluation via Localization MedVQA), a comprehensive benchmark designed to evaluate LMMs' localization abilities and hallucination robustness. HEAL-MedVQA features (i) two innovative evaluation protocols to assess visual and textual shortcut learning, and (ii) a dataset of 67K VQA pairs, with doctor-annotated anatomical segmentation masks for pathological regions. To improve visual reasoning, we propose the Localize-before-Answer (LobA) framework, which trains LMMs to localize target regions of interest and self-prompt to emphasize segmented pathological areas, generating grounded and reliable answers. Experimental results demonstrate that our approach significantly outperforms state-of-the-art biomedical LMMs on the challenging HEAL-MedVQA benchmark, advancing robustness in medical VQA.
>
---
#### [replaced 078] Mask Image Watermarking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12739v2](http://arxiv.org/pdf/2504.12739v2)**

> **作者:** Runyi Hu; Jie Zhang; Shiqian Zhao; Nils Lukas; Jiwei Li; Qing Guo; Han Qiu; Tianwei Zhang
>
> **备注:** 26 pages, 20 figures
>
> **摘要:** We present MaskMark, a simple, efficient, and flexible framework for image watermarking. MaskMark has two variants: (1) MaskMark-D, which supports global watermark embedding, watermark localization, and local watermark extraction for applications such as tamper detection; (2) MaskMark-ED, which focuses on local watermark embedding and extraction, offering enhanced robustness in small regions to support fine-grined image protection. MaskMark-D builds on the classical encoder-distortion layer-decoder training paradigm. In MaskMark-D, we introduce a simple masking mechanism during the decoding stage that enables both global and local watermark extraction. During training, the decoder is guided by various types of masks applied to watermarked images before extraction, helping it learn to localize watermarks and extract them from the corresponding local areas. MaskMark-ED extends this design by incorporating the mask into the encoding stage as well, guiding the encoder to embed the watermark in designated local regions, which improves robustness under regional attacks. Extensive experiments show that MaskMark achieves state-of-the-art performance in global and local watermark extraction, watermark localization, and multi-watermark embedding. It outperforms all existing baselines, including the recent leading model WAM for local watermarking, while preserving high visual quality of the watermarked images. In addition, MaskMark is highly efficient and adaptable. It requires only 20 hours of training on a single A6000 GPU, achieving 15x computational efficiency compared to WAM. By simply adjusting the distortion layer, MaskMark can be quickly fine-tuned to meet varying robustness requirements.
>
---
#### [replaced 079] MediConfusion: Can you trust your AI radiologist? Probing the reliability of multimodal medical foundation models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15477v2](http://arxiv.org/pdf/2409.15477v2)**

> **作者:** Mohammad Shahab Sepehri; Zalan Fabian; Maryam Soltanolkotabi; Mahdi Soltanolkotabi
>
> **备注:** 24 Pages, 9 figures, The Thirteenth International Conference on Learning Representations (ICLR) 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have tremendous potential to improve the accuracy, availability, and cost-effectiveness of healthcare by providing automated solutions or serving as aids to medical professionals. Despite promising first steps in developing medical MLLMs in the past few years, their capabilities and limitations are not well-understood. Recently, many benchmark datasets have been proposed that test the general medical knowledge of such models across a variety of medical areas. However, the systematic failure modes and vulnerabilities of such models are severely underexplored with most medical benchmarks failing to expose the shortcomings of existing models in this safety-critical domain. In this paper, we introduce MediConfusion, a challenging medical Visual Question Answering (VQA) benchmark dataset, that probes the failure modes of medical MLLMs from a vision perspective. We reveal that state-of-the-art models are easily confused by image pairs that are otherwise visually dissimilar and clearly distinct for medical experts. Strikingly, all available models (open-source or proprietary) achieve performance below random guessing on MediConfusion, raising serious concerns about the reliability of existing medical MLLMs for healthcare deployment. We also extract common patterns of model failure that may help the design of a new generation of more trustworthy and reliable MLLMs in healthcare.
>
---
#### [replaced 080] Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03654v2](http://arxiv.org/pdf/2502.03654v2)**

> **作者:** Indrashis Das; Mahmoud Safari; Steven Adriaensen; Frank Hutter
>
> **备注:** 8 pages, excluding references and appendix; v2: slight improvement in presentation. Equation (4) added, with proof in Appendix A. Appendices B (Flipped Mish) and I (Machine Translation) added. Figure 9 added to Appendix C. Appendix D extended with Heatmaps 12 and 13
>
> **摘要:** Activation functions are fundamental elements of deep learning architectures as they significantly influence training dynamics. ReLU, while widely used, is prone to the dying neuron problem, which has been mitigated by variants such as LeakyReLU, PReLU, and ELU that better handle negative neuron outputs. Recently, self-gated activations like GELU and Swish have emerged as state-of-the-art alternatives, leveraging their smoothness to ensure stable gradient flow and prevent neuron inactivity. In this work, we introduce the Gompertz Linear Unit (GoLU), a novel self-gated activation function defined as $\mathrm{GoLU}(x) = x \, \mathrm{Gompertz}(x)$, where $\mathrm{Gompertz}(x) = e^{-e^{-x}}$. The GoLU activation leverages the right-skewed asymmetry in the Gompertz function to reduce variance in the latent space more effectively compared to GELU and Swish, while preserving robust gradient flow. Extensive experiments across diverse tasks, including Image Classification, Language Modeling, Semantic Segmentation, Object Detection, Instance Segmentation, and Diffusion, highlight GoLU's superior performance relative to state-of-the-art activation functions, establishing GoLU as a robust alternative to existing activation functions.
>
---
#### [replaced 081] CLIMB-3D: Continual Learning for Imbalanced 3D Instance Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17429v2](http://arxiv.org/pdf/2502.17429v2)**

> **作者:** Vishal Thengane; Jean Lahoud; Hisham Cholakkal; Rao Muhammad Anwer; Lu Yin; Xiatian Zhu; Salman Khan
>
> **备注:** Code: https://github.com/vgthengane/CLIMB3D
>
> **摘要:** While 3D instance segmentation (3DIS) has advanced significantly, existing methods typically assume that all object classes are known in advance and are uniformly distributed. However, this assumption is unrealistic in dynamic, real-world environments where new classes emerge gradually and exhibit natural imbalance. Although some approaches have addressed class emergence, they often overlook class imbalance, resulting in suboptimal performance -- particularly on rare categories. To tackle this challenge, we propose CLIMB-3D, a unified framework for \textbf{CL}ass-incremental \textbf{Imb}alance-aware \textbf{3D}IS. Building upon established exemplar replay (ER) strategies, we show that ER alone is insufficient to achieve robust performance under constrained memory conditions. To mitigate this, we introduce a novel pseudo-label generator (PLG) that extends supervision to previously learned categories by leveraging predictions from a frozen prior model. Despite its promise, PLG tends to bias towards frequent classes. Therefore, we propose a class-balanced re-weighting (CBR) scheme, that estimates object frequencies from pseudo-labels and dynamically adjusts training bias -- without requiring access to past data. We design and evaluate three incremental scenarios for 3DIS on the challenging ScanNet200 dataset, and additionally on semantic segmentation on ScanNetV2. Our approach achieves state-of-the-art results, surpassing prior work by up to 16.76\% mAP for instance segmentation and approximately 30\% mIoU for semantic segmentation, demonstrating strong generalization across both frequent and rare classes.
>
---
