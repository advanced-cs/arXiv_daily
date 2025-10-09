# 计算机视觉 cs.CV

- **最新发布 114 篇**

- **更新 72 篇**

## 最新发布

#### [new 001] Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding
- **分类: cs.CV**

- **简介: 论文提出Lumina-DiMOO，一个基于离散扩散的多模态大模型，用于统一处理生成与理解任务。该模型采用全离散扩散架构，提升采样效率，支持文本到图像、图像到图像生成及图像理解等多模态任务。论文任务属多模态AI领域，旨在解决统一建模与高效生成问题。**

- **链接: [http://arxiv.org/pdf/2510.06308v1](http://arxiv.org/pdf/2510.06308v1)**

> **作者:** Yi Xin; Qi Qin; Siqi Luo; Kaiwen Zhu; Juncheng Yan; Yan Tai; Jiayi Lei; Yuewen Cao; Keqi Wang; Yibin Wang; Jinbin Bai; Qian Yu; Dengyang Jiang; Yuandong Pu; Haoxing Chen; Le Zhuo; Junjun He; Gen Luo; Tianbin Li; Ming Hu; Jin Ye; Shenglong Ye; Bo Zhang; Chang Xu; Wenhai Wang; Hongsheng Li; Guangtao Zhai; Tianfan Xue; Bin Fu; Xiaohong Liu; Yu Qiao; Yihao Liu
>
> **备注:** 33 pages, 13 figures, 10 tables
>
> **摘要:** We introduce Lumina-DiMOO, an open-source foundational model for seamless multi-modal generation and understanding. Lumina-DiMOO sets itself apart from prior unified models by utilizing a fully discrete diffusion modeling to handle inputs and outputs across various modalities. This innovative approach allows Lumina-DiMOO to achieve higher sampling efficiency compared to previous autoregressive (AR) or hybrid AR-Diffusion paradigms and adeptly support a broad spectrum of multi-modal tasks, including text-to-image generation, image-to-image generation (e.g., image editing, subject-driven generation, and image inpainting, etc.), as well as image understanding. Lumina-DiMOO achieves state-of-the-art performance on multiple benchmarks, surpassing existing open-source unified multi-modal models. To foster further advancements in multi-modal and discrete diffusion model research, we release our code and checkpoints to the community. Project Page: https://synbol.github.io/Lumina-DiMOO.
>
---
#### [new 002] MATRIX: Mask Track Alignment for Interaction-aware Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决现有视频生成模型在多实例交互建模上的不足。作者提出MATRIX方法，通过结合多实例掩码轨迹对模型注意力机制进行对齐正则化，并构建了交互感知评估协议InterGenEval。实验表明，该方法有效提升了视频生成中的交互准确性和语义一致性，减少了漂移和幻觉问题。**

- **链接: [http://arxiv.org/pdf/2510.07310v1](http://arxiv.org/pdf/2510.07310v1)**

> **作者:** Siyoon Jin; Seongchan Kim; Dahyun Chung; Jaeho Lee; Hyunwook Choi; Jisu Nam; Jiyoung Kim; Seungryong Kim
>
> **备注:** Project Page is available at: https://cvlab-kaist.github.io/MATRIX/
>
> **摘要:** Video DiTs have advanced video generation, yet they still struggle to model multi-instance or subject-object interactions. This raises a key question: How do these models internally represent interactions? To answer this, we curate MATRIX-11K, a video dataset with interaction-aware captions and multi-instance mask tracks. Using this dataset, we conduct a systematic analysis that formalizes two perspectives of video DiTs: semantic grounding, via video-to-text attention, which evaluates whether noun and verb tokens capture instances and their relations; and semantic propagation, via video-to-video attention, which assesses whether instance bindings persist across frames. We find both effects concentrate in a small subset of interaction-dominant layers. Motivated by this, we introduce MATRIX, a simple and effective regularization that aligns attention in specific layers of video DiTs with multi-instance mask tracks from the MATRIX-11K dataset, enhancing both grounding and propagation. We further propose InterGenEval, an evaluation protocol for interaction-aware video generation. In experiments, MATRIX improves both interaction fidelity and semantic alignment while reducing drift and hallucination. Extensive ablations validate our design choices. Codes and weights will be released.
>
---
#### [new 003] Evaluating Fundus-Specific Foundation Models for Diabetic Macular Edema Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决糖尿病黄斑水肿（DME）检测中数据不足导致的深度学习应用难题。论文比较了多种基础模型（如RETFound、FLAIR）与传统迁移学习方法在多个数据集上的表现，发现轻量级CNN（如EfficientNet-B0）在多数设置下表现更优，表明基础模型在细粒度眼科任务中未必更有效。**

- **链接: [http://arxiv.org/pdf/2510.07277v1](http://arxiv.org/pdf/2510.07277v1)**

> **作者:** Franco Javier Arellano; José Ignacio Orlando
>
> **备注:** Accepted for publication at SIPAIM 2025
>
> **摘要:** Diabetic Macular Edema (DME) is a leading cause of vision loss among patients with Diabetic Retinopathy (DR). While deep learning has shown promising results for automatically detecting this condition from fundus images, its application remains challenging due the limited availability of annotated data. Foundation Models (FM) have emerged as an alternative solution. However, it is unclear if they can cope with DME detection in particular. In this paper, we systematically compare different FM and standard transfer learning approaches for this task. Specifically, we compare the two most popular FM for retinal images--RETFound and FLAIR--and an EfficientNet-B0 backbone, across different training regimes and evaluation settings in IDRiD, MESSIDOR-2 and OCT-and-Eye-Fundus-Images (OEFI). Results show that despite their scale, FM do not consistently outperform fine-tuned CNNs in this task. In particular, an EfficientNet-B0 ranked first or second in terms of area under the ROC and precision/recall curves in most evaluation settings, with RETFound only showing promising results in OEFI. FLAIR, on the other hand, demonstrated competitive zero-shot performance, achieving notable AUC-PR scores when prompted appropriately. These findings reveal that FM might not be a good tool for fine-grained ophthalmic tasks such as DME detection even after fine-tuning, suggesting that lightweight CNNs remain strong baselines in data-scarce environments.
>
---
#### [new 004] HARP-NeXt: High-Speed and Accurate Range-Point Fusion Network for 3D LiDAR Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 本文属于3D LiDAR语义分割任务，旨在解决现有方法在准确性和速度之间的权衡问题。作者提出HARP-NeXt网络，通过高效预处理、Conv-SE-NeXt模块和多尺度range-point融合结构，在保持高精度的同时显著提升推理速度，适用于资源受限的嵌入式系统。**

- **链接: [http://arxiv.org/pdf/2510.06876v1](http://arxiv.org/pdf/2510.06876v1)**

> **作者:** Samir Abou Haidar; Alexandre Chariot; Mehdi Darouich; Cyril Joly; Jean-Emmanuel Deschaud
>
> **备注:** Accepted at IROS 2025 (IEEE/RSJ International Conference on Intelligent Robots and Systems)
>
> **摘要:** LiDAR semantic segmentation is crucial for autonomous vehicles and mobile robots, requiring high accuracy and real-time processing, especially on resource-constrained embedded systems. Previous state-of-the-art methods often face a trade-off between accuracy and speed. Point-based and sparse convolution-based methods are accurate but slow due to the complexity of neighbor searching and 3D convolutions. Projection-based methods are faster but lose critical geometric information during the 2D projection. Additionally, many recent methods rely on test-time augmentation (TTA) to improve performance, which further slows the inference. Moreover, the pre-processing phase across all methods increases execution time and is demanding on embedded platforms. Therefore, we introduce HARP-NeXt, a high-speed and accurate LiDAR semantic segmentation network. We first propose a novel pre-processing methodology that significantly reduces computational overhead. Then, we design the Conv-SE-NeXt feature extraction block to efficiently capture representations without deep layer stacking per network stage. We also employ a multi-scale range-point fusion backbone that leverages information at multiple abstraction levels to preserve essential geometric details, thereby enhancing accuracy. Experiments on the nuScenes and SemanticKITTI benchmarks show that HARP-NeXt achieves a superior speed-accuracy trade-off compared to all state-of-the-art methods, and, without relying on ensemble models or TTA, is comparable to the top-ranked PTv3, while running 24$\times$ faster. The code is available at https://github.com/SamirAbouHaidar/HARP-NeXt
>
---
#### [new 005] Improving Artifact Robustness for CT Deep Learning Models Without Labeled Artifact Images via Domain Adaptation
- **分类: cs.CV; q-bio.TO**

- **简介: 该论文属于医学图像分类任务，旨在解决CT图像中因新出现的伪影导致深度学习模型性能下降的问题。作者通过域适应方法（DANN），在无需标注新伪影图像的情况下，提升模型对含伪影图像的分类鲁棒性。实验表明该方法有效，且具备对未见噪声的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.06584v1](http://arxiv.org/pdf/2510.06584v1)**

> **作者:** Justin Cheung; Samuel Savine; Calvin Nguyen; Lin Lu; Alhassan S. Yasin
>
> **备注:** 8 pages, 12 figures, 1 table
>
> **摘要:** Deep learning models which perform well on images from their training distribution can degrade substantially when applied to new distributions. If a CT scanner introduces a new artifact not present in the training labels, the model may misclassify the images. Although modern CT scanners include design features which mitigate these artifacts, unanticipated or difficult-to-mitigate artifacts can still appear in practice. The direct solution of labeling images from this new distribution can be costly. As a more accessible alternative, this study evaluates domain adaptation as an approach for training models that maintain classification performance despite new artifacts, even without corresponding labels. We simulate ring artifacts from detector gain error in sinogram space and evaluate domain adversarial neural networks (DANN) against baseline and augmentation-based approaches on the OrganAMNIST abdominal CT dataset. Our results demonstrate that baseline models trained only on clean images fail to generalize to images with ring artifacts, and traditional augmentation with other distortion types provides no improvement on unseen artifact domains. In contrast, the DANN approach successfully maintains high classification accuracy on ring artifact images using only unlabeled artifact data during training, demonstrating the viability of domain adaptation for artifact robustness. The domain-adapted model achieved classification performance on ring artifact test data comparable to models explicitly trained with labeled artifact images, while also showing unexpected generalization to uniform noise. These findings provide empirical evidence that domain adaptation can effectively address distribution shift in medical imaging without requiring expensive expert labeling of new artifact distributions, suggesting promise for deployment in clinical settings where novel artifacts may emerge.
>
---
#### [new 006] DADO: A Depth-Attention framework for Object Discovery
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的无监督目标发现任务，旨在无需人工标注的情况下识别和定位图像中的物体。针对注意力图噪声和复杂场景问题，论文提出DADO模型，结合注意力机制与深度模型，并引入动态加权策略以自适应强调关键特征。实验表明，DADO在标准数据集上优于现有方法，具有更高的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.07089v1](http://arxiv.org/pdf/2510.07089v1)**

> **作者:** Federico Gonzalez; Estefania Talavera; Petia Radeva
>
> **备注:** 21st International Conference in Computer Analysis of Images and Patterns (CAIP 2025)
>
> **摘要:** Unsupervised object discovery, the task of identifying and localizing objects in images without human-annotated labels, remains a significant challenge and a growing focus in computer vision. In this work, we introduce a novel model, DADO (Depth-Attention self-supervised technique for Discovering unseen Objects), which combines an attention mechanism and a depth model to identify potential objects in images. To address challenges such as noisy attention maps or complex scenes with varying depth planes, DADO employs dynamic weighting to adaptively emphasize attention or depth features based on the global characteristics of each image. We evaluated DADO on standard benchmarks, where it outperforms state-of-the-art methods in object discovery accuracy and robustness without the need for fine-tuning.
>
---
#### [new 007] Label-frugal satellite image change detection with generative virtual exemplar learning
- **分类: cs.CV**

- **简介: 论文属于遥感图像变化检测任务，旨在减少对大量人工标注数据的依赖。为解决标注成本高的问题，提出一种基于主动学习的新方法，通过生成虚拟样例（virtual exemplars），选择最具代表性和挑战性的样本供人工标注，从而提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.06926v1](http://arxiv.org/pdf/2510.06926v1)**

> **作者:** Hichem Sahbi
>
> **摘要:** Change detection is a major task in remote sensing which consists in finding all the occurrences of changes in multi-temporal satellite or aerial images. The success of existing methods, and particularly deep learning ones, is tributary to the availability of hand-labeled training data that capture the acquisition conditions and the subjectivity of the user (oracle). In this paper, we devise a novel change detection algorithm, based on active learning. The main contribution of our work resides in a new model that measures how important is each unlabeled sample, and provides an oracle with only the most critical samples (also referred to as virtual exemplars) for further labeling. These exemplars are generated, using an invertible graph convnet, as the optimum of an adversarial loss that (i) measures representativity, diversity and ambiguity of the data, and thereby (ii) challenges (the most) the current change detection criteria, leading to a better re-estimate of these criteria in the subsequent iterations of active learning. Extensive experiments show the positive impact of our label-efficient learning model against comparative methods.
>
---
#### [new 008] Superpixel Integrated Grids for Fast Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决超像素因不规则分布导致深度学习方法受限的问题。作者提出SIGRID数据结构，结合颜色与形状信息，在降低输入维度的同时保持甚至提升了分割性能，兼顾效率与准确率。**

- **链接: [http://arxiv.org/pdf/2510.06487v1](http://arxiv.org/pdf/2510.06487v1)**

> **作者:** Jack Roberts; Jeova Farias Sales Rocha Neto
>
> **摘要:** Superpixels have long been used in image simplification to enable more efficient data processing and storage. However, despite their computational potential, their irregular spatial distribution has often forced deep learning approaches to rely on specialized training algorithms and architectures, undermining the original motivation for superpixelations. In this work, we introduce a new superpixel-based data structure, SIGRID (Superpixel-Integrated Grid), as an alternative to full-resolution images in segmentation tasks. By leveraging classical shape descriptors, SIGRID encodes both color and shape information of superpixels while substantially reducing input dimensionality. We evaluate SIGRIDs on four benchmark datasets using two popular convolutional segmentation architectures. Our results show that, despite compressing the original data, SIGRIDs not only match but in some cases surpass the performance of pixel-level representations, all while significantly accelerating model training. This demonstrates that SIGRIDs achieve a favorable balance between accuracy and computational efficiency.
>
---
#### [new 009] MV-Performer: Taming Video Diffusion Model for Faithful and Synchronized Multi-view Performer Synthesis
- **分类: cs.CV**

- **简介: 该论文属于多视角视频生成任务，旨在解决从单目视频生成360度同步多视角人体视频的问题。作者提出了MV-Performer框架，利用多视角人类数据集和条件信号（如法线图），设计了多视角视频扩散模型，并优化推理过程，以提升生成效果与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.07190v1](http://arxiv.org/pdf/2510.07190v1)**

> **作者:** Yihao Zhi; Chenghong Li; Hongjie Liao; Xihe Yang; Zhengwentai Sun; Jiahao Chang; Xiaodong Cun; Wensen Feng; Xiaoguang Han
>
> **备注:** Accepted by SIGGRAPH Asia 2025 conference track
>
> **摘要:** Recent breakthroughs in video generation, powered by large-scale datasets and diffusion techniques, have shown that video diffusion models can function as implicit 4D novel view synthesizers. Nevertheless, current methods primarily concentrate on redirecting camera trajectory within the front view while struggling to generate 360-degree viewpoint changes. In this paper, we focus on human-centric subdomain and present MV-Performer, an innovative framework for creating synchronized novel view videos from monocular full-body captures. To achieve a 360-degree synthesis, we extensively leverage the MVHumanNet dataset and incorporate an informative condition signal. Specifically, we use the camera-dependent normal maps rendered from oriented partial point clouds, which effectively alleviate the ambiguity between seen and unseen observations. To maintain synchronization in the generated videos, we propose a multi-view human-centric video diffusion model that fuses information from the reference video, partial rendering, and different viewpoints. Additionally, we provide a robust inference procedure for in-the-wild video cases, which greatly mitigates the artifacts induced by imperfect monocular depth estimation. Extensive experiments on three datasets demonstrate our MV-Performer's state-of-the-art effectiveness and robustness, setting a strong model for human-centric 4D novel view synthesis.
>
---
#### [new 010] Uncertainty Quantification In Surface Landmines and UXO Classification Using MC Dropout
- **分类: cs.CV; cs.AI; cs.LG; stat.OT**

- **简介: 该论文属于地雷与未爆物分类任务，旨在解决深度学习模型在复杂环境下易受干扰、可靠性不足的问题。作者采用MC Dropout方法，在ResNet-50模型中量化预测不确定性，提升分类可靠性。实验表明该方法能有效识别不可靠预测，增强排雷决策的安全性。**

- **链接: [http://arxiv.org/pdf/2510.06238v1](http://arxiv.org/pdf/2510.06238v1)**

> **作者:** Sagar Lekhak; Emmett J. Ientilucci; Dimah Dera; Susmita Ghosh
>
> **备注:** This work has been accepted and presented at IGARSS 2025 and will appear in the IEEE IGARSS 2025 proceedings
>
> **摘要:** Detecting surface landmines and unexploded ordnances (UXOs) using deep learning has shown promise in humanitarian demining. However, deterministic neural networks can be vulnerable to noisy conditions and adversarial attacks, leading to missed detection or misclassification. This study introduces the idea of uncertainty quantification through Monte Carlo (MC) Dropout, integrated into a fine-tuned ResNet-50 architecture for surface landmine and UXO classification, which was tested on a simulated dataset. Integrating the MC Dropout approach helps quantify epistemic uncertainty, providing an additional metric for prediction reliability, which could be helpful to make more informed decisions in demining operations. Experimental results on clean, adversarially perturbed, and noisy test images demonstrate the model's ability to flag unreliable predictions under challenging conditions. This proof-of-concept study highlights the need for uncertainty quantification in demining, raises awareness about the vulnerability of existing neural networks in demining to adversarial threats, and emphasizes the importance of developing more robust and reliable models for practical applications.
>
---
#### [new 011] StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 论文提出StaR-KVQA，用于解决隐式知识视觉问答（IK-KVQA）任务中的推理监督不足、泛化能力差的问题。该方法通过构建结构化推理轨迹（包括符号关系路径和自然语言解释），实现推理过程的可解释性与一致性，提升准确率与跨领域泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.06638v1](http://arxiv.org/pdf/2510.06638v1)**

> **作者:** Zhihao Wen; Wenkang Wei; Yuan Fang; Xingtong Yu; Hui Zhang; Weicheng Zhu; Xin Zhang
>
> **摘要:** Knowledge-based Visual Question Answering (KVQA) requires models to ground entities in images and reason over factual knowledge. We study its implicit-knowledge variant, IK-KVQA, where a multimodal large language model (MLLM) is the sole knowledge source, without external retrieval. Yet, MLLMs lack explicit reasoning supervision and produce inconsistent justifications, and generalize poorly after standard supervised fine-tuning (SFT). We present StaR-KVQA (Structured Reasoning Traces for IK-KVQA), which supervises structured traces - dual symbolic relation paths plus path-grounded natural-language explanations - so that reasoning becomes transparent and verifiable. With one open-source MLLM, StaR-KVQA constructs and selects path-grounded reasoning traces to form a trace-enriched dataset, then fine-tunes via structured self-distillation to align generation with supervision; no external retrievers, verifiers, or curated knowledge bases (KBs) are used, traces are built offline, and inference is a single autoregressive pass. Across benchmarks, StaR-KVQA improves both accuracy and interpretability, achieving up to +11.3% higher answer accuracy on OK-VQA over the strongest baseline while exhibiting robust cross-domain generalization.
>
---
#### [new 012] A Bridge from Audio to Video: Phoneme-Viseme Alignment Allows Every Face to Speak Multiple Languages
- **分类: cs.CV**

- **简介: 该论文属于语音驱动说话人脸合成任务，旨在解决非英语语言生成效果差的问题。作者提出MuEx框架，利用音素-可视素对齐机制，实现跨语言的音视频同步，并构建了包含12种语言的多语言说话人脸基准进行验证。**

- **链接: [http://arxiv.org/pdf/2510.06612v1](http://arxiv.org/pdf/2510.06612v1)**

> **作者:** Zibo Su; Kun Wei; Jiahua Li; Xu Yang; Cheng Deng
>
> **摘要:** Speech-driven talking face synthesis (TFS) focuses on generating lifelike facial animations from audio input. Current TFS models perform well in English but unsatisfactorily in non-English languages, producing wrong mouth shapes and rigid facial expressions. The terrible performance is caused by the English-dominated training datasets and the lack of cross-language generalization abilities. Thus, we propose Multilingual Experts (MuEx), a novel framework featuring a Phoneme-Guided Mixture-of-Experts (PG-MoE) architecture that employs phonemes and visemes as universal intermediaries to bridge audio and video modalities, achieving lifelike multilingual TFS. To alleviate the influence of linguistic differences and dataset bias, we extract audio and video features as phonemes and visemes respectively, which are the basic units of speech sounds and mouth movements. To address audiovisual synchronization issues, we introduce the Phoneme-Viseme Alignment Mechanism (PV-Align), which establishes robust cross-modal correspondences between phonemes and visemes. In addition, we build a Multilingual Talking Face Benchmark (MTFB) comprising 12 diverse languages with 95.04 hours of high-quality videos for training and evaluating multilingual TFS performance. Extensive experiments demonstrate that MuEx achieves superior performance across all languages in MTFB and exhibits effective zero-shot generalization to unseen languages without additional training.
>
---
#### [new 013] SDQM: Synthetic Data Quality Metric for Object Detection Dataset Evaluation
- **分类: cs.CV; cs.AI; cs.IT; cs.LG; math.IT**

- **简介: 该论文属于目标检测任务，旨在解决合成数据质量评估问题。现有方法依赖模型训练，成本高且相关性弱。作者提出SDQM，一种无需训练模型即可评估合成数据质量的高效指标。实验表明，SDQM与YOLOv11的mAP得分高度相关，并能提供改进数据质量的可行见解，提升合成数据生成效率。**

- **链接: [http://arxiv.org/pdf/2510.06596v1](http://arxiv.org/pdf/2510.06596v1)**

> **作者:** Ayush Zenith; Arnold Zumbrun; Neel Raut; Jing Lin
>
> **摘要:** The performance of machine learning models depends heavily on training data. The scarcity of large-scale, well-annotated datasets poses significant challenges in creating robust models. To address this, synthetic data generated through simulations and generative models has emerged as a promising solution, enhancing dataset diversity and improving the performance, reliability, and resilience of models. However, evaluating the quality of this generated data requires an effective metric. This paper introduces the Synthetic Dataset Quality Metric (SDQM) to assess data quality for object detection tasks without requiring model training to converge. This metric enables more efficient generation and selection of synthetic datasets, addressing a key challenge in resource-constrained object detection tasks. In our experiments, SDQM demonstrated a strong correlation with the mean Average Precision (mAP) scores of YOLOv11, a leading object detection model, while previous metrics only exhibited moderate or weak correlations. Additionally, it provides actionable insights for improving dataset quality, minimizing the need for costly iterative training. This scalable and efficient metric sets a new standard for evaluating synthetic data. The code for SDQM is available at https://github.com/ayushzenith/SDQM
>
---
#### [new 014] Cluster Paths: Navigating Interpretability in Neural Networks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于可解释性研究任务，旨在解决神经网络决策过程不透明的问题。作者提出“聚类路径”方法，通过聚类神经网络各层激活值，生成输入样本的路径表示，并设计四个指标评估路径质量。实验表明该方法能有效识别模型依赖的视觉概念、检测异常样本，并在多种模型上验证了其解释性和稳定性。**

- **链接: [http://arxiv.org/pdf/2510.06541v1](http://arxiv.org/pdf/2510.06541v1)**

> **作者:** Nicholas M. Kroeger; Vincent Bindschaedler
>
> **摘要:** While modern deep neural networks achieve impressive performance in vision tasks, they remain opaque in their decision processes, risking unwarranted trust, undetected biases and unexpected failures. We propose cluster paths, a post-hoc interpretability method that clusters activations at selected layers and represents each input as its sequence of cluster IDs. To assess these cluster paths, we introduce four metrics: path complexity (cognitive load), weighted-path purity (class alignment), decision-alignment faithfulness (predictive fidelity), and path agreement (stability under perturbations). In a spurious-cue CIFAR-10 experiment, cluster paths identify color-based shortcuts and collapse when the cue is removed. On a five-class CelebA hair-color task, they achieve 90% faithfulness and maintain 96% agreement under Gaussian noise without sacrificing accuracy. Scaling to a Vision Transformer pretrained on ImageNet, we extend cluster paths to concept paths derived from prompting a large language model on minimal path divergences. Finally, we show that cluster paths can serve as an effective out-of-distribution (OOD) detector, reliably flagging anomalous samples before the model generates over-confident predictions. Cluster paths uncover visual concepts, such as color palettes, textures, or object contexts, at multiple network depths, demonstrating that cluster paths scale to large vision models while generating concise and human-readable explanations.
>
---
#### [new 015] HSNet: Heterogeneous Subgraph Network for Single Image Super-resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像超分辨率任务，旨在解决现有方法在结构灵活性和计算复杂度上的不足。论文提出HSNet，通过构建异质子图网络，分解全局图为多个子图，利用子图集合和聚合策略提升图像重建质量，同时降低计算开销，实现性能与效率的平衡。**

- **链接: [http://arxiv.org/pdf/2510.06564v1](http://arxiv.org/pdf/2510.06564v1)**

> **作者:** Qiongyang Hu; Wenyang Liu; Wenbin Zou; Yuejiao Su; Lap-Pui Chau; Yi Wang
>
> **摘要:** Existing deep learning approaches for image super-resolution, particularly those based on CNNs and attention mechanisms, often suffer from structural inflexibility. Although graph-based methods offer greater representational adaptability, they are frequently impeded by excessive computational complexity. To overcome these limitations, this paper proposes the Heterogeneous Subgraph Network (HSNet), a novel framework that efficiently leverages graph modeling while maintaining computational feasibility. The core idea of HSNet is to decompose the global graph into manageable sub-components. First, we introduce the Constructive Subgraph Set Block (CSSB), which generates a diverse set of complementary subgraphs. Rather than relying on a single monolithic graph, CSSB captures heterogeneous characteristics of the image by modeling different relational patterns and feature interactions, producing a rich ensemble of both local and global graph structures. Subsequently, the Subgraph Aggregation Block (SAB) integrates the representations embedded across these subgraphs. Through adaptive weighting and fusion of multi-graph features, SAB constructs a comprehensive and discriminative representation that captures intricate interdependencies. Furthermore, a Node Sampling Strategy (NSS) is designed to selectively retain the most salient features, thereby enhancing accuracy while reducing computational overhead. Extensive experiments demonstrate that HSNet achieves state-of-the-art performance, effectively balancing reconstruction quality with computational efficiency. The code will be made publicly available.
>
---
#### [new 016] General and Efficient Visual Goal-Conditioned Reinforcement Learning using Object-Agnostic Masks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉目标条件强化学习任务，旨在解决现有目标表示方法泛化性差、收敛慢等问题。作者提出一种对象无关的掩码目标表示方法，通过掩码生成密集奖励，提升了学习效率和泛化能力，并实现了从仿真到真实机器人的迁移应用。**

- **链接: [http://arxiv.org/pdf/2510.06277v1](http://arxiv.org/pdf/2510.06277v1)**

> **作者:** Fahim Shahriar; Cheryl Wang; Alireza Azimi; Gautham Vasan; Hany Hamed Elanwar; A. Rupam Mahmood; Colin Bellinger
>
> **摘要:** Goal-conditioned reinforcement learning (GCRL) allows agents to learn diverse objectives using a unified policy. The success of GCRL, however, is contingent on the choice of goal representation. In this work, we propose a mask-based goal representation system that provides object-agnostic visual cues to the agent, enabling efficient learning and superior generalization. In contrast, existing goal representation methods, such as target state images, 3D coordinates, and one-hot vectors, face issues of poor generalization to unseen objects, slow convergence, and the need for special cameras. Masks can be processed to generate dense rewards without requiring error-prone distance calculations. Learning with ground truth masks in simulation, we achieved 99.9% reaching accuracy on training and unseen test objects. Our proposed method can be utilized to perform pick-up tasks with high accuracy, without using any positional information of the target. Moreover, we demonstrate learning from scratch and sim-to-real transfer applications using two different physical robots, utilizing pretrained open vocabulary object detection models for mask generation.
>
---
#### [new 017] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Reranking
- **分类: cs.CV; cs.LG**

- **简介: 论文提出EDJE，一种高效的视觉-语言联合编码器，用于大规模多模态重排序任务。旨在解决现有模型因视觉特征提取昂贵而难以部署的问题，通过离线预计算并压缩视觉token，实现快速、低存储的在线推理。**

- **链接: [http://arxiv.org/pdf/2510.06820v1](http://arxiv.org/pdf/2510.06820v1)**

> **作者:** Mitchell Keren Taraday; Shahaf Wagner; Chaim Baskin
>
> **备注:** preprint
>
> **摘要:** Multimodal retrieval still leans on embedding-based models like CLIP for fast vector search over pre-computed image embeddings. Yet, unlike text retrieval, where joint-encoder rerankers are standard, comparable vision--language rerankers are largely absent. We find that seminal joint encoders such as BLIP are severely bottlenecked by an expensive visual feature-extraction stage, preventing practical deployment at scale. Motivated by this bottleneck, we introduce EDJE, an Efficient Discriminative Joint Encoder that precomputes vision tokens offline and compresses them via a lightweight attention-based adapter, so online inference runs only a compact joint encoder over a small set of visual tokens plus the text. EDJE preserves strong retrieval performance while drastically reducing storage and online compute, enabling high-throughput inference. Specifically, EDJE processes 50k image--text pairs/second while requiring 49kB of disk storage per image, matching prior art on Flickr (zero-shot) and COCO (fine-tuned) retrieval. The implementation and checkpoints will be made publicly available shortly.
>
---
#### [new 018] MoRe: Monocular Geometry Refinement via Graph Optimization for Cross-View Consistency
- **分类: cs.CV**

- **简介: 该论文属于3D视觉任务，旨在解决单目几何先验的尺度模糊问题。通过构建基于图优化的MoRe框架，利用帧间特征匹配与局部平面逼近，提升跨视角一致性与尺度对齐。方法无需训练，优化3D结构并增强新视角合成效果，尤其适用于稀疏视角渲染场景。**

- **链接: [http://arxiv.org/pdf/2510.07119v1](http://arxiv.org/pdf/2510.07119v1)**

> **作者:** Dongki Jung; Jaehoon Choi; Yonghan Lee; Sungmin Eum; Heesung Kwon; Dinesh Manocha
>
> **摘要:** Monocular 3D foundation models offer an extensible solution for perception tasks, making them attractive for broader 3D vision applications. In this paper, we propose MoRe, a training-free Monocular Geometry Refinement method designed to improve cross-view consistency and achieve scale alignment. To induce inter-frame relationships, our method employs feature matching between frames to establish correspondences. Rather than applying simple least squares optimization on these matched points, we formulate a graph-based optimization framework that performs local planar approximation using the estimated 3D points and surface normals estimated by monocular foundation models. This formulation addresses the scale ambiguity inherent in monocular geometric priors while preserving the underlying 3D structure. We further demonstrate that MoRe not only enhances 3D reconstruction but also improves novel view synthesis, particularly in sparse view rendering scenarios.
>
---
#### [new 019] TransFIRA: Transfer Learning for Face Image Recognizability Assessment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像质量评估任务，旨在解决无约束环境下人脸图像的可识别性预测问题。现有方法依赖人工特征或复杂生成流程，与识别模型的决策不一致。论文提出TransFIRA，基于迁移学习，通过类中心相似度和角度分离度定义可识别性，在嵌入空间中实现高效、可解释的评估，并扩展至身体识别，提升了准确性和适用范围。**

- **链接: [http://arxiv.org/pdf/2510.06353v1](http://arxiv.org/pdf/2510.06353v1)**

> **作者:** Allen Tu; Kartik Narayan; Joshua Gleason; Jennifer Xu; Matthew Meyn; Tom Goldstein; Vishal M. Patel
>
> **备注:** Project Page: https://transfira.github.io/
>
> **摘要:** Face recognition in unconstrained environments such as surveillance, video, and web imagery must contend with extreme variation in pose, blur, illumination, and occlusion, where conventional visual quality metrics fail to predict whether inputs are truly recognizable to the deployed encoder. Existing FIQA methods typically rely on visual heuristics, curated annotations, or computationally intensive generative pipelines, leaving their predictions detached from the encoder's decision geometry. We introduce TransFIRA (Transfer Learning for Face Image Recognizability Assessment), a lightweight and annotation-free framework that grounds recognizability directly in embedding space. TransFIRA delivers three advances: (i) a definition of recognizability via class-center similarity (CCS) and class-center angular separation (CCAS), yielding the first natural, decision-boundary--aligned criterion for filtering and weighting; (ii) a recognizability-informed aggregation strategy that achieves state-of-the-art verification accuracy on BRIAR and IJB-C while nearly doubling correlation with true recognizability, all without external labels, heuristics, or backbone-specific training; and (iii) new extensions beyond faces, including encoder-grounded explainability that reveals how degradations and subject-specific factors affect recognizability, and the first recognizability-aware body recognition assessment. Experiments confirm state-of-the-art results on faces, strong performance on body recognition, and robustness under cross-dataset shifts. Together, these contributions establish TransFIRA as a unified, geometry-driven framework for recognizability assessment -- encoder-specific, accurate, interpretable, and extensible across modalities -- significantly advancing FIQA in accuracy, explainability, and scope.
>
---
#### [new 020] Improving the Spatial Resolution of GONG Solar Images to GST Quality Using Deep Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像超分辨率任务，旨在提升GONG低分辨率太阳Hα图像的空间分辨率，使其接近GST高分辨率观测质量。论文采用基于GAN的Real-ESRGAN模型，结合残差密集块和相对判别器，对齐GONG与GST图像对，有效恢复太阳黑子、日珥等细节，取得了较好的重建效果。**

- **链接: [http://arxiv.org/pdf/2510.06281v1](http://arxiv.org/pdf/2510.06281v1)**

> **作者:** Chenyang Li; Qin Li; Haimin Wang; Bo Shen
>
> **备注:** 5 pages; accepted as a workshop paper in ICDM 2025
>
> **摘要:** High-resolution (HR) solar imaging is crucial for capturing fine-scale dynamic features such as filaments and fibrils. However, the spatial resolution of the full-disk H$\alpha$ images is limited and insufficient to resolve these small-scale structures. To address this, we propose a GAN-based superresolution approach to enhance low-resolution (LR) full-disk H$\alpha$ images from the Global Oscillation Network Group (GONG) to a quality comparable with HR observations from the Big Bear Solar Observatory/Goode Solar Telescope (BBSO/GST). We employ Real-ESRGAN with Residual-in-Residual Dense Blocks and a relativistic discriminator. We carefully aligned GONG-GST pairs. The model effectively recovers fine details within sunspot penumbrae and resolves fine details in filaments and fibrils, achieving an average mean squared error (MSE) of 467.15, root mean squared error (RMSE) of 21.59, and cross-correlation (CC) of 0.7794. Slight misalignments between image pairs limit quantitative performance, which we plan to address in future work alongside dataset expansion to further improve reconstruction quality.
>
---
#### [new 021] MSITrack: A Challenging Benchmark for Multispectral Single Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的目标跟踪任务，旨在解决真实场景中RGB跟踪器受限于遮挡、相似物干扰等问题。作者构建了大规模多光谱跟踪数据集MSITrack，包含300个视频、129k帧及55类物体，提升目标辨识度并推动多光谱跟踪研究。**

- **链接: [http://arxiv.org/pdf/2510.06619v1](http://arxiv.org/pdf/2510.06619v1)**

> **作者:** Tao Feng; Tingfa Xu; Haolin Qin; Tianhao Li; Shuaihao Han; Xuyang Zou; Zhan Lv; Jianan Li
>
> **摘要:** Visual object tracking in real-world scenarios presents numerous challenges including occlusion, interference from similar objects and complex backgrounds-all of which limit the effectiveness of RGB-based trackers. Multispectral imagery, which captures pixel-level spectral reflectance, enhances target discriminability. However, the availability of multispectral tracking datasets remains limited. To bridge this gap, we introduce MSITrack, the largest and most diverse multispectral single object tracking dataset to date. MSITrack offers the following key features: (i) More Challenging Attributes-including interference from similar objects and similarity in color and texture between targets and backgrounds in natural scenarios, along with a wide range of real-world tracking challenges; (ii) Richer and More Natural Scenes-spanning 55 object categories and 300 distinct natural scenes, MSITrack far exceeds the scope of existing benchmarks. Many of these scenes and categories are introduced to the multispectral tracking domain for the first time; (iii) Larger Scale-300 videos comprising over 129k frames of multispectral imagery. To ensure annotation precision, each frame has undergone meticulous processing, manual labeling and multi-stage verification. Extensive evaluations using representative trackers demonstrate that the multispectral data in MSITrack significantly improves performance over RGB-only baselines, highlighting its potential to drive future advancements in the field. The MSITrack dataset is publicly available at: https://github.com/Fengtao191/MSITrack.
>
---
#### [new 022] Concept Retrieval -- What and How?
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，旨在根据图像的中心概念进行检索，超越传统视觉或语义相似性方法。论文定义了概念检索的问题与评估指标，提出通过嵌入空间邻居的双模态高斯分布建模来识别概念，并验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.07058v1](http://arxiv.org/pdf/2510.07058v1)**

> **作者:** Ori nizan; Oren Shrout; Ayellet Tal
>
> **摘要:** A concept may reflect either a concrete or abstract idea. Given an input image, this paper seeks to retrieve other images that share its central concepts, capturing aspects of the underlying narrative. This goes beyond conventional retrieval or clustering methods, which emphasize visual or semantic similarity. We formally define the problem, outline key requirements, and introduce appropriate evaluation metrics. We propose a novel approach grounded in two key observations: (1) While each neighbor in the embedding space typically shares at least one concept with the query, not all neighbors necessarily share the same concept with one another. (2) Modeling this neighborhood with a bimodal Gaussian distribution uncovers meaningful structure that facilitates concept identification. Qualitative, quantitative, and human evaluations confirm the effectiveness of our approach. See the package on PyPI: https://pypi.org/project/coret/
>
---
#### [new 023] Bayesian Modelling of Multi-Year Crop Type Classification Using Deep Neural Networks and Hidden Markov Models
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在解决多时相卫星图像中土地覆盖类型的时间不一致性问题。作者提出一种结合Transformer Encoder与隐马尔可夫模型的贝叶斯建模方法，通过建模年际间作物类型的时序关系，提高分类性能与时间连贯性。实验表明该方法在多作物类型分类中表现优异。**

- **链接: [http://arxiv.org/pdf/2510.07008v1](http://arxiv.org/pdf/2510.07008v1)**

> **作者:** Gianmarco Perantoni; Giulio Weikmann; Lorenzo Bruzzone
>
> **备注:** 5 pages, 1 figure, accepted conference paper at IEEE International Geoscience and Remote Sensing Symposium, 7-12 July 2024, Athens, Greece
>
> **摘要:** The temporal consistency of yearly land-cover maps is of great importance to model the evolution and change of the land cover over the years. In this paper, we focus the attention on a novel approach to classification of yearly satellite image time series (SITS) that combines deep learning with Bayesian modelling, using Hidden Markov Models (HMMs) integrated with Transformer Encoder (TE) based DNNs. The proposed approach aims to capture both i) intricate temporal correlations in yearly SITS and ii) specific patterns in multiyear crop type sequences. It leverages the cascade classification of an HMM layer built on top of the TE, discerning consistent yearly crop-type sequences. Validation on a multiyear crop type classification dataset spanning 47 crop types and six years of Sentinel-2 acquisitions demonstrates the importance of modelling temporal consistency in the predicted labels. HMMs enhance the overall performance and F1 scores, emphasising the effectiveness of the proposed approach.
>
---
#### [new 024] Adaptive Stain Normalization for Cross-Domain Medical Histology
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决染色和成像条件差异导致的域移问题。作者提出了一种可训练的染色归一化模型，基于Beer-Lambert定律和非负矩阵分解，提取染色不变的结构信息，提升跨域病理图像分析性能。**

- **链接: [http://arxiv.org/pdf/2510.06592v1](http://arxiv.org/pdf/2510.06592v1)**

> **作者:** Tianyue Xu; Yanlin Wu; Abhai K. Tripathi; Matthew M. Ippolito; Benjamin D. Haeffele
>
> **备注:** Accepted to the 28th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI 2025)
>
> **摘要:** Deep learning advances have revolutionized automated digital pathology analysis. However, differences in staining protocols and imaging conditions can introduce significant color variability. In deep learning, such color inconsistency often reduces performance when deploying models on data acquired under different conditions from the training data, a challenge known as domain shift. Many existing methods attempt to address this problem via color normalization but suffer from several notable drawbacks such as introducing artifacts or requiring careful choice of a template image for stain mapping. To address these limitations, we propose a trainable color normalization model that can be integrated with any backbone network for downstream tasks such as object detection and classification. Based on the physics of the imaging process per the Beer-Lambert law, our model architecture is derived via algorithmic unrolling of a nonnegative matrix factorization (NMF) model to extract stain-invariant structural information from the original pathology images, which serves as input for further processing. Experimentally, we evaluate the method on publicly available pathology datasets and an internally curated collection of malaria blood smears for cross-domain object detection and classification, where our method outperforms many state-of-the-art stain normalization methods. Our code is available at https://github.com/xutianyue/BeerLaNet.
>
---
#### [new 025] Limited-Angle Tomography Reconstruction via Projector Guided 3D Diffusion
- **分类: cs.CV**

- **简介: 该论文属于电子断层扫描重建任务，旨在解决有限角度下因缺失楔形数据导致的重建伪影问题。作者提出TEMDiff方法，利用3D扩散模型结合FIB-SEM数据训练，无需高质量TEM真实数据即可提升重建效果，并验证其在极窄角度下的适用性。**

- **链接: [http://arxiv.org/pdf/2510.06516v1](http://arxiv.org/pdf/2510.06516v1)**

> **作者:** Zhantao Deng; Mériem Er-Rafik; Anna Sushko; Cécile Hébert; Pascal Fua
>
> **备注:** 10 pages, 11 figures
>
> **摘要:** Limited-angle electron tomography aims to reconstruct 3D shapes from 2D projections of Transmission Electron Microscopy (TEM) within a restricted range and number of tilting angles, but it suffers from the missing-wedge problem that causes severe reconstruction artifacts. Deep learning approaches have shown promising results in alleviating these artifacts, yet they typically require large high-quality training datasets with known 3D ground truth which are difficult to obtain in electron microscopy. To address these challenges, we propose TEMDiff, a novel 3D diffusion-based iterative reconstruction framework. Our method is trained on readily available volumetric FIB-SEM data using a simulator that maps them to TEM tilt series, enabling the model to learn realistic structural priors without requiring clean TEM ground truth. By operating directly on 3D volumes, TEMDiff implicitly enforces consistency across slices without the need for additional regularization. On simulated electron tomography datasets with limited angular coverage, TEMDiff outperforms state-of-the-art methods in reconstruction quality. We further demonstrate that a trained TEMDiff model generalizes well to real-world TEM tilts obtained under different conditions and can recover accurate structures from tilt ranges as narrow as 8 degrees, with 2-degree increments, without any retraining or fine-tuning.
>
---
#### [new 026] Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods
- **分类: cs.CV**

- **简介: 该论文属于视觉token压缩评估任务，旨在解决现有基准不适用于压缩方法评估的问题。作者发现简单下采样优于复杂方法，并提出VTC-Bench框架，通过数据过滤机制去除噪声，实现更公平准确的评估。**

- **链接: [http://arxiv.org/pdf/2510.07143v1](http://arxiv.org/pdf/2510.07143v1)**

> **作者:** Chenfei Liao; Wensong Wang; Zichen Wen; Xu Zheng; Yiyu Wang; Haocong He; Yuanhuiyi Lyu; Lutao Jiang; Xin Zou; Yuqian Fu; Bin Ren; Linfeng Zhang; Xuming Hu
>
> **摘要:** Recent endeavors to accelerate inference in Multimodal Large Language Models (MLLMs) have primarily focused on visual token compression. The effectiveness of these methods is typically assessed by measuring the accuracy drop on established benchmarks, comparing model performance before and after compression. However, these benchmarks are originally designed to assess the perception and reasoning capabilities of MLLMs, rather than to evaluate compression techniques. As a result, directly applying them to visual token compression introduces a task mismatch. Strikingly, our investigation reveals that simple image downsampling consistently outperforms many advanced compression methods across multiple widely used benchmarks. Through extensive experiments, we make the following observations: (i) Current benchmarks are noisy for the visual token compression task. (ii) Down-sampling is able to serve as a data filter to evaluate the difficulty of samples in the visual token compression task. Motivated by these findings, we introduce VTC-Bench, an evaluation framework that incorporates a data filtering mechanism to denoise existing benchmarks, thereby enabling fairer and more accurate assessment of visual token compression methods. All data and code are available at https://github.com/Chenfei-Liao/VTC-Bench.
>
---
#### [new 027] Vision Transformer for Transient Noise Classification
- **分类: cs.CV; astro-ph.IM; cs.LG; gr-qc**

- **简介: 该论文属于机器学习与引力波数据分析任务，旨在解决LIGO数据中瞬态噪声分类问题。为应对新增的两类噪声，作者采用预训练视觉Transformer（ViT-B/32）模型，在结合Gravity Spy与O3a数据集上实现22+2类噪声的高效分类，准确率达92.26%。**

- **链接: [http://arxiv.org/pdf/2510.06273v1](http://arxiv.org/pdf/2510.06273v1)**

> **作者:** Divyansh Srivastava; Andrzej Niedzielski
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Transient noise (glitches) in LIGO data hinders the detection of gravitational waves (GW). The Gravity Spy project has categorized these noise events into various classes. With the O3 run, there is the inclusion of two additional noise classes and thus a need to train new models for effective classification. We aim to classify glitches in LIGO data into 22 existing classes from the first run plus 2 additional noise classes from O3a using the Vision Transformer (ViT) model. We train a pre-trained Vision Transformer (ViT-B/32) model on a combined dataset consisting of the Gravity Spy dataset with the additional two classes from the LIGO O3a run. We achieve a classification efficiency of 92.26%, demonstrating the potential of Vision Transformer to improve the accuracy of gravitational wave detection by effectively distinguishing transient noise. Key words: gravitational waves --vision transformer --machine learning
>
---
#### [new 028] Lung Infection Severity Prediction Using Transformers with Conditional TransMix Augmentation and Cross-Attention
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决肺部感染（如肺炎）的严重程度预测问题。作者提出了一种基于Transformer的新方法QCross-Att-PVT，并结合条件在线数据增强策略Conditional Online TransMix，提升模型在CT和X光图像上的预测准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.06887v1](http://arxiv.org/pdf/2510.06887v1)**

> **作者:** Bouthaina Slika; Fadi Dornaika; Fares Bougourzi; Karim Hammoudi
>
> **摘要:** Lung infections, particularly pneumonia, pose serious health risks that can escalate rapidly, especially during pandemics. Accurate AI-based severity prediction from medical imaging is essential to support timely clinical decisions and optimize patient outcomes. In this work, we present a novel method applicable to both CT scans and chest X-rays for assessing lung infection severity. Our contributions are twofold: (i) QCross-Att-PVT, a Transformer-based architecture that integrates parallel encoders, a cross-gated attention mechanism, and a feature aggregator to capture rich multi-scale features; and (ii) Conditional Online TransMix, a custom data augmentation strategy designed to address dataset imbalance by generating mixed-label image patches during training. Evaluated on two benchmark datasets, RALO CXR and Per-COVID-19 CT, our method consistently outperforms several state-of-the-art deep learning models. The results emphasize the critical role of data augmentation and gated attention in improving both robustness and predictive accuracy. This approach offers a reliable, adaptable tool to support clinical diagnosis, disease monitoring, and personalized treatment planning. The source code of this work is available at https://github.com/bouthainas/QCross-Att-PVT.
>
---
#### [new 029] Enhanced Self-Distillation Framework for Efficient Spiking Neural Network Training
- **分类: cs.CV**

- **简介: 该论文属于神经网络训练方法研究，旨在解决脉冲神经网络（SNN）训练效率低、资源消耗大的问题。作者提出了一种增强的自蒸馏框架，结合基于速率的反向传播，通过分离可靠与不可靠知识，优化模型结构，降低训练复杂度，同时保持高性能。实验验证了方法在多个数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.06254v1](http://arxiv.org/pdf/2510.06254v1)**

> **作者:** Xiaochen Zhao; Chengting Yu; Kairong Yu; Lei Liu; Aili Wang
>
> **摘要:** Spiking Neural Networks (SNNs) exhibit exceptional energy efficiency on neuromorphic hardware due to their sparse activation patterns. However, conventional training methods based on surrogate gradients and Backpropagation Through Time (BPTT) not only lag behind Artificial Neural Networks (ANNs) in performance, but also incur significant computational and memory overheads that grow linearly with the temporal dimension. To enable high-performance SNN training under limited computational resources, we propose an enhanced self-distillation framework, jointly optimized with rate-based backpropagation. Specifically, the firing rates of intermediate SNN layers are projected onto lightweight ANN branches, and high-quality knowledge generated by the model itself is used to optimize substructures through the ANN pathways. Unlike traditional self-distillation paradigms, we observe that low-quality self-generated knowledge may hinder convergence. To address this, we decouple the teacher signal into reliable and unreliable components, ensuring that only reliable knowledge is used to guide the optimization of the model. Extensive experiments on CIFAR-10, CIFAR-100, CIFAR10-DVS, and ImageNet demonstrate that our method reduces training complexity while achieving high-performance SNN training. Our code is available at https://github.com/Intelli-Chip-Lab/enhanced-self-distillation-framework-for-snn.
>
---
#### [new 030] Heptapod: Language Modeling on Visual Signals
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决现有自回归模型在图像生成效果不佳的问题。作者提出了Heptapod模型，采用因果注意力机制，进行二维分布预测，结合了自回归建模与自监督学习的优势，提升了图像生成质量，在ImageNet数据集上取得了更好的FID分数。**

- **链接: [http://arxiv.org/pdf/2510.06673v1](http://arxiv.org/pdf/2510.06673v1)**

> **作者:** Yongxin Zhu; Jiawei Chen; Yuanzhe Chen; Zhuo Chen; Dongya Jia; Jian Cong; Xiaobin Zhuang; Yuping Wang; Yuxuan Wang
>
> **摘要:** We introduce Heptapod, an image autoregressive model that adheres to the foundational principles of language modeling. Heptapod employs \textbf{causal attention}, \textbf{eliminates reliance on CFG}, and \textbf{eschews the trend of semantic tokenizers}. Our key innovation is \textit{next 2D distribution prediction}: a causal Transformer with reconstruction-focused visual tokenizer, learns to predict the distribution over the entire 2D spatial grid of images at each timestep. This learning objective unifies the sequential modeling of autoregressive framework with the holistic self-supervised learning of masked autoencoding, enabling the model to capture comprehensive image semantics via generative training. On the ImageNet generation benchmark, Heptapod achieves an FID of $2.70$, significantly outperforming previous causal autoregressive approaches. We hope our work inspires a principled rethinking of language modeling on visual signals and beyond.
>
---
#### [new 031] Efficient High-Resolution Image Editing with Hallucination-Aware Loss and Adaptive Tiling
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像编辑任务，旨在解决高分辨率图像编辑中的内存占用高和计算成本大的问题。论文提出了MobilePicasso系统，通过分阶段编辑、潜空间投影和自适应分块上采样方法，有效提升了编辑效率与质量，同时降低了内存消耗。**

- **链接: [http://arxiv.org/pdf/2510.06295v1](http://arxiv.org/pdf/2510.06295v1)**

> **作者:** Young D. Kwon; Abhinav Mehrotra; Malcolm Chadwick; Alberto Gil Ramos; Sourav Bhattacharya
>
> **备注:** Preprint. Under review
>
> **摘要:** High-resolution (4K) image-to-image synthesis has become increasingly important for mobile applications. Existing diffusion models for image editing face significant challenges, in terms of memory and image quality, when deployed on resource-constrained devices. In this paper, we present MobilePicasso, a novel system that enables efficient image editing at high resolutions, while minimising computational cost and memory usage. MobilePicasso comprises three stages: (i) performing image editing at a standard resolution with hallucination-aware loss, (ii) applying latent projection to overcome going to the pixel space, and (iii) upscaling the edited image latent to a higher resolution with adaptive context-preserving tiling. Our user study with 46 participants reveals that MobilePicasso not only improves image quality by 18-48% but reduces hallucinations by 14-51% over existing methods. MobilePicasso demonstrates significantly lower latency, e.g., up to 55.8$\times$ speed-up, yet with a small increase in runtime memory, e.g., a mere 9% increase over prior work. Surprisingly, the on-device runtime of MobilePicasso is observed to be faster than a server-based high-resolution image editing model running on an A100 GPU.
>
---
#### [new 032] Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决现有生成模型因使用VAE压缩导致深度图边缘和细节出现“飞行像素”的问题。论文提出“Pixel-Perfect Depth”模型，直接在像素空间进行扩散生成，避免VAE引入的伪影，并引入语义引导的扩散变压器（SP-DiT）和级联设计，提升效率与精度。**

- **链接: [http://arxiv.org/pdf/2510.07316v1](http://arxiv.org/pdf/2510.07316v1)**

> **作者:** Gangwei Xu; Haotong Lin; Hongcheng Luo; Xianqi Wang; Jingfeng Yao; Lianghui Zhu; Yuechuan Pu; Cheng Chi; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Sida Peng; Xin Yang
>
> **备注:** NeurIPS 2025. Project page: https://pixel-perfect-depth.github.io/
>
> **摘要:** This paper presents Pixel-Perfect Depth, a monocular depth estimation model based on pixel-space diffusion generation that produces high-quality, flying-pixel-free point clouds from estimated depth maps. Current generative depth estimation models fine-tune Stable Diffusion and achieve impressive performance. However, they require a VAE to compress depth maps into latent space, which inevitably introduces \textit{flying pixels} at edges and details. Our model addresses this challenge by directly performing diffusion generation in the pixel space, avoiding VAE-induced artifacts. To overcome the high complexity associated with pixel-space generation, we introduce two novel designs: 1) Semantics-Prompted Diffusion Transformers (SP-DiT), which incorporate semantic representations from vision foundation models into DiT to prompt the diffusion process, thereby preserving global semantic consistency while enhancing fine-grained visual details; and 2) Cascade DiT Design that progressively increases the number of tokens to further enhance efficiency and accuracy. Our model achieves the best performance among all published generative models across five benchmarks, and significantly outperforms all other models in edge-aware point cloud evaluation.
>
---
#### [new 033] Text2Interact: High-Fidelity and Diverse Text-to-Two-Person Interaction Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到交互生成任务，旨在根据文本生成高质量的双人互动动作。主要解决双人互动数据不足及文本条件建模不够细粒度的问题。论文提出了Text2Interact框架，包含InterCompose（合成双人互动数据）和InterActor（文本驱动的互动建模），提升了生成动作的多样性、真实性和文本一致性。**

- **链接: [http://arxiv.org/pdf/2510.06504v1](http://arxiv.org/pdf/2510.06504v1)**

> **作者:** Qingxuan Wu; Zhiyang Dou; Chuan Guo; Yiming Huang; Qiao Feng; Bing Zhou; Jian Wang; Lingjie Liu
>
> **摘要:** Modeling human-human interactions from text remains challenging because it requires not only realistic individual dynamics but also precise, text-consistent spatiotemporal coupling between agents. Currently, progress is hindered by 1) limited two-person training data, inadequate to capture the diverse intricacies of two-person interactions; and 2) insufficiently fine-grained text-to-interaction modeling, where language conditioning collapses rich, structured prompts into a single sentence embedding. To address these limitations, we propose our Text2Interact framework, designed to generate realistic, text-aligned human-human interactions through a scalable high-fidelity interaction data synthesizer and an effective spatiotemporal coordination pipeline. First, we present InterCompose, a scalable synthesis-by-composition pipeline that aligns LLM-generated interaction descriptions with strong single-person motion priors. Given a prompt and a motion for an agent, InterCompose retrieves candidate single-person motions, trains a conditional reaction generator for another agent, and uses a neural motion evaluator to filter weak or misaligned samples-expanding interaction coverage without extra capture. Second, we propose InterActor, a text-to-interaction model with word-level conditioning that preserves token-level cues (initiation, response, contact ordering) and an adaptive interaction loss that emphasizes contextually relevant inter-person joint pairs, improving coupling and physical plausibility for fine-grained interaction modeling. Extensive experiments show consistent gains in motion diversity, fidelity, and generalization, including out-of-distribution scenarios and user studies. We will release code and models to facilitate reproducibility.
>
---
#### [new 034] OBS-Diff: Accurate Pruning For Diffusion Models in One-Shot
- **分类: cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决扩散模型因计算成本高而难以部署的问题。作者提出了OBS-Diff，一种适用于扩散模型的一次性剪枝方法，通过改进经典OBS算法、引入时间步感知的Hessian矩阵和分组剪枝策略，实现了高效压缩与推理加速。**

- **链接: [http://arxiv.org/pdf/2510.06751v1](http://arxiv.org/pdf/2510.06751v1)**

> **作者:** Junhan Zhu; Hesong Wang; Mingluo Su; Zefang Wang; Huan Wang
>
> **摘要:** Large-scale text-to-image diffusion models, while powerful, suffer from prohibitive computational cost. Existing one-shot network pruning methods can hardly be directly applied to them due to the iterative denoising nature of diffusion models. To bridge the gap, this paper presents OBS-Diff, a novel one-shot pruning framework that enables accurate and training-free compression of large-scale text-to-image diffusion models. Specifically, (i) OBS-Diff revitalizes the classic Optimal Brain Surgeon (OBS), adapting it to the complex architectures of modern diffusion models and supporting diverse pruning granularity, including unstructured, N:M semi-structured, and structured (MHA heads and FFN neurons) sparsity; (ii) To align the pruning criteria with the iterative dynamics of the diffusion process, by examining the problem from an error-accumulation perspective, we propose a novel timestep-aware Hessian construction that incorporates a logarithmic-decrease weighting scheme, assigning greater importance to earlier timesteps to mitigate potential error accumulation; (iii) Furthermore, a computationally efficient group-wise sequential pruning strategy is proposed to amortize the expensive calibration process. Extensive experiments show that OBS-Diff achieves state-of-the-art one-shot pruning for diffusion models, delivering inference acceleration with minimal degradation in visual quality.
>
---
#### [new 035] Quantum-enhanced Computer Vision: Going Beyond Classical Algorithms
- **分类: cs.CV**

- **简介: 该论文属于综述任务，旨在介绍量子增强计算机视觉（QeCV）这一新兴领域。它探讨了如何利用量子计算提升计算机视觉性能，特别是在处理复杂问题时超越经典算法的能力。论文总结了QeCV的研究现状，介绍了量子计算的基本原理、相关工具及编程方法，并讨论了未来挑战和社会影响。**

- **链接: [http://arxiv.org/pdf/2510.07317v1](http://arxiv.org/pdf/2510.07317v1)**

> **作者:** Natacha Kuete Meli; Shuteng Wang; Marcel Seelbach Benkner; Michele Sasdelli; Tat-Jun Chin; Tolga Birdal; Michael Moeller; Vladislav Golyanik
>
> **备注:** 44 pages, 23 figures and 6 tables
>
> **摘要:** Quantum-enhanced Computer Vision (QeCV) is a new research field at the intersection of computer vision, optimisation theory, machine learning and quantum computing. It has high potential to transform how visual signals are processed and interpreted with the help of quantum computing that leverages quantum-mechanical effects in computations inaccessible to classical (i.e. non-quantum) computers. In scenarios where existing non-quantum methods cannot find a solution in a reasonable time or compute only approximate solutions, quantum computers can provide, among others, advantages in terms of better time scalability for multiple problem classes. Parametrised quantum circuits can also become, in the long term, a considerable alternative to classical neural networks in computer vision. However, specialised and fundamentally new algorithms must be developed to enable compatibility with quantum hardware and unveil the potential of quantum computational paradigms in computer vision. This survey contributes to the existing literature on QeCV with a holistic review of this research field. It is designed as a quantum computing reference for the computer vision community, targeting computer vision students, scientists and readers with related backgrounds who want to familiarise themselves with QeCV. We provide a comprehensive introduction to QeCV, its specifics, and methodologies for formulations compatible with quantum hardware and QeCV methods, leveraging two main quantum computational paradigms, i.e. gate-based quantum computing and quantum annealing. We elaborate on the operational principles of quantum computers and the available tools to access, program and simulate them in the context of QeCV. Finally, we review existing quantum computing tools and learning materials and discuss aspects related to publishing and reviewing QeCV papers, open challenges and potential social implications.
>
---
#### [new 036] Semantic Segmentation Algorithm Based on Light Field and LiDAR Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态语义分割任务，旨在解决复杂场景下（如遮挡）目标分割不准确的问题。作者提出了一个融合光场图像与激光雷达点云的新数据集及Mlpfseg网络，通过特征补全和深度感知模块提升分割效果。**

- **链接: [http://arxiv.org/pdf/2510.06687v1](http://arxiv.org/pdf/2510.06687v1)**

> **作者:** Jie Luo; Yuxuan Jiang; Xin Jin; Mingyu Liu; Yihui Fan
>
> **摘要:** Semantic segmentation serves as a cornerstone of scene understanding in autonomous driving but continues to face significant challenges under complex conditions such as occlusion. Light field and LiDAR modalities provide complementary visual and spatial cues that are beneficial for robust perception; however, their effective integration is hindered by limited viewpoint diversity and inherent modality discrepancies. To address these challenges, the first multimodal semantic segmentation dataset integrating light field data and point cloud data is proposed. Based on this dataset, we proposed a multi-modal light field point-cloud fusion segmentation network(Mlpfseg), incorporating feature completion and depth perception to segment both camera images and LiDAR point clouds simultaneously. The feature completion module addresses the density mismatch between point clouds and image pixels by performing differential reconstruction of point-cloud feature maps, enhancing the fusion of these modalities. The depth perception module improves the segmentation of occluded objects by reinforcing attention scores for better occlusion awareness. Our method outperforms image-only segmentation by 1.71 Mean Intersection over Union(mIoU) and point cloud-only segmentation by 2.38 mIoU, demonstrating its effectiveness.
>
---
#### [new 037] TTRV: Test-Time Reinforcement Learning for Vision Language Models
- **分类: cs.CV**

- **简介: 论文提出TTRV，一种无需标注数据的测试时强化学习方法，用于提升视觉语言模型的理解能力。其任务为视觉语言理解，旨在解决传统强化学习依赖标注数据的问题。通过改进GRPO框架，设计基于输出频率的奖励机制，并控制输出多样性，实现模型推理时自适应优化。实验表明该方法在多个视觉任务上显著提升性能，优于GPT-4o等强模型。**

- **链接: [http://arxiv.org/pdf/2510.06783v1](http://arxiv.org/pdf/2510.06783v1)**

> **作者:** Akshit Singh; Shyam Marjit; Wei Lin; Paul Gavrikov; Serena Yeung-Levy; Hilde Kuehne; Rogerio Feris; Sivan Doveh; James Glass; M. Jehanzeb Mirza
>
> **摘要:** Existing methods for extracting reward signals in Reinforcement Learning typically rely on labeled data and dedicated training splits, a setup that contrasts with how humans learn directly from their environment. In this work, we propose TTRV to enhance vision language understanding by adapting the model on the fly at inference time, without the need for any labeled data. Concretely, we enhance the Group Relative Policy Optimization (GRPO) framework by designing rewards based on the frequency of the base model's output, while inferring on each test sample multiple times. Further, we also propose to control the diversity of the model's output by simultaneously rewarding the model for obtaining low entropy of the output empirical distribution. Our approach delivers consistent gains across both object recognition and visual question answering (VQA), with improvements of up to 52.4% and 29.8%, respectively, and average boosts of 24.6% and 10.0% across 16 datasets.Remarkably, on image recognition, TTRV applied to InternVL 8B surpasses GPT-4o by an average of 2.3% over 8 benchmarks, while remaining highly competitive on VQA, demonstrating that test-time reinforcement learning can match or exceed the strongest proprietary models. Finally, we find many interesting properties of test-time RL for VLMs: for example, even in extremely data-constrained scenarios, where adaptation is performed on a single randomly chosen unlabeled test example, TTRV still yields non-trivial improvements of up to 5.5% in recognition tasks.
>
---
#### [new 038] SpecGuard: Spectral Projection-based Advanced Invisible Watermarking
- **分类: cs.CV**

- **简介: 该论文属于图像水印任务，旨在解决现有方法在抗干扰和鲁棒性方面的不足。论文提出了SpecGuard，通过频域转换和频谱投影技术，在图像高频部分嵌入水印，提高了水印的不可见性和抗攻击能力，并通过实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.07302v1](http://arxiv.org/pdf/2510.07302v1)**

> **作者:** Inzamamul Alam; Md Tanvir Islam; Khan Muhammad; Simon S. Woo
>
> **备注:** ICCV 2025 Accepted Paper
>
> **摘要:** Watermarking embeds imperceptible patterns into images for authenticity verification. However, existing methods often lack robustness against various transformations primarily including distortions, image regeneration, and adversarial perturbation, creating real-world challenges. In this work, we introduce SpecGuard, a novel watermarking approach for robust and invisible image watermarking. Unlike prior approaches, we embed the message inside hidden convolution layers by converting from the spatial domain to the frequency domain using spectral projection of a higher frequency band that is decomposed by wavelet projection. Spectral projection employs Fast Fourier Transform approximation to transform spatial data into the frequency domain efficiently. In the encoding phase, a strength factor enhances resilience against diverse attacks, including adversarial, geometric, and regeneration-based distortions, ensuring the preservation of copyrighted information. Meanwhile, the decoder leverages Parseval's theorem to effectively learn and extract the watermark pattern, enabling accurate retrieval under challenging transformations. We evaluate the proposed SpecGuard based on the embedded watermark's invisibility, capacity, and robustness. Comprehensive experiments demonstrate the proposed SpecGuard outperforms the state-of-the-art models. To ensure reproducibility, the full code is released on \href{https://github.com/inzamamulDU/SpecGuard_ICCV_2025}{\textcolor{blue}{\textbf{GitHub}}}.
>
---
#### [new 039] Milestone Determination for Autonomous Railway Operation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉与铁路自动化任务，旨在解决自动驾驶铁路中缺乏高质量时序数据的问题。通过引入“里程碑”概念，利用路线关键点构建规则模型，简化视觉学习过程，提升铁路自动化系统的安全性和效率。**

- **链接: [http://arxiv.org/pdf/2510.06229v1](http://arxiv.org/pdf/2510.06229v1)**

> **作者:** Josh Hunter; John McDermid; Simon Burton; Poppy Fynes; Mia Dempster
>
> **备注:** Paper submitted and partially accepted to ICART 2025, paper is 8 pages and has 1 figure, 2 tables
>
> **摘要:** In the field of railway automation, one of the key challenges has been the development of effective computer vision systems due to the limited availability of high-quality, sequential data. Traditional datasets are restricted in scope, lacking the spatio temporal context necessary for real-time decision-making, while alternative solutions introduce issues related to realism and applicability. By focusing on route-specific, contextually relevant cues, we can generate rich, sequential datasets that align more closely with real-world operational logic. The concept of milestone determination allows for the development of targeted, rule-based models that simplify the learning process by eliminating the need for generalized recognition of dynamic components, focusing instead on the critical decision points along a route. We argue that this approach provides a practical framework for training vision agents in controlled, predictable environments, facilitating safer and more efficient machine learning systems for railway automation.
>
---
#### [new 040] OBJVanish: Physically Realizable Text-to-3D Adv. Generation of LiDAR-Invisible Objects
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶安全任务，旨在解决LiDAR 3D检测器对物理世界中隐藏物体的漏洞问题。作者提出OBJVanish方法，通过文本生成不可被LiDAR检测到的3D模型，揭示检测系统漏洞。论文结合仿真与物理实验，验证方法有效性，强调安全关键系统中的潜在风险。**

- **链接: [http://arxiv.org/pdf/2510.06952v1](http://arxiv.org/pdf/2510.06952v1)**

> **作者:** Bing Li; Wuqi Wang; Yanan Zhang; Jingzheng Li; Haigen Min; Wei Feng; Xingyu Zhao; Jie Zhang; Qing Guo
>
> **摘要:** LiDAR-based 3D object detectors are fundamental to autonomous driving, where failing to detect objects poses severe safety risks. Developing effective 3D adversarial attacks is essential for thoroughly testing these detection systems and exposing their vulnerabilities before real-world deployment. However, existing adversarial attacks that add optimized perturbations to 3D points have two critical limitations: they rarely cause complete object disappearance and prove difficult to implement in physical environments. We introduce the text-to-3D adversarial generation method, a novel approach enabling physically realizable attacks that can generate 3D models of objects truly invisible to LiDAR detectors and be easily realized in the real world. Specifically, we present the first empirical study that systematically investigates the factors influencing detection vulnerability by manipulating the topology, connectivity, and intensity of individual pedestrian 3D models and combining pedestrians with multiple objects within the CARLA simulation environment. Building on the insights, we propose the physically-informed text-to-3D adversarial generation (Phy3DAdvGen) that systematically optimizes text prompts by iteratively refining verbs, objects, and poses to produce LiDAR-invisible pedestrians. To ensure physical realizability, we construct a comprehensive object pool containing 13 3D models of real objects and constrain Phy3DAdvGen to generate 3D objects based on combinations of objects in this set. Extensive experiments demonstrate that our approach can generate 3D pedestrians that evade six state-of-the-art (SOTA) LiDAR 3D detectors in both CARLA simulation and physical environments, thereby highlighting vulnerabilities in safety-critical applications.
>
---
#### [new 041] From Captions to Keyframes: Efficient Video Summarization via Caption- and Context-Aware Frame Scoring
- **分类: cs.CV**

- **简介: 该论文属于视频摘要任务，旨在解决长视频处理中的效率问题。通过结合视觉信息与文本描述，提出KeyScore和STACFP方法，实现高效帧选择，在减少计算量的同时保持语义与上下文完整性。**

- **链接: [http://arxiv.org/pdf/2510.06509v1](http://arxiv.org/pdf/2510.06509v1)**

> **作者:** Shih-Yao Lin; Sibendu Paul; Caren Chen
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Efficient video-language understanding requires selecting a small set of frames that retain semantic and contextual information from long videos. We propose KeyScore, a multimodal frame scoring framework that jointly leverages captions and visual context to estimate frame-level importance. By combining semantic similarity, temporal diversity, and contextual drop impact, KeyScore identifies the most informative frames for downstream tasks such as retrieval, captioning, and video-language reasoning. To complement KeyScore, we introduce STACFP (Spatio-Temporal Adaptive Clustering for Frame Proposals), which generates compact and diverse frame candidates for long-form videos. Together, these modules achieve up to 99\% frame reduction compared to full-frame inference and substantially outperform standard 8-frame encoders on MSRVTT, MSVD, and DiDeMo. Our results demonstrate that emphasizing multimodal alignment between visual and textual signals enables scalable, efficient, and caption-grounded video understanding -- without explicit video summarization.
>
---
#### [new 042] DeRainMamba: A Frequency-Aware State Space Model with Detail Enhancement for Image Deraining
- **分类: cs.CV**

- **简介: 该论文属于图像去雨任务，旨在解决雨 streaks 降低图像质量的问题。作者提出了 DeRainMamba 模型，结合频域感知模块（FASSM）和多方向感知卷积（MDPConv），有效去除雨 streaks 同时保留图像细节，提升了去雨效果和效率。**

- **链接: [http://arxiv.org/pdf/2510.06746v1](http://arxiv.org/pdf/2510.06746v1)**

> **作者:** Zhiliang Zhu; Tao Zeng; Tao Yang; Guoliang Luo; Jiyong Zeng
>
> **备注:** accepted by IEEE SPL
>
> **摘要:** Image deraining is crucial for improving visual quality and supporting reliable downstream vision tasks. Although Mamba-based models provide efficient sequence modeling, their limited ability to capture fine-grained details and lack of frequency-domain awareness restrict further improvements. To address these issues, we propose DeRainMamba, which integrates a Frequency-Aware State-Space Module (FASSM) and Multi-Directional Perception Convolution (MDPConv). FASSM leverages Fourier transform to distinguish rain streaks from high-frequency image details, balancing rain removal and detail preservation. MDPConv further restores local structures by capturing anisotropic gradient features and efficiently fusing multiple convolution branches. Extensive experiments on four public benchmarks demonstrate that DeRainMamba consistently outperforms state-of-the-art methods in PSNR and SSIM, while requiring fewer parameters and lower computational costs. These results validate the effectiveness of combining frequency-domain modeling and spatial detail enhancement within a state-space framework for single image deraining.
>
---
#### [new 043] ChainMPQ: Interleaved Text-Image Reasoning Chains for Mitigating Relation Hallucinations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型任务，旨在缓解模型在关系推理中的幻觉问题。通过构建基于关键词的多视角问题链，结合图文记忆提升关系推理能力，形成ChainMPQ方法，有效减少关系幻觉。**

- **链接: [http://arxiv.org/pdf/2510.06292v1](http://arxiv.org/pdf/2510.06292v1)**

> **作者:** Yike Wu; Yiwei Wang; Yujun Cai
>
> **摘要:** While Large Vision-Language Models (LVLMs) achieve strong performance in multimodal tasks, hallucinations continue to hinder their reliability. Among the three categories of hallucinations, which include object, attribute, and relation, relation hallucinations account for the largest proportion but have received the least attention. To address this issue, we propose ChainMPQ (Multi-Perspective Questions guided Interleaved Chain of Image and Text), a training-free method that improves relational inference in LVLMs by utilizing accumulated textual and visual memories. ChainMPQ first extracts subject and object keywords from the question to enhance the corresponding image regions. It then constructs multi-perspective questions that focus on the three core components of a relationship: the subject, the object, and the relation that links them. These questions are sequentially input to the model, with textual and visual memories from earlier steps providing supporting context for subsequent ones, thereby forming an interleaved chain of images and text that guides progressive relational reasoning. Experiments on multiple LVLMs and benchmarks show that ChainMPQ substantially reduces relation hallucinations, while ablation studies further validate the effectiveness of its three core modules.
>
---
#### [new 044] VA-Adapter: Adapting Ultrasound Foundation Model to Echocardiography Probe Guidance
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决心脏超声图像获取质量低的问题。通过设计参数高效的视觉-动作适配器（VA-Adapter），使预训练的超声基础模型能学习探头调整策略，提升对初级超声师的实时操作指导性能。**

- **链接: [http://arxiv.org/pdf/2510.06809v1](http://arxiv.org/pdf/2510.06809v1)**

> **作者:** Teng Wang; Haojun Jiang; Yuxuan Wang; Zhenguo Sun; Shiji Song; Gao Huang
>
> **摘要:** Echocardiography is a critical tool for detecting heart diseases. Recently, ultrasound foundation models have demonstrated remarkable capabilities in cardiac ultrasound image analysis. However, obtaining high-quality ultrasound images is a prerequisite for accurate diagnosis. Due to the exceptionally high operational difficulty of cardiac ultrasound, there is a shortage of highly skilled personnel, which hinders patients from receiving timely examination services. In this paper, we aim to adapt the medical knowledge learned by foundation models from vast datasets to the probe guidance task, which is designed to provide real-time operational recommendations for junior sonographers to acquire high-quality ultrasound images. Moreover, inspired by the practice where experts optimize action decisions based on past explorations, we meticulously design a parameter-efficient Vision-Action Adapter (VA-Adapter) to enable foundation model's image encoder to encode vision-action sequences, thereby enhancing guidance performance. With built-in sequential reasoning capabilities in a compact design, the VA-Adapter enables a pre-trained ultrasound foundation model to learn precise probe adjustment strategies by fine-tuning only a small subset of parameters. Extensive experiments demonstrate that the VA-Adapter can surpass strong probe guidance models. Our code will be released after acceptance.
>
---
#### [new 045] Generating Surface for Text-to-3D using 2D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本生成3D建模任务，旨在解决复杂几何形状的高质量3D内容生成问题。作者提出DirectGaussian方法，利用2D高斯点渲染和多视角法线、纹理先验，结合曲率约束优化，实现高保真、多样化的3D对象表面生成。**

- **链接: [http://arxiv.org/pdf/2510.06967v1](http://arxiv.org/pdf/2510.06967v1)**

> **作者:** Huanning Dong; Fan Li; Ping Kuang; Jianwen Min
>
> **摘要:** Recent advancements in Text-to-3D modeling have shown significant potential for the creation of 3D content. However, due to the complex geometric shapes of objects in the natural world, generating 3D content remains a challenging task. Current methods either leverage 2D diffusion priors to recover 3D geometry, or train the model directly based on specific 3D representations. In this paper, we propose a novel method named DirectGaussian, which focuses on generating the surfaces of 3D objects represented by surfels. In DirectGaussian, we utilize conditional text generation models and the surface of a 3D object is rendered by 2D Gaussian splatting with multi-view normal and texture priors. For multi-view geometric consistency problems, DirectGaussian incorporates curvature constraints on the generated surface during optimization process. Through extensive experiments, we demonstrate that our framework is capable of achieving diverse and high-fidelity 3D content creation.
>
---
#### [new 046] Through the Perspective of LiDAR: A Feature-Enriched and Uncertainty-Aware Annotation Pipeline for Terrestrial Point Cloud Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云语义分割任务，旨在解决地面激光扫描数据手动标注成本高的问题。作者提出了一种半自动化、结合不确定性评估的标注流程，通过球面投影、特征增强、集成学习与可视化工具，减少标注工作量并保持精度。同时构建了Mangrove3D数据集，并验证了方法在多个数据集上的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.06582v1](http://arxiv.org/pdf/2510.06582v1)**

> **作者:** Fei Zhang; Rob Chancia; Josie Clapp; Amirhossein Hassanzadeh; Dimah Dera; Richard MacKenzie; Jan van Aardt
>
> **摘要:** Accurate semantic segmentation of terrestrial laser scanning (TLS) point clouds is limited by costly manual annotation. We propose a semi-automated, uncertainty-aware pipeline that integrates spherical projection, feature enrichment, ensemble learning, and targeted annotation to reduce labeling effort, while sustaining high accuracy. Our approach projects 3D points to a 2D spherical grid, enriches pixels with multi-source features, and trains an ensemble of segmentation networks to produce pseudo-labels and uncertainty maps, the latter guiding annotation of ambiguous regions. The 2D outputs are back-projected to 3D, yielding densely annotated point clouds supported by a three-tier visualization suite (2D feature maps, 3D colorized point clouds, and compact virtual spheres) for rapid triage and reviewer guidance. Using this pipeline, we build Mangrove3D, a semantic segmentation TLS dataset for mangrove forests. We further evaluate data efficiency and feature importance to address two key questions: (1) how much annotated data are needed and (2) which features matter most. Results show that performance saturates after ~12 annotated scans, geometric features contribute the most, and compact nine-channel stacks capture nearly all discriminative power, with the mean Intersection over Union (mIoU) plateauing at around 0.76. Finally, we confirm the generalization of our feature-enrichment strategy through cross-dataset tests on ForestSemantic and Semantic3D. Our contributions include: (i) a robust, uncertainty-aware TLS annotation pipeline with visualization tools; (ii) the Mangrove3D dataset; and (iii) empirical guidance on data efficiency and feature importance, thus enabling scalable, high-quality segmentation of TLS point clouds for ecological monitoring and beyond. The dataset and processing scripts are publicly available at https://fz-rit.github.io/through-the-lidars-eye/.
>
---
#### [new 047] Graph Conditioned Diffusion for Controllable Histopathology Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像生成任务，旨在解决现有扩散模型在可控生成医学图像（如组织病理学图像）时缺乏语义结构控制的问题。作者提出基于图的扩散模型，利用图结构表示图像中的关键对象及其关系，通过Transformer集成到扩散模型中，实现对生成内容的精细控制，并验证了生成图像在分割任务中的可用性。**

- **链接: [http://arxiv.org/pdf/2510.07129v1](http://arxiv.org/pdf/2510.07129v1)**

> **作者:** Sarah Cechnicka; Matthew Baugh; Weitong Zhang; Mischa Dombrowski; Zhe Li; Johannes C. Paetzold; Candice Roufosse; Bernhard Kainz
>
> **摘要:** Recent advances in Diffusion Probabilistic Models (DPMs) have set new standards in high-quality image synthesis. Yet, controlled generation remains challenging, particularly in sensitive areas such as medical imaging. Medical images feature inherent structure such as consistent spatial arrangement, shape or texture, all of which are critical for diagnosis. However, existing DPMs operate in noisy latent spaces that lack semantic structure and strong priors, making it difficult to ensure meaningful control over generated content. To address this, we propose graph-based object-level representations for Graph-Conditioned-Diffusion. Our approach generates graph nodes corresponding to each major structure in the image, encapsulating their individual features and relationships. These graph representations are processed by a transformer module and integrated into a diffusion model via the text-conditioning mechanism, enabling fine-grained control over generation. We evaluate this approach using a real-world histopathology use case, demonstrating that our generated data can reliably substitute for annotated patient data in downstream segmentation tasks. The code is available here.
>
---
#### [new 048] Ming-UniVision: Joint Image Understanding and Generation with a Unified Continuous Tokenizer
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言统一建模任务，旨在解决视觉理解与生成中离散潜在空间带来的语义表达受限问题。作者提出MingTok，一种基于连续潜在空间的视觉分词器，并构建了Ming-UniVision模型，通过统一的自回归预测框架，实现视觉理解和生成任务的统一建模，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.06590v1](http://arxiv.org/pdf/2510.06590v1)**

> **作者:** Ziyuan Huang; DanDan Zheng; Cheng Zou; Rui Liu; Xiaolong Wang; Kaixiang Ji; Weilong Chai; Jianxin Sun; Libin Wang; Yongjie Lv; Taozhi Huang; Jiajia Liu; Qingpei Guo; Ming Yang; Jingdong Chen; Jun Zhou
>
> **备注:** Code released at https://github.com/inclusionAI/Ming-UniVision
>
> **摘要:** Visual tokenization remains a core challenge in unifying visual understanding and generation within the autoregressive paradigm. Existing methods typically employ tokenizers in discrete latent spaces to align with the tokens from large language models, where the quantization errors can limit semantic expressiveness and degrade the capability of vision-language understanding. To address this, we introduce MingTok, a new family of visual tokenizers with a continuous latent space, for unified autoregressive generation and understanding. While understanding tasks favor discriminative high-dimensional features, generation tasks prefer compact low-level codes. Thus, to reconcile these competing demands, MingTok adopts a three-stage sequential architecture involving low-level encoding, semantic expansion, and visual reconstruction. Built on top of it, Ming-UniVision eliminates the need for task-specific visual representations, and unifies diverse vision-language tasks under a single autoregrsssive prediction paradigm. By formulating both understanding and generation as next-token prediction in a shared continuous space, it seamlessly supports multi-round, in-context tasks such as iterative understanding, generation and editing. Empirically, we find that using a unified continuous visual representation reconciles the competing requirements on the tokenizers by the understanding and generation tasks, thereby leading to state-of-the-art level performance across both domains. We hope our findings will facilitate unified visual tokenization in the continuous domain. Inference code and model weights are released to benefit community.
>
---
#### [new 049] IAR2: Improving Autoregressive Visual Generation with Semantic-Detail Associated Token Prediction
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决视觉自回归模型生成图像时忽略视觉数据结构特性的问题。论文提出IAR2框架，采用语义-细节双码本和分层预测策略，提升生成质量与计算效率，实现了ImageNet上1.50的FID新纪录。**

- **链接: [http://arxiv.org/pdf/2510.06928v1](http://arxiv.org/pdf/2510.06928v1)**

> **作者:** Ran Yi; Teng Hu; Zihan Su; Lizhuang Ma
>
> **摘要:** Autoregressive models have emerged as a powerful paradigm for visual content creation, but often overlook the intrinsic structural properties of visual data. Our prior work, IAR, initiated a direction to address this by reorganizing the visual codebook based on embedding similarity, thereby improving generation robustness. However, it is constrained by the rigidity of pre-trained codebooks and the inaccuracies of hard, uniform clustering. To overcome these limitations, we propose IAR2, an advanced autoregressive framework that enables a hierarchical semantic-detail synthesis process. At the core of IAR2 is a novel Semantic-Detail Associated Dual Codebook, which decouples image representations into a semantic codebook for global semantic information and a detail codebook for fine-grained refinements. It expands the quantization capacity from a linear to a polynomial scale, significantly enhancing expressiveness. To accommodate this dual representation, we propose a Semantic-Detail Autoregressive Prediction scheme coupled with a Local-Context Enhanced Autoregressive Head, which performs hierarchical prediction-first the semantic token, then the detail token-while leveraging a local context window to enhance spatial coherence. Furthermore, for conditional generation, we introduce a Progressive Attention-Guided Adaptive CFG mechanism that dynamically modulates the guidance scale for each token based on its relevance to the condition and its temporal position in the generation sequence, improving conditional alignment without sacrificing realism. Extensive experiments demonstrate that IAR2 sets a new state-of-the-art for autoregressive image generation, achieving a FID of 1.50 on ImageNet. Our model not only surpasses previous methods in performance but also demonstrates superior computational efficiency, highlighting the effectiveness of our structured, coarse-to-fine generation strategy.
>
---
#### [new 050] No MoCap Needed: Post-Training Motion Diffusion Models with Reinforcement Learning using Only Textual Prompts
- **分类: cs.CV**

- **简介: 该论文属于文本驱动人体动作生成任务，旨在解决无需动作捕捉数据的动作迁移问题。作者提出一种基于强化学习的微调框架，利用预训练文本-动作检索网络作为奖励信号，通过去噪扩散策略优化，实现仅用文本提示调整预训练扩散模型，有效提升生成动作的质量与多样性，同时保持原始分布性能。方法在多个数据集和模型架构上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2510.06988v1](http://arxiv.org/pdf/2510.06988v1)**

> **作者:** Girolamo Macaluso; Lorenzo Mandelli; Mirko Bicchierai; Stefano Berretti; Andrew D. Bagdanov
>
> **摘要:** Diffusion models have recently advanced human motion generation, producing realistic and diverse animations from textual prompts. However, adapting these models to unseen actions or styles typically requires additional motion capture data and full retraining, which is costly and difficult to scale. We propose a post-training framework based on Reinforcement Learning that fine-tunes pretrained motion diffusion models using only textual prompts, without requiring any motion ground truth. Our approach employs a pretrained text-motion retrieval network as a reward signal and optimizes the diffusion policy with Denoising Diffusion Policy Optimization, effectively shifting the model's generative distribution toward the target domain without relying on paired motion data. We evaluate our method on cross-dataset adaptation and leave-one-out motion experiments using the HumanML3D and KIT-ML datasets across both latent- and joint-space diffusion architectures. Results from quantitative metrics and user studies show that our approach consistently improves the quality and diversity of generated motions, while preserving performance on the original distribution. Our approach is a flexible, data-efficient, and privacy-preserving solution for motion adaptation.
>
---
#### [new 051] EigenScore: OOD Detection using Covariance in Diffusion Models
- **分类: cs.CV**

- **简介: 论文提出EigenScore，用于扩散模型中的分布外（OOD）检测。通过利用后验协方差的特征值谱，捕捉分布偏移信号，实现更可靠的OOD检测。方法基于扩散模型，通过前向计算估计主特征值，提升检测性能，尤其在近似OOD场景下表现稳健。**

- **链接: [http://arxiv.org/pdf/2510.07206v1](http://arxiv.org/pdf/2510.07206v1)**

> **作者:** Shirin Shoushtari; Yi Wang; Xiao Shi; M. Salman Asif; Ulugbek S. Kamilov
>
> **摘要:** Out-of-distribution (OOD) detection is critical for the safe deployment of machine learning systems in safety-sensitive domains. Diffusion models have recently emerged as powerful generative models, capable of capturing complex data distributions through iterative denoising. Building on this progress, recent work has explored their potential for OOD detection. We propose EigenScore, a new OOD detection method that leverages the eigenvalue spectrum of the posterior covariance induced by a diffusion model. We argue that posterior covariance provides a consistent signal of distribution shift, leading to larger trace and leading eigenvalues on OOD inputs, yielding a clear spectral signature. We further provide analysis explicitly linking posterior covariance to distribution mismatch, establishing it as a reliable signal for OOD detection. To ensure tractability, we adopt a Jacobian-free subspace iteration method to estimate the leading eigenvalues using only forward evaluations of the denoiser. Empirically, EigenScore achieves SOTA performance, with up to 5% AUROC improvement over the best baseline. Notably, it remains robust in near-OOD settings such as CIFAR-10 vs CIFAR-100, where existing diffusion-based methods often fail.
>
---
#### [new 052] Online Generic Event Boundary Detection
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于在线通用事件边界检测（On-GEBD）任务，旨在实时识别视频中事件的边界。它解决了现有方法需处理完整视频的问题，提出Estimator框架，结合事件预测与误差分析，实现对细微事件变化的即时检测。**

- **链接: [http://arxiv.org/pdf/2510.06855v1](http://arxiv.org/pdf/2510.06855v1)**

> **作者:** Hyungrok Jung; Daneul Kim; Seunggyun Lim; Jeany Son; Jonghyun Choi
>
> **备注:** ICCV 2025
>
> **摘要:** Generic Event Boundary Detection (GEBD) aims to interpret long-form videos through the lens of human perception. However, current GEBD methods require processing complete video frames to make predictions, unlike humans processing data online and in real-time. To bridge this gap, we introduce a new task, Online Generic Event Boundary Detection (On-GEBD), aiming to detect boundaries of generic events immediately in streaming videos. This task faces unique challenges of identifying subtle, taxonomy-free event changes in real-time, without the access to future frames. To tackle these challenges, we propose a novel On-GEBD framework, Estimator, inspired by Event Segmentation Theory (EST) which explains how humans segment ongoing activity into events by leveraging the discrepancies between predicted and actual information. Our framework consists of two key components: the Consistent Event Anticipator (CEA), and the Online Boundary Discriminator (OBD). Specifically, the CEA generates a prediction of the future frame reflecting current event dynamics based solely on prior frames. Then, the OBD measures the prediction error and adaptively adjusts the threshold using statistical tests on past errors to capture diverse, subtle event transitions. Experimental results demonstrate that Estimator outperforms all baselines adapted from recent online video understanding models and achieves performance comparable to prior offline-GEBD methods on the Kinetics-GEBD and TAPOS datasets.
>
---
#### [new 053] RGBD Gaze Tracking Using Transformer for Feature Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于眼动追踪任务，旨在解决利用RGBD图像进行视线角度估计的问题。作者设计了基于Transformer的特征融合模型，并构建了新数据集。实验比较了不同模型结构的效果，结果显示在多个数据集上取得了较好的精度表现。**

- **链接: [http://arxiv.org/pdf/2510.06298v1](http://arxiv.org/pdf/2510.06298v1)**

> **作者:** Tobias J. Bauer
>
> **备注:** Master Thesis with 125 pages, 59 figures, 17 tables
>
> **摘要:** Subject of this thesis is the implementation of an AI-based Gaze Tracking system using RGBD images that contain both color (RGB) and depth (D) information. To fuse the features extracted from the images, a module based on the Transformer architecture is used. The combination of RGBD input images and Transformers was chosen because it has not yet been investigated. Furthermore, a new dataset is created for training the AI models as existing datasets either do not contain depth information or only contain labels for Gaze Point Estimation that are not suitable for the task of Gaze Angle Estimation. Various model configurations are trained, validated and evaluated on a total of three different datasets. The trained models are then to be used in a real-time pipeline to estimate the gaze direction and thus the gaze point of a person in front of a computer screen. The AI model architecture used in this thesis is based on an earlier work by Lian et al. It uses a Generative Adversarial Network (GAN) to simultaneously remove depth map artifacts and extract head pose features. Lian et al. achieve a mean Euclidean error of 38.7mm on their own dataset ShanghaiTechGaze+. In this thesis, a model architecture with a Transformer module for feature fusion achieves a mean Euclidean error of 55.3mm on the same dataset, but we show that using no pre-trained GAN module leads to a mean Euclidean error of 30.1mm. Replacing the Transformer module with a Multilayer Perceptron (MLP) improves the error to 26.9mm. These results are coherent with the ones on the other two datasets. On the ETH-XGaze dataset, the model with Transformer module achieves a mean angular error of 3.59{\deg} and without Transformer module 3.26{\deg}, whereas the fundamentally different model architecture used by the dataset authors Zhang et al. achieves a mean angular error of 2.04{\deg}. On the OTH-Gaze-Estimation dataset created for...
>
---
#### [new 054] Does Physics Knowledge Emerge in Frontier Models?
- **分类: cs.CV**

- **简介: 该论文研究前沿视觉-语言模型（VLMs）是否具备物理知识，特别是在物理动态理解与预测方面的能力。论文属于视觉-语言模型与物理推理交叉任务，旨在解决当前模型在物理推理能力不明确的问题。作者在三个物理模拟数据集上测试六个前沿模型，并设计诊断子测试，分析感知与物理推理能力的关系，发现模型的感知与物理技能未能有效结合，缺乏因果理解。**

- **链接: [http://arxiv.org/pdf/2510.06251v1](http://arxiv.org/pdf/2510.06251v1)**

> **作者:** Ieva Bagdonaviciute; Vibhav Vineet
>
> **备注:** 8 pages, 7 figures. Preprint
>
> **摘要:** Leading Vision-Language Models (VLMs) show strong results in visual perception and general reasoning, but their ability to understand and predict physical dynamics remains unclear. We benchmark six frontier VLMs on three physical simulation datasets - CLEVRER, Physion, and Physion++ - where the evaluation tasks test whether a model can predict outcomes or hypothesize about alternative situations. To probe deeper, we design diagnostic subtests that isolate perception (objects, colors, occluders) from physics reasoning (motion prediction, spatial relations). Intuitively, stronger diagnostic performance should support higher evaluation accuracy. Yet our analysis reveals weak correlations: models that excel at perception or physics reasoning do not consistently perform better on predictive or counterfactual evaluation. This counterintuitive gap exposes a central limitation of current VLMs: perceptual and physics skills remain fragmented and fail to combine into causal understanding, underscoring the need for architectures that bind perception and reasoning more tightly.
>
---
#### [new 055] Extreme Amodal Face Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于极端非完整目标检测任务，旨在从单张图像中推断视野外的人脸位置。现有方法依赖视频序列或生成模型，而该文提出一种基于热图的高效、无采样方法，利用上下文线索和粗到细解码器预测不可见人脸，性能优于生成模型。**

- **链接: [http://arxiv.org/pdf/2510.06791v1](http://arxiv.org/pdf/2510.06791v1)**

> **作者:** Changlin Song; Yunzhong Hou; Michael Randall Barnes; Rahul Shome; Dylan Campbell
>
> **摘要:** Extreme amodal detection is the task of inferring the 2D location of objects that are not fully visible in the input image but are visible within an expanded field-of-view. This differs from amodal detection, where the object is partially visible within the input image, but is occluded. In this paper, we consider the sub-problem of face detection, since this class provides motivating applications involving safety and privacy, but do not tailor our method specifically to this class. Existing approaches rely on image sequences so that missing detections may be interpolated from surrounding frames or make use of generative models to sample possible completions. In contrast, we consider the single-image task and propose a more efficient, sample-free approach that makes use of the contextual cues from the image to infer the presence of unseen faces. We design a heatmap-based extreme amodal object detector that addresses the problem of efficiently predicting a lot (the out-of-frame region) from a little (the image) with a selective coarse-to-fine decoder. Our method establishes strong results for this new task, even outperforming less efficient generative approaches.
>
---
#### [new 056] Lattice-allocated Real-time Line Segment Feature Detection and Tracking Using Only an Event-based Camera
- **分类: cs.CV**

- **简介: 该论文属于实时线段特征检测与跟踪任务，旨在解决仅使用事件相机在高事件率下实现高效、准确的线段提取问题。论文提出了一种基于晶格分配的处理流程，包含速度不变事件表示、基于拟合评分的线段检测及端点扰动跟踪方法，实现了无需额外帧相机的实时操作，并在多个数据集上验证了性能优势。**

- **链接: [http://arxiv.org/pdf/2510.06829v1](http://arxiv.org/pdf/2510.06829v1)**

> **作者:** Mikihiro Ikura; Arren Glover; Masayoshi Mizuno; Chiara Bartolozzi
>
> **备注:** 12 pages, 13 figures, 6 tables, ICCV Workshop NeVi2025
>
> **摘要:** Line segment extraction is effective for capturing geometric features of human-made environments. Event-based cameras, which asynchronously respond to contrast changes along edges, enable efficient extraction by reducing redundant data. However, recent methods often rely on additional frame cameras or struggle with high event rates. This research addresses real-time line segment detection and tracking using only a modern, high-resolution (i.e., high event rate) event-based camera. Our lattice-allocated pipeline consists of (i) velocity-invariant event representation, (ii) line segment detection based on a fitting score, (iii) and line segment tracking by perturbating endpoints. Evaluation using ad-hoc recorded dataset and public datasets demonstrates real-time performance and higher accuracy compared to state-of-the-art event-only and event-frame hybrid baselines, enabling fully stand-alone event camera operation in real-world settings.
>
---
#### [new 057] Temporal Prompting Matters: Rethinking Referring Video Object Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，旨在解决需大量标注的RVOS问题。通过分解任务为指代表达、视频和分割三个因素，提出Tenet框架，结合现有检测与跟踪模型生成并评估时序提示，有效适配图像分割模型至视频领域，减少训练成本并提升效果。**

- **链接: [http://arxiv.org/pdf/2510.07319v1](http://arxiv.org/pdf/2510.07319v1)**

> **作者:** Ci-Siang Lin; Min-Hung Chen; I-Jieh Liu; Chien-Yi Wang; Sifei Liu; Yu-Chiang Frank Wang
>
> **摘要:** Referring Video Object Segmentation (RVOS) aims to segment the object referred to by the query sentence in the video. Most existing methods require end-to-end training with dense mask annotations, which could be computation-consuming and less scalable. In this work, we rethink the RVOS problem and aim to investigate the key to this task. Based on existing foundation segmentation models, we decompose the RVOS task into referring, video, and segmentation factors, and propose a Temporal Prompt Generation and Selection (Tenet) framework to address the referring and video factors while leaving the segmentation problem to foundation models. To efficiently adapt image-based foundation segmentation models to referring video object segmentation, we leverage off-the-shelf object detectors and trackers to produce temporal prompts associated with the referring sentence. While high-quality temporal prompts could be produced, they can not be easily identified from confidence scores. To tackle this issue, we propose Prompt Preference Learning to evaluate the quality of the produced temporal prompts. By taking such prompts to instruct image-based foundation segmentation models, we would be able to produce high-quality masks for the referred object, enabling efficient model adaptation to referring video object segmentation. Experiments on RVOS benchmarks demonstrate the effectiveness of the Tenet framework.
>
---
#### [new 058] Self-supervised Physics-guided Model with Implicit Representation Regularization for Fast MRI Reconstruction
- **分类: cs.CV**

- **简介: 论文提出了一种自监督物理引导的MRI快速重建方法UnrollINR，无需外部训练数据，通过结合物理模型与隐式神经表示，实现高质量、快速的MRI图像重建，解决了数据采样不足导致的重建难题。**

- **链接: [http://arxiv.org/pdf/2510.06611v1](http://arxiv.org/pdf/2510.06611v1)**

> **作者:** Jingran Xu; Yuanyuan Liu; Yanjie Zhu
>
> **摘要:** Magnetic Resonance Imaging (MRI) is a vital clinical diagnostic tool, yet its widespread application is limited by prolonged scan times. Fast MRI reconstruction techniques effectively reduce acquisition duration by reconstructing high-fidelity MR images from undersampled k-space data. In recent years, deep learning-based methods have demonstrated remarkable progress in this field, with self-supervised and unsupervised learning approaches proving particularly valuable in scenarios where fully sampled data are difficult to obtain. This paper proposes a novel zero-shot self-supervised reconstruction framework named UnrollINR, which enables scan-specific MRI reconstruction without relying on external training data. The method adopts a physics-guided unrolled iterative reconstruction architecture and introduces Implicit Neural Representation (INR) as a regularization prior to effectively constrain the solution space. By combining a deep unrolled structure with the powerful implicit representation capability of INR, the model's interpretability and reconstruction performance are enhanced. Experimental results demonstrate that even at a high acceleration rate of 10, UnrollINR achieves superior reconstruction performance compared to the supervised learning method, validating the superiority of the proposed method.
>
---
#### [new 059] U-Bench: A Comprehensive Understanding of U-Net through 100-Variant Benchmarking
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决U-Net变体缺乏系统评估的问题。作者构建了U-Bench，首个大规模、统计严谨的基准，评估100种U-Net变体在28个数据集上的性能，提出U-Score指标，并提供模型选择建议与开源资源，推动公平、可复现的模型评估。**

- **链接: [http://arxiv.org/pdf/2510.07041v1](http://arxiv.org/pdf/2510.07041v1)**

> **作者:** Fenghe Tang; Chengqi Dong; Wenxin Ma; Zikang Xu; Heqin Zhu; Zihang Jiang; Rongsheng Wang; Yuhao Wang; Chenxu Wu; Shaohua Kevin Zhou
>
> **备注:** 54 pages. The project can be accessed at: https://fenghetan9.github.io/ubench. Code is available at: https://github.com/FengheTan9/U-Bench
>
> **摘要:** Over the past decade, U-Net has been the dominant architecture in medical image segmentation, leading to the development of thousands of U-shaped variants. Despite its widespread adoption, there is still no comprehensive benchmark to systematically evaluate their performance and utility, largely because of insufficient statistical validation and limited consideration of efficiency and generalization across diverse datasets. To bridge this gap, we present U-Bench, the first large-scale, statistically rigorous benchmark that evaluates 100 U-Net variants across 28 datasets and 10 imaging modalities. Our contributions are threefold: (1) Comprehensive Evaluation: U-Bench evaluates models along three key dimensions: statistical robustness, zero-shot generalization, and computational efficiency. We introduce a novel metric, U-Score, which jointly captures the performance-efficiency trade-off, offering a deployment-oriented perspective on model progress. (2) Systematic Analysis and Model Selection Guidance: We summarize key findings from the large-scale evaluation and systematically analyze the impact of dataset characteristics and architectural paradigms on model performance. Based on these insights, we propose a model advisor agent to guide researchers in selecting the most suitable models for specific datasets and tasks. (3) Public Availability: We provide all code, models, protocols, and weights, enabling the community to reproduce our results and extend the benchmark with future methods. In summary, U-Bench not only exposes gaps in previous evaluations but also establishes a foundation for fair, reproducible, and practically relevant benchmarking in the next decade of U-Net-based segmentation models. The project can be accessed at: https://fenghetan9.github.io/ubench. Code is available at: https://github.com/FengheTan9/U-Bench.
>
---
#### [new 060] LogSTOP: Temporal Scores over Prediction Sequences for Matching and Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频和音频理解任务，旨在解决如何从局部属性的检测分数中推导出时间属性评分的问题。作者提出了LogSTOP方法，利用线性时序逻辑计算时间属性的评分，并在查询匹配和排序检索任务中取得了优于现有方法的性能。**

- **链接: [http://arxiv.org/pdf/2510.06512v1](http://arxiv.org/pdf/2510.06512v1)**

> **作者:** Avishree Khare; Hideki Okamoto; Bardh Hoxha; Georgios Fainekos; Rajeev Alur
>
> **摘要:** Neural models such as YOLO and HuBERT can be used to detect local properties such as objects ("car") and emotions ("angry") in individual frames of videos and audio clips respectively. The likelihood of these detections is indicated by scores in [0, 1]. Lifting these scores to temporal properties over sequences can be useful for several downstream applications such as query matching (e.g., "does the speaker eventually sound happy in this audio clip?"), and ranked retrieval (e.g., "retrieve top 5 videos with a 10 second scene where a car is detected until a pedestrian is detected"). In this work, we formalize this problem of assigning Scores for TempOral Properties (STOPs) over sequences, given potentially noisy score predictors for local properties. We then propose a scoring function called LogSTOP that can efficiently compute these scores for temporal properties represented in Linear Temporal Logic. Empirically, LogSTOP, with YOLO and HuBERT, outperforms Large Vision / Audio Language Models and other Temporal Logic-based baselines by at least 16% on query matching with temporal properties over objects-in-videos and emotions-in-speech respectively. Similarly, on ranked retrieval with temporal properties over objects and actions in videos, LogSTOP with Grounding DINO and SlowR50 reports at least a 19% and 16% increase in mean average precision and recall over zero-shot text-to-video retrieval baselines respectively.
>
---
#### [new 061] Addressing the ID-Matching Challenge in Long Video Captioning
- **分类: cs.CV**

- **简介: 该论文属于视频描述生成任务，旨在解决长视频中人物身份匹配（ID-Matching）问题。现有方法在此任务上泛化能力差，依赖点对点匹配。论文提出RICE方法，利用LVLM（如GPT-4o）提升ID-Matching性能，通过新基准评估，验证了图像信息与个体描述增强的有效性，显著提高了ID-Matching的精度与召回率。**

- **链接: [http://arxiv.org/pdf/2510.06973v1](http://arxiv.org/pdf/2510.06973v1)**

> **作者:** Zhantao Yang; Huangji Wang; Ruili Feng; Han Zhang; Yuting Hu; Shangwen Zhu; Junyan Li; Yu Liu; Fan Cheng
>
> **摘要:** Generating captions for long and complex videos is both critical and challenging, with significant implications for the growing fields of text-to-video generation and multi-modal understanding. One key challenge in long video captioning is accurately recognizing the same individuals who appear in different frames, which we refer to as the ID-Matching problem. Few prior works have focused on this important issue. Those that have, usually suffer from limited generalization and depend on point-wise matching, which limits their overall effectiveness. In this paper, unlike previous approaches, we build upon LVLMs to leverage their powerful priors. We aim to unlock the inherent ID-Matching capabilities within LVLMs themselves to enhance the ID-Matching performance of captions. Specifically, we first introduce a new benchmark for assessing the ID-Matching capabilities of video captions. Using this benchmark, we investigate LVLMs containing GPT-4o, revealing key insights that the performance of ID-Matching can be improved through two methods: 1) enhancing the usage of image information and 2) increasing the quantity of information of individual descriptions. Based on these insights, we propose a novel video captioning method called Recognizing Identities for Captioning Effectively (RICE). Extensive experiments including assessments of caption quality and ID-Matching performance, demonstrate the superiority of our approach. Notably, when implemented on GPT-4o, our RICE improves the precision of ID-Matching from 50% to 90% and improves the recall of ID-Matching from 15% to 80% compared to baseline. RICE makes it possible to continuously track different individuals in the captions of long videos.
>
---
#### [new 062] DreamOmni2: Multimodal Instruction-based Editing and Generation
- **分类: cs.CV**

- **简介: 该论文提出DreamOmni2，致力于解决指令图像编辑与生成中的细节不足与概念局限问题，新增支持图文指令的多模态任务，涵盖具象与抽象概念，提升实用价值。**

- **链接: [http://arxiv.org/pdf/2510.06679v1](http://arxiv.org/pdf/2510.06679v1)**

> **作者:** Bin Xia; Bohao Peng; Yuechen Zhang; Junjia Huang; Jiyang Liu; Jingyao Li; Haoru Tan; Sitong Wu; Chengyao Wang; Yitong Wang; Xinglong Wu; Bei Yu; Jiaya Jia
>
> **摘要:** Recent advancements in instruction-based image editing and subject-driven generation have garnered significant attention, yet both tasks still face limitations in meeting practical user needs. Instruction-based editing relies solely on language instructions, which often fail to capture specific editing details, making reference images necessary. Meanwhile, subject-driven generation is limited to combining concrete objects or people, overlooking broader, abstract concepts. To address these challenges, we propose two novel tasks: multimodal instruction-based editing and generation. These tasks support both text and image instructions and extend the scope to include both concrete and abstract concepts, greatly enhancing their practical applications. We introduce DreamOmni2, tackling two primary challenges: data creation and model framework design. Our data synthesis pipeline consists of three steps: (1) using a feature mixing method to create extraction data for both abstract and concrete concepts, (2) generating multimodal instruction-based editing training data using the editing and extraction models, and (3) further applying the extraction model to create training data for multimodal instruction-based editing. For the framework, to handle multi-image input, we propose an index encoding and position encoding shift scheme, which helps the model distinguish images and avoid pixel confusion. Additionally, we introduce joint training with the VLM and our generation/editing model to better process complex instructions. In addition, we have proposed comprehensive benchmarks for these two new tasks to drive their development. Experiments show that DreamOmni2 has achieved impressive results. Models and codes will be released.
>
---
#### [new 063] VUGEN: Visual Understanding priors for GENeration
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决现有视觉语言模型在图像生成中存在理解与生成表示不一致或结构复杂的问题。论文提出VUGEN框架，利用预训练视觉理解先验，通过降维、对齐生成和解码，实现高效高质量图像生成，并在COCO数据集上取得了更好的性能表现。**

- **链接: [http://arxiv.org/pdf/2510.06529v1](http://arxiv.org/pdf/2510.06529v1)**

> **作者:** Xiangyi Chen; Théophane Vallaeys; Maha Elbayad; John Nguyen; Jakob Verbeek
>
> **摘要:** Recent advances in Vision-Language Models (VLMs) have enabled unified understanding across text and images, yet equipping these models with robust image generation capabilities remains challenging. Existing approaches often rely on reconstruction-oriented autoencoders or complex bridging mechanisms, leading to misalignment between understanding and generation representations, or architectural complexity. In this work, we propose VUGEN, a novel framework that explicitly leverages VLM's pretrained visual understanding priors for efficient and high-quality image generation. Our approach first transforms the high-dimensional latent space of the VLM's native vision encoder into a lower-dimensional, tractable distribution that maximally preserves visual information. The VLM is then trained to sample within this reduced latent space, ensuring alignment with its visual understanding capabilities. Finally, a dedicated pixel decoder maps these generated latents back to the image space. We find that a VAE-free pixel diffusion decoder to be on par or better than commonly used complex latent diffusion decoders that internally rely on VAE latents. Extensive experiments demonstrate that VUGEN achieves superior image generation performance, improving DPG Bench from 71.17 to 74.32 and FID from 11.86 to 9.06 on COCO, while fully preserving the VLM's original understanding capabilities.
>
---
#### [new 064] Explaining raw data complexity to improve satellite onboard processing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感任务，旨在解决在卫星上使用原始传感器数据进行AI处理的问题。通过模拟生成原始数据并比较两种模型在不同数据上的表现，发现原始数据在高置信度下检测效果较差，建议改进AI架构以提升边界识别能力。**

- **链接: [http://arxiv.org/pdf/2510.06858v1](http://arxiv.org/pdf/2510.06858v1)**

> **作者:** Adrien Dorise; Marjorie Bellizzi; Adrien Girard; Benjamin Francesconi; Stéphane May
>
> **备注:** Preprint: European Data Handling & Data Processing Conference (EDHPC) 2025
>
> **摘要:** With increasing processing power, deploying AI models for remote sensing directly onboard satellites is becoming feasible. However, new constraints arise, mainly when using raw, unprocessed sensor data instead of preprocessed ground-based products. While current solutions primarily rely on preprocessed sensor images, few approaches directly leverage raw data. This study investigates the effects of utilising raw data on deep learning models for object detection and classification tasks. We introduce a simulation workflow to generate raw-like products from high-resolution L1 imagery, enabling systemic evaluation. Two object detection models (YOLOv11s and YOLOX-S) are trained on both raw and L1 datasets, and their performance is compared using standard detection metrics and explainability tools. Results indicate that while both models perform similarly at low to medium confidence thresholds, the model trained on raw data struggles with object boundary identification at high confidence levels. It suggests that adapting AI architectures with improved contouring methods can enhance object detection on raw images, improving onboard AI for remote sensing.
>
---
#### [new 065] TDiff: Thermal Plug-And-Play Prior with Patch-Based Diffusion
- **分类: cs.CV**

- **简介: 论文提出了一种基于扩散模型的热成像修复方法TDiff，旨在解决低分辨率、固定模式噪声等问题。通过局部块训练和空间平滑融合，实现去噪、超分辨率和去模糊任务。属于图像恢复领域，适用于多任务热图像处理。**

- **链接: [http://arxiv.org/pdf/2510.06460v1](http://arxiv.org/pdf/2510.06460v1)**

> **作者:** Piyush Dashpute; Niki Nezakati; Wolfgang Heidrich; Vishwanath Saragadam
>
> **摘要:** Thermal images from low-cost cameras often suffer from low resolution, fixed pattern noise, and other localized degradations. Available datasets for thermal imaging are also limited in both size and diversity. To address these challenges, we propose a patch-based diffusion framework (TDiff) that leverages the local nature of these distortions by training on small thermal patches. In this approach, full-resolution images are restored by denoising overlapping patches and blending them using smooth spatial windowing. To our knowledge, this is the first patch-based diffusion framework that models a learned prior for thermal image restoration across multiple tasks. Experiments on denoising, super-resolution, and deblurring demonstrate strong results on both simulated and real thermal data, establishing our method as a unified restoration pipeline.
>
---
#### [new 066] WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉生成与机器人操控任务，旨在解决腕视图数据稀缺问题。通过提出WristWorld模型，利用锚视图生成几何一致的腕视图视频，提升VLA模型的操作性能。**

- **链接: [http://arxiv.org/pdf/2510.07313v1](http://arxiv.org/pdf/2510.07313v1)**

> **作者:** Zezhong Qian; Xiaowei Chi; Yuming Li; Shizun Wang; Zhiyuan Qin; Xiaozhu Ju; Sirui Han; Shanghang Zhang
>
> **摘要:** Wrist-view observations are crucial for VLA models as they capture fine-grained hand-object interactions that directly enhance manipulation performance. Yet large-scale datasets rarely include such recordings, resulting in a substantial gap between abundant anchor views and scarce wrist views. Existing world models cannot bridge this gap, as they require a wrist-view first frame and thus fail to generate wrist-view videos from anchor views alone. Amid this gap, recent visual geometry models such as VGGT emerge with geometric and cross-view priors that make it possible to address extreme viewpoint shifts. Inspired by these insights, we propose WristWorld, the first 4D world model that generates wrist-view videos solely from anchor views. WristWorld operates in two stages: (i) Reconstruction, which extends VGGT and incorporates our Spatial Projection Consistency (SPC) Loss to estimate geometrically consistent wrist-view poses and 4D point clouds; (ii) Generation, which employs our video generation model to synthesize temporally coherent wrist-view videos from the reconstructed perspective. Experiments on Droid, Calvin, and Franka Panda demonstrate state-of-the-art video generation with superior spatial consistency, while also improving VLA performance, raising the average task completion length on Calvin by 3.81% and closing 42.4% of the anchor-wrist view gap.
>
---
#### [new 067] AIM 2025 Challenge on Real-World RAW Image Denoising
- **分类: cs.CV**

- **简介: 该论文介绍了AIM 2025真实场景RAW图像去噪挑战赛，属于图像恢复任务。旨在推动基于数据合成的高效去噪技术发展，解决低光环境下不同相机拍摄的RAW图像噪声问题。论文构建了新的评估基准，结合全参考与非参考指标评估模型性能，促进夜间自动驾驶等领域的技术进步。**

- **链接: [http://arxiv.org/pdf/2510.06601v1](http://arxiv.org/pdf/2510.06601v1)**

> **作者:** Feiran Li; Jiacheng Li; Marcos V. Conde; Beril Besbinar; Vlad Hosu; Daisuke Iso; Radu Timofte
>
> **摘要:** We introduce the AIM 2025 Real-World RAW Image Denoising Challenge, aiming to advance efficient and effective denoising techniques grounded in data synthesis. The competition is built upon a newly established evaluation benchmark featuring challenging low-light noisy images captured in the wild using five different DSLR cameras. Participants are tasked with developing novel noise synthesis pipelines, network architectures, and training methodologies to achieve high performance across different camera models. Winners are determined based on a combination of performance metrics, including full-reference measures (PSNR, SSIM, LPIPS), and non-reference ones (ARNIQA, TOPIQ). By pushing the boundaries of camera-agnostic low-light RAW image denoising trained on synthetic data, the competition promotes the development of robust and practical models aligned with the rapid progress in digital photography. We expect the competition outcomes to influence multiple domains, from image restoration to night-time autonomous driving.
>
---
#### [new 068] Ensemble Deep Learning and LLM-Assisted Reporting for Automated Skin Lesion Diagnosis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于皮肤病诊断任务，旨在解决现有AI系统在皮肤病变诊断中的同质性架构、数据集偏差及沟通障碍问题。论文提出了一种集成深度学习与大语言模型的统一框架，通过异构模型集成提升诊断可靠性，并嵌入语言模型生成临床报告，实现诊断与患者教育一体化，从而提升早期干预率。**

- **链接: [http://arxiv.org/pdf/2510.06260v1](http://arxiv.org/pdf/2510.06260v1)**

> **作者:** Sher Khan; Raz Muhammad; Adil Hussain; Muhammad Sajjad; Muhammad Rashid
>
> **摘要:** Cutaneous malignancies demand early detection for favorable outcomes, yet current diagnostics suffer from inter-observer variability and access disparities. While AI shows promise, existing dermatological systems are limited by homogeneous architectures, dataset biases across skin tones, and fragmented approaches that treat natural language processing as separate post-hoc explanations rather than integral to clinical decision-making. We introduce a unified framework that fundamentally reimagines AI integration for dermatological diagnostics through two synergistic innovations. First, a purposefully heterogeneous ensemble of architecturally diverse convolutional neural networks provides complementary diagnostic perspectives, with an intrinsic uncertainty mechanism flagging discordant cases for specialist review -- mimicking clinical best practices. Second, we embed large language model capabilities directly into the diagnostic workflow, transforming classification outputs into clinically meaningful assessments that simultaneously fulfill medical documentation requirements and deliver patient-centered education. This seamless integration generates structured reports featuring precise lesion characterization, accessible diagnostic reasoning, and actionable monitoring guidance -- empowering patients to recognize early warning signs between visits. By addressing both diagnostic reliability and communication barriers within a single cohesive system, our approach bridges the critical translational gap that has prevented previous AI implementations from achieving clinical impact. The framework represents a significant advancement toward deployable dermatological AI that enhances diagnostic precision while actively supporting the continuum of care from initial detection through patient education, ultimately improving early intervention rates for skin lesions.
>
---
#### [new 069] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决视觉风格提示中的内容泄漏问题。作者提出StyleKeeper方法，通过扩展无分类器引导和引入负向视觉查询引导，减少不想要的内容传递，提升生成图像的风格控制与文本匹配度。**

- **链接: [http://arxiv.org/pdf/2510.06827v1](http://arxiv.org/pdf/2510.06827v1)**

> **作者:** Jaeseok Jeong; Junho Kim; Gayoung Lee; Yunjey Choi; Youngjung Uh
>
> **备注:** Accepted to ICCV 2025; CVPRW AI4CC 2024 (Best Paper + Oral)
>
> **摘要:** In the domain of text-to-image generation, diffusion models have emerged as powerful tools. Recently, studies on visual prompting, where images are used as prompts, have enabled more precise control over style and content. However, existing methods often suffer from content leakage, where undesired elements of the visual style prompt are transferred along with the intended style. To address this issue, we 1) extend classifier-free guidance (CFG) to utilize swapping self-attention and propose 2) negative visual query guidance (NVQG) to reduce the transfer of unwanted contents. NVQG employs negative score by intentionally simulating content leakage scenarios that swap queries instead of key and values of self-attention layers from visual style prompts. This simple yet effective method significantly reduces content leakage. Furthermore, we provide careful solutions for using a real image as visual style prompts. Through extensive evaluation across various styles and text prompts, our method demonstrates superiority over existing approaches, reflecting the style of the references, and ensuring that resulting images match the text prompts. Our code is available \href{https://github.com/naver-ai/StyleKeeper}{here}.
>
---
#### [new 070] TalkCuts: A Large-Scale Dataset for Multi-Shot Human Speech Video Generation
- **分类: cs.CV**

- **简介: 该论文属于多镜头语音视频生成任务，旨在解决现有数据集单一视角限制的问题。作者构建了大规模数据集TalkCuts，包含164k高质量视频片段，并提出Orator框架实现多模态视频生成。实验表明，该数据集能提升生成视频的连贯性和视觉效果，为可控多镜头语音视频生成提供基础。**

- **链接: [http://arxiv.org/pdf/2510.07249v1](http://arxiv.org/pdf/2510.07249v1)**

> **作者:** Jiaben Chen; Zixin Wang; Ailing Zeng; Yang Fu; Xueyang Yu; Siyuan Cen; Julian Tanke; Yihang Chen; Koichi Saito; Yuki Mitsufuji; Chuang Gan
>
> **备注:** Project page: https://talkcuts.github.io/
>
> **摘要:** In this work, we present TalkCuts, a large-scale dataset designed to facilitate the study of multi-shot human speech video generation. Unlike existing datasets that focus on single-shot, static viewpoints, TalkCuts offers 164k clips totaling over 500 hours of high-quality human speech videos with diverse camera shots, including close-up, half-body, and full-body views. The dataset includes detailed textual descriptions, 2D keypoints and 3D SMPL-X motion annotations, covering over 10k identities, enabling multimodal learning and evaluation. As a first attempt to showcase the value of the dataset, we present Orator, an LLM-guided multi-modal generation framework as a simple baseline, where the language model functions as a multi-faceted director, orchestrating detailed specifications for camera transitions, speaker gesticulations, and vocal modulation. This architecture enables the synthesis of coherent long-form videos through our integrated multi-modal video generation module. Extensive experiments in both pose-guided and audio-driven settings show that training on TalkCuts significantly enhances the cinematographic coherence and visual appeal of generated multi-shot speech videos. We believe TalkCuts provides a strong foundation for future work in controllable, multi-shot speech video generation and broader multimodal learning.
>
---
#### [new 071] Continual Action Quality Assessment via Adaptive Manifold-Aligned Graph Regularization
- **分类: cs.CV**

- **简介: 该论文属于动作质量评估任务，旨在解决现实场景中质量分布非平稳导致模型泛化能力差的问题。作者提出了Continual AQA（CAQA）框架，结合持续学习与自适应流形对齐图正则化方法（MAGR++），缓解灾难性遗忘并提升模型适应能力。**

- **链接: [http://arxiv.org/pdf/2510.06842v1](http://arxiv.org/pdf/2510.06842v1)**

> **作者:** Kanglei Zhou; Qingyi Pan; Xingxing Zhang; Hubert P. H. Shum; Frederick W. B. Li; Xiaohui Liang; Liyuan Wang
>
> **备注:** Extended Version of MAGR (ECCV 2024 Oral Presentation)
>
> **摘要:** Action Quality Assessment (AQA) quantifies human actions in videos, supporting applications in sports scoring, rehabilitation, and skill evaluation. A major challenge lies in the non-stationary nature of quality distributions in real-world scenarios, which limits the generalization ability of conventional methods. We introduce Continual AQA (CAQA), which equips AQA with Continual Learning (CL) capabilities to handle evolving distributions while mitigating catastrophic forgetting. Although parameter-efficient fine-tuning of pretrained models has shown promise in CL for image classification, we find it insufficient for CAQA. Our empirical and theoretical analyses reveal two insights: (i) Full-Parameter Fine-Tuning (FPFT) is necessary for effective representation learning; yet (ii) uncontrolled FPFT induces overfitting and feature manifold shift, thereby aggravating forgetting. To address this, we propose Adaptive Manifold-Aligned Graph Regularization (MAGR++), which couples backbone fine-tuning that stabilizes shallow layers while adapting deeper ones with a two-step feature rectification pipeline: a manifold projector to translate deviated historical features into the current representation space, and a graph regularizer to align local and global distributions. We construct four CAQA benchmarks from three datasets with tailored evaluation protocols and strong baselines, enabling systematic cross-dataset comparison. Extensive experiments show that MAGR++ achieves state-of-the-art performance, with average correlation gains of 3.6% offline and 12.2% online over the strongest baseline, confirming its robustness and effectiveness. Our code is available at https://github.com/ZhouKanglei/MAGRPP.
>
---
#### [new 072] Road Surface Condition Detection with Machine Learning using New York State Department of Transportation Camera Images and Weather Forecast Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于道路状态检测任务，旨在解决人工观测道路条件效率低的问题。作者利用纽约州交通部的摄像头图像和天气数据，训练了卷积神经网络和随机森林模型，自动识别六种道路表面状况。模型在未见过的摄像头数据上达到了81.5%的准确率，具有良好的泛化能力，可为交通管理部门提供决策支持。**

- **链接: [http://arxiv.org/pdf/2510.06440v1](http://arxiv.org/pdf/2510.06440v1)**

> **作者:** Carly Sutter; Kara J. Sulia; Nick P. Bassill; Christopher D. Wirz; Christopher D. Thorncroft; Jay C. Rothenberger; Vanessa Przybylo; Mariana G. Cains; Jacob Radford; David Aaron Evans
>
> **摘要:** The New York State Department of Transportation (NYSDOT) has a network of roadside traffic cameras that are used by both the NYSDOT and the public to observe road conditions. The NYSDOT evaluates road conditions by driving on roads and observing live cameras, tasks which are labor-intensive but necessary for making critical operational decisions during winter weather events. However, machine learning models can provide additional support for the NYSDOT by automatically classifying current road conditions across the state. In this study, convolutional neural networks and random forests are trained on camera images and weather data to predict road surface conditions. Models are trained on a hand-labeled dataset of ~22,000 camera images, each classified by human labelers into one of six road surface conditions: severe snow, snow, wet, dry, poor visibility, or obstructed. Model generalizability is prioritized to meet the operational needs of the NYSDOT decision makers, and the weather-related road surface condition model in this study achieves an accuracy of 81.5% on completely unseen cameras.
>
---
#### [new 073] Enhancing Concept Localization in CLIP-based Concept Bottleneck Models
- **分类: cs.CV**

- **简介: 该论文属于可解释AI（XAI）任务，旨在解决基于CLIP的概念瓶颈模型（CBMs）中概念幻觉问题。作者提出CHILI方法，通过局部可解释性抑制概念幻觉，提升模型解释的可信度与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.07115v1](http://arxiv.org/pdf/2510.07115v1)**

> **作者:** Rémi Kazmierczak; Steve Azzolin; Eloïse Berthier; Goran Frehse; Gianni Franchi
>
> **摘要:** This paper addresses explainable AI (XAI) through the lens of Concept Bottleneck Models (CBMs) that do not require explicit concept annotations, relying instead on concepts extracted using CLIP in a zero-shot manner. We show that CLIP, which is central in these techniques, is prone to concept hallucination, incorrectly predicting the presence or absence of concepts within an image in scenarios used in numerous CBMs, hence undermining the faithfulness of explanations. To mitigate this issue, we introduce Concept Hallucination Inhibition via Localized Interpretability (CHILI), a technique that disentangles image embeddings and localizes pixels corresponding to target concepts. Furthermore, our approach supports the generation of saliency-based explanations that are more interpretable.
>
---
#### [new 074] SIGMA-GEN: Structure and Identity Guided Multi-subject Assembly for Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决多主体身份保留生成的问题。现有方法难以在单次生成中保留多个身份并结合结构和空间约束。论文提出了SIGMA-GEN框架和SIGMA-SET27K数据集，实现了单次生成中多身份保留，并支持多种用户引导方式。**

- **链接: [http://arxiv.org/pdf/2510.06469v1](http://arxiv.org/pdf/2510.06469v1)**

> **作者:** Oindrila Saha; Vojtech Krs; Radomir Mech; Subhransu Maji; Kevin Blackburn-Matzen; Matheus Gadelha
>
> **备注:** Webpage: https://oindrilasaha.github.io/SIGMA-Gen/
>
> **摘要:** We present SIGMA-GEN, a unified framework for multi-identity preserving image generation. Unlike prior approaches, SIGMA-GEN is the first to enable single-pass multi-subject identity-preserved generation guided by both structural and spatial constraints. A key strength of our method is its ability to support user guidance at various levels of precision -- from coarse 2D or 3D boxes to pixel-level segmentations and depth -- with a single model. To enable this, we introduce SIGMA-SET27K, a novel synthetic dataset that provides identity, structure, and spatial information for over 100k unique subjects across 27k images. Through extensive evaluation we demonstrate that SIGMA-GEN achieves state-of-the-art performance in identity preservation, image generation quality, and speed. Code and visualizations at https://oindrilasaha.github.io/SIGMA-Gen/
>
---
#### [new 075] Resolution scaling governs DINOv3 transfer performance in chest radiograph classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决自监督学习模型在胸片分类中的有效性问题。通过比较DINOv3、DINOv2和ImageNet初始化模型在不同分辨率下的迁移性能，发现DINOv3结合ConvNeXt-B在512x512分辨率下表现最佳，尤其对细微病变检测有显著优势。**

- **链接: [http://arxiv.org/pdf/2510.07191v1](http://arxiv.org/pdf/2510.07191v1)**

> **作者:** Soroosh Tayebi Arasteh; Mina Shaigan; Christiane Kuhl; Jakob Nikolas Kather; Sven Nebelung; Daniel Truhn
>
> **摘要:** Self-supervised learning (SSL) has advanced visual representation learning, but its value in chest radiography, a high-volume imaging modality with fine-grained findings, remains unclear. Meta's DINOv3 extends earlier SSL models through Gram-anchored self-distillation. Whether these design choices improve transfer learning for chest radiography has not been systematically tested. We benchmarked DINOv3 against DINOv2 and ImageNet initialization across seven datasets (n>814,000). Two representative backbones were evaluated: ViT-B/16 and ConvNeXt-B. Images were analyzed at 224x224, 512x512, and 1024x1024 pixels. We additionally assessed frozen features from a 7B model. The primary outcome was mean AUROC across labels. At 224x224, DINOv3 and DINOv2 achieved comparable performance on adult datasets. Increasing resolution to 512x512 yielded consistent improvements for DINOv3 over both DINOv2 and ImageNet. In contrast, results in pediatric cohort showed no differences across initializations. Across all settings, ConvNeXt-B outperformed ViT-B/16. Models using frozen DINOv3-7B features underperformed relative to fully finetuned 86-89M-parameter backbones, highlighting the importance of domain adaptation. Scaling to 1024x1024 did not further improve accuracy. Resolution-related gains were most evident for boundary-dependent and small focal abnormalities. In chest radiography, higher input resolution is critical for leveraging the benefits of modern self-supervised models. 512x512 pixels represent a practical upper limit where DINOv3-initialized ConvNeXt-B networks provide the strongest performance, while larger inputs offer minimal return on cost. Clinically, these findings support use of finetuned, mid-sized backbones at 512x512 for chest radiograph interpretation, with the greatest gains expected in detecting subtle or boundary-centered lesions relevant to emergency and critical care settings.
>
---
#### [new 076] User to Video: A Model for Spammer Detection Inspired by Video Classification Technology
- **分类: cs.CV**

- **简介: 该论文属于社交网络中的垃圾用户检测任务，旨在解决识别恶意账号的问题。作者提出UVSD模型，将用户行为转化为“视频”形式，通过像素化用户、生成行为图像并构建视频序列，结合视频分类技术检测垃圾用户。实验表明该方法优于现有技术。**

- **链接: [http://arxiv.org/pdf/2510.06233v1](http://arxiv.org/pdf/2510.06233v1)**

> **作者:** Haoyang Zhang; Zhou Yang; Yucai Pang
>
> **备注:** Accepted by International Joint Conference on Neural Networks (IJCNN) 2025
>
> **摘要:** This article is inspired by video classification technology. If the user behavior subspace is viewed as a frame image, consecutive frame images are viewed as a video. Following this novel idea, a model for spammer detection based on user videoization, called UVSD, is proposed. Firstly, a user2piexl algorithm for user pixelization is proposed. Considering the adversarial behavior of user stances, the user is viewed as a pixel, and the stance is quantified as the pixel's RGB. Secondly, a behavior2image algorithm is proposed for transforming user behavior subspace into frame images. Low-rank dense vectorization of subspace user relations is performed using representation learning, while cutting and diffusion algorithms are introduced to complete the frame imageization. Finally, user behavior videos are constructed based on temporal features. Subsequently, a video classification algorithm is combined to identify the spammers. Experiments using publicly available datasets, i.e., WEIBO and TWITTER, show an advantage of the UVSD model over state-of-the-art methods.
>
---
#### [new 077] Validation of Various Normalization Methods for Brain Tumor Segmentation: Can Federated Learning Overcome This Heterogeneity?
- **分类: cs.CV; cs.DC**

- **简介: 该论文属于医学图像分析任务，旨在解决因数据异质性（由不同MRI归一化方法引起）导致的联邦学习模型性能下降问题。作者通过模拟非独立同分布数据，评估不同归一化方法对脑肿瘤分割模型的影响，并验证联邦学习在该情况下的有效性。**

- **链接: [http://arxiv.org/pdf/2510.07126v1](http://arxiv.org/pdf/2510.07126v1)**

> **作者:** Jan Fiszer; Dominika Ciupek; Maciej Malawski
>
> **摘要:** Deep learning (DL) has been increasingly applied in medical imaging, however, it requires large amounts of data, which raises many challenges related to data privacy, storage, and transfer. Federated learning (FL) is a training paradigm that overcomes these issues, though its effectiveness may be reduced when dealing with non-independent and identically distributed (non-IID) data. This study simulates non-IID conditions by applying different MRI intensity normalization techniques to separate data subsets, reflecting a common cause of heterogeneity. These subsets are then used for training and testing models for brain tumor segmentation. The findings provide insights into the influence of the MRI intensity normalization methods on segmentation models, both training and inference. Notably, the FL methods demonstrated resilience to inconsistently normalized data across clients, achieving the 3D Dice score of 92%, which is comparable to a centralized model (trained using all data). These results indicate that FL is a solution to effectively train high-performing models without violating data privacy, a crucial concern in medical applications. The code is available at: https://github.com/SanoScience/fl-varying-normalization.
>
---
#### [new 078] Transforming Noise Distributions with Histogram Matching: Towards a Single Denoiser for All
- **分类: cs.CV**

- **简介: 该论文属于图像去噪任务，旨在解决现有高斯去噪器对分布外噪声泛化能力有限的问题。作者提出一种基于直方图匹配的噪声分布转换方法，并结合噪声变换与去噪的相互增强循环，提升去噪效果。通过局部直方图匹配、块内置换和频域变换等策略，使单一去噪器能有效应对多种噪声类型。**

- **链接: [http://arxiv.org/pdf/2510.06757v1](http://arxiv.org/pdf/2510.06757v1)**

> **作者:** Sheng Fu; Junchao Zhang; Kailun Yang
>
> **备注:** 12 pages
>
> **摘要:** Supervised Gaussian denoisers exhibit limited generalization when confronted with out-of-distribution noise, due to the diverse distributional characteristics of different noise types. To bridge this gap, we propose a histogram matching approach that transforms arbitrary noise towards a target Gaussian distribution with known intensity. Moreover, a mutually reinforcing cycle is established between noise transformation and subsequent denoising. This cycle progressively refines the noise to be converted, making it approximate the real noise, thereby enhancing the noise transformation effect and further improving the denoising performance. We tackle specific noise complexities: local histogram matching handles signal-dependent noise, intrapatch permutation processes channel-related noise, and frequency-domain histogram matching coupled with pixel-shuffle down-sampling breaks spatial correlation. By applying these transformations, a single Gaussian denoiser gains remarkable capability to handle various out-of-distribution noises, including synthetic noises such as Poisson, salt-and-pepper and repeating pattern noises, as well as complex real-world noises. Extensive experiments demonstrate the superior generalization and effectiveness of our method.
>
---
#### [new 079] Automated Neural Architecture Design for Industrial Defect Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于工业表面缺陷检测任务，旨在解决缺陷形态多样导致的检测难题。作者提出AutoNAD框架，自动设计神经网络结构，融合卷积、Transformer和MLP，提升检测效果，并集成多尺度特征聚合与延迟感知策略，优化效率与精度。**

- **链接: [http://arxiv.org/pdf/2510.06669v1](http://arxiv.org/pdf/2510.06669v1)**

> **作者:** Yuxi Liu; Yunfeng Ma; Yi Tang; Min Liu; Shuai Jiang; Yaonan Wang
>
> **摘要:** Industrial surface defect detection (SDD) is critical for ensuring product quality and manufacturing reliability. Due to the diverse shapes and sizes of surface defects, SDD faces two main challenges: intraclass difference and interclass similarity. Existing methods primarily utilize manually designed models, which require extensive trial and error and often struggle to address both challenges effectively. To overcome this, we propose AutoNAD, an automated neural architecture design framework for SDD that jointly searches over convolutions, transformers, and multi-layer perceptrons. This hybrid design enables the model to capture both fine-grained local variations and long-range semantic context, addressing the two key challenges while reducing the cost of manual network design. To support efficient training of such a diverse search space, AutoNAD introduces a cross weight sharing strategy, which accelerates supernet convergence and improves subnet performance. Additionally, a searchable multi-level feature aggregation module (MFAM) is integrated to enhance multi-scale feature learning. Beyond detection accuracy, runtime efficiency is essential for industrial deployment. To this end, AutoNAD incorporates a latency-aware prior to guide the selection of efficient architectures. The effectiveness of AutoNAD is validated on three industrial defect datasets and further applied within a defect imaging and detection platform. Code will be available at https://github.com/Yuxi104/AutoNAD.
>
---
#### [new 080] Few-Shot Adaptation Benchmark for Remote Sensing Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于遥感视觉语言模型任务，旨在解决小样本适应能力不足的问题。作者构建了首个系统基准，评估十种遥感场景分类数据集上五种小样本方法在三种先进模型上的表现，发现不同模型适应性差异显著，并提供开源框架推动后续研究。**

- **链接: [http://arxiv.org/pdf/2510.07135v1](http://arxiv.org/pdf/2510.07135v1)**

> **作者:** Karim El Khoury; Maxime Zanella; Christophe De Vleeschouwer; Benoit Macq
>
> **摘要:** Remote Sensing Vision-Language Models (RSVLMs) have shown remarkable potential thanks to large-scale pretraining, achieving strong zero-shot performance on various tasks. However, their ability to generalize in low-data regimes, such as few-shot learning, remains insufficiently explored. In this work, we present the first structured benchmark for evaluating few-shot adaptation methods on RSVLMs. We conduct comprehensive experiments across ten remote sensing scene classification datasets, applying five widely used few-shot adaptation strategies to three state-of-the-art RSVLMs with varying backbones. Our findings reveal that models with similar zero-shot performance can exhibit markedly different behavior under few-shot adaptation, with some RSVLMs being inherently more amenable to such adaptation than others. The variability of performance and the absence of a clear winner among existing methods highlight the need for the development of more robust methods for few-shot adaptation tailored to RS. To facilitate future research, we provide a reproducible benchmarking framework and open-source code to systematically evaluate RSVLMs under few-shot conditions. The source code is publicly available on Github: https://github.com/elkhouryk/fewshot_RSVLMs
>
---
#### [new 081] Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities
- **分类: cs.CV; cs.AI; cs.CL; 68T50**

- **简介: 该论文属于历史文献OCR评估任务，旨在解决现有OCR评估方法无法准确衡量历史文本数字化质量的问题。论文提出了新的评估框架，引入了HCPR和AIR等新指标，评估了12个多模态LLM模型，发现Gemini和Qwen表现较好，但也存在“过度历史化”问题。**

- **链接: [http://arxiv.org/pdf/2510.06743v1](http://arxiv.org/pdf/2510.06743v1)**

> **作者:** Maria Levchenko
>
> **备注:** The First Workshop on Natural Language Processing and Language Models for Digital Humanities (LM4DH 2025). RANLP 2025
>
> **摘要:** Digital humanities scholars increasingly use Large Language Models for historical document digitization, yet lack appropriate evaluation frameworks for LLM-based OCR. Traditional metrics fail to capture temporal biases and period-specific errors crucial for historical corpus creation. We present an evaluation methodology for LLM-based historical OCR, addressing contamination risks and systematic biases in diplomatic transcription. Using 18th-century Russian Civil font texts, we introduce novel metrics including Historical Character Preservation Rate (HCPR) and Archaic Insertion Rate (AIR), alongside protocols for contamination control and stability testing. We evaluate 12 multimodal LLMs, finding that Gemini and Qwen models outperform traditional OCR while exhibiting over-historicization: inserting archaic characters from incorrect historical periods. Post-OCR correction degrades rather than improves performance. Our methodology provides digital humanities practitioners with guidelines for model selection and quality assessment in historical corpus digitization.
>
---
#### [new 082] Learning Global Representation from Queries for Vectorized HD Map Construction
- **分类: cs.CV; cs.AI**

- **简介: 论文任务是矢量化高精地图构建，旨在解决现有方法忽略全局表示的问题。作者提出MapGR模型，包含全局表示学习和引导模块，提升地图构建精度，在nuScenes和Argoverse2数据集上验证了效果。**

- **链接: [http://arxiv.org/pdf/2510.06969v1](http://arxiv.org/pdf/2510.06969v1)**

> **作者:** Shoumeng Qiu; Xinrun Li; Yang Long; Xiangyang Xue; Varun Ojha; Jian Pu
>
> **备注:** 16 pages
>
> **摘要:** The online construction of vectorized high-definition (HD) maps is a cornerstone of modern autonomous driving systems. State-of-the-art approaches, particularly those based on the DETR framework, formulate this as an instance detection problem. However, their reliance on independent, learnable object queries results in a predominantly local query perspective, neglecting the inherent global representation within HD maps. In this work, we propose \textbf{MapGR} (\textbf{G}lobal \textbf{R}epresentation learning for HD \textbf{Map} construction), an architecture designed to learn and utilize a global representations from queries. Our method introduces two synergistic modules: a Global Representation Learning (GRL) module, which encourages the distribution of all queries to better align with the global map through a carefully designed holistic segmentation task, and a Global Representation Guidance (GRG) module, which endows each individual query with explicit, global-level contextual information to facilitate its optimization. Evaluations on the nuScenes and Argoverse2 datasets validate the efficacy of our approach, demonstrating substantial improvements in mean Average Precision (mAP) compared to leading baselines.
>
---
#### [new 083] A deep multiple instance learning approach based on coarse labels for high-resolution land-cover mapping
- **分类: cs.CV**

- **简介: 该论文属于高分辨率地表覆盖分类任务，旨在解决弱标签数据下的分类问题。利用低分辨率标签训练高分辨率遥感影像分类器，通过深度多示例学习方法，隐式学习高分辨率标签。实验验证了所提方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.06769v1](http://arxiv.org/pdf/2510.06769v1)**

> **作者:** Gianmarco Perantoni; Lorenzo Bruzzone
>
> **备注:** 14 pages, 4 figures, accepted conference paper at SPIE REMOTE SENSING, 3-7 September 2023, Amsterdam, Netherlands
>
> **摘要:** The quantity and the quality of the training labels are central problems in high-resolution land-cover mapping with machine-learning-based solutions. In this context, weak labels can be gathered in large quantities by leveraging on existing low-resolution or obsolete products. In this paper, we address the problem of training land-cover classifiers using high-resolution imagery (e.g., Sentinel-2) and weak low-resolution reference data (e.g., MODIS -derived land-cover maps). Inspired by recent works in Deep Multiple Instance Learning (DMIL), we propose a method that trains pixel-level multi-class classifiers and predicts low-resolution labels (i.e., patch-level classification), where the actual high-resolution labels are learned implicitly without direct supervision. This is achieved with flexible pooling layers that are able to link the semantics of the pixels in the high-resolution imagery to the low-resolution reference labels. Then, the Multiple Instance Learning (MIL) problem is re-framed in a multi-class and in a multi-label setting. In the former, the low-resolution annotation represents the majority of the pixels in the patch. In the latter, the annotation only provides us information on the presence of one of the land-cover classes in the patch and thus multiple labels can be considered valid for a patch at a time, whereas the low-resolution labels provide us only one label. Therefore, the classifier is trained with a Positive-Unlabeled Learning (PUL) strategy. Experimental results on the 2020 IEEE GRSS Data Fusion Contest dataset show the effectiveness of the proposed framework compared to standard training strategies.
>
---
#### [new 084] SCas4D: Structural Cascaded Optimization for Boosting Persistent 4D Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于动态场景建模任务，旨在解决持久动态场景中跟踪与新视角合成的难题。通过提出SCas4D级联优化框架，利用3D高斯点群中的结构模式，从粗到细逐步优化形变，显著提升计算效率与效果，在少得多的训练迭代中达到现有方法水平，并适用于关节物体分割、新视角合成等任务。**

- **链接: [http://arxiv.org/pdf/2510.06694v1](http://arxiv.org/pdf/2510.06694v1)**

> **作者:** Jipeng Lyu; Jiahua Dong; Yu-Xiong Wang
>
> **备注:** Published in Transactions on Machine Learning Research (06/2025)
>
> **摘要:** Persistent dynamic scene modeling for tracking and novel-view synthesis remains challenging due to the difficulty of capturing accurate deformations while maintaining computational efficiency. We propose SCas4D, a cascaded optimization framework that leverages structural patterns in 3D Gaussian Splatting for dynamic scenes. The key idea is that real-world deformations often exhibit hierarchical patterns, where groups of Gaussians share similar transformations. By progressively refining deformations from coarse part-level to fine point-level, SCas4D achieves convergence within 100 iterations per time frame and produces results comparable to existing methods with only one-twentieth of the training iterations. The approach also demonstrates effectiveness in self-supervised articulated object segmentation, novel view synthesis, and dense point tracking tasks.
>
---
#### [new 085] GenPilot: A Multi-Agent System for Test-Time Prompt Optimization in Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂提示词生成效果差、现有优化方法不足的问题。作者提出了GenPilot，一种模型无关的多智能体系统，通过错误分析、自适应探索、精细验证和记忆模块实现测试时提示优化，提升生成图像的语义一致性和结构完整性。**

- **链接: [http://arxiv.org/pdf/2510.07217v1](http://arxiv.org/pdf/2510.07217v1)**

> **作者:** Wen Ye; Zhaocheng Liu; Yuwei Gui; Tingyu Yuan; Yunyue Su; Bowen Fang; Chaoyang Zhao; Qiang Liu; Liang Wang
>
> **备注:** 30 pages, 21 figures, accepted to EMNLP 2025 findings
>
> **摘要:** Text-to-image synthesis has made remarkable progress, yet accurately interpreting complex and lengthy prompts remains challenging, often resulting in semantic inconsistencies and missing details. Existing solutions, such as fine-tuning, are model-specific and require training, while prior automatic prompt optimization (APO) approaches typically lack systematic error analysis and refinement strategies, resulting in limited reliability and effectiveness. Meanwhile, test-time scaling methods operate on fixed prompts and on noise or sample numbers, limiting their interpretability and adaptability. To solve these, we introduce a flexible and efficient test-time prompt optimization strategy that operates directly on the input text. We propose a plug-and-play multi-agent system called GenPilot, integrating error analysis, clustering-based adaptive exploration, fine-grained verification, and a memory module for iterative optimization. Our approach is model-agnostic, interpretable, and well-suited for handling long and complex prompts. Simultaneously, we summarize the common patterns of errors and the refinement strategy, offering more experience and encouraging further exploration. Experiments on DPG-bench and Geneval with improvements of up to 16.9% and 5.7% demonstrate the strong capability of our methods in enhancing the text and image consistency and structural coherence of generated images, revealing the effectiveness of our test-time prompt optimization strategy. The code is available at https://github.com/27yw/GenPilot.
>
---
#### [new 086] multimodars: A Rust-powered toolkit for multi-modality cardiac image fusion and registration
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像处理任务，旨在解决多模态心脏影像（如血管内成像与CCTA）融合与配准问题。现有方法缺乏灵活、高效的开源工具支持多状态分析。作者开发了multimodars，一个基于Rust的高性能工具包，实现确定性配准、支持可重复实验，并提供与AIVUS-CAA等软件的兼容接口。**

- **链接: [http://arxiv.org/pdf/2510.06241v1](http://arxiv.org/pdf/2510.06241v1)**

> **作者:** Anselm W. Stark; Marc Ilic; Ali Mokhtari; Pooya Mohammadi Kazaj; Christoph Graeni; Isaac Shiri
>
> **摘要:** Combining complementary imaging modalities is critical to build reliable 3D coronary models: intravascular imaging gives sub-millimetre resolution but limited whole-vessel context, while CCTA supplies 3D geometry but suffers from limited spatial resolution and artefacts (e.g., blooming). Prior work demonstrated intravascular/CCTA fusion, yet no open, flexible toolkit is tailored for multi-state analysis (rest/stress, pre-/post-stenting) while offering deterministic behaviour, high performance, and easy pipeline integration. multimodars addresses this gap with deterministic alignment algorithms, a compact NumPy-centred data model, and an optimised Rust backend suitable for scalable, reproducible experiments. The package accepts CSV/NumPy inputs including data formats produced by the AIVUS-CAA software
>
---
#### [new 087] Scalable deep fusion of spaceborne lidar and synthetic aperture radar for global forest structural complexity mapping
- **分类: cs.CV; cs.LG; stat.AP**

- **简介: 该论文属于遥感与生态监测任务，旨在解决森林结构复杂性难以连续高分辨率制图的问题。作者提出一种融合星载激光雷达与合成孔径雷达的深度学习框架，实现了全球25米分辨率的森林结构复杂性多时相制图，并支持扩展预测其他森林变量。**

- **链接: [http://arxiv.org/pdf/2510.06299v1](http://arxiv.org/pdf/2510.06299v1)**

> **作者:** Tiago de Conto; John Armston; Ralph Dubayah
>
> **摘要:** Forest structural complexity metrics integrate multiple canopy attributes into a single value that reflects habitat quality and ecosystem function. Spaceborne lidar from the Global Ecosystem Dynamics Investigation (GEDI) has enabled mapping of structural complexity in temperate and tropical forests, but its sparse sampling limits continuous high-resolution mapping. We present a scalable, deep learning framework fusing GEDI observations with multimodal Synthetic Aperture Radar (SAR) datasets to produce global, high-resolution (25 m) wall-to-wall maps of forest structural complexity. Our adapted EfficientNetV2 architecture, trained on over 130 million GEDI footprints, achieves high performance (global R2 = 0.82) with fewer than 400,000 parameters, making it an accessible tool that enables researchers to process datasets at any scale without requiring specialized computing infrastructure. The model produces accurate predictions with calibrated uncertainty estimates across biomes and time periods, preserving fine-scale spatial patterns. It has been used to generate a global, multi-temporal dataset of forest structural complexity from 2015 to 2022. Through transfer learning, this framework can be extended to predict additional forest structural variables with minimal computational cost. This approach supports continuous, multi-temporal monitoring of global forest structural dynamics and provides tools for biodiversity conservation and ecosystem management efforts in a changing climate.
>
---
#### [new 088] CML-Bench: A Framework for Evaluating and Enhancing LLM-Powered Movie Scripts Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于自然语言处理与电影脚本生成任务，旨在解决大语言模型（LLMs）在生成电影剧本时缺乏情感深度与叙事连贯性的问题。作者构建了CML-Dataset和评估框架CML-Bench，并提出CML-Instruction提升LLMs生成质量。**

- **链接: [http://arxiv.org/pdf/2510.06231v1](http://arxiv.org/pdf/2510.06231v1)**

> **作者:** Mingzhe Zheng; Dingjie Song; Guanyu Zhou; Jun You; Jiahao Zhan; Xuran Ma; Xinyuan Song; Ser-Nam Lim; Qifeng Chen; Harry Yang
>
> **备注:** 24 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency in generating highly structured texts. However, while exhibiting a high degree of structural organization, movie scripts demand an additional layer of nuanced storytelling and emotional depth-the 'soul' of compelling cinema-that LLMs often fail to capture. To investigate this deficiency, we first curated CML-Dataset, a dataset comprising (summary, content) pairs for Cinematic Markup Language (CML), where 'content' consists of segments from esteemed, high-quality movie scripts and 'summary' is a concise description of the content. Through an in-depth analysis of the intrinsic multi-shot continuity and narrative structures within these authentic scripts, we identified three pivotal dimensions for quality assessment: Dialogue Coherence (DC), Character Consistency (CC), and Plot Reasonableness (PR). Informed by these findings, we propose the CML-Bench, featuring quantitative metrics across these dimensions. CML-Bench effectively assigns high scores to well-crafted, human-written scripts while concurrently pinpointing the weaknesses in screenplays generated by LLMs. To further validate our benchmark, we introduce CML-Instruction, a prompting strategy with detailed instructions on character dialogue and event logic, to guide LLMs to generate more structured and cinematically sound scripts. Extensive experiments validate the effectiveness of our benchmark and demonstrate that LLMs guided by CML-Instruction generate higher-quality screenplays, with results aligned with human preferences.
>
---
#### [new 089] A Total Variation Regularized Framework for Epilepsy-Related MRI Image Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决癫痫相关脑MRI中局灶性皮质发育不良（FCD）病变难以检测的问题。作者提出一种结合Dice损失和各向异性全变分（TV）项的新损失函数，提升分割精度与平滑性。模型在85例患者数据上验证，表现优于基线方法，显著减少假阳性。**

- **链接: [http://arxiv.org/pdf/2510.06276v1](http://arxiv.org/pdf/2510.06276v1)**

> **作者:** Mehdi Rabiee; Sergio Greco; Reza Shahbazian; Irina Trubitsyna
>
> **摘要:** Focal Cortical Dysplasia (FCD) is a primary cause of drug-resistant epilepsy and is difficult to detect in brain {magnetic resonance imaging} (MRI) due to the subtle and small-scale nature of its lesions. Accurate segmentation of FCD regions in 3D multimodal brain MRI images is essential for effective surgical planning and treatment. However, this task remains highly challenging due to the limited availability of annotated FCD datasets, the extremely small size and weak contrast of FCD lesions, the complexity of handling 3D multimodal inputs, and the need for output smoothness and anatomical consistency, which is often not addressed by standard voxel-wise loss functions. This paper presents a new framework for segmenting FCD regions in 3D brain MRI images. We adopt state-of-the-art transformer-enhanced encoder-decoder architecture and introduce a novel loss function combining Dice loss with an anisotropic {Total Variation} (TV) term. This integration encourages spatial smoothness and reduces false positive clusters without relying on post-processing. The framework is evaluated on a public FCD dataset with 85 epilepsy patients and demonstrates superior segmentation accuracy and consistency compared to standard loss formulations. The model with the proposed TV loss shows an 11.9\% improvement on the Dice coefficient and 13.3\% higher precision over the baseline model. Moreover, the number of false positive clusters is reduced by 61.6%
>
---
#### [new 090] GPT-5 Model Corrected GPT-4V's Chart Reading Errors, Not Prompting
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文属于图表理解任务，旨在解决GPT-4V在图表阅读中的错误问题。研究对比了GPT-5与GPT-4V在107个可视化问题上的推理准确率，发现GPT-5在模型架构上显著提升了准确性，而提示变体效果有限。**

- **链接: [http://arxiv.org/pdf/2510.06782v1](http://arxiv.org/pdf/2510.06782v1)**

> **作者:** Kaichun Yang; Jian Chen
>
> **摘要:** We present a quantitative evaluation to understand the effect of zero-shot large-language model (LLMs) and prompting uses on chart reading tasks. We asked LLMs to answer 107 visualization questions to compare inference accuracies between the agentic GPT-5 and multimodal GPT-4V, for difficult image instances, where GPT-4V failed to produce correct answers. Our results show that model architecture dominates the inference accuracy: GPT5 largely improved accuracy, while prompt variants yielded only small effects. Pre-registration of this work is available here: https://osf.io/u78td/?view_only=6b075584311f48e991c39335c840ded3; the Google Drive materials are here:https://drive.google.com/file/d/1ll8WWZDf7cCNcfNWrLViWt8GwDNSvVrp/view.
>
---
#### [new 091] SER-Diff: Synthetic Error Replay Diffusion for Incremental Brain Tumor Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出SER-Diff，属于增量学习任务，旨在解决脑肿瘤分割模型在适应新数据时遗忘旧知识的问题。通过结合扩散模型生成合成误差图进行回放，并采用双损失训练，有效缓解灾难性遗忘，提升分割准确性和一致性。**

- **链接: [http://arxiv.org/pdf/2510.06283v1](http://arxiv.org/pdf/2510.06283v1)**

> **作者:** Sashank Makanaboyina
>
> **摘要:** Incremental brain tumor segmentation is critical for models that must adapt to evolving clinical datasets without retraining on all prior data. However, catastrophic forgetting, where models lose previously acquired knowledge, remains a major obstacle. Recent incremental learning frameworks with knowledge distillation partially mitigate forgetting but rely heavily on generative replay or auxiliary storage. Meanwhile, diffusion models have proven effective for refining tumor segmentations, but have not been explored in incremental learning contexts. We propose Synthetic Error Replay Diffusion (SER-Diff), the first framework that unifies diffusion-based refinement with incremental learning. SER-Diff leverages a frozen teacher diffusion model to generate synthetic error maps from past tasks, which are replayed during training on new tasks. A dual-loss formulation combining Dice loss for new data and knowledge distillation loss for replayed errors ensures both adaptability and retention. Experiments on BraTS2020, BraTS2021, and BraTS2023 demonstrate that SER-Diff consistently outperforms prior methods. It achieves the highest Dice scores of 95.8\%, 94.9\%, and 94.6\%, along with the lowest HD95 values of 4.4 mm, 4.7 mm, and 4.9 mm, respectively. These results indicate that SER-Diff not only mitigates catastrophic forgetting but also delivers more accurate and anatomically coherent segmentations across evolving datasets.
>
---
#### [new 092] Active Next-Best-View Optimization for Risk-Averse Path Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人路径规划任务，旨在解决不确定环境中安全导航问题。通过结合风险规避与主动感知，提出一种统一框架：利用3D高斯辐射场构建风险地图，并在SE(3)流形上优化下一最佳视角，以提升路径安全性与感知效率。**

- **链接: [http://arxiv.org/pdf/2510.06481v1](http://arxiv.org/pdf/2510.06481v1)**

> **作者:** Amirhossein Mollaei Khass; Guangyi Liu; Vivek Pandey; Wen Jiang; Boshu Lei; Kostas Daniilidis; Nader Motee
>
> **摘要:** Safe navigation in uncertain environments requires planning methods that integrate risk aversion with active perception. In this work, we present a unified framework that refines a coarse reference path by constructing tail-sensitive risk maps from Average Value-at-Risk statistics on an online-updated 3D Gaussian-splat Radiance Field. These maps enable the generation of locally safe and feasible trajectories. In parallel, we formulate Next-Best-View (NBV) selection as an optimization problem on the SE(3) pose manifold, where Riemannian gradient descent maximizes an expected information gain objective to reduce uncertainty most critical for imminent motion. Our approach advances the state-of-the-art by coupling risk-averse path refinement with NBV planning, while introducing scalable gradient decompositions that support efficient online updates in complex environments. We demonstrate the effectiveness of the proposed framework through extensive computational studies.
>
---
#### [new 093] Introspection in Learned Semantic Scene Graph Localisation
- **分类: cs.LG; cs.AI; cs.CV; cs.RO; I.2.10; I.2.9; I.4.8; I.5.2; I.5.1**

- **简介: 该论文属于语义定位任务，旨在研究语义如何影响定位性能与鲁棒性。作者在自监督对比学习框架下训练定位网络，并通过事后分析探究模型是否过滤环境噪声、关注显著地标。他们验证了解释性方法的可靠性，发现集成梯度与注意力权重最有效，并揭示模型隐式弱化常见物体的权重，最终实现对场景定义的可解释、鲁棒定位。**

- **链接: [http://arxiv.org/pdf/2510.07053v1](http://arxiv.org/pdf/2510.07053v1)**

> **作者:** Manshika Charvi Bissessur; Efimia Panagiotaki; Daniele De Martini
>
> **备注:** IEEE IROS 2025 Workshop FAST
>
> **摘要:** This work investigates how semantics influence localisation performance and robustness in a learned self-supervised, contrastive semantic localisation framework. After training a localisation network on both original and perturbed maps, we conduct a thorough post-hoc introspection analysis to probe whether the model filters environmental noise and prioritises distinctive landmarks over routine clutter. We validate various interpretability methods and present a comparative reliability analysis. Integrated gradients and Attention Weights consistently emerge as the most reliable probes of learned behaviour. A semantic class ablation further reveals an implicit weighting in which frequent objects are often down-weighted. Overall, the results indicate that the model learns noise-robust, semantically salient relations about place definition, thereby enabling explainable registration under challenging visual and structural variations.
>
---
#### [new 094] StruSR: Structure-Aware Symbolic Regression with Physics-Informed Taylor Guidance
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于符号回归任务，旨在从时间序列数据中发现符合物理规律的解析表达式。传统方法难以提取系统结构先验，导致表达式不够准确。论文提出StruSR，利用物理信息神经网络（PINN）提取局部结构先验，结合泰勒展开与遗传编程，提升表达式的物理一致性与结构准确性，优化符号回归的收敛速度与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.06635v1](http://arxiv.org/pdf/2510.06635v1)**

> **作者:** Yunpeng Gong; Sihan Lan; Can Yang; Kunpeng Xu; Min Jiang
>
> **摘要:** Symbolic regression aims to find interpretable analytical expressions by searching over mathematical formula spaces to capture underlying system behavior, particularly in scientific modeling governed by physical laws. However, traditional methods lack mechanisms for extracting structured physical priors from time series observations, making it difficult to capture symbolic expressions that reflect the system's global behavior. In this work, we propose a structure-aware symbolic regression framework, called StruSR, that leverages trained Physics-Informed Neural Networks (PINNs) to extract locally structured physical priors from time series data. By performing local Taylor expansions on the outputs of the trained PINN, we obtain derivative-based structural information to guide symbolic expression evolution. To assess the importance of expression components, we introduce a masking-based attribution mechanism that quantifies each subtree's contribution to structural alignment and physical residual reduction. These sensitivity scores steer mutation and crossover operations within genetic programming, preserving substructures with high physical or structural significance while selectively modifying less informative components. A hybrid fitness function jointly minimizes physics residuals and Taylor coefficient mismatch, ensuring consistency with both the governing equations and the local analytical behavior encoded by the PINN. Experiments on benchmark PDE systems demonstrate that StruSR improves convergence speed, structural fidelity, and expression interpretability compared to conventional baselines, offering a principled paradigm for physics-grounded symbolic discovery.
>
---
#### [new 095] Stacked Regression using Off-the-shelf, Stimulus-tuned and Fine-tuned Neural Networks for Predicting fMRI Brain Responses to Movies (Algonauts 2025 Report)
- **分类: eess.IV; cs.AI; cs.CV; q-bio.NC**

- **简介: 该论文属于脑活动预测任务，旨在解决根据电影刺激预测fMRI脑响应的问题。作者结合多模态表示（包括语言模型、视觉模型等），使用堆叠回归融合各模型预测结果，并通过调整文本输入、模型调优提升性能，最终在挑战赛中排名第十。**

- **链接: [http://arxiv.org/pdf/2510.06235v1](http://arxiv.org/pdf/2510.06235v1)**

> **作者:** Robert Scholz; Kunal Bagga; Christine Ahrends; Carlo Alberto Barbano
>
> **摘要:** We present our submission to the Algonauts 2025 Challenge, where the goal is to predict fMRI brain responses to movie stimuli. Our approach integrates multimodal representations from large language models, video encoders, audio models, and vision-language models, combining both off-the-shelf and fine-tuned variants. To improve performance, we enhanced textual inputs with detailed transcripts and summaries, and we explored stimulus-tuning and fine-tuning strategies for language and vision models. Predictions from individual models were combined using stacked regression, yielding solid results. Our submission, under the team name Seinfeld, ranked 10th. We make all code and resources publicly available, contributing to ongoing efforts in developing multimodal encoding models for brain activity.
>
---
#### [new 096] Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人领域的综述任务，旨在解决如何将视觉、语言和动作统一应用于机器人系统以提升泛化能力的问题。论文系统回顾了VLA模型的架构、学习方法、数据策略及硬件平台，提供实际部署指导，并整理了相关资源，助力真实场景应用。**

- **链接: [http://arxiv.org/pdf/2510.07077v1](http://arxiv.org/pdf/2510.07077v1)**

> **作者:** Kento Kawaharazuka; Jihoon Oh; Jun Yamada; Ingmar Posner; Yuke Zhu
>
> **备注:** Accepted to IEEE Access, website: https://vla-survey.github.io
>
> **摘要:** Amid growing efforts to leverage advances in large language models (LLMs) and vision-language models (VLMs) for robotics, Vision-Language-Action (VLA) models have recently gained significant attention. By unifying vision, language, and action data at scale, which have traditionally been studied separately, VLA models aim to learn policies that generalise across diverse tasks, objects, embodiments, and environments. This generalisation capability is expected to enable robots to solve novel downstream tasks with minimal or no additional task-specific data, facilitating more flexible and scalable real-world deployment. Unlike previous surveys that focus narrowly on action representations or high-level model architectures, this work offers a comprehensive, full-stack review, integrating both software and hardware components of VLA systems. In particular, this paper provides a systematic review of VLAs, covering their strategy and architectural transition, architectures and building blocks, modality-specific processing techniques, and learning paradigms. In addition, to support the deployment of VLAs in real-world robotic applications, we also review commonly used robot platforms, data collection strategies, publicly available datasets, data augmentation methods, and evaluation benchmarks. Throughout this comprehensive survey, this paper aims to offer practical guidance for the robotics community in applying VLAs to real-world robotic systems. All references categorized by training approach, evaluation method, modality, and dataset are available in the table on our project website: https://vla-survey.github.io .
>
---
#### [new 097] SaFeR-VLM: Toward Safety-aware Fine-grained Reasoning in Multimodal Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于多模态模型安全推理任务，旨在解决模型在面对对抗性或不安全提示时的安全风险问题。论文提出了SaFeR-VLM框架，通过安全感知的数据集、反思机制、奖励建模和优化方法，将安全性融入推理过程，提升了模型在安全性和帮助性上的表现。**

- **链接: [http://arxiv.org/pdf/2510.06871v1](http://arxiv.org/pdf/2510.06871v1)**

> **作者:** Huahui Yi; Kun Wang; Qiankun Li; Miao Yu; Liang Lin; Gongli Xi; Hao Wu; Xuming Hu; Kang Li; Yang Liu
>
> **摘要:** Multimodal Large Reasoning Models (MLRMs) demonstrate impressive cross-modal reasoning but often amplify safety risks under adversarial or unsafe prompts, a phenomenon we call the \textit{Reasoning Tax}. Existing defenses mainly act at the output level and do not constrain the reasoning process, leaving models exposed to implicit risks. In this paper, we propose SaFeR-VLM, a safety-aligned reinforcement learning framework that embeds safety directly into multimodal reasoning. The framework integrates four components: (I) QI-Safe-10K, a curated dataset emphasizing safety-critical and reasoning-sensitive cases; (II) safety-aware rollout, where unsafe generations undergo reflection and correction instead of being discarded; (III) structured reward modeling with multi-dimensional weighted criteria and explicit penalties for hallucinations and contradictions; and (IV) GRPO optimization, which reinforces both safe and corrected trajectories. This unified design shifts safety from a passive safeguard to an active driver of reasoning, enabling scalable and generalizable safety-aware reasoning. SaFeR-VLM further demonstrates robustness against both explicit and implicit risks, supporting dynamic and interpretable safety decisions beyond surface-level filtering. SaFeR-VLM-3B achieves average performance $70.13$ and $78.97$ on safety and helpfulness across six benchmarks, surpassing both same-scale and $>10\times$ larger models such as Skywork-R1V3-38B, Qwen2.5VL-72B, and GLM4.5V-106B. Remarkably, SaFeR-VLM-7B benefits from its increased scale to surpass GPT-5-mini and Gemini-2.5-Flash by \num{6.47} and \num{16.76} points respectively on safety metrics, achieving this improvement without any degradation in helpfulness performance. Our codes are available at https://github.com/HarveyYi/SaFeR-VLM.
>
---
#### [new 098] Unsupervised Backdoor Detection and Mitigation for Spiking Neural Networks
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 该论文属于安全任务，旨在解决脉冲神经网络（SNN）中隐蔽的后门攻击问题。作者提出了一种无监督的后门检测框架TMPBD和一种缓解机制NDSBM，分别用于检测和抑制恶意行为，实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2510.06629v1](http://arxiv.org/pdf/2510.06629v1)**

> **作者:** Jiachen Li; Bang Wu; Xiaoyu Xia; Xiaoning Liu; Xun Yi; Xiuzhen Zhang
>
> **备注:** To appear in The 28th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2025)
>
> **摘要:** Spiking Neural Networks (SNNs) have gained increasing attention for their superior energy efficiency compared to Artificial Neural Networks (ANNs). However, their security aspects, particularly under backdoor attacks, have received limited attention. Existing defense methods developed for ANNs perform poorly or can be easily bypassed in SNNs due to their event-driven and temporal dependencies. This paper identifies the key blockers that hinder traditional backdoor defenses in SNNs and proposes an unsupervised post-training detection framework, Temporal Membrane Potential Backdoor Detection (TMPBD), to overcome these challenges. TMPBD leverages the maximum margin statistics of temporal membrane potential (TMP) in the final spiking layer to detect target labels without any attack knowledge or data access. We further introduce a robust mitigation mechanism, Neural Dendrites Suppression Backdoor Mitigation (NDSBM), which clamps dendritic connections between early convolutional layers to suppress malicious neurons while preserving benign behaviors, guided by TMP extracted from a small, clean, unlabeled dataset. Extensive experiments on multiple neuromorphic benchmarks and state-of-the-art input-aware dynamic trigger attacks demonstrate that TMPBD achieves 100% detection accuracy, while NDSBM reduces the attack success rate from 100% to 8.44%, and to 2.81% when combined with detection, without degrading clean accuracy.
>
---
#### [new 099] Real-Time Glass Detection and Reprojection using Sensor Fusion Onboard Aerial Robots
- **分类: cs.RO; cs.CV; cs.SY; eess.SY**

- **简介: 论文任务为透明障碍物实时检测与地图构建。针对无人机在透明物体前导航困难的问题，提出融合ToF相机与超声波传感器的轻量级方案，实现低功耗无人机上实时透明障碍物感知。**

- **链接: [http://arxiv.org/pdf/2510.06518v1](http://arxiv.org/pdf/2510.06518v1)**

> **作者:** Malakhi Hopkins; Varun Murali; Vijay Kumar; Camillo J Taylor
>
> **备注:** 8 pages, 8 figures, submitted to ICRA 2026
>
> **摘要:** Autonomous aerial robots are increasingly being deployed in real-world scenarios, where transparent obstacles present significant challenges to reliable navigation and mapping. These materials pose a unique problem for traditional perception systems because they lack discernible features and can cause conventional depth sensors to fail, leading to inaccurate maps and potential collisions. To ensure safe navigation, robots must be able to accurately detect and map these transparent obstacles. Existing methods often rely on large, expensive sensors or algorithms that impose high computational burdens, making them unsuitable for low Size, Weight, and Power (SWaP) robots. In this work, we propose a novel and computationally efficient framework for detecting and mapping transparent obstacles onboard a sub-300g quadrotor. Our method fuses data from a Time-of-Flight (ToF) camera and an ultrasonic sensor with a custom, lightweight 2D convolution model. This specialized approach accurately detects specular reflections and propagates their depth into corresponding empty regions of the depth map, effectively rendering transparent obstacles visible. The entire pipeline operates in real-time, utilizing only a small fraction of a CPU core on an embedded processor. We validate our system through a series of experiments in both controlled and real-world environments, demonstrating the utility of our method through experiments where the robot maps indoor environments containing glass. Our work is, to our knowledge, the first of its kind to demonstrate a real-time, onboard transparent obstacle mapping system on a low-SWaP quadrotor using only the CPU.
>
---
#### [new 100] High-Rate Mixout: Revisiting Mixout for Robust Domain Generalization
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于领域泛化任务，旨在解决模型在分布偏移下泛化能力不足的问题。论文提出“High-Rate Mixout”方法，通过高概率随机替换微调权重以保持预训练知识，减少计算开销，提升跨领域性能。实验表明其效果接近集成方法，但训练成本显著降低。**

- **链接: [http://arxiv.org/pdf/2510.06955v1](http://arxiv.org/pdf/2510.06955v1)**

> **作者:** Masih Aminbeidokhti; Heitor Rapela Medeiros; Eric Granger; Marco Pedersoli
>
> **备注:** WACV 2026: Winter Conference on Applications of Computer Vision 2026
>
> **摘要:** Ensembling fine-tuned models initialized from powerful pre-trained weights is a common strategy to improve robustness under distribution shifts, but it comes with substantial computational costs due to the need to train and store multiple models. Dropout offers a lightweight alternative by simulating ensembles through random neuron deactivation; however, when applied to pre-trained models, it tends to over-regularize and disrupt critical representations necessary for generalization. In this work, we investigate Mixout, a stochastic regularization technique that provides an alternative to Dropout for domain generalization. Rather than deactivating neurons, Mixout mitigates overfitting by probabilistically swapping a subset of fine-tuned weights with their pre-trained counterparts during training, thereby maintaining a balance between adaptation and retention of prior knowledge. Our study reveals that achieving strong performance with Mixout on domain generalization benchmarks requires a notably high masking probability of 0.9 for ViTs and 0.8 for ResNets. While this may seem like a simple adjustment, it yields two key advantages for domain generalization: (1) higher masking rates more strongly penalize deviations from the pre-trained parameters, promoting better generalization to unseen domains; and (2) high-rate masking substantially reduces computational overhead, cutting gradient computation by up to 45% and gradient memory usage by up to 90%. Experiments across five domain generalization benchmarks, PACS, VLCS, OfficeHome, TerraIncognita, and DomainNet, using ResNet and ViT architectures, show that our approach, High-rate Mixout, achieves out-of-domain accuracy comparable to ensemble-based methods while significantly reducing training costs.
>
---
#### [new 101] UniFField: A Generalizable Unified Neural Feature Field for Visual, Semantic, and Spatial Uncertainties in Any Scene
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人感知任务，旨在解决复杂环境中3D场景理解与不确定性建模问题。作者提出UniFField，一种统一的神经特征场，融合视觉、语义与几何特征，并预测各模态的不确定性，支持零样本迁移与增量式场景建模，提升机器人决策鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.06754v1](http://arxiv.org/pdf/2510.06754v1)**

> **作者:** Christian Maurer; Snehal Jauhri; Sophie Lueth; Georgia Chalvatzaki
>
> **备注:** Project website: https://sites.google.com/view/uniffield
>
> **摘要:** Comprehensive visual, geometric, and semantic understanding of a 3D scene is crucial for successful execution of robotic tasks, especially in unstructured and complex environments. Additionally, to make robust decisions, it is necessary for the robot to evaluate the reliability of perceived information. While recent advances in 3D neural feature fields have enabled robots to leverage features from pretrained foundation models for tasks such as language-guided manipulation and navigation, existing methods suffer from two critical limitations: (i) they are typically scene-specific, and (ii) they lack the ability to model uncertainty in their predictions. We present UniFField, a unified uncertainty-aware neural feature field that combines visual, semantic, and geometric features in a single generalizable representation while also predicting uncertainty in each modality. Our approach, which can be applied zero shot to any new environment, incrementally integrates RGB-D images into our voxel-based feature representation as the robot explores the scene, simultaneously updating uncertainty estimation. We evaluate our uncertainty estimations to accurately describe the model prediction errors in scene reconstruction and semantic feature prediction. Furthermore, we successfully leverage our feature predictions and their respective uncertainty for an active object search task using a mobile manipulator robot, demonstrating the capability for robust decision-making.
>
---
#### [new 102] The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于科学机器学习任务，旨在解决机器学习模型在不同分辨率数据上的泛化能力问题。论文发现当前的机器学习操作符（MLOs）无法实现“零样本超分辨率”，即在未训练的高分辨率数据上表现不佳，容易受到混叠效应影响。为解决该问题，作者提出了一种简单高效的数据驱动多分辨率训练方法，有效提升了模型在不同分辨率上的推理能力。**

- **链接: [http://arxiv.org/pdf/2510.06646v1](http://arxiv.org/pdf/2510.06646v1)**

> **作者:** Mansi Sakarvadia; Kareem Hegazy; Amin Totounferoush; Kyle Chard; Yaoqing Yang; Ian Foster; Michael W. Mahoney
>
> **摘要:** A core challenge in scientific machine learning, and scientific computing more generally, is modeling continuous phenomena which (in practice) are represented discretely. Machine-learned operators (MLOs) have been introduced as a means to achieve this modeling goal, as this class of architecture can perform inference at arbitrary resolution. In this work, we evaluate whether this architectural innovation is sufficient to perform "zero-shot super-resolution," namely to enable a model to serve inference on higher-resolution data than that on which it was originally trained. We comprehensively evaluate both zero-shot sub-resolution and super-resolution (i.e., multi-resolution) inference in MLOs. We decouple multi-resolution inference into two key behaviors: 1) extrapolation to varying frequency information; and 2) interpolating across varying resolutions. We empirically demonstrate that MLOs fail to do both of these tasks in a zero-shot manner. Consequently, we find MLOs are not able to perform accurate inference at resolutions different from those on which they were trained, and instead they are brittle and susceptible to aliasing. To address these failure modes, we propose a simple, computationally-efficient, and data-driven multi-resolution training protocol that overcomes aliasing and that provides robust multi-resolution generalization.
>
---
#### [new 103] TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言模型在机器人领域的应用任务，旨在解决现有模型在几何推理中精度不足的问题。论文提出了TIGeR框架，通过集成外部工具实现精确几何计算，提升了机器人操作的厘米级精度。**

- **链接: [http://arxiv.org/pdf/2510.07181v1](http://arxiv.org/pdf/2510.07181v1)**

> **作者:** Yi Han; Cheng Chi; Enshen Zhou; Shanyu Rong; Jingkun An; Pengwei Wang; Zhongyuan Wang; Lu Sheng; Shanghang Zhang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, trajectory generation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks.
>
---
#### [new 104] Bionetta: Efficient Client-Side Zero-Knowledge Machine Learning Proving
- **分类: cs.CR; cs.CV**

- **简介: 论文提出Bionetta，一个基于UltraGroth的零知识机器学习框架，旨在提升客户端证明效率，实现在移动设备上运行。相比EZKL等工具，其优势在于适用于原生EVM智能合约部署，尽管预处理成本较高，但显著降低了证明大小与验证开销，推动隐私保护机器学习应用发展。**

- **链接: [http://arxiv.org/pdf/2510.06784v1](http://arxiv.org/pdf/2510.06784v1)**

> **作者:** Dmytro Zakharov; Oleksandr Kurbatov; Artem Sdobnov; Lev Soukhanov; Yevhenii Sekhin; Vitalii Volovyk; Mykhailo Velykodnyi; Mark Cherepovskyi; Kyrylo Baibula; Lasha Antadze; Pavlo Kravchenko; Volodymyr Dubinin; Yaroslav Panasenko
>
> **摘要:** In this report, we compare the performance of our UltraGroth-based zero-knowledge machine learning framework Bionetta to other tools of similar purpose such as EZKL, Lagrange's deep-prove, or zkml. The results show a significant boost in the proving time for custom-crafted neural networks: they can be proven even on mobile devices, enabling numerous client-side proving applications. While our scheme increases the cost of one-time preprocessing steps, such as circuit compilation and generating trusted setup, our approach is, to the best of our knowledge, the only one that is deployable on the native EVM smart contracts without overwhelming proof size and verification overheads.
>
---
#### [new 105] Revisiting Mixout: An Overlooked Path to Robust Finetuning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉模型微调任务，旨在解决微调过程中在分布偏移下鲁棒性下降的问题。作者重新审视Mixout方法，提出改进的GMixout，通过动态锚点和调节掩码频率增强鲁棒性，实验证明其在多个基准上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.06982v1](http://arxiv.org/pdf/2510.06982v1)**

> **作者:** Masih Aminbeidokhti; Heitor Rapela Medeiros; Eric Granger; Marco Pedersoli
>
> **摘要:** Finetuning vision foundation models often improves in-domain accuracy but comes at the cost of robustness under distribution shift. We revisit Mixout, a stochastic regularizer that intermittently replaces finetuned weights with their pretrained reference, through the lens of a single-run, weight-sharing implicit ensemble. This perspective reveals three key levers that govern robustness: the \emph{masking anchor}, \emph{resampling frequency}, and \emph{mask sparsity}. Guided by this analysis, we introduce GMixout, which (i) replaces the fixed anchor with an exponential moving-average snapshot that adapts during training, and (ii) regulates masking period via an explicit resampling-frequency hyperparameter. Our sparse-kernel implementation updates only a small fraction of parameters with no inference-time overhead, enabling training on consumer-grade GPUs. Experiments on benchmarks covering covariate shift, corruption, and class imbalance, ImageNet / ImageNet-LT, DomainNet, iWildCam, and CIFAR100-C, GMixout consistently improves in-domain accuracy beyond zero-shot performance while surpassing both Model Soups and strong parameter-efficient finetuning baselines under distribution shift.
>
---
#### [new 106] Conditional Denoising Diffusion Model-Based Robust MR Image Reconstruction from Highly Undersampled Data
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像重建任务，旨在解决MRI成像中因数据欠采样导致的图像质量下降问题。作者提出一种基于条件去噪扩散模型的方法，通过在每一步扩散过程中嵌入测量模型并使用配对数据训练，结合生成模型与数据一致性约束，提升了重建图像的质量，在多个指标上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.06335v1](http://arxiv.org/pdf/2510.06335v1)**

> **作者:** Mohammed Alsubaie; Wenxi Liu; Linxia Gu; Ovidiu C. Andronesi; Sirani M. Perera; Xianqi Li
>
> **摘要:** Magnetic Resonance Imaging (MRI) is a critical tool in modern medical diagnostics, yet its prolonged acquisition time remains a critical limitation, especially in time-sensitive clinical scenarios. While undersampling strategies can accelerate image acquisition, they often result in image artifacts and degraded quality. Recent diffusion models have shown promise for reconstructing high-fidelity images from undersampled data by learning powerful image priors; however, most existing approaches either (i) rely on unsupervised score functions without paired supervision or (ii) apply data consistency only as a post-processing step. In this work, we introduce a conditional denoising diffusion framework with iterative data-consistency correction, which differs from prior methods by embedding the measurement model directly into every reverse diffusion step and training the model on paired undersampled-ground truth data. This hybrid design bridges generative flexibility with explicit enforcement of MRI physics. Experiments on the fastMRI dataset demonstrate that our framework consistently outperforms recent state-of-the-art deep learning and diffusion-based methods in SSIM, PSNR, and LPIPS, with LPIPS capturing perceptual improvements more faithfully. These results demonstrate that integrating conditional supervision with iterative consistency updates yields substantial improvements in both pixel-level fidelity and perceptual realism, establishing a principled and practical advance toward robust, accelerated MRI reconstruction.
>
---
#### [new 107] FEAorta: A Fully Automated Framework for Finite Element Analysis of the Aorta From 3D CT Images
- **分类: eess.IV; cs.CE; cs.CV; cs.LG**

- **简介: 该论文属于医学图像处理与生物力学分析任务，旨在解决胸主动脉瘤破裂风险评估中建模耗时和计算负担重的问题。作者开发了FEAorta框架，实现从3D CT图像自动生成主动脉有限元网格，并结合深度学习与FEA加速应力计算，提升临床适用性。**

- **链接: [http://arxiv.org/pdf/2510.06621v1](http://arxiv.org/pdf/2510.06621v1)**

> **作者:** Jiasong Chen; Linchen Qian; Ruonan Gong; Christina Sun; Tongran Qin; Thuy Pham; Caitlin Martin; Mohammad Zafar; John Elefteriades; Wei Sun; Liang Liang
>
> **摘要:** Aortic aneurysm disease ranks consistently in the top 20 causes of death in the U.S. population. Thoracic aortic aneurysm is manifested as an abnormal bulging of thoracic aortic wall and it is a leading cause of death in adults. From the perspective of biomechanics, rupture occurs when the stress acting on the aortic wall exceeds the wall strength. Wall stress distribution can be obtained by computational biomechanical analyses, especially structural Finite Element Analysis. For risk assessment, probabilistic rupture risk of TAA can be calculated by comparing stress with material strength using a material failure model. Although these engineering tools are currently available for TAA rupture risk assessment on patient specific level, clinical adoption has been limited due to two major barriers: labor intensive 3D reconstruction current patient specific anatomical modeling still relies on manual segmentation, making it time consuming and difficult to scale to a large patient population, and computational burden traditional FEA simulations are resource intensive and incompatible with time sensitive clinical workflows. The second barrier was successfully overcome by our team through the development of the PyTorch FEA library and the FEA DNN integration framework. By incorporating the FEA functionalities within PyTorch FEA and applying the principle of static determinacy, we reduced the FEA based stress computation time to approximately three minutes per case. Moreover, by integrating DNN and FEA through the PyTorch FEA library, our approach further decreases the computation time to only a few seconds per case. This work focuses on overcoming the first barrier through the development of an end to end deep neural network capable of generating patient specific finite element meshes of the aorta directly from 3D CT images.
>
---
#### [new 108] TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型在具身视觉跟踪（EVT）任务中的应用。解决现有方法在遮挡或干扰物下跟踪失败的问题，提出TrackVLA++模型，引入空间推理机制（Polar-CoT）和目标识别记忆模块（TIM），提升跟踪的时空连续性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.07134v1](http://arxiv.org/pdf/2510.07134v1)**

> **作者:** Jiahang Liu; Yunpeng Qi; Jiazhao Zhang; Minghan Li; Shaoan Wang; Kui Wu; Hanjing Ye; Hong Zhang; Zhibo Chen; Fangwei Zhong; Zhizheng Zhang; He Wang
>
> **备注:** Project page: https://pku-epic.github.io/TrackVLA-plus-plus-Web/
>
> **摘要:** Embodied Visual Tracking (EVT) is a fundamental ability that underpins practical applications, such as companion robots, guidance robots and service assistants, where continuously following moving targets is essential. Recent advances have enabled language-guided tracking in complex and unstructured scenes. However, existing approaches lack explicit spatial reasoning and effective temporal memory, causing failures under severe occlusions or in the presence of similar-looking distractors. To address these challenges, we present TrackVLA++, a novel Vision-Language-Action (VLA) model that enhances embodied visual tracking with two key modules, a spatial reasoning mechanism and a Target Identification Memory (TIM). The reasoning module introduces a Chain-of-Thought paradigm, termed Polar-CoT, which infers the target's relative position and encodes it as a compact polar-coordinate token for action prediction. Guided by these spatial priors, the TIM employs a gated update strategy to preserve long-horizon target memory, ensuring spatiotemporal consistency and mitigating target loss during extended occlusions. Extensive experiments show that TrackVLA++ achieves state-of-the-art performance on public benchmarks across both egocentric and multi-camera settings. On the challenging EVT-Bench DT split, TrackVLA++ surpasses the previous leading approach by 5.1 and 12, respectively. Furthermore, TrackVLA++ exhibits strong zero-shot generalization, enabling robust real-world tracking in dynamic and occluded scenarios.
>
---
#### [new 109] Capture and Interact: Rapid 3D Object Acquisition and Rendering with Gaussian Splatting in Unity
- **分类: cs.GR; cs.CV**

- **简介: 论文提出了一种基于3D高斯点绘的端到端流程，用于实现从手机视频快速获取三维物体，并在Unity中进行实时交互渲染。该工作属于三维重建与实时渲染任务，旨在解决移动设备上3D内容创建与展示的效率问题。**

- **链接: [http://arxiv.org/pdf/2510.06802v1](http://arxiv.org/pdf/2510.06802v1)**

> **作者:** Islomjon Shukhratov; Sergey Gorinsky
>
> **摘要:** Capturing and rendering three-dimensional (3D) objects in real time remain a significant challenge, yet hold substantial potential for applications in augmented reality, digital twin systems, remote collaboration and prototyping. We present an end-to-end pipeline that leverages 3D Gaussian Splatting (3D GS) to enable rapid acquisition and interactive rendering of real-world objects using a mobile device, cloud processing and a local computer. Users scan an object with a smartphone video, upload it for automated 3D reconstruction, and visualize it interactively in Unity at an average of 150 frames per second (fps) on a laptop. The system integrates mobile capture, cloud-based 3D GS and Unity rendering to support real-time telepresence. Our experiments show that the pipeline processes scans in approximately 10 minutes on a graphics processing unit (GPU) achieving real-time rendering on the laptop.
>
---
#### [new 110] Angular Constraint Embedding via SpherePair Loss for Constrained Clustering
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于**约束聚类任务**，旨在解决现有深度约束聚类方法在端到端建模和欧氏嵌入学习中的局限性。作者提出**SpherePair损失函数**，通过角度空间嵌入保留成对约束，实现更优聚类效果，并具备理论保证。方法无需预设簇数、可扩展性强，适用于真实场景。**

- **链接: [http://arxiv.org/pdf/2510.06907v1](http://arxiv.org/pdf/2510.06907v1)**

> **作者:** Shaojie Zhang; Ke Chen
>
> **备注:** Accepted by NeurIPS 2025, 6 Figures and 1 Table in Main text, 18 Figures and 5 Tables in Appendices
>
> **摘要:** Constrained clustering integrates domain knowledge through pairwise constraints. However, existing deep constrained clustering (DCC) methods are either limited by anchors inherent in end-to-end modeling or struggle with learning discriminative Euclidean embedding, restricting their scalability and real-world applicability. To avoid their respective pitfalls, we propose a novel angular constraint embedding approach for DCC, termed SpherePair. Using the SpherePair loss with a geometric formulation, our method faithfully encodes pairwise constraints and leads to embeddings that are clustering-friendly in angular space, effectively separating representation learning from clustering. SpherePair preserves pairwise relations without conflict, removes the need to specify the exact number of clusters, generalizes to unseen data, enables rapid inference of the number of clusters, and is supported by rigorous theoretical guarantees. Comparative evaluations with state-of-the-art DCC methods on diverse benchmarks, along with empirical validation of theoretical insights, confirm its superior performance, scalability, and overall real-world effectiveness. Code is available at \href{https://github.com/spherepaircc/SpherePairCC/tree/main}{our repository}.
>
---
#### [new 111] On knot detection via picture recognition
- **分类: cs.LG; cs.CV; math.GT; Primary: 57K10, 68T07, secondary: 57K14, 68T45**

- **简介: 该论文属于图像识别与拓扑分类任务，旨在通过图像识别技术自动识别绳结类型。论文结合机器学习（如CNN和Transformer）与传统算法（如Jones多项式），从图像中提取结构信息并预测交叉数，探索了从视觉感知到符号重建的两阶段方法，以实现鲁棒的绳结分类。**

- **链接: [http://arxiv.org/pdf/2510.06284v1](http://arxiv.org/pdf/2510.06284v1)**

> **作者:** Anne Dranowski; Yura Kabkov; Daniel Tubbenhauer
>
> **备注:** 21 pages, many figures, comments welcome
>
> **摘要:** Our goal is to one day take a photo of a knot and have a phone automatically recognize it. In this expository work, we explain a strategy to approximate this goal, using a mixture of modern machine learning methods (in particular convolutional neural networks and transformers for image recognition) and traditional algorithms (to compute quantum invariants like the Jones polynomial). We present simple baselines that predict crossing number directly from images, showing that even lightweight CNN and transformer architectures can recover meaningful structural information. The longer-term aim is to combine these perception modules with symbolic reconstruction into planar diagram (PD) codes, enabling downstream invariant computation for robust knot classification. This two-stage approach highlights the complementarity between machine learning, which handles noisy visual data, and invariants, which enforce rigorous topological distinctions.
>
---
#### [new 112] Control-Augmented Autoregressive Diffusion for Data Assimilation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于数据同化任务，旨在解决现有方法在处理混沌时空偏微分方程时计算成本高且预测易漂移的问题。作者提出了一种控制增强的自回归扩散模型框架，通过引入轻量级控制器网络，在推理过程中进行实时修正，提升了稳定性、准确性和物理保真度。**

- **链接: [http://arxiv.org/pdf/2510.06637v1](http://arxiv.org/pdf/2510.06637v1)**

> **作者:** Prakhar Srivastava; Farrin Marouf Sofian; Francesco Immorlano; Kushagra Pandey; Stephan Mandt
>
> **摘要:** Despite recent advances in test-time scaling and finetuning of diffusion models, guidance in Auto-Regressive Diffusion Models (ARDMs) remains underexplored. We introduce an amortized framework that augments pretrained ARDMs with a lightweight controller network, trained offline by previewing future ARDM rollouts and learning stepwise controls that anticipate upcoming observations under a terminal cost objective. We evaluate this framework in the context of data assimilation (DA) for chaotic spatiotemporal partial differential equations (PDEs), a setting where existing methods are often computationally prohibitive and prone to forecast drift under sparse observations. Our approach reduces DA inference to a single forward rollout with on-the-fly corrections, avoiding expensive adjoint computations and/or optimizations during inference. We demonstrate that our method consistently outperforms four state-of-the-art baselines in stability, accuracy, and physical fidelity across two canonical PDEs and six observation regimes. We will release code and checkpoints publicly.
>
---
#### [new 113] Sharpness-Aware Data Generation for Zero-shot Quantization
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于零样本量化任务，旨在无真实训练数据的情况下，生成优化量化模型锐度的合成数据。通过最大化梯度匹配来减少模型锐度，提升泛化能力。实验表明其方法在CIFAR-100和ImageNet上优于现有技术。**

- **链接: [http://arxiv.org/pdf/2510.07018v1](http://arxiv.org/pdf/2510.07018v1)**

> **作者:** Dung Hoang-Anh; Cuong Pham Trung Le; Jianfei Cai; Thanh-Toan Do
>
> **摘要:** Zero-shot quantization aims to learn a quantized model from a pre-trained full-precision model with no access to original real training data. The common idea in zero-shot quantization approaches is to generate synthetic data for quantizing the full-precision model. While it is well-known that deep neural networks with low sharpness have better generalization ability, none of the previous zero-shot quantization works considers the sharpness of the quantized model as a criterion for generating training data. This paper introduces a novel methodology that takes into account quantized model sharpness in synthetic data generation to enhance generalization. Specifically, we first demonstrate that sharpness minimization can be attained by maximizing gradient matching between the reconstruction loss gradients computed on synthetic and real validation data, under certain assumptions. We then circumvent the problem of the gradient matching without real validation set by approximating it with the gradient matching between each generated sample and its neighbors. Experimental evaluations on CIFAR-100 and ImageNet datasets demonstrate the superiority of the proposed method over the state-of-the-art techniques in low-bit quantization settings.
>
---
#### [new 114] Surgeons Are Indian Males and Speech Therapists Are White Females: Auditing Biases in Vision-Language Models for Healthcare Professionals
- **分类: cs.CY; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言模型（VLM）中的偏见审计任务，旨在解决医疗领域中AI模型对医疗职业与人口属性间刻板印象的反映问题。作者构建了职业分类体系和职业感知提示集，评估模型在多个医疗角色上的性别、种族等人口统计偏见，揭示了关键领域AI应用的潜在公平性和合规风险。**

- **链接: [http://arxiv.org/pdf/2510.06280v1](http://arxiv.org/pdf/2510.06280v1)**

> **作者:** Zohaib Hasan Siddiqui; Dayam Nadeem; Mohammad Masudur Rahman; Mohammad Nadeem; Shahab Saquib Sohail; Beenish Moalla Chaudhry
>
> **摘要:** Vision language models (VLMs), such as CLIP and OpenCLIP, can encode and reflect stereotypical associations between medical professions and demographic attributes learned from web-scale data. We present an evaluation protocol for healthcare settings that quantifies associated biases and assesses their operational risk. Our methodology (i) defines a taxonomy spanning clinicians and allied healthcare roles (e.g., surgeon, cardiologist, dentist, nurse, pharmacist, technician), (ii) curates a profession-aware prompt suite to probe model behavior, and (iii) benchmarks demographic skew against a balanced face corpus. Empirically, we observe consistent demographic biases across multiple roles and vision models. Our work highlights the importance of bias identification in critical domains such as healthcare as AI-enabled hiring and workforce analytics can have downstream implications for equity, compliance, and patient trust.
>
---
## 更新

#### [replaced 001] Multi-modal Segment Assemblage Network for Ad Video Editing with Importance-Coherence Reward
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2209.12164v2](http://arxiv.org/pdf/2209.12164v2)**

> **作者:** Yolo Yunlong Tang; Siting Xu; Teng Wang; Qin Lin; Qinglin Lu; Feng Zheng
>
> **备注:** Accepted by ACCV 2022
>
> **摘要:** Advertisement video editing aims to automatically edit advertising videos into shorter videos while retaining coherent content and crucial information conveyed by advertisers. It mainly contains two stages: video segmentation and segment assemblage. The existing method performs well at video segmentation stages but suffers from the problems of dependencies on extra cumbersome models and poor performance at the segment assemblage stage. To address these problems, we propose M-SAN (Multi-modal Segment Assemblage Network) which can perform efficient and coherent segment assemblage task end-to-end. It utilizes multi-modal representation extracted from the segments and follows the Encoder-Decoder Ptr-Net framework with the Attention mechanism. Importance-coherence reward is designed for training M-SAN. We experiment on the Ads-1k dataset with 1000+ videos under rich ad scenarios collected from advertisers. To evaluate the methods, we propose a unified metric, Imp-Coh@Time, which comprehensively assesses the importance, coherence, and duration of the outputs at the same time. Experimental results show that our method achieves better performance than random selection and the previous method on the metric. Ablation experiments further verify that multi-modal representation and importance-coherence reward significantly improve the performance. Ads-1k dataset is available at: https://github.com/yunlong10/Ads-1k
>
---
#### [replaced 002] Platonic Transformers: A Solid Choice For Equivariance
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2510.03511v2](http://arxiv.org/pdf/2510.03511v2)**

> **作者:** Mohammad Mohaiminul Islam; Rishabh Anand; David R. Wessels; Friso de Kruiff; Thijs P. Kuipers; Rex Ying; Clara I. Sánchez; Sharvaree Vadgama; Georg Bökman; Erik J. Bekkers
>
> **摘要:** While widespread, Transformers lack inductive biases for geometric symmetries common in science and computer vision. Existing equivariant methods often sacrifice the efficiency and flexibility that make Transformers so effective through complex, computationally intensive designs. We introduce the Platonic Transformer to resolve this trade-off. By defining attention relative to reference frames from the Platonic solid symmetry groups, our method induces a principled weight-sharing scheme. This enables combined equivariance to continuous translations and Platonic symmetries, while preserving the exact architecture and computational cost of a standard Transformer. Furthermore, we show that this attention is formally equivalent to a dynamic group convolution, which reveals that the model learns adaptive geometric filters and enables a highly scalable, linear-time convolutional variant. Across diverse benchmarks in computer vision (CIFAR-10), 3D point clouds (ScanObjectNN), and molecular property prediction (QM9, OMol25), the Platonic Transformer achieves competitive performance by leveraging these geometric constraints at no additional cost.
>
---
#### [replaced 003] Towards a Multimodal Large Language Model with Pixel-Level Insight for Biomedicine
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.09278v3](http://arxiv.org/pdf/2412.09278v3)**

> **作者:** Xiaoshuang Huang; Lingdong Shen; Jia Liu; Fangxin Shang; Hongxiang Li; Haifeng Huang; Yehui Yang
>
> **备注:** Accepted by AAAI2025
>
> **摘要:** In recent years, Multimodal Large Language Models (MLLM) have achieved notable advancements, demonstrating the feasibility of developing an intelligent biomedical assistant. However, current biomedical MLLMs predominantly focus on image-level understanding and restrict interactions to textual commands, thus limiting their capability boundaries and the flexibility of usage. In this paper, we introduce a novel end-to-end multimodal large language model for the biomedical domain, named MedPLIB, which possesses pixel-level understanding. Excitingly, it supports visual question answering (VQA), arbitrary pixel-level prompts (points, bounding boxes, and free-form shapes), and pixel-level grounding. We propose a novel Mixture-of-Experts (MoE) multi-stage training strategy, which divides MoE into separate training phases for a visual-language expert model and a pixel-grounding expert model, followed by fine-tuning using MoE. This strategy effectively coordinates multitask learning while maintaining the computational cost at inference equivalent to that of a single expert model. To advance the research of biomedical MLLMs, we introduce the Medical Complex Vision Question Answering Dataset (MeCoVQA), which comprises an array of 8 modalities for complex medical imaging question answering and image region understanding. Experimental results indicate that MedPLIB has achieved state-of-the-art outcomes across multiple medical visual language tasks. More importantly, in zero-shot evaluations for the pixel grounding task, MedPLIB leads the best small and large models by margins of 19.7 and 15.6 respectively on the mDice metric. The codes, data, and model checkpoints will be made publicly available at https://github.com/ShawnHuang497/MedPLIB.
>
---
#### [replaced 004] Erasing More Than Intended? How Concept Erasure Degrades the Generation of Non-Target Concepts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.09833v2](http://arxiv.org/pdf/2501.09833v2)**

> **作者:** Ibtihel Amara; Ahmed Imtiaz Humayun; Ivana Kajic; Zarana Parekh; Natalie Harris; Sarah Young; Chirag Nagpal; Najoung Kim; Junfeng He; Cristina Nader Vasconcelos; Deepak Ramachandran; Golnoosh Farnadi; Katherine Heller; Mohammad Havaei; Negar Rostamzadeh
>
> **备注:** Accepted for publication at ICCV 2025
>
> **摘要:** Concept erasure techniques have recently gained significant attention for their potential to remove unwanted concepts from text-to-image models. While these methods often demonstrate promising results in controlled settings, their robustness in real-world applications and suitability for deployment remain uncertain. In this work, we (1) identify a critical gap in evaluating sanitized models, particularly in assessing their performance across diverse concept dimensions, and (2) systematically analyze the failure modes of text-to-image models post-erasure. We focus on the unintended consequences of concept removal on non-target concepts across different levels of interconnected relationships including visually similar, binomial, and semantically related concepts. To address this, we introduce EraseBench, a comprehensive benchmark for evaluating post-erasure performance. EraseBench includes over 100 curated concepts, targeted evaluation prompts, and a robust set of metrics to assess both effectiveness and side effects of erasure. Our findings reveal a phenomenon of concept entanglement, where erasure leads to unintended suppression of non-target concepts, causing spillover degradation that manifests as distortions and a decline in generation quality.
>
---
#### [replaced 005] TFM Dataset: A Novel Multi-task Dataset and Integrated Pipeline for Automated Tear Film Break-Up Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.05615v2](http://arxiv.org/pdf/2510.05615v2)**

> **作者:** Guangrong Wan; Jun liu; Qiyang Zhou; Tang tang; Lianghao Shi; Wenjun Luo; TingTing Xu
>
> **摘要:** Tear film break-up (TFBU) analysis is critical for diagnosing dry eye syndrome, but automated TFBU segmentation remains challenging due to the lack of annotated datasets and integrated solutions. This paper introduces the Tear Film Multi-task (TFM) Dataset, the first comprehensive dataset for multi-task tear film analysis, comprising 15 high-resolution videos (totaling 6,247 frames) annotated with three vision tasks: frame-level classification ('clear', 'closed', 'broken', 'blur'), Placido Ring detection, and pixel-wise TFBU area segmentation. Leveraging this dataset, we first propose TF-Net, a novel and efficient baseline segmentation model. TF-Net incorporates a MobileOne-mini backbone with re-parameterization techniques and an enhanced feature pyramid network to achieve a favorable balance between accuracy and computational efficiency for real-time clinical applications. We further establish benchmark performance on the TFM segmentation subset by comparing TF-Net against several state-of-the-art medical image segmentation models. Furthermore, we design TF-Collab, a novel integrated real-time pipeline that synergistically leverages models trained on all three tasks of the TFM dataset. By sequentially orchestrating frame classification for BUT determination, pupil region localization for input standardization, and TFBU segmentation, TF-Collab fully automates the analysis. Experimental results demonstrate the effectiveness of the proposed TF-Net and TF-Collab, providing a foundation for future research in ocular surface diagnostics. Our code and the TFM datasets are available at https://github.com/glory-wan/TF-Net
>
---
#### [replaced 006] Domain Generalization by Rejecting Extreme Augmentations
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2310.06670v2](http://arxiv.org/pdf/2310.06670v2)**

> **作者:** Masih Aminbeidokhti; Fidel A. Guerrero Peña; Heitor Rapela Medeiros; Thomas Dubail; Eric Granger; Marco Pedersoli
>
> **备注:** WACV 2024: Winter Conference on Applications of Computer Vision 2024
>
> **摘要:** Data augmentation is one of the most effective techniques for regularizing deep learning models and improving their recognition performance in a variety of tasks and domains. However, this holds for standard in-domain settings, in which the training and test data follow the same distribution. For the out-of-domain case, where the test data follow a different and unknown distribution, the best recipe for data augmentation is unclear. In this paper, we show that for out-of-domain and domain generalization settings, data augmentation can provide a conspicuous and robust improvement in performance. To do that, we propose a simple training procedure: (i) use uniform sampling on standard data augmentation transformations; (ii) increase the strength transformations to account for the higher data variance expected when working out-of-domain, and (iii) devise a new reward function to reject extreme transformations that can harm the training. With this procedure, our data augmentation scheme achieves a level of accuracy that is comparable to or better than state-of-the-art methods on benchmark domain generalization datasets. Code: https://github.com/Masseeh/DCAug
>
---
#### [replaced 007] DiffMI: Breaking Face Recognition Privacy via Diffusion-Driven Training-Free Model Inversion
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.18015v3](http://arxiv.org/pdf/2504.18015v3)**

> **作者:** Hanrui Wang; Shuo Wang; Chun-Shien Lu; Isao Echizen
>
> **摘要:** Face recognition poses serious privacy risks due to its reliance on sensitive and immutable biometric data. While modern systems mitigate privacy risks by mapping facial images to embeddings (commonly regarded as privacy-preserving), model inversion attacks reveal that identity information can still be recovered, exposing critical vulnerabilities. However, existing attacks are often computationally expensive and lack generalization, especially those requiring target-specific training. Even training-free approaches suffer from limited identity controllability, hindering faithful reconstruction of nuanced or unseen identities. In this work, we propose DiffMI, the first diffusion-driven, training-free model inversion attack. DiffMI introduces a novel pipeline combining robust latent code initialization, a ranked adversarial refinement strategy, and a statistically grounded, confidence-aware optimization objective. DiffMI applies directly to unseen target identities and face recognition models, offering greater adaptability than training-dependent approaches while significantly reducing computational overhead. Our method achieves 84.42%--92.87% attack success rates against inversion-resilient systems and outperforms the best prior training-free GAN-based approach by 4.01%--9.82%. The implementation is available at https://github.com/azrealwang/DiffMI.
>
---
#### [replaced 008] RespoDiff: Dual-Module Bottleneck Transformation for Responsible & Faithful T2I Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.15257v2](http://arxiv.org/pdf/2509.15257v2)**

> **作者:** Silpa Vadakkeeveetil Sreelatha; Sauradip Nag; Muhammad Awais; Serge Belongie; Anjan Dutta
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** The rapid advancement of diffusion models has enabled high-fidelity and semantically rich text-to-image generation; however, ensuring fairness and safety remains an open challenge. Existing methods typically improve fairness and safety at the expense of semantic fidelity and image quality. In this work, we propose RespoDiff, a novel framework for responsible text-to-image generation that incorporates a dual-module transformation on the intermediate bottleneck representations of diffusion models. Our approach introduces two distinct learnable modules: one focused on capturing and enforcing responsible concepts, such as fairness and safety, and the other dedicated to maintaining semantic alignment with neutral prompts. To facilitate the dual learning process, we introduce a novel score-matching objective that enables effective coordination between the modules. Our method outperforms state-of-the-art methods in responsible generation by ensuring semantic alignment while optimizing both objectives without compromising image fidelity. Our approach improves responsible and semantically coherent generation by 20% across diverse, unseen prompts. Moreover, it integrates seamlessly into large-scale models like SDXL, enhancing fairness and safety. Code will be released upon acceptance.
>
---
#### [replaced 009] Fully Spiking Neural Networks for Unified Frame-Event Object Tracking
- **分类: cs.CV; cs.NE**

- **链接: [http://arxiv.org/pdf/2505.20834v2](http://arxiv.org/pdf/2505.20834v2)**

> **作者:** Jingjun Yang; Liangwei Fan; Jinpu Zhang; Xiangkai Lian; Hui Shen; Dewen Hu
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** The integration of image and event streams offers a promising approach for achieving robust visual object tracking in complex environments. However, current fusion methods achieve high performance at the cost of significant computational overhead and struggle to efficiently extract the sparse, asynchronous information from event streams, failing to leverage the energy-efficient advantages of event-driven spiking paradigms. To address this challenge, we propose the first fully Spiking Frame-Event Tracking framework called SpikeFET. This network achieves synergistic integration of convolutional local feature extraction and Transformer-based global modeling within the spiking paradigm, effectively fusing frame and event data. To overcome the degradation of translation invariance caused by convolutional padding, we introduce a Random Patchwork Module (RPM) that eliminates positional bias through randomized spatial reorganization and learnable type encoding while preserving residual structures. Furthermore, we propose a Spatial-Temporal Regularization (STR) strategy that overcomes similarity metric degradation from asymmetric features by enforcing spatio-temporal consistency among temporal template features in latent space. Extensive experiments across multiple benchmarks demonstrate that the proposed framework achieves superior tracking accuracy over existing methods while significantly reducing power consumption, attaining an optimal balance between performance and efficiency.
>
---
#### [replaced 010] LLMVA-GEBC: Large Language Model with Video Adapter for Generic Event Boundary Captioning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2306.10354v2](http://arxiv.org/pdf/2306.10354v2)**

> **作者:** Yolo Yunlong Tang; Jinrui Zhang; Xiangchen Wang; Teng Wang; Feng Zheng
>
> **备注:** Winner solution to Generic Event Boundary Captioning task in LOVEU Challenge (CVPR 2023 workshop)
>
> **摘要:** Our winning entry for the CVPR 2023 Generic Event Boundary Captioning (GEBC) competition is detailed in this paper. Unlike conventional video captioning tasks, GEBC demands that the captioning model possess an understanding of immediate changes in status around the designated video boundary, making it a difficult task. This paper proposes an effective model LLMVA-GEBC (Large Language Model with Video Adapter for Generic Event Boundary Captioning): (1) We utilize a pretrained LLM for generating human-like captions with high quality. (2) To adapt the model to the GEBC task, we take the video Q-former as an adapter and train it with the frozen visual feature extractors and LLM. Our proposed method achieved a 76.14 score on the test set and won the first place in the challenge. Our code is available at https://github.com/zjr2000/LLMVA-GEBC .
>
---
#### [replaced 011] RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18468v5](http://arxiv.org/pdf/2504.18468v5)**

> **作者:** Georgios Kouros; Minye Wu; Tinne Tuytelaars
>
> **摘要:** In this work, we address specular appearance in inverse rendering using 2D Gaussian splatting with deferred shading and argue for a refinement stage to improve specular detail, thereby bridging the gap with reconstruction-only methods. Our pipeline estimates editable material properties and environment illumination while employing a directional residual pass that captures leftover view-dependent effects for further refining novel view synthesis. In contrast to per-Gaussian shading with shortest-axis normals and normal residuals, which tends to result in more noisy geometry and specular appearance, a pixel-deferred surfel formulation with specular residuals yields sharper highlights, cleaner materials, and improved editability. We evaluate our approach on rendering and reconstruction quality on three popular datasets featuring glossy objects, and also demonstrate high-quality relighting and material editing.
>
---
#### [replaced 012] Video-in-the-Loop: Span-Grounded Long Video QA with Interleaved Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.04022v3](http://arxiv.org/pdf/2510.04022v3)**

> **作者:** Chendong Wang; Donglin Bai; Yifan Yang; Xiao Jin; Anlan Zhang; Rui Wang; Shiqi Jiang; Yuqing Yang; Hao Wu; Qi Dai; Chong Luo; Ting Cao; Lili Qiu; Suman Banerjee
>
> **摘要:** We present \emph{Video-in-the-Loop} (ViTL), a two-stage long-video QA framework that preserves a fixed token budget by first \emph{localizing} question-relevant interval(s) with a low-fps skim and then \emph{answering} via span-aware reallocation of visual tokens at higher effective frame rate, emitting an interleaved output with both spans and the final option for direct attribution. We also introduce \dataname{}, which converts description based event graphs into \emph{span-grounded} multiple-choice QA by pairing each question with \emph{ground-truth} time span(s) and related reasoning. ViTL is trained end-to-end with an interleaved group-relative objective that couples temporal IoU for localization with answer correctness, allowing credit to flow from answers back to spans without increasing compute. Under fixed token budgets, ViTL attains up to 8.6% with 50% less frame input on long-video QA and temporal grounding (e.g., Charades-STA, ActivityNet-Captions) and ablations show that span-aware token reallocation consistently surpasses uniform sampling. Together, \dataname{} and ViTL provide an interpretable, compute-efficient recipe for scalable long-video QA.
>
---
#### [replaced 013] MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17832v3](http://arxiv.org/pdf/2502.17832v3)**

> **作者:** Hyeonjeong Ha; Qiusi Zhan; Jeonghwan Kim; Dimitrios Bralios; Saikrishna Sanniboina; Nanyun Peng; Kai-Wei Chang; Daniel Kang; Heng Ji
>
> **备注:** Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG
>
> **摘要:** Multimodal large language models with Retrieval Augmented Generation (RAG) have significantly advanced tasks such as multimodal question answering by grounding responses in external text and images. This grounding improves factuality, reduces hallucination, and extends reasoning beyond parametric knowledge. However, this reliance on external knowledge poses a critical yet underexplored safety risk: knowledge poisoning attacks, where adversaries deliberately inject adversarial multimodal content into external knowledge bases to steer model toward generating incorrect or even harmful responses. To expose such vulnerabilities, we propose MM-PoisonRAG, the first framework to systematically design knowledge poisoning in multimodal RAG. We introduce two complementary attack strategies: Localized Poisoning Attack (LPA), which implants targeted multimodal misinformation to manipulate specific queries, and Globalized Poisoning Attack (GPA), which inserts a single adversarial knowledge to broadly disrupt reasoning and induce nonsensical responses across all queries. Comprehensive experiments across tasks, models, and access settings show that LPA achieves targeted manipulation with attack success rates of up to 56%, while GPA completely disrupts model generation to 0% accuracy with just a single adversarial knowledge injection. Our results reveal the fragility of multimodal RAG and highlight the urgent need for defenses against knowledge poisoning.
>
---
#### [replaced 014] A Deep Learning System for Rapid and Accurate Warning of Acute Aortic Syndrome on Non-contrast CT in China
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.15222v5](http://arxiv.org/pdf/2406.15222v5)**

> **作者:** Yujian Hu; Yilang Xiang; Yan-Jie Zhou; Yangyan He; Dehai Lang; Shifeng Yang; Xiaolong Du; Chunlan Den; Youyao Xu; Gaofeng Wang; Zhengyao Ding; Jingyong Huang; Wenjun Zhao; Xuejun Wu; Donglin Li; Qianqian Zhu; Zhenjiang Li; Chenyang Qiu; Ziheng Wu; Yunjun He; Chen Tian; Yihui Qiu; Zuodong Lin; Xiaolong Zhang; Yuan He; Zhenpeng Yuan; Xiaoxiang Zhou; Rong Fan; Ruihan Chen; Wenchao Guo; Jianpeng Zhang; Tony C. W. Mok; Zi Li; Mannudeep K. Kalra; Le Lu; Wenbo Xiao; Xiaoqiang Li; Yun Bian; Chengwei Shao; Guofu Wang; Wei Lu; Zhengxing Huang; Minfeng Xu; Hongkun Zhang
>
> **摘要:** The accurate and timely diagnosis of acute aortic syndromes (AAS) in patients presenting with acute chest pain remains a clinical challenge. Aortic CT angiography (CTA) is the imaging protocol of choice in patients with suspected AAS. However, due to economic and workflow constraints in China, the majority of suspected patients initially undergo non-contrast CT as the initial imaging testing, and CTA is reserved for those at higher risk. In this work, we present an artificial intelligence-based warning system, iAorta, using non-contrast CT for AAS identification in China, which demonstrates remarkably high accuracy and provides clinicians with interpretable warnings. iAorta was evaluated through a comprehensive step-wise study. In the multi-center retrospective study (n = 20,750), iAorta achieved a mean area under the receiver operating curve (AUC) of 0.958 (95% CI 0.950-0.967). In the large-scale real-world study (n = 137,525), iAorta demonstrated consistently high performance across various non-contrast CT protocols, achieving a sensitivity of 0.913-0.942 and a specificity of 0.991-0.993. In the prospective comparative study (n = 13,846), iAorta demonstrated the capability to significantly shorten the time to correct diagnostic pathway. For the prospective pilot deployment that we conducted, iAorta correctly identified 21 out of 22 patients with AAS among 15,584 consecutive patients presenting with acute chest pain and under non-contrast CT protocol in the emergency department (ED) and enabled the average diagnostic time of these 21 AAS positive patients to be 102.1 (75-133) mins. Last, the iAorta can help avoid delayed or missed diagnosis of AAS in settings where non-contrast CT remains the unavoidable the initial or only imaging test in resource-constrained regions and in patients who cannot or did not receive intravenous contrast.
>
---
#### [replaced 015] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding
- **分类: cs.LG; cs.AI; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.15946v3](http://arxiv.org/pdf/2505.15946v3)**

> **作者:** Yuxiang Wei; Yanteng Zhang; Xi Xiao; Tianyang Wang; Xiao Wang; Vince D. Calhoun
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Decoding visual experiences from fMRI offers a powerful avenue to understand human perception and develop advanced brain-computer interfaces. However, current progress often prioritizes maximizing reconstruction fidelity while overlooking interpretability, an essential aspect for deriving neuroscientific insight. To address this gap, we propose MoRE-Brain, a neuro-inspired framework designed for high-fidelity, adaptable, and interpretable visual reconstruction. MoRE-Brain uniquely employs a hierarchical Mixture-of-Experts architecture where distinct experts process fMRI signals from functionally related voxel groups, mimicking specialized brain networks. The experts are first trained to encode fMRI into the frozen CLIP space. A finetuned diffusion model then synthesizes images, guided by expert outputs through a novel dual-stage routing mechanism that dynamically weighs expert contributions across the diffusion process. MoRE-Brain offers three main advancements: First, it introduces a novel Mixture-of-Experts architecture grounded in brain network principles for neuro-decoding. Second, it achieves efficient cross-subject generalization by sharing core expert networks while adapting only subject-specific routers. Third, it provides enhanced mechanistic insight, as the explicit routing reveals precisely how different modeled brain regions shape the semantic and spatial attributes of the reconstructed image. Extensive experiments validate MoRE-Brain's high reconstruction fidelity, with bottleneck analyses further demonstrating its effective utilization of fMRI signals, distinguishing genuine neural decoding from over-reliance on generative priors. Consequently, MoRE-Brain marks a substantial advance towards more generalizable and interpretable fMRI-based visual decoding. Code will be publicly available soon: https://github.com/yuxiangwei0808/MoRE-Brain.
>
---
#### [replaced 016] The Percept-V Challenge: Can Multimodal LLMs Crack Simple Perception Problems?
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.21143v2](http://arxiv.org/pdf/2508.21143v2)**

> **作者:** Samrajnee Ghosh; Naman Agarwal; Hemanshu Garg; Chinmay Mittal; Mausam; Parag Singla
>
> **摘要:** Cognitive science research treats visual perception, the ability to understand and make sense of a visual input, as one of the early developmental signs of intelligence. Its TVPS-4 framework categorizes and tests human perception into seven skills such as visual discrimination, and form constancy. Do Multimodal Large Language Models (MLLMs) match up to humans in basic perception? Even though there are many benchmarks that evaluate MLLMs on advanced reasoning and knowledge skills, there is limited research that focuses evaluation on simple perception. In response, we introduce Percept-V, a dataset containing 6000 program-generated uncontaminated images divided into 30 domains, where each domain tests one or more TVPS-4 skills. Our focus is on perception, so we make our domains quite simple and the reasoning and knowledge required for solving them are minimal. Since modern-day MLLMs can solve much more complex tasks, our a-priori expectation is that they will solve these domains very easily. Contrary to our belief, our experiments show a weak performance of SoTA proprietary and open-source MLLMs compared to very high human performance on Percept-V. We find that as number of objects in the image increases, performance goes down rather fast. Our experiments also identify the perception skills that are considerably harder for all models.
>
---
#### [replaced 017] SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models
- **分类: cs.CR; cs.AI; cs.CV; I.2**

- **链接: [http://arxiv.org/pdf/2510.05173v2](http://arxiv.org/pdf/2510.05173v2)**

> **作者:** Peigui Qi; Kunsheng Tang; Wenbo Zhou; Weiming Zhang; Nenghai Yu; Tianwei Zhang; Qing Guo; Jie Zhang
>
> **备注:** Accepted by ACM CCS 2025
>
> **摘要:** Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce \textbf{SafeGuider}, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, \textbf{SafeGuider} generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.
>
---
#### [replaced 018] PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations
- **分类: cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2505.19256v4](http://arxiv.org/pdf/2505.19256v4)**

> **作者:** Vivek Gopalakrishnan; Neel Dey; Polina Golland
>
> **备注:** NeurIPS 2025. Code available at https://github.com/eigenvivek/polypose
>
> **摘要:** Determining the 3D pose of a patient from a limited set of 2D X-ray images is a critical task in interventional settings. While preoperative volumetric imaging (e.g., CT and MRI) provides precise 3D localization and visualization of anatomical targets, these modalities cannot be acquired during procedures, where fast 2D imaging (X-ray) is used instead. To integrate volumetric guidance into intraoperative procedures, we present PolyPose, a simple and robust method for deformable 2D/3D registration. PolyPose parameterizes complex 3D deformation fields as a composition of rigid transforms, leveraging the biological constraint that individual bones do not bend in typical motion. Unlike existing methods that either assume no inter-joint movement or fail outright in this under-determined setting, our polyrigid formulation enforces anatomically plausible priors that respect the piecewise-rigid nature of human movement. This approach eliminates the need for expensive deformation regularizers that require patient- and procedure-specific hyperparameter optimization. Across extensive experiments on diverse datasets from orthopedic surgery and radiotherapy, we show that this strong inductive bias enables PolyPose to successfully align the patient's preoperative volume to as few as two X-rays, thereby providing crucial 3D guidance in challenging sparse-view and limited-angle settings where current registration methods fail. Additional visualizations, tutorials, and code are available at https://polypose.csail.mit.edu.
>
---
#### [replaced 019] Decomposed Global Optimization for Robust Point Matching with Low-Dimensional Branching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.08589v2](http://arxiv.org/pdf/2405.08589v2)**

> **作者:** Wei Lian; Zhesen Cui; Fei Ma; Hang Pan; Wangmeng Zuo; Jianmei Zhang
>
> **摘要:** Numerous applications require algorithms that can align partially overlapping point sets while maintaining invariance to geometric transformations (e.g., similarity, affine, rigid). This paper introduces a novel global optimization method for this task by minimizing the objective function of the Robust Point Matching (RPM) algorithm. We first reveal that the original RPM objective is a cubic polynomial. Through a concise variable substitution, we transform this objective into a quadratic function. By leveraging the convex envelope of bilinear monomials, we derive a tight lower bound for this quadratic function. This lower bound problem conveniently and efficiently decomposes into two parts: a standard linear assignment problem (solvable in polynomial time) and a low-dimensional convex quadratic program. Furthermore, we devise a specialized Branch-and-Bound (BnB) algorithm that branches exclusively on the transformation parameters, which significantly accelerates convergence by confining the search space. Experiments on 2D and 3D synthetic and real-world data demonstrate that our method, compared to state-of-the-art approaches, exhibits superior robustness to non-rigid deformations, positional noise, and outliers, particularly in scenarios where outliers are distinct from inliers.
>
---
#### [replaced 020] Unlocking Dataset Distillation with Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.03881v4](http://arxiv.org/pdf/2403.03881v4)**

> **作者:** Brian B. Moser; Federico Raue; Sebastian Palacio; Stanislav Frolov; Andreas Dengel
>
> **摘要:** Dataset distillation seeks to condense datasets into smaller but highly representative synthetic samples. While diffusion models now lead all generative benchmarks, current distillation methods avoid them and rely instead on GANs or autoencoders, or, at best, sampling from a fixed diffusion prior. This trend arises because naive backpropagation through the long denoising chain leads to vanishing gradients, which prevents effective synthetic sample optimization. To address this limitation, we introduce Latent Dataset Distillation with Diffusion Models (LD3M), the first method to learn gradient-based distilled latents and class embeddings end-to-end through a pre-trained latent diffusion model. A linearly decaying skip connection, injected from the initial noisy state into every reverse step, preserves the gradient signal across dozens of timesteps without requiring diffusion weight fine-tuning. Across multiple ImageNet subsets at 128x128 and 256x256, LD3M improves downstream accuracy by up to 4.8 percentage points (1 IPC) and 4.2 points (10 IPC) over the prior state-of-the-art. The code for LD3M is provided at https://github.com/Brian-Moser/prune_and_distill.
>
---
#### [replaced 021] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.03993v3](http://arxiv.org/pdf/2510.03993v3)**

> **作者:** Yaxin Hou; Bo Han; Yuheng Jia; Hui Liu; Junhui Hou
>
> **备注:** The paper is accepted by NeurIPS 2025
>
> **摘要:** Current long-tailed semi-supervised learning methods assume that labeled data exhibit a long-tailed distribution, and unlabeled data adhere to a typical predefined distribution (i.e., long-tailed, uniform, or inverse long-tailed). However, the distribution of the unlabeled data is generally unknown and may follow an arbitrary distribution. To tackle this challenge, we propose a Controllable Pseudo-label Generation (CPG) framework, expanding the labeled dataset with the progressively identified reliable pseudo-labels from the unlabeled dataset and training the model on the updated labeled dataset with a known distribution, making it unaffected by the unlabeled data distribution. Specifically, CPG operates through a controllable self-reinforcing optimization cycle: (i) at each training step, our dynamic controllable filtering mechanism selectively incorporates reliable pseudo-labels from the unlabeled dataset into the labeled dataset, ensuring that the updated labeled dataset follows a known distribution; (ii) we then construct a Bayes-optimal classifier using logit adjustment based on the updated labeled data distribution; (iii) this improved classifier subsequently helps identify more reliable pseudo-labels in the next training step. We further theoretically prove that this optimization cycle can significantly reduce the generalization error under some conditions. Additionally, we propose a class-aware adaptive augmentation module to further improve the representation of minority classes, and an auxiliary branch to maximize data utilization by leveraging all labeled and unlabeled samples. Comprehensive evaluations on various commonly used benchmark datasets show that CPG achieves consistent improvements, surpassing state-of-the-art methods by up to $\textbf{15.97%}$ in accuracy. The code is available at https://github.com/yaxinhou/CPG.
>
---
#### [replaced 022] CaRDiff: Video Salient Object Ranking Chain of Thought Reasoning for Saliency Prediction with Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.12009v2](http://arxiv.org/pdf/2408.12009v2)**

> **作者:** Yolo Yunlong Tang; Gen Zhan; Li Yang; Yiting Liao; Chenliang Xu
>
> **备注:** Accepted to AAAI 2025
>
> **摘要:** Video saliency prediction aims to identify the regions in a video that attract human attention and gaze, driven by bottom-up features from the video and top-down processes like memory and cognition. Among these top-down influences, language plays a crucial role in guiding attention by shaping how visual information is interpreted. Existing methods primarily focus on modeling perceptual information while neglecting the reasoning process facilitated by language, where ranking cues are crucial outcomes of this process and practical guidance for saliency prediction. In this paper, we propose CaRDiff (Caption, Rank, and generate with Diffusion), a framework that imitates the process by integrating a multimodal large language model (MLLM), a grounding module, and a diffusion model, to enhance video saliency prediction. Specifically, we introduce a novel prompting method VSOR-CoT (Video Salient Object Ranking Chain of Thought), which utilizes an MLLM with a grounding module to caption video content and infer salient objects along with their rankings and positions. This process derives ranking maps that can be sufficiently leveraged by the diffusion model to decode the saliency maps for the given video accurately. Extensive experiments show the effectiveness of VSOR-CoT in improving the performance of video saliency prediction. The proposed CaRDiff performs better than state-of-the-art models on the MVS dataset and demonstrates cross-dataset capabilities on the DHF1k dataset through zero-shot evaluation.
>
---
#### [replaced 023] Generative AI for Cel-Animation: A Survey
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.06250v4](http://arxiv.org/pdf/2501.06250v4)**

> **作者:** Yolo Yunlong Tang; Junjia Guo; Pinxin Liu; Zhiyuan Wang; Hang Hua; Jia-Xing Zhong; Yunzhong Xiao; Chao Huang; Luchuan Song; Susan Liang; Yizhi Song; Liu He; Jing Bi; Mingqian Feng; Xinyang Li; Zeliang Zhang; Chenliang Xu
>
> **备注:** Accepted by ICCV 2025 AISTORY Workshop
>
> **摘要:** Traditional Celluloid (Cel) Animation production pipeline encompasses multiple essential steps, including storyboarding, layout design, keyframe animation, inbetweening, and colorization, which demand substantial manual effort, technical expertise, and significant time investment. These challenges have historically impeded the efficiency and scalability of Cel-Animation production. The rise of generative artificial intelligence (GenAI), encompassing large language models, multimodal models, and diffusion models, offers innovative solutions by automating tasks such as inbetween frame generation, colorization, and storyboard creation. This survey explores how GenAI integration is revolutionizing traditional animation workflows by lowering technical barriers, broadening accessibility for a wider range of creators through tools like AniDoc, ToonCrafter, and AniSora, and enabling artists to focus more on creative expression and artistic innovation. Despite its potential, challenges like visual consistency, stylistic coherence, and ethical considerations persist. Additionally, this paper explores future directions and advancements in AI-assisted animation. For further exploration and resources, please visit our GitHub repository: https://github.com/yunlong10/Awesome-AI4Animation
>
---
#### [replaced 024] Color Bind: Exploring Color Perception in Text-to-Image Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19791v2](http://arxiv.org/pdf/2508.19791v2)**

> **作者:** Shay Shomer Chai; Wenxuan Peng; Bharath Hariharan; Hadar Averbuch-Elor
>
> **备注:** Project webpage: https://tau-vailab.github.io/color-edit/
>
> **摘要:** Text-to-image generation has recently seen remarkable success, granting users with the ability to create high-quality images through the use of text. However, contemporary methods face challenges in capturing the precise semantics conveyed by complex multi-object prompts. Consequently, many works have sought to mitigate such semantic misalignments, typically via inference-time schemes that modify the attention layers of the denoising networks. However, prior work has mostly utilized coarse metrics, such as the cosine similarity between text and image CLIP embeddings, or human evaluations, which are challenging to conduct on a larger-scale. In this work, we perform a case study on colors -- a fundamental attribute commonly associated with objects in text prompts, which offer a rich test bed for rigorous evaluation. Our analysis reveals that pretrained models struggle to generate images that faithfully reflect multiple color attributes-far more so than with single-color prompts-and that neither inference-time techniques nor existing editing methods reliably resolve these semantic misalignments. Accordingly, we introduce a dedicated image editing technique, mitigating the issue of multi-object semantic alignment for prompts containing multiple colors. We demonstrate that our approach significantly boosts performance over a wide range of metrics, considering images generated by various text-to-image diffusion-based techniques.
>
---
#### [replaced 025] Efficient Universal Models for Medical Image Segmentation via Weakly Supervised In-Context Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.05899v2](http://arxiv.org/pdf/2510.05899v2)**

> **作者:** Jiesi Hu; Yanwu Yang; Zhiyu Ye; Jinyan Zhou; Jianfeng Cao; Hanyang Peng; Ting Ma
>
> **摘要:** Universal models for medical image segmentation, such as interactive and in-context learning (ICL) models, offer strong generalization but require extensive annotations. Interactive models need repeated user prompts for each image, while ICL relies on dense, pixel-level labels. To address this, we propose Weakly Supervised In-Context Learning (WS-ICL), a new ICL paradigm that leverages weak prompts (e.g., bounding boxes or points) instead of dense labels for context. This approach significantly reduces annotation effort by eliminating the need for fine-grained masks and repeated user prompting for all images. We evaluated the proposed WS-ICL model on three held-out benchmarks. Experimental results demonstrate that WS-ICL achieves performance comparable to regular ICL models at a significantly lower annotation cost. In addition, WS-ICL is highly competitive even under the interactive paradigm. These findings establish WS-ICL as a promising step toward more efficient and unified universal models for medical image segmentation. Our code and model are publicly available at https://github.com/jiesihu/Weak-ICL.
>
---
#### [replaced 026] acia-workflows: Automated Single-cell Imaging Analysis for Scalable and Deep Learning-based Live-cell Imaging Analysis Workflows
- **分类: cs.CV; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2510.05886v2](http://arxiv.org/pdf/2510.05886v2)**

> **作者:** Johannes Seiffarth; Keitaro Kasahara; Michelle Bund; Benita Lückel; Richard D. Paul; Matthias Pesch; Lennart Witting; Michael Bott; Dietrich Kohlheyer; Katharina Nöh
>
> **摘要:** Live-cell imaging (LCI) technology enables the detailed spatio-temporal characterization of living cells at the single-cell level, which is critical for advancing research in the life sciences, from biomedical applications to bioprocessing. High-throughput setups with tens to hundreds of parallel cell cultivations offer the potential for robust and reproducible insights. However, these insights are obscured by the large amount of LCI data recorded per experiment. Recent advances in state-of-the-art deep learning methods for cell segmentation and tracking now enable the automated analysis of such large data volumes, offering unprecedented opportunities to systematically study single-cell dynamics. The next key challenge lies in integrating these powerful tools into accessible, flexible, and user-friendly workflows that support routine application in biological research. In this work, we present acia-workflows, a platform that combines three key components: (1) the Automated live-Cell Imaging Analysis (acia) Python library, which supports the modular design of image analysis pipelines offering eight deep learning segmentation and tracking approaches; (2) workflows that assemble the image analysis pipeline, its software dependencies, documentation, and visualizations into a single Jupyter Notebook, leading to accessible, reproducible and scalable analysis workflows; and (3) a collection of application workflows showcasing the analysis and customization capabilities in real-world applications. Specifically, we present three workflows to investigate various types of microfluidic LCI experiments ranging from growth rate comparisons to precise, minute-resolution quantitative analyses of individual dynamic cells responses to changing oxygen conditions. Our collection of more than ten application workflows is open source and publicly available at https://github.com/JuBiotech/acia-workflows.
>
---
#### [replaced 027] VGGT-X: When VGGT Meets Dense Novel View Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.25191v2](http://arxiv.org/pdf/2509.25191v2)**

> **作者:** Yang Liu; Chuanchen Luo; Zimo Tang; Junran Peng; Zhaoxiang Zhang
>
> **备注:** Project Page: https://dekuliutesla.github.io/vggt-x.github.io/
>
> **摘要:** We study the problem of applying 3D Foundation Models (3DFMs) to dense Novel View Synthesis (NVS). Despite significant progress in Novel View Synthesis powered by NeRF and 3DGS, current approaches remain reliant on accurate 3D attributes (e.g., camera poses and point clouds) acquired from Structure-from-Motion (SfM), which is often slow and fragile in low-texture or low-overlap captures. Recent 3DFMs showcase orders of magnitude speedup over the traditional pipeline and great potential for online NVS. But most of the validation and conclusions are confined to sparse-view settings. Our study reveals that naively scaling 3DFMs to dense views encounters two fundamental barriers: dramatically increasing VRAM burden and imperfect outputs that degrade initialization-sensitive 3D training. To address these barriers, we introduce VGGT-X, incorporating a memory-efficient VGGT implementation that scales to 1,000+ images, an adaptive global alignment for VGGT output enhancement, and robust 3DGS training practices. Extensive experiments show that these measures substantially close the fidelity gap with COLMAP-initialized pipelines, achieving state-of-the-art results in dense COLMAP-free NVS and pose estimation. Additionally, we analyze the causes of remaining gaps with COLMAP-initialized rendering, providing insights for the future development of 3D foundation models and dense NVS. Our project page is available at https://dekuliutesla.github.io/vggt-x.github.io/
>
---
#### [replaced 028] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning
- **分类: cs.CV; cs.LG; I.4.9**

- **链接: [http://arxiv.org/pdf/2509.25033v2](http://arxiv.org/pdf/2509.25033v2)**

> **作者:** Wenhao Li; Qiangchang Wang; Xianjing Meng; Zhibin Wu; Yilong Yin
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Few-shot learning (FSL) aims to recognize novel concepts from only a few labeled support samples. Recent studies enhance support features by incorporating additional semantic information or designing complex semantic fusion modules. However, they still suffer from hallucinating semantics that contradict the visual evidence due to the lack of grounding in actual instances, resulting in noisy guidance and costly corrections. To address these issues, we propose a novel framework, bridging Vision and Text with LLMs for Few-Shot Learning (VT-FSL), which constructs precise cross-modal prompts conditioned on Large Language Models (LLMs) and support images, seamlessly integrating them through a geometry-aware alignment. It mainly consists of Cross-modal Iterative Prompting (CIP) and Cross-modal Geometric Alignment (CGA). Specifically, the CIP conditions an LLM on both class names and support images to generate precise class descriptions iteratively in a single structured reasoning pass. These descriptions not only enrich the semantic understanding of novel classes but also enable the zero-shot synthesis of semantically consistent images. The descriptions and synthetic images act respectively as complementary textual and visual prompts, providing high-level class semantics and low-level intra-class diversity to compensate for limited support data. Furthermore, the CGA jointly aligns the fused textual, support, and synthetic visual representations by minimizing the kernelized volume of the 3-dimensional parallelotope they span. It captures global and nonlinear relationships among all representations, enabling structured and consistent multimodal integration. The proposed VT-FSL method establishes new state-of-the-art performance across ten diverse benchmarks, including standard, cross-domain, and fine-grained few-shot learning scenarios. Code is available at https://github.com/peacelwh/VT-FSL.
>
---
#### [replaced 029] Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.05034v2](http://arxiv.org/pdf/2510.05034v2)**

> **作者:** Yolo Yunlong Tang; Jing Bi; Pinxin Liu; Zhenyu Pan; Zhangyun Tan; Qianxiang Shen; Jiani Liu; Hang Hua; Junjia Guo; Yunzhong Xiao; Chao Huang; Zhiyuan Wang; Susan Liang; Xinyi Liu; Yizhi Song; Yuhe Nie; Jia-Xing Zhong; Bozheng Li; Daiqing Qi; Ziyun Zeng; Ali Vosoughi; Luchuan Song; Zeliang Zhang; Daiki Shimada; Han Liu; Jiebo Luo; Chenliang Xu
>
> **备注:** The 1st version
>
> **摘要:** Video understanding represents the most challenging frontier in computer vision, requiring models to reason about complex spatiotemporal relationships, long-term dependencies, and multimodal evidence. The recent emergence of Video-Large Multimodal Models (Video-LMMs), which integrate visual encoders with powerful decoder-based language models, has demonstrated remarkable capabilities in video understanding tasks. However, the critical phase that transforms these models from basic perception systems into sophisticated reasoning engines, post-training, remains fragmented across the literature. This survey provides the first comprehensive examination of post-training methodologies for Video-LMMs, encompassing three fundamental pillars: supervised fine-tuning (SFT) with chain-of-thought, reinforcement learning (RL) from verifiable objectives, and test-time scaling (TTS) through enhanced inference computation. We present a structured taxonomy that clarifies the roles, interconnections, and video-specific adaptations of these techniques, addressing unique challenges such as temporal localization, spatiotemporal grounding, long video efficiency, and multimodal evidence integration. Through systematic analysis of representative methods, we synthesize key design principles, insights, and evaluation protocols while identifying critical open challenges in reward design, scalability, and cost-performance optimization. We further curate essential benchmarks, datasets, and metrics to facilitate rigorous assessment of post-training effectiveness. This survey aims to provide researchers and practitioners with a unified framework for advancing Video-LMM capabilities. Additional resources and updates are maintained at: https://github.com/yunlong10/Awesome-Video-LMM-Post-Training
>
---
#### [replaced 030] Guardians of Image Quality: Benchmarking Defenses Against Adversarial Attacks on Image Quality Metrics
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2408.01541v2](http://arxiv.org/pdf/2408.01541v2)**

> **作者:** Alexander Gushchin; Khaled Abud; Georgii Bychkov; Ekaterina Shumitskaya; Anna Chistyakova; Sergey Lavrushkin; Bader Rasheed; Kirill Malyshev; Dmitriy Vatolin; Anastasia Antsiferova
>
> **摘要:** In the field of Image Quality Assessment (IQA), the adversarial robustness of the metrics poses a critical concern. This paper presents a comprehensive benchmarking study of various defense mechanisms in response to the rise in adversarial attacks on IQA. We systematically evaluate 25 defense strategies, including adversarial purification, adversarial training, and certified robustness methods. We applied 14 adversarial attack algorithms of various types in both non-adaptive and adaptive settings and tested these defenses against them. We analyze the differences between defenses and their applicability to IQA tasks, considering that they should preserve IQA scores and image quality. The proposed benchmark aims to guide future developments and accepts submissions of new methods, with the latest results available online: https://videoprocessing.ai/benchmarks/iqa-defenses.html.
>
---
#### [replaced 031] Longitudinal Flow Matching for Trajectory Modeling
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2510.03569v2](http://arxiv.org/pdf/2510.03569v2)**

> **作者:** Mohammad Mohaiminul Islam; Thijs P. Kuipers; Sharvaree Vadgama; Coen de Vente; Afsana Khan; Clara I. Sánchez; Erik J. Bekkers
>
> **摘要:** Generative models for sequential data often struggle with sparsely sampled and high-dimensional trajectories, typically reducing the learning of dynamics to pairwise transitions. We propose Interpolative Multi-Marginal Flow Matching (IMMFM), a framework that learns continuous stochastic dynamics jointly consistent with multiple observed time points. IMMFM employs a piecewise-quadratic interpolation path as a smooth target for flow matching and jointly optimizes drift and a data-driven diffusion coefficient, supported by a theoretical condition for stable learning. This design captures intrinsic stochasticity, handles irregular sparse sampling, and yields subject-specific trajectories. Experiments on synthetic benchmarks and real-world longitudinal neuroimaging datasets show that IMMFM outperforms existing methods in both forecasting accuracy and further downstream tasks.
>
---
#### [replaced 032] Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples
- **分类: cs.NE; cs.AI; cs.CR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2209.03358v4](http://arxiv.org/pdf/2209.03358v4)**

> **作者:** Nuo Xu; Kaleel Mahmood; Haowen Fang; Ethan Rathbun; Caiwen Ding; Wujie Wen
>
> **备注:** Accepted manuscript. Published in *Neurocomputing*, Volume 656, 2025, Article 131506. Available online 12 September 2025. DOI: 10.1016/j.neucom.2025.131506
>
> **摘要:** Spiking neural networks (SNNs) have drawn much attention for their high energy efficiency and recent advances in classification performance. However, unlike traditional deep learning, the robustness of SNNs to adversarial examples remains underexplored. This work advances the adversarial attack side of SNNs and makes three major contributions. First, we show that successful white-box attacks on SNNs strongly depend on the surrogate gradient estimation technique, even for adversarially trained models. Second, using the best single surrogate gradient estimator, we study the transferability of adversarial examples between SNNs and state-of-the-art architectures such as Vision Transformers (ViTs) and CNNs. Our analysis reveals two major gaps: no existing white-box attack leverages multiple surrogate estimators, and no single attack effectively fools both SNNs and non-SNN models simultaneously. Third, we propose the Mixed Dynamic Spiking Estimation (MDSE) attack, which dynamically combines multiple surrogate gradients to overcome these gaps. MDSE produces adversarial examples that fool both SNN and non-SNN models, achieving up to 91.4% higher effectiveness on SNN/ViT ensembles and a 3x boost on adversarially trained SNN ensembles over Auto-PGD. Experiments span three datasets (CIFAR-10, CIFAR-100, ImageNet) and nineteen classifiers, and we will release code and models upon publication.
>
---
#### [replaced 033] Sustainable Self-evolution Adversarial Training
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.02270v2](http://arxiv.org/pdf/2412.02270v2)**

> **作者:** Wenxuan Wang; Chenglei Wang; Huihui Qi; Menghao Ye; Xuelin Qian; Peng Wang; Yanning Zhang
>
> **备注:** Accepted to ACMMM 2024
>
> **摘要:** With the wide application of deep neural network models in various computer vision tasks, there has been a proliferation of adversarial example generation strategies aimed at deeply exploring model security. However, existing adversarial training defense models, which rely on single or limited types of attacks under a one-time learning process, struggle to adapt to the dynamic and evolving nature of attack methods. Therefore, to achieve defense performance improvements for models in long-term applications, we propose a novel Sustainable Self-Evolution Adversarial Training (SSEAT) framework. Specifically, we introduce a continual adversarial defense pipeline to realize learning from various kinds of adversarial examples across multiple stages. Additionally, to address the issue of model catastrophic forgetting caused by continual learning from ongoing novel attacks, we propose an adversarial data replay module to better select more diverse and key relearning data. Furthermore, we design a consistency regularization strategy to encourage current defense models to learn more from previously trained ones, guiding them to retain more past knowledge and maintain accuracy on clean samples. Extensive experiments have been conducted to verify the efficacy of the proposed SSEAT defense method, which demonstrates superior defense performance and classification accuracy compared to competitors.Code is available at https://github.com/aup520/SSEAT
>
---
#### [replaced 034] Intelligent Healthcare Imaging Platform: A VLM-Based Framework for Automated Medical Image Analysis and Clinical Report Generation
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.13590v2](http://arxiv.org/pdf/2509.13590v2)**

> **作者:** Samer Al-Hamadani
>
> **备注:** 32 pages, 14 figures, 6 tables
>
> **摘要:** The rapid advancement of artificial intelligence (AI) in healthcare imaging has revolutionized diagnostic medicine and clinical decision-making processes. This work presents an intelligent multimodal framework for medical image analysis that leverages Vision-Language Models (VLMs) in healthcare diagnostics. The framework integrates Google Gemini 2.5 Flash for automated tumor detection and clinical report generation across multiple imaging modalities including CT, MRI, X-ray, and Ultrasound. The system combines visual feature extraction with natural language processing to enable contextual image interpretation, incorporating coordinate verification mechanisms and probabilistic Gaussian modeling for anomaly distribution. Multi-layered visualization techniques generate detailed medical illustrations, overlay comparisons, and statistical representations to enhance clinical confidence, with location measurement achieving 80 pixels average deviation. Result processing utilizes precise prompt engineering and textual analysis to extract structured clinical information while maintaining interpretability. Experimental evaluations demonstrated high performance in anomaly detection across multiple modalities. The system features a user-friendly Gradio interface for clinical workflow integration and demonstrates zero-shot learning capabilities to reduce dependence on large datasets. This framework represents a significant advancement in automated diagnostic support and radiological workflow efficiency, though clinical validation and multi-center evaluation are necessary prior to widespread adoption.
>
---
#### [replaced 035] VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.10979v4](http://arxiv.org/pdf/2411.10979v4)**

> **作者:** Yolo Yunlong Tang; Junjia Guo; Hang Hua; Susan Liang; Mingqian Feng; Xinyang Li; Rui Mao; Chao Huang; Jing Bi; Zeliang Zhang; Pooyan Fazli; Chenliang Xu
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** The advancement of Multimodal Large Language Models (MLLMs) has enabled significant progress in multimodal understanding, expanding their capacity to analyze video content. However, existing evaluation benchmarks for MLLMs primarily focus on abstract video comprehension, lacking a detailed assessment of their ability to understand video compositions, the nuanced interpretation of how visual elements combine and interact within highly compiled video contexts. We introduce VidComposition, a new benchmark specifically designed to evaluate the video composition understanding capabilities of MLLMs using carefully curated compiled videos and cinematic-level annotations. VidComposition includes 982 videos with 1706 multiple-choice questions, covering various compositional aspects such as camera movement, angle, shot size, narrative structure, character actions and emotions, etc. Our comprehensive evaluation of 33 open-source and proprietary MLLMs reveals a significant performance gap between human and model capabilities. This highlights the limitations of current MLLMs in understanding complex, compiled video compositions and offers insights into areas for further improvement. The leaderboard and evaluation code are available at https://yunlong10.github.io/VidComposition/.
>
---
#### [replaced 036] WAFT: Warping-Alone Field Transforms for Optical Flow
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21526v2](http://arxiv.org/pdf/2506.21526v2)**

> **作者:** Yihan Wang; Jia Deng
>
> **摘要:** We introduce Warping-Alone Field Transforms (WAFT), a simple and effective method for optical flow. WAFT is similar to RAFT but replaces cost volume with high-resolution warping, achieving better accuracy with lower memory cost. This design challenges the conventional wisdom that constructing cost volumes is necessary for strong performance. WAFT is a simple and flexible meta-architecture with minimal inductive biases and reliance on custom designs. Compared with existing methods, WAFT ranks 1st on Spring, Sintel, and KITTI benchmarks, achieves the best zero-shot generalization on KITTI, while being up to 4.1x faster than methods with similar performance. Code and model weights are available at https://github.com/princeton-vl/WAFT.
>
---
#### [replaced 037] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20426v2](http://arxiv.org/pdf/2505.20426v2)**

> **作者:** Yolo Yunlong Tang; Pinxin Liu; Mingqian Feng; Zhangyun Tan; Rui Mao; Chao Huang; Jing Bi; Yunzhong Xiao; Susan Liang; Hang Hua; Ali Vosoughi; Luchuan Song; Zeliang Zhang; Chenliang Xu
>
> **备注:** Accepted to NeurIPS 2025 DB Track
>
> **摘要:** Understanding perspective is fundamental to human visual perception, yet the extent to which multimodal large language models (MLLMs) internalize perspective geometry remains unclear. We introduce MMPerspective, the first benchmark specifically designed to systematically evaluate MLLMs' understanding of perspective through 10 carefully crafted tasks across three complementary dimensions: Perspective Perception, Reasoning, and Robustness. Our benchmark comprises 2,711 real-world and synthetic image instances with 5,083 question-answer pairs that probe key capabilities, such as vanishing point perception and counting, perspective type reasoning, line relationship understanding in 3D space, invariance to perspective-preserving transformations, etc. Through a comprehensive evaluation of 43 state-of-the-art MLLMs, we uncover significant limitations: while models demonstrate competence on surface-level perceptual tasks, they struggle with compositional reasoning and maintaining spatial consistency under perturbations. Our analysis further reveals intriguing patterns between model architecture, scale, and perspective capabilities, highlighting both robustness bottlenecks and the benefits of chain-of-thought prompting. MMPerspective establishes a valuable testbed for diagnosing and advancing spatial understanding in vision-language systems. Resources available at: https://yunlong10.github.io/MMPerspective/
>
---
#### [replaced 038] GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.14230v2](http://arxiv.org/pdf/2501.14230v2)**

> **作者:** Hanrui Wang; Ching-Chun Chang; Chun-Shien Lu; Christopher Leckie; Isao Echizen
>
> **摘要:** Deep neural networks are highly vulnerable to adversarial examples that inputs with small, carefully crafted perturbations that cause misclassification, making adversarial attacks an essential tool for robustness evaluation. Existing black-box attacks fall into three categories: query-only, transfer-only, and query-and-transfer, and vary in perturbation pattern and optimization strategy. However, no prior method jointly achieves query-and-transfer guidance, pixel-wise sparsity, and training-free direct optimization, leaving a gap between black-box flexibility and white-box precision. We present GreedyPixel, a new attack framework that fills this gap by combining a surrogate-derived pixel priority map with greedy, per-pixel optimization refined by query feedback. This design reduces the exponential brute-force search space to a tractable linear procedure, guarantees monotonic loss decrease and convergence to a coordinate-wise optimum, and concentrates perturbations on robust, semantically meaningful pixels to improve perceptual quality. Extensive experiments on CIFAR-10 and ImageNet under both white-box and black-box settings demonstrate that GreedyPixel achieves state-of-the-art attack success rates and produces visually imperceptible perturbations. Our results show that GreedyPixel bridges the precision gap between white-box and black-box attacks and provides a practical framework for fine-grained robustness evaluation. The implementation is available at https://github.com/azrealwang/greedypixel.
>
---
#### [replaced 039] Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.04759v2](http://arxiv.org/pdf/2510.04759v2)**

> **作者:** Chi Yan; Dan Xu
>
> **备注:** Project Page: https://yanchi-3dv.github.io/PG-Occ
>
> **摘要:** The 3D occupancy prediction task has witnessed remarkable progress in recent years, playing a crucial role in vision-based autonomous driving systems. While traditional methods are limited to fixed semantic categories, recent approaches have moved towards predicting text-aligned features to enable open-vocabulary text queries in real-world scenes. However, there exists a trade-off in text-aligned scene modeling: sparse Gaussian representation struggles to capture small objects in the scene, while dense representation incurs significant computational overhead. To address these limitations, we present PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables open-vocabulary 3D occupancy prediction. Our framework employs progressive online densification, a feed-forward strategy that gradually enhances the 3D Gaussian representation to capture fine-grained scene details. By iteratively enhancing the representation, the framework achieves increasingly precise and detailed scene understanding. Another key contribution is the introduction of an anisotropy-aware sampling strategy with spatio-temporal fusion, which adaptively assigns receptive fields to Gaussians at different scales and stages, enabling more effective feature aggregation and richer scene information capture. Through extensive evaluations, we demonstrate that PG-Occ achieves state-of-the-art performance with a relative 14.3% mIoU improvement over the previous best performing method. Code and pretrained models will be released upon publication on our project page: https://yanchi-3dv.github.io/PG-Occ
>
---
#### [replaced 040] LoDisc: Learning Global-Local Discriminative Features for Self-Supervised Fine-Grained Visual Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.04066v2](http://arxiv.org/pdf/2403.04066v2)**

> **作者:** Jialu Shi; Zhiqiang Wei; Jie Nie; Lei Huang
>
> **备注:** Accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)
>
> **摘要:** The self-supervised contrastive learning strategy has attracted considerable attention due to its exceptional ability in representation learning. However, current contrastive learning tends to learn global coarse-grained representations of the image that benefit generic object recognition, whereas such coarse-grained features are insufficient for fine-grained visual recognition. In this paper, we incorporate subtle local fine-grained feature learning into global self-supervised contrastive learning through a pure self-supervised global-local fine-grained contrastive learning framework. Specifically, a novel pretext task called local discrimination (LoDisc) is proposed to explicitly supervise the self-supervised model's focus toward local pivotal regions, which are captured by a simple but effective location-wise mask sampling strategy. We show that the LoDisc pretext task can effectively enhance fine-grained clues in important local regions and that the global-local framework further refines the fine-grained feature representations of images. Extensive experimental results on different fine-grained object recognition tasks demonstrate that the proposed method can lead to a decent improvement in different evaluation settings. The proposed method is also effective for general object recognition tasks.
>
---
#### [replaced 041] AerialVG: A Challenging Benchmark for Aerial Visual Grounding by Exploring Positional Relations
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.07836v4](http://arxiv.org/pdf/2504.07836v4)**

> **作者:** Junli Liu; Qizhi Chen; Zhigang Wang; Yiwen Tang; Yiting Zhang; Chi Yan; Dong Wang; Xuelong Li; Bin Zhao
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Visual grounding (VG) aims to localize target objects in an image based on natural language descriptions. In this paper, we propose AerialVG, a new task focusing on visual grounding from aerial views. Compared to traditional VG, AerialVG poses new challenges, \emph{e.g.}, appearance-based grounding is insufficient to distinguish among multiple visually similar objects, and positional relations should be emphasized. Besides, existing VG models struggle when applied to aerial imagery, where high-resolution images cause significant difficulties. To address these challenges, we introduce the first AerialVG dataset, consisting of 5K real-world aerial images, 50K manually annotated descriptions, and 103K objects. Particularly, each annotation in AerialVG dataset contains multiple target objects annotated with relative spatial relations, requiring models to perform comprehensive spatial reasoning. Furthermore, we propose an innovative model especially for the AerialVG task, where a Hierarchical Cross-Attention is devised to focus on target regions, and a Relation-Aware Grounding module is designed to infer positional relations. Experimental results validate the effectiveness of our dataset and method, highlighting the importance of spatial reasoning in aerial visual grounding. The code and dataset will be released.
>
---
#### [replaced 042] HBSplat: Robust Sparse-View Gaussian Reconstruction with Hybrid-Loss Guided Depth and Bidirectional Warping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24893v3](http://arxiv.org/pdf/2509.24893v3)**

> **作者:** Yu Ma; Guoliang Wei; Haihong Xiao; Yue Cheng
>
> **备注:** 14 pages, 21 figures
>
> **摘要:** Novel View Synthesis (NVS) from sparse views presents a formidable challenge in 3D reconstruction, where limited multi-view constraints lead to severe overfitting, geometric distortion, and fragmented scenes. While 3D Gaussian Splatting (3DGS) delivers real-time, high-fidelity rendering, its performance drastically deteriorates under sparse inputs, plagued by floating artifacts and structural failures. To address these challenges, we introduce HBSplat, a unified framework that elevates 3DGS by seamlessly integrating robust structural cues, virtual view constraints, and occluded region completion. Our core contributions are threefold: a Hybrid-Loss Depth Estimation module that ensures multi-view consistency by leveraging dense matching priors and integrating reprojection, point propagation, and smoothness constraints; a Bidirectional Warping Virtual View Synthesis method that enforces substantially stronger constraints by creating high-fidelity virtual views through bidirectional depth-image warping and multi-view fusion; and an Occlusion-Aware Reconstruction component that recovers occluded areas using a depth-difference mask and a learning-based inpainting model. Extensive evaluations on LLFF, Blender, and DTU benchmarks validate that HBSplat sets a new state-of-the-art, achieving up to 21.13 dB PSNR and 0.189 LPIPS, while maintaining real-time inference. Code is available at: https://github.com/eternalland/HBSplat.
>
---
#### [replaced 043] Point2RBox-v3: Self-Bootstrapping from Point Annotations via Integrated Pseudo-Label Refinement and Utilization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.26281v2](http://arxiv.org/pdf/2509.26281v2)**

> **作者:** Teng Zhang; Ziqian Fan; Mingxin Liu; Xin Zhang; Xudong Lu; Wentong Li; Yue Zhou; Yi Yu; Xiang Li; Junchi Yan; Xue Yang
>
> **备注:** 19pages, 5figures, 6tables
>
> **摘要:** Driven by the growing need for Oriented Object Detection (OOD), learning from point annotations under a weakly-supervised framework has emerged as a promising alternative to costly and laborious manual labeling. In this paper, we discuss two deficiencies in existing point-supervised methods: inefficient utilization and poor quality of pseudo labels. Therefore, we present Point2RBox-v3. At the core are two principles: 1) Progressive Label Assignment (PLA). It dynamically estimates instance sizes in a coarse yet intelligent manner at different stages of the training process, enabling the use of label assignment methods. 2) Prior-Guided Dynamic Mask Loss (PGDM-Loss). It is an enhancement of the Voronoi Watershed Loss from Point2RBox-v2, which overcomes the shortcomings of Watershed in its poor performance in sparse scenes and SAM's poor performance in dense scenes. To our knowledge, Point2RBox-v3 is the first model to employ dynamic pseudo labels for label assignment, and it creatively complements the advantages of SAM model with the watershed algorithm, which achieves excellent performance in both sparse and dense scenes. Our solution gives competitive performance, especially in scenarios with large variations in object size or sparse object occurrences: 66.09%/56.86%/41.28%/46.40%/19.60%/45.96% on DOTA-v1.0/DOTA-v1.5/DOTA-v2.0/DIOR/STAR/RSAR.
>
---
#### [replaced 044] Spatiotemporal Tile-based Attention-guided LSTMs for Traffic Video Prediction
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/1910.11030v4](http://arxiv.org/pdf/1910.11030v4)**

> **作者:** Tu Nguyen
>
> **备注:** Neurips 2019 Traffic4Cast Challenge, v4: added formal proofs
>
> **摘要:** This extended abstract describes our solution for the Traffic4Cast Challenge 2019. The task requires modeling both fine-grained (pixel-level) and coarse (region-level) spatial structure while preserving temporal relationships across long sequences. Building on Conv-LSTM ideas, we introduce a tile-aware, cascaded-memory Conv-LSTM augmented with cross-frame additive attention and a memory-flexible training scheme: frames are sampled per spatial tile so the model learns tile-local dynamics and per-tile memory cells can be updated sparsely, paged, or compressed to scale to large maps. We provide a compact theoretical analysis (tight softmax/attention Lipschitz bound and a tiling error lower bound) explaining stability and the memory-accuracy tradeoffs, and empirically demonstrate improved scalability and competitive forecasting performance on large-scale traffic heatmaps.
>
---
#### [replaced 045] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20444v2](http://arxiv.org/pdf/2505.20444v2)**

> **作者:** Haoran Li; Yingjie Qin; Baoyuan Ou; Lai Xu; Ruiwen Xu
>
> **备注:** NeurIPS 2025
>
> **摘要:** Vision-Language Models (VLMs) have made significant progress in multimodal tasks. However, their performance often deteriorates in long-context scenarios, particularly long videos. While Rotary Position Embedding (RoPE) has been widely adopted for length generalization in Large Language Models (LLMs), extending vanilla RoPE to capture the intricate spatial-temporal dependencies in videos remains an unsolved challenge. Existing methods typically allocate different frequencies within RoPE to encode 3D positional information. However, these allocation strategies mainly rely on heuristics, lacking in-depth theoretical analysis. In this paper, we first study how different allocation strategies impact the long-context capabilities of VLMs. Our analysis reveals that current multimodal RoPEs fail to reliably capture semantic similarities over extended contexts. To address this issue, we propose HoPE, a Hybrid of Position Embedding designed to improve the long-context capabilities of VLMs. HoPE introduces a hybrid frequency allocation strategy for reliable semantic modeling over arbitrarily long contexts, and a dynamic temporal scaling mechanism to facilitate robust learning and flexible inference across diverse context lengths. Extensive experiments across four video benchmarks on long video understanding and retrieval tasks demonstrate that HoPE consistently outperforms existing methods, confirming its effectiveness. Our code is available at https://github.com/hrlics/HoPE.
>
---
#### [replaced 046] Unified Unsupervised Anomaly Detection via Matching Cost Filtering
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2510.03363v2](http://arxiv.org/pdf/2510.03363v2)**

> **作者:** Zhe Zhang; Mingxiu Cai; Gaochang Wu; Jing Zhang; Lingqiao Liu; Dacheng Tao; Tianyou Chai; Xiatian Zhu
>
> **备注:** 63 pages (main paper and supplementary material), 39 figures, 58 tables
>
> **摘要:** Unsupervised anomaly detection (UAD) aims to identify image- and pixel-level anomalies using only normal training data, with wide applications such as industrial inspection and medical analysis, where anomalies are scarce due to privacy concerns and cold-start constraints. Existing methods, whether reconstruction-based (restoring normal counterparts) or embedding-based (pretrained representations), fundamentally conduct image- or feature-level matching to generate anomaly maps. Nonetheless, matching noise has been largely overlooked, limiting their detection ability. Beyond earlier focus on unimodal RGB-based UAD, recent advances expand to multimodal scenarios, e.g., RGB-3D and RGB-Text, enabled by point cloud sensing and vision-language models. Despite shared challenges, these lines remain largely isolated, hindering a comprehensive understanding and knowledge transfer. In this paper, we advocate unified UAD for both unimodal and multimodal settings in the matching perspective. Under this insight, we present Unified Cost Filtering (UCF), a generic post-hoc refinement framework for refining anomaly cost volume of any UAD model. The cost volume is constructed by matching a test sample against normal samples from the same or different modalities, followed by a learnable filtering module with multi-layer attention guidance from the test sample, mitigating matching noise and highlighting subtle anomalies. Comprehensive experiments on 22 diverse benchmarks demonstrate the efficacy of UCF in enhancing a variety of UAD methods, consistently achieving new state-of-the-art results in both unimodal (RGB) and multimodal (RGB-3D, RGB-Text) UAD scenarios. Code and models will be released at https://github.com/ZHE-SAPI/CostFilter-AD.
>
---
#### [replaced 047] Polyp-Gen: Realistic and Diverse Polyp Image Generation for Endoscopic Dataset Expansion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.16679v3](http://arxiv.org/pdf/2501.16679v3)**

> **作者:** Shengyuan Liu; Zhen Chen; Qiushi Yang; Weihao Yu; Di Dong; Jiancong Hu; Yixuan Yuan
>
> **备注:** Accepted by ICRA 2025
>
> **摘要:** Automated diagnostic systems (ADS) have shown significant potential in the early detection of polyps during endoscopic examinations, thereby reducing the incidence of colorectal cancer. However, due to high annotation costs and strict privacy concerns, acquiring high-quality endoscopic images poses a considerable challenge in the development of ADS. Despite recent advancements in generating synthetic images for dataset expansion, existing endoscopic image generation algorithms failed to accurately generate the details of polyp boundary regions and typically required medical priors to specify plausible locations and shapes of polyps, which limited the realism and diversity of the generated images. To address these limitations, we present Polyp-Gen, the first full-automatic diffusion-based endoscopic image generation framework. Specifically, we devise a spatial-aware diffusion training scheme with a lesion-guided loss to enhance the structural context of polyp boundary regions. Moreover, to capture medical priors for the localization of potential polyp areas, we introduce a hierarchical retrieval-based sampling strategy to match similar fine-grained spatial features. In this way, our Polyp-Gen can generate realistic and diverse endoscopic images for building reliable ADS. Extensive experiments demonstrate the state-of-the-art generation quality, and the synthetic images can improve the downstream polyp detection task. Additionally, our Polyp-Gen has shown remarkable zero-shot generalizability on other datasets. The source code is available at https://github.com/CUHK-AIM-Group/Polyp-Gen.
>
---
#### [replaced 048] MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20772v2](http://arxiv.org/pdf/2505.20772v2)**

> **作者:** Hongjia Liu; Rongzhen Zhao; Haohan Chen; Joni Pajarinen
>
> **摘要:** Learning object-level, structured representations is widely regarded as a key to better generalization in vision and underpins the design of next-generation Pre-trained Vision Models (PVMs). Mainstream Object-Centric Learning (OCL) methods adopt Slot Attention or its variants to iteratively aggregate objects' super-pixels into a fixed set of query feature vectors, termed slots. However, their reliance on a static slot count leads to an object being represented as multiple parts when the number of objects varies. We introduce MetaSlot, a plug-and-play Slot Attention variant that adapts to variable object counts. MetaSlot (i) maintains a codebook that holds prototypes of objects in a dataset by vector-quantizing the resulting slot representations; (ii) removes duplicate slots from the traditionally aggregated slots by quantizing them with the codebook; and (iii) injects progressively weaker noise into the Slot Attention iterations to accelerate and stabilize the aggregation. MetaSlot is a general Slot Attention variant that can be seamlessly integrated into existing OCL architectures. Across multiple public datasets and tasks--including object discovery and recognition--models equipped with MetaSlot achieve significant performance gains and markedly interpretable slot representations, compared with existing Slot Attention variants.
>
---
#### [replaced 049] KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.19807v3](http://arxiv.org/pdf/2506.19807v3)**

> **作者:** Baochang Ren; Shuofei Qiao; Da Zheng; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at https://github.com/zjunlp/KnowRL.
>
---
#### [replaced 050] Generative Pre-trained Autoregressive Diffusion Transformer
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07344v5](http://arxiv.org/pdf/2505.07344v5)**

> **作者:** Yuan Zhang; Jiacheng Jiang; Guoqing Ma; Zhiying Lu; Haoyang Huang; Jianlong Yuan; Nan Duan; Daxin Jiang
>
> **摘要:** In this work, we present GPDiT, a Generative Pre-trained Autoregressive Diffusion Transformer that unifies the strengths of diffusion and autoregressive modeling for long-range video synthesis, within a continuous latent space. Instead of predicting discrete tokens, GPDiT autoregressively predicts future latent frames using a diffusion loss, enabling natural modeling of motion dynamics and semantic consistency across frames. This continuous autoregressive framework not only enhances generation quality but also endows the model with representation capabilities. Additionally, we introduce a lightweight causal attention variant and a parameter-free rotation-based time-conditioning mechanism, improving both the training and inference efficiency. Extensive experiments demonstrate that GPDiT achieves strong performance in video generation quality, video representation ability, and few-shot learning tasks, highlighting its potential as an effective framework for video modeling in continuous space.
>
---
#### [replaced 051] Uncertainty-Aware Remaining Lifespan Prediction from Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13430v3](http://arxiv.org/pdf/2506.13430v3)**

> **作者:** Tristan Kenneweg; Philip Kenneweg; Barbara Hammer
>
> **备注:** Submitted to ISVC 2025
>
> **摘要:** Predicting mortality-related outcomes from images offers the prospect of accessible, noninvasive, and scalable health screening. We present a method that leverages pretrained vision transformer foundation models to estimate remaining lifespan from facial and whole-body images, alongside robust uncertainty quantification. We show that predictive uncertainty varies systematically with the true remaining lifespan, and that this uncertainty can be effectively modeled by learning a Gaussian distribution for each sample. Our approach achieves state-of-the-art mean absolute error (MAE) of 7.41 years on an established dataset, and further achieves 4.91 and 4.99 years MAE on two new, higher-quality datasets curated and published in this work. Importantly, our models provide calibrated uncertainty estimates, as demonstrated by a bucketed expected calibration error of 0.82 years on the Faces Dataset. While not intended for clinical deployment, these results highlight the potential of extracting medically relevant signals from images. We make all code and datasets available to facilitate further research.
>
---
#### [replaced 052] V2Xum-LLM: Cross-Modal Video Summarization with Temporal Prompt Instruction Tuning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.12353v3](http://arxiv.org/pdf/2404.12353v3)**

> **作者:** Hang Hua; Yolo Yunlong Tang; Chenliang Xu; Jiebo Luo
>
> **备注:** Accepted to AAAI 2025
>
> **摘要:** Video summarization aims to create short, accurate, and cohesive summaries of longer videos. Despite the existence of various video summarization datasets, a notable limitation is their limited amount of source videos, which hampers the effective training of advanced large vision-language models (VLMs). Additionally, most existing datasets are created for video-to-video summarization, overlooking the contemporary need for multimodal video content summarization. Recent efforts have been made to expand from unimodal to multimodal video summarization, categorizing the task into three sub-tasks based on the summary's modality: video-to-video (V2V), video-to-text (V2T), and a combination of video and text summarization (V2VT). However, the textual summaries in previous multimodal datasets are inadequate. To address these issues, we introduce Instruct-V2Xum, a cross-modal video summarization dataset featuring 30,000 diverse videos sourced from YouTube, with lengths ranging from 40 to 940 seconds and an average summarization ratio of 16.39%. Each video summary in Instruct-V2Xum is paired with a textual summary that references specific frame indexes, facilitating the generation of aligned video and textual summaries. In addition, we propose a new video summarization framework named V2Xum-LLM. V2Xum-LLM, specifically V2Xum-LLaMA in this study, is the first framework that unifies different video summarization tasks into one large language model's (LLM) text decoder and achieves task-controllable video summarization with temporal prompts and task instructions. Experiments show that V2Xum-LLaMA outperforms strong baseline models on multiple video summarization tasks. Furthermore, we propose an enhanced evaluation metric for V2V and V2VT summarization tasks.
>
---
#### [replaced 053] Robot Learning from Any Images
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.22970v2](http://arxiv.org/pdf/2509.22970v2)**

> **作者:** Siheng Zhao; Jiageng Mao; Wei Chow; Zeyu Shangguan; Tianheng Shi; Rong Xue; Yuxi Zheng; Yijia Weng; Yang You; Daniel Seita; Leonidas Guibas; Sergey Zakharov; Vitor Guizilini; Yue Wang
>
> **备注:** CoRL 2025 camera ready
>
> **摘要:** We introduce RoLA, a framework that transforms any in-the-wild image into an interactive, physics-enabled robotic environment. Unlike previous methods, RoLA operates directly on a single image without requiring additional hardware or digital assets. Our framework democratizes robotic data generation by producing massive visuomotor robotic demonstrations within minutes from a wide range of image sources, including camera captures, robotic datasets, and Internet images. At its core, our approach combines a novel method for single-view physical scene recovery with an efficient visual blending strategy for photorealistic data collection. We demonstrate RoLA's versatility across applications like scalable robotic data generation and augmentation, robot learning from Internet images, and single-image real-to-sim-to-real systems for manipulators and humanoids. Video results are available at https://sihengz02.github.io/RoLA .
>
---
#### [replaced 054] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18369v3](http://arxiv.org/pdf/2506.18369v3)**

> **作者:** Yeongtak Oh; Dohyun Chung; Juhyeon Shin; Sangha Park; Johan Barthelemy; Jisoo Mok; Sungroh Yoon
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Recent multi-modal large language models (MLLMs) often struggle to generate personalized image captions, even when trained on high-quality captions. In this work, we observe that such limitations persist in existing post-training-based MLLM personalization methods. Specifically, despite being post-tuned with large-scale caption data through supervised fine-tuning (SFT), these models frequently fail to produce faithful descriptions in real-world scenarios, such as multi-concept image captioning. However, acquiring large-scale, high-quality captions for such complex settings is both costly and difficult. To address the data-centric nature of SFT, we propose a reinforcement learning (RL)-based post-training framework. To the best of our knowledge, this is the first RL-based approach to post-train MLLMs for personalized image captioning. Our method significantly enhances both visual recognition and personalized generation capabilities of MLLMs, and consistently outperforms existing SFT-based baselines, especially in the challenging multi-concept image captioning task. Project page: https://github.com/oyt9306/RePIC
>
---
#### [replaced 055] Train-Free Segmentation in MRI with Cubical Persistent Homology
- **分类: eess.IV; cs.CG; cs.CV; cs.LG; 55N31, 68-04, 92-08, 68U10**

- **链接: [http://arxiv.org/pdf/2401.01160v2](http://arxiv.org/pdf/2401.01160v2)**

> **作者:** Anton François; Raphaël Tinarrage
>
> **备注:** Preprint, 36 pages, 18 figures, 4 tables. For associated code, see https://github.com/antonfrancois/gliomaSegmentation_TDA
>
> **摘要:** We present a new general framework for segmentation of MRI scans based on Topological Data Analysis (TDA), offering several advantages over traditional machine learning approaches. The pipeline proceeds in three steps, first identifying the whole object to segment via automatic thresholding, then detecting a distinctive subset whose topology is known in advance, and finally deducing the various components of the segmentation. Unlike most prior TDA uses in medical image segmentation, which are typically embedded within deep networks, our approach is a standalone method tailored to MRI. A key ingredient is the localization of representative cycles from the persistence diagram, which enables interpretable mappings from topological features to anatomical components. In particular, the method offers the ability to perform segmentation without the need for large annotated datasets. Its modular design makes it adaptable to a wide range of data segmentation challenges. We validate the framework on three applications: glioblastoma segmentation in brain MRI, where a sphere is to be detected; myocardium in cardiac MRI, forming a cylinder; and cortical plate detection in fetal brain MRI, whose 2D slices are circles. We compare our method with established supervised and unsupervised baselines.
>
---
#### [replaced 056] Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20612v3](http://arxiv.org/pdf/2505.20612v3)**

> **作者:** Peter Robicheaux; Matvei Popov; Anish Madan; Isaac Robinson; Joseph Nelson; Deva Ramanan; Neehar Peri
>
> **备注:** The first two authors contributed equally. This work has been accepted to the Neural Information Processing Systems (NeurIPS) 2025 Datasets & Benchmark Track. Project Page: https://rf100-vl.org/
>
> **摘要:** Vision-language models (VLMs) trained on internet-scale data achieve remarkable zero-shot detection performance on common objects like car, truck, and pedestrian. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. Rather than simply re-training VLMs on more visual data, we argue that one should align VLMs to new concepts with annotation instructions containing a few visual examples and rich textual descriptions. To this end, we introduce Roboflow100-VL, a large-scale collection of 100 multi-modal object detection datasets with diverse concepts not commonly found in VLM pre-training. We evaluate state-of-the-art models on our benchmark in zero-shot, few-shot, semi-supervised, and fully-supervised settings, allowing for comparison across data regimes. Notably, we find that VLMs like GroundingDINO and Qwen2.5-VL achieve less than 2% zero-shot accuracy on challenging medical imaging datasets within Roboflow100-VL, demonstrating the need for few-shot concept alignment. Lastly, we discuss our recent CVPR 2025 Foundational FSOD competition and share insights from the community. Notably, the winning team significantly outperforms our baseline by 17 mAP! Our code and dataset are available at https://github.com/roboflow/rf100-vl and https://universe.roboflow.com/rf100-vl/.
>
---
#### [replaced 057] Is My Data in Your AI? Membership Inference Test (MINT) applied to Face Biometrics
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.09225v4](http://arxiv.org/pdf/2402.09225v4)**

> **作者:** Daniel DeAlcala; Aythami Morales; Julian Fierrez; Gonzalo Mancera; Ruben Tolosana; Javier Ortega-Garcia
>
> **备注:** 11 pages main text + 2 pages references and 1 pages appendix
>
> **摘要:** This article introduces the Membership Inference Test (MINT), a novel approach that aims to empirically assess if given data was used during the training of AI/ML models. Specifically, we propose two MINT architectures designed to learn the distinct activation patterns that emerge when an Audited Model is exposed to data used during its training process. These architectures are based on Multilayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs). The experimental framework focuses on the challenging task of Face Recognition, considering three state-of-the-art Face Recognition systems. Experiments are carried out using six publicly available databases, comprising over 22 million face images in total. Different experimental scenarios are considered depending on the context of the AI model to test. Our proposed MINT approach achieves promising results, with up to 90\% accuracy, indicating the potential to recognize if an AI model has been trained with specific data. The proposed MINT approach can serve to enforce privacy and fairness in several AI applications, e.g., revealing if sensitive or private data was used for training or tuning Large Language Models (LLMs).
>
---
#### [replaced 058] Rethinking Inter-LoRA Orthogonality in Adapter Merging: Insights from Orthogonal Monte Carlo Dropout
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.03262v2](http://arxiv.org/pdf/2510.03262v2)**

> **作者:** Andi Zhang; Xuan Ding; Haofan Wang; Steven McDonagh; Samuel Kaski
>
> **摘要:** We propose Orthogonal Monte Carlo Dropout, a mechanism that enforces strict orthogonality when combining sparse semantic vectors without extra time complexity. Low-Rank Adaptation (LoRA), a popular fine-tuning method for large models, typically trains a module to represent a specific concept such as an object or a style. When multiple LoRA modules are merged, for example to generate an object in a particular style, their outputs (semantic vectors) may interfere with each other. Our method guarantees that merged LoRA modules remain orthogonal and thus free from direct interference. However, empirical analysis reveals that such orthogonality does not lead to the semantic disentanglement highlighted in prior work on compositional adaptation. This finding suggests that inter-LoRA orthogonality alone may be insufficient for achieving true semantic compositionality, prompting a re-examination of its role in adapter merging.
>
---
#### [replaced 059] Taming Diffusion Models for Image Restoration: A Review
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.10353v3](http://arxiv.org/pdf/2409.10353v3)**

> **作者:** Ziwei Luo; Fredrik K. Gustafsson; Zheng Zhao; Jens Sjölund; Thomas B. Schön
>
> **备注:** This paper has been published in Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences
>
> **摘要:** Diffusion models have achieved remarkable progress in generative modelling, particularly in enhancing image quality to conform to human preferences. Recently, these models have also been applied to low-level computer vision for photo-realistic image restoration (IR) in tasks such as image denoising, deblurring, dehazing, etc. In this review paper, we introduce key constructions in diffusion models and survey contemporary techniques that make use of diffusion models in solving general IR tasks. Furthermore, we point out the main challenges and limitations of existing diffusion-based IR frameworks and provide potential directions for future work.
>
---
#### [replaced 060] Human Action Recognition from Point Clouds over Time
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.05506v2](http://arxiv.org/pdf/2510.05506v2)**

> **作者:** James Dickens
>
> **摘要:** Recent research into human action recognition (HAR) has focused predominantly on skeletal action recognition and video-based methods. With the increasing availability of consumer-grade depth sensors and Lidar instruments, there is a growing opportunity to leverage dense 3D data for action recognition, to develop a third way. This paper presents a novel approach for recognizing actions from 3D videos by introducing a pipeline that segments human point clouds from the background of a scene, tracks individuals over time, and performs body part segmentation. The method supports point clouds from both depth sensors and monocular depth estimation. At the core of the proposed HAR framework is a novel backbone for 3D action recognition, which combines point-based techniques with sparse convolutional networks applied to voxel-mapped point cloud sequences. Experiments incorporate auxiliary point features including surface normals, color, infrared intensity, and body part parsing labels, to enhance recognition accuracy. Evaluation on the NTU RGB- D 120 dataset demonstrates that the method is competitive with existing skeletal action recognition algorithms. Moreover, combining both sensor-based and estimated depth inputs in an ensemble setup, this approach achieves 89.3% accuracy when different human subjects are considered for training and testing, outperforming previous point cloud action recognition methods.
>
---
#### [replaced 061] Lossy Neural Compression for Geospatial Analytics: A Review
- **分类: eess.SP; cs.AI; cs.CV; cs.LG; physics.geo-ph**

- **链接: [http://arxiv.org/pdf/2503.01505v2](http://arxiv.org/pdf/2503.01505v2)**

> **作者:** Carlos Gomes; Isabelle Wittmann; Damien Robert; Johannes Jakubik; Tim Reichelt; Michele Martone; Stefano Maurogiovanni; Rikard Vinge; Jonas Hurst; Erik Scheurer; Rocco Sedona; Thomas Brunschwiler; Stefan Kesselheim; Matej Batic; Philip Stier; Jan Dirk Wegner; Gabriele Cavallaro; Edzer Pebesma; Michael Marszalek; Miguel A Belenguer-Plomer; Kennedy Adriko; Paolo Fraccaro; Romeo Kienzler; Rania Briq; Sabrina Benassou; Michele Lazzarini; Conrad M Albrecht
>
> **备注:** self-consistent review paper
>
> **摘要:** Over the past decades, there has been an explosion in the amount of available Earth Observation (EO) data. The unprecedented coverage of the Earth's surface and atmosphere by satellite imagery has resulted in large volumes of data that must be transmitted to ground stations, stored in data centers, and distributed to end users. Modern Earth System Models (ESMs) face similar challenges, operating at high spatial and temporal resolutions, producing petabytes of data per simulated day. Data compression has gained relevance over the past decade, with neural compression (NC) emerging from deep learning and information theory, making EO data and ESM outputs ideal candidates due to their abundance of unlabeled data. In this review, we outline recent developments in NC applied to geospatial data. We introduce the fundamental concepts of NC including seminal works in its traditional applications to image and video compression domains with focus on lossy compression. We discuss the unique characteristics of EO and ESM data, contrasting them with "natural images", and explain the additional challenges and opportunities they present. Moreover, we review current applications of NC across various EO modalities and explore the limited efforts in ESM compression to date. The advent of self-supervised learning (SSL) and foundation models (FM) has advanced methods to efficiently distill representations from vast unlabeled data. We connect these developments to NC for EO, highlighting the similarities between the two fields and elaborate on the potential of transferring compressed feature representations for machine--to--machine communication. Based on insights drawn from this review, we devise future directions relevant to applications in EO and ESM.
>
---
#### [replaced 062] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03538v3](http://arxiv.org/pdf/2506.03538v3)**

> **作者:** Chengqi Li; Zhihao Shi; Yangdi Lu; Wenbo He; Xiangyu Xu
>
> **备注:** NeurIPS 2025 Spotlight; Project page: https://steveli88.github.io/AsymGS/
>
> **摘要:** 3D reconstruction from in-the-wild images remains a challenging task due to inconsistent lighting conditions and transient distractors. Existing methods typically rely on heuristic strategies to handle the low-quality training data, which often struggle to produce stable and consistent reconstructions, frequently resulting in visual artifacts. In this work, we propose \modelname{}, a novel framework that leverages the stochastic nature of these artifacts: they tend to vary across different training runs due to minor randomness. Specifically, our method trains two 3D Gaussian Splatting (3DGS) models in parallel, enforcing a consistency constraint that encourages convergence on reliable scene geometry while suppressing inconsistent artifacts. To prevent the two models from collapsing into similar failure modes due to confirmation bias, we introduce a divergent masking strategy that applies two complementary masks: a multi-cue adaptive mask and a self-supervised soft mask, which leads to an asymmetric training process of the two models, reducing shared error modes. In addition, to improve the efficiency of model training, we introduce a lightweight variant called Dynamic EMA Proxy, which replaces one of the two models with a dynamically updated Exponential Moving Average (EMA) proxy, and employs an alternating masking strategy to preserve divergence. Extensive experiments on challenging real-world datasets demonstrate that our method consistently outperforms existing approaches while achieving high efficiency. See the project website at https://steveli88.github.io/AsymGS.
>
---
#### [replaced 063] Maximising the Utility of Validation Sets for Imbalanced Noisy-label Meta-learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2208.08132v4](http://arxiv.org/pdf/2208.08132v4)**

> **作者:** Dung Anh Hoang; Cuong Nguyen; Belagiannis Vasileios; Thanh-Toan Do; Gustavo Carneiro
>
> **摘要:** Meta-learning is an effective method to handle imbalanced and noisy-label learning, but it depends on a validation set containing randomly selected, manually labelled and balanced distributed samples. The random selection and manual labelling and balancing of this validation set is not only sub-optimal for meta-learning, but it also scales poorly with the number of classes. Hence, recent meta-learning papers have proposed ad-hoc heuristics to automatically build and label this validation set, but these heuristics are still sub-optimal for meta-learning. In this paper, we analyse the meta-learning algorithm and propose new criteria to characterise the utility of the validation set, based on: 1) the informativeness of the validation set; 2) the class distribution balance of the set; and 3) the correctness of the labels of the set. Furthermore, we propose a new imbalanced noisy-label meta-learning (INOLML) algorithm that automatically builds a validation set by maximising its utility using the criteria above. Our method shows significant improvements over previous meta-learning approaches and sets the new state-of-the-art on several benchmarks.
>
---
#### [replaced 064] GPS-MTM: Capturing Pattern of Normalcy in GPS-Trajectories with self-supervised learning
- **分类: cs.LG; cs.AI; cs.CV; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.24031v2](http://arxiv.org/pdf/2509.24031v2)**

> **作者:** Umang Garg; Bowen Zhang; Anantajit Subrahmanya; Chandrakanth Gudavalli; BS Manjunath
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Foundation models have driven remarkable progress in text, vision, and video understanding, and are now poised to unlock similar breakthroughs in trajectory modeling. We introduce the GPSMasked Trajectory Transformer (GPS-MTM), a foundation model for large-scale mobility data that captures patterns of normalcy in human movement. Unlike prior approaches that flatten trajectories into coordinate streams, GPS-MTM decomposes mobility into two complementary modalities: states (point-of-interest categories) and actions (agent transitions). Leveraging a bi-directional Transformer with a self-supervised masked modeling objective, the model reconstructs missing segments across modalities, enabling it to learn rich semantic correlations without manual labels. Across benchmark datasets, including Numosim-LA, Urban Anomalies, and Geolife, GPS-MTM consistently outperforms on downstream tasks such as trajectory infilling and next-stop prediction. Its advantages are most pronounced in dynamic tasks (inverse and forward dynamics), where contextual reasoning is critical. These results establish GPS-MTM as a robust foundation model for trajectory analytics, positioning mobility data as a first-class modality for large-scale representation learning. Code is released for further reference.
>
---
#### [replaced 065] Efficient Flow Matching using Latent Variables
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.04486v3](http://arxiv.org/pdf/2505.04486v3)**

> **作者:** Anirban Samaddar; Yixuan Sun; Viktor Nilsson; Sandeep Madireddy
>
> **摘要:** Flow matching models have shown great potential in image generation tasks among probabilistic generative models. However, most flow matching models in the literature do not explicitly utilize the underlying clustering structure in the target data when learning the flow from a simple source distribution like the standard Gaussian. This leads to inefficient learning, especially for many high-dimensional real-world datasets, which often reside in a low-dimensional manifold. To this end, we present $\texttt{Latent-CFM}$, which provides efficient training strategies by conditioning on the features extracted from data using pretrained deep latent variable models. Through experiments on synthetic data from multi-modal distributions and widely used image benchmark datasets, we show that $\texttt{Latent-CFM}$ exhibits improved generation quality with significantly less training and computation than state-of-the-art flow matching models by adopting pretrained lightweight latent variable models. Beyond natural images, we consider generative modeling of spatial fields stemming from physical processes. Using a 2d Darcy flow dataset, we demonstrate that our approach generates more physically accurate samples than competing approaches. In addition, through latent space analysis, we demonstrate that our approach can be used for conditional image generation conditioned on latent features, which adds interpretability to the generation process.
>
---
#### [replaced 066] OneVision: An End-to-End Generative Framework for Multi-view E-commerce Vision Search
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.05759v2](http://arxiv.org/pdf/2510.05759v2)**

> **作者:** Zexin Zheng; Huangyu Dai; Lingtao Mao; Xinyu Sun; Zihan Liang; Ben Chen; Yuqing Ding; Chenyi Lei; Wenwu Ou; Han Li; Kun Gai
>
> **备注:** Some of the online experimental results in the paper are significantly different from the actual results, and need to be re-experimented and revised before submission. The current version is prone to misunderstanding
>
> **摘要:** Traditional vision search, similar to search and recommendation systems, follows the multi-stage cascading architecture (MCA) paradigm to balance efficiency and conversion. Specifically, the query image undergoes feature extraction, recall, pre-ranking, and ranking stages, ultimately presenting the user with semantically similar products that meet their preferences. This multi-view representation discrepancy of the same object in the query and the optimization objective collide across these stages, making it difficult to achieve Pareto optimality in both user experience and conversion. In this paper, an end-to-end generative framework, OneVision, is proposed to address these problems. OneVision builds on VRQ, a vision-aligned residual quantization encoding, which can align the vastly different representations of an object across multiple viewpoints while preserving the distinctive features of each product as much as possible. Then a multi-stage semantic alignment scheme is adopted to maintain strong visual similarity priors while effectively incorporating user-specific information for personalized preference generation. In offline evaluations, OneVision performs on par with online MCA, while improving inference efficiency by 21% through dynamic pruning. In A/B tests, it achieves significant online improvements: +2.15% item CTR, +2.27% CVR, and +3.12% order volume. These results demonstrate that a semantic ID centric, generative architecture can unify retrieval and personalization while simplifying the serving pathway.
>
---
#### [replaced 067] Gaze Estimation for Human-Robot Interaction: Analysis Using the NICO Platform
- **分类: cs.CV; cs.RO; I.4.9**

- **链接: [http://arxiv.org/pdf/2509.24001v2](http://arxiv.org/pdf/2509.24001v2)**

> **作者:** Matej Palider; Omar Eldardeer; Viktor Kocur
>
> **备注:** Code available at http://github.com/kocurvik/nico_gaze
>
> **摘要:** This paper evaluates the current gaze estimation methods within an HRI context of a shared workspace scenario. We introduce a new, annotated dataset collected with the NICO robotic platform. We evaluate four state-of-the-art gaze estimation models. The evaluation shows that the angular errors are close to those reported on general-purpose benchmarks. However, when expressed in terms of distance in the shared workspace the best median error is 16.48 cm quantifying the practical limitations of current methods. We conclude by discussing these limitations and offering recommendations on how to best integrate gaze estimation as a modality in HRI systems.
>
---
#### [replaced 068] DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting
- **分类: cs.CV; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.15690v3](http://arxiv.org/pdf/2507.15690v3)**

> **作者:** Hung Nguyen; Runfa Li; An Le; Truong Nguyen
>
> **备注:** Accepted to VCIP 2025
>
> **摘要:** Sparse-view 3D Gaussian Splatting (3DGS) presents significant challenges in reconstructing high-quality novel views, as it often overfits to the widely-varying high-frequency (HF) details of the sparse training views. While frequency regularization can be a promising approach, its typical reliance on Fourier transforms causes difficult parameter tuning and biases towards detrimental HF learning. We propose DWTGS, a framework that rethinks frequency regularization by leveraging wavelet-space losses that provide additional spatial supervision. Specifically, we supervise only the low-frequency (LF) LL subbands at multiple DWT levels, while enforcing sparsity on the HF HH subband in a self-supervised manner. Experiments across benchmarks show that DWTGS consistently outperforms Fourier-based counterparts, as this LF-centric strategy improves generalization and reduces HF hallucinations.
>
---
#### [replaced 069] BIM-Constrained Optimization for Accurate Localization and Deviation Correction in Construction Monitoring
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17693v2](http://arxiv.org/pdf/2504.17693v2)**

> **作者:** Asier Bikandi-Noya; Muhammad Shaheer; Hriday Bavle; Jayan Jevanesan; Holger Voos; Jose Luis Sanchez-Lopez
>
> **摘要:** Augmented reality (AR) applications for construction monitoring rely on real-time environmental tracking to visualize architectural elements. However, construction sites present significant challenges for traditional tracking methods due to featureless surfaces, dynamic changes, and drift accumulation, leading to misalignment between digital models and the physical world. This paper proposes a BIM-aware drift correction method to address these challenges. Instead of relying solely on SLAM-based localization, we align ``as-built" detected planes from the real-world environment with ``as-planned" architectural planes in BIM. Our method performs robust plane matching and computes a transformation (TF) between SLAM (S) and BIM (B) origin frames using optimization techniques, minimizing drift over time. By incorporating BIM as prior structural knowledge, we can achieve improved long-term localization and enhanced AR visualization accuracy in noisy construction environments. The method is evaluated through real-world experiments, showing significant reductions in drift-induced errors and optimized alignment consistency. On average, our system achieves a reduction of 52.24% in angular deviations and a reduction of 60.8% in the distance error of the matched walls compared to the initial manual alignment by the user.
>
---
#### [replaced 070] SubGrapher: Visual Fingerprinting of Chemical Structures
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19695v2](http://arxiv.org/pdf/2504.19695v2)**

> **作者:** Lucas Morin; Gerhard Ingmar Meijer; Valéry Weber; Luc Van Gool; Peter W. J. Staar
>
> **摘要:** Automatic extraction of chemical structures from scientific literature plays a crucial role in accelerating research across fields ranging from drug discovery to materials science. Patent documents, in particular, contain molecular information in visual form, which is often inaccessible through traditional text-based searches. In this work, we introduce SubGrapher, a method for the visual fingerprinting of chemical structure images. Unlike conventional Optical Chemical Structure Recognition (OCSR) models that attempt to reconstruct full molecular graphs, SubGrapher focuses on extracting molecular fingerprints directly from chemical structure images. Using learning-based instance segmentation, SubGrapher identifies functional groups and carbon backbones, constructing a substructure-based fingerprint that enables chemical structure retrieval. Our approach is evaluated against state-of-the-art OCSR and fingerprinting methods, demonstrating superior retrieval performance and robustness across diverse molecular depictions. The dataset, models, and code are publicly available.
>
---
#### [replaced 071] Optimizing Breast Cancer Detection in Mammograms: A Comprehensive Study of Transfer Learning, Resolution Reduction, and Multi-View Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19945v3](http://arxiv.org/pdf/2503.19945v3)**

> **作者:** Daniel G. P. Petrini; Hae Yong Kim
>
> **备注:** 31 pages
>
> **摘要:** Mammography, an X-ray-based imaging technique, remains central to the early detection of breast cancer. Recent advances in artificial intelligence have enabled increasingly sophisticated computer-aided diagnostic methods, evolving from patch-based classifiers to whole-image approaches and then to multi-view architectures that jointly analyze complementary projections. Despite this progress, several critical questions remain unanswered. In this study, we systematically investigate these issues by addressing five key research questions: (1) the role of patch classifiers in performance, (2) the transferability of natural-image-trained backbones, (3) the advantages of learn-to-resize over conventional downscaling, (4) the contribution of multi-view integration, and (5) the robustness of findings across varying image quality. Beyond benchmarking, our experiments demonstrate clear performance gains over prior work. For the CBIS-DDSM dataset, we improved single-view AUC from 0.8153 to 0.8343, and multiple-view AUC from 0.8483 to 0.8658. Using a new comparative method, we also observed a 0.0217 AUC increase when extending from single to multiple-view analysis. On the complete VinDr-Mammo dataset, the multiple-view approach further improved results, achieving a 0.0492 AUC increase over single view and reaching 0.8511 AUC overall. These results establish new state-of-the-art benchmarks, providing clear evidence of the advantages of multi-view architectures for mammogram interpretation. Beyond performance, our analysis offers principled insights into model design and transfer learning strategies, contributing to the development of more accurate and reliable breast cancer screening tools. The inference code and trained models are publicly available at https://github.com/dpetrini/multiple-view.
>
---
#### [replaced 072] Empowering LLMs with Pseudo-Untrimmed Videos for Audio-Visual Temporal Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.16276v3](http://arxiv.org/pdf/2403.16276v3)**

> **作者:** Yolo Yunlong Tang; Daiki Shimada; Jing Bi; Mingqian Feng; Hang Hua; Chenliang Xu
>
> **备注:** Accepted to AAAI 2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in natural language and multimodal domains. By fine-tuning multimodal LLMs with temporal annotations from well-annotated datasets, e.g., dense video captioning datasets, their temporal understanding capacity in video-language tasks can be obtained. However, there is a notable lack of untrimmed audio-visual video datasets with precise temporal annotations for events. This deficiency hinders LLMs from learning the alignment between time, audio-visual events, and text tokens, thus impairing their ability to temporally localize audio-visual events in videos. To address this gap, we introduce PU-VALOR, a comprehensive audio-visual dataset comprising over 114,000 pseudo-untrimmed videos with detailed temporal annotations. PU-VALOR is derived from the large-scale but coarse-annotated audio-visual dataset VALOR, through a subtle method involving event-based video clustering, random temporal scaling, and permutation. By fine-tuning a multimodal LLM on PU-VALOR, we developed AVicuna, a model capable of aligning audio-visual events with temporal intervals and corresponding text tokens. AVicuna excels in temporal localization and time-aware dialogue capabilities. Our experiments demonstrate that AVicuna effectively handles temporal understanding in audio-visual videos and achieves state-of-the-art performance on open-ended video QA, audio-visual QA, and audio-visual event dense localization tasks.
>
---
