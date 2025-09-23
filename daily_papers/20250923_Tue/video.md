# 计算机视觉 cs.CV

- **最新发布 250 篇**

- **更新 138 篇**

## 最新发布

#### [new 001] Benchmarking and Mitigating MCQA Selection Bias of Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文聚焦视觉语言模型在多选题任务中的选择偏差问题，构建了不同难度的MCQA基准，并提出了一种无需再训练的推理时去偏方法，有效减少了偏差并提升了模型在复杂任务中的准确性。**

- **链接: [http://arxiv.org/pdf/2509.16805v1](http://arxiv.org/pdf/2509.16805v1)**

> **作者:** Md. Atabuzzaman; Ali Asgarov; Chris Thomas
>
> **备注:** Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved strong performance on vision-language tasks, particularly Visual Question Answering (VQA). While prior work has explored unimodal biases in VQA, the problem of selection bias in Multiple-Choice Question Answering (MCQA), where models may favor specific option tokens (e.g., "A") or positions, remains underexplored. In this paper, we investigate both the presence and nature of selection bias in LVLMs through fine-grained MCQA benchmarks spanning easy, medium, and hard difficulty levels, defined by the semantic similarity of the options. We further propose an inference-time logit-level debiasing method that estimates an ensemble bias vector from general and contextual prompts and applies confidence-adaptive corrections to the model's output. Our method mitigates bias without retraining and is compatible with frozen LVLMs. Extensive experiments across several state-of-the-art models reveal consistent selection biases that intensify with task difficulty, and show that our mitigation approach significantly reduces bias while improving accuracy in challenging settings. This work offers new insights into the limitations of LVLMs in MCQA and presents a practical approach to improve their robustness in fine-grained visual reasoning. Datasets and code are available at: https://github.com/Atabuzzaman/Selection-Bias-of-LVLMs
>
---
#### [new 002] FitPro: A Zero-Shot Framework for Interactive Text-based Pedestrian Retrieval in Open World
- **分类: cs.CV**

- **简介: 该论文提出FitPro，一种面向开放世界场景的零样本交互式文本检索行人框架。针对现有方法在语义理解和跨场景适应性上的不足，设计了特征对比解码、增量语义挖掘和查询感知分层检索三个模块，显著提升了模型泛化与交互性能。**

- **链接: [http://arxiv.org/pdf/2509.16674v1](http://arxiv.org/pdf/2509.16674v1)**

> **作者:** Zengli Luo; Canlong Zhang; Xiaochun Lu; Zhixin Li
>
> **备注:** 15pages,6 figures
>
> **摘要:** Text-based Pedestrian Retrieval (TPR) aims to retrieve specific target pedestrians in visual scenes according to natural language descriptions. Although existing methods have achieved progress under constrained settings, interactive retrieval in the open-world scenario still suffers from limited model generalization and insufficient semantic understanding. To address these challenges, we propose FitPro, an open-world interactive zero-shot TPR framework with enhanced semantic comprehension and cross-scene adaptability. FitPro has three innovative components: Feature Contrastive Decoding (FCD), Incremental Semantic Mining (ISM), and Query-aware Hierarchical Retrieval (QHR). The FCD integrates prompt-guided contrastive decoding to generate high-quality structured pedestrian descriptions from denoised images, effectively alleviating semantic drift in zero-shot scenarios. The ISM constructs holistic pedestrian representations from multi-view observations to achieve global semantic modeling in multi-turn interactions,thereby improving robustness against viewpoint shifts and fine-grained variations in descriptions. The QHR dynamically optimizes the retrieval pipeline according to query types, enabling efficient adaptation to multi-modal and multi-view inputs. Extensive experiments on five public datasets and two evaluation protocols demonstrate that FitPro significantly overcomes the generalization limitations and semantic modeling constraints of existing methods in interactive retrieval, paving the way for practical deployment. The code and data will be released at https://github.com/ lilo4096/FitPro-Interactive-Person-Retrieval.
>
---
#### [new 003] Segment-to-Act: Label-Noise-Robust Action-Prompted Video Segmentation Towards Embodied Intelligence
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文研究动作提示视频分割任务，旨在解决标签噪声问题。工作包括引入两类噪声、构建首个带噪声基准ActiSeg-NL、适配六种噪声学习策略，并提出PMHM机制提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16677v1](http://arxiv.org/pdf/2509.16677v1)**

> **作者:** Wenxin Li; Kunyu Peng; Di Wen; Ruiping Liu; Mengfei Duan; Kai Luo; Kailun Yang
>
> **备注:** The established benchmark and source code will be made publicly available at https://github.com/mylwx/ActiSeg-NL
>
> **摘要:** Embodied intelligence relies on accurately segmenting objects actively involved in interactions. Action-based video object segmentation addresses this by linking segmentation with action semantics, but it depends on large-scale annotations and prompts that are costly, inconsistent, and prone to multimodal noise such as imprecise masks and referential ambiguity. To date, this challenge remains unexplored. In this work, we take the first step by studying action-based video object segmentation under label noise, focusing on two sources: textual prompt noise (category flips and within-category noun substitutions) and mask annotation noise (perturbed object boundaries to mimic imprecise supervision). Our contributions are threefold. First, we introduce two types of label noises for the action-based video object segmentation task. Second, we build up the first action-based video object segmentation under a label noise benchmark ActiSeg-NL and adapt six label-noise learning strategies to this setting, and establish protocols for evaluating them under textual, boundary, and mixed noise. Third, we provide a comprehensive analysis linking noise types to failure modes and robustness gains, and we introduce a Parallel Mask Head Mechanism (PMHM) to address mask annotation noise. Qualitative evaluations further reveal characteristic failure modes, including boundary leakage and mislocalization under boundary perturbations, as well as occasional identity substitutions under textual flips. Our comparative analysis reveals that different learning strategies exhibit distinct robustness profiles, governed by a foreground-background trade-off where some achieve balanced performance while others prioritize foreground accuracy at the cost of background precision. The established benchmark and source code will be made publicly available at https://github.com/mylwx/ActiSeg-NL.
>
---
#### [new 004] 3D Gaussian Flats: Hybrid 2D/3D Photometric Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出一种混合2D/3D高斯表示方法，用于光度场景重建。针对现有方法在处理无纹理平面时的不足，结合平面和自由形式高斯优化，提升视觉质量和几何精度，在室内场景重建任务中表现出色。**

- **链接: [http://arxiv.org/pdf/2509.16423v1](http://arxiv.org/pdf/2509.16423v1)**

> **作者:** Maria Taktasheva; Lily Goli; Alessandro Fiorini; Zhen; Li; Daniel Rebain; Andrea Tagliasacchi
>
> **摘要:** Recent advances in radiance fields and novel view synthesis enable creation of realistic digital twins from photographs. However, current methods struggle with flat, texture-less surfaces, creating uneven and semi-transparent reconstructions, due to an ill-conditioned photometric reconstruction objective. Surface reconstruction methods solve this issue but sacrifice visual quality. We propose a novel hybrid 2D/3D representation that jointly optimizes constrained planar (2D) Gaussians for modeling flat surfaces and freeform (3D) Gaussians for the rest of the scene. Our end-to-end approach dynamically detects and refines planar regions, improving both visual fidelity and geometric accuracy. It achieves state-of-the-art depth estimation on ScanNet++ and ScanNetv2, and excels at mesh extraction without overfitting to a specific camera model, showing its effectiveness in producing high-quality reconstruction of indoor scenes.
>
---
#### [new 005] Unified Multimodal Coherent Field: Synchronous Semantic-Spatial-Vision Fusion for Brain Tumor Segmentation
- **分类: cs.CV**

- **简介: 该论文针对脑肿瘤分割任务，旨在解决多模态MRI图像中肿瘤区域边界模糊、层次结构复杂的问题。提出UMCF方法，在统一3D潜在空间中同步融合视觉、语义和空间信息，并引入医学先验知识，提升分割精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.17520v1](http://arxiv.org/pdf/2509.17520v1)**

> **作者:** Mingda Zhang; Yuyang Zheng; Ruixiang Tang; Jingru Qiu; Haiyan Ding
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Brain tumor segmentation requires accurate identification of hierarchical regions including whole tumor (WT), tumor core (TC), and enhancing tumor (ET) from multi-sequence magnetic resonance imaging (MRI) images. Due to tumor tissue heterogeneity, ambiguous boundaries, and contrast variations across MRI sequences, methods relying solely on visual information or post-hoc loss constraints show unstable performance in boundary delineation and hierarchy preservation. To address this challenge, we propose the Unified Multimodal Coherent Field (UMCF) method. This method achieves synchronous interactive fusion of visual, semantic, and spatial information within a unified 3D latent space, adaptively adjusting modal contributions through parameter-free uncertainty gating, with medical prior knowledge directly participating in attention computation, avoiding the traditional "process-then-concatenate" separated architecture. On Brain Tumor Segmentation (BraTS) 2020 and 2021 datasets, UMCF+nnU-Net achieves average Dice coefficients of 0.8579 and 0.8977 respectively, with an average 4.18% improvement across mainstream architectures. By deeply integrating clinical knowledge with imaging features, UMCF provides a new technical pathway for multimodal information fusion in precision medicine.
>
---
#### [new 006] Catching the Details: Self-Distilled RoI Predictors for Fine-Grained MLLM Perception
- **分类: cs.CV**

- **简介: 该论文针对多模态大语言模型（MLLM）在细粒度视觉感知任务中面临的问题，提出了一种无需标注的SD-RPN方法。通过利用模型中间层注意力生成伪RoI标签，训练轻量级RPN网络，高效精准定位关键区域，提升了TextVQA等任务的性能。**

- **链接: [http://arxiv.org/pdf/2509.16944v1](http://arxiv.org/pdf/2509.16944v1)**

> **作者:** Yuheng Shi; Xiaohuan Pei; Minjing Dong; Chang Xu
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) require high-resolution visual information to perform fine-grained perception, yet processing entire high-resolution images is computationally prohibitive. While recent methods leverage a Region-of-Interest (RoI) mechanism to focus on salient areas, they typically present a difficult trade-off: training-based approaches depend on large-scale annotated datasets, while training-free methods that utilize the model's internal attention are computationally inefficient and less accurate, requiring either multi-pass prefill stages or reliance on the slow auto-regressive decoding process. In this paper, we propose an efficient, annotation-free Self-Distilled Region Proposal Network (SD-RPN) that resolves this trade-off. The SD-RPN is built around a pipeline that transforms the noisy attention maps from the MLLM's middle layers into high-quality pseudo-RoI labels by explicitly denoising the signal and resolving ambiguity. We use these labels to train a lightweight Region Proposal Network (RPN) that learns a more precise localization. This RPN is also highly efficient, predicting the RoI in a single forward pass using features from the MLLM's middle layers, decoupling RoI identification from the auto-regressive generation and avoiding costly multi-pass operations.To validate our approach, we integrate the framework into the LLaVA-1.5 architecture. Despite being trained on only a few (e.g. 10K) question-answer pairs, our method demonstrates exceptional data efficiency and generalization, achieving over a 10% absolute accuracy improvement on unseen benchmarks, including TextVQA, DocVQA, and V-Star. Our work presents a practical and scalable solution for enhancing the fine-grained perception of MLLMs without requiring costly supervision or full model fine-tuning. Code is available at https://github.com/YuHengsss/SD-RPN.
>
---
#### [new 007] Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration
- **分类: cs.CV; 68T45; I.2.10**

- **简介: 该论文提出多尺度时间预测（MSTP）任务，旨在解决视觉-语言模型在多尺度场景状态预测中的困难。针对通用和手术场景，构建首个MSTP基准，并提出增量生成与多智能体协作方法（IG-MC），通过模块化设计提升预测精度与一致性。**

- **链接: [http://arxiv.org/pdf/2509.17429v1](http://arxiv.org/pdf/2509.17429v1)**

> **作者:** Zhitao Zeng; Guojian Yuan; Junyuan Mao; Yuxuan Wang; Xiaoshuang Jia; Yueming Jin
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Accurate temporal prediction is the bridge between comprehensive scene understanding and embodied artificial intelligence. However, predicting multiple fine-grained states of a scene at multiple temporal scales is difficult for vision-language models. We formalize the Multi-Scale Temporal Prediction (MSTP) task in general and surgical scenes by decomposing multi-scale into two orthogonal dimensions: the temporal scale, forecasting states of humans and surgery at varying look-ahead intervals, and the state scale, modeling a hierarchy of states in general and surgical scenes. For example, in general scenes, states of contact relationships are finer-grained than states of spatial relationships. In surgical scenes, medium-level steps are finer-grained than high-level phases yet remain constrained by their encompassing phase. To support this unified task, we introduce the first MSTP Benchmark, featuring synchronized annotations across multiple state scales and temporal scales. We further propose a method, Incremental Generation and Multi-agent Collaboration (IG-MC), which integrates two key innovations. First, we present a plug-and-play incremental generation module that continuously synthesizes up-to-date visual previews at expanding temporal scales to inform multiple decision-making agents, keeping decisions and generated visuals synchronized and preventing performance degradation as look-ahead intervals lengthen. Second, we present a decision-driven multi-agent collaboration framework for multi-state prediction, comprising generation, initiation, and multi-state assessment agents that dynamically trigger and evaluate prediction cycles to balance global coherence and local fidelity.
>
---
#### [new 008] StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出StableGuard，针对扩散模型生成内容的版权保护与篡改定位问题。通过将二值水印嵌入生成过程，设计MPW-VAE和MoE-GFN模块，实现端到端的统一解决方案，提升水印验证与篡改检测效果。**

- **链接: [http://arxiv.org/pdf/2509.17993v1](http://arxiv.org/pdf/2509.17993v1)**

> **作者:** Haoxin Yang; Bangzhen Liu; Xuemiao Xu; Cheng Xu; Yuyang Yu; Zikai Huang; Yi Wang; Shengfeng He
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** The advancement of diffusion models has enhanced the realism of AI-generated content but also raised concerns about misuse, necessitating robust copyright protection and tampering localization. Although recent methods have made progress toward unified solutions, their reliance on post hoc processing introduces considerable application inconvenience and compromises forensic reliability. We propose StableGuard, a novel framework that seamlessly integrates a binary watermark into the diffusion generation process, ensuring copyright protection and tampering localization in Latent Diffusion Models through an end-to-end design. We develop a Multiplexing Watermark VAE (MPW-VAE) by equipping a pretrained Variational Autoencoder (VAE) with a lightweight latent residual-based adapter, enabling the generation of paired watermarked and watermark-free images. These pairs, fused via random masks, create a diverse dataset for training a tampering-agnostic forensic network. To further enhance forensic synergy, we introduce a Mixture-of-Experts Guided Forensic Network (MoE-GFN) that dynamically integrates holistic watermark patterns, local tampering traces, and frequency-domain cues for precise watermark verification and tampered region detection. The MPW-VAE and MoE-GFN are jointly optimized in a self-supervised, end-to-end manner, fostering a reciprocal training between watermark embedding and forensic accuracy. Extensive experiments demonstrate that StableGuard consistently outperforms state-of-the-art methods in image fidelity, watermark verification, and tampering localization.
>
---
#### [new 009] SynergyNet: Fusing Generative Priors and State-Space Models for Facial Beauty Prediction
- **分类: cs.CV**

- **简介: 该论文针对面部美感预测任务，旨在解决局部细节与全局结构建模的矛盾。提出MD-Net双流架构，融合预训练扩散模型的生成先验和Vision Mamba的高效全局建模能力，通过跨注意力机制实现性能突破，在SCUT-FBP5500上达到0.9235的Pearson相关系数。**

- **链接: [http://arxiv.org/pdf/2509.17172v1](http://arxiv.org/pdf/2509.17172v1)**

> **作者:** Djamel Eddine Boukhari
>
> **摘要:** The automated prediction of facial beauty is a benchmark task in affective computing that requires a sophisticated understanding of both local aesthetic details (e.g., skin texture) and global facial harmony (e.g., symmetry, proportions). Existing models, based on either Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs), exhibit inherent architectural biases that limit their performance; CNNs excel at local feature extraction but struggle with long-range dependencies, while ViTs model global relationships at a significant computational cost. This paper introduces the \textbf{Mamba-Diffusion Network (MD-Net)}, a novel dual-stream architecture that resolves this trade-off by delegating specialized roles to state-of-the-art models. The first stream leverages a frozen U-Net encoder from a pre-trained latent diffusion model, providing a powerful generative prior for fine-grained aesthetic qualities. The second stream employs a Vision Mamba (Vim), a modern state-space model, to efficiently capture global facial structure with linear-time complexity. By synergistically integrating these complementary representations through a cross-attention mechanism, MD-Net creates a holistic and nuanced feature space for prediction. Evaluated on the SCUT-FBP5500 benchmark, MD-Net sets a new state-of-the-art, achieving a Pearson Correlation of \textbf{0.9235} and demonstrating the significant potential of hybrid architectures that fuse generative and sequential modeling paradigms for complex visual assessment tasks.
>
---
#### [new 010] From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MIR基准，旨在提升多模态大模型在多图交织推理任务中的能力。针对现有基准忽略图文交织关系的问题，构建包含交织文本的多图推理数据集，并采用“由易到难”的分阶段训练策略，增强模型跨模态理解与推理能力。**

- **链接: [http://arxiv.org/pdf/2509.17040v1](http://arxiv.org/pdf/2509.17040v1)**

> **作者:** Hang Du; Jiayang Zhang; Guoshun Nan; Wendi Deng; Zhenyan Chen; Chenyang Zhang; Wang Xiao; Shan Huang; Yuqi Pan; Tao Qi; Sicong Leng
>
> **摘要:** Multi-image Interleaved Reasoning aims to improve Multi-modal Large Language Models (MLLMs) ability to jointly comprehend and reason across multiple images and their associated textual contexts, introducing unique challenges beyond single-image or non-interleaved multi-image tasks. While current multi-image benchmarks overlook interleaved textual contexts and neglect distinct relationships between individual images and their associated texts, enabling models to reason over multi-image interleaved data may significantly enhance their comprehension of complex scenes and better capture cross-modal correlations. To bridge this gap, we introduce a novel benchmark MIR, requiring joint reasoning over multiple images accompanied by interleaved textual contexts to accurately associate image regions with corresponding texts and logically connect information across images. To enhance MLLMs ability to comprehend multi-image interleaved data, we introduce reasoning steps for each instance within the benchmark and propose a stage-wise curriculum learning strategy. This strategy follows an "easy to hard" approach, progressively guiding models from simple to complex scenarios, thereby enhancing their ability to handle challenging tasks. Extensive experiments benchmarking multiple MLLMs demonstrate that our method significantly enhances models reasoning performance on MIR and other established benchmarks. We believe that MIR will encourage further research into multi-image interleaved reasoning, facilitating advancements in MLLMs capability to handle complex inter-modal tasks.Our code and dataset are available at https://github.com/Shelly-coder239/MIRBench.
>
---
#### [new 011] SAEC: Scene-Aware Enhanced Edge-Cloud Collaborative Industrial Vision Inspection with Multimodal LLM
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SAEC，一种基于多模态大模型的工业视觉检测框架，旨在解决高精度与资源约束的矛盾。通过轻量化场景复杂度估计和自适应边云调度，提升检测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.17136v1](http://arxiv.org/pdf/2509.17136v1)**

> **作者:** Yuhao Tian; Zheming Yang
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Industrial vision inspection requires high accuracy under stringent resource constraints, yet existing approaches face a fundamental trade-off. Multimodal LLMs (MLLMs) deliver strong reasoning capabilities but incur prohibitive computational costs, while lightweight edge models often fail on complex cases. In this paper, we present SAEC, a scene-aware enhanced edge-cloud collaborative industrial vision inspection framework with MLLM. The framework is composed of three synergistic components: (1) Efficient MLLM Fine-Tuning for Complex Defect Inspection, (2) Lightweight Multiscale Scene-Complexity Estimation, and (3) Adaptive Edge-Cloud Scheduler. Together, these modules enable robust defect detection by tailoring multimodal reasoning to scene complexity and dynamically balancing computation between edge and cloud resources. Experimental results on MVTec AD and KSDD2 datasets demonstrate that SAEC attains 85.11% and 82.72% accuracy, surpassing Qwen by 22.1% and 20.8%, and LLaVA by 33.3% and 31.6%. It also reduces runtime by up to 22.4% and cuts energy per correct decision by 40%-74%. The code is available at https://github.com/YuHao-Tian/SAEC.
>
---
#### [new 012] Predicting Depth Maps from Single RGB Images and Addressing Missing Information in Depth Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度估计任务，旨在解决从单张RGB图像生成深度图以及修复深度图中缺失信息的问题。研究提出了一种多层训练算法，成功在Cityscapes数据集上实现了深度图的生成与修复。**

- **链接: [http://arxiv.org/pdf/2509.17686v1](http://arxiv.org/pdf/2509.17686v1)**

> **作者:** Mohamad Mofeed Chaar; Jamal Raiyn; Galia Weidl
>
> **备注:** 8 pages, 10 figures, VEHITS conference 2025
>
> **摘要:** Depth imaging is a crucial area in Autonomous Driving Systems (ADS), as it plays a key role in detecting and measuring objects in the vehicle's surroundings. However, a significant challenge in this domain arises from missing information in Depth images, where certain points are not measurable due to gaps or inconsistencies in pixel data. Our research addresses two key tasks to overcome this challenge. First, we developed an algorithm using a multi-layered training approach to generate Depth images from a single RGB image. Second, we addressed the issue of missing information in Depth images by applying our algorithm to rectify these gaps, resulting in Depth images with complete and accurate data. We further tested our algorithm on the Cityscapes dataset and successfully resolved the missing information in its Depth images, demonstrating the effectiveness of our approach in real-world urban environments.
>
---
#### [new 013] Overview of PlantCLEF 2023: Image-based Plant Identification at Global Scale
- **分类: cs.CV**

- **简介: 该论文介绍了PlantCLEF2023挑战，旨在通过多图像与元数据分类解决全球8万种植物的自动识别问题。面对类别众多、分布不均等挑战，论文概述了资源、评估方法及研究团队采用的深度学习技术，推动植物物种高效识别系统的构建。**

- **链接: [http://arxiv.org/pdf/2509.17622v1](http://arxiv.org/pdf/2509.17622v1)**

> **作者:** Herve Goeau; Pierre Bonnet; Alexis Joly
>
> **备注:** 10 pages, 1 figure, CLEF 2023 Conference and Labs of the Evaluation Forum, September 18 to 21, 2023, Thessaloniki, Greece
>
> **摘要:** The world is estimated to be home to over 300,000 species of vascular plants. In the face of the ongoing biodiversity crisis, expanding our understanding of these species is crucial for the advancement of human civilization, encompassing areas such as agriculture, construction, and pharmacopoeia. However, the labor-intensive process of plant identification undertaken by human experts poses a significant obstacle to the accumulation of new data and knowledge. Fortunately, recent advancements in automatic identification, particularly through the application of deep learning techniques, have shown promising progress. Despite challenges posed by data-related issues such as a vast number of classes, imbalanced class distribution, erroneous identifications, duplications, variable visual quality, and diverse visual contents (such as photos or herbarium sheets), deep learning approaches have reached a level of maturity which gives us hope that in the near future we will have an identification system capable of accurately identifying all plant species worldwide. The PlantCLEF2023 challenge aims to contribute to this pursuit by addressing a multi-image (and metadata) classification problem involving an extensive set of classes (80,000 plant species). This paper provides an overview of the challenge's resources and evaluations, summarizes the methods and systems employed by participating research groups, and presents an analysis of key findings.
>
---
#### [new 014] MO R-CNN: Multispectral Oriented R-CNN for Object Detection in Remote Sensing Image
- **分类: cs.CV**

- **简介: 该论文提出MO R-CNN，用于遥感图像中的多光谱定向目标检测。针对现有方法计算复杂、内存消耗大的问题，设计了轻量级框架，包含异构特征提取网络（HFEN）、单模态监督（SMS）和条件多模态标签融合（CMLF），提升检测性能并降低资源消耗。**

- **链接: [http://arxiv.org/pdf/2509.16957v1](http://arxiv.org/pdf/2509.16957v1)**

> **作者:** Leiyu Wang; Biao Jin; Feng Huang; Liqiong Chen; Zhengyong Wang; Xiaohai He; Honggang Chen
>
> **摘要:** Oriented object detection for multi-spectral imagery faces significant challenges due to differences both within and between modalities. Although existing methods have improved detection accuracy through complex network architectures, their high computational complexity and memory consumption severely restrict their performance. Motivated by the success of large kernel convolutions in remote sensing, we propose MO R-CNN, a lightweight framework for multi-spectral oriented detection featuring heterogeneous feature extraction network (HFEN), single modality supervision (SMS), and condition-based multimodal label fusion (CMLF). HFEN leverages inter-modal differences to adaptively align, merge, and enhance multi-modal features. SMS constrains multi-scale features and enables the model to learn from multiple modalities. CMLF fuses multimodal labels based on specific rules, providing the model with a more robust and consistent supervisory signal. Experiments on the DroneVehicle, VEDAI and OGSOD datasets prove the superiority of our method. The source code is available at:https://github.com/Iwill-github/MORCNN.
>
---
#### [new 015] Captioning for Text-Video Retrieval via Dual-Group Direct Preference Optimization
- **分类: cs.CV**

- **简介: 该论文针对文本-视频检索任务，旨在解决传统生成的辅助字幕过于通用、难以区分视觉相似视频的问题。提出CaRe-DPO框架，采用双组直接偏好优化方法，通过检索相关性分数优化字幕生成，提升细粒度检索性能。**

- **链接: [http://arxiv.org/pdf/2509.16560v1](http://arxiv.org/pdf/2509.16560v1)**

> **作者:** Ji Soo Lee; Byungoh Ko; Jaewon Cho; Howoong Lee; Jaewoon Byun; Hyunwoo J. Kim
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** In text-video retrieval, auxiliary captions are often used to enhance video understanding, bridging the gap between the modalities. While recent advances in multi-modal large language models (MLLMs) have enabled strong zero-shot caption generation, we observe that such captions tend to be generic and indistinguishable across visually similar videos, limiting their utility for fine-grained retrieval. Moreover, conventional captioning approaches are typically evaluated using language generation metrics, such as BLEU, which are not typically tailored for retrieval tasks that require making discriminative distinctions between candidates. To address this, we propose $\textbf{CaRe-DPO}$, a retrieval framework that directly optimizes caption generation using retrieval relevance scores. At its core is Dual-Group Direct Preference Optimization (DG-DPO), a novel learning strategy that supervises captioning by modeling preferences across groups of distinct video and caption pairs. In addition, we present an MLLM-based retrieval model that incorporates role-embeddings to better distinguish between textual inputs with different functional roles, such as an auxiliary caption and a text query. Through extensive experiments, we demonstrate that CaRe-DPO significantly enhances retrieval performance by effectively leveraging auxiliary knowledge to generate fine-grained captions for retrieval. Code is available at https://github.com/mlvlab/CaReDPO.
>
---
#### [new 016] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Lattice Boltzmann Model（LBM）用于视觉跟踪任务，旨在解决真实场景中像素动态性建模问题。通过预测-更新网络分解像素动态，并利用碰撞-流过程估计像素位置与可见性，实现实时高效的在线视觉跟踪。**

- **链接: [http://arxiv.org/pdf/2509.16527v1](http://arxiv.org/pdf/2509.16527v1)**

> **作者:** Guangze Zheng; Shijie Lin; Haobo Zuo; Si Si; Ming-Shan Wang; Changhong Fu; Jia Pan
>
> **备注:** NeurIPS 2025. Project page: https://george-zhuang.github.io/lbm/
>
> **摘要:** This work proposes the Lattice Boltzmann Model (LBM) to learn real-world pixel dynamicity for visual tracking. LBM decomposes visual representations into dynamic pixel lattices and solves pixel motion states through collision-streaming processes. Specifically, the high-dimensional distribution of the target pixels is acquired through a multilayer predict-update network to estimate the pixel positions and visibility. The predict stage formulates lattice collisions among the spatial neighborhood of target pixels and develops lattice streaming within the temporal visual context. The update stage rectifies the pixel distributions with online visual representations. Compared with existing methods, LBM demonstrates practical applicability in an online and real-time manner, which can efficiently adapt to real-world visual tracking tasks. Comprehensive evaluations of real-world point tracking benchmarks such as TAP-Vid and RoboTAP validate LBM's efficiency. A general evaluation of large-scale open-world object tracking benchmarks such as TAO, BFT, and OVT-B further demonstrates LBM's real-world practicality.
>
---
#### [new 017] Detection and Simulation of Urban Heat Islands Using a Fine-Tuned Geospatial Foundation Model
- **分类: cs.CV; cs.AI; I.2.6; I.5.4; I.6.8**

- **简介: 该论文属于城市热岛检测与模拟任务，旨在解决传统模型预测精度低的问题。研究通过微调地理空间基础模型，预测未来城市地表温度，并评估植被策略的影响，实现了高精度的温度下放和模式匹配。**

- **链接: [http://arxiv.org/pdf/2509.16617v1](http://arxiv.org/pdf/2509.16617v1)**

> **作者:** David Kreismann
>
> **备注:** 12 pages, 4 figures, to appear in GI LNI (SKILL 2025)
>
> **摘要:** As urbanization and climate change progress, urban heat island effects are becoming more frequent and severe. To formulate effective mitigation plans, cities require detailed air temperature data. However, predictive analytics methods based on conventional machine learning models and limited data infrastructure often provide inaccurate predictions, especially in underserved areas. In this context, geospatial foundation models trained on unstructured global data demonstrate strong generalization and require minimal fine-tuning, offering an alternative for predictions where traditional approaches are limited. This study fine-tunes a geospatial foundation model to predict urban land surface temperatures under future climate scenarios and explores its response to land cover changes using simulated vegetation strategies. The fine-tuned model achieved pixel-wise downscaling errors below 1.74 {\deg}C and aligned with ground truth patterns, demonstrating an extrapolation capacity up to 3.62 {\deg}C.
>
---
#### [new 018] Evict3R: Training-Free Token Eviction for Memory-Bounded Streaming Visual Geometry Transformers
- **分类: cs.CV**

- **简介: 该论文针对流式视觉几何变换器（如StreamVGGT）在处理3D感知任务时KV缓存无界增长的问题，提出了一种无需训练的推理时令牌驱逐策略Evict3R。通过丢弃冗余令牌以降低内存占用，在保证精度的前提下显著减少内存消耗，使长序列推理更高效。**

- **链接: [http://arxiv.org/pdf/2509.17650v1](http://arxiv.org/pdf/2509.17650v1)**

> **作者:** Soroush Mahdi; Fardin Ayar; Ehsan Javanmardi; Manabu Tsukada; Mahdi Javanmardi
>
> **摘要:** Streaming visual transformers like StreamVGGT achieve strong 3D perception but suffer from unbounded growth of key value (KV) memory, which limits scalability. We propose a training-free, inference-time token eviction policy that bounds memory by discarding redundant tokens while keeping the most informative ones. Our method uses significantly less memory with little to no drop in accuracy: on 7-Scenes with long sequences it reduces peak memory from 18.63 GB to 9.39 GB while accuracy and completeness drop by only 0.003. Under strict memory budgets, eviction enables denser frame sampling, which improves reconstruction accuracy compared to the baseline. Experiments across video depth estimation (Sintel, KITTI), 3D reconstruction (7-Scenes, NRGBD), and camera pose estimation (Sintel, TUM-dynamics) show that our approach closely matches StreamVGGT at a fraction of the memory and makes long-horizon streaming inference more practical.
>
---
#### [new 019] Describe-to-Score: Text-Guided Efficient Image Complexity Assessment
- **分类: cs.CV**

- **简介: 该论文提出D2S框架，用于图像复杂度评估任务。针对现有方法依赖视觉特征、忽视语义信息的问题，引入图文融合策略，通过生成图像描述并进行特征对齐，提升评估的准确性和泛化性，同时保持推理效率。**

- **链接: [http://arxiv.org/pdf/2509.16609v1](http://arxiv.org/pdf/2509.16609v1)**

> **作者:** Shipeng Liu; Zhonglin Zhang; Dengfeng Chen; Liang Zhao
>
> **摘要:** Accurately assessing image complexity (IC) is critical for computer vision, yet most existing methods rely solely on visual features and often neglect high-level semantic information, limiting their accuracy and generalization. We introduce vision-text fusion for IC modeling. This approach integrates visual and textual semantic features, increasing representational diversity. It also reduces the complexity of the hypothesis space, which enhances both accuracy and generalization in complexity assessment. We propose the D2S (Describe-to-Score) framework, which generates image captions with a pre-trained vision-language model. We propose the feature alignment and entropy distribution alignment mechanisms, D2S guides semantic information to inform complexity assessment while bridging the gap between vision and text modalities. D2S utilizes multi-modal information during training but requires only the vision branch during inference, thereby avoiding multi-modal computational overhead and enabling efficient assessment. Experimental results demonstrate that D2S outperforms existing methods on the IC9600 dataset and maintains competitiveness on no-reference image quality assessment (NR-IQA) benchmark, validating the effectiveness and efficiency of multi-modal fusion in complexity-related tasks. Code is available at: https://github.com/xauat-liushipeng/D2S
>
---
#### [new 020] ContextFlow: Training-Free Video Object Editing via Adaptive Context Enrichment
- **分类: cs.CV**

- **简介: 该论文提出ContextFlow，一种无需训练的视频对象编辑框架，旨在解决编辑过程中的保真度与时间一致性问题。通过引入高阶Rectified Flow求解器和自适应上下文增强机制，有效缓解了特征替换导致的上下文冲突，并通过数据驱动方法定位关键层，提升了编辑效果。**

- **链接: [http://arxiv.org/pdf/2509.17818v1](http://arxiv.org/pdf/2509.17818v1)**

> **作者:** Yiyang Chen; Xuanhua He; Xiujun Ma; Yue Ma
>
> **备注:** The project page is at https://yychen233.github.io/ContextFlow-page
>
> **摘要:** Training-free video object editing aims to achieve precise object-level manipulation, including object insertion, swapping, and deletion. However, it faces significant challenges in maintaining fidelity and temporal consistency. Existing methods, often designed for U-Net architectures, suffer from two primary limitations: inaccurate inversion due to first-order solvers, and contextual conflicts caused by crude "hard" feature replacement. These issues are more challenging in Diffusion Transformers (DiTs), where the unsuitability of prior layer-selection heuristics makes effective guidance challenging. To address these limitations, we introduce ContextFlow, a novel training-free framework for DiT-based video object editing. In detail, we first employ a high-order Rectified Flow solver to establish a robust editing foundation. The core of our framework is Adaptive Context Enrichment (for specifying what to edit), a mechanism that addresses contextual conflicts. Instead of replacing features, it enriches the self-attention context by concatenating Key-Value pairs from parallel reconstruction and editing paths, empowering the model to dynamically fuse information. Additionally, to determine where to apply this enrichment (for specifying where to edit), we propose a systematic, data-driven analysis to identify task-specific vital layers. Based on a novel Guidance Responsiveness Metric, our method pinpoints the most influential DiT blocks for different tasks (e.g., insertion, swapping), enabling targeted and highly effective guidance. Extensive experiments show that ContextFlow significantly outperforms existing training-free methods and even surpasses several state-of-the-art training-based approaches, delivering temporally coherent, high-fidelity results.
>
---
#### [new 021] Spectral Compressive Imaging via Chromaticity-Intensity Decomposition
- **分类: cs.CV**

- **简介: 该论文针对压缩光谱成像中的重建问题，提出一种基于色度-强度分解的框架CIDNet。通过分离光照不变的色度和空间平滑强度，结合双相机CASSI系统与混合Transformer，提升了高光谱图像的重建质量。**

- **链接: [http://arxiv.org/pdf/2509.16690v1](http://arxiv.org/pdf/2509.16690v1)**

> **作者:** Xiaodong Wang; Zijun He; Ping Wang; Lishun Wang; Yanan Hu; Xin Yuan
>
> **摘要:** In coded aperture snapshot spectral imaging (CASSI), the captured measurement entangles spatial and spectral information, posing a severely ill-posed inverse problem for hyperspectral images (HSIs) reconstruction. Moreover, the captured radiance inherently depends on scene illumination, making it difficult to recover the intrinsic spectral reflectance that remains invariant to lighting conditions. To address these challenges, we propose a chromaticity-intensity decomposition framework, which disentangles an HSI into a spatially smooth intensity map and a spectrally variant chromaticity cube. The chromaticity encodes lighting-invariant reflectance, enriched with high-frequency spatial details and local spectral sparsity. Building on this decomposition, we develop CIDNet, a Chromaticity-Intensity Decomposition unfolding network within a dual-camera CASSI system. CIDNet integrates a hybrid spatial-spectral Transformer tailored to reconstruct fine-grained and sparse spectral chromaticity and a degradation-aware, spatially-adaptive noise estimation module that captures anisotropic noise across iterative stages. Extensive experiments on both synthetic and real-world CASSI datasets demonstrate that our method achieves superior performance in both spectral and chromaticity fidelity. Code and models will be publicly available.
>
---
#### [new 022] Does Audio Matter for Modern Video-LLMs and Their Benchmarks?
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文研究视频大模型中音频的作用，指出当前多数评估忽略音频。作者分析音频对视频理解的影响，提出轻量压缩方法，并构建了含音频的评测集，推动真实场景下的音视频联合建模。**

- **链接: [http://arxiv.org/pdf/2509.17901v1](http://arxiv.org/pdf/2509.17901v1)**

> **作者:** Geewook Kim; Minjoon Seo
>
> **备注:** 5 pages, 2 figures, under review. Project page: https://github.com/naver-ai/LLaVA-AV-SSM
>
> **摘要:** Modern multimodal large language models often claim "video understanding," yet most evaluations use muted videos or simply discard audio. We ask a direct question: how much does audio actually matter for contemporary Video-LLMs and the benchmarks that certify them? We audit widely used suites and observe that many items are even solvable from a single frame, rendering audio largely redundant. Building on LLaVA-OneVision architecture, we attach a speech/audio encoder (e.g., Whisper) and analyze when audio helps, while addressing audio token explosion with a lightweight Mamba-based state-space token compressor. We find that audio yields minimal gains on recent video benchmarks but is decisive on curated, audio-sensitive subsets. To enable faithful evaluation, we release AVQA-Hard and Music-AVQA-Hard, our model, and code. Our findings surface a growing gap between current academic practice and real-world expectations, and provide practical tools for scalable audio-visual Video-LLMs. We will fully open-source our work at https://github.com/naver-ai/LLaVA-AV-SSM.
>
---
#### [new 023] Adaptive Fast-and-Slow Visual Program Reasoning for Long-Form VideoQA
- **分类: cs.CV**

- **简介: 该论文提出FS-VisPR框架，用于长视频问答任务。针对现有方法依赖闭源模型、缺乏系统推理的问题，设计了快慢结合的视觉程序推理机制，并构建高质量数据集与参数优化策略，提升效率和可靠性。**

- **链接: [http://arxiv.org/pdf/2509.17743v1](http://arxiv.org/pdf/2509.17743v1)**

> **作者:** Chenglin Li; Feng Han; FengTao; Ruilin Li; Qianglong Chen; Jingqi Tong; Yin Zhang; Jiaqi Wang
>
> **摘要:** Large language models (LLMs) have shown promise in generating program workflows for visual tasks. However, previous approaches often rely on closed-source models, lack systematic reasoning, and struggle with long-form video question answering (videoQA). To address these challenges, we introduce the FS-VisPR framework, an adaptive visual program reasoning approach that balances fast reasoning for simple queries with slow reasoning for difficult ones. First, we design efficient visual modules (e.g., key clip retrieval and subtitle retrieval) to support long-form video tasks. Then, we construct a diverse and high-quality fast-slow reasoning dataset with a strong LLM to align open-source language models' ability to generate visual program workflows as FS-LLM. Next, we design a fast-slow reasoning framework with FS-LLM: Simple queries are directly solved by VideoLLMs, while difficult ones invoke visual program reasoning, motivated by human-like reasoning processes. During this process, low-confidence fast-thinking answers will trigger a second-stage slow-reasoning process, and a fallback mechanism to fast reasoning is activated if the program execution fails. Moreover, we improve visual programs through parameter search during both training and inference. By adjusting the parameters of the visual modules within the program, multiple variants are generated: during training, programs that yield correct answers are selected, while during inference, the program with the highest confidence result is applied. Experiments show that FS-VisPR improves both efficiency and reliability in visual program workflows. It achieves 50.4% accuracy on LVBench, surpassing GPT-4o, matching the performance of Qwen2.5VL-72B on VideoMME.
>
---
#### [new 024] Improved mmFormer for Liver Fibrosis Staging via Missing-Modality Compensation
- **分类: cs.CV**

- **简介: 该论文针对医学影像中多模态MRI数据缺失问题，提出改进的mmFormer模型。通过引入缺失模态补偿模块和交叉验证集成策略，提升肝纤维化分期任务的鲁棒性与准确性，在CARE 2025挑战赛LiFS任务中取得良好效果。**

- **链接: [http://arxiv.org/pdf/2509.16436v1](http://arxiv.org/pdf/2509.16436v1)**

> **作者:** Zhejia Zhang; Junjie Wang; Le Zhang
>
> **摘要:** In real-world clinical settings, magnetic resonance imaging (MRI) frequently suffers from missing modalities due to equipment variability or patient cooperation issues, which can significantly affect model performance. To address this issue, we propose a multimodal MRI classification model based on the mmFormer architecture with an adaptive module for handling arbitrary combinations of missing modalities. Specifically, this model retains the hybrid modality-specific encoders and the modality-correlated encoder from mmFormer to extract consistent lesion features across available modalities. In addition, we integrate a missing-modality compensation module which leverages zero-padding, modality availability masks, and a Delta Function with learnable statistical parameters to dynamically synthesize proxy features for recovering missing information. To further improve prediction performance, we adopt a cross-validation ensemble strategy by training multiple models on different folds and applying soft voting during inference. This method is evaluated on the test set of Comprehensive Analysis & Computing of REal-world medical images (CARE) 2025 challenge, targeting the Liver Fibrosis Staging (LiFS) task based on non-contrast dynamic MRI scans including T1-weighted imaging (T1WI), T2-weighted imaging (T2WI), and diffusion-weighted imaging (DWI). For Cirrhosis Detection and Substantial Fibrosis Detection on in-distribution vendors, our model obtains accuracies of 66.67%, and 74.17%, and corresponding area under the curve (AUC) scores of 71.73% and 68.48%, respectively.
>
---
#### [new 025] AHA -- Predicting What Matters Next: Online Highlight Detection Without Looking Ahead
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Aha，一种用于在线视频高亮检测的框架，解决实时视频流中无法预知未来帧的问题。Aha利用多模态模型和Dynamic SinkCache机制，在不依赖全视频的情况下实现SOTA性能，适用于机器人等实时任务场景。**

- **链接: [http://arxiv.org/pdf/2509.16421v1](http://arxiv.org/pdf/2509.16421v1)**

> **作者:** Aiden Chang; Celso De Melo; Stephanie M. Lukin
>
> **备注:** Accepted at NeurIPS 2025, 32 pages, 5 figures
>
> **摘要:** Real-time understanding of continuous video streams is essential for intelligent agents operating in high-stakes environments, including autonomous vehicles, surveillance drones, and disaster response robots. Yet, most existing video understanding and highlight detection methods assume access to the entire video during inference, making them unsuitable for online or streaming scenarios. In particular, current models optimize for offline summarization, failing to support step-by-step reasoning needed for real-time decision-making. We introduce Aha, an autoregressive highlight detection framework that predicts the relevance of each video frame against a task described in natural language. Without accessing future video frames, Aha utilizes a multimodal vision-language model and lightweight, decoupled heads trained on a large, curated dataset of human-centric video labels. To enable scalability, we introduce the Dynamic SinkCache mechanism that achieves constant memory usage across infinite-length streams without degrading performance on standard benchmarks. This encourages the hidden representation to capture high-level task objectives, enabling effective frame-level rankings for informativeness, relevance, and uncertainty with respect to the natural language task. Aha achieves state-of-the-art (SOTA) performance on highlight detection benchmarks, surpassing even prior offline, full-context approaches and video-language models by +5.9% on TVSum and +8.3% on Mr.Hisum in mAP (mean Average Precision). We explore Aha's potential for real-world robotics applications given a task-oriented natural language input and a continuous, robot-centric video. Both experiments demonstrate Aha's potential effectiveness as a real-time reasoning module for downstream planning and long-horizon understanding.
>
---
#### [new 026] TS-P$^2$CL: Plug-and-Play Dual Contrastive Learning for Vision-Guided Medical Time Series Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TS-P²CL，用于视觉引导的医疗时间序列分类任务。针对跨个体异质性导致的泛化能力差问题，将1D信号转为2D伪图像，并采用双对比学习策略，提升模型的领域不变特征学习能力。**

- **链接: [http://arxiv.org/pdf/2509.17802v1](http://arxiv.org/pdf/2509.17802v1)**

> **作者:** Qi'ao Xu; Pengfei Wang; Bo Zhong; Tianwen Qian; Xiaoling Wang; Ye Wang; Hong Yu
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Medical time series (MedTS) classification is pivotal for intelligent healthcare, yet its efficacy is severely limited by poor cross-subject generation due to the profound cross-individual heterogeneity. Despite advances in architectural innovations and transfer learning techniques, current methods remain constrained by modality-specific inductive biases that limit their ability to learn universally invariant representations. To overcome this, we propose TS-P$^2$CL, a novel plug-and-play framework that leverages the universal pattern recognition capabilities of pre-trained vision models. We introduce a vision-guided paradigm that transforms 1D physiological signals into 2D pseudo-images, establishing a bridge to the visual domain. This transformation enables implicit access to rich semantic priors learned from natural images. Within this unified space, we employ a dual-contrastive learning strategy: intra-modal consistency enforces temporal coherence, while cross-modal alignment aligns time-series dynamics with visual semantics, thereby mitigating individual-specific biases and learning robust, domain-invariant features. Extensive experiments on six MedTS datasets demonstrate that TS-P$^2$CL consistently outperforms fourteen methods in both subject-dependent and subject-independent settings.
>
---
#### [new 027] From Restoration to Reconstruction: Rethinking 3D Gaussian Splatting for Underwater Scenes
- **分类: cs.CV**

- **简介: 该论文针对水下场景三维重建任务，解决物理模型简化导致的渲染质量与几何精度低的问题。提出R-Splatting框架，融合图像修复与3D高斯溅射，并引入光照生成和不确定性透明度优化，提升重建效果。**

- **链接: [http://arxiv.org/pdf/2509.17789v1](http://arxiv.org/pdf/2509.17789v1)**

> **作者:** Guoxi Huang; Haoran Wang; Zipeng Qi; Wenjun Lu; David Bull; Nantheera Anantrasirichai
>
> **摘要:** Underwater image degradation poses significant challenges for 3D reconstruction, where simplified physical models often fail in complex scenes. We propose \textbf{R-Splatting}, a unified framework that bridges underwater image restoration (UIR) with 3D Gaussian Splatting (3DGS) to improve both rendering quality and geometric fidelity. Our method integrates multiple enhanced views produced by diverse UIR models into a single reconstruction pipeline. During inference, a lightweight illumination generator samples latent codes to support diverse yet coherent renderings, while a contrastive loss ensures disentangled and stable illumination representations. Furthermore, we propose \textit{Uncertainty-Aware Opacity Optimization (UAOO)}, which models opacity as a stochastic function to regularize training. This suppresses abrupt gradient responses triggered by illumination variation and mitigates overfitting to noisy or view-specific artifacts. Experiments on Seathru-NeRF and our new BlueCoral3D dataset demonstrate that R-Splatting outperforms strong baselines in both rendering quality and geometric accuracy.
>
---
#### [new 028] A Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion
- **分类: cs.CV**

- **简介: 该论文针对RGB-T人群计数任务，旨在提升复杂环境下人群定位精度。提出Dual Modulation框架，包含空间调制注意力（SMA）和自适应融合调制（AFM），分别优化局部关注与多模态融合，有效提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.17079v1](http://arxiv.org/pdf/2509.17079v1)**

> **作者:** Yuhong Feng; Hongtao Chen; Qi Zhang; Jie Chen; Zhaoxi He; Mingzhe Liu; Jianghai Liao
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Accurate RGB-Thermal (RGB-T) crowd counting is crucial for public safety in challenging conditions. While recent Transformer-based methods excel at capturing global context, their inherent lack of spatial inductive bias causes attention to spread to irrelevant background regions, compromising crowd localization precision. Furthermore, effectively bridging the gap between these distinct modalities remains a major hurdle. To tackle this, we propose the Dual Modulation Framework, comprising two modules: Spatially Modulated Attention (SMA), which improves crowd localization by using a learnable Spatial Decay Mask to penalize attention between distant tokens and prevent focus from spreading to the background; and Adaptive Fusion Modulation (AFM), which implements a dynamic gating mechanism to prioritize the most reliable modality for adaptive cross-modal fusion. Extensive experiments on RGB-T crowd counting datasets demonstrate the superior performance of our method compared to previous works. Code available at https://github.com/Cht2924/RGBT-Crowd-Counting.
>
---
#### [new 029] $\mathtt{M^3VIR}$: A Large-Scale Multi-Modality Multi-View Synthesized Benchmark Dataset for Image Restoration and Content Creation
- **分类: cs.CV**

- **简介: 该论文提出了$\mathtt{M^3VIR}$，一个面向图像修复与内容生成的多模态、多视角合成数据集。针对现有数据集在游戏内容真实性及可控视频生成方面的不足，提供了高保真游戏场景和多任务基准，推动AI在云游戏中的应用研究。**

- **链接: [http://arxiv.org/pdf/2509.16873v1](http://arxiv.org/pdf/2509.16873v1)**

> **作者:** Yuanzhi Li; Lebin Zhou; Nam Ling; Zhenghao Chen; Wei Wang; Wei Jiang
>
> **摘要:** The gaming and entertainment industry is rapidly evolving, driven by immersive experiences and the integration of generative AI (GAI) technologies. Training such models effectively requires large-scale datasets that capture the diversity and context of gaming environments. However, existing datasets are often limited to specific domains or rely on artificial degradations, which do not accurately capture the unique characteristics of gaming content. Moreover, benchmarks for controllable video generation remain absent. To address these limitations, we introduce $\mathtt{M^3VIR}$, a large-scale, multi-modal, multi-view dataset specifically designed to overcome the shortcomings of current resources. Unlike existing datasets, $\mathtt{M^3VIR}$ provides diverse, high-fidelity gaming content rendered with Unreal Engine 5, offering authentic ground-truth LR-HR paired and multi-view frames across 80 scenes in 8 categories. It includes $\mathtt{M^3VIR\_MR}$ for super-resolution (SR), novel view synthesis (NVS), and combined NVS+SR tasks, and $\mathtt{M^3VIR\_{MS}}$, the first multi-style, object-level ground-truth set enabling research on controlled video generation. Additionally, we benchmark several state-of-the-art SR and NVS methods to establish performance baselines. While no existing approaches directly handle controlled video generation, $\mathtt{M^3VIR}$ provides a benchmark for advancing this area. By releasing the dataset, we aim to facilitate research in AI-powered restoration, compression, and controllable content generation for next-generation cloud gaming and entertainment.
>
---
#### [new 030] GraDeT-HTR: A Resource-Efficient Bengali Handwritten Text Recognition System utilizing Grapheme-based Tokenizer and Decoder-only Transformer
- **分类: cs.CV**

- **简介: 该论文提出GraDeT-HTR，一种高效的孟加拉手写文本识别系统。针对孟加拉文复杂结构和数据稀缺问题，采用基于图符的分词器与解码器型Transformer，提升识别精度并取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.18081v1](http://arxiv.org/pdf/2509.18081v1)**

> **作者:** Md. Mahmudul Hasan; Ahmed Nesar Tahsin Choudhury; Mahmudul Hasan; Md. Mosaddek Khan
>
> **备注:** 7 pages. Accepted at the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) System Demonstrations. Equal Contribution: Md. Mahmudul Hasan and Ahmed Nesar Tahsin Choudhury
>
> **摘要:** Despite Bengali being the sixth most spoken language in the world, handwritten text recognition (HTR) systems for Bengali remain severely underdeveloped. The complexity of Bengali script--featuring conjuncts, diacritics, and highly variable handwriting styles--combined with a scarcity of annotated datasets makes this task particularly challenging. We present GraDeT-HTR, a resource-efficient Bengali handwritten text recognition system based on a Grapheme-aware Decoder-only Transformer architecture. To address the unique challenges of Bengali script, we augment the performance of a decoder-only transformer by integrating a grapheme-based tokenizer and demonstrate that it significantly improves recognition accuracy compared to conventional subword tokenizers. Our model is pretrained on large-scale synthetic data and fine-tuned on real human-annotated samples, achieving state-of-the-art performance on multiple benchmark datasets.
>
---
#### [new 031] Efficient Rectified Flow for Image Fusion
- **分类: cs.CV**

- **简介: 该论文针对图像融合任务，提出RFfusion方法，基于Rectified Flow实现一步采样，减少计算复杂度。同时设计任务专用VAE架构和两阶段训练策略，提升融合质量与推理效率。**

- **链接: [http://arxiv.org/pdf/2509.16549v1](http://arxiv.org/pdf/2509.16549v1)**

> **作者:** Zirui Wang; Jiayi Zhang; Tianwei Guan; Yuhan Zhou; Xingyuan Li; Minjing Dong; Jinyuan Liu
>
> **摘要:** Image fusion is a fundamental and important task in computer vision, aiming to combine complementary information from different modalities to fuse images. In recent years, diffusion models have made significant developments in the field of image fusion. However, diffusion models often require complex computations and redundant inference time, which reduces the applicability of these methods. To address this issue, we propose RFfusion, an efficient one-step diffusion model for image fusion based on Rectified Flow. We incorporate Rectified Flow into the image fusion task to straighten the sampling path in the diffusion model, achieving one-step sampling without the need for additional training, while still maintaining high-quality fusion results. Furthermore, we propose a task-specific variational autoencoder (VAE) architecture tailored for image fusion, where the fusion operation is embedded within the latent space to further reduce computational complexity. To address the inherent discrepancy between conventional reconstruction-oriented VAE objectives and the requirements of image fusion, we introduce a two-stage training strategy. This approach facilitates the effective learning and integration of complementary information from multi-modal source images, thereby enabling the model to retain fine-grained structural details while significantly enhancing inference efficiency. Extensive experiments demonstrate that our method outperforms other state-of-the-art methods in terms of both inference speed and fusion quality. Code is available at https://github.com/zirui0625/RFfusion.
>
---
#### [new 032] Multimodal Medical Image Classification via Synergistic Learning Pre-training
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态医学图像分类任务，旨在解决标注数据稀缺下的模态融合问题。提出一种协同预训练框架，结合一致性、重构和对齐学习，并设计了多模态微调方法与分布偏移策略，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.17492v1](http://arxiv.org/pdf/2509.17492v1)**

> **作者:** Qinghua Lin; Guang-Hai Liu; Zuoyong Li; Yang Li; Yuting Jiang; Xiang Wu
>
> **摘要:** Multimodal pathological images are usually in clinical diagnosis, but computer vision-based multimodal image-assisted diagnosis faces challenges with modality fusion, especially in the absence of expert-annotated data. To achieve the modality fusion in multimodal images with label scarcity, we propose a novel ``pretraining + fine-tuning" framework for multimodal semi-supervised medical image classification. Specifically, we propose a synergistic learning pretraining framework of consistency, reconstructive, and aligned learning. By treating one modality as an augmented sample of another modality, we implement a self-supervised learning pre-train, enhancing the baseline model's feature representation capability. Then, we design a fine-tuning method for multimodal fusion. During the fine-tuning stage, we set different encoders to extract features from the original modalities and provide a multimodal fusion encoder for fusion modality. In addition, we propose a distribution shift method for multimodal fusion features, which alleviates the prediction uncertainty and overfitting risks caused by the lack of labeled samples. We conduct extensive experiments on the publicly available gastroscopy image datasets Kvasir and Kvasirv2. Quantitative and qualitative results demonstrate that the proposed method outperforms the current state-of-the-art classification methods. The code will be released at: https://github.com/LQH89757/MICS.
>
---
#### [new 033] Stencil: Subject-Driven Generation with Context Guidance
- **分类: cs.CV**

- **简介: 该论文提出Stencil框架，用于主题驱动的文本到图像生成任务。针对现有微调方法在质量和效率间的权衡问题，Stencil结合轻量模型微调与大模型冻结推理，实现高效且高保真度的主题图像生成。**

- **链接: [http://arxiv.org/pdf/2509.17120v1](http://arxiv.org/pdf/2509.17120v1)**

> **作者:** Gordon Chen; Ziqi Huang; Cheston Tan; Ziwei Liu
>
> **备注:** Accepted as Spotlight at ICIP 2025
>
> **摘要:** Recent text-to-image diffusion models can generate striking visuals from text prompts, but they often fail to maintain subject consistency across generations and contexts. One major limitation of current fine-tuning approaches is the inherent trade-off between quality and efficiency. Fine-tuning large models improves fidelity but is computationally expensive, while fine-tuning lightweight models improves efficiency but compromises image fidelity. Moreover, fine-tuning pre-trained models on a small set of images of the subject can damage the existing priors, resulting in suboptimal results. To this end, we present Stencil, a novel framework that jointly employs two diffusion models during inference. Stencil efficiently fine-tunes a lightweight model on images of the subject, while a large frozen pre-trained model provides contextual guidance during inference, injecting rich priors to enhance generation with minimal overhead. Stencil excels at generating high-fidelity, novel renditions of the subject in less than a minute, delivering state-of-the-art performance and setting a new benchmark in subject-driven generation.
>
---
#### [new 034] Looking in the mirror: A faithful counterfactual explanation method for interpreting deep image classification models
- **分类: cs.CV**

- **简介: 该论文提出Mirror-CFE方法，用于生成深度图像分类模型的反事实解释。针对现有方法依赖外部模型、忽略分类器自身特征空间的问题，Mirror-CFE直接在分类器特征空间中操作，生成具有高保真度和可解释性的反事实样本，提升模型可解释性。**

- **链接: [http://arxiv.org/pdf/2509.16822v1](http://arxiv.org/pdf/2509.16822v1)**

> **作者:** Townim Faisal Chowdhury; Vu Minh Hieu Phan; Kewen Liao; Nanyu Dong; Minh-Son To; Anton Hengel; Johan Verjans; Zhibin Liao
>
> **备注:** Accepted at IEEE/CVF International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Counterfactual explanations (CFE) for deep image classifiers aim to reveal how minimal input changes lead to different model decisions, providing critical insights for model interpretation and improvement. However, existing CFE methods often rely on additional image encoders and generative models to create plausible images, neglecting the classifier's own feature space and decision boundaries. As such, they do not explain the intrinsic feature space and decision boundaries learned by the classifier. To address this limitation, we propose Mirror-CFE, a novel method that generates faithful counterfactual explanations by operating directly in the classifier's feature space, treating decision boundaries as mirrors that ``reflect'' feature representations in the mirror. Mirror-CFE learns a mapping function from feature space to image space while preserving distance relationships, enabling smooth transitions between source images and their counterfactuals. Through extensive experiments on four image datasets, we demonstrate that Mirror-CFE achieves superior performance in validity while maintaining input resemblance compared to state-of-the-art explanation methods. Finally, mirror-CFE provides interpretable visualization of the classifier's decision process by generating step-wise transitions that reveal how features evolve as classification confidence changes.
>
---
#### [new 035] From Canopy to Ground via ForestGen3D: Learning Cross-Domain Generation of 3D Forest Structure from Aerial-to-Terrestrial LiDAR
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ForestGen3D，一种基于条件扩散模型的生成框架，旨在从航拍LiDAR（ALS）数据生成高保真地面LiDAR（TLS）风格的三维森林结构，解决大规模低成本获取森林细部结构的问题。**

- **链接: [http://arxiv.org/pdf/2509.16346v1](http://arxiv.org/pdf/2509.16346v1)**

> **作者:** Juan Castorena; E. Louise Loudermilk; Scott Pokswinski; Rodman Linn
>
> **摘要:** The 3D structure of living and non-living components in ecosystems plays a critical role in determining ecological processes and feedbacks from both natural and human-driven disturbances. Anticipating the effects of wildfire, drought, disease, or atmospheric deposition depends on accurate characterization of 3D vegetation structure, yet widespread measurement remains prohibitively expensive and often infeasible. We introduce ForestGen3D, a novel generative modeling framework that synthesizes high-fidelity 3D forest structure using only aerial LiDAR (ALS) inputs. ForestGen3D is based on conditional denoising diffusion probabilistic models (DDPMs) trained on co-registered ALS/TLS (terrestrial LiDAR) data. The model learns to generate TLS-like 3D point clouds conditioned on sparse ALS observations, effectively reconstructing occluded sub-canopy detail at scale. To ensure ecological plausibility, we introduce a geometric containment prior based on the convex hull of ALS observations and provide theoretical and empirical guarantees that generated structures remain spatially consistent. We evaluate ForestGen3D at tree, plot, and landscape scales using real-world data from mixed conifer ecosystems, and show that it produces high-fidelity reconstructions that closely match TLS references in terms of geometric similarity and biophysical metrics, such as tree height, DBH, crown diameter and crown volume. Additionally, we demonstrate that the containment property can serve as a practical proxy for generation quality in settings where TLS ground truth is unavailable. Our results position ForestGen3D as a scalable tool for ecological modeling, wildfire simulation, and structural fuel characterization in ALS-only environments.
>
---
#### [new 036] InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention
- **分类: cs.CV**

- **简介: 该论文针对Layout-to-Image（L2I）生成任务，提出InstanceAssemble方法，通过实例组装注意力机制结合布局条件和多模态内容控制，提升图像生成的精度与可控性，并构建了Denselayout数据集和LGS评估指标。**

- **链接: [http://arxiv.org/pdf/2509.16691v1](http://arxiv.org/pdf/2509.16691v1)**

> **作者:** Qiang Xiang; Shuang Sun; Binglei Li; Dejia Song; Huaxia Li; Nemo Chen; Xu Tang; Yao Hu; Junping Zhang
>
> **备注:** Accepted in NeurIPS 2025
>
> **摘要:** Diffusion models have demonstrated remarkable capabilities in generating high-quality images. Recent advancements in Layout-to-Image (L2I) generation have leveraged positional conditions and textual descriptions to facilitate precise and controllable image synthesis. Despite overall progress, current L2I methods still exhibit suboptimal performance. Therefore, we propose InstanceAssemble, a novel architecture that incorporates layout conditions via instance-assembling attention, enabling position control with bounding boxes (bbox) and multimodal content control including texts and additional visual content. Our method achieves flexible adaption to existing DiT-based T2I models through light-weighted LoRA modules. Additionally, we propose a Layout-to-Image benchmark, Denselayout, a comprehensive benchmark for layout-to-image generation, containing 5k images with 90k instances in total. We further introduce Layout Grounding Score (LGS), an interpretable evaluation metric to more precisely assess the accuracy of L2I generation. Experiments demonstrate that our InstanceAssemble method achieves state-of-the-art performance under complex layout conditions, while exhibiting strong compatibility with diverse style LoRA modules.
>
---
#### [new 037] Advancing Reference-free Evaluation of Video Captions with Factual Analysis
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视频描述生成任务，旨在解决传统评估方法依赖人工标注参考文本的问题。提出了VC-Inspector，一个无需参考的、基于事实的视频描述质量评估框架，利用大语言模型生成伪数据训练多模态评估模型，实现更客观、可扩展的评估。**

- **链接: [http://arxiv.org/pdf/2509.16538v1](http://arxiv.org/pdf/2509.16538v1)**

> **作者:** Shubhashis Roy Dipta; Tz-Ying Wu; Subarna Tripathi
>
> **摘要:** Video captions offer concise snapshots of actors, objects, and actions within a video, serving as valuable assets for applications such as question answering and event localization. However, acquiring human annotations for video captions is costly or even impractical, especially when dealing with diverse video domains. Existing models trained on supervised datasets face challenges in evaluating performance across different domains due to the reliance on reference-based evaluation protocols, which necessitate ground truth captions. This assumption is unrealistic for evaluating videos in the wild. To address these limitations, we propose a reference-free evaluation framework that does not require ground truth captions, focusing on factual grounding to ensure accurate assessment of caption quality. We introduce VC-Inspector, a novel caption quality evaluator that is both reference-free and factually grounded. Utilizing large language models, we generate pseudo captions of varying quality based on supervised data, which are subsequently used to train a multimodal model (i.e., Qwen2.5-VL) as the evaluator. Our approach demonstrates superior alignment with human judgments on the VATEX-Eval dataset, outperforming existing methods. The performance also generalizes to image caption datasets, Flickr8K-Expert and Flickr8K-CF, when viewing images as 1-frame videos. Overall, VC-Inspector offers a scalable and generalizable solution for evaluating the factual accuracy of video captions, paving the way for more effective and objective assessment methodologies in diverse video domains.
>
---
#### [new 038] ST-GS: Vision-Based 3D Semantic Occupancy Prediction with Spatial-Temporal Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ST-GS框架，用于基于视觉的3D语义占用预测任务。针对现有方法在多视角空间交互和时序一致性上的不足，设计了空间聚合策略与几何感知时序融合方案，提升了建模效果和时序稳定性。**

- **链接: [http://arxiv.org/pdf/2509.16552v1](http://arxiv.org/pdf/2509.16552v1)**

> **作者:** Xiaoyang Yan; Muleilan Pei; Shaojie Shen
>
> **摘要:** 3D occupancy prediction is critical for comprehensive scene understanding in vision-centric autonomous driving. Recent advances have explored utilizing 3D semantic Gaussians to model occupancy while reducing computational overhead, but they remain constrained by insufficient multi-view spatial interaction and limited multi-frame temporal consistency. To overcome these issues, in this paper, we propose a novel Spatial-Temporal Gaussian Splatting (ST-GS) framework to enhance both spatial and temporal modeling in existing Gaussian-based pipelines. Specifically, we develop a guidance-informed spatial aggregation strategy within a dual-mode attention mechanism to strengthen spatial interaction in Gaussian representations. Furthermore, we introduce a geometry-aware temporal fusion scheme that effectively leverages historical context to improve temporal continuity in scene completion. Extensive experiments on the large-scale nuScenes occupancy prediction benchmark showcase that our proposed approach not only achieves state-of-the-art performance but also delivers markedly better temporal consistency compared to existing Gaussian-based methods.
>
---
#### [new 039] CardiacCLIP: Video-based CLIP Adaptation for LVEF Prediction in a Few-shot Manner
- **分类: cs.CV**

- **简介: 该论文提出CardiacCLIP，用于小样本下基于超声视频的LVEF预测。针对现有方法依赖大规模标注数据和忽略时序动态的问题，引入MFL和EchoZoom机制，提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2509.17065v1](http://arxiv.org/pdf/2509.17065v1)**

> **作者:** Yao Du; Jiarong Guo; Xiaomeng Li
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Echocardiography is a vital non-invasive modality for cardiac assessment, with left ventricular ejection fraction (LVEF) serving as a key indicator of heart function. Existing LVEF estimation methods depend on large-scale annotated video datasets, which are costly and limit adaptability across various clinical settings. Recent vision-language models for echocardiography, such as EchoCLIP, apply image-to-text pretraining but fail to capture crucial temporal dynamics and localized cardiac structures essential for accurate diagnosis. To address these challenges, we propose CardiacCLIP, a video-based framework that enhances LVEF prediction through attention-based frame aggregation and multi-resolution input scaling. Specifically, we introduce MFL (Multi Frame Learning), a novel attention-based mechanism for selectively fusing informative frames, and EchoZoom, a multi-scale feature extraction strategy that refines spatial representations of cardiac structures. As a novel adaptation of CLIP models for few-shot echocardiogram video analysis, our approach significantly improves diagnostic accuracy, reducing MAE by 2.07 on the EchoNet-Dynamic dataset under 1-shot setting. The code is available at https://github.com/xmed-lab/CardiacCLIP.
>
---
#### [new 040] V-CECE: Visual Counterfactual Explanations via Conceptual Edits
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出V-CECE，一种无需训练的黑盒反事实生成框架，通过语义编辑生成可解释的视觉反事实解释。利用预训练扩散模型，在不依赖分类器内部结构的情况下，实现符合人类推理的图像编辑，揭示了人类与神经模型在判断上的差异。**

- **链接: [http://arxiv.org/pdf/2509.16567v1](http://arxiv.org/pdf/2509.16567v1)**

> **作者:** Nikolaos Spanos; Maria Lymperaiou; Giorgos Filandrianos; Konstantinos Thomas; Athanasios Voulodimos; Giorgos Stamou
>
> **备注:** Accepted in NeurIPS 2025
>
> **摘要:** Recent black-box counterfactual generation frameworks fail to take into account the semantic content of the proposed edits, while relying heavily on training to guide the generation process. We propose a novel, plug-and-play black-box counterfactual generation framework, which suggests step-by-step edits based on theoretical guarantees of optimal edits to produce human-level counterfactual explanations with zero training. Our framework utilizes a pre-trained image editing diffusion model, and operates without access to the internals of the classifier, leading to an explainable counterfactual generation process. Throughout our experimentation, we showcase the explanatory gap between human reasoning and neural model behavior by utilizing both Convolutional Neural Network (CNN), Vision Transformer (ViT) and Large Vision Language Model (LVLM) classifiers, substantiated through a comprehensive human evaluation.
>
---
#### [new 041] Selecting Optimal Camera Views for Gait Analysis: A Multi-Metric Assessment of 2D Projections
- **分类: cs.CV**

- **简介: 该论文研究2D无标记步态分析中摄像头视角（正面 vs 侧面）对精度的影响。通过对比3D运动捕捉数据，评估不同视角在多种指标下的表现。结果显示侧视更优于矢状面参数，正视更适合躯干对称性分析，为临床部署提供依据。**

- **链接: [http://arxiv.org/pdf/2509.17805v1](http://arxiv.org/pdf/2509.17805v1)**

> **作者:** Dong Chen; Huili Peng; Yong Hu; Kenneth MC. Cheung
>
> **摘要:** Objective: To systematically quantify the effect of the camera view (frontal vs. lateral) on the accuracy of 2D markerless gait analysis relative to 3D motion capture ground truth. Methods: Gait data from 18 subjects were recorded simultaneously using frontal, lateral and 3D motion capture systems. Pose estimation used YOLOv8. Four metrics were assessed to evaluate agreement: Dynamic Time Warping (DTW) for temporal alignment, Maximum Cross-Correlation (MCC) for signal similarity, Kullback-Leibler Divergence (KLD) for distribution differences, and Information Entropy (IE) for complexity. Wilcoxon signed-rank tests (significance: $p < 0.05$) and Cliff's delta ($\delta$) were used to measure statistical differences and effect sizes. Results: Lateral views significantly outperformed frontal views for sagittal plane kinematics: step length (DTW: $53.08 \pm 24.50$ vs. $69.87 \pm 25.36$, $p = 0.005$) and knee rotation (DTW: $106.46 \pm 38.57$ vs. $155.41 \pm 41.77$, $p = 0.004$). Frontal views were superior for symmetry parameters: trunk rotation (KLD: $0.09 \pm 0.06$ vs. $0.30 \pm 0.19$, $p < 0.001$) and wrist-to-hipmid distance (MCC: $105.77 \pm 29.72$ vs. $75.20 \pm 20.38$, $p = 0.003$). Effect sizes were medium-to-large ($\delta: 0.34$--$0.76$). Conclusion: Camera view critically impacts gait parameter accuracy. Lateral views are optimal for sagittal kinematics; frontal views excel for trunk symmetry. Significance: This first systematic evidence enables data-driven camera deployment in 2D gait analysis, enhancing clinical utility. Future implementations should leverage both views via disease-oriented setups.
>
---
#### [new 042] AlignedGen: Aligning Style Across Generated Images
- **分类: cs.CV**

- **简介: 该论文提出AlignedGen，旨在解决扩散模型在生成图像时风格一致性不足的问题。通过引入ShiftPE和AAS技术，提升DiT模型的跨图风格对齐能力，并支持外部图像作为风格参考，实现高质量、一致性的文本到图像生成。**

- **链接: [http://arxiv.org/pdf/2509.17088v1](http://arxiv.org/pdf/2509.17088v1)**

> **作者:** Jiexuan Zhang; Yiheng Du; Qian Wang; Weiqi Li; Yu Gu; Jian Zhang
>
> **摘要:** Despite their generative power, diffusion models struggle to maintain style consistency across images conditioned on the same style prompt, hindering their practical deployment in creative workflows. While several training-free methods attempt to solve this, they are constrained to the U-Net architecture, which not only leads to low-quality results and artifacts like object repetition but also renders them incompatible with superior Diffusion Transformer (DiT). To address these issues, we introduce AlignedGen, a novel training-free framework that enhances style consistency across images generated by DiT models. Our work first reveals a critical insight: naive attention sharing fails in DiT due to conflicting positional signals from improper position embeddings. We introduce Shifted Position Embedding (ShiftPE), an effective solution that resolves this conflict by allocating a non-overlapping set of positional indices to each image. Building on this foundation, we develop Advanced Attention Sharing (AAS), a suite of three techniques meticulously designed to fully unleash the potential of attention sharing within the DiT. Furthermore, to broaden the applicability of our method, we present an efficient query, key, and value feature extraction algorithm, enabling our method to seamlessly incorporate external images as style references. Extensive experimental results validate that our method effectively enhances style consistency across generated images while maintaining precise text-to-image alignment.
>
---
#### [new 043] Towards Generalized Synapse Detection Across Invertebrate Species
- **分类: cs.CV**

- **简介: 该论文针对跨无脊椎动物物种的突触检测任务，旨在解决大规模、高精度突触识别难题。作者提出了轻量级模型SimpSyn，并构建了多物种EM数据集，验证了其在突触位点检测中的优越性能和实用性。**

- **链接: [http://arxiv.org/pdf/2509.17041v1](http://arxiv.org/pdf/2509.17041v1)**

> **作者:** Samia Mohinta; Daniel Franco-Barranco; Shi Yan Lee; Albert Cardona
>
> **摘要:** Behavioural differences across organisms, whether healthy or pathological, are closely tied to the structure of their neural circuits. Yet, the fine-scale synaptic changes that give rise to these variations remain poorly understood, in part due to persistent challenges in detecting synapses reliably and at scale. Volume electron microscopy (EM) offers the resolution required to capture synaptic architecture, but automated detection remains difficult due to sparse annotations, morphological variability, and cross-dataset domain shifts. To address this, we make three key contributions. First, we curate a diverse EM benchmark spanning four datasets across two invertebrate species: adult and larval Drosophila melanogaster, and Megaphragma viggianii (micro-WASP). Second, we propose SimpSyn, a single-stage Residual U-Net trained to predict dual-channel spherical masks around pre- and post-synaptic sites, designed to prioritize training and inference speeds and annotation efficiency over architectural complexity. Third, we benchmark SimpSyn against Buhmann et al.'s Synful [1], a state-of-the-art multi-task model that jointly infers synaptic pairs. Despite its simplicity, SimpSyn consistently outperforms Synful in F1-score across all volumes for synaptic site detection. While generalization across datasets remains limited, SimpSyn achieves competitive performance when trained on the combined cohort. Finally, ablations reveal that simple post-processing strategies - such as local peak detection and distance-based filtering - yield strong performance without complex test-time heuristics. Taken together, our results suggest that lightweight models, when aligned with task structure, offer a practical and scalable solution for synapse detection in large-scale connectomic pipelines.
>
---
#### [new 044] Clothing agnostic Pre-inpainting Virtual Try-ON
- **分类: cs.CV**

- **简介: 该论文提出CaP-VTON，用于虚拟试衣任务，旨在解决衣物轮廓残留和下半身检测不准确的问题。通过引入多类别掩码和皮肤修复模块，提升了合成效果的自然性和一致性，尤其在短袖生成方面性能优于Leffa。**

- **链接: [http://arxiv.org/pdf/2509.17654v1](http://arxiv.org/pdf/2509.17654v1)**

> **作者:** Sehyun Kim; Hye Jun Lee; Jiwoo Lee; Taemin Lee
>
> **摘要:** With the development of deep learning technology, virtual try-on technology has become an important application value in the fields of e-commerce, fashion, and entertainment. The recently proposed Leffa has improved the texture distortion problem of diffu-sion-based models, but there are limitations in that the bottom detection inaccuracy and the existing clothing silhouette remain in the synthesis results. To solve this problem, this study proposes CaP-VTON (Clothing agnostic Pre-inpainting Virtual Try-ON). CaP-VTON has improved the naturalness and consistency of whole-body clothing syn-thesis by integrating multi-category masking based on Dress Code and skin inpainting based on Stable Diffusion. In particular, a generate skin module was introduced to solve the skin restoration problem that occurs when long-sleeved images are converted into short-sleeved or sleeveless ones, and high-quality restoration was implemented consider-ing the human body posture and color. As a result, CaP-VTON recorded 92.5\%, which is 15.4\% better than Leffa in short-sleeved synthesis accuracy, and showed the performance of consistently reproducing the style and shape of reference clothing in visual evaluation. These structures maintain model-agnostic properties and are applicable to various diffu-sion-based virtual inspection systems, and can contribute to applications that require high-precision virtual wearing, such as e-commerce, custom styling, and avatar creation.
>
---
#### [new 045] VCE: Safe Autoregressive Image Generation via Visual Contrast Exploitation
- **分类: cs.CV**

- **简介: 该论文提出VCE框架，用于安全的自回归图像生成。针对现有方法难以移除NSFW内容的问题，设计了对比图像对构建和DPO训练策略，有效擦除危险概念，同时保留安全内容，在风格擦除、内容过滤等任务中取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.16986v1](http://arxiv.org/pdf/2509.16986v1)**

> **作者:** Feng Han; Chao Gong; Zhipeng Wei; Jingjing Chen; Yu-Gang Jiang
>
> **摘要:** Recently, autoregressive image generation models have wowed audiences with their remarkable capability in creating surprisingly realistic images. Models such as GPT-4o and LlamaGen can not only produce images that faithfully mimic renowned artistic styles like Ghibli, Van Gogh, or Picasso, but also potentially generate Not-Safe-For-Work (NSFW) content, raising significant concerns regarding copyright infringement and ethical use. Despite these concerns, methods to safeguard autoregressive text-to-image models remain underexplored. Previous concept erasure methods, primarily designed for diffusion models that operate in denoising latent space, are not directly applicable to autoregressive models that generate images token by token. To address this critical gap, we propose Visual Contrast Exploitation (VCE), a novel framework comprising: (1) an innovative contrastive image pair construction paradigm that precisely decouples unsafe concepts from their associated content semantics, and (2) a sophisticated DPO-based training approach that enhances the model's ability to identify and leverage visual contrastive features from image pairs, enabling precise concept erasure. Our comprehensive experiments across three challenging tasks-artist style erasure, explicit content erasure, and object removal-demonstrate that our method effectively secures the model, achieving state-of-the-art results while erasing unsafe concepts and maintaining the integrity of unrelated safe concepts. The code and models are available at https://github.com/Maplebb/VCE.
>
---
#### [new 046] Penalizing Boundary Activation for Object Completeness in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成中的物体不完整问题，分析发现训练中的RandomCrop导致物体断裂。提出一种无需训练的方法，在去噪早期惩罚图像边界激活值，提升预训练扩散模型的物体完整性与图像质量。**

- **链接: [http://arxiv.org/pdf/2509.16968v1](http://arxiv.org/pdf/2509.16968v1)**

> **作者:** Haoyang Xu; Tianhao Zhao; Sibei Yang; Yutian Li
>
> **摘要:** Diffusion models have emerged as a powerful technique for text-to-image (T2I) generation, creating high-quality, diverse images across various domains. However, a common limitation in these models is the incomplete display of objects, where fragments or missing parts undermine the model's performance in downstream applications. In this study, we conduct an in-depth analysis of the incompleteness issue and reveal that the primary factor behind incomplete object generation is the usage of RandomCrop during model training. This widely used data augmentation method, though enhances model generalization ability, disrupts object continuity during training. To address this, we propose a training-free solution that penalizes activation values at image boundaries during the early denoising steps. Our method is easily applicable to pre-trained Stable Diffusion models with minimal modifications and negligible computational overhead. Extensive experiments demonstrate the effectiveness of our method, showing substantial improvements in object integrity and image quality.
>
---
#### [new 047] OmniInsert: Mask-Free Video Insertion of Any Reference via Diffusion Transformer Models
- **分类: cs.CV**

- **简介: 该论文提出OmniInsert，用于无掩码视频插入任务。针对数据稀缺、主体-场景平衡和插入和谐性问题，构建了InsertPipe数据管道和InsertBench基准，并设计了特征注入机制、渐进训练策略及优化方法，提升了插入效果。**

- **链接: [http://arxiv.org/pdf/2509.17627v1](http://arxiv.org/pdf/2509.17627v1)**

> **作者:** Jinshu Chen; Xinghui Li; Xu Bai; Tianxiang Ma; Pengze Zhang; Zhuowei Chen; Gen Li; Lijie Liu; Songtao Zhao; Bingchuan Li; Qian He
>
> **备注:** Github Page: https://phantom-video.github.io/OmniInsert/
>
> **摘要:** Recent advances in video insertion based on diffusion models are impressive. However, existing methods rely on complex control signals but struggle with subject consistency, limiting their practical applicability. In this paper, we focus on the task of Mask-free Video Insertion and aim to resolve three key challenges: data scarcity, subject-scene equilibrium, and insertion harmonization. To address the data scarcity, we propose a new data pipeline InsertPipe, constructing diverse cross-pair data automatically. Building upon our data pipeline, we develop OmniInsert, a novel unified framework for mask-free video insertion from both single and multiple subject references. Specifically, to maintain subject-scene equilibrium, we introduce a simple yet effective Condition-Specific Feature Injection mechanism to distinctly inject multi-source conditions and propose a novel Progressive Training strategy that enables the model to balance feature injection from subjects and source video. Meanwhile, we design the Subject-Focused Loss to improve the detailed appearance of the subjects. To further enhance insertion harmonization, we propose an Insertive Preference Optimization methodology to optimize the model by simulating human preferences, and incorporate a Context-Aware Rephraser module during reference to seamlessly integrate the subject into the original scenes. To address the lack of a benchmark for the field, we introduce InsertBench, a comprehensive benchmark comprising diverse scenes with meticulously selected subjects. Evaluation on InsertBench indicates OmniInsert outperforms state-of-the-art closed-source commercial solutions. The code will be released.
>
---
#### [new 048] A Novel Metric for Detecting Memorization in Generative Models for Brain MRI Synthesis
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出DeepSSIM，一种用于检测生成模型在脑MRI合成中记忆训练数据的新度量方法。针对生成模型可能泄露敏感患者信息的问题，通过自监督学习和结构保持增强，实现了更准确的记忆检测，在实验中显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16582v1](http://arxiv.org/pdf/2509.16582v1)**

> **作者:** Antonio Scardace; Lemuel Puglisi; Francesco Guarnera; Sebastiano Battiato; Daniele Ravì
>
> **摘要:** Deep generative models have emerged as a transformative tool in medical imaging, offering substantial potential for synthetic data generation. However, recent empirical studies highlight a critical vulnerability: these models can memorize sensitive training data, posing significant risks of unauthorized patient information disclosure. Detecting memorization in generative models remains particularly challenging, necessitating scalable methods capable of identifying training data leakage across large sets of generated samples. In this work, we propose DeepSSIM, a novel self-supervised metric for quantifying memorization in generative models. DeepSSIM is trained to: i) project images into a learned embedding space and ii) force the cosine similarity between embeddings to match the ground-truth SSIM (Structural Similarity Index) scores computed in the image space. To capture domain-specific anatomical features, training incorporates structure-preserving augmentations, allowing DeepSSIM to estimate similarity reliably without requiring precise spatial alignment. We evaluate DeepSSIM in a case study involving synthetic brain MRI data generated by a Latent Diffusion Model (LDM) trained under memorization-prone conditions, using 2,195 MRI scans from two publicly available datasets (IXI and CoRR). Compared to state-of-the-art memorization metrics, DeepSSIM achieves superior performance, improving F1 scores by an average of +52.03% over the best existing method. Code and data of our approach are publicly available at the following link: https://github.com/brAIn-science/DeepSSIM.
>
---
#### [new 049] Development and validation of an AI foundation model for endoscopic diagnosis of esophagogastric junction adenocarcinoma: a cohort and deep learning study
- **分类: cs.CV**

- **简介: 该论文属于医学图像诊断任务，旨在解决食管胃结合部腺癌（EGJA）早期诊断依赖操作者经验的问题。研究开发了一个基于DINOv2和ResNet50的AI模型，利用内镜图像进行EGJA的筛查与分期诊断，并在多中心数据集上验证了其高准确性。**

- **链接: [http://arxiv.org/pdf/2509.17660v1](http://arxiv.org/pdf/2509.17660v1)**

> **作者:** Yikun Ma; Bo Li; Ying Chen; Zijie Yue; Shuchang Xu; Jingyao Li; Lei Ma; Liang Zhong; Duowu Zou; Leiming Xu; Yunshi Zhong; Xiaobo Li; Weiqun Ding; Minmin Zhang; Dongli He; Zhenghong Li; Ye Chen; Ye Zhao; Jialong Zhuo; Xiaofen Wu; Lisha Yi; Miaojing Shi; Huihui Sun
>
> **摘要:** The early detection of esophagogastric junction adenocarcinoma (EGJA) is crucial for improving patient prognosis, yet its current diagnosis is highly operator-dependent. This paper aims to make the first attempt to develop an artificial intelligence (AI) foundation model-based method for both screening and staging diagnosis of EGJA using endoscopic images. In this cohort and learning study, we conducted a multicentre study across seven Chinese hospitals between December 28, 2016 and December 30, 2024. It comprises 12,302 images from 1,546 patients; 8,249 of them were employed for model training, while the remaining were divided into the held-out (112 patients, 914 images), external (230 patients, 1,539 images), and prospective (198 patients, 1,600 images) test sets for evaluation. The proposed model employs DINOv2 (a vision foundation model) and ResNet50 (a convolutional neural network) to extract features of global appearance and local details of endoscopic images for EGJA staging diagnosis. Our model demonstrates satisfactory performance for EGJA staging diagnosis across three test sets, achieving an accuracy of 0.9256, 0.8895, and 0.8956, respectively. In contrast, among representative AI models, the best one (ResNet50) achieves an accuracy of 0.9125, 0.8382, and 0.8519 on the three test sets, respectively; the expert endoscopists achieve an accuracy of 0.8147 on the held-out test set. Moreover, with the assistance of our model, the overall accuracy for the trainee, competent, and expert endoscopists improves from 0.7035, 0.7350, and 0.8147 to 0.8497, 0.8521, and 0.8696, respectively. To our knowledge, our model is the first application of foundation models for EGJA staging diagnosis and demonstrates great potential in both diagnostic accuracy and efficiency.
>
---
#### [new 050] Revisiting Vision Language Foundations for No-Reference Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文研究无参考图像质量评估（NR-IQA）任务，探讨现代视觉基础模型在该任务中的表现。系统评估了六种预训练模型，并发现SigLIP2表现优异，同时提出可学习激活选择机制，提升了模型泛化能力与性能。**

- **链接: [http://arxiv.org/pdf/2509.17374v1](http://arxiv.org/pdf/2509.17374v1)**

> **作者:** Ankit Yadav; Ta Duc Huy; Lingqiao Liu
>
> **备注:** 23 pages, 16 figures
>
> **摘要:** Large-scale vision language pre-training has recently shown promise for no-reference image-quality assessment (NR-IQA), yet the relative merits of modern Vision Transformer foundations remain poorly understood. In this work, we present the first systematic evaluation of six prominent pretrained backbones, CLIP, SigLIP2, DINOv2, DINOv3, Perception, and ResNet, for the task of No-Reference Image Quality Assessment (NR-IQA), each finetuned using an identical lightweight MLP head. Our study uncovers two previously overlooked factors: (1) SigLIP2 consistently achieves strong performance; and (2) the choice of activation function plays a surprisingly crucial role, particularly for enhancing the generalization ability of image quality assessment models. Notably, we find that simple sigmoid activations outperform commonly used ReLU and GELU on several benchmarks. Motivated by this finding, we introduce a learnable activation selection mechanism that adaptively determines the nonlinearity for each channel, eliminating the need for manual activation design, and achieving new state-of-the-art SRCC on CLIVE, KADID10K, and AGIQA3K. Extensive ablations confirm the benefits across architectures and regimes, establishing strong, resource-efficient NR-IQA baselines.
>
---
#### [new 051] Optimal Transport for Handwritten Text Recognition in a Low-Resource Regime
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对手写文本识别（HTR）任务，旨在解决低资源场景下标注数据不足的问题。提出一种基于最优传输的迭代自举框架，通过对齐视觉特征与语义表示，提升低资源条件下的识别准确率。**

- **链接: [http://arxiv.org/pdf/2509.16977v1](http://arxiv.org/pdf/2509.16977v1)**

> **作者:** Petros Georgoulas Wraight; Giorgos Sfikas; Ioannis Kordonis; Petros Maragos; George Retsinas
>
> **摘要:** Handwritten Text Recognition (HTR) is a task of central importance in the field of document image understanding. State-of-the-art methods for HTR require the use of extensive annotated sets for training, making them impractical for low-resource domains like historical archives or limited-size modern collections. This paper introduces a novel framework that, unlike the standard HTR model paradigm, can leverage mild prior knowledge of lexical characteristics; this is ideal for scenarios where labeled data are scarce. We propose an iterative bootstrapping approach that aligns visual features extracted from unlabeled images with semantic word representations using Optimal Transport (OT). Starting with a minimal set of labeled examples, the framework iteratively matches word images to text labels, generates pseudo-labels for high-confidence alignments, and retrains the recognizer on the growing dataset. Numerical experiments demonstrate that our iterative visual-semantic alignment scheme significantly improves recognition accuracy on low-resource HTR benchmarks.
>
---
#### [new 052] Task-Oriented Communications for 3D Scene Representation: Balancing Timeliness and Fidelity
- **分类: cs.CV; cs.NI**

- **简介: 该论文研究实时3D场景表示中时延与质量的平衡问题，提出结合AoI和语义信息的PPO框架优化图像选择。实验验证了在动态环境中提升表示保真度并保持低延迟的效果。**

- **链接: [http://arxiv.org/pdf/2509.17282v1](http://arxiv.org/pdf/2509.17282v1)**

> **作者:** Xiangmin Xu; Zhen Meng; Kan Chen; Jiaming Yang; Emma Li; Philip G. Zhao; David Flynn
>
> **备注:** Submitted to IEEE Transactions on Mobile Computing
>
> **摘要:** Real-time Three-dimensional (3D) scene representation is a foundational element that supports a broad spectrum of cutting-edge applications, including digital manufacturing, Virtual, Augmented, and Mixed Reality (VR/AR/MR), and the emerging metaverse. Despite advancements in real-time communication and computing, achieving a balance between timeliness and fidelity in 3D scene representation remains a challenge. This work investigates a wireless network where multiple homogeneous mobile robots, equipped with cameras, capture an environment and transmit images to an edge server over channels for 3D representation. We propose a contextual-bandit Proximal Policy Optimization (PPO) framework incorporating both Age of Information (AoI) and semantic information to optimize image selection for representation, balancing data freshness and representation quality. Two policies -- the $\omega$-threshold and $\omega$-wait policies -- together with two benchmark methods are evaluated, timeliness embedding and weighted sum, on standard datasets and baseline 3D scene representation models. Experimental results demonstrate improved representation fidelity while maintaining low latency, offering insight into the model's decision-making process. This work advances real-time 3D scene representation by optimizing the trade-off between timeliness and fidelity in dynamic environments.
>
---
#### [new 053] Training-Free Label Space Alignment for Universal Domain Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究通用域适应（UniDA）任务，旨在解决源域与目标域标签空间差异及目标私有类问题。提出一种无需训练的标签空间对齐方法，利用CLIP等视觉-语言模型的零样本能力，过滤噪声标签并构建通用分类器，显著提升了跨域适应性能。**

- **链接: [http://arxiv.org/pdf/2509.17452v1](http://arxiv.org/pdf/2509.17452v1)**

> **作者:** Dujin Lee; Sojung An; Jungmyung Wi; Kuniaki Saito; Donghyun Kim
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** Universal domain adaptation (UniDA) transfers knowledge from a labeled source domain to an unlabeled target domain, where label spaces may differ and the target domain may contain private classes. Previous UniDA methods primarily focused on visual space alignment but often struggled with visual ambiguities due to content differences, which limited their robustness and generalizability. To overcome this, we introduce a novel approach that leverages the strong \textit{zero-shot capabilities} of recent vision-language foundation models (VLMs) like CLIP, concentrating solely on label space alignment to enhance adaptation stability. CLIP can generate task-specific classifiers based only on label names. However, adapting CLIP to UniDA is challenging because the label space is not fully known in advance. In this study, we first utilize generative vision-language models to identify unknown categories in the target domain. Noise and semantic ambiguities in the discovered labels -- such as those similar to source labels (e.g., synonyms, hypernyms, hyponyms) -- complicate label alignment. To address this, we propose a training-free label-space alignment method for UniDA (\ours). Our method aligns label spaces instead of visual spaces by filtering and refining noisy labels between the domains. We then construct a \textit{universal classifier} that integrates both shared knowledge and target-private class information, thereby improving generalizability under domain shifts. The results reveal that the proposed method considerably outperforms existing UniDA techniques across key DomainBed benchmarks, delivering an average improvement of \textcolor{blue}{+7.9\%}in H-score and \textcolor{blue}{+6.1\%} in H$^3$-score. Furthermore, incorporating self-training further enhances performance and achieves an additional (\textcolor{blue}{+1.6\%}) increment in both H- and H$^3$-scores.
>
---
#### [new 054] Optimized Learned Image Compression for Facial Expression Recognition
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于图像压缩与表情识别任务，旨在解决压缩导致特征退化和识别精度下降的问题。提出端到端模型及定制损失函数，优化压缩与识别性能的平衡。实验表明，联合优化显著提升了准确率与压缩效率。**

- **链接: [http://arxiv.org/pdf/2509.17262v1](http://arxiv.org/pdf/2509.17262v1)**

> **作者:** Xiumei Li; Marc Windsheimer; Misha Sadeghi; Björn Eskofier; André Kaup
>
> **备注:** Accepted at ICIP 2025
>
> **摘要:** Efficient data compression is crucial for the storage and transmission of visual data. However, in facial expression recognition (FER) tasks, lossy compression often leads to feature degradation and reduced accuracy. To address these challenges, this study proposes an end-to-end model designed to preserve critical features and enhance both compression and recognition performance. A custom loss function is introduced to optimize the model, tailored to balance compression and recognition performance effectively. This study also examines the influence of varying loss term weights on this balance. Experimental results indicate that fine-tuning the compression model alone improves classification accuracy by 0.71% and compression efficiency by 49.32%, while joint optimization achieves significant gains of 4.04% in accuracy and 89.12% in efficiency. Moreover, the findings demonstrate that the jointly optimized classification model maintains high accuracy on both compressed and uncompressed data, while the compression model reliably preserves image details, even at high compression rates.
>
---
#### [new 055] The 1st Solution for 7th LSVOS RVOS Track: SaSaSa2VA
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对RVOS任务，旨在通过自然语言指导视频中的目标分割与跟踪。基于Sa2VA，提出SaSaSa2VA方法，解决稀疏采样和单[SEG]令牌限制问题，采用分割增强与测试时集成策略，在7th LSVOS挑战赛中排名第一。**

- **链接: [http://arxiv.org/pdf/2509.16972v1](http://arxiv.org/pdf/2509.16972v1)**

> **作者:** Quanzhu Niu; Dengxian Gong; Shihao Chen; Tao Zhang; Yikang Zhou; Haobo Yuan; Lu Qi; Xiangtai Li; Shunping Ji
>
> **备注:** 1st place report of 7th LSVOS RVOS track in ICCV 2025. The code is released in Sa2VA repository: https://github.com/magic-research/Sa2VA
>
> **摘要:** Referring video object segmentation (RVOS) requires segmenting and tracking objects in videos conditioned on natural-language expressions, demanding fine-grained understanding of both appearance and motion. Building on Sa2VA, which couples a Multi-modal Large Language Model (MLLM) with the video segmentation model SAM2, we identify two key bottlenecks that limit segmentation performance: sparse frame sampling and reliance on a single [SEG] token for an entire video. We propose Segmentation Augmented and Selective Averaged Sa2VA SaSaSa2VA to address these issues. On the 7th LSVOS Challenge (RVOS track), SaSaSa2VA achieves a $J\&F$ of 67.45, ranking first and surpassing the runner-up by 2.80 points. This result and ablation studies demonstrate that efficient segmentation augmentation and test-time ensembling substantially enhance grounded MLLMs for RVOS. The code is released in Sa2VA repository: https://github.com/magic-research/Sa2VA.
>
---
#### [new 056] CSDformer: A Conversion Method for Fully Spike-Driven Transformer
- **分类: cs.CV**

- **简介: 该论文提出CSDformer，一种全脉冲驱动Transformer的转换方法。针对现有脉冲神经网络训练成本高、硬件不友好问题，设计NReLU函数和延迟积分-发放神经元，通过量化与时间分解技术实现高效转换，在ImageNet等数据集上取得优异性能，显著降低计算复杂度和训练开销。**

- **链接: [http://arxiv.org/pdf/2509.17461v1](http://arxiv.org/pdf/2509.17461v1)**

> **作者:** Yuhao Zhang; Chengjun Zhang; Di Wu; Jie Yang; Mohamad Sawan
>
> **摘要:** Spike-based transformer is a novel architecture aiming to enhance the performance of spiking neural networks while mitigating the energy overhead inherent to transformers. However, methods for generating these models suffer from critical limitations: excessive training costs introduced by direct training methods, or unavoidably hardware-unfriendly operations in existing conversion methods. In this paper, we propose CSDformer, a novel conversion method for fully spike-driven transformers. We tailor a conversion-oriented transformer-based architecture and propose a new function NReLU to replace softmax in self-attention. Subsequently, this model is quantized and trained, and converted into a fully spike-driven model with temporal decomposition technique. Also, we propose delayed Integrate-andFire neurons to reduce conversion errors and improve the performance of spiking models. We evaluate CSDformer on ImageNet, CIFAR-10 and CIFAR-100 datasets and achieve 76.36% top-1 accuracy under 7 time-steps on ImageNet, demonstrating superiority over state-of-the-art models. Furthermore, CSDformer eliminates the need for training SNNs, thereby reducing training costs (reducing computational resource by 75% and accelerating training speed by 2-3$\times$). To the best of our knowledge, this is the first fully spike-driven transformer-based model developed via conversion method, achieving high performance under ultra-low latency, while dramatically reducing both computational complexity and training overhead.
>
---
#### [new 057] AgriDoctor: A Multimodal Intelligent Assistant for Agriculture
- **分类: cs.CV**

- **简介: 该论文提出AgriDoctor，一个用于农业的多模态智能助手，旨在解决作物病害诊断和农业知识交互问题。通过构建包含图像、知识和提示的AgriMM基准数据集，并集成路由、分类、检测等模块，实现优于现有模型的农业任务性能。**

- **链接: [http://arxiv.org/pdf/2509.17044v1](http://arxiv.org/pdf/2509.17044v1)**

> **作者:** Mingqing Zhang; Zhuoning Xu; Peijie Wang; Rongji Li; Liang Wang; Qiang Liu; Jian Xu; Xuyao Zhang; Shu Wu; Liang Wang
>
> **摘要:** Accurate crop disease diagnosis is essential for sustainable agriculture and global food security. Existing methods, which primarily rely on unimodal models such as image-based classifiers and object detectors, are limited in their ability to incorporate domain-specific agricultural knowledge and lack support for interactive, language-based understanding. Recent advances in large language models (LLMs) and large vision-language models (LVLMs) have opened new avenues for multimodal reasoning. However, their performance in agricultural contexts remains limited due to the absence of specialized datasets and insufficient domain adaptation. In this work, we propose AgriDoctor, a modular and extensible multimodal framework designed for intelligent crop disease diagnosis and agricultural knowledge interaction. As a pioneering effort to introduce agent-based multimodal reasoning into the agricultural domain, AgriDoctor offers a novel paradigm for building interactive and domain-adaptive crop health solutions. It integrates five core components: a router, classifier, detector, knowledge retriever and LLMs. To facilitate effective training and evaluation, we construct AgriMM, a comprehensive benchmark comprising 400000 annotated disease images, 831 expert-curated knowledge entries, and 300000 bilingual prompts for intent-driven tool selection. Extensive experiments demonstrate that AgriDoctor, trained on AgriMM, significantly outperforms state-of-the-art LVLMs on fine-grained agricultural tasks, establishing a new paradigm for intelligent and sustainable farming applications.
>
---
#### [new 058] LLM-Assisted Semantic Guidance for Sparsely Annotated Remote Sensing Object Detection
- **分类: cs.CV**

- **简介: 该论文针对遥感目标检测中稀疏标注的问题，提出一种基于大语言模型（LLM）语义引导的框架。通过引入LLM生成的语义先验，设计了类感知伪标签分配机制和自适应难例重加权模块，有效提升了稀疏标注下的检测性能。**

- **链接: [http://arxiv.org/pdf/2509.16970v1](http://arxiv.org/pdf/2509.16970v1)**

> **作者:** Wei Liao; Chunyan Xu; Chenxu Wang; Zhen Cui
>
> **摘要:** Sparse annotation in remote sensing object detection poses significant challenges due to dense object distributions and category imbalances. Although existing Dense Pseudo-Label methods have demonstrated substantial potential in pseudo-labeling tasks, they remain constrained by selection ambiguities and inconsistencies in confidence estimation.In this paper, we introduce an LLM-assisted semantic guidance framework tailored for sparsely annotated remote sensing object detection, exploiting the advanced semantic reasoning capabilities of large language models (LLMs) to distill high-confidence pseudo-labels.By integrating LLM-generated semantic priors, we propose a Class-Aware Dense Pseudo-Label Assignment mechanism that adaptively assigns pseudo-labels for both unlabeled and sparsely labeled data, ensuring robust supervision across varying data distributions. Additionally, we develop an Adaptive Hard-Negative Reweighting Module to stabilize the supervised learning branch by mitigating the influence of confounding background information. Extensive experiments on DOTA and HRSC2016 demonstrate that the proposed method outperforms existing single-stage detector-based frameworks, significantly improving detection performance under sparse annotations.
>
---
#### [new 059] Can multimodal representation learning by alignment preserve modality-specific information?
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究多模态表示学习中对齐方法是否能保留模态特异性信息。针对遥感数据融合任务，分析了自监督对齐策略的信息丢失问题，并通过理论和实验验证，旨在改进多模态卫星数据的对比学习方法。**

- **链接: [http://arxiv.org/pdf/2509.17943v1](http://arxiv.org/pdf/2509.17943v1)**

> **作者:** Romain Thoreau; Jessie Levillain; Dawa Derksen
>
> **备注:** Accepted as a workshop paper at MACLEAN - ECML/PKDD 2025
>
> **摘要:** Combining multimodal data is a key issue in a wide range of machine learning tasks, including many remote sensing problems. In Earth observation, early multimodal data fusion methods were based on specific neural network architectures and supervised learning. Ever since, the scarcity of labeled data has motivated self-supervised learning techniques. State-of-the-art multimodal representation learning techniques leverage the spatial alignment between satellite data from different modalities acquired over the same geographic area in order to foster a semantic alignment in the latent space. In this paper, we investigate how this methods can preserve task-relevant information that is not shared across modalities. First, we show, under simplifying assumptions, when alignment strategies fundamentally lead to an information loss. Then, we support our theoretical insight through numerical experiments in more realistic settings. With those theoretical and empirical evidences, we hope to support new developments in contrastive learning for the combination of multimodal satellite data. Our code and data is publicly available at https://github.com/Romain3Ch216/alg_maclean_25.
>
---
#### [new 060] Seeing Culture: A Benchmark for Visual Reasoning and Grounding
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文提出了Seeing Culture Benchmark（SCB），用于评估视觉-语言模型在文化推理与定位任务中的能力。针对现有数据集文化代表性不足和缺乏深度推理的问题，SCB通过两阶段任务（多选VQA+分割）测试模型对东南亚138种文化元素的理解，包含1065张图像和3178个问题，推动跨模态文化理解研究。**

- **链接: [http://arxiv.org/pdf/2509.16517v1](http://arxiv.org/pdf/2509.16517v1)**

> **作者:** Burak Satar; Zhixin Ma; Patrick A. Irawan; Wilfried A. Mulyawan; Jing Jiang; Ee-Peng Lim; Chong-Wah Ngo
>
> **备注:** Accepted to EMNLP 2025 Main Conference, https://seeingculture-benchmark.github.io/
>
> **摘要:** Multimodal vision-language models (VLMs) have made substantial progress in various tasks that require a combined understanding of visual and textual content, particularly in cultural understanding tasks, with the emergence of new cultural datasets. However, these datasets frequently fall short of providing cultural reasoning while underrepresenting many cultures. In this paper, we introduce the Seeing Culture Benchmark (SCB), focusing on cultural reasoning with a novel approach that requires VLMs to reason on culturally rich images in two stages: i) selecting the correct visual option with multiple-choice visual question answering (VQA), and ii) segmenting the relevant cultural artifact as evidence of reasoning. Visual options in the first stage are systematically organized into three types: those originating from the same country, those from different countries, or a mixed group. Notably, all options are derived from a singular category for each type. Progression to the second stage occurs only after a correct visual option is chosen. The SCB benchmark comprises 1,065 images that capture 138 cultural artifacts across five categories from seven Southeast Asia countries, whose diverse cultures are often overlooked, accompanied by 3,178 questions, of which 1,093 are unique and meticulously curated by human annotators. Our evaluation of various VLMs reveals the complexities involved in cross-modal cultural reasoning and highlights the disparity between visual reasoning and spatial grounding in culturally nuanced scenarios. The SCB serves as a crucial benchmark for identifying these shortcomings, thereby guiding future developments in the field of cultural reasoning. https://github.com/buraksatar/SeeingCulture
>
---
#### [new 061] Prototype-Based Pseudo-Label Denoising for Source-Free Domain Adaptation in Remote Sensing Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文研究遥感图像语义分割中的源无关域适应（SFDA）任务，旨在解决目标域伪标签噪声问题。提出ProSFDA框架，通过原型加权伪标签和原型对比策略，提升自训练效果，有效缓解域偏移，实验表明方法优于现有技术。**

- **链接: [http://arxiv.org/pdf/2509.16942v1](http://arxiv.org/pdf/2509.16942v1)**

> **作者:** Bin Wang; Fei Deng; Zeyu Chen; Zhicheng Yu; Yiguang Liu
>
> **摘要:** Source-Free Domain Adaptation (SFDA) enables domain adaptation for semantic segmentation of Remote Sensing Images (RSIs) using only a well-trained source model and unlabeled target domain data. However, the lack of ground-truth labels in the target domain often leads to the generation of noisy pseudo-labels. Such noise impedes the effective mitigation of domain shift (DS). To address this challenge, we propose ProSFDA, a prototype-guided SFDA framework. It employs prototype-weighted pseudo-labels to facilitate reliable self-training (ST) under pseudo-labels noise. We, in addition, introduce a prototype-contrast strategy that encourages the aggregation of features belonging to the same class, enabling the model to learn discriminative target domain representations without relying on ground-truth supervision. Extensive experiments show that our approach substantially outperforms existing methods.
>
---
#### [new 062] MRN: Harnessing 2D Vision Foundation Models for Diagnosing Parkinson's Disease with Limited 3D MR Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MRN方法，利用2D视觉基础模型对有限的3D MR数据进行帕金森病诊断。通过多ROI处理与对比学习，有效提升分类性能，在MICCAI 2025挑战赛中取得86.0%准确率。**

- **链接: [http://arxiv.org/pdf/2509.17566v1](http://arxiv.org/pdf/2509.17566v1)**

> **作者:** Ding Shaodong; Liu Ziyang; Zhou Yijun; Liu Tao
>
> **备注:** First-place solution of the classification track for MICCAI'2025 PDCADxFoundation Challenge
>
> **摘要:** The automatic diagnosis of Parkinson's disease is in high clinical demand due to its prevalence and the importance of targeted treatment. Current clinical practice often relies on diagnostic biomarkers in QSM and NM-MRI images. However, the lack of large, high-quality datasets makes training diagnostic models from scratch prone to overfitting. Adapting pre-trained 3D medical models is also challenging, as the diversity of medical imaging leads to mismatches in voxel spacing and modality between pre-training and fine-tuning data. In this paper, we address these challenges by leveraging 2D vision foundation models (VFMs). Specifically, we crop multiple key ROIs from NM and QSM images, process each ROI through separate branches to compress the ROI into a token, and then combine these tokens into a unified patient representation for classification. Within each branch, we use 2D VFMs to encode axial slices of the 3D ROI volume and fuse them into the ROI token, guided by an auxiliary segmentation head that steers the feature extraction toward specific brain nuclei. Additionally, we introduce multi-ROI supervised contrastive learning, which improves diagnostic performance by pulling together representations of patients from the same class while pushing away those from different classes. Our approach achieved first place in the MICCAI 2025 PDCADxFoundation challenge, with an accuracy of 86.0% trained on a dataset of only 300 labeled QSM and NM-MRI scans, outperforming the second-place method by 5.5%.These results highlight the potential of 2D VFMs for clinical analysis of 3D MR images.
>
---
#### [new 063] Evaluation of Ensemble Learning Techniques for handwritten OCR Improvement
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于OCR改进任务，旨在提高手写历史病历的数字化准确性。研究通过集成学习方法结合OCR技术，验证了集成学习能提升识别精度，且训练数据量对结果影响不大。**

- **链接: [http://arxiv.org/pdf/2509.16221v1](http://arxiv.org/pdf/2509.16221v1)**

> **作者:** Martin Preiß
>
> **摘要:** For the bachelor project 2021 of Professor Lippert's research group, handwritten entries of historical patient records needed to be digitized using Optical Character Recognition (OCR) methods. Since the data will be used in the future, a high degree of accuracy is naturally required. Especially in the medical field this has even more importance. Ensemble Learning is a method that combines several machine learning models and is claimed to be able to achieve an increased accuracy for existing methods. For this reason, Ensemble Learning in combination with OCR is investigated in this work in order to create added value for the digitization of the patient records. It was possible to discover that ensemble learning can lead to an increased accuracy for OCR, which methods were able to achieve this and that the size of the training data set did not play a role here.
>
---
#### [new 064] Interpreting Attention Heads for Image-to-Text Information Flow in Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究视觉-语言模型（LVLMs）中图像到文本的信息流机制，属于视觉问答任务。针对信息流难以解释的问题，提出“头归因”方法，分析注意力头在信息传递中的作用，揭示语义内容驱动的关键注意力模式及信息流动路径。**

- **链接: [http://arxiv.org/pdf/2509.17588v1](http://arxiv.org/pdf/2509.17588v1)**

> **作者:** Jinyeong Kim; Seil Kang; Jiwoo Park; Junhyeok Kim; Seong Jae Hwang
>
> **摘要:** Large Vision-Language Models (LVLMs) answer visual questions by transferring information from images to text through a series of attention heads. While this image-to-text information flow is central to visual question answering, its underlying mechanism remains difficult to interpret due to the simultaneous operation of numerous attention heads. To address this challenge, we propose head attribution, a technique inspired by component attribution methods, to identify consistent patterns among attention heads that play a key role in information transfer. Using head attribution, we investigate how LVLMs rely on specific attention heads to identify and answer questions about the main object in an image. Our analysis reveals that a distinct subset of attention heads facilitates the image-to-text information flow. Remarkably, we find that the selection of these heads is governed by the semantic content of the input image rather than its visual appearance. We further examine the flow of information at the token level and discover that (1) text information first propagates to role-related tokens and the final token before receiving image information, and (2) image information is embedded in both object-related and background tokens. Our work provides evidence that image-to-text information flow follows a structured process, and that analysis at the attention-head level offers a promising direction toward understanding the mechanisms of LVLMs.
>
---
#### [new 065] Guided and Unguided Conditional Diffusion Mechanisms for Structured and Semantically-Aware 3D Point Cloud Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究3D点云生成任务，旨在解决现有方法在几何生成中语义信息不足的问题。提出基于扩散模型的框架，在生成过程中直接嵌入每一点的语义条件，实现几何与语义的联合生成，提升结构和语义准确性。**

- **链接: [http://arxiv.org/pdf/2509.17206v1](http://arxiv.org/pdf/2509.17206v1)**

> **作者:** Gunner Stone; Sushmita Sarker; Alireza Tavakkoli
>
> **摘要:** Generating realistic 3D point clouds is a fundamental problem in computer vision with applications in remote sensing, robotics, and digital object modeling. Existing generative approaches primarily capture geometry, and when semantics are considered, they are typically imposed post hoc through external segmentation or clustering rather than integrated into the generative process itself. We propose a diffusion-based framework that embeds per-point semantic conditioning directly within generation. Each point is associated with a conditional variable corresponding to its semantic label, which guides the diffusion dynamics and enables the joint synthesis of geometry and semantics. This design produces point clouds that are both structurally coherent and segmentation-aware, with object parts explicitly represented during synthesis. Through a comparative analysis of guided and unguided diffusion processes, we demonstrate the significant impact of conditional variables on diffusion dynamics and generation quality. Extensive experiments validate the efficacy of our approach, producing detailed and accurate 3D point clouds tailored to specific parts and features.
>
---
#### [new 066] COLA: Context-aware Language-driven Test-time Adaptation
- **分类: cs.CV**

- **简介: 该论文研究测试时适应（TTA）任务，旨在解决源域与目标域标签不共享的限制问题。提出COLA方法，通过引入上下文感知模块和类平衡伪标签策略，使预训练视觉-语言模型能高效适应多个目标域，提升跨域性能。**

- **链接: [http://arxiv.org/pdf/2509.17598v1](http://arxiv.org/pdf/2509.17598v1)**

> **作者:** Aiming Zhang; Tianyuan Yu; Liang Bai; Jun Tang; Yanming Guo; Yirun Ruan; Yun Zhou; Zhihe Lu
>
> **摘要:** Test-time adaptation (TTA) has gained increasing popularity due to its efficacy in addressing ``distribution shift'' issue while simultaneously protecting data privacy. However, most prior methods assume that a paired source domain model and target domain sharing the same label space coexist, heavily limiting their applicability. In this paper, we investigate a more general source model capable of adaptation to multiple target domains without needing shared labels. This is achieved by using a pre-trained vision-language model (VLM), \egno, CLIP, that can recognize images through matching with class descriptions. While the zero-shot performance of VLMs is impressive, they struggle to effectively capture the distinctive attributes of a target domain. To that end, we propose a novel method -- Context-aware Language-driven TTA (COLA). The proposed method incorporates a lightweight context-aware module that consists of three key components: a task-aware adapter, a context-aware unit, and a residual connection unit for exploring task-specific knowledge, domain-specific knowledge from the VLM and prior knowledge of the VLM, respectively. It is worth noting that the context-aware module can be seamlessly integrated into a frozen VLM, ensuring both minimal effort and parameter efficiency. Additionally, we introduce a Class-Balanced Pseudo-labeling (CBPL) strategy to mitigate the adverse effects caused by class imbalance. We demonstrate the effectiveness of our method not only in TTA scenarios but also in class generalisation tasks. The source code is available at https://github.com/NUDT-Bai-Group/COLA-TTA.
>
---
#### [new 067] CGTGait: Collaborative Graph and Transformer for Gait Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文提出CGTGait，用于步态情感识别任务。针对现有方法难以捕捉长时序特征的问题，结合图卷积与Transformer提取时空特征，并引入双向跨流融合模块，有效提升性能且降低计算复杂度约82.2%。**

- **链接: [http://arxiv.org/pdf/2509.16623v1](http://arxiv.org/pdf/2509.16623v1)**

> **作者:** Junjie Zhou; Haijun Xiong; Junhao Lu; Ziyu Lin; Bin Feng
>
> **备注:** Accepted by IJCB2025
>
> **摘要:** Skeleton-based gait emotion recognition has received significant attention due to its wide-ranging applications. However, existing methods primarily focus on extracting spatial and local temporal motion information, failing to capture long-range temporal representations. In this paper, we propose \textbf{CGTGait}, a novel framework that collaboratively integrates graph convolution and transformers to extract discriminative spatiotemporal features for gait emotion recognition. Specifically, CGTGait consists of multiple CGT blocks, where each block employs graph convolution to capture frame-level spatial topology and the transformer to model global temporal dependencies. Additionally, we introduce a Bidirectional Cross-Stream Fusion (BCSF) module to effectively aggregate posture and motion spatiotemporal features, facilitating the exchange of complementary information between the two streams. We evaluate our method on two widely used datasets, Emotion-Gait and ELMD, demonstrating that our CGTGait achieves state-of-the-art or at least competitive performance while reducing computational complexity by approximately \textbf{82.2\%} (only requiring 0.34G FLOPs) during testing. Code is available at \small{https://github.com/githubzjj1/CGTGait.}
>
---
#### [new 068] Eye Gaze Tells You Where to Compute: Gaze-Driven Efficient VLMs
- **分类: cs.CV**

- **简介: 该论文提出GazeVLM，一种无需训练的高效视觉-语言模型框架。通过利用人类眼动数据指导计算分配，减少冗余视觉token，提升推理效率。实验表明，在保持答案质量的同时，显著降低计算量和token数量，适用于AR/VR等边缘设备。**

- **链接: [http://arxiv.org/pdf/2509.16476v1](http://arxiv.org/pdf/2509.16476v1)**

> **作者:** Qinyu Chen; Jiawen Qi
>
> **备注:** 11 pages
>
> **摘要:** Vision-Language Models (VLMs) deliver impressive performance in understanding visual content with language instructions. However, redundancy in vision tokens results in the degenerated inference efficiency of VLMs, which hinders real-time use on edge consumer devices such as AR/VR devices. Existing efficiency methods commonly prune visual tokens using learned saliency, sparse attention schedules, or controller policies, but they often require architectural modification or access to intermediate activations. These pipelines add inference-time modules that increase compute and memory and often lead to an accuracy trade-off. Moreover, they also suffer from misalignment between the prompts and the region of interest in the images. Without human guidance, the model may focus on the wrong regions and miss small, high-frequency details when prompts or scenes change. In this paper, we propose GazeVLM, a training-free framework that uses the human eye gaze as a natural supervisory signal to allocate computation where it matters. By extracting gaze-driven regions of interest (ROIs) and optionally combining them with a low-resolution global view, GazeVLM mimics fovea-periphery perception to cut redundant visual tokens while preserving task-relevant details. We evaluate the visual question answering tasks on Qwen2.5-VL-3B/7B on the VOILA-COCO benchmark with human gaze. Quality of the answer is assessed by GPT-4o pairwise judging and a weighted score over coverage, accuracy, details, and fluency. Efficiency is measured by token counts and FLOPs. GazeVLM reduces visual tokens by up to 93.1%, total tokens by up to 59.6%, and FLOPs by 50%, while keeping better answer quality relative to full-resolution baselines. Our results show that aligning model computation with human gaze offers a simple, plug-and-play path toward efficient VLM inference on consumer devices.
>
---
#### [new 069] Min: Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对持续学习中的类别增量学习（CIL）任务，旨在解决预训练模型在适应新任务时的参数漂移问题。提出Min方法，通过信息论指导学习并混合有益噪声，抑制低相关特征，提升模型泛化能力，实验显示其在多个数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.16738v1](http://arxiv.org/pdf/2509.16738v1)**

> **作者:** Kai Jiang; Zhengyan Shi; Dell Zhang; Hongyuan Zhang; Xuelong Li
>
> **备注:** Accepted by NeurIPS 2025. Source Code will be released in the next version
>
> **摘要:** Class Incremental Learning (CIL) aims to continuously learn new categories while retaining the knowledge of old ones. Pre-trained models (PTMs) show promising capabilities in CIL. However, existing approaches that apply lightweight fine-tuning to backbones still induce parameter drift, thereby compromising the generalization capability of pre-trained models. Parameter drift can be conceptualized as a form of noise that obscures critical patterns learned for previous tasks. However, recent researches have shown that noise is not always harmful. For example, the large number of visual patterns learned from pre-training can be easily abused by a single task, and introducing appropriate noise can suppress some low-correlation features, thus leaving a margin for future tasks. To this end, we propose learning beneficial noise for CIL guided by information theory and propose Mixture of Noise (Min), aiming to mitigate the degradation of backbone generalization from adapting new tasks. Specifically, task-specific noise is learned from high-dimension features of new tasks. Then, a set of weights is adjusted dynamically for optimal mixture of different task noise. Finally, Min embeds the beneficial noise into the intermediate features to mask the response of inefficient patterns. Extensive experiments on six benchmark datasets demonstrate that Min achieves state-of-the-art performance in most incremental settings, with particularly outstanding results in 50-steps incremental settings. This shows the significant potential for beneficial noise in continual learning.
>
---
#### [new 070] Efficient 3D Scene Reconstruction and Simulation from Sparse Endoscopic Views
- **分类: cs.CV**

- **简介: 该论文提出基于高斯泼溅的框架，用于从稀疏内窥镜视图高效重建和模拟手术场景。针对视角受限导致的几何失真问题，引入虚拟相机正则化方法，并结合稀疏控制节点的物质点法实现快速物理仿真。**

- **链接: [http://arxiv.org/pdf/2509.17027v1](http://arxiv.org/pdf/2509.17027v1)**

> **作者:** Zhenya Yang
>
> **备注:** Workshop Paper of AECAI@MICCAI 2025
>
> **摘要:** Surgical simulation is essential for medical training, enabling practitioners to develop crucial skills in a risk-free environment while improving patient safety and surgical outcomes. However, conventional methods for building simulation environments are cumbersome, time-consuming, and difficult to scale, often resulting in poor details and unrealistic simulations. In this paper, we propose a Gaussian Splatting-based framework to directly reconstruct interactive surgical scenes from endoscopic data while ensuring efficiency, rendering quality, and realism. A key challenge in this data-driven simulation paradigm is the restricted movement of endoscopic cameras, which limits viewpoint diversity. As a result, the Gaussian Splatting representation overfits specific perspectives, leading to reduced geometric accuracy. To address this issue, we introduce a novel virtual camera-based regularization method that adaptively samples virtual viewpoints around the scene and incorporates them into the optimization process to mitigate overfitting. An effective depth-based regularization is applied to both real and virtual views to further refine the scene geometry. To enable fast deformation simulation, we propose a sparse control node-based Material Point Method, which integrates physical properties into the reconstructed scene while significantly reducing computational costs. Experimental results on representative surgical data demonstrate that our method can efficiently reconstruct and simulate surgical scenes from sparse endoscopic views. Notably, our method takes only a few minutes to reconstruct the surgical scene and is able to produce physically plausible deformations in real-time with user-defined interactions.
>
---
#### [new 071] MMPart: Harnessing Multi-Modal Large Language Models for Part-Aware 3D Generation
- **分类: cs.CV**

- **简介: 该论文提出MMPart，用于从单张图像生成结构感知的3D模型。针对现有方法缺乏部件控制与遮挡区域想象的问题，利用多模态大模型生成提示并指导生成过程，实现部件级建模与高质量重建。**

- **链接: [http://arxiv.org/pdf/2509.16768v1](http://arxiv.org/pdf/2509.16768v1)**

> **作者:** Omid Bonakdar; Nasser Mozayani
>
> **摘要:** Generative 3D modeling has advanced rapidly, driven by applications in VR/AR, metaverse, and robotics. However, most methods represent the target object as a closed mesh devoid of any structural information, limiting editing, animation, and semantic understanding. Part-aware 3D generation addresses this problem by decomposing objects into meaningful components, but existing pipelines face challenges: in existing methods, the user has no control over which objects are separated and how model imagine the occluded parts in isolation phase. In this paper, we introduce MMPart, an innovative framework for generating part-aware 3D models from a single image. We first use a VLM to generate a set of prompts based on the input image and user descriptions. In the next step, a generative model generates isolated images of each object based on the initial image and the previous step's prompts as supervisor (which control the pose and guide model how imagine previously occluded areas). Each of those images then enters the multi-view generation stage, where a number of consistent images from different views are generated. Finally, a reconstruction model converts each of these multi-view images into a 3D model.
>
---
#### [new 072] FROQ: Observing Face Recognition Models for Efficient Quality Assessment
- **分类: cs.CV**

- **简介: 该论文提出FROQ，一种无需训练的半监督人脸图像质量评估方法。针对传统方法依赖大量训练或效率低的问题，FROQ利用人脸识别模型中间表示并结合伪标签校准，实现了高效且性能优异的质量评估。**

- **链接: [http://arxiv.org/pdf/2509.17689v1](http://arxiv.org/pdf/2509.17689v1)**

> **作者:** Žiga Babnik; Deepak Kumar Jain; Peter Peer; Vitomir Štruc
>
> **备注:** Presented at the International Joint Conference on Biometrics (IJCB 2025)
>
> **摘要:** Face Recognition (FR) plays a crucial role in many critical (high-stakes) applications, where errors in the recognition process can lead to serious consequences. Face Image Quality Assessment (FIQA) techniques enhance FR systems by providing quality estimates of face samples, enabling the systems to discard samples that are unsuitable for reliable recognition or lead to low-confidence recognition decisions. Most state-of-the-art FIQA techniques rely on extensive supervised training to achieve accurate quality estimation. In contrast, unsupervised techniques eliminate the need for additional training but tend to be slower and typically exhibit lower performance. In this paper, we introduce FROQ (Face Recognition Observer of Quality), a semi-supervised, training-free approach that leverages specific intermediate representations within a given FR model to estimate face-image quality, and combines the efficiency of supervised FIQA models with the training-free approach of unsupervised methods. A simple calibration step based on pseudo-quality labels allows FROQ to uncover specific representations, useful for quality assessment, in any modern FR model. To generate these pseudo-labels, we propose a novel unsupervised FIQA technique based on sample perturbations. Comprehensive experiments with four state-of-the-art FR models and eight benchmark datasets show that FROQ leads to highly competitive results compared to the state-of-the-art, achieving both strong performance and efficient runtime, without requiring explicit training.
>
---
#### [new 073] RCTDistill: Cross-Modal Knowledge Distillation Framework for Radar-Camera 3D Object Detection with Temporal Fusion
- **分类: cs.CV**

- **简介: 该论文针对雷达-相机融合的3D目标检测任务，提出RCTDistill框架，通过跨模态知识蒸馏与时间融合，解决传感器误差和动态物体时序对齐问题，提升检测性能并实现高速推理。**

- **链接: [http://arxiv.org/pdf/2509.17712v1](http://arxiv.org/pdf/2509.17712v1)**

> **作者:** Geonho Bang; Minjae Seong; Jisong Kim; Geunju Baek; Daye Oh; Junhyung Kim; Junho Koh; Jun Won Choi
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Radar-camera fusion methods have emerged as a cost-effective approach for 3D object detection but still lag behind LiDAR-based methods in performance. Recent works have focused on employing temporal fusion and Knowledge Distillation (KD) strategies to overcome these limitations. However, existing approaches have not sufficiently accounted for uncertainties arising from object motion or sensor-specific errors inherent in radar and camera modalities. In this work, we propose RCTDistill, a novel cross-modal KD method based on temporal fusion, comprising three key modules: Range-Azimuth Knowledge Distillation (RAKD), Temporal Knowledge Distillation (TKD), and Region-Decoupled Knowledge Distillation (RDKD). RAKD is designed to consider the inherent errors in the range and azimuth directions, enabling effective knowledge transfer from LiDAR features to refine inaccurate BEV representations. TKD mitigates temporal misalignment caused by dynamic objects by aligning historical radar-camera BEV features with current LiDAR representations. RDKD enhances feature discrimination by distilling relational knowledge from the teacher model, allowing the student to differentiate foreground and background features. RCTDistill achieves state-of-the-art radar-camera fusion performance on both the nuScenes and View-of-Delft (VoD) datasets, with the fastest inference speed of 26.2 FPS.
>
---
#### [new 074] Leveraging RGB Images for Pre-Training of Event-Based Hand Pose Estimation
- **分类: cs.CV**

- **简介: 该论文提出RPEP，用于事件相机的手部姿态估计预训练。针对事件数据标注稀缺问题，利用RGB图像生成伪事件数据，并改进运动分解和反向约束策略，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.16949v1](http://arxiv.org/pdf/2509.16949v1)**

> **作者:** Ruicong Liu; Takehiko Ohkawa; Tze Ho Elden Tse; Mingfang Zhang; Angela Yao; Yoichi Sato
>
> **摘要:** This paper presents RPEP, the first pre-training method for event-based 3D hand pose estimation using labeled RGB images and unpaired, unlabeled event data. Event data offer significant benefits such as high temporal resolution and low latency, but their application to hand pose estimation is still limited by the scarcity of labeled training data. To address this, we repurpose real RGB datasets to train event-based estimators. This is done by constructing pseudo-event-RGB pairs, where event data is generated and aligned with the ground-truth poses of RGB images. Unfortunately, existing pseudo-event generation techniques assume stationary objects, thus struggling to handle non-stationary, dynamically moving hands. To overcome this, RPEP introduces a novel generation strategy that decomposes hand movements into smaller, step-by-step motions. This decomposition allows our method to capture temporal changes in articulation, constructing more realistic event data for a moving hand. Additionally, RPEP imposes a motion reversal constraint, regularizing event generation using reversed motion. Extensive experiments show that our pre-trained model significantly outperforms state-of-the-art methods on real event data, achieving up to 24% improvement on EvRealHands. Moreover, it delivers strong performance with minimal labeled samples for fine-tuning, making it well-suited for practical deployment.
>
---
#### [new 075] VaseVQA: Multimodal Agent and Benchmark for Ancient Greek Pottery
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VaseVL，一个针对古希腊陶器的多模态评估系统，旨在解决大模型在文化遗产领域缺乏专业推理能力的问题。通过构建VaseVQA基准和类型引导的奖励机制，提升了风格分类与历史归属的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.17191v1](http://arxiv.org/pdf/2509.17191v1)**

> **作者:** Jinchao Ge; Tengfei Cheng; Biao Wu; Zeyu Zhang; Shiya Huang; Judith Bishop; Gillian Shepherd; Meng Fang; Ling Chen; Yang Zhao
>
> **摘要:** Analyzing cultural-heritage artifacts remains challenging for MLLMs: general models lack domain expertise, and SFT often overfits superficial patterns, yielding brittle reasoning for authentication and historical attribution. This raises the question of how to equip MLLMs with robust, expert-level reasoning for ancient Greek pottery. We present VaseVL, an SFT-then-RL system that turns evaluation into supervision: we construct a taxonomy of question types, probe the SFT model to localize type-specific performance gaps, and optimize with type-conditioned, compositionality-oriented rewards targeting those gaps. We also release VaseVQA, a comprehensive benchmark of 31,773 images designed to probe deep understanding. Experiments show state-of-the-art results on style classification and historical attribution with marked gains in compositional robustness over SFT-only baselines, validating diagnosis-guided, taxonomy-conditioned reward engineering and providing a reusable resource for future research. Code and dataset will be available at https://github.com/AIGeeksGroup/VaseVQA.
>
---
#### [new 076] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EmbodiedSplat，旨在解决Embodied AI中Sim-to-Real导航的挑战。通过使用iPhone采集场景并结合3D Gaussian Splatting与Habitat-Sim，实现个性化策略训练，提升真实世界导航成功率。**

- **链接: [http://arxiv.org/pdf/2509.17430v1](http://arxiv.org/pdf/2509.17430v1)**

> **作者:** Gunjan Chhablani; Xiaomeng Ye; Muhammad Zubair Irshad; Zsolt Kira
>
> **备注:** 16 pages, 18 figures, paper accepted at ICCV, 2025
>
> **摘要:** The field of Embodied AI predominantly relies on simulation for training and evaluation, often using either fully synthetic environments that lack photorealism or high-fidelity real-world reconstructions captured with expensive hardware. As a result, sim-to-real transfer remains a major challenge. In this paper, we introduce EmbodiedSplat, a novel approach that personalizes policy training by efficiently capturing the deployment environment and fine-tuning policies within the reconstructed scenes. Our method leverages 3D Gaussian Splatting (GS) and the Habitat-Sim simulator to bridge the gap between realistic scene capture and effective training environments. Using iPhone-captured deployment scenes, we reconstruct meshes via GS, enabling training in settings that closely approximate real-world conditions. We conduct a comprehensive analysis of training strategies, pre-training datasets, and mesh reconstruction techniques, evaluating their impact on sim-to-real predictivity in real-world scenarios. Experimental results demonstrate that agents fine-tuned with EmbodiedSplat outperform both zero-shot baselines pre-trained on large-scale real-world datasets (HM3D) and synthetically generated datasets (HSSD), achieving absolute success rate improvements of 20\% and 40\% on real-world Image Navigation task. Moreover, our approach yields a high sim-vs-real correlation (0.87--0.97) for the reconstructed meshes, underscoring its effectiveness in adapting policies to diverse environments with minimal effort. Project page: https://gchhablani.github.io/embodied-splat
>
---
#### [new 077] SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出SmokeSeer，用于视频中烟雾去除与3D场景重建。利用RGB和热成像信息，基于3D高斯溅射技术，分解烟雾与非烟雾成分，解决真实场景中多密度、时变烟雾的处理问题。**

- **链接: [http://arxiv.org/pdf/2509.17329v1](http://arxiv.org/pdf/2509.17329v1)**

> **作者:** Neham Jain; Andrew Jong; Sebastian Scherer; Ioannis Gkioulekas
>
> **备注:** Project website: https://imaging.cs.cmu.edu/smokeseer
>
> **摘要:** Smoke in real-world scenes can severely degrade the quality of images and hamper visibility. Recent methods for image restoration either rely on data-driven priors that are susceptible to hallucinations, or are limited to static low-density smoke. We introduce SmokeSeer, a method for simultaneous 3D scene reconstruction and smoke removal from a video capturing multiple views of a scene. Our method uses thermal and RGB images, leveraging the fact that the reduced scattering in thermal images enables us to see through the smoke. We build upon 3D Gaussian splatting to fuse information from the two image modalities, and decompose the scene explicitly into smoke and non-smoke components. Unlike prior approaches, SmokeSeer handles a broad range of smoke densities and can adapt to temporally varying smoke. We validate our approach on synthetic data and introduce a real-world multi-view smoke dataset with RGB and thermal images. We provide open-source code and data at the project website.
>
---
#### [new 078] When Confidence Fails: Revisiting Pseudo-Label Selection in Semi-supervised Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对半监督语义分割中伪标签选择问题，提出置信度可分学习（CSL），通过凸优化方法建立样本特定决策边界，并引入随机掩码缓解低置信度区域信息丢失。实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16704v1](http://arxiv.org/pdf/2509.16704v1)**

> **作者:** Pan Liu; Jinshi Liu
>
> **摘要:** While significant advances exist in pseudo-label generation for semi-supervised semantic segmentation, pseudo-label selection remains understudied. Existing methods typically use fixed confidence thresholds to retain high-confidence predictions as pseudo-labels. However, these methods cannot cope with network overconfidence tendency, where correct and incorrect predictions overlap significantly in high-confidence regions, making separation challenging and amplifying model cognitive bias. Meanwhile, the direct discarding of low-confidence predictions disrupts spatial-semantic continuity, causing critical context loss. We propose Confidence Separable Learning (CSL) to address these limitations. CSL formulates pseudo-label selection as a convex optimization problem within the confidence distribution feature space, establishing sample-specific decision boundaries to distinguish reliable from unreliable predictions. Additionally, CSL introduces random masking of reliable pixels to guide the network in learning contextual relationships from low-reliability regions, thereby mitigating the adverse effects of discarding uncertain predictions. Extensive experimental results on the Pascal, Cityscapes, and COCO benchmarks show that CSL performs favorably against state-of-the-art methods. Code and model weights are available at https://github.com/PanLiuCSU/CSL.
>
---
#### [new 079] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SQS，一种用于自动驾驶的查询驱动稀疏感知模型预训练方法。通过基于查询的高斯表示和自监督投影学习，提升占用预测和3D目标检测性能，实验表明其在多个任务上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16588v1](http://arxiv.org/pdf/2509.16588v1)**

> **作者:** Haiming Zhang; Yiyao Zhu; Wending Zhou; Xu Yan; Yingjie Cai; Bingbing Liu; Shuguang Cui; Zhen Li
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** Sparse Perception Models (SPMs) adopt a query-driven paradigm that forgoes explicit dense BEV or volumetric construction, enabling highly efficient computation and accelerated inference. In this paper, we introduce SQS, a novel query-based splatting pre-training specifically designed to advance SPMs in autonomous driving. SQS introduces a plug-in module that predicts 3D Gaussian representations from sparse queries during pre-training, leveraging self-supervised splatting to learn fine-grained contextual features through the reconstruction of multi-view images and depth maps. During fine-tuning, the pre-trained Gaussian queries are seamlessly integrated into downstream networks via query interaction mechanisms that explicitly connect pre-trained queries with task-specific queries, effectively accommodating the diverse requirements of occupancy prediction and 3D object detection. Extensive experiments on autonomous driving benchmarks demonstrate that SQS delivers considerable performance gains across multiple query-based 3D perception tasks, notably in occupancy prediction and 3D object detection, outperforming prior state-of-the-art pre-training approaches by a significant margin (i.e., +1.3 mIoU on occupancy prediction and +1.0 NDS on 3D detection).
>
---
#### [new 080] A Cross-Hierarchical Multi-Feature Fusion Network Based on Multiscale Encoder-Decoder for Hyperspectral Change Detection
- **分类: cs.CV**

- **简介: 该论文针对高光谱变化检测任务，旨在解决现有方法多尺度特征利用不足和差分特征融合效率低的问题。提出一种基于多尺度编码-解码架构的跨层次多特征融合网络（CHMFFN），通过引入注意力模块和自适应融合机制，提升复杂变化的表征能力，实验验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2509.16988v1](http://arxiv.org/pdf/2509.16988v1)**

> **作者:** Mingshuai Sheng; Bhatti Uzair Aslam; Junfeng Zhang; Siling Feng; Yonis Gulzar
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Hyperspectral change detection (HCD) aims to accurately identify land-cover changes in hyperspectral images of the same area acquired at different times, with key applications in environmental monitoring and disaster assessment. To address limitations of existing methods, such as insufficient use of multiscale features and low efficiency in differential feature fusion, this paper proposes a cross-hierarchical multi-feature fusion network (CHMFFN) based on a multiscale encoder-decoder architecture. The front-end adopts a multiscale feature extraction subnetwork, built on an encoder-decoder backbone with residual connections and a dual-core channel-spatial attention (DCCSA) module to extract spectral-spatial-temporal features (SSTF). The encoder captures multiscale features from shallow details to deep semantics via residual blocks and convolutional kernels with varying receptive fields. The decoder restores spatial resolution and suppresses noise information through skip connections integrating encoder features. Additionally, a spectral-temporal change feature learning (STCFL) module learns cross-temporal change features at different levels, strengthening inter-temporal difference capture. An adaptive fusion of advanced features (AFAF) module dynamically balances hierarchical differential features via adaptive weights, enhancing representation of complex changes. Experiments on four public hyperspectral datasets show CHMFFN outperforms state-of-the-art methods, verifying its effectiveness.
>
---
#### [new 081] Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文提出Anytime Person Re-Identification（AT-ReID）任务，旨在解决现有ReID在不同时间场景下检索效果不足的问题。作者构建了首个大规模AT-USTC数据集，并设计了统一模型Uni-AT，包含多场景特征学习、属性专家模块和动态权重策略，提升了跨场景的检索性能。**

- **链接: [http://arxiv.org/pdf/2509.16635v1](http://arxiv.org/pdf/2509.16635v1)**

> **作者:** Xulin Li; Yan Lu; Bin Liu; Jiaze Li; Qinhong Yang; Tao Gong; Qi Chu; Mang Ye; Nenghai Yu
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** In real applications, person re-identification (ReID) is expected to retrieve the target person at any time, including both daytime and nighttime, ranging from short-term to long-term. However, existing ReID tasks and datasets can not meet this requirement, as they are constrained by available time and only provide training and evaluation for specific scenarios. Therefore, we investigate a new task called Anytime Person Re-identification (AT-ReID), which aims to achieve effective retrieval in multiple scenarios based on variations in time. To address the AT-ReID problem, we collect the first large-scale dataset, AT-USTC, which contains 403k images of individuals wearing multiple clothes captured by RGB and IR cameras. Our data collection spans 21 months, and 270 volunteers were photographed on average 29.1 times across different dates or scenes, 4-15 times more than current datasets, providing conditions for follow-up investigations in AT-ReID. Further, to tackle the new challenge of multi-scenario retrieval, we propose a unified model named Uni-AT, which comprises a multi-scenario ReID (MS-ReID) framework for scenario-specific features learning, a Mixture-of-Attribute-Experts (MoAE) module to alleviate inter-scenario interference, and a Hierarchical Dynamic Weighting (HDW) strategy to ensure balanced training across all scenarios. Extensive experiments show that our model leads to satisfactory results and exhibits excellent generalization to all scenarios.
>
---
#### [new 082] Single-Image Depth from Defocus with Coded Aperture and Diffusion Posterior Sampling
- **分类: cs.CV**

- **简介: 该论文提出一种基于编码孔径的单幅图像去模糊深度估计方法，利用扩散先验替代手工设计的先验进行正则化。通过可微前向模型和扩散后验采样，实现了无需配对训练数据的鲁棒RGBD重建，优于传统优化和U-Net基线方法。**

- **链接: [http://arxiv.org/pdf/2509.17427v1](http://arxiv.org/pdf/2509.17427v1)**

> **作者:** Hodaka Kawachi; Jose Reinaldo Cunha Santos A. V. Silva Neto; Yasushi Yagi; Hajime Nagahara; Tomoya Nakamura
>
> **摘要:** We propose a single-snapshot depth-from-defocus (DFD) reconstruction method for coded-aperture imaging that replaces hand-crafted priors with a learned diffusion prior used purely as regularization. Our optimization framework enforces measurement consistency via a differentiable forward model while guiding solutions with the diffusion prior in the denoised image domain, yielding higher accuracy and stability than clas- sical optimization. Unlike U-Net-style regressors, our approach requires no paired defocus-RGBD training data and does not tie training to a specific camera configuration. Experiments on comprehensive simulations and a prototype camera demonstrate consistently strong RGBD reconstructions across noise levels, outperforming both U-Net baselines and a classical coded- aperture DFD method.
>
---
#### [new 083] DragOSM: Extract Building Roofs and Footprints from Aerial Images by Aligning Historical Labels
- **分类: cs.CV; I.5.4**

- **简介: 该论文提出DragOSM方法，用于从倾斜航拍图像中对齐历史标注（如OpenStreetMap）以提取建筑物屋顶和轮廓。针对历史标注位置偏差及标注不全的问题，引入对齐标记并建模为去噪过程，通过迭代优化提升标注精度，并构建ReBO数据集验证效果。**

- **链接: [http://arxiv.org/pdf/2509.17951v1](http://arxiv.org/pdf/2509.17951v1)**

> **作者:** Kai Li; Xingxing Weng; Yupeng Deng; Yu Meng; Chao Pang; Gui-Song Xia; Xiangyu Zhao
>
> **备注:** 17 Pages
>
> **摘要:** Extracting polygonal roofs and footprints from remote sensing images is critical for large-scale urban analysis. Most existing methods rely on segmentation-based models that assume clear semantic boundaries of roofs, but these approaches struggle in off- nadir images, where the roof and footprint are significantly displaced, and facade pixels are fused with the roof boundary. With the increasing availability of open vector map annotations, e.g., OpenStreetMap, utilizing historical labels for off-nadir image annotation has become viable because remote sensing images are georeferenced once captured. However, these historical labels commonly suffer from significant positional discrepancies with new images and only have one annotation (roof or footprint), which fails to describe the correct structures of a building. To address these discrepancies, we first introduce a concept of an alignment token, which encodes the correction vector to guide the label correction. Based on this concept, we then propose Drag OpenStreetMap Labels (DragOSM), a novel model designed to align dislocated historical labels with roofs and footprints. Specifically, DragOSM formulates the label alignment as an interactive denoising process, modeling the positional discrepancy as a Gaussian distribution. During training, it learns to correct these errors by simulating misalignment with random Gaussian perturbations; during inference, it iteratively refines the positions of input labels. To validate our method, we further present a new dataset, Repairing Buildings in OSM (ReBO), comprising 179,265 buildings with both OpenStreetMap and manually corrected annotations across 5,473 images from 41 cities. Experimental results on ReBO demonstrate the effectiveness of DragOSM. Code, dataset, and trained models are publicly available at https://github.com/likaiucas/DragOSM.git.
>
---
#### [new 084] SmaRT: Style-Modulated Robust Test-Time Adaptation for Cross-Domain Brain Tumor Segmentation in MRI
- **分类: cs.CV**

- **简介: 该论文提出SmaRT框架，用于MRI脑肿瘤分割任务，旨在解决跨域场景下因设备、协议和人群差异导致的模型性能下降问题。通过风格调制增强、双分支动量策略和结构先验，实现无源域的鲁棒测试时自适应，提升分割精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.17925v1](http://arxiv.org/pdf/2509.17925v1)**

> **作者:** Yuanhan Wang; Yifei Chen; Shuo Jiang; Wenjing Yu; Mingxuan Liu; Beining Wu; Jinying Zong; Feiwei Qin; Changmiao Wang; Qiyuan Tian
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Reliable brain tumor segmentation in MRI is indispensable for treatment planning and outcome monitoring, yet models trained on curated benchmarks often fail under domain shifts arising from scanner and protocol variability as well as population heterogeneity. Such gaps are especially severe in low-resource and pediatric cohorts, where conventional test-time or source-free adaptation strategies often suffer from instability and structural inconsistency. We propose SmaRT, a style-modulated robust test-time adaptation framework that enables source-free cross-domain generalization. SmaRT integrates style-aware augmentation to mitigate appearance discrepancies, a dual-branch momentum strategy for stable pseudo-label refinement, and structural priors enforcing consistency, integrity, and connectivity. This synergy ensures both adaptation stability and anatomical fidelity under extreme domain shifts. Extensive evaluations on sub-Saharan Africa and pediatric glioma datasets show that SmaRT consistently outperforms state-of-the-art methods, with notable gains in Dice accuracy and boundary precision. Overall, SmaRT bridges the gap between algorithmic advances and equitable clinical applicability, supporting robust deployment of MRI-based neuro-oncology tools in diverse clinical environments. Our source code is available at https://github.com/baiyou1234/SmaRT.
>
---
#### [new 085] PRNU-Bench: A Novel Benchmark and Model for PRNU-Based Camera Identification
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文提出PRNU-Bench，用于基于PRNU的相机识别任务。针对现实场景下的相机识别问题，构建了一个包含13K张图片的新基准数据集，并设计了一种结合去噪自编码器和卷积网络的混合模型，通过Hadamard乘积实现更优的1:N相机验证。**

- **链接: [http://arxiv.org/pdf/2509.17581v1](http://arxiv.org/pdf/2509.17581v1)**

> **作者:** Florinel Alin Croitoru; Vlad Hondru; Radu Tudor Ionescu
>
> **摘要:** We propose a novel benchmark for camera identification via Photo Response Non-Uniformity (PRNU) estimation. The benchmark comprises 13K photos taken with 120+ cameras, where the training and test photos are taken in different scenarios, enabling ``in-the-wild'' evaluation. In addition, we propose a novel PRNU-based camera identification model that employs a hybrid architecture, comprising a denoising autoencoder to estimate the PRNU signal and a convolutional network that can perform 1:N verification of camera devices. Instead of using a conventional approach based on contrastive learning, our method takes the Hadamard product between reference and query PRNU signals as input. This novel design leads to significantly better results compared with state-of-the-art models based on denoising autoencoders and contrastive learning. We release our dataset and code at: https://github.com/CroitoruAlin/PRNU-Bench.
>
---
#### [new 086] GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出GeoSVR，一种基于稀疏体素的显式框架，用于几何精确的表面重建。针对现有方法在表示能力上的瓶颈，GeoSVR通过体素不确定性深度约束和稀疏体素表面正则化，提升了重建的准确性、细节保留和完整性。**

- **链接: [http://arxiv.org/pdf/2509.18090v1](http://arxiv.org/pdf/2509.18090v1)**

> **作者:** Jiahe Li; Jiawei Zhang; Youmin Zhang; Xiao Bai; Jin Zheng; Xiaohan Yu; Lin Gu
>
> **备注:** Accepted at NeurIPS 2025 (Spotlight). Project page: https://fictionarry.github.io/GeoSVR-project/
>
> **摘要:** Reconstructing accurate surfaces with radiance fields has achieved remarkable progress in recent years. However, prevailing approaches, primarily based on Gaussian Splatting, are increasingly constrained by representational bottlenecks. In this paper, we introduce GeoSVR, an explicit voxel-based framework that explores and extends the under-investigated potential of sparse voxels for achieving accurate, detailed, and complete surface reconstruction. As strengths, sparse voxels support preserving the coverage completeness and geometric clarity, while corresponding challenges also arise from absent scene constraints and locality in surface refinement. To ensure correct scene convergence, we first propose a Voxel-Uncertainty Depth Constraint that maximizes the effect of monocular depth cues while presenting a voxel-oriented uncertainty to avoid quality degradation, enabling effective and robust scene constraints yet preserving highly accurate geometries. Subsequently, Sparse Voxel Surface Regularization is designed to enhance geometric consistency for tiny voxels and facilitate the voxel-based formation of sharp and accurate surfaces. Extensive experiments demonstrate our superior performance compared to existing methods across diverse challenging scenarios, excelling in geometric accuracy, detail preservation, and reconstruction completeness while maintaining high efficiency. Code is available at https://github.com/Fictionarry/GeoSVR.
>
---
#### [new 087] VidCLearn: A Continual Learning Approach for Text-to-Video Generation
- **分类: cs.CV**

- **简介: 该论文提出VidCLearn，一种面向文本到视频生成的持续学习框架。针对现有模型难以增量学习新数据的问题，设计了师生架构与生成回放机制，并引入时序一致性损失和视频检索模块，以提升生成效果与计算效率。**

- **链接: [http://arxiv.org/pdf/2509.16956v1](http://arxiv.org/pdf/2509.16956v1)**

> **作者:** Luca Zanchetta; Lorenzo Papa; Luca Maiano; Irene Amerini
>
> **摘要:** Text-to-video generation is an emerging field in generative AI, enabling the creation of realistic, semantically accurate videos from text prompts. While current models achieve impressive visual quality and alignment with input text, they typically rely on static knowledge, making it difficult to incorporate new data without retraining from scratch. To address this limitation, we propose VidCLearn, a continual learning framework for diffusion-based text-to-video generation. VidCLearn features a student-teacher architecture where the student model is incrementally updated with new text-video pairs, and the teacher model helps preserve previously learned knowledge through generative replay. Additionally, we introduce a novel temporal consistency loss to enhance motion smoothness and a video retrieval module to provide structural guidance at inference. Our architecture is also designed to be more computationally efficient than existing models while retaining satisfactory generation performance. Experimental results show VidCLearn's superiority over baseline methods in terms of visual quality, semantic alignment, and temporal coherence.
>
---
#### [new 088] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception
- **分类: cs.CV**

- **简介: 该论文提出MAESTRO，用于多任务3D感知（包括目标检测、BEV地图分割和3D占用预测）。针对任务冲突导致性能下降的问题，设计了CPG、TSFG和SPA三个模块，通过自适应特征增强与抑制提升各任务性能。**

- **链接: [http://arxiv.org/pdf/2509.17462v1](http://arxiv.org/pdf/2509.17462v1)**

> **作者:** Changwon Kang; Jisong Kim; Hongjae Shin; Junseo Park; Jun Won Choi
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** The goal of multi-task learning is to learn to conduct multiple tasks simultaneously based on a shared data representation. While this approach can improve learning efficiency, it may also cause performance degradation due to task conflicts that arise when optimizing the model for different objectives. To address this challenge, we introduce MAESTRO, a structured framework designed to generate task-specific features and mitigate feature interference in multi-task 3D perception, including 3D object detection, bird's-eye view (BEV) map segmentation, and 3D occupancy prediction. MAESTRO comprises three components: the Class-wise Prototype Generator (CPG), the Task-Specific Feature Generator (TSFG), and the Scene Prototype Aggregator (SPA). CPG groups class categories into foreground and background groups and generates group-wise prototypes. The foreground and background prototypes are assigned to the 3D object detection task and the map segmentation task, respectively, while both are assigned to the 3D occupancy prediction task. TSFG leverages these prototype groups to retain task-relevant features while suppressing irrelevant features, thereby enhancing the performance for each task. SPA enhances the prototype groups assigned for 3D occupancy prediction by utilizing the information produced by the 3D object detection head and the map segmentation head. Extensive experiments on the nuScenes and Occ3D benchmarks demonstrate that MAESTRO consistently outperforms existing methods across 3D object detection, BEV map segmentation, and 3D occupancy prediction tasks.
>
---
#### [new 089] Multi-Agent Amodal Completion: Direct Synthesis with Fine-Grained Semantic Guidance
- **分类: cs.CV; cs.MA**

- **简介: 该论文聚焦于多智能体非模态补全任务，旨在解决遮挡物体不可见部分生成中的数据依赖、泛化性和误差累积问题。提出协作式多智能体推理框架，结合精细语义引导，直接生成高质量RGBA输出，实现更准确的图像补全。**

- **链接: [http://arxiv.org/pdf/2509.17757v1](http://arxiv.org/pdf/2509.17757v1)**

> **作者:** Hongxing Fan; Lipeng Wang; Haohua Chen; Zehuan Huang; Jiangtao Wu; Lu Sheng
>
> **摘要:** Amodal completion, generating invisible parts of occluded objects, is vital for applications like image editing and AR. Prior methods face challenges with data needs, generalization, or error accumulation in progressive pipelines. We propose a Collaborative Multi-Agent Reasoning Framework based on upfront collaborative reasoning to overcome these issues. Our framework uses multiple agents to collaboratively analyze occlusion relationships and determine necessary boundary expansion, yielding a precise mask for inpainting. Concurrently, an agent generates fine-grained textual descriptions, enabling Fine-Grained Semantic Guidance. This ensures accurate object synthesis and prevents the regeneration of occluders or other unwanted elements, especially within large inpainting areas. Furthermore, our method directly produces layered RGBA outputs guided by visible masks and attention maps from a Diffusion Transformer, eliminating extra segmentation. Extensive evaluations demonstrate our framework achieves state-of-the-art visual quality.
>
---
#### [new 090] Enhanced Detection of Tiny Objects in Aerial Images
- **分类: cs.CV**

- **简介: 该论文针对航拍图像中微小目标检测效果差的问题，提出三种增强策略：调整输入分辨率、数据增强和引入注意力机制，并设计了MoonNet网络，提升了YOLOv8的检测精度。**

- **链接: [http://arxiv.org/pdf/2509.17078v1](http://arxiv.org/pdf/2509.17078v1)**

> **作者:** Kihyun Kim; Michalis Lazarou; Tania Stathaki
>
> **摘要:** While one-stage detectors like YOLOv8 offer fast training speed, they often under-perform on detecting small objects as a trade-off. This becomes even more critical when detecting tiny objects in aerial imagery due to low-resolution targets and cluttered backgrounds. To address this, we introduce three enhancement strategies -- input image resolution adjustment, data augmentation, and attention mechanisms -- that can be easily implemented on YOLOv8. We demonstrate that image size enlargement and the proper use of augmentation can lead to enhancement. Additionally, we designed a Mixture of Orthogonal Neural-modules Network (MoonNet) pipeline which consists of attention-augmented CNNs. Two well-known attention modules, the Squeeze-and-Excitation Block (SE Block) and the Convolutional Block Attention Module (CBAM), were integrated into the backbone of YOLOv8 with an increased number of channels, and the MoonNet backbone obtained improved detection accuracy compared to the original YOLOv8. MoonNet further proved its adaptability and potential by achieving state-of-the-art performance on a tiny-object benchmark when integrated with the YOLC model. Our codes are available at: https://github.com/Kihyun11/MoonNet
>
---
#### [new 091] ProDyG: Progressive Dynamic Scene Reconstruction via Gaussian Splatting from Monocular Videos
- **分类: cs.CV**

- **简介: 该论文提出ProDyG，用于在线动态场景重建任务。针对现有SLAM方法处理动态部分不足、依赖RGB-D输入及缺乏全局一致性的问题，通过分离静态与动态部分，结合运动掩码和渐进式Motion Scaffolds图，实现了高细节的实时动态三维重建。**

- **链接: [http://arxiv.org/pdf/2509.17864v1](http://arxiv.org/pdf/2509.17864v1)**

> **作者:** Shi Chen; Erik Sandström; Sandro Lombardi; Siyuan Li; Martin R. Oswald
>
> **摘要:** Achieving truly practical dynamic 3D reconstruction requires online operation, global pose and map consistency, detailed appearance modeling, and the flexibility to handle both RGB and RGB-D inputs. However, existing SLAM methods typically merely remove the dynamic parts or require RGB-D input, while offline methods are not scalable to long video sequences, and current transformer-based feedforward methods lack global consistency and appearance details. To this end, we achieve online dynamic scene reconstruction by disentangling the static and dynamic parts within a SLAM system. The poses are tracked robustly with a novel motion masking strategy, and dynamic parts are reconstructed leveraging a progressive adaptation of a Motion Scaffolds graph. Our method yields novel view renderings competitive to offline methods and achieves on-par tracking with state-of-the-art dynamic SLAM methods.
>
---
#### [new 092] Person Identification from Egocentric Human-Object Interactions using 3D Hand Pose
- **分类: cs.CV; cs.ET; cs.HC; cs.LG**

- **简介: 该论文提出I2S框架，通过分析第一视角视频中的3D手部姿态，实现基于人-物交互的用户身份识别。属于AR场景下的身份认证任务，旨在提升个性化辅助技术的安全性与实时性。工作包括特征提取、多阶段识别流程设计及轻量化模型优化。**

- **链接: [http://arxiv.org/pdf/2509.16557v1](http://arxiv.org/pdf/2509.16557v1)**

> **作者:** Muhammad Hamza; Danish Hamid; Muhammad Tahir Akram
>
> **备注:** 21 pages, 8 figures, 7 tables. Preprint of a manuscript submitted to CCF Transactions on Pervasive Computing and Interaction (Springer), currently under review
>
> **摘要:** Human-Object Interaction Recognition (HOIR) and user identification play a crucial role in advancing augmented reality (AR)-based personalized assistive technologies. These systems are increasingly being deployed in high-stakes, human-centric environments such as aircraft cockpits, aerospace maintenance, and surgical procedures. This research introduces I2S (Interact2Sign), a multi stage framework designed for unobtrusive user identification through human object interaction recognition, leveraging 3D hand pose analysis in egocentric videos. I2S utilizes handcrafted features extracted from 3D hand poses and per forms sequential feature augmentation: first identifying the object class, followed by HOI recognition, and ultimately, user identification. A comprehensive feature extraction and description process was carried out for 3D hand poses, organizing the extracted features into semantically meaningful categories: Spatial, Frequency, Kinematic, Orientation, and a novel descriptor introduced in this work, the Inter-Hand Spatial Envelope (IHSE). Extensive ablation studies were conducted to determine the most effective combination of features. The optimal configuration achieved an impressive average F1-score of 97.52% for user identification, evaluated on a bimanual object manipulation dataset derived from the ARCTIC and H2O datasets. I2S demonstrates state-of-the-art performance while maintaining a lightweight model size of under 4 MB and a fast inference time of 0.1 seconds. These characteristics make the proposed framework highly suitable for real-time, on-device authentication in security-critical, AR-based systems.
>
---
#### [new 093] I2VWM: Robust Watermarking for Image to Video Generation
- **分类: cs.CV**

- **简介: 该论文提出I2VWM，一种针对图像到视频生成的跨模态数字水印框架。旨在解决现有方法在追踪源图像上的不足，通过引入鲁棒扩散距离和优化对齐模块，提升水印在时间维度上的鲁棒性与不可感知性。**

- **链接: [http://arxiv.org/pdf/2509.17773v1](http://arxiv.org/pdf/2509.17773v1)**

> **作者:** Guanjie Wang; Zehua Ma; Han Fang; Weiming Zhang
>
> **备注:** 10 pages
>
> **摘要:** The rapid progress of image-guided video generation (I2V) has raised concerns about its potential misuse in misinformation and fraud, underscoring the urgent need for effective digital watermarking. While existing watermarking methods demonstrate robustness within a single modality, they fail to trace source images in I2V settings. To address this gap, we introduce the concept of Robust Diffusion Distance, which measures the temporal persistence of watermark signals in generated videos. Building on this, we propose I2VWM, a cross-modal watermarking framework designed to enhance watermark robustness across time. I2VWM leverages a video-simulation noise layer during training and employs an optical-flow-based alignment module during inference. Experiments on both open-source and commercial I2V models demonstrate that I2VWM significantly improves robustness while maintaining imperceptibility, establishing a new paradigm for cross-modal watermarking in the era of generative video. \href{https://github.com/MrCrims/I2VWM-Robust-Watermarking-for-Image-to-Video-Generation}{Code Released.}
>
---
#### [new 094] When Color-Space Decoupling Meets Diffusion for Adverse-Weather Image Restoration
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对恶劣天气图像修复任务，提出LCDiff框架，通过解耦亮度与色度信息，并结合引导扩散模型，提升了修复效果和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.17024v1](http://arxiv.org/pdf/2509.17024v1)**

> **作者:** Wenxuan Fang; Jili Fan; Chao Wang; Xiantao Hu; Jiangwei Weng; Ying Tai; Jian Yang; Jun Li
>
> **摘要:** Adverse Weather Image Restoration (AWIR) is a highly challenging task due to the unpredictable and dynamic nature of weather-related degradations. Traditional task-specific methods often fail to generalize to unseen or complex degradation types, while recent prompt-learning approaches depend heavily on the degradation estimation capabilities of vision-language models, resulting in inconsistent restorations. In this paper, we propose \textbf{LCDiff}, a novel framework comprising two key components: \textit{Lumina-Chroma Decomposition Network} (LCDN) and \textit{Lumina-Guided Diffusion Model} (LGDM). LCDN processes degraded images in the YCbCr color space, separately handling degradation-related luminance and degradation-invariant chrominance components. This decomposition effectively mitigates weather-induced degradation while preserving color fidelity. To further enhance restoration quality, LGDM leverages degradation-related luminance information as a guiding condition, eliminating the need for explicit degradation prompts. Additionally, LGDM incorporates a \textit{Dynamic Time Step Loss} to optimize the denoising network, ensuring a balanced recovery of both low- and high-frequency features in the image. Finally, we present DriveWeather, a comprehensive all-weather driving dataset designed to enable robust evaluation. Extensive experiments demonstrate that our approach surpasses state-of-the-art methods, setting a new benchmark in AWIR. The dataset and code are available at: https://github.com/fiwy0527/LCDiff.
>
---
#### [new 095] Are VLMs Ready for Lane Topology Awareness in Autonomous Driving?
- **分类: cs.CV**

- **简介: 该论文评估视觉语言模型（VLMs）在自动驾驶中道路拓扑理解的能力，设计了四个基于鸟瞰图的拓扑诊断任务。研究发现，尽管大模型表现较好，但在时序推理上仍有不足，表明空间推理仍是瓶颈，并指出模型规模和示例数量对性能有正向影响。**

- **链接: [http://arxiv.org/pdf/2509.16654v1](http://arxiv.org/pdf/2509.16654v1)**

> **作者:** Xin Chen; Jia He; Maozheng Li; Dongliang Xu; Tianyu Wang; Yixiao Chen; Zhixin Lin; Yue Yao
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Vision-Language Models (VLMs) have recently shown remarkable progress in multimodal reasoning, yet their applications in autonomous driving remain limited. In particular, the ability to understand road topology, a key requirement for safe navigation, has received relatively little attention. While some recent works have begun to explore VLMs in driving contexts, their performance on topology reasoning is far from satisfactory. In this work, we systematically evaluate VLMs' capabilities in road topology understanding. Specifically, multi-view images are projected into unified ground-plane coordinate system and fused into bird's-eye-view (BEV) lanes. Based on these BEV lanes, we formulate four topology-related diagnostic VQA tasks, which together capture essential components of spatial topology reasoning. Through extensive evaluation, we find that while frontier closed-source models (e.g., GPT-4o) achieve relatively high accuracy in some tasks, they still fail in some temporal questions that humans can answer (e.g., GPT-4o achieve only 67.8% in vector, a two-class classification problem). Furthermore, we find open-source VLMs, even at 30B scale, struggle significantly. These results indicate that spatial reasoning remains a fundamental bottleneck for current VLMs. We also find that the model's capability is positively correlated with model size, length of reasoning tokens and shots provided as examples, showing direction for future research.
>
---
#### [new 096] Tensor-Based Self-Calibration of Cameras via the TrifocalCalib Method
- **分类: cs.CV**

- **简介: 该论文提出TrifocalCalib方法，用于无需标定目标和相机运动约束的自标定任务，解决相机内参实时估计问题。基于三焦张量构建方程，提升了标定精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.17620v1](http://arxiv.org/pdf/2509.17620v1)**

> **作者:** Gregory Schroeder; Mohamed Sabry; Cristina Olaverri-Monreal
>
> **摘要:** Estimating camera intrinsic parameters without prior scene knowledge is a fundamental challenge in computer vision. This capability is particularly important for applications such as autonomous driving and vehicle platooning, where precalibrated setups are impractical and real-time adaptability is necessary. To advance the state-of-the-art, we present a set of equations based on the calibrated trifocal tensor, enabling projective camera self-calibration from minimal image data. Our method, termed TrifocalCalib, significantly improves accuracy and robustness compared to both recent learning-based and classical approaches. Unlike many existing techniques, our approach requires no calibration target, imposes no constraints on camera motion, and simultaneously estimates both focal length and principal point. Evaluations in both procedurally generated synthetic environments and structured dataset-based scenarios demonstrate the effectiveness of our approach. To support reproducibility, we make the code publicly available.
>
---
#### [new 097] Thermal Imaging-based Real-time Fall Detection using Motion Flow and Attention-enhanced Convolutional Recurrent Architecture
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于热成像的实时跌倒检测方法，采用增强注意力机制的双向卷积LSTM模型。旨在解决老年人跌倒检测中的隐私、可靠性和实时性问题，通过多组实验验证了模型在多个数据集上的优越性能。**

- **链接: [http://arxiv.org/pdf/2509.16479v1](http://arxiv.org/pdf/2509.16479v1)**

> **作者:** Christopher Silver; Thangarajah Akilan
>
> **摘要:** Falls among seniors are a major public health issue. Existing solutions using wearable sensors, ambient sensors, and RGB-based vision systems face challenges in reliability, user compliance, and practicality. Studies indicate that stakeholders, such as older adults and eldercare facilities, prefer non-wearable, passive, privacy-preserving, and real-time fall detection systems that require no user interaction. This study proposes an advanced thermal fall detection method using a Bidirectional Convolutional Long Short-Term Memory (BiConvLSTM) model, enhanced with spatial, temporal, feature, self, and general attention mechanisms. Through systematic experimentation across hundreds of model variations exploring the integration of attention mechanisms, recurrent modules, and motion flow, we identified top-performing architectures. Among them, BiConvLSTM achieved state-of-the-art performance with a ROC-AUC of $99.7\%$ on the TSF dataset and demonstrated robust results on TF-66, a newly emerged, diverse, and privacy-preserving benchmark. These results highlight the generalizability and practicality of the proposed model, setting new standards for thermal fall detection and paving the way toward deployable, high-performance solutions.
>
---
#### [new 098] DT-NeRF: A Diffusion and Transformer-Based Optimization Approach for Neural Radiance Fields in 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出DT-NeRF方法，结合扩散模型与Transformer优化神经辐射场（NeRF），用于3D重建任务。旨在提升稀疏视角下的细节恢复和多视角一致性，实验表明其在多个指标上优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.17232v1](http://arxiv.org/pdf/2509.17232v1)**

> **作者:** Bo Liu; Runlong Li; Li Zhou; Yan Zhou
>
> **备注:** 15 pages
>
> **摘要:** This paper proposes a Diffusion Model-Optimized Neural Radiance Field (DT-NeRF) method, aimed at enhancing detail recovery and multi-view consistency in 3D scene reconstruction. By combining diffusion models with Transformers, DT-NeRF effectively restores details under sparse viewpoints and maintains high accuracy in complex geometric scenes. Experimental results demonstrate that DT-NeRF significantly outperforms traditional NeRF and other state-of-the-art methods on the Matterport3D and ShapeNet datasets, particularly in metrics such as PSNR, SSIM, Chamfer Distance, and Fidelity. Ablation experiments further confirm the critical role of the diffusion and Transformer modules in the model's performance, with the removal of either module leading to a decline in performance. The design of DT-NeRF showcases the synergistic effect between modules, providing an efficient and accurate solution for 3D scene reconstruction. Future research may focus on further optimizing the model, exploring more advanced generative models and network architectures to enhance its performance in large-scale dynamic scenes.
>
---
#### [new 099] Explainable Gait Abnormality Detection Using Dual-Dataset CNN-LSTM Models
- **分类: cs.CV**

- **简介: 该论文提出一种双数据集的CNN-LSTM框架，用于可解释的步态异常检测。针对现有模型缺乏可解释性和依赖单一数据集的问题，结合GAVD和OU-MVLP数据，并通过SHAP和Grad-CAM提供解释性，在测试中表现出高准确率和召回率。**

- **链接: [http://arxiv.org/pdf/2509.16472v1](http://arxiv.org/pdf/2509.16472v1)**

> **作者:** Parth Agarwal; Sangaa Chatterjee; Md Faisal Kabir; Suman Saha
>
> **备注:** The paper got accepted in ICMLA-2025. It is a camera-ready version
>
> **摘要:** Gait is a key indicator in diagnosing movement disorders, but most models lack interpretability and rely on single datasets. We propose a dual-branch CNN-LSTM framework a 1D branch on joint-based features from GAVD and a 3D branch on silhouettes from OU-MVLP. Interpretability is provided by SHAP (temporal attributions) and Grad-CAM (spatial localization).On held-out sets, the system achieves 98.6% accuracy with strong recall and F1. This approach advances explainable gait analysis across both clinical and biometric domains.
>
---
#### [new 100] Surgical-MambaLLM: Mamba2-enhanced Multimodal Large Language Model for VQLA in Robotic Surgery
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Surgical-MambaLLM，用于机器人手术中的视觉问答定位（VQLA）任务。针对现有方法在跨模态依赖和空间感知上的不足，结合Mamba2设计CBMI模块和SIP扫描模式，提升模型对术中图像的理解能力。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16618v1](http://arxiv.org/pdf/2509.16618v1)**

> **作者:** Pengfei Hao; Hongqiu Wang; Shuaibo Li; Zhaohu Xing; Guang Yang; Kaishun Wu; Lei Zhu
>
> **备注:** Early accepted by MICCAI2025
>
> **摘要:** In recent years, Visual Question Localized-Answering in robotic surgery (Surgical-VQLA) has gained significant attention for its potential to assist medical students and junior doctors in understanding surgical scenes. Recently, the rapid development of Large Language Models (LLMs) has provided more promising solutions for this task. However, current methods struggle to establish complex dependencies between text and visual details, and have difficulty perceiving the spatial information of surgical scenes. To address these challenges, we propose a novel method, Surgical-MambaLLM, which is the first to combine Mamba2 with LLM in the surgical domain, that leverages Mamba2's ability to effectively capture cross-modal dependencies and perceive spatial information in surgical scenes, thereby enhancing the LLMs' understanding of surgical images. Specifically, we propose the Cross-modal Bidirectional Mamba2 Integration (CBMI) module to leverage Mamba2 for effective multimodal fusion, with its cross-modal integration capabilities. Additionally, tailored to the geometric characteristics of surgical scenes, we design the Surgical Instrument Perception (SIP) scanning mode for Mamba2 to scan the surgical images, enhancing the model's spatial understanding of the surgical scene. Extensive experiments demonstrate that our Surgical-MambaLLM model outperforms the state-of-the-art methods on the EndoVis17-VQLA and EndoVis18-VQLA datasets, significantly improving the performance of the Surgical-VQLA task.
>
---
#### [new 101] Accurate and Efficient Low-Rank Model Merging in Core Space
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Core Space合并框架，用于高效且准确地合并低秩适配（如LoRA）模型。针对现有方法在合并时效率低下的问题，通过在公共对齐基上进行合并，保留低秩效率并提升任务准确性，适用于视觉和语言任务。**

- **链接: [http://arxiv.org/pdf/2509.17786v1](http://arxiv.org/pdf/2509.17786v1)**

> **作者:** Aniello Panariello; Daniel Marczak; Simone Magistri; Angelo Porrello; Bartłomiej Twardowski; Andrew D. Bagdanov; Simone Calderara; Joost van de Weijer
>
> **备注:** Accepted at 39th Conference on Neural Information Processing Systems (NeurIPS 2025), San Diego, USA
>
> **摘要:** In this paper, we address the challenges associated with merging low-rank adaptations of large neural networks. With the rise of parameter-efficient adaptation techniques, such as Low-Rank Adaptation (LoRA), model fine-tuning has become more accessible. While fine-tuning models with LoRA is highly efficient, existing merging methods often sacrifice this efficiency by merging fully-sized weight matrices. We propose the Core Space merging framework, which enables the merging of LoRA-adapted models within a common alignment basis, thereby preserving the efficiency of low-rank adaptation while substantially improving accuracy across tasks. We further provide a formal proof that projection into Core Space ensures no loss of information and provide a complexity analysis showing the efficiency gains. Extensive empirical results demonstrate that Core Space significantly improves existing merging techniques and achieves state-of-the-art results on both vision and language tasks while utilizing a fraction of the computational resources. Codebase is available at https://github.com/apanariello4/core-space-merging.
>
---
#### [new 102] L2M-Reg: Building-level Uncertainty-aware Registration of Outdoor LiDAR Point Clouds and Semantic 3D City Models
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于LiDAR点云与语义3D城市模型的配准任务，旨在解决LoD2模型不确定性导致的建筑级配准难题。提出了L2M-Reg方法，通过平面对应、约束模型和自适应估计提高配准精度与效率。**

- **链接: [http://arxiv.org/pdf/2509.16832v1](http://arxiv.org/pdf/2509.16832v1)**

> **作者:** Ziyang Xu; Benedikt Schwab; Yihui Yang; Thomas H. Kolbe; Christoph Holst
>
> **备注:** submit to ISPRS Journal of Photogrammetry and Remote Sensing
>
> **摘要:** Accurate registration between LiDAR (Light Detection and Ranging) point clouds and semantic 3D city models is a fundamental topic in urban digital twinning and a prerequisite for downstream tasks, such as digital construction, change detection and model refinement. However, achieving accurate LiDAR-to-Model registration at individual building level remains challenging, particularly due to the generalization uncertainty in semantic 3D city models at the Level of Detail 2 (LoD2). This paper addresses this gap by proposing L2M-Reg, a plane-based fine registration method that explicitly accounts for model uncertainty. L2M-Reg consists of three key steps: establishing reliable plane correspondence, building a pseudo-plane-constrained Gauss-Helmert model, and adaptively estimating vertical translation. Experiments on three real-world datasets demonstrate that L2M-Reg is both more accurate and computationally efficient than existing ICP-based and plane-based methods. Overall, L2M-Reg provides a novel building-level solution regarding LiDAR-to-Model registration when model uncertainty is present.
>
---
#### [new 103] Towards a Transparent and Interpretable AI Model for Medical Image Classifications
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类中的AI可解释性研究，旨在解决AI模型在医疗领域应用中的不透明问题。作者通过多种XAI方法和数据集模拟，验证了XAI提升AI决策透明度和临床实用性的能力，并探讨了相关挑战。**

- **链接: [http://arxiv.org/pdf/2509.16685v1](http://arxiv.org/pdf/2509.16685v1)**

> **作者:** Binbin Wen; Yihang Wu; Tareef Daqqaq; Ahmad Chaddad
>
> **备注:** Published in Cognitive Neurodynamics
>
> **摘要:** The integration of artificial intelligence (AI) into medicine is remarkable, offering advanced diagnostic and therapeutic possibilities. However, the inherent opacity of complex AI models presents significant challenges to their clinical practicality. This paper focuses primarily on investigating the application of explainable artificial intelligence (XAI) methods, with the aim of making AI decisions transparent and interpretable. Our research focuses on implementing simulations using various medical datasets to elucidate the internal workings of the XAI model. These dataset-driven simulations demonstrate how XAI effectively interprets AI predictions, thus improving the decision-making process for healthcare professionals. In addition to a survey of the main XAI methods and simulations, ongoing challenges in the XAI field are discussed. The study highlights the need for the continuous development and exploration of XAI, particularly from the perspective of diverse medical datasets, to promote its adoption and effectiveness in the healthcare domain.
>
---
#### [new 104] Learning from Gene Names, Expression Values and Images: Contrastive Masked Text-Image Pretraining for Spatial Transcriptomics Representation Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CoMTIP，一种用于空间转录组学的对比掩码图文预训练框架。任务是学习跨模态表征以关联组织图像与基因表达。解决了现有方法忽视基因数值和视觉上下文的问题，通过联合建模图像、基因名及表达值，提升了下游任务性能并实现了零样本预测。**

- **链接: [http://arxiv.org/pdf/2509.16892v1](http://arxiv.org/pdf/2509.16892v1)**

> **作者:** Jiahe Qian; Yaoyu Fang; Ziqiao Weng; Xinkun Wang; Lee A. Cooper; Bo Zhou
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Spatial transcriptomics aims to connect high-resolution histology images with spatially resolved gene expression. To achieve better performance on downstream tasks such as gene expression prediction, large-scale pre-training is required to obtain generalisable representations that can bridge histology and transcriptomics across tissues, protocols, and laboratories. Existing cross-modal pre-training approaches for spatial transcriptomics rely on either gene names or expression values in isolation, which strips the gene branch of essential semantics and breaks the association between each gene and its quantitative magnitude. In addition, by restricting supervision to image-text alignment, these methods ignore intrinsic visual cues that are critical for learning robust image features. We present CoMTIP, the first Contrastive Masked Text-Image Pretraining framework that jointly learns from images, gene names, and expression values while capturing fine-grained visual context for spatial transcriptomics. The vision branch uses Masked Feature Modeling to reconstruct occluded patches and learn context-aware image embeddings. The text branch applies a scalable Gene-Text Encoder that processes all gene sentences in parallel, enriches each gene and its numerical value with dedicated embeddings, and employs Pair-aware Adversarial Training (PAAT) to preserve correct gene-value associations. Image and text representations are aligned in a shared InfoNCE-optimised space. Experiments on public spatial transcriptomics datasets show that CoMTIP not only surpasses previous methods on diverse downstream tasks but also achieves zero-shot gene expression prediction, a capability that existing approaches do not provide.
>
---
#### [new 105] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs
- **分类: cs.CV**

- **简介: 该论文提出TempSamp-R1，针对视频语言模型在时间定位任务中的不足，通过强化微调框架优化时序采样。利用真实标注提供精确指导，并采用非线性优势计算和混合推理训练，提升性能与稳定性，在多个基准数据集上取得新SOTA。**

- **链接: [http://arxiv.org/pdf/2509.18056v1](http://arxiv.org/pdf/2509.18056v1)**

> **作者:** Yunheng Li; Jing Cheng; Shaoyong Jia; Hangyi Kuang; Shaohui Jiao; Qibin Hou; Ming-Ming Cheng
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** This paper introduces TempSamp-R1, a new reinforcement fine-tuning framework designed to improve the effectiveness of adapting multimodal large language models (MLLMs) to video temporal grounding tasks. We reveal that existing reinforcement learning methods, such as Group Relative Policy Optimization (GRPO), rely on on-policy sampling for policy updates. However, in tasks with large temporal search spaces, this strategy becomes both inefficient and limited in performance, as it often fails to identify temporally accurate solutions. To address this limitation, TempSamp-R1 leverages ground-truth annotations as off-policy supervision to provide temporally precise guidance, effectively compensating for the sparsity and misalignment in on-policy solutions. To further stabilize training and reduce variance in reward-based updates, TempSamp-R1 provides a non-linear soft advantage computation method that dynamically reshapes the reward feedback via an asymmetric transformation. By employing a hybrid Chain-of-Thought (CoT) training paradigm, TempSamp-R1 optimizes a single unified model to support both CoT and non-CoT inference modes, enabling efficient handling of queries with varying reasoning complexity. Experimental results demonstrate that TempSamp-R1 outperforms GRPO-based baselines, establishing new state-of-the-art performance on benchmark datasets: Charades-STA (R1@0.7: 52.9%, +2.7%), ActivityNet Captions (R1@0.5: 56.0%, +5.3%), and QVHighlights (mAP: 30.0%, +3.0%). Moreover, TempSamp-R1 shows robust few-shot generalization capabilities under limited data. Code: https://github.com/HVision-NKU/TempSamp-R1
>
---
#### [new 106] RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation
- **分类: cs.CV**

- **简介: 该论文提出RLGF方法，用于自动驾驶合成视频生成任务。针对现有模型几何失真影响感知性能的问题，引入基于几何反馈的强化学习，通过优化扩散模型显著减少几何误差，提升3D目标检测效果。**

- **链接: [http://arxiv.org/pdf/2509.16500v1](http://arxiv.org/pdf/2509.16500v1)**

> **作者:** Tianyi Yan; Wencheng Han; Xia Zhou; Xueyang Zhang; Kun Zhan; Cheng-zhong Xu; Jianbing Shen
>
> **备注:** NeurIPS 2025
>
> **摘要:** Synthetic data is crucial for advancing autonomous driving (AD) systems, yet current state-of-the-art video generation models, despite their visual realism, suffer from subtle geometric distortions that limit their utility for downstream perception tasks. We identify and quantify this critical issue, demonstrating a significant performance gap in 3D object detection when using synthetic versus real data. To address this, we introduce Reinforcement Learning with Geometric Feedback (RLGF), RLGF uniquely refines video diffusion models by incorporating rewards from specialized latent-space AD perception models. Its core components include an efficient Latent-Space Windowing Optimization technique for targeted feedback during diffusion, and a Hierarchical Geometric Reward (HGR) system providing multi-level rewards for point-line-plane alignment, and scene occupancy coherence. To quantify these distortions, we propose GeoScores. Applied to models like DiVE on nuScenes, RLGF substantially reduces geometric errors (e.g., VP error by 21\%, Depth error by 57\%) and dramatically improves 3D object detection mAP by 12.7\%, narrowing the gap to real-data performance. RLGF offers a plug-and-play solution for generating geometrically sound and reliable synthetic videos for AD development.
>
---
#### [new 107] Interpreting vision transformers via residual replacement model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉Transformer（ViT）的可解释性研究任务，旨在解决ViT如何表示和处理信息的问题。通过分析6.6K个特征并提出残差替换模型，用可解释特征替代原始计算，实现对ViT机制的直观理解，并用于消除虚假相关性。**

- **链接: [http://arxiv.org/pdf/2509.17401v1](http://arxiv.org/pdf/2509.17401v1)**

> **作者:** Jinyeong Kim; Junhyeok Kim; Yumin Shim; Joohyeok Kim; Sunyoung Jung; Seong Jae Hwang
>
> **摘要:** How do vision transformers (ViTs) represent and process the world? This paper addresses this long-standing question through the first systematic analysis of 6.6K features across all layers, extracted via sparse autoencoders, and by introducing the residual replacement model, which replaces ViT computations with interpretable features in the residual stream. Our analysis reveals not only a feature evolution from low-level patterns to high-level semantics, but also how ViTs encode curves and spatial positions through specialized feature types. The residual replacement model scalably produces a faithful yet parsimonious circuit for human-scale interpretability by significantly simplifying the original computations. As a result, this framework enables intuitive understanding of ViT mechanisms. Finally, we demonstrate the utility of our framework in debiasing spurious correlations.
>
---
#### [new 108] Informative Text-Image Alignment for Visual Affordance Learning with Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦于视觉可用性学习任务，旨在解决文本与图像特征对齐不足导致性能欠佳的问题。提出基于互信息约束的框架，通过优化图文特征对齐，提升少样本下可用性区域识别效果，在AGD20K数据集上取得SOTA。**

- **链接: [http://arxiv.org/pdf/2509.17074v1](http://arxiv.org/pdf/2509.17074v1)**

> **作者:** Qian Zhang; Lin Zhang; Xing Fang; Mingxin Zhang; Zhiyuan Wei; Ran Song; Wei Zhang
>
> **备注:** Submitted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Visual affordance learning is crucial for robots to understand and interact effectively with the physical world. Recent advances in this field attempt to leverage pre-trained knowledge of vision-language foundation models to learn affordance properties with limited training data, providing a novel paradigm for visual affordance learning. However, these methods overlook the significance of maintaining feature alignment between visual images and language descriptions for identifying affordance areas with textual guidance, and thus may lead to suboptimal results. In this paper, we present an informative framework for text-guided affordance learning, which involves information-based constraints to achieve text-image alignment at feature level. Specifically, we design an affordance mutual information constraint that helps learn appropriate textual prompts and task-oriented visual features simultaneously by maximizing the mutual information between the features of the affordance areas in the input images and the corresponding textual prompts. In addition, we propose an object-level information constraint that maximizes the mutual information between the visual features of a given object and the text features of the category it belongs to. This enables the model to capture high-quality representations for the object, providing more reliable semantic priors for identifying affordance regions. Experimental results on the AGD20K dataset show that the proposed method outperforms existing approaches and achieves the new state-of-the-art in one-shot affordance learning.
>
---
#### [new 109] ComposeMe: Attribute-Specific Image Prompts for Controllable Human Image Generation
- **分类: cs.CV**

- **简介: 该论文提出ComposeMe，用于可控的人像生成任务。旨在解决现有方法在精细控制发型、服装等属性时的模块化不足问题。通过引入属性特定的图像提示和多属性交叉参考训练策略，实现了对多个视觉因素的解耦控制。**

- **链接: [http://arxiv.org/pdf/2509.18092v1](http://arxiv.org/pdf/2509.18092v1)**

> **作者:** Guocheng Gordon Qian; Daniil Ostashev; Egor Nemchinov; Avihay Assouline; Sergey Tulyakov; Kuan-Chieh Jackson Wang; Kfir Aberman
>
> **备注:** Accepted to SIGGRAPH Asia 2025, webpage: https://snap-research.github.io/composeme/
>
> **摘要:** Generating high-fidelity images of humans with fine-grained control over attributes such as hairstyle and clothing remains a core challenge in personalized text-to-image synthesis. While prior methods emphasize identity preservation from a reference image, they lack modularity and fail to provide disentangled control over specific visual attributes. We introduce a new paradigm for attribute-specific image prompting, in which distinct sets of reference images are used to guide the generation of individual aspects of human appearance, such as hair, clothing, and identity. Our method encodes these inputs into attribute-specific tokens, which are injected into a pre-trained text-to-image diffusion model. This enables compositional and disentangled control over multiple visual factors, even across multiple people within a single image. To promote natural composition and robust disentanglement, we curate a cross-reference training dataset featuring subjects in diverse poses and expressions, and propose a multi-attribute cross-reference training strategy that encourages the model to generate faithful outputs from misaligned attribute inputs while adhering to both identity and textual conditioning. Extensive experiments show that our method achieves state-of-the-art performance in accurately following both visual and textual prompts. Our framework paves the way for more configurable human image synthesis by combining visual prompting with text-driven generation. Webpage is available at: https://snap-research.github.io/composeme/.
>
---
#### [new 110] Vision-Based Driver Drowsiness Monitoring: Comparative Analysis of YOLOv5-v11 Models
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究基于YOLO算法的驾驶员疲劳检测任务，旨在提高检测准确性与效率。使用UTA-RLDD数据集对7种YOLO模型进行评估，并结合EAR方法分析性能权衡，为实际部署提供指导。**

- **链接: [http://arxiv.org/pdf/2509.17498v1](http://arxiv.org/pdf/2509.17498v1)**

> **作者:** Dilshara Herath; Chinthaka Abeyrathne; Prabhani Jayaweera
>
> **备注:** Drowsiness Detection using state of the art YOLO algorithms
>
> **摘要:** Driver drowsiness remains a critical factor in road accidents, accounting for thousands of fatalities and injuries each year. This paper presents a comprehensive evaluation of real-time, non-intrusive drowsiness detection methods, focusing on computer vision based YOLO (You Look Only Once) algorithms. A publicly available dataset namely, UTA-RLDD was used, containing both awake and drowsy conditions, ensuring variability in gender, eyewear, illumination, and skin tone. Seven YOLO variants (v5s, v9c, v9t, v10n, v10l, v11n, v11l) are fine-tuned, with performance measured in terms of Precision, Recall, mAP0.5, and mAP 0.5-0.95. Among these, YOLOv9c achieved the highest accuracy (0.986 mAP 0.5, 0.978 Recall) while YOLOv11n strikes the optimal balance between precision (0.954) and inference efficiency, making it highly suitable for embedded deployment. Additionally, we implement an Eye Aspect Ratio (EAR) approach using Dlib's facial landmarks, which despite its low computational footprint exhibits reduced robustness under pose variation and occlusions. Our findings illustrate clear trade offs between accuracy, latency, and resource requirements, and offer practical guidelines for selecting or combining detection methods in autonomous driving and industrial safety applications.
>
---
#### [new 111] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文研究文本到图像扩散模型中的跨模态注意力机制，提出Seg4Diff框架分析MM-DiT结构，识别语义对齐层，并通过微调提升分割与生成效果，推动视觉感知与生成的统一。**

- **链接: [http://arxiv.org/pdf/2509.18096v1](http://arxiv.org/pdf/2509.18096v1)**

> **作者:** Chaehyun Kim; Heeseong Shin; Eunbeen Hong; Heeji Yoon; Anurag Arnab; Paul Hongsuck Seo; Sunghwan Hong; Seungryong Kim
>
> **备注:** NeurIPS 2025. Project page: https://cvlab-kaist.github.io/Seg4Diff/
>
> **摘要:** Text-to-image diffusion models excel at translating language prompts into photorealistic images by implicitly grounding textual concepts through their cross-modal attention mechanisms. Recent multi-modal diffusion transformers extend this by introducing joint self-attention over concatenated image and text tokens, enabling richer and more scalable cross-modal alignment. However, a detailed understanding of how and where these attention maps contribute to image generation remains limited. In this paper, we introduce Seg4Diff (Segmentation for Diffusion), a systematic framework for analyzing the attention structures of MM-DiT, with a focus on how specific layers propagate semantic information from text to image. Through comprehensive analysis, we identify a semantic grounding expert layer, a specific MM-DiT block that consistently aligns text tokens with spatially coherent image regions, naturally producing high-quality semantic segmentation masks. We further demonstrate that applying a lightweight fine-tuning scheme with mask-annotated image data enhances the semantic grouping capabilities of these layers and thereby improves both segmentation performance and generated image fidelity. Our findings demonstrate that semantic grouping is an emergent property of diffusion transformers and can be selectively amplified to advance both segmentation and generation performance, paving the way for unified models that bridge visual perception and generation.
>
---
#### [new 112] OS-DiffVSR: Towards One-step Latent Diffusion Model for High-detailed Real-world Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文提出OS-DiffVSR，用于真实世界视频超分辨率（VSR）任务。针对扩散模型在视频质量与推理效率间的权衡问题，设计了一种一步式潜扩散模型，并引入相邻帧对抗训练和多帧融合机制，提升了视频质量和时序一致性。**

- **链接: [http://arxiv.org/pdf/2509.16507v1](http://arxiv.org/pdf/2509.16507v1)**

> **作者:** Hanting Li; Huaao Tang; Jianhong Han; Tianxiong Zhou; Jiulong Cui; Haizhen Xie; Yan Chen; Jie Hu
>
> **摘要:** Recently, latent diffusion models has demonstrated promising performance in real-world video super-resolution (VSR) task, which can reconstruct high-quality videos from distorted low-resolution input through multiple diffusion steps. Compared to image super-resolution (ISR), VSR methods needs to process each frame in a video, which poses challenges to its inference efficiency. However, video quality and inference efficiency have always been a trade-off for the diffusion-based VSR methods. In this work, we propose One-Step Diffusion model for real-world Video Super-Resolution, namely OS-DiffVSR. Specifically, we devise a novel adjacent frame adversarial training paradigm, which can significantly improve the quality of synthetic videos. Besides, we devise a multi-frame fusion mechanism to maintain inter-frame temporal consistency and reduce the flicker in video. Extensive experiments on several popular VSR benchmarks demonstrate that OS-DiffVSR can even achieve better quality than existing diffusion-based VSR methods that require dozens of sampling steps.
>
---
#### [new 113] SlowFast-SCI: Slow-Fast Deep Unfolding Learning for Spectral Compressive Imaging
- **分类: cs.CV**

- **简介: 该论文提出SlowFast-SCI，用于光谱压缩成像任务。针对现有方法在新光学配置下适应性差、计算量大的问题，设计了双速框架：慢学习预训练主干网络，快学习嵌入轻量模块，实现测试时自适应优化，提升跨域适应性和效率。**

- **链接: [http://arxiv.org/pdf/2509.16509v1](http://arxiv.org/pdf/2509.16509v1)**

> **作者:** Haijin Zeng; Xuan Lu; Yurong Zhang; Yongyong Chen; Jingyong Su; Jie Liu
>
> **备注:** 12 pages
>
> **摘要:** Humans learn in two complementary ways: a slow, cumulative process that builds broad, general knowledge, and a fast, on-the-fly process that captures specific experiences. Existing deep-unfolding methods for spectral compressive imaging (SCI) mirror only the slow component-relying on heavy pre-training with many unfolding stages-yet they lack the rapid adaptation needed to handle new optical configurations. As a result, they falter on out-of-distribution cameras, especially in bespoke spectral setups unseen during training. This depth also incurs heavy computation and slow inference. To bridge this gap, we introduce SlowFast-SCI, a dual-speed framework seamlessly integrated into any deep unfolding network beyond SCI systems. During slow learning, we pre-train or reuse a priors-based backbone and distill it via imaging guidance into a compact fast-unfolding model. In the fast learning stage, lightweight adaptation modules are embedded within each block and trained self-supervised at test time via a dual-domain loss-without retraining the backbone. To the best of our knowledge, SlowFast-SCI is the first test-time adaptation-driven deep unfolding framework for efficient, self-adaptive spectral reconstruction. Its dual-stage design unites offline robustness with on-the-fly per-sample calibration-yielding over 70% reduction in parameters and FLOPs, up to 5.79 dB PSNR improvement on out-of-distribution data, preserved cross-domain adaptability, and a 4x faster adaptation speed. In addition, its modularity integrates with any deep-unfolding network, paving the way for self-adaptive, field-deployable imaging and expanded computational imaging modalities. Code and models are available at https://github.com/XuanLu11/SlowFast-SCI.
>
---
#### [new 114] Learning Attribute-Aware Hash Codes for Fine-Grained Image Retrieval via Query Optimization
- **分类: cs.CV**

- **简介: 该论文聚焦细粒度图像检索任务，旨在提升哈希码的判别性与可解释性。提出一种基于可学习查询的属性感知哈希方法，通过优化框架和辅助分支建模高阶属性交互，有效生成具有属性语义的低比特哈希码，提升了检索精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.17049v1](http://arxiv.org/pdf/2509.17049v1)**

> **作者:** Peng Wang; Yong Li; Lin Zhao; Xiu-Shen Wei
>
> **摘要:** Fine-grained hashing has become a powerful solution for rapid and efficient image retrieval, particularly in scenarios requiring high discrimination between visually similar categories. To enable each hash bit to correspond to specific visual attributes, we propoe a novel method that harnesses learnable queries for attribute-aware hash codes learning. This method deploys a tailored set of queries to capture and represent nuanced attribute-level information within the hashing process, thereby enhancing both the interpretability and relevance of each hash bit. Building on this query-based optimization framework, we incorporate an auxiliary branch to help alleviate the challenges of complex landscape optimization often encountered with low-bit hash codes. This auxiliary branch models high-order attribute interactions, reinforcing the robustness and specificity of the generated hash codes. Experimental results on benchmark datasets demonstrate that our method generates attribute-aware hash codes and consistently outperforms state-of-the-art techniques in retrieval accuracy and robustness, especially for low-bit hash codes, underscoring its potential in fine-grained image hashing tasks.
>
---
#### [new 115] SFN-YOLO: Towards Free-Range Poultry Detection via Scale-aware Fusion Networks
- **分类: cs.CV**

- **简介: 该论文提出SFN-YOLO，用于自由放养环境中家禽检测。针对多尺度、遮挡和复杂背景问题，设计了尺度感知融合网络，并构建新数据集M-SCOPE。实验表明模型高效且泛化性强，支持智能养殖自动化。**

- **链接: [http://arxiv.org/pdf/2509.17086v1](http://arxiv.org/pdf/2509.17086v1)**

> **作者:** Jie Chen; Yuhong Feng; Tao Dai; Mingzhe Liu; Hongtao Chen; Zhaoxi He; Jiancong Bai
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Detecting and localizing poultry is essential for advancing smart poultry farming. Despite the progress of detection-centric methods, challenges persist in free-range settings due to multiscale targets, obstructions, and complex or dynamic backgrounds. To tackle these challenges, we introduce an innovative poultry detection approach named SFN-YOLO that utilizes scale-aware fusion. This approach combines detailed local features with broader global context to improve detection in intricate environments. Furthermore, we have developed a new expansive dataset (M-SCOPE) tailored for varied free-range conditions. Comprehensive experiments demonstrate our model achieves an mAP of 80.7% with just 7.2M parameters, which is 35.1% fewer than the benchmark, while retaining strong generalization capability across different domains. The efficient and real-time detection capabilities of SFN-YOLO support automated smart poultry farming. The code and dataset can be accessed at https://github.com/chenjessiee/SFN-YOLO.
>
---
#### [new 116] SAMSON: 3rd Place Solution of LSVOS 2025 VOS Challenge
- **分类: cs.CV**

- **简介: 该论文提出SAMSON，用于大规模视频目标分割（LSVOS）任务，旨在解决长视频中目标重识别、遮挡和误差累积问题。方法引入长期记忆模块和SAM2Long后处理策略，提升分割稳定性与准确性，在MOSE赛道中取得第3名。**

- **链接: [http://arxiv.org/pdf/2509.17500v1](http://arxiv.org/pdf/2509.17500v1)**

> **作者:** Yujie Xie; Hongyang Zhang; Zhihui Liu; Shihai Ruan
>
> **摘要:** Large-scale Video Object Segmentation (LSVOS) addresses the challenge of accurately tracking and segmenting objects in long video sequences, where difficulties stem from object reappearance, small-scale targets, heavy occlusions, and crowded scenes. Existing approaches predominantly adopt SAM2-based frameworks with various memory mechanisms for complex video mask generation. In this report, we proposed Segment Anything with Memory Strengthened Object Navigation (SAMSON), the 3rd place solution in the MOSE track of ICCV 2025, which integrates the strengths of stateof-the-art VOS models into an effective paradigm. To handle visually similar instances and long-term object disappearance in MOSE, we incorporate a long-term memorymodule for reliable object re-identification. Additionly, we adopt SAM2Long as a post-processing strategy to reduce error accumulation and enhance segmentation stability in long video sequences. Our method achieved a final performance of 0.8427 in terms of J &F in the test-set leaderboard.
>
---
#### [new 117] Ambiguous Medical Image Segmentation Using Diffusion Schrödinger Bridge
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Segmentation Schrödinger Bridge (SSB)，用于解决医学图像分割中边界模糊和标注多样性问题。通过建模图像-掩码联合动态，设计新损失函数并引入Diversity Divergence Index ($D_{DDI}$)，实现了结构完整性与分割多样性的平衡，在多个数据集上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.17187v1](http://arxiv.org/pdf/2509.17187v1)**

> **作者:** Lalith Bharadwaj Baru; Kamalaker Dadi; Tapabrata Chakraborti; Raju S. Bapi
>
> **备注:** MICCAI 2025 (11 pages, 2 figures, 1 table, and 26 references)
>
> **摘要:** Accurate segmentation of medical images is challenging due to unclear lesion boundaries and mask variability. We introduce \emph{Segmentation Sch\"{o}dinger Bridge (SSB)}, the first application of Sch\"{o}dinger Bridge for ambiguous medical image segmentation, modelling joint image-mask dynamics to enhance performance. SSB preserves structural integrity, delineates unclear boundaries without additional guidance, and maintains diversity using a novel loss function. We further propose the \emph{Diversity Divergence Index} ($D_{DDI}$) to quantify inter-rater variability, capturing both diversity and consensus. SSB achieves state-of-the-art performance on LIDC-IDRI, COCA, and RACER (in-house) datasets.
>
---
#### [new 118] SimToken: A Simple Baseline for Referring Audio-Visual Segmentation
- **分类: cs.CV**

- **简介: 该论文针对“基于自然语言的音视频分割”任务，提出SimToken框架，结合多模态大语言模型与SAM，通过生成语义token指导视频目标分割，并引入语义对齐损失提升性能。**

- **链接: [http://arxiv.org/pdf/2509.17537v1](http://arxiv.org/pdf/2509.17537v1)**

> **作者:** Dian Jin; Yanghao Zhou; Jinxing Zhou; Jiaqi Ma; Ruohao Guo; Dan Guo
>
> **摘要:** Referring Audio-Visual Segmentation (Ref-AVS) aims to segment specific objects in videos based on natural language expressions involving audio, vision, and text information. This task poses significant challenges in cross-modal reasoning and fine-grained object localization. In this paper, we propose a simple framework, SimToken, that integrates a multimodal large language model (MLLM) with the Segment Anything Model (SAM). The MLLM is guided to generate a special semantic token representing the referred object. This compact token, enriched with contextual information from all modalities, acts as a prompt to guide SAM to segment objectsacross video frames. To further improve semantic learning, we introduce a novel target-consistent semantic alignment loss that aligns token embeddings from different expressions but referring to the same object. Experiments on the Ref-AVS benchmark demonstrate that our approach achieves superior performance compared to existing methods.Code will be available at https://github.com/DianJin-HFUT/SimToken
>
---
#### [new 119] Automated Labeling of Intracranial Arteries with Uncertainty Quantification Using Deep Learning
- **分类: cs.CV; cs.LG; I.4.0**

- **简介: 该论文研究颅内动脉自动标注任务，旨在解决人工标注耗时且易受操作者差异影响的问题。提出基于深度学习的框架，结合不确定性量化，提升标注准确性和可靠性，并验证其在血流动力学分析中的临床实用性。**

- **链接: [http://arxiv.org/pdf/2509.17726v1](http://arxiv.org/pdf/2509.17726v1)**

> **作者:** Javier Bisbal; Patrick Winter; Sebastian Jofre; Aaron Ponce; Sameer A. Ansari; Ramez Abdalla; Michael Markl; Oliver Welin Odeback; Sergio Uribe; Cristian Tejos; Julio Sotelo; Susanne Schnell; David Marlevi
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Accurate anatomical labeling of intracranial arteries is essential for cerebrovascular diagnosis and hemodynamic analysis but remains time-consuming and subject to interoperator variability. We present a deep learning-based framework for automated artery labeling from 3D Time-of-Flight Magnetic Resonance Angiography (3D ToF-MRA) segmentations (n=35), incorporating uncertainty quantification to enhance interpretability and reliability. We evaluated three convolutional neural network architectures: (1) a UNet with residual encoder blocks, reflecting commonly used baselines in vascular labeling; (2) CS-Net, an attention-augmented UNet incorporating channel and spatial attention mechanisms for enhanced curvilinear structure recognition; and (3) nnUNet, a self-configuring framework that automates preprocessing, training, and architectural adaptation based on dataset characteristics. Among these, nnUNet achieved the highest labeling performance (average Dice score: 0.922; average surface distance: 0.387 mm), with improved robustness in anatomically complex vessels. To assess predictive confidence, we implemented test-time augmentation (TTA) and introduced a novel coordinate-guided strategy to reduce interpolation errors during augmented inference. The resulting uncertainty maps reliably indicated regions of anatomical ambiguity, pathological variation, or manual labeling inconsistency. We further validated clinical utility by comparing flow velocities derived from automated and manual labels in co-registered 4D Flow MRI datasets, observing close agreement with no statistically significant differences. Our framework offers a scalable, accurate, and uncertainty-aware solution for automated cerebrovascular labeling, supporting downstream hemodynamic analysis and facilitating clinical integration.
>
---
#### [new 120] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming
- **分类: cs.CV**

- **简介: 该论文提出4DGCPro，一种高效的分层4D高斯压缩框架，用于渐进式体视频流传输。旨在解决高质量体视频在移动设备上实时解码与渲染的难题，通过感知加权表示和熵优化训练，实现单模型多比特率、多质量灵活流媒体传输。**

- **链接: [http://arxiv.org/pdf/2509.17513v1](http://arxiv.org/pdf/2509.17513v1)**

> **作者:** Zihan Zheng; Zhenlong Wu; Houqiang Zhong; Yuan Tian; Ning Cao; Lan Xu; Jiangchao Yao; Xiaoyun Zhang; Qiang Hu; Wenjun Zhang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Achieving seamless viewing of high-fidelity volumetric video, comparable to 2D video experiences, remains an open challenge. Existing volumetric video compression methods either lack the flexibility to adjust quality and bitrate within a single model for efficient streaming across diverse networks and devices, or struggle with real-time decoding and rendering on lightweight mobile platforms. To address these challenges, we introduce 4DGCPro, a novel hierarchical 4D Gaussian compression framework that facilitates real-time mobile decoding and high-quality rendering via progressive volumetric video streaming in a single bitstream. Specifically, we propose a perceptually-weighted and compression-friendly hierarchical 4D Gaussian representation with motion-aware adaptive grouping to reduce temporal redundancy, preserve coherence, and enable scalable multi-level detail streaming. Furthermore, we present an end-to-end entropy-optimized training scheme, which incorporates layer-wise rate-distortion (RD) supervision and attribute-specific entropy modeling for efficient bitstream generation. Extensive experiments show that 4DGCPro enables flexible quality and multiple bitrate within a single model, achieving real-time decoding and rendering on mobile devices while outperforming existing methods in RD performance across multiple datasets. Project Page: https://mediax-sjtu.github.io/4DGCPro
>
---
#### [new 121] Overview of PlantCLEF 2025: Multi-Species Plant Identification in Vegetation Quadrat Images
- **分类: cs.CV**

- **简介: 论文介绍了PlantCLEF 2025挑战，旨在通过AI加速植被样方图像中多物种植物的识别。任务是基于单标签数据进行弱标注多标签分类，提供了大规模训练数据和预训练模型，并评估了多种方法的效果。**

- **链接: [http://arxiv.org/pdf/2509.17602v1](http://arxiv.org/pdf/2509.17602v1)**

> **作者:** Giulio Martellucci; Herve Goeau; Pierre Bonnet; Fabrice Vinatier; Alexis Joly
>
> **备注:** 13 pages, 4 figures, CLEF 2025 Conference and Labs of the Evaluation Forum, September 09 to 12, 2024, Madrid, Spain
>
> **摘要:** Quadrat images are essential for ecological studies, as they enable standardized sampling, the assessment of plant biodiversity, long-term monitoring, and large-scale field campaigns. These images typically cover an area of fifty centimetres or one square meter, and botanists carefully identify all the species present. Integrating AI could help specialists accelerate their inventories and expand the spatial coverage of ecological studies. To assess progress in this area, the PlantCLEF 2025 challenge relies on a new test set of 2,105 high-resolution multi-label images annotated by experts and covering around 400 species. It also provides a large training set of 1.4 million individual plant images, along with vision transformer models pre-trained on this data. The task is formulated as a (weakly labelled) multi-label classification problem, where the goal is to predict all species present in a quadrat image using single-label training data. This paper provides a detailed description of the data, the evaluation methodology, the methods and models used by participants, and the results achieved.
>
---
#### [new 122] Depth Edge Alignment Loss: DEALing with Depth in Weakly Supervised Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文研究弱监督语义分割任务，旨在减少对昂贵像素级标注的依赖。提出一种模型无关的深度边缘对齐损失（DEAL），利用图像级监督和可用的深度信息生成像素级标签，提升分割性能，并在多个数据集上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.17702v1](http://arxiv.org/pdf/2509.17702v1)**

> **作者:** Patrick Schmidt; Vasileios Belagiannis; Lazaros Nalpantidis
>
> **备注:** Submitted to IEEE
>
> **摘要:** Autonomous robotic systems applied to new domains require an abundance of expensive, pixel-level dense labels to train robust semantic segmentation models under full supervision. This study proposes a model-agnostic Depth Edge Alignment Loss to improve Weakly Supervised Semantic Segmentation models across different datasets. The methodology generates pixel-level semantic labels from image-level supervision, avoiding expensive annotation processes. While weak supervision is widely explored in traditional computer vision, our approach adds supervision with pixel-level depth information, a modality commonly available in robotic systems. We demonstrate how our approach improves segmentation performance across datasets and models, but can also be combined with other losses for even better performance, with improvements up to +5.439, +1.274 and +16.416 points in mean Intersection over Union on the PASCAL VOC / MS COCO validation, and the HOPE static onboarding split, respectively. Our code will be made publicly available.
>
---
#### [new 123] CoBEVMoE: Heterogeneity-aware Feature Fusion with Dynamic Mixture-of-Experts for Collaborative Perception
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出CoBEVMoE，用于多智能体协作感知任务。针对异构特征融合问题，设计了动态混合专家架构，在BEV空间中建模特征相似性与异质性，并引入DEML损失提升表现。**

- **链接: [http://arxiv.org/pdf/2509.17107v1](http://arxiv.org/pdf/2509.17107v1)**

> **作者:** Lingzhao Kong; Jiacheng Lin; Siyu Li; Kai Luo; Zhiyong Li; Kailun Yang
>
> **备注:** The source code will be made publicly available at https://github.com/godk0509/CoBEVMoE
>
> **摘要:** Collaborative perception aims to extend sensing coverage and improve perception accuracy by sharing information among multiple agents. However, due to differences in viewpoints and spatial positions, agents often acquire heterogeneous observations. Existing intermediate fusion methods primarily focus on aligning similar features, often overlooking the perceptual diversity among agents. To address this limitation, we propose CoBEVMoE, a novel collaborative perception framework that operates in the Bird's Eye View (BEV) space and incorporates a Dynamic Mixture-of-Experts (DMoE) architecture. In DMoE, each expert is dynamically generated based on the input features of a specific agent, enabling it to extract distinctive and reliable cues while attending to shared semantics. This design allows the fusion process to explicitly model both feature similarity and heterogeneity across agents. Furthermore, we introduce a Dynamic Expert Metric Loss (DEML) to enhance inter-expert diversity and improve the discriminability of the fused representation. Extensive experiments on the OPV2V and DAIR-V2X-C datasets demonstrate that CoBEVMoE achieves state-of-the-art performance. Specifically, it improves the IoU for Camera-based BEV segmentation by +1.5% on OPV2V and the AP@50 for LiDAR-based 3D object detection by +3.0% on DAIR-V2X-C, verifying the effectiveness of expert-based heterogeneous feature modeling in multi-agent collaborative perception. The source code will be made publicly available at https://github.com/godk0509/CoBEVMoE.
>
---
#### [new 124] Echo-Path: Pathology-Conditioned Echo Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Echo-Path，一种生成式框架，用于生成特定心脏病理条件下的超声心动图视频。旨在解决病理数据稀缺问题，提升自动化诊断模型的性能。**

- **链接: [http://arxiv.org/pdf/2509.17190v1](http://arxiv.org/pdf/2509.17190v1)**

> **作者:** Kabir Hamzah Muhammad; Marawan Elbatel; Yi Qin; Xiaomeng Li
>
> **备注:** 10 pages, 3 figures, MICCAI-AMAI2025 Workshop
>
> **摘要:** Cardiovascular diseases (CVDs) remain the leading cause of mortality globally, and echocardiography is critical for diagnosis of both common and congenital cardiac conditions. However, echocardiographic data for certain pathologies are scarce, hindering the development of robust automated diagnosis models. In this work, we propose Echo-Path, a novel generative framework to produce echocardiogram videos conditioned on specific cardiac pathologies. Echo-Path can synthesize realistic ultrasound video sequences that exhibit targeted abnormalities, focusing here on atrial septal defect (ASD) and pulmonary arterial hypertension (PAH). Our approach introduces a pathology-conditioning mechanism into a state-of-the-art echo video generator, allowing the model to learn and control disease-specific structural and motion patterns in the heart. Quantitative evaluation demonstrates that the synthetic videos achieve low distribution distances, indicating high visual fidelity. Clinically, the generated echoes exhibit plausible pathology markers. Furthermore, classifiers trained on our synthetic data generalize well to real data and, when used to augment real training sets, it improves downstream diagnosis of ASD and PAH by 7\% and 8\% respectively. Code, weights and dataset are available here https://github.com/Marshall-mk/EchoPathv1
>
---
#### [new 125] TractoTransformer: Diffusion MRI Streamline Tractography using CNN and Transformer Networks
- **分类: cs.CV**

- **简介: 该论文提出TractoTransformer，结合CNN与Transformer网络，用于白质纤维追踪任务。旨在解决扩散MRI中因交叉、合并等问题导致的纤维轨迹重建不准确问题。通过融合空间信息和轨迹上下文，提升了白质路径映射的精度与完整性。**

- **链接: [http://arxiv.org/pdf/2509.16429v1](http://arxiv.org/pdf/2509.16429v1)**

> **作者:** Itzik Waizman; Yakov Gusakov; Itay Benou; Tammy Riklin Raviv
>
> **摘要:** White matter tractography is an advanced neuroimaging technique that reconstructs the 3D white matter pathways of the brain from diffusion MRI data. It can be framed as a pathfinding problem aiming to infer neural fiber trajectories from noisy and ambiguous measurements, facing challenges such as crossing, merging, and fanning white-matter configurations. In this paper, we propose a novel tractography method that leverages Transformers to model the sequential nature of white matter streamlines, enabling the prediction of fiber directions by integrating both the trajectory context and current diffusion MRI measurements. To incorporate spatial information, we utilize CNNs that extract microstructural features from local neighborhoods around each voxel. By combining these complementary sources of information, our approach improves the precision and completeness of neural pathway mapping compared to traditional tractography models. We evaluate our method with the Tractometer toolkit, achieving competitive performance against state-of-the-art approaches, and present qualitative results on the TractoInferno dataset, demonstrating strong generalization to real-world data.
>
---
#### [new 126] Degradation-Aware All-in-One Image Restoration via Latent Prior Encoding
- **分类: cs.CV**

- **简介: 该论文提出一种基于潜在先验编码的通用图像修复方法，旨在解决真实场景中多种复杂退化问题。通过自动学习退化感知的潜在表示，无需外部提示即可实现自适应特征选择、空间定位与语义修复，提升了对未知混合退化的泛化能力与效率。**

- **链接: [http://arxiv.org/pdf/2509.17792v1](http://arxiv.org/pdf/2509.17792v1)**

> **作者:** S M A Sharif; Abdur Rehman; Fayaz Ali Dharejo; Radu Timofte; Rizwan Ali Naqvi
>
> **摘要:** Real-world images often suffer from spatially diverse degradations such as haze, rain, snow, and low-light, significantly impacting visual quality and downstream vision tasks. Existing all-in-one restoration (AIR) approaches either depend on external text prompts or embed hand-crafted architectural priors (e.g., frequency heuristics); both impose discrete, brittle assumptions that weaken generalization to unseen or mixed degradations. To address this limitation, we propose to reframe AIR as learned latent prior inference, where degradation-aware representations are automatically inferred from the input without explicit task cues. Based on latent priors, we formulate AIR as a structured reasoning paradigm: (1) which features to route (adaptive feature selection), (2) where to restore (spatial localization), and (3) what to restore (degradation semantics). We design a lightweight decoding module that efficiently leverages these latent encoded cues for spatially-adaptive restoration. Extensive experiments across six common degradation tasks, five compound settings, and previously unseen degradations demonstrate that our method outperforms state-of-the-art (SOTA) approaches, achieving an average PSNR improvement of 1.68 dB while being three times more efficient.
>
---
#### [new 127] PM25Vision: A Large-Scale Benchmark Dataset for Visual Estimation of Air Quality
- **分类: cs.CV**

- **简介: 该论文提出了PM25Vision，一个用于从街景图像估算空气质量（PM2.5浓度）的大规模基准数据集。它包含11,114张图像与3,261个监测站11年的PM2.5数据，空间精度达5公里。论文还提供了数据处理流程和基线模型性能，推动视觉空气质量估计任务的发展。**

- **链接: [http://arxiv.org/pdf/2509.16519v1](http://arxiv.org/pdf/2509.16519v1)**

> **作者:** Yang Han
>
> **摘要:** We introduce PM25Vision (PM25V), the largest and most comprehensive dataset to date for estimating air quality - specifically PM2.5 concentrations - from street-level images. The dataset contains over 11,114 images matched with timestamped and geolocated PM2.5 readings across 3,261 AQI monitoring stations and 11 years, significantly exceeding the scale of previous benchmarks. The spatial accuracy of this dataset has reached 5 kilometers, far exceeding the city-level accuracy of many datasets. We describe the data collection, synchronization, and cleaning pipelines, and provide baseline model performances using CNN and transformer architectures. Our dataset is publicly available.
>
---
#### [new 128] MedGS: Gaussian Splatting for Multi-Modal 3D Medical Imaging
- **分类: cs.CV**

- **简介: 该论文提出MedGS，一种基于高斯点绘的半监督神经隐式表面重建框架，用于多模态3D医学影像。旨在解决传统方法在噪声和帧间信息缺失下的建模与插值问题，实现更高效、鲁棒且高保真的医学图像重建与可视化。**

- **链接: [http://arxiv.org/pdf/2509.16806v1](http://arxiv.org/pdf/2509.16806v1)**

> **作者:** Kacper Marzol; Ignacy Kolton; Weronika Smolak-Dyżewska; Joanna Kaleta; Marcin Mazur; Przemysław Spurek
>
> **摘要:** Multi-modal three-dimensional (3D) medical imaging data, derived from ultrasound, magnetic resonance imaging (MRI), and potentially computed tomography (CT), provide a widely adopted approach for non-invasive anatomical visualization. Accurate modeling, registration, and visualization in this setting depend on surface reconstruction and frame-to-frame interpolation. Traditional methods often face limitations due to image noise and incomplete information between frames. To address these challenges, we present MedGS, a semi-supervised neural implicit surface reconstruction framework that employs a Gaussian Splatting (GS)-based interpolation mechanism. In this framework, medical imaging data are represented as consecutive two-dimensional (2D) frames embedded in 3D space and modeled using Gaussian-based distributions. This representation enables robust frame interpolation and high-fidelity surface reconstruction across imaging modalities. As a result, MedGS offers more efficient training than traditional neural implicit methods. Its explicit GS-based representation enhances noise robustness, allows flexible editing, and supports precise modeling of complex anatomical structures with fewer artifacts. These features make MedGS highly suitable for scalable and practical applications in medical imaging.
>
---
#### [new 129] DA-Font: Few-Shot Font Generation via Dual-Attention Hybrid Integration
- **分类: cs.CV**

- **简介: 该论文提出DA-Font，用于少量样本字体生成任务，旨在解决现有方法生成字体时存在的笔画错误、模糊等问题。通过引入双注意混合模块（DAHM）及两种协同注意力块，并设计新的损失函数，提升了字体结构和风格的准确性与一致性。**

- **链接: [http://arxiv.org/pdf/2509.16632v1](http://arxiv.org/pdf/2509.16632v1)**

> **作者:** Weiran Chen; Guiqian Zhu; Ying Li; Yi Ji; Chunping Liu
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Few-shot font generation aims to create new fonts with a limited number of glyph references. It can be used to significantly reduce the labor cost of manual font design. However, due to the variety and complexity of font styles, the results generated by existing methods often suffer from visible defects, such as stroke errors, artifacts and blurriness. To address these issues, we propose DA-Font, a novel framework which integrates a Dual-Attention Hybrid Module (DAHM). Specifically, we introduce two synergistic attention blocks: the component attention block that leverages component information from content images to guide the style transfer process, and the relation attention block that further refines spatial relationships through interacting the content feature with both original and stylized component-wise representations. These two blocks collaborate to preserve accurate character shapes and stylistic textures. Moreover, we also design a corner consistency loss and an elastic mesh feature loss to better improve geometric alignment. Extensive experiments show that our DA-Font outperforms the state-of-the-art methods across diverse font styles and characters, demonstrating its effectiveness in enhancing structural integrity and local fidelity. The source code can be found at \href{https://github.com/wrchen2001/DA-Font}{\textit{https://github.com/wrchen2001/DA-Font}}.
>
---
#### [new 130] CommonForms: A Large, Diverse Dataset for Form Field Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出了CommonForms，一个用于表单字段检测的大规模数据集，并开发了FFDNet模型。任务是识别页面中的表单字段（如文本框、复选框等）。工作包括数据集构建、模型训练与评估，实现了高精度且低成本的检测方法。**

- **链接: [http://arxiv.org/pdf/2509.16506v1](http://arxiv.org/pdf/2509.16506v1)**

> **作者:** Joe Barrow
>
> **摘要:** This paper introduces CommonForms, a web-scale dataset for form field detection. It casts the problem of form field detection as object detection: given an image of a page, predict the location and type (Text Input, Choice Button, Signature) of form fields. The dataset is constructed by filtering Common Crawl to find PDFs that have fillable elements. Starting with 8 million documents, the filtering process is used to arrive at a final dataset of roughly 55k documents that have over 450k pages. Analysis shows that the dataset contains a diverse mixture of languages and domains; one third of the pages are non-English, and among the 14 classified domains, no domain makes up more than 25% of the dataset. In addition, this paper presents a family of form field detectors, FFDNet-Small and FFDNet-Large, which attain a very high average precision on the CommonForms test set. Each model cost less than $500 to train. Ablation results show that high-resolution inputs are crucial for high-quality form field detection, and that the cleaning process improves data efficiency over using all PDFs that have fillable fields in Common Crawl. A qualitative analysis shows that they outperform a popular, commercially available PDF reader that can prepare forms. Unlike the most popular commercially available solutions, FFDNet can predict checkboxes in addition to text and signature fields. This is, to our knowledge, the first large scale dataset released for form field detection, as well as the first open source models. The dataset, models, and code will be released at https://github.com/jbarrow/commonforms
>
---
#### [new 131] Visual Instruction Pretraining for Domain-Specific Foundation Models
- **分类: cs.CV**

- **简介: 该论文提出视觉指令预训练（ViTP），通过语言推理增强视觉感知，解决基础模型在下游领域中感知与推理闭环不足的问题。采用视觉Transformer，在遥感和医学影像任务中取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.17562v1](http://arxiv.org/pdf/2509.17562v1)**

> **作者:** Yuxuan Li; Yicheng Zhang; Wenhao Tang; Yimian Dai; Ming-Ming Cheng; Xiang Li; Jian Yang
>
> **摘要:** Modern computer vision is converging on a closed loop in which perception, reasoning and generation mutually reinforce each other. However, this loop remains incomplete: the top-down influence of high-level reasoning on the foundational learning of low-level perceptual features is not yet underexplored. This paper addresses this gap by proposing a new paradigm for pretraining foundation models in downstream domains. We introduce Visual insTruction Pretraining (ViTP), a novel approach that directly leverages reasoning to enhance perception. ViTP embeds a Vision Transformer (ViT) backbone within a Vision-Language Model and pretrains it end-to-end using a rich corpus of visual instruction data curated from target downstream domains. ViTP is powered by our proposed Visual Robustness Learning (VRL), which compels the ViT to learn robust and domain-relevant features from a sparse set of visual tokens. Extensive experiments on 16 challenging remote sensing and medical imaging benchmarks demonstrate that ViTP establishes new state-of-the-art performance across a diverse range of downstream tasks. The code is available at github.com/zcablii/ViTP.
>
---
#### [new 132] Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment
- **分类: cs.CV; cs.LG**

- **简介: 该论文聚焦于**自动疼痛评估**任务，旨在解决数据不平衡和生成模型控制精度不足的问题。提出了**3DPain**（大规模合成数据集）和**ViTPain**（基于视觉Transformer的跨模态蒸馏框架），通过可控的3D人脸生成与热图标注提升评估的准确性与临床可靠性。**

- **链接: [http://arxiv.org/pdf/2509.16727v1](http://arxiv.org/pdf/2509.16727v1)**

> **作者:** Xin Lei Lin; Soroush Mehraban; Abhishek Moturu; Babak Taati
>
> **摘要:** Automated pain assessment from facial expressions is crucial for non-communicative patients, such as those with dementia. Progress has been limited by two challenges: (i) existing datasets exhibit severe demographic and label imbalance due to ethical constraints, and (ii) current generative models cannot precisely control facial action units (AUs), facial structure, or clinically validated pain levels. We present 3DPain, a large-scale synthetic dataset specifically designed for automated pain assessment, featuring unprecedented annotation richness and demographic diversity. Our three-stage framework generates diverse 3D meshes, textures them with diffusion models, and applies AU-driven face rigging to synthesize multi-view faces with paired neutral and pain images, AU configurations, PSPI scores, and the first dataset-level annotations of pain-region heatmaps. The dataset comprises 82,500 samples across 25,000 pain expression heatmaps and 2,500 synthetic identities balanced by age, gender, and ethnicity. We further introduce ViTPain, a Vision Transformer based cross-modal distillation framework in which a heatmap-trained teacher guides a student trained on RGB images, enhancing accuracy, interpretability, and clinical reliability. Together, 3DPain and ViTPain establish a controllable, diverse, and clinically grounded foundation for generalizable automated pain assessment.
>
---
#### [new 133] Visual Detector Compression via Location-Aware Discriminant Analysis
- **分类: cs.CV**

- **简介: 该论文针对目标检测模型的压缩问题，提出一种基于位置感知判别分析的主动压缩方法。通过最大化检测相关判别信息并结合定位信息，有效剪枝冗余神经元/滤波器，提升模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2509.17968v1](http://arxiv.org/pdf/2509.17968v1)**

> **作者:** Qizhen Lan; Jung Im Choi; Qing Tian
>
> **摘要:** Deep neural networks are powerful, yet their high complexity greatly limits their potential to be deployed on billions of resource-constrained edge devices. Pruning is a crucial network compression technique, yet most existing methods focus on classification models, with limited attention to detection. Even among those addressing detection, there is a lack of utilization of essential localization information. Also, many pruning methods passively rely on pre-trained models, in which useful and useless components are intertwined, making it difficult to remove the latter without harming the former at the neuron/filter level. To address the above issues, in this paper, we propose a proactive detection-discriminants-based network compression approach for deep visual detectors, which alternates between two steps: (1) maximizing and compressing detection-related discriminants and aligning them with a subset of neurons/filters immediately before the detection head, and (2) tracing the detection-related discriminating power across the layers and discarding features of lower importance. Object location information is exploited in both steps. Extensive experiments, employing four advanced detection models and four state-of-the-art competing methods on the KITTI and COCO datasets, highlight the superiority of our approach. Remarkably, our compressed models can even beat the original base models with a substantial reduction in complexity.
>
---
#### [new 134] A$^2$M$^2$-Net: Adaptively Aligned Multi-Scale Moment for Few-Shot Action Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对少样本动作识别任务，提出A$^2$M$^2$-Net方法，通过自适应对齐和多尺度二阶矩模块，有效建模视频动态特征，解决时间错位问题，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.17638v1](http://arxiv.org/pdf/2509.17638v1)**

> **作者:** Zilin Gao; Qilong Wang; Bingbing Zhang; Qinghua Hu; Peihua Li
>
> **备注:** 27 pages, 13 figures, 7 tables
>
> **摘要:** Thanks to capability to alleviate the cost of large-scale annotation, few-shot action recognition (FSAR) has attracted increased attention of researchers in recent years. Existing FSAR approaches typically neglect the role of individual motion pattern in comparison, and under-explore the feature statistics for video dynamics. Thereby, they struggle to handle the challenging temporal misalignment in video dynamics, particularly by using 2D backbones. To overcome these limitations, this work proposes an adaptively aligned multi-scale second-order moment network, namely A$^2$M$^2$-Net, to describe the latent video dynamics with a collection of powerful representation candidates and adaptively align them in an instance-guided manner. To this end, our A$^2$M$^2$-Net involves two core components, namely, adaptive alignment (A$^2$ module) for matching, and multi-scale second-order moment (M$^2$ block) for strong representation. Specifically, M$^2$ block develops a collection of semantic second-order descriptors at multiple spatio-temporal scales. Furthermore, A$^2$ module aims to adaptively select informative candidate descriptors while considering the individual motion pattern. By such means, our A$^2$M$^2$-Net is able to handle the challenging temporal misalignment problem by establishing an adaptive alignment protocol for strong representation. Notably, our proposed method generalizes well to various few-shot settings and diverse metrics. The experiments are conducted on five widely used FSAR benchmarks, and the results show our A$^2$M$^2$-Net achieves very competitive performance compared to state-of-the-arts, demonstrating its effectiveness and generalization.
>
---
#### [new 135] DepTR-MOT: Unveiling the Potential of Depth-Informed Trajectory Refinement for Multi-Object Tracking
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文针对多目标跟踪（MOT）任务，旨在解决2D方法在遮挡和密集场景下的跟踪不稳定性问题。提出DepTR-MOT，通过引入实例级深度信息，提升轨迹鲁棒性，并在机器人跟踪数据集上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.17323v1](http://arxiv.org/pdf/2509.17323v1)**

> **作者:** Buyin Deng; Lingxin Huang; Kai Luo; Fei Teng; Kailun Yang
>
> **备注:** The source code will be made publicly available at https://github.com/warriordby/DepTR-MOT
>
> **摘要:** Visual Multi-Object Tracking (MOT) is a crucial component of robotic perception, yet existing Tracking-By-Detection (TBD) methods often rely on 2D cues, such as bounding boxes and motion modeling, which struggle under occlusions and close-proximity interactions. Trackers relying on these 2D cues are particularly unreliable in robotic environments, where dense targets and frequent occlusions are common. While depth information has the potential to alleviate these issues, most existing MOT datasets lack depth annotations, leading to its underexploited role in the domain. To unveil the potential of depth-informed trajectory refinement, we introduce DepTR-MOT, a DETR-based detector enhanced with instance-level depth information. Specifically, we propose two key innovations: (i) foundation model-based instance-level soft depth label supervision, which refines depth prediction, and (ii) the distillation of dense depth maps to maintain global depth consistency. These strategies enable DepTR-MOT to output instance-level depth during inference, without requiring foundation models and without additional computational cost. By incorporating depth cues, our method enhances the robustness of the TBD paradigm, effectively resolving occlusion and close-proximity challenges. Experiments on both the QuadTrack and DanceTrack datasets demonstrate the effectiveness of our approach, achieving HOTA scores of 27.59 and 44.47, respectively. In particular, results on QuadTrack, a robotic platform MOT dataset, highlight the advantages of our method in handling occlusion and close-proximity challenges in robotic tracking. The source code will be made publicly available at https://github.com/warriordby/DepTR-MOT.
>
---
#### [new 136] ISCS: Parameter-Guided Channel Ordering and Grouping for Learned Image Compression
- **分类: cs.CV**

- **简介: 该论文针对学习型图像压缩任务，旨在提升编码效率和计算效率。通过分析预训练VAE模型中通道的重要性，提出了一种通用的通道排序与分组方法ISCS，减少冗余并提高比特率效率，无需依赖数据集特定实验。**

- **链接: [http://arxiv.org/pdf/2509.16853v1](http://arxiv.org/pdf/2509.16853v1)**

> **作者:** Jinhao Wang; Cihan Ruan; Nam Ling; Wei Wang; Wei Jiang
>
> **摘要:** Prior studies in learned image compression (LIC) consistently show that only a small subset of latent channels is critical for reconstruction, while many others carry limited information. Exploiting this imbalance could improve both coding and computational efficiency, yet existing approaches often rely on costly, dataset-specific ablation tests and typically analyze channels in isolation, ignoring their interdependencies. We propose a generalizable, dataset-agnostic method to identify and organize important channels in pretrained VAE-based LIC models. Instead of brute-force empirical evaluations, our approach leverages intrinsic parameter statistics-weight variances, bias magnitudes, and pairwise correlations-to estimate channel importance. This analysis reveals a consistent organizational structure, termed the Invariant Salient Channel Space (ISCS), where Salient-Core channels capture dominant structures and Salient-Auxiliary channels provide complementary details. Building on ISCS, we introduce a deterministic channel ordering and grouping strategy that enables slice-parallel decoding, reduces redundancy, and improves bitrate efficiency. Experiments across multiple LIC architectures demonstrate that our method effectively reduces bitrate and computation while maintaining reconstruction quality, providing a practical and modular enhancement to existing learned compression frameworks.
>
---
#### [new 137] CAMBench-QR : A Structure-Aware Benchmark for Post-Hoc Explanations with QR Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CAMBench-QR，一个用于评估视觉解释结构感知能力的基准，利用QR码的几何特征测试CAM方法是否关注关键结构并避免背景干扰，解决了现有解释方法表面合理但结构不忠实的问题。**

- **链接: [http://arxiv.org/pdf/2509.16745v1](http://arxiv.org/pdf/2509.16745v1)**

> **作者:** Ritabrata Chakraborty; Avijit Dasgupta; Sandeep Chaurasia
>
> **备注:** 9 pages, 5 figures, 6 tables
>
> **摘要:** Visual explanations are often plausible but not structurally faithful. We introduce CAMBench-QR, a structure-aware benchmark that leverages the canonical geometry of QR codes (finder patterns, timing lines, module grid) to test whether CAM methods place saliency on requisite substructures while avoiding background. CAMBench-QR synthesizes QR/non-QR data with exact masks and controlled distortions, and reports structure-aware metrics (Finder/Timing Mass Ratios, Background Leakage, coverage AUCs, Distance-to-Structure) alongside causal occlusion, insertion/deletion faithfulness, robustness, and latency. We benchmark representative, efficient CAMs (LayerCAM, EigenGrad-CAM, XGrad-CAM) under two practical regimes of zero-shot and last-block fine-tuning. The benchmark, metrics, and training recipes provide a simple, reproducible yardstick for structure-aware evaluation of visual explanations. Hence we propose that CAMBENCH-QR can be used as a litmus test of whether visual explanations are truly structure-aware.
>
---
#### [new 138] Unlocking Hidden Potential in Point Cloud Networks with Attention-Guided Grouping-Feature Coordination
- **分类: cs.CV**

- **简介: 该论文针对点云分析任务，旨在提升传统点云网络的性能。提出GF-Core模块协调分组与特征提取，并设计自监督预训练策略。在保持结构简洁的前提下，显著提升了分类精度。**

- **链接: [http://arxiv.org/pdf/2509.16639v1](http://arxiv.org/pdf/2509.16639v1)**

> **作者:** Shangzhuo Xie; Qianqian Yang
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Point cloud analysis has evolved with diverse network architectures, while existing works predominantly focus on introducing novel structural designs. However, conventional point-based architectures - processing raw points through sequential sampling, grouping, and feature extraction layers - demonstrate underutilized potential. We notice that substantial performance gains can be unlocked through strategic module integration rather than structural modifications. In this paper, we propose the Grouping-Feature Coordination Module (GF-Core), a lightweight separable component that simultaneously regulates both grouping layer and feature extraction layer to enable more nuanced feature aggregation. Besides, we introduce a self-supervised pretraining strategy specifically tailored for point-based inputs to enhance model robustness in complex point cloud analysis scenarios. On ModelNet40 dataset, our method elevates baseline networks to 94.0% accuracy, matching advanced frameworks' performance while preserving architectural simplicity. On three variants of the ScanObjectNN dataset, we obtain improvements of 2.96%, 6.34%, and 6.32% respectively.
>
---
#### [new 139] AutoArabic: A Three-Stage Framework for Localizing Video-Text Retrieval Benchmarks
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出AutoArabic框架，用于将英文视频-文本检索基准（如DiDeMo）翻译成阿拉伯语，解决阿拉伯语资源不足的问题。利用大语言模型实现自动翻译和错误检测，并评估不同后期编辑预算对性能的影响。**

- **链接: [http://arxiv.org/pdf/2509.16438v1](http://arxiv.org/pdf/2509.16438v1)**

> **作者:** Mohamed Eltahir; Osamah Sarraj; Abdulrahman Alfrihidi; Taha Alshatiri; Mohammed Khurd; Mohammed Bremoo; Tanveer Hussain
>
> **备注:** Accepted at ArabicNLP 2025 (EMNLP 2025 workshop)
>
> **摘要:** Video-to-text and text-to-video retrieval are dominated by English benchmarks (e.g. DiDeMo, MSR-VTT) and recent multilingual corpora (e.g. RUDDER), yet Arabic remains underserved, lacking localized evaluation metrics. We introduce a three-stage framework, AutoArabic, utilizing state-of-the-art large language models (LLMs) to translate non-Arabic benchmarks into Modern Standard Arabic, reducing the manual revision required by nearly fourfold. The framework incorporates an error detection module that automatically flags potential translation errors with 97% accuracy. Applying the framework to DiDeMo, a video retrieval benchmark produces DiDeMo-AR, an Arabic variant with 40,144 fluent Arabic descriptions. An analysis of the translation errors is provided and organized into an insightful taxonomy to guide future Arabic localization efforts. We train a CLIP-style baseline with identical hyperparameters on the Arabic and English variants of the benchmark, finding a moderate performance gap (about 3 percentage points at Recall@1), indicating that Arabic localization preserves benchmark difficulty. We evaluate three post-editing budgets (zero/ flagged-only/ full) and find that performance improves monotonically with more post-editing, while the raw LLM output (zero-budget) remains usable. To ensure reproducibility to other languages, we made the code available at https://github.com/Tahaalshatiri/AutoArabic.
>
---
#### [new 140] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SD-VLM，旨在提升视觉语言模型（VLM）的3D空间理解能力。针对2D图像空间表征不足的问题，研究构建了包含700K问答对的MSMU数据集，并引入深度位置编码方法，显著提升了模型在空间测量和推理任务中的性能。**

- **链接: [http://arxiv.org/pdf/2509.17664v1](http://arxiv.org/pdf/2509.17664v1)**

> **作者:** Pingyi Chen; Yujing Lou; Shen Cao; Jinhui Guo; Lubin Fan; Yue Wu; Lin Yang; Lizhuang Ma; Jieping Ye
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** While vision language models (VLMs) excel in 2D semantic visual understanding, their ability to quantitatively reason about 3D spatial relationships remains under-explored, due to the deficiency of 2D images' spatial representation ability. In this paper, we analyze the problem hindering VLMs' spatial understanding abilities and propose SD-VLM, a novel framework that significantly enhances fundamental spatial perception abilities of VLMs through two key contributions: (1) propose Massive Spatial Measuring and Understanding (MSMU) dataset with precise spatial annotations, and (2) introduce a simple depth positional encoding method strengthening VLMs' spatial awareness. MSMU dataset covers massive quantitative spatial tasks with 700K QA pairs, 2.5M physical numerical annotations, and 10K chain-of-thought augmented samples. We have trained SD-VLM, a strong generalist VLM which shows superior quantitative spatial measuring and understanding capability. SD-VLM not only achieves state-of-the-art performance on our proposed MSMU-Bench, but also shows spatial generalization abilities on other spatial understanding benchmarks including Q-Spatial and SpatialRGPT-Bench. Extensive experiments demonstrate that SD-VLM outperforms GPT-4o and Intern-VL3-78B by 26.91% and 25.56% respectively on MSMU-Bench. Code and models are released at https://github.com/cpystan/SD-VLM.
>
---
#### [new 141] Cross-Corpus and Cross-domain Handwriting Assessment of NeuroDegenerative Diseases via Time-Series-to-Image Conversion
- **分类: cs.CV**

- **简介: 该论文属于神经退行性疾病的手写评估任务，旨在解决跨数据集和跨模态（时间序列与图像）的模型泛化问题。提出结合时间序列和图像的联合分类框架，基于ResNet50实现高精度检测PD等疾病，尤其在绘图和书写任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.16474v1](http://arxiv.org/pdf/2509.16474v1)**

> **作者:** Gabrielle Chavez; Laureano Moro-Velazquez; Ankur Butala; Najim Dehak; Thomas Thebaud
>
> **备注:** 5 pages, 2 figures, submitted to International Conference on Acoustics, Speech, and Signal Processing (ICASSP)
>
> **摘要:** Handwriting is significantly affected by neurological disorders (ND) such as Parkinson's disease (PD) and Alzheimer's disease (AD). Prior works have analyzed handwriting tasks using feature-based approaches or computer-vision techniques, but these methods have struggled to generalize across multiple datasets, particularly between temporal features represented as time-series and images. We propose a framework that leverages both time-series and images of handwriting through a joint classifier, based on a ResNet50 pretrained on ImageNet-1k. Binary classification experiments demonstrate state-of-the-art performances on existing time-series and image datasets, with significant improvement on specific drawing and writing tasks from the NeuroLogical Signals (NLS) dataset. In particular, the proposed model demonstrates improved performance on Draw Clock and Spiral tasks. Additionally, cross-dataset and multi-dataset experiments were consistently able to achieve high F1 scores, up to 98 for PD detection, highlighting the potential of the proposed model to generalize over different forms of handwriting signals, and enhance the detection of motor deficits in ND.
>
---
#### [new 142] Introducing Resizable Region Packing Problem in Image Generation, with a Heuristic Solution
- **分类: cs.CV**

- **简介: 该论文提出图像生成中的可调整锚定区域装箱（RARP）问题，属于合成数据生成任务。针对如何在图像画布中合理放置任意形状和大小的区域，设计了一种贪心启发式算法，并验证其有效性，用于生成大规模异常检测数据集。**

- **链接: [http://arxiv.org/pdf/2509.16363v1](http://arxiv.org/pdf/2509.16363v1)**

> **作者:** Hrishikesh Sharma
>
> **摘要:** The problem of image data generation in computer vision has traditionally been a harder problem to solve, than discriminative problems. Such data generation entails placing relevant objects of appropriate sizes each, at meaningful location in a scene canvas. There have been two classes of popular approaches to such generation: graphics based, and generative models-based. Optimization problems are known to lurk in the background for both these classes of approaches. In this paper, we introduce a novel, practically useful manifestation of the classical Bin Packing problem in the context of generation of synthetic image data. We conjecture that the newly introduced problem, Resizable Anchored Region Packing(RARP) Problem, is NP-hard, and provide detailed arguments about our conjecture. As a first solution, we present a novel heuristic algorithm that is generic enough and therefore scales and packs arbitrary number of arbitrary-shaped regions at arbitrary locations, into an image canvas. The algorithm follows greedy approach to iteratively pack region pairs in a careful way, while obeying the optimization constraints. The algorithm is validated by an implementation that was used to generate a large-scale synthetic anomaly detection dataset, with highly varying degree of bin packing parameters per image sample i.e. RARP instance. Visual inspection of such data and checking of the correctness of each solution proves the effectiveness of our algorithm. With generative modeling being on rise in deep learning, and synthetic data generation poised to become mainstream, we expect that the newly introduced problem will be valued in the imaging scientific community.
>
---
#### [new 143] StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对水下立体深度估计任务，旨在解决领域适应与多模态融合问题。提出StereoAdapter框架，通过参数高效的LoRA方法和递归立体优化模块，在无大量标注数据情况下提升水下3D重建精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16415v1](http://arxiv.org/pdf/2509.16415v1)**

> **作者:** Zhengri Wu; Yiran Wang; Yu Wen; Zeyu Zhang; Biao Wu; Hao Tang
>
> **摘要:** Underwater stereo depth estimation provides accurate 3D geometry for robotics tasks such as navigation, inspection, and mapping, offering metric depth from low-cost passive cameras while avoiding the scale ambiguity of monocular methods. However, existing approaches face two critical challenges: (i) parameter-efficiently adapting large vision foundation encoders to the underwater domain without extensive labeled data, and (ii) tightly fusing globally coherent but scale-ambiguous monocular priors with locally metric yet photometrically fragile stereo correspondences. To address these challenges, we propose StereoAdapter, a parameter-efficient self-supervised framework that integrates a LoRA-adapted monocular foundation encoder with a recurrent stereo refinement module. We further introduce dynamic LoRA adaptation for efficient rank selection and pre-training on the synthetic UW-StereoDepth-40K dataset to enhance robustness under diverse underwater conditions. Comprehensive evaluations on both simulated and real-world benchmarks show improvements of 6.11% on TartanAir and 5.12% on SQUID compared to state-of-the-art methods, while real-world deployment with the BlueROV2 robot further demonstrates the consistent robustness of our approach. Code: https://github.com/AIGeeksGroup/StereoAdapter. Website: https://aigeeksgroup.github.io/StereoAdapter.
>
---
#### [new 144] Text-Scene: A Scene-to-Language Parsing Framework for 3D Scene Understanding
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Text-Scene框架，用于3D场景理解任务。针对3D场景语言解析数据不足和复杂性问题，设计自动将3D场景转为文本描述的方法，并构建InPlan3D基准测试3D任务规划能力。**

- **链接: [http://arxiv.org/pdf/2509.16721v1](http://arxiv.org/pdf/2509.16721v1)**

> **作者:** Haoyuan Li; Rui Liu; Hehe Fan; Yi Yang
>
> **备注:** 19 pages, 12 figures, 6 tables
>
> **摘要:** Enabling agents to understand and interact with complex 3D scenes is a fundamental challenge for embodied artificial intelligence systems. While Multimodal Large Language Models (MLLMs) have achieved significant progress in 2D image understanding, extending such capabilities to 3D scenes remains difficult: 1) 3D environment involves richer concepts such as spatial relationships, affordances, physics, layout, and so on, 2) the absence of large-scale 3D vision-language datasets has posed a significant obstacle. In this paper, we introduce Text-Scene, a framework that automatically parses 3D scenes into textual descriptions for scene understanding. Given a 3D scene, our model identifies object attributes and spatial relationships, and then generates a coherent summary of the whole scene, bridging the gap between 3D observation and language without requiring human-in-the-loop intervention. By leveraging both geometric analysis and MLLMs, Text-Scene produces descriptions that are accurate, detailed, and human-interpretable, capturing object-level details and global-level context. Experimental results on benchmarks demonstrate that our textual parses can faithfully represent 3D scenes and benefit downstream tasks. To evaluate the reasoning capability of MLLMs, we present InPlan3D, a comprehensive benchmark for 3D task planning, consisting of 3174 long-term planning tasks across 636 indoor scenes. We emphasize clarity and accessibility in our approach, aiming to make 3D scene content understandable through language. Code and datasets will be released.
>
---
#### [new 145] HyPlaneHead: Rethinking Tri-plane-like Representations in Full-Head Image Synthesis
- **分类: cs.CV**

- **简介: 该论文针对全头图像合成任务中Tri-plane表示存在的特征纠缠、映射不均和通道干扰问题，提出HyPlaneHead方法。通过引入结合平面与球面优势的Hy-Plane表示，并优化特征映射策略与生成方式，有效提升了图像质量和生成效率。**

- **链接: [http://arxiv.org/pdf/2509.16748v1](http://arxiv.org/pdf/2509.16748v1)**

> **作者:** Heyuan Li; Kenkun Liu; Lingteng Qiu; Qi Zuo; Keru Zheng; Zilong Dong; Xiaoguang Han
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Tri-plane-like representations have been widely adopted in 3D-aware GANs for head image synthesis and other 3D object/scene modeling tasks due to their efficiency. However, querying features via Cartesian coordinate projection often leads to feature entanglement, which results in mirroring artifacts. A recent work, SphereHead, attempted to address this issue by introducing spherical tri-planes based on a spherical coordinate system. While it successfully mitigates feature entanglement, SphereHead suffers from uneven mapping between the square feature maps and the spherical planes, leading to inefficient feature map utilization during rendering and difficulties in generating fine image details. Moreover, both tri-plane and spherical tri-plane representations share a subtle yet persistent issue: feature penetration across convolutional channels can cause interference between planes, particularly when one plane dominates the others. These challenges collectively prevent tri-plane-based methods from reaching their full potential. In this paper, we systematically analyze these problems for the first time and propose innovative solutions to address them. Specifically, we introduce a novel hybrid-plane (hy-plane for short) representation that combines the strengths of both planar and spherical planes while avoiding their respective drawbacks. We further enhance the spherical plane by replacing the conventional theta-phi warping with a novel near-equal-area warping strategy, which maximizes the effective utilization of the square feature map. In addition, our generator synthesizes a single-channel unified feature map instead of multiple feature maps in separate channels, thereby effectively eliminating feature penetration. With a series of technical improvements, our hy-plane representation enables our method, HyPlaneHead, to achieve state-of-the-art performance in full-head image synthesis.
>
---
#### [new 146] When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using Small VLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视觉问答（VQA）任务，旨在解决小规模视觉语言模型（S-VLMs）性能不足的问题。提出Model Parity Aligner（MPA），通过无标签数据和知识迁移策略，有效缩小S-VLMs与大规模模型间的性能差距，提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.16633v1](http://arxiv.org/pdf/2509.16633v1)**

> **作者:** Abhirama Subramanyam Penamakuri; Navlika Singh; Piyush Arora; Anand Mishra
>
> **备注:** Accepted to EMNLP (Main) 2025
>
> **摘要:** Large Vision-Language Models (L-VLMs) have demonstrated remarkable performance in various vision and language tasks, including visual question answering (VQA). However, their high computational cost makes them impractical for resource-constrained settings and inference-heavy applications. In contrast, Small Vision-Language Models (S-VLMs) offer efficiency but suffer from a significant performance gap compared to their larger counterparts. In this work, we introduce the Model Parity Aligner (MPA), a novel framework designed to systematically improve S-VLMs by leveraging unlabeled images and effective knowledge transfer from L-VLMs. Instead of traditional knowledge distillation methods that rely on labeled training data, MPA employs a strategic parity-based approach that precisely identifies the knowledge disparities between S-VLMs and L-VLMs, and optimizes training by targeting only these disparities. We conduct extensive experiments on four diverse VQA benchmarks, namely TextVQA, ST-VQA, ChartQA, and OKVQA, each of which requires specialized reasoning capabilities such as text recognition, chart interpretation, and commonsense and factual understanding. Our results demonstrate that MPA consistently enhances the performance of S-VLMs on all benchmarks, reducing the performance gap while maintaining computational efficiency. We make our code publicly available.
>
---
#### [new 147] Beyond Diagnosis: Evaluating Multimodal LLMs for Pathology Localization in Chest Radiographs
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究了多模态大语言模型（GPT-4、GPT-5、MedGemma）在胸部X光片中病变定位任务的表现。通过空间网格提示方法，评估模型对九种病变的定位准确性，发现其性能低于专用CNN和放射科医生基准，揭示了当前模型在医学影像中的潜力与局限性。**

- **链接: [http://arxiv.org/pdf/2509.18015v1](http://arxiv.org/pdf/2509.18015v1)**

> **作者:** Advait Gosai; Arun Kavishwar; Stephanie L. McNamara; Soujanya Samineni; Renato Umeton; Alexander Chowdhury; William Lotter
>
> **摘要:** Recent work has shown promising performance of frontier large language models (LLMs) and their multimodal counterparts in medical quizzes and diagnostic tasks, highlighting their potential for broad clinical utility given their accessible, general-purpose nature. However, beyond diagnosis, a fundamental aspect of medical image interpretation is the ability to localize pathological findings. Evaluating localization not only has clinical and educational relevance but also provides insight into a model's spatial understanding of anatomy and disease. Here, we systematically assess two general-purpose MLLMs (GPT-4 and GPT-5) and a domain-specific model (MedGemma) in their ability to localize pathologies on chest radiographs, using a prompting pipeline that overlays a spatial grid and elicits coordinate-based predictions. Averaged across nine pathologies in the CheXlocalize dataset, GPT-5 exhibited a localization accuracy of 49.7%, followed by GPT-4 (39.1%) and MedGemma (17.7%), all lower than a task-specific CNN baseline (59.9%) and a radiologist benchmark (80.1%). Despite modest performance, error analysis revealed that GPT-5's predictions were largely in anatomically plausible regions, just not always precisely localized. GPT-4 performed well on pathologies with fixed anatomical locations, but struggled with spatially variable findings and exhibited anatomically implausible predictions more frequently. MedGemma demonstrated the lowest performance on all pathologies, showing limited capacity to generalize to this novel task. Our findings highlight both the promise and limitations of current MLLMs in medical imaging and underscore the importance of integrating them with task-specific tools for reliable use.
>
---
#### [new 148] ME-Mamba: Multi-Expert Mamba with Efficient Knowledge Capture and Fusion for Multimodal Survival Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ME-Mamba，用于多模态生存分析任务。旨在解决病理图像与基因组数据融合中的信息丢失问题。设计了三个专家模块分别提取和融合多模态特征，实现高效、准确的癌症生存预测。**

- **链接: [http://arxiv.org/pdf/2509.16900v1](http://arxiv.org/pdf/2509.16900v1)**

> **作者:** Chengsheng Zhang; Linhao Qu; Xiaoyu Liu; Zhijian Song
>
> **摘要:** Survival analysis using whole-slide images (WSIs) is crucial in cancer research. Despite significant successes, pathology images typically only provide slide-level labels, which hinders the learning of discriminative representations from gigapixel WSIs. With the rapid advancement of high-throughput sequencing technologies, multimodal survival analysis integrating pathology images and genomics data has emerged as a promising approach. We propose a Multi-Expert Mamba (ME-Mamba) system that captures discriminative pathological and genomic features while enabling efficient integration of both modalities. This approach achieves complementary information fusion without losing critical information from individual modalities, thereby facilitating accurate cancer survival analysis. Specifically, we first introduce a Pathology Expert and a Genomics Expert to process unimodal data separately. Both experts are designed with Mamba architectures that incorporate conventional scanning and attention-based scanning mechanisms, allowing them to extract discriminative features from long instance sequences containing substantial redundant or irrelevant information. Second, we design a Synergistic Expert responsible for modality fusion. It explicitly learns token-level local correspondences between the two modalities via Optimal Transport, and implicitly enhances distribution consistency through a global cross-modal fusion loss based on Maximum Mean Discrepancy. The fused feature representations are then passed to a mamba backbone for further integration. Through the collaboration of the Pathology Expert, Genomics Expert, and Synergistic Expert, our method achieves stable and accurate survival analysis with relatively low computational complexity. Extensive experimental results on five datasets in The Cancer Genome Atlas (TCGA) demonstrate our state-of-the-art performance.
>
---
#### [new 149] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology
- **分类: cs.CV**

- **简介: 该论文提出一种结合语义分割图与局部组织图像的扩散模型，用于生成高保真且异质性强的病理图像。旨在解决合成数据中组织多样性不足和标注依赖问题，通过自监督方法扩展至无标注数据，显著提升下游分割任务性能。**

- **链接: [http://arxiv.org/pdf/2509.17847v1](http://arxiv.org/pdf/2509.17847v1)**

> **作者:** Saghir Alfasly; Wataru Uegami; MD Enamul Hoq; Ghazal Alabtah; H. R. Tizhoosh
>
> **备注:** NeurIPS 2025
>
> **摘要:** Synthetic data generation in histopathology faces unique challenges: preserving tissue heterogeneity, capturing subtle morphological features, and scaling to unannotated datasets. We present a latent diffusion model that generates realistic heterogeneous histopathology images through a novel dual-conditioning approach combining semantic segmentation maps with tissue-specific visual crops. Unlike existing methods that rely on text prompts or abstract visual embeddings, our approach preserves critical morphological details by directly incorporating raw tissue crops from corresponding semantic regions. For annotated datasets (i.e., Camelyon16, Panda), we extract patches ensuring 20-80% tissue heterogeneity. For unannotated data (i.e., TCGA), we introduce a self-supervised extension that clusters whole-slide images into 100 tissue types using foundation model embeddings, automatically generating pseudo-semantic maps for training. Our method synthesizes high-fidelity images with precise region-wise annotations, achieving superior performance on downstream segmentation tasks. When evaluated on annotated datasets, models trained on our synthetic data show competitive performance to those trained on real data, demonstrating the utility of controlled heterogeneous tissue generation. In quantitative evaluation, prompt-guided synthesis reduces Frechet Distance by up to 6X on Camelyon16 (from 430.1 to 72.0) and yields 2-3x lower FD across Panda and TCGA. Downstream DeepLabv3+ models trained solely on synthetic data attain test IoU of 0.71 and 0.95 on Camelyon16 and Panda, within 1-2% of real-data baselines (0.72 and 0.96). By scaling to 11,765 TCGA whole-slide images without manual annotations, our framework offers a practical solution for an urgent need for generating diverse, annotated histopathology data, addressing a critical bottleneck in computational pathology.
>
---
#### [new 150] SPFSplatV2: Efficient Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views
- **分类: cs.CV**

- **简介: 该论文提出SPFSplatV2，用于从稀疏视角图像高效生成3D高斯点，无需真实相机位姿。通过共享特征提取和掩码注意力机制，实现姿态自监督预测与几何重建，提升了新视角合成性能。**

- **链接: [http://arxiv.org/pdf/2509.17246v1](http://arxiv.org/pdf/2509.17246v1)**

> **作者:** Ranran Huang; Krystian Mikolajczyk
>
> **摘要:** We introduce SPFSplatV2, an efficient feed-forward framework for 3D Gaussian splatting from sparse multi-view images, requiring no ground-truth poses during training and inference. It employs a shared feature extraction backbone, enabling simultaneous prediction of 3D Gaussian primitives and camera poses in a canonical space from unposed inputs. A masked attention mechanism is introduced to efficiently estimate target poses during training, while a reprojection loss enforces pixel-aligned Gaussian primitives, providing stronger geometric constraints. We further demonstrate the compatibility of our training framework with different reconstruction architectures, resulting in two model variants. Remarkably, despite the absence of pose supervision, our method achieves state-of-the-art performance in both in-domain and out-of-domain novel view synthesis, even under extreme viewpoint changes and limited image overlap, and surpasses recent methods that rely on geometric supervision for relative pose estimation. By eliminating dependence on ground-truth poses, our method offers the scalability to leverage larger and more diverse datasets. Code and pretrained models will be available on our project page: https://ranrhuang.github.io/spfsplatv2/.
>
---
#### [new 151] Agentic Reasoning for Robust Vision Systems via Increased Test-Time Compute
- **分类: cs.CV; cs.AI; cs.MA**

- **简介: 该论文提出视觉推理代理（VRA），通过“思考-批评-行动”循环，提升现成视觉系统的鲁棒性与准确性，无需重新训练。适用于高风险视觉任务，如遥感和医疗诊断。**

- **链接: [http://arxiv.org/pdf/2509.16343v1](http://arxiv.org/pdf/2509.16343v1)**

> **作者:** Chung-En; Yu; Brian Jalaian; Nathaniel D. Bastian
>
> **摘要:** Developing trustworthy intelligent vision systems for high-stakes domains, \emph{e.g.}, remote sensing and medical diagnosis, demands broad robustness without costly retraining. We propose \textbf{Visual Reasoning Agent (VRA)}, a training-free, agentic reasoning framework that wraps off-the-shelf vision-language models \emph{and} pure vision systems in a \emph{Think--Critique--Act} loop. While VRA incurs significant additional test-time computation, it achieves up to 40\% absolute accuracy gains on challenging visual reasoning benchmarks. Future work will optimize query routing and early stopping to reduce inference overhead while preserving reliability in vision tasks.
>
---
#### [new 152] Automatic Intermodal Loading Unit Identification using Computer Vision: A Scoping Review
- **分类: cs.CV**

- **简介: 该论文综述了63项基于计算机视觉的多式联运装卸单元（ILUs）识别研究，旨在解决港口高效识别问题。回顾了1990-2025年的技术演进，并指出数据集缺乏导致性能差异大，呼吁标准化术语和开放数据以推动领域发展。**

- **链接: [http://arxiv.org/pdf/2509.17707v1](http://arxiv.org/pdf/2509.17707v1)**

> **作者:** Emre Gülsoylu; Alhassan Abdelhalim; Derya Kara Boztas; Ole Grasse; Carlos Jahn; Simone Frintrop; Janick Edinger
>
> **备注:** Submission to Transport Reviews. 36 pages, 2 figures, 4 tables
>
> **摘要:** The standardisation of Intermodal Loading Units (ILUs), such as containers, semi-trailers and swap bodies, has revolutionised global trade yet their efficient and robust identification remains a critical bottleneck in high-throughput ports and terminals. This paper reviews 63 empirical studies that propose computer vision (CV) based solutions. It covers the last 35 years (1990-2025), tracing the field's evolution from early digital image processing (DIP) and traditional machine learning (ML) to the current dominance of deep learning (DL) techniques. While CV offers cost-effective alternatives for other types of identification techniques, its development is hindered by the lack of publicly available benchmarking datasets. This results in high variance for the reported results such as end-to-end accuracy ranging from 5 % to 96 %. Beyond dataset limitations, this review highlights the emerging challenges especially introduced by the shift from character-based text recognition to scene-text spotting and the integration of mobile cameras (e.g. drones, sensor equipped ground vehicles) for dynamic terminal monitoring. To advance the field, the paper calls for standardised terminology, open-access datasets, shared source code, while outlining future research directions such as contextless text recognition optimised for ISO6346 codes.
>
---
#### [new 153] IPF-RDA: An Information-Preserving Framework for Robust Data Augmentation
- **分类: cs.CV**

- **简介: 该论文提出IPF-RDA，一种信息保留的数据增强框架，旨在解决数据增强引入分布偏移和噪声的问题。通过估计类别判别信息并自适应保留关键信息，提升多种数据增强方法的鲁棒性和性能。**

- **链接: [http://arxiv.org/pdf/2509.16678v1](http://arxiv.org/pdf/2509.16678v1)**

> **作者:** Suorong Yang; Hongchao Yang; Suhan Guo; Furao Shen; Jian Zhao
>
> **备注:** IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** Data augmentation is widely utilized as an effective technique to enhance the generalization performance of deep models. However, data augmentation may inevitably introduce distribution shifts and noises, which significantly constrain the potential and deteriorate the performance of deep networks. To this end, we propose a novel information-preserving framework, namely IPF-RDA, to enhance the robustness of data augmentations in this paper. IPF-RDA combines the proposal of (i) a new class-discriminative information estimation algorithm that identifies the points most vulnerable to data augmentation operations and corresponding importance scores; And (ii) a new information-preserving scheme that preserves the critical information in the augmented samples and ensures the diversity of augmented data adaptively. We divide data augmentation methods into three categories according to the operation types and integrate these approaches into our framework accordingly. After being integrated into our framework, the robustness of data augmentation methods can be enhanced and their full potential can be unleashed. Extensive experiments demonstrate that although being simple, IPF-RDA consistently improves the performance of numerous commonly used state-of-the-art data augmentation methods with popular deep models on a variety of datasets, including CIFAR-10, CIFAR-100, Tiny-ImageNet, CUHK03, Market1501, Oxford Flower, and MNIST, where its performance and scalability are stressed. The implementation is available at https://github.com/Jackbrocp/IPF-RDA.
>
---
#### [new 154] Dual-View Alignment Learning with Hierarchical-Prompt for Class-Imbalance Multi-Label Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对类别不平衡的多标签图像分类任务，提出HP-DVAL方法。通过双视角对齐学习和层次化提示策略，利用视觉-语言预训练模型缓解长尾与少样本问题，在MS-COCO和VOC2007数据集上取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2509.17747v1](http://arxiv.org/pdf/2509.17747v1)**

> **作者:** Sheng Huang; Jiexuan Yan; Beiyan Liu; Bo Liu; Richang Hong
>
> **备注:** accepted by IEEE Transactions on Image Processing
>
> **摘要:** Real-world datasets often exhibit class imbalance across multiple categories, manifesting as long-tailed distributions and few-shot scenarios. This is especially challenging in Class-Imbalanced Multi-Label Image Classification (CI-MLIC) tasks, where data imbalance and multi-object recognition present significant obstacles. To address these challenges, we propose a novel method termed Dual-View Alignment Learning with Hierarchical Prompt (HP-DVAL), which leverages multi-modal knowledge from vision-language pretrained (VLP) models to mitigate the class-imbalance problem in multi-label settings. Specifically, HP-DVAL employs dual-view alignment learning to transfer the powerful feature representation capabilities from VLP models by extracting complementary features for accurate image-text alignment. To better adapt VLP models for CI-MLIC tasks, we introduce a hierarchical prompt-tuning strategy that utilizes global and local prompts to learn task-specific and context-related prior knowledge. Additionally, we design a semantic consistency loss during prompt tuning to prevent learned prompts from deviating from general knowledge embedded in VLP models. The effectiveness of our approach is validated on two CI-MLIC benchmarks: MS-COCO and VOC2007. Extensive experimental results demonstrate the superiority of our method over SOTA approaches, achieving mAP improvements of 10.0\% and 5.2\% on the long-tailed multi-label image classification task, and 6.8\% and 2.9\% on the multi-label few-shot image classification task.
>
---
#### [new 155] The SAGES Critical View of Safety Challenge: A Global Benchmark for AI-Assisted Surgical Quality Assessment
- **分类: cs.CV; 68T07; I.2.10; J.3**

- **简介: 该论文介绍了SAGES Critical View of Safety挑战赛，旨在通过AI评估腹腔镜胆囊切除术中的安全操作质量。任务是推动AI在手术质量评估中的应用，解决性能、主观评估不确定性和临床变异性问题。工作包括构建标注数据集、开发EndoGlacier框架，并分析方法趋势以指导未来研究。**

- **链接: [http://arxiv.org/pdf/2509.17100v1](http://arxiv.org/pdf/2509.17100v1)**

> **作者:** Deepak Alapatt; Jennifer Eckhoff; Zhiliang Lyu; Yutong Ban; Jean-Paul Mazellier; Sarah Choksi; Kunyi Yang; 2024 CVS Challenge Consortium; Quanzheng Li; Filippo Filicori; Xiang Li; Pietro Mascagni; Daniel A. Hashimoto; Guy Rosman; Ozanan Meireles; Nicolas Padoy
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Advances in artificial intelligence (AI) for surgical quality assessment promise to democratize access to expertise, with applications in training, guidance, and accreditation. This study presents the SAGES Critical View of Safety (CVS) Challenge, the first AI competition organized by a surgical society, using the CVS in laparoscopic cholecystectomy, a universally recommended yet inconsistently performed safety step, as an exemplar of surgical quality assessment. A global collaboration across 54 institutions in 24 countries engaged hundreds of clinicians and engineers to curate 1,000 videos annotated by 20 surgical experts according to a consensus-validated protocol. The challenge addressed key barriers to real-world deployment in surgery, including achieving high performance, capturing uncertainty in subjective assessment, and ensuring robustness to clinical variability. To enable this scale of effort, we developed EndoGlacier, a framework for managing large, heterogeneous surgical video and multi-annotator workflows. Thirteen international teams participated, achieving up to a 17\% relative gain in assessment performance, over 80\% reduction in calibration error, and a 17\% relative improvement in robustness over the state-of-the-art. Analysis of results highlighted methodological trends linked to model performance, providing guidance for future research toward robust, clinically deployable AI for surgical quality assessment.
>
---
#### [new 156] Follow-Your-Emoji-Faster: Towards Efficient, Fine-Controllable, and Expressive Freestyle Portrait Animation
- **分类: cs.CV**

- **简介: 该论文提出Follow-Your-Emoji-Faster，一个高效的基于扩散的自由风格肖像动画框架。任务是通过面部关键点驱动肖像表情变化，解决身份保持、表情迁移和长期稳定性问题。方法包括表情感知关键点、细粒度面部损失及加速策略，提升生成效率与质量。**

- **链接: [http://arxiv.org/pdf/2509.16630v1](http://arxiv.org/pdf/2509.16630v1)**

> **作者:** Yue Ma; Zexuan Yan; Hongyu Liu; Hongfa Wang; Heng Pan; Yingqing He; Junkun Yuan; Ailing Zeng; Chengfei Cai; Heung-Yeung Shum; Zhifeng Li; Wei Liu; Linfeng Zhang; Qifeng Chen
>
> **备注:** accepted by IJCV2025. project page:https://follow-your-emoji.github.io
>
> **摘要:** We present Follow-Your-Emoji-Faster, an efficient diffusion-based framework for freestyle portrait animation driven by facial landmarks. The main challenges in this task are preserving the identity of the reference portrait, accurately transferring target expressions, and maintaining long-term temporal consistency while ensuring generation efficiency. To address identity preservation and accurate expression retargeting, we enhance Stable Diffusion with two key components: a expression-aware landmarks as explicit motion signals, which improve motion alignment, support exaggerated expressions, and reduce identity leakage; and a fine-grained facial loss that leverages both expression and facial masks to better capture subtle expressions and faithfully preserve the reference appearance. With these components, our model supports controllable and expressive animation across diverse portrait types, including real faces, cartoons, sculptures, and animals. However, diffusion-based frameworks typically struggle to efficiently generate long-term stable animation results, which remains a core challenge in this task. To address this, we propose a progressive generation strategy for stable long-term animation, and introduce a Taylor-interpolated cache, achieving a 2.6X lossless acceleration. These two strategies ensure that our method produces high-quality results efficiently, making it user-friendly and accessible. Finally, we introduce EmojiBench++, a more comprehensive benchmark comprising diverse portraits, driving videos, and landmark sequences. Extensive evaluations on EmojiBench++ demonstrate that Follow-Your-Emoji-Faster achieves superior performance in both animation quality and controllability. The code, training dataset and benchmark will be found in https://follow-your-emoji.github.io/.
>
---
#### [new 157] 4D-MoDe: Towards Editable and Scalable Volumetric Streaming via Motion-Decoupled 4D Gaussian Compression
- **分类: cs.CV**

- **简介: 该论文提出4D-MoDe，一种用于可编辑、可扩展的体积视频流的运动解耦4D高斯压缩方法。通过分离静态背景和动态前景，减少冗余并支持选择性流媒体传输，解决了大规模高质量动态体积内容存储与传输效率低的问题。**

- **链接: [http://arxiv.org/pdf/2509.17506v1](http://arxiv.org/pdf/2509.17506v1)**

> **作者:** Houqiang Zhong; Zihan Zheng; Qiang Hu; Yuan Tian; Ning Cao; Lan Xu; Xiaoyun Zhang; Zhengxue Cheng; Li Song; Wenjun Zhang
>
> **摘要:** Volumetric video has emerged as a key medium for immersive telepresence and augmented/virtual reality, enabling six-degrees-of-freedom (6DoF) navigation and realistic spatial interactions. However, delivering high-quality dynamic volumetric content at scale remains challenging due to massive data volume, complex motion, and limited editability of existing representations. In this paper, we present 4D-MoDe, a motion-decoupled 4D Gaussian compression framework designed for scalable and editable volumetric video streaming. Our method introduces a layered representation that explicitly separates static backgrounds from dynamic foregrounds using a lookahead-based motion decomposition strategy, significantly reducing temporal redundancy and enabling selective background/foreground streaming. To capture continuous motion trajectories, we employ a multi-resolution motion estimation grid and a lightweight shared MLP, complemented by a dynamic Gaussian compensation mechanism to model emergent content. An adaptive grouping scheme dynamically inserts background keyframes to balance temporal consistency and compression efficiency. Furthermore, an entropy-aware training pipeline jointly optimizes the motion fields and Gaussian parameters under a rate-distortion (RD) objective, while employing range-based and KD-tree compression to minimize storage overhead. Extensive experiments on multiple datasets demonstrate that 4D-MoDe consistently achieves competitive reconstruction quality with an order of magnitude lower storage cost (e.g., as low as \textbf{11.4} KB/frame) compared to state-of-the-art methods, while supporting practical applications such as background replacement and foreground-only streaming.
>
---
#### [new 158] PRISM: Precision-Recall Informed Data-Free Knowledge Distillation via Generative Diffusion
- **分类: cs.CV**

- **简介: 该论文提出PRISM，用于数据无关知识蒸馏任务。针对生成大规模图像时的模式崩溃和精度-召回挑战，引入能量引导分布对齐和多样化提示工程，提升合成数据质量与覆盖性，实现更有效的知识迁移和领域泛化。**

- **链接: [http://arxiv.org/pdf/2509.16897v1](http://arxiv.org/pdf/2509.16897v1)**

> **作者:** Xuewan He; Jielei Wang; Zihan Cheng; Yuchen Su; Shiyue Huang; Guoming Lu
>
> **摘要:** Data-free knowledge distillation (DFKD) transfers knowledge from a teacher to a student without access to the real in-distribution (ID) data. While existing methods perform well on small-scale images, they suffer from mode collapse when synthesizing large-scale images, resulting in limited knowledge transfer. Recently, leveraging advanced generative models to synthesize photorealistic images has emerged as a promising alternative. Nevertheless, directly using off-the-shelf diffusion to generate datasets faces the precision-recall challenges: 1) ensuring synthetic data aligns with the real distribution, and 2) ensuring coverage of the real ID manifold. In response, we propose PRISM, a precision-recall informed synthesis method. Specifically, we introduce Energy-guided Distribution Alignment to avoid the generation of out-of-distribution samples, and design the Diversified Prompt Engineering to enhance coverage of the real ID manifold. Extensive experiments on various large-scale image datasets demonstrate the superiority of PRISM. Moreover, we demonstrate that models trained with PRISM exhibit strong domain generalization.
>
---
#### [new 159] Point-RTD: Replaced Token Denoising for Pretraining Transformer Models on Point Clouds
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Point-RTD，一种用于点云预训练的替换去噪策略。不同于传统遮蔽重建方法，Point-RTD通过破坏点云token并利用生成器-判别器架构进行去噪，提升模型结构先验学习能力，在多个基准测试中显著优于PointMAE。**

- **链接: [http://arxiv.org/pdf/2509.17207v1](http://arxiv.org/pdf/2509.17207v1)**

> **作者:** Gunner Stone; Youngsook Choi; Alireza Tavakkoli; Ankita Shukla
>
> **摘要:** Pre-training strategies play a critical role in advancing the performance of transformer-based models for 3D point cloud tasks. In this paper, we introduce Point-RTD (Replaced Token Denoising), a novel pretraining strategy designed to improve token robustness through a corruption-reconstruction framework. Unlike traditional mask-based reconstruction tasks that hide data segments for later prediction, Point-RTD corrupts point cloud tokens and leverages a discriminator-generator architecture for denoising. This shift enables more effective learning of structural priors and significantly enhances model performance and efficiency. On the ShapeNet dataset, Point-RTD reduces reconstruction error by over 93% compared to PointMAE, and achieves more than 14x lower Chamfer Distance on the test set. Our method also converges faster and yields higher classification accuracy on ShapeNet, ModelNet10, and ModelNet40 benchmarks, clearly outperforming the baseline Point-MAE framework in every case.
>
---
#### [new 160] Multi-needle Localization for Pelvic Seed Implant Brachytherapy based on Tip-handle Detection and Matching
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文针对盆腔种子植入近距离治疗中多针定位难题，提出基于针尖-手柄检测与匹配的方法。采用HRNet构建无锚点网络检测针尖和手柄，并通过贪婪匹配合并算法重构3D针路径，在100例患者数据上表现优于分割方法。**

- **链接: [http://arxiv.org/pdf/2509.17931v1](http://arxiv.org/pdf/2509.17931v1)**

> **作者:** Zhuo Xiao; Fugen Zhou; Jingjing Wang; Chongyu He; Bo Liu; Haitao Sun; Zhe Ji; Yuliang Jiang; Junjie Wang; Qiuwen Wu
>
> **摘要:** Accurate multi-needle localization in intraoperative CT images is crucial for optimizing seed placement in pelvic seed implant brachytherapy. However, this task is challenging due to poor image contrast and needle adhesion. This paper presents a novel approach that reframes needle localization as a tip-handle detection and matching problem to overcome these difficulties. An anchor-free network, based on HRNet, is proposed to extract multi-scale features and accurately detect needle tips and handles by predicting their centers and orientations using decoupled branches for heatmap regression and polar angle prediction. To associate detected tips and handles into individual needles, a greedy matching and merging (GMM) method designed to solve the unbalanced assignment problem with constraints (UAP-C) is presented. The GMM method iteratively selects the most probable tip-handle pairs and merges them based on a distance metric to reconstruct 3D needle paths. Evaluated on a dataset of 100 patients, the proposed method demonstrates superior performance, achieving higher precision and F1 score compared to a segmentation-based method utilizing the nnUNet model,thereby offering a more robust and accurate solution for needle localization in complex clinical scenarios.
>
---
#### [new 161] Enhancing Scientific Visual Question Answering via Vision-Caption aware Supervised Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文聚焦科学视觉问答（VQA）任务，旨在提升小规模视觉语言模型的性能。提出VCASFT方法，结合图像标题与问答对进行指令微调，并构建HiSciVQA数据集及LLM评估方案，验证方法在多语言场景下的有效性。**

- **链接: [http://arxiv.org/pdf/2509.16628v1](http://arxiv.org/pdf/2509.16628v1)**

> **作者:** Janak Kapuriya; Anwar Shaikh; Arnav Goel; Medha Hira; Apoorv Singh; Jay Saraf; Sanjana; Vaibhav Nauriyal; Avinash Anand; Zhengkui Wang; Rajiv Ratn Shah
>
> **摘要:** In this study, we introduce Vision-Caption aware Supervised FineTuning (VCASFT), a novel learning paradigm designed to enhance the performance of smaller Vision Language Models(VLMs) on scientific visual question answering(VQA) tasks. VCASFT leverages image captions as zero-shot prompts alongside question-answer pairs and instruction-tunes models to yield significant performance improvements. To comprehensively evaluate VCASFT, we benchmark it on ScienceQA, which consists of questions across diverse languages, subjects, and fields, demonstrating its adaptability and effectiveness in a variety of educational contexts. Additionally, to further demonstrate the effectiveness of this technique on lowresource languages, we developed HiSciVQA, a dataset comprising 2,245 high-quality, hand-annotated Hindi multimodal Q&A pairs. This dataset addresses the critical need for low-resource language Q&A datasets and serves as a foundation for testing VCASFT. Additionally, we introduce a novel LLM-based evaluation scheme to evaluate VLMs on HiSciVQA which offers deeper insights into model effectiveness surpassing traditional n-gram matching accuracy metrics. We are committed to advancing the field by open-sourcing all code files and the HiSciVQA dataset for the research community.
>
---
#### [new 162] Chat-CBM: Towards Interactive Concept Bottleneck Models with Frozen Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出Chat-CBM，一种结合冻结大语言模型的交互式概念瓶颈模型。旨在解决传统CBM在用户干预和新知识融入上的局限性，通过语义分类器提升可解释性和交互性，实验证明其预测性能和用户互动性更优。**

- **链接: [http://arxiv.org/pdf/2509.17522v1](http://arxiv.org/pdf/2509.17522v1)**

> **作者:** Hangzhou He; Lei Zhu; Kaiwen Li; Xinliang Zhang; Jiakui Hu; Ourui Fu; Zhengjian Yao; Yanye Lu
>
> **摘要:** Concept Bottleneck Models (CBMs) provide inherent interpretability by first predicting a set of human-understandable concepts and then mapping them to labels through a simple classifier. While users can intervene in the concept space to improve predictions, traditional CBMs typically employ a fixed linear classifier over concept scores, which restricts interventions to manual value adjustments and prevents the incorporation of new concepts or domain knowledge at test time. These limitations are particularly severe in unsupervised CBMs, where concept activations are often noisy and densely activated, making user interventions ineffective. We introduce Chat-CBM, which replaces score-based classifiers with a language-based classifier that reasons directly over concept semantics. By grounding prediction in the semantic space of concepts, Chat-CBM preserves the interpretability of CBMs while enabling richer and more intuitive interventions, such as concept correction, addition or removal of concepts, incorporation of external knowledge, and high-level reasoning guidance. Leveraging the language understanding and few-shot capabilities of frozen large language models, Chat-CBM extends the intervention interface of CBMs beyond numerical editing and remains effective even in unsupervised settings. Experiments on nine datasets demonstrate that Chat-CBM achieves higher predictive performance and substantially improves user interactivity while maintaining the concept-based interpretability of CBMs.
>
---
#### [new 163] Emergent 3D Correspondence from Neural Shape Representation
- **分类: cs.CV**

- **简介: 该论文提出一种基于层次化神经语义表示（HNSR）的3D语义对应方法，任务是准确匹配3D形状的语义结构。通过结合全局特征与局部几何特征，并采用渐进式匹配策略，实现了无监督、跨类别的鲁棒对应，应用于分割、配准等任务。**

- **链接: [http://arxiv.org/pdf/2509.17431v1](http://arxiv.org/pdf/2509.17431v1)**

> **作者:** Keyu Du; Jingyu Hu; Haipeng Li; Hao Xu; Haibing Huang; Chi-Wing Fu; Shuaicheng Liu
>
> **备注:** This paper is accepted by Siggraph Asia 2025 conference track
>
> **摘要:** This paper presents a new approach to estimate accurate and robust 3D semantic correspondence with the hierarchical neural semantic representation. Our work has three key contributions. First, we design the hierarchical neural semantic representation (HNSR), which consists of a global semantic feature to capture high-level structure and multi-resolution local geometric features to preserve fine details, by carefully harnessing 3D priors from pre-trained 3D generative models. Second, we design a progressive global-to-local matching strategy, which establishes coarse semantic correspondence using the global semantic feature, then iteratively refines it with local geometric features, yielding accurate and semantically-consistent mappings. Third, our framework is training-free and broadly compatible with various pre-trained 3D generative backbones, demonstrating strong generalization across diverse shape categories. Our method also supports various applications, such as shape co-segmentation, keypoint matching, and texture transfer, and generalizes well to structurally diverse shapes, with promising results even in cross-category scenarios. Both qualitative and quantitative evaluations show that our method outperforms previous state-of-the-art techniques.
>
---
#### [new 164] NeuS-QA: Grounding Long-Form Video Understanding in Temporal Logic and Neuro-Symbolic Reasoning
- **分类: cs.CV**

- **简介: 该论文提出NeuS-QA，用于长视频问答（LVQA）任务。针对传统方法在时序推理和因果关系上的不足，NeuS-QA通过将问题转化为时序逻辑表达，并结合神经符号推理验证视频片段，从而提升模型的可解释性和推理能力。**

- **链接: [http://arxiv.org/pdf/2509.18041v1](http://arxiv.org/pdf/2509.18041v1)**

> **作者:** Sahil Shah; S P Sharan; Harsh Goel; Minkyu Choi; Mustafa Munir; Manvik Pasula; Radu Marculescu; Sandeep Chinchali
>
> **摘要:** Long-Form Video Question Answering (LVQA) poses challenges beyond traditional visual question answering (VQA), which is often limited to static images or short video clips. While current vision-language models (VLMs) perform well in those settings, they struggle with complex queries in LVQA over long videos involving multi-step temporal reasoning and causality. Vanilla approaches, which sample frames uniformly and feed them to a VLM with the question, incur significant token overhead, forcing severe downsampling. As a result, the model often misses fine-grained visual structure, subtle event transitions, or key temporal cues, ultimately leading to incorrect answers. To address these limitations, recent works have explored query-adaptive frame sampling, hierarchical keyframe selection, and agent-based iterative querying. However, these methods remain fundamentally heuristic: they lack explicit temporal representations and cannot enforce or verify logical event relationships. As a result, there are no formal guarantees that the sampled context actually encodes the compositional or causal logic demanded by the question. To address these foundational gaps, we introduce NeuS-QA, a training-free, plug-and-play neuro-symbolic pipeline for LVQA. NeuS-QA translates a natural language question into a formal temporal logic expression, constructs a video automaton from frame-level semantic propositions, and applies model checking to rigorously identify video segments satisfying the question's logical requirements. Only these logic-verified segments are submitted to the VLM, thus improving interpretability, reducing hallucinations, and enabling compositional reasoning without modifying or fine-tuning the model. Experiments on LongVideoBench and CinePile show NeuS-QA improves performance by over 10%, especially on questions involving event ordering, causality, and multi-step compositional reasoning.
>
---
#### [new 165] Trainee Action Recognition through Interaction Analysis in CCATT Mixed-Reality Training
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动作识别任务，旨在解决混合现实训练中团队协作评估的主观性和不一致性问题。研究结合认知任务分析与多模态学习分析，构建了基于交互行为的自动评估框架，提升了CCATT培训的客观性与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.17888v1](http://arxiv.org/pdf/2509.17888v1)**

> **作者:** Divya Mereddy; Marcos Quinones-Grueiro; Ashwin T S; Eduardo Davalos; Gautam Biswas; Kent Etherton; Tyler Davis; Katelyn Kay; Jill Lear; Benjamin Goldberg
>
> **摘要:** This study examines how Critical Care Air Transport Team (CCATT) members are trained using mixed-reality simulations that replicate the high-pressure conditions of aeromedical evacuation. Each team - a physician, nurse, and respiratory therapist - must stabilize severely injured soldiers by managing ventilators, IV pumps, and suction devices during flight. Proficient performance requires clinical expertise and cognitive skills, such as situational awareness, rapid decision-making, effective communication, and coordinated task management, all of which must be maintained under stress. Recent advances in simulation and multimodal data analytics enable more objective and comprehensive performance evaluation. In contrast, traditional instructor-led assessments are subjective and may overlook critical events, thereby limiting generalizability and consistency. However, AI-based automated and more objective evaluation metrics still demand human input to train the AI algorithms to assess complex team dynamics in the presence of environmental noise and the need for accurate re-identification in multi-person tracking. To address these challenges, we introduce a systematic, data-driven assessment framework that combines Cognitive Task Analysis (CTA) with Multimodal Learning Analytics (MMLA). We have developed a domain-specific CTA model for CCATT training and a vision-based action recognition pipeline using a fine-tuned Human-Object Interaction model, the Cascade Disentangling Network (CDN), to detect and track trainee-equipment interactions over time. These interactions automatically yield performance indicators (e.g., reaction time, task duration), which are mapped onto a hierarchical CTA model tailored to CCATT operations, enabling interpretable, domain-relevant performance evaluations.
>
---
#### [new 166] Geodesic Prototype Matching via Diffusion Maps for Interpretable Fine-Grained Recognition
- **分类: cs.CV**

- **简介: 该论文针对细粒度识别任务中基于原型方法的语义相似性不足问题，提出GeoProto框架。通过扩散映射提取类内流形结构，并引入可微Nystrom插值，使模型能捕捉深层特征的内在几何关系，提升可解释性和识别性能。**

- **链接: [http://arxiv.org/pdf/2509.17050v1](http://arxiv.org/pdf/2509.17050v1)**

> **作者:** Junhao Jia; Yunyou Liu; Yifei Sun; Huangwei Chen; Feiwei Qin; Changmiao Wang; Yong Peng
>
> **摘要:** Nonlinear manifolds are widespread in deep visual features, where Euclidean distances often fail to capture true similarity. This limitation becomes particularly severe in prototype-based interpretable fine-grained recognition, where subtle semantic distinctions are essential. To address this challenge, we propose a novel paradigm for prototype-based recognition that anchors similarity within the intrinsic geometry of deep features. Specifically, we distill the latent manifold structure of each class into a diffusion space and introduce a differentiable Nystr\"om interpolation, making the geometry accessible to both unseen samples and learnable prototypes. To ensure efficiency, we employ compact per-class landmark sets with periodic updates. This design keeps the embedding aligned with the evolving backbone, enabling fast and scalable inference. Extensive experiments on the CUB-200-2011 and Stanford Cars datasets show that our GeoProto framework produces prototypes focusing on semantically aligned parts, significantly outperforming Euclidean prototype networks.
>
---
#### [new 167] Animalbooth: multimodal feature enhancement for animal subject personalization
- **分类: cs.CV**

- **简介: 该论文提出AnimalBooth，用于个性化动物图像生成。针对跨域特征对齐导致的身份漂移问题，引入Animal Net、自适应注意力模块和频控特征集成模块，提升身份保真度与图像质量，并构建AnimalBench数据集推动研究。**

- **链接: [http://arxiv.org/pdf/2509.16702v1](http://arxiv.org/pdf/2509.16702v1)**

> **作者:** Chen Liu; Haitao Wu; Kafeng Wang; Xiaowang Zhang
>
> **摘要:** Personalized animal image generation is challenging due to rich appearance cues and large morphological variability. Existing approaches often exhibit feature misalignment across domains, which leads to identity drift. We present AnimalBooth, a framework that strengthens identity preservation with an Animal Net and an adaptive attention module, mitigating cross domain alignment errors. We further introduce a frequency controlled feature integration module that applies Discrete Cosine Transform filtering in the latent space to guide the diffusion process, enabling a coarse to fine progression from global structure to detailed texture. To advance research in this area, we curate AnimalBench, a high resolution dataset for animal personalization. Extensive experiments show that AnimalBooth consistently outperforms strong baselines on multiple benchmarks and improves both identity fidelity and perceptual quality.
>
---
#### [new 168] ConfidentSplat: Confidence-Weighted Depth Fusion for Accurate 3D Gaussian Splatting SLAM
- **分类: cs.CV; 68T20, 68U20**

- **简介: 该论文提出ConfidentSplat，一种基于3D高斯溅射的SLAM系统，旨在解决RGB-only重建中的深度估计不准确问题。通过引入置信度加权融合机制，结合多视角几何与单目先验，提升重建精度和新视角合成效果。**

- **链接: [http://arxiv.org/pdf/2509.16863v1](http://arxiv.org/pdf/2509.16863v1)**

> **作者:** Amanuel T. Dufera; Yuan-Li Cai
>
> **摘要:** We introduce ConfidentSplat, a novel 3D Gaussian Splatting (3DGS)-based SLAM system for robust, highfidelity RGB-only reconstruction. Addressing geometric inaccuracies in existing RGB-only 3DGS SLAM methods that stem from unreliable depth estimation, ConfidentSplat incorporates a core innovation: a confidence-weighted fusion mechanism. This mechanism adaptively integrates depth cues from multiview geometry with learned monocular priors (Omnidata ViT), dynamically weighting their contributions based on explicit reliability estimates-derived predominantly from multi-view geometric consistency-to generate high-fidelity proxy depth for map supervision. The resulting proxy depth guides the optimization of a deformable 3DGS map, which efficiently adapts online to maintain global consistency following pose updates from a DROID-SLAM-inspired frontend and backend optimizations (loop closure, global bundle adjustment). Extensive validation on standard benchmarks (TUM-RGBD, ScanNet) and diverse custom mobile datasets demonstrates significant improvements in reconstruction accuracy (L1 depth error) and novel view synthesis fidelity (PSNR, SSIM, LPIPS) over baselines, particularly in challenging conditions. ConfidentSplat underscores the efficacy of principled, confidence-aware sensor fusion for advancing state-of-the-art dense visual SLAM.
>
---
#### [new 169] Artificial Satellite Trails Detection Using U-Net Deep Neural Network and Line Segment Detector Algorithm
- **分类: cs.CV; astro-ph.IM**

- **简介: 该论文属于图像处理任务，旨在解决天文图像中人造卫星轨迹干扰问题。作者结合U-Net和LSD算法，构建检测模型，在模拟与真实数据上验证了其高检测率和精度。**

- **链接: [http://arxiv.org/pdf/2509.16771v1](http://arxiv.org/pdf/2509.16771v1)**

> **作者:** Xiaohan Chen; Hongrui Gu; Cunshi Wang; Haiyang Mu; Jie Zheng; Junju Du; Jing Ren; Zhou Fan; Jing Li
>
> **备注:** 15 pages, 7 figures, 2 tables, PASP accepted
>
> **摘要:** With the rapid increase in the number of artificial satellites, astronomical imaging is experiencing growing interference. When these satellites reflect sunlight, they produce streak-like artifacts in photometry images. Such satellite trails can introduce false sources and cause significant photometric errors. As a result, accurately identifying the positions of satellite trails in observational data has become essential. In this work, we propose a satellite trail detection model that combines the U-Net deep neural network for image segmentation with the Line Segment Detector (LSD) algorithm. The model is trained on 375 simulated images of satellite trails, generated using data from the Mini-SiTian Array. Experimental results show that for trails with a signal-to-noise ratio (SNR) greater than 3, the detection rate exceeds 99. Additionally, when applied to real observational data from the Mini-SiTian Array, the model achieves a recall of 79.57 and a precision of 74.56.
>
---
#### [new 170] FG-Attn: Leveraging Fine-Grained Sparsity In Diffusion Transformers
- **分类: cs.CV; cs.AR**

- **简介: 该论文针对视频生成中扩散变换器的计算瓶颈问题，提出FG-Attn方法。通过细粒度稀疏注意力机制和异步加载操作，减少冗余计算，在H100 GPU上实现视频生成速度提升1.41-1.65倍。**

- **链接: [http://arxiv.org/pdf/2509.16518v1](http://arxiv.org/pdf/2509.16518v1)**

> **作者:** Sankeerth Durvasula; Kavya Sreedhar; Zain Moustafa; Suraj Kothawade; Ashish Gondimalla; Suvinay Subramanian; Narges Shahidi; Nandita Vijaykumar
>
> **摘要:** Generating realistic videos with diffusion transformers demands significant computation, with attention layers the central bottleneck; even producing a short clip requires running a transformer over a very long sequence of embeddings, e.g., more than 30K embeddings for a 5-second video, incurring significant latency. Prior work aims to mitigate this bottleneck by exploiting sparsity in the attention layers to reduce computation. However, these works typically rely on block-sparse attention, which skips score computation only when all entries in a block of attention scores (corresponding to M queries and M keys, with M = 64 typically) are zero. This coarse-granular skipping of attention scores does not fully exploit sparsity in the attention map and leaves room for improvement. In this work, we propose FG-Attn, a sparse attention mechanism for long-context diffusion transformers that leverages sparsity at a fine granularity. Unlike block-sparse attention, which skips entire MxM blocks, our approach skips computations at the granularity of Mx1 slices of the attention map. Each slice is produced by query-key dot products between a block of query vectors and a single key. To implement our proposed sparse attention mechanism, we develop a new efficient bulk-load operation called asynchronous-gather load. This load operation gathers a sparse set of relevant key-value vectors from memory and arranges them into packed tiles in the GPU's shared memory. Only a sparse set of keys relevant to those queries are loaded into shared memory when computing attention for a block of queries, in contrast to loading full blocks of key tokens in block-sparse attention. Our fine-grained sparse attention, applied to video diffusion models, achieves an average 1.55X (up to 1.65X) speedup for 5 second, 480p videos, and an average 1.41X (up to 1.49X) for 5 second, 720p videos on a single H100 GPU.
>
---
#### [new 171] Parameter-efficient fine-tuning (PEFT) of Vision Foundation Models for Atypical Mitotic Figure Classification
- **分类: cs.CV; 68T07; I.2.10; I.4.9; I.5.4**

- **简介: 该论文研究利用参数高效微调（PEFT）的视觉基础模型（如Virchow、UNI）进行非典型有丝分裂图像分类，旨在解决其检测中形态细微、类别不平衡等问题。通过LoRA方法优化模型性能，在MIDOG 2025挑战中取得88.37%的平衡准确率。**

- **链接: [http://arxiv.org/pdf/2509.16935v1](http://arxiv.org/pdf/2509.16935v1)**

> **作者:** Lavish Ramchandani; Gunjan Deotale; Dev Kumar Das
>
> **备注:** MIDOG'25
>
> **摘要:** Atypical mitotic figures (AMFs) are rare abnormal cell divisions associated with tumor aggressiveness and poor prognosis. Their detection remains a significant challenge due to subtle morphological cues, class imbalance, and inter-observer variability among pathologists. The MIDOG 2025 challenge introduced a dedicated track for atypical mitosis classification, enabling systematic evaluation of deep learning methods. In this study, we investigated the use of large vision foundation models, including Virchow, Virchow2, and UNI, with Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. We conducted extensive experiments with different LoRA ranks, as well as random and group-based data splits, to analyze robustness under varied conditions. Our best approach, Virchow with LoRA rank 8 and ensemble of three-fold cross-validation, achieved a balanced accuracy of 88.37% on the preliminary test set, ranking joint 9th in the challenge leaderboard. These results highlight the promise of foundation models with efficient adaptation strategies for the classification of atypical mitosis, while underscoring the need for improvements in specificity and domain generalization.
>
---
#### [new 172] Overview of PlantCLEF 2022: Image-based plant identification at global scale
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决全球范围内基于图像的植物自动识别问题。工作内容包括提出PlantCLEF2022挑战赛，使用多图像和元数据对8万个植物物种进行分类，并分析参与团队的方法与结果。**

- **链接: [http://arxiv.org/pdf/2509.17632v1](http://arxiv.org/pdf/2509.17632v1)**

> **作者:** Herve Goeau; Pierre Bonnet; Alexis Joly
>
> **备注:** 13 pages, 2 figures, CLEF 2022 Conference and Labs of the Evaluation Forum, September 05 to 08, 2022, Bologna, Italy
>
> **摘要:** It is estimated that there are more than 300,000 species of vascular plants in the world. Increasing our knowledge of these species is of paramount importance for the development of human civilization (agriculture, construction, pharmacopoeia, etc.), especially in the context of the biodiversity crisis. However, the burden of systematic plant identification by human experts strongly penalizes the aggregation of new data and knowledge. Since then, automatic identification has made considerable progress in recent years as highlighted during all previous editions of PlantCLEF. Deep learning techniques now seem mature enough to address the ultimate but realistic problem of global identification of plant biodiversity in spite of many problems that the data may present (a huge number of classes, very strongly unbalanced classes, partially erroneous identifications, duplications, variable visual quality, diversity of visual contents such as photos or herbarium sheets, etc). The PlantCLEF2022 challenge edition proposes to take a step in this direction by tackling a multi-image (and metadata) classification problem with a very large number of classes (80k plant species). This paper presents the resources and evaluations of the challenge, summarizes the approaches and systems employed by the participating research groups, and provides an analysis of key findings.
>
---
#### [new 173] Domain Adaptive Object Detection for Space Applications with Real-Time Constraints
- **分类: cs.CV**

- **简介: 该论文研究面向太空应用的实时目标检测任务，旨在解决合成数据与真实数据域差异导致性能下降的问题。提出结合半监督域适应方法，通过域不变特征学习和轻量级检测模型提升真实场景下的检测精度。**

- **链接: [http://arxiv.org/pdf/2509.17593v1](http://arxiv.org/pdf/2509.17593v1)**

> **作者:** Samet Hicsonmez; Abd El Rahman Shabayek; Arunkumar Rathinam; Djamila Aouada
>
> **备注:** Advanced Space Technologies in Robotics and Automation (ASTRA) 2025
>
> **摘要:** Object detection is essential in space applications targeting Space Domain Awareness and also applications involving relative navigation scenarios. Current deep learning models for Object Detection in space applications are often trained on synthetic data from simulators, however, the model performance drops significantly on real-world data due to the domain gap. However, domain adaptive object detection is an overlooked problem in the community. In this work, we first show the importance of domain adaptation and then explore Supervised Domain Adaptation (SDA) to reduce this gap using minimal labeled real data. We build on a recent semi-supervised adaptation method and tailor it for object detection. Our approach combines domain-invariant feature learning with a CNN-based domain discriminator and invariant risk minimization using a domain-independent regression head. To meet real-time deployment needs, we test our method on a lightweight Single Shot Multibox Detector (SSD) with MobileNet backbone and on the more advanced Fully Convolutional One-Stage object detector (FCOS) with ResNet-50 backbone. We evaluated on two space datasets, SPEED+ and SPARK. The results show up to 20-point improvements in average precision (AP) with just 250 labeled real images.
>
---
#### [new 174] DINOv3-Diffusion Policy: Self-Supervised Large Visual Model for Visuomotor Diffusion Policy Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究了DINOv3自监督视觉模型在机器人操作任务中的扩散策略学习效果。通过对比自监督与监督预训练模型，验证了DINOv3在多种训练模式下的性能优势，证明其可作为高效、通用的感知前端，提升策略成功率。**

- **链接: [http://arxiv.org/pdf/2509.17684v1](http://arxiv.org/pdf/2509.17684v1)**

> **作者:** ThankGod Egbe; Peng Wang; Zhihao Guo; Zidong Chen
>
> **摘要:** This paper evaluates DINOv3, a recent large-scale self-supervised vision backbone, for visuomotor diffusion policy learning in robotic manipulation. We investigate whether a purely self-supervised encoder can match or surpass conventional supervised ImageNet-pretrained backbones (e.g., ResNet-18) under three regimes: training from scratch, frozen, and finetuned. Across four benchmark tasks (Push-T, Lift, Can, Square) using a unified FiLM-conditioned diffusion policy, we find that (i) finetuned DINOv3 matches or exceeds ResNet-18 on several tasks, (ii) frozen DINOv3 remains competitive, indicating strong transferable priors, and (iii) self-supervised features improve sample efficiency and robustness. These results support self-supervised large visual models as effective, generalizable perceptual front-ends for action diffusion policies, motivating further exploration of scalable label-free pretraining in robotic manipulation. Compared to using ResNet18 as a backbone, our approach with DINOv3 achieves up to a 10% absolute increase in test-time success rates on challenging tasks such as Can, and on-the-par performance in tasks like Lift, PushT, and Square.
>
---
#### [new 175] Enhancing Semantic Segmentation with Continual Self-Supervised Pre-training
- **分类: cs.CV**

- **简介: 该论文针对语义分割任务，研究如何在有限数据下高效扩展自监督预训练模型。提出GLARE方法，通过局部和区域一致性约束，结合轻量适配模块，实现计算高效的持续自监督预训练，提升下游分割性能。**

- **链接: [http://arxiv.org/pdf/2509.17816v1](http://arxiv.org/pdf/2509.17816v1)**

> **作者:** Brown Ebouky; Ajad Chhatkuli; Cristiano Malossi; Christoph Studer; Roy Assaf; Andrea Bartezzaghi
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Self-supervised learning (SSL) has emerged as a central paradigm for training foundation models by leveraging large-scale unlabeled datasets, often producing representations with strong generalization capabilities. These models are typically pre-trained on general-purpose datasets such as ImageNet and subsequently adapted to various downstream tasks through finetuning. While recent advances have explored parameter-efficient strategies for adapting pre-trained models, extending SSL pre-training itself to new domains - particularly under limited data regimes and for dense prediction tasks - remains underexplored. In this work, we address the problem of adapting vision foundation models to new domains in an unsupervised and data-efficient manner, specifically targeting downstream semantic segmentation. We propose GLARE (Global Local and Regional Enforcement), a novel continual self-supervised pre-training task designed to enhance downstream segmentation performance. GLARE introduces patch-level augmentations to encourage local consistency and incorporates a regional consistency constraint that leverages spatial semantics in the data. For efficient continual pre-training, we initialize Vision Transformers (ViTs) with weights from existing SSL models and update only lightweight adapter modules - specifically UniAdapter - while keeping the rest of the backbone frozen. Experiments across multiple semantic segmentation benchmarks on different domains demonstrate that GLARE consistently improves downstream performance with minimal computational and parameter overhead.
>
---
#### [new 176] Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion
- **分类: cs.CV**

- **简介: 该论文针对多焦点图像融合任务中决策图边界不精确的问题，提出基于神经动力学驱动的耦合神经P系统模型ND-CNPFuse。通过将输入映射为可解释的脉冲矩阵并比较脉冲数量，直接生成高精度决策图，无需后处理，在多个数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.17704v1](http://arxiv.org/pdf/2509.17704v1)**

> **作者:** Bo Li; Yunkuo Lei; Tingting Bao; Yaxian Wang; Lingling Zhang; Jun Liu
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Multi-focus image fusion (MFIF) is a crucial technique in image processing, with a key challenge being the generation of decision maps with precise boundaries. However, traditional methods based on heuristic rules and deep learning methods with black-box mechanisms are difficult to generate high-quality decision maps. To overcome this challenge, we introduce neurodynamics-driven coupled neural P (CNP) systems, which are third-generation neural computation models inspired by spiking mechanisms, to enhance the accuracy of decision maps. Specifically, we first conduct an in-depth analysis of the model's neurodynamics to identify the constraints between the network parameters and the input signals. This solid analysis avoids abnormal continuous firing of neurons and ensures the model accurately distinguishes between focused and unfocused regions, generating high-quality decision maps for MFIF. Based on this analysis, we propose a \textbf{N}eurodynamics-\textbf{D}riven \textbf{CNP} \textbf{F}usion model (\textbf{ND-CNPFuse}) tailored for the challenging MFIF task. Unlike current ideas of decision map generation, ND-CNPFuse distinguishes between focused and unfocused regions by mapping the source image into interpretable spike matrices. By comparing the number of spikes, an accurate decision map can be generated directly without any post-processing. Extensive experimental results show that ND-CNPFuse achieves new state-of-the-art performance on four classical MFIF datasets, including Lytro, MFFW, MFI-WHU, and Real-MFF. The code is available at https://github.com/MorvanLi/ND-CNPFuse.
>
---
#### [new 177] KRAST: Knowledge-Augmented Robotic Action Recognition with Structured Text for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于机器人动作识别任务，旨在提升室内日常动作识别的准确性。通过引入知识增强的提示学习框架，将结构化文本描述嵌入预训练视觉-语言模型，实现仅用RGB视频输入即可达到95%以上的识别准确率，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16452v1](http://arxiv.org/pdf/2509.16452v1)**

> **作者:** Son Hai Nguyen; Diwei Wang; Jinhyeok Jang; Hyewon Seo
>
> **摘要:** Accurate vision-based action recognition is crucial for developing autonomous robots that can operate safely and reliably in complex, real-world environments. In this work, we advance video-based recognition of indoor daily actions for robotic perception by leveraging vision-language models (VLMs) enriched with domain-specific knowledge. We adapt a prompt-learning framework in which class-level textual descriptions of each action are embedded as learnable prompts into a frozen pre-trained VLM backbone. Several strategies for structuring and encoding these textual descriptions are designed and evaluated. Experiments on the ETRI-Activity3D dataset demonstrate that our method, using only RGB video inputs at test time, achieves over 95\% accuracy and outperforms state-of-the-art approaches. These results highlight the effectiveness of knowledge-augmented prompts in enabling robust action recognition with minimal supervision.
>
---
#### [new 178] From Benchmarks to Reality: Advancing Visual Anomaly Detection by the VAND 3.0 Challenge
- **分类: cs.CV**

- **简介: 该论文聚焦视觉异常检测任务，旨在解决实际应用中的分布偏移和小样本问题。通过VAND 3.0挑战赛，设置了两个赛道分别探索鲁棒检测方法与视觉语言模型的少样本能力，推动了领域进展。**

- **链接: [http://arxiv.org/pdf/2509.17615v1](http://arxiv.org/pdf/2509.17615v1)**

> **作者:** Lars Heckler-Kram; Ashwin Vaidya; Jan-Hendrik Neudeck; Ulla Scheler; Dick Ameln; Samet Akcay; Paula Ramos
>
> **摘要:** Visual anomaly detection is a strongly application-driven field of research. Consequently, the connection between academia and industry is of paramount importance. In this regard, we present the VAND 3.0 Challenge to showcase current progress in anomaly detection across different practical settings whilst addressing critical issues in the field. The challenge hosted two tracks, fostering the development of anomaly detection methods robust against real-world distribution shifts (Category 1) and exploring the capabilities of Vision Language Models within the few-shot regime (Category 2), respectively. The participants' solutions reached significant improvements over previous baselines by combining or adapting existing approaches and fusing them with novel pipelines. While for both tracks the progress in large pre-trained vision (language) backbones played a pivotal role for the performance increase, scaling up anomaly detection methods more efficiently needs to be addressed by future research to meet real-time and computational constraints on-site.
>
---
#### [new 179] Octree Latent Diffusion for Semantic 3D Scene Generation and Completion
- **分类: cs.CV**

- **简介: 该论文提出Octree Latent Semantic Diffusion框架，统一解决3D语义场景的补全、扩展与生成问题。通过双八叉树图潜空间表示，实现跨室内外场景的零样本泛化和高效推理，无需微调即可利用LiDAR数据完成高质量语义生成。**

- **链接: [http://arxiv.org/pdf/2509.16483v1](http://arxiv.org/pdf/2509.16483v1)**

> **作者:** Xujia Zhang; Brendan Crowe; Christoffer Heckman
>
> **摘要:** The completion, extension, and generation of 3D semantic scenes are an interrelated set of capabilities that are useful for robotic navigation and exploration. Existing approaches seek to decouple these problems and solve them oneoff. Additionally, these approaches are often domain-specific, requiring separate models for different data distributions, e.g. indoor vs. outdoor scenes. To unify these techniques and provide cross-domain compatibility, we develop a single framework that can perform scene completion, extension, and generation in both indoor and outdoor scenes, which we term Octree Latent Semantic Diffusion. Our approach operates directly on an efficient dual octree graph latent representation: a hierarchical, sparse, and memory-efficient occupancy structure. This technique disentangles synthesis into two stages: (i) structure diffusion, which predicts binary split signals to construct a coarse occupancy octree, and (ii) latent semantic diffusion, which generates semantic embeddings decoded by a graph VAE into voxellevel semantic labels. To perform semantic scene completion or extension, our model leverages inference-time latent inpainting, or outpainting respectively. These inference-time methods use partial LiDAR scans or maps to condition generation, without the need for retraining or finetuning. We demonstrate highquality structure, coherent semantics, and robust completion from single LiDAR scans, as well as zero-shot generalization to out-of-distribution LiDAR data. These results indicate that completion-through-generation in a dual octree graph latent space is a practical and scalable alternative to regression-based pipelines for real-world robotic perception tasks.
>
---
#### [new 180] MoCLIP-Lite: Efficient Video Recognition by Fusing CLIP with Motion Vectors
- **分类: cs.CV**

- **简介: 该论文提出MoCLIP-Lite，用于高效视频动作识别。针对现有模型计算成本高、依赖大量预训练的问题，结合CLIP图像编码器与运动矢量特征，采用轻量融合框架，在UCF101上达到89.2%的Top-1精度，显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2509.17084v1](http://arxiv.org/pdf/2509.17084v1)**

> **作者:** Binhua Huang; Nan Wang; Arjun Parakash; Soumyabrata Dev
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Video action recognition is a fundamental task in computer vision, but state-of-the-art models are often computationally expensive and rely on extensive video pre-training. In parallel, large-scale vision-language models like Contrastive Language-Image Pre-training (CLIP) offer powerful zero-shot capabilities on static images, while motion vectors (MV) provide highly efficient temporal information directly from compressed video streams. To synergize the strengths of these paradigms, we propose MoCLIP-Lite, a simple yet powerful two-stream late fusion framework for efficient video recognition. Our approach combines features from a frozen CLIP image encoder with features from a lightweight, supervised network trained on raw MV. During fusion, both backbones are frozen, and only a tiny Multi-Layer Perceptron (MLP) head is trained, ensuring extreme efficiency. Through comprehensive experiments on the UCF101 dataset, our method achieves a remarkable 89.2% Top-1 accuracy, significantly outperforming strong zero-shot (65.0%) and MV-only (66.5%) baselines. Our work provides a new, highly efficient baseline for video understanding that effectively bridges the gap between large static models and dynamic, low-cost motion cues. Our code and models are available at https://github.com/microa/MoCLIP-Lite.
>
---
#### [new 181] Accurate Thyroid Cancer Classification using a Novel Binary Pattern Driven Local Discrete Cosine Transform Descriptor
- **分类: cs.CV; cs.LG; eess.IV; I.2.1; I.5.2**

- **简介: 该论文提出一种用于甲状腺癌分类的CAD系统，任务是医学图像分类。针对甲状腺超声图像纹理特征提取难题，设计了结合ILBP与LDCT的BPD-LDCT新特征描述子，并使用非线性SVM进行分类，显著提升了分类准确率。**

- **链接: [http://arxiv.org/pdf/2509.16382v1](http://arxiv.org/pdf/2509.16382v1)**

> **作者:** Saurabh Saini; Kapil Ahuja; Marc C. Steinbach; Thomas Wick
>
> **备注:** 15 Pages, 7 Figures, 5 Tables
>
> **摘要:** In this study, we develop a new CAD system for accurate thyroid cancer classification with emphasis on feature extraction. Prior studies have shown that thyroid texture is important for segregating the thyroid ultrasound images into different classes. Based upon our experience with breast cancer classification, we first conjuncture that the Discrete Cosine Transform (DCT) is the best descriptor for capturing textural features. Thyroid ultrasound images are particularly challenging as the gland is surrounded by multiple complex anatomical structures leading to variations in tissue density. Hence, we second conjuncture the importance of localization and propose that the Local DCT (LDCT) descriptor captures the textural features best in this context. Another disadvantage of complex anatomy around the thyroid gland is scattering of ultrasound waves resulting in noisy and unclear textures. Hence, we third conjuncture that one image descriptor is not enough to fully capture the textural features and propose the integration of another popular texture capturing descriptor (Improved Local Binary Pattern, ILBP) with LDCT. ILBP is known to be noise resilient as well. We term our novel descriptor as Binary Pattern Driven Local Discrete Cosine Transform (BPD-LDCT). Final classification is carried out using a non-linear SVM. The proposed CAD system is evaluated on the only two publicly available thyroid cancer datasets, namely TDID and AUITD. The evaluation is conducted in two stages. In Stage I, thyroid nodules are categorized as benign or malignant. In Stage II, the malignant cases are further sub-classified into TI-RADS (4) and TI-RADS (5). For Stage I classification, our proposed model demonstrates exceptional performance of nearly 100% on TDID and 97% on AUITD. In Stage II classification, the proposed model again attains excellent classification of close to 100% on TDID and 99% on AUITD.
>
---
#### [new 182] Rethinking Evaluation of Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文针对红外小目标检测任务，提出改进评估方法。当前评估存在指标碎片化、忽视错误分析和泛化性不足的问题。为此，论文引入了融合像素与目标级的混合指标，系统性错误分析方法，并强调跨数据集评估，以构建更全面的分析框架。**

- **链接: [http://arxiv.org/pdf/2509.16888v1](http://arxiv.org/pdf/2509.16888v1)**

> **作者:** Youwei Pang; Xiaoqi Zhao; Lihe Zhang; Huchuan Lu; Georges El Fakhri; Xiaofeng Liu; Shijian Lu
>
> **备注:** NeurIPS 2025; Evaluation Toolkit: https://github.com/lartpang/PyIRSTDMetrics
>
> **摘要:** As an essential vision task, infrared small target detection (IRSTD) has seen significant advancements through deep learning. However, critical limitations in current evaluation protocols impede further progress. First, existing methods rely on fragmented pixel- and target-level specific metrics, which fails to provide a comprehensive view of model capabilities. Second, an excessive emphasis on overall performance scores obscures crucial error analysis, which is vital for identifying failure modes and improving real-world system performance. Third, the field predominantly adopts dataset-specific training-testing paradigms, hindering the understanding of model robustness and generalization across diverse infrared scenarios. This paper addresses these issues by introducing a hybrid-level metric incorporating pixel- and target-level performance, proposing a systematic error analysis method, and emphasizing the importance of cross-dataset evaluation. These aim to offer a more thorough and rational hierarchical analysis framework, ultimately fostering the development of more effective and robust IRSTD models. An open-source toolkit has be released to facilitate standardized benchmarking.
>
---
#### [new 183] ChartHal: A Fine-grained Framework Evaluating Hallucination of Large Vision Language Models in Chart Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ChartHal，一个用于评估大视觉语言模型在图表理解中幻觉现象的细粒度基准。针对现有研究对图表理解和幻觉结合不足的问题，构建了包含1062个样本的人工标注数据集，并分析了主流模型的幻觉表现，揭示了其在处理矛盾或缺失信息时的严重问题。**

- **链接: [http://arxiv.org/pdf/2509.17481v1](http://arxiv.org/pdf/2509.17481v1)**

> **作者:** Xingqi Wang; Yiming Cui; Xin Yao; Shijin Wang; Guoping Hu; Xiaoyu Qin
>
> **摘要:** Large Vision-Language Models (LVLMs) have recently demonstrated remarkable progress, yet hallucination remains a critical barrier, particularly in chart understanding, which requires sophisticated perceptual and cognitive abilities as well as rigorous factual accuracy. While prior work has investigated hallucinations and chart comprehension independently, their intersection remains largely unexplored. To address this gap, we present ChartHal, a benchmark that features a fine-grained taxonomy of hallucination scenarios in chart understanding, along with a human-validated dataset of 1,062 samples. Our evaluation shows that state-of-the-art LVLMs suffer from severe hallucinations on ChartHal, including proprietary models such as GPT-5 and o4-mini, which achieve only 34.46% and 22.79% accuracy, respectively. Further analysis reveals that questions involving information absent from or contradictory to charts are especially likely to trigger hallucinations, underscoring the urgent need for more robust mitigation strategies. Code and data are available at https://github.com/ymcui/ChartHal .
>
---
#### [new 184] WISE: Weak-Supervision-Guided Step-by-Step Explanations for Multimodal LLMs in Image Classification
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型（MLLMs）在图像分类中的可解释性不足问题，提出WISE方法。通过弱监督引导生成基于概念的逐步解释（MCoT），增强模型对图像内部特征的理解，提升分类准确率和可解释性。**

- **链接: [http://arxiv.org/pdf/2509.17740v1](http://arxiv.org/pdf/2509.17740v1)**

> **作者:** Yiwen Jiang; Deval Mehta; Siyuan Yan; Yaling Shen; Zimu Wang; Zongyuan Ge
>
> **备注:** Accepted at EMNLP 2025 (Main)
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown promise in visual-textual reasoning, with Multimodal Chain-of-Thought (MCoT) prompting significantly enhancing interpretability. However, existing MCoT methods rely on rationale-rich datasets and largely focus on inter-object reasoning, overlooking the intra-object understanding crucial for image classification. To address this gap, we propose WISE, a Weak-supervision-guided Step-by-step Explanation method that augments any image classification dataset with MCoTs by reformulating the concept-based representations from Concept Bottleneck Models (CBMs) into concise, interpretable reasoning chains under weak supervision. Experiments across ten datasets show that our generated MCoTs not only improve interpretability by 37% but also lead to gains in classification accuracy when used to fine-tune MLLMs. Our work bridges concept-based interpretability and generative MCoT reasoning, providing a generalizable framework for enhancing MLLMs in fine-grained visual understanding.
>
---
#### [new 185] Preconditioned Deformation Grids
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出一种新的动态表面重建方法——Preconditioned Deformation Grids，用于从点云序列中估计形变场。针对现有方法依赖复杂正则化或大量训练数据的问题，该方法通过多分辨率体素网格和Sobolev预条件优化，实现了无需显式对应关系的高精度重建。**

- **链接: [http://arxiv.org/pdf/2509.18097v1](http://arxiv.org/pdf/2509.18097v1)**

> **作者:** Julian Kaltheuner; Alexander Oebel; Hannah Droege; Patrick Stotko; Reinhard Klein
>
> **备注:** GitHub: https://github.com/vc-bonn/preconditioned-deformation-grids
>
> **摘要:** Dynamic surface reconstruction of objects from point cloud sequences is a challenging field in computer graphics. Existing approaches either require multiple regularization terms or extensive training data which, however, lead to compromises in reconstruction accuracy as well as over-smoothing or poor generalization to unseen objects and motions. To address these lim- itations, we introduce Preconditioned Deformation Grids, a novel technique for estimating coherent deformation fields directly from unstructured point cloud sequences without requiring or forming explicit correspondences. Key to our approach is the use of multi-resolution voxel grids that capture the overall motion at varying spatial scales, enabling a more flexible deformation representation. In conjunction with incorporating grid-based Sobolev preconditioning into gradient-based optimization, we show that applying a Chamfer loss between the input point clouds as well as to an evolving template mesh is sufficient to obtain accurate deformations. To ensure temporal consistency along the object surface, we include a weak isometry loss on mesh edges which complements the main objective without constraining deformation fidelity. Extensive evaluations demonstrate that our method achieves superior results, particularly for long sequences, compared to state-of-the-art techniques.
>
---
#### [new 186] ProtoVQA: An Adaptable Prototypical Framework for Explainable Fine-Grained Visual Question Answering
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ProtoVQA，一种可解释的细粒度视觉问答框架。针对VQA任务中模型缺乏可解释性的问题，设计了问题感知原型、空间约束匹配和共享原型结构，并引入VLAS评估指标，提升答案与图像区域的对齐质量，增强系统的透明性和可信度。**

- **链接: [http://arxiv.org/pdf/2509.16680v1](http://arxiv.org/pdf/2509.16680v1)**

> **作者:** Xingjian Diao; Weiyi Wu; Keyi Kong; Peijun Qing; Xinwen Xu; Ming Cheng; Soroush Vosoughi; Jiang Gui
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Visual Question Answering (VQA) is increasingly used in diverse applications ranging from general visual reasoning to safety-critical domains such as medical imaging and autonomous systems, where models must provide not only accurate answers but also explanations that humans can easily understand and verify. Prototype-based modeling has shown promise for interpretability by grounding predictions in semantically meaningful regions for purely visual reasoning tasks, yet remains underexplored in the context of VQA. We present ProtoVQA, a unified prototypical framework that (i) learns question-aware prototypes that serve as reasoning anchors, connecting answers to discriminative image regions, (ii) applies spatially constrained matching to ensure that the selected evidence is coherent and semantically relevant, and (iii) supports both answering and grounding tasks through a shared prototype backbone. To assess explanation quality, we propose the Visual-Linguistic Alignment Score (VLAS), which measures how well the model's attended regions align with ground-truth evidence. Experiments on Visual7W show that ProtoVQA yields faithful, fine-grained explanations while maintaining competitive accuracy, advancing the development of transparent and trustworthy VQA systems.
>
---
#### [new 187] UIPro: Unleashing Superior Interaction Capability For GUI Agents
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出UIPro，一种通用GUI代理，旨在解决现有方法在场景有限、数据不足和动作空间异构等问题。通过构建大规模多平台数据集和统一动作空间，提升了GUI交互能力。**

- **链接: [http://arxiv.org/pdf/2509.17328v1](http://arxiv.org/pdf/2509.17328v1)**

> **作者:** Hongxin Li; Jingran Su; Jingfan Chen; Zheng Ju; Yuntao Chen; Qing Li; Zhaoxiang Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Building autonomous agents that perceive and operate graphical user interfaces (GUIs) like humans has long been a vision in the field of artificial intelligence. Central to these agents is the capability for GUI interaction, which involves GUI understanding and planning capabilities. Existing methods have tried developing GUI agents based on the multi-modal comprehension ability of vision-language models (VLMs). However, the limited scenario, insufficient size, and heterogeneous action spaces hinder the progress of building generalist GUI agents. To resolve these issues, this paper proposes \textbf{UIPro}, a novel generalist GUI agent trained with extensive multi-platform and multi-task GUI interaction data, coupled with a unified action space. We first curate a comprehensive dataset encompassing 20.6 million GUI understanding tasks to pre-train UIPro, granting it a strong GUI grounding capability, which is key to downstream GUI agent tasks. Subsequently, we establish a unified action space to harmonize heterogeneous GUI agent task datasets and produce a merged dataset to foster the action prediction ability of UIPro via continued fine-tuning. Experimental results demonstrate UIPro's superior performance across multiple GUI task benchmarks on various platforms, highlighting the effectiveness of our approach.
>
---
#### [new 188] Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics
- **分类: cs.CV**

- **简介: 该论文提出RPLIF模型，通过引入神经元的不应期机制改进LIF神经元模型。属于脉冲神经网络（SNN）研究任务，旨在解决传统模型忽略生物神经不应期导致的过激发和干扰问题。方法提升了SNN的鲁棒性和效率，并在多个数据集上取得先进性能。**

- **链接: [http://arxiv.org/pdf/2509.17769v1](http://arxiv.org/pdf/2509.17769v1)**

> **作者:** Yang Li; Xinyi Zeng; Zhe Xue; Pinxian Zeng; Zikai Zhang; Yan Wang
>
> **摘要:** As the third generation of neural networks, spiking neural networks (SNNs) have recently gained widespread attention for their biological plausibility, energy efficiency, and effectiveness in processing neuromorphic datasets. To better emulate biological neurons, various models such as Integrate-and-Fire (IF) and Leaky Integrate-and-Fire (LIF) have been widely adopted in SNNs. However, these neuron models overlook the refractory period, a fundamental characteristic of biological neurons. Research on excitable neurons reveal that after firing, neurons enter a refractory period during which they are temporarily unresponsive to subsequent stimuli. This mechanism is critical for preventing over-excitation and mitigating interference from aberrant signals. Therefore, we propose a simple yet effective method to incorporate the refractory period into spiking LIF neurons through spike-triggered threshold dynamics, termed RPLIF. Our method ensures that each spike accurately encodes neural information, effectively preventing neuron over-excitation under continuous inputs and interference from anomalous inputs. Incorporating the refractory period into LIF neurons is seamless and computationally efficient, enhancing robustness and efficiency while yielding better performance with negligible overhead. To the best of our knowledge, RPLIF achieves state-of-the-art performance on Cifar10-DVS(82.40%) and N-Caltech101(83.35%) with fewer timesteps and demonstrates superior performance on DVS128 Gesture(97.22%) at low latency.
>
---
#### [new 189] Tailored Transformation Invariance for Industrial Anomaly Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对工业异常检测任务，提出LWinNN方法，在kNN基础上引入局部窗口机制，以有限平移不变性提升检测准确率并减少训练与测试时间，为后续研究提供新基线。**

- **链接: [http://arxiv.org/pdf/2509.17670v1](http://arxiv.org/pdf/2509.17670v1)**

> **作者:** Mariette Schönfeld; Wannes Meert; Hendrik Blockeel
>
> **摘要:** Industrial Anomaly Detection (IAD) is a subproblem within Computer Vision Anomaly Detection that has been receiving increasing amounts of attention due to its applicability to real-life scenarios. Recent research has focused on how to extract the most informative features, contrasting older kNN-based methods that use only pretrained features. These recent methods are much more expensive to train however and could complicate real-life application. Careful study of related work with regards to transformation invariance leads to the idea that popular benchmarks require robustness to only minor translations. With this idea we then formulate LWinNN, a local window based approach that creates a middle ground between kNN based methods that have either complete or no translation invariance. Our experiments demonstrate that this small change increases accuracy considerably, while simultaneously decreasing both train and test time. This teaches us two things: first, the gap between kNN-based approaches and more complex state-of-the-art methodology can still be narrowed by effective usage of the limited data available. Second, our assumption of requiring only limited translation invariance highlights potential areas of interest for future work and the need for more spatially diverse benchmarks, for which our method can hopefully serve as a new baseline. Our code can be found at https://github.com/marietteschonfeld/LWinNN .
>
---
#### [new 190] SLAM-Former: Putting SLAM into One Transformer
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SLAM-Former，将完整的SLAM功能集成到一个Transformer中，实现单目图像的实时建图与跟踪，并通过前后端协同优化提升几何一致性。**

- **链接: [http://arxiv.org/pdf/2509.16909v1](http://arxiv.org/pdf/2509.16909v1)**

> **作者:** Yijun Yuan; Zhuoguang Chen; Kenan Li; Weibang Wang; Hang Zhao
>
> **备注:** Project Page:https://tsinghua-mars-lab.github.io/SLAM-Former
>
> **摘要:** We present SLAM-Former, a novel neural approach that integrates full SLAM capabilities into a single transformer. Similar to traditional SLAM systems, SLAM-Former comprises both a frontend and a backend that operate in tandem. The frontend processes sequential monocular images in real-time for incremental mapping and tracking, while the backend performs global refinement to ensure a geometrically consistent result. This alternating execution allows the frontend and backend to mutually promote one another, enhancing overall system performance. Comprehensive experimental results demonstrate that SLAM-Former achieves superior or highly competitive performance compared to state-of-the-art dense SLAM methods.
>
---
#### [new 191] Automated Facility Enumeration for Building Compliance Checking using Door Detection and Large Language Models
- **分类: cs.CV; cs.AI; cs.ET**

- **简介: 该论文提出了一种结合门检测与大语言模型的建筑合规检查新方法，用于自动统计设施数量并验证其是否符合法规要求。通过引入链式推理流程，提升了自动化水平和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.17283v1](http://arxiv.org/pdf/2509.17283v1)**

> **作者:** Licheng Zhan; Bach Le; Naveed Akhtar; Tuan Ngo
>
> **摘要:** Building compliance checking (BCC) is a critical process for ensuring that constructed facilities meet regulatory standards. A core component of BCC is the accurate enumeration of facility types and their spatial distribution. Despite its importance, this problem has been largely overlooked in the literature, posing a significant challenge for BCC and leaving a critical gap in existing workflows. Performing this task manually is time-consuming and labor-intensive. Recent advances in large language models (LLMs) offer new opportunities to enhance automation by combining visual recognition with reasoning capabilities. In this paper, we introduce a new task for BCC: automated facility enumeration, which involves validating the quantity of each facility type against statutory requirements. To address it, we propose a novel method that integrates door detection with LLM-based reasoning. We are the first to apply LLMs to this task and further enhance their performance through a Chain-of-Thought (CoT) pipeline. Our approach generalizes well across diverse datasets and facility types. Experiments on both real-world and synthetic floor plan data demonstrate the effectiveness and robustness of our method.
>
---
#### [new 192] FakeChain: Exposing Shallow Cues in Multi-Step Deepfake Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦多步骤深度伪造检测任务，旨在解决现有检测模型在面对混合伪造方法时泛化能力差的问题。作者构建了FakeChain基准数据集，包含1-3步的伪造样本，并分析发现检测性能严重依赖最后一步操作，强调需考虑伪造过程的历史与序列信息。**

- **链接: [http://arxiv.org/pdf/2509.16602v1](http://arxiv.org/pdf/2509.16602v1)**

> **作者:** Minji Heo; Simon S. Woo
>
> **摘要:** Multi-step or hybrid deepfakes, created by sequentially applying different deepfake creation methods such as Face-Swapping, GAN-based generation, and Diffusion methods, can pose an emerging and unforseen technical challenge for detection models trained on single-step forgeries. While prior studies have mainly focused on detecting isolated single manipulation, little is known about the detection model behavior under such compositional, hybrid, and complex manipulation pipelines. In this work, we introduce \textbf{FakeChain}, a large-scale benchmark comprising 1-, 2-, and 3-Step forgeries synthesized using five state-of-the-art representative generators. Using this approach, we analyze detection performance and spectral properties across hybrid manipulation at different step, along with varying generator combinations and quality settings. Surprisingly, our findings reveal that detection performance highly depends on the final manipulation type, with F1-score dropping by up to \textbf{58.83\%} when it differs from training distribution. This clearly demonstrates that detectors rely on last-stage artifacts rather than cumulative manipulation traces, limiting generalization. Such findings highlight the need for detection models to explicitly consider manipulation history and sequences. Our results highlight the importance of benchmarks such as FakeChain, reflecting growing synthesis complexity and diversity in real-world scenarios. Our sample code is available here\footnote{https://github.com/minjihh/FakeChain}.
>
---
#### [new 193] Breaking the Discretization Barrier of Continuous Physics Simulation Learning
- **分类: cs.CV**

- **简介: 该论文提出CoPS，一种纯数据驱动的方法，用于从部分观测中建模连续物理模拟。任务是解决时空连续建模问题，克服离散化限制。工作包括设计多尺度图ODE和神经自修正模块，提升复杂动力学建模能力。**

- **链接: [http://arxiv.org/pdf/2509.17955v1](http://arxiv.org/pdf/2509.17955v1)**

> **作者:** Fan Xu; Hao Wu; Nan Wang; Lilan Peng; Kun Wang; Wei Gong; Xibin Zhao
>
> **摘要:** The modeling of complicated time-evolving physical dynamics from partial observations is a long-standing challenge. Particularly, observations can be sparsely distributed in a seemingly random or unstructured manner, making it difficult to capture highly nonlinear features in a variety of scientific and engineering problems. However, existing data-driven approaches are often constrained by fixed spatial and temporal discretization. While some researchers attempt to achieve spatio-temporal continuity by designing novel strategies, they either overly rely on traditional numerical methods or fail to truly overcome the limitations imposed by discretization. To address these, we propose CoPS, a purely data-driven methods, to effectively model continuous physics simulation from partial observations. Specifically, we employ multiplicative filter network to fuse and encode spatial information with the corresponding observations. Then we customize geometric grids and use message-passing mechanism to map features from original spatial domain to the customized grids. Subsequently, CoPS models continuous-time dynamics by designing multi-scale graph ODEs, while introducing a Markov-based neural auto-correction module to assist and constrain the continuous extrapolations. Comprehensive experiments demonstrate that CoPS advances the state-of-the-art methods in space-time continuous modeling across various scenarios.
>
---
#### [new 194] Stable Video-Driven Portraits
- **分类: cs.CV**

- **简介: 该论文研究视频驱动的肖像动画任务，旨在解决传统方法在表情控制、时间一致性和泛化性上的不足。提出基于扩散模型的新框架，利用面部关键区域作为运动信号，引入时空注意力机制和信号融合策略，实现高质量、可控的肖像动画生成。**

- **链接: [http://arxiv.org/pdf/2509.17476v1](http://arxiv.org/pdf/2509.17476v1)**

> **作者:** Mallikarjun B. R.; Fei Yin; Vikram Voleti; Nikita Drobyshev; Maksim Lapin; Aaryaman Vasishta; Varun Jampani
>
> **备注:** https://stable-video-driven-portraits.github.io/
>
> **摘要:** Portrait animation aims to generate photo-realistic videos from a single source image by reenacting the expression and pose from a driving video. While early methods relied on 3D morphable models or feature warping techniques, they often suffered from limited expressivity, temporal inconsistency, and poor generalization to unseen identities or large pose variations. Recent advances using diffusion models have demonstrated improved quality but remain constrained by weak control signals and architectural limitations. In this work, we propose a novel diffusion based framework that leverages masked facial regions specifically the eyes, nose, and mouth from the driving video as strong motion control cues. To enable robust training without appearance leakage, we adopt cross identity supervision. To leverage the strong prior from the pretrained diffusion model, our novel architecture introduces minimal new parameters that converge faster and help in better generalization. We introduce spatial temporal attention mechanisms that allow inter frame and intra frame interactions, effectively capturing subtle motions and reducing temporal artifacts. Our model uses history frames to ensure continuity across segments. At inference, we propose a novel signal fusion strategy that balances motion fidelity with identity preservation. Our approach achieves superior temporal consistency and accurate expression control, enabling high-quality, controllable portrait animation suitable for real-world applications.
>
---
#### [new 195] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UniPixel，一种统一的多模态模型，旨在解决像素级视觉推理任务中细粒度感知与语言语义对齐的问题。它能灵活处理视觉输入，生成掩码并进行条件推理，在10个基准上验证了其在像素级分割、指代和对象理解等任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.18094v1](http://arxiv.org/pdf/2509.18094v1)**

> **作者:** Ye Liu; Zongyang Ma; Junfu Pu; Zhongang Qi; Yang Wu; Ying Shan; Chang Wen Chen
>
> **备注:** NeurIPS 2025 Camera Ready. Project Page: https://polyu-chenlab.github.io/unipixel/
>
> **摘要:** Recent advances in Large Multi-modal Models (LMMs) have demonstrated their remarkable success as general-purpose multi-modal assistants, with particular focuses on holistic image- and video-language understanding. Conversely, less attention has been given to scaling fine-grained pixel-level understanding capabilities, where the models are expected to realize pixel-level alignment between visual signals and language semantics. Some previous studies have applied LMMs to related tasks such as region-level captioning and referring expression segmentation. However, these models are limited to performing either referring or segmentation tasks independently and fail to integrate these fine-grained perception capabilities into visual reasoning. To bridge this gap, we propose UniPixel, a large multi-modal model capable of flexibly comprehending visual prompt inputs and generating mask-grounded responses. Our model distinguishes itself by seamlessly integrating pixel-level perception with general visual understanding capabilities. Specifically, UniPixel processes visual prompts and generates relevant masks on demand, and performs subsequent reasoning conditioning on these intermediate pointers during inference, thereby enabling fine-grained pixel-level reasoning. The effectiveness of our approach has been verified on 10 benchmarks across a diverse set of tasks, including pixel-level referring/segmentation and object-centric understanding in images/videos. A novel PixelQA task that jointly requires referring, segmentation, and question answering is also designed to verify the flexibility of our method.
>
---
#### [new 196] An Empirical Study on the Robustness of YOLO Models for Underwater Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究水下目标检测中YOLO模型的鲁棒性，针对水下畸变导致检测性能下降的问题，系统评估了YOLOv8至YOLOv12在六种模拟环境下的表现，并提出增强鲁棒性的训练策略。**

- **链接: [http://arxiv.org/pdf/2509.17561v1](http://arxiv.org/pdf/2509.17561v1)**

> **作者:** Edwine Nabahirwa; Wei Song; Minghua Zhang; Shufan Chen
>
> **备注:** 28 Pages, 12 Figures
>
> **摘要:** Underwater object detection (UOD) remains a critical challenge in computer vision due to underwater distortions which degrade low-level features and compromise the reliability of even state-of-the-art detectors. While YOLO models have become the backbone of real-time object detection, little work has systematically examined their robustness under these uniquely challenging conditions. This raises a critical question: Are YOLO models genuinely robust when operating under the chaotic and unpredictable conditions of underwater environments? In this study, we present one of the first comprehensive evaluations of recent YOLO variants (YOLOv8-YOLOv12) across six simulated underwater environments. Using a unified dataset of 10,000 annotated images from DUO and Roboflow100, we not only benchmark model robustness but also analyze how distortions affect key low-level features such as texture, edges, and color. Our findings show that (1) YOLOv12 delivers the strongest overall performance but is highly vulnerable to noise, and (2) noise disrupts edge and texture features, explaining the poor detection performance in noisy images. Class imbalance is a persistent challenge in UOD. Experiments revealed that (3) image counts and instance frequency primarily drive detection performance, while object appearance exerts only a secondary influence. Finally, we evaluated lightweight training-aware strategies: noise-aware sample injection, which improves robustness in both noisy and real-world conditions, and fine-tuning with advanced enhancement, which boosts accuracy in enhanced domains but slightly lowers performance in original data, demonstrating strong potential for domain adaptation, respectively. Together, these insights provide practical guidance for building resilient and cost-efficient UOD systems.
>
---
#### [new 197] Explainable AI for Analyzing Person-Specific Patterns in Facial Recognition Tasks
- **分类: cs.CV; cs.AI; 68T10; I.2.10; I.4.m**

- **简介: 该论文属于可解释AI任务，旨在分析人脸识别系统中个体特有的识别模式。提出了LEAM方法，通过定位关键面部区域揭示模型的识别机制，并验证了这些区域在个体间的显著差异性，为个性化隐私保护提供基础。**

- **链接: [http://arxiv.org/pdf/2509.17457v1](http://arxiv.org/pdf/2509.17457v1)**

> **作者:** Paweł Jakub Borsukiewicz; Jordan Samhi; Jacques Klein; Tegawendé F. Bissyandé
>
> **备注:** 22 pages; 24 tables; 11 figures
>
> **摘要:** The proliferation of facial recognition systems presents major privacy risks, driving the need for effective countermeasures. Current adversarial techniques apply generalized methods rather than adapting to individual facial characteristics, limiting their effectiveness and inconspicuousness. In this work, we introduce Layer Embedding Activation Mapping (LEAM), a novel technique that identifies which facial areas contribute most to recognition at an individual level. Unlike adversarial attack methods that aim to fool recognition systems, LEAM is an explainability technique designed to understand how these systems work, providing insights that could inform future privacy protection research. We integrate LEAM with a face parser to analyze data from 1000 individuals across 9 pre-trained facial recognition models. Our analysis reveals that while different layers within facial recognition models vary significantly in their focus areas, these models generally prioritize similar facial regions across architectures when considering their overall activation patterns, which show significantly higher similarity between images of the same individual (Bhattacharyya Coefficient: 0.32-0.57) vs. different individuals (0.04-0.13), validating the existence of person-specific recognition patterns. Our results show that facial recognition models prioritize the central region of face images (with nose areas accounting for 18.9-29.7% of critical recognition regions), while still distributing attention across multiple facial fragments. Proper selection of relevant facial areas was confirmed using validation occlusions, based on just 1% of the most relevant, LEAM-identified, image pixels, which proved to be transferable across different models. Our findings establish the foundation for future individually tailored privacy protection systems centered around LEAM's choice of areas to be perturbed.
>
---
#### [new 198] MedCutMix: A Data-Centric Approach to Improve Radiology Vision-Language Pre-training with Disease Awareness
- **分类: cs.CV**

- **简介: 该论文提出MedCutMix，一种面向放射学视觉-语言预训练（VLP）的多模态数据增强方法。针对医学数据多样性不足的问题，通过结合诊断文本与图像的跨模态注意力机制提升模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.16673v1](http://arxiv.org/pdf/2509.16673v1)**

> **作者:** Sinuo Wang; Yutong Xie; Yuyuan Liu; Qi Wu
>
> **摘要:** Vision-Language Pre-training (VLP) is drawing increasing interest for its ability to minimize manual annotation requirements while enhancing semantic understanding in downstream tasks. However, its reliance on image-text datasets poses challenges due to privacy concerns and the high cost of obtaining paired annotations. Data augmentation emerges as a viable strategy to address this issue, yet existing methods often fall short of capturing the subtle and complex variations in medical data due to limited diversity. To this end, we propose MedCutMix, a novel multi-modal disease-centric data augmentation method. MedCutMix performs diagnostic sentence CutMix within medical reports and establishes the cross-attention between the diagnostic sentence and medical image to guide attentive manifold mix within the imaging modality. Our approach surpasses previous methods across four downstream radiology diagnosis datasets, highlighting its effectiveness in enhancing performance and generalizability in radiology VLP.
>
---
#### [new 199] DocIQ: A Benchmark Dataset and Feature Fusion Network for Document Image Quality Assessment
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文针对文档图像质量评估（DIQA）任务，提出了DIQA-5000数据集和一个融合多级特征的无参考质量评估模型。通过引入布局特征和独立评分头，有效提升了质量预测精度，优于现有通用IQA方法。**

- **链接: [http://arxiv.org/pdf/2509.17012v1](http://arxiv.org/pdf/2509.17012v1)**

> **作者:** Zhichao Ma; Fan Huang; Lu Zhao; Fengjun Guo; Guangtao Zhai; Xiongkuo Min
>
> **摘要:** Document image quality assessment (DIQA) is an important component for various applications, including optical character recognition (OCR), document restoration, and the evaluation of document image processing systems. In this paper, we introduce a subjective DIQA dataset DIQA-5000. The DIQA-5000 dataset comprises 5,000 document images, generated by applying multiple document enhancement techniques to 500 real-world images with diverse distortions. Each enhanced image was rated by 15 subjects across three rating dimensions: overall quality, sharpness, and color fidelity. Furthermore, we propose a specialized no-reference DIQA model that exploits document layout features to maintain quality perception at reduced resolutions to lower computational cost. Recognizing that image quality is influenced by both low-level and high-level visual features, we designed a feature fusion module to extract and integrate multi-level features from document images. To generate multi-dimensional scores, our model employs independent quality heads for each dimension to predict score distributions, allowing it to learn distinct aspects of document image quality. Experimental results demonstrate that our method outperforms current state-of-the-art general-purpose IQA models on both DIQA-5000 and an additional document image dataset focused on OCR accuracy.
>
---
#### [new 200] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images
- **分类: cs.CV**

- **简介: 该论文提出DiffEye，基于扩散模型生成自然图像下连续的眼动轨迹。任务是建模真实眼动的多样性和连续性，解决传统方法丢失轨迹信息、忽视个体差异的问题。引入CPE模块，结合视觉语义特征与眼动数据，首次在自然图像上实现高质量连续眼动生成。**

- **链接: [http://arxiv.org/pdf/2509.16767v1](http://arxiv.org/pdf/2509.16767v1)**

> **作者:** Ozgur Kara; Harris Nisar; James M. Rehg
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Numerous models have been developed for scanpath and saliency prediction, which are typically trained on scanpaths, which model eye movement as a sequence of discrete fixation points connected by saccades, while the rich information contained in the raw trajectories is often discarded. Moreover, most existing approaches fail to capture the variability observed among human subjects viewing the same image. They generally predict a single scanpath of fixed, pre-defined length, which conflicts with the inherent diversity and stochastic nature of real-world visual attention. To address these challenges, we propose DiffEye, a diffusion-based training framework designed to model continuous and diverse eye movement trajectories during free viewing of natural images. Our method builds on a diffusion model conditioned on visual stimuli and introduces a novel component, namely Corresponding Positional Embedding (CPE), which aligns spatial gaze information with the patch-based semantic features of the visual input. By leveraging raw eye-tracking trajectories rather than relying on scanpaths, DiffEye captures the inherent variability in human gaze behavior and generates high-quality, realistic eye movement patterns, despite being trained on a comparatively small dataset. The generated trajectories can also be converted into scanpaths and saliency maps, resulting in outputs that more accurately reflect the distribution of human visual attention. DiffEye is the first method to tackle this task on natural images using a diffusion model while fully leveraging the richness of raw eye-tracking data. Our extensive evaluation shows that DiffEye not only achieves state-of-the-art performance in scanpath generation but also enables, for the first time, the generation of continuous eye movement trajectories. Project webpage: https://diff-eye.github.io/
>
---
#### [new 201] Uncertainty-Supervised Interpretable and Robust Evidential Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对医学图像分割中的不确定性估计问题，提出一种自监督方法。通过引入不确定性与图像梯度的关系原则，设计了两种监督损失，提升了模型预测的可解释性与鲁棒性，并在实验中验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2509.17098v1](http://arxiv.org/pdf/2509.17098v1)**

> **作者:** Yuzhu Li; An Sui; Fuping Wu; Xiahai Zhuang
>
> **摘要:** Uncertainty estimation has been widely studied in medical image segmentation as a tool to provide reliability, particularly in deep learning approaches. However, previous methods generally lack effective supervision in uncertainty estimation, leading to low interpretability and robustness of the predictions. In this work, we propose a self-supervised approach to guide the learning of uncertainty. Specifically, we introduce three principles about the relationships between the uncertainty and the image gradients around boundaries and noise. Based on these principles, two uncertainty supervision losses are designed. These losses enhance the alignment between model predictions and human interpretation. Accordingly, we introduce novel quantitative metrics for evaluating the interpretability and robustness of uncertainty. Experimental results demonstrate that compared to state-of-the-art approaches, the proposed method can achieve competitive segmentation performance and superior results in out-of-distribution (OOD) scenarios while significantly improving the interpretability and robustness of uncertainty estimation. Code is available via https://github.com/suiannaius/SURE.
>
---
#### [new 202] MirrorSAM2: Segment Mirror in Videos with Depth Perception
- **分类: cs.CV**

- **简介: 该论文提出MirrorSAM2，用于RGB-D视频中的镜面分割。针对反射模糊和纹理混淆等问题，设计了四个模块，实现自动、精细的镜面分割，在VMD和DVMD基准上达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.17220v1](http://arxiv.org/pdf/2509.17220v1)**

> **作者:** Mingchen Xu; Yukun Lai; Ze Ji; Jing Wu
>
> **备注:** 8 pages
>
> **摘要:** This paper presents MirrorSAM2, the first framework that adapts Segment Anything Model 2 (SAM2) to the task of RGB-D video mirror segmentation. MirrorSAM2 addresses key challenges in mirror detection, such as reflection ambiguity and texture confusion, by introducing four tailored modules: a Depth Warping Module for RGB and depth alignment, a Depth-guided Multi-Scale Point Prompt Generator for automatic prompt generation, a Frequency Detail Attention Fusion Module to enhance structural boundaries, and a Mirror Mask Decoder with a learnable mirror token for refined segmentation. By fully leveraging the complementarity between RGB and depth, MirrorSAM2 extends SAM2's capabilities to the prompt-free setting. To our knowledge, this is the first work to enable SAM2 for automatic video mirror segmentation. Experiments on the VMD and DVMD benchmark demonstrate that MirrorSAM2 achieves SOTA performance, even under challenging conditions such as small mirrors, weak boundaries, and strong reflections.
>
---
#### [new 203] Real-Time Fish Detection in Indonesian Marine Ecosystems Using Lightweight YOLOv10-nano Architecture
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测任务，旨在解决印尼海域鱼类监测效率低的问题。研究采用轻量级YOLOv10-nano模型，在Bunaken国家海洋公园数据上实现高精度（mAP50: 0.966）和实时性（29.29 FPS），适用于资源受限的海洋保护场景。**

- **链接: [http://arxiv.org/pdf/2509.17406v1](http://arxiv.org/pdf/2509.17406v1)**

> **作者:** Jonathan Wuntu; Muhamad Dwisnanto Putro; Rendy Syahputra
>
> **摘要:** Indonesia's marine ecosystems, part of the globally recognized Coral Triangle, are among the richest in biodiversity, requiring efficient monitoring tools to support conservation. Traditional fish detection methods are time-consuming and demand expert knowledge, prompting the need for automated solutions. This study explores the implementation of YOLOv10-nano, a state-of-the-art deep learning model, for real-time marine fish detection in Indonesian waters, using test data from Bunaken National Marine Park. YOLOv10's architecture, featuring improvements like the CSPNet backbone, PAN for feature fusion, and Pyramid Spatial Attention Block, enables efficient and accurate object detection even in complex environments. The model was evaluated on the DeepFish and OpenImages V7-Fish datasets. Results show that YOLOv10-nano achieves a high detection accuracy with mAP50 of 0.966 and mAP50:95 of 0.606 while maintaining low computational demand (2.7M parameters, 8.4 GFLOPs). It also delivered an average inference speed of 29.29 FPS on the CPU, making it suitable for real-time deployment. Although OpenImages V7-Fish alone provided lower accuracy, it complemented DeepFish in enhancing model robustness. Overall, this study demonstrates YOLOv10-nano's potential for efficient, scalable marine fish monitoring and conservation applications in data-limited environments.
>
---
#### [new 204] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文提出HyRF，用于高效高质量的新视角合成。针对3DGS内存消耗大的问题，结合显式高斯和神经场，分解场景参数并设计混合渲染方案，在保持实时性能的同时显著减小模型规模并提升细节重建质量。**

- **链接: [http://arxiv.org/pdf/2509.17083v1](http://arxiv.org/pdf/2509.17083v1)**

> **作者:** Zipeng Wang; Dan Xu
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS) has emerged as a powerful alternative to NeRF-based approaches, enabling real-time, high-quality novel view synthesis through explicit, optimizable 3D Gaussians. However, 3DGS suffers from significant memory overhead due to its reliance on per-Gaussian parameters to model view-dependent effects and anisotropic shapes. While recent works propose compressing 3DGS with neural fields, these methods struggle to capture high-frequency spatial variations in Gaussian properties, leading to degraded reconstruction of fine details. We present Hybrid Radiance Fields (HyRF), a novel scene representation that combines the strengths of explicit Gaussians and neural fields. HyRF decomposes the scene into (1) a compact set of explicit Gaussians storing only critical high-frequency parameters and (2) grid-based neural fields that predict remaining properties. To enhance representational capacity, we introduce a decoupled neural field architecture, separately modeling geometry (scale, opacity, rotation) and view-dependent color. Additionally, we propose a hybrid rendering scheme that composites Gaussian splatting with a neural field-predicted background, addressing limitations in distant scene representation. Experiments demonstrate that HyRF achieves state-of-the-art rendering quality while reducing model size by over 20 times compared to 3DGS and maintaining real-time performance. Our project page is available at https://wzpscott.github.io/hyrf/.
>
---
#### [new 205] Pre-Trained CNN Architecture for Transformer-Based Image Caption Generation Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像描述生成任务，旨在解决传统RNN/LSTM模型训练慢、长序列信息保留差的问题。工作包括：基于Transformer构建并训练图像描述模型，结合EfficientNetB0提取图像特征，并在Flickr30k数据集上实现高效并行化处理。**

- **链接: [http://arxiv.org/pdf/2509.17365v1](http://arxiv.org/pdf/2509.17365v1)**

> **作者:** Amanuel Tafese Dufera
>
> **摘要:** Automatic image captioning, a multifaceted task bridging computer vision and natural lan- guage processing, aims to generate descriptive textual content from visual input. While Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks have achieved significant advancements, they present limitations. The inherent sequential nature of RNNs leads to sluggish training and inference times. LSTMs further struggle with retaining information from earlier sequence elements when dealing with very long se- quences. This project presents a comprehensive guide to constructing and comprehending transformer models for image captioning. Transformers employ self-attention mechanisms, capturing both short- and long-range dependencies within the data. This facilitates efficient parallelization during both training and inference phases. We leverage the well-established Transformer architecture, recognized for its effectiveness in managing sequential data, and present a meticulous methodology. Utilizing the Flickr30k dataset, we conduct data pre- processing, construct a model architecture that integrates an EfficientNetB0 CNN for fea- ture extraction, and train the model with attention mechanisms incorporated. Our approach exemplifies the utilization of parallelization for efficient training and inference. You can find the project on GitHub.
>
---
#### [new 206] ADVEDM:Fine-grained Adversarial Attack against VLM-based Embodied Agents
- **分类: cs.CV**

- **简介: 该论文针对基于视觉-语言模型（VLM）的具身智能体，提出ADVEDM攻击框架。旨在解决现有对抗攻击依赖强假设或破坏语义导致无效输出的问题。通过精细修改关键物体感知，在物理世界中引发有效但错误的决策，提升攻击实用性与威胁性。**

- **链接: [http://arxiv.org/pdf/2509.16645v1](http://arxiv.org/pdf/2509.16645v1)**

> **作者:** Yichen Wang; Hangtao Zhang; Hewen Pan; Ziqi Zhou; Xianlong Wang; Peijin Guo; Lulu Xue; Shengshan Hu; Minghui Li; Leo Yu Zhang
>
> **摘要:** Vision-Language Models (VLMs), with their strong reasoning and planning capabilities, are widely used in embodied decision-making (EDM) tasks in embodied agents, such as autonomous driving and robotic manipulation. Recent research has increasingly explored adversarial attacks on VLMs to reveal their vulnerabilities. However, these attacks either rely on overly strong assumptions, requiring full knowledge of the victim VLM, which is impractical for attacking VLM-based agents, or exhibit limited effectiveness. The latter stems from disrupting most semantic information in the image, which leads to a misalignment between the perception and the task context defined by system prompts. This inconsistency interrupts the VLM's reasoning process, resulting in invalid outputs that fail to affect interactions in the physical world. To this end, we propose a fine-grained adversarial attack framework, ADVEDM, which modifies the VLM's perception of only a few key objects while preserving the semantics of the remaining regions. This attack effectively reduces conflicts with the task context, making VLMs output valid but incorrect decisions and affecting the actions of agents, thus posing a more substantial safety threat in the physical world. We design two variants of based on this framework, ADVEDM-R and ADVEDM-A, which respectively remove the semantics of a specific object from the image and add the semantics of a new object into the image. The experimental results in both general scenarios and EDM tasks demonstrate fine-grained control and excellent attack performance.
>
---
#### [new 207] Diff-GNSS: Diffusion-based Pseudorange Error Estimation
- **分类: cs.CV; cs.ET**

- **简介: 该论文提出Diff-GNSS，一种基于扩散模型的伪距误差估计方法，旨在解决GNSS在复杂环境中因多径效应导致的定位精度下降问题。通过粗到精的框架，结合Mamba模块与条件扩散模型，提升了误差预测的准确性与可控性。**

- **链接: [http://arxiv.org/pdf/2509.17397v1](http://arxiv.org/pdf/2509.17397v1)**

> **作者:** Jiaqi Zhu; Shouyi Lu; Ziyao Li; Guirong Zhuo; Lu Xiong
>
> **摘要:** Global Navigation Satellite Systems (GNSS) are vital for reliable urban positioning. However, multipath and non-line-of-sight reception often introduce large measurement errors that degrade accuracy. Learning-based methods for predicting and compensating pseudorange errors have gained traction, but their performance is limited by complex error distributions. To address this challenge, we propose Diff-GNSS, a coarse-to-fine GNSS measurement (pseudorange) error estimation framework that leverages a conditional diffusion model to capture such complex distributions. Firstly, a Mamba-based module performs coarse estimation to provide an initial prediction with appropriate scale and trend. Then, a conditional denoising diffusion layer refines the estimate, enabling fine-grained modeling of pseudorange errors. To suppress uncontrolled generative diversity and achieve controllable synthesis, three key features related to GNSS measurement quality are used as conditions to precisely guide the reverse denoising process. We further incorporate per-satellite uncertainty modeling within the diffusion stage to assess the reliability of the predicted errors. We have collected and publicly released a real-world dataset covering various scenes. Experiments on public and self-collected datasets show that DiffGNSS consistently outperforms state-of-the-art baselines across multiple metrics. To the best of our knowledge, this is the first application of diffusion models to pseudorange error estimation. The proposed diffusion-based refinement module is plug-and-play and can be readily integrated into existing networks to markedly improve estimation accuracy.
>
---
#### [new 208] SAM-DCE: Addressing Token Uniformity and Semantic Over-Smoothing in Medical Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割任务，旨在解决SAM模型在医疗领域中的token均匀性与语义过平滑问题。提出SAM-DCE方法，在提升类间可分性的同时增强局部判别和全局语义一致性。**

- **链接: [http://arxiv.org/pdf/2509.16886v1](http://arxiv.org/pdf/2509.16886v1)**

> **作者:** Yingzhen Hu; Yiheng Zhong; Ruobing Li; Yingxue Su; Jiabao An; Feilong Tang; Jionglong Su; Imran Razzak
>
> **摘要:** The Segment Anything Model (SAM) demonstrates impressive zero-shot segmentation ability on natural images but encounters difficulties in medical imaging due to domain shifts, anatomical variability, and its reliance on user-provided prompts. Recent prompt-free adaptations alleviate the need for expert intervention, yet still suffer from limited robustness and adaptability, often overlooking the issues of semantic over-smoothing and token uniformity. We propose SAM-DCE, which balances local discrimination and global semantics while mitigating token uniformity, enhancing inter-class separability, and enriching mask decoding with fine-grained, consistent representations. Extensive experiments on diverse medical benchmarks validate its effectiveness.
>
---
#### [new 209] CARINOX: Inference-time Scaling with Category-Aware Reward-based Initial Noise Optimization and Exploration
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成中的组合对齐问题，提出CARINOX框架，在推理阶段结合噪声优化与探索，并通过类别感知的奖励函数提升生成效果。实验表明其在多个基准上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.17458v1](http://arxiv.org/pdf/2509.17458v1)**

> **作者:** Seyed Amir Kasaei; Ali Aghayari; Arash Marioriyad; Niki Sepasian; Shayan Baghayi Nejad; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **摘要:** Text-to-image diffusion models, such as Stable Diffusion, can produce high-quality and diverse images but often fail to achieve compositional alignment, particularly when prompts describe complex object relationships, attributes, or spatial arrangements. Recent inference-time approaches address this by optimizing or exploring the initial noise under the guidance of reward functions that score text-image alignment without requiring model fine-tuning. While promising, each strategy has intrinsic limitations when used alone: optimization can stall due to poor initialization or unfavorable search trajectories, whereas exploration may require a prohibitively large number of samples to locate a satisfactory output. Our analysis further shows that neither single reward metrics nor ad-hoc combinations reliably capture all aspects of compositionality, leading to weak or inconsistent guidance. To overcome these challenges, we present Category-Aware Reward-based Initial Noise Optimization and Exploration (CARINOX), a unified framework that combines noise optimization and exploration with a principled reward selection procedure grounded in correlation with human judgments. Evaluations on two complementary benchmarks covering diverse compositional challenges show that CARINOX raises average alignment scores by +16% on T2I-CompBench++ and +11% on the HRS benchmark, consistently outperforming state-of-the-art optimization and exploration-based methods across all major categories, while preserving image quality and diversity. The project page is available at https://amirkasaei.com/carinox/{this URL}.
>
---
#### [new 210] SISMA: Semantic Face Image Synthesis with Mamba
- **分类: cs.CV**

- **简介: 该论文提出SISMA，一种基于Mamba的语义人脸图像生成模型，用于语义图像合成任务。旨在解决扩散模型计算成本高、速度慢的问题，通过语义掩码控制形状，实现高质量生成且运算更高效。**

- **链接: [http://arxiv.org/pdf/2509.17651v1](http://arxiv.org/pdf/2509.17651v1)**

> **作者:** Filippo Botti; Alex Ergasti; Tomaso Fontanini; Claudio Ferrari; Massimo Bertozzi; Andrea Prati
>
> **摘要:** Diffusion Models have become very popular for Semantic Image Synthesis (SIS) of human faces. Nevertheless, their training and inference is computationally expensive and their computational requirements are high due to the quadratic complexity of attention layers. In this paper, we propose a novel architecture called SISMA, based on the recently proposed Mamba. SISMA generates high quality samples by controlling their shape using a semantic mask at a reduced computational demand. We validated our approach through comprehensive experiments with CelebAMask-HQ, revealing that our architecture not only achieves a better FID score yet also operates at three times the speed of state-of-the-art architectures. This indicates that the proposed design is a viable, lightweight substitute to transformer-based models.
>
---
#### [new 211] Neural-MMGS: Multi-modal Neural Gaussian Splats for Large-Scale Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出Neural-MMGS，用于大规模场景重建任务。针对多模态数据（图像、LiDAR、语义）融合效率低的问题，设计了一种紧凑的可学习嵌入方式，将多模态信息编码到高斯分布中，从而降低内存消耗并提升重建质量。**

- **链接: [http://arxiv.org/pdf/2509.17762v1](http://arxiv.org/pdf/2509.17762v1)**

> **作者:** Sitian Shen; Georgi Pramatarov; Yifu Tao; Daniele De Martini
>
> **摘要:** This paper proposes Neural-MMGS, a novel neural 3DGS framework for multimodal large-scale scene reconstruction that fuses multiple sensing modalities in a per-gaussian compact, learnable embedding. While recent works focusing on large-scale scene reconstruction have incorporated LiDAR data to provide more accurate geometric constraints, we argue that LiDAR's rich physical properties remain underexplored. Similarly, semantic information has been used for object retrieval, but could provide valuable high-level context for scene reconstruction. Traditional approaches append these properties to Gaussians as separate parameters, increasing memory usage and limiting information exchange across modalities. Instead, our approach fuses all modalities -- image, LiDAR, and semantics -- into a compact, learnable embedding that implicitly encodes optical, physical, and semantic features in each Gaussian. We then train lightweight neural decoders to map these embeddings to Gaussian parameters, enabling the reconstruction of each sensing modality with lower memory overhead and improved scalability. We evaluate Neural-MMGS on the Oxford Spires and KITTI-360 datasets. On Oxford Spires, we achieve higher-quality reconstructions, while on KITTI-360, our method reaches competitive results with less storage consumption compared with current approaches in LiDAR-based novel-view synthesis.
>
---
#### [new 212] Active View Selection for Scene-level Multi-view Crowd Counting and Localization with Limited Labels
- **分类: cs.CV**

- **简介: 该论文研究多视角人群计数与定位任务，旨在解决场景级视图选择问题。提出独立视图选择（IVS）和主动视图选择（AVS）方法，在有限标注和跨场景设置下优化视图选择与下游任务联合优化，提升计数与定位性能。**

- **链接: [http://arxiv.org/pdf/2509.16684v1](http://arxiv.org/pdf/2509.16684v1)**

> **作者:** Qi Zhang; Bin Li; Antoni B. Chan; Hui Huang
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Multi-view crowd counting and localization fuse the input multi-views for estimating the crowd number or locations on the ground. Existing methods mainly focus on accurately predicting on the crowd shown in the input views, which neglects the problem of choosing the `best' camera views to perceive all crowds well in the scene. Besides, existing view selection methods require massive labeled views and images, and lack the ability for cross-scene settings, reducing their application scenarios. Thus, in this paper, we study the view selection issue for better scene-level multi-view crowd counting and localization results with cross-scene ability and limited label demand, instead of input-view-level results. We first propose an independent view selection method (IVS) that considers view and scene geometries in the view selection strategy and conducts the view selection, labeling, and downstream tasks independently. Based on IVS, we also put forward an active view selection method (AVS) that jointly optimizes the view selection, labeling, and downstream tasks. In AVS, we actively select the labeled views and consider both the view/scene geometries and the predictions of the downstream task models in the view selection process. Experiments on multi-view counting and localization tasks demonstrate the cross-scene and the limited label demand advantages of the proposed active view selection method (AVS), outperforming existing methods and with wider application scenarios.
>
---
#### [new 213] VideoArtGS: Building Digital Twins of Articulated Objects from Monocular Video
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VideoArtGS，旨在从单目视频中重建可动对象的高精度数字孪生。任务包括几何重建、部件分割与运动参数估计。工作重点在于设计运动先验引导和混合中心-网格模块，以提升重建精度并减少误差。**

- **链接: [http://arxiv.org/pdf/2509.17647v1](http://arxiv.org/pdf/2509.17647v1)**

> **作者:** Yu Liu; Baoxiong Jia; Ruijie Lu; Chuyue Gan; Huayu Chen; Junfeng Ni; Song-Chun Zhu; Siyuan Huang
>
> **摘要:** Building digital twins of articulated objects from monocular video presents an essential challenge in computer vision, which requires simultaneous reconstruction of object geometry, part segmentation, and articulation parameters from limited viewpoint inputs. Monocular video offers an attractive input format due to its simplicity and scalability; however, it's challenging to disentangle the object geometry and part dynamics with visual supervision alone, as the joint movement of the camera and parts leads to ill-posed estimation. While motion priors from pre-trained tracking models can alleviate the issue, how to effectively integrate them for articulation learning remains largely unexplored. To address this problem, we introduce VideoArtGS, a novel approach that reconstructs high-fidelity digital twins of articulated objects from monocular video. We propose a motion prior guidance pipeline that analyzes 3D tracks, filters noise, and provides reliable initialization of articulation parameters. We also design a hybrid center-grid part assignment module for articulation-based deformation fields that captures accurate part motion. VideoArtGS demonstrates state-of-the-art performance in articulation and mesh reconstruction, reducing the reconstruction error by about two orders of magnitude compared to existing methods. VideoArtGS enables practical digital twin creation from monocular video, establishing a new benchmark for video-based articulated object reconstruction. Our work is made publicly available at: https://videoartgs.github.io.
>
---
#### [new 214] From Coated to Uncoated: Scanning Electron Microscopy Corrections to Estimate True Surface Pore Size in Nanoporous Membranes
- **分类: cond-mat.mtrl-sci; cs.CV; physics.app-ph; physics.chem-ph; physics.ins-det**

- **简介: 该论文研究SEM成像条件对纳米多孔膜表面孔隙率和孔径测量的影响，提出数字膨胀校正方法，揭示真实孔结构。任务是修正SEM测量偏差，解决孔隙低估问题。**

- **链接: [http://arxiv.org/pdf/2509.16471v1](http://arxiv.org/pdf/2509.16471v1)**

> **作者:** Sima Zeinali Danalou; Dian Yu; Niher R. Sarker; Hooman Chamani; Jane Y. Howe; Patrick C. Lee; Jay R. Werber
>
> **摘要:** Scanning electron microscopy (SEM) is the premier method for characterizing the nanoscale surface pores in ultrafiltration (UF) membranes and the support layers of reverse osmosis (RO) membranes. Based on SEM, the conventional understanding is that membranes typically have low surface porosities of <10%. We hypothesized that high acceleration voltage during SEM imaging and sputter metal coatings required for SEM have led to systematic underestimations of porosity and pore size. We showed that imaging a commercial UF membrane at 1, 5, and 10 kV reduced measured porosity from 10.3% (1 kV) to 6.3% (10 kV), while increasing Pt coating thickness from 1.5 to 5 nm lowered porosity by 54% for the UF membrane (12.9% to 5.8%) and 46% for an RO support (13.1% to 7.0%). To account for coating thickness, we developed a digital correction method that simulates pore dilation, enabling the pore structure to be estimated for uncoated membranes. Dilation yielded uncoated porosity values of 23% for the UF membrane and 20% for the RO support, about 3-fold greater than values observed with a 4 nm coating. Mean pore diameters were 2-fold greater for the UF membrane and 1.5-fold greater for the RO support. Critically, dilation-derived pore-size distributions agreed with low-flux dextran-retention data fitted with the Bungay-Brenner model. Our results suggest that surface porosities and pore sizes of nanoporous membranes are much larger than previously understood, with major implications for structure/transport relationships. For future nanoscale pore analysis of membranes (and other nanoporous materials), we recommend low acceleration voltage (1 kV), minimal coatings (1-2 nm), and digital dilation to account for coating artifacts
>
---
#### [new 215] LenslessMic: Audio Encryption and Authentication via Lensless Computational Imaging
- **分类: cs.CR; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出LenslessMic，一种基于无透镜相机的音频加密与认证方法。通过光学硬件实现物理层安全，解决传统音频加密依赖软件的问题，提供高安全性与音质保障，并通过低成本原型验证效果。**

- **链接: [http://arxiv.org/pdf/2509.16418v1](http://arxiv.org/pdf/2509.16418v1)**

> **作者:** Petr Grinberg; Eric Bezzam; Paolo Prandoni; Martin Vetterli
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** With society's increasing reliance on digital data sharing, the protection of sensitive information has become critical. Encryption serves as one of the privacy-preserving methods; however, its realization in the audio domain predominantly relies on signal processing or software methods embedded into hardware. In this paper, we introduce LenslessMic, a hybrid optical hardware-based encryption method that utilizes a lensless camera as a physical layer of security applicable to multiple types of audio. We show that LenslessMic enables (1) robust authentication of audio recordings and (2) encryption strength that can rival the search space of 256-bit digital standards, while maintaining high-quality signals and minimal loss of content information. The approach is validated with a low-cost Raspberry Pi prototype and is open-sourced together with datasets to facilitate research in the area.
>
---
#### [new 216] MRADNET: a Compact Radar Object Detector with MetaFormer
- **分类: eess.SP; cs.CV**

- **简介: 该论文提出mRadNet，一种用于雷达目标检测的轻量级模型。针对车载嵌入式系统对模型紧凑性和效率的需求，采用U-net结构与MetaFormer模块，结合分离卷积和注意力机制，提升检测性能并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.16223v1](http://arxiv.org/pdf/2509.16223v1)**

> **作者:** Huaiyu Chen; Fahed Hassanat; Robert Laganiere; Martin Bouchard
>
> **备注:** 5 pages, 2 figures, submitted to IEEE Icassp 2026
>
> **摘要:** Frequency-modulated continuous wave radars have gained increasing popularity in the automotive industry. Its robustness against adverse weather conditions makes it a suitable choice for radar object detection in advanced driver assistance systems. These real-time embedded systems have requirements for the compactness and efficiency of the model, which have been largely overlooked in previous work. In this work, we propose mRadNet, a novel radar object detection model with compactness in mind. mRadNet employs a U-net style architecture with MetaFormer blocks, in which separable convolution and attention token mixers are used to capture both local and global features effectively. More efficient token embedding and merging strategies are introduced to further facilitate the lightweight design of the model. The performance of mRadNet is validated on the CRUW dataset, improving state-of-the-art performance.
>
---
#### [new 217] Beat on Gaze: Learning Stylized Generation of Gaze and Head Dynamics
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于音频驱动的3D面部动画任务，旨在解决现有方法忽视注视、头部运动与语音之间协调的问题。提出StyGazeTalk方法，通过LSTM和风格编码器生成多样化的注视与头部动态，并构建高精度多模态数据集，提升动画的真实性和风格表现力。**

- **链接: [http://arxiv.org/pdf/2509.17168v1](http://arxiv.org/pdf/2509.17168v1)**

> **作者:** Chengwei Shi; Chong Cao; Xin Tong; Xukun Shen
>
> **备注:** arXiv submission
>
> **摘要:** Head and gaze dynamics are crucial in expressive 3D facial animation for conveying emotion and intention. However, existing methods frequently address facial components in isolation, overlooking the intricate coordination between gaze, head motion, and speech. The scarcity of high-quality gaze-annotated datasets hinders the development of data-driven models capable of capturing realistic, personalized gaze control. To address these challenges, we propose StyGazeTalk, an audio-driven method that generates synchronized gaze and head motion styles. We extract speaker-specific motion traits from gaze-head sequences with a multi-layer LSTM structure incorporating a style encoder, enabling the generation of diverse animation styles. We also introduce a high-precision multimodal dataset comprising eye-tracked gaze, audio, head pose, and 3D facial parameters, providing a valuable resource for training and evaluating head and gaze control models. Experimental results demonstrate that our method generates realistic, temporally coherent, and style-aware head-gaze motions, significantly advancing the state-of-the-art in audio-driven facial animation.
>
---
#### [new 218] FlagEval Findings Report: A Preliminary Evaluation of Large Reasoning Models on Automatically Verifiable Textual and Visual Questions
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于对大型推理模型的评估任务，旨在测试其在文本和视觉问题上的推理能力。作者构建了ROME基准，提供无污染的数据评估视觉语言模型，发布评测数据与结果，推动模型推理性能研究。**

- **链接: [http://arxiv.org/pdf/2509.17177v1](http://arxiv.org/pdf/2509.17177v1)**

> **作者:** Bowen Qin; Chen Yue; Fang Yin; Hui Wang; JG Yao; Jiakang Liu; Jing-Shu Zheng; Miguel Hu Chen; Richeng Xuan; Shibei Meng; Shiqi Zhou; Teng Dai; Tong-Shuai Ren; Wei Cui; Xi Yang; Xialin Du; Xiaojing Xu; Xue Sun; Xuejing Li; Yaming Liu; Yesheng Liu; Ying Liu; Yonghua Lin; Yu Zhao; Yunduo Zhang; Yuwen Luo; Zheqi He; Zhiyuan He; Zhongyuan Wang
>
> **备注:** 23 pages in main text
>
> **摘要:** We conduct a moderate-scale contamination-free (to some extent) evaluation of current large reasoning models (LRMs) with some preliminary findings. We also release ROME, our evaluation benchmark for vision language models intended to test reasoning from visual clues. We attach links to the benchmark, evaluation data, and other updates on this website: https://flageval-baai.github.io/LRM-Eval/
>
---
#### [new 219] Is It Certainly a Deepfake? Reliability Analysis in Detection & Generation Ecosystem
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于深度伪造检测任务，旨在解决检测器预测不确定性问题。通过分析生成与检测过程中的不确定性，提出基于贝叶斯神经网络和蒙特卡洛dropout的方法，量化不确定性并提升检测可靠性，为可信合成媒体检测提供关键支持。**

- **链接: [http://arxiv.org/pdf/2509.17550v1](http://arxiv.org/pdf/2509.17550v1)**

> **作者:** Neslihan Kose; Anthony Rhodes; Umur Aybars Ciftci; Ilke Demir
>
> **备注:** Accepted for publication at the ICCV 2025 STREAM workshop
>
> **摘要:** As generative models are advancing in quality and quantity for creating synthetic content, deepfakes begin to cause online mistrust. Deepfake detectors are proposed to counter this effect, however, misuse of detectors claiming fake content as real or vice versa further fuels this misinformation problem. We present the first comprehensive uncertainty analysis of deepfake detectors, systematically investigating how generative artifacts influence prediction confidence. As reflected in detectors' responses, deepfake generators also contribute to this uncertainty as their generative residues vary, so we cross the uncertainty analysis of deepfake detectors and generators. Based on our observations, the uncertainty manifold holds enough consistent information to leverage uncertainty for deepfake source detection. Our approach leverages Bayesian Neural Networks and Monte Carlo dropout to quantify both aleatoric and epistemic uncertainties across diverse detector architectures. We evaluate uncertainty on two datasets with nine generators, with four blind and two biological detectors, compare different uncertainty methods, explore region- and pixel-based uncertainty, and conduct ablation studies. We conduct and analyze binary real/fake, multi-class real/fake, source detection, and leave-one-out experiments between the generator/detector combinations to share their generalization capability, model calibration, uncertainty, and robustness against adversarial attacks. We further introduce uncertainty maps that localize prediction confidence at the pixel level, revealing distinct patterns correlated with generator-specific artifacts. Our analysis provides critical insights for deploying reliable deepfake detection systems and establishes uncertainty quantification as a fundamental requirement for trustworthy synthetic media detection.
>
---
#### [new 220] Event-Based Visual Teach-and-Repeat via Fast Fourier-Domain Cross-Correlation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出首个基于事件相机的视觉教-复现导航系统，解决传统帧相机因固定帧率导致的响应延迟问题。通过频域互相关方法实现高速（>300Hz）事件流匹配，提升机器人实时定位与导航性能。**

- **链接: [http://arxiv.org/pdf/2509.17287v1](http://arxiv.org/pdf/2509.17287v1)**

> **作者:** Gokul B. Nair; Alejandro Fontan; Michael Milford; Tobias Fischer
>
> **备注:** 8 Pages, 4 Figures, Under Review
>
> **摘要:** Visual teach-and-repeat navigation enables robots to autonomously traverse previously demonstrated paths by comparing current sensory input with recorded trajectories. However, conventional frame-based cameras fundamentally limit system responsiveness: their fixed frame rates (typically 30-60 Hz) create inherent latency between environmental changes and control responses. Here we present the first event-camera-based visual teach-and-repeat system. To achieve this, we develop a frequency-domain cross-correlation framework that transforms the event stream matching problem into computationally efficient Fourier space multiplications, capable of exceeding 300Hz processing rates, an order of magnitude faster than frame-based approaches. By exploiting the binary nature of event frames and applying image compression techniques, we further enhance the computational speed of the cross-correlation process without sacrificing localization accuracy. Extensive experiments using a Prophesee EVK4 HD event camera mounted on an AgileX Scout Mini robot demonstrate successful autonomous navigation across 4000+ meters of indoor and outdoor trajectories. Our system achieves ATEs below 24 cm while maintaining consistent high-frequency control updates. Our evaluations show that our approach achieves substantially higher update rates compared to conventional frame-based systems, underscoring the practical viability of event-based perception for real-time robotic navigation.
>
---
#### [new 221] Qwen3-Omni Technical Report
- **分类: cs.CL; cs.AI; cs.CV; eess.AS**

- **简介: 该论文介绍了Qwen3-Omni，一种在文本、图像、音频和视频任务中均保持SOTA性能的多模态模型。通过Thinker-Talker MoE架构统一感知与生成，优化了流式合成延迟，并引入专用模型增强多模态推理与音频描述能力。**

- **链接: [http://arxiv.org/pdf/2509.17765v1](http://arxiv.org/pdf/2509.17765v1)**

> **作者:** Jin Xu; Zhifang Guo; Hangrui Hu; Yunfei Chu; Xiong Wang; Jinzheng He; Yuxuan Wang; Xian Shi; Ting He; Xinfa Zhu; Yuanjun Lv; Yongqi Wang; Dake Guo; He Wang; Linhan Ma; Pei Zhang; Xinyu Zhang; Hongkun Hao; Zishan Guo; Baosong Yang; Bin Zhang; Ziyang Ma; Xipin Wei; Shuai Bai; Keqin Chen; Xuejing Liu; Peng Wang; Mingkun Yang; Dayiheng Liu; Xingzhang Ren; Bo Zheng; Rui Men; Fan Zhou; Bowen Yu; Jianxin Yang; Le Yu; Jingren Zhou; Junyang Lin
>
> **备注:** https://github.com/QwenLM/Qwen3-Omni
>
> **摘要:** We present Qwen3-Omni, a single multimodal model that, for the first time, maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to single-modal counterparts. Qwen3-Omni matches the performance of same-sized single-modal models within the Qwen series and excels particularly on audio tasks. Across 36 audio and audio-visual benchmarks, Qwen3-Omni achieves open-source SOTA on 32 benchmarks and overall SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker MoE architecture that unifies perception and generation across text, images, audio, and video, yielding fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. To reduce first-packet latency in streaming synthesis, Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme. Leveraging the representational capacity of these codebooks, we replace computationally intensive block-wise diffusion with a lightweight causal ConvNet, enabling streaming from the first codec frame. In cold-start settings, Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms. To further strengthen multimodal reasoning, we introduce a Thinking model that explicitly reasons over inputs from any modality. Since the research community currently lacks a general-purpose audio captioning model, we fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which produces detailed, low-hallucination captions for arbitrary audio inputs. Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, and Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0 license.
>
---
#### [new 222] SOLAR: Switchable Output Layer for Accuracy and Robustness in Once-for-All Training
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对Once-for-All训练中因参数共享导致性能下降的问题，提出SOLAR方法，为每个子网络分配独立分类头，提升准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16833v1](http://arxiv.org/pdf/2509.16833v1)**

> **作者:** Shaharyar Ahmed Khan Tareen; Lei Fan; Xiaojing Yuan; Qin Lin; Bin Hu
>
> **备注:** 10 pages, 7 figures, 6 tables
>
> **摘要:** Once-for-All (OFA) training enables a single super-net to generate multiple sub-nets tailored to diverse deployment scenarios, supporting flexible trade-offs among accuracy, robustness, and model-size without retraining. However, as the number of supported sub-nets increases, excessive parameter sharing in the backbone limits representational capacity, leading to degraded calibration and reduced overall performance. To address this, we propose SOLAR (Switchable Output Layer for Accuracy and Robustness in Once-for-All Training), a simple yet effective technique that assigns each sub-net a separate classification head. By decoupling the logit learning process across sub-nets, the Switchable Output Layer (SOL) reduces representational interference and improves optimization, without altering the shared backbone. We evaluate SOLAR on five datasets (SVHN, CIFAR-10, STL-10, CIFAR-100, and TinyImageNet) using four super-net backbones (ResNet-34, WideResNet-16-8, WideResNet-40-2, and MobileNetV2) for two OFA training frameworks (OATS and SNNs). Experiments show that SOLAR outperforms the baseline methods: compared to OATS, it improves accuracy of sub-nets up to 1.26 %, 4.71 %, 1.67 %, and 1.76 %, and robustness up to 9.01 %, 7.71 %, 2.72 %, and 1.26 % on SVHN, CIFAR-10, STL-10, and CIFAR-100, respectively. Compared to SNNs, it improves TinyImageNet accuracy by up to 2.93 %, 2.34 %, and 1.35 % using ResNet-34, WideResNet-16-8, and MobileNetV2 backbones (with 8 sub-nets), respectively.
>
---
#### [new 223] Development of a Mobile Application for at-Home Analysis of Retinal Fundus Images
- **分类: cs.HC; cs.CV**

- **简介: 该论文提出一款移动应用，用于居家监测视网膜眼底图像与年龄相关疾病相关的指标。通过机器学习模型分析血管扭曲度、糖尿病视网膜病变、青光眼和黄斑水肿等指标，实现趋势观察，无需直接诊断。**

- **链接: [http://arxiv.org/pdf/2509.16814v1](http://arxiv.org/pdf/2509.16814v1)**

> **作者:** Mattea Reid; Zuhairah Zainal; Khaing Zin Than; Danielle Chan; Jonathan Chan
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Machine learning is gaining significant attention as a diagnostic tool in medical imaging, particularly in the analysis of retinal fundus images. However, this approach is not yet clinically applicable, as it still depends on human validation from a professional. Therefore, we present the design for a mobile application that monitors metrics related to retinal fundus images correlating to age-related conditions. The purpose of this platform is to observe for a change in these metrics over time, offering early insights into potential ocular diseases without explicitly delivering diagnostics. Metrics analysed include vessel tortuosity, as well as signs of glaucoma, retinopathy and macular edema. To evaluate retinopathy grade and risk of macular edema, a model was trained on the Messidor dataset and compared to a similar model trained on the MAPLES-DR dataset. Information from the DeepSeeNet glaucoma detection model, as well as tortuosity calculations, is additionally incorporated to ultimately present a retinal fundus image monitoring platform. As a result, the mobile application permits monitoring of trends or changes in ocular metrics correlated to age-related conditions with regularly uploaded photographs.
>
---
#### [new 224] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DriveDPO，用于端到端自动驾驶策略学习。针对模仿学习方法安全性不足的问题，通过安全评分与人类模仿的联合优化，实现轨迹级偏好对齐，提升驾驶安全性与可靠性，在NAVSIM基准上取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.17940v1](http://arxiv.org/pdf/2509.17940v1)**

> **作者:** Shuyao Shang; Yuntao Chen; Yuqi Wang; Yingyan Li; Zhaoxiang Zhang
>
> **备注:** NeurIPS 2025
>
> **摘要:** End-to-end autonomous driving has substantially progressed by directly predicting future trajectories from raw perception inputs, which bypasses traditional modular pipelines. However, mainstream methods trained via imitation learning suffer from critical safety limitations, as they fail to distinguish between trajectories that appear human-like but are potentially unsafe. Some recent approaches attempt to address this by regressing multiple rule-driven scores but decoupling supervision from policy optimization, resulting in suboptimal performance. To tackle these challenges, we propose DriveDPO, a Safety Direct Preference Optimization Policy Learning framework. First, we distill a unified policy distribution from human imitation similarity and rule-based safety scores for direct policy optimization. Further, we introduce an iterative Direct Preference Optimization stage formulated as trajectory-level preference alignment. Extensive experiments on the NAVSIM benchmark demonstrate that DriveDPO achieves a new state-of-the-art PDMS of 90.0. Furthermore, qualitative results across diverse challenging scenarios highlight DriveDPO's ability to produce safer and more reliable driving behaviors.
>
---
#### [new 225] ViTCAE: ViT-based Class-conditioned Autoencoder
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出ViTCAE，一种基于Vision Transformer的类条件自编码器框架。针对现有方法中Class token利用率低和注意力机制静态的问题，重新定义Class token为生成核心，并引入动态注意力机制和自适应温度调度，提升生成控制与效率。**

- **链接: [http://arxiv.org/pdf/2509.16554v1](http://arxiv.org/pdf/2509.16554v1)**

> **作者:** Vahid Jebraeeli; Hamid Krim; Derya Cansever
>
> **备注:** -
>
> **摘要:** Vision Transformer (ViT) based autoencoders often underutilize the global Class token and employ static attention mechanisms, limiting both generative control and optimization efficiency. This paper introduces ViTCAE, a framework that addresses these issues by re-purposing the Class token into a generative linchpin. In our architecture, the encoder maps the Class token to a global latent variable that dictates the prior distribution for local, patch-level latent variables, establishing a robust dependency where global semantics directly inform the synthesis of local details. Drawing inspiration from opinion dynamics, we treat each attention head as a dynamical system of interacting tokens seeking consensus. This perspective motivates a convergence-aware temperature scheduler that adaptively anneals each head's influence function based on its distributional stability. This process enables a principled head-freezing mechanism, guided by theoretically-grounded diagnostics like an attention evolution distance and a consensus/cluster functional. This technique prunes converged heads during training to significantly improve computational efficiency without sacrificing fidelity. By unifying a generative Class token with an adaptive attention mechanism rooted in multi-agent consensus theory, ViTCAE offers a more efficient and controllable approach to transformer-based generation.
>
---
#### [new 226] VAInpaint: Zero-Shot Video-Audio inpainting framework with LLMs-driven Module
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出VAInpaint，一种基于LLM的视频-音频补全框架，旨在解决从视频中精准移除对象及其对应音频的问题。通过分割模型生成掩码，并结合视频补全与文本驱动的音频分离模型实现零样本音视频修复。**

- **链接: [http://arxiv.org/pdf/2509.17022v1](http://arxiv.org/pdf/2509.17022v1)**

> **作者:** Kam Man Wu; Zeyue Tian; Liya Ji; Qifeng Chen
>
> **摘要:** Video and audio inpainting for mixed audio-visual content has become a crucial task in multimedia editing recently. However, precisely removing an object and its corresponding audio from a video without affecting the rest of the scene remains a significant challenge. To address this, we propose VAInpaint, a novel pipeline that first utilizes a segmentation model to generate masks and guide a video inpainting model in removing objects. At the same time, an LLM then analyzes the scene globally, while a region-specific model provides localized descriptions. Both the overall and regional descriptions will be inputted into an LLM, which will refine the content and turn it into text queries for our text-driven audio separation model. Our audio separation model is fine-tuned on a customized dataset comprising segmented MUSIC instrument images and VGGSound backgrounds to enhance its generalization performance. Experiments show that our method achieves performance comparable to current benchmarks in both audio and video inpainting.
>
---
#### [new 227] MetaEmbed: Scaling Multimodal Retrieval at Test-Time with Flexible Late Interaction
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文提出MetaEmbed框架，用于多模态检索任务。针对现有方法在表达性和效率间的权衡问题，引入可学习的Meta Tokens，在测试时通过选择不同数量的向量实现质量与效率的平衡，并在多个基准上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.18095v1](http://arxiv.org/pdf/2509.18095v1)**

> **作者:** Zilin Xiao; Qi Ma; Mengting Gu; Chun-cheng Jason Chen; Xintao Chen; Vicente Ordonez; Vijai Mohan
>
> **摘要:** Universal multimodal embedding models have achieved great success in capturing semantic relevance between queries and candidates. However, current methods either condense queries and candidates into a single vector, potentially limiting the expressiveness for fine-grained information, or produce too many vectors that are prohibitively expensive for multi-vector retrieval. In this work, we introduce MetaEmbed, a new framework for multimodal retrieval that rethinks how multimodal embeddings are constructed and interacted with at scale. During training, a fixed number of learnable Meta Tokens are appended to the input sequence. At test-time, their last-layer contextualized representations serve as compact yet expressive multi-vector embeddings. Through the proposed Matryoshka Multi-Vector Retrieval training, MetaEmbed learns to organize information by granularity across multiple vectors. As a result, we enable test-time scaling in multimodal retrieval, where users can balance retrieval quality against efficiency demands by selecting the number of tokens used for indexing and retrieval interactions. Extensive evaluations on the Massive Multimodal Embedding Benchmark (MMEB) and the Visual Document Retrieval Benchmark (ViDoRe) confirm that MetaEmbed achieves state-of-the-art retrieval performance while scaling robustly to models with 32B parameters.
>
---
#### [new 228] High Resolution UDF Meshing via Iterative Networks
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出一种迭代神经网络，用于高分辨率 Unsigned Distance Fields (UDF) 的网格化。针对传统方法在噪声和复杂区域中易产生缺失表面和孔洞的问题，通过多轮推理和邻域信息传播，提升表面恢复的准确性和完整性。属于3D重建任务。**

- **链接: [http://arxiv.org/pdf/2509.17212v1](http://arxiv.org/pdf/2509.17212v1)**

> **作者:** Federico Stella; Nicolas Talabot; Hieu Le; Pascal Fua
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Unsigned Distance Fields (UDFs) are a natural implicit representation for open surfaces but, unlike Signed Distance Fields (SDFs), are challenging to triangulate into explicit meshes. This is especially true at high resolutions where neural UDFs exhibit higher noise levels, which makes it hard to capture fine details. Most current techniques perform within single voxels without reference to their neighborhood, resulting in missing surface and holes where the UDF is ambiguous or noisy. We show that this can be remedied by performing several passes and by reasoning on previously extracted surface elements to incorporate neighborhood information. Our key contribution is an iterative neural network that does this and progressively improves surface recovery within each voxel by spatially propagating information from increasingly distant neighbors. Unlike single-pass methods, our approach integrates newly detected surfaces, distance values, and gradients across multiple iterations, effectively correcting errors and stabilizing extraction in challenging regions. Experiments on diverse 3D models demonstrate that our method produces significantly more accurate and complete meshes than existing approaches, particularly for complex geometries, enabling UDF surface extraction at higher resolutions where traditional methods fail.
>
---
#### [new 229] Sight Over Site: Perception-Aware Reinforcement Learning for Efficient Robotic Inspection
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人自主巡检任务，旨在解决传统导航方式忽视视觉感知导致的低效问题。提出一种端到端的强化学习框架，以目标可见性为核心目标，使机器人找到最短可见路径。方法在仿真中训练，实现实验验证。**

- **链接: [http://arxiv.org/pdf/2509.17877v1](http://arxiv.org/pdf/2509.17877v1)**

> **作者:** Richard Kuhlmann; Jakob Wolfram; Boyang Sun; Jiaxu Xing; Davide Scaramuzza; Marc Pollefeys; Cesar Cadena
>
> **摘要:** Autonomous inspection is a central problem in robotics, with applications ranging from industrial monitoring to search-and-rescue. Traditionally, inspection has often been reduced to navigation tasks, where the objective is to reach a predefined location while avoiding obstacles. However, this formulation captures only part of the real inspection problem. In real-world environments, the inspection targets may become visible well before their exact coordinates are reached, making further movement both redundant and inefficient. What matters more for inspection is not simply arriving at the target's position, but positioning the robot at a viewpoint from which the target becomes observable. In this work, we revisit inspection from a perception-aware perspective. We propose an end-to-end reinforcement learning framework that explicitly incorporates target visibility as the primary objective, enabling the robot to find the shortest trajectory that guarantees visual contact with the target without relying on a map. The learned policy leverages both perceptual and proprioceptive sensing and is trained entirely in simulation, before being deployed to a real-world robot. We further develop an algorithm to compute ground-truth shortest inspection paths, which provides a reference for evaluation. Through extensive experiments, we show that our method outperforms existing classical and learning-based navigation approaches, yielding more efficient inspection trajectories in both simulated and real-world settings. The project is avialable at https://sight-over-site.github.io/
>
---
#### [new 230] R-Net: A Reliable and Resource-Efficient CNN for Colorectal Cancer Detection with XAI Integration
- **分类: q-bio.TO; cs.AI; cs.CV**

- **简介: 该论文提出R-Net，一种轻量级CNN，用于结直肠癌检测与分类。针对传统SOTA CNN计算资源消耗大的问题，R-Net在保证高精度（99.37%）的同时集成XAI技术（如SHAP、LIME、Grad-CAM），提升模型可解释性，并对比多种CNN和集成模型效果。**

- **链接: [http://arxiv.org/pdf/2509.16251v1](http://arxiv.org/pdf/2509.16251v1)**

> **作者:** Rokonozzaman Ayon; Md Taimur Ahad; Bo Song; Yan Li
>
> **摘要:** State-of-the-art (SOTA) Convolutional Neural Networks (CNNs) are criticized for their extensive computational power, long training times, and large datasets. To overcome this limitation, we propose a reasonable network (R-Net), a lightweight CNN only to detect and classify colorectal cancer (CRC) using the Enteroscope Biopsy Histopathological Hematoxylin and Eosin Image Dataset (EBHI). Furthermore, six SOTA CNNs, including Multipath-based CNNs (DenseNet121, ResNet50), Depth-based CNNs (InceptionV3), width-based multi-connection CNNs (Xception), depth-wise separable convolutions (MobileNetV2), spatial exploitation-based CNNs (VGG16), Transfer learning, and two ensemble models are also tested on the same dataset. The ensemble models are a multipath-depth-width combination (DenseNet121-InceptionV3-Xception) and a multipath-depth-spatial combination (ResNet18-InceptionV3-VGG16). However, the proposed R-Net lightweight achieved 99.37% accuracy, outperforming MobileNet (95.83%) and ResNet50 (96.94%). Most importantly, to understand the decision-making of R-Net, Explainable AI such as SHAP, LIME, and Grad-CAM are integrated to visualize which parts of the EBHI image contribute to the detection and classification process of R-Net. The main novelty of this research lies in building a reliable, lightweight CNN R-Net that requires fewer computing resources yet maintains strong prediction results. SOTA CNNs, transfer learning, and ensemble models also extend our knowledge on CRC classification and detection. XAI functionality and the impact of pixel intensity on correct and incorrect classification images are also some novelties in CRC detection and classification.
>
---
#### [new 231] Detection of Misreporting Attacks on Software-Defined Immersive Environments
- **分类: cs.NI; cs.CV**

- **简介: 该论文属于网络安全任务，旨在解决SDN沉浸式环境中因交换机错误报告导致的负载失衡问题。提出了一种结合无监督异常评分与监督分类的混合机器学习检测框架，通过识别时序不一致行为，实现对隐蔽攻击的高效检测。**

- **链接: [http://arxiv.org/pdf/2509.18040v1](http://arxiv.org/pdf/2509.18040v1)**

> **作者:** Sourya Saha; Md Nurul Absur; Shima Yousefi; Saptarshi Debroy
>
> **备注:** 7 Pages, 7 Images, will appear in CNSM 2025
>
> **摘要:** The ability to centrally control network infrastructure using a programmable middleware has made Software-Defined Networking (SDN) ideal for emerging applications, such as immersive environments. However, such flexibility introduces new vulnerabilities, such as switch misreporting led load imbalance, which in turn make such immersive environment vulnerable to severe quality degradation. In this paper, we present a hybrid machine learning (ML)-based network anomaly detection framework that identifies such stealthy misreporting by capturing temporal inconsistencies in switch-reported loads, and thereby counter potentially catastrophic quality degradation of hosted immersive application. The detection system combines unsupervised anomaly scoring with supervised classification to robustly distinguish malicious behavior. Data collected from a realistic testbed deployment under both benign and adversarial conditions is used to train and evaluate the model. Experimental results show that the framework achieves high recall in detecting misreporting behavior, making it effective for early and reliable detection in SDN environments.
>
---
#### [new 232] The Iconicity of the Generated Image
- **分类: cs.CY; cs.CV**

- **简介: 该论文研究生成式AI在图像生成中是否受标志性图像影响。通过数据归因、语义相似性分析和用户研究，发现AI难以再现标志性图像，揭示了人类与AI在视觉学习上的差异。属于视觉生成任务，旨在探讨AI对经典图像的学习机制。**

- **链接: [http://arxiv.org/pdf/2509.16473v1](http://arxiv.org/pdf/2509.16473v1)**

> **作者:** Nanne van Noord; Noa Garcia
>
> **备注:** Work presented at EA-AI 2025, May 2025, Venice
>
> **摘要:** How humans interpret and produce images is influenced by the images we have been exposed to. Similarly, visual generative AI models are exposed to many training images and learn to generate new images based on this. Given the importance of iconic images in human visual communication, as they are widely seen, reproduced, and used as inspiration, we may expect that they may similarly have a proportionally large influence within the generative AI process. In this work we explore this question through a three-part analysis, involving data attribution, semantic similarity analysis, and a user-study. Our findings indicate that iconic images do not have an obvious influence on the generative process, and that for many icons it is challenging to reproduce an image which resembles it closely. This highlights an important difference in how humans and visual generative AI models draw on and learn from prior visual communication.
>
---
#### [new 233] Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出一种名为Contract-and-Broadcast Self-Attention (CBSA) 的机制，旨在同时提升Transformer中注意力机制的可解释性和计算效率。通过统一优化目标，将所有token压缩为少量代表性token，实现线性扩展并保持性能。**

- **链接: [http://arxiv.org/pdf/2509.16875v1](http://arxiv.org/pdf/2509.16875v1)**

> **作者:** Qishuai Wen; Zhiyuan Huang; Chun-Guang Li
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Attention mechanisms in Transformers have gained significant empirical success. Nonetheless, the optimization objectives underlying their forward pass are still unclear. Additionally, the quadratic complexity of self-attention is increasingly prohibitive. Unlike the prior work on addressing the interpretability or efficiency issue separately, we propose a unified optimization objective to alleviate both issues simultaneously. By unrolling the optimization over the objective, we derive an inherently interpretable and efficient attention mechanism, which compresses all tokens into low-dimensional structures by contracting a few representative tokens and then broadcasting the contractions back. This Contract-and-Broadcast Self-Attention (CBSA) mechanism can not only scale linearly but also generalize existing attention mechanisms as its special cases. Experiments further demonstrate comparable performance and even superior advantages of CBSA on several visual tasks. Code is available at this https URL.
>
---
#### [new 234] Automated Coral Spawn Monitoring for Reef Restoration: The Coral Spawn and Larvae Imaging Camera System (CSLICS)
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CSLICS系统，用于自动化监测珊瑚产卵和幼虫，解决人工计数劳动强度大、效率低的问题。通过低成本摄像头和目标检测技术实现自动计数，提升了珊瑚养殖效率与生态修复能力。**

- **链接: [http://arxiv.org/pdf/2509.17299v1](http://arxiv.org/pdf/2509.17299v1)**

> **作者:** Dorian Tsai; Christopher A. Brunner; Riki Lamont; F. Mikaela Nordborg; Andrea Severati; Java Terry; Karen Jackel; Matthew Dunbabin; Tobias Fischer; Scarlett Raine
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Coral aquaculture for reef restoration requires accurate and continuous spawn counting for resource distribution and larval health monitoring, but current methods are labor-intensive and represent a critical bottleneck in the coral production pipeline. We propose the Coral Spawn and Larvae Imaging Camera System (CSLICS), which uses low cost modular cameras and object detectors trained using human-in-the-loop labeling approaches for automated spawn counting in larval rearing tanks. This paper details the system engineering, dataset collection, and computer vision techniques to detect, classify and count coral spawn. Experimental results from mass spawning events demonstrate an F1 score of 82.4\% for surface spawn detection at different embryogenesis stages, 65.3\% F1 score for sub-surface spawn detection, and a saving of 5,720 hours of labor per spawning event compared to manual sampling methods at the same frequency. Comparison of manual counts with CSLICS monitoring during a mass coral spawning event on the Great Barrier Reef demonstrates CSLICS' accurate measurement of fertilization success and sub-surface spawn counts. These findings enhance the coral aquaculture process and enable upscaling of coral reef restoration efforts to address climate change threats facing ecosystems like the Great Barrier Reef.
>
---
#### [new 235] Fusing Spectral Correlation Density Imaging with Deep Learning for Intelligent Fault Diagnosis in Rotating Machinery
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于故障诊断任务，旨在解决旋转机械轴承故障的早期检测问题。通过将振动信号转化为SCD图像，并结合三种CNN模型进行分类，实现了高精度的多工况故障识别，验证了方法在不同环境下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16580v1](http://arxiv.org/pdf/2509.16580v1)**

> **作者:** Dilshara Herath; Chinthaka Abeyrathne; Chamindu Adithya; Chathura Seneviratne
>
> **摘要:** Bearing fault diagnosis in rotating machinery is critical for ensuring operational reliability, therefore early fault detection is essential to avoid catastrophic failures and expensive emergency repairs. Traditional methods like Fast Fourier Transform (FFT) often fail to capture the complex, non-stationary nature of vibration signals. This study leverages the cyclostationary properties of vibration data through Spectral Correlation Density (SCD) images to enhance fault detection and apply deep learning for classification. Using a publicly available dataset with bearing faults seeded in two distinct housings (A and B) under varying load conditions (0 Nm, 2 Nm, 4 Nm), we processed vibration signals into 2D SCD images to reveal fault-specific periodicities, such as broadband spectra (2000--8000 Hz) for larger faults. Three convolutional neural network (CNN) models, Custom CNN, ResNet152V2, and EfficientNetB0, were developed to classify seven bearing conditions. The custom CNN achieved the highest accuracies of 96.58\% and 94.95\% on Housing A and B, respectively, followed by ResNet152V2 at 96.49\% and 95.35\%, and EfficientNetB0 at 94.16\% and 91.65\%, respectively. The models' high accuracies across different housings demonstrate a robust solution suitable for cost-effective condition monitoring deployable near sensing platforms, contributing to applied machine learning for edge intelligence and showcasing effective signal processing strategies for handling complex, potentially large-scale vibration data.
>
---
#### [new 236] TASO: Task-Aligned Sparse Optimization for Parameter-Efficient Model Adaptation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出TASO，一种基于预训练模型权重重要性信息的任务对齐稀疏优化方法，旨在减少LoRA微调中的冗余参数，提升参数效率和微调效果。**

- **链接: [http://arxiv.org/pdf/2509.17688v1](http://arxiv.org/pdf/2509.17688v1)**

> **作者:** Daiye Miao; Yufang Liu; Jie Wang; Changzhi Sun; Yunke Zhang; Demei Yan; Shaokang Dong; Qi Zhang; Yuanbin Wu
>
> **备注:** Accepted to EMNLP 2025 (Main Conference),13 pages,10 figures
>
> **摘要:** LoRA has become one of the most widely used parameter-efficient fine-tuning methods due to its simplicity and effectiveness. However, numerous studies have shown that LoRA often introduces substantial parameter redundancy, which not only increases the number of trainable parameters but also hinders the effectiveness of fine-tuning. Since identifying redundant parameters in LoRA is inherently difficult, how to eliminate them efficiently and accurately remains a challenging problem. In this paper, we propose TASO, a redundancy reduction method that leverages importance information from the pretrained model's weights to mitigate LoRA redundancy. Specifically, we estimate parameter importance on downstream tasks and identify task-specific core regions based on the distribution of importance scores. The location information of these core regions is then used to determine the sparse structure of LoRA modules, enabling redundancy removal before fine-tuning. Our approach significantly reduces the number of trainable parameters required for task adaptation, while providing a novel task-aligned perspective for LoRA redundancy reduction. Experimental results demonstrate that, with a parameter budget comparable to LoRA with rank $r = 1$, TASO consistently outperforms standard LoRA across multiple tasks, achieving strong fine-tuning performance while effectively eliminating redundant parameters.
>
---
#### [new 237] Vision Language Models Are Not (Yet) Spelling Correctors
- **分类: cs.CL; cs.CV**

- **简介: 该论文聚焦视觉语言模型在图像文本拼写纠错任务中的表现，提出首个真实场景下的中英文视觉拼写纠错基准ReViCo，并通过实验分析当前VLM的不足，探索两种改进方法以提升纠错性能。**

- **链接: [http://arxiv.org/pdf/2509.17418v1](http://arxiv.org/pdf/2509.17418v1)**

> **作者:** Junhong Liang; Bojun Zhang
>
> **摘要:** Spelling correction from visual input poses unique challenges for vision language models (VLMs), as it requires not only detecting but also correcting textual errors directly within images. We present ReViCo (Real Visual Correction), the first benchmark that systematically evaluates VLMs on real-world visual spelling correction across Chinese and English. ReViCo contains naturally occurring errors collected from real-world image data and supports fine-grained evaluation at both image and token levels. Through comprehensive experiments on representative cascaded (Qwen) and native (InternVL) open-source models, as well as closed-source systems (GPT-4o, Claude), we show that current VLMs fall significantly short of human performance, particularly in correction. To address these limitations, we explore two solution paradigms: a Joint OCR-Correction pipeline and a Background Information enhanced approach, both of which yield consistent performance gains. Our analysis highlights fundamental limitations of existing architectures and provides actionable insights for advancing multimodal spelling correction.
>
---
#### [new 238] A study on Deep Convolutional Neural Networks, transfer learning, and Mnet model for Cervical Cancer Detection
- **分类: q-bio.TO; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决宫颈癌早期检测中模型计算成本高、缺乏可解释性的问题。研究开发了一个轻量级CNN模型S-Net，并结合XAI技术提升模型可解释性，同时对比分析了多种SOTA模型在迁移学习下的表现。**

- **链接: [http://arxiv.org/pdf/2509.16250v1](http://arxiv.org/pdf/2509.16250v1)**

> **作者:** Saifuddin Sagor; Md Taimur Ahad; Faruk Ahmed; Rokonozzaman Ayon; Sanzida Parvin
>
> **摘要:** Early and accurate detection through Pap smear analysis is critical to improving patient outcomes and reducing mortality of Cervical cancer. State-of-the-art (SOTA) Convolutional Neural Networks (CNNs) require substantial computational resources, extended training time, and large datasets. In this study, a lightweight CNN model, S-Net (Simple Net), is developed specifically for cervical cancer detection and classification using Pap smear images to address these limitations. Alongside S-Net, six SOTA CNNs were evaluated using transfer learning, including multi-path (DenseNet201, ResNet152), depth-based (Serasnet152), width-based multi-connection (Xception), depth-wise separable convolutions (MobileNetV2), and spatial exploitation-based (VGG19). All models, including S-Net, achieved comparable accuracy, with S-Net reaching 99.99%. However, S-Net significantly outperforms the SOTA CNNs in terms of computational efficiency and inference time, making it a more practical choice for real-time and resource-constrained applications. A major limitation in CNN-based medical diagnosis remains the lack of transparency in the decision-making process. To address this, Explainable AI (XAI) techniques, such as SHAP, LIME, and Grad-CAM, were employed to visualize and interpret the key image regions influencing model predictions. The novelty of this study lies in the development of a highly accurate yet computationally lightweight model (S-Net) caPable of rapid inference while maintaining interpretability through XAI integration. Furthermore, this work analyzes the behavior of SOTA CNNs, investigates the effects of negative transfer learning on Pap smear images, and examines pixel intensity patterns in correctly and incorrectly classified samples.
>
---
#### [new 239] Joint Optimization of Memory Frequency, Computing Frequency, Transmission Power and Task Offloading for Energy-efficient DNN Inference
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究如何通过联合优化内存频率和计算频率，降低设备在深度神经网络推理中的能耗。属于能效优化任务，解决高延迟和高能耗问题，提出了基于模型与数据驱动的分析方法，并通过仿真验证了方案的有效性。**

- **链接: [http://arxiv.org/pdf/2509.17970v1](http://arxiv.org/pdf/2509.17970v1)**

> **作者:** Yunchu Han; Zhaojun Nan; Sheng Zhou; Zhisheng Niu
>
> **摘要:** Deep neural networks (DNNs) have been widely applied in diverse applications, but the problems of high latency and energy overhead are inevitable on resource-constrained devices. To address this challenge, most researchers focus on the dynamic voltage and frequency scaling (DVFS) technique to balance the latency and energy consumption by changing the computing frequency of processors. However, the adjustment of memory frequency is usually ignored and not fully utilized to achieve efficient DNN inference, which also plays a significant role in the inference time and energy consumption. In this paper, we first investigate the impact of joint memory frequency and computing frequency scaling on the inference time and energy consumption with a model-based and data-driven method. Then by combining with the fitting parameters of different DNN models, we give a preliminary analysis for the proposed model to see the effects of adjusting memory frequency and computing frequency simultaneously. Finally, simulation results in local inference and cooperative inference cases further validate the effectiveness of jointly scaling the memory frequency and computing frequency to reduce the energy consumption of devices.
>
---
#### [new 240] Computational Scaffolding of Composition, Value, and Color for Disciplined Drawing
- **分类: cs.HC; cs.CV**

- **简介: 该论文提出ArtKrit，一个辅助数字艺术家通过构图、明暗和色彩三步骤进行规范绘画训练的工具，旨在帮助初学者和中级用户提升技术技能并获得即时反馈。**

- **链接: [http://arxiv.org/pdf/2509.17268v1](http://arxiv.org/pdf/2509.17268v1)**

> **作者:** Jiaju Ma; Chau Vu; Asya Lyubavina; Catherine Liu; Jingyi Li
>
> **备注:** Accepted to UIST 2025 (Best Paper)
>
> **摘要:** One way illustrators engage in disciplined drawing - the process of drawing to improve technical skills - is through studying and replicating reference images. However, for many novice and intermediate digital artists, knowing how to approach studying a reference image can be challenging. It can also be difficult to receive immediate feedback on their works-in-progress. To help these users develop their professional vision, we propose ArtKrit, a tool that scaffolds the process of replicating a reference image into three main steps: composition, value, and color. At each step, our tool offers computational guidance, such as adaptive composition line generation, and automatic feedback, such as value and color accuracy. Evaluating this tool with intermediate digital artists revealed that ArtKrit could flexibly accommodate their unique workflows. Our code and supplemental materials are available at https://majiaju.io/artkrit .
>
---
#### [new 241] ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出ComposableNav，用于解决机器人在动态环境中遵循复杂指令导航的问题。针对指令组合爆炸的挑战，通过可组合扩散模型学习并组合不同运动基元，在未见过的指令组合下生成有效轨迹，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.17941v1](http://arxiv.org/pdf/2509.17941v1)**

> **作者:** Zichao Hu; Chen Tang; Michael J. Munje; Yifeng Zhu; Alex Liu; Shuijing Liu; Garrett Warnell; Peter Stone; Joydeep Biswas
>
> **备注:** Conference on Robot Learning (CoRL) 2025 Project site: https://amrl.cs.utexas.edu/ComposableNav/
>
> **摘要:** This paper considers the problem of enabling robots to navigate dynamic environments while following instructions. The challenge lies in the combinatorial nature of instruction specifications: each instruction can include multiple specifications, and the number of possible specification combinations grows exponentially as the robot's skill set expands. For example, "overtake the pedestrian while staying on the right side of the road" consists of two specifications: "overtake the pedestrian" and "walk on the right side of the road." To tackle this challenge, we propose ComposableNav, based on the intuition that following an instruction involves independently satisfying its constituent specifications, each corresponding to a distinct motion primitive. Using diffusion models, ComposableNav learns each primitive separately, then composes them in parallel at deployment time to satisfy novel combinations of specifications unseen in training. Additionally, to avoid the onerous need for demonstrations of individual motion primitives, we propose a two-stage training procedure: (1) supervised pre-training to learn a base diffusion model for dynamic navigation, and (2) reinforcement learning fine-tuning that molds the base model into different motion primitives. Through simulation and real-world experiments, we show that ComposableNav enables robots to follow instructions by generating trajectories that satisfy diverse and unseen combinations of specifications, significantly outperforming both non-compositional VLM-based policies and costmap composing baselines. Videos and additional materials can be found on the project page: https://amrl.cs.utexas.edu/ComposableNav/
>
---
#### [new 242] Long-Tailed Out-of-Distribution Detection with Refined Separate Class Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对长尾分布下异常检测（OOD）任务，提出RSCL方法。通过动态温度调整和信息异常挖掘，解决现有SCL方法中静态温度和无意义异常影响的问题，提升OOD检测性能与分类准确率。**

- **链接: [http://arxiv.org/pdf/2509.17034v1](http://arxiv.org/pdf/2509.17034v1)**

> **作者:** Shuai Feng; Yuxin Ge; Yuntao Du; Mingcai Chen; Lei Feng
>
> **摘要:** Out-of-distribution (OOD) detection is crucial for deploying robust machine learning models. However, when training data follows a long-tailed distribution, the model's ability to accurately detect OOD samples is significantly compromised, due to the confusion between OOD samples and head/tail classes. To distinguish OOD samples from both head and tail classes, the separate class learning (SCL) approach has emerged as a promising solution, which separately conduct head-specific and tail-specific class learning. To this end, we examine the limitations of existing works of SCL and reveal that the OOD detection performance is notably influenced by the use of static scaling temperature value and the presence of uninformative outliers. To mitigate these limitations, we propose a novel approach termed Refined Separate Class Learning (RSCL), which leverages dynamic class-wise temperature adjustment to modulate the temperature parameter for each in-distribution class and informative outlier mining to identify diverse types of outliers based on their affinity with head and tail classes. Extensive experiments demonstrate that RSCL achieves superior OOD detection performance while improving the classification accuracy on in-distribution data.
>
---
#### [new 243] Mano Report
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文提出Mano，一个基于多模态模型的GUI智能体，旨在解决GUI自动化中视觉复杂性、环境动态性和多步决策难题。通过构建高保真模拟环境和三阶段训练框架（SFT+离线/在线强化学习），结合验证模块提升鲁棒性，在多个基准测试中取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.17336v1](http://arxiv.org/pdf/2509.17336v1)**

> **作者:** Tianyu Fu; Anyang Su; Chenxu Zhao; Hanning Wang; Minghui Wu; Zhe Yu; Fei Hu; Mingjia Shi; Wei Dong; Jiayao Wang; Yuyang Chen; Ruiyang Yu; Siran Peng; Menglin Li; Nan Huang; Haitian Wei; Jiawei Yu; Yi Xin; Xilin Zhao; Kai Gu; Ping Jiang; Sifan Zhou; Shuo Wang
>
> **摘要:** Graphical user interfaces (GUIs) are the primary medium for human-computer interaction, yet automating GUI interactions remains challenging due to the complexity of visual elements, dynamic environments, and the need for multi-step reasoning. Existing methods based on vision-language models (VLMs) often suffer from limited resolution, domain mismatch, and insufficient sequential decisionmaking capability. To address these issues, we propose Mano, a robust GUI agent built upon a multi-modal foundation model pre-trained on extensive web and computer system data. Our approach integrates a novel simulated environment for high-fidelity data generation, a three-stage training pipeline (supervised fine-tuning, offline reinforcement learning, and online reinforcement learning), and a verification module for error recovery. Mano demonstrates state-of-the-art performance on multiple GUI benchmarks, including Mind2Web and OSWorld, achieving significant improvements in success rate and operational accuracy. Our work provides new insights into the effective integration of reinforcement learning with VLMs for practical GUI agent deployment, highlighting the importance of domain-specific data, iterative training, and holistic reward design.
>
---
#### [new 244] PhysHDR: When Lighting Meets Materials and Scene Geometry in HDR Reconstruction
- **分类: cs.GR; cs.AI; cs.CV; cs.LG; cs.MM; eess.IV; Artificial intelligence, Computer vision, Machine learning, Deep
  learning; I.3.3; I.4.5**

- **简介: 该论文聚焦于低动态范围（LDR）到高动态范围（HDR）图像重建任务，旨在解决现有方法缺乏对光照、材质和场景几何的显式建模问题。提出了PhysHDR，一种基于扩散模型的方法，通过引入光照、深度信息及材质属性损失，提升了HDR重建质量。**

- **链接: [http://arxiv.org/pdf/2509.16869v1](http://arxiv.org/pdf/2509.16869v1)**

> **作者:** Hrishav Bakul Barua; Kalin Stefanov; Ganesh Krishnasamy; KokSheik Wong; Abhinav Dhall
>
> **备注:** Submitted to IEEE
>
> **摘要:** Low Dynamic Range (LDR) to High Dynamic Range (HDR) image translation is a fundamental task in many computational vision problems. Numerous data-driven methods have been proposed to address this problem; however, they lack explicit modeling of illumination, lighting, and scene geometry in images. This limits the quality of the reconstructed HDR images. Since lighting and shadows interact differently with different materials, (e.g., specular surfaces such as glass and metal, and lambertian or diffuse surfaces such as wood and stone), modeling material-specific properties (e.g., specular and diffuse reflectance) has the potential to improve the quality of HDR image reconstruction. This paper presents PhysHDR, a simple yet powerful latent diffusion-based generative model for HDR image reconstruction. The denoising process is conditioned on lighting and depth information and guided by a novel loss to incorporate material properties of surfaces in the scene. The experimental results establish the efficacy of PhysHDR in comparison to a number of recent state-of-the-art methods.
>
---
#### [new 245] Intra-Cluster Mixup: An Effective Data Augmentation Technique for Complementary-Label Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究互补标签学习（CLL）任务，旨在解决传统Mixup数据增强在该任务中引入噪声的问题。提出Intra-Cluster Mixup（ICM），通过仅混合相近样本提升模型性能，在多个数据集上取得显著效果。**

- **链接: [http://arxiv.org/pdf/2509.17971v1](http://arxiv.org/pdf/2509.17971v1)**

> **作者:** Tan-Ha Mai; Hsuan-Tien Lin
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** In this paper, we investigate the challenges of complementary-label learning (CLL), a specialized form of weakly-supervised learning (WSL) where models are trained with labels indicating classes to which instances do not belong, rather than standard ordinary labels. This alternative supervision is appealing because collecting complementary labels is generally cheaper and less labor-intensive. Although most existing research in CLL emphasizes the development of novel loss functions, the potential of data augmentation in this domain remains largely underexplored. In this work, we uncover that the widely-used Mixup data augmentation technique is ineffective when directly applied to CLL. Through in-depth analysis, we identify that the complementary-label noise generated by Mixup negatively impacts the performance of CLL models. We then propose an improved technique called Intra-Cluster Mixup (ICM), which only synthesizes augmented data from nearby examples, to mitigate the noise effect. ICM carries the benefits of encouraging complementary label sharing of nearby examples, and leads to substantial performance improvements across synthetic and real-world labeled datasets. In particular, our wide spectrum of experimental results on both balanced and imbalanced CLL settings justifies the potential of ICM in allying with state-of-the-art CLL algorithms, achieving significant accuracy increases of 30% and 10% on MNIST and CIFAR datasets, respectively.
>
---
#### [new 246] CoUn: Empowering Machine Unlearning via Contrastive Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出CoUn，一种基于对比学习的机器遗忘（MU）方法。旨在有效移除特定数据的影响，同时保留其余数据的知识。通过调整数据表示，提升遗忘效果，并在多个实验中验证其优越性。**

- **链接: [http://arxiv.org/pdf/2509.16391v1](http://arxiv.org/pdf/2509.16391v1)**

> **作者:** Yasser H. Khalil; Mehdi Setayesh; Hongliang Li
>
> **摘要:** Machine unlearning (MU) aims to remove the influence of specific "forget" data from a trained model while preserving its knowledge of the remaining "retain" data. Existing MU methods based on label manipulation or model weight perturbations often achieve limited unlearning effectiveness. To address this, we introduce CoUn, a novel MU framework inspired by the observation that a model retrained from scratch using only retain data classifies forget data based on their semantic similarity to the retain data. CoUn emulates this behavior by adjusting learned data representations through contrastive learning (CL) and supervised learning, applied exclusively to retain data. Specifically, CoUn (1) leverages semantic similarity between data samples to indirectly adjust forget representations using CL, and (2) maintains retain representations within their respective clusters through supervised learning. Extensive experiments across various datasets and model architectures show that CoUn consistently outperforms state-of-the-art MU baselines in unlearning effectiveness. Additionally, integrating our CL module into existing baselines empowers their unlearning effectiveness.
>
---
#### [new 247] Neural Atlas Graphs for Dynamic Scene Decomposition and Editing
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文提出Neural Atlas Graphs（NAGs），用于动态场景分解与编辑。针对现有方法在编辑性与复杂度间的权衡问题，NAGs结合神经图与图节点表示，实现高分辨率2D编辑与3D空间关系建模，提升PSNR性能，并支持多样场景的环境编辑任务。**

- **链接: [http://arxiv.org/pdf/2509.16336v1](http://arxiv.org/pdf/2509.16336v1)**

> **作者:** Jan Philipp Schneider; Pratik Singh Bisht; Ilya Chugunov; Andreas Kolb; Michael Moeller; Felix Heide
>
> **摘要:** Learning editable high-resolution scene representations for dynamic scenes is an open problem with applications across the domains from autonomous driving to creative editing - the most successful approaches today make a trade-off between editability and supporting scene complexity: neural atlases represent dynamic scenes as two deforming image layers, foreground and background, which are editable in 2D, but break down when multiple objects occlude and interact. In contrast, scene graph models make use of annotated data such as masks and bounding boxes from autonomous-driving datasets to capture complex 3D spatial relationships, but their implicit volumetric node representations are challenging to edit view-consistently. We propose Neural Atlas Graphs (NAGs), a hybrid high-resolution scene representation, where every graph node is a view-dependent neural atlas, facilitating both 2D appearance editing and 3D ordering and positioning of scene elements. Fit at test-time, NAGs achieve state-of-the-art quantitative results on the Waymo Open Dataset - by 5 dB PSNR increase compared to existing methods - and make environmental editing possible in high resolution and visual quality - creating counterfactual driving scenarios with new backgrounds and edited vehicle appearance. We find that the method also generalizes beyond driving scenes and compares favorably - by more than 7 dB in PSNR - to recent matting and video editing baselines on the DAVIS video dataset with a diverse set of human and animal-centric scenes.
>
---
#### [new 248] A Chain-of-thought Reasoning Breast Ultrasound Dataset Covering All Histopathology Categories
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出了BUS-CoT，一个包含11,439张乳腺超声图像的数据集，覆盖99种组织病理类型。旨在解决AI在乳腺病变诊断中数据规模和标注不足的问题，并促进链式推理研究，提升罕见病例的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.17046v1](http://arxiv.org/pdf/2509.17046v1)**

> **作者:** Haojun Yu; Youcheng Li; Zihan Niu; Nan Zhang; Xuantong Gong; Huan Li; Zhiying Zou; Haifeng Qi; Zhenxiao Cao; Zijie Lan; Xingjian Yuan; Jiating He; Haokai Zhang; Shengtao Zhang; Zicheng Wang; Dong Wang; Ziwei Zhao; Congying Chen; Yong Wang; Wangyan Qin; Qingli Zhu
>
> **摘要:** Breast ultrasound (BUS) is an essential tool for diagnosing breast lesions, with millions of examinations per year. However, publicly available high-quality BUS benchmarks for AI development are limited in data scale and annotation richness. In this work, we present BUS-CoT, a BUS dataset for chain-of-thought (CoT) reasoning analysis, which contains 11,439 images of 10,019 lesions from 4,838 patients and covers all 99 histopathology types. To facilitate research on incentivizing CoT reasoning, we construct the reasoning processes based on observation, feature, diagnosis and pathology labels, annotated and verified by experienced experts. Moreover, by covering lesions of all histopathology types, we aim to facilitate robust AI systems in rare cases, which can be error-prone in clinical practice.
>
---
#### [new 249] Learning Neural Antiderivatives
- **分类: cs.LG; cs.CV; cs.GR**

- **简介: 该论文研究了在连续神经场中学习函数的多重反导数表示问题，提出并分析多种神经积分方法，克服传统离散网格限制。任务涉及微分与积分算子的学习，应用于重构与渲染等下游任务。**

- **链接: [http://arxiv.org/pdf/2509.17755v1](http://arxiv.org/pdf/2509.17755v1)**

> **作者:** Fizza Rubab; Ntumba Elie Nsampi; Martin Balint; Felix Mujkanovic; Hans-Peter Seidel; Tobias Ritschel; Thomas Leimkühler
>
> **摘要:** Neural fields offer continuous, learnable representations that extend beyond traditional discrete formats in visual computing. We study the problem of learning neural representations of repeated antiderivatives directly from a function, a continuous analogue of summed-area tables. Although widely used in discrete domains, such cumulative schemes rely on grids, which prevents their applicability in continuous neural contexts. We introduce and analyze a range of neural methods for repeated integration, including both adaptations of prior work and novel designs. Our evaluation spans multiple input dimensionalities and integration orders, assessing both reconstruction quality and performance in downstream tasks such as filtering and rendering. These results enable integrating classical cumulative operators into modern neural systems and offer insights into learning tasks involving differential and integral operators.
>
---
#### [new 250] HARE: an entity and relation centric evaluation framework for histopathology reports
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出HARE，一个面向组织病理学报告的评估框架，旨在解决生成报告临床质量评估缺乏领域特定指标的问题。工作包括构建标注数据集、开发实体与关系模型（HARE-NER/RE），并提出新评估指标，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.16326v1](http://arxiv.org/pdf/2509.16326v1)**

> **作者:** Yunsoo Kim; Michal W. S. Ong; Alex Shavick; Honghan Wu; Adam P. Levine
>
> **备注:** Accepted to EMNLP2025 Findings
>
> **摘要:** Medical domain automated text generation is an active area of research and development; however, evaluating the clinical quality of generated reports remains a challenge, especially in instances where domain-specific metrics are lacking, e.g. histopathology. We propose HARE (Histopathology Automated Report Evaluation), a novel entity and relation centric framework, composed of a benchmark dataset, a named entity recognition (NER) model, a relation extraction (RE) model, and a novel metric, which prioritizes clinically relevant content by aligning critical histopathology entities and relations between reference and generated reports. To develop the HARE benchmark, we annotated 813 de-identified clinical diagnostic histopathology reports and 652 histopathology reports from The Cancer Genome Atlas (TCGA) with domain-specific entities and relations. We fine-tuned GatorTronS, a domain-adapted language model to develop HARE-NER and HARE-RE which achieved the highest overall F1-score (0.915) among the tested models. The proposed HARE metric outperformed traditional metrics including ROUGE and Meteor, as well as radiology metrics such as RadGraph-XL, with the highest correlation and the best regression to expert evaluations (higher than the second best method, GREEN, a large language model based radiology report evaluator, by Pearson $r = 0.168$, Spearman $\rho = 0.161$, Kendall $\tau = 0.123$, $R^2 = 0.176$, $RMSE = 0.018$). We release HARE, datasets, and the models at https://github.com/knowlab/HARE to foster advancements in histopathology report generation, providing a robust framework for improving the quality of reports.
>
---
## 更新

#### [replaced 001] Investigating Long-term Training for Remote Sensing Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.15143v3](http://arxiv.org/pdf/2407.15143v3)**

> **作者:** JongHyun Park; Yechan Kim; Moongu Jeon
>
> **摘要:** Recently, numerous methods have achieved impressive performance in remote sensing object detection, relying on convolution or transformer architectures. Such detectors typically have a feature backbone to extract useful features from raw input images. A common practice in current detectors is initializing the backbone with pre-trained weights available online. Fine-tuning the backbone is typically required to generate features suitable for remote-sensing images. While the prolonged training could lead to over-fitting, hindering the extraction of basic visual features, it can enable models to gradually extract deeper insights and richer representations from remote sensing data. Striking a balance between these competing factors is critical for achieving optimal performance. In this study, we aim to investigate the performance and characteristics of remote sensing object detection models under very long training schedules, and propose a novel method named Dynamic Backbone Freezing (DBF) for feature backbone fine-tuning on remote sensing object detection under long-term training. Our method addresses the dilemma of whether the backbone should extract low-level generic features or possess specific knowledge of the remote sensing domain, by introducing a module called 'Freezing Scheduler' to manage the update of backbone features during long-term training dynamically. Extensive experiments on DOTA and DIOR-R show that our approach enables more accurate model learning while substantially reducing computational costs in long-term training. Besides, it can be seamlessly adopted without additional effort due to its straightforward design. The code is available at https://github.com/unique-chan/dbf.
>
---
#### [replaced 002] Multi-viewregulated gaussian splatting for novel view synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.02103v2](http://arxiv.org/pdf/2410.02103v2)**

> **作者:** Xiaobiao Du; Yida Wang; Xin Yu
>
> **备注:** Project Page:https://xiaobiaodu.github.io/mvgs-project/
>
> **摘要:** Recent works in volume rendering, \textit{e.g.} NeRF and 3D Gaussian Splatting (3DGS), significantly advance the rendering quality and efficiency with the help of the learned implicit neural radiance field or 3D Gaussians. Rendering on top of an explicit representation, the vanilla 3DGS and its variants deliver real-time efficiency by optimizing the parametric model with single-view supervision per iteration during training which is adopted from NeRF. Consequently, certain views are overfitted, leading to unsatisfying appearance in novel-view synthesis and imprecise 3D geometries. To solve aforementioned problems, we propose a new 3DGS optimization method embodying four key novel contributions: 1) We transform the conventional single-view training paradigm into a multi-view training strategy. With our proposed multi-view regulation, 3D Gaussian attributes are further optimized without overfitting certain training views. As a general solution, we improve the overall accuracy in a variety of scenarios and different Gaussian variants. 2) Inspired by the benefit introduced by additional views, we further propose a cross-intrinsic guidance scheme, leading to a coarse-to-fine training procedure concerning different resolutions. 3) Built on top of our multi-view regulated training, we further propose a cross-ray densification strategy, densifying more Gaussian kernels in the ray-intersect regions from a selection of views. 4) By further investigating the densification strategy, we found that the effect of densification should be enhanced when certain views are distinct dramatically. As a solution, we propose a novel multi-view augmented densification strategy, where 3D Gaussians are encouraged to get densified to a sufficient number accordingly, resulting in improved reconstruction accuracy.
>
---
#### [replaced 003] Few-Shot Image Quality Assessment via Adaptation of Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.05381v2](http://arxiv.org/pdf/2409.05381v2)**

> **作者:** Xudong Li; Zihao Huang; Yan Zhang; Yunhang Shen; Ke Li; Xiawu Zheng; Liujuan Cao; Rongrong Ji
>
> **摘要:** Image Quality Assessment (IQA) remains an unresolved challenge in computer vision due to complex distortions, diverse image content, and limited data availability. Existing Blind IQA (BIQA) methods largely rely on extensive human annotations, which are labor-intensive and costly due to the demanding nature of creating IQA datasets. To reduce this dependency, we propose the Gradient-Regulated Meta-Prompt IQA Framework (GRMP-IQA), designed to efficiently adapt the visual-language pre-trained model, CLIP, to IQA tasks, achieving high accuracy even with limited data. GRMP-IQA consists of two core modules: (i) Meta-Prompt Pre-training Module and (ii) Quality-Aware Gradient Regularization. The Meta Prompt Pre-training Module leverages a meta-learning paradigm to pre-train soft prompts with shared meta-knowledge across different distortions, enabling rapid adaptation to various IQA tasks. On the other hand, the Quality-Aware Gradient Regularization is designed to adjust the update gradients during fine-tuning, focusing the model's attention on quality-relevant features and preventing overfitting to semantic information. Extensive experiments on standard BIQA datasets demonstrate the superior performance to the state-of-the-art BIQA methods under limited data setting. Notably, utilizing just 20% of the training data, GRMP-IQA is competitive with most existing fully supervised BIQA approaches.
>
---
#### [replaced 004] InfiniBench: A Benchmark for Large Multi-Modal Models in Long-Form Movies and TV Shows
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.19875v4](http://arxiv.org/pdf/2406.19875v4)**

> **作者:** Kirolos Ataallah; Eslam Abdelrahman; Mahmoud Ahmed; Chenhui Gou; Khushbu Pahwa; Jian Ding; Mohamed Elhoseiny
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Understanding long-form videos, such as movies and TV episodes ranging from tens of minutes to two hours, remains a significant challenge for multi-modal models. Existing benchmarks often fail to test the full range of cognitive skills needed to process these temporally rich and narratively complex inputs. Therefore, we introduce InfiniBench, a comprehensive benchmark designed to evaluate the capabilities of models in long video understanding rigorously. InfiniBench offers:(1) Over 1,000 hours of video content, with an average video length of 53 minutes. (2) The largest set of question-answer pairs for long video comprehension, totaling around 87.7 K. (3) Eight diverse skills that span both grounding-based (e.g., scene transitions, character actions) and reasoning-based (e.g., deep context understanding, multi-event linking). (4) Rich annotation formats, including both multiple-choice and open-ended questions. We conducted an in-depth evaluation across both commercial (GPT-4o, Gemini 2.0 Flash) and most recent open-source vision-language models such as Qwen2.5-VL, InternVL3.0). Results reveal that:(1) Models struggle across the board: Even the best model, GPT-4o, achieves only 47.1 % on grounding-based skills, with most models performing near or just above random chance. (2) Strong reliance on world knowledge: Models achieve surprisingly high scores using only metadata (e.g., video titles), highlighting a tendency to rely on pre-trained knowledge rather than actual visual or temporal understanding. (3) Multi-Modal Importance: When provided with full video and subtitle context, however, models show substantial improvements, confirming the critical role of multimodal input in video understanding. InfiniBench is publicly available at https://vision-cair.github.io/Infinibench
>
---
#### [replaced 005] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; 68T07, 68T45, 68T50, 68T09, 68U10, 94A29, 94A34, 94A08, 94A17; I.2.10; I.2.7; I.5.4; I.4.9; I.4; H.5.1; H.3.3**

- **链接: [http://arxiv.org/pdf/2503.16980v5](http://arxiv.org/pdf/2503.16980v5)**

> **作者:** Haichao Zhang; Yun Fu
>
> **备注:** Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Token-based video representation has emerged as a promising approach for enabling large language models (LLMs) to interpret video content. However, existing token reduction techniques, such as pruning and merging, often disrupt essential positional embeddings and rely on continuous visual tokens sampled from nearby pixels with similar spatial-temporal locations. By removing only a small fraction of tokens, these methods still produce relatively lengthy continuous sequences, which falls short of the extreme compression required to balance computational efficiency and token count in video LLMs. In this paper, we introduce the novel task of Extreme Short Token Reduction, which aims to represent entire videos using a minimal set of discrete tokens. We propose VQToken, a neural discrete token representation framework that (i) applies adaptive vector quantization to continuous ViT embeddings to learn a compact codebook and (ii) preserves spatial-temporal positions via a token hash function by assigning each grid-level token to its nearest codebook entry. On the Extreme Short Token Reduction task, our VQToken compresses sequences to just 0.07 percent of their original length while incurring only a 0.66 percent drop in accuracy on the NextQA-MC benchmark. It also achieves comparable performance on ActNet-QA, Long Video Bench, and VideoMME. We further introduce the Token Information Density (TokDense) metric and formalize fixed-length and adaptive-length subtasks, achieving state-of-the-art results in both settings. Our approach dramatically lowers theoretical complexity, increases information density, drastically reduces token counts, and enables efficient video LLMs in resource-constrained environments.
>
---
#### [replaced 006] SafeEraser: Enhancing Safety in Multimodal Large Language Models through Multimodal Machine Unlearning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12520v3](http://arxiv.org/pdf/2502.12520v3)**

> **作者:** Junkai Chen; Zhijie Deng; Kening Zheng; Yibo Yan; Shuliang Liu; PeiJun Wu; Peijie Jiang; Jia Liu; Xuming Hu
>
> **摘要:** As Multimodal Large Language Models (MLLMs) develop, their potential security issues have become increasingly prominent. Machine Unlearning (MU), as an effective strategy for forgetting specific knowledge in training data, has been widely used in privacy protection. However, MU for safety in MLLM has yet to be fully explored. To address this issue, we propose SAFEERASER, a safety unlearning benchmark for MLLMs, consisting of 3,000 images and 28.8K VQA pairs. We comprehensively evaluate unlearning methods from two perspectives: forget quality and model utility. Our findings show that existing MU methods struggle to maintain model performance while implementing the forget operation and often suffer from over-forgetting. Hence, we introduce Prompt Decouple (PD) Loss to alleviate over-forgetting through decouple prompt during unlearning process. To quantitatively measure over-forgetting mitigated by PD Loss, we propose a new metric called Safe Answer Refusal Rate (SARR). Experimental results demonstrate that combining PD Loss with existing unlearning methods can effectively prevent over-forgetting and achieve a decrease of 79.5% in the SARR metric of LLaVA-7B and LLaVA-13B, while maintaining forget quality and model utility. Our code and dataset will be released upon acceptance. Warning: This paper contains examples of harmful language and images, and reader discretion is recommended.
>
---
#### [replaced 007] How Good are Foundation Models in Step-by-Step Embodied Reasoning?
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.15293v2](http://arxiv.org/pdf/2509.15293v2)**

> **作者:** Dinura Dissanayake; Ahmed Heakl; Omkar Thawakar; Noor Ahsan; Ritesh Thawkar; Ketan More; Jean Lahoud; Rao Anwer; Hisham Cholakkal; Ivan Laptev; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** Project page: https://mbzuai-oryx.github.io/FoMER-Bench/
>
> **摘要:** Embodied agents operating in the physical world must make decisions that are not only effective but also safe, spatially coherent, and grounded in context. While recent advances in large multimodal models (LMMs) have shown promising capabilities in visual understanding and language generation, their ability to perform structured reasoning for real-world embodied tasks remains underexplored. In this work, we aim to understand how well foundation models can perform step-by-step reasoning in embodied environments. To this end, we propose the Foundation Model Embodied Reasoning (FoMER) benchmark, designed to evaluate the reasoning capabilities of LMMs in complex embodied decision-making scenarios. Our benchmark spans a diverse set of tasks that require agents to interpret multimodal observations, reason about physical constraints and safety, and generate valid next actions in natural language. We present (i) a large-scale, curated suite of embodied reasoning tasks, (ii) a novel evaluation framework that disentangles perceptual grounding from action reasoning, and (iii) empirical analysis of several leading LMMs under this setting. Our benchmark includes over 1.1k samples with detailed step-by-step reasoning across 10 tasks and 8 embodiments, covering three different robot types. Our results highlight both the potential and current limitations of LMMs in embodied reasoning, pointing towards key challenges and opportunities for future research in robot intelligence. Our data and code will be made publicly available.
>
---
#### [replaced 008] Neural Antidote: Class-Wise Prompt Tuning for Purifying Backdoors in CLIP
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.19269v2](http://arxiv.org/pdf/2502.19269v2)**

> **作者:** Jiawei Kong; Hao Fang; Sihang Guo; Chenxi Qing; Kuofeng Gao; Bin Chen; Shu-Tao Xia; Ke Xu
>
> **摘要:** While pre-trained Vision-Language Models (VLMs) such as CLIP exhibit impressive representational capabilities for multimodal data, recent studies have revealed their vulnerability to backdoor attacks. To alleviate the threat, existing defense strategies primarily focus on fine-tuning the entire suspicious model. However, the substantial model parameters increase the difficulty of reaching a stable and consistent optimization direction, limiting their resistance against state-of-the-art attacks and often resulting in a degradation of clean accuracy. To address this challenge, we propose Class-wise Backdoor Prompt Tuning (CBPT), an efficient and effective defense mechanism that operates on text prompts to indirectly purify poisoned CLIP. Specifically, we first employ the advanced contrastive learning via carefully crafted positive and negative samples, to effectively invert the backdoor triggers that are potentially adopted by the attacker. Once the dummy trigger is established, we leverage three well-designed loss functions to optimize these class-wise text prompts, modifying the model's decision boundary and further reclassifying the feature regions affected by backdoor triggers. Extensive experiments demonstrate that CBPT significantly mitigates backdoor threats while preserving model utility, e.g. an average Clean Accuracy (CA) of 58.83% and an Attack Success Rate (ASR) of 0.39% across seven mainstream backdoor attacks. These results underscore the superiority of our prompt purifying design to strengthen CLIP's robustness against backdoor attacks.
>
---
#### [replaced 009] CLIP-IN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02329v2](http://arxiv.org/pdf/2508.02329v2)**

> **作者:** Ziteng Wang; Siqi Yang; Limeng Qiao; Lin Ma
>
> **备注:** NeurIPS 2025 Main
>
> **摘要:** Despite the success of Vision-Language Models (VLMs) like CLIP in aligning vision and language, their proficiency in detailed, fine-grained visual comprehension remains a key challenge. We present CLIP-IN, a novel framework that bolsters CLIP's fine-grained perception through two core innovations. Firstly, we leverage instruction-editing datasets, originally designed for image manipulation, as a unique source of hard negative image-text pairs. Coupled with a symmetric hard negative contrastive loss, this enables the model to effectively distinguish subtle visual-semantic differences. Secondly, CLIP-IN incorporates long descriptive captions, utilizing rotary positional encodings to capture rich semantic context often missed by standard CLIP. Our experiments demonstrate that CLIP-IN achieves substantial gains on the MMVP benchmark and various fine-grained visual recognition tasks, without compromising robust zero-shot performance on broader classification and retrieval tasks. Critically, integrating CLIP-IN's visual representations into Multimodal Large Language Models significantly reduces visual hallucinations and enhances reasoning abilities. This work underscores the considerable potential of synergizing targeted, instruction-based contrastive learning with comprehensive descriptive information to elevate the fine-grained understanding of VLMs.
>
---
#### [replaced 010] Image-to-Brain Signal Generation for Visual Prosthesis with CLIP Guided Multimodal Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00787v3](http://arxiv.org/pdf/2509.00787v3)**

> **作者:** Ganxi Xu; Jinyi Long; Jia Zhang
>
> **摘要:** Visual prostheses hold great promise for restoring vision in blind individuals. While researchers have successfully utilized M/EEG signals to evoke visual perceptions during the brain decoding stage of visual prostheses, the complementary process of converting images into M/EEG signals in the brain encoding stage remains largely unexplored, hindering the formation of a complete functional pipeline. In this work, we present, to our knowledge, the first image-to-brain signal framework that generates M/EEG from images by leveraging denoising diffusion probabilistic models enhanced with cross-attention mechanisms. Specifically, the proposed framework comprises two key components: a pretrained CLIP visual encoder that extracts rich semantic representations from input images, and a cross-attention enhanced U-Net diffusion model that reconstructs brain signals through iterative denoising. Unlike conventional generative models that rely on simple concatenation for conditioning, our cross-attention modules capture the complex interplay between visual features and brain signal representations, enabling fine-grained alignment during generation. We evaluate the framework on two multimodal benchmark datasets and demonstrate that it generates biologically plausible brain signals. We also present visualizations of M/EEG topographies across all subjects in both datasets, providing intuitive demonstrations of intra-subject and inter-subject variations in brain signals.
>
---
#### [replaced 011] Revisiting Speech-Lip Alignment: A Phoneme-Aware Speech Encoder for Robust Talking Head Synthesis
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.05803v2](http://arxiv.org/pdf/2504.05803v2)**

> **作者:** Yihuan Huang; Jiajun Liu; Yanzhen Ren; Wuyang Liu; Zongkun Sun
>
> **摘要:** Speech-driven talking head synthesis tasks commonly use general acoustic features as guided speech features. However, we discovered that these features suffer from phoneme-viseme alignment ambiguity, which refers to the uncertainty and imprecision in matching phonemes with visemes. To overcome this limitation, we propose a phoneme-aware speech encoder (PASE) that explicitly enforces accurate phoneme-viseme correspondence. PASE first captures fine-grained speech and visual features, then introduces a prediction-reconstruction task to improve robustness under noise and modality absence. Furthermore, a phoneme-level alignment module guided by phoneme embeddings and contrastive learning ensures discriminative audio and visual alignment. Experimental results show that PASE achieves state-of-the-art performance in both NeRF and 3DGS rendering models. Its lip sync accuracy improves by 13.7% and 14.2% compared to the acoustic feature, producing results close to the ground truth videos.
>
---
#### [replaced 012] Single-step Diffusion for Image Compression at Ultra-Low Bitrates
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16572v2](http://arxiv.org/pdf/2506.16572v2)**

> **作者:** Chanung Park; Joo Chan Lee; Jong Hwan Ko
>
> **摘要:** Although there have been significant advancements in image compression techniques, such as standard and learned codecs, these methods still suffer from severe quality degradation at extremely low bits per pixel. While recent diffusion-based models provided enhanced generative performance at low bitrates, they often yields limited perceptual quality and prohibitive decoding latency due to multiple denoising steps. In this paper, we propose the single-step diffusion model for image compression that delivers high perceptual quality and fast decoding at ultra-low bitrates. Our approach incorporates two key innovations: (i) Vector-Quantized Residual (VQ-Residual) training, which factorizes a structural base code and a learned residual in latent space, capturing both global geometry and high-frequency details; and (ii) rate-aware noise modulation, which tunes denoising strength to match the desired bitrate. Extensive experiments show that ours achieves comparable compression performance to state-of-the-art methods while improving decoding speed by about 50x compared to prior diffusion-based methods, greatly enhancing the practicality of generative codecs.
>
---
#### [replaced 013] Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning
- **分类: cs.CV; cs.CG**

- **链接: [http://arxiv.org/pdf/2509.07493v2](http://arxiv.org/pdf/2509.07493v2)**

> **作者:** Wenzhi Guo; Bing Wang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as a powerful paradigm for photorealistic view synthesis, representing scenes with spatially distributed Gaussian primitives. While highly effective for rendering, achieving accurate and complete surface reconstruction remains challenging due to the unstructured nature of the representation and the absence of explicit geometric supervision. In this work, we propose DiGS, a unified framework that embeds Signed Distance Field (SDF) learning directly into the 3DGS pipeline, thereby enforcing strong and interpretable surface priors. By associating each Gaussian with a learnable SDF value, DiGS explicitly aligns primitives with underlying geometry and improves cross-view consistency. To further ensure dense and coherent coverage, we design a geometry-guided grid growth strategy that adaptively distributes Gaussians along geometry-consistent regions under a multi-scale hierarchy. Extensive experiments on standard benchmarks, including DTU, Mip-NeRF 360, and Tanks& Temples, demonstrate that DiGS consistently improves reconstruction accuracy and completeness while retaining high rendering fidelity.
>
---
#### [replaced 014] LoFT: Parameter-Efficient Fine-Tuning for Long-tailed Semi-Supervised Learning in Open-World Scenarios
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09926v2](http://arxiv.org/pdf/2509.09926v2)**

> **作者:** Zhiyuan Huang; Jiahao Chen; Yurou Liu; Bing Su
>
> **摘要:** Long-tailed learning has garnered increasing attention due to its wide applicability in real-world scenarios. Among existing approaches, Long-Tailed Semi-Supervised Learning (LTSSL) has emerged as an effective solution by incorporating a large amount of unlabeled data into the imbalanced labeled dataset. However, most prior LTSSL methods are designed to train models from scratch, which often leads to issues such as overconfidence and low-quality pseudo-labels. To address these challenges, we extend LTSSL into the foundation model fine-tuning paradigm and propose a novel framework: LoFT (Long-tailed semi-supervised learning via parameter-efficient Fine-Tuning). We demonstrate that fine-tuned foundation models can generate more reliable pseudolabels, thereby benefiting imbalanced learning. Furthermore, we explore a more practical setting by investigating semi-supervised learning under open-world conditions, where the unlabeled data may include out-of-distribution (OOD) samples. To handle this problem, we propose LoFT-OW (LoFT under Open-World scenarios) to improve the discriminative ability. Experimental results on multiple benchmarks demonstrate that our method achieves superior performance compared to previous approaches, even when utilizing only 1\% of the unlabeled data compared with previous works.
>
---
#### [replaced 015] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19972v2](http://arxiv.org/pdf/2508.19972v2)**

> **作者:** Seongheon Park; Yixuan Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Object hallucination in large vision-language models presents a significant challenge to their safe deployment in real-world applications. Recent works have proposed object-level hallucination scores to estimate the likelihood of object hallucination; however, these methods typically adopt either a global or local perspective in isolation, which may limit detection reliability. In this paper, we introduce GLSim, a novel training-free object hallucination detection framework that leverages complementary global and local embedding similarity signals between image and text modalities, enabling more accurate and reliable hallucination detection in diverse scenarios. We comprehensively benchmark existing object hallucination detection methods and demonstrate that GLSim achieves superior detection performance, outperforming competitive baselines by a significant margin.
>
---
#### [replaced 016] Survey of Video Diffusion Models: Foundations, Implementations, and Applications
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.16081v2](http://arxiv.org/pdf/2504.16081v2)**

> **作者:** Yimu Wang; Xuye Liu; Wei Pang; Li Ma; Shuai Yuan; Paul Debevec; Ning Yu
>
> **备注:** Accepted by TMLR
>
> **摘要:** Recent advances in diffusion models have revolutionized video generation, offering superior temporal consistency and visual quality compared to traditional generative adversarial networks-based approaches. While this emerging field shows tremendous promise in applications, it faces significant challenges in motion consistency, computational efficiency, and ethical considerations. This survey provides a comprehensive review of diffusion-based video generation, examining its evolution, technical foundations, and practical applications. We present a systematic taxonomy of current methodologies, analyze architectural innovations and optimization strategies, and investigate applications across low-level vision tasks such as denoising and super-resolution. Additionally, we explore the synergies between diffusionbased video generation and related domains, including video representation learning, question answering, and retrieval. Compared to the existing surveys (Lei et al., 2024a;b; Melnik et al., 2024; Cao et al., 2023; Xing et al., 2024c) which focus on specific aspects of video generation, such as human video synthesis (Lei et al., 2024a) or long-form content generation (Lei et al., 2024b), our work provides a broader, more updated, and more fine-grained perspective on diffusion-based approaches with a special section for evaluation metrics, industry solutions, and training engineering techniques in video generation. This survey serves as a foundational resource for researchers and practitioners working at the intersection of diffusion models and video generation, providing insights into both the theoretical frameworks and practical implementations that drive this rapidly evolving field. A structured list of related works involved in this survey is also available on https://github.com/Eyeline-Research/Survey-Video-Diffusion.
>
---
#### [replaced 017] Anatomical feature-prioritized loss for enhanced MR to CT translation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.10328v3](http://arxiv.org/pdf/2410.10328v3)**

> **作者:** Arthur Longuefosse; Baudouin Denis de Senneville; Gael Dournes; Ilyes Benlala; Pascal Desbarats; Fabien Baldacci
>
> **摘要:** In medical image synthesis, the precision of localized structural details is crucial, particularly when addressing specific clinical requirements such as the identification and measurement of fine structures. Traditional methods for image translation and synthesis are generally optimized for global image reconstruction but often fall short in providing the finesse required for detailed local analysis. This study represents a step toward addressing this challenge by introducing a novel anatomical feature-prioritized (AFP) loss function into the synthesis process. This method enhances reconstruction by focusing on clinically significant structures, utilizing features from a pre-trained model designed for a specific downstream task, such as the segmentation of particular anatomical regions. The AFP loss function can replace or complement global reconstruction methods, ensuring a balanced emphasis on both global image fidelity and local structural details. Various implementations of this loss function are explored, including its integration into different synthesis networks such as GAN-based and CNN-based models. Our approach is applied and evaluated in two contexts: lung MR to CT translation, focusing on high-quality reconstruction of bronchial structures, using a private dataset; and pelvis MR to CT synthesis, targeting the accurate representation of organs and muscles, utilizing a public dataset from the Synthrad2023 challenge. We leverage embeddings from pre-trained segmentation models specific to these anatomical regions to demonstrate the capability of the AFP loss to prioritize and accurately reconstruct essential features. This tailored approach shows promising potential for enhancing the specificity and practicality of medical image synthesis in clinical applications.
>
---
#### [replaced 018] PROFUSEme: PROstate Cancer Biochemical Recurrence Prediction via FUSEd Multi-modal Embeddings
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.14051v2](http://arxiv.org/pdf/2509.14051v2)**

> **作者:** Suhang You; Carla Pitarch-Abaigar; Sanket Kachole; Sumedh Sonawane; Juhyung Ha; Anish Sudarshan Gada; David Crandall; Rakesh Shiradkar; Spyridon Bakas
>
> **备注:** 11 pages, 1 figure, method paper for CHIMERA 2025 Challenge
>
> **摘要:** Almost 30% of prostate cancer (PCa) patients undergoing radical prostatectomy (RP) experience biochemical recurrence (BCR), characterized by increased prostate specific antigen (PSA) and associated with increased mortality. Accurate early prediction of BCR, at the time of RP, would contribute to prompt adaptive clinical decision-making and improved patient outcomes. In this work, we propose prostate cancer BCR prediction via fused multi-modal embeddings (PROFUSEme), which learns cross-modal interactions of clinical, radiology, and pathology data, following an intermediate fusion configuration in combination with Cox Proportional Hazard regressors. Quantitative evaluation of our proposed approach reveals superior performance, when compared with late fusion configurations, yielding a mean C-index of 0.861 ($\sigma=0.112$) on the internal 5-fold nested cross-validation framework, and a C-index of 0.7107 on the hold out data of CHIMERA 2025 challenge validation leaderboard.
>
---
#### [replaced 019] VocSegMRI: Multimodal Learning for Precise Vocal Tract Segmentation in Real-time MRI
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.13767v2](http://arxiv.org/pdf/2509.13767v2)**

> **作者:** Daiqi Liu; Tomás Arias-Vergara; Johannes Enk; Fangxu Xing; Maureen Stone; Jerry L. Prince; Jana Hutter; Andreas Maier; Jonghye Woo; Paula Andrea Pérez-Toro
>
> **备注:** Preprint submitted to ICASSP
>
> **摘要:** Accurately segmenting articulatory structures in real-time magnetic resonance imaging (rtMRI) remains challenging, as most existing methods rely almost entirely on visual cues. Yet synchronized acoustic and phonological signals provide complementary context that can enrich visual information and improve precision. In this paper, we introduce VocSegMRI, a multimodal framework that integrates video, audio, and phonological inputs through cross-attention fusion for dynamic feature alignment. To further enhance cross-modal representation, we incorporate a contrastive learning objective that improves segmentation performance even when the audio modality is unavailable at inference. Evaluated on a sub-set of USC-75 rtMRI dataset, our approach achieves state-of-the-art performance, with a Dice score of 0.95 and a 95th percentile Hausdorff Distance (HD_95) of 4.20 mm, outperforming both unimodal and multimodal baselines. Ablation studies confirm the contributions of cross-attention and contrastive learning to segmentation precision and robustness. These results highlight the value of integrative multimodal modeling for accurate vocal tract analysis.
>
---
#### [replaced 020] Look, Focus, Act: Efficient and Robust Robot Learning via Human Gaze and Foveated Vision Transformers
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15833v2](http://arxiv.org/pdf/2507.15833v2)**

> **作者:** Ian Chuang; Jinyu Zou; Andrew Lee; Dechen Gao; Iman Soltani
>
> **备注:** Project page: https://ian-chuang.github.io/gaze-av-aloha/
>
> **摘要:** Human vision is a highly active process driven by gaze, which directs attention to task-relevant regions through foveation, dramatically reducing visual processing. In contrast, robot learning systems typically rely on passive, uniform processing of raw camera images. In this work, we explore how incorporating human-like active gaze into robotic policies can enhance efficiency and robustness. We develop GIAVA (Gaze Integrated Active-Vision ALOHA), a robot vision system that emulates human head and neck movement, and gaze adjustment for foveated processing. Extending the AV-ALOHA robot platform, we introduce a framework for simultaneously collecting eye-tracking, perspective control, and robot manipulation demonstration data from a human operator. We also open-source a simulation benchmark and dataset for training robot policies that incorporate human gaze. Inspired by recent work in foveated image segmentation and given the widespread use of Vision Transformers (ViTs) in robot learning, we integrate gaze information into ViTs using a foveated patch tokenization scheme. Compared to uniform patch tokenization, this significantly reduces the number of tokens, and thus computation. Our results show that our method for foveated robot vision drastically reduces computational overhead, and enhances robustness to background distractors. Notably, on certain high-precision tasks, foveated vision also improves performance, as reflected in higher success rates. Together, these findings suggest that human-inspired foveated visual processing offers untapped potential and should be further considered as a useful inductive bias in robotic vision systems. https://ian-chuang.github.io/gaze-av-aloha/
>
---
#### [replaced 021] Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00416v2](http://arxiv.org/pdf/2507.00416v2)**

> **作者:** Tao Lin; Gen Li; Yilei Zhong; Yanwen Zou; Yuxin Du; Jiting Liu; Encheng Gu; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising framework for enabling generalist robots capable of perceiving, reasoning, and acting in the real world. These models usually build upon pretrained Vision-Language Models (VLMs), which excel at semantic understanding due to large-scale image and text pretraining. However, existing VLMs typically lack precise spatial understanding capabilities, as they are primarily tuned on 2D image-text pairs without 3D supervision. To address this limitation, recent approaches have incorporated explicit 3D inputs such as point clouds or depth maps, but this necessitates additional depth sensors or pre-trained depth estimation models, which may yield defective results. In contrast, our work introduces a plug-and-play module that implicitly incorporates 3D geometry features into VLA models by leveraging an off-the-shelf visual geometry foundation model. This integration provides the model with depth-aware visual representations, improving its ability to understand the geometric structure of the scene and the spatial relationships among objects from RGB images alone. We evaluate our method on a set of spatially challenging tasks in both simulation and the real world. Extensive evaluations show that our method significantly improves the performance of state-of-the-art VLA models across diverse scenarios.
>
---
#### [replaced 022] Generating 360° Video is What You Need For a 3D Scene
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02045v2](http://arxiv.org/pdf/2504.02045v2)**

> **作者:** Zhaoyang Zhang; Yannick Hold-Geoffroy; Miloš Hašan; Ziwen Chen; Fujun Luan; Julie Dorsey; Yiwei Hu
>
> **摘要:** Generating 3D scenes is still a challenging task due to the lack of readily available scene data. Most existing methods only produce partial scenes and provide limited navigational freedom. We introduce a practical and scalable solution that uses 360{\deg} video as an intermediate scene representation, capturing the full-scene context and ensuring consistent visual content throughout the generation. We propose WorldPrompter, a generative pipeline that synthesizes traversable 3D scenes from text prompts. WorldPrompter incorporates a conditional 360{\deg} panoramic video generator, capable of producing a 128-frame video that simulates a person walking through and capturing a virtual environment. The resulting video is then reconstructed as Gaussian splats by a fast feedforward 3D reconstructor, enabling a true walkable experience within the 3D scene. Experiments demonstrate that our panoramic video generation model, trained with a mix of image and video data, achieves convincing spatial and temporal consistency for static scenes. This is validated by an average COLMAP matching rate of 94.6\%, allowing for high-quality panoramic Gaussian splat reconstruction and improved navigation throughout the scene. Qualitative and quantitative results also show it outperforms the state-of-the-art 360{\deg} video generators and 3D scene generation models.
>
---
#### [replaced 023] LEMUR Neural Network Dataset: Towards Seamless AutoML
- **分类: cs.LG; cs.AI; cs.CV; cs.DL**

- **链接: [http://arxiv.org/pdf/2504.10552v3](http://arxiv.org/pdf/2504.10552v3)**

> **作者:** Arash Torabi Goodarzi; Roman Kochnev; Waleed Khalid; Hojjat Torabi Goudarzi; Furui Qin; Tolgay Atinc Uzun; Yashkumar Sanjaybhai Dhameliya; Yash Kanubhai Kathiriya; Zofia Antonina Bentyn; Dmitry Ignatov; Radu Timofte
>
> **摘要:** Neural networks have become the backbone of modern AI, yet designing, evaluating, and comparing them remains labor-intensive. While many datasets exist for training models, there are few standardized collections of the models themselves. We present LEMUR, an open-source dataset and framework that brings together a large collection of PyTorch-based neural networks across tasks such as classification, segmentation, detection, and natural language processing. Each model follows a common template, with configurations and results logged in a structured database to ensure consistency and reproducibility. LEMUR integrates Optuna for automated hyperparameter optimization, provides statistical analysis and visualization tools, and exposes an API for seamless access to performance data. The framework also supports extensibility, enabling researchers to add new models, datasets, or metrics without breaking compatibility. By standardizing implementations and unifying evaluation, LEMUR aims to accelerate AutoML research, facilitate fair benchmarking, and lower the barrier to large-scale neural network experimentation.
>
---
#### [replaced 024] ViLReF: An Expert Knowledge Enabled Vision-Language Retinal Foundation Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.10894v4](http://arxiv.org/pdf/2408.10894v4)**

> **作者:** Shengzhu Yang; Jiawei Du; Jia Guo; Weihang Zhang; Hanruo Liu; Huiqi Li; Ningli Wang
>
> **摘要:** Subtle semantic differences in retinal image and text data present great challenges for pre-training visual-language models. Moreover, false negative samples, i.e., image-text pairs having the same semantics but incorrectly regarded as negatives, disrupt the visual-language pre-training process and affect the model's learning ability. This work aims to develop a retinal foundation model, called ViLReF, by pre-training on a paired dataset comprising 451,956 retinal images and corresponding diagnostic text reports. In our vision-language pre-training strategy, we leverage expert knowledge to facilitate the extraction of labels and propose a novel constraint, the Weighted Similarity Coupling Loss, to adjust the speed of pushing sample pairs further apart dynamically within the feature space. Furthermore, we employ a batch expansion module with dynamic memory queues, maintained by momentum encoders, to supply extra samples and compensate for the vacancies caused by eliminating false negatives. Extensive experiments are conducted on multiple datasets for downstream classification and segmentation tasks. The experimental results demonstrate the powerful zero-shot and transfer learning capabilities of ViLReF, verifying the effectiveness of our pre-training strategy. Our ViLReF model is available at: https://github.com/T6Yang/ViLReF.
>
---
#### [replaced 025] Pulling Back the Curtain on ReLU Networks
- **分类: cs.LG; cs.CV; cs.NE; I.2.6; I.4.10**

- **链接: [http://arxiv.org/pdf/2507.22832v4](http://arxiv.org/pdf/2507.22832v4)**

> **作者:** Maciej Satkiewicz
>
> **备注:** 12 pages, 3-page appendix, 4 figures, under review; v4 changes: wording improvements, clarification of arguments and of the Hypothesis 1
>
> **摘要:** Since any ReLU network is piecewise affine, its hidden units can be characterized by their pullbacks through the active subnetwork, i.e., by their gradients (up to bias terms). However, gradients of deeper neurons are notoriously misaligned, which obscures the network's internal representations. We posit that models do align gradients with data, yet this is concealed by the intrinsic noise of the ReLU hard gating. We validate this intuition by applying soft gating in the backward pass only, reducing the local impact of weakly excited neurons. The resulting modified gradients, which we call "excitation pullbacks", exhibit striking perceptual alignment on a number of ImageNet-pretrained architectures, while the rudimentary pixel-space gradient ascent quickly produces easily interpretable input- and target-specific features. Inspired by these findings, we formulate the "path stability" hypothesis, claiming that the binary activation patterns largely stabilize during training and get encoded in the pre-activation distribution of the final model. When true, excitation pullbacks become aligned with the gradients of a kernel machine that mainly determines the network's decision. This provides a theoretical justification for the apparent faithfulness of the feature attributions based on excitation pullbacks, potentially even leading to mechanistic interpretability of deep models. Incidentally, we give a possible explanation for the effectiveness of Batch Normalization and Deep Features, together with a novel perspective on the network's internal memory and generalization properties. We release the code and an interactive app for easier exploration of the excitation pullbacks.
>
---
#### [replaced 026] Reflecting on the State of Rehearsal-free Continual Learning with Pretrained Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.09384v2](http://arxiv.org/pdf/2406.09384v2)**

> **作者:** Lukas Thede; Karsten Roth; Olivier J. Hénaff; Matthias Bethge; Zeynep Akata
>
> **备注:** 3rd Conference on Lifelong Learning Agents (CoLLAs) 2024
>
> **摘要:** With the advent and recent ubiquity of foundation models, continual learning (CL) has recently shifted from continual training from scratch to the continual adaptation of pretrained models, seeing particular success on rehearsal-free CL benchmarks (RFCL). To achieve this, most proposed methods adapt and restructure parameter-efficient finetuning techniques (PEFT) to suit the continual nature of the problem. Based most often on input-conditional query-mechanisms or regularizations on top of prompt- or adapter-based PEFT, these PEFT-style RFCL (P-RFCL) approaches report peak performances; often convincingly outperforming existing CL techniques. However, on the other end, critical studies have recently highlighted competitive results by training on just the first task or via simple non-parametric baselines. Consequently, questions arise about the relationship between methodological choices in P-RFCL and their reported high benchmark scores. In this work, we tackle these questions to better understand the true drivers behind strong P-RFCL performances, their placement w.r.t. recent first-task adaptation studies, and their relation to preceding CL standards such as EWC or SI. In particular, we show: (1) P-RFCL techniques relying on input-conditional query mechanisms work not because, but rather despite them by collapsing towards standard PEFT shortcut solutions. (2) Indeed, we show how most often, P-RFCL techniques can be matched by a simple and lightweight PEFT baseline. (3) Using this baseline, we identify the implicit bound on tunable parameters when deriving RFCL approaches from PEFT methods as a potential denominator behind P-RFCL efficacy. Finally, we (4) better disentangle continual versus first-task adaptation, and (5) motivate standard RFCL techniques s.a. EWC or SI in light of recent P-RFCL methods.
>
---
#### [replaced 027] 3D Cell Oversegmentation Correction via Geo-Wasserstein Divergence
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01890v3](http://arxiv.org/pdf/2502.01890v3)**

> **作者:** Peter Chen; Bryan Chang; Olivia Annette Creasey; Julie Beth Sneddon; Zev Gartner; Yining Liu
>
> **摘要:** 3D cell segmentation methods are often hindered by \emph{oversegmentation}, where a single cell is incorrectly split into multiple fragments. This degrades the final segmentation quality and is notoriously difficult to resolve, as oversegmentation errors often resemble natural gaps between adjacent cells. Our work makes two key contributions. First, for 3D cell segmentation, we are the first work to formulate oversegmentation as a concrete problem and propose a geometric framework to identify and correct these errors. Our approach builds a pre-trained classifier using both 2D geometric and 3D topological features extracted from flawed 3D segmentation results. Second, we introduce a novel metric, Geo-Wasserstein divergence, to quantify changes in 2D geometries. This captures the evolving trends of cell mask shape in a geometry-aware manner. We validate our method through extensive experiments on in-domain plant datasets, including both synthesized and real oversegmented cases, as well as on out-of-domain animal datasets to demonstrate transfer learning performance. An ablation study further highlights the contribution of the Geo-Wasserstein divergence. A clear pipeline is provided for end-users to build pre-trained models to any labeled dataset.
>
---
#### [replaced 028] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06970v2](http://arxiv.org/pdf/2506.06970v2)**

> **作者:** Pengfei Zhao; Rongbo Luan; Wei Zhang; Peng Wu; Sifeng He
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Despite Contrastive Language-Image Pretraining (CLIP)'s remarkable capability to retrieve content across modalities, a substantial modality gap persists in its feature space. Intriguingly, we discover that off-the-shelf MLLMs (Multimodal Large Language Models) demonstrate powerful inherent modality alignment properties. While recent MLLM-based retrievers with unified architectures partially mitigate this gap, their reliance on coarse modality alignment mechanisms fundamentally limits their potential. In this work, We introduce MAPLE (Modality-Aligned Preference Learning for Embeddings), a novel framework that leverages the fine grained alignment priors inherent in MLLM to guide cross modal representation learning. MAPLE formulates the learning process as reinforcement learning with two key components: (1) Automatic preference data construction using off-the-shelf MLLM, and (2) a new Relative Preference Alignment (RPA) loss, which adapts Direct Preference Optimization (DPO) to the embedding learning setting. Experimental results show that our preference-guided alignment achieves substantial gains in fine-grained cross-modal retrieval, underscoring its effectiveness in handling nuanced semantic distinctions.
>
---
#### [replaced 029] BiPrompt-SAM: Enhancing Image Segmentation via Explicit Selection between Point and Text Prompts
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.19769v3](http://arxiv.org/pdf/2503.19769v3)**

> **作者:** Suzhe Xu; Jialin Peng; Chengyuan Zhang
>
> **备注:** metrics went wrong
>
> **摘要:** Segmentation is a fundamental task in computer vision, with prompt-driven methods gaining prominence due to their flexibility. The Segment Anything Model (SAM) excels at point-prompted segmentation, while text-based models, often leveraging powerful multimodal encoders like BEIT-3, provide rich semantic understanding. However, effectively combining these complementary modalities remains a challenge. This paper introduces BiPrompt-SAM, a novel dual-modal prompt segmentation framework employing an explicit selection mechanism. We leverage SAM's ability to generate multiple mask candidates from a single point prompt and use a text-guided mask (generated via EVF-SAM with BEIT-3) to select the point-generated mask that best aligns spatially, measured by Intersection over Union (IoU). This approach, interpretable as a simplified Mixture of Experts (MoE), effectively fuses spatial precision and semantic context without complex model modifications. Notably, our method achieves strong zero-shot performance on the Endovis17 medical dataset (89.55% mDice, 81.46% mIoU) using only a single point prompt per instance. This significantly reduces annotation burden compared to bounding boxes and aligns better with practical clinical workflows, demonstrating the method's effectiveness without domain-specific training. On the RefCOCO series, BiPrompt-SAM attained 87.1%, 86.5%, and 85.8% IoU, significantly outperforming existing approaches. Experiments show BiPrompt-SAM excels in scenarios requiring both spatial accuracy and semantic disambiguation, offering a simple, effective, and interpretable perspective on multi-modal prompt fusion.
>
---
#### [replaced 030] AIM 2025 challenge on Inverse Tone Mapping Report: Methods and Results
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.13479v2](http://arxiv.org/pdf/2508.13479v2)**

> **作者:** Chao Wang; Francesco Banterle; Bin Ren; Radu Timofte; Xin Lu; Yufeng Peng; Chengjie Ge; Zhijing Sun; Ziang Zhou; Zihao Li; Zishun Liao; Qiyu Kang; Xueyang Fu; Zheng-Jun Zha; Zhijing Sun; Xingbo Wang; Kean Liu; Senyan Xu; Yang Qiu; Yifan Ding; Gabriel Eilertsen; Jonas Unger; Zihao Wang; Ke Wu; Jinshan Pan; Zhen Liu; Zhongyang Li; Shuaicheng Liu; S. M Nadim Uddin
>
> **摘要:** This paper presents a comprehensive review of the AIM 2025 Challenge on Inverse Tone Mapping (ITM). The challenge aimed to push forward the development of effective ITM algorithms for HDR image reconstruction from single LDR inputs, focusing on perceptual fidelity and numerical consistency. A total of \textbf{67} participants submitted \textbf{319} valid results, from which the best five teams were selected for detailed analysis. This report consolidates their methodologies and performance, with the lowest PU21-PSNR among the top entries reaching 29.22 dB. The analysis highlights innovative strategies for enhancing HDR reconstruction quality and establishes strong benchmarks to guide future research in inverse tone mapping.
>
---
#### [replaced 031] Both Text and Images Leaked! A Systematic Analysis of Data Contamination in Multimodal LLM
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.03823v3](http://arxiv.org/pdf/2411.03823v3)**

> **作者:** Dingjie Song; Sicheng Lai; Mingxuan Wang; Shunian Chen; Lichao Sun; Benyou Wang
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** The rapid advancement of multimodal large language models (MLLMs) has significantly enhanced performance across benchmarks. However, data contamination-unintentional memorization of benchmark data during model training-poses critical challenges for fair evaluation. Existing detection methods for unimodal large language models (LLMs) are inadequate for MLLMs due to multimodal data complexity and multi-phase training. We systematically analyze multimodal data contamination using our analytical framework, MM-Detect, which defines two contamination categories-unimodal and cross-modal-and effectively quantifies contamination severity across multiple-choice and caption-based Visual Question Answering tasks. Evaluations on twelve MLLMs and five benchmarks reveal significant contamination, particularly in proprietary models and older benchmarks. Crucially, contamination sometimes originates during unimodal pre-training rather than solely from multimodal fine-tuning. Our insights refine contamination understanding, guiding evaluation practices and improving multimodal model reliability.
>
---
#### [replaced 032] QA-HFL: Quality-Aware Hierarchical Federated Learning for Resource-Constrained Mobile Devices with Heterogeneous Image Quality
- **分类: cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05411v2](http://arxiv.org/pdf/2506.05411v2)**

> **作者:** Sajid Hussain; Muhammad Sohail; Nauman Ali Khan
>
> **备注:** Due to some technical issues
>
> **摘要:** This paper introduces QA-HFL, a quality-aware hierarchical federated learning framework that efficiently handles heterogeneous image quality across resource-constrained mobile devices. Our approach trains specialized local models for different image quality levels and aggregates their features using a quality-weighted fusion mechanism, while incorporating differential privacy protection. Experiments on MNIST demonstrate that QA-HFL achieves 92.31% accuracy after just three federation rounds, significantly outperforming state-of-the-art methods like FedRolex (86.42%). Under strict privacy constraints, our approach maintains 30.77% accuracy with formal differential privacy guarantees. Counter-intuitively, low-end devices contributed most significantly (63.5%) to the final model despite using 100 fewer parameters than high-end counterparts. Our quality-aware approach addresses accuracy decline through device-specific regularization, adaptive weighting, intelligent client selection, and server-side knowledge distillation, while maintaining efficient communication with a 4.71% compression ratio. Statistical analysis confirms that our approach significantly outperforms baseline methods (p 0.01) under both standard and privacy-constrained conditions.
>
---
#### [replaced 033] Photography Perspective Composition: Towards Aesthetic Perspective Recommendation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20655v2](http://arxiv.org/pdf/2505.20655v2)**

> **作者:** Lujian Yao; Siming Zheng; Xinbin Yuan; Zhuoxuan Cai; Pu Wu; Jinwei Chen; Bo Li; Peng-Tao Jiang
>
> **摘要:** Traditional photography composition approaches are dominated by 2D cropping-based methods. However, these methods fall short when scenes contain poorly arranged subjects. Professional photographers often employ perspective adjustment as a form of 3D recomposition, modifying the projected 2D relationships between subjects while maintaining their actual spatial positions to achieve better compositional balance. Inspired by this artistic practice, we propose photography perspective composition (PPC), extending beyond traditional cropping-based methods. However, implementing the PPC faces significant challenges: the scarcity of perspective transformation datasets and undefined assessment criteria for perspective quality. To address these challenges, we present three key contributions: (1) An automated framework for building PPC datasets through expert photographs. (2) A video generation approach that demonstrates the transformation process from suboptimal to optimal perspectives. (3) A perspective quality assessment (PQA) model constructed based on human performance. Our approach is concise and requires no additional prompt instructions or camera trajectories, helping and guiding ordinary users to enhance their composition skills.
>
---
#### [replaced 034] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20168v2](http://arxiv.org/pdf/2506.20168v2)**

> **作者:** Zhentao He; Can Zhang; Ziheng Wu; Zhenghao Chen; Yufei Zhan; Yifan Li; Zhao Zhang; Xian Wang; Minghui Qiu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent advancements in multimodal large language models have enhanced document understanding by integrating textual and visual information. However, existing models exhibit incompleteness within their paradigm in real-world scenarios, particularly under visual degradation. In such conditions, the current response paradigm often fails to adequately perceive visual degradation and ambiguity, leading to overreliance on linguistic priors or misaligned visual-textual reasoning. This difficulty in recognizing uncertainty frequently results in the generation of hallucinatory content, especially when a precise answer is not feasible. To better demonstrate and analyze this phenomenon and problem, we propose KIE-HVQA, the first benchmark dedicated to evaluating OCR hallucination in degraded document understanding. This dataset includes test samples spanning identity cards and invoices, with simulated real-world degradations for OCR reliability. This setup allows for evaluating models' capacity, under degraded input, to distinguish reliable visual information and answer accordingly, thereby highlighting the challenge of avoiding hallucination on uncertain data. To achieve vision-faithful reasoning and thereby avoid the aforementioned issues, we further introduce a GRPO-based framework featuring a novel reward mechanism. By incorporating a self-awareness of visual uncertainty and an analysis method that initiates refusal to answer to increase task difficulty within our supervised fine-tuning and reinforcement learning framework, we successfully mitigated hallucinations in ambiguous regions. Experiments on Qwen2.5-VL demonstrate that our 7B-parameter model achieves a 22\% absolute improvement in hallucination-free accuracy over GPT-4o on KIE-HVQA and there is no significant performance drop in standard tasks, highlighting both effectiveness and robustness.
>
---
#### [replaced 035] Seeing 3D Through 2D Lenses: 3D Few-Shot Class-Incremental Learning via Cross-Modal Geometric Rectification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.14958v2](http://arxiv.org/pdf/2509.14958v2)**

> **作者:** Tuo Xiang; Xuemiao Xu; Bangzhen Liu; Jinyi Li; Yong Li; Shengfeng He
>
> **备注:** ICCV2025
>
> **摘要:** The rapid growth of 3D digital content necessitates expandable recognition systems for open-world scenarios. However, existing 3D class-incremental learning methods struggle under extreme data scarcity due to geometric misalignment and texture bias. While recent approaches integrate 3D data with 2D foundation models (e.g., CLIP), they suffer from semantic blurring caused by texture-biased projections and indiscriminate fusion of geometric-textural cues, leading to unstable decision prototypes and catastrophic forgetting. To address these issues, we propose Cross-Modal Geometric Rectification (CMGR), a framework that enhances 3D geometric fidelity by leveraging CLIP's hierarchical spatial semantics. Specifically, we introduce a Structure-Aware Geometric Rectification module that hierarchically aligns 3D part structures with CLIP's intermediate spatial priors through attention-driven geometric fusion. Additionally, a Texture Amplification Module synthesizes minimal yet discriminative textures to suppress noise and reinforce cross-modal consistency. To further stabilize incremental prototypes, we employ a Base-Novel Discriminator that isolates geometric variations. Extensive experiments demonstrate that our method significantly improves 3D few-shot class-incremental learning, achieving superior geometric coherence and robustness to texture bias across cross-domain and within-domain settings.
>
---
#### [replaced 036] TempFlow-GRPO: When Timing Matters for GRPO in Flow Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04324v2](http://arxiv.org/pdf/2508.04324v2)**

> **作者:** Xiaoxuan He; Siming Fu; Yuke Zhao; Wanli Li; Jian Yang; Dacheng Yin; Fengyun Rao; Bo Zhang
>
> **摘要:** Recent flow matching models for text-to-image generation have achieved remarkable quality, yet their integration with reinforcement learning for human preference alignment remains suboptimal, hindering fine-grained reward-based optimization. We observe that the key impediment to effective GRPO training of flow models is the temporal uniformity assumption in existing approaches: sparse terminal rewards with uniform credit assignment fail to capture the varying criticality of decisions across generation timesteps, resulting in inefficient exploration and suboptimal convergence. To remedy this shortcoming, we introduce \textbf{TempFlow-GRPO} (Temporal Flow GRPO), a principled GRPO framework that captures and exploits the temporal structure inherent in flow-based generation. TempFlow-GRPO introduces three key innovations: (i) a trajectory branching mechanism that provides process rewards by concentrating stochasticity at designated branching points, enabling precise credit assignment without requiring specialized intermediate reward models; (ii) a noise-aware weighting scheme that modulates policy optimization according to the intrinsic exploration potential of each timestep, prioritizing learning during high-impact early stages while ensuring stable refinement in later phases; and (iii) a seed group strategy that controls for initialization effects to isolate exploration contributions. These innovations endow the model with temporally-aware optimization that respects the underlying generative dynamics, leading to state-of-the-art performance in human preference alignment and text-to-image benchmarks.
>
---
#### [replaced 037] Unified Framework for Pre-trained Neural Network Compression via Decomposition and Optimized Rank Selection
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.03555v2](http://arxiv.org/pdf/2409.03555v2)**

> **作者:** Ali Aghababaei-Harandi; Massih-Reza Amini
>
> **摘要:** Despite their high accuracy, complex neural networks demand significant computational resources, posing challenges for deployment on resource constrained devices such as mobile phones and embedded systems. Compression algorithms have been developed to address these challenges by reducing model size and computational demands while maintaining accuracy. Among these approaches, factorization methods based on tensor decomposition are theoretically sound and effective. However, they face difficulties in selecting the appropriate rank for decomposition. This paper tackles this issue by presenting a unified framework that simultaneously applies decomposition and rank selection, employing a composite compression loss within defined rank constraints. Our method includes an automatic rank search in a continuous space, efficiently identifying optimal rank configurations for the pre-trained model by eliminating the need for additional training data and reducing computational overhead in the search step. Combined with a subsequent fine-tuning step, our approach maintains the performance of highly compressed models on par with their original counterparts. Using various benchmark datasets and models, we demonstrate the efficacy of our method through a comprehensive analysis.
>
---
#### [replaced 038] Walk and Read Less: Improving the Efficiency of Vision-and-Language Navigation via Tuning-Free Multimodal Token Pruning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.15250v2](http://arxiv.org/pdf/2509.15250v2)**

> **作者:** Wenda Qin; Andrea Burns; Bryan A. Plummer; Margrit Betke
>
> **备注:** Accepted to EMNLP 2025. Data and code to be released at https://github.com/wdqin/VLN-NAP
>
> **摘要:** Large models achieve strong performance on Vision-and-Language Navigation (VLN) tasks, but are costly to run in resource-limited environments. Token pruning offers appealing tradeoffs for efficiency with minimal performance loss by reducing model input size, but prior work overlooks VLN-specific challenges. For example, information loss from pruning can effectively increase computational cost due to longer walks. Thus, the inability to identify uninformative tokens undermines the supposed efficiency gains from pruning. To address this, we propose Navigation-Aware Pruning (NAP), which uses navigation-specific traits to simplify the pruning process by pre-filtering tokens into foreground and background. For example, image views are filtered based on whether the agent can navigate in that direction. We also extract navigation-relevant instructions using a Large Language Model. After filtering, we focus pruning on background tokens, minimizing information loss. To further help avoid increases in navigation length, we discourage backtracking by removing low-importance navigation nodes. Experiments on standard VLN benchmarks show NAP significantly outperforms prior work, preserving higher success rates while saving more than 50% FLOPS.
>
---
#### [replaced 039] QVGen: Pushing the Limit of Quantized Video Generative Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11497v3](http://arxiv.org/pdf/2505.11497v3)**

> **作者:** Yushi Huang; Ruihao Gong; Jing Liu; Yifu Ding; Chengtao Lv; Haotong Qin; Jun Zhang
>
> **备注:** Our code will be released upon acceptance
>
> **摘要:** Video diffusion models (DMs) have enabled high-quality video synthesis. Yet, their substantial computational and memory demands pose serious challenges to real-world deployment, even on high-end GPUs. As a commonly adopted solution, quantization has proven notable success in reducing cost for image DMs, while its direct application to video DMs remains ineffective. In this paper, we present QVGen, a novel quantization-aware training (QAT) framework tailored for high-performance and inference-efficient video DMs under extremely low-bit quantization (e.g., 4-bit or below). We begin with a theoretical analysis demonstrating that reducing the gradient norm is essential to facilitate convergence for QAT. To this end, we introduce auxiliary modules ($\Phi$) to mitigate large quantization errors, leading to significantly enhanced convergence. To eliminate the inference overhead of $\Phi$, we propose a rank-decay strategy that progressively eliminates $\Phi$. Specifically, we repeatedly employ singular value decomposition (SVD) and a proposed rank-based regularization $\mathbf{\gamma}$ to identify and decay low-contributing components. This strategy retains performance while zeroing out inference overhead. Extensive experiments across $4$ state-of-the-art (SOTA) video DMs, with parameter sizes ranging from $1.3$B $\sim14$B, show that QVGen is the first to reach full-precision comparable quality under 4-bit settings. Moreover, it significantly outperforms existing methods. For instance, our 3-bit CogVideoX-2B achieves improvements of $+25.28$ in Dynamic Degree and $+8.43$ in Scene Consistency on VBench.
>
---
#### [replaced 040] AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04504v3](http://arxiv.org/pdf/2503.04504v3)**

> **作者:** Sunghyun Ahn; Youngwan Jo; Kijung Lee; Sein Kwon; Inpyo Hong; Sanghyun Park
>
> **摘要:** Video anomaly detection (VAD) is crucial for video analysis and surveillance in computer vision. However, existing VAD models rely on learned normal patterns, which makes them difficult to apply to diverse environments. Consequently, users should retrain models or develop separate AI models for new environments, which requires expertise in machine learning, high-performance hardware, and extensive data collection, limiting the practical usability of VAD. To address these challenges, this study proposes customizable video anomaly detection (C-VAD) technique and the AnyAnomaly model. C-VAD considers user-defined text as an abnormal event and detects frames containing a specified event in a video. We effectively implemented AnyAnomaly using a context-aware visual question answering without fine-tuning the large vision language model. To validate the effectiveness of the proposed model, we constructed C-VAD datasets and demonstrated the superiority of AnyAnomaly. Furthermore, our approach showed competitive results on VAD benchmarks, achieving state-of-the-art performance on UBnormal and UCF-Crime and surpassing other methods in generalization across all datasets. Our code is available online at github.com/SkiddieAhn/Paper-AnyAnomaly.
>
---
#### [replaced 041] Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12667v2](http://arxiv.org/pdf/2505.12667v2)**

> **作者:** Zihan Su; Xuerui Qiu; Hongbin Xu; Tangyu Jiang; Junhao Zhuang; Chun Yuan; Ming Li; Shengfeng He; Fei Richard Yu
>
> **备注:** Safa-Sora is accepted by NeurIPS 2025
>
> **摘要:** The explosive growth of generative video models has amplified the demand for reliable copyright preservation of AI-generated content. Despite its popularity in image synthesis, invisible generative watermarking remains largely underexplored in video generation. To address this gap, we propose Safe-Sora, the first framework to embed graphical watermarks directly into the video generation process. Motivated by the observation that watermarking performance is closely tied to the visual similarity between the watermark and cover content, we introduce a hierarchical coarse-to-fine adaptive matching mechanism. Specifically, the watermark image is divided into patches, each assigned to the most visually similar video frame, and further localized to the optimal spatial region for seamless embedding. To enable spatiotemporal fusion of watermark patches across video frames, we develop a 3D wavelet transform-enhanced Mamba architecture with a novel spatiotemporal local scanning strategy, effectively modeling long-range dependencies during watermark embedding and retrieval. To the best of our knowledge, this is the first attempt to apply state space models to watermarking, opening new avenues for efficient and robust watermark protection. Extensive experiments demonstrate that Safe-Sora achieves state-of-the-art performance in terms of video quality, watermark fidelity, and robustness, which is largely attributed to our proposals. Code is publicly available at https://github.com/Sugewud/Safe-Sora
>
---
#### [replaced 042] LRQ-DiT: Log-Rotation Post-Training Quantization of Diffusion Transformers for Image and Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03485v2](http://arxiv.org/pdf/2508.03485v2)**

> **作者:** Lianwei Yang; Haokun Lin; Tianchen Zhao; Yichen Wu; Hongyu Zhu; Ruiqi Xie; Zhenan Sun; Yu Wang; Qingyi Gu
>
> **摘要:** Diffusion Transformers (DiTs) have achieved impressive performance in text-to-image and text-to-video generation. However, their high computational cost and large parameter sizes pose significant challenges for usage in resource-constrained scenarios. Effective compression of models has become a crucial issue that urgently needs to be addressed. Post-training quantization (PTQ) is a promising solution to reduce memory usage and accelerate inference, but existing PTQ methods suffer from severe performance degradation under extreme low-bit settings. After experiments and analysis, we identify two key obstacles to low-bit PTQ for DiTs: (1) the weights of DiT models follow a Gaussian-like distribution with long tails, causing uniform quantization to poorly allocate intervals and leading to significant quantization errors. This issue has been observed in the linear layer weights of different DiT models, which deeply limits the performance. (2) two types of activation outliers in DiT models: (i) Mild Outliers with slightly elevated values, and (ii) Salient Outliers with large magnitudes concentrated in specific channels, which disrupt activation quantization. To address these issues, we propose LRQ-DiT, an efficient and accurate post-training quantization framework for image and video generation. First, we introduce Twin-Log Quantization (TLQ), a log-based method that allocates more quantization intervals to the intermediate dense regions, effectively achieving alignment with the weight distribution and reducing quantization errors. Second, we propose an Adaptive Rotation Scheme (ARS) that dynamically applies Hadamard or outlier-aware rotations based on activation fluctuation, effectively mitigating the impact of both types of outliers.Extensive experiments on various text-to-image and text-to-video DiT models demonstrate that LRQ-DiT preserves high generation quality.
>
---
#### [replaced 043] Contextual Gesture: Co-Speech Gesture Video Generation through Context-aware Gesture Representation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.07239v3](http://arxiv.org/pdf/2502.07239v3)**

> **作者:** Pinxin Liu; Pengfei Zhang; Hyeongwoo Kim; Pablo Garrido; Ari Shapiro; Kyle Olszewski
>
> **备注:** Accepted to ACM MM 2025. Project Page: https://andypinxinliu.github.io/Contextual-Gesture/
>
> **摘要:** Co-speech gesture generation is crucial for creating lifelike avatars and enhancing human-computer interactions by synchronizing gestures with speech. Despite recent advancements, existing methods struggle with accurately identifying the rhythmic or semantic triggers from audio for generating contextualized gesture patterns and achieving pixel-level realism. To address these challenges, we introduce Contextual Gesture, a framework that improves co-speech gesture video generation through three innovative components: (1) a chronological speech-gesture alignment that temporally connects two modalities, (2) a contextualized gesture tokenization that incorporate speech context into motion pattern representation through distillation, and (3) a structure-aware refinement module that employs edge connection to link gesture keypoints to improve video generation. Our extensive experiments demonstrate that Contextual Gesture not only produces realistic and speech-aligned gesture videos but also supports long-sequence generation and video gesture editing applications, shown in Fig.1.
>
---
#### [replaced 044] IMAIA: Interactive Maps AI Assistant for Travel Planning and Geo-Spatial Intelligence
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06993v2](http://arxiv.org/pdf/2507.06993v2)**

> **作者:** Jieren Deng; Zhizhang Hu; Ziyan He; Aleksandar Cvetkovic; Pak Kiu Chung; Dragomir Yankov; Chiqun Zhang
>
> **摘要:** Map applications are still largely point-and-click, making it difficult to ask map-centric questions or connect what a camera sees to the surrounding geospatial context with view-conditioned inputs. We introduce IMAIA, an interactive Maps AI Assistant that enables natural-language interaction with both vector (street) maps and satellite imagery, and augments camera inputs with geospatial intelligence to help users understand the world. IMAIA comprises two complementary components. Maps Plus treats the map as first-class context by parsing tiled vector/satellite views into a grid-aligned representation that a language model can query to resolve deictic references (e.g., ``the flower-shaped building next to the park in the top-right''). Places AI Smart Assistant (PAISA) performs camera-aware place understanding by fusing image--place embeddings with geospatial signals (location, heading, proximity) to ground a scene, surface salient attributes, and generate concise explanations. A lightweight multi-agent design keeps latency low and exposes interpretable intermediate decisions. Across map-centric QA and camera-to-place grounding tasks, IMAIA improves accuracy and responsiveness over strong baselines while remaining practical for user-facing deployments. By unifying language, maps, and geospatial cues, IMAIA moves beyond scripted tools toward conversational mapping that is both spatially grounded and broadly usable.
>
---
#### [replaced 045] A Multimodal and Multi-centric Head and Neck Cancer Dataset for Segmentation, Diagnosis and Outcome Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00367v3](http://arxiv.org/pdf/2509.00367v3)**

> **作者:** Numan Saeed; Salma Hassan; Shahad Hardan; Ahmed Aly; Darya Taratynova; Umair Nawaz; Ufaq Khan; Muhammad Ridzuan; Vincent Andrearczyk; Adrien Depeursinge; Yutong Xie; Thomas Eugene; Raphaël Metz; Mélanie Dore; Gregory Delpon; Vijay Ram Kumar Papineni; Kareem Wahid; Cem Dede; Alaa Mohamed Shawky Ali; Carlos Sjogreen; Mohamed Naser; Clifton D. Fuller; Valentin Oreiller; Mario Jreige; John O. Prior; Catherine Cheze Le Rest; Olena Tankyevych; Pierre Decazes; Su Ruan; Stephanie Tanadini-Lang; Martin Vallières; Hesham Elhalawani; Ronan Abgral; Romain Floch; Kevin Kerleguer; Ulrike Schick; Maelle Mauguen; David Bourhis; Jean-Christophe Leclere; Amandine Sambourg; Arman Rahmim; Mathieu Hatt; Mohammad Yaqub
>
> **备注:** 10 pages, 5 figures. Numan Saeed is the corresponding author. Numan Saeed, Salma Hassan and Shahad Hardan contributed equally to this work. Project page: https://hecktor25.grand-challenge.org/
>
> **摘要:** We present a publicly available multimodal dataset for head and neck cancer research, comprising 1123 annotated Positron Emission Tomography/Computed Tomography (PET/CT) studies from patients with histologically confirmed disease, acquired from 10 international medical centers. All studies contain co-registered PET/CT scans with varying acquisition protocols, reflecting real-world clinical diversity from a long-term, multi-institution retrospective collection. Primary gross tumor volumes (GTVp) and involved lymph nodes (GTVn) were manually segmented by experienced radiation oncologists and radiologists following established guidelines. We provide anonymized NifTi files, expert-annotated segmentation masks, comprehensive clinical metadata, and radiotherapy dose distributions for a patient subset. The metadata include TNM staging, HPV status, demographics, long-term follow-up outcomes, survival times, censoring indicators, and treatment information. To demonstrate its utility, we benchmark three key clinical tasks: automated tumor segmentation, recurrence-free survival prediction, and HPV status classification, using state-of-the-art deep learning models like UNet, SegResNet, and multimodal prognostic frameworks.
>
---
#### [replaced 046] Dynamic Classifier-Free Diffusion Guidance via Online Feedback
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.16131v2](http://arxiv.org/pdf/2509.16131v2)**

> **作者:** Pinelopi Papalampidi; Olivia Wiles; Ira Ktena; Aleksandar Shtedritski; Emanuele Bugliarello; Ivana Kajic; Isabela Albuquerque; Aida Nematzadeh
>
> **摘要:** Classifier-free guidance (CFG) is a cornerstone of text-to-image diffusion models, yet its effectiveness is limited by the use of static guidance scales. This "one-size-fits-all" approach fails to adapt to the diverse requirements of different prompts; moreover, prior solutions like gradient-based correction or fixed heuristic schedules introduce additional complexities and fail to generalize. In this work, we challeng this static paradigm by introducing a framework for dynamic CFG scheduling. Our method leverages online feedback from a suite of general-purpose and specialized small-scale latent-space evaluations, such as CLIP for alignment, a discriminator for fidelity and a human preference reward model, to assess generation quality at each step of the reverse diffusion process. Based on this feedback, we perform a greedy search to select the optimal CFG scale for each timestep, creating a unique guidance schedule tailored to every prompt and sample. We demonstrate the effectiveness of our approach on both small-scale models and the state-of-the-art Imagen 3, showing significant improvements in text alignment, visual quality, text rendering and numerical reasoning. Notably, when compared against the default Imagen 3 baseline, our method achieves up to 53.8% human preference win-rate for overall preference, a figure that increases up to to 55.5% on prompts targeting specific capabilities like text rendering. Our work establishes that the optimal guidance schedule is inherently dynamic and prompt-dependent, and provides an efficient and generalizable framework to achieve it.
>
---
#### [replaced 047] Subjective Camera 1.0: Bridging Human Cognition and Visual Reconstruction through Sequence-Aware Sketch-Guided Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23711v3](http://arxiv.org/pdf/2506.23711v3)**

> **作者:** Haoyang Chen; Dongfang Sun; Caoyuan Ma; Shiqin Wang; Kewei Zhang; Zheng Wang; Zhixiang Wang
>
> **摘要:** We introduce the concept of a subjective camera to reconstruct meaningful moments that physical cameras fail to capture. We propose Subjective Camera 1.0, a framework for reconstructing real-world scenes from readily accessible subjective readouts, i.e., textual descriptions and progressively drawn rough sketches. Built on optimization-based alignment of diffusion models, our approach avoids large-scale paired training data and mitigates generalization issues. To address the challenge of integrating multiple abstract concepts in real-world scenarios, we design a Sequence-Aware Sketch-Guided Diffusion framework with three loss terms for concept-wise sequential optimization, following the natural order of subjective readouts. Experiments on two datasets demonstrate that our method achieves state-of-the-art performance in image quality as well as spatial and semantic alignment with target scenes. User studies with 40 participants further confirm that our approach is consistently preferred. Our project page is at: subjective-camera.github.io
>
---
#### [replaced 048] FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.21059v3](http://arxiv.org/pdf/2502.21059v3)**

> **作者:** Ziyi Zhang; Zhen Sun; Zongmin Zhang; Jihui Guo; Xinlei He
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most MLLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, MLLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute the jailbreak attack on MLLMs. Our evaluations on Advbench show that FC-Attack attains an attack success rate of up to 96% via images and up to 78% via videos across multiple MLLMs. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. We also find that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop.
>
---
#### [replaced 049] A Unified Deep Learning Framework for Motion Correction in Medical Imaging
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.14204v3](http://arxiv.org/pdf/2409.14204v3)**

> **作者:** Jian Wang; Razieh Faghihpirayesh; Danny Joca; Polina Golland; Ali Gholipour
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Deep learning has shown significant value in image registration, however, current techniques are either limited by the type and range of motion they can handle, or require iterative inference and/or retraining for new imaging data. To address these limitations, we introduce UniMo, a Unified Motion Correction framework that leverages deep neural networks to correct diverse motion in medical imaging. UniMo employs an alternating optimization scheme for a unified loss function to train an integrated model of 1) an equivariant neural network for global rigid motion correction and 2) an encoder-decoder network for local deformations. It features a geometric deformation augmenter that 1) enhances the robustness of global correction by addressing local deformations from non-rigid motion or geometric distortions, and 2) generates augmented data to improve training. UniMo is a hybrid model that uses both image intensities and shapes to achieve robust performance amid appearance variations, and therefore generalizes to multiple imaging modalities without retraining. We trained and tested UniMo to track motion in fetal magnetic resonance imaging, a challenging application due to 1) both large rigid and non-rigid motion, and 2) wide variations in image appearance. We then evaluated the trained model, without retraining, on MedMNIST, lung CT, and BraTS datasets. Results show that UniMo surpassed existing motion correction methods in accuracy, and notably enabled one-time training on a single modality while maintaining high stability and adaptability across unseen datasets. By offering a unified solution to motion correction, UniMo marks a significant advance in medical imaging, especially in applications with combined bulk and local motion. The code is available at: https://github.com/IntelligentImaging/UNIMO
>
---
#### [replaced 050] KNN-MMD: Cross Domain Wireless Sensing via Local Distribution Alignment
- **分类: cs.CV; cs.AI; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.04783v4](http://arxiv.org/pdf/2412.04783v4)**

> **作者:** Zijian Zhao; Zhijie Cai; Tingwei Chen; Xiaoyang Li; Hang Li; Qimei Chen; Guangxu Zhu
>
> **摘要:** Wireless sensing has recently found widespread applications in diverse environments, including homes, offices, and public spaces. By analyzing patterns in channel state information (CSI), it is possible to infer human actions for tasks such as person identification, gesture recognition, and fall detection. However, CSI is highly sensitive to environmental changes, where even minor alterations can significantly distort the CSI patterns. This sensitivity often leads to performance degradation or outright failure when applying wireless sensing models trained in one environment to another. To address this challenge, Domain Alignment (DAL) has been widely adopted for cross-domain classification tasks, as it focuses on aligning the global distributions of the source and target domains in feature space. Despite its popularity, DAL often neglects inter-category relationships, which can lead to misalignment between categories across domains, even when global alignment is achieved. To overcome these limitations, we propose K-Nearest Neighbors Maximum Mean Discrepancy (KNN-MMD), a novel few-shot method for cross-domain wireless sensing. Our approach begins by constructing a help set using KNN from the target domain, enabling local alignment between the source and target domains within each category using MMD. Additionally, we address a key instability issue commonly observed in cross-domain methods, where model performance fluctuates sharply between epochs. Further, most existing methods struggle to determine an optimal stopping point during training due to the absence of labeled data from the target domain. Our method resolves this by excluding the support set from the target domain during training and employing it as a validation set to determine the stopping criterion.The dataset and code are publicly available at https://github.com/RS2002/KNN-MMD .
>
---
#### [replaced 051] The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.12594v2](http://arxiv.org/pdf/2509.12594v2)**

> **作者:** Titong Jiang; Xuefeng Jiang; Yuan Ma; Xin Wen; Bailin Li; Kun Zhan; Peng Jia; Yahui Liu; Sheng Sun; Xianpeng Lang
>
> **备注:** Under review. Project site: https://liauto-research.github.io/LightVLA
>
> **摘要:** We present LightVLA, a simple yet effective differentiable token pruning framework for vision-language-action (VLA) models. While VLA models have shown impressive capability in executing real-world robotic tasks, their deployment on resource-constrained platforms is often bottlenecked by the heavy attention-based computation over large sets of visual tokens. LightVLA addresses this challenge through adaptive, performance-driven pruning of visual tokens: It generates dynamic queries to evaluate visual token importance, and adopts Gumbel softmax to enable differentiable token selection. Through fine-tuning, LightVLA learns to preserve the most informative visual tokens while pruning tokens which do not contribute to task execution, thereby improving efficiency and performance simultaneously. Notably, LightVLA requires no heuristic magic numbers and introduces no additional trainable parameters, making it compatible with modern inference frameworks. Experimental results demonstrate that LightVLA outperforms different VLA models and existing token pruning methods across diverse tasks on the LIBERO benchmark, achieving higher success rates with substantially reduced computational overhead. Specifically, LightVLA reduces FLOPs and latency by 59.1% and 38.2% respectively, with a 2.6% improvement in task success rate. Meanwhile, we also investigate the learnable query-based token pruning method LightVLA* with additional trainable parameters, which also achieves satisfactory performance. Our work reveals that as VLA pursues optimal performance, LightVLA spontaneously learns to prune tokens from a performance-driven perspective. To the best of our knowledge, LightVLA is the first work to apply adaptive visual token pruning to VLA tasks with the collateral goals of efficiency and performance, marking a significant step toward more efficient, powerful and practical real-time robotic systems.
>
---
#### [replaced 052] A Semantic Segmentation Algorithm for Pleural Effusion Based on DBIF-AUNet
- **分类: cs.CV; 68T45, 92C55; I.4.6; I.5.4; J.3**

- **链接: [http://arxiv.org/pdf/2508.06191v2](http://arxiv.org/pdf/2508.06191v2)**

> **作者:** Ruixiang Tang; Mingda Zhang; Jianglong Qin; Yan Song; Yi Wu; Wei Wu
>
> **备注:** 12 pages, 6 figures, 2 tables
>
> **摘要:** Pleural effusion semantic segmentation can significantly enhance the accuracy and timeliness of clinical diagnosis and treatment by precisely identifying disease severity and lesion areas. Currently, semantic segmentation of pleural effusion CT images faces multiple challenges. These include similar gray levels between effusion and surrounding tissues, blurred edges, and variable morphology. Existing methods often struggle with diverse image variations and complex edges, primarily because direct feature concatenation causes semantic gaps. To address these challenges, we propose the Dual-Branch Interactive Fusion Attention model (DBIF-AUNet). This model constructs a densely nested skip-connection network and innovatively refines the Dual-Domain Feature Disentanglement module (DDFD). The DDFD module orthogonally decouples the functions of dual-domain modules to achieve multi-scale feature complementarity and enhance characteristics at different levels. Concurrently, we design a Branch Interaction Attention Fusion module (BIAF) that works synergistically with the DDFD. This module dynamically weights and fuses global, local, and frequency band features, thereby improving segmentation robustness. Furthermore, we implement a nested deep supervision mechanism with hierarchical adaptive hybrid loss to effectively address class imbalance. Through validation on 1,622 pleural effusion CT images from Southwest Hospital, DBIF-AUNet achieved IoU and Dice scores of 80.1% and 89.0% respectively. These results outperform state-of-the-art medical image segmentation models U-Net++ and Swin-UNet by 5.7%/2.7% and 2.2%/1.5% respectively, demonstrating significant optimization in segmentation accuracy for complex pleural effusion CT images.
>
---
#### [replaced 053] TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17098v2](http://arxiv.org/pdf/2505.17098v2)**

> **作者:** Yanshu Li; Jianjiang Yang; Tian Yun; Pinyuan Feng; Jinfa Huang; Ruixiang Tang
>
> **备注:** EMNLP2025 Main, 28 pages, 11 figures, 19 tables
>
> **摘要:** Multimodal in-context learning (ICL) has emerged as a key mechanism for harnessing the capabilities of large vision-language models (LVLMs). However, its effectiveness remains highly sensitive to the quality of input ICL sequences, particularly for tasks involving complex reasoning or open-ended generation. A major limitation is our limited understanding of how LVLMs actually exploit these sequences during inference. To bridge this gap, we systematically interpret multimodal ICL through the lens of task mapping, which reveals how local and global relationships within and among demonstrations guide model reasoning. Building on this insight, we present TACO, a lightweight transformer-based model equipped with task-aware attention that dynamically configures ICL sequences. By injecting task-mapping signals into the autoregressive decoding process, TACO creates a bidirectional synergy between sequence construction and task reasoning. Experiments on five LVLMs and nine datasets demonstrate that TACO consistently surpasses baselines across diverse ICL tasks. These results position task mapping as a novel and valuable perspective for interpreting and improving multimodal ICL.
>
---
#### [replaced 054] AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11003v2](http://arxiv.org/pdf/2509.11003v2)**

> **作者:** Gurutva Patle; Nilay Girgaonkar; Nagabhushan Somraj; Rajiv Soundararajan
>
> **备注:** SIGGRAPH Asia 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has shown impressive results in real-time novel view synthesis. However, it often struggles under sparse-view settings, producing undesirable artifacts such as floaters, inaccurate geometry, and overfitting due to limited observations. We find that a key contributing factor is uncontrolled densification, where adding Gaussian primitives rapidly without guidance can harm geometry and cause artifacts. We propose AD-GS, a novel alternating densification framework that interleaves high and low densification phases. During high densification, the model densifies aggressively, followed by photometric loss based training to capture fine-grained scene details. Low densification then primarily involves aggressive opacity pruning of Gaussians followed by regularizing their geometry through pseudo-view consistency and edge-aware depth smoothness. This alternating approach helps reduce overfitting by carefully controlling model capacity growth while progressively refining the scene representation. Extensive experiments on challenging datasets demonstrate that AD-GS significantly improves rendering quality and geometric consistency compared to existing methods. The source code for our model can be found on our project page: https://gurutvapatle.github.io/publications/2025/ADGS.html .
>
---
#### [replaced 055] LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04653v2](http://arxiv.org/pdf/2504.04653v2)**

> **作者:** Yimu Wang; Mozhgan Nasr Azadani; Sean Sedwards; Krzysztof Czarnecki
>
> **备注:** To appear at EMNLP 2025
>
> **摘要:** Redundancy of visual tokens in multi-modal large language models (MLLMs) significantly reduces their computational efficiency. Recent approaches, such as resamplers and summarizers, have sought to reduce the number of visual tokens, but at the cost of visual reasoning ability. To address this, we propose LEO-MINI, a novel MLLM that significantly reduces the number of visual tokens and simultaneously boosts visual reasoning capabilities. For efficiency, LEO-MINI incorporates CoTR, a novel token reduction module to consolidate a large number of visual tokens into a smaller set of tokens, using the similarity between visual tokens, text tokens, and a compact learnable query. For effectiveness, to scale up the model's ability with minimal computational overhead, LEO-MINI employs MMoE, a novel mixture of multi-modal experts module. MMOE employs a set of LoRA experts with a novel router to switch between them based on the input text and visual tokens instead of only using the input hidden state. MMoE also includes a general LoRA expert that is always activated to learn general knowledge for LLM reasoning. For extracting richer visual features, MMOE employs a set of vision experts trained on diverse domain-specific data. To demonstrate LEO-MINI's improved efficiency and performance, we evaluate it against existing efficient MLLMs on various benchmark vision-language tasks.
>
---
#### [replaced 056] ZoDIAC: Zoneout Dropout Injection Attention Calculation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2206.14263v2](http://arxiv.org/pdf/2206.14263v2)**

> **作者:** Zanyar Zohourianshahzadi; Jugal Kalita
>
> **备注:** This work has been published in IEEE AIxSET 2024 and is available conference proceedings
>
> **摘要:** In the past few years the transformer model has been utilized for a variety of tasks such as image captioning, image classification natural language generation, and natural language understanding. As a key component of the transformer model, self-attention calculates the attention values by mapping the relationships among the head elements of the source and target sequence, yet there is no explicit mechanism to refine and intensify the attention values with respect to the context of the input and target sequences. Based on this intuition, we introduce a novel refine and intensify attention mechanism that is called Zoneup Dropout Injection Attention Calculation (ZoDIAC), in which the intensities of attention values in the elements of the input source and target sequences are first refined using GELU and dropout and then intensified using a proposed zoneup process which includes the injection of a learned scalar factor. Our extensive experiments show that ZoDIAC achieves statistically significant higher scores under all image captioning metrics using various feature extractors in comparison to the conventional self-attention module in the transformer model on the MS-COCO dataset. Our proposed ZoDIAC attention modules can be used as a drop-in replacement for the attention components in all transformer models. The code for our experiments is publicly available at: https://github.com/zanyarz/zodiac
>
---
#### [replaced 057] Optimal Transport for Rectified Flow Image Editing: Unifying Inversion-Based and Direct Methods
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02363v2](http://arxiv.org/pdf/2508.02363v2)**

> **作者:** Marian Lupascu; Mihai-Sorin Stupariu
>
> **备注:** 27 pages, 26 figures, WACV conference
>
> **摘要:** Image editing in rectified flow models remains challenging due to the fundamental trade-off between reconstruction fidelity and editing flexibility. While inversion-based methods suffer from trajectory deviation, recent inversion-free approaches like FlowEdit offer direct editing pathways but can benefit from additional guidance to improve structure preservation. In this work, we demonstrate that optimal transport theory provides a unified framework for improving both paradigms in rectified flow editing. We introduce a zero-shot transport-guided inversion framework that leverages optimal transport during the reverse diffusion process, and extend optimal transport principles to enhance inversion-free methods through transport-optimized velocity field corrections. Incorporating transport-based guidance can effectively balance reconstruction accuracy and editing controllability across different rectified flow editing approaches. For inversion-based editing, our method achieves high-fidelity reconstruction with LPIPS scores of 0.001 and SSIM of 0.992 on face editing benchmarks, observing 7.8% to 12.9% improvements over RF-Inversion on LSUN datasets. For inversion-free editing with FlowEdit on FLUX and Stable Diffusion 3, we demonstrate consistent improvements in semantic consistency and structure preservation across diverse editing scenarios. Our semantic face editing experiments show an 11.2% improvement in identity preservation and enhanced perceptual quality. The unified optimal transport framework produces visually compelling edits with superior detail preservation across both inversion-based and direct editing paradigms. Code is available for RF-Inversion and FlowEdit at: https://github.com/marianlupascu/OT-RF
>
---
#### [replaced 058] How Much Do Large Language Models Know about Human Motion? A Case Study in 3D Avatar Control
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.21531v2](http://arxiv.org/pdf/2505.21531v2)**

> **作者:** Kunhang Li; Jason Naradowsky; Yansong Feng; Yusuke Miyao
>
> **摘要:** We explore the human motion knowledge of Large Language Models (LLMs) through 3D avatar control. Given a motion instruction, we prompt LLMs to first generate a high-level movement plan with consecutive steps (High-level Planning), then specify body part positions in each step (Low-level Planning), which we linearly interpolate into avatar animations. Using 20 representative motion instructions that cover fundamental movements and balance body part usage, we conduct comprehensive evaluations, including human and automatic scoring of both high-level movement plans and generated animations, as well as automatic comparison with oracle positions in low-level planning. Our findings show that LLMs are strong at interpreting high-level body movements but struggle with precise body part positioning. While decomposing motion queries into atomic components improves planning, LLMs face challenges in multi-step movements involving high-degree-of-freedom body parts. Furthermore, LLMs provide reasonable approximations for general spatial descriptions, but fall short in handling precise spatial specifications. Notably, LLMs demonstrate promise in conceptualizing creative motions and distinguishing culturally specific motion patterns.
>
---
#### [replaced 059] GeoSplat: A Deep Dive into Geometry-Constrained Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.05075v2](http://arxiv.org/pdf/2509.05075v2)**

> **作者:** Yangming Li; Chaoyu Liu; Lihao Liu; Simon Masnou; Carola-Bibiane Schönlieb
>
> **摘要:** A few recent works explored incorporating geometric priors to regularize the optimization of Gaussian splatting, further improving its performance. However, those early studies mainly focused on the use of low-order geometric priors (e.g., normal vector), and they are also unreliably estimated by noise-sensitive methods, like local principal component analysis. To address their limitations, we first present GeoSplat, a general geometry-constrained optimization framework that exploits both first-order and second-order geometric quantities to improve the entire training pipeline of Gaussian splatting, including Gaussian initialization, gradient update, and densification. As an example, we initialize the scales of 3D Gaussian primitives in terms of principal curvatures, leading to a better coverage of the object surface than random initialization. Secondly, based on certain geometric structures (e.g., local manifold), we introduce efficient and noise-robust estimation methods that provide dynamic geometric priors for our framework. We conduct extensive experiments on multiple datasets for novel view synthesis, showing that our framework: GeoSplat, significantly improves the performance of Gaussian splatting and outperforms previous baselines.
>
---
#### [replaced 060] Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13146v3](http://arxiv.org/pdf/2502.13146v3)**

> **作者:** Shuo Xing; Peiran Li; Yuping Wang; Ruizheng Bai; Yueqi Wang; Chan-Wei Hu; Chengxuan Qian; Huaxiu Yao; Zhengzhong Tu
>
> **备注:** Published at EMNLP 2025
>
> **摘要:** The emergence of large Vision Language Models (VLMs) has broadened the scope and capabilities of single-modal Large Language Models (LLMs) by integrating visual modalities, thereby unlocking transformative cross-modal applications in a variety of real-world scenarios. Despite their impressive performance, VLMs are prone to significant hallucinations, particularly in the form of cross-modal inconsistencies. Building on the success of Reinforcement Learning from Human Feedback (RLHF) in aligning LLMs, recent advancements have focused on applying direct preference optimization (DPO) on carefully curated datasets to mitigate these issues. Yet, such approaches typically introduce preference signals in a brute-force manner, neglecting the crucial role of visual information in the alignment process. In this paper, we introduce Re-Align, a novel alignment framework that leverages image retrieval to construct a dual-preference dataset, effectively incorporating both textual and visual preference signals. We further introduce rDPO, an extension of the standard direct preference optimization that incorporates an additional visual preference objective during fine-tuning. Our experimental results demonstrate that Re-Align not only mitigates hallucinations more effectively than previous methods but also yields significant performance gains in general visual question-answering (VQA) tasks. Moreover, we show that Re-Align maintains robustness and scalability across a wide range of VLM sizes and architectures. This work represents a significant step forward in aligning multimodal LLMs, paving the way for more reliable and effective cross-modal applications. We release all the code in https://github.com/taco-group/Re-Align.
>
---
#### [replaced 061] DynSTG-Mamba: Dynamic Spatio-Temporal Graph Mamba with Cross-Graph Knowledge Distillation for Gait Disorders Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13156v2](http://arxiv.org/pdf/2503.13156v2)**

> **作者:** Zakariae Zrimek; Youssef Mourchid; Mohammed El Hassouni
>
> **备注:** After receiving detailed feedback from journalreviewers, we identified limitations in the initial approach and substantially improved the methodology and contributions of the work. To ensure clarity and avoid confusion between the initial and revised versions, we are withdrawing this submission. A new version reflecting these improvements will be submitted and made available on arXiv shortly
>
> **摘要:** Gait disorder recognition plays a crucial role in the early diagnosis and monitoring of movement disorders. Existing approaches, including spatio-temporal graph convolutional networks (ST-GCNs), often face high memory demands and struggle to capture complex spatio-temporal dependencies, limiting their efficiency in clinical applications. To address these challenges, we introduce DynSTG-Mamba (Dynamic Spatio-Temporal Graph Mamba), a novel framework that combines DF-STGNN and STG-Mamba to enhance motion sequence modeling. The DF-STGNN incorporates a dynamic spatio-temporal filter that adaptively adjusts spatial connections between skeletal joints and temporal interactions across different movement phases. This approach ensures better feature propagation through dynamic graph structures by considering the hierarchical nature and dynamics of skeletal gait data. Meanwhile, STG-Mamba, an extension of Mamba adapted for skeletal motion data, ensures a continuous propagation of states, facilitating the capture of long-term dependencies while reducing computational complexity. To reduce the number of model parameters and computational costs while maintaining consistency, we propose Cross-Graph Relational Knowledge Distillation, a novel knowledge transfer mechanism that aligns relational information between teacher (large architecture) and student models (small architecture) while using shared memory. This ensures that the interactions and movement patterns of the joints are accurately preserved in the motion sequences. We validate our DynSTG-Mamba on KOA-NM, PD-WALK, and ATAXIA datasets, where it outperforms state-of-the-art approaches by achieving in terms of Accuracy, F1-score, and Recall. Our results highlight the efficiency and robustness of our approach, offering a lightweight yet highly accurate solution for automated gait analysis and movement disorder assessment.
>
---
#### [replaced 062] Wavelet-Space Representations for Neural Super-Resolution in Rendering Pipelines
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.16024v3](http://arxiv.org/pdf/2508.16024v3)**

> **作者:** Prateek Poudel; Prashant Aryal; Kirtan Kunwar; Navin Nepal; Dinesh Baniya Kshatri
>
> **摘要:** We investigate the use of wavelet-space feature decomposition in neural super-resolution for rendering pipelines. Building on recent neural upscaling frameworks, we introduce a formulation that predicts stationary wavelet coefficients rather than directly regressing RGB values. This frequency-aware decomposition separates low- and high-frequency components, enabling sharper texture recovery and reducing blur in challenging regions. Unlike conventional wavelet transforms, our use of the stationary wavelet transform (SWT) preserves spatial alignment across subbands, allowing the network to integrate G-buffer attributes and temporally warped history frames in a shift-invariant manner. The predicted coefficients are recombined through inverse wavelet synthesis, producing resolution-consistent reconstructions across arbitrary scale factors. We conduct extensive evaluations and ablations, showing that incorporating SWT improves both fidelity and perceptual quality with only modest overhead, while remaining compatible with standard rendering architectures. Taken together, our results suggest that wavelet-domain neural super-resolution provides a principled and efficient path toward higher-quality real-time rendering, with broader implications for neural rendering and graphics applications.
>
---
#### [replaced 063] EmoGist: Efficient In-Context Learning for Visual Emotion Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14660v2](http://arxiv.org/pdf/2505.14660v2)**

> **作者:** Ronald Seoh; Dan Goldwasser
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** In this paper, we introduce EmoGist, a training-free, in-context learning method for performing visual emotion classification with LVLMs. The key intuition of our approach is that context-dependent definition of emotion labels could allow more accurate predictions of emotions, as the ways in which emotions manifest within images are highly context dependent and nuanced. EmoGist pre-generates multiple descriptions of emotion labels, by analyzing the clusters of example images belonging to each label. At test time, we retrieve a version of description based on the cosine similarity of test image to cluster centroids, and feed it together with the test image to a fast LVLM for classification. Through our experiments, we show that EmoGist allows up to 12 points improvement in micro F1 scores with the multi-label Memotion dataset, and up to 8 points in macro F1 in the multi-class FI dataset.
>
---
#### [replaced 064] SINF: Semantic Neural Network Inference with Semantic Subgraphs
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.01259v3](http://arxiv.org/pdf/2310.01259v3)**

> **作者:** A. Q. M. Sazzad Sayyed; Francesco Restuccia
>
> **备注:** 12 pages, 13 figures, conference format
>
> **摘要:** This paper proposes Semantic Inference (SINF) that creates semantic subgraphs in a Deep Neural Network(DNN) based on a new Discriminative Capability Score (DCS) to drastically reduce the DNN computational load with limited performance loss.~We evaluate the performance SINF on VGG16, VGG19, and ResNet50 DNNs trained on CIFAR100 and a subset of the ImageNet dataset. Moreover, we compare its performance against 6 state-of-the-art pruning approaches. Our results show that (i) on average, SINF reduces the inference time of VGG16, VGG19, and ResNet50 respectively by up to 29%, 35%, and 15% with only 3.75%, 0.17%, and 6.75% accuracy loss for CIFAR100 while for ImageNet benchmark, the reduction in inference time is 18%, 22%, and 9% for accuracy drop of 3%, 2.5%, and 6%; (ii) DCS achieves respectively up to 3.65%, 4.25%, and 2.36% better accuracy with VGG16, VGG19, and ResNet50 with respect to existing discriminative scores for CIFAR100 and the same for ImageNet is 8.9%, 5.8%, and 5.2% respectively. Through experimental evaluation on Raspberry Pi and NVIDIA Jetson Nano, we show SINF is about 51% and 38% more energy efficient and takes about 25% and 17% less inference time than the base model for CIFAR100 and ImageNet.
>
---
#### [replaced 065] DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration with Dynamic Blur-Level Mapping and Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13797v2](http://arxiv.org/pdf/2507.13797v2)**

> **作者:** Huu-Phu Do; Yu-Wei Chen; Yi-Cheng Liao; Chi-Wei Hsiao; Han-Yang Wang; Wei-Chen Chiu; Ching-Chun Huang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Blind Face Restoration aims to recover high-fidelity, detail-rich facial images from unknown degraded inputs, presenting significant challenges in preserving both identity and detail. Pre-trained diffusion models have been increasingly used as image priors to generate fine details. Still, existing methods often use fixed diffusion sampling timesteps and a global guidance scale, assuming uniform degradation. This limitation and potentially imperfect degradation kernel estimation frequently lead to under- or over-diffusion, resulting in an imbalance between fidelity and quality. We propose DynFaceRestore, a novel blind face restoration approach that learns to map any blindly degraded input to Gaussian blurry images. By leveraging these blurry images and their respective Gaussian kernels, we dynamically select the starting timesteps for each blurry image and apply closed-form guidance during the diffusion sampling process to maintain fidelity. Additionally, we introduce a dynamic guidance scaling adjuster that modulates the guidance strength across local regions, enhancing detail generation in complex areas while preserving structural fidelity in contours. This strategy effectively balances the trade-off between fidelity and quality. DynFaceRestore achieves state-of-the-art performance in both quantitative and qualitative evaluations, demonstrating robustness and effectiveness in blind face restoration. Project page at https://nycu-acm.github.io/DynFaceRestore/
>
---
#### [replaced 066] MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.15548v2](http://arxiv.org/pdf/2509.15548v2)**

> **作者:** Deming Li; Kaiwen Jiang; Yutao Tang; Ravi Ramamoorthi; Rama Chellappa; Cheng Peng
>
> **摘要:** In-the-wild photo collections often contain limited volumes of imagery and exhibit multiple appearances, e.g., taken at different times of day or seasons, posing significant challenges to scene reconstruction and novel view synthesis. Although recent adaptations of Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have improved in these areas, they tend to oversmooth and are prone to overfitting. In this paper, we present MS-GS, a novel framework designed with Multi-appearance capabilities in Sparse-view scenarios using 3DGS. To address the lack of support due to sparse initializations, our approach is built on the geometric priors elicited from monocular depth estimations. The key lies in extracting and utilizing local semantic regions with a Structure-from-Motion (SfM) points anchored algorithm for reliable alignment and geometry cues. Then, to introduce multi-view constraints, we propose a series of geometry-guided supervision at virtual views in a fine-grained and coarse scheme to encourage 3D consistency and reduce overfitting. We also introduce a dataset and an in-the-wild experiment setting to set up more realistic benchmarks. We demonstrate that MS-GS achieves photorealistic renderings under various challenging sparse-view and multi-appearance conditions and outperforms existing approaches significantly across different datasets.
>
---
#### [replaced 067] IPGPhormer: Interpretable Pathology Graph-Transformer for Survival Analysis
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.12381v2](http://arxiv.org/pdf/2508.12381v2)**

> **作者:** Guo Tang; Songhan Jiang; Jinpeng Lu; Linghan Cai; Yongbing Zhang
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Pathological images play an essential role in cancer prognosis, while survival analysis, which integrates computational techniques, can predict critical clinical events such as patient mortality or disease recurrence from whole-slide images (WSIs). Recent advancements in multiple instance learning have significantly improved the efficiency of survival analysis. However, existing methods often struggle to balance the modeling of long-range spatial relationships with local contextual dependencies and typically lack inherent interpretability, limiting their clinical utility. To address these challenges, we propose the Interpretable Pathology Graph-Transformer (IPGPhormer), a novel framework that captures the characteristics of the tumor microenvironment and models their spatial dependencies across the tissue. IPGPhormer uniquely provides interpretability at both tissue and cellular levels without requiring post-hoc manual annotations, enabling detailed analyses of individual WSIs and cross-cohort assessments. Comprehensive evaluations on four public benchmark datasets demonstrate that IPGPhormer outperforms state-of-the-art methods in both predictive accuracy and interpretability. In summary, our method, IPGPhormer, offers a promising tool for cancer prognosis assessment, paving the way for more reliable and interpretable decision-support systems in pathology. The code is publicly available at https://anonymous.4open.science/r/IPGPhormer-6EEB.
>
---
#### [replaced 068] DD-Ranking: Rethinking the Evaluation of Dataset Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13300v3](http://arxiv.org/pdf/2505.13300v3)**

> **作者:** Zekai Li; Xinhao Zhong; Samir Khaki; Zhiyuan Liang; Yuhao Zhou; Mingjia Shi; Ziqiao Wang; Xuanlei Zhao; Wangbo Zhao; Ziheng Qin; Mengxuan Wu; Pengfei Zhou; Haonan Wang; David Junhao Zhang; Jia-Wei Liu; Shaobo Wang; Dai Liu; Linfeng Zhang; Guang Li; Kun Wang; Zheng Zhu; Zhiheng Ma; Joey Tianyi Zhou; Jiancheng Lv; Yaochu Jin; Peihao Wang; Kaipeng Zhang; Lingjuan Lyu; Yiran Huang; Zeynep Akata; Zhiwei Deng; Xindi Wu; George Cazenavette; Yuzhang Shang; Justin Cui; Jindong Gu; Qian Zheng; Hao Ye; Shuo Wang; Xiaobo Wang; Yan Yan; Angela Yao; Mike Zheng Shou; Tianlong Chen; Hakan Bilen; Baharan Mirzasoleiman; Manolis Kellis; Konstantinos N. Plataniotis; Zhangyang Wang; Bo Zhao; Yang You; Kai Wang
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** In recent years, dataset distillation has provided a reliable solution for data compression, where models trained on the resulting smaller synthetic datasets achieve performance comparable to those trained on the original datasets. To further improve the performance of synthetic datasets, various training pipelines and optimization objectives have been proposed, greatly advancing the field of dataset distillation. Recent decoupled dataset distillation methods introduce soft labels and stronger data augmentation during the post-evaluation phase and scale dataset distillation up to larger datasets (e.g., ImageNet-1K). However, this raises a question: Is accuracy still a reliable metric to fairly evaluate dataset distillation methods? Our empirical findings suggest that the performance improvements of these methods often stem from additional techniques rather than the inherent quality of the images themselves, with even randomly sampled images achieving superior results. Such misaligned evaluation settings severely hinder the development of DD. Therefore, we propose DD-Ranking, a unified evaluation framework, along with new general evaluation metrics to uncover the true performance improvements achieved by different methods. By refocusing on the actual information enhancement of distilled datasets, DD-Ranking provides a more comprehensive and fair evaluation standard for future research advancements.
>
---
#### [replaced 069] TennisTV: Do Multimodal Large Language Models Understand Tennis Rallies?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.15602v2](http://arxiv.org/pdf/2509.15602v2)**

> **作者:** Zhongyuan Bao; Lejun Zhang
>
> **摘要:** Multimodal large language models (MLLMs) excel at general video understanding but struggle with fast, high-frequency sports like tennis, where rally clips are short yet information-dense. To systematically evaluate MLLMs in this challenging domain, we present TennisTV, the first and most comprehensive benchmark for tennis video understanding. TennisTV models each rally as a temporal-ordered sequence of consecutive stroke events, using automated pipelines for filtering and question generation. It covers 9 tasks from the stroke level to the rally level and includes 2943 human-verified questions. Evaluating 17 representative MLLMs, we provide the first systematic assessment of tennis video understanding. Results reveal substantial shortcomings and yield two key insights: (i) frame-sampling density should be tailored and balanced across tasks, and (ii) improving temporal grounding is essential for stronger reasoning.
>
---
#### [replaced 070] DeepInsert: Early Layer Bypass for Efficient and Performant Multimodal Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.19327v2](http://arxiv.org/pdf/2504.19327v2)**

> **作者:** Moulik Choraria; Xinbo Wu; Akhil Bhimaraju; Nitesh Sekhar; Yue Wu; Xu Zhang; Prateek Singhal; Lav R. Varshney
>
> **摘要:** The hyperscaling of data and parameter count in transformer models is yielding diminishing performance improvement, especially when weighed against training costs. Such plateauing underlines a growing need for more efficient finetuning and inference, without sacrificing performance. This is particularly pressing for multimodal learning, where the overhead of processing multimodal tokens alongside language data often limits the practical viability of these systems. In parallel, advances in representation learning and interpretability have deepened our understanding of how such models process and encode information. Notably, recent work has uncovered implicit cross-modal alignment in the deeper layers of large pretrained models. Interestingly, this aligns with our own observations that models naturally defer most cross-modal token interactions to deeper stages of computation. Building on this, we propose a simple modification. Instead of concatenation with the language prompt at the start, we insert multimodal tokens directly into the middle, allowing them to entirely bypass the early layers. Our results with diverse modalities: 1) LLaVA \& BLIP for vision, 2) LTU for audio, and 3) MoLCA for molecular data, indicate that our method reduces computational costs during both training and inference, while at the very least, preserving, if not surpassing the performance of existing baselines. Our work has important implications for scaling and composing pretrained models in a resource-efficient manner.
>
---
#### [replaced 071] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.12720v3](http://arxiv.org/pdf/2508.12720v3)**

> **作者:** Kangjie Chen; Yingji Zhong; Zhihao Li; Jiaqi Lin; Youyu Chen; Minghan Qin; Haoqian Wang
>
> **备注:** Accepted by NeurIPS 2025. Project page: https://chenkangjie1123.github.io/Co-Adaptation-3DGS/, Code at: https://github.com/chenkangjie1123/Co-Adaptation-of-3DGS
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated impressive performance in novel view synthesis under dense-view settings. However, in sparse-view scenarios, despite the realistic renderings in training views, 3DGS occasionally manifests appearance artifacts in novel views. This paper investigates the appearance artifacts in sparse-view 3DGS and uncovers a core limitation of current approaches: the optimized Gaussians are overly-entangled with one another to aggressively fit the training views, which leads to a neglect of the real appearance distribution of the underlying scene and results in appearance artifacts in novel views. The analysis is based on a proposed metric, termed Co-Adaptation Score (CA), which quantifies the entanglement among Gaussians, i.e., co-adaptation, by computing the pixel-wise variance across multiple renderings of the same viewpoint, with different random subsets of Gaussians. The analysis reveals that the degree of co-adaptation is naturally alleviated as the number of training views increases. Based on the analysis, we propose two lightweight strategies to explicitly mitigate the co-adaptation in sparse-view 3DGS: (1) random gaussian dropout; (2) multiplicative noise injection to the opacity. Both strategies are designed to be plug-and-play, and their effectiveness is validated across various methods and benchmarks. We hope that our insights into the co-adaptation effect will inspire the community to achieve a more comprehensive understanding of sparse-view 3DGS.
>
---
#### [replaced 072] Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03262v3](http://arxiv.org/pdf/2503.03262v3)**

> **作者:** Nadya Abdel Madjid; Abdulrahman Ahmad; Murad Mebrahtu; Yousef Babaa; Abdelmoamen Nasser; Sumbal Malik; Bilal Hassan; Naoufel Werghi; Jorge Dias; Majid Khonji
>
> **摘要:** As the potential for autonomous vehicles to be integrated on a large scale into modern traffic systems continues to grow, ensuring safe navigation in dynamic environments is crucial for smooth integration. To guarantee safety and prevent collisions, autonomous vehicles must be capable of accurately predicting the trajectories of surrounding traffic agents. Over the past decade, significant efforts from both academia and industry have been dedicated to designing solutions for precise trajectory forecasting. These efforts have produced a diverse range of approaches, raising questions about the differences between these methods and whether trajectory prediction challenges have been fully addressed. This paper reviews a substantial portion of recent trajectory prediction methods proposing a taxonomy to classify existing solutions. A general overview of the prediction pipeline is also provided, covering input and output modalities, modeling features, and prediction paradigms existing in the literature. In addition, the paper discusses active research areas within trajectory prediction, addresses the posed research questions, and highlights the remaining research gaps and challenges.
>
---
#### [replaced 073] EarthGPT-X: A Spatial MLLM for Multi-level Multi-Source Remote Sensing Imagery Understanding with Visual Prompting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12795v2](http://arxiv.org/pdf/2504.12795v2)**

> **作者:** Wei Zhang; Miaoxin Cai; Yaqian Ning; Tong Zhang; Yin Zhuang; Shijian Lu; He Chen; Jun Li; Xuerui Mao
>
> **摘要:** Recent advances in natural-domain multi-modal large language models (MLLMs) have demonstrated effective spatial reasoning through visual and textual prompting. However, their direct transfer to remote sensing (RS) is hindered by heterogeneous sensing physics, diverse modalities, and unique spatial scales. Existing RS MLLMs are mainly limited to optical imagery and plain language interaction, preventing flexible and scalable real-world applications. In this article, EarthGPT-X is proposed, the first flexible spatial MLLM that unifies multi-source RS imagery comprehension and accomplishes both coarse-grained and fine-grained visual tasks under diverse visual prompts in a single framework. Distinct from prior models, EarthGPT-X introduces: 1) a dual-prompt mechanism combining text instructions with various visual prompts (i.e., point, box, and free-form) to mimic the versatility of referring in human life; 2) a comprehensive multi-source multi-level prompting dataset, the model advances beyond holistic image understanding to support hierarchical spatial reasoning, including scene-level understanding and fine-grained object attributes and relational analysis; 3) a cross-domain one-stage fusion training strategy, enabling efficient and consistent alignment across modalities and tasks. Extensive experiments demonstrate that EarthGPT-X substantially outperforms prior nature and RS MLLMs, establishing the first framework capable of multi-source, multi-task, and multi-level interpretation using visual prompting in RS scenarios.
>
---
#### [replaced 074] X-GAN: A Generative AI-Powered Unsupervised Model for Main Vessel Segmentation of Glaucoma Screening
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06743v5](http://arxiv.org/pdf/2503.06743v5)**

> **作者:** Cheng Huang; Weizheng Xie; Tsengdar J. Lee; Jui-Kai Wang; Karanjit Kooner; Ning Zhang; Jia Zhang
>
> **摘要:** Structural changes in main retinal blood vessels serve as critical biomarkers for the onset and progression of glaucoma. Identifying these vessels is vital for vascular modeling yet highly challenging. This paper proposes X-GAN, a generative AI-powered unsupervised segmentation model designed for extracting main blood vessels from Optical Coherence Tomography Angiography (OCTA) images. The process begins with the Space Colonization Algorithm (SCA) to rapidly generate a skeleton of vessels, featuring their radii. By synergistically integrating the generative adversarial network (GAN) with biostatistical modeling of vessel radii, X-GAN enables a fast reconstruction of both 2D and 3D representations of the vessels. Based on this reconstruction, X-GAN achieves nearly 100\% segmentation accuracy without relying on labeled data or high-performance computing resources. Experimental results confirm X-GAN's superiority in evaluating main vessel segmentation compared to existing deep learning models.
>
---
#### [replaced 075] GroundFlow: A Plug-in Module for Temporal Reasoning on 3D Point Cloud Sequential Grounding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21188v3](http://arxiv.org/pdf/2506.21188v3)**

> **作者:** Zijun Lin; Shuting He; Cheston Tan; Bihan Wen
>
> **摘要:** Sequential grounding in 3D point clouds (SG3D) refers to locating sequences of objects by following text instructions for a daily activity with detailed steps. Current 3D visual grounding (3DVG) methods treat text instructions with multiple steps as a whole, without extracting useful temporal information from each step. However, the instructions in SG3D often contain pronouns such as "it", "here" and "the same" to make language expressions concise. This requires grounding methods to understand the context and retrieve relevant information from previous steps to correctly locate object sequences. Due to the lack of an effective module for collecting related historical information, state-of-the-art 3DVG methods face significant challenges in adapting to the SG3D task. To fill this gap, we propose GroundFlow -- a plug-in module for temporal reasoning on 3D point cloud sequential grounding. Firstly, we demonstrate that integrating GroundFlow improves the task accuracy of 3DVG baseline methods by a large margin (+7.5\% and +10.2\%) in the SG3D benchmark, even outperforming a 3D large language model pre-trained on various datasets. Furthermore, we selectively extract both short-term and long-term step information based on its relevance to the current instruction, enabling GroundFlow to take a comprehensive view of historical information and maintain its temporal understanding advantage as step counts increase. Overall, our work introduces temporal reasoning capabilities to existing 3DVG models and achieves state-of-the-art performance in the SG3D benchmark across five datasets.
>
---
#### [replaced 076] GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07417v4](http://arxiv.org/pdf/2503.07417v4)**

> **作者:** Minwen Liao; Hao Bo Dong; Xinyi Wang; Kurban Ubul; Yihua Shao; Ziyang Yan
>
> **摘要:** Low-light enhancement has wide applications in autonomous driving, 3D reconstruction, remote sensing, surveillance, and so on, which can significantly improve information utilization. However, most existing methods lack generalization and are limited to specific tasks such as image recovery. To address these issues, we propose Gated-Mechanism Mixture-of-Experts (GM-MoE), the first framework to introduce a mixture-of-experts network for low-light image enhancement. GM-MoE comprises a dynamic gated weight conditioning network and three sub-expert networks, each specializing in a distinct enhancement task. Combining a self-designed gated mechanism that dynamically adjusts the weights of the sub-expert networks for different data domains. Additionally, we integrate local and global feature fusion within sub-expert networks to enhance image quality by capturing multi-scale features. Experimental results demonstrate that the GM-MoE achieves superior generalization with respect to 25 compared approaches, reaching state-of-the-art performance on PSNR on 5 benchmarks and SSIM on 4 benchmarks, respectively.
>
---
#### [replaced 077] Proxy-Embedding as an Adversarial Teacher: An Embedding-Guided Bidirectional Attack for Referring Expression Segmentation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16157v2](http://arxiv.org/pdf/2506.16157v2)**

> **作者:** Xingbai Chen; Tingchao Fu; Renyang Liu; Wei Zhou; Chao Yi
>
> **备注:** 20pages, 5figures
>
> **摘要:** Referring Expression Segmentation (RES) enables precise object segmentation in images based on natural language descriptions, offering high flexibility and broad applicability in real-world vision tasks. Despite its impressive performance, the robustness of RES models against adversarial examples remains largely unexplored. While prior adversarial attack methods have explored adversarial robustness on conventional segmentation models, they perform poorly when directly applied to RES models, failing to expose vulnerabilities in its multimodal structure. In practical open-world scenarios, users typically issue multiple, diverse referring expressions to interact with the same image, highlighting the need for adversarial examples that generalize across varied textual inputs. Furthermore, from the perspective of privacy protection, ensuring that RES models do not segment sensitive content without explicit authorization is a crucial aspect of enhancing the robustness and security of multimodal vision-language systems. To address these challenges, we present PEAT, an Embedding-Guided Bidirectional Attack for RES models. Extensive experiments across multiple RES architectures and standard benchmarks show that PEAT consistently outperforms competitive baselines.
>
---
#### [replaced 078] Superpixel Graph Contrastive Clustering with Semantic-Invariant Augmentations for Hyperspectral Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.01799v2](http://arxiv.org/pdf/2403.01799v2)**

> **作者:** Jianhan Qi; Yuheng Jia; Hui Liu; Junhui Hou
>
> **摘要:** Hyperspectral images (HSI) clustering is an important but challenging task. The state-of-the-art (SOTA) methods usually rely on superpixels, however, they do not fully utilize the spatial and spectral information in HSI 3-D structure, and their optimization targets are not clustering-oriented. In this work, we first use 3-D and 2-D hybrid convolutional neural networks to extract the high-order spatial and spectral features of HSI through pre-training, and then design a superpixel graph contrastive clustering (SPGCC) model to learn discriminative superpixel representations. Reasonable augmented views are crucial for contrastive clustering, and conventional contrastive learning may hurt the cluster structure since different samples are pushed away in the embedding space even if they belong to the same class. In SPGCC, we design two semantic-invariant data augmentations for HSI superpixels: pixel sampling augmentation and model weight augmentation. Then sample-level alignment and clustering-center-level contrast are performed for better intra-class similarity and inter-class dissimilarity of superpixel embeddings. We perform clustering and network optimization alternatively. Experimental results on several HSI datasets verify the advantages of the proposed SPGCC compared to SOTA methods. Our code is available at https://github.com/jhqi/spgcc.
>
---
#### [replaced 079] Show-o2: Improved Native Unified Multimodal Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15564v3](http://arxiv.org/pdf/2506.15564v3)**

> **作者:** Jinheng Xie; Zhenheng Yang; Mike Zheng Shou
>
> **备注:** NeurIPS 2025. (v3: update to include video understanding, OneIG, and more ablation study results)
>
> **摘要:** This paper presents improved native unified multimodal models, \emph{i.e.,} Show-o2, that leverage autoregressive modeling and flow matching. Built upon a 3D causal variational autoencoder space, unified visual representations are constructed through a dual-path of spatial (-temporal) fusion, enabling scalability across image and video modalities while ensuring effective multimodal understanding and generation. Based on a language model, autoregressive modeling and flow matching are natively applied to the language head and flow head, respectively, to facilitate text token prediction and image/video generation. A two-stage training recipe is designed to effectively learn and scale to larger models. The resulting Show-o2 models demonstrate versatility in handling a wide range of multimodal understanding and generation tasks across diverse modalities, including text, images, and videos. Code and models are released at https://github.com/showlab/Show-o.
>
---
#### [replaced 080] Disentangling Content from Style to Overcome Shortcut Learning: A Hybrid Generative-Discriminative Learning Framework
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.11598v3](http://arxiv.org/pdf/2509.11598v3)**

> **作者:** Siming Fu; Sijun Dong; Xiaoliang Meng
>
> **摘要:** Despite the remarkable success of Self-Supervised Learning (SSL), its generalization is fundamentally hindered by Shortcut Learning, where models exploit superficial features like texture instead of intrinsic structure. We experimentally verify this flaw within the generative paradigm (e.g., MAE) and argue it is a systemic issue also affecting discriminative methods, identifying it as the root cause of their failure on unseen domains. While existing methods often tackle this at a surface level by aligning or separating domain-specific features, they fail to alter the underlying learning mechanism that fosters shortcut dependency. To address this at its core, we propose HyGDL (Hybrid Generative-Discriminative Learning Framework), a hybrid framework that achieves explicit content-style disentanglement. Our approach is guided by the Invariance Pre-training Principle: forcing a model to learn an invariant essence by systematically varying a bias (e.g., style) at the input while keeping the supervision signal constant. HyGDL operates on a single encoder and analytically defines style as the component of a representation that is orthogonal to its style-invariant content, derived via vector projection. This is operationalized through a synergistic design: (1) a self-distillation objective learns a stable, style-invariant content direction; (2) an analytical projection then decomposes the representation into orthogonal content and style vectors; and (3) a style-conditioned reconstruction objective uses these vectors to restore the image, providing end-to-end supervision. Unlike prior methods that rely on implicit heuristics, this principled disentanglement allows HyGDL to learn truly robust representations, demonstrating superior performance on benchmarks designed to diagnose shortcut learning.
>
---
#### [replaced 081] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05255v2](http://arxiv.org/pdf/2507.05255v2)**

> **作者:** Yana Wei; Liang Zhao; Jianjian Sun; Kangheng Lin; Jisheng Yin; Jingcheng Hu; Yinmin Zhang; En Yu; Haoran Lv; Zejia Weng; Jia Wang; Chunrui Han; Yuang Peng; Qi Han; Zheng Ge; Xiangyu Zhang; Daxin Jiang; Vishal M. Patel
>
> **备注:** NeurIPS 2025
>
> **摘要:** The remarkable reasoning capability of large language models (LLMs) stems from cognitive behaviors that emerge through reinforcement with verifiable rewards. This work investigates how to transfer this principle to Multimodal LLMs (MLLMs) to unlock advanced visual reasoning. We introduce a two-stage paradigm built on Qwen2.5-VL-7B: a massive linguistic cold-start fine-tuning, followed by multimodal reinforcement learning (RL) spanning nearly 1,000 steps, surpassing all previous open-source efforts in scale. This pioneering work reveals three fundamental insights: 1) Behavior transfer emerges surprisingly early in cold start due to linguistic mental imagery. 2) Cold start broadly memorizes visual behaviors, while RL critically discerns and scales up effective patterns. 3) Transfer strategically favors high-utility behaviors such as visual reflection. Our resulting model, Open-Vision-Reasoner (OVR), achieves state-of-the-art performance on a suite of reasoning benchmarks, including 95.3% on MATH500, 51.8% on MathVision and 54.6% on MathVerse. We release our model, data, and training dynamics to catalyze the development of more capable, behavior-aligned multimodal reasoners.
>
---
#### [replaced 082] VisText-Mosquito: A Unified Multimodal Benchmark Dataset for Visual Detection, Segmentation, and Textual Reasoning on Mosquito Breeding Sites
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14629v2](http://arxiv.org/pdf/2506.14629v2)**

> **作者:** Md. Adnanul Islam; Md. Faiyaz Abdullah Sayeedi; Md. Asaduzzaman Shuvo; Shahanur Rahman Bappy; Md Asiful Islam; Swakkhar Shatabda
>
> **摘要:** Mosquito-borne diseases pose a major global health risk, requiring early detection and proactive control of breeding sites to prevent outbreaks. In this paper, we present VisText-Mosquito, a multimodal dataset that integrates visual and textual data to support automated detection, segmentation, and reasoning for mosquito breeding site analysis. The dataset includes 1,828 annotated images for object detection, 142 images for water surface segmentation, and natural language reasoning texts linked to each image. The YOLOv9s model achieves the highest precision of 0.92926 and mAP@50 of 0.92891 for object detection, while YOLOv11n-Seg reaches a segmentation precision of 0.91587 and mAP@50 of 0.79795. For reasoning generation, we tested a range of large vision-language models (LVLMs) in both zero-shot and few-shot settings. Our fine-tuned Mosquito-LLaMA3-8B model achieved the best results, with a final loss of 0.0028, a BLEU score of 54.7, BERTScore of 0.91, and ROUGE-L of 0.85. This dataset and model framework emphasize the theme "Prevention is Better than Cure", showcasing how AI-based detection can proactively address mosquito-borne disease risks. The dataset and implementation code are publicly available at GitHub: https://github.com/adnanul-islam-jisun/VisText-Mosquito
>
---
#### [replaced 083] PASS: Path-selective State Space Model for Event-based Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.16953v2](http://arxiv.org/pdf/2409.16953v2)**

> **作者:** Jiazhou Zhou; Kanghao Chen; Lei Zhang; Lin Wang
>
> **备注:** Accepted by NeurIPS 2025. Main paper: 10 pages; Supplementary: 6 pages
>
> **摘要:** Event cameras are bio-inspired sensors that capture intensity changes asynchronously with distinct advantages, such as high temporal resolution. Existing methods for event-based object/action recognition predominantly sample and convert event representation at every fixed temporal interval (or frequency). However, they are constrained to processing a limited number of event lengths and show poor frequency generalization, thus not fully leveraging the event's high temporal resolution. In this paper, we present our PASS framework, exhibiting superior capacity for spatiotemporal event modeling towards a larger number of event lengths and generalization across varying inference temporal frequencies. Our key insight is to learn adaptively encoded event features via the state space models (SSMs), whose linear complexity and generalization on input frequency make them ideal for processing high temporal resolution events. Specifically, we propose a Path-selective Event Aggregation and Scan (PEAS) module to encode events into features with fixed dimensions by adaptively scanning and selecting aggregated event presentations. On top of it, we introduce a novel Multi-faceted Selection Guiding (MSG) loss to minimize the randomness and redundancy of the encoded features during the PEAS selection process. Our method outperforms prior methods on five public datasets and shows strong generalization across varying inference frequencies with less accuracy drop (ours -8.62% vs. -20.69% for the baseline). Overall, PASS exhibits strong long spatiotemporal modeling for a broader distribution of event length (1-10^9), precise temporal perception, and generalization for real-world
>
---
#### [replaced 084] SD-VSum: A Method and Dataset for Script-Driven Video Summarization
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.03319v2](http://arxiv.org/pdf/2505.03319v2)**

> **作者:** Manolis Mylonas; Evlampios Apostolidis; Vasileios Mezaris
>
> **备注:** In ACM Multimedia 2025, DOI:10.1145/3746027.3755821
>
> **摘要:** In this work, we introduce the task of script-driven video summarization, which aims to produce a summary of the full-length video by selecting the parts that are most relevant to a user-provided script outlining the visual content of the desired summary. Following, we extend a recently-introduced large-scale dataset for generic video summarization (VideoXum) by producing natural language descriptions of the different human-annotated summaries that are available per video. In this way we make it compatible with the introduced task, since the available triplets of ``video, summary and summary description'' can be used for training a method that is able to produce different summaries for a given video, driven by the provided script about the content of each summary. Finally, we develop a new network architecture for script-driven video summarization (SD-VSum), that employs a cross-modal attention mechanism for aligning and fusing information from the visual and text modalities. Our experimental evaluations demonstrate the advanced performance of SD-VSum against SOTA approaches for query-driven and generic (unimodal and multimodal) summarization from the literature, and document its capacity to produce video summaries that are adapted to each user's needs about their content.
>
---
#### [replaced 085] DCFFSNet: Deep Connectivity Feature Fusion Separation Network for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.18407v2](http://arxiv.org/pdf/2507.18407v2)**

> **作者:** Mingda Zhang; Xun Ye; Ruixiang Tang; Haiyan Ding
>
> **备注:** 16 pages , 11 figures
>
> **摘要:** Medical image segmentation leverages topological connectivity theory to enhance edge precision and regional consistency. However, existing deep networks integrating connectivity often forcibly inject it as an additional feature module, resulting in coupled feature spaces with no standardized mechanism to quantify different feature strengths. To address these issues, we propose DCFFSNet (Dual-Connectivity Feature Fusion-Separation Network). It introduces an innovative feature space decoupling strategy. This strategy quantifies the relative strength between connectivity features and other features. It then builds a deep connectivity feature fusion-separation architecture. This architecture dynamically balances multi-scale feature expression. Experiments were conducted on the ISIC2018, DSB2018, and MoNuSeg datasets. On ISIC2018, DCFFSNet outperformed the next best model (CMUNet) by 1.3% (Dice) and 1.2% (IoU). On DSB2018, it surpassed TransUNet by 0.7% (Dice) and 0.9% (IoU). On MoNuSeg, it exceeded CSCAUNet by 0.8% (Dice) and 0.9% (IoU). The results demonstrate that DCFFSNet exceeds existing mainstream methods across all metrics. It effectively resolves segmentation fragmentation and achieves smooth edge transitions. This significantly enhances clinical usability.
>
---
#### [replaced 086] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12434v3](http://arxiv.org/pdf/2505.12434v3)**

> **作者:** Qi Wang; Yanrui Yu; Ye Yuan; Rui Mao; Tianfei Zhou
>
> **备注:** Accepted by NeurIPS 2025. Code: https://github.com/QiWang98/VideoRFT
>
> **摘要:** Reinforcement fine-tuning (RFT) has shown great promise in achieving humanlevel reasoning capabilities of Large Language Models (LLMs), and has recently been extended to MLLMs. Nevertheless, reasoning about videos, which is a fundamental aspect of human intelligence, remains a persistent challenge due to the complex logic, temporal and causal structures inherent in video data. To fill this gap, we propose VIDEORFT, a novel approach that extends the RFT paradigm to cultivate human-like video reasoning capabilities in MLLMs. VIDEORFT follows the standard two-stage scheme in RFT: supervised fine-tuning (SFT) with chain-of-thought (CoT) annotations, followed by reinforcement learning (RL) to improve generalization. A central challenge to achieve this in the video domain lies in the scarcity of large-scale, high-quality video CoT datasets. We address this by building a multi-expert, cognition-inspired CoT curation pipeline. First, we devise a cognition-inspired prompting strategy to elicit a reasoning LLM to generate preliminary CoTs based solely on rich, structured, and literal representations of video content. Subsequently, these CoTs are revised by a MLLM conditioned on the actual video, ensuring visual consistency and reducing visual hallucinations. This pipeline results in two new datasets, i.e.VideoRFT-CoT-102K for SFT and VideoRFT-RL-310K for RL. To further strengthen the RL phase, we introduce a novel semantic-consistency reward that explicitly promotes the alignment between textual reasoning and visual evidence. This reward encourages the model to produce coherent, context-aware reasoning outputs grounded in visual input. Extensive experiments show that VIDEORFT achieves state-of-the-art performance on six video reasoning benchmarks.
>
---
#### [replaced 087] SpecVLM: Fast Speculative Decoding in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.11815v2](http://arxiv.org/pdf/2509.11815v2)**

> **作者:** Haiduo Huang; Fuwei Yang; Zhenhua Liu; Xuanwu Yin; Dong Li; Pengju Ren; Emad Barsoum
>
> **摘要:** Speculative decoding is a powerful way to accelerate autoregressive large language models (LLMs), but directly porting it to vision-language models (VLMs) faces unique systems constraints: the prefill stage is dominated by visual tokens whose count scales with image resolution and video length, inflating both compute and memory, especially the key-value (KV) cache. We study speculative decoding for VLMs and introduce SpecVLM, a practical system that (1) establishes a strong EAGLE-2-style baseline, EagleVLM, delivering 1.5--2.3x end-to-end speedups over full autoregressive inference, and (2) further accelerates VLM inference with an elastic visual compressor that adaptively selects among pruning, pooling, convolution, and resampler primitives to balance FLOPs/parameters and accuracy per input. To avoid costly offline distillation corpora, we propose an online-logit distillation protocol that trains the draft model with on-the-fly teacher logits and penultimate features using a combined cross-entropy and Smooth L1 objective, eliminating storage and preprocessing while remaining compute-efficient. This protocol reveals a training-time scaling effect: longer online training monotonically increases the draft model's average accepted length, improving speculative efficiency. Empirically, SpecVLM achieves additional acceleration, culminating in 2.5--2.9x end-to-end speedups within 5 epochs across LLaVA and MMMU, consistently over resolutions and task difficulties, while preserving the target model's output distribution (lossless decoding). Our code is available at https://github.com/haiduo/SpecVLM.
>
---
#### [replaced 088] Exploring the Design Space of 3D MLLMs for CT Report Generation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.21535v2](http://arxiv.org/pdf/2506.21535v2)**

> **作者:** Mohammed Baharoon; Jun Ma; Congyu Fang; Augustin Toma; Bo Wang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have emerged as a promising way to automate Radiology Report Generation (RRG). In this work, we systematically investigate the design space of 3D MLLMs, including visual input representation, projectors, Large Language Models (LLMs), and fine-tuning techniques for 3D CT report generation. We also introduce two knowledge-based report augmentation methods that improve performance on the GREEN score by up to 10%, achieving the 2nd place on the MICCAI 2024 AMOS-MM challenge. Our results on the 1,687 cases from the AMOS-MM dataset show that RRG is largely independent of the size of LLM under the same training protocol. We also show that larger volume size does not always improve performance if the original ViT was pre-trained on a smaller volume size. Lastly, we show that using a segmentation mask along with the CT volume improves performance. The code is publicly available at https://github.com/bowang-lab/AMOS-MM-Solution
>
---
#### [replaced 089] Test-Time Multimodal Backdoor Detection by Contrastive Prompting
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.15269v3](http://arxiv.org/pdf/2405.15269v3)**

> **作者:** Yuwei Niu; Shuo He; Qi Wei; Zongyu Wu; Feng Liu; Lei Feng
>
> **备注:** Accepted to ICML2025
>
> **摘要:** While multimodal contrastive learning methods (e.g., CLIP) can achieve impressive zero-shot classification performance, recent research has revealed that these methods are vulnerable to backdoor attacks. To defend against backdoor attacks on CLIP, existing defense methods focus on either the pre-training stage or the fine-tuning stage, which would unfortunately cause high computational costs due to numerous parameter updates and are not applicable in black-box settings. In this paper, we provide the first attempt at a computationally efficient backdoor detection method to defend against backdoored CLIP in the \emph{inference} stage. We empirically find that the visual representations of backdoored images are \emph{insensitive} to \emph{benign} and \emph{malignant} changes in class description texts. Motivated by this observation, we propose BDetCLIP, a novel test-time backdoor detection method based on contrastive prompting. Specifically, we first prompt a language model (e.g., GPT-4) to produce class-related description texts (benign) and class-perturbed random texts (malignant) by specially designed instructions. Then, the distribution difference in cosine similarity between images and the two types of class description texts can be used as the criterion to detect backdoor samples. Extensive experiments validate that our proposed BDetCLIP is superior to state-of-the-art backdoor detection methods, in terms of both effectiveness and efficiency. Our codes are publicly available at: https://github.com/Purshow/BDetCLIP.
>
---
#### [replaced 090] Diversity-Guided MLP Reduction for Efficient Large Vision Transformers
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.08591v2](http://arxiv.org/pdf/2506.08591v2)**

> **作者:** Chengchao Shen; Hourun Zhu; Gongfan Fang; Jianxin Wang; Xinchao Wang
>
> **摘要:** Transformer models achieve excellent scaling property, where the performance is improved with the increment of model capacity. However, large-scale model parameters lead to an unaffordable cost of computing and memory. We analyze popular transformer architectures and find that multilayer perceptron (MLP) modules take up the majority of model parameters. To this end, we focus on the recoverability of the compressed models and propose a Diversity-Guided MLP Reduction (DGMR) method to significantly reduce the parameters of large vision transformers with only negligible performance degradation. Specifically, we conduct a Gram-Schmidt weight pruning strategy to eliminate redundant neurons of MLP hidden layer, while preserving weight diversity for better performance recover during distillation. Compared to the model trained from scratch, our pruned model only requires 0.06\% data of LAION-2B (for the training of large vision transformers) without labels (ImageNet-1K) to recover the original performance. Experimental results on several state-of-the-art large vision transformers demonstrate that our method achieves a more than 57.0\% parameter and FLOPs reduction in a near lossless manner. Notably, for EVA-CLIP-E (4.4B), our method accomplishes a 71.5\% parameter and FLOPs reduction without performance degradation. The source code and trained weights are available at https://github.com/visresearch/DGMR.
>
---
#### [replaced 091] TinyDef-DETR: A DETR-based Framework for Defect Detection in Transmission Lines from UAV Imagery
- **分类: cs.CV; cs.AI; cs.CE**

- **链接: [http://arxiv.org/pdf/2509.06035v4](http://arxiv.org/pdf/2509.06035v4)**

> **作者:** Feng Shen; Jiaming Cui; Shuai Zhou; Wenqiang Li; Ruifeng Qin
>
> **摘要:** Automated defect detection from UAV imagery of transmission lines is a challenging task due to the small size, ambiguity, and complex backgrounds of defects. This paper proposes TinyDef-DETR, a DETR-based framework designed to achieve accurate and efficient detection of transmission line defects from UAV-acquired images. The model integrates four major components: an edge-enhanced ResNet backbone to strengthen boundary-sensitive representations, a stride-free space-to-depth module to enable detail-preserving downsampling, a cross-stage dual-domain multi-scale attention mechanism to jointly model global context and local cues, and a Focaler-Wise-SIoU regression loss to improve the localization of small and difficult targets. Together, these designs effectively mitigate the limitations of conventional detectors. Extensive experiments on both public and real-world datasets demonstrate that TinyDef-DETR achieves superior detection performance and strong generalization capability, while maintaining modest computational overhead. The accuracy and efficiency of TinyDef-DETR make it a suitable method for UAV-based transmission line defect detection, particularly in scenarios involving small and ambiguous targets.
>
---
#### [replaced 092] RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13289v2](http://arxiv.org/pdf/2505.13289v2)**

> **作者:** Alonso Urbano; David W. Romero; Max Zimmer; Sebastian Pokutta
>
> **摘要:** Real world data often exhibits unknown, instance-specific symmetries that rarely exactly match a transformation group $G$ fixed a priori. Class-pose decompositions aim to create disentangled representations by factoring inputs into invariant features and a pose $g\in G$ defined relative to a training-dependent, arbitrary canonical representation. We introduce RECON, a class-pose agnostic $\textit{canonical orientation normalization}$ that corrects arbitrary canonicals via a simple right-multiplication, yielding $\textit{natural}$, data-aligned canonicalizations. This enables (i) unsupervised discovery of instance-specific symmetry distributions, (ii) detection of out-of-distribution poses, and (iii) test-time canonicalization, granting group invariance to pre-trained models without retraining and irrespective of model architecture, improving downstream performance. We demonstrate results on 2D image benchmarks and --for the first time-- extend symmetry discovery to 3D groups.
>
---
#### [replaced 093] DescriptorMedSAM: Language-Image Fusion with Multi-Aspect Text Guidance for Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13806v2](http://arxiv.org/pdf/2503.13806v2)**

> **作者:** Wenjie Zhang; Liming Luo; Mengnan He; Jiarui Hai; Jiancheng Ye
>
> **摘要:** Accurate organ segmentation is essential for clinical tasks such as radiotherapy planning and disease monitoring. Recent foundation models like MedSAM achieve strong results using point or bounding-box prompts but still require manual interaction. We propose DescriptorMedSAM, a lightweight extension of MedSAM that incorporates structured text prompts, ranging from simple organ names to combined shape and location descriptors to enable click-free segmentation. DescriptorMedSAM employs a CLIP text encoder to convert radiology-style descriptors into dense embeddings, which are fused with visual tokens via a cross-attention block and a multi-scale feature extractor. We designed four descriptor types: Name (N), Name + Shape (NS), Name + Location (NL), and Name + Shape + Location (NSL), and evaluated them on the FLARE 2022 dataset under zero-shot and few-shot settings, where organs unseen during training must be segmented with minimal additional data. NSL prompts achieved the highest performance, with a Dice score of 0.9405 under full supervision, a 76.31% zero-shot retention ratio, and a 97.02% retention ratio after fine-tuning with only 50 labeled slices per unseen organ. Adding shape and location cues consistently improved segmentation accuracy, especially for small or morphologically complex structures. We demonstrate that structured language prompts can effectively replace spatial interactions, delivering strong zero-shot performance and rapid few-shot adaptation. By quantifying the role of descriptor, this work lays the groundwork for scalable, prompt-aware segmentation models that generalize across diverse anatomical targets with minimal annotation effort.
>
---
#### [replaced 094] Interpretability-Aware Pruning for Efficient Medical Image Analysis
- **分类: cs.CV; cs.AI; cs.ET; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.08330v2](http://arxiv.org/pdf/2507.08330v2)**

> **作者:** Nikita Malik; Pratinav Seth; Neeraj Kumar Singh; Chintan Chitroda; Vinay Kumar Sankarapu
>
> **备注:** Accepted at The 1st MICCAI Workshop on Efficient Medical AI 2025
>
> **摘要:** Deep learning has driven significant advances in medical image analysis, yet its adoption in clinical practice remains constrained by the large size and lack of transparency in modern models. Advances in interpretability techniques such as DL-Backtrace, Layer-wise Relevance Propagation, and Integrated Gradients make it possible to assess the contribution of individual components within neural networks trained on medical imaging tasks. In this work, we introduce an interpretability-guided pruning framework that reduces model complexity while preserving both predictive performance and transparency. By selectively retaining only the most relevant parts of each layer, our method enables targeted compression that maintains clinically meaningful representations. Experiments across multiple medical image classification benchmarks demonstrate that this approach achieves high compression rates with minimal loss in accuracy, paving the way for lightweight, interpretable models suited for real-world deployment in healthcare settings.
>
---
#### [replaced 095] Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2412.03526v3](http://arxiv.org/pdf/2412.03526v3)**

> **作者:** Hanxue Liang; Jiawei Ren; Ashkan Mirzaei; Antonio Torralba; Ziwei Liu; Igor Gilitschenski; Sanja Fidler; Cengiz Oztireli; Huan Ling; Zan Gojcic; Jiahui Huang
>
> **备注:** Project website: https://research.nvidia.com/labs/toronto-ai/bullet-timer/
>
> **摘要:** Recent advancements in static feed-forward scene reconstruction have demonstrated significant progress in high-quality novel view synthesis. However, these models often struggle with generalizability across diverse environments and fail to effectively handle dynamic content. We present BTimer (short for BulletTimer), the first motion-aware feed-forward model for real-time reconstruction and novel view synthesis of dynamic scenes. Our approach reconstructs the full scene in a 3D Gaussian Splatting representation at a given target ('bullet') timestamp by aggregating information from all the context frames. Such a formulation allows BTimer to gain scalability and generalization by leveraging both static and dynamic scene datasets. Given a casual monocular dynamic video, BTimer reconstructs a bullet-time scene within 150ms while reaching state-of-the-art performance on both static and dynamic scene datasets, even compared with optimization-based approaches.
>
---
#### [replaced 096] Evaluating Fairness in Large Vision-Language Models Across Diverse Demographic Attributes and Prompts
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.17974v3](http://arxiv.org/pdf/2406.17974v3)**

> **作者:** Xuyang Wu; Yuan Wang; Hsin-Tai Wu; Zhiqiang Tao; Yi Fang
>
> **备注:** EMNLP Findings
>
> **摘要:** Large vision-language models (LVLMs) have recently achieved significant progress, demonstrating strong capabilities in open-world visual understanding. However, it is not yet clear how LVLMs address demographic biases in real life, especially the disparities across attributes such as gender, skin tone, age and race. In this paper, We empirically investigate \emph{visual fairness} in several mainstream LVLMs by auditing their performance disparities across demographic attributes using public fairness benchmark datasets (e.g., FACET, UTKFace). Our fairness evaluation framework employs direct and single-choice question prompt on visual question-answering/classification tasks. Despite advancements in visual understanding, our zero-shot prompting results show that both open-source and closed-source LVLMs continue to exhibit fairness issues across different prompts and demographic groups. Furthermore, we propose a potential multi-modal Chain-of-thought (CoT) based strategy for unfairness mitigation, applicable to both open-source and closed-source LVLMs. This approach enhances transparency and offers a scalable solution for addressing fairness, providing a solid foundation for future research and practical efforts in unfairness mitigation. The dataset and code used in this study are publicly available at this GitHub Repository.
>
---
#### [replaced 097] UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08787v4](http://arxiv.org/pdf/2505.08787v4)**

> **作者:** Hanjung Kim; Jaehyun Kang; Hyolim Kang; Meedeum Cho; Seon Joo Kim; Youngwoon Lee
>
> **备注:** CoRL 2025. Project Page: https://kimhanjung.github.io/UniSkill/
>
> **摘要:** Mimicry is a fundamental learning mechanism in humans, enabling individuals to learn new tasks by observing and imitating experts. However, applying this ability to robots presents significant challenges due to the inherent differences between human and robot embodiments in both their visual appearance and physical capabilities. While previous methods bridge this gap using cross-embodiment datasets with shared scenes and tasks, collecting such aligned data between humans and robots at scale is not trivial. In this paper, we propose UniSkill, a novel framework that learns embodiment-agnostic skill representations from large-scale cross-embodiment video data without any labels, enabling skills extracted from human video prompts to effectively transfer to robot policies trained only on robot data. Our experiments in both simulation and real-world environments show that our cross-embodiment skills successfully guide robots in selecting appropriate actions, even with unseen video prompts. The project website can be found at: https://kimhanjung.github.io/UniSkill.
>
---
#### [replaced 098] Elevating Visual Perception in Multimodal LLMs with Auxiliary Embedding Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09585v2](http://arxiv.org/pdf/2412.09585v2)**

> **作者:** Jitesh Jain; Zhengyuan Yang; Humphrey Shi; Jianfeng Gao; Jianwei Yang
>
> **备注:** Project Page: https://praeclarumjj3.github.io/visper_lm/
>
> **摘要:** In recent times, the standard practice for developing MLLMs is to feed features from vision encoder(s) into the LLM and train with natural language supervision. This approach often causes models to lean towards language comprehension and undermine the rich visual perception signals present in the data, which are critical for tasks involving spatial reasoning in the domain of embodied AI and robotics. Is it possible to optimize both at the same time? In this work, we propose VisPer-LM, the first approach that infuses visual perception knowledge from expert vision encoders into the LLM's (of an MLLM) hidden representations. We start by investigating MLLMs trained solely with natural language supervision and identify a positive correlation between the quality of visual representations within these models and their downstream performance. Given this insight, we formulate the objective during the pretraining stage in MLLMs as a coupled optimization of predictive visual embedding and next (text) token prediction. Moreover, through extensive probing, we observe improved visual representation quality due to embedding optimization, underscoring the effectiveness of our probing setup. We demonstrate that our VisPer-LM outperforms the single and multi-encoder baselines, proving our approach's superiority over explicitly feeding the corresponding features to the LLM. In particular, VisPer-LM boosts performance by an average margin of up to 2.5% on various benchmarks, with a notable improvement of 8.7% on the Depth task in CV-Bench.
>
---
#### [replaced 099] An Efficient Dual-Line Decoder Network with Multi-Scale Convolutional Attention for Multi-organ Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.17007v2](http://arxiv.org/pdf/2508.17007v2)**

> **作者:** Riad Hassan; M. Rubaiyat Hossain Mondal; Sheikh Iqbal Ahamed; Fahad Mostafa; Md Mostafijur Rahman
>
> **备注:** After revision, minor ablation studies have been added in the published version in Biomedical Signal Processing and Control (BSPC)
>
> **摘要:** Proper segmentation of organs-at-risk is important for radiation therapy, surgical planning, and diagnostic decision-making in medical image analysis. While deep learning-based segmentation architectures have made significant progress, they often fail to balance segmentation accuracy with computational efficiency. Most of the current state-of-the-art methods either prioritize performance at the cost of high computational complexity or compromise accuracy for efficiency. This paper addresses this gap by introducing an efficient dual-line decoder segmentation network (EDLDNet). The proposed method features a noisy decoder, which learns to incorporate structured perturbation at training time for better model robustness, yet at inference time only the noise-free decoder is executed, leading to lower computational cost. Multi-Scale convolutional Attention Modules (MSCAMs), Attention Gates (AGs), and Up-Convolution Blocks (UCBs) are further utilized to optimize feature representation and boost segmentation performance. By leveraging multi-scale segmentation masks from both decoders, we also utilize a mutation-based loss function to enhance the model's generalization. Our approach outperforms SOTA segmentation architectures on four publicly available medical imaging datasets. EDLDNet achieves SOTA performance with an 84.00% Dice score on the Synapse dataset, surpassing baseline model like UNet by 13.89% in Dice score while significantly reducing Multiply-Accumulate Operations (MACs) by 89.7%. Compared to recent approaches like EMCAD, our EDLDNet not only achieves higher Dice score but also maintains comparable computational efficiency. The outstanding performance across diverse datasets establishes EDLDNet's strong generalization, computational efficiency, and robustness. The source code, pre-processed data, and pre-trained weights will be available at https://github.com/riadhassan/EDLDNet .
>
---
#### [replaced 100] PoseBench3D: A Cross-Dataset Analysis Framework for 3D Human Pose Estimation via Pose Lifting Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10888v2](http://arxiv.org/pdf/2505.10888v2)**

> **作者:** Saad Manzur; Bryan Vela; Brandon Vela; Aditya Agrawal; Lan-Anh Dang-Vu; David Li; Wayne Hayes
>
> **备注:** Code: https://github.com/bryanjvela/PoseBench3D
>
> **摘要:** Reliable three-dimensional human pose estimation (3D HPE) remains challenging due to the differences in viewpoints, environments, and camera conventions among datasets. As a result, methods that achieve near-optimal in-dataset accuracy often degrade on unseen datasets. In practice, however, systems must adapt to diverse viewpoints, environments, and camera setups--conditions that differ significantly from those encountered during training, which is often the case in real-world scenarios. Measuring cross-dataset performance is a vital process, but extremely labor-intensive when done manually for human pose estimation. To address these challenges, we automate this evaluation using PoseBench3D, a standardized testing framework that enables consistent and fair cross-dataset comparisons on previously unseen data. PoseBench3D streamlines testing across four widely used 3D HPE datasets via a single, configurable interface. Using this framework, we re-evaluate 18 methods and report over 100 cross-dataset results under Protocol 1: MPJPE and Protocol 2: PA-MPJPE, revealing systematic generalization gaps and the impact of common preprocessing and dataset setup choices. The PoseBench3D code is found at: https://github.com/bryanjvela/PoseBench3D
>
---
#### [replaced 101] ProReason: Multi-Modal Proactive Reasoning with Decoupled Eyesight and Wisdom
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.14138v4](http://arxiv.org/pdf/2410.14138v4)**

> **作者:** Jingqi Zhou; Sheng Wang; Jingwei Dong; Kai Liu; Lei Li; Jiahui Gao; Jiyue Jiang; Lingpeng Kong; Chuan Wu
>
> **摘要:** Large vision-language models (LVLMs) have witnessed significant progress on visual understanding tasks. However, they often prioritize language knowledge over image information on visual reasoning tasks, incurring performance degradation. To tackle this issue, we first identify the drawbacks of existing solutions (i.e., limited multi-modal reasoning capacities, and insufficient and irrelevant visual descriptions). We then decompose visual reasoning process into two stages: proactive visual perception (i.e., eyesight) and textual reasoning (i.e., wisdom), and introduce a novel visual reasoning framework named ProReason. This framework features decoupled vision-reasoning capabilities and multi-run proactive perception. Briefly, given a multi-modal question, ProReason iterates proactive information collection and reasoning until the answer can be concluded with necessary and sufficient visual descriptions. Notably, the disassociation of capabilities allows seamless integration of existing large language models (LLMs) to compensate for the reasoning deficits of LVLMs. Our extensive experiments demonstrate that ProReason outperforms existing multi-step reasoning frameworks on various benchmarks for both open-source and closed-source models, with the average performance gain reaching 13.2%. Besides, the integration of LLMs allows ProReason to produce high-quality visual reasoning data, which empowers ProReason-distilled models (i.e., ProReason-VL and ProReason-Q3) to achieve superior performance in downstream tasks. Our insights into existing solutions and the decoupled perspective for feasible integration of LLMs illuminate future research on visual reasoning techniques, especially LLM-assisted ones.
>
---
#### [replaced 102] Co-STAR: Collaborative Curriculum Self-Training with Adaptive Regularization for Source-Free Video Domain Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11669v2](http://arxiv.org/pdf/2504.11669v2)**

> **作者:** Amirhossein Dadashzadeh; Parsa Esmati; Majid Mirmehdi
>
> **摘要:** Recent advances in Source-Free Unsupervised Video Domain Adaptation (SFUVDA) leverage vision-language models to enhance pseudo-label generation. However, challenges such as noisy pseudo-labels and over-confident predictions limit their effectiveness in adapting well across domains. We propose Co-STAR, a novel framework that integrates curriculum learning with collaborative self-training between a source-trained teacher and a contrastive vision-language model (CLIP). Our curriculum learning approach employs a reliability-based weight function that measures bidirectional prediction alignment between the teacher and CLIP, balancing between confident and uncertain predictions. This function preserves uncertainty for difficult samples, while prioritizing reliable pseudo-labels when the predictions from both models closely align. To further improve adaptation, we propose Adaptive Curriculum Regularization, which modifies the learning priority of samples in a probabilistic, adaptive manner based on their confidence scores and prediction stability, mitigating overfitting to noisy and over-confident samples. Extensive experiments across multiple video domain adaptation benchmarks demonstrate that Co-STAR consistently outperforms state-of-the-art SFUVDA methods. Code is available at: https://github.com/Plrbear/Co-Star
>
---
#### [replaced 103] MOSEv2: A More Challenging Dataset for Video Object Segmentation in Complex Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05630v2](http://arxiv.org/pdf/2508.05630v2)**

> **作者:** Henghui Ding; Kaining Ying; Chang Liu; Shuting He; Xudong Jiang; Yu-Gang Jiang; Philip H. S. Torr; Song Bai
>
> **备注:** MOSEv2 Dataset Report, Project Page: https://mose.video/, Baseline & metric code: https://github.com/henghuiding/MOSE-api
>
> **摘要:** Video object segmentation (VOS) aims to segment specified target objects throughout a video. Although state-of-the-art methods have achieved impressive performance (e.g., 90+% J&F) on benchmarks such as DAVIS and YouTube-VOS, these datasets primarily contain salient, dominant, and isolated objects, limiting their generalization to real-world scenarios. To bridge this gap, the coMplex video Object SEgmentation (MOSEv1) dataset was introduced to facilitate VOS research in complex scenes. Building on the foundations and insights of MOSEv1, we present MOSEv2, a significantly more challenging dataset designed to further advance VOS methods under real-world conditions. MOSEv2 consists of 5,024 videos and 701,976 high-quality masks for 10,074 objects across 200 categories. Compared to its predecessor, MOSEv2 introduces much greater scene complexity, including {more frequent object disappearance and reappearance, severe occlusions and crowding, smaller objects, as well as a range of new challenges such as adverse weather (e.g., rain, snow, fog), low-light scenes (e.g., nighttime, underwater), multi-shot sequences, camouflaged objects, non-physical targets (e.g., shadows, reflections), and scenarios requiring external knowledge.} We benchmark 20 representative VOS methods under 5 different settings and observe consistent performance drops on MOSEv2. For example, SAM2 drops from 76.4% on MOSEv1 to only 50.9% on MOSEv2. We further evaluate 9 video object tracking methods and observe similar declines, demonstrating that MOSEv2 poses challenges across tasks. These results highlight that despite strong performance on existing datasets, current VOS methods still fall short under real-world complexities. Based on our analysis of the observed challenges, we further propose several practical tricks that enhance model performance. MOSEv2 is publicly available at https://MOSE.video.
>
---
#### [replaced 104] DriveSplat: Decoupled Driving Scene Reconstruction with Geometry-enhanced Partitioned Neural Gaussians
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15376v3](http://arxiv.org/pdf/2508.15376v3)**

> **作者:** Cong Wang; Xianda Guo; Wenbo Xu; Wei Tian; Ruiqi Song; Chenming Zhang; Lingxi Li; Long Chen
>
> **摘要:** In the realm of driving scenarios, the presence of rapidly moving vehicles, pedestrians in motion, and large-scale static backgrounds poses significant challenges for 3D scene reconstruction. Recent methods based on 3D Gaussian Splatting address the motion blur problem by decoupling dynamic and static components within the scene. However, these decoupling strategies overlook background optimization with adequate geometry relationships and rely solely on fitting each training view by adding Gaussians. Therefore, these models exhibit limited robustness in rendering novel views and lack an accurate geometric representation. To address the above issues, we introduce DriveSplat, a high-quality reconstruction method for driving scenarios based on neural Gaussian representations with dynamic-static decoupling. To better accommodate the predominantly linear motion patterns of driving viewpoints, a region-wise voxel initialization scheme is employed, which partitions the scene into near, middle, and far regions to enhance close-range detail representation. Deformable neural Gaussians are introduced to model non-rigid dynamic actors, whose parameters are temporally adjusted by a learnable deformation network. The entire framework is further supervised by depth and normal priors from pre-trained models, improving the accuracy of geometric structures. Our method has been rigorously evaluated on the Waymo and KITTI datasets, demonstrating state-of-the-art performance in novel-view synthesis for driving scenarios.
>
---
#### [replaced 105] FOCUS: Unified Vision-Language Modeling for Interactive Editing Driven by Referential Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16806v2](http://arxiv.org/pdf/2506.16806v2)**

> **作者:** Fan Yang; Yousong Zhu; Xin Li; Yufei Zhan; Hongyin Zhao; Shurong Zheng; Yaowei Wang; Ming Tang; Jinqiao Wang
>
> **摘要:** Recent Large Vision Language Models (LVLMs) demonstrate promising capabilities in unifying visual understanding and generative modeling, enabling both accurate content understanding and flexible editing. However, current approaches treat "what to see" and "how to edit" separately: they either perform isolated object segmentation or utilize segmentation masks merely as conditional prompts for local edit generation tasks, often relying on multiple disjointed models. To bridge these gaps, we introduce FOCUS, a unified LVLM that integrates segmentation-aware perception and controllable object-centric generation within an end-to-end framework. FOCUS employs a dual-branch visual encoder to simultaneously capture global semantic context and fine-grained spatial details. In addition, we leverage a MoVQGAN-based visual tokenizer to produce discrete visual tokens that enhance generation quality. To enable accurate and controllable image editing, we propose a progressive multi-stage training pipeline, where segmentation masks are jointly optimized and used as spatial condition prompts to guide the diffusion decoder. This strategy aligns visual encoding, segmentation, and generation modules, effectively bridging segmentation-aware perception with fine-grained visual synthesis. Extensive experiments across three core tasks, including multimodal understanding, referring segmentation accuracy, and controllable image generation, demonstrate that FOCUS achieves strong performance by jointly optimizing visual perception and generative capabilities.
>
---
#### [replaced 106] Automatic Real-time Vehicle Classification by Image Colour Component Based Template Matching
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2210.06586v3](http://arxiv.org/pdf/2210.06586v3)**

> **作者:** Ahmet Orun
>
> **备注:** The paper may clash with another submission and dispute of copyright
>
> **摘要:** Selection of appropriate template matching algorithms to run effectively on real-time low-cost systems is always major issue. This is due to unpredictable changes in image scene which often necessitate more sophisticated real-time algorithms to retain image consistency. Inefficiency of low cost auxiliary hardware and time limitations are the major constraints in using these sorts of algorithms. The real-time system introduced here copes with these problems utilising a fast running template matching algorithm, which makes use of best colour band selection. The system uses fast running real-time algorithms to achieve template matching and vehicle classification at about 4 frames /sec. on low-cost hardware. The colour image sequences have been taken by a fixed CCTV camera overlooking a busy multi-lane road
>
---
#### [replaced 107] DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11196v2](http://arxiv.org/pdf/2505.11196v2)**

> **作者:** Yuang Ai; Qihang Fan; Xuefeng Hu; Zhenheng Yang; Ran He; Huaibo Huang
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Diffusion Transformer (DiT), a promising diffusion model for visual generation, demonstrates impressive performance but incurs significant computational overhead. Intriguingly, analysis of pre-trained DiT models reveals that global self-attention is often redundant, predominantly capturing local patterns-highlighting the potential for more efficient alternatives. In this paper, we revisit convolution as an alternative building block for constructing efficient and expressive diffusion models. However, naively replacing self-attention with convolution typically results in degraded performance. Our investigations attribute this performance gap to the higher channel redundancy in ConvNets compared to Transformers. To resolve this, we introduce a compact channel attention mechanism that promotes the activation of more diverse channels, thereby enhancing feature diversity. This leads to Diffusion ConvNet (DiCo), a family of diffusion models built entirely from standard ConvNet modules, offering strong generative performance with significant efficiency gains. On class-conditional ImageNet generation benchmarks, DiCo-XL achieves an FID of 2.05 at 256x256 resolution and 2.53 at 512x512, with a 2.7x and 3.1x speedup over DiT-XL/2, respectively. Furthermore, experimental results on MS-COCO demonstrate that the purely convolutional DiCo exhibits strong potential for text-to-image generation. Code: https://github.com/shallowdream204/DiCo.
>
---
#### [replaced 108] Ensemble YOLO Framework for Multi-Domain Mitotic Figure Detection in Histopathology Images
- **分类: eess.IV; cs.CV; 68T07; I.4.9; I.5.4**

- **链接: [http://arxiv.org/pdf/2509.02957v2](http://arxiv.org/pdf/2509.02957v2)**

> **作者:** Navya Sri Kelam; Akash Parekh; Saikiran Bonthu; Nitin Singhal
>
> **备注:** 4 pages, MIDOG25 Challenge
>
> **摘要:** The reliable identification of mitotic figures in whole-slide histopathological images remains difficult, owing to their low prevalence, substantial morphological heterogeneity, and the inconsistencies introduced by tissue processing and staining procedures. The MIDOG competition series provides standardized benchmarks for evaluating detection approaches across diverse domains, thus motivating the development of generalizable deep learning models. In this work, we investigate the performance of two modern one-stage detectors, YOLOv5 and YOLOv8, trained on MIDOG++, CMC, and CCMCT datasets. To enhance robustness, training incorporated stain-invariant color perturbations and texture-preserving augmentations. Ininternal validation, YOLOv5 achieved higher precision (84.3%), while YOLOv8 offered improved recall (82.6%), reflecting architectural trade-offs between anchor-based and anchor-free detections. To capitalize on their complementary strengths, weemployed an ensemble of the two models, which improved sensitivity (85.3%) while maintaining competitive precision, yielding the best F1 score of 83.1%. On the preliminary MIDOG 2025 test leaderboard, our ensemble ranked 5th with an F1 score of 79.2%, precision of 73.6%, and recall of 85.8%, confirming that the proposed strategy generalizes effectively across unseen test data. These findings highlight the effectiveness of combining anchor-based and anchor-free object detectors to advance automated mitosis detection in digital pathology.
>
---
#### [replaced 109] MS-YOLO: A Multi-Scale Model for Accurate and Efficient Blood Cell Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03972v2](http://arxiv.org/pdf/2506.03972v2)**

> **作者:** Guohua Wu; Shengqi Chen; Pengchao Deng; Wenting Yu
>
> **备注:** There is a disagreement among the authors regarding the content and submission of the manuscript, which needs to be resolved before it can be made public
>
> **摘要:** Complete blood cell detection holds significant value in clinical diagnostics. Conventional manual microscopy methods suffer from time inefficiency and diagnostic inaccuracies. Existing automated detection approaches remain constrained by high deployment costs and suboptimal accuracy. While deep learning has introduced powerful paradigms to this field, persistent challenges in detecting overlapping cells and multi-scale objects hinder practical deployment. This study proposes the multi-scale YOLO (MS-YOLO), a blood cell detection model based on the YOLOv11 framework, incorporating three key architectural innovations to enhance detection performance. Specifically, the multi-scale dilated residual module (MS-DRM) replaces the original C3K2 modules to improve multi-scale discriminability; the dynamic cross-path feature enhancement module (DCFEM) enables the fusion of hierarchical features from the backbone with aggregated features from the neck to enhance feature representations; and the light adaptive-weight downsampling module (LADS) improves feature downsampling through adaptive spatial weighting while reducing computational complexity. Experimental results on the CBC benchmark demonstrate that MS-YOLO achieves precise detection of overlapping cells and multi-scale objects, particularly small targets such as platelets, achieving an mAP@50 of 97.4% that outperforms existing models. Further validation on the supplementary WBCDD dataset confirms its robust generalization capability. Additionally, with a lightweight architecture and real-time inference efficiency, MS-YOLO meets clinical deployment requirements, providing reliable technical support for standardized blood pathology assessment.
>
---
#### [replaced 110] In-Context Edit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20690v2](http://arxiv.org/pdf/2504.20690v2)**

> **作者:** Zechuan Zhang; Ji Xie; Yu Lu; Zongxin Yang; Yi Yang
>
> **备注:** Accepted by NeurIPS 2025, there will be future updates for camera ready version. Code: https://github.com/River-Zhang/ICEdit
>
> **摘要:** Instruction-based image editing enables precise modifications via natural language prompts, but existing methods face a precision-efficiency tradeoff: fine-tuning demands massive datasets (>10M) and computational resources, while training-free approaches suffer from weak instruction comprehension. We address this by proposing ICEdit, which leverages the inherent comprehension and generation abilities of large-scale Diffusion Transformers (DiTs) through three key innovations: (1) An in-context editing paradigm without architectural modifications; (2) Minimal parameter-efficient fine-tuning for quality improvement; (3) Early Filter Inference-Time Scaling, which uses VLMs to select high-quality noise samples for efficiency. Experiments show that ICEdit achieves state-of-the-art editing performance with only 0.1\% of the training data and 1\% trainable parameters compared to previous methods. Our approach establishes a new paradigm for balancing precision and efficiency in instructional image editing. Codes and demos can be found in https://river-zhang.github.io/ICEdit-gh-pages/.
>
---
#### [replaced 111] SAM2-ELNet: Label Enhancement and Automatic Annotation for Remote Sensing Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12404v2](http://arxiv.org/pdf/2503.12404v2)**

> **作者:** Jianhao Yang; Wenshuo Yu; Yuanchao Lv; Jiance Sun; Bokang Sun; Mingyang Liu
>
> **备注:** published in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
>
> **摘要:** Remote sensing image segmentation is crucial for environmental monitoring, disaster assessment, and resource management, but its performance largely depends on the quality of the dataset. Although several high-quality datasets are broadly accessible, data scarcity remains for specialized tasks like marine oil spill segmentation. Such tasks still rely on manual annotation, which is both time-consuming and influenced by subjective human factors. The segment anything model 2 (SAM2) has strong potential as an automatic annotation framework but struggles to perform effectively on heterogeneous, low-contrast remote sensing imagery. To address these challenges, we introduce a novel label enhancement and automatic annotation framework, termed SAM2-ELNet (Enhancement and Labeling Network). Specifically, we employ the frozen Hiera backbone from the pretrained SAM2 as the encoder, while fine-tuning the adapter and decoder for different remote sensing tasks. In addition, the proposed framework includes a label quality evaluator for filtering, ensuring the reliability of the generated labels. We design a series of experiments targeting resource-limited remote sensing tasks and evaluate our method on two datasets: the Deep-SAR Oil Spill (SOS) dataset with Synthetic Aperture Radar (SAR) imagery, and the CHN6-CUG Road dataset with Very High Resolution (VHR) optical imagery. The proposed framework can enhance coarse annotations and generate reliable training data under resource-limited conditions. Fine-tuned on only 30% of the training data, it generates automatically labeled data. A model trained solely on these achieves slightly lower performance than using the full original annotations, while greatly reducing labeling costs and offering a practical solution for large-scale remote sensing interpretation.
>
---
#### [replaced 112] GarmentDiffusion: 3D Garment Sewing Pattern Generation with Multimodal Diffusion Transformers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21476v4](http://arxiv.org/pdf/2504.21476v4)**

> **作者:** Xinyu Li; Qi Yao; Yuanda Wang
>
> **备注:** The 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)
>
> **摘要:** Garment sewing patterns are fundamental design elements that bridge the gap between design concepts and practical manufacturing. The generative modeling of sewing patterns is crucial for creating diversified garments. However, existing approaches are limited either by reliance on a single input modality or by suboptimal generation efficiency. In this work, we present GarmentDiffusion, a new generative model capable of producing centimeter-precise, vectorized 3D sewing patterns from multimodal inputs (text, image, and incomplete sewing pattern). Our method efficiently encodes 3D sewing pattern parameters into compact edge token representations, achieving a sequence length that is 10 times shorter than that of the autoregressive SewingGPT in DressCode. By employing a diffusion transformer, we simultaneously denoise all edge tokens along the temporal axis, while maintaining a constant number of denoising steps regardless of dataset-specific edge and panel statistics. With all combination of designs of our model, the sewing pattern generation speed is accelerated by 100 times compared to SewingGPT. We achieve new state-of-the-art results on DressCodeData, as well as on the largest sewing pattern dataset, namely GarmentCodeData. The project website is available at https://shenfu-research.github.io/Garment-Diffusion/.
>
---
#### [replaced 113] DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases
- **分类: q-bio.NC; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01426v2](http://arxiv.org/pdf/2509.01426v2)**

> **作者:** Mo Wang; Kaining Peng; Jingsheng Tang; Hongkai Wen; Quanying Liu
>
> **备注:** Accepted as a poster at NeurIPS 2025 with scores 5554
>
> **摘要:** Brain atlases are essential for reducing the dimensionality of neuroimaging data and enabling interpretable analysis. However, most existing atlases are predefined, group-level templates with limited flexibility and resolution. We present Deep Cluster Atlas (DCA), a graph-guided deep embedding clustering framework for generating individualized, voxel-wise brain parcellations. DCA combines a pretrained autoencoder with spatially regularized deep clustering to produce functionally coherent and spatially contiguous regions. Our method supports flexible control over resolution and anatomical scope, and generalizes to arbitrary brain structures. We further introduce a standardized benchmarking platform for atlas evaluation, using multiple large-scale fMRI datasets. Across multiple datasets and scales, DCA outperforms state-of-the-art atlases, improving functional homogeneity by 98.8% and silhouette coefficient by 29%, and achieves superior performance in downstream tasks such as autism diagnosis and cognitive decoding. We also observe that a fine-tuned pretrained model achieves superior results on the corresponding task. Codes and models are available at https://github.com/ncclab-sustech/DCA .
>
---
#### [replaced 114] Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14899v2](http://arxiv.org/pdf/2504.14899v2)**

> **作者:** Chenjie Cao; Jingkai Zhou; Shikai Li; Jingyun Liang; Chaohui Yu; Fan Wang; Xiangyang Xue; Yanwei Fu
>
> **备注:** Project page: https://github.com/ewrfcas/Uni3C. Accepted by Siggraph Asian 2025
>
> **摘要:** Camera and human motion controls have been extensively studied for video generation, but existing approaches typically address them separately, suffering from limited data with high-quality annotations for both aspects. To overcome this, we present Uni3C, a unified 3D-enhanced framework for precise control of both camera and human motion in video generation. Uni3C includes two key contributions. First, we propose a plug-and-play control module trained with a frozen video generative backbone, PCDController, which utilizes unprojected point clouds from monocular depth to achieve accurate camera control. By leveraging the strong 3D priors of point clouds and the powerful capacities of video foundational models, PCDController shows impressive generalization, performing well regardless of whether the inference backbone is frozen or fine-tuned. This flexibility enables different modules of Uni3C to be trained in specific domains, i.e., either camera control or human motion control, reducing the dependency on jointly annotated data. Second, we propose a jointly aligned 3D world guidance for the inference phase that seamlessly integrates both scenic point clouds and SMPL-X characters to unify the control signals for camera and human motion, respectively. Extensive experiments confirm that PCDController enjoys strong robustness in driving camera motion for fine-tuned backbones of video generation. Uni3C substantially outperforms competitors in both camera controllability and human motion quality. Additionally, we collect tailored validation sets featuring challenging camera movements and human actions to validate the effectiveness of our method.
>
---
#### [replaced 115] Learning to Align: Addressing Character Frequency Distribution Shifts in Handwritten Text Recognition
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09846v2](http://arxiv.org/pdf/2506.09846v2)**

> **作者:** Panagiotis Kaliosis; John Pavlopoulos
>
> **备注:** EMNLP 2025 Findings, 18 pages, 10 figures, 11 tables
>
> **摘要:** Handwritten text recognition aims to convert visual input into machine-readable text, and it remains challenging due to the evolving and context-dependent nature of handwriting. Character sets change over time, and character frequency distributions shift across historical periods or regions, often causing models trained on broad, heterogeneous corpora to underperform on specific subsets. To tackle this, we propose a novel loss function that incorporates the Wasserstein distance between the character frequency distribution of the predicted text and a target distribution empirically derived from training data. By penalizing divergence from expected distributions, our approach enhances both accuracy and robustness under temporal and contextual intra-dataset shifts. Furthermore, we demonstrate that character distribution alignment can also improve existing models at inference time without requiring retraining by integrating it as a scoring function in a guided decoding scheme. Experimental results across multiple datasets and architectures confirm the effectiveness of our method in boosting generalization and performance. We open source our code at https://github.com/pkaliosis/fada.
>
---
#### [replaced 116] Convergence analysis of equilibrium methods for inverse problems
- **分类: math.NA; cs.CV; cs.NA**

- **链接: [http://arxiv.org/pdf/2306.01421v2](http://arxiv.org/pdf/2306.01421v2)**

> **作者:** Daniel Obmann; Gyeongha Hwang; Markus Haltmeier
>
> **摘要:** Solving inverse problems \(Ax = y\) is central to a variety of practically important fields such as medical imaging, remote sensing, and non-destructive testing. The most successful and theoretically best-understood method is convex variational regularization, where approximate but stable solutions are defined as minimizers of \( \|A(\cdot) - y^\delta\|^2 / 2 + \alpha \mathcal{R}(\cdot)\), with \(\mathcal{R}\) a regularization functional. Recent methods such as deep equilibrium models and plug-and-play approaches, however, go beyond variational regularization. Motivated by these innovations, we introduce implicit non-variational (INV) regularization, where approximate solutions are defined as solutions of \(A^*(A x - y^\delta) + \alpha R(x) = 0\) for some regularization operator \(R\). When the regularization operator is the gradient of a functional, INV reduces to classical variational regularization. However, in methods like DEQ and PnP, \(R\) is not a gradient field, and the existing theoretical foundation remains incomplete. To address this, we establish stability and convergence results in this broader setting, including convergence rates and stability estimates measured via a absolute Bregman distance.
>
---
#### [replaced 117] A Modular Robotic System for Autonomous Exploration and Semantic Updating in Large-Scale Indoor Environments
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15493v3](http://arxiv.org/pdf/2409.15493v3)**

> **作者:** Sai Haneesh Allu; Itay Kadosh; Tyler Summers; Yu Xiang
>
> **备注:** 10 pages, 9 figures, 5 tables. Project page is available at https://irvlutd.github.io/SemanticMapping/
>
> **摘要:** We present a modular robotic system for autonomous exploration and semantic updating of large-scale unknown environments. Our approach enables a mobile robot to build, revisit, and update a hybrid semantic map that integrates a 2D occupancy grid for geometry with a topological graph for object semantics. Unlike prior methods that rely on manual teleoperation or precollected datasets, our two-phase approach achieves end-to-end autonomy: first, a modified frontier-based exploration algorithm with dynamic search windows constructs a geometric map; second, using a greedy trajectory planner, environments are revisited, and object semantics are updated using open-vocabulary object detection and segmentation. This modular system, compatible with any metric SLAM framework, supports continuous operation by efficiently updating the semantic graph to reflect short-term and long-term changes such as object relocation, removal, or addition. We validate the approach on a Fetch robot in real-world indoor environments of approximately $8,500$m$^2$ and $117$m$^2$, demonstrating robust and scalable semantic mapping and continuous adaptation, marking a fully autonomous integration of exploration, mapping, and semantic updating on a physical robot.
>
---
#### [replaced 118] ReasonPlan: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO; 68T40(Primary), 68T45, 68T50(Secondary); I.2.9; I.2.10; I.5.1**

- **链接: [http://arxiv.org/pdf/2505.20024v2](http://arxiv.org/pdf/2505.20024v2)**

> **作者:** Xueyi Liu; Zuodong Zhong; Yuxin Guo; Yun-Fu Liu; Zhiguo Su; Qichao Zhang; Junli Wang; Yinfeng Gao; Yupeng Zheng; Qiao Lin; Huiyong Chen; Dongbin Zhao
>
> **备注:** 18 pages; 9 figures; https://github.com/Liuxueyi/ReasonPlan
>
> **摘要:** Due to the powerful vision-language reasoning and generalization abilities, multimodal large language models (MLLMs) have garnered significant attention in the field of end-to-end (E2E) autonomous driving. However, their application to closed-loop systems remains underexplored, and current MLLM-based methods have not shown clear superiority to mainstream E2E imitation learning approaches. In this work, we propose ReasonPlan, a novel MLLM fine-tuning framework designed for closed-loop driving through holistic reasoning with a self-supervised Next Scene Prediction task and supervised Decision Chain-of-Thought process. This dual mechanism encourages the model to align visual representations with actionable driving context, while promoting interpretable and causally grounded decision making. We curate a planning-oriented decision reasoning dataset, namely PDR, comprising 210k diverse and high-quality samples. Our method outperforms the mainstream E2E imitation learning method by a large margin of 19% L2 and 16.1 driving score on Bench2Drive benchmark. Furthermore, ReasonPlan demonstrates strong zero-shot generalization on unseen DOS benchmark, highlighting its adaptability in handling zero-shot corner cases. Code and dataset will be found in https://github.com/Liuxueyi/ReasonPlan.
>
---
#### [replaced 119] Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04039v2](http://arxiv.org/pdf/2506.04039v2)**

> **作者:** Jiulong Wu; Zhengliang Shi; Shuaiqiang Wang; Jizhou Huang; Dawei Yin; Lingyong Yan; Min Cao; Min Zhang
>
> **备注:** This paper is accepted by EMNLP2025
>
> **摘要:** Large Visual Language Models (LVLMs) have demonstrated impressive capabilities across multiple tasks. However, their trustworthiness is often challenged by hallucinations, which can be attributed to the modality misalignment and the inherent hallucinations of their underlying Large Language Models (LLMs) backbone. Existing preference alignment methods focus on aligning model responses with human preferences while neglecting image-text modality alignment, resulting in over-reliance on LLMs and hallucinations. In this paper, we propose Entity-centric Multimodal Preference Optimization (EMPO), which achieves enhanced modality alignment compared to existing human preference alignment methods. Besides, to overcome the scarcity of high-quality multimodal preference data, we utilize open-source instruction datasets to automatically construct high-quality preference data across three aspects: image, instruction, and response. Experiments on two human preference datasets and five multimodal hallucination benchmarks demonstrate the effectiveness of EMPO, e.g., reducing hallucination rates by 85.9\% on Object-HalBench and 49.8\% on MM-HalBench.
>
---
#### [replaced 120] Side Effects of Erasing Concepts from Diffusion Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.15124v3](http://arxiv.org/pdf/2508.15124v3)**

> **作者:** Shaswati Saha; Sourajit Saha; Manas Gaur; Tejas Gokhale
>
> **备注:** Findings of the Association for Computational Linguistics: EMNLP 2025
>
> **摘要:** Concerns about text-to-image (T2I) generative models infringing on privacy, copyright, and safety have led to the development of concept erasure techniques (CETs). The goal of an effective CET is to prohibit the generation of undesired "target" concepts specified by the user, while preserving the ability to synthesize high-quality images of other concepts. In this work, we demonstrate that concept erasure has side effects and CETs can be easily circumvented. For a comprehensive measurement of the robustness of CETs, we present the Side Effect Evaluation (SEE) benchmark that consists of hierarchical and compositional prompts describing objects and their attributes. The dataset and an automated evaluation pipeline quantify side effects of CETs across three aspects: impact on neighboring concepts, evasion of targets, and attribute leakage. Our experiments reveal that CETs can be circumvented by using superclass-subclass hierarchy, semantically similar prompts, and compositional variants of the target. We show that CETs suffer from attribute leakage and a counterintuitive phenomenon of attention concentration or dispersal. We release our benchmark and evaluation tools to aid future work on robust concept erasure.
>
---
#### [replaced 121] CameraVDP: Perceptual Display Assessment with Uncertainty Estimation via Camera and Visual Difference Prediction
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.08947v2](http://arxiv.org/pdf/2509.08947v2)**

> **作者:** Yancheng Cai; Robert Wanat; Rafal Mantiuk
>
> **备注:** Accepted by SIGGRAPH Asia 2025
>
> **摘要:** Accurate measurement of images produced by electronic displays is critical for the evaluation of both traditional and computational displays. Traditional display measurement methods based on sparse radiometric sampling and fitting a model are inadequate for capturing spatially varying display artifacts, as they fail to capture high-frequency and pixel-level distortions. While cameras offer sufficient spatial resolution, they introduce optical, sampling, and photometric distortions. Furthermore, the physical measurement must be combined with a model of a visual system to assess whether the distortions are going to be visible. To enable perceptual assessment of displays, we propose a combination of a camera-based reconstruction pipeline with a visual difference predictor, which account for both the inaccuracy of camera measurements and visual difference prediction. The reconstruction pipeline combines HDR image stacking, MTF inversion, vignetting correction, geometric undistortion, homography transformation, and color correction, enabling cameras to function as precise display measurement instruments. By incorporating a Visual Difference Predictor (VDP), our system models the visibility of various stimuli under different viewing conditions for the human visual system. We validate the proposed CameraVDP framework through three applications: defective pixel detection, color fringing awareness, and display non-uniformity evaluation. Our uncertainty analysis framework enables the estimation of the theoretical upper bound for defect pixel detection performance and provides confidence intervals for VDP quality scores.
>
---
#### [replaced 122] QuizRank: Picking Images by Quizzing VLMs
- **分类: cs.HC; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.15059v2](http://arxiv.org/pdf/2509.15059v2)**

> **作者:** Tenghao Ji; Eytan Adar
>
> **摘要:** Images play a vital role in improving the readability and comprehension of Wikipedia articles by serving as `illustrative aids.' However, not all images are equally effective and not all Wikipedia editors are trained in their selection. We propose QuizRank, a novel method of image selection that leverages large language models (LLMs) and vision language models (VLMs) to rank images as learning interventions. Our approach transforms textual descriptions of the article's subject into multiple-choice questions about important visual characteristics of the concept. We utilize these questions to quiz the VLM: the better an image can help answer questions, the higher it is ranked. To further improve discrimination between visually similar items, we introduce a Contrastive QuizRank that leverages differences in the features of target (e.g., a Western Bluebird) and distractor concepts (e.g., Mountain Bluebird) to generate questions. We demonstrate the potential of VLMs as effective visual evaluators by showing a high congruence with human quiz-takers and an effective discriminative ranking of images.
>
---
#### [replaced 123] SCORP: Scene-Consistent Object Refinement via Proxy Generation and Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23835v2](http://arxiv.org/pdf/2506.23835v2)**

> **作者:** Ziwei Chen; Ziling Liu; Zitong Huang; Mingqi Gao; Feng Zheng
>
> **备注:** 8 pages with 6 figures. Project page: https://polysummit.github.io/scorp.github.io/
>
> **摘要:** Viewpoint missing of objects is common in scene reconstruction, as camera paths typically prioritize capturing the overall scene structure rather than individual objects. This makes it highly challenging to achieve high-fidelity object-level modeling while maintaining accurate scene-level representation. Addressing this issue is critical for advancing downstream tasks requiring high-fidelity object reconstruction. In this paper, we introduce Scene-Consistent Object Refinement via Proxy Generation and Tuning (SCORP), a novel 3D enhancement framework that leverages 3D generative priors to recover fine-grained object geometry and appearance under missing views. Starting with proxy generation by substituting degraded objects using a 3D generation model, SCORP then progressively refines geometry and texture by aligning each proxy to its degraded counterpart in 7-DoF pose, followed by correcting spatial and appearance inconsistencies through registration-constrained enhancement. This two-stage proxy tuning ensures the high-fidelity geometry and appearance of the original object in unseen views while maintaining consistency in spatial positioning, observed geometry, and appearance. Across challenging benchmarks, SCORP achieves consistent gains over recent state-of-the-art baselines on both novel view synthesis and geometry completion tasks. SCORP is available at https://github.com/PolySummit/SCORP.
>
---
#### [replaced 124] Conformal In-Context Reverse Classification Accuracy: Efficient Estimation of Segmentation Quality with Statistical Guarantees
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04522v3](http://arxiv.org/pdf/2503.04522v3)**

> **作者:** Matias Cosarinsky; Ramiro Billot; Lucas Mansilla; Gabriel Jimenez; Nicolas Gaggión; Guanghui Fu; Tom Tirer; Enzo Ferrante
>
> **摘要:** Assessing the quality of automatic image segmentation is crucial in clinical practice, but often very challenging due to the limited availability of ground truth annotations. Reverse Classification Accuracy (RCA) is an approach that estimates the quality of new predictions on unseen samples by training a segmenter on those predictions, and then evaluating it against existing annotated images. In this work, we introduce Conformal In-Context RCA, a novel method for automatically estimating segmentation quality with statistical guarantees in the absence of ground-truth annotations, which consists of two main innovations. First, In-Context RCA, which leverages recent in-context learning models for image segmentation and incorporates retrieval-augmentation techniques to select the most relevant reference images. This approach enables efficient quality estimation with minimal reference data while avoiding the need of training additional models. Second, we introduce Conformal RCA, which extends both the original RCA framework and In-Context RCA to go beyond point estimation. Using tools from split conformal prediction, Conformal RCA produces prediction intervals for segmentation quality providing statistical guarantees that the true score lies within the estimated interval with a user-specified probability. Validated across 10 different medical imaging tasks in various organs and modalities, our methods demonstrate robust performance and computational efficiency, offering a promising solution for automated quality control in clinical workflows, where fast and reliable segmentation assessment is essential. The code is available at https://github.com/mcosarinsky/Conformal-In-Context-RCA.
>
---
#### [replaced 125] On the Suitability of Reinforcement Fine-Tuning to Visual Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.05682v2](http://arxiv.org/pdf/2504.05682v2)**

> **作者:** Xiaxu Chen; Wei Li; Chunxu Liu; Chi Xie; Xiaoyan Hu; Chengqian Ma; Feng Zhu; Rui Zhao
>
> **摘要:** Reinforcement Fine-Tuning (RFT) is proved to be greatly valuable for enhancing the reasoning ability of LLMs. Researchers have been starting to apply RFT to MLLMs, hoping it will also enhance the capabilities of visual understanding. However, these works are at a very early stage and have not examined how suitable RFT actually is for visual tasks. In this work, we endeavor to understand the suitabilities and limitations of RFT for visual tasks, through experimental analysis and observations. We start by quantitative comparisons on various tasks, which shows RFT is generally better than SFT on visual tasks. %especially when the number of training samples are limited. To check whether such advantages are brought up by the reasoning process, we design a new reward that encourages the model to ``think'' more, whose results show more thinking can be beneficial for complicated tasks but harmful for simple tasks. We hope this study can provide more insight for the rapid advancements on this topic.
>
---
#### [replaced 126] TextOCVP: Object-Centric Video Prediction with Language Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11655v2](http://arxiv.org/pdf/2502.11655v2)**

> **作者:** Angel Villar-Corrales; Gjergj Plepi; Sven Behnke
>
> **摘要:** Understanding and forecasting future scene states is critical for autonomous agents to plan and act effectively in complex environments. Object-centric models, with structured latent spaces, have shown promise in modeling object dynamics and predicting future scene states, but often struggle to scale beyond simple synthetic datasets and to integrate external guidance, limiting their applicability in robotics. To address these limitations, we propose TextOCVP, an object-centric model for video prediction guided by textual descriptions. TextOCVP parses an observed scene into object representations, called slots, and utilizes a text-conditioned transformer predictor to forecast future object states and video frames. Our approach jointly models object dynamics and interactions while incorporating textual guidance, enabling accurate and controllable predictions. TextOCVP's structured latent space offers a more precise control of the forecasting process, outperforming several video prediction baselines on two datasets. Additionally, we show that structured object-centric representations provide superior robustness to novel scene configurations, as well as improved controllability and interpretability, enabling more precise and understandable predictions. Videos and code are available at https://play-slot.github.io/TextOCVP.
>
---
#### [replaced 127] Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20134v4](http://arxiv.org/pdf/2502.20134v4)**

> **作者:** Itay Benou; Tammy Riklin-Raviv
>
> **摘要:** Modern deep neural networks have now reached human-level performance across a variety of tasks. However, unlike humans they lack the ability to explain their decisions by showing where and telling what concepts guided them. In this work, we present a unified framework for transforming any vision neural network into a spatially and conceptually interpretable model. We introduce a spatially-aware concept bottleneck layer that projects "black-box" features of pre-trained backbone models into interpretable concept maps, without requiring human labels. By training a classification layer over this bottleneck, we obtain a self-explaining model that articulates which concepts most influenced its prediction, along with heatmaps that ground them in the input image. Accordingly, we name this method "Spatially-Aware and Label-Free Concept Bottleneck Model" (SALF-CBM). Our results show that the proposed SALF-CBM: (1) Outperforms non-spatial CBM methods, as well as the original backbone, on a variety of classification tasks; (2) Produces high-quality spatial explanations, outperforming widely used heatmap-based methods on a zero-shot segmentation task; (3) Facilitates model exploration and debugging, enabling users to query specific image regions and refine the model's decisions by locally editing its concept maps.
>
---
#### [replaced 128] Alias-Free Latent Diffusion Models: Improving Fractional Shift Equivariance of Diffusion Latent Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09419v2](http://arxiv.org/pdf/2503.09419v2)**

> **作者:** Yifan Zhou; Zeqi Xiao; Shuai Yang; Xingang Pan
>
> **备注:** Code is available at: https://github.com/SingleZombie/AFLDM
>
> **摘要:** Latent Diffusion Models (LDMs) are known to have an unstable generation process, where even small perturbations or shifts in the input noise can lead to significantly different outputs. This hinders their applicability in applications requiring consistent results. In this work, we redesign LDMs to enhance consistency by making them shift-equivariant. While introducing anti-aliasing operations can partially improve shift-equivariance, significant aliasing and inconsistency persist due to the unique challenges in LDMs, including 1) aliasing amplification during VAE training and multiple U-Net inferences, and 2) self-attention modules that inherently lack shift-equivariance. To address these issues, we redesign the attention modules to be shift-equivariant and propose an equivariance loss that effectively suppresses the frequency bandwidth of the features in the continuous domain. The resulting alias-free LDM (AF-LDM) achieves strong shift-equivariance and is also robust to irregular warping. Extensive experiments demonstrate that AF-LDM produces significantly more consistent results than vanilla LDM across various applications, including video editing and image-to-image translation.
>
---
#### [replaced 129] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.15566v2](http://arxiv.org/pdf/2509.15566v2)**

> **作者:** Shaojie Zhang; Ruoceng Zhang; Pei Fu; Shaokang Wang; Jiahui Yang; Xin Du; Shiqi Cui; Bin Qin; Ying Huang; Zhenbo Luo; Jian Luan
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** In the field of AI-driven human-GUI interaction automation, while rapid advances in multimodal large language models and reinforcement fine-tuning techniques have yielded remarkable progress, a fundamental challenge persists: their interaction logic significantly deviates from natural human-GUI communication patterns. To fill this gap, we propose "Blink-Think-Link" (BTL), a brain-inspired framework for human-GUI interaction that mimics the human cognitive process between users and graphical interfaces. The system decomposes interactions into three biologically plausible phases: (1) Blink - rapid detection and attention to relevant screen areas, analogous to saccadic eye movements; (2) Think - higher-level reasoning and decision-making, mirroring cognitive planning; and (3) Link - generation of executable commands for precise motor control, emulating human action selection mechanisms. Additionally, we introduce two key technical innovations for the BTL framework: (1) Blink Data Generation - an automated annotation pipeline specifically optimized for blink data, and (2) BTL Reward -- the first rule-based reward mechanism that enables reinforcement learning driven by both process and outcome. Building upon this framework, we develop a GUI agent model named BTL-UI, which demonstrates consistent state-of-the-art performance across both static GUI understanding and dynamic interaction tasks in comprehensive benchmarks. These results provide conclusive empirical validation of the framework's efficacy in developing advanced GUI Agents.
>
---
#### [replaced 130] BaseBoostDepth: Exploiting Larger Baselines For Self-supervised Monocular Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.20437v2](http://arxiv.org/pdf/2407.20437v2)**

> **作者:** Kieran Saunders; Luis J. Manso; George Vogiatzis
>
> **摘要:** In the domain of multi-baseline stereo, the conventional understanding is that, in general, increasing baseline separation substantially enhances the accuracy of depth estimation. However, prevailing self-supervised depth estimation architectures primarily use minimal frame separation and a constrained stereo baseline. Larger frame separations can be employed; however, we show this to result in diminished depth quality due to various factors, including significant changes in brightness, and increased areas of occlusion. In response to these challenges, our proposed method, BaseBoostDepth, incorporates a curriculum learning-inspired optimization strategy to effectively leverage larger frame separations. However, we show that our curriculum learning-inspired strategy alone does not suffice, as larger baselines still cause pose estimation drifts. Therefore, we introduce incremental pose estimation to enhance the accuracy of pose estimations, resulting in significant improvements across all depth metrics. Additionally, to improve the robustness of the model, we introduce error-induced reconstructions, which optimize reconstructions with added error to the pose estimations. Ultimately, our final depth network achieves state-of-the-art performance on KITTI and SYNS-patches datasets across image-based, edge-based, and point cloud-based metrics without increasing computational complexity at test time. The project website can be found at https://kieran514.github.io/BaseBoostDepth-Project.
>
---
#### [replaced 131] PPORLD-EDNetLDCT: A Proximal Policy Optimization-Based Reinforcement Learning Framework for Adaptive Low-Dose CT Denoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.03185v2](http://arxiv.org/pdf/2509.03185v2)**

> **作者:** Debopom Sutradhar; Ripon Kumar Debnath; Mohaimenul Azam Khan Raiaan; Yan Zhang; Reem E. Mohamed; Sami Azam
>
> **备注:** 20 pages, 5 figures, 5 tables
>
> **摘要:** Low-dose computed tomography (LDCT) is critical for minimizing radiation exposure, but it often leads to increased noise and reduced image quality. Traditional denoising methods, such as iterative optimization or supervised learning, often fail to preserve image quality. To address these challenges, we introduce PPORLD-EDNetLDCT, a reinforcement learning-based (RL) approach with Encoder-Decoder for LDCT. Our method utilizes a dynamic RL-based approach in which an advanced posterior policy optimization (PPO) algorithm is used to optimize denoising policies in real time, based on image quality feedback, trained via a custom gym environment. The experimental results on the low dose CT image and projection dataset demonstrate that the proposed PPORLD-EDNetLDCT model outperforms traditional denoising techniques and other DL-based methods, achieving a peak signal-to-noise ratio of 41.87, a structural similarity index measure of 0.9814 and a root mean squared error of 0.00236. Moreover, in NIH-AAPM-Mayo Clinic Low Dose CT Challenge dataset our method achieved a PSNR of 41.52, SSIM of 0.9723 and RMSE of 0.0051. Furthermore, we validated the quality of denoising using a classification task in the COVID-19 LDCT dataset, where the images processed by our method improved the classification accuracy to 94%, achieving 4% higher accuracy compared to denoising without RL-based denoising.
>
---
#### [replaced 132] Block-Fused Attention-Driven Adaptively-Pooled ResNet Model for Improved Cervical Cancer Classification
- **分类: eess.IV; cs.CV; cs.LG; I.2.1; I.5.2**

- **链接: [http://arxiv.org/pdf/2405.01600v3](http://arxiv.org/pdf/2405.01600v3)**

> **作者:** Saurabh Saini; Kapil Ahuja; Akshat S. Chauhan
>
> **备注:** 32 Pages, 12 Tables, 14 Figures
>
> **摘要:** Cervical cancer is the second most common cancer among women and a leading cause of mortality. Many attempts have been made to develop an effective Computer Aided Diagnosis (CAD) system; however, their performance remains limited. Using pretrained ResNet-50/101/152, we propose a novel CAD system that significantly outperforms prior approaches. Our novel model has three key components. First, we extract detailed features (color, edges, and texture) from early convolution blocks and the abstract features (shapes and objects) from later blocks, as both are equally important. This dual-level feature extraction is a new paradigm in cancer classification. Second, a non-parametric 3D attention module is uniquely embedded within each block for feature enhancement. Third, we design a theoretically motivated innovative adaptive pooling strategy for feature selection that applies Global Max Pooling to detailed features and Global Average Pooling to abstract features. These components form our Proposed Block-Fused Attention-Driven Adaptively-Pooled ResNet (BF-AD-AP-ResNet) model. To further strengthen learning, we introduce a Tri-Stream model, which unifies the enhanced features from three BF-AD-AP-ResNets. An SVM classifier is employed for final classification. We evaluate our models on two public datasets, IARC and AnnoCerv. On IARC, the base ResNets achieve an average performance of 90.91%, while our model achieves an excellent performance of 98.63%. On AnnoCerv, the base ResNets reach to 87.68%, and our model improves this significantly, reaching 93.39%. Our approach outperforms the best existing method on IARC by an average of 14.55%. For AnnoCerv, no prior competitive works are available. Additionally, we introduce a novel SHAP+LIME explainability method, accurately identifying the cancerous region in 97% of cases, ensuring model reliability for real-world use.
>
---
#### [replaced 133] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15235v2](http://arxiv.org/pdf/2509.15235v2)**

> **作者:** Jialiang Kang; Han Shu; Wenshuo Li; Yingjie Zhai; Xinghao Chen
>
> **备注:** NeurIPS 2025
>
> **摘要:** Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), yet its application to vision-language models (VLMs) remains underexplored, with existing methods achieving only modest speedups (<1.5x). This gap is increasingly significant as multimodal capabilities become central to large-scale models. We hypothesize that large VLMs can effectively filter redundant image information layer by layer without compromising textual comprehension, whereas smaller draft models struggle to do so. To address this, we introduce Vision-Aware Speculative Decoding (ViSpec), a novel framework tailored for VLMs. ViSpec employs a lightweight vision adaptor module to compress image tokens into a compact representation, which is seamlessly integrated into the draft model's attention mechanism while preserving original image positional information. Additionally, we extract a global feature vector for each input image and augment all subsequent text tokens with this feature to enhance multimodal coherence. To overcome the scarcity of multimodal datasets with long assistant responses, we curate a specialized training dataset by repurposing existing datasets and generating extended outputs using the target VLM with modified prompts. Our training strategy mitigates the risk of the draft model exploiting direct access to the target model's hidden states, which could otherwise lead to shortcut learning when training solely on target model outputs. Extensive experiments validate ViSpec, achieving, to our knowledge, the first substantial speedup in VLM speculative decoding. Code is available at https://github.com/KangJialiang/ViSpec.
>
---
#### [replaced 134] Unsupervised Interpretable Basis Extraction for Concept-Based Visual Explanations
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2303.10523v3](http://arxiv.org/pdf/2303.10523v3)**

> **作者:** Alexandros Doumanoglou; Stylianos Asteriadis; Dimitrios Zarpalas
>
> **备注:** 15 pages, Original version accepted to IEEE Transactions on Artificial Intelligence, Special Issue on New Developments in Explainable and Interpretable AI, This version contains improvements in the presentation style
>
> **摘要:** An important line of research attempts to explain CNN image classifier predictions and intermediate layer representations in terms of human-understandable concepts. Previous work supports that deep representations are linearly separable with respect to their concept label, implying that the feature space has directions where intermediate representations may be projected onto, to become more understandable. These directions are called interpretable, and when considered as a set, they may form an interpretable feature space basis. Compared to previous top-down probing approaches which use concept annotations to identify the interpretable directions one at a time, in this work, we take a bottom-up approach, identifying the directions from the structure of the feature space, collectively, without relying on supervision from concept labels. Instead, we learn the directions by optimizing for a sparsity property that holds for any interpretable basis. We experiment with existing popular CNNs and demonstrate the effectiveness of our method in extracting an interpretable basis across network architectures and training datasets. We make extensions to existing basis interpretability metrics and show that intermediate layer representations become more interpretable when transformed with the extracted bases. Finally, we compare the bases extracted with our method with the bases derived with supervision and find that, in one aspect, unsupervised basis extraction has a strength that constitutes a limitation of learning the basis with supervision, and we provide potential directions for future research.
>
---
#### [replaced 135] HyperTTA: Test-Time Adaptation for Hyperspectral Image Classification under Distribution Shifts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.08436v2](http://arxiv.org/pdf/2509.08436v2)**

> **作者:** Xia Yue; Anfeng Liu; Ning Chen; Chenjia Huang; Hui Liu; Zhou Huang; Leyuan Fang
>
> **摘要:** Hyperspectral image (HSI) classification models are highly sensitive to distribution shifts caused by real-world degradations such as noise, blur, compression, and atmospheric effects. To address this challenge, we propose HyperTTA (Test-Time Adaptable Transformer for Hyperspectral Degradation), a unified framework that enhances model robustness under diverse degradation conditions. First, we construct a multi-degradation hyperspectral benchmark that systematically simulates nine representative degradations, enabling comprehensive evaluation of robust classification. Based on this benchmark, we develop a Spectral--Spatial Transformer Classifier (SSTC) with a multi-level receptive field mechanism and label smoothing regularization to capture multi-scale spatial context and improve generalization. Furthermore, we introduce a lightweight test-time adaptation strategy, the Confidence-aware Entropy-minimized LayerNorm Adapter (CELA), which dynamically updates only the affine parameters of LayerNorm layers by minimizing prediction entropy on high-confidence unlabeled target samples. This strategy ensures reliable adaptation without access to source data or target labels. Experiments on two benchmark datasets demonstrate that HyperTTA outperforms state-of-the-art baselines across a wide range of degradation scenarios. Code will be made available publicly.
>
---
#### [replaced 136] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.11653v2](http://arxiv.org/pdf/2506.11653v2)**

> **作者:** Emre Kavak; Tom Nuno Wolf; Christian Wachinger
>
> **摘要:** Dataset bias often leads deep learning models to exploit spurious correlations instead of task-relevant signals. We introduce the Standard Anti-Causal Model (SAM), a unifying causal framework that characterizes bias mechanisms and yields a conditional independence criterion for causal stability. Building on this theory, we propose DISCO$_m$ and sDISCO, efficient and scalable estimators of conditional distance correlation that enable independence regularization in black-box models. Across five diverse datasets, our methods consistently outperform or are competitive in existing bias mitigation approaches, while requiring fewer hyperparameters and scaling seamlessly to multi-bias scenarios. This work bridges causal theory and practical deep learning, providing both a principled foundation and effective tools for robust prediction. Source Code: https://github.com/***.
>
---
#### [replaced 137] SWA-PF: Semantic-Weighted Adaptive Particle Filter for Memory-Efficient 4-DoF UAV Localization in GNSS-Denied Environments
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.13795v2](http://arxiv.org/pdf/2509.13795v2)**

> **作者:** Jiayu Yuan; Ming Dai; Enhui Zheng; Chao Su; Nanxing Chen; Qiming Hu; Shibo Zhu; Yibin Cao
>
> **摘要:** Vision-based Unmanned Aerial Vehicle (UAV) localization systems have been extensively investigated for Global Navigation Satellite System (GNSS)-denied environments. However, existing retrieval-based approaches face limitations in dataset availability and persistent challenges including suboptimal real-time performance, environmental sensitivity, and limited generalization capability, particularly in dynamic or temporally varying environments. To overcome these limitations, we present a large-scale Multi-Altitude Flight Segments dataset (MAFS) for variable altitude scenarios and propose a novel Semantic-Weighted Adaptive Particle Filter (SWA-PF) method. This approach integrates robust semantic features from both UAV-captured images and satellite imagery through two key innovations: a semantic weighting mechanism and an optimized particle filtering architecture. Evaluated using our dataset, the proposed method achieves 10x computational efficiency gain over feature extraction methods, maintains global positioning errors below 10 meters, and enables rapid 4 degree of freedom (4-DoF) pose estimation within seconds using accessible low-resolution satellite maps. Code and dataset will be available at https://github.com/YuanJiayuuu/SWA-PF.
>
---
#### [replaced 138] The Sound of Simulation: Learning Multimodal Sim-to-Real Robot Policies with Generative Audio
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02864v2](http://arxiv.org/pdf/2507.02864v2)**

> **作者:** Renhao Wang; Haoran Geng; Tingle Li; Feishi Wang; Gopala Anumanchipalli; Trevor Darrell; Boyi Li; Pieter Abbeel; Jitendra Malik; Alexei A. Efros
>
> **备注:** Conference on Robot Learning 2025
>
> **摘要:** Robots must integrate multiple sensory modalities to act effectively in the real world. Yet, learning such multimodal policies at scale remains challenging. Simulation offers a viable solution, but while vision has benefited from high-fidelity simulators, other modalities (e.g. sound) can be notoriously difficult to simulate. As a result, sim-to-real transfer has succeeded primarily in vision-based tasks, with multimodal transfer still largely unrealized. In this work, we tackle these challenges by introducing MultiGen, a framework that integrates large-scale generative models into traditional physics simulators, enabling multisensory simulation. We showcase our framework on the dynamic task of robot pouring, which inherently relies on multimodal feedback. By synthesizing realistic audio conditioned on simulation video, our method enables training on rich audiovisual trajectories -- without any real robot data. We demonstrate effective zero-shot transfer to real-world pouring with novel containers and liquids, highlighting the potential of generative modeling to both simulate hard-to-model modalities and close the multimodal sim-to-real gap.
>
---
