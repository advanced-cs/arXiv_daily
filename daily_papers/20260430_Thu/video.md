# 计算机视觉 cs.CV

- **最新发布 93 篇**

- **更新 73 篇**

## 最新发布

#### [new 001] SynSur: An end-to-end generative pipeline for synthetic industrial surface defect generation and detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于工业表面缺陷检测任务，旨在解决标注数据稀缺问题。通过合成缺陷生成与标注管道，提升数据量并增强模型性能。**

- **链接: [https://arxiv.org/pdf/2604.26633](https://arxiv.org/pdf/2604.26633)**

> **作者:** Paul Julius Kühn; Mika Pommeranz; Arjan Kuijper; Saptarshi Neil Sinha
>
> **摘要:** The bottleneck in learning-based industrial defect detection is often limited not by model capacity, but by the scarcity of labeled defect data: defects are rare, annotations are expensive, and collecting balanced training sets is slow. We present an end-to-end pipeline for synthetic defect generation and annotation, combining Vision-Language-Model-based prompts, LoRA-adapted diffusion, mask-guided inpainting, and sample filtering with automatic label derivation, and demonstrates the potential of real data with realistic synthetic samples to overcome data scarcity. The evaluation is conducted on, a challenging dataset of pitting defects on ball screw drives, and then on a subset of the Mobile phone screen surface defect segmentation dataset (MSD) dataset to test cross-domain transfer. Beyond downstream detector performance, we analyze key stages of the pipeline, including prompt construction, LoRA selection, and sample filtering with DreamSim and CLIPScore, to understand which synthetic samples are both realistic and useful. Experiments with YOLOv26, YOLOX, and LW-DETR show that synthetic-only training does not replace real data. When combined with real data, synthetic defects can preserve performance and yield modest gains in selected BSData training regimes. The MSD transfer study shows that the overall pipeline structure carries over to a second industrial inspection domain, while also highlighting the importance of domain-specific adaptation and annotation-quality control. Overall, the paper provides an end-to-end assessment of diffusion-based industrial defect synthesis and shows that its strongest value lies in strengthening scarce real datasets rather than substituting for them.
>
---
#### [new 002] MixerCA: An Efficient and Accurate Model for High-Performance Hyperspectral Image Classification
- **分类: cs.CV**

- **简介: 本文提出MixerCA模型，用于高光谱图像分类任务，解决传统方法在处理复杂空间和光谱特征上的不足。通过深度卷积和自注意力机制，提升分类效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.26138](https://arxiv.org/pdf/2604.26138)**

> **作者:** Mohammed Q. Alkhatib; Ali Jamali
>
> **备注:** Preprint accepted for publication in "Remote Sensing Applications: Society and Environment" Journal
>
> **摘要:** Over the past decade, hyperspectral image (HSI) classification has drawn considerable interest due to HSIs' ability to effectively distinguish terrestrial objects by capturing detailed, continuous spectral information. The strong performance of recent deep learning techniques in tasks like image classification and semantic segmentation has led to their growing use in HSI classification, due to their ability to capture complex spatial and spectral features more effectively than traditional methods. This paper presents MixerCA, a novel lightweight model for HSI classification that leverages depthwise convolution and a self-attention mechanism. MixerCA integrates depth-wise convolutions, token and channel mixing, and coordinate attention into a unified structure to decouple spatial and channel interactions, maintain consistent resolution throughout the network, and directly process HSI patches. Extensive experiments on four hyperspectral benchmark datasets reveal MixerCA's clear advantages over several competing algorithms, including 2D-CNN, 3D-CNN, Tri-CNN, HybridSN, ViT, and Swin Transformer. The source code is publicly available at this https URL.
>
---
#### [new 003] MesonGS++: Post-training Compression of 3D Gaussian Splatting with Hyperparameter Searching
- **分类: cs.CV; cs.GR; cs.MM**

- **简介: 该论文属于3D重建任务，解决3D高斯点云存储成本高的问题。提出MesonGS++，通过联合优化剪枝、编码和量化等技术，实现高效压缩并精确控制存储大小。**

- **链接: [https://arxiv.org/pdf/2604.26799](https://arxiv.org/pdf/2604.26799)**

> **作者:** Shuzhao Xie; Junchen Ge; Weixiang Zhang; Jiahang Liu; Chen Tang; Yunpeng Bai; Shijia Ge; Jingyan Jiang; Yuzhi Huang; Fengnian Yang; Cong Zhang; Xiaoyi Fan; Zhi Wang
>
> **备注:** this https URL
>
> **摘要:** 3D Gaussian Splatting (3DGS) achieves high-quality novel view synthesis with real-time rendering, but its storage cost remains prohibitive for practical deployment. Existing post-training compression methods still rely on many coupled hyperparameters across pruning, transformation, quantization, and entropy coding, making it difficult to control the final compressed size and fully exploit the rate-distortion trade-off. We propose MesonGS++, a size-aware post-training codec for 3D Gaussian compression. On the codec side, MesonGS++ combines joint importance-based pruning, octree geometry coding, attribute transformation, selective vector quantization for higher-degree spherical harmonics, and group-wise mixed-precision quantization with entropy coding. On the configuration side, it treats the reserve ratio and bit-width allocation as the dominant rate-distortion knobs and jointly optimizes them under a target storage budget via discrete sampling and 0--1 integer linear programming. We further propose a linear size estimator and a CUDA parallel quantization operator to accelerate the hyperparameter searching process. Extensive experiments show that MesonGS++ achieves over 34$\times$ compression while preserving rendering fidelity, outperforming state-of-the-art post-training methods and accurately meeting target size budgets. Remarkably, without any training, MesonGS++ can even surpass the PSNR of vanilla 3DGS at a 20$\times$ compression rate on the Stump scene. Our code is available at this https URL
>
---
#### [new 004] DenseStep2M: A Scalable, Training-Free Pipeline for Dense Instructional Video Annotation
- **分类: cs.CV**

- **简介: 该论文提出DenseStep2M，一个无需训练的管道，用于从指令视频中提取高质量步骤标注，解决视频理解中的时间对齐与活动推理问题。**

- **链接: [https://arxiv.org/pdf/2604.26565](https://arxiv.org/pdf/2604.26565)**

> **作者:** Mingji Ge; Qirui Chen; Zeqian Li; Weidi Xie
>
> **摘要:** Long-term video understanding requires interpreting complex temporal events and reasoning over procedural activities. While instructional video corpora, like HowTo100M, offer rich resources for model training, they present significant challenges, including noisy ASR transcripts and inconsistent temporal alignments between narration and visual content. In this work, we introduce an automated, training-free pipeline to extract high-quality procedural annotations from in-the-wild instructional videos. Our approach segments videos into coherent shots, filters poorly aligned content, and leverages state-of-the-art multimodal and large language models (Qwen2.5-VL and DeepSeek-R1) to generate structured, temporally grounded procedural steps. This pipeline yields DenseStep2M, a large-scale dataset comprising approximately 100K videos and 2M detailed instructional steps, designed to support comprehensive long-form video understanding. To rigorously evaluate our pipeline, we curate DenseCaption100, a benchmark of high-quality, human-written captions. Evaluations demonstrate strong alignment between our auto-generated steps and human annotations. Furthermore, we validate the utility of DenseStep2M across three core downstream tasks: dense video captioning, procedural step grounding, and cross-modal retrieval. Models fine-tuned on DenseStep2M achieve substantial gains in captioning quality and temporal localization, while exhibiting robust zero-shot generalization across egocentric, exocentric, and mixed-perspective domains. These results underscore the effectiveness of DenseStep2M in facilitating advanced multimodal alignment and long-term activity reasoning. Our dataset is available at this https URL.
>
---
#### [new 005] Report of the 5th PVUW Challenge: Towards More Diverse Modalities in Pixel-Level Understanding
- **分类: cs.CV**

- **简介: 该论文报告了PVUW挑战赛的最新成果，聚焦像素级视频理解任务，解决复杂场景下的目标跟踪与定位问题，提出多模态解决方案。**

- **链接: [https://arxiv.org/pdf/2604.26031](https://arxiv.org/pdf/2604.26031)**

> **作者:** Chang Liu; Henghui Ding; Nikhila Ravi; Yunchao Wei; Shuting He; Song Bai; Philip Torr; Leilei Cao; Jinrong Zhang; Deshui Miao; Xusheng He; Dengxian Gong; Zhiyu Wang; Mingqi Gao; Jihwan Hong; Canyang Wu; Weili Guan; Jianlong Wu; Liqiang Nie; Xingsen Huang; Yameng Gu; Xiaogang Yu; Xin Li; Ming-Hsuan Yang; Sijie Li; Jungong Han; Quanzhu Niu; Shihao Chen; Yuanzheng Wu; Yikang Zhou; Tao Zhang; Haobo Yuan; Lu Qi; Shunping Ji; Chao Yang; Chao Tian; Guoqing Zhu; Kai Yang; Zhifan Mo; Haijun Zhang; Xudong Kang; Shutao Li; Jaeyoung Do
>
> **备注:** Official Report of the 5th PVUW Challenge on CVPR 2026
>
> **摘要:** This report summarizes the objectives, datasets, and top-performing methodologies of the 2026 Pixel-level Video Understanding in the Wild (PVUW) Challenge, hosted at CVPR 2026, which evaluates state-of-the-art models under highly unconstrained conditions. To provide a comprehensive assessment, the 2026 edition features three specialized tracks: the MOSE track for tracking objects within densely cluttered and severely occluded scenarios; the MeViS-Text track for localizing targets via motion-focused linguistic expressions; and the newly inaugurated MeViS-Audio track, which pioneers acoustic-driven object segmentation. By introducing previously unreleased challenging data and analyzing the cutting-edge, multimodal solutions submitted by participants, this report highlights the community's latest technical advancements and charts promising future directions for robust video scene comprehension.
>
---
#### [new 006] GaitKD: A Universal Decoupled Distillation Framework for Efficient Gait Recognition
- **分类: cs.CV**

- **简介: 该论文属于行为识别任务，旨在解决高效步态识别问题。提出GaitKD框架，通过解耦决策和边界知识迁移，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.26255](https://arxiv.org/pdf/2604.26255)**

> **作者:** Yuqi Li; Qian Zhou; Huiran Duan; Jingjie Wang; Shunli Zhang; Chuanguang Yang; Guoying Zhao; Yingli Tian
>
> **摘要:** Gait recognition is an attractive biometric modality for long-range and contact-free identification, but high-performing gait models often rely on deep and computationally expensive architectures that are difficult to deploy in practice. Knowledge distillation (KD) offers a natural way to transfer knowledge from a powerful teacher to an efficient student; however, standard KD is often less effective for part-structured gait models, where supervision is formed from both part-wise classification logits and part-wise retrieval embeddings. In this paper, we propose GaitKD, a distillation framework that decouples gait knowledge transfer into two complementary components: decision-level distillation and boundary-level distillation. Specifically, GaitKD aligns the teacher and student through part-calibrated logit distillation to transfer inter-class decision relations, while preserving the teacher-induced partitioning of the embedding space through an activation-boundary objective instead of direct feature regression. With a simple aligned part-wise design, GaitKD supports heterogeneous teacher-student gait models without introducing additional inference cost. Experimental results across multiple gait recognition benchmarks and teacher-student configurations show consistent improvements over strong gait baselines. Our study demonstrates that the two transfer components are complementary, and boundary-preserving distillation provides more stable performance than direct feature regression. Source code is available at this https URL
>
---
#### [new 007] GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents
- **分类: cs.CV**

- **简介: 该论文提出GLM-5V-Turbo，旨在构建多模态代理的原生基础模型。解决多模态感知与语言推理结合的问题，通过整合视觉、文本等多模态信息提升代理能力。**

- **链接: [https://arxiv.org/pdf/2604.26752](https://arxiv.org/pdf/2604.26752)**

> **作者:** GLM-V Team; Wenyi Hong; Xiaotao Gu; Ziyang Pan; Zhen Yang; Yuting Wang; Yue Wang; Yuanchang Yue; Yu Wang; Yanling Wang; Yan Wang; Xijun Liu; Wenmeng Yu; Weihan Wang; Wei Li; Shuaiqi Duan; Sheng Yang; Ruiliang Lv; Mingdao Liu; Lihang Pan; Ke Ning; Junhui Ji; Jinjiang Wang; Jing Chen; Jiazheng Xu; Jiale Zhu; Jiale Cheng; Ji Qi; Guobing Gan; Guo Wang; Cong Yao; Zijun Dou; Zihao Zhou; Zihan Wang; Zhiqi Ge; Zhijie Li; Zhenyu Hou; Zhao Xue; Zehui Wang; Zehai He; Yusen Liu; Yukuo Cen; Yuchen Li; Yuan Wang; Yijian Lu; Yanzi Wang; Yadong Xue; Xinyu Zhang; Xinyu Liu; Wenkai Li; Tianyu Tong; Tianshu Zhang; Shengdong Yan; Qinkai Zheng; Mingde Xu; Licheng Bao; Jiaxing Xu; Jiaxin Fan; Jiawen Qian; Jiali Chen; Jiahui Lin; Haozhi Zheng; Haoran Wang; Haochen Li; Fan Yang; Dan Zhang; Chuangxin Zhao; Chengcheng Wu; Boyan Shi; Bowei Jia; Baoxu Wang; Peng Zhang; Debing Liu; Bin Xu; Juanzi Li; Minlie Huang; Yuxiao Dong; Jie Tang
>
> **摘要:** We present GLM-5V-Turbo, a step toward native foundation models for multimodal agents. As foundation models are increasingly deployed in real environments, agentic capability depends not only on language reasoning, but also on the ability to perceive, interpret, and act over heterogeneous contexts such as images, videos, webpages, documents, GUIs. GLM-5V-Turbo is built around this objective: multimodal perception is integrated as a core component of reasoning, planning, tool use, and execution, rather than as an auxiliary interface to a language model. This report summarizes the main improvements behind GLM-5V-Turbo across model design, multimodal training, reinforcement learning, toolchain expansion, and integration with agent frameworks. These developments lead to strong performance in multimodal coding, visual tool use, and framework-based agentic tasks, while preserving competitive text-only coding capability. More importantly, our development process offers practical insights for building multimodal agents, highlighting the central role of multimodal perception, hierarchical optimization, and reliable end-to-end verification.
>
---
#### [new 008] Point Cloud Registration via Probabilistic Self-Update Local Correspondence and Line Vector Sets
- **分类: cs.CV**

- **简介: 该论文属于点云配准任务，解决3D数据融合问题。提出一种基于概率自更新局部对应和线向量集的快速有效算法，提升配准精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.26318](https://arxiv.org/pdf/2604.26318)**

> **作者:** Kuo-Liang Chung; Yu-Cheng Lin; Wu-Chi Chen
>
> **摘要:** Point cloud registration (PCR) is a fundamental task for integrating 3D observations in remote sensing applications. This paper proposes a fast and effective PCR algorithm utilizing probabilistic self-updating local correspondence and line vector sets. Our dual RANSAC interaction model comprises a global RANSAC evaluating the global correspondence set and a local RANSAC operating on dynamically updated local sets. Initially, these local sets are constructed using angle histogram statistics and line vector length preservation techniques. To improve accuracy, a probabilistic self-updating strategy refines the local sets after each interaction round. To reduce runtime, we introduce a global early termination condition that optimally balances accuracy and efficiency. Finally, a weighted singular value decomposition estimates the registration solution. Evaluations on public datasets demonstrate our algorithm achieves superior time efficiency and at least a 10% root mean square error improvement over state-of-the-art methods. The C++ source code is publicly available at this https URL.
>
---
#### [new 009] GateMOT: Q-Gated Attention for Dense Object Tracking
- **分类: cs.CV**

- **简介: 该论文提出GateMOT，解决密集目标跟踪中注意力机制计算成本高的问题。通过Q-Gated Attention实现高效特征选择，提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2604.26353](https://arxiv.org/pdf/2604.26353)**

> **作者:** Mingjin Lv; Zelin Liu; Feifei Shao; Yi-Ping Phoebe Chen; Junqing Yu; Wei Yang; Zikai Song
>
> **摘要:** While large models demonstrate the strong representational power of vanilla attention, this core mechanism cannot be directly applied to Dense Object Tracking: its quadratic all-to-all interactions are computationally prohibitive for dense motion estimation on high-resolution features. This mismatch prevents Dense Object Tracking from fully leveraging attention-based modeling in crowded and occlusion-heavy scenes. To address this challenge, we introduce GateMOT, an online tracking framework centered on Q-Gated Attention (Q-Attention), an efficient and spatially aware attention variant. Our key idea is to repurpose the Query from a similarity-conditioning term into a learnable gating unit. This Gating-Query (Gating-Q) produces a probabilistic gate that modulates Key features in an element-wise manner, enabling explicit relevance selection instead of costly global aggregation. Built on this mechanism, parallel Q-Attention heads transform one shared feature map into task-specific yet consistent representations for detection, motion, and re-identification, yielding a tightly coupled multi-task decoder with linear-complexity gating operations. GateMOT achieves state-of-the-art HOTA of 48.4, MOTA of 67.8, and IDF1 of 64.5 on BEE24, and demonstrates strong performance on additional Dense Object Tracking benchmarks. These results show that Q-Attention is a simple, effective, and transferable building block for attention-based tracking in dense tracking scenarios.
>
---
#### [new 010] Multi-Stage Bi-Atrial Segmentation Framework from 3D Late Gadolinium-Enhanced MRI using V-Net Family Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于心脏MRI图像分割任务，旨在解决3D LGE MRI中双心房的多类分割问题。通过多阶段V-Net模型实现粗细分割，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.26251](https://arxiv.org/pdf/2604.26251)**

> **作者:** Hao Wen; Jingsu Kang
>
> **备注:** 6 pages, 2 figures, technical report for participating the MBAS2024 challenge hosted on the MICCAI2024 conference
>
> **摘要:** We report our multi-stage framework designed for the problem of multi-class bi-atrial segmentation from 3D late gadolinium-enhanced (LGE) MRI of the human heart. The pipeline consists of a preprocessing step using multidimensional contrast limited adaptive histogram equalization (MCLAHE); coarse region segmentation from MCLAHE-enhanced and down-sampled MRI using a V-Net family model; and fine segmentation from the coarse region using another V-Net model. Asymmetric loss is adopted to optimize the model weights.
>
---
#### [new 011] Edge AI for Automotive Vulnerable Road User Safety: Deployable Detection via Knowledge Distillation
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文属于目标检测任务，旨在解决边缘设备上VRU安全检测的模型精度与计算约束矛盾。通过知识蒸馏方法，将大模型知识迁移至小模型，实现高效准确检测。**

- **链接: [https://arxiv.org/pdf/2604.26857](https://arxiv.org/pdf/2604.26857)**

> **作者:** Akshay Karjol; Darrin M. Hanna
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Deploying accurate object detection for Vulnerable Road User (VRU) safety on edge hardware requires balancing model capacity against computational constraints. Large models achieve high accuracy but fail under INT8 quantization required for edge deployment, while small models sacrifice detection performance. This paper presents a knowledge distillation (KD) framework that trains a compact YOLOv8-S student (11.2M parameters) to mimic a YOLOv8-L teacher (43.7M parameters), achieving 3.9x compression while preserving quantization robustness. We evaluate on full-scale BDD100K (70K training images) with Post-Training Quantization to INT8. The teacher suffers catastrophic degradation under INT8 (-23% mAP), while the KD student retains accuracy (-5.6% mAP). Analysis reveals that KD transfers precision calibration rather than raw detection capacity: the KD student achieves 0.748 precision versus 0.653 for direct training at INT8, a 14.5% gain at equivalent recall, reducing false alarms by 44% versus the collapsed teacher. At INT8, the KD student exceeds the teacher's FP32 precision (0.748 vs. 0.718) in a model 3.9x smaller. These findings establish knowledge distillation as a requirement for deploying accurate, safety-critical VRU detection on edge hardware.
>
---
#### [new 012] The Unseen Adversaries: Robust and Generalized Defense Against Adversarial Patches
- **分类: cs.CV**

- **简介: 该论文属于对抗样本防御任务，旨在解决物理世界中对抗补丁和自然噪声的联合威胁。工作包括构建新数据集，评估分类器有效性，发现独立防御效果不佳。**

- **链接: [https://arxiv.org/pdf/2604.26317](https://arxiv.org/pdf/2604.26317)**

> **作者:** Vishesh Kumar; Akshay Agarwal
>
> **备注:** Accepted at AISTATS 2026
>
> **摘要:** The vulnerabilities of deep neural networks against singularities have raised serious concerns regarding their deployment in the physical world. One of the most prominent and impactful physical-world adversarial perturbations is the attachment of patches to clean images, known as an adversarial patch attack. Similarly, natural noises such as Gaussian and Salt\&Pepper are highly prevalent in the real world. The current research need arises from the above vulnerabilities and the lack of efforts to tackle these two singularities independently and, especially, in combination. In this research, we have, for the first time, combined these two prominent singularities and proposed a novel dataset. Using this dataset, we have conducted a benchmark study of singularity data-point detection using features from several convolutional neural networks. For classification, rather than the popular neural network-based parameter tuning, we have used traditional yet effective machine learning classifiers. The extensive experiments across various in- and out-of-distribution (OOD) singularities reveal several interesting findings about the effectiveness of classifiers and show that it is hard to defend against adversaries when they are treated independently, and inefficient classifiers are selected.
>
---
#### [new 013] CheXthought: A global multimodal dataset of clinical chain-of-thought reasoning and visual attention for chest X-ray interpretation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CheXthought数据集，用于胸部X光解读的多模态临床推理研究，解决AI在医学影像分析中缺乏真实推理过程和视觉注意机制的问题。**

- **链接: [https://arxiv.org/pdf/2604.26288](https://arxiv.org/pdf/2604.26288)**

> **作者:** Sonali Sharma; Jin Long; George Shih; Sarah Eid; Christian Bluethgen; Francine L. Jacobson; Emily B. Tsai; Global Radiology Consortium; Ahmed M. Alaa; Curtis P. Langlotz
>
> **备注:** 51 pages, 7 figures, 10 tables
>
> **摘要:** Chest X-ray interpretation is one of the most frequently performed diagnostic tasks in medicine and a primary target for AI development, yet current vision--language models are primarily trained on datasets of paired images and reports, not the cognitive processes and visual attention that underlie clinical reasoning. Here, we present CheXthought, a global, multimodal resource containing 103,592 chain-of-thought reasoning traces and 6,609,082 synchronized visual attention annotations across 50,312 multi-read chest X-rays from 501 radiologists in 71 countries. Our analysis reveals clinical reasoning patterns in how experts deploy distinct visual search strategies, integrate clinical context, and communicate uncertainty. We demonstrate the clinical utility of CheXthought across four dimensions. First, CheXthought reasoning significantly outperforms state--of--the--art vision--language model chain-of-thought in factual accuracy and spatial grounding. Second, visual attention data used as an inference--time hint recovers missed findings and significantly reduces hallucinations. Third, models trained on CheXthought data achieve significantly stronger pathology classification, visual faithfulness, temporal reasoning and uncertainty communication. Fourth, leveraging CheXthought's multi-reader annotations, we predict both human--human and human--AI disagreement directly from an image, enabling transparent communication of case difficulty, uncertainty and model reliability. These findings establish CheXthought as a resource for advancing multimodal clinical reasoning and the development of more transparent, interpretable vision--language models.
>
---
#### [new 014] HOI-aware Adaptive Network for Weakly-supervised Action Segmentation
- **分类: cs.CV**

- **简介: 该论文属于动作分割任务，旨在解决弱监督下相似动作识别模糊的问题。通过引入时空关联的HOI信息，设计自适应网络提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.26227](https://arxiv.org/pdf/2604.26227)**

> **作者:** Runzhong Zhang; Suchen Wang; Yueqi Duan; Yansong Tang; Yue Zhang; Yap-Peng Tan
>
> **备注:** Accepted to IJCAI 2023
>
> **摘要:** In this paper, we propose an HOI-aware adaptive network named AdaAct for weakly-supervised action segmentation. Most existing methods learn a fixed network to predict the action of each frame with the neighboring frames. However, this would result in ambiguity when estimating similar actions, such as pouring juice and pouring coffee. To address this, we aim to exploit temporally global but spatially local human-object interactions (HOI) as video-level prior knowledge for action segmentation. The long-term HOI sequence provides crucial contextual information to distinguish ambiguous actions, where our network dynamically adapts to the given HOI sequence at test time. More specifically, we first design a video HOI encoder that extracts, selects, and integrates the most representative HOI throughout the video. Then, we propose a two-branch HyperNetwork to learn an adaptive temporal encoder, which automatically adjusts the parameters based on the HOI information of various videos on the fly. Extensive experiments on two widely-used datasets including Breakfast and 50Salads demonstrate the effectiveness of our method under different evaluation metrics.
>
---
#### [new 015] Why Domain Matters: A Preliminary Study of Domain Effects in Underwater Object Detection
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决水下域迁移问题。针对现有基准未能反映真实场景因素的缺陷，提出一种基于图像、场景和采集特征的域标注框架，支持更准确的性能评估与故障分析。**

- **链接: [https://arxiv.org/pdf/2604.26174](https://arxiv.org/pdf/2604.26174)**

> **作者:** Melanie Wille; Dimity Miller; Tobias Fischer; Scarlett Raine
>
> **备注:** Poster Presentation at ICRA 2026 Workshop S2S
>
> **摘要:** Domain shift, where deviations between training and deployment data distributions degrade model performance, is a key challenge in underwater environments. Existing benchmarks testing performance for underwater domain shift simulate variability through synthetic style transfer. This fails to capture intrinsic scene factors such as visibility, illumination, scene composition, or acquisition factors, limiting analysis of real-world effects. We propose a labeling framework that defines underwater domains using measurable image, scene, and acquisition characteristics. Unlike prior benchmarks, it captures physically meaningful factors, enabling semantically consistent image grouping and supporting domain-specific evaluation of detection performance including failure analysis. We validate this on public datasets, showing systematic variations across domain factors and revealing hidden failure modes.
>
---
#### [new 016] EnerGS: Energy-Based Gaussian Splatting with Partial Geometric Priors
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决大场景下几何先验不完整导致的重建质量下降问题。提出EnerGS方法，通过能量场形式软性引导优化过程。**

- **链接: [https://arxiv.org/pdf/2604.26238](https://arxiv.org/pdf/2604.26238)**

> **作者:** Rui Song; Tianhui Cai; Markus Gross; Yun Zhang; Walter Zimmer; Zhiyu Huang; Olaf Wysocki; Jiaqi Ma
>
> **摘要:** 3D Gaussian Splatting (3DGS) has been widely adopted for scene reconstruction, where training inherently constitutes a highly coupled and non-convex optimization problem. Recent works commonly incorporate geometric priors, such as LiDAR measurements, either for initialization or as training constraints, with the goal of improving photometric reconstruction quality. However, in large-scale outdoor scenarios, such geometric supervision is often spatially incomplete and uneven, which limits its effectiveness as a reliable prior and can even be detrimental to the final reconstruction. To address this challenge, we model partially observable geometry as a continuous energy field induced by geometric evidence and propose EnerGS. Rather than enforcing geometry as a hard constraint, EnerGS provides a soft geometric guidance for the optimization of Gaussian primitives, allowing geometric information to steer the optimization process without directly restricting the solution space. Extensive experiments on large-scale outdoor scenes demonstrate that, under both sparse multi-view and monocular settings, EnerGS consistently improves photometric quality and geometric stability, while effectively mitigating overfitting during 3DGS training.
>
---
#### [new 017] TAP into the Patch Tokens: Leveraging Vision Foundation Model Features for AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在提升对AI生成和修复图像的识别能力。通过改进特征提取与分类器设计，提出TAP方法，显著提升了检测性能。**

- **链接: [https://arxiv.org/pdf/2604.26772](https://arxiv.org/pdf/2604.26772)**

> **作者:** Ahmed Abdullah; Nikolas Ebert; Oliver Wasenmüller
>
> **备注:** This paper has been accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2026
>
> **摘要:** Recent methods demonstrate that large-scale pretrained models, such as CLIP vision transformers, effectively detect AI-generated images (AIGIs) from unseen generative models when used as feature extractors. Many state-of-the-art methods for AI-generated image detection build upon the original CLIP-ViT to enhance this generalization. Since CLIP's release, numerous vision foundation models (VFMs) have emerged, incorporating architectural improvements and different training paradigms. Despite these advances, their potential for AIGI detection and AI image forensics remains largely unexplored. In this work, we present a comprehensive benchmark across multiple VFM families, covering diverse pretraining objectives, input resolutions, and model scales. We systematically evaluate their out-of-the-box performance for detecting fully-generated AI-images and AI-inpainted images, and discover that the best model outperforms the original CLIP by more than 12% in accuracy, beating established approaches in the process. To fully leverage the features of a modern VFM, we propose a simple redesign of the classifier head by utilizing tunable attention pooling (TAP), which aggregates output tokens into a refined global representation. Integrating TAP with the latest VFMs yields substantial performance gains across several AIGI detection benchmarks, establishing a new state-of-the-art on two challenging benchmarks for in-the-wild detection of AI-generated and -inpainted images.
>
---
#### [new 018] Last-Layer-Centric Feature Recombination: Unleashing 3D Geometric Knowledge in DINOv3 for Monocular Depth Estimation
- **分类: cs.CV**

- **简介: 该论文针对单目深度估计任务，解决几何信息利用不足的问题。通过分析DINOv3的层间特征，提出LFR模块，提升深度预测性能。**

- **链接: [https://arxiv.org/pdf/2604.26454](https://arxiv.org/pdf/2604.26454)**

> **作者:** Gongshu Wang; Zhirui Wang; Kan Yang
>
> **备注:** 18page, 6 figure, 6 table
>
> **摘要:** Monocular depth estimation (MDE) is a fundamental yet inherently ill-posed task. Recent vision foundation models (VFMs), particularly DINO-based transformers, have significantly improved accuracy and generalization for dense prediction. Prior works generally follow a unified paradigm: sampling a fixed set of intermediate transformer layers at uniform intervals to build multi-scale features. This common practice implicitly assumes that geometric information is uniformly distributed across layers, which may underutilize the structural 3D cues encoded in VFMs. In this study, we present a systematic layer-wise analysis of DINOv3, revealing that 3D information is distributed non-uniformly: deeper layers exhibit stronger depth predictability and better capture inter-sample geometric variation. Motivated by this, we introduce a Last-Layer-Centric Feature Recombination (LFR) module to enhance geometric expressiveness. LFR treats the final layer as a geometric anchor and adaptively selects complementary intermediate layers according to a minimal-similarity criterion. Selected features are fused with the last-layer representation via compact linear this http URL experiments show that LFR module consistently improves MDE accuracy and achieves state-of-the-art performance. Our analysis sheds light on how geometric knowledge is organized within VFMs and offers an efficient strategy for unlocking their potential in dense 3D tasks.
>
---
#### [new 019] Which Face and Whose Identity? Solving the Dual Challenge of Deepfake Proactive Forensics in Multi-Face Scenarios
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，解决多人脸场景下的伪造定位与溯源问题。提出DAWF框架，实现对伪造区域和伪造者的精准识别。**

- **链接: [https://arxiv.org/pdf/2604.26342](https://arxiv.org/pdf/2604.26342)**

> **作者:** Lei Zhang; Zhiqing Guo; Dan Ma; Gaobo Yang
>
> **摘要:** Unlike single-face forgeries, deepfakes in complex multi-person interaction scenarios (such as group photos and multi-person meetings) more closely reflect real-world threats. Although existing proactive forensics solutions demonstrate good performance, they heavily rely on a "single-face" setting, making it difficult to effectively address the problems of deepfake localization and source tracing in complex multi-person environments. To address this challenge, we propose the Deep Attributable Watermarking Framework (DAWF). This framework adopts a novel multi-face encoder-decoder architecture that bypasses the cumbersome offline pre-processing steps of traditional forensics, facilitating efficient in-network parallel watermark embedding and cross-face collaborative processing. Crucially, we propose a selective regional supervision loss. This innovative mechanism guides the decoder to focus exclusively on the facial regions tampered with by deepfakes. Leveraging this mechanism alongside the embedded identity payloads, DAWF realizes the "which + who" goal, answering the dual questions of which facial region was forged and who was forged. Extensive experiments on challenging multi-face datasets show that DAWF achieves excellent deepfake localization and traceability in complex multi-person scenes.
>
---
#### [new 020] ViBE: Visual-to-M/EEG Brain Encoding via Spatio-Temporal VAE and Distribution-Aligned Projection
- **分类: cs.CV**

- **简介: 该论文提出ViBE框架，用于从视觉刺激生成M/EEG信号，解决脑编码任务中的神经响应重建与跨模态对齐问题。**

- **链接: [https://arxiv.org/pdf/2604.26218](https://arxiv.org/pdf/2604.26218)**

> **作者:** Ganxi Xu; Zhao-Rong Lai; Yuting Tang; Yonghao Song; Shuyan Zhou; Guoxu Zhou; Boyu Wang; Jian Zhu; Jinyi Long
>
> **摘要:** Brain encoding models not only serve to decipher how visual stimuli are transformed into neural responses, but also represent a critical step toward visual prostheses that restore vision for patients with severe vision disorders. Brain encoding involves two fundamental steps: achieving faithful reconstruction of neural responses and establishing cross-modal alignment between visual stimuli and neural responses. To this end, we propose ViBE, a novel brain encoding framework for generating magnetoencephalography (MEG) and electroencephalography (EEG) signals from visual stimuli. Specifically, we first design a spatio-temporal convolutional variational autoencoder (TSC-VAE) that captures the spatio-temporal characteristics of M/EEG signals for effective neural response reconstruction. To bridge the modality gap between visual features and neural representations, we employ Q-Former to map CLIP image embeddings to the TSC-VAE latent space, producing neural proxy embeddings. For comprehensive cross-modal alignment, we combine mean squared error (MSE) loss for point-wise feature matching with sliced Wasserstein distance (SWD) for probability distribution alignment between the neural proxy embeddings and TSC-VAE latent embeddings. We conduct extensive experiments on the THINGS-EEG2 and THINGS-MEG datasets, demonstrating the effectiveness of our approach in generating high-quality M/EEG signals from visual stimuli.
>
---
#### [new 021] Decoupled Prototype Matching with Vision Foundation Models for Few-Shot Industrial Object Detection
- **分类: cs.CV**

- **简介: 该论文属于工业目标检测任务，解决少量标注样本下的目标检测问题。通过构建类别原型并匹配特征，实现无需大量标注数据的高效检测。**

- **链接: [https://arxiv.org/pdf/2604.26404](https://arxiv.org/pdf/2604.26404)**

> **作者:** Hari Prasanth S. M.; Nilusha Jayawickrama; Risto Ojala
>
> **备注:** This article is submitted to Journal of Intelligent Manufacturing, and is currently in under review
>
> **摘要:** Industrial object detection systems typically rely on large annotated datasets, which are expensive to collect and challenging to maintain in industrial scenarios where the inventory of objects changes frequently. This work addresses the challenge of few-shot object detection in such industrial scenarios, where only a limited number of labeled samples are available for newly introduced objects. We present a detection framework that leverages vision foundation models to recognize objects with minimal supervision. The method constructs class prototypes from a small set of reference samples by extracting feature representations. For a given query scene during inference, object regions are generated using a segmentation model, and feature embeddings are extracted and matched with class prototypes using similarity matching. We evaluate the detection method on three established industrial datasets from the Benchmark for 6D Object Pose Estimation benchmark following the official 2D object detection evaluation protocol. We demonstrate competitive detection performance, improving AP by 6.9% compared to the state-of-the-art training-free detection methods. Furthermore, the presented method is able to onboard new objects using only a few reference images, without requiring any CAD models or large annotated datasets. These properties make the approach well-suited for real-world industrial applications.
>
---
#### [new 022] Lifting Embodied World Models for Planning and Control
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，解决高维动作空间下世界模型难以控制和规划的问题。通过训练轻量策略将高层指令映射为低层动作，提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2604.26182](https://arxiv.org/pdf/2604.26182)**

> **作者:** Alex N. Wang; Trevor Darrell; Pavel Izmailov; Yutong Bai; Amir Bar
>
> **摘要:** World models of embodied agents predict future observations conditioned on an action taken by the agent. For complex embodiments, action spaces are high-dimensional and difficult to specify: for example, precisely controlling a human agent requires specifying the motion of each joint. This makes the world model hard to control and expensive to plan with as search-based methods like CEM scale poorly with action dimensionality. To address this issue, we train a lightweight policy that maps high-level actions to sequences of low-level joint actions. Composing this policy with the frozen world model produces a lifted world model that predicts a sequence of future observations from a single high-level action. We instantiate this framework for a human-like embodiment, defining the high-level action space as a small set of 2D waypoints annotated on the current observation frame, each specifying a near-term goal position for a leaf joint (pelvis, head, hands). Waypoints are low-dimensional, visually interpretable, and easy to specify manually or to search over. We show that the lifted world model substantially outperforms searching directly in low-level joint space ($3.8\times$ lower mean joint error to the goal pose), while remaining more compute-efficient and generalizing to environments unseen by the policy.
>
---
#### [new 023] Attribution-Guided Multimodal Deepfake Detection via Cross-Modal Forensic Fingerprints
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决多模态检测中难以识别伪造来源的问题。提出AMDD框架，结合跨模态指纹一致性损失，提升检测与归属准确性。**

- **链接: [https://arxiv.org/pdf/2604.26453](https://arxiv.org/pdf/2604.26453)**

> **作者:** Wasim Ahmad; Wei Zhang; Xuerui Mao
>
> **摘要:** Audio-visual deepfakes have reached a level of realism that makes perceptual detection unreliable, threatening media integrity and biometric security. While multimodal detection has shown promise, most approaches are binary classification tasks that often latch onto dataset-specific artifacts rather than genuine generative traces. We argue that a detector incapable of identifying how a video was forged is likely learning the wrong signal. Unlike binary detection, attribution-guided learning imposes a stronger geometric constraint on the shared embedding space, forcing the model to encode generator-specific forensic content rather than shortcuts. We propose the Attribution-Guided Multimodal Deepfake Detection (AMDD) framework, which jointly learns to detect and attribute manipulation. AMDD treats generator attribution as a structured regularization that constrains representation geometry toward forensically meaningful features. We introduce a Cross-Modal Forensic Fingerprint Consistency (CMFFC) loss to enforce alignment between generator-induced artifacts in visual and audio streams. This exploits the fact that coherent manipulation leaves correlated traces across modalities, grounded in the physical coupling between speech and facial articulation that synthetic pipelines routinely disrupt. Architecturally, we pair a ResNet50 with temporal attention for visual encoding against a pretrained ResNet18 for mel spectrograms, closing the encoder capacity gap found in prior models. On FakeAVCeleb, AMDD achieves 99.7% balanced accuracy and 99.8% AUC with 95.9% attribution accuracy. Cross-dataset evaluation on DeepfakeTIMIT, DFDM, and LAV-DF confirms that real video detection generalizes robustly, while fake detection on unseen generators remains an open challenge that we analyze in depth.
>
---
#### [new 024] Uncertainty-Aware Pedestrian Attribute Recognition via Evidential Deep Learning
- **分类: cs.CV**

- **简介: 该论文属于行人属性识别任务，解决传统方法无法评估低质量样本预测可靠性的问题。通过引入证据深度学习，提升系统在复杂场景下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26873](https://arxiv.org/pdf/2604.26873)**

> **作者:** Zhuofan Lou; Shihang Zhang; Fangle Zhu; Shengjie Ye; Pingyu Wang
>
> **备注:** 11 pages, 6 figures, 5 tables
>
> **摘要:** We propose UAPAR, an Uncertainty-Aware Pedestrian Attribute Recognition framework. To the best of our knowledge, this is the first EDL-based uncertainty-aware framework for pedestrian attribute recognition (PAR). Unlike conventional deterministic methods, which fail to assess prediction reliability on low-quality samples, UAPAR effectively identifies unreliable predictions and thus enhances system robustness in complex real-world scenarios. To achieve this, UAPAR incorporates Evidential Deep Learning (EDL) into a CLIP-based architecture. Specifically, a Region-Aware Evidence Reasoning module employs cross-attention and spatial prior masks to capture fine-grained local features, which are further processed by an evidence head to estimate attribute-wise epistemic uncertainty. To further enhance training robustness, we develop an uncertainty-guided dual-stage curriculum learning strategy to alleviate the adverse effects of severe label noise during training. Extensive experiments on the PA100K, PETA, RAPv1, and RAPv2 datasets demonstrate that UAPAR achieves competitive or superior performance. Furthermore, qualitative results confirm that the proposed framework generates uncertainty estimates that are predictive of challenging or erroneous samples.
>
---
#### [new 025] GIFGuard: Proactive Forensics against Deepfakes in Facial GIFs via Spatiotemporal Watermarking
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决动画GIF图像的真实性问题。提出GIFGuard框架，通过时空水印技术实现对深度伪造GIF的主动取证。**

- **链接: [https://arxiv.org/pdf/2604.26519](https://arxiv.org/pdf/2604.26519)**

> **作者:** Shupeng Che; Zhiqing Guo; Changtao Miao; Dan Ma; Gaobo Yang
>
> **摘要:** The rapid evolution of deepfake technology poses an unprecedented threat to the authenticity of Graphics Interchange Format (GIF) imagery, which serves as a representative of short-loop temporal media in social networks. However, existing proactive forensics works are designed for static images, which limits their applicability to animated GIFs. To bridge this gap, we propose GIFGuard, the first spatiotemporal watermarking framework tailored for deepfake proactive forensics in GIFs. In the embedding stage, we propose the Spatiotemporal Adaptive Residual Encoder (STARE) to ensure robustness against high-level semantic tampering. It employs a 3D convolutional backbone with adaptive channel recalibration to capture globally coherent temporal dependencies. In the extraction stage, we design the Deep Integrity Restoration Decoder (DIRD). It utilizes a spatiotemporal hourglass architecture equipped with 3D attention to restore latent features, allowing for the accurate extraction of watermark signals even under severe facial manipulation. Furthermore, we construct GIFfaces, the first large-scale benchmark dataset curated for GIF proactive forensics to facilitate research in this domain. Extensive results show that GIFGuard achieves high-fidelity visual quality and remarkable robustness performance against deepfakes. Related code and dataset will be released.
>
---
#### [new 026] ACPO: Anchor-Constrained Perceptual Optimization for Diffusion Models with No-Reference Quality Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型训练中主观视觉质量和语义一致性不足的问题。通过引入无参考感知优化框架，提升生成质量同时保持稳定性与多样性。**

- **链接: [https://arxiv.org/pdf/2604.26348](https://arxiv.org/pdf/2604.26348)**

> **作者:** Yang Yang; Feifan Meng; Han Fang; Weiming Zhang
>
> **备注:** 14 pages, 9 figures, 11 tables
>
> **摘要:** Diffusion models have achieved remarkable success in image generation, yet their training is predominantly driven by full-reference objectives that enforce pixel-wise similarity to ground-truth this http URL supervision, while effective for fidelity, may insufficient in terms of subjective visual perception quality and text-image semantic consistency. In this work, we investigate the problem of incorporating no-reference perceptual quality into diffusion training. A key challenge is that directly optimizing perceptual signals, such as those provided by no-reference image quality assessment (NR-IQA) models, introduces a mismatch with the original diffusion objective, leading to training instability and distributional drift during fine-tuning. To address this issue, we propose an anchor-constrained optimization framework that enables stable perceptual adaptation. Specifically, we leverage a learned NR-IQA model as a perceptual guidance signal, while introducing an anchor-based regularization that enforces consistency with the base diffusion model in terms of noise prediction. This design effectively balances perceptual quality improvement and generative fidelity, allowing controlled adaptation toward perceptually favorable outputs without compromising the original generative behavior. Extensive experiments demonstrate that our method consistently enhances perceptual quality while preserving generation diversity and training stability, highlighting the effectiveness of anchor-constrained perceptual optimization for diffusion models.
>
---
#### [new 027] QYOLO: Lightweight Object Detection via Quantum Inspired Shared Channel Mixing
- **分类: cs.CV; cs.AI; cs.ET**

- **简介: 该论文属于目标检测任务，旨在减少模型参数和计算量。通过引入量子启发的通道混合机制，替换部分骨干网络模块，实现模型压缩。**

- **链接: [https://arxiv.org/pdf/2604.26435](https://arxiv.org/pdf/2604.26435)**

> **作者:** Garvit Kumar Mittal; Sahil Tomar; Sandeep Kumar
>
> **摘要:** The rapid advancement of object detection architectures has positioned single stage detectors as the dominant solution for real-time visual perception. A primary source of computational overhead in these models lies in the deep backbone stages, where C2f bottleneck modules at high stride levels accumulate a disproportionate share of parameters due to quadratic scaling with channel width. This work introduces QYOLO, a quantum-inspired channel mixing framework that achieves genuine architectural compression by replacing the two deepest backbone C2f modules at P4/16 (512 channels) and P5/32 (1024 channels) with a compact QMixBlock. The proposed block performs global channel recalibration through a sinusoidal mixing mechanism with shared learnable parameters across both backbone stages, enforcing consistent channel importance without requiring independent per-stage parameter sets. The neck and detection head remain fully classical and unchanged. Evaluation on the VisDrone2019 benchmark demonstrates that QYOLOv8n achieves a 20.2% reduction in parameter count (3.01M to 2.40M) and 12.3% GFLOPs reduction with only 0.4 pp mAP@50 degradation. QYOLOv8s achieves 21.8% reduction with 0.1 pp degradation. When combined with knowledge distillation, full accuracy parity is recovered at no cost to compression. An expanded backbone plus neck variant achieved 38 to 41% reduction at the cost of greater accuracy degradation, motivating the backbone-only final design.
>
---
#### [new 028] State Beyond Appearance: Diagnosing and Improving State Consistency in Dial-Based Measurement Reading
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，解决MLLM在基于表盘的读数任务中状态不一致的问题。通过分析和提出TriSCA框架提升状态一致性。**

- **链接: [https://arxiv.org/pdf/2604.26614](https://arxiv.org/pdf/2604.26614)**

> **作者:** Yuanze Hu; Gen Li; Yuqin Lan; Qingchen Yu; Zhichao Yang; Junwei Jing; Zhaoxin Fan; Xiaotie Deng
>
> **摘要:** Multimodal large language models (MLLMs) have achieved impressive progress on general multimodal tasks, yet they remain brittle on dial-based measurement reading. In this paper, we study this problem through controlled benchmarks and feature-space probing, and show that current MLLMs not only achieve unsatisfactory accuracy on dial-based readout, but also suffer sharp performance drops under viewpoint and illumination changes even when the underlying dial state remains fixed. Our probing analysis further reveals that same-state samples under appearance variation are not consistently clustered, while neighboring states fail to preserve the local structure implied by continuous dial values. These findings suggest that existing MLLMs largely ignore the intrinsic state geometry of dial measurement tasks and instead rely on superficial appearance cues. Motivated by this diagnosis, we propose TriSCA, a tri-level state-consistent alignment framework for dial-based measurement reading. Specifically, TriSCA consists of state-distance-aware representation alignment, metadata-grounded observation-to-state supervision, and state-aware objective alignment. Extensive ablation studies and evaluation experiments on controlled clock and gauge benchmarks, together with evaluation on an external real-world benchmark, demonstrate the effectiveness of our method.
>
---
#### [new 029] Generalized Disguise Makeup Presentation Attack Detection Using an Attention-Guided Patch-Based Framework
- **分类: cs.CV**

- **简介: 该论文属于人脸活体检测任务，旨在解决伪装妆容攻击的检测问题。通过注意力引导的局部分析框架，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.26025](https://arxiv.org/pdf/2604.26025)**

> **作者:** Fateme Taraghi; Atefe Aghaei; Mohsen Ebrahimi Moghaddam
>
> **摘要:** Despite significant advances in facial recognition systems, they remain vulnerable to face presentation attacks. Among them, disguise makeup attacks are particularly challenging, as they use advanced cosmetics, prosthetic components, and artificial materials to realistically alter facial appearance, often making detection difficult even for humans. Despite their importance, this problem remains underexplored, and publicly available datasets are limited. To address this, we propose a generalized disguise makeup presentation attack detection framework. The method adopts a two-phase design in which a style-invariant full-face model, trained with metric learning and enhanced by a whitening transformation, extracts region attention scores via Grad-CAM. These scores guide a patch-based phase that performs localized analysis using region-specific subnetworks trained with metric learning for fine-grained discrimination. We also construct a new, diverse dataset of live and disguise makeup faces collected under real-world conditions, covering variations in subjects, environments, and disguise materials. Experimental results demonstrate strong generalization across both the collected dataset and SIW-Mv2, achieving 8.97% ACER and 9.76% EER on the collected dataset, and 0% ACER on Obfuscation and Impersonation and 1.34% on Cosmetics attacks of SIW-Mv2. The proposed method consistently outperforms prior works while maintaining robust performance across other spoof types.
>
---
#### [new 030] Privacy-Preserving Clothing Classification using Vision Transformer for Thermal Comfort Estimation
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于隐私保护任务，旨在解决摄像头图像在HVAC控制中的隐私问题。通过引入Vision Transformer，在加密图像上实现高精度的衣物分类，保障隐私同时保持准确率。**

- **链接: [https://arxiv.org/pdf/2604.26184](https://arxiv.org/pdf/2604.26184)**

> **作者:** Tatsuya Chuman; Yousuke Udagawa; Hitoshi Kiya
>
> **备注:** To be appeared in 2026 IEEE International Conference on Consumer Electronics - Taiwan (ICCE-TW 2026)
>
> **摘要:** A privacy-preserving clothing classification scheme is presented to enable secure occupant-centric control (OCC) systems. Although the utilization of camera images for HVAC control has been widely studied to optimize thermal comfort, privacy protection of occupant images has not been considered in prior works. While various privacy-preserving methods have been proposed for image classification, applying conventional schemes results in severe accuracy degradation. In this paper, we introduce a privacy-preserving classification method using Vision Transformer (ViT) applied to clothing insulation estimation. In an experiment using the DeepFashion dataset categorized by clothing insulation, while the conventional pixel-based method suffers a severe accuracy drop, our scheme maintains a high accuracy on encrypted images, showing no degradation from plain images across all categories.
>
---
#### [new 031] SpatialFusion: Endowing Unified Image Generation with Intrinsic 3D Geometric Awareness
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决模型缺乏内在三维空间理解的问题。通过引入3D几何感知，提升图像生成的空间一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2604.26341](https://arxiv.org/pdf/2604.26341)**

> **作者:** Haiyi Qiu; Kaihang Pan; Jiacheng Li; Juncheng Li; Siliang Tang; Yueting Zhuang
>
> **摘要:** Recent unified image generation models have achieved remarkable success by employing MLLMs for semantic understanding and diffusion backbones for image generation. However, these models remain fundamentally limited in spatially-aware tasks due to a lack of intrinsic spatial understanding and the absence of explicit geometric guidance during generation. In this paper, we propose SpatialFusion, a novel framework that internalizes 3D geometric awareness into unified image generation models. Specifically, we first employ a Mixture-of-Transformers (MoT) architecture to augment the MLLM with a parallel spatial transformer to enhance 3D geometric modeling capability. By sharing self-attention with the MLLM, the spatial transformer learns to derive metric-depth maps of target images from rich semantic contexts. These explicit geometric scaffolds are then injected into the diffusion backbone through a specialized depth adapter, providing precise spatial constraints for spatially-coherent image generation. Through a progressive two-stage training strategy, SpatialFusion significantly enhances performance on spatially-aware benchmarks, notably outperforming leading models such as GPT-4o. Additionally, it achieves generalized performance gains across both text-to-image generation and image editing scenarios, all while maintaining negligible inference overhead.
>
---
#### [new 032] CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于联邦域泛化行人重识别任务，解决跨域风格差异导致的模型过拟合问题。提出CO-EVO框架，通过语义锚定与风格扩展协同进化提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.26363](https://arxiv.org/pdf/2604.26363)**

> **作者:** Fengchun Zhang; Qiang Ma; Liuyu Xiang; Jinshan Lai; Tingxuan Huang; Jianwei Hu
>
> **备注:** Accepted at ACL 2026 (Main Conference)
>
> **摘要:** Federated domain generalization for person re-identification (FedDG-ReID) aims to collaboratively train a pedestrian retrieval model across multiple decentralized source domains such that it can generalize to unseen target environments without compromising raw data privacy. However, this task is significantly challenged by the inherent stylistic gaps across decentralized clients. Without global supervision, models easily succumb to shortcut learning where representations overfit to domain specific camera biases rather than universal identity features. We propose CO-EVO, a novel federated framework that resolves this semantic-style conflict through a co-evolutionary mechanism. On the semantic side, Camera-Invariant Semantic Anchoring (CSA) learns identity prompts with cross-camera consistency to establish purified and domain-agnostic anchors that filter out local imaging noise. On the visual side, Global Style Diversification (GSD), powered by a Global Camera-Style Bank (GCSB), synthesizes realistic perturbations to expand the visual boundaries of training data. The core of CO-EVO is its co-evolutionary loop where purified anchors act as gravitational centers to guide the image encoder toward robust anatomical attributes amidst diverse style variations. Extensive experiments demonstrate that CO-EVO achieves state-of-the-art (SOTA) performance, proving that the synergy between semantic purification and style expansion is essential for robust cross-domain generalization. Our code is available at: this https URL.
>
---
#### [new 033] Graph-based Semantic Calibration Network for Unaligned UAV RGBT Image Semantic Segmentation and A Large-scale Benchmark
- **分类: cs.CV**

- **简介: 该论文属于无人机RGBT图像语义分割任务，解决跨模态空间错位和细粒度物体语义混淆问题，提出GSCNet模型并构建大规模基准数据集。**

- **链接: [https://arxiv.org/pdf/2604.26893](https://arxiv.org/pdf/2604.26893)**

> **作者:** Fangqiang Fan; Zhicheng Zhao; Xiaoliang Ma; Chenglong Li; Jin Tang
>
> **备注:** 13 pages,13 figures
>
> **摘要:** Fine-grained RGBT image semantic segmentation is crucial for all-weather unmanned aerial vehicle (UAV) scene understanding. However, UAV RGBT semantic segmentation faces two coupled challenges: cross-modal spatial misalignment caused by sensor parallax and platform vibration, and severe semantic confusion among fine-grained ground objects under top-down aerial views. To address these issues, we propose a Graph-based Semantic Calibration Network (GSCNet) for unaligned UAV RGBT image semantic segmentation. Specifically, we design a Feature Decoupling and Alignment Module (FDAM) that decouples each modality into shared structural and private perceptual components and performs deformable alignment in the shared subspace, enabling robust spatial correction with reduced modality appearance interference. Moreover, we propose a Semantic Graph Calibration Module (SGCM) that explicitly encodes the hierarchical taxonomy and co-occurrence regularities among ground-object categories in UAV scenes into a structured category graph, and incorporates these priors into graph-attention reasoning to calibrate predictions of visually similar and rare this http URL addition, we construct the Unaligned RGB-Thermal Fine-grained (URTF) benchmark, to the best of our knowledge, the largest and most fine-grained benchmark for unaligned UAV RGBT image semantic segmentation, containing over 25,000 image pairs across 61 categories with realistic cross-modal misalignment. Extensive experiments on URTF demonstrate that GSCNet significantly outperforms state-of-the-art methods, with notable gains on fine-grained categories. The dataset is available at this https URL.
>
---
#### [new 034] SEAL: Semantic-aware Single-image Sticker Personalization with a Large-scale Sticker-tag Dataset
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决单图贴纸个性化中的视觉纠缠和结构僵化问题。提出SEAL模块，结合语义约束与结构限制，提升身份保留与上下文控制能力。**

- **链接: [https://arxiv.org/pdf/2604.26883](https://arxiv.org/pdf/2604.26883)**

> **作者:** Changhyun Roh; Yonghyun Jeong; Jonghyun Lee; Chanho Eom; Jihyong Oh
>
> **备注:** The last two authors are co-corresponding authors. Please visit our project page at this https URL
>
> **摘要:** Synthesizing a target concept from a single reference image is challenging in diffusion-based personalized text-to-image generation, particularly for sticker personalization where prompts often require explicit attribute edits. With only one reference, test-time fine-tuning (TTF) methods tend to overfit, producing \textit{visual entanglement}, where background artifacts are absorbed into the learned concept, and \textit{structural rigidity}, where the model memorizes reference-specific spatial configurations and loses contextual controllability. To address these issues, we introduce \textbf{SE}mantic-aware single-image sticker person\textbf{AL}ization (\textbf{SEAL}), a plug-and-play, architecture-agnostic adaptation module that integrates into existing personalization pipelines without modifying their U-Net-based diffusion backbones. SEAL applies three components during embedding adaptation: (1) a Semantic-guided Spatial Attention Loss, (2) a Split-merge Token Strategy, and (3) Structure-aware Layer Restriction. To support sticker-domain personalization with attribute-level control, we present StickerBench, a large-scale sticker image dataset with structured tags under a six-attribute schema (Appearance, Emotion, Action, Camera Composition, Style, Background). These annotations provide a consistent interface for varying context while keeping target identity fixed, enabling systematic evaluation of identity disentanglement and contextual controllability. Experiments show that SEAL consistently improves identity preservation while maintaining contextual controllability, highlighting the importance of explicit spatial and structural constraints during test-time adaptation. The code, StickerBench, and project page will be publicly released.
>
---
#### [new 035] Topology-Aware Representation Alignment for Semi-Supervised Vision-Language Learning
- **分类: cs.CV; cs.LG; math.AT**

- **简介: 该论文属于视觉-语言学习任务，旨在解决领域泛化能力差的问题。通过引入拓扑感知的对齐方法，提升多模态表示的全局结构建模能力。**

- **链接: [https://arxiv.org/pdf/2604.26370](https://arxiv.org/pdf/2604.26370)**

> **作者:** Junwon You; Mihyun Jang; Sangwoo Mo; Jae-Hun Jung
>
> **备注:** 30 pages, 10 figures, 24 tables
>
> **摘要:** Vision-language models have shown strong performance, but they often generalize poorly to specialized domains. While semi-supervised vision-language learning mitigates this limitation by leveraging a small set of labeled image-text pairs together with abundant unlabeled images, existing methods remain fundamentally pairwise and fail to model the global structure of multimodal representation manifolds. Existing topology-based alignment methods rely on persistence diagram matching, which neither guarantees geometric alignment nor utilizes the image-text pairing information central to vision-language learning. We propose Topology-Aware Multimodal Representation Alignment (ToMA), a framework that uses persistent homology to identify topologically salient edges and aligns them across modalities through available cross-modal correspondences. ToMA leverages both H_0-death edges and lightweight H_1-birth edges, allowing it to capture both connectivity and cycle structure without constructing 2-simplices. Experiments show that ToMA yields stable gains, with clear improvements on remote sensing and modest but consistent benefits on fashion retrieval. Additional analysis shows that ToMA is more stable than alternative topology-based objectives and that lightweight H_1-birth edges provide useful higher-order structural signals.
>
---
#### [new 036] A Multistage Extraction Pipeline for Long Scanned Financial Documents: An Empirical Study in Industrial KYC Workflows
- **分类: cs.CV**

- **简介: 该论文属于金融文档信息提取任务，解决长篇多语言扫描文件在KYC流程中的结构化信息提取问题。提出多阶段处理框架，提升提取准确性。**

- **链接: [https://arxiv.org/pdf/2604.26462](https://arxiv.org/pdf/2604.26462)**

> **作者:** Yuxuan Han; Yuanxing Zhang; Yushuo Wang; Yichao Jin; Kenneth Zhu Ke; Jingyuan Zhao
>
> **摘要:** Structured information extraction from long, multilingual scanned financial documents is a core requirement in industrial KYC and compliance workflows. These documents are typically non machine readable, noisy, and visually heterogeneous. They usually span dozens of pages while containing only sparse task relevant information. Although recent vision-language models achieve strong benchmark performance, directly applying them end to end to full financial reports often leads to unreliable extraction under real world conditions. We present a multistage extraction framework that integrates image preprocessing, multilingual OCR, hybrid page-level retrieval, and compact VLM-based structured extraction. The design separates page localization from multimodal reasoning, enabling more accurate extraction from complex multipage documents. We evaluated the framework on 120 production KYC documents comprising about 3000 multilingual scanned pages. Across multiple OCR-VLM combinations, the proposed pipeline consistently outperforms direct PDF-to-VLM baselines, improving field-level accuracy by up to 31.9 percentage points. The best configuration, PaddleOCR with MiniCPM2.6, achieves 87.27 percent accuracy. Ablation studies show that page-level retrieval is the dominant factor in performance improvements, particularly for complex financial statements and non-English documents.
>
---
#### [new 037] RADIO-ViPE: Online Tightly Coupled Multi-Modal Fusion for Open-Vocabulary Semantic SLAM in Dynamic Environments
- **分类: cs.CV**

- **简介: 该论文提出RADIO-ViPE，用于动态环境中开放词汇语义SLAM的任务，解决无标定、单目视频下的多模态融合问题，实现几何感知的语义定位与关联。**

- **链接: [https://arxiv.org/pdf/2604.26067](https://arxiv.org/pdf/2604.26067)**

> **作者:** Zaid Nasser; Mikhail Iumanov; Tianhao Li; Maxim Popov; Jaafar Mahmoud; Sergey Kolyubin
>
> **摘要:** We present RADIO-ViPE (Reduce All Domains Into One -- Video Pose Engine), an online semantic SLAM system that enables geometry-aware open-vocabulary grounding, associating arbitrary natural language queries with localized 3D regions and objects in dynamic environments. Unlike existing approaches that require calibrated, posed RGB-D input, RADIO-ViPE operates directly on raw monocular RGB video streams, requiring no prior camera intrinsics, depth sensors, or pose initialization. The system tightly couples multi-modal embeddings -- spanning vision and language -- derived from agglomerative foundation models (e.g., RADIO) with geometric scene information. This coupling takes place in initialization, optimization and factor graph connections to improve the consistency of the map from multiple modalities. The optimization is wrapped within adaptive robust kernels, designed to handle both actively moving objects and agent-displaced scene elements (e.g., furniture rearranged during ego-centric session). Experiments demonstrate that RADIO-ViPE achieves state-of-the-art results on the dynamic TUM-RGBD benchmark while maintaining competitive performance against offline open-vocabulary methods that rely on calibrated data and static scene assumptions. RADIO-ViPE bridges a critical gap in real-world deployment, enabling robust open-vocabulary semantic grounding for autonomous robotics and unconstrained in-the-wild video streams. Project page: this https URL
>
---
#### [new 038] Three-Step Nav: A Hierarchical Global-Local Planner for Zero-Shot Vision-and-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决零样本导航中路径偏移、过早停止等问题。提出Three-Step Nav，通过全局、当前和回溯三步策略提升导航性能。**

- **链接: [https://arxiv.org/pdf/2604.26946](https://arxiv.org/pdf/2604.26946)**

> **作者:** Wanrong Zheng; Yunhao Ge; Laurent Itti
>
> **备注:** Accepted to AISTATS 2026. Code: this https URL
>
> **摘要:** Breakthrough progress in vision-based navigation through unknown environments has been achieved by using multimodal large language models (MLLMs). These models can plan a sequence of motions by evaluating the current view at each time step against the task and goal given to the agent. However, current zero-shot Vision-and-Language Navigation (VLN) agents powered by MLLMs still tend to drift off course, halt prematurely, and achieve low overall success rates. We propose Three-Step Nav to counteract these failures with a three-view protocol: First, "look forward" to extract global landmarks and sketch a coarse plan. Then, "look now" to align the current visual observation with the next sub-goal for fine-grained guidance. Finally, "look backward" audits the entire trajectory to correct accumulated drift before stopping. Requiring no gradient updates or task-specific fine-tuning, our planner drops into existing VLN pipelines with minimal overhead. Three-Step Nav achieves state-of-the-art zero-shot performance on the R2R-CE and RxR-CE dataset. Our code is available at this https URL.
>
---
#### [new 039] FASH-iCNN: Making Editorial Fashion Identity Inspectable Through Multimodal CNN Probing
- **分类: cs.CV; cs.HC; cs.IR; cs.MM**

- **简介: 该论文提出FASH-iCNN，属于时尚识别任务，旨在揭示时尚编辑文化逻辑。通过多模态CNN分析服装图像，识别品牌、时期和色彩传统，揭示视觉特征对编辑身份的影响。**

- **链接: [https://arxiv.org/pdf/2604.26186](https://arxiv.org/pdf/2604.26186)**

> **作者:** Morayo Danielle Adeyemi; Ryan A. Rossi; Franck Dernoncourt
>
> **备注:** 5 pages, 4 tables, 1 figure. Under review
>
> **摘要:** Fashion AI systems routinely encode the aesthetic logic of specific houses, editors, and historical moments without disclosing it. We present FASH-iCNN, a multimodal system trained on 87,547 Vogue runway images across 15 fashion houses spanning 1991-2024 that makes this cultural logic inspectable. Given a photograph of a garment, the system recovers which house produced it, which era it belongs to, and which color tradition it reflects. A clothing-only model identifies the fashion house at 78.2% top-1 across 14 houses, the decade at 88.6% top-1, and the specific year at 58.3% top-1 across 34 years with a mean error of just 2.2 years. Probing which visual channels carry this signal reveals a sharp dissociation: removing color costs only 10.6pp of house identity accuracy, while removing texture costs 37.6pp, establishing texture and luminance as the primary carriers of editorial identity. FASH-iCNN treats editorial culture as the signal rather than background noise, identifying which houses, eras, and color traditions shaped each output so that users can see not just what the system predicts but which houses, editors, and historical moments are encoded in that prediction.
>
---
#### [new 040] Beyond Shortcuts: Mitigating Visual Illusions in Frozen VLMs via Qualitative Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型在面对视觉幻觉时的感知脆弱问题。通过提出SQI框架，增强模型的视觉 grounding 能力，提升对幻觉的识别与理解。**

- **链接: [https://arxiv.org/pdf/2604.26250](https://arxiv.org/pdf/2604.26250)**

> **作者:** Hao Guo; Fei Wang; Junjie Chen; Yiqi Nie; Jiaqi Zhao; Qiankun Li; Subin Huang
>
> **备注:** 4 pages, 2 figures, and 1 table. This is a methodology paper for the DataCV 2026 Challenge (CVPR Workshops), Task 1, where our method ranked 2nd
>
> **摘要:** While Vision-Language Models (VLMs) have achieved state-of-the-art performance in general visual tasks, their perceptual robustness remains remarkably brittle when confronted with optical illusions. These failures are often attributed to shortcut heuristics, where models prioritize linguistic priors and memorized prototypes over direct visual evidence. In this work, we propose Structured Qualitative Inference (SQI), a training-free, data-centric framework designed to fortify visual grounding in frozen VLMs. SQI addresses perceptual anomalies through three systematic modules: (1) Axiomatic Constraint Injection, which suppresses erroneous metric estimations and quantitative hallucinations; (2) Hierarchical Scene Decomposition, which decouples target visual manifolds from complex background distractors; and (3) Counterfactual Self-Verification, an adversarial reasoning step that mitigates confirmation bias. By orchestrating these qualitative constraints at inference time, SQI effectively aligns high-level linguistic reasoning with low-level visual perception. Our framework was evaluated on the DataCV 2026 Challenge (Task I: Classic Illusion Understanding), where it ranked 2nd place overall. Experimental results demonstrate that SQI not only significantly enhances accuracy across diverse illusion categories but also provides superior diagnostic interpretability without any model fine-tuning. Our success underscores the potential of structured qualitative grounding as a robust paradigm for developing next-generation, illusion-resistant vision-language systems.
>
---
#### [new 041] ViCrop-Det: Spatial Attention Entropy Guided Cropping for Training-Free Small-Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测任务，旨在解决小物体检测中因空间异质性导致的特征退化问题。提出ViCrop-Det框架，通过空间注意力熵引导动态裁剪，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.26806](https://arxiv.org/pdf/2604.26806)**

> **作者:** Hui Wang; Hongze Li; Wei Chen; Xiaojin Zhang
>
> **摘要:** Transformer-based architectures have established a dominant paradigm in global semantic perception; however, they remain fundamentally constrained by the profound spatial heterogeneity inherent in natural images. Specifically, the imposition of a uniform global receptive field across regions of varying information density inevitably leads to local feature degradation, particularly in dense conflict zones populated by microscopic targets. To address this mechanistic limitation, we propose ViCrop-Det, a training-free inference framework that introduces adaptive spatial trust region shrinkage. Inspired by the use of attention entropy in anomaly segmentation, ViCrop-Det leverages the detection decoder's cross-attention distribution as an endogenous probe. By utilizing Spatial Attention Entropy (SAE) to heuristically evaluate local spatial ambiguity, the framework executes dynamic spatial routing, allocating a fixed computational budget exclusively to regions exhibiting both high target saliency and high cognitive uncertainty. By shrinking the spatial trust region and injecting high-frequency localized observations, ViCrop-Det actively resolves spatial ambiguity and recovers fine-grained features without requiring architectural modifications. Extensive evaluations on VisDrone and DOTA-v1.5 demonstrate that ViCrop-Det yields competitive performance enhancements, consistently adding +1-3 mAP@50 to RT-DETR-R50 and Deformable DETR with a marginal 20-23\% latency overhead. On MS COCO, $AP_{S}$ improves while $AP_{M}/AP_{L}$ remains stable, indicating precise fine-scale refinement without compromising the global spatial prior. Under compute-matched settings, our adaptive routing strategy comprehensively surpasses uniform slicing baselines, achieving a highly optimized accuracy-speed trade-off.
>
---
#### [new 042] Hearing the Room Through the Shape of the Drum: Modal-Guided Sound Recovery from Multi-Point Surface Vibrations
- **分类: cs.CV**

- **简介: 该论文属于声源恢复任务，解决复杂物体振动响应差的问题。通过多点振动捕捉与物理模型，提升声源重建效果。**

- **链接: [https://arxiv.org/pdf/2604.26678](https://arxiv.org/pdf/2604.26678)**

> **作者:** Shai Bagon; Matan Kichler; Mark Sheinin
>
> **备注:** Oral presentation at The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026
>
> **摘要:** Optical vibration sensing enables recovering the scene sound directly from the surface vibration of nearby objects, turning everyday objects into ``visual microphones''. However, most prior methods had focused on capturing the vibrations of specific objects with highly favorable vibration responses. These include objects where the surface vibrations are generated by the object itself (e.g., speaker membrane or guitar body) or objects consisting of a thin membrane which is highly reactive to sound (e.g., a chip bag or the leaf of a plant). In this paper, we tackle sound recovery for a more challenging class of solid objects whose vibration responses are poor or highly resonant. We simultaneously capture vibrations for multiple surface points on the object using a speckle-based vibrometry imaging system. Then, we derive a novel physics-guided vibration formation model that relates the scene sound source to the captured multi-point multi-axis vibrations via the object's vibrational modes. The model is then used to reverse the resonant transfer function of the vibrating object, fusing multiple vibration signals to estimate the original sound source in the scene. We evaluate our approach by recovering sound from a variety of everyday objects, demonstrating that it significantly outperforms traditional single-point speckle vibrometry in challenging scenarios and other signal-processing-based methods for multi-signal fusing.
>
---
#### [new 043] Event-based Liveness Detection using Temporal Ocular Dynamics: An Exploratory Approach
- **分类: cs.CV**

- **简介: 该论文属于人脸活体检测任务，旨在解决传统方法在不同传感器和攻击场景下性能下降的问题。通过使用事件相机分析眼部动态，提升活体检测的准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26285](https://arxiv.org/pdf/2604.26285)**

> **作者:** Nicolas Mastropasqua; Ignacio Bugueno-Cordova; Rodrigo Verschae; Daniel Acevedo; Pablo Negri
>
> **备注:** Accepted at FG 2026 FME Workshop
>
> **摘要:** Face liveness detection has been extensively studied using RGB cameras, achieving strong performance under controlled conditions but often failing to generalize across sensors and attack scenarios. In this work, we explore event cameras as an alternative sensing modality for liveness detection based on temporal ocular dynamics. Event cameras capture sparse, asynchronous changes in brightness with microsecond resolution, enabling precise analysis of fast eye movements such as saccades. Replay attacks cannot faithfully reproduce these dynamics due to temporal resampling and display artifacts, leading to distinctive spatio-temporal patterns in the event domain. We design a data collection protocol to extend RGBE-Gaze with replay-attack recordings, yielding an event-based fake counterpart for liveness detection. We analyze event-driven temporal features from eye regions and evaluate their effectiveness for ocular motion segmentation and liveness classification. Our results show that event-based representations enable reliable discrimination between genuine and replayed sequences, achieving up to 95.37% top-1 accuracy with a spiking convolutional neural network. These preliminary findings highlight the potential of event-based sensing for robust and low-latency liveness detection.
>
---
#### [new 044] Delineating Knowledge Boundaries for Honest Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型在面对未知问题时缺乏拒绝能力的问题。通过构建数据集并优化模型，提升其识别知识边界的能力。**

- **链接: [https://arxiv.org/pdf/2604.26419](https://arxiv.org/pdf/2604.26419)**

> **作者:** Junru Song; Yimeng Hu; Yijing Chen; Huining Li; Qian Li; Lizhen Cui; Yuntao Du
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable multimodal performance yet remain prone to factual hallucinations, particularly in long-tail or specialized domains. Moreover, current models exhibit a weak capacity to refuse queries that exceed their parametric knowledge. In this paper, we propose a systematic framework to enhance the refusal capability of VLMs when facing such unknown questions. We first curate a model-specific "Visual-Idk" (Visual-I don't know) dataset, leveraging multi-sample consistency probing to distinguish between known and unknown facts. We then align the model using supervised fine-tuning followed by preference-aware optimization (e.g., DPO, ORPO) to effectively delineate its knowledge boundaries. Results on the Visual-Idk dataset show our method improves the Truthful Rate from 57.9\% to 67.3\%. Additionally, internal probing also demonstrates that the model genuinely recognizes its boundaries instead of just memorizing refusal patterns. Our framework further generalizes to out-of-distribution medical and perceptual domains, providing a robust path toward more trustworthy and prudent visual assistants.
>
---
#### [new 045] World2VLM: Distilling World Model Imagination into VLMs for Dynamic Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决动态空间推理问题。通过将世界模型的时空想象融入VLM，提升其在动态场景下的推理能力。**

- **链接: [https://arxiv.org/pdf/2604.26934](https://arxiv.org/pdf/2604.26934)**

> **作者:** Wanyue Zhang; Wenxiang Wu; Wang Xu; Jiaxin Luo; Helu Zhi; Yibin Huang; Shuo Ren; Zitao Liu; Jiajun Zhang
>
> **备注:** The code is available at this https URL. The dataset is available at this https URL
>
> **摘要:** Vision-language models (VLMs) have shown strong performance on static visual understanding, yet they still struggle with dynamic spatial reasoning that requires imagining how scenes evolve under egocentric motion. Recent efforts address this limitation either by scaling spatial supervision with synthetic data or by coupling VLMs with world models at inference time. However, the former often lacks explicit modeling of motion-conditioned state transitions, while the latter incurs substantial computational overhead. In this work, we propose World2VLM, a training framework that distills spatial imagination from a generative world model into a vision-language model. Given an initial observation and a parameterized camera trajectory, we use a view-consistent world model to synthesize geometrically aligned future views and derive structured supervision for both forward (action-to-outcome) and inverse (outcome-to-action) spatial reasoning. We post-train the VLM with a two-stage recipe on a compact dataset generated by this pipeline and evaluate it on multiple spatial reasoning benchmarks. World2VLM delivers consistent improvements over the base model across diverse benchmarks, including SAT-Real, SAT-Synthesized, VSI-Bench, and MindCube. It also outperforms the test-time world-model-coupled methods while eliminating the need for expensive inference-time generation. Our results suggest that world models can serve not only as inference-time tools, but also as effective training-time teachers, enabling VLMs to internalize spatial imagination in a scalable and efficient manner.
>
---
#### [new 046] 3D-LENS: A 3D Lifting-based Elevated Novel-view Synthesis method for Single-View Aerial-Ground Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于AG-ReID任务，解决单视角下跨视点重识别问题。提出3D-LENS方法，结合3D重建与表示学习，提升跨视角检索性能。**

- **链接: [https://arxiv.org/pdf/2604.26520](https://arxiv.org/pdf/2604.26520)**

> **作者:** William Grolleau; Astrid Sabourin; Guillaume Lapouge; Catherine Achard
>
> **摘要:** Aerial-Ground Re-Identification (AG-ReID) is constrained by the viewpoint-domain gap, as drastic viewpoint disparities occlude or distort discriminative features, making cross-viewpoint image retrieval challenging. While existing methods rely on paired cross-view annotations, real-world deployments, such as wilderness search-and-rescue (SAR), often lack target-domain data, requiring retrieval from ground-level references alone. To our knowledge, we are the first to address this challenge by formalizing the Single-View AG-ReID (SV AG-ReID) setting, where models trained on a single real viewpoint must generalize to an unseen viewpoint. We propose 3D Lifting-based Elevated Novel-view Synthesis (3D-LENS), a unified framework combining geometrically-consistent novel view synthesis that leverages large-scale 3D mesh reconstruction, with a robust representation learning scheme to mitigate synthetic-to-real bias. Unlike 2D generative baselines that suffer from geometric inconsistencies or prior 3D methods that are restricted to class-specific templates, our approach ensures view-consistent synthesis across diverse categories without predefined templates that fail to capture fine-grained details, such as carried objects. Extensive experiments demonstrate that our method achieves state-of-the-art performance on SV AG-ReID scenarios. Code and data will be released at this https URL.
>
---
#### [new 047] Beyond Fixed Formulas: Data-Driven Linear Predictor for Efficient Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于扩散模型优化任务，旨在降低采样成本。针对现有方法依赖固定公式的不足，提出L2P框架，通过可学习的线性预测器实现高效加速。**

- **链接: [https://arxiv.org/pdf/2604.26365](https://arxiv.org/pdf/2604.26365)**

> **作者:** Zhirong Shen; Rui Huang; Jiacheng Liu; Chang Zou; Peiliang Cai; Shikang Zheng; Zhengyi Shi; Liang Feng; Linfeng Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** To address the high sampling cost of Diffusion Transformers (DiTs), feature caching offers a training-free acceleration method. However, existing methods rely on hand-crafted forecasting formulas that fail under aggressive skipping. We propose L2P (Learnable Linear Predictor), a simple data-driven caching framework that replaces fixed coefficients with learnable per-timestep weights. Rapidly trained in ~20 seconds on a single GPU, L2P accurately reconstructs current features from past trajectories. L2P significantly outperforms existing baselines: it achieves a 4.55x FLOPs reduction and 4.15x latency speedup on FLUX.1-dev, and maintains high visual fidelity under up to 7.18x acceleration on Qwen-Image models, where prior methods show noticeable quality degradation. Our results show learning linear predictors is highly effective for efficient DiT inference. Code is available at this https URL.
>
---
#### [new 048] Color-Encoded Illumination for High-Speed Volumetric Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D动态场景重建任务，解决传统相机帧率限制导致无法捕捉高速场景的问题。通过颜色编码照明和动态高斯点云方法，实现多视角高速体积重建。**

- **链接: [https://arxiv.org/pdf/2604.26920](https://arxiv.org/pdf/2604.26920)**

> **作者:** David Novikov; Eilon Vaknin; Narek Tumanyan; Mark Sheinin
>
> **备注:** accepted to IEEE CVPR 2026 as a highlight
>
> **摘要:** The task of capturing and rendering 3D dynamic scenes from 2D images has become increasingly popular in recent years. However, most conventional cameras are bandwidth-limited to 30-60 FPS, restricting these methods to static or slowly evolving scenes. While overcoming bandwidth limitations is difficult for general scenes, recent years have seen a flurry of computational imaging methods that yield high-speed videos using conventional cameras for specific applications (e.g., motion capture and particle image velocimetry). However, most of these methods require modifications to a camera's optics or the addition of mechanically moving components, limiting them to a single-view high-speed capture. Consequently, these methods cannot be readily used to capture a 3D representation of rapid scene motion. In this paper, we propose a novel method to capture and reconstruct a volumetric representation of a high-speed scene using only unaugmented low-speed cameras. Instead of modifying the hardware or optics of each individual camera, we encode high-speed scene dynamics by illuminating the scene with a rapid, sequential color-coded sequence. This results in simultaneous multi-view capture of the scene, where high-speed temporal information is encoded in the spatial intensity and color variations of the captured images. To construct a high-speed volumetric representation of the dynamic scene, we develop a novel dynamic Gaussian Splatting-based approach that decodes the temporal information from the images. We evaluate our approach on simulated scenes and real-world experiments using a multi-camera imaging setup, showing first-of-a-kind high-speed volumetric scene reconstructions.
>
---
#### [new 049] Semantic Foam: Unifying Spatial and Semantic Scene Decomposition
- **分类: cs.CV**

- **简介: 该论文属于场景分解任务，旨在提升3D场景的语义分割效果。针对现有方法在分割质量和跨视角一致性上的不足，提出Semantic Foam，结合空间结构与语义特征，提高分割性能。**

- **链接: [https://arxiv.org/pdf/2604.26262](https://arxiv.org/pdf/2604.26262)**

> **作者:** Amr Sharafeldin; Shrisudhan Govindarajan; Thomas Walker; Aryan Mikaeili; Daniel Rebain; Kwang Moo Yi; Andrea Tagliasacchi
>
> **备注:** 15 pages, 10 figures, Accepted to CVPR 2026 (Highlight) , Project page: this http URL
>
> **摘要:** Modern scene reconstruction methods, such as 3D Gaussian Splatting, enable photo-realistic novel view synthesis at real-time speeds. However, their adoption in interactive graphics applications remains limited due to the difficulty of interacting with these representations compared to traditional, human-authored 3D assets. While prior work has attempted to impose semantic decomposition on these models, significant challenges remain in segmentation quality and cross-view this http URL address these limitations, we introduce Semantic Foam, which extends the recently proposed Radiant Foam representation to semantic decomposition tasks. Our approach leverages the inherent spatial structure of Radiant Foam's volumetric Voronoi mesh and augments it with an explicit semantic feature field defined at the cell level. This design enables direct spatial regularization, improving consistency across views and mitigating artifacts caused by occlusion and inconsistent supervision, which are common issues in point-based this http URL results demonstrate that our method achieves superior object-level segmentation performance compared to state-of-the-art approaches such as Gaussian Grouping and this http URL page: this http URL
>
---
#### [new 050] Cross-Domain Transfer of Hyperspectral Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像语义分割任务，解决数据不足导致的模型性能问题。通过跨领域迁移，利用遥感预训练模型提升近场感知效果，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.26478](https://arxiv.org/pdf/2604.26478)**

> **作者:** Nick Theisen; Peer Neubert
>
> **备注:** Accepted for publication at ICPR 2026
>
> **摘要:** Hyperspectral imaging (HSI) semantic segmentation typically relies on in-domain training, but limited data availability often restricts model performance in real-world applications. Current approaches to leverage foundation models in proximal sensing use cross-modality techniques, bridging RGB and HSI to exploit vision foundation models. However, these methods either discard spectral information or introduce architectural complexity. We propose cross-domain transfer as an alternative, reusing HSI foundation models - originally trained in remote sensing - for proximal sensing applications. By eliminating the need to bridge modality gaps, our approach preserves spectral information while maintaining a simple architecture. Using the HS3-Bench benchmark, we systematically evaluate and compare conventional in-domain, in-modality training, cross-modality transfer and cross-domain transfer strategies. Our results demonstrate that cross-domain transfer achieves large performance improvements over in-domain, in-modality training, reduces the performance gap to cross-modality approaches and maintains strong performance in limited data settings. Thus, this work advances more effective HSI semantic segmentation in diverse applications.
>
---
#### [new 051] Robust Alignment: Harmonizing Clean Accuracy and Adversarial Robustness in Adversarial Training
- **分类: cs.CV**

- **简介: 该论文属于对抗训练任务，旨在解决干净准确率与对抗鲁棒性之间的权衡问题。通过引入鲁棒对齐机制，提升模型在保持高准确率的同时增强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26496](https://arxiv.org/pdf/2604.26496)**

> **作者:** Yanyun Wang; Qingqing Ye; Li Liu; Zi Liang; Haibo Hu
>
> **备注:** 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition - Findings Track (CVPR'26 Findings)
>
> **摘要:** Adversarial Training (AT) is one of the most effective methods for developing robust deep neural networks (DNNs). However, AT faces a trade-off problem between clean accuracy and adversarial robustness. In this work, we reveal a surprising phenomenon for the first time: Varying input perturbation intensities for training samples near decision boundaries in AT have minimal impact on model robustness. This finding directly exposes the inconsistency between accuracy and robustness score fluctuations, leading us to identify the misalignment between input and latent spaces as a critical driver of the robustness-accuracy trade-off. To mitigate this misalignment for harmonizing accuracy and robustness, we define Robust Alignment as a new AT target, encouraging the model perception to change with input perturbations provided the final label prediction remains unchanged, which can be achieved via two novel ideas. First, we suggest a reduced and fixed perturbation intensity for those boundary samples, which facilitates the model to utilize the perturbations as learnable patterns, instead of noises that complicate decision boundaries meaninglessly. Second, we propose a Domain Interpolation Consistency Adversarial Regularization (DICAR), based on rigorous theoretical derivations, which explicitly introduces semantic alignment between input and latent spaces into AT. Based on these two ideas, we end up with a new Robust Alignment Adversarial Training (RAAT) method, effectively harmonizing accuracy and robustness. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet with ResNet-18, PreActResNet-18, and WideResNet-28-10 demonstrate the effectiveness of RAAT in improving the trade-off beyond four common baselines and a total of 14 related state-of-the-art (SOTA) works.
>
---
#### [new 052] MTCurv: Deep learning for direct microtubule curvature mapping in noisy fluorescence microscopy images
- **分类: cs.CV; q-bio.CB**

- **简介: 该论文属于生物图像分析任务，旨在解决微管弯曲度准确提取的问题。通过深度学习方法直接回归弯曲度图，提升在噪声和低对比度下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26517](https://arxiv.org/pdf/2604.26517)**

> **作者:** Achraf Ait Laydi; Sidi Mohamed Sid'El Moctar; Yousef El Mourabit; Hélène Bouvrais
>
> **备注:** Accepted for presentation at the International Conference on Pattern Recognition (ICPR) 2026
>
> **摘要:** Accurate quantification of the geometry of curvilinear biological structures is essential for understanding cellular mechanics and disease-related morphological alterations. Microtubule curvature is a key descriptor of filament rigidity and mechanical perturbations. However, reliable curvature extraction from fluorescence microscopy images remains challenging due to noise, low contrast, and partial filament visibility. Existing approaches rely on segmentation pipelines with pre or post-processing, which are highly sensitive to segmentation errors and often fail under adverse imaging conditions. In this work, we propose MTCurv, a deep learning framework for direct, segmenta-tion-free regression of microtubule curvature maps from noisy microscopy images. Leveraging a synthetic dataset with pixel-wise curvature annotations, we reformulated curvature estimation as a regression problem and adapted an attention-based residual U-Net. To reduce hallucinations and enforce spatial coherence, we introduced a gradient-aware loss combining Mean Squared Error with a gradient consistency term. Beyond model and loss design, we evaluated commonly used regression and image quality metrics, revealing that many perceptual and blind metrics are poorly suited for curvature estimation. Correlation-based metrics, particularly Spearman correlation, emerged as more reliable indicators of curvature prediction quality. Experiments on two datasets of increasing difficulty demonstrated that MTCurv accurately recovers local microtubule curvatures, even in the presence of background fluorescence. Ablation studies highlighted the contribution of both residual encoding and attention-based decoding. Overall, this work provides a practical tool for filament curvature analysis and methodological insights for geometry-aware regression in biomedical imaging. Datasets and code are made available.
>
---
#### [new 053] AirZoo: A Unified Large-Scale Dataset for Grounding Aerial Geometric 3D Vision
- **分类: cs.CV**

- **简介: 该论文提出AirZoo数据集，解决航空几何3D视觉训练数据不足的问题，涵盖大规模、多样化场景和精确标注，用于图像检索、跨视图匹配和3D重建任务。**

- **链接: [https://arxiv.org/pdf/2604.26567](https://arxiv.org/pdf/2604.26567)**

> **作者:** Xiaoya Cheng; Rouwan Wu; Xinyi Liu; Zeyu Cui; Yan Liu; Na Zhao; Yu Liu; Maojun Zhang; Shen Yan
>
> **摘要:** Despite the rapid progress in data-driven 3D vision, aerial geometric 3D vision remains a formidable challenge due to the severe scarcity of large-scale, high-fidelity training data. Existing benchmarks, predominantly biased toward ground-level or object-centric views, do not account for complex viewpoint transformations and diverse environmental conditions in UAV-based sensing. To bridge this critical gap, we propose AirZoo, a unified large-scale dataset and benchmark for grounding aerial geometric 3D vision. AirZoo possesses three appealing properties: 1) Scalable Generation Pipeline: Leveraging freely available, world-scale photogrammetric 3D meshes, it renders vast outdoor environments with customizable UAV flight trajectories and configurable weather/illumination. 2) Comprehensive Scene Diversity: It provides the most extensive coverage of region types to date (spanning 378 regions across 22 countries), systematically encompassing both highly structured urban landscapes and complex unstructured natural environments. 3) Rich Geometric Annotations: Each frame provides synchronized, pixel-level metric depth and precise 6-DoF geo-referenced poses, essential for geometry-aware learning. Through three rigorous evaluation tracks -- aerial image retrieval, cross-view matching, and multi-view 3D reconstruction -- we demonstrate that AirZoo serves as a powerful pre-training engine. Extensive experiments on both public and newly collected real-world benchmarks reveal that fine-tuning on AirZoo yields substantial performance gains for SoTA models (e.g., MegaLoc, RoMa, VGGT, and Depth Anything 3), establishing a new performance upper bound for aerial spatial intelligence.
>
---
#### [new 054] Breaking the Rigid Prior: Towards Articulated 3D Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于3D异常检测任务，针对关节物体的刚性先验失效问题，提出ArtiAD基准和SPA-SDF方法，有效区分姿态变化与结构缺陷。**

- **链接: [https://arxiv.org/pdf/2604.26868](https://arxiv.org/pdf/2604.26868)**

> **作者:** Jinye Gan; Bozhong Zheng; Xiaohao Xu; Junye Ren; Zixuan Zhang; Na Ni; Yingna Wu
>
> **摘要:** Existing 3D anomaly detection methods are built on a rigid prior: normal geometry is pose-invariant and can be canonicalized through registration or alignment. This prior does not hold for articulated objects with hinge or sliding joints, where valid pose changes induce structured geometric variations that cannot be collapsed to a single canonical template, causing pose-induced deformations to be misidentified as anomalies while true structural defects are obscured. No existing benchmark addresses this challenge. We introduce ArtiAD, the first large-scale benchmark for articulated 3D anomaly detection, comprising 15,229 point clouds across 39 object categories with dense joint-angle variations and six structural anomaly types. Each sample is annotated with its joint configuration and part-level motion labels, enabling explicit disentanglement of pose-induced geometry from structural defects. ArtiAD also provides a seen/unseen articulation split to evaluate both interpolation and extrapolation to novel joint configurations. We propose Shape-Pose-Aware Signed Distance Field (SPA-SDF), a baseline that replaces the rigid prior with a continuous pose-conditioned implicit field, factorized into an articulation-independent structural prior and a Fourier-encoded joint embedding. At inference, the articulation state is recovered by minimizing reconstruction energy, and anomalies are identified as point-wise deviations from the learned manifold. SPA-SDF achieves 0.884 object-level AUROC on seen configurations and 0.874 on unseen configurations, substantially outperforming all rigid-based baselines. Our code and benchmark will be publicly released to facilitate future research.
>
---
#### [new 055] Federated Medical Image Classification under Class and Domain Imbalance exploiting Synthetic Sample Generation
- **分类: cs.CV**

- **简介: 该论文属于医疗图像分类任务，解决因类别和领域不平衡导致的模型性能下降问题。通过生成合成样本提升罕见病理和不同设备的覆盖，增强模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.26324](https://arxiv.org/pdf/2604.26324)**

> **作者:** Martina Pavan; Matteo Caligiuri; Francesco Barbato; Pietro Zanuttigh
>
> **备注:** Accepted at ICPR 2026, 13 pages, 3 figures, 5 tables
>
> **摘要:** Exploiting deep learning in medical imaging faces critical challenges, including strict privacy constraints, heterogeneous imaging devices with varying acquisition properties, and class imbalance due to the uneven prevalence of pathologies. In this work, we propose FedSSG, a novel Federated Learning framework that addresses domain shifts caused by diverse imaging devices while mitigating the under-representation of rare pathologies. The key contribution is a strategy for generating synthetic samples and distributing them across clients to improve coverage of both underrepresented pathologies and imaging devices. Experimental results demonstrate that our approach significantly enhances model performance and generalization across heterogeneous institutions, with minimal computational overhead at the client side.
>
---
#### [new 056] FruitProM-V2: Robust Probabilistic Maturity Estimation and Detection of Fruits and Vegetables
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于果实成熟度估计任务，旨在解决传统分类方法在相邻成熟阶段边界模糊的问题。通过将成熟度建模为连续变量并使用概率检测头进行预测，提升估计的可靠性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26084](https://arxiv.org/pdf/2604.26084)**

> **作者:** Rahul Harsha Cheppally; Sidharth Rai; Sudan Baral; Benjamin Vail; Ajay Sharda
>
> **摘要:** Accurate fruit maturity identification is essential for determining harvest timing, as incorrect assessment directly affects yield and post-harvest quality. Although ripening is a continuous biological process, vision-based maturity estimation is typically formulated as a multi-class classification task, which imposes sharp boundaries between visually similar stages. To examine this limitation, we perform an annotation reliability study with two independent annotators on a held-out tomato dataset and observe disagreement concentrated near adjacent maturity stages. Motivated by this observation, we model maturity as a latent continuous variable and predict it probabilistically using a distributional detection head, converting the distribution into class probabilities through the cumulative distribution function (CDF). The proposed formulation maintains comparable performance to a standard detector under clean labels while better representing uncertainty. Furthermore, when controlled label noise is introduced during training, the probabilistic model demonstrates improved robustness relative to the baseline, indicating that explicitly modeling maturity uncertainty leads to more reliable visual maturity estimation.
>
---
#### [new 057] $\text{PKS}^4$:Parallel Kinematic Selective State Space Scanners for Efficient Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频中时间建模的高计算成本问题。提出PKS⁴模块，通过线性复杂度扫描替代注意力机制，提升效率并保持空间结构。**

- **链接: [https://arxiv.org/pdf/2604.26461](https://arxiv.org/pdf/2604.26461)**

> **作者:** Lingjie Zeng; Hailun Zhang; Xiwen Wang; Qijun Zhao
>
> **摘要:** Temporal modeling remains a fundamental challenge in video understanding, particularly as sequence lengths scale. Traditional video models relying on dense spatiotemporal attention suffer from quadratic computational costs for long videos. To circumvent these costs, recent approaches adapt image models for videos via Parameter-Efficient Fine-Tuning (PEFT) methods such as adapters. However, deeply inserting these modules incurs prohibitive activation memory overhead during back-propagation. While recent efficient State Space Models (SSMs) introduce linear complexity, they disrupt 2D spatial relationships and rely on extensive masked pre-training to recover spatial awareness. To overcome these limitations, we propose Parallel Kinematic Selective State Space Scanners (PKS$^4$). We retain a standard 2D vision backbone for spatial semantics and insert a single plug-and-play PKS$^4$ module with linear-complexity temporal scanning, avoiding temporal attention and multi-layer adapters. We first extract kinematic priors via a Kinematic Prior Encoder, which captures local displacements and motion boundaries through inter-frame correlations and differences. These priors drive linear-complexity SSMs to track underlying kinematic states, adaptively modulating update speeds and read-write strategies at each time step. Instead of global scanning, we deploy parallel scanners along the temporal dimension for each spatial location, preserving spatial structures while reducing overhead. Experiments on spatial-heavy and temporal-heavy action recognition benchmarks show that PKS$^4$ achieves state-of-the-art performance. Remarkably, our method converges in merely $20$ epochs, achieving approximately $10\times$ lower training compute than pure video SSMs, establishing a new paradigm for efficient video understanding.
>
---
#### [new 058] Bridge: Basis-Driven Causal Inference Marries VFMs for Domain Generalization
- **分类: cs.CV**

- **简介: 该论文属于目标检测领域的域泛化任务，旨在解决因源域与目标域分布差异导致的性能下降问题。提出Bridge框架，结合因果推理与视觉基础模型，减少混淆因素影响，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.26820](https://arxiv.org/pdf/2604.26820)**

> **作者:** Mingbo Hong; Feng Liu; Caroline Gevaert; George Vosselman; Hao Cheng
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Detectors often suffer from degraded performance, primarily due to the distributional gap between the source and target domains. This issue is especially evident in single-source domains with limited data, as models tend to rely on confounders (e.g., illumination, co-occurrence, and style) from the source domain, leading to spurious correlations that hinder generalization. To this end, this paper proposes a novel Basis-driven framework for domain generalization, namely \textbf{\textit{Bridge}}, that incorporates causal inference into object detection. By learning the low-rank bases for front-door adjustment, \textbf{\textit{Bridge}} blocks confounders' effects to mitigate spurious correlations, while simultaneously refining representations by filtering redundant and task-irrelevant components. \textbf{\textit{Bridge}} can be seamlessly integrated with both discriminative (e.g., DINOv2/3, SAM) and generative (e.g., Stable Diffusion) Vision Foundation Models (VFMs). Extensive experiments across multiple domain generalization object detection datasets, i.e., Cross-Camera, Adverse Weather, Real-to-Artistic, Diverse Weather Datasets, and Diverse Weather DroneVehicle (our newly augmented real-world UAV-based benchmark), underscore the superiority of our proposed method over previous state-of-the-art approaches. The project page is available at: this https URL.
>
---
#### [new 059] Star-Fusion: A Multi-modal Transformer Architecture for Discrete Celestial Orientation via Spherical Topology
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Star-Fusion，解决航天器自主导航中的星体姿态确定问题。针对传统方法计算量大、易受噪声影响，以及深度学习模型对球面拓扑的不适应，该工作采用多模态Transformer架构，提升定位精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.26582](https://arxiv.org/pdf/2604.26582)**

> **作者:** May Hammad; Menatallh Hammad
>
> **摘要:** Reliable celestial attitude determination is a critical requirement for autonomous spacecraft navigation, yet traditional "Lost-in-Space" (LIS) algorithms often suffer from high computational overhead and sensitivity to sensor-induced noise. While deep learning has emerged as a promising alternative, standard regression models are often confounded by the non-Euclidean topology of the celestial sphere and by the periodic boundary conditions of Right Ascension (RA) and Declination (Dec). In this paper, we present Star-Fusion, a multi-modal architecture that reformulates orientation estimation as a discrete topological classification task. Our approach leverages spherical K-Means clustering to partition the celestial sphere into K topologically consistent regions, effectively mitigating coordinate wrapping artifacts. The proposed architecture employs a tripartite fusion strategy: a SwinV2-Tiny transformer backbone for photometric feature extraction, a convolutional heatmap branch for spatial grounding, and a coordinate-based MLP for geometric anchoring. Experimental evaluations on a synthetic Hipparcos-derived dataset demonstrate that Star-Fusion achieves a Top-1 accuracy of 93.4% and a Top-3 accuracy of 97.8%. Furthermore, the model exhibits high computational efficiency, maintaining an inference latency of 18.4 ms on resource-constrained COTS hardware, making it a viable candidate for real-time onboard deployment in next-generation satellite constellations.
>
---
#### [new 060] Evaluating the Alignment Between GeoAI Explanations and Domain Knowledge in Satellite-Based Flood Mapping
- **分类: cs.CV; cs.AI**

- **简介: 论文属于GeoAI解释性评估任务，旨在解决深度学习模型决策过程不透明的问题。通过ADAGE框架，评估模型解释与遥感领域知识的对齐程度，提升模型在实际应用中的可信度与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.26051](https://arxiv.org/pdf/2604.26051)**

> **作者:** Hyunho Lee; Wenwen Li
>
> **备注:** 21 pages, 6 figures, 5 tables
>
> **摘要:** The increasing number of satellites has improved the temporal resolution of Earth observation, making satellite-based flood mapping a promising approach for operational flood monitoring. Deep learning-based approaches for flood mapping using satellite imagery, an important application within Geospatial Artificial Intelligence (GeoAI), have shown improved predictive performance by learning complex spatial and spectral patterns from large volumes of remote sensing data. However, the opaque decision-making processes of deep learning models remain a major barrier to their integration into critical scientific and operational workflows. This highlights the need for a systematic assessment of whether model explanations align with established domain knowledge in remote sensing. To address this research gap, this study introduces the ADAGE (Alignment between Domain Knowledge And GeoAI Explanation Evaluation) framework. The proposed framework is designed to systematically evaluate how well explanations of deep learning models align with established remote sensing knowledge, particularly regarding the distinctive spectral properties of the Earth's surface. The ADAGE framework employs Channel-Group SHAP (SHapley Additive exPlanations) method to estimate the contributions of grouped input channels to pixel-level predictions. Experiments on two satellite-based flood mapping tasks demonstrate that the ADAGE framework can (1) quantitatively assess the alignment between model explanations and reference explanations derived from domain knowledge and (2) help domain experts identify misaligned explanations through alignment scores. This study contributes to bridging the gap between explainability and domain knowledge in GeoAI for Earth observation, enhancing the applicability of GeoAI models in scientific and operational workflows.
>
---
#### [new 061] AnimateAnyMesh++: A Flexible 4D Foundation Model for High-Fidelity Text-Driven Mesh Animation
- **分类: cs.CV**

- **简介: 该论文提出AnimateAnyMesh++，解决高保真文本驱动的3D网格动画生成问题。通过扩展数据集、优化模型架构，提升动画质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.26917](https://arxiv.org/pdf/2604.26917)**

> **作者:** Zijie Wu; Chaohui Yu; Fan Wang; Xiang Bai
>
> **备注:** 14 pages, TPAMI submission, code url: this https URL
>
> **摘要:** Recent advances in 4D content generation have attracted increasing attention, yet creating high-quality animated 3D models remains challenging due to the complexity of modeling spatio-temporal distributions and the scarcity of 4D training data. We present AnimateAnyMesh++, a feed-forward framework for text-driven animation of arbitrary 3D meshes with substantial upgrades in data, architecture, and generative capability. First, we expand the DyMesh-XL dataset by mining dynamic content from Objaverse-XL, increasing the number of unique identities from 60K to 300K and substantially broadening category and motion diversity. Second, we redesign DyMeshVAE-Flex with power-law topology-aware attention and vertex-normal enhanced features, which significantly improves trajectory reconstruction, local geometry preservation, and mitigates trajectory-sticking artifacts. Third, we introduce architectural changes to both DyMeshVAE-Flex and the rectified-flow (RF) generator to support variable-length sequence training and generation, enabling longer animations while preserving reconstruction fidelity. Extensive experiments demonstrate that AnimateAnyMesh++ generates semantically accurate and temporally coherent mesh animations within seconds, surpassing prior approaches in quality and efficiency. The enlarged DyMesh-XL, the upgraded DyMeshVAE-Flex, and variable-length RF together deliver consistent gains across benchmarks and in-the-wild meshes. We will release code, models, and the expanded DyMesh-XL upon acceptance of this manuscript to facilitate research in 4D content creation.
>
---
#### [new 062] Sparsity as a Key: Unlocking New Insights from Latent Structures for Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 该论文属于视觉模型的分布外检测任务，旨在解决现有方法依赖纠缠特征的问题。通过引入稀疏自编码器，提取结构化潜在特征，实现有效检测。**

- **链接: [https://arxiv.org/pdf/2604.26409](https://arxiv.org/pdf/2604.26409)**

> **作者:** Ahyoung Oh; Wonseok Shin; Songkuk Kim
>
> **备注:** 8 pages, 6 figures, supplementary material included, CVPR 2026
>
> **摘要:** Sparse Autoencoders (SAEs) have demonstrated significant success in interpreting Large Language Models (LLMs) by decomposing dense representations into sparse, semantic components. However, their potential for analyzing Vision Transformers (ViTs) remains largely under-explored. In this work, we present the first application of SAEs to the ViT [CLS] token for out-of-distribution (OOD) detection, addressing the limitation of existing methods that rely on entangled feature representations. We propose a novel framework utilizing a Top-k SAE to disentangle the dense [CLS] features into a structured latent space. Through this analysis, we reveal that in-distribution (ID) data exhibits consistent, class-specific activation patterns, which we formalize as Class Activation Profiles (CAPs). Our study uncovers a key structural invariant: while ID samples preserve a stable pattern within CAPs, OOD samples systematically disrupt this structure. Leveraging this insight, we introduce a scoring function based on the divergence of core energy profiles to quantify the deviation from ideal activation profiles. Our method achieves strong results on the FPR95 metric, critical for safety-sensitive applications across multiple benchmarks, while also achieving competitive AUROC. Overall, our findings demonstrate that the sparse, disentangled features revealed by SAEs can serve as a powerful, interpretable tool for robust OOD detection in vision models.
>
---
#### [new 063] Delta Score Matters! Spatial Adaptive Multi Guidance in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型中因全局统一引导导致的细节与伪影矛盾问题。通过几何分析提出SAMG方法，实现自适应引导，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.26503](https://arxiv.org/pdf/2604.26503)**

> **作者:** Haosen Li; Wenshuo Chen; Lei Wang; Shaofeng Liang; Bowen Tian; Soning Lai; Yutao Yue
>
> **摘要:** Diffusion models have achieved remarkable success in synthesizing complex static and temporal visuals, a breakthrough largely driven by Classifier-Free Guidance (CFG). However, despite its pivotal role in aligning generated content with textual prompts, standard CFG relies on a globally uniform scalar. This homogeneous amplification traps models in a well-documented "detail-artifact dilemma": low guidance scales fail to inject intricate semantics, while high scales inevitably cause structural degradation, color over-saturation, and temporal inconsistencies in videos. In this paper, we expose the physical root of this flaw through the lens of differential geometry. By analyzing Tweedie's Formula, we reveal that CFG intrinsically performs a tangential linear extrapolation. Because the natural data manifold is highly curved, this uniform linear step introduces a severe orthogonal deviation. To keep the generation trajectory safely bounded, we formulate a theoretical upper bound for spatial and adaptive guidance. Based on these geometric insights, we propose Spatial Adaptive Multi Guidance (SAMG), a training-free and virtually zero-cost sampling algorithm. SAMG dynamically computes point-wise conditional guidance energy, applying a conservative minimum scale to high-energy boundary regions to preserve delicate micro-textures, while deploying an aggressive maximum scale in low-energy regions to maximize semantic injection. Extensive experiments across diverse image (SD 1.5, SDXL, SD3.5 Medium) and video (CogVideoX, ModelScope) architectures demonstrate that SAMG effectively resolves the detail-artifact dilemma, achieving superior semantic alignment, structural integrity, and temporal smoothness without any computational overhead.
>
---
#### [new 064] MetaSR: Content-Adaptive Metadata Orchestration for Generative Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MetaSR，解决生成式超分辨率中的内容自适应元数据协调问题，通过Diffusion Transformer框架提升图像质量并降低传输比特率。**

- **链接: [https://arxiv.org/pdf/2604.26244](https://arxiv.org/pdf/2604.26244)**

> **作者:** Jiaqi Guo; Mingzhen Li; Haohong Wang; Aggelos K. Katsaggelos
>
> **摘要:** We study generative super-resolution (SR) in real-world scenarios where content and degradations vary across domains, genres, and segments. For example, images and videos may alternate between text overlays, fast motion, smooth cartoons, and low-light faces, each benefiting from different forms of side information. Existing metadata-guided SR methods typically use a fixed conditioning design, which is suboptimal when useful cues are content dependent and transmission budgets are limited. We propose MetaSR, a Diffusion Transformer (DiT)-based framework that selects and injects task-relevant metadata to guide SR under resource constraints. Specifically, we use the DiT's own VAE and transformer backbone to fuse heterogeneous metadata, and adopt an efficient distillation strategy that enables one-step diffusion inference. Experiments across diverse content buckets and degradation regimes show that MetaSR outperforms reference solutions by up to 1.0~dB PSNR while achieving up to 50\% transmission bitrate saving at matched quality. We assess these gains under a rate--distortion optimization (RDO) framework that jointly accounts for sender-side bitrate and receiver/display quality metrics (e.g., PSNR and SSIM).
>
---
#### [new 065] CurEvo: Curriculum-Guided Self-Evolution for Video Understanding
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出CurEvo框架，解决视频理解中自进化学习的结构化问题，通过课程学习引导模型逐步提升，提升视频问答任务的准确性与语义得分。**

- **链接: [https://arxiv.org/pdf/2604.26707](https://arxiv.org/pdf/2604.26707)**

> **作者:** Guiyi Zeng; Junqing Yu; Yi-Ping Phoebe Chen; Xu Chen; Wei Yang; Zikai Song
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Recent advances in self-evolution video understanding frameworks have demonstrated the potential of autonomous learning without human annotations. However, existing methods often suffer from weakly controlled optimization and uncontrolled difficulty progression, as they lack structured guidance throughout the iterative learning process. To address these limitations, we propose CurEvo, a curriculum-guided self-evolution framework that introduces curriculum learning into self-evolution to achieve more structured and progressive model improvement. CurEvo dynamically regulates task difficulty, refines evaluation criteria, and balances data diversity according to model competence, forming a curriculum-guided feedback loop that aligns learning complexity with model capability. Built upon this principle, we develop a multi-dimensional adaptive QA framework that jointly evolves question generation and answer evaluation across perception, recognition, and understanding dimensions, ensuring coherent and measurable curriculum progression. Through this integration, CurEvo transforms weakly controlled self-evolution into a more structured learning process for autonomous video understanding. Across seven backbones, CurEvo consistently improves both benchmark accuracy and evaluator-based semantic score on four VideoQA benchmarks, validating the effectiveness of curriculum-guided self-evolution for video understanding.
>
---
#### [new 066] ProcFunc: Function-Oriented Abstractions for Procedural 3D Generation in Python
- **分类: cs.CV**

- **简介: 该论文提出ProcFunc，用于Blender的Python过程式3D生成。解决过程式3D生成代码编写复杂的问题，提供易用函数库，提升生成效率与多样性。**

- **链接: [https://arxiv.org/pdf/2604.26943](https://arxiv.org/pdf/2604.26943)**

> **作者:** Alexander Raistrick; Karhan Kayan; Jack Nugent; David Yan; Lingjie Mei; Meenal Parakh; Hongyu Wen; Dylan Li; Yiming Zuo; Erich Liang; Jia Deng
>
> **摘要:** We introduce ProcFunc, a library for Blender-based procedural 3D generation in Python. ProcFunc provides a library of easy-to-use Python functions, which streamline creating, combining, analyzing, and executing procedural generation code. ProcFunc makes it easy to create large-scale diverse training data, by combinatorial compositions of semantic components. VLMs can use ProcFunc to edit procedural material and geometry code and can create new procedural code with significantly fewer coding errors. Finally, as an example use case, we use ProcFunc to develop a new procedural generator of indoor rooms, which includes a collection of new compositional procedural materials. We demonstrate the detail, runtime efficiency, and diversity of this room generator, as well as its use for 3D synthetic data generation. Please visit this https URL for source code.
>
---
#### [new 067] DepthPilot: From Controllability to Interpretability in Colonoscopy Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视频生成任务，旨在提升生成视频的可解释性。通过引入深度约束和自适应样条去噪模块，解决生成内容与临床实际不一致的问题。**

- **链接: [https://arxiv.org/pdf/2604.26232](https://arxiv.org/pdf/2604.26232)**

> **作者:** Junhu Fu; Ke Chen; Weidong Guo; Shuyu Liang; Jie Xu; Chen Ma; Kehao Wang; Shengli Lin; Zeju Li; Yuanyuan Wang; Yi Guo; Shuo Li
>
> **摘要:** Controllable medical video generation has achieved remarkable progress, but it still lacks interpretability, which requires the alignment of generated contents with physical priors and faithful clinical manifestations. To push the boundaries from mere controllability to interpretability, we propose DepthPilot, the first interpretable framework for colonoscopy video generation. This work takes a step toward trustworthy generation through two synergistic paradigms. To achieve explicit geometric grounding, DepthPilot devises a prior distribution alignment strategy, injecting depth constraints into the diffusion backbone via parameter-efficient fine-tuning to ensure anatomical fidelity. To enhance intrinsic nonlinear modeling under these geometric constraints, DepthPilot employs an adaptive spline denoising module, replacing fixed linear weights with learnable spline functions to capture complex spatio-temporal dynamics. Extensive evaluations across three public datasets and in-house clinical data confirm DepthPilot's robust ability to produce physically consistent videos. It achieves FID scores below 15 across all benchmarks and ranks first in clinician assessments, bridging the gap between "visually realistic" and "clinically interpretable". Moreover, DepthPilot-generated videos are expected to enable reliable 3D reconstruction, facilitating surgical navigation and blind region identification, and serve as a foundation toward the colorectal world model.
>
---
#### [new 068] Camera-RFID Fusion for Robust Asset Tracking in Forested Environments
- **分类: cs.CV**

- **简介: 该论文属于资产追踪任务，旨在解决森林环境中RFID与视觉系统定位精度低的问题。通过融合摄像头与RFID数据，提升追踪准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26241](https://arxiv.org/pdf/2604.26241)**

> **作者:** John Hateley; Sriram Narasimhan; Omid Abari
>
> **备注:** 11 Pages, 10 Figures, Submitted and awaiting acceptance at IEEE RFID
>
> **摘要:** Passive RFID tags offer a cost-effective and scalable solution for tracking numerous deployed assets. However, in forested environments, signal attenuation and multipath effects generally limit RFID spatial accuracy to the meter level. Conversely, while cameras employing stereo vision can achieve centimeter-level precision, relying solely on computer vision fails to resolve issues arising from spatial association ambiguity and partial occlusions in dense settings. Fusing these modalities allows systems to harness the high-accuracy benefits of vision while retaining the robust, non-line-of-sight identification advantages of RFID. Yet, a primary challenge in achieving this, which is the central focus of this paper, lies in accurately associating the disparate trajectories generated by these two sensors. To overcome this limitation, we introduce a novel camera--RFID fusion framework that integrates depth and object information with advanced trajectory-matching algorithms. By successfully bridging the meter-to-centimeter accuracy gap, the proposed approach helps achieve reliable tag localization even when assets temporarily leave the camera's field of view. To the best of our knowledge, this represents the first application of camera--RFID fusion for asset tracking in natural forested environments.
>
---
#### [new 069] FunFace: Feature Utility and Norm Estimation for Face Recognition
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在解决模型在不同质量样本上的性能问题。提出FunFace损失函数，结合生物特征效用与特征范数，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.26598](https://arxiv.org/pdf/2604.26598)**

> **作者:** Žiga Babnik; Fadi Boutros; Naser Damer; Deepak Kumar Jain; Peter Peer; Vitomir Štruc
>
> **摘要:** Face Recognition (FR) is used in a variety of application domains, from entertainment and banking to security and surveillance. Such applications rely on the FR model to be robust and perform well in a variety of settings. To achieve this, state-of-the-art FR models typically use expressive adaptive margin loss functions, which tie the feature norm to concepts related to sample quality, such as recognizability and perceptual image quality. Recently, through the development of Face Image Quality Assessment (FIQA) techniques, biometric utility has become the preferred measure of face-image quality and has been shown to be a better predictor of the usefulness of samples for face recognition compared to more human-centric aspects, such as resolution, blur, and lighting, tied to general image quality. While image quality expressed through feature norms exhibits a certain level of correlation with biometric utility, it does not fully encapsulate all aspects of utility. To address this point, we propose a new adaptive margin loss, FunFace (Face Recognition Through Utility and Norm Estimation), which incorporates biometric utility, estimated by the Certainty Ratio, into the adaptive margin, taking inspiration from AdaFace. We show that FunFace (when used to train a face recognition model) achieves competitive results to other state-of-the-art FR models on benchmarks containing high-quality samples, while surpassing them on low quality benchmarks.
>
---
#### [new 070] High-Dimensional Noise to Low-Dimensional Manifolds: A Manifold-Space Diffusion Framework for Degraded Hyperspectral Image Classification
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像分类任务，旨在解决复杂退化条件下数据分布失真的问题。通过构建流形空间扩散框架，实现降维与特征稳定优化。**

- **链接: [https://arxiv.org/pdf/2604.26279](https://arxiv.org/pdf/2604.26279)**

> **作者:** Boxiang Yang; Ning Chen; Xia Yue; Yichang Luo; Yingbo Fan; Haoyuan Zhang; Haoyu Ma; Jun Yue; Shanjun Mao
>
> **摘要:** Recently, Hyperspectral Image (HSI) classification has attracted increasing attention in remote sensing. However, HSI data are inherently high-dimensional but low-rank, with discriminative information concentrated on a low-dimensional latent manifold. In real-world remote sensing scenarios, the superposition of multiple degradation factors disrupts this intrinsic manifold structure, driving samples away from their original low-dimensional distribution and introducing substantial redundant and non-discriminative variations. To better handle this challenge, this paper proposes a manifold-space diffusion framework (MSDiff) for robust hyperspectral classification under complex degradation conditions. Specifically, the proposed method first maps high-dimensional, degradation-affected HSI data into a compact low-dimensional manifold through a discriminative spectral-spatial reconstruction task, preserving class semantics and reducing redundant variations. A diffusion-based generative model is then applied to regularize the spectral-spatial distribution within the manifold, enabling progressive refinement and stabilization of latent features against residual degradations. The key advantage of the proposed framework lies in performing diffusion-based distribution modeling directly on the low-dimensional manifold, effectively decoupling degradation-induced disturbances from intrinsic discriminative structures and enhancing representation stability under complex degradations. Experimental results on multiple hyperspectral benchmarks demonstrate consistent performance improvements over state-of-the-art methods under diverse composite degradation settings. The code will be available at this https URL
>
---
#### [new 071] MemOVCD: Training-Free Open-Vocabulary Change Detection via Cross-Temporal Memory Reasoning and Global-Local Adaptive Rectification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于开放词汇变化检测任务，旨在解决时间耦合不足和全局语义不连续的问题。提出MemOVCD框架，通过跨时记忆推理和全局局部自适应校正实现有效变化检测。**

- **链接: [https://arxiv.org/pdf/2604.26774](https://arxiv.org/pdf/2604.26774)**

> **作者:** Zuzheng Kuang; Honghao Chang; Boqiang Liang; Haoqian Wang; Lijun He; Fan Li; Haixia Bi
>
> **摘要:** Open-vocabulary change detection aims to identify semantic changes in bi-temporal remote sensing images without predefined categories. Recent methods combine foundation models such as SAM, DINO and CLIP, but typically process each timestamp independently or interact only at the final comparison stage. Such paradigms suffer from insufficient temporal coupling during semantic reasoning, which limits their ability to distinguish genuine semantic changes from non-semantic appearance discrepancies. In addition, patch-dominant inference on high-resolution images often weakens global semantic continuity and produces fragmented change regions. To address these issues, we propose MemOVCD, a training-free open-vocabulary change detection framework based on cross-temporal memory reasoning and global-local adaptive rectification. Specifically, we reformulate bi-temporal change detection as a two-frame tracking problem and introduce weighted bidirectional propagation to aggregate semantic evidence from both temporal directions. To stabilize memory propagation across large temporal gaps, we construct histogram-aligned transition frames to smooth abrupt appearance changes. Moreover, a global-local adaptive rectification strategy adaptively fuses local and global-view predictions, improving spatial consistency while preserving fine-grained details. Experiments on five benchmarks demonstrate that MemOVCD achieves favorable performance on two change detection tasks, validating its effectiveness and generalization under diverse open-vocabulary settings.
>
---
#### [new 072] Featurising Pixels from Dynamic 3D Scenes with Linear In-Context Learners
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出LILA框架，解决动态3D场景中像素级特征提取问题。通过线性上下文学习，利用视频中的时空线索，提升像素级视觉任务性能。**

- **链接: [https://arxiv.org/pdf/2604.26488](https://arxiv.org/pdf/2604.26488)**

> **作者:** Nikita Araslanov; Martin Sundermeyer; Hidenobu Matsuki; David Joseph Tan; Federico Tombari
>
> **备注:** To appear at CVPR 2026 (oral). Project website: this https URL
>
> **摘要:** One of the most exciting applications of vision models involve pixel-level reasoning. Despite the abundance of vision foundation models, we still lack representations that effectively embed spatio-temporal properties of visual scenes at the pixel level. Existing frameworks either train on image-based pretext tasks, which do not account for dynamic elements, or on video sequences for action-level reasoning, which does not scale to dense pixel-level prediction. We present a framework that learns pixel-accurate feature descriptors from videos, LILA. The core element of our training framework is linear in-context learning. LILA leverages spatio-temporal cue maps -- depth and motion -- estimated with off-the-shelf networks. Despite the noisy nature of those cues, LILA trains effectively on uncurated video datasets, embedding semantic and geometric properties in a temporally consistent manner. We demonstrate compelling empirical benefits of the learned representation across a diverse suite of vision tasks: video object segmentation, surface normal estimation and semantic segmentation.
>
---
#### [new 073] A Multimodal Pre-trained Network for Integrated EEG-Video Seizure Detection
- **分类: cs.CV**

- **简介: 该论文属于癫痫检测任务，解决小鼠模型中癫痫发作识别问题。通过融合EEG和视频数据，提出EEGVFusion框架，提升检测准确性和降低误报率。**

- **链接: [https://arxiv.org/pdf/2604.26379](https://arxiv.org/pdf/2604.26379)**

> **作者:** Tong Lu; Ke Xu; Zimo Zhang; Zitong Zhao; Danwei Weng; Ruiyu Wang; Miao Liu; Zizuo Zhang; Jingyi Yao; Yixuan Zhao; Wenchao Zhang; Min Wang; Guoming Luan; Minmin Luo; Zhifeng Yue
>
> **摘要:** Reliable seizure detection in mouse models is essential for preclinical epilepsy research, yet manual review of synchronized video-EEG recordings is labor-intensive and single-modality systems fail for complementary reasons: video-based methods are easily confounded by benign behaviors, whereas EEG-based methods are vulnerable to ictal motion artifacts. We present EEGVFusion, a multimodal framework that combines self-supervised EEG representation learning, spatio-temporal video encoding, optimal-transport alignment, and bidirectional cross-attention to integrate neural and behavioral evidence. We also curate an expert-annotated dataset of synchronized EEG and video recordings comprising 93 sessions from 15 mice for training and evaluation. In the random-session split, EEGVFusion achieved a Balanced Accuracy of 0.9957 with perfect event sensitivity and an Event FAR of 0.6250 FP/h, indicating strong seizure detection performance with a low false-alarm burden. In a single held-out-subject evaluation with Subject 110 reserved for testing, EEGVFusion achieved a Balanced Accuracy of 0.9718 and reduced Event FAR from 2.7250 FP/h for the EEG-only counterpart to 0.4833 FP/h while preserving perfect event sensitivity. Targeted ablations further showed that EEG pre-training and OT alignment help reduce false alarms while preserving event sensitivity.
>
---
#### [new 074] Sample Selection Using Multi-Task Autoencoders in Federated Learning with Non-IID Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于联邦学习任务，旨在解决非独立同分布数据中的噪声样本问题。通过多任务自编码器和多种方法进行样本选择，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2604.26116](https://arxiv.org/pdf/2604.26116)**

> **作者:** Emre Ardıç; Yakup Genç
>
> **备注:** Published in Engineering Science and Technology, an International Journal, 61 (2025), 101920. DOI: this https URL and Codes: this https URL
>
> **摘要:** Federated learning is a machine learning paradigm in which multiple devices collaboratively train a model under the supervision of a central server while ensuring data privacy. However, its performance is often hindered by redundant, malicious, or abnormal samples, leading to model degradation and inefficiency. To overcome these issues, we propose novel sample selection methods for image classification, employing a multitask autoencoder to estimate sample contributions through loss and feature analysis. Our approach incorporates unsupervised outlier detection, using one-class support vector machine (OCSVM), isolation forest (IF), and adaptive loss threshold (AT) methods managed by a central server to filter noisy samples on clients. We also propose a multi-class deep support vector data description (SVDD) loss controlled by a central server to enhance feature-based sample selection. We validate our methods on CIFAR10 and MNIST datasets across varying numbers of clients, non-IID distributions, and noise levels up to 40%. The results show significant accuracy improvements with loss-based sample selection, achieving gains of up to 7.02% on CIFAR10 with OCSVM and 1.83% on MNIST with AT. Additionally, our federated SVDD loss further improves feature-based sample selection, yielding accuracy gains of up to 0.99% on CIFAR10 with OCSVM. These results show the effectiveness of our methods in improving model accuracy across various client counts and noise conditions.
>
---
#### [new 075] SnapPose3D: Diffusion-Based Single-Frame 2D-to-3D Lifting of Human Poses
- **分类: cs.CV**

- **简介: 该论文属于3D人体姿态估计任务，解决2D到3D姿态提升中的深度模糊和关节不确定性问题。提出SnapPose3D框架，通过扩散模型生成并聚合多个假设，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.26620](https://arxiv.org/pdf/2604.26620)**

> **作者:** Alessandro Simoni; Riccardo Catalini; Davide Di Nucci; Guido Borghi; Davide Davoli; Lorenzo Garattoni; Gianpiero Francesca; Yuki Kawana; Roberto Vezzani
>
> **备注:** Accepted at ICPR 2026
>
> **摘要:** Depth ambiguity and joint uncertainty are the two main obstacles in obtaining accurate human pose predictions by 2D-to-3D lifting methods proposed in the literature. In particular, these issues are caused by 2D joint locations that can be mapped to multiple 3D positions, inducing multiple possible final poses. Following these considerations, we propose leveraging diffusion-based models generation capability to predict multiple hypotheses and aggregate them in a final accurate pose. Therefore, we introduce SnapPose3D, a pose-lifting framework trained deterministically to denoise 3D poses conditioned on both visual context and 2D pose features. SnapPose3D adopts a probabilistic approach during inference, generating multiple hypotheses through random sampling from a unit Gaussian distribution. Unlike most previous methods that address pose ambiguity by processing temporal sequences, SnapPose3D uses single frames as input, avoiding tracking and limiting computational cost, data acquisition complexity, and the need for online, real-time applications. We extensively evaluate SnapPose3D on well-known benchmarks for the 3D human pose estimation task showing its ability to generate and aggregate accurate hypotheses that lead to state-of-the-art results.
>
---
#### [new 076] Motion-Driven Multi-Object Tracking of Model Organisms in Space Science Experiments
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，解决微重力环境下生物视频的多动物跟踪难题。提出ART-Track框架，提升轨迹稳定性与身份一致性。**

- **链接: [https://arxiv.org/pdf/2604.26321](https://arxiv.org/pdf/2604.26321)**

> **作者:** Jianing You; Han Wang; Kang Liu; Jiale Ding; Fengjie Chu; Zihan Guo; Shengyang Li
>
> **备注:** 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
>
> **摘要:** Automated animal behavior analysis relies on long-term, interpretable individual trajectories; however, multi-animal tracking in space science experimental videos remains highly challenging due to weak appearance cues, low-quality imaging, complex maneuvering behaviors, and frequent interactions. To address this problem, we first construct the SpaceAnimal-MOT dataset to characterize the motion complexity and long-term identity preservation challenges in biological videos acquired under microgravity conditions. We then propose ART-Track (Adaptive Robust Tracking), a motion-driven tracking framework tailored to this setting. Specifically, multi-model motion estimation is introduced to handle abrupt maneuvers and nonlinear motion, motion-state-driven association is designed to reduce identity switches under dense interactions and temporary mismatch, and uncertainty-adaptive fusion is used to dynamically balance spatial and motion cues when prediction reliability varies. Experimental results show that ART-Track significantly reduces identity switches on zebrafish and fruitfly sequences, while maintaining more stable association under occlusion, deformation, and high-density interactions, thereby providing a more reliable tracking foundation for downstream quantitative behavior analysis. The code is publicly available at this https URL.
>
---
#### [new 077] A Data-Centric Framework for Intraoperative Fluorescence Lifetime Imaging for Glioma Surgical Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决术中胶质瘤边界评估问题。通过数据驱动的AI框架提升FLIm分类准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.26147](https://arxiv.org/pdf/2604.26147)**

> **作者:** Silvia Noble Anbunesan; Mohamed Abul Hassan; Jinyi Qi; Lisanne Kraft; Han Sung Lee; Orin Bloch; Laura Marcu
>
> **摘要:** Accurate intraoperative assessment of glioma infiltration is essential for maximizing tumor resection while preserving functional brain tissue. Fluorescence lifetime imaging (FLIm) offers real-time, label-free biochemical contrast, but its clinical utility is challenged by biological heterogeneity, class imbalance, and variability in histopathological labeling. We present a data-centric AI (DC-AI) framework that integrates confident learning (CL), class refinement, and targeted label evaluation to develop a robust multi-class FLIm classifier for glioblastoma (GBM) resection margins. FLIm data were collected from 192 tissue margins across 31 newly diagnosed IDH-wildtype GBM patients and initially labeled into seven tumor cellularity classes by an expert neuropathologist. CL was applied to quantify FLIm point-level confidence, identify label inconsistencies, and guide iterative class merging into a three-class scheme ("low", "moderate", "high"). The resulting high-fidelity dataset enabled training a model that achieved 96% accuracy in the three-class task. SHAP analysis revealed class-specific FLIm feature importance, highlighting distinct optical signatures across the infiltration spectrum. Targeted FLIm analysis further identified biological (e.g., gray matter composition) and acquisition-related (e.g., blood contamination) contributors to low-confidence predictions. Blinded re-evaluation of margins flagged by CL demonstrated intra-pathologist variability, underscoring the value of selective relabeling rather than exhaustive review. Together, these findings demonstrate that a DC-AI framework can systematically improve data reliability, enhance model robustness, and refine biological interpretation of FLIm signals, supporting the development of clinically actionable optical tools for real-time glioma margin assessment.
>
---
#### [new 078] Are Data Augmentation and Segmentation Always Necessary? Insights from COVID-19 X-Rays and a Methodology Thereof
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决COVID-19检测中的模型可靠性问题。通过分析肺部分割与数据增强的影响，提出SDL-COVID方法提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2604.26437](https://arxiv.org/pdf/2604.26437)**

> **作者:** Aman Swaraj; Arnav Agarwal; Hitendra Singh Bhadouria; Sandeep Kumar; Karan Verma
>
> **摘要:** Purpose: Rapid and reliable diagnostic tools are crucial for managing respiratory diseases like COVID-19, where chest X-ray analysis coupled with artificial intelligence techniques has proven invaluable. However, most existing works on X-ray images have not considered lung segmentation, raising concerns about their reliability. Additionally, some have employed disproportionate and impractical augmentation techniques, making models less generalized and prone to overfitting. This study presents a critical analysis of both issues and proposes a methodology (SDL-COVID) for more reliable classification of chest X-rays for COVID-19 detection. Methods: We use class activation mapping to obtain a visual understanding of the predictions made by Convolutional Neural Networks (CNNs), validating the necessity of lung segmentation. To analyze the effect of data augmentation, deep learning models are implemented on two levels: one for an augmented dataset and another for a non-augmented dataset. Results: Careful analysis of X-ray images and their corresponding heat maps under expert medical supervision reveals that lung segmentation is necessary for accurate COVID-19 prediction. Regarding data augmentation, test accuracy significantly drops beyond a certain threshold with additional augmented images, indicating model overfitting. Conclusion: Our proposed methodology, SDL-COVID, achieves a precision of 95.21% and a lower false negative rate, ensuring its reliability for COVID-19 detection using chest X-rays.
>
---
#### [new 079] Multiple Consistent 2D-3D Mappings for Robust Zero-Shot 3D Visual Grounding
- **分类: cs.CV**

- **简介: 该论文针对零样本3D视觉定位任务，解决3D提议质量差和空间冗余问题，提出MCM-VG框架，通过多一致2D-3D映射提升定位精度与推理可靠性。**

- **链接: [https://arxiv.org/pdf/2604.26261](https://arxiv.org/pdf/2604.26261)**

> **作者:** Yufei Yin; Jie Zheng; Qianke Meng; Zhou Yu; Minghao Chen; Jiajun Ding; Min Tan; Yuling Xi; Zhiwen Chen; Chengfei Lv
>
> **摘要:** Zero-shot 3D Visual Grounding (3DVG) is a critical capability for open-world embodied AI. However, existing methods are fundamentally bottlenecked by the poor quality of open-vocabulary 3D proposals, suffering from inaccurate categories and imprecise geometries, as well as the spatial redundancy of exhaustive multi-view reasoning. To address these challenges, we propose MCM-VG, a novel framework that achieves robust zero-shot 3DVG by explicitly establishing Multiple Consistent 2D-3D Mappings. Instead of passively relying on noisy 3D segments, MCM-VG enforces 2D-3D consistency across three fundamental dimensions to achieve precise target localization and reliable reasoning. First, a Semantic Alignment module corrects category mismatches via LLM-driven query parsing and coarse-to-fine 2D-3D matching. Second, an Instance Rectification module leverages VLM-guided 2D segmentations to reconstruct missing targets, back-projecting these reliable visual priors to establish accurate 3D geometries. Finally, to eliminate spatial redundancy, a Viewpoint Distillation module clusters 3D camera directions to extract optimal frames. By pairing these optimal RGB frames with Bird's Eye View maps into concise visual prompt sets, we formulate the final target disambiguation as a multiple-choice reasoning task for Vision-Language Models. Extensive evaluations on ScanRefer and Nr3D benchmarks demonstrate that MCM-VG sets a new state-of-the-art for zero-shot 3D visual grounding. Remarkably, it achieves 62.0\% and 53.6\% in Acc@0.25 and Acc@0.5 on ScanRefer, outperforming previous baselines by substantial margins of 6.4\% and 4.0\%.
>
---
#### [new 080] Seamless Indoor-Outdoor Mapping for INGENIOUS First Responders
- **分类: cs.CV**

- **简介: 该论文属于室内室外无缝建模任务，旨在解决灾难响应中需融合室内外三维模型的问题。通过结合无人机与地面定位系统，实现自动注册与实时可视化。**

- **链接: [https://arxiv.org/pdf/2604.26368](https://arxiv.org/pdf/2604.26368)**

> **作者:** Jürgen Wohlfeil; Henry Meißner; Adrian Schischmanow; Thomas Kraft; Dirk Baumbach; Ines Ernst; Dennis Dahlke
>
> **摘要:** In several applications it is desired to have 3D models not only from the outdoor spaces but also from inside the building. In the context of First Responder enhancement in large scale natural and man-made disasters, a method is presented to achieve this goal with a high degree of automation. Therefore an autonomously flying aerial mapping system is combined with a person-carried indoor positioning system. Automatically recognized markers (AprilTags) are geo-referenced by the aerial system and their coordinates are sent to the ground-based system. By looking at the AprilTags before entering the building, the ground-based system is registered to world coordinates. Without the further need of any global positioning, it creates a point cloud from the indoor spaces that fits with the point could from the aerial view. This allows a co-visualization of both point-clouds as a seamless indoor-outdoor 3D model in real time.
>
---
#### [new 081] Seeking Consensus: Geometric-Semantic On-the-Fly Recalibration for Open-Vocabulary Remote Sensing Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像的开放词汇语义分割任务，旨在解决场景分布差异导致的语义模糊和前景激活不全问题。提出SeeCo框架，通过几何与语义共识学习实现无需训练的在线校准。**

- **链接: [https://arxiv.org/pdf/2604.26221](https://arxiv.org/pdf/2604.26221)**

> **作者:** Guanchun Wang; Chenxiao Wu; Xiangrong Zhang; Zelin Peng; Jianxun Lai; Tianyang Zhang; Xu Tang
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) in remote sensing images is a promising task that employs textual descriptions for identifying undefined land cover categories. Despite notable advances, existing methods typically employ a static inference paradigm, overlooking the distinct distribution of each scene, resulting in semantic ambiguity in diverse land covers and incomplete foreground activation. Motivated by this, we propose Seeking Consensus, termed SeeCo, a plug-and-play framework to boost the performance of training-free OVSS models in remote sensing images, which recalibrates arbitrary OVSS models on-the-fly by seeking dual consensus: geometric consensus learning (GCL) through multi-view consistent observations and semantic consensus learning (SCL) via textual description adaptive calibration, which assists collaborative recalibration of visual and textual semantics. The two consensus are injected via an online consensus injector (OCI), effectively alleviating the under-activation and semantic bias. SeeCo requires no specific training process, yet recalibrates semantic-geometric alignment for each unique scene during inference. Extensive experiments on eight remote sensing OVSS benchmarks show consistent gains, proving its effectiveness and universality.
>
---
#### [new 082] Virtual-reality based patient-specific simulation of spine surgical procedures: A fast, highly automated and high-fidelity system for surgical education and planning
- **分类: cs.CV**

- **简介: 该论文属于医学影像与虚拟现实结合的任务，旨在解决传统手术培训中缺乏个性化模拟的问题。通过AI技术生成患者特定的3D模型，并进行VR手术模拟，提升培训效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.26781](https://arxiv.org/pdf/2604.26781)**

> **作者:** Raj Kumar Ranabhat; Tayler D Ross; Tony Jiao; Jeremie Larouche; Joel Finkelstein; Michael Hardisty
>
> **摘要:** Surgical training involves didactic teaching, mentor-led learning, surgical skills laboratories, and direct exposure to surgery; however, increasing clinical pressures have limited operating room (OR) exposure. This work leverages virtual reality (VR) to provide a safe and immersive training environment. Existing VR training is often based on standardized scenarios not tailored to individual clinical cases. This study addresses this limitation using artificial intelligence (AI) based computer vision methods to generate patient-specific simulations from computed tomography (CT) and magnetic resonance imaging (MRI). This study focuses on patient-specific spinal decompression simulation for spinal stenosis in a virtual operating room. The objectives were (1) automatic creation of 3D anatomical models and (2) VR simulation of spinal decompression procedures including laminectomy, disc resection, and foraminotomy. Model construction required multimodal fusion (registration) of CT and MRI and segmentation of relevant structures. Segmentation was evaluated using the Dice Similarity Coefficient (DSC), and registration accuracy using Target Registration Error (TRE). Qualitative feedback was obtained from surgeons and trainees. High-fidelity patient-specific 3D models were generated efficiently (approximately 2.5 minutes per case, N = 15). Segmentation accuracy was high, with a DSC of 0.95 (+/- 0.03) for vertebral bone and 0.895 (+/- 0.02) for soft tissue structures. Registration accuracy showed a mean TRE of 1.73 (+/- 0.42) mm. Semi-structured interviews indicated improved spatial understanding, increased procedural confidence, and strong perceived educational value. This platform significantly reduced the time and costs of patient-specific modelling, thereby facilitating pre-operative planning, post-procedural assessments, and comprehensive surgical simulation.
>
---
#### [new 083] MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉语言模型任务，旨在解决诊断记忆缺失问题。通过引入隐式记忆演化框架，提升模型的诊断准确性。**

- **链接: [https://arxiv.org/pdf/2604.26283](https://arxiv.org/pdf/2604.26283)**

> **作者:** Chunzheng Zhu; Jiaqi Zeng; Junyu Jiang; Jianxin Lin; Yijun Wang
>
> **备注:** Medical latent reasoning; Memory evolution
>
> **摘要:** High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose ours, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model's hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR), which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privileged-autonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets demonstrate that ours, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-the-art methods, particularly chain-of-thought paradigms, in diagnostic accuracy.
>
---
#### [new 084] Learning Sparse BRDF Measurement Samples from Image
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于BRDF重建任务，旨在用少量测量样本来提高材质外观的重建质量。通过学习采样策略，优化测量位置以提升重建效果。**

- **链接: [https://arxiv.org/pdf/2604.26740](https://arxiv.org/pdf/2604.26740)**

> **作者:** Wen Cao
>
> **摘要:** Accurate BRDF acquisition is important for realistic rendering, but dense gonioreflectometer measurements are slow and expensive. We study how to select a small number of BRDF measurements that are most useful for reconstructing material appearance under a learned reflectance prior. Our method combines a set encoder for sparse coordinate-value observations, a pretrained hypernetwork-based BRDF reconstructor, and a differentiable renderer. During sampler training, the reconstructor is kept fixed and gradients from BRDF-space and rendered-image losses are used to optimize measurement locations. This separates sample selection from prior fitting and encourages the sampler to choose directions that are informative under the learned material distribution. Experiments on the MERL dataset show that the proposed sampler improves low-budget reconstruction quality at 8 and 16 measurements compared with neural reconstruction baselines, while PCA-based methods remain strong at larger budgets. We further analyze the effect of image-space supervision, co-optimization, and image-only latent fitting for unseen materials.
>
---
#### [new 085] OmniTrend: Content-Context Modeling for Scalable Social Popularity Prediction
- **分类: cs.CV**

- **简介: 该论文属于社会流行度预测任务，旨在解决现有方法无法区分内容吸引力与暴露上下文的问题。提出OmniTrend框架，分别建模内容与上下文因素，提升可解释性与跨平台迁移能力。**

- **链接: [https://arxiv.org/pdf/2604.26252](https://arxiv.org/pdf/2604.26252)**

> **作者:** Liliang Ye; Guiyi Zeng; Yunyao Zhang; Yi-Ping Phoebe Chen; Junqing Yu; Zikai Song
>
> **摘要:** Predicting social media popularity requires understanding both the intrinsic appeal of content and the external context that determines how it is exposed to users. Existing methods focus on content signals but do not separate them from exposure-related patterns, which causes the learned representations to absorb platform-specific visibility effects and weakens both interpretability and cross-platform transfer. This paper introduces OmniTrend, a unified framework that models popularity as the joint outcome of content attractiveness and contextual exposure. The content module learns cross-modal representations from visual, audio, and textual cues to quantify intrinsic appeal, while the context module estimates exposure from exogenous signals such as posting time, author activity, topical trends, and retrieval-based neighborhood statistics. OmniTrend learns separate predictors for content attractiveness and contextual exposure and integrates them in the final popularity estimate, which makes the role of each factor explicit and supports robust transfer across image and video platforms.
>
---
#### [new 086] Adaptive Transform Coding for Semantic Compression
- **分类: eess.IV; cs.CV; cs.IT; eess.SP**

- **简介: 该论文属于视觉数据压缩任务，解决机器感知下的语义特征高效编码问题。提出自适应变换编码方法，提升异构特征分布的压缩效率。**

- **链接: [https://arxiv.org/pdf/2604.26492](https://arxiv.org/pdf/2604.26492)**

> **作者:** Andriy Enttsel; Vincent Corlay
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Visual data compression is shifting from human-centered reconstruction to machine-oriented representation coding. In this setting, an image is often mapped to a compact semantic embedding, which is then compressed and transmitted for downstream inference. We propose an adaptive transform-coding method for semantic-feature compression motivated by the conditional rate-distortion function of a Gaussian mixture model. The scheme uses mode-dependent transforms and quantizers selected according to the inferred source component, enabling more efficient coding of heterogeneous feature distributions. Evaluations on features from widely used vision backbones and foundation models show that the proposed method outperforms or is competitive with state-of-the-art neural compression methods while preserving flexibility and interpretability.
>
---
#### [new 087] SAND: Spatially Adaptive Network Depth for Fast Sampling of Neural Implicit Surfaces
- **分类: cs.GR; cs.CV; eess.IV**

- **简介: 该论文属于几何建模任务，解决隐式神经表示计算成本高的问题。提出SAND框架，通过自适应网络深度提升采样效率。**

- **链接: [https://arxiv.org/pdf/2604.25936](https://arxiv.org/pdf/2604.25936)**

> **作者:** Chuanxiang Yang; Junhui Hou; Yuan Liu; Siyu Ren; Guangshun Wei; Taku Komura; Yuanfeng Zhou; Wenping Wang
>
> **摘要:** Implicit neural representations are powerful for geometric modeling, but their practical use is often limited by the high computational cost of network evaluations. We observe that implicit representations require progressively lower accuracy as query points move farther from the target surface, and that even within the same iso-surface, representation difficulty varies spatially with local geometric complexity. However, conventional neural implicit models evaluate all query points with the same network depth and computational cost, ignoring this spatial variation and thereby incurring substantial computational waste. Motivated by this observation, we propose an efficient neural implicit geometry representation framework with spatially adaptive network depth (SAND). SAND leverages a volumetric network-depth map together with a tailed multi-layer perceptron (T-MLP) to model implicit representation. The volumetric depth map records, for each spatial region, the network depth required to achieve sufficient accuracy, while the T-MLP is a modified MLP designed to learn implicit functions such as signed distance functions, where an output branch, referred to as a tail, is attached to each hidden layer. This design allows network evaluation to terminate adaptively without traversing the full network and directs computational resources to geometrically important and complex regions, improving efficiency while preserving high-fidelity representations. Extensive experimental results demonstrate that our approach can significantly improve the inference-time query speed of implicit neural representations.
>
---
#### [new 088] Progressive Semantic Communication for Efficient Edge-Cloud Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CV; cs.DC; cs.NI**

- **简介: 该论文属于边缘-云视觉语言模型任务，解决资源受限设备部署难题。提出渐进语义通信框架，通过自适应压缩减少传输成本，提升效率与语义一致性。**

- **链接: [https://arxiv.org/pdf/2604.26508](https://arxiv.org/pdf/2604.26508)**

> **作者:** Cyril Shih-Huan Hsu; Wig Yuan-Cheng Cheng; Chrysa Papagianni
>
> **备注:** Under review. Extended version with additional figures and appendices
>
> **摘要:** Deploying Vision-Language Models (VLMs) on edge devices remains challenging due to their substantial computational and memory demands, which exceed the capabilities of resource-constrained embedded platforms. Conversely, fully offloading inference to the cloud is often impractical in bandwidth-limited environments, where transmitting raw visual data introduces substantial latency overhead. While recent edge-cloud collaborative architectures attempt to partition VLM workloads across devices, they typically rely on transmitting fixed-size representations, lacking adaptability to dynamic network conditions and failing to fully exploit semantic redundancy. In this paper, we propose a progressive semantic communication framework for edge-cloud VLM inference, using a Meta AutoEncoder that compresses visual tokens into adaptive, progressively refinable representations, enabling plug-and-play deployment with off-the-shelf VLMs without additional fine-tuning. This design allows flexible transmission at different information levels, providing a controllable trade-off between communication cost and semantic fidelity. We implement a full end-to-end edge-cloud system comprising an embedded NXP i.MX95 platform and a GPU server, communicating over bandwidth-constrained networks. Experimental results show that, at 1 Mbps uplink, the proposed progressive scheme significantly reduces network latency compared to full-edge and full-cloud solutions, while maintaining high semantic consistency even under high compression. The implementation code will be released upon publication at this https URL.
>
---
#### [new 089] Circular Phase Representation and Geometry-Aware Optimization for Ptychographic Image Reconstruction
- **分类: eess.IV; cs.CV; physics.optics**

- **简介: 该论文属于ptychographic图像重建任务，旨在解决深度学习中相位预测的周期性问题。提出基于单位圆的相位表示和几何感知优化方法，提升重建精度与速度。**

- **链接: [https://arxiv.org/pdf/2604.26664](https://arxiv.org/pdf/2604.26664)**

> **作者:** Carson Yu Liu; Jun Cheng; Chien-Chun Chen; Steve F. Shu
>
> **摘要:** Traditional iterative reconstruction methods are accurate but computationally expensive, limiting their use in high-throughput and real-time ptychography. Recent deep learning approaches improve speed, but often predict phase as a Euclidean scalar despite its $2\pi$ periodicity, which can introduce wrapping artifacts, discontinuities at $\pm\pi$, and a mismatch between the loss and the underlying signal geometry. We present a deep learning framework for ptychographic reconstruction that models phase on the unit circle using cosine and sine components. Phase error is optimized with a differentiable geodesic loss, which avoids branch-cut discontinuities and provides bounded gradients. The network further incorporates saturation-aware dual-gain input scaling, parallel encoder branches, and three decoders for amplitude, cosine, and sine prediction, together with a composite loss that promotes circular consistency and structural fidelity. Experiments on synthetic and experimental datasets show consistent improvements in both amplitude and phase reconstruction over existing deep learning methods. Frequency-domain analysis further shows better preservation of mid- and high-frequency phase content. The proposed method also provides substantial speedup over iterative solvers while maintaining physically consistent reconstructions.
>
---
#### [new 090] KAYRA: A Microservice Architecture for AI-Assisted Karyotyping with Cloud and On-Premise Deployment
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出KAYRA，一个用于染色体核型分析的微服务架构系统，解决临床环境中数据隐私与部署灵活性问题，采用多模型AI实现高精度分割、分类与旋转识别。**

- **链接: [https://arxiv.org/pdf/2604.26869](https://arxiv.org/pdf/2604.26869)**

> **作者:** Attila Pintér; Javier Rico; Attila Répai; Jalal Al-Afandi; Adrienn Éva Borsy; András Kozma; Hajnalka Andrikovics; György Cserey
>
> **摘要:** We present KAYRA, an end-to-end karyotyping system that operates inside the operational constraints of a clinical cytogenetic laboratory. KAYRA is architected as a containerized microservice pipeline whose ML stack combines an EfficientNet-B5 + U-Net semantic segmenter, a Mask R-CNN (ResNet-50 + FPN) instance detector, and a ResNet-18 classifier, orchestrated through a cascaded ROI-narrowing strategy that focuses each downstream model on the chromosome-bearing region. The same container images are deployed both as a cloud service and as an on-premise installation, supporting clinical environments where patient-data egress is not permitted as well as those where it is. A pilot clinical evaluation against two commercial reference karyotyping systems on 459 chromosomes from 10 metaphase spreads shows segmentation accuracy of 98.91 % (vs. 78.21 % / 40.52 %), classification accuracy of 89.1 % (vs. 86.9 % / 54.5 %), and rotation accuracy of 89.76 % (vs. 94.55 % / 78.43 %). KAYRA improves over the older density-thresholding reference on all three axes (p < 0.0001 for segmentation and classification by Fisher's exact test on chromosome-level counts), and on segmentation also against the modern AI- supported reference (p < 0.0001); on classification the difference vs. the modern AI reference is not statistically significant at the present test-set size (p = 0.34). The system reaches TRL 6 maturity and integrates the human-in-the-loop expert-review workflow that diagnostic cytogenetic practice requires. The thesis of this paper is that a multi-model cytogenetic AI service can be packaged as a microservice architecture supporting flexible deployment - cloud-hosted or on-premise - while delivering strong empirical performance on a pilot clinical evaluation.
>
---
#### [new 091] Grounding vs. Compositionality: On the Non-Complementarity of Reasoning in Neuro-Symbolic Systems
- **分类: cs.AI; cs.CV; cs.LG; cs.LO**

- **简介: 该论文属于神经符号系统研究，旨在解决神经网络在组合泛化上的不足。通过提出iLTN模型，验证了符号接地不足以实现泛化，需显式学习推理能力。**

- **链接: [https://arxiv.org/pdf/2604.26521](https://arxiv.org/pdf/2604.26521)**

> **作者:** Mahnoor Shahid; Hannes Rothe
>
> **备注:** Accepted at AAAI MAKE 2026
>
> **摘要:** Compositional generalization remains a foundational weakness of modern neural networks, limiting their robustness and applicability in domains requiring out-of-distribution reasoning. A central, yet unverified, assumption in neuro-symbolic AI is that compositional reasoning will emerge as a byproduct of successful symbol grounding. This work presents the first systematic empirical analysis to challenge this assumption by disentangling the contributions of grounding and reasoning. To operationalize this investigation, we introduce the Iterative Logic Tensor Network ($i$LTN), a fully differentiable architecture designed for multi-step deduction. Using a formal taxonomy of generalization -- probing for novel entities, unseen relations, and complex rule compositions -- we demonstrate that a model trained solely on a grounding objective fails to generalize. In contrast, our full $i$LTN, trained jointly on perceptual grounding and multi-step reasoning, achieves high zero-shot accuracy across all tasks. Our findings provide conclusive evidence that symbol grounding, while necessary, is insufficient for generalization, establishing that reasoning is not an emergent property but a distinct capability that requires an explicit learning objective.
>
---
#### [new 092] Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出X-WAM，解决机器人行动与4D世界建模的统一问题，通过视频先验和异步去噪提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.26694](https://arxiv.org/pdf/2604.26694)**

> **作者:** Jun Guo; Qiwei Li; Peiyan Li; Zilong Chen; Nan Sun; Yifei Su; Heyun Wang; Yuan Zhang; Xinghang Li; Huaping Liu
>
> **备注:** Project website: this https URL
>
> **摘要:** We propose X-WAM, a Unified 4D World Model that unifies real-time robotic action execution and high-fidelity 4D world synthesis (video + 3D reconstruction) in a single framework, addressing the critical limitations of prior unified world models (e.g., UWM) that only model 2D pixel-space and fail to balance action efficiency and world modeling quality. To leverage the strong visual priors of pretrained video diffusion models, X-WAM imagines the future world by predicting multi-view RGB-D videos, and obtains spatial information efficiently through a lightweight structural adaptation: replicating the final few blocks of the pretrained Diffusion Transformer into a dedicated depth prediction branch for the reconstruction of future spatial information. Moreover, we propose Asynchronous Noise Sampling (ANS) to jointly optimize generation quality and action decoding efficiency. ANS applies a specialized asynchronous denoising schedule during inference, which rapidly decodes actions with fewer steps to enable efficient real-time execution, while dedicating the full sequence of steps to generate high-fidelity video. Rather than entirely decoupling the timesteps during training, ANS samples from their joint distribution to align with the inference distribution. Pretrained on over 5,800 hours of robotic data, X-WAM achieves 79.2% and 90.7% average success rate on RoboCasa and RoboTwin 2.0 benchmarks, while producing high-fidelity 4D reconstruction and generation surpassing existing methods in both visual and geometric metrics.
>
---
#### [new 093] 3D Generation for Embodied AI and Robotic Simulation: A Survey
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D生成任务，旨在解决 embodied AI 和机器人仿真中的内容生成问题。工作包括分类3D生成在数据生成、仿真环境构建和 sim2real 桥接中的角色，并分析当前瓶颈。**

- **链接: [https://arxiv.org/pdf/2604.26509](https://arxiv.org/pdf/2604.26509)**

> **作者:** Tianwei Ye; Yifan Mao; Minwen Liao; Jian Liu; Chunchao Guo; Dazhao Du; Quanxin Shou; Fangqi Zhu; Song Guo
>
> **备注:** 26 pages, 11 figures, 8 tables. Project Page: this https URL
>
> **摘要:** Embodied AI and robotic systems increasingly depend on scalable, diverse, and physically grounded 3D content for simulation-based training and real-world deployment. While 3D generative modeling has advanced rapidly, embodied applications impose requirements far beyond visual realism: generated objects must carry kinematic structure and material properties, scenes must support interaction and task execution, and the resulting content must bridge the gap between simulation and reality. This survey presents the first survey of 3D generation for embodied AI and organizes the literature around three roles that 3D generation plays in embodied systems. In \emph{Data Generator}, 3D generation produces simulation-ready objects and assets, including articulated, physically grounded, and deformable content for downstream interaction; in \emph{Simulation Environments}, it constructs interactive and task-oriented worlds, spanning structure-aware, controllable, and agentic scene generation; and in \emph{Sim2Real Bridge}, it supports digital twin reconstruction, data augmentation, and synthetic demonstrations for downstream robot learning and real-world transfer. We also show that the field is shifting from visual realism toward interaction readiness, and we identify the main bottlenecks, including limited physical annotations, the gap between geometric quality and physical validity, fragmented evaluation, and the persistent sim-to-real divide, that must be addressed for 3D generation to become a dependable foundation for embodied intelligence. Our project page is at this https URL.
>
---
## 更新

#### [replaced 001] Revisiting Human-in-the-Loop Object Retrieval with Pre-Trained Vision Transformers
- **分类: cs.CV; cs.HC; cs.IR**

- **链接: [https://arxiv.org/pdf/2604.00809](https://arxiv.org/pdf/2604.00809)**

> **作者:** Kawtar Zaher; Olivier Buisson; Alexis Joly
>
> **摘要:** Building on existing approaches, we revisit Human-in-the-Loop Object Retrieval, a task that consists of iteratively retrieving images containing objects of a class-of-interest, specified by a user-provided query. Starting from a large unlabeled image collection, the aim is to rapidly identify diverse instances of an object category relying solely on the initial query and the user's Relevance Feedback, with no prior labels. The retrieval process is formulated as a binary classification task, where the system continuously learns to distinguish between relevant and non-relevant images to the query, through iterative user interaction. This interaction is guided by an Active Learning loop: at each iteration, the system selects informative samples for user annotation, thereby refining the retrieval performance. This task is particularly challenging in multi-object datasets, where the object of interest may occupy only a small region of the image within a complex, cluttered scene. Unlike object-centered settings where global descriptors often suffice, multi-object images require more adapted, localized descriptors. In this work, we formulate and revisit the Human-in-the-Loop Object Retrieval task by leveraging pre-trained ViT representations, and addressing key design questions, including which object instances to consider in an image, what form the annotations should take, how Active Selection should be applied, and which representation strategies best capture the object's features. We compare several representation strategies across multi-object datasets highlighting trade-offs between capturing the global context and focusing on fine-grained local object details. Our results offer practical insights for the design of effective interactive retrieval pipelines based on Active Learning for object class retrieval.
>
---
#### [replaced 002] Robust Grounding with MLLMs Against Occlusion and Small Objects via Language-Guided Semantic Cues
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2604.24036](https://arxiv.org/pdf/2604.24036)**

> **作者:** Beomchan Park; Seongho Kim; Hyunjun Kim; Sungjune Park; Yong Man Ro
>
> **备注:** 4 pages, 2 figures, ICASSP 2026
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have enhanced grounding capabilities in general scenes, their robustness in crowded scenes remains underexplored. Crowded scenes entail visual challenges (i.e., occlusion and small objects), which impair object semantics and degrade grounding performance. In contrast, language expressions are immune to such degradation and preserve object semantics. In light of these observations, we propose a novel method that overcomes such constraints by leveraging Language-Guided Semantic Cues (LGSCs). Specifically, our approach introduces a Semantic Cue Extractor (SCE) to derive semantic cues of objects from the visual pipeline of an MLLM. We then guide these cues using corresponding text embeddings to produce LGSCs as linguistic semantic priors. Subsequently, they are reintegrated into the original visual pipeline to refine object semantics. Extensive experiments and analyses demonstrate that incorporating LGSCs into an MLLM effectively improves grounding accuracy in crowded scenes.
>
---
#### [replaced 003] Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于CAD生成任务，旨在解决命令序列无法支持实体选择及拓扑错误的问题。提出Pointer-CAD框架，通过指针选择几何实体，提升模型精度。**

- **链接: [https://arxiv.org/pdf/2603.04337](https://arxiv.org/pdf/2603.04337)**

> **作者:** Dacheng Qi; Chenyu Wang; Jingwei Xu; Tianzhe Chu; Zibo Zhao; Wen Liu; Wenrui Ding; Yi Ma; Shenghua Gao
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Constructing computer-aided design (CAD) models is labor-intensive but essential for engineering and manufacturing. Recent advances in Large Language Models (LLMs) have inspired the LLM-based CAD generation by representing CAD as command sequences. But these methods struggle in practical scenarios because command sequence representation does not support entity selection (e.g. faces or edges), limiting its ability to support complex editing operations such as chamfer or fillet. Further, the discretization of a continuous variable during sketch and extrude operations may result in topological errors. To address these limitations, we present Pointer-CAD, a novel LLM-based CAD generation framework that leverages a pointer-based command sequence representation to explicitly incorporate the geometric information of B-rep models into sequential modeling. In particular, Pointer-CAD decomposes CAD model generation into steps, conditioning the generation of each subsequent step on both the textual description and the B-rep generated from previous steps. Whenever an operation requires the selection of a specific geometric entity, the LLM predicts a Pointer that selects the most feature-consistent candidate from the available set. Such a selection operation also reduces the quantization error in the command sequence-based representation. To support the training of Pointer-CAD, we develop a data annotation pipeline that produces expert-level natural language descriptions and apply it to build a dataset of approximately 575K CAD models. Extensive experimental results demonstrate that Pointer-CAD effectively supports the generation of complex geometric structures and reduces segmentation error to an extremely low level, achieving a significant improvement over prior command sequence methods, thereby significantly mitigating the topological inaccuracies introduced by quantization error.
>
---
#### [replaced 004] VLM Judges Can Rank but Cannot Score: Task-Dependent Uncertainty in Multimodal Evaluation
- **分类: cs.LG; cs.CL; cs.CV; stat.ML**

- **简介: 该论文研究VLM作为评判者时的评估不确定性问题，通过置信预测方法分析不同任务下的评分可靠性，揭示任务依赖性及评分与排序的解耦现象。**

- **链接: [https://arxiv.org/pdf/2604.25235](https://arxiv.org/pdf/2604.25235)**

> **作者:** Divake Kumar; Sina Tayebati; Devashri Naik; Ranganath Krishnan; Amit Ranjan Trivedi
>
> **摘要:** Vision-language models (VLMs) are increasingly used as automated judges for multimodal systems, yet their scores provide no indication of reliability. We study this problem through conformal prediction, a distribution-free framework that converts a judge's point score into a calibrated prediction interval using only score-token log-probabilities, with no retraining. We present the first systematic analysis of conformal prediction for VLM-as-a-Judge across 3 judges and 14 visual task categories. Our results show that evaluation uncertainty is strongly task-dependent: intervals cover ~40% of the score range for aesthetics and natural images but expand to ~70% for chart and mathematical reasoning, yielding a quantitative reliability map for multimodal evaluation. We further identify a failure mode not captured by standard evaluation metrics, ranking-scoring decoupling, where judges achieve high ranking correlation while producing wide, uninformative intervals, correctly ordering responses but failing to assign reliable absolute scores. Finally, we show that interval width is driven primarily by task difficulty and annotation quality, i.e., the same judge and method yield 4.5x narrower intervals on a clean, multi-annotator captioning benchmark. Code: this https URL
>
---
#### [replaced 005] Image Compression with Bubble-Aware Frame Rate Adaptation for Energy-Efficient Video Capsule Endoscopy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.25464](https://arxiv.org/pdf/2604.25464)**

> **作者:** Oliver Bause; Jörg Gamerdinger; Julia Werner; Oliver Bringmann
>
> **备注:** 7 pages, 8 figures, EMBC2026
>
> **摘要:** Video Capsule Endoscopy (VCE) is a promising method for improving the medical examination of the small intestine in the gastrointestinal tract. A key challenge is their limited size, resulting in a short battery lifetime which conflicts with high energy consumption for image capturing and transmission to an on-body device. Thus, we propose an image compression pipeline that substantially reduces the transmitted data while preserving diagnostic image quality. Furthermore, we exploit characteristics of the compression process to identify frames with low diagnostic value mainly caused by bubbles, without requiring additional image analysis. For low-visibility frames, a dynamic bubble-aware frame rate adaptation strategy reduces image acquisition and transmission during these phases while preserving sensitivity to potential anomalies. The proposed compression and frame rate adaptation are evaluated on a RISC-V platform using the Kvasir-Capsule and Galar datasets. The compression method achieves a compression ratio of 5.748 (82.6%) at a peak signal-to-noise ratio of 40.3 dB, indicating negligible loss of visual quality. The compression accomplished a mean energy reduction of the whole system by 20.58%. Additionally, the proposed bubble-aware frame rate adaptation reduced the energy consumption by up to 40%. These results demonstrate the potential of our method to increase the applicability of VCE.
>
---
#### [replaced 006] RetroMotion: Retrocausal Motion Forecasting Models are Instructable
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于多智能体运动预测任务，解决复杂场景下轨迹预测的不确定性问题。通过分解分布并引入回溯因果信息流，提升预测准确性与指令适应能力。**

- **链接: [https://arxiv.org/pdf/2505.20414](https://arxiv.org/pdf/2505.20414)**

> **作者:** Royden Wagner; Omer Sahin Tas; Felix Hauser; Marlon Steiner; Dominik Strutz; Abhishek Vivekanandan; Jaime Villa; Yinzhe Shen; Carlos Fernandez; Christoph Stiller
>
> **备注:** CVPRW26
>
> **摘要:** Motion forecasts of road users (i.e., agents) vary in complexity depending on the number of agents, scene constraints, and interactions. In particular, the output space of joint trajectory distributions grows exponentially with the number of agents. Therefore, we decompose multi-agent motion forecasts into (1) marginal distributions for all modeled agents and (2) joint distributions for interacting agents. Using a transformer model, we generate joint distributions by re-encoding marginal distributions followed by pairwise modeling. This incorporates a retrocausal flow of information from later points in marginal trajectories to earlier points in joint trajectories. For each time step, we model the positional uncertainty using compressed exponential power distributions. Notably, our method achieves strong results in the Waymo Interaction Prediction Challenge and generalizes well to the Argoverse 2 and V2X-Seq datasets. Additionally, our method provides an interface for issuing instructions. We show that standard motion forecasting training implicitly enables the model to follow instructions and adapt them to the scene context. GitHub repository: this https URL
>
---
#### [replaced 007] GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.09547](https://arxiv.org/pdf/2508.09547)**

> **作者:** Fengyi Wu; Yifei Dong; Yilong Dai; Guangyu Chen; Qifeng Wu; Huiting Huang; Hang Wang; Qi Dai; Alexander G. Hauptmann; Zhi-Qi Cheng
>
> **备注:** Accepted to ACL 2026 Findings. 22 pages, 12 figures, Code: this https URL
>
> **摘要:** We introduce Goal-Conditioned Visual Navigation Instruction Generation (GoViG), a new task that aims to generate contextually coherent navigation instructions solely from egocentric visual observations of initial and goal states. Unlike prior work relying on structured inputs, such as semantic annotations or environmental maps, GoViG exclusively leverages raw egocentric visual data, improving adaptability to unseen and unstructured environments. Our method addresses this task by decomposing it into two interconnected subtasks: (1) navigation visualization, predicting intermediate visual states bridging the initial and goal views; and (2) instruction generation, synthesizing coherent instructions grounded in observed and anticipated visuals. Both subtasks are integrated within an autoregressive multimodal LLM trained with tailored objectives to ensure spatial accuracy and linguistic clarity. Furthermore, we introduce two multimodal reasoning strategies, one-pass and interleaved reasoning, to mimic incremental human navigation cognition. To comprehensively evaluate our method, we propose the R2R-Goal dataset, combining diverse synthetic and real-world trajectories. Empirical results demonstrate significant performance improvements over state-of-the-art methods in BLEU-4 and CIDEr scores along with robust cross-domain generalization.
>
---
#### [replaced 008] StereoSpace: Depth-Free Synthesis of Stereo Geometry via End-to-End Diffusion in a Canonical Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10959](https://arxiv.org/pdf/2512.10959)**

> **作者:** Tjark Behrens; Anton Obukhov; Bingxin Ke; Fabio Tosi; Matteo Poggi; Konrad Schindler
>
> **备注:** CVPR 2026 Findings. Project page: this https URL
>
> **摘要:** We introduce StereoSpace, a diffusion-based framework for monocular-to-stereo synthesis that models geometry purely through viewpoint conditioning, without explicit depth or warping. A canonical rectified space and the conditioning guide the generator to infer correspondences and fill disocclusions end-to-end. To ensure fair and leakage-free evaluation, we introduce an end-to-end protocol that excludes any ground truth or proxy geometry estimates at test time. The protocol emphasizes metrics reflecting downstream relevance: iSQoE for perceptual comfort and MEt3R for geometric consistency. StereoSpace surpasses other methods from the warp & inpaint, latent-warping, and warped-conditioning categories, achieving sharp parallax and strong robustness on layered and non-Lambertian scenes. This establishes viewpoint-conditioned diffusion as a scalable, depth-free solution for stereo generation.
>
---
#### [replaced 009] ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models
- **分类: cs.LG; cs.AI; cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2405.13729](https://arxiv.org/pdf/2405.13729)**

> **作者:** Rui Xu; Jiepeng Wang; Hao Pan; Yang Liu; Xin Tong; Shiqing Xin; Changhe Tu; Taku Komura; Wenping Wang
>
> **备注:** ACM Transactions on Graphics, SIGGRAPH 2026
>
> **摘要:** In this paper, we study an under-explored but important factor of diffusion generative models, i.e., the combinatorial complexity. Data samples are generally high-dimensional, and for various structured generation tasks, additional attributes are combined to associate with data samples. We show that the space spanned by the combination of dimensions and attributes can be insufficiently covered by existing training schemes of diffusion generative models, potentially limiting test time performance. We present a simple fix to this problem by constructing stochastic processes that fully exploit the combinatorial structures, hence the name ComboStoc. Using this simple strategy, we show that network training is significantly accelerated across diverse data modalities, including images and 3D structured shapes. Moreover, ComboStoc enables a new way of test time generation which uses asynchronous time steps for different dimensions and attributes, thus allowing for varying degrees of control over them. Our code is available at: this https URL
>
---
#### [replaced 010] Triple-Phase Sequential Fusion Network for Hepatobiliary Phase Liver MRI Synthesis
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.22904](https://arxiv.org/pdf/2604.22904)**

> **作者:** Qiuli Wang; Xinhuan Sun; Fengxi Chen; Yongxu Liu; Jie Cheng; Lin Chen; Jiafei Chen; Yue Zhang; Xiaoming Li; Wei Chen
>
> **备注:** 7 figures, 7 tables
>
> **摘要:** Gadoxetate disodium-enhanced MRI is essential for the detection and characterization of hepatocellular carcinoma. However, acquisition of the hepatobiliary phase (HBP) requires a prolonged post-contrast delay, which reduces workflow efficiency and increases the risk of motion artifacts. In this study, we propose a Triple-Phase Sequential Fusion Network (TriPF-Net) to synthesize HBP images by leveraging the sequential information from pre-HBP sequences: while T1-weighted imaging serves as the indispensable baseline, the model adaptively integrates arterial-phase (AP) and venous-phase (VP) features when available. By modeling the tissue-specific contrast uptake and excretion dynamics across these three phases, TriPF-Net ensures robust HBP synthesis even under the stochastic absence of one or both dynamic contrast-enhanced sequences. The framework comprises an Enhanced Region-Guided Encoder and a Dynamic Feature Unification Module, optimized with a Region-Guided Sequential Fusion Loss to maintain physiological consistency. In addition, clinical variables, including age, sex, total bilirubin, and albumin, are incorporated to enhance physiological consistency. Compared with conventional methods, TriPF-Net achieved superior performance on datasets from two centers. On the internal dataset, the model achieved an MAE of 10.65, a PSNR of 23.27, and an SSIM of 0.76. On the external validation dataset, the corresponding values were 12.41, 23.11, and 0.78, respectively. This flexible solution enhances clinical workflow and lesion depiction, potentially eliminating the need for delayed HBP acquisition in HCC imaging.
>
---
#### [replaced 011] HumanOmni-Speaker: Identifying Who said What and When
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21664](https://arxiv.org/pdf/2603.21664)**

> **作者:** Detao Bai; Shimin Yao; Weixuan Chen; Zhiheng Ma; Xihan Wei; Jingren Zhou
>
> **摘要:** While Omni-modal Large Language Models have made strides in joint sensory processing, they fundamentally struggle with a cornerstone of human interaction: deciphering complex, multi-person conversational dynamics to accurately answer ``Who said what and when.'' Current models suffer from an ``illusion of competence'' -- they exploit visual biases in conventional benchmarks to bypass genuine cross-modal alignment, while relying on sparse, low-frame-rate visual sampling that destroys crucial high-frequency dynamics like lip movements. To shatter this illusion, we introduce Visual-Registered Speaker Diarization and Recognition (VR-SDR) and the HumanOmni-Speaker Benchmark. By strictly eliminating visual shortcuts, this rigorous paradigm demands true end-to-end spatio-temporal identity binding using only natural language queries. To overcome the underlying architectural perception gap, we propose HumanOmni-Speaker, powered by a Visual Delta Encoder. By sampling raw video at 25 fps and explicitly compressing inter-frame motion residuals into just 6 tokens per frame, it captures fine-grained visemes and speaker trajectories without triggering a catastrophic token explosion. Ultimately, HumanOmni-Speaker demonstrates strong multimodal synergy, natively enabling end-to-end lip-reading and high-precision spatial localization without intrusive cropping, and achieving superior performance across a wide spectrum of speaker-centric tasks.
>
---
#### [replaced 012] SkyReels-Text: Fine-Grained Font-Controllable Text Editing for Poster Design
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13285](https://arxiv.org/pdf/2511.13285)**

> **作者:** Yunjie Yu; Jingchen Wu; Junchen Zhu; Chunze Lin; Guibin Chen
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Artistic design, particularly poster design, often demands rapid yet precise modification of textual content while preserving visual harmony and typographic intent, especially across diverse font styles. Although modern image editing models have grown increasingly powerful, they still fall short in fine-grained, font-aware text manipulation, limiting their utility in professional workflows. To address this issue, we present SkyReels-Text, a novel font-controllable framework for precise poster text editing. Our method enables simultaneous editing of multiple text regions, each rendered in distinct typographic styles, while preserving the visual appearance of non-edited regions. Notably, our model requires neither font labels nor test-time fine-tuning: users can simply provide cropped glyph patches corresponding to their desired typography - even if the font is not included in any standard library. Extensive experiments on multiple benchmarks demonstrate that SkyReels-Text achieves state-of-the-art performance in both text fidelity and visual realism, offering unprecedented control over font families and stylistic nuances. This work bridges the gap between general-purpose image editing and professional-grade typographic design. Code and models are publicly available at this https URL.
>
---
#### [replaced 013] OVGGT: O(1) Constant-Cost Streaming Visual Geometry Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05959](https://arxiv.org/pdf/2603.05959)**

> **作者:** Si-Yu Lu; Po-Ting Chen; Hui-Che Hsu; Sin-Ye Jhong; Wen-Huang Cheng; Yung-Yao Chen
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** Reconstructing 3D geometry from streaming video requires continuous inference under bounded resources. Recent geometric foundation models achieve impressive reconstruction quality through all-to-all attention, yet their quadratic cost confines them to short, offline sequences. Causal-attention variants such as StreamVGGT enable single-pass streaming but accumulate an ever-growing KV cache, exhausting GPU memory within hundreds of frames and precluding the long-horizon deployment that motivates streaming inference in the first place. We present OVGGT, a training-free framework that bounds both memory and compute to a fixed budget regardless of sequence length. Our approach combines Self-Selective Caching, which leverages FFN residual magnitudes to compress the KV cache while remaining fully compatible with FlashAttention, with Dynamic Anchor Protection, which shields coordinate-critical tokens from eviction to suppress geometric drift over extended trajectories. Extensive experiments on indoor, outdoor, and ultra-long sequence benchmarks demonstrate that OVGGT processes arbitrarily long videos within a constant VRAM envelope while achieving state-of-the-art 3D geometric accuracy. Project page: this https URL Code: this https URL
>
---
#### [replaced 014] FILTR: Extracting Topological Features from Pretrained 3D Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.22334](https://arxiv.org/pdf/2604.22334)**

> **作者:** Louis Martinez; Maks Ovsjanikov
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advances in pretraining 3D point cloud encoders (e.g., Point-BERT, Point-MAE) have produced powerful models, whose abilities are typically evaluated on geometric or semantic tasks. At the same time, topological descriptors have been shown to provide informative summaries of a shape's multiscale structure. In this paper we pose the question whether topological information can be derived from features produced by 3D encoders. To address this question, we first introduce DONUT, a synthetic benchmark with controlled topological complexity, and propose FILTR (Filtration Transformer), a learnable framework to predict persistence diagrams directly from frozen encoders. FILTR adapts a transformer decoder to treat diagram generation as a set prediction task. Our analysis on DONUT reveals that existing encoders retain only limited global topological signals, yet FILTR successfully leverages information produced by these encoders to approximate persistence diagrams. Our approach enables, for the first time, data-driven extraction of persistence diagrams from raw point clouds through an efficient learnable feed-forward mechanism.
>
---
#### [replaced 015] CommFuse: Hiding Tail Latency via Communication Decomposition and Fusion for Distributed LLM Training
- **分类: cs.LG; cs.CV; cs.DC**

- **链接: [https://arxiv.org/pdf/2604.24013](https://arxiv.org/pdf/2604.24013)**

> **作者:** Rezaul Karim; Austin Wen; Wang Zongzuo; Weiwei Zhang; Yang Liu; Walid Ahmed
>
> **备注:** Slightly modified the title, and corresponding minor wording change in the content
>
> **摘要:** The rapid growth in the size of large language models has necessitated the partitioning of computational workloads across accelerators such as GPUs, TPUs, and NPUs. However, these parallelization strategies incur substantial data communication overhead significantly hindering computational efficiency. While communication-computation overlap presents a promising direction, existing data slicing based solutions suffer from tail latency. To overcome this limitation, this research introduces a novel communication-computation overlap technique to eliminate this tail latency in state of the art overlap methods for distributed LLM training. The aim of this technique is to effectively mitigate communication bottleneck of tensor parallelism and data parallelism for distributed training and inference. In particular, we propose a novel method termed CommFuse that replaces conventional collective operations of reduce-scatter and all-gather with decomposed peer-to-peer (P2P) communication and schedules partitioned computations to enable fine-grained overlap. Our method provides an exact algorithm for reducing communication overhead that eliminates tail latency. Moreover, it presents a versatile solution compatible with data-parallel training and various tensor-level parallelism strategies, including TPSP and UP. Experimental evaluations demonstrate that our technique consistently achieves lower latency, superior Model FLOPS Utilization (MFU), and high throughput.
>
---
#### [replaced 016] ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ReGATE方法，用于加速多模态大语言模型训练。针对训练中计算成本高的问题，通过自适应剪枝减少token使用，提升效率。**

- **链接: [https://arxiv.org/pdf/2507.21420](https://arxiv.org/pdf/2507.21420)**

> **作者:** Chaoyu Li; Yogesh Kulkarni; Pooyan Fazli
>
> **备注:** ACL 2026. Project page: this https URL
>
> **摘要:** The computational cost of training multimodal large language models (MLLMs) grows rapidly with the number of processed tokens. Existing efficiency methods mainly target inference via token reduction or merging, offering limited benefits during training. We introduce ReGATE (Reference-Guided Adaptive Token Elision), an adaptive token pruning method for accelerating MLLM training. ReGATE adopts a teacher-student framework, in which a frozen teacher LLM provides per-token guidance losses that are fused with an exponential moving average of the student's difficulty estimates. This adaptive scoring mechanism dynamically selects informative tokens while skipping redundant ones in the forward pass, substantially reducing computation without altering the model architecture. Across three representative MLLMs, ReGATE matches the peak accuracy of standard training on MVBench up to 2$\times$ faster, using only 38% of the tokens. With extended training, it even surpasses the baseline across multiple multimodal benchmarks, cutting total token usage by over 41%.
>
---
#### [replaced 017] Towards Redundancy Reduction in Diffusion Models for Efficient Video Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23980](https://arxiv.org/pdf/2509.23980)**

> **作者:** Jinpei Guo; Yifei Ji; Shengwei Wang; Zheng Chen; Yufei Wang; Sizhuo Ma; Yong Guo; Baiang Li; Jusheng Zhang; Yulun Zhang; Jian Wang
>
> **摘要:** Diffusion models have recently shown promising results for video super-resolution (VSR). However, directly adapting generative diffusion models to VSR can result in redundancy, since low-quality videos already preserve substantial content information. Such redundancy leads to increased computational overhead and learning burden, as the model performs superfluous operations and must learn to filter out irrelevant information. To address this problem, we propose OASIS, an efficient $\textbf{o}$ne-step diffusion model with $\textbf{a}$ttention $\textbf{s}$pecialization for real-world v$\textbf{i}$deo $\textbf{s}$uper-resolution. OASIS incorporates an attention specialization routing that assigns attention heads to different patterns according to their intrinsic behaviors. This routing mitigates redundancy while effectively preserving pretrained knowledge, allowing diffusion models to better adapt to VSR and achieve stronger performance. Moreover, we propose a simple yet effective progressive training strategy, which starts with temporally consistent degradations and then shifts to inconsistent settings. This strategy facilitates learning under complex degradations. Extensive experiments demonstrate that OASIS achieves state-of-the-art performance on both synthetic and real-world datasets. OASIS also provides superior inference speed, offering a $\textbf{6.2$\times$}$ speedup over one-step diffusion baselines such as SeedVR2. The code will be available at \href{this https URL}{this https URL}.
>
---
#### [replaced 018] ViPO: Visual Preference Optimization at Scale
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.24953](https://arxiv.org/pdf/2604.24953)**

> **作者:** Ming Li; Jie Wu; Justin Cui; Xiaojie Li; Rui Wang; Chen Chen
>
> **备注:** ICLR 2026 Paper. Project Page: this https URL ; Code: this https URL
>
> **摘要:** While preference optimization is crucial for improving visual generative models, how to effectively scale this paradigm remains largely unexplored. Current open-source preference datasets contain conflicting preference patterns, where winners excel in some dimensions but underperform in others. Naively optimizing on such noisy datasets fails to learn preferences, hindering effective scaling. To enhance robustness against noise, we propose Poly-DPO, which extends the DPO objective with an additional polynomial term that dynamically adjusts model confidence based on dataset characteristics, enabling effective learning across diverse data distributions. Beyond biased patterns, existing datasets suffer from low resolution, limited prompt diversity, and imbalanced distributions. To facilitate large-scale visual preference optimization by tackling data bottlenecks, we construct ViPO, a massive-scale preference dataset with 1M image pairs at 1024px across five categories and 300K video pairs at 720p+ across three categories. State-of-the-art generative models and diverse prompts ensure reliable preference signals with balanced distributions. Remarkably, when applying Poly-DPO to our high-quality dataset, the optimal configuration converges to standard DPO. This convergence validates dataset quality and Poly-DPO's adaptive nature: sophisticated optimization becomes unnecessary with sufficient data quality, yet remains valuable for imperfect datasets. We validate our approach across visual generation models. On noisy datasets like Pick-a-Pic V2, Poly-DPO achieves 6.87 and 2.32 gains over Diffusion-DPO on GenEval for SD1.5 and SDXL, respectively. For ViPO, models achieve performance far exceeding those trained on existing open-source preference datasets. These results confirm that addressing both algorithmic adaptability and data quality is essential for scaling visual preference optimization.
>
---
#### [replaced 019] COMMA: Coordinate-aware Modulated Mamba Network for 3D Dispersed Vessel Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.02332](https://arxiv.org/pdf/2503.02332)**

> **作者:** Gen Shi; Hui Zhang; Jie Tian
>
> **备注:** Accepted by IEEE TIP
>
> **摘要:** Accurate segmentation of 3D vascular structures is essential for various medical imaging applications. The dispersed nature of vascular structures leads to inherent spatial uncertainty and necessitates location awareness, yet most current 3D medical segmentation models rely on the patch-wise training strategy that usually loses this spatial context. In this study, we introduce the Coordinate-aware Modulated Mamba Network (COMMA) and contribute a manually labeled dataset of 570 cases, the largest publicly available 3D vessel dataset to date. COMMA leverages both entire and cropped patch data through global and local branches, ensuring robust and efficient spatial location awareness. Specifically, COMMA employs a channel-compressed Mamba (ccMamba) block to encode entire image data, capturing long-range dependencies while optimizing computational costs. Additionally, we propose a coordinate-aware modulated (CaM) block to enhance interactions between the global and local branches, allowing the local branch to better perceive spatial information. We evaluate COMMA on six datasets, covering two imaging modalities and five types of vascular tissues. The results demonstrate COMMA's superior performance compared to state-of-the-art methods with computational efficiency, especially in segmenting small vessels. Ablation studies further highlight the importance of our proposed modules and spatial information. The code and data will be open source at this https URL.
>
---
#### [replaced 020] Assessing the Utility of Volumetric Motion Fields for Radar-based Precipitation Nowcasting with Physics-informed Deep Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.13589](https://arxiv.org/pdf/2603.13589)**

> **作者:** Peter Pavlík; Anna Bou Ezzeddine; Viera Rozinajová
>
> **备注:** To be submitted to a fitting journal
>
> **摘要:** Estimating motion from spatiotemporal geoscientific data is a fundamental component of many environmental modeling and forecasting tasks. In this work, we propose a physics-informed deep learning framework for estimating altitude-wise motion fields directly from volumetric radar reflectivity data. The model utilizes a fully differentiable semi-Lagrangian extrapolation operator to process three-dimensional inputs as independent horizontal slice sequences, enabling efficient inference of horizontal motion across multiple altitude levels. Using a multi-year radar dataset from Central Europe, we evaluate the impact of altitude-wise motion estimation on extrapolation-based precipitation forecasting and conduct a systematic dataset-scale analysis of inter-altitude motion consistency. The results show that the estimated motion fields exhibit strong vertical coherence, with high correlation across altitude levels, which results in limited improvement over traditional two-dimensional approach in this setting. The proposed framework provides a general tool for efficiently analyzing motion structure in volumetric geospatial data. The findings indicate that, in regions dominated by vertically coherent precipitation systems, the added complexity of volumetric motion modeling may offer limited benefit, warranting careful consideration in the design of efficient spatiotemporal advection models.
>
---
#### [replaced 021] Rethinking Cross-Dose PET Denoising: Mitigating Averaging Effects via Residual Noise Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.16925](https://arxiv.org/pdf/2604.16925)**

> **作者:** Yichao Liu; Zongru Shao; Yueyang Teng; Junwen Guo
>
> **摘要:** Cross-dose denoising for low-dose positron emission tomography (LDPET) has been proposed to address the limited generalization of models trained at a single noise level. In practice, neural networks trained on a specific dose level often fail to generalize to other dose conditions due to variations in noise magnitude and statistical properties. Conventional "one-size-for-all" models attempt to handle this variability but tend to learn averaged representations across noise levels, resulting in degraded performance. In this work, we analyze this limitation and show that standard training formulations implicitly optimize an expectation over heterogeneous noise distributions. To this end, we propose a unified residual noise learning framework that estimates noise directly from low-dose PET images rather than predicting full-dose images. Experiments on large-scale multi-dose PET datasets from two medical centers demonstrate that the proposed method outperforms the "one-size-for-all" model, individual dose-specific U-Net models, and dose-conditioned approaches, achieving improved denoising performance. These results indicate that residual noise learning effectively mitigates the averaging effect and enhances generalization for cross-dose PET denoising.
>
---
#### [replaced 022] A Multimodal Depth-Aware Method For Embodied Reference Understanding
- **分类: cs.CV; cs.HC; cs.RO**

- **简介: 该论文属于Embodied Reference Understanding任务，旨在解决多目标场景下的指代消歧问题。通过结合语言模型、深度图和深度感知模块，提升目标检测的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.08278](https://arxiv.org/pdf/2510.08278)**

> **作者:** Fevziye Irem Eyiokur; Dogucan Yaman; Hazım Kemal Ekenel; Alexander Waibel
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Embodied Reference Understanding requires identifying a target object in a visual scene based on both language instructions and pointing cues. While prior works have shown progress in open-vocabulary object detection, they often fail in ambiguous scenarios where multiple candidate objects exist in the scene. To address these challenges, we propose a novel ERU framework that jointly leverages LLM-based data augmentation, depth-map modality, and a depth-aware decision module. This design enables robust integration of linguistic and embodied cues, improving disambiguation in complex or cluttered environments. Experimental results on two datasets demonstrate that our approach significantly outperforms existing baselines, achieving more accurate and reliable referent detection.
>
---
#### [replaced 023] Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦于多模态关系知识推理任务，解决当前模型在抽象信息理解上的不足。提出自动数据生成引擎和增强训练框架，提升模型在STAR任务中的表现。**

- **链接: [https://arxiv.org/pdf/2510.21828](https://arxiv.org/pdf/2510.21828)**

> **作者:** Yichi Zhang; Zhuo Chen; Lingbing Guo; Wen Zhang; Huajun Chen
>
> **备注:** Accepted by Findings of ACL 2026
>
> **摘要:** Understanding and reasoning with abstractive information from the visual modality presents significant challenges for current multi-modal large language models (MLLMs). Among the various forms of abstractive information, Multi-Modal Relational Knowledge (MMRK), which represents abstract relational structures between multi-modal entities using node-edge formats, remains largely under-explored. In particular, STructured and Abstractive Reasoning (STAR) on such data has received little attention from the research community. To bridge the dual gaps in large-scale high-quality data and capability enhancement methodologies, this paper makes the following key contributions: (i). An automatic STAR data engine capable of synthesizing images with MMRK to build multi-modal instruction data with reliable chain-of-thought thinking for various STAR tasks and (ii). A comprehsive two-stage capability enhancement training framework, accompanied by a suite of evaluation protocols tailored to different STAR tasks. Based upon these contributions, we introduce STAR-64K, a dataset comprising 64K high-quality multi-modal instruction samples, and conduct experiments across 5 open-source MLLMs. Experimental results show that our two-stage enhancement framework enables smaller 3B/7B models to significantly outperform GPT-4o in STAR. Additionally, we provide in-depth analysis regarding the effectiveness of various designs, data transferability, and scalability.
>
---
#### [replaced 024] COP-GEN: Latent Diffusion Transformer for Copernicus Earth Observation Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03239](https://arxiv.org/pdf/2603.03239)**

> **作者:** Miguel Espinosa; Eva Gmelich Meijling; Valerio Marsocci; Elliot J. Crowley; Mikolaj Czerkawski
>
> **摘要:** Earth observation applications increasingly rely on data from multiple sensors, including optical, radar, elevation, and land-cover. Relationships between modalities are fundamental for data integration but are inherently non-injective: identical conditioning information can correspond to multiple physically plausible observations, and should be parametrised as conditional distributions. Deterministic models, by contrast, collapse toward conditional means and fail to represent the uncertainty and variability required for tasks such as data completion and cross-sensor translation. We introduce COP-GEN, a multimodal latent diffusion transformer that models the joint distribution of heterogeneous EO modalities at their native spatial resolutions. By parameterising cross-modal mappings as conditional distributions, COP-GEN enables flexible any-to-any conditional generation, including zero-shot modality translation without task-specific retraining. Experiments show that COP-GEN generates diverse yet physically consistent realisations while maintaining strong peak fidelity across optical, radar, and elevation modalities. Qualitative and quantitative analyses demonstrate that the model captures meaningful cross-modal structure and adapts its output uncertainty as conditioning information increases. We release a stochastic benchmark built from multi-temporal Sentinel-2 observations that enables distribution-level comparison of generative EO models. On this benchmark, COP-GEN covers 90% of the real observation manifold and 63% of its per-band reflectance range, while the strongest competing method collapses to 2.8% and 18%, respectively. These results highlight the importance of stochastic generative modeling for EO and motivate evaluation protocols beyond single-reference, pointwise metrics. Website: this https URL
>
---
#### [replaced 025] Benchmarking Deep Learning and Vision Foundation Models for Atypical vs. Normal Mitosis Classification with Cross-Dataset Evaluation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21444](https://arxiv.org/pdf/2506.21444)**

> **作者:** Sweta Banerjee; Viktoria Weiss; Taryn A. Donovan; Rutger H.J. Fick; Thomas Conrad; Jonas Ammeling; Nils Porsche; Robert Klopfleisch; Christopher Kaltenecker; Katharina Breininger; Marc Aubreville; Christof A. Bertram
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL
>
> **摘要:** Atypical mitosis marks a deviation in the cell division process that has been shown be an independent prognostic marker for tumor malignancy. However, atypical mitosis classification remains challenging due to low prevalence, at times subtle morphological differences from normal mitotic figures, low inter-rater agreement among pathologists, and class imbalance in datasets. Building on the Atypical Mitosis dataset for Breast Cancer (AMi-Br), this study presents a comprehensive benchmark comparing deep learning approaches for automated atypical mitotic figure (AMF) classification, including end-to-end trained deep learning models, foundation models with linear probing, and foundation models fine-tuned with low-rank adaptation (LoRA). For rigorous evaluation, we further introduce two new held-out AMF datasets - AtNorM-Br, a dataset of mitotic figures from the TCGA breast cancer cohort, and AtNorM-MD, a multi-domain dataset of mitotic figures from a subset of the MIDOG++ training set. We found average balanced accuracy values of up to 0.8135, 0.7788, and 0.7723 on the in-domain AMi-Br and the out-of-domain AtNorm-Br and AtNorM-MD datasets, respectively. Our work shows that atypical mitotic figure classification, while being a challenging problem, can be effectively addressed through the use of recent advances in transfer learning and model fine-tuning techniques. We make all code and data used in this paper available in this github repository: this https URL.
>
---
#### [replaced 026] Time Blindness: Why Video-Language Models Can't See What Humans Can?
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.24867](https://arxiv.org/pdf/2505.24867)**

> **作者:** Ujjwal Upadhyay; Mukul Ranjan; Zhiqiang Shen; Mohamed Elhoseiny
>
> **备注:** Accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026 Project page at this https URL
>
> **摘要:** Recent advances in vision-language models (VLMs) have made impressive strides in understanding spatio-temporal relationships in videos. However, when spatial information is obscured, these models struggle to capture purely temporal patterns. We introduce $\textbf{SpookyBench}$, a benchmark where information is encoded solely in temporal sequences of noise-like frames, mirroring natural phenomena from biological signaling to covert communication. Interestingly, while humans can recognize shapes, text, and patterns in these sequences with over 98% accuracy, state-of-the-art VLMs achieve 0% accuracy. This performance gap highlights a critical limitation: an over-reliance on frame-level spatial features and an inability to extract meaning from temporal cues. Furthermore, when trained in data sets with low spatial signal-to-noise ratios (SNR), temporal understanding of models degrades more rapidly than human perception, especially in tasks requiring fine-grained temporal reasoning. Overcoming this limitation will require novel architectures or training paradigms that decouple spatial dependencies from temporal processing. Our systematic analysis shows that this issue persists across model scales and architectures. We release SpookyBench to catalyze research in temporal pattern recognition and bridge the gap between human and machine video understanding. Dataset and code has been made available on our project website: this https URL.
>
---
#### [replaced 027] Graph Propagated Projection Unlearning: A Unified Framework for Vision and Audio Discriminative Models
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出GPPU，解决深度学习模型中类级信息擦除问题，通过图传播和投影实现高效、不可逆的模型遗忘，适用于视觉和音频任务。**

- **链接: [https://arxiv.org/pdf/2604.13127](https://arxiv.org/pdf/2604.13127)**

> **作者:** Shreyansh Pathak; Jyotishman Das
>
> **备注:** This submission has been withdrawn because it is posted accidentally without full author approval. A revised version may be submitted with full approval anytime soon
>
> **摘要:** The need to selectively and efficiently erase learned information from deep neural networks is becoming increasingly important for privacy, regulatory compliance, and adaptive system design. We introduce Graph-Propagated Projection Unlearning (GPPU), a unified and scalable algorithm for class-level unlearning that operates across both vision and audio models. GPPU employs graph-based propagation to identify class-specific directions in the feature space and projects representations onto the orthogonal subspace, followed by targeted fine-tuning, to ensure that target class information is effectively and irreversibly removed. Through comprehensive evaluations on six vision datasets and two large-scale audio benchmarks spanning a variety of architectures including CNNs, Vision Transformers, and Audio Transformers, we demonstrate that GPPU achieves highly efficient unlearning, realizing 10-20x speedups over prior methodologies while preserving model utility on retained classes. Our framework provides a principled and modality-agnostic approach to machine unlearning, evaluated at a scale that has received limited attention in prior work, contributing toward more efficient and responsible deep learning.
>
---
#### [replaced 028] GNC-Pose: Geometry-Aware GNC-PnP for Accurate 6D Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.06565](https://arxiv.org/pdf/2512.06565)**

> **作者:** Xiujin Liu
>
> **备注:** 1 figures, 2 tables, 14pages
>
> **摘要:** We present GNC-Pose, a fully learning-free monocular 6D object pose estimation pipeline for textured objects that combines rendering-based initialization, geometry-aware correspondence weighting, and robust GNC optimization. Starting from coarse 2D-3D correspondences obtained through feature matching and rendering-based alignment, our method builds upon the Graduated Non-Convexity (GNC) principle and introduces a geometry-aware, cluster-based weighting mechanism that assigns robust per point confidence based on the 3D structural consistency of the model. This geometric prior and weighting strategy significantly stabilizes the optimization under severe outlier contamination. A final LM refinement further improve accuracy. We tested GNC-Pose on The YCB Object and Model Set, despite requiring no learned features, training data, or category-specific priors, GNC-Pose achieves competitive accuracy compared with both learning-based and learning-free methods, and offers a simple, robust, and practical solution for learning-free 6D pose estimation.
>
---
#### [replaced 029] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
- **分类: cs.CL; cs.AI; cs.CR; cs.CV; cs.SE**

- **简介: 该论文属于安全研究任务，旨在分析CUA的安全威胁。通过文献综述，定义CUA、分类威胁、提出防御策略，并总结评估方法，为安全设计提供指导。**

- **链接: [https://arxiv.org/pdf/2505.10924](https://arxiv.org/pdf/2505.10924)**

> **作者:** Ada Chen; Yongjiang Wu; Junyuan Zhang; Jingyu Xiao; Shu Yang; Jen-tse Huang; Kun Wang; Wenxuan Wang; Shuai Wang
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Recently, AI-driven interactions with computing devices have advanced from basic prototype tools to sophisticated, LLM-based systems that emulate human-like operations in graphical user interfaces. We are now witnessing the emergence of \emph{Computer-Using Agents} (CUAs), capable of autonomously performing tasks such as navigating desktop applications, web pages, and mobile apps. However, as these agents grow in capability, they also introduce novel safety and security risks. Vulnerabilities in LLM-driven reasoning, with the added complexity of integrating multiple software components and multimodal inputs, further complicate the security landscape. In this paper, we present a systematization of knowledge on the safety and security threats of CUAs. We conduct a comprehensive literature review and distill our findings along four research objectives: \textit{\textbf{(i)}} define the CUA that suits safety analysis; \textit{\textbf{(ii)} } categorize current safety threats among CUAs; \textit{\textbf{(iii)}} propose a comprehensive taxonomy of existing defensive strategies; \textit{\textbf{(iv)}} summarize prevailing benchmarks, datasets, and evaluation metrics used to assess the safety and performance of CUAs. Building on these insights, our work provides future researchers with a structured foundation for exploring unexplored vulnerabilities and offers practitioners actionable guidance in designing and deploying secure Computer-Using Agents.
>
---
#### [replaced 030] ViTaPEs: Visuotactile Position Encodings for Cross-Modal Alignment in Multimodal Transformers
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出ViTaPEs，解决多模态对齐问题，通过双阶段位置编码提升视觉与触觉信息融合效果，增强模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.20032](https://arxiv.org/pdf/2505.20032)**

> **作者:** Fotios Lygerakis; Ozan Özdenizci; Elmar Rückert
>
> **摘要:** Tactile sensing provides local essential information that is complementary to visual perception, such as texture, compliance, and force. Despite recent advances in visuotactile representation learning, challenges remain in fusing these modalities and generalizing across tasks and environments without heavy reliance on pre-trained vision-language models. Moreover, existing methods do not study positional encodings, thereby overlooking the multi-stage spatial reasoning needed to capture fine-grained visuotactile correlations. We introduce ViTaPEs, a transformer-based architecture for learning task-agnostic visuotactile representations from paired vision and tactile inputs. Our key idea is a two-stage positional injection: local (modality-specific) positional encodings are added within each stream, and a global positional encoding is added on the joint token sequence immediately before attention, providing a shared positional vocabulary at the stage where cross-modal interaction occurs. We make the positional injection points explicit and conduct controlled ablations that isolate their effect before a token-wise nonlinearity versus immediately before self-attention. Experiments on multiple large-scale real-world datasets show that ViTaPEs not only surpasses state-of-the-art baselines across various recognition tasks but also demonstrates zero-shot generalization to unseen, out-of-domain scenarios. We further demonstrate the transfer-learning strength of \emph{ViTaPEs} in a robotic grasping task, where it outperforms state-of-the-art baselines in predicting grasp success. Project page: this https URL
>
---
#### [replaced 031] PASR: Pose-Aware 3D Shape Retrieval from Occluded Single Views
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.22658](https://arxiv.org/pdf/2604.22658)**

> **作者:** Jiaxin Shi; Guofeng Zhang; Wufei Ma; Naifu Liang; Adam Kortylewski; Alan Yuille
>
> **摘要:** Single-view 3D shape retrieval is a fundamental yet challenging task that is increasingly important with the growth of available 3D data. Existing approaches largely fall into two categories: those using contrastive learning to map point cloud features into existing vision-language spaces and those that learn a common embedding space for 2D images and 3D shapes. However, these feed-forward, holistic alignments are often difficult to interpret, which in turn limits their robustness and generalization to real-world applications. To address this problem, we propose Pose-Aware 3D Shape Retrieval (PASR), a framework that formulates retrieval as a feature-level analysis-by-synthesis problem by distilling knowledge from a 2D foundation model (DINOv3) into a 3D encoder. By aligning pose-conditioned 3D projections with 2D feature maps, our method bridges the gap between real-world images and synthetic meshes. During inference, PASR performs a test-time optimization via analysis-by-synthesis, jointly searching for the shape and pose that best reconstruct the patch-level feature map of the input image. This synthesis-based optimization is inherently robust to partial occlusion and sensitive to fine-grained geometric details. PASR substantially outperforms existing methods on both clean and occluded 3D shape retrieval datasets by a wide margin. Additionally, PASR demonstrates strong multi-task capabilities, achieving robust shape retrieval, competitive pose estimation, and accurate category classification within a single framework.
>
---
#### [replaced 032] Causal Disentanglement for Full-Reference Image Quality Assessment
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.21654](https://arxiv.org/pdf/2604.21654)**

> **作者:** Zhen Zhang; Jielei Chu; Tian Zhang; Fengmao Lv; Tianrui Li
>
> **摘要:** Existing deep network-based full-reference image quality assessment (FR-IQA) models typically work by performing pairwise comparisons of deep features from the reference and distorted images. In this paper, we approach this problem from a different perspective and propose a novel FR-IQA paradigm based on causal inference and decoupled representation learning. Unlike typical feature comparison-based FR-IQA models, our approach formulates degradation estimation as a causal disentanglement process guided by intervention on latent representations. We first decouple degradation and content representations by exploiting the content invariance between the reference and distorted images. Second, inspired by the human visual masking effect, we design a masking module to model the causal relationship between image content and degradation features, thereby extracting content-influenced degradation features from distorted images. Finally, quality scores are predicted from these degradation features using either supervised regression or label-free dimensionality reduction. Extensive experiments demonstrate that our method achieves highly competitive performance on standard IQA benchmarks across fully supervised, few-label, and label-free settings. Furthermore, we evaluate the approach on diverse non-standard natural image domains with scarce data, including underwater, radiographic, medical, neutron, and screen-content images. Benefiting from its ability to perform scenario-specific training and prediction without labeled IQA data, our method exhibits superior cross-domain generalization compared to existing training-free FR-IQA models.
>
---
#### [replaced 033] The Surprising Effectiveness of Canonical Knowledge Distillation for Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.25530](https://arxiv.org/pdf/2604.25530)**

> **作者:** Muhammad Ali; Kevin Alexander Laube; Madan Ravi Ganesh; Lukas Schott; Niclas Popp; Thomas Brox
>
> **备注:** Presented at Efficient Computer Vision (ECV) Workshop, CVPR 2026. 5 pages, 3 figures
>
> **摘要:** Recent knowledge distillation (KD) methods for semantic segmentation introduce increasingly complex hand-crafted objectives, yet are typically evaluated under fixed iteration schedules. These objectives substantially increase per-iteration cost, meaning equal iteration counts do not correspond to equal training budgets. It is therefore unclear whether reported gains reflect stronger distillation signals or simply greater compute. We show that iteration-based comparisons are misleading: when wall-clock compute is matched, canonical logit- and feature-based KD outperform recent segmentation-specific methods. Under extended training, feature-based distillation achieves state-of-the-art ResNet-18 performance on Cityscapes and ADE20K. A PSPNet ResNet-18 student closely approaches its ResNet-101 teacher despite using only one quarter of the parameters, reaching 99% of the teacher's mIoU on Cityscapes (79.0 vs 79.8) and 92% on ADE20K. Our results challenge the prevailing assumption that KD for segmentation requires task-specific mechanisms and suggest that scaling, rather than complex hand-crafted objectives, should guide future method design.
>
---
#### [replaced 034] Contrastive Semantic Projection: Faithful Neuron Labeling with Contrastive Examples
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.22477](https://arxiv.org/pdf/2604.22477)**

> **作者:** Oussama Bouanani; Jim Berend; Wojciech Samek; Sebastian Lapuschkin; Maximilian Dreyer
>
> **摘要:** Neuron labeling assigns textual descriptions to internal units of deep networks. Existing approaches typically rely on highly activating examples, often yielding broad or misleading labels by focusing on dominant but incidental visual factors. Prior work such as FALCON introduced contrastive examples -- inputs that are semantically similar to activating examples but elicit low activations -- to sharpen explanations, but it primarily addresses subspace-level interpretability rather than scalable neuron-level labeling. We revisit contrastive explanations for neuron-level labeling in two stages: (1) candidate label generation with vision language models (VLMs) and (2) label assignment with CLIP-like encoders. First, we show that providing contrastive image sets to VLMs yields candidate labels that are more specific and more faithful. Second, we introduce Contrastive Semantic Projection (CSP), an extension of SemanticLens that incorporates contrastive examples directly into its CLIP-based scoring and selection pipeline. Across extensive experiments and a case study on melanoma detection, contrastive labeling improves both faithfulness and semantic granularity over state-of-the-art baselines. Our results demonstrate that contrastive examples are a simple yet powerful and currently underutilized component of neuron labeling and analysis pipelines.
>
---
#### [replaced 035] TopoMamba: Topology-Aware Scanning and Fusion for Segmenting Heterogeneous Medical Visual Media
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.25545](https://arxiv.org/pdf/2604.25545)**

> **作者:** Fuchen Zheng; Chengpei Xu; Long Ma; Weixuan Li; Junhua Zhou; Xuhang Chen; Weihuang Liu; Haolun Li; Quanjun Li; Zhenxi Zhang; Lei Zhao; Chi-Man Pun; Shoujun Zhou
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Visual state-space models (SSMs) have shown strong potential for medical image segmentation, yet their effectiveness is often limited by two practical issues: axis-biased scan ordering weakens the modeling of oblique and curved structures, and naive multi-branch fusion tends to amplify redundant responses. We present TopoMamba, a topology-aware scan-and-fuse framework for segmenting heterogeneous medical visual media. The method combines a diagonal/anti-diagonal TopoA-Scan branch with the standard Cross-Scan branch to provide complementary structural priors, and introduces ScanCache, a device-aware caching mechanism that amortizes explicit scan-index construction across recurring resolutions. To fuse heterogeneous scan features efficiently, we further propose a lightweight HSIC Gate that regulates branch interaction using a dependence-aware scalar gating rule. We also instantiate a volumetric TopoMamba-3D for practical 3D clinical segmentation. Experiments on Synapse CT, ISIC 2017 dermoscopy, and CVC-ClinicDB endoscopy show that TopoMamba consistently improves segmentation quality over strong CNN, Transformer, and SSM baselines, with particularly clear gains on thin or curved targets such as the pancreas and gallbladder, while maintaining favorable deployment efficiency under dynamic input resolutions. These results suggest that topology-aware scan ordering and lightweight dependence-aware fusion form an effective and practical design for medical multimedia segmentation. The code will be made publicly available.
>
---
#### [replaced 036] Action Hints: Semantic Typicality and Context Uniqueness for Generalizable Skeleton-based Video Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.11058](https://arxiv.org/pdf/2509.11058)**

> **作者:** Canhui Tang; Sanping Zhou; Haoyue Shi; Le Wang
>
> **备注:** The paper has been accepted by Pattern Recognition (PR)
>
> **摘要:** Zero-Shot Video Anomaly Detection (ZS-VAD) requires temporally localizing anomalies without target domain training data, which is a crucial task due to various practical concerns, e.g., data privacy or new surveillance deployments. Skeleton-based approach has inherent generalizable advantages in achieving ZS-VAD as it eliminates domain disparities both in background and human appearance. However, existing methods only learn low-level skeleton representation and rely on the domain-limited normality boundary, which cannot generalize well to new scenes with different normal and abnormal behavior patterns. In this paper, we propose a novel zero-shot video anomaly detection framework, unlocking the potential of skeleton data via action typicality and uniqueness learning. Firstly, we introduce a language-guided semantic typicality modeling module that projects skeleton snippets into action semantic space and distills LLM's knowledge of typical normal and abnormal behaviors during training. Secondly, we propose a test-time context uniqueness analysis module to finely analyze the spatio-temporal differences between skeleton snippets and then derive scene-adaptive boundaries. Without using any training samples from the target domain, our method achieves state-of-the-art results against skeleton-based methods on four large-scale VAD datasets: ShanghaiTech, UBnormal, NWPU, and UCF-Crime, featuring over 100 unseen surveillance scenes.
>
---
#### [replaced 037] NTIRE 2026 3D Restoration and Reconstruction in Real-world Adverse Conditions: RealX3D Challenge Results
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04135](https://arxiv.org/pdf/2604.04135)**

> **作者:** Shuhong Liu; Chenyu Bao; Ziteng Cui; Xuangeng Chu; Bin Ren; Lin Gu; Xiang Chen; Mingrui Li; Long Ma; Marcos V. Conde; Radu Timofte; Yun Liu; Ryo Umagami; Tomohiro Hashimoto; Zijian Hu; Yuan Gan; Tianhan Xu; Yusuke Kurose; Tatsuya Harada; Junwei Yuan; Gengjia Chang; Xining Ge; Mache You; Qida Cao; Zeliang Li; Xinyuan Hu; Hongde Gu; Changyue Shi; Jiajun Ding; Zhou Yu; Jun Yu; Seungsang Oh; Fei Wang; Donggun Kim; Zhiliang Wu; Seho Ahn; Xinye Zheng; Kun Li; Yanyan Wei; Weisi Lin; Dizhe Zhang; Yuchao Chen; Meixi Song; Hanqing Wang; Haoran Feng; Lu Qi; Jiaao Shan; Yang Gu; Jiacheng Liu; Shiyu Liu; Kui Jiang; Junjun Jiang; Runyu Zhu; Sixun Dong; Qingxia Ye; Zhiqiang Zhang; Zhihua Xu; Zhiwei Wang; Phan The Son; Zhimiao Shi; Zixuan Guo; Xueming Fu; Lixia Han; Changhe Liu; Zhenyu Zhao; Manabu Tsukada; Zheng Zhang; Zihan Zhai; Tingting Li; Ziyang Zheng; Yuhao Liu; Dingju Wang; Jeongbin You; Younghyuk Kim; Il-Youp Kwak; Mingzhe Lyu; Junbo Yang; Wenhan Yang; Hongsen Zhang; Jinqiang Cui; Hong Zhang; Haojie Guo; Hantang Li; Qiang Zhu; Bowen He; Xiandong Meng; Debin Zhao; Xiaopeng Fan; Wei Zhou; Linzhe Jiang; Linfeng Li; Louzhe Xu; Qi Xu; Hang Song; Chenkun Guo; Weizhi Nie; Yufei Li; Xingan Zhan; Zhanqi Shi; Dufeng Zhang
>
> **摘要:** This paper presents a comprehensive review of the NTIRE 2026 3D Restoration and Reconstruction (3DRR) Challenge, detailing the proposed methods and results. The challenge seeks to identify robust reconstruction pipelines that are robust under real-world adverse conditions, specifically extreme low-light and smoke-degraded environments, as captured by our RealX3D benchmark. A total of 279 participants registered for the competition, of whom 33 teams submitted valid results. We thoroughly evaluate the submitted approaches against state-of-the-art baselines, revealing significant progress in 3D reconstruction under adverse conditions. Our analysis highlights shared design principles among top-performing methods and provides insights into effective strategies for handling 3D scene degradation.
>
---
#### [replaced 038] Perception Test 2025: Challenge Summary and a Unified VQA Extension
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.06287](https://arxiv.org/pdf/2601.06287)**

> **作者:** Joseph Heyward; Nikhil Parthasarathy; Tyler Zhu; Aravindh Mahendran; João Carreira; Dima Damen; Andrew Zisserman; Viorica Pătrăucean
>
> **摘要:** The Third Perception Test challenge was organised as a full-day workshop alongside the IEEE/CVF International Conference on Computer Vision (ICCV) 2025. Its primary goal is to benchmark state-of-the-art video models and measure the progress in multimodal perception. This year, the workshop featured 2 guest tracks as well: KiVA (an image understanding challenge) and Physic-IQ (a video generation challenge). In this report, we summarise the results from the main Perception Test challenge, detailing both the existing tasks as well as novel additions to the benchmark. In this iteration, we placed an emphasis on task unification, as this poses a more challenging test for current SOTA multimodal models. The challenge included five consolidated tracks: unified video QA, unified object and point tracking, unified action and sound localisation, grounded video QA, and hour-long video QA, alongside an analysis and interpretability track that is still open for submissions. Notably, the unified video QA track introduced a novel subset that reformulates traditional perception tasks (such as point tracking and temporal action localisation) as multiple-choice video QA questions that video-language models can natively tackle. The unified object and point tracking merged the original object tracking and point tracking tasks, whereas the unified action and sound localisation merged the original temporal action localisation and temporal sound localisation tracks. Accordingly, we required competitors to use unified approaches rather than engineered pipelines with task-specific models. By proposing such a unified challenge, Perception Test 2025 highlights the significant difficulties existing models face when tackling diverse perception tasks through unified interfaces.
>
---
#### [replaced 039] SciMDR: Advancing Scientific Multimodal Document Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出SciMDR，解决科学多模态文档推理数据集构建中的规模、真实性和准确性难题，通过合成与再定位框架生成高质量QA对，提升模型在科学任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.12249](https://arxiv.org/pdf/2603.12249)**

> **作者:** Ziyu Chen; Yilun Zhao; Chengye Wang; Rilyn Han; Manasi Patwardhan; Arman Cohan
>
> **备注:** ACL 2026
>
> **摘要:** Constructing scientific multimodal document reasoning datasets for foundation model training involves an inherent trade-off among scale, faithfulness, and realism. To address this challenge, we introduce the synthesize-and-reground framework, a two-stage pipeline comprising: (1) Claim-Centric QA Synthesis, which generates faithful, isolated QA pairs and reasoning on focused segments, and (2) Document-Scale Regrounding, which programmatically re-embeds these pairs into full-document tasks to ensure realistic complexity. Using this framework, we construct SciMDR, a large-scale training dataset for cross-modal comprehension, comprising 300K QA pairs with explicit reasoning chains across 20K scientific papers. We further construct SciMDR-Eval, an expert-annotated benchmark to evaluate multimodal comprehension within full-length scientific workflows. Experiments demonstrate that models fine-tuned on SciMDR achieve significant improvements across multiple scientific QA benchmarks, particularly in those tasks requiring complex document-level reasoning.
>
---
#### [replaced 040] Contrastive Heliophysical Image Pretraining for Solar Dynamics Observatory Records
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22958](https://arxiv.org/pdf/2511.22958)**

> **作者:** Shiyu Shen; Zhe Gao; Taifeng Chai; Yang Huang; Bin Pan
>
> **备注:** arXiv admin note: This submission has been withdrawn due to violation of arXiv policies for acceptable submissions
>
> **摘要:** Deep learning has revolutionized solar image analysis, yet most approaches train task-specific encoders from scratch or rely on natural-image pretraining that ignores the unique characteristics of Solar Dynamics Observatory (SDO) data. We introduce SolarCHIP, a family of contrastively pretrained visual backbones tailored to multi-instrument SDO observations. SolarCHIP addresses three key challenges in solar imaging: multimodal sensing across AIA and HMI instruments, weak inter-class separability due to slow temporal evolution, and strong intra-class variability with sparse activity signals. Our pretraining framework employs a multi-granularity contrastive objective that jointly aligns (1) global class tokens across co-temporal AIA-HMI pairs to enhance temporal discrimination, (2) local patch tokens at fixed spatial indices to enforce position-consistent, modality-invariant features, and (3) intra-sample patches across different spatial locations to preserve fine-grained spatial structure. We train both CNN- and Vision Transformer-based autoencoders and demonstrate their effectiveness on two downstream tasks: cross-modal translation between HMI and AIA passbands via ControlNet, and full-disk flare classification. Experimental results show that SolarCHIP achieves state-of-the-art performance across both tasks, with particularly strong gains in low-resource settings where labeled data is limited. Ablation studies confirm that each contrastive component contributes essential discriminative capacity at different granularities. By publicly releasing pretrained weights and training code, we provide the heliophysics community with a practical, plug-and-play feature extractor that reduces computational requirements, improves label efficiency, and establishes a reusable foundation for diverse solar imaging applications.
>
---
#### [replaced 041] Glance-or-Gaze: Incentivizing LMMs to Adaptively Focus Search via Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.13942](https://arxiv.org/pdf/2601.13942)**

> **作者:** Hongbo Bai; Yujin Zhou; Yile Wu; Chi-Min Chan; Pengcheng Wen; Kunhao Pan; Sirui Han; Yike Guo
>
> **摘要:** Large Multimodal Models (LMMs) have achieved remarkable success in visual understanding, yet they struggle with knowledge-intensive queries involving long-tail entities or evolving information due to static parametric knowledge. Recent search-augmented approaches attempt to address this limitation, but existing methods rely on indiscriminate whole-image retrieval that introduces substantial visual redundancy and noise, and lack deep iterative reflection, limiting their effectiveness on complex visual queries. To overcome these challenges, we propose Glance-or-Gaze (GoG), a fully autonomous framework that shifts from passive perception to active visual planning. GoG introduces a Selective Gaze mechanism that dynamically chooses whether to glance at global context or gaze into high-value regions, filtering irrelevant information before retrieval. We design a dual-stage training strategy: Reflective GoG Behavior Alignment via supervised fine-tuning instills the fundamental GoG paradigm, while Complexity-Adaptive Reinforcement Learning further enhances the model's capability to handle complex queries through iterative reasoning. Experiments across six benchmarks demonstrate state-of-the-art performance. Ablation studies confirm that both Selective Gaze and complexity-adaptive RL are essential for effective visual search.
>
---
#### [replaced 042] Video Compression Meets Video Generation: Latent Inter-Frame Pruning with Attention Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.05811](https://arxiv.org/pdf/2603.05811)**

> **作者:** Dennis Menn; Yuedong Yang; Bokun Wang; Xiwen Wei; Mustafa Munir; Feng Liang; Radu Marculescu; Chenfeng Xu; Diana Marculescu
>
> **摘要:** Current video generation models suffer from high computational latency, making real-time applications prohibitively costly. In this paper, we address this limitation by exploiting the temporal redundancy inherent in video latent patches. To this end, we propose the Latent Inter-frame Pruning with Attention Recovery (LIPAR) framework, which detects and skips recomputing duplicated latent patches. Additionally, we introduce a novel Attention Recovery mechanism that approximates the attention values of pruned tokens, thereby removing visual artifacts arising from naively applying the pruning method. Empirically, our method increases video editing throughput by $1.53\times$, achieving an average of 19.3 FPS on an NVIDIA RTX 4090 with the 1.3B Self-Forcing model (4-step denoising, FP16). The proposed method does not compromise generation quality and can be seamlessly integrated with the model without additional training. Our approach effectively bridges the gap between traditional compression algorithms and modern generative pipelines.
>
---
#### [replaced 043] PointTransformerX: Portable and Efficient 3D Point Cloud Processing without Sparse Algorithms
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.24169](https://arxiv.org/pdf/2604.24169)**

> **作者:** Laurenz Reichardt; Nikolas Ebert; Oliver Wasenmüller
>
> **备注:** This paper has been accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2026
>
> **摘要:** 3D point cloud perception remains tightly coupled to custom CUDA operators for spatial operations, limiting portability and efficiency on non-NVIDIA, AMD, and embedded hardware. We introduce PointTransformerX (PTX), a fully PyTorch-native vision transformer backbone for 3D point clouds, removing all custom CUDA operators and external libraries while retaining competitive accuracy. PTX introduces 3D-GS-RoPE, a rotary positional embedding that encodes 3D spatial relationships directly in self-attention without neighborhood construction, and further replaces sparse convolutional patch embedding with a linear projection. PTX explores inference-time scaling of attention windows to improve accuracy without retraining. With a redesigned feed-forward network, PTX achieves 98.7\% of PointTransformer V3's accuracy on ScanNet with 79.2\% fewer parameters and executing 1.6\times faster while requiring just 253 MB memory. PTX runs natively on NVIDIA GPUs, AMD GPUs (ROCm), and CPUs, providing an efficient and portable foundation for point cloud perception.
>
---
#### [replaced 044] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于医疗AI评估任务，旨在解决现有基准测试可靠性不足的问题。提出MedCheck框架，对53个医学LLM基准进行分析，揭示数据完整性、临床相关性和安全性等问题。**

- **链接: [https://arxiv.org/pdf/2508.04325](https://arxiv.org/pdf/2508.04325)**

> **作者:** Wenting Chen; Guo Yu; Yiu-Fai Cheung; Meidan Ding; Jie Liu; Zizhan Ma; Wenxuan Wang; Linlin Shen
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Large language models (LLMs) show significant potential in healthcare, prompting numerous benchmarks to evaluate their capabilities. However, concerns persist regarding the reliability of these benchmarks, which often lack clinical fidelity, robust data management, and safety-oriented evaluation metrics. To address these shortcomings, we introduce MedCheck, the first lifecycle-oriented assessment framework specifically designed for medical benchmarks. Our framework deconstructs a benchmark's development into five continuous stages, from design to governance, and provides a comprehensive checklist of 46 medically-tailored criteria. Using MedCheck, we conducted an in-depth empirical evaluation of 53 medical LLM benchmarks. Our analysis uncovers widespread, systemic issues, including a profound disconnect from clinical practice, a crisis of data integrity due to unmitigated contamination risks, and a systematic neglect of safety-critical evaluation dimensions like model robustness and uncertainty awareness. Based on these findings, MedCheck serves as both a diagnostic tool for existing benchmarks and an actionable guideline to foster a more standardized, reliable, and transparent approach to evaluating AI in healthcare.
>
---
#### [replaced 045] StreamAgent: Towards Anticipatory Agents for Streaming Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01875](https://arxiv.org/pdf/2508.01875)**

> **作者:** Haolin Yang; Feilong Tang; Lingxiao Zhao; Xinlin Zhuang; Yifan Lu; Xiang An; Ming Hu; Xiaofeng Zhang; Abdalla Swikir; Junjun He; Zongyuan Ge; Muhammad Haris Khan; Imran Razzak
>
> **摘要:** Real-time streaming video understanding in domains such as autonomous driving and intelligent surveillance poses challenges beyond conventional offline video processing, requiring continuous perception, proactive decision making, and responsive interaction based on dynamically evolving visual content. However, existing methods rely on alternating perception-reaction or asynchronous triggers, lacking task-driven planning and future anticipation, which limits their real-time responsiveness and proactive decision making in evolving video streams. To this end, we propose a StreamAgent that anticipates the temporal intervals and spatial regions expected to contain future task-relevant information to enable proactive and goal-driven responses. Specifically, we integrate question semantics and historical observations through prompting the anticipatory agent to anticipate the temporal progression of key events, align current observations with the expected future evidence, and subsequently adjust the perception action (e.g., attending to task-relevant regions or continuously tracking in subsequent frames). To enable efficient inference, we design a streaming KV-cache memory mechanism that constructs a hierarchical memory structure for selective recall of relevant tokens, enabling efficient semantic retrieval while reducing the overhead of storing all tokens in the traditional KV-cache. Extensive experiments on streaming and long video understanding tasks demonstrate that our method outperforms existing methods in response accuracy and real-time efficiency, highlighting its practical value for real-world streaming scenarios.
>
---
#### [replaced 046] q3-MuPa: Quick, Quiet, Quantitative Multi-Parametric MRI using Physics-Informed Diffusion Models
- **分类: physics.med-ph; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23726](https://arxiv.org/pdf/2512.23726)**

> **作者:** Shishuai Wang; Florian Wiesinger; Noemi Sgambelluri; Carolin Pirkl; Stefan Klein; Juan A. Hernandez-Tamames; Dirk H.J. Poot
>
> **摘要:** The 3D fast silent multi-parametric mapping sequence with zero echo time (MuPa-ZTE) is a novel quantitative MRI (qMRI) acquisition that enables nearly silent scanning by using a 3D phyllotaxis sampling scheme. MuPa-ZTE improves patient comfort and motion robustness, and generates quantitative maps of T1, T2, and proton density using the acquired weighted image series. In this work, we propose a diffusion model-based qMRI mapping method that leverages both a deep generative model and physics-based data consistency to further improve the mapping performance. Furthermore, our method enables additional acquisition acceleration, allowing high-quality qMRI mapping from a fourfold-accelerated MuPa-ZTE scan (approximately 1 minute). Specifically, we trained a denoising diffusion probabilistic model (DDPM) to map MuPa-ZTE image series to qMRI maps, and we incorporated the MuPa-ZTE forward signal model as an explicit data consistency (DC) constraint during inference. We compared our mapping method against a baseline dictionary matching approach and a purely data-driven diffusion model. The diffusion models were trained entirely on synthetic data generated from digital brain phantoms, eliminating the need for large real-scan datasets. We evaluated on synthetic data, a NISM/ISMRM phantom, healthy volunteers, and a patient with brain metastases. The results demonstrated that our method produces 3D qMRI maps with high accuracy, reduced noise and better preservation of structural details. Notably, it generalised well to real scans despite training on synthetic data alone. The combination of the MuPa-ZTE acquisition and our physics-informed diffusion model is termed q3-MuPa, a quick, quiet, and quantitative multi-parametric mapping framework, and our findings highlight its strong clinical potential.
>
---
#### [replaced 047] Omni2Sound: Towards Unified Video-Text-to-Audio Generation
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文属于视频-文本到音频生成任务，解决数据稀缺与多任务竞争问题。提出SoundAtlas数据集和Omni2Sound模型，实现统一的音视频生成。**

- **链接: [https://arxiv.org/pdf/2601.02731](https://arxiv.org/pdf/2601.02731)**

> **作者:** Yusheng Dai; Zehua Chen; Yuxuan Jiang; Baolong Gao; Qiuhong Ke; Jianfei Cai; Jun Zhu
>
> **摘要:** Training a unified model integrating video-to-audio (V2A), text-to-audio (T2A), and joint video-text-to-audio (VT2A) generation offers significant application flexibility, yet faces two unexplored foundational challenges: (1) the scarcity of high-quality audio captions with tight V-A-T alignment, leading to severe semantic conflict between multimodal conditions, and (2) cross-task and intra-task competition, manifesting as an adverse V2A-T2A performance trade-off and modality bias in the VT2A task. First, to address data scarcity, we introduce SoundAtlas, a large-scale dataset (470k pairs) that significantly outperforms existing benchmarks and even human experts in quality. Powered by a novel agentic pipeline, it integrates Vision-to-Language Compression to mitigate visual bias of MLLMs, a Junior-Senior Agent Handoff for a 5$\times$ cost reduction, and rigorous Post-hoc Filtering to ensure fidelity. Consequently, SoundAtlas delivers semantically rich and temporally detailed captions with tight V-A-T alignment. Second, we propose Omni2Sound, a unified VT2A diffusion model supporting flexible input modalities. To resolve the inherent cross-task and intra-task competition, we design a three-stage multi-task progressive training schedule that converts cross-task competition into joint optimization and mitigates modality bias in the VT2A task, maintaining both audio-visual alignment and off-screen audio generation faithfulness. Finally, we construct VGGSound-Omni, a comprehensive benchmark for unified evaluation, including challenging off-screen tracks. With a standard DiT backbone, Omni2Sound achieves unified SOTA performance across all three tasks within a single model, demonstrating strong generalization across benchmarks with heterogeneous input conditions.
>
---
#### [replaced 048] Consist-Retinex: One-Step Noise-Emphasized Consistency Training Accelerates High-Quality Retinex Enhancement
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.08982](https://arxiv.org/pdf/2512.08982)**

> **作者:** Jian Xu; Wei Chen; Shigui Li; Delu Zeng; John Paisley; Qibin Zhao
>
> **摘要:** Retinex-based low-light image enhancement benefits from separating reflectance and illumination, yet recent generative approaches often rely on iterative sampling and are difficult to deploy under strict latency budgets. Consistency models offer a natural route to one-step restoration, but direct adaptation to Retinex-factorized enhancement is unstable: one-step inference is evaluated at the high-noise endpoint, whereas standard training schedules provide little supervision there, and temporal self-consistency alone does not determine the correct conditional target. We propose Consist-Retinex, which first uses a Retinex Transformer Decomposition Network (TDN) to obtain paired reflectance and illumination maps, then trains two conditional consistency models with a Retinex-aware dual objective and adaptive noise-emphasized fixed-point sampling. The dual objective combines trajectory consistency with paired ground-truth component alignment, while the sampling rule concentrates supervision near the inference endpoint without discarding full-range noise coverage. We further provide an endpoint error bound, an anchoring-propagation result, and a high-noise sample-allocation analysis that explain why endpoint supervision and temporal consistency are complementary for one-step Retinex enhancement. Experiments on paired and unpaired low-light benchmarks show that Consist-Retinex obtains the best VE-LOL-L scores among the compared methods under one-step inference and remains competitive on LOL, with substantially reduced sampling and consistency-stage training cost in the reported setup.
>
---
#### [replaced 049] Incoherent Deformation, Not Capacity: Diagnosing and Mitigating Overfitting in Dynamic Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.16747](https://arxiv.org/pdf/2604.16747)**

> **作者:** Ahmad Droby
>
> **备注:** 10 pages, 6 figures, 2 tables
>
> **摘要:** Dynamic 3D Gaussian Splatting methods achieve strong training-view PSNR on monocular video but generalize poorly: on the D-NeRF benchmark we measure an average train-test PSNR gap of 6.18 dB, rising to 11 dB on individual scenes. We report two findings that together account for most of that gap. Finding 1 (the role of splitting). A systematic ablation of the Adaptive Density Control pipeline (split, clone, prune, frequency, threshold, schedule) shows that splitting is responsible for over 80% of the gap: disabling split collapses the cloud from 44K to 3K Gaussians and the gap from 6.18 dB to 1.15 dB. Across all threshold-varying ablations, gap is log-linear in count (r = 0.995, bootstrap 95% CI [0.99, 1.00]), which suggests a capacity-based explanation. Finding 2 (the role of deformation coherence). We show that the capacity explanation is incomplete. A local-smoothness penalty on the per-Gaussian deformation field -- Elastic Energy Regularization (EER) -- reduces the gap by 40.8% while growing the cloud by 85%. Measuring per-Gaussian strain directly on trained checkpoints, EER reduces mean strain by 99.72% (median 99.80%) across all 8 scenes; on 8/8 scenes the median Gaussian under EER is less strained than the 1st-percentile (best-behaved) Gaussian under baseline. Alongside EER, we evaluate two further regularizers: GAD, a loss-rate-aware densification threshold, and PTDrop, a jitter-weighted Gaussian dropout. GAD+EER reduces the gap by 48%; adding PTDrop and a soft growth cap reaches 57%. We confirm that coherence generalizes to (a) a different deformation architecture (Deformable-3DGS, +40.6% gap reduction at re-tuned lambda), and (b) real monocular video (4 HyperNeRF scenes, reducing the mean PSNR gap by 14.9% at the same lambda as D-NeRF, with near-zero quality cost). The overfitting in dynamic 3DGS is driven by incoherent deformation, not parameter count.
>
---
#### [replaced 050] CAGE-SGG: Counterfactual Active Graph Evidence for Open-Vocabulary Scene Graph Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.22274](https://arxiv.org/pdf/2604.22274)**

> **作者:** Suiyang Guang; Chenyu Liu; Ruohan Zhang; Siyuan Chen
>
> **备注:** This manuscript has been withdrawn by the authors because we found a methodological flaw in the formulation and evaluation of the proposed approach. The issue affects the reliability of the experimental results and the conclusions drawn from them. Therefore, the authors consider the current version unsuitable for citation or further use
>
> **摘要:** Open-vocabulary scene graph generation (SGG) aims to describe visual scenes with flexible and fine-grained relation phrases beyond a fixed predicate vocabulary. While recent vision-language models greatly expand the semantic coverage of SGG, they also introduce a critical reliability issue: predicted relations may be driven by language priors or object co-occurrence rather than grounded visual evidence. In this paper, we propose an evidence-rounded open-vocabulary SGG framework based on counterfactual relation verification. Instead of directly accepting plausible relation proposals, our method verifies whether each candidate relation is supported by relation-pecific visual, geometric, and contextual evidence. Specifically, we first generate open-vocabulary relation candidates with a vision-language proposer, then decompose predicate phrases into soft evidence bases such as support, contact, containment, depth, motion, and state. A relation-conditioned evidence encoder extracts predicate-relevant cues, while a counterfactual verifier tests whether the relation score decreases when necessary vidence is removed and remains stable under irrelevant perturbations. We further introduce contradiction-aware predicate learning and graph-level preference optimization to improve fine-grained discrimination and global graph consistency. Experiments on conventional, open-vocabulary, and panoptic SGG benchmarks show that our method consistently improves standard recall-based metrics, unseen predicate generalization, and counterfactual grounding quality. These results demonstrate that moving from relation generation to relation verification leads to more reliable, interpretable, and evidence-grounded scene graphs.
>
---
#### [replaced 051] The devil is in the details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20340](https://arxiv.org/pdf/2512.20340)**

> **作者:** Qingdong He; Xueqin Chen; Yanjie Pan; Peng Tang; Pengcheng Xu; Zhenye Gan; Chengjie Wang; Xiaobin Hu; Jiangning Zhang; Yabiao Wang
>
> **备注:** Accepted by CVPR 2026 (Main Conference)
>
> **摘要:** Although diffusion transformer (DiT)-based video virtual try-on (VVT) has made significant progress in synthesizing realistic videos, existing methods still struggle to capture fine-grained garment dynamics and preserve background integrity across video frames. They also incur high computational costs due to additional interaction modules introduced into DiTs, while the limited scale and quality of existing public datasets also restrict model generalization and effective training. To address these challenges, we propose a novel framework, KeyTailor, along with a large-scale, high-definition dataset, ViT-HD. The core idea of KeyTailor is a keyframe-driven details injection strategy, motivated by the fact that keyframes inherently contain both foreground dynamics and background consistency. Specifically, KeyTailor adopts an instruction-guided keyframe sampling strategy to filter informative frames from the input video. Subsequently,two tailored keyframe-driven modules, the garment details enhancement module and the collaborative background optimization module, are employed to distill garment dynamics into garment-related latents and to optimize the integrity of background latents, both guided by this http URL enriched details are then injected into standard DiT blocks together with pose, mask, and noise latents, enabling efficient and realistic try-on video synthesis. This design ensures consistency without explicitly modifying the DiT architecture, while simultaneously avoiding additional complexity. In addition, our dataset ViT-HD comprises 15, 070 high-quality video samples at a resolution of 810*1080, covering diverse garments. Extensive experiments demonstrate that KeyTailor outperforms state-of-the-art baselines in terms of garment fidelity and background integrity across both dynamic and static scenarios.
>
---
#### [replaced 052] Rethinking Satellite Image Restoration for Onboard AI: A Lightweight Learning-Based Approach
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.12807](https://arxiv.org/pdf/2604.12807)**

> **作者:** Adrien Dorise; Marjorie Bellizzi; Omar Hlimi
>
> **备注:** Accepted at AI4SPACE@CVPR conference
>
> **摘要:** Satellite image restoration aims to improve image quality by compensating for degradations (e.g., noise and blur) introduced by the imaging system and acquisition conditions. As a fundamental preprocessing step, restoration directly impacts both ground-based product generation and emerging onboard AI applications. Traditional restoration pipelines based on sequential physical models are computationally intensive and slow, making them unsuitable for onboard environments. In this paper, we introduce ConvBEERS: a Convolutional Board-ready Embedded and Efficient Restoration model for Space to investigate whether a light and non-generative residual convolutional network, trained on simulated satellite data, can match or surpass a traditional ground-processing restoration pipeline across multiple operating conditions. Experiments conducted on simulated datasets and real Pleiades-HR imagery demonstrate that the proposed approach achieves competitive image quality, with a +6.9dB PSNR improvement. Evaluation on a downstream object detection task demonstrates that restoration significantly improves performance, with up to +5.1% mAP@50. In addition, successful deployment on a Xilinx Versal VCK190 FPGA validates its practical feasibility for satellite onboard processing, with a ~41x reduction in latency compared to the traditional pipeline. These results demonstrate the relevance of using lightweight CNNs to achieve competitive restoration quality while addressing real-world constraints in spaceborne systems.
>
---
#### [replaced 053] MINOS: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态生成评估任务，旨在解决现有评估模型数据质量低、跨任务性能差的问题。通过构建高质量数据集并采用联合训练和偏好对齐策略，提出Minos模型，实现优异的评估效果。**

- **链接: [https://arxiv.org/pdf/2506.02494](https://arxiv.org/pdf/2506.02494)**

> **作者:** Junzhe Zhang; Huixuan Zhang; Xinyu Hu; Li Lin; Mingqi Gao; Shi Qiu; Xiaojun Wan
>
> **备注:** Accepted to the Findings of ACL 2026
>
> **摘要:** Evaluation is important for multimodal generation tasks, while traditional multimodal evaluation metrics suffer from several limitations. With the rapid progress of MLLMs, there is growing interest in applying MLLMs to build general evaluation systems. However, existing researches often simply collect large-scale evaluation data for training, while overlooking the quality of evaluation data. What's more, current proposed evaluation models often struggle to achieve consistently strong performance across both image-to-text (I2T) and text-to-image (T2I) tasks. In this paper, through rigorous quality control strategies, we construct a comprehensive multimodal evaluation dataset, Minos-57K, with evaluation samples across 15 datasets, for developing the multimodal evaluation model Minos with SFT and preference alignment training strategies. Notably, despite using less than half the scale of the training data of prior work, our model achieves state-of-the-art evaluation performance across 16 out-of-domain datasets covering both I2T and T2I tasks among all open-source multimodal evaluation models and remain competitive with closed-source models. Extensive experiments demonstrate the importance of leveraging quality control process, jointly training on evaluation data from both I2T and T2I generation tasks and further preference alignment.
>
---
#### [replaced 054] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20032](https://arxiv.org/pdf/2511.20032)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng; Zhixing Tan
>
> **备注:** CVPR 2026
>
> **摘要:** Visual attention serves as the primary mechanism through which MLLMs interpret visual information; however, its limited localization capability often leads to hallucinations. We observe that although MLLMs can accurately extract visual semantics from visual tokens, they fail to fully leverage this advantage during subsequent inference. To address this limitation, we propose Vision-Guided Attention (VGA), a training-free method that first constructs precise visual grounding by exploiting the semantic content of visual tokens, and then uses this grounding to guide the model's focus toward relevant visual regions. In image captioning, VGA further refines this guidance dynamically during generation by suppressing regions that have already been described. In VGA, each token undergoes only a single forward pass, introducing a negligible latency overhead. In addition, VGA is fully compatible with efficient attention implementations such as FlashAttention. Extensive experiments across diverse MLLMs and multiple hallucination benchmarks demonstrate that VGA achieves state-of-the-art dehallucination performance. Further analysis confirms that explicit visual guidance plays a crucial role in enhancing the visual understanding capabilities of MLLMs.
>
---
#### [replaced 055] Efficient Zero-Shot Inpainting with Decoupled Diffusion Guidance
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.18365](https://arxiv.org/pdf/2512.18365)**

> **作者:** Badr Moufad; Navid Bagheri Shouraki; Alain Oliviero Durmus; Thomas Hirtz; Eric Moulines; Jimmy Olsson; Yazid Janati
>
> **摘要:** Diffusion models have emerged as powerful priors for image editing tasks such as inpainting and local modification, where the objective is to generate realistic content that remains consistent with observed regions. In particular, zero-shot approaches that leverage a pretrained diffusion model, without any retraining, have been shown to achieve highly effective reconstructions. However, state-of-the-art zero-shot methods typically rely on a sequence of surrogate likelihood functions, whose scores are used as proxies for the ideal score. This procedure however requires vector-Jacobian products through the denoiser at every reverse step, introducing significant memory and runtime overhead. To address this issue, we propose a new likelihood surrogate that yields simple and efficient to sample Gaussian posterior transitions, sidestepping the backpropagation through the denoiser network. Our extensive experiments show that our method achieves strong observation consistency compared with fine-tuned baselines and produces coherent, high-quality reconstructions, all while significantly reducing inference cost. Code is available at this https URL.
>
---
#### [replaced 056] Learning Vision-Based Omnidirectional Navigation: A Teacher-Student Approach Using Monocular Depth Estimation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人导航任务，解决工业环境中2D LiDAR传感器无法感知三维障碍物的问题。通过教师-学生框架，利用单目深度估计实现无LiDAR的全景导航。**

- **链接: [https://arxiv.org/pdf/2603.01999](https://arxiv.org/pdf/2603.01999)**

> **作者:** Jan Finke; Wayne Paul Martis; Adrian Schmelter; Lars Erbach; Christian Jestel; Marvin Wiedemann
>
> **摘要:** Reliable obstacle avoidance in industrial settings demands 3D scene understanding, but widely used 2D LiDAR sensors perceive only a single horizontal slice of the environment, missing critical obstacles above or below the scan plane. We present a teacher-student framework for vision-based mobile robot navigation that eliminates the need for LiDAR sensors. A teacher policy trained via Proximal Policy Optimization (PPO) in NVIDIA Isaac Lab leverages privileged 2D LiDAR observations that account for the full robot footprint to learn robust navigation. The learned behavior is distilled into a student policy that relies solely on monocular depth maps predicted by a fine-tuned Depth Anything V2 model from four RGB cameras. The complete inference pipeline, comprising monocular depth estimation (MDE), policy execution, and motor control, runs entirely onboard an NVIDIA Jetson Orin AGX mounted on a DJI RoboMaster platform, requiring no external computation for inference. In simulation, the student achieves success rates of 82-96.5%, consistently outperforming the standard 2D LiDAR teacher (50-89%). In real-world experiments, the MDE-based student outperforms the 2D LiDAR teacher when navigating around obstacles with complex 3D geometries, such as overhanging structures and low-profile objects, that fall outside the single scan plane of a 2D LiDAR.
>
---
#### [replaced 057] U-FaceBP: Uncertainty-aware Bayesian Ensemble Deep Learning for Face Video-based Blood Pressure Estimation
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2412.10679](https://arxiv.org/pdf/2412.10679)**

> **作者:** Yusuke Akamatsu; Akinori F. Ebihara; Terumi Umematsu
>
> **备注:** Accepted to IEEE Transactions on Instrumentation and Measurement
>
> **摘要:** Blood pressure (BP) measurement is crucial for daily health assessment. Remote photoplethysmography (rPPG), which extracts pulse waves from face videos captured by a camera, has the potential to enable convenient BP measurement without specialized medical devices. However, there are various uncertainties in BP estimation using rPPG, leading to limited estimation performance and reliability. In this paper, we propose U-FaceBP, an uncertainty-aware Bayesian ensemble deep learning method for face video-based BP estimation. U-FaceBP models aleatoric and epistemic uncertainties in face video-based BP estimation with a Bayesian neural network (BNN). Additionally, we design U-FaceBP as an ensemble method, estimating BP from rPPG signals, PPG signals derived from face videos, and face images using multiple BNNs. Large-scale experiments on two datasets involving 1197 subjects from diverse racial groups demonstrate that U-FaceBP outperforms state-of-the-art BP estimation methods. Furthermore, we show that the uncertainty estimates provided by U-FaceBP are informative and useful for guiding modality fusion, assessing prediction reliability, and analyzing performance across racial groups.
>
---
#### [replaced 058] At FullTilt: Real-Time Open-Set 3D Macromolecule Detection Directly from Tilted 2D Projections
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10766](https://arxiv.org/pdf/2604.10766)**

> **作者:** Ming-Yang Ho; Alberto Bartesaghi
>
> **摘要:** Open-set 3D macromolecule detection in cryogenic electron tomography eliminates the need for target-specific model retraining. However, strict VRAM constraints prohibit processing an entire 3D tomogram, forcing current methods to rely on slow sliding-window inference over extracted subvolumes. To overcome this, we propose FullTilt, an end-to-end framework that redefines 3D detection by operating directly on aligned 2D tilt-series. Because a tilt-series contains significantly fewer images than slices in a reconstructed tomogram, FullTilt eliminates redundant volumetric computation, accelerating inference by orders of magnitude. To process the entire tilt-series simultaneously, we introduce a tilt-series encoder to efficiently fuse cross-view information. We further propose a multiclass visual prompt encoder for flexible prompting, a tilt-aware query initializer to effectively anchor 3D queries, and an auxiliary geometric primitives module to enhance the model's understanding of multi-view geometry while improving robustness to adverse imaging artifacts. Extensive evaluations on three real-world datasets demonstrate that FullTilt achieves state-of-the-art zero-shot performance while drastically reducing runtime and VRAM requirements, paving the way for rapid, large-scale visual proteomics analysis. All code and data will be publicly available upon publication.
>
---
#### [replaced 059] OnSiteVRU: A High-Resolution Trajectory Dataset for High-Density Vulnerable Road Users
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出OnSiteVRU数据集，用于高密度VRU轨迹研究，解决现有数据不足问题，提升自动驾驶系统安全性。**

- **链接: [https://arxiv.org/pdf/2503.23365](https://arxiv.org/pdf/2503.23365)**

> **作者:** Zhangcun Yan; Jianqiang Li; Peng Hang; Jian Sun
>
> **摘要:** With the acceleration of urbanization and the growth of transportation demands, the safety of vulnerable road users (VRUs, such as pedestrians and cyclists) in mixed traffic flows has become increasingly prominent, necessitating high-precision and diverse trajectory data to support the development and optimization of autonomous driving systems. However, existing datasets fall short in capturing the diversity and dynamics of VRU behaviors, making it difficult to meet the research demands of complex traffic environments. To address this gap, this study developed the OnSiteVRU datasets, which cover a variety of scenarios, including intersections, road segments, and urban villages. These datasets provide trajectory data for motor vehicles, electric bicycles, and human-powered bicycles, totaling approximately 17,429 trajectories with a precision of 0.04 seconds. The datasets integrate both aerial-view natural driving data and onboard real-time dynamic detection data, along with environmental information such as traffic signals, obstacles, and real-time maps, enabling a comprehensive reconstruction of interaction events. The results demonstrate that VRU\_Data outperforms traditional datasets in terms of VRU density and scene coverage, offering a more comprehensive representation of VRU behavioral characteristics. This provides critical support for traffic flow modeling, trajectory prediction, and autonomous driving virtual testing. The dataset is publicly available for download at: this https URL.
>
---
#### [replaced 060] FASTER: Rethinking Real-Time Flow VLAs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型的实时执行任务，解决反应延迟问题。通过提出FASTER方法，优化采样策略以缩短反应时间，提升机器人实时响应能力。**

- **链接: [https://arxiv.org/pdf/2603.19199](https://arxiv.org/pdf/2603.19199)**

> **作者:** Yuxiang Lu; Zhe Liu; Xianzhe Fan; Zhenya Yang; Jinghua Hou; Junyi Li; Kaixin Ding; Hengshuang Zhao
>
> **备注:** Project page: this https URL
>
> **摘要:** Real-time execution is crucial for deploying Vision-Language-Action (VLA) models in the physical world. Existing asynchronous inference methods primarily optimize trajectory smoothness, but neglect the critical latency in reacting to environmental changes. By rethinking the notion of reaction in action chunking policies, this paper presents a systematic analysis of the factors governing reaction time. We show that reaction time follows a uniform distribution determined jointly by the Time to First Action (TTFA) and the execution horizon. Moreover, we reveal that the standard practice of applying a constant schedule in flow-based VLAs can be inefficient and forces the system to complete all sampling steps before any movement can start, forming the bottleneck in reaction latency. To overcome this issue, we propose Fast Action Sampling for ImmediaTE Reaction (FASTER). By introducing a Horizon-Aware Schedule, FASTER adaptively prioritizes near-term actions during flow sampling, compressing the denoising of the immediate reaction by tenfold (e.g., in $\pi_{0.5}$ and X-VLA) into a single step, while preserving the quality of long-horizon trajectory. Coupled with a streaming client-server pipeline, FASTER substantially reduces the effective reaction latency on real robots, especially when deployed on consumer-grade GPUs. Real-world experiments, including a highly dynamic table tennis task, prove that FASTER unlocks unprecedented real-time responsiveness for generalist policies, enabling rapid generation of accurate and smooth trajectories.
>
---
#### [replaced 061] Value-Guided Iterative Refinement and the DIQ-H Benchmark for Evaluating VLM Robustness
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.03992](https://arxiv.org/pdf/2512.03992)**

> **作者:** Hanwen Wan; Zexin Lin; Yixuan Deng; Xiaoqiang Ji
>
> **摘要:** Vision-Language Models (VLMs) are essential for embodied AI and safety-critical applications, such as robotics and autonomous systems. However, existing benchmarks primarily focus on static or curated visual inputs, neglecting the challenges posed by adversarial conditions, value misalignment, and error propagation in continuous deployment. Current benchmarks either overlook the impact of real-world perturbations, or fail to account for the cumulative effect of inconsistent reasoning over time. To address these gaps, we introduce the Degraded Image Quality Leading to Hallucinations (DIQ-H) benchmark, the first to evaluate VLMs under adversarial visual conditions in continuous sequences. DIQ-H simulates real-world stressors including motion blur, sensor noise, and compression artifacts, and measures how these corruptions lead to persistent errors and misaligned outputs across time. The benchmark explicitly models error propagation and its long-term value consistency. To enhance scalability and reduce costs for safety-critical evaluation, we propose the Value-Guided Iterative Refinement (VIR) framework, which automates the generation of high-quality, ethically aligned ground truth annotations. VGIR leverages lightweight VLMs to detect and refine value misalignment, improving accuracy from 72.2% to 83.3%, representing a 15.3% relative improvement. The DIQ-H benchmark and VGIR framework provide a robust platform for embodied AI safety assessment, revealing vulnerabilities in error recovery, ethical consistency, and temporal value alignment.
>
---
#### [replaced 062] Uncertainty-Aware Information Pursuit for Interpretable and Reliable Medical Image Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.16742](https://arxiv.org/pdf/2506.16742)**

> **作者:** Md Nahiduzzaman; Steven Korevaar; Zongyuan Ge; Feng Xia; Alireza Bab-Hadiashar; Ruwan Tennakoon
>
> **备注:** Accepted to IEEE Transactions on Medical Imaging (IEEE TMI 2025)
>
> **摘要:** To be adopted in safety-critical domains like medical image analysis, AI systems must provide human-interpretable decisions. Variational Information Pursuit (V-IP) offers an interpretable-by-design framework by sequentially querying input images for human-understandable concepts, using their presence or absence to make predictions. However, existing V-IP methods overlook sample-specific uncertainty in concept predictions, which can arise from ambiguous features or model limitations, leading to suboptimal query selection and reduced robustness. In this paper, we propose an interpretable and uncertainty-aware framework for medical imaging that addresses these limitations by accounting for upstream uncertainties in concept-based, interpretable-by-design models. Specifically, we introduce two uncertainty-aware models, EUAV-IP and IUAV-IP, that integrate uncertainty estimates into the V-IP querying process to prioritize more reliable concepts per sample. EUAV-IP skips uncertain concepts via masking, while IUAV-IP incorporates uncertainty into query selection implicitly for more informed and clinically aligned decisions. Our approach allows models to make reliable decisions based on a subset of concepts tailored to each individual sample, without human intervention, while maintaining overall interpretability. We evaluate our methods on five medical imaging datasets across four modalities: dermoscopy, X-ray, ultrasound, and blood cell imaging. The proposed IUAV-IP model achieves state-of-the-art accuracy among interpretable-by-design approaches on four of the five datasets, and generates more concise explanations by selecting fewer yet more informative concepts. These advances enable more reliable and clinically meaningful outcomes, enhancing model trustworthiness and supporting safer AI deployment in healthcare.
>
---
#### [replaced 063] Foundation Model-Driven Semantic Change Detection in Remote Sensing Imagery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.13780](https://arxiv.org/pdf/2602.13780)**

> **作者:** Hengtong Shen; Li Yan; Hong Xie; Yaxuan Wei; Xinhao Li; Wenfei Shen; Peixian Lv; Fei Tan
>
> **摘要:** Remote sensing (RS) change detection is essential for interpreting surface dynamics. Semantic change detection (SCD) further enables pixel-level understanding of multi-class transitions, yet remains sensitive to pseudo-changes induced by imaging conditions. Recent RS foundation models extract semantically consistent features across temporal and environmental variations, which is critical for mitigating pseudo-changes. However, existing SCD methods are often rigid and backbone-specific, lacking the flexibility to integrate diverse multi-scale features from emerging foundation models. To this end, we introduce a modular Cascaded Gated Decoder (CG-Decoder) that bridges various backbones and SCD tasks, processing multi-scale features in a coarse-to-fine manner while enabling adaptive change extraction. Building upon the RS foundation model PerA, we present PerASCD, a unified SCD framework. We further propose a Soft Semantic Consistency Loss (SSCLoss) to mitigate numerical instability in mixed-precision training. Extensive experiments on SECOND and LandsatSCD show that PerASCD achieves new state-of-the-art Sek scores (26.11% and 65.21%), surpassing the previous best by 0.61% and 4.95%, respectively. It also demonstrates exceptional data efficiency (outperforming the full-data baseline with 50% data), seamless cross-backbone generalization, and enhanced interpretability. Our approach maintains robust semantic consistency under radiometric variations, providing a reliable SCD solution. Code: this https URL.
>
---
#### [replaced 064] A Diffeomorphism Groupoid and Algebroid Framework for Discontinuous Image Registration
- **分类: math.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11806](https://arxiv.org/pdf/2603.11806)**

> **作者:** Lili Bao; Bin Xiao; Shihui Ying; Stefan Sommer
>
> **摘要:** In this paper, we propose a novel mathematical framework for piecewise diffeomorphic image registration that involves discontinuous sliding motion using a diffeomorphism groupoid and algebroid approach. The traditional Large Deformation Diffeomorphic Metric Mapping (LDDMM) registration method builds on Lie groups, which assume continuity and smoothness in velocity fields, limiting its applicability in handling discontinuous sliding motion. To overcome this limitation, we extend the diffeomorphism Lie groups to a framework of discontinuous diffeomorphism Lie groupoids, allowing for discontinuities along sliding boundaries while maintaining diffeomorphism within homogeneous regions. We provide a rigorous analysis of the associated mathematical structures, including Lie algebroids and their duals, and derive specific Euler-Arnold equations to govern optimal flows for discontinuous deformations. Numerical tests are performed to validate the efficiency of the proposed approach.
>
---
#### [replaced 065] An Affordable, Wearable Stereo-Eye-Tracking Platform
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.24331](https://arxiv.org/pdf/2604.24331)**

> **作者:** Alexander Zimmer; Yasmeen Abdrabou; Enkelejda Kasneci
>
> **摘要:** Research on video-based eye-tracking has long explored stereo and glint-based methods, yet existing wearable eye trackers - both commercial and open-source - offer limited flexibility for algorithm development and comparative evaluation. We present an affordable, wearable stereo eye-tracking platform built from off-the-shelf and 3D-printable components that explicitly targets this gap. The system combines four infrared eye cameras, infrared illumination, an optional scene camera, and software support for calibration and synchronized data acquisition. By design, the platform supports multiple eye-tracking paradigms, including stereo, glint-based, and binocular approaches, within a single hardware configuration. Rather than optimizing for end-user robustness, the platform prioritizes modularity and extensibility for research use. This paper focuses on the hardware architecture and calibration pipeline and demonstrates the feasibility of the approach using a prototype implementation. All hardware designs and documentation are made openly available.
>
---
#### [replaced 066] Real-time Global Illumination for Dynamic 3D Gaussian Scenes
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.17897](https://arxiv.org/pdf/2503.17897)**

> **作者:** Chenxiao Hu; Meng Gai; Guoping Wang; Sheng Li
>
> **备注:** accepted by IEEE Transactions on Visualization and Computer Graphics
>
> **摘要:** We present a real-time global illumination approach along with a pipeline for dynamic 3D Gaussian models and meshes. Building on a formulated surface light transport model for 3D Gaussians, we address key performance challenges with a fast compound stochastic ray-tracing algorithm and an optimized 3D Gaussian rasterizer. Our pipeline integrates multiple real-time techniques to accelerate performance and achieve high-quality lighting effects. Our approach enables real-time rendering of dynamic scenes with interactively editable materials and dynamic lighting of diverse multi-lights settings, capturing mutual multi-bounce light transport (indirect illumination) between 3D Gaussians and mesh. Additionally, we present a real-time renderer with an interactive user interface, validating our approach and demonstrating its practicality and high efficiency with over 40 fps in scenes including both 3D Gaussians and mesh. Furthermore, our work highlights the potential of 3D Gaussians in real-time applications with dynamic lighting, offering insights into performance and optimization.
>
---
#### [replaced 067] Co-generation of Layout and Shape from Text via Autoregressive 3D Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.16552](https://arxiv.org/pdf/2604.16552)**

> **作者:** Zhenggang Tang; Yuehao Wang; Yuchen Fan; Jun-Kun Chen; Yu-Ying Yeh; Kihyuk Sohn; Zhangyang Wang; Qixing Huang; Alexander Schwing; Rakesh Ranjan; Dilin Wang; Zhicheng Yan
>
> **摘要:** Recent text-to-scene generation approaches largely reduced the manual efforts required to create 3D scenes. However, their focus is either to generate a scene layout or to generate objects, and few generate both. The generated scene layout is often simple even with LLM's help. Moreover, the generated scene is often inconsistent with the text input that contains non-trivial descriptions of the shape, appearance, and spatial arrangement of the objects. We present a new paradigm of sequential text-to-scene generation and propose a novel generative model for interactive scene creation. At the core is a 3D Autoregressive Diffusion model 3D-ARD+, which unifies the autoregressive generation over a multimodal token sequence and diffusion generation of next-object 3D latents. To generate the next object, the model uses one autoregressive step to generate the coarse-grained 3D latents in the scene space, conditioned on both the current seen text instructions and already synthesized 3D scene. It then uses a second step to generate the 3D latents in the smaller object space, which can be decoded into fine-grained object geometry and appearance. We curate a large dataset of 230K indoor scenes with paired text instructions for training. We evaluate 7B 3D-ARD+, on challenging scenes, and showcase the model can generate and place objects following non-trivial spatial layout and semantics prescribed by the text instructions.
>
---
#### [replaced 068] ChartVerse: Scaling Chart Reasoning via Reliable Programmatic Synthesis from Scratch
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.13606](https://arxiv.org/pdf/2601.13606)**

> **作者:** Zheng Liu; Honglin Lin; Chonghan Qin; Xiaoyang Wang; Xin Gao; Yu Li; Mengzhang Cai; Yun Zhu; Zhanping Zhong; Qizhi Pei; Zhuoshi Pan; Xiaoran Shang; Bin Cui; Conghui He; Wentao Zhang; Lijun Wu
>
> **备注:** 29 pages
>
> **摘要:** Chart reasoning is a critical capability for Vision Language Models (VLMs). However, the development of open-source models is severely hindered by the lack of high-quality training data. Existing datasets suffer from a dual challenge: synthetic charts are often simplistic and repetitive, while the associated QA pairs are prone to hallucinations and lack the reasoning depth required for complex tasks. To bridge this gap, we propose ChartVerse, a scalable framework designed to synthesize complex charts and reliable reasoning data from scratch. (1) To address the bottleneck of simple patterns, we first introduce Rollout Posterior Entropy (RPE), a novel metric that quantifies chart complexity. Guided by RPE, we develop complexity-aware chart coder to autonomously synthesize diverse, high-complexity charts via executable programs. (2) To guarantee reasoning rigor, we develop truth-anchored inverse QA synthesis. Diverging from standard generation, we adopt an answer-first paradigm: we extract deterministic answers directly from the source code, generate questions conditional on these anchors, and enforce strict consistency verification. To further elevate difficulty and reasoning depth, we filter samples based on model fail-rate and distill high-quality Chain-of-Thought (CoT) reasoning. We curate ChartVerse-SFT-600K and ChartVerse-RL-40K using Qwen3-VL-30B-A3B-Thinking as the teacher. Experimental results demonstrate that ChartVerse-8B achieves state-of-the-art performance, notably surpassing its teacher and rivaling the stronger Qwen3-VL-32B-Thinking. We release our code, model weights, and datasets in this https URL.
>
---
#### [replaced 069] Bridging Visual and Wireless Sensing via a Unified Radiation Field for 3D Radio Map Construction
- **分类: cs.NI; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.19216](https://arxiv.org/pdf/2601.19216)**

> **作者:** Chaozheng Wen; Jingwen Tong; Zehong Lin; Chenghong Bian; Jun Zhang
>
> **备注:** The code for this work will be publicly available at: this https URL
>
> **摘要:** The emerging applications of next-generation wireless networks demand high-fidelity environmental intelligence. 3D radio maps bridge physical environments and electromagnetic propagation for spectrum planning and environment-aware sensing. However, most existing methods treat visual and wireless data as independent modalities and fail to leverage shared electromagnetic propagation principles. To bridge this gap, we propose URF-GS, a unified radio-optical radiation field framework based on 3D Gaussian splatting and inverse rendering for 3D radio map construction. By fusing cross-modal observations, our method recovers scene geometry and material properties to predict radio signals under arbitrary transceiver configurations without retraining. Experiments demonstrate up to a 24.7% improvement in spatial spectrum accuracy and a 10x increase in sample efficiency compared with NeRF-based methods. We further showcase URF-GS in Wi-Fi AP deployment and robot path planning tasks. This unified visual-wireless representation supports holistic radiation field modeling for future wireless communication systems.
>
---
#### [replaced 070] FLARE: Fully Integration of Vision-Language Representations for Deep Cross-Modal Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.09925](https://arxiv.org/pdf/2504.09925)**

> **作者:** Zheng Liu; Mengjie Liu; Jingzhou Chen; Jingwei Xu; Bin Cui; Conghui He; Wentao Zhang
>
> **摘要:** We introduce FLARE, a family of vision language models (VLMs) with a fully vision-language alignment and integration paradigm. Unlike existing approaches that rely on single MLP projectors for modality alignment and defer cross-modal interaction to LLM decoding, FLARE achieves deep, dynamic integration throughout the pipeline. Our key contributions include: (1) Text-Guided Vision Encoding that incorporates textual information during vision encoding to achieve pixel-level alignment; (2) Context-Aware Alignment Decoding that aggregates visual features conditioned on textual context during decoding for query-level integration; (3) Dual-Semantic Mapping Loss to supervise feature mapping from both modalities and enable modality-level bridging; and (4) Text-Driven VQA Synthesis that leverages high-quality text to generate VQA pairs and synthesize corresponding images, enabling data-level optimization. We train FLARE at 3B and 8B scales under both fixed and dynamic resolution settings, demonstrating that our full-modality alignment significantly outperforms existing methods while maintaining strong generalizability. FLARE 3B surpasses Cambrian-1 8B and Florence-VL 8B using only 630 vision tokens. Ablation studies reveal that FLARE achieves superior performance over existing methods with minimal computational cost. Even without dynamic resolution, FLARE outperforms LLaVA-NeXT, validating the effectiveness of our approach. We release our code, model weights, and dataset in this https URL.
>
---
#### [replaced 071] Inferix: A Block-Diffusion based Next-Generation Inference Engine for World Simulation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.20714](https://arxiv.org/pdf/2511.20714)**

> **作者:** Inferix Team; Tianyu Feng; Yizeng Han; Jiahao He; Yuanyu He; Xi Lin; Teng Liu; Hanfeng Lu; Jiasheng Tang; Wei Wang; Zhiyuan Wang; Jichao Wu; Mingyang Yang; Yinghao Yu; Zeyu Zhang; Bohan Zhuang
>
> **摘要:** World models serve as core simulators for fields such as agentic AI, embodied AI, and gaming, capable of generating long, physically realistic, and interactive high-quality videos. Moreover, scaling these models could unlock emergent capabilities in visual perception, understanding, and reasoning, paving the way for a new paradigm that moves beyond current LLM-centric vision foundation models. A key breakthrough empowering them is the semi-autoregressive (block-diffusion) decoding paradigm, which merges the strengths of diffusion and autoregressive methods by generating video tokens in block-applying diffusion within each block while conditioning on previous ones, resulting in more coherent and stable video sequences. Crucially, it overcomes limitations of standard video diffusion by reintroducing LLM-style KV Cache management, enabling efficient, variable-length, and high-quality generation. Therefore, Inferix is specifically designed as a next-generation inference engine to enable immersive world synthesis through optimized semi-autoregressive decoding processes. This dedicated focus on world simulation distinctly sets it apart from systems engineered for high-concurrency scenarios (like vLLM or SGLang) and from classic video diffusion models (such as xDiTs). Inferix further enhances its offering with interactive video streaming and profiling, enabling real-time interaction and realistic simulation to accurately model world dynamics. Additionally, it supports efficient benchmarking through seamless integration of LV-Bench, a new fine-grained evaluation benchmark tailored for minute-long video generation scenarios. We hope the community will work together to advance Inferix and foster world model exploration.
>
---
#### [replaced 072] R2RGEN: Real-to-Real 3D Data Generation for Spatially Generalized Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决空间泛化问题。通过生成真实世界3D数据，提升策略在不同空间配置下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.08547](https://arxiv.org/pdf/2510.08547)**

> **作者:** Xiuwei Xu; Angyuan Ma; Hankun Li; Bingyao Yu; Zheng Zhu; Jie Zhou; Jiwen Lu
>
> **备注:** Accepted to RSS 2026. Project page: this https URL
>
> **摘要:** Towards the aim of generalized robotic manipulation, spatial generalization is the most fundamental capability that requires the policy to work robustly under different spatial distribution of objects, environment and agent itself. To achieve this, substantial human demonstrations need to be collected to cover different spatial configurations for training a generalized visuomotor policy via imitation learning. Prior works explore a promising direction that leverages data generation to acquire abundant spatially diverse data from minimal source demonstrations. However, most approaches face significant sim-to-real gap and are often limited to constrained settings, such as fixed-base scenarios and predefined camera viewpoints. In this paper, we propose a real-to-real 3D data generation framework (R2RGen) that directly augments the pointcloud observation-action pairs to generate real-world data. R2RGen is simulator- and rendering-free, thus being efficient and plug-and-play. Specifically, we propose a unified three-stage framework, which (1) pre-processes source demonstrations under different camera setups in a shared 3D space with scene / trajectory parsing; (2) augments objects and robot's position with a group-wise backtracking strategy; (3) aligns the distribution of generated data with real-world 3D sensor using camera-aware post-processing. Empirically, R2RGen substantially enhances data efficiency on extensive experiments and demonstrates strong potential for scaling and application on mobile manipulation.
>
---
#### [replaced 073] ELIQ: A Label-Free Framework for Quality Assessment of Evolving AI-Generated Images
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2602.03558](https://arxiv.org/pdf/2602.03558)**

> **作者:** Xinyue Li; Zhiming Xu; Min Tang; Zhaolin Cai; Sijing Wu; Xiongkuo Min; Yitong Chen; Guangtao Zhai
>
> **摘要:** Generative text-to-image models are advancing at an unprecedented pace, continuously shifting the perceptual quality ceiling and rendering previously collected labels unreliable for newer generations. To address this, we present ELIQ, a Label-free Framework for Quality Assessment of Evolving AI-generated Images. Specifically, ELIQ focuses on visual quality and prompt-image alignment, automatically constructs positive and aspect-specific negative pairs to cover both conventional distortions and AIGC-specific distortion modes, enabling transferable supervision without human annotations. Building on these pairs, ELIQ adapts a pre-trained multimodal model into a quality-aware critic via instruction tuning and predicts two-dimensional quality using lightweight gated fusion and a Quality Query Transformer. Experiments across multiple benchmarks demonstrate that ELIQ consistently outperforms existing label-free methods, generalizes from AI-generated content (AIGC) to user-generated content (UGC) scenarios without modification, and paves the way for scalable and label-free quality assessment under continuously evolving generative models. The code will be released upon publication.
>
---
