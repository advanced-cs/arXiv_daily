# 计算机视觉 cs.CV

- **最新发布 164 篇**

- **更新 83 篇**

## 最新发布

#### [new 001] Cell Phantom Video Generation in Elliptical Fourier Descriptor Domain
- **分类: cs.CV**

- **简介: 该论文属于细胞追踪任务，旨在解决标注数据不足的问题。通过在椭圆傅里叶描述子域生成时间一致的细胞幻影视频，提升合成数据质量。**

- **链接: [https://arxiv.org/pdf/2605.22563](https://arxiv.org/pdf/2605.22563)**

> **作者:** Francesco Benedetto; Roberto Basla; Luca Magri; Giacomo Boracchi
>
> **备注:** 6 pages, Accepted at the International Conference on Image Processing (ICIP) 2026
>
> **摘要:** Training Deep Neural Networks for tracking individual cells in biomedical videos requires a large amount of annotated data. The annotation of videos for cell tracking is very time consuming and often requires domain expertise; this explains the limited availability of public annotated data to address important medical problems like tissue repair or cancer treatment. Generating synthetic videos along with their Ground Truth annotations is a promising solution that relies, as a foundational first step, on the synthesis of single cell annotations (or phantoms). Phantoms need to be time consistent, as they have to replicate biological processes that are specific to the cell types. In this work, we propose a novel framework for generating videos of cell phantoms in the Elliptical Fourier Descriptors (EFDs) domain, a compact and geometrically interpretable representation for 2D closed contours. We represent the cell phantom evolution as a multivariate time series of EFD coefficients, introducing a strong prior for cell morphology and enabling the efficient generation of sequences that evolve coherently in time. Our experimental validation proves that modelling the temporal evolution in EFD space enables the generation of biologically plausible phantom videos. Our method can be used in generative pipelines for synthesizing annotated data for cell tracking, thus strongly mitigating the annotation effort for creating new datasets. Our code is available for download here: this https URL.
>
---
#### [new 002] MotionDPS: Motion-Compensated 3D Brain MRI Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D脑部MRI重建任务，旨在解决运动伪影问题。通过联合估计图像、运动参数和线圈灵敏度，提升图像质量与运动鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.22121](https://arxiv.org/pdf/2605.22121)**

> **作者:** Antonio Ortiz-Gonzalez; Erich Kobler; Lukas Schletter; Alexander Effland
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Magnetic resonance imaging (MRI) is highly susceptible to patient motion due to its relatively long acquisition times and the fact that data are acquired sequentially in k-space. Even small patient movements introduce phase inconsistencies across measurements, leading to severe artifacts such as blurring, ghosting, and geometric distortions that can compromise diagnostic quality. Retrospective motion compensation remains challenging, particularly in accelerated acquisitions, due to the ill-posed nature of the joint reconstruction and motion estimation problem. In this work, we propose a unified Bayesian framework for motion-compensated 3D MRI that jointly estimates the anatomical image, rigid-body motion parameters, and coil sensitivity maps directly from motion-corrupted k-space data. Our approach integrates pretrained 3D complex-valued score-based diffusion models as expressive anatomical image priors within a physics-based forward model. Inference is performed by alternating diffusion posterior image updates with efficient proximal optimization steps for motion and coil sensitivity estimation, enabling fully unsupervised reconstruction without the need for paired motion-free training data. Experiments on simulated and real-motion brain MRI datasets demonstrate that the proposed method achieves improved image quality and motion robustness compared to state-of-the-art classical and learning-based motion correction techniques, particularly in the presence of severe motion and high acceleration.
>
---
#### [new 003] Event-Illumination Collaborative Low-light Image Enhancement with a High-resolution Real-world Dataset
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决事件信号噪声大和全局光照信息缺失的问题。提出EIC-LIE框架，融合事件与光照信息，提升增强效果。**

- **链接: [https://arxiv.org/pdf/2605.22186](https://arxiv.org/pdf/2605.22186)**

> **作者:** Senyan Xu; Zhijing Sun; Kean Liu; Xin Lu; Ruixuan Jiang; Mingyang Huang; Xueyang Fu; Zheng-Jun Zha
>
> **摘要:** Event-based low-light image enhancement (LIE) methods mainly focus on incorporating high dynamic range (HDR) information from events while overlooking the essential global illumination in images and the inherent noise sensitivity of event signals in real-world scenarios. To address these issues, we propose EIC-LIE, an event-illumination collaborative LIE framework. Concretely, we first design an Event-Illumination Collaborative Interaction (EICI) module, which contains two key processes: forward gathering, which gathers HDR features across varying lighting conditions, and backward injection, which provides complementary content for illumination and event representations. Next, we introduce an Illumination-aware Event Filter (IAEF) that dynamically reduces event noise based on brightness statistics derived from images. Additionally, we build a beam-splitter-based hybrid imaging system to collect high-quality event-image pairs with temporal synchronization from dynamic scenes, providing the first high-resolution, real-world event-based LIE dataset. Extensive experiments show that our EIC-LIE outperforms state-of-the-art methods on five real-world and synthetic datasets, significantly surpassing previous methods with improvements of up to 1.24dB in PSNR and 0.069 in SSIM. The code and dataset are released at this https URL.
>
---
#### [new 004] Virtual 3D H&E Staining from Phase-contrast Back-illumination Interference Tomography
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于虚拟染色任务，旨在将BIT成像数据转换为H&E图像。解决shift-variant对比度和缺乏验证基准的问题，提出新框架提升结构保真度和分割精度。**

- **链接: [https://arxiv.org/pdf/2605.22000](https://arxiv.org/pdf/2605.22000)**

> **作者:** Anthony Song; Boyan Zhou; Mayank Golhar; Marisa Morakis; Alex Baras; Nicholas Durr
>
> **摘要:** Three-dimensional (3D) histopathology of unprocessed tissues has the potential to transform disease management by enabling volumetric characterization of tissue microarchitecture and in-vivo assessment. Back-illumination Interference Tomography (BIT) is a new phase microscopy technology that provides rapid, non-destructive volumetric imaging of unprocessed tissues. However, translating BIT volumes into clinically interpretable H&E images remains challenging, particularly due to shift-variant contrast and the absence of quantitative validation benchmarks. We introduce HistoBIT3D, the first voxel-wise paired BIT and fluorescence-labeled nuclei dataset, enabling quantitative evaluation of structural preservation in unsupervised virtual staining against ground-truth nuclear distributions. Using this dataset, we present a novel virtual staining framework that translates BIT volumes with shift-variant contrast into realistic H&E volumes by leveraging bidirectional multiscale content consistency and cross-domain style reuse to enhance structural fidelity and perceptual realism. Our method achieves state-of-the-art realism metrics while significantly improving 3D nuclei segmentation accuracy and boundary preservation under zero-shot Cellpose evaluation. Together, these contributions establish a quantitatively validated, structurally faithful, and scalable pipeline for 3D virtual H&E staining, advancing the paradigm of slide-free, volumetric computational histopathology. Our data and code are available at: this https URL.
>
---
#### [new 005] Exposing Vulnerabilities in Visible-Infrared VLMs: A Unified Geometric Adversarial Framework with Cross-Task Transferability
- **分类: cs.CV**

- **简介: 该论文属于可见-红外视觉语言模型安全研究，旨在解决其对抗鲁棒性不足的问题。提出CFGPatch框架，通过几何对抗补丁攻击提升模型脆弱性。**

- **链接: [https://arxiv.org/pdf/2605.22273](https://arxiv.org/pdf/2605.22273)**

> **作者:** Xiang Chen; Yuxian Dong; Chao Li; Chengyin Hu; Jiaju Han; Fengyu Zhang; Yiwei Wei; Jiahuan Long; Jiujiang Guo
>
> **摘要:** Vision-language models (VLMs) have achieved strong performance across diverse multimodal tasks, but their adversarial robustness in visible-infrared (VIS-IR) scenarios remains underexplored. This gap is critical because VIS-IR sensing is widely used in real-world perception systems to support reliable understanding under challenging imaging conditions. To address this cross-modal threat setting, we propose CFGPatch, a curved-edge fractal geometric adversarial patch framework for attacking VIS-IR VLMs. CFGPatch builds on triangular fractal geometry and replaces rigid straight-edged primitives with Bezier-curved elements, preserving multi-scale fractal self-similarity while introducing smoother contours, richer directional variation, and more flexible shape deformation. In addition, we design a modality-specific Fraser-spiral rendering mechanism to inject fine-grained texture distortions and misleading perceptual cues into visible and infrared images. By coupling global curved-fractal geometry with local spiral-based appearance interference, CFGPatch disrupts both shape perception and texture interpretation. We further adopt expectation over transformation (EOT) to improve robustness against common image-level transformations. Extensive experiments show that CFGPatch effectively fools VIS-IR VLMs and consistently outperforms standard patch baselines in attack effectiveness and robustness. Moreover, adversarial samples optimized for zero-shot classification transfer well to image captioning and visual question answering, demonstrating strong cross-task transferability and generalizability across downstream tasks.
>
---
#### [new 006] Rethinking Noise-Robust Training for Frozen Vision Foundation Models: A Cross-Dataset Benchmark with a Case Study of Small-Loss Failure
- **分类: cs.CV**

- **简介: 该论文研究医疗图像中冻结视觉基础模型的噪声鲁棒训练问题，通过基准测试分析不同方法在噪声数据下的表现，提出方法选择应考虑具体场景。**

- **链接: [https://arxiv.org/pdf/2605.22591](https://arxiv.org/pdf/2605.22591)**

> **作者:** Zitong Li; Haoyu Wang
>
> **摘要:** Frozen Vision Foundation Models (VFMs) with lightweight classification heads are increasingly used in medical imaging because they offer efficient and reproducible deployment. Yet noisy-label learning methods for this frozen-feature regime remain poorly understood, and most existing methods still rely on a small-loss assumption inherited from end-to-end training. We present a controlled benchmark of eight noisy-label methods across five medical datasets, three backbones, two noise types, and five noise rates (150 conditions, 6,000 training runs), evaluated with balanced accuracy. The benchmark shows that there is no universal winner: Friedman ranking over the 150 conditions yields $\chi^2 = 333.2$ ($p = 4.77 \times 10^{-68}$), ELR wins the most conditions (49/150), while CUFIT attains the best mean rank (2.51). The practical cost of method choice grows sharply with noise severity, from 4.5pp on clean data to 18.8pp at asymmetric 40\% noise. To explain these benchmark-level patterns, we revisit the small-loss assumption in a representative high-risk regime. Under frozen DINOv2 features, clean and noisy loss distributions overlap by 53--61\%, and matched-rate clean-sample detection shows that prediction agreement is markedly more stable than loss ranking under asymmetric noise (3pp vs.\ 13pp precision drop). On ISIC2019 with asymmetric 40\% noise, Co-Teaching reaches 68\% overall accuracy while collapsing to 35.1\% balanced accuracy with zero recall on three minority classes. Together, these results recast noisy-label learning for frozen VFMs as a regime-aware method-selection problem rather than a search for a single dominant algorithm. We conclude with evidence-based guidance and a low-regret feature-space selector for practical recommendation.
>
---
#### [new 007] VEELA: A Clinically-Constrained Benchmark for Liver Vessel Segmentation in Computed Tomography Angiography
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于肝脏血管分割任务，旨在解决CTA中血管分割的挑战。通过构建VEELA数据集并设计多维度评估框架，提升分割的准确性与临床实用性。**

- **链接: [https://arxiv.org/pdf/2605.22357](https://arxiv.org/pdf/2605.22357)**

> **作者:** Ziya Ata Yazıcı; N. Sinem Gezer; İlkay Öksüz; İlker Özgür Koska; Tuğçe Toprak; Pervin Bulucu; Ufuk Beşenk; A. Emre Kavur; Pierre-Henri Conze; Hazım Kemal Ekenel; Oğuz Dicle; Mustafa Ege Şeker; Mustafa Said Kartal; Ariorad Moniri; Orhan Özkan; Osman Faruk Bayram; Hakan Polat; Musa Balcı; Ece Tuğba Cebeci; Baran Cılga; Kardelen Peçenek; M. Alper Selver
>
> **备注:** 27 pages, 25 figures, 5 tables
>
> **摘要:** Accurate segmentation of hepatic and portal vessels in contrast-enhanced computed tomography angiography (CTA) remains challenging due to complex vascular topology, peripheral visibility limitations, and acquisition-induced ambiguities. While existing public datasets offer valuable benchmarks, few include clinically realistic annotation constraints. We introduce VEELA (Vessel Extraction and Extrication for Liver Analysis), a rigorously curated liver vessel dataset derived from 40 CTA scans inherited from the CHAOS grand-challenge cohort. All vessels were manually delineated slice-by-slice under multi-expert consensus, using a strict visibility-driven annotation policy and avoiding anatomically inferred interpolation. This design explicitly captures anatomical variability and imaging-related uncertainty. As a continuation of the CHAOS challenge, VEELA enables reproducible cross-benchmark evaluation while extending the scope to fine-grained hepatic and portal vessel segmentation. We further establish a standardized benchmarking framework and analyze complementary evaluation metrics, including topology-aware (clDice), overlap-based (IoU), boundary-sensitive (NSD), and geometry-aware (area, length) measures. Our results demonstrate that different metrics capture distinct aspects of vascular integrity, underscoring the necessity of multi-perspective evaluation for clinically meaningful vessel segmentation. VEELA is publicly released to facilitate reproducible research and support the development of robust vascular segmentation methods. Researchers can access the evaluation metrics, dataset, and submission platform at this https URL.
>
---
#### [new 008] Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于遥感图像超分辨率任务，解决传统方法效率低、灵活性差的问题。提出FlowGS框架，通过概率流和高斯点云实现高效、灵活的图像重建。**

- **链接: [https://arxiv.org/pdf/2605.22147](https://arxiv.org/pdf/2605.22147)**

> **作者:** Jiangwei Mo; Xi Lu; Hanlin Wu
>
> **摘要:** High-resolution remote sensing images (RSIs) are crucial for Earth observation applications, yet acquiring them is often limited by sensor constraints and costs. In recent years, generative super-resolution (SR) methods, particularly diffusion models, have made significant progress. However, they typically require slow iterative inference with 40--1000 steps and exhibit limited flexibility in continuous-scale SR settings. To address these issues, we propose FlowGS, a generative reconstruction framework for arbitrary-scale SR of RSIs. FlowGS models the high-frequency detail representations between high- and low-resolution images and learns a continuous probability flow from noise to detail priors via flow matching (FM) constrained by shortcut consistency, thereby reducing generative complexity and improving inference efficiency. Additionally, we employ 2D Gaussian splatting to construct a continuous feature field, thereby enabling flexible reconstruction at arbitrary query locations. Experimental results show that FlowGS delivers competitive perceptual quality compared with existing methods in both continuous-scale and fixed-scale SR settings, with substantially improved inference efficiency.
>
---
#### [new 009] EventGait: Towards Robust Gait Recognition with Event Streams
- **分类: cs.CV**

- **简介: 该论文属于行为识别任务，旨在解决传统相机在复杂环境下gait识别效果差的问题。通过事件相机和双流框架EventGait，提升识别鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.22139](https://arxiv.org/pdf/2605.22139)**

> **作者:** Senyan Xu; Shuai Chen; Chuanfu Shen; Kean Liu; Zhijing Sun; Chengzhi Cao; Xueyang Fu
>
> **摘要:** Gait recognition enables non-intrusive, privacy-preserving identification but suffers in uncontrolled environments due to illumination and motion sensitivity of conventional cameras. In this work, we explore gait recognition using event cameras, which offer microsecond temporal resolution and high dynamic range, naturally capturing robust dynamic cues and suppressing static noise. Existing event-based approaches typically aggregate event streams into event images over long time windows, thereby discarding fine-grained motion dynamics critical for gait recognition. Therefore, we propose \textbf{EventGait}, an end-to-end dual-stream framework that separately models motion and shape while preserving the advantages of events. Our dynamic stream leverages a Mixture of Spiking Experts (MoSE) with diverse neuron constants for robust dynamic perception across complex motion and illumination scenes, while the static stream learns dense shape representations via Cross-modal Structure Alignment (CroSA) with large vision foundation models. To address the absence of large-scale event-based gait datasets, we introduce a synthesis pipeline and release two new benchmarks: SUSTech1K-E and CCGR-Mini-E. Extensive experiments have shown that event-based gait recognition not only achieves results comparable to camera-based gait recognition under normal conditions but also significantly outperforms it in low-light scenarios. Our approach sets a new state of the art on both synthesized and real-world event-based gait benchmarks, highlighting the robustness and potential of event-driven gait analysis. The code and datasets are released at this https URL.
>
---
#### [new 010] Direct content-based retrieval from music scores images
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于音乐评分内容检索任务，旨在解决传统依赖元数据搜索的问题。工作包括分析评分特征、构建查询数据集，并测试多种基于内容的检索方法。**

- **链接: [https://arxiv.org/pdf/2605.22255](https://arxiv.org/pdf/2605.22255)**

> **作者:** Noelia Luna-Barahona; Antonio Ríos-Vila; David Rizo; Jorge Calvo-Zaragoza
>
> **备注:** 17 pages (14 pages + references), 3 figures (with subfigures)
>
> **摘要:** The digitization of musical scores plays a crucial role in their preservation and accessibility, yet information retrieval still depends mainly on metadata searches, such as by title or composer. Content based search in music score images remains underexplored compared to text documents, despite its potential value for musicians, musicologists, and educators. This work contributes to the field by first studying which characteristics of a score are most relevant for search and by defining a systematic method to build query datasets from any annotated corpus. We also consider diverse methods for content-based search on music score images, ranging from transcription-based approaches relying on Optical Music Recognition (OMR), to a transcription-free Transformer model trained to recognize queries directly from score images, and a text-prompted Large Language Model. Our experiments evaluate these models on four corpora exhibiting diverse characteristics in terms of dataset size, image quality, and typesetting mechanisms. Overall, each method excels under different conditions: OMR-based pipelines achieve higher in-domain retrieval, whereas transcription-free models handle domain variability more effectively.
>
---
#### [new 011] ORBIS: Output-Guided Token Reduction with Distribution-Aware Matching for Video Diffusion Acceleration
- **分类: cs.CV; cs.AR**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散模型计算成本高的问题。通过提出ORBIS框架，结合输出引导的token缩减和分布感知匹配算法，显著提升加速效果。**

- **链接: [https://arxiv.org/pdf/2605.22015](https://arxiv.org/pdf/2605.22015)**

> **作者:** Hangyeol Lee; Joo-Young Kim
>
> **摘要:** Diffusion Transformer (DiT) has emerged as a powerful model architecture for generating high-quality images and videos. In the case of video DiT, 3D Spatio-Temporal Attention increases token length in proportion to the number of frames, sharply increasing computational cost. Token reduction methods mitigate this cost by exploiting spatial redundancy, but existing approaches rely on inaccurate similarity estimates and lightweight matching algorithms, resulting in poor matching quality and only marginal acceleration. To overcome these limitations, we propose ORBIS, an SW-HW co-designed accelerator for video DiT. ORBIS leverages the output activation from the previous timestep to obtain more accurate inter-token similarity, substantially improving matching quality and enabling a higher token reduction ratio. We further introduce a Distribution-Aware Token Matching (DATM) algorithm that captures global token distribution and explicitly minimizes token-pair loss for additional gains. To fully hide DATM latency, we design specialized, deeply pipelined hardware and minimize its hardware cost through quantization, occupying only 2.4% of total area with negligible accuracy loss. Extensive experiments show that ORBIS achieves about 2x higher token reduction ratio than the state-of-the-art approach, AsymRnR, while delivering up to 4.5x speedup and 79.3% energy reduction compared to an NVIDIA A100 GPU.
>
---
#### [new 012] GenHAR: Generalizing Cross-domain Human Activity Recognition for Last-mile Delivery
- **分类: cs.CV**

- **简介: 该论文属于人类活动识别任务，旨在解决跨领域数据分布差异导致的性能下降问题。提出GenHAR框架，通过学习领域不变特征提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.22086](https://arxiv.org/pdf/2605.22086)**

> **作者:** Zhiqing Hong; Zelong Li; Xiubin Fan; Guang Yang; Baoshen Guo; Haotian Wang; Tian He; Desheng Zhang
>
> **摘要:** Human Activity Recognition (HAR) has shown remarkable effectiveness in various applications, such as smart healthcare and intelligent manufacturing. However, a major challenge faced by HAR is the distribution shift across different sensor data domains, which often leads to decreased performance when deployed for real-world applications. To address this issue, this paper introduces GenHAR, a novel framework designed to mitigate the domain gap by learning domain-invariant sensor representations. GenHAR aims to enhance the generalization capabilities of HAR on target domains purely with data from the source domain. The key novelty of GenHAR lies in two aspects. Firstly, GenHAR tokenizes sensor data and learns correlations among frequency sensor channel dimensions to improve the robustness of HAR models. Secondly, GenHAR improves the efficiency via selective masking and an efficient attention mechanism. We conduct a systematic analysis of GenHAR by comparing it with state-of-the-art HAR methods on real-world human activity datasets. Results show that GenHAR outperforms state-of-the-art methods by 9.97% in accuracy, and reduces Floating Point Operations by 6.4 times. Moreover, we deploy GenHAR at a leading logistics company in 4 cities, and have detected 2.15 billion real-time activities. We release our code at: this https URL.
>
---
#### [new 013] GA-VLN: Geometry-Aware BEV Representation for Efficient Vision-Language Navigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言导航任务，旨在解决现有方法依赖密集视频导致计算冗余和空间推理不足的问题。通过引入GA-BEV特征表示，提升导航效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.22036](https://arxiv.org/pdf/2605.22036)**

> **作者:** Jiahao Yang; Zihan Wang; Xiangyang Li; Xing Zhu; Yujun Shen; Yinghao Xu; Shuqiang Jiang
>
> **摘要:** Despite significant progress in Vision-Language Navigation (VLN), existing approaches still rely on dense RGB videos that produce excessive patch tokens and lack explicit spatial structure, resulting in substantial computational overhead and limited spatial reasoning. To address these issues, we introduce the Geometry-Aware BEV (GA-BEV) - a compact, 3D-grounded feature representation that integrates both explicit and implicit geometric cues into multimodal large language model (MLLM) - based navigation systems. We construct BEV spatial maps from RGB-D inputs by projecting visual features into 3D space and aggregating them into an agent-centric layout that preserves geometric consistency while reducing token redundancy. To further enrich geometric understanding, we incorporate features from a pretrained 3D foundation model into the BEV space, injecting structural priors learned from large-scale 3D reconstruction tasks. Together, these complementary cues - explicit depth-based projection and implicit learned priors - yield compact yet spatially expressive representations that substantially improve navigation efficiency and performance. Experiments show that our method achieves state-of-the-art results using only navigation data, without DAgger augmentation or mixed VQA training, demonstrating the robustness and data efficiency of the proposed GA-VLN framework.
>
---
#### [new 014] Echo4DIR: 4D Implicit Heart Reconstruction from 2D Echocardiography Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于4D心脏重建任务，解决2D超声视频重建4D心脏几何的难题。通过引入隐式表示和注意力机制，实现高精度、连续的4D心脏模型重建。**

- **链接: [https://arxiv.org/pdf/2605.22066](https://arxiv.org/pdf/2605.22066)**

> **作者:** Yanan Liu; Qinya Li; Hao Zhang; Kangjian He; Xuan Yang; Hao Li; Dan Xu; Lei Li
>
> **摘要:** Reconstructing 4D (3D+t) cardiac geometry from sparse 2D echocardiography is highly desirable yet fundamentally challenged by geometric ambiguity and temporal discontinuity. To tackle these issues, we propose Echo4DIR, a novel test-time 4D implicit reconstruction framework. Specifically, we learn robust 3D shape priors from statistical shape models (SSMs) via a cardiac conditional SDF, constructing an Epipolar Mask Encoder module with epipolar cross attention to effectively fuse multi-view features. To bridge the synthetic-to-real domain gap, we introduce a self-supervised SDF-tailored differentiable rendering strategy for patient-specific 3D shape adaptation using uncalibrated clinical masks without requiring 3D ground truth. Crucially, the inherent continuity of implicit representation overcomes sparse observations, enabling anatomically reliable geometry at arbitrary resolutions. Furthermore, to empower our framework with physically continuous 4D extension, we introduce a Radial SDF Alignment strategy that strictly locks shape evolution to the predicted velocity field, fundamentally eliminating mesh drift. Extensive experiments on synthetic benchmarks and real clinical datasets demonstrate that Echo4DIR achieves state-of-the-art 4D cardiac mesh reconstruction, notably yielding an impressive clinical overlap of up to 98.35% Dice and 96.75% IoU.
>
---
#### [new 015] BodyReLux: Temporally Consistent Full-Body Video Relighting
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出BodyReLux，解决人体视频动态光照重渲染问题。通过混合数据集和光照条件控制，实现高质量、时间一致的视频重渲染。**

- **链接: [https://arxiv.org/pdf/2605.21766](https://arxiv.org/pdf/2605.21766)**

> **作者:** Li Ma; Mingming He; Xueming Yu; David M. George; Ahmet Levent Taşel; Paul Debevec; Julien Philip
>
> **备注:** Siggraph 2026 Journal Track. Project page: this https URL
>
> **摘要:** Being able to relight human performance is a fundamental task for post production and content creation. We present BodyReLux, a subject-specific video diffusion-based framework for relighting full-body human performances in a temporally consistent way. Our model is trained on a hybrid dataset of pixel-aligned video relighting pairs, covering a diverse combination of lighting conditions, performances and viewpoints. To acquire such dataset, we combine traditional static One-Light-at-a-Time (OLAT) capture and a novel dynamic performance capture in which two smoothly varying lighting sequences are rapidly interleaved. Because the lighting operates above the human flicker-fusion threshold, the interleaving does not appear to strobe. We train our video relighting model from a pretrained text-to-video model to fully leverage the generative priors for producing high quality videos. To achieve accurate lighting control, we introduce a new lighting conditioning method that represents each light source as a token. We further condition on sequences of lighting using masked attention to support dynamic lighting control. Together with a carefully designed data augmentation pipeline, we achieve photorealistic, robust, and temporally consistent video relighting of subject-specific human performances.
>
---
#### [new 016] RiT: Vanilla Diffusion Transformers Suffice in Representation Space
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型在表示空间中的学习效率问题。通过分析不同特征空间的统计特性，提出RiT模型，利用DINOv2特征提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.21981](https://arxiv.org/pdf/2605.21981)**

> **作者:** Le Zhang; Ning Mang; Aishwarya Agrawal
>
> **摘要:** Flow matching with $x$-prediction -- regressing the clean data point rather than the ambient velocity -- is known to exploit low-dimensional manifold structure effectively in pixel space \cite{li2025back}. We ask whether a pretrained representation space, while containing a low-dimensional data manifold of comparable intrinsic dimensionality, offers a distribution more favorable for flow-matching learning. Comparing pixel, SD-VAE, and DINOv2 features along four geometric axes, we find that pixel and DINOv2 share nearly identical intrinsic dimensionalities (both $\hat{d}\!\approx\!33$) yet DINOv2 exhibits $7.3\times$ higher effective rank, $35\times$ better covariance conditioning, $11.5\times$ lower excess kurtosis, and $1.7\times$ lower on-manifold interpolation error; SD-VAE latents are consistently intermediate, indicating that the advantage stems from representation-learning objectives rather than mere compression. These statistical properties render the flow-matching regression well-conditioned and remove the need for the specialized prediction heads or Riemannian transport used by prior DINOv2 diffusion methods. We propose the \emph{Representation Image Transformer} (RiT): a vanilla Diffusion Transformer trained by $x$-prediction on frozen DINOv2 features, augmented only by a dimension-aware noise schedule and joint \texttt{[CLS]}-patch modeling. On ImageNet $256{\times}256$, RiT attains FID 1.45 without guidance and 1.14 with classifier-free guidance, outperforming DiT$^\text{DH}$-XL with $19\%$ fewer parameters (676M vs.\ 839M). The resulting ODE is efficiently solvable at coarse discretizations: with classifier-free guidance, $5$ Heun steps already reach FID 2.0 and $10$ steps reach 1.25, without distillation or consistency training. Code at this https URL.
>
---
#### [new 017] Bounding-Box Trajectories Matter for Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在解决因场景变化导致的检测困难问题。通过引入边界框轨迹，提出TrajVAD框架，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.21957](https://arxiv.org/pdf/2605.21957)**

> **作者:** Inpyo Song; Jangwon Lee
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** Video anomaly detection is critical for public safety and security, yet remains highly challenging despite extensive research due to large variations in appearance, viewpoint, and scene dynamics. Among existing approaches, human pose-based methods have emerged as a major line of research, showing strong performance since many anomalies in public datasets involve humans and pose representations are robust to appearance changes while providing compact motion descriptions. However, these methods often overlook bounding-box trajectories, although such information is inherently available in pose-based pipelines. In this paper, we explicitly leverage these trajectories as a primary anomaly cue. We present TrajVAD, a framework that models multi-class bounding-box trajectories using normalizing flows to learn normal kinematic patterns. Its trajectory-only variant (TrajVAD-T) eliminates pose estimation and surpasses all compared pose-based methods on ShanghaiTech in AP (87.7%), while achieving the best results on MSAD. An extended version (TrajVAD-P) incorporates pose information and further improves performance to 88.6% AUROC and 90.9% AP on ShanghaiTech, highlighting bounding-box trajectories as an effective yet underexplored modality for video anomaly detection.
>
---
#### [new 018] H-Flow: Self-supervised Human Scene Flow via Physics-inspired Joint Multi-modal Learning
- **分类: cs.CV**

- **简介: 该论文提出H-Flow，解决人体场景流估计问题，结合骨骼运动与表面变形，利用物理先验进行自监督学习。**

- **链接: [https://arxiv.org/pdf/2605.22629](https://arxiv.org/pdf/2605.22629)**

> **作者:** Zhanbo Huang; Xiaoming Liu; Yu Kong
>
> **备注:** 19 pages, 7 figures, 4 tables
>
> **摘要:** Parametric human models capture global pose but cannot represent the non-rigid surface dynamics of clothing and soft tissue. Generic scene flow estimates dense motion but breaks down on articulated bodies, where pixel-level supervision is also intractable to acquire. We introduce H-Flow, a dense human scene flow that captures both skeletal kinematics and surface deformation. A unified multi-head transformer estimates flow from monocular video, jointly predicting pose and depth as companion outputs. The challenge lies in the lack of supervision. In place of unattainable labels, we anchor the network in the physics of human motion, encoding geometric, structural, and biomechanical priors as cross-modal training objectives. We further introduce DynAct4D, a high-fidelity synthetic benchmark providing dense flow annotations across diverse subjects, garments, and motions. On standard benchmarks, H-Flow outperforms scene-flow and parametric baselines, and generalizes zero-shot to in-the-wild video. Code, models, and the DynAct4D benchmark will be released upon publication
>
---
#### [new 019] SceneAligner: 3D-Grounded Floorplan Localization in the Wild
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于场景定位任务，旨在解决在复杂环境中根据图像定位到建筑平面图的问题。通过构建3D场景并生成密度图，实现与平面图的对齐，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.22581](https://arxiv.org/pdf/2605.22581)**

> **作者:** Junhyeong Cho; Ruojin Cai; Hadar Averbuch-Elor
>
> **备注:** Project Page: this https URL
>
> **摘要:** Many public buildings provide floorplans with a "you are here" indicator to help visitors orient themselves. Floorplan localization seeks to computationally replicate this capability by determining where visual observations were captured within a floorplan. However, existing methods typically assume controlled small-scale environments and precise vectorized floorplans, limiting their ability to operate in large-scale buildings and rasterized floorplans. In this work, we present an approach for performing floorplan localization in the wild by grounding the task in a reconstructed 3D representation of the scene. Given an unconstrained image collection, our method reconstructs a gravity-aligned 3D scene and projects it into a 2D density map that serves as a floorplan proxy. Floorplan localization is then formulated as aligning this proxy with the input floorplan via a 2D similarity transform. To bridge the appearance gap between density maps and architectural floorplans, we adapt a 2D foundation model to learn cross-modal correspondences, introducing a fine-tuning scheme that encourages semantically aligned matches while preserving structural consistency. Extensive experiments demonstrate substantial improvements over prior methods, including in extremely sparse settings with as little as a single input image. Our code and data will be publicly available.
>
---
#### [new 020] Slimmable ConvNeXt: Width-Adaptive Inference for Efficient Multi-Device Deployment
- **分类: cs.CV**

- **简介: 该论文提出Slimmable ConvNeXt，解决多设备高效部署问题，通过宽度自适应推理实现单一模型多性能配置，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.22677](https://arxiv.org/pdf/2605.22677)**

> **作者:** Janek Haberer; Jon Eike Wilhelm; Olaf Landsiedel
>
> **备注:** Accepted at Mobile AI Workshop 2026 (CVPR'26 Workshop)
>
> **摘要:** Deploying vision models across devices with varying resource constraints, or even on a single device where available compute fluctuates due to battery state, thermal throttling, or latency deadlines, typically requires training and maintaining separate models. Width-adaptive inference addresses this by training a single set of shared weights containing multiple nested subnetworks of increasing capacity, but prior CNN-based approaches required switchable batch normalization, while recent scalable methods have focused on Vision Transformers. We present Slimmable ConvNeXt, which shows that ConvNeXt's modern design, specifically LayerNorm and inverted bottlenecks, makes it particularly suited for channel-width slimming, eliminating the normalization overhead of classical slimmable networks and producing a simpler training pipeline than both prior CNN and ViT approaches. On ImageNet-1k, Slimmable ConvNeXt-T with 3 subnetworks achieves 80.8% top-1 accuracy at 4.5 GMACs and 77.4% at 1.2 GMACs, trained from scratch for 600 epochs. At comparable compute, this exceeds HydraViT's 6-head subnetwork (78.4% at 4.6 GMACs) by 2.4 percentage points and its 3-head configuration (73.0% at 1.3 GMACs) by 4.4 percentage points, while also outperforming MatFormer-S (78.6%) and SortedNet-S (78.2%) at the same GMACs. Scaling to Slimmable ConvNeXt-B further improves maximum accuracy to 82.8% at 15.35 GMACs.
>
---
#### [new 021] Improving Viewpoint-Invariance and Temporal Consistency for Action Detection
- **分类: cs.CV**

- **简介: 该论文属于动作检测任务，旨在解决视角不变性和时间一致性问题。通过两阶段方法，提取运动特征并构建多尺度时序编码器，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.22695](https://arxiv.org/pdf/2605.22695)**

> **作者:** Yannick Porto; Renato Martins; Thomas Chalumeau; Cedric Demonceaux
>
> **备注:** Accepted at ICIP 2026. Code and trained models are available at: this https URL
>
> **摘要:** Viewpoint change invariance and action temporal consistency are critical aspects for the effective deployment of human action detection of untrimmed videos. Existing appearance-based video detection methods often struggle with limited viewpoint diversity during training, while motion-based detection approaches frequently fail to model fine-grained temporal relationships across consecutive motion windows. This paper introduces a novel two-stage action detection approach designed to improve both view-invariance and global temporal coherence properties. In the first stage, we extract motion features from augmented virtual viewpoints, solely used at training. Then, the second stage introduces a new view-invariant, multi-scale temporal encoder based on selective state-space sequence modelling to aggregate information across viewpoints and time scales. Experiments on PKU-MMD and BABEL benchmarks demonstrate that this approach significantly outperforms state-of-the-art methods in all considered splits. Code and trained models are available at: this https URL
>
---
#### [new 022] Case-Aware Medical Image Classification with Multimodal Knowledge Graphs and Reliability-Guided Refinement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗图像分类任务，旨在解决现有方法依赖单一视觉证据、无法有效利用相似病例和外部知识的问题。通过多模态知识图谱和可靠性引导优化，提升分类效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.22547](https://arxiv.org/pdf/2605.22547)**

> **作者:** Yiming Xu; Yixuan Liu; Yuhang Zhang; Ling Zheng; Yihan Wang; Qi Song
>
> **摘要:** Deep learning has brought significant progress to medical image classification, yet most existing methods still rely on isolated visual evidence and cannot effectively leverage similar cases or external knowledge. In clinical practice, diagnosis is typically supported by historical similar cases and their associated symptoms. To simulate this diagnostic process, we propose a framework that performs case-aware reasoning using multimodal knowledge graphs for explainable medical image diagnosis. Given an input image, our method constructs a multimodal knowledge graph from adaptively retrieved similar cases, enabling more effective utilization of related samples. We further introduce a knowledge propagation and injection mechanism, where an image-centric Graph Attention Network propagates knowledge semantics to obtain case-based features, followed by a bidirectional cross-modal attention mechanism that injects these features into visual representations for cross-modal alignment. To mitigate noisy retrieval, we design a confidence-calibrated decision refinement scheme that estimates the reliability of each retrieved case by jointly considering prediction confidence and sample similarity, adaptively adjusting its contribution to the final prediction and providing interpretable case-level evidence. Extensive experiments on multiple medical imaging datasets show that our approach consistently outperforms strong baselines, and ablation studies validate the effectiveness of each component. The source code is publicly available at this https URL.
>
---
#### [new 023] Beyond Chamfer Distance: Granular Order-aware Evaluation Metric For Online Mapping
- **分类: cs.CV**

- **简介: 该论文针对在线地图估计任务，解决现有评估方法对点序敏感性不足和粒度不够的问题。提出SOSPA和PLD两个新指标，提升评估精度与分析能力。**

- **链接: [https://arxiv.org/pdf/2605.22578](https://arxiv.org/pdf/2605.22578)**

> **作者:** Chouaib Bencheikh Lehocine; Adam Lilja; Junsheng Fu; Lars Hammarstrand
>
> **摘要:** Online map estimation is a crucial component of autonomous driving systems that reduces the reliance on costly high-definition maps. State-of-the-art (SOTA) methods commonly predict map elements as ordered sequences of points that form polylines and polygons. The evaluation of these methods relies predominantly on mean average precision (mAP) based on thresholded Chamfer distance (CD). This framework lacks sensitivity to point ordering and provides limited granularity in assessing geometric quality, making it difficult to distinguish which methods truly excel over others. In this work, we address these limitations on two fronts. For the single-instance similarity measure, we introduce sequence optimal sub-pattern assignment (SOSPA), an order-aware metric that enables fine-grained evaluation of individual geometries while satisfying all metric axioms. For the multi-instance evaluation framework, we propose polyline localisation and detection (PLD), a soft metric that jointly captures detection quality and geometric accuracy, replacing the hard thresholding of mAP with a principled soft assignment. Through evaluations on nuScenes, we demonstrate that PLD effectively ranks SOTA online mapping methods (MapTRv2, StreamMapNet, MapTracker) while providing a decomposed error analysis. This analysis identifies detection capability as the dominant bottleneck in current methods, revealing a performance trend that mAP fails to capture. Code for evaluation using our metrics will be released.
>
---
#### [new 024] Conceptualizing Embeddings: Sparse Disentanglement for Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型解释任务，旨在解决嵌入表示语义纠缠问题。提出CEDAR方法，在不增加维度的情况下实现语义解耦，提升可解释性。**

- **链接: [https://arxiv.org/pdf/2605.22679](https://arxiv.org/pdf/2605.22679)**

> **作者:** Piotr Kubaty; Patryk Marszałek; Łukasz Struski; Adam Wróbel; Jacek Tabor; Marek Śmieja
>
> **摘要:** Vision-language models learn powerful multimodal embeddings, yet their internal semantics remain opaque. While sparse autoencoders (SAEs) can extract interpretable features, they rely on expanding the representation dimension, which compromises the original geometry and introduces redundancy. We introduce CEDAR (Conceptual Embedding Disentanglement via Adaptive Rotation), a post-hoc method that reveals the compositional structure of pretrained embeddings without increasing dimensionality. By learning an invertible transformation with a top-$k$ sparsity bottleneck, CEDAR concentrates semantic information into axis-aligned disentangled coordinates. In CLIP-like architecture, individual coordinates can be interpreted with textual concepts, while for generative models such as BLIP, they can be decoded into natural language descriptions. Experiments demonstrate that CEDAR achieves a competitive reconstruction-sparsity trade-off while producing explanations that are more interpretable and better aligned with human perception. Our results suggest that the apparent entanglement in vision-language representations can be resolved through a suitable change of basis, eliminating the need for overcomplete expansions.
>
---
#### [new 025] Distributed Image Compression with Multimodal Side Information at Extremely Low Bitrates
- **分类: cs.CV**

- **简介: 该论文属于分布式图像压缩任务，旨在解决低比特率下重建质量差的问题。通过引入多模态侧信息，提升细节和全局感知质量。**

- **链接: [https://arxiv.org/pdf/2605.22061](https://arxiv.org/pdf/2605.22061)**

> **作者:** Guojun Xu; Mingyang Zhang; Jianwen Xiang; Cheng Tan; Yanchao Yang; Junwei Zhou
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Distributed Image Compression (DIC) is crucial for multi-view transmission, especially when operating at extremely low bitrates (< 0.1 bpp). Its core challenge is effectively utilizing side information to achieve high-quality reconstruction under strict bitrate budgets. However, existing DIC approaches struggle to exploit global context and object-level details from side information, leading to local blurring and the loss of fine details in the reconstruction. To address these limitations, we propose a Multimodal DIC framework (MDIC), which, for the first time, leverages side information in a multimodal manner into the DIC paradigm, effectively preserving fine-grained local details and enhancing global perceptual quality in reconstructed images. Specifically, we introduce a text-to-image diffusion-based decoder conditioned on textual side information extracted from correlated images to capture shared global semantics. Moreover, we design a feature-mask generator, supervised by a multimodal fine-grained alignment task, to strengthen the exploitation of visual side information. The generated mask serves two purposes: first, it guides the extraction of fine-grained details from losslessly transmitted side information to preserve the semantic consistency of reconstructed details; second, it regulates the extraction of clustered feature representations from the quantized VQ-VAE embeddings, compensating for category information lost under the extreme compression of the primary image. Extensive experiments on the widely used KITTI Stereo and Cityscapes datasets demonstrate that MDIC achieves state-of-the-art perceptual quality at extremely low bitrates.
>
---
#### [new 026] D3Seg: Dependency-Aware Diffusion for Brain Tumor Segmentation with Missing Modalities
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决MRI模态缺失下的脑肿瘤分割问题。提出D3Seg模型，通过多跳模态图融合、扩散补全和概率决策优化提升分割性能。**

- **链接: [https://arxiv.org/pdf/2605.22249](https://arxiv.org/pdf/2605.22249)**

> **作者:** Danish Ali; Ajmal Mian; Naveed Akhtar; Ghulam Mubashar Hassan
>
> **摘要:** Accurate brain tumor segmentation using multiparametric MRI is critical for effective treatment planning. However, in clinical settings, complete acquisition of all MRI sequences is not always possible. The absence of certain MRI modalities results in substantial performance degradation in existing segmentation methods, which typically rely on naive feature concatenation or direct fusion strategies. To address this limitation, we propose a novel segmentation model D3Seg which is designed to maintain stable performance under missing-modality settings. D3Seg introduces Multi-hop Modality Graph Fusion (MMGF) to model higher order inter-modality dependencies, a lightweight diffusion-based imputation mechanism to compensate for missing T1ce representations in latent space, and probability-space decision refinement to mitigate dominant class overconfidence and improve delineation of underrepresented tumor subregions. Extensive evaluation on BraTS 2023 dataset demonstrates that our D3Seg model consistently improves segmentation performance under missing modality configurations. The proposed model achieves approximately 1.5-2.0% Dice improvement on enhancing tumor (ET) and around 1.0% on tumor core (TC) across multiple missing modality configurations compared to the current state-of-the-art model, while maintaining computational efficiency.
>
---
#### [new 027] Guided Trajectory Optimization with Sparse Scaling for Test-Time Diffusion
- **分类: cs.CV**

- **简介: 该论文属于扩散模型生成任务，解决测试时缩放效率问题。提出RTS方法，通过奖励引导和稀疏缩放提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.21907](https://arxiv.org/pdf/2605.21907)**

> **作者:** Gang Dai; Yining Huang; Yiming Xia; Guohao Chen; Shuaicheng Niu
>
> **摘要:** The efficient Test-Time Scaling (TTS) paradigm offers a promising perspective for enhancing the generation performance of diffusion models. However, current solutions are limited to a static, pre-defined noise pool and suffer from inflexible noise exploration across the denoising trajectory. To bridge this gap, we propose RTS, a novel Reward-guided Trajectory Scaling method to fully unlock the generative potential of diffusion models. Unlike existing methods, RTS facilitates the synthesis of refined, high-fidelity images via two core innovations: 1) a reward-guided noise optimization strategy to actively direct the search towards promising regions; and 2) a sparse test-time scaling framework together with a PCA-driven curvature analysis scheme to prioritize key intermediate steps in the entire denoising space, effectively compressing the search space. Experiments show our approach outperforms baselines by 15.6% across GenEval Score, and a 60.4% enhancement in ImageReward score, setting a new SOTA while providing a practical guideline for more effective test-time scaling across diffusion-specific architectures.
>
---
#### [new 028] Learning Emergent Modular Representations in Multi-modality Medical Vision Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态医学视觉任务，解决跨模态特征统计差异导致的表示崩溃问题。提出DEX模块化网络，通过专家与导演机制实现模态专精与协同，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.21861](https://arxiv.org/pdf/2605.21861)**

> **作者:** Yuting He; Chenyu You; Shuo Li
>
> **备注:** Accepted by KDD 2026
>
> **摘要:** Multi-modality medical vision (MV) foundation models (FM) are fundamentally challenged by pronounced Non-IID feature statistics across heterogeneous imaging modalities. Monolithic self-supervised optimization on such data induces conflicting gradients, driving representations to collapse toward modality-dominant shortcuts. This work reframes this failure as an imbalance between specialization and coordination in emergent modularity, and proposes Director-Experts (DEX), a modular network that explicitly regulates these dynamics in stacked modules. Each DEX module comprises a pool of experts, dynamically adapted by our image-wise activation strategy, autonomously specializing in modality-dominant statistics, together with a director, updated via our group exponential moving average, which distills multi-expert knowledge into a shared space for semantic integration across modalities, thus driving the emergence of modular representations. We curate a new benchmark, Medical Vision Universe, over 4 million images across 10 modalities, which provides a FM-level pre-training with the broadest coverage of distinct imaging modalities to our DEX. Extensive evaluations on 26 downstream tasks demonstrate improved optimization behavior and transferability, indicating DEX as a principled step toward general-purpose multi-modality medical AI. Our code and dataset will be opened at this https URL.
>
---
#### [new 029] Towards Clinically Interpretable Ophthalmic VQA via Spatially-Grounded Lesion Evidence
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉问答任务，旨在提升眼科学VQA的临床可解释性。通过构建包含空间定位病灶的基准，增强模型推理透明度与准确性。**

- **链接: [https://arxiv.org/pdf/2605.22414](https://arxiv.org/pdf/2605.22414)**

> **作者:** Xingyue Wang; Bo Liu; Meng Wang; Zhixuan Zhang; Chengcheng Zhu; Huazhu Fu; Jiang Liu
>
> **摘要:** Visual Question Answering (VQA) holds great promise for clinical support, particularly in ophthalmology, where retinal fundus photography is essential for diagnosis. However, ophthalmic VQA benchmarks primarily emphasize answer accuracy, neglecting the explicit visual evidence necessary for clinical interpretability. In this work, we introduce FundusGround, a new benchmark for clinically interpretable ophthalmic VQA with spatially-grounded lesion evidence. Specifically, we propose a three-stage pipeline that collects 10,719 fundus images with 15,595 image-level meticulously annotated lesions. To ensure anatomical consistency and clinical validity, all lesions are spatially localized using the Early Treatment Diabetic Retinopathy Study (ETDRS) grid, enabling standardized mapping to nine clinically meaningful retinal regions. Built upon this structured lesion evidence, 72,706 questions are then generated spanning four formats: open-ended, closed-ended, single-choice, and multiple-choice. We further benchmark multiple general- and medical- large vision-language models using dual metrics for answer accuracy and lesion-level reasoning. The experiments demonstrate that incorporating lesion-level visual evidence consistently improves model performance and transparency, highlighting the necessity of explicit spatial grounding for reliable and explainable ophthalmic VQA.
>
---
#### [new 030] SegGuidedNet: Sub-Region-Aware Attention Supervision for Interpretable Brain Tumor Segmentation
- **分类: cs.CV**

- **简介: 该论文属于脑肿瘤分割任务，旨在解决多模态MRI中肿瘤子区域分割的挑战。提出SegGuidedNet，通过注意力监督提升分割精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.22572](https://arxiv.org/pdf/2605.22572)**

> **作者:** Hasaan Maqsood; Saif Ur Rehman Khan; Sebastian Vollmer; Andreas Dengel; Muhammad Nabeel Asim
>
> **摘要:** Accurate segmentation of brain tumour sub-regions from multi-parametric MRI is critical for treatment planning yet remains challenging due to morphological variability, class imbalance, and overlapping appearances of tumour regions across imaging sequences. We propose SegGuidedNet, a three-dimensional residual encoder--decoder network introducing a novel SegAttentionGate module that explicitly supervises the decoder to produce spatially discriminative attention maps for each tumour sub-region necrotic core, peritumoral oedema, and enhancing tumour via a lightweight auxiliary loss, adding less than 0.2% parameter overhead. This sub-region supervision maintains decoder discriminability between visually ambiguous classes while providing free-of-cost spatial interpretability at inference without any post-hoc explanation method. Evaluated independently on BraTS2021 and BraTS2023 GLI across 251 held-out subjects each, SegGuidedNet achieves mean Dice of 0.905 (ET= 0.873, TC=0.906, WT=0.935) and 0.897 (ET=0.859, TC=0.902, WT=0.931) respectively, surpassing ensemble-based nnU-Net and HNF-Netv2 as a single model and approaching Swin UNETR a 10-model ensemble within 2--4 Dice points at a fraction of the inference cost. The consistency of results across two benchmark editions further confirms the generalisability of the proposed approach, offering competitive accuracy with built-in interpretability in a lightweight, clinically practical framework.
>
---
#### [new 031] SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决扩散Transformer在超训练分辨率下性能下降的问题。提出SEGA方法，通过动态调整注意力机制提升高分辨率生成效果。**

- **链接: [https://arxiv.org/pdf/2605.22668](https://arxiv.org/pdf/2605.22668)**

> **作者:** Javad Rajabi; Kimia Shaban; Koorosh Roohi; David B. Lindell; Babak Taati
>
> **备注:** 27 pages, 14 figures. Project page: this https URL
>
> **摘要:** Diffusion transformers (DiTs) have emerged as a dominant architecture for text-to-image generation, yet their performance drops when generating at resolutions beyond their training range. Existing training-free approaches mitigate this by modifying inference-time attention behavior, often through Rotary Position Embeddings (RoPE) extrapolation combined with attention scaling. However, these strategies apply a uniform and content-agnostic scaling across RoPE components with distinct frequency characteristics, inducing a trade-off between preserving global structure and recovering fine detail. We introduce SEGA, a training-free method that dynamically scales attention across RoPE components according to the latent's spatial-frequency structure at each denoising step. This adaptive scaling improves both structural coherence and fine-detail fidelity. Experiments show that SEGA consistently improves high-resolution synthesis across multiple target resolutions, outperforming state-of-the-art training-free baselines.
>
---
#### [new 032] MRecover: A Conditional Generative Model for Recovering Motion-Corrupted MR images Using AI Generated Contrast
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像修复任务，解决运动伪影导致的MRI数据质量问题。通过生成模型MRecover，将T1w图像转换为高质量TSE图像，提升数据可用性和诊断效果。**

- **链接: [https://arxiv.org/pdf/2605.21669](https://arxiv.org/pdf/2605.21669)**

> **作者:** Jinghang Li; Tales Santini; Courtney Clark; Bruno de Almeida; Cong Chu; Salem Alkhateeb; Andrea Sajewski; Jacob Berardinelli; Hecheng Jin; Tobias Campos; Jeremy J. Berardo; Joseph Mettenburg; Ariel Gildengers; Howard J. Aizenstein; Minjie Wu; Tamer S. Ibrahim
>
> **摘要:** Hippocampal subfield segmentation requires high-resolution T2w turbo spin echo (TSE) MRI, yet this sequence is susceptible to motion artifacts, leading to substantial data loss. We developed a conditional generative model (MRecover) that synthesizes routinely acquired T1w images to create TSE images with autoregressive slice conditioning for volumetric consistency. Trained on 7T MRI data (n=577), the model achieved high in-domain fidelity (n=148, SSIM=0.84, FSIM=0.94) and generalized well to out-of-domain 3T data: subfield volumes from synthesized and the as-acquired images closely matched: (n=416, r=0.87-0.97) and yielded 31.8% more analyzable subjects in the motion-affected ADNI3 dataset after quality control (593 vs 450). The synthesized images also achieved larger effect sizes due to increasing the sample size for diagnostic group differences in hippocampal subfield atrophy (whole hippocampus $\epsilon^2$= 0.121-0.100 vs. 0.086-0.062, left-right hemispheres). Project page: this https URL
>
---
#### [new 033] ConvNeXt-FD: A Fractal-Based Deep Model for Robust Biomedical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于 biomedical image segmentation 任务，旨在解决医学图像分割中的边界敏感性和形状保真问题。提出 ConvNeXt-FD 模型，结合 Fractal Dimension 思想优化损失函数，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2605.22002](https://arxiv.org/pdf/2605.22002)**

> **作者:** Joao Batista Florindo; Amanda Pontes de Oliveira Ornelas
>
> **摘要:** Biomedical image segmentation is a critical task in medical diagnosis and treatment planning, enabling precise delineation of anatomical structures and pathological regions. Despite significant advancements, challenges persist due to the inherent variability, noise, and complex morphology present in diverse medical imaging modalities. This paper introduces ConvNeXt-FD, a novel deep learning architecture for robust biomedical image segmentation, built upon a U-Net-like encoder-decoder framework leveraging the powerful ConvNeXt backbone. Our approach integrates a hybrid loss function combining the Dice coefficient with a boundary-aware regularization term inspired by a differentiable formulation of Fractal Dimension, designed to enhance the model's sensitivity to object boundaries and shape fidelity. We rigorously evaluate ConvNeXt-FD across six distinct biomedical datasets: BUSI (Breast Ultrasound Images), DDTI (Thyroid Ultrasound Images), FluoCells (Fluorescent Cell Images), IDRiD (Diabetic Retinopathy Images for Optic Disc Segmentation), ISIC2018 (Skin Lesion Images), and MoNuSeg (Nuclei Segmentation). Experimental results demonstrate that ConvNeXt-FD, particularly when initialized with ImageNet pre-trained weights, achieves competitive and often superior performance compared to existing state-of-the-art methods across various metrics, including Dice, Jaccard, Accuracy, Sensitivity, Specificity, and False Positive Rate. The integration of ConvNeXt as a strong encoder, coupled with the boundary-aware regularization, proves effective in capturing both high-level semantic features and fine-grained boundary details, leading to more accurate and reliable segmentations in challenging biomedical contexts.
>
---
#### [new 034] GALAR-TemporalNet v2: Anatomy-Guided Dual-Branch Temporal Classification with Bidirectional Mamba and Dual-Graph GCN for Video Capsule Endoscopy -- after competition results
- **分类: cs.CV**

- **简介: 该论文属于视频胶囊内镜的多标签时序分类任务，解决器官定位与病灶检测问题。提出GALAR-TemporalNet v2模型，优化结构与损失函数，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2605.22209](https://arxiv.org/pdf/2605.22209)**

> **作者:** Jiye Won; Seangmin Lee; Soon Ki Jung
>
> **备注:** 7 pages, 2 figures. Post-competition preprint for the ICPR 2026 RARE-VISION Challenge
>
> **摘要:** Video Capsule Endoscopy (VCE) poses a challenging multi-label temporal classification problem, requiring simultaneous localization of 8 anatomical regions and detection of 9 pathological findings across tens of thousands of frames. We present GALAR-TemporalNet v2, a hierarchical temporal model that addresses three core challenges: extreme class imbalance, long-range temporal dependencies, and pathology--anatomy entanglement. Our architecture combines windowed self-attention for local modeling, a Dual-Graph GCN for global frame relationships, and Bidirectional Mamba for selective boundary context encoding. A novel anatomy prototype residual pathway decouples pathological deviation signals from normal organ appearance, and a frame-level GCN skip connection stabilizes training of visually confusable rare classes. The competition version, GALAR-TemporalNet, achieved an overall mAP@0.5 of 0.2644 and mAP@0.95 of 0.2353 on the RARE-VISION test set. Following the competition, the redesigned GALAR-TemporalNet v2 -- incorporating a restructured pathology branch, refined loss functions, and extended post-processing -- improved these results to mAP@0.5 of 0.3409 and mAP@0.95 of 0.3333.
>
---
#### [new 035] Physiology and Anatomy Aware Inverse Inference of Myocardial Infarction for Cardiac Digital Twin
- **分类: cs.CV**

- **简介: 该论文属于心肌梗死定位任务，旨在解决传统方法对心肌瘢痕和电生理变化描述不足的问题。通过构建心脏数字孪生框架，结合影像与ECG数据，提升定位准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2605.22044](https://arxiv.org/pdf/2605.22044)**

> **作者:** Mengxiao Wang; Yilin Lyu; Julia Camps; Ching Hui Sia; Mark Yan-Yee Chan; Yanrui Jin; Shuzhi Sam Ge; Chengliang Liu; Lei Li
>
> **备注:** Early-accepted by MICCAI 2026. This version corresponds to the submitted version. The final version will be available on Springer Link
>
> **摘要:** Accurate localization of myocardial infarction is essential for risk stratification. While LGE-MRI remains the gold standard, it is resource-intensive. Integrating cine MRI with ECG enables a more detailed representation of infarct properties. Existing inverse MI inference methods overlook realistic scar morphology and cardiac repolarization, reducing sensitivity to subtle ECG variations and interpretability of infarct-induced electrophysiological changes. In this paper, we propose a novel framework for noninvasive MI localization using cardiac digital twins. To bridge the domain gap between simulation and reality, we introduce an anatomy-aware stochastic infarct synthesis strategy to synthesize realistic, irregular scars with border zones, mimicking ischemic transmural progression. We then construct a virtual cohort to simulate QRS-T waveforms, capturing both depolarization and repolarization dynamics. Furthermore, we design a Physiology and Anatomy Aware Network (PAA-Net) that jointly encodes 3D myocardial geometry and multi-lead ECGs to infer infarct areas with varying localizations, sizes, spatial extents, and transmuralities. Experimental results demonstrate that our framework significantly outperforms existing methods in inverse inference, achieving Dice scores of 0.7391 and 0.5503 for scar and border zone segmentation, respectively, while further enhancing the interpretability of the ECG-infarct relationship. Our code will be released upon acceptance.
>
---
#### [new 036] AgroTools: A Benchmark for Tool-Augmented Multimodal Agents in Agriculture
- **分类: cs.CV**

- **简介: 该论文提出AgroTools，一个用于评估农业中工具增强多模态代理的基准。解决农业决策中多模态系统工具使用能力不足的问题，通过构建包含多种任务和工具的数据集进行模型评测。**

- **链接: [https://arxiv.org/pdf/2605.22366](https://arxiv.org/pdf/2605.22366)**

> **作者:** Zi Ye; Yibin Wen; Xiaoya Fan; Xinyu Zhang; Jing Wu; Kun Zeng; Zurong Mai; Jiarui Zhang; Bohan Shi; Juepeng Zheng; Jianxi Huang; Yutong Lu; Haohuan Fu
>
> **摘要:** Agricultural decision-making increasingly requires multimodal systems that can transform visual observations into reliable, executable actions. However, existing agricultural multimodal benchmarks mainly evaluate final-answer correctness and provide limited support for assessing whether models can use external tools to complete precision-sensitive workflows. In this paper, we introduce AgroTools, a benchmark for evaluating tool-augmented multimodal agents in agriculture. AgroTools contains 539 question-answer instances paired with 1,097 heterogeneous agricultural images, spanning five task families and an executable environment of 14 agricultural tools. Each query is annotated with structured tool-use traces, enabling a dual-view evaluation of both process-level execution quality and outcome-level task success. We benchmark 9 open-source and 4 closed-source multimodal large language models on AgroTools. Results show that current models remain far from reliable in agricultural tool-use settings, with clear bottlenecks in tool planning, argument generation, execution recovery, and final-answer synthesis. We hope AgroTools will support future research on multimodal agents for high-precision agricultural applications. The benchmark and evaluation are available at this https URL.
>
---
#### [new 037] Bernini: Latent Semantic Planning for Video Diffusion
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文提出Bernini框架，融合MLLM与扩散模型，解决视频生成与编辑问题。MLLM负责语义规划，扩散模型根据语义生成像素，提升生成质量与编辑能力。**

- **链接: [https://arxiv.org/pdf/2605.22344](https://arxiv.org/pdf/2605.22344)**

> **作者:** Bernini Team; Chenchen Liu; Junyi Chen; Lei Li; Lu Chi; Mingzhen Sun; Zhuoying Li; Yi Fu; Ruoyu Guo; Yiheng Wu; Ge Bai; Zehuan Yuan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Multimodal large language models (MLLMs) and diffusion models have each reached remarkable maturity: MLLMs excel at reasoning over heterogeneous multimodal inputs with strong semantic grounding, while diffusion models synthesize images and videos with photorealistic fidelity. We argue that these two families can be unified through a simple division of labor: MLLMs perform semantic planning, while diffusion models render pixels from high-level semantic guidance and low-level visual features. Building on this idea, we propose Bernini, a unified framework for video generation and editing. An MLLM-based planner predicts the target semantic representation directly in the ViT embedding space, and a DiT-based renderer synthesizes pixels conditioned on this plan, augmented by text features and, for editing, source VAE features for detail preservation. Because semantics serve as the interface, the planner and renderer can be trained separately and only lightly co-trained, preserving the pretrained strengths of both components while keeping training efficient. To better handle multiple visual inputs, we introduce Segment-Aware 3D Rotary Positional Embedding (SA-3D RoPE), and further incorporate chain-of-thought reasoning in the planner to better transfer understanding into generation. Bernini achieves state-of-the-art performance across a wide range of video generation and editing benchmarks, with the MLLM's pretrained understanding translating into strong generalization on challenging editing tasks.
>
---
#### [new 038] EasyVFX: Frequency-Driven Decoupling for Resource-Efficient VFX Generation
- **分类: cs.CV**

- **简介: 该论文提出EasyVFX，解决资源受限下的高质量VFX生成问题。通过频域解耦，降低计算需求，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.22051](https://arxiv.org/pdf/2605.22051)**

> **作者:** Yue Ma; Xu Ye; Qinghe Wang; Yucheng Wang; Hongyu Liu; Yinhan Zhang; Xinyu Wang; Yuanpeng Che; Shanhui Mo; Paul Liang; Fangneng Zhan; Qifeng Chen
>
> **备注:** Accepted by SIGGRAPH 2026. Project page: this https URL
>
> **摘要:** Generating high-fidelity visual effects (VFX) typically demands massive datasets and prohibitive computational power due to the intricate coupling of spatial textures and temporal dynamics. In this paper, we introduce EasyVFX, a resource-efficient framework that achieves realistic VFX synthesis under stringent constraints. Our core philosophy lies in frequency-domain decomposition: we observe that the complexity of VFX can be significantly mitigated by decoupling high-frequency components, which represent intricate spatial appearances, from low-frequency components that encapsulate global motion dynamics. This spectral disentanglement transforms a high-dimensional learning problem into manageable sub-tasks, thereby lowering the optimization barrier and reducing data dependency. Building upon this insight, we propose a two-stage training paradigm. First, we design a Frequency-aware Mixture-of-Experts (Freq-MoE) architecture. By utilizing a soft routing mechanism, our model assigns specialized experts to distinct spectral bands, enabling them to cultivate robust priors for appearance and motion dynamics. This specialization allows the model to acquire foundational VFX knowledge with fewer GPU resources. Second, we introduce a Test-Time Training strategy powered by a novel Frequency-constraint Loss. This allows the pre-trained model to swiftly adapt to specific, unseen effects through localized optimizations, requiring only about 100 steps on a single GPU. Experimental results demonstrate that EasyVFX produces structurally consistent and visually stunning effects, proving that frequency-aware learning is a key catalyst for democratizing professional-grade VFX.
>
---
#### [new 039] Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following
- **分类: cs.CV**

- **简介: 该论文属于视觉任务，解决 gaze following 问题。针对 VFMs 在 gaze reasoning 上的不足，提出新训练机制提升其性能，尤其在目标不显著时表现更优。**

- **链接: [https://arxiv.org/pdf/2605.22607](https://arxiv.org/pdf/2605.22607)**

> **作者:** Shijing Wang; Yaping Huang; Chaoqun Cui; David Wong; Yihua Cheng; Alexandros Neophytou; Hyung Jin Chang
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Gaze following requires both scene understanding and gaze reasoning to localize the gaze target of an in-scene person. Recently, vision foundation models (VFMs) have demonstrated strong performance on this task, enabling simpler architectures while outperforming prior methods. However, we observe a key limitation of VFM-based approaches: while VFMs substantially improve scene understanding, they contribute little to gaze reasoning. As a result, existing methods often rely on semantically salient objects rather than true gaze cues, leading to degraded performance when targets are not salient. To address this, we propose a novel training mechanism to enhance gaze reasoning in VFMs for gaze following. Our method includes: (1) a head-conditioned local LoRA, which enables localized adaptation to preserve scene token learning while improving head token learning for gaze reasoning; and (2) an out-of-cone penalty, which injects gaze cues into head tokens while aligning them with scene tokens. Experiments on the GazeFollow and VAT datasets demonstrate that our method achieves state-of-the-art performance, with particularly strong improvements when gaze targets are not semantically salient. Our findings offer valuable insights for advancing future gaze following research. We will release the code once the paper is accepted.
>
---
#### [new 040] VGenST-Bench: A Benchmark for Spatio-Temporal Reasoning via Active Video Synthesis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大语言模型的时空推理任务，旨在解决现有数据集评估能力不足的问题。通过主动生成视频数据构建基准，提升对模型细粒度时空理解的评估能力。**

- **链接: [https://arxiv.org/pdf/2605.22570](https://arxiv.org/pdf/2605.22570)**

> **作者:** Jinho Park; Youbin Kim; Hogun Park; Eunbyung Park
>
> **备注:** 82 pages, 91 figures (7 in main paper, 84 in appendix). Project page: this https URL
>
> **摘要:** Spatio-temporal reasoning is a core capability for Multimodal Large Language Models (MLLMs) operating in the real world. As such, evaluating it precisely has become an essential challenge. However, existing spatio-temporal reasoning benchmark datasets primarily rely on static image sets or passively curated video data, which limits the evaluation of fine-grained reasoning capabilities. In this paper, we introduce VGenST-Bench, a video benchmark that employs generative models to actively synthesize highly controlled and diverse evaluation scenarios. To construct VGenST-Bench, we propose a multi-agent pipeline incorporating a human quality control stage, ensuring the quality of all generated videos and QA pairs. We establish a comprehensive 3x2x2 video taxonomy, encompassing Spatial Scale, Perspective, and Scene Dynamics to span diverse scenarios. Furthermore, we design a hierarchical task suite that decouples low-level visual perception from high-level spatio-temporal reasoning. By shifting the paradigm from passive curation to active synthesis, VGenST-Bench enables fine-grained diagnosis of spatio-temporal understanding in MLLMs.
>
---
#### [new 041] GazePrior: Zero-Shot AR/VR Eye Tracking via Learned 3D Gaze Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于AR/VR眼动追踪任务，解决新设备训练数据不足的问题。通过构建3D眼动先验模型，实现无需额外数据的高质量合成数据生成。**

- **链接: [https://arxiv.org/pdf/2605.22359](https://arxiv.org/pdf/2605.22359)**

> **作者:** Corentin Dumery; David Colmenares; Alexander Fix; Pascal Fua; Ali Behrooz; Jogendra Kundu
>
> **备注:** Project page: this https URL
>
> **摘要:** Eye tracking (ET) is a foundational technology for advanced AR/VR applications. However, training ET models for every new ET device is challenging: real data collection is costly and time-consuming, while existing synthetic data generation methods lack realism. To remove the need for additional data collection while maintaining data quality, we introduce a data-driven 3D prior that models the distribution of human eyes across diverse identities, gaze directions, and light settings. This model, which we coin GazePrior, then enables sparse-input 3D reconstruction of annotated data collected with previous ET devices, which can in turn be rendered from the cameras of any target ET device. Our approach synthesizes data with the realism, diversity and ground-truth accuracy of real data collection without its prohibitive costs. Our experiments demonstrate that ET models trained with our synthesized data outperform previous zero-shot methods, achieving higher accuracy and robustness.
>
---
#### [new 042] REACH: Hand Pose Estimation from Room Corners
- **分类: cs.CV**

- **简介: 该论文属于3D手部姿态估计任务，解决远距离、低分辨率和遮挡下的手部姿态恢复问题。通过结合手体协调与多视角信息，提出REACH-Net模型，并构建了REACH数据集进行训练与测试。**

- **链接: [https://arxiv.org/pdf/2605.22231](https://arxiv.org/pdf/2605.22231)**

> **作者:** Shu Nakamura; Ryo Kawahara; Genki Kinoshita; Ryosuke Hirai; Yasutomo Kawanishi; Shohei Nobuhara; Ko Nishino
>
> **摘要:** We introduce a novel 3D hand pose estimator that can accurately recover the shape and pose of people's hands in a room from afar, typically from fixed cameras at room corners, in extremely low-resolution and frequently occluded views. Our key idea is to fully leverage hand-body coordination, its temporal progression, and multiview observations. We achieve this with a novel Transformer-based model, in which hand and body configurations are modeled through correlations between their visual features expressed as per-view tokens, and their temporal coordination is exploited in an autoregressive manner. We introduce a novel dataset, which we refer to as REACH, Room-Environment dataset Annotated with Chest cameras for Hand pose estimation, to train and test our method. REACH is a first-of-its-kind large-scale hand pose dataset that captures accurate hand movements of 50 participants across a wide variety of daily activities. In order to avoid interfering with natural movements while annotating the hands with accurate shape and pose, we leverage concealed chest cameras. Through extensive experiments, including comparative studies with existing methods, we show that our model, REACH-Net, achieves highly accurate 3D hand pose estimation from afar. These results broaden the horizon of 3D hand pose estimation, especially towards "in-the-wild" continuous human behavior analysis.
>
---
#### [new 043] Multi-scale interaction network for stereo image super-resolution
- **分类: cs.CV**

- **简介: 该论文属于立体图像超分辨率任务，旨在提升图像分辨率并充分利用双目信息。通过设计多尺度交互网络，增强视图内和视图间特征提取，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2605.21913](https://arxiv.org/pdf/2605.21913)**

> **作者:** Liyi Xu; Lin Qi
>
> **摘要:** Stereo image super-resolution aims to generate high-resolution images by leveraging complementary information from binocular systems. Although previous studies have achieved impressive results, the potential of intra-view and cross-view information has not been fully exploited. To address this issue, we propose a novel multi-scale interaction network for stereo image super-resolution. Specifically, we design a Multi-scale Spatial-Channel Attention Module that utilizes multi-scale large separable kernel attention and simple channel attention to improve intra-view feature extraction. Additionally, we propose a Dual-View Epipolar Attention Module, utilizing an optimal transport algorithm to achieve more accurate matching along the epipolar line. Extensive experimental and ablation studies show that our method achieves competitive results that outperform most SOTA methods.
>
---
#### [new 044] VISTA: Validation-Guided Integration of Spatial and Temporal Foundation Models with Anatomical Decoding for Rare-Pathology VCE Event Detection -- after competition results
- **分类: cs.CV**

- **简介: 该论文针对罕见病理事件检测任务，提出VISTA框架，融合时空模型与解剖信息，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.22096](https://arxiv.org/pdf/2605.22096)**

> **作者:** Bo-Cheng Qiu; Fang-Ying Lin; Ming-Han Sun; Yu-Fan Lin; Chia-Ming Lee; Chih-Chung Hsu
>
> **摘要:** Capsule endoscopy event detection is challenging because clinically relevant findings are sparse, visually heterogeneous, and evaluated at the event level rather than by frame accuracy. We propose VISTA, a metric-aligned multi-backbone framework for the RAREVISION task. VISTA combines EndoFM-LV for temporal context and DINOv3 ViTL/16 for frame-level visual semantics, followed by a Diverse Head Ensemble (DHE), Validation-Guided Weighted Fusion (VGWF), and Anatomy-Aware Temporal Event Decoding (ATED). The original official submission achieved hidden-test temporal mAP@0.5 of 0.3530 and mAP@0.95 of 0.3235. After the competition, extending local threshold refinement with a global coarse search improved performance to 0.3726 mAP@0.5 and 0.3431 mAP@0.95, ranking Team ACVLab second in the post-competition evaluation.
>
---
#### [new 045] MM-Conv: A Multimodal Dataset and Benchmark for Context-Aware Grounding in 3D Dialogue
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言接地任务，解决对话中模糊指代的问题。构建了3D对话基准并提出两阶段接地方法，提升指代理解效果。**

- **链接: [https://arxiv.org/pdf/2605.21796](https://arxiv.org/pdf/2605.21796)**

> **作者:** Anna Deichler; Jim O'Regan; Fethiye Irmak Dogan; Lubos Marcinek; Anna Klezovich; Iolanda Leite; Jonas Beskow
>
> **备注:** Extended version of the paper published at LREC 2026 (Palma de Mallorca, Spain), with expanded VLM baselines and inter-annotator agreement analysis
>
> **摘要:** Grounding language in the physical world requires AI systems to interpret references that emerge dynamically during conversation. While current vision-language models (VLMs) excel at static image tasks, they struggle to resolve ambiguous expressions in spontaneous, multi-turn dialogue. We address this gap by introducing (1) a benchmark for referential communication in dynamic 3D environments, built from 6.7 hours of egocentric VR interaction with synchronized speech, motion, gaze, and 3D scene geometry, and (2) a two-stage grounding pipeline that explicitly resolves conversational ambiguity before visual localization. The benchmark includes over 4,200 manually verified referring expressions spanning full, partitive, and pronominal types. Our contextual rewriting approach improves grounding performance by 11-22 percentage points on average, with a pure detector (GroundingDINO) reaching 56.7% on pronominals after rewriting, nearly double the best end-to-end baseline. Results demonstrate that decoupling linguistic reasoning from visual perception is more effective than end-to-end approaches for conversational grounding.
>
---
#### [new 046] Rethinking Token Reduction for Diffusion Models via Output-Similarity-Awareness
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散模型中token减少效率与质量的平衡问题。提出DiTo方法，通过输出相似性引导的token减少策略，提升生成质量与速度。**

- **链接: [https://arxiv.org/pdf/2605.22011](https://arxiv.org/pdf/2605.22011)**

> **作者:** Hangyeol Lee; Hyojeong Lee; Joo-Young Kim
>
> **摘要:** Diffusion Transformers (DiTs) achieve superior image generation quality but suffer from quadratic computational complexity relative to token count. While various token reduction (TR) methods have been proposed to mitigate this cost, they overlook the primary objective of generative models: minimizing recovery error, which requires reflecting output token similarity. They rely solely on input token similarity inherited from reduction-only ViT paradigms, leading to a fundamental misalignment with this objective. To bridge this gap, we propose DiTo, a novel TR paradigm that shifts the focus toward output-centric token reduction. Based on the observation that output token similarity is consistently preserved across adjacent timesteps, DiTo utilizes prior-step similarities as an effective proxy to establish token correspondences at a Matching timestep, which are then reused across multiple subsequent Reduction timesteps. To optimize this interleaved scheduling, we propose Pair Match Ratio (PMR)-guided Interval Scheduling to determine the optimal matching frequency. Furthermore, to mitigate localized approximation errors and resulting blocking artifacts caused by repeated reuse, we propose Frequency-aware Token Matching by incorporating a selection-frequency penalty. Extensive experiments demonstrate that DiTo consistently outperforms existing TR methods with 1.6-3.9 dB higher PSNR at comparable speedups, achieving a superior Pareto frontier.
>
---
#### [new 047] Segment Anything with Motion, Geometry, and Semantic Adaptation for Complex Nonlinear Visual Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪任务，解决传统方法在复杂场景下的泛化能力不足问题。通过改进SAM 2，引入运动、几何和语义线索，提升跟踪稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.22538](https://arxiv.org/pdf/2605.22538)**

> **作者:** Deyi Zhu; Yuji Wang; Yong Liu; Yansong Tang; Bingyao Yu; Jiwen Lu; Jie Zhou
>
> **摘要:** Traditional visual object tracking (VOT) methods typically rely on task-specific supervised training, limiting their generalization to unseen objects and challenging scenarios with distractors, occlusion, and nonlinear motion. Recent vision foundation models, exemplified by SAM 2, learn strong video understanding priors from large-scale pretraining and offer a promising foundation for building more robust and generalizable trackers. However, directly applying SAM 2 to VOT remains suboptimal, as it does not explicitly model target motion dynamics or enforce geometric and semantic consistency across frames, both of which are essential for reliable tracking. To address this issue, we propose SAMOSA, a new tracking framework that adapts SAM 2 to complex VOT scenarios by explicitly leveraging motion, geometry, and semantic cues. Specifically, we introduce a lightweight nonlinear motion predictor to model target dynamics and guide mask selection as well as memory filtering. We further exploit semantic cues to detect target shifts and recover from tracking failures, while geometric cues are incorporated as structural constraints to improve tracking stability. In this way, SAMOSA bridges the gap between the implicit video understanding prior of SAM 2 and explicit tracking-oriented modeling. Extensive experiments show that SAMOSA consistently outperforms state-of-the-art SAM 2--based approaches on general benchmarks, demonstrates stronger generalization than supervised VOT methods, and achieves substantial gains on anti-UAV datasets, which typify complex nonlinear motion scenarios. Our code is available at this https URL.
>
---
#### [new 048] Flat-Pack Bench: Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决现有基准对细粒度时空推理评估不足的问题。通过构建家具组装基准，评估模型在时间顺序、状态定位等任务上的表现。**

- **链接: [https://arxiv.org/pdf/2605.21625](https://arxiv.org/pdf/2605.21625)**

> **作者:** Aditya Chetan; Eric Cai; Peeyush Kushwaha; Bharath Raj Nagoor Kani; Utkarsh Mall; Qianqian Wang; Noah Snavely; Bharath Hariharan
>
> **备注:** CVPR 2026
>
> **摘要:** The emergence of Large Vision-Language Models (LVLMs) has significantly advanced video understanding capabilities. However, existing benchmarks focus predominantly on coarse-grained tasks such as action segmentation, classification, captioning, and retrieval. Furthermore, these benchmarks often rely on entities that can be easily identified verbally, like household objects, animals, human subjects, etc., limiting their applicability to complex, in-the-wild video scenarios. But, many applications such as furniture assembly, cooking, etc., require step-by-step fine-grained spatio-temporal understanding of the video, which is not sufficiently evaluated in current benchmarks. To address this gap, we introduce Flat-Pack Bench, a novel benchmark centered on furniture assembly tasks. Our benchmark evaluates LVLMs on nuanced tasks, including temporal ordering of assembly actions, temporal localization of assembly state, understanding part mating, and tracking, using multiple-choice questions paired with visual prompts highlighting relevant parts as references for fine-grained questions. Our experiments reveal that state-of-the-art LVLMs struggle significantly with fine-grained spatio-temporal reasoning, highlighting their limitations in effectively leveraging temporal information from videos, limited tracking ability, and understanding of spatial interactions like physical contact.
>
---
#### [new 049] Diffusion-guided Generalizable Enhancer for Urban Scene Reconstruction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于城市场景重建任务，解决现有方法在大视角变化下质量下降的问题。提出GenRe，利用扩散模型提升3D表示质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.22420](https://arxiv.org/pdf/2605.22420)**

> **作者:** Henry Che; Jingkang Wang; Yun Chen; Ze Yang; Sivabalan Manivasagam; Raquel Urtasun
>
> **备注:** ICRA 2026. Project page: this https URL
>
> **摘要:** Urban scene reconstruction from real-world observations has emerged as a powerful tool for self-driving development and testing. While current neural rendering approaches achieve high-fidelity rendering along the recorded trajectories, their quality degrades significantly under large viewpoint shifts, limiting the applicability for closed-loop simulation. Recent works have shown promising results in using diffusion models to enhance quality at these challenging viewpoints and distill improvements back into 3D representations. However, they often require costly per-scene optimization, and the distilled representations remain fragile and fail to generalize beyond limited synthesized views. To address these limitations, we propose GenRe, a novel diffusion-guided generalizable enhancer for urban scene reconstruction. GenRe takes as input any pretrained 3D Gaussian representation and fixes the deficiencies within a few minutes. By learning to distill generative priors across diverse scenes, GenRe produces robust and high-fidelity representation efficiently that generalizes reliably to challenging unseen viewpoints (e.g., lane change). Experiments show that GenRe outperforms existing methods in both quality and efficiency and benefits various downstream tasks, enabling robust and scalable sensor simulation for autonomous driving.
>
---
#### [new 050] MuKV: Multi-Grained KV Cache Compression for Long Streaming Video Question-Answering
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文针对长视频问答任务，解决LLM在处理长视频时的内存和效率问题。提出MuKV方法，通过多粒度KV缓存压缩提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2605.22269](https://arxiv.org/pdf/2605.22269)**

> **作者:** Junbin Xiao; Jiajun Chen; Tianxiang Sun; Xun Yang; Angela Yao
>
> **备注:** To appear at CVPR'26. Code is available at this https URL
>
> **摘要:** Long streaming video QA remains challenging due to growing visual tokens and limited reasoning length of large language models (LLMs). KV-caching stores the Key-Value (KV) of the historical tokens via LLM prefill and enables more efficient streaming QA. However, existing methods cache every one or two frames, causing redundant memory usage and losing fine-grained spatial details within frame or temporal contexts across frames. This paper proposes MuKV, a method that features a multi-grained KV cache compression module and a semi-hierarchical retrieval approach to improve both efficiency and accuracy for long streaming VideoQA. For the offline KV cache, MuKV extracts visual representations at patch-, frame-, and segment-levels. The multiple levels of granularity preserve both local cues and global temporal context, while maintaining efficiency with a dual signal token compression mechanism guided by self-attention and frequency. For online QA, MuKV designs a semi-hierarchical retrieval method to retrieve relevant KV caches for answer generation. Experiments on long-streaming VideoQA benchmarks show that MuKV significantly improves answer accuracy, without sacrificing memory and online QA efficiency. Moreover, our compression mechanism alone brings consistent benefits across answer accuracy, memory, and QA efficiency over baselines, showcasing highly effective contribution.
>
---
#### [new 051] MLLMs Know When Before Speaking: Revealing and Recovering Temporal Grounding via Attention Cues
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦视频时间定位任务，解决MLLMs时间戳预测不可靠的问题。通过分析注意力机制，提出一种无需参数更新的推理框架，提升模型时间定位性能。**

- **链接: [https://arxiv.org/pdf/2605.21954](https://arxiv.org/pdf/2605.21954)**

> **作者:** Dazhao Du; Liao Duan; Jian Liu; Tao Han; Yujia Zhang; Eric Liu; Xi Chen; Song Guo
>
> **备注:** Project Website: this https URL
>
> **摘要:** Video temporal grounding (VTG), which localizes the start and end times of a queried event in an untrimmed video, is a key test of whether multimodal large language models (MLLMs) understand not only what happens but also when it happens. Although modern MLLMs describe video content fluently, their timestamp predictions remain unreliable, while existing remedies either require costly post-training on temporal annotations or rely on coarse training-free heuristics. In this work, we probe the cross-modal attention of MLLMs and uncover a perception-generation gap. Our key finding is that MLLMs often know the target interval during prefill, but lose this signal when generating the final answer. In the prefill stage, a sparse set of attention heads, which we call \emph{Temporal Grounding Heads} (TG-Heads), concentrates query-to-video attention on the ground-truth interval. During autoregressive decoding, however, the answer tokens shift attention away from this interval toward visually salient but query-irrelevant segments. This observation motivates an inference-time read-then-regenerate framework. We first convert TG-Head prefill attention into a debiased frame-level relevance signal and extract the high-attention interval it highlights. We then re-invoke the MLLM with visual context restricted to this interval, using video cropping or attention masking to suppress distractors. Without parameter updates and architectural changes, our framework consistently improves MiMo-VL-7B, Qwen3-VL-8B, and TimeLens-8B on three VTG benchmarks, with gains of up to +3.5 mIoU. The project website can be found at this https URL.
>
---
#### [new 052] Detection of Virus and Small Cell Patches in Foci Images Using Switchable Convolution and Feature Pyramid Networks
- **分类: cs.CV**

- **简介: 该论文属于生物医学图像目标检测任务，旨在准确识别和计数病毒和小细胞斑点。通过结合FPN和可切换卷积改进YOLOv2，提升多尺度特征表示与细粒度目标检测能力。**

- **链接: [https://arxiv.org/pdf/2605.22290](https://arxiv.org/pdf/2605.22290)**

> **作者:** Amrita Singh; Snehasis Mukherjee
>
> **摘要:** Accurate detection and counting of virus patches in focus-forming unit (FFU) images, also known as foci images, are important for quantifying viral infection and analyzing cellular structures. This task is challenging because biomedical targets often vary substantially in size, density, contrast, and shape. In this paper, we propose an enhanced YOLOv2-based detector that integrates a Feature Pyramid Network (FPN) to improve multi-scale feature representation. We also incorporate a switchable atrous convolution mechanism to adapt the receptive field for fine-grained targets in dense microscopy images. The proposed method is evaluated on biomedical foci image datasets for virus patch and small cell patch detection. For small cell patch detection, the model achieves a mean average precision (mAP) of 40.5% at a 25% Intersection over Union (IoU) threshold. For FFU virus patch detection, the model achieves an mAP of 68%. These results indicate that combining FPN-based feature fusion with switchable convolution improves the suitability of YOLOv2 for specialized biomedical object detection tasks
>
---
#### [new 053] SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于空间推理任务，旨在解决视觉退化下模型鲁棒性不足的问题。构建了SpaceDG数据集和基准，评估并提升模型在退化场景下的空间智能表现。**

- **链接: [https://arxiv.org/pdf/2605.22536](https://arxiv.org/pdf/2605.22536)**

> **作者:** Xiaolong Zhou; Yifei Liu; Ziyang Gong; Jiarui Li; Qiyue Zhao; Muyao Niu; Yuanyuan Gao; Le Ma; Xue Yang; Hongjie Zhang; Zhihang Zhong
>
> **摘要:** Multimodal Large Language Models (MLLMs) have made rapid progress in spatial intelligence, yet existing spatial reasoning benchmarks largely assume pristine visual inputs and overlook the degradations that commonly occur in real-world deployment, such as motion blur, low light, adverse weather, lens distortion, and compression artifacts. This raises a fundamental question: how robust is the spatial intelligence of current MLLMs when visual observations are imperfect? To answer this question, we introduce SpaceDG, the first large-scale dataset for degradation-aware spatial understanding. It is constructed with a physically grounded degradation synthesis engine that embeds degradation formation process into 3D Gaussian Splatting (3DGS) rendering, enabling realistic simulation of nine degradation types. The resulting dataset contains approximately 1M QA pairs from nearly 1,000 indoor scenes. We further introduce SpaceDG-Bench, an human-verified benchmark with 1,102 questions spanning 11 reasoning categories and 9 visual degradation types, yielding over 10K VQA instances. Evaluating 25 open- and closed-source MLLMs reveals that visual degradations consistently and substantially impair spatial reasoning, exposing a critical robustness gap. Finally, we show that finetuning on SpaceDG markedly improves degradation robustness and can even surpass human performance under degraded conditions without any performance drop on clean images, highlighting the promise of degradation-aware training for robust spatial intelligence.
>
---
#### [new 054] Broken Memories: Detecting and Mitigating Memorization in Diffusion Models with Degraded Generations
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型中的记忆问题。通过分析生成过程中的数值稳定性，提出检测与缓解记忆的方法，有效降低记忆率并保持图像质量。**

- **链接: [https://arxiv.org/pdf/2605.22050](https://arxiv.org/pdf/2605.22050)**

> **作者:** Yuanmin Huang; Mi Zhang; Chen Chen; Feifei Li; Geng Hong; Xiaoyu You; Min Yang
>
> **备注:** KDD 2026, extended version
>
> **摘要:** While diffusion models excel at generating high-quality images, their tendency to memorize training data poses significant privacy and copyright risks. In this work, we for the first time identify that memorization induces internal numerical instability, often manifesting as visually ``broken'' artifacts. Inspired by stability analysis in numerical methods, we introduce empirical stability regions based on latent update norms to quantitatively characterize stable behavior during generation. Leveraging this, we propose a principled, on-the-fly framework for step-wise detection and adaptive mitigation. Our approach suppresses memorization without altering prompts or guidance, thereby preserving semantic fidelity and image quality. Extensive experiments on Stable Diffusion 1.4 demonstrate that our method achieves an AUC $>0.999$ detection performance and a $0.0\%$ memorization rate after mitigation with negligible overhead ($\approx0.01$s per image).
>
---
#### [new 055] EvoVid: Temporal-Centric Self-Evolution for Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出EvoVid，解决视频大语言模型自进化问题，通过时间感知奖励机制，实现从原始视频中自主学习，提升视频理解能力。**

- **链接: [https://arxiv.org/pdf/2605.21931](https://arxiv.org/pdf/2605.21931)**

> **作者:** Shiqi Huang; Ziyue Wang; Zhongrong Zuo; Han Qiu; Qi She; Bihan Wen
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent Video Large Language Models (Video-LLMs) have demonstrated strong capabilities in video reasoning through reinforcement learning (RL). However, existing RL pipelines rely heavily on human-annotated tasks and solutions, making them costly to scale and fundamentally constrained by human expertise. Self-evolving frameworks have recently emerged as a promising alternative through autonomous Questioner-Solver self-play. Unfortunately, these approaches are primarily designed for static modalities such as text and images, fundamentally failing to capture the temporal dynamics that are central to video reasoning. In this work, we propose $\textbf{EvoVid}$, a temporal-centric self-evolving framework that enables Video-LLMs to improve directly from raw, unannotated videos. Specifically, we introduce two complementary temporal-centric rewards: a temporal-aware Questioner reward that encourages temporally dependent question generation through temporal perturbation sensitivity, and a temporal-grounded Solver reward that provides automatic temporal supervision via inherent video segment localization. Extensive experiments across four base models and six benchmarks demonstrate consistent improvements over both base models and existing self-evolving baselines, achieving competitive performance with supervised methods. These results highlight temporal-centric self-evolution as an effective and scalable paradigm for video understanding and reasoning.
>
---
#### [new 056] Cross-Domain Human Action Recognition from Multiview Motion and Textual Descriptions
- **分类: cs.CV**

- **简介: 该论文属于零样本动作识别任务，旨在解决跨域场景下动作识别性能下降的问题。通过结合多视角运动信息和文本描述，提升模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.22697](https://arxiv.org/pdf/2605.22697)**

> **作者:** Yannick Porto; Renato Martins; Thomas Chalumeau; Cedric Demonceaux
>
> **备注:** Accepted to ICPR 2026. Code and trained models available at: this https URL
>
> **摘要:** Robustness to domain changes is a key capability for effective deployment of human action recognition systems in real-world scenarios, where action categories at inference can present important domain shifts or even unseen actions from training. In this context, improving the recognition capabilities of Zero-Shot Action Recognition models (ZSAR), without requiring strong annotation efforts, remains a central challenge. Most ZSAR approaches assume that actions are observed under geometric conditions similar to those seen during training. In practice, variations in human body orientation and camera viewpoint add a significant domain gap in ZSAR, substantially limiting generalization to novel action-motion combinations. In this context, this paper presents a novel orientation-aware action recognition approach with improved cross-domain capabilities. Our approach combines motion cues of multiple camera viewpoints and text descriptions of human actions in the training phase. We present a new orientation-aware motion encoding network to learn different motion features, and adapt a specific orientation-aware text prompt to match the corresponding features at inference. Extensive experiments demonstrate that the proposed method consistently improves ZSAR performance across different recognition benchmarks, outperforming recent state-of-the-art zero-shot approaches on NTU-RGB+D, BABEL, NW-UCLA, and on two surveillance datasets. In addition, the learned representations exhibit strong transfer learning capabilities, yielding competitive performance on both cross-domain and same-domain recognition of seen actions. Code and trained models are available at: this https URL
>
---
#### [new 057] SegCompass: Exploring Interpretable Alignment with Sparse Autoencoders for Enhanced Reasoning Segmentation
- **分类: cs.CV; cs.LG; cs.MM; eess.IV**

- **简介: 该论文提出SegCompass，解决推理分割中的可解释性问题。通过稀疏自编码器建立显式对齐路径，提升模型透明度与分割精度。**

- **链接: [https://arxiv.org/pdf/2605.22658](https://arxiv.org/pdf/2605.22658)**

> **作者:** Zhenyu Lu; Liupeng Li; Jinpeng Wang; Haoqian Kang; Yan Feng; Ke Chen; Yaowei Wang
>
> **备注:** Accepted by CVPR 2026. 15 pages, 9 figures, 6 tables
>
> **摘要:** While large language models provide strong compositional reasoning, existing reasoning segmentation pipelines fail to transparently connect this reasoning to visual perception. Current methods, such as latent query alignment, are end-to-end yet opaque "black boxes". Conversely, textual localization readout is merely readable, not truly interpretable, often functioning as an unconstrained post-hoc step. To bridge this interpretability gap, we propose SegCompass, an end-to-end model that leverages a Sparse Autoencoder (SAE) to forge an explicit, interpretable, and differentiable alignment pathway. Given an image-instruction pair, SegCompass first generates a chain-of-thought (CoT) trace. The core of our method is an SAE that maps both the CoT and visual tokens into a shared, high-dimensional sparse concept space. A query codebook selects salient concepts from this space, which are then spatially grounded by a slot mapper into a multi-slot heatmap that guides the final mask decoder. The entire model is trained jointly, unifying reinforcement learning for the reasoning path with standard segmentation supervision. This SAE-driven interface provides a "white-box" connection that is significantly more traceable than latent queries and more coherent than textual readouts. Extensive experiments on five challenging benchmarks demonstrate that SegCompass matches or surpasses state-of-the-art performance. Crucially, our visual and quantitative analyses show a strong correlation between the quality of the learned sparse concepts and final mask accuracy, confirming that SegCompass achieves superior results through its enhanced and inspectable alignment. Code is available at this https URL.
>
---
#### [new 058] COCOTree: A Dataset and Benchmark for Open Tree-Structured Visual Decomposition
- **分类: cs.CV**

- **简介: 该论文提出COCOTree，解决开放树状视觉分解任务，通过自动化生成大量标注数据，构建大规模基准，提升结构化图像分析的精度与灵活性。**

- **链接: [https://arxiv.org/pdf/2605.22068](https://arxiv.org/pdf/2605.22068)**

> **作者:** Junhyub Lee; Seunghun Chae; Hyosu Kim
>
> **摘要:** We formalize and enable the task of open tree decomposition, which segments an image into hierarchical trees of visual components with unconstrained granularity and flexibility. Specifically, we provide the foundation benchmark for this new paradigm with the following three key contributions. First, we overcome the prohibitively high cognitive and physical bottlenecks of manual annotation by developing a fully automated generation pipeline that synergizes the semantic reasoning of Large Vision-Language Models (LVLMs) with the precise geometric grounding of SAM 3. Second, leveraging this pipeline, we construct COCOTree, a massive-scale benchmark featuring over 21K images and 1.8M structural nodes. By embracing an open-vocabulary space of over 3.5K unique labels, it successfully captures the long-tail distribution of complex physical assemblies. Notably, rigorous human evaluation confirms our generated annotations demonstrate strong alignment with human structural judgment. Third, we establish a standardized evaluation protocol by proposing the Open Tree Quality (OTQ) metric, which jointly assesses mask precision, label accuracy, and structural consistency. We release our dataset and benchmark code at this https URL.
>
---
#### [new 059] Translating Signals to Languages for sEMG-Based Activity Recognition
- **分类: cs.CV**

- **简介: 该论文属于sEMG信号活动识别任务，旨在提升识别准确性。通过将sEMG信号转换为语言，利用大语言模型进行活动识别，提出LLM-sEMG框架。**

- **链接: [https://arxiv.org/pdf/2605.22403](https://arxiv.org/pdf/2605.22403)**

> **作者:** Ming Wang; Haoxuan Qu; Qiuhong Ke; Wei Zhou; Hossein Rahmani; Jun Liu
>
> **摘要:** Surface electromyography (sEMG) signal-based activity recognition has attracted increasing research attention in recent years. To develop accurate sEMG signal-based activity recognizers, numerous approaches have been proposed. Some studies focus on designing larger and more expressive model architectures to enhance the representational capacity of sEMG signals, while others aim to enrich model priors through large-scale pretraining, thereby improving recognition performance. Recently, large language models (LLMs) have shown remarkable generalization and reasoning capabilities in natural language processing, whose implicit knowledge, learned from extensive linguistic descriptions of actions, opens new possibilities for interpreting sEMG signals and inferring activity intentions. Motivated by this, we propose LLM-sEMG, a novel framework that leverages LLMs as sEMG activity recognizers. Within this framework, we design a language-oriented mapping mechanism that converts continuous sEMG sequences into sEMG language, integrating several strategies to further facilitate the signal-to-language mapping process. Extensive experiments demonstrate that the proposed framework achieves highly accurate sEMG signal-based activity recognition using large language models.
>
---
#### [new 060] Robustness of breast lesion segmentation under MRI undersampling improves with k-space-aware deep learning
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像分割任务，旨在提升乳腺病变在MRI欠采样和噪声下的分割鲁棒性。通过设计k空间感知的深度学习模型，对比不同方法在不同数据条件下的表现，验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2605.22327](https://arxiv.org/pdf/2605.22327)**

> **作者:** Lukas T. Rotkopf; Marco Schlimbach; Julius C. Holzschuh; Heinz-Peter Schlemmer; Jens Kleesiek; Moritz Rempe
>
> **摘要:** Purpose: To assess whether breast lesion segmentation can be learned directly from acquired MRI k-space, and whether doing so improves robustness when data are accelerated or noisy. Materials and Methods: This retrospective study used public breast dynamic contrast-enhanced MRI (DCE-MRI) datasets with acquired and synthetic k-space, together with a within-dataset synthetic control. We compared four 3D U-Net variants: a hybrid k-space-to-image model, a native k-space model, and magnitude and complex image-space baselines. Models were evaluated under increasing undersampling and added complex Gaussian k-space noise. The primary outcome was patient-level Dice similarity coefficient under cross-validation, with the hybrid model prespecified as the main comparison against the magnitude image-space baseline. Results: At full sampling, the hybrid and image-space models performed similarly. As acceleration increased, the hybrid model retained substantially more segmentation accuracy and significantly outperformed the magnitude image-space baseline across moderate to high undersampling levels. The same pattern was observed when noise was added directly to k-space: the hybrid model degraded more slowly, whereas the image-space baseline failed under heavier noise. This advantage was reproduced in the within-dataset synthetic control. Feature analysis suggested that the k-space stage and image-space stage played complementary roles, with frequency-domain filtering concentrated before image-domain lesion localization. Conclusion: K-space-aware deep learning improves the robustness of breast lesion segmentation under MRI undersampling and k-space noise, while matching image-space methods at full sampling.
>
---
#### [new 061] One Sentence, One Drama: Personalized Short-Form Drama Generation via Multi-Agent Systems
- **分类: cs.CV**

- **简介: 该论文属于短剧生成任务，解决叙事节奏、空间一致性和质量控制问题。提出多智能体框架，通过结构化模块和迭代优化生成高质量短剧。**

- **链接: [https://arxiv.org/pdf/2605.22144](https://arxiv.org/pdf/2605.22144)**

> **作者:** Yufei Shi; Weilong Yan; Naixuan Huang; Yucheng Chen; Chenyu Zhang; Tao He; Si Yong Yeo; Ming Li
>
> **摘要:** Existing approaches for digital short-drama production typically rely on one-shot LLM generated scripts and loosely coupled pipelines, which fail to satisfy three key requirements of short-drama generation: (1) narrative pacing, resulting in weak hooks, insufficient escalation, and unattractive endings; (2) spatial consistency, leading to drifting scene layouts and inconsistent character positions across clips; and (3) production-level quality control, requiring extensive manual review and correction across script and visual stages. We present One Sentence, One Drama, a hierarchical multi-agent framework that transforms a user's single-sentence idea into a fully produced short drama through structured intermediate modules and iterative refinement. Our approach is built upon three key components: (1) a multi-agent debate-based story generation module that enforces short-drama pacing and narrative coherence; (2) a 3D-grounded first-frame generation mechanism that establishes a shared spatial reference for consistent character positioning and scene layout across clips; and (3) multi-stage reviewer loops that perform comprehensive error detection and targeted revision across script, visual, and video generation stages. We also introduce scene-level BGM matching and scene transition planning to improve the audience's immersive experience. To systematically evaluate this task, we introduce Short-Drama-Bench, a benchmark that extends standard video quality metrics with short-drama-specific criteria. Experimental results demonstrate that our method significantly outperforms existing pipelines in narrative quality, cross-clip consistency, and overall viewing experience.
>
---
#### [new 062] Moment-Reenacting: Inverse Motion Degradation with Cross-shutter Guidance
- **分类: cs.CV**

- **简介: 该论文属于图像去模糊任务，解决运动退化问题。通过联合利用全局快门模糊与滚动快门失真，提出统一框架进行运动逆过程重建。**

- **链接: [https://arxiv.org/pdf/2605.22423](https://arxiv.org/pdf/2605.22423)**

> **作者:** Ji Xiang; Lin Guixu; Yin Zhengwei; Zhao Jiancheng; Zheng Yinqiang
>
> **备注:** Accepted by TPAMI
>
> **摘要:** Motion degradation, manifested as blur in global shutter (GS) images or rolling shutter (RS) distortion in RS counterparts, remains a fundamental challenge in computational imaging, especially under fast motion or low-light conditions. While prior works have treated blur decomposition and RS temporal super-resolution as separate tasks, this separation fails to exploit their intrinsic complementarity. In this paper, we propose a unified framework to invert motion degradation and reenact imaging moment by jointly leveraging the complementary characteristics of GS blur and RS distortion. To this end, we introduce a novel dual-shutter setup that captures synchronized blur-RS image pairs and demonstrate that this combination effectively resolves temporal and spatial ambiguities inherent in both modalities. For allowing flexible performance-cost trade-offs, we further extend this dual-shutter setup to a stereo Blur-RS configuration with a narrow baseline. In addition, we construct a triaxial imaging system to collect a real-world dataset with aligned GS-RS pairs and ground-truth high-speed frames, enabling robust training and evaluation beyond synthetic data. Our proposed network explicitly disentangles motion into context-aware and temporally-sensitive representations via a dual-stream motion interpretation module, followed by a self-prompted frame reconstruction stage. Extensive experiments validate the superiority and generalizability of our approach, establishing a new paradigm for realistic high-speed video reconstruction under complex motion degradations. Codes and more resources are available at this https URL.
>
---
#### [new 063] Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning for Video Temporal Grounding
- **分类: cs.CV**

- **简介: 该论文属于视频时间定位任务，解决现有方法在时间戳生成上的不稳定性问题。提出F2G框架，通过事件识别与边界测量分离提升定位准确性。**

- **链接: [https://arxiv.org/pdf/2605.21973](https://arxiv.org/pdf/2605.21973)**

> **作者:** Zelin Zheng; Xinyan Liu; Ruixin Li; Antoni B. Chan; Guorong Li; Qingming Huang; Laiyun Qing
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Current Video-LLM approaches for Video Temporal Grounding (VTG) typically rely on direct timestamp generation from an unstructured visual-token stream, often leading to brittle numerics and inconsistent boundaries. To address this, we propose Foresee-to-Ground (F2G), a framework that reformulates VTG as a verifiable Identify-then-Measure problem. F2G integrates Predictive Temporal Perception with Evidence-Driven Reasoning: it learns boundary-sensitive temporal representations to build a video-wide evidence pool of candidate event segments, and exposes these segments to the LLM as citable evidence units that bind boundary prediction to explicit event hypotheses. By decoupling event identification from precise boundary measurement, F2G stabilizes grounding and makes predictions verifiable. Extensive experiments demonstrate that F2G consistently improves grounding accuracy across diverse benchmarks, transfers robustly across different Video-LLM backbones, and preserves general video understanding capabilities.
>
---
#### [new 064] Lens: Rethinking Training Efficiency for Foundational Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文提出Lens模型，解决文本到图像生成的训练效率问题。通过优化数据和架构，提升性能并减少计算需求。**

- **链接: [https://arxiv.org/pdf/2605.21573](https://arxiv.org/pdf/2605.21573)**

> **作者:** Dong Chen; Fangyun Wei; Ziyu Wan; Dongdong Chen; Jiawei Zhang; Jinjing Zhao; Sirui Zhang; Yang Yue; Zhiyang Liang; Baining Guo; Chong Luo; Jianmin Bao; Ji Li; Lei Shi; Qinhong Yang; Xiuyu Wu; Xuelu Feng; Yan Lu; Yanchen Dong; Yitong Wang; Yunuo Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** We introduce Lens, a 3.8B-parameter T2I model that achieves performance competitive with, and in several cases surpassing, state-of-the-art models with more than 6B parameters across various benchmarks, while requiring significantly less training compute. For example, Lens requires only about 19.3% of the training compute used by Z-Image. The training efficiency of Lens stems from two key strategies beyond its compact model size. First, we maximize data information density per training batch by (i) training on Lens-800M, a dataset of 800M densely captioned image-text pairs whose captions are generated by GPT-4.1 and contain approximately 109 words on average, providing richer semantic supervision than conventional short captions, and (ii) constructing each batch from images with multiple resolutions and diverse aspect ratios, thereby enlarging the effective visual coverage of each optimization step. Second, we improve convergence speed through careful architectural choices, including adopting a semantic VAE that provides better latent representations and employing a strong language encoder that accelerates optimization while enabling multilingual generalization from English-only training data. After pre-training, we apply RL with taxonomy-driven prompts (Lens-RL-8K) and structured reward rubrics to suppress artifacts and improve visual quality, a reasoner module with training-free system prompt search to better align user requests with the model, and distillation-based acceleration for 4-step inference. Through efficient training and systematic optimization, Lens generalizes to arbitrary aspect ratios from 1:2 to 2:1 and resolutions up to 1440^2, and supports prompts in several commonly used languages. Thanks to its compact size, Lens generates a 1024^2 image in 3.15 seconds on a single NVIDIA H100 GPU, while its distilled turbo version performs 4-step generation in 0.84 seconds.
>
---
#### [new 065] QuantSR+: Pushing the Limit of Quantized Image Super-Resolution Networks
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，解决低比特量化导致的性能下降问题。提出QuantSR+框架，通过优化量化、网络设计和训练策略，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.22351](https://arxiv.org/pdf/2605.22351)**

> **作者:** Haotong Qin; Xudong Ma; Xianglong Liu; Jie Luo; Jinyang Guo; Michele Magno; Yulun Zhang
>
> **摘要:** Low-bit quantization is widely used to compress super-resolution (SR) models and reduce storage and computation costs for deployment on resource-limited devices. However, when SR models are pushed to ultra-low precision (2-4 bits), performance can drop sharply due to diminished representational capacity and the detail-sensitive nature of SR. To address these issues, we propose QuantSR+, a unified framework that improves quantization operators, network design, and training optimization, achieving better trade-offs between accuracy and efficiency than prior low-bit SR methods. QuantSR+ mainly relies on three technical contributions: (1) Redistribution-driven Bit Determination (RBD), which reshapes quantization distributions in both forward and backward passes to preserve representation fidelity; (2) Quantized Slimmable Architecture (QSA), which begins with an over-parameterized model and progressively prunes less critical blocks to meet efficiency budgets while pushing the accuracy performance; and (3) Slimming-guided Function-localized Distillation (SFD), which enforces block-aware feature alignment via a direct loss and a progressive, function-local training schedule to capture quantization effects better and speed up convergence. Extensive experiments show that QuantSR+ achieves state-of-the-art performance against both specialized quantized SR methods and generic quantization approaches. For SwinIR-S on Urban100 (x4), it improves PSNR by 0.29 dB over the 2-bit SOTA baseline. Meanwhile, it delivers strong efficiency gains at 2-bit, reducing operations by up to 87.9% and storage by 89.4%. QuantSR+ is effective for both convolutional and transformer-based SR models, indicating broad applicability.
>
---
#### [new 066] Learning Spatiotemporal Sensitivity in Video LLMs via Counterfactual Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频语言模型任务，旨在解决模型依赖静态线索而非动态信息的问题。通过引入对抗性强化学习框架CRPO，提升模型对时空动态的敏感性。**

- **链接: [https://arxiv.org/pdf/2605.21988](https://arxiv.org/pdf/2605.21988)**

> **作者:** Dazhao Du; Jian Liu; Jialong Qin; Tao Han; Bohai Gu; Fangqi Zhu; Yujia Zhang; Eric Liu; Xi Chen; Song Guo
>
> **备注:** Project website: this https URL
>
> **摘要:** Video large language models (Video LLMs) achieve strong benchmark accuracy, yet often answer video questions through shortcuts such as single-frame cues and language priors rather than by tracking spatiotemporal dynamics. This issue is exacerbated in RL post-training, where correctness-only rewards can further reinforce shortcut policies that obtain high reward without tracking video dynamics. We address this by asking a controlled counterfactual question: if the visual world changed while the question remained fixed, should the answer change or stay the same? Based on this view, we propose \textbf{Counterfactual Relational Policy Optimization (CRPO)}, a dual-branch RL framework for improving \emph{spatiotemporal sensitivity}. CRPO constructs counterfactual videos through horizontal flips and temporal reversals, trains on both original and counterfactual branches, and introduces a \textbf{Counterfactual Relation Reward (CRR)} between their answers. CRR encourages answers to change for dynamic questions and remain unchanged for static questions. This cross-branch constraint makes it difficult for shortcut policies to be consistently rewarded across both branches. To evaluate this property, we introduce \textbf{DyBench}, a paired counterfactual video benchmark with 3,014 videos covering reversible dynamics, moving direction, and event sequence, together with a strict pair-accuracy metric that prevents fixed-answer shortcuts from inflating scores. Experiments show that CRPO outperforms prior RL methods on spatiotemporal-sensitive evaluations while maintaining competitive general video performance. On Qwen3-VL-8B, CRPO improves DyBench P-Acc by +7.7 and TimeBlind I-Acc by +8.2 over the base model, indicating improved spatiotemporal sensitivity rather than stronger reliance on static shortcuts. The project website can be found at this https URL .
>
---
#### [new 067] Balancing Uncertainty and Diversity of Samples: Leveraging Diversity of Least, High Confidence Samples for Effective Active Learning
- **分类: cs.CV**

- **简介: 该论文属于主动学习任务，旨在解决标注数据稀缺问题。通过结合不确定性和多样性，提出新的采样方法以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22169](https://arxiv.org/pdf/2605.22169)**

> **作者:** Vipul Arya; S.H. Shabbeer Basha; Srikrishna U N; Sunainha Vijay; Snehasis Mukherjee
>
> **摘要:** Deep learning models, including Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), have achieved state-of-the-art performance on various computer vision tasks such as object classification, detection, segmentation, generation, and many more. However, these models are data-hungry as they require more training data to learn millions or billions of parameters. Especially for supervised learning tasks, curating a large number of labeled samples for model training is an expensive and time-consuming task. Active Learning (AL) has been used to address this problem for many years. Existing active learning methods aim at choosing the samples for annotation from a pool of unlabeled samples that are either diverse or uncertain. Choosing such samples may hinder the model's performance as we pool based on one dimension, i.e., either diverse or uncertain. In this paper, we propose four novel hybrid sampling methods for pooling both easy and hard samples, which are also diverse. To verify the efficacy of the proposed methods, extensive experiments are conducted using high and low-confidence samples separately. We observe from our experiments that the proposed hybrid sampling method, Least Confident and Diverse (LCD), consistently performs better compared to state-of-the-art methods. It is observed that selecting uncertain and diverse instances helps the model learn more distinct features. The codes related to this study will be available at this https URL.
>
---
#### [new 068] LVDrive: Latent Visual Representation Enhanced Vision-Language-Action Autonomous Driving Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LVDrive，一种增强潜在视觉表示的视觉-语言-动作模型，用于自动驾驶。针对现有模型依赖稀疏动作监督和过度关注像素重建的问题，通过引入未来场景预测任务，提升场景理解与轨迹生成能力。**

- **链接: [https://arxiv.org/pdf/2605.22089](https://arxiv.org/pdf/2605.22089)**

> **作者:** Xiaodong Mei; Diankun Zhang; Hongwei Xie; Guang Chen; Hangjun Ye; Dan Xu
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising framework for end-to-end autonomous driving. However, existing VLAs typically rely on sparse action supervision, which underutilizes their powerful scene understanding and reasoning capabilities. Recent attempts to incorporate dense visual supervision via world modeling often overemphasize pixel-level image reconstruction, neglecting semantically meaningful scene representation learning. In this work, we propose LVDrive, a Latent Visual representation enhanced VLA framework for autonomous driving. LVDrive introduces a future scene prediction task into the VLA paradigm, where future representations are learned entirely in a high-level latent space under auxiliary supervision from a pretrained vision backbone. Departing from inefficient autoregressive generation, we jointly model future scene and motion prediction within a unified embedding space, processed in a single forward pass to conduct the future-aware reasoning. We further design a two-stage trajectory decoding strategy that explicitly leverages the learned latent future representations to refine trajectory generation. Extensive experiments on the challenging Bench2Drive benchmark demonstrate that LVDrive achieves significant improvements in closed-loop driving performance, outperforming both action supervised methods and image-reconstruction-based world model approaches.
>
---
#### [new 069] MOTOR: A Multimodal Dataset for Two-Wheeler Rider Behavior Understanding
- **分类: cs.CV**

- **简介: 该论文提出MOTOR数据集，用于研究两轮车骑行行为，解决两轮车安全分析不足的问题。整合多模态数据，提升行为识别与合法性分类性能。**

- **链接: [https://arxiv.org/pdf/2605.22550](https://arxiv.org/pdf/2605.22550)**

> **作者:** Varun A. Paturkar; Shankar Gangisetty; C.V. Jawahar
>
> **摘要:** Two-wheelers account for a disproportionately high share of road fatalities in the Global South. Research on two-wheeler rider behavior, however, lags far behind four-wheelers, where multimodal datasets have driven major advances in Advanced Driver Assistance Systems (ADAS). To address this gap, we present the MOtorized TwO-wheeler Rider (MOTOR) dataset, the first large-scale, multi-view, multimodal resource dedicated to two-wheelers in dense, unstructured traffic. MOTOR comprises 1,629 sequences (25+ hours of video data) collected from 16 riders and integrates synchronized front, rear, and helmet videos, rider eye-gaze from wearable trackers, on-road audio, and telemetry (GPS, accelerometer, gyroscope). Rich annotations capture traffic context, rider state, 12 riding maneuvers spanning conventional and unconventional behaviors, and legality labels (Legal, Illegal, Unspecified). We benchmark rider behavior recognition and maneuver legality classification using state-of-the-art video action recognition backbones (CNN and Transformer-based), extended with multimodal fusion, and find that combining RGB, gaze, and telemetry consistently yields the best performance. MOTOR thus provides a unique foundation for advancing safety-critical understanding of two-wheeler riding. It offers the research community a benchmark to develop and evaluate models for behavior analysis, legality-aware prediction, and intelligent transportation systems. Dataset and code is available at https: //varuniiith.this http URL
>
---
#### [new 070] SO-Mamba: State-Ownership Mamba for Unrolled MRI Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于MRI重建任务，旨在解决加速重建中细节恢复与结构一致性问题。提出SO-Mamba模型，通过状态所有权机制优化Mamba结构，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2605.22031](https://arxiv.org/pdf/2605.22031)**

> **作者:** Pengcheng Fang; Hongli Chen; Fangfang Tang; Feng Liu; Xiaohao Cai; Shanshan Shan
>
> **摘要:** Accelerated MRI reconstruction requires recovering missing details while preserving anatomically coherent structures across large spatial regions. State-space models such as Mamba provide efficient long-range modeling, making them attractive learned regularizers for unrolled reconstruction. However, in a data-consistency-coupled unrolled solver, different stages operate on different reconstruction iterates, where the resident carrier should preserve coherent reconstruction content across stages while stage-dependent non-resident evidence is tied to the current update. Treating these roles uniformly can place persistent resident-carrier evidence and update-dependent non-resident evidence into the same recurrent content route. We therefore propose SO-Mamba, a state-ownership Mamba regularizer that assigns reconstruction evidence within each Mamba stage to recurrent residency, state-interface access, and non-state output correction. SO-Mamba implements this ownership rule with a State-Ownership Router (SOR), which constructs a resident carrier for recurrent content and routes non-resident evidence to affine modulation of the B/C state interfaces and an output correction outlet. The resident carrier supplies the Mamba content route, while the non-resident evidence stream adapts the state interfaces and contributes through the output outlet without entering the recurrent content route. We further introduce a two-level outer-band leakage diagnostic that separates hidden-state storage from readout expression by measuring outer-band energy in the selective-scan state trajectory and the post-scan Mamba readout. Experiments on five public MRI reconstruction benchmarks spanning diverse anatomies, sampling patterns, and coil configurations show that SO-Mamba consistently improves over CNN-, Transformer-, and Mamba-based baselines with competitive computational efficiency.
>
---
#### [new 071] No Pose, No Problem in 4D: Feed-Forward Dynamic Gaussians from Unposed Multi-View Videos
- **分类: cs.CV**

- **简介: 该论文属于动态3D重建任务，解决多视角、无姿态、动态场景的重建问题。提出NoPo4D系统，通过速度分解和双向运动编码实现高效准确的重建。**

- **链接: [https://arxiv.org/pdf/2605.22190](https://arxiv.org/pdf/2605.22190)**

> **作者:** Matteo Balice; Yanik Kunzi; Chenyangguang Zhang; Matteo Matteucci; Marc Pollefeys; Sungwhan Hong
>
> **备注:** this https URL
>
> **摘要:** Recent feed-forward 3D gaussian splatting methods have made dramatic progress on individual aspects of 3D scene reconstruction, but no existing method jointly addresses dynamic content, multi-view input, and unknown camera poses in a single feed-forward pass. Methods that handle dynamics either require accurate camera poses or accept only monocular input; pose-free multi-view methods address only static scenes; and per-scene optimization methods bridge some of these gaps but at minutes-to-hours cost per scene. We introduce NoPo4D, the first feed-forward system that addresses this empty quadrant. Building on a pretrained geometry backbone and recent 4D Gaussian frameworks, NoPo4D introduces a velocity decomposition that splits Gaussian motion into per-pixel image-plane shifts and depth changes, allowing direct supervision from pseudo ground-truth optical flow on the 2D component. This sidesteps both the differentiable rendering that couples prior posed methods to pose accuracy and the 3D motion ground truth that prior pose-free methods require. The system is rounded out by a bidirectional motion encoder for cross-view and cross-frame feature aggregation, and view-dependent opacity that mitigates cross-view and cross-timestep Gaussian misalignments. On four multi-view dynamic benchmarks, NoPo4D consistently outperforms prior feed-forward baselines, and with an optional post-optimization stage surpasses per-scene optimization methods, while running orders of magnitude faster.
>
---
#### [new 072] Matching with Deliberation: Test-Time Evolutionary Hierarchical Multi-Agents for Zero-Shot Compositional Image Retrieval
- **分类: cs.CV**

- **简介: 该论文属于零样本组合图像检索任务，解决视觉连续性和语义执行的问题。提出PDF框架，通过多智能体和自进化机制提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.22478](https://arxiv.org/pdf/2605.22478)**

> **作者:** Xingtian Pei; Yukun Song; Changwei Wang; Shunpeng Chen; Rongtao Xu; Shibiao Xu
>
> **备注:** 10 pages, 5 figures,4 tables
>
> **摘要:** Zero-Shot Compositional Image Retrieval (ZS-CIR) requires both preserving the visual continuity of the reference image and faithfully executing the semantic variables specified in the modification text, which constitutes the core challenge of the task. Existing methods often suffer from Perception Myopia in a single space, or fall into Logic Drift in iterative collaboration due to the perception ceiling of the underlying retriever. To address this issue, we propose a one-stop hierarchical Perception-to-Deliberation Framework (PDF), which, to the best of our knowledge, is the first to introduce experience self-evolution and Test-Time Scaling Law (TTS) into ZS-CIR. Relying on a hierarchical multi-agent architecture, PDF first utilizes an Intent Routing Manager to dynamically dispatch multi-view Worker perception signals based on modification intents to construct a high-recall candidate pool. Subsequently, the Decision Manager combines a Training-free Reasoning Policy Distillation mechanism with a Tournament-style TTS strategy to achieve self-evolving fine-grained reasoning, yielding the final retrieval results. Experimental results demonstrate that PDF achieves SOTA performance on three benchmark datasets: CIRR, CIRCO, and FashionIQ. This study indicates that experience-driven self-evolution and TTS represent a highly promising and scalable path for achieving zero-shot fine-grained multimedia retrieval. The code will be made publicly available upon acceptance.
>
---
#### [new 073] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决分布偏移下的行为表示学习问题。提出BehaviorVLA框架，通过时序一致的行为表示提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.22671](https://arxiv.org/pdf/2605.22671)**

> **作者:** Bing Hu; Zaijing Li; Rui Shao; Junda Chen; April Hua Liu; Wei-Shi Zheng; Liqiang Nie
>
> **摘要:** Vision-Language-Action (VLA) models often suffer from performance degradation under distribution shifts, as they struggle to learn generalized behavior representations across varying environments. While existing approaches attempt to construct behavior representations through action-centric latent variables, they are often limited by short-horizon temporal fragmentation and static execution-alignment, leading to inconsistent behaviors in complex scenarios. To address these limitations, we propose \textbf{BehaviorVLA}, a framework that facilitates robust manipulation through the learning of a temporally coherent behavioral representations. Our approach features two symmetric components: (1) the \textbf{Visuomotor Behavior Encoder (VBE)}, which utilizes a causal Mamba-based architecture to aggregate long-horizon trajectory information into a unified behavior representation; and (2) the \textbf{Phase-conditioned Behavior Decoder (PBD)}, which decodes this representation into precise actions by dynamically aligning task-level priors with real-time execution progress. Experiments on RoboTwin 2.0, LIBERO, and CALVIN demonstrate state-of-the-art success rates of 58\%, 98\%, and 4.36 (this http URL), respectively. Notably, in real-world sim-to-real transfer, BehaviorVLA matches the performance of OpenVLA-OFT using only 50\% of the demonstration data, showcasing its superior data efficiency and generalization.
>
---
#### [new 074] From Recognition to Reasoning: Benchmarking and Enhancing MLLMs on Real-World Receipt Document Understanding
- **分类: cs.CV**

- **简介: 该论文聚焦于视觉信息抽取任务，解决现有基准不足的问题。构建了ReceiptBench基准并提出两阶段训练框架，提升模型在结构化理解上的表现。**

- **链接: [https://arxiv.org/pdf/2605.22413](https://arxiv.org/pdf/2605.22413)**

> **作者:** Yandi Wang; Libin Zhan; Ziwei Huang; Tiancheng Luo; Yuxuan Jiang; Wang Dong; Leilei Gan; Jun Chen
>
> **摘要:** Extracting structured information from visual documents (Visual Information Extraction, VIE) is a cornerstone of business automation. While recent Multimodal Large Language Models (MLLMs) have shown promising capabilities, existing benchmarks suffer from critical limitations in scale and realism, lack semantic granularity, and fail to cover diverse document types. To bridge this gap, we introduce ReceiptBench, a large-scale, human-annotated benchmark consisting of 10k diverse receipts, organizing information extraction into four hierarchical sub-tasks: (1) Basic Perception for raw text spotting, (2) Format Normalization for strictly following standardization instructions, (3) Semantic Reasoning for inferring implicit attributes from context, and (4) Structure Parsing for handling nested line items. Furthermore, we propose a two-stage training framework incorporating Metric-Aware Group Relative Policy Optimization (GRPO), which translates rigorous evaluation constraints into reinforcement learning signals to enhance structural consistency. Extensive experiments demonstrate that our method yields state-of-the-art performance, surpassing leading proprietary models on complex reasoning tasks. We release our datasets and code at this https URL.
>
---
#### [new 075] Accelerating Vision Foundation Models with Drop-in Depthwise Convolution
- **分类: cs.CV**

- **简介: 该论文属于视觉模型加速任务，旨在降低ViT的推理成本。通过引入深度卷积层替换部分注意力头，实现速度提升且性能损失小。**

- **链接: [https://arxiv.org/pdf/2605.22132](https://arxiv.org/pdf/2605.22132)**

> **作者:** Carmelo Scribano; Mohammad Mahdi; Nedyalko Prisadnikov; Yuqian Fu; Giorgia Franchini; Danda Pani Paudel; Marko Bertogna; Luc Van Gool
>
> **备注:** Accepted at ICPR 2026
>
> **摘要:** Pretrained vision foundation models deliver strong performance across tasks with limited fine-tuning. However, their Vision Transformer (ViT) backbones impose high inference costs, limiting deployment on resource-constrained devices. In this work, we accelerate large-scale pretrained ViTs while preserving their feature extraction capabilities by exploiting the intrinsic convolution-like behavior of some attention heads. Specifically, we introduce an efficient depthwise convolution-based layer that serves as a drop-in replacement for these heads. Additionally, we propose simple strategies to identify which heads can be replaced and introduce a fine-tuning procedure that recovers downstream task performance. Across both image classification and segmentation tasks, our method achieves 17-20\% percent inference speedup with minimal performance degradation. We validate the approach through detailed derivations, extensive experiments, and efficiency benchmarks. The reference implementation is publicly available.
>
---
#### [new 076] Ultra-High-Definition Image Quality Assessment via Graph Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，解决UHD图像盲评估中计算成本高和局部与全局关系弱的问题。通过图表示学习建模区域间结构依赖，提升质量预测精度。**

- **链接: [https://arxiv.org/pdf/2605.22192](https://arxiv.org/pdf/2605.22192)**

> **作者:** Shaode Yu; Enqi Chen; Ming Huang; Xuemin Ren; Songnan Zhao; Zhicheng Zhang; Qiurui Sun
>
> **摘要:** Blind image quality assessment (BIQA) for ultrahighdefinition (UHD) images remains challenging because native-resolution inference is computationally expensive, whereas aggressive resizing or isolated cropping may suppress scale-sensitive distortions and weaken the relationship between local artifacts and global scene context. This paper aims to improve UHD-BIQA by explicitly modeling the structural dependencies among sampled image regions rather than treating them as independent views, and a graph representation learning framework UHD-GCN-BIQA is proposed. The framework samples aspect-ratio-aligned patches from each UHD image, encodes them as graph nodes, and constructs a hybrid k-nearest-neighbor graph using spatial proximity and feature similarity. Residual graph convolution is used to propagate contextual information across regions, and gated attention pooling aggregates patchlevel evidence into an imagelevel quality prediction. An exponential moving average normalized multiobjective loss function is adopted to stabilize the joint optimization of regression, correlation, and ranking objectives. Experiments on the UHD-IQA benchmark show that UHD-GCN-BIQA achieves PLCC = 0.7784, SRCC = 0.8019, and RMSE = 0.0519, obtaining competitive correlation performance and the lowest RMSE among the compared methods. These results indicate that graph-based region relation modeling is effective for UHD image quality assessment, particularly for improving absolute quality score estimation under high-resolution visual content.
>
---
#### [new 077] Synthetic Data Alone is Enough? Rethinking Data Scarcity in Pediatric Rare Disease Recognition
- **分类: cs.CV**

- **简介: 该论文属于儿科罕见病识别任务，旨在解决数据稀缺问题。通过使用合成数据进行训练，验证其在 pediatric 罕见病识别中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.22767](https://arxiv.org/pdf/2605.22767)**

> **作者:** Ganlin Feng; Yuxi Long; Erin Lou; Lianghong Chen; Zihao Jing; Pingzhao Hu; Wei Xu
>
> **备注:** CVPR 2026 CV4CHL workshop
>
> **摘要:** Children with rare genetic diseases often exhibit distinctive facial phenotypes, yet developing computer vision systems for early diagnosis remains challenging due to extreme data scarcity, privacy constraints, and limited data sharing in pediatric settings. These challenges not only hinder automated diagnosis but also restrict the availability of visual resources for clinical genetic counseling. While prior work has shown that synthetic data can augment real datasets and preserve phenotype-level semantics, it remains unclear whether synthetic data alone is sufficient for learning in ultra-low-resource pediatric settings. In this work, we study the synthetic-only regime for pediatric rare disease recognition. Under a controlled experimental setup, models are trained exclusively on phenotype-aware synthetic facial images at increasing scales. We find that synthetic-only training achieves performance comparable to real-data-only baselines at sufficient scale across multiple backbones, suggesting that high-fidelity synthetic data can approximate clinically meaningful distributions. These findings together further enable the use of synthetic pediatric facial images as privacy-preserving resources for genetic education and counseling, supporting clinician training and patient communication. Our results highlight the potential of computer vision to improve data efficiency and expand accessible visual tools in children's healthcare.
>
---
#### [new 078] Zero-Shot Temporal Action Localization Through Textual Guidance
- **分类: cs.CV**

- **简介: 该论文属于零样本时序动作定位任务，解决未见动作分类与定位问题。通过引入文本信息增强动作细粒度区分，提出TEGU方法提升定位效果。**

- **链接: [https://arxiv.org/pdf/2605.22201](https://arxiv.org/pdf/2605.22201)**

> **作者:** Benedetta Liberatori; Alessandro Conti; Lorenzo Vaquero; Paolo Rota; Yiming Wang; Elisa Ricci
>
> **备注:** Accepted to FG 2026
>
> **摘要:** Zero-shot temporal action localization (ZS-TAL) consists of classifying and localizing actions in untrimmed videos, where action classes are unseen at training time. Existing work uses Vision and Language Models (VLMs), taking advantage of their strong zero-shot transfer capabilities. Yet, these models face evident challenges with fine-grained action classification, making it difficult to directly use them to distinguish between the presence and absence of an action. Most current methods for ZS-TAL address these challenges by training models on large-scale video datasets, which require annotated data and often result in limited generalization performance. Recently, approaches discarding the use of labeled data have emerged as an alternative. Following this direction, we propose a novel approach, ``Textual Guidance for finer localization of actions in videos'' (TEGU), that compensates for the lack of supervision from training data by exploiting rich textual information derived from large language models and structured text extracted from captions. This additional linguistic context can improve fine-grained discrimination by providing richer cues about fine-grained action differences within videos. We validate the effectiveness of the proposed method by conducting experiments on the THUMOS14 and the ActivityNet-v1.3 datasets. Our results show that, by exploiting rich textual information for improved action localization, TEGU outperforms state-of-the-art ZS-TAL approaches that do not involve training
>
---
#### [new 079] Visual-Advantage On-Policy Distillation for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，解决知识蒸馏中视觉依赖性不足的问题。提出VA-OPD方法，通过视觉优势增强模型对视觉信息的依赖。**

- **链接: [https://arxiv.org/pdf/2605.21924](https://arxiv.org/pdf/2605.21924)**

> **作者:** Ruiqi Liu; Xiaolei Lv; Gengsheng Li; Ximo Zhu; Zhiheng Wang; Zhengbo Zhang; Junkai Chen; Zhiheng Li; Bo Li; Jun Gao; Shu Wu
>
> **摘要:** On-policy knowledge distillation has proven effective for language models, yet its application to vision-language models (VLMs) remains underexplored. We observe that standard on-policy distillation can improve a student's output quality while failing to strengthen its reliance on visual input: on vision-critical tokens, the student's predictions remain largely unchanged whether or not fine-grained visual detail is present, even though the teacher's predictions depend heavily on this http URL make this difference observable, we introduce visual advantage (VA), the token-level log-probability difference when the teacher scores a student-generated rollout with versus without access to fine-grained visual detail. VA is concentrated in a small minority of tokens, and these high-VA tokens are the ones that actually carry the visual supervision signal. This motivates a distillation objective that treats them differently from language scaffolding, so their contribution is not diluted by the abundant surrounding language this http URL propose Visual-Advantage On-Policy Distillation (VA-OPD), which uses VA at two granularities: rollout-level reweighting by trajectory-averaged VA, and token-level KL averaged within high-VA and low-VA groups separately. We train on two math datasets (Geometry3K and ViRL39K) and evaluate on eight benchmarks covering both mathematical reasoning and visual understanding, across three teacher sizes (4B, 8B, and 32B) on the Qwen3-VL family. VA-OPD improves over standard on-policy distillation on every benchmark, with the gain growing monotonically along both the teacher-size and data-scale axes, suggesting that these factors compound consistently.
>
---
#### [new 080] SceneGraphGrounder: Zero-Shot 3D Visual Grounding via Structured Scene Graph Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D视觉定位任务，解决零样本环境下从自然语言定位物体的问题。提出SceneGraphGrounder框架，通过结构化图匹配实现空间一致的语义推理。**

- **链接: [https://arxiv.org/pdf/2605.21788](https://arxiv.org/pdf/2605.21788)**

> **作者:** Xuefei Sun; Xujia Zhang; Brendan Crowe; Doncey Albin; Christoffer Heckman
>
> **摘要:** Zero-shot 3D visual grounding requires localizing objects in unstructured environments from free-form natural language. Recent vision-language model (VLM) approaches achieve promising results but rely on view-dependent reasoning or implicit representations, limiting spatial consistency and interpretability for compositional queries. We propose SceneGraphGrounder, a framework that reformulates 3D grounding as structured graph matching over a reconstructed 3D scene graph. To enable this formulation, we introduce a visual marker prompting strategy that enables a VLM to infer object-object relationships from 2D views, which are subsequently lifted into a persistent 3D scene graph encoding both spatial and semantic relations. Given a query, we construct a query graph and perform constrained alignment with the scene graph, ensuring multi-view consistency and interpretable reasoning. Experiments on the ScanRefer benchmark demonstrate that our method achieves competitive performance among zero-shot approaches, using only RGB-D inputs. We further validate our framework through real-world deployment on a mobile robot, demonstrating robust spatial reasoning in long-horizon physical environments. We will make our code publicly available upon acceptance.
>
---
#### [new 081] Enhancing Multimodal Large Language Models for Safety-Critical Driving Video Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于驾驶视频分析任务，旨在提升MLLM对安全关键事件的识别能力。通过融合视频与传感器数据生成伪标签，优化模型以准确检测和描述高风险动态事件。**

- **链接: [https://arxiv.org/pdf/2605.22185](https://arxiv.org/pdf/2605.22185)**

> **作者:** Tomaso Trinci; Henrique Piñeiro Monteagudo; Leonardo Taccari
>
> **备注:** Accepted at the 2026 IEEE International Conference on Intelligent Transportation Systems (ITSC 2026)
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in general visual understanding. However, their application to safety-critical driving scenarios remains limited by an inability to accurately perceive and reason about rare high-stakes dynamic events, such as collisions or near-collisions. To address this, we introduce a pipeline that enhances MLLM perception by fusing downsampled video frames with synchronized high-frequency telematics data (IMU and GPS) and semantic insights from specialized computer vision models. Our pipeline generates high-quality pseudo-labels, including descriptive captions and question-answer pairs, specifically designed to train MLLMs to identify and describe Safety-Critical Events (SCEs) in real-world driving footage. We show the effectiveness of our approach fine-tuning the open-source QwenVL-2.5 model via DoRA adapters: our experiments demonstrate significant improvements in identifying and explaining safety-critical events, with fewer than 50M trainable parameters and limited computational budget.
>
---
#### [new 082] Swift Sampling: Selecting Temporal Surprises via Taylor Series
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Swift Sampling，用于视频中关键帧选择。解决长视频信息冗余问题，通过泰勒展开预测视觉特征，识别突变帧。属于视频摘要或关键帧检测任务。**

- **链接: [https://arxiv.org/pdf/2605.22678](https://arxiv.org/pdf/2605.22678)**

> **作者:** Dahye Kim; Bhuvan Sachdeva; Karan Uppal; Naman Gupta; Vineeth N. Balasubramanian; Deepti Ghadiyaram
>
> **摘要:** While most frames in long-form video are redundant, the critical information resides in temporal surprises: moments where the actual visual features deviate from their predicted evolution. Inspired by the human brain's predictive coding, we introduce Swift Sampling, an elegant, training-free frame selection algorithm that automatically identifies high-information moments in a video. Specifically, we model a video as a differentiable trajectory in the visual latent space and compute the velocity and acceleration of its features. Then, we apply Taylor expansion to project the expected path of subsequent frames. Frames that diverge sharply from this predicted manifold are identified as temporally surprising frames and selected for sampling. Unlike prior training-free methods that rely on auxiliary networks or video-specific hyperparameter tuning, Swift Sampling is incredibly lightweight, adding only 0.02x additional computational cost over baseline making it 30x cheaper overhead than leading baselines. Across three long-video question answering benchmarks and 10 different downstream tasks, Swift Sampling outperforms uniform sampling and prior query-agnostic baselines. It is especially powerful for long videos with limited frame budgets improving accuracy by up to +12.5 points.
>
---
#### [new 083] ForeSplat: Optimization-Aware Foresight for Feed-Forward 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决feed-forward 3DGS模型优化能力不足的问题。提出ForeSplat框架，通过优化感知训练提升初始化质量，减少优化步骤，实现高效高保真重建。**

- **链接: [https://arxiv.org/pdf/2605.22020](https://arxiv.org/pdf/2605.22020)**

> **作者:** Yuke Li; Weihang Liu; Cheng Zhang; Yuefeng Zhang; Jiadi Cui; Zixuan Wang; Junran Ding; Haoyu Wu; Yujiao Shi; Jingyi Yu; Xin Lou
>
> **摘要:** Feed-forward 3D Gaussian Splatting (3DGS) models offer fast single-pass reconstruction,but scaling them to match per-scene optimization quality is fundamentally hindered by the scarcity of large-scale 3D annotations.A practical compromise is predict-then-refine,where post-prediction optimization compensates for the limited capacity of the feed-forward this http URL,standard feed-forward 3DGS is trained solely for zero-step rendering error,ignoring whether its output constitutes a good initialization for the downstream this http URL present ForeSplat,an optimization-aware training framework that equips feed-forward 3DGS models to produce initializations explicitly designed for rapid,effective this http URL offloading part of the scene-modeling burden to the optimizer,ForeSplat substantially reduces the capacity pressure on the feed-forward model,making high-quality reconstruction feasible even with compact this http URL its core is MetaGrad,a lightweight multi-anchor meta-gradient training rule that bypasses costly higher-order differentiation through the 3DGS this http URL unrolls a short inner-loop refinement trajectory,samples anchor states,and back-propagates aggregated first-order gradients to the prediction head as a surrogate optimization-aware this http URL fine-tuning adds no inference cost and enables high-quality reconstruction within seconds after a few refinement this http URL instantiate ForeSplat on diverse backbones,including AnySplat,Pi3X,and a distilled variant tailored for edge this http URL all tested architectures,a ForeSplat-trained initialization converges in fewer refinement steps and reaches a higher peak reconstruction quality than its vanilla counterpart,even fully this http URL framework consistently bridges the gap between amortized prediction and per-scene optimization,establishing a practical path toward lightweight,high-fidelity 3D reconstruction.
>
---
#### [new 084] JMed48k: A Multi-Profession Japanese Medical Licensing Benchmark for Vision-Language Model Evaluation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出JMed48k，一个用于评估视觉语言模型的日本医疗执照基准数据集。任务是评估模型在医学领域的表现，解决视觉信息对模型影响的问题，通过构建包含文本和图像的题库进行分析。**

- **链接: [https://arxiv.org/pdf/2605.22080](https://arxiv.org/pdf/2605.22080)**

> **作者:** Yue Xun; Junyu Liu; Qian Niu; Xinyi Wang; Zheng Yuan; Zirui Li; Zequn Zhang; Bowen Zhao; Shujun Wang; Irene Li; Kan Hatakeyama-Sato; Yusuke Iwasawa; Yutaka Matsuo
>
> **摘要:** We introduce JMed48k, a multi-profession Japanese healthcare licensing benchmark for evaluating vision-language models. Built from official PDF materials released by the Japanese Ministry of Health, Labour and Welfare, JMed48k contains 48,862 exam questions and 20,142 images from 11 national licensing examinations between 2005 and 2025, with visual content annotated under an 8-type taxonomy. From this corpus, we derive JMed48k-Eval, a recent five-year evaluation subset with 12,484 scored questions, including 9,905 text-only questions and 2,579 questions with images. We evaluate 21 proprietary, open-source, and medical-specific models, reporting text-only and with-image performance separately. Because these subsets contain different questions, we further introduce a paired image-removal audit that evaluates questions with images before and after removing visual content to explore four answer-transition states. The audit shows that proprietary and open source models gain substantially from images, whereas medical-specific systems show limited observable use of visual evidence, with many correct answers persisting after image removal. Even among proprietary models, the net image-removal effect varies sevenfold across professions, from +5.7 points on Physician questions to +39.8 points on Public Health Nurse questions. We release JMed48k to support reproducible, profession-stratified evaluation of vision-language models in medical licensing settings.
>
---
#### [new 085] MaSC: A Masked Similarity Metric for Evaluating Concept-Driven Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成的评估任务，解决概念驱动生成中概念保留与提示遵循的评价问题。提出MaSC，通过掩码分离主体与背景进行更准确的评估。**

- **链接: [https://arxiv.org/pdf/2605.22469](https://arxiv.org/pdf/2605.22469)**

> **作者:** Patryk Bartkowiak; Lennart Petersen; Bartosz Kotrys; Dominik Michels; Soren Pirk; Wojtek Palubicki
>
> **备注:** 20 pages, 2 figures, 7 tables
>
> **摘要:** Evaluating single-concept personalization in text-to-image diffusion requires measuring both concept preservation, which captures identity fidelity to a reference, and prompt following, which captures whether the generated scene matches the prompt. Existing metrics commonly compute these signals using global image or text-image embeddings, such as CLIP-I, DINO, and CLIP-T. We show that such metrics correlate poorly with human perception because they attend to the image as a whole instead of separating the concept subject from the background. We introduce MaSC, a masked similarity metric that uses externally provided foreground concept masks to decompose evaluation into subject-specific concept preservation and background-based prompt following. MaSC computes both scores from frozen SigLIP2 SO400M-NaFlex features: concept preservation is measured by masked max-cosine matching between foreground reference patches and generated-image patches, while prompt following is measured by comparing a background-only pooled image embedding to a subject-stripped prompt embedding. On DreamBench++ human ratings, MaSC achieves Krippendorff alpha = 0.471 for concept preservation, outperforming all tested non-LLM baselines and GPT-4V, and approaching GPT-4o. On ORIDa, a real-photo identity-preservation benchmark across physical environments, MaSC achieves AUC = 0.992, nearly perfectly distinguishing same-subject from cross-subject pairs. Its prompt-following score also outperforms the CLIP-T baseline shipped with DreamBench++. These results show that spatially decomposed aggregation is a strong design principle for evaluating concept-driven generation.
>
---
#### [new 086] Two-Stage Multimodal Framework for Emotion Mimicry Intensity Prediction
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文属于情感强度预测任务，旨在通过多模态数据预测六种情绪强度。工作包括提出分阶段融合框架，结合文本、音频、视觉和运动信息，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2605.21869](https://arxiv.org/pdf/2605.21869)**

> **作者:** Dinithi Dissanayake; Shaveen Silva; Ovindu Atukorala; Prasanth Sasikumar; Suranga Nanayakkara
>
> **备注:** 10th Affective & Behavior Analysis in-the-wild, CVPR Workshop 2026
>
> **摘要:** We present our submission to the Hume-ABAW10 Emotional Mimicry Intensity (EMI) Challenge, which aims to predict six continuous emotion intensity dimensions: Admiration, Amusement, Determination, Empathic Pain, Excitement, and Joy, from in-the-wild multimodal video clips. We propose a staged multimodal framework that combines textual, acoustic, and visual representations, with an optional motion branch. Our approach first trains modality-specific encoders independently and then fuses their learned representations through a lightweight regressor with modality dropout and controlled encoder adaptation. Across our submitted systems, the best validation performance is obtained by the text--audio--vision--motion fusion model under the expanded 4:1 split, achieving an average Pearson correlation of 0.4722. Although the motion branch yields only very slight gains, its behavior can be interesting to study. Our team was placed third in the EMI challenge, achieving an average Pearson correlation of 0.57 for the test set. Overall, we provide a practical and reproducible baseline for EMI prediction.
>
---
#### [new 087] AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出AnyMo，解决可穿戴设备在非受控环境下运动建模的问题。通过生成合成数据和跨模态对齐，提升运动识别与理解性能。**

- **链接: [https://arxiv.org/pdf/2605.22715](https://arxiv.org/pdf/2605.22715)**

> **作者:** Baiyu Chen; Zechen Li; Wilson Wongso; Lihuan Li; Xiachong Lin; Hao Xue; Benjamin Tag; Flora Salim
>
> **摘要:** As wearable and mobile devices become increasingly embedded in daily life, they offer a practical way to continuously sense human motion in the wild. But inertial signals are highly dependent on the sensing setup, including body location, mounting position, sensor orientation, device hardware, and sampling protocol. This setup dependence makes it difficult to learn motion representations that transfer across devices and datasets, and limits the broader use of wearable IMUs beyond closed-set recognition. We introduce AnyMo, a geometry-aware framework for setup-agnostic human motion modeling. AnyMo uses physics-grounded IMU simulation over dense body-surface placements to generate diverse and plausible synthetic signals, pre-trains a graph encoder from paired synthetic placement views and masked partial observations, tokenizes multi-position IMU into full-body motion tokens, and aligns these tokens with an LLM for motion-language understanding. We evaluate AnyMo on three complementary tasks: zero-shot activity recognition across 14 unseen downstream datasets, cross-modal retrieval, and wearable IMU motion captioning, where it improves average Accuracy/F1/R@2 by 11.7\%/11.6\%/22.6\% on HAR, increases zero-shot IMU-to-text and text-to-IMU retrieval MRR by 15.9\% and 28.6\%, respectively, and improves zero-shot captioning BERT-F1 by 18.8\%. These results support AnyMo as a generalist model for wearable motion understanding in the wild. Project page: this https URL.
>
---
#### [new 088] SDGBiasBench: Benchmarking and Mitigating Vision--Language Models' Biases in Sustainable Development Goals
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言推理任务，旨在解决SDG模型中的偏见问题。通过构建基准测试集，评估并减轻模型对先验知识的依赖，提升预测公平性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.21919](https://arxiv.org/pdf/2605.21919)**

> **作者:** Zihang Lin; Huaiyuan Qin; Muli Yang; Hongyuan Zhu
>
> **摘要:** Assessing progress toward the Sustainable Development Goals (SDGs) requires multi-step reasoning over visual cues, contextual knowledge, and development indicators, where incomplete evidence use and imperfect evidence integration can introduce hidden prediction biases. Real-world SDG monitoring further spans both qualitative judgments and quantitative estimation. However, existing benchmarks typically evaluate these aspects in isolation, obscuring systematic biases that emerge when models substitute priors for evidence. To address this gap, we propose SDGBiasBench, a large-scale benchmark suite for SDG-oriented vision-language reasoning. Spanning 500k expert-involved multiple-choice questions and 50k regression tasks, the benchmark enables comprehensive assessment of both decision-level and estimation-level bias in Vision--Language Models (VLMs). Evaluations on SDGBiasBench reveal an intrinsic SDG bias in current VLMs, where predictions are frequently driven by SDG specific priors rather than reliable multi-modal cues. To mitigate such bias, we propose CADE (Contrastive Adaptive Debias Ensemble), a training-free, plug-and-play method that leverages modality-specific answer priors. CADE yields significant gains on the proposed benchmark, improving multiple-choice accuracy by up to 25% and reducing regression MAE by up to 12 points across multiple VLMs. We hope our work can foster the development of more fair and reliable AI systems for sustainable development.
>
---
#### [new 089] Dual-Integrated Low-Latency Single-Lens Infrared Computational Imaging for Object Detection
- **分类: cs.CV; physics.optics**

- **简介: 该论文属于红外目标检测任务，旨在解决计算成像中延迟高、精度与速度难以兼顾的问题。提出PDI-Net框架，集成重建与检测，并引入物理先验，提升效率与准确率。**

- **链接: [https://arxiv.org/pdf/2605.21964](https://arxiv.org/pdf/2605.21964)**

> **作者:** Xuquan Wang; Guishuo Yang; Dapeng Yan; Yujie Xing; Xuanyu Qian; Kai Zhang; Xiong Dun; Jiande Sun; Zhanshan Wang; Xinbin Cheng
>
> **备注:** 15 pages, 11 figures; supplementary material: 3 pages, 2 figures
>
> **摘要:** Computational imaging enables compact infrared systems, but deep-learning pipelines that combine image reconstruction and object detection often introduce substantial inference latency. Most existing acceleration strategies compress the reconstruction network while overlooking physical priors from the optical path, leaving a trade-off between accuracy and speed. We present Physics-aware Dual-Integrated Network (PDI-Net), a low-latency framework that integrates infrared reconstruction with object detection and further embeds optical priors into the learning process. PDI-Net uses a supervised U-Net during training, while a semi-U-Net encoder shares features directly with a YOLO-based detector during inference, avoiding full image reconstruction. To bridge the gap between fidelity-oriented reconstruction features and detection-oriented semantics, we introduce a physics-aware large-small bridge (PALS-Bridge), which uses field-dependent point spread function priors to adaptively modulate multiscale convolutional branches. A physics-informed optical degradation simulation pipeline is also developed for training and validation. The method is deployed on a single-lens infrared camera, reducing system weight by about 50% compared with traditional multi-lens designs. On the M3FD benchmark under low-SNR conditions, PDI-Net reduces inference time by 84.06% compared with the Rec+Det with pruning strategy while improving mAP@0.5:0.95 by 5.07%. These results demonstrate compact, low-latency computational infrared imaging for real-time object detection on resource-constrained platforms.
>
---
#### [new 090] CrossVLA: Cross-Paradigm Post-Training and Inference Optimization for Vision-Language-Action Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言-动作模型的优化任务，解决DPO在连续动作模型中的应用问题，提出CrossVLA方法提升性能并分析推理效率。**

- **链接: [https://arxiv.org/pdf/2605.21854](https://arxiv.org/pdf/2605.21854)**

> **作者:** Zhi Liu
>
> **备注:** Workshop draft, 14 pages, 4 figures. Code, ckpts, data: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have rapidly converged on a small set of architectural patterns: discrete-token autoregression (e.g. OpenVLA) and continuous-action flow-matching (e.g. pi-0.5). Yet preference alignment via Direct Preference Optimisation (DPO) -- the de-facto post-training step in language models -- has been studied almost exclusively on autoregressive VLAs. We present CrossVLA, an empirical study of cross-paradigm VLA post-training. Three contributions: (i) a surrogate flow-matching log-probability estimator that lets DPO operate on continuous-action backbones without probability-flow ODE integration; (ii) a head-to-head comparison of LoRA and DoRA as the parameter-efficient layer for VLA DPO, finding DoRA improves over OpenVLA SFT by a mean +10.4 pp across LIBERO 4-suite (600 trials, 3 seeds) -- per-suite +20.0 Object, +11.0 Long-horizon, +8.0 Goal, +2.7 Spatial -- with zero seed variance on Object (38/50 on each of 3 seeds); (iii) an inference-time anatomy showing the denoise loop dominates 78.6% of sample_actions latency and prefix-K/V caching a la VLA-Cache caps at a 21% acceleration ceiling -- both chunk-level and token-level cache strategies degrade success rate to 0-80% in our benchmarks. We further pretrain a multi-view + temporal projection head on 6000 LIBERO frames, achieving 99.5% k-NN recall@1 for same-task retrieval (36x over random), available as a downstream initialisation. All code, ckpts, training logs, and reproduction scripts are open at this https URL.
>
---
#### [new 091] The Neglected Baseline in Model Interpretation
- **分类: cs.CV; cs.SE**

- **简介: 该论文属于模型解释任务，旨在解决现有方法忽略基线导致解释不准确的问题。通过重新定义解释原则，统一多种方法，并提出改进的解释方法。**

- **链接: [https://arxiv.org/pdf/2605.22417](https://arxiv.org/pdf/2605.22417)**

> **作者:** Yongjin Cui; Xiaohui Fan
>
> **摘要:** We observe that existing model interpretation methods generally ignore the baseline, and such neglect often results in imprecise or even incorrect interpretation. In this paper, we reformulate the task of model interpretation and the interpretation principles for model interpretation results to demonstrate the importance of the baseline. We further unify gradient-based methods, Integrated Gradients (IG) methods, and Taylor expansion, clarifying the connections among them and explicitly identifying the baseline for each method. On this basis, we analyze the flaws and errors in related model interpretation methods (IG, LayerCAM, ODAM, Difference Map). We advocate evaluating the quality of model interpretation results precisely through the attribution error between the attribution result and the attribution target, rather than adopting flawed evaluation methods, such as those based on marginal-effect or the assumption of perfect model performance. We revise IG and develope a model interpretation method with a clear and reasonable baseline, achieving better results. Our method supports model interpretation based on features from any layer. Interpretation based on features from different layers are all reasonable, and the differences among these results reflect varying degrees of feature extraction at different feature extraction stages.
>
---
#### [new 092] GLeVE: Graph-Guided Lesion Grounding with Proposal Verification in 3D CT
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决3D CT与放射报告之间语义-空间差异的问题。提出GLeVE框架，通过图推理和层次细化提升病变定位准确性。**

- **链接: [https://arxiv.org/pdf/2605.22619](https://arxiv.org/pdf/2605.22619)**

> **作者:** Shuo Jiang; Yuhao Hong; Chunbo Jiang; Weihong Chen; Huangwei Chen; Shenghao Zhu; Beining Wu; Mingxuan Liu; Zhu Zhu; Feiwei Qin; Min Tan; Yifei Chen
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Grounding radiology report descriptions to 3D CT volumes is essential for verifiable clinical interpretation, yet remains challenging due to the semantic-spatial gap between free-text narratives and volumetric anatomy. Existing report-assisted and vision-language grounding methods typically rely on phrase-level alignment or dense pixel supervision, resulting in limited lesion-wise correspondence and suboptimal localization accuracy. We propose GLeVE, a graph-guided lesion grounding framework with anatomical prior verification and octree-based autoregressive refinement. GLeVE treats each lesion description as an atomic semantic unit and encodes organ attribution, attributes, and inter-lesion relations through relation-aware graph reasoning to produce discriminative lesion-wise queries. Anatomy-aware proposal generation with region-level verification enforces one-to-one text-lesion alignment, while hierarchical octree refinement progressively improves boundary delineation. Experiments on AbdomenAtlas 3.0 demonstrate consistent gains over classical multimodal foundation models and report-supervised baselines in both segmentation accuracy and lesion-level localization.
>
---
#### [new 093] Universal CT Representations from Anatomy to Disease Phenotype through Agglomerative Pretraining
- **分类: cs.CV**

- **简介: 该论文提出FlexiCT，一个通过分层预训练的CT基础模型，解决医学影像任务中模型碎片化问题，实现多任务泛化与疾病表型分析。**

- **链接: [https://arxiv.org/pdf/2605.21906](https://arxiv.org/pdf/2605.21906)**

> **作者:** Yuheng Li; Yuan Gao; Haoyu Dong; Yuxiang Lai; Shansong Wang; Mojtaba Safari; James E. Baciak; Xiaofeng Yang
>
> **摘要:** Computed tomography (CT) is a central to three-dimensional medical imaging, yet CT-based artificial intelligence remains fragmented across task-specific models for segmentation, classification, registration, and report analysis. Here we present FlexiCT, a family of CT foundation models trained by agglomerative continual pretraining on 266,227 CT volumes from 56 publicly available datasets, forming a large-scale public resource for CT representation learning. FlexiCT uses agglomerative pretraining across three stages: two-dimensional axial pretraining, three-dimensional anatomical pretraining and report-guided semantic alignment. This training strategy supports slice-level, volume-level and vision-language analysis. Across five downstream task families (segmentation, classification, registration, vision-language understanding and clinical retrieval), FlexiCT matches or exceeds prior task-specific approaches on multiple benchmarks. Its embeddings further organize CT scans along gradients associated with various tumor stages, suggesting that CT foundation models can capture imaging features relevant to disease phenotype characterization. Code is available at this https URL
>
---
#### [new 094] Cambrian-P: Pose-Grounded Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决多模态大模型缺乏相机姿态信息的问题。通过引入相机姿态作为监督信号，提升视频空间推理能力。**

- **链接: [https://arxiv.org/pdf/2605.22819](https://arxiv.org/pdf/2605.22819)**

> **作者:** Jihan Yang; Zifan Zhao; Xichen Pan; Shusheng Yang; Junyi Zhang; Bingyi Kang; Hu Xu; Saining Xie
>
> **备注:** Project Page: this https URL
>
> **摘要:** Camera pose matters. The position and orientation of each viewpoint define a shared spatial coordinate frame that relates observations across video frames. Yet this signal is largely absent from multimodal LLMs (MLLMs) for video understanding, which process frames as isolated 2D snapshots, instead of the persistent scene humans perceive. We revisit pose as a lightweight supervisory signal and introduce Cambrian-P, a video MLLM augmented with per-frame learnable camera tokens and a pose regression head. With a carefully designed sampling scheme, the model achieves substantial gains of 4.5-6.5% on spatial reasoning benchmarks such as VSI-Bench, generalizes across eight additional spatial and general video QA benchmarks, and, as a byproduct, achieves state of the art streaming pose estimation on ScanNet. Surprisingly, training on pseudo-annotated poses from in-the-wild video further improves general video QA benchmarks, showing pose helps beyond spatial reasoning. Together, these results position camera pose as a fundamental signal for video models that reason about the physical world.
>
---
#### [new 095] Thermo-VL: Extending Vision-Language Models to Thermal Infrared Perception
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在解决低光环境下视觉感知不足的问题。通过引入热红外信息，增强模型的多光谱感知能力。**

- **链接: [https://arxiv.org/pdf/2605.21882](https://arxiv.org/pdf/2605.21882)**

> **作者:** Rusiru Thushara; Yasiru Ranasinghe; Jay Paranjape; Vishal M. Patel
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** Vision-language models (VLMs) often fail under low illumination because their visual grounding is learned predominantly from RGB imagery, whereas thermal infrared preserves complementary scene structure when visible cues degrade. We present Thermo-VL, a wavelength-aware VLM that augments a frozen Molmo-7B backbone with a trainable thermal encoder and a text-guided dual-attention fusion module. Given aligned RGB tokens, thermal tokens, and prompt embeddings, the fusion module conditions thermal features on both language and RGB context, then injects a gated residual into the frozen RGB stream so thermal evidence can be incorporated without disrupting Molmo's pretrained RGB-language interface. We train the model with the standard language-modeling objective together with auxiliary alignment and regularization losses that improve cross-modal grounding and reduce over-reliance on RGB. We also introduce a pixel-aligned RGB-thermal instruction-tuning dataset and Thermo-VL-Bench, a manually screened RGB-thermal VQA benchmark for low-light and cross-spectrum reasoning. Experiments show strong gains on challenging thermal-only and RGB+thermal reasoning tasks, highlighting the value of prompt-conditioned multispectral fusion. Our dataset and code are publicly available at: this https URL
>
---
#### [new 096] OSS: Open Suturing Skills Vision-Based Assessment Challenge 2024-2025
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于开放手术技能评估任务，旨在通过视频分析提升手术训练效果。研究构建了数据集并举办挑战赛，探索自动评估方法，解决技能分类、评分及工具跟踪问题。**

- **链接: [https://arxiv.org/pdf/2605.22200](https://arxiv.org/pdf/2605.22200)**

> **作者:** Hanna Hoffmann; Setareh Bady; Claas de Boer; Max Kirchner; Jan Egger; Rainer Röhrig; Frank Hölzle; Lennart Johannes Gruber; Kunpeng Xie; Marlon Neuhaus; Victor Alves; Guilherme Barbosa; Leonardo Barroso; João Carvalho; Hao Chen; Gabriella d'Albenzio; André Ferreira; Nuno Gomes; Yuichiro Hayashi; Kousuke Hirasawa; Rebecca Hisey; Seungjae Hong; Seoi Jeong; Tiago Jesus; Daehong Kang; Satoshi Kasai; Shunsuke Kikuchi; Takayuki Kitasaka; Satoshi Kondo; Hyoun-Joong Kong; Youngbin Kong; Atsushi Kouno; Shlomi Laufer; Kyu Eun Lee; Bining Long; Nooshin Maghsoodi; Hiroki Matsuzaki; Evangelos Mazomenos; Ori Meiraz; Kensaku Mori; Marina Music; Masahiro Oda; Roi Papo; Jieun Park; Rafael Piexoto; Saeid Rezaei; Mariana Ribeiro; Soyeon Shin; Yang Shu; Idan Smoller; Danail Stoyanov; Yihui Wang; Xinkai Zhao; Sebastian Bodenstedt; Isabel Funke; Stefanie Speidel; Behrus Hinrichs-Puladi
>
> **备注:** Stefanie Speidel and Behrus Hinrichs-Puladi jointly supervised this work. Submitted to MEDIA
>
> **摘要:** Achieving high levels of surgical skill through effective training is essential for optimal patient outcomes. Automated, data-driven skill assessment holds significant potential to improve surgical training. While machine learning-based methods are increasingly popular for assessing skills in minimally invasive surgery, their application to open surgery remains limited. We present the results of a dedicated MICCAI challenge designed to benchmark and advance vision-based skill assessment in open surgery. The challenge dataset comprises videos of an open suturing training task recorded with a static GoPro camera in a dry-lab setting, with instrument trajectories available in addition to the primary video modality. The OSS Challenge was hosted over two consecutive years, comprising two and three independent tasks, respectively: (1) classifying skill level into four classes, (2) predicting the full Objective Structured Assessment of Technical Skills across eight categories, and (3) tracking hands and surgical tools. Participants submitted diverse solutions including deep learning-based video models, tracking-driven methods, and hybrid approaches. General-purpose spatiotemporal video models consistently achieved the strongest performance, though conceptually diverse approaches reached competitive levels when well-executed. Predicting fine-grained OSATS scores remains challenging but benefits substantially from increased training data. Keypoint tracking proves difficult given frequent occlusions and out-of-frame instances, limiting current applicability for motion-based skill analysis. This work benchmarks innovative and diverse solutions for surgical skill assessment, highlighting both the promise and current limitations of video-based evaluation in open surgery and identifying critical directions for advancing automated skill assessment toward clinical impact.
>
---
#### [new 097] GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation
- **分类: cs.CV**

- **简介: 该论文提出GenEvolve，解决开放性图像生成问题，通过工具协调的视觉经验蒸馏实现自进化，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.21605](https://arxiv.org/pdf/2605.21605)**

> **作者:** Sixiang Chen; Zhaohu Xing; Tian Ye; Xinyu Geng; Yunlong Lin; Jianyu Lai; Xuanhua He; Fuxiang Zhai; Jialin Gao; Lei Zhu
>
> **摘要:** Open-ended image generation is no longer a simple prompt-to-image problem. High-quality generation often requires an agent to combine a model's internal generative ability with external resources. As requests become more diverse and demanding, we aim to develop a general image-generation agent that can self-evolve through trajectories and use tools more effectively across varied generation challenges. To this end, we propose GenEvolve, a self-evolving framework based on Tool-Orchestrated Visual Experience Distillation. In GenEvolve, each generation attempt is modeled as a tool-orchestrated trajectory, where the agent gathers evidence, selects references, invokes generation skills, and composes them into a prompt-reference program. Unlike existing agentic generation methods that mainly rely on image-level scalar rewards, GenEvolve compares multiple trajectories for the same request and abstracts best-worst differences into structured visual experience, provided only to a privileged teacher branch. Inspired by on-policy self-distillation, Visual Experience Distillation provides dense token-level supervision, helping the student internalize better search, knowledge activation, reference selection, and prompt construction. We further construct GenEvolve-Data and GenEvolve-Bench. Experiments on public benchmarks and GenEvolve-Bench show substantial gains over strong baselines, achieving state-of-the-art performance among current image-generation frameworks. Our website is as follows: this https URL
>
---
#### [new 098] MotiMotion: Motion-Controlled Video Generation with Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文提出MotiMotion，解决视频生成中运动控制不自然的问题。通过视觉推理和置信度控制，生成更合理、连贯的视频内容。**

- **链接: [https://arxiv.org/pdf/2605.22818](https://arxiv.org/pdf/2605.22818)**

> **作者:** Lee Hsin-Ying; Hanwen Jiang; Yiqun Mei; Jing Shi; Ming-Hsuan Yang; Zhixin Shu
>
> **备注:** ICML 2026. Project page: this https URL
>
> **摘要:** Current motion-controlled image-to-video generation models rigidly follow user-provided trajectories that are often sparse, imprecise, and causally incomplete. Such reliance often yields unnatural or implausible outcomes, especially by missing secondary causal consequences. To address this, we introduce MotiMotion, a novel framework that reformulates motion control as a reasoning-then-generation problem. To encourage causally grounded and commonsense-consistent interactions, we leverage a training-free vision-language reasoner to refine image-space coordinates of primary trajectories and to hallucinate plausible secondary motions. To further improve motion naturalness, we propose a confidence-aware control scheme that modulates guidance strength, enabling the model to closely follow high-confidence plans while correcting artifacts under low-confidence inputs with its internal generative priors. To support systematic evaluation, we curate a new image-to-video benchmark, MotiBench, consisting of interaction-centric scenes where new events are triggered by motion. Both VLM-based evaluation and a human study on MotiBench demonstrate that MotiMotion produces videos with more plausible object behaviors and interaction, and is preferred over existing approaches.
>
---
#### [new 099] FastTab: A Fast Table Recognizer with a Tiny Recursive Module and 1D Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FastTab，用于表格结构识别（TSR），解决表格布局和分隔符精准定位问题，采用轻量模块和1D Transformer提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.22422](https://arxiv.org/pdf/2605.22422)**

> **作者:** Laziz Hamdi; Amine Tamasna; Pascal Boisson; Thierry Paquet
>
> **摘要:** Table structure recognition (TSR) requires both table-level coherence (row/column counts, headers, spanning cells) and precise separator localization. We introduce FastTab, a grid-centric TSR model that avoids autoregressive HTML decoding by combining (i) a lightweight Tiny Recursive Module (TRM) for global reasoning and (ii) axial 1D Transformer encoders that capture long-range dependencies along rows and columns. The model predicts row/column counts, header rows, and separators to construct a grid, then infers rowspan/colspan using ROI-aligned cell features. Across four benchmarks (PubTabNet, FinTabNet, PubTables-1M, and SciTSR), FastTab achieves competitive structure recovery performance while operating at low-latency inference. We further study robustness under pixel-level anonymisation and show an extension to curved separators for camera-captured documents. The source code will be made publicly available at this https URL .
>
---
#### [new 100] Seizure-Semiology-Suite (S3): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding
- **分类: cs.CV**

- **简介: 该论文提出Seizure-Semiology-Suite，用于癫痫发作行为理解的多模态数据集与基准，解决医学视频分析中癫痫症状识别问题，通过任务评估和模型优化提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2605.21852](https://arxiv.org/pdf/2605.21852)**

> **作者:** Lina Zhang; Tonmoy Monsoor; Peizheng Li; Jiarui Cui; Xinyi Peng; Chong Han; Prateik Sinha; Siyuan Dai; Jessica Nichole Pasqua; Colin M McCrimmon; Weiting Liu; Hailey Marie Miranda; Bing Hu; Xiangting Wu; Tengyou Xu; Chunhan Li; Jiaye Tian; Jiarui Tang; Detao Ma; Lingye Kong; Junnan Lyu; Jungang Li; Yan Zan; Junhua Huang; Rajarshi Mazumder; Vwani Roychowdhury
>
> **备注:** Accepted to ICML 2026 as a Spotlight presentation
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have demonstrated remarkable proficiency in general video understanding, their capacity to interpret involuntary, and spatio-temporally evolving pathologic motor behaviors such as seizure semiology remains largely untested. To address this gap, we introduce Seizure-Semiology-Suite, a clinically grounded dataset and benchmark for fine-grained, structured seizure semiology understanding. The dataset includes 438 seizure videos annotated with over 35,000 dense labels covering 20 ILAE-defined semiological features. Building on this dataset, we propose a seven-task hierarchical benchmark that systematically evaluates MLLMs from low-level visual perception to temporal sequencing, narrative report generation, and seizure diagnosis. To enable clinically meaningful evaluation of generated reports, we further introduce the Report Quality Index for Seizure Semiology (Seizure-RQI). Extensive baselines across 11 open-weight MLLMs reveal systematic weaknesses in laterality reasoning, temporal localization, symptom sequencing, and clinically faithful reporting. We show that seizure-specific fine-tuning substantially improves performance across tasks, and that a two-stage neuro-symbolic framework achieves an F1 score of 0.96 on epileptic versus non-epileptic seizure classification. Seizure-Semiology-Suite establishes a rigorous benchmark for evaluating multimodal models in safety-critical medical video understanding and guides the development of clinically reliable, domain-adaptive multimodal intelligence.
>
---
#### [new 101] Spectral Tail Auxiliary Learning for AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决真实与生成图像辨别困难的问题。通过分析频谱尾部异常提升现象，提出STAL框架，提升检测的泛化性和稳定性。**

- **链接: [https://arxiv.org/pdf/2605.22751](https://arxiv.org/pdf/2605.22751)**

> **作者:** Xingyi Li; Jiahui Zhang; Yiheng Li; Yun Cao; Wenhao Wang
>
> **摘要:** As generative image models evolve rapidly, the perceptual gap between generated and real images continues to narrow, making AI-generated image detection increasingly challenging. Many existing methods exploit frequency-domain cues for detection, typically described as frequency-domain artifacts or high-frequency discrepancies. However, the specific and recurring spectral regularities remain insufficiently understood and characterized. In this paper, we systematically analyze the one-dimensional radial log-power spectra of real and generated images. We find that generated images do not necessarily exhibit higher or lower energy across the entire spectrum or high-band range. Instead, their spectra deviate from the power-law decay and show an anomalous uplift in the ultra-high-frequency tail. We term this phenomenon spectral tail uplift. We further attribute this phenomenon to nonlinear harmonic accumulation in trained generative models, suggesting that it can serve as a structural cue across generative architectures. Based on this observation, we propose Spectral Tail Auxiliary Learning (STAL), a frequency-domain auxiliary supervision framework for generalizable AI-generated image detection. STAL transfers spectral-tail cues from a tail-aware frequency teacher to a spatial detector during training, while all frequency-domain modules are discarded at inference time. Consequently, STAL introduces no inference overhead. Extensive experiments on 9 public datasets show that STAL achieves strong generalization and stability across generators, data distributions, and real-world scenarios.
>
---
#### [new 102] OPERA: An Agent for Image Restoration with End-to-End Joint Planning-Execution Optimization
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，解决复杂退化场景下的恢复问题。提出OPERA框架，通过联合优化规划与执行，提升恢复效果。**

- **链接: [https://arxiv.org/pdf/2605.22104](https://arxiv.org/pdf/2605.22104)**

> **作者:** Feng Zhu; Shuyang Xie; Yihan Zeng; Ming Liu; Wangmeng Zuo
>
> **摘要:** Real-world image restoration is challenging due to complex and interacting mixed degradations. Recent agent-based approaches address this problem by composing multiple task-specific restoration tools. However, empirical analysis reveals that their performance is fundamentally limited by implicitly constrained planning spaces and the lack of coordination among independently pretrained tools. To address these issues, we propose OPERA (Optimized Planning-Execution Restoration Agent), a framework that jointly optimizes restoration planning and tool execution in an end-to-end manner. On the planning side, OPERA uses reinforcement learning to directly optimize tool composition over a combinatorial plan space, with the final restoration quality as the reward. On the execution side, OPERA introduces agent-guided co-training of restoration tools, enabling them to learn cooperative behaviors under sequential composition. Extensive experiments on multi-degradation benchmarks and real-world datasets demonstrate that OPERA consistently outperforms both all-in-one restoration models and existing agent-based methods across diverse and complex degradation scenarios.
>
---
#### [new 103] PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出PhysX-Omni，解决3D生成中物理属性不足或类别单一的问题，实现刚体、柔体和关节物体的统一物理3D生成。**

- **链接: [https://arxiv.org/pdf/2605.21572](https://arxiv.org/pdf/2605.21572)**

> **作者:** Ziang Cao; Yinghao Liu; Haitian Li; Runmao Yao; Fangzhou Hong; Zhaoxi Chen; Liang Pan; Ziwei Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** Simulation-ready physical 3D assets have emerged as a promising direction owing to their broad applicability in downstream tasks. However, most existing 3D generation methods either neglect physical properties or are limited to a single asset category, e.g., rigid, deformable, or articulated objects. To address these limitations, we introduce PhysX-Omni, a unified framework for simulation-ready physical 3D generation across diverse asset types. Specifically, we develop a novel and efficient geometry representation tailored for Vision-Language Models, which directly encodes high-resolution 3D structures without compression, significantly improving generation performance. In addition, we construct the first general simulation-ready 3D dataset, PhysXVerse, covering diverse indoor and outdoor categories. Furthermore, to comprehensively and flexibly evaluate both generative and understanding capabilities in the wild, we propose PhysX-Bench, which encompasses six key attributes: geometry, absolute scale, material, affordance, kinematics, and function description. Extensive experiments with conventional metrics and PhysX-Bench show that PhysX-Omni performs strongly in both generation and understanding. Moreover, additional studies further validate the potential of PhysX-Omni for applications in simulation-ready scene generation and robotic policy learning. We believe PhysX-Omni can significantly advance a wide range of downstream applications, particularly in embodied AI and physics-based simulation.
>
---
#### [new 104] SADGE: Structure and Appearance Domain Gap Estimation of Synthetic and Real Data
- **分类: cs.CV**

- **简介: 该论文提出SADGE，用于评估合成与真实数据之间的领域差异，解决合成数据有效性预测问题。通过结合外观和结构相似性，提升下游任务性能预测准确性。**

- **链接: [https://arxiv.org/pdf/2605.22467](https://arxiv.org/pdf/2605.22467)**

> **作者:** Patryk Bartkowiak; Bartosz Kotrys; Dominik Michels; Soren Pirk; Wojtek Palubicki
>
> **摘要:** We propose SADGE, a quantitative similarity metric that predicts the performance of synthetic image datasets for common computer vision tasks without downstream model training. Estimating whether a synthetic dataset will lead to a model that performs well on real-world data remains a bottleneck in model development. Existing evaluation metrics (e.g., PSNR, FID, CLIP) primarily measure semantic alignment between real and synthetic images (Appearance Similarity Score). Less commonly, structural similarity between images is considered to assess the domain gap (Geometric Similarity Score). However, to the best of our knowledge there exists no studies that evaluate which similarity metric is the best downstream predictor for a given synthetic dataset. In this paper, we show over a wide variety of different synthetic datasets and downstream tasks that neither appearance nor geometry alone can reliably predict downstream performance; rather, it is their non-linear interplay that dictates synthetic data utility. Specifically, we measure how commonly used Appearance and Geometric Similarity metrics computed between synthetic and real images correlate with downstream performance in object detection, semantic segmentation, and pose estimation. Across five public synthetic-to-real benchmark families and 15 dataset-level variants (79k image pairs), SADGE achieves the strongest association with downstream transfer performance under both linear and rank-based criteria, reaching Pearson r=0.88 and Spearman rho=0.77. We compute for each combination of geometry-based methods and appearance-based approaches SADGE scores across all benchmark families. The best configuration is obtained by fusing DINOv3 appearance similarity with MASt3R geometric consistency through a constrained bilinear interaction, outperforming both the strongest geometry-only baseline and the strongest appearance-only baseline .
>
---
#### [new 105] HyLoVQA: Dynamic Hypernetwork-Generated Low-Rank Adaptation for Continual Visual Question Answering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于持续视觉问答任务，解决模型在连续学习中遗忘旧知识的问题。提出HyLoVQA，通过动态生成低秩适配器和记忆锚点实现高效适应。**

- **链接: [https://arxiv.org/pdf/2605.22035](https://arxiv.org/pdf/2605.22035)**

> **作者:** Yiran Wang; Chenyi Xiong; Ziyue Qin; Miao Zhang; Kui Xiao; Zhifei Li
>
> **备注:** Accepted by IJCAI 2026
>
> **摘要:** Continual Visual Question Answering (VQA) requires learning from non-stationary streams of visual inputs and questions while preserving past knowledge. Most prior methods adapt by updating a largely shared parameter set. This often leads to cross-level task interference, hindering accurate adaptation to the current task and object. To address this limitation, we propose HyLoVQA. It maintains a drift-resilient memory bank of anchors. The bank stores the content of visual objects and textual tasks, and they are updated using current input features. Conditioned on retrieved anchors, a hypernetwork generates lightweight Low-Rank Adaptation (LoRA) adapters. This ensures parameter efficiency, allowing the model to adapt to each task and object dynamically. Additionally, we formulate an alignment loss that aligns semantic discrepancies in the feature space with functional changes in the parameter space, thereby constraining LoRA adapters to remain focused on the current task and object. Extensive experiments on VQA v2 and NExT-QA under both standard and compositional settings demonstrate the superiority of HyLoVQA over prior state-of-the-art methods.
>
---
#### [new 106] AesFormer: Transform Everyday Photos into Beautiful Memories
- **分类: cs.CV**

- **简介: 该论文提出AesFormer，解决日常摄影中审美缺陷问题，通过结构重建提升照片美感，包含两个阶段的框架和基准数据集。**

- **链接: [https://arxiv.org/pdf/2605.22126](https://arxiv.org/pdf/2605.22126)**

> **作者:** Tianxiang Du; Hulingxiao He; Yuxin Peng
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** In everyday photography, aesthetically appealing moments are often captured with structural flaws (e.g., composition, camera viewpoint, or pose) that existing retouching and portrait enhancement methods cannot fix. We formulate Aesthetic Photo Reconstruction (APR) as improving a photo's aesthetic quality via structural reconstruction while preserving subject identity and scene semantics. Although recent advances in image editing models make APR feasible, they often lack aesthetic understanding, yielding edits that are semantically plausible yet aesthetically weak. To address this, we propose AesFormer, a two-stage framework that decouples aesthetic planning from image editing. In Stage 1, an aesthetic action model (AesThinker) analyzes the input along seven progressive photographic dimensions and outputs executable editing actions; we further apply GRPO-A to encourage broad exploration over diverse action plans beyond SFT. In Stage 2, an action-conditioned editor (AesEditor) performs structural edits guided by these actions. To support APR, we build a video-based corpus-mining pipeline (VCMP) and construct AesRecon, a benchmark of 9,071 strictly aligned (poor, good) image pairs. Experiments show that AesFormer substantially improves APR performance and is competitive with Nano Banana Pro. Code is available at this https URL.
>
---
#### [new 107] 4D-GSW: Kinematic-Aware Spatio-Temporal Consistent Watermarking for 4D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于4D内容版权保护任务，解决4DGS中水印导致的物理不一致问题。提出4D-GSW框架，通过STC和HMM-MRF确保水印的时空一致性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.22342](https://arxiv.org/pdf/2605.22342)**

> **作者:** Sifan Zhou; Hang Zhang; Yuhang Wang; Ming Li
>
> **备注:** 9 pages main paper, 7 figures, 18 pages in total
>
> **摘要:** While 4D Gaussian Splatting (4DGS) has revolutionized high-fidelity dynamic reconstruction, safeguarding the intellectual property of these assets remains an open challenge. Conventional steganographic techniques often neglect the underlying kinematic manifolds, triggering non-physical artifacts such as severe temporal flickering and "FVD collapse". To address this, we propose \textbf{4D-GSW}, a kinematic-aware watermarking framework designed to embed robust copyright information while preserving high spatio-temporal consistency. Unlike prior 4D steganography that primarily focuses on opacity-guided invisibility, our approach explicitly addresses the physical coherence of motion trajectories. We introduce a \textbf{Spatio-Temporal Curvature (STC)} metric to identify "Dynamic Instants," adaptively gating watermark gradient injection to shield critical motion manifolds from non-physical perturbations. To ensure global coherence across complex deformations, we formulate a joint \textbf{HMM-MRF energy minimization} model that synchronizes watermark phases within both temporal trajectories and spatial neighborhoods. Furthermore, an \textbf{anisotropic gradient routing} mechanism ensures that watermark embedding remains strictly decoupled from photometric reconstruction fidelity. Extensive experiments have demonstrated the superior performance of our method in robustly hiding watermarks while resisting various attacks and maintaining high rendering quality and spatiotemporal consistency.
>
---
#### [new 108] Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI生成内容检测任务，解决图像与视频检测不一致的问题。通过统一训练和跨模态对齐，提升检测器的泛化能力与性能。**

- **链接: [https://arxiv.org/pdf/2605.21977](https://arxiv.org/pdf/2605.21977)**

> **作者:** Zhengcen Li; Chenyang Jiang; Liangxu Su; Tong Shao; Shiyang Zhou; Ming Tao; Jingyong Su
>
> **摘要:** AI-generated content (AIGC) is rapidly improving, creating an urgent need for detectors that generalize across data sources, deployment pipelines, and visual modalities. A strongly generalizable detector should remain robust under distributional variations. However, we identify a consistent failure mode: SOTA AI-generated image detectors often collapse when applied to frames extracted from videos. Through systematic analysis, we show that this cross-modal gap arises from both entangled synthesis-agnostic video processing shifts, including color conversion, codec compression, resizing, and blur, and model-specific fingerprints introduced by modern video generators. Motivated by these findings, we propose VINA (Video as Natural Augmentation), a unified AIGC detection framework that jointly trains on image and video data. VINA uses video frames as physically grounded natural augmentations and further introduces a cross-modal supervised contrastive objective to align image and video representations under a shared real/fake decision boundary. Extensive experiments on 14 image, video, and in-the-wild benchmarks show that VINA delivers bidirectional gains, improves robustness and transferability, and achieves state-of-the-art performance across nearly all evaluated settings without complex augmentation or dataset-specific tuning.
>
---
#### [new 109] WorldKV: Efficient World Memory with World Retrieval and Compression
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决持久世界一致性与实时性冲突的问题。提出WorldKV框架，通过检索与压缩提升效率和存储，实现高保真与高速度。**

- **链接: [https://arxiv.org/pdf/2605.22718](https://arxiv.org/pdf/2605.22718)**

> **作者:** Jung Yi; Minjae Kim; Paul Hyunbin Cho; Wooseok Jang; Sangdoo Yun; Seungryong Kim
>
> **备注:** Project Page: this https URL
>
> **摘要:** Autoregressive video diffusion models have enabled real-time, action-conditioned world generation. However, sustaining a persistent world, where revisiting a previously seen viewpoint yields consistent content, remains an open problem. Full KV-cache attention preserves this consistency but breaks real-time constraints: memory footprint and attention cost grow linearly with rollout length. Sliding window inference restores throughput but discards long-term consistency. We propose WorldKV, a training-free framework with two components: World Retrieval and World Compression. World Retrieval stores evicted KV-cache chunks in GPU/CPU memory and selectively retrieves scene-relevant chunks via camera/ action correspondence, inserting them back into the native attention window without re-encoding. World Compression prunes redundant tokens within each chunk via key-key similarity to an anchor frame, halving per-chunk storage to fit 2x more history under a fixed budget. On Matrix-Game-2.0 and LingBot- World-Fast, WorldKV matches or exceeds full-KV memory fidelity at roughly 2x the throughput, and is competitive with memory-trained baselines without any fine-tuning. Project Page: this https URL
>
---
#### [new 110] BEiTScore: Reference-free Image Captioning Evaluation with an Efficient Cross-Encoder Model
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于图像描述评估任务，解决现有评价方法计算成本高或敏感性不足的问题。提出一种高效交叉编码器模型，提升评估精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.21728](https://arxiv.org/pdf/2605.21728)**

> **作者:** Gonçalo Gomes; Bruno Martins; Chrysoula Zerva
>
> **摘要:** Image captioning evaluation remains a significant challenge, as vision-language models evolve toward more challenging capabilities such as generating long-form and context-rich descriptions. State-of-the-art evaluation metrics involve extensive computational costs associated with the use of Large Language Models (LLMs) as judges, or instead suffer from the limitations of standard CLIP-based encoders, such as strict token limits, lack of fine-grained sensitivity, or lack of compositional generalization by treating captions as ``bags-of-words.'' We propose a new learned metric that tackles the aforementioned challenges, based on a lightweight cross-encoder that is initialized from a visual question-answering model checkpoint, balancing a strong weight initialization with computational efficiency. Our training scheme uses a carefully assembled data mixture for supervised learning, featuring adversarial LLM-based data augmentations to enhance model sensitivity to fine-grained visual-linguistic errors. We also introduce a new benchmark designed to assess detailed captioning evaluation across diverse scenarios. Experimental results demonstrate that the proposed metric achieves state-of-the-art performance while maintaining the efficiency required for large-scale benchmarking, quality-aware decoding, or reward guidance.
>
---
#### [new 111] From Baseline to Follow-Up: Counterfactual Spine DXA Image Synthesis in UK Biobank Using a Causal Hierarchical Variational Autoencoder
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像生成任务，旨在解决DXA图像中可控且可解释的解剖变异学习问题。通过因果分层变分自编码器，从UKBiobank数据中生成符合生理逻辑的脊柱DXA图像。**

- **链接: [https://arxiv.org/pdf/2605.22649](https://arxiv.org/pdf/2605.22649)**

> **作者:** Yilin Zhang; Nicholas C. Harvey; Nicholas R. Fuggle; Rahman Attar
>
> **备注:** 7 pages, 4 figures, 3 tables. Accepted at the 48th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC 2026)
>
> **摘要:** Dual-energy X-ray absorptiometry (DXA) is widely used for large-scale skeletal assessment, yet learning controllable and interpretable factor-specific anatomical variation remains challenging. We propose a metadata-conditioned causal hierarchical variational autoencoder (CHVAE) for causally consistent generation of anteroposterior (AP) spine DXA images from the UK Biobank (UKB). The model is trained on 3,743 raw AP spine scans from the first imaging visit and conditioned on basic participant attributes and lumbar morphometry. Causal consistency is evaluated in a baseline-to-follow-up setting using abduction--action--prediction (AAP): latent variables are abducted from baseline images, age is intervened to the repeat-imaging value, and the resulting counterfactual follow-up morphometry is compared with observed repeat-imaging measurements. Results show strong absolute-level agreement for key vertebral morphometry variables under age intervention, supporting intervention-aligned synthesis of anatomically plausible DXA images.
>
---
#### [new 112] Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出Sensor2Sensor，解决自动驾驶系统数据不足问题，通过生成模型将单目行车记录仪视频转换为多模态传感器数据。**

- **链接: [https://arxiv.org/pdf/2605.22809](https://arxiv.org/pdf/2605.22809)**

> **作者:** Jiahao Wang; Bo Sun; Yijing Bai; Vincent Casser; Songyou Peng; Zehao Zhu; Meng-Li Shih; Xander Masotto; Shih-Yang Su; Kanaad V Parvate; Tiancheng Ge; Linn Bieske; Dragomir Anguelov; Mingxing Tan; Chiyu Max Jiang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Robust training and validation of Autonomous Driving Systems (ADS) require massive, diverse datasets. Proprietary data collected by Autonomous Vehicle (AV) fleets, while high-fidelity, are limited in scale, diversity of sensor configurations, as well as geographic and long-tail-behavioral coverage. In contrast, in-the-wild data from sources like dashcams offers immense scale and diversity, capturing critical long-tail scenarios and novel environments. However, this unstructured, in-the-wild video data is incompatible with ADS expecting structured, multi-modal sensor inputs for validation and training. To bridge this data gap, we propose Sensor2Sensor, a novel generative modeling paradigm that translates in-the-wild monocular dashcam videos into a high-fidelity, multi-modal sensor suite (AV logs) comprising multi-view camera images and LiDAR point clouds. A core challenge is the lack of paired training data. We address this by converting real AV logs into dashcam-style videos via 4D Gaussian Splatting (4DGS) reconstruction and novel-view rendering. Sensor2Sensor then utilizes a diffusion architecture to perform the generative conversion. We perform comprehensive quantitative evaluations on the fidelity and realism of the generated sensor data. We demonstrate Sensor2Sensor's practical utility by converting challenging in-the-wild internet and dashcam footage into realistic, multi-modal data formats, further unlocking vast external data sources for AV development.
>
---
#### [new 113] AtomicMotion: Learning Human Motion From Different Human Parts
- **分类: cs.CV**

- **简介: 该论文属于人体姿态重建任务，旨在从稀疏的头部和手部轨迹中准确恢复全身姿态。针对现有方法误差累积和关节协调不自然的问题，提出AtomicMotion框架，通过分解身体结构、预训练策略和运动注意机制提升重建精度与生物合理性。**

- **链接: [https://arxiv.org/pdf/2605.22631](https://arxiv.org/pdf/2605.22631)**

> **作者:** Runzhen Liu; Chuhua Xian; Fa-Ting Hong
>
> **摘要:** Accurately reconstructing full-body poses from sparse head and hand trajectories is a foundational challenge for immersive AR/VR telepresence. Current methods often struggle with error accumulation and unnatural joint coordination, primarily because they treat the human body as a monolithic entity, thereby failing to capture the fine-grained ``atomic intents'' embedded in subtle signal variations and overlooking the inherent structural topology. To bridge this gap, we present AtomicMotion, a framework designed to decouple and re-integrate body dynamics through three core innovations. First, we introduce a logical body partitioning scheme that decomposes the skeleton into five distinct clusters based on functional intent; this ensures that each partition preserves internal joint synergies while isolating local motion primitives. Second, to robustly map sparse inputs to high-dimensional poses, we employ a masked full-body pre-conditioning strategy during training, forcing the model to internalize global skeletal topology and latent kinematic constraints. Finally, addressing the limitations of vanilla spatial attention, which often ignores fixed physiological connectivity, we propose Kinematic Attention. By embedding the classical kinematic tree structure into the attention mechanism, we ensure biological plausibility in the synthesized motions. Extensive evaluations on the AMASS dataset demonstrate that AtomicMotion significantly outperforms existing baselines, yielding higher reconstruction fidelity and superior biomechanical realism.
>
---
#### [new 114] GeoWeaver: Grounding Visual Tokens with Geometric Evidence before Scene Reasoning
- **分类: cs.CV**

- **简介: 该论文提出GeoWeaver，解决视觉-语言模型中几何信息不足的问题。通过几何接地框架，提升空间推理能力，增强模型对物理几何的感知。**

- **链接: [https://arxiv.org/pdf/2605.22558](https://arxiv.org/pdf/2605.22558)**

> **作者:** Deshui Miao; Xingsen Huang; Yameng Gu; Xin Li; Haijun Zhang; Ming-Hsuan Yang
>
> **摘要:** Spatio-temporal reasoning in vision-language models requires visual representations that preserve physical geometry rather than merely semantic appearance. Recent multimodal models incorporate geometric information through structural branches, 3D-aware supervision, reasoning-stage fusion, or long-horizon memory. While these approaches demonstrate the importance of geometry for spatial intelligence, they typically treat geometric cues as a shared signal across all visual tokens. We note that this overlooks a finer-grained challenge: different visual tokens require different geometric evidence depending on their spatial roles. To address this limitation, we introduce GeoWeaver, a pre-reasoning geometric grounding framework that treats geometry as a representational prerequisite for spatio-temporal reasoning. GeoWeaver constructs a multi-level geometry bank from a frozen geometry encoder and performs token-adaptive geometric evidence allocation, enabling each visual token to retrieve the most relevant geometric abstractions. The selected evidence is incorporated into visual tokens via a residual grounding operation prior to language modeling, yielding geometry-grounded representations for downstream reasoning. Extensive evaluations on spatial reasoning benchmarks demonstrate that GeoWeaver consistently enhances geometry-aware reasoning while retaining general multimodal capabilities. This indicates that geometric information yields the greatest benefit not as a late-fusion auxiliary signal but as a fundamental prerequisite that shapes the representational foundation on which large language models perform reasoning. All source code and models will be released at this https URL .
>
---
#### [new 115] FashionLens: Toward Versatile Fashion Image Retrieval via Task-Adaptive Learning
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于时尚图像检索任务，旨在解决现有方法无法处理多样化查询的问题。提出FashionLens框架，通过多模态大模型和自适应策略实现通用检索。**

- **链接: [https://arxiv.org/pdf/2605.22552](https://arxiv.org/pdf/2605.22552)**

> **作者:** Haokun Wen; Xuemeng Song; Xinghao Xie; Xiaolin Chen; Xiangyu Zhao; Weili Guan
>
> **摘要:** Fashion image retrieval is a cornerstone of modern e-commerce systems. A unified framework that supports diverse query formats and search intentions is highly desired in practice. However, existing approaches focus on narrow retrieval tasks and do not fully capture such diversity. Therefore, in this work, we aim to develop a unified framework capable of handling diverse realistic fashion retrieval scenarios, achieving truly versatile fashion image retrieval. To establish a data foundation, we first introduce U-FIRE, a comprehensive benchmark that consolidates fragmented fashion datasets into a unified collection, supplemented by two manually curated datasets for testing generalization. Building upon this, we propose FashionLens, a unified framework based on Multimodal Large Language Models. To handle divergent matching objectives, we design a Proposal-Guided Spherical Query Calibrator that dynamically shifts query representations into task-aligned metric spaces via adaptive spherical linear interpolation. Additionally, to mitigate the optimization imbalance caused by varying task complexities and data scales, we develop a Gradient-Guided Adaptive Sampling strategy that automatically re-weights tasks based on realtime learning difficulty and the data scale prior. Experiments on U-FIRE show that FashionLens achieves state-of-the-art performance across diverse retrieval scenarios and generalizes robustly to unseen tasks. The data and code are publicly released at this https URL.
>
---
#### [new 116] Training-Free Fine-Grained Semantic Segmentations in Low Data Regimes: A FungiTastic Baseline
- **分类: cs.CV**

- **简介: 该论文属于细粒度语义分割任务，解决低数据量下类间相似导致的分割难题。提出无需训练的两阶段框架，分离分割与分类，提升分割效率与精度。**

- **链接: [https://arxiv.org/pdf/2605.22492](https://arxiv.org/pdf/2605.22492)**

> **作者:** Sebastian Cavada; Francesco Pelosin; Lapo Faggi
>
> **备注:** Accepted at the 13th Workshop on Fine-Grained Visual Categorization, CVPR 2026
>
> **摘要:** Fine-grained semantic segmentation requires both precise localization and discrimination between visually similar classes. In FungiTastic, this problem is further complicated by a long-tailed distribution and strong variation in image acquisition conditions. We propose a training-free two-stage framework that decouples segmentation from classification. SAM3 first produces class-agnostic mushroom masks using macro-taxonomic prompts, and DINOv3 then assigns fine-grained labels through prototype matching in the embedding space. To improve this stage, we apply a simple transformation of the DINOv3 feature space that improves prototype-based classification. Compared with class-specific prompting, our approach is more scalable and keeps the segmentation cost low. We report results from one-shot to few-hundred-shot regimes, providing, to the best of our knowledge, the first baseline for fine-grained semantic segmentation in low-data settings.
>
---
#### [new 117] FRED: A Multi-Modal Autonomous Driving Dataset for Flooded Road Environments
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出FRED数据集，用于洪水环境下的自动驾驶研究，解决水害场景感知问题，整合多模态传感器数据以支持检测与定位任务。**

- **链接: [https://arxiv.org/pdf/2605.22018](https://arxiv.org/pdf/2605.22018)**

> **作者:** Connor Malone; Sebastien Demmel; Sebastien Glaser
>
> **摘要:** The Flooded Road Environments Dataset (FRED) is, to our knowledge, the first multi-modal autonomous driving dataset specifically targeting the collection of data from scenarios involving water hazards on the road. The dataset contains images from a 2.3 MP FLIR Blackfly USB3 camera, 64-beam 360$^\circ$ point clouds from an Ouster OS1-64 LiDAR, and data from an iXblue ATLANS-C IMU corrected by a Geoflex RTK GNSS, from five separate locations captured both during and after flooding events. The data has been released in two formats: a KITTI-style format for easy integration with existing data tools, and the RTMaps format for direct replay of the vehicle's data capture. We provide semantic labels to enable the training and evaluation of both single-sensor and sensor-fusion methods for water hazard detection. Position and velocity, as well as data captured under dry conditions, are provided to enable the development of location-based detection methods that may incorporate maps, and to evaluate other tasks such as localisation and SLAM.
>
---
#### [new 118] MAVEN: A Multi-stage Agentic Annotation Pipeline for Video Reasoning Tasks
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MAVEN，解决视频推理任务中高质量标注不足的问题。通过多阶段智能标注流程生成结构化数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.21917](https://arxiv.org/pdf/2605.21917)**

> **作者:** Han Zhang; Wanting Jiang; Tomasz Kornuta; Tian Zheng; Vidya Murali
>
> **备注:** CVPR 2026 Workshop
>
> **摘要:** Training Vision Language Models (VLMs) for video event reasoning requires high-quality structured annotations capturing not only what happened, but when, where, why, and with what consequence, at a scale manual labelling cannot support. We present MAVEN (Multi-stage Agentic Video Event aNnotation), a multi-stage agentic pipeline that turns raw videos into multi-task training data with Chain-of-Thought (CoT) reasoning traces, organized around a designated Event of Focus. At its core, MAVEN synthesizes a Multi-Scale Spatio-Temporal Event Description (MSTED) from three complementary caption levels; this explicit intermediate serves as the sole input to downstream Q&A generation across multiple task formats. Crucially, MAVEN supports agent-driven domain adaptation: given a new video dataset and target question examples, the agent redesigns all prompts top-down without manual re-engineering. A hierarchical refinement loop further classifies annotation errors against a taxonomy, traces root causes to the originating pipeline stage, and applies targeted edits that rewrite prompts or modify the pipeline structure itself, iteratively improving data quality. We apply MAVEN to label over 5,300 traffic videos and fine-tune Cosmos-Reason2-8B on the resulting data. On a private CCTV evaluation set, fine-tuning surpasses both Gemini 2.5 Pro and 3.1 Flash, including a $+38.8$-point gain in MCQ accuracy over zero-shot. On AccidentBench, CCTV-only training lifts Cosmos-Reason2 by $+10.7$ MCQ points and matches Gemini 2.5 Pro despite seeing no dashcam videos; adding agent-adapted dashcam annotations narrows the gap to Gemini 3.1 Flash, and RL post-training pushes overall performance past both Gemini baselines. Qualitative results on warehouse surveillance and public safety videos further show the agentic workflow readily adapts the pipeline to new domains.
>
---
#### [new 119] TextTeacher: What Can Language Teach About Images?
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉分类任务，旨在提升图像分类模型性能。通过引入文本嵌入作为辅助目标，提出TextTeacher方法，利用语言模型的语义知识优化视觉模型，提高准确率并保持高效。**

- **链接: [https://arxiv.org/pdf/2605.22098](https://arxiv.org/pdf/2605.22098)**

> **作者:** Tobias Christian Nauen; Stanislav Frolov; Brian Bernhard Moser; Federico Raue; Ahmed Anwar; Andreas Dengel
>
> **备注:** Published at TMLR
>
> **摘要:** The platonic representation hypothesis suggests that sufficiently large models converge to a shared representation geometry, even across modalities. Motivated by this, we ask: Can the semantic knowledge of a language model efficiently improve a vision model? As an answer, we introduce TextTeacher, a simple auxiliary objective that injects text embeddings as additional information into image classification training. TextTeacher uses readily available image captions, a pre-trained and frozen text encoder, and a lightweight projection to produce semantic anchors that efficiently guide representations during training while leaving the inference-time model unchanged. On ImageNet with standard ViT backbones, TextTeacher improves accuracy by up to +2.7 percentage points (p.p.) and yields consistent transfer gains (on average +1.0 p.p.) under the same recipe and compute. It outperforms vision knowledge distillation, yielding more accuracy at a constant compute budget or similar accuracy, but 33% faster. Our analysis indicates that TextTeacher acts as a feature-space preconditioner, shaping deeper layers in the first stages of training, and aiding generalization by supplying complementary semantic cues. TextTeacher adds negligible overhead, requires no costly multimodal training of the target model and preserves the simplicity and latency of pure vision models. Project page with code and captions: this https URL
>
---
#### [new 120] DecQ: Detail-Condensing Queries for Enhanced Reconstruction and Generation in Representation Autoencoders
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决RAE中重建与生成的权衡问题。提出DecQ框架，通过轻量查询提升重建质量并加速生成。**

- **链接: [https://arxiv.org/pdf/2605.22777](https://arxiv.org/pdf/2605.22777)**

> **作者:** Tianhang Wang; Yitong Chen; Wei Song; Zuxuan Wu; Min Li; Jiaqi Wang
>
> **摘要:** Representation Autoencoders (RAEs) leverage frozen vision foundation models (VFMs) as tokenizer encoders, providing robust high-level representations that facilitate fast convergence and high-quality generation in latent diffusion models. However, freezing the VFM inherently constrains its spatial reconstruction capacity, limiting fine-grained generation and image editing; in contrast, incorporating reconstruction-oriented signals via fine-tuning disrupts the pretrained semantic space and degrades generative fidelity. To address this trade-off, we propose DecQ, a simple yet effective framework for RAEs. Specifically, DecQ introduces lightweight detail-condensing queries that extract fine-grained information from intermediate VFM features through condenser modules. These queries are incorporated into the decoder to support reconstruction and are jointly generated with patch tokens during generative modeling. By aggregating information from both shallow and deep layers, DecQ effectively mitigates the reconstruction--generation trade-off, improving both reconstruction quality and generative performance. Our experiments demonstrate that: (1) with only 8 additional queries and 3.9% extra computation, DecQ improves reconstruction over the frozen DINOv2-based RAE, increasing PSNR from 19.13 dB to 22.76 dB; and (2) for generative modeling, DecQ achieves 3.3$\times$ faster convergence than RAE, attaining an FID of 1.41 without guidance and 1.05 with guidance.
>
---
#### [new 121] Ablate-to-Validate: Are Vision-Language Models Really Using Continuous Thought Tokens?
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型研究，旨在验证模型是否真正利用连续思维令牌进行推理。通过提出TRT测试方法，发现模型性能提升可能源于其他因素而非令牌内容本身。**

- **链接: [https://arxiv.org/pdf/2605.21642](https://arxiv.org/pdf/2605.21642)**

> **作者:** Tianyi Zhang; Mahtab Bigverdi; Ranjay Krishna
>
> **摘要:** Vision-language models (VLMs) are increasingly augmented with continuous or latent non-textual tokens intended to support "visual thinking." Despite improved task accuracy, this alone does not show that models actually use these tokens for reasoning -- gains may arise from confounds such as added context length, special-token anchoring, or training-time regularization. We formalize a diagnostic principle, Ablate-to-Validate, for testing whether latent-token content is genuinely utilized, and instantiate it as the Token Replacement Test (TRT), a standardized suite of content-replacement ablations. TRT holds the prompt, image, token budget, and decoding fixed while replacing intermediate tokens with zero, random, first-repeat, or oracle alternatives, isolating whether performance depends on token content or merely on token presence. As a controlled testbed, we study relative depth reasoning with LLaVA-13B and Qwen2.5-VL-3B, training models to predict and consume continuous or discrete depth spans across multiple frozen encoders (SigLIP2, CLIP, DINOv2) and token budgets. We additionally apply TRT to three off-the-shelf visual-thinking systems (Mirage, Mull-Tokens, CoVT) on BLINK, VSP, and CV-Bench. Across all settings, accuracy gains are a misleading proxy for latent-token reasoning: VLMs retain most improvement even when token content is corrupted or replaced, revealing a persistent gap between having a latent channel and using it as an information bottleneck. We recommend TRT as a standard diagnostic alongside accuracy for any method introducing continuous thought tokens.
>
---
#### [new 122] Pre-VLA: Preemptive Runtime Verification for Reliable Vision-Language-Action and World-Model Rollouts
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Pre-VLA，用于解决视觉-语言-动作系统中的动作可靠性问题，通过预验证提升执行成功率和减少错误累积。**

- **链接: [https://arxiv.org/pdf/2605.22446](https://arxiv.org/pdf/2605.22446)**

> **作者:** Zhen Sun; Yongjian Guo; Haoran Sun; Luqiao Wang; Wei Lu; Jiachi Ji; Shengzhe Ji; Junwu Xiong; Zhijun Meng
>
> **摘要:** While large vision-language-action (VLA) models and generative world models (WM) have advanced long-horizon embodied intelligence, their practical deployment remains challenged by uncertainty in learning-based action generation. Low-quality actions may cause physical failures during execution or lead to misleading world-model rollouts with redundant rendering costs. To address this issue, we propose Pre-VLA, a unified runtime verification architecture that performs preemptive action validity assessment before physical execution or world-model imagination. Pre-VLA leverages an efficient multimodal backbone with modality-aware pooling and a lightweight dual-branch head to predict both safety confidence and critic-derived advantage scores for candidate action chunks. To handle severe class imbalance and unstable boundary decisions, we train Pre-VLA with a multi-task objective combining Focal classification, advantage regression, and soft-threshold calibration. During deployment, a dual-mode preemptive resampling scheduler filters low-quality actions and triggers adaptive resampling under a limited computation budget. Experiments on the LIBERO benchmark show that Pre-VLA improves the average closed-loop success rate across four suites from 30.79\% to 37.62\% over RynnVLA-002, reduces task execution steps, achieves 183.9 ms average forward verification time per action chunk, and mitigates error accumulation in world-model rollouts.
>
---
#### [new 123] 3D LULC classification using multispectral LiDAR and deep learning: current and prospective schemes
- **分类: cs.CV**

- **简介: 该论文属于3D地物分类任务，解决缺乏公开NMCA标准数据集的问题，提出新分类方案和基准数据集，并评估深度学习模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22328](https://arxiv.org/pdf/2605.22328)**

> **作者:** Narges Takhtkeshha; Aldino Rizaldy; Markus Hollaus; Juha Hyyppä; Fabio Remondino; Gottfried Mandlburger
>
> **摘要:** Land Use Land Cover (LULC) classification is essential for national 3D mapping, geospatial analysis, and sustainable planning. Multispectral (MS) LiDAR provides synchronized spatial-spectral information, and deep learning (DL) enables 3D point cloud semantic segmentation; however, adoption is limited by the lack of publicly available urban and suburban MS LiDAR datasets aligned with National Mapping and Cadastral Agencies (NMCAs) classification schemes. This study addresses these gaps by introducing L1 and L2 NMCA-aligned LULC classification schemes and a new benchmark MS LiDAR dataset. We evaluate seven state-of-the-art DL models and perform spectral ablation studies at both levels of detail. Results show that Point Transformer V3 achieves the best performance, with mIoU of 79.4% (L1, 8 classes) and 58.9% (L2, 20 classes) using a dual-wavelength LiDAR system (532 nm and 1064 nm). Ablation results show that multispectral information improves performance over geometry-only inputs, with gains of 1.1 percentage points at L1 and 7.8 points at L2. These results highlight the value of LiDAR reflectance for fine-grained material discrimination and support the evolution of NMCA LULC schemes toward higher semantic detail. The Loosdorf-MSL dataset contributes a new benchmark for consistent national and international LULC mapping.
>
---
#### [new 124] Supervised Classification Heads as Semantic Prototypes: Unlocking Vision-Language Alignment via Weight Recycling
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言对齐任务，旨在解决传统方法依赖大量配对数据的问题。通过重用预训练模型的分类头作为语义原型，提升零样本和少样本分类及跨模态检索性能。**

- **链接: [https://arxiv.org/pdf/2605.22484](https://arxiv.org/pdf/2605.22484)**

> **作者:** David Méndez; Roberto Confalonieri; Natalia Díaz Rodríguez
>
> **摘要:** Vision-Language Models (VLMs) excel at tasks like zero-shot classification and cross-modal retrieval by mapping images and text to a shared space, but this requires expensive end-to-end training with massive paired datasets. Current post-hoc alignment methods reduce computational costs by connecting pretrained encoders through lightweight mappings, yet still demand substantial paired data. In this work, we investigate the potential of repurposing the classification heads of pretrained vision models as semantic prototypes. The recycling of these weights, typically discarded after pretraining, unlocks two distinct capabilities: it enables zero-shot alignment by using weights as semantic anchors, and serves as a robust data augmentation strategy by mixing these prototypes with real image-text pairs. We demonstrate that integrating our approach with several state-of-the-art post-hoc alignment techniques consistently boosts accuracy in cross-modal retrieval, zero- and few-shot classification tasks.
>
---
#### [new 125] Look-Closer-Then-Diagnose: Confidence-Aware Ultrasound VQA via Active Zooming
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像问答任务，旨在提升超声图像的诊断准确性。针对现有模型无法主动聚焦病灶及忽略标注主观性的问题，提出一种结合主动缩放和不确定性感知的框架，显著提高了病灶定位效果。**

- **链接: [https://arxiv.org/pdf/2605.21652](https://arxiv.org/pdf/2605.21652)**

> **作者:** Yue Zhou; Erxuan Wu; Yikang Sun; Hongjoo Lee; Yuan Bi; Huixiong Xu; Zhongliang Jiang
>
> **摘要:** Vision-Language Models (VLMs) have significantly advanced medical visual question answering, yet their performance in ultrasound remains suboptimal. In clinical practice, sonographers explicitly focus on lesion regions to formulate reports, though diagnostic interpretations sometimes vary due to inherent subjectivity. However, existing VLMs are not explicitly structured to interactively zoom into lesions prior to diagnosis; moreover, they typically treat annotations as unbiased ground truths, failing to account for their inherent subjectivity and ambiguity. In this paper, we propose a framework specifically designed to consider the sonographer's cognitive workflow. We first introduce a structured Zoom-then-Diagnose paradigm, which replicates the interactive search process to enable lesion-focused reasoning. Furthermore, within the Group Relative Policy Optimization (GRPO) framework, we introduce an uncertainty-aware reward derived from stochastic group-wise rollouts to estimate prediction consistency as a proxy for model confidence. Together, these two components encourage the model to reinforce accurate predictions on clear cases while remaining cautious under ambiguity. Experiments across liver, breast, and thyroid datasets show that our framework improves lesion localization by 39.3\%, demonstrating that our model has learned the ability to actively look closer and diagnose.
>
---
#### [new 126] Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于情感理解任务，旨在解析和增强大视觉语言模型中的情感机制。通过构建数据集和因果归因框架，揭示情感信息的流动路径，并优化情感表达。**

- **链接: [https://arxiv.org/pdf/2605.21980](https://arxiv.org/pdf/2605.21980)**

> **作者:** Chengsheng Zhang; Chenghao Sun; Zhining Xie; Xinmei Tian
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) represent a significant leap towards empathetic agents, demonstrating remarkable capabilities in emotion understanding. However, the internal mechanisms governing how LVLMs translate abstract visual stimuli into coherent emotional narratives remain largely unexplored, primarily due to the scarcity of visual counterfactuals and the diffuse nature of emotional expression. In this paper, we bridge this gap by introducing a steering-vector-based causal attribution framework tailored for descriptive emotional reasoning. To this end, we construct a specialized dataset to demystify the emotional circuits underlying the three-stage ``Adapt-Aggregate-Execute'' mechanism. Crucially, we discover a functional decoupling: visual emotional cues are aggregated in middle layers via sentiment-specific attention heads, but are subsequently translated into narrative generation in deep layers through emotion-general pathways. Guided by these insights, we regulate the emotional information routing to strengthen attention flow and amplify the semantic activation to consolidate expression. Extensive experiments on the comprehensive MER-UniBench demonstrate that our methods significantly improve performance via inference-time intervention, effectively mitigating emotional hallucinations and corroborating the causal fidelity of the discovered circuits.
>
---
#### [new 127] A Robust Semantic Segmentation Pipeline for the CVPR 2026 8th UG2+ Challenge Track 2
- **分类: cs.CV**

- **简介: 该论文针对恶劣天气下的语义分割任务，提出一种半监督分割流程，仅使用WeatherProof数据集进行训练，并通过测试时增强提升性能。**

- **链接: [https://arxiv.org/pdf/2605.22216](https://arxiv.org/pdf/2605.22216)**

> **作者:** Jinming Chai; Libo Yan; Licheng Jiao; Fang Liu
>
> **摘要:** This report presents our solution for the WeatherProof Dataset Challenge, namely CVPR 2026 8th UG2+ Challenge Track 2: Semantic Segmentation in Adverse Weather. For the semantic segmentation task under adverse weather conditions, we propose a semi-supervised segmentation pipeline. Our method is trained exclusively on the WeatherProof dataset, without using any additional external data. Specifically, we adopt UniMatch V2 as the baseline model and treat all degraded-weather images as unlabeled data for semi-supervised training, thereby fully exploiting the data distribution provided by the challenge. During inference, we further apply test-time augmentation to improve the robustness and segmentation accuracy of the final predictions. The code is publicly available at: this https URL.
>
---
#### [new 128] Improving 3D Labeling in Self-Driving by Inferring Vehicle Information using Vision Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，旨在提升自动驾驶中的车辆标注精度。通过视觉语言模型进行零样本推理，提高车辆信息识别和3D边界框标注质量。**

- **链接: [https://arxiv.org/pdf/2605.21747](https://arxiv.org/pdf/2605.21747)**

> **作者:** Steven Chen; Shivesh Khaitan; Nemanja Djuric
>
> **备注:** To appear in Proceedings of the IEEE Intelligent Vehicles Symposium (IV), 2026. Accepted for oral presentation
>
> **摘要:** We present an approach to improve 3D vehicle labeling in self-driving applications through zero-shot inference of vehicle information, leveraging Vehicle Make and Model Recognition (VMMR) methods. The proposed approach utilizes a Vision Language Model (VLM) to both infer a vehicle's make, model, and generation from image crops, and output accurate 3D bounding box dimensions to seed manual labeling. We evaluate the impact of iterative prompt engineering and the choice of different VLMs on both vehicle bounding box inference and make/model/generation recognition. When compared to strong baselines, the proposed approach not only shows high accuracy, but also excels in mitigating specific failure modes where VLMs provide better dimensions than initial lidar-aided human annotated labels (e.g., in cases of significant vehicle occlusion). Experiments on both public and proprietary data strongly suggest that our conclusions are generalizable across different labelers and datasets. The results demonstrate that integrating VLMs into the labeling process can reduce manual labeling time while increasing label quality.
>
---
#### [new 129] PIU: Proximity-guided Identity Unlearning in ID-Conditioned Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于隐私保护任务，解决身份条件扩散模型中个体信息被遗忘的问题。提出PIU框架，通过近邻引导的身份替换实现有效身份删除。**

- **链接: [https://arxiv.org/pdf/2605.22311](https://arxiv.org/pdf/2605.22311)**

> **作者:** Jose Edgar Hernandez Cancino Estrada; Mauro Díaz Lupone; Žiga Emeršič; Vitomir Štruc; Peter Peer; Darian Tomašević
>
> **摘要:** Identity-conditioned diffusion models enable high-quality and identity-consistent face generation, but they also raise severe privacy concerns, as models may continue to synthesize individuals despite their right to be forgotten. While machine unlearning has been extensively studied for concept and data removal, identity unlearning remains largely unexplored, particularly in models conditioned directly on identity embeddings rather than text prompts. In this work, we study identity unlearning in Arc2Face, a state-of-the-art identity-conditioned latent diffusion model for face generation, and introduce Proximity-guided Identity Unlearning (PIU), an anchor-guided framework for identity unlearning. Specifically, we formulate identity removal as an identity replacement objective that reassigns the source identity to a selected anchor identity in the learned identity space, and we complement it with a proximity-based anchor selection strategy motivated by the geometry of ArcFace representations. We further show that effective unlearning can be achieved through localized fine-tuning of a small subset of identity-sensitive cross-attention layers. Experiments across many target identities show that our framework effectively suppresses generation of the target identity while preserving realism and identity consistency for retained identities, as validated by improved performance on unlearning and image-quality metrics, together with qualitative evaluation. The source code for the PIU framework is publicly available at this https URL .
>
---
#### [new 130] What Does the Caption Really Say? Counterfactual Phrase Intervention for Compositional Data Selection in Vision-Language Pretraining
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言预训练任务，旨在解决数据选择中 compositional supervision 与全局过滤不匹配的问题。提出 CPI 方法，通过句法级干预提升数据质量，优化模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22651](https://arxiv.org/pdf/2605.22651)**

> **作者:** Hyejin Go; Semi Lee; Hyesong Choi
>
> **备注:** 11 pages, 2 figures, 4 tables. Preprint
>
> **摘要:** CLIP-style contrastive pretraining typically curates web-scale image-text pairs using sample-level filtering signals, often based on pair-level alignment. We show that this signal saturates: once coarse mismatches are removed, stricter global filtering no longer tracks the compositional supervision provided by the retained captions. The reason is structural - a global score conflates whether a pair is broadly plausible with whether the individual object, attribute, and relation phrases inside the caption materially support the image-text match. The latter is what compositional generalization demands, yet pair-level filters are blind to it. We address this with Counterfactual Phrase Intervention (CPI), a phrase-level curation framework that converts controlled nonce-token substitutions into image-conditioned phrase-sensitivity scores. CPI uses global alignment only for coarse mismatch removal, then ranks the surviving pool by whether caption phrases measurably affect the image-text score under controlled substitution. We frame CPI as a first-order phrase-sensitivity signal rather than a grounding or identification result, and evaluate it at CC3M scale. Ranking by this signal yields a 50%-data subset that improves VL-CheckList-VG Relation by +1.91 over the full-data baseline and +1.00 over alignment-only filtering at matched budget, while improving SugarCrepe overall and preserving general transfer. CPI is loss-orthogonal: applied unchanged to NegCLIP, it further improves VL-CheckList-VG Relation by +3.84, with additional CE-CLIP gains in the main text.
>
---
#### [new 131] Making the Discrete Continuous: Synthetic RAW Augmentations for Fine-Grained Evaluation of Person Detection Performance in Low Light
- **分类: cs.CV; cs.AI; cs.LG; physics.optics**

- **简介: 该论文属于目标检测任务，解决低光环境下行人检测性能评估问题。通过合成低光数据增强，提升模型在低密度区域的评估能力。**

- **链接: [https://arxiv.org/pdf/2605.22455](https://arxiv.org/pdf/2605.22455)**

> **作者:** Valeria Pais; Malena Mendilaharzu; Daniele Faccio; Luis Oala; Christoph Clausen; Bruno Sanguinetti
>
> **备注:** Accepted non-archival paper at the CVPR 2026 AUTOPILOT Workshop (Autonomous Understanding Through Open-world Perception and Integrated Language Models for On-road Tasks)
>
> **摘要:** Real-world deployment of AI vision models is both fueled and limited by the data available for training and testing. Real datasets are sparse and uneven: long-tailed or unbalanced distributions hinder generalization, and the low number of samples in low density regions makes it hard to run evaluations. Synthetic data can fill these gaps, providing us with a way to sample the input space more continuously and improve data coverage for benchmarks. Focusing on the autonomous driving safety-critical case of pedestrian detection in the dark, we show how synthetic low-light samples can be used to better characterize the performance of a state-of-the-art object detection model as a function of the scene illumination. We use a synthetic RAW image augmentation technique to generate low-light samples that match the noise model of the camera sensor. Performance metrics on real and synthetic low-light data are similar, indicating that the AI model finds it hard to distinguish between them.
>
---
#### [new 132] Diverse Yet Consistent: Context-Guided Diffusion with Energy-Based Joint Refinement for Multi-Agent Motion Prediction
- **分类: cs.CV**

- **简介: 该论文属于多智能体运动预测任务，解决生成多样化且一致的交互轨迹问题。通过扩散模型和能量函数优化，提升预测的多样性和交互一致性。**

- **链接: [https://arxiv.org/pdf/2605.22017](https://arxiv.org/pdf/2605.22017)**

> **作者:** Lei Chu; Yuhuan Zhao
>
> **备注:** MEIS-- CVPR
>
> **摘要:** Deepgenerative models havebecomeapromisingapproach for human motion prediction due to their ability to capture multimodal distributions and represent diverse human be haviors. However, generating predictions that are both di verse and jointly consistent among interacting agents re mains challenging. In addition, most existing approaches are primarily evaluated using single-agent (marginal) met rics, which fail to fully reflect the joint dynamics of multi agent interactions. We propose a diffusion-based frame work that improves multi-agent motion prediction by lever aging rich contextual information from historical trajecto ries. This information is incorporated through a guidance mechanism to enhance the diversity and expressiveness of predicted motions. To further enforce interaction consis tency, we introduce an energy-based formulation that re fines the joint trajectory distribution while preserving the plausibility of individual trajectories. Extensive experi ments on four benchmark datasets demonstrate that our approach consistently outperforms existing methods. No tably, our approach substantially improves both marginal (ADE/FDE) and joint (JADE/JFDE) metrics on ETH/UCY over strong marginal baselines. Compared with prior joint prediction methods, it delivers significant gains in marginal metrics while maintaining competitive joint performance.
>
---
#### [new 133] AVI-HT: Adaptive Vision-IMU Fusion for 3D Hand Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出AVI-HT，用于3D手部姿态跟踪，解决视觉遮挡下的跟踪问题。通过融合视觉与IMU数据，提升精度和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.21714](https://arxiv.org/pdf/2605.21714)**

> **作者:** Ziyi Kou; Ankit Kumar; Mia Huang; Taylor Niehues; Vatsal Mehta; Ergys Ristani; Li Guan
>
> **摘要:** We present AVI-HT, an adaptive visual-IMU fusion approach for tracking 3D hand poses by jointly modeling the egocentric image with on-glove 6-DoF IMU signals. AVI-HT achieves significantly improved accuracy and availability, particularly in hand-object interaction (HOI) scenarios involving heavy visual occlusion. Two complementary ingredients underpin its success: (1) synchronized multi-modal training data pairing on-body vision-IMU sensor streams with ground-truth 3D hand poses from a motion-capture system, and (2) a cross-sensor deep attention mechanism that adaptively modulates the trust assigned to the vision and individual IMU sensors. To evaluate AVI-HT in real-world settings, we conduct extensive experiments on our DexGloveHOI dataset that consists of 100K+ pairwise vision-IMU samples with synchronized 3D annotated poses, in which users manipulate a variety of objects during daily tasks. We compare against multiple single- and multi-modal tracking approaches under two hand models (UmeTrack, MANO). The results show that AVI-HT reduces mean keypoint error by 16.1% and its wrist-aligned variant by 24.2% over the baselines. Ablation studies further reveal the per-finger contribution of IMU sensors across activity types, and the model's sensitivity to IMU noise and temporal misalignment in vision-IMU fusion.
>
---
#### [new 134] UniVL: Unified Vision-Language Embedding for Spatially Grounded Contextual Image Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出UniVL，解决空间语义引导的图像生成任务，通过统一视觉语言嵌入减少计算，无需独立文本编码器。**

- **链接: [https://arxiv.org/pdf/2605.21611](https://arxiv.org/pdf/2605.21611)**

> **作者:** Jiayun Wang; Yu Wang; Weijie Gan; Zhenting Wang; Wei Wei
>
> **摘要:** We introduce spatially grounded contextual image generation, a controllable image generation task that reframes the conditioning paradigm. Instead of supplying a reference image and a global text prompt through two separate encoders, one for vision and one for language, UniVL is trained to bind semantics to spatial locations directly from a single unified visual input, where the textual instruction is rendered onto the spatial mask. This removes the need for a standalone text encoder at inference time. The resulting model supports contextual image generation by following user-specified instructions about what should appear where, while substantially reducing computation. To address this task, we propose a framework in which the UniVL encoder, adapted from an optical-character-recognition-pretrained backbone, reads the unified condition optically and produces a UniVL embedding, fVIL, that fuses visual and semantic intent with spatial locations in a single token sequence. A two-stage pipeline first aligns UniVL with the VAE embedding space and then conditions a pretrained diffusion backbone entirely on UniVL embeddings, eliminating the standalone text encoder, such as T5. Although this reframing uses a deliberately minimal text interface, it yields strong empirical gains. On UniVL-ImgGen, a benchmark of 477K mask-annotated images that we construct for training and evaluation, UniVL improves image quality over text-prompted baselines, reducing FID from 14 to 11 and increasing PSNR from 16 to 20. It also eliminates the text encoder entirely, reducing inference TFLOPs by up to 52% and runtime by up to 44%. Additional ablation studies validate the contributions of the proposed components, paving the way for efficient, spatially grounded image generation with a unified conditioning paradigm.
>
---
#### [new 135] AgroVG: A Large-Scale Multi-Source Benchmark for Agricultural Visual Grounding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AgroVG基准，解决农业视觉定位任务中的目标小、重复、遮挡等问题，通过多源数据和多种评估协议提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2605.22034](https://arxiv.org/pdf/2605.22034)**

> **作者:** Haocheng Li; Juepeng Zheng; Zenghao Yang; Kaiqi Du; Guilong Xiao; Gengmeng Pu; Haohuan Fu; Jianxi Huang
>
> **备注:** 45 pages,12 figures
>
> **摘要:** Visual grounding, the task of localizing objects described by natural-language expressions, is a foundational capability for agricultural AI systems, enabling applications such as selective weeding, disease monitoring, and targeted harvesting. Reliable evaluation of agricultural visual grounding remains challenging because agricultural targets are often small, repetitive, occluded, or irregularly shaped, and instructions may refer to one, many, or no objects in an image. Evaluating this capability therefore requires jointly testing localization accuracy, target-set completeness, and existence-aware abstention. To address these challenges, we introduce \textbf{AgroVG}, a multi-source benchmark that formulates agricultural grounding as generalized set prediction: given an image and a referring expression, a model must return all matching target instances or abstain when no target is present. AgroVG contains 10{,}071 annotation-grounded image-query pairs from ten source datasets across six target families: crop/weed, fruit, wheat head, pest, plant disease, and tree canopy. It supports bounding-box grounding (T1) across all six families and instance-mask grounding (T2) on sources with reliable instance-level pixel annotations, with queries covering single-target, multi-target, and target-absent regimes. AgroVG further provides task-specific protocols for box-set matching and query-level mask coverage. Zero-shot evaluation of 26 model configurations spanning closed-source MLLMs, open-source VLMs, and specialized grounding systems reveals persistent gaps: the best multi-target Set-$F_1$ reaches only 0.35, and the best positive-query mask success rate at IoU@0.75 remains below 0.17. Data and code are available at this https URL .
>
---
#### [new 136] Which Way Did It Move? Diagnosing and Overcoming Directional Motion Blindness in Video-LLMs
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，解决Video-LLMs在识别运动方向上的缺陷，即方向运动盲。通过分析并提出DeltaDirect方法提升模型对运动方向的准确率。**

- **链接: [https://arxiv.org/pdf/2605.22823](https://arxiv.org/pdf/2605.22823)**

> **作者:** Jongseo Lee; Hyuntak Lee; Sunghun Kim; Sooa Kim; Jihoon Chung; Jinwoo Choi
>
> **备注:** Preprint. 59 pages, including appendix. Code: this https URL
>
> **摘要:** Video Large Language Models (Video-LLMs) have made rapid progress on temporal video understanding, yet many fail at a basic perceptual primitive: signed image-plane motion direction. On simple videos of a single object moving left, right, up, or down, most Video-LLMs perform near chance, with above-chance cases largely attributable to prediction biases rather than genuine direction understanding. We call this failure directional motion blindness. We localize the failure by tracing motion direction information through the Video-LLM pipeline. Motion direction remains linearly accessible from the vision encoder, projector, and LLM hidden states, but the readout fails to bind this signal to the correct verbal answer option, revealing a direction binding gap. Although synthetic motion direction instruction tuning reduces this gap on the source domain, motion direction concept vector analysis shows that visual complexity weakens the signal magnitude and limits out-of-domain generalization. We introduce MoDirect, a dataset family for motion direction instruction tuning and evaluation, and DeltaDirect, a diagnosis-driven, projector-level objective that predicts normalized 2-D motion vectors from adjacent-frame feature deltas. On MoDirect-SynBench, instruction tuning with DeltaDirect improves motion direction accuracy from 25.9% to 85.4%. On MoDirect-RealBench, DeltaDirect improves real-world motion direction accuracy by 21.9 points over the vanilla baseline without real-world tuning data, while preserving standard video-understanding performance. Code: this https URL
>
---
#### [new 137] PointLLM-R: Enhancing 3D Point Cloud Reasoning via Chain-of-Thought
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文属于3D点云理解任务，旨在解决点云数据缺乏显式推理的问题。通过构建包含推理路径的指令数据集，提升模型的3D多模态推理能力。**

- **链接: [https://arxiv.org/pdf/2605.22013](https://arxiv.org/pdf/2605.22013)**

> **作者:** Chaoqi Chen; Qile Xu; Wenjun Zhou; Hui Huang
>
> **摘要:** Understanding 3D point clouds through language remains a fundamental challenge in computer graphics and visual computing, due to the irregular structure of point cloud data and the lack of explicit reasoning in existing 3D multimodal models. While Chain-of-Thought (CoT) reasoning has shown strong effectiveness in LLMs and image-based MLLMs, its extension to 3D understanding remains largely underexplored. In this paper, we propose a data-centric framework for constructing large-scale CoT supervision tailored to 3D point cloud understanding. Our framework consists of a two-stage pipeline that first refines point-text instruction data via vision-language-model-based quality evaluation and reference-guided refinement, and then synthesizes high-quality reasoning paths through Human-in-the-Loop Prompt Optimization (HiLPO). Using this approach, we build PoCoTI, a CoT-enhanced point-text instruction-following dataset containing 55K samples with explicit reasoning paths. Fine-tuning PointLLM on PoCoTI yields PointLLM-R, a reasoning-capable 3D multimodal language model. Extensive experiments on generative 3D classification and captioning demonstrate that PointLLM-R achieves state-of-the-art performance and generalizes robustly to real-world scanned point clouds and multi-turn dialogue scenarios.
>
---
#### [new 138] TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D重建任务，解决稀疏视角下场景重建质量低的问题。通过引入TPS优化点云对齐，提升3DGS初始化效果，增强细节和颜色保真度。**

- **链接: [https://arxiv.org/pdf/2605.22069](https://arxiv.org/pdf/2605.22069)**

> **作者:** Hyeseong Kim; Geonhui Son; Deukhee Lee; Dosik Hwang
>
> **备注:** Accepted to CVPR 2025, Project page: this https URL
>
> **摘要:** Novel view synthesis from sparse-view inputs poses a significant challenge in 3D computer vision, particularly for achieving high-quality scene reconstructions with limited viewpoints. We introduce TWINGS, a framework that enhances 3D Gaussian Splatting (3DGS) by directly addressing point sparsity. We employ Thin Plate Splines (TPS), a smooth non-rigid deformation model that minimizes bending energy to estimate a globally coherent warp from control-point correspondences, to align backprojected points from estimated depth with triangulated 3D control points, yielding calibrated backprojected points. By sampling these calibrated points near the control points, TWINGS provides a fast and geometrically accurate initialization for 3DGS, ultimately improving structural detail preservation and color fidelity in reconstructed scenes. Extensive experiments on DTU, LLFF, and Mip-NeRF360 demonstrate that TWINGS consistently outperforms existing methods, delivering detailed and accurate reconstructions under sparse-view scenarios.
>
---
#### [new 139] EvoIR-Agent: Self-Evolving Image Restoration Agentic System via Experience-Driven Learning
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，解决多模态大模型在无经验情况下的零样本规划问题。提出EvoIR-Agent，通过经验池和自进化机制提升修复效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.22208](https://arxiv.org/pdf/2605.22208)**

> **作者:** Kailin Zhuang; Jiawei Wu; Zhi Jin
>
> **摘要:** Multimodal Large Language Model (MLLM)-driven image restoration agent demonstrates effectiveness in degradation coupling scenarios by flexibly selecting tools and determining removal orders. However, their zero-shot planning often fails without experience, necessitating severe trial-and-error overhead to achieve satisfactory outcomes. Currently, two paradigms are employed to address this issue, yet a dilemma persists: Training-based methods embed intrinsic experience into parameters, achieving high inference efficiency but lacking compatibility with new tools or degradation. In contrast, training-free methods utilize explicit experience storage for compatibility but still incur trial-and-error overhead due to naive experience. To resolve the dilemma, we propose EvoIR-Agent, which first systematically formulates the experience components of a training-free image restoration agent. Subsequently, a hierarchical experience pool is constructed, which enables coarse-to-fine guidance for diverse tools and removal orders. Furthermore, a self-evolving mechanism is introduced to update the pool from scratch using accumulated records, thereby greatly improving performance and efficiency. Extensive experiments reveal that EvoIR-Agent achieves a significant lead in the full reference metrics and yields a remarkable Pareto-optimal balance between performance and efficiency compared to the state-of-the-art methods.
>
---
#### [new 140] Tackle CSM in JPEG Steganalysis with Data Adaptation
- **分类: eess.IV; cs.AI; cs.CV; cs.MM; eess.SP**

- **简介: 该论文属于图像隐写分析任务，解决真实场景下的载体源不匹配问题。通过引入TADA框架，从少量未标注数据中学习未知处理流程，提升模型鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.21523](https://arxiv.org/pdf/2605.21523)**

> **作者:** Rony Abecidan; Vincent Itier; Jérémie Boulanger; Patrick Bas; Tomáš Pevný
>
> **备注:** ACM Workshop on Information Hiding and Multimedia Security, (IH&MMSec '26), Jun 2026, Florence, Italy
>
> **摘要:** Steganalysis models excel on benchmark datasets but struggle in the wild when analyzed images are produced by a processing pipeline unseen during training. This problem known as Cover Source Mismatch (CSM) is particularly hard in realistic settings where practitioners (1) have access to only a small, unlabeled dataset, (2) are unsure of the processing techniques applied to these images, and (3) lack information on the proportion of covers and stegos in that set. To answer this challenge, we introduce TADA (Target Alignment through Data Adaptation), a framework learning to emulate the unknown processing pipeline from a small unlabeled target set. This architecture is trained with a loss combining residual covariance alignment, residual distribution matching, and a $\ell^2$ loss constraining the emulator to produce realistic images. Across toy and operational targets, TADA yields substantial gains in robustness to CSM and improves operational generalization compared to strong holistic and atomistic baselines. Additional resources are available at this link: this https URL
>
---
#### [new 141] Time-varying rPPG signal separation via block-sparse signal model
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于rPPG信号提取任务，旨在解决弱信号和光照噪声干扰问题。通过构建时变块稀疏模型实现有效信号分离。**

- **链接: [https://arxiv.org/pdf/2605.22425](https://arxiv.org/pdf/2605.22425)**

> **作者:** Kosuke Kurihara; Yoshihiro Maeda; Daisuke Sugimura; Takayuki Hamamoto
>
> **备注:** Accepted by IEEE International Conference on Image Processing (ICIP 2026)
>
> **摘要:** Remote photoplethysmography (rPPG) enables non-contact measurement of cardiac pulse signals by analyzing subtle color changes in facial videos. Nevertheless, extracting rPPG signals remains challenging because of their extremely weak signal strength and susceptibility to illumination noise. In this paper, we propose an rPPG signal extraction method that exploits the quasi-periodic characteristics of rPPG signals. Our approach models quasi-periodicity of the rPPG signal, which arises from the stable cardiac cycle, as a block-sparse structure in the time-frequency domain. To incorporate a block-sparse model and enable adaptive signal separation under illumination fluctuations, we construct a time-varying signal separation framework. Experiments using a public dataset demonstrate the effectiveness of our method.
>
---
#### [new 142] VRXU-net: A Deep Learning Approach for Brain Ischemic Stroke Lesion Detection and Segmentation in T1W MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于脑卒中病灶检测与分割任务，旨在解决T1W MRI中病灶边界识别困难的问题。提出VRXU-net模型，结合VGG和U-Net结构，提升分割精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.21633](https://arxiv.org/pdf/2605.21633)**

> **作者:** Sayed Amir Mousavi Mobarakeh
>
> **摘要:** When the blood supply to the brain is obstructed by a clot, oxygen delivery to brain tissues becomes insufficient, leading to cellular necrosis. In healthcare settings, accurately identifying and delineating ischemic lesion boundaries is essential for treatment and surgical planning. However, ischemic stroke lesions vary widely in shape, size, and location, and in grayscale MRI modalities such as T1W they may resemble surrounding brain structures. This makes lesion detection and segmentation a challenging task for clinicians. This study introduces a novel VRU-Net architecture, derived from visual features, residual connections, and a U-shaped network, for detecting and segmenting ischemic stroke lesions in 3D magnetic resonance imaging scans. The proposed method first uses a modified VGG model to identify ischemic stroke in separate 2D slices. Then, a U-shaped segmentation model with residual blocks segments the lesion in each slice. This procedure is applied independently to the axial, sagittal, and coronal planes, and the final output is generated by aggregating the three segmentation results. To improve both performance and processing speed, a high-performance classifier is applied before the segmentation model in a sequential framework. This strategy reduces unnecessary segmentation of non-lesion slices and improves overall accuracy. In addition, decomposing 3D images into 2D slices reduces model complexity while allowing information from three anatomical planes to support more accurate lesion localization. The proposed model is trained on the Anatomical Tracings of Lesions After Stroke dataset and outperforms state-of-the-art models in terms of accuracy and Dice coefficient. Moreover, the segmentation output provides feedback that helps the classification model reduce false-positive predictions.
>
---
#### [new 143] Seeing the Poem: Image-Semantic Detection of AI-Generated Modern Chinese Poetry with MLLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于AI生成诗歌检测任务，旨在解决传统方法在检测现代中文诗歌上的不足。通过引入图像语义信息，提升LLM检测效果。**

- **链接: [https://arxiv.org/pdf/2605.22654](https://arxiv.org/pdf/2605.22654)**

> **作者:** Shanshan Wang; Fengying Ye; Hanjia Lyu; Caiwen Gou; Junchao Wu; Jingming Yao; Chengzhong Xu; Jiebo Luo; Derek F. Wong
>
> **摘要:** Previous detection studies have shown that LLMs cannot be effectively used as detectors, but these studies have not addressed modern Chinese poetry. Moreover, no relevant research has explored the performance of LLMs in detecting modern Chinese poetry. This paper evaluates and enhances the performance of LLMs as detectors for modern Chinese poetry, and proposes an image-semantic guided poetry detection method. Compared with traditional detection approaches, our method innovatively incorporates images that reflect the content of the poetry. Through example-driven approaches, our method effectively integrates information such as meaning, imagery, and feeling from the image, then forms a complementary judgment with the poem text. Experimental results demonstrate that the LLM detectors based on our method outperform baseline detectors based on plain text, and even surpass the best-performing traditional detector, RoBERTa. The Gemini detector using our method achieves a Macro-F1 score of 85.65%, reaching the state-of-the-art level. The performance improvements of different LLM detectors on multiple LLMs-generated data prove the effectiveness of our method.
>
---
#### [new 144] Imagine2Real: Towards Zero-shot Humanoid-Object Interaction via Video Generative Priors
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人形机器人与物体交互任务，解决3D数据稀缺和迁移复杂的问题。提出Imagine2Real框架，通过4D点轨迹和关键点追踪实现零样本物理部署。**

- **链接: [https://arxiv.org/pdf/2605.22272](https://arxiv.org/pdf/2605.22272)**

> **作者:** Jiahe Chen; ZiRui Wang; Feiyu Jia; Xiao Chen; Xiaojie Niu; Weishuai Zeng; Tianfan Xue; Xiaowei Zhou; Jiangmiao Pang; Jingbo Wang
>
> **摘要:** Whole-body Humanoid-Object Interaction (HOI) is bottlenecked by the scarcity of high-fidelity 3D data. While video generative priors offer a promising alternative, existing methods suffer from \textit{Representation Misalignment} due to their reliance on geometric priors (e.g., explicit CAD models), and \textit{Retargeting Complexity} arising from intensive morphing and morphological mismatch. We propose Imagine2Real, a zero-shot HOI framework for flexible, geometry-free interaction. To resolve misalignment, we formulate robot and object motions as unified 4D point trajectories. To overcome retargeting complexity, our Keypoints Tracker tracks only sparse critical points (base, hands, and object), entirely bypassing the error-amplifying retargeting process. To maintain natural gaits despite these sparse signals, we utilize the latent space of a Behavior Foundation Model (BFM) as the tracker's search domain. Using a progressive training strategy, Imagine2Real learns robust behaviors with simple tracking rewards, enabling zero-shot physical deployment within a motion capture(mocap) system.
>
---
#### [new 145] Mapping Tomato Cropping Systems in California Using AlphaEarth Geospatial Embeddings and Deep Learning Analysis
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于作物分类任务，旨在解决加州加工番茄种植系统精准映射问题。通过使用AlphaEarth嵌入和深度学习方法，实现高效、准确的田块级番茄识别。**

- **链接: [https://arxiv.org/pdf/2605.21804](https://arxiv.org/pdf/2605.21804)**

> **作者:** Mohammadreza Narimani; Alireza Pourreza; Parastoo Farajpoor
>
> **备注:** 5 pages, 3 figures, 1 table. Preprint submitted to ASABE 2026 AIM
>
> **摘要:** Field-scale crop maps support supply-chain forecasting and policy, yet statewide crop identification still often depends on retrospective surveys or remote-sensing workflows built around hand-engineered spectral features. Those pipelines can be accurate, but they require repeated preprocessing and often lose robustness across years. This study evaluated whether Google DeepMind's AlphaEarth geospatial embeddings can serve as an analysis-ready alternative for mapping processing tomato systems in California. LandIQ 2018 crop polygons were used to assemble a balanced reference dataset of 4,742 tomato and 4,742 non-tomato fields. For each polygon, 64-band AlphaEarth embedding chips were extracted and aligned with binary masks, then divided into spatially independent training (n = 6,638), validation (n = 1,422), and test (n = 1,424) sets. A U-Net segmentation model was trained on AWS SageMaker using a composite masked binary cross-entropy and soft Dice loss. To complement hard predictions, Monte Carlo dropout was retained at inference and repeated 100 times per chip to estimate predictive mean and variance. On the independent test set, the model achieved 99.19% pixel accuracy, 98.69% precision, 99.40% recall, 99.04% F1 score, 98.11% intersection over union, and 99.02% chip accuracy. Uncertainty maps were consistently highest near field edges and low within field interiors. The results show that AlphaEarth embeddings retain crop-relevant spatial and temporal structure and can support accurate, field-scale tomato mapping without manual feature engineering.
>
---
#### [new 146] AwareVLN: Reasoning with Self-awareness for Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决导航模型对环境和任务理解不足的问题。提出AwareVLN框架，通过自意识推理机制提升导航效果。**

- **链接: [https://arxiv.org/pdf/2605.22816](https://arxiv.org/pdf/2605.22816)**

> **作者:** Wenxuan Guo; Xiuwei Xu; Yichen Liu; Xiangyu Li; Hang Yin; Huangxing Chen; Wenzhao Zheng; Jianjiang Feng; Jie Zhou; Jiwen Lu
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Vision-and-Language Navigation (VLN) requires an agent to ground language instructions to its own movement within a visual environment. While state-of-the-art methods leverage the reasoning capabilities of Vision-Language Models (VLMs) for end-to-end action prediction, they often lack an explicit and explainable understanding of the relationships between the agent, the instruction, and the scene. Conversely, explicitly building a scene map for heuristic planning is intuitively appealing but relies on additional 3D sensors and hinders large-scale vision-language pre-training. To bridge this gap, we propose AwareVLN, a novel framework that equips the navigation model with a self-aware reasoning mechanism, enabling it to understand the agent's state and task progress in a fully end-to-end and data-driven manner. Our approach features two key innovations: (1) a structural reasoning module that fosters spatial and task-oriented self-awareness, and (2) an automatic data engine with progress division for effective training. Extensive experiments on various datasets in Habitat simulator show our AwareVLN significantly outperforms previous state-of-the-art vision-language navigation methods. Project page: this https URL.
>
---
#### [new 147] An Open Multi-Center Whole-Body FDG PET/CT Foundation Model for Tumor Segmentation
- **分类: eess.IV; cs.AI; cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像分割任务，旨在解决PET/CT肿瘤分割中数据依赖性强、标注成本高的问题。提出一种多中心、全身体积的FDG PET/CT基础模型，提升分割效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.21835](https://arxiv.org/pdf/2605.21835)**

> **作者:** Xiaofeng Liu; Qianru Zhang; Thibault Marin; Menghua Xia; Chi Liu; Georges El Fakhri; Jinsong Ouyang
>
> **备注:** Code available at: this https URL
>
> **摘要:** The synergistic interpretation of anatomical information from computed tomography (CT) and metabolic information from positron emission tomography (PET) is important to oncologic imaging. However, existing deep learning methods for PET/CT remain largely task-specific, are often trained on single-center cohorts, or adopt dual-branch fusion schemes that delay cross-modal interaction and underutilize early spatial correspondence between PET and CT. To address these limitations, we present an open-source, multi-center, whole-body FDG PET/CT foundation model utilizing 4,997 harmonized scans from four public datasets. Our framework employs hierarchical UNet-shaped backbones with early channel-wise concatenation, enabling anatomical and metabolic features to interact from the first embedding layer onward. We further introduce a masked autoencoding objective based on zero-mean imputation, combined with a weighted global reconstruction loss. This design avoids non-physical intensity discontinuities at masked-region boundaries that arise from learnable mask tokens. On downstream AutoPET lesion segmentation, the proposed models demonstrate strong label efficiency: with only 10\% of the labeled training data, they achieve performance comparable to models trained from scratch on the full dataset. Under extreme 5-shot linear probing, joint PET/CT pretraining also achieves higher Dice scores than separated-modality pretraining. This multi-center foundation model demonstrates label efficiency and cross-modality representation learning for PET/CT tumor segmentation. It provides a robust, open-source basis for advancing automated oncologic imaging, significantly reducing the need for large-scale manual annotations in clinical practice.
>
---
#### [new 148] ST-SimDiff: Balancing Spatiotemporal Similarity and Difference for Efficient Video Understanding with MLLMs
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频处理中的计算效率问题。通过平衡时空相似性与差异性，提出ST-SimDiff框架，减少冗余并保留关键信息。**

- **链接: [https://arxiv.org/pdf/2605.22158](https://arxiv.org/pdf/2605.22158)**

> **作者:** Bingjun Luo; Tony Wang; Chaoqi Chen; Xinpeng Ding
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) face significant computational overhead when processing long videos due to the massive number of visual tokens required. To improve efficiency, existing methods primarily reduce redundancy by pruning or merging tokens based on importance or similarity. However, these approaches largely overlook a critical dimension of video content, i.e., changes and turning points, and they lack a collaborative model for spatio-temporal relationships. To address this, we propose a new perspective: similarity is for identifying redundancy, while difference is for capturing key events. Based on this, we designed a training-free framework named ST-SimDiff. We first construct a spatio-temporal graph from the visual tokens to uniformly model their complex associations. Subsequently, we employ a parallel dual-selection strategy: 1) similarity-based selection uses community detection to retain representative tokens, compressing static information; 2) temporal difference-based selection precisely locates content-changing points to preserve tokens that capture key dynamic shifts. This allows it to preserve both static and dynamic content with a minimal number of tokens. Extensive experiments show our method significantly outperforms state-of-the-art approaches while substantially reducing computational costs. Our code is available in this https URL.
>
---
#### [new 149] LACO: Adaptive Latent Communication for Collaborative Driving
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于协同驾驶任务，解决多车协作中的通信延迟与信息丢失问题。提出LACO框架，通过潜空间推理和高效信息选择提升协作性能。**

- **链接: [https://arxiv.org/pdf/2605.22504](https://arxiv.org/pdf/2605.22504)**

> **作者:** Tianhao Chen; Yuheng Wu; Dongman Lee
>
> **摘要:** Collaborative driving aims to improve safety and efficiency by enabling connected vehicles to coordinate under partial observability. Recent approaches have evolved from sharing visual features for perception to exchanging language-based reasoning through foundation models for behavioral coordination. Though communicating in language provides intuitive information, it introduces two challenges: high latency caused by autoregressive decoding and information loss caused by compressing rich internal representations into discrete tokens. To address these challenges, we analyze latent communication in collaborative driving under inherent limitations of multi-agent settings. Our analysis reveals agent identity confusion, where direct fusion of latent states entangles decision representations across vehicles. Motivated by this, we propose LACO, a training-free \textbf{LA}tent \textbf{CO}mmunication paradigm that seamlessly adapts pretrained driving models to collaborative settings. LACO introduces Iterative Latent Deliberation (ILD) for latent reasoning, Cross-Horizon Saliency Attribution (CHSA) for communication-efficient information selection, and Structured Semantic Knowledge Distillation (SSKD) to stabilize ego-centric decision making. Closed-loop experiments in CARLA show that LACO notably reduces communication and inference latency while maintaining strong collaborative driving performance.
>
---
#### [new 150] HyperBench: Standardizing and Scaling Synthetic Evaluation for Hyperspectral Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于高光谱超分辨率任务，旨在解决合成评估标准不统一的问题。提出HyperBench框架，实现标准化、大规模的合成实验与结果对比。**

- **链接: [https://arxiv.org/pdf/2605.21671](https://arxiv.org/pdf/2605.21671)**

> **作者:** Ritik Shah; Marco F. Duarte
>
> **摘要:** Hyperspectral super-resolution (HSR) reconstructs a high-spatial-resolution hyperspectral image by fusing a low-resolution hyperspectral image (LR-HSI) with a high-resolution multispectral image (HR-MSI). In the absence of real-world paired data, HSR methods are evaluated almost exclusively on synthetic experiments derived from hyperspectral datasets through Wald's protocol. Despite the protocol's widespread adoption, its practical implementation varies markedly across research works, typically relying on a single (usually Gaussian) or very few point spread functions (PSFs), one or two spectral response functions (SRFs), and a couple of spatial downsampling factors. As a result, reported performance figures are difficult to compare across the literature, in addition to being often difficult to reproduce; furthermore, they may not generalize across realistic sensing conditions. We introduce HyperBench, a unified and extensible framework that standardizes synthetic experimentation for HSR. HyperBench supports diverse degradation configurations spanning ten PSFs, four SRFs derived from operational multispectral sensors, configurable spatial downsampling factors, and matched additive white Gaussian noise; its goal is to automate large-scale evaluation and structured logging. By decoupling model development from experimental design, the framework enables reproducible, apples-to-apples cross-method comparison with minimal friction. We use HyperBench to evaluate six recently proposed HSR methods across a 70-configuration sweep on four widely used hyperspectral scenes and observe that the inter-method PSNR spread widens from approximately 5 dB on the easiest PSF to over 13 dB on the hardest - a fragility that is structurally invisible to the prevailing single-configuration evaluation protocol. HyperBench code is available at this https URL .
>
---
#### [new 151] Faithful-MR1: Faithful Multimodal Reasoning via Anchoring and Reinforcing Visual Attention
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决视觉证据感知与使用不忠实的问题。提出Faithful-MR1框架，通过锚定和强化视觉注意力提升多模态推理的准确性。**

- **链接: [https://arxiv.org/pdf/2605.22072](https://arxiv.org/pdf/2605.22072)**

> **作者:** Changyuan Tian; Zhicong Lu; Huaxing Liu; Xiang Wang; Shuai Li; Yu Chen; Wenqian Lv; Zichuan Lin; Juncheng Diao; Deheng Ye
>
> **备注:** 20 pages, 7 figures, 3 tables. Preprint
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for advancing complex reasoning in large language models, and recent work extends RLVR to multimodal large language models (MLLMs). This transfer, however, surfaces a faithfulness challenge: faithful perception of task-relevant visual evidence and faithful use of that evidence during reasoning, leading to unsatisfactory gains on multimodal benchmarks. Specifically, existing perception supervision often operates on textual descriptions rather than natively on image regions, and faithful use is largely overlooked, exposing the perception-reasoning disconnect where correctly perceived evidence is dropped or contradicted during reasoning. To close these gaps, we propose Faithful-MR1, a training framework that anchors and reinforces visual attention to address both halves of faithful multimodal reasoning. The Anchoring stage turns perception into an explicit pre-reasoning subtask, supervising a dedicated <Focus> token's attention directly against image regions rather than through textual descriptions. The Reinforcing stage exposes faithful use through counterfactual image intervention, rewarding answer-correct trajectories that concentrate visual attention where vision causally matters. Extensive experiments demonstrate that Faithful-MR1 outperforms recent multimodal reasoning baselines on both Qwen2.5-VL-Instruct 3B and 7B backbones while using substantially less training data.
>
---
#### [new 152] The Double Dilemma in Multi-Task Radiology Report Generation: A Gradient Dynamics Analysis and Solution
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于多任务放射学报告生成任务，针对线性标量化策略在平衡临床监督与报告平滑性上的不足，提出CAME-Grad优化器以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22635](https://arxiv.org/pdf/2605.22635)**

> **作者:** Erjian Zhang; Yatong Hao; Liejun Wang; Zhiqing Guo
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While multi-task learning based automatic radiology report generation (RRG) is widely adopted to ensure clinical consistency, most focus on architectural designs yet remain limited to coarse linear scalarization strategies. These strategies cannot effectively balance the hard constraints of discriminative clinical supervision with the smoothness requirements of report generation. To address these problems, we analyze the failure mechanism of linear scalarization from the perspective of gradient dynamics, utilizing the stochastic differential equation (SDE) framework to characterize it as a "Double Dilemma" of drift term deviation and diffusion term decay. Based on this, we propose a backbone-agnostic optimizer named Conflict-Averse Magnitude-Enhanced Gradient Descent (CAME-Grad). Through conflict-averse direction rectification and magnitude-enhanced energy injection, the algorithm not only ensures geometric validity, but also avoids local optimal solutions. Then, the adaptive gradient fusion mechanism is used to establish a dynamic balance between the theoretical optimal direction and the task-specific inductive bias. Experiments show that as a universal plug-and-play optimizer, CAME-Grad brings substantial and consistent improvements across eight diverse RRG methods, elevating overall clinical efficacy performance by an average of 2.3\% on MIMIC-CXR and 1.9\% on IU X-Ray. Our code is available at this https URL.
>
---
#### [new 153] LatentOmni: Rethinking Omni-Modal Understanding via Unified Audio-Visual Latent Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态理解任务，旨在解决跨模态推理中细粒度证据提取困难的问题。提出LatentOmni框架，通过统一潜在空间实现音频视觉联合推理。**

- **链接: [https://arxiv.org/pdf/2605.22012](https://arxiv.org/pdf/2605.22012)**

> **作者:** Yifan Dai; Zhenhua Wu; Bohan Zeng; Daili Hua; Jialing Liu; Bozhou Li; Yuran Wang; Chengzhuo Tong; Hao Liang; Xiaochen Ma; Junbo Niu; Tianyu Guo; Yang Shi; Yue Ding; Yiyan Ji; Bingyin Mei; Yushuo Guan; Yuanxing Zhang; Pengfei Wan; Fangcheng Fu; Wentao Zhang
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Joint audio-visual reasoning is essential for omnimodal understanding, yet current multimodal large language models (MLLMs) still struggle when reasoning requires fine-grained evidence from both modalities. A central limitation is that explicit text-based chain-of-thought (CoT) compresses continuous audio-visual signals into discrete tokens, weakening temporal grounding and shifting intermediate reasoning toward language priors. We argue that a unified latent space is a better medium for such reasoning because it preserves dense sensory information while remaining compatible with autoregressive generation. Based on this insight, we propose \textbf{LatentOmni}, a cross-modal reasoning framework that interleaves textual reasoning with audio-visual latent states. LatentOmni introduces feature-level supervision to align latent reasoning states with task-relevant sensory features and uses Omni-Sync Position Embedding (OSPE) to maintain temporal consistency between latent audio and visual states. We further construct \textbf{LatentOmni-Instruct-35K}, a dataset of audio-visual interleaved reasoning trajectories for supervising latent-space reasoning. Comprehensive evaluation across multiple audio-visual reasoning benchmarks demonstrates that LatentOmni achieves the best performance among the evaluated open-source models and consistently outperforms the Explicit Text CoT baseline, supporting latent-space joint reasoning as a promising path toward stronger omnimodal understanding.
>
---
#### [new 154] An Evidence Hierarchy for Bayesian Object Classification via OSINT-Aided Heterogeneous Sensor Fusion
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于目标分类任务，旨在解决传感器检测与分类CBRNE威胁的难题。通过构建证据层次和融合OSINT信息，提升分类准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.22259](https://arxiv.org/pdf/2605.22259)**

> **作者:** Jan Nausner; Michael Hubner
>
> **备注:** 6 pages, 1 figure; \c{opyright} 2026 The Authors. Submitted to the 2026 IEEE International Conference on Multisensor Fusion and Integration (MFI 2026). Under review
>
> **摘要:** Heterogeneous sensor fusion is vital for detecting, localizing, and classifying CBRNE threats. However, individual sensors are often only capable of detecting a subset of relevant threats with varying reliability or can even provide only indirect threat indications, making threat classification challenging. Furthermore, high clutter rates on the sensor side present a great challenge for fusion systems. Additionally, the limited availability of high quality datasets hinders the advancement of learning-based detection and classification models in smart sensors. To mitigate these sensor related shortcomings, a context-aware and domain knowledge-enhanced fusion process is proposed. First, a novel evidence hierarchy is established that enables modeling of direct, indicative, and contextual information. Second, contextual information about the environment is introduced into the fusion process, by collecting, processing, and exploiting OSINT inputs. Third, all levels of the evidence hierarchy are used to craft a Bayesian threat type classification mechanism with domain knowledge-informed priors. The proposed methodology is evaluated in simulated scenarios, and the results demonstrate the benefit of the proposed fusion approach in terms of robustness to clutter and prior mismatch, with an overall classification accuracy of up to 95%.
>
---
#### [new 155] Enhancing Visual Token Representations for Video Large Language Models via Training-Free Spatial-Temporal Pooling and Gridding
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于视频大模型任务，旨在提升视觉token表示。针对现有方法压缩视觉token时丢失时空信息的问题，提出ST-GridPool方法，结合时空分层和归一化池化，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22078](https://arxiv.org/pdf/2605.22078)**

> **作者:** Bingjun Luo; Tony Wang; Hanqi Chen; Xinpeng Ding
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have significantly advanced video understanding tasks, yet challenges remain in efficiently compressing visual tokens while preserving spatiotemporal interactions. Existing methods, such as LLaVA family, utilize simplistic pooling or interpolation techniques that overlook the intricate dynamics of visual tokens. To bridge this gap, we propose ST-GridPool, a novel training-free visual token enhancement method designed specifically for Video LLMs. Our approach integrates Pyramid Temporal Gridding (PTG), which captures multi-grained spatiotemporal interactions through hierarchical temporal gridding, and Norm-based Spatial Pooling (NSP), which preserves high-information visual regions by leveraging the correlation between token norms and semantic richness. Extensive experiments on various benchmarks demonstrate that ST-GridPool consistently enhances performance of Video LLMs without requiring costly retraining. Our method offers an efficient and plug-and-play solution for improving visual token representations. Our code is available in this https URL.
>
---
#### [new 156] GesVLA: Gesture-Aware Vision-Language-Action Model Embedded Representations
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决复杂场景中空间歧义问题。通过引入手势作为辅助指令，构建GesVLA模型，提升目标定位和人机交互效率。**

- **链接: [https://arxiv.org/pdf/2605.22812](https://arxiv.org/pdf/2605.22812)**

> **作者:** Wenxuan Guo; Ziyuan Li; Meng Zhang; Yichen Liu; Yimeng Dong; Chuxi Xu; Yunfei Wei; Ze Chen; Erjin Zhou; Jianjiang Feng
>
> **备注:** Project page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong potential for general-purpose robot manipulation by unifying perception and action. However, existing VLA systems primarily rely on textual instructions and struggle to resolve spatial ambiguity in complex scenes with multiple similar objects. To address this limitation, we introduce gesture as a parallel instruction modality and propose a Gesture-aware Vision-Language-Action model (GesVLA). Our approach encodes gesture features directly into the latent space, enabling them to participate in both high-level reasoning and low-level action generation, and adopts a dual-VLM architecture to achieve tight coupling between gesture representations and action policies. At the data level, we construct a scalable gesture data generation pipeline by rendering hand models onto real-world scene images. This reduces the sim-to-real visual gap while producing rich data with diverse motion patterns and corresponding pointing annotations. In addition, we employ a two-stage training strategy to equip the model with both gesture perception and action prediction capabilities. We evaluate our approach on multiple real-world robotic tasks, including a controlled block manipulation task for validation and more practical scenarios such as product and produce selection. Experimental results show that incorporating gesture consistently improves target grounding accuracy and human-robot interaction efficiency, especially in complex and cluttered environments. Project page: this https URL.
>
---
#### [new 157] A Task-Agnostic Algebraic Integrity Metric for Event-Camera Streams Toward SOTIF-Compliant Perception using Pearson Correlation Coefficient
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对自动驾驶感知中的事件流完整性评估问题，提出基于皮尔逊相关系数的统一度量框架，解决无任务依赖的质量评估难题。**

- **链接: [https://arxiv.org/pdf/2605.21500](https://arxiv.org/pdf/2605.21500)**

> **作者:** Arthur de Miranda Neto
>
> **备注:** 12 pages, 6 figures, 3 tables, 14 equations. Theoretical framework paper with procedural-synthetic illustrations; empirical validation on real datasets reserved for follow-up. Code and demonstration video available
>
> **摘要:** Event cameras have emerged as a high-bandwidth, low-latency sensing modality for safety-critical perception in automated driving systems (ADS), offering microsecond temporal resolution, 120-140 dB dynamic range, and intrinsic absence of motion blur. However, no task-agnostic quality metric currently operates directly on the asynchronous event stream: state-of-the-art proxies require a downstream task (e.g., detection accuracy, tracking error) to assess stream integrity, which is incompatible with the certification requirements of ISO 21448 (SOTIF) and ISO/PAS 8800:2024. The recent BiasBench benchmark (CVPR 2025) explicitly identifies this gap. This work proposes a unified algebraic framework that lifts the Pearson Correlation Coefficient (PCC), historically used in two prior works for redundancy filtering and ROI selection on frame-based images, to the three standard event representations: Time Surface, Event Frame, and Voxel Grid. The framework yields three metrics: (i) r-TS for stream integrity monitoring against an ego-motion-predicted Time Surface, (ii) r2-EF for adaptive ROI selection requiring only integer comparisons, and (iii) r-VG for temporal redundancy gating. A structural isomorphism is established between the contrast-threshold mechanism of the event camera (|Delta L| >= C) and the PCC-based change criterion, the three lifted metrics are formalized, and pipeline latency and information loss are analyzed symmetrically against the raw stream. Illustrative behavior of each metric is demonstrated on a procedural-synthetic event stream, generated by direct simulation of the emission model rather than drawn from any real or video-derived dataset, including a tunnel-dip integrity-anomaly scenario in which r_C drops from 0.93 (coherent flow) to below 0 (alarm). An explicit epistemic convention ([ESTABLISHED], [SOLID], [HYPOTH.], [OPEN]) delineates the status of every contribution.
>
---
#### [new 158] Impact of Atmospheric Turbulence and Pointing Error on Earth Observation
- **分类: cs.NI; cs.AI; cs.CV**

- **简介: 该论文属于遥感图像处理任务，旨在解决大气湍流和指向误差对地球观测图像质量的影响问题。通过构建增强的图像模拟器，生成真实退化图像，评估目标检测模型在恶劣条件下的性能。**

- **链接: [https://arxiv.org/pdf/2605.22268](https://arxiv.org/pdf/2605.22268)**

> **作者:** Celia Sánchez-de-Miguel; Antonio M. Mercado-Martínez; Beatriz Soret; Antonio Jurado-Navas; Miguel Castillo-Vázquez
>
> **备注:** Conference
>
> **摘要:** Earth Observation (EO) imagery is often degraded by atmospheric turbulence and pointing jitter; yet, these effects are rarely considered in datasets used to train AI-based detection models. Based on prior work, this paper presents an enhanced image simulator that enables the incorporation of vertical-path atmospheric turbulence and satellite pointing jitter, arising from platform and sensor vibrations, to generate physically realistic distorted images. As a case study, vessel detection is evaluated using YOLOv8 and RetinaNet on images generated by the proposed simulator under different levels of turbulence and pointing errors. Results show that YOLOv8 recall decreases from 91% under ideal conditions to 60% in the presence of weak turbulence, and falls below 40% under strong turbulence or jitter. In contrast, RetinaNet demonstrates greater robustness, maintaining approximately 75% recall across degraded conditions. These results highlight the importance of incorporating realistic physical degradations into EO training datasets to ensure reliable performance of AI-based models in operational environments, as demonstrated in maritime surveillance applications.
>
---
#### [new 159] Entropy-Guided Self-Supervised Learning for Medical Image Classification
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决标注数据少、类内差异大等问题。通过结合自监督学习和迁移学习，提出一种集成方法，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2605.21970](https://arxiv.org/pdf/2605.21970)**

> **作者:** Joao Florindo; Viviane Moura
>
> **摘要:** Accurate and robust medical image classification is paramount for early disease diagnosis and treatment planning. However, challenges such as limited annotated data, high intra-class variability, and subtle inter-class differences often hinder the performance of deep learning models. This paper introduces a synergistic deep learning framework that leverages the strengths of self-supervised learning and transfer learning for enhanced medical image classification. Our approach employs two distinct ConvNeXt-Tiny models: one pre-trained on a large-scale natural image dataset (ImageNet) and another pre-trained using an entropy-guided Masked Autoencoder (MAE) on the target medical dataset. Both models are then fine-tuned on specific medical image classification tasks. A final ensemble strategy, based on averaging predicted probabilities, is utilized to combine the complementary insights from these two models. Rigorous experimental validation across four diverse medical imaging datasets (Breast Ultrasound Images (BUSI), International Skin Imaging Collaboration (ISIC) 2018, Kvasir, and COVID) demonstrates the superior performance and robustness of our ensemble approach. The MAE pre-training significantly improves feature learning on domain-specific data, while the ImageNet pre-training provides strong generalizable features. The ensemble consistently achieves state-of-the-art results, outperforming individual models and existing methods, highlighting the efficacy of combining diverse pre-training strategies for challenging medical image analysis.
>
---
#### [new 160] Decoupling Ego-Motion from Target Dynamics via Dual-Interval Motion Cues for UAV Detection
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于目标检测任务，解决UAV视频中因自身运动导致的检测困难问题。通过分离目标运动与相机干扰，提出双间隔运动特征提取和轻量注意力机制，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.22605](https://arxiv.org/pdf/2605.22605)**

> **作者:** Liuyang Wang; Feitian Zhang
>
> **摘要:** Object detection from Unmanned Aerial Vehicles (UAVs) is challenged by severe ego-motion, camera jitter, and large scale variations. While modern detectors perform well on static images, their direct application to UAV video often fails, particularly for small objects in dynamic scenes. Existing motion-based methods either rely on computationally expensive optical flow or use single-interval differencing, which is sensitive to jitter and limited in capturing diverse motion patterns. We propose a vision-only motion-guided detection framework that decouples target motion from camera-induced disturbances. A homography-based Global Motion Compensation (GMC) first aligns adjacent frames. We then introduce a Dual-Interval Motion Extraction strategy that captures both short-term and long-term motion cues. To integrate these cues, a lightweight Motion-Guided Attention (MGA) module enhances feature representations within a Feature Pyramid Network. Experiments on the VisDrone-VID dataset demonstrate consistent improvements over a strong YOLOv8 baseline under severe ego-motion. Ablation studies further confirm the effectiveness of the dual-interval design and the proposed motion-guided attention mechanism.
>
---
#### [new 161] Don't Collapse Your Features: Why CenterLoss Hurts OOD Detection and Multi-Scale Mahalanobis Wins
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于OOD检测任务，旨在提升模型识别分布外输入的能力。针对现有方法因优化分类准确率而忽视不确定性问题，提出GOEN框架，结合多尺度特征与马氏距离，有效提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.21493](https://arxiv.org/pdf/2605.21493)**

> **作者:** Rahul D Ray
>
> **摘要:** The ability to detect out-of-distribution (OOD) inputs is fundamental to safe deployment of machine learning systems. Yet, current methods often rely on feature representations that are optimised solely for classification accuracy, neglecting the distinct requirements of epistemic uncertainty. We introduce GOEN (Geometry-Optimised Epistemic Network), a simple pipeline that combines multi-scale features, L2 normalisation, Mahalanobis distance, and a calibration head trained with real hard OOD examples. Through systematic ablation we uncover a counter-intuitive finding: CenterLoss, a popular regulariser for feature compactness, significantly degrades OOD detection performance, reducing average OOD AUROC from 0.9483 to 0.9366 despite improving classification accuracy. The best variant, GOEN-NoCenterLoss, achieves an average OOD AUROC of 0.9483, surpassing all baselines including deep ensembles (0.8827), KNN (0.8967), and ODIN (0.8870) on CIFAR-10 benchmarks, while maintaining competitive in-distribution accuracy. Our results challenge the prevailing assumption that better classification geometry automatically leads to better epistemic uncertainty. Instead, we show that overly tight feature clusters compress inter-class margins and distort the covariance structure needed for effective OOD detection. GOEN is efficient, training in under 20 minutes on a single GPU, and provides a practical blueprint for building AI systems that reliably recognise their own limitations.
>
---
#### [new 162] CryoNet: A Deep Learning Framework for Multi-Modal Debris-Covered Glacier Mapping. A Case Study of the Poiqu Basin, Central Himalaya
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于冰川识别任务，旨在解决碎屑覆盖冰川自动划分难题。通过融合多源数据构建CryoNet模型，提升冰川与冰湖的识别精度。**

- **链接: [https://arxiv.org/pdf/2605.21527](https://arxiv.org/pdf/2605.21527)**

> **作者:** Farzaneh Barzegar; Tobias Bolch; Norbert Kuehtreiber; Silvia L. Ullo
>
> **备注:** 15 pages, 10 figures, 5 tables. Preprint submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS); currently under review
>
> **摘要:** Glaciers play a critical role as freshwater reserves and indicators of climate change, yet their automatic delineation, especially for debris-covered glaciers, remains challenging due to spectral similarity with surrounding terrain. This study introduces CryoNet, a deep learning framework that leverages a rich multi-modal dataset combining Sentinel-2 optical imagery, DEM-derived topographic variables, spectral indices, Principal Component Analysis (PCA), InSAR coherence and phase, tasseled-cap features, and GLCM texture to discriminate clean-ice glaciers, debris-covered glaciers, and glacial lakes. CryoNet is an encoder-decoder CNN with nested skip connections and spatial-channel Squeeze-and-Excitation (scSE) attention, built upon a ResNet101 encoder to capture hierarchical contextual and spatial features. The study is conducted in the Poiqu Basin in the central Himalaya, and transferability is evaluated by applying the trained model to the Mont Blanc Massif in the Alps. We additionally analyse the importance of each data layer in improving glacier mapping performance. The proposed model achieves an overall IoU of 90.52%, mean Recall of 98.08%, and mean Precision of 92.26%. For debris-covered glaciers specifically, CryoNet obtains an IoU of 90.46%, a recall of 95.79%, and a precision of 94.21%. Across both per-class and overall metrics, CryoNet surpasses DeepLabV3+, SegFormer, and U-Net, taken as state-of-the-art (SOTA) references, demonstrating its effectiveness for robust glacier mapping in complex high-mountain environments.
>
---
#### [new 163] Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?
- **分类: cs.AI; cs.CV; cs.CY**

- **简介: 该论文属于多模态语言模型的人格感知任务，旨在解决模型是否真正理解行为而非仅凭表面模式判断的问题。研究提出新任务、数据集和评估体系，揭示模型存在偏见与推理不足的问题。**

- **链接: [https://arxiv.org/pdf/2605.22109](https://arxiv.org/pdf/2605.22109)**

> **作者:** Caixin Kang; Tianyu Yan; Sitong Gong; Mingfang Zhang; Liangyang Ouyang; Ruicong Liu; Bo Zheng; Huchuan Lu; Kaipeng Zhang; Yoichi Sato; Yifei Huang
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly deployed in human-facing roles where personality perception is critical, yet existing benchmarks evaluate this capability solely on numerical Big Five score prediction, leaving open whether models truly perceive personality through behavioral understanding or merely prejudge through superficial pattern matching. We address this gap with three contributions. (i) A new task: we formalize Grounded Personality Reasoning (GPR), which requires MLLMs to anchor each Big Five rating in observable evidence through a chain of rating, reasoning, and grounding. (ii) A new dataset: we release MM-OCEAN (1,104 videos, 5,320 MCQs), produced by a multi-agent pipeline with human verification, with timestamped behavioral observations, evidence-grounded trait analyses, and seven categories of cue-grounding MCQs. (iii) Benchmark and analysis: we design a three-tier evaluation (rating, reasoning, grounding) plus four sample-level failure-mode metrics: Prejudice Rate (PR), Confabulation Rate (CR), Integration-failure Rate (IR), and Holistic-grounding Rate (HR), and benchmark 27 MLLMs (13 closed, 14 open). The analysis uncovers a striking Prejudice Gap: across the field, 51% of correct ratings are not grounded in retrieved cues, and the Holistic-Grounding Rate spans only 0-33.5%. These findings expose a disconnect between getting the right score and reasoning for the right reason, charting a roadmap for grounded social cognition in MLLMs.
>
---
#### [new 164] Hierarchical Variational Policies for Reward-Guided Diffusion
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散模型在下游任务中计算成本高的问题。提出分层变分策略，降低推理成本并提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.21661](https://arxiv.org/pdf/2605.21661)**

> **作者:** Kushagra Pandey; Farrin Marouf Sofian; Jan Niklas Groeneveld; Felix Draxler; Stephan Mandt
>
> **摘要:** Adapting pretrained diffusion models to downstream objectives such as inverse problems often requires expensive test-time guidance or optimization. We propose a principled framework for generating high-quality reward-aligned samples at substantially reduced inference cost. Our approach formulates test-time adaptation as a hierarchical variational model, where control is amortized into a lightweight yet expressive stochastic policy. This formulation naturally supports few-step diffusion sampling: large step sizes enable fast inference, while the learned policy maintains sample quality by providing structured per-step control. The resulting fully amortized sampler achieves a strong quality--speed tradeoff, matching or exceeding recent test-time scaling baselines while requiring significantly less compute. For example, on 4x super-resolution, our method achieves better perceptual quality with more than 5x faster inference compared to the best-performing baseline. We further extend our approach to a semi-amortized regime that combines cheap amortized proposals with limited test-time optimization, achieving state-of-the-art perceptual quality across several challenging inverse problems.
>
---
## 更新

#### [replaced 001] LFX: Towards Unified Light Field Dense Semantic Segmentation and Salient Object Detection
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出LFX，首个统一的光场感知框架，解决不同光场表示下语义分割和显著目标检测的问题，通过视差角度建模实现跨表示适应。**

- **链接: [https://arxiv.org/pdf/2503.00747](https://arxiv.org/pdf/2503.00747)**

> **作者:** Fei Teng; Lingxin Huang; Buyin Deng; Kai Luo; Boyuan Zheng; Zheng Fang; Hong Zheng; Kunyu Peng; Jiaming Zhang; Yaonan Wang; Kailun Yang
>
> **备注:** The source code will be made publicly available at this https URL
>
> **摘要:** Light field cameras capture multi-view observations within a single exposure. However, existing studies are typically tailored to specific LF representations, leaving the field without a unified learning framework. To bridge this gap, we present LFX, the first unified framework for LF perception. LFX establishes a representation-invariant feature modulation space, enabling it to adapt to heterogeneous LF representations and diverse perception tasks. Specifically, we propose Field-of-Parallax Angular Subspace Modeling (FoP-ASM), which assigns an independent angular marker to each auxiliary view, enabling view-wise independent modeling. Meanwhile, shared manifold subspace constraints and regularization losses enforce globally consistent semantic modulation across views. Extensive evaluations across three LF benchmarks show that LFX achieves state-of-the-art results across distinct LF representations, outperforming representation-specific methods by up to 12% and 20% with 0.029/0.027 MAE for salient object detection, and achieving 84.37 mIoU for semantic segmentation. The source code will be made publicly available at this https URL.
>
---
#### [replaced 002] MagicFuse: Single Image Fusion for Visual and Semantic Reinforcement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01760](https://arxiv.org/pdf/2602.01760)**

> **作者:** Hao Zhang; Yanping Zha; Zizhuo Li; Meiqi Gong; Jiayi Ma
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** This paper focuses on a highly practical scenario: how to continue benefiting from the advantages of multi-modal image fusion under harsh conditions when only visible imaging sensors are available. To achieve this goal, we propose a novel concept of single-image fusion, which extends conventional data-level fusion to the knowledge level. Specifically, we develop MagicFuse, a novel single image fusion framework capable of deriving a comprehensive cross-spectral scene representation from a single low-quality visible image. MagicFuse first introduces an intra-spectral knowledge reinforcement branch and a cross-spectral knowledge generation branch based on the diffusion models. They mine scene information obscured in the visible spectrum and learn thermal radiation distribution patterns transferred to the infrared spectrum, respectively. Building on them, we design a multi-domain knowledge fusion branch that integrates the probabilistic noise from the diffusion streams of these two branches, from which a cross-spectral scene representation can be obtained through successive sampling. Then, we impose both visual and semantic constraints to ensure that this scene representation can satisfy human observation while supporting downstream semantic decision-making. Extensive experiments show that our MagicFuse achieves visual and semantic representation performance comparable to or even better than state-of-the-art fusion methods with multi-modal inputs, despite relying solely on a single degraded visible image. The code is publicly available at this https URL.
>
---
#### [replaced 003] VDE Bench: Evaluating The Capability of Image Editing Models to Modify Visual Documents
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2602.00122](https://arxiv.org/pdf/2602.00122)**

> **作者:** Hongzhu Yi; Yujia Yang; Yuanxiang Wang; Tong Li; Zhenyu Guan; Tianyu Zong; Jiahuan Chen; Chenxi Bao; Tiankun Yang; Haopeng Jin; Yixuan Yuan; Xinming Wang; Tao Yu; Ruilin Gao; Ruiwen Tao; Haijin Liang; Jin Ma; Jinwen Luo; Yeshani; Xinyu Zuo; Jungang Xu
>
> **摘要:** In recent years, image editing models have made significant progress, enabling users to manipulate visual content in a flexible and interactive manner through natural language instructions. However, an important yet underexplored research direction remains dense visual document image editing, which involves modifying textual content within images while faithfully preserving the original text style and background context. Existing methods primarily focus on English scenarios and images with relatively sparse text, and thus cannot adequately address dense, structurally complex documents or non-Latin scripts such as Chinese. To bridge this gap, we propose VDE Bench (Visual Doc Edit Bench), a rigorously human annotated and evaluated benchmark specifically designed to assess the performance of image editing models on bilingual Chinese-English and complex visual document editing tasks. The benchmark comprises a high quality dataset of 942 instruction based image editing samples, whose seed images encompass dense Chinese and English text documents including academic papers, posters, presentation slides, examination materials, and newspapers. Furthermore, we introduce a novel evaluation framework that systematically quantifies editing performance at the OCR parsing level, thereby enabling fine grained assessment of text modification accuracy. Based on this benchmark, we conduct a comprehensive evaluation of representative image editing models. Human verification demonstrates a high degree of consistency between human judgments and automated evaluation metrics. VDE Bench constitutes the first systematic benchmark for evaluating the performance of image editing models on bilingual dense text visual documents.
>
---
#### [replaced 004] How Well Do Models Follow Visual Instructions? VIBE: A Systematic Benchmark for Visual Instruction-Driven Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01851](https://arxiv.org/pdf/2602.01851)**

> **作者:** Huanyu Zhang; Xuehai Bai; Chengzu Li; Chen Liang; Haochen Tian; Haodong Li; Ruichuan An; Yifan Zhang; Anna Korhonen; Zhang Zhang; Liang Wang; Tieniu Tan
>
> **备注:** this https URL
>
> **摘要:** Recent generative models have achieved remarkable progress in image editing. However, existing systems and benchmarks remain largely text-guided. In contrast, human communication is inherently multimodal, where visual instructions such as sketches efficiently convey spatial and structural intent. To address this gap, we introduce VIBE, the Visual Instruction Benchmark for Image Editing with a three-level interaction hierarchy that captures deictic grounding, morphological manipulation, and causal reasoning. Across these levels, we curate high-quality and diverse test cases that reflect progressively increasing complexity in visual instruction following. We further propose a robust LMM-as-a-judge evaluation framework with task-specific metrics to enable scalable and fine-grained assessment. Through a comprehensive evaluation of 17 representative open-source and proprietary image editing models, we find that proprietary models exhibit early-stage visual instruction-following capabilities and consistently outperform open-source models. However, performance degrades markedly with increasing task difficulty even for the strongest systems, highlighting promising directions for future research.
>
---
#### [replaced 005] Improved DDIM Sampling with Moment Matching Gaussian Mixtures
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2311.04938](https://arxiv.org/pdf/2311.04938)**

> **作者:** Prasad Gabbur
>
> **备注:** 34 pages, 12 figures; Accepted to TMLR; Code open sourced
>
> **摘要:** We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ, class-conditional models trained on ImageNet, and text-to-image generation using Stable Diffusion v2.1 on COYO700M datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of 6.94 and IS of 207.85 with a GMM kernel compared to 10.15 and 196.73 respectively with a Gaussian kernel. Further, we derive novel SDE samplers for rectified flow matching models and experiment with the proposed approach. We see improvements using both 1-rectified flow and 2-rectified flow models. Code: this https URL.
>
---
#### [replaced 006] SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.01466](https://arxiv.org/pdf/2605.01466)**

> **作者:** Zhaoyang Li; Zhichao You; Tianrui Li
>
> **备注:** Accepted as a Spotlight paper at ICML 2026; camera-ready version
>
> **摘要:** Although multi-modal learning has advanced point cloud completion, the theoretical mechanisms remain unclear. Recent works attribute success to the connection between modalities, yet we identify that standard hard projection severs this connection: projecting a sparse point cloud onto the image plane yields an extremely sparse support, which hinders visual prior propagation, a failure mode we term Cross-Modal Entropy Collapse. To address this practical limitation, we propose SplAttN, which replaces hard projection with Differentiable Gaussian Splatting to produce a dense, continuous image-plane representation. By reformulating projection as continuous density estimation, SplAttN avoids collapsed sparse support, facilitates gradient flow, and improves cross-modal connection learnability. Extensive experiments show that SplAttN achieves state-of-the-art performance on PCN and ShapeNet-55/34. Crucially, we utilize the real-world KITTI benchmark as a stress test for multi-modal reliance. Counter-factual evaluation reveals that while baselines degenerate into unimodal template retrievers insensitive to visual removal, SplAttN maintains a robust dependency on visual cues, validating that our method establishes an effective cross-modal connection. Code is available at this https URL.
>
---
#### [replaced 007] CHOIR: Contact-aware 4D Hand-Object Interaction Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.20992](https://arxiv.org/pdf/2605.20992)**

> **作者:** Hao Xu; Yilin Liu; Yinqiao Wang; Chi-Wing Fu; Niloy J. Mitra
>
> **摘要:** We ask whether everyday open-world monocular videos can be turned into reusable 4D interaction primitives: articulated hand motion, object shape with 6D pose over time, and the when/where of contact. Such a capability would enable scalable mining of real interactions and, beyond reconstruction, support scene-aware synthesis and planning. However, reconstructing hand-object interaction (HOI) from challenging monocular videos remains difficult: methods often assume known objects or curated scenes, and separately estimated hands and objects easily become misaligned under clutter, occlusion, and unseen object geometries. Targeting this setting, we present CHOIR, a Contact-aware HOI Reconstruction framework for a monocular camera, using contact as an explicit coupling signal between hands and objects. CHOIR first initializes a coarse, contact-agnostic 4D HOI sequence from open-world visual priors. It then introduces a generative HOI spatial rectification module to predict ray-depth corrections and rectify hand-object relative placement, then derive initial per-frame contact correspondences on the rectified geometry. Last, a contact-aware joint optimization with dynamically updated contact constraints enforces geometric, temporal, and contact consistency. Experiments on controlled and challenging videos show that CHOIR improves object reconstruction, physical plausibility, and temporal consistency over state-of-the-art methods.
>
---
#### [replaced 008] A strongly annotated passive acoustic dataset for tropical bird monitoring
- **分类: cs.SD; cs.CV**

- **简介: 该论文提出PteroSet数据集，用于热带鸟类监测。解决监督学习所需标注数据稀缺的问题，包含大量音频标注，支持机器学习任务。**

- **链接: [https://arxiv.org/pdf/2605.20578](https://arxiv.org/pdf/2605.20578)**

> **作者:** Daniela Ruiz; Juan Sebastián Ulloa; Zhongqi Miao; Nicolás Betancourt; Maria Paula Toro-Gómez; Andrés Hernández; Bruno Demuro; Eliana Barona-Cortés; Angela Mendoza-Henao; Andrés Sierra-Ricaurte; Sebastián Pérez-Peña; Rahul Dodhia; Pablo Arbeláez; Juan M. Lavista Ferres
>
> **摘要:** Passive acoustic monitoring enables continuous, non-invasive biodiversity assessment across diverse ecosystems. The scale of these datasets has driven the adoption of machine learning, with supervised approaches showing strong performance. However, supervised methods require time-resolved annotated datasets, which remain scarce, especially in complex tropical soundscapes. We present PteroSet, a curated dataset of strongly annotated Neotropical bird vocalizations recorded in Puerto Asis (Putumayo) and Pivijay (Magdalena), Colombia, between 2023 and 2025. The dataset comprises 563 recordings (73.62 h) and 15,372 time-frequency annotations, including 6,702 events identified to the species level across 168 species. We release the annotations in a COCO-inspired JSON schema that unifies audio files, taxonomic categories, and labels for machine learning workflows. Beyond providing annotated data, PteroSet serves as a realistic benchmark that highlights key characteristics of tropical soundscapes, including acoustic co-occurrence and domain shift across recording sites. We provide a deep learning baseline for binary bird detection, demonstrating PteroSet's usability and the challenges it presents.
>
---
#### [replaced 009] SplatWeaver: Learning to Allocate Gaussian Primitives for Generalizable Novel View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.07287](https://arxiv.org/pdf/2605.07287)**

> **作者:** Yecong Wan; Fan Li; Mingwen Shao; Wangmeng Zuo
>
> **备注:** Project Page: this https URL
>
> **摘要:** Generalizable novel view synthesis aims to render unseen views from uncalibrated input images without requiring per-scene optimization. Recent feed-forward approaches based on 3D Gaussian Splatting have achieved promising efficiency and rendering quality. However, most of them assign a fixed number of Gaussians to each pixel or voxel, ignoring the spatially varying complexity of real-world scenes. Such uniform allocation often wastes Gaussian primitives in smooth regions while providing insufficient capacity for fine structures, complex geometry, and high-frequency details. This motivates us to predict region-dependent primitive cardinalities rather than impose a fixed primitive budget everywhere, enabling a more expressive 3D scene representation. Therefore, we propose SplatWeaver, a generalizable novel view synthesis framework that is able to dynamically allocate Gaussian primitives over different regions in a feed-forward manner. Specifically, SplatWeaver introduces cardinality Gaussian experts and a pixel-level routing scheme, wherein each expert specializes in producing a specific number of primitives from 0 to M, and the routing scheme coordinates these experts to adaptively determine how many Gaussian primitives should be allocated to each spatial location. Moreover, SplatWeaver incorporates a high-frequency prior with attendant guidance module and routing regularization to stabilize expert selection and promote complexity-aware allocation. By leveraging high-frequency cues, the routing process is encouraged to assign more Gaussian primitives to fine structures and textured regions, while suppressing redundancy in smooth areas. Extensive experiments across diverse scenarios show that SplatWeaver consistently outperforms state-of-the-art methods, delivering more faithful novel-view renderings with fewer Gaussian primitives. Project Page: this https URL
>
---
#### [replaced 010] Flow of Truth: Proactive Temporal Forensics for Image-to-Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.15003](https://arxiv.org/pdf/2604.15003)**

> **作者:** Yuzhuo Chen; Zehua Ma; Han Fang; Hengyi Wang; Guanjie Wang; Weiming Zhang
>
> **摘要:** The rapid rise of image-to-video (I2V) generation enables realistic videos to be created from a single image but also brings new forensic demands. Unlike static images, I2V content evolves over time, requiring forensics to move beyond 2D pixel-level tampering localization toward tracing how pixels flow and transform throughout the video. As frames progress, embedded traces drift and deform, making traditional spatial forensics ineffective. To address this unexplored dimension, we present **Flow of Truth**, the first proactive framework focusing on temporal forensics in I2V generation. A key challenge lies in discovering a forensic signature that can evolve consistently with the generation process, which is inherently a creative transformation rather than a deterministic reconstruction. Despite this intrinsic difficulty, we innovatively redefine video generation as *the motion of pixels through time rather than the synthesis of frames*. Building on this view, we propose a learnable forensic template that follows pixel motion and a template-guided flow module that decouples motion from image content, enabling robust temporal tracing. Experiments show that Flow of Truth generalizes across commercial and open-source I2V models, substantially improving temporal forensics performance.
>
---
#### [replaced 011] Demystifying Transition Matching: When and Why It Can Beat Flow Matching
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.17991](https://arxiv.org/pdf/2510.17991)**

> **作者:** Jaihoon Kim; Rajarshi Saha; Minhyuk Sung; Youngsuk Park
>
> **备注:** Code: this https URL (AISTATS 2026)
>
> **摘要:** Flow Matching (FM) underpins many state-of-the-art generative models, yet recent results indicate that Transition Matching (TM) can achieve higher quality with fewer sampling steps. This work answers the question of when and why TM outperforms FM. First, when the target is a unimodal Gaussian distribution, we prove that TM attains strictly lower KL divergence than FM for finite number of steps. The improvement arises from stochastic difference latent updates in TM, which preserve target covariance that deterministic FM underestimates. We then characterize convergence rates, showing that TM achieves faster convergence than FM under a fixed compute budget, establishing its advantage in the unimodal Gaussian setting. Second, we extend the analysis to Gaussian mixtures and identify local-unimodality regimes in which the sampling dynamics approximate the unimodal case, where TM can outperform FM. The approximation error decreases as the minimal distance between component means increases, highlighting that TM is favored when the modes are well separated. However, when the target variance approaches zero, each TM update converges to the FM update, and the performance advantage of TM diminishes. In summary, we show that TM outperforms FM when the target distribution has well-separated modes and non-negligible variances. We validate our theoretical results with controlled experiments on Gaussian distributions, and extend the comparison to real-world applications in image and video generation.
>
---
#### [replaced 012] Next-Acceleration-Scale Prediction for Autoregressive MRI Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2605.19354](https://arxiv.org/pdf/2605.19354)**

> **作者:** Yilmaz Korkmaz; Vishal M. Patel
>
> **摘要:** MRI reconstruction is an inherently ill-posed inverse problem, since incomplete measurements admit many plausible solutions. This ambiguity becomes more severe under high acceleration, where pixel-domain continuous predictors tend to average over feasible reconstructions and suppress high-frequency anatomy. We address this limitation by moving reconstruction to discrete multi-scale latent space and posing it as autoregressive next-acceleration-scale prediction. Leveraging discrete priors proven effective in visual autoregressive modeling, our method restricts the solution to compact sequences of codebook tokens, enabling sharp reconstructions even from extremely sparse measurements. This discrete autoregressive formulation also aligns naturally with modern large language model post-training techniques. Building on this observation, we introduce on-policy privileged information distillation for visual autoregressive modeling, where a teacher is provided training only privileged context that is unavailable at inference, in our case fully sampled acquisitions, and supervises a student trained on its own rollouts, leading to consistent reconstruction gains. Through extensive experiments on the fastMRI benchmark, we show that our approach delivers improved reconstruction performance across diverse sampling patterns under extreme undersampling. Project website is \href{this https URL}{here}.
>
---
#### [replaced 013] UIKA: Fast Universal Head Avatar from Pose-Free Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.07603](https://arxiv.org/pdf/2601.07603)**

> **作者:** Zijian Wu; Boyao Zhou; Liangxiao Hu; Hongyu Liu; Yuan Sun; Xuan Wang; Xun Cao; Yujun Shen; Hao Zhu
>
> **备注:** CVPR 2026 Highlight. Code: this https URL
>
> **摘要:** We present UIKA, a feed-forward animatable Gaussian head model from an arbitrary number of pose-free inputs, including a single image, multi-view captures, and smartphone-captured videos. Unlike the traditional avatar method, which requires a studio-level multi-view capture system and reconstructs a human-specific model through a long-time optimization process, we rethink the task through the lenses of model representation, network design, and data preparation. First, we introduce a UV-guided avatar modeling strategy, in which each input image is associated with a pixel-wise facial correspondence estimation. Such correspondence estimation allows us to reproject each valid pixel color from screen space to UV space, which is independent of camera pose and character expression. Furthermore, we design learnable UV tokens on which the attention mechanism can be applied at both the screen and UV levels. The learned UV tokens can be decoded into canonical Gaussian attributes using aggregated UV information from all input views. To train our large avatar model, we additionally prepare a large-scale, identity-rich synthetic training dataset. Our method significantly outperforms existing approaches in both monocular and multi-view settings.
>
---
#### [replaced 014] AutoRubric-T2I: Robust Rule-Based Reward Model for Text-to-Image Alignment
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.17602](https://arxiv.org/pdf/2605.17602)**

> **作者:** Kuei-Chun Kao; Daixuan Huo; Yuanhao Ban; Cho-Jui Hsieh
>
> **备注:** 27 pages
>
> **摘要:** Aligning Text-to-Image (T2I) generation models with human preferences increasingly relies on image reward models that score or rank generated images according to prompt alignment and perceptual quality. Existing reward models are commonly trained as Bradley-Terry (BT) preference models on large-scale human preference corpora, making them costly to train, difficult to adapt, and opaque in their evaluation criteria. Meanwhile, Vision-Language Model (VLM) judges can provide more fine-grained assessments through textual rubrics, but their manually designed or heuristically generated scoring rules may fail to reliably reflect human preferences. In this paper, we propose AutoRubric-T2I, the first rubric learning framework in T2I that automatically synthesizes and selects explicit rubrics for guiding VLM judges. AutoRubric-T2I first synthesizes reasoning traces from preference pairs into candidate rubrics, then uses a VLM judge to score paired images under each rubric, producing pairwise rubric-score differences for preference learning. To remove noisy and redundant rules, we further employ a $\ell_1$-Regularized Logistic Regression Refiner, which selects the Top-$N$ most discriminative rubrics. Extensive evaluations show that AutoRubric-T2I produces high-quality, interpretable reward signals using less than 0.01% of the annotated preference data, substantially reducing the need for large-scale reward-model training. On image reward benchmarks such as MMRB2, AutoRubric-T2I outperforms strong reward model baselines. We further validate AutoRubric-T2I as an RL reward on downstream T2I tasks, including TIIF and UniGenBench++, where it improves generation quality over scalar reward models using the Flow-GRPO pipeline on diffusion models.
>
---
#### [replaced 015] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究模型融合任务，解决共享知识过量累积问题。提出SVC方法，通过调整奇异值平衡谱分布，提升融合效果。**

- **链接: [https://arxiv.org/pdf/2602.05536](https://arxiv.org/pdf/2602.05536)**

> **作者:** Yayuan Li; Ze Peng; Jian Zhang; Jintao Guo; Yue Duan; Yinghuan Shi
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Model merging combines multiple fine-tuned models into a single model by adding their weight updates, providing a lightweight alternative to retraining. Existing methods primarily target resolving conflicts between task updates, leaving the failure mode of over-counting shared knowledge unaddressed. We show that when tasks share aligned spectral directions (i.e., overlapping singular vectors), a simple linear combination repeatedly accumulates these directions, inflating the singular values and biasing the merged model toward shared subspaces. To mitigate this issue, we propose Singular Value Calibration (SVC), a training-free and data-free post-processing method that quantifies subspace overlap and rescales inflated singular values to restore a balanced spectrum. Across vision and language benchmarks, SVC consistently improves strong merging baselines and achieves state-of-the-art performance. Furthermore, by modifying only the singular values, SVC improves the performance of Task Arithmetic by 13.0%. Code is available at this https URL.
>
---
#### [replaced 016] Findings of the Counter Turing Test: AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.20787](https://arxiv.org/pdf/2605.20787)**

> **作者:** Rajarshi Roy; Nasrin Imanpour; Ashhar Aziz; Shashwat Bajpai; Gurpreet Singh; Shwetangshu Biswas; Kapil Wanaskar; Parth Patwa; Subhankar Ghosh; Shreyas Dixit; Nilesh Ranjan Pal; Vipula Rawte; Ritvik Garimella; Amitava Das; Amit Sheth; Vasu Sharma; Aishwarya Naresh Reganti; Vinija Jain; Aman Chadha
>
> **备注:** Defactify4 @AAAI 2025
>
> **摘要:** The rapid advancements in generative AI technologies, such as Stable Diffusion, DALL-E, and Midjourney, have significantly transformed the creation of synthetic visual content. While these models enable innovation across industries, they also pose serious challenges, including misinformation, disinformation, and biased content generation. The increasing realism of AI-generated images makes their detection a pressing concern for researchers, policymakers, and industry stakeholders. In this paper, we present the findings of the Defactify 4.0 workshop, which introduced the Counter Turing Test (CT2) for AI-Generated Image Detection. The competition consisted of two key tasks: (1) binary classification of images as either AI-generated or real and (2) identification of the specific generative model responsible for an AI-generated image. To facilitate this, we developed the MS COCOAI dataset, consisting of 50,000 synthetic images from multiple generative models alongside real-world images from the MS COCO dataset. Participants employed diverse detection strategies, including convolutional neural networks (CNNs), Vision Transformers (ViTs), frequency-based analysis, contrastive learning, and multimodal techniques. The results demonstrated that while AI-generated images can be detected with high accuracy (F1-score > 0.83), identifying the exact model used remains significantly more challenging (highest F1-score: 0.4986). These findings highlight the need for improved model fingerprinting, adversarial robustness, and real-time detection mechanisms.
>
---
#### [replaced 017] SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10719](https://arxiv.org/pdf/2512.10719)**

> **作者:** Peizheng Li; Zhenghao Zhang; David Holtz; Hang Yu; Yutong Yang; Yuzhi Lai; Rui Song; Andreas Geiger; Andreas Zell
>
> **摘要:** End-to-end autonomous driving methods built on vision language models (VLMs) have undergone rapid development driven by their universal visual understanding and strong reasoning capabilities obtained from the large-scale pretraining. However, we find that current VLMs struggle to understand fine-grained 3D spatial relationships which is a fundamental requirement for systems interacting with the physical world. To address this issue, we propose SpaceDrive, a spatial-aware VLM-based driving framework that treats spatial information as explicit positional encodings (PEs) instead of textual digit tokens, enabling joint reasoning over semantic and spatial representations. SpaceDrive employs a universal positional encoder to all 3D coordinates derived from multi-view depth estimation, historical ego-states, and text prompts. These 3D PEs are first superimposed to augment the corresponding 2D visual tokens. Meanwhile, they serve as a task-agnostic coordinate representation, replacing the digit-wise numerical tokens as both inputs and outputs for the VLM. This mechanism enables the model to better index specific visual semantics in spatial reasoning and directly regress trajectory coordinates rather than generating digit-by-digit, thereby enhancing planning accuracy. Extensive experiments validate that SpaceDrive achieves state-of-the-art open-loop performance on the nuScenes dataset and the second-best Driving Score of 78.02 on the Bench2Drive closed-loop benchmark over existing VLM-based methods. Code is available at: this https URL.
>
---
#### [replaced 018] AlignPose: Generalizable 6D Pose Estimation via Multi-view Feature-metric Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20538](https://arxiv.org/pdf/2512.20538)**

> **作者:** Anna Šárová Mikeštíková; Médéric Fourmy; Martin Cífka; Josef Sivic; Vladimir Petrik
>
> **备注:** CVPR 2026
>
> **摘要:** Single-view RGB model-based object pose estimation methods achieve strong generalization but are fundamentally limited by depth ambiguity, clutter, and occlusions. Multi-view pose estimation methods have the potential to solve these issues, but existing works rely on precise single-view pose estimates or lack generalization to unseen objects. We address these challenges via the following three contributions. First, we introduce AlignPose, a 6D object pose estimation method that aggregates information from multiple extrinsically calibrated RGB views and does not require any object-specific training or symmetry annotation. Second, the key component of this approach is a new multi-view feature-metric refinement specifically designed for object pose. It optimizes a single, consistent world-frame object pose by minimizing the feature discrepancy between on-the-fly rendered object features and observed image features across all views simultaneously. Third, we report extensive experiments on six datasets (YCB-V, T-LESS, HouseCat6D, ITODD-MV, IPD, XYZ-IBD) using the BOP benchmark evaluation and show that AlignPose outperforms other published methods, especially on challenging industrial datasets where multiple views are readily available in practice.
>
---
#### [replaced 019] Towards Initialization-free Calibrated Bundle Adjustment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.23808](https://arxiv.org/pdf/2506.23808)**

> **作者:** Carl Olsson; Amanda Nilsson
>
> **摘要:** A recent series of works has shown that initialization-free BA can be achieved using pseudo Object Space Error (pOSE) as a surrogate objective. The initial reconstruction-step optimizes an objective where all terms are projectively invariant and it cannot incorporate knowledge of the camera calibration. As a result, the solution is only determined up to a projective transformation of the scene and the process requires more data for successful reconstruction. In contrast, we present a method that is able to use the known camera calibration thereby producing near metric solutions, that is, reconstructions that are accurate up to a similarity transformation. To achieve this we introduce pairwise relative rotation estimates that carry information about camera calibration. These are only invariant to similarity transformations, thus encouraging solutions that preserve metric features of the real scene. Our method can be seen as integrating rotation averaging into the pOSE framework striving towards initialization-free calibrated SfM. Our experimental evaluation shows that we are able to reliably optimize our objective, achieving convergence to the global minimum with high probability from random starting solutions, resulting in accurate near metric reconstructions.
>
---
#### [replaced 020] Label tree semantic losses for rich multi-class medical image segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.15777](https://arxiv.org/pdf/2507.15777)**

> **作者:** Junwen Wang; Oscar MacCormac; William Rochford; Aaron Kujawa; Jonathan Shapey; Tom Vercauteren
>
> **摘要:** Rich and accurate medical image segmentation is poised to underpin the next generation of AI-defined clinical practice by delineating critical anatomy for pre-operative planning, guiding real-time intra-operative navigation, and supporting precise post-operative assessment. However, commonly used learning methods for medical and surgical imaging segmentation tasks penalise all errors equivalently and thus fail to exploit any inter-class semantics in the label space. This becomes particularly problematic as the cardinality and richness of labels increases to include subtly different classes. In this work, we propose two tree-based semantic loss functions which take advantage of a hierarchical organisation of the labels. We further incorporate our losses in a recently proposed approach for training with sparse, background-free annotations to extend the applicability of our proposed losses. Extensive experiments are reported on two medical and surgical imaging segmentation tasks, namely head MRI for whole brain parcellation with full supervision and neurosurgical hyperspectral imaging for scene understanding with sparse annotations. Results demonstrate consistent improvements over the evaluated task-specific baselines, with the strongest support for the Wasserstein-based compound loss in whole-brain parcellation and for hierarchy-weighted top-level supervision in the sparse HSI setting.
>
---
#### [replaced 021] SPIRAL: Self-Evolving Action-Conditioned Video Generation via Reflective Planning Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08403](https://arxiv.org/pdf/2603.08403)**

> **作者:** Yu Yang; Yue Liao; Jianbiao Mei; Baisen Wang; Xuemeng Yang; Licheng Wen; Jiangning Zhang; Xiangtai Li; Liang Lv; Hanlin Chen; Botian Shi; Yong Liu; Shuicheng Yan; Gim Hee Lee
>
> **备注:** 42 Pages, 21 Figures, Project page at this https URL
>
> **摘要:** Long-horizon action-conditioned video generation aims to synthesize temporally coherent videos that follow complex action instructions over extended horizons, requiring procedural ordering, persistent action execution, and scene consistency beyond conventional TI2V's short-term fidelity. Existing single-shot video generation models typically operate in an open-loop manner, leading to incomplete action execution, hallucinated motions, and temporal drift. To address this, we propose SPIRAL, a closed-loop framework that performs sequential planning and iterative reflection for action-conditioned long-horizon video generation. Specifically, SPIRAL instantiates a think-act-reflect process: a PlanAgent decomposes high-level goals into sub-actions, which condition a VideoGenerator to synthesize each segment alongside a memory context, while a CriticAgent evaluates intermediate video segments to provide corrective feedback for iterative refinement. This closed-loop design further supports self-evolution by utilizing PlanAgent-proposed actions and CriticAgent-derived rewards for GRPO-based post-training to enhance the video generator's long-horizon consistency. Moreover, we introduce ActVideoGen-Dataset for task-specific training, and establish ActVideoGen-Bench as a dedicated evaluation suite for measuring action quality and temporal coherence. Experiments across multiple TI2V backbones alongside the self-evolving strategy show consistent gains on ActVideoGen-Bench and VBench, demonstrating the effectiveness of SPIRAL.
>
---
#### [replaced 022] Temporal Aware Pruning for Efficient Diffusion-based Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.17837](https://arxiv.org/pdf/2605.17837)**

> **作者:** Sheng Li; Yang Sui; Junhao Ran; Bo Yuan; Yue Dai; Xulong Tang
>
> **摘要:** Video diffusion models have recently enabled high-quality video generation with ViT-based architectures, but remain computationally intensive because generation requires attention computation over long spatiotemporal sequences. Token pruning has proven effective for ViTs and VLMs. However, most prior pruning methods are attention-based and operate per frame, failing to ensure the vital temporal coherence across frames in video generation tasks. In practice, naively adopting attention-only pruning causes noticeable degradation due to worsened background consistency, flickering, and reduced image quality. To address this, we propose TAPE, a training-free Temporal Aware Pruning for Efficient diffusion-based video generation. TAPE (i) applies temporal smoothing to align token-importance across adjacent frames and suppress selection jitter; and (ii) performs token reselection in selected layers to align token pruning with layers' diverse semantic focus and avoid error accumulation in specific areas; it also (iii) adopt a timestep-level budget scheduling that prunes aggressively at early noisy steps and relaxes pruning during fidelity-critical refinement. The experimental results show that TAPE delivers significant speedups while preserving high visual fidelity, outperforming prior token reduction approaches.
>
---
#### [replaced 023] LiWi: Layering in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.14552](https://arxiv.org/pdf/2605.14552)**

> **作者:** Yu He; Fang Li; Haoyang Tong; Lichen Ma; Xinyuan Shan; Jingling Fu; Dong Chen; Luohang Liu; Junshi Huang; Yan Li
>
> **备注:** Project Page this https URL
>
> **摘要:** Recent advances in generative models have empowered impressive layered image generation, yet their success is largely confined to graphic design domains. The layering of in-the-wild images remains an underexplored problem, limiting fine-grained editing and applications of images in real-world scenarios. Specifically, challenges remain in scalable layered data and the modeling of object interaction in natural images, such as illumination effects and structural boundary. To address these bottlenecks, we propose a novel framework for high-fidelity natural image decomposition. First, we introduce an Agent-driven Data Decomposition (ADD) pipeline that orchestrates agents and tools to synthesize layered data without manual intervention. Utilizing this pipeline, we construct a large-scale dataset, named LiWi-100k, with over 100,000 high-quality layered in-the-wild images. Second, we present a novel framework that jointly improves photometric fidelity and alpha boundary accuracy. Specifically, shadow-guided learning explicitly models the illumination effects, and degradation-restoration objective provides boundary-correction supervision by recovering clean foreground image from degraded one. Extensive experiments demonstrate that our framework achieves state-of-the-art (SoTA) performance in natural image decomposition, outperforming existing models in RGB L1 and Alpha IoU metrics. We will soon release our code and dataset.
>
---
#### [replaced 024] What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01334](https://arxiv.org/pdf/2602.01334)**

> **作者:** Yan Ma; Weiyu Zhang; Tianle Li; Linge Du; Xuyang Shen; Pengfei Liu
>
> **备注:** ICML 2026 camera ready. Code: this https URL
>
> **摘要:** Vision tool-use reinforcement learning (RL) can equip vision language models with visual operators such as crop-and-zoom and achieves strong performance gains, yet it remains unclear whether these gains are driven by improvements in tool use or evolving intrinsic capabilities. We introduce MED (Measure--Explain--Diagnose), a coarse-to-fine framework that disentangles intrinsic capability changes from tool-induced effects, decomposes the tool-induced performance difference into gain and harm terms, and probes the mechanisms driving their evolution. Across checkpoint-level analyses in the crop-and-zoom setting on two VLMs with different tool priors and six benchmarks, we find that improvements are dominated by intrinsic learning, while tool-use RL mainly reduces tool-induced harm (e.g., fewer call-induced errors and weaker tool schema interference) and yields limited progress in tool-based correction of intrinsic failures. Overall, in the crop-and-zoom setting studied here, current vision tool-use RL learns to coexist safely with tools rather than master them.
>
---
#### [replaced 025] Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02214](https://arxiv.org/pdf/2602.02214)**

> **作者:** Hongzhou Zhu; Min Zhao; Guande He; Hang Su; Chongxuan Li; Jun Zhu
>
> **备注:** Project page and the code: \href{this https URL}{this https URL}; this https URL. ICML 2026
>
> **摘要:** To achieve real-time interactive video generation, current methods distill pretrained bidirectional video diffusion models into few-step autoregressive (AR) models, facing an architectural gap when full attention is replaced by causal attention. However, existing approaches do not bridge this gap theoretically. They initialize the AR student via ODE distillation, which requires frame-level injectivity, where each noisy frame must map to a unique clean frame under the PF-ODE of an AR teacher. Distilling an AR student from a bidirectional teacher violates this condition, preventing recovery of the teacher's flow map and instead inducing a conditional-expectation solution, which degrades performance. To address this issue, we propose Causal Forcing, which uses an autoregressive teacher for ODE initialization to bridge the architectural gap, and then applies the same DMD procedure as in Self Forcing. Empirical results show that our method outperforms all baselines across all metrics, surpassing the SOTA Self Forcing by 19.3\% in Dynamic Degree, 8.7\% in VisionReward, and 16.7\% in Instruction Following. Project page: \href{this https URL}{this https URL}; the code: \href{this https URL}{this https URL}.
>
---
#### [replaced 026] Lens Privacy Sealing: A New Benchmark and Method for Physical Privacy-Preserving Action Recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.19578](https://arxiv.org/pdf/2605.19578)**

> **作者:** Mengyuan Liu; Ziyi Wang; Peiming Li; Junsong Yuan
>
> **备注:** Accepted by IEEE Transactions on Image Processing (TIP), 2026
>
> **摘要:** RGB camera-based surveillance systems enable human action recognition for public safety and healthcare, yet raise serious privacy concerns. Existing methods rely on post-capture algorithms, which fail to protect privacy during data acquisition. We propose Lens Privacy Sealing (LPS), a simple hardware solution that physically obscures camera lenses with adjustable laminating film, providing pre-sensor privacy protection at minimal cost. Unlike software methods or expensive engineered optics, LPS achieves strong privacy through stochastic multi-layer scattering that is physically irreversible. We introduce the P$^3$AR dataset for privacy-preserving action recognition, featuring both large-scale replay-captured (P$^3$AR-NTU, 114K videos) and real-world collected (P$^3$AR-PKU) subsets with privacy attribute annotations. To handle video degradation from LPS, we propose MSPNet, a single-stage framework incorporating Inter-Frame Noise Suppressor (IFNS) and Cross-Frame Semantic Aggregator (CFSA), enhanced by contrastive language-image pre-training for robust semantic extraction. Extensive experiments demonstrate that MSPNet with IFNS and CFSA nearly doubles action recognition accuracy compared to baseline methods while suppressing identity recognition to low levels. Comprehensive validation shows LPS achieves a superior privacy-utility trade-off compared to state-of-the-art hardware methods, resists reconstruction attacks including PSF inversion and data-driven recovery, and generalizes robustly across optical configurations and challenging environments. Code is available at this https URL.
>
---
#### [replaced 027] Neuroscience-inspired Staged Representation Learning with Disentangled Coarse- and Fine-Grained Semantics for EEG Visual Decoding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.16923](https://arxiv.org/pdf/2605.16923)**

> **作者:** Xiang Gao; Hui Tian; Yanming Zhu; Xuefei Yin; Alan Wee-Chung Liew
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Decoding visual information from electroencephalography (EEG) signals remains a fundamental challenge in brain-computer interfaces and medical rehabilitation. Existing EEG visual decoding methods mainly focus on learning a single global EEG embedding for cross-modal alignment, but they largely overlook the staged and hierarchical characteristics of human visual processing. To address this limitation, we propose a neuroscience-inspired staged representation learning framework that reformulates EEG visual decoding as a stage-specific representation decomposition problem. The proposed framework organizes EEG representation learning into three complementary phases: low-level visual representation learning, high-level semantic representation learning, and integrative information fusion. To strengthen semantic modeling, we further introduce a multimodal dual-level semantic learning mechanism that separates coarse label-level semantics from fine image-level visual-semantic information. In addition, semantic latent channels are introduced as computational representation channels generated from observed visual EEG signals, expanding the channel-level semantic representation space for structured semantic abstraction and cross-modal alignment. Extensive experiments on the THINGS-EEG benchmark demonstrate that the proposed method achieves superior performance under subject-dependent zero-shot evaluation and improved exact retrieval under subject-independent zero-shot evaluation. Additional analyses, including layer-wise retrieval, temporal accumulation, expanded multi-image retrieval, and ablation studies, further support the effectiveness of staged decomposition and structured semantic modeling. These results suggest that explicitly modeling staged perceptual, semantic, and integrative representations provides an effective neuroscience-inspired framework for EEG-based visual decoding.
>
---
#### [replaced 028] Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples
- **分类: cs.NE; cs.AI; cs.CR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2209.03358](https://arxiv.org/pdf/2209.03358)**

> **作者:** Nuo Xu; Kaleel Mahmood; Haowen Fang; Ethan Rathbun; Caiwen Ding; Wujie Wen
>
> **备注:** Accepted manuscript. Published in *Neurocomputing*, Volume 656, 2025, Article 131506. Available online 12 September 2025. DOI: https://doi.org/10.1016/j.neucom.2025.131506
>
> **摘要:** Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and recent advances in classification performance. However, unlike traditional deep learning approaches, the study of SNN robustness to adversarial examples remains relatively underdeveloped. In this work, we advance the adversarial attack side of SNNs through three contributions. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient estimator, even for adversarially trained SNNs. Second, using the best single surrogate gradient estimator, we analyze the transferability of adversarial attacks across SNNs, Vision Transformers (ViTs) and CNNs. Our analysis reveals two key gaps: no existing white-box attack exploits multiple surrogate gradient estimators for SNNs, and no single-model attack reliably generates adversarial examples that simultaneously fool both SNN and non-SNN models. For our third contribution, we develop the Mixed Dynamic Spiking Estimation (MDSE) attack to address these issues. MDSE uses a dynamic gradient estimation scheme to fully exploit multiple surrogate gradient estimator functions and generates adversarial examples capable of fooling SNN and non-SNN models simultaneously. MDSE is up to 91.4% more effective on SNN/ViT model ensembles and provides a 3x boost on adversarially trained SNN ensembles compared to conventional white-box attacks like Auto-PGD. Experiments cover three datasets (CIFAR-10, CIFAR-100, ImageNet) and nineteen classifier models (seven per CIFAR dataset, five for ImageNet). Our implementation of MDSE and the evaluated models is publicly available at this https URL.
>
---
#### [replaced 029] ViPS: Video-informed Pose Spaces for Auto-Rigged Meshes
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2604.17623](https://arxiv.org/pdf/2604.17623)**

> **作者:** Honglin Chen; Karran Pandey; Rundi Wu; Matheus Gadelha; Yannick Hold-Geoffroy; Ayush Tewari; Niloy J. Mitra; Changxi Zheng; Paul Guerrero
>
> **备注:** Project page: this https URL
>
> **摘要:** Kinematic rigs provide a structured interface for articulating 3D meshes but lack any associated pose space, i.e., an explicit representation of the plausible manifold of joint configurations for a given mesh. Without such a pose space, stochastic sampling or manual manipulation of raw rig parameters easily results in semantic and/or geometric violations, such as anatomical hyperextension and non-physical self-intersections. We propose Video-informed Pose Spaces (ViPS), a feedforward framework that discovers the latent distribution of valid articulations for auto-rigged meshes by distilling motion priors from a pretrained video diffusion model. Unlike existing methods that rely on scarce, artist-authored 4D datasets, or focus on reconstructing instances of individual motions, ViPS transfers generative video model priors into a universal distribution over the given rig parameterization. Differentiable geometric validators applied to the skinned mesh enforce shape-specific integrity without requiring manual regularizers. Our feedforward model reveals a smooth, compact, and controllable pose space. This, in turn, supports sampling for diverse shape variations, manifold projection for inverse kinematics, and temporally coherent trajectories for animation and keyframing. Further, the distilled 3D pose samples serve as semantic proxies to guide video diffusion, effectively closing the loop between generative 2D priors and structured 3D kinematic control. Our evaluations show that ViPS, trained solely using video priors, matches the performance of state-of-the-art models trained on synthetic artist-created 4D data in both plausibility and diversity. Additionally, as a universal model, ViPS exhibits robust zero-shot generalization to out-of-distribution species and unseen skeletal topologies.
>
---
#### [replaced 030] Do Vision Models Encode Object-Level Semantic Relatedness? A Cognitive Psychology-Inspired Benchmark
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/1709.03806](https://arxiv.org/pdf/1709.03806)**

> **作者:** Hansang Lee; Haeil Lee; Junmo Kim
>
> **摘要:** Modern vision models have achieved strong object-recognition performance, yet it remains unclear whether their representations encode object-level semantic relatedness, the meaningful connection between object concepts that supports human visual cognition. Existing benchmarks predominantly target category prediction or rely on image--text matching, leaving the visual representation itself underexamined. Drawing on cognitive psychology, we recast semantic relatedness as a triplet-ranking task and study two image-only test beds: POPORO, an existing 400-triplet psychological stimulus set repurposed for representation evaluation, and PoporoIN, a newly constructed and manually curated 1,000-triplet ImageNet-validation extension. Each triplet is annotated along two orthogonal axes: a related-target axis distinguishing Categorical Relatedness (CR, taxonomic) from conTextual Relatedness (TR, thematic), and a distractor axis distinguishing Color-matched Distractors (CD) from Shape-matched Distractors (SD). Twenty pretrained models spanning supervised, self-supervised, vision--language, and generative paradigms were evaluated by cosine similarity in an inference-only protocol. Transformer-based representations exceeded convolutional counterparts by up to 18.30 percentage points on PoporoIN at comparable ImageNet accuracy, and vision--language encoders exceeded vision-only counterparts by up to 22.50 percentage points under matched ImageNet accuracy on POPORO. Across paradigms, models recognized taxonomic targets more reliably than thematic ones and were more easily misled by shape-matched than by color-matched distractors. The benchmarks expose representational properties that classification accuracy alone does not fully predict, bridging cognitive psychology and visual representation evaluation.
>
---
#### [replaced 031] SFN-YOLO: Towards Free-Range Poultry Detection via Scale-aware Fusion Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17086](https://arxiv.org/pdf/2509.17086)**

> **作者:** Jie Chen; Yuhong Feng; Tao Dai; Hao Wang; Hongtao Chen; Zhaoxi He; Mingzhe Liu; Jiancong Bai
>
> **摘要:** Detecting and localizing poultry is essential for advancing smart poultry farming. Despite the progress of detection-centric methods, challenges persist in free-range settings due to multiscale targets, obstructions, and complex or dynamic backgrounds. To tackle these challenges, we introduce an innovative poultry detection approach named SFN-YOLO that utilizes scale-aware fusion. This approach combines detailed local features with broader global context to improve detection in intricate environments. Furthermore, we have developed a new expansive dataset (M-SCOPE) tailored for varied free-range conditions. Comprehensive experiments demonstrate our model achieves an mAP of 80.7% with just 7.2M parameters, which is 35.1% fewer than the benchmark, while retaining strong generalization capability across different domains. The efficient and real-time detection capabilities of SFN-YOLO support automated smart poultry farming.
>
---
#### [replaced 032] From Spherical to Gaussian: A Comparative Analysis of Point Cloud Cropping Strategies in Large-Scale 3D Environments
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.02098](https://arxiv.org/pdf/2605.02098)**

> **作者:** Maximilian Kellner; Dominik Merkle; Michael Brunklaus; Alexander Reiterer
>
> **摘要:** Large-scale 3D point clouds can consist of hundreds of millions of points. Even after downsampling, these point clouds are too large for modern 3D neural networks. In order to develop a semantic understanding of the scene, the point clouds are divided into smaller subclouds that can be processed. Typically, this division is done using spherical crops, resulting in a loss of surrounding geometric context. To address this issue, we propose alternative methods that produce subclouds with larger crop sizes while maintaining a similar number of points. Specifically, we compare exponential, Gaussian, and linear cropping methods with the spherical method. We evaluated three 3D deep learning model architectures using multiple indoor and outdoor environment datasets. Our results demonstrate that altering the cropping strategy can enhance model performance, especially for large-scale outdoor scenes, yielding new state-of-the-art results. Code is available at this https URL
>
---
#### [replaced 033] MedFM-Robust: Benchmarking Robustness of Medical Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.19027](https://arxiv.org/pdf/2605.19027)**

> **作者:** Xiangxiang Cui; Tianjin Huang; Yifang Wang; Lijie Hu; Lu Yin
>
> **备注:** MICCAI2026
>
> **摘要:** Medical foundation models (MedFMs) have emerged as transformative tools in healthcare, demonstrating capabilities across diverse clinical applications. These models can be broadly categorized into two paradigms: Medical Vision-Language Models (Med-VLMs) and segmentation foundation models. Med-VLMs range from medical-specialized models such as LLaVA-Med and MedGemma, to general-purpose models like GPT-4o and Gemini, all capable of medical image understanding tasks including visual question answering (VQA), report generation, and visual grounding. Concurrently, the Segment Anything Model (SAM) has catalyzed a new generation of medical segmentation models, with adaptations like SAM-Med2D and MedSAM. The widespread clinical deployment of these models thus necessitates rigorous evaluation of their reliability under real-world conditions.
>
---
#### [replaced 034] 4D Radar Semantic Segmentation of People in Field Conditions Using Temporal Multi-View Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于4D雷达语义分割任务，旨在解决恶劣环境下人员检测问题。通过设计TMVA4D网络，利用雷达数据实现准确的人员分类。**

- **链接: [https://arxiv.org/pdf/2404.05307](https://arxiv.org/pdf/2404.05307)**

> **作者:** Mikael Skog; Oleksandr Kotlyar; Vladimír Kubelka; Martin Magnusson
>
> **摘要:** Reliable people detection is crucial for the safe autonomy of mobile robots and heavy vehicles, both on roads and in industrial settings like mining and construction. However, common sensors like cameras or lidars are prone to failure in adverse conditions such as dust, fog, or smoke, which limits their use in real-world robotic systems. Radar, on the other hand, delivers robust measurements in a wide range of environmental conditions. In particular, modern high-resolution 4D imaging radars provide 4D point clouds across range, azimuth, and elevation, as well as per-point Doppler velocity data, well suited for robot perception. We propose TMVA4D, a family of artificial neural network architectures based on CNN and ConvLSTM encoders that leverage the 4D radar modality for semantic segmentation. The architectures are trained to distinguish between background and person classes using a series of 2D projections of the 4D radar data, encompassing elevation, azimuth, range, and Doppler velocity dimensions. Evaluated across several operational sites, our models achieve promising performance (Dice 75.9%, IoU 61.2% for class person) even in low-visibility conditions. The data and code will be made publicly available upon publication.
>
---
#### [replaced 035] Skarimva: Skeleton-based Action Recognition is a Multi-view Application
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23231](https://arxiv.org/pdf/2602.23231)**

> **作者:** Daniel Bermuth; Alexander Poeppel; Wolfgang Reif
>
> **摘要:** Human action recognition plays an important role when developing intelligent interactions between humans and machines. While there is a lot of active research on improving the machine learning algorithms for skeleton-based action recognition, not much attention has been given to the quality of the input skeleton data itself. This work demonstrates that by making use of multiple camera views to triangulate more accurate 3D~skeletons, the performance of state-of-the-art action recognition models can be improved significantly. This suggests that the quality of the input data is currently a limiting factor for the performance of these models. Based on these results, it is argued that the cost-benefit ratio of using multiple cameras is very favorable in most practical use-cases, therefore future research in skeleton-based action recognition should consider multi-view applications as the standard setup.
>
---
#### [replaced 036] Vendi Novelty Scores for Out-of-Distribution Detection
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.10062](https://arxiv.org/pdf/2602.10062)**

> **作者:** Amey P. Pasarkar; Adji Bousso Dieng
>
> **摘要:** Out-of-distribution (OOD) detection is critical for the safe deployment of machine learning systems. Existing post-hoc detectors typically rely on model confidence scores or likelihood estimates in feature space, often under restrictive distributional assumptions. In this work, we introduce a third paradigm and formulate OOD detection from a diversity perspective. We propose the Vendi Novelty Score (VNS), an OOD detector based on the Vendi Scores (VS), a family of similarity-based diversity metrics. VNS quantifies how much a test sample increases the VS of the in-distribution feature set, providing a principled notion of novelty that does not require density modeling. VNS is linear-time, non-parametric, and naturally combines class-conditional (local) and dataset-level (global) novelty signals. Across multiple image classification benchmarks and network architectures, VNS achieves state-of-the-art OOD detection performance. Remarkably, VNS retains this performance when computed using only 1% of the training data, enabling deployment in memory- or access-constrained settings.
>
---
#### [replaced 037] The Expense of Seeing: Attaining Trustworthy Multimodal Reasoning Within the Monolithic Paradigm
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.20665](https://arxiv.org/pdf/2604.20665)**

> **作者:** Karan Goyal
>
> **备注:** Addresses practical viability of Vlabel construction. Writing is grounded. Acknowledgement is duly added
>
> **摘要:** The rapid proliferation of Vision-Language Models (VLMs) is often framed as enabling unified multimodal knowledge discovery but rests on an under-examined assumption: that current VLMs faithfully synthesise multimodal data. We argue they often do not, and this gap reflects a trustworthiness problem in the dominant Vision Encoder-Projector-LLM paradigm. Rather than extracting grounded knowledge from visual inputs, state-of-the-art models frequently exhibit functional blindness, i.e., exploiting strong language priors to bypass severe visual representation bottlenecks. In this work, we challenge the conventional methodology of multimodal evaluation, which relies on data ablation or new dataset creation and therefore conflates dataset biases with architectural incapacity. We propose an information-theoretic departure: the Modality Translation Protocol, designed to quantify what we call the Expense of Seeing. By translating semantic payloads rather than ablating them, we formulate three novel metrics -- the Toll (ToS), Curse (CoS), and Fallacy (FoS) of Seeing -- culminating in the Semantic Sufficiency Criterion (SSC). Furthermore, we hypothesise a Divergence Law of Multimodal Scaling: as the underlying language engines scale to unprecedented reasoning capabilities, the penalty of the visual knowledge bottleneck may increase rather than diminish. We argue the community should move beyond "multimodal gain" as a primary evaluation target. By elevating the SSC from a passive diagnostic constraint to an active architectural blueprint, we provide a foundation for guiding the next generation of AI systems toward genuine multimodal reasoning.
>
---
#### [replaced 038] Not All Starting Points Are Equal: Pre-trained Priors and Their Outsized Impact on Person Identification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.17640](https://arxiv.org/pdf/2507.17640)**

> **作者:** Thomas M. Metz; Matthew Q. Hill; Alice J. O'Toole
>
> **摘要:** Recent years have seen an explosion of diverse general purpose pre-training methodologies for computer vision. However, the impact that these pre-training methodologies have on person identification tasks (re-id) remains under-explored. We show that under equated domain adaptation pipelines, there is dramatic variance in person identification outcomes using different starting models (architectures and pre-trained weights). We show that a range of intuitive explanations for differing downstream performance on a range of re-id tests are insufficient and propose that pre-trained weights serve as a strong prior to the weights learned during domain adaptation. This framework allows for domain adapted solutions to be viewed as a maximum probability point estimate of the Gibbs posterior with the pre-trained weights acting as a prior. Under this framework, we show that large, pre-trained foundation models with simple domain adaptation achieve SOTA solutions on a range of re-id datasets (Market, PRCC, DeepChange, BTS) with solutions that are very close in the parameter space to the starting parameters. Moreover, we perform ablations on these solutions and show that they can be reached with small transfer sets and with varying transfer datasets but are sensitive to choice of optimizer, weight-decay, and loss function. Ultimately, we propose that the simple approach of direct fine-tuning using large vision foundation models (CLIP, Dino, EVA, AIM, etc.) needs to serve as an important baseline for future work in re-id.
>
---
#### [replaced 039] LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20785](https://arxiv.org/pdf/2511.20785)**

> **作者:** Zuhao Yang; Sudong Wang; Kaichen Zhang; Keming Wu; Sicong Leng; Yifan Zhang; Bo Li; Chengwei Qin; Shijian Lu; Xingxuan Li; Lidong Bing
>
> **备注:** CVPR 2026
>
> **摘要:** Large multimodal models (LMMs) have shown great potential for video reasoning with textual Chain-of-Thought. However, they remain vulnerable to hallucinations, especially when processing long-form videos where evidence is sparse and temporally dispersed. Inspired by how humans comprehend long videos - by first skimming globally and then examining relevant clips for details - we introduce LongVT, an end-to-end agentic framework that enables "Thinking with Long Videos" via interleaved Multimodal Chain-of-Tool-Thought. Specifically, we exploit LMMs' inherent temporal grounding ability as a native video cropping tool to zoom in on a specific video clip and resample finer-grained video frames. This global-to-local reasoning loop continues until answers are grounded in retrieved visual evidence. Given the scarcity of fine-grained question-answering (QA) data for the long video reasoning task, we curate and will release a data suite named VideoSIAH to facilitate both training and evaluation. Specifically, our training dataset consists of 247.9K samples for tool-integrated cold-start supervised fine-tuning, 1.6K samples for agentic reinforcement learning, and 15.4K samples for agentic reinforcement fine-tuning, respectively. Our evaluation benchmark consists of 1,280 QA pairs that are carefully curated through a semi-automatic data pipeline with human-in-the-loop validation. With a meticulously designed three-stage training strategy and extensive empirical validation, LongVT consistently outperforms existing strong baselines across four challenging long-video understanding and reasoning benchmarks. Our codes, data, and model checkpoints are publicly available at this https URL .
>
---
#### [replaced 040] DriveMA: Rethinking Language Interfaces in Driving VLAs with One-Step Meta-Actions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.21273](https://arxiv.org/pdf/2605.21273)**

> **作者:** Weicheng Zheng; Yixin Huang; Qiao Sun; Derun Li; Hang zhao
>
> **备注:** We withdraw this submission because the current version contains a mismatch between the paper title, conceptual framing, and the intended contribution of the work. To avoid potential misunderstanding by readers, the authors have decided to withdraw this version and substantially revise the title, organization, and presentation before any future submission
>
> **摘要:** Driving Vision-Language-Action Models (Driving VLAs) commonly introduce natural-language reasoning as an intermediate interface for end-to-end planning, but reasoning-centric interfaces face three practical bottlenecks: obtaining high-quality reasoning annotations is difficult, generating and understanding long reasoning chains is challenging for compact models, and inference latency is substantially increased. In this paper, we rethink the design of language interfaces in Driving VLAs and show that concise one-step meta-actions are a simple yet effective alternative to verbose reasoning. Meta-actions provide semantic decision grounding while remaining low-entropy, and being automatically derivable from expert trajectories, enabling scalable supervision and reliable trajectory conditioning. Building on this interface, we propose DriveMA, which combines action-centric supervised training with a turn-level credit-assignment reinforcement learning framework that jointly optimizes meta-action correctness, trajectory quality, and trajectory--meta-action consistency. Experiments show that DriveMA already achieves a new state of the art on the Waymo End-to-End Driving Challenge with a 2B model, reaching a Rater Feedback Score (RFS) of 8.060, while its 4B version further improves the state of the art to 8.079; DriveMA also obtains competitive performance on NAVSIM. Ablations demonstrate that one-step meta-actions offer a better practical trade-off between expressiveness, predictability, and inference efficiency than natural-language reasoning or finer-grained action sequences. Code, data, and models will be released to facilitate future research.
>
---
#### [replaced 041] Towards Selection of Large Multimodal Models as Engines for Burned-in Protected Health Information Detection in Medical Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.02014](https://arxiv.org/pdf/2511.02014)**

> **作者:** Tuan Truong; Guillermo Jimenez Perez; Pedro Osorio; Matthias Lenga
>
> **备注:** Accepted at EMBC 2026
>
> **摘要:** The detection of Protected Health Information (PHI) in medical imaging is critical for safeguarding patient privacy and ensuring compliance with regulatory frameworks. Traditional detection methodologies predominantly utilize Optical Character Recognition (OCR) models in conjunction with named entity recognition. However, recent advancements in Large Multimodal Model (LMM) present new opportunities for enhanced text extraction and semantic analysis. In this study, we systematically benchmark three prominent closed and open-sourced LMMs, namely GPT-4o, Gemini 2.5 Flash, and Qwen 2.5 7B, utilizing two distinct pipeline configurations: one dedicated to text analysis alone and another integrating both OCR and semantic analysis. Our results indicate that LMM exhibits superior OCR efficacy (WER: 0.03-0.05, CER: 0.02-0.03) compared to conventional models like EasyOCR. However, this improvement in OCR performance does not consistently correlate with enhanced overall PHI detection accuracy. The strongest performance gains are observed on test cases with complex imprint patterns. In scenarios where text regions are well readable with sufficient contrast, and strong LMMs are employed for text analysis after OCR, different pipeline configurations yield similar results. Furthermore, we provide empirically grounded recommendations for LMM selection tailored to specific operational constraints and propose a deployment strategy that leverages scalable and modular infrastructure.
>
---
#### [replaced 042] DanceHMR: Hand-Aware Whole-Body Human Mesh Recovery from Monocular Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18102](https://arxiv.org/pdf/2605.18102)**

> **作者:** Wenhao Shen; Ming Zhou; Hengyuan Zhang; Siyuan Bian; Youjiang Xu; Xi Lin
>
> **备注:** I would like to withdraw my arXiv paper submission due to company-related approval and authorization requirements
>
> **摘要:** Monocular video human mesh recovery is essential for digital humans, avatar animation, and embodied simulation, where both temporal stability and expressive whole-body motion are required. Existing video HMR methods produce coherent body motion but often overlook detailed hand articulation, while image-based whole-body methods recover SMPL-X meshes independently per frame, often leading to jittery and inaccurate hand motion. We present a temporally coherent whole-body HMR framework for challenging in-the-wild monocular videos. Our model unifies body context and part-specific hand observations through residual body-hand fusion, enabling stable body motion and detailed hand recovery within a single temporal architecture. We further introduce close-up-aware augmentation to improve robustness under upper-body framing. Experiments on whole-body and body-only benchmarks demonstrate improved hand reconstruction and competitive body accuracy. Our method also produces temporally stable and 2D-consistent SMPL-X motion in challenging real-world videos.
>
---
#### [replaced 043] VChain: Chain-of-Visual-Thought for Reasoning in Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.05094](https://arxiv.org/pdf/2510.05094)**

> **作者:** Ziqi Huang; Ning Yu; Gordon Chen; Haonan Qiu; Paul Debevec; Ziwei Liu
>
> **备注:** ACL 2026 (Findings Paper), ICCV 2025 Workshop Outstanding Paper Award, Project page: this https URL
>
> **摘要:** Recent video generation models can produce smooth and visually appealing clips, but they often struggle to synthesize complex dynamics with a coherent chain of consequences. Accurately modeling visual outcomes and state transitions over time remains a core challenge. In contrast, large language and multimodal models (e.g., GPT-4o) exhibit strong visual state reasoning and future prediction capabilities. To bridge these strengths, we introduce VChain, a novel inference-time chain-of-visual-thought framework that injects visual reasoning signals from multimodal models into video generation. Specifically, VChain contains a dedicated pipeline that leverages large multimodal models to generate a sparse set of critical keyframes as snapshots, which are then used to guide the sparse inference-time visual-state adaptation of a pre-trained video generator only at these key moments. Our approach is tuning-efficient, introduces minimal overhead and avoids dense supervision. Extensive experiments on complex, multi-step scenarios show that VChain significantly enhances the quality of generated videos.
>
---
#### [replaced 044] InfVSR: Breaking Length Limits of Generic Video Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00948](https://arxiv.org/pdf/2510.00948)**

> **作者:** Ziqing Zhang; Kai Liu; Zheng Chen; Xi Li; Yucong Chen; Bingnan Duan; Linghe Kong; Yulun Zhang
>
> **备注:** Code and model are available at this https URL
>
> **摘要:** Real-world videos often extend over thousands of frames. Existing generative video super-resolution (VSR) approaches, however, face two persistent challenges when processing long sequences: (1) inefficiency due to the heavy cost of multi-step denoising for full-length sequences; and (2) poor consistency is hindered by temporal decomposition that causes artifacts and discontinuities. To break these limits, we propose InfVSR, which reformulates VSR as an autoregressive-one-step-diffusion paradigm, and enables streaming inference with video diffusion priors. First, we adapt the pretrained DiT into a causal structure, maintaining both local and global coherence via rolling KV-cache and joint visual guidance. Second, we distill the diffusion process into a single step efficiently, with patch-wise pixel supervision and cross-chunk distribution matching. To fill the gap in long-form video evaluation, we build a new benchmark tailored for extended sequences and further introduce semantic-level metrics to comprehensively assess temporal consistency. Our method pushes the frontier of long-form VSR, achieves state-of-the-art quality with enhanced semantic consistency, and delivers up to 58x speed-up over existing methods such as MGLD-VSR. Our code and models are available at this https URL.
>
---
#### [replaced 045] Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多模态语言模型研究，旨在解决 embodied 模型能力瓶颈问题。通过构建 BEAR 基准进行细粒度技能评估，发现感知与时空建模是主要问题，并提出 BEAR-Agent 提升性能。**

- **链接: [https://arxiv.org/pdf/2510.08759](https://arxiv.org/pdf/2510.08759)**

> **作者:** Yu Qi; Haibo Zhao; Ziyu Guo; Siyuan Ma; Ziyan Chen; Yaokun Han; Renrui Zhang; Zitiantao Lin; Yizhe Zhu; Shiji Xin; Yijian Huang; Boce Hu; Kai Cheng; Peiheng Wang; Jiazheng Liu; Jiayi Zhang; Yizhe Zhu; Wenqing Wang; Yiran Qin; Haojie Huang; Lawson L.S. Wong
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Understanding the capability bottlenecks of embodied multimodal large language models (MLLMs) is crucial for improving embodied agents. However, existing embodied benchmarks mainly focus on task-level evaluation and fail to provide actionable insights into the underlying causes of model failures. To address this limitation, we introduce BEAR, a benchmark that decomposes embodied tasks into 14 atomic skills for fine-grained skill-level evaluation. BEAR comprises 4,469 interleaved image-video-text samples spanning 14 skills across 6 categories, ranging from low-level perception to high-level planning. We evaluate 20 MLLMs on BEAR under a hierarchical skill-level diagnosis framework and uncover two key findings: (1) perceptual capabilities are major bottlenecks behind reasoning failures, and (2) current models suffer from unstable spatiotemporal modeling that remains largely unexposed in prior benchmarks. Motivated by these findings, we further propose BEAR-Agent, a multimodal conversational agent that augments MLLMs with visual and spatial reasoning tools. BEAR-Agent substantially improves performance across embodied skills, achieving a relative improvement of 17.5% on GPT-5 over the base model on BEAR, while also outperforming strong baselines in both simulation and real-world robotic experiments. Project page: this https URL
>
---
#### [replaced 046] DocAtlas: Multilingual Document Understanding Across 80+ Languages
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出DocAtlas，解决多语言文档理解问题，尤其针对低资源语言。构建跨82语言的高质量OCR数据集，采用双管道生成结构化标注，提升模型多语言适应能力。**

- **链接: [https://arxiv.org/pdf/2605.12623](https://arxiv.org/pdf/2605.12623)**

> **作者:** Ahmed Heakl; Youssef Mohamed; Abdullah Sohail; Rania Elbadry; Ahmed Nassar; Peter W. J. Staar; Fahad Shahbaz Khan; Imran Razzak; Salman Khan
>
> **备注:** Under submission
>
> **摘要:** Multilingual document understanding remains limited for low-resource languages due to scarce training data and model-based annotation pipelines that perpetuate existing biases. We introduce DocAtlas, a framework that constructs high-fidelity OCR datasets and benchmarks covering 82 languages and 9 evaluation tasks. Our dual pipelines, differential rendering of native DOCX documents and synthetic LaTeX-based generation for right-to-left scripts produce precise structural annotations in a unified DocTag format encoding layout, text, and component types, without learned models for core annotation. Evaluating 16 state-of-the-art models reveals persistent gaps in low-resource scripts. We show that Direct Preference Optimization (DPO) using rendering-derived ground truth as positive signal achieves stable multilingual adaptation, improving both in-domain (+1.9%) and out-of-domain (+1.8%) accuracy without measurable base-language degradation, where supervised fine-tuning degrades out-of-domain performance by up to 21%. Our best variant, DocAtlas-DeepSeek, improves +1.7% over the strongest baseline. Code is available at this https URL .
>
---
#### [replaced 047] Quantifying Rodda and Graham Gait Classification from 3D Makerless Kinematics derived from a Single-view Video in a Heterogeneous Pediatric Clinical Cohort
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.11314](https://arxiv.org/pdf/2605.11314)**

> **作者:** Lauhitya Reddy; Seth Donahue; Jeremy Bauer; Susan Sienko; Anita Bagley; Joseph Krzak; Maura Eveld; Karen Kruger; Ross Chafetz; Vedant Kulkarni; Hyeokhyen Kwon
>
> **备注:** 29 pages, 8 figures, 9 tables (including 1 supplementary table); manuscript prepared in PLOS ONE format
>
> **摘要:** Cerebral Palsy (CP) is a neurological disorder of movement and the most common cause of lifelong physical disability in childhood. Approximately 75% of children with CP are ambulatory, and accurate gait assessment is central to preserving walking function, which deteriorates by mid-adulthood in a quarter to half of adults with CP. The Rodda and Graham classification system quantifies sagittal-plane gait deviations using ankle and knee z-scores derived from 3D Instrumented Gait Analysis (3D-IGA), but 3D-IGA is expensive and limited to specialized centers, while observational assessment shows only moderate inter-rater agreement. We developed a markerless gait analysis pipeline that quantifies Rodda and Graham knee and ankle z-scores directly from single-view clinical gait videos. Across 1,058 bilateral limb samples from 529 trials of 152 children (88 male, 63 female; age 12.1 $\pm$ 4.0 years; 60 distinct primary diagnoses, cerebral palsy the most common at $n=54$), the sagittal-view model achieved $R^2 = 0.80 \pm 0.02$ and CCC $= 0.89 \pm 0.02$ for knee z-scores and $R^2 = 0.57 \pm 0.02$ and CCC $= 0.72 \pm 0.02$ for ankle z-scores against 3D-IGA. Binary screening for excess knee flexion achieves AUROC $= 0.88$, correctly identifying 83% of affected children, and applying Rodda and Graham rules yields $43 \pm 1$% 7-class accuracy with macro-AUROC $= 0.78 \pm 0.01$, ankle prediction error remaining the primary bottleneck. Beyond cross-sectional screening, continuous z-scores support longitudinal trajectory tracking across visits, providing a quantitative substrate for monitoring disease progression and treatment response unavailable from observational scales. These results demonstrate the feasibility of video-based z-score estimation, excess-flexion screening, and longitudinal trajectory tracking as a path toward scalable, objective gait assessment in low-resource clinical settings.
>
---
#### [replaced 048] PartCo: Part-Level Correspondence Priors Enhance Category Discovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22769](https://arxiv.org/pdf/2509.22769)**

> **作者:** Fernando Julio Cendra; Kai Han
>
> **备注:** ICML 2026, Project page: this https URL
>
> **摘要:** Generalized Category Discovery (GCD) aims to identify both known and novel categories within unlabeled data by leveraging a set of labeled examples from known categories. Existing GCD methods primarily depend on semantic labels and global image representations, often overlooking the detailed part-level cues that are crucial for distinguishing closely related categories. In this paper, we introduce PartCo, short for Part-Level Correspondence Prior, a novel framework that enhances category discovery by incorporating part-level visual feature correspondences. By leveraging part-level relationships, PartCo captures finer-grained semantic structures, enabling a more nuanced understanding of category relationships. Importantly, PartCo seamlessly integrates with existing GCD methods without requiring significant modifications. Our extensive experiments on multiple benchmark datasets demonstrate that PartCo significantly improves the performance of current GCD approaches, outperforming most existing methods by bridging the gap between semantic labels and part-level visual compositions, thereby setting new benchmarks for GCD.
>
---
#### [replaced 049] Depth Augmented and FE Free 3D/2D Liver Registration for Laparoscopic Liver AR
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17517](https://arxiv.org/pdf/2602.17517)**

> **作者:** Hanyuan Zhang; Lucas He; Runlong He; Weixi Yi; Abdolrahim Kadkhodamohammadi; Danail Stoyanov; Brian R. Davidson; Evangelos B. Mazomenos; Matthew J. Clarkson
>
> **摘要:** Augmented reality (AR) guidance in laparoscopic liver surgery requires accurate registration of preoperative 3D models to intraoperative 2D video, but remains challenging due to partial visibility, specularities, and tissue deformation. Existing methods often rely on contour-based rigid initialization and finite-element (FE) models for deformable registration, increasing modeling and engineering complexity. We present a depth-augmented, FE-free 3D--2D registration pipeline that combines robust rigid initialization with patient-specific non-rigid refinement. For rigid alignment, we adapt the RefineNet module of FoundationPose to laparoscopic liver scenes by using multi-class contour maps and monocular depth for relative pose refinement. For deformable alignment, we construct a patient-specific statistical deformation model from non-rigid ICP (NICP) correspondences and optimize pose and shape parameters using a coarse-to-fine L-BFGS-B strategy. On a public clinical laparoscopic liver dataset, the proposed method achieves a mean target registration error (TRE) of 14.73\,mm under a controlled manual-contour setting designed to isolate registration performance. Ablation studies show that monocular depth improves rigid initialization over contour-only inputs, while tumor-mapping analysis indicates that good surface alignment does not necessarily translate into lower target localization error. On an external dataset without ground truth, the method produces visually plausible overlays for qualitative assessment. These results suggest that depth-augmented pose refinement and FE-free statistical deformation modeling provide a promising alternative to FE-based pipelines for controlled 3D--2D liver registration in surgical AR.
>
---
#### [replaced 050] Dual-Anchoring: Addressing State Drift in Vision-Language Navigation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.17473](https://arxiv.org/pdf/2604.17473)**

> **作者:** Kangyi Wu; Pengna Li; Kailin Lyu; Xi Lin; Lin Zhao; Qingrong He; Jinjun Wang; Jianyi Liu
>
> **摘要:** Vision-Language Navigation(VLN) requires an agent to navigate through 3D environments by following natural language instructions. While recent Video Large Language Models(Video-LLMs) have largely advanced VLN, they remain highly susceptible to State Drift in long scenarios. In these cases, the agent's internal state drifts away from the true task execution state, leading to aimless wandering and failure to execute essential maneuvers in the instruction. We attribute this failure to two distinct cognitive deficits: Progress Drift, where the agent fails to distinguish completed sub-goals from remaining ones, and Memory Drift, where the agent's history representations degrade, making it lose track of visited landmarks. In this paper, we propose a Dual-Anchoring Framework that explicitly anchors the instruction progress and history representations. First, to address progress drift, we introduce Instruction Progress Anchoring, which supervises the agent to generate structured text tokens that delineate completed versus remaining sub-goals. Second, to mitigate memory drift, we propose Memory Landmark Anchoring, which utilizes a Landmark-Centric World Model to retrospectively predict object-centric embeddings extracted by the Segment Anything Model, compelling the agent to explicitly verify past observations and preserve distinct representations of visited landmarks. Facilitating this framework, we curate two extensive datasets: 3.6 million samples with explicit progress descriptions, and 937k grounded landmark data for retrospective verification. Extensive experiments in both simulation and real-world environments demonstrate the superiority of our method, achieving a 15.2% improvement in Success Rate and a remarkable 24.7% gain on long-horizon trajectories. To facilitate further research, we will release our code, data generation pipelines, and the collected datasets.
>
---
#### [replaced 051] ParaVT: Taming the Tool Prior Paradox for Parallel Tool Use in Agentic Video Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.20342](https://arxiv.org/pdf/2605.20342)**

> **作者:** Zuhao Yang; Kaichen Zhang; Sudong Wang; Keming Wu; Zhongyu Yang; Bo Li; Xiaojuan Qi; Shijian Lu; Xingxuan Li; Lidong Bing
>
> **备注:** Project Page: this https URL
>
> **摘要:** Training large multimodal models (LMMs) via reinforcement learning (RL) to natively invoke video-processing tools (e.g., cropping) has become a promising route to long-video understanding. However, existing native-RL methods dispatch tool calls sequentially (i.e., one per turn): a single wrong crop propagates errors without peer correction, multi-turn tool calls corrupt context, and inference cost scales linearly with the number of turns. We introduce ParaVT, the first multi-agent end-to-end RL-trained framework for Parallel Video Tool calling, dispatching multiple time-window crops in a single turn for cleaner context and better fault tolerance. Yet applying standard RL to ParaVT reveals an obstacle we term the Tool Prior Paradox: the pretrained tool priors that enable tool exploration also destabilize cold-started structural format and expose the skip-tool reward shortcut under temperature sampling. A cross-model contrast on a weaker-prior LMM supports this claim: format stays stable but RL elicits zero tool calls, indicating that prior strength is the shared driver of both format collapse and tool exploration. We propose PARA-GRPO (Parseability-Anchored and Ratio-gAted GRPO), which augments standard RL with two complementary mechanisms: (i) a targeted format reward applied only at the structural-token positions most prone to collapse, and (ii) a per-prompt frame-budget randomization that creates training prompts where calling the tool yields a measurable reward signal over skipping it. Across six long-video understanding benchmarks, ParaVT improves over the Qwen3-VL baseline by +7.9% on average, with PARA-GRPO lifting training-time format compliance from 0.13 to 0.64. As tool capabilities become increasingly internalized in modern LMMs, RL must cooperate with the resulting priors, and ParaVT offers a general recipe for agentic RL. Code, data, and model weights are publicly available.
>
---
#### [replaced 052] When Simultaneous Localization and Mapping Meets Wireless Communications: A Survey
- **分类: cs.RO; cs.CV; cs.IT; cs.MA**

- **简介: 该论文属于SLAM与无线通信融合任务，解决两者协同优化问题，分析了视觉SLAM与无线信号的相互影响及集成方法。**

- **链接: [https://arxiv.org/pdf/2602.06995](https://arxiv.org/pdf/2602.06995)**

> **作者:** Konstantinos Gounis; Sotiris A. Tegos; Dimitrios Tyrovolas; Panagiotis D. Diamantoulakis; George K. Karagiannidis
>
> **摘要:** This paper surveys the state-of-the-art in the nexus of SLAM and Wireless Communications, attributing the bidirectional impact of each with a focus on visual SLAM (V-SLAM) integration. We provide an overview of key concepts related to wireless signal propagation, geometric channel modeling, and radio frequency (RF)-based localization and sensing. In addition to this, we show image processing techniques that can detect landmarks, proactively predicting optimal paths for wireless channels. Several dimensions are considered, including the prerequisites, techniques, background, and future directions and challenges of the intersection between SLAM and wireless communications. We analyze estimation and control approaches such as Bayesian filters, feature-based pose estimation, perception-aware motion control, spatial methods for signal processing such as vector fields, and key technological aspects. We expose techniques and items towards enabling a highly effective retrieval of the autonomous robot state. Among other interesting findings, we observe that monocular V-SLAM would benefit from RF relevant information, as the latter can serve as a proxy for the scale ambiguity resolution. Conversely, we find that wireless communications in the context of 5G and beyond can potentially benefit from visual odometry that is central in SLAM. Moreover, we examine other sources besides the camera for SLAM and describe the twofold relation with wireless communications. Finally, integrated solutions performing joint communications and SLAM appear to be in their infancy: theoretical and practical advancements are required to add higher-level localization and semantic perception capabilities to RF and multi-antenna technologies.
>
---
#### [replaced 053] RobuQ: Pushing DiTs to W1.58A2 via Robust Activation Quantization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23582](https://arxiv.org/pdf/2509.23582)**

> **作者:** Kaicheng Yang; Xun Zhang; Haotong Qin; Yucheng Lin; Kaisen Yang; Xianglong Yan; Yulun Zhang
>
> **备注:** Accepted by ICML2026
>
> **摘要:** Diffusion Transformers (DiTs) have recently emerged as a powerful backbone for image generation, demonstrating superior scalability and performance over U-Net architectures. However, their practical deployment is hindered by substantial computational and memory costs. While Quantization-Aware Training (QAT) has shown promise for U-Nets, its application to DiTs faces unique challenges, primarily due to the sensitivity and distributional complexity of activations. In this work, we identify activation quantization as the primary bottleneck for pushing DiTs to extremely low-bit settings. To address this, we propose a systematic QAT framework for DiTs, named RobuQ. We start by establishing a strong ternary weight (W1.58A4) DiT baseline. Building upon this, we propose RobustQuantizer to achieve robust activation quantization. Our theoretical analyses show that the Hadamard transform can convert unknown per-token distributions into per-token normal distributions, providing a strong foundation for this method. Furthermore, we propose AMPN, the first Activation-only Mixed-Precision Network pipeline for DiTs. This method applies ternary weights across the entire network while allocating different activation precisions to each layer to eliminate information bottlenecks. Through extensive experiments on unconditional and conditional image generation, our RobuQ framework achieves state-of-the-art performance for DiT quantization in sub-4-bit quantization configuration. To the best of our knowledge, RobuQ is the first achieving stable and competitive image generation on large datasets like ImageNet-1K with activations quantized to average 2 bits. The code and models will be available at this https URL .
>
---
#### [replaced 054] Enhancing Event-based Object Detection with Monocular Normal Maps
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02127](https://arxiv.org/pdf/2508.02127)**

> **作者:** Mingjie Liu; Hanqing Liu; Luoping Cui; Chuang Zhu
>
> **摘要:** Object detection in autonomous driving is frequently compromised by complex illumination. While event cameras offer a robust solution, they are susceptible to sudden contrast changes such as reflections which often trigger dense, misleading event signals. To overcome this, we leverage RGB-derived surface normal maps as explicit geometric constraints. Crucially, even when RGB degrades, they preserve low-frequency structural priors that effectively assist in event-based detection. Consequently, we present NRE-Net, a trimodal framework that integrates structural priors from surface Normal maps, appearance context from RGB images, and high-frequency dynamics from Events. The Adaptive Dual-stream Fusion Module (ADFM) first aligns geometric and appearance cues, followed by the Event-modality Aware Fusion Module (EAFM) which selectively integrates event dynamics. Extensive evaluations on DSEC-Det-sub and PKU-DAVIS-SOD demonstrate that incorporating geometric priors yields an additional 3.0% AP50 gain over dual-modal baselines, while our approach consistently outperforms fusion methods such as SFNet (+2.7%) and SODFormer (+7.1%).
>
---
#### [replaced 055] IVGT: Implicit Visual Geometry Transformer for Neural Scene Representation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出IVGT，解决无姿态多视角图像中连续几何与外观重建问题，通过隐式建模实现场景的神经表示和空间查询。**

- **链接: [https://arxiv.org/pdf/2605.16258](https://arxiv.org/pdf/2605.16258)**

> **作者:** Yuqi Wu; Tianyu Hu; Wenzhao Zheng; Yuanhui Huang; Haowen Sun; Jie Zhou; Jiwen Lu
>
> **备注:** Code: this https URL
>
> **摘要:** Reconstructing coherent 3D geometry and appearance from unposed multi-view images is a fundamental yet challenging problem in computer vision. Most existing visual geometry foundation models predict explicit geometry by regressing pixel-aligned pointmaps, often suffering from redundancy and limited geometric continuity. We propose IVGT, an Implicit Visual Geometry Transformer that implicitly models continuous and coherent geometry from pose-free multi-view images. This formulation learns a continuous neural scene representation in a canonical coordinate system and supports continuous spatial queries at any 3D positions, retrieving local features to predict signed distance (SDF) values and colors using lightweight decoders. It allows direct extraction of continuous and coherent surface geometry, enabling rendering of RGB images, depth maps, and surface normal maps from arbitrary viewpoints. We train IVGT via multi-dataset joint optimization with 2D supervision and 3D geometric regularization. IVGT demonstrates generalization across scenes and achieves strong performance on various tasks, including mesh and point cloud reconstruction, novel view synthesis, depth and surface normal estimation, and camera pose estimation.
>
---
#### [replaced 056] Attend Locally, Remember Linearly: Linear Attention as Cross-Frame Memory for Autoregressive Video Diffusion
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.16579](https://arxiv.org/pdf/2605.16579)**

> **作者:** Kunyang Li; Mubarak Shah; Yuzhang Shang
>
> **摘要:** Autoregressive (AR) video diffusion is a powerful paradigm for streaming and interactive video generation. However, its reliance on softmax self-attention leads to quadratic compute complexity in sequence length and memory usage due to key-value caching, which limits its scalability to long video horizons. Existing remedies (e.g., sparse attention and KV-cache compression) reduce per-step cost but still rely on a linearly growing cache or irreversibly discard past context, and thus fail to address linear memory growth and streaming context management. To address this scalability bottleneck, we propose ARL2 (Attend Locally, Remember Linearly), a hybrid attention module that replaces quadratic cross-frame attention with a fixed-size recurrent state. We decompose self-attention into two branches: an intra-frame softmax branch for spatial detail and local dependencies, and an inter-frame gated recurrent linear branch that maintains a fixed-size state for streaming context. Our key insight is that softmax attention captures fine-grained local interactions, while a recurrent state provides controllable long-range memory. This design achieves linear-time scaling with constant memory while improving temporal consistency over the full-softmax model. To prevent noisy intermediate states from corrupting memory, we update the recurrent state only after the denoised pass. To avoid within-frame information asymmetry, all tokens share the same pre-update state rather than sequential updates. To the best of our knowledge, this is the first work to convert a pretrained AR video diffusion model into a hybrid linear attention architecture, through an efficient two-stage training scheme for AR video. With 75% of layers replaced by hybrid linear attention, the model achieves up to 2.26 wall-clock speedup and 54% memory reduction, while maintaining comparable quality with improving temporal consistency.
>
---
#### [replaced 057] U-CECE: A Universal Multi-Resolution Framework for Conceptual Counterfactual Explanations
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.08295](https://arxiv.org/pdf/2604.08295)**

> **作者:** Angeliki Dimitriou; Nikolaos Chaidos; Maria Lymperaiou; Giorgos Filandrianos; Giorgos Stamou
>
> **摘要:** As AI models grow more complex, explainability is essential for building trust, yet concept-based counterfactual methods still face a trade-off between expressivity and efficiency. Representing underlying concepts as atomic sets is fast but misses relational context, whereas full graph representations are more faithful but require solving the NP-hard Graph Edit Distance (GED) problem. We propose U-CECE, a unified, model-agnostic multi-resolution framework for conceptual counterfactual explanations that adapts to data regime and compute budget. U-CECE spans three levels of expressivity: atomic concepts for broad explanations, relational sets-of-sets for simple interactions, and structural graphs for full semantic structure. At the structural level, both a precision-oriented transductive mode based on supervised Graph Neural Networks (GNNs) and a scalable inductive mode based on unsupervised graph autoencoders (GAEs) are supported. Experiments on the structurally divergent CUB and Visual Genome datasets characterize the efficiency-expressivity trade-off across levels, while human surveys and LVLM-based evaluation show that the retrieved structural counterfactuals are semantically equivalent to, and often preferred over, exact GED-based ground-truth explanations.
>
---
#### [replaced 058] VisPhyWorld: Probing Physical Reasoning via Code-Driven Video Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.13294](https://arxiv.org/pdf/2602.13294)**

> **作者:** Jiarong Liang; Max Ku; Ka-Hei Hui; Ping Nie; Wenhu Chen
>
> **摘要:** Evaluating whether Multimodal Large Language Models (MLLMs) genuinely reason about physical dynamics remains challenging. Most existing benchmarks rely on recognition-style protocols such as Visual Question Answering (VQA) and Violation of Expectation (VoE), which can often be answered without committing to an explicit, testable physical hypothesis. We propose VisPhyWorld, an execution-based framework that evaluates physical reasoning by requiring models to generate executable simulator code from visual observations. By producing runnable code, the inferred world representation is directly inspectable, editable, and falsifiable. This separates physical reasoning from rendering. Building on this framework, we introduce VisPhyBench, comprising 209 evaluation scenes derived from 108 physical templates and a systematic protocol that evaluates how well models reconstruct appearance and reproduce physically plausible motion. Our pipeline produces valid reconstructed videos in 97.7% of benchmark runs before fallback. Experiments show that while state-of-the-art MLLMs achieve strong semantic scene understanding, they struggle to accurately infer physical parameters and to simulate consistent physical dynamics. Our code is available this https URL
>
---
#### [replaced 059] Weakly Supervised Cross-Modal Learning for 4D Radar Scene Flow Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18507](https://arxiv.org/pdf/2605.18507)**

> **作者:** Jingyun Fu; Zhiyu Xiang; Na Zhao
>
> **备注:** Accepted by ICML2026
>
> **摘要:** Due to the difficulty of obtaining ground-truth data for 4D radar scene flow estimation, previous methods typically rely on either self-supervised losses or cross-modal supervision using 3D LiDAR data, 2D images, and odometry. However, self-supervised approaches often yield suboptimal results due to radar's inherently low-fidelity measurements, while existing cross-modal supervised methods introduce complex multi-task architecture and require costly LiDAR sensors to generate pseudo radar scene flow labels from pretrained 3D tracking models. To overcome these limitations, we propose a task-specific iterative framework for weakly supervised radar scene flow learning, using only images and odometry for auxiliary supervision during training. Specially, we establish two novel instance-aware self-supervised losses by exploiting off-the-shelf 2D tracking and segmentation algorithms to obtain tracked instance masks, which are back-projected into 3D space to provide instance-level semantic guidance; for static regions, we integrate vehicle odometry with radar's intrinsic motion cues to construct a rigid static loss. Extensive experiments on the real-world View-of-Delft (VoD) dataset demonstrate that our method not only surpasses state-of-the-art cross-modal supervised approaches that rely on 3D multi-object tracking on dense LiDAR point clouds but also outperforms existing fully supervised scene flow estimation methods. The code is open-sourced at \href{this https URL}{this https URL}.
>
---
#### [replaced 060] Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.23224](https://arxiv.org/pdf/2601.23224)**

> **作者:** Xiangyu Zeng; Zhiqiu Zhang; Yuhan Zhu; Xinhao Li; Zikang Wang; Changlian Ma; Qingyu Zhang; Zizheng Huang; Kun Ouyang; Tianxiang Jiang; Ziang Yan; Yi Wang; Hongjie Zhang; Yali Wang; Limin Wang
>
> **备注:** 27 pages, 15 figures, 15 tables
>
> **摘要:** Existing multimodal large language models for long-video understanding predominantly rely on uniform sampling and single-turn inference, limiting their ability to identify sparse yet critical evidence amid extensive redundancy. We introduce Video-o3, a novel framework that supports iterative discovery of salient visual clues, fine-grained inspection of key segments, and adaptive termination once sufficient evidence is acquired. Technically, we address two core challenges in interleaved tool invocation. First, to mitigate attention dispersion induced by the heterogeneity of reasoning and tool-calling, we propose Task-Decoupled Attention Masking, which isolates per-step concentration while preserving shared global context. Second, to control context length growth in multi-turn interactions, we introduce a Verifiable Trajectory-Guided Reward that balances exploration coverage with reasoning efficiency. To support training at scale, we further develop a data synthesis pipeline and construct Seeker-173K, comprising 173K high-quality tool-interaction trajectories for effective supervised and reinforcement learning. Extensive experiments show that Video-o3 substantially outperforms state-of-the-art methods, achieving 72.1% accuracy on MLVU and 46.5% on Video-Holmes. These results demonstrate Video-o3's strong multi-hop evidence-seeking and reasoning capabilities, and validate the effectiveness of native tool invocation in long-video scenarios.
>
---
#### [replaced 061] Identifiable Token Correspondence for World Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2605.16457](https://arxiv.org/pdf/2605.16457)**

> **作者:** Youngin Kim; Ray Sun; Inho Kim; Bumsoo Park; Hyun Oh Song
>
> **摘要:** Token-based transformer world models have shown strong performance in visual reinforcement learning, but often suffer from temporal inconsistency in long-horizon rollouts, including object duplication, disappearance, and transmutation. A key reason is that most existing approaches treat next-frame prediction purely as a token generation problem, without considering the persistence of tokens across time. We introduce Identifiable Token Correspondence (ITC), a decoding step for token-based transformer world models that formulates next-frame prediction as a structured assignment problem with latent token correspondence variables: each next-frame token is explained either by copying a token from the previous frame or by generating a new one. ITC leaves the transformer architecture and training procedure unchanged and can be added on top of existing backbones. Our experiments show state-of-the-art performance on 4 challenging benchmarks. The proposed method achieves a return of 72.5% and a score of 35.6% on the Craftax-classic benchmark, significantly surpassing the previous best of 67.4% and 27.9%. We release our source code on this https URL.
>
---
#### [replaced 062] Divergence is Uncertainty: A Closed-Form Posterior Covariance for Flow Matching
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2605.00941](https://arxiv.org/pdf/2605.00941)**

> **作者:** Jiarui Xing; Song Wang; Jian Wang
>
> **备注:** 9 Pages, 5 figures
>
> **摘要:** Flow matching has become a leading framework for generative modeling, but quantifying the uncertainty of its samples remains an open problem. Existing approaches retrain the model with auxiliary variance heads, maintain costly ensembles, or propagate approximate covariance through many integration steps, trading off training cost, inference cost, or accuracy. We show that none of these trade-offs is necessary. By extending Tweedie's formula from the denoising setting to the flow matching interpolant, we derive an exact, closed-form expression for the posterior covariance at every point along the generative trajectory. The result depends on a single quantity, namely the divergence of the learned velocity field, which can be computed post-hoc on any pre-trained flow matching model, requiring no retraining and no architectural modification. For one-step generators such as MeanFlow, the same formula yields the end-to-end generation uncertainty in a single forward pass, eliminating the multi-step variance propagation required by all prior methods. Experiments on MNIST confirm that the resulting per-pixel uncertainty maps are semantically meaningful, concentrating on digit boundaries where inter-sample variation is highest, and that the scalar uncertainty score tracks actual prediction error, all at roughly $10^4 \times$ less total compute than ensembling or Monte Carlo dropout.
>
---
#### [replaced 063] Decoupling Endpoint and Semantic Transition Learning for Zero-Shot Composed Image Retrieval
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.08389](https://arxiv.org/pdf/2605.08389)**

> **作者:** Mingyu Liu; Sihan Huang; Yijia Fan; Yinlin Yan; Quan Zhang; Jian-Fang Hu; Jianhuang Lai
>
> **摘要:** Zero-shot composed image retrieval (ZS-CIR) retrieves a target image from a reference image and a text modification without human-annotated CIR triplets. Projection-based ZS-CIR methods are attractive because they do not rely on LLMs at inference and remain lightweight, but they often underperform LLM-based approaches on complex semantic modifications. This gap reflects a semantic transition bottleneck in projection-based ZS-CIR: endpoint-level matching can let the edit text act as a target-side attribute cue rather than grounding it as a source-conditioned semantic transition. We further show that adding semantic transition supervision to the same text adapter creates an endpoint--transition conflict between endpoint alignment and semantic transition alignment. To address this conflict, DeCIR decouples endpoint and transition learning. It constructs paired forward/reverse edit tuples from image-caption pairs, trains separate low-rank text adapter branches for endpoint alignment and semantic transition alignment, and merges them with Low-Rank Directional Merge (LRDM) into one deployable adapter. Extensive experiments on CIRR, CIRCO, FashionIQ, and GeneCIS demonstrate that DeCIR consistently improves projection-based ZS-CIR without increasing inference complexity.
>
---
#### [replaced 064] Transporting Task Vectors across Different Architectures without Training
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.12952](https://arxiv.org/pdf/2602.12952)**

> **作者:** Filippo Rinaldi; Aniello Panariello; Giacomo Salici; Angelo Porrello; Simone Calderara
>
> **备注:** Accepted at the International Conference on Machine Learning (ICML), 2026
>
> **摘要:** Adapting large pre-trained models to downstream tasks often produces task-specific parameter updates that are expensive to relearn for every model variant. While recent work has shown that such updates can be transferred between models with identical architectures, transferring them across models of different widths remains unexplored. In this work, we introduce Theseus, a training-free method for transporting task updates across heterogeneous-width models. Rather than matching parameters, we characterize a task update by the functional effect it induces on intermediate representations. We formalize task-vector transport as a functional matching problem on observed activations and show that, after aligning representation spaces via orthogonal Procrustes analysis, it admits a stable closed-form solution that preserves the geometry of the update. We evaluate Theseus on vision and language models across different widths, showing consistent improvements over baselines without additional training or backpropagation. Our results show that task updates can be meaningfully transferred across architectures when task identity is defined functionally rather than parametrically. Code is available at this https URL.
>
---
#### [replaced 065] RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.00392](https://arxiv.org/pdf/2605.00392)**

> **作者:** Ben Wan; Yan Feng; Zihan Tang; Weizhe Huang; Yuting Zeng; Jia Wang; Tongxuan Liu
>
> **备注:** 21 pages, accepted by ICML2026
>
> **摘要:** DeepSeek-OCR leverages visual-text compression to reduce long-text processing costs and accelerate inference, yet visual tokens remain prone to redundant textual and structural information. Moreover, current token pruning methods for conventional vision-language models (VLMs) fail to preserve textual fidelity due to improper compression mechanisms. By analyzing the decoding process of DeepSeek-OCR, we find that a distinct two-stage reading trajectory: the model initially prioritizes the majority of high-norm tokens, then subsequently redistributes its attention to the remaining ones. Motivated by this insight, we propose RTPrune, a two-stage token pruning method tailored for DeepSeek-OCR. In the first stage, we prioritize high-norm visual tokens that capture salient textual and structural information. In the second stage, the remaining tokens are paired and merged based on optimal transport theory to achieve efficient feature aggregation. We further introduce a dynamic pruning ratio that adapts to token similarity and textual density for OCR tasks, enabling a better efficiency-accuracy trade-off. Extensive experiments demonstrate state-of-the-art performance, as evidenced by 99.47% accuracy and 1.23$\times$ faster prefill on OmniDocBench, achieved with 84.25% token retention when applied to DeepSeek-OCR-Large.
>
---
#### [replaced 066] Universal Skeleton Understanding via Differentiable Rendering and MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.18003](https://arxiv.org/pdf/2603.18003)**

> **作者:** Ziyi Wang; Peiming Li; Xinshun Wang; Yang Tang; Kai-Kuang Ma; Mengyuan Liu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Multimodal large language models (MLLMs) exhibit strong visual-language reasoning, yet cannot process structured, non-visual data such as human skeletons. Existing methods either compress skeleton dynamics into lossy feature vectors for text alignment, or quantize motion into discrete tokens that generalize poorly across heterogeneous skeleton formats. We present SkeletonLLM, which achieves universal skeleton understanding by translating arbitrary skeleton sequences into the MLLM's native visual modality. At its core is DrAction, a differentiable, format-agnostic renderer that converts skeletal kinematics into compact image sequences. Because the pipeline is end-to-end differentiable, MLLM gradients can directly guide the rendering to produce task-informative visual tokens. To further enhance reasoning capabilities, we introduce a cooperative training strategy: Causal Reasoning Distillation transfers structured, step-by-step reasoning from a teacher model, while Discriminative Finetuning sharpens decision boundaries between confusable actions. SkeletonLLM demonstrates strong generalization \revise{in open-vocabulary action recognition, while its learned reasoning capabilities naturally extend to motion captioning and question answering across heterogeneous skeleton formats} -- suggesting a viable path for applying MLLMs to non-native modalities. Code: this https URL.
>
---
#### [replaced 067] Energy-based Tissue Manifolds for Longitudinal Multiparametric MRI Analysis
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.07180](https://arxiv.org/pdf/2604.07180)**

> **作者:** Kartikay Tehlan; Lukas Förner; Sina Wendrich; Nico Schmutzenhofer; Michael Frühwald; Matthias Wagner; Nassir Navab; Thomas Wendler
>
> **备注:** The code is available at this https URL
>
> **摘要:** We propose a geometric framework for longitudinal multi-parametric MRI analysis based on patient-specific energy modelling in sequence space. Rather than operating on images with spatial networks, each voxel is represented by its multi-sequence intensity vector ($T1$, $T1c$, $T2$, FLAIR, ADC), and a compact implicit neural representation is trained via denoising score matching to learn an energy function $E_{\theta}(\mathbf{u})$ over $\mathbb{R}^d$ from a single baseline scan. The learned energy landscape provides a differential-geometric description of tissue regimes without segmentation labels. Local minima define tissue basins, gradient magnitude reflects proximity to regime boundaries, and Laplacian curvature characterises local constraint structure. Importantly, this baseline energy manifold is treated as a fixed geometric reference: it encodes the set of contrast combinations observed at diagnosis and is not retrained at follow-up. Longitudinal assessment is therefore formulated as evaluation of subsequent scans relative to this baseline geometry. Rather than comparing anatomical segmentations, we analyse how the distribution of MRI sequence vectors evolves under the baseline energy function. In a paediatric case with later recurrence, follow-up scans show progressive deviation in energy and directional displacement in sequence space toward the baseline tumour-associated regime before clear radiological reappearance. In a case with stable disease, voxel distributions remain confined to established low-energy basins without systematic drift. The presented cases serve as proof-of-concept that patient-specific energy manifolds can function as geometric reference systems for longitudinal mpMRI analysis without explicit segmentation or supervised classification, providing a foundation for further investigation of manifold-based tissue-at-risk tracking in neuro-oncology.
>
---
#### [replaced 068] HumanSplatHMR: Closing the Loop Between Human Mesh Recovery and Gaussian Splatting Avatar
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.02784](https://arxiv.org/pdf/2605.02784)**

> **作者:** Yeheng Zong; Pou-Chun Kung; Yike Pan; Seth Isaacson; Yizhou Chen; Ram Vasudevan; Katherine A. Skinner
>
> **备注:** Project page: this https URL
>
> **摘要:** Accurately recovering human pose and appearance from video is an essential component of scene reconstruction, with applications to motion capture, motion prediction, virtual reality, and digital twinning. Despite significant interest in building realistic human avatars from video, this paper demonstrates that existing methods do not accurately recover the 3D geometry of humans. ViT-based approaches are not consistently reliable and can overfit to 2D views, while NeRF- and Gaussian Splatting-based avatars treat pose and appearance separately, limiting rendering generalization to new poses. To resolve these shortcomings, this paper proposes HumanSplatHMR, a joint optimization framework that refines 3D human poses while simultaneously learning a high-fidelity avatar for novel-view and novel-pose synthesis. Our key insight is to close the loop between geometric pose estimation and differentiable rendering. Unlike prior human avatar methods that rely on accurate human pose obtained through motion capture systems or offline refinement, which are impractical in in-the-wild scenarios, our approach uses only human mesh estimates from a state-of-the-art human pose estimator to better reflect real-world conditions. Therefore, instead of using the human pose only as a deformation prior, HumanSplatHMR backpropagates photometric, segmentation, and depth losses through a differentiable renderer to the pose parameters and global position. This coupling refines the global 3D pose over time, improving accuracy and alignment while producing better renderings from novel views. Experiments show consistent improvements over pose recovery baselines that omit image-level refinement and avatar baselines that decouple pose estimation from avatar reconstruction.
>
---
#### [replaced 069] Structural Anchor Pruning: Training-Free Multi-Vector Compression for Visual Document Retrieval
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于视觉文档检索任务，解决多向量索引存储过高的问题。提出SAP方法，在无需训练的情况下高效压缩视觉标记，保留高检索性能。**

- **链接: [https://arxiv.org/pdf/2601.20107](https://arxiv.org/pdf/2601.20107)**

> **作者:** Zhuchenyang Liu; Ziyu Hu; Yao Zhang; Yu Xiao
>
> **备注:** methodology revision and new title
>
> **摘要:** Recent Vision-Language Models (e.g., ColPali) enable fine-grained Visual Document Retrieval (VDR) but incur prohibitive multi-vector index storage overhead. Existing training-free pruning methods either rely on heuristic layer choices or degrade sharply under aggressive compression, leading prior work to argue that effective high-compression pruning requires query-dependent training. We challenge this view with Structural Anchor Pruning (SAP), a self-calibrating, training-free, and query-agnostic index-time pruning framework with three components: (i) Score Retention (SR), a white-box per-layer compression diagnostic; (ii) SR-guided window selection, a procedure that automatically locates the structural pruning region for any backbone with no per-model hyperparameters; and (iii) a visual in-degree centrality scorer that identifies anchor patches within the selected window. On the ViDoRe v1/v2 benchmarks across three architectures spanning 18, 28, and 36 backbone layers, SAP retains over 90\% of NDCG@5 while pruning more than 90\% of visual tokens, without any per-model parameter tuning. Our layer-resolved SR analysis reveals an Alignment-Aggregation Divergence: the document's visual structure is preserved as a stable ``Structural Plateau'' within the backbone, but the final layers reshape this representation into a sparse, query-aligned form that is no longer suitable for pruning. This is the mechanistic reason SAP succeeds where final-layer methods fail.
>
---
#### [replaced 070] SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for Topological Crack Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.14926](https://arxiv.org/pdf/2605.14926)**

> **作者:** Hanxu Zhang; Chen Jia; Hui Liu; Xu Cheng; Fan Shi; Shengyong Chen
>
> **备注:** Accept by ICML2026
>
> **摘要:** Achieving pixel-level accurate segmentation of structural cracks across diverse scenarios remains a formidable challenge. Existing methods face significant bottlenecks in balancing crack topology modeling with computational efficiency, often failing to reconcile high segmentation quality with low resource demands. To address these limitations, we propose the Ultra-Compact Structure-Calibrated Vision RWKV (SCRWKV), a network that achieves high-precision modeling via a novel Structure-Field Encoder (SFE) backbone while maintaining linear complexity. The SFE integrates the Adaptive Multi-scale Cascaded Modulator (AMCM) to enhance texture representation and utilizes the Structure-Calibrated Insight Unit (SCIU) as its core engine. Specifically, the SCIU employs the Geometry-guided Bidirectional Structure Transformation (GBST) to capture topological correlations and integrates the Dynamic Self-Calibrating Decay (DSCD) into Dy-WKV to suppress noise propagation. Furthermore, we introduce a lightweight Cross-Scale Harmonic Fusion (CSHF) decoder to achieve precise feature aggregation. Systematic evaluations on multiple benchmarks characterized by complex textures and severe interference demonstrate that SCRWKV, with only 1.22M parameters, significantly outperforms SOTA methods. Achieving an F1 score of 0.8428 and mIoU of 0.8512 on the TUT dataset, the model confirms its robust potential for efficient real-world deployment. The code is available at this https URL.
>
---
#### [replaced 071] RelWitness: Open-Vocabulary 3D Scene Graph Generation with Visual-Geometric Relation Witnesses
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.20823](https://arxiv.org/pdf/2605.20823)**

> **作者:** Minh Anh Nguyen; Quang Huy Tran; Bao Ngoc Le; Tuan Kiet Pham; Sui Yang Guang
>
> **摘要:** Open-vocabulary 3D scene graph generation seeks to describe object instances and their relations with flexible natural-language predicates. The central difficulty is not only vocabulary expansion, but supervision reliability: relation annotations in 3D scene graph datasets are selective, and many valid object-pair relations are unannotated. We propose RelWitness, a framework for open-vocabulary 3D scene graph generation from posed RGB-D sequences under incomplete relation supervision. The key concept is a relation witness: a concrete visual-geometric cue that makes a relation observable in the captured scene. Support relations require contact and vertical ordering; containment requires enclosure; proximity requires metric closeness; orientation requires facing direction; and stable relations should persist across views where both objects are visible. RelWitness constructs relation witness records from RGB views, depth maps, reconstructed 3D geometry, role-sensitive text, object-prior null views, and multi-view consistency. A visual-geometric witness verifier assigns unannotated relation candidates to verified missing positives, reliable negatives, or uncertain unlabeled cases. A witness-guided positive-unlabeled objective then learns from incomplete annotations without turning every missing label into a negative. We further introduce witness-consistent decoding and an RGB-D missing-relation audit protocol. Simulated manuscript-planning experiments on 3DSSG/3RScan and ScanNet-derived open-vocabulary splits show the intended behavior: improved unseen-relation recognition, higher witness precision, lower hallucination, and reduced redundant relation phrases. All numerical results are planning values and must be replaced by reproduced measurements before submission
>
---
#### [replaced 072] VDFP: Video Deflickering with Flicker-banding Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.21079](https://arxiv.org/pdf/2605.21079)**

> **作者:** Zhiyi Zhou; Libo Zhu; Zihan Zhou; Yulun Zhang; Xiaokang Yang
>
> **备注:** Our dataset and code will be released at this https URL
>
> **摘要:** Capturing digital screens with smartphones frequently induces severe banding due to hardware synchronization mismatches. Existing video restoration methods struggle with these structured, periodic luminance fluctuations, often resulting in residual artifacts or over-smoothed textures. We firstly construct DeViD, a real-world dataset in various scenes to deal with the lack of available datasets. Then we propose VDFP (Video Deflickering with Flicker-banding Priors), a novel perception-guided generation framework. First, we introduce a Degradation Field Modeling Based on Rolling Shutter Mechanism (DFM) capable of synthesizing complex multi-banding scenarios. Second, we present a spatial-temporal continuous prior perception (CPP). Unlike traditional binary segmentation, this module is optimized via a Flicker-Aware Mean Squared Error (FA-MSE) to capture the luminance transitions. By zero-initializing an augmented input layer, our model preserves pre-trained generative priors as well as spatial-temporal prior perception. Extensive experiments demonstrate that VDFP significantly outperforms other methods, eliminating complex banding with high-fidelity spatial details and temporal consistency. Our dataset and code will be released at this https URL.
>
---
#### [replaced 073] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2605.20302](https://arxiv.org/pdf/2605.20302)**

> **作者:** Panagiotis Koromilas; Theodoros Giannakopoulos; Mihalis A. Nicolaou; Yannis Panagakis
>
> **备注:** 43rd International Conference on Machine Learning (ICML 2026); Code: this https URL
>
> **摘要:** Supervised classification has a theoretical optimum, Neural Collapse (NC), yet neither of its two dominant paradigms reaches it in practice. Cross entropy (CE) leaves radial degrees of freedom unconstrained and converges to a degenerate geometry, while supervised contrastive learning (SCL) drives features toward NC during pretraining but discards this structure in a post hoc linear probing phase. We show that both paradigms are different appearances of the same method that contrasts prototypes on the unit hypersphere, and that closing the gap requires fixing each at its point of failure. From the CE side, we propose NTCE and NONL, two normalized losses that import contrastive optimization's missing ingredients into classifier learning: a large effective negative set and decoupled alignment and uniformity terms. From the SCL side, we prove that SCL's objective already optimizes throughout training for a principled classifier whose weights are the class mean embeddings, making linear probing both redundant and harmful. Empirically, on four benchmarks including ImageNet-1K, NTCE and NONL surpass CE accuracy, closely approximate NC ($\geq 95\%$), and match CE's converged NC on 4/5 metrics in under $7.5\%$ of its iterations, while SCL with fixed prototypes matches linear probing without the hours-long classifier training phase. The learned geometry yields $+5.5\%$ mean relative improvement in transfer learning, up to $+8.7\%$ under severe class imbalance, and improved robustness to corruptions on ImageNet-C. Our work recasts supervised learning as prototype learning on the hypersphere, with NC reached by design.
>
---
#### [replaced 074] Revisiting Integration of Image and Metadata for DICOM Series Classification: Cross-Attention and Dictionary Learning
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23833](https://arxiv.org/pdf/2602.23833)**

> **作者:** Tuan Truong; Melanie Dohmen; Sara Lorio; Matthias Lenga
>
> **备注:** Early acceptance at MICCAI 2026
>
> **摘要:** Automated identification of DICOM image series is essential for large-scale medical image analysis, quality control, protocol harmonization, and reliable downstream processing. However, DICOM series classification remains challenging due to heterogeneous slice content, variable series length, and entirely missing, incomplete or inconsistent DICOM metadata. We propose an end-to-end multimodal framework for DICOM series classification that jointly models image content and acquisition metadata while explicitly accounting for all these challenges. (i) Images and metadata are encoded with modality-aware modules and fused using a bi-directional cross-modal attention mechanism. (ii) Metadata is processed by a sparse, missingness-aware encoder based on learnable feature dictionaries and value-conditioned modulation. By design, the approach does not require any form of imputation. (iii) Variability in series length and image data dimensions is handled via a 2.5D visual encoder and attention operating on equidistantly sampled slices. We evaluate the proposed approach on the publicly available Duke Liver MRI dataset and a large multi-institutional in-house cohort, assessing both in-domain performance and out-of-domain generalization. Across all evaluation settings, the proposed method consistently outperforms relevant image only, metadata-only and multimodal 2D/3D baselines. The results demonstrate that explicitly modeling metadata sparsity and cross-modal interactions improves robustness for DICOM series classification.
>
---
#### [replaced 075] RE-VLM: Event-Augmented Vision-Language Model for Scene Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.19329](https://arxiv.org/pdf/2605.19329)**

> **作者:** Hanqing Liu; Mingjie Liu; Luoping Cui; Endian Lin; Donghong Jiang; Chuang Zhu
>
> **备注:** 10 pages, 6 figures, 6 tables
>
> **摘要:** Conventional vision-language models (VLMs) struggle to interpret scenes captured under adverse conditions (e.g., low light, high dynamic range, or fast motion) because standard RGB images degrade in such environments. Event cameras provide a complementary modality: they asynchronously record per-pixel brightness changes with high temporal resolution and wide dynamic range, preserving motion cues where frames fail. We propose RE-VLM, the first dual-stream vision-language model that jointly leverages RGB images and event streams for robust scene understanding across both normal and challenging conditions. RE-VLM employs parallel RGB and event encoders together with a progressive training strategy that aligns heterogeneous visual features with language. To address the scarcity of RGB-Event-Text supervision, we further propose a graph-driven pipeline that converts synchronized RGB-Event streams into verifiable scene graphs, from which we synthesize captions and question-answer (QA) pairs. To develop and evaluate RE-VLM, we construct two datasets: PEOD-Chat, targeting illumination-challenged scenes, and RGBE-Chat, covering diverse scenarios. On captioning and VQA benchmarks, RE-VLM consistently outperforms state-of-the-art RGB-only and event-only models with comparable parameter counts, with particularly large gains under challenging conditions. These results demonstrate the effectiveness of event-augmented VLMs in achieving robust vision-language understanding across a wide range of real-world environments.
>
---
#### [replaced 076] SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
- **分类: cs.RO; cs.AI; cs.CV; cs.GR; eess.SY**

- **简介: 该论文提出SONIC模型，解决人形机器人全身控制问题。通过扩大模型规模、数据和计算，实现自然运动跟踪，提升控制性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.07820](https://arxiv.org/pdf/2511.07820)**

> **作者:** Zhengyi Luo; Ye Yuan; Tingwu Wang; Chenran Li; Fernando Castañeda; Sirui Chen; Zi-Ang Cao; Jiefeng Li; David Minor; Qingwei Ben; Jinhyung Park; David Sami; Zi Wang; Xingye Da; Runyu Ding; Cyrus Hogg; Lina Song; Edy Lim; Eugene Jeong; Tairan He; Haoru Xue; Wenli Xiao; Simon Yuen; Jan Kautz; Yan Chang; Umar Iqbal; Linxi "Jim" Fan; Yuke Zhu
>
> **备注:** Project page: this https URL
>
> **摘要:** Despite the rise of billion-parameter foundation models trained across thousands of GPUs, similar scaling gains have not been shown for humanoid control. Current neural controllers for humanoids remain modest in size, target a limited set of behaviors, and are trained on a handful of GPUs. We show that scaling model capacity, data, and compute yields a generalist humanoid controller capable of natural, robust whole-body movements. We position motion tracking as a scalable task for humanoid control, leveraging dense supervision from diverse motion-capture data to acquire human motion priors without manual reward engineering. We build a foundation model for motion tracking by scaling along three axes: network size (1.2M to 42M parameters), dataset volume (100M+ frames from 700 hours of motion capture), and compute (21k GPU hours). Beyond demonstrating the benefits of scale, we further show downstream utility through: (1) a real-time kinematic planner bridging motion tracking to tasks such as navigation, enabling natural and interactive control, and (2) a unified token space supporting VR teleoperation and vision-language-action (VLA) models with a single policy. Through this interface, we demonstrate autonomous VLA-driven whole-body loco-manipulation requiring coordinated hand and foot placement. Scaling motion tracking exhibits favorable properties: performance improves steadily with compute and data diversity, and learned policies generalize to unseen motions, establishing motion tracking at scale as a practical foundation for humanoid control.
>
---
#### [replaced 077] Ray-Aware Pointer Memory with Adaptive Updates for Streaming 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.05749](https://arxiv.org/pdf/2605.05749)**

> **作者:** Feifei Li; Qi Song; Chi Zhang; Rui Huang
>
> **摘要:** Dense 3D reconstruction from continuous image streams requires both accurate geometric aggregation and stable long-term memory management. Recent feed-forward reconstruction frameworks integrate observations through persistent memory representations, yet most rely primarily on appearance-based similarity when updating memory. Such appearance-driven integration often leads to redundant accumulation of observations and unstable geometry when viewpoint changes occur. In this work, we propose a ray-aware pointer memory for streaming 3D reconstruction that explicitly models both spatial location and viewing direction within a unified memory representation. Each memory pointer stores its 3D position, associated ray direction, and feature embedding, allowing the system to reason jointly about geometric proximity and viewpoint consistency. Based on this representation, we introduce an adaptive pointer update strategy that replaces traditional fusion-based memory compression with a retain-or-replace mechanism. Instead of averaging nearby observations, the system selectively retains informative pointers while discarding redundant ones, preserving distinctive geometric structures while maintaining bounded memory growth. Furthermore, the joint reasoning over spatial distance and ray-direction discrepancy enables the system to distinguish between local redundancy, novel observations, and potential loop revisits in a unified manner. When loop candidates are detected, pose refinement is triggered to enforce global geometric consistency across the reconstruction. Extensive experiments demonstrate that the proposed ray-aware memory design significantly improves long-term reconstruction stability and camera pose accuracy while maintaining efficient streaming inference. Our approach provides a principled framework for scalable and drift-resistant online 3D reconstruction from image streams.
>
---
#### [replaced 078] AEGIS: A Holistic Benchmark for Evaluating Forensic Analysis of AI-Generated Academic Images
- **分类: cs.CV; cs.CY**

- **链接: [https://arxiv.org/pdf/2604.28177](https://arxiv.org/pdf/2604.28177)**

> **作者:** Bo Zhang; Tzu-Yen Ma; Zichen Tang; Junpeng Ding; Zirui Wang; Yizhuo Zhao; Peilin Gao; Zijie Xi; Zixin Ding; Haiyang Sun; Haocheng Gao; Yuan Liu; Liangjia Wang; Yiling Huang; Yujie Wang; Yuyue Zhang; Ronghui Xi; Yuanze Li; Jiacheng Liu; Zhongjun Yang; Haihong E
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** We introduce AEGIS, A holistic benchmark for Evaluating forensic analysis of AI-Generated academic ImageS. Compared to existing benchmarks, AEGIS features three key advances: (1) Domain-Specific Complexity: covering seven academic categories with 39 fine-grained subtypes, exposing intrinsic forensic difficulty, where even GPT-5.1 reaches 48.80% overall performance and expert models achieve only limited localization accuracy (IoU 30.09%); (2) Diverse Forgery Simulations: modeling four prevalent academic forgery strategies across 25 generative models, with 11 yielding average forensic accuracy below 50%, showing that forensics lag behind generative advances; and (3) Multi-Dimensional Forensic Evaluation: jointly assessing detection, reasoning, and localization, revealing complementary strengths between model families, with multimodal large language models (MLLMs) at 84.74% accuracy in textual artifact recognition and expert detectors peaking at 79.54% accuracy in binary authenticity detection. By evaluating 25 leading MLLMs, nine expert models, and one unified multimodal understanding and generation model, AEGIS serves as a diagnostic testbed exposing fundamental limitations in academic image forensics.
>
---
#### [replaced 079] X-OmniClaw Technical Report: A Unified Mobile Agent for Multimodal Understanding and Interaction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.05765](https://arxiv.org/pdf/2605.05765)**

> **作者:** Xiaoming Ren; Ru Zhen; Chao Li; Yang Song; Qiuxia Hou; Yanhao Zhang; Peng Liu; Qi Qi; Quanlong Zheng; Qi Wu; Zhenyi Liao; Binqiang Pan; Haobo Ji; Haonan Lu
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Inspired by the development of OpenClaw, there is a growing demand for mobile-based personal agents capable of handling complex and intuitive interactions. In this technical report, we introduce X-OmniClaw, a unified mobile agent designed for multimodal understanding and interaction in the Android ecosystem. This unified architecture of perception, memory, and action enables the agent to handle complex mobile tasks with high contextual awareness. Specifically, Omni Perception provides a unified multimodal ingress pipeline that integrates UI states, real-world visual contexts, and speech inputs, leveraging a temporal alignment module to decompose raw data into structured multimodal intent representations. Omni Memory leverages multimodal memory optimization to enhance personalized intelligence by integrating runtime working memory for task continuity with long-term personal memory distilled from local data, enabling highly context-aware and personalized interactions. Finally, Omni Action employs a hybrid grounding strategy that combines structural XML metadata with visual perception for robust interaction. Through Behavior Cloning and Trajectory Replay, the system captures user navigation as reusable skills, enabling precise direct-access execution. Demonstrations across diverse scenarios show that X-OmniClaw effectively enhances interaction efficiency and task reliability, providing a practical architectural blueprint for the next generation of mobile-native personal assistants.
>
---
#### [replaced 080] OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.24762](https://arxiv.org/pdf/2604.24762)**

> **作者:** Boyang Wang; Guangyi Xu; Jiahui Zhang; Zhipeng Tang; Zezhou Cheng
>
> **摘要:** Shot Boundary Detection (SBD) aims to automatically identify shot changes and divide a video into coherent shots. While SBD was widely studied in the literature, existing methods often produce non-interpretable boundaries on transitions, miss subtle yet harmful discontinuities, and rely on noisy, low-diversity annotations and outdated benchmarks. To alleviate these limitations, we propose OmniShotCut to formulate SBD as structured relational prediction, jointly estimating shot ranges with intra-shot relations and inter-shot relations, by a shot query-based dense video Transformer. To avoid imprecise manual labeling, we adopt a fully synthetic transition synthesis pipeline that automatically reproduces major transition families with precise boundaries and parameterized variants. We also introduce OmniShotCutBench, a modern wide-domain benchmark enabling holistic and diagnostic evaluation. Experiments on the benchmarks demonstrate the effectiveness and generality of our method.
>
---
#### [replaced 081] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.16416](https://arxiv.org/pdf/2505.16416)**

> **作者:** Chengcheng Wang; Jianyuan Guo; Hongguang Li; Yuchuan Tian; Ying Nie; Chang Xu; Kai Han
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Rotary Position Embedding (RoPE) is widely adopted in large language models, but when applied to vision-language models (VLMs) it couples text and image position indices and can introduce spurious cross-modal relative-position bias. We propose Per-Token Distance (PTD) to quantify cross-modal positional disentanglement, and prove that PTD = 0 is a sufficient condition to eliminate the geometric attention bias induced by RoPE. Guided by this criterion, we introduce Circle-RoPE, which remaps 2D image-token coordinates onto an annulus orthogonal to the text position axis, yielding a cone-like geometry where each text token is equidistant to all image tokens while preserving intra-image spatial structure. We further propose Alternating Geometry Encoding (AGE) to combine complementary geometric priors by alternating the decoupled geometry of Circle-RoPE and the grid-based prior of standard RoPE across layers. This design enables cross-modal positional disentanglement while preserving fine-grained intra-image spatial structure. Experiments on diverse VLM backbones and multimodal benchmarks show consistent gains in spatial grounding and visual reasoning. The code is available at this https URL.
>
---
#### [replaced 082] Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.06676](https://arxiv.org/pdf/2602.06676)**

> **作者:** Bo Du; Xiaochen Ma; Xuekang Zhu; Zhe Yang; Chaogun Niu; Chenfan Qu; Mingqi Fang; Zhenming Wang; Jingjing Liu; Jian Liu; Ji-Zhe Zhou
>
> **摘要:** Fake Image Detection (FID), aiming at unified detection across four image forensic subdomains, is critical in real-world forensic scenarios. Compared with ensemble approaches, monolithic FID models are theoretically more promising, but to date, consistently yield inferior performance in practice. In this work, we identify the intrinsic distinctness of artifacts across subdomains, a critical barrier we term the ``Ji-Zhe phenomenon". Driven by this phenomenon, we diagnose the cause of this underperformance for the first time: the collapse of the artifact feature space. The core challenge for developing a practical monolithic FID model thus boils down to the ``unified-yet-discriminative" reconstruction of the artifact feature space. To address this paradoxical challenge, we hypothesize that high-level semantics can serve as a structural prior for the reconstruction, and further propose Semantic-Induced Constrained Adaptation (SICA), the first monolithic FID paradigm. Extensive experiments on our OpenMMSec dataset demonstrate that SICA outperforms 15 state-of-the-art methods and reconstructs the target unified-yet-discriminative artifact feature space in a near-orthogonal manner, thus firmly validating our hypothesis. The code and dataset are available at: this https URL.
>
---
#### [replaced 083] Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17186](https://arxiv.org/pdf/2602.17186)**

> **作者:** Seulbi Lee; Sangheum Hwang
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Large Vision Language Models (LVLMs) have achieved remarkable progress, yet they often suffer from language bias, producing answers without relying on visual evidence. While prior work attempts to mitigate this issue through decoding strategies, architectural modifications, or curated instruction data, they typically lack a quantitative measure of how much individual training samples or tokens actually benefit from the image. In this work, we introduce Visual Information Gain (VIG), a perplexity-based metric that measures the reduction in prediction uncertainty provided by visual input. VIG enables fine-grained analysis at both sample and token levels, effectively highlighting visually grounded elements such as colors, spatial relations, and attributes. Leveraging this, we propose a VIG-guided selective training scheme that prioritizes high-VIG samples and tokens. This approach improves visual grounding and mitigates language bias, achieving superior performance with significantly reduced supervision by focusing exclusively on visually informative samples and tokens.
>
---
