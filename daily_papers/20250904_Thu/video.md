# 计算机视觉 cs.CV

- **最新发布 87 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] LiGuard: A Streamlined Open-Source Framework for Rapid & Interactive Lidar Research
- **分类: cs.CV**

- **简介: 该论文提出LiGuard框架，解决激光雷达研究中代码重复、灵活性不足的问题。通过内置数据处理、交互式算法调整及结构化目录，提升开发效率与成果共享性。**

- **链接: [http://arxiv.org/pdf/2509.02902v1](http://arxiv.org/pdf/2509.02902v1)**

> **作者:** Muhammad Shahbaz; Shaurya Agarwal
>
> **摘要:** There is a growing interest in the development of lidar-based autonomous mobility and Intelligent Transportation Systems (ITS). To operate and research on lidar data, researchers often develop code specific to application niche. This approach leads to duplication of efforts across studies that, in many cases, share multiple methodological steps such as data input/output (I/O), pre/post processing, and common algorithms in multi-stage solutions. Moreover, slight changes in data, algorithms, and/or research focus may force major revisions in the code. To address these challenges, we present LiGuard, an open-source software framework that allows researchers to: 1) rapidly develop code for their lidar-based projects by providing built-in support for data I/O, pre/post processing, and commonly used algorithms, 2) interactively add/remove/reorder custom algorithms and adjust their parameters, and 3) visualize results for classification, detection, segmentation, and tracking tasks. Moreover, because it creates all the code files in structured directories, it allows easy sharing of entire projects or even the individual components to be reused by other researchers. The effectiveness of LiGuard is demonstrated via case studies.
>
---
#### [new 002] DeepSea MOT: A benchmark dataset for multi-object tracking on deep-sea video
- **分类: cs.CV**

- **简介: 该论文构建首个公开深海视频多目标跟踪基准数据集，评估检测与跟踪模型性能，提供生成流程及计算工具。**

- **链接: [http://arxiv.org/pdf/2509.03499v1](http://arxiv.org/pdf/2509.03499v1)**

> **作者:** Kevin Barnard; Elaine Liu; Kristine Walz; Brian Schlining; Nancy Jacobsen Stout; Lonny Lundsten
>
> **备注:** 5 pages, 3 figures, dataset available at https://huggingface.co/datasets/MBARI-org/DeepSea-MOT
>
> **摘要:** Benchmarking multi-object tracking and object detection model performance is an essential step in machine learning model development, as it allows researchers to evaluate model detection and tracker performance on human-generated 'test' data, facilitating consistent comparisons between models and trackers and aiding performance optimization. In this study, a novel benchmark video dataset was developed and used to assess the performance of several Monterey Bay Aquarium Research Institute object detection models and a FathomNet single-class object detection model together with several trackers. The dataset consists of four video sequences representing midwater and benthic deep-sea habitats. Performance was evaluated using Higher Order Tracking Accuracy, a metric that balances detection, localization, and association accuracy. To the best of our knowledge, this is the first publicly available benchmark for multi-object tracking in deep-sea video footage. We provide the benchmark data, a clearly documented workflow for generating additional benchmark videos, as well as example Python notebooks for computing metrics.
>
---
#### [new 003] Human Preference-Aligned Concept Customization Benchmark via Decomposed Evaluation
- **分类: cs.CV**

- **简介: 该论文提出D-GPTScore方法，通过分解评估标准和MLLM细粒度评估，解决概念定制中人类偏好与现有指标不匹配的问题，并发布CC-AlignBench基准数据集，支持单/多概念任务评估。**

- **链接: [http://arxiv.org/pdf/2509.03385v1](http://arxiv.org/pdf/2509.03385v1)**

> **作者:** Reina Ishikawa; Ryo Fujii; Hideo Saito; Ryo Hachiuma
>
> **备注:** Accepted to ICCV Workshop 2025
>
> **摘要:** Evaluating concept customization is challenging, as it requires a comprehensive assessment of fidelity to generative prompts and concept images. Moreover, evaluating multiple concepts is considerably more difficult than evaluating a single concept, as it demands detailed assessment not only for each individual concept but also for the interactions among concepts. While humans can intuitively assess generated images, existing metrics often provide either overly narrow or overly generalized evaluations, resulting in misalignment with human preference. To address this, we propose Decomposed GPT Score (D-GPTScore), a novel human-aligned evaluation method that decomposes evaluation criteria into finer aspects and incorporates aspect-wise assessments using Multimodal Large Language Model (MLLM). Additionally, we release Human Preference-Aligned Concept Customization Benchmark (CC-AlignBench), a benchmark dataset containing both single- and multi-concept tasks, enabling stage-wise evaluation across a wide difficulty range -- from individual actions to multi-person interactions. Our method significantly outperforms existing approaches on this benchmark, exhibiting higher correlation with human preferences. This work establishes a new standard for evaluating concept customization and highlights key challenges for future research. The benchmark and associated materials are available at https://github.com/ReinaIshikawa/D-GPTScore.
>
---
#### [new 004] EdgeAttNet: Towards Barb-Aware Filament Segmentation
- **分类: cs.CV; astro-ph.SR; eess.IV**

- **简介: 论文提出EdgeAttNet，用于H-alpha图像中太阳丝状体的分割，解决现有方法在捕捉细小barbs结构上的不足，通过引入可学习边缘图和改进注意力机制，提升分割精度和效率。**

- **链接: [http://arxiv.org/pdf/2509.02964v1](http://arxiv.org/pdf/2509.02964v1)**

> **作者:** Victor Solomon; Piet Martens; Jingyu Liu; Rafal Angryk
>
> **摘要:** Accurate segmentation of solar filaments in H-alpha observations is critical for determining filament chirality, a key factor in the behavior of Coronal Mass Ejections (CMEs). However, existing methods often fail to capture fine-scale filament structures, particularly barbs, due to a limited ability to model long-range dependencies and spatial detail. We propose EdgeAttNet, a segmentation architecture built on a U-Net backbone by introducing a novel, learnable edge map derived directly from the input image. This edge map is incorporated into the model by linearly transforming the attention Key and Query matrices with the edge information, thereby guiding the self-attention mechanism at the network's bottleneck to more effectively capture filament boundaries and barbs. By explicitly integrating this structural prior into the attention computations, EdgeAttNet enhances spatial sensitivity and segmentation accuracy while reducing the number of trainable parameters. Trained end-to-end, EdgeAttNet outperforms U-Net and other U-Net-based transformer baselines on the MAGFILO dataset. It achieves higher segmentation accuracy and significantly better recognition of filament barbs, with faster inference performance suitable for practical deployment.
>
---
#### [new 005] LGBP-OrgaNet: Learnable Gaussian Band Pass Fusion of CNN and Transformer Features for Robust Organoid Segmentation and Tracking
- **分类: cs.CV; cs.AI**

- **简介: 本文提出LGBP-OrgaNet，结合CNN与Transformer特征，通过可学习高斯带通融合模块和双向交叉融合块，实现器官类器官的自动化分割与跟踪，解决传统方法破坏结构的问题，提升分割准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.03221v1](http://arxiv.org/pdf/2509.03221v1)**

> **作者:** Jing Zhang; Siying Tao; Jiao Li; Tianhe Wang; Junchen Wu; Ruqian Hao; Xiaohui Du; Ruirong Tan; Rui Li
>
> **摘要:** Organoids replicate organ structure and function, playing a crucial role in fields such as tumor treatment and drug screening. Their shape and size can indicate their developmental status, but traditional fluorescence labeling methods risk compromising their structure. Therefore, this paper proposes an automated, non-destructive approach to organoid segmentation and tracking. We introduced the LGBP-OrgaNet, a deep learning-based system proficient in accurately segmenting, tracking, and quantifying organoids. The model leverages complementary information extracted from CNN and Transformer modules and introduces the innovative feature fusion module, Learnable Gaussian Band Pass Fusion, to merge data from two branches. Additionally, in the decoder, the model proposes a Bidirectional Cross Fusion Block to fuse multi-scale features, and finally completes the decoding through progressive concatenation and upsampling. SROrga demonstrates satisfactory segmentation accuracy and robustness on organoids segmentation datasets, providing a potent tool for organoid research.
>
---
#### [new 006] Transformer-Guided Content-Adaptive Graph Learning for Hyperspectral Unmixing
- **分类: cs.CV**

- **简介: 该论文提出T-CAGU框架，结合Transformer和内容自适应图神经网络，解决高光谱解混中全局依赖与局部一致性的矛盾，提升长程交互和边界细节保留。**

- **链接: [http://arxiv.org/pdf/2509.03376v1](http://arxiv.org/pdf/2509.03376v1)**

> **作者:** Hui Chen; Liangyu Liu; Xianchao Xiu; Wanquan Liu
>
> **摘要:** Hyperspectral unmixing (HU) targets to decompose each mixed pixel in remote sensing images into a set of endmembers and their corresponding abundances. Despite significant progress in this field using deep learning, most methods fail to simultaneously characterize global dependencies and local consistency, making it difficult to preserve both long-range interactions and boundary details. This letter proposes a novel transformer-guided content-adaptive graph unmixing framework (T-CAGU), which overcomes these challenges by employing a transformer to capture global dependencies and introducing a content-adaptive graph neural network to enhance local relationships. Unlike previous work, T-CAGU integrates multiple propagation orders to dynamically learn the graph structure, ensuring robustness against noise. Furthermore, T-CAGU leverages a graph residual mechanism to preserve global information and stabilize training. Experimental results demonstrate its superiority over the state-of-the-art methods. Our code is available at https://github.com/xianchaoxiu/T-CAGU.
>
---
#### [new 007] AutoDetect: Designing an Autoencoder-based Detection Method for Poisoning Attacks on Object Detection Applications in the Military Domain
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AutoDetect，一种基于自编码器的中毒攻击检测方法，针对军事目标检测系统中的数据中毒问题，通过图像切片重建误差区分干净与中毒样本，提升检测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.03179v1](http://arxiv.org/pdf/2509.03179v1)**

> **作者:** Alma M. Liezenga; Stefan Wijnja; Puck de Haan; Niels W. T. Brink; Jip J. van Stijn; Yori Kamphuis; Klamer Schutte
>
> **备注:** To be presented at SPIE: Sensors + Imaging, Artificial Intelligence for Security and Defence Applications II
>
> **摘要:** Poisoning attacks pose an increasing threat to the security and robustness of Artificial Intelligence systems in the military domain. The widespread use of open-source datasets and pretrained models exacerbates this risk. Despite the severity of this threat, there is limited research on the application and detection of poisoning attacks on object detection systems. This is especially problematic in the military domain, where attacks can have grave consequences. In this work, we both investigate the effect of poisoning attacks on military object detectors in practice, and the best approach to detect these attacks. To support this research, we create a small, custom dataset featuring military vehicles: MilCivVeh. We explore the vulnerability of military object detectors for poisoning attacks by implementing a modified version of the BadDet attack: a patch-based poisoning attack. We then assess its impact, finding that while a positive attack success rate is achievable, it requires a substantial portion of the data to be poisoned -- raising questions about its practical applicability. To address the detection challenge, we test both specialized poisoning detection methods and anomaly detection methods from the visual industrial inspection domain. Since our research shows that both classes of methods are lacking, we introduce our own patch detection method: AutoDetect, a simple, fast, and lightweight autoencoder-based method. Our method shows promising results in separating clean from poisoned samples using the reconstruction error of image slices, outperforming existing methods, while being less time- and memory-intensive. We urge that the availability of large, representative datasets in the military domain is a prerequisite to further evaluate risks of poisoning attacks and opportunities patch detection.
>
---
#### [new 008] Time-Scaling State-Space Models for Dense Video Captioning
- **分类: cs.CV**

- **简介: 该论文针对密集视频字幕生成任务，解决长视频处理中的计算复杂度、内存限制及离线依赖问题。提出时间扩展状态空间模型，结合长序列与递归特性，支持在线生成，FLOPs减少7倍。**

- **链接: [http://arxiv.org/pdf/2509.03426v1](http://arxiv.org/pdf/2509.03426v1)**

> **作者:** AJ Piergiovanni; Ganesh Satish Mallya; Dahun Kim; Anelia Angelova
>
> **备注:** BMVC 2025
>
> **摘要:** Dense video captioning is a challenging video understanding task which aims to simultaneously segment the video into a sequence of meaningful consecutive events and to generate detailed captions to accurately describe each event. Existing methods often encounter difficulties when working with the long videos associated with dense video captioning, due to the computational complexity and memory limitations. Furthermore, traditional approaches require the entire video as input, in order to produce an answer, which precludes online processing of the video. We address these challenges by time-scaling State-Space Models (SSMs) to even longer sequences than before. Our approach, State-Space Models with Transfer State, combines both the long-sequence and recurrent properties of SSMs and addresses the main limitation of SSMs which are otherwise not able to sustain their state for very long contexts, effectively scaling SSMs further in time. The proposed model is particularly suitable for generating captions on-the-fly, in an online or streaming manner, without having to wait for the full video to be processed, which is more beneficial in practice. When applied to dense video captioning, our approach scales well with video lengths and uses 7x fewer FLOPs.
>
---
#### [new 009] Multi-Scale Deep Learning for Colon Histopathology: A Hybrid Graph-Transformer Approach
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出混合图-Transformer架构，结合胶囊网络与CNN，利用多尺度特征及自监督学习，提升结肠癌组织病理图像分类准确率，助力早期检测。**

- **链接: [http://arxiv.org/pdf/2509.02851v1](http://arxiv.org/pdf/2509.02851v1)**

> **作者:** Sadra Saremi; Amirhossein Ahmadkhan Kordbacheh
>
> **摘要:** Colon cancer also known as Colorectal cancer, is one of the most malignant types of cancer worldwide. Early-stage detection of colon cancer is highly crucial to prevent its deterioration. This research presents a hybrid multi-scale deep learning architecture that synergizes capsule networks, graph attention mechanisms, transformer modules, and residual learning to advance colon cancer classification on the Lung and Colon Cancer Histopathological Image Dataset (LC25000) dataset. The proposed model in this paper utilizes the HG-TNet model that introduces a hybrid architecture that joins strength points in transformers and convolutional neural networks to capture multi-scale features in histopathological images. Mainly, a transformer branch extracts global contextual bonds by partitioning the image into patches by convolution-based patch embedding and then processing these patches through a transformer encoder. Analogously, a dedicated CNN branch captures fine-grained, local details through successive Incorporation these diverse features, combined with a self-supervised rotation prediction objective, produce a robust diagnostic representation that surpasses standard architectures in performance. Results show better performance not only in accuracy or loss function but also in these algorithms by utilizing capsule networks to preserve spatial orders and realize how each element individually combines and forms whole structures.
>
---
#### [new 010] Towards Realistic Hand-Object Interaction with Gravity-Field Based Diffusion Bridge
- **分类: cs.CV**

- **简介: 该论文提出GravityDB方法，通过引力场模拟手-物体交互，解决现有方法存在的穿透、间隙及手部变形捕捉不足问题，结合语义信息提升交互合理性。**

- **链接: [http://arxiv.org/pdf/2509.03114v1](http://arxiv.org/pdf/2509.03114v1)**

> **作者:** Miao Xu; Xiangyu Zhu; Xusheng Liang; Zidu Wang; Jinlin Wu; Zhen Lei
>
> **摘要:** Existing reconstruction or hand-object pose estimation methods are capable of producing coarse interaction states. However, due to the complex and diverse geometry of both human hands and objects, these approaches often suffer from interpenetration or leave noticeable gaps in regions that are supposed to be in contact. Moreover, the surface of a real human hand undergoes non-negligible deformations during interaction, which are difficult to capture and represent with previous methods. To tackle these challenges, we formulate hand-object interaction as an attraction-driven process and propose a Gravity-Field Based Diffusion Bridge (GravityDB) to simulate interactions between a deformable hand surface and rigid objects. Our approach effectively resolves the aforementioned issues by generating physically plausible interactions that are free of interpenetration, ensure stable grasping, and capture realistic hand deformations. Furthermore, we incorporate semantic information from textual descriptions to guide the construction of the gravitational field, enabling more semantically meaningful interaction regions. Extensive qualitative and quantitative experiments on multiple datasets demonstrate the effectiveness of our method.
>
---
#### [new 011] Joint Training of Image Generator and Detector for Road Defect Detection
- **分类: cs.CV**

- **简介: 该论文提出JTGD方法，联合训练图像生成器与检测器，解决边缘设备上道路缺陷检测的高效问题。通过双判别器和CLIP-FID损失提升生成质量，减少参数量，实现无集成和TTA的高精度检测。**

- **链接: [http://arxiv.org/pdf/2509.03465v1](http://arxiv.org/pdf/2509.03465v1)**

> **作者:** Kuan-Chuan Peng
>
> **备注:** This paper is accepted to ICCV 2025 Workshop on Representation Learning with Very Limited Resources: When Data, Modalities, Labels, and Computing Resources are Scarce as an oral paper
>
> **摘要:** Road defect detection is important for road authorities to reduce the vehicle damage caused by road defects. Considering the practical scenarios where the defect detectors are typically deployed on edge devices with limited memory and computational resource, we aim at performing road defect detection without using ensemble-based methods or test-time augmentation (TTA). To this end, we propose to Jointly Train the image Generator and Detector for road defect detection (dubbed as JTGD). We design the dual discriminators for the generative model to enforce both the synthesized defect patches and overall images to look plausible. The synthesized image quality is improved by our proposed CLIP-based Fr\'echet Inception Distance loss. The generative model in JTGD is trained jointly with the detector to encourage the generative model to synthesize harder examples for the detector. Since harder synthesized images of better quality caused by the aforesaid design are used in the data augmentation, JTGD outperforms the state-of-the-art method in the RDD2022 road defect detection benchmark across various countries under the condition of no ensemble and TTA. JTGD only uses less than 20% of the number of parameters compared with the competing baseline, which makes it more suitable for deployment on edge devices in practice.
>
---
#### [new 012] Resilient Multimodal Industrial Surface Defect Detection with Uncertain Sensors Availability
- **分类: cs.CV**

- **简介: 该论文针对工业表面缺陷检测中的多模态传感器缺失问题，提出交叉模态提示学习与对称对比学习方法，提升RGB与3D模态融合鲁棒性，实验表明在模态缺失场景下性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.02962v1](http://arxiv.org/pdf/2509.02962v1)**

> **作者:** Shuai Jiang; Yunfeng Ma; Jingyu Zhou; Yuan Bian; Yaonan Wang; Min Liu
>
> **备注:** Accepted to IEEE/ASME Transactions on Mechatronics
>
> **摘要:** Multimodal industrial surface defect detection (MISDD) aims to identify and locate defect in industrial products by fusing RGB and 3D modalities. This article focuses on modality-missing problems caused by uncertain sensors availability in MISDD. In this context, the fusion of multiple modalities encounters several troubles, including learning mode transformation and information vacancy. To this end, we first propose cross-modal prompt learning, which includes: i) the cross-modal consistency prompt serves the establishment of information consistency of dual visual modalities; ii) the modality-specific prompt is inserted to adapt different input patterns; iii) the missing-aware prompt is attached to compensate for the information vacancy caused by dynamic modalities-missing. In addition, we propose symmetric contrastive learning, which utilizes text modality as a bridge for fusion of dual vision modalities. Specifically, a paired antithetical text prompt is designed to generate binary text semantics, and triple-modal contrastive pre-training is offered to accomplish multimodal learning. Experiment results show that our proposed method achieves 73.83% I-AUROC and 93.05% P-AUROC with a total missing rate 0.7 for RGB and 3D modalities (exceeding state-of-the-art methods 3.84% and 5.58% respectively), and outperforms existing approaches to varying degrees under different missing types and rates. The source code will be available at https://github.com/SvyJ/MISDD-MM.
>
---
#### [new 013] Enhancing Robustness in Post-Processing Watermarking: An Ensemble Attack Network Using CNNs and Transformers
- **分类: cs.CV**

- **简介: 该论文针对后处理水印的鲁棒性问题，提出集成CNN与Transformer的攻击网络，通过空间与频率域组合提升抗攻击能力，在WAVES基准测试中显著增强水印鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.03006v1](http://arxiv.org/pdf/2509.03006v1)**

> **作者:** Tzuhsuan Huang; Cheng Yu Yeo; Tsai-Ling Huang; Hong-Han Shuai; Wen-Huang Cheng; Jun-Cheng Chen
>
> **备注:** 10 pages
>
> **摘要:** Recent studies on deep watermarking have predominantly focused on in-processing watermarking, which integrates the watermarking process into image generation. However, post-processing watermarking, which embeds watermarks after image generation, offers more flexibility. It can be applied to outputs from any generative model (e.g. GANs, diffusion models) without needing access to the model's internal structure. It also allows users to embed unique watermarks into individual images. Therefore, this study focuses on post-processing watermarking and enhances its robustness by incorporating an ensemble attack network during training. We construct various versions of attack networks using CNN and Transformer in both spatial and frequency domains to investigate how each combination influences the robustness of the watermarking model. Our results demonstrate that combining a CNN-based attack network in the spatial domain with a Transformer-based attack network in the frequency domain yields the highest robustness in watermarking models. Extensive evaluation on the WAVES benchmark, using average bit accuracy as the metric, demonstrates that our ensemble attack network significantly enhances the robustness of baseline watermarking methods under various stress tests. In particular, for the Regeneration Attack defined in WAVES, our method improves StegaStamp by 18.743%. The code is released at:https://github.com/aiiu-lab/DeepRobustWatermark.
>
---
#### [new 014] High-Fidelity Digital Twins for Bridging the Sim2Real Gap in LiDAR-Based ITS Perception
- **分类: cs.CV**

- **简介: 该论文提出高保真数字孪生框架，生成合成数据以解决Sim2Real领域偏移问题，提升LiDAR感知性能，通过分布对齐指标验证效果。**

- **链接: [http://arxiv.org/pdf/2509.02904v1](http://arxiv.org/pdf/2509.02904v1)**

> **作者:** Muhammad Shahbaz; Shaurya Agarwal
>
> **摘要:** Sim2Real domain transfer offers a cost-effective and scalable approach for developing LiDAR-based perception (e.g., object detection, tracking, segmentation) in Intelligent Transportation Systems (ITS). However, perception models trained in simulation often under perform on real-world data due to distributional shifts. To address this Sim2Real gap, this paper proposes a high-fidelity digital twin (HiFi DT) framework that incorporates real-world background geometry, lane-level road topology, and sensor-specific specifications and placement. We formalize the domain adaptation challenge underlying Sim2Real learning and present a systematic method for constructing simulation environments that yield in-domain synthetic data. An off-the-shelf 3D object detector is trained on HiFi DT-generated synthetic data and evaluated on real data. Our experiments show that the DT-trained model outperforms the equivalent model trained on real data by 4.8%. To understand this gain, we quantify distributional alignment between synthetic and real data using multiple metrics, including Chamfer Distance (CD), Maximum Mean Discrepancy (MMD), Earth Mover's Distance (EMD), and Fr'echet Distance (FD), at both raw-input and latent-feature levels. Results demonstrate that HiFi DTs substantially reduce domain shift and improve generalization across diverse evaluation scenarios. These findings underscore the significant role of digital twins in enabling reliable, simulation-based LiDAR perception for real-world ITS applications.
>
---
#### [new 015] Information transmission: Inferring change area from change moment in time series remote sensing images
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CAIM-Net，通过时间序列分析与空间变化检测的内在关系，从变化时刻推断变化区域，解决传统方法将变化区域与时刻视为独立任务导致的不一致性问题，提升生态动态分析的准确性。**

- **链接: [http://arxiv.org/pdf/2509.03112v1](http://arxiv.org/pdf/2509.03112v1)**

> **作者:** Jialu Li; Chen Wu; Meiqi Hu
>
> **摘要:** Time series change detection is a critical task for exploring ecosystem dynamics using time series remote sensing images, because it can simultaneously indicate where and when change occur. While deep learning has shown excellent performance in this domain, it continues to approach change area detection and change moment identification as distinct tasks. Given that change area can be inferred from change moment, we propose a time series change detection network, named CAIM-Net (Change Area Inference from Moment Network), to ensure consistency between change area and change moment results. CAIM-Net infers change area from change moment based on the intrinsic relationship between time series analysis and spatial change detection. The CAIM-Net comprises three key steps: Difference Extraction and Enhancement, Coarse Change Moment Extraction, and Fine Change Moment Extraction and Change Area Inference. In the Difference Extraction and Enhancement, a lightweight encoder with batch dimension stacking is designed to rapidly extract difference features. Subsequently, boundary enhancement convolution is applied to amplify these difference features. In the Coarse Change Moment Extraction, the enhanced difference features from the first step are used to spatiotemporal correlation analysis, and then two distinct methods are employed to determine coarse change moments. In the Fine Change Moment Extraction and Change Area Inference, a multiscale temporal Class Activation Mapping (CAM) module first increases the weight of the change-occurring moment from coarse change moments. Then the weighted change moment is used to infer change area based on the fact that pixels with the change moment must have undergone a change.
>
---
#### [new 016] Scalable and Loosely-Coupled Multimodal Deep Learning for Breast Cancer Subtyping
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对乳腺癌分子分型任务，提出可扩展松耦合多模态框架，整合CNV、临床记录及WSI双表示，结合新融合策略，提升亚型分类性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.03408v1](http://arxiv.org/pdf/2509.03408v1)**

> **作者:** Mohammed Amer; Mohamed A. Suliman; Tu Bui; Nuria Garcia; Serban Georgescu
>
> **摘要:** Healthcare applications are inherently multimodal, benefiting greatly from the integration of diverse data sources. However, the modalities available in clinical settings can vary across different locations and patients. A key area that stands to gain from multimodal integration is breast cancer molecular subtyping, an important clinical task that can facilitate personalized treatment and improve patient prognosis. In this work, we propose a scalable and loosely-coupled multimodal framework that seamlessly integrates data from various modalities, including copy number variation (CNV), clinical records, and histopathology images, to enhance breast cancer subtyping. While our primary focus is on breast cancer, our framework is designed to easily accommodate additional modalities, offering the flexibility to scale up or down with minimal overhead without requiring re-training of existing modalities, making it applicable to other types of cancers as well. We introduce a dual-based representation for whole slide images (WSIs), combining traditional image-based and graph-based WSI representations. This novel dual approach results in significant performance improvements. Moreover, we present a new multimodal fusion strategy, demonstrating its ability to enhance performance across a range of multimodal conditions. Our comprehensive results show that integrating our dual-based WSI representation with CNV and clinical health records, along with our pipeline and fusion strategy, outperforms state-of-the-art methods in breast cancer subtyping.
>
---
#### [new 017] Lesion-Aware Visual-Language Fusion for Automated Image Captioning of Ulcerative Colitis Endoscopic Examinations
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种病变感知的视觉-语言融合框架，用于溃疡性结肠炎内镜图像的自动描述生成。通过整合ResNet、Grad-CAM、CBAM与T5解码器，并注入临床元数据作为提示，提升描述质量与MES分类准确性，支持可靠内镜报告。**

- **链接: [http://arxiv.org/pdf/2509.03011v1](http://arxiv.org/pdf/2509.03011v1)**

> **作者:** Alexis Ivan Lopez Escamilla; Gilberto Ochoa; Sharib Al
>
> **备注:** Miccai Demi Conference 2025
>
> **摘要:** We present a lesion-aware image captioning framework for ulcerative colitis (UC). The model integrates ResNet embeddings, Grad-CAM heatmaps, and CBAM-enhanced attention with a T5 decoder. Clinical metadata (MES score 0-3, vascular pattern, bleeding, erythema, friability, ulceration) is injected as natural-language prompts to guide caption generation. The system produces structured, interpretable descriptions aligned with clinical practice and provides MES classification and lesion tags. Compared with baselines, our approach improves caption quality and MES classification accuracy, supporting reliable endoscopic reporting.
>
---
#### [new 018] SOPSeg: Prompt-based Small Object Instance Segmentation in Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文提出SOPSeg框架，解决遥感图像中小目标实例分割的挑战，通过区域自适应放大、定制解码器和定向提示机制提升精度，构建专用数据集以推动相关研究。**

- **链接: [http://arxiv.org/pdf/2509.03002v1](http://arxiv.org/pdf/2509.03002v1)**

> **作者:** Chenhao Wang; Yingrui Ji; Yu Meng; Yunjian Zhang; Yao Zhu
>
> **摘要:** Extracting small objects from remote sensing imagery plays a vital role in various applications, including urban planning, environmental monitoring, and disaster management. While current research primarily focuses on small object detection, instance segmentation for small objects remains underexplored, with no dedicated datasets available. This gap stems from the technical challenges and high costs of pixel-level annotation for small objects. While the Segment Anything Model (SAM) demonstrates impressive zero-shot generalization, its performance on small-object segmentation deteriorates significantly, largely due to the coarse 1/16 feature resolution that causes severe loss of fine spatial details. To this end, we propose SOPSeg, a prompt-based framework specifically designed for small object segmentation in remote sensing imagery. It incorporates a region-adaptive magnification strategy to preserve fine-grained details, and employs a customized decoder that integrates edge prediction and progressive refinement for accurate boundary delineation. Moreover, we introduce a novel prompting mechanism tailored to the oriented bounding boxes widely adopted in remote sensing applications. SOPSeg outperforms existing methods in small object segmentation and facilitates efficient dataset construction for remote sensing tasks. We further construct a comprehensive small object instance segmentation dataset based on SODA-A, and will release both the model and dataset to support future research.
>
---
#### [new 019] OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation
- **分类: cs.CV**

- **简介: 该论文提出OneCAT模型，用于多模态理解与生成任务，解决传统模型需外部组件（如ViT）导致的效率低下问题。通过纯解码器架构、模态专用MoE结构及多尺度视觉自回归机制，实现高效高分辨率处理，提升多模态生成、编辑与理解性能。**

- **链接: [http://arxiv.org/pdf/2509.03498v1](http://arxiv.org/pdf/2509.03498v1)**

> **作者:** Han Li; Xinyu Peng; Yaoming Wang; Zelin Peng; Xin Chen; Rongxiang Weng; Jingang Wang; Xunliang Cai; Wenrui Dai; Hongkai Xiong
>
> **备注:** technical report
>
> **摘要:** We introduce OneCAT, a unified multimodal model that seamlessly integrates understanding, generation, and editing within a novel, pure decoder-only transformer architecture. Our framework uniquely eliminates the need for external components such as Vision Transformers (ViT) or vision tokenizer during inference, leading to significant efficiency gains, especially for high-resolution inputs. This is achieved through a modality-specific Mixture-of-Experts (MoE) structure trained with a single autoregressive (AR) objective, which also natively supports dynamic resolutions. Furthermore, we pioneer a multi-scale visual autoregressive mechanism within the Large Language Model (LLM) that drastically reduces decoding steps compared to diffusion-based methods while maintaining state-of-the-art performance. Our findings demonstrate the powerful potential of pure autoregressive modeling as a sufficient and elegant foundation for unified multimodal intelligence. As a result, OneCAT sets a new performance standard, outperforming existing open-source unified multimodal models across benchmarks for multimodal generation, editing, and understanding.
>
---
#### [new 020] Preserving instance continuity and length in segmentation through connectivity-aware loss computation
- **分类: cs.CV; I.4.6; I.2.10**

- **简介: 该论文针对生物医学分割中细长结构连续性与长度保持问题，提出两种新型损失函数及实验优化策略，减少分割不连续性，提升下游应用准确性。**

- **链接: [http://arxiv.org/pdf/2509.03154v1](http://arxiv.org/pdf/2509.03154v1)**

> **作者:** Karol Szustakowski; Luk Frank; Julia Esser; Jan Gründemann; Marie Piraud
>
> **备注:** \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** In many biomedical segmentation tasks, the preservation of elongated structure continuity and length is more important than voxel-wise accuracy. We propose two novel loss functions, Negative Centerline Loss and Simplified Topology Loss, that, applied to Convolutional Neural Networks (CNNs), help preserve connectivity of output instances. Moreover, we discuss characteristics of experiment design, such as downscaling and spacing correction, that help obtain continuous segmentation masks. We evaluate our approach on a 3D light-sheet fluorescence microscopy dataset of axon initial segments (AIS), a task prone to discontinuity due to signal dropout. Compared to standard CNNs and existing topology-aware losses, our methods reduce the number of segmentation discontinuities per instance, particularly in regions with missing input signal, resulting in improved instance length calculation in downstream applications. Our findings demonstrate that structural priors embedded in the loss design can significantly enhance the reliability of segmentation for biological applications.
>
---
#### [new 021] SynBT: High-quality Tumor Synthesis for Breast Tumor Segmentation by 3D Diffusion Model
- **分类: cs.CV**

- **简介: 该论文提出SynBT，一种3D扩散模型，用于生成高质量乳腺肿瘤合成图像，解决现有方法在大体积肿瘤（如MRI大FOV）合成效果差的问题。通过结合patch-to-volume自编码器和掩码条件扩散模型，提升分割模型Dice得分2-3%。**

- **链接: [http://arxiv.org/pdf/2509.03267v1](http://arxiv.org/pdf/2509.03267v1)**

> **作者:** Hongxu Yang; Edina Timko; Levente Lippenszky; Vanda Czipczer; Lehel Ferenczi
>
> **备注:** Accepted by MICCAI 2025 Deep-Breath Workshop. Supported by IHI SYNTHIA project
>
> **摘要:** Synthetic tumors in medical images offer controllable characteristics that facilitate the training of machine learning models, leading to an improved segmentation performance. However, the existing methods of tumor synthesis yield suboptimal performances when tumor occupies a large spatial volume, such as breast tumor segmentation in MRI with a large field-of-view (FOV), while commonly used tumor generation methods are based on small patches. In this paper, we propose a 3D medical diffusion model, called SynBT, to generate high-quality breast tumor (BT) in contrast-enhanced MRI images. The proposed model consists of a patch-to-volume autoencoder, which is able to compress the high-resolution MRIs into compact latent space, while preserving the resolution of volumes with large FOV. Using the obtained latent space feature vector, a mask-conditioned diffusion model is used to synthesize breast tumors within selected regions of breast tissue, resulting in realistic tumor appearances. We evaluated the proposed method for a tumor segmentation task, which demonstrated the proposed high-quality tumor synthesis method can facilitate the common segmentation models with performance improvement of 2-3% Dice Score on a large public dataset, and therefore provides benefits for tumor segmentation in MRI images.
>
---
#### [new 022] PRECISE-AS: Personalized Reinforcement Learning for Efficient Point-of-Care Echocardiography in Aortic Stenosis Diagnosis
- **分类: cs.CV**

- **简介: 该论文提出基于强化学习的主动视频选择框架，解决资源有限地区主动脉瓣狭窄诊断中POCUS操作复杂、视频冗余的问题，通过动态优化影像采集提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.02898v1](http://arxiv.org/pdf/2509.02898v1)**

> **作者:** Armin Saadat; Nima Hashemi; Hooman Vaseli; Michael Y. Tsang; Christina Luong; Michiel Van de Panne; Teresa S. M. Tsang; Purang Abolmaesumi
>
> **备注:** To be published in MICCAI 2025
>
> **摘要:** Aortic stenosis (AS) is a life-threatening condition caused by a narrowing of the aortic valve, leading to impaired blood flow. Despite its high prevalence, access to echocardiography (echo), the gold-standard diagnostic tool, is often limited due to resource constraints, particularly in rural and underserved areas. Point-of-care ultrasound (POCUS) offers a more accessible alternative but is restricted by operator expertise and the challenge of selecting the most relevant imaging views. To address this, we propose a reinforcement learning (RL)-driven active video acquisition framework that dynamically selects each patient's most informative echo videos. Unlike traditional methods that rely on a fixed set of videos, our approach continuously evaluates whether additional imaging is needed, optimizing both accuracy and efficiency. Tested on data from 2,572 patients, our method achieves 80.6% classification accuracy while using only 47% of the echo videos compared to a full acquisition. These results demonstrate the potential of active feature acquisition to enhance AS diagnosis, making echocardiographic assessments more efficient, scalable, and personalized. Our source code is available at: https://github.com/Armin-Saadat/PRECISE-AS.
>
---
#### [new 023] Strefer: Empowering Video LLMs with Space-Time Referring and Reasoning via Synthetic Instruction Data
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **简介: 该论文提出Strefer框架，通过合成指令数据增强视频大模型的时空指代与推理能力，解决现有模型在细粒度时空理解上的不足，提升其处理时间事件和空间手势引用的能力。**

- **链接: [http://arxiv.org/pdf/2509.03501v1](http://arxiv.org/pdf/2509.03501v1)**

> **作者:** Honglu Zhou; Xiangyu Peng; Shrikant Kendre; Michael S. Ryoo; Silvio Savarese; Caiming Xiong; Juan Carlos Niebles
>
> **备注:** This technical report serves as the archival version of our paper accepted at the ICCV 2025 Workshop. For more information, please visit our project website: https://strefer.github.io/
>
> **摘要:** Next-generation AI companions must go beyond general video understanding to resolve spatial and temporal references in dynamic, real-world environments. Existing Video Large Language Models (Video LLMs), while capable of coarse-level comprehension, struggle with fine-grained, spatiotemporal reasoning, especially when user queries rely on time-based event references for temporal anchoring, or gestural cues for spatial anchoring to clarify object references and positions. To bridge this critical gap, we introduce Strefer, a synthetic instruction data generation framework designed to equip Video LLMs with spatiotemporal referring and reasoning capabilities. Strefer produces diverse instruction-tuning data using a data engine that pseudo-annotates temporally dense, fine-grained video metadata, capturing rich spatial and temporal information in a structured manner, including subjects, objects, their locations as masklets, and their action descriptions and timelines. Our approach enhances the ability of Video LLMs to interpret spatial and temporal references, fostering more versatile, space-time-aware reasoning essential for real-world AI companions. Without using proprietary models, costly human annotation, or the need to annotate large volumes of new videos, experimental evaluations show that models trained with data produced by Strefer outperform baselines on tasks requiring spatial and temporal disambiguation. Additionally, these models exhibit enhanced space-time-aware reasoning, establishing a new foundation for perceptually grounded, instruction-tuned Video LLMs.
>
---
#### [new 024] Heatmap Guided Query Transformers for Robust Astrocyte Detection across Immunostains and Resolutions
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出混合CNN-Transformer模型，通过热图引导查询机制与轻量Transformer模块，解决星形胶质细胞在多染色和分辨率下的检测难题，优于Faster R-CNN等方法，实现高灵敏度与低假阳性。**

- **链接: [http://arxiv.org/pdf/2509.03323v1](http://arxiv.org/pdf/2509.03323v1)**

> **作者:** Xizhe Zhang; Jiayang Zhu
>
> **摘要:** Astrocytes are critical glial cells whose altered morphology and density are hallmarks of many neurological disorders. However, their intricate branching and stain dependent variability make automated detection of histological images a highly challenging task. To address these challenges, we propose a hybrid CNN Transformer detector that combines local feature extraction with global contextual reasoning. A heatmap guided query mechanism generates spatially grounded anchors for small and faint astrocytes, while a lightweight Transformer module improves discrimination in dense clusters. Evaluated on ALDH1L1 and GFAP stained astrocyte datasets, the model consistently outperformed Faster R-CNN, YOLOv11 and DETR, achieving higher sensitivity with fewer false positives, as confirmed by FROC analysis. These results highlight the potential of hybrid CNN Transformer architectures for robust astrocyte detection and provide a foundation for advanced computational pathology tools.
>
---
#### [new 025] Unveiling the Response of Large Vision-Language Models to Visually Absent Tokens
- **分类: cs.CV; cs.AI**

- **简介: 论文研究大视觉语言模型对视觉缺失文本的反应，解决其误将文本视为图像导致错误响应的问题。通过发现VA神经元并开发检测模块，提出修正方法，提升模型准确性。**

- **链接: [http://arxiv.org/pdf/2509.03025v1](http://arxiv.org/pdf/2509.03025v1)**

> **作者:** Sohee Kim; Soohyun Ryu; Joonhyung Park; Eunho Yang
>
> **备注:** accepted to EMNLP 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) generate contextually relevant responses by jointly interpreting visual and textual inputs. However, our finding reveals they often mistakenly perceive text inputs lacking visual evidence as being part of the image, leading to erroneous responses. In light of this finding, we probe whether LVLMs possess an internal capability to determine if textual concepts are grounded in the image, and discover a specific subset of Feed-Forward Network (FFN) neurons, termed Visual Absence-aware (VA) neurons, that consistently signal the visual absence through a distinctive activation pattern. Leveraging these patterns, we develop a detection module that systematically classifies whether an input token is visually grounded. Guided by its prediction, we propose a method to refine the outputs by reinterpreting question prompts or replacing the detected absent tokens during generation. Extensive experiments show that our method effectively mitigates the models' tendency to falsely presume the visual presence of text input and its generality across various LVLMs.
>
---
#### [new 026] DCDB: Dynamic Conditional Dual Diffusion Bridge for Ill-posed Multi-Tasks
- **分类: cs.CV**

- **简介: 论文提出DCDB框架，解决不适定多任务中任务相关性弱和静态条件控制不足的问题，通过解耦扩散与条件生成，动态调整统计特性，嵌入时间信息，提升学习效果。**

- **链接: [http://arxiv.org/pdf/2509.03044v1](http://arxiv.org/pdf/2509.03044v1)**

> **作者:** Chengjie Huang; Jiafeng Yan; Jing Li; Lu Bai
>
> **备注:** 15 pages,6 figures
>
> **摘要:** Conditional diffusion models have made impressive progress in the field of image processing, but the characteristics of constructing data distribution pathways make it difficult to exploit the intrinsic correlation between tasks in multi-task scenarios, which is even worse in ill-posed tasks with a lack of training data. In addition, traditional static condition control makes it difficult for networks to learn in multi-task scenarios with its dynamically evolving characteristics. To address these challenges, we propose a dynamic conditional double diffusion bridge training paradigm to build a general framework for ill-posed multi-tasks. Firstly, this paradigm decouples the diffusion and condition generation processes, avoiding the dependence of the diffusion model on supervised data in ill-posed tasks. Secondly, generated by the same noise schedule, dynamic conditions are used to gradually adjust their statistical characteristics, naturally embed time-related information, and reduce the difficulty of network learning. We analyze the learning objectives of the network under different conditional forms in the single-step denoising process and compare the changes in its attention weights in the network, demonstrating the superiority of our dynamic conditions. Taking dehazing and visible-infrared fusion as typical ill-posed multi-task scenarios, we achieve the best performance in multiple indicators on public datasets. The code has been publicly released at: https://anonymous.4open.science/r/DCDB-D3C2.
>
---
#### [new 027] Temporally-Aware Diffusion Model for Brain Progression Modelling with Bidirectional Temporal Regularisation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出TADM-3D，一种3D扩散模型，通过脑龄引导和双向时间正则化，解决MRI纵向预测中时间关系建模、插值不足及3D上下文忽略问题，提升脑部进展预测准确性。**

- **链接: [http://arxiv.org/pdf/2509.03141v1](http://arxiv.org/pdf/2509.03141v1)**

> **作者:** Mattia Litrico; Francesco Guarnera; Mario Valerio Giuffrida; Daniele Ravì; Sebastiano Battiato
>
> **摘要:** Generating realistic MRIs to accurately predict future changes in the structure of brain is an invaluable tool for clinicians in assessing clinical outcomes and analysing the disease progression at the patient level. However, current existing methods present some limitations: (i) some approaches fail to explicitly capture the relationship between structural changes and time intervals, especially when trained on age-imbalanced datasets; (ii) others rely only on scan interpolation, which lack clinical utility, as they generate intermediate images between timepoints rather than future pathological progression; and (iii) most approaches rely on 2D slice-based architectures, thereby disregarding full 3D anatomical context, which is essential for accurate longitudinal predictions. We propose a 3D Temporally-Aware Diffusion Model (TADM-3D), which accurately predicts brain progression on MRI volumes. To better model the relationship between time interval and brain changes, TADM-3D uses a pre-trained Brain-Age Estimator (BAE) that guides the diffusion model in the generation of MRIs that accurately reflect the expected age difference between baseline and generated follow-up scans. Additionally, to further improve the temporal awareness of TADM-3D, we propose the Back-In-Time Regularisation (BITR), by training TADM-3D to predict bidirectionally from the baseline to follow-up (forward), as well as from the follow-up to baseline (backward). Although predicting past scans has limited clinical applications, this regularisation helps the model generate temporally more accurate scans. We train and evaluate TADM-3D on the OASIS-3 dataset, and we validate the generalisation performance on an external test set from the NACC dataset. The code will be available upon acceptance.
>
---
#### [new 028] Count2Density: Crowd Density Estimation without Location-level Annotations
- **分类: cs.CV; cs.LG**

- **简介: 论文提出Count2Density，解决人群密度估计中位置级标注困难的问题，通过伪密度图生成和对比正则化，在半监督下超越现有方法。**

- **链接: [http://arxiv.org/pdf/2509.03170v1](http://arxiv.org/pdf/2509.03170v1)**

> **作者:** Mattia Litrico; Feng Chen; Michael Pound; Sotirios A Tsaftaris; Sebastiano Battiato; Mario Valerio Giuffrida
>
> **摘要:** Crowd density estimation is a well-known computer vision task aimed at estimating the density distribution of people in an image. The main challenge in this domain is the reliance on fine-grained location-level annotations, (i.e. points placed on top of each individual) to train deep networks. Collecting such detailed annotations is both tedious, time-consuming, and poses a significant barrier to scalability for real-world applications. To alleviate this burden, we present Count2Density: a novel pipeline designed to predict meaningful density maps containing quantitative spatial information using only count-level annotations (i.e., total number of people) during training. To achieve this, Count2Density generates pseudo-density maps leveraging past predictions stored in a Historical Map Bank, thereby reducing confirmation bias. This bank is initialised using an unsupervised saliency estimator to provide an initial spatial prior and is iteratively updated with an EMA of predicted density maps. These pseudo-density maps are obtained by sampling locations from estimated crowd areas using a hypergeometric distribution, with the number of samplings determined by the count-level annotations. To further enhance the spatial awareness of the model, we add a self-supervised contrastive spatial regulariser to encourage similar feature representations within crowded regions while maximising dissimilarity with background regions. Experimental results demonstrate that our approach significantly outperforms cross-domain adaptation methods and achieves better results than recent state-of-the-art approaches in semi-supervised settings across several datasets. Additional analyses validate the effectiveness of each individual component of our pipeline, confirming the ability of Count2Density to effectively retrieve spatial information from count-level annotations and enabling accurate subregion counting.
>
---
#### [new 029] High Cursive Complex Character Recognition using GAN External Classifier
- **分类: cs.CV**

- **简介: 该论文针对复杂连笔手写字符识别任务，提出ADA-GAN模型。通过GAN生成对抗样本增强数据，解决传统CNN在复杂字符分类中准确率下降的问题，提升模型对复杂字符的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.03062v1](http://arxiv.org/pdf/2509.03062v1)**

> **作者:** S M Rafiuddin
>
> **备注:** Comments: 10 pages, 8 figures, published in the Proceedings of the 2nd International Conference on Computing Advancements (ICCA 2022). Paper introduces ADA-GAN with an external classifier for complex cursive handwritten character recognition, evaluated on MNIST and BanglaLekha datasets, showing improved robustness compared to CNN baselines
>
> **摘要:** Handwritten characters can be trickier to classify due to their complex and cursive nature compared to simple and non-cursive characters. We present an external classifier along with a Generative Adversarial Network that can classify highly cursive and complex characters. The generator network produces fake handwritten character images, which are then used to augment the training data after adding adversarially perturbed noise and achieving a confidence score above a threshold with the discriminator network. The results show that the accuracy of convolutional neural networks decreases as character complexity increases, but our proposed model, ADA-GAN, remains more robust and effective for both cursive and complex characters.
>
---
#### [new 030] 2nd Place Solution for CVPR2024 E2E Challenge: End-to-End Autonomous Driving Using Vision Language Model
- **分类: cs.CV; cs.RO**

- **简介: 论文提出结合端到端架构与视觉语言模型（VLM）的自动驾驶方法，仅用单目摄像头在CVPR2024挑战中取得第二名，验证了视觉驱动方案的有效性。**

- **链接: [http://arxiv.org/pdf/2509.02659v1](http://arxiv.org/pdf/2509.02659v1)**

> **作者:** Zilong Guo; Yi Luo; Long Sha; Dongxu Wang; Panqu Wang; Chenyang Xu; Yi Yang
>
> **备注:** 2nd place in CVPR 2024 End-to-End Driving at Scale Challenge
>
> **摘要:** End-to-end autonomous driving has drawn tremendous attention recently. Many works focus on using modular deep neural networks to construct the end-to-end archi-tecture. However, whether using powerful large language models (LLM), especially multi-modality Vision Language Models (VLM) could benefit the end-to-end driving tasks remain a question. In our work, we demonstrate that combining end-to-end architectural design and knowledgeable VLMs yield impressive performance on the driving tasks. It is worth noting that our method only uses a single camera and is the best camera-only solution across the leaderboard, demonstrating the effectiveness of vision-based driving approach and the potential for end-to-end driving tasks.
>
---
#### [new 031] KEPT: Knowledge-Enhanced Prediction of Trajectories from Consecutive Driving Frames with Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出KEPT框架，通过结合时间频率-空间融合编码器、检索增强和CoT提示，解决短时间轨迹预测中VLMs对场景动态和领域知识建模不足的问题，实现高精度和安全性的轨迹预测。**

- **链接: [http://arxiv.org/pdf/2509.02966v1](http://arxiv.org/pdf/2509.02966v1)**

> **作者:** Yujin Wang; Tianyi Wang; Quanfeng Liu; Wenxian Fan; Junfeng Jiao; Christian Claudel; Yunbing Yan; Bingzhao Gao; Jianqiang Wang; Hong Chen
>
> **摘要:** Accurate short-horizon trajectory prediction is pivotal for safe and reliable autonomous driving, yet existing vision-language models (VLMs) often fail to effectively ground their reasoning in scene dynamics and domain knowledge. To address this challenge, this paper introduces KEPT, a knowledge-enhanced VLM framework that predicts ego trajectories directly from consecutive front-view driving frames. KEPT couples a temporal frequency-spatial fusion (TFSF) video encoder, trained via self-supervised learning with hard-negative mining, with a scalable k-means + HNSW retrieval stack that supplies scene-aligned exemplars. Retrieved priors are embedded into chain-of-thought (CoT) prompts with explicit planning constraints, while a triple-stage fine-tuning schedule incrementally aligns the language head to metric spatial cues, physically feasible motion, and temporally conditioned front-view planning. Evaluated on nuScenes dataset, KEPT achieves state-of-the-art performance across open-loop protocols: under NoAvg, it achieves 0.70m average L2 with a 0.21\% collision rate; under TemAvg with lightweight ego status, it attains 0.31m average L2 and a 0.07\% collision rate. Ablation studies show that all three fine-tuning stages contribute complementary benefits, and that using Top-2 retrieved exemplars yields the best accuracy-safety trade-off. The k-means-clustered HNSW index delivers sub-millisecond retrieval latency, supporting practical deployment. These results indicate that retrieval-augmented, CoT-guided VLMs offer a promising, data-efficient pathway toward interpretable and trustworthy autonomous driving.
>
---
#### [new 032] InstaDA: Augmenting Instance Segmentation Data with Dual-Agent System
- **分类: cs.CV**

- **简介: 论文提出InstaDA，针对实例分割数据标注困难和类别不平衡问题，设计双代理系统（文本代理优化提示，图像代理生成新实例），提升数据多样性和分布，实验显示指标提升。**

- **链接: [http://arxiv.org/pdf/2509.02973v1](http://arxiv.org/pdf/2509.02973v1)**

> **作者:** Xianbao Hou; Yonghao He; Zeyd Boukhers; John See; Hu Su; Wei Sui; Cong Yang
>
> **摘要:** Acquiring high-quality instance segmentation data is challenging due to the labor-intensive nature of the annotation process and significant class imbalances within datasets. Recent studies have utilized the integration of Copy-Paste and diffusion models to create more diverse datasets. However, these studies often lack deep collaboration between large language models (LLMs) and diffusion models, and underutilize the rich information within the existing training data. To address these limitations, we propose InstaDA, a novel, training-free Dual-Agent system designed to augment instance segmentation datasets. First, we introduce a Text-Agent (T-Agent) that enhances data diversity through collaboration between LLMs and diffusion models. This agent features a novel Prompt Rethink mechanism, which iteratively refines prompts based on the generated images. This process not only fosters collaboration but also increases image utilization and optimizes the prompts themselves. Additionally, we present an Image-Agent (I-Agent) aimed at enriching the overall data distribution. This agent augments the training set by generating new instances conditioned on the training images. To ensure practicality and efficiency, both agents operate as independent and automated workflows, enhancing usability. Experiments conducted on the LVIS 1.0 validation set indicate that InstaDA achieves significant improvements, with an increase of +4.0 in box average precision (AP) and +3.3 in mask AP compared to the baseline. Furthermore, it outperforms the leading model, DiverGen, by +0.3 in box AP and +0.1 in mask AP, with a notable +0.7 gain in box AP on common categories and mask AP gains of +0.2 on common categories and +0.5 on frequent categories.
>
---
#### [new 033] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出基于梯度自反思的方法，通过检测视觉标记并整合到对比解码框架，缓解多模态模型中的文本-视觉和共现偏差，无需额外资源，有效提升准确率。**

- **链接: [http://arxiv.org/pdf/2509.03113v1](http://arxiv.org/pdf/2509.03113v1)**

> **作者:** Shan Wang; Maying Shen; Nadine Chang; Chuong Nguyen; Hongdong Li; Jose M. Alvarez
>
> **摘要:** Hallucinations in multimodal large language model are caused by the text-visual bias and the co-occurrence bias. The former reflects an over-reliance on text information in the decision-making process, while the latter arises from the statistical object-pairing patterns abstracted from the training data. Existing mitigation methods heuristically address these biases without understanding the fluctuating bias level across the instances. We first propose estimating the influence of respective token types (visual, prompt, and previous outputs) using a gradient-based self-reflection method. The estimated token influence further enables the detection of object-related visual tokens and their integration into an influence-aware contrastive decoding framework to mitigate both types of biases simultaneously. Our method operates without the need for additional resources, such as costly fine-tuning, extra models, or data statistics. Extensive experiments show it effectively reduces hallucinations, achieving up to a 92% accuracy increase on LLaVA-QA90.
>
---
#### [new 034] PI3DETR: Parametric Instance Detection of 3D Point Cloud Edges with a Geometry-Aware 3DETR
- **分类: cs.CV**

- **简介: 论文提出PI3DETR，直接从点云预测3D参数化曲线（如直线、圆弧），解决传统多阶段处理的复杂性与噪声敏感问题。通过几何感知匹配和专用损失函数实现统一检测，提升鲁棒性，适用于LiDAR数据。**

- **链接: [http://arxiv.org/pdf/2509.03262v1](http://arxiv.org/pdf/2509.03262v1)**

> **作者:** Fabio F. Oberweger; Michael Schwingshackl; Vanessa Staderini
>
> **摘要:** We present PI3DETR, an end-to-end framework that directly predicts 3D parametric curve instances from raw point clouds, avoiding the intermediate representations and multi-stage processing common in prior work. Extending 3DETR, our model introduces a geometry-aware matching strategy and specialized loss functions that enable unified detection of differently parameterized curve types, including cubic B\'ezier curves, line segments, circles, and arcs, in a single forward pass. Optional post-processing steps further refine predictions without adding complexity. This streamlined design improves robustness to noise and varying sampling densities, addressing critical challenges in real world LiDAR and 3D sensing scenarios. PI3DETR sets a new state-of-the-art on the ABC dataset and generalizes effectively to real sensor data, offering a simple yet powerful solution for 3D edge and curve estimation.
>
---
#### [new 035] A comprehensive Persian offline handwritten database for investigating the effects of heritability and family relationships on handwriting
- **分类: cs.CV**

- **简介: 论文创建了一个波斯语离线手写数据库，包含210个家庭成员的多种手写样本，用于研究遗传性和家庭关系对书写的影响，旨在探讨书写是否具有遗传性及家庭关系如何影响书写风格。**

- **链接: [http://arxiv.org/pdf/2509.03510v1](http://arxiv.org/pdf/2509.03510v1)**

> **作者:** Abbas Zohrevand; Javad Sadri; Zahra Imani
>
> **摘要:** This paper introduces a comprehensive database for research and investigation on the effects of inheritance on handwriting. A database has been created that can be used to answer questions such as: Is there a genetic component to handwriting? Is handwriting inherited? Do family relationships affect handwriting? Varieties of samples of handwritten components such as: digits, letters, shapes and free paragraphs of 210 families including (grandparents, parents, uncles, aunts, siblings, cousins, nephews and nieces) have been collected using specially designed forms, and family relationships of all writers are captured. To the best of our knowledge, no such database is presently available. Based on comparisons and investigation of features of handwritings of family members, similarities among their features and writing styles are detected. Our database is freely available to the pattern recognition community and hope it will pave the way for investigations on the effects of inheritance and family relationships on handwritings.
>
---
#### [new 036] MedLiteNet: Lightweight Hybrid Medical Image Segmentation Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MedLiteNet，解决皮肤病变分割中CNN接收野有限、Transformer复杂度高的问题。通过轻量化CNN-Transformer混合架构，结合多尺度上下文聚合与边界感知注意力机制，实现高效精准的医学图像分割。**

- **链接: [http://arxiv.org/pdf/2509.03041v1](http://arxiv.org/pdf/2509.03041v1)**

> **作者:** Pengyang Yu; Haoquan Wang; Gerard Marks; Tahar Kechadi; Laurence T. Yang; Sahraoui Dhelim; Nyothiri Aung
>
> **摘要:** Accurate skin-lesion segmentation remains a key technical challenge for computer-aided diagnosis of skin cancer. Convolutional neural networks, while effective, are constrained by limited receptive fields and thus struggle to model long-range dependencies. Vision Transformers capture global context, yet their quadratic complexity and large parameter budgets hinder use on the small-sample medical datasets common in dermatology. We introduce the MedLiteNet, a lightweight CNN Transformer hybrid tailored for dermoscopic segmentation that achieves high precision through hierarchical feature extraction and multi-scale context aggregation. The encoder stacks depth-wise Mobile Inverted Bottleneck blocks to curb computation, inserts a bottleneck-level cross-scale token-mixing unit to exchange information between resolutions, and embeds a boundary-aware self-attention module to sharpen lesion contours.
>
---
#### [new 037] Backdoor Poisoning Attack Against Face Spoofing Attack Detection Methods
- **分类: cs.CV**

- **简介: 论文研究人脸识别反欺骗检测中的后门攻击，提出一种通过嵌入欺骗特征到活体图像中，绕过检测的攻击方法，验证其对现有系统的威胁。**

- **链接: [http://arxiv.org/pdf/2509.03108v1](http://arxiv.org/pdf/2509.03108v1)**

> **作者:** Shota Iwamatsu; Koichi Ito; Takafumi Aoki
>
> **备注:** 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)
>
> **摘要:** Face recognition systems are robust against environmental changes and noise, and thus may be vulnerable to illegal authentication attempts using user face photos, such as spoofing attacks. To prevent such spoofing attacks, it is crucial to discriminate whether the input image is a live user image or a spoofed image prior to the face recognition process. Most existing spoofing attack detection methods utilize deep learning, which necessitates a substantial amount of training data. Consequently, if malicious data is injected into a portion of the training dataset, a specific spoofing attack may be erroneously classified as live, leading to false positives.In this paper, we propose a novel backdoor poisoning attack method to demonstrate the latent threat of backdoor poisoning within face anti-spoofing detection. The proposed method enables certain spoofing attacks to bypass detection by embedding features extracted from the spoofing attack's face image into a live face image without inducing any perceptible visual alterations.Through experiments conducted on public datasets, we demonstrate that the proposed method constitutes a realistic threat to existing spoofing attack detection systems.
>
---
#### [new 038] STAR: A Fast and Robust Rigid Registration Framework for Serial Histopathological Images
- **分类: cs.CV**

- **简介: 该论文提出STAR框架，解决连续组织切片图像刚性配准问题，克服传统方法计算复杂、难以复现的局限。通过集成预处理与分层策略，实现跨染色和组织类型的高效配准，适用于病理分析与AI数据准备。**

- **链接: [http://arxiv.org/pdf/2509.02952v1](http://arxiv.org/pdf/2509.02952v1)**

> **作者:** Zeyu Liu; Shengwei Ding
>
> **备注:** The code is available at https://github.com/Rowerliu/STAR
>
> **摘要:** Registration of serial whole-slide histopathological images (WSIs) is critical for enabling direct comparison across diverse stains and for preparing paired datasets in artificial intelligence (AI) workflows such as virtual staining and biomarker prediction. While existing methods often rely on complex deformable or deep learning approaches that are computationally intensive and difficult to reproduce, lightweight rigid frameworks-sufficient for many consecutive-section scenarios-remain underdeveloped. We introduce STAR (Serial Tissue Alignment for Rigid registration), a fast and robust open-source framework for multi-WSI alignment. STAR integrates stain-conditioned preprocessing with a hierarchical coarse-to-fine correlation strategy, adaptive kernel scaling, and built-in quality control, achieving reliable rigid registration across heterogeneous tissue types and staining protocols, including hematoxylin-eosin (H&E), special histochemical stains (e.g., PAS, PASM, Masson's), and immunohistochemical (IHC) markers (e.g., CD31, KI67). Evaluated on the ANHIR 2019 and ACROBAT 2022 datasets spanning multiple organs and scanning conditions, STAR consistently produced stable alignments within minutes per slide, demonstrating robustness to cross-stain variability and partial tissue overlap. Beyond benchmarks, we present case studies on H&E-IHC alignment, construction of multi-IHC panels, and typical failure modes, underscoring both utility and limitations. Released as an open and lightweight tool, STAR provides a reproducible baseline that lowers the barrier for clinical adoption and enables large-scale paired data preparation for next-generation computational pathology.
>
---
#### [new 039] Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?
- **分类: cs.CV**

- **简介: 该论文提出T2I-CoReBench基准，评估文本到图像模型的组合与推理能力，解决现有基准无法全面评估复杂场景与多步推理的问题，通过高密度场景和多步推理测试，涵盖12维分类。**

- **链接: [http://arxiv.org/pdf/2509.03516v1](http://arxiv.org/pdf/2509.03516v1)**

> **作者:** Ouxiang Li; Yuan Wang; Xinting Hu; Huijuan Huang; Rui Chen; Jiarong Ou; Xin Tao; Pengfei Wan; Fuli Feng
>
> **备注:** Project Page: https://t2i-corebench.github.io/
>
> **摘要:** Text-to-image (T2I) generation aims to synthesize images from textual prompts, which jointly specify what must be shown and imply what can be inferred, thereby corresponding to two core capabilities: composition and reasoning. However, with the emerging advances of T2I models in reasoning beyond composition, existing benchmarks reveal clear limitations in providing comprehensive evaluations across and within these capabilities. Meanwhile, these advances also enable models to handle more complex prompts, whereas current benchmarks remain limited to low scene density and simplified one-to-one reasoning. To address these limitations, we propose T2I-CoReBench, a comprehensive and complex benchmark that evaluates both composition and reasoning capabilities of T2I models. To ensure comprehensiveness, we structure composition around scene graph elements (instance, attribute, and relation) and reasoning around the philosophical framework of inference (deductive, inductive, and abductive), formulating a 12-dimensional evaluation taxonomy. To increase complexity, driven by the inherent complexities of real-world scenarios, we curate each prompt with high compositional density for composition and multi-step inference for reasoning. We also pair each prompt with a checklist that specifies individual yes/no questions to assess each intended element independently to facilitate fine-grained and reliable evaluation. In statistics, our benchmark comprises 1,080 challenging prompts and around 13,500 checklist questions. Experiments across 27 current T2I models reveal that their composition capability still remains limited in complex high-density scenarios, while the reasoning capability lags even further behind as a critical bottleneck, with all models struggling to infer implicit elements from prompts. Our project page: https://t2i-corebench.github.io/.
>
---
#### [new 040] Background Matters Too: A Language-Enhanced Adversarial Framework for Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对行人重识别任务，解决前景定位与背景噪声抑制难题。提出融合前景-背景双分支模型，通过语义对齐与对抗学习策略增强特征判别性，提升复杂场景下的重识别性能。**

- **链接: [http://arxiv.org/pdf/2509.03032v1](http://arxiv.org/pdf/2509.03032v1)**

> **作者:** Kaicong Huang; Talha Azfar; Jack M. Reilly; Thomas Guggisberg; Ruimin Ke
>
> **摘要:** Person re-identification faces two core challenges: precisely locating the foreground target while suppressing background noise and extracting fine-grained features from the target region. Numerous visual-only approaches address these issues by partitioning an image and applying attention modules, yet they rely on costly manual annotations and struggle with complex occlusions. Recent multimodal methods, motivated by CLIP, introduce semantic cues to guide visual understanding. However, they focus solely on foreground information, but overlook the potential value of background cues. Inspired by human perception, we argue that background semantics are as important as the foreground semantics in ReID, as humans tend to eliminate background distractions while focusing on target appearance. Therefore, this paper proposes an end-to-end framework that jointly models foreground and background information within a dual-branch cross-modal feature extraction pipeline. To help the network distinguish between the two domains, we propose an intra-semantic alignment and inter-semantic adversarial learning strategy. Specifically, we align visual and textual features that share the same semantics across domains, while simultaneously penalizing similarity between foreground and background features to enhance the network's discriminative power. This strategy drives the model to actively suppress noisy background regions and enhance attention toward identity-relevant foreground cues. Comprehensive experiments on two holistic and two occluded ReID benchmarks demonstrate the effectiveness and generality of the proposed method, with results that match or surpass those of current state-of-the-art approaches.
>
---
#### [new 041] Parameter-Efficient Adaptation of mPLUG-Owl2 via Pixel-Level Visual Prompts for NR-IQA
- **分类: cs.CV**

- **简介: 该论文提出基于像素级视觉提示的参数高效方法，用于无参考图像质量评估（NR-IQA），通过冻结基础模型仅训练60万参数，实现低资源下的高性能评估。**

- **链接: [http://arxiv.org/pdf/2509.03494v1](http://arxiv.org/pdf/2509.03494v1)**

> **作者:** Yahya Benmahane; Mohammed El Hassouni
>
> **摘要:** In this paper, we propose a novel parameter-efficient adaptation method for No- Reference Image Quality Assessment (NR-IQA) using visual prompts optimized in pixel-space. Unlike full fine-tuning of Multimodal Large Language Models (MLLMs), our approach trains only 600K parameters at most (< 0.01% of the base model), while keeping the underlying model fully frozen. During inference, these visual prompts are combined with images via addition and processed by mPLUG-Owl2 with the textual query "Rate the technical quality of the image." Evaluations across distortion types (synthetic, realistic, AI-generated) on KADID- 10k, KonIQ-10k, and AGIQA-3k demonstrate competitive performance against full finetuned methods and specialized NR-IQA models, achieving 0.93 SRCC on KADID-10k. To our knowledge, this is the first work to leverage pixel-space visual prompts for NR-IQA, enabling efficient MLLM adaptation for low-level vision tasks. The source code is publicly available at https: // github. com/ yahya-ben/ mplug2-vp-for-nriqa .
>
---
#### [new 042] AIVA: An AI-based Virtual Companion for Emotion-aware Interaction
- **分类: cs.CV**

- **简介: 论文提出情感感知虚拟助手AIVA，解决LLMs无法处理非语言情感问题，通过多模态感知网络、情感提示工程及TTS动画模块实现沉浸式交互。**

- **链接: [http://arxiv.org/pdf/2509.03212v1](http://arxiv.org/pdf/2509.03212v1)**

> **作者:** Chenxi Li
>
> **摘要:** Recent advances in Large Language Models (LLMs) have significantly improved natural language understanding and generation, enhancing Human-Computer Interaction (HCI). However, LLMs are limited to unimodal text processing and lack the ability to interpret emotional cues from non-verbal signals, hindering more immersive and empathetic interactions. This work explores integrating multimodal sentiment perception into LLMs to create emotion-aware agents. We propose \ours, an AI-based virtual companion that captures multimodal sentiment cues, enabling emotionally aligned and animated HCI. \ours introduces a Multimodal Sentiment Perception Network (MSPN) using a cross-modal fusion transformer and supervised contrastive learning to provide emotional cues. Additionally, we develop an emotion-aware prompt engineering strategy for generating empathetic responses and integrate a Text-to-Speech (TTS) system and animated avatar module for expressive interactions. \ours provides a framework for emotion-aware agents with applications in companion robotics, social care, mental health, and human-centered AI.
>
---
#### [new 043] VQualA 2025 Challenge on Engagement Prediction for Short Videos: Methods and Results
- **分类: cs.CV; cs.MM; cs.SI**

- **简介: 该论文介绍VQualA 2025挑战赛，旨在通过多模态特征建模短视频用户参与度，解决UGC内容流行度预测问题。使用真实互动数据集，吸引97参与者，推动该领域研究进展。**

- **链接: [http://arxiv.org/pdf/2509.02969v1](http://arxiv.org/pdf/2509.02969v1)**

> **作者:** Dasong Li; Sizhuo Ma; Hang Hua; Wenjie Li; Jian Wang; Chris Wei Zhou; Fengbin Guan; Xin Li; Zihao Yu; Yiting Lu; Ru-Ling Liao; Yan Ye; Zhibo Chen; Wei Sun; Linhan Cao; Yuqin Cao; Weixia Zhang; Wen Wen; Kaiwei Zhang; Zijian Chen; Fangfang Lu; Xiongkuo Min; Guangtao Zhai; Erjia Xiao; Lingfeng Zhang; Zhenjie Su; Hao Cheng; Yu Liu; Renjing Xu; Long Chen; Xiaoshuai Hao; Zhenpeng Zeng; Jianqin Wu; Xuxu Wang; Qian Yu; Bo Hu; Weiwei Wang; Pinxin Liu; Yunlong Tang; Luchuan Song; Jinxi He; Jiaru Wu; Hanjia Lyu
>
> **备注:** ICCV 2025 VQualA workshop EVQA track
>
> **摘要:** This paper presents an overview of the VQualA 2025 Challenge on Engagement Prediction for Short Videos, held in conjunction with ICCV 2025. The challenge focuses on understanding and modeling the popularity of user-generated content (UGC) short videos on social media platforms. To support this goal, the challenge uses a new short-form UGC dataset featuring engagement metrics derived from real-world user interactions. This objective of the Challenge is to promote robust modeling strategies that capture the complex factors influencing user engagement. Participants explored a variety of multi-modal features, including visual content, audio, and metadata provided by creators. The challenge attracted 97 participants and received 15 valid test submissions, contributing significantly to progress in short-form UGC video engagement prediction.
>
---
#### [new 044] PointAD+: Learning Hierarchical Representations for Zero-shot 3D Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出PointAD+框架，通过隐式（渲染）与显式（几何）分层表示学习，结合跨层次对比对齐，解决零样本3D异常检测问题，提升对未见对象的异常识别能力。**

- **链接: [http://arxiv.org/pdf/2509.03277v1](http://arxiv.org/pdf/2509.03277v1)**

> **作者:** Qihang Zhou; Shibo He; Jiangtao Yan; Wenchao Meng; Jiming Chen
>
> **备注:** Submitted to TPAMI
>
> **摘要:** In this paper, we aim to transfer CLIP's robust 2D generalization capabilities to identify 3D anomalies across unseen objects of highly diverse class semantics. To this end, we propose a unified framework to comprehensively detect and segment 3D anomalies by leveraging both point- and pixel-level information. We first design PointAD, which leverages point-pixel correspondence to represent 3D anomalies through their associated rendering pixel representations. This approach is referred to as implicit 3D representation, as it focuses solely on rendering pixel anomalies but neglects the inherent spatial relationships within point clouds. Then, we propose PointAD+ to further broaden the interpretation of 3D anomalies by introducing explicit 3D representation, emphasizing spatial abnormality to uncover abnormal spatial relationships. Hence, we propose G-aggregation to involve geometry information to enable the aggregated point representations spatially aware. To simultaneously capture rendering and spatial abnormality, PointAD+ proposes hierarchical representation learning, incorporating implicit and explicit anomaly semantics into hierarchical text prompts: rendering prompts for the rendering layer and geometry prompts for the geometry layer. A cross-hierarchy contrastive alignment is further introduced to promote the interaction between the rendering and geometry layers, facilitating mutual anomaly learning. Finally, PointAD+ integrates anomaly semantics from both layers to capture the generalized anomaly semantics. During the test, PointAD+ can integrate RGB information in a plug-and-play manner and further improve its detection performance. Extensive experiments demonstrate the superiority of PointAD+ in ZS 3D anomaly detection across unseen objects with highly diverse class semantics, achieving a holistic understanding of abnormality.
>
---
#### [new 045] InfraDiffusion: zero-shot depth map restoration with diffusion models and prompted segmentation from sparse infrastructure point clouds
- **分类: cs.CV**

- **简介: 论文提出InfraDiffusion框架，通过扩散模型和提示分割，解决低光环境下点云稀疏噪声导致的砖级分割难题，无需任务训练提升深度图质量，实现基础设施自动化检测。**

- **链接: [http://arxiv.org/pdf/2509.03324v1](http://arxiv.org/pdf/2509.03324v1)**

> **作者:** Yixiong Jing; Cheng Zhang; Haibing Wu; Guangming Wang; Olaf Wysocki; Brian Sheil
>
> **摘要:** Point clouds are widely used for infrastructure monitoring by providing geometric information, where segmentation is required for downstream tasks such as defect detection. Existing research has automated semantic segmentation of structural components, while brick-level segmentation (identifying defects such as spalling and mortar loss) has been primarily conducted from RGB images. However, acquiring high-resolution images is impractical in low-light environments like masonry tunnels. Point clouds, though robust to dim lighting, are typically unstructured, sparse, and noisy, limiting fine-grained segmentation. We present InfraDiffusion, a zero-shot framework that projects masonry point clouds into depth maps using virtual cameras and restores them by adapting the Denoising Diffusion Null-space Model (DDNM). Without task-specific training, InfraDiffusion enhances visual clarity and geometric consistency of depth maps. Experiments on masonry bridge and tunnel point cloud datasets show significant improvements in brick-level segmentation using the Segment Anything Model (SAM), underscoring its potential for automated inspection of masonry assets. Our code and data is available at https://github.com/Jingyixiong/InfraDiffusion-official-implement.
>
---
#### [new 046] A Data-Driven RetinaNet Model for Small Object Detection in Aerial Images
- **分类: cs.CV; cs.LG**

- **简介: 本文提出DDR-Net，用于航拍图像中小目标检测，解决数据有限下的检测难题。通过数据驱动优化特征与锚框，结合创新采样技术，提升模型性能，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2509.02928v1](http://arxiv.org/pdf/2509.02928v1)**

> **作者:** Zhicheng Tang; Jinwen Tang; Yi Shang
>
> **摘要:** In the realm of aerial imaging, the ability to detect small objects is pivotal for a myriad of applications, encompassing environmental surveillance, urban design, and crisis management. Leveraging RetinaNet, this work unveils DDR-Net: a data-driven, deep-learning model devised to enhance the detection of diminutive objects. DDR-Net introduces novel, data-driven techniques to autonomously ascertain optimal feature maps and anchor estimations, cultivating a tailored and proficient training process while maintaining precision. Additionally, this paper presents an innovative sampling technique to bolster model efficacy under limited data training constraints. The model's enhanced detection capabilities support critical applications including wildlife and habitat monitoring, traffic flow optimization, and public safety improvements through accurate identification of small objects like vehicles and pedestrians. DDR-Net significantly reduces the cost and time required for data collection and training, offering efficient performance even with limited data. Empirical assessments over assorted aerial avian imagery datasets demonstrate that DDR-Net markedly surpasses RetinaNet and alternative contemporary models. These innovations advance current aerial image analysis technologies and promise wide-ranging impacts across multiple sectors including agriculture, security, and archaeology.
>
---
#### [new 047] TRELLIS-Enhanced Surface Features for Comprehensive Intracranial Aneurysm Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出利用TRELLIS生成的表面特征，解决颅内动脉瘤检测、分割与血流预测中因标注数据不足导致的性能瓶颈，通过跨域特征迁移提升模型准确性与分割质量。**

- **链接: [http://arxiv.org/pdf/2509.03095v1](http://arxiv.org/pdf/2509.03095v1)**

> **作者:** Clément Hervé; Paul Garnier; Jonathan Viquerat; Elie Hachem
>
> **摘要:** Intracranial aneurysms pose a significant clinical risk yet are difficult to detect, delineate and model due to limited annotated 3D data. We propose a cross-domain feature-transfer approach that leverages the latent geometric embeddings learned by TRELLIS, a generative model trained on large-scale non-medical 3D datasets, to augment neural networks for aneurysm analysis. By replacing conventional point normals or mesh descriptors with TRELLIS surface features, we systematically enhance three downstream tasks: (i) classifying aneurysms versus healthy vessels in the Intra3D dataset, (ii) segmenting aneurysm and vessel regions on 3D meshes, and (iii) predicting time-evolving blood-flow fields using a graph neural network on the AnXplore dataset. Our experiments show that the inclusion of these features yields strong gains in accuracy, F1-score and segmentation quality over state-of-the-art baselines, and reduces simulation error by 15\%. These results illustrate the broader potential of transferring 3D representations from general-purpose generative models to specialized medical tasks.
>
---
#### [new 048] PercepTwin: Modeling High-Fidelity Digital Twins for Sim2Real LiDAR-based Perception for Intelligent Transportation Systems
- **分类: cs.CV**

- **简介: 论文提出PercepTwin方法，构建高保真数字孪生，解决LiDAR感知需大量标注数据的问题，通过合成数据集和开源资源提升Sim2Real学习效果。**

- **链接: [http://arxiv.org/pdf/2509.02903v1](http://arxiv.org/pdf/2509.02903v1)**

> **作者:** Muhammad Shahbaz; Shaurya Agarwal
>
> **摘要:** LiDAR-based perception in intelligent transportation systems (ITS), for tasks such as object detection, tracking, and semantic and instance segmentation, is predominantly solved by deep neural network models which often require large-scale labeled datasets during training to achieve generalization. However, creating these datasets is costly. time consuming and require human labor before the datasets are ready for training models. This hinders scalability of the LiDAR-based perception systems in ITS. Sim2Real learning offers scalable alternative, however, its effectiveness is dependent on the fidelity of the source simulation(s) to real-world, in terms of environment structure, actor dynamics, and sensor emulations. In response, this paper introduces a rigorous and reproducible methodology for creating large-scale, high-quality synthetic datasets using High-Fidelity Digital Twins (HiFi DTs). The proposed workflow outlines the steps, tools, and best practices for digitally replicating real-world environments, encompassing static geometry modeling, road infrastructure replication, and dynamic traffic scenario generation. Leveraging open-source and readily available resources such as satellite imagery and OpenStreetMap data, alongside specific sensor configurations, this paper provides practical, detailed guidance for constructing robust synthetic environments. These environments subsequently facilitate scalable, cost-effective, and diverse dataset generation, forming a reliable foundation for robust Sim2Real learning.
>
---
#### [new 049] Single Domain Generalization in Diabetic Retinopathy: A Neuro-Symbolic Learning Approach
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出KG-DG框架，结合神经网络与符号推理，解决糖尿病视网膜病变分类中的单/多领域泛化问题，通过临床本体和血管分割特征融合，提升模型在未见领域的准确率。**

- **链接: [http://arxiv.org/pdf/2509.02918v1](http://arxiv.org/pdf/2509.02918v1)**

> **作者:** Midhat Urooj; Ayan Banerjee; Farhat Shaikh; Kuntal Thakur; Sandeep Gupta
>
> **备注:** Accepted in ANSyA 2025: 1st International Workshop on Advanced Neuro-Symbolic Applications
>
> **摘要:** Domain generalization remains a critical challenge in medical imaging, where models trained on single sources often fail under real-world distribution shifts. We propose KG-DG, a neuro-symbolic framework for diabetic retinopathy (DR) classification that integrates vision transformers with expert-guided symbolic reasoning to enable robust generalization across unseen domains. Our approach leverages clinical lesion ontologies through structured, rule-based features and retinal vessel segmentation, fusing them with deep visual representations via a confidence-weighted integration strategy. The framework addresses both single-domain generalization (SDG) and multi-domain generalization (MDG) by minimizing the KL divergence between domain embeddings, thereby enforcing alignment of high-level clinical semantics. Extensive experiments across four public datasets (APTOS, EyePACS, Messidor-1, Messidor-2) demonstrate significant improvements: up to a 5.2% accuracy gain in cross-domain settings and a 6% improvement over baseline ViT models. Notably, our symbolic-only model achieves a 63.67% average accuracy in MDG, while the complete neuro-symbolic integration achieves the highest accuracy compared to existing published baselines and benchmarks in challenging SDG scenarios. Ablation studies reveal that lesion-based features (84.65% accuracy) substantially outperform purely neural approaches, confirming that symbolic components act as effective regularizers beyond merely enhancing interpretability. Our findings establish neuro-symbolic integration as a promising paradigm for building clinically robust, and domain-invariant medical AI systems.
>
---
#### [new 050] SPENet: Self-guided Prototype Enhancement Network for Few-shot Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出SPENet，解决少样本医学图像分割中忽略类别内差异和跨图像差异的问题。通过多级原型生成与查询引导的局部原型增强模块，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2509.02993v1](http://arxiv.org/pdf/2509.02993v1)**

> **作者:** Chao Fan; Xibin Jia; Anqi Xiao; Hongyuan Yu; Zhenghan Yang; Dawei Yang; Hui Xu; Yan Huang; Liang Wang
>
> **备注:** Accepted by MICCAI2025
>
> **摘要:** Few-Shot Medical Image Segmentation (FSMIS) aims to segment novel classes of medical objects using only a few labeled images. Prototype-based methods have made significant progress in addressing FSMIS. However, they typically generate a single global prototype for the support image to match with the query image, overlooking intra-class variations. To address this issue, we propose a Self-guided Prototype Enhancement Network (SPENet). Specifically, we introduce a Multi-level Prototype Generation (MPG) module, which enables multi-granularity measurement between the support and query images by simultaneously generating a global prototype and an adaptive number of local prototypes. Additionally, we observe that not all local prototypes in the support image are beneficial for matching, especially when there are substantial discrepancies between the support and query images. To alleviate this issue, we propose a Query-guided Local Prototype Enhancement (QLPE) module, which adaptively refines support prototypes by incorporating guidance from the query image, thus mitigating the negative effects of such discrepancies. Extensive experiments on three public medical datasets demonstrate that SPENet outperforms existing state-of-the-art methods, achieving superior performance.
>
---
#### [new 051] Decoding Visual Neural Representations by Multimodal with Dynamic Balancing
- **分类: cs.CV**

- **简介: 该论文提出多模态框架整合EEG、图像与文本，解决低信噪比EEG解码及跨模态对齐问题。引入文本语义增强对齐，设计适配器模块与动态平衡策略，结合正则化提升泛化。在ThingsEEG上超越SOTA，Top-1/Top-5提升2.0%/4.7%。**

- **链接: [http://arxiv.org/pdf/2509.03433v1](http://arxiv.org/pdf/2509.03433v1)**

> **作者:** Kaili sun; Xingyu Miao; Bing Zhai; Haoran Duan; Yang Long
>
> **摘要:** In this work, we propose an innovative framework that integrates EEG, image, and text data, aiming to decode visual neural representations from low signal-to-noise ratio EEG signals. Specifically, we introduce text modality to enhance the semantic correspondence between EEG signals and visual content. With the explicit semantic labels provided by text, image and EEG features of the same category can be more closely aligned with the corresponding text representations in a shared multimodal space. To fully utilize pre-trained visual and textual representations, we propose an adapter module that alleviates the instability of high-dimensional representation while facilitating the alignment and fusion of cross-modal features. Additionally, to alleviate the imbalance in multimodal feature contributions introduced by the textual representations, we propose a Modal Consistency Dynamic Balance (MCDB) strategy that dynamically adjusts the contribution weights of each modality. We further propose a stochastic perturbation regularization (SPR) term to enhance the generalization ability of semantic perturbation-based models by introducing dynamic Gaussian noise in the modality optimization process. The evaluation results on the ThingsEEG dataset show that our method surpasses previous state-of-the-art methods in both Top-1 and Top-5 accuracy metrics, improving by 2.0\% and 4.7\% respectively.
>
---
#### [new 052] TinyDrop: Tiny Model Guided Token Dropping for Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TinyDrop框架，通过轻量模型引导的token dropping减少ViT推理计算量，解决大模型高成本问题，无需训练且兼容多架构，在保持精度前提下降低FLOPs达80%。**

- **链接: [http://arxiv.org/pdf/2509.03379v1](http://arxiv.org/pdf/2509.03379v1)**

> **作者:** Guoxin Wang; Qingyuan Wang; Binhua Huang; Shaowu Chen; Deepu John
>
> **摘要:** Vision Transformers (ViTs) achieve strong performance in image classification but incur high computational costs from processing all image tokens. To reduce inference costs in large ViTs without compromising accuracy, we propose TinyDrop, a training-free token dropping framework guided by a lightweight vision model. The guidance model estimates the importance of tokens while performing inference, thereby selectively discarding low-importance tokens if large vit models need to perform attention calculations. The framework operates plug-and-play, requires no architectural modifications, and is compatible with diverse ViT architectures. Evaluations on standard image classification benchmarks demonstrate that our framework reduces FLOPs by up to 80% for ViTs with minimal accuracy degradation, highlighting its generalization capability and practical utility for efficient ViT-based classification.
>
---
#### [new 053] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?
- **分类: cs.CV**

- **简介: 该论文研究视频多模态模型是否利用运动信息进行像素级视觉定位。提出MoCentric-Bench基准，设计运动中心探针技术，评估模型对运动与语言交互的理解，建立单图像基线并探索适配方法，推动密集时空定位研究。**

- **链接: [http://arxiv.org/pdf/2509.02807v1](http://arxiv.org/pdf/2509.02807v1)**

> **作者:** Mennatullah Siam
>
> **备注:** Work under review in NeurIPS 2025 with the title "Are we using Motion in Referring Segmentation? A Motion-Centric Evaluation"
>
> **摘要:** Multi-modal large language models (MLLMs) have shown impressive generalization across tasks using images and text modalities. While their extension to video has enabled tasks such as video question answering and video captioning, their pixel-level visual grounding abilities are less studied. In this work, we raise the pertinent question of whether motion is used in pixel-level visual grounding and whether video MLLMs can segment objects based on natural language expressions describing their motion patterns. We identify the shortcomings in the current benchmarks, where we show that a single frame can often suffice for capturing the motion referring expression without any temporal reasoning. To address this, we introduce four motion-centric probing techniques, particularly designed for the visual grounding task, to study video MLLMs' ability to identify true motion from a fake one and their ability to grasp the motion order. Consequently, we provide a motion-centric benchmark, MoCentric-Bench. It ensures that video MLLMs are evaluated towards leveraging the interaction between motion and language rather than being dominated by static appearance cues emphasized in existing visual grounding datasets. We further establish strong single-image baselines that are on par with or outperform prior methods. Finally, we explore simple motion-centric adaptation techniques that provide state-of-the-art performance on our MoCentric-Bench. Our motion-centric benchmark, evaluation and findings challenge future models to improve dense spatiotemporal grounding and pixel-level understanding within videos. Code and datasets will be made publicly available at https://github.com/MSiam/PixFoundation-2.0.git.
>
---
#### [new 054] RTGMFF: Enhanced fMRI-based Brain Disorder Diagnosis via ROI-driven Text Generation and Multimodal Feature Fusion
- **分类: cs.CV**

- **简介: 该论文提出RTGMFF框架，通过ROI驱动文本生成与多模态特征融合，解决fMRI脑疾病诊断中的低信噪比、个体差异及文本注释不足问题，提升诊断准确率。**

- **链接: [http://arxiv.org/pdf/2509.03214v1](http://arxiv.org/pdf/2509.03214v1)**

> **作者:** Junhao Jia; Yifei Sun; Yunyou Liu; Cheng Yang; Changmiao Wang; Feiwei Qin; Yong Peng; Wenwen Min
>
> **摘要:** Functional magnetic resonance imaging (fMRI) is a powerful tool for probing brain function, yet reliable clinical diagnosis is hampered by low signal-to-noise ratios, inter-subject variability, and the limited frequency awareness of prevailing CNN- and Transformer-based models. Moreover, most fMRI datasets lack textual annotations that could contextualize regional activation and connectivity patterns. We introduce RTGMFF, a framework that unifies automatic ROI-level text generation with multimodal feature fusion for brain-disorder diagnosis. RTGMFF consists of three components: (i) ROI-driven fMRI text generation deterministically condenses each subject's activation, connectivity, age, and sex into reproducible text tokens; (ii) Hybrid frequency-spatial encoder fuses a hierarchical wavelet-mamba branch with a cross-scale Transformer encoder to capture frequency-domain structure alongside long-range spatial dependencies; and (iii) Adaptive semantic alignment module embeds the ROI token sequence and visual features in a shared space, using a regularized cosine-similarity loss to narrow the modality gap. Extensive experiments on the ADHD-200 and ABIDE benchmarks show that RTGMFF surpasses current methods in diagnostic accuracy, achieving notable gains in sensitivity, specificity, and area under the ROC curve. Code is available at https://github.com/BeistMedAI/RTGMFF.
>
---
#### [new 055] Isolated Bangla Handwritten Character Classification using Transfer Learning
- **分类: cs.CV**

- **简介: 该论文针对孟加拉语手写字符分类任务，解决基本与复合字符识别问题。采用迁移学习结合3DCNN、ResNet和MobileNet模型，在Bangla Lekha数据集上实现99.82%训练准确率和99.46%测试准确率，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.03061v1](http://arxiv.org/pdf/2509.03061v1)**

> **作者:** Abdul Karim; S M Rafiuddin; Jahidul Islam Razin; Tahira Alam
>
> **备注:** Comments: 13 pages, 14 figures, published in the Proceedings of the 2nd International Conference on Computing Advancements (ICCA 2022), IEEE. Strong experimental section with comparisons across models (3DCNN, ResNet50, MobileNet)
>
> **摘要:** Bangla language consists of fifty distinct characters and many compound characters. Several notable studies have been performed to recognize Bangla characters, both handwritten and optical. Our approach uses transfer learning to classify the basic, distinct, as well as compound Bangla handwritten characters while avoiding the vanishing gradient problem. Deep Neural Network techniques such as 3D Convolutional Neural Network (3DCNN), Residual Neural Network (ResNet), and MobileNet are applied to generate an end-to-end classification of all possible standard formations of handwritten characters in the Bangla language. The Bangla Lekha Isolated dataset, which contains 166,105 Bangla character image samples categorized into 84 distinct classes, is used for this classification model. The model achieved 99.82% accuracy on training data and 99.46% accuracy on test data. Comparisons with various state-of-the-art benchmarks of Bangla handwritten character classification show that the proposed model achieves better accuracy in classifying the data.
>
---
#### [new 056] Empowering Lightweight MLLMs with Reasoning via Long CoT SFT
- **分类: cs.CV**

- **简介: 该论文旨在提升轻量级多模态语言模型（MLLMs）的推理能力，解决其现有方法效果不足的问题。通过结合长链式思维（CoT）监督微调（SFT）与强化学习（RL），验证了长CoT数据对MLLMs推理能力提升的关键作用。**

- **链接: [http://arxiv.org/pdf/2509.03321v1](http://arxiv.org/pdf/2509.03321v1)**

> **作者:** Linyu Ou
>
> **摘要:** While Reinforcement Learning with Verifiable Rewards has enhanced the reasoning of large-scale language models (LLMs), its efficacy for lightweight multimodal language models (MLLMs) with fewer than seven billion parameters remains underexplored. This paper investigates the role of long Chain-of-Thought (long CoT) data in enhancing the reasoning abilities of such MLLMs. Our findings demonstrate that Supervised Fine-Tuning (SFT) with long CoT data significantly improves MLLM reasoning. Furthermore, we observe that after this initial SFT phase, MLLMs can achieve additional performance gains through a subsequent RL stage. We conclude that a SFT stage with long CoT data is a critical prerequisite for developing the reasoning capabilities of lightweight MLLMs.
>
---
#### [new 057] PPORLD-EDNetLDCT: A Proximal Policy Optimization-Based Reinforcement Learning Framework for Adaptive Low-Dose CT Denoising
- **分类: cs.CV**

- **简介: 该论文提出PPORLD-EDNetLDCT框架，基于PPO强化学习和编码器-解码器结构，解决低剂量CT降噪中传统方法图像质量差的问题，通过实时优化策略提升PSNR、SSIM等指标。**

- **链接: [http://arxiv.org/pdf/2509.03185v1](http://arxiv.org/pdf/2509.03185v1)**

> **作者:** Debopom Sutradhar; Ripon Kumar Debnath; Mohaimenul Azam Khan Raiaan; Yan Zhang; Reem E. Mohamed; Sami Azam
>
> **备注:** 20 pages, 5 figures, 5 tables. Submitted to Computers in Biology and Medicine for peer review
>
> **摘要:** Low-dose computed tomography (LDCT) is critical for minimizing radiation exposure, but it often leads to increased noise and reduced image quality. Traditional denoising methods, such as iterative optimization or supervised learning, often fail to preserve image quality. To address these challenges, we introduce PPORLD-EDNetLDCT, a reinforcement learning-based (RL) approach with Encoder-Decoder for LDCT. Our method utilizes a dynamic RL-based approach in which an advanced posterior policy optimization (PPO) algorithm is used to optimize denoising policies in real time, based on image quality feedback, trained via a custom gym environment. The experimental results on the low dose CT image and projection dataset demonstrate that the proposed PPORLD-EDNetLDCT model outperforms traditional denoising techniques and other DL-based methods, achieving a peak signal-to-noise ratio of 41.87, a structural similarity index measure of 0.9814 and a root mean squared error of 0.00236. Moreover, in NIH-AAPM-Mayo Clinic Low Dose CT Challenge dataset our method achived a PSNR of 41.52, SSIM of 0.9723 and RMSE of 0.0051. Furthermore, we validated the quality of denoising using a classification task in the COVID-19 LDCT dataset, where the images processed by our method improved the classification accuracy to 94\%, achieving 4\% higher accuracy compared to denoising without RL-based denoising. This method offers a promising solution for safer and more accurate LDCT imaging.
>
---
#### [new 058] Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey
- **分类: cs.RO; cs.CV**

- **简介: 该论文系统综述基于大VLM的VLA模型在机器人操作中的应用，解决传统方法在复杂环境中的不足，通过分类架构、分析集成领域及提出未来方向，整合现有研究并填补关键空白。**

- **链接: [http://arxiv.org/pdf/2508.13073v2](http://arxiv.org/pdf/2508.13073v2)**

> **作者:** Rui Shao; Wei Li; Lingsen Zhang; Renshan Zhang; Zhiyang Liu; Ran Chen; Liqiang Nie
>
> **备注:** Project Page: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
>
> **摘要:** Robotic manipulation, a key frontier in robotics and embodied AI, requires precise motor control and multimodal understanding, yet traditional rule-based methods fail to scale or generalize in unstructured, novel environments. In recent years, Vision-Language-Action (VLA) models, built upon Large Vision-Language Models (VLMs) pretrained on vast image-text datasets, have emerged as a transformative paradigm. This survey provides the first systematic, taxonomy-oriented review of large VLM-based VLA models for robotic manipulation. We begin by clearly defining large VLM-based VLA models and delineating two principal architectural paradigms: (1) monolithic models, encompassing single-system and dual-system designs with differing levels of integration; and (2) hierarchical models, which explicitly decouple planning from execution via interpretable intermediate representations. Building on this foundation, we present an in-depth examination of large VLM-based VLA models: (1) integration with advanced domains, including reinforcement learning, training-free optimization, learning from human videos, and world model integration; (2) synthesis of distinctive characteristics, consolidating architectural traits, operational strengths, and the datasets and benchmarks that support their development; (3) identification of promising directions, including memory mechanisms, 4D perception, efficient adaptation, multi-agent cooperation, and other emerging capabilities. This survey consolidates recent advances to resolve inconsistencies in existing taxonomies, mitigate research fragmentation, and fill a critical gap through the systematic integration of studies at the intersection of large VLMs and robotic manipulation. We provide a regularly updated project page to document ongoing progress: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
>
---
#### [new 059] Is Synthetic Image Augmentation Useful for Imbalanced Classification Problems? Case-Study on the MIDOG2025 Atypical Cell Detection Competition
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文研究异常有丝分裂分类任务，针对高度不平衡的病理图像数据，比较了合成数据增强与不同预训练模型的效果，发现合成平衡无显著提升，而ImageNet预训练模型表现更优。**

- **链接: [http://arxiv.org/pdf/2509.02612v1](http://arxiv.org/pdf/2509.02612v1)**

> **作者:** Leire Benito-Del-Valle; Pedro A. Moreno-Sánchez; Itziar Egusquiza; Itsaso Vitoria; Artzai Picón; Cristina López-Saratxaga; Adrian Galdran
>
> **备注:** version 0, to be updated; submitted to midog 2025
>
> **摘要:** The MIDOG 2025 challenge extends prior work on mitotic figure detection by introducing a new Track 2 on atypical mitosis classification. This task aims to distinguish normal from atypical mitotic figures in histopathology images, a clinically relevant but highly imbalanced and cross-domain problem. We investigated two complementary backbones: (i) ConvNeXt-Small, pretrained on ImageNet, and (ii) a histopathology-specific ViT from Lunit trained via self-supervision. To address the strong prevalence imbalance (9408 normal vs. 1741 atypical), we synthesized additional atypical examples to approximate class balance and compared models trained with real-only vs. real+synthetic data. Using five-fold cross-validation, both backbones reached strong performance (mean AUROC approximately 95 percent), with ConvNeXt achieving slightly higher peaks while Lunit exhibited greater fold-to-fold stability. Synthetic balancing, however, did not lead to consistent improvements. On the organizers' preliminary hidden test set, explicitly designed as an out-of-distribution debug subset, ConvNeXt attained the highest AUROC (95.4 percent), whereas Lunit remained competitive on balanced accuracy. These findings suggest that both ImageNet and domain-pretrained backbones are viable for atypical mitosis classification, with domain-pretraining conferring robustness and ImageNet pretraining reaching higher peaks, while naive synthetic balancing has limited benefit. Full hidden test set results will be reported upon challenge completion.
>
---
#### [new 060] Ensemble of Pathology Foundation Models for MIDOG 2025 Track 2: Atypical Mitosis Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对医学图像中异常有丝分裂分类任务，提出基于病理基础模型的集成方法。通过预训练模型高效微调、鱼眼变换增强特征及傅里叶域适配，结合多模型集成，提升对肿瘤侵袭性评估关键指标的判别准确性。**

- **链接: [http://arxiv.org/pdf/2509.02591v1](http://arxiv.org/pdf/2509.02591v1)**

> **作者:** Mieko Ochi; Bae Yuan
>
> **摘要:** Mitotic figures are classified into typical and atypical variants, with atypical counts correlating strongly with tumor aggressiveness. Accurate differentiation is therefore essential for patient prognostication and resource allocation, yet remains challenging even for expert pathologists. Here, we leveraged Pathology Foundation Models (PFMs) pre-trained on large histopathology datasets and applied parameter-efficient fine-tuning via low-rank adaptation. During training, we employ a fisheye transform to emphasize mitoses and Fourier Domain Adaptation using ImageNet target images. Finally, we ensembled multiple PFMs to integrate complementary morphological insights, achieving a high balanced accuracy on the Preliminary Evaluation Phase dataset.
>
---
#### [new 061] Efficient Active Training for Deep LiDAR Odometry
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出主动训练框架，用于提升LiDAR里程计的效率与泛化能力。针对传统方法需大量多样数据的问题，通过ITSS和AIS策略选择性提取训练样本，以52%数据量达到全数据集效果，优化训练过程并增强模型对复杂环境的适应性。**

- **链接: [http://arxiv.org/pdf/2509.03211v1](http://arxiv.org/pdf/2509.03211v1)**

> **作者:** Beibei Zhou; Zhiyuan Zhang; Zhenbo Song; Jianhui Guo; Hui Kong
>
> **摘要:** Robust and efficient deep LiDAR odometry models are crucial for accurate localization and 3D reconstruction, but typically require extensive and diverse training data to adapt to diverse environments, leading to inefficiencies. To tackle this, we introduce an active training framework designed to selectively extract training data from diverse environments, thereby reducing the training load and enhancing model generalization. Our framework is based on two key strategies: Initial Training Set Selection (ITSS) and Active Incremental Selection (AIS). ITSS begins by breaking down motion sequences from general weather into nodes and edges for detailed trajectory analysis, prioritizing diverse sequences to form a rich initial training dataset for training the base model. For complex sequences that are difficult to analyze, especially under challenging snowy weather conditions, AIS uses scene reconstruction and prediction inconsistency to iteratively select training samples, refining the model to handle a wide range of real-world scenarios. Experiments across datasets and weather conditions validate our approach's effectiveness. Notably, our method matches the performance of full-dataset training with just 52\% of the sequence volume, demonstrating the training efficiency and robustness of our active training paradigm. By optimizing the training process, our approach sets the stage for more agile and reliable LiDAR odometry systems, capable of navigating diverse environmental conditions with greater precision.
>
---
#### [new 062] Application of Quantum Convolutional Neural Networks for MRI-Based Brain Tumor Detection and Classification
- **分类: physics.med-ph; cs.CV**

- **简介: 该论文探索量子卷积神经网络（QCNN）在MRI脑肿瘤检测与分类中的应用，解决类别不平衡问题，构建二分类和多分类模型，提升肿瘤识别准确率。**

- **链接: [http://arxiv.org/pdf/2509.02582v1](http://arxiv.org/pdf/2509.02582v1)**

> **作者:** Sugih Pratama Nugraha; Ariiq Islam Alfajri; Tony Sumaryada; Duong Thanh Tai; Nissren Tamam; Abdelmoneim Sulieman; Sitti Yani
>
> **摘要:** This study explores the application of Quantum Convolutional Neural Networks (QCNNs) for brain tumor classification using MRI images, leveraging quantum computing for enhanced computational efficiency. A dataset of 3,264 MRI images, including glioma, meningioma, pituitary tumors, and non-tumor cases, was utilized. The data was split into 80% training and 20% testing, with an oversampling technique applied to address class imbalance. The QCNN model consists of quantum convolution layers, flatten layers, and dense layers, with a filter size of 2, depth of 4, and 4 qubits, trained over 10 epochs. Two models were developed: a binary classification model distinguishing tumor presence and a multiclass classification model categorizing tumor types. The binary model achieved 88% accuracy, improving to 89% after data balancing, while the multiclass model achieved 52% accuracy, increasing to 62% after oversampling. Despite strong binary classification performance, the multiclass model faced challenges due to dataset complexity and quantum circuit limitations. These findings suggest that QCNNs hold promise for medical imaging applications, particularly in binary classification. However, further refinements, including optimized quantum circuit architectures and hybrid classical-quantum approaches, are necessary to enhance multiclass classification accuracy and improve QCNN applicability in clinical settings.
>
---
#### [new 063] Solutions for Mitotic Figure Detection and Atypical Classification in MIDOG 2025
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对MIDOG 2025挑战的两个任务：有丝分裂图像检测与异常分类，提出两阶段检测-分类框架和多模型集成策略，提升检测精度与分类鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.02597v1](http://arxiv.org/pdf/2509.02597v1)**

> **作者:** Shuting Xu; Runtong Liu; Zhixuan Chen; Junlin Hou; Hao Chen
>
> **摘要:** Deep learning has driven significant advances in mitotic figure analysis within computational pathology. In this paper, we present our approach to the Mitosis Domain Generalization (MIDOG) 2025 Challenge, which consists of two distinct tasks, i.e., mitotic figure detection and atypical mitosis classification. For the mitotic figure detection task, we propose a two-stage detection-classification framework that first localizes candidate mitotic figures and subsequently refines the predictions using a dedicated classification module. For the atypical mitosis classification task, we employ an ensemble strategy that integrates predictions from multiple state-of-the-art deep learning architectures to improve robustness and accuracy. Extensive experiments demonstrate the effectiveness of our proposed methods across both tasks.
>
---
#### [new 064] ConvNeXt with Histopathology-Specific Augmentations for Mitotic Figure Classification
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于ConvNeXt的轻量模型，结合组织病理学特定增强（弹性变换、染色调整）和平衡采样，解决有丝分裂图分类中的形态差异、类别不平衡及领域转移问题，达到0.8961平衡准确率。**

- **链接: [http://arxiv.org/pdf/2509.02595v1](http://arxiv.org/pdf/2509.02595v1)**

> **作者:** Hana Feki; Alice Blondel; Thomas Walter
>
> **摘要:** Accurate mitotic figure classification is crucial in computational pathology, as mitotic activity informs cancer grading and patient prognosis. Distinguishing atypical mitotic figures (AMFs), which indicate higher tumor aggressiveness, from normal mitotic figures (NMFs) remains challenging due to subtle morphological differences and high intra-class variability. This task is further complicated by domain shifts, including variations in organ, tissue type, and scanner, as well as limited annotations and severe class imbalance. To address these challenges in Track 2 of the MIDOG 2025 Challenge, we propose a solution based on the lightweight ConvNeXt architecture, trained on all available datasets (AMi-Br, AtNorM-Br, AtNorM-MD, and OMG-Octo) to maximize domain coverage. Robustness is enhanced through a histopathology-specific augmentation pipeline, including elastic and stain-specific transformations, and balanced sampling to mitigate class imbalance. A grouped 5-fold cross-validation strategy ensures reliable evaluation. On the preliminary leaderboard, our model achieved a balanced accuracy of 0.8961, ranking among the top entries. These results highlight that broad domain exposure combined with targeted augmentation strategies is key to building accurate and generalizable mitotic figure classifiers.
>
---
#### [new 065] Adaptive Learning Strategies for Mitotic Figure Classification in MIDOG2025 Challenge
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对MIDOG2025挑战中的不典型有丝分裂图分类任务，解决形态模糊和扫描差异导致的检测难题。通过视觉提示调优与测试时增强结合染色归一化，提升模型在多样化成像条件下的鲁棒性与准确率。**

- **链接: [http://arxiv.org/pdf/2509.02640v1](http://arxiv.org/pdf/2509.02640v1)**

> **作者:** Biwen Meng; Xi Long; Jingxin Liu
>
> **摘要:** Atypical mitotic figures (AMFs) are clinically relevant indicators of abnormal cell division, yet their reliable detection remains challenging due to morphological ambiguity and scanner variability. In this work, we investigated three variants of adapting the pathology foundation model UNI2-h for the MIDOG2025 Track 2 challenge. Starting from a LoRA-based baseline, we found that visual prompt tuning (VPT) substantially improved generalization, and that further integrating test-time augmentation (TTA) with Vahadane and Macenko stain normalization provided the best robustness. Our final submission achieved a balanced accuracy of 0.8837 and an ROC-AUC of 0.9513 on the preliminary leaderboard, ranking within the top 10 teams. These results demonstrate that prompt-based adaptation combined with stain-normalization TTA offers an effective strategy for atypical mitosis classification under diverse imaging conditions.
>
---
#### [new 066] Prompt-Guided Patch UNet-VAE with Adversarial Supervision for Adrenal Gland Segmentation in Computed Tomography Medical Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出一种结合VAE-UNet、对抗训练和合成块的框架，解决CT图像中肾上腺分割的挑战，提升边界敏感区域的分割精度与重建质量。**

- **链接: [http://arxiv.org/pdf/2509.03188v1](http://arxiv.org/pdf/2509.03188v1)**

> **作者:** Hania Ghouse; Muzammil Behzad
>
> **摘要:** Segmentation of small and irregularly shaped abdominal organs, such as the adrenal glands in CT imaging, remains a persistent challenge due to severe class imbalance, poor spatial context, and limited annotated data. In this work, we propose a unified framework that combines variational reconstruction, supervised segmentation, and adversarial patch-based feedback to address these limitations in a principled and scalable manner. Our architecture is built upon a VAE-UNet backbone that jointly reconstructs input patches and generates voxel-level segmentation masks, allowing the model to learn disentangled representations of anatomical structure and appearance. We introduce a patch-based training pipeline that selectively injects synthetic patches generated from the learned latent space, and systematically study the effects of varying synthetic-to-real patch ratios during training. To further enhance output fidelity, the framework incorporates perceptual reconstruction loss using VGG features, as well as a PatchGAN-style discriminator for adversarial supervision over spatial realism. Comprehensive experiments on the BTCV dataset demonstrate that our approach improves segmentation accuracy, particularly in boundary-sensitive regions, while maintaining strong reconstruction quality. Our findings highlight the effectiveness of hybrid generative-discriminative training regimes for small-organ segmentation and provide new insights into balancing realism, diversity, and anatomical consistency in data-scarce scenarios.
>
---
#### [new 067] EclipseTouch: Touch Segmentation on Ad Hoc Surfaces using Worn Infrared Shadow Casting
- **分类: cs.HC; cs.CV; cs.GR; cs.RO**

- **简介: 论文提出EclipseTouch技术，通过红外阴影投射与摄像头结合，实现对非专用表面的高精度触摸检测，解决混合现实系统中未仪器化表面的触控难题。**

- **链接: [http://arxiv.org/pdf/2509.03430v1](http://arxiv.org/pdf/2509.03430v1)**

> **作者:** Vimal Mollyn; Nathan DeVrio; Chris Harrison
>
> **备注:** Accepted to UIST 2025
>
> **摘要:** The ability to detect touch events on uninstrumented, everyday surfaces has been a long-standing goal for mixed reality systems. Prior work has shown that virtual interfaces bound to physical surfaces offer performance and ergonomic benefits over tapping at interfaces floating in the air. A wide variety of approaches have been previously developed, to which we contribute a new headset-integrated technique called \systemname. We use a combination of a computer-triggered camera and one or more infrared emitters to create structured shadows, from which we can accurately estimate hover distance (mean error of 6.9~mm) and touch contact (98.0\% accuracy). We discuss how our technique works across a range of conditions, including surface material, interaction orientation, and environmental lighting.
>
---
#### [new 068] RF-DETR for Robust Mitotic Figure Detection: A MIDOG 2025 Track 1 Approach
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出RF-DETR单阶段检测方法，用于解决不同扫描仪和染色协议下有丝分裂图检测的领域迁移问题，通过硬负样本挖掘和数据平衡提升模型泛化能力，参与MIDOG 2025挑战赛Track 1。**

- **链接: [http://arxiv.org/pdf/2509.02599v1](http://arxiv.org/pdf/2509.02599v1)**

> **作者:** Piotr Giedziun; Jan Sołtysik; Mateusz Górczany; Norbert Ropiak; Marcin Przymus; Piotr Krajewski; Jarosław Kwiecień; Artur Bartczak; Izabela Wasiak; Mateusz Maniewski
>
> **备注:** Challenge report for MIDOG 2025 Track 1
>
> **摘要:** Mitotic figure detection in histopathology images remains challenging due to significant domain shifts across different scanners, staining protocols, and tissue types. This paper presents our approach for the MIDOG 2025 challenge Track 1, focusing on robust mitotic figure detection across diverse histological contexts. While we initially planned a two-stage approach combining high-recall detection with subsequent classification refinement, time constraints led us to focus on optimizing a single-stage detection pipeline. We employed RF-DETR (Roboflow Detection Transformer) with hard negative mining, trained on MIDOG++ dataset. On the preliminary test set, our method achieved an F1 score of 0.789 with a recall of 0.839 and precision of 0.746, demonstrating effective generalization across unseen domains. The proposed solution offers insights into the importance of training data balance and hard negative mining for addressing domain shift challenges in mitotic figure detection.
>
---
#### [new 069] Ensemble YOLO Framework for Multi-Domain Mitotic Figure Detection in Histopathology Images
- **分类: eess.IV; cs.CV; 68T07; I.4.9; I.5.4**

- **简介: 论文提出基于YOLOv5和YOLOv8集成框架，解决多领域病理图像中稀少、形态多样的有丝分裂图检测问题。通过颜色扰动和纹理增强提升模型鲁棒性，结合模型优势提高检测敏感度与精度。**

- **链接: [http://arxiv.org/pdf/2509.02957v1](http://arxiv.org/pdf/2509.02957v1)**

> **作者:** Navya Sri Kelam; Akash Parekh; Saikiran Bonthu; Nitin Singhal
>
> **备注:** 3pages, MIDOG25 Challenge
>
> **摘要:** Accurate detection of mitotic figures in whole slide histopathological images remains a challenging task due to their scarcity, morphological heterogeneity, and the variability introduced by tissue preparation and staining protocols. The MIDOG competition series provides standardized benchmarks for evaluating detection approaches across diverse domains, thus motivating the development of generalizable deep learning models. In this work, we investigate the performance of two modern one-stage detectors, YOLOv5 and YOLOv8, trained on MIDOG++, CMC, and CCMCT datasets. To enhance robustness, training incorporated stain-invariant color perturbations and texture preserving augmentations. In internal validation, YOLOv5 achieved superior precision, while YOLOv8 provided improved recall, reflecting architectural trade-offs between anchor-based and anchor-free detection. To capitalize on these complementary strengths, we employed an ensemble of the two models, which improved sensitivity without a major reduction in precision. These findings highlight the effectiveness of ensemble strategies built upon contemporary object detectors to advance automated mitosis detection in digital pathology.
>
---
#### [new 070] Challenges and Lessons from MIDOG 2025: A Two-Stage Approach to Domain-Robust Mitotic Figure Detection
- **分类: eess.IV; cs.CV**

- **简介: 论文提出两阶段方法解决跨领域有丝分裂图检测问题，结合Faster R-CNN与三种分类器集成以减少假阳性。尽管召回率高，但精度低，揭示了领域泛化和形态区分的挑战。**

- **链接: [http://arxiv.org/pdf/2509.02630v1](http://arxiv.org/pdf/2509.02630v1)**

> **作者:** Euiseop Song; Jaeyoung Park; Jaewoo Park
>
> **摘要:** Mitotic figure detection remains a challenging task in computational pathology due to domain variability and morphological complexity. This paper describes our participation in the MIDOG 2025 challenge, focusing on robust mitotic figure detection across diverse tissue domains. We developed a two-stage pipeline combining Faster R-CNN for candidate detection with an ensemble of three classifiers (DenseNet-121, EfficientNet-v2, InceptionResNet-v2) for false positive reduction. Our best submission achieved F1-score 0.2237 (Recall: 0.9528, Precision: 0.1267) using a Faster R-CNN trained solely on MIDOG++ dataset. While our high recall demonstrates effective mitotic figure detection, the critically low precision (12.67%) reveals fundamental challenges in distinguishing true mitoses from morphologically similar imposters across diverse domains. Analysis of six submission variants showed that subsequent optimization attempts were counterproductive, highlighting the omplexity of domain generalization in histopathology. This work provides valuable insights into the practical challenges of developing robust mitotic figure detection algorithms and emphasizes the importance of effective false positive suppression strategies.
>
---
#### [new 071] Deep Self-knowledge Distillation: A hierarchical supervised learning for coronary artery segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 本文提出Deep Self-knowledge Distillation方法，通过分层监督和双损失函数提升冠状动脉分割性能，解决现有模型性能差、泛化不足问题，在XCAD和DCA1数据集上取得更好效果。**

- **链接: [http://arxiv.org/pdf/2509.03173v1](http://arxiv.org/pdf/2509.03173v1)**

> **作者:** Mingfeng Lin
>
> **摘要:** Coronary artery disease is a leading cause of mortality, underscoring the critical importance of precise diagnosis through X-ray angiography. Manual coronary artery segmentation from these images is time-consuming and inefficient, prompting the development of automated models. However, existing methods, whether rule-based or deep learning models, struggle with issues like poor performance and limited generalizability. Moreover, current knowledge distillation methods applied in this field have not fully exploited the hierarchical knowledge of the model, leading to certain information waste and insufficient enhancement of the model's performance capabilities for segmentation tasks. To address these issues, this paper introduces Deep Self-knowledge Distillation, a novel approach for coronary artery segmentation that leverages hierarchical outputs for supervision. By combining Deep Distribution Loss and Pixel-wise Self-knowledge Distillation Loss, our method enhances the student model's segmentation performance through a hierarchical learning strategy, effectively transferring knowledge from the teacher model. Our method combines a loosely constrained probabilistic distribution vector with tightly constrained pixel-wise supervision, providing dual regularization for the segmentation model while also enhancing its generalization and robustness. Extensive experiments on XCAD and DCA1 datasets demonstrate that our approach outperforms the dice coefficient, accuracy, sensitivity and IoU compared to other models in comparative evaluations.
>
---
#### [new 072] Normal and Atypical Mitosis Image Classifier using Efficient Vision Transformer
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对正常与异常有丝分裂图像分类任务，提出基于EfficientViT-L2混合模型的解决方案，通过统一数据集、交叉验证及图像增强技术，实现高准确率（0.85）和AUC（0.942）的分类模型。**

- **链接: [http://arxiv.org/pdf/2509.02589v1](http://arxiv.org/pdf/2509.02589v1)**

> **作者:** Xuan Qi; Dominic Labella; Thomas Sanford; Maxwell Lee
>
> **备注:** for grandchallenge midog 2025 track 2 abstract
>
> **摘要:** We tackle atypical versus normal mitosis classification in the MIDOG 2025 challenge using EfficientViT-L2, a hybrid CNN--ViT architecture optimized for accuracy and efficiency. A unified dataset of 13,938 nuclei from seven cancer types (MIDOG++ and AMi-Br) was used, with atypical mitoses comprising ~15. To assess domain generalization, we applied leave-one-cancer-type-out cross-validation with 5-fold ensembles, using stain-deconvolution for image augmentation. For challenge submissions, we trained an ensemble with the same 5-fold split but on all cancer types. In the preliminary evaluation phase, this model achieved balanced accuracy of 0.859, ROC AUC of 0.942, and raw accuracy of 0.85, demonstrating competitive and well-balanced performance across metrics.
>
---
#### [new 073] sam-llm: interpretable lane change trajectoryprediction via parametric finetuning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SAM-LLM模型，通过参数化微调结合LLM与物理模型，解决车道变更轨迹预测的可解释性与计算效率问题，实现高精度且连续的轨迹生成。**

- **链接: [http://arxiv.org/pdf/2509.03462v1](http://arxiv.org/pdf/2509.03462v1)**

> **作者:** Zhuo Cao; Yunxiao Shi; Min Xu
>
> **备注:** 5 pages
>
> **摘要:** This work introduces SAM-LLM, a novel hybrid architecture that bridges the gap between the contextual reasoning of Large Language Models (LLMs) and the physical precision of kinematic lane change models for autonomous driving. The system is designed for interpretable lane change trajectory prediction by finetuning an LLM to output the core physical parameters of a trajectory model instead of raw coordinates. For lane-keeping scenarios, the model predicts discrete coordinates, but for lane change maneuvers, it generates the parameters for an enhanced Sinusoidal Acceleration Model (SAM), including lateral displacement, maneuver duration, initial lateral velocity, and longitudinal velocity change. This parametric approach yields a complete, continuous, and physically plausible trajectory model that is inherently interpretable and computationally efficient, achieving an 80% reduction in output size compared to coordinate-based methods. The SAM-LLM achieves a state-of-the-art overall intention prediction accuracy of 98.73%, demonstrating performance equivalent to traditional LLM predictors while offering significant advantages in explainability and resource efficiency.
>
---
#### [new 074] Uncertainty-aware Test-Time Training (UT$^3$) for Efficient On-the-fly Domain Adaptive Dense Regression
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UT³框架，用于领域自适应的密集回归（如单目深度估计），通过不确定性感知的自监督减少推理延迟，提升实时应用效率。**

- **链接: [http://arxiv.org/pdf/2509.03012v1](http://arxiv.org/pdf/2509.03012v1)**

> **作者:** Uddeshya Upadhyay
>
> **摘要:** Deep neural networks (DNNs) are increasingly being used in autonomous systems. However, DNNs do not generalize well to domain shift. Adapting to a continuously evolving environment is a safety-critical challenge inevitably faced by all autonomous systems deployed to the real world. Recent work on test-time training proposes methods that adapt to a new test distribution on the fly by optimizing the DNN model for each test input using self-supervision. However, these techniques result in a sharp increase in inference time as multiple forward and backward passes are required for a single test sample (for test-time training) before finally making the prediction based on the fine-tuned features. This is undesirable for real-world robotics applications where these models may be deployed to resource constraint hardware with strong latency requirements. In this work, we propose a new framework (called UT$^3$) that leverages test-time training for improved performance in the presence of continuous domain shift while also decreasing the inference time, making it suitable for real-world applications. Our method proposes an uncertainty-aware self-supervision task for efficient test-time training that leverages the quantified uncertainty to selectively apply the training leading to sharp improvements in the inference time while performing comparably to standard test-time training protocol. Our proposed protocol offers a continuous setting to identify the selected keyframes, allowing the end-user to control how often to apply test-time training. We demonstrate the efficacy of our method on a dense regression task - monocular depth estimation.
>
---
#### [new 075] Toward a robust lesion detection model in breast DCE-MRI: adapting foundation models to high-risk women
- **分类: physics.med-ph; cs.CV; cs.LG**

- **简介: 该论文提出基于MST和KAN的乳腺MRI病变分类方法，针对高危人群提升良恶性区分准确率，通过自监督预训练和可解释模型实现AUC 0.80。**

- **链接: [http://arxiv.org/pdf/2509.02710v1](http://arxiv.org/pdf/2509.02710v1)**

> **作者:** Gabriel A. B. do Nascimento; Vincent Dong; Guilherme J. Cavalcante; Alex Nguyen; Thaís G. do Rêgo; Yuri Malheiros; Telmo M. Silva Filho; Carla R. Zeballos Torrez; James C. Gee; Anne Marie McCarthy; Andrew D. A. Maidment; Bruno Barufaldi
>
> **摘要:** Accurate breast MRI lesion detection is critical for early cancer diagnosis, especially in high-risk populations. We present a classification pipeline that adapts a pretrained foundation model, the Medical Slice Transformer (MST), for breast lesion classification using dynamic contrast-enhanced MRI (DCE-MRI). Leveraging DINOv2-based self-supervised pretraining, MST generates robust per-slice feature embeddings, which are then used to train a Kolmogorov--Arnold Network (KAN) classifier. The KAN provides a flexible and interpretable alternative to conventional convolutional networks by enabling localized nonlinear transformations via adaptive B-spline activations. This enhances the model's ability to differentiate benign from malignant lesions in imbalanced and heterogeneous clinical datasets. Experimental results demonstrate that the MST+KAN pipeline outperforms the baseline MST classifier, achieving AUC = 0.80 \pm 0.02 while preserving interpretability through attention-based heatmaps. Our findings highlight the effectiveness of combining foundation model embeddings with advanced classification strategies for building robust and generalizable breast MRI analysis tools.
>
---
#### [new 076] Generalist versus Specialist Vision Foundation Models for Ocular Disease and Oculomics
- **分类: eess.IV; cs.CV; J.3; I.2.10**

- **简介: 该论文比较通用与专用视觉基础模型在眼科疾病检测中的表现，评估其适应性及效率，发现专用模型在准确性和数据效率上更优，但通用模型差距缩小，提示领域扩展可能提升其临床应用价值。**

- **链接: [http://arxiv.org/pdf/2509.03421v1](http://arxiv.org/pdf/2509.03421v1)**

> **作者:** Yukun Zhou; Paul Nderitu; Jocelyn Hui Lin Goh; Justin Engelmann; Siegfried K. Wagner; Anran Ran; Hongyang Jiang; Lie Ju; Ke Zou; Sahana Srinivasan; Hyunmin Kim; Takahiro Ninomiya; Zheyuan Wang; Gabriel Dawei Yang; Eden Ruffell; Dominic Williamson; Rui Santos; Gabor Mark Somfai; Carol Y. Cheung; Tien Yin Wong; Daniel C. Alexander; Yih Chung Tham; Pearse A. Keane
>
> **备注:** 39 pages, 8 Figures
>
> **摘要:** Medical foundation models, pre-trained with large-scale clinical data, demonstrate strong performance in diverse clinically relevant applications. RETFound, trained on nearly one million retinal images, exemplifies this approach in applications with retinal images. However, the emergence of increasingly powerful and multifold larger generalist foundation models such as DINOv2 and DINOv3 raises the question of whether domain-specific pre-training remains essential, and if so, what gap persists. To investigate this, we systematically evaluated the adaptability of DINOv2 and DINOv3 in retinal image applications, compared to two specialist RETFound models, RETFound-MAE and RETFound-DINOv2. We assessed performance on ocular disease detection and systemic disease prediction using two adaptation strategies: fine-tuning and linear probing. Data efficiency and adaptation efficiency were further analysed to characterise trade-offs between predictive performance and computational cost. Our results show that although scaling generalist models yields strong adaptability across diverse tasks, RETFound-DINOv2 consistently outperforms these generalist foundation models in ocular-disease detection and oculomics tasks, demonstrating stronger generalisability and data efficiency. These findings suggest that specialist retinal foundation models remain the most effective choice for clinical applications, while the narrowing gap with generalist foundation models suggests that continued data and model scaling can deliver domain-relevant gains and position them as strong foundations for future medical foundation models.
>
---
#### [new 077] Robult: Leveraging Redundancy and Modality Specific Features for Robust Multimodal Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对多模态学习中缺失模态和小样本问题，提出Robult框架。通过结合模态特异性保留与冗余信息利用，设计软PU对比损失和潜在重建损失，提升模型鲁棒性与半监督性能，适用于实际多模态场景。**

- **链接: [http://arxiv.org/pdf/2509.03477v1](http://arxiv.org/pdf/2509.03477v1)**

> **作者:** Duy A. Nguyen; Abhi Kamboj; Minh N. Do
>
> **备注:** Accepted and presented at IJCAI 2025 in Montreal, Canada
>
> **摘要:** Addressing missing modalities and limited labeled data is crucial for advancing robust multimodal learning. We propose Robult, a scalable framework designed to mitigate these challenges by preserving modality-specific information and leveraging redundancy through a novel information-theoretic approach. Robult optimizes two core objectives: (1) a soft Positive-Unlabeled (PU) contrastive loss that maximizes task-relevant feature alignment while effectively utilizing limited labeled data in semi-supervised settings, and (2) a latent reconstruction loss that ensures unique modality-specific information is retained. These strategies, embedded within a modular design, enhance performance across various downstream tasks and ensure resilience to incomplete modalities during inference. Experimental results across diverse datasets validate that Robult achieves superior performance over existing approaches in both semi-supervised learning and missing modality contexts. Furthermore, its lightweight design promotes scalability and seamless integration with existing architectures, making it suitable for real-world multimodal applications.
>
---
#### [new 078] ProMQA-Assembly: Multimodal Procedural QA Dataset on Assembly
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出ProMQA-Assembly数据集，用于多模态装配任务问答，解决缺乏实际应用场景评估的问题。通过半自动标注与任务图构建，提供基准测试，推动程序化助手发展。**

- **链接: [http://arxiv.org/pdf/2509.02949v1](http://arxiv.org/pdf/2509.02949v1)**

> **作者:** Kimihiro Hasegawa; Wiradee Imrattanatrai; Masaki Asada; Susan Holm; Yuran Wang; Vincent Zhou; Ken Fukuda; Teruko Mitamura
>
> **备注:** 29 pages. Code and data: https://github.com/kimihiroh/promqa-assembly
>
> **摘要:** Assistants on assembly tasks have a large potential to benefit humans from everyday tasks to industrial settings. However, no testbeds support application-oriented system evaluation in a practical setting, especially in assembly. To foster the development, we propose a new multimodal QA dataset on assembly activities. Our dataset, ProMQA-Assembly, consists of 391 QA pairs that require the multimodal understanding of human-activity recordings and their instruction manuals in an online-style manner. In the development, we adopt a semi-automated QA annotation approach, where LLMs generate candidates and humans verify them, as a cost-effective method, and further improve it by integrating fine-grained action labels to diversify question types. Furthermore, we create instruction task graphs for the target tasks of assembling toy vehicles. These newly created task graphs are used in our benchmarking experiment, as well as to facilitate the human verification process in the QA annotation. Utilizing our dataset, we benchmark models, including competitive proprietary multimodal models. Our results suggest great room for improvement for the current models. We believe our new evaluation dataset can contribute to the further development of procedural-activity assistants.
>
---
#### [new 079] DUViN: Diffusion-Based Underwater Visual Navigation via Knowledge-Transferred Depth Features
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DUViN方法，解决水下自主导航中无地图、障碍避障及地形高度保持问题。通过知识迁移的深度特征与扩散模型，实现端到端4-DoF控制，分阶段训练应对域迁移挑战，验证了模拟与真实环境下的有效性。**

- **链接: [http://arxiv.org/pdf/2509.02983v1](http://arxiv.org/pdf/2509.02983v1)**

> **作者:** Jinghe Yang; Minh-Quan Le; Mingming Gong; Ye Pu
>
> **摘要:** Autonomous underwater navigation remains a challenging problem due to limited sensing capabilities and the difficulty of constructing accurate maps in underwater environments. In this paper, we propose a Diffusion-based Underwater Visual Navigation policy via knowledge-transferred depth features, named DUViN, which enables vision-based end-to-end 4-DoF motion control for underwater vehicles in unknown environments. DUViN guides the vehicle to avoid obstacles and maintain a safe and perception awareness altitude relative to the terrain without relying on pre-built maps. To address the difficulty of collecting large-scale underwater navigation datasets, we propose a method that ensures robust generalization under domain shifts from in-air to underwater environments by leveraging depth features and introducing a novel model transfer strategy. Specifically, our training framework consists of two phases: we first train the diffusion-based visual navigation policy on in-air datasets using a pre-trained depth feature extractor. Secondly, we retrain the extractor on an underwater depth estimation task and integrate the adapted extractor into the trained navigation policy from the first step. Experiments in both simulated and real-world underwater environments demonstrate the effectiveness and generalization of our approach. The experimental videos are available at https://www.youtube.com/playlist?list=PLqt2s-RyCf1gfXJgFzKjmwIqYhrP4I-7Y.
>
---
#### [new 080] MitoDetect++: A Domain-Robust Pipeline for Mitosis Detection and Atypical Subtyping
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出MitoDetect++，解决计算病理学中自动检测和分类有丝分裂图（区分正常与非典型）的任务。采用U-Net与ViT架构，结合注意力模块、LoRA微调及数据增强，实现跨域鲁棒性，验证域平衡准确率达0.892。**

- **链接: [http://arxiv.org/pdf/2509.02586v1](http://arxiv.org/pdf/2509.02586v1)**

> **作者:** Esha Sadia Nasir; Jiaqi Lv; Mostafa Jahanifer; Shan E Ahmed Raza
>
> **摘要:** Automated detection and classification of mitotic figures especially distinguishing atypical from normal remain critical challenges in computational pathology. We present MitoDetect++, a unified deep learning pipeline designed for the MIDOG 2025 challenge, addressing both mitosis detection and atypical mitosis classification. For detection (Track 1), we employ a U-Net-based encoder-decoder architecture with EfficientNetV2-L as the backbone, enhanced with attention modules, and trained via combined segmentation losses. For classification (Track 2), we leverage the Virchow2 vision transformer, fine-tuned efficiently using Low-Rank Adaptation (LoRA) to minimize resource consumption. To improve generalization and mitigate domain shifts, we integrate strong augmentations, focal loss, and group-aware stratified 5-fold cross-validation. At inference, we deploy test-time augmentation (TTA) to boost robustness. Our method achieves a balanced accuracy of 0.892 across validation domains, highlighting its clinical applicability and scalability across tasks.
>
---
#### [new 081] A Single Detect Focused YOLO Framework for Robust Mitotic Figure Detection
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出SDF-YOLO框架，针对计算病理学中的有丝分裂图检测任务，解决域变化导致的检测不鲁棒问题。通过单检测头、坐标注意力和跨通道特征混合改进YOLOv11，实现高准确性和效率。**

- **链接: [http://arxiv.org/pdf/2509.02637v1](http://arxiv.org/pdf/2509.02637v1)**

> **作者:** Yasemin Topuz; M. Taha Gökcan; Serdar Yıldız; Songül Varlı
>
> **摘要:** Mitotic figure detection is a crucial task in computational pathology, as mitotic activity serves as a strong prognostic marker for tumor aggressiveness. However, domain variability that arises from differences in scanners, tissue types, and staining protocols poses a major challenge to the robustness of automated detection methods. In this study, we introduce SDF-YOLO (Single Detect Focused YOLO), a lightweight yet domain-robust detection framework designed specifically for small, rare targets such as mitotic figures. The model builds on YOLOv11 with task-specific modifications, including a single detection head aligned with mitotic figure scale, coordinate attention to enhance positional sensitivity, and improved cross-channel feature mixing. Experiments were conducted on three datasets that span human and canine tumors: MIDOG ++, canine cutaneous mast cell tumor (CCMCT), and canine mammary carcinoma (CMC). When submitted to the preliminary test set for the MIDOG2025 challenge, SDF-YOLO achieved an average precision (AP) of 0.799, with a precision of 0.758, a recall of 0.775, an F1 score of 0.766, and an FROC-AUC of 5.793, demonstrating both competitive accuracy and computational efficiency. These results indicate that SDF-YOLO provides a reliable and efficient framework for robust mitotic figure detection across diverse domains.
>
---
#### [new 082] Pan-Cancer mitotic figures detection and domain generalization: MIDOG 2025 Challenge
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对癌症病理切片中分裂图检测任务，通过发布新数据集和采用最新训练方法，解决跨领域泛化问题，实现Track-1 F1-Score 0.8407和Track-2平衡准确率0.9107。**

- **链接: [http://arxiv.org/pdf/2509.02585v1](http://arxiv.org/pdf/2509.02585v1)**

> **作者:** Zhuoyan Shen; Esther Bär; Maria Hawkins; Konstantin Bräutigam; Charles-Antoine Collins-Fekete
>
> **摘要:** This report details our submission to the Mitotic Domain Generalization (MIDOG) 2025 challenge, which addresses the critical task of mitotic figure detection in histopathology for cancer prognostication. Following the "Bitter Lesson"\cite{sutton2019bitterlesson} principle that emphasizes data scale over algorithmic novelty, we have publicly released two new datasets to bolster training data for both conventional \cite{Shen2024framework} and atypical mitoses \cite{shen_2025_16780587}. Besides, we implement up-to-date training methodologies for both track and reach a Track-1 F1-Score of 0.8407 on our test set, as well as a Track-2 balanced accuracy of 0.9107 for atypical mitotic cell classification.
>
---
#### [new 083] Team Westwood Solution for MIDOG 2025 Challenge
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对MIDOG 2025挑战赛的有丝分裂检测与异常分类任务，提出结合nnUNetV2与多CNN模型（EfficientNet系列、InceptionV3）的随机森林集成方案，实现高灵敏度检测与分类准确率。**

- **链接: [http://arxiv.org/pdf/2509.02600v1](http://arxiv.org/pdf/2509.02600v1)**

> **作者:** Tengyou Xu; Haochen Yang; Xiang 'Anthony' Chen; Hongyan Gu; Mohammad Haeri
>
> **备注:** 2 pages, 2 figures
>
> **摘要:** This abstract presents our solution (Team Westwood) for mitosis detection and atypical mitosis classification in the MItosis DOmain Generalization (MIDOG) 2025 challenge. For mitosis detection, we trained an nnUNetV2 for initial mitosis candidate screening with high sensitivity, followed by a random forest classifier ensembling predictions of three convolutional neural networks (CNNs): EfficientNet-b3, EfficientNet-b5, and EfficientNetV2-s. For the atypical mitosis classification, we trained another random forest classifier ensembling the predictions of three CNNs: EfficientNet-b3, EfficientNet-b5, and InceptionV3. On the preliminary test set, our solution achieved an F1 score of 0.7450 for track 1 mitosis detection, and a balanced accuracy of 0.8722 for track 2 atypical mitosis classification.
>
---
#### [new 084] Sequential Hard Mining: a data-centric approach for Mitosis Detection
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于数据中心的Sequential Hard Mining方法，用于有丝分裂检测。针对大规模标注数据训练挑战，结合提升技术高效采样，提出候选方案解决MIDOG 2025挑战的两个赛道问题。**

- **链接: [http://arxiv.org/pdf/2509.02588v1](http://arxiv.org/pdf/2509.02588v1)**

> **作者:** Maxime W. Lafarge; Viktor H. Koelzer
>
> **摘要:** With a continuously growing availability of annotated datasets of mitotic figures in histology images, finding the best way to optimally use with this unprecedented amount of data to optimally train deep learning models has become a new challenge. Here, we build upon previously proposed approaches with a focus on efficient sampling of training data inspired by boosting techniques and present our candidate solutions for the two tracks of the MIDOG 2025 challenge.
>
---
#### [new 085] Robust Pan-Cancer Mitotic Figure Detection with YOLOv12
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出基于YOLOv12的胰腺癌分裂图检测方法，解决病理图像中分裂图识别困难、观察者差异大的问题，在MIDOG 2025挑战中实现0.801 F1分数，无需外部数据。**

- **链接: [http://arxiv.org/pdf/2509.02593v1](http://arxiv.org/pdf/2509.02593v1)**

> **作者:** Raphaël Bourgade; Guillaume Balezo; Thomas Walter
>
> **摘要:** Mitotic figures represent a key histoprognostic feature in tumor pathology, providing crucial insights into tumor aggressiveness and proliferation. However, their identification remains challenging, subject to significant inter-observer variability, even among experienced pathologists. To address this issue, the MItosis DOmain Generalization (MIDOG) 2025 challenge marks the third edition of an international competition aiming to develop robust mitosis detection algorithms. In this paper, we present a mitotic figures detection approach based on the YOLOv12 object detection architecture, achieving a $F_1$-score of 0.801 on the preliminary test set of the MIDOG 2025 challenge, without relying on external data.
>
---
#### [new 086] Foundation Model-Driven Classification of Atypical Mitotic Figures with Domain-Aware Training Strategies
- **分类: eess.IV; cs.CV**

- **简介: 论文针对医学图像中的正常与非典型有丝分裂图二分类任务，提出基于H-optimus-0基础模型的解决方案，结合LoRA微调、MixUp增强及领域适应策略，提升分类准确性。**

- **链接: [http://arxiv.org/pdf/2509.02601v1](http://arxiv.org/pdf/2509.02601v1)**

> **作者:** Piotr Giedziun; Jan Sołtysik; Mateusz Górczany; Norbert Ropiak; Marcin Przymus; Piotr Krajewski; Jarosław Kwiecień; Artur Bartczak; Izabela Wasiak; Mateusz Maniewski
>
> **摘要:** We present a solution for the MIDOG 2025 Challenge Track~2, addressing binary classification of normal mitotic figures (NMFs) versus atypical mitotic figures (AMFs). The approach leverages pathology-specific foundation model H-optimus-0, selected based on recent cross-domain generalization benchmarks and our empirical testing, with Low-Rank Adaptation (LoRA) fine-tuning and MixUp augmentation. Implementation includes soft labels based on multi-expert consensus, hard negative mining, and adaptive focal loss, metric learning and domain adaptation. The method demonstrates both the promise and challenges of applying foundation models to this complex classification task, achieving reasonable performance in the preliminary evaluation phase.
>
---
#### [new 087] SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data
- **分类: cs.HC; cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出一种基于智能手机与智能手表的臂部姿态估计方法，利用UWB测距与IMU惯性数据互补，解决现有方案的隐私问题及设备复杂性，实现无需训练数据的高精度姿态追踪（中位误差11cm）。**

- **链接: [http://arxiv.org/pdf/2509.03451v1](http://arxiv.org/pdf/2509.03451v1)**

> **作者:** Nathan DeVrio; Vimal Mollyn; Chris Harrison
>
> **备注:** The first two listed authors contributed equally. Published at UIST 2023
>
> **摘要:** The ability to track a user's arm pose could be valuable in a wide range of applications, including fitness, rehabilitation, augmented reality input, life logging, and context-aware assistants. Unfortunately, this capability is not readily available to consumers. Systems either require cameras, which carry privacy issues, or utilize multiple worn IMUs or markers. In this work, we describe how an off-the-shelf smartphone and smartwatch can work together to accurately estimate arm pose. Moving beyond prior work, we take advantage of more recent ultra-wideband (UWB) functionality on these devices to capture absolute distance between the two devices. This measurement is the perfect complement to inertial data, which is relative and suffers from drift. We quantify the performance of our software-only approach using off-the-shelf devices, showing it can estimate the wrist and elbow joints with a \hl{median positional error of 11.0~cm}, without the user having to provide training data.
>
---
## 更新

#### [replaced 001] Automated Parsing of Engineering Drawings for Structured Information Extraction Using a Fine-tuned Document Understanding Transformer
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.01530v3](http://arxiv.org/pdf/2505.01530v3)**

> **作者:** Muhammad Tayyab Khan; Zane Yong; Lequn Chen; Jun Ming Tan; Wenhe Feng; Seung Ki Moon
>
> **备注:** This manuscript has been accepted for publication at IEEE International Conference on Industrial Engineering and Engineering Management (IEEM)
>
> **摘要:** Accurate extraction of key information from 2D engineering drawings is crucial for high-precision manufacturing. Manual extraction is slow and labor-intensive, while traditional Optical Character Recognition (OCR) techniques often struggle with complex layouts and overlapping symbols, resulting in unstructured outputs. To address these challenges, this paper proposes a novel hybrid deep learning framework for structured information extraction by integrating an Oriented Bounding Box (OBB) detection model with a transformer-based document parsing model (Donut). An in-house annotated dataset is used to train YOLOv11 for detecting nine key categories: Geometric Dimensioning and Tolerancing (GD&T), General Tolerances, Measures, Materials, Notes, Radii, Surface Roughness, Threads, and Title Blocks. Detected OBBs are cropped into images and labeled to fine-tune Donut for structured JSON output. Fine-tuning strategies include a single model trained across all categories and category-specific models. Results show that the single model consistently outperforms category-specific ones across all evaluation metrics, achieving higher precision (94.77% for GD&T), recall (100% for most categories), and F1 score (97.3%), while reducing hallucinations (5.23%). The proposed framework improves accuracy, reduces manual effort, and supports scalable deployment in precision-driven industries.
>
---
#### [replaced 002] Enhancing Diffusion Model Stability for Image Restoration via Gradient Management
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.06656v2](http://arxiv.org/pdf/2507.06656v2)**

> **作者:** Hongjie Wu; Mingqin Zhang; Linchao He; Ji-Zhe Zhou; Jiancheng Lv
>
> **备注:** Accepted to ACM Multimedia 2025
>
> **摘要:** Diffusion models have shown remarkable promise for image restoration by leveraging powerful priors. Prominent methods typically frame the restoration problem within a Bayesian inference framework, which iteratively combines a denoising step with a likelihood guidance step. However, the interactions between these two components in the generation process remain underexplored. In this paper, we analyze the underlying gradient dynamics of these components and identify significant instabilities. Specifically, we demonstrate conflicts between the prior and likelihood gradient directions, alongside temporal fluctuations in the likelihood gradient itself. We show that these instabilities disrupt the generative process and compromise restoration performance. To address these issues, we propose Stabilized Progressive Gradient Diffusion (SPGD), a novel gradient management technique. SPGD integrates two synergistic components: (1) a progressive likelihood warm-up strategy to mitigate gradient conflicts; and (2) adaptive directional momentum (ADM) smoothing to reduce fluctuations in the likelihood gradient. Extensive experiments across diverse restoration tasks demonstrate that SPGD significantly enhances generation stability, leading to state-of-the-art performance in quantitative metrics and visually superior results. Code is available at https://github.com/74587887/SPGD.
>
---
#### [replaced 003] GS-TG: 3D Gaussian Splatting Accelerator with Tile Grouping for Reducing Redundant Sorting while Preserving Rasterization Efficiency
- **分类: cs.AR; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00911v2](http://arxiv.org/pdf/2509.00911v2)**

> **作者:** Joongho Jo; Jongsun Park
>
> **备注:** DAC 2025
>
> **摘要:** 3D Gaussian Splatting (3D-GS) has emerged as a promising alternative to neural radiance fields (NeRF) as it offers high speed as well as high image quality in novel view synthesis. Despite these advancements, 3D-GS still struggles to meet the frames per second (FPS) demands of real-time applications. In this paper, we introduce GS-TG, a tile-grouping-based accelerator that enhances 3D-GS rendering speed by reducing redundant sorting operations and preserving rasterization efficiency. GS-TG addresses a critical trade-off issue in 3D-GS rendering: increasing the tile size effectively reduces redundant sorting operations, but it concurrently increases unnecessary rasterization computations. So, during sorting of the proposed approach, GS-TG groups small tiles (for making large tiles) to share sorting operations across tiles within each group, significantly reducing redundant computations. During rasterization, a bitmask assigned to each Gaussian identifies relevant small tiles, to enable efficient sharing of sorting results. Consequently, GS-TG enables sorting to be performed as if a large tile size is used by grouping tiles during the sorting stage, while allowing rasterization to proceed with the original small tiles by using bitmasks in the rasterization stage. GS-TG is a lossless method requiring no retraining or fine-tuning and it can be seamlessly integrated with previous 3D-GS optimization techniques. Experimental results show that GS-TG achieves an average speed-up of 1.54 times over state-of-the-art 3D-GS accelerators.
>
---
#### [replaced 004] Performance is not All You Need: Sustainability Considerations for Algorithms
- **分类: cs.CV; cs.PF**

- **链接: [http://arxiv.org/pdf/2509.00045v2](http://arxiv.org/pdf/2509.00045v2)**

> **作者:** Xiang Li; Chong Zhang; Hongpeng Wang; Shreyank Narayana Gowda; Yushi Li; Xiaobo Jin
>
> **备注:** 18 pages, 6 figures. Accepted Chinese Conference on Pattern Recognition and Computer Vision 2025
>
> **摘要:** This work focuses on the high carbon emissions generated by deep learning model training, specifically addressing the core challenge of balancing algorithm performance and energy consumption. It proposes an innovative two-dimensional sustainability evaluation system. Different from the traditional single performance-oriented evaluation paradigm, this study pioneered two quantitative indicators that integrate energy efficiency ratio and accuracy: the sustainable harmonic mean (FMS) integrates accumulated energy consumption and performance parameters through the harmonic mean to reveal the algorithm performance under unit energy consumption; the area under the sustainability curve (ASC) constructs a performance-power consumption curve to characterize the energy efficiency characteristics of the algorithm throughout the cycle. To verify the universality of the indicator system, the study constructed benchmarks in various multimodal tasks, including image classification, segmentation, pose estimation, and batch and online learning. Experiments demonstrate that the system can provide a quantitative basis for evaluating cross-task algorithms and promote the transition of green AI research from theory to practice. Our sustainability evaluation framework code can be found here, providing methodological support for the industry to establish algorithm energy efficiency standards.
>
---
#### [replaced 005] UPGS: Unified Pose-aware Gaussian Splatting for Dynamic Scene Deblurring
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00831v2](http://arxiv.org/pdf/2509.00831v2)**

> **作者:** Zhijing Wu; Longguang Wang
>
> **摘要:** Reconstructing dynamic 3D scenes from monocular video has broad applications in AR/VR, robotics, and autonomous navigation, but often fails due to severe motion blur caused by camera and object motion. Existing methods commonly follow a two-step pipeline, where camera poses are first estimated and then 3D Gaussians are optimized. Since blurring artifacts usually undermine pose estimation, pose errors could be accumulated to produce inferior reconstruction results. To address this issue, we introduce a unified optimization framework by incorporating camera poses as learnable parameters complementary to 3DGS attributes for end-to-end optimization. Specifically, we recast camera and object motion as per-primitive SE(3) affine transformations on 3D Gaussians and formulate a unified optimization objective. For stable optimization, we introduce a three-stage training schedule that optimizes camera poses and Gaussians alternatively. Particularly, 3D Gaussians are first trained with poses being fixed, and then poses are optimized with 3D Gaussians being untouched. Finally, all learnable parameters are optimized together. Extensive experiments on the Stereo Blur dataset and challenging real-world sequences demonstrate that our method achieves significant gains in reconstruction quality and pose estimation accuracy over prior dynamic deblurring methods.
>
---
#### [replaced 006] A Coarse-to-Fine Approach to Multi-Modality 3D Occupancy Grounding
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.01197v2](http://arxiv.org/pdf/2508.01197v2)**

> **作者:** Zhan Shi; Song Wang; Junbo Chen; Jianke Zhu
>
> **摘要:** Visual grounding aims to identify objects or regions in a scene based on natural language descriptions, essential for spatially aware perception in autonomous driving. However, existing visual grounding tasks typically depend on bounding boxes that often fail to capture fine-grained details. Not all voxels within a bounding box are occupied, resulting in inaccurate object representations. To address this, we introduce a benchmark for 3D occupancy grounding in challenging outdoor scenes. Built on the nuScenes dataset, it integrates natural language with voxel-level occupancy annotations, offering more precise object perception compared to the traditional grounding task. Moreover, we propose GroundingOcc, an end-to-end model designed for 3D occupancy grounding through multi-modal learning. It combines visual, textual, and point cloud features to predict object location and occupancy information from coarse to fine. Specifically, GroundingOcc comprises a multimodal encoder for feature extraction, an occupancy head for voxel-wise predictions, and a grounding head to refine localization. Additionally, a 2D grounding module and a depth estimation module enhance geometric understanding, thereby boosting model performance. Extensive experiments on the benchmark demonstrate that our method outperforms existing baselines on 3D occupancy grounding. The dataset is available at https://github.com/RONINGOD/GroundingOcc.
>
---
#### [replaced 007] MSA2-Net: Utilizing Self-Adaptive Convolution Module to Extract Multi-Scale Information in Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.01498v2](http://arxiv.org/pdf/2509.01498v2)**

> **作者:** Chao Deng; Xiaosen Li; Xiao Qin
>
> **摘要:** The nnUNet segmentation framework adeptly adjusts most hyperparameters in training scripts automatically, but it overlooks the tuning of internal hyperparameters within the segmentation network itself, which constrains the model's ability to generalize. Addressing this limitation, this study presents a novel Self-Adaptive Convolution Module that dynamically adjusts the size of the convolution kernels depending on the unique fingerprints of different datasets. This adjustment enables the MSA2-Net, when equipped with this module, to proficiently capture both global and local features within the feature maps. Self-Adaptive Convolution Module is strategically integrated into two key components of the MSA2-Net: the Multi-Scale Convolution Bridge and the Multi-Scale Amalgamation Decoder. In the MSConvBridge, the module enhances the ability to refine outputs from various stages of the CSWin Transformer during the skip connections, effectively eliminating redundant data that could potentially impair the decoder's performance. Simultaneously, the MSADecoder, utilizing the module, excels in capturing detailed information of organs varying in size during the decoding phase. This capability ensures that the decoder's output closely reproduces the intricate details within the feature maps, thus yielding highly accurate segmentation images. MSA2-Net, bolstered by this advanced architecture, has demonstrated exceptional performance, achieving Dice coefficient scores of 86.49\%, 92.56\%, 93.37\%, and 92.98\% on the Synapse, ACDC, Kvasir, and Skin Lesion Segmentation (ISIC2017) datasets, respectively. This underscores MSA2-Net's robustness and precision in medical image segmentation tasks across various datasets.
>
---
#### [replaced 008] Sequential keypoint density estimator: an overlooked baseline of skeleton-based video anomaly detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18368v3](http://arxiv.org/pdf/2506.18368v3)**

> **作者:** Anja Delić; Matej Grcić; Siniša Šegvić
>
> **备注:** ICCV 2025 Highlight
>
> **摘要:** Detecting anomalous human behaviour is an important visual task in safety-critical applications such as healthcare monitoring, workplace safety, or public surveillance. In these contexts, abnormalities are often reflected with unusual human poses. Thus, we propose SeeKer, a method for detecting anomalies in sequences of human skeletons. Our method formulates the skeleton sequence density through autoregressive factorization at the keypoint level. The corresponding conditional distributions represent probable keypoint locations given prior skeletal motion. We formulate the joint distribution of the considered skeleton as causal prediction of conditional Gaussians across its constituent keypoints. A skeleton is flagged as anomalous if its keypoint locations surprise our model (i.e. receive a low density). In practice, our anomaly score is a weighted sum of per-keypoint log-conditionals, where the weights account for the confidence of the underlying keypoint detector. Despite its conceptual simplicity, SeeKer surpasses all previous methods on the UBnormal and MSAD-HR datasets while delivering competitive performance on the ShanghaiTech dataset.
>
---
#### [replaced 009] Novel Category Discovery with X-Agent Attention for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01275v2](http://arxiv.org/pdf/2509.01275v2)**

> **作者:** Jiahao Li; Yang Lu; Yachao Zhang; Fangyong Wang; Yuan Xie; Yanyun Qu
>
> **备注:** Accepted by ACMMM2025
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) conducts pixel-level classification via text-driven alignment, where the domain discrepancy between base category training and open-vocabulary inference poses challenges in discriminative modeling of latent unseen category. To address this challenge, existing vision-language model (VLM)-based approaches demonstrate commendable performance through pre-trained multi-modal representations. However, the fundamental mechanisms of latent semantic comprehension remain underexplored, making the bottleneck for OVSS. In this work, we initiate a probing experiment to explore distribution patterns and dynamics of latent semantics in VLMs under inductive learning paradigms. Building on these insights, we propose X-Agent, an innovative OVSS framework employing latent semantic-aware ``agent'' to orchestrate cross-modal attention mechanisms, simultaneously optimizing latent semantic dynamic and amplifying its perceptibility. Extensive benchmark evaluations demonstrate that X-Agent achieves state-of-the-art performance while effectively enhancing the latent semantic saliency.
>
---
#### [replaced 010] On the representation of stack operators by mathematical morphology
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09766v2](http://arxiv.org/pdf/2504.09766v2)**

> **作者:** Diego Marcondes
>
> **摘要:** This paper introduces the class of grey-scale image stack operators as those that (a) map binary-images into binary-images and (b) commute on average with cross-sectioning. Equivalently, stack operators are 1-Lipchitz extensions of set operators which can be represented by applying a characteristic set operator to the cross-sections of the image and adding. In particular, they are a generalisation of stack filters, for which the characteristic set operators are increasing. Our main result is that stack operators inherit lattice properties of the characteristic set operators. We focus on the case of translation-invariant and locally defined stack operators and show the main result by deducing the characteristic function, kernel, and basis representation of stack operators. The results of this paper have implications on the design of image operators, since imply that to solve some grey-scale image processing problems it is enough to design an operator for performing the desired transformation on binary images, and then considering its extension given by a stack operator. We leave many topics for future research regarding the machine learning of stack operators and the characterisation of the image processing problems that can be solved by them.
>
---
#### [replaced 011] LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.12615v2](http://arxiv.org/pdf/2503.12615v2)**

> **作者:** Alessio Spagnoletti; Jean Prost; Andrés Almansa; Nicolas Papadakis; Marcelo Pereyra
>
> **备注:** 27 pages, 24 figures, International Conference on Computer Vision, ICCV 2025
>
> **摘要:** Text-to-image latent diffusion models (LDMs) have recently emerged as powerful generative models with great potential for solving inverse problems in imaging. However, leveraging such models in a Plug & Play (PnP), zero-shot manner remains challenging because it requires identifying a suitable text prompt for the unknown image of interest. Also, existing text-to-image PnP approaches are highly computationally expensive. We herein address these challenges by proposing a novel PnP inference paradigm specifically designed for embedding generative models within stochastic inverse solvers, with special attention to Latent Consistency Models (LCMs), which distill LDMs into fast generators. We leverage our framework to propose LAtent consisTency INverse sOlver (LATINO), the first zero-shot PnP framework to solve inverse problems with priors encoded by LCMs. Our conditioning mechanism avoids automatic differentiation and reaches SOTA quality in as little as 8 neural function evaluations. As a result, LATINO delivers remarkably accurate solutions and is significantly more memory and computationally efficient than previous approaches. We then embed LATINO within an empirical Bayesian framework that automatically calibrates the text prompt from the observed measurements by marginal maximum likelihood estimation. Extensive experiments show that prompt self-calibration greatly improves estimation, allowing LATINO with PRompt Optimization to define new SOTAs in image reconstruction quality and computational efficiency. The code is available at https://latino-pro.github.io
>
---
#### [replaced 012] Discrete Noise Inversion for Next-scale Autoregressive Text-based Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01984v2](http://arxiv.org/pdf/2509.01984v2)**

> **作者:** Quan Dao; Xiaoxiao He; Ligong Han; Ngan Hoai Nguyen; Amin Heyrani Nobar; Faez Ahmed; Han Zhang; Viet Anh Nguyen; Dimitris Metaxas
>
> **备注:** update affiliation
>
> **摘要:** Visual autoregressive models (VAR) have recently emerged as a promising class of generative models, achieving performance comparable to diffusion models in text-to-image generation tasks. While conditional generation has been widely explored, the ability to perform prompt-guided image editing without additional training is equally critical, as it supports numerous practical real-world applications. This paper investigates the text-to-image editing capabilities of VAR by introducing Visual AutoRegressive Inverse Noise (VARIN), the first noise inversion-based editing technique designed explicitly for VAR models. VARIN leverages a novel pseudo-inverse function for argmax sampling, named Location-aware Argmax Inversion (LAI), to generate inverse Gumbel noises. These inverse noises enable precise reconstruction of the source image and facilitate targeted, controllable edits aligned with textual prompts. Extensive experiments demonstrate that VARIN effectively modifies source images according to specified prompts while significantly preserving the original background and structural details, thus validating its efficacy as a practical editing approach.
>
---
#### [replaced 013] D2-Mamba: Dual-Scale Fusion and Dual-Path Scanning with SSMs for Shadow Removal
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.12750v2](http://arxiv.org/pdf/2508.12750v2)**

> **作者:** Linhao Li; Boya Jin; Zizhe Li; Lanqing Guo; Hao Cheng; Bo Li; Yongfeng Dong
>
> **备注:** Paper Under Review
>
> **摘要:** Shadow removal aims to restore images that are partially degraded by shadows, where the degradation is spatially localized and non-uniform. Unlike general restoration tasks that assume global degradation, shadow removal can leverage abundant information from non-shadow regions for guidance. However, the transformation required to correct shadowed areas often differs significantly from that of well-lit regions, making it challenging to apply uniform correction strategies. This necessitates the effective integration of non-local contextual cues and adaptive modeling of region-specific transformations. To this end, we propose a novel Mamba-based network featuring dual-scale fusion and dual-path scanning to selectively propagate contextual information based on transformation similarity across regions. Specifically, the proposed Dual-Scale Fusion Mamba Block (DFMB) enhances multi-scale feature representation by fusing original features with low-resolution features, effectively reducing boundary artifacts. The Dual-Path Mamba Group (DPMG) captures global features via horizontal scanning and incorporates a mask-aware adaptive scanning strategy, which improves structural continuity and fine-grained region modeling. Experimental results demonstrate that our method significantly outperforms existing state-of-the-art approaches on shadow removal benchmarks.
>
---
#### [replaced 014] Soft-TransFormers for Continual Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16073v2](http://arxiv.org/pdf/2411.16073v2)**

> **作者:** Haeyong Kang; Chang D. Yoo
>
> **摘要:** Inspired by the Well-initialized Lottery Ticket Hypothesis (WLTH), which provides suboptimal fine-tuning solutions, we propose a novel fully fine-tuned continual learning (CL) method referred to as Soft-TransFormers (Soft-TF). Soft-TF sequentially learns and selects an optimal soft-network for each task. During sequential training in CL, a well-initialized Soft-TF mask optimizes the weights of sparse layers to obtain task-adaptive soft (real-valued) networks, while keeping the well-pre-trained layer parameters frozen. In inference, the identified task-adaptive network of Soft-TF masks the parameters of the pre-trained network, mapping to an optimal solution for each task and minimizing Catastrophic Forgetting (CF) - the soft-masking preserves the knowledge of the pre-trained network. Extensive experiments on the Vision Transformer (ViT) and the Language Transformer (Bert) demonstrate the effectiveness of Soft-TF, achieving state-of-the-art performance across Vision and Language Class Incremental Learning (CIL) scenarios.
>
---
#### [replaced 015] GroundingDINO-US-SAM: Text-Prompted Multi-Organ Segmentation in Ultrasound with LoRA-Tuned Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23903v2](http://arxiv.org/pdf/2506.23903v2)**

> **作者:** Hamza Rasaee; Taha Koleilat; Hassan Rivaz
>
> **备注:** 11 pages, 3 figures, 7 tables
>
> **摘要:** Accurate and generalizable object segmentation in ultrasound imaging remains a significant challenge due to anatomical variability, diverse imaging protocols, and limited annotated data. In this study, we propose a prompt-driven vision-language model (VLM) that integrates Grounding DINO with SAM2 to enable object segmentation across multiple ultrasound organs. A total of 18 public ultrasound datasets, encompassing the breast, thyroid, liver, prostate, kidney, and paraspinal muscle, were utilized. These datasets were divided into 15 for fine-tuning and validation of Grounding DINO using Low Rank Adaptation (LoRA) to the ultrasound domain, and 3 were held out entirely for testing to evaluate performance in unseen distributions. Comprehensive experiments demonstrate that our approach outperforms state-of-the-art segmentation methods, including UniverSeg, MedSAM, MedCLIP-SAM, BiomedParse, and SAMUS on most seen datasets while maintaining strong performance on unseen datasets without additional fine-tuning. These results underscore the promise of VLMs in scalable and robust ultrasound image analysis, reducing dependence on large, organ-specific annotated datasets. We will publish our code on code.sonography.ai after acceptance.
>
---
#### [replaced 016] GalaxAlign: Mimicking Citizen Scientists' Multimodal Guidance for Galaxy Morphology Analysis
- **分类: cs.CV; astro-ph.GA; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19475v2](http://arxiv.org/pdf/2411.19475v2)**

> **作者:** Ruoqi Wang; Haitao Wang; Qiong Luo
>
> **备注:** ACM MM 2025
>
> **摘要:** Galaxy morphology analysis involves studying galaxies based on their shapes and structures. For such studies, fundamental tasks include identifying and classifying galaxies in astronomical images, as well as retrieving visually or structurally similar galaxies through similarity search. Existing methods either directly train domain-specific foundation models on large, annotated datasets or fine-tune vision foundation models on a smaller set of images. The former is effective but costly, while the latter is more resource-efficient but often yields lower accuracy. To address these challenges, we introduce GalaxAlign, a multimodal approach inspired by how citizen scientists identify galaxies in astronomical images by following textual descriptions and matching schematic symbols. Specifically, GalaxAlign employs a tri-modal alignment framework to align three types of data during fine-tuning: (1) schematic symbols representing galaxy shapes and structures, (2) textual labels for these symbols, and (3) galaxy images. By incorporating multimodal instructions, GalaxAlign eliminates the need for expensive pretraining and enhances the effectiveness of fine-tuning. Experiments on galaxy classification and similarity search demonstrate that our method effectively fine-tunes general pre-trained models for astronomical tasks by incorporating domain-specific multi-modal knowledge. Code is available at https://github.com/RapidsAtHKUST/GalaxAlign.
>
---
#### [replaced 017] Point Cloud Recombination: Systematic Real Data Augmentation Using Robotic Targets for LiDAR Perception Validation
- **分类: cs.RO; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.02476v2](http://arxiv.org/pdf/2505.02476v2)**

> **作者:** Hubert Padusinski; Christian Steinhauser; Christian Scherl; Julian Gaal; Jacob Langner
>
> **备注:** Pre-print for IEEE IAVVC 2025
>
> **摘要:** The validation of LiDAR-based perception of intelligent mobile systems operating in open-world applications remains a challenge due to the variability of real environmental conditions. Virtual simulations allow the generation of arbitrary scenes under controlled conditions but lack physical sensor characteristics, such as intensity responses or material-dependent effects. In contrast, real-world data offers true sensor realism but provides less control over influencing factors, hindering sufficient validation. Existing approaches address this problem with augmentation of real-world point cloud data by transferring objects between scenes. However, these methods do not consider validation and remain limited in controllability because they rely on empirical data. We solve these limitations by proposing Point Cloud Recombination, which systematically augments captured point cloud scenes by integrating point clouds acquired from physical target objects measured in controlled laboratory environments. Thus enabling the creation of vast amounts and varieties of repeatable, physically accurate test scenes with respect to phenomena-aware occlusions with registered 3D meshes. Using the Ouster OS1-128 Rev7 sensor, we demonstrate the augmentation of real-world urban and rural scenes with humanoid targets featuring varied clothing and poses, for repeatable positioning. We show that the recombined scenes closely match real sensor outputs, enabling targeted testing, scalable failure analysis, and improved system safety. By providing controlled yet sensor-realistic data, our method enables trustworthy conclusions about the limitations of specific sensors in compound with their algorithms, e.g., object detection.
>
---
#### [replaced 018] Open-Set LiDAR Panoptic Segmentation Guided by Uncertainty-Aware Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.13265v3](http://arxiv.org/pdf/2506.13265v3)**

> **作者:** Rohit Mohan; Julia Hindel; Florian Drews; Claudius Gläser; Daniele Cattaneo; Abhinav Valada
>
> **摘要:** Autonomous vehicles that navigate in open-world environments may encounter previously unseen object classes. However, most existing LiDAR panoptic segmentation models rely on closed-set assumptions, failing to detect unknown object instances. In this work, we propose ULOPS, an uncertainty-guided open-set panoptic segmentation framework that leverages Dirichlet-based evidential learning to model predictive uncertainty. Our architecture incorporates separate decoders for semantic segmentation with uncertainty estimation, embedding with prototype association, and instance center prediction. During inference, we leverage uncertainty estimates to identify and segment unknown instances. To strengthen the model's ability to differentiate between known and unknown objects, we introduce three uncertainty-driven loss functions. Uniform Evidence Loss to encourage high uncertainty in unknown regions. Adaptive Uncertainty Separation Loss ensures a consistent difference in uncertainty estimates between known and unknown objects at a global scale. Contrastive Uncertainty Loss refines this separation at the fine-grained level. To evaluate open-set performance, we extend benchmark settings on KITTI-360 and introduce a new open-set evaluation for nuScenes. Extensive experiments demonstrate that ULOPS consistently outperforms existing open-set LiDAR panoptic segmentation methods.
>
---
#### [replaced 019] Structure-preserving contrastive learning for spatial time series
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06380v4](http://arxiv.org/pdf/2502.06380v4)**

> **作者:** Yiru Jiao; Sander van Cranenburgh; Simeon Calvert; Hans van Lint
>
> **备注:** TL;DR: Preserving certain structures of similarity relations in spatio-temporal data can improve downstream task performance via contrastive learning
>
> **摘要:** The effectiveness of neural network models largely relies on learning meaningful latent patterns from data, where self-supervised learning of informative representations can enhance model performance and generalisability. However, self-supervised representation learning for spatially characterised time series, which are ubiquitous in transportation domain, poses unique challenges due to the necessity of maintaining fine-grained spatio-temporal similarities in the latent space. In this study, we introduce two structure-preserving regularisers for the contrastive learning of spatial time series: one regulariser preserves the topology of similarities between instances, and the other preserves the graph geometry of similarities across spatial and temporal dimensions. To balance the contrastive learning objective and the need for structure preservation, we propose a dynamic weighting mechanism that adaptively manages this trade-off and stabilises training. We validate the proposed method through extensive experiments, including multivariate time series classification to demonstrate its general applicability, as well as macroscopic and microscopic traffic prediction to highlight its particular usefulness in encoding traffic interactions. Across all tasks, our method preserves the similarity structures more effectively and improves state-of-the-art task performances. This method can be integrated with an arbitrary neural network model and is particularly beneficial for time series data with spatial or geographical features. Furthermore, our findings suggest that well-preserved similarity structures in the latent space indicate more informative and useful representations. This provides insights to design more effective neural networks for data-driven transportation research. Our code is made openly accessible with all resulting data at https://github.com/yiru-jiao/spclt
>
---
#### [replaced 020] AstroClearNet: Deep image prior for multi-frame astronomical image restoration
- **分类: astro-ph.IM; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06463v2](http://arxiv.org/pdf/2504.06463v2)**

> **作者:** Yashil Sukurdeep; Fausto Navarro; Tamás Budavári
>
> **摘要:** Recovering high-fidelity images of the night sky from blurred observations is a fundamental problem in astronomy, where traditional methods typically fall short. In ground-based astronomy, combining multiple exposures to enhance signal-to-noise ratios is further complicated by variations in the point-spread function caused by atmospheric turbulence. In this work, we present a self-supervised multi-frame method, based on deep image priors, for denoising, deblurring, and coadding ground-based exposures. Central to our approach is a carefully designed convolutional neural network that integrates information across multiple observations and enforces physically motivated constraints. We demonstrate the method's potential by processing Hyper Suprime-Cam exposures, yielding promising preliminary results with sharper restored images.
>
---
#### [replaced 021] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting
- **分类: cs.CV; cs.GR; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.00177v3](http://arxiv.org/pdf/2412.00177v3)**

> **作者:** Xiaoyan Xing; Konrad Groh; Sezer Karaoglu; Theo Gevers; Anand Bhattad
>
> **备注:** Corrects an evaluation bug in Table 1 due to a data normalization error. Thanks to the Sony PlayStation team for discovering and reporting the issue. The paper's core contributions, qualitative results, and user study are unaffected. We also include a minor update to the method to further improve result quality. Project page: https://luminet-relight.github.io/
>
> **摘要:** We introduce LumiNet, a novel architecture that leverages generative models and latent intrinsic representations for effective lighting transfer. Given a source image and a target lighting image, LumiNet synthesizes a relit version of the source scene that captures the target's lighting. Our approach makes two key contributions: a data curation strategy from the StyleGAN-based relighting model for our training, and a modified diffusion-based ControlNet that processes both latent intrinsic properties from the source image and latent extrinsic properties from the target image. We further improve lighting transfer through a learned adaptor (MLP) that injects the target's latent extrinsic properties via cross-attention and fine-tuning. Unlike traditional ControlNet, which generates images with conditional maps from a single scene, LumiNet processes latent representations from two different images - preserving geometry and albedo from the source while transferring lighting characteristics from the target. Experiments demonstrate that our method successfully transfers complex lighting phenomena including specular highlights and indirect illumination across scenes with varying spatial layouts and materials, outperforming existing approaches on challenging indoor scenes using only images as input.
>
---
#### [replaced 022] FastCache: Fast Caching for Diffusion Transformer Through Learnable Linear Approximation
- **分类: cs.LG; cs.AI; cs.CV; cs.MM; cs.PF**

- **链接: [http://arxiv.org/pdf/2505.20353v2](http://arxiv.org/pdf/2505.20353v2)**

> **作者:** Dong Liu; Yanxuan Yu; Jiayi Zhang; Yifan Li; Ben Lengerich; Ying Nian Wu
>
> **摘要:** Diffusion Transformers (DiT) are powerful generative models but remain computationally intensive due to their iterative structure and deep transformer stacks. To alleviate this inefficiency, we propose FastCache, a hidden-state-level caching and compression framework that accelerates DiT inference by exploiting redundancy within the model's internal representations. FastCache introduces a dual strategy: (1) a spatial-aware token selection mechanism that adaptively filters redundant tokens based on hidden state saliency, and (2) a transformer-level cache that reuses latent activations across timesteps when changes are statistically insignificant. These modules work jointly to reduce unnecessary computation while preserving generation fidelity through learnable linear approximation. Theoretical analysis shows that FastCache maintains bounded approximation error under a hypothesis-testing-based decision rule. Empirical evaluations across multiple DiT variants demonstrate substantial reductions in latency and memory usage, with best generation output quality compared to other cache methods, as measured by FID and t-FID. Code implementation of FastCache is available on GitHub at https://github.com/NoakLiu/FastCache-xDiT.
>
---
#### [replaced 023] LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.00790v3](http://arxiv.org/pdf/2507.00790v3)**

> **作者:** Huaqiu Li; Yong Wang; Tongwen Huang; Hailang Huang; Haoqian Wang; Xiangxiang Chu
>
> **摘要:** Unified image restoration is a significantly challenging task in low-level vision. Existing methods either make tailored designs for specific tasks, limiting their generalizability across various types of degradation, or rely on training with paired datasets, thereby suffering from closed-set constraints. To address these issues, we propose a novel, dataset-free, and unified approach through recurrent posterior sampling utilizing a pretrained latent diffusion model. Our method incorporates the multimodal understanding model to provide sematic priors for the generative model under a task-blind condition. Furthermore, it utilizes a lightweight module to align the degraded input with the generated preference of the diffusion model, and employs recurrent refinement for posterior sampling. Extensive experiments demonstrate that our method outperforms state-of-the-art methods, validating its effectiveness and robustness. Our code and data are available at https://github.com/AMAP-ML/LD-RPS.
>
---
#### [replaced 024] Learning a Neural Association Network for Self-supervised Multi-Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.11514v2](http://arxiv.org/pdf/2411.11514v2)**

> **作者:** Shuai Li; Michael Burke; Subramanian Ramamoorthy; Juergen Gall
>
> **备注:** BMVC2025 poster
>
> **摘要:** This paper introduces a novel framework to learn data association for multi-object tracking in a self-supervised manner. Fully-supervised learning methods are known to achieve excellent tracking performances, but acquiring identity-level annotations is tedious and time-consuming. Motivated by the fact that in real-world scenarios object motion can be usually represented by a Markov process, we present a novel expectation maximization (EM) algorithm that trains a neural network to associate detections for tracking, without requiring prior knowledge of their temporal correspondences. At the core of our method lies a neural Kalman filter, with an observation model conditioned on associations of detections parameterized by a neural network. Given a batch of frames as input, data associations between detections from adjacent frames are predicted by a neural network followed by a Sinkhorn normalization that determines the assignment probabilities of detections to states. Kalman smoothing is then used to obtain the marginal probability of observations given the inferred states, producing a training objective to maximize this marginal probability using gradient descent. The proposed framework is fully differentiable, allowing the underlying neural model to be trained end-to-end. We evaluate our approach on the challenging MOT17, MOT20, and BDD100K datasets and achieve state-of-the-art results in comparison to self-supervised trackers using public detections.
>
---
#### [replaced 025] Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04416v2](http://arxiv.org/pdf/2508.04416v2)**

> **作者:** Haoji Zhang; Xin Gu; Jiawen Li; Chixiang Ma; Sule Bai; Chubin Zhang; Bowen Zhang; Zhichao Zhou; Dongliang He; Yansong Tang
>
> **摘要:** The video reasoning ability of multimodal large language models (MLLMs) is crucial for downstream tasks like video question answering and temporal grounding. While recent approaches have explored text-based chain-of-thought (CoT) reasoning for MLLMs, these methods often suffer from limited cross-modal interaction and increased hallucination, especially with longer videos or reasoning chains. To address these challenges, we propose Video Intelligence via Tool-Augmented Learning (VITAL), a novel end-to-end agentic video reasoning framework. With a visual toolbox, the model can densely sample new video frames on demand and generate multimodal CoT for precise long video reasoning. We observe that temporal grounding and question answering are mutually beneficial for video understanding tasks. Therefore, we construct two high-quality multi-task video reasoning datasets MTVR-CoT-72k for supervised fine-tuning and MTVR-RL-110k for reinforcement learning. Moreover, we propose a Difficulty-aware Group Relative Policy Optimization algorithm (DGRPO) to mitigate difficulty imbalance in multi-task reinforcement learning. Extensive experiments on 11 challenging video understanding benchmarks demonstrate the advanced reasoning ability of VITAL, outperforming existing methods in video question answering and temporal grounding tasks, especially in long video scenarios. Code is available at https://zhang9302002.github.io/thinkingwithvideos-page/.
>
---
#### [replaced 026] Multimodal Iterative RAG for Knowledge Visual Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.00798v2](http://arxiv.org/pdf/2509.00798v2)**

> **作者:** Changin Choi; Wonseok Lee; Jungmin Ko; Wonjong Rhee
>
> **摘要:** While Multimodal Large Language Models (MLLMs) have significantly advanced multimodal understanding, their performance remains limited on knowledge-intensive visual questions that require external knowledge beyond the image. Retrieval-Augmented Generation (RAG) has become a promising solution for providing models with external knowledge, its conventional single-pass framework often fails to gather sufficient knowledge. To overcome this limitation, we propose MI-RAG, a Multimodal Iterative RAG framework that leverages reasoning to enhance retrieval and update reasoning over newly retrieved knowledge across modalities. At each iteration, MI-RAG leverages an accumulated reasoning record to dynamically formulate a multi-query. These queries then drive a joint search across heterogeneous knowledge bases containing both visually-grounded and textual knowledge. The newly acquired knowledge is synthesized into the reasoning record, progressively refining understanding across iterations. Experiments on challenging benchmarks, including Encyclopedic VQA, InfoSeek, and OK-VQA, show that MI-RAG significantly improves both retrieval recall and answer accuracy, establishing a scalable approach for compositional reasoning in knowledge-intensive VQA.
>
---
#### [replaced 027] Rethinking Data Protection in the (Generative) Artificial Intelligence Era
- **分类: cs.LG; cs.AI; cs.CR; cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.03034v4](http://arxiv.org/pdf/2507.03034v4)**

> **作者:** Yiming Li; Shuo Shao; Yu He; Junfeng Guo; Tianwei Zhang; Zhan Qin; Pin-Yu Chen; Michael Backes; Philip Torr; Dacheng Tao; Kui Ren
>
> **备注:** Perspective paper for a broader scientific audience. The first two authors contributed equally to this paper. 13 pages
>
> **摘要:** The (generative) artificial intelligence (AI) era has profoundly reshaped the meaning and value of data. No longer confined to static content, data now permeates every stage of the AI lifecycle from the training samples that shape model parameters to the prompts and outputs that drive real-world model deployment. This shift renders traditional notions of data protection insufficient, while the boundaries of what needs safeguarding remain poorly defined. Failing to safeguard data in AI systems can inflict societal and individual, underscoring the urgent need to clearly delineate the scope of and rigorously enforce data protection. In this perspective, we propose a four-level taxonomy, including non-usability, privacy preservation, traceability, and deletability, that captures the diverse protection needs arising in modern (generative) AI models and systems. Our framework offers a structured understanding of the trade-offs between data utility and control, spanning the entire AI pipeline, including training datasets, model weights, system prompts, and AI-generated content. We analyze representative technical approaches at each level and reveal regulatory blind spots that leave critical assets exposed. By offering a structured lens to align future AI technologies and governance with trustworthy data practices, we underscore the urgency of rethinking data protection for modern AI techniques and provide timely guidance for developers, researchers, and regulators alike.
>
---
#### [replaced 028] Towards Cardiac MRI Foundation Models: Comprehensive Visual-Tabular Representations for Whole-Heart Assessment and Beyond
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13037v4](http://arxiv.org/pdf/2504.13037v4)**

> **作者:** Yundi Zhang; Paul Hager; Che Liu; Suprosanna Shit; Chen Chen; Daniel Rueckert; Jiazhen Pan
>
> **摘要:** Cardiac magnetic resonance imaging is the gold standard for non-invasive cardiac assessment, offering rich spatio-temporal views of the cardiac anatomy and physiology. Patient-level health factors, such as demographics, metabolic, and lifestyle, are known to substantially influence cardiovascular health and disease risk, yet remain uncaptured by CMR alone. To holistically understand cardiac health and to enable the best possible interpretation of an individual's disease risk, CMR and patient-level factors must be jointly exploited within an integrated framework. Recent multi-modal approaches have begun to bridge this gap, yet they often rely on limited spatio-temporal data and focus on isolated clinical tasks, thereby hindering the development of a comprehensive representation for cardiac health evaluation. To overcome these limitations, we introduce ViTa, a step toward foundation models that delivers a comprehensive representation of the heart and a precise interpretation of individual disease risk. Leveraging data from 42,000 UK Biobank participants, ViTa integrates 3D+T cine stacks from short-axis and long-axis views, enabling a complete capture of the cardiac cycle. These imaging data are then fused with detailed tabular patient-level factors, enabling context-aware insights. This multi-modal paradigm supports a wide spectrum of downstream tasks, including cardiac phenotype and physiological feature prediction, segmentation, and classification of cardiac and metabolic diseases within a single unified framework. By learning a shared latent representation that bridges rich imaging features and patient context, ViTa moves beyond traditional, task-specific models toward a universal, patient-specific understanding of cardiac health, highlighting its potential to advance clinical utility and scalability in cardiac analysis.
>
---
#### [replaced 029] CompSlider: Compositional Slider for Disentangled Multiple-Attribute Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01028v2](http://arxiv.org/pdf/2509.01028v2)**

> **作者:** Zixin Zhu; Kevin Duarte; Mamshad Nayeem Rizve; Chengyuan Xu; Ratheesh Kalarot; Junsong Yuan
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** In text-to-image (T2I) generation, achieving fine-grained control over attributes - such as age or smile - remains challenging, even with detailed text prompts. Slider-based methods offer a solution for precise control of image attributes. Existing approaches typically train individual adapter for each attribute separately, overlooking the entanglement among multiple attributes. As a result, interference occurs among different attributes, preventing precise control of multiple attributes together. To address this challenge, we aim to disentangle multiple attributes in slider-based generation to enbale more reliable and independent attribute manipulation. Our approach, CompSlider, can generate a conditional prior for the T2I foundation model to control multiple attributes simultaneously. Furthermore, we introduce novel disentanglement and structure losses to compose multiple attribute changes while maintaining structural consistency within the image. Since CompSlider operates in the latent space of the conditional prior and does not require retraining the foundation model, it reduces the computational burden for both training and inference. We evaluate our approach on a variety of image attributes and highlight its generality by extending to video generation.
>
---
#### [replaced 030] Multimodal Medical Image Binding via Shared Text Embeddings
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18072v2](http://arxiv.org/pdf/2506.18072v2)**

> **作者:** Yunhao Liu; Suyang Xi; Shiqi Liu; Hong Ding; Chicheng Jin; Chong Zhong; Junjun He; Catherine C. Liu; Yiqing Shen
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Medical image analysis increasingly relies on the integration of multiple imaging modalities to capture complementary anatomical and functional information, enabling more accurate diagnosis and treatment planning. Achieving aligned feature representations across these diverse modalities is therefore important for effective multimodal analysis. While contrastive language-image pre-training (CLIP) and its variant have enabled image-text alignments, they require explicitly paired data between arbitrary two modalities, which is difficult to acquire in medical contexts. To address the gap, we present Multimodal Medical Image Binding with Text (M\textsuperscript{3}Bind), a novel pre-training framework that enables seamless alignment of multiple medical imaging modalities through a shared text representation space without requiring explicit paired data between any two medical image modalities. Specifically, based on the insight that different images can naturally bind with text, M\textsuperscript{3}Bind first fine-tunes pre-trained CLIP-like image-text models to align their modality-specific text embedding space while preserving their original image-text alignments. Subsequently, we distill these modality-specific text encoders into a unified model, creating a shared text embedding space. Experiments on X-ray, CT, retina, ECG, and pathological images on multiple downstream tasks demonstrate that M\textsuperscript{3}Bind achieves state-of-the-art performance in zero-shot, few-shot classification and cross-modal retrieval tasks compared to its CLIP-like counterparts. These results validate M\textsuperscript{3}Bind's effectiveness in achieving cross-image-modal alignment for medical analysis.
>
---
#### [replaced 031] GAEA: A Geolocation Aware Conversational Assistant
- **分类: cs.CV; cs.LG; I.4; I.2.7; I.5**

- **链接: [http://arxiv.org/pdf/2503.16423v3](http://arxiv.org/pdf/2503.16423v3)**

> **作者:** Ron Campos; Ashmal Vayani; Parth Parag Kulkarni; Rohit Gupta; Aizan Zafar; Aritra Dutta; Mubarak Shah
>
> **备注:** The dataset and code used in this submission is available at: https://ucf-crcv.github.io/GAEA/
>
> **摘要:** Image geolocalization, in which an AI model traditionally predicts the precise GPS coordinates of an image, is a challenging task with many downstream applications. However, the user cannot utilize the model to further their knowledge beyond the GPS coordinates; the model lacks an understanding of the location and the conversational ability to communicate with the user. In recent days, with the tremendous progress of large multimodal models (LMMs) -- proprietary and open-source -- researchers have attempted to geolocalize images via LMMs. However, the issues remain unaddressed; beyond general tasks, for more specialized downstream tasks, such as geolocalization, LMMs struggle. In this work, we propose solving this problem by introducing a conversational model, GAEA, that provides information regarding the location of an image as the user requires. No large-scale dataset enabling the training of such a model exists. Thus, we propose GAEA-1.4M, a comprehensive dataset comprising over 800k images and approximately 1.4M question-answer pairs, constructed by leveraging OpenStreetMap (OSM) attributes and geographical context clues. For quantitative evaluation, we propose a diverse benchmark, GAEA-Bench, comprising 3.5k image-text pairs to evaluate conversational capabilities equipped with diverse question types. We consider 11 state-of-the-art open-source and proprietary LMMs and demonstrate that GAEA significantly outperforms the best open-source model, LLaVA-OneVision, by 18.2% and the best proprietary model, GPT-4o, by 7.2%. Our dataset, model and codes are available.
>
---
#### [replaced 032] See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2509.02028v2](http://arxiv.org/pdf/2509.02028v2)**

> **作者:** Halima Bouzidi; Haoyu Liu; Mohammad Abdullah Al Faruque
>
> **备注:** 12 pages, 1 figure, 3 tables
>
> **摘要:** Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.
>
---
#### [replaced 033] Texture or Semantics? Vision-Language Models Get Lost in Font Recognition
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23768v3](http://arxiv.org/pdf/2503.23768v3)**

> **作者:** Zhecheng Li; Guoxian Song; Yujun Cai; Zhen Xiong; Junsong Yuan; Yiwei Wang
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Modern Vision-Language Models (VLMs) exhibit remarkable visual and linguistic capabilities, achieving impressive performance in various tasks such as image recognition and object localization. However, their effectiveness in fine-grained tasks remains an open question. In everyday scenarios, individuals encountering design materials, such as magazines, typography tutorials, research papers, or branding content, may wish to identify aesthetically pleasing fonts used in the text. Given their multimodal capabilities and free accessibility, many VLMs are often considered potential tools for font recognition. This raises a fundamental question: Do VLMs truly possess the capability to recognize fonts? To investigate this, we introduce the Font Recognition Benchmark (FRB), a compact and well-structured dataset comprising 15 commonly used fonts. FRB includes two versions: (i) an easy version, where 10 sentences are rendered in different fonts, and (ii) a hard version, where each text sample consists of the names of the 15 fonts themselves, introducing a stroop effect that challenges model perception. Through extensive evaluation of various VLMs on font recognition tasks, we arrive at the following key findings: (i) Current VLMs exhibit limited font recognition capabilities, with many state-of-the-art models failing to achieve satisfactory performance and being easily affected by the stroop effect introduced by textual information. (ii) Few-shot learning and Chain-of-Thought (CoT) prompting provide minimal benefits in improving font recognition accuracy across different VLMs. (iii) Attention analysis sheds light on the inherent limitations of VLMs in capturing semantic features.
>
---
#### [replaced 034] Refinement of Monocular Depth Maps via Multi-View Differentiable Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.03861v2](http://arxiv.org/pdf/2410.03861v2)**

> **作者:** Laura Fink; Linus Franke; Bernhard Egger; Joachim Keinert; Marc Stamminger
>
> **备注:** 8 pages main paper + 3 pages of references + 6 pages appendix
>
> **摘要:** Accurate depth estimation is at the core of many applications in computer graphics, vision, and robotics. Current state-of-the-art monocular depth estimators, trained on extensive datasets, generalize well but lack 3D consistency needed for many applications. In this paper, we combine the strength of those generalizing monocular depth estimation techniques with multi-view data by framing this as an analysis-by-synthesis optimization problem to lift and refine such relative depth maps to accurate error-free depth maps. After an initial global scale estimation through structure-from-motion point clouds, we further refine the depth map through optimization enforcing multi-view consistency via photometric and geometric losses with differentiable rendering of the meshed depth map. In a two-stage optimization, scaling is further refined first, and afterwards artifacts and errors in the depth map are corrected via nearby-view photometric supervision. Our evaluation shows that our method is able to generate detailed, high-quality, view consistent, accurate depth maps, also in challenging indoor scenarios, and outperforms state-of-the-art multi-view depth reconstruction approaches on such datasets. Project page and source code can be found at https://lorafib.github.io/ref_depth/.
>
---
#### [replaced 035] Deeply Supervised Flow-Based Generative Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14494v2](http://arxiv.org/pdf/2503.14494v2)**

> **作者:** Inkyu Shin; Chenglin Yang; Liang-Chieh Chen
>
> **备注:** Accepted to ICCV 2025. Project website at https://deepflow-project.github.io/
>
> **摘要:** Flow based generative models have charted an impressive path across multiple visual generation tasks by adhering to a simple principle: learning velocity representations of a linear interpolant. However, we observe that training velocity solely from the final layer output underutilizes the rich inter layer representations, potentially impeding model convergence. To address this limitation, we introduce DeepFlow, a novel framework that enhances velocity representation through inter layer communication. DeepFlow partitions transformer layers into balanced branches with deep supervision and inserts a lightweight Velocity Refiner with Acceleration (VeRA) block between adjacent branches, which aligns the intermediate velocity features within transformer blocks. Powered by the improved deep supervision via the internal velocity alignment, DeepFlow converges 8 times faster on ImageNet with equivalent performance and further reduces FID by 2.6 while halving training time compared to previous flow based models without a classifier free guidance. DeepFlow also outperforms baselines in text to image generation tasks, as evidenced by evaluations on MSCOCO and zero shot GenEval.
>
---
#### [replaced 036] Grid-Reg: Detector-Free Gridized Feature Learning and Matching for Large-Scale SAR-Optical Image Registration
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04233v2](http://arxiv.org/pdf/2507.04233v2)**

> **作者:** Xiaochen Wei; Weiwei Guo; Zenghui Zhang; Wenxian Yu
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** It is highly challenging to register large-scale, heterogeneous SAR and optical images, particularly across platforms, due to significant geometric, radiometric, and temporal differences, which most existing methods struggle to address. To overcome these challenges, we propose Grid-Reg, a grid-based multimodal registration framework comprising a domain-robust descriptor extraction network, Hybrid Siamese Correlation Metric Learning Network (HSCMLNet), and a grid-based solver (Grid-Solver) for transformation parameter estimation. In heterogeneous imagery with large modality gaps and geometric differences, obtaining accurate correspondences is inherently difficult. To robustly measure similarity between gridded patches, HSCMLNet integrates a hybrid Siamese module with a correlation metric learning module (CMLModule) based on equiangular unit basis vectors (EUBVs), together with a manifold consistency loss to promote modality-invariant, discriminative feature learning. The Grid-Solver estimates transformation parameters by minimizing a global grid matching loss through a progressive dual-loop search strategy to reliably find patch correspondences across entire images. Furthermore, we curate a challenging benchmark dataset for SAR-to-optical registration using real-world UAV MiniSAR data and Google Earth optical imagery. Extensive experiments demonstrate that our proposed approach achieves superior performance over state-of-the-art methods.
>
---
#### [replaced 037] AIM 2025 Rip Current Segmentation (RipSeg) Challenge Report
- **分类: cs.CV; cs.AI; I.4.0; I.4.9**

- **链接: [http://arxiv.org/pdf/2508.13401v2](http://arxiv.org/pdf/2508.13401v2)**

> **作者:** Andrei Dumitriu; Florin Miron; Florin Tatui; Radu Tudor Ionescu; Radu Timofte; Aakash Ralhan; Florin-Alexandru Vasluianu; Shenyang Qian; Mitchell Harley; Imran Razzak; Yang Song; Pu Luo; Yumei Li; Cong Xu; Jinming Chai; Kexin Zhang; Licheng Jiao; Lingling Li; Siqi Yu; Chao Zhang; Kehuan Song; Fang Liu; Puhua Chen; Xu Liu; Jin Hu; Jinyang Xu; Biao Liu
>
> **备注:** Challenge report paper from AIM2025 Workshop at ICCVW 2025
>
> **摘要:** This report presents an overview of the AIM 2025 RipSeg Challenge, a competition designed to advance techniques for automatic rip current segmentation in still images. Rip currents are dangerous, fast-moving flows that pose a major risk to beach safety worldwide, making accurate visual detection an important and underexplored research task. The challenge builds on RipVIS, the largest available rip current dataset, and focuses on single-class instance segmentation, where precise delineation is critical to fully capture the extent of rip currents. The dataset spans diverse locations, rip current types, and camera orientations, providing a realistic and challenging benchmark. In total, $75$ participants registered for this first edition, resulting in $5$ valid test submissions. Teams were evaluated on a composite score combining $F_1$, $F_2$, $AP_{50}$, and $AP_{[50:95]}$, ensuring robust and application-relevant rankings. The top-performing methods leveraged deep learning architectures, domain adaptation techniques, pretrained models, and domain generalization strategies to improve performance under diverse conditions. This report outlines the dataset details, competition framework, evaluation metrics, and final results, providing insights into the current state of rip current segmentation. We conclude with a discussion of key challenges, lessons learned from the submissions, and future directions for expanding RipSeg.
>
---
#### [replaced 038] Faster and Better: Reinforced Collaborative Distillation and Self-Learning for Infrared-Visible Image Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02424v2](http://arxiv.org/pdf/2509.02424v2)**

> **作者:** Yuhao Wang; Lingjuan Miao; Zhiqiang Zhou; Yajun Qiao; Lei Zhang
>
> **摘要:** Infrared and visible image fusion plays a critical role in enhancing scene perception by combining complementary information from different modalities. Despite recent advances, achieving high-quality image fusion with lightweight models remains a significant challenge. To bridge this gap, we propose a novel collaborative distillation and self-learning framework for image fusion driven by reinforcement learning. Unlike conventional distillation, this approach not only enables the student model to absorb image fusion knowledge from the teacher model, but more importantly, allows the student to perform self-learning on more challenging samples to enhance its capabilities. Particularly, in our framework, a reinforcement learning agent explores and identifies a more suitable training strategy for the student.The agent takes both the student's performance and the teacher-student gap as inputs, which leads to the generation of challenging samples to facilitate the student's self-learning. Simultaneously, it dynamically adjusts the teacher's guidance strength based on the student's state to optimize the knowledge transfer. Experimental results demonstrate that our method can significantly improve student performance and achieve better fusion results compared to existing techniques.
>
---
#### [replaced 039] WildFireCan-MMD: A Multimodal Dataset for Classification of User-Generated Content During Wildfires in Canada
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13231v3](http://arxiv.org/pdf/2504.13231v3)**

> **作者:** Braeden Sherritt; Isar Nejadgholi; Efstratios Aivaliotis; Khaled Mslmani; Marzieh Amini
>
> **摘要:** Rapid information access is vital during wildfires, yet traditional data sources are slow and costly. Social media offers real-time updates, but extracting relevant insights remains a challenge. In this work, we focus on multimodal wildfire social media data, which, although existing in current datasets, is currently underrepresented in Canadian contexts. We present WildFireCan-MMD, a new multimodal dataset of X posts from recent Canadian wildfires, annotated across twelve key themes. We evaluate zero-shot vision-language models on this dataset and compare their results with those of custom-trained and baseline classifiers. We show that while baseline methods and zero-shot prompting offer quick deployment, custom-trained models outperform them when labelled data is available. Our best-performing custom model reaches 84.48% f-score, outperforming VLMs and baseline classifiers. We also demonstrate how this model can be used to uncover trends during wildfires, through the collection and analysis of a large unlabeled dataset. Our dataset facilitates future research in wildfire response, and our findings highlight the importance of tailored datasets and task-specific training. Importantly, such datasets should be localized, as disaster response requirements vary across regions and contexts.
>
---
#### [replaced 040] Hues and Cues: Human vs. CLIP
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02305v2](http://arxiv.org/pdf/2509.02305v2)**

> **作者:** Nuria Alabau-Bosque; Jorge Vila-Tomás; Paula Daudén-Oliver; Pablo Hernández-Cámara; Jose Manuel Jaén-Lorites; Valero Laparra; Jesús Malo
>
> **备注:** 4 pages, 3 figures. 8th annual conference on Cognitive Computational Neuroscience
>
> **摘要:** Playing games is inherently human, and a lot of games are created to challenge different human characteristics. However, these tasks are often left out when evaluating the human-like nature of artificial models. The objective of this work is proposing a new approach to evaluate artificial models via board games. To this effect, we test the color perception and color naming capabilities of CLIP by playing the board game Hues & Cues and assess its alignment with humans. Our experiments show that CLIP is generally well aligned with human observers, but our approach brings to light certain cultural biases and inconsistencies when dealing with different abstraction levels that are hard to identify with other testing strategies. Our findings indicate that assessing models with different tasks like board games can make certain deficiencies in the models stand out in ways that are difficult to test with the commonly used benchmarks.
>
---
#### [replaced 041] Domain Consistency Representation Learning for Lifelong Person Re-Identification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.19954v4](http://arxiv.org/pdf/2409.19954v4)**

> **作者:** Shiben Liu; Huijie Fan; Qiang Wang; Weihong Ren; Yandong Tang; Yang Cong
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Lifelong person re-identification (LReID) exhibits a contradictory relationship between intra-domain discrimination and inter-domain gaps when learning from continuous data. Intra-domain discrimination focuses on individual nuances (i.e., clothing type, accessories, etc.), while inter-domain gaps emphasize domain consistency. Achieving a trade-off between maximizing intra-domain discrimination and minimizing inter-domain gaps is a crucial challenge for improving LReID performance. Most existing methods strive to reduce inter-domain gaps through knowledge distillation to maintain domain consistency. However, they often ignore intra-domain discrimination. To address this challenge, we propose a novel domain consistency representation learning (DCR) model that explores global and attribute-wise representations as a bridge to balance intra-domain discrimination and inter-domain gaps. At the intra-domain level, we explore the complementary relationship between global and attribute-wise representations to improve discrimination among similar identities. Excessive learning intra-domain discrimination can lead to catastrophic forgetting. We further develop an attribute-oriented anti-forgetting (AF) strategy that explores attribute-wise representations to enhance inter-domain consistency, and propose a knowledge consolidation (KC) strategy to facilitate knowledge transfer. Extensive experiments show that our DCR achieves superior performance compared to state-of-the-art LReID methods. Our code is available at https://github.com/LiuShiBen/DCR.
>
---
#### [replaced 042] JARVIS: A Neuro-Symbolic Commonsense Reasoning Framework for Conversational Embodied Agents
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2208.13266v4](http://arxiv.org/pdf/2208.13266v4)**

> **作者:** Kaizhi Zheng; Kaiwen Zhou; Jing Gu; Yue Fan; Jialu Wang; Zonglin Di; Xuehai He; Xin Eric Wang
>
> **备注:** 19th International Conference on Neurosymbolic Learning and Reasoning
>
> **摘要:** Building a conversational embodied agent to execute real-life tasks has been a long-standing yet quite challenging research goal, as it requires effective human-agent communication, multi-modal understanding, long-range sequential decision making, etc. Traditional symbolic methods have scaling and generalization issues, while end-to-end deep learning models suffer from data scarcity and high task complexity, and are often hard to explain. To benefit from both worlds, we propose JARVIS, a neuro-symbolic commonsense reasoning framework for modular, generalizable, and interpretable conversational embodied agents. First, it acquires symbolic representations by prompting large language models (LLMs) for language understanding and sub-goal planning, and by constructing semantic maps from visual observations. Then the symbolic module reasons for sub-goal planning and action generation based on task- and action-level common sense. Extensive experiments on the TEACh dataset validate the efficacy and efficiency of our JARVIS framework, which achieves state-of-the-art (SOTA) results on all three dialog-based embodied tasks, including Execution from Dialog History (EDH), Trajectory from Dialog (TfD), and Two-Agent Task Completion (TATC) (e.g., our method boosts the unseen Success Rate on EDH from 6.1\% to 15.8\%). Moreover, we systematically analyze the essential factors that affect the task performance and also demonstrate the superiority of our method in few-shot settings. Our JARVIS model ranks first in the Alexa Prize SimBot Public Benchmark Challenge.
>
---
#### [replaced 043] Spotlighter: Revisiting Prompt Tuning from a Representative Mining View
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.00905v2](http://arxiv.org/pdf/2509.00905v2)**

> **作者:** Yutong Gao; Maoyuan Shao; Xinyang Huang; Chuang Zhu; Lijuan Sun; Yu Weng; Xuan Liu; Guoshun Nan
>
> **备注:** Accepted as EMNLP 2025 Findings
>
> **摘要:** CLIP's success has demonstrated that prompt tuning can achieve robust cross-modal semantic alignment for tasks ranging from open-domain recognition to fine-grained classification. However, redundant or weakly relevant feature components introduce noise and incur unnecessary computational costs. In this work, we propose Spotlighter, a lightweight token-selection framework that simultaneously enhances accuracy and efficiency in prompt tuning. Spotlighter evaluates each visual token's activation from both sample-wise and semantic-wise perspectives and retains only the top-scoring tokens for downstream prediction. A class-specific semantic memory bank of learned prototypes refines this selection, ensuring semantic representativeness and compensating for discarded features. To further prioritize informative signals, we introduce a two-level ranking mechanism that dynamically weights token--prototype interactions. Across 11 few-shot benchmarks, Spotlighter outperforms CLIP by up to 11.19\% in harmonic mean accuracy and achieves up to 0.8K additional FPS, with only 21 extra parameters. These results establish Spotlighter as an effective and scalable baseline for prompt tuning. Code for our method will be available at https://github.com/greatest-gourmet/Spotlighter.
>
---
#### [replaced 044] DeepTopoNet: A Framework for Subglacial Topography Estimation on the Greenland Ice Sheets
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.23980v2](http://arxiv.org/pdf/2505.23980v2)**

> **作者:** Bayu Adhi Tama; Mansa Krishna; Homayra Alam; Mostafa Cham; Omar Faruque; Gong Cheng; Jianwu Wang; Mathieu Morlighem; Vandana Janeja
>
> **备注:** Accepted as Full Application Track Paper in SIGSPATIAL 2025
>
> **摘要:** Understanding Greenland's subglacial topography is critical for projecting the future mass loss of the ice sheet and its contribution to global sea-level rise. However, the complex and sparse nature of observational data, particularly information about the bed topography under the ice sheet, significantly increases the uncertainty in model projections. Bed topography is traditionally measured by airborne ice-penetrating radar that measures the ice thickness directly underneath the aircraft, leaving data gap of tens of kilometers in between flight lines. This study introduces a deep learning framework, which we call as DeepTopoNet, that integrates radar-derived ice thickness observations and BedMachine Greenland data through a novel dynamic loss-balancing mechanism. Among all efforts to reconstruct bed topography, BedMachine has emerged as one of the most widely used datasets, combining mass conservation principles and ice thickness measurements to generate high-resolution bed elevation estimates. The proposed loss function adaptively adjusts the weighting between radar and BedMachine data, ensuring robustness in areas with limited radar coverage while leveraging the high spatial resolution of BedMachine predictions i.e. bed estimates. Our approach incorporates gradient-based and trend surface features to enhance model performance and utilizes a CNN architecture designed for subgrid-scale predictions. By systematically testing on the Upernavik Isstr{\o}m) region, the model achieves high accuracy, outperforming baseline methods in reconstructing subglacial terrain. This work demonstrates the potential of deep learning in bridging observational gaps, providing a scalable and efficient solution to inferring subglacial topography.
>
---
#### [replaced 045] HydroVision: Predicting Optically Active Parameters in Surface Water Using Computer Vision
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.01882v2](http://arxiv.org/pdf/2509.01882v2)**

> **作者:** Shubham Laxmikant Deshmukh; Matthew Wilchek; Feras A. Batarseh
>
> **备注:** This paper is under peer review for IEEE Journal of Oceanic Engineering
>
> **摘要:** Ongoing advancements in computer vision, particularly in pattern recognition and scene classification, have enabled new applications in environmental monitoring. Deep learning now offers non-contact methods for assessing water quality and detecting contamination, both critical for disaster response and public health protection. This work introduces HydroVision, a deep learning-based scene classification framework that estimates optically active water quality parameters including Chlorophyll-Alpha, Chlorophylls, Colored Dissolved Organic Matter (CDOM), Phycocyanins, Suspended Sediments, and Turbidity from standard Red-Green-Blue (RGB) images of surface water. HydroVision supports early detection of contamination trends and strengthens monitoring by regulatory agencies during external environmental stressors, industrial activities, and force majeure events. The model is trained on more than 500,000 seasonally varied images collected from the United States Geological Survey Hydrologic Imagery Visualization and Information System between 2022 and 2024. This approach leverages widely available RGB imagery as a scalable, cost-effective alternative to traditional multispectral and hyperspectral remote sensing. Four state-of-the-art convolutional neural networks (VGG-16, ResNet50, MobileNetV2, DenseNet121) and a Vision Transformer are evaluated through transfer learning to identify the best-performing architecture. DenseNet121 achieves the highest validation performance, with an R2 score of 0.89 in predicting CDOM, demonstrating the framework's promise for real-world water quality monitoring across diverse conditions. While the current model is optimized for well-lit imagery, future work will focus on improving robustness under low-light and obstructed scenarios to expand its operational utility.
>
---
#### [replaced 046] LanternNet: A Hub-and-Spoke System to Seek and Suppress Spotted Lanternfly Populations
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20800v3](http://arxiv.org/pdf/2507.20800v3)**

> **作者:** Vinil Polepalli
>
> **摘要:** The invasive spotted lanternfly (SLF) poses a significant threat to agriculture and ecosystems, causing widespread damage. Current control methods, such as egg scraping, pesticides, and quarantines, prove labor-intensive, environmentally hazardous, and inadequate for long-term SLF suppression. This research introduces LanternNet, a novel autonomous robotic Hub-and-Spoke system designed for scalable detection and suppression of SLF populations. A central, tree-mimicking hub utilizes a YOLOv8 computer vision model for precise SLF identification. Three specialized robotic spokes perform targeted tasks: pest neutralization, environmental monitoring, and navigation/mapping. Field deployment across multiple infested sites over 5 weeks demonstrated LanternNet's efficacy. Quantitative analysis revealed significant reductions (p < 0.01, paired t-tests) in SLF populations and corresponding improvements in tree health indicators across the majority of test sites. Compared to conventional methods, LanternNet offers substantial cost advantages and improved scalability. Furthermore, the system's adaptability for enhanced autonomy and targeting of other invasive species presents significant potential for broader ecological impact. LanternNet demonstrates the transformative potential of integrating robotics and AI for advanced invasive species management and improved environmental outcomes.
>
---
#### [replaced 047] Beyond Feature Mapping GAP: Integrating Real HDRTV Priors for Superior SDRTV-to-HDRTV Conversion
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.10775v2](http://arxiv.org/pdf/2411.10775v2)**

> **作者:** Gang He; Kepeng Xu; Li Xu; Wenxin Yu; Xianyun Wu
>
> **备注:** accepted by IJCAI 2025
>
> **摘要:** The rise of HDR-WCG display devices has highlighted the need to convert SDRTV to HDRTV, as most video sources are still in SDR. Existing methods primarily focus on designing neural networks to learn a single-style mapping from SDRTV to HDRTV. However, the limited information in SDRTV and the diversity of styles in real-world conversions render this process an ill-posed problem, thereby constraining the performance and generalization of these methods. Inspired by generative approaches, we propose a novel method for SDRTV to HDRTV conversion guided by real HDRTV priors. Despite the limited information in SDRTV, introducing real HDRTV as reference priors significantly constrains the solution space of the originally high-dimensional ill-posed problem. This shift transforms the task from solving an unreferenced prediction problem to making a referenced selection, thereby markedly enhancing the accuracy and reliability of the conversion process. Specifically, our approach comprises two stages: the first stage employs a Vector Quantized Generative Adversarial Network to capture HDRTV priors, while the second stage matches these priors to the input SDRTV content to recover realistic HDRTV outputs. We evaluate our method on public datasets, demonstrating its effectiveness with significant improvements in both objective and subjective metrics across real and synthetic datasets.
>
---
#### [replaced 048] Comparing Next-Day Wildfire Predictability of MODIS and VIIRS Satellite Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08580v4](http://arxiv.org/pdf/2503.08580v4)**

> **作者:** Justus Karlsson; Yonghao Xu; Amanda Berg; Leif Haglund
>
> **摘要:** Multiple studies have performed next-day fire prediction using satellite imagery. Two main satellites are used to detect wildfires: MODIS and VIIRS. Both satellites provide fire mask products, called MOD14 and VNP14, respectively. Studies have used one or the other, but there has been no comparison between them to determine which might be more suitable for next-day fire prediction. In this paper, we first evaluate how well VIIRS and MODIS data can be used to forecast wildfire spread one day ahead. We find that the model using VIIRS as input and VNP14 as target achieves the best results. Interestingly, the model using MODIS as input and VNP14 as target performs significantly better than using VNP14 as input and MOD14 as target. Next, we discuss why MOD14 might be harder to use for predicting next-day fires. We find that the MOD14 fire mask is highly stochastic and does not correlate with reasonable fire spread patterns. This is detrimental for machine learning tasks, as the model learns irrational patterns. Therefore, we conclude that MOD14 is unsuitable for next-day fire prediction and that VNP14 is a much better option. However, using MODIS input and VNP14 as target, we achieve a significant improvement in predictability. This indicates that an improved fire detection model is possible for MODIS. The full code and dataset is available online: https://github.com/justuskarlsson/wildfire-mod14-vnp14
>
---
#### [replaced 049] Aligning Machine and Human Visual Representations across Abstraction Levels
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.06509v4](http://arxiv.org/pdf/2409.06509v4)**

> **作者:** Lukas Muttenthaler; Klaus Greff; Frieda Born; Bernhard Spitzer; Simon Kornblith; Michael C. Mozer; Klaus-Robert Müller; Thomas Unterthiner; Andrew K. Lampinen
>
> **备注:** 91 pages
>
> **摘要:** Deep neural networks have achieved success across a wide range of applications, including as models of human behavior and neural representations in vision tasks. However, neural network training and human learning differ in fundamental ways, and neural networks often fail to generalize as robustly as humans do raising questions regarding the similarity of their underlying representations. What is missing for modern learning systems to exhibit more human-aligned behavior? We highlight a key misalignment between vision models and humans: whereas human conceptual knowledge is hierarchically organized from fine- to coarse-scale distinctions, model representations do not accurately capture all these levels of abstraction. To address this misalignment, we first train a teacher model to imitate human judgments, then transfer human-aligned structure from its representations to refine the representations of pretrained state-of-the-art vision foundation models via finetuning. These human-aligned models more accurately approximate human behavior and uncertainty across a wide range of similarity tasks, including a new dataset of human judgments spanning multiple levels of semantic abstractions. They also perform better on a diverse set of machine learning tasks, increasing generalization and out-of-distribution robustness. Thus, infusing neural networks with additional human knowledge yields a best-of-both-worlds representation that is both more consistent with human cognitive judgments and more practically useful, thus paving the way toward more robust, interpretable, and human-aligned artificial intelligence systems.
>
---
#### [replaced 050] Bridging the Domain Gap for Flight-Ready Spaceborne Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.11661v2](http://arxiv.org/pdf/2409.11661v2)**

> **作者:** Tae Ha Park; Simone D'Amico
>
> **备注:** Accepted to Journal of Spacecraft and Rockets
>
> **摘要:** This work presents Spacecraft Pose Network v3 (SPNv3), a Neural Network (NN) for monocular pose estimation of a known, non-cooperative target spacecraft. SPNv3 is designed and trained to be computationally efficient while providing robustness to spaceborne images that have not been observed during offline training and validation on the ground. These characteristics are essential to deploying NNs on space-grade edge devices. They are achieved through careful NN design choices, and an extensive trade-off analysis reveals features such as data augmentation, transfer learning and vision transformer architecture as a few of those that contribute to simultaneously maximizing robustness and minimizing computational overhead. Experiments demonstrate that the final SPNv3 can achieve state-of-the-art pose accuracy on hardware-in-the-loop images from a robotic testbed while having trained exclusively on computer-generated synthetic images, effectively bridging the domain gap between synthetic and real imagery. At the same time, SPNv3 runs well above the update frequency of modern satellite navigation filters when tested on a representative graphical processing unit system with flight heritage. Overall, SPNv3 is an efficient, flight-ready NN model readily applicable to close-range rendezvous and proximity operations with target resident space objects.
>
---
#### [replaced 051] TruthLens: Visual Grounding for Universal DeepFake Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15867v3](http://arxiv.org/pdf/2503.15867v3)**

> **作者:** Rohit Kundu; Shan Jia; Vishal Mohanty; Athula Balachandran; Amit K. Roy-Chowdhury
>
> **摘要:** Detecting DeepFakes has become a crucial research area as the widespread use of AI image generators enables the effortless creation of face-manipulated and fully synthetic content, while existing methods are often limited to binary classification (real vs. fake) and lack interpretability. To address these challenges, we propose TruthLens, a novel, unified, and highly generalizable framework that goes beyond traditional binary classification, providing detailed, textual reasoning for its predictions. Distinct from conventional methods, TruthLens performs MLLM grounding. TruthLens uses a task-driven representation integration strategy that unites global semantic context from a multimodal large language model (MLLM) with region-specific forensic cues through explicit cross-modal adaptation of a vision-only model. This enables nuanced, region-grounded reasoning for both face-manipulated and fully synthetic content, and supports fine-grained queries such as "Does the eyes/nose/mouth look real or fake?"- capabilities beyond pretrained MLLMs alone. Extensive experiments across diverse datasets demonstrate that TruthLens sets a new benchmark in both forensic interpretability and detection accuracy, generalizing to seen and unseen manipulations alike. By unifying high-level scene understanding with fine-grained region grounding, TruthLens delivers transparent DeepFake forensics, bridging a critical gap in the literature.
>
---
#### [replaced 052] Repurposing SAM for User-Defined Semantics Aware Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.02420v3](http://arxiv.org/pdf/2312.02420v3)**

> **作者:** Rohit Kundu; Sudipta Paul; Arindam Dutta; Amit K. Roy-Chowdhury
>
> **摘要:** The Segment Anything Model (SAM) excels at generating precise object masks from input prompts but lacks semantic awareness, failing to associate its generated masks with specific object categories. To address this limitation, we propose U-SAM, a novel framework that imbibes semantic awareness into SAM, enabling it to generate targeted masks for user-specified object categories. Given only object class names as input from the user, U-SAM provides pixel-level semantic annotations for images without requiring any labeled/unlabeled samples from the test data distribution. Our approach leverages synthetically generated or web crawled images to accumulate semantic information about the desired object classes. We then learn a mapping function between SAM's mask embeddings and object class labels, effectively enhancing SAM with granularity-specific semantic recognition capabilities. As a result, users can obtain meaningful and targeted segmentation masks for specific objects they request, rather than generic and unlabeled masks. We evaluate U-SAM on PASCAL VOC 2012 and MSCOCO-80, achieving significant mIoU improvements of +17.95% and +5.20%, respectively, over state-of-the-art methods. By transforming SAM into a semantically aware segmentation model, U-SAM offers a practical and flexible solution for pixel-level annotation across diverse and unseen domains in a resource-constrained environment.
>
---
#### [replaced 053] Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18829v2](http://arxiv.org/pdf/2504.18829v2)**

> **作者:** Jiayi Chen; Yubin Ke; Lin Peng; He Wang
>
> **备注:** Accepted by Robotics: Science and Systems (RSS 2025)
>
> **摘要:** Generalizable dexterous grasping with suitable grasp types is a fundamental skill for intelligent robots. Developing such skills requires a large-scale and high-quality dataset that covers numerous grasp types (i.e., at least those categorized by the GRASP taxonomy), but collecting such data is extremely challenging. Existing automatic grasp synthesis methods are often limited to specific grasp types or object categories, hindering scalability. This work proposes an efficient pipeline capable of synthesizing contact-rich, penetration-free, and physically plausible grasps for any grasp type, object, and articulated hand. Starting from a single human-annotated template for each hand and grasp type, our pipeline tackles the complicated synthesis problem with two stages: optimize the object to fit the hand template first, and then locally refine the hand to fit the object in simulation. To validate the synthesized grasps, we introduce a contact-aware control strategy that allows the hand to apply the appropriate force at each contact point to the object. Those validated grasps can also be used as new grasp templates to facilitate future synthesis. Experiments show that our method significantly outperforms previous type-unaware grasp synthesis baselines in simulation. Using our algorithm, we construct a dataset containing 10.7k objects and 9.5M grasps, covering 31 grasp types in the GRASP taxonomy. Finally, we train a type-conditional generative model that successfully performs the desired grasp type from single-view object point clouds, achieving an 82.3% success rate in real-world experiments. Project page: https://pku-epic.github.io/Dexonomy.
>
---
#### [replaced 054] Survey on Hand Gesture Recognition from Visual Input
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.11992v3](http://arxiv.org/pdf/2501.11992v3)**

> **作者:** Manousos Linardakis; Iraklis Varlamis; Georgios Th. Papadopoulos
>
> **备注:** 37 pages
>
> **摘要:** Hand gesture recognition has become an important research area, driven by the growing demand for human-computer interaction in fields such as sign language recognition, virtual and augmented reality, and robotics. Despite the rapid growth of the field, there are few surveys that comprehensively cover recent research developments, available solutions, and benchmark datasets. This survey addresses this gap by examining the latest advancements in hand gesture and 3D hand pose recognition from various types of camera input data including RGB images, depth images, and videos from monocular or multiview cameras, examining the differing methodological requirements of each approach. Furthermore, an overview of widely used datasets is provided, detailing their main characteristics and application domains. Finally, open challenges such as achieving robust recognition in real-world environments, handling occlusions, ensuring generalization across diverse users, and addressing computational efficiency for real-time applications are highlighted to guide future research directions. By synthesizing the objectives, methodologies, and applications of recent studies, this survey offers valuable insights into current trends, challenges, and opportunities for future research in human hand gesture recognition.
>
---
#### [replaced 055] Embedding Similarity Guided License Plate Super Resolution
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.01483v3](http://arxiv.org/pdf/2501.01483v3)**

> **作者:** Abderrezzaq Sendjasni; Mohamed-Chaker Larabi
>
> **备注:** Accepted in Neurocomputing
>
> **摘要:** Super-resolution (SR) techniques play a pivotal role in enhancing the quality of low-resolution images, particularly for applications such as security and surveillance, where accurate license plate recognition is crucial. This study proposes a novel framework that combines pixel-based loss with embedding similarity learning to address the unique challenges of license plate super-resolution (LPSR). The introduced pixel and embedding consistency loss (PECL) integrates a Siamese network and applies contrastive loss to force embedding similarities to improve perceptual and structural fidelity. By effectively balancing pixel-wise accuracy with embedding-level consistency, the framework achieves superior alignment of fine-grained features between high-resolution (HR) and super-resolved (SR) license plates. Extensive experiments on the CCPD and PKU dataset validate the efficacy of the proposed framework, demonstrating consistent improvements over state-of-the-art methods in terms of PSNR, SSIM, LPIPS, and optical character recognition (OCR) accuracy. These results highlight the potential of embedding similarity learning to advance both perceptual quality and task-specific performance in extreme super-resolution scenarios.
>
---
#### [replaced 056] ViDDAR: Vision Language Model-Based Task-Detrimental Content Detection for Augmented Reality
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12553v2](http://arxiv.org/pdf/2501.12553v2)**

> **作者:** Yanming Xiu; Tim Scargill; Maria Gorlatova
>
> **备注:** The paper has been accepted to the 2025 IEEE Conference on Virtual Reality and 3D User Interfaces (IEEE VR), and selected for publication in the 2025 IEEE Transactions on Visualization and Computer Graphics (TVCG) special issue
>
> **摘要:** In Augmented Reality (AR), virtual content enhances user experience by providing additional information. However, improperly positioned or designed virtual content can be detrimental to task performance, as it can impair users' ability to accurately interpret real-world information. In this paper we examine two types of task-detrimental virtual content: obstruction attacks, in which virtual content prevents users from seeing real-world objects, and information manipulation attacks, in which virtual content interferes with users' ability to accurately interpret real-world information. We provide a mathematical framework to characterize these attacks and create a custom open-source dataset for attack evaluation. To address these attacks, we introduce ViDDAR (Vision language model-based Task-Detrimental content Detector for Augmented Reality), a comprehensive full-reference system that leverages Vision Language Models (VLMs) and advanced deep learning techniques to monitor and evaluate virtual content in AR environments, employing a user-edge-cloud architecture to balance performance with low latency. To the best of our knowledge, ViDDAR is the first system to employ VLMs for detecting task-detrimental content in AR settings. Our evaluation results demonstrate that ViDDAR effectively understands complex scenes and detects task-detrimental content, achieving up to 92.15% obstruction detection accuracy with a detection latency of 533 ms, and an 82.46% information manipulation content detection accuracy with a latency of 9.62 s.
>
---
#### [replaced 057] HodgeFormer: Transformers for Learnable Operators on Triangular Meshes through Data-Driven Hodge Matrices
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01839v2](http://arxiv.org/pdf/2509.01839v2)**

> **作者:** Akis Nousias; Stavros Nousias
>
> **备注:** 13 pages, 11 figures, 9 tables
>
> **摘要:** Currently, prominent Transformer architectures applied on graphs and meshes for shape analysis tasks employ traditional attention layers that heavily utilize spectral features requiring costly eigenvalue decomposition-based methods. To encode the mesh structure, these methods derive positional embeddings, that heavily rely on eigenvalue decomposition based operations, e.g. on the Laplacian matrix, or on heat-kernel signatures, which are then concatenated to the input features. This paper proposes a novel approach inspired by the explicit construction of the Hodge Laplacian operator in Discrete Exterior Calculus as a product of discrete Hodge operators and exterior derivatives, i.e. $(L := \star_0^{-1} d_0^T \star_1 d_0)$. We adjust the Transformer architecture in a novel deep learning layer that utilizes the multi-head attention mechanism to approximate Hodge matrices $\star_0$, $\star_1$ and $\star_2$ and learn families of discrete operators $L$ that act on mesh vertices, edges and faces. Our approach results in a computationally-efficient architecture that achieves comparable performance in mesh segmentation and classification tasks, through a direct learning framework, while eliminating the need for costly eigenvalue decomposition operations or complex preprocessing operations.
>
---
#### [replaced 058] MedDINOv3: How to adapt vision foundation models for medical image segmentation?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02379v2](http://arxiv.org/pdf/2509.02379v2)**

> **作者:** Yuheng Li; Yizhou Wu; Yuxiang Lai; Mingzhe Hu; Xiaofeng Yang
>
> **摘要:** Accurate segmentation of organs and tumors in CT and MRI scans is essential for diagnosis, treatment planning, and disease monitoring. While deep learning has advanced automated segmentation, most models remain task-specific, lacking generalizability across modalities and institutions. Vision foundation models (FMs) pretrained on billion-scale natural images offer powerful and transferable representations. However, adapting them to medical imaging faces two key challenges: (1) the ViT backbone of most foundation models still underperform specialized CNNs on medical image segmentation, and (2) the large domain gap between natural and medical images limits transferability. We introduce MedDINOv3, a simple and effective framework for adapting DINOv3 to medical segmentation. We first revisit plain ViTs and design a simple and effective architecture with multi-scale token aggregation. Then, we perform domain-adaptive pretraining on CT-3M, a curated collection of 3.87M axial CT slices, using a multi-stage DINOv3 recipe to learn robust dense features. MedDINOv3 matches or exceeds state-of-the-art performance across four segmentation benchmarks, demonstrating the potential of vision foundation models as unified backbones for medical image segmentation. The code is available at https://github.com/ricklisz/MedDINOv3.
>
---
#### [replaced 059] PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.05085v2](http://arxiv.org/pdf/2411.05085v2)**

> **作者:** Daniel C. Castro; Aurelia Bustos; Shruthi Bannur; Stephanie L. Hyland; Kenza Bouzid; Maria Teodora Wetscherek; Maria Dolores Sánchez-Valverde; Lara Jaques-Pérez; Lourdes Pérez-Rodríguez; Kenji Takeda; José María Salinas; Javier Alvarez-Valle; Joaquín Galant Herrero; Antonio Pertusa
>
> **摘要:** Radiology report generation (RRG) aims to create free-text radiology reports from clinical imaging. Grounded radiology report generation (GRRG) extends RRG by including the localisation of individual findings on the image. Currently, there are no manually annotated chest X-ray (CXR) datasets to train GRRG models. In this work, we present a dataset called PadChest-GR (Grounded-Reporting) derived from PadChest aimed at training GRRG models for CXR images. We curate a public bi-lingual dataset of 4,555 CXR studies with grounded reports (3,099 abnormal and 1,456 normal), each containing complete lists of sentences describing individual present (positive) and absent (negative) findings in English and Spanish. In total, PadChest-GR contains 7,037 positive and 3,422 negative finding sentences. Every positive finding sentence is associated with up to two independent sets of bounding boxes labelled by different readers and has categorical labels for finding type, locations, and progression. To the best of our knowledge, PadChest-GR is the first manually curated dataset designed to train GRRG models for understanding and interpreting radiological images and generated text. By including detailed localization and comprehensive annotations of all clinically relevant findings, it provides a valuable resource for developing and evaluating GRRG models from CXR images. PadChest-GR can be downloaded under request from https://bimcv.cipf.es/bimcv-projects/padchest-gr/
>
---
#### [replaced 060] Mitigating Hallucination in Large Vision-Language Models through Aligning Attention Distribution to Information Flow
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14257v2](http://arxiv.org/pdf/2505.14257v2)**

> **作者:** Jianfei Zhao; Feng Zhang; Xin Sun; Chong Feng
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Due to the unidirectional masking mechanism, Decoder-Only models propagate information from left to right. LVLMs (Large Vision-Language Models) follow the same architecture, with visual information gradually integrated into semantic representations during forward propagation. Through systematic analysis, we observe that the majority of the visual information is absorbed into the semantic representations. However, the model's attention distribution does not exhibit sufficient emphasis on semantic representations. This misalignment between the attention distribution and the actual information flow undermines the model's visual understanding ability and contributes to hallucinations. To address this issue, we enhance the model's visual understanding by leveraging the core information embedded in semantic representations. Specifically, we identify attention heads that focus on core semantic representations based on their attention distributions. Then, through a two-stage optimization paradigm, we propagate the advantages of these attention heads across the entire model, aligning the attention distribution with the actual information flow. We evaluate our method on three image captioning benchmarks using five different LVLMs, demonstrating its effectiveness in significantly reducing hallucinations. Further experiments reveal a trade-off between reduced hallucinations and richer details. Notably, our method allows for manual adjustment of the model's conservativeness, enabling flexible control to meet diverse real-world requirements.
>
---
#### [replaced 061] Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12348v2](http://arxiv.org/pdf/2506.12348v2)**

> **作者:** Zaiqiang Wu; I-Chao Shen; Takeo Igarashi
>
> **摘要:** Per-garment virtual try-on methods collect garment-specific datasets and train networks tailored to each garment to achieve superior results. However, these approaches often struggle with loose-fitting garments due to two key limitations: (1) They rely on human body semantic maps to align garments with the body, but these maps become unreliable when body contours are obscured by loose-fitting garments, resulting in degraded outcomes; (2) They train garment synthesis networks on a per-frame basis without utilizing temporal information, leading to noticeable jittering artifacts. To address the first limitation, we propose a two-stage approach for robust semantic map estimation. First, we extract a garment-invariant representation from the raw input image. This representation is then passed through an auxiliary network to estimate the semantic map. This enhances the robustness of semantic map estimation under loose-fitting garments during garment-specific dataset generation. To address the second limitation, we introduce a recurrent garment synthesis framework that incorporates temporal dependencies to improve frame-to-frame coherence while maintaining real-time performance. We conducted qualitative and quantitative evaluations to demonstrate that our method outperforms existing approaches in both image quality and temporal coherence. Ablation studies further validate the effectiveness of the garment-invariant representation and the recurrent synthesis framework.
>
---
#### [replaced 062] Mind the Third Eye! Benchmarking Privacy Awareness in MLLM-powered Smartphone Agents
- **分类: cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.19493v2](http://arxiv.org/pdf/2508.19493v2)**

> **作者:** Zhixin Lin; Jungang Li; Shidong Pan; Yibo Shi; Yue Yao; Dongliang Xu
>
> **摘要:** Smartphones bring significant convenience to users but also enable devices to extensively record various types of personal information. Existing smartphone agents powered by Multimodal Large Language Models (MLLMs) have achieved remarkable performance in automating different tasks. However, as the cost, these agents are granted substantial access to sensitive users' personal information during this operation. To gain a thorough understanding of the privacy awareness of these agents, we present the first large-scale benchmark encompassing 7,138 scenarios to the best of our knowledge. In addition, for privacy context in scenarios, we annotate its type (e.g., Account Credentials), sensitivity level, and location. We then carefully benchmark seven available mainstream smartphone agents. Our results demonstrate that almost all benchmarked agents show unsatisfying privacy awareness (RA), with performance remaining below 60% even with explicit hints. Overall, closed-source agents show better privacy ability than open-source ones, and Gemini 2.0-flash achieves the best, achieving an RA of 67%. We also find that the agents' privacy detection capability is highly related to scenario sensitivity level, i.e., the scenario with a higher sensitivity level is typically more identifiable. We hope the findings enlighten the research community to rethink the unbalanced utility-privacy tradeoff about smartphone agents. Our code and benchmark are available at https://zhixin-l.github.io/SAPA-Bench.
>
---
#### [replaced 063] A Multimodal and Multi-centric Head and Neck Cancer Dataset for Tumor Segmentation and Outcome Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00367v2](http://arxiv.org/pdf/2509.00367v2)**

> **作者:** Numan Saeed; Salma Hassan; Shahad Hardan; Ahmed Aly; Darya Taratynova; Umair Nawaz; Ufaq Khan; Muhammad Ridzuan; Vincent Andrearczyk; Adrien Depeursinge; Mathieu Hatt; Thomas Eugene; Raphaël Metz; Mélanie Dore; Gregory Delpon; Vijay Ram Kumar Papineni; Kareem Wahid; Cem Dede; Alaa Mohamed Shawky Ali; Carlos Sjogreen; Mohamed Naser; Clifton D. Fuller; Valentin Oreiller; Mario Jreige; John O. Prior; Catherine Cheze Le Rest; Olena Tankyevych; Pierre Decazes; Su Ruan; Stephanie Tanadini-Lang; Martin Vallières; Hesham Elhalawani; Ronan Abgral; Romain Floch; Kevin Kerleguer; Ulrike Schick; Maelle Mauguen; Arman Rahmim; Mohammad Yaqub
>
> **备注:** 10 pages, 5 figures. Numan Saeed is the corresponding author. Numan Saeed, Salma Hassan and Shahad Hardan contributed equally to this work. Project page: https://hecktor25.grand-challenge.org/
>
> **摘要:** We describe a publicly available multimodal dataset of annotated Positron Emission Tomography/Computed Tomography (PET/CT) studies for head and neck cancer research. The dataset includes 1123 FDG-PET/CT studies from patients with histologically confirmed head and neck cancer, acquired from 10 international medical centers. All examinations consisted of co-registered PET/CT scans with varying acquisition protocols, reflecting real-world clinical diversity across institutions. Primary gross tumor volumes (GTVp) and involved lymph nodes (GTVn) were manually segmented by experienced radiation oncologists and radiologists following standardized guidelines and quality control measures. We provide anonymized NifTi files of all studies, along with expert-annotated segmentation masks, radiotherapy dose distribution for a subset of patients, and comprehensive clinical metadata. This metadata includes TNM staging, HPV status, demographics (age and gender), long-term follow-up outcomes, survival times, censoring indicators, and treatment information. We demonstrate how this dataset can be used for three key clinical tasks: automated tumor segmentation, recurrence-free survival prediction, and HPV status classification, providing benchmark results using state-of-the-art deep learning models, including UNet, SegResNet, and multimodal prognostic frameworks.
>
---
#### [replaced 064] Towards a Universal Synthetic Video Detector: From Face or Background Manipulations to Fully AI-Generated Content
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.12278v2](http://arxiv.org/pdf/2412.12278v2)**

> **作者:** Rohit Kundu; Hao Xiong; Vishal Mohanty; Athula Balachandran; Amit K. Roy-Chowdhury
>
> **摘要:** Existing DeepFake detection techniques primarily focus on facial manipulations, such as face-swapping or lip-syncing. However, advancements in text-to-video (T2V) and image-to-video (I2V) generative models now allow fully AI-generated synthetic content and seamless background alterations, challenging face-centric detection methods and demanding more versatile approaches. To address this, we introduce the \underline{U}niversal \underline{N}etwork for \underline{I}dentifying \underline{T}ampered and synth\underline{E}tic videos (\texttt{UNITE}) model, which, unlike traditional detectors, captures full-frame manipulations. \texttt{UNITE} extends detection capabilities to scenarios without faces, non-human subjects, and complex background modifications. It leverages a transformer-based architecture that processes domain-agnostic features extracted from videos via the SigLIP-So400M foundation model. Given limited datasets encompassing both facial/background alterations and T2V/I2V content, we integrate task-irrelevant data alongside standard DeepFake datasets in training. We further mitigate the model's tendency to over-focus on faces by incorporating an attention-diversity (AD) loss, which promotes diverse spatial attention across video frames. Combining AD loss with cross-entropy improves detection performance across varied contexts. Comparative evaluations demonstrate that \texttt{UNITE} outperforms state-of-the-art detectors on datasets (in cross-data settings) featuring face/background manipulations and fully synthetic T2V/I2V videos, showcasing its adaptability and generalizable detection capabilities.
>
---
#### [replaced 065] Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.00062v2](http://arxiv.org/pdf/2509.00062v2)**

> **作者:** Justin Jung
>
> **备注:** Comments: 6 pages, LaTeX; typos corrected, figure added
>
> **摘要:** Generating realistic sparse multi-category 3D voxel structures is difficult due to the cubic memory scaling of voxel structures and moreover the significant class imbalance caused by sparsity. We introduce Scaffold Diffusion, a generative model designed for sparse multi-category 3D voxel structures. By treating voxels as tokens, Scaffold Diffusion uses a discrete diffusion language model to generate 3D voxel structures. We show that discrete diffusion language models can be extended beyond inherently sequential domains such as text to generate spatially coherent 3D structures. We evaluate on Minecraft house structures from the 3D-Craft dataset and demonstrate that, unlike prior baselines and an auto-regressive formulation, Scaffold Diffusion produces realistic and coherent structures even when trained on data with over 98% sparsity. We provide an interactive viewer where readers can visualize generated samples and the generation process: https://scaffold.deepexploration.org/
>
---
#### [replaced 066] C-DiffDet+: Fusing Global Scene Context with Generative Denoising for High-Fidelity Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00578v2](http://arxiv.org/pdf/2509.00578v2)**

> **作者:** Abdellah Zakaria Sellam; Ilyes Benaissa; Salah Eddine Bekhouche; Abdenour Hadid; Vito Renó; Cosimo Distante
>
> **摘要:** Fine-grained object detection in challenging visual domains, such as vehicle damage assessment, presents a formidable challenge even for human experts to resolve reliably. While DiffusionDet has advanced the state-of-the-art through conditional denoising diffusion, its performance remains limited by local feature conditioning in context-dependent scenarios. We address this fundamental limitation by introducing Context-Aware Fusion (CAF), which leverages cross-attention mechanisms to integrate global scene context with local proposal features directly. The global context is generated using a separate dedicated encoder that captures comprehensive environmental information, enabling each object proposal to attend to scene-level understanding. Our framework significantly enhances the generative detection paradigm by enabling each object proposal to attend to comprehensive environmental information. Experimental results demonstrate an improvement over state-of-the-art models on the CarDD benchmark, establishing new performance benchmarks for context-aware object detection in fine-grained domains
>
---
