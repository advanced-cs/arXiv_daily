# 计算机视觉 cs.CV

- **最新发布 79 篇**

- **更新 36 篇**

## 最新发布

#### [new 001] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining
- **分类: cs.CV**

- **简介: 该论文针对工业异常检测中ImageNet预训练特征不适用的问题，提出ADPretrain框架，通过在专用数据集RealIAD上使用角度与范数对比损失学习异常感知表征，显著提升多种检测方法性能。**

- **链接: [http://arxiv.org/pdf/2511.05245v1](http://arxiv.org/pdf/2511.05245v1)**

> **作者:** Xincheng Yao; Yan Luo; Zefeng Qian; Chongyang Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** The current mainstream and state-of-the-art anomaly detection (AD) methods are substantially established on pretrained feature networks yielded by ImageNet pretraining. However, regardless of supervised or self-supervised pretraining, the pretraining process on ImageNet does not match the goal of anomaly detection (i.e., pretraining in natural images doesn't aim to distinguish between normal and abnormal). Moreover, natural images and industrial image data in AD scenarios typically have the distribution shift. The two issues can cause ImageNet-pretrained features to be suboptimal for AD tasks. To further promote the development of the AD field, pretrained representations specially for AD tasks are eager and very valuable. To this end, we propose a novel AD representation learning framework specially designed for learning robust and discriminative pretrained representations for industrial anomaly detection. Specifically, closely surrounding the goal of anomaly detection (i.e., focus on discrepancies between normals and anomalies), we propose angle- and norm-oriented contrastive losses to maximize the angle size and norm difference between normal and abnormal features simultaneously. To avoid the distribution shift from natural images to AD images, our pretraining is performed on a large-scale AD dataset, RealIAD. To further alleviate the potential shift between pretraining data and downstream AD datasets, we learn the pretrained AD representations based on the class-generalizable representation, residual features. For evaluation, based on five embedding-based AD methods, we simply replace their original features with our pretrained representations. Extensive experiments on five AD datasets and five backbones consistently show the superiority of our pretrained features. The code is available at https://github.com/xcyao00/ADPretrain.
>
---
#### [new 002] Dense Motion Captioning
- **分类: cs.CV; I.2.10; I.4.8; I.5.4**

- **简介: 论文提出“密集动作字幕生成”任务，解决3D人体动作理解不足的问题，构建了首个大规模复杂动作数据集CompMo，并设计DEMO模型，实现动作时序定位与自然语言描述生成，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.05369v1](http://arxiv.org/pdf/2511.05369v1)**

> **作者:** Shiyao Xu; Benedetta Liberatori; Gül Varol; Paolo Rota
>
> **备注:** 12 pages, 5 figures, accepted to 3DV 2026
>
> **摘要:** Recent advances in 3D human motion and language integration have primarily focused on text-to-motion generation, leaving the task of motion understanding relatively unexplored. We introduce Dense Motion Captioning, a novel task that aims to temporally localize and caption actions within 3D human motion sequences. Current datasets fall short in providing detailed temporal annotations and predominantly consist of short sequences featuring few actions. To overcome these limitations, we present the Complex Motion Dataset (CompMo), the first large-scale dataset featuring richly annotated, complex motion sequences with precise temporal boundaries. Built through a carefully designed data generation pipeline, CompMo includes 60,000 motion sequences, each composed of multiple actions ranging from at least two to ten, accurately annotated with their temporal extents. We further present DEMO, a model that integrates a large language model with a simple motion adapter, trained to generate dense, temporally grounded captions. Our experiments show that DEMO substantially outperforms existing methods on CompMo as well as on adapted benchmarks, establishing a robust baseline for future research in 3D motion understanding and captioning.
>
---
#### [new 003] Pressure2Motion: Hierarchical Motion Synthesis from Ground Pressure with Text Guidance
- **分类: cs.CV**

- **简介: Pressure2Motion提出一种从地面压力序列和文本提示合成人体运动的新方法，解决传统动作捕捉依赖昂贵设备的问题，通过双层特征提取与分层扩散模型，实现无设备、隐私友好的高精度运动生成。**

- **链接: [http://arxiv.org/pdf/2511.05038v1](http://arxiv.org/pdf/2511.05038v1)**

> **作者:** Zhengxuan Li; Qinhui Yang; Yiyu Zhuang; Chuan Guo; Xinxin Zuo; Xiaoxiao Long; Yao Yao; Xun Cao; Qiu Shen; Hao Zhu
>
> **摘要:** We present Pressure2Motion, a novel motion capture algorithm that synthesizes human motion from a ground pressure sequence and text prompt. It eliminates the need for specialized lighting setups, cameras, or wearable devices, making it suitable for privacy-preserving, low-light, and low-cost motion capture scenarios. Such a task is severely ill-posed due to the indeterminate nature of the pressure signals to full-body motion. To address this issue, we introduce Pressure2Motion, a generative model that leverages pressure features as input and utilizes a text prompt as a high-level guiding constraint. Specifically, our model utilizes a dual-level feature extractor that accurately interprets pressure data, followed by a hierarchical diffusion model that discerns broad-scale movement trajectories and subtle posture adjustments. Both the physical cues gained from the pressure sequence and the semantic guidance derived from descriptive texts are leveraged to guide the motion generation with precision. To the best of our knowledge, Pressure2Motion is a pioneering work in leveraging both pressure data and linguistic priors for motion generation, and the established MPL benchmark is the first benchmark for this task. Experiments show our method generates high-fidelity, physically plausible motions, establishing a new state-of-the-art for this task. The codes and benchmarks will be publicly released upon publication.
>
---
#### [new 004] Canonical Space Representation for 4D Panoptic Segmentation of Articulated Objects
- **分类: cs.CV; I.2.10; I.4.6; I.5.1; I.5.4**

- **简介: 该论文针对4D关节物体全景分割任务，提出Artic4D数据集与CanonSeg4D框架，通过将物体部分对齐至学习的规范空间，实现跨帧一致分割，提升动态场景下的分割精度。**

- **链接: [http://arxiv.org/pdf/2511.05356v1](http://arxiv.org/pdf/2511.05356v1)**

> **作者:** Manuel Gomes; Bogdan Raducanu; Miguel Oliveira
>
> **备注:** 32 pages, 6 figures, 4 tables, submitted to Expert Systems With Applications
>
> **摘要:** Articulated object perception presents significant challenges in computer vision, particularly because most existing methods ignore temporal dynamics despite the inherently dynamic nature of such objects. The use of 4D temporal data has not been thoroughly explored in articulated object perception and remains unexamined for panoptic segmentation. The lack of a benchmark dataset further hurt this field. To this end, we introduce Artic4D as a new dataset derived from PartNet Mobility and augmented with synthetic sensor data, featuring 4D panoptic annotations and articulation parameters. Building on this dataset, we propose CanonSeg4D, a novel 4D panoptic segmentation framework. This approach explicitly estimates per-frame offsets mapping observed object parts to a learned canonical space, thereby enhancing part-level segmentation. The framework employs this canonical representation to achieve consistent alignment of object parts across sequential frames. Comprehensive experiments on Artic4D demonstrate that the proposed CanonSeg4D outperforms state of the art approaches in panoptic segmentation accuracy in more complex scenarios. These findings highlight the effectiveness of temporal modeling and canonical alignment in dynamic object understanding, and pave the way for future advances in 4D articulated object perception.
>
---
#### [new 005] EventFlow: Real-Time Neuromorphic Event-Driven Classification of Two-Phase Boiling Flow Regimes
- **分类: cs.CV; 76T10, 68T07; I.2.10; I.4.8; I.4.9**

- **简介: 论文提出EventFlow框架，利用神经形态传感器的事件数据实时分类两相沸腾流型，解决传统光学方法延迟高、计算重的问题。基于事件的LSTM模型实现97.6%准确率与0.28ms低延迟，支持稳定实时反馈。**

- **链接: [http://arxiv.org/pdf/2511.05467v1](http://arxiv.org/pdf/2511.05467v1)**

> **作者:** Sanghyeon Chang; Srikar Arani; Nishant Sai Nuthalapati; Youngjoon Suh; Nicholas Choi; Siavash Khodakarami; Md Rakibul Hasan Roni; Nenad Miljkovic; Aparna Chandramowlishwaran; Yoonjin Won
>
> **备注:** 19 pages, 6 figures, Under review in Droplet (Manuscript ID: DRO-2025-0045.R1)
>
> **摘要:** Flow boiling is an efficient heat transfer mechanism capable of dissipating high heat loads with minimal temperature variation, making it an ideal thermal management method. However, sudden shifts between flow regimes can disrupt thermal performance and system reliability, highlighting the need for accurate and low-latency real-time monitoring. Conventional optical imaging methods are limited by high computational demands and insufficient temporal resolution, making them inadequate for capturing transient flow behavior. To address this, we propose a real-time framework based on signals from neuromorphic sensors for flow regime classification. Neuromorphic sensors detect changes in brightness at individual pixels, which typically correspond to motion at edges, enabling fast and efficient detection without full-frame reconstruction, providing event-based information. We develop five classification models using both traditional image data and event-based data, demonstrating that models leveraging event data outperform frame-based approaches due to their sensitivity to dynamic flow features. Among these models, the event-based long short-term memory model provides the best balance between accuracy and speed, achieving 97.6% classification accuracy with a processing time of 0.28 ms. Our asynchronous processing pipeline supports continuous, low-latency predictions and delivers stable output through a majority voting mechanisms, enabling reliable real-time feedback for experimental control and intelligent thermal management.
>
---
#### [new 006] From Linear Probing to Joint-Weighted Token Hierarchy: A Foundation Model Bridging Global and Cellular Representations in Biomarker Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出JWTH模型，解决病理基础模型忽视细胞形态的问题，通过联合加权令牌层次结构融合全局与细胞级表征，提升生物标志物检测的准确率与可解释性。**

- **链接: [http://arxiv.org/pdf/2511.05150v1](http://arxiv.org/pdf/2511.05150v1)**

> **作者:** Jingsong Liu; Han Li; Nassir Navab; Peter J. Schüffler
>
> **摘要:** AI-based biomarkers can infer molecular features directly from hematoxylin & eosin (H&E) slides, yet most pathology foundation models (PFMs) rely on global patch-level embeddings and overlook cell-level morphology. We present a PFM model, JWTH (Joint-Weighted Token Hierarchy), which integrates large-scale self-supervised pretraining with cell-centric post-tuning and attention pooling to fuse local and global tokens. Across four tasks involving four biomarkers and eight cohorts, JWTH achieves up to 8.3% higher balanced accuracy and 1.2% average improvement over prior PFMs, advancing interpretable and robust AI-based biomarker detection in digital pathology.
>
---
#### [new 007] CPO: Condition Preference Optimization for Controllable Image Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Condition Preference Optimization（CPO），用于提升文本到图像生成的可控性。通过优化控制条件而非生成图像的偏好，避免了DPO中图像质量等干扰因素，显著提升分割、姿态、边缘等多类控制任务的精度，且计算更高效。**

- **链接: [http://arxiv.org/pdf/2511.04753v1](http://arxiv.org/pdf/2511.04753v1)**

> **作者:** Zonglin Lyu; Ming Li; Xinxin Liu; Chen Chen
>
> **摘要:** To enhance controllability in text-to-image generation, ControlNet introduces image-based control signals, while ControlNet++ improves pixel-level cycle consistency between generated images and the input control signal. To avoid the prohibitive cost of back-propagating through the sampling process, ControlNet++ optimizes only low-noise timesteps (e.g., $t < 200$) using a single-step approximation, which not only ignores the contribution of high-noise timesteps but also introduces additional approximation errors. A straightforward alternative for optimizing controllability across all timesteps is Direct Preference Optimization (DPO), a fine-tuning method that increases model preference for more controllable images ($I^{w}$) over less controllable ones ($I^{l}$). However, due to uncertainty in generative models, it is difficult to ensure that win--lose image pairs differ only in controllability while keeping other factors, such as image quality, fixed. To address this, we propose performing preference learning over control conditions rather than generated images. Specifically, we construct winning and losing control signals, $\mathbf{c}^{w}$ and $\mathbf{c}^{l}$, and train the model to prefer $\mathbf{c}^{w}$. This method, which we term \textit{Condition Preference Optimization} (CPO), eliminates confounding factors and yields a low-variance training objective. Our approach theoretically exhibits lower contrastive loss variance than DPO and empirically achieves superior results. Moreover, CPO requires less computation and storage for dataset curation. Extensive experiments show that CPO significantly improves controllability over the state-of-the-art ControlNet++ across multiple control types: over $10\%$ error rate reduction in segmentation, $70$--$80\%$ in human pose, and consistent $2$--$5\%$ reductions in edge and depth maps.
>
---
#### [new 008] Deep learning models are vulnerable, but adversarial examples are even more vulnerable
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究对抗样本的脆弱性，发现其对遮挡更敏感，提出SMCE指标量化置信度波动，并设计SWM-AED检测方法，有效识别对抗样本，提升模型鲁棒性，避免传统对抗训练的过拟合问题。**

- **链接: [http://arxiv.org/pdf/2511.05073v1](http://arxiv.org/pdf/2511.05073v1)**

> **作者:** Jun Li; Yanwei Xu; Keran Li; Xiaoli Zhang
>
> **备注:** 25 pages,12 figures
>
> **摘要:** Understanding intrinsic differences between adversarial examples and clean samples is key to enhancing DNN robustness and detection against adversarial attacks. This study first empirically finds that image-based adversarial examples are notably sensitive to occlusion. Controlled experiments on CIFAR-10 used nine canonical attacks (e.g., FGSM, PGD) to generate adversarial examples, paired with original samples for evaluation. We introduce Sliding Mask Confidence Entropy (SMCE) to quantify model confidence fluctuation under occlusion. Using 1800+ test images, SMCE calculations supported by Mask Entropy Field Maps and statistical distributions show adversarial examples have significantly higher confidence volatility under occlusion than originals. Based on this, we propose Sliding Window Mask-based Adversarial Example Detection (SWM-AED), which avoids catastrophic overfitting of conventional adversarial training. Evaluations across classifiers and attacks on CIFAR-10 demonstrate robust performance, with accuracy over 62% in most cases and up to 96.5%.
>
---
#### [new 009] Cross-domain EEG-based Emotion Recognition with Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文提出EmotionCLIP，将EEG情绪识别转化为EEG-文本对比匹配任务，设计SST-LegoViT提取多维特征，利用对比学习提升跨被试与跨时间泛化能力，在SEED数据集上取得领先准确率。**

- **链接: [http://arxiv.org/pdf/2511.05293v1](http://arxiv.org/pdf/2511.05293v1)**

> **作者:** Rui Yan; Yibo Li; Han Ding; Fei Wang
>
> **备注:** 5 pages
>
> **摘要:** Electroencephalogram (EEG)-based emotion recognition is vital for affective computing but faces challenges in feature utilization and cross-domain generalization. This work introduces EmotionCLIP, which reformulates recognition as an EEG-text matching task within the CLIP framework. A tailored backbone, SST-LegoViT, captures spatial, spectral, and temporal features using multi-scale convolution and Transformer modules. Experiments on SEED and SEED-IV datasets show superior cross-subject accuracies of 88.69% and 73.50%, and cross-time accuracies of 88.46% and 77.54%, outperforming existing models. Results demonstrate the effectiveness of multimodal contrastive learning for robust EEG emotion recognition.
>
---
#### [new 010] Learning Fourier shapes to probe the geometric world of deep neural networks
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出一种基于傅里叶级数的可微框架，用优化几何形状探查深度神经网络的几何感知能力，揭示形状可作为语义载体、解释工具和新型对抗攻击手段，填补了DNN几何理解的空白。**

- **链接: [http://arxiv.org/pdf/2511.04970v1](http://arxiv.org/pdf/2511.04970v1)**

> **作者:** Jian Wang; Yixing Yong; Haixia Bi; Lijun He; Fan Li
>
> **备注:** 20 pages, 5 figures
>
> **摘要:** While both shape and texture are fundamental to visual recognition, research on deep neural networks (DNNs) has predominantly focused on the latter, leaving their geometric understanding poorly probed. Here, we show: first, that optimized shapes can act as potent semantic carriers, generating high-confidence classifications from inputs defined purely by their geometry; second, that they are high-fidelity interpretability tools that precisely isolate a model's salient regions; and third, that they constitute a new, generalizable adversarial paradigm capable of deceiving downstream visual tasks. This is achieved through an end-to-end differentiable framework that unifies a powerful Fourier series to parameterize arbitrary shapes, a winding number-based mapping to translate them into the pixel grid required by DNNs, and signal energy constraints that enhance optimization efficiency while ensuring physically plausible shapes. Our work provides a versatile framework for probing the geometric world of DNNs and opens new frontiers for challenging and understanding machine perception.
>
---
#### [new 011] $\mathbf{S^2LM}$: Towards Semantic Steganography via Large Language Models
- **分类: cs.CV; cs.CR**

- **简介: 论文提出语义隐写任务，利用大语言模型（LLM）将句子级语义信息嵌入图像，解决传统隐写仅支持比特级低语义内容的问题，并构建基准IVT与模型S²LM，实现高语义隐蔽通信。**

- **链接: [http://arxiv.org/pdf/2511.05319v1](http://arxiv.org/pdf/2511.05319v1)**

> **作者:** Huanqi Wu; Huangbiao Xu; Runfeng Xie; Jiaxin Cai; Kaixin Zhang; Xiao Ke
>
> **备注:** 35 Pages, 20 Figures
>
> **摘要:** Although steganography has made significant advancements in recent years, it still struggles to embed semantically rich, sentence-level information into carriers. However, in the era of AIGC, the capacity of steganography is more critical than ever. In this work, we present Sentence-to-Image Steganography, an instance of Semantic Steganography, a novel task that enables the hiding of arbitrary sentence-level messages within a cover image. Furthermore, we establish a benchmark named Invisible Text (IVT), comprising a diverse set of sentence-level texts as secret messages for evaluation. Finally, we present $\mathbf{S^2LM}$: Semantic Steganographic Language Model, which utilizes large language models (LLMs) to embed high-level textual information, such as sentences or even paragraphs, into images. Unlike traditional bit-level counterparts, $\mathrm{S^2LM}$ enables the integration of semantically rich content through a newly designed pipeline in which the LLM is involved throughout the entire process. Both quantitative and qualitative experiments demonstrate that our method effectively unlocks new semantic steganographic capabilities for LLMs. The source code will be released soon.
>
---
#### [new 012] Dynamic Residual Encoding with Slide-Level Contrastive Learning for End-to-End Whole Slide Image Representation
- **分类: cs.CV; cs.AI; I.4.9; I.2.10**

- **简介: 该论文面向全切片图像（WSI）端到端表征学习，解决因图像规模大导致的梯度计算难题。提出动态残差编码与滑片级对比学习（DRE-SLCL），利用内存库融合采样与历史特征，提升癌症分型、识别与突变预测性能。**

- **链接: [http://arxiv.org/pdf/2511.05034v1](http://arxiv.org/pdf/2511.05034v1)**

> **作者:** Jing Jin; Xu Liu; Te Gao; Zhihong Shi; Yixiong Liang; Ruiqing Zheng; Hulin Kuang; Min Zeng; Shichao Kan
>
> **备注:** 8pages, 3figures, published to ACM Digital Library
>
> **摘要:** Whole Slide Image (WSI) representation is critical for cancer subtyping, cancer recognition and mutation prediction.Training an end-to-end WSI representation model poses significant challenges, as a standard gigapixel slide can contain tens of thousands of image tiles, making it difficult to compute gradients of all tiles in a single mini-batch due to current GPU limitations. To address this challenge, we propose a method of dynamic residual encoding with slide-level contrastive learning (DRE-SLCL) for end-to-end WSI representation. Our approach utilizes a memory bank to store the features of tiles across all WSIs in the dataset. During training, a mini-batch usually contains multiple WSIs. For each WSI in the batch, a subset of tiles is randomly sampled and their features are computed using a tile encoder. Then, additional tile features from the same WSI are selected from the memory bank. The representation of each individual WSI is generated using a residual encoding technique that incorporates both the sampled features and those retrieved from the memory bank. Finally, the slide-level contrastive loss is computed based on the representations and histopathology reports ofthe WSIs within the mini-batch. Experiments conducted over cancer subtyping, cancer recognition, and mutation prediction tasks proved the effectiveness of the proposed DRE-SLCL method.
>
---
#### [new 013] Global 3D Reconstruction of Clouds & Tropical Cyclones
- **分类: cs.CV; physics.ao-ph**

- **简介: 该论文提出一种基于预训练-微调框架的机器学习方法，首次实现全球范围内从2D卫星图像重建3D云结构，尤其针对热带气旋强风暴，解决观测缺失与结构解析难题，提升强度预报能力。**

- **链接: [http://arxiv.org/pdf/2511.04773v1](http://arxiv.org/pdf/2511.04773v1)**

> **作者:** Shirin Ermis; Cesar Aybar; Lilli Freischem; Stella Girtsou; Kyriaki-Margarita Bintsi; Emiliano Diaz Salas-Porras; Michael Eisinger; William Jones; Anna Jungbluth; Benoit Tremblay
>
> **摘要:** Accurate forecasting of tropical cyclones (TCs) remains challenging due to limited satellite observations probing TC structure and difficulties in resolving cloud properties involved in TC intensification. Recent research has demonstrated the capabilities of machine learning methods for 3D cloud reconstruction from satellite observations. However, existing approaches have been restricted to regions where TCs are uncommon, and are poorly validated for intense storms. We introduce a new framework, based on a pre-training--fine-tuning pipeline, that learns from multiple satellites with global coverage to translate 2D satellite imagery into 3D cloud maps of relevant cloud properties. We apply our model to a custom-built TC dataset to evaluate performance in the most challenging and relevant conditions. We show that we can - for the first time - create global instantaneous 3D cloud maps and accurately reconstruct the 3D structure of intense storms. Our model not only extends available satellite observations but also provides estimates when observations are missing entirely. This is crucial for advancing our understanding of TC intensification and improving forecasts.
>
---
#### [new 014] FreeControl: Efficient, Training-Free Structural Control via One-Step Attention Extraction
- **分类: cs.CV**

- **简介: FreeControl提出一种无训练的扩散模型结构控制方法，通过单步注意力提取与潜变量解耦，实现高效、高保真的语义结构引导，支持多源图像组合控制，仅增5%计算开销。**

- **链接: [http://arxiv.org/pdf/2511.05219v1](http://arxiv.org/pdf/2511.05219v1)**

> **作者:** Jiang Lin; Xinyu Chen; Song Wu; Zhiqiu Zhang; Jizhi Zhang; Ye Wang; Qiang Tang; Qian Wang; Jian Yang; Zili Yi
>
> **备注:** Accepted by NIPS 2025
>
> **摘要:** Controlling the spatial and semantic structure of diffusion-generated images remains a challenge. Existing methods like ControlNet rely on handcrafted condition maps and retraining, limiting flexibility and generalization. Inversion-based approaches offer stronger alignment but incur high inference cost due to dual-path denoising. We present FreeControl, a training-free framework for semantic structural control in diffusion models. Unlike prior methods that extract attention across multiple timesteps, FreeControl performs one-step attention extraction from a single, optimally chosen key timestep and reuses it throughout denoising. This enables efficient structural guidance without inversion or retraining. To further improve quality and stability, we introduce Latent-Condition Decoupling (LCD): a principled separation of the key timestep and the noised latent used in attention extraction. LCD provides finer control over attention quality and eliminates structural artifacts. FreeControl also supports compositional control via reference images assembled from multiple sources - enabling intuitive scene layout design and stronger prompt alignment. FreeControl introduces a new paradigm for test-time control, enabling structurally and semantically aligned, visually coherent generation directly from raw images, with the flexibility for intuitive compositional design and compatibility with modern diffusion models at approximately 5 percent additional cost.
>
---
#### [new 015] SnowyLane: Robust Lane Detection on Snow-covered Rural Roads Using Infrastructural Elements
- **分类: cs.CV**

- **简介: 该论文提出SnowyLane方法，针对积雪道路车道线缺失问题，通过检测路侧指示杆间接推断车道轨迹，采用贝塞尔曲线建模，并构建了8万帧积雪场景合成数据集，提升恶劣天气下车道检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.05108v1](http://arxiv.org/pdf/2511.05108v1)**

> **作者:** Jörg Gamerdinger; Benedict Wetzel; Patrick Schulz; Sven Teufel; Oliver Bringmann
>
> **摘要:** Lane detection for autonomous driving in snow-covered environments remains a major challenge due to the frequent absence or occlusion of lane markings. In this paper, we present a novel, robust and realtime capable approach that bypasses the reliance on traditional lane markings by detecting roadside features,specifically vertical roadside posts called delineators, as indirect lane indicators. Our method first perceives these posts, then fits a smooth lane trajectory using a parameterized Bezier curve model, leveraging spatial consistency and road geometry. To support training and evaluation in these challenging scenarios, we introduce SnowyLane, a new synthetic dataset containing 80,000 annotated frames capture winter driving conditions, with varying snow coverage, and lighting conditions. Compared to state-of-the-art lane detection systems, our approach demonstrates significantly improved robustness in adverse weather, particularly in cases with heavy snow occlusion. This work establishes a strong foundation for reliable lane detection in winter scenarios and contributes a valuable resource for future research in all-weather autonomous driving. The dataset is available at https://ekut-es.github.io/snowy-lane
>
---
#### [new 016] Photo Dating by Facial Age Aggregation
- **分类: cs.CV**

- **简介: 该论文提出“照片年代推断”任务，通过聚合图像中多人脸的年龄与身份信息，结合职业时间先验，估计拍摄年份。发布160万标注人脸数据集CSFD-1.6M，显著超越场景基线方法。**

- **链接: [http://arxiv.org/pdf/2511.05464v1](http://arxiv.org/pdf/2511.05464v1)**

> **作者:** Jakub Paplham; Vojtech Franc
>
> **摘要:** We introduce a novel method for Photo Dating which estimates the year a photograph was taken by leveraging information from the faces of people present in the image. To facilitate this research, we publicly release CSFD-1.6M, a new dataset containing over 1.6 million annotated faces, primarily from movie stills, with identity and birth year annotations. Uniquely, our dataset provides annotations for multiple individuals within a single image, enabling the study of multi-face information aggregation. We propose a probabilistic framework that formally combines visual evidence from modern face recognition and age estimation models, and career-based temporal priors to infer the photo capture year. Our experiments demonstrate that aggregating evidence from multiple faces consistently improves the performance and the approach significantly outperforms strong, scene-based baselines, particularly for images containing several identifiable individuals.
>
---
#### [new 017] OregairuChar: A Benchmark Dataset for Character Appearance Frequency Analysis in My Teen Romantic Comedy SNAFU
- **分类: cs.CV; cs.AI**

- **简介: 论文提出OregairuChar数据集，用于分析动漫《我的青春恋爱物语果然有问题》中角色出场频率，解决角色 prominence 的量化研究问题。通过标注1600帧图像，评估目标检测模型，实现角色随时间的出现模式分析。**

- **链接: [http://arxiv.org/pdf/2511.05263v1](http://arxiv.org/pdf/2511.05263v1)**

> **作者:** Qi Sun; Dingju Zhou; Lina Zhang
>
> **摘要:** The analysis of character appearance frequency is essential for understanding narrative structure, character prominence, and story progression in anime. In this work, we introduce OregairuChar, a benchmark dataset designed for appearance frequency analysis in the anime series My Teen Romantic Comedy SNAFU. The dataset comprises 1600 manually selected frames from the third season, annotated with 2860 bounding boxes across 11 main characters. OregairuChar captures diverse visual challenges, including occlusion, pose variation, and inter-character similarity, providing a realistic basis for appearance-based studies. To enable quantitative research, we benchmark several object detection models on the dataset and leverage their predictions for fine-grained, episode-level analysis of character presence over time. This approach reveals patterns of character prominence and their evolution within the narrative. By emphasizing appearance frequency, OregairuChar serves as a valuable resource for exploring computational narrative dynamics and character-centric storytelling in stylized media.
>
---
#### [new 018] Medical Referring Image Segmentation via Next-Token Mask Prediction
- **分类: cs.CV**

- **简介: 该论文提出NTP-MRISeg，将医学指代表达图像分割（MRIS）重构为自回归next-token预测任务，统一图像、文本与掩码编码，无需复杂融合模块，并提出NkTP、TCL和HET策略提升分割精度，实现端到端最优性能。**

- **链接: [http://arxiv.org/pdf/2511.05044v1](http://arxiv.org/pdf/2511.05044v1)**

> **作者:** Xinyu Chen; Yiran Wang; Gaoyang Pang; Jiafu Hao; Chentao Yue; Luping Zhou; Yonghui Li
>
> **备注:** This work has been submitted to the IEEE Transactions on Medical Imaging for possible publication
>
> **摘要:** Medical Referring Image Segmentation (MRIS) involves segmenting target regions in medical images based on natural language descriptions. While achieving promising results, recent approaches usually involve complex design of multimodal fusion or multi-stage decoders. In this work, we propose NTP-MRISeg, a novel framework that reformulates MRIS as an autoregressive next-token prediction task over a unified multimodal sequence of tokenized image, text, and mask representations. This formulation streamlines model design by eliminating the need for modality-specific fusion and external segmentation models, supports a unified architecture for end-to-end training. It also enables the use of pretrained tokenizers from emerging large-scale multimodal models, enhancing generalization and adaptability. More importantly, to address challenges under this formulation-such as exposure bias, long-tail token distributions, and fine-grained lesion edges-we propose three novel strategies: (1) a Next-k Token Prediction (NkTP) scheme to reduce cumulative prediction errors, (2) Token-level Contrastive Learning (TCL) to enhance boundary sensitivity and mitigate long-tail distribution effects, and (3) a memory-based Hard Error Token (HET) optimization strategy that emphasizes difficult tokens during training. Extensive experiments on the QaTa-COV19 and MosMedData+ datasets demonstrate that NTP-MRISeg achieves new state-of-the-art performance, offering a streamlined and effective alternative to traditional MRIS pipelines.
>
---
#### [new 019] GroupKAN: Rethinking Nonlinearity with Grouped Spline-based KAN Modeling for Efficient Medical Image Segmentation
- **分类: cs.CV**

- **简介: 论文面向医学图像分割任务，针对U-KAN计算复杂度高问题，提出GroupKAN，通过分组样条变换与激活模块降低复杂度，提升效率与可解释性，在三个医学数据集上以更少参数实现更高精度。**

- **链接: [http://arxiv.org/pdf/2511.05477v1](http://arxiv.org/pdf/2511.05477v1)**

> **作者:** Guojie Li; Anwar P. P. Abdul Majeed; Muhammad Ateeq; Anh Nguyen; Fan Zhang
>
> **摘要:** Medical image segmentation requires models that are accurate, lightweight, and interpretable. Convolutional architectures lack adaptive nonlinearity and transparent decision-making, whereas Transformer architectures are hindered by quadratic complexity and opaque attention mechanisms. U-KAN addresses these challenges using Kolmogorov-Arnold Networks, achieving higher accuracy than both convolutional and attention-based methods, fewer parameters than Transformer variants, and improved interpretability compared to conventional approaches. However, its O(C^2) complexity due to full-channel transformations limits its scalability as the number of channels increases. To overcome this, we introduce GroupKAN, a lightweight segmentation network that incorporates two novel, structured functional modules: (1) Grouped KAN Transform, which partitions channels into G groups for multivariate spline mappings, reducing complexity to O(C^2/G), and (2) Grouped KAN Activation, which applies shared spline-based mappings within each channel group for efficient, token-wise nonlinearity. Evaluated on three medical benchmarks (BUSI, GlaS, and CVC), GroupKAN achieves an average IoU of 79.80 percent, surpassing U-KAN by +1.11 percent while requiring only 47.6 percent of the parameters (3.02M vs 6.35M), and shows improved interpretability.
>
---
#### [new 020] Automatic segmentation of colorectal liver metastases for ultrasound-based navigated resection
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对超声引导肝切除中结直肠癌肝转移灶分割难题，提出基于裁剪区域的3D U-Net自动分割方法，实现近实时、高精度、无配准导航，显著提升效率并降低操作依赖。**

- **链接: [http://arxiv.org/pdf/2511.05253v1](http://arxiv.org/pdf/2511.05253v1)**

> **作者:** Tiziano Natali; Karin A. Olthof; Niels F. M. Kok; Koert F. D. Kuhlmann; Theo J. M. Ruers; Matteo Fusaglia
>
> **摘要:** Introduction: Accurate intraoperative delineation of colorectal liver metastases (CRLM) is crucial for achieving negative resection margins but remains challenging using intraoperative ultrasound (iUS) due to low contrast, noise, and operator dependency. Automated segmentation could enhance precision and efficiency in ultrasound-based navigation workflows. Methods: Eighty-five tracked 3D iUS volumes from 85 CRLM patients were used to train and evaluate a 3D U-Net implemented via the nnU-Net framework. Two variants were compared: one trained on full iUS volumes and another on cropped regions around tumors. Segmentation accuracy was assessed using Dice Similarity Coefficient (DSC), Hausdorff Distance (HDist.), and Relative Volume Difference (RVD) on retrospective and prospective datasets. The workflow was integrated into 3D Slicer for real-time intraoperative use. Results: The cropped-volume model significantly outperformed the full-volume model across all metrics (AUC-ROC = 0.898 vs 0.718). It achieved median DSC = 0.74, recall = 0.79, and HDist. = 17.1 mm comparable to semi-automatic segmentation but with ~4x faster execution (~ 1 min). Prospective intraoperative testing confirmed robust and consistent performance, with clinically acceptable accuracy for real-time surgical guidance. Conclusion: Automatic 3D segmentation of CRLM in iUS using a cropped 3D U-Net provides reliable, near real-time results with minimal operator input. The method enables efficient, registration-free ultrasound-based navigation for hepatic surgery, approaching expert-level accuracy while substantially reducing manual workload and procedure time.
>
---
#### [new 021] GSE: Evaluating Sticker Visual Semantic Similarity via a General Sticker Encoder
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出“贴纸语义相似性”任务，构建首个标注基准Triple-S，并设计轻量模型GSE，以解决贴纸符号化内容语义理解难题，提升情感分类与贴纸检索性能。**

- **链接: [http://arxiv.org/pdf/2511.04977v1](http://arxiv.org/pdf/2511.04977v1)**

> **作者:** Heng Er Metilda Chee; Jiayin Wang; Zhiqiang Guo; Weizhi Ma; Min Zhang
>
> **摘要:** Stickers have become a popular form of visual communication, yet understanding their semantic relationships remains challenging due to their highly diverse and symbolic content. In this work, we formally {define the Sticker Semantic Similarity task} and introduce {Triple-S}, the first benchmark for this task, consisting of 905 human-annotated positive and negative sticker pairs. Through extensive evaluation, we show that existing pretrained vision and multimodal models struggle to capture nuanced sticker semantics. To address this, we propose the {General Sticker Encoder (GSE)}, a lightweight and versatile model that learns robust sticker embeddings using both Triple-S and additional datasets. GSE achieves superior performance on unseen stickers, and demonstrates strong results on downstream tasks such as emotion classification and sticker-to-sticker retrieval. By releasing both Triple-S and GSE, we provide standardized evaluation tools and robust embeddings, enabling future research in sticker understanding, retrieval, and multimodal content generation. The Triple-S benchmark and GSE have been publicly released and are available here.
>
---
#### [new 022] DeepEyesV2: Toward Agentic Multimodal Model
- **分类: cs.CV; cs.AI**

- **简介: DeepEyesV2旨在构建能主动调用外部工具（如代码、搜索）的多模态智能体模型，解决传统模型工具使用不鲁棒的问题。提出两阶段训练法，构建专用数据集与RealX-Bench基准，验证其在感知、推理与搜索任务中的自适应工具调用能力。**

- **链接: [http://arxiv.org/pdf/2511.05271v1](http://arxiv.org/pdf/2511.05271v1)**

> **作者:** Jack Hong; Chenxiao Zhao; ChengLin Zhu; Weiheng Lu; Guohai Xu; Xing Yu
>
> **备注:** Homepage: https://visual-agent.github.io/
>
> **摘要:** Agentic multimodal models should not only comprehend text and images, but also actively invoke external tools, such as code execution environments and web search, and integrate these operations into reasoning. In this work, we introduce DeepEyesV2 and explore how to build an agentic multimodal model from the perspectives of data construction, training methods, and model evaluation. We observe that direct reinforcement learning alone fails to induce robust tool-use behavior. This phenomenon motivates a two-stage training pipeline: a cold-start stage to establish tool-use patterns, and reinforcement learning stage to further refine tool invocation. We curate a diverse, moderately challenging training dataset, specifically including examples where tool use is beneficial. We further introduce RealX-Bench, a comprehensive benchmark designed to evaluate real-world multimodal reasoning, which inherently requires the integration of multiple capabilities, including perception, search, and reasoning. We evaluate DeepEyesV2 on RealX-Bench and other representative benchmarks, demonstrating its effectiveness across real-world understanding, mathematical reasoning, and search-intensive tasks. Moreover, DeepEyesV2 exhibits task-adaptive tool invocation, tending to use image operations for perception tasks and numerical computations for reasoning tasks. Reinforcement learning further enables complex tool combinations and allows model to selectively invoke tools based on context. We hope our study can provide guidance for community in developing agentic multimodal models.
>
---
#### [new 023] Validating Vision Transformers for Otoscopy: Performance and Data-Leakage Effects
- **分类: cs.CV**

- **简介: 该论文研究Vision Transformers在耳镜图像诊断中的应用，旨在提升耳部疾病诊断准确率。发现初始高精度源于数据泄漏，修正后模型准确率降至83%左右，强调医疗AI中数据处理的严谨性。**

- **链接: [http://arxiv.org/pdf/2511.04872v1](http://arxiv.org/pdf/2511.04872v1)**

> **作者:** James Ndubuisi; Fernando Auat; Marta Vallejo
>
> **摘要:** This study evaluates the efficacy of vision transformer models, specifically Swin transformers, in enhancing the diagnostic accuracy of ear diseases compared to traditional convolutional neural networks. With a reported 27% misdiagnosis rate among specialist otolaryngologists, improving diagnostic accuracy is crucial. The research utilised a real-world dataset from the Department of Otolaryngology at the Clinical Hospital of the Universidad de Chile, comprising otoscopic videos of ear examinations depicting various middle and external ear conditions. Frames were selected based on the Laplacian and Shannon entropy thresholds, with blank frames removed. Initially, Swin v1 and Swin v2 transformer models achieved accuracies of 100% and 99.1%, respectively, marginally outperforming the ResNet model (99.5%). These results surpassed metrics reported in related studies. However, the evaluation uncovered a critical data leakage issue in the preprocessing step, affecting both this study and related research using the same raw dataset. After mitigating the data leakage, model performance decreased significantly. Corrected accuracies were 83% for both Swin v1 and Swin v2, and 82% for the ResNet model. This finding highlights the importance of rigorous data handling in machine learning studies, especially in medical applications. The findings indicate that while vision transformers show promise, it is essential to find an optimal balance between the benefits of advanced model architectures and those derived from effective data preprocessing. This balance is key to developing a reliable machine learning model for diagnosing ear diseases.
>
---
#### [new 024] Visual Spatial Tuning
- **分类: cs.CV**

- **简介: 该论文提出Visual Spatial Tuning（VST），解决VLMs空间能力弱且增强时损害通用性能的问题。通过构建大规模空间感知与推理数据集（VST-P/R）并采用渐进式训练，显著提升模型空间理解与推理能力，无损通用性能。**

- **链接: [http://arxiv.org/pdf/2511.05491v1](http://arxiv.org/pdf/2511.05491v1)**

> **作者:** Rui Yang; Ziyu Zhu; Yanwei Li; Jingjia Huang; Shen Yan; Siyuan Zhou; Zhe Liu; Xiangtai Li; Shuangye Li; Wenqian Wang; Yi Lin; Hengshuang Zhao
>
> **摘要:** Capturing spatial relationships from visual inputs is a cornerstone of human-like general intelligence. Several previous studies have tried to enhance the spatial awareness of Vision-Language Models (VLMs) by adding extra expert encoders, which brings extra overhead and usually harms general capabilities. To enhance the spatial ability in general architectures, we introduce Visual Spatial Tuning (VST), a comprehensive framework to cultivate VLMs with human-like visuospatial abilities, from spatial perception to reasoning. We first attempt to enhance spatial perception in VLMs by constructing a large-scale dataset termed VST-P, which comprises 4.1 million samples spanning 19 skills across single views, multiple images, and videos. Then, we present VST-R, a curated dataset with 135K samples that instruct models to reason in space. In particular, we adopt a progressive training pipeline: supervised fine-tuning to build foundational spatial knowledge, followed by reinforcement learning to further improve spatial reasoning abilities. Without the side-effect to general capabilities, the proposed VST consistently achieves state-of-the-art results on several spatial benchmarks, including $34.8\%$ on MMSI-Bench and $61.2\%$ on VSIBench. It turns out that the Vision-Language-Action models can be significantly enhanced with the proposed spatial tuning paradigm, paving the way for more physically grounded AI.
>
---
#### [new 025] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: LiveStar面向实时视频流理解，解决在线Video-LLMs响应延迟与语义不连贯问题，提出自适应解码、记忆压缩与响应时机判断机制，并构建OmniStar数据集，显著提升响应速度与语义准确性。**

- **链接: [http://arxiv.org/pdf/2511.05299v1](http://arxiv.org/pdf/2511.05299v1)**

> **作者:** Zhenyu Yang; Kairui Zhang; Yuhang Hu; Bing Wang; Shengsheng Qian; Bin Wen; Fan Yang; Tingting Gao; Weiming Dong; Changsheng Xu
>
> **备注:** NeurIPS 2025 Accepted
>
> **摘要:** Despite significant progress in Video Large Language Models (Video-LLMs) for offline video understanding, existing online Video-LLMs typically struggle to simultaneously process continuous frame-by-frame inputs and determine optimal response timing, often compromising real-time responsiveness and narrative coherence. To address these limitations, we introduce LiveStar, a pioneering live streaming assistant that achieves always-on proactive responses through adaptive streaming decoding. Specifically, LiveStar incorporates: (1) a training strategy enabling incremental video-language alignment for variable-length video streams, preserving temporal consistency across dynamically evolving frame sequences; (2) a response-silence decoding framework that determines optimal proactive response timing via a single forward pass verification; (3) memory-aware acceleration via peak-end memory compression for online inference on 10+ minute videos, combined with streaming key-value cache to achieve 1.53x faster inference. We also construct an OmniStar dataset, a comprehensive dataset for training and benchmarking that encompasses 15 diverse real-world scenarios and 5 evaluation tasks for online video understanding. Extensive experiments across three benchmarks demonstrate LiveStar's state-of-the-art performance, achieving an average 19.5% improvement in semantic correctness with 18.1% reduced timing difference compared to existing online Video-LLMs, while improving FPS by 12.0% across all five OmniStar tasks. Our model and dataset can be accessed at https://github.com/yzy-bupt/LiveStar.
>
---
#### [new 026] Geometry Denoising with Preferred Normal Vectors
- **分类: cs.CV; math.OC**

- **简介: 该论文提出一种基于优先法向量标签的几何去噪方法，将去噪与法向量聚类分割联合优化，通过总变差正则化和Split Bregman算法求解，利用二阶形状微积分更新顶点，提升几何重建质量。**

- **链接: [http://arxiv.org/pdf/2511.04848v1](http://arxiv.org/pdf/2511.04848v1)**

> **作者:** Manuel Weiß; Lukas Baumgärtner; Roland Herzog; Stephan Schmidt
>
> **摘要:** We introduce a new paradigm for geometry denoising using prior knowledge about the surface normal vector. This prior knowledge comes in the form of a set of preferred normal vectors, which we refer to as label vectors. A segmentation problem is naturally embedded in the denoising process. The segmentation is based on the similarity of the normal vector to the elements of the set of label vectors. Regularization is achieved by a total variation term. We formulate a split Bregman (ADMM) approach to solve the resulting optimization problem. The vertex update step is based on second-order shape calculus.
>
---
#### [new 027] PALM: A Dataset and Baseline for Learning Multi-subject Hand Prior
- **分类: cs.CV**

- **简介: 论文提出PALM数据集（13k手部扫描、90k多视角图像），解决个性化手部数字建模中数据匮乏与复杂几何建模难题，并构建PALM-Net基线模型，实现单图可重光照手部avatar个性化生成。**

- **链接: [http://arxiv.org/pdf/2511.05403v1](http://arxiv.org/pdf/2511.05403v1)**

> **作者:** Zicong Fan; Edoardo Remelli; David Dimond; Fadime Sener; Liuhao Ge; Bugra Tekin; Cem Keskin; Shreyas Hampali
>
> **摘要:** The ability to grasp objects, signal with gestures, and share emotion through touch all stem from the unique capabilities of human hands. Yet creating high-quality personalized hand avatars from images remains challenging due to complex geometry, appearance, and articulation, particularly under unconstrained lighting and limited views. Progress has also been limited by the lack of datasets that jointly provide accurate 3D geometry, high-resolution multiview imagery, and a diverse population of subjects. To address this, we present PALM, a large-scale dataset comprising 13k high-quality hand scans from 263 subjects and 90k multi-view images, capturing rich variation in skin tone, age, and geometry. To show its utility, we present a baseline PALM-Net, a multi-subject prior over hand geometry and material properties learned via physically based inverse rendering, enabling realistic, relightable single-image hand avatar personalization. PALM's scale and diversity make it a valuable real-world resource for hand modeling and related research.
>
---
#### [new 028] PreResQ-R1: Towards Fine-Grained Rank-and-Score Reinforcement Learning for Visual Quality Assessment via Preference-Response Disentangled Policy Optimization
- **分类: cs.CV**

- **简介: 该论文提出PreResQ-R1，面向视觉质量评估任务，解决现有方法推理浅层、评分不准问题，通过解耦偏好与响应的强化学习框架，实现细粒度评分与排序统一优化，在图像与视频质量评估上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2511.05393v1](http://arxiv.org/pdf/2511.05393v1)**

> **作者:** Zehui Feng; Tian Qiu; Tong Wu; Junxuan Li; Huayuan Xu; Ting Han
>
> **备注:** 27 pages, 14 figures, under review as a conference paper
>
> **摘要:** Visual Quality Assessment (QA) seeks to predict human perceptual judgments of visual fidelity. While recent multimodal large language models (MLLMs) show promise in reasoning about image and video quality, existing approaches mainly rely on supervised fine-tuning or rank-only objectives, resulting in shallow reasoning, poor score calibration, and limited cross-domain generalization. We propose PreResQ-R1, a Preference-Response Disentangled Reinforcement Learning framework that unifies absolute score regression and relative ranking consistency within a single reasoning-driven optimization scheme. Unlike prior QA methods, PreResQ-R1 introduces a dual-branch reward formulation that separately models intra-sample response coherence and inter-sample preference alignment, optimized via Group Relative Policy Optimization (GRPO). This design encourages fine-grained, stable, and interpretable chain-of-thought reasoning about perceptual quality. To extend beyond static imagery, we further design a global-temporal and local-spatial data flow strategy for Video Quality Assessment. Remarkably, with reinforcement fine-tuning on only 6K images and 28K videos, PreResQ-R1 achieves state-of-the-art results across 10 IQA and 5 VQA benchmarks under both SRCC and PLCC metrics, surpassing by margins of 5.30% and textbf2.15% in IQA task, respectively. Beyond quantitative gains, it produces human-aligned reasoning traces that reveal the perceptual cues underlying quality judgments. Code and model are available.
>
---
#### [new 029] 3D Gaussian Point Encoders
- **分类: cs.CV**

- **简介: 该论文提出3D高斯点编码器，用显式高斯混合替代PointNet等隐式表示，解决其计算低效问题，通过自然梯度与知识蒸馏优化，并融合高斯泼溅技术，显著提升速度与效率，实现CPU实时推理。**

- **链接: [http://arxiv.org/pdf/2511.04797v1](http://arxiv.org/pdf/2511.04797v1)**

> **作者:** Jim James; Ben Wilson; Simon Lucey; James Hays
>
> **备注:** 10 pages, 3 figures, 3 tables
>
> **摘要:** In this work, we introduce the 3D Gaussian Point Encoder, an explicit per-point embedding built on mixtures of learned 3D Gaussians. This explicit geometric representation for 3D recognition tasks is a departure from widely used implicit representations such as PointNet. However, it is difficult to learn 3D Gaussian encoders in end-to-end fashion with standard optimizers. We develop optimization techniques based on natural gradients and distillation from PointNets to find a Gaussian Basis that can reconstruct PointNet activations. The resulting 3D Gaussian Point Encoders are faster and more parameter efficient than traditional PointNets. As in the 3D reconstruction literature where there has been considerable interest in the move from implicit (e.g., NeRF) to explicit (e.g., Gaussian Splatting) representations, we can take advantage of computational geometry heuristics to accelerate 3D Gaussian Point Encoders further. We extend filtering techniques from 3D Gaussian Splatting to construct encoders that run 2.7 times faster as a comparable accuracy PointNet while using 46% less memory and 88% fewer FLOPs. Furthermore, we demonstrate the effectiveness of 3D Gaussian Point Encoders as a component in Mamba3D, running 1.27 times faster and achieving a reduction in memory and FLOPs by 42% and 54% respectively. 3D Gaussian Point Encoders are lightweight enough to achieve high framerates on CPU-only devices.
>
---
#### [new 030] What's on Your Plate? Inferring Chinese Cuisine Intake from Wearable IMUs
- **分类: cs.CV; cs.LG**

- **简介: 论文提出CuisineSense，通过智能手表与智能眼镜的IMU数据，识别中国菜摄入行为，解决传统方法偏差大、现有穿戴设备对中餐分类不足的问题，实现非侵入式饮食监测。**

- **链接: [http://arxiv.org/pdf/2511.05292v1](http://arxiv.org/pdf/2511.05292v1)**

> **作者:** Jiaxi Yin; Pengcheng Wang; Han Ding; Fei Wang
>
> **备注:** 5 pages
>
> **摘要:** Accurate food intake detection is vital for dietary monitoring and chronic disease prevention. Traditional self-report methods are prone to recall bias, while camera-based approaches raise concerns about privacy. Furthermore, existing wearable-based methods primarily focus on a limited number of food types, such as hamburgers and pizza, failing to address the vast diversity of Chinese cuisine. To bridge this gap, we propose CuisineSense, a system that classifies Chinese food types by integrating hand motion cues from a smartwatch with head dynamics from smart glasses. To filter out irrelevant daily activities, we design a two-stage detection pipeline. The first stage identifies eating states by distinguishing characteristic temporal patterns from non-eating behaviors. The second stage then conducts fine-grained food type recognition based on the motions captured during food intake. To evaluate CuisineSense, we construct a dataset comprising 27.5 hours of IMU recordings across 11 food categories and 10 participants. Experiments demonstrate that CuisineSense achieves high accuracy in both eating state detection and food classification, offering a practical solution for unobtrusive, wearable-based dietary monitoring.The system code is publicly available at https://github.com/joeeeeyin/CuisineSense.git.
>
---
#### [new 031] IndicVisionBench: Benchmarking Cultural and Multilingual Understanding in VLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出IndicVisionBench，首个聚焦印度次大陆的多语言、文化导向视觉语言模型评测基准，涵盖13种文化主题、10种印度语言及3类多模态任务，揭示现有模型在文化多样性场景中的显著性能差距。**

- **链接: [http://arxiv.org/pdf/2511.04727v1](http://arxiv.org/pdf/2511.04727v1)**

> **作者:** Ali Faraz; Akash; Shaharukh Khan; Raja Kolla; Akshat Patidar; Suranjan Goswami; Abhinav Ravi; Chandra Khatri; Shubham Agarwal
>
> **摘要:** Vision-language models (VLMs) have demonstrated impressive generalization across multimodal tasks, yet most evaluation benchmarks remain Western-centric, leaving open questions about their performance in culturally diverse and multilingual settings. To address this gap, we introduce IndicVisionBench, the first large-scale benchmark centered on the Indian subcontinent. Covering English and 10 Indian languages, our benchmark spans 3 multimodal tasks, including Optical Character Recognition (OCR), Multimodal Machine Translation (MMT), and Visual Question Answering (VQA), covering 6 kinds of question types. Our final benchmark consists of a total of ~5K images and 37K+ QA pairs across 13 culturally grounded topics. In addition, we release a paired parallel corpus of annotations across 10 Indic languages, creating a unique resource for analyzing cultural and linguistic biases in VLMs. We evaluate a broad spectrum of 8 models, from proprietary closed-source systems to open-weights medium and large-scale models. Our experiments reveal substantial performance gaps, underscoring the limitations of current VLMs in culturally diverse contexts. By centering cultural diversity and multilinguality, IndicVisionBench establishes a reproducible evaluation framework that paves the way for more inclusive multimodal research.
>
---
#### [new 032] The Potential of Copernicus Satellites for Disaster Response: Retrieving Building Damage from Sentinel-1 and Sentinel-2
- **分类: cs.CV**

- **简介: 该论文研究利用Copernicus计划的Sentinel-1/2中分辨率遥感影像进行建筑损毁检测，解决VHR影像稀缺下的快速灾损评估问题。构建了xBD-S12数据集，验证中分辨率数据的有效性，并发现复杂模型优势有限。**

- **链接: [http://arxiv.org/pdf/2511.05461v1](http://arxiv.org/pdf/2511.05461v1)**

> **作者:** Olivier Dietrich; Merlin Alfredsson; Emilia Arens; Nando Metzger; Torben Peters; Linus Scheibenreif; Jan Dirk Wegner; Konrad Schindler
>
> **摘要:** Natural disasters demand rapid damage assessment to guide humanitarian response. Here, we investigate whether medium-resolution Earth observation images from the Copernicus program can support building damage assessment, complementing very-high resolution imagery with often limited availability. We introduce xBD-S12, a dataset of 10,315 pre- and post-disaster image pairs from both Sentinel-1 and Sentinel-2, spatially and temporally aligned with the established xBD benchmark. In a series of experiments, we demonstrate that building damage can be detected and mapped rather well in many disaster scenarios, despite the moderate 10$\,$m ground sampling distance. We also find that, for damage mapping at that resolution, architectural sophistication does not seem to bring much advantage: more complex model architectures tend to struggle with generalization to unseen disasters, and geospatial foundation models bring little practical benefit. Our results suggest that Copernicus images are a viable data source for rapid, wide-area damage assessment and could play an important role alongside VHR imagery. We release the xBD-S12 dataset, code, and trained models to support further research.
>
---
#### [new 033] Semantic-Guided Natural Language and Visual Fusion for Cross-Modal Interaction Based on Tiny Object Detection
- **分类: cs.CV**

- **简介: 该论文面向小目标检测任务，提出一种语义引导的跨模态融合方法，结合BERT与PRB-FPN-Net等高效骨干网络，实现文本与视觉特征精准对齐，显著提升检测精度与效率，优于YOLO-World且参数更少。**

- **链接: [http://arxiv.org/pdf/2511.05474v1](http://arxiv.org/pdf/2511.05474v1)**

> **作者:** Xian-Hong Huang; Hui-Kai Su; Chi-Chia Sun; Jun-Wei Hsieh
>
> **摘要:** This paper introduces a cutting-edge approach to cross-modal interaction for tiny object detection by combining semantic-guided natural language processing with advanced visual recognition backbones. The proposed method integrates the BERT language model with the CNN-based Parallel Residual Bi-Fusion Feature Pyramid Network (PRB-FPN-Net), incorporating innovative backbone architectures such as ELAN, MSP, and CSP to optimize feature extraction and fusion. By employing lemmatization and fine-tuning techniques, the system aligns semantic cues from textual inputs with visual features, enhancing detection precision for small and complex objects. Experimental validation using the COCO and Objects365 datasets demonstrates that the model achieves superior performance. On the COCO2017 validation set, it attains a 52.6% average precision (AP), outperforming YOLO-World significantly while maintaining half the parameter consumption of Transformer-based models like GLIP. Several test on different of backbones such ELAN, MSP, and CSP further enable efficient handling of multi-scale objects, ensuring scalability and robustness in resource-constrained environments. This study underscores the potential of integrating natural language understanding with advanced backbone architectures, setting new benchmarks in object detection accuracy, efficiency, and adaptability to real-world challenges.
>
---
#### [new 034] Shared Latent Representation for Joint Text-to-Audio-Visual Synthesis
- **分类: cs.CV**

- **简介: 该论文提出一种文本到口型视频合成框架，通过共享潜在表征联合生成语音与面部动作，解决TTS预测特征与真实音频间的分布偏移问题，实现无真实语音输入下的高精度音视同步与身份保持。**

- **链接: [http://arxiv.org/pdf/2511.05432v1](http://arxiv.org/pdf/2511.05432v1)**

> **作者:** Dogucan Yaman; Seymanur Akti; Fevziye Irem Eyiokur; Alexander Waibel
>
> **摘要:** We propose a text-to-talking-face synthesis framework leveraging latent speech representations from HierSpeech++. A Text-to-Vec module generates Wav2Vec2 embeddings from text, which jointly condition speech and face generation. To handle distribution shifts between clean and TTS-predicted features, we adopt a two-stage training: pretraining on Wav2Vec2 embeddings and finetuning on TTS outputs. This enables tight audio-visual alignment, preserves speaker identity, and produces natural, expressive speech and synchronized facial motion without ground-truth audio at inference. Experiments show that conditioning on TTS-predicted latent features outperforms cascaded pipelines, improving both lip-sync and visual realism.
>
---
#### [new 035] CLM: Removing the GPU Memory Barrier for 3D Gaussian Splatting
- **分类: cs.CV; D.4; I.3.2; I.3.7**

- **简介: CLM针对3D高斯溅射内存超限问题，提出一种CPU-GPU协同的内存管理机制，通过感知访问模式实现高效数据流水与通信压缩，使消费级GPU可渲染亿级高斯点的大场景，突破显存瓶颈。**

- **链接: [http://arxiv.org/pdf/2511.04951v1](http://arxiv.org/pdf/2511.04951v1)**

> **作者:** Hexu Zhao; Xiwen Min; Xiaoteng Liu; Moonjun Gong; Yiming Li; Ang Li; Saining Xie; Jinyang Li; Aurojit Panda
>
> **备注:** Accepted to appear in the 2026 ACM International Conference on Architectural Support for Programming Languages and Operating Systems
>
> **摘要:** 3D Gaussian Splatting (3DGS) is an increasingly popular novel view synthesis approach due to its fast rendering time, and high-quality output. However, scaling 3DGS to large (or intricate) scenes is challenging due to its large memory requirement, which exceed most GPU's memory capacity. In this paper, we describe CLM, a system that allows 3DGS to render large scenes using a single consumer-grade GPU, e.g., RTX4090. It does so by offloading Gaussians to CPU memory, and loading them into GPU memory only when necessary. To reduce performance and communication overheads, CLM uses a novel offloading strategy that exploits observations about 3DGS's memory access pattern for pipelining, and thus overlap GPU-to-CPU communication, GPU computation and CPU computation. Furthermore, we also exploit observation about the access pattern to reduce communication volume. Our evaluation shows that the resulting implementation can render a large scene that requires 100 million Gaussians on a single RTX4090 and achieve state-of-the-art reconstruction quality.
>
---
#### [new 036] Knowledge-based anomaly detection for identifying network-induced shape artifacts
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于知识的异常检测方法，用于识别合成医学图像中的网络诱导形态伪影。通过分析解剖边界角度梯度分布，结合隔离森林，实现对合成数据质量的自动评估，提升合成数据的临床可用性。**

- **链接: [http://arxiv.org/pdf/2511.04729v1](http://arxiv.org/pdf/2511.04729v1)**

> **作者:** Rucha Deshpande; Tahsin Rahman; Miguel Lago; Adarsh Subbaswamy; Jana G. Delfino; Ghada Zamzmi; Elim Thompson; Aldo Badano; Seyed Kahaki
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Synthetic data provides a promising approach to address data scarcity for training machine learning models; however, adoption without proper quality assessments may introduce artifacts, distortions, and unrealistic features that compromise model performance and clinical utility. This work introduces a novel knowledge-based anomaly detection method for detecting network-induced shape artifacts in synthetic images. The introduced method utilizes a two-stage framework comprising (i) a novel feature extractor that constructs a specialized feature space by analyzing the per-image distribution of angle gradients along anatomical boundaries, and (ii) an isolation forest-based anomaly detector. We demonstrate the effectiveness of the method for identifying network-induced shape artifacts in two synthetic mammography datasets from models trained on CSAW-M and VinDr-Mammo patient datasets respectively. Quantitative evaluation shows that the method successfully concentrates artifacts in the most anomalous partition (1st percentile), with AUC values of 0.97 (CSAW-syn) and 0.91 (VMLO-syn). In addition, a reader study involving three imaging scientists confirmed that images identified by the method as containing network-induced shape artifacts were also flagged by human readers with mean agreement rates of 66% (CSAW-syn) and 68% (VMLO-syn) for the most anomalous partition, approximately 1.5-2 times higher than the least anomalous partition. Kendall-Tau correlations between algorithmic and human rankings were 0.45 and 0.43 for the two datasets, indicating reasonable agreement despite the challenging nature of subtle artifact detection. This method is a step forward in the responsible use of synthetic data, as it allows developers to evaluate synthetic images for known anatomic constraints and pinpoint and address specific issues to improve the overall quality of a synthetic dataset.
>
---
#### [new 037] 4D3R: Motion-Aware Neural Reconstruction and Rendering of Dynamic Scenes from Monocular Videos
- **分类: cs.CV; cs.AI**

- **简介: 4D3R针对单目视频动态场景的无位姿神经重建与渲染任务，提出双阶段框架：利用基础模型初估位姿与几何，再通过运动感知束调整与高效高斯溅射建模动态运动，显著提升重建质量并降低计算成本。**

- **链接: [http://arxiv.org/pdf/2511.05229v1](http://arxiv.org/pdf/2511.05229v1)**

> **作者:** Mengqi Guo; Bo Xu; Yanyan Li; Gim Hee Lee
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Novel view synthesis from monocular videos of dynamic scenes with unknown camera poses remains a fundamental challenge in computer vision and graphics. While recent advances in 3D representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown promising results for static scenes, they struggle with dynamic content and typically rely on pre-computed camera poses. We present 4D3R, a pose-free dynamic neural rendering framework that decouples static and dynamic components through a two-stage approach. Our method first leverages 3D foundational models for initial pose and geometry estimation, followed by motion-aware refinement. 4D3R introduces two key technical innovations: (1) a motion-aware bundle adjustment (MA-BA) module that combines transformer-based learned priors with SAM2 for robust dynamic object segmentation, enabling more accurate camera pose refinement; and (2) an efficient Motion-Aware Gaussian Splatting (MA-GS) representation that uses control points with a deformation field MLP and linear blend skinning to model dynamic motion, significantly reducing computational cost while maintaining high-quality reconstruction. Extensive experiments on real-world dynamic datasets demonstrate that our approach achieves up to 1.8dB PSNR improvement over state-of-the-art methods, particularly in challenging scenarios with large dynamic objects, while reducing computational requirements by 5x compared to previous dynamic scene representations.
>
---
#### [new 038] DeepForgeSeal: Latent Space-Driven Semi-Fragile Watermarking for Deepfake Detection Using Multi-Agent Adversarial Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 论文提出DeepForgeSeal，一种基于潜空间与多智能体对抗强化学习的半脆弱水印方法，用于深度伪造检测。通过动态平衡鲁棒性与敏感性，提升对未知伪造与恶意篡改的检测能力，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2511.04949v1](http://arxiv.org/pdf/2511.04949v1)**

> **作者:** Tharindu Fernando; Clinton Fookes; Sridha Sridharan
>
> **摘要:** Rapid advances in generative AI have led to increasingly realistic deepfakes, posing growing challenges for law enforcement and public trust. Existing passive deepfake detectors struggle to keep pace, largely due to their dependence on specific forgery artifacts, which limits their ability to generalize to new deepfake types. Proactive deepfake detection using watermarks has emerged to address the challenge of identifying high-quality synthetic media. However, these methods often struggle to balance robustness against benign distortions with sensitivity to malicious tampering. This paper introduces a novel deep learning framework that harnesses high-dimensional latent space representations and the Multi-Agent Adversarial Reinforcement Learning (MAARL) paradigm to develop a robust and adaptive watermarking approach. Specifically, we develop a learnable watermark embedder that operates in the latent space, capturing high-level image semantics, while offering precise control over message encoding and extraction. The MAARL paradigm empowers the learnable watermarking agent to pursue an optimal balance between robustness and fragility by interacting with a dynamic curriculum of benign and malicious image manipulations simulated by an adversarial attacker agent. Comprehensive evaluations on the CelebA and CelebA-HQ benchmarks reveal that our method consistently outperforms state-of-the-art approaches, achieving improvements of over 4.5% on CelebA and more than 5.3% on CelebA-HQ under challenging manipulation scenarios.
>
---
#### [new 039] How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对3D点云Transformer冗余问题，提出gitmerge3D方法，通过全局图融合将令牌数减少90-95%，显著提升效率而不损性能，首次揭示其过度令牌化现象，推动高效3D基础架构设计。**

- **链接: [http://arxiv.org/pdf/2511.05449v1](http://arxiv.org/pdf/2511.05449v1)**

> **作者:** Tuan Anh Tran; Duy M. H. Nguyen; Hoai-Chau Tran; Michael Barz; Khoa D. Doan; Roger Wattenhofer; Ngo Anh Vien; Mathias Niepert; Daniel Sonntag; Paul Swoboda
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Recent advances in 3D point cloud transformers have led to state-of-the-art results in tasks such as semantic segmentation and reconstruction. However, these models typically rely on dense token representations, incurring high computational and memory costs during training and inference. In this work, we present the finding that tokens are remarkably redundant, leading to substantial inefficiency. We introduce gitmerge3D, a globally informed graph token merging method that can reduce the token count by up to 90-95% while maintaining competitive performance. This finding challenges the prevailing assumption that more tokens inherently yield better performance and highlights that many current models are over-tokenized and under-optimized for scalability. We validate our method across multiple 3D vision tasks and show consistent improvements in computational efficiency. This work is the first to assess redundancy in large-scale 3D transformer models, providing insights into the development of more efficient 3D foundation architectures. Our code and checkpoints are publicly available at https://gitmerge3d.github.io
>
---
#### [new 040] Accurate online action and gesture recognition system using detectors and Deep SPD Siamese Networks
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文提出一种基于SPD矩阵与Siamese网络的在线骨架动作识别系统，解决传统方法不适用于连续流数据的问题，通过检测器定位动作区间并分类器识别动作，在手势与行为识别任务中实现高精度在线识别。**

- **链接: [http://arxiv.org/pdf/2511.05250v1](http://arxiv.org/pdf/2511.05250v1)**

> **作者:** Mohamed Sanim Akremi; Rim Slama; Hedi Tabia
>
> **摘要:** Online continuous motion recognition is a hot topic of research since it is more practical in real life application cases. Recently, Skeleton-based approaches have become increasingly popular, demonstrating the power of using such 3D temporal data. However, most of these works have focused on segment-based recognition and are not suitable for the online scenarios. In this paper, we propose an online recognition system for skeleton sequence streaming composed from two main components: a detector and a classifier, which use a Semi-Positive Definite (SPD) matrix representation and a Siamese network. The powerful statistical representations for the skeletal data given by the SPD matrices and the learning of their semantic similarity by the Siamese network enable the detector to predict time intervals of the motions throughout an unsegmented sequence. In addition, they ensure the classifier capability to recognize the motion in each predicted interval. The proposed detector is flexible and able to identify the kinetic state continuously. We conduct extensive experiments on both hand gesture and body action recognition benchmarks to prove the accuracy of our online recognition system which in most cases outperforms state-of-the-art performances.
>
---
#### [new 041] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start
- **分类: cs.CV**

- **简介: 该论文针对真实恶劣天气图像恢复问题，构建高保真数据集HFLS-Weather，提出双层强化学习框架，通过局部优化与全局调度实现无配对监督的自适应恢复，显著提升泛化性能。**

- **链接: [http://arxiv.org/pdf/2511.05095v1](http://arxiv.org/pdf/2511.05095v1)**

> **作者:** Fuyang Liu; Jiaqi Xu; Xiaowei Hu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Adverse weather severely impairs real-world visual perception, while existing vision models trained on synthetic data with fixed parameters struggle to generalize to complex degradations. To address this, we first construct HFLS-Weather, a physics-driven, high-fidelity dataset that simulates diverse weather phenomena, and then design a dual-level reinforcement learning framework initialized with HFLS-Weather for cold-start training. Within this framework, at the local level, weather-specific restoration models are refined through perturbation-driven image quality optimization, enabling reward-based learning without paired supervision; at the global level, a meta-controller dynamically orchestrates model selection and execution order according to scene degradation. This framework enables continuous adaptation to real-world conditions and achieves state-of-the-art performance across a wide range of adverse weather scenarios. Code is available at https://github.com/xxclfy/AgentRL-Real-Weather
>
---
#### [new 042] Walk the Lines 2: Contour Tracking for Detailed Segmentation
- **分类: cs.CV**

- **简介: 论文提出Walk the Lines 2（WtL2），一种改进的轮廓追踪算法，用于红外与RGB图像中物体的精细分割，替代传统NMS，通过追踪生成单像素宽闭合轮廓，提升IoU与细节保留，拓展了原方法在多模态场景的应用。**

- **链接: [http://arxiv.org/pdf/2511.05210v1](http://arxiv.org/pdf/2511.05210v1)**

> **作者:** André Peter Kelm; Max Braeschke; Emre Gülsoylu; Simone Frintrop
>
> **备注:** 11 pages, 6 figures. Accepted at CAIP 2025: 21st International Conference on Computer Analysis of Images and Patterns, Las Palmas de Gran Canaria, Spain, September 22-25, 2025. To appear in: Proceedings Part I, Lecture Notes in Computer Science (LNCS), Springer Nature Switzerland
>
> **摘要:** This paper presents Walk the Lines 2 (WtL2), a unique contour tracking algorithm specifically adapted for detailed segmentation of infrared (IR) ships and various objects in RGB.1 This extends the original Walk the Lines (WtL) [12], which focused solely on detailed ship segmentation in color. These innovative WtLs can replace the standard non-maximum suppression (NMS) by using contour tracking to refine the object contour until a 1-pixel-wide closed shape can be binarized, forming a segmentable area in foreground-background scenarios. WtL2 broadens the application range of WtL beyond its original scope, adapting to IR and expanding to diverse objects within the RGB context. To achieve IR segmentation, we adapt its input, the object contour detector, to IR ships. In addition, the algorithm is enhanced to process a wide range of RGB objects, outperforming the latest generation of contour-based methods when achieving a closed object contour, offering high peak Intersection over Union (IoU) with impressive details. This positions WtL2 as a compelling method for specialized applications that require detailed segmentation or high-quality samples, potentially accelerating progress in several niche areas of image segmentation.
>
---
#### [new 043] No Pose Estimation? No Problem: Pose-Agnostic and Instance-Aware Test-Time Adaptation for Monocular Depth Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对单目深度估计在域偏移场景下的适应难题，提出PITTA框架，无需位姿信息，通过实例感知掩码和边缘增强实现高效测试时自适应，显著提升模型在动态环境中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.05055v1](http://arxiv.org/pdf/2511.05055v1)**

> **作者:** Mingyu Sung; Hyeonmin Choe; Il-Min Kim; Sangseok Yun; Jae Mo Kang
>
> **摘要:** Monocular depth estimation (MDE), inferring pixel-level depths in single RGB images from a monocular camera, plays a crucial and pivotal role in a variety of AI applications demanding a three-dimensional (3D) topographical scene. In the real-world scenarios, MDE models often need to be deployed in environments with different conditions from those for training. Test-time (domain) adaptation (TTA) is one of the compelling and practical approaches to address the issue. Although there have been notable advancements in TTA for MDE, particularly in a self-supervised manner, existing methods are still ineffective and problematic when applied to diverse and dynamic environments. To break through this challenge, we propose a novel and high-performing TTA framework for MDE, named PITTA. Our approach incorporates two key innovative strategies: (i) pose-agnostic TTA paradigm for MDE and (ii) instance-aware image masking. Specifically, PITTA enables highly effective TTA on a pretrained MDE network in a pose-agnostic manner without resorting to any camera pose information. Besides, our instance-aware masking strategy extracts instance-wise masks for dynamic objects (e.g., vehicles, pedestrians, etc.) from a segmentation mask produced by a pretrained panoptic segmentation network, by removing static objects including background components. To further boost performance, we also present a simple yet effective edge extraction methodology for the input image (i.e., a single monocular image) and depth map. Extensive experimental evaluations on DrivingStereo and Waymo datasets with varying environmental conditions demonstrate that our proposed framework, PITTA, surpasses the existing state-of-the-art techniques with remarkable performance improvements in MDE during TTA.
>
---
#### [new 044] A Dual-stage Prompt-driven Privacy-preserving Paradigm for Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对行人重识别（Re-ID）中的隐私问题，提出双阶段提示驱动隐私保护框架DPPP，通过生成大规模虚拟数据集GenePerson并设计提示解耦机制PDM，实现域不变特征学习，显著提升模型泛化性能。**

- **链接: [http://arxiv.org/pdf/2511.05092v1](http://arxiv.org/pdf/2511.05092v1)**

> **作者:** Ruolin Li; Min Liu; Yuan Bian; Zhaoyang Li; Yuzhen Li; Xueping Wang; Yaonan Wang
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** With growing concerns over data privacy, researchers have started using virtual data as an alternative to sensitive real-world images for training person re-identification (Re-ID) models. However, existing virtual datasets produced by game engines still face challenges such as complex construction and poor domain generalization, making them difficult to apply in real scenarios. To address these challenges, we propose a Dual-stage Prompt-driven Privacy-preserving Paradigm (DPPP). In the first stage, we generate rich prompts incorporating multi-dimensional attributes such as pedestrian appearance, illumination, and viewpoint that drive the diffusion model to synthesize diverse data end-to-end, building a large-scale virtual dataset named GenePerson with 130,519 images of 6,641 identities. In the second stage, we propose a Prompt-driven Disentanglement Mechanism (PDM) to learn domain-invariant generalization features. With the aid of contrastive learning, we employ two textual inversion networks to map images into pseudo-words representing style and content, respectively, thereby constructing style-disentangled content prompts to guide the model in learning domain-invariant content features at the image level. Experiments demonstrate that models trained on GenePerson with PDM achieve state-of-the-art generalization performance, surpassing those on popular real and virtual Re-ID datasets.
>
---
#### [new 045] Pattern-Aware Diffusion Synthesis of fMRI/dMRI with Tissue and Microstructural Refinement
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PDS模型，解决fMRI与dMRI模态缺失问题，通过模式感知扩散框架与组织-微结构精炼网络，实现高保真跨模态合成，在多个数据集上达到SOTA性能并验证临床诊断价值。**

- **链接: [http://arxiv.org/pdf/2511.04963v1](http://arxiv.org/pdf/2511.04963v1)**

> **作者:** Xiongri Shen; Jiaqi Wang; Yi Zhong; Zhenxi Song; Leilei Zhao; Yichen Wei; Lingyan Liang; Shuqiang Wang; Baiying Lei; Demao Deng; Zhiguo Zhang
>
> **摘要:** Magnetic resonance imaging (MRI), especially functional MRI (fMRI) and diffusion MRI (dMRI), is essential for studying neurodegenerative diseases. However, missing modalities pose a major barrier to their clinical use. Although GAN- and diffusion model-based approaches have shown some promise in modality completion, they remain limited in fMRI-dMRI synthesis due to (1) significant BOLD vs. diffusion-weighted signal differences between fMRI and dMRI in time/gradient axis, and (2) inadequate integration of disease-related neuroanatomical patterns during generation. To address these challenges, we propose PDS, introducing two key innovations: (1) a pattern-aware dual-modal 3D diffusion framework for cross-modality learning, and (2) a tissue refinement network integrated with a efficient microstructure refinement to maintain structural fidelity and fine details. Evaluated on OASIS-3, ADNI, and in-house datasets, our method achieves state-of-the-art results, with PSNR/SSIM scores of 29.83 dB/90.84\% for fMRI synthesis (+1.54 dB/+4.12\% over baselines) and 30.00 dB/77.55\% for dMRI synthesis (+1.02 dB/+2.2\%). In clinical validation, the synthesized data show strong diagnostic performance, achieving 67.92\%/66.02\%/64.15\% accuracy (NC vs. MCI vs. AD) in hybrid real-synthetic experiments. Code is available in \href{https://github.com/SXR3015/PDS}{PDS GitHub Repository}
>
---
#### [new 046] EETnet: a CNN for Gaze Detection and Tracking for Smart-Eyewear
- **分类: cs.CV**

- **简介: EETnet提出一种轻量级CNN，用于基于事件相机的瞳孔检测与追踪，解决嵌入式设备上低功耗实时眼动追踪问题，设计了分类与回归双模型，并实现微控制器部署。**

- **链接: [http://arxiv.org/pdf/2511.04779v1](http://arxiv.org/pdf/2511.04779v1)**

> **作者:** Andrea Aspesi; Andrea Simpsi; Aaron Tognoli; Simone Mentasti; Luca Merigo; Matteo Matteucci
>
> **备注:** International Joint Conference on Neural Networks (IJCNN), 2025
>
> **摘要:** Event-based cameras are becoming a popular solution for efficient, low-power eye tracking. Due to the sparse and asynchronous nature of event data, they require less processing power and offer latencies in the microsecond range. However, many existing solutions are limited to validation on powerful GPUs, with no deployment on real embedded devices. In this paper, we present EETnet, a convolutional neural network designed for eye tracking using purely event-based data, capable of running on microcontrollers with limited resources. Additionally, we outline a methodology to train, evaluate, and quantize the network using a public dataset. Finally, we propose two versions of the architecture: a classification model that detects the pupil on a grid superimposed on the original image, and a regression model that operates at the pixel level.
>
---
#### [new 047] TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: TimeSearch-R提出一种自验证强化学习框架，将视频时序搜索融入文本-视频推理过程，解决传统方法搜索策略僵化、推理不完整问题，通过GRPO-CSV提升搜索完备性，在多个长视频理解基准上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2511.05489v1](http://arxiv.org/pdf/2511.05489v1)**

> **作者:** Junwen Pan; Qizhe Zhang; Rui Zhang; Ming Lu; Xin Wan; Yuan Zhang; Chang Liu; Qi She
>
> **备注:** 22 pages, 17 figures. Official code: https://github.com/Time-Search/TimeSearch-R
>
> **摘要:** Temporal search aims to identify a minimal set of relevant frames from tens of thousands based on a given query, serving as a foundation for accurate long-form video understanding. Existing works attempt to progressively narrow the search space. However, these approaches typically rely on a hand-crafted search process, lacking end-to-end optimization for learning optimal search strategies. In this paper, we propose TimeSearch-R, which reformulates temporal search as interleaved text-video thinking, seamlessly integrating searching video clips into the reasoning process through reinforcement learning (RL). However, applying RL training methods, such as Group Relative Policy Optimization (GRPO), to video reasoning can result in unsupervised intermediate search decisions. This leads to insufficient exploration of the video content and inconsistent logical reasoning. To address these issues, we introduce GRPO with Completeness Self-Verification (GRPO-CSV), which gathers searched video frames from the interleaved reasoning process and utilizes the same policy model to verify the adequacy of searched frames, thereby improving the completeness of video reasoning. Additionally, we construct datasets specifically designed for the SFT cold-start and RL training of GRPO-CSV, filtering out samples with weak temporal dependencies to enhance task difficulty and improve temporal search capabilities. Extensive experiments demonstrate that TimeSearch-R achieves significant improvements on temporal search benchmarks such as Haystack-LVBench and Haystack-Ego4D, as well as long-form video understanding benchmarks like VideoMME and MLVU. Notably, TimeSearch-R establishes a new state-of-the-art on LongVideoBench with 4.1% improvement over the base model Qwen2.5-VL and 2.0% over the advanced video reasoning model Video-R1. Our code is available at https://github.com/Time-Search/TimeSearch-R.
>
---
#### [new 048] Data Efficiency and Transfer Robustness in Biomedical Image Segmentation: A Study of Redundancy and Forgetting with Cellpose
- **分类: cs.CV; cs.AI; cs.LG; I.2.10; I.4.6**

- **简介: 该论文研究生物医学图像分割中的数据效率与迁移鲁棒性问题，以Cellpose为对象，提出数据量化策略DQ，发现训练数据高度冗余，且跨域微调引发灾难性遗忘，通过选择性重放和域排序有效缓解。**

- **链接: [http://arxiv.org/pdf/2511.04803v1](http://arxiv.org/pdf/2511.04803v1)**

> **作者:** Shuo Zhao; Jianxu Chen
>
> **备注:** Accepted to IEEE BIBM 2025 Workshop; 6 pages; 4 figures; 5 tables; IEEEtran class. Code: https://github.com/MMV-Lab/biomedseg-efficiency
>
> **摘要:** Generalist biomedical image segmentation models such as Cellpose are increasingly applied across diverse imaging modalities and cell types. However, two critical challenges remain underexplored: (1) the extent of training data redundancy and (2) the impact of cross domain transfer on model retention. In this study, we conduct a systematic empirical analysis of these challenges using Cellpose as a case study. First, to assess data redundancy, we propose a simple dataset quantization (DQ) strategy for constructing compact yet diverse training subsets. Experiments on the Cyto dataset show that image segmentation performance saturates with only 10% of the data, revealing substantial redundancy and potential for training with minimal annotations. Latent space analysis using MAE embeddings and t-SNE confirms that DQ selected patches capture greater feature diversity than random sampling. Second, to examine catastrophic forgetting, we perform cross domain finetuning experiments and observe significant degradation in source domain performance, particularly when adapting from generalist to specialist domains. We demonstrate that selective DQ based replay reintroducing just 5-10% of the source data effectively restores source performance, while full replay can hinder target adaptation. Additionally, we find that training domain sequencing improves generalization and reduces forgetting in multi stage transfer. Our findings highlight the importance of data centric design in biomedical image segmentation and suggest that efficient training requires not only compact subsets but also retention aware learning strategies and informed domain ordering. The code is available at https://github.com/MMV-Lab/biomedseg-efficiency.
>
---
#### [new 049] Clinical-ComBAT: a diffusion-weighted MRI harmonization method for clinical applications
- **分类: cs.CV; stat.AP**

- **简介: 论文提出Clinical-ComBAT，一种面向临床的DW-MRI数据校正方法，解决多中心扫描偏差问题。通过非线性模型、独立站点校准和自适应先验，提升小样本临床数据的可比性与规范建模能力。**

- **链接: [http://arxiv.org/pdf/2511.04871v1](http://arxiv.org/pdf/2511.04871v1)**

> **作者:** Gabriel Girard; Manon Edde; Félix Dumais; Yoan David; Matthieu Dumont; Guillaume Theaud; Jean-Christophe Houde; Arnaud Boré; Maxime Descoteaux; Pierre-Marc Jodoin
>
> **备注:** 39 pages, 11 figures
>
> **摘要:** Diffusion-weighted magnetic resonance imaging (DW-MRI) derived scalar maps are effective for assessing neurodegenerative diseases and microstructural properties of white matter in large number of brain conditions. However, DW-MRI inherently limits the combination of data from multiple acquisition sites without harmonization to mitigate scanner-specific biases. While the widely used ComBAT method reduces site effects in research, its reliance on linear covariate relationships, homogeneous populations, fixed site numbers, and well populated sites constrains its clinical use. To overcome these limitations, we propose Clinical-ComBAT, a method designed for real-world clinical scenarios. Clinical-ComBAT harmonizes each site independently, enabling flexibility as new data and clinics are introduced. It incorporates a non-linear polynomial data model, site-specific harmonization referenced to a normative site, and variance priors adaptable to small cohorts. It further includes hyperparameter tuning and a goodness-of-fit metric for harmonization assessment. We demonstrate its effectiveness on simulated and real data, showing improved alignment of diffusion metrics and enhanced applicability for normative modeling.
>
---
#### [new 050] Learning to Restore Multi-Degraded Images via Ingredient Decoupling and Task-Aware Path Adaptation
- **分类: cs.CV**

- **简介: 该论文面向多退化图像恢复任务，解决单一模型难以处理复合退化（如雨、噪、雾共存）的问题。提出IMDNet，通过退化成分解耦与任务感知路径自适应，实现高效多退化联合恢复，兼顾单退化性能。**

- **链接: [http://arxiv.org/pdf/2511.04920v1](http://arxiv.org/pdf/2511.04920v1)**

> **作者:** Hu Gao; Xiaoning Lei; Ying Zhang; Xichen Xu; Guannan Jiang; Lizhuang Ma
>
> **摘要:** Image restoration (IR) aims to recover clean images from degraded observations. Despite remarkable progress, most existing methods focus on a single degradation type, whereas real-world images often suffer from multiple coexisting degradations, such as rain, noise, and haze coexisting in a single image, which limits their practical effectiveness. In this paper, we propose an adaptive multi-degradation image restoration network that reconstructs images by leveraging decoupled representations of degradation ingredients to guide path selection. Specifically, we design a degradation ingredient decoupling block (DIDBlock) in the encoder to separate degradation ingredients statistically by integrating spatial and frequency domain information, enhancing the recognition of multiple degradation types and making their feature representations independent. In addition, we present fusion block (FBlock) to integrate degradation information across all levels using learnable matrices. In the decoder, we further introduce a task adaptation block (TABlock) that dynamically activates or fuses functional branches based on the multi-degradation representation, flexibly selecting optimal restoration paths under diverse degradation conditions. The resulting tightly integrated architecture, termed IMDNet, is extensively validated through experiments, showing superior performance on multi-degradation restoration while maintaining strong competitiveness on single-degradation tasks.
>
---
#### [new 051] Another BRIXEL in the Wall: Towards Cheaper Dense Features
- **分类: cs.CV; cs.LG**

- **简介: 论文提出BRIXEL，一种轻量级知识蒸馏方法，解决DINOv3生成高分辨率稠密特征计算成本高的问题，让学生模型学习复现自身高分辨率特征，在固定分辨率下显著超越基线，大幅降低算力开销。**

- **链接: [http://arxiv.org/pdf/2511.05168v1](http://arxiv.org/pdf/2511.05168v1)**

> **作者:** Alexander Lappe; Martin A. Giese
>
> **摘要:** Vision foundation models achieve strong performance on both global and locally dense downstream tasks. Pretrained on large images, the recent DINOv3 model family is able to produce very fine-grained dense feature maps, enabling state-of-the-art performance. However, computing these feature maps requires the input image to be available at very high resolution, as well as large amounts of compute due to the squared complexity of the transformer architecture. To address these issues, we propose BRIXEL, a simple knowledge distillation approach that has the student learn to reproduce its own feature maps at higher resolution. Despite its simplicity, BRIXEL outperforms the baseline DINOv3 models by large margins on downstream tasks when the resolution is kept fixed. Moreover, it is able to produce feature maps that are very similar to those of the teacher at a fraction of the computational cost. Code and model weights are available at https://github.com/alexanderlappe/BRIXEL.
>
---
#### [new 052] SurgiATM: A Physics-Guided Plug-and-Play Model for Deep Learning-Based Smoke Removal in Laparoscopic Surgery
- **分类: cs.CV**

- **简介: 论文提出SurgiATM，一种无额外参数的物理引导插件模块，用于腹腔镜手术烟雾去除，融合物理模型与深度学习，提升现有去烟模型的精度与泛化性，无需修改原架构。**

- **链接: [http://arxiv.org/pdf/2511.05059v1](http://arxiv.org/pdf/2511.05059v1)**

> **作者:** Mingyu Sheng; Jianan Fan; Dongnan Liu; Guoyan Zheng; Ron Kikinis; Weidong Cai
>
> **备注:** 10 pages, 5 figures, 6 tables. Code available at https://github.com/MingyuShengSMY/SurgiATM
>
> **摘要:** During laparoscopic surgery, smoke generated by tissue cauterization can significantly degrade the visual quality of endoscopic frames, increasing the risk of surgical errors and hindering both clinical decision-making and computer-assisted visual analysis. Consequently, removing surgical smoke is critical to ensuring patient safety and maintaining operative efficiency. In this study, we propose the Surgical Atmospheric Model (SurgiATM) for surgical smoke removal. SurgiATM statistically bridges a physics-based atmospheric model and data-driven deep learning models, combining the superior generalizability of the former with the high accuracy of the latter. Furthermore, SurgiATM is designed as a lightweight, plug-and-play module that can be seamlessly integrated into diverse surgical desmoking architectures to enhance their accuracy and stability, better meeting clinical requirements. It introduces only two hyperparameters and no additional trainable weights, preserving the original network architecture with minimal computational and modification overhead. We conduct extensive experiments on three public surgical datasets with ten desmoking methods, involving multiple network architectures and covering diverse procedures, including cholecystectomy, partial nephrectomy, and diaphragm dissection. The results demonstrate that incorporating SurgiATM commonly reduces the restoration errors of existing models and relatively enhances their generalizability, without adding any trainable layers or weights. This highlights the convenience, low cost, effectiveness, and generalizability of the proposed method. The code for SurgiATM is released at https://github.com/MingyuShengSMY/SurgiATM.
>
---
#### [new 053] Early Alzheimer's Disease Detection from Retinal OCT Images: A UK Biobank Study
- **分类: cs.CV; cs.LG**

- **简介: 该论文首次利用深度学习直接分类视网膜OCT B扫图像，实现阿尔茨海默病早期预测。基于UK Biobank数据，通过数据增强与加权损失优化模型，ResNet-34达AUC 0.62，揭示了早期视网膜结构变化，为无创早筛提供基线。**

- **链接: [http://arxiv.org/pdf/2511.05106v1](http://arxiv.org/pdf/2511.05106v1)**

> **作者:** Yasemin Turkan; F. Boray Tek; M. Serdar Nazlı; Öykü Eren
>
> **摘要:** Alterations in retinal layer thickness, measurable using Optical Coherence Tomography (OCT), have been associated with neurodegenerative diseases such as Alzheimer's disease (AD). While previous studies have mainly focused on segmented layer thickness measurements, this study explored the direct classification of OCT B-scan images for the early detection of AD. To our knowledge, this is the first application of deep learning to raw OCT B-scans for AD prediction in the literature. Unlike conventional medical image classification tasks, early detection is more challenging than diagnosis because imaging precedes clinical diagnosis by several years. We fine-tuned and evaluated multiple pretrained models, including ImageNet-based networks and the OCT-specific RETFound transformer, using subject-level cross-validation datasets matched for age, sex, and imaging instances from the UK Biobank cohort. To reduce overfitting in this small, high-dimensional dataset, both standard and OCT-specific augmentation techniques were applied, along with a year-weighted loss function that prioritized cases diagnosed within four years of imaging. ResNet-34 produced the most stable results, achieving an AUC of 0.62 in the 4-year cohort. Although below the threshold for clinical application, our explainability analyses confirmed localized structural differences in the central macular subfield between the AD and control groups. These findings provide a baseline for OCT-based AD prediction, highlight the challenges of detecting subtle retinal biomarkers years before AD diagnosis, and point to the need for larger datasets and multimodal approaches.
>
---
#### [new 054] Rethinking Metrics and Diffusion Architecture for 3D Point Cloud Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文面向3D点云生成任务，指出传统评估指标（如CD）缺乏鲁棒性，提出密度感知CD和表面法向一致性新指标，并设计Diffusion Point Transformer架构，显著提升生成质量，实现SOTA。**

- **链接: [http://arxiv.org/pdf/2511.05308v1](http://arxiv.org/pdf/2511.05308v1)**

> **作者:** Matteo Bastico; David Ryckelynck; Laurent Corté; Yannick Tillier; Etienne Decencière
>
> **备注:** This paper has been accepted at International Conference on 3D Vision (3DV) 2026
>
> **摘要:** As 3D point clouds become a cornerstone of modern technology, the need for sophisticated generative models and reliable evaluation metrics has grown exponentially. In this work, we first expose that some commonly used metrics for evaluating generated point clouds, particularly those based on Chamfer Distance (CD), lack robustness against defects and fail to capture geometric fidelity and local shape consistency when used as quality indicators. We further show that introducing samples alignment prior to distance calculation and replacing CD with Density-Aware Chamfer Distance (DCD) are simple yet essential steps to ensure the consistency and robustness of point cloud generative model evaluation metrics. While existing metrics primarily focus on directly comparing 3D Euclidean coordinates, we present a novel metric, named Surface Normal Concordance (SNC), which approximates surface similarity by comparing estimated point normals. This new metric, when combined with traditional ones, provides a more comprehensive evaluation of the quality of generated samples. Finally, leveraging recent advancements in transformer-based models for point cloud analysis, such as serialized patch attention , we propose a new architecture for generating high-fidelity 3D structures, the Diffusion Point Transformer. We perform extensive experiments and comparisons on the ShapeNet dataset, showing that our model outperforms previous solutions, particularly in terms of quality of generated point clouds, achieving new state-of-the-art. Code available at https://github.com/matteo-bastico/DiffusionPointTransformer.
>
---
#### [new 055] MUSE: Multi-Scale Dense Self-Distillation for Nucleus Detection and Classification
- **分类: cs.CV**

- **简介: MUSE提出一种自监督方法，用于组织病理学中的核检测与分类，解决标注依赖强、未标注数据利用不足的问题，通过多尺度局部自蒸馏NuLo实现跨尺度表征学习，超越现有监督与基础模型。**

- **链接: [http://arxiv.org/pdf/2511.05170v1](http://arxiv.org/pdf/2511.05170v1)**

> **作者:** Zijiang Yang; Hanqing Chao; Bokai Zhao; Yelin Yang; Yunshuo Zhang; Dongmei Fu; Junping Zhang; Le Lu; Ke Yan; Dakai Jin; Minfeng Xu; Yun Bian; Hui Jiang
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Nucleus detection and classification (NDC) in histopathology analysis is a fundamental task that underpins a wide range of high-level pathology applications. However, existing methods heavily rely on labor-intensive nucleus-level annotations and struggle to fully exploit large-scale unlabeled data for learning discriminative nucleus representations. In this work, we propose MUSE (MUlti-scale denSE self-distillation), a novel self-supervised learning method tailored for NDC. At its core is NuLo (Nucleus-based Local self-distillation), a coordinate-guided mechanism that enables flexible local self-distillation based on predicted nucleus positions. By removing the need for strict spatial alignment between augmented views, NuLo allows critical cross-scale alignment, thus unlocking the capacity of models for fine-grained nucleus-level representation. To support MUSE, we design a simple yet effective encoder-decoder architecture and a large field-of-view semi-supervised fine-tuning strategy that together maximize the value of unlabeled pathology images. Extensive experiments on three widely used benchmarks demonstrate that MUSE effectively addresses the core challenges of histopathological NDC. The resulting models not only surpass state-of-the-art supervised baselines but also outperform generic pathology foundation models.
>
---
#### [new 056] Beta Distribution Learning for Reliable Roadway Crash Risk Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于卫星影像的贝塔分布深度学习模型，用于道路事故风险评估。解决传统方法忽略空间复杂性与不确定性的问题，实现高精度、可解释且带置信度的风险预测，提升交通安全决策可靠性。**

- **链接: [http://arxiv.org/pdf/2511.04886v1](http://arxiv.org/pdf/2511.04886v1)**

> **作者:** Ahmad Elallaf; Nathan Jacobs; Xinyue Ye; Mei Chen; Gongbo Liang
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Roadway traffic accidents represent a global health crisis, responsible for over a million deaths annually and costing many countries up to 3% of their GDP. Traditional traffic safety studies often examine risk factors in isolation, overlooking the spatial complexity and contextual interactions inherent in the built environment. Furthermore, conventional Neural Network-based risk estimators typically generate point estimates without conveying model uncertainty, limiting their utility in critical decision-making. To address these shortcomings, we introduce a novel geospatial deep learning framework that leverages satellite imagery as a comprehensive spatial input. This approach enables the model to capture the nuanced spatial patterns and embedded environmental risk factors that contribute to fatal crash risks. Rather than producing a single deterministic output, our model estimates a full Beta probability distribution over fatal crash risk, yielding accurate and uncertainty-aware predictions--a critical feature for trustworthy AI in safety-critical applications. Our model outperforms baselines by achieving a 17-23% improvement in recall, a key metric for flagging potential dangers, while delivering superior calibration. By providing reliable and interpretable risk assessments from satellite imagery alone, our method enables safer autonomous navigation and offers a highly scalable tool for urban planners and policymakers to enhance roadway safety equitably and cost-effectively.
>
---
#### [new 057] Splatography: Sparse multi-view dynamic Gaussian Splatting for filmmaking challenges
- **分类: cs.CV; cs.GR; cs.MM**

- **简介: 该论文提出Splatography，解决电影制作中稀疏视角下动态3D重建难题。通过前景/背景分解与分阶段训练，仅用稀疏掩码实现高保真、可分割的动态高斯泼溅重建，性能优于SOTA且模型更小。**

- **链接: [http://arxiv.org/pdf/2511.05152v1](http://arxiv.org/pdf/2511.05152v1)**

> **作者:** Adrian Azzarelli; Nantheera Anantrasirichai; David R Bull
>
> **摘要:** Deformable Gaussian Splatting (GS) accomplishes photorealistic dynamic 3-D reconstruction from dense multi-view video (MVV) by learning to deform a canonical GS representation. However, in filmmaking, tight budgets can result in sparse camera configurations, which limits state-of-the-art (SotA) methods when capturing complex dynamic features. To address this issue, we introduce an approach that splits the canonical Gaussians and deformation field into foreground and background components using a sparse set of masks for frames at t=0. Each representation is separately trained on different loss functions during canonical pre-training. Then, during dynamic training, different parameters are modeled for each deformation field following common filmmaking practices. The foreground stage contains diverse dynamic features so changes in color, position and rotation are learned. While, the background containing film-crew and equipment, is typically dimmer and less dynamic so only changes in point position are learned. Experiments on 3-D and 2.5-D entertainment datasets show that our method produces SotA qualitative and quantitative results; up to 3 PSNR higher with half the model size on 3-D scenes. Unlike the SotA and without the need for dense mask supervision, our method also produces segmented dynamic reconstructions including transparent and dynamic textures. Code and video comparisons are available online: https://interims-git.github.io/
>
---
#### [new 058] Challenges in 3D Data Synthesis for Training Neural Networks on Topological Features
- **分类: cs.CV**

- **简介: 该论文面向拓扑数据分析（TDA）中的神经网络训练，解决3D标注数据稀缺问题，提出基于Repulsive Surface算法生成可控拓扑结构（如孔数）的合成3D数据集，并训练3D卷积变换器估算亏格，揭示几何复杂性对模型泛化的影响。**

- **链接: [http://arxiv.org/pdf/2511.04972v1](http://arxiv.org/pdf/2511.04972v1)**

> **作者:** Dylan Peek; Matthew P. Skerritt; Siddharth Pritam; Stephan Chalup
>
> **备注:** 10 pages
>
> **摘要:** Topological Data Analysis (TDA) involves techniques of analyzing the underlying structure and connectivity of data. However, traditional methods like persistent homology can be computationally demanding, motivating the development of neural network-based estimators capable of reducing computational overhead and inference time. A key barrier to advancing these methods is the lack of labeled 3D data with class distributions and diversity tailored specifically for supervised learning in TDA tasks. To address this, we introduce a novel approach for systematically generating labeled 3D datasets using the Repulsive Surface algorithm, allowing control over topological invariants, such as hole count. The resulting dataset offers varied geometry with topological labeling, making it suitable for training and benchmarking neural network estimators. This paper uses a synthetic 3D dataset to train a genus estimator network, created using a 3D convolutional transformer architecture. An observed decrease in accuracy as deformations increase highlights the role of not just topological complexity, but also geometric complexity, when training generalized estimators. This dataset fills a gap in labeled 3D datasets and generation for training and evaluating models and techniques for TDA.
>
---
#### [new 059] AI Assisted AR Assembly: Object Recognition and Computer Vision for Augmented Reality Assisted Assembly
- **分类: cs.CV; cs.AI; cs.HC; H.5.2; H.5.1; I.4.8; I.2.6**

- **简介: 该论文提出一种AI辅助AR装配系统，利用深度学习实现物体识别，实时在AR中框出组件及其目标位置，解决人工查找与排序问题，以乐高积木组装为案例验证可行性。**

- **链接: [http://arxiv.org/pdf/2511.05394v1](http://arxiv.org/pdf/2511.05394v1)**

> **作者:** Alexander Htet Kyaw; Haotian Ma; Sasa Zivkovic; Jenny Sabin
>
> **备注:** Accepted to the Association for Computing Machinery (ACM) Symposium on Computational Fabrication (SCF '25)
>
> **摘要:** We present an AI-assisted Augmented Reality assembly workflow that uses deep learning-based object recognition to identify different assembly components and display step-by-step instructions. For each assembly step, the system displays a bounding box around the corresponding components in the physical space, and where the component should be placed. By connecting assembly instructions with the real-time location of relevant components, the system eliminates the need for manual searching, sorting, or labeling of different components before each assembly. To demonstrate the feasibility of using object recognition for AR-assisted assembly, we highlight a case study involving the assembly of LEGO sculptures.
>
---
#### [new 060] A benchmark multimodal oro-dental dataset for large vision-language models
- **分类: cs.CV; cs.AI**

- **简介: 该论文构建了首个大规模多模态口腔牙科数据集，包含影像与文本记录，用于训练和评估视觉-语言模型。工作包括数据收集、标注及微调Qwen-VL模型，提升口腔异常分类与诊断报告生成性能，推动AI在牙科中的应用。**

- **链接: [http://arxiv.org/pdf/2511.04948v1](http://arxiv.org/pdf/2511.04948v1)**

> **作者:** Haoxin Lv; Ijazul Haq; Jin Du; Jiaxin Ma; Binnian Zhu; Xiaobing Dang; Chaoan Liang; Ruxu Du; Yingjie Zhang; Muhammad Saqib
>
> **摘要:** The advancement of artificial intelligence in oral healthcare relies on the availability of large-scale multimodal datasets that capture the complexity of clinical practice. In this paper, we present a comprehensive multimodal dataset, comprising 8775 dental checkups from 4800 patients collected over eight years (2018-2025), with patients ranging from 10 to 90 years of age. The dataset includes 50000 intraoral images, 8056 radiographs, and detailed textual records, including diagnoses, treatment plans, and follow-up notes. The data were collected under standard ethical guidelines and annotated for benchmarking. To demonstrate its utility, we fine-tuned state-of-the-art large vision-language models, Qwen-VL 3B and 7B, and evaluated them on two tasks: classification of six oro-dental anomalies and generation of complete diagnostic reports from multimodal inputs. We compared the fine-tuned models with their base counterparts and GPT-4o. The fine-tuned models achieved substantial gains over these baselines, validating the dataset and underscoring its effectiveness in advancing AI-driven oro-dental healthcare solutions. The dataset is publicly available, providing an essential resource for future research in AI dentistry.
>
---
#### [new 061] Self-Supervised Implicit Attention Priors for Point Cloud Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对点云重建中稀疏区域难以恢复的问题，提出自监督隐式注意力先验，通过输入点云自身学习形状先验，并结合隐式神经表示与RIMLS算法，实现高保真、抗噪声的表面重建。**

- **链接: [http://arxiv.org/pdf/2511.04864v1](http://arxiv.org/pdf/2511.04864v1)**

> **作者:** Kyle Fogarty; Chenyue Cai; Jing Yang; Zhilin Guo; Cengiz Öztireli
>
> **备注:** Accepted at 3DV 2026
>
> **摘要:** Recovering high-quality surfaces from irregular point cloud is ill-posed unless strong geometric priors are available. We introduce an implicit self-prior approach that distills a shape-specific prior directly from the input point cloud itself and embeds it within an implicit neural representation. This is achieved by jointly training a small dictionary of learnable embeddings with an implicit distance field; at every query location, the field attends to the dictionary via cross-attention, enabling the network to capture and reuse repeating structures and long-range correlations inherent to the shape. Optimized solely with self-supervised point cloud reconstruction losses, our approach requires no external training data. To effectively integrate this learned prior while preserving input fidelity, the trained field is then sampled to extract densely distributed points and analytic normals via automatic differentiation. We integrate the resulting dense point cloud and corresponding normals into a robust implicit moving least squares (RIMLS) formulation. We show this hybrid strategy preserves fine geometric details in the input data, while leveraging the learned prior to regularize sparse regions. Experiments show that our method outperforms both classical and learning-based approaches in generating high-fidelity surfaces with superior detail preservation and robustness to common data degradations.
>
---
#### [new 062] An Active Learning Pipeline for Biomedical Image Instance Segmentation with Minimal Human Intervention
- **分类: cs.CV; cs.AI; cs.LG; 68T07, 68U10; I.2.10; I.4.6; J.3**

- **简介: 该论文针对生物医学图像实例分割中标注数据稀缺的问题，提出一种主动学习流水线，结合基础模型生成伪标签与nnU-Net自配置，仅需极少人工标注即可实现高性能分割。**

- **链接: [http://arxiv.org/pdf/2511.04811v1](http://arxiv.org/pdf/2511.04811v1)**

> **作者:** Shuo Zhao; Yu Zhou; Jianxu Chen
>
> **备注:** 6 pages, 4 figures, presented at Bildverarbeitung f\"ur die Medizin (BVM) 2025, Wiesbaden, Germany
>
> **摘要:** Biomedical image segmentation is critical for precise structure delineation and downstream analysis. Traditional methods often struggle with noisy data, while deep learning models such as U-Net have set new benchmarks in segmentation performance. nnU-Net further automates model configuration, making it adaptable across datasets without extensive tuning. However, it requires a substantial amount of annotated data for cross-validation, posing a challenge when only raw images but no labels are available. Large foundation models offer zero-shot generalizability, but may underperform on specific datasets with unique characteristics, limiting their direct use for analysis. This work addresses these bottlenecks by proposing a data-centric AI workflow that leverages active learning and pseudo-labeling to combine the strengths of traditional neural networks and large foundation models while minimizing human intervention. The pipeline starts by generating pseudo-labels from a foundation model, which are then used for nnU-Net's self-configuration. Subsequently, a representative core-set is selected for minimal manual annotation, enabling effective fine-tuning of the nnU-Net model. This approach significantly reduces the need for manual annotations while maintaining competitive performance, providing an accessible solution for biomedical researchers to apply state-of-the-art AI techniques in their segmentation tasks. The code is available at https://github.com/MMV-Lab/AL_BioMed_img_seg.
>
---
#### [new 063] Sharing the Learned Knowledge-base to Estimate Convolutional Filter Parameters for Continual Image Restoration
- **分类: cs.CV**

- **简介: 该论文面向持续图像修复任务，解决新任务学习时遗忘旧任务且计算开销大的问题。提出一种无需修改主干网络的卷积层改进方法，共享历史知识库参数，在不增加推理负担下提升新旧任务性能。**

- **链接: [http://arxiv.org/pdf/2511.05421v1](http://arxiv.org/pdf/2511.05421v1)**

> **作者:** Aupendu Kar; Krishnendu Ghosh; Prabir Kumar Biswas
>
> **备注:** This paper has been accepted to ACM ICVGIP 2025
>
> **摘要:** Continual learning is an emerging topic in the field of deep learning, where a model is expected to learn continuously for new upcoming tasks without forgetting previous experiences. This field has witnessed numerous advancements, but few works have been attempted in the direction of image restoration. Handling large image sizes and the divergent nature of various degradation poses a unique challenge in the restoration domain. However, existing works require heavily engineered architectural modifications for new task adaptation, resulting in significant computational overhead. Regularization-based methods are unsuitable for restoration, as different restoration challenges require different kinds of feature processing. In this direction, we propose a simple modification of the convolution layer to adapt the knowledge from previous restoration tasks without touching the main backbone architecture. Therefore, it can be seamlessly applied to any deep architecture without any structural modifications. Unlike other approaches, we demonstrate that our model can increase the number of trainable parameters without significantly increasing computational overhead or inference time. Experimental validation demonstrates that new restoration tasks can be introduced without compromising the performance of existing tasks. We also show that performance on new restoration tasks improves by adapting the knowledge from the knowledge base created by previous restoration tasks. The code is available at https://github.com/aupendu/continual-restore.
>
---
#### [new 064] DARN: Dynamic Adaptive Regularization Networks for Efficient and Robust Foundation Model Adaptation
- **分类: cs.CV**

- **简介: 该论文提出DARN，用于高效鲁棒地适配地理空间基础模型，解决传统方法忽略图像异质性问题。通过动态预测任务难度并自适应调整dropout与通道激活，提升微调效率与泛化能力，在多个基准上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2511.04766v1](http://arxiv.org/pdf/2511.04766v1)**

> **作者:** Dhenenjay Yadav; Rohan Sawai
>
> **摘要:** Foundation models (FMs) offer powerful representations for geospatial analysis, but adapting them effectively remains challenging. Standard adaptation methods, whether full fine-tuning or efficient frozen-backbone approaches, typically employ decoders with fixed regularization strategies, failing to account for the significant heterogeneity in satellite imagery. We introduce Dynamic Adaptive Regularization Networks (DARN), a novel decoder architecture designed to address this limitation. DARN integrates three key innovations: (1) a lightweight Task Complexity Predictor (TCP) that estimates per-sample difficulty, (2) Adaptive Dropout Modulation (ADM), dynamically adjusting dropout rates (from 0.1 to 0.5) based on predicted complexity, and (3) Dynamic Capacity Gating (DCG) that modulates channel activation. We provide theoretical justifications linking DARN's optimization to stationary point convergence and its mechanism to adaptive information bottlenecks. Empirically, DARN demonstrates exceptional performance across both major adaptation paradigms. In full fine-tuning (unfrozen backbone), DARN achieves a new state-of-the-art on the multi-task GeoBench benchmark (86.66% mIoU, +5.56 pp over prior SOTA). In efficient adaptation (frozen backbone), DARN achieves SOTA-competitive accuracy (90.5% mIoU on Sen1Floods11) while delivering substantial advantages crucial for real-world deployment: superior out-of-distribution (OOD) generalization (+9.5 pp mIoU on AI4SmallFarms), enhanced robustness (17% relative reduction in corruption error), and improved performance on minority classes. DARN offers a more intelligent, robust, and efficient approach to leveraging FMs in critical geospatial applications.
>
---
#### [new 065] Role-SynthCLIP: A Role Play Driven Diverse Synthetic Data Approach
- **分类: cs.CV**

- **简介: 论文提出Role-SynthCLIP，通过多角色提示引导多模态大模型生成语义多样化的图文配对，解决合成数据语义单一、冗余问题，在不增加数据量下显著提升CLIP模型性能。**

- **链接: [http://arxiv.org/pdf/2511.05057v1](http://arxiv.org/pdf/2511.05057v1)**

> **作者:** Yuanxiang Huangfu; Chaochao Wang; Weilei Wang
>
> **摘要:** The effectiveness of Contrastive Language-Image Pre-training (CLIP) models critically depends on the semantic diversity and quality of their training data. However, while existing synthetic data generation methods primarily focus on increasing data volume, such emphasis often leads to limited semantic diversity and redundant or shallow captions. To address this limitation, we propose Role-SynthCLIP, a novel data synthesis framework that leverages multi-perspective role-playing prompts (e.g., a compositional analyst, an interpreter of image context) to guide Multimodal Large Language Models (MLLMs) in generating semantically diverse captions from distinct viewpoints. This mechanism enhances the semantic diversity and fine-grained image-text alignment of synthetic pairs, thereby improving caption expressiveness and accuracy while keeping the total number of image-text pairs unchanged. Experimental results demonstrate the effectiveness and efficiency of our method. A CLIP-B/16 model trained on only 1 million Role-SynthCLIP pairs achieves a Recall@1 of 64.1% on the MS COCO validation set, surpassing the best existing synthetic data baseline (trained on 5M pairs) by 2.8 percentage points. The code and trained models are released at https://github.com/huangfu170/Role-SynthCLIP.
>
---
#### [new 066] Towards Mitigating Hallucinations in Large Vision-Language Models by Refining Textual Embeddings
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉-语言模型中语言模态偏差导致的幻觉问题，提出通过平均池化视觉特征 refine 文本嵌入，增强视觉对齐，有效降低幻觉，属视觉-语言对齐与幻觉抑制任务。**

- **链接: [http://arxiv.org/pdf/2511.05017v1](http://arxiv.org/pdf/2511.05017v1)**

> **作者:** Aakriti Agrawal; Gouthaman KV; Rohith Aralikatti; Gauri Jagatap; Jiaxin Yuan; Vijay Kamarshi; Andrea Fanelli; Furong Huang
>
> **摘要:** In this work, we identify an inherent bias in prevailing LVLM architectures toward the language modality, largely resulting from the common practice of simply appending visual embeddings to the input text sequence. To address this, we propose a simple yet effective method that refines textual embeddings by integrating average-pooled visual features. Our approach demonstrably improves visual grounding and significantly reduces hallucinations on established benchmarks. While average pooling offers a straightforward, robust, and efficient means of incorporating visual information, we believe that more sophisticated fusion methods could further enhance visual grounding and cross-modal alignment. Given that the primary focus of this work is to highlight the modality imbalance and its impact on hallucinations -- and to show that refining textual embeddings with visual information mitigates this issue -- we leave exploration of advanced fusion strategies for future work.
>
---
#### [new 067] Multi-modal Loop Closure Detection with Foundation Models in Severely Unstructured Environments
- **分类: cs.CV; cs.AI; I.2.9; I.2.10**

- **简介: 论文提出MPRF，利用视觉与LiDAR基础模型实现严酷非结构环境中的多模态回环检测，融合高效检索与6自由度位姿估计，提升低纹理区域的精度与鲁棒性，统一位姿识别与定位任务。**

- **链接: [http://arxiv.org/pdf/2511.05404v1](http://arxiv.org/pdf/2511.05404v1)**

> **作者:** Laura Alejandra Encinar Gonzalez; John Folkesson; Rudolph Triebel; Riccardo Giubilato
>
> **备注:** Under review for ICRA 2026
>
> **摘要:** Robust loop closure detection is a critical component of Simultaneous Localization and Mapping (SLAM) algorithms in GNSS-denied environments, such as in the context of planetary exploration. In these settings, visual place recognition often fails due to aliasing and weak textures, while LiDAR-based methods suffer from sparsity and ambiguity. This paper presents MPRF, a multimodal pipeline that leverages transformer-based foundation models for both vision and LiDAR modalities to achieve robust loop closure in severely unstructured environments. Unlike prior work limited to retrieval, MPRF integrates a two-stage visual retrieval strategy with explicit 6-DoF pose estimation, combining DINOv2 features with SALAD aggregation for efficient candidate screening and SONATA-based LiDAR descriptors for geometric verification. Experiments on the S3LI dataset and S3LI Vulcano dataset show that MPRF outperforms state-of-the-art retrieval methods in precision while enhancing pose estimation robustness in low-texture regions. By providing interpretable correspondences suitable for SLAM back-ends, MPRF achieves a favorable trade-off between accuracy, efficiency, and reliability, demonstrating the potential of foundation models to unify place recognition and pose estimation. Code and models will be released at github.com/DLR-RM/MPRF.
>
---
#### [new 068] UHDRes: Ultra-High-Definition Image Restoration via Dual-Domain Decoupled Spectral Modulation
- **分类: eess.IV; cs.CV**

- **简介: UHDRes针对超高清图像去模糊、去雾等复原任务，提出双域解耦频谱调制框架，显式增强幅度谱、隐式恢复相位，结合空间频谱融合与共享门控网络，在仅400K参数下实现SOTA性能与低延迟。**

- **链接: [http://arxiv.org/pdf/2511.05009v1](http://arxiv.org/pdf/2511.05009v1)**

> **作者:** S. Zhao; W. Lu; B. Wang; T. Wang; K. Zhang; H. Zhao
>
> **摘要:** Ultra-high-definition (UHD) images often suffer from severe degradations such as blur, haze, rain, or low-light conditions, which pose significant challenges for image restoration due to their high resolution and computational demands. In this paper, we propose UHDRes, a novel lightweight dual-domain decoupled spectral modulation framework for UHD image restoration. It explicitly models the amplitude spectrum via lightweight spectrum-domain modulation, while restoring phase implicitly through spatial-domain refinement. We introduce the spatio-spectral fusion mechanism, which first employs a multi-scale context aggregator to extract local and global spatial features, and then performs spectral modulation in a decoupled manner. It explicitly enhances amplitude features in the frequency domain while implicitly restoring phase information through spatial refinement. Additionally, a shared gated feed-forward network is designed to efficiently promote feature interaction through shared-parameter convolutions and adaptive gating mechanisms. Extensive experimental comparisons on five public UHD benchmarks demonstrate that our UHDRes achieves the state-of-the-art restoration performance with only 400K parameters, while significantly reducing inference latency and memory usage. The codes and models are available at https://github.com/Zhao0100/UHDRes.
>
---
#### [new 069] SiamMM: A Mixture Model Perspective on Deep Unsupervised Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文将深度无监督聚类与统计混合模型关联，提出SiamMM模型，提升聚类性能，实现自监督学习新SOTA，并揭示标签错误。**

- **链接: [http://arxiv.org/pdf/2511.05462v1](http://arxiv.org/pdf/2511.05462v1)**

> **作者:** Xiaodong Wang; Jing Huang; Kevin J Liang
>
> **摘要:** Recent studies have demonstrated the effectiveness of clustering-based approaches for self-supervised and unsupervised learning. However, the application of clustering is often heuristic, and the optimal methodology remains unclear. In this work, we establish connections between these unsupervised clustering methods and classical mixture models from statistics. Through this framework, we demonstrate significant enhancements to these clustering methods, leading to the development of a novel model named SiamMM. Our method attains state-of-the-art performance across various self-supervised learning benchmarks. Inspection of the learned clusters reveals a strong resemblance to unseen ground truth labels, uncovering potential instances of mislabeling.
>
---
#### [new 070] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究文本到图像生成模型的安全防护问题，发现负提示与微调方法存在兼容性矛盾。提出用概念反演生成隐式负嵌入替代负提示，无需修改现有模型，显著提升对裸露与暴力内容的防御效果，同时保留原始语义。**

- **链接: [http://arxiv.org/pdf/2511.04834v1](http://arxiv.org/pdf/2511.04834v1)**

> **作者:** Jiwoo Shin; Byeonghu Na; Mina Kang; Wonhyeok Choi; Il-chul Moon
>
> **备注:** Accepted at NeurIPS 2025 Workshop on Generative and Protective AI for Content Creation
>
> **摘要:** Recent advances in text-to-image generative models have raised concerns about their potential to produce harmful content when provided with malicious input text prompts. To address this issue, two main approaches have emerged: (1) fine-tuning the model to unlearn harmful concepts and (2) training-free guidance methods that leverage negative prompts. However, we observe that combining these two orthogonal approaches often leads to marginal or even degraded defense performance. This observation indicates a critical incompatibility between two paradigms, which hinders their combined effectiveness. In this work, we address this issue by proposing a conceptually simple yet experimentally robust method: replacing the negative prompts used in training-free methods with implicit negative embeddings obtained through concept inversion. Our method requires no modification to either approach and can be easily integrated into existing pipelines. We experimentally validate its effectiveness on nudity and violence benchmarks, demonstrating consistent improvements in defense success rate while preserving the core semantics of input prompts.
>
---
#### [new 071] Quantifying the Risk of Transferred Black Box Attacks
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于对抗攻击风险量化任务，旨在解决黑盒迁移攻击难以准确评估的问题。提出基于CKA相似性选择代理模型的靶向测试框架，结合回归估计器实现高效、可操作的对抗风险量化。**

- **链接: [http://arxiv.org/pdf/2511.05102v1](http://arxiv.org/pdf/2511.05102v1)**

> **作者:** Disesdi Susanna Cox; Niklas Bunzel
>
> **摘要:** Neural networks have become pervasive across various applications, including security-related products. However, their widespread adoption has heightened concerns regarding vulnerability to adversarial attacks. With emerging regulations and standards emphasizing security, organizations must reliably quantify risks associated with these attacks, particularly regarding transferred adversarial attacks, which remain challenging to evaluate accurately. This paper investigates the complexities involved in resilience testing against transferred adversarial attacks. Our analysis specifically addresses black-box evasion attacks, highlighting transfer-based attacks due to their practical significance and typically high transferability between neural network models. We underline the computational infeasibility of exhaustively exploring high-dimensional input spaces to achieve complete test coverage. As a result, comprehensive adversarial risk mapping is deemed impractical. To mitigate this limitation, we propose a targeted resilience testing framework that employs surrogate models strategically selected based on Centered Kernel Alignment (CKA) similarity. By leveraging surrogate models exhibiting both high and low CKA similarities relative to the target model, the proposed approach seeks to optimize coverage of adversarial subspaces. Risk estimation is conducted using regression-based estimators, providing organizations with realistic and actionable risk quantification.
>
---
#### [new 072] Ada-FCN: Adaptive Frequency-Coupled Network for fMRI-Based Brain Disorder Classification
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对fMRI脑疾病分类中忽视频域特异性的问题，提出Ada-FCN框架，通过自适应分解学习区域特异性频带，并构建频域耦合功能网络，提升诊断精度。**

- **链接: [http://arxiv.org/pdf/2511.04718v1](http://arxiv.org/pdf/2511.04718v1)**

> **作者:** Yue Xun; Jiaxing Xu; Wenbo Gao; Chen Yang; Shujun Wang
>
> **备注:** 11 pages, 2 figures, conference
>
> **摘要:** Resting-state fMRI has become a valuable tool for classifying brain disorders and constructing brain functional connectivity networks by tracking BOLD signals across brain regions. However, existing mod els largely neglect the multi-frequency nature of neuronal oscillations, treating BOLD signals as monolithic time series. This overlooks the cru cial fact that neurological disorders often manifest as disruptions within specific frequency bands, limiting diagnostic sensitivity and specificity. While some methods have attempted to incorporate frequency informa tion, they often rely on predefined frequency bands, which may not be optimal for capturing individual variability or disease-specific alterations. To address this, we propose a novel framework featuring Adaptive Cas cade Decomposition to learn task-relevant frequency sub-bands for each brain region and Frequency-Coupled Connectivity Learning to capture both intra- and nuanced cross-band interactions in a unified functional network. This unified network informs a novel message-passing mecha nism within our Unified-GCN, generating refined node representations for diagnostic prediction. Experimental results on the ADNI and ABIDE datasets demonstrate superior performance over existing methods. The code is available at https://github.com/XXYY20221234/Ada-FCN.
>
---
#### [new 073] Neural Image Abstraction Using Long Smoothing B-Splines
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出一种基于长光滑B样条的神经图像抽象方法，集成至DiffVG框架，实现可微分、长路径的矢量图形生成，解决传统方法路径不连续、控制困难问题，并支持风格化图像抽象与文本生成。**

- **链接: [http://arxiv.org/pdf/2511.05360v1](http://arxiv.org/pdf/2511.05360v1)**

> **作者:** Daniel Berio; Michael Stroh; Sylvain Calinon; Frederic Fol Leymarie; Oliver Deussen; Ariel Shamir
>
> **摘要:** We integrate smoothing B-splines into a standard differentiable vector graphics (DiffVG) pipeline through linear mapping, and show how this can be used to generate smooth and arbitrarily long paths within image-based deep learning systems. We take advantage of derivative-based smoothing costs for parametric control of fidelity vs. simplicity tradeoffs, while also enabling stylization control in geometric and image spaces. The proposed pipeline is compatible with recent vector graphics generation and vectorization methods. We demonstrate the versatility of our approach with four applications aimed at the generation of stylized vector graphics: stylized space-filling path generation, stroke-based image abstraction, closed-area image abstraction, and stylized text generation.
>
---
#### [new 074] LG-NuSegHop: A Local-to-Global Self-Supervised Pipeline For Nuclei Instance Segmentation
- **分类: eess.IV; cs.CV; q-bio.BM**

- **简介: 该论文提出LG-NuSegHop，一种无需标注数据的自监督核分割方法，解决组织图像中核形态多样与标注昂贵的问题，通过局部-全局协同流程实现高精度、可解释的核实例分割，性能媲美监督方法。**

- **链接: [http://arxiv.org/pdf/2511.04892v1](http://arxiv.org/pdf/2511.04892v1)**

> **作者:** Vasileios Magoulianitis; Catherine A. Alexander; Jiaxin Yang; C. -C. Jay Kuo
>
> **备注:** 42 pages, 8 figures, 7 tables
>
> **摘要:** Nuclei segmentation is the cornerstone task in histology image reading, shedding light on the underlying molecular patterns and leading to disease or cancer diagnosis. Yet, it is a laborious task that requires expertise from trained physicians. The large nuclei variability across different organ tissues and acquisition processes challenges the automation of this task. On the other hand, data annotations are expensive to obtain, and thus, Deep Learning (DL) models are challenged to generalize to unseen organs or different domains. This work proposes Local-to-Global NuSegHop (LG-NuSegHop), a self-supervised pipeline developed on prior knowledge of the problem and molecular biology. There are three distinct modules: (1) a set of local processing operations to generate a pseudolabel, (2) NuSegHop a novel data-driven feature extraction model and (3) a set of global operations to post-process the predictions of NuSegHop. Notably, even though the proposed pipeline uses { no manually annotated training data} or domain adaptation, it maintains a good generalization performance on other datasets. Experiments in three publicly available datasets show that our method outperforms other self-supervised and weakly supervised methods while having a competitive standing among fully supervised methods. Remarkably, every module within LG-NuSegHop is transparent and explainable to physicians.
>
---
#### [new 075] On Flow Matching KL Divergence
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文研究流匹配（Flow Matching）的统计效率，推导了$L_2$损失与KL散度之间的确定性上界，证明其在总变差距离下可达近最小方差效率，理论支撑其与扩散模型相当的性能。**

- **链接: [http://arxiv.org/pdf/2511.05480v1](http://arxiv.org/pdf/2511.05480v1)**

> **作者:** Maojiang Su; Jerry Yao-Chieh Hu; Sophia Pi; Han Liu
>
> **摘要:** We derive a deterministic, non-asymptotic upper bound on the Kullback-Leibler (KL) divergence of the flow-matching distribution approximation. In particular, if the $L_2$ flow-matching loss is bounded by $\epsilon^2 > 0$, then the KL divergence between the true data distribution and the estimated distribution is bounded by $A_1 \epsilon + A_2 \epsilon^2$. Here, the constants $A_1$ and $A_2$ depend only on the regularities of the data and velocity fields. Consequently, this bound implies statistical convergence rates of Flow Matching Transformers under the Total Variation (TV) distance. We show that, flow matching achieves nearly minimax-optimal efficiency in estimating smooth distributions. Our results make the statistical efficiency of flow matching comparable to that of diffusion models under the TV distance. Numerical studies on synthetic and learned velocities corroborate our theory.
>
---
#### [new 076] DAFM: Dynamic Adaptive Fusion for Multi-Model Collaboration in Composed Image Retrieval
- **分类: cs.GR; cs.CV**

- **简介: 该论文针对组合图像检索（CIR）任务，提出DAFM方法，通过动态自适应融合多模型特征，解决单模型难以兼顾全局与细节、缺乏自适应权重分配导致的嵌入漂移问题，显著提升检索精度。**

- **链接: [http://arxiv.org/pdf/2511.05020v1](http://arxiv.org/pdf/2511.05020v1)**

> **作者:** Yawei Cai; Jiapeng Mi; Nan Ji; Haotian Rong; Yawei Zhang; Zhangti Li; Wenbin Guo; Rensong Xie
>
> **备注:** 10 pages,4 figures
>
> **摘要:** Composed Image Retrieval (CIR) is a cross-modal task that aims to retrieve target images from large-scale databases using a reference image and a modification text. Most existing methods rely on a single model to perform feature fusion and similarity matching. However, this paradigm faces two major challenges. First, one model alone can't see the whole picture and the tiny details at the same time; it has to handle different tasks with the same weights, so it often misses the small but important links between image and text. Second, the absence of dynamic weight allocation prevents adaptive leveraging of complementary model strengths, so the resulting embedding drifts away from the target and misleads the nearest-neighbor search in CIR. To address these limitations, we propose Dynamic Adaptive Fusion (DAFM) for multi-model collaboration in CIR. Rather than optimizing a single method in isolation, DAFM exploits the complementary strengths of heterogeneous models and adaptively rebalances their contributions. This not only maximizes retrieval accuracy but also ensures that the performance gains are independent of the fusion order, highlighting the robustness of our approach. Experiments on the CIRR and FashionIQ benchmarks demonstrate consistent improvements. Our method achieves a Recall@10 of 93.21 and an Rmean of 84.43 on CIRR, and an average Rmean of 67.48 on FashionIQ, surpassing recent strong baselines by up to 4.5%. These results confirm that dynamic multi-model collaboration provides an effective and general solution for CIR.
>
---
#### [new 077] PySlyde: A Lightweight, Open-Source Toolkit for Pathology Preprocessing
- **分类: q-bio.QM; cs.CV; eess.IV**

- **简介: PySlyde是一款轻量级开源Python工具包，旨在解决病理全切片图像（WSI）预处理流程碎片化问题，统一组织检测、切片、染色归一化等关键步骤，提升数据标准化与可复现性，助力AI模型开发。**

- **链接: [http://arxiv.org/pdf/2511.05183v1](http://arxiv.org/pdf/2511.05183v1)**

> **作者:** Gregory Verghese; Anthony Baptista; Chima Eke; Holly Rafique; Mengyuan Li; Fathima Mohamed; Ananya Bhalla; Lucy Ryan; Michael Pitcher; Enrico Parisini; Concetta Piazzese; Liz Ing-Simmons; Anita Grigoriadis
>
> **摘要:** The integration of artificial intelligence (AI) into pathology is advancing precision medicine by improving diagnosis, treatment planning, and patient outcomes. Digitised whole-slide images (WSIs) capture rich spatial and morphological information vital for understanding disease biology, yet their gigapixel scale and variability pose major challenges for standardisation and analysis. Robust preprocessing, covering tissue detection, tessellation, stain normalisation, and annotation parsing is critical but often limited by fragmented and inconsistent workflows. We present PySlyde, a lightweight, open-source Python toolkit built on OpenSlide to simplify and standardise WSI preprocessing. PySlyde provides an intuitive API for slide loading, annotation management, tissue detection, tiling, and feature extraction, compatible with modern pathology foundation models. By unifying these processes, it streamlines WSI preprocessing, enhances reproducibility, and accelerates the generation of AI-ready datasets, enabling researchers to focus on model development and downstream analysis.
>
---
#### [new 078] Cross-Lingual SynthDocs: A Large-Scale Synthetic Corpus for Any to Arabic OCR and Document Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Cross-Lingual SynthDocs，一个大规模合成阿拉伯语文档数据集，解决阿拉伯语OCR与文档理解资源匮乏问题。通过真实背景、双语布局与带变音符号字体生成250万样本，显著提升Qwen-2.5-VL在多模态任务上的性能。**

- **链接: [http://arxiv.org/pdf/2511.04699v1](http://arxiv.org/pdf/2511.04699v1)**

> **作者:** Haneen Al-Homoud; Asma Ibrahim; Murtadha Al-Jubran; Fahad Al-Otaibi; Yazeed Al-Harbi; Daulet Toibazar; Kesen Wang; Pedro J. Moreno
>
> **摘要:** Cross-Lingual SynthDocs is a large-scale synthetic corpus designed to address the scarcity of Arabic resources for Optical Character Recognition (OCR) and Document Understanding (DU). The dataset comprises over 2.5 million of samples, including 1.5 million textual data, 270K fully annotated tables, and hundred thousands of real data based charts. Our pipeline leverages authentic scanned backgrounds, bilingual layouts, and diacritic aware fonts to capture the typographic and structural complexity of Arabic documents. In addition to text, the corpus includes variety of rendered styles for charts and tables. Finetuning Qwen-2.5-VL on SynthDocs yields consistent improvements in Word Error Rate (WER) and Character Error Rate (CER) in terms of OCR across multiple public Arabic benchmarks, Tree-Edit Distance Similarity (TEDS) and Chart Extraction Score (CharTeX) improved as well in other modalities. SynthDocs provides a scalable, visually realistic resource for advancing research in multilingual document analysis.
>
---
#### [new 079] EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 论文提出EverydayVLA，一种低成本（<300美元）视觉-语言-动作模型，用于机器人操作。它联合输出离散与连续动作，通过自适应规划提升在复杂场景中的可靠性，在LIBERO和真实场景中均超越现有方法，推动家用与科研场景的普及。**

- **链接: [http://arxiv.org/pdf/2511.05397v1](http://arxiv.org/pdf/2511.05397v1)**

> **作者:** Samarth Chopra; Alex McMoil; Ben Carnovale; Evan Sokolson; Rajkumar Kubendran; Samuel Dickerson
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** While Vision-Language-Action (VLA) models map visual inputs and language instructions directly to robot actions, they often rely on costly hardware and struggle in novel or cluttered scenes. We introduce EverydayVLA, a 6-DOF manipulator that can be assembled for under $300, capable of modest payloads and workspace. A single unified model jointly outputs discrete and continuous actions, and our adaptive-horizon ensemble monitors motion uncertainty to trigger on-the-fly re-planning for safe, reliable operation. On LIBERO, EverydayVLA matches state-of-the-art success rates, and in real-world tests it outperforms prior methods by 49% in-distribution and 34.9% out-of-distribution. By combining a state-of-the-art VLA with cost-effective hardware, EverydayVLA democratizes access to a robotic foundation model and paves the way for economical use in homes and research labs alike. Experiment videos and details: https://everydayvla.github.io/
>
---
## 更新

#### [replaced 001] Holistic Evaluation of Multimodal LLMs on Spatial Intelligence
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.13142v3](http://arxiv.org/pdf/2508.13142v3)**

> **作者:** Zhongang Cai; Yubo Wang; Qingping Sun; Ruisi Wang; Chenyang Gu; Wanqi Yin; Zhiqian Lin; Zhitao Yang; Chen Wei; Oscar Qian; Hui En Pang; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Jiaqi Li; Xiangyu Fan; Hanming Deng; Lewei Lu; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: https://github.com/EvolvingLMMs-Lab/EASI/
>
> **摘要:** Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence. We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a standardized protocol for the fair evaluation of state-of-the-art proprietary and open-source models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in spatial intelligence (SI), yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail even the most advanced multimodal models.
>
---
#### [replaced 002] FreeSeg-Diff: Training-Free Open-Vocabulary Segmentation with Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.20105v2](http://arxiv.org/pdf/2403.20105v2)**

> **作者:** Barbara Toniella Corradini; Mustafa Shukor; Paul Couairon; Guillaume Couairon; Franco Scarselli; Matthieu Cord
>
> **摘要:** Foundation models have exhibited unprecedented capabilities in tackling many domains and tasks. Models such as CLIP are currently widely used to bridge cross-modal representations, and text-to-image diffusion models are arguably the leading models in terms of realistic image generation. Image generative models are trained on massive datasets that provide them with powerful internal spatial representations. In this work, we explore the potential benefits of such representations, beyond image generation, in particular, for dense visual prediction tasks. We focus on the task of image segmentation, which is traditionally solved by training models on closed-vocabulary datasets, with pixel-level annotations. To avoid the annotation cost or training large diffusion models, we constraint our setup to be zero-shot and training-free. In a nutshell, our pipeline leverages different and relatively small-sized, open-source foundation models for zero-shot open-vocabulary segmentation. The pipeline is as follows: the image is passed to both a captioner model (i.e. BLIP) and a diffusion model (i.e., Stable Diffusion Model) to generate a text description and visual representation, respectively. The features are clustered and binarized to obtain class agnostic masks for each object. These masks are then mapped to a textual class, using the CLIP model to support open-vocabulary. Finally, we add a refinement step that allows to obtain a more precise segmentation mask. Our approach (dubbed FreeSeg-Diff), which does not rely on any training, outperforms many training-based approaches on both Pascal VOC and COCO datasets. In addition, we show very competitive results compared to the recent weakly-supervised segmentation approaches. We provide comprehensive experiments showing the superiority of diffusion model features compared to other pretrained models. Project page: https://bcorrad.github.io/freesegdiff/
>
---
#### [replaced 003] THEval. Evaluation Framework for Talking Head Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2511.04520v2](http://arxiv.org/pdf/2511.04520v2)**

> **作者:** Nabyl Quignon; Baptiste Chopin; Yaohui Wang; Antitza Dantcheva
>
> **摘要:** Video generation has achieved remarkable progress, with generated videos increasingly resembling real ones. However, the rapid advance in generation has outpaced the development of adequate evaluation metrics. Currently, the assessment of talking head generation primarily relies on limited metrics, evaluating general video quality, lip synchronization, and on conducting user studies. Motivated by this, we propose a new evaluation framework comprising 8 metrics related to three dimensions (i) quality, (ii) naturalness, and (iii) synchronization. In selecting the metrics, we place emphasis on efficiency, as well as alignment with human preferences. Based on this considerations, we streamline to analyze fine-grained dynamics of head, mouth, and eyebrows, as well as face quality. Our extensive experiments on 85,000 videos generated by 17 state-of-the-art models suggest that while many algorithms excel in lip synchronization, they face challenges with generating expressiveness and artifact-free details. These videos were generated based on a novel real dataset, that we have curated, in order to mitigate bias of training data. Our proposed benchmark framework is aimed at evaluating the improvement of generative methods. Original code, dataset and leaderboards will be publicly released and regularly updated with new methods, in order to reflect progress in the field.
>
---
#### [replaced 004] GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.18090v2](http://arxiv.org/pdf/2509.18090v2)**

> **作者:** Jiahe Li; Jiawei Zhang; Youmin Zhang; Xiao Bai; Jin Zheng; Xiaohan Yu; Lin Gu
>
> **备注:** Accepted at NeurIPS 2025 (Spotlight). Project page: https://fictionarry.github.io/GeoSVR-project/
>
> **摘要:** Reconstructing accurate surfaces with radiance fields has achieved remarkable progress in recent years. However, prevailing approaches, primarily based on Gaussian Splatting, are increasingly constrained by representational bottlenecks. In this paper, we introduce GeoSVR, an explicit voxel-based framework that explores and extends the under-investigated potential of sparse voxels for achieving accurate, detailed, and complete surface reconstruction. As strengths, sparse voxels support preserving the coverage completeness and geometric clarity, while corresponding challenges also arise from absent scene constraints and locality in surface refinement. To ensure correct scene convergence, we first propose a Voxel-Uncertainty Depth Constraint that maximizes the effect of monocular depth cues while presenting a voxel-oriented uncertainty to avoid quality degradation, enabling effective and robust scene constraints yet preserving highly accurate geometries. Subsequently, Sparse Voxel Surface Regularization is designed to enhance geometric consistency for tiny voxels and facilitate the voxel-based formation of sharp and accurate surfaces. Extensive experiments demonstrate our superior performance compared to existing methods across diverse challenging scenarios, excelling in geometric accuracy, detail preservation, and reconstruction completeness while maintaining high efficiency. Code is available at https://github.com/Fictionarry/GeoSVR.
>
---
#### [replaced 005] Cyst-X: A Federated AI System Outperforms Clinical Guidelines to Detect Pancreatic Cancer Precursors and Reduce Unnecessary Surgery
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22017v3](http://arxiv.org/pdf/2507.22017v3)**

> **作者:** Hongyi Pan; Gorkem Durak; Elif Keles; Deniz Seyithanoglu; Zheyuan Zhang; Alpay Medetalibeyoglu; Halil Ertugrul Aktas; Andrea Mia Bejar; Ziliang Hong; Yavuz Taktak; Gulbiz Dagoglu Kartal; Mehmet Sukru Erturk; Timurhan Cebeci; Maria Jaramillo Gonzalez; Yury Velichko; Lili Zhao; Emil Agarunov; Federica Proietto Salanitri; Concetto Spampinato; Pallavi Tiwari; Ziyue Xu; Sachin Jambawalikar; Ivo G. Schoots; Marco J. Bruno; Chenchan Huang; Candice W. Bolan; Tamas Gonda; Frank H. Miller; Rajesh N. Keswani; Michael B. Wallace; Ulas Bagci
>
> **摘要:** Pancreatic cancer is projected to be the second-deadliest cancer by 2030, making early detection critical. Intraductal papillary mucinous neoplasms (IPMNs), key cancer precursors, present a clinical dilemma, as current guidelines struggle to stratify malignancy risk, leading to unnecessary surgeries or missed diagnoses. Here, we developed Cyst-X, an AI framework for IPMN risk prediction trained on a unique, multi-center dataset of 1,461 MRI scans from 764 patients. Cyst-X achieves significantly higher accuracy (AUC = 0.82) than both the established Kyoto guidelines (AUC = 0.75) and expert radiologists, particularly in correct identification of high-risk lesions. Clinically, this translates to a 20% increase in cancer detection sensitivity (87.8% vs. 64.1%) for high-risk lesions. We demonstrate that this performance is maintained in a federated learning setting, allowing for collaborative model training without compromising patient privacy. To accelerate research in early pancreatic cancer detection, we publicly release the Cyst-X dataset and models, providing the first large-scale, multi-center MRI resource for pancreatic cyst analysis.
>
---
#### [replaced 006] SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16601v2](http://arxiv.org/pdf/2502.16601v2)**

> **作者:** Feng Lu; Tong Jin; Xiangyuan Lan; Lijun Zhang; Yunpeng Liu; Yaowei Wang; Chun Yuan
>
> **备注:** accepted by T-PAMI
>
> **摘要:** Recent studies show that the visual place recognition (VPR) method using pre-trained visual foundation models can achieve promising performance. In our previous work, we propose a novel method to realize seamless adaptation of foundation models to VPR (SelaVPR). This method can produce both global and local features that focus on discriminative landmarks to recognize places for two-stage VPR by a parameter-efficient adaptation approach. Although SelaVPR has achieved competitive results, we argue that the previous adaptation is inefficient in training time and GPU memory usage, and the re-ranking paradigm is also costly in retrieval latency and storage usage. In pursuit of higher efficiency and better performance, we propose an extension of the SelaVPR, called SelaVPR++. Concretely, we first design a parameter-, time-, and memory-efficient adaptation method that uses lightweight multi-scale convolution (MultiConv) adapters to refine intermediate features from the frozen foundation backbone. This adaptation method does not back-propagate gradients through the backbone during training, and the MultiConv adapter facilitates feature interactions along the spatial axes and introduces proper local priors, thus achieving higher efficiency and better performance. Moreover, we propose an innovative re-ranking paradigm for more efficient VPR. Instead of relying on local features for re-ranking, which incurs huge overhead in latency and storage, we employ compact binary features for initial retrieval and robust floating-point (global) features for re-ranking. To obtain such binary features, we propose a similarity-constrained deep hashing method, which can be easily integrated into the VPR pipeline. Finally, we improve our training strategy and unify the training protocol of several common training datasets to merge them for better training of VPR models. Extensive experiments show that ......
>
---
#### [replaced 007] ControlGS: Consistent Structural Compression Control for Deployment-Aware Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10473v3](http://arxiv.org/pdf/2505.10473v3)**

> **作者:** Fengdi Zhang; Yibao Sun; Hongkun Cao; Ruqi Huang
>
> **摘要:** 3D Gaussian Splatting (3DGS) is a highly deployable real-time method for novel view synthesis. In practice, it requires a universal, consistent control mechanism that adjusts the trade-off between rendering quality and model compression without scene-specific tuning, enabling automated deployment across different device performances and communication bandwidths. In this work, we present ControlGS, a control-oriented optimization framework that maps the trade-off between Gaussian count and rendering quality to a continuous, scene-agnostic, and highly responsive control axis. Extensive experiments across a wide range of scene scales and types (from small objects to large outdoor scenes) demonstrate that, by adjusting a globally unified control hyperparameter, ControlGS can flexibly generate models biased toward either structural compactness or high fidelity, regardless of the specific scene scale or complexity, while achieving markedly higher rendering quality with the same or fewer Gaussians compared to potential competing methods. Project page: https://zhang-fengdi.github.io/ControlGS/
>
---
#### [replaced 008] Generative Autoregressive Transformers for Model-Agnostic Federated MRI Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.04521v3](http://arxiv.org/pdf/2502.04521v3)**

> **作者:** Valiyeh A. Nezhad; Gokberk Elmas; Bilal Kabas; Fuat Arslan; Emine U. Saritas; Tolga Çukur
>
> **摘要:** While learning-based models hold great promise for MRI reconstruction, single-site models trained on limited local datasets often show poor generalization. This has motivated collaborative training across institutions via federated learning (FL)-a privacy-preserving framework that aggregates model updates instead of sharing raw data. Conventional FL requires architectural homogeneity, restricting sites from using models tailored to their resources or needs. To address this limitation, we propose FedGAT, a model-agnostic FL technique that first collaboratively trains a global generative prior for MR images, adapted from a natural image foundation model composed of a variational autoencoder (VAE) and a transformer that generates images via spatial-scale autoregression. We fine-tune the transformer module after injecting it with a lightweight site-specific prompting mechanism, keeping the VAE frozen, to efficiently adapt the model to multi-site MRI data. In a second tier, each site independently trains its preferred reconstruction model by augmenting local data with synthetic MRI data from other sites, generated by site-prompting the tuned prior. This decentralized augmentation improves generalization while preserving privacy. Experiments on multi-institutional datasets show that FedGAT outperforms state-of-the-art FL baselines in both within- and cross-site reconstruction performance under model-heterogeneous settings.
>
---
#### [replaced 009] USIGAN: Unbalanced Self-Information Feature Transport for Weakly Paired Image IHC Virtual Staining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05843v2](http://arxiv.org/pdf/2507.05843v2)**

> **作者:** Yue Peng; Bing Xiong; Fuqiang Chen; De Eybo; RanRan Zhang; Wanming Hu; Jing Cai; Wenjian Qin
>
> **摘要:** Immunohistochemical (IHC) virtual staining is a task that generates virtual IHC images from H\&E images while maintaining pathological semantic consistency with adjacent slices. This task aims to achieve cross-domain mapping between morphological structures and staining patterns through generative models, providing an efficient and cost-effective solution for pathological analysis. However, under weakly paired conditions, spatial heterogeneity between adjacent slices presents significant challenges. This can lead to inaccurate one-to-many mappings and generate results that are inconsistent with the pathological semantics of adjacent slices. To address this issue, we propose a novel unbalanced self-information feature transport for IHC virtual staining, named USIGAN, which extracts global morphological semantics without relying on positional correspondence.By removing weakly paired terms in the joint marginal distribution, we effectively mitigate the impact of weak pairing on joint distributions, thereby significantly improving the content consistency and pathological semantic consistency of the generated results. Moreover, we design the Unbalanced Optimal Transport Consistency (UOT-CTM) mechanism and the Pathology Self-Correspondence (PC-SCM) mechanism to construct correlation matrices between H\&E and generated IHC in image-level and real IHC and generated IHC image sets in intra-group level.. Experiments conducted on two publicly available datasets demonstrate that our method achieves superior performance across multiple clinically significant metrics, such as IoD and Pearson-R correlation, demonstrating better clinical relevance.
>
---
#### [replaced 010] InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback
- **分类: cs.CL; cs.AI; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2502.15027v3](http://arxiv.org/pdf/2502.15027v3)**

> **作者:** Henry Hengyuan Zhao; Wenqi Pei; Yifei Tao; Haiyang Mei; Mike Zheng Shou
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Existing benchmarks do not test Large Multimodal Models (LMMs) on their interactive intelligence with human users, which is vital for developing general-purpose AI assistants. We design InterFeedback, an interactive framework, which can be applied to any LMM and dataset to assess this ability autonomously. On top of this, we introduce InterFeedback-Bench which evaluates interactive intelligence using two representative datasets, MMMU-Pro and MathVerse, to test 10 different open-source LMMs. Additionally, we present InterFeedback-Human, a newly collected dataset of 120 cases designed for manually testing interactive performance in leading models such as OpenAI-o1 and Claude-Sonnet-4. Our evaluation results indicate that even the state-of-the-art LMM, OpenAI-o1, struggles to refine its responses based on human feedback, achieving an average score of less than 50%. Our findings point to the need for methods that can enhance LMMs' capabilities to interpret and benefit from feedback.
>
---
#### [replaced 011] When Are Concepts Erased From Diffusion Models?
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17013v5](http://arxiv.org/pdf/2505.17013v5)**

> **作者:** Kevin Lu; Nicky Kriplani; Rohit Gandikota; Minh Pham; David Bau; Chinmay Hegde; Niv Cohen
>
> **备注:** Accepted to NeurIPS 2025. Our code, data, and results are available at https://unerasing.baulab.info/
>
> **摘要:** In concept erasure, a model is modified to selectively prevent it from generating a target concept. Despite the rapid development of new methods, it remains unclear how thoroughly these approaches remove the target concept from the model. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) interfering with the model's internal guidance processes, and (ii) reducing the unconditional likelihood of generating the target concept, potentially removing it entirely. To assess whether a concept has been truly erased from the model, we introduce a comprehensive suite of independent probing techniques: supplying visual context, modifying the diffusion trajectory, applying classifier guidance, and analyzing the model's alternative generations that emerge in place of the erased concept. Our results shed light on the value of exploring concept erasure robustness outside of adversarial text inputs, and emphasize the importance of comprehensive evaluations for erasure in diffusion models.
>
---
#### [replaced 012] Self-supervised Deep Unrolled Model with Implicit Neural Representation Regularization for Accelerating MRI Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.06611v2](http://arxiv.org/pdf/2510.06611v2)**

> **作者:** Jingran Xu; Yuanyuan Liu; Yuanbiao Yang; Zhuo-Xu Cui; Jing Cheng; Qingyong Zhu; Nannan Zhang; Yihang Zhou; Dong Liang; Yanjie Zhu
>
> **摘要:** Magnetic resonance imaging (MRI) is a vital clinical diagnostic tool, yet its application is limited by prolonged scan times. Accelerating MRI reconstruction addresses this issue by reconstructing high-fidelity MR images from undersampled k-space measurements. In recent years, deep learning-based methods have demonstrated remarkable progress. However, most methods rely on supervised learning, which requires large amounts of fully-sampled training data that are difficult to obtain. This paper proposes a novel zero-shot self-supervised reconstruction method named UnrollINR, which enables scan-specific MRI reconstruction without external training data. UnrollINR adopts a physics-guided unrolled reconstruction architecture and introduces implicit neural representation (INR) as a regularization prior to effectively constrain the solution space. This method overcomes the local bias limitation of CNNs in traditional deep unrolled methods and avoids the instability associated with relying solely on INR's implicit regularization in highly ill-posed scenarios. Consequently, UnrollINR significantly improves MRI reconstruction performance under high acceleration rates. Experimental results show that even at a high acceleration rate of 10, UnrollINR achieves superior reconstruction performance compared to supervised and self-supervised learning methods, validating its effectiveness and superiority.
>
---
#### [replaced 013] Consistency Trajectory Matching for One-Step Generative Super-Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20349v5](http://arxiv.org/pdf/2503.20349v5)**

> **作者:** Weiyi You; Mingyang Zhang; Leheng Zhang; Xingyu Zhou; Kexuan Shi; Shuhang Gu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Current diffusion-based super-resolution (SR) approaches achieve commendable performance at the cost of high inference overhead. Therefore, distillation techniques are utilized to accelerate the multi-step teacher model into one-step student model. Nevertheless, these methods significantly raise training costs and constrain the performance of the student model by the teacher model. To overcome these tough challenges, we propose Consistency Trajectory Matching for Super-Resolution (CTMSR), a distillation-free strategy that is able to generate photo-realistic SR results in one step. Concretely, we first formulate a Probability Flow Ordinary Differential Equation (PF-ODE) trajectory to establish a deterministic mapping from low-resolution (LR) images with noise to high-resolution (HR) images. Then we apply the Consistency Training (CT) strategy to directly learn the mapping in one step, eliminating the necessity of pre-trained diffusion model. To further enhance the performance and better leverage the ground-truth during the training process, we aim to align the distribution of SR results more closely with that of the natural images. To this end, we propose to minimize the discrepancy between their respective PF-ODE trajectories from the LR image distribution by our meticulously designed Distribution Trajectory Matching (DTM) loss, resulting in improved realism of our recovered HR images. Comprehensive experimental results demonstrate that the proposed methods can attain comparable or even superior capabilities on both synthetic and real datasets while maintaining minimal inference latency.
>
---
#### [replaced 014] Towards Explainable Fake Image Detection with Multi-Modal Large Language Models
- **分类: cs.CV; cs.CL; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2504.14245v2](http://arxiv.org/pdf/2504.14245v2)**

> **作者:** Yikun Ji; Yan Hong; Jiahui Zhan; Haoxing Chen; jun lan; Huijia Zhu; Weiqiang Wang; Liqing Zhang; Jianfu Zhang
>
> **备注:** Accepted to ACM MM 2025; 14 pages including Appendix
>
> **摘要:** Progress in image generation raises significant public security concerns. We argue that fake image detection should not operate as a "black box". Instead, an ideal approach must ensure both strong generalization and transparency. Recent progress in Multi-modal Large Language Models (MLLMs) offers new opportunities for reasoning-based AI-generated image detection. In this work, we evaluate the capabilities of MLLMs in comparison to traditional detection methods and human evaluators, highlighting their strengths and limitations. Furthermore, we design six distinct prompts and propose a framework that integrates these prompts to develop a more robust, explainable, and reasoning-driven detection system. The code is available at https://github.com/Gennadiyev/mllm-defake.
>
---
#### [replaced 015] Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.17643v4](http://arxiv.org/pdf/2311.17643v4)**

> **作者:** Alexander Becker; Rodrigo Caye Daudt; Dominik Narnhofer; Torben Peters; Nando Metzger; Jan Dirk Wegner; Konrad Schindler
>
> **摘要:** Recent approaches to arbitrary-scale single image super-resolution (ASR) use neural fields to represent continuous signals that can be sampled at arbitrary resolutions. However, point-wise queries of neural fields do not naturally match the point spread function (PSF) of pixels, which may cause aliasing in the super-resolved image. Existing methods attempt to mitigate this by approximating an integral version of the field at each scaling factor, compromising both fidelity and generalization. In this work, we introduce neural heat fields, a novel neural field formulation that inherently models a physically exact PSF. Our formulation enables analytically correct anti-aliasing at any desired output resolution, and -- unlike supersampling -- at no additional cost. Building on this foundation, we propose Thera, an end-to-end ASR method that substantially outperforms existing approaches, while being more parameter-efficient and offering strong theoretical guarantees. The project page is at https://therasr.github.io.
>
---
#### [replaced 016] Med-Banana-50K: A Cross-modality Large-Scale Dataset for Text-guided Medical Image Editing
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2511.00801v3](http://arxiv.org/pdf/2511.00801v3)**

> **作者:** Zhihui Chen; Mengling Feng
>
> **摘要:** Medical image editing has emerged as a pivotal technology with broad applications in data augmentation, model interpretability, medical education, and treatment simulation. However, the lack of large-scale, high-quality, and openly accessible datasets tailored for medical contexts with strict anatomical and clinical constraints has significantly hindered progress in this domain. To bridge this gap, we introduce Med-Banana-50K, a comprehensive dataset of over 50k medically curated image edits spanning chest X-ray, brain MRI, and fundus photography across 23 diseases. Each sample supports bidirectional lesion editing (addition and removal) and is constructed using Gemini-2.5-Flash-Image based on real clinical images. A key differentiator of our dataset is the medically grounded quality control protocol: we employ an LLM-as-Judge evaluation framework with criteria such as instruction compliance, structural plausibility, image realism, and fidelity preservation, alongside iterative refinement over up to five rounds. Additionally, Med-Banana-50K includes around 37,000 failed editing attempts with full evaluation logs to support preference learning and alignment research. By offering a large-scale, medically rigorous, and fully documented resource, Med-Banana-50K establishes a critical foundation for developing and evaluating reliable medical image editing systems. Our dataset and code are publicly available. [https://github.com/richardChenzhihui/med-banana-50k].
>
---
#### [replaced 017] USF-MAE: Ultrasound Self-Supervised Foundation Model with Masked Autoencoding
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22990v2](http://arxiv.org/pdf/2510.22990v2)**

> **作者:** Youssef Megahed; Robin Ducharme; Aylin Erman; Mark Walker; Steven Hawken; Adrian D. C. Chan
>
> **备注:** 18 pages, 8 figures, 2 tables
>
> **摘要:** Ultrasound imaging is one of the most widely used diagnostic modalities, offering real-time, radiation-free assessment across diverse clinical domains. However, interpretation of ultrasound images remains challenging due to high noise levels, operator dependence, and limited field of view, resulting in substantial inter-observer variability. Current Deep Learning approaches are hindered by the scarcity of large labeled datasets and the domain gap between general and sonographic images, which limits the transferability of models pretrained on non-medical data. To address these challenges, we introduce the Ultrasound Self-Supervised Foundation Model with Masked Autoencoding (USF-MAE), the first large-scale self-supervised MAE framework pretrained exclusively on ultrasound data. The model was pre-trained on 370,000 2D and 3D ultrasound images curated from 46 open-source datasets, collectively termed OpenUS-46, spanning over twenty anatomical regions. This curated dataset has been made publicly available to facilitate further research and reproducibility. Using a Vision Transformer encoder-decoder architecture, USF-MAE reconstructs masked image patches, enabling it to learn rich, modality-specific representations directly from unlabeled data. The pretrained encoder was fine-tuned on three public downstream classification benchmarks: BUS-BRA (breast cancer), MMOTU-2D (ovarian tumors), and GIST514-DB (gastrointestinal stromal tumors). Across all tasks, USF-MAE consistently outperformed conventional CNN and ViT baselines, achieving F1-scores of 81.6%, 79.6%, and 82.4%, respectively. Despite not using labels during pretraining, USF-MAE approached the performance of the supervised foundation model UltraSam on breast cancer classification and surpassed it on the other tasks, demonstrating strong cross-anatomical generalization.
>
---
#### [replaced 018] TRACE: Textual Relevance Augmentation and Contextual Encoding for Multimodal Hate Detection
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17902v2](http://arxiv.org/pdf/2504.17902v2)**

> **作者:** Girish A. Koushik; Helen Treharne; Aditya Joshi; Diptesh Kanojia
>
> **备注:** Accepted to Special Track on AI for Social Impact (AISI) at AAAI 2026
>
> **摘要:** Social media memes are a challenging domain for hate detection because they intertwine visual and textual cues into culturally nuanced messages. To tackle these challenges, we introduce TRACE, a hierarchical multimodal framework that leverages visually grounded context augmentation, along with a novel caption-scoring network to emphasize hate-relevant content, and parameter-efficient fine-tuning of CLIP's text encoder. Our experiments demonstrate that selectively fine-tuning deeper text encoder layers significantly enhances performance compared to simpler projection-layer fine-tuning methods. Specifically, our framework achieves state-of-the-art accuracy (0.807) and F1-score (0.806) on the widely-used Hateful Memes dataset, matching the performance of considerably larger models while maintaining efficiency. Moreover, it achieves superior generalization on the MultiOFF offensive meme dataset (F1-score 0.673), highlighting robustness across meme categories. Additional analyses confirm that robust visual grounding and nuanced text representations significantly reduce errors caused by benign confounders. We publicly release our code to facilitate future research.
>
---
#### [replaced 019] On Scaling Up 3D Gaussian Splatting Training
- **分类: cs.CV; I.4.5**

- **链接: [http://arxiv.org/pdf/2406.18533v2](http://arxiv.org/pdf/2406.18533v2)**

> **作者:** Hexu Zhao; Haoyang Weng; Daohan Lu; Ang Li; Jinyang Li; Aurojit Panda; Saining Xie
>
> **备注:** ICLR 2025 Oral; Homepage: https://daohanlu.github.io/scaling-up-3dgs/
>
> **摘要:** 3D Gaussian Splatting (3DGS) is increasingly popular for 3D reconstruction due to its superior visual quality and rendering speed. However, 3DGS training currently occurs on a single GPU, limiting its ability to handle high-resolution and large-scale 3D reconstruction tasks due to memory constraints. We introduce Grendel, a distributed system designed to partition 3DGS parameters and parallelize computation across multiple GPUs. As each Gaussian affects a small, dynamic subset of rendered pixels, Grendel employs sparse all-to-all communication to transfer the necessary Gaussians to pixel partitions and performs dynamic load balancing. Unlike existing 3DGS systems that train using one camera view image at a time, Grendel supports batched training with multiple views. We explore various optimization hyperparameter scaling strategies and find that a simple sqrt(batch size) scaling rule is highly effective. Evaluations using large-scale, high-resolution scenes show that Grendel enhances rendering quality by scaling up 3DGS parameters across multiple GPUs. On the Rubble dataset, we achieve a test PSNR of 27.28 by distributing 40.4 million Gaussians across 16 GPUs, compared to a PSNR of 26.28 using 11.2 million Gaussians on a single GPU. Grendel is an open-source project available at: https://github.com/nyu-systems/Grendel-GS
>
---
#### [replaced 020] KARMA: Efficient Structural Defect Segmentation via Kolmogorov-Arnold Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.08186v3](http://arxiv.org/pdf/2508.08186v3)**

> **作者:** Md Meftahul Ferdaus; Mahdi Abdelguerfi; Elias Ioup; Steven Sloan; Kendall N. Niles; Ken Pathak
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Semantic segmentation of structural defects in civil infrastructure remains challenging due to variable defect appearances, harsh imaging conditions, and significant class imbalance. Current deep learning methods, despite their effectiveness, typically require millions of parameters, rendering them impractical for real-time inspection systems. We introduce KARMA (Kolmogorov-Arnold Representation Mapping Architecture), a highly efficient semantic segmentation framework that models complex defect patterns through compositions of one-dimensional functions rather than conventional convolutions. KARMA features three technical innovations: (1) a parameter-efficient Tiny Kolmogorov-Arnold Network (TiKAN) module leveraging low-rank factorization for KAN-based feature transformation; (2) an optimized feature pyramid structure with separable convolutions for multi-scale defect analysis; and (3) a static-dynamic prototype mechanism that enhances feature representation for imbalanced classes. Extensive experiments on benchmark infrastructure inspection datasets demonstrate that KARMA achieves competitive or superior mean IoU performance compared to state-of-the-art approaches, while using significantly fewer parameters (0.959M vs. 31.04M, a 97% reduction). Operating at 0.264 GFLOPS, KARMA maintains inference speeds suitable for real-time deployment, enabling practical automated infrastructure inspection systems without compromising accuracy. The source code can be accessed at the following URL: https://github.com/faeyelab/karma.
>
---
#### [replaced 021] ZERO: Industry-ready Vision Foundation Model with Multi-modal Prompts
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.04270v4](http://arxiv.org/pdf/2507.04270v4)**

> **作者:** Sangbum Choi; Kyeongryeol Go; Taewoong Jang
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Foundation models have revolutionized AI, yet they struggle with zero-shot deployment in real-world industrial settings due to a lack of high-quality, domain-specific datasets. To bridge this gap, Superb AI introduces ZERO, an industry-ready vision foundation model that leverages multi-modal prompting (textual and visual) for generalization without retraining. Trained on a compact yet representative 0.9 million annotated samples from a proprietary billion-scale industrial dataset, ZERO demonstrates competitive performance on academic benchmarks like LVIS-Val and significantly outperforms existing models across 37 diverse industrial datasets. Furthermore, ZERO achieved 2nd place in the CVPR 2025 Object Instance Detection Challenge and 4th place in the Foundational Few-shot Object Detection Challenge, highlighting its practical deployability and generalizability with minimal adaptation and limited data. To the best of our knowledge, ZERO is the first vision foundation model explicitly built for domain-specific, zero-shot industrial applications.
>
---
#### [replaced 022] EditInfinity: Image Editing with Binary-Quantized Generative Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.20217v3](http://arxiv.org/pdf/2510.20217v3)**

> **作者:** Jiahuan Wang; Yuxin Chen; Jun Yu; Guangming Lu; Wenjie Pei
>
> **备注:** 28 pages, 13 figures, accepted by The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Adapting pretrained diffusion-based generative models for text-driven image editing with negligible tuning overhead has demonstrated remarkable potential. A classical adaptation paradigm, as followed by these methods, first infers the generative trajectory inversely for a given source image by image inversion, then performs image editing along the inferred trajectory guided by the target text prompts. However, the performance of image editing is heavily limited by the approximation errors introduced during image inversion by diffusion models, which arise from the absence of exact supervision in the intermediate generative steps. To circumvent this issue, we investigate the parameter-efficient adaptation of binary-quantized generative models for image editing, and leverage their inherent characteristic that the exact intermediate quantized representations of a source image are attainable, enabling more effective supervision for precise image inversion. Specifically, we propose EditInfinity, which adapts \emph{Infinity}, a binary-quantized generative model, for image editing. We propose an efficient yet effective image inversion mechanism that integrates text prompting rectification and image style preservation, enabling precise image inversion. Furthermore, we devise a holistic smoothing strategy which allows our EditInfinity to perform image editing with high fidelity to source images and precise semantic alignment to the text prompts. Extensive experiments on the PIE-Bench benchmark across `add', `change', and `delete' editing operations, demonstrate the superior performance of our model compared to state-of-the-art diffusion-based baselines. Code available at: https://github.com/yx-chen-ust/EditInfinity.
>
---
#### [replaced 023] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering
- **分类: cs.IR; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16470v2](http://arxiv.org/pdf/2505.16470v2)**

> **作者:** Kuicai Dong; Yujing Chang; Shijie Huang; Yasheng Wang; Ruiming Tang; Yong Liu
>
> **备注:** Paper accepted to NeurIPS 2025 DB
>
> **摘要:** Document Visual Question Answering (DocVQA) faces dual challenges in processing lengthy multimodal documents (text, images, tables) and performing cross-modal reasoning. Current document retrieval-augmented generation (DocRAG) methods remain limited by their text-centric approaches, frequently missing critical visual information. The field also lacks robust benchmarks for assessing multimodal evidence selection and integration. We introduce MMDocRAG, a comprehensive benchmark featuring 4,055 expert-annotated QA pairs with multi-page, cross-modal evidence chains. Our framework introduces innovative metrics for evaluating multimodal quote selection and enables answers that interleave text with relevant visual elements. Through large-scale experiments with 60 VLM/LLM models and 14 retrieval systems, we identify persistent challenges in multimodal evidence retrieval, selection, and integration.Key findings reveal advanced proprietary LVMs show superior performance than open-sourced alternatives. Also, they show moderate advantages using multimodal inputs over text-only inputs, while open-source alternatives show significant performance degradation. Notably, fine-tuned LLMs achieve substantial improvements when using detailed image descriptions. MMDocRAG establishes a rigorous testing ground and provides actionable insights for developing more robust multimodal DocVQA systems. Our benchmark and code are available at https://mmdocrag.github.io/MMDocRAG/.
>
---
#### [replaced 024] Towards Understanding the Mechanisms of Classifier-Free Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19210v2](http://arxiv.org/pdf/2505.19210v2)**

> **作者:** Xiang Li; Rongrong Wang; Qing Qu
>
> **摘要:** Classifier-free guidance (CFG) is a core technique powering state-of-the-art image generation systems, yet its underlying mechanisms remain poorly understood. In this work, we begin by analyzing CFG in a simplified linear diffusion model, where we show its behavior closely resembles that observed in the nonlinear case. Our analysis reveals that linear CFG improves generation quality via three distinct components: (i) a mean-shift term that approximately steers samples in the direction of class means, (ii) a positive Contrastive Principal Components (CPC) term that amplifies class-specific features, and (iii) a negative CPC term that suppresses generic features prevalent in unconditional data. We then verify that these insights in real-world, nonlinear diffusion models: over a broad range of noise levels, linear CFG resembles the behavior of its nonlinear counterpart. Although the two eventually diverge at low noise levels, we discuss how the insights from the linear analysis still shed light on the CFG's mechanism in the nonlinear regime.
>
---
#### [replaced 025] Improving Diagnostic Performance on Small and Imbalanced Datasets Using Class-Based Input Image Composition
- **分类: cs.CV; cs.AI; cs.DB**

- **链接: [http://arxiv.org/pdf/2511.03891v2](http://arxiv.org/pdf/2511.03891v2)**

> **作者:** Hlali Azzeddine; Majid Ben Yakhlef; Soulaiman El Hazzat
>
> **摘要:** Small, imbalanced datasets and poor input image quality can lead to high false predictions rates with deep learning models. This paper introduces Class-Based Image Composition, an approach that allows us to reformulate training inputs through a fusion of multiple images of the same class into combined visual composites, named Composite Input Images (CoImg). That enhances the intra-class variance and improves the valuable information density per training sample and increases the ability of the model to distinguish between subtle disease patterns. Our method was evaluated on the Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods (OCTDL) (Kulyabin et al., 2024), which contains 2,064 high-resolution optical coherence tomography (OCT) scans of the human retina, representing seven distinct diseases with a significant class imbalance. We constructed a perfectly class-balanced version of this dataset, named Co-OCTDL, where each scan is resented as a 3x1 layout composite image. To assess the effectiveness of this new representation, we conducted a comparative analysis between the original dataset and its variant using a VGG16 model. A fair comparison was ensured by utilizing the identical model architecture and hyperparameters for all experiments. The proposed approach markedly improved diagnostic results.The enhanced Dataset achieved near-perfect accuracy (99.6%) with F1-score (0.995) and AUC (0.9996), compared to a baseline model trained on raw dataset. The false prediction rate was also significantly lower, this demonstrates that the method can producehigh-quality predictions even for weak datasets affected by class imbalance or small sample size.
>
---
#### [replaced 026] LoRA-Edge: Tensor-Train-Assisted LoRA for Practical CNN Fine-Tuning on Edge Devices
- **分类: cs.CV; cs.AR**

- **链接: [http://arxiv.org/pdf/2511.03765v2](http://arxiv.org/pdf/2511.03765v2)**

> **作者:** Hyunseok Kwak; Kyeongwon Lee; Jae-Jin Lee; Woojoo Lee
>
> **备注:** 8 pages, 6 figures, 2 tables, DATE 2026 accepted paper
>
> **摘要:** On-device fine-tuning of CNNs is essential to withstand domain shift in edge applications such as Human Activity Recognition (HAR), yet full fine-tuning is infeasible under strict memory, compute, and energy budgets. We present LoRA-Edge, a parameter-efficient fine-tuning (PEFT) method that builds on Low-Rank Adaptation (LoRA) with tensor-train assistance. LoRA-Edge (i) applies Tensor-Train Singular Value Decomposition (TT-SVD) to pre-trained convolutional layers, (ii) selectively updates only the output-side core with zero-initialization to keep the auxiliary path inactive at the start, and (iii) fuses the update back into dense kernels, leaving inference cost unchanged. This design preserves convolutional structure and reduces the number of trainable parameters by up to two orders of magnitude compared to full fine-tuning. Across diverse HAR datasets and CNN backbones, LoRA-Edge achieves accuracy within 4.7% of full fine-tuning while updating at most 1.49% of parameters, consistently outperforming prior parameter-efficient baselines under similar budgets. On a Jetson Orin Nano, TT-SVD initialization and selective-core training yield 1.4-3.8x faster convergence to target F1. LoRA-Edge thus makes structure-aligned, parameter-efficient on-device CNN adaptation practical for edge platforms.
>
---
#### [replaced 027] FunOTTA: On-the-Fly Adaptation on Cross-Domain Fundus Image via Stable Test-time Training
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.04396v3](http://arxiv.org/pdf/2407.04396v3)**

> **作者:** Qian Zeng; Le Zhang; Yipeng Liu; Ce Zhu; Fan Zhang
>
> **备注:** 13 pages, 8 figures, 7 tables
>
> **摘要:** Fundus images are essential for the early screening and detection of eye diseases. While deep learning models using fundus images have significantly advanced the diagnosis of multiple eye diseases, variations in images from different imaging devices and locations (known as domain shifts) pose challenges for deploying pre-trained models in real-world applications. To address this, we propose a novel Fundus On-the-fly Test-Time Adaptation (FunOTTA) framework that effectively generalizes a fundus image diagnosis model to unseen environments, even under strong domain shifts. FunOTTA stands out for its stable adaptation process by performing dynamic disambiguation in the memory bank while minimizing harmful prior knowledge bias. We also introduce a new training objective during adaptation that enables the classifier to incrementally adapt to target patterns with reliable class conditional estimation and consistency regularization. We compare our method with several state-of-the-art test-time adaptation (TTA) pipelines. Experiments on cross-domain fundus image benchmarks across two diseases demonstrate the superiority of the overall framework and individual components under different backbone networks. Code is available at https://github.com/Casperqian/FunOTTA.
>
---
#### [replaced 028] Faithful Contouring: Near-Lossless 3D Voxel Representation Free from Iso-surface
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2511.04029v2](http://arxiv.org/pdf/2511.04029v2)**

> **作者:** Yihao Luo; Xianglong He; Chuanyu Pan; Yiwen Chen; Jiaqi Wu; Yangguang Li; Wanli Ouyang; Yuanming Hu; Guang Yang; ChoonHwai Yap
>
> **摘要:** Accurate and efficient voxelized representations of 3D meshes are the foundation of 3D reconstruction and generation. However, existing representations based on iso-surface heavily rely on water-tightening or rendering optimization, which inevitably compromise geometric fidelity. We propose Faithful Contouring, a sparse voxelized representation that supports 2048+ resolutions for arbitrary meshes, requiring neither converting meshes to field functions nor extracting the isosurface during remeshing. It achieves near-lossless fidelity by preserving sharpness and internal structures, even for challenging cases with complex geometry and topology. The proposed method also shows flexibility for texturing, manipulation, and editing. Beyond representation, we design a dual-mode autoencoder for Faithful Contouring, enabling scalable and detail-preserving shape reconstruction. Extensive experiments show that Faithful Contouring surpasses existing methods in accuracy and efficiency for both representation and reconstruction. For direct representation, it achieves distance errors at the $10^{-5}$ level; for mesh reconstruction, it yields a 93\% reduction in Chamfer Distance and a 35\% improvement in F-score over strong baselines, confirming superior fidelity as a representation for 3D learning tasks.
>
---
#### [replaced 029] Learning to Navigate Socially Through Proactive Risk Perception
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.07871v3](http://arxiv.org/pdf/2510.07871v3)**

> **作者:** Erjia Xiao; Lingfeng Zhang; Yingbo Tang; Hao Cheng; Renjing Xu; Wenbo Ding; Lei Zhou; Long Chen; Hangjun Ye; Xiaoshuai Hao
>
> **摘要:** In this report, we describe the technical details of our submission to the IROS 2025 RoboSense Challenge Social Navigation Track. This track focuses on developing RGBD-based perception and navigation systems that enable autonomous agents to navigate safely, efficiently, and socially compliantly in dynamic human-populated indoor environments. The challenge requires agents to operate from an egocentric perspective using only onboard sensors including RGB-D observations and odometry, without access to global maps or privileged information, while maintaining social norm compliance such as safe distances and collision avoidance. Building upon the Falcon model, we introduce a Proactive Risk Perception Module to enhance social navigation performance. Our approach augments Falcon with collision risk understanding that learns to predict distance-based collision risk scores for surrounding humans, which enables the agent to develop more robust spatial awareness and proactive collision avoidance behaviors. The evaluation on the Social-HM3D benchmark demonstrates that our method improves the agent's ability to maintain personal space compliance while navigating toward goals in crowded indoor scenes with dynamic human agents, achieving 2nd place among 16 participating teams in the challenge.
>
---
#### [replaced 030] Dark Transformer: A Video Transformer for Action Recognition in the Dark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.12805v2](http://arxiv.org/pdf/2407.12805v2)**

> **作者:** Anwaar Ulhaq
>
> **备注:** 8 Figures, 12 Pages
>
> **摘要:** Recognizing human actions in adverse lighting conditions presents significant challenges in computer vision, with wide-ranging applications in visual surveillance and nighttime driving. Existing methods tackle action recognition and dark enhancement separately, limiting the potential for end-to-end learning of spatiotemporal representations for video action classification. This paper introduces Dark Transformer, a novel video transformer-based approach for action recognition in low-light environments. Dark Transformer leverages spatiotemporal self-attention mechanisms in cross-domain settings to enhance cross-domain action recognition. By extending video transformers to learn cross-domain knowledge, Dark Transformer achieves state-of-the-art performance on benchmark action recognition datasets, including InFAR, XD145, and ARID. The proposed approach demonstrates significant promise in addressing the challenges of action recognition in adverse lighting conditions, offering practical implications for real-world applications.
>
---
#### [replaced 031] NVIDIA Nemotron Nano V2 VL
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2511.03929v2](http://arxiv.org/pdf/2511.03929v2)**

> **作者:** NVIDIA; :; Amala Sanjay Deshmukh; Kateryna Chumachenko; Tuomas Rintamaki; Matthieu Le; Tyler Poon; Danial Mohseni Taheri; Ilia Karmanov; Guilin Liu; Jarno Seppanen; Guo Chen; Karan Sapra; Zhiding Yu; Adi Renduchintala; Charles Wang; Peter Jin; Arushi Goel; Mike Ranzinger; Lukas Voegtle; Philipp Fischer; Timo Roman; Wei Ping; Boxin Wang; Zhuolin Yang; Nayeon Lee; Shaokun Zhang; Fuxiao Liu; Zhiqi Li; Di Zhang; Greg Heinrich; Hongxu Yin; Song Han; Pavlo Molchanov; Parth Mannan; Yao Xu; Jane Polak Scowcroft; Tom Balough; Subhashree Radhakrishnan; Paris Zhang; Sean Cha; Ratnesh Kumar; Zaid Pervaiz Bhat; Jian Zhang; Darragh Hanley; Pritam Biswas; Jesse Oliver; Kevin Vasques; Roger Waleffe; Duncan Riach; Oluwatobi Olabiyi; Ameya Sunil Mahabaleshwarkar; Bilal Kartal; Pritam Gundecha; Khanh Nguyen; Alexandre Milesi; Eugene Khvedchenia; Ran Zilberstein; Ofri Masad; Natan Bagrov; Nave Assaf; Tomer Asida; Daniel Afrimi; Amit Zuker; Netanel Haber; Zhiyu Cheng; Jingyu Xin; Di Wu; Nik Spirin; Maryam Moosaei; Roman Ageev; Vanshil Atul Shah; Yuting Wu; Daniel Korzekwa; Unnikrishnan Kizhakkemadam Sreekumar; Wanli Jiang; Padmavathy Subramanian; Alejandra Rico; Sandip Bhaskar; Saeid Motiian; Kedi Wu; Annie Surla; Chia-Chih Chen; Hayden Wolff; Matthew Feinberg; Melissa Corpuz; Marek Wawrzos; Eileen Long; Aastha Jhunjhunwala; Paul Hendricks; Farzan Memarian; Benika Hall; Xin-Yu Wang; David Mosallanezhad; Soumye Singhal; Luis Vega; Katherine Cheung; Krzysztof Pawelec; Michael Evans; Katherine Luna; Jie Lou; Erick Galinkin; Akshay Hazare; Kaustubh Purandare; Ann Guan; Anna Warno; Chen Cui; Yoshi Suhara; Shibani Likhite; Seph Mard; Meredith Price; Laya Sleiman; Saori Kaji; Udi Karpas; Kari Briski; Joey Conway; Michael Lightstone; Jan Kautz; Mohammad Shoeybi; Mostofa Patwary; Jonathen Cohen; Oleksii Kuchaiev; Andrew Tao; Bryan Catanzaro
>
> **摘要:** We introduce Nemotron Nano V2 VL, the latest model of the Nemotron vision-language series designed for strong real-world document understanding, long video comprehension, and reasoning tasks. Nemotron Nano V2 VL delivers significant improvements over our previous model, Llama-3.1-Nemotron-Nano-VL-8B, across all vision and text domains through major enhancements in model architecture, datasets, and training recipes. Nemotron Nano V2 VL builds on Nemotron Nano V2, a hybrid Mamba-Transformer LLM, and innovative token reduction techniques to achieve higher inference throughput in long document and video scenarios. We are releasing model checkpoints in BF16, FP8, and FP4 formats and sharing large parts of our datasets, recipes and training code.
>
---
#### [replaced 032] MOSAIC: Generating Consistent, Privacy-Preserving Scenes from Multiple Depth Views in Multi-Room Environments
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13816v3](http://arxiv.org/pdf/2503.13816v3)**

> **作者:** Zhixuan Liu; Haokun Zhu; Rui Chen; Jonathan Francis; Soonmin Hwang; Ji Zhang; Jean Oh
>
> **摘要:** We introduce a diffusion-based approach for generating privacy-preserving digital twins of multi-room indoor environments from depth images only. Central to our approach is a novel Multi-view Overlapped Scene Alignment with Implicit Consistency (MOSAIC) model that explicitly considers cross-view dependencies within the same scene in the probabilistic sense. MOSAIC operates through a multi-channel inference-time optimization that avoids error accumulation common in sequential or single-room constraints in panorama-based approaches. MOSAIC scales to complex scenes with zero extra training and provably reduces the variance during denoising process when more overlapping views are added, leading to improved generation quality. Experiments show that MOSAIC outperforms state-of-the-art baselines on image fidelity metrics in reconstructing complex multi-room environments. Resources and code are at https://mosaic-cmubig.github.io
>
---
#### [replaced 033] MMDocIR: Benchmarking Multimodal Retrieval for Long Documents
- **分类: cs.IR; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08828v3](http://arxiv.org/pdf/2501.08828v3)**

> **作者:** Kuicai Dong; Yujing Chang; Xin Deik Goh; Dexun Li; Ruiming Tang; Yong Liu
>
> **备注:** Paper accepted to EMNLP-2025(Main)
>
> **摘要:** Multimodal document retrieval aims to identify and retrieve various forms of multimodal content, such as figures, tables, charts, and layout information from extensive documents. Despite its increasing popularity, there is a notable lack of a comprehensive and robust benchmark to effectively evaluate the performance of systems in such tasks. To address this gap, this work introduces a new benchmark, named MMDocIR, that encompasses two distinct tasks: page-level and layout-level retrieval. The former evaluates the performance of identifying the most relevant pages within a long document, while the later assesses the ability of detecting specific layouts, providing a more fine-grained measure than whole-page analysis. A layout refers to a variety of elements, including textual paragraphs, equations, figures, tables, or charts. The MMDocIR benchmark comprises a rich dataset featuring 1,685 questions annotated by experts and 173,843 questions with bootstrapped labels, making it a valuable resource in multimodal document retrieval for both training and evaluation. Through rigorous experiments, we demonstrate that (i) visual retrievers significantly outperform their text counterparts, (ii) MMDocIR training set effectively enhances the performance of multimodal document retrieval and (iii) text retrievers leveraging VLM-text significantly outperforms retrievers relying on OCR-text. Our dataset is available at https://mmdocrag.github.io/MMDocIR/.
>
---
#### [replaced 034] GAITEX: Human motion dataset of impaired gait and rehabilitation exercises using inertial and optical sensors
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.21069v2](http://arxiv.org/pdf/2507.21069v2)**

> **作者:** Andreas Spilz; Heiko Oppel; Jochen Werner; Kathrin Stucke-Straub; Felix Capanni; Michael Munz
>
> **摘要:** Wearable inertial measurement units (IMUs) provide a cost-effective approach to assessing human movement in clinical and everyday environments. However, developing the associated classification models for robust assessment of physiotherapeutic exercise and gait analysis requires large, diverse datasets that are costly and time-consuming to collect. We present a multimodal dataset of physiotherapeutic and gait-related exercises, including correct and clinically relevant variants, recorded from 19 healthy subjects using synchronized IMUs and optical marker-based motion capture (MoCap). It contains data from nine IMUs and 68 markers tracking full-body kinematics. Four markers per IMU allow direct comparison between IMU- and MoCap-derived orientations. We additionally provide processed IMU orientations aligned to common segment coordinate systems, subject-specific OpenSim models, inverse kinematics outputs, and visualization tools for IMU-derived orientations. The dataset is fully annotated with movement quality ratings and timestamped segmentations. It supports various machine learning tasks such as exercise evaluation, gait classification, temporal segmentation, and biomechanical parameter estimation. Code for postprocessing, alignment, inverse kinematics, and technical validation is provided to promote reproducibility.
>
---
#### [replaced 035] Dual Teacher-Student Learning for Semi-supervised Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11018v2](http://arxiv.org/pdf/2505.11018v2)**

> **作者:** Pengchen Zhang; Alan J. X. Guo; Sipin Luo; Zhe Han; Lin Guo
>
> **摘要:** Semi-supervised learning reduces the costly manual annotation burden in medical image segmentation. A popular approach is the mean teacher (MT) strategy, which applies consistency regularization using a temporally averaged teacher model. In this work, the MT strategy is reinterpreted as a form of self-paced learning in the context of supervised learning, where agreement between the teacher's predictions and the ground truth implicitly guides the model from easy to hard. Extending this insight to semi-supervised learning, we propose dual teacher-student learning (DTSL). It regulates the learning pace on unlabeled data using two signals: a temporally averaged signal from an in-group teacher and a cross-architectural signal from a student in a second, distinct model group. Specifically, a novel consensus label generator (CLG) creates the pseudo-labels from the agreement between these two signals, establishing an effective learning curriculum. Extensive experiments on four benchmark datasets demonstrate that the proposed method consistently outperforms existing state-of-the-art approaches. Remarkably, on three of the four datasets, our semi-supervised method with limited labeled data surpasses its fully supervised counterparts, validating the effectiveness of our self-paced learning design.
>
---
#### [replaced 036] Diffusion Denoised Hyperspectral Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21890v2](http://arxiv.org/pdf/2505.21890v2)**

> **作者:** Sunil Kumar Narayanan; Lingjun Zhao; Lu Gan; Yongsheng Chen
>
> **备注:** Accepted to 3DV 2026
>
> **摘要:** Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise determination of nutritional elements of samples. Recently, 3D reconstruction methods have been used to create implicit neural representations of HSI scenes, which can help localize the target object's nutrient composition spatially and spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit representation that can be used to render hyperspectral channel compositions of each spatial location from any viewing direction. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of hyperspectral scenes across the full spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of DD-HGS. The results demonstrate that DD-HGS achieves new state-of-the-art performance among previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
>
---
