# 计算机视觉 cs.CV

- **最新发布 99 篇**

- **更新 74 篇**

## 最新发布

#### [new 001] Temporally-Aware Supervised Contrastive Learning for Polyp Counting in Colonoscopy
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决结肠镜下息肉计数问题。通过引入时间感知的监督对比学习，提升息肉跟踪与聚类的准确性。**

- **链接: [http://arxiv.org/pdf/2507.02493v1](http://arxiv.org/pdf/2507.02493v1)**

> **作者:** Luca Parolari; Andrea Cherubini; Lamberto Ballan; Carlo Biffi
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Automated polyp counting in colonoscopy is a crucial step toward automated procedure reporting and quality control, aiming to enhance the cost-effectiveness of colonoscopy screening. Counting polyps in a procedure involves detecting and tracking polyps, and then clustering tracklets that belong to the same polyp entity. Existing methods for polyp counting rely on self-supervised learning and primarily leverage visual appearance, neglecting temporal relationships in both tracklet feature learning and clustering stages. In this work, we introduce a paradigm shift by proposing a supervised contrastive loss that incorporates temporally-aware soft targets. Our approach captures intra-polyp variability while preserving inter-polyp discriminability, leading to more robust clustering. Additionally, we improve tracklet clustering by integrating a temporal adjacency constraint, reducing false positive re-associations between visually similar but temporally distant tracklets. We train and validate our method on publicly available datasets and evaluate its performance with a leave-one-out cross-validation strategy. Results demonstrate a 2.2x reduction in fragmentation rate compared to prior approaches. Our results highlight the importance of temporal awareness in polyp counting, establishing a new state-of-the-art. Code is available at https://github.com/lparolari/temporally-aware-polyp-counting.
>
---
#### [new 002] AnyI2V: Animating Any Conditional Image with Motion Control
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决动态运动与空间约束控制不足的问题。提出AnyI2V框架，无需训练即可根据用户定义的运动轨迹生成视频。**

- **链接: [http://arxiv.org/pdf/2507.02857v1](http://arxiv.org/pdf/2507.02857v1)**

> **作者:** Ziye Li; Hao Luo; Xincheng Shuai; Henghui Ding
>
> **备注:** ICCV 2025, Project Page: https://henghuiding.com/AnyI2V/
>
> **摘要:** Recent advancements in video generation, particularly in diffusion models, have driven notable progress in text-to-video (T2V) and image-to-video (I2V) synthesis. However, challenges remain in effectively integrating dynamic motion signals and flexible spatial constraints. Existing T2V methods typically rely on text prompts, which inherently lack precise control over the spatial layout of generated content. In contrast, I2V methods are limited by their dependence on real images, which restricts the editability of the synthesized content. Although some methods incorporate ControlNet to introduce image-based conditioning, they often lack explicit motion control and require computationally expensive training. To address these limitations, we propose AnyI2V, a training-free framework that animates any conditional images with user-defined motion trajectories. AnyI2V supports a broader range of modalities as the conditional image, including data types such as meshes and point clouds that are not supported by ControlNet, enabling more flexible and versatile video generation. Additionally, it supports mixed conditional inputs and enables style transfer and editing via LoRA and text prompts. Extensive experiments demonstrate that the proposed AnyI2V achieves superior performance and provides a new perspective in spatial- and motion-controlled video generation. Code is available at https://henghuiding.com/AnyI2V/.
>
---
#### [new 003] LMPNet for Weakly-supervised Keypoint Discovery
- **分类: cs.CV**

- **简介: 该论文属于弱监督关键点检测任务，旨在通过类别标签自动发现语义关键点。提出LMPNet模型，利用滤波器学习非重复局部模式，并通过聚类生成关键点预测。**

- **链接: [http://arxiv.org/pdf/2507.02308v1](http://arxiv.org/pdf/2507.02308v1)**

> **作者:** Pei Guo; Ryan Farrell
>
> **摘要:** In this work, we explore the task of semantic object keypoint discovery weakly-supervised by only category labels. This is achieved by transforming discriminatively-trained intermediate layer filters into keypoint detectors. We begin by identifying three preferred characteristics of keypoint detectors: (i) spatially sparse activations, (ii) consistency and (iii) diversity. Instead of relying on hand-crafted loss terms, a novel computationally-efficient leaky max pooling (LMP) layer is proposed to explicitly encourage final conv-layer filters to learn "non-repeatable local patterns" that are well aligned with object keypoints. Informed by visualizations, a simple yet effective selection strategy is proposed to ensure consistent filter activations and attention mask-out is then applied to force the network to distribute its attention to the whole object instead of just the most discriminative region. For the final keypoint prediction, a learnable clustering layer is proposed to group keypoint proposals into keypoint predictions. The final model, named LMPNet, is highly interpretable in that it directly manipulates network filters to detect predefined concepts. Our experiments show that LMPNet can (i) automatically discover semantic keypoints that are robust to object pose and (ii) achieves strong prediction accuracy comparable to a supervised pose estimation model.
>
---
#### [new 004] ViRefSAM: Visual Reference-Guided Segment Anything Model for Remote Sensing Segmentation
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分割任务，解决SAM在遥感场景中需手动提示和域适应差的问题。通过引入参考图像实现自动分割，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.02294v1](http://arxiv.org/pdf/2507.02294v1)**

> **作者:** Hanbo Bi; Yulong Xu; Ya Li; Yongqiang Mao; Boyuan Tong; Chongyang Li; Chunbo Lang; Wenhui Diao; Hongqi Wang; Yingchao Feng; Xian Sun
>
> **摘要:** The Segment Anything Model (SAM), with its prompt-driven paradigm, exhibits strong generalization in generic segmentation tasks. However, applying SAM to remote sensing (RS) images still faces two major challenges. First, manually constructing precise prompts for each image (e.g., points or boxes) is labor-intensive and inefficient, especially in RS scenarios with dense small objects or spatially fragmented distributions. Second, SAM lacks domain adaptability, as it is pre-trained primarily on natural images and struggles to capture RS-specific semantics and spatial characteristics, especially when segmenting novel or unseen classes. To address these issues, inspired by few-shot learning, we propose ViRefSAM, a novel framework that guides SAM utilizing only a few annotated reference images that contain class-specific objects. Without requiring manual prompts, ViRefSAM enables automatic segmentation of class-consistent objects across RS images. Specifically, ViRefSAM introduces two key components while keeping SAM's original architecture intact: (1) a Visual Contextual Prompt Encoder that extracts class-specific semantic clues from reference images and generates object-aware prompts via contextual interaction with target images; and (2) a Dynamic Target Alignment Adapter, integrated into SAM's image encoder, which mitigates the domain gap by injecting class-specific semantics into target image features, enabling SAM to dynamically focus on task-relevant regions. Extensive experiments on three few-shot segmentation benchmarks, including iSAID-5$^i$, LoveDA-2$^i$, and COCO-20$^i$, demonstrate that ViRefSAM enables accurate and automatic segmentation of unseen classes by leveraging only a few reference images and consistently outperforms existing few-shot segmentation methods across diverse datasets.
>
---
#### [new 005] Understanding Trade offs When Conditioning Synthetic Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决合成数据生成中的质量与控制问题。通过对比不同条件策略，提升合成数据在目标检测中的效果。**

- **链接: [http://arxiv.org/pdf/2507.02217v1](http://arxiv.org/pdf/2507.02217v1)**

> **作者:** Brandon Trabucco; Qasim Wani; Benjamin Pikus; Vasu Sharma
>
> **摘要:** Learning robust object detectors from only a handful of images is a critical challenge in industrial vision systems, where collecting high quality training data can take months. Synthetic data has emerged as a key solution for data efficient visual inspection and pick and place robotics. Current pipelines rely on 3D engines such as Blender or Unreal, which offer fine control but still require weeks to render a small dataset, and the resulting images often suffer from a large gap between simulation and reality. Diffusion models promise a step change because they can generate high quality images in minutes, yet precise control, especially in low data regimes, remains difficult. Although many adapters now extend diffusion beyond plain text prompts, the effect of different conditioning schemes on synthetic data quality is poorly understood. We study eighty diverse visual concepts drawn from four standard object detection benchmarks and compare two conditioning strategies: prompt based and layout based. When the set of conditioning cues is narrow, prompt conditioning yields higher quality synthetic data; as diversity grows, layout conditioning becomes superior. When layout cues match the full training distribution, synthetic data raises mean average precision by an average of thirty four percent and by as much as one hundred seventy seven percent compared with using real data alone.
>
---
#### [new 006] LaCo: Efficient Layer-wise Compression of Visual Tokens for Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型的视觉令牌压缩任务，旨在提升压缩效率。提出LaCo框架，在视觉编码器中间层实现高效令牌压缩。**

- **链接: [http://arxiv.org/pdf/2507.02279v1](http://arxiv.org/pdf/2507.02279v1)**

> **作者:** Juntao Liu; Liqiang Niu; Wenchao Chen; Jie Zhou; Fandong Meng
>
> **摘要:** Existing visual token compression methods for Multimodal Large Language Models (MLLMs) predominantly operate as post-encoder modules, limiting their potential for efficiency gains. To address this limitation, we propose LaCo (Layer-wise Visual Token Compression), a novel framework that enables effective token compression within the intermediate layers of the vision encoder. LaCo introduces two core components: 1) a layer-wise pixel-shuffle mechanism that systematically merges adjacent tokens through space-to-channel transformations, and 2) a residual learning architecture with non-parametric shortcuts that preserves critical visual information during compression. Extensive experiments indicate that our LaCo outperforms all existing methods when compressing tokens in the intermediate layers of the vision encoder, demonstrating superior effectiveness. In addition, compared to external compression, our method improves training efficiency beyond 20% and inference throughput over 15% while maintaining strong performance.
>
---
#### [new 007] Flow-CDNet: A Novel Network for Detecting Both Slow and Fast Changes in Bitemporal Images
- **分类: cs.CV**

- **简介: 该论文属于变化检测任务，旨在同时检测遥感图像中的慢变和快变区域。提出Flow-CDNet网络，结合光流与残差网络，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.02307v1](http://arxiv.org/pdf/2507.02307v1)**

> **作者:** Haoxuan Li; Chenxu Wei; Haodong Wang; Xiaomeng Hu; Boyuan An; Lingyan Ran; Baosen Zhang; Jin Jin; Omirzhan Taukebayev; Amirkhan Temirbayev; Junrui Liu; Xiuwei Zhang
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Change detection typically involves identifying regions with changes between bitemporal images taken at the same location. Besides significant changes, slow changes in bitemporal images are also important in real-life scenarios. For instance, weak changes often serve as precursors to major hazards in scenarios like slopes, dams, and tailings ponds. Therefore, designing a change detection network that simultaneously detects slow and fast changes presents a novel challenge. In this paper, to address this challenge, we propose a change detection network named Flow-CDNet, consisting of two branches: optical flow branch and binary change detection branch. The first branch utilizes a pyramid structure to extract displacement changes at multiple scales. The second one combines a ResNet-based network with the optical flow branch's output to generate fast change outputs. Subsequently, to supervise and evaluate this new change detection framework, a self-built change detection dataset Flow-Change, a loss function combining binary tversky loss and L2 norm loss, along with a new evaluation metric called FEPE are designed. Quantitative experiments conducted on Flow-Change dataset demonstrated that our approach outperforms the existing methods. Furthermore, ablation experiments verified that the two branches can promote each other to enhance the detection performance.
>
---
#### [new 008] Addressing Camera Sensors Faults in Vision-Based Navigation: Simulation and Dataset Development
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉导航任务，旨在解决相机传感器故障导致的导航可靠性问题。通过仿真和数据集开发，支持AI故障检测算法的研究。**

- **链接: [http://arxiv.org/pdf/2507.02602v1](http://arxiv.org/pdf/2507.02602v1)**

> **作者:** Riccardo Gallon; Fabian Schiemenz; Alessandra Menicucci; Eberhard Gill
>
> **备注:** Submitted to Acta Astronautica
>
> **摘要:** The increasing importance of Vision-Based Navigation (VBN) algorithms in space missions raises numerous challenges in ensuring their reliability and operational robustness. Sensor faults can lead to inaccurate outputs from navigation algorithms or even complete data processing faults, potentially compromising mission objectives. Artificial Intelligence (AI) offers a powerful solution for detecting such faults, overcoming many of the limitations associated with traditional fault detection methods. However, the primary obstacle to the adoption of AI in this context is the lack of sufficient and representative datasets containing faulty image data. This study addresses these challenges by focusing on an interplanetary exploration mission scenario. A comprehensive analysis of potential fault cases in camera sensors used within the VBN pipeline is presented. The causes and effects of these faults are systematically characterized, including their impact on image quality and navigation algorithm performance, as well as commonly employed mitigation strategies. To support this analysis, a simulation framework is introduced to recreate faulty conditions in synthetically generated images, enabling a systematic and controlled reproduction of faulty data. The resulting dataset of fault-injected images provides a valuable tool for training and testing AI-based fault detection algorithms. The final link to the dataset will be added after an embargo period. For peer-reviewers, this private link is available.
>
---
#### [new 009] CrowdTrack: A Benchmark for Difficult Multiple Pedestrian Tracking in Real Scenarios
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多行人跟踪任务，旨在解决复杂场景下跟踪效果差的问题。提出CrowdTrack数据集，包含真实场景的33个视频和5185条轨迹，用于提升算法在复杂环境中的性能。**

- **链接: [http://arxiv.org/pdf/2507.02479v1](http://arxiv.org/pdf/2507.02479v1)**

> **作者:** Teng Fu; Yuwen Chen; Zhuofan Chen; Mengyang Zhao; Bin Li; Xiangyang Xue
>
> **摘要:** Multi-object tracking is a classic field in computer vision. Among them, pedestrian tracking has extremely high application value and has become the most popular research category. Existing methods mainly use motion or appearance information for tracking, which is often difficult in complex scenarios. For the motion information, mutual occlusions between objects often prevent updating of the motion state; for the appearance information, non-robust results are often obtained due to reasons such as only partial visibility of the object or blurred images. Although learning how to perform tracking in these situations from the annotated data is the simplest solution, the existing MOT dataset fails to satisfy this solution. Existing methods mainly have two drawbacks: relatively simple scene composition and non-realistic scenarios. Although some of the video sequences in existing dataset do not have the above-mentioned drawbacks, the number is far from adequate for research purposes. To this end, we propose a difficult large-scale dataset for multi-pedestrian tracking, shot mainly from the first-person view and all from real-life complex scenarios. We name it ``CrowdTrack'' because there are numerous objects in most of the sequences. Our dataset consists of 33 videos, containing a total of 5,185 trajectories. Each object is annotated with a complete bounding box and a unique object ID. The dataset will provide a platform to facilitate the development of algorithms that remain effective in complex situations. We analyzed the dataset comprehensively and tested multiple SOTA models on our dataset. Besides, we analyzed the performance of the foundation models on our dataset. The dataset and project code is released at: https://github.com/loseevaya/CrowdTrack .
>
---
#### [new 010] Wildlife Target Re-Identification Using Self-supervised Learning in Non-Urban Settings
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于野生动物个体再识别任务，解决标注数据依赖问题。通过自监督学习从相机陷阱数据中提取特征，提升模型在有限数据下的鲁棒性与性能。**

- **链接: [http://arxiv.org/pdf/2507.02403v1](http://arxiv.org/pdf/2507.02403v1)**

> **作者:** Mufhumudzi Muthivhi; Terence L. van Zyl
>
> **备注:** Accepted for publication in IEEE Xplore and ISIF FUSION 2025 proceedings:
>
> **摘要:** Wildlife re-identification aims to match individuals of the same species across different observations. Current state-of-the-art (SOTA) models rely on class labels to train supervised models for individual classification. This dependence on annotated data has driven the curation of numerous large-scale wildlife datasets. This study investigates self-supervised learning Self-Supervised Learning (SSL) for wildlife re-identification. We automatically extract two distinct views of an individual using temporal image pairs from camera trap data without supervision. The image pairs train a self-supervised model from a potentially endless stream of video data. We evaluate the learnt representations against supervised features on open-world scenarios and transfer learning in various wildlife downstream tasks. The analysis of the experimental results shows that self-supervised models are more robust even with limited data. Moreover, self-supervised features outperform supervision across all downstream tasks. The code is available here https://github.com/pxpana/SSLWildlife.
>
---
#### [new 011] UniMC: Taming Diffusion Transformer for Unified Keypoint-Guided Multi-Class Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决非刚性物体（如动物）及多实例生成难题。提出UniMC框架和HAIG-2.9M数据集以提升生成效果。**

- **链接: [http://arxiv.org/pdf/2507.02713v1](http://arxiv.org/pdf/2507.02713v1)**

> **作者:** Qin Guo; Ailing Zeng; Dongxu Yue; Ceyuan Yang; Yang Cao; Hanzhong Guo; Fei Shen; Wei Liu; Xihui Liu; Dan Xu
>
> **摘要:** Although significant advancements have been achieved in the progress of keypoint-guided Text-to-Image diffusion models, existing mainstream keypoint-guided models encounter challenges in controlling the generation of more general non-rigid objects beyond humans (e.g., animals). Moreover, it is difficult to generate multiple overlapping humans and animals based on keypoint controls solely. These challenges arise from two main aspects: the inherent limitations of existing controllable methods and the lack of suitable datasets. First, we design a DiT-based framework, named UniMC, to explore unifying controllable multi-class image generation. UniMC integrates instance- and keypoint-level conditions into compact tokens, incorporating attributes such as class, bounding box, and keypoint coordinates. This approach overcomes the limitations of previous methods that struggled to distinguish instances and classes due to their reliance on skeleton images as conditions. Second, we propose HAIG-2.9M, a large-scale, high-quality, and diverse dataset designed for keypoint-guided human and animal image generation. HAIG-2.9M includes 786K images with 2.9M instances. This dataset features extensive annotations such as keypoints, bounding boxes, and fine-grained captions for both humans and animals, along with rigorous manual inspection to ensure annotation accuracy. Extensive experiments demonstrate the high quality of HAIG-2.9M and the effectiveness of UniMC, particularly in heavy occlusions and multi-class scenarios.
>
---
#### [new 012] Heeding the Inner Voice: Aligning ControlNet Training via Intermediate Features Feedback
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在提升生成图像的空间控制精度。针对现有方法忽略中间阶段的问题，提出InnerControl，通过中间特征反馈增强对齐效果。**

- **链接: [http://arxiv.org/pdf/2507.02321v1](http://arxiv.org/pdf/2507.02321v1)**

> **作者:** Nina Konovalova; Maxim Nikolaev; Andrey Kuznetsov; Aibek Alanov
>
> **备注:** code available at https://github.com/ControlGenAI/InnerControl
>
> **摘要:** Despite significant progress in text-to-image diffusion models, achieving precise spatial control over generated outputs remains challenging. ControlNet addresses this by introducing an auxiliary conditioning module, while ControlNet++ further refines alignment through a cycle consistency loss applied only to the final denoising steps. However, this approach neglects intermediate generation stages, limiting its effectiveness. We propose InnerControl, a training strategy that enforces spatial consistency across all diffusion steps. Our method trains lightweight convolutional probes to reconstruct input control signals (e.g., edges, depth) from intermediate UNet features at every denoising step. These probes efficiently extract signals even from highly noisy latents, enabling pseudo ground truth controls for training. By minimizing the discrepancy between predicted and target conditions throughout the entire diffusion process, our alignment loss improves both control fidelity and generation quality. Combined with established techniques like ControlNet++, InnerControl achieves state-of-the-art performance across diverse conditioning methods (e.g., edges, depth).
>
---
#### [new 013] Determination Of Structural Cracks Using Deep Learning Frameworks
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于结构裂缝检测任务，旨在解决人工检测效率低、误差大的问题。通过改进的残差U-Net模型及集成学习方法提升检测精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.02416v1](http://arxiv.org/pdf/2507.02416v1)**

> **作者:** Subhasis Dasgupta; Jaydip Sen; Tuhina Halder
>
> **备注:** This is the accepted version of the paper presented in IEEE CONIT 2025 held on 20th June 2025. This is not the camera-ready version. There are 6 pages in this paper and it contains 7 figures and 1 table
>
> **摘要:** Structural crack detection is a critical task for public safety as it helps in preventing potential structural failures that could endanger lives. Manual detection by inexperienced personnel can be slow, inconsistent, and prone to human error, which may compromise the reliability of assessments. The current study addresses these challenges by introducing a novel deep-learning architecture designed to enhance the accuracy and efficiency of structural crack detection. In this research, various configurations of residual U-Net models were utilized. These models, due to their robustness in capturing fine details, were further integrated into an ensemble with a meta-model comprising convolutional blocks. This unique combination aimed to boost prediction efficiency beyond what individual models could achieve. The ensemble's performance was evaluated against well-established architectures such as SegNet and the traditional U-Net. Results demonstrated that the residual U-Net models outperformed their predecessors, particularly with low-resolution imagery, and the ensemble model exceeded the performance of individual models, proving it as the most effective. The assessment was based on the Intersection over Union (IoU) metric and DICE coefficient. The ensemble model achieved the highest scores, signifying superior accuracy. This advancement suggests way for more reliable automated systems in structural defects monitoring tasks.
>
---
#### [new 014] ESTR-CoT: Towards Explainable and Accurate Event Stream based Scene Text Recognition with Chain-of-Thought Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于事件流场景文本识别任务，旨在解决识别准确性和可解释性不足的问题。提出ESTR-CoT框架，结合视觉编码与语言模型，实现端到端优化和逻辑推理。**

- **链接: [http://arxiv.org/pdf/2507.02200v1](http://arxiv.org/pdf/2507.02200v1)**

> **作者:** Xiao Wang; Jingtao Jiang; Qiang Chen; Lan Chen; Lin Zhu; Yaowei Wang; Yonghong Tian; Jin Tang
>
> **备注:** A Strong Baseline for Reasoning based Event Stream Scene Text Recognition
>
> **摘要:** Event stream based scene text recognition is a newly arising research topic in recent years which performs better than the widely used RGB cameras in extremely challenging scenarios, especially the low illumination, fast motion. Existing works either adopt end-to-end encoder-decoder framework or large language models for enhanced recognition, however, they are still limited by the challenges of insufficient interpretability and weak contextual logical reasoning. In this work, we propose a novel chain-of-thought reasoning based event stream scene text recognition framework, termed ESTR-CoT. Specifically, we first adopt the vision encoder EVA-CLIP (ViT-G/14) to transform the input event stream into tokens and utilize a Llama tokenizer to encode the given generation prompt. A Q-former is used to align the vision token to the pre-trained large language model Vicuna-7B and output both the answer and chain-of-thought (CoT) reasoning process simultaneously. Our framework can be optimized using supervised fine-tuning in an end-to-end manner. In addition, we also propose a large-scale CoT dataset to train our framework via a three stage processing (i.e., generation, polish, and expert verification). This dataset provides a solid data foundation for the development of subsequent reasoning-based large models. Extensive experiments on three event stream STR benchmark datasets (i.e., EventSTR, WordArt*, IC15*) fully validated the effectiveness and interpretability of our proposed framework. The source code and pre-trained models will be released on https://github.com/Event-AHU/ESTR-CoT.
>
---
#### [new 015] Neural Network-based Study for Rice Leaf Disease Recognition and Classification: A Comparative Analysis Between Feature-based Model and Direct Imaging Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于水稻叶部病害识别任务，旨在解决传统方法效果不足的问题。通过对比特征分析模型与直接图像模型，验证特征提取方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.02322v1](http://arxiv.org/pdf/2507.02322v1)**

> **作者:** Farida Siddiqi Prity; Mirza Raquib; Saydul Akbar Murad; Md. Jubayar Alam Rafi; Md. Khairul Bashar Bhuiyan; Anupam Kumar Bairagi
>
> **摘要:** Rice leaf diseases significantly reduce productivity and cause economic losses, highlighting the need for early detection to enable effective management and improve yields. This study proposes Artificial Neural Network (ANN)-based image-processing techniques for timely classification and recognition of rice diseases. Despite the prevailing approach of directly inputting images of rice leaves into ANNs, there is a noticeable absence of thorough comparative analysis between the Feature Analysis Detection Model (FADM) and Direct Image-Centric Detection Model (DICDM), specifically when it comes to evaluating the effectiveness of Feature Extraction Algorithms (FEAs). Hence, this research presents initial experiments on the Feature Analysis Detection Model, utilizing various image Feature Extraction Algorithms, Dimensionality Reduction Algorithms (DRAs), Feature Selection Algorithms (FSAs), and Extreme Learning Machine (ELM). The experiments are carried out on datasets encompassing bacterial leaf blight, brown spot, leaf blast, leaf scald, Sheath blight rot, and healthy leaf, utilizing 10-fold Cross-Validation method. A Direct Image-Centric Detection Model is established without the utilization of any FEA, and the evaluation of classification performance relies on different metrics. Ultimately, an exhaustive contrast is performed between the achievements of the Feature Analysis Detection Model and Direct Image-Centric Detection Model in classifying rice leaf diseases. The results reveal that the highest performance is attained using the Feature Analysis Detection Model. The adoption of the proposed Feature Analysis Detection Model for detecting rice leaf diseases holds excellent potential for improving crop health, minimizing yield losses, and enhancing overall productivity and sustainability of rice farming.
>
---
#### [new 016] Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Deepfake Video Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度伪造视频检测任务，旨在解决传统方法忽视像素级时间不一致的问题。通过提取像素级时间频率特征并结合注意力机制，提升检测精度。**

- **链接: [http://arxiv.org/pdf/2507.02398v1](http://arxiv.org/pdf/2507.02398v1)**

> **作者:** Taehoon Kim; Jongwook Choi; Yonghyun Jeong; Haeun Noh; Jaejun Yoo; Seungryul Baek; Jongwon Choi
>
> **备注:** accepted by iccv 2025. code is will be available at https://github.com/rama0126/PwTF-DVD
>
> **摘要:** We introduce a deepfake video detection approach that exploits pixel-wise temporal inconsistencies, which traditional spatial frequency-based detectors often overlook. Traditional detectors represent temporal information merely by stacking spatial frequency spectra across frames, resulting in the failure to detect temporal artifacts in the pixel plane. Our approach performs a 1D Fourier transform on the time axis for each pixel, extracting features highly sensitive to temporal inconsistencies, especially in areas prone to unnatural movements. To precisely locate regions containing the temporal artifacts, we introduce an attention proposal module trained in an end-to-end manner. Additionally, our joint transformer module effectively integrates pixel-wise temporal frequency features with spatio-temporal context features, expanding the range of detectable forgery artifacts. Our framework represents a significant advancement in deepfake video detection, providing robust performance across diverse and challenging detection scenarios.
>
---
#### [new 017] Are Synthetic Videos Useful? A Benchmark for Retrieval-Centric Evaluation of Synthetic Videos
- **分类: cs.CV**

- **简介: 该论文属于文本到视频生成任务，旨在评估合成视频在检索中的实用性。通过构建SynTVA数据集，研究其语义对齐与检索性能的关系。**

- **链接: [http://arxiv.org/pdf/2507.02316v1](http://arxiv.org/pdf/2507.02316v1)**

> **作者:** Zecheng Zhao; Selena Song; Tong Chen; Zhi Chen; Shazia Sadiq; Yadan Luo
>
> **备注:** 7 pages, 10 figures
>
> **摘要:** Text-to-video (T2V) synthesis has advanced rapidly, yet current evaluation metrics primarily capture visual quality and temporal consistency, offering limited insight into how synthetic videos perform in downstream tasks such as text-to-video retrieval (TVR). In this work, we introduce SynTVA, a new dataset and benchmark designed to evaluate the utility of synthetic videos for building retrieval models. Based on 800 diverse user queries derived from MSRVTT training split, we generate synthetic videos using state-of-the-art T2V models and annotate each video-text pair along four key semantic alignment dimensions: Object \& Scene, Action, Attribute, and Prompt Fidelity. Our evaluation framework correlates general video quality assessment (VQA) metrics with these alignment scores, and examines their predictive power for downstream TVR performance. To explore pathways of scaling up, we further develop an Auto-Evaluator to estimate alignment quality from existing metrics. Beyond benchmarking, our results show that SynTVA is a valuable asset for dataset augmentation, enabling the selection of high-utility synthetic samples that measurably improve TVR outcomes. Project page and dataset can be found at https://jasoncodemaker.github.io/SynTVA/.
>
---
#### [new 018] APT: Adaptive Personalized Training for Diffusion Models with Limited Data
- **分类: cs.CV; cs.AI; 60J60, 68T07; I.2.6; I.2.10; I.4.9**

- **简介: 该论文属于扩散模型个性化任务，解决有限数据下的过拟合和先验知识丢失问题。提出APT框架，通过自适应训练、表征稳定和注意力对齐提升生成质量。**

- **链接: [http://arxiv.org/pdf/2507.02687v1](http://arxiv.org/pdf/2507.02687v1)**

> **作者:** JungWoo Chae; Jiyoon Kim; JaeWoong Choi; Kyungyul Kim; Sangheum Hwang
>
> **备注:** CVPR 2025 camera ready. Project page: https://lgcnsai.github.io/apt
>
> **摘要:** Personalizing diffusion models using limited data presents significant challenges, including overfitting, loss of prior knowledge, and degradation of text alignment. Overfitting leads to shifts in the noise prediction distribution, disrupting the denoising trajectory and causing the model to lose semantic coherence. In this paper, we propose Adaptive Personalized Training (APT), a novel framework that mitigates overfitting by employing adaptive training strategies and regularizing the model's internal representations during fine-tuning. APT consists of three key components: (1) Adaptive Training Adjustment, which introduces an overfitting indicator to detect the degree of overfitting at each time step bin and applies adaptive data augmentation and adaptive loss weighting based on this indicator; (2)Representation Stabilization, which regularizes the mean and variance of intermediate feature maps to prevent excessive shifts in noise prediction; and (3) Attention Alignment for Prior Knowledge Preservation, which aligns the cross-attention maps of the fine-tuned model with those of the pretrained model to maintain prior knowledge and semantic coherence. Through extensive experiments, we demonstrate that APT effectively mitigates overfitting, preserves prior knowledge, and outperforms existing methods in generating high-quality, diverse images with limited reference data.
>
---
#### [new 019] Continual Multiple Instance Learning with Enhanced Localization for Histopathological Whole Slide Image Analysis
- **分类: cs.CV**

- **简介: 该论文属于病理图像分析任务，解决持续多实例学习中的定位与遗忘问题。提出CoMEL框架，提升定位准确性和模型适应性。**

- **链接: [http://arxiv.org/pdf/2507.02395v1](http://arxiv.org/pdf/2507.02395v1)**

> **作者:** Byung Hyun Lee; Wongi Jeong; Woojae Han; Kyoungbun Lee; Se Young Chun
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Multiple instance learning (MIL) significantly reduced annotation costs via bag-level weak labels for large-scale images, such as histopathological whole slide images (WSIs). However, its adaptability to continual tasks with minimal forgetting has been rarely explored, especially on instance classification for localization. Weakly incremental learning for semantic segmentation has been studied for continual localization, but it focused on natural images, leveraging global relationships among hundreds of small patches (e.g., $16 \times 16$) using pre-trained models. This approach seems infeasible for MIL localization due to enormous amounts ($\sim 10^5$) of large patches (e.g., $256 \times 256$) and no available global relationships such as cancer cells. To address these challenges, we propose Continual Multiple Instance Learning with Enhanced Localization (CoMEL), an MIL framework for both localization and adaptability with minimal forgetting. CoMEL consists of (1) Grouped Double Attention Transformer (GDAT) for efficient instance encoding, (2) Bag Prototypes-based Pseudo-Labeling (BPPL) for reliable instance pseudo-labeling, and (3) Orthogonal Weighted Low-Rank Adaptation (OWLoRA) to mitigate forgetting in both bag and instance classification. Extensive experiments on three public WSI datasets demonstrate superior performance of CoMEL, outperforming the prior arts by up to $11.00\%$ in bag-level accuracy and up to $23.4\%$ in localization accuracy under the continual MIL setup.
>
---
#### [new 020] MAC-Lookup: Multi-Axis Conditional Lookup Model for Underwater Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于 underwater image enhancement 任务，旨在解决水下图像颜色失真和细节模糊问题。提出 MAC-Lookup 模型，结合颜色校正与多轴增强，提升图像质量。**

- **链接: [http://arxiv.org/pdf/2507.02270v1](http://arxiv.org/pdf/2507.02270v1)**

> **作者:** Fanghai Yi; Zehong Zheng; Zexiao Liang; Yihang Dong; Xiyang Fang; Wangyu Wu; Xuhang Chen
>
> **备注:** Accepted by IEEE SMC 2025
>
> **摘要:** Enhancing underwater images is crucial for exploration. These images face visibility and color issues due to light changes, water turbidity, and bubbles. Traditional prior-based methods and pixel-based methods often fail, while deep learning lacks sufficient high-quality datasets. We introduce the Multi-Axis Conditional Lookup (MAC-Lookup) model, which enhances visual quality by improving color accuracy, sharpness, and contrast. It includes Conditional 3D Lookup Table Color Correction (CLTCC) for preliminary color and quality correction and Multi-Axis Adaptive Enhancement (MAAE) for detail refinement. This model prevents over-enhancement and saturation while handling underwater challenges. Extensive experiments show that MAC-Lookup excels in enhancing underwater images by restoring details and colors better than existing methods. The code is https://github.com/onlycatdoraemon/MAC-Lookup.
>
---
#### [new 021] Parametric shape models for vessels learned from segmentations via differentiable voxelization
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在整合不同表示形式的血管结构。通过可微分体素化，从分割数据中学习参数化形状模型，实现高保真网格重建。**

- **链接: [http://arxiv.org/pdf/2507.02576v1](http://arxiv.org/pdf/2507.02576v1)**

> **作者:** Alina F. Dima; Suprosanna Shit; Huaqi Qiu; Robbie Holland; Tamara T. Mueller; Fabio Antonio Musio; Kaiyuan Yang; Bjoern Menze; Rickmer Braren; Marcus Makowski; Daniel Rueckert
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Vessels are complex structures in the body that have been studied extensively in multiple representations. While voxelization is the most common of them, meshes and parametric models are critical in various applications due to their desirable properties. However, these representations are typically extracted through segmentations and used disjointly from each other. We propose a framework that joins the three representations under differentiable transformations. By leveraging differentiable voxelization, we automatically extract a parametric shape model of the vessels through shape-to-segmentation fitting, where we learn shape parameters from segmentations without the explicit need for ground-truth shape parameters. The vessel is parametrized as centerlines and radii using cubic B-splines, ensuring smoothness and continuity by construction. Meshes are differentiably extracted from the learned shape parameters, resulting in high-fidelity meshes that can be manipulated post-fit. Our method can accurately capture the geometry of complex vessels, as demonstrated by the volumetric fits in experiments on aortas, aneurysms, and brain vessels.
>
---
#### [new 022] PosDiffAE: Position-aware Diffusion Auto-encoder For High-Resolution Brain Tissue Classification Incorporating Artifact Restoration
- **分类: cs.CV**

- **简介: 该论文属于高分辨率脑组织分类任务，旨在解决图像噪声和伪影问题。通过结合扩散模型与自编码器，构建位置感知的潜在空间，并实现无监督的伪影修复。**

- **链接: [http://arxiv.org/pdf/2507.02405v1](http://arxiv.org/pdf/2507.02405v1)**

> **作者:** Ayantika Das; Moitreya Chaudhuri; Koushik Bhat; Keerthi Ram; Mihail Bota; Mohanasankar Sivaprakasam
>
> **备注:** Published in IEEE Journal of Biomedical and Health Informatics (Early Access Available) https://ieeexplore.ieee.org/document/10989734
>
> **摘要:** Denoising diffusion models produce high-fidelity image samples by capturing the image distribution in a progressive manner while initializing with a simple distribution and compounding the distribution complexity. Although these models have unlocked new applicabilities, the sampling mechanism of diffusion does not offer means to extract image-specific semantic representation, which is inherently provided by auto-encoders. The encoding component of auto-encoders enables mapping between a specific image and its latent space, thereby offering explicit means of enforcing structures in the latent space. By integrating an encoder with the diffusion model, we establish an auto-encoding formulation, which learns image-specific representations and offers means to organize the latent space. In this work, First, we devise a mechanism to structure the latent space of a diffusion auto-encoding model, towards recognizing region-specific cellular patterns in brain images. We enforce the representations to regress positional information of the patches from high-resolution images. This creates a conducive latent space for differentiating tissue types of the brain. Second, we devise an unsupervised tear artifact restoration technique based on neighborhood awareness, utilizing latent representations and the constrained generation capability of diffusion models during inference. Third, through representational guidance and leveraging the inference time steerable noising and denoising capability of diffusion, we devise an unsupervised JPEG artifact restoration technique.
>
---
#### [new 023] Confidence-driven Gradient Modulation for Multimodal Human Activity Recognition: A Dynamic Contrastive Dual-Path Learning Approach
- **分类: cs.CV**

- **简介: 该论文属于多模态人体活动识别任务，旨在解决跨模态对齐和模态贡献不平衡问题。提出DCDP-HAR框架，包含双路径结构、对比学习和置信度梯度调制策略。**

- **链接: [http://arxiv.org/pdf/2507.02826v1](http://arxiv.org/pdf/2507.02826v1)**

> **作者:** Panpan Ji; Junni Song; Hang Xiao; Hanyu Liu; Chao Li
>
> **摘要:** Sensor-based Human Activity Recognition (HAR) is a core technology that enables intelligent systems to perceive and interact with their environment. However, multimodal HAR systems still encounter key challenges, such as difficulties in cross-modal feature alignment and imbalanced modality contributions. To address these issues, we propose a novel framework called the Dynamic Contrastive Dual-Path Network (DCDP-HAR). The framework comprises three key components. First, a dual-path feature extraction architecture is employed, where ResNet and DenseNet branches collaboratively process multimodal sensor data. Second, a multi-stage contrastive learning mechanism is introduced to achieve progressive alignment from local perception to semantic abstraction. Third, we present a confidence-driven gradient modulation strategy that dynamically monitors and adjusts the learning intensity of each modality branch during backpropagation, effectively alleviating modality competition. In addition, a momentum-based gradient accumulation strategy is adopted to enhance training stability. We conduct ablation studies to validate the effectiveness of each component and perform extensive comparative experiments on four public benchmark datasets.
>
---
#### [new 024] SurgVisAgent: Multimodal Agentic Model for Versatile Surgical Visual Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像增强任务，旨在解决手术中多种视觉干扰问题。通过构建多模态智能代理模型，实现多样化的图像优化。**

- **链接: [http://arxiv.org/pdf/2507.02252v1](http://arxiv.org/pdf/2507.02252v1)**

> **作者:** Zeyu Lei; Hongyuan Yu; Jinlin Wu; Zhen Chen
>
> **摘要:** Precise surgical interventions are vital to patient safety, and advanced enhancement algorithms have been developed to assist surgeons in decision-making. Despite significant progress, these algorithms are typically designed for single tasks in specific scenarios, limiting their effectiveness in complex real-world situations. To address this limitation, we propose SurgVisAgent, an end-to-end intelligent surgical vision agent built on multimodal large language models (MLLMs). SurgVisAgent dynamically identifies distortion categories and severity levels in endoscopic images, enabling it to perform a variety of enhancement tasks such as low-light enhancement, overexposure correction, motion blur elimination, and smoke removal. Specifically, to achieve superior surgical scenario understanding, we design a prior model that provides domain-specific knowledge. Additionally, through in-context few-shot learning and chain-of-thought (CoT) reasoning, SurgVisAgent delivers customized image enhancements tailored to a wide range of distortion types and severity levels, thereby addressing the diverse requirements of surgeons. Furthermore, we construct a comprehensive benchmark simulating real-world surgical distortions, on which extensive experiments demonstrate that SurgVisAgent surpasses traditional single-task models, highlighting its potential as a unified solution for surgical assistance.
>
---
#### [new 025] Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于安全攻击任务，旨在通过视觉上下文注入破解多模态大语言模型，提出VisCo攻击方法提升攻击效果。**

- **链接: [http://arxiv.org/pdf/2507.02844v1](http://arxiv.org/pdf/2507.02844v1)**

> **作者:** Ziqi Miao; Yi Ding; Lijun Li; Jing Shao
>
> **备注:** 16 pages
>
> **摘要:** With the emergence of strong visual-language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: visual-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct visual-focused strategies, dynamically generating auxiliary images when necessary to construct a visual-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which performs a toxicity score of 2.48 and an ASR of 22.2%. The code is available at https://github.com/Dtc7w3PQ/Visco-Attack.
>
---
#### [new 026] IGDNet: Zero-Shot Robust Underexposed Image Enhancement via Illumination-Guided and Denoising
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像增强任务，旨在解决无监督下欠曝图像修复问题。提出IGDNet方法，无需配对数据，通过分解与去噪模块提升图像质量。**

- **链接: [http://arxiv.org/pdf/2507.02445v1](http://arxiv.org/pdf/2507.02445v1)**

> **作者:** Hailong Yan; Junjian Huang; Tingwen Huang
>
> **备注:** Submitted to IEEE Transactions on Artificial Intelligence (TAI) on Oct.31, 2024
>
> **摘要:** Current methods for restoring underexposed images typically rely on supervised learning with paired underexposed and well-illuminated images. However, collecting such datasets is often impractical in real-world scenarios. Moreover, these methods can lead to over-enhancement, distorting well-illuminated regions. To address these issues, we propose IGDNet, a Zero-Shot enhancement method that operates solely on a single test image, without requiring guiding priors or training data. IGDNet exhibits strong generalization ability and effectively suppresses noise while restoring illumination. The framework comprises a decomposition module and a denoising module. The former separates the image into illumination and reflection components via a dense connection network, while the latter enhances non-uniformly illuminated regions using an illumination-guided pixel adaptive correction method. A noise pair is generated through downsampling and refined iteratively to produce the final result. Extensive experiments on four public datasets demonstrate that IGDNet significantly improves visual quality under complex lighting conditions. Quantitative results on metrics like PSNR (20.41dB) and SSIM (0.860dB) show that it outperforms 14 state-of-the-art unsupervised methods. The code will be released soon.
>
---
#### [new 027] Prompt Disentanglement via Language Guidance and Representation Alignment for Domain Generalization
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于领域泛化任务，旨在解决跨领域特征不变性问题。通过语言引导和表示对齐，设计有效提示以提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.02288v1](http://arxiv.org/pdf/2507.02288v1)**

> **作者:** De Cheng; Zhipeng Xu; Xinyang Jiang; Dongsheng Li; Nannan Wang; Xinbo Gao
>
> **摘要:** Domain Generalization (DG) seeks to develop a versatile model capable of performing effectively on unseen target domains. Notably, recent advances in pre-trained Visual Foundation Models (VFMs), such as CLIP, have demonstrated considerable potential in enhancing the generalization capabilities of deep learning models. Despite the increasing attention toward VFM-based domain prompt tuning within DG, the effective design of prompts capable of disentangling invariant features across diverse domains remains a critical challenge. In this paper, we propose addressing this challenge by leveraging the controllable and flexible language prompt of the VFM. Noting that the text modality of VFMs is naturally easier to disentangle, we introduce a novel framework for text feature-guided visual prompt tuning. This framework first automatically disentangles the text prompt using a large language model (LLM) and then learns domain-invariant visual representation guided by the disentangled text feature. However, relying solely on language to guide visual feature disentanglement has limitations, as visual features can sometimes be too complex or nuanced to be fully captured by descriptive text. To address this, we introduce Worst Explicit Representation Alignment (WERA), which extends text-guided visual prompts by incorporating an additional set of abstract prompts. These prompts enhance source domain diversity through stylized image augmentations, while alignment constraints ensure that visual representations remain consistent across both the original and augmented distributions. Experiments conducted on major DG datasets, including PACS, VLCS, OfficeHome, DomainNet, and TerraInc, demonstrate that our proposed method outperforms state-of-the-art DG methods.
>
---
#### [new 028] MC-INR: Efficient Encoding of Multivariate Scientific Simulation Data using Meta-Learning and Clustered Implicit Neural Representations
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于科学数据编码任务，旨在解决INR在处理多变量非结构化数据时的局限性。提出MC-INR框架，结合元学习与聚类实现高效编码。**

- **链接: [http://arxiv.org/pdf/2507.02494v1](http://arxiv.org/pdf/2507.02494v1)**

> **作者:** Hyunsoo Son; Jeonghyun Noh; Suemin Jeon; Chaoli Wang; Won-Ki Jeong
>
> **备注:** 5 pages
>
> **摘要:** Implicit Neural Representations (INRs) are widely used to encode data as continuous functions, enabling the visualization of large-scale multivariate scientific simulation data with reduced memory usage. However, existing INR-based methods face three main limitations: (1) inflexible representation of complex structures, (2) primarily focusing on single-variable data, and (3) dependence on structured grids. Thus, their performance degrades when applied to complex real-world datasets. To address these limitations, we propose a novel neural network-based framework, MC-INR, which handles multivariate data on unstructured grids. It combines meta-learning and clustering to enable flexible encoding of complex structures. To further improve performance, we introduce a residual-based dynamic re-clustering mechanism that adaptively partitions clusters based on local error. We also propose a branched layer to leverage multivariate data through independent branches simultaneously. Experimental results demonstrate that MC-INR outperforms existing methods on scientific data encoding tasks.
>
---
#### [new 029] Underwater Monocular Metric Depth Estimation: Real-World Benchmarks and Synthetic Fine-Tuning
- **分类: cs.CV**

- **简介: 该论文属于 underwater monocular metric depth estimation 任务，旨在解决水下环境深度估计可靠性低的问题。通过构建基准和合成数据微调模型，提升水下深度预测性能。**

- **链接: [http://arxiv.org/pdf/2507.02148v1](http://arxiv.org/pdf/2507.02148v1)**

> **作者:** Zijie Cai; Christopher Metzler
>
> **摘要:** Monocular depth estimation has recently advanced to provide not only relative but also metric depth predictions. However, its reliability in underwater environments remains limited due to light attenuation and scattering, color distortion, turbidity, and the lack of high-quality metric ground-truth data. In this paper, we present a comprehensive benchmark of zero-shot and fine-tuned monocular metric depth estimation models on real-world underwater datasets with metric depth annotations, such as FLSea and SQUID. We evaluate a diverse set of state-of-the-art models across a range of underwater conditions with different ranges. Our results show that large-scale models trained on terrestrial (real or synthetic) data, while effective in in-air settings, perform poorly underwater due to significant domain shifts. To address this, we fine-tune Depth Anything V2 with a ViT-S backbone encoder on a synthetic underwater variant of the Hypersim dataset, which we generated using a physically based underwater image formation model. We demonstrate our fine-tuned model consistently improves performance across all benchmarks and outperforms baselines trained only on the clean in-air Hypersim dataset. Our study provides a detailed evaluation and visualization for monocular metric depth estimation in underwater scenes, highlighting the importance of domain adaptation and scale-aware supervision for achieving robust and generalizable metric depth predictions in challenging underwater environments for future research.
>
---
#### [new 030] A Novel Tuning Method for Real-time Multiple-Object Tracking Utilizing Thermal Sensor with Complexity Motion Pattern
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，解决热成像中因低级特征表达不足导致的跟踪困难问题，提出一种优化双阶段超参数的调优方法。**

- **链接: [http://arxiv.org/pdf/2507.02408v1](http://arxiv.org/pdf/2507.02408v1)**

> **作者:** Duong Nguyen-Ngoc Tran; Long Hoang Pham; Chi Dai Tran; Quoc Pham-Nam Ho; Huy-Hung Nguyen; Jae Wook Jeon
>
> **摘要:** Multi-Object Tracking in thermal images is essential for surveillance systems, particularly in challenging environments where RGB cameras struggle due to low visibility or poor lighting conditions. Thermal sensors enhance recognition tasks by capturing infrared signatures, but a major challenge is their low-level feature representation, which makes it difficult to accurately detect and track pedestrians. To address this, the paper introduces a novel tuning method for pedestrian tracking, specifically designed to handle the complex motion patterns in thermal imagery. The proposed framework optimizes two-stages, ensuring that each stage is tuned with the most suitable hyperparameters to maximize tracking performance. By fine-tuning hyperparameters for real-time tracking, the method achieves high accuracy without relying on complex reidentification or motion models. Extensive experiments on PBVS Thermal MOT dataset demonstrate that the approach is highly effective across various thermal camera conditions, making it a robust solution for real-world surveillance applications.
>
---
#### [new 031] IMASHRIMP: Automatic White Shrimp (Penaeus vannamei) Biometrical Analysis from Laboratory Images Using Computer Vision and Deep Learning
- **分类: cs.CV; I.2.10; I.4.8**

- **简介: 该论文属于图像分析任务，旨在自动化白虾形态分析，解决人工误差和效率问题。通过深度学习和计算机视觉技术，实现精准测量与分类。**

- **链接: [http://arxiv.org/pdf/2507.02519v1](http://arxiv.org/pdf/2507.02519v1)**

> **作者:** Abiam Remache González; Meriem Chagour; Timon Bijan Rüth; Raúl Trapiella Cañedo; Marina Martínez Soler; Álvaro Lorenzo Felipe; Hyun-Suk Shin; María-Jesús Zamorano Serrano; Ricardo Torres; Juan-Antonio Castillo Parra; Eduardo Reyes Abad; Miguel-Ángel Ferrer Ballester; Juan-Manuel Afonso López; Francisco-Mario Hernández Tejera; Adrian Penate-Sanchez
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** This paper introduces IMASHRIMP, an adapted system for the automated morphological analysis of white shrimp (Penaeus vannamei}, aimed at optimizing genetic selection tasks in aquaculture. Existing deep learning and computer vision techniques were modified to address the specific challenges of shrimp morphology analysis from RGBD images. IMASHRIMP incorporates two discrimination modules, based on a modified ResNet-50 architecture, to classify images by the point of view and determine rostrum integrity. It is proposed a "two-factor authentication (human and IA)" system, it reduces human error in view classification from 0.97% to 0% and in rostrum detection from 12.46% to 3.64%. Additionally, a pose estimation module was adapted from VitPose to predict 23 key points on the shrimp's skeleton, with separate networks for lateral and dorsal views. A morphological regression module, using a Support Vector Machine (SVM) model, was integrated to convert pixel measurements to centimeter units. Experimental results show that the system effectively reduces human error, achieving a mean average precision (mAP) of 97.94% for pose estimation and a pixel-to-centimeter conversion error of 0.07 (+/- 0.1) cm. IMASHRIMP demonstrates the potential to automate and accelerate shrimp morphological analysis, enhancing the efficiency of genetic selection and contributing to more sustainable aquaculture practices.The code are available at https://github.com/AbiamRemacheGonzalez/ImaShrimp-public
>
---
#### [new 032] Cross-domain Hyperspectral Image Classification based on Bi-directional Domain Adaptation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于跨域高光谱图像分类任务，解决不同场景下类别光谱偏移问题。提出BiDA框架，通过双向域适应提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.02268v1](http://arxiv.org/pdf/2507.02268v1)**

> **作者:** Yuxiang Zhang; Wei Li; Wen Jia; Mengmeng Zhang; Ran Tao; Shunlin Liang
>
> **摘要:** Utilizing hyperspectral remote sensing technology enables the extraction of fine-grained land cover classes. Typically, satellite or airborne images used for training and testing are acquired from different regions or times, where the same class has significant spectral shifts in different scenes. In this paper, we propose a Bi-directional Domain Adaptation (BiDA) framework for cross-domain hyperspectral image (HSI) classification, which focuses on extracting both domain-invariant features and domain-specific information in the independent adaptive space, thereby enhancing the adaptability and separability to the target scene. In the proposed BiDA, a triple-branch transformer architecture (the source branch, target branch, and coupled branch) with semantic tokenizer is designed as the backbone. Specifically, the source branch and target branch independently learn the adaptive space of source and target domains, a Coupled Multi-head Cross-attention (CMCA) mechanism is developed in coupled branch for feature interaction and inter-domain correlation mining. Furthermore, a bi-directional distillation loss is designed to guide adaptive space learning using inter-domain correlation. Finally, we propose an Adaptive Reinforcement Strategy (ARS) to encourage the model to focus on specific generalized feature extraction within both source and target scenes in noise condition. Experimental results on cross-temporal/scene airborne and satellite datasets demonstrate that the proposed BiDA performs significantly better than some state-of-the-art domain adaptation approaches. In the cross-temporal tree species classification task, the proposed BiDA is more than 3\%$\sim$5\% higher than the most advanced method. The codes will be available from the website: https://github.com/YuxiangZhang-BIT/IEEE_TCSVT_BiDA.
>
---
#### [new 033] Multi-Label Classification Framework for Hurricane Damage Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多标签分类任务，旨在解决飓风后损伤类型复杂、传统方法不足的问题。通过融合ResNet和注意力机制，提升损伤评估的准确性。**

- **链接: [http://arxiv.org/pdf/2507.02265v1](http://arxiv.org/pdf/2507.02265v1)**

> **作者:** Zhangding Liu; Neda Mohammadi; John E. Taylor
>
> **备注:** 9 pages, 3 figures. Accepted at the ASCE International Conference on Computing in Civil Engineering (i3CE 2025)
>
> **摘要:** Hurricanes cause widespread destruction, resulting in diverse damage types and severities that require timely and accurate assessment for effective disaster response. While traditional single-label classification methods fall short of capturing the complexity of post-hurricane damage, this study introduces a novel multi-label classification framework for assessing damage using aerial imagery. The proposed approach integrates a feature extraction module based on ResNet and a class-specific attention mechanism to identify multiple damage types within a single image. Using the Rescuenet dataset from Hurricane Michael, the proposed method achieves a mean average precision of 90.23%, outperforming existing baseline methods. This framework enhances post-hurricane damage assessment, enabling more targeted and efficient disaster response and contributing to future strategies for disaster mitigation and resilience. This paper has been accepted at the ASCE International Conference on Computing in Civil Engineering (i3CE 2025), and the camera-ready version will appear in the official conference proceedings.
>
---
#### [new 034] F^2TTA: Free-Form Test-Time Adaptation on Cross-Domain Medical Image Classification via Image-Level Disentangled Prompt Tuning
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究自由形式的测试时适应（F²TTA）任务，旨在解决跨域医学图像分类中数据片段不规则的问题。提出I-DiPT框架，结合图像不变和特定提示，提升模型适应能力。**

- **链接: [http://arxiv.org/pdf/2507.02437v1](http://arxiv.org/pdf/2507.02437v1)**

> **作者:** Wei Li; Jingyang Zhang; Lihao Liu; Guoan Wang; Junjun He; Yang Chen; Lixu Gu
>
> **备注:** This paper has been submitted to relevant journals
>
> **摘要:** Test-Time Adaptation (TTA) has emerged as a promising solution for adapting a source model to unseen medical sites using unlabeled test data, due to the high cost of data annotation. Existing TTA methods consider scenarios where data from one or multiple domains arrives in complete domain units. However, in clinical practice, data usually arrives in domain fragments of arbitrary lengths and in random arrival orders, due to resource constraints and patient variability. This paper investigates a practical Free-Form Test-Time Adaptation (F$^{2}$TTA) task, where a source model is adapted to such free-form domain fragments, with shifts occurring between fragments unpredictably. In this setting, these shifts could distort the adaptation process. To address this problem, we propose a novel Image-level Disentangled Prompt Tuning (I-DiPT) framework. I-DiPT employs an image-invariant prompt to explore domain-invariant representations for mitigating the unpredictable shifts, and an image-specific prompt to adapt the source model to each test image from the incoming fragments. The prompts may suffer from insufficient knowledge representation since only one image is available for training. To overcome this limitation, we first introduce Uncertainty-oriented Masking (UoM), which encourages the prompts to extract sufficient information from the incoming image via masked consistency learning driven by the uncertainty of the source model representations. Then, we further propose a Parallel Graph Distillation (PGD) method that reuses knowledge from historical image-specific and image-invariant prompts through parallel graph networks. Experiments on breast cancer and glaucoma classification demonstrate the superiority of our method over existing TTA approaches in F$^{2}$TTA. Code is available at https://github.com/mar-cry/F2TTA.
>
---
#### [new 035] Team RAS in 9th ABAW Competition: Multimodal Compound Expression Recognition Approach
- **分类: cs.CV**

- **简介: 该论文属于情感计算中的复合表情识别任务，旨在无需领域适应的情况下准确识别复杂情绪。通过多模态融合与零样本学习方法提升识别效果。**

- **链接: [http://arxiv.org/pdf/2507.02205v1](http://arxiv.org/pdf/2507.02205v1)**

> **作者:** Elena Ryumina; Maxim Markitantov; Alexandr Axyonov; Dmitry Ryumin; Mikhail Dolgushin; Alexey Karpov
>
> **备注:** 8
>
> **摘要:** Compound Expression Recognition (CER), a subfield of affective computing, aims to detect complex emotional states formed by combinations of basic emotions. In this work, we present a novel zero-shot multimodal approach for CER that combines six heterogeneous modalities into a single pipeline: static and dynamic facial expressions, scene and label matching, scene context, audio, and text. Unlike previous approaches relying on task-specific training data, our approach uses zero-shot components, including Contrastive Language-Image Pretraining (CLIP)-based label matching and Qwen-VL for semantic scene understanding. We further introduce a Multi-Head Probability Fusion (MHPF) module that dynamically weights modality-specific predictions, followed by a Compound Expressions (CE) transformation module that uses Pair-Wise Probability Aggregation (PPA) and Pair-Wise Feature Similarity Aggregation (PFSA) methods to produce interpretable compound emotion outputs. Evaluated under multi-corpus training, the proposed approach shows F1 scores of 46.95% on AffWild2, 49.02% on Acted Facial Expressions in The Wild (AFEW), and 34.85% on C-EXPR-DB via zero-shot testing, which is comparable to the results of supervised approaches trained on target data. This demonstrates the effectiveness of the proposed approach for capturing CE without domain adaptation. The source code is publicly available.
>
---
#### [new 036] Perception Activator: An intuitive and portable framework for brain cognitive exploration
- **分类: cs.CV**

- **简介: 该论文属于脑视觉解码任务，旨在解决现有方法在语义对齐上的不足。通过引入fMRI信号提升目标检测与分割性能，验证了其在多对象语义和空间定位上的价值。**

- **链接: [http://arxiv.org/pdf/2507.02311v1](http://arxiv.org/pdf/2507.02311v1)**

> **作者:** Le Xu; Qi Zhang; Qixian Zhang; Hongyun Zhang; Duoqian Miao; Cairong Zhao
>
> **摘要:** Recent advances in brain-vision decoding have driven significant progress, reconstructing with high fidelity perceived visual stimuli from neural activity, e.g., functional magnetic resonance imaging (fMRI), in the human visual cortex. Most existing methods decode the brain signal using a two-level strategy, i.e., pixel-level and semantic-level. However, these methods rely heavily on low-level pixel alignment yet lack sufficient and fine-grained semantic alignment, resulting in obvious reconstruction distortions of multiple semantic objects. To better understand the brain's visual perception patterns and how current decoding models process semantic objects, we have developed an experimental framework that uses fMRI representations as intervention conditions. By injecting these representations into multi-scale image features via cross-attention, we compare both downstream performance and intermediate feature changes on object detection and instance segmentation tasks with and without fMRI information. Our results demonstrate that incorporating fMRI signals enhances the accuracy of downstream detection and segmentation, confirming that fMRI contains rich multi-object semantic cues and coarse spatial localization information-elements that current models have yet to fully exploit or integrate.
>
---
#### [new 037] LiteReality: Graphics-Ready 3D Scene Reconstruction from RGB-D Scans
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文提出LiteReality，解决室内场景3D重建任务，通过RGB-D扫描生成紧凑、真实且可交互的3D虚拟场景。**

- **链接: [http://arxiv.org/pdf/2507.02861v1](http://arxiv.org/pdf/2507.02861v1)**

> **作者:** Zhening Huang; Xiaoyang Wu; Fangcheng Zhong; Hengshuang Zhao; Matthias Nießner; Joan Lasenby
>
> **备注:** Project Page: https://litereality.github.io; Video: https://www.youtube.com/watch?v=ecK9m3LXg2c&feature=youtu.be
>
> **摘要:** We propose LiteReality, a novel pipeline that converts RGB-D scans of indoor environments into compact, realistic, and interactive 3D virtual replicas. LiteReality not only reconstructs scenes that visually resemble reality but also supports key features essential for graphics pipelines -- such as object individuality, articulation, high-quality physically based rendering materials, and physically based interaction. At its core, LiteReality first performs scene understanding and parses the results into a coherent 3D layout and objects with the help of a structured scene graph. It then reconstructs the scene by retrieving the most visually similar 3D artist-crafted models from a curated asset database. Next, the Material Painting module enhances realism by recovering high-quality, spatially varying materials. Finally, the reconstructed scene is integrated into a simulation engine with basic physical properties to enable interactive behavior. The resulting scenes are compact, editable, and fully compatible with standard graphics pipelines, making them suitable for applications in AR/VR, gaming, robotics, and digital twins. In addition, LiteReality introduces a training-free object retrieval module that achieves state-of-the-art similarity performance on the Scan2CAD benchmark, along with a robust material painting module capable of transferring appearances from images of any style to 3D assets -- even under severe misalignment, occlusion, and poor lighting. We demonstrate the effectiveness of LiteReality on both real-life scans and public datasets. Project page: https://litereality.github.io; Video: https://www.youtube.com/watch?v=ecK9m3LXg2c
>
---
#### [new 038] FMOcc: TPV-Driven Flow Matching for 3D Occupancy Prediction with Selective State Space Model
- **分类: cs.CV**

- **简介: 该论文属于3D语义占用预测任务，旨在解决遮挡和远距离场景预测准确率低的问题。提出FMOcc模型，结合流匹配与选择性状态空间模型提升性能。**

- **链接: [http://arxiv.org/pdf/2507.02250v1](http://arxiv.org/pdf/2507.02250v1)**

> **作者:** Jiangxia Chen; Tongyuan Huang; Ke Song
>
> **摘要:** 3D semantic occupancy prediction plays a pivotal role in autonomous driving. However, inherent limitations of fewframe images and redundancy in 3D space compromise prediction accuracy for occluded and distant scenes. Existing methods enhance performance by fusing historical frame data, which need additional data and significant computational resources. To address these issues, this paper propose FMOcc, a Tri-perspective View (TPV) refinement occupancy network with flow matching selective state space model for few-frame 3D occupancy prediction. Firstly, to generate missing features, we designed a feature refinement module based on a flow matching model, which is called Flow Matching SSM module (FMSSM). Furthermore, by designing the TPV SSM layer and Plane Selective SSM (PS3M), we selectively filter TPV features to reduce the impact of air voxels on non-air voxels, thereby enhancing the overall efficiency of the model and prediction capability for distant scenes. Finally, we design the Mask Training (MT) method to enhance the robustness of FMOcc and address the issue of sensor data loss. Experimental results on the Occ3D-nuScenes and OpenOcc datasets show that our FMOcc outperforms existing state-of-theart methods. Our FMOcc with two frame input achieves notable scores of 43.1% RayIoU and 39.8% mIoU on Occ3D-nuScenes validation, 42.6% RayIoU on OpenOcc with 5.4 G inference memory and 330ms inference time.
>
---
#### [new 039] Privacy-preserving Preselection for Face Identification Based on Packing
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于隐私保护下的面部识别任务，旨在解决密文域中人脸检索效率低的问题。提出PFIP方案，通过预选和打包机制提升检索效率。**

- **链接: [http://arxiv.org/pdf/2507.02414v1](http://arxiv.org/pdf/2507.02414v1)**

> **作者:** Rundong Xin; Taotao Wang; Jin Wang; Chonghe Zhao; Jing Wang
>
> **备注:** This paper has been accepted for publication in SecureComm 2025
>
> **摘要:** Face identification systems operating in the ciphertext domain have garnered significant attention due to increasing privacy concerns and the potential recovery of original facial data. However, as the size of ciphertext template libraries grows, the face retrieval process becomes progressively more time-intensive. To address this challenge, we propose a novel and efficient scheme for face retrieval in the ciphertext domain, termed Privacy-Preserving Preselection for Face Identification Based on Packing (PFIP). PFIP incorporates an innovative preselection mechanism to reduce computational overhead and a packing module to enhance the flexibility of biometric systems during the enrollment stage. Extensive experiments conducted on the LFW and CASIA datasets demonstrate that PFIP preserves the accuracy of the original face recognition model, achieving a 100% hit rate while retrieving 1,000 ciphertext face templates within 300 milliseconds. Compared to existing approaches, PFIP achieves a nearly 50x improvement in retrieval efficiency.
>
---
#### [new 040] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决现有技术缺乏解释性和泛化能力的问题。通过构建数据集和设计训练框架，提出AIGI-Holmes模型，提升检测效果与可解释性。**

- **链接: [http://arxiv.org/pdf/2507.02664v1](http://arxiv.org/pdf/2507.02664v1)**

> **作者:** Ziyin Zhou; Yunpeng Luo; Yuanchen Wu; Ke Sun; Jiayi Ji; Ke Yan; Shouhong Ding; Xiaoshuai Sun; Yunsheng Wu; Rongrong Ji
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** The rapid development of AI-generated content (AIGC) technology has led to the misuse of highly realistic AI-generated images (AIGI) in spreading misinformation, posing a threat to public information security. Although existing AIGI detection techniques are generally effective, they face two issues: 1) a lack of human-verifiable explanations, and 2) a lack of generalization in the latest generation technology. To address these issues, we introduce a large-scale and comprehensive dataset, Holmes-Set, which includes the Holmes-SFTSet, an instruction-tuning dataset with explanations on whether images are AI-generated, and the Holmes-DPOSet, a human-aligned preference dataset. Our work introduces an efficient data annotation method called the Multi-Expert Jury, enhancing data generation through structured MLLM explanations and quality control via cross-model evaluation, expert defect filtering, and human preference modification. In addition, we propose Holmes Pipeline, a meticulously designed three-stage training framework comprising visual expert pre-training, supervised fine-tuning, and direct preference optimization. Holmes Pipeline adapts multimodal large language models (MLLMs) for AIGI detection while generating human-verifiable and human-aligned explanations, ultimately yielding our model AIGI-Holmes. During the inference stage, we introduce a collaborative decoding strategy that integrates the model perception of the visual expert with the semantic reasoning of MLLMs, further enhancing the generalization capabilities. Extensive experiments on three benchmarks validate the effectiveness of our AIGI-Holmes.
>
---
#### [new 041] No time to train! Training-Free Reference-Based Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于无监督实例分割任务，旨在解决依赖大量标注数据的问题。通过利用基础模型的语义先验，提出一种无需训练的方法，实现跨域少样本分割。**

- **链接: [http://arxiv.org/pdf/2507.02798v1](http://arxiv.org/pdf/2507.02798v1)**

> **作者:** Miguel Espinosa; Chenhongyi Yang; Linus Ericsson; Steven McDonagh; Elliot J. Crowley
>
> **备注:** Preprint
>
> **摘要:** The performance of image segmentation models has historically been constrained by the high cost of collecting large-scale annotated data. The Segment Anything Model (SAM) alleviates this original problem through a promptable, semantics-agnostic, segmentation paradigm and yet still requires manual visual-prompts or complex domain-dependent prompt-generation rules to process a new image. Towards reducing this new burden, our work investigates the task of object segmentation when provided with, alternatively, only a small set of reference images. Our key insight is to leverage strong semantic priors, as learned by foundation models, to identify corresponding regions between a reference and a target image. We find that correspondences enable automatic generation of instance-level segmentation masks for downstream tasks and instantiate our ideas via a multi-stage, training-free method incorporating (1) memory bank construction; (2) representation aggregation and (3) semantic-aware feature matching. Our experiments show significant improvements on segmentation metrics, leading to state-of-the-art performance on COCO FSOD (36.8% nAP), PASCAL VOC Few-Shot (71.2% nAP50) and outperforming existing training-free approaches on the Cross-Domain FSOD benchmark (22.4% nAP).
>
---
#### [new 042] From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频编辑任务，旨在解决自动剪辑缺乏连贯性的问题。提出HIVE框架，结合多模态理解与场景分割，提升剪辑质量。**

- **链接: [http://arxiv.org/pdf/2507.02790v1](http://arxiv.org/pdf/2507.02790v1)**

> **作者:** Xiangfeng Wang; Xiao Li; Yadong Wei; Xueyu Song; Yang Song; Xiaoqiang Xia; Fangrui Zeng; Zaiyi Chen; Liu Liu; Gu Xu; Tong Xu
>
> **摘要:** The rapid growth of online video content, especially on short video platforms, has created a growing demand for efficient video editing techniques that can condense long-form videos into concise and engaging clips. Existing automatic editing methods predominantly rely on textual cues from ASR transcripts and end-to-end segment selection, often neglecting the rich visual context and leading to incoherent outputs. In this paper, we propose a human-inspired automatic video editing framework (HIVE) that leverages multimodal narrative understanding to address these limitations. Our approach incorporates character extraction, dialogue analysis, and narrative summarization through multimodal large language models, enabling a holistic understanding of the video content. To further enhance coherence, we apply scene-level segmentation and decompose the editing process into three subtasks: highlight detection, opening/ending selection, and pruning of irrelevant content. To facilitate research in this area, we introduce DramaAD, a novel benchmark dataset comprising over 800 short drama episodes and 500 professionally edited advertisement clips. Experimental results demonstrate that our framework consistently outperforms existing baselines across both general and advertisement-oriented editing tasks, significantly narrowing the quality gap between automatic and human-edited videos.
>
---
#### [new 043] Detecting Multiple Diseases in Multiple Crops Using Deep Learning
- **分类: cs.CV; cs.AI; cs.ET**

- **简介: 该论文属于多作物多病害检测任务，旨在解决农业中疾病早期识别问题。通过构建统一数据集并训练深度学习模型，提升检测准确率与覆盖范围。**

- **链接: [http://arxiv.org/pdf/2507.02517v1](http://arxiv.org/pdf/2507.02517v1)**

> **作者:** Vivek Yadav; Anugrah Jain
>
> **摘要:** India, as a predominantly agrarian economy, faces significant challenges in agriculture, including substantial crop losses caused by diseases, pests, and environmental stress. Early detection and accurate identification of diseases across different crops are critical for improving yield and ensuring food security. This paper proposes a deep learning based solution for detecting multiple diseases in multiple crops, aimed to cover India's diverse agricultural landscape. We first create a unified dataset encompassing images of 17 different crops and 34 different diseases from various available repositories. Proposed deep learning model is trained on this dataset and outperforms the state-of-the-art in terms of accuracy and the number of crops, diseases covered. We achieve a significant detection accuracy, i.e., 99 percent for our unified dataset which is 7 percent more when compared to state-of-the-art handling 14 crops and 26 different diseases only. By improving the number of crops and types of diseases that can be detected, proposed solution aims to provide a better product for Indian farmers.
>
---
#### [new 044] USAD: An Unsupervised Data Augmentation Spatio-Temporal Attention Diffusion Network
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人体活动识别任务，解决标签数据少、特征提取不足和轻量设备性能差的问题。提出USAD网络，结合数据增强、多分支时空交互和自适应损失函数，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.02827v1](http://arxiv.org/pdf/2507.02827v1)**

> **作者:** Ying Yu; Hang Xiao; Siyao Li; Jiarui Li; Haotian Tang; Hanyu Liu; Chao Li
>
> **摘要:** The primary objective of human activity recognition (HAR) is to infer ongoing human actions from sensor data, a task that finds broad applications in health monitoring, safety protection, and sports analysis. Despite proliferating research, HAR still faces key challenges, including the scarcity of labeled samples for rare activities, insufficient extraction of high-level features, and suboptimal model performance on lightweight devices. To address these issues, this paper proposes a comprehensive optimization approach centered on multi-attention interaction mechanisms. First, an unsupervised, statistics-guided diffusion model is employed to perform data augmentation, thereby alleviating the problems of labeled data scarcity and severe class imbalance. Second, a multi-branch spatio-temporal interaction network is designed, which captures multi-scale features of sequential data through parallel residual branches with 3*3, 5*5, and 7*7 convolutional kernels. Simultaneously, temporal attention mechanisms are incorporated to identify critical time points, while spatial attention enhances inter-sensor interactions. A cross-branch feature fusion unit is further introduced to improve the overall feature representation capability. Finally, an adaptive multi-loss function fusion strategy is integrated, allowing for dynamic adjustment of loss weights and overall model optimization. Experimental results on three public datasets, WISDM, PAMAP2, and OPPORTUNITY, demonstrate that the proposed unsupervised data augmentation spatio-temporal attention diffusion network (USAD) achieves accuracies of 98.84%, 93.81%, and 80.92% respectively, significantly outperforming existing approaches. Furthermore, practical deployment on embedded devices verifies the efficiency and feasibility of the proposed method.
>
---
#### [new 045] Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决视频扩散模型推理速度慢的问题。提出EasyCache框架，通过运行时自适应缓存减少冗余计算，显著提升效率并保持画质。**

- **链接: [http://arxiv.org/pdf/2507.02860v1](http://arxiv.org/pdf/2507.02860v1)**

> **作者:** Xin Zhou; Dingkang Liang; Kaijin Chen; Tianrui Feng; Xiwu Chen; Hongkai Lin; Yikang Ding; Feiyang Tan; Hengshuang Zhao; Xiang Bai
>
> **备注:** The code is made available at https://github.com/H-EmbodVis/EasyCache. Project page: https://h-embodvis.github.io/EasyCache/
>
> **摘要:** Video generation models have demonstrated remarkable performance, yet their broader adoption remains constrained by slow inference speeds and substantial computational costs, primarily due to the iterative nature of the denoising process. Addressing this bottleneck is essential for democratizing advanced video synthesis technologies and enabling their integration into real-world applications. This work proposes EasyCache, a training-free acceleration framework for video diffusion models. EasyCache introduces a lightweight, runtime-adaptive caching mechanism that dynamically reuses previously computed transformation vectors, avoiding redundant computations during inference. Unlike prior approaches, EasyCache requires no offline profiling, pre-computation, or extensive parameter tuning. We conduct comprehensive studies on various large-scale video generation models, including OpenSora, Wan2.1, and HunyuanVideo. Our method achieves leading acceleration performance, reducing inference time by up to 2.1-3.3$\times$ compared to the original baselines while maintaining high visual fidelity with a significant up to 36% PSNR improvement compared to the previous SOTA method. This improvement makes our EasyCache a efficient and highly accessible solution for high-quality video generation in both research and practical applications. The code is available at https://github.com/H-EmbodVis/EasyCache.
>
---
#### [new 046] Mesh Silksong: Auto-Regressive Mesh Generation as Weaving Silk
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于三维网格生成任务，解决现有方法冗余问题，通过单次访问顶点实现高效压缩与高质量网格生成。**

- **链接: [http://arxiv.org/pdf/2507.02477v1](http://arxiv.org/pdf/2507.02477v1)**

> **作者:** Gaochao Song; Zibo Zhao; Haohan Weng; Jingbo Zeng; Rongfei Jia; Shenghua Gao
>
> **备注:** 9 pages main text, 14 pages appendix, 23 figures
>
> **摘要:** We introduce Mesh Silksong, a compact and efficient mesh representation tailored to generate the polygon mesh in an auto-regressive manner akin to silk weaving. Existing mesh tokenization methods always produce token sequences with repeated vertex tokens, wasting the network capability. Therefore, our approach tokenizes mesh vertices by accessing each mesh vertice only once, reduces the token sequence's redundancy by 50\%, and achieves a state-of-the-art compression rate of approximately 22\%. Furthermore, Mesh Silksong produces polygon meshes with superior geometric properties, including manifold topology, watertight detection, and consistent face normals, which are critical for practical applications. Experimental results demonstrate the effectiveness of our approach, showcasing not only intricate mesh generation but also significantly improved geometric integrity.
>
---
#### [new 047] Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于3D重建任务，解决多视角图像中信息丢失问题。提出Point3R框架，通过显式空间指针记忆实现高效在线3D重建。**

- **链接: [http://arxiv.org/pdf/2507.02863v1](http://arxiv.org/pdf/2507.02863v1)**

> **作者:** Yuqi Wu; Wenzhao Zheng; Jie Zhou; Jiwen Lu
>
> **备注:** Code is available at: https://github.com/YkiWu/Point3R
>
> **摘要:** Dense 3D scene reconstruction from an ordered sequence or unordered image collections is a critical step when bringing research in computer vision into practical scenarios. Following the paradigm introduced by DUSt3R, which unifies an image pair densely into a shared coordinate system, subsequent methods maintain an implicit memory to achieve dense 3D reconstruction from more images. However, such implicit memory is limited in capacity and may suffer from information loss of earlier frames. We propose Point3R, an online framework targeting dense streaming 3D reconstruction. To be specific, we maintain an explicit spatial pointer memory directly associated with the 3D structure of the current scene. Each pointer in this memory is assigned a specific 3D position and aggregates scene information nearby in the global coordinate system into a changing spatial feature. Information extracted from the latest frame interacts explicitly with this pointer memory, enabling dense integration of the current observation into the global coordinate system. We design a 3D hierarchical position embedding to promote this interaction and design a simple yet effective fusion mechanism to ensure that our pointer memory is uniform and efficient. Our method achieves competitive or state-of-the-art performance on various tasks with low training costs. Code is available at: https://github.com/YkiWu/Point3R.
>
---
#### [new 048] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling
- **分类: cs.CV**

- **简介: 该论文属于动态场景建模任务，解决多视角视频合成中大尺度与细粒度运动建模难题。提出LocalDyGS，通过局部空间分解和静态动态特征解耦实现更真实的动态场景重建。**

- **链接: [http://arxiv.org/pdf/2507.02363v1](http://arxiv.org/pdf/2507.02363v1)**

> **作者:** Jiahao Wu; Rui Peng; Jianbo Jiao; Jiayu Yang; Luyang Tang; Kaiqiang Xiong; Jie Liang; Jinbo Yan; Runling Liu; Ronggang Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Due to the complex and highly dynamic motions in the real world, synthesizing dynamic videos from multi-view inputs for arbitrary viewpoints is challenging. Previous works based on neural radiance field or 3D Gaussian splatting are limited to modeling fine-scale motion, greatly restricting their application. In this paper, we introduce LocalDyGS, which consists of two parts to adapt our method to both large-scale and fine-scale motion scenes: 1) We decompose a complex dynamic scene into streamlined local spaces defined by seeds, enabling global modeling by capturing motion within each local space. 2) We decouple static and dynamic features for local space motion modeling. A static feature shared across time steps captures static information, while a dynamic residual field provides time-specific features. These are combined and decoded to generate Temporal Gaussians, modeling motion within each local space. As a result, we propose a novel dynamic scene reconstruction framework to model highly dynamic real-world scenes more realistically. Our method not only demonstrates competitive performance on various fine-scale datasets compared to state-of-the-art (SOTA) methods, but also represents the first attempt to model larger and more complex highly dynamic scenes. Project page: https://wujh2001.github.io/LocalDyGS/.
>
---
#### [new 049] Structure-aware Semantic Discrepancy and Consistency for 3D Medical Image Self-supervised Learning
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像自监督学习任务，旨在解决结构变化下的语义一致性与差异性问题。提出S²DC框架，通过优化传输策略和邻域相似性提升结构感知表示。**

- **链接: [http://arxiv.org/pdf/2507.02581v1](http://arxiv.org/pdf/2507.02581v1)**

> **作者:** Tan Pan; Zhaorui Tan; Kaiyu Guo; Dongli Xu; Weidi Xu; Chen Jiang; Xin Guo; Yuan Qi; Yuan Cheng
>
> **备注:** Accepted by ICCV25
>
> **摘要:** 3D medical image self-supervised learning (mSSL) holds great promise for medical analysis. Effectively supporting broader applications requires considering anatomical structure variations in location, scale, and morphology, which are crucial for capturing meaningful distinctions. However, previous mSSL methods partition images with fixed-size patches, often ignoring the structure variations. In this work, we introduce a novel perspective on 3D medical images with the goal of learning structure-aware representations. We assume that patches within the same structure share the same semantics (semantic consistency) while those from different structures exhibit distinct semantics (semantic discrepancy). Based on this assumption, we propose an mSSL framework named $S^2DC$, achieving Structure-aware Semantic Discrepancy and Consistency in two steps. First, $S^2DC$ enforces distinct representations for different patches to increase semantic discrepancy by leveraging an optimal transport strategy. Second, $S^2DC$ advances semantic consistency at the structural level based on neighborhood similarity distribution. By bridging patch-level and structure-level representations, $S^2DC$ achieves structure-aware representations. Thoroughly evaluated across 10 datasets, 4 tasks, and 3 modalities, our proposed method consistently outperforms the state-of-the-art methods in mSSL.
>
---
#### [new 050] UVLM: Benchmarking Video Language Model for Underwater World Understanding
- **分类: cs.CV**

- **简介: 该论文提出UVLM基准，解决水下环境理解问题。通过构建多样化数据集和任务，提升视频语言模型在水下场景的性能。**

- **链接: [http://arxiv.org/pdf/2507.02373v1](http://arxiv.org/pdf/2507.02373v1)**

> **作者:** Xizhe Xue; Yang Zhou; Dawei Yan; Ying Li; Haokui Zhang; Rong Xiao
>
> **备注:** 13 pages, 4 figures, 3 tables
>
> **摘要:** Recently, the remarkable success of large language models (LLMs) has achieved a profound impact on the field of artificial intelligence. Numerous advanced works based on LLMs have been proposed and applied in various scenarios. Among them, video language models (VidLMs) are particularly widely used. However, existing works primarily focus on terrestrial scenarios, overlooking the highly demanding application needs of underwater observation. To overcome this gap, we introduce UVLM, an under water observation benchmark which is build through a collaborative approach combining human expertise and AI models. To ensure data quality, we have conducted in-depth considerations from multiple perspectives. First, to address the unique challenges of underwater environments, we selected videos that represent typical underwater challenges including light variations, water turbidity, and diverse viewing angles to construct the dataset. Second, to ensure data diversity, the dataset covers a wide range of frame rates, resolutions, 419 classes of marine animals, and various static plants and terrains. Next, for task diversity, we adopted a structured design where observation targets are categorized into two major classes: biological and environmental. Each category includes content observation and change/action observation, totaling 20 distinct task types. Finally, we designed several challenging evaluation metrics to enable quantitative comparison and analysis of different methods. Experiments on two representative VidLMs demonstrate that fine-tuning VidLMs on UVLM significantly improves underwater world understanding while also showing potential for slight improvements on existing in-air VidLM benchmarks, such as VideoMME and Perception text. The dataset and prompt engineering will be released publicly.
>
---
#### [new 051] MedFormer: Hierarchical Medical Vision Transformer with Content-Aware Dual Sparse Selection Attention
- **分类: cs.CV**

- **简介: 该论文属于医学图像识别任务，旨在解决模型泛化性差和计算效率低的问题。提出MedFormer，采用金字塔结构和内容感知的双稀疏注意力机制，提升性能与效率。**

- **链接: [http://arxiv.org/pdf/2507.02488v1](http://arxiv.org/pdf/2507.02488v1)**

> **作者:** Zunhui Xia; Hongxing Li; Libin Lan
>
> **备注:** 13 pages, 9 figures, 9 tables
>
> **摘要:** Medical image recognition serves as a key way to aid in clinical diagnosis, enabling more accurate and timely identification of diseases and abnormalities. Vision transformer-based approaches have proven effective in handling various medical recognition tasks. However, these methods encounter two primary challenges. First, they are often task-specific and architecture-tailored, limiting their general applicability. Second, they usually either adopt full attention to model long-range dependencies, resulting in high computational costs, or rely on handcrafted sparse attention, potentially leading to suboptimal performance. To tackle these issues, we present MedFormer, an efficient medical vision transformer with two key ideas. First, it employs a pyramid scaling structure as a versatile backbone for various medical image recognition tasks, including image classification and dense prediction tasks such as semantic segmentation and lesion detection. This structure facilitates hierarchical feature representation while reducing the computation load of feature maps, highly beneficial for boosting performance. Second, it introduces a novel Dual Sparse Selection Attention (DSSA) with content awareness to improve computational efficiency and robustness against noise while maintaining high performance. As the core building technique of MedFormer, DSSA is explicitly designed to attend to the most relevant content. In addition, a detailed theoretical analysis has been conducted, demonstrating that MedFormer has superior generality and efficiency in comparison to existing medical vision transformers. Extensive experiments on a variety of imaging modality datasets consistently show that MedFormer is highly effective in enhancing performance across all three above-mentioned medical image recognition tasks. The code is available at https://github.com/XiaZunhui/MedFormer.
>
---
#### [new 052] Linear Attention with Global Context: A Multipole Attention Mechanism for Vision and Physics
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉与物理模拟任务，旨在解决Transformer模型在高分辨率输入下的高计算复杂度问题。提出MANO机制，实现线性时间与内存复杂度的注意力计算。**

- **链接: [http://arxiv.org/pdf/2507.02748v1](http://arxiv.org/pdf/2507.02748v1)**

> **作者:** Alex Colagrande; Paul Caillon; Eva Feillet; Alexandre Allauzen
>
> **备注:** Accepted at ECLR Workshop at ICCV 2025
>
> **摘要:** Transformers have become the de facto standard for a wide range of tasks, from image classification to physics simulations. Despite their impressive performance, the quadratic complexity of standard Transformers in both memory and time with respect to the input length makes them impractical for processing high-resolution inputs. Therefore, several variants have been proposed, the most successful relying on patchification, downsampling, or coarsening techniques, often at the cost of losing the finest-scale details. In this work, we take a different approach. Inspired by state-of-the-art techniques in $n$-body numerical simulations, we cast attention as an interaction problem between grid points. We introduce the Multipole Attention Neural Operator (MANO), which computes attention in a distance-based multiscale fashion. MANO maintains, in each attention head, a global receptive field and achieves linear time and memory complexity with respect to the number of grid points. Empirical results on image classification and Darcy flows demonstrate that MANO rivals state-of-the-art models such as ViT and Swin Transformer, while reducing runtime and peak memory usage by orders of magnitude. We open source our code for reproducibility at https://github.com/AlexColagrande/MANO.
>
---
#### [new 053] RichControl: Structure- and Appearance-Rich Training-Free Spatial Control for Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决条件图像与生成图像结构不一致的问题。提出一种无需训练的特征注入框架，提升结构和外观控制效果。**

- **链接: [http://arxiv.org/pdf/2507.02792v1](http://arxiv.org/pdf/2507.02792v1)**

> **作者:** Liheng Zhang; Lexi Pang; Hang Ye; Xiaoxuan Ma; Yizhou Wang
>
> **摘要:** Text-to-image (T2I) diffusion models have shown remarkable success in generating high-quality images from text prompts. Recent efforts extend these models to incorporate conditional images (e.g., depth or pose maps) for fine-grained spatial control. Among them, feature injection methods have emerged as a training-free alternative to traditional fine-tuning approaches. However, they often suffer from structural misalignment, condition leakage, and visual artifacts, especially when the condition image diverges significantly from natural RGB distributions. By revisiting existing methods, we identify a core limitation: the synchronous injection of condition features fails to account for the trade-off between domain alignment and structural preservation during denoising. Inspired by this observation, we propose a flexible feature injection framework that decouples the injection timestep from the denoising process. At its core is a structure-rich injection module, which enables the model to better adapt to the evolving interplay between alignment and structure preservation throughout the diffusion steps, resulting in more faithful structural generation. In addition, we introduce appearance-rich prompting and a restart refinement strategy to further enhance appearance control and visual quality. Together, these designs enable training-free generation that is both structure-rich and appearance-rich. Extensive experiments show that our approach achieves state-of-the-art performance across diverse zero-shot conditioning scenarios.
>
---
#### [new 054] AvatarMakeup: Realistic Makeup Transfer for 3D Animatable Head Avatars
- **分类: cs.CV**

- **简介: 该论文属于3D虚拟形象个性化定制任务，旨在解决3D头像妆容转移中的真实感、一致性与细节控制问题。提出AvatarMakeup方法，结合扩散模型与全局UV优化，实现高质量妆容迁移。**

- **链接: [http://arxiv.org/pdf/2507.02419v1](http://arxiv.org/pdf/2507.02419v1)**

> **作者:** Yiming Zhong; Xiaolin Zhang; Ligang Liu; Yao Zhao; Yunchao Wei
>
> **摘要:** Similar to facial beautification in real life, 3D virtual avatars require personalized customization to enhance their visual appeal, yet this area remains insufficiently explored. Although current 3D Gaussian editing methods can be adapted for facial makeup purposes, these methods fail to meet the fundamental requirements for achieving realistic makeup effects: 1) ensuring a consistent appearance during drivable expressions, 2) preserving the identity throughout the makeup process, and 3) enabling precise control over fine details. To address these, we propose a specialized 3D makeup method named AvatarMakeup, leveraging a pretrained diffusion model to transfer makeup patterns from a single reference photo of any individual. We adopt a coarse-to-fine idea to first maintain the consistent appearance and identity, and then to refine the details. In particular, the diffusion model is employed to generate makeup images as supervision. Due to the uncertainties in diffusion process, the generated images are inconsistent across different viewpoints and expressions. Therefore, we propose a Coherent Duplication method to coarsely apply makeup to the target while ensuring consistency across dynamic and multiview effects. Coherent Duplication optimizes a global UV map by recoding the averaged facial attributes among the generated makeup images. By querying the global UV map, it easily synthesizes coherent makeup guidance from arbitrary views and expressions to optimize the target avatar. Given the coarse makeup avatar, we further enhance the makeup by incorporating a Refinement Module into the diffusion model to achieve high makeup quality. Experiments demonstrate that AvatarMakeup achieves state-of-the-art makeup transfer quality and consistency throughout animation.
>
---
#### [new 055] SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出SciGA-145k数据集，用于支持图形摘要的设计与推荐，解决可视化科学传播难题。**

- **链接: [http://arxiv.org/pdf/2507.02212v1](http://arxiv.org/pdf/2507.02212v1)**

> **作者:** Takuro Kawada; Shunsuke Kitada; Sota Nemoto; Hitoshi Iyatomi
>
> **备注:** 21 pages, 15 figures, 4 tables. Project Page: https://iyatomilab.github.io/SciGA/
>
> **摘要:** Graphical Abstracts (GAs) play a crucial role in visually conveying the key findings of scientific papers. While recent research has increasingly incorporated visual materials such as Figure 1 as de facto GAs, their potential to enhance scientific communication remains largely unexplored. Moreover, designing effective GAs requires advanced visualization skills, creating a barrier to their widespread adoption. To tackle these challenges, we introduce SciGA-145k, a large-scale dataset comprising approximately 145,000 scientific papers and 1.14 million figures, explicitly designed for supporting GA selection and recommendation as well as facilitating research in automated GA generation. As a preliminary step toward GA design support, we define two tasks: 1) Intra-GA recommendation, which identifies figures within a given paper that are well-suited to serve as GAs, and 2) Inter-GA recommendation, which retrieves GAs from other papers to inspire the creation of new GAs. We provide reasonable baseline models for these tasks. Furthermore, we propose Confidence Adjusted top-1 ground truth Ratio (CAR), a novel recommendation metric that offers a fine-grained analysis of model behavior. CAR addresses limitations in traditional ranking-based metrics by considering cases where multiple figures within a paper, beyond the explicitly labeled GA, may also serve as GAs. By unifying these tasks and metrics, our SciGA-145k establishes a foundation for advancing visual scientific communication while contributing to the development of AI for Science.
>
---
#### [new 056] DexVLG: Dexterous Vision-Language-Grasp Model at Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DexVLG模型，解决机器人基于语言指令进行精细抓取的任务。通过大规模数据训练，实现高精度的物体部分抓取。**

- **链接: [http://arxiv.org/pdf/2507.02747v1](http://arxiv.org/pdf/2507.02747v1)**

> **作者:** Jiawei He; Danshi Li; Xinqiang Yu; Zekun Qi; Wenyao Zhang; Jiayi Chen; Zhaoxiang Zhang; Zhizheng Zhang; Li Yi; He Wang
>
> **摘要:** As large models gain traction, vision-language-action (VLA) systems are enabling robots to tackle increasingly complex tasks. However, limited by the difficulty of data collection, progress has mainly focused on controlling simple gripper end-effectors. There is little research on functional grasping with large models for human-like dexterous hands. In this paper, we introduce DexVLG, a large Vision-Language-Grasp model for Dexterous grasp pose prediction aligned with language instructions using single-view RGBD input. To accomplish this, we generate a dataset of 170 million dexterous grasp poses mapped to semantic parts across 174,000 objects in simulation, paired with detailed part-level captions. This large-scale dataset, named DexGraspNet 3.0, is used to train a VLM and flow-matching-based pose head capable of producing instruction-aligned grasp poses for tabletop objects. To assess DexVLG's performance, we create benchmarks in physics-based simulations and conduct real-world experiments. Extensive testing demonstrates DexVLG's strong zero-shot generalization capabilities-achieving over 76% zero-shot execution success rate and state-of-the-art part-grasp accuracy in simulation-and successful part-aligned grasps on physical objects in real-world scenarios.
>
---
#### [new 057] Weakly-supervised Contrastive Learning with Quantity Prompts for Moving Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决标注成本高、目标小且背景弱的问题。提出一种弱监督对比学习方法，仅需目标数量提示即可提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.02454v1](http://arxiv.org/pdf/2507.02454v1)**

> **作者:** Weiwei Duan; Luping Ji; Shengjia Chen; Sicheng Zhu; Jianghong Huang; Mao Ye
>
> **摘要:** Different from general object detection, moving infrared small target detection faces huge challenges due to tiny target size and weak background contrast.Currently, most existing methods are fully-supervised, heavily relying on a large number of manual target-wise annotations. However, manually annotating video sequences is often expensive and time-consuming, especially for low-quality infrared frame images. Inspired by general object detection, non-fully supervised strategies ($e.g.$, weakly supervised) are believed to be potential in reducing annotation requirements. To break through traditional fully-supervised frameworks, as the first exploration work, this paper proposes a new weakly-supervised contrastive learning (WeCoL) scheme, only requires simple target quantity prompts during model training.Specifically, in our scheme, based on the pretrained segment anything model (SAM), a potential target mining strategy is designed to integrate target activation maps and multi-frame energy accumulation.Besides, contrastive learning is adopted to further improve the reliability of pseudo-labels, by calculating the similarity between positive and negative samples in feature subspace.Moreover, we propose a long-short term motion-aware learning scheme to simultaneously model the local motion patterns and global motion trajectory of small targets.The extensive experiments on two public datasets (DAUB and ITSDT-15K) verify that our weakly-supervised scheme could often outperform early fully-supervised methods. Even, its performance could reach over 90\% of state-of-the-art (SOTA) fully-supervised ones.
>
---
#### [new 058] AuroraLong: Bringing RNNs Back to Efficient Open-Ended Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频处理的高计算和内存成本问题。通过引入线性RNN模型替代Transformer，提升效率与吞吐量。**

- **链接: [http://arxiv.org/pdf/2507.02591v1](http://arxiv.org/pdf/2507.02591v1)**

> **作者:** Weili Xu; Enxin Song; Wenhao Chai; Xuexiang Wen; Tian Ye; Gaoang Wang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** The challenge of long video understanding lies in its high computational complexity and prohibitive memory cost, since the memory and computation required by transformer-based LLMs scale quadratically with input sequence length. We propose AuroraLong to address this challenge by replacing the LLM component in MLLMs with a linear RNN language model that handles input sequence of arbitrary length with constant-size hidden states. To further increase throughput and efficiency, we combine visual token merge with linear RNN models by reordering the visual tokens by their sizes in ascending order. Despite having only 2B parameters and being trained exclusively on public data, AuroraLong achieves performance comparable to Transformer-based models of similar size trained on private datasets across multiple video benchmarks. This demonstrates the potential of efficient, linear RNNs to democratize long video understanding by lowering its computational entry barrier. To our best knowledge, we are the first to use a linear RNN based LLM backbone in a LLaVA-like model for open-ended video understanding.
>
---
#### [new 059] Two-Steps Neural Networks for an Automated Cerebrovascular Landmark Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决脑血管分叉点自动检测问题。通过两阶段神经网络提高检测准确率。**

- **链接: [http://arxiv.org/pdf/2507.02349v1](http://arxiv.org/pdf/2507.02349v1)**

> **作者:** Rafic Nader; Vincent L'Allinec; Romain Bourcier; Florent Autrusseau
>
> **摘要:** Intracranial aneurysms (ICA) commonly occur in specific segments of the Circle of Willis (CoW), primarily, onto thirteen major arterial bifurcations. An accurate detection of these critical landmarks is necessary for a prompt and efficient diagnosis. We introduce a fully automated landmark detection approach for CoW bifurcations using a two-step neural networks process. Initially, an object detection network identifies regions of interest (ROIs) proximal to the landmark locations. Subsequently, a modified U-Net with deep supervision is exploited to accurately locate the bifurcations. This two-step method reduces various problems, such as the missed detections caused by two landmarks being close to each other and having similar visual characteristics, especially when processing the complete MRA Time-of-Flight (TOF). Additionally, it accounts for the anatomical variability of the CoW, which affects the number of detectable landmarks per scan. We assessed the effectiveness of our approach using two cerebral MRA datasets: our In-House dataset which had varying numbers of landmarks, and a public dataset with standardized landmark configuration. Our experimental results demonstrate that our method achieves the highest level of performance on a bifurcation detection task.
>
---
#### [new 060] From Pixels to Damage Severity: Estimating Earthquake Impacts Using Semantic Segmentation of Social Media Images
- **分类: cs.CV; cs.SI**

- **简介: 该论文属于图像分割任务，旨在解决地震后社交媒体图像中损伤程度评估的主观性问题。通过构建分割数据集并优化模型，实现更客观的损伤分析。**

- **链接: [http://arxiv.org/pdf/2507.02781v1](http://arxiv.org/pdf/2507.02781v1)**

> **作者:** Danrong Zhang; Huili Huang; N. Simrill Smith; Nimisha Roy; J. David Frost
>
> **摘要:** In the aftermath of earthquakes, social media images have become a crucial resource for disaster reconnaissance, providing immediate insights into the extent of damage. Traditional approaches to damage severity assessment in post-earthquake social media images often rely on classification methods, which are inherently subjective and incapable of accounting for the varying extents of damage within an image. Addressing these limitations, this study proposes a novel approach by framing damage severity assessment as a semantic segmentation problem, aiming for a more objective analysis of damage in earthquake-affected areas. The methodology involves the construction of a segmented damage severity dataset, categorizing damage into three degrees: undamaged structures, damaged structures, and debris. Utilizing this dataset, the study fine-tunes a SegFormer model to generate damage severity segmentations for post-earthquake social media images. Furthermore, a new damage severity scoring system is introduced, quantifying damage by considering the varying degrees of damage across different areas within images, adjusted for depth estimation. The application of this approach allows for the quantification of damage severity in social media images in a more objective and comprehensive manner. By providing a nuanced understanding of damage, this study enhances the ability to offer precise guidance to disaster reconnaissance teams, facilitating more effective and targeted response efforts in the aftermath of earthquakes.
>
---
#### [new 061] SIU3R: Simultaneous Scene Understanding and 3D Reconstruction Beyond Feature Alignment
- **分类: cs.CV**

- **简介: 该论文属于场景理解与3D重建任务，解决传统方法依赖2D-3D对齐导致的语义信息损失问题，提出无需对齐的SIU3R框架。**

- **链接: [http://arxiv.org/pdf/2507.02705v1](http://arxiv.org/pdf/2507.02705v1)**

> **作者:** Qi Xu; Dongxu Wei; Lingzhe Zhao; Wenpu Li; Zhangchi Huang; Shunping Ji; Peidong Liu
>
> **摘要:** Simultaneous understanding and 3D reconstruction plays an important role in developing end-to-end embodied intelligent systems. To achieve this, recent approaches resort to 2D-to-3D feature alignment paradigm, which leads to limited 3D understanding capability and potential semantic information loss. In light of this, we propose SIU3R, the first alignment-free framework for generalizable simultaneous understanding and 3D reconstruction from unposed images. Specifically, SIU3R bridges reconstruction and understanding tasks via pixel-aligned 3D representation, and unifies multiple understanding tasks into a set of unified learnable queries, enabling native 3D understanding without the need of alignment with 2D models. To encourage collaboration between the two tasks with shared representation, we further conduct in-depth analyses of their mutual benefits, and propose two lightweight modules to facilitate their interaction. Extensive experiments demonstrate that our method achieves state-of-the-art performance not only on the individual tasks of 3D reconstruction and understanding, but also on the task of simultaneous understanding and 3D reconstruction, highlighting the advantages of our alignment-free framework and the effectiveness of the mutual benefit designs.
>
---
#### [new 062] PLOT: Pseudo-Labeling via Video Object Tracking for Scalable Monocular 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于单目3D目标检测任务，旨在解决数据稀缺和2D到3D歧义问题。提出一种基于视频对象跟踪的伪标签框架，提升检测鲁棒性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2507.02393v1](http://arxiv.org/pdf/2507.02393v1)**

> **作者:** Seokyeong Lee; Sithu Aung; Junyong Choi; Seungryong Kim; Ig-Jae Kim; Junghyun Cho
>
> **备注:** 18 pages, 16 figures
>
> **摘要:** Monocular 3D object detection (M3OD) has long faced challenges due to data scarcity caused by high annotation costs and inherent 2D-to-3D ambiguity. Although various weakly supervised methods and pseudo-labeling methods have been proposed to address these issues, they are mostly limited by domain-specific learning or rely solely on shape information from a single observation. In this paper, we propose a novel pseudo-labeling framework that uses only video data and is more robust to occlusion, without requiring a multi-view setup, additional sensors, camera poses, or domain-specific training. Specifically, we explore a technique for aggregating the pseudo-LiDARs of both static and dynamic objects across temporally adjacent frames using object point tracking, enabling 3D attribute extraction in scenarios where 3D data acquisition is infeasible. Extensive experiments demonstrate that our method ensures reliable accuracy and strong scalability, making it a practical and effective solution for M3OD.
>
---
#### [new 063] Red grape detection with accelerated artificial neural networks in the FPGA's programmable logic
- **分类: cs.CV; cs.AI; cs.DC; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决机器人在移动中检测物体速度慢的问题。通过在FPGA中部署加速的神经网络模型，提升检测效率。**

- **链接: [http://arxiv.org/pdf/2507.02443v1](http://arxiv.org/pdf/2507.02443v1)**

> **作者:** Sandro Costa Magalhães; Marco Almeida; Filipe Neves dos Santos; António Paulo Moreira; Jorge Dias
>
> **备注:** Submitted to ROBOT'2025
>
> **摘要:** Robots usually slow down for canning to detect objects while moving. Additionally, the robot's camera is configured with a low framerate to track the velocity of the detection algorithms. This would be constrained while executing tasks and exploring, making robots increase the task execution time. AMD has developed the Vitis-AI framework to deploy detection algorithms into FPGAs. However, this tool does not fully use the FPGAs' PL. In this work, we use the FINN architecture to deploy three ANNs, MobileNet v1 with 4-bit quantisation, CNV with 2-bit quantisation, and CNV with 1-bit quantisation (BNN), inside an FPGA's PL. The models were trained on the RG2C dataset. This is a self-acquired dataset released in open access. MobileNet v1 performed better, reaching a success rate of 98 % and an inference speed of 6611 FPS. In this work, we proved that we can use FPGAs to speed up ANNs and make them suitable for attention mechanisms.
>
---
#### [new 064] Large Language Models for Crash Detection in Video: A Survey of Methods, Datasets, and Challenges
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，旨在解决交通事故检测问题。通过综述LLM和VLM方法，分析数据集、模型架构及挑战，为未来研究提供基础。**

- **链接: [http://arxiv.org/pdf/2507.02074v1](http://arxiv.org/pdf/2507.02074v1)**

> **作者:** Sanjeda Akter; Ibne Farabi Shihab; Anuj Sharma
>
> **摘要:** Crash detection from video feeds is a critical problem in intelligent transportation systems. Recent developments in large language models (LLMs) and vision-language models (VLMs) have transformed how we process, reason about, and summarize multimodal information. This paper surveys recent methods leveraging LLMs for crash detection from video data. We present a structured taxonomy of fusion strategies, summarize key datasets, analyze model architectures, compare performance benchmarks, and discuss ongoing challenges and opportunities. Our review provides a foundation for future research in this fast-growing intersection of video understanding and foundation models.
>
---
#### [new 065] Holistic Tokenizer for Autoregressive Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自回归图像生成任务，旨在解决传统分步生成模型难以捕捉全局关系的问题。提出Hita tokenizer，通过整体到局部的结构提升生成效果。**

- **链接: [http://arxiv.org/pdf/2507.02358v1](http://arxiv.org/pdf/2507.02358v1)**

> **作者:** Anlin Zheng; Haochen Wang; Yucheng Zhao; Weipeng Deng; Tiancai Wang; Xiangyu Zhang; Xiaojuan Qi
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** The vanilla autoregressive image generation model generates visual tokens in a step-by-step fashion, which limits the ability to capture holistic relationships among token sequences. Moreover, most visual tokenizers map local image patches into latent tokens, leading to limited global information. To address this, we introduce \textit{Hita}, a novel image tokenizer for autoregressive (AR) image generation. It introduces a holistic-to-local tokenization scheme with learnable holistic queries and local patch tokens. Besides, Hita incorporates two key strategies for improved alignment with the AR generation process: 1) it arranges a sequential structure with holistic tokens at the beginning followed by patch-level tokens while using causal attention to maintain awareness of previous tokens; and 2) before feeding the de-quantized tokens into the decoder, Hita adopts a lightweight fusion module to control information flow to prioritize holistic tokens. Extensive experiments show that Hita accelerates the training speed of AR generators and outperforms those trained with vanilla tokenizers, achieving \textbf{2.59 FID} and \textbf{281.9 IS} on the ImageNet benchmark. A detailed analysis of the holistic representation highlights its ability to capture global image properties such as textures, materials, and shapes. Additionally, Hita also demonstrates effectiveness in zero-shot style transfer and image in-painting. The code is available at \href{https://github.com/CVMI-Lab/Hita}{https://github.com/CVMI-Lab/Hita}
>
---
#### [new 066] MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details
- **分类: cs.CV**

- **简介: 该论文属于单目几何估计任务，解决从单张图像恢复精确度量尺度和细节的3D点云问题。通过改进模型并引入数据优化方法提升重建精度。**

- **链接: [http://arxiv.org/pdf/2507.02546v1](http://arxiv.org/pdf/2507.02546v1)**

> **作者:** Ruicheng Wang; Sicheng Xu; Yue Dong; Yu Deng; Jianfeng Xiang; Zelong Lv; Guangzhong Sun; Xin Tong; Jiaolong Yang
>
> **备注:** Project page: https://wangrc.site/MoGe2Page/
>
> **摘要:** We propose MoGe-2, an advanced open-domain geometry estimation model that recovers a metric scale 3D point map of a scene from a single image. Our method builds upon the recent monocular geometry estimation approach, MoGe, which predicts affine-invariant point maps with unknown scales. We explore effective strategies to extend MoGe for metric geometry prediction without compromising the relative geometry accuracy provided by the affine-invariant point representation. Additionally, we discover that noise and errors in real data diminish fine-grained detail in the predicted geometry. We address this by developing a unified data refinement approach that filters and completes real data from different sources using sharp synthetic labels, significantly enhancing the granularity of the reconstructed geometry while maintaining the overall accuracy. We train our model on a large corpus of mixed datasets and conducted comprehensive evaluations, demonstrating its superior performance in achieving accurate relative geometry, precise metric scale, and fine-grained detail recovery -- capabilities that no previous methods have simultaneously achieved.
>
---
#### [new 067] RefTok: Reference-Based Tokenization for Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频中时间冗余和依赖关系处理的问题。提出RefTok方法，通过参考帧进行编码解码，提升视频生成质量。**

- **链接: [http://arxiv.org/pdf/2507.02862v1](http://arxiv.org/pdf/2507.02862v1)**

> **作者:** Xiang Fan; Xiaohang Sun; Kushan Thakkar; Zhu Liu; Vimal Bhat; Ranjay Krishna; Xiang Hao
>
> **摘要:** Effectively handling temporal redundancy remains a key challenge in learning video models. Prevailing approaches often treat each set of frames independently, failing to effectively capture the temporal dependencies and redundancies inherent in videos. To address this limitation, we introduce RefTok, a novel reference-based tokenization method capable of capturing complex temporal dynamics and contextual information. Our method encodes and decodes sets of frames conditioned on an unquantized reference frame. When decoded, RefTok preserves the continuity of motion and the appearance of objects across frames. For example, RefTok retains facial details despite head motion, reconstructs text correctly, preserves small patterns, and maintains the legibility of handwriting from the context. Across 4 video datasets (K600, UCF-101, BAIR Robot Pushing, and DAVIS), RefTok significantly outperforms current state-of-the-art tokenizers (Cosmos and MAGVIT) and improves all evaluated metrics (PSNR, SSIM, LPIPS) by an average of 36.7% at the same or higher compression ratios. When a video generation model is trained using RefTok's latents on the BAIR Robot Pushing task, the generations not only outperform MAGVIT-B but the larger MAGVIT-L, which has 4x more parameters, across all generation metrics by an average of 27.9%.
>
---
#### [new 068] DreamComposer++: Empowering Diffusion Models with Multi-View Conditions for 3D Content Generation
- **分类: cs.CV**

- **简介: 该论文属于3D内容生成任务，旨在解决单视图生成不可控的问题。通过引入多视图条件，提升扩散模型生成可控新视图的能力。**

- **链接: [http://arxiv.org/pdf/2507.02299v1](http://arxiv.org/pdf/2507.02299v1)**

> **作者:** Yunhan Yang; Shuo Chen; Yukun Huang; Xiaoyang Wu; Yuan-Chen Guo; Edmund Y. Lam; Hengshuang Zhao; Tong He; Xihui Liu
>
> **备注:** Accepted by TPAMI, extension of CVPR 2024 paper DreamComposer
>
> **摘要:** Recent advancements in leveraging pre-trained 2D diffusion models achieve the generation of high-quality novel views from a single in-the-wild image. However, existing works face challenges in producing controllable novel views due to the lack of information from multiple views. In this paper, we present DreamComposer++, a flexible and scalable framework designed to improve current view-aware diffusion models by incorporating multi-view conditions. Specifically, DreamComposer++ utilizes a view-aware 3D lifting module to extract 3D representations of an object from various views. These representations are then aggregated and rendered into the latent features of target view through the multi-view feature fusion module. Finally, the obtained features of target view are integrated into pre-trained image or video diffusion models for novel view synthesis. Experimental results demonstrate that DreamComposer++ seamlessly integrates with cutting-edge view-aware diffusion models and enhances their abilities to generate controllable novel views from multi-view conditions. This advancement facilitates controllable 3D object reconstruction and enables a wide range of applications.
>
---
#### [new 069] Bootstrapping Grounded Chain-of-Thought in Multimodal LLMs for Data-Efficient Model Adaptation
- **分类: cs.CV**

- **简介: 该论文属于多模态模型适应任务，旨在解决MLLM在有限数据下难以适应专业视觉任务的问题。通过引入带边界框的CoT数据提升推理准确性。**

- **链接: [http://arxiv.org/pdf/2507.02859v1](http://arxiv.org/pdf/2507.02859v1)**

> **作者:** Jiaer Xia; Bingkui Tong; Yuhang Zang; Rui Shao; Kaiyang Zhou
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in interpreting images using natural language. However, without using large-scale datasets for retraining, these models are difficult to adapt to specialized vision tasks, e.g., chart understanding. This problem is caused by a mismatch between pre-training and downstream datasets: pre-training datasets primarily concentrate on scenes and objects but contain limited information about specialized, non-object images, such as charts and tables. In this paper, we share an interesting finding that training an MLLM with Chain-of-Thought (CoT) reasoning data can facilitate model adaptation in specialized vision tasks, especially under data-limited regimes. However, we identify a critical issue within CoT data distilled from pre-trained MLLMs, i.e., the data often contains multiple factual errors in the reasoning steps. To address the problem, we propose Grounded Chain-of-Thought (GCoT), a simple bootstrapping-based approach that aims to inject grounding information (i.e., bounding boxes) into CoT data, essentially making the reasoning steps more faithful to input images. We evaluate our approach on five specialized vision tasks, which cover a variety of visual formats including charts, tables, receipts, and reports. The results demonstrate that under data-limited regimes our approach significantly improves upon fine-tuning and distillation.
>
---
#### [new 070] Prompt learning with bounding box constraints for medical image segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决标注数据获取困难的问题。通过结合基础模型与弱监督分割，仅使用边界框注释自动生成提示，提升分割效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.02743v1](http://arxiv.org/pdf/2507.02743v1)**

> **作者:** Mélanie Gaillochet; Mehrdad Noori; Sahar Dastani; Christian Desrosiers; Hervé Lombaert
>
> **备注:** Accepted to IEEE Transactions on Biomedical Engineering (TMBE), 14 pages
>
> **摘要:** Pixel-wise annotations are notoriously labourious and costly to obtain in the medical domain. To mitigate this burden, weakly supervised approaches based on bounding box annotations-much easier to acquire-offer a practical alternative. Vision foundation models have recently shown noteworthy segmentation performance when provided with prompts such as points or bounding boxes. Prompt learning exploits these models by adapting them to downstream tasks and automating segmentation, thereby reducing user intervention. However, existing prompt learning approaches depend on fully annotated segmentation masks. This paper proposes a novel framework that combines the representational power of foundation models with the annotation efficiency of weakly supervised segmentation. More specifically, our approach automates prompt generation for foundation models using only bounding box annotations. Our proposed optimization scheme integrates multiple constraints derived from box annotations with pseudo-labels generated by the prompted foundation model. Extensive experiments across multimodal datasets reveal that our weakly supervised method achieves an average Dice score of 84.90% in a limited data setting, outperforming existing fully-supervised and weakly-supervised approaches. The code is available at https://github.com/Minimel/box-prompt-learning-VFM.git
>
---
#### [new 071] CanonSwap: High-Fidelity and Consistent Video Face Swapping via Canonical Space Modulation
- **分类: cs.CV**

- **简介: 该论文属于视频人脸交换任务，旨在解决身份迁移与动态属性保持不一致的问题。提出CanonSwap框架，通过解耦运动与外观信息，实现高质量且一致的视频换脸。**

- **链接: [http://arxiv.org/pdf/2507.02691v1](http://arxiv.org/pdf/2507.02691v1)**

> **作者:** Xiangyang Luo; Ye Zhu; Yunfei Liu; Lijian Lin; Cong Wan; Zijian Cai; Shao-Lun Huang; Yu Li
>
> **备注:** ICCV Accepted
>
> **摘要:** Video face swapping aims to address two primary challenges: effectively transferring the source identity to the target video and accurately preserving the dynamic attributes of the target face, such as head poses, facial expressions, lip-sync, \etc. Existing methods mainly focus on achieving high-quality identity transfer but often fall short in maintaining the dynamic attributes of the target face, leading to inconsistent results. We attribute this issue to the inherent coupling of facial appearance and motion in videos. To address this, we propose CanonSwap, a novel video face-swapping framework that decouples motion information from appearance information. Specifically, CanonSwap first eliminates motion-related information, enabling identity modification within a unified canonical space. Subsequently, the swapped feature is reintegrated into the original video space, ensuring the preservation of the target face's dynamic attributes. To further achieve precise identity transfer with minimal artifacts and enhanced realism, we design a Partial Identity Modulation module that adaptively integrates source identity features using a spatial mask to restrict modifications to facial regions. Additionally, we introduce several fine-grained synchronization metrics to comprehensively evaluate the performance of video face swapping methods. Extensive experiments demonstrate that our method significantly outperforms existing approaches in terms of visual quality, temporal consistency, and identity preservation. Our project page are publicly available at https://luoxyhappy.github.io/CanonSwap/.
>
---
#### [new 072] Partial Weakly-Supervised Oriented Object Detection
- **分类: cs.CV**

- **简介: 该论文属于定向目标检测任务，解决标注成本高的问题。提出PWOOD框架，利用部分弱标注数据提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.02751v1](http://arxiv.org/pdf/2507.02751v1)**

> **作者:** Mingxin Liu; Peiyuan Zhang; Yuan Liu; Wei Zhang; Yue Zhou; Ning Liao; Ziyang Gong; Junwei Luo; Zhirui Wang; Yi Yu; Xue Yang
>
> **备注:** 10 pages, 5 figures, 4 tables, source code: https://github.com/VisionXLab/PWOOD
>
> **摘要:** The growing demand for oriented object detection (OOD) across various domains has driven significant research in this area. However, the high cost of dataset annotation remains a major concern. Current mainstream OOD algorithms can be mainly categorized into three types: (1) fully supervised methods using complete oriented bounding box (OBB) annotations, (2) semi-supervised methods using partial OBB annotations, and (3) weakly supervised methods using weak annotations such as horizontal boxes or points. However, these algorithms inevitably increase the cost of models in terms of annotation speed or annotation cost. To address this issue, we propose:(1) the first Partial Weakly-Supervised Oriented Object Detection (PWOOD) framework based on partially weak annotations (horizontal boxes or single points), which can efficiently leverage large amounts of unlabeled data, significantly outperforming weakly supervised algorithms trained with partially weak annotations, also offers a lower cost solution; (2) Orientation-and-Scale-aware Student (OS-Student) model capable of learning orientation and scale information with only a small amount of orientation-agnostic or scale-agnostic weak annotations; and (3) Class-Agnostic Pseudo-Label Filtering strategy (CPF) to reduce the model's sensitivity to static filtering thresholds. Comprehensive experiments on DOTA-v1.0/v1.5/v2.0 and DIOR datasets demonstrate that our PWOOD framework performs comparably to, or even surpasses, traditional semi-supervised algorithms.
>
---
#### [new 073] Automatic Labelling for Low-Light Pedestrian Detection
- **分类: cs.CV**

- **简介: 该论文属于低光行人检测任务，旨在解决低光条件下缺乏标注数据的问题。通过红外-RGB自动标注管道生成标签，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.02513v1](http://arxiv.org/pdf/2507.02513v1)**

> **作者:** Dimitrios Bouzoulas; Eerik Alamikkotervo; Risto Ojala
>
> **摘要:** Pedestrian detection in RGB images is a key task in pedestrian safety, as the most common sensor in autonomous vehicles and advanced driver assistance systems is the RGB camera. A challenge in RGB pedestrian detection, that does not appear to have large public datasets, is low-light conditions. As a solution, in this research, we propose an automated infrared-RGB labeling pipeline. The proposed pipeline consists of 1) Infrared detection, where a fine-tuned model for infrared pedestrian detection is used 2) Label transfer process from the infrared detections to their RGB counterparts 3) Training object detection models using the generated labels for low-light RGB pedestrian detection. The research was performed using the KAIST dataset. For the evaluation, object detection models were trained on the generated autolabels and ground truth labels. When compared on a previously unseen image sequence, the results showed that the models trained on generated labels outperformed the ones trained on ground-truth labels in 6 out of 9 cases for the mAP@50 and mAP@50-95 metrics. The source code for this research is available at https://github.com/BouzoulasDimitrios/IR-RGB-Automated-LowLight-Pedestrian-Labeling
>
---
#### [new 074] MAGIC: Mask-Guided Diffusion Inpainting with Multi-Level Perturbations and Context-Aware Alignment for Few-Shot Anomaly Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于异常生成任务，旨在解决少样本下保持背景完整、精确覆盖掩码和生成语义合理异常的问题。提出MAGIC方法，结合多级扰动与上下文对齐，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2507.02314v1](http://arxiv.org/pdf/2507.02314v1)**

> **作者:** JaeHyuck Choi; MinJun Kim; JeHyeong Hong
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Few-shot anomaly generation is emerging as a practical solution for augmenting the scarce anomaly data in industrial quality control settings. An ideal generator would meet three demands at once, namely (i) keep the normal background intact, (ii) inpaint anomalous regions to tightly overlap with the corresponding anomaly masks, and (iii) generate anomalous regions in a semantically valid location, while still producing realistic, diverse appearances from only a handful of real examples. Existing diffusion-based methods usually satisfy at most two of these requirements: global anomaly generators corrupt the background, whereas mask-guided ones often falter when the mask is imprecise or misplaced. We propose MAGIC--Mask-guided inpainting with multi-level perturbations and Context-aware alignment--to resolve all three issues. At its core, MAGIC fine-tunes a Stable Diffusion inpainting backbone that preserves normal regions and ensures strict adherence of the synthesized anomaly to the supplied mask, directly addressing background corruption and misalignment. To offset the diversity loss that fine-tuning can cause, MAGIC adds two complementary perturbation strategies: (i) Gaussian prompt-level perturbation applied during fine-tuning and inference that broadens the global appearance of anomalies while avoiding low-fidelity textual appearances, and (ii) mask-guided spatial noise injection that enriches local texture variations. Additionally, the context-aware mask alignment module forms semantic correspondences and relocates masks so that every anomaly remains plausibly contained within the host object, eliminating out-of-boundary artifacts. Under a consistent identical evaluation protocol on the MVTec-AD dataset, MAGIC outperforms previous state-of-the-arts in downstream anomaly tasks.
>
---
#### [new 075] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于面部动画生成任务，旨在提升单目视频生成的高质量可动画化人脸形象。通过引入高维高斯分布（HyperGaussians）解决细节表现不足的问题。**

- **链接: [http://arxiv.org/pdf/2507.02803v1](http://arxiv.org/pdf/2507.02803v1)**

> **作者:** Gent Serifi; Marcel C. Bühler
>
> **备注:** Project page: https://gserifi.github.io/HyperGaussians
>
> **摘要:** We introduce HyperGaussians, a novel extension of 3D Gaussian Splatting for high-quality animatable face avatars. Creating such detailed face avatars from videos is a challenging problem and has numerous applications in augmented and virtual reality. While tremendous successes have been achieved for static faces, animatable avatars from monocular videos still fall in the uncanny valley. The de facto standard, 3D Gaussian Splatting (3DGS), represents a face through a collection of 3D Gaussian primitives. 3DGS excels at rendering static faces, but the state-of-the-art still struggles with nonlinear deformations, complex lighting effects, and fine details. While most related works focus on predicting better Gaussian parameters from expression codes, we rethink the 3D Gaussian representation itself and how to make it more expressive. Our insights lead to a novel extension of 3D Gaussians to high-dimensional multivariate Gaussians, dubbed 'HyperGaussians'. The higher dimensionality increases expressivity through conditioning on a learnable local embedding. However, splatting HyperGaussians is computationally expensive because it requires inverting a high-dimensional covariance matrix. We solve this by reparameterizing the covariance matrix, dubbed the 'inverse covariance trick'. This trick boosts the efficiency so that HyperGaussians can be seamlessly integrated into existing models. To demonstrate this, we plug in HyperGaussians into the state-of-the-art in fast monocular face avatars: FlashAvatar. Our evaluation on 19 subjects from 4 face datasets shows that HyperGaussians outperform 3DGS numerically and visually, particularly for high-frequency details like eyeglass frames, teeth, complex facial movements, and specular reflections.
>
---
#### [new 076] FairHuman: Boosting Hand and Face Quality in Human Image Generation with Minimum Potential Delay Fairness in Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决人类图像中面部和手部细节生成质量低的问题。通过多目标微调和公平优化策略提升生成效果。**

- **链接: [http://arxiv.org/pdf/2507.02714v1](http://arxiv.org/pdf/2507.02714v1)**

> **作者:** Yuxuan Wang; Tianwei Cao; Huayu Zhang; Zhongjiang He; Kongming Liang; Zhanyu Ma
>
> **备注:** ICCV 2025
>
> **摘要:** Image generation has achieved remarkable progress with the development of large-scale text-to-image models, especially diffusion-based models. However, generating human images with plausible details, such as faces or hands, remains challenging due to insufficient supervision of local regions during training. To address this issue, we propose FairHuman, a multi-objective fine-tuning approach designed to enhance both global and local generation quality fairly. Specifically, we first construct three learning objectives: a global objective derived from the default diffusion objective function and two local objectives for hands and faces based on pre-annotated positional priors. Subsequently, we derive the optimal parameter updating strategy under the guidance of the Minimum Potential Delay (MPD) criterion, thereby attaining fairness-ware optimization for this multi-objective problem. Based on this, our proposed method can achieve significant improvements in generating challenging local details while maintaining overall quality. Extensive experiments showcase the effectiveness of our method in improving the performance of human image generation under different scenarios.
>
---
#### [new 077] High-Fidelity Differential-information Driven Binary Vision Transformer
- **分类: cs.CV**

- **简介: 该论文属于视觉Transformer的二值化任务，旨在解决二值化导致的性能下降问题。提出DIDB-ViT，通过差分信息和频率分解提升模型精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.02222v1](http://arxiv.org/pdf/2507.02222v1)**

> **作者:** Tian Gao; Zhiyuan Zhang; Kaijie Yin; Xu-Cheng Zhong; Hui Kong
>
> **摘要:** The binarization of vision transformers (ViTs) offers a promising approach to addressing the trade-off between high computational/storage demands and the constraints of edge-device deployment. However, existing binary ViT methods often suffer from severe performance degradation or rely heavily on full-precision modules. To address these issues, we propose DIDB-ViT, a novel binary ViT that is highly informative while maintaining the original ViT architecture and computational efficiency. Specifically, we design an informative attention module incorporating differential information to mitigate information loss caused by binarization and enhance high-frequency retention. To preserve the fidelity of the similarity calculations between binary Q and K tensors, we apply frequency decomposition using the discrete Haar wavelet and integrate similarities across different frequencies. Additionally, we introduce an improved RPReLU activation function to restructure the activation distribution, expanding the model's representational capacity. Experimental results demonstrate that our DIDB-ViT significantly outperforms state-of-the-art network quantization methods in multiple ViT architectures, achieving superior image classification and segmentation performance.
>
---
#### [new 078] Spotlighting Partially Visible Cinematic Language for Video-to-Audio Generation via Self-distillation
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于视频到音频生成任务，解决部分可见场景下忽略电影语言导致性能下降的问题，通过自蒸馏方法提升模型对视听关联的捕捉能力。**

- **链接: [http://arxiv.org/pdf/2507.02271v1](http://arxiv.org/pdf/2507.02271v1)**

> **作者:** Feizhen Huang; Yu Wu; Yutian Lin; Bo Du
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Video-to-Audio (V2A) Generation achieves significant progress and plays a crucial role in film and video post-production. However, current methods overlook the cinematic language, a critical component of artistic expression in filmmaking. As a result, their performance deteriorates in scenarios where Foley targets are only partially visible. To address this challenge, we propose a simple self-distillation approach to extend V2A models to cinematic language scenarios. By simulating the cinematic language variations, the student model learns to align the video features of training pairs with the same audio-visual correspondences, enabling it to effectively capture the associations between sounds and partial visual information. Our method not only achieves impressive improvements under partial visibility across all evaluation metrics, but also enhances performance on the large-scale V2A dataset, VGGSound.
>
---
#### [new 079] Lightweight Shrimp Disease Detection Research Based on YOLOv8n
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决虾类疾病智能检测效率低的问题。通过改进YOLOv8n模型，提升检测精度与计算效率。**

- **链接: [http://arxiv.org/pdf/2507.02354v1](http://arxiv.org/pdf/2507.02354v1)**

> **作者:** Fei Yuhuan; Wang Gengchen; Liu Fenghao; Zang Ran; Sun Xufei; Chang Hao
>
> **备注:** in Chinese language
>
> **摘要:** Shrimp diseases are one of the primary causes of economic losses in shrimp aquaculture. To prevent disease transmission and enhance intelligent detection efficiency in shrimp farming, this paper proposes a lightweight network architecture based on YOLOv8n. First, by designing the RLDD detection head and C2f-EMCM module, the model reduces computational complexity while maintaining detection accuracy, improving computational efficiency. Subsequently, an improved SegNext_Attention self-attention mechanism is introduced to further enhance the model's feature extraction capability, enabling more precise identification of disease characteristics. Extensive experiments, including ablation studies and comparative evaluations, are conducted on a self-constructed shrimp disease dataset, with generalization tests extended to the URPC2020 dataset. Results demonstrate that the proposed model achieves a 32.3% reduction in parameters compared to the original YOLOv8n, with a mAP@0.5 of 92.7% (3% improvement over YOLOv8n). Additionally, the model outperforms other lightweight YOLO-series models in mAP@0.5, parameter count, and model size. Generalization experiments on the URPC2020 dataset further validate the model's robustness, showing a 4.1% increase in mAP@0.5 compared to YOLOv8n. The proposed method achieves an optimal balance between accuracy and efficiency, providing reliable technical support for intelligent disease detection in shrimp aquaculture.
>
---
#### [new 080] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning
- **分类: cs.CV**

- **简介: 该论文属于人体交互重建任务，解决复杂场景下人体姿态与接触关系估计问题。通过结合外观和社交距离先验，提出双分支优化框架，提升交互动作的准确性。**

- **链接: [http://arxiv.org/pdf/2507.02565v1](http://arxiv.org/pdf/2507.02565v1)**

> **作者:** Buzhen Huang; Chen Li; Chongyang Xu; Dongyue Lu; Jinnan Chen; Yangang Wang; Gim Hee Lee
>
> **摘要:** Due to visual ambiguities and inter-person occlusions, existing human pose estimation methods cannot recover plausible close interactions from in-the-wild videos. Even state-of-the-art large foundation models~(\eg, SAM) cannot accurately distinguish human semantics in such challenging scenarios. In this work, we find that human appearance can provide a straightforward cue to address these obstacles. Based on this observation, we propose a dual-branch optimization framework to reconstruct accurate interactive motions with plausible body contacts constrained by human appearances, social proxemics, and physical laws. Specifically, we first train a diffusion model to learn the human proxemic behavior and pose prior knowledge. The trained network and two optimizable tensors are then incorporated into a dual-branch optimization framework to reconstruct human motions and appearances. Several constraints based on 3D Gaussians, 2D keypoints, and mesh penetrations are also designed to assist the optimization. With the proxemics prior and diverse constraints, our method is capable of estimating accurate interactions from in-the-wild videos captured in complex environments. We further build a dataset with pseudo ground-truth interaction annotations, which may promote future research on pose estimation and human behavior understanding. Experimental results on several benchmarks demonstrate that our method outperforms existing approaches. The code and data are available at https://www.buzhenhuang.com/works/CloseApp.html.
>
---
#### [new 081] LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with TriMap Video Diffusion
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在从稀疏视角生成可泛化的语言嵌入场景。通过引入TriMap视频扩散模型和LQC编码器，实现多模态一致性与开放语言查询支持。**

- **链接: [http://arxiv.org/pdf/2507.02813v1](http://arxiv.org/pdf/2507.02813v1)**

> **作者:** Fangfu Liu; Hao Li; Jiawei Chi; Hanyang Wang; Minghui Yang; Fudong Wang; Yueqi Duan
>
> **备注:** Project page: https://liuff19.github.io/LangScene-X
>
> **摘要:** Recovering 3D structures with open-vocabulary scene understanding from 2D images is a fundamental but daunting task. Recent developments have achieved this by performing per-scene optimization with embedded language information. However, they heavily rely on the calibrated dense-view reconstruction paradigm, thereby suffering from severe rendering artifacts and implausible semantic synthesis when limited views are available. In this paper, we introduce a novel generative framework, coined LangScene-X, to unify and generate 3D consistent multi-modality information for reconstruction and understanding. Powered by the generative capability of creating more consistent novel observations, we can build generalizable 3D language-embedded scenes from only sparse views. Specifically, we first train a TriMap video diffusion model that can generate appearance (RGBs), geometry (normals), and semantics (segmentation maps) from sparse inputs through progressive knowledge integration. Furthermore, we propose a Language Quantized Compressor (LQC), trained on large-scale image datasets, to efficiently encode language embeddings, enabling cross-scene generalization without per-scene retraining. Finally, we reconstruct the language surface fields by aligning language information onto the surface of 3D scenes, enabling open-ended language queries. Extensive experiments on real-world data demonstrate the superiority of our LangScene-X over state-of-the-art methods in terms of quality and generalizability. Project Page: https://liuff19.github.io/LangScene-X.
>
---
#### [new 082] TABNet: A Triplet Augmentation Self-Recovery Framework with Boundary-Aware Pseudo-Labels for Medical Image Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，解决稀疏标注（如涂鸦标注）下模型性能不足的问题，提出TAB Net框架提升分割效果。**

- **链接: [http://arxiv.org/pdf/2507.02399v1](http://arxiv.org/pdf/2507.02399v1)**

> **作者:** Peilin Zhang; Shaouxan Wua; Jun Feng; Zhuo Jin; Zhizezhang Gao; Jingkun Chen; Yaqiong Xing; Xiao Zhang
>
> **摘要:** Background and objective: Medical image segmentation is a core task in various clinical applications. However, acquiring large-scale, fully annotated medical image datasets is both time-consuming and costly. Scribble annotations, as a form of sparse labeling, provide an efficient and cost-effective alternative for medical image segmentation. However, the sparsity of scribble annotations limits the feature learning of the target region and lacks sufficient boundary supervision, which poses significant challenges for training segmentation networks. Methods: We propose TAB Net, a novel weakly-supervised medical image segmentation framework, consisting of two key components: the triplet augmentation self-recovery (TAS) module and the boundary-aware pseudo-label supervision (BAP) module. The TAS module enhances feature learning through three complementary augmentation strategies: intensity transformation improves the model's sensitivity to texture and contrast variations, cutout forces the network to capture local anatomical structures by masking key regions, and jigsaw augmentation strengthens the modeling of global anatomical layout by disrupting spatial continuity. By guiding the network to recover complete masks from diverse augmented inputs, TAS promotes a deeper semantic understanding of medical images under sparse supervision. The BAP module enhances pseudo-supervision accuracy and boundary modeling by fusing dual-branch predictions into a loss-weighted pseudo-label and introducing a boundary-aware loss for fine-grained contour refinement. Results: Experimental evaluations on two public datasets, ACDC and MSCMR seg, demonstrate that TAB Net significantly outperforms state-of-the-art methods for scribble-based weakly supervised segmentation. Moreover, it achieves performance comparable to that of fully supervised methods.
>
---
#### [new 083] Learning few-step posterior samplers by unfolding and distillation of diffusion models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，旨在提升扩散模型的后验采样效率。通过深度展开与模型蒸馏，将扩散模型转化为少步条件模型，解决计算效率与灵活性问题。**

- **链接: [http://arxiv.org/pdf/2507.02686v1](http://arxiv.org/pdf/2507.02686v1)**

> **作者:** Charlesquin Kemajou Mbakam; Jonathan Spence; Marcelo Pereyra
>
> **备注:** 28 pages, 16 figures, 10 tables
>
> **摘要:** Diffusion models (DMs) have emerged as powerful image priors in Bayesian computational imaging. Two primary strategies have been proposed for leveraging DMs in this context: Plug-and-Play methods, which are zero-shot and highly flexible but rely on approximations; and specialized conditional DMs, which achieve higher accuracy and faster inference for specific tasks through supervised training. In this work, we introduce a novel framework that integrates deep unfolding and model distillation to transform a DM image prior into a few-step conditional model for posterior sampling. A central innovation of our approach is the unfolding of a Markov chain Monte Carlo (MCMC) algorithm - specifically, the recently proposed LATINO Langevin sampler (Spagnoletti et al., 2025) - representing the first known instance of deep unfolding applied to a Monte Carlo sampling scheme. We demonstrate our proposed unfolded and distilled samplers through extensive experiments and comparisons with the state of the art, where they achieve excellent accuracy and computational efficiency, while retaining the flexibility to adapt to variations in the forward model at inference time.
>
---
#### [new 084] Energy-Based Transformers are Scalable Learners and Thinkers
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出Energy-Based Transformers（EBTs），解决模型学习与推理能力提升问题。通过无监督学习实现系统2思维，提升语言和图像任务性能。**

- **链接: [http://arxiv.org/pdf/2507.02092v1](http://arxiv.org/pdf/2507.02092v1)**

> **作者:** Alexi Gladstone; Ganesh Nanduru; Md Mofijul Islam; Peixuan Han; Hyeonjeong Ha; Aman Chadha; Yilun Du; Heng Ji; Jundong Li; Tariq Iqbal
>
> **摘要:** Inference-time computation techniques, analogous to human System 2 Thinking, have recently become popular for improving model performances. However, most existing approaches suffer from several limitations: they are modality-specific (e.g., working only in text), problem-specific (e.g., verifiable domains like math and coding), or require additional supervision/training on top of unsupervised pretraining (e.g., verifiers or verifiable rewards). In this paper, we ask the question "Is it possible to generalize these System 2 Thinking approaches, and develop models that learn to think solely from unsupervised learning?" Interestingly, we find the answer is yes, by learning to explicitly verify the compatibility between inputs and candidate-predictions, and then re-framing prediction problems as optimization with respect to this verifier. Specifically, we train Energy-Based Transformers (EBTs) -- a new class of Energy-Based Models (EBMs) -- to assign an energy value to every input and candidate-prediction pair, enabling predictions through gradient descent-based energy minimization until convergence. Across both discrete (text) and continuous (visual) modalities, we find EBTs scale faster than the dominant Transformer++ approach during training, achieving an up to 35% higher scaling rate with respect to data, batch size, parameters, FLOPs, and depth. During inference, EBTs improve performance with System 2 Thinking by 29% more than the Transformer++ on language tasks, and EBTs outperform Diffusion Transformers on image denoising while using fewer forward passes. Further, we find that EBTs achieve better results than existing models on most downstream tasks given the same or worse pretraining performance, suggesting that EBTs generalize better than existing approaches. Consequently, EBTs are a promising new paradigm for scaling both the learning and thinking capabilities of models.
>
---
#### [new 085] MultiGen: Using Multimodal Generation in Simulation to Learn Multimodal Policies in Real
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决多模态策略在现实中的迁移问题。通过整合生成模型到物理模拟器中，实现多感官仿真，提升机器人多模态感知与决策能力。**

- **链接: [http://arxiv.org/pdf/2507.02864v1](http://arxiv.org/pdf/2507.02864v1)**

> **作者:** Renhao Wang; Haoran Geng; Tingle Li; Feishi Wang; Gopala Anumanchipalli; Philipp Wu; Trevor Darrell; Boyi Li; Pieter Abbeel; Jitendra Malik; Alexei A. Efros
>
> **摘要:** Robots must integrate multiple sensory modalities to act effectively in the real world. Yet, learning such multimodal policies at scale remains challenging. Simulation offers a viable solution, but while vision has benefited from high-fidelity simulators, other modalities (e.g. sound) can be notoriously difficult to simulate. As a result, sim-to-real transfer has succeeded primarily in vision-based tasks, with multimodal transfer still largely unrealized. In this work, we tackle these challenges by introducing MultiGen, a framework that integrates large-scale generative models into traditional physics simulators, enabling multisensory simulation. We showcase our framework on the dynamic task of robot pouring, which inherently relies on multimodal feedback. By synthesizing realistic audio conditioned on simulation video, our method enables training on rich audiovisual trajectories -- without any real robot data. We demonstrate effective zero-shot transfer to real-world pouring with novel containers and liquids, highlighting the potential of generative modeling to both simulate hard-to-model modalities and close the multimodal sim-to-real gap.
>
---
#### [new 086] Real-time Image-based Lighting of Glints
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于实时渲染任务，解决动态材料与环境光下的闪烁效果问题。提出一种高效近似方法，通过环境图过滤和概率采样实现实时glints渲染。**

- **链接: [http://arxiv.org/pdf/2507.02674v1](http://arxiv.org/pdf/2507.02674v1)**

> **作者:** Tom Kneiphof; Reinhard Klein
>
> **摘要:** Image-based lighting is a widely used technique to reproduce shading under real-world lighting conditions, especially in real-time rendering applications. A particularly challenging scenario involves materials exhibiting a sparkling or glittering appearance, caused by discrete microfacets scattered across their surface. In this paper, we propose an efficient approximation for image-based lighting of glints, enabling fully dynamic material properties and environment maps. Our novel approach is grounded in real-time glint rendering under area light illumination and employs standard environment map filtering techniques. Crucially, our environment map filtering process is sufficiently fast to be executed on a per-frame basis. Our method assumes that the environment map is partitioned into few homogeneous regions of constant radiance. By filtering the corresponding indicator functions with the normal distribution function, we obtain the probabilities for individual microfacets to reflect light from each region. During shading, these probabilities are utilized to hierarchically sample a multinomial distribution, facilitated by our novel dual-gated Gaussian approximation of binomial distributions. We validate that our real-time approximation is close to ground-truth renderings for a range of material properties and lighting conditions, and demonstrate robust and stable performance, with little overhead over rendering glints from a single directional light. Compared to rendering smooth materials without glints, our approach requires twice as much memory to store the prefiltered environment map.
>
---
#### [new 087] Grounding Intelligence in Movement
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于AI与运动建模任务，旨在解决运动数据建模不足的问题，提出将运动作为核心建模目标，以提升智能系统对世界的理解能力。**

- **链接: [http://arxiv.org/pdf/2507.02771v1](http://arxiv.org/pdf/2507.02771v1)**

> **作者:** Melanie Segado; Felipe Parodi; Jordan K. Matelsky; Michael L. Platt; Eva B. Dyer; Konrad P. Kording
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Recent advances in machine learning have dramatically improved our ability to model language, vision, and other high-dimensional data, yet they continue to struggle with one of the most fundamental aspects of biological systems: movement. Across neuroscience, medicine, robotics, and ethology, movement is essential for interpreting behavior, predicting intent, and enabling interaction. Despite its core significance in our intelligence, movement is often treated as an afterthought rather than as a rich and structured modality in its own right. This reflects a deeper fragmentation in how movement data is collected and modeled, often constrained by task-specific goals and domain-specific assumptions. But movement is not domain-bound. It reflects shared physical constraints, conserved morphological structures, and purposeful dynamics that cut across species and settings. We argue that movement should be treated as a primary modeling target for AI. It is inherently structured and grounded in embodiment and physics. This structure, often allowing for compact, lower-dimensional representations (e.g., pose), makes it more interpretable and computationally tractable to model than raw, high-dimensional sensory inputs. Developing models that can learn from and generalize across diverse movement data will not only advance core capabilities in generative modeling and control, but also create a shared foundation for understanding behavior across biological and artificial systems. Movement is not just an outcome, it is a window into how intelligent systems engage with the world.
>
---
#### [new 088] MISCGrasp: Leveraging Multiple Integrated Scales and Contrastive Learning for Enhanced Volumetric Grasping
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，旨在解决不同形状和大小物体的自适应抓取问题。通过多尺度特征提取与对比学习，提升抓取性能。**

- **链接: [http://arxiv.org/pdf/2507.02672v1](http://arxiv.org/pdf/2507.02672v1)**

> **作者:** Qingyu Fan; Yinghao Cai; Chao Li; Chunting Jiao; Xudong Zheng; Tao Lu; Bin Liang; Shuo Wang
>
> **备注:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025
>
> **摘要:** Robotic grasping faces challenges in adapting to objects with varying shapes and sizes. In this paper, we introduce MISCGrasp, a volumetric grasping method that integrates multi-scale feature extraction with contrastive feature enhancement for self-adaptive grasping. We propose a query-based interaction between high-level and low-level features through the Insight Transformer, while the Empower Transformer selectively attends to the highest-level features, which synergistically strikes a balance between focusing on fine geometric details and overall geometric structures. Furthermore, MISCGrasp utilizes multi-scale contrastive learning to exploit similarities among positive grasp samples, ensuring consistency across multi-scale features. Extensive experiments in both simulated and real-world environments demonstrate that MISCGrasp outperforms baseline and variant methods in tabletop decluttering tasks. More details are available at https://miscgrasp.github.io/.
>
---
#### [new 089] Fair Deepfake Detectors Can Generalize
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于深度伪造检测任务，解决模型在不同场景下的泛化能力和群体公平性之间的冲突问题。通过引入DAID框架提升检测效果。**

- **链接: [http://arxiv.org/pdf/2507.02645v1](http://arxiv.org/pdf/2507.02645v1)**

> **作者:** Harry Cheng; Ming-Hui Liu; Yangyang Guo; Tianyi Wang; Liqiang Nie; Mohan Kankanhalli
>
> **备注:** 14 pages, version 1
>
> **摘要:** Deepfake detection models face two critical challenges: generalization to unseen manipulations and demographic fairness among population groups. However, existing approaches often demonstrate that these two objectives are inherently conflicting, revealing a trade-off between them. In this paper, we, for the first time, uncover and formally define a causal relationship between fairness and generalization. Building on the back-door adjustment, we show that controlling for confounders (data distribution and model capacity) enables improved generalization via fairness interventions. Motivated by this insight, we propose Demographic Attribute-insensitive Intervention Detection (DAID), a plug-and-play framework composed of: i) Demographic-aware data rebalancing, which employs inverse-propensity weighting and subgroup-wise feature normalization to neutralize distributional biases; and ii) Demographic-agnostic feature aggregation, which uses a novel alignment loss to suppress sensitive-attribute signals. Across three cross-domain benchmarks, DAID consistently achieves superior performance in both fairness and generalization compared to several state-of-the-art detectors, validating both its theoretical foundation and practical effectiveness.
>
---
#### [new 090] A robust and versatile deep learning model for prediction of the arterial input function in dynamic small animal $\left[^{18}\text{F}\right]$FDG PET imaging
- **分类: eess.IV; cs.CV; physics.med-ph; q-bio.QM**

- **简介: 该论文属于医学影像分析任务，旨在解决小动物PET中动脉输入函数（AIF）的非侵入性预测问题。通过深度学习模型FC-DLIF，无需血液采样即可准确估计AIF。**

- **链接: [http://arxiv.org/pdf/2507.02367v1](http://arxiv.org/pdf/2507.02367v1)**

> **作者:** Christian Salomonsen; Luigi Tommaso Luppino; Fredrik Aspheim; Kristoffer Wickstrøm; Elisabeth Wetzer; Michael Kampffmeyer; Rodrigo Berzaghi; Rune Sundset; Robert Jenssen; Samuel Kuttner
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** Dynamic positron emission tomography (PET) and kinetic modeling are pivotal in advancing tracer development research in small animal studies. Accurate kinetic modeling requires precise input function estimation, traditionally achieved via arterial blood sampling. However, arterial cannulation in small animals like mice, involves intricate, time-consuming, and terminal procedures, precluding longitudinal studies. This work proposes a non-invasive, fully convolutional deep learning-based approach (FC-DLIF) to predict input functions directly from PET imaging, potentially eliminating the need for blood sampling in dynamic small-animal PET. The proposed FC-DLIF model includes a spatial feature extractor acting on the volumetric time frames of the PET sequence, extracting spatial features. These are subsequently further processed in a temporal feature extractor that predicts the arterial input function. The proposed approach is trained and evaluated using images and arterial blood curves from [$^{18}$F]FDG data using cross validation. Further, the model applicability is evaluated on imaging data and arterial blood curves collected using two additional radiotracers ([$^{18}$F]FDOPA, and [$^{68}$Ga]PSMA). The model was further evaluated on data truncated and shifted in time, to simulate shorter, and shifted, PET scans. The proposed FC-DLIF model reliably predicts the arterial input function with respect to mean squared error and correlation. Furthermore, the FC-DLIF model is able to predict the arterial input function even from truncated and shifted samples. The model fails to predict the AIF from samples collected using different radiotracers, as these are not represented in the training data. Our deep learning-based input function offers a non-invasive and reliable alternative to arterial blood sampling, proving robust and flexible to temporal shifts and different scan durations.
>
---
#### [new 091] Generative Latent Diffusion for Efficient Spatiotemporal Data Reduction
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于数据压缩任务，旨在解决生成模型在压缩中的可控性和精度不足问题。通过结合变分自编码器和条件扩散模型，实现高效时空数据压缩。**

- **链接: [http://arxiv.org/pdf/2507.02129v1](http://arxiv.org/pdf/2507.02129v1)**

> **作者:** Xiao Li; Liangji Zhu; Anand Rangarajan; Sanjay Ranka
>
> **备注:** 10 pages
>
> **摘要:** Generative models have demonstrated strong performance in conditional settings and can be viewed as a form of data compression, where the condition serves as a compact representation. However, their limited controllability and reconstruction accuracy restrict their practical application to data compression. In this work, we propose an efficient latent diffusion framework that bridges this gap by combining a variational autoencoder with a conditional diffusion model. Our method compresses only a small number of keyframes into latent space and uses them as conditioning inputs to reconstruct the remaining frames via generative interpolation, eliminating the need to store latent representations for every frame. This approach enables accurate spatiotemporal reconstruction while significantly reducing storage costs. Experimental results across multiple datasets show that our method achieves up to 10 times higher compression ratios than rule-based state-of-the-art compressors such as SZ3, and up to 63 percent better performance than leading learning-based methods under the same reconstruction error.
>
---
#### [new 092] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于模型微调任务，旨在解决持续领域自适应预训练中的计算成本高、数据顺序敏感及模型泛化性差的问题。提出DoMIX框架，利用LoRA实现高效、并行的领域适配。**

- **链接: [http://arxiv.org/pdf/2507.02302v1](http://arxiv.org/pdf/2507.02302v1)**

> **作者:** Dohoon Kim; Donghun Kang; Taesup Moon
>
> **备注:** 22 pages, 5 figures, ACL 2025 Main
>
> **摘要:** Domain-Adaptive Pre-training (DAP) has recently gained attention for its effectiveness in fine-tuning pre-trained models. Building on this, continual DAP has been explored to develop pre-trained models capable of incrementally incorporating different domain datasets. However, existing continual DAP methods face several limitations: (1) high computational cost and GPU memory usage during training; (2) sensitivity to incremental data order; and (3) providing a single, generalized model for all end tasks, which contradicts the essence of DAP. In this paper, we propose DoMIX, a novel approach that addresses these challenges by leveraging LoRA modules, a representative parameter-efficient fine-tuning (PEFT) method. Our approach enables efficient and parallel domain-adaptive pre-training that is robust to domain order and effectively utilizes accumulated knowledge to provide tailored pre-trained models for specific tasks. We also demonstrate that our method can be extended beyond the DAP setting to standard LLM fine-tuning scenarios. Code is available at https://github.com/dohoonkim-ai/DoMIX.
>
---
#### [new 093] MEGANet-W: A Wavelet-Driven Edge-Guided Attention Framework for Weak Boundary Polyp Detection
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于结直肠息肉分割任务，旨在解决弱边界检测难题。提出MEGANet-W，通过小波边缘引导注意力机制提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.02668v1](http://arxiv.org/pdf/2507.02668v1)**

> **作者:** Zhe Yee Tan
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Colorectal polyp segmentation is critical for early detection of colorectal cancer, yet weak and low contrast boundaries significantly limit automated accuracy. Existing deep models either blur fine edge details or rely on handcrafted filters that perform poorly under variable imaging conditions. We propose MEGANet-W, a Wavelet Driven Edge Guided Attention Network that injects directional, parameter free Haar wavelet edge maps into each decoder stage to recalibrate semantic features. Our two main contributions are: (1) a two-level Haar wavelet head for multi orientation edge extraction; and (2) Wavelet Edge Guided Attention (WEGA) modules that fuse wavelet cues with reverse and input branches. On five public polyp datasets, MEGANetW consistently outperforms existing methods, improving mIoU by up to 2.3% and mDice by 1.2%, while introducing no additional learnable parameters.
>
---
#### [new 094] 3D Heart Reconstruction from Sparse Pose-agnostic 2D Echocardiographic Slices
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D心脏重建任务，旨在解决2D超声图像难以准确评估心室体积的问题。通过创新框架，从稀疏2D切片重建个性化3D心脏模型，提升体积估算精度。**

- **链接: [http://arxiv.org/pdf/2507.02411v1](http://arxiv.org/pdf/2507.02411v1)**

> **作者:** Zhurong Chen; Jinhua Chen; Wei Zhuo; Wufeng Xue; Dong Ni
>
> **备注:** 10 pages
>
> **摘要:** Echocardiography (echo) plays an indispensable role in the clinical practice of heart diseases. However, ultrasound imaging typically provides only two-dimensional (2D) cross-sectional images from a few specific views, making it challenging to interpret and inaccurate for estimation of clinical parameters like the volume of left ventricle (LV). 3D ultrasound imaging provides an alternative for 3D quantification, but is still limited by the low spatial and temporal resolution and the highly demanding manual delineation. To address these challenges, we propose an innovative framework for reconstructing personalized 3D heart anatomy from 2D echo slices that are frequently used in clinical practice. Specifically, a novel 3D reconstruction pipeline is designed, which alternatively optimizes between the 3D pose estimation of these 2D slices and the 3D integration of these slices using an implicit neural network, progressively transforming a prior 3D heart shape into a personalized 3D heart model. We validate the method with two datasets. When six planes are used, the reconstructed 3D heart can lead to a significant improvement for LV volume estimation over the bi-plane method (error in percent: 1.98\% VS. 20.24\%). In addition, the whole reconstruction framework makes even an important breakthrough that can estimate RV volume from 2D echo slices (with an error of 5.75\% ). This study provides a new way for personalized 3D structure and function analysis from cardiac ultrasound and is of great potential in clinical practice.
>
---
#### [new 095] Holistic Continual Learning under Concept Drift with Adaptive Memory Realignment
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于持续学习任务，解决概念漂移下的知识遗忘问题。提出AMR方法，通过自适应记忆重对齐，提升模型在动态数据中的稳定性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.02310v1](http://arxiv.org/pdf/2507.02310v1)**

> **作者:** Alif Ashrafee; Jedrzej Kozal; Michal Wozniak; Bartosz Krawczyk
>
> **摘要:** Traditional continual learning methods prioritize knowledge retention and focus primarily on mitigating catastrophic forgetting, implicitly assuming that the data distribution of previously learned tasks remains static. This overlooks the dynamic nature of real-world data streams, where concept drift permanently alters previously seen data and demands both stability and rapid adaptation. We introduce a holistic framework for continual learning under concept drift that simulates realistic scenarios by evolving task distributions. As a baseline, we consider Full Relearning (FR), in which the model is retrained from scratch on newly labeled samples from the drifted distribution. While effective, this approach incurs substantial annotation and computational overhead. To address these limitations, we propose Adaptive Memory Realignment (AMR), a lightweight alternative that equips rehearsal-based learners with a drift-aware adaptation mechanism. AMR selectively removes outdated samples of drifted classes from the replay buffer and repopulates it with a small number of up-to-date instances, effectively realigning memory with the new distribution. This targeted resampling matches the performance of FR while reducing the need for labeled data and computation by orders of magnitude. To enable reproducible evaluation, we introduce four concept-drift variants of standard vision benchmarks: Fashion-MNIST-CD, CIFAR10-CD, CIFAR100-CD, and Tiny-ImageNet-CD, where previously seen classes reappear with shifted representations. Comprehensive experiments on these datasets using several rehearsal-based baselines show that AMR consistently counters concept drift, maintaining high accuracy with minimal overhead. These results position AMR as a scalable solution that reconciles stability and plasticity in non-stationary continual learning environments.
>
---
#### [new 096] Embedding-Based Federated Data Sharing via Differentially Private Conditional VAEs
- **分类: cs.LG; cs.CV; eess.IV**

- **简介: 该论文属于医疗图像领域，解决数据隐私与共享问题。通过DP-CVAE生成高保真嵌入，实现安全高效的数据共享与多任务支持。**

- **链接: [http://arxiv.org/pdf/2507.02671v1](http://arxiv.org/pdf/2507.02671v1)**

> **作者:** Francesco Di Salvo; Hanh Huyen My Nguyen; Christian Ledig
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Deep Learning (DL) has revolutionized medical imaging, yet its adoption is constrained by data scarcity and privacy regulations, limiting access to diverse datasets. Federated Learning (FL) enables decentralized training but suffers from high communication costs and is often restricted to a single downstream task, reducing flexibility. We propose a data-sharing method via Differentially Private (DP) generative models. By adopting foundation models, we extract compact, informative embeddings, reducing redundancy and lowering computational overhead. Clients collaboratively train a Differentially Private Conditional Variational Autoencoder (DP-CVAE) to model a global, privacy-aware data distribution, supporting diverse downstream tasks. Our approach, validated across multiple feature extractors, enhances privacy, scalability, and efficiency, outperforming traditional FL classifiers while ensuring differential privacy. Additionally, DP-CVAE produces higher-fidelity embeddings than DP-CGAN while requiring $5{\times}$ fewer parameters.
>
---
#### [new 097] L-VAE: Variational Auto-Encoder with Learnable Beta for Disentangled Representation
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出L-VAE模型，用于学习解耦表示，解决传统VAE在重建与解耦间的平衡问题。通过学习损失函数权重实现动态优化。**

- **链接: [http://arxiv.org/pdf/2507.02619v1](http://arxiv.org/pdf/2507.02619v1)**

> **作者:** Hazal Mogultay Ozcan; Sinan Kalkan; Fatos T. Yarman-Vural
>
> **备注:** The paper is under revision at Machine Vision and Applications
>
> **摘要:** In this paper, we propose a novel model called Learnable VAE (L-VAE), which learns a disentangled representation together with the hyperparameters of the cost function. L-VAE can be considered as an extension of \b{eta}-VAE, wherein the hyperparameter, \b{eta}, is empirically adjusted. L-VAE mitigates the limitations of \b{eta}-VAE by learning the relative weights of the terms in the loss function to control the dynamic trade-off between disentanglement and reconstruction losses. In the proposed model, the weight of the loss terms and the parameters of the model architecture are learned concurrently. An additional regularization term is added to the loss function to prevent bias towards either reconstruction or disentanglement losses. Experimental analyses show that the proposed L-VAE finds an effective balance between reconstruction fidelity and disentangling the latent dimensions. Comparisons of the proposed L-VAE against \b{eta}-VAE, VAE, ControlVAE, DynamicVAE, and {\sigma}-VAE on datasets, such as dSprites, MPI3D-complex, Falcor3D, and Isaac3D reveals that L-VAE consistently provides the best or the second best performances measured by a set of disentanglement metrics. Moreover, qualitative experiments on CelebA dataset, confirm the success of the L-VAE model for disentangling the facial attributes.
>
---
#### [new 098] TubuleTracker: a high-fidelity shareware software to quantify angiogenesis architecture and maturity
- **分类: q-bio.QM; cs.CV; q-bio.CB**

- **简介: 该论文属于医学图像分析任务，旨在解决 angiogenesis 分析效率低、主观性强的问题。研究开发了 tubuleTracker 软件，实现快速客观的血管网络量化分析。**

- **链接: [http://arxiv.org/pdf/2507.02024v1](http://arxiv.org/pdf/2507.02024v1)**

> **作者:** Danish Mahmood; Stephanie Buczkowski; Sahaj Shah; Autumn Anthony; Rohini Desetty; Carlo R Bartoli
>
> **备注:** Abstract word count = [285] Total word count = [3910] Main body text = [2179] References = [30] Table = [0] Figures = [4]
>
> **摘要:** Background: In vitro endothelial cell culture is widely used to study angiogenesis. Histomicrographic images of cell networks are often analyzed manually, a process that is time-consuming and subjective. Automated tools like ImageJ (NIH) can assist, but are often slow and inaccurate. Additionally, as endothelial networks grow more complex, traditional architectural metrics may not fully reflect network maturity. To address these limitations, we developed tubuleTracker, a software tool that quantifies endothelial network architecture and maturity rapidly and objectively. Methods: Human umbilical vein endothelial cells were cultured in an extracellular matrix, and 54 images were acquired using phase contrast microscopy. Each image was analyzed manually by three independent reviewers, and by both ImageJ and tubuleTracker. Key metrics included tubule count, total length, node count, tubule area, and vessel circularity. In parallel, trained scientists rated each image for angiogenesis maturity on a 1-5 scale (1 = most mature). Results: Analysis time per image differed significantly: manual (8 min), ImageJ (58+/-4 s), and tubuleTracker (6+/-2 s) (p<0.0001). Significant differences were also found in tubule count (manual 168+/-SD, tubuleTracker 92+/-SD, ImageJ 433+/-SD), length, and node count (all p<0.0001). tubuleTracker's metrics varied significantly across angiogenesis maturity scores, including tubule count, length, node count, area, and circularity (all p<0.0001). Conclusions: tubuleTracker was faster and more consistent than both manual and ImageJ-based analysis. Vessel circularity proved especially effective in capturing angiogenesis maturity. tubuleTracker is available as free shareware for the biomedical research community.
>
---
#### [new 099] CineMyoPS: Segmenting Myocardial Pathologies from Cine Cardiac MR
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在仅通过 cine CMR 图像分割心肌病灶（瘢痕和水肿），解决多序列 CMR 获取耗时的问题。**

- **链接: [http://arxiv.org/pdf/2507.02289v1](http://arxiv.org/pdf/2507.02289v1)**

> **作者:** Wangbin Ding; Lei Li; Junyi Qiu; Bogen Lin; Mingjing Yang; Liqin Huang; Lianming Wu; Sihan Wang; Xiahai Zhuang
>
> **摘要:** Myocardial infarction (MI) is a leading cause of death worldwide. Late gadolinium enhancement (LGE) and T2-weighted cardiac magnetic resonance (CMR) imaging can respectively identify scarring and edema areas, both of which are essential for MI risk stratification and prognosis assessment. Although combining complementary information from multi-sequence CMR is useful, acquiring these sequences can be time-consuming and prohibitive, e.g., due to the administration of contrast agents. Cine CMR is a rapid and contrast-free imaging technique that can visualize both motion and structural abnormalities of the myocardium induced by acute MI. Therefore, we present a new end-to-end deep neural network, referred to as CineMyoPS, to segment myocardial pathologies, \ie scars and edema, solely from cine CMR images. Specifically, CineMyoPS extracts both motion and anatomy features associated with MI. Given the interdependence between these features, we design a consistency loss (resembling the co-training strategy) to facilitate their joint learning. Furthermore, we propose a time-series aggregation strategy to integrate MI-related features across the cardiac cycle, thereby enhancing segmentation accuracy for myocardial pathologies. Experimental results on a multi-center dataset demonstrate that CineMyoPS achieves promising performance in myocardial pathology segmentation, motion estimation, and anatomy segmentation.
>
---
## 更新

#### [replaced 001] Consistent Story Generation with Asymmetry Zigzag Sampling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09612v3](http://arxiv.org/pdf/2506.09612v3)**

> **作者:** Mingxiao Li; Mang Ning; Marie-Francine Moens
>
> **备注:** 17 pages, 9. figures
>
> **摘要:** Text-to-image generation models have made significant progress in producing high-quality images from textual descriptions, yet they continue to struggle with maintaining subject consistency across multiple images, a fundamental requirement for visual storytelling. Existing methods attempt to address this by either fine-tuning models on large-scale story visualization datasets, which is resource-intensive, or by using training-free techniques that share information across generations, which still yield limited success. In this paper, we introduce a novel training-free sampling strategy called Zigzag Sampling with Asymmetric Prompts and Visual Sharing to enhance subject consistency in visual story generation. Our approach proposes a zigzag sampling mechanism that alternates between asymmetric prompting to retain subject characteristics, while a visual sharing module transfers visual cues across generated images to %further enforce consistency. Experimental results, based on both quantitative metrics and qualitative evaluations, demonstrate that our method significantly outperforms previous approaches in generating coherent and consistent visual stories. The code is available at https://github.com/Mingxiao-Li/Asymmetry-Zigzag-StoryDiffusion.
>
---
#### [replaced 002] HAPI: A Model for Learning Robot Facial Expressions from Human Preferences
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.17046v2](http://arxiv.org/pdf/2503.17046v2)**

> **作者:** Dongsheng Yang; Qianying Liu; Wataru Sato; Takashi Minato; Chaoran Liu; Shin'ya Nishida
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Automatic robotic facial expression generation is crucial for human-robot interaction, as handcrafted methods based on fixed joint configurations often yield rigid and unnatural behaviors. Although recent automated techniques reduce the need for manual tuning, they tend to fall short by not adequately bridging the gap between human preferences and model predictions-resulting in a deficiency of nuanced and realistic expressions due to limited degrees of freedom and insufficient perceptual integration. In this work, we propose a novel learning-to-rank framework that leverages human feedback to address this discrepancy and enhanced the expressiveness of robotic faces. Specifically, we conduct pairwise comparison annotations to collect human preference data and develop the Human Affective Pairwise Impressions (HAPI) model, a Siamese RankNet-based approach that refines expression evaluation. Results obtained via Bayesian Optimization and online expression survey on a 35-DOF android platform demonstrate that our approach produces significantly more realistic and socially resonant expressions of Anger, Happiness, and Surprise than those generated by baseline and expert-designed methods. This confirms that our framework effectively bridges the gap between human preferences and model predictions while robustly aligning robotic expression generation with human affective responses.
>
---
#### [replaced 003] Text-Aware Image Restoration with Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09993v2](http://arxiv.org/pdf/2506.09993v2)**

> **作者:** Jaewon Min; Jin Hyeon Kim; Paul Hyunbin Cho; Jaeeun Lee; Jihye Park; Minkyu Park; Sangpil Kim; Hyunhee Park; Seungryong Kim
>
> **备注:** Project page: https://cvlab-kaist.github.io/TAIR/
>
> **摘要:** Image restoration aims to recover degraded images. However, existing diffusion-based restoration methods, despite great success in natural image restoration, often struggle to faithfully reconstruct textual regions in degraded images. Those methods frequently generate plausible but incorrect text-like patterns, a phenomenon we refer to as text-image hallucination. In this paper, we introduce Text-Aware Image Restoration (TAIR), a novel restoration task that requires the simultaneous recovery of visual contents and textual fidelity. To tackle this task, we present SA-Text, a large-scale benchmark of 100K high-quality scene images densely annotated with diverse and complex text instances. Furthermore, we propose a multi-task diffusion framework, called TeReDiff, that integrates internal features from diffusion models into a text-spotting module, enabling both components to benefit from joint training. This allows for the extraction of rich text representations, which are utilized as prompts in subsequent denoising steps. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art restoration methods, achieving significant gains in text recognition accuracy. See our project page: https://cvlab-kaist.github.io/TAIR/
>
---
#### [replaced 004] HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01737v2](http://arxiv.org/pdf/2507.01737v2)**

> **作者:** Lin Wu; Zhixiang Chen; Jianglin Lan
>
> **备注:** preprint
>
> **摘要:** Generating realistic 3D human-object interactions (HOIs) remains a challenging task due to the difficulty of modeling detailed interaction dynamics. Existing methods treat human and object motions independently, resulting in physically implausible and causally inconsistent behaviors. In this work, we present HOI-Dyn, a novel framework that formulates HOI generation as a driver-responder system, where human actions drive object responses. At the core of our method is a lightweight transformer-based interaction dynamics model that explicitly predicts how objects should react to human motion. To further enforce consistency, we introduce a residual-based dynamics loss that mitigates the impact of dynamics prediction errors and prevents misleading optimization signals. The dynamics model is used only during training, preserving inference efficiency. Through extensive qualitative and quantitative experiments, we demonstrate that our approach not only enhances the quality of HOI generation but also establishes a feasible metric for evaluating the quality of generated interactions.
>
---
#### [replaced 005] Towards Universal & Efficient Model Compression via Exponential Torque Pruning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22015v3](http://arxiv.org/pdf/2506.22015v3)**

> **作者:** Sarthak Ketanbhai Modi; Zi Pong Lim; Shourya Kuchhal; Yushi Cao; Yupeng Cheng; Yon Shin Teo; Shang-Wei Lin; Zhiming Li
>
> **摘要:** The rapid growth in complexity and size of modern deep neural networks (DNNs) has increased challenges related to computational costs and memory usage, spurring a growing interest in efficient model compression techniques. Previous state-of-the-art approach proposes using a Torque-inspired regularization which forces the weights of neural modules around a selected pivot point. Whereas, we observe that the pruning effect of this approach is far from perfect, as the post-trained network is still dense and also suffers from high accuracy drop. In this work, we attribute such ineffectiveness to the default linear force application scheme, which imposes inappropriate force on neural module of different distances. To efficiently prune the redundant and distant modules while retaining those that are close and necessary for effective inference, in this work, we propose Exponential Torque Pruning (ETP), which adopts an exponential force application scheme for regularization. Experimental results on a broad range of domains demonstrate that, though being extremely simple, ETP manages to achieve significantly higher compression rate than the previous state-of-the-art pruning strategies with negligible accuracy drop.
>
---
#### [replaced 006] CMD-HAR: Cross-Modal Disentanglement for Wearable Human Activity Recognition
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21843v2](http://arxiv.org/pdf/2503.21843v2)**

> **作者:** Hanyu Liu; Siyao Li; Ying Yu; Yixuan Jiang; Hang Xiao; Jingxi Long; Haotian Tang; Chao Li
>
> **摘要:** Human Activity Recognition (HAR) is a fundamental technology for numerous human - centered intelligent applications. Although deep learning methods have been utilized to accelerate feature extraction, issues such as multimodal data mixing, activity heterogeneity, and complex model deployment remain largely unresolved. The aim of this paper is to address issues such as multimodal data mixing, activity heterogeneity, and complex model deployment in sensor-based human activity recognition. We propose a spatiotemporal attention modal decomposition alignment fusion strategy to tackle the problem of the mixed distribution of sensor data. Key discriminative features of activities are captured through cross-modal spatio-temporal disentangled representation, and gradient modulation is combined to alleviate data heterogeneity. In addition, a wearable deployment simulation system is constructed. We conducted experiments on a large number of public datasets, demonstrating the effectiveness of the model.
>
---
#### [replaced 007] SURE-VQA: Systematic Understanding of Robustness Evaluation in Medical VQA Tasks
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19688v3](http://arxiv.org/pdf/2411.19688v3)**

> **作者:** Kim-Celine Kahl; Selen Erkan; Jeremias Traub; Carsten T. Lüth; Klaus Maier-Hein; Lena Maier-Hein; Paul F. Jaeger
>
> **备注:** TMLR 07/2025
>
> **摘要:** Vision-Language Models (VLMs) have great potential in medical tasks, like Visual Question Answering (VQA), where they could act as interactive assistants for both patients and clinicians. Yet their robustness to distribution shifts on unseen data remains a key concern for safe deployment. Evaluating such robustness requires a controlled experimental setup that allows for systematic insights into the model's behavior. However, we demonstrate that current setups fail to offer sufficiently thorough evaluations. To address this gap, we introduce a novel framework, called SURE-VQA, centered around three key requirements to overcome current pitfalls and systematically analyze VLM robustness: 1) Since robustness on synthetic shifts does not necessarily translate to real-world shifts, it should be measured on real-world shifts that are inherent to the VQA data; 2) Traditional token-matching metrics often fail to capture underlying semantics, necessitating the use of large language models (LLMs) for more accurate semantic evaluation; 3) Model performance often lacks interpretability due to missing sanity baselines, thus meaningful baselines should be reported that allow assessing the multimodal impact on the VLM. To demonstrate the relevance of this framework, we conduct a study on the robustness of various Fine-Tuning (FT) methods across three medical datasets with four types of distribution shifts. Our study highlights key insights into robustness: 1) No FT method consistently outperforms others in robustness, and 2) robustness trends are more stable across FT methods than across distribution shifts. Additionally, we find that simple sanity baselines that do not use the image data can perform surprisingly well and confirm LoRA as the best-performing FT method on in-distribution data. Code is provided at https://github.com/IML-DKFZ/sure-vqa.
>
---
#### [replaced 008] Understanding-informed Bias Mitigation for Fair CMR Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17089v2](http://arxiv.org/pdf/2503.17089v2)**

> **作者:** Tiarna Lee; Esther Puyol-Antón; Bram Ruijsink; Pier-Giorgio Masci; Louise Keehn; Phil Chowienczyk; Emily Haseler; Miaojing Shi; Andrew P. King
>
> **摘要:** Artificial intelligence (AI) is increasingly being used for medical imaging tasks. However, there can be biases in AI models, particularly when they are trained using imbalanced training datasets. One such example has been the strong ethnicity bias effect in cardiac magnetic resonance (CMR) image segmentation models. Although this phenomenon has been reported in a number of publications, little is known about the effectiveness of bias mitigation algorithms in this domain. We aim to investigate the impact of common bias mitigation methods to address bias between Black and White subjects in AI-based CMR segmentation models. Specifically, we use oversampling, importance reweighing and Group DRO as well as combinations of these techniques to mitigate the ethnicity bias. Second, motivated by recent findings on the root causes of AI-based CMR segmentation bias, we evaluate the same methods using models trained and evaluated on cropped CMR images. We find that bias can be mitigated using oversampling, significantly improving performance for the underrepresented Black subjects whilst not significantly reducing the majority White subjects' performance. Using cropped images increases performance for both ethnicities and reduces the bias, whilst adding oversampling as a bias mitigation technique with cropped images reduces the bias further. When testing the models on an external clinical validation set, we find high segmentation performance and no statistically significant bias.
>
---
#### [replaced 009] RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22800v2](http://arxiv.org/pdf/2506.22800v2)**

> **作者:** Sicong Du; Jiarun Liu; Qifeng Chen; Hao-Xiang Chen; Tai-Jiang Mu; Sheng Yang
>
> **摘要:** A single-pass driving clip frequently results in incomplete scanning of the road structure, making reconstructed scene expanding a critical requirement for sensor simulators to effectively regress driving actions. Although contemporary 3D Gaussian Splatting (3DGS) techniques achieve remarkable reconstruction quality, their direct extension through the integration of diffusion priors often introduces cumulative physical inconsistencies and compromises training efficiency. To address these limitations, we present RGE-GS, a novel expansive reconstruction framework that synergizes diffusion-based generation with reward-guided Gaussian integration. The RGE-GS framework incorporates two key innovations: First, we propose a reward network that learns to identify and prioritize consistently generated patterns prior to reconstruction phases, thereby enabling selective retention of diffusion outputs for spatial stability. Second, during the reconstruction process, we devise a differentiated training strategy that automatically adjust Gaussian optimization progress according to scene converge metrics, which achieving better convergence than baseline methods. Extensive evaluations of publicly available datasets demonstrate that RGE-GS achieves state-of-the-art performance in reconstruction quality. Our source-code will be made publicly available at https://github.com/CN-ADLab/RGE-GS.
>
---
#### [replaced 010] TiCoSS: Tightening the Coupling between Semantic Segmentation and Stereo Matching within A Joint Learning Framework
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2407.18038v4](http://arxiv.org/pdf/2407.18038v4)**

> **作者:** Guanfeng Tang; Zhiyuan Wu; Jiahang Li; Ping Zhong; We Ye; Xieyuanli Chen; Huiming Lu; Rui Fan
>
> **摘要:** Semantic segmentation and stereo matching, respectively analogous to the ventral and dorsal streams in our human brain, are two key components of autonomous driving perception systems. Addressing these two tasks with separate networks is no longer the mainstream direction in developing computer vision algorithms, particularly with the recent advances in large vision models and embodied artificial intelligence. The trend is shifting towards combining them within a joint learning framework, especially emphasizing feature sharing between the two tasks. The major contributions of this study lie in comprehensively tightening the coupling between semantic segmentation and stereo matching. Specifically, this study introduces three novelties: (1) a tightly coupled, gated feature fusion strategy, (2) a hierarchical deep supervision strategy, and (3) a coupling tightening loss function. The combined use of these technical contributions results in TiCoSS, a state-of-the-art joint learning framework that simultaneously tackles semantic segmentation and stereo matching. Through extensive experiments on the KITTI and vKITTI2 datasets, along with qualitative and quantitative analyses, we validate the effectiveness of our developed strategies and loss function, and demonstrate its superior performance compared to prior arts, with a notable increase in mIoU by over 9%. Our source code will be publicly available at mias.group/TiCoSS upon publication.
>
---
#### [replaced 011] Anatomical Foundation Models for Brain MRIs
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; 68T07; I.2.6**

- **链接: [http://arxiv.org/pdf/2408.07079v4](http://arxiv.org/pdf/2408.07079v4)**

> **作者:** Carlo Alberto Barbano; Matteo Brunello; Benoit Dufumier; Marco Grangetto
>
> **备注:** Updated version; added ablation study
>
> **摘要:** Deep Learning (DL) in neuroimaging has become increasingly relevant for detecting neurological conditions and neurodegenerative disorders. One of the most predominant biomarkers in neuroimaging is represented by brain age, which has been shown to be a good indicator for different conditions, such as Alzheimer's Disease. Using brain age for weakly supervised pre-training of DL models in transfer learning settings has also recently shown promising results, especially when dealing with data scarcity of different conditions. On the other hand, anatomical information of brain MRIs (e.g. cortical thickness) can provide important information for learning good representations that can be transferred to many downstream tasks. In this work, we propose AnatCL, an anatomical foundation model for brain MRIs that i.) leverages anatomical information in a weakly contrastive learning approach, and ii.) achieves state-of-the-art performances across many different downstream tasks. To validate our approach we consider 12 different downstream tasks for the diagnosis of different conditions such as Alzheimer's Disease, autism spectrum disorder, and schizophrenia. Furthermore, we also target the prediction of 10 different clinical assessment scores using structural MRI data. Our findings show that incorporating anatomical information during pre-training leads to more robust and generalizable representations. Pre-trained models can be found at: https://github.com/EIDOSLAB/AnatCL.
>
---
#### [replaced 012] Towards autonomous photogrammetric forest inventory using a lightweight under-canopy robotic drone
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12073v2](http://arxiv.org/pdf/2501.12073v2)**

> **作者:** Väinö Karjalainen; Niko Koivumäki; Teemu Hakala; Jesse Muhojoki; Eric Hyyppä; Anand George; Juha Suomalainen; Eija Honkavaara
>
> **备注:** 36 pages, 11 Figures
>
> **摘要:** Drones are increasingly used in forestry to capture high-resolution remote sensing data, supporting enhanced monitoring, assessment, and decision-making processes. While operations above the forest canopy are already highly automated, flying inside forests remains challenging, primarily relying on manual piloting. Inside dense forests, reliance on the Global Navigation Satellite System (GNSS) for localization is not feasible. Additionally, the drone must autonomously adjust its flight path to avoid collisions. Recently, advancements in robotics have enabled autonomous drone flights in GNSS-denied obstacle-rich areas. In this article, a step towards autonomous forest data collection is taken by building a prototype of a robotic under-canopy drone utilizing state-of-the-art open-source methods and validating its performance for data collection inside forests. Specifically, the study focused on camera-based autonomous flight under the forest canopy and photogrammetric post-processing of the data collected with the low-cost onboard stereo camera. The autonomous flight capability of the prototype was evaluated through multiple test flights at boreal forests. The tree parameter estimation capability was studied by performing diameter at breast height (DBH) estimation. The prototype successfully carried out flights in selected challenging forest environments, and the experiments showed excellent performance in forest 3D modeling with a miniaturized stereoscopic photogrammetric system. The DBH estimation achieved a root mean square error (RMSE) of 3.33 cm (12.79 \%) across all trees. For trees with a DBH less than 30 cm, the RMSE was 1.16 cm (5.74 \%). The results provide valuable insights into autonomous under-canopy forest mapping and highlight the critical next steps for advancing lightweight robotic drone systems for mapping complex forest environments.
>
---
#### [replaced 013] Uncertainty-Guided Coarse-to-Fine Tumor Segmentation with Anatomy-Aware Post-Processing
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.12215v2](http://arxiv.org/pdf/2504.12215v2)**

> **作者:** Ilkin Sevgi Isler; David Mohaisen; Curtis Lisle; Damla Turgut; Ulas Bagci
>
> **备注:** 6 pages, 2 figures, to appear in IEEE ADSCA 2025
>
> **摘要:** Reliable tumor segmentation in thoracic computed tomography (CT) remains challenging due to boundary ambiguity, class imbalance, and anatomical variability. We propose an uncertainty-guided, coarse-to-fine segmentation framework that combines full-volume tumor localization with refined region-of-interest (ROI) segmentation, enhanced by anatomically aware post-processing. The first-stage model generates a coarse prediction, followed by anatomically informed filtering based on lung overlap, proximity to lung surfaces, and component size. The resulting ROIs are segmented by a second-stage model trained with uncertainty-aware loss functions to improve accuracy and boundary calibration in ambiguous regions. Experiments on private and public datasets demonstrate improvements in Dice and Hausdorff scores, with fewer false positives and enhanced spatial interpretability. These results highlight the value of combining uncertainty modeling and anatomical priors in cascaded segmentation pipelines for robust and clinically meaningful tumor delineation. On the Orlando dataset, our framework improved Swin UNETR Dice from 0.4690 to 0.6447. Reduction in spurious components was strongly correlated with segmentation gains, underscoring the value of anatomically informed post-processing.
>
---
#### [replaced 014] Evaluating Robustness of Monocular Depth Estimation with Procedural Scene Perturbations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00981v2](http://arxiv.org/pdf/2507.00981v2)**

> **作者:** Jack Nugent; Siyang Wu; Zeyu Ma; Beining Han; Meenal Parakh; Abhishek Joshi; Lingjie Mei; Alexander Raistrick; Xinyuan Li; Jia Deng
>
> **备注:** Fixing display of figure on Safari browsers
>
> **摘要:** Recent years have witnessed substantial progress on monocular depth estimation, particularly as measured by the success of large models on standard benchmarks. However, performance on standard benchmarks does not offer a complete assessment, because most evaluate accuracy but not robustness. In this work, we introduce PDE (Procedural Depth Evaluation), a new benchmark which enables systematic robustness evaluation. PDE uses procedural generation to create 3D scenes that test robustness to various controlled perturbations, including object, camera, material and lighting changes. Our analysis yields interesting findings on what perturbations are challenging for state-of-the-art depth models, which we hope will inform further research. Code and data are available at https://github.com/princeton-vl/proc-depth-eval.
>
---
#### [replaced 015] Privacy-Preserving in Connected and Autonomous Vehicles Through Vision to Text Transformation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15854v2](http://arxiv.org/pdf/2506.15854v2)**

> **作者:** Abdolazim Rezaei; Mehdi Sookhak; Ahmad Patooghy
>
> **摘要:** Connected and Autonomous Vehicles (CAVs) rely on a range of devices that often process privacy-sensitive data. Among these, roadside units play a critical role particularly through the use of AI-equipped (AIE) cameras for applications such as violation detection. However, the privacy risks associated with captured imagery remain a major concern, as such data can be misused for identity theft, profiling, or unauthorized commercial purposes. While traditional techniques such as face blurring and obfuscation have been applied to mitigate privacy risks, individual privacy remains at risk, as individuals can still be tracked using other features such as their clothing. This paper introduces a novel privacy-preserving framework that leverages feedback-based reinforcement learning (RL) and vision-language models (VLMs) to protect sensitive visual information captured by AIE cameras. The main idea is to convert images into semantically equivalent textual descriptions, ensuring that scene-relevant information is retained while visual privacy is preserved. A hierarchical RL strategy is employed to iteratively refine the generated text, enhancing both semantic accuracy and privacy. Evaluation results demonstrate significant improvements in both privacy protection and textual quality, with the Unique Word Count increasing by approximately 77\% and Detail Density by around 50\% compared to existing approaches.
>
---
#### [replaced 016] Deep Transfer Learning for Kidney Cancer Diagnosis
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.04318v2](http://arxiv.org/pdf/2408.04318v2)**

> **作者:** Yassine Habchi; Hamza Kheddar; Yassine Himeur; Mohamed Chahine Ghanem; Abdelkrim Boukabou; Shadi Atalla; Wathiq Mansoor; Hussain Al-Ahmad
>
> **摘要:** Incurable diseases continue to pose major challenges to global healthcare systems, with their prevalence shaped by lifestyle, economic, social, and genetic factors. Among these, kidney disease remains a critical global health issue, requiring ongoing research to improve early diagnosis and treatment. In recent years, deep learning (DL) has shown promise in medical imaging and diagnostics, driving significant progress in automatic kidney cancer (KC) detection. However, the success of DL models depends heavily on the availability of high-quality, domain-specific datasets, which are often limited and expensive to acquire. Moreover, DL models demand substantial computational power and storage, restricting their real-world clinical use. To overcome these barriers, transfer learning (TL) has emerged as an effective approach, enabling the reuse of pre-trained models from related domains to enhance KC diagnosis. This paper presents a comprehensive survey of DL-based TL frameworks for KC detection, systematically reviewing key methodologies, their advantages, and limitations, and analyzing their practical performance. It further discusses challenges in applying TL to medical imaging and highlights emerging trends that could influence future research. This review demonstrates the transformative role of TL in precision medicine, particularly oncology, by improving diagnostic accuracy, lowering computational demands, and supporting the integration of AI-powered tools in healthcare. The insights provided offer valuable guidance for researchers and practitioners, paving the way for future advances in KC diagnostics and personalized treatment strategies.
>
---
#### [replaced 017] Assessing the Uncertainty and Robustness of the Laptop Refurbishing Software
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.03782v2](http://arxiv.org/pdf/2409.03782v2)**

> **作者:** Chengjie Lu; Jiahui Wu; Shaukat Ali; Mikkel Labori Olsen
>
> **备注:** 17 pages, 6 figures, 4 tables
>
> **摘要:** Refurbishing laptops extends their lives while contributing to reducing electronic waste, which promotes building a sustainable future. To this end, the Danish Technological Institute (DTI) focuses on the research and development of several robotic applications empowered with software, including laptop refurbishing. Cleaning represents a major step in refurbishing and involves identifying and removing stickers from laptop surfaces. Software plays a crucial role in the cleaning process. For instance, the software integrates various object detection models to identify and remove stickers from laptops automatically. However, given the diversity in types of stickers (e.g., shapes, colors, locations), identification of the stickers is highly uncertain, thereby requiring explicit quantification of uncertainty associated with the identified stickers. Such uncertainty quantification can help reduce risks in removing stickers, which, for example, could otherwise result in software faults damaging laptop surfaces. For uncertainty quantification, we adopted the Monte Carlo Dropout method to evaluate six sticker detection models (SDMs) from DTI using three datasets: the original image dataset from DTI and two datasets generated with vision language models, i.e., DALL-E-3 and Stable Diffusion-3. In addition, we presented novel robustness metrics concerning detection accuracy and uncertainty to assess the robustness of the SDMs based on adversarial datasets generated from the three datasets using a dense adversary method. Our evaluation results show that different SDMs perform differently regarding different metrics. Based on the results, we provide SDM selection guidelines and lessons learned from various perspectives.
>
---
#### [replaced 018] Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15075v2](http://arxiv.org/pdf/2505.15075v2)**

> **作者:** Hao Wang; Pinzhi Huang; Jihan Yang; Saining Xie; Daisuke Kawahara
>
> **备注:** https://github.com/nlp-waseda/traveling-across-languages
>
> **摘要:** The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models.
>
---
#### [replaced 019] DeltaEdit: Exploring Text-free Training for Text-Driven Image Manipulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2303.06285v2](http://arxiv.org/pdf/2303.06285v2)**

> **作者:** Yueming Lyu; Tianwei Lin; Fu Li; Dongliang He; Jing Dong; Tieniu Tan
>
> **备注:** Code is available at https://github.com/Yueming6568/DeltaEdit
>
> **摘要:** Text-driven image manipulation remains challenging in training or inference flexibility. Conditional generative models depend heavily on expensive annotated training data. Meanwhile, recent frameworks, which leverage pre-trained vision-language models, are limited by either per text-prompt optimization or inference-time hyper-parameters tuning. In this work, we propose a novel framework named \textit{DeltaEdit} to address these problems. Our key idea is to investigate and identify a space, namely delta image and text space that has well-aligned distribution between CLIP visual feature differences of two images and CLIP textual embedding differences of source and target texts. Based on the CLIP delta space, the DeltaEdit network is designed to map the CLIP visual features differences to the editing directions of StyleGAN at training phase. Then, in inference phase, DeltaEdit predicts the StyleGAN's editing directions from the differences of the CLIP textual features. In this way, DeltaEdit is trained in a text-free manner. Once trained, it can well generalize to various text prompts for zero-shot inference without bells and whistles.
>
---
#### [replaced 020] The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05673v3](http://arxiv.org/pdf/2502.05673v3)**

> **作者:** Ping Liu; Jiawei Du
>
> **备注:** Dr. Jiawei Du is the corresponding author
>
> **摘要:** Dataset distillation, which condenses large-scale datasets into compact synthetic representations, has emerged as a critical solution for training modern deep learning models efficiently. While prior surveys focus on developments before 2023, this work comprehensively reviews recent advances, emphasizing scalability to large-scale datasets such as ImageNet-1K and ImageNet-21K. We categorize progress into a few key methodologies: trajectory matching, gradient matching, distribution matching, scalable generative approaches, and decoupling optimization mechanisms. As a comprehensive examination of recent dataset distillation advances, this survey highlights breakthrough innovations: the SRe2L framework for efficient and effective condensation, soft label strategies that significantly enhance model accuracy, and lossless distillation techniques that maximize compression while maintaining performance. Beyond these methodological advancements, we address critical challenges, including robustness against adversarial and backdoor attacks, effective handling of non-IID data distributions. Additionally, we explore emerging applications in video and audio processing, multi-modal learning, medical imaging, and scientific computing, highlighting its domain versatility. By offering extensive performance comparisons and actionable research directions, this survey equips researchers and practitioners with practical insights to advance efficient and generalizable dataset distillation, paving the way for future innovations.
>
---
#### [replaced 021] Non-rigid Motion Correction for MRI Reconstruction via Coarse-To-Fine Diffusion Models
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15057v3](http://arxiv.org/pdf/2505.15057v3)**

> **作者:** Frederic Wang; Jonathan I. Tamir
>
> **备注:** ICIP 2025
>
> **摘要:** Magnetic Resonance Imaging (MRI) is highly susceptible to motion artifacts due to the extended acquisition times required for k-space sampling. These artifacts can compromise diagnostic utility, particularly for dynamic imaging. We propose a novel alternating minimization framework that leverages a bespoke diffusion model to jointly reconstruct and correct non-rigid motion-corrupted k-space data. The diffusion model uses a coarse-to-fine denoising strategy to capture large overall motion and reconstruct the lower frequencies of the image first, providing a better inductive bias for motion estimation than that of standard diffusion models. We demonstrate the performance of our approach on both real-world cine cardiac MRI datasets and complex simulated rigid and non-rigid deformations, even when each motion state is undersampled by a factor of 64x. Additionally, our method is agnostic to sampling patterns, anatomical variations, and MRI scanning protocols, as long as some low frequency components are sampled during each motion state.
>
---
#### [replaced 022] AirV2X: Unified Air-Ground Vehicle-to-Everything Collaboration
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19283v3](http://arxiv.org/pdf/2506.19283v3)**

> **作者:** Xiangbo Gao; Yuheng Wu; Fengze Yang; Xuewen Luo; Keshu Wu; Xinghao Chen; Yuping Wang; Chenxi Liu; Yang Zhou; Zhengzhong Tu
>
> **摘要:** While multi-vehicular collaborative driving demonstrates clear advantages over single-vehicle autonomy, traditional infrastructure-based V2X systems remain constrained by substantial deployment costs and the creation of "uncovered danger zones" in rural and suburban areas. We present AirV2X-Perception, a large-scale dataset that leverages Unmanned Aerial Vehicles (UAVs) as a flexible alternative or complement to fixed Road-Side Units (RSUs). Drones offer unique advantages over ground-based perception: complementary bird's-eye-views that reduce occlusions, dynamic positioning capabilities that enable hovering, patrolling, and escorting navigation rules, and significantly lower deployment costs compared to fixed infrastructure. Our dataset comprises 6.73 hours of drone-assisted driving scenarios across urban, suburban, and rural environments with varied weather and lighting conditions. The AirV2X-Perception dataset facilitates the development and standardized evaluation of Vehicle-to-Drone (V2D) algorithms, addressing a critical gap in the rapidly expanding field of aerial-assisted autonomous driving systems. The dataset and development kits are open-sourced at https://github.com/taco-group/AirV2X-Perception.
>
---
#### [replaced 023] CAD-Editor: A Locate-then-Infill Framework with Automated Training Data Synthesis for Text-Based CAD Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03997v2](http://arxiv.org/pdf/2502.03997v2)**

> **作者:** Yu Yuan; Shizhao Sun; Qi Liu; Jiang Bian
>
> **摘要:** Computer Aided Design (CAD) is indispensable across various industries. \emph{Text-based CAD editing}, which automates the modification of CAD models based on textual instructions, holds great potential but remains underexplored. Existing methods primarily focus on design variation generation or text-based CAD generation, either lacking support for text-based control or neglecting existing CAD models as constraints. We introduce \emph{CAD-Editor}, the first framework for text-based CAD editing. To address the challenge of demanding triplet data with accurate correspondence for training, we propose an automated data synthesis pipeline. This pipeline utilizes design variation models to generate pairs of original and edited CAD models and employs Large Vision-Language Models (LVLMs) to summarize their differences into editing instructions. To tackle the composite nature of text-based CAD editing, we propose a locate-then-infill framework that decomposes the task into two focused sub-tasks: locating regions requiring modification and infilling these regions with appropriate edits. Large Language Models (LLMs) serve as the backbone for both sub-tasks, leveraging their capabilities in natural language understanding and CAD knowledge. Experiments show that CAD-Editor achieves superior performance both quantitatively and qualitatively. The code is available at \url {https://github.com/microsoft/CAD-Editor}.
>
---
#### [replaced 024] Task-Adapter++: Task-specific Adaptation with Order-aware Alignment for Few-shot Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06002v2](http://arxiv.org/pdf/2505.06002v2)**

> **作者:** Congqi Cao; Peiheng Han; Yueran zhang; Yating Yu; Qinyi Lv; Lingtong Min; Yanning zhang
>
> **备注:** extended work of Task-Adapter
>
> **摘要:** Large-scale pre-trained models have achieved remarkable success in language and image tasks, leading an increasing number of studies to explore the application of pre-trained image models, such as CLIP, in the domain of few-shot action recognition (FSAR). However, current methods generally suffer from several problems: 1) Direct fine-tuning often undermines the generalization capability of the pre-trained model; 2) The exploration of task-specific information is insufficient in the visual tasks; 3) The semantic order information is typically overlooked during text modeling; 4) Existing cross-modal alignment techniques ignore the temporal coupling of multimodal information. To address these, we propose Task-Adapter++, a parameter-efficient dual adaptation method for both image and text encoders. Specifically, to make full use of the variations across different few-shot learning tasks, we design a task-specific adaptation for the image encoder so that the most discriminative information can be well noticed during feature extraction. Furthermore, we leverage large language models (LLMs) to generate detailed sequential sub-action descriptions for each action class, and introduce semantic order adapters into the text encoder to effectively model the sequential relationships between these sub-actions. Finally, we develop an innovative fine-grained cross-modal alignment strategy that actively maps visual features to reside in the same temporal stage as semantic descriptions. Extensive experiments fully demonstrate the effectiveness and superiority of the proposed method, which achieves state-of-the-art performance on 5 benchmarks consistently. The code is open-sourced at https://github.com/Jaulin-Bage/Task-Adapter-pp.
>
---
#### [replaced 025] LLaVA-SP: Enhancing Visual Representation with Visual Spatial Tokens for MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00505v2](http://arxiv.org/pdf/2507.00505v2)**

> **作者:** Haoran Lou; Chunxiao Fan; Ziyan Liu; Yuexin Wu; Xinliang Wang
>
> **备注:** ICCV
>
> **摘要:** The architecture of multimodal large language models (MLLMs) commonly connects a vision encoder, often based on CLIP-ViT, to a large language model. While CLIP-ViT works well for capturing global image features, it struggles to model local relationships between adjacent patches, leading to weaker visual representation, which in turn affects the detailed understanding ability of MLLMs. To solve this, we propose LLaVA-SP, which \textbf{ only adds six spatial visual tokens} to the original visual tokens to enhance the visual representation. Our approach offers three key advantages: 1)We propose a novel Projector, which uses convolutional kernels to derive visual spatial tokens from ViT patch features, simulating two visual spatial ordering approaches: ``from central region to global" and ``from abstract to specific". Then, a cross-attention mechanism is applied to fuse fine-grained visual information, enriching the overall visual representation. 2) We present two model variants: LLaVA-SP-Cropping, which focuses on detail features through progressive cropping, and LLaVA-SP-Pooling, which captures global semantics through adaptive pooling, enabling the model to handle diverse visual understanding tasks. 3) Extensive experiments show that LLaVA-SP, fine-tuned with LoRA, achieves significant performance improvements across various multimodal benchmarks, outperforming the state-of-the-art LLaVA-1.5 model in multiple tasks with nearly identical inference latency. The code and models are available at https://github.com/CnFaker/LLaVA-SP.
>
---
#### [replaced 026] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16765v3](http://arxiv.org/pdf/2411.16765v3)**

> **作者:** Shester Gueuwou; Xiaodan Du; Greg Shakhnarovich; Karen Livescu; Alexander H. Liu
>
> **备注:** Fixed Figure 1. ACL 2025
>
> **摘要:** Sign language processing has traditionally relied on task-specific models, limiting the potential for transfer learning across tasks. Pre-training methods for sign language have typically focused on either supervised pre-training, which cannot take advantage of unlabeled data, or context-independent (frame or video segment) representations, which ignore the effects of relationships across time in sign language. We introduce SHuBERT (Sign Hidden-Unit BERT), a self-supervised contextual representation model learned from approximately 1,000 hours of American Sign Language video. SHuBERT adapts masked token prediction objectives to multi-stream visual sign language input, learning to predict multiple targets corresponding to clustered hand, face, and body pose streams. SHuBERT achieves state-of-the-art performance across multiple tasks including sign language translation, isolated sign language recognition, and fingerspelling detection.
>
---
#### [replaced 027] Enhancing Fetal Plane Classification Accuracy with Data Augmentation Using Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.15248v2](http://arxiv.org/pdf/2501.15248v2)**

> **作者:** Yueying Tian; Elif Ucurum; Xudong Han; Rupert Young; Chris Chatwin; Philip Birch
>
> **摘要:** Ultrasound imaging is widely used in medical diagnosis, especially for fetal health assessment. However, the availability of high-quality annotated ultrasound images is limited, which restricts the training of machine learning models. In this paper, we investigate the use of diffusion models to generate synthetic ultrasound images to improve the performance on fetal plane classification. We train different classifiers first on synthetic images and then fine-tune them with real images. Extensive experimental results demonstrate that incorporating generated images into training pipelines leads to better classification accuracy than training with real images alone. The findings suggest that generating synthetic data using diffusion models can be a valuable tool in overcoming the challenges of data scarcity in ultrasound medical imaging.
>
---
#### [replaced 028] Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18248v2](http://arxiv.org/pdf/2506.18248v2)**

> **作者:** Jongoh Jeong; Hunmin Yang; Jaeseok Jeong; Kuk-Jin Yoon
>
> **摘要:** Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).
>
---
#### [replaced 029] Diffusion-Based Generative Models for 3D Occupancy Prediction in Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23115v2](http://arxiv.org/pdf/2505.23115v2)**

> **作者:** Yunshen Wang; Yicheng Liu; Tianyuan Yuan; Yingshi Liang; Xiuyu Yang; Honggang Zhang; Hang Zhao
>
> **备注:** ICRA 2025
>
> **摘要:** Accurately predicting 3D occupancy grids from visual inputs is critical for autonomous driving, but current discriminative methods struggle with noisy data, incomplete observations, and the complex structures inherent in 3D scenes. In this work, we reframe 3D occupancy prediction as a generative modeling task using diffusion models, which learn the underlying data distribution and incorporate 3D scene priors. This approach enhances prediction consistency, noise robustness, and better handles the intricacies of 3D spatial structures. Our extensive experiments show that diffusion-based generative models outperform state-of-the-art discriminative approaches, delivering more realistic and accurate occupancy predictions, especially in occluded or low-visibility regions. Moreover, the improved predictions significantly benefit downstream planning tasks, highlighting the practical advantages of our method for real-world autonomous driving applications.
>
---
#### [replaced 030] Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23918v3](http://arxiv.org/pdf/2506.23918v3)**

> **作者:** Zhaochen Su; Peng Xia; Hangyu Guo; Zhenhua Liu; Yan Ma; Xiaoye Qu; Jiaqi Liu; Yanshu Li; Kaide Zeng; Zhengyuan Yang; Linjie Li; Yu Cheng; Heng Ji; Junxian He; Yi R. Fung
>
> **备注:** Preprint in progress. We maintain a real-time GitHub repository tracking progress at: https://github.com/zhaochen0110/Awesome_Think_With_Images
>
> **摘要:** Recent progress in multimodal reasoning has been significantly advanced by textual Chain-of-Thought (CoT), a paradigm where models conduct reasoning within language. This text-centric approach, however, treats vision as a static, initial context, creating a fundamental "semantic gap" between rich perceptual data and discrete symbolic thought. Human cognition often transcends language, utilizing vision as a dynamic mental sketchpad. A similar evolution is now unfolding in AI, marking a fundamental paradigm shift from models that merely think about images to those that can truly think with images. This emerging paradigm is characterized by models leveraging visual information as intermediate steps in their thought process, transforming vision from a passive input into a dynamic, manipulable cognitive workspace. In this survey, we chart this evolution of intelligence along a trajectory of increasing cognitive autonomy, which unfolds across three key stages: from external tool exploration, through programmatic manipulation, to intrinsic imagination. To structure this rapidly evolving field, our survey makes four key contributions. (1) We establish the foundational principles of the think with image paradigm and its three-stage framework. (2) We provide a comprehensive review of the core methods that characterize each stage of this roadmap. (3) We analyze the critical landscape of evaluation benchmarks and transformative applications. (4) We identify significant challenges and outline promising future directions. By providing this structured overview, we aim to offer a clear roadmap for future research towards more powerful and human-aligned multimodal AI.
>
---
#### [replaced 031] Weakly Supervised Segmentation Framework for Thyroid Nodule Based on High-confidence Labels and High-rationality Losses
- **分类: cs.CV; J.3.3**

- **链接: [http://arxiv.org/pdf/2502.19707v2](http://arxiv.org/pdf/2502.19707v2)**

> **作者:** Jianning Chi; Zelan Li; Geng Lin; MingYang Sun; Xiaosheng Yu
>
> **备注:** 24 pages, 14 figures, 7 tables
>
> **摘要:** Weakly supervised segmentation methods can delineate thyroid nodules in ultrasound images efficiently using training data with coarse labels, but suffer from: 1) low-confidence pseudo-labels that follow topological priors, introducing significant label noise, and 2) low-rationality loss functions that rigidly compare segmentation with labels, ignoring discriminative information for nodules with diverse and complex shapes. To solve these issues, we clarify the objective and references for weakly supervised ultrasound image segmentation, presenting a framework with high-confidence pseudo-labels to represent topological and anatomical information and high-rationality losses to capture multi-level discriminative features. Specifically, we fuse geometric transformations of four-point annotations and MedSAM model results prompted by specific annotations to generate high-confidence box, foreground, and background labels. Our high-rationality learning strategy includes: 1) Alignment loss measuring spatial consistency between segmentation and box label, and topological continuity within the foreground label, guiding the network to perceive nodule location; 2) Contrastive loss pulling features from labeled foreground regions while pushing features from labeled foreground and background regions, guiding the network to learn nodule and background feature distribution; 3) Prototype correlation loss measuring consistency between correlation maps derived by comparing features with foreground and background prototypes, refining uncertain regions to accurate nodule edges. Experimental results show that our method achieves state-of-the-art performance on the TN3K and DDTI datasets. The code is available at https://github.com/bluehenglee/MLI-MSC.
>
---
#### [replaced 032] MTCNet: Motion and Topology Consistency Guided Learning for Mitral Valve Segmentationin 4D Ultrasound
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00660v2](http://arxiv.org/pdf/2507.00660v2)**

> **作者:** Rusi Chen; Yuanting Yang; Jiezhi Yao; Hongning Song; Ji Zhang; Yongsong Zhou; Yuhao Huang; Ronghao Yang; Dan Jia; Yuhan Zhang; Xing Tao; Haoran Dou; Qing Zhou; Xin Yang; Dong Ni
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Mitral regurgitation is one of the most prevalent cardiac disorders. Four-dimensional (4D) ultrasound has emerged as the primary imaging modality for assessing dynamic valvular morphology. However, 4D mitral valve (MV) analysis remains challenging due to limited phase annotations, severe motion artifacts, and poor imaging quality. Yet, the absence of inter-phase dependency in existing methods hinders 4D MV analysis. To bridge this gap, we propose a Motion-Topology guided consistency network (MTCNet) for accurate 4D MV ultrasound segmentation in semi-supervised learning (SSL). MTCNet requires only sparse end-diastolic and end-systolic annotations. First, we design a cross-phase motion-guided consistency learning strategy, utilizing a bi-directional attention memory bank to propagate spatio-temporal features. This enables MTCNet to achieve excellent performance both per- and inter-phase. Second, we devise a novel topology-guided correlation regularization that explores physical prior knowledge to maintain anatomically plausible. Therefore, MTCNet can effectively leverage structural correspondence between labeled and unlabeled phases. Extensive evaluations on the first largest 4D MV dataset, with 1408 phases from 160 patients, show that MTCNet performs superior cross-phase consistency compared to other advanced methods (Dice: 87.30%, HD: 1.75mm). Both the code and the dataset are available at https://github.com/crs524/MTCNet.
>
---
#### [replaced 033] Escaping Platos Cave: JAM for Aligning Independently Trained Vision and Language Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01201v2](http://arxiv.org/pdf/2507.01201v2)**

> **作者:** Hyoseo; Yoon; Yisong Yue; Been Kim
>
> **摘要:** Independently trained vision and language models inhabit disjoint representational spaces, shaped by their respective modalities, objectives, and architectures. Yet an emerging hypothesis - the Platonic Representation Hypothesis - suggests that such models may nonetheless converge toward a shared statistical model of reality. This compatibility, if it exists, raises a fundamental question: can we move beyond post-hoc statistical detection of alignment and explicitly optimize for it between such disjoint representations? We cast this Platonic alignment problem as a multi-objective optimization task - preserve each modality's native structure while aligning for mutual coherence. We introduce the Joint Autoencoder Modulator (JAM) framework that jointly trains modality-specific autoencoders on the latent representations of pre-trained single modality models, encouraging alignment through both reconstruction and cross-modal objectives. By analogy, this framework serves as a method to escape Plato's Cave, enabling the emergence of shared structure from disjoint inputs. We evaluate this framework across three critical design axes: (i) the alignment objective - comparing contrastive loss (Con), its hard-negative variant (NegCon), and our Spread loss, (ii) the layer depth at which alignment is most effective, and (iii) the impact of foundation model scale on representational convergence. Our findings show that our lightweight Pareto-efficient framework reliably induces alignment, even across frozen, independently trained representations, offering both theoretical insight and practical pathways for transforming generalist unimodal foundations into specialist multimodal models.
>
---
#### [replaced 034] ZeroStereo: Zero-Shot Stereo Matching from Single Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08654v3](http://arxiv.org/pdf/2501.08654v3)**

> **作者:** Xianqi Wang; Hao Yang; Gangwei Xu; Junda Cheng; Min Lin; Yong Deng; Jinliang Zang; Yurui Chen; Xin Yang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** State-of-the-art supervised stereo matching methods have achieved remarkable performance on various benchmarks. However, their generalization to real-world scenarios remains challenging due to the scarcity of annotated real-world stereo data. In this paper, we propose ZeroStereo, a novel stereo image generation pipeline for zero-shot stereo matching. Our approach synthesizes high-quality right images from arbitrary single images by leveraging pseudo disparities generated by a monocular depth estimation model. Unlike previous methods that address occluded regions by filling missing areas with neighboring pixels or random backgrounds, we fine-tune a diffusion inpainting model to recover missing details while preserving semantic structure. Additionally, we propose Training-Free Confidence Generation, which mitigates the impact of unreliable pseudo labels without additional training, and Adaptive Disparity Selection, which ensures a diverse and realistic disparity distribution while preventing excessive occlusion and foreground distortion. Experiments demonstrate that models trained with our pipeline achieve state-of-the-art zero-shot generalization across multiple datasets with only a dataset volume comparable to Scene Flow. Code: https://github.com/Windsrain/ZeroStereo.
>
---
#### [replaced 035] Stereo Any Video: Temporally Consistent Stereo Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05549v2](http://arxiv.org/pdf/2503.05549v2)**

> **作者:** Junpeng Jing; Weixun Luo; Ye Mao; Krystian Mikolajczyk
>
> **备注:** Accepted at ICCV2025
>
> **摘要:** This paper introduces Stereo Any Video, a powerful framework for video stereo matching. It can estimate spatially accurate and temporally consistent disparities without relying on auxiliary information such as camera poses or optical flow. The strong capability is driven by rich priors from monocular video depth models, which are integrated with convolutional features to produce stable representations. To further enhance performance, key architectural innovations are introduced: all-to-all-pairs correlation, which constructs smooth and robust matching cost volumes, and temporal convex upsampling, which improves temporal coherence. These components collectively ensure robustness, accuracy, and temporal consistency, setting a new standard in video stereo matching. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple datasets both qualitatively and quantitatively in zero-shot settings, as well as strong generalization to real-world indoor and outdoor scenarios.
>
---
#### [replaced 036] Illuminant and light direction estimation using Wasserstein distance method
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05802v2](http://arxiv.org/pdf/2503.05802v2)**

> **作者:** Selcuk Yazar
>
> **摘要:** Illumination estimation remains a pivotal challenge in image processing, particularly for robotics, where robust environmental perception is essential under varying lighting conditions. Traditional approaches, such as RGB histograms and GIST descriptors, often fail in complex scenarios due to their sensitivity to illumination changes. This study introduces a novel method utilizing the Wasserstein distance, rooted in optimal transport theory, to estimate illuminant and light direction in images. Experiments on diverse images indoor scenes, black-and-white photographs, and night images demonstrate the method's efficacy in detecting dominant light sources and estimating their directions, outperforming traditional statistical methods in complex lighting environments. The approach shows promise for applications in light source localization, image quality assessment, and object detection enhancement. Future research may explore adaptive thresholding and integrate gradient analysis to enhance accuracy, offering a scalable solution for real-world illumination challenges in robotics and beyond.
>
---
#### [replaced 037] Adapter-Enhanced Semantic Prompting for Continual Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.11074v3](http://arxiv.org/pdf/2412.11074v3)**

> **作者:** Baocai Yin; Ji Zhao; Huajie Jiang; Ningning Hou; Yongli Hu; Amin Beheshti; Ming-Hsuan Yang; Yuankai Qi
>
> **备注:** This work has been submitted to the IJCV for possible publication
>
> **摘要:** Continual learning (CL) enables models to adapt to evolving data streams. A major challenge of CL is catastrophic forgetting, where new knowledge will overwrite previously acquired knowledge. Traditional methods usually retain the past data for replay or add additional branches in the model to learn new knowledge, which has high memory requirements. In this paper, we propose a novel lightweight CL framework, Adapter-Enhanced Semantic Prompting (AESP), which integrates prompt tuning and adapter techniques. Specifically, we design semantic-guided prompts to enhance the generalization ability of visual features and utilize adapters to efficiently fuse the semantic information, aiming to learn more adaptive features for the continual learning task. Furthermore, to choose the right task prompt for feature adaptation, we have developed a novel matching mechanism for prompt selection. Extensive experiments on three CL datasets demonstrate that our approach achieves favorable performance across multiple metrics, showing its potential for advancing CL.
>
---
#### [replaced 038] Modality-agnostic, patient-specific digital twins modeling temporally varying digestive motion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01909v2](http://arxiv.org/pdf/2507.01909v2)**

> **作者:** Jorge Tapias Gomez; Nishant Nadkarni; Lando S. Bosma; Jue Jiang; Ergys D. Subashi; William P. Segars; James M. Balter; Mert R Sabuncu; Neelam Tyagi; Harini Veeraraghavan
>
> **备注:** 7 Pages, 6 figures, 4 tables
>
> **摘要:** Objective: Clinical implementation of deformable image registration (DIR) requires voxel-based spatial accuracy metrics such as manually identified landmarks, which are challenging to implement for highly mobile gastrointestinal (GI) organs. To address this, patient-specific digital twins (DT) modeling temporally varying motion were created to assess the accuracy of DIR methods. Approach: 21 motion phases simulating digestive GI motion as 4D sequences were generated from static 3D patient scans using published analytical GI motion models through a semi-automated pipeline. Eleven datasets, including six T2w FSE MRI (T2w MRI), two T1w 4D golden-angle stack-of-stars, and three contrast-enhanced CT scans. The motion amplitudes of the DTs were assessed against real patient stomach motion amplitudes extracted from independent 4D MRI datasets. The generated DTs were then used to assess six different DIR methods using target registration error, Dice similarity coefficient, and the 95th percentile Hausdorff distance using summary metrics and voxel-level granular visualizations. Finally, for a subset of T2w MRI scans from patients treated with MR-guided radiation therapy, dose distributions were warped and accumulated to assess dose warping errors, including evaluations of DIR performance in both low- and high-dose regions for patient-specific error estimation. Main results: Our proposed pipeline synthesized DTs modeling realistic GI motion, achieving mean and maximum motion amplitudes and a mean log Jacobian determinant within 0.8 mm and 0.01, respectively, similar to published real-patient gastric motion data. It also enables the extraction of detailed quantitative DIR performance metrics and rigorous validation of dose mapping accuracy. Significance: The pipeline enables rigorously testing DIR tools for dynamic, anatomically complex regions enabling granular spatial and dosimetric accuracies.
>
---
#### [replaced 039] FeatSharp: Your Vision Model Features, Sharper
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16025v2](http://arxiv.org/pdf/2502.16025v2)**

> **作者:** Mike Ranzinger; Greg Heinrich; Pavlo Molchanov; Jan Kautz; Bryan Catanzaro; Andrew Tao
>
> **备注:** ICML 2025 Version
>
> **摘要:** The feature maps of vision encoders are fundamental to myriad modern AI tasks, ranging from core perception algorithms (e.g. semantic segmentation, object detection, depth perception, etc.) to modern multimodal understanding in vision-language models (VLMs). Currently, in computer vision, the frontier of general purpose vision backbones is Vision Transformers (ViT), typically trained using contrastive loss (e.g. CLIP). A key problem with most off-the-shelf ViTs, particularly CLIP, is that these models are inflexibly low resolution. Most run at $224 \times 224$px, while the "high-resolution" versions are around $378-448$px, but still inflexible. We introduce a novel method to coherently and cheaply upsample the feature maps of low-resolution vision encoders while picking up on fine-grained details that would otherwise be lost due to resolution. We demonstrate the effectiveness of this approach on core perception tasks as well as within agglomerative model training using RADIO as a way of providing richer targets for distillation. Code available at https://github.com/NVlabs/FeatSharp .
>
---
#### [replaced 040] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.16236v3](http://arxiv.org/pdf/2410.16236v3)**

> **作者:** Yuxuan Cai; Jiangning Zhang; Haoyang He; Xinwei He; Ao Tong; Zhenye Gan; Chengjie Wang; Zhucun Xue; Yong Liu; Xiang Bai
>
> **备注:** ICCV'25
>
> **摘要:** The success of Large Language Models (LLMs) has inspired the development of Multimodal Large Language Models (MLLMs) for unified understanding of vision and language. However, the increasing model size and computational complexity of large-scale MLLMs (l-MLLMs) limit their use in resource-constrained scenarios. Although small-scale MLLMs (s-MLLMs) are designed to reduce computational costs, they typically suffer from performance degradation. To mitigate this limitation, we propose a novel LLaVA-KD framework to transfer knowledge from l-MLLMs to s-MLLMs. Specifically, we introduce Multimodal Distillation (MDist) to transfer teacher model's robust representations across both visual and linguistic modalities, and Relation Distillation (RDist) to transfer teacher model's ability to capture visual token relationships. Additionally, we propose a three-stage training scheme to fully exploit the potential of the proposed distillation strategy: 1) Distilled Pre-Training to strengthen the alignment between visual-linguistic representations in s-MLLMs, 2) Supervised Fine-Tuning to equip the s-MLLMs with multimodal understanding capacity, and 3) Distilled Fine-Tuning to refine s-MLLM's knowledge. Our approach significantly improves s-MLLMs performance without altering the model architecture. Extensive experiments and ablation studies validate the effectiveness of each proposed component. Code will be available at https://github.com/Fantasyele/LLaVA-KD.
>
---
#### [replaced 041] TAROT: Targeted Data Selection via Optimal Transport
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2412.00420v2](http://arxiv.org/pdf/2412.00420v2)**

> **作者:** Lan Feng; Fan Nie; Yuejiang Liu; Alexandre Alahi
>
> **摘要:** We propose TAROT, a targeted data selection framework grounded in optimal transport theory. Previous targeted data selection methods primarily rely on influence-based greedy heuristics to enhance domain-specific performance. While effective on limited, unimodal data (i.e., data following a single pattern), these methods struggle as target data complexity increases. Specifically, in multimodal distributions, these heuristics fail to account for multiple inherent patterns, leading to suboptimal data selection. This work identifies two primary factors contributing to this limitation: (i) the disproportionate impact of dominant feature components in high-dimensional influence estimation, and (ii) the restrictive linear additive assumptions inherent in greedy selection strategies. To address these challenges, TAROT incorporates whitened feature distance to mitigate dominant feature bias, providing a more reliable measure of data influence. Building on this, TAROT uses whitened feature distance to quantify and minimize the optimal transport distance between the selected data and target domains. Notably, this minimization also facilitates the estimation of optimal selection ratios. We evaluate TAROT across multiple tasks, including semantic segmentation, motion prediction, and instruction tuning. Results consistently show that TAROT outperforms state-of-the-art methods, highlighting its versatility across various deep learning tasks. Code is available at https://github.com/vita-epfl/TAROT.
>
---
#### [replaced 042] SPACE-SUIT: An Artificial Intelligence Based Chromospheric Feature Extractor and Classifier for SUIT
- **分类: astro-ph.SR; astro-ph.IM; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.08589v2](http://arxiv.org/pdf/2412.08589v2)**

> **作者:** Pranava Seth; Vishal Upendran; Megha Anand; Janmejoy Sarkar; Soumya Roy; Priyadarshan Chaki; Pratyay Chowdhury; Borishan Ghosh; Durgesh Tripathi
>
> **备注:** Published in Solar Physics
>
> **摘要:** The Solar Ultraviolet Imaging Telescope(SUIT) onboard Aditya-L1 is an imager that observes the solar photosphere and chromosphere through observations in the wavelength range of 200-400 nm. A comprehensive understanding of the plasma and thermodynamic properties of chromospheric and photospheric morphological structures requires a large sample statistical study, necessitating the development of automatic feature detection methods. To this end, we develop the feature detection algorithm SPACE-SUIT: Solar Phenomena Analysis and Classification using Enhanced vision techniques for SUIT, to detect and classify the solar chromospheric features to be observed from SUIT's Mg II k filter. Specifically, we target plage regions, sunspots, filaments, and off-limb structures. SPACE uses YOLO, a neural network-based model to identify regions of interest. We train and validate SPACE using mock-SUIT images developed from Interface Region Imaging Spectrometer(IRIS) full-disk mosaic images in Mg II k line, while we also perform detection on Level-1 SUIT data. SPACE achieves an approximate precision of 0.788, recall 0.863 and MAP of 0.874 on the validation mock SUIT FITS dataset. Given the manual labeling of our dataset, we perform "self-validation" by applying statistical measures and Tamura features on the ground truth and predicted bounding boxes. We find the distributions of entropy, contrast, dissimilarity, and energy to show differences in the features. These differences are qualitatively captured by the detected regions predicted by SPACE and validated with the observed SUIT images, even in the absence of labeled ground truth. This work not only develops a chromospheric feature extractor but also demonstrates the effectiveness of statistical metrics and Tamura features for distinguishing chromospheric features, offering independent validation for future detection schemes.
>
---
#### [replaced 043] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.14171v2](http://arxiv.org/pdf/2412.14171v2)**

> **作者:** Jihan Yang; Shusheng Yang; Anjali W. Gupta; Rilyn Han; Li Fei-Fei; Saining Xie
>
> **备注:** Project page: https://vision-x-nyu.github.io/thinking-in-space.github.io/
>
> **摘要:** Humans possess the visual-spatial intelligence to remember spaces from sequential visual observations. However, can Multimodal Large Language Models (MLLMs) trained on million-scale video datasets also ``think in space'' from videos? We present a novel video-based visual-spatial intelligence benchmark (VSI-Bench) of over 5,000 question-answer pairs, and find that MLLMs exhibit competitive - though subhuman - visual-spatial intelligence. We probe models to express how they think in space both linguistically and visually and find that while spatial reasoning capabilities remain the primary bottleneck for MLLMs to reach higher benchmark performance, local world models and spatial awareness do emerge within these models. Notably, prevailing linguistic reasoning techniques (e.g., chain-of-thought, self-consistency, tree-of-thoughts) fail to improve performance, whereas explicitly generating cognitive maps during question-answering enhances MLLMs' spatial distance ability.
>
---
#### [replaced 044] Bi-modality medical images synthesis by a bi-directional discrete process matching method
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.03977v3](http://arxiv.org/pdf/2409.03977v3)**

> **作者:** Zhe Xiong; Qiaoqiao Ding; Xiaoqun Zhang
>
> **摘要:** Recently, medical image synthesis gains more and more popularity, along with the rapid development of generative models. Medical image synthesis aims to generate an unacquired image modality, often from other observed data modalities. Synthesized images can be used for clinical diagnostic assistance, data augmentation for model training and validation or image quality improving. In the meanwhile, the flow-based models are among the successful generative models for the ability of generating realistic and high-quality synthetic images. However, most flow-based models require to calculate flow ordinary different equation (ODE) evolution steps in synthesis process, for which the performances are significantly limited by heavy computation time due to a large number of time iterations. In this paper, we propose a novel flow-based model, namely bi-directional Discrete Process Matching (Bi-DPM) to accomplish the bi-modality image synthesis tasks. Different to other flow matching based models, we propose to utilize both forward and backward ODE flows and enhance the consistency on the intermediate images over a few discrete time steps, resulting in a synthesis process maintaining high-quality generations for both modalities under the guidance of paired data. Our experiments on three datasets of MRI T1/T2 and CT/MRI demonstrate that Bi-DPM outperforms other state-of-the-art flow-based methods for bi-modality image synthesis, delivering higher image quality with accurate anatomical regions.
>
---
#### [replaced 045] MV2DFusion: Leveraging Modality-Specific Object Semantics for Multi-Modal 3D Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05945v2](http://arxiv.org/pdf/2408.05945v2)**

> **作者:** Zitian Wang; Zehao Huang; Yulu Gao; Naiyan Wang; Si Liu
>
> **摘要:** The rise of autonomous vehicles has significantly increased the demand for robust 3D object detection systems. While cameras and LiDAR sensors each offer unique advantages--cameras provide rich texture information and LiDAR offers precise 3D spatial data--relying on a single modality often leads to performance limitations. This paper introduces MV2DFusion, a multi-modal detection framework that integrates the strengths of both worlds through an advanced query-based fusion mechanism. By introducing an image query generator to align with image-specific attributes and a point cloud query generator, MV2DFusion effectively combines modality-specific object semantics without biasing toward one single modality. Then the sparse fusion process can be accomplished based on the valuable object semantics, ensuring efficient and accurate object detection across various scenarios. Our framework's flexibility allows it to integrate with any image and point cloud-based detectors, showcasing its adaptability and potential for future advancements. Extensive evaluations on the nuScenes and Argoverse2 datasets demonstrate that MV2DFusion achieves state-of-the-art performance, particularly excelling in long-range detection scenarios.
>
---
#### [replaced 046] PAD: Phase-Amplitude Decoupling Fusion for Multi-Modal Land Cover Classification
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.19136v2](http://arxiv.org/pdf/2504.19136v2)**

> **作者:** Huiling Zheng; Xian Zhong; Bin Liu; Yi Xiao; Bihan Wen; Xiaofeng Li
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** The fusion of Synthetic Aperture Radar (SAR) and RGB imagery for land cover classification remains challenging due to modality heterogeneity and underutilized spectral complementarity. Existing methods often fail to decouple shared structural features from modality-complementary radiometric attributes, causing feature conflicts and information loss. To address this, we propose Phase-Amplitude Decoupling (PAD), a frequency-aware framework that separates phase (modality-shared) and amplitude (modality-complementary) components in the Fourier domain, thus reinforcing shared structures while preserving complementary characteristics to improve fusion quality. Unlike prior approaches that overlook the distinct physical properties encoded in frequency spectra, PAD is the first to introduce explicit amplitude-phase decoupling for multi-modal fusion. Specifically, PAD comprises two key components: 1) Phase Spectrum Correction (PSC), which aligns cross-modal phase features via convolution-guided scaling to enhance geometric consistency; and 2) Amplitude Spectrum Fusion (ASF), which dynamically integrates high-frequency and low-frequency patterns using frequency-adaptive multilayer perceptrons, leveraging SAR's morphological sensitivity and RGB's spectral richness. Extensive experiments on WHU-OPT-SAR and DDHR-SK datasets demonstrate state-of-the-art performance. Our work establishes a new paradigm for physics-aware multi-modal fusion in remote sensing. The code will be available at https://github.com/RanFeng2/PAD.
>
---
#### [replaced 047] BANet: Bilateral Aggregation Network for Mobile Stereo Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.03259v2](http://arxiv.org/pdf/2503.03259v2)**

> **作者:** Gangwei Xu; Jiaxin Liu; Xianqi Wang; Junda Cheng; Yong Deng; Jinliang Zang; Yurui Chen; Xin Yang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** State-of-the-art stereo matching methods typically use costly 3D convolutions to aggregate a full cost volume, but their computational demands make mobile deployment challenging. Directly applying 2D convolutions for cost aggregation often results in edge blurring, detail loss, and mismatches in textureless regions. Some complex operations, like deformable convolutions and iterative warping, can partially alleviate this issue; however, they are not mobile-friendly, limiting their deployment on mobile devices. In this paper, we present a novel bilateral aggregation network (BANet) for mobile stereo matching that produces high-quality results with sharp edges and fine details using only 2D convolutions. Specifically, we first separate the full cost volume into detailed and smooth volumes using a spatial attention map, then perform detailed and smooth aggregations accordingly, ultimately fusing both to obtain the final disparity map. Experimental results demonstrate that our BANet-2D significantly outperforms other mobile-friendly methods, achieving 35.3\% higher accuracy on the KITTI 2015 leaderboard than MobileStereoNet-2D, with faster runtime on mobile devices. Code: \textcolor{magenta}{https://github.com/gangweix/BANet}.
>
---
#### [replaced 048] ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20323v4](http://arxiv.org/pdf/2502.20323v4)**

> **作者:** Xuangeng Chu; Nabarun Goswami; Ziteng Cui; Hanqin Wang; Tatsuya Harada
>
> **备注:** More video demonstrations, code, models and data can be found on our project website: http://xg-chu.site/project_artalk/
>
> **摘要:** Speech-driven 3D facial animation aims to generate realistic lip movements and facial expressions for 3D head models from arbitrary audio clips. Although existing diffusion-based methods are capable of producing natural motions, their slow generation speed limits their application potential. In this paper, we introduce a novel autoregressive model that achieves real-time generation of highly synchronized lip movements and realistic head poses and eye blinks by learning a mapping from speech to a multi-scale motion codebook. Furthermore, our model can adapt to unseen speaking styles, enabling the creation of 3D talking avatars with unique personal styles beyond the identities seen during training. Extensive evaluations and user studies demonstrate that our method outperforms existing approaches in lip synchronization accuracy and perceived quality.
>
---
#### [replaced 049] Privacy-Preserving Operating Room Workflow Analysis using Digital Twins
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.12552v2](http://arxiv.org/pdf/2504.12552v2)**

> **作者:** Alejandra Perez; Han Zhang; Yu-Chun Ku; Lalithkumar Seenivasan; Roger Soberanis; Jose L. Porras; Richard Day; Jeff Jopling; Peter Najjar; Mathias Unberath
>
> **摘要:** The operating room (OR) is a complex environment where optimizing workflows is critical to reduce costs and improve patient outcomes. While computer vision approaches for automatic recognition of perioperative events can identify bottlenecks for OR optimization, privacy concerns limit the use of OR videos for automated event detection. We propose a two-stage pipeline for privacy-preserving OR video analysis and event detection. First, we leverage vision foundation models for depth estimation and semantic segmentation to generate de-identified Digital Twins (DT) of the OR from conventional RGB videos. Second, we employ the SafeOR model, a fused two-stream approach that processes segmentation masks and depth maps for OR event detection. Evaluation on an internal dataset of 38 simulated surgical trials with five event classes shows that our DT-based approach achieves performance on par with -- and sometimes better than -- raw RGB video-based models for OR event detection. Digital Twins enable privacy-preserving OR workflow analysis, facilitating the sharing of de-identified data across institutions and potentially enhancing model generalizability by mitigating domain-specific appearance differences.
>
---
#### [replaced 050] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23897v3](http://arxiv.org/pdf/2506.23897v3)**

> **作者:** Longliang Liu; Miaojie Feng; Junda Cheng; Jijun Xiang; Xuan Zhu; Xin Yang
>
> **摘要:** Panoramic optical flow enables a comprehensive understanding of temporal dynamics across wide fields of view. However, severe distortions caused by sphere-to-plane projections, such as the equirectangular projection (ERP), significantly degrade the performance of conventional perspective-based optical flow methods, especially in polar regions. To address this challenge, we propose PriOr-Flow, a novel dual-branch framework that leverages the low-distortion nature of the orthogonal view to enhance optical flow estimation in these regions. Specifically, we introduce the Dual-Cost Collaborative Lookup (DCCL) operator, which jointly retrieves correlation information from both the primitive and orthogonal cost volumes, effectively mitigating distortion noise during cost volume construction. Furthermore, our Ortho-Driven Distortion Compensation (ODDC) module iteratively refines motion features from both branches, further suppressing polar distortions. Extensive experiments demonstrate that PriOr-Flow is compatible with various perspective-based iterative optical flow methods and consistently achieves state-of-the-art performance on publicly available panoramic optical flow datasets, setting a new benchmark for wide-field motion estimation. The code is publicly available at: https://github.com/longliangLiu/PriOr-Flow.
>
---
#### [replaced 051] Skip-Vision: Efficient and Scalable Acceleration of Vision-Language Models via Adaptive Token Skipping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21817v3](http://arxiv.org/pdf/2503.21817v3)**

> **作者:** Weili Zeng; Ziyuan Huang; Kaixiang Ji; Yichao Yan
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Transformer-based models have driven significant advancements in Multimodal Large Language Models (MLLMs), yet their computational costs surge drastically when scaling resolution, training data, and model parameters. A key bottleneck stems from the proliferation of visual tokens required for fine-grained image understanding. We propose Skip-Vision, a unified framework addressing both training and inference inefficiencies in vision-language models. On top of conventional token compression approaches, our method introduces two complementary acceleration strategies. For training acceleration, we observe that Feed-Forward Network (FFN) computations on visual tokens induce marginal feature updates. This motivates our Skip-FFN strategy, which bypasses FFN layers for redundant visual tokens. For inference acceleration, we design a selective KV-cache removal mechanism that prunes the skipped key-value pairs during decoding while preserving model performance. Experimental results demonstrate that Skip-Vision reduces training time by up to 35\%, inference FLOPs by 75\%, and latency by 45\%, while achieving comparable or superior performance to existing methods. Our work provides a practical solution for scaling high-performance MLLMs with enhanced efficiency.
>
---
#### [replaced 052] Customizable ROI-Based Deep Image Compression
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.00373v3](http://arxiv.org/pdf/2507.00373v3)**

> **作者:** Jian Jin; Fanxin Xia; Feng Ding; Xinfeng Zhang; Meiqin Liu; Yao Zhao; Weisi Lin; Lili Meng
>
> **摘要:** Region of Interest (ROI)-based image compression optimizes bit allocation by prioritizing ROI for higher-quality reconstruction. However, as the users (including human clients and downstream machine tasks) become more diverse, ROI-based image compression needs to be customizable to support various preferences. For example, different users may define distinct ROI or require different quality trade-offs between ROI and non-ROI. Existing ROI-based image compression schemes predefine the ROI, making it unchangeable, and lack effective mechanisms to balance reconstruction quality between ROI and non-ROI. This work proposes a paradigm for customizable ROI-based deep image compression. First, we develop a Text-controlled Mask Acquisition (TMA) module, which allows users to easily customize their ROI for compression by just inputting the corresponding semantic \emph{text}. It makes the encoder controlled by text. Second, we design a Customizable Value Assign (CVA) mechanism, which masks the non-ROI with a changeable extent decided by users instead of a constant one to manage the reconstruction quality trade-off between ROI and non-ROI. Finally, we present a Latent Mask Attention (LMA) module, where the latent spatial prior of the mask and the latent Rate-Distortion Optimization (RDO) prior of the image are extracted and fused in the latent space, and further used to optimize the latent representation of the source image. Experimental results demonstrate that our proposed customizable ROI-based deep image compression paradigm effectively addresses the needs of customization for ROI definition and mask acquisition as well as the reconstruction quality trade-off management between the ROI and non-ROI.
>
---
#### [replaced 053] Towards an Explainable Comparison and Alignment of Feature Embeddings
- **分类: cs.LG; cs.AI; cs.CV; math.SP**

- **链接: [http://arxiv.org/pdf/2506.06231v2](http://arxiv.org/pdf/2506.06231v2)**

> **作者:** Mohammad Jalali; Bahar Dibaei Nia; Farzan Farnia
>
> **摘要:** While several feature embedding models have been developed in the literature, comparisons of these embeddings have largely focused on their numerical performance in classification-related downstream applications. However, an interpretable comparison of different embeddings requires identifying and analyzing mismatches between sample groups clustered within the embedding spaces. In this work, we propose the \emph{Spectral Pairwise Embedding Comparison (SPEC)} framework to compare embeddings and identify their differences in clustering a reference dataset. Our approach examines the kernel matrices derived from two embeddings and leverages the eigendecomposition of the difference kernel matrix to detect sample clusters that are captured differently by the two embeddings. We present a scalable implementation of this kernel-based approach, with computational complexity that grows linearly with the sample size. Furthermore, we introduce an optimization problem using this framework to align two embeddings, ensuring that clusters identified in one embedding are also captured in the other model. We provide numerical results demonstrating the SPEC's application to compare and align embeddings on large-scale datasets such as ImageNet and MS-COCO. The project page is available at https://mjalali.github.io/SPEC/.
>
---
#### [replaced 054] AI Flow: Perspectives, Scenarios, and Approaches
- **分类: cs.AI; cs.CL; cs.CV; cs.DC; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.12479v2](http://arxiv.org/pdf/2506.12479v2)**

> **作者:** Hongjun An; Wenhan Hu; Sida Huang; Siqi Huang; Ruanjun Li; Yuanzhi Liang; Jiawei Shao; Yiliang Song; Zihan Wang; Cheng Yuan; Chi Zhang; Hongyuan Zhang; Wenhao Zhuang; Xuelong Li
>
> **备注:** Authors are with Institute of Artificial Intelligence (TeleAI), China Telecom, China. Author names are listed alphabetically by surname. This work was conducted at TeleAI, facilitated by Dr. Jiawei Shao (e-mail: shaojw2@chinatelecom.cn) under the leadership of Prof. Xuelong Li. The corresponding author is Prof. Xuelong Li (e-mail: xuelong li@ieee.org), the CTO and Chief Scientist of China Telecom
>
> **摘要:** Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems.
>
---
#### [replaced 055] RFWNet: A Lightweight Remote Sensing Object Detector Integrating Multiscale Receptive Fields and Foreground Focus Mechanism
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.00545v2](http://arxiv.org/pdf/2503.00545v2)**

> **作者:** Yujie Lei; Wenjie Sun; Sen Jia; Qingquan Li; Jie Zhang
>
> **摘要:** Challenges in remote sensing object detection(RSOD), such as high interclass similarity, imbalanced foreground-background distribution, and the small size of objects in remote sensing images, significantly hinder detection accuracy. Moreover, the tradeoff between model accuracy and computational complexity poses additional constraints on the application of RSOD algorithms. To address these issues, this study proposes an efficient and lightweight RSOD algorithm integrating multiscale receptive fields and foreground focus mechanism, named robust foreground weighted network(RFWNet). Specifically, we proposed a lightweight backbone network receptive field adaptive selection network (RFASNet), leveraging the rich context information of remote sensing images to enhance class separability. Additionally, we developed a foreground-background separation module(FBSM)consisting of a background redundant information filtering module (BRIFM) and a foreground information enhancement module (FIEM) to emphasize critical regions within images while filtering redundant background information. Finally, we designed a loss function, the weighted CIoU-Wasserstein loss (LWCW),which weights the IoU-based loss by using the normalized Wasserstein distance to mitigate model sensitivity to small object position deviations. The comprehensive experimental results demonstrate that RFWNet achieved 95.3% and 73.2% mean average precision (mAP) with 6.0 M parameters on the DOTA V1.0 and NWPU VHR-10 datasets, respectively, with an inference speed of 52 FPS.
>
---
#### [replaced 056] Stronger, Steadier & Superior: Geometric Consistency in Depth VFM Forges Domain Generalized Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12753v2](http://arxiv.org/pdf/2504.12753v2)**

> **作者:** Siyu Chen; Ting Han; Changshe Zhang; Xin Luo; Meiliu Wu; Guorong Cai; Jinhe Su
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Vision Foundation Models (VFMs) have delivered remarkable performance in Domain Generalized Semantic Segmentation (DGSS). However, recent methods often overlook the fact that visual cues are susceptible, whereas the underlying geometry remains stable, rendering depth information more robust. In this paper, we investigate the potential of integrating depth information with features from VFMs, to improve the geometric consistency within an image and boost the generalization performance of VFMs. We propose a novel fine-tuning DGSS framework, named DepthForge, which integrates the visual cues from frozen DINOv2 or EVA02 and depth cues from frozen Depth Anything V2. In each layer of the VFMs, we incorporate depth-aware learnable tokens to continuously decouple domain-invariant visual and spatial information, thereby enhancing depth awareness and attention of the VFMs. Finally, we develop a depth refinement decoder and integrate it into the model architecture to adaptively refine multi-layer VFM features and depth-aware learnable tokens. Extensive experiments are conducted based on various DGSS settings and five different datsets as unseen target domains. The qualitative and quantitative results demonstrate that our method significantly outperforms alternative approaches with stronger performance, steadier visual-spatial attention, and superior generalization ability. In particular, DepthForge exhibits outstanding performance under extreme conditions (e.g., night and snow). Code is available at https://github.com/anonymouse-xzrptkvyqc/DepthForge.
>
---
#### [replaced 057] RGC-VQA: An Exploration Database for Robotic-Generated Video Quality Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23852v2](http://arxiv.org/pdf/2506.23852v2)**

> **作者:** Jianing Jin; Jiangyong Ying; Huiyu Duan; Liu Yang; Sijing Wu; Yunhao Li; Yushuo Zheng; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** As camera-equipped robotic platforms become increasingly integrated into daily life, robotic-generated videos have begun to appear on streaming media platforms, enabling us to envision a future where humans and robots coexist. We innovatively propose the concept of Robotic-Generated Content (RGC) to term these videos generated from egocentric perspective of robots. The perceptual quality of RGC videos is critical in human-robot interaction scenarios, and RGC videos exhibit unique distortions and visual requirements that differ markedly from those of professionally-generated content (PGC) videos and user-generated content (UGC) videos. However, dedicated research on quality assessment of RGC videos is still lacking. To address this gap and to support broader robotic applications, we establish the first Robotic-Generated Content Database (RGCD), which contains a total of 2,100 videos drawn from three robot categories and sourced from diverse platforms. A subjective VQA experiment is conducted subsequently to assess human visual perception of robotic-generated videos. Finally, we conduct a benchmark experiment to evaluate the performance of 11 state-of-the-art VQA models on our database. Experimental results reveal significant limitations in existing VQA models when applied to complex, robotic-generated content, highlighting a critical need for RGC-specific VQA models. Our RGCD is publicly available at: https://github.com/IntMeGroup/RGC-VQA.
>
---
#### [replaced 058] LUDO: Low-Latency Understanding of Deformable Objects using Point Cloud Occupancy Functions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.08777v5](http://arxiv.org/pdf/2411.08777v5)**

> **作者:** Pit Henrich; Franziska Mathis-Ullrich; Paul Maria Scheikl
>
> **备注:** Published in IEEE Transactions on Robotics (T-RO)
>
> **摘要:** Accurately determining the shape of deformable objects and the location of their internal structures is crucial for medical tasks that require precise targeting, such as robotic biopsies. We introduce LUDO, a method for accurate low-latency understanding of deformable objects. LUDO reconstructs objects in their deformed state, including their internal structures, from a single-view point cloud observation in under 30 ms using occupancy networks. LUDO provides uncertainty estimates for its predictions. Additionally, it provides explainability by highlighting key features in its input observations. Both uncertainty and explainability are important for safety-critical applications such as surgery. We evaluate LUDO in real-world robotic experiments, achieving a success rate of 98.9% for puncturing various regions of interest (ROIs) inside deformable objects. We compare LUDO to a popular baseline and show its superior ROI localization accuracy, training time, and memory requirements. LUDO demonstrates the potential to interact with deformable objects without the need for deformable registration methods.
>
---
#### [replaced 059] COEF-VQ: Cost-Efficient Video Quality Understanding through a Cascaded Multimodal LLM Framework
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.10435v2](http://arxiv.org/pdf/2412.10435v2)**

> **作者:** Xin Dong; Sen Jia; Ming Rui Wang; Yan Li; Zhenheng Yang; Bingfeng Deng; Hongyu Xiong
>
> **摘要:** Recently, with the emergence of recent Multimodal Large Language Model (MLLM) technology, it has become possible to exploit its video understanding capability on different classification tasks. In practice, we face the difficulty of huge requirements for GPU resource if we need to deploy MLLMs online. In this paper, we propose COEF-VQ, a novel cascaded MLLM framework designed to enhance video quality understanding on the short-video platform while optimizing computational efficiency. Our approach integrates an entropy-based pre-filtering stage, where a lightweight model assesses uncertainty and selectively filters cases before passing them to the more computationally intensive MLLM for final evaluation. By prioritizing high-uncertainty samples for deeper analysis, our framework significantly reduces GPU usage while maintaining the strong classification performance of a full MLLM deployment. To demonstrate the effectiveness of COEF-VQ, we deploy this new framework onto the video management platform (VMP) at the short-video platform, and perform a series of detailed experiments on two in-house tasks related to video quality understanding. We show that COEF-VQ leads to substantial performance gains from the offline evaluation in these two tasks and effectively enhances platform safety with limit resource consumption, significantly reducing inappropriate content video view rate by 9.9% in a online A/B test without affecting engagement. Post-launch monitoring confirmed sustained improvements, validating its real-world impact.
>
---
#### [replaced 060] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01439v2](http://arxiv.org/pdf/2507.01439v2)**

> **作者:** Shaocheng Yan; Pengcheng Shi; Zhenjun Zhao; Kaixin Wang; Kuang Cao; Ji Wu; Jiayuan Li
>
> **备注:** ICCV-2025 Accepted Paper
>
> **摘要:** Robust estimation is essential in correspondence-based Point Cloud Registration (PCR). Existing methods using maximal clique search in compatibility graphs achieve high recall but suffer from exponential time complexity, limiting their use in time-sensitive applications. To address this challenge, we propose a fast and robust estimator, TurboReg, built upon a novel lightweight clique, TurboClique, and a highly parallelizable Pivot-Guided Search (PGS) algorithm. First, we define the TurboClique as a 3-clique within a highly-constrained compatibility graph. The lightweight nature of the 3-clique allows for efficient parallel searching, and the highly-constrained compatibility graph ensures robust spatial consistency for stable transformation estimation. Next, PGS selects matching pairs with high SC$^2$ scores as pivots, effectively guiding the search toward TurboCliques with higher inlier ratios. Moreover, the PGS algorithm has linear time complexity and is significantly more efficient than the maximal clique search with exponential time complexity. Extensive experiments show that TurboReg achieves state-of-the-art performance across multiple real-world datasets, with substantial speed improvements. For example, on the 3DMatch+FCGF dataset, TurboReg (1K) operates $208.22\times$ faster than 3DMAC while also achieving higher recall. Our code is accessible at \href{https://github.com/Laka-3DV/TurboReg}{\texttt{TurboReg}}.
>
---
#### [replaced 061] Lightweight Structure-Aware Attention for Visual Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2211.16289v2](http://arxiv.org/pdf/2211.16289v2)**

> **作者:** Heeseung Kwon; Francisco M. Castro; Manuel J. Marin-Jimenez; Nicolas Guil; Karteek Alahari
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Attention operator has been widely used as a basic brick in visual understanding since it provides some flexibility through its adjustable kernels. However, this operator suffers from inherent limitations: (1) the attention kernel is not discriminative enough, resulting in high redundancy, and (2) the complexity in computation and memory is quadratic in the sequence length. In this paper, we propose a novel attention operator, called Lightweight Structure-aware Attention (LiSA), which has a better representation power with log-linear complexity. Our operator transforms the attention kernels to be more discriminative by learning structural patterns. These structural patterns are encoded by exploiting a set of relative position embeddings (RPEs) as multiplicative weights, thereby improving the representation power of the attention kernels. Additionally, the RPEs are approximated to obtain log-linear complexity. Our experiments and analyses demonstrate that the proposed operator outperforms self-attention and other existing operators, achieving state-of-the-art results on ImageNet-1K and other downstream tasks such as video action recognition on Kinetics-400, object detection \& instance segmentation on COCO, and semantic segmentation on ADE-20K.
>
---
#### [replaced 062] Rejoining fragmented ancient bamboo slips with physics-driven deep learning
- **分类: cs.CV; cond-mat.mtrl-sci**

- **链接: [http://arxiv.org/pdf/2505.08601v2](http://arxiv.org/pdf/2505.08601v2)**

> **作者:** Jinchi Zhu; Zhou Zhao; Hailong Lei; Xiaoguang Wang; Jialiang Lu; Jing Li; Qianqian Tang; Jiachen Shen; Gui-Song Xia; Bo Du; Yongchao Xu
>
> **摘要:** Bamboo slips are a crucial medium for recording ancient civilizations in East Asia, and offers invaluable archaeological insights for reconstructing the Silk Road, studying material culture exchanges, and global history. However, many excavated bamboo slips have been fragmented into thousands of irregular pieces, making their rejoining a vital yet challenging step for understanding their content. Here we introduce WisePanda, a physics-driven deep learning framework designed to rejoin fragmented bamboo slips. Based on the physics of fracture and material deterioration, WisePanda automatically generates synthetic training data that captures the physical properties of bamboo fragmentations. This approach enables the training of a matching network without requiring manually paired samples, providing ranked suggestions to facilitate the rejoining process. Compared to the leading curve matching method, WisePanda increases Top-50 matching accuracy from 36% to 52% among more than one thousand candidate fragments. Archaeologists using WisePanda have experienced substantial efficiency improvements (approximately 20 times faster) when rejoining fragmented bamboo slips. This research demonstrates that incorporating physical principles into deep learning models can significantly enhance their performance, transforming how archaeologists restore and study fragmented artifacts. WisePanda provides a new paradigm for addressing data scarcity in ancient artifact restoration through physics-driven machine learning.
>
---
#### [replaced 063] ODE$_t$(ODE$_l$): Shortcutting the Time and Length in Diffusion and Flow Models for Faster Sampling
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21714v2](http://arxiv.org/pdf/2506.21714v2)**

> **作者:** Denis Gudovskiy; Wenzhao Zheng; Tomoyuki Okuno; Yohei Nakata; Kurt Keutzer
>
> **备注:** Preprint. Github page: github.com/gudovskiy/odelt
>
> **摘要:** Recently, continuous normalizing flows (CNFs) and diffusion models (DMs) have been studied using the unified theoretical framework. Although such models can generate high-quality data points from a noise distribution, the sampling demands multiple iterations to solve an ordinary differential equation (ODE) with high computational complexity. Most existing methods focus on reducing the number of time steps during the sampling process to improve efficiency. In this work, we explore a complementary direction in which the quality-complexity tradeoff can be dynamically controlled in terms of time steps and in the length of the neural network. We achieve this by rewiring the blocks in the transformer-based architecture to solve an inner discretized ODE w.r.t. its length. Then, we employ time- and length-wise consistency terms during flow matching training, and as a result, the sampling can be performed with an arbitrary number of time steps and transformer blocks. Unlike others, our ODE$_t$(ODE$_l$) approach is solver-agnostic in time dimension and decreases both latency and memory usage. Compared to the previous state of the art, image generation experiments on CelebA-HQ and ImageNet show a latency reduction of up to 3$\times$ in the most efficient sampling mode, and a FID score improvement of up to 3.5 points for high-quality sampling. We release our code and model weights with fully reproducible experiments.
>
---
#### [replaced 064] MaizeField3D: A Curated 3D Point Cloud and Procedural Model Dataset of Field-Grown Maize from a Diversity Panel
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07813v3](http://arxiv.org/pdf/2503.07813v3)**

> **作者:** Elvis Kimara; Mozhgan Hadadi; Jackson Godbersen; Aditya Balu; Talukder Jubery; Yawei Li; Adarsh Krishnamurthy; Patrick S. Schnable; Baskar Ganapathysubramanian
>
> **备注:** Elvis Kimara and Mozhgan Hadadi contributed equally to this work
>
> **摘要:** The development of artificial intelligence (AI) and machine learning (ML) based tools for 3D phenotyping, especially for maize, has been limited due to the lack of large and diverse 3D datasets. 2D image datasets fail to capture essential structural details such as leaf architecture, plant volume, and spatial arrangements that 3D data provide. To address this limitation, we present MaizeField3D (https://baskargroup.github.io/MaizeField3D/), a curated dataset of 3D point clouds of field-grown maize plants from a diverse genetic panel, designed to be AI-ready for advancing agricultural research. Our dataset includes 1,045 high-quality point clouds of field-grown maize collected using a terrestrial laser scanner (TLS). Point clouds of 520 plants from this dataset were segmented and annotated using a graph-based segmentation method to isolate individual leaves and stalks, ensuring consistent labeling across all samples. This labeled data was then used for fitting procedural models that provide a structured parametric representation of the maize plants. The leaves of the maize plants in the procedural models are represented using Non-Uniform Rational B-Spline (NURBS) surfaces that were generated using a two-step optimization process combining gradient-free and gradient-based methods. We conducted rigorous manual quality control on all datasets, correcting errors in segmentation, ensuring accurate leaf ordering, and validating metadata annotations. The dataset also includes metadata detailing plant morphology and quality, alongside multi-resolution subsampled point cloud data (100k, 50k, 10k points), which can be readily used for different downstream computational tasks. MaizeField3D will serve as a comprehensive foundational dataset for AI-driven phenotyping, plant structural analysis, and 3D applications in agricultural research.
>
---
#### [replaced 065] Similarity Memory Prior is All You Need for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00585v2](http://arxiv.org/pdf/2507.00585v2)**

> **作者:** Tang Hao; Guo ZhiQing; Wang LieJun; Liu Chao
>
> **摘要:** In recent years, it has been found that "grandmother cells" in the primary visual cortex (V1) of macaques can directly recognize visual input with complex shapes. This inspires us to examine the value of these cells in promoting the research of medical image segmentation. In this paper, we design a Similarity Memory Prior Network (Sim-MPNet) for medical image segmentation. Specifically, we propose a Dynamic Memory Weights-Loss Attention (DMW-LA), which matches and remembers the category features of specific lesions or organs in medical images through the similarity memory prior in the prototype memory bank, thus helping the network to learn subtle texture changes between categories. DMW-LA also dynamically updates the similarity memory prior in reverse through Weight-Loss Dynamic (W-LD) update strategy, effectively assisting the network directly extract category features. In addition, we propose the Double-Similarity Global Internal Enhancement Module (DS-GIM) to deeply explore the internal differences in the feature distribution of input data through cosine similarity and euclidean distance. Extensive experiments on four public datasets show that Sim-MPNet has better segmentation performance than other state-of-the-art methods. Our code is available on https://github.com/vpsg-research/Sim-MPNet.
>
---
#### [replaced 066] Towards a Novel Measure of User Trust in XAI Systems
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.05766v2](http://arxiv.org/pdf/2405.05766v2)**

> **作者:** Miquel Miró-Nicolau; Gabriel Moyà-Alcover; Antoni Jaume-i-Capó; Manuel González-Hidalgo; Adel Ghazel; Maria Gemma Sempere Campello; Juan Antonio Palmer Sancho
>
> **摘要:** The increasing reliance on Deep Learning models, combined with their inherent lack of transparency, has spurred the development of a novel field of study known as eXplainable AI (XAI) methods. These methods seek to enhance the trust of end-users in automated systems by providing insights into the rationale behind their decisions. This paper presents a novel trust measure in XAI systems, allowing their refinement. Our proposed metric combines both performance metrics and trust indicators from an objective perspective. To validate this novel methodology, we conducted three case studies showing an improvement respect the state-of-the-art, with an increased sensitiviy to different scenarios.
>
---
#### [replaced 067] Fairer Analysis and Demographically Balanced Face Generation for Fairer Face Verification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03349v2](http://arxiv.org/pdf/2412.03349v2)**

> **作者:** Alexandre Fournier-Montgieux; Michael Soumm; Adrian Popescu; Bertrand Luvison; Hervé Le Borgne
>
> **备注:** Published in WACV2025
>
> **摘要:** Face recognition and verification are two computer vision tasks whose performances have advanced with the introduction of deep representations. However, ethical, legal, and technical challenges due to the sensitive nature of face data and biases in real-world training datasets hinder their development. Generative AI addresses privacy by creating fictitious identities, but fairness problems remain. Using the existing DCFace SOTA framework, we introduce a new controlled generation pipeline that improves fairness. Through classical fairness metrics and a proposed in-depth statistical analysis based on logit models and ANOVA, we show that our generation pipeline improves fairness more than other bias mitigation approaches while slightly improving raw performance.
>
---
#### [replaced 068] PAID: Pairwise Angular-Invariant Decomposition for Continual Test-Time Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02453v2](http://arxiv.org/pdf/2506.02453v2)**

> **作者:** Kunyu Wang; Xueyang Fu; Yuanfei Bao; Chengjie Ge; Chengzhi Cao; Wei Zhai; Zheng-Jun Zha
>
> **摘要:** Continual Test-Time Adaptation (CTTA) aims to online adapt a pre-trained model to changing environments during inference. Most existing methods focus on exploiting target data, while overlooking another crucial source of information, the pre-trained weights, which encode underutilized domain-invariant priors. This paper takes the geometric attributes of pre-trained weights as a starting point, systematically analyzing three key components: magnitude, absolute angle, and pairwise angular structure. We find that the pairwise angular structure remains stable across diverse corrupted domains and encodes domain-invariant semantic information, suggesting it should be preserved during adaptation. Based on this insight, we propose PAID (Pairwise Angular-Invariant Decomposition), a prior-driven CTTA method that decomposes weight into magnitude and direction, and introduces a learnable orthogonal matrix via Householder reflections to globally rotate direction while preserving the pairwise angular structure. During adaptation, only the magnitudes and the orthogonal matrices are updated. PAID achieves consistent improvements over recent SOTA methods on four widely used CTTA benchmarks, demonstrating that preserving pairwise angular structure offers a simple yet effective principle for CTTA.
>
---
#### [replaced 069] MAD: Makeup All-in-One with Cross-Domain Diffusion Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02545v2](http://arxiv.org/pdf/2504.02545v2)**

> **作者:** Bo-Kai Ruan; Hong-Han Shuai
>
> **备注:** Accepted by CVPRW2025
>
> **摘要:** Existing makeup techniques often require designing multiple models to handle different inputs and align features across domains for different makeup tasks, e.g., beauty filter, makeup transfer, and makeup removal, leading to increased complexity. Another limitation is the absence of text-guided makeup try-on, which is more user-friendly without needing reference images. In this study, we make the first attempt to use a single model for various makeup tasks. Specifically, we formulate different makeup tasks as cross-domain translations and leverage a cross-domain diffusion model to accomplish all tasks. Unlike existing methods that rely on separate encoder-decoder configurations or cycle-based mechanisms, we propose using different domain embeddings to facilitate domain control. This allows for seamless domain switching by merely changing embeddings with a single model, thereby reducing the reliance on additional modules for different tasks. Moreover, to support precise text-to-makeup applications, we introduce the MT-Text dataset by extending the MT dataset with textual annotations, advancing the practicality of makeup technologies.
>
---
#### [replaced 070] Good Representation, Better Explanation: Role of Convolutional Neural Networks in Transformer-Based Remote Sensing Image Captioning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.16095v2](http://arxiv.org/pdf/2502.16095v2)**

> **作者:** Swadhin Das; Saarthak Gupta; Kamal Kumar; Raksha Sharma
>
> **摘要:** Remote Sensing Image Captioning (RSIC) is the process of generating meaningful descriptions from remote sensing images. Recently, it has gained significant attention, with encoder-decoder models serving as the backbone for generating meaningful captions. The encoder extracts essential visual features from the input image, transforming them into a compact representation, while the decoder utilizes this representation to generate coherent textual descriptions. Recently, transformer-based models have gained significant popularity due to their ability to capture long-range dependencies and contextual information. The decoder has been well explored for text generation, whereas the encoder remains relatively unexplored. However, optimizing the encoder is crucial as it directly influences the richness of extracted features, which in turn affects the quality of generated captions. To address this gap, we systematically evaluate twelve different convolutional neural network (CNN) architectures within a transformer-based encoder framework to assess their effectiveness in RSIC. The evaluation consists of two stages: first, a numerical analysis categorizes CNNs into different clusters, based on their performance. The best performing CNNs are then subjected to human evaluation from a human-centric perspective by a human annotator. Additionally, we analyze the impact of different search strategies, namely greedy search and beam search, to ensure the best caption. The results highlight the critical role of encoder selection in improving captioning performance, demonstrating that specific CNN architectures significantly enhance the quality of generated descriptions for remote sensing images. By providing a detailed comparison of multiple encoders, this study offers valuable insights to guide advances in transformer-based image captioning models.
>
---
#### [replaced 071] Sequence-aware Pre-training for Echocardiography Probe Movement Guidance
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.15026v2](http://arxiv.org/pdf/2408.15026v2)**

> **作者:** Haojun Jiang; Teng Wang; Zhenguo Sun; Yulin Wang; Yang Yue; Yu Sun; Ning Jia; Meng Li; Shaqi Luo; Shiji Song; Gao Huang
>
> **备注:** Tech Report
>
> **摘要:** Echocardiography is an essential medical technique for diagnosing cardiovascular diseases, but its high operational complexity has led to a shortage of trained professionals. To address this issue, we introduce a novel probe movement guidance algorithm that has the potential to be applied in guiding robotic systems or novices with probe pose adjustment for high-quality standard plane image acquisition.Cardiac ultrasound faces two major challenges: (1) the inherently complex structure of the heart, and (2) significant individual variations. Previous works have only learned the population-averaged structure of the heart rather than personalized cardiac structures, leading to a performance bottleneck. Clinically, we observe that sonographers dynamically adjust their interpretation of a patient's cardiac anatomy based on prior scanning sequences, consequently refining their scanning strategies. Inspired by this, we propose a novel sequence-aware self-supervised pre-training method. Specifically, our approach learns personalized three-dimensional cardiac structural features by predicting the masked-out image features and probe movement actions in a scanning sequence. We hypothesize that if the model can predict the missing content it has acquired a good understanding of personalized cardiac structure. Extensive experiments on a large-scale expert scanning dataset with 1.31 million samples demonstrate that our proposed sequence-aware paradigm can effectively reduce probe guidance errors compared to other advanced baseline methods. Our code will be released after acceptance.
>
---
#### [replaced 072] Learning Traffic Anomalies from Generative Models on Real-Time Observations
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01391v3](http://arxiv.org/pdf/2502.01391v3)**

> **作者:** Fotis I. Giasemis; Alexandros Sopasakis
>
> **摘要:** Accurate detection of traffic anomalies is crucial for effective urban traffic management and congestion mitigation. We use the Spatiotemporal Generative Adversarial Network (STGAN) framework combining Graph Neural Networks and Long Short-Term Memory networks to capture complex spatial and temporal dependencies in traffic data. We apply STGAN to real-time, minute-by-minute observations from 42 traffic cameras across Gothenburg, Sweden, collected over several months in 2020. The images are processed to compute a flow metric representing vehicle density, which serves as input for the model. Training is conducted on data from April to November 2020, and validation is performed on a separate dataset from November 14 to 23, 2020. Our results demonstrate that the model effectively detects traffic anomalies with high precision and low false positive rates. The detected anomalies include camera signal interruptions, visual artifacts, and extreme weather conditions affecting traffic flow.
>
---
#### [replaced 073] Self-Guidance: Boosting Flow and Diffusion Generation on Their Own
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05827v4](http://arxiv.org/pdf/2412.05827v4)**

> **作者:** Tiancheng Li; Weijian Luo; Zhiyang Chen; Liyuan Ma; Guo-Jun Qi
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Proper guidance strategies are essential to achieve high-quality generation results without retraining diffusion and flow-based text-to-image models. Existing guidance either requires specific training or strong inductive biases of diffusion model networks, potentially limiting their applications. Motivated by the observation that artifact outliers can be detected by a significant decline in the density from a noisier to a cleaner noise level, we propose Self-Guidance (SG), which improves the image quality by suppressing the generation of low-quality samples. SG only relies on the sampling probabilities of its own diffusion model at different noise levels with no need of any guidance-specific training. This makes it flexible to be used in a plug-and-play manner with other sampling algorithms. We also introduce a more efficient approximation of SG, named SG-prev, which reuses the output from the immediately previous diffusion step to avoid doubling sampling time. We conduct experiments on text-to-image and text-to-video generation with different architectures, including UNet and transformer models. With open-sourced diffusion models such as Stable Diffusion 3.5 and FLUX, SG exceeds existing algorithms on multiple metrics, including both FID and Human Preference Score. SG-prev also achieves strong results over both the baseline and the SG with only one forward pass. Moreover, we find that SG and SG-prev both have a surprisingly positive effect on the generation of physiologically correct human body structures such as hands, faces, and arms, showing their ability of eliminating human body artifacts with minimal efforts. We will release our code along with this paper.
>
---
#### [replaced 074] KeyNode-Driven Geometry Coding for Real-World Scanned Human Dynamic Mesh Compression
- **分类: cs.CV; cs.MM; eess.SP**

- **链接: [http://arxiv.org/pdf/2501.01717v2](http://arxiv.org/pdf/2501.01717v2)**

> **作者:** Huong Hoang; Truong Nguyen; Pamela Cosman
>
> **摘要:** The compression of real-world scanned 3D human dynamic meshes is an emerging research area, driven by applications such as telepresence, virtual reality, and 3D digital streaming. Unlike synthesized dynamic meshes with fixed topology, scanned dynamic meshes often not only have varying topology across frames but also scan defects such as holes and outliers, increasing the complexity of prediction and compression. Additionally, human meshes often combine rigid and non-rigid motions, making accurate prediction and encoding significantly more difficult compared to objects that exhibit purely rigid motion. To address these challenges, we propose a compression method designed for real-world scanned human dynamic meshes, leveraging embedded key nodes. The temporal motion of each vertex is formulated as a distance-weighted combination of transformations from neighboring key nodes, requiring the transmission of solely the key nodes' transformations. To enhance the quality of the KeyNode-driven prediction, we introduce an octree-based residual coding scheme and a Dual-direction prediction mode, which uses I-frames from both directions. Extensive experiments demonstrate that our method achieves significant improvements over the state-of-the-art, with an average bitrate savings of 58.43% across the evaluated sequences, particularly excelling at low bitrates.
>
---
