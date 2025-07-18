# 计算机视觉 cs.CV

- **最新发布 107 篇**

- **更新 57 篇**

## 最新发布

#### [new 001] Taming Diffusion Transformer for Real-Time Mobile Video Generation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于视频生成任务，旨在解决DiT模型计算成本高、难以实现实时移动设备应用的问题。通过压缩VAE、模型剪枝和对抗性步骤蒸馏等优化，实现高效实时视频生成。**

- **链接: [http://arxiv.org/pdf/2507.13343v1](http://arxiv.org/pdf/2507.13343v1)**

> **作者:** Yushu Wu; Yanyu Li; Anil Kag; Ivan Skorokhodov; Willi Menapace; Ke Ma; Arpit Sahni; Ju Hu; Aliaksandr Siarohin; Dhritiman Sagar; Yanzhi Wang; Sergey Tulyakov
>
> **备注:** 9 pages, 4 figures, 5 tables
>
> **摘要:** Diffusion Transformers (DiT) have shown strong performance in video generation tasks, but their high computational cost makes them impractical for resource-constrained devices like smartphones, and real-time generation is even more challenging. In this work, we propose a series of novel optimizations to significantly accelerate video generation and enable real-time performance on mobile platforms. First, we employ a highly compressed variational autoencoder (VAE) to reduce the dimensionality of the input data without sacrificing visual quality. Second, we introduce a KD-guided, sensitivity-aware tri-level pruning strategy to shrink the model size to suit mobile platform while preserving critical performance characteristics. Third, we develop an adversarial step distillation technique tailored for DiT, which allows us to reduce the number of inference steps to four. Combined, these optimizations enable our model to achieve over 10 frames per second (FPS) generation on an iPhone 16 Pro Max, demonstrating the feasibility of real-time, high-quality video generation on mobile devices.
>
---
#### [new 002] Diffuman4D: 4D Consistent Human View Synthesis from Sparse-View Videos with Spatio-Temporal Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于4D人体视图合成任务，解决稀疏视角视频生成视频时的时空不一致问题。通过滑动迭代去噪提升模型一致性，实现高质量新视角视频生成。**

- **链接: [http://arxiv.org/pdf/2507.13344v1](http://arxiv.org/pdf/2507.13344v1)**

> **作者:** Yudong Jin; Sida Peng; Xuan Wang; Tao Xie; Zhen Xu; Yifan Yang; Yujun Shen; Hujun Bao; Xiaowei Zhou
>
> **备注:** Project page: https://diffuman4d.github.io/
>
> **摘要:** This paper addresses the challenge of high-fidelity view synthesis of humans with sparse-view videos as input. Previous methods solve the issue of insufficient observation by leveraging 4D diffusion models to generate videos at novel viewpoints. However, the generated videos from these models often lack spatio-temporal consistency, thus degrading view synthesis quality. In this paper, we propose a novel sliding iterative denoising process to enhance the spatio-temporal consistency of the 4D diffusion model. Specifically, we define a latent grid in which each latent encodes the image, camera pose, and human pose for a certain viewpoint and timestamp, then alternately denoising the latent grid along spatial and temporal dimensions with a sliding window, and finally decode the videos at target viewpoints from the corresponding denoised latents. Through the iterative sliding, information flows sufficiently across the latent grid, allowing the diffusion model to obtain a large receptive field and thus enhance the 4D consistency of the output, while making the GPU memory consumption affordable. The experiments on the DNA-Rendering and ActorsHQ datasets demonstrate that our method is able to synthesize high-quality and consistent novel-view videos and significantly outperforms the existing approaches. See our project page for interactive demos and video results: https://diffuman4d.github.io/ .
>
---
#### [new 003] FIQ: Fundamental Question Generation with the Integration of Question Embeddings for Video Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频问答任务，旨在解决现有方法依赖事件中心标注导致场景理解不足的问题。通过生成基础问题对和引入视觉对齐模块，增强模型的推理能力。**

- **链接: [http://arxiv.org/pdf/2507.12816v1](http://arxiv.org/pdf/2507.12816v1)**

> **作者:** Ju-Young Oh; Ho-Joong Kim; Seong-Whan Lee
>
> **备注:** SMC 2025
>
> **摘要:** Video question answering (VQA) is a multimodal task that requires the interpretation of a video to answer a given question. Existing VQA methods primarily utilize question and answer (Q&A) pairs to learn the spatio-temporal characteristics of video content. However, these annotations are typically event-centric, which is not enough to capture the broader context of each video. The absence of essential details such as object types, spatial layouts, and descriptive attributes restricts the model to learning only a fragmented scene representation. This issue limits the model's capacity for generalization and higher-level reasoning. In this paper, we propose a fundamental question generation with the integration of question embeddings for video question answering (FIQ), a novel approach designed to strengthen the reasoning ability of the model by enhancing the fundamental understanding of videos. FIQ generates Q&A pairs based on descriptions extracted from videos, enriching the training data with fundamental scene information. Generated Q&A pairs enable the model to understand the primary context, leading to enhanced generalizability and reasoning ability. Furthermore, we incorporate a VQ-CAlign module that assists task-specific question embeddings with visual features, ensuring that essential domain-specific details are preserved to increase the adaptability of downstream tasks. Experiments on SUTD-TrafficQA demonstrate that our FIQ achieves state-of-the-art performance compared to existing baseline methods.
>
---
#### [new 004] Feature-Enhanced TResNet for Fine-Grained Food Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于细粒度食品图像分类任务，旨在解决相似形状食品间细微特征识别难题。通过改进TResNet模型，融合StyleRM和DCA技术提升分类精度。**

- **链接: [http://arxiv.org/pdf/2507.12828v1](http://arxiv.org/pdf/2507.12828v1)**

> **作者:** Lulu Liu; Zhiyong Xiao
>
> **摘要:** Food is not only a core component of humans' daily diets, but also an important carrier of cultural heritage and emotional bonds. With the development of technology, the need for accurate classification of food images has grown, which is crucial for a variety of application scenarios. However, existing Convolutional Neural Networks (CNNs) face significant challenges when dealing with fine-grained food images that are similar in shape but subtle in detail. To address this challenge, this study presents an innovative method for classifying food images, named Feature-Enhanced TResNet (FE-TResNet), specifically designed to address fine-grained food images and accurately capture subtle features within them. The FE-TResNet method is based on the TResNet model and integrates Style-based Recalibration Module (StyleRM) and Deep Channel-wise Attention (DCA) technologies to enhance feature extraction capabilities. In experimental validation on Chinese food image datasets ChineseFoodNet and CNFOOD-241, the FE-TResNet method significantly improved classification accuracy, achieving rates of 81.37% and 80.29%, respectively, demonstrating its effectiveness and superiority in fine-grained food image classification.
>
---
#### [new 005] Demographic-aware fine-grained classification of pediatric wrist fractures
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在提高儿童腕部骨折的诊断准确性。针对数据有限的问题，结合X光图像与患者信息，采用细粒度学习方法提升识别效果。**

- **链接: [http://arxiv.org/pdf/2507.12964v1](http://arxiv.org/pdf/2507.12964v1)**

> **作者:** Ammar Ahmed; Ali Shariq Imran; Zenun Kastrati; Sher Muhammad Daudpota
>
> **摘要:** Wrist pathologies are frequently observed, particularly among children who constitute the majority of fracture cases. However, diagnosing these conditions is time-consuming and requires specialized expertise. Computer vision presents a promising avenue, contingent upon the availability of extensive datasets, a notable challenge in medical imaging. Therefore, reliance solely on one modality, such as images, proves inadequate, especially in an era of diverse and plentiful data types. In this study, we employ a multifaceted approach to address the challenge of recognizing wrist pathologies using an extremely limited dataset. Initially, we approach the problem as a fine-grained recognition task, aiming to identify subtle X-ray pathologies that conventional CNNs overlook. Secondly, we enhance network performance by fusing patient metadata with X-ray images. Thirdly, rather than pre-training on a coarse-grained dataset like ImageNet, we utilize weights trained on a fine-grained dataset. While metadata integration has been used in other medical domains, this is a novel application for wrist pathologies. Our results show that a fine-grained strategy and metadata integration improve diagnostic accuracy by 2% with a limited dataset and by over 10% with a larger fracture-focused dataset.
>
---
#### [new 006] Funnel-HOI: Top-Down Perception for Zero-Shot HOI Detection
- **分类: cs.CV**

- **简介: 该论文属于人-物交互检测任务，解决数据稀疏导致的零样本识别问题。提出Funnel-HOI框架，通过自顶向下机制和新型损失函数提升交互表示与分类效果。**

- **链接: [http://arxiv.org/pdf/2507.12628v1](http://arxiv.org/pdf/2507.12628v1)**

> **作者:** Sandipan Sarma; Agney Talwarr; Arijit Sur
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Human-object interaction detection (HOID) refers to localizing interactive human-object pairs in images and identifying the interactions. Since there could be an exponential number of object-action combinations, labeled data is limited - leading to a long-tail distribution problem. Recently, zero-shot learning emerged as a solution, with end-to-end transformer-based object detectors adapted for HOID becoming successful frameworks. However, their primary focus is designing improved decoders for learning entangled or disentangled interpretations of interactions. We advocate that HOI-specific cues must be anticipated at the encoder stage itself to obtain a stronger scene interpretation. Consequently, we build a top-down framework named Funnel-HOI inspired by the human tendency to grasp well-defined concepts first and then associate them with abstract concepts during scene understanding. We first probe an image for the presence of objects (well-defined concepts) and then probe for actions (abstract concepts) associated with them. A novel asymmetric co-attention mechanism mines these cues utilizing multimodal information (incorporating zero-shot capabilities) and yields stronger interaction representations at the encoder level. Furthermore, a novel loss is devised that considers objectaction relatedness and regulates misclassification penalty better than existing loss functions for guiding the interaction classifier. Extensive experiments on the HICO-DET and V-COCO datasets across fully-supervised and six zero-shot settings reveal our state-of-the-art performance, with up to 12.4% and 8.4% gains for unseen and rare HOI categories, respectively.
>
---
#### [new 007] CT-ScanGaze: A Dataset and Baselines for 3D Volumetric Scanpath Modeling
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决CT图像眼动数据不足及2D扫描路径预测的局限性。提出CT-ScanGaze数据集和3D扫描路径模型CT-Searcher。**

- **链接: [http://arxiv.org/pdf/2507.12591v1](http://arxiv.org/pdf/2507.12591v1)**

> **作者:** Trong-Thang Pham; Akash Awasthi; Saba Khan; Esteban Duran Marti; Tien-Phat Nguyen; Khoa Vo; Minh Tran; Ngoc Son Nguyen; Cuong Tran Van; Yuki Ikebe; Anh Totti Nguyen; Anh Nguyen; Zhigang Deng; Carol C. Wu; Hien Van Nguyen; Ngan Le
>
> **备注:** ICCV 2025
>
> **摘要:** Understanding radiologists' eye movement during Computed Tomography (CT) reading is crucial for developing effective interpretable computer-aided diagnosis systems. However, CT research in this area has been limited by the lack of publicly available eye-tracking datasets and the three-dimensional complexity of CT volumes. To address these challenges, we present the first publicly available eye gaze dataset on CT, called CT-ScanGaze. Then, we introduce CT-Searcher, a novel 3D scanpath predictor designed specifically to process CT volumes and generate radiologist-like 3D fixation sequences, overcoming the limitations of current scanpath predictors that only handle 2D inputs. Since deep learning models benefit from a pretraining step, we develop a pipeline that converts existing 2D gaze datasets into 3D gaze data to pretrain CT-Searcher. Through both qualitative and quantitative evaluations on CT-ScanGaze, we demonstrate the effectiveness of our approach and provide a comprehensive assessment framework for 3D scanpath prediction in medical imaging.
>
---
#### [new 008] Revisiting Reliability in the Reasoning-based Pose Estimation Benchmark
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人体姿态估计任务，针对RPE基准的可靠性问题，解决数据不一致和质量缺陷，通过精修标注提升评估一致性与可复现性。**

- **链接: [http://arxiv.org/pdf/2507.13314v1](http://arxiv.org/pdf/2507.13314v1)**

> **作者:** Junsu Kim; Naeun Kim; Jaeho Lee; Incheol Park; Dongyoon Han; Seungryul Baek
>
> **备注:** To be presented as a poster at MMFM 2025
>
> **摘要:** The reasoning-based pose estimation (RPE) benchmark has emerged as a widely adopted evaluation standard for pose-aware multimodal large language models (MLLMs). Despite its significance, we identified critical reproducibility and benchmark-quality issues that hinder fair and consistent quantitative evaluations. Most notably, the benchmark utilizes different image indices from those of the original 3DPW dataset, forcing researchers into tedious and error-prone manual matching processes to obtain accurate ground-truth (GT) annotations for quantitative metrics (\eg, MPJPE, PA-MPJPE). Furthermore, our analysis reveals several inherent benchmark-quality limitations, including significant image redundancy, scenario imbalance, overly simplistic poses, and ambiguous textual descriptions, collectively undermining reliable evaluations across diverse scenarios. To alleviate manual effort and enhance reproducibility, we carefully refined the GT annotations through meticulous visual matching and publicly release these refined annotations as an open-source resource, thereby promoting consistent quantitative evaluations and facilitating future advancements in human pose-aware multimodal reasoning.
>
---
#### [new 009] SOD-YOLO: Enhancing YOLO-Based Detection of Small Objects in UAV Imagery
- **分类: cs.CV; I.4**

- **简介: 该论文属于目标检测任务，旨在解决无人机图像中小目标检测困难的问题。通过改进YOLOv8模型，引入ASF机制、P2层和Soft-NMS，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.12727v1](http://arxiv.org/pdf/2507.12727v1)**

> **作者:** Peijun Wang; Jinhua Zhao
>
> **摘要:** Small object detection remains a challenging problem in the field of object detection. To address this challenge, we propose an enhanced YOLOv8-based model, SOD-YOLO. This model integrates an ASF mechanism in the neck to enhance multi-scale feature fusion, adds a Small Object Detection Layer (named P2) to provide higher-resolution feature maps for better small object detection, and employs Soft-NMS to refine confidence scores and retain true positives. Experimental results demonstrate that SOD-YOLO significantly improves detection performance, achieving a 36.1% increase in mAP$_{50:95}$ and 20.6% increase in mAP$_{50}$ on the VisDrone2019-DET dataset compared to the baseline model. These enhancements make SOD-YOLO a practical and efficient solution for small object detection in UAV imagery. Our source code, hyper-parameters, and model weights are available at https://github.com/iamwangxiaobai/SOD-YOLO.
>
---
#### [new 010] Decoupled PROB: Decoupled Query Initialization Tasks and Objectness-Class Learning for Open World Object Detection
- **分类: cs.CV**

- **简介: 该论文属于开放世界目标检测任务，解决对象性与分类学习冲突问题。提出Decoupled PROB模型，通过ETOP和TDQI提升性能。**

- **链接: [http://arxiv.org/pdf/2507.13085v1](http://arxiv.org/pdf/2507.13085v1)**

> **作者:** Riku Inoue; Masamitsu Tsuchiya; Yuji Yasui
>
> **备注:** This paper has been accepted to WACV 2025 (Tucson, Arizona, USA), February 28-March 4 2025
>
> **摘要:** Open World Object Detection (OWOD) is a challenging computer vision task that extends standard object detection by (1) detecting and classifying unknown objects without supervision, and (2) incrementally learning new object classes without forgetting previously learned ones. The absence of ground truths for unknown objects makes OWOD tasks particularly challenging. Many methods have addressed this by using pseudo-labels for unknown objects. The recently proposed Probabilistic Objectness transformer-based open-world detector (PROB) is a state-of-the-art model that does not require pseudo-labels for unknown objects, as it predicts probabilistic objectness. However, this method faces issues with learning conflicts between objectness and class predictions. To address this issue and further enhance performance, we propose a novel model, Decoupled PROB. Decoupled PROB introduces Early Termination of Objectness Prediction (ETOP) to stop objectness predictions at appropriate layers in the decoder, resolving the learning conflicts between class and objectness predictions in PROB. Additionally, we introduce Task-Decoupled Query Initialization (TDQI), which efficiently extracts features of known and unknown objects, thereby improving performance. TDQI is a query initialization method that combines query selection and learnable queries, and it is a module that can be easily integrated into existing DETR-based OWOD models. Extensive experiments on OWOD benchmarks demonstrate that Decoupled PROB surpasses all existing methods across several metrics, significantly improving performance.
>
---
#### [new 011] AutoPartGen: Autogressive 3D Part Generation and Discovery
- **分类: cs.CV**

- **简介: 该论文属于3D生成任务，解决如何自动生成和发现物体的3D部件问题。工作是提出AutoPartGen模型，通过自回归方式生成高质量的3D部件并自动确定其数量与类型。**

- **链接: [http://arxiv.org/pdf/2507.13346v1](http://arxiv.org/pdf/2507.13346v1)**

> **作者:** Minghao Chen; Jianyuan Wang; Roman Shapovalov; Tom Monnier; Hyunyoung Jung; Dilin Wang; Rakesh Ranjan; Iro Laina; Andrea Vedaldi
>
> **备注:** Project page: https://silent-chen.github.io/AutoPartGen/
>
> **摘要:** We introduce AutoPartGen, a model that generates objects composed of 3D parts in an autoregressive manner. This model can take as input an image of an object, 2D masks of the object's parts, or an existing 3D object, and generate a corresponding compositional 3D reconstruction. Our approach builds upon 3DShape2VecSet, a recent latent 3D representation with powerful geometric expressiveness. We observe that this latent space exhibits strong compositional properties, making it particularly well-suited for part-based generation tasks. Specifically, AutoPartGen generates object parts autoregressively, predicting one part at a time while conditioning on previously generated parts and additional inputs, such as 2D images, masks, or 3D objects. This process continues until the model decides that all parts have been generated, thus determining automatically the type and number of parts. The resulting parts can be seamlessly assembled into coherent objects or scenes without requiring additional optimization. We evaluate both the overall 3D generation capabilities and the part-level generation quality of AutoPartGen, demonstrating that it achieves state-of-the-art performance in 3D part generation.
>
---
#### [new 012] MCoT-RE: Multi-Faceted Chain-of-Thought and Re-Ranking for Training-Free Zero-Shot Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，解决训练-free 零样本组合图像检索问题。提出MCoT-RE框架，通过多方面思维链和重排序提升检索精度。**

- **链接: [http://arxiv.org/pdf/2507.12819v1](http://arxiv.org/pdf/2507.12819v1)**

> **作者:** Jeong-Woo Park; Seong-Whan Lee
>
> **备注:** 6 pages, 4 figures, 2025 IEEE International Conference on Systems, Man, and Cybernetics
>
> **摘要:** Composed Image Retrieval (CIR) is the task of retrieving a target image from a gallery using a composed query consisting of a reference image and a modification text. Among various CIR approaches, training-free zero-shot methods based on pre-trained models are cost-effective but still face notable limitations. For example, sequential VLM-LLM pipelines process each modality independently, which often results in information loss and limits cross-modal interaction. In contrast, methods based on multimodal large language models (MLLMs) often focus exclusively on applying changes indicated by the text, without fully utilizing the contextual visual information from the reference image. To address these issues, we propose multi-faceted Chain-of-Thought with re-ranking (MCoT-RE), a training-free zero-shot CIR framework. MCoT-RE utilizes multi-faceted Chain-of-Thought to guide the MLLM to balance explicit modifications and contextual visual cues, generating two distinct captions: one focused on modification and the other integrating comprehensive visual-textual context. The first caption is used to filter candidate images. Subsequently, we combine these two captions and the reference image to perform multi-grained re-ranking. This two-stage approach facilitates precise retrieval by aligning with the textual modification instructions while preserving the visual context of the reference image. Through extensive experiments, MCoT-RE achieves state-of-the-art results among training-free methods, yielding improvements of up to 6.24% in Recall@10 on FashionIQ and 8.58% in Recall@1 on CIRR.
>
---
#### [new 013] VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决视觉token冗余问题。通过动态调整图像分辨率和强化学习，提升OCR任务性能并减少计算资源消耗。**

- **链接: [http://arxiv.org/pdf/2507.13348v1](http://arxiv.org/pdf/2507.13348v1)**

> **作者:** Senqiao Yang; Junyi Li; Xin Lai; Bei Yu; Hengshuang Zhao; Jiaya Jia
>
> **备注:** Code and models are available at https://github.com/dvlab-research/VisionThink
>
> **摘要:** Recent advancements in vision-language models (VLMs) have improved performance by increasing the number of visual tokens, which are often significantly longer than text tokens. However, we observe that most real-world scenarios do not require such an extensive number of visual tokens. While the performance drops significantly in a small subset of OCR-related tasks, models still perform accurately in most other general VQA tasks with only 1/4 resolution. Therefore, we propose to dynamically process distinct samples with different resolutions, and present a new paradigm for visual token compression, namely, VisionThink. It starts with a downsampled image and smartly decides whether it is sufficient for problem solving. Otherwise, the model could output a special token to request the higher-resolution image. Compared to existing Efficient VLM methods that compress tokens using fixed pruning ratios or thresholds, VisionThink autonomously decides whether to compress tokens case by case. As a result, it demonstrates strong fine-grained visual understanding capability on OCR-related tasks, and meanwhile saves substantial visual tokens on simpler tasks. We adopt reinforcement learning and propose the LLM-as-Judge strategy to successfully apply RL to general VQA tasks. Moreover, we carefully design a reward function and penalty mechanism to achieve a stable and reasonable image resize call ratio. Extensive experiments demonstrate the superiority, efficiency, and effectiveness of our method. Our code is available at https://github.com/dvlab-research/VisionThink.
>
---
#### [new 014] FantasyPortrait: Enhancing Multi-Character Portrait Animation with Expression-Augmented Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于面部动画生成任务，解决单图多角色表情动画生成问题。提出FantasyPortrait框架，采用扩散Transformer和掩码交叉注意力机制，提升动画质量与多角色协同性。**

- **链接: [http://arxiv.org/pdf/2507.12956v1](http://arxiv.org/pdf/2507.12956v1)**

> **作者:** Qiang Wang; Mengchao Wang; Fan Jiang; Yaqi Fan; Yonggang Qi; Mu Xu
>
> **备注:** https://fantasy-amap.github.io/fantasy-portrait/
>
> **摘要:** Producing expressive facial animations from static images is a challenging task. Prior methods relying on explicit geometric priors (e.g., facial landmarks or 3DMM) often suffer from artifacts in cross reenactment and struggle to capture subtle emotions. Furthermore, existing approaches lack support for multi-character animation, as driving features from different individuals frequently interfere with one another, complicating the task. To address these challenges, we propose FantasyPortrait, a diffusion transformer based framework capable of generating high-fidelity and emotion-rich animations for both single- and multi-character scenarios. Our method introduces an expression-augmented learning strategy that utilizes implicit representations to capture identity-agnostic facial dynamics, enhancing the model's ability to render fine-grained emotions. For multi-character control, we design a masked cross-attention mechanism that ensures independent yet coordinated expression generation, effectively preventing feature interference. To advance research in this area, we propose the Multi-Expr dataset and ExprBench, which are specifically designed datasets and benchmarks for training and evaluating multi-character portrait animations. Extensive experiments demonstrate that FantasyPortrait significantly outperforms state-of-the-art methods in both quantitative metrics and qualitative evaluations, excelling particularly in challenging cross reenactment and multi-character contexts. Our project page is https://fantasy-amap.github.io/fantasy-portrait/.
>
---
#### [new 015] LoViC: Efficient Long Video Generation with Context Compression
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决长视频生成中自注意力复杂度高的问题。提出LoViC框架，通过上下文压缩实现高效、连贯的长视频生成。**

- **链接: [http://arxiv.org/pdf/2507.12952v1](http://arxiv.org/pdf/2507.12952v1)**

> **作者:** Jiaxiu Jiang; Wenbo Li; Jingjing Ren; Yuping Qiu; Yong Guo; Xiaogang Xu; Han Wu; Wangmeng Zuo
>
> **备注:** Project page: https://jiangjiaxiu.github.io/lovic/
>
> **摘要:** Despite recent advances in diffusion transformers (DiTs) for text-to-video generation, scaling to long-duration content remains challenging due to the quadratic complexity of self-attention. While prior efforts -- such as sparse attention and temporally autoregressive models -- offer partial relief, they often compromise temporal coherence or scalability. We introduce LoViC, a DiT-based framework trained on million-scale open-domain videos, designed to produce long, coherent videos through a segment-wise generation process. At the core of our approach is FlexFormer, an expressive autoencoder that jointly compresses video and text into unified latent representations. It supports variable-length inputs with linearly adjustable compression rates, enabled by a single query token design based on the Q-Former architecture. Additionally, by encoding temporal context through position-aware mechanisms, our model seamlessly supports prediction, retradiction, interpolation, and multi-shot generation within a unified paradigm. Extensive experiments across diverse tasks validate the effectiveness and versatility of our approach.
>
---
#### [new 016] Efficient Adaptation of Pre-trained Vision Transformer underpinned by Approximately Orthogonal Fine-Tuning Strategy
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉Transformer的参数高效微调任务，旨在提升模型泛化能力。通过引入近正交微调策略，使低秩适配矩阵更接近预训练主干的特性，从而增强模型性能。**

- **链接: [http://arxiv.org/pdf/2507.13260v1](http://arxiv.org/pdf/2507.13260v1)**

> **作者:** Yiting Yang; Hao Luo; Yuan Sun; Qingsen Yan; Haokui Zhang; Wei Dong; Guoqing Wang; Peng Wang; Yang Yang; Hengtao Shen
>
> **备注:** This paper is accepted by ICCV 2025
>
> **摘要:** A prevalent approach in Parameter-Efficient Fine-Tuning (PEFT) of pre-trained Vision Transformers (ViT) involves freezing the majority of the backbone parameters and solely learning low-rank adaptation weight matrices to accommodate downstream tasks. These low-rank matrices are commonly derived through the multiplication structure of down-projection and up-projection matrices, exemplified by methods such as LoRA and Adapter. In this work, we observe an approximate orthogonality among any two row or column vectors within any weight matrix of the backbone parameters; however, this property is absent in the vectors of the down/up-projection matrices. Approximate orthogonality implies a reduction in the upper bound of the model's generalization error, signifying that the model possesses enhanced generalization capability. If the fine-tuned down/up-projection matrices were to exhibit this same property as the pre-trained backbone matrices, could the generalization capability of fine-tuned ViTs be further augmented? To address this question, we propose an Approximately Orthogonal Fine-Tuning (AOFT) strategy for representing the low-rank weight matrices. This strategy employs a single learnable vector to generate a set of approximately orthogonal vectors, which form the down/up-projection matrices, thereby aligning the properties of these matrices with those of the backbone. Extensive experimental results demonstrate that our method achieves competitive performance across a range of downstream image classification tasks, confirming the efficacy of the enhanced generalization capability embedded in the down/up-projection matrices.
>
---
#### [new 017] Spatially Grounded Explanations in Vision Language Models for Document Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文档视觉问答任务，旨在提升模型解释的透明度与可复现性。提出EaGERS方法，通过空间定位增强模型推理过程的可解释性。**

- **链接: [http://arxiv.org/pdf/2507.12490v1](http://arxiv.org/pdf/2507.12490v1)**

> **作者:** Maximiliano Hormazábal Lagos; Héctor Cerezo-Costas; Dimosthenis Karatzas
>
> **备注:** This work has been accepted for presentation at the 16th Conference and Labs of the Evaluation Forum (CLEF 2025) and will be published in the proceedings by Springer in the Lecture Notes in Computer Science (LNCS) series. Please cite the published version when available
>
> **摘要:** We introduce EaGERS, a fully training-free and model-agnostic pipeline that (1) generates natural language rationales via a vision language model, (2) grounds these rationales to spatial sub-regions by computing multimodal embedding similarities over a configurable grid with majority voting, and (3) restricts the generation of responses only from the relevant regions selected in the masked image. Experiments on the DocVQA dataset demonstrate that our best configuration not only outperforms the base model on exact match accuracy and Average Normalized Levenshtein Similarity metrics but also enhances transparency and reproducibility in DocVQA without additional model fine-tuning.
>
---
#### [new 018] A Privacy-Preserving Semantic-Segmentation Method Using Domain-Adaptation Technique
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于图像语义分割任务，旨在解决加密图像训练中的隐私与精度问题。通过域适应技术改进ViT结构，实现隐私保护与高精度分割。**

- **链接: [http://arxiv.org/pdf/2507.12730v1](http://arxiv.org/pdf/2507.12730v1)**

> **作者:** Homare Sueyoshi; Kiyoshi Nishikawa; Hitoshi Kiya
>
> **备注:** 4 pages, 5 figures, 1 table. Accepted to GCCE 2025
>
> **摘要:** We propose a privacy-preserving semantic-segmentation method for applying perceptual encryption to images used for model training in addition to test images. This method also provides almost the same accuracy as models without any encryption. The above performance is achieved using a domain-adaptation technique on the embedding structure of the Vision Transformer (ViT). The effectiveness of the proposed method was experimentally confirmed in terms of the accuracy of semantic segmentation when using a powerful semantic-segmentation model with ViT called Segmentation Transformer.
>
---
#### [new 019] Reconstruct, Inpaint, Finetune: Dynamic Novel-view Synthesis from Monocular Videos
- **分类: cs.CV**

- **简介: 该论文属于动态场景的新型视角合成任务，解决单目视频生成新视角图像的问题。提出方法结合重建、修复和微调，提升合成效果与效率。**

- **链接: [http://arxiv.org/pdf/2507.12646v1](http://arxiv.org/pdf/2507.12646v1)**

> **作者:** Kaihua Chen; Tarasha Khurana; Deva Ramanan
>
> **备注:** Project page: https://cog-nvs.github.io/
>
> **摘要:** We explore novel-view synthesis for dynamic scenes from monocular videos. Prior approaches rely on costly test-time optimization of 4D representations or do not preserve scene geometry when trained in a feed-forward manner. Our approach is based on three key insights: (1) covisible pixels (that are visible in both the input and target views) can be rendered by first reconstructing the dynamic 3D scene and rendering the reconstruction from the novel-views and (2) hidden pixels in novel views can be "inpainted" with feed-forward 2D video diffusion models. Notably, our video inpainting diffusion model (CogNVS) can be self-supervised from 2D videos, allowing us to train it on a large corpus of in-the-wild videos. This in turn allows for (3) CogNVS to be applied zero-shot to novel test videos via test-time finetuning. We empirically verify that CogNVS outperforms almost all prior art for novel-view synthesis of dynamic scenes from monocular videos.
>
---
#### [new 020] Label-Consistent Dataset Distillation with Detector-Guided Refinement
- **分类: cs.CV**

- **简介: 该论文属于数据集蒸馏任务，旨在解决生成数据标签不一致和结构不足的问题。通过引入检测器引导的优化框架，提升生成数据的质量与一致性。**

- **链接: [http://arxiv.org/pdf/2507.13074v1](http://arxiv.org/pdf/2507.13074v1)**

> **作者:** Yawen Zou; Guang Li; Zi Wang; Chunzhi Gu; Chao Zhang
>
> **摘要:** Dataset distillation (DD) aims to generate a compact yet informative dataset that achieves performance comparable to the original dataset, thereby reducing demands on storage and computational resources. Although diffusion models have made significant progress in dataset distillation, the generated surrogate datasets often contain samples with label inconsistencies or insufficient structural detail, leading to suboptimal downstream performance. To address these issues, we propose a detector-guided dataset distillation framework that explicitly leverages a pre-trained detector to identify and refine anomalous synthetic samples, thereby ensuring label consistency and improving image quality. Specifically, a detector model trained on the original dataset is employed to identify anomalous images exhibiting label mismatches or low classification confidence. For each defective image, multiple candidates are generated using a pre-trained diffusion model conditioned on the corresponding image prototype and label. The optimal candidate is then selected by jointly considering the detector's confidence score and dissimilarity to existing qualified synthetic samples, thereby ensuring both label accuracy and intra-class diversity. Experimental results demonstrate that our method can synthesize high-quality representative images with richer details, achieving state-of-the-art performance on the validation set.
>
---
#### [new 021] Deep Learning-Based Fetal Lung Segmentation from Diffusion-weighted MRI Images and Lung Maturity Evaluation for Fetal Growth Restriction
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决胎儿肺部成熟度评估的自动化问题。通过深度学习实现肺部分割与IVIM参数分析，提升临床诊断效率。**

- **链接: [http://arxiv.org/pdf/2507.13106v1](http://arxiv.org/pdf/2507.13106v1)**

> **作者:** Zhennan Xiao; Katharine Brudkiewicz; Zhen Yuan; Rosalind Aughwane; Magdalena Sokolska; Joanna Chappell; Trevor Gaunt; Anna L. David; Andrew P. King; Andrew Melbourne
>
> **摘要:** Fetal lung maturity is a critical indicator for predicting neonatal outcomes and the need for post-natal intervention, especially for pregnancies affected by fetal growth restriction. Intra-voxel incoherent motion analysis has shown promising results for non-invasive assessment of fetal lung development, but its reliance on manual segmentation is time-consuming, thus limiting its clinical applicability. In this work, we present an automated lung maturity evaluation pipeline for diffusion-weighted magnetic resonance images that consists of a deep learning-based fetal lung segmentation model and a model-fitting lung maturity assessment. A 3D nnU-Net model was trained on manually segmented images selected from the baseline frames of 4D diffusion-weighted MRI scans. The segmentation model demonstrated robust performance, yielding a mean Dice coefficient of 82.14%. Next, voxel-wise model fitting was performed based on both the nnU-Net-predicted and manual lung segmentations to quantify IVIM parameters reflecting tissue microstructure and perfusion. The results suggested no differences between the two. Our work shows that a fully automated pipeline is possible for supporting fetal lung maturity assessment and clinical decision-making.
>
---
#### [new 022] Local Representative Token Guided Merging for Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决稳定扩散模型生成效率低的问题。提出ReToM方法通过局部代表性令牌合并，提升效率并保持图像质量。**

- **链接: [http://arxiv.org/pdf/2507.12771v1](http://arxiv.org/pdf/2507.12771v1)**

> **作者:** Min-Jeong Lee; Hee-Dong Kim; Seong-Whan Lee
>
> **备注:** 6 pages
>
> **摘要:** Stable diffusion is an outstanding image generation model for text-to-image, but its time-consuming generation process remains a challenge due to the quadratic complexity of attention operations. Recent token merging methods improve efficiency by reducing the number of tokens during attention operations, but often overlook the characteristics of attention-based image generation models, limiting their effectiveness. In this paper, we propose local representative token guided merging (ReToM), a novel token merging strategy applicable to any attention mechanism in image generation. To merge tokens based on various contextual information, ReToM defines local boundaries as windows within attention inputs and adjusts window sizes. Furthermore, we introduce a representative token, which represents the most representative token per window by computing similarity at a specific timestep and selecting the token with the highest average similarity. This approach preserves the most salient local features while minimizing computational overhead. Experimental results show that ReToM achieves a 6.2% improvement in FID and higher CLIP scores compared to the baseline, while maintaining comparable inference time. We empirically demonstrate that ReToM is effective in balancing visual quality and computational efficiency.
>
---
#### [new 023] Think-Before-Draw: Decomposing Emotion Semantics & Fine-Grained Controllable Expressive Talking Head Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于情感对话头生成任务，解决文本驱动方法难以自然表达情绪的问题。通过引入思维链和渐进去噪策略，提升情感表达的细腻度与真实性。**

- **链接: [http://arxiv.org/pdf/2507.12761v1](http://arxiv.org/pdf/2507.12761v1)**

> **作者:** Hanlei Shi; Leyuan Qu; Yu Liu; Di Gao; Yuhua Zheng; Taihao Li
>
> **摘要:** Emotional talking-head generation has emerged as a pivotal research area at the intersection of computer vision and multimodal artificial intelligence, with its core value lying in enhancing human-computer interaction through immersive and empathetic engagement.With the advancement of multimodal large language models, the driving signals for emotional talking-head generation has shifted from audio and video to more flexible text. However, current text-driven methods rely on predefined discrete emotion label texts, oversimplifying the dynamic complexity of real facial muscle movements and thus failing to achieve natural emotional expressiveness.This study proposes the Think-Before-Draw framework to address two key challenges: (1) In-depth semantic parsing of emotions--by innovatively introducing Chain-of-Thought (CoT), abstract emotion labels are transformed into physiologically grounded facial muscle movement descriptions, enabling the mapping from high-level semantics to actionable motion features; and (2) Fine-grained expressiveness optimization--inspired by artists' portrait painting process, a progressive guidance denoising strategy is proposed, employing a "global emotion localization--local muscle control" mechanism to refine micro-expression dynamics in generated videos.Our experiments demonstrate that our approach achieves state-of-the-art performance on widely-used benchmarks, including MEAD and HDTF. Additionally, we collected a set of portrait images to evaluate our model's zero-shot generation capability.
>
---
#### [new 024] Continuous Marine Tracking via Autonomous UAV Handoff
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于海洋动物跟踪任务，解决单无人机续航与环境干扰问题，通过多无人机协同实现连续跟踪。**

- **链接: [http://arxiv.org/pdf/2507.12763v1](http://arxiv.org/pdf/2507.12763v1)**

> **作者:** Heegyeong Kim; Alice James; Avishkar Seth; Endrowednes Kuantama; Jane Williamson; Yimeng Feng; Richard Han
>
> **备注:** 6 pages, 5 figures, to be published in DroNet '25: Proceedings of the 10th Workshop on Micro Aerial Vehicle Networks, Systems, and Applications
>
> **摘要:** This paper introduces an autonomous UAV vision system for continuous, real-time tracking of marine animals, specifically sharks, in dynamic marine environments. The system integrates an onboard computer with a stabilised RGB-D camera and a custom-trained OSTrack pipeline, enabling visual identification under challenging lighting, occlusion, and sea-state conditions. A key innovation is the inter-UAV handoff protocol, which enables seamless transfer of tracking responsibilities between drones, extending operational coverage beyond single-drone battery limitations. Performance is evaluated on a curated shark dataset of 5,200 frames, achieving a tracking success rate of 81.9\% during real-time flight control at 100 Hz, and robustness to occlusion, illumination variation, and background clutter. We present a seamless UAV handoff framework, where target transfer is attempted via high-confidence feature matching, achieving 82.9\% target coverage. These results confirm the viability of coordinated UAV operations for extended marine tracking and lay the groundwork for scalable, autonomous monitoring.
>
---
#### [new 025] FAR-Net: Multi-Stage Fusion Network with Enhanced Semantic Alignment and Adaptive Reconciliation for Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，解决组合图像检索（CIR）中多模态融合问题。提出FAR-Net框架，结合语义对齐与自适应校正模块，提升检索效果。**

- **链接: [http://arxiv.org/pdf/2507.12823v1](http://arxiv.org/pdf/2507.12823v1)**

> **作者:** Jeong-Woo Park; Young-Eun Kim; Seong-Whan Lee
>
> **备注:** 6 pages, 3 figures, 3 tables
>
> **摘要:** Composed image retrieval (CIR) is a vision language task that retrieves a target image using a reference image and modification text, enabling intuitive specification of desired changes. While effectively fusing visual and textual modalities is crucial, existing methods typically adopt either early or late fusion. Early fusion tends to excessively focus on explicitly mentioned textual details and neglect visual context, whereas late fusion struggles to capture fine-grained semantic alignments between image regions and textual tokens. To address these issues, we propose FAR-Net, a multi-stage fusion framework designed with enhanced semantic alignment and adaptive reconciliation, integrating two complementary modules. The enhanced semantic alignment module (ESAM) employs late fusion with cross-attention to capture fine-grained semantic relationships, while the adaptive reconciliation module (ARM) applies early fusion with uncertainty embeddings to enhance robustness and adaptability. Experiments on CIRR and FashionIQ show consistent performance gains, improving Recall@1 by up to 2.4% and Recall@50 by 1.04% over existing state-of-the-art methods, empirically demonstrating that FAR Net provides a robust and scalable solution to CIR tasks.
>
---
#### [new 026] Argus: Leveraging Multiview Images for Improved 3-D Scene Understanding With Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D场景理解任务，旨在解决3D点云信息丢失问题。通过融合多视角图像与相机位姿，构建更全面的3D场景表示。**

- **链接: [http://arxiv.org/pdf/2507.12916v1](http://arxiv.org/pdf/2507.12916v1)**

> **作者:** Yifan Xu; Chao Zhang; Hanqi Jiang; Xiaoyan Wang; Ruifei Ma; Yiwei Li; Zihao Wu; Zeju Li; Xiangde Liu
>
> **备注:** Accepted by TNNLS2025
>
> **摘要:** Advancements in foundation models have made it possible to conduct applications in various downstream tasks. Especially, the new era has witnessed a remarkable capability to extend Large Language Models (LLMs) for tackling tasks of 3D scene understanding. Current methods rely heavily on 3D point clouds, but the 3D point cloud reconstruction of an indoor scene often results in information loss. Some textureless planes or repetitive patterns are prone to omission and manifest as voids within the reconstructed 3D point clouds. Besides, objects with complex structures tend to introduce distortion of details caused by misalignments between the captured images and the dense reconstructed point clouds. 2D multi-view images present visual consistency with 3D point clouds and provide more detailed representations of scene components, which can naturally compensate for these deficiencies. Based on these insights, we propose Argus, a novel 3D multimodal framework that leverages multi-view images for enhanced 3D scene understanding with LLMs. In general, Argus can be treated as a 3D Large Multimodal Foundation Model (3D-LMM) since it takes various modalities as input(text instructions, 2D multi-view images, and 3D point clouds) and expands the capability of LLMs to tackle 3D tasks. Argus involves fusing and integrating multi-view images and camera poses into view-as-scene features, which interact with the 3D features to create comprehensive and detailed 3D-aware scene embeddings. Our approach compensates for the information loss while reconstructing 3D point clouds and helps LLMs better understand the 3D world. Extensive experiments demonstrate that our method outperforms existing 3D-LMMs in various downstream tasks.
>
---
#### [new 027] MS-DGCNN++: A Multi-Scale Fusion Dynamic Graph Neural Network with Biological Knowledge Integration for LiDAR Tree Species Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于点云分类任务，解决森林LiDAR数据中树种识别问题。提出MS-DGCNN++模型，通过多尺度融合和生物知识整合提升分类精度。**

- **链接: [http://arxiv.org/pdf/2507.12602v1](http://arxiv.org/pdf/2507.12602v1)**

> **作者:** Said Ohamouddou; Abdellatif El Afia; Hanaa El Afia; Raddouane Chiheb
>
> **摘要:** Tree species classification from terrestrial LiDAR point clouds is challenging because of the complex multi-scale geometric structures in forest environments. Existing approaches using multi-scale dynamic graph convolutional neural networks (MS-DGCNN) employ parallel multi-scale processing, which fails to capture the semantic relationships between the hierarchical levels of the tree architecture. We present MS-DGCNN++, a hierarchical multiscale fusion dynamic graph convolutional network that uses semantically meaningful feature extraction at local, branch, and canopy scales with cross-scale information propagation. Our method employs scale-specific feature engineering, including standard geometric features for the local scale, normalized relative vectors for the branch scale, and distance information for the canopy scale. This hierarchical approach replaces uniform parallel processing with semantically differentiated representations that are aligned with the natural tree structure. Under the same proposed tree species data augmentation strategy for all experiments, MS-DGCNN++ achieved an accuracy of 94.96 \% on STPCTLS, outperforming DGCNN, MS-DGCNN, and the state-of-the-art model PPT. On FOR-species20K, it achieves 67.25\% accuracy (6.1\% improvement compared to MS-DGCNN). For standard 3D object recognition, our method outperformed DGCNN and MS-DGCNN with overall accuracies of 93.15\% on ModelNet40 and 94.05\% on ModelNet10. With lower parameters and reduced complexity compared to state-of-the-art transformer approaches, our method is suitable for resource-constrained applications while maintaining a competitive accuracy. Beyond tree classification, the method generalizes to standard 3D object recognition, establishing it as a versatile solution for diverse point cloud processing applications. The implementation code is publicly available at https://github.com/said-ohamouddou/MS-DGCNN2.
>
---
#### [new 028] Integrated Oculomics and Lipidomics Reveal Microvascular Metabolic Signatures Associated with Cardiovascular Health in a Healthy Cohort
- **分类: cs.CV**

- **简介: 该论文属于心血管疾病早期检测任务，旨在通过整合眼底与脂质组学数据，发现非侵入性生物标志物，揭示微血管代谢特征与心血管健康的关系。**

- **链接: [http://arxiv.org/pdf/2507.12663v1](http://arxiv.org/pdf/2507.12663v1)**

> **作者:** Inamullah; Ernesto Elias Vidal Rosas; Imran Razzak; Shoaib Jameel
>
> **摘要:** Cardiovascular disease (CVD) remains the leading global cause of mortality, yet current risk stratification methods often fail to detect early, subclinical changes. Previous studies have generally not integrated retinal microvasculature characteristics with comprehensive serum lipidomic profiles as potential indicators of CVD risk. In this study, an innovative imaging omics framework was introduced, combining retinal microvascular traits derived through deep learning based image processing with serum lipidomic data to highlight asymptomatic biomarkers of cardiovascular risk beyond the conventional lipid panel. This represents the first large scale, covariate adjusted and stratified correlation analysis conducted in a healthy population, which is essential for identifying early indicators of disease. Retinal phenotypes were quantified using automated image analysis tools, while serum lipid profiling was performed by Ultra High Performance Liquid Chromatography Electrospray ionization High resolution mass spectrometry (UHPLC ESI HRMS). Strong, age- and sex-independent correlations were established, particularly between average artery width, vessel density, and lipid subclasses such as triacylglycerols (TAGs), diacylglycerols (DAGs), and ceramides (Cers). These associations suggest a converging mechanism of microvascular remodeling under metabolic stress. By linking detailed vascular structural phenotypes to specific lipid species, this study fills a critical gap in the understanding of early CVD pathogenesis. This integration not only offers a novel perspective on microvascular metabolic associations but also presents a significant opportunity for the identification of robust, non-invasive biomarkers. Ultimately, these findings may support improved early detection, targeted prevention, and personalized approaches in cardiovascular healthcare.
>
---
#### [new 029] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放词汇遥感实例分割任务，解决现有方法在识别新类别和跨数据集泛化能力不足的问题。提出SCORE框架，融合多粒度场景上下文提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.12857v1](http://arxiv.org/pdf/2507.12857v1)**

> **作者:** Shiqi Huang; Shuting He; Huaiyuan Qin; Bihan Wen
>
> **备注:** ICCV 2025
>
> **摘要:** Most existing remote sensing instance segmentation approaches are designed for close-vocabulary prediction, limiting their ability to recognize novel categories or generalize across datasets. This restricts their applicability in diverse Earth observation scenarios. To address this, we introduce open-vocabulary (OV) learning for remote sensing instance segmentation. While current OV segmentation models perform well on natural image datasets, their direct application to remote sensing faces challenges such as diverse landscapes, seasonal variations, and the presence of small or ambiguous objects in aerial imagery. To overcome these challenges, we propose $\textbf{SCORE}$ ($\textbf{S}$cene $\textbf{C}$ontext matters in $\textbf{O}$pen-vocabulary $\textbf{RE}$mote sensing instance segmentation), a framework that integrates multi-granularity scene context, i.e., regional context and global context, to enhance both visual and textual representations. Specifically, we introduce Region-Aware Integration, which refines class embeddings with regional context to improve object distinguishability. Additionally, we propose Global Context Adaptation, which enriches naive text embeddings with remote sensing global context, creating a more adaptable and expressive linguistic latent space for the classifier. We establish new benchmarks for OV remote sensing instance segmentation across diverse datasets. Experimental results demonstrate that, our proposed method achieves SOTA performance, which provides a robust solution for large-scale, real-world geospatial analysis. Our code is available at https://github.com/HuangShiqi128/SCORE.
>
---
#### [new 030] DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉里程计任务，解决单目VO的鲁棒性与泛化性问题。通过融合DINOv2特征与几何信息，提出DINO-VO系统，提升定位精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.13145v1](http://arxiv.org/pdf/2507.13145v1)**

> **作者:** Maulana Bisyir Azhari; David Hyunchul Shim
>
> **备注:** 8 pages, 6 figures. Accepted for publication in IEEE Robotics and Automation Letters (RA-L), July 2025
>
> **摘要:** Learning-based monocular visual odometry (VO) poses robustness, generalization, and efficiency challenges in robotics. Recent advances in visual foundation models, such as DINOv2, have improved robustness and generalization in various vision tasks, yet their integration in VO remains limited due to coarse feature granularity. In this paper, we present DINO-VO, a feature-based VO system leveraging DINOv2 visual foundation model for its sparse feature matching. To address the integration challenge, we propose a salient keypoints detector tailored to DINOv2's coarse features. Furthermore, we complement DINOv2's robust-semantic features with fine-grained geometric features, resulting in more localizable representations. Finally, a transformer-based matcher and differentiable pose estimation layer enable precise camera motion estimation by learning good matches. Against prior detector-descriptor networks like SuperPoint, DINO-VO demonstrates greater robustness in challenging environments. Furthermore, we show superior accuracy and generalization of the proposed feature descriptors against standalone DINOv2 coarse features. DINO-VO outperforms prior frame-to-frame VO methods on the TartanAir and KITTI datasets and is competitive on EuRoC dataset, while running efficiently at 72 FPS with less than 1GB of memory usage on a single GPU. Moreover, it performs competitively against Visual SLAM systems on outdoor driving scenarios, showcasing its generalization capabilities.
>
---
#### [new 031] ATL-Diff: Audio-Driven Talking Head Generation with Early Landmarks-Guide Noise Diffusion
- **分类: cs.CV**

- **简介: 该论文属于音频驱动的说话头生成任务，解决面部动画与音频同步问题。提出ATL-Diff框架，通过地标引导噪声扩散提升生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2507.12804v1](http://arxiv.org/pdf/2507.12804v1)**

> **作者:** Hoang-Son Vo; Quang-Vinh Nguyen; Seungwon Kim; Hyung-Jeong Yang; Soonja Yeom; Soo-Hyung Kim
>
> **摘要:** Audio-driven talking head generation requires precise synchronization between facial animations and audio signals. This paper introduces ATL-Diff, a novel approach addressing synchronization limitations while reducing noise and computational costs. Our framework features three key components: a Landmark Generation Module converting audio to facial landmarks, a Landmarks-Guide Noise approach that decouples audio by distributing noise according to landmarks, and a 3D Identity Diffusion network preserving identity characteristics. Experiments on MEAD and CREMA-D datasets demonstrate that ATL-Diff outperforms state-of-the-art methods across all metrics. Our approach achieves near real-time processing with high-quality animations, computational efficiency, and exceptional preservation of facial nuances. This advancement offers promising applications for virtual assistants, education, medical communication, and digital platforms. The source code is available at: \href{https://github.com/sonvth/ATL-Diff}{https://github.com/sonvth/ATL-Diff}
>
---
#### [new 032] Federated Learning for Commercial Image Sources
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像分类任务，旨在解决联邦学习中的数据隐私与分布异构问题。构建了首个专为联邦学习设计的图像数据集，并提出两种新算法Fed-Cyclic和Fed-Star。**

- **链接: [http://arxiv.org/pdf/2507.12903v1](http://arxiv.org/pdf/2507.12903v1)**

> **作者:** Shreyansh Jain; Koteswar Rao Jerripothula
>
> **备注:** Published in the Proceedings of IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023 with DOI: 10.1109/WACV56688.2023.00647
>
> **摘要:** Federated Learning is a collaborative machine learning paradigm that enables multiple clients to learn a global model without exposing their data to each other. Consequently, it provides a secure learning platform with privacy-preserving capabilities. This paper introduces a new dataset containing 23,326 images collected from eight different commercial sources and classified into 31 categories, similar to the Office-31 dataset. To the best of our knowledge, this is the first image classification dataset specifically designed for Federated Learning. We also propose two new Federated Learning algorithms, namely Fed-Cyclic and Fed-Star. In Fed-Cyclic, a client receives weights from its previous client, updates them through local training, and passes them to the next client, thus forming a cyclic topology. In Fed-Star, a client receives weights from all other clients, updates its local weights through pre-aggregation (to address statistical heterogeneity) and local training, and sends its updated local weights to all other clients, thus forming a star-like topology. Our experiments reveal that both algorithms perform better than existing baselines on our newly introduced dataset.
>
---
#### [new 033] AnyPos: Automated Task-Agnostic Actions for Bimanual Manipulation
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于双臂操作任务，解决任务特定数据依赖问题。提出AnyPos模型与ATARA框架，实现无需任务指导的自动化动作学习，提升效率与成功率。**

- **链接: [http://arxiv.org/pdf/2507.12768v1](http://arxiv.org/pdf/2507.12768v1)**

> **作者:** Hengkai Tan; Yao Feng; Xinyi Mao; Shuhe Huang; Guodong Liu; Zhongkai Hao; Hang Su; Jun Zhu
>
> **摘要:** Vision-language-action (VLA) models have shown promise on task-conditioned control in complex settings such as bimanual manipulation. However, the heavy reliance on task-specific human demonstrations limits their generalization and incurs high data acquisition costs. In this work, we present a new notion of task-agnostic action paradigm that decouples action execution from task-specific conditioning, enhancing scalability, efficiency, and cost-effectiveness. To address the data collection challenges posed by this paradigm -- such as low coverage density, behavioral redundancy, and safety risks -- we introduce ATARA (Automated Task-Agnostic Random Actions), a scalable self-supervised framework that accelerates collection by over $ 30\times $ compared to human teleoperation. To further enable effective learning from task-agnostic data, which often suffers from distribution mismatch and irrelevant trajectories, we propose AnyPos, an inverse dynamics model equipped with Arm-Decoupled Estimation and a Direction-Aware Decoder (DAD). We additionally integrate a video-conditioned action validation module to verify the feasibility of learned policies across diverse manipulation tasks. Extensive experiments show that the AnyPos-ATARA pipeline yields a 51% improvement in test accuracy and achieves 30-40% higher success rates in downstream tasks such as lifting, pick-and-place, and clicking, using replay-based video validation. Project Page: https://embodiedfoundation.github.io/vidar_anypos
>
---
#### [new 034] RGB Pre-Training Enhanced Unobservable Feature Latent Diffusion Model for Spectral Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于光谱重建任务，旨在从RGB图像重建高光谱图像。通过构建联合分布模型，解决未观测特征估计问题，提升重建效果。**

- **链接: [http://arxiv.org/pdf/2507.12967v1](http://arxiv.org/pdf/2507.12967v1)**

> **作者:** Keli Deng; Jie Nie; Yuntao Qian
>
> **摘要:** Spectral reconstruction (SR) is a crucial problem in image processing that requires reconstructing hyperspectral images (HSIs) from the corresponding RGB images. A key difficulty in SR is estimating the unobservable feature, which encapsulates significant spectral information not captured by RGB imaging sensors. The solution lies in effectively constructing the spectral-spatial joint distribution conditioned on the RGB image to complement the unobservable feature. Since HSIs share a similar spatial structure with the corresponding RGB images, it is rational to capitalize on the rich spatial knowledge in RGB pre-trained models for spectral-spatial joint distribution learning. To this end, we extend the RGB pre-trained latent diffusion model (RGB-LDM) to an unobservable feature LDM (ULDM) for SR. As the RGB-LDM and its corresponding spatial autoencoder (SpaAE) already excel in spatial knowledge, the ULDM can focus on modeling spectral structure. Moreover, separating the unobservable feature from the HSI reduces the redundant spectral information and empowers the ULDM to learn the joint distribution in a compact latent space. Specifically, we propose a two-stage pipeline consisting of spectral structure representation learning and spectral-spatial joint distribution learning to transform the RGB-LDM into the ULDM. In the first stage, a spectral unobservable feature autoencoder (SpeUAE) is trained to extract and compress the unobservable feature into a 3D manifold aligned with RGB space. In the second stage, the spectral and spatial structures are sequentially encoded by the SpeUAE and the SpaAE, respectively. The ULDM is then acquired to model the distribution of the coded unobservable feature with guidance from the corresponding RGB images. Experimental results on SR and downstream relighting tasks demonstrate that our proposed method achieves state-of-the-art performance.
>
---
#### [new 035] Unified Medical Image Segmentation with State Space Modeling Snake
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决多尺度结构异质性带来的挑战。提出Mamba Snake框架，结合状态空间建模和能量图先验，提升分割精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.12760v1](http://arxiv.org/pdf/2507.12760v1)**

> **作者:** Ruicheng Zhang; Haowei Guo; Kanghui Tian; Jun Zhou; Mingliang Yan; Zeyu Zhang; Shen Zhao
>
> **备注:** This paper has been accepted by ACM MM 2025
>
> **摘要:** Unified Medical Image Segmentation (UMIS) is critical for comprehensive anatomical assessment but faces challenges due to multi-scale structural heterogeneity. Conventional pixel-based approaches, lacking object-level anatomical insight and inter-organ relational modeling, struggle with morphological complexity and feature conflicts, limiting their efficacy in UMIS. We propose Mamba Snake, a novel deep snake framework enhanced by state space modeling for UMIS. Mamba Snake frames multi-contour evolution as a hierarchical state space atlas, effectively modeling macroscopic inter-organ topological relationships and microscopic contour refinements. We introduce a snake-specific vision state space module, the Mamba Evolution Block (MEB), which leverages effective spatiotemporal information aggregation for adaptive refinement of complex morphologies. Energy map shape priors further ensure robust long-range contour evolution in heterogeneous data. Additionally, a dual-classification synergy mechanism is incorporated to concurrently optimize detection and segmentation, mitigating under-segmentation of microstructures in UMIS. Extensive evaluations across five clinical datasets reveal Mamba Snake's superior performance, with an average Dice improvement of 3\% over state-of-the-art methods.
>
---
#### [new 036] Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于自动驾驶任务，解决长时序预测与复杂场景泛化问题。提出一种无需额外传感器的模型，采用连续自回归结构，提升预测性能。**

- **链接: [http://arxiv.org/pdf/2507.13162v1](http://arxiv.org/pdf/2507.13162v1)**

> **作者:** Arian Mousakhan; Sudhanshu Mittal; Silvio Galesso; Karim Farid; Thomas Brox
>
> **备注:** Project page: https://lmb-freiburg.github.io/orbis.github.io/
>
> **摘要:** Existing world models for autonomous driving struggle with long-horizon generation and generalization to challenging scenarios. In this work, we develop a model using simple design choices, and without additional supervision or sensors, such as maps, depth, or multiple cameras. We show that our model yields state-of-the-art performance, despite having only 469M parameters and being trained on 280h of video data. It particularly stands out in difficult scenarios like turning maneuvers and urban traffic. We test whether discrete token models possibly have advantages over continuous models based on flow matching. To this end, we set up a hybrid tokenizer that is compatible with both approaches and allows for a side-by-side comparison. Our study concludes in favor of the continuous autoregressive model, which is less brittle on individual design choices and more powerful than the model built on discrete tokens. Code, models and qualitative results are publicly available at https://lmb-freiburg.github.io/orbis.github.io/.
>
---
#### [new 037] Beyond Fully Supervised Pixel Annotations: Scribble-Driven Weakly-Supervised Framework for Image Manipulation Localization
- **分类: cs.CV**

- **简介: 该论文属于图像篡改定位任务，解决依赖像素级标注的问题。通过引入草图标注监督，提出新数据集和弱监督框架，提升检测性能与效率。**

- **链接: [http://arxiv.org/pdf/2507.13018v1](http://arxiv.org/pdf/2507.13018v1)**

> **作者:** Songlin Li; Guofeng Yu; Zhiqing Guo; Yunfeng Diao; Dan Ma; Gaobo Yang; Liejun Wang
>
> **摘要:** Deep learning-based image manipulation localization (IML) methods have achieved remarkable performance in recent years, but typically rely on large-scale pixel-level annotated datasets. To address the challenge of acquiring high-quality annotations, some recent weakly supervised methods utilize image-level labels to segment manipulated regions. However, the performance is still limited due to insufficient supervision signals. In this study, we explore a form of weak supervision that improves the annotation efficiency and detection performance, namely scribble annotation supervision. We re-annotated mainstream IML datasets with scribble labels and propose the first scribble-based IML (Sc-IML) dataset. Additionally, we propose the first scribble-based weakly supervised IML framework. Specifically, we employ self-supervised training with a structural consistency loss to encourage the model to produce consistent predictions under multi-scale and augmented inputs. In addition, we propose a prior-aware feature modulation module (PFMM) that adaptively integrates prior information from both manipulated and authentic regions for dynamic feature adjustment, further enhancing feature discriminability and prediction consistency in complex scenes. We also propose a gated adaptive fusion module (GAFM) that utilizes gating mechanisms to regulate information flow during feature fusion, guiding the model toward emphasizing potential tampered regions. Finally, we propose a confidence-aware entropy minimization loss (${\mathcal{L}}_{ {CEM }}$). This loss dynamically regularizes predictions in weakly annotated or unlabeled regions based on model uncertainty, effectively suppressing unreliable predictions. Experimental results show that our method outperforms existing fully supervised approaches in terms of average performance both in-distribution and out-of-distribution.
>
---
#### [new 038] Hierarchical Rectified Flow Matching with Mini-Batch Couplings
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生成模型任务，解决多模态数据建模问题。通过引入分层流匹配与小批量耦合，调整层次间分布复杂度，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2507.13350v1](http://arxiv.org/pdf/2507.13350v1)**

> **作者:** Yichi Zhang; Yici Yan; Alex Schwing; Zhizhen Zhao
>
> **备注:** Project Page: https://riccizz.github.io/HRF_coupling
>
> **摘要:** Flow matching has emerged as a compelling generative modeling approach that is widely used across domains. To generate data via a flow matching model, an ordinary differential equation (ODE) is numerically solved via forward integration of the modeled velocity field. To better capture the multi-modality that is inherent in typical velocity fields, hierarchical flow matching was recently introduced. It uses a hierarchy of ODEs that are numerically integrated when generating data. This hierarchy of ODEs captures the multi-modal velocity distribution just like vanilla flow matching is capable of modeling a multi-modal data distribution. While this hierarchy enables to model multi-modal velocity distributions, the complexity of the modeled distribution remains identical across levels of the hierarchy. In this paper, we study how to gradually adjust the complexity of the distributions across different levels of the hierarchy via mini-batch couplings. We show the benefits of mini-batch couplings in hierarchical rectified flow matching via compelling results on synthetic and imaging data. Code is available at https://riccizz.github.io/HRF_coupling.
>
---
#### [new 039] LanePerf: a Performance Estimation Framework for Lane Detection
- **分类: cs.CV**

- **简介: 该论文属于车道检测任务，解决模型在新环境下的性能评估问题。通过引入LanePerf框架，结合图像和车道特征，实现无需真实标签的性能估计。**

- **链接: [http://arxiv.org/pdf/2507.12894v1](http://arxiv.org/pdf/2507.12894v1)**

> **作者:** Yin Wu; Daniel Slieter; Ahmed Abouelazm; Christian Hubschneider; J. Marius Zöllner
>
> **备注:** Accepted in IEEE ITSC 2025
>
> **摘要:** Lane detection is a critical component of Advanced Driver-Assistance Systems (ADAS) and Automated Driving System (ADS), providing essential spatial information for lateral control. However, domain shifts often undermine model reliability when deployed in new environments. Ensuring the robustness and safety of lane detection models typically requires collecting and annotating target domain data, which is resource-intensive. Estimating model performance without ground-truth labels offers a promising alternative for efficient robustness assessment, yet remains underexplored in lane detection. While previous work has addressed performance estimation in image classification, these methods are not directly applicable to lane detection tasks. This paper first adapts five well-performing performance estimation methods from image classification to lane detection, building a baseline. Addressing the limitations of prior approaches that solely rely on softmax scores or lane features, we further propose a new Lane Performance Estimation Framework (LanePerf), which integrates image and lane features using a pretrained image encoder and a DeepSets-based architecture, effectively handling zero-lane detection scenarios and large domain-shift cases. Extensive experiments on the OpenLane dataset, covering diverse domain shifts (scenes, weather, hours), demonstrate that our LanePerf outperforms all baselines, achieving a lower MAE of 0.117 and a higher Spearman's rank correlation coefficient of 0.727. These findings pave the way for robust, label-free performance estimation in ADAS, supporting more efficient testing and improved safety in challenging driving scenarios.
>
---
#### [new 040] SE-VLN: A Self-Evolving Vision-Language Navigation Framework Based on Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决传统方法缺乏持续进化能力的问题。提出SE-VLN框架，通过记忆、推理和反思模块实现导航 agent 的自我演化。**

- **链接: [http://arxiv.org/pdf/2507.13152v1](http://arxiv.org/pdf/2507.13152v1)**

> **作者:** Xiangyu Dong; Haoran Zhao; Jiang Gao; Haozhou Li; Xiaoguang Ma; Yaoming Zhou; Fuhai Chen; Juan Liu
>
> **摘要:** Recent advances in vision-language navigation (VLN) were mainly attributed to emerging large language models (LLMs). These methods exhibited excellent generalization capabilities in instruction understanding and task reasoning. However, they were constrained by the fixed knowledge bases and reasoning abilities of LLMs, preventing fully incorporating experiential knowledge and thus resulting in a lack of efficient evolutionary capacity. To address this, we drew inspiration from the evolution capabilities of natural agents, and proposed a self-evolving VLN framework (SE-VLN) to endow VLN agents with the ability to continuously evolve during testing. To the best of our knowledge, it was the first time that an multimodal LLM-powered self-evolving VLN framework was proposed. Specifically, SE-VLN comprised three core modules, i.e., a hierarchical memory module to transfer successful and failure cases into reusable knowledge, a retrieval-augmented thought-based reasoning module to retrieve experience and enable multi-step decision-making, and a reflection module to realize continual evolution. Comprehensive tests illustrated that the SE-VLN achieved navigation success rates of 57% and 35.2% in unseen environments, representing absolute performance improvements of 23.9% and 15.0% over current state-of-the-art methods on R2R and REVERSE datasets, respectively. Moreover, the SE-VLN showed performance improvement with increasing experience repository, elucidating its great potential as a self-evolving agent framework for VLN.
>
---
#### [new 041] VideoITG: Multimodal Video Understanding with Instructed Temporal Grounding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，解决长视频中帧选择不精准的问题。提出VideoITG方法，通过指令引导的时空定位实现更有效的帧选取。**

- **链接: [http://arxiv.org/pdf/2507.13353v1](http://arxiv.org/pdf/2507.13353v1)**

> **作者:** Shihao Wang; Guo Chen; De-an Huang; Zhiqi Li; Minghan Li; Guilin Li; Jose M. Alvarez; Lei Zhang; Zhiding Yu
>
> **备注:** Technical Report
>
> **摘要:** Recent studies have revealed that selecting informative and relevant video frames can significantly improve the performance of Video Large Language Models (Video-LLMs). Current methods, such as reducing inter-frame redundancy, employing separate models for image-text relevance assessment, or utilizing temporal video grounding for event localization, substantially adopt unsupervised learning paradigms, whereas they struggle to address the complex scenarios in long video understanding. We propose Instructed Temporal Grounding for Videos (VideoITG), featuring customized frame sampling aligned with user instructions. The core of VideoITG is the VidThinker pipeline, an automated annotation framework that explicitly mimics the human annotation process. First, it generates detailed clip-level captions conditioned on the instruction; then, it retrieves relevant video segments through instruction-guided reasoning; finally, it performs fine-grained frame selection to pinpoint the most informative visual evidence. Leveraging VidThinker, we construct the VideoITG-40K dataset, containing 40K videos and 500K instructed temporal grounding annotations. We then design a plug-and-play VideoITG model, which takes advantage of visual language alignment and reasoning capabilities of Video-LLMs, for effective frame selection in a discriminative manner. Coupled with Video-LLMs, VideoITG achieves consistent performance improvements across multiple multimodal video understanding benchmarks, showing its superiority and great potentials for video understanding.
>
---
#### [new 042] Channel-wise Motion Features for Efficient Motion Segmentation
- **分类: cs.CV**

- **简介: 该论文属于运动分割任务，旨在提高实时性。通过提出通道运动特征，仅使用姿态网络，提升效率并减少参数。**

- **链接: [http://arxiv.org/pdf/2507.13082v1](http://arxiv.org/pdf/2507.13082v1)**

> **作者:** Riku Inoue; Masamitsu Tsuchiya; Yuji Yasui
>
> **备注:** This paper has been accepted to IROS 2024 (Abu Dhabi, UAE), October 14-18, 2024
>
> **摘要:** For safety-critical robotics applications such as autonomous driving, it is important to detect all required objects accurately in real-time. Motion segmentation offers a solution by identifying dynamic objects from the scene in a class-agnostic manner. Recently, various motion segmentation models have been proposed, most of which jointly use subnetworks to estimate Depth, Pose, Optical Flow, and Scene Flow. As a result, the overall computational cost of the model increases, hindering real-time performance. In this paper, we propose a novel cost-volume-based motion feature representation, Channel-wise Motion Features. By extracting depth features of each instance in the feature map and capturing the scene's 3D motion information, it offers enhanced efficiency. The only subnetwork used to build Channel-wise Motion Features is the Pose Network, and no others are required. Our method not only achieves about 4 times the FPS of state-of-the-art models in the KITTI Dataset and Cityscapes of the VCAS-Motion Dataset, but also demonstrates equivalent accuracy while reducing the parameters to about 25$\%$.
>
---
#### [new 043] A Real-Time System for Egocentric Hand-Object Interaction Detection in Industrial Domains
- **分类: cs.CV**

- **简介: 该论文属于工业场景下的实时手物交互检测任务，旨在解决快速准确识别手与物体交互的问题。提出结合动作识别与目标检测的级联架构，提升检测效率与精度。**

- **链接: [http://arxiv.org/pdf/2507.13326v1](http://arxiv.org/pdf/2507.13326v1)**

> **作者:** Antonio Finocchiaro; Alessandro Sebastiano Catinello; Michele Mazzamuto; Rosario Leonardi; Antonino Furnari; Giovanni Maria Farinella
>
> **备注:** 12 pages, 4 figures, In International Conference on Image Analysis and Processing
>
> **摘要:** Hand-object interaction detection remains an open challenge in real-time applications, where intuitive user experiences depend on fast and accurate detection of interactions with surrounding objects. We propose an efficient approach for detecting hand-objects interactions from streaming egocentric vision that operates in real time. Our approach consists of an action recognition module and an object detection module for identifying active objects upon confirmed interaction. Our Mamba model with EfficientNetV2 as backbone for action recognition achieves 38.52% p-AP on the ENIGMA-51 benchmark at 30fps, while our fine-tuned YOLOWorld reaches 85.13% AP for hand and object. We implement our models in a cascaded architecture where the action recognition and object detection modules operate sequentially. When the action recognition predicts a contact state, it activates the object detection module, which in turn performs inference on the relevant frame to detect and classify the active object.
>
---
#### [new 044] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于模型压缩任务，解决扩散模型低比特量化中的性能下降问题。通过引入LES和PTS方法提升量化效果，确保低比特下生成质量与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.12933v1](http://arxiv.org/pdf/2507.12933v1)**

> **作者:** Dongyeun Lee; Jiwan Hur; Hyounguk Shon; Jae Young Lee; Junmo Kim
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Diffusion models have achieved remarkable success in image generation but come with significant computational costs, posing challenges for deployment in resource-constrained environments. Recent post-training quantization (PTQ) methods have attempted to mitigate this issue by focusing on the iterative nature of diffusion models. However, these approaches often overlook outliers, leading to degraded performance at low bit-widths. In this paper, we propose a DMQ which combines Learned Equivalent Scaling (LES) and channel-wise Power-of-Two Scaling (PTS) to effectively address these challenges. Learned Equivalent Scaling optimizes channel-wise scaling factors to redistribute quantization difficulty between weights and activations, reducing overall quantization error. Recognizing that early denoising steps, despite having small quantization errors, crucially impact the final output due to error accumulation, we incorporate an adaptive timestep weighting scheme to prioritize these critical steps during learning. Furthermore, identifying that layers such as skip connections exhibit high inter-channel variance, we introduce channel-wise Power-of-Two Scaling for activations. To ensure robust selection of PTS factors even with small calibration set, we introduce a voting algorithm that enhances reliability. Extensive experiments demonstrate that our method significantly outperforms existing works, especially at low bit-widths such as W4A6 (4-bit weight, 6-bit activation) and W4A8, maintaining high image generation quality and model stability. The code is available at https://github.com/LeeDongYeun/dmq.
>
---
#### [new 045] R^2MoE: Redundancy-Removal Mixture of Experts for Lifelong Concept Learning
- **分类: cs.CV**

- **简介: 该论文属于持续视觉概念学习任务，解决灾难性遗忘和参数膨胀问题。提出R²MoE框架，通过专家选择与冗余消除实现高效学习。**

- **链接: [http://arxiv.org/pdf/2507.13107v1](http://arxiv.org/pdf/2507.13107v1)**

> **作者:** Xiaohan Guo; Yusong Cai; Zejia Liu; Zhengning Wang; Lili Pan; Hongliang Li
>
> **摘要:** Enabling large-scale generative models to continuously learn new visual concepts is essential for personalizing pre-trained models to meet individual user preferences. Existing approaches for continual visual concept learning are constrained by two fundamental challenges: catastrophic forgetting and parameter expansion. In this paper, we propose Redundancy-Removal Mixture of Experts (R^2MoE), a parameter-efficient framework for lifelong visual concept learning that effectively learns new concepts while incurring minimal parameter overhead. Our framework includes three key innovative contributions: First, we propose a mixture-of-experts framework with a routing distillation mechanism that enables experts to acquire concept-specific knowledge while preserving the gating network's routing capability, thereby effectively mitigating catastrophic forgetting. Second, we propose a strategy for eliminating redundant layer-wise experts that reduces the number of expert parameters by fully utilizing previously learned experts. Third, we employ a hierarchical local attention-guided inference approach to mitigate interference between generated visual concepts. Extensive experiments have demonstrated that our method generates images with superior conceptual fidelity compared to the state-of-the-art (SOTA) method, achieving an impressive 87.8\% reduction in forgetting rates and 63.3\% fewer parameters on the CustomConcept 101 dataset. Our code is available at {https://github.com/learninginvision/R2MoE}
>
---
#### [new 046] VITA: Vision-to-Action Flow Matching Policy
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VITA，一种基于视觉到动作流匹配的策略，用于解决视觉-动作控制任务中的模态差异和生成效率问题。通过结构化动作潜空间实现端到端学习，提升操控性能并降低延迟。**

- **链接: [http://arxiv.org/pdf/2507.13231v1](http://arxiv.org/pdf/2507.13231v1)**

> **作者:** Dechen Gao; Boqi Zhao; Andrew Lee; Ian Chuang; Hanchu Zhou; Hang Wang; Zhe Zhao; Junshan Zhang; Iman Soltani
>
> **备注:** Project page: https://ucd-dare.github.io/VITA/
>
> **摘要:** We present VITA, a Vision-To-Action flow matching policy that evolves latent visual representations into latent actions for visuomotor control. Traditional flow matching and diffusion policies sample from standard source distributions (e.g., Gaussian noise) and require additional conditioning mechanisms like cross-attention to condition action generation on visual information, creating time and space overheads. VITA proposes a novel paradigm that treats latent images as the flow source, learning an inherent mapping from vision to action while eliminating separate conditioning modules and preserving generative modeling capabilities. Learning flows between fundamentally different modalities like vision and action is challenging due to sparse action data lacking semantic structures and dimensional mismatches between high-dimensional visual representations and raw actions. We address this by creating a structured action latent space via an autoencoder as the flow matching target, up-sampling raw actions to match visual representation shapes. Crucially, we supervise flow matching with both encoder targets and final action outputs through flow latent decoding, which backpropagates action reconstruction loss through sequential flow matching ODE solving steps for effective end-to-end learning. Implemented as simple MLP layers, VITA is evaluated on challenging bi-manual manipulation tasks on the ALOHA platform, including 5 simulation and 2 real-world tasks. Despite its simplicity, MLP-only VITA outperforms or matches state-of-the-art generative policies while reducing inference latency by 50-130% compared to conventional flow matching policies requiring different conditioning mechanisms or complex architectures. To our knowledge, VITA is the first MLP-only flow matching policy capable of solving complex bi-manual manipulation tasks like those in ALOHA benchmarks.
>
---
#### [new 047] HairShifter: Consistent and High-Fidelity Video Hair Transfer via Anchor-Guided Animation
- **分类: cs.CV**

- **简介: 该论文属于视频发型迁移任务，解决视频中发型转移的时序一致性和空间保真度问题。提出HairShifter框架，结合图像迁移与多尺度解码器，实现高质量视频发型迁移。**

- **链接: [http://arxiv.org/pdf/2507.12758v1](http://arxiv.org/pdf/2507.12758v1)**

> **作者:** Wangzheng Shi; Yinglin Zheng; Yuxin Lin; Jianmin Bao; Ming Zeng; Dong Chen
>
> **摘要:** Hair transfer is increasingly valuable across domains such as social media, gaming, advertising, and entertainment. While significant progress has been made in single-image hair transfer, video-based hair transfer remains challenging due to the need for temporal consistency, spatial fidelity, and dynamic adaptability. In this work, we propose HairShifter, a novel "Anchor Frame + Animation" framework that unifies high-quality image hair transfer with smooth and coherent video animation. At its core, HairShifter integrates a Image Hair Transfer (IHT) module for precise per-frame transformation and a Multi-Scale Gated SPADE Decoder to ensure seamless spatial blending and temporal coherence. Our method maintains hairstyle fidelity across frames while preserving non-hair regions. Extensive experiments demonstrate that HairShifter achieves state-of-the-art performance in video hairstyle transfer, combining superior visual quality, temporal consistency, and scalability. The code will be publicly available. We believe this work will open new avenues for video-based hairstyle transfer and establish a robust baseline in this field.
>
---
#### [new 048] DiffClean: Diffusion-based Makeup Removal for Accurate Age Estimation
- **分类: cs.CV**

- **简介: 该论文属于人脸识别与年龄估计任务，解决面部妆容影响年龄判断的问题。通过扩散模型去除妆容，提升年龄识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.13292v1](http://arxiv.org/pdf/2507.13292v1)**

> **作者:** Ekta Balkrishna Gavas; Chinmay Hegde; Nasir Memon; Sudipta Banerjee
>
> **摘要:** Accurate age verification can protect underage users from unauthorized access to online platforms and e-commerce sites that provide age-restricted services. However, accurate age estimation can be confounded by several factors, including facial makeup that can induce changes to alter perceived identity and age to fool both humans and machines. In this work, we propose DiffClean which erases makeup traces using a text-guided diffusion model to defend against makeup attacks. DiffClean improves age estimation (minor vs. adult accuracy by 4.8%) and face verification (TMR by 8.9% at FMR=0.01%) over competing baselines on digitally simulated and real makeup images.
>
---
#### [new 049] Predicting Soccer Penalty Kick Direction Using Human Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于动作预测任务，旨在通过分析罚球前球员动作预测射门方向。研究构建了标注数据集，并提出深度学习模型进行预测。**

- **链接: [http://arxiv.org/pdf/2507.12617v1](http://arxiv.org/pdf/2507.12617v1)**

> **作者:** David Freire-Obregón; Oliverio J. Santana; Javier Lorenzo-Navarro; Daniel Hernández-Sosa; Modesto Castrillón-Santana
>
> **备注:** Accepted at 23rd International Conference on Image Analysis and Processing (ICIAP 2025)
>
> **摘要:** Action anticipation has become a prominent topic in Human Action Recognition (HAR). However, its application to real-world sports scenarios remains limited by the availability of suitable annotated datasets. This work presents a novel dataset of manually annotated soccer penalty kicks to predict shot direction based on pre-kick player movements. We propose a deep learning classifier to benchmark this dataset that integrates HAR-based feature embeddings with contextual metadata. We evaluate twenty-two backbone models across seven architecture families (MViTv2, MViTv1, SlowFast, Slow, X3D, I3D, C2D), achieving up to 63.9% accuracy in predicting shot direction (left or right), outperforming the real goalkeepers' decisions. These results demonstrate the dataset's value for anticipatory action recognition and validate our model's potential as a generalizable approach for sports-based predictive tasks.
>
---
#### [new 050] World Model-Based End-to-End Scene Generation for Accident Anticipation in Autonomous Driving
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶中的事故预判任务，旨在解决数据不足和环境干扰导致的感知问题。通过生成场景和动态预测模型提升事故预判准确性与提前量。**

- **链接: [http://arxiv.org/pdf/2507.12762v1](http://arxiv.org/pdf/2507.12762v1)**

> **作者:** Yanchen Guan; Haicheng Liao; Chengyue Wang; Xingcheng Liu; Jiaxun Zhang; Zhenning Li
>
> **摘要:** Reliable anticipation of traffic accidents is essential for advancing autonomous driving systems. However, this objective is limited by two fundamental challenges: the scarcity of diverse, high-quality training data and the frequent absence of crucial object-level cues due to environmental disruptions or sensor deficiencies. To tackle these issues, we propose a comprehensive framework combining generative scene augmentation with adaptive temporal reasoning. Specifically, we develop a video generation pipeline that utilizes a world model guided by domain-informed prompts to create high-resolution, statistically consistent driving scenarios, particularly enriching the coverage of edge cases and complex interactions. In parallel, we construct a dynamic prediction model that encodes spatio-temporal relationships through strengthened graph convolutions and dilated temporal operators, effectively addressing data incompleteness and transient visual noise. Furthermore, we release a new benchmark dataset designed to better capture diverse real-world driving risks. Extensive experiments on public and newly released datasets confirm that our framework enhances both the accuracy and lead time of accident anticipation, offering a robust solution to current data and modeling limitations in safety-critical autonomous driving applications.
>
---
#### [new 051] Differential-informed Sample Selection Accelerates Multimodal Contrastive Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态对比学习任务，旨在解决样本选择效率低的问题。提出DISSect方法，通过分析模型差异提升样本选择质量，加速训练过程。**

- **链接: [http://arxiv.org/pdf/2507.12998v1](http://arxiv.org/pdf/2507.12998v1)**

> **作者:** Zihua Zhao; Feng Hong; Mengxi Chen; Pengyi Chen; Benyuan Liu; Jiangchao Yao; Ya Zhang; Yanfeng Wang
>
> **摘要:** The remarkable success of contrastive-learning-based multimodal models has been greatly driven by training on ever-larger datasets with expensive compute consumption. Sample selection as an alternative efficient paradigm plays an important direction to accelerate the training process. However, recent advances on sample selection either mostly rely on an oracle model to offline select a high-quality coreset, which is limited in the cold-start scenarios, or focus on online selection based on real-time model predictions, which has not sufficiently or efficiently considered the noisy correspondence. To address this dilemma, we propose a novel Differential-Informed Sample Selection (DISSect) method, which accurately and efficiently discriminates the noisy correspondence for training acceleration. Specifically, we rethink the impact of noisy correspondence on contrastive learning and propose that the differential between the predicted correlation of the current model and that of a historical model is more informative to characterize sample quality. Based on this, we construct a robust differential-based sample selection and analyze its theoretical insights. Extensive experiments on three benchmark datasets and various downstream tasks demonstrate the consistent superiority of DISSect over current state-of-the-art methods. Source code is available at: https://github.com/MediaBrain-SJTU/DISSect.
>
---
#### [new 052] DeQA-Doc: Adapting DeQA-Score to Document Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于文档质量评估任务，旨在解决现有方法评分不准确的问题。通过改进DeQA-Score，提出DeQA-Doc框架，提升文档图像质量评估的准确性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.12796v1](http://arxiv.org/pdf/2507.12796v1)**

> **作者:** Junjie Gao; Runze Liu; Yingzhe Peng; Shujian Yang; Jin Zhang; Kai Yang; Zhiyuan You
>
> **摘要:** Document quality assessment is critical for a wide range of applications including document digitization, OCR, and archival. However, existing approaches often struggle to provide accurate and robust quality scores, limiting their applicability in practical scenarios. With the rapid progress in Multi-modal Large Language Models (MLLMs), recent MLLM-based methods have achieved remarkable performance in image quality assessment. In this work, we extend this success to the document domain by adapting DeQA-Score, a state-of-the-art MLLM-based image quality scorer, for document quality assessment. We propose DeQA-Doc, a framework that leverages the visual language capabilities of MLLMs and a soft label strategy to regress continuous document quality scores. To adapt DeQA-Score to DeQA-Doc, we adopt two complementary solutions to construct soft labels without the variance information. Also, we relax the resolution constrains to support the large resolution of document images. Finally, we introduce ensemble methods to further enhance the performance. Extensive experiments demonstrate that DeQA-Doc significantly outperforms existing baselines, offering accurate and generalizable document quality assessment across diverse degradation types. Codes and model weights are available in https://github.com/Junjie-Gao19/DeQA-Doc.
>
---
#### [new 053] $π^3$: Scalable Permutation-Equivariant Visual Geometry Learning
- **分类: cs.CV**

- **简介: 该论文提出$\pi^3$，解决视觉几何重建任务中的参考视角依赖问题，通过排列等变架构实现无参考的相机位姿和点图预测。**

- **链接: [http://arxiv.org/pdf/2507.13347v1](http://arxiv.org/pdf/2507.13347v1)**

> **作者:** Yifan Wang; Jianjun Zhou; Haoyi Zhu; Wenzheng Chang; Yang Zhou; Zizun Li; Junyi Chen; Jiangmiao Pang; Chunhua Shen; Tong He
>
> **备注:** Project page: https://yyfz.github.io/pi3/
>
> **摘要:** We introduce $\pi^3$, a feed-forward neural network that offers a novel approach to visual geometry reconstruction, breaking the reliance on a conventional fixed reference view. Previous methods often anchor their reconstructions to a designated viewpoint, an inductive bias that can lead to instability and failures if the reference is suboptimal. In contrast, $\pi^3$ employs a fully permutation-equivariant architecture to predict affine-invariant camera poses and scale-invariant local point maps without any reference frames. This design makes our model inherently robust to input ordering and highly scalable. These advantages enable our simple and bias-free approach to achieve state-of-the-art performance on a wide range of tasks, including camera pose estimation, monocular/video depth estimation, and dense point map reconstruction. Code and models are publicly available.
>
---
#### [new 054] SEMT: Static-Expansion-Mesh Transformer Network Architecture for Remote Sensing Image Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于远程感知图像描述任务，旨在提升卫星图像自动描述生成效果。提出SEMT架构，融合静态扩展、记忆增强自注意力等技术，优化模型性能。**

- **链接: [http://arxiv.org/pdf/2507.12845v1](http://arxiv.org/pdf/2507.12845v1)**

> **作者:** Khang Truong; Lam Pham; Hieu Tang; Jasmin Lampert; Martin Boyer; Son Phan; Truong Nguyen
>
> **摘要:** Image captioning has emerged as a crucial task in the intersection of computer vision and natural language processing, enabling automated generation of descriptive text from visual content. In the context of remote sensing, image captioning plays a significant role in interpreting vast and complex satellite imagery, aiding applications such as environmental monitoring, disaster assessment, and urban planning. This motivates us, in this paper, to present a transformer based network architecture for remote sensing image captioning (RSIC) in which multiple techniques of Static Expansion, Memory-Augmented Self-Attention, Mesh Transformer are evaluated and integrated. We evaluate our proposed models using two benchmark remote sensing image datasets of UCM-Caption and NWPU-Caption. Our best model outperforms the state-of-the-art systems on most of evaluation metrics, which demonstrates potential to apply for real-life remote sensing image systems.
>
---
#### [new 055] MVA 2025 Small Multi-Object Tracking for Spotting Birds Challenge: Dataset, Methods, and Results
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于小目标多目标跟踪任务，解决小目标检测与跟踪难题。构建了SMOT4SB数据集，提出SO-HOTA评估指标，并举办MVA2025挑战赛，提升跟踪性能。**

- **链接: [http://arxiv.org/pdf/2507.12832v1](http://arxiv.org/pdf/2507.12832v1)**

> **作者:** Yuki Kondo; Norimichi Ukita; Riku Kanayama; Yuki Yoshida; Takayuki Yamaguchi; Xiang Yu; Guang Liang; Xinyao Liu; Guan-Zhang Wang; Wei-Ta Chu; Bing-Cheng Chuang; Jia-Hua Lee; Pin-Tseng Kuo; I-Hsuan Chu; Yi-Shein Hsiao; Cheng-Han Wu; Po-Yi Wu; Jui-Chien Tsou; Hsuan-Chi Liu; Chun-Yi Lee; Yuan-Fu Yang; Kosuke Shigematsu; Asuka Shin; Ba Tran
>
> **备注:** This paper is the official challenge report for SMOT4SB and is published in the proceedings of MVA 2025 (19th International Conference on Machine Vision and Applications). Official challenge page: https://www.mva-org.jp/mva2025/challenge
>
> **摘要:** Small Multi-Object Tracking (SMOT) is particularly challenging when targets occupy only a few dozen pixels, rendering detection and appearance-based association unreliable. Building on the success of the MVA2023 SOD4SB challenge, this paper introduces the SMOT4SB challenge, which leverages temporal information to address limitations of single-frame detection. Our three main contributions are: (1) the SMOT4SB dataset, consisting of 211 UAV video sequences with 108,192 annotated frames under diverse real-world conditions, designed to capture motion entanglement where both camera and targets move freely in 3D; (2) SO-HOTA, a novel metric combining Dot Distance with HOTA to mitigate the sensitivity of IoU-based metrics to small displacements; and (3) a competitive MVA2025 challenge with 78 participants and 308 submissions, where the winning method achieved a 5.1x improvement over the baseline. This work lays a foundation for advancing SMOT in UAV scenarios with applications in bird strike avoidance, agriculture, fisheries, and ecological monitoring.
>
---
#### [new 056] DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决标注变异问题。提出DiffOSeg框架，同时实现专家共识与个体偏好，提升分割准确性。**

- **链接: [http://arxiv.org/pdf/2507.13087v1](http://arxiv.org/pdf/2507.13087v1)**

> **作者:** Han Zhang; Xiangde Luo; Yong Chen; Kang Li
>
> **摘要:** Annotation variability remains a substantial challenge in medical image segmentation, stemming from ambiguous imaging boundaries and diverse clinical expertise. Traditional deep learning methods producing single deterministic segmentation predictions often fail to capture these annotator biases. Although recent studies have explored multi-rater segmentation, existing methods typically focus on a single perspective -- either generating a probabilistic ``gold standard'' consensus or preserving expert-specific preferences -- thus struggling to provide a more omni view. In this study, we propose DiffOSeg, a two-stage diffusion-based framework, which aims to simultaneously achieve both consensus-driven (combining all experts' opinions) and preference-driven (reflecting experts' individual assessments) segmentation. Stage I establishes population consensus through a probabilistic consensus strategy, while Stage II captures expert-specific preference via adaptive prompts. Demonstrated on two public datasets (LIDC-IDRI and NPC-170), our model outperforms existing state-of-the-art methods across all evaluated metrics. Source code is available at https://github.com/string-ellipses/DiffOSeg .
>
---
#### [new 057] Analysis of Image-and-Text Uncertainty Propagation in Multimodal Large Language Models with Cardiac MR-Based Applications
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决MLLM中图像与文本不确定性传播问题。提出MUPM模型，分析不确定性的关系，并应用于心脏MRI临床场景。**

- **链接: [http://arxiv.org/pdf/2507.12945v1](http://arxiv.org/pdf/2507.12945v1)**

> **作者:** Yucheng Tang; Yunguan Fu; Weixi Yi; Yipei Wang; Daniel C. Alexander; Rhodri Davies; Yipeng Hu
>
> **备注:** It is accepted by 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025
>
> **摘要:** Multimodal large language models (MLLMs) can process and integrate information from multimodality sources, such as text and images. However, interrelationship among input modalities, uncertainties due to individual uni-modal data and potential clinical applications following such an uncertainty decomposition are yet fully understood in the context of large-scale MLLMs. In this work, we propose a multimodal uncertainty propagation model (MUPM) based on uncertainty propagation, to characterise the relationship among the uncertainties arising from image-only, text-only, and joint image-text variations in MLLM inputs. Using real clinical data consisting of cardiac MR scans and digital health records, we describe that MUPMs can be optimised robustly with a few samples. We then show that the fitted MUPMs are generalisable across different input data distributions and, perhaps surprisingly, across different downstream tasks. Such a transferability may be explained by the shared pretraining, comparatively light MLLM fine-tuning, along with the low-dimensional nature of the MUPMs. More importantly, this learned transferability, quantifying the relationship between these uncertainties, led to direct clinical applications in which uncertainties may be estimated and thus analysed robustly for varying data or even a novel set of cardiac disease prediction tasks. In addition, we show experimentally the efficiency in multimodal data required for estimating the overall uncertainty and its ability to identify redundant factors, both of which are considered practical yet clinically useful applications with the proposed MUPMs. Codes are available at https://github.com/yucheng722/MUPM.
>
---
#### [new 058] Compact Vision Transformer by Reduction of Kernel Complexity
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解决Transformer模型计算成本高的问题。通过引入KCR-Transformer，实现通道选择以降低FLOPs，同时保持或提升性能。**

- **链接: [http://arxiv.org/pdf/2507.12780v1](http://arxiv.org/pdf/2507.12780v1)**

> **作者:** Yancheng Wang; Yingzhen Yang
>
> **摘要:** Self-attention and transformer architectures have become foundational components in modern deep learning. Recent efforts have integrated transformer blocks into compact neural architectures for computer vision, giving rise to various efficient vision transformers. In this work, we introduce Transformer with Kernel Complexity Reduction, or KCR-Transformer, a compact transformer block equipped with differentiable channel selection, guided by a novel and sharp theoretical generalization bound. KCR-Transformer performs input/output channel selection in the MLP layers of transformer blocks to reduce the computational cost. Furthermore, we provide a rigorous theoretical analysis establishing a tight generalization bound for networks equipped with KCR-Transformer blocks. Leveraging such strong theoretical results, the channel pruning by KCR-Transformer is conducted in a generalization-aware manner, ensuring that the resulting network retains a provably small generalization error. Our KCR-Transformer is compatible with many popular and compact transformer networks, such as ViT and Swin, and it reduces the FLOPs of the vision transformers while maintaining or even improving the prediction accuracy. In the experiments, we replace all the transformer blocks in the vision transformers with KCR-Transformer blocks, leading to KCR-Transformer networks with different backbones. The resulting TCR-Transformers achieve superior performance on various computer vision tasks, achieving even better performance than the original models with even less FLOPs and parameters.
>
---
#### [new 059] $S^2M^2$: Scalable Stereo Matching Model for Reliable Depth Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于立体匹配任务，解决跨分辨率和视差范围的通用性问题。提出S²M²模型，结合多尺度Transformer和新损失函数，实现高精度与高效性。**

- **链接: [http://arxiv.org/pdf/2507.13229v1](http://arxiv.org/pdf/2507.13229v1)**

> **作者:** Junhong Min; Youngpil Jeon; Jimin Kim; Minyong Choi
>
> **备注:** 8 pages, 5 figures, ICCV accepted paper
>
> **摘要:** The pursuit of a generalizable stereo matching model, capable of performing across varying resolutions and disparity ranges without dataset-specific fine-tuning, has revealed a fundamental trade-off. Iterative local search methods achieve high scores on constrained benchmarks, but their core mechanism inherently limits the global consistency required for true generalization. On the other hand, global matching architectures, while theoretically more robust, have been historically rendered infeasible by prohibitive computational and memory costs. We resolve this dilemma with $S^2M^2$: a global matching architecture that achieves both state-of-the-art accuracy and high efficiency without relying on cost volume filtering or deep refinement stacks. Our design integrates a multi-resolution transformer for robust long-range correspondence, trained with a novel loss function that concentrates probability on feasible matches. This approach enables a more robust joint estimation of disparity, occlusion, and confidence. $S^2M^2$ establishes a new state of the art on the Middlebury v3 and ETH3D benchmarks, significantly outperforming prior methods across most metrics while reconstructing high-quality details with competitive efficiency.
>
---
#### [new 060] Camera-based implicit mind reading by capturing higher-order semantic dynamics of human gaze within environmental context
- **分类: cs.CV**

- **简介: 该论文属于情感识别任务，旨在解决传统方法依赖显性信号和静态分析的不足。通过摄像头捕捉眼动与环境语义的动态关系，实现无感知、实时情绪识别。**

- **链接: [http://arxiv.org/pdf/2507.12889v1](http://arxiv.org/pdf/2507.12889v1)**

> **作者:** Mengke Song; Yuge Xie; Qi Cui; Luming Li; Xinyu Liu; Guotao Wang; Chenglizhao Chen; Shanchen Pang
>
> **摘要:** Emotion recognition,as a step toward mind reading,seeks to infer internal states from external cues.Most existing methods rely on explicit signals-such as facial expressions,speech,or gestures-that reflect only bodily responses and overlook the influence of environmental context.These cues are often voluntary,easy to mask,and insufficient for capturing deeper,implicit emotions. Physiological signal-based approaches offer more direct access to internal states but require complex sensors that compromise natural behavior and limit scalability.Gaze-based methods typically rely on static fixation analysis and fail to capture the rich,dynamic interactions between gaze and the environment,and thus cannot uncover the deep connection between emotion and implicit behavior.To address these limitations,we propose a novel camera-based,user-unaware emotion recognition approach that integrates gaze fixation patterns with environmental semantics and temporal dynamics.Leveraging standard HD cameras,our method unobtrusively captures users'eye appearance and head movements in natural settings-without the need for specialized hardware or active user participation.From these visual cues,the system estimates gaze trajectories over time and space, providing the basis for modeling the spatial, semantic,and temporal dimensions of gaze behavior. This allows us to capture the dynamic interplay between visual attention and the surrounding environment,revealing that emotions are not merely physiological responses but complex outcomes of human-environment interactions.The proposed approach enables user-unaware,real-time,and continuous emotion recognition,offering high generalizability and low deployment cost.
>
---
#### [new 061] Mono-InternVL-1.5: Towards Cheaper and Faster Monolithic Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在解决单体模型训练不稳定和数据成本高的问题。通过引入视觉参数空间和优化预训练方法，提升了模型性能并降低了成本。**

- **链接: [http://arxiv.org/pdf/2507.12566v1](http://arxiv.org/pdf/2507.12566v1)**

> **作者:** Gen Luo; Wenhan Dou; Wenhao Li; Zhaokai Wang; Xue Yang; Changyao Tian; Hao Li; Weiyun Wang; Wenhai Wang; Xizhou Zhu; Yu Qiao; Jifeng Dai
>
> **摘要:** This paper focuses on monolithic Multimodal Large Language Models (MLLMs), which integrate visual encoding and language decoding into a single model. Existing structures and pre-training strategies for monolithic MLLMs often suffer from unstable optimization and catastrophic forgetting. To address these challenges, our key idea is to embed a new visual parameter space into a pre-trained LLM, enabling stable learning of visual knowledge from noisy data via delta tuning. Based on this principle, we first introduce Mono-InternVL, an advanced monolithic MLLM that incorporates a set of visual experts through a multimodal mixture-of-experts architecture. In addition, we design an innovative Endogenous Visual Pre-training (EViP) for Mono-InternVL to maximize its visual capabilities via progressive learning. Mono-InternVL achieves competitive performance against existing MLLMs but also leads to relatively expensive data cost. Therefore, we further present Mono-InternVL-1.5, a cheaper and stronger monolithic MLLM equipped with an improved EViP (EViP++). EViP++ introduces additional visual attention experts to Mono-InternVL-1.5 and re-organizes the pre-training process in an efficient manner. During inference, it includes a fused CUDA kernel to speed up its MoE operations. With these designs, Mono-InternVL-1.5 significantly reduces training and inference costs, while still maintaining competitive performance with Mono-InternVL. To evaluate our approach, we conduct extensive experiments across 15 benchmarks. Results demonstrate that Mono-InternVL outperforms existing monolithic MLLMs on 12 out of 15 benchmarks, e.g., +114-point improvement over Emu3 on OCRBench. Compared to its modular counterpart, i.e., InternVL-1.5, Mono-InternVL-1.5 achieves similar multimodal performance while reducing first-token latency by up to 69%. Code and models are released at https://github.com/OpenGVLab/Mono-InternVL.
>
---
#### [new 062] Leveraging Language Prior for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决背景模糊、目标小且稀疏的问题。通过引入语言先验和多模态数据提升检测效果。**

- **链接: [http://arxiv.org/pdf/2507.13113v1](http://arxiv.org/pdf/2507.13113v1)**

> **作者:** Pranav Singh; Pravendra Singh
>
> **摘要:** IRSTD (InfraRed Small Target Detection) detects small targets in infrared blurry backgrounds and is essential for various applications. The detection task is challenging due to the small size of the targets and their sparse distribution in infrared small target datasets. Although existing IRSTD methods and datasets have led to significant advancements, they are limited by their reliance solely on the image modality. Recent advances in deep learning and large vision-language models have shown remarkable performance in various visual recognition tasks. In this work, we propose a novel multimodal IRSTD framework that incorporates language priors to guide small target detection. We leverage language-guided attention weights derived from the language prior to enhance the model's ability for IRSTD, presenting a novel approach that combines textual information with image data to improve IRSTD capabilities. Utilizing the state-of-the-art GPT-4 vision model, we generate text descriptions that provide the locations of small targets in infrared images, employing careful prompt engineering to ensure improved accuracy. Due to the absence of multimodal IR datasets, existing IRSTD methods rely solely on image data. To address this shortcoming, we have curated a multimodal infrared dataset that includes both image and text modalities for small target detection, expanding upon the popular IRSTD-1k and NUDT-SIRST datasets. We validate the effectiveness of our approach through extensive experiments and comprehensive ablation studies. The results demonstrate significant improvements over the state-of-the-art method, with relative percentage differences of 9.74%, 13.02%, 1.25%, and 67.87% in IoU, nIoU, Pd, and Fa on the NUAA-SIRST subset, and 4.41%, 2.04%, 2.01%, and 113.43% on the IRSTD-1k subset of the LangIR dataset, respectively.
>
---
#### [new 063] MindJourney: Test-Time Scaling with World Models for Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于空间推理任务，解决VLM在3D场景理解上的不足。通过结合世界模型与VLM，实现测试时的动态推理，提升空间任务性能。**

- **链接: [http://arxiv.org/pdf/2507.12508v1](http://arxiv.org/pdf/2507.12508v1)**

> **作者:** Yuncong Yang; Jiageng Liu; Zheyuan Zhang; Siyuan Zhou; Reuben Tan; Jianwei Yang; Yilun Du; Chuang Gan
>
> **备注:** Project Page: https://umass-embodied-agi.github.io/MindJourney
>
> **摘要:** Spatial reasoning in 3D space is central to human cognition and indispensable for embodied tasks such as navigation and manipulation. However, state-of-the-art vision-language models (VLMs) struggle frequently with tasks as simple as anticipating how a scene will look after an egocentric motion: they perceive 2D images but lack an internal model of 3D dynamics. We therefore propose MindJourney, a test-time scaling framework that grants a VLM with this missing capability by coupling it to a controllable world model based on video diffusion. The VLM iteratively sketches a concise camera trajectory, while the world model synthesizes the corresponding view at each step. The VLM then reasons over this multi-view evidence gathered during the interactive exploration. Without any fine-tuning, our MindJourney achieves over an average 8% performance boost on the representative spatial reasoning benchmark SAT, showing that pairing VLMs with world models for test-time scaling offers a simple, plug-and-play route to robust 3D reasoning. Meanwhile, our method also improves upon the test-time inference VLMs trained through reinforcement learning, which demonstrates the potential of our method that utilizes world models for test-time scaling.
>
---
#### [new 064] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning
- **分类: cs.CV**

- **简介: 该论文属于可见-红外行人重识别任务，解决缺乏交叉模态标签时的模型训练问题。通过构建异构专家协作一致性学习框架，提升跨模态特征提取与身份识别能力。**

- **链接: [http://arxiv.org/pdf/2507.12942v1](http://arxiv.org/pdf/2507.12942v1)**

> **作者:** Yafei Zhang; Lingqi Kong; Huafeng Li; Jie Wen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** To reduce the reliance of visible-infrared person re-identification (ReID) models on labeled cross-modal samples, this paper explores a weakly supervised cross-modal person ReID method that uses only single-modal sample identity labels, addressing scenarios where cross-modal identity labels are unavailable. To mitigate the impact of missing cross-modal labels on model performance, we propose a heterogeneous expert collaborative consistency learning framework, designed to establish robust cross-modal identity correspondences in a weakly supervised manner. This framework leverages labeled data from each modality to independently train dedicated classification experts. To associate cross-modal samples, these classification experts act as heterogeneous predictors, predicting the identities of samples from the other modality. To improve prediction accuracy, we design a cross-modal relationship fusion mechanism that effectively integrates predictions from different experts. Under the implicit supervision provided by cross-modal identity correspondences, collaborative and consistent learning among the experts is encouraged, significantly enhancing the model's ability to extract modality-invariant features and improve cross-modal identity recognition. Experimental results on two challenging datasets validate the effectiveness of the proposed method.
>
---
#### [new 065] cIDIR: Conditioned Implicit Neural Representation for Regularized Deformable Image Registration
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像配准任务，解决传统方法中正则化参数调优计算成本高的问题。提出cIDIR框架，通过隐式神经表示条件化注册过程，提升配准精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.12953v1](http://arxiv.org/pdf/2507.12953v1)**

> **作者:** Sidaty El Hadramy; Oumeymah Cherkaoui; Philippe C. Cattin
>
> **摘要:** Regularization is essential in deformable image registration (DIR) to ensure that the estimated Deformation Vector Field (DVF) remains smooth, physically plausible, and anatomically consistent. However, fine-tuning regularization parameters in learning-based DIR frameworks is computationally expensive, often requiring multiple training iterations. To address this, we propose cIDI, a novel DIR framework based on Implicit Neural Representations (INRs) that conditions the registration process on regularization hyperparameters. Unlike conventional methods that require retraining for each regularization hyperparameter setting, cIDIR is trained over a prior distribution of these hyperparameters, then optimized over the regularization hyperparameters by using the segmentations masks as an observation. Additionally, cIDIR models a continuous and differentiable DVF, enabling seamless integration of advanced regularization techniques via automatic differentiation. Evaluated on the DIR-LAB dataset, $\operatorname{cIDIR}$ achieves high accuracy and robustness across the dataset.
>
---
#### [new 066] AthleticsPose: Authentic Sports Motion Dataset on Athletic Field and Evaluation of Monocular 3D Pose Estimation Ability
- **分类: cs.CV**

- **简介: 该论文属于单目3D姿态估计任务，旨在解决体育动作数据不足和模型可靠性问题。构建了AthleticsPose数据集，并验证了其在体育分析中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.12905v1](http://arxiv.org/pdf/2507.12905v1)**

> **作者:** Tomohiro Suzuki; Ryota Tanaka; Calvin Yeung; Keisuke Fujii
>
> **备注:** 9 pages, 5 figures, 5 tables
>
> **摘要:** Monocular 3D pose estimation is a promising, flexible alternative to costly motion capture systems for sports analysis. However, its practical application is hindered by two factors: a lack of realistic sports datasets and unclear reliability for sports tasks. To address these challenges, we introduce the AthleticsPose dataset, a new public dataset featuring ``real'' motions captured from 23 athletes performing various athletics events on an athletic field. Using this dataset, we trained a representative 3D pose estimation model and performed a comprehensive evaluation. Our results show that the model trained on AthleticsPose significantly outperforms a baseline model trained on an imitated sports motion dataset, reducing MPJPE by approximately 75 %. These results show the importance of training on authentic sports motion data, as models based on imitated motions do not effectively transfer to real-world motions. Further analysis reveals that estimation accuracy is sensitive to camera view and subject scale. In case studies of kinematic indicators, the model demonstrated the potential to capture individual differences in knee angles but struggled with higher-speed metrics, such as knee-drive velocity, due to prediction biases. This work provides the research community with a valuable dataset and clarifies the potential and practical limitations of using monocular 3D pose estimation for sports motion analysis. Our dataset, code, and checkpoints are available at https://github.com/SZucchini/AthleticsPose.
>
---
#### [new 067] Imbalance in Balance: Online Concept Balancing in Generation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉生成任务，解决复杂概念响应不稳定的问题。通过设计IMBA损失函数实现在线概念平衡，提升模型表现。**

- **链接: [http://arxiv.org/pdf/2507.13345v1](http://arxiv.org/pdf/2507.13345v1)**

> **作者:** Yukai Shi; Jiarong Ou; Rui Chen; Haotian Yang; Jiahao Wang; Xin Tao; Pengfei Wan; Di Zhang; Kun Gai
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** In visual generation tasks, the responses and combinations of complex concepts often lack stability and are error-prone, which remains an under-explored area. In this paper, we attempt to explore the causal factors for poor concept responses through elaborately designed experiments. We also design a concept-wise equalization loss function (IMBA loss) to address this issue. Our proposed method is online, eliminating the need for offline dataset processing, and requires minimal code changes. In our newly proposed complex concept benchmark Inert-CompBench and two other public test sets, our method significantly enhances the concept response capability of baseline models and yields highly competitive results with only a few codes.
>
---
#### [new 068] FashionPose: Text to Pose to Relight Image Generation for Personalized Fashion Visualization
- **分类: cs.CV**

- **简介: 该论文属于服装可视化任务，解决个性化姿态与光照控制问题。通过文本驱动生成人体姿态和图像，并实现高效逼真的光影调整。**

- **链接: [http://arxiv.org/pdf/2507.13311v1](http://arxiv.org/pdf/2507.13311v1)**

> **作者:** Chuancheng Shi; Yixiang Chen; Burong Lei; Jichao Chen
>
> **摘要:** Realistic and controllable garment visualization is critical for fashion e-commerce, where users expect personalized previews under diverse poses and lighting conditions. Existing methods often rely on predefined poses, limiting semantic flexibility and illumination adaptability. To address this, we introduce FashionPose, the first unified text-to-pose-to-relighting generation framework. Given a natural language description, our method first predicts a 2D human pose, then employs a diffusion model to generate high-fidelity person images, and finally applies a lightweight relighting module, all guided by the same textual input. By replacing explicit pose annotations with text-driven conditioning, FashionPose enables accurate pose alignment, faithful garment rendering, and flexible lighting control. Experiments demonstrate fine-grained pose synthesis and efficient, consistent relighting, providing a practical solution for personalized virtual fashion display.
>
---
#### [new 069] Variance-Based Pruning for Accelerating and Compressing Trained Networks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决结构化剪枝后性能下降的问题。提出基于方差的剪枝方法，在少量微调下实现模型加速与压缩。**

- **链接: [http://arxiv.org/pdf/2507.12988v1](http://arxiv.org/pdf/2507.12988v1)**

> **作者:** Uranik Berisha; Jens Mehnert; Alexandru Paul Condurache
>
> **备注:** Accepted at IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Increasingly expensive training of ever larger models such as Vision Transfomers motivate reusing the vast library of already trained state-of-the-art networks. However, their latency, high computational costs and memory demands pose significant challenges for deployment, especially on resource-constrained hardware. While structured pruning methods can reduce these factors, they often require costly retraining, sometimes for up to hundreds of epochs, or even training from scratch to recover the lost accuracy resulting from the structural modifications. Maintaining the provided performance of trained models after structured pruning and thereby avoiding extensive retraining remains a challenge. To solve this, we introduce Variance-Based Pruning, a simple and structured one-shot pruning technique for efficiently compressing networks, with minimal finetuning. Our approach first gathers activation statistics, which are used to select neurons for pruning. Simultaneously the mean activations are integrated back into the model to preserve a high degree of performance. On ImageNet-1k recognition tasks, we demonstrate that directly after pruning DeiT-Base retains over 70% of its original performance and requires only 10 epochs of fine-tuning to regain 99% of the original accuracy while simultaneously reducing MACs by 35% and model size by 36%, thus speeding up the model by 1.44x.
>
---
#### [new 070] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于植物建模任务，旨在解决叶子形状和变形难以建模的问题。提出NeuraLeaf模型，分离2D形状与3D变形，并构建新数据集以提升重建精度。**

- **链接: [http://arxiv.org/pdf/2507.12714v1](http://arxiv.org/pdf/2507.12714v1)**

> **作者:** Yang Yang; Dongni Mao; Hiroaki Santo; Yasuyuki Matsushita; Fumio Okura
>
> **备注:** IEEE/CVF International Conference on Computer Vision (ICCV 2025), Project: https://neuraleaf-yang.github.io/
>
> **摘要:** We develop a neural parametric model for 3D leaves for plant modeling and reconstruction that are essential for agriculture and computer graphics. While neural parametric models are actively studied for humans and animals, plant leaves present unique challenges due to their diverse shapes and flexible deformation. To this problem, we introduce a neural parametric model for leaves, NeuraLeaf. Capitalizing on the fact that flattened leaf shapes can be approximated as a 2D plane, NeuraLeaf disentangles the leaves' geometry into their 2D base shapes and 3D deformations. This representation allows learning from rich sources of 2D leaf image datasets for the base shapes, and also has the advantage of simultaneously learning textures aligned with the geometry. To model the 3D deformation, we propose a novel skeleton-free skinning model and create a newly captured 3D leaf dataset called DeformLeaf. We show that NeuraLeaf successfully generates a wide range of leaf shapes with deformation, resulting in accurate model fitting to 3D observations like depth maps and point clouds. Our implementation and dataset are available at https://neuraleaf-yang.github.io/.
>
---
#### [new 071] Advancing Complex Wide-Area Scene Understanding with Hierarchical Coresets Selection
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的场景理解任务，旨在解决VLMs在复杂大范围场景适应性不足的问题。提出HCS机制，通过分层核心选择提升模型对未见场景的理解能力。**

- **链接: [http://arxiv.org/pdf/2507.13061v1](http://arxiv.org/pdf/2507.13061v1)**

> **作者:** Jingyao Wang; Yiming Chen; Lingyu Si; Changwen Zheng
>
> **摘要:** Scene understanding is one of the core tasks in computer vision, aiming to extract semantic information from images to identify objects, scene categories, and their interrelationships. Although advancements in Vision-Language Models (VLMs) have driven progress in this field, existing VLMs still face challenges in adaptation to unseen complex wide-area scenes. To address the challenges, this paper proposes a Hierarchical Coresets Selection (HCS) mechanism to advance the adaptation of VLMs in complex wide-area scene understanding. It progressively refines the selected regions based on the proposed theoretically guaranteed importance function, which considers utility, representativeness, robustness, and synergy. Without requiring additional fine-tuning, HCS enables VLMs to achieve rapid understandings of unseen scenes at any scale using minimal interpretable regions while mitigating insufficient feature density. HCS is a plug-and-play method that is compatible with any VLM. Experiments demonstrate that HCS achieves superior performance and universality in various tasks.
>
---
#### [new 072] Resurrect Mask AutoRegressive Modeling for Efficient and Scalable Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决MAR模型性能不足的问题。通过改进架构和优化 tokenizer，提出MaskGIL模型，在保持高质量的同时显著提升生成效率。**

- **链接: [http://arxiv.org/pdf/2507.13032v1](http://arxiv.org/pdf/2507.13032v1)**

> **作者:** Yi Xin; Le Zhuo; Qi Qin; Siqi Luo; Yuewen Cao; Bin Fu; Yangfan He; Hongsheng Li; Guangtao Zhai; Xiaohong Liu; Peng Gao
>
> **备注:** 24 pages, 10 figures, 10 tables
>
> **摘要:** AutoRegressive (AR) models have made notable progress in image generation, with Masked AutoRegressive (MAR) models gaining attention for their efficient parallel decoding. However, MAR models have traditionally underperformed when compared to standard AR models. This study refines the MAR architecture to improve image generation quality. We begin by evaluating various image tokenizers to identify the most effective one. Subsequently, we introduce an improved Bidirectional LLaMA architecture by replacing causal attention with bidirectional attention and incorporating 2D RoPE, which together form our advanced model, MaskGIL. Scaled from 111M to 1.4B parameters, MaskGIL achieves a FID score of 3.71, matching state-of-the-art AR models in the ImageNet 256x256 benchmark, while requiring only 8 inference steps compared to the 256 steps of AR models. Furthermore, we develop a text-driven MaskGIL model with 775M parameters for generating images from text at various resolutions. Beyond image generation, MaskGIL extends to accelerate AR-based generation and enable real-time speech-to-image conversion. Our codes and models are available at https://github.com/synbol/MaskGIL.
>
---
#### [new 073] Synthesizing Reality: Leveraging the Generative AI-Powered Platform Midjourney for Construction Worker Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测任务，旨在解决施工场景中数据不足的问题。通过Midjourney生成12,000张合成图像，提升DNN训练效果。**

- **链接: [http://arxiv.org/pdf/2507.13221v1](http://arxiv.org/pdf/2507.13221v1)**

> **作者:** Hongyang Zhao; Tianyu Liang; Sina Davari; Daeho Kim
>
> **备注:** This work was presented at ASCE International Conference on Computing in Civil Engineering (i3CE) 2024 and is currently under consideration for publication in ASCE proceedings
>
> **摘要:** While recent advancements in deep neural networks (DNNs) have substantially enhanced visual AI's capabilities, the challenge of inadequate data diversity and volume remains, particularly in construction domain. This study presents a novel image synthesis methodology tailored for construction worker detection, leveraging the generative-AI platform Midjourney. The approach entails generating a collection of 12,000 synthetic images by formulating 3000 different prompts, with an emphasis on image realism and diversity. These images, after manual labeling, serve as a dataset for DNN training. Evaluation on a real construction image dataset yielded promising results, with the model attaining average precisions (APs) of 0.937 and 0.642 at intersection-over-union (IoU) thresholds of 0.5 and 0.5 to 0.95, respectively. Notably, the model demonstrated near-perfect performance on the synthetic dataset, achieving APs of 0.994 and 0.919 at the two mentioned thresholds. These findings reveal both the potential and weakness of generative AI in addressing DNN training data scarcity.
>
---
#### [new 074] AnyCap Project: A Unified Framework, Dataset, and Benchmark for Controllable Omni-modal Captioning
- **分类: cs.CV**

- **简介: 该论文属于多模态caption任务，解决可控性不足与评估不准确的问题。提出AnyCap框架、数据集和评估基准，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2507.12841v1](http://arxiv.org/pdf/2507.12841v1)**

> **作者:** Yiming Ren; Zhiqiang Lin; Yu Li; Gao Meng; Weiyun Wang; Junjie Wang; Zicheng Lin; Jifeng Dai; Yujiu Yang; Wenhai Wang; Ruihang Chu
>
> **摘要:** Controllable captioning is essential for precise multimodal alignment and instruction following, yet existing models often lack fine-grained control and reliable evaluation protocols. To address this gap, we present the AnyCap Project, an integrated solution spanning model, dataset, and evaluation. We introduce AnyCapModel (ACM), a lightweight plug-and-play framework that enhances the controllability of existing foundation models for omni-modal captioning without retraining the base model. ACM reuses the original captions from base models while incorporating user instructions and modality features to generate improved captions. To remedy the data scarcity in controllable multimodal captioning, we build AnyCapDataset (ACD), covering three modalities, 28 user-instruction types, and 300\,k high-quality data entries. We further propose AnyCapEval, a new benchmark that provides more reliable evaluation metrics for controllable captioning by decoupling content accuracy and stylistic fidelity. ACM markedly improves caption quality across a diverse set of base models on AnyCapEval. Notably, ACM-8B raises GPT-4o\'s content scores by 45\% and style scores by 12\%, and it also achieves substantial gains on widely used benchmarks such as MIA-Bench and VidCapBench.
>
---
#### [new 075] Best Practices for Large-Scale, Pixel-Wise Crop Mapping and Transfer Learning Workflows
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于作物分类任务，旨在解决大规模像素级作物映射问题。通过比较传统方法与迁移学习技术，评估不同预处理和模型效果，提出最优工作流程。**

- **链接: [http://arxiv.org/pdf/2507.12590v1](http://arxiv.org/pdf/2507.12590v1)**

> **作者:** Judy Long; Tao Liu; Sean Alexander Woznicki; Miljana Marković; Oskar Marko; Molly Sears
>
> **备注:** A review article. 41 pages, 22 figures. Preprint
>
> **摘要:** Crop mapping involves identifying and classifying crop types using spatial data, primarily derived from remote sensing imagery. This study presents the first comprehensive review of large-scale, pixel-wise crop mapping workflows, encompassing both conventional supervised methods and emerging transfer learning approaches. To identify the optimal supervised crop mapping workflows, we conducted systematic experiments, comparing six widely adopted satellite image-based preprocessing methods, alongside eleven supervised pixel-wise classification models. Additionally, we assessed the synergistic impact of varied training sample sizes and variable combinations. Moreover, we identified optimal transfer learning techniques for different magnitudes of domain shift. The evaluation of best methods was conducted across five diverse agricultural sites. Landsat 8 served as the primary satellite data source. Labels come from CDL trusted pixels and field surveys. Our findings reveal three key insights. First, fine-scale interval preprocessing paired with Transformer models consistently delivered optimal performance for both supervised and transferable workflows. RF offered rapid training and competitive performance in conventional supervised learning and direct transfer to similar domains. Second, transfer learning techniques enhanced workflow adaptability, with UDA being effective for homogeneous crop classes while fine-tuning remains robust across diverse scenarios. Finally, workflow choice depends heavily on the availability of labeled samples. With a sufficient sample size, supervised training typically delivers more accurate and generalizable results. Below a certain threshold, transfer learning that matches the level of domain shift is a viable alternative to achieve crop mapping. Repository: Best-Practices-for-Large-Scale-Pixel-Wise-Crop-Mapping-and-Transfer-Learning-Workflows
>
---
#### [new 076] Domain-Enhanced Dual-Branch Model for Efficient and Interpretable Accident Anticipation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于交通事故预测任务，旨在提高预测精度与可解释性。通过双分支模型融合视频与文本数据，结合大模型进行特征聚合，提升系统效率与响应能力。**

- **链接: [http://arxiv.org/pdf/2507.12755v1](http://arxiv.org/pdf/2507.12755v1)**

> **作者:** Yanchen Guan; Haicheng Liao; Chengyue Wang; Bonan Wang; Jiaxun Zhang; Jia Hu; Zhenning Li
>
> **摘要:** Developing precise and computationally efficient traffic accident anticipation system is crucial for contemporary autonomous driving technologies, enabling timely intervention and loss prevention. In this paper, we propose an accident anticipation framework employing a dual-branch architecture that effectively integrates visual information from dashcam videos with structured textual data derived from accident reports. Furthermore, we introduce a feature aggregation method that facilitates seamless integration of multimodal inputs through large models (GPT-4o, Long-CLIP), complemented by targeted prompt engineering strategies to produce actionable feedback and standardized accident archives. Comprehensive evaluations conducted on benchmark datasets (DAD, CCD, and A3D) validate the superior predictive accuracy, enhanced responsiveness, reduced computational overhead, and improved interpretability of our approach, thus establishing a new benchmark for state-of-the-art performance in traffic accident anticipation.
>
---
#### [new 077] Simulate, Refocus and Ensemble: An Attention-Refocusing Scheme for Domain Generalization
- **分类: cs.CV**

- **简介: 该论文属于领域泛化任务，旨在解决CLIP在跨域任务中难以关注不变特征的问题。提出SRE方法通过模拟域转移、注意力重聚焦和集成学习提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.12851v1](http://arxiv.org/pdf/2507.12851v1)**

> **作者:** Ziyi Wang; Zhi Gao; Jin Chen; Qingjie Zhao; Xinxiao Wu; Jiebo Luo
>
> **备注:** \c{opyright} 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Domain generalization (DG) aims to learn a model from source domains and apply it to unseen target domains with out-of-distribution data. Owing to CLIP's strong ability to encode semantic concepts, it has attracted increasing interest in domain generalization. However, CLIP often struggles to focus on task-relevant regions across domains, i.e., domain-invariant regions, resulting in suboptimal performance on unseen target domains. To address this challenge, we propose an attention-refocusing scheme, called Simulate, Refocus and Ensemble (SRE), which learns to reduce the domain shift by aligning the attention maps in CLIP via attention refocusing. SRE first simulates domain shifts by performing augmentation on the source data to generate simulated target domains. SRE then learns to reduce the domain shifts by refocusing the attention in CLIP between the source and simulated target domains. Finally, SRE utilizes ensemble learning to enhance the ability to capture domain-invariant attention maps between the source data and the simulated target data. Extensive experimental results on several datasets demonstrate that SRE generally achieves better results than state-of-the-art methods. The code is available at: https://github.com/bitPrincy/SRE-DG.
>
---
#### [new 078] Semantic-guided Fine-tuning of Foundation Model for Long-tailed Visual Recognition
- **分类: cs.CV**

- **简介: 该论文属于长尾视觉识别任务，旨在解决类别样本不平衡导致的性能下降问题。提出Sage方法，通过语义引导优化视觉编码器，并引入补偿因子纠正预测偏差。**

- **链接: [http://arxiv.org/pdf/2507.12807v1](http://arxiv.org/pdf/2507.12807v1)**

> **作者:** Yufei Peng; Yonggang Zhang; Yiu-ming Cheung
>
> **摘要:** The variance in class-wise sample sizes within long-tailed scenarios often results in degraded performance in less frequent classes. Fortunately, foundation models, pre-trained on vast open-world datasets, demonstrate strong potential for this task due to their generalizable representation, which promotes the development of adaptive strategies on pre-trained models in long-tailed learning. Advanced fine-tuning methods typically adjust visual encoders while neglecting the semantics derived from the frozen text encoder, overlooking the visual and textual alignment. To strengthen this alignment, we propose a novel approach, Semantic-guided fine-tuning of foundation model for long-tailed visual recognition (Sage), which incorporates semantic guidance derived from textual modality into the visual fine-tuning process. Specifically, we introduce an SG-Adapter that integrates class descriptions as semantic guidance to guide the fine-tuning of the visual encoder. The introduced guidance is passesed through the attention mechanism and enables the model to focus more on semantically relevant content, strengthening the alignment between the visual and textual modalities. Due to the inconsistent class-conditional distributions neglected by the existing loss function, the resulting prediction bias causes performance improvements for the tail class less than for the head class, even when the multi-modal alignment is enhanced. To address this challenge, we propose a novel distribution mismatch-aware compensation factor, which is specifically designed to rectify the prediction bias caused by the ignored inconsistent distribution based on our theoretical analysis, and is seamlessly integrated into the loss function. Extensive experiments on benchmark datasets demonstrate the effectiveness of the proposed Sage in enhancing performance in long-tailed learning.
>
---
#### [new 079] GLAD: Generalizable Tuning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的微调任务，旨在解决少样本下的过拟合问题。提出GLAD框架，结合LoRA与梯度正则化，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.13089v1](http://arxiv.org/pdf/2507.13089v1)**

> **作者:** Yuqi Peng; Pengfei Wang; Jianzhuang Liu; Shifeng Chen
>
> **备注:** ICCV 2025 workshop
>
> **摘要:** Pre-trained vision-language models, such as CLIP, show impressive zero-shot recognition ability and can be easily transferred to specific downstream tasks via prompt tuning, even with limited training data. However, existing prompt tuning methods face two main challenges: (1) In few-shot scenarios, data scarcity often leads to overfitting, making the model sensitive to changes in the input domain. (2) To mitigate overfitting, these methods typically rely on complex task-specific model architectures and sensitive hyperparameter tuning, severely restricting their general applicability. To address these issues, we propose a simpler and more general framework called GLAD (Generalizable LoRA tuning with RegulArized GraDient). We show that merely applying LoRA achieves performance in downstream tasks comparable to current state-of-the-art prompt-based methods. While LoRA is effective and easy to use, it remains susceptible to overfitting in few-shot learning scenarios. To mitigate this risk, we introduce a gradient-based regularization technique. This technique effectively steers the optimization trajectory, encouraging the model to find a more stable parameter region that is robust to variations in data distribution. Through extensive experiments conducted on 15 benchmark datasets, we demonstrate that GLAD outperforms previous tuning approaches in terms of base-to-novel class generalization, image domain generalization, and cross-dataset generalization. The code will be publicly available.
>
---
#### [new 080] City-VLM: Towards Multidomain Perception Scene Understanding via Multimodal Incomplete Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于户外场景理解任务，解决现有模型在多模态数据融合和大规模场景适应上的不足，提出City-VLM模型与SVM-City数据集。**

- **链接: [http://arxiv.org/pdf/2507.12795v1](http://arxiv.org/pdf/2507.12795v1)**

> **作者:** Penglei Sun; Yaoxian Song; Xiangru Zhu; Xiang Liu; Qiang Wang; Yue Liu; Changqun Xia; Tiefeng Li; Yang Yang; Xiaowen Chu
>
> **摘要:** Scene understanding enables intelligent agents to interpret and comprehend their environment. While existing large vision-language models (LVLMs) for scene understanding have primarily focused on indoor household tasks, they face two significant limitations when applied to outdoor large-scale scene understanding. First, outdoor scenarios typically encompass larger-scale environments observed through various sensors from multiple viewpoints (e.g., bird view and terrestrial view), while existing indoor LVLMs mainly analyze single visual modalities within building-scale contexts from humanoid viewpoints. Second, existing LVLMs suffer from missing multidomain perception outdoor data and struggle to effectively integrate 2D and 3D visual information. To address the aforementioned limitations, we build the first multidomain perception outdoor scene understanding dataset, named \textbf{\underline{SVM-City}}, deriving from multi\textbf{\underline{S}}cale scenarios with multi\textbf{\underline{V}}iew and multi\textbf{\underline{M}}odal instruction tuning data. It contains $420$k images and $4, 811$M point clouds with $567$k question-answering pairs from vehicles, low-altitude drones, high-altitude aerial planes, and satellite. To effectively fuse the multimodal data in the absence of one modality, we introduce incomplete multimodal learning to model outdoor scene understanding and design the LVLM named \textbf{\underline{City-VLM}}. Multimodal fusion is realized by constructing a joint probabilistic distribution space rather than implementing directly explicit fusion operations (e.g., concatenation). Experimental results on three typical outdoor scene understanding tasks show City-VLM achieves $18.14 \%$ performance surpassing existing LVLMs in question-answering tasks averagely. Our method demonstrates pragmatic and generalization performance across multiple outdoor scenes.
>
---
#### [new 081] WhoFi: Deep Person Re-Identification via Wi-Fi Channel Signal Encoding
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于行人重识别任务，旨在解决视觉方法在光照、遮挡等环境下的性能问题。通过Wi-Fi信号和深度学习方法提取生物特征，提升识别效果。**

- **链接: [http://arxiv.org/pdf/2507.12869v1](http://arxiv.org/pdf/2507.12869v1)**

> **作者:** Danilo Avola; Daniele Pannone; Dario Montagnini; Emad Emam
>
> **摘要:** Person Re-Identification is a key and challenging task in video surveillance. While traditional methods rely on visual data, issues like poor lighting, occlusion, and suboptimal angles often hinder performance. To address these challenges, we introduce WhoFi, a novel pipeline that utilizes Wi-Fi signals for person re-identification. Biometric features are extracted from Channel State Information (CSI) and processed through a modular Deep Neural Network (DNN) featuring a Transformer-based encoder. The network is trained using an in-batch negative loss function to learn robust and generalizable biometric signatures. Experiments on the NTU-Fi dataset show that our approach achieves competitive results compared to state-of-the-art methods, confirming its effectiveness in identifying individuals via Wi-Fi signals.
>
---
#### [new 082] From Neck to Head: Bio-Impedance Sensing for Head Pose Estimation
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于头姿估计任务，旨在无需视线的头姿追踪。通过生物阻抗传感和深度学习方法，实现高精度头姿估计。**

- **链接: [http://arxiv.org/pdf/2507.12884v1](http://arxiv.org/pdf/2507.12884v1)**

> **作者:** Mengxi Liu; Lala Shakti Swarup Ray; Sizhen Bian; Ko Watanabe; Ankur Bhatt; Joanna Sorysz; Russel Torah; Bo Zhou; Paul Lukowicz
>
> **摘要:** We present NeckSense, a novel wearable system for head pose tracking that leverages multi-channel bio-impedance sensing with soft, dry electrodes embedded in a lightweight, necklace-style form factor. NeckSense captures dynamic changes in tissue impedance around the neck, which are modulated by head rotations and subtle muscle activations. To robustly estimate head pose, we propose a deep learning framework that integrates anatomical priors, including joint constraints and natural head rotation ranges, into the loss function design. We validate NeckSense on 7 participants using the current SOTA pose estimation model as ground truth. Our system achieves a mean per-vertex error of 25.9 mm across various head movements with a leave-one-person-out cross-validation method, demonstrating that a compact, line-of-sight-free bio-impedance wearable can deliver head-tracking performance comparable to SOTA vision-based methods.
>
---
#### [new 083] 3DKeyAD: High-Resolution 3D Point Cloud Anomaly Detection via Keypoint-Guided Point Clustering
- **分类: cs.CV**

- **简介: 该论文属于3D点云异常检测任务，解决高分辨率点云中局部结构差异难以捕捉的问题。通过关键点引导聚类和多原型对齐，实现精确异常定位。**

- **链接: [http://arxiv.org/pdf/2507.13110v1](http://arxiv.org/pdf/2507.13110v1)**

> **作者:** Zi Wang; Katsuya Hotta; Koichiro Kamide; Yawen Zou; Chao Zhang; Jun Yu
>
> **摘要:** High-resolution 3D point clouds are highly effective for detecting subtle structural anomalies in industrial inspection. However, their dense and irregular nature imposes significant challenges, including high computational cost, sensitivity to spatial misalignment, and difficulty in capturing localized structural differences. This paper introduces a registration-based anomaly detection framework that combines multi-prototype alignment with cluster-wise discrepancy analysis to enable precise 3D anomaly localization. Specifically, each test sample is first registered to multiple normal prototypes to enable direct structural comparison. To evaluate anomalies at a local level, clustering is performed over the point cloud, and similarity is computed between features from the test sample and the prototypes within each cluster. Rather than selecting cluster centroids randomly, a keypoint-guided strategy is employed, where geometrically informative points are chosen as centroids. This ensures that clusters are centered on feature-rich regions, enabling more meaningful and stable distance-based comparisons. Extensive experiments on the Real3D-AD benchmark demonstrate that the proposed method achieves state-of-the-art performance in both object-level and point-level anomaly detection, even using only raw features.
>
---
#### [new 084] Leveraging Pre-Trained Visual Models for AI-Generated Video Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成视频检测任务，旨在解决识别AI生成视频的问题。通过利用预训练视觉模型提取特征，实现高效检测。**

- **链接: [http://arxiv.org/pdf/2507.13224v1](http://arxiv.org/pdf/2507.13224v1)**

> **作者:** Keerthi Veeramachaneni; Praveen Tirupattur; Amrit Singh Bedi; Mubarak Shah
>
> **摘要:** Recent advances in Generative AI (GenAI) have led to significant improvements in the quality of generated visual content. As AI-generated visual content becomes increasingly indistinguishable from real content, the challenge of detecting the generated content becomes critical in combating misinformation, ensuring privacy, and preventing security threats. Although there has been substantial progress in detecting AI-generated images, current methods for video detection are largely focused on deepfakes, which primarily involve human faces. However, the field of video generation has advanced beyond DeepFakes, creating an urgent need for methods capable of detecting AI-generated videos with generic content. To address this gap, we propose a novel approach that leverages pre-trained visual models to distinguish between real and generated videos. The features extracted from these pre-trained models, which have been trained on extensive real visual content, contain inherent signals that can help distinguish real from generated videos. Using these extracted features, we achieve high detection performance without requiring additional model training, and we further improve performance by training a simple linear classification layer on top of the extracted features. We validated our method on a dataset we compiled (VID-AID), which includes around 10,000 AI-generated videos produced by 9 different text-to-video models, along with 4,000 real videos, totaling over 7 hours of video content. Our evaluation shows that our approach achieves high detection accuracy, above 90% on average, underscoring its effectiveness. Upon acceptance, we plan to publicly release the code, the pre-trained models, and our dataset to support ongoing research in this critical area.
>
---
#### [new 085] RS-TinyNet: Stage-wise Feature Fusion Network for Detecting Tiny Objects in Remote Sensing Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像中微小目标检测任务，旨在解决因目标尺寸小、背景复杂导致的检测困难问题。通过设计多阶段特征融合网络提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.13120v1](http://arxiv.org/pdf/2507.13120v1)**

> **作者:** Xiaozheng Jiang; Wei Zhang; Xuerui Mao
>
> **摘要:** Detecting tiny objects in remote sensing (RS) imagery has been a long-standing challenge due to their extremely limited spatial information, weak feature representations, and dense distributions across complex backgrounds. Despite numerous efforts devoted, mainstream detectors still underperform in such scenarios. To bridge this gap, we introduce RS-TinyNet, a multi-stage feature fusion and enhancement model explicitly tailored for RS tiny object detection in various RS scenarios. RS-TinyNet comes with two novel designs: tiny object saliency modeling and feature integrity reconstruction. Guided by these principles, we design three step-wise feature enhancement modules. Among them, the multi-dimensional collaborative attention (MDCA) module employs multi-dimensional attention to enhance the saliency of tiny objects. Additionally, the auxiliary reversible branch (ARB) and a progressive fusion detection head (PFDH) module are introduced to preserve information flow and fuse multi-level features to bridge semantic gaps and retain structural detail. Comprehensive experiments on public RS dataset AI-TOD show that our RS-TinyNet surpasses existing state-of-the-art (SOTA) detectors by 4.0% AP and 6.5% AP75. Evaluations on DIOR benchmark dataset further validate its superior detection performance in diverse RS scenarios. These results demonstrate that the proposed multi-stage feature fusion strategy offers an effective and practical solution for tiny object detection in complex RS environments.
>
---
#### [new 086] Transformer-based Spatial Grounding: A Comprehensive Survey
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言定位任务，旨在系统梳理基于Transformer的空间接地方法，分析模型架构、数据集和评估指标，为研究者提供指导。**

- **链接: [http://arxiv.org/pdf/2507.12739v1](http://arxiv.org/pdf/2507.12739v1)**

> **作者:** Ijazul Haq; Muhammad Saqib; Yingjie Zhang
>
> **摘要:** Spatial grounding, the process of associating natural language expressions with corresponding image regions, has rapidly advanced due to the introduction of transformer-based models, significantly enhancing multimodal representation and cross-modal alignment. Despite this progress, the field lacks a comprehensive synthesis of current methodologies, dataset usage, evaluation metrics, and industrial applicability. This paper presents a systematic literature review of transformer-based spatial grounding approaches from 2018 to 2025. Our analysis identifies dominant model architectures, prevalent datasets, and widely adopted evaluation metrics, alongside highlighting key methodological trends and best practices. This study provides essential insights and structured guidance for researchers and practitioners, facilitating the development of robust, reliable, and industry-ready transformer-based spatial grounding models.
>
---
#### [new 087] HRSeg: High-Resolution Visual Perception and Enhancement for Reasoning Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，解决低分辨率感知限制问题。提出HRSeg模型，通过高分辨率感知与增强模块提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.12883v1](http://arxiv.org/pdf/2507.12883v1)**

> **作者:** Weihuang Lin; Yiwei Ma; Xiaoshuai Sun; Shuting He; Jiayi Ji; Liujuan Cao; Rongrong Ji
>
> **摘要:** The reasoning segmentation task involves segmenting objects within an image by interpreting implicit user instructions, which may encompass subtleties such as contextual cues and open-world knowledge. Despite significant advancements made by existing approaches, they remain constrained by low perceptual resolution, as visual encoders are typically pre-trained at lower resolutions. Furthermore, simply interpolating the positional embeddings of visual encoders to enhance perceptual resolution yields only marginal performance improvements while incurring substantial computational costs. To address this, we propose HRSeg, an efficient model with high-resolution fine-grained perception. It features two key innovations: High-Resolution Perception (HRP) and High-Resolution Enhancement (HRE). The HRP module processes high-resolution images through cropping, integrating local and global features for multi-granularity quality. The HRE module enhances mask features by integrating fine-grained information from high-resolution images, refining their alignment with text features for precise segmentation. Extensive ablation studies validate the effectiveness of our modules, while comprehensive experiments on multiple benchmark datasets demonstrate HRSeg's superior performance.
>
---
#### [new 088] FORTRESS: Function-composition Optimized Real-Time Resilient Structural Segmentation via Kolmogorov-Arnold Enhanced Spatial Attention Networks
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于结构缺陷分割任务，解决实时高精度分割难题。提出FORTRESS架构，结合深度可分离卷积与Kolmogorov-Arnold网络，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.12675v1](http://arxiv.org/pdf/2507.12675v1)**

> **作者:** Christina Thrainer; Md Meftahul Ferdaus; Mahdi Abdelguerfi; Christian Guetl; Steven Sloan; Kendall N. Niles; Ken Pathak
>
> **摘要:** Automated structural defect segmentation in civil infrastructure faces a critical challenge: achieving high accuracy while maintaining computational efficiency for real-time deployment. This paper presents FORTRESS (Function-composition Optimized Real-Time Resilient Structural Segmentation), a new architecture that balances accuracy and speed by using a special method that combines depthwise separable convolutions with adaptive Kolmogorov-Arnold Network integration. FORTRESS incorporates three key innovations: a systematic depthwise separable convolution framework achieving a 3.6x parameter reduction per layer, adaptive TiKAN integration that selectively applies function composition transformations only when computationally beneficial, and multi-scale attention fusion combining spatial, channel, and KAN-enhanced features across decoder levels. The architecture achieves remarkable efficiency gains with 91% parameter reduction (31M to 2.9M), 91% computational complexity reduction (13.7 to 1.17 GFLOPs), and 3x inference speed improvement while delivering superior segmentation performance. Evaluation on benchmark infrastructure datasets demonstrates state-of-the-art results with an F1- score of 0.771 and a mean IoU of 0.677, significantly outperforming existing methods including U-Net, SA-UNet, and U- KAN. The dual optimization strategy proves essential for optimal performance, establishing FORTRESS as a robust solution for practical structural defect segmentation in resource-constrained environments where both accuracy and computational efficiency are paramount. Comprehensive architectural specifications are provided in the Supplemental Material. Source code is available at URL: https://github.com/faeyelab/fortress-paper-code.
>
---
#### [new 089] A Deep-Learning Framework for Land-Sliding Classification from Remote Sensing Image
- **分类: cs.CV**

- **简介: 该论文属于土地滑坡分类任务，旨在解决遥感图像中滑坡自动检测的问题。通过结合数据增强、EfficientNet\_Large和SVM提升分类性能。**

- **链接: [http://arxiv.org/pdf/2507.12939v1](http://arxiv.org/pdf/2507.12939v1)**

> **作者:** Hieu Tang; Truong Vo; Dong Pham; Toan Nguyen; Lam Pham; Truong Nguyen
>
> **摘要:** The use of satellite imagery combined with deep learning to support automatic landslide detection is becoming increasingly widespread. However, selecting an appropriate deep learning architecture to optimize performance while avoiding overfitting remains a critical challenge. To address these issues, we propose a deep-learning based framework for landslide detection from remote sensing image in this paper. The proposed framework presents an effective combination of the online an offline data augmentation to tackle the imbalanced data, a backbone EfficientNet\_Large deep learning model for extracting robust embedding features, and a post-processing SVM classifier to balance and enhance the classification performance. The proposed model achieved an F1-score of 0.8938 on the public test set of the Zindi challenge.
>
---
#### [new 090] fastWDM3D: Fast and Accurate 3D Healthy Tissue Inpainting
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D健康组织修复任务，旨在解决DDPMs采样速度慢的问题。通过结合DDPM与GAN及方差保持噪声调度，提出fastWDM3D模型，在少量时间步内实现快速高精度修复。**

- **链接: [http://arxiv.org/pdf/2507.13146v1](http://arxiv.org/pdf/2507.13146v1)**

> **作者:** Alicia Durrer; Florentin Bieder; Paul Friedrich; Bjoern Menze; Philippe C. Cattin; Florian Kofler
>
> **备注:** Philippe C. Cattin and Florian Kofler: equal contribution
>
> **摘要:** Healthy tissue inpainting has significant applications, including the generation of pseudo-healthy baselines for tumor growth models and the facilitation of image registration. In previous editions of the BraTS Local Synthesis of Healthy Brain Tissue via Inpainting Challenge, denoising diffusion probabilistic models (DDPMs) demonstrated qualitatively convincing results but suffered from low sampling speed. To mitigate this limitation, we adapted a 2D image generation approach, combining DDPMs with generative adversarial networks (GANs) and employing a variance-preserving noise schedule, for the task of 3D inpainting. Our experiments showed that the variance-preserving noise schedule and the selected reconstruction losses can be effectively utilized for high-quality 3D inpainting in a few time steps without requiring adversarial training. We applied our findings to a different architecture, a 3D wavelet diffusion model (WDM3D) that does not include a GAN component. The resulting model, denoted as fastWDM3D, obtained a SSIM of 0.8571, a MSE of 0.0079, and a PSNR of 22.26 on the BraTS inpainting test set. Remarkably, it achieved these scores using only two time steps, completing the 3D inpainting process in 1.81 s per image. When compared to other DDPMs used for healthy brain tissue inpainting, our model is up to 800 x faster while still achieving superior performance metrics. Our proposed method, fastWDM3D, represents a promising approach for fast and accurate healthy tissue inpainting. Our code is available at https://github.com/AliciaDurrer/fastWDM3D.
>
---
#### [new 091] InSight: AI Mobile Screening Tool for Multiple Eye Disease Detection using Multimodal Fusion
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于多疾病眼病检测任务，旨在解决资源有限地区筛查困难的问题。通过融合多模态数据和改进模型结构，提升诊断准确性和效率。**

- **链接: [http://arxiv.org/pdf/2507.12669v1](http://arxiv.org/pdf/2507.12669v1)**

> **作者:** Ananya Raghu; Anisha Raghu; Alice S. Tang; Yannis M. Paulus; Tyson N. Kim; Tomiko T. Oskotsky
>
> **摘要:** Background/Objectives: Age-related macular degeneration, glaucoma, diabetic retinopathy (DR), diabetic macular edema, and pathological myopia affect hundreds of millions of people worldwide. Early screening for these diseases is essential, yet access to medical care remains limited in low- and middle-income countries as well as in resource-limited settings. We develop InSight, an AI-based app that combines patient metadata with fundus images for accurate diagnosis of five common eye diseases to improve accessibility of screenings. Methods: InSight features a three-stage pipeline: real-time image quality assessment, disease diagnosis model, and a DR grading model to assess severity. Our disease diagnosis model incorporates three key innovations: (a) Multimodal fusion technique (MetaFusion) combining clinical metadata and images; (b) Pretraining method leveraging supervised and self-supervised loss functions; and (c) Multitask model to simultaneously predict 5 diseases. We make use of BRSET (lab-captured images) and mBRSET (smartphone-captured images) datasets, both of which also contain clinical metadata for model training/evaluation. Results: Trained on a dataset of BRSET and mBRSET images, the image quality checker achieves near-100% accuracy in filtering out low-quality fundus images. The multimodal pretrained disease diagnosis model outperforms models using only images by 6% in balanced accuracy for BRSET and 4% for mBRSET. Conclusions: The InSight pipeline demonstrates robustness across varied image conditions and has high diagnostic accuracy across all five diseases, generalizing to both smartphone and lab captured images. The multitask model contributes to the lightweight nature of the pipeline, making it five times computationally efficient compared to having five individual models corresponding to each disease.
>
---
#### [new 092] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言导航任务，旨在解决虚拟与物理环境间的差距问题。通过构建VLN-PE平台，评估不同方法在物理机器人中的表现，揭示实际部署挑战。**

- **链接: [http://arxiv.org/pdf/2507.13019v1](http://arxiv.org/pdf/2507.13019v1)**

> **作者:** Liuyi Wang; Xinyuan Xia; Hui Zhao; Hanqing Wang; Tai Wang; Yilun Chen; Chengju Liu; Qijun Chen; Jiangmiao Pang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent Vision-and-Language Navigation (VLN) advancements are promising, but their idealized assumptions about robot movement and control fail to reflect physically embodied deployment challenges. To bridge this gap, we introduce VLN-PE, a physically realistic VLN platform supporting humanoid, quadruped, and wheeled robots. For the first time, we systematically evaluate several ego-centric VLN methods in physical robotic settings across different technical pipelines, including classification models for single-step discrete action prediction, a diffusion model for dense waypoint prediction, and a train-free, map-based large language model (LLM) integrated with path planning. Our results reveal significant performance degradation due to limited robot observation space, environmental lighting variations, and physical challenges like collisions and falls. This also exposes locomotion constraints for legged robots in complex environments. VLN-PE is highly extensible, allowing seamless integration of new scenes beyond MP3D, thereby enabling more comprehensive VLN evaluation. Despite the weak generalization of current models in physical deployment, VLN-PE provides a new pathway for improving cross-embodiment's overall adaptability. We hope our findings and tools inspire the community to rethink VLN limitations and advance robust, practical VLN models. The code is available at https://crystalsixone.github.io/vln_pe.github.io/.
>
---
#### [new 093] MUPAX: Multidimensional Problem Agnostic eXplainable AI
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出MUPAX，一种通用的可解释AI方法，解决模型可解释性问题。它在多种数据模态中实现精准、一致的特征重要性分析，提升模型准确性。**

- **链接: [http://arxiv.org/pdf/2507.13090v1](http://arxiv.org/pdf/2507.13090v1)**

> **作者:** Vincenzo Dentamaro; Felice Franchini; Giuseppe Pirlo; Irina Voiculescu
>
> **摘要:** Robust XAI techniques should ideally be simultaneously deterministic, model agnostic, and guaranteed to converge. We propose MULTIDIMENSIONAL PROBLEM AGNOSTIC EXPLAINABLE AI (MUPAX), a deterministic, model agnostic explainability technique, with guaranteed convergency. MUPAX measure theoretic formulation gives principled feature importance attribution through structured perturbation analysis that discovers inherent input patterns and eliminates spurious relationships. We evaluate MUPAX on an extensive range of data modalities and tasks: audio classification (1D), image classification (2D), volumetric medical image analysis (3D), and anatomical landmark detection, demonstrating dimension agnostic effectiveness. The rigorous convergence guarantees extend to any loss function and arbitrary dimensions, making MUPAX applicable to virtually any problem context for AI. By contrast with other XAI methods that typically decrease performance when masking, MUPAX not only preserves but actually enhances model accuracy by capturing only the most important patterns of the original data. Extensive benchmarking against the state of the XAI art demonstrates MUPAX ability to generate precise, consistent and understandable explanations, a crucial step towards explainable and trustworthy AI systems. The source code will be released upon publication.
>
---
#### [new 094] HairFormer: Transformer-Based Dynamic Neural Hair Simulation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于动态头发模拟任务，解决跨不同发型、体型和动作的通用性问题。提出基于Transformer的两阶段神经网络，实现高保真且无穿透的头发动态模拟。**

- **链接: [http://arxiv.org/pdf/2507.12600v1](http://arxiv.org/pdf/2507.12600v1)**

> **作者:** Joy Xiaoji Zhang; Jingsen Zhu; Hanyu Chen; Steve Marschner
>
> **摘要:** Simulating hair dynamics that generalize across arbitrary hairstyles, body shapes, and motions is a critical challenge. Our novel two-stage neural solution is the first to leverage Transformer-based architectures for such a broad generalization. We propose a Transformer-powered static network that predicts static draped shapes for any hairstyle, effectively resolving hair-body penetrations and preserving hair fidelity. Subsequently, a dynamic network with a novel cross-attention mechanism fuses static hair features with kinematic input to generate expressive dynamics and complex secondary motions. This dynamic network also allows for efficient fine-tuning of challenging motion sequences, such as abrupt head movements. Our method offers real-time inference for both static single-frame drapes and dynamic drapes over pose sequences. Our method demonstrates high-fidelity and generalizable dynamic hair across various styles, guided by physics-informed losses, and can resolve penetrations even for complex, unseen long hairstyles, highlighting its broad generalization.
>
---
#### [new 095] DASViT: Differentiable Architecture Search for Vision Transformer
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于神经网络架构搜索任务，旨在解决Vision Transformer设计效率低、创新性不足的问题。通过提出DASViT方法，实现更高效、更优的ViT架构搜索。**

- **链接: [http://arxiv.org/pdf/2507.13079v1](http://arxiv.org/pdf/2507.13079v1)**

> **作者:** Pengjin Wu; Ferrante Neri; Zhenhua Feng
>
> **备注:** Accepted to the International Joint Conference on Neural Networks (IJCNN) 2025
>
> **摘要:** Designing effective neural networks is a cornerstone of deep learning, and Neural Architecture Search (NAS) has emerged as a powerful tool for automating this process. Among the existing NAS approaches, Differentiable Architecture Search (DARTS) has gained prominence for its efficiency and ease of use, inspiring numerous advancements. Since the rise of Vision Transformers (ViT), researchers have applied NAS to explore ViT architectures, often focusing on macro-level search spaces and relying on discrete methods like evolutionary algorithms. While these methods ensure reliability, they face challenges in discovering innovative architectural designs, demand extensive computational resources, and are time-intensive. To address these limitations, we introduce Differentiable Architecture Search for Vision Transformer (DASViT), which bridges the gap in differentiable search for ViTs and uncovers novel designs. Experiments show that DASViT delivers architectures that break traditional Transformer encoder designs, outperform ViT-B/16 on multiple datasets, and achieve superior efficiency with fewer parameters and FLOPs.
>
---
#### [new 096] WaveletInception Networks for Drive-by Vibration-Based Infrastructure Health Monitoring
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于基础设施健康监测任务，解决驱动振动信号分析问题。提出WaveletInception-BiLSTM网络，融合时频特征与操作条件，实现自动化、高精度的结构健康评估。**

- **链接: [http://arxiv.org/pdf/2507.12969v1](http://arxiv.org/pdf/2507.12969v1)**

> **作者:** Reza Riahi Samani; Alfredo Nunez; Bart De Schutter
>
> **摘要:** This paper presents a novel deep learning-based framework for infrastructure health monitoring using drive-by vibration response signals. Recognizing the importance of spectral and temporal information, we introduce the WaveletInception-BiLSTM network. The WaveletInception feature extractor utilizes a Learnable Wavelet Packet Transform (LWPT) as the stem for extracting vibration signal features, incorporating spectral information in the early network layers. This is followed by 1D Inception networks that extract multi-scale, high-level features at deeper layers. The extracted vibration signal features are then integrated with operational conditions via a Long Short-term Memory (LSTM) layer. The resulting feature extraction network effectively analyzes drive-by vibration signals across various measurement speeds without preprocessing and uses LSTM to capture interrelated temporal dependencies among different modes of information and to create feature vectors for health condition estimation. The estimator head is designed with a sequential modeling architecture using bidirectional LSTM (BiLSTM) networks, capturing bi-directional temporal relationships from drive-by measurements. This architecture allows for a high-resolution, beam-level assessment of infrastructure health conditions. A case study focusing on railway track stiffness estimation with simulated drive-by vibration signals shows that the model significantly outperforms state-of-the-art methods in estimating railway ballast and railpad stiffness parameters. Results underscore the potential of this approach for accurate, localized, and fully automated drive-by infrastructure health monitoring.
>
---
#### [new 097] From Variability To Accuracy: Conditional Bernoulli Diffusion Models with Consensus-Driven Correction for Thin Structure Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，解决薄结构分割不准确问题。通过条件伯努利扩散模型和共识校正提高分割精度与连续性。**

- **链接: [http://arxiv.org/pdf/2507.12985v1](http://arxiv.org/pdf/2507.12985v1)**

> **作者:** Jinseo An; Min Jin Lee; Kyu Won Shim; Helen Hong
>
> **备注:** Early accepted at MICCAI 2025
>
> **摘要:** Accurate segmentation of orbital bones in facial computed tomography (CT) images is essential for the creation of customized implants for reconstruction of defected orbital bones, particularly challenging due to the ambiguous boundaries and thin structures such as the orbital medial wall and orbital floor. In these ambiguous regions, existing segmentation approaches often output disconnected or under-segmented results. We propose a novel framework that corrects segmentation results by leveraging consensus from multiple diffusion model outputs. Our approach employs a conditional Bernoulli diffusion model trained on diverse annotation patterns per image to generate multiple plausible segmentations, followed by a consensus-driven correction that incorporates position proximity, consensus level, and gradient direction similarity to correct challenging regions. Experimental results demonstrate that our method outperforms existing methods, significantly improving recall in ambiguous regions while preserving the continuity of thin structures. Furthermore, our method automates the manual process of segmentation result correction and can be applied to image-guided surgical planning and surgery.
>
---
#### [new 098] Improving Diagnostic Accuracy of Pigmented Skin Lesions With CNNs: an Application on the DermaMNIST Dataset
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在提升色素性皮肤病变的诊断准确性。通过使用CNN模型对DermaMNIST数据集进行多类分类，验证了模型的有效性。**

- **链接: [http://arxiv.org/pdf/2507.12961v1](http://arxiv.org/pdf/2507.12961v1)**

> **作者:** Nerma Kadric; Amila Akagic; Medina Kapo
>
> **摘要:** Pigmented skin lesions represent localized areas of increased melanin and can indicate serious conditions like melanoma, a major contributor to skin cancer mortality. The MedMNIST v2 dataset, inspired by MNIST, was recently introduced to advance research in biomedical imaging and includes DermaMNIST, a dataset for classifying pigmented lesions based on the HAM10000 dataset. This study assesses ResNet-50 and EfficientNetV2L models for multi-class classification using DermaMNIST, employing transfer learning and various layer configurations. One configuration achieves results that match or surpass existing methods. This study suggests that convolutional neural networks (CNNs) can drive progress in biomedical image analysis, significantly enhancing diagnostic accuracy.
>
---
#### [new 099] Pathology-Guided Virtual Staining Metric for Evaluation and Training
- **分类: eess.IV; cs.CV; cs.SY; eess.SY**

- **简介: 该论文属于虚拟染色评估任务，解决传统评估方法无法准确反映病理特征的问题。提出PaPIS指标，结合深度学习与Retinex理论，提升虚拟染色质量评估与模型训练效果。**

- **链接: [http://arxiv.org/pdf/2507.12624v1](http://arxiv.org/pdf/2507.12624v1)**

> **作者:** Qiankai Wang; James E. D. Tweel; Parsin Haji Reza; Anita Layton
>
> **备注:** 19 pages, 10 figures. Intended for submission to the Journal of Imaging Informatics in Medicine (JIIM)
>
> **摘要:** Virtual staining has emerged as a powerful alternative to traditional histopathological staining techniques, enabling rapid, reagent-free image transformations. However, existing evaluation methods predominantly rely on full-reference image quality assessment (FR-IQA) metrics such as structural similarity, which are originally designed for natural images and often fail to capture pathology-relevant features. Expert pathology reviews have also been used, but they are inherently subjective and time-consuming. In this study, we introduce PaPIS (Pathology-Aware Perceptual Image Similarity), a novel FR-IQA metric specifically tailored for virtual staining evaluation. PaPIS leverages deep learning-based features trained on cell morphology segmentation and incorporates Retinex-inspired feature decomposition to better reflect histological perceptual quality. Comparative experiments demonstrate that PaPIS more accurately aligns with pathology-relevant visual cues and distinguishes subtle cellular structures that traditional and existing perceptual metrics tend to overlook. Furthermore, integrating PaPIS as a guiding loss function in a virtual staining model leads to improved histological fidelity. This work highlights the critical need for pathology-aware evaluation frameworks to advance the development and clinical readiness of virtual staining technologies.
>
---
#### [new 100] SpectraLift: Physics-Guided Spectral-Inversion Network for Self-Supervised Hyperspectral Image Super-Resolution
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于高光谱图像超分辨率任务，解决如何在无PSF或真实HR-HSI情况下融合MSI与HSI的问题。提出SpectraLift框架，利用SRF进行自监督训练，实现高效高质图像重建。**

- **链接: [http://arxiv.org/pdf/2507.13339v1](http://arxiv.org/pdf/2507.13339v1)**

> **作者:** Ritik Shah; Marco F. Duarte
>
> **摘要:** High-spatial-resolution hyperspectral images (HSI) are essential for applications such as remote sensing and medical imaging, yet HSI sensors inherently trade spatial detail for spectral richness. Fusing high-spatial-resolution multispectral images (HR-MSI) with low-spatial-resolution hyperspectral images (LR-HSI) is a promising route to recover fine spatial structures without sacrificing spectral fidelity. Most state-of-the-art methods for HSI-MSI fusion demand point spread function (PSF) calibration or ground truth high resolution HSI (HR-HSI), both of which are impractical to obtain in real world settings. We present SpectraLift, a fully self-supervised framework that fuses LR-HSI and HR-MSI inputs using only the MSI's Spectral Response Function (SRF). SpectraLift trains a lightweight per-pixel multi-layer perceptron (MLP) network using ($i$)~a synthetic low-spatial-resolution multispectral image (LR-MSI) obtained by applying the SRF to the LR-HSI as input, ($ii$)~the LR-HSI as the output, and ($iii$)~an $\ell_1$ spectral reconstruction loss between the estimated and true LR-HSI as the optimization objective. At inference, SpectraLift uses the trained network to map the HR-MSI pixel-wise into a HR-HSI estimate. SpectraLift converges in minutes, is agnostic to spatial blur and resolution, and outperforms state-of-the-art methods on PSNR, SAM, SSIM, and RMSE benchmarks.
>
---
#### [new 101] Physically Based Neural LiDAR Resimulation
- **分类: cs.RO; cs.CV; cs.GR; eess.IV**

- **简介: 该论文属于LiDAR模拟任务，解决现有方法对传感器特性建模不足的问题，通过建模滚动快门、激光功率变化等实现更精确的LiDAR仿真。**

- **链接: [http://arxiv.org/pdf/2507.12489v1](http://arxiv.org/pdf/2507.12489v1)**

> **作者:** Richard Marcus; Marc Stamminger
>
> **备注:** Accepted at ITSC 2025, Gold Coast Australia
>
> **摘要:** Methods for Novel View Synthesis (NVS) have recently found traction in the field of LiDAR simulation and large-scale 3D scene reconstruction. While solutions for faster rendering or handling dynamic scenes have been proposed, LiDAR specific effects remain insufficiently addressed. By explicitly modeling sensor characteristics such as rolling shutter, laser power variations, and intensity falloff, our method achieves more accurate LiDAR simulation compared to existing techniques. We demonstrate the effectiveness of our approach through quantitative and qualitative comparisons with state-of-the-art methods, as well as ablation studies that highlight the importance of each sensor model component. Beyond that, we show that our approach exhibits advanced resimulation capabilities, such as generating high resolution LiDAR scans in the camera perspective. Our code and the resulting dataset are available at https://github.com/richardmarcus/PBNLiDAR.
>
---
#### [new 102] Multimodal-Guided Dynamic Dataset Pruning for Robust and Efficient Data-Centric Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于数据增强任务，旨在解决数据质量差和冗余问题。提出一种动态数据集剪枝方法，结合任务难度与跨模态语义一致性，提升模型训练效率和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.12750v1](http://arxiv.org/pdf/2507.12750v1)**

> **作者:** Suorong Yang; Peijia Li; Yujie Liu; Zhiming Xu; Peng Ye; Wanli Ouyang; Furao Shen; Dongzhan Zhou
>
> **摘要:** Modern deep models are trained on large real-world datasets, where data quality varies and redundancy is common. Data-centric approaches such as dataset pruning have shown promise in improving training efficiency and model performance. However, most existing methods rely on static heuristics or task-specific metrics, limiting their robustness and generalizability across domains. In this work, we introduce a dynamic dataset pruning framework that adaptively selects training samples based on both task-driven difficulty and cross-modality semantic consistency. By incorporating supervision from pretrained multimodal foundation models, our approach captures training dynamics while effectively filtering out uninformative samples. Our work highlights the potential of integrating cross-modality alignment for robust sample selection, advancing data-centric learning toward more efficient and robust practices across application domains.
>
---
#### [new 103] TRIQA: Image Quality Assessment by Contrastive Pretraining on Ordered Distortion Triplets
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像质量评估任务，解决无参考IQA数据不足的问题。通过对比三元组预训练，构建高效模型，提升预测性能。**

- **链接: [http://arxiv.org/pdf/2507.12687v1](http://arxiv.org/pdf/2507.12687v1)**

> **作者:** Rajesh Sureddi; Saman Zadtootaghaj; Nabajeet Barman; Alan C. Bovik
>
> **备注:** 5 pages
>
> **摘要:** Image Quality Assessment (IQA) models aim to predict perceptual image quality in alignment with human judgments. No-Reference (NR) IQA remains particularly challenging due to the absence of a reference image. While deep learning has significantly advanced this field, a major hurdle in developing NR-IQA models is the limited availability of subjectively labeled data. Most existing deep learning-based NR-IQA approaches rely on pre-training on large-scale datasets before fine-tuning for IQA tasks. To further advance progress in this area, we propose a novel approach that constructs a custom dataset using a limited number of reference content images and introduces a no-reference IQA model that incorporates both content and quality features for perceptual quality prediction. Specifically, we train a quality-aware model using contrastive triplet-based learning, enabling efficient training with fewer samples while achieving strong generalization performance across publicly available datasets. Our repository is available at https://github.com/rajeshsureddi/triqa.
>
---
#### [new 104] Pixel Perfect MegaMed: A Megapixel-Scale Vision-Language Foundation Model for Generating High Resolution Medical Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决高分辨率医疗图像合成中细节丢失的问题。工作包括设计多尺度Transformer模型，实现文本到高分辨率医学图像的生成。**

- **链接: [http://arxiv.org/pdf/2507.12698v1](http://arxiv.org/pdf/2507.12698v1)**

> **作者:** Zahra TehraniNasab; Amar Kumar; Tal Arbel
>
> **摘要:** Medical image synthesis presents unique challenges due to the inherent complexity and high-resolution details required in clinical contexts. Traditional generative architectures such as Generative Adversarial Networks (GANs) or Variational Auto Encoder (VAEs) have shown great promise for high-resolution image generation but struggle with preserving fine-grained details that are key for accurate diagnosis. To address this issue, we introduce Pixel Perfect MegaMed, the first vision-language foundation model to synthesize images at resolutions of 1024x1024. Our method deploys a multi-scale transformer architecture designed specifically for ultra-high resolution medical image generation, enabling the preservation of both global anatomical context and local image-level details. By leveraging vision-language alignment techniques tailored to medical terminology and imaging modalities, Pixel Perfect MegaMed bridges the gap between textual descriptions and visual representations at unprecedented resolution levels. We apply our model to the CheXpert dataset and demonstrate its ability to generate clinically faithful chest X-rays from text prompts. Beyond visual quality, these high-resolution synthetic images prove valuable for downstream tasks such as classification, showing measurable performance gains when used for data augmentation, particularly in low-data regimes. Our code is accessible through the project website - https://tehraninasab.github.io/pixelperfect-megamed.
>
---
#### [new 105] Tensor-Tensor Products, Group Representations, and Semidefinite Programming
- **分类: math.OC; cs.CV; cs.NA; math.NA; math.RT; 90C22, 15A69, 65F99**

- **简介: 该论文研究第三阶张量的$\star_M$-乘积，探讨其正半定性与半定规划，结合群表示理论解决不变半定规划问题，应用于非负二次型和低秩张量补全。**

- **链接: [http://arxiv.org/pdf/2507.12729v1](http://arxiv.org/pdf/2507.12729v1)**

> **作者:** Alex Dunbar; Elizabeth Newman
>
> **备注:** 34 Pages, 7 figures
>
> **摘要:** The $\star_M$-family of tensor-tensor products is a framework which generalizes many properties from linear algebra to third order tensors. Here, we investigate positive semidefiniteness and semidefinite programming under the $\star_M$-product. Critical to our investigation is a connection between the choice of matrix M in the $\star_M$-product and the representation theory of an underlying group action. Using this framework, third order tensors equipped with the $\star_M$-product are a natural setting for the study of invariant semidefinite programs. As applications of the M-SDP framework, we provide a characterization of certain nonnegative quadratic forms and solve low-rank tensor completion problems.
>
---
#### [new 106] Unleashing Vision Foundation Models for Coronary Artery Segmentation: Parallel ViT-CNN Encoding and Variational Fusion
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决冠状动脉准确分割难题。通过结合ViT与CNN的并行编码及变分融合方法，提升分割精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.12938v1](http://arxiv.org/pdf/2507.12938v1)**

> **作者:** Caixia Dong; Duwei Dai; Xinyi Han; Fan Liu; Xu Yang; Zongfang Li; Songhua Xu
>
> **摘要:** Accurate coronary artery segmentation is critical for computeraided diagnosis of coronary artery disease (CAD), yet it remains challenging due to the small size, complex morphology, and low contrast with surrounding tissues. To address these challenges, we propose a novel segmentation framework that leverages the power of vision foundation models (VFMs) through a parallel encoding architecture. Specifically, a vision transformer (ViT) encoder within the VFM captures global structural features, enhanced by the activation of the final two ViT blocks and the integration of an attention-guided enhancement (AGE) module, while a convolutional neural network (CNN) encoder extracts local details. These complementary features are adaptively fused using a cross-branch variational fusion (CVF) module, which models latent distributions and applies variational attention to assign modality-specific weights. Additionally, we introduce an evidential-learning uncertainty refinement (EUR) module, which quantifies uncertainty using evidence theory and refines uncertain regions by incorporating multi-scale feature aggregation and attention mechanisms, further enhancing segmentation accuracy. Extensive evaluations on one in-house and two public datasets demonstrate that the proposed framework significantly outperforms state-of-the-art methods, achieving superior performance in accurate coronary artery segmentation and showcasing strong generalization across multiple datasets. The code is available at https://github.com/d1c2x3/CAseg.
>
---
#### [new 107] Dual LiDAR-Based Traffic Movement Count Estimation at a Signalized Intersection: Deployment, Data Collection, and Preliminary Analysis
- **分类: eess.SY; cs.CV; cs.SY**

- **简介: 该论文属于交通流量计数任务，旨在解决传统方法在恶劣环境下精度不足的问题。通过部署双LiDAR系统，实现更准确的车辆分类与流向统计。**

- **链接: [http://arxiv.org/pdf/2507.13073v1](http://arxiv.org/pdf/2507.13073v1)**

> **作者:** Saswat Priyadarshi Nayak; Guoyuan Wu; Kanok Boriboonsomsin; Matthew Barth
>
> **备注:** 7 Pages, 8 Figures. This paper has been accepted for publication at the 2025 IEEE ITSC. Copyright IEEE
>
> **摘要:** Traffic Movement Count (TMC) at intersections is crucial for optimizing signal timings, assessing the performance of existing traffic control measures, and proposing efficient lane configurations to minimize delays, reduce congestion, and promote safety. Traditionally, methods such as manual counting, loop detectors, pneumatic road tubes, and camera-based recognition have been used for TMC estimation. Although generally reliable, camera-based TMC estimation is prone to inaccuracies under poor lighting conditions during harsh weather and nighttime. In contrast, Light Detection and Ranging (LiDAR) technology is gaining popularity in recent times due to reduced costs and its expanding use in 3D object detection, tracking, and related applications. This paper presents the authors' endeavor to develop, deploy and evaluate a dual-LiDAR system at an intersection in the city of Rialto, California, for TMC estimation. The 3D bounding box detections from the two LiDARs are used to classify vehicle counts based on traffic directions, vehicle movements, and vehicle classes. This work discusses the estimated TMC results and provides insights into the observed trends and irregularities. Potential improvements are also discussed that could enhance not only TMC estimation, but also trajectory forecasting and intent prediction at intersections.
>
---
## 更新

#### [replaced 001] Vidi: Large Multimodal Models for Video Understanding and Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15681v3](http://arxiv.org/pdf/2504.15681v3)**

> **作者:** Vidi Team; Celong Liu; Chia-Wen Kuo; Dawei Du; Fan Chen; Guang Chen; Jiamin Yuan; Lingxi Zhang; Lu Guo; Lusha Li; Longyin Wen; Qingyu Chen; Rachel Deng; Sijie Zhu; Stuart Siew; Tong Jin; Wei Lu; Wen Zhong; Xiaohui Shen; Xin Gu; Xing Mei; Xueqiong Qu; Zhenfang Chen
>
> **摘要:** Humans naturally share information with those they are connected to, and video has become one of the dominant mediums for communication and expression on the Internet. To support the creation of high-quality large-scale video content, a modern pipeline requires a comprehensive understanding of both the raw input materials (e.g., the unedited footage captured by cameras) and the editing components (e.g., visual effects). In video editing scenarios, models must process multiple modalities (e.g., vision, audio, text) with strong background knowledge and handle flexible input lengths (e.g., hour-long raw videos), which poses significant challenges for traditional models. In this report, we introduce Vidi, a family of Large Multimodal Models (LMMs) for a wide range of video understand editing scenarios. The first release focuses on temporal retrieval, i.e., identifying the time ranges within the input videos corresponding to a given text query, which plays a critical role in intelligent editing. The model is capable of processing hour-long videos with strong temporal understanding capability, e.g., retrieve time ranges for certain queries. To support a comprehensive evaluation in real-world scenarios, we also present the VUE-TR benchmark, which introduces five key advancements. 1) Video duration: significantly longer than videos of existing temporal retrival datasets, 2) Audio support: includes audio-based queries, 3) Query format: diverse query lengths/formats, 4) Annotation quality: ground-truth time ranges are manually annotated. 5) Evaluation metric: a refined IoU metric to support evaluation over multiple time ranges. Remarkably, Vidi significantly outperforms leading proprietary models, e.g., GPT-4o and Gemini, on the temporal retrieval task, indicating its superiority in video editing scenarios.
>
---
#### [replaced 002] Task-Specific Generative Dataset Distillation with Difficulty-Guided Sampling
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.03331v2](http://arxiv.org/pdf/2507.03331v2)**

> **作者:** Mingzhuo Li; Guang Li; Jiafeng Mao; Linfeng Ye; Takahiro Ogawa; Miki Haseyama
>
> **备注:** Accepted by The ICCV 2025 Workshop on Curated Data for Efficient Learning
>
> **摘要:** To alleviate the reliance of deep neural networks on large-scale datasets, dataset distillation aims to generate compact, high-quality synthetic datasets that can achieve comparable performance to the original dataset. The integration of generative models has significantly advanced this field. However, existing approaches primarily focus on aligning the distilled dataset with the original one, often overlooking task-specific information that can be critical for optimal downstream performance. In this paper, focusing on the downstream task of classification, we propose a task-specific sampling strategy for generative dataset distillation that incorporates the concept of difficulty to consider the requirements of the target task better. The final dataset is sampled from a larger image pool with a sampling distribution obtained by matching the difficulty distribution of the original dataset. A logarithmic transformation is applied as a pre-processing step to correct for distributional bias. The results of extensive experiments demonstrate the effectiveness of our method and suggest its potential for enhancing performance on other downstream tasks. The code is available at https://github.com/SumomoTaku/DiffGuideSamp.
>
---
#### [replaced 003] Global urban visual perception varies across demographics and personalities
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12758v3](http://arxiv.org/pdf/2505.12758v3)**

> **作者:** Matias Quintana; Youlong Gu; Xiucheng Liang; Yujun Hou; Koichi Ito; Yihan Zhu; Mahmoud Abdelrahman; Filip Biljecki
>
> **备注:** Under review
>
> **摘要:** Understanding people's preferences is crucial for urban planning, yet current approaches often combine responses from multi-cultural populations, obscuring demographic differences and risking amplifying biases. We conducted a large-scale urban visual perception survey of streetscapes worldwide using street view imagery, examining how demographics -- including gender, age, income, education, race and ethnicity, and, for the first time, personality traits -- shape perceptions among 1,000 participants with balanced demographics from five countries and 45 nationalities. This dataset, Street Perception Evaluation Considering Socioeconomics (SPECS), reveals demographic- and personality-based differences across six traditional indicators (safe, lively, wealthy, beautiful, boring, depressing) and four new ones (live nearby, walk, cycle, green). Location-based sentiments further shape these preferences. Machine learning models trained on existing global datasets tend to overestimate positive indicators and underestimate negative ones compared to human responses, underscoring the need for local context. Our study aspires to rectify the myopic treatment of street perception, which rarely considers demographics or personality traits.
>
---
#### [replaced 004] PhenoBench: A Comprehensive Benchmark for Cell Phenotyping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03532v3](http://arxiv.org/pdf/2507.03532v3)**

> **作者:** Nora Koreuber; Jannik Franzen; Fabian H. Reith; Claudia Winklmayr; Jerome Luescher; Elias Baumann; Christian M. Schuerch; Dagmar Kainmueller; Josef Lorenz Rumberger
>
> **备注:** accepted for presentation at MICCAI 2025
>
> **摘要:** Digital pathology has seen the advent of a wealth of foundational models (FM), yet to date their performance on cell phenotyping has not been benchmarked in a unified manner. We therefore propose PhenoBench: A comprehensive benchmark for cell phenotyping on Hematoxylin and Eosin (H&E) stained histopathology images. We provide both PhenoCell, a new H&E dataset featuring 14 granular cell types identified by using multiplexed imaging, and ready-to-use fine-tuning and benchmarking code that allows the systematic evaluation of multiple prominent pathology FMs in terms of dense cell phenotype predictions in different generalization scenarios. We perform extensive benchmarking of existing FMs, providing insights into their generalization behavior under technical vs. medical domain shifts. Furthermore, while FMs achieve macro F1 scores > 0.70 on previously established benchmarks such as Lizard and PanNuke, on PhenoCell, we observe scores as low as 0.20. This indicates a much more challenging task not captured by previous benchmarks, establishing PhenoCell as a prime asset for future benchmarking of FMs and supervised models alike. Code and data are available on GitHub.
>
---
#### [replaced 005] SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks
- **分类: cs.LG; astro-ph.GA; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.14048v2](http://arxiv.org/pdf/2501.14048v2)**

> **作者:** Sneh Pandya; Purvik Patel; Brian D. Nord; Mike Walmsley; Aleksandra Ćiprijanović
>
> **备注:** 25 pages, 5 figures, 4 tables. code available at: https://github.com/deepskies/SIDDA
>
> **摘要:** Modern neural networks (NNs) often do not generalize well in the presence of a "covariate shift"; that is, in situations where the training and test data distributions differ, but the conditional distribution of classification labels remains unchanged. In such cases, NN generalization can be reduced to a problem of learning more domain-invariant features. Domain adaptation (DA) methods include a range of techniques aimed at achieving this; however, these methods have struggled with the need for extensive hyperparameter tuning, which then incurs significant computational costs. In this work, we introduce SIDDA, an out-of-the-box DA training algorithm built upon the Sinkhorn divergence, that can achieve effective domain alignment with minimal hyperparameter tuning and computational overhead. We demonstrate the efficacy of our method on multiple simulated and real datasets of varying complexity, including simple shapes, handwritten digits, and real astronomical observations. SIDDA is compatible with a variety of NN architectures, and it works particularly well in improving classification accuracy and model calibration when paired with equivariant neural networks (ENNs). We find that SIDDA enhances the generalization capabilities of NNs, achieving up to a $\approx40\%$ improvement in classification accuracy on unlabeled target data. We also study the efficacy of DA on ENNs with respect to the varying group orders of the dihedral group $D_N$, and find that the model performance improves as the degree of equivariance increases. Finally, we find that SIDDA enhances model calibration on both source and target data--achieving over an order of magnitude improvement in the ECE and Brier score. SIDDA's versatility, combined with its automated approach to domain alignment, has the potential to advance multi-dataset studies by enabling the development of highly generalizable models.
>
---
#### [replaced 006] Token Communications: A Large Model-Driven Framework for Cross-modal Context-aware Semantic Communications
- **分类: cs.MM; cs.CV; cs.IT; eess.SP; math.IT**

- **链接: [http://arxiv.org/pdf/2502.12096v4](http://arxiv.org/pdf/2502.12096v4)**

> **作者:** Li Qiao; Mahdi Boloursaz Mashhadi; Zhen Gao; Rahim Tafazolli; Mehdi Bennis; Dusit Niyato
>
> **备注:** Accepted at IEEE Wireless Communications Magazine
>
> **摘要:** In this paper, we introduce token communications (TokCom), a large model-driven framework to leverage cross-modal context information in generative semantic communications (GenSC). TokCom is a new paradigm, motivated by the recent success of generative foundation models and multimodal large language models (GFM/MLLMs), where the communication units are tokens, enabling efficient transformer-based token processing at the transmitter and receiver. In this paper, we introduce the potential opportunities and challenges of leveraging context in GenSC, explore how to integrate GFM/MLLMs-based token processing into semantic communication systems to leverage cross-modal context effectively at affordable complexity, present the key principles for efficient TokCom at various layers in future wireless networks. In a typical image semantic communication setup, we demonstrate a significant improvement of the bandwidth efficiency, achieved by TokCom by leveraging the context information among tokens. Finally, the potential research directions are identified to facilitate adoption of TokCom in future wireless networks.
>
---
#### [replaced 007] Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18248v3](http://arxiv.org/pdf/2506.18248v3)**

> **作者:** Jongoh Jeong; Hunmin Yang; Jaeseok Jeong; Kuk-Jin Yoon
>
> **备注:** Preprint
>
> **摘要:** Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).
>
---
#### [replaced 008] Physical Annotation for Automated Optical Inspection: A Concept for In-Situ, Pointer-Based Training Data Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05026v2](http://arxiv.org/pdf/2506.05026v2)**

> **作者:** Oliver Krumpek; Oliver Heimann; Jörg Krüger
>
> **摘要:** This paper introduces a novel physical annotation system designed to generate training data for automated optical inspection. The system uses pointer-based in-situ interaction to transfer the valuable expertise of trained inspection personnel directly into a machine learning (ML) training pipeline. Unlike conventional screen-based annotation methods, our system captures physical trajectories and contours directly on the object, providing a more intuitive and efficient way to label data. The core technology uses calibrated, tracked pointers to accurately record user input and transform these spatial interactions into standardised annotation formats that are compatible with open-source annotation software. Additionally, a simple projector-based interface projects visual guidance onto the object to assist users during the annotation process, ensuring greater accuracy and consistency. The proposed concept bridges the gap between human expertise and automated data generation, enabling non-IT experts to contribute to the ML training pipeline and preventing the loss of valuable training samples. Preliminary evaluation results confirm the feasibility of capturing detailed annotation trajectories and demonstrate that integration with CVAT streamlines the workflow for subsequent ML tasks. This paper details the system architecture, calibration procedures and interface design, and discusses its potential contribution to future ML data generation for automated optical inspection.
>
---
#### [replaced 009] PhysX: Physical-Grounded 3D Asset Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.12465v2](http://arxiv.org/pdf/2507.12465v2)**

> **作者:** Ziang Cao; Zhaoxi Chen; Liang Pan; Ziwei Liu
>
> **备注:** Project page: https://physx-3d.github.io/
>
> **摘要:** 3D modeling is moving from virtual to physical. Existing 3D generation primarily emphasizes geometries and textures while neglecting physical-grounded modeling. Consequently, despite the rapid development of 3D generative models, the synthesized 3D assets often overlook rich and important physical properties, hampering their real-world application in physical domains like simulation and embodied AI. As an initial attempt to address this challenge, we propose \textbf{PhysX}, an end-to-end paradigm for physical-grounded 3D asset generation. 1) To bridge the critical gap in physics-annotated 3D datasets, we present PhysXNet - the first physics-grounded 3D dataset systematically annotated across five foundational dimensions: absolute scale, material, affordance, kinematics, and function description. In particular, we devise a scalable human-in-the-loop annotation pipeline based on vision-language models, which enables efficient creation of physics-first assets from raw 3D assets.2) Furthermore, we propose \textbf{PhysXGen}, a feed-forward framework for physics-grounded image-to-3D asset generation, injecting physical knowledge into the pre-trained 3D structural space. Specifically, PhysXGen employs a dual-branch architecture to explicitly model the latent correlations between 3D structures and physical properties, thereby producing 3D assets with plausible physical predictions while preserving the native geometry quality. Extensive experiments validate the superior performance and promising generalization capability of our framework. All the code, data, and models will be released to facilitate future research in generative physical AI.
>
---
#### [replaced 010] Intriguing Properties of Robust Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.04245v2](http://arxiv.org/pdf/2412.04245v2)**

> **作者:** Bernd Prach; Christoph H. Lampert
>
> **摘要:** Despite extensive research since the community learned about adversarial examples 10 years ago, we still do not know how to train high-accuracy classifiers that are guaranteed to be robust to small perturbations of their inputs. Previous works often argued that this might be because no classifier exists that is robust and accurate at the same time. However, in computer vision this assumption does not match reality where humans are usually accurate and robust on most tasks of interest. We offer an alternative explanation and show that in certain settings robust generalization is only possible with unrealistically large amounts of data. Specifically, we find a setting where a robust classifier exists, it is easy to learn an accurate classifier, yet it requires an exponential amount of data to learn a robust classifier. Based on this theoretical result, we evaluate the influence of the amount of training data on datasets such as CIFAR-10. Our findings indicate that the amount of training data is the main factor determining the robust performance. Furthermore we show that there are low magnitude directions in the data which are useful for non-robust generalization but are not available for robust classifiers. We provide code at https://github.com/berndprach/IntriguingProperties.
>
---
#### [replaced 011] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14953v2](http://arxiv.org/pdf/2503.14953v2)**

> **作者:** Yang Liu; Wentao Feng; Zhuoyao Liu; Shudong Huang; Jiancheng Lv
>
> **摘要:** Enabling Visual Semantic Models to effectively handle multi-view description matching has been a longstanding challenge. Existing methods typically learn a set of embeddings to find the optimal match for each view's text and compute similarity. However, the visual and text embeddings learned through these approaches have limited information capacity and are prone to interference from locally similar negative samples. To address this issue, we argue that the information capacity of embeddings is crucial and propose Dense-to-Sparse Feature Distilled Visual Semantic Embedding (D2S-VSE), which enhances the information capacity of sparse text by leveraging dense text distillation. Specifically, D2S-VSE is a two-stage framework. In the pre-training stage, we align images with dense text to enhance the information capacity of visual semantic embeddings. In the fine-tuning stage, we optimize two tasks simultaneously, distilling dense text embeddings to sparse text embeddings while aligning images and sparse texts, enhancing the information capacity of sparse text embeddings. Our proposed D2S-VSE model is extensively evaluated on the large-scale MS-COCO and Flickr30K datasets, demonstrating its superiority over recent state-of-the-art methods.
>
---
#### [replaced 012] An Event-based Algorithm for Simultaneous 6-DOF Camera Pose Tracking and Mapping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2301.00618v4](http://arxiv.org/pdf/2301.00618v4)**

> **作者:** Masoud Dayani Najafabadi; Mohammad Reza Ahmadzadeh
>
> **摘要:** Compared to regular cameras, Dynamic Vision Sensors or Event Cameras can output compact visual data based on a change in the intensity in each pixel location asynchronously. In this paper, we study the application of current image-based SLAM techniques to these novel sensors. To this end, the information in adaptively selected event windows is processed to form motion-compensated images. These images are then used to reconstruct the scene and estimate the 6-DOF pose of the camera. We also propose an inertial version of the event-only pipeline to assess its capabilities. We compare the results of different configurations of the proposed algorithm against the ground truth for sequences of two publicly available event datasets. We also compare the results of the proposed event-inertial pipeline with the state-of-the-art and show it can produce comparable or more accurate results provided the map estimate is reliable.
>
---
#### [replaced 013] Golden Noise for Diffusion Models: A Learning Framework
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.09502v5](http://arxiv.org/pdf/2411.09502v5)**

> **作者:** Zikai Zhou; Shitong Shao; Lichen Bai; Shufei Zhang; Zhiqiang Xu; Bo Han; Zeke Xie
>
> **摘要:** Text-to-image diffusion model is a popular paradigm that synthesizes personalized images by providing a text prompt and a random Gaussian noise. While people observe that some noises are ``golden noises'' that can achieve better text-image alignment and higher human preference than others, we still lack a machine learning framework to obtain those golden noises. To learn golden noises for diffusion sampling, we mainly make three contributions in this paper. First, we identify a new concept termed the \textit{noise prompt}, which aims at turning a random Gaussian noise into a golden noise by adding a small desirable perturbation derived from the text prompt. Following the concept, we first formulate the \textit{noise prompt learning} framework that systematically learns ``prompted'' golden noise associated with a text prompt for diffusion models. Second, we design a noise prompt data collection pipeline and collect a large-scale \textit{noise prompt dataset}~(NPD) that contains 100k pairs of random noises and golden noises with the associated text prompts. With the prepared NPD as the training dataset, we trained a small \textit{noise prompt network}~(NPNet) that can directly learn to transform a random noise into a golden noise. The learned golden noise perturbation can be considered as a kind of prompt for noise, as it is rich in semantic information and tailored to the given text prompt. Third, our extensive experiments demonstrate the impressive effectiveness and generalization of NPNet on improving the quality of synthesized images across various diffusion models, including SDXL, DreamShaper-xl-v2-turbo, and Hunyuan-DiT. Moreover, NPNet is a small and efficient controller that acts as a plug-and-play module with very limited additional inference and computational costs, as it just provides a golden noise instead of a random noise without accessing the original pipeline.
>
---
#### [replaced 014] EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12440v2](http://arxiv.org/pdf/2507.12440v2)**

> **作者:** Ruihan Yang; Qinxi Yu; Yecheng Wu; Rui Yan; Borui Li; An-Chieh Cheng; Xueyan Zou; Yunhao Fang; Hongxu Yin; Sifei Liu; Song Han; Yao Lu; Xiaolong Wang
>
> **备注:** More videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
> **摘要:** Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Ego Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Ego Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
---
#### [replaced 015] USIS16K: High-Quality Dataset for Underwater Salient Instance Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19472v2](http://arxiv.org/pdf/2506.19472v2)**

> **作者:** Lin Hong; Xin Wang; Yihao Li; Xia Wang
>
> **备注:** 8 pages 10 figures
>
> **摘要:** Inspired by the biological visual system that selectively allocates attention to efficiently identify salient objects or regions, underwater salient instance segmentation (USIS) aims to jointly address the problems of where to look (saliency prediction) and what is there (instance segmentation) in underwater scenarios. However, USIS remains an underexplored challenge due to the inaccessibility and dynamic nature of underwater environments, as well as the scarcity of large-scale, high-quality annotated datasets. In this paper, we introduce USIS16K, a large-scale dataset comprising 16,151 high-resolution underwater images collected from diverse environmental settings and covering 158 categories of underwater objects. Each image is annotated with high-quality instance-level salient object masks, representing a significant advance in terms of diversity, complexity, and scalability. Furthermore, we provide benchmark evaluations on underwater object detection and USIS tasks using USIS16K. To facilitate future research in this domain, the dataset and benchmark models are publicly available.
>
---
#### [replaced 016] BPD-Neo: An MRI Dataset for Lung-Trachea Segmentation with Clinical Data for Neonatal Bronchopulmonary Dysplasia
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23305v2](http://arxiv.org/pdf/2506.23305v2)**

> **作者:** Rachit Saluja; Arzu Kovanlikaya; Candace Chien; Lauren Kathryn Blatt; Jeffrey M. Perlman; Stefan Worgall; Mert R. Sabuncu; Jonathan P. Dyke
>
> **备注:** Adding link to Zenodo repo for dataset
>
> **摘要:** Bronchopulmonary dysplasia (BPD) is a common complication among preterm neonates, with portable X-ray imaging serving as the standard diagnostic modality in neonatal intensive care units (NICUs). However, lung magnetic resonance imaging (MRI) offers a non-invasive alternative that avoids sedation and radiation while providing detailed insights into the underlying mechanisms of BPD. Leveraging high-resolution 3D MRI data, advanced image processing and semantic segmentation algorithms can be developed to assist clinicians in identifying the etiology of BPD. In this dataset, we present MRI scans paired with corresponding semantic segmentations of the lungs and trachea for 40 neonates, the majority of whom are diagnosed with BPD. The imaging data consist of free-breathing 3D stack-of-stars radial gradient echo acquisitions, known as the StarVIBE series. Additionally, we provide comprehensive clinical data and baseline segmentation models, validated against clinical assessments, to support further research and development in neonatal lung imaging.
>
---
#### [replaced 017] LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10200v4](http://arxiv.org/pdf/2503.10200v4)**

> **作者:** Boyu Chen; Zhengrong Yue; Siran Chen; Zikang Wang; Yang Liu; Peng Li; Yali Wang
>
> **备注:** accepted in ICCV 2025
>
> **摘要:** Existing MLLMs encounter significant challenges in modeling the temporal context within long videos. Currently, mainstream Agent-based methods use external tools to assist a single MLLM in answering long video questions. Despite such tool-based support, a solitary MLLM still offers only a partial understanding of long videos, resulting in limited performance. In order to better address long video tasks, we introduce LVAgent, the first framework enabling multi-round dynamic collaboration of MLLM agents in long video understanding. Our method consists of four key steps: 1) Selection: We pre-select appropriate agents from the model library to form optimal agent teams based on different tasks. 2) Perception: We design an effective retrieval scheme for long videos to improve the coverage of critical temporal segments while maintaining computational efficiency. 3) Action: Agents answer long video questions and exchange reasons. 4) Reflection: We evaluate each agent's performance in each round of discussion and optimize the agent team for dynamic collaboration. The agents iteratively refine their answers by multi-round dynamical collaboration of MLLM agents. LVAgent is the first agent system method that outperforms all closed-source models (like GPT-4o) and open-source models (like InternVL-2.5 and Qwen2-VL) in the long video understanding tasks. Our LVAgent achieves an accuracy of 80\% on four mainstream long video understanding tasks. Notably, LVAgent improves accuracy by 13.3\% on LongVideoBench. Code is available at https://github.com/64327069/LVAgent.
>
---
#### [replaced 018] Model-Agnostic, Temperature-Informed Sampling Enhances Cross-Year Crop Mapping with Deep Learning
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.12885v3](http://arxiv.org/pdf/2506.12885v3)**

> **作者:** Mehmet Ozgur Turkoglu; Selene Ledain; Helge Aasen
>
> **备注:** under review
>
> **摘要:** Crop type classification using optical satellite time series remains limited in its ability to generalize across seasons, particularly when crop phenology shifts due to inter-annual weather variability. This hampers real-world applicability in scenarios where current-year labels are unavailable. In addition, uncertainty quantification is often overlooked, which reduces the reliability of such approaches for operational crop monitoring. Inspired by ecophysiological principles of plant growth, we propose a simple, model-agnostic Thermal-Time-based Temporal Sampling (T3S) method that replaces calendar time with thermal time. By subsampling time series in this biologically meaningful way, our method highlights key periods within the growing season while reducing temporal redundancy and noise. We evaluate the T3S on a multi-year Sentinel-2 dataset covering the entirety of Switzerland, which allows us to assess all applied methods on unseen years. Compared to state-of-the-art baselines, our approach yields substantial improvements in classification accuracy and, critically, provides well-calibrated uncertainty estimates. Moreover, the T3S method excels in low-data regimes and enables significantly more accurate early-season classification. With just 10% of the training labels, it outperforms the current baseline in both accuracy and uncertainty calibration, and by the end of June, it achieves a performance similar to the full-season baseline model.
>
---
#### [replaced 019] A Controllable Appearance Representation for Flexible Transfer and Editing
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15028v2](http://arxiv.org/pdf/2504.15028v2)**

> **作者:** Santiago Jimenez-Navarro; Julia Guerrero-Viu; Belen Masia
>
> **备注:** EGSR 2025 - Symposium Track
>
> **摘要:** We present a method that computes an interpretable representation of material appearance within a highly compact, disentangled latent space. This representation is learned in a self-supervised fashion using an adapted FactorVAE. We train our model with a carefully designed unlabeled dataset, avoiding possible biases induced by human-generated labels. Our model demonstrates strong disentanglement and interpretability by effectively encoding material appearance and illumination, despite the absence of explicit supervision. Then, we use our representation as guidance for training a lightweight IP-Adapter to condition a diffusion pipeline that transfers the appearance of one or more images onto a target geometry, and allows the user to further edit the resulting appearance. Our approach offers fine-grained control over the generated results: thanks to the well-structured compact latent space, users can intuitively manipulate attributes such as hue or glossiness in image space to achieve the desired final appearance.
>
---
#### [replaced 020] SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.14987v3](http://arxiv.org/pdf/2410.14987v3)**

> **作者:** Zhewei Dai; Shilei Zeng; Haotian Liu; Xurui Li; Feng Xue; Yu Zhou
>
> **备注:** Accepted at ICCV2025
>
> **摘要:** We introduce SeaS, a unified industrial generative model for automatically creating diverse anomalies, authentic normal products, and precise anomaly masks. While extensive research exists, most efforts either focus on specific tasks, i.e., anomalies or normal products only, or require separate models for each anomaly type. Consequently, prior methods either offer limited generative capability or depend on a vast array of anomaly-specific models. We demonstrate that U-Net's differentiated learning ability captures the distinct visual traits of slightly-varied normal products and diverse anomalies, enabling us to construct a unified model for all tasks. Specifically, we first introduce an Unbalanced Abnormal (UA) Text Prompt, comprising one normal token and multiple anomaly tokens. More importantly, our Decoupled Anomaly Alignment (DA) loss decouples anomaly attributes and binds them to distinct anomaly tokens of UA, enabling SeaS to create unseen anomalies by recombining these attributes. Furthermore, our Normal-image Alignment (NA) loss aligns the normal token to normal patterns, making generated normal products globally consistent and locally varied. Finally, SeaS produces accurate anomaly masks by fusing discriminative U-Net features with high-resolution VAE features. SeaS sets a new benchmark for industrial generation, significantly enhancing downstream applications, with average improvements of $+8.66\%$ pixel-level AP for synthesis-based AD approaches, $+1.10\%$ image-level AP for unsupervised AD methods, and $+12.79\%$ IoU for supervised segmentation models. Code is available at \href{https://github.com/HUST-SLOW/SeaS}{https://github.com/HUST-SLOW/SeaS}.
>
---
#### [replaced 021] Uncertainty quantification for White Matter Hyperintensity segmentation detects silent failures and improves automated Fazekas quantification
- **分类: eess.IV; cs.CV; cs.LG; I.4.10; I.4.6; I.2.10; I.2.6; J.3; G.3**

- **链接: [http://arxiv.org/pdf/2411.17571v2](http://arxiv.org/pdf/2411.17571v2)**

> **作者:** Ben Philps; Maria del C. Valdes Hernandez; Chen Qin; Una Clancy; Eleni Sakka; Susana Munoz Maniega; Mark E. Bastin; Angela C. C. Jochems; Joanna M. Wardlaw; Miguel O. Bernabeu; Alzheimers Disease Neuroimaging Initiative
>
> **备注:** 34 pages (or 19 not including appendix) 28 figures (or 10 not including appendix)
>
> **摘要:** White Matter Hyperintensities (WMH) are key neuroradiological markers of small vessel disease present in brain MRI. Assessment of WMH is important in research and clinics. However, WMH are challenging to segment due to their high variability in shape, location, size, poorly defined borders, and similar intensity profile to other pathologies (e.g stroke lesions) and artefacts (e.g head motion). In this work, we assess the utility and semantic properties of the most effective techniques for uncertainty quantification (UQ) in segmentation for the WMH segmentation task across multiple test-time data distributions. We find UQ techniques reduce 'silent failure' by identifying in UQ maps small WMH clusters in the deep white matter that are unsegmented by the model. A combination of Stochastic Segmentation Networks with Deep Ensembles also yields the highest Dice and lowest Absolute Volume Difference % (AVD) score and can highlight areas where there is ambiguity between WMH and stroke lesions. We further demonstrate the downstream utility of UQ, proposing a novel method for classification of the clinical Fazekas score using spatial features extracted from voxelwise WMH probability and UQ maps. We show that incorporating WMH uncertainty information improves Fazekas classification performance and calibration. Our model with (UQ and spatial WMH features)/(spatial WMH features)/(WMH volume only) achieves a balanced accuracy score of 0.74/0.67/0.62, and root brier score of 0.65/0.72/0.74 in the Deep WMH and balanced accuracy of 0.74/0.73/0.71 and root brier score of 0.64/0.66/0.68 in the Periventricular region. We further demonstrate that stochastic UQ techniques with high sample diversity can improve the detection of poor quality segmentations.
>
---
#### [replaced 022] Salvaging the Overlooked: Leveraging Class-Aware Contrastive Learning for Multi-Class Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.04769v2](http://arxiv.org/pdf/2412.04769v2)**

> **作者:** Lei Fan; Junjie Huang; Donglin Di; Anyang Su; Tianyou Song; Maurice Pagnucco; Yang Song
>
> **备注:** Accepted by ICCV2025, https://lgc-ad.github.io/
>
> **摘要:** For anomaly detection (AD), early approaches often train separate models for individual classes, yielding high performance but posing challenges in scalability and resource management. Recent efforts have shifted toward training a single model capable of handling multiple classes. However, directly extending early AD methods to multi-class settings often results in degraded performance. In this paper, we investigate this performance degradation observed in reconstruction-based methods, identifying the key issue: inter-class confusion. This confusion emerges when a model trained in multi-class scenarios incorrectly reconstructs samples from one class as those of another, thereby exacerbating reconstruction errors. To this end, we propose a simple yet effective modification, called class-aware contrastive learning (CCL). By explicitly leveraging raw object category information (\eg carpet or wood) as supervised signals, we introduce local CL to refine multiscale dense features, and global CL to obtain more compact feature representations of normal patterns, thereby effectively adapting the models to multi-class settings. Experiments across five datasets validate the effectiveness of our approach, demonstrating significant improvements and superior performance compared to state-of-the-art methods. Notably, ablation studies indicate that pseudo-class labels can achieve comparable performance.
>
---
#### [replaced 023] MMOne: Representing Multiple Modalities in One Scene
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11129v2](http://arxiv.org/pdf/2507.11129v2)**

> **作者:** Zhifeng Gu; Bing Wang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Humans perceive the world through multimodal cues to understand and interact with the environment. Learning a scene representation for multiple modalities enhances comprehension of the physical world. However, modality conflicts, arising from inherent distinctions among different modalities, present two critical challenges: property disparity and granularity disparity. To address these challenges, we propose a general framework, MMOne, to represent multiple modalities in one scene, which can be readily extended to additional modalities. Specifically, a modality modeling module with a novel modality indicator is proposed to capture the unique properties of each modality. Additionally, we design a multimodal decomposition mechanism to separate multi-modal Gaussians into single-modal Gaussians based on modality differences. We address the essential distinctions among modalities by disentangling multimodal information into shared and modality-specific components, resulting in a more compact and efficient multimodal scene representation. Extensive experiments demonstrate that our method consistently enhances the representation capability for each modality and is scalable to additional modalities. The code is available at https://github.com/Neal2020GitHub/MMOne.
>
---
#### [replaced 024] Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20755v2](http://arxiv.org/pdf/2505.20755v2)**

> **作者:** Yifei Wang; Weimin Bai; Colin Zhang; Debing Zhang; Weijian Luo; He Sun
>
> **摘要:** In this paper, we unify more than 10 existing one-step diffusion distillation approaches, such as Diff-Instruct, DMD, SIM, SiD, $f$-distill, etc, inside a theory-driven framework which we name the \textbf{\emph{Uni-Instruct}}. Uni-Instruct is motivated by our proposed diffusion expansion theory of the $f$-divergence family. Then we introduce key theories that overcome the intractability issue of the original expanded $f$-divergence, resulting in an equivalent yet tractable loss that effectively trains one-step diffusion models by minimizing the expanded $f$-divergence family. The novel unification introduced by Uni-Instruct not only offers new theoretical contributions that help understand existing approaches from a high-level perspective but also leads to state-of-the-art one-step diffusion generation performances. On the CIFAR10 generation benchmark, Uni-Instruct achieves record-breaking Frechet Inception Distance (FID) values of \textbf{\emph{1.46}} for unconditional generation and \textbf{\emph{1.38}} for conditional generation. On the ImageNet-$64\times 64$ generation benchmark, Uni-Instruct achieves a new SoTA one-step generation FID of \textbf{\emph{1.02}}, which outperforms its 79-step teacher diffusion with a significant improvement margin of 1.33 (1.02 vs 2.35). We also apply Uni-Instruct on broader tasks like text-to-3D generation. For text-to-3D generation, Uni-Instruct gives decent results, which slightly outperforms previous methods, such as SDS and VSD, in terms of both generation quality and diversity. Both the solid theoretical and empirical contributions of Uni-Instruct will potentially help future studies on one-step diffusion distillation and knowledge transferring of diffusion models.
>
---
#### [replaced 025] Dynamic EventNeRF: Reconstructing General Dynamic Scenes from Multi-view RGB and Event Streams
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.06770v4](http://arxiv.org/pdf/2412.06770v4)**

> **作者:** Viktor Rudnev; Gereon Fox; Mohamed Elgharib; Christian Theobalt; Vladislav Golyanik
>
> **备注:** 17 pages, 13 figures, 7 tables; CVPRW 2025
>
> **摘要:** Volumetric reconstruction of dynamic scenes is an important problem in computer vision. It is especially challenging in poor lighting and with fast motion. This is partly due to limitations of RGB cameras: To capture frames under low lighting, the exposure time needs to be increased, which leads to more motion blur. In contrast, event cameras, which record changes in pixel brightness asynchronously, are much less dependent on lighting, making them more suitable for recording fast motion. We hence propose the first method to spatiotemporally reconstruct a scene from sparse multi-view event streams and sparse RGB frames. We train a sequence of cross-faded time-conditioned NeRF models, one per short recording segment. The individual segments are supervised with a set of event- and RGB-based losses and sparse-view regularisation. We assemble a real-world multi-view camera rig with six static event cameras around the object and record a benchmark multi-view event stream dataset of challenging motions. Our work outperforms RGB-based baselines, producing state-of-the-art results, and opens up the topic of multi-view event-based reconstruction as a new path for fast scene capture beyond RGB cameras. The code and the data are released at https://4dqv.mpi-inf.mpg.de/DynEventNeRF/
>
---
#### [replaced 026] Escaping Plato's Cave: JAM for Aligning Independently Trained Vision and Language Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01201v4](http://arxiv.org/pdf/2507.01201v4)**

> **作者:** Lauren Hyoseo Yoon; Yisong Yue; Been Kim
>
> **摘要:** Independently trained vision and language models inhabit disjoint representational spaces, shaped by their respective modalities, objectives, and architectures. Yet an emerging hypothesis - the Platonic Representation Hypothesis - suggests that such models may nonetheless converge toward a shared statistical model of reality. This compatibility, if it exists, raises a fundamental question: can we move beyond post-hoc statistical detection of alignment and explicitly optimize for it between such disjoint representations? We cast this Platonic alignment problem as a multi-objective optimization task - preserve each modality's native structure while aligning for mutual coherence. We introduce the Joint Autoencoder Modulator (JAM) framework that jointly trains modality-specific autoencoders on the latent representations of pre-trained single modality models, encouraging alignment through both reconstruction and cross-modal objectives. By analogy, this framework serves as a method to escape Plato's Cave, enabling the emergence of shared structure from disjoint inputs. We evaluate this framework across three critical design axes: (i) the alignment objective - comparing contrastive loss (Con), its hard-negative variant (NegCon), and our Spread loss, (ii) the layer depth at which alignment is most effective, and (iii) the impact of foundation model scale on representational convergence. Our findings show that our lightweight Pareto-efficient framework reliably induces alignment, even across frozen, independently trained representations, offering both theoretical insight and practical pathways for transforming generalist unimodal foundations into specialist multimodal models.
>
---
#### [replaced 027] KeyRe-ID: Keypoint-Guided Person Re-Identification using Part-Aware Representation in Videos
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.07393v3](http://arxiv.org/pdf/2507.07393v3)**

> **作者:** Jinseong Kim; Jeonghoon Song; Gyeongseon Baek; Byeongjoon Noh
>
> **备注:** 10 pages, 2 figures,
>
> **摘要:** We propose \textbf{KeyRe-ID}, a keypoint-guided video-based person re-identification framework consisting of global and local branches that leverage human keypoints for enhanced spatiotemporal representation learning. The global branch captures holistic identity semantics through Transformer-based temporal aggregation, while the local branch dynamically segments body regions based on keypoints to generate fine-grained, part-aware features. Extensive experiments on MARS and iLIDS-VID benchmarks demonstrate state-of-the-art performance, achieving 91.73\% mAP and 97.32\% Rank-1 accuracy on MARS, and 96.00\% Rank-1 and 100.0\% Rank-5 accuracy on iLIDS-VID. The code for this work will be publicly available on GitHub upon publication.
>
---
#### [replaced 028] Deep Blur Multi-Model (DeepBlurMM) -- a strategy to mitigate the impact of image blur on deep learning model performance in histopathology image analysis
- **分类: eess.IV; cs.CV; I.4; J.3**

- **链接: [http://arxiv.org/pdf/2405.09298v4](http://arxiv.org/pdf/2405.09298v4)**

> **作者:** Yujie Xiang; Bojing Liu; Mattias Rantalainen
>
> **摘要:** AI-based models for histopathology whole slide image (WSI) analysis are increasingly common, but unsharp or blurred areas within WSI can significantly reduce prediction performance. In this study, we investigated the effect of image blur on deep learning models and introduced a mixture of experts (MoE) strategy that combines predictions from multiple expert models trained on data with varying blur levels. Using H&E-stained WSIs from 2,093 breast cancer patients, we benchmarked performance on grade classification and IHC biomarker prediction with both CNN- (CNN_CLAM and MoE-CNN_CLAM) and Vision Transformer-based (UNI_CLAM and MoE-UNI_CLAM) models. Our results show that baseline models' performance consistently decreased with increasing blur, but expert models trained on blurred tiles and especially our proposed MoE approach substantially improved performance, and outperformed baseline models in a range of simulated scenarios. MoE-CNN_CLAM outperformed the baseline CNN_CLAM under moderate (AUC: 0.868 vs. 0.702) and mixed blur conditions (AUC: 0.890 vs. 0.875). MoE-UNI_CLAM outperformed the baseline UNI_CLAM model in both moderate (AUC: 0.950 vs. 0.928) and mixed blur conditions (AUC: 0.944 vs. 0.931). This MoE method has the potential to enhance the reliability of AI-based pathology models under variable image quality, supporting broader application in both research and clinical settings.
>
---
#### [replaced 029] Learning Lens Blur Fields
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2310.11535v2](http://arxiv.org/pdf/2310.11535v2)**

> **作者:** Esther Y. H. Lin; Zhecheng Wang; Rebecca Lin; Daniel Miau; Florian Kainz; Jiawen Chen; Xuaner Cecilia Zhang; David B. Lindell; Kiriakos N. Kutulakos
>
> **摘要:** Optical blur is an inherent property of any lens system and is challenging to model in modern cameras because of their complex optical elements. To tackle this challenge, we introduce a high-dimensional neural representation of blur$-$$\textit{the lens blur field}$$-$and a practical method for acquiring it. The lens blur field is a multilayer perceptron (MLP) designed to (1) accurately capture variations of the lens 2D point spread function over image plane location, focus setting and, optionally, depth and (2) represent these variations parametrically as a single, sensor-specific function. The representation models the combined effects of defocus, diffraction, aberration, and accounts for sensor features such as pixel color filters and pixel-specific micro-lenses. To learn the real-world blur field of a given device, we formulate a generalized non-blind deconvolution problem that directly optimizes the MLP weights using a small set of focal stacks as the only input. We also provide a first-of-its-kind dataset of 5D blur fields$-$for smartphone cameras, camera bodies equipped with a variety of lenses, etc. Lastly, we show that acquired 5D blur fields are expressive and accurate enough to reveal, for the first time, differences in optical behavior of smartphone devices of the same make and model. Code and data can be found at blur-fields.github.io.
>
---
#### [replaced 030] ProDisc-VAD: An Efficient System for Weakly-Supervised Anomaly Detection in Video Surveillance Applications
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02179v3](http://arxiv.org/pdf/2505.02179v3)**

> **作者:** Tao Zhu; Qi Yu; Xinru Dong; Shiyu Li; Yue Liu; Jinlong Jiang; Lei Shu
>
> **备注:** arXiv admin comment: This version has been removed by arXiv administrators as the submitter did not have the rights to agree to the license at the time of submission
>
> **摘要:** Weakly-supervised video anomaly detection (WS-VAD) using Multiple Instance Learning (MIL) suffers from label ambiguity, hindering discriminative feature learning. We propose ProDisc-VAD, an efficient framework tackling this via two synergistic components. The Prototype Interaction Layer (PIL) provides controlled normality modeling using a small set of learnable prototypes, establishing a robust baseline without being overwhelmed by dominant normal data. The Pseudo-Instance Discriminative Enhancement (PIDE) loss boosts separability by applying targeted contrastive learning exclusively to the most reliable extreme-scoring instances (highest/lowest scores). ProDisc-VAD achieves strong AUCs (97.98% ShanghaiTech, 87.12% UCF-Crime) using only 0.4M parameters, over 800x fewer than recent ViT-based methods like VadCLIP. Code is available at https://github.com/modadundun/ProDisc-VAD.
>
---
#### [replaced 031] Real-Time Inverse Kinematics for Generating Multi-Constrained Movements of Virtual Human Characters
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.00792v2](http://arxiv.org/pdf/2507.00792v2)**

> **作者:** Hendric Voss; Stefan Kopp
>
> **摘要:** Generating accurate and realistic virtual human movements in real-time is of high importance for a variety of applications in computer graphics, interactive virtual environments, robotics, and biomechanics. This paper introduces a novel real-time inverse kinematics (IK) solver specifically designed for realistic human-like movement generation. Leveraging the automatic differentiation and just-in-time compilation of TensorFlow, the proposed solver efficiently handles complex articulated human skeletons with high degrees of freedom. By treating forward and inverse kinematics as differentiable operations, our method effectively addresses common challenges such as error accumulation and complicated joint limits in multi-constrained problems, which are critical for realistic human motion modeling. We demonstrate the solver's effectiveness on the SMPLX human skeleton model, evaluating its performance against widely used iterative-based IK algorithms, like Cyclic Coordinate Descent (CCD), FABRIK, and the nonlinear optimization algorithm IPOPT. Our experiments cover both simple end-effector tasks and sophisticated, multi-constrained problems with realistic joint limits. Results indicate that our IK solver achieves real-time performance, exhibiting rapid convergence, minimal computational overhead per iteration, and improved success rates compared to existing methods. The project code is available at https://github.com/hvoss-techfak/TF-JAX-IK
>
---
#### [replaced 032] Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.19697v3](http://arxiv.org/pdf/2502.19697v3)**

> **作者:** Yuan Bian; Min Liu; Yunqi Yi; Xueping Wang; Yaonan Wang
>
> **摘要:** Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.
>
---
#### [replaced 033] Comparative Evaluation of Radiomics and Deep Learning Models for Disease Detection in Chest Radiography
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.12249v2](http://arxiv.org/pdf/2504.12249v2)**

> **作者:** Zhijin He; Alan B. McMillan
>
> **备注:** revised abstract; added statistical analysis; one figure removed, three tables added; clarification of dataset usage, experimental design, and model training strategy; revised methods with details; revised discussion; defined all abbreviations; correction of typographical and numerical inconsistencies; overall language review
>
> **摘要:** The application of artificial intelligence (AI) in medical imaging has revolutionized diagnostic practices, enabling advanced analysis and interpretation of radiological data. This study presents a comprehensive evaluation of radiomics-based and deep learning-based approaches for disease detection in chest radiography, focusing on COVID-19, lung opacity, and viral pneumonia. While deep learning models, particularly convolutional neural networks and vision transformers, learn directly from image data, radiomics-based models extract handcrafted features, offering potential advantages in data-limited scenarios. We systematically compared the diagnostic performance of various AI models, including Decision Trees, Gradient Boosting, Random Forests, Support Vector Machines, and Multi-Layer Perceptrons for radiomics, against state-of-the-art deep learning models such as InceptionV3, EfficientNetL, and ConvNeXtXLarge. Performance was evaluated across multiple sample sizes. At 24 samples, EfficientNetL achieved an AUC of 0.839, outperforming SVM with an AUC of 0.762. At 4000 samples, InceptionV3 achieved the highest AUC of 0.996, compared to 0.885 for Random Forest. A Scheirer-Ray-Hare test confirmed significant main and interaction effects of model type and sample size on all metrics. Post hoc Mann-Whitney U tests with Bonferroni correction further revealed consistent performance advantages for deep learning models across most conditions. These findings provide statistically validated, data-driven recommendations for model selection in diagnostic AI. Deep learning models demonstrated higher performance and better scalability with increasing data availability, while radiomics-based models may remain useful in low-data contexts. This study addresses a critical gap in AI-based diagnostic research by offering practical guidance for deploying AI models across diverse clinical environments.
>
---
#### [replaced 034] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.04106v3](http://arxiv.org/pdf/2412.04106v3)**

> **作者:** Haoning Wu; Ziheng Zhao; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **备注:** Accepted by ICCV 2025; Project Page: https://haoningwu3639.github.io/MRGen/
>
> **摘要:** Training medical image segmentation models for rare yet clinically important imaging modalities is challenging due to the scarcity of annotated data, and manual mask annotations can be costly and labor-intensive to acquire. This paper investigates leveraging generative models to synthesize data, for training segmentation models for underrepresented modalities, particularly on annotation-scarce MRI. Concretely, our contributions are threefold: (i) we introduce MRGen-DB, a large-scale radiology image-text dataset comprising extensive samples with rich metadata, including modality labels, attributes, regions, and organs information, with a subset featuring pixel-wise mask annotations; (ii) we present MRGen, a diffusion-based data engine for controllable medical image synthesis, conditioned on text prompts and segmentation masks. MRGen can generate realistic images for diverse MRI modalities lacking mask annotations, facilitating segmentation training in low-source domains; (iii) extensive experiments across multiple modalities demonstrate that MRGen significantly improves segmentation performance on unannotated modalities by providing high-quality synthetic data. We believe that our method bridges a critical gap in medical image analysis, extending segmentation capabilities to scenarios that are challenging to acquire manual annotations. The codes, models, and data will be publicly available at https://haoningwu3639.github.io/MRGen/
>
---
#### [replaced 035] Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12318v2](http://arxiv.org/pdf/2507.12318v2)**

> **作者:** Samuel Lavoie; Michael Noukhovitch; Aaron Courville
>
> **备注:** In submission, 22 pages, 7 tables, 12 figures
>
> **摘要:** We argue that diffusion models' success in modeling complex distributions is, for the most part, coming from their input conditioning. This paper investigates the representation used to condition diffusion models from the perspective that ideal representations should improve sample fidelity, be easy to generate, and be compositional to allow out-of-training samples generation. We introduce Discrete Latent Code (DLC), an image representation derived from Simplicial Embeddings trained with a self-supervised learning objective. DLCs are sequences of discrete tokens, as opposed to the standard continuous image embeddings. They are easy to generate and their compositionality enables sampling of novel images beyond the training distribution. Diffusion models trained with DLCs have improved generation fidelity, establishing a new state-of-the-art for unconditional image generation on ImageNet. Additionally, we show that composing DLCs allows the image generator to produce out-of-distribution samples that coherently combine the semantics of images in diverse ways. Finally, we showcase how DLCs can enable text-to-image generation by leveraging large-scale pretrained language models. We efficiently finetune a text diffusion language model to generate DLCs that produce novel samples outside of the image generator training distribution.
>
---
#### [replaced 036] RetinaLogos: Fine-Grained Synthesis of High-Resolution Retinal Images Through Captions
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12887v3](http://arxiv.org/pdf/2505.12887v3)**

> **作者:** Junzhi Ning; Cheng Tang; Kaijing Zhou; Diping Song; Lihao Liu; Ming Hu; Wei Li; Huihui Xu; Yanzhou Su; Tianbin Li; Jiyao Liu; Jin Ye; Sheng Zhang; Yuanfeng Ji; Junjun He
>
> **摘要:** The scarcity of high-quality, labelled retinal imaging data, which presents a significant challenge in the development of machine learning models for ophthalmology, hinders progress in the field. Existing methods for synthesising Colour Fundus Photographs (CFPs) largely rely on predefined disease labels, which restricts their ability to generate images that reflect fine-grained anatomical variations, subtle disease stages, and diverse pathological features beyond coarse class categories. To overcome these challenges, we first introduce an innovative pipeline that creates a large-scale, captioned retinal dataset comprising 1.4 million entries, called RetinaLogos-1400k. Specifically, RetinaLogos-1400k uses the visual language model(VLM) to describe retinal conditions and key structures, such as optic disc configuration, vascular distribution, nerve fibre layers, and pathological features. Building on this dataset, we employ a novel three-step training framework, RetinaLogos, which enables fine-grained semantic control over retinal images and accurately captures different stages of disease progression, subtle anatomical variations, and specific lesion types. Through extensive experiments, our method demonstrates superior performance across multiple datasets, with 62.07% of text-driven synthetic CFPs indistinguishable from real ones by ophthalmologists. Moreover, the synthetic data improves accuracy by 5%-10% in diabetic retinopathy grading and glaucoma detection. Codes are available at https://github.com/uni-medical/retina-text2cfp.
>
---
#### [replaced 037] Cascaded Multi-Scale Attention for Enhanced Multi-Scale Feature Extraction and Interaction with Low-Resolution Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02197v2](http://arxiv.org/pdf/2412.02197v2)**

> **作者:** Xiangyong Lu; Masanori Suganuma; Takayuki Okatani
>
> **备注:** 9 pages, 4 figures, 5 tables
>
> **摘要:** In real-world applications of image recognition tasks, such as human pose estimation, cameras often capture objects, like human bodies, at low resolutions. This scenario poses a challenge in extracting and leveraging multi-scale features, which is often essential for precise inference. To address this challenge, we propose a new attention mechanism, named cascaded multi-scale attention (CMSA), tailored for use in CNN-ViT hybrid architectures, to handle low-resolution inputs effectively. The design of CMSA enables the extraction and seamless integration of features across various scales without necessitating the downsampling of the input image or feature maps. This is achieved through a novel combination of grouped multi-head self-attention mechanisms with window-based local attention and cascaded fusion of multi-scale features over different scales. This architecture allows for the effective handling of features across different scales, enhancing the model's ability to perform tasks such as human pose estimation, head pose estimation, and more with low-resolution images. Our experimental results show that the proposed method outperforms existing state-of-the-art methods in these areas with fewer parameters, showcasing its potential for broad application in real-world scenarios where capturing high-resolution images is not feasible. Code is available at https://github.com/xyongLu/CMSA.
>
---
#### [replaced 038] Fine-grained Image Retrieval via Dual-Vision Adaptation
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.16273v2](http://arxiv.org/pdf/2506.16273v2)**

> **作者:** Xin Jiang; Meiqi Cao; Hao Tang; Fei Shen; Zechao Li
>
> **摘要:** Fine-Grained Image Retrieval~(FGIR) faces challenges in learning discriminative visual representations to retrieve images with similar fine-grained features. Current leading FGIR solutions typically follow two regimes: enforce pairwise similarity constraints in the semantic embedding space, or incorporate a localization sub-network to fine-tune the entire model. However, such two regimes tend to overfit the training data while forgetting the knowledge gained from large-scale pre-training, thus reducing their generalization ability. In this paper, we propose a Dual-Vision Adaptation (DVA) approach for FGIR, which guides the frozen pre-trained model to perform FGIR through collaborative sample and feature adaptation. Specifically, we design Object-Perceptual Adaptation, which modifies input samples to help the pre-trained model perceive critical objects and elements within objects that are helpful for category prediction. Meanwhile, we propose In-Context Adaptation, which introduces a small set of parameters for feature adaptation without modifying the pre-trained parameters. This makes the FGIR task using these adjusted features closer to the task solved during the pre-training. Additionally, to balance retrieval efficiency and performance, we propose Discrimination Perception Transfer to transfer the discriminative knowledge in the object-perceptual adaptation to the image encoder using the knowledge distillation mechanism. Extensive experiments show that DVA has fewer learnable parameters and performs well on three in-distribution and three out-of-distribution fine-grained datasets.
>
---
#### [replaced 039] STF: Spatial Temporal Fusion for Trajectory Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.18149v2](http://arxiv.org/pdf/2311.18149v2)**

> **作者:** Pengqian Han; Jiamou Liu; Tianzhe Bao; Yifei Wang
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Trajectory prediction is a challenging task that aims to predict the future trajectory of vehicles or pedestrians over a short time horizon based on their historical positions. The main reason is that the trajectory is a kind of complex data, including spatial and temporal information, which is crucial for accurate prediction. Intuitively, the more information the model can capture, the more precise the future trajectory can be predicted. However, previous works based on deep learning methods processed spatial and temporal information separately, leading to inadequate spatial information capture, which means they failed to capture the complete spatial information. Therefore, it is of significance to capture information more fully and effectively on vehicle interactions. In this study, we introduced an integrated 3D graph that incorporates both spatial and temporal edges. Based on this, we proposed the integrated 3D graph, which considers the cross-time interaction information. In specific, we design a Spatial-Temporal Fusion (STF) model including Multi-layer perceptions (MLP) and Graph Attention (GAT) to capture the spatial and temporal information historical trajectories simultaneously on the 3D graph. Our experiment on the ApolloScape Trajectory Datasets shows that the proposed STF outperforms several baseline methods, especially on the long-time-horizon trajectory prediction.
>
---
#### [replaced 040] Generating Synthetic Data via Augmentations for Improved Facial Resemblance in DreamBooth and InstantID
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.03557v2](http://arxiv.org/pdf/2505.03557v2)**

> **作者:** Koray Ulusan; Benjamin Kiefer
>
> **备注:** Accepted to CVPR 2025 Workshop "Synthetic Data for Computer Vision Workshop", https://syndata4cv.github.io/ Revised version
>
> **摘要:** Personalizing Stable Diffusion for professional portrait generation from amateur photos faces challenges in maintaining facial resemblance. This paper evaluates the impact of augmentation strategies on two personalization methods: DreamBooth and InstantID. We compare classical augmentations (flipping, cropping, color adjustments) with generative augmentation using InstantID's synthetic images to enrich training data. Using SDXL and a new FaceDistance metric based on FaceNet, we quantitatively assess facial similarity. Results show classical augmentations can cause artifacts harming identity retention, while InstantID improves fidelity when balanced with real images to avoid overfitting. A user study with 97 participants confirms high photorealism and preferences for InstantID's polished look versus DreamBooth's identity accuracy. Our findings inform effective augmentation strategies for personalized text-to-image generation.
>
---
#### [replaced 041] Site-Level Fine-Tuning with Progressive Layer Freezing: Towards Robust Prediction of Bronchopulmonary Dysplasia from Day-1 Chest Radiographs in Extremely Preterm Infants
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12269v2](http://arxiv.org/pdf/2507.12269v2)**

> **作者:** Sybelle Goedicke-Fritz; Michelle Bous; Annika Engel; Matthias Flotho; Pascal Hirsch; Hannah Wittig; Dino Milanovic; Dominik Mohr; Mathias Kaspar; Sogand Nemat; Dorothea Kerner; Arno Bücker; Andreas Keller; Sascha Meyer; Michael Zemlin; Philipp Flotho
>
> **备注:** S.G.-F., M.B., and A.E. contributed equally to this work and share first authorship. M.Z. and P.F. contributed equally to this work and share senior authorship
>
> **摘要:** Bronchopulmonary dysplasia (BPD) is a chronic lung disease affecting 35% of extremely low birth weight infants. Defined by oxygen dependence at 36 weeks postmenstrual age, it causes lifelong respiratory complications. However, preventive interventions carry severe risks, including neurodevelopmental impairment, ventilator-induced lung injury, and systemic complications. Therefore, early BPD prognosis and prediction of BPD outcome is crucial to avoid unnecessary toxicity in low risk infants. Admission radiographs of extremely preterm infants are routinely acquired within 24h of life and could serve as a non-invasive prognostic tool. In this work, we developed and investigated a deep learning approach using chest X-rays from 163 extremely low-birth-weight infants ($\leq$32 weeks gestation, 401-999g) obtained within 24 hours of birth. We fine-tuned a ResNet-50 pretrained specifically on adult chest radiographs, employing progressive layer freezing with discriminative learning rates to prevent overfitting and evaluated a CutMix augmentation and linear probing. For moderate/severe BPD outcome prediction, our best performing model with progressive freezing, linear probing and CutMix achieved an AUROC of 0.78 $\pm$ 0.10, balanced accuracy of 0.69 $\pm$ 0.10, and an F1-score of 0.67 $\pm$ 0.11. In-domain pre-training significantly outperformed ImageNet initialization (p = 0.031) which confirms domain-specific pretraining to be important for BPD outcome prediction. Routine IRDS grades showed limited prognostic value (AUROC 0.57 $\pm$ 0.11), confirming the need of learned markers. Our approach demonstrates that domain-specific pretraining enables accurate BPD prediction from routine day-1 radiographs. Through progressive freezing and linear probing, the method remains computationally feasible for site-level implementation and future federated learning deployments.
>
---
#### [replaced 042] ZIP: Scalable Crowd Counting via Zero-Inflated Poisson Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19955v2](http://arxiv.org/pdf/2506.19955v2)**

> **作者:** Yiming Ma; Victor Sanchez; Tanaya Guha
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Most crowd counting methods directly regress blockwise density maps using Mean Squared Error (MSE) losses. This practice has two key limitations: (1) it fails to account for the extreme spatial sparsity of annotations -- over 95% of 8x8 blocks are empty across standard benchmarks, so supervision signals in informative regions are diluted by the predominant zeros; (2) MSE corresponds to a Gaussian error model that poorly matches discrete, non-negative count data. To address these issues, we introduce ZIP, a scalable crowd counting framework that models blockwise counts with a Zero-Inflated Poisson likelihood: a zero-inflation term learns the probability a block is structurally empty (handling excess zeros), while the Poisson component captures expected counts when people are present (respecting discreteness). We provide a generalization analysis showing a tighter risk bound for ZIP than MSE-based losses and DMCount provided that the training resolution is moderately large. To assess the scalability of ZIP, we instantiate it on backbones spanning over 100x in parameters/compute. Experiments on ShanghaiTech A & B, UCF-QNRF, and NWPU-Crowd demonstrate that ZIP consistently surpasses state-of-the-art methods across all model scales.
>
---
#### [replaced 043] 4D-MISR: A unified model for low-dose super-resolution imaging via feature fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.09953v3](http://arxiv.org/pdf/2507.09953v3)**

> **作者:** Zifei Wang; Zian Mao; Xiaoya He; Xi Huang; Haoran Zhang; Chun Cheng; Shufen Chu; Tingzheng Hou; Xiaoqin Zeng; Yujun Xie
>
> **摘要:** While electron microscopy offers crucial atomic-resolution insights into structure-property relationships, radiation damage severely limits its use on beam-sensitive materials like proteins and 2D materials. To overcome this challenge, we push beyond the electron dose limits of conventional electron microscopy by adapting principles from multi-image super-resolution (MISR) that have been widely used in remote sensing. Our method fuses multiple low-resolution, sub-pixel-shifted views and enhances the reconstruction with a convolutional neural network (CNN) that integrates features from synthetic, multi-angle observations. We developed a dual-path, attention-guided network for 4D-STEM that achieves atomic-scale super-resolution from ultra-low-dose data. This provides robust atomic-scale visualization across amorphous, semi-crystalline, and crystalline beam-sensitive specimens. Systematic evaluations on representative materials demonstrate comparable spatial resolution to conventional ptychography under ultra-low-dose conditions. Our work expands the capabilities of 4D-STEM, offering a new and generalizable method for the structural analysis of radiation-vulnerable materials.
>
---
#### [replaced 044] DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04447v2](http://arxiv.org/pdf/2507.04447v2)**

> **作者:** Wenyao Zhang; Hongsi Liu; Zekun Qi; Yunnan Wang; Xinqiang Yu; Jiazhao Zhang; Runpei Dong; Jiawei He; He Wang; Zhizheng Zhang; Li Yi; Wenjun Zeng; Xin Jin
>
> **摘要:** Recent advances in vision-language-action (VLA) models have shown promise in integrating image generation with action prediction to improve generalization and reasoning in robot manipulation. However, existing methods are limited to challenging image-based forecasting, which suffers from redundant information and lacks comprehensive and critical world knowledge, including dynamic, spatial and semantic information. To address these limitations, we propose DreamVLA, a novel VLA framework that integrates comprehensive world knowledge forecasting to enable inverse dynamics modeling, thereby establishing a perception-prediction-action loop for manipulation tasks. Specifically, DreamVLA introduces a dynamic-region-guided world knowledge prediction, integrated with the spatial and semantic cues, which provide compact yet comprehensive representations for action planning. This design aligns with how humans interact with the world by first forming abstract multimodal reasoning chains before acting. To mitigate interference among the dynamic, spatial and semantic information during training, we adopt a block-wise structured attention mechanism that masks their mutual attention, preventing information leakage and keeping each representation clean and disentangled. Moreover, to model the conditional distribution over future actions, we employ a diffusion-based transformer that disentangles action representations from shared latent features. Extensive experiments on both real-world and simulation environments demonstrate that DreamVLA achieves 76.7% success rate on real robot tasks and 4.44 average length on the CALVIN ABC-D benchmarks.
>
---
#### [replaced 045] OscNet v1.5: Energy Efficient Hopfield Network on CMOS Oscillators for Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12610v2](http://arxiv.org/pdf/2506.12610v2)**

> **作者:** Wenxiao Cai; Zongru Li; Iris Wang; Yu-Neng Wang; Thomas H. Lee
>
> **摘要:** Machine learning has achieved remarkable advancements but at the cost of significant computational resources. This has created an urgent need for a novel and energy-efficient computational fabric and corresponding algorithms. CMOS Oscillator Networks (OscNet) is a brain inspired and specially designed hardware for low energy consumption. In this paper, we propose a Hopfield Network based machine learning algorithm that can be implemented on OscNet. The network is trained using forward propagation alone to learn sparsely connected weights, yet achieves an 8% improvement in accuracy compared to conventional deep learning models on MNIST dataset. OscNet v1.5 achieves competitive accuracy on MNIST and is well-suited for implementation using CMOS-compatible ring oscillator arrays with SHIL. In oscillator-based inference, we utilize only 24% of the connections used in a fully connected Hopfield network, with merely a 0.1% drop in accuracy. OscNet v1.5 relies solely on forward propagation and employs sparse connections, making it an energy-efficient machine learning pipeline designed for oscillator computing fabric. The repository for OscNet family is: https://github.com/RussRobin/OscNet .
>
---
#### [replaced 046] Monocular 3D Hand Pose Estimation with Implicit Camera Alignment
- **分类: cs.CV; cs.GR; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.11133v2](http://arxiv.org/pdf/2506.11133v2)**

> **作者:** Christos Pantazopoulos; Spyridon Thermos; Gerasimos Potamianos
>
> **备注:** Code is available at the project page https://cpantazop.github.io/HandRepo/
>
> **摘要:** Estimating the 3D hand articulation from a single color image is an important problem with applications in Augmented Reality (AR), Virtual Reality (VR), Human-Computer Interaction (HCI), and robotics. Apart from the absence of depth information, occlusions, articulation complexity, and the need for camera parameters knowledge pose additional challenges. In this work, we propose an optimization pipeline for estimating the 3D hand articulation from 2D keypoint input, which includes a keypoint alignment step and a fingertip loss to overcome the need to know or estimate the camera parameters. We evaluate our approach on the EgoDexter and Dexter+Object benchmarks to showcase that it performs competitively with the state-of-the-art, while also demonstrating its robustness when processing "in-the-wild" images without any prior camera knowledge. Our quantitative analysis highlights the sensitivity of the 2D keypoint estimation accuracy, despite the use of hand priors. Code is available at the project page https://cpantazop.github.io/HandRepo/
>
---
#### [replaced 047] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03558v3](http://arxiv.org/pdf/2412.03558v3)**

> **作者:** Zehuan Huang; Yuan-Chen Guo; Xingqiao An; Yunhan Yang; Yangguang Li; Zi-Xin Zou; Ding Liang; Xihui Liu; Yan-Pei Cao; Lu Sheng
>
> **备注:** Project page: https://huanngzh.github.io/MIDI-Page/
>
> **摘要:** This paper introduces MIDI, a novel paradigm for compositional 3D scene generation from a single image. Unlike existing methods that rely on reconstruction or retrieval techniques or recent approaches that employ multi-stage object-by-object generation, MIDI extends pre-trained image-to-3D object generation models to multi-instance diffusion models, enabling the simultaneous generation of multiple 3D instances with accurate spatial relationships and high generalizability. At its core, MIDI incorporates a novel multi-instance attention mechanism, that effectively captures inter-object interactions and spatial coherence directly within the generation process, without the need for complex multi-step processes. The method utilizes partial object images and global scene context as inputs, directly modeling object completion during 3D generation. During training, we effectively supervise the interactions between 3D instances using a limited amount of scene-level data, while incorporating single-object data for regularization, thereby maintaining the pre-trained generalization ability. MIDI demonstrates state-of-the-art performance in image-to-scene generation, validated through evaluations on synthetic data, real-world scene data, and stylized scene images generated by text-to-image diffusion models.
>
---
#### [replaced 048] Fetuses Made Simple: Modeling and Tracking of Fetal Shape and Pose
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17858v3](http://arxiv.org/pdf/2506.17858v3)**

> **作者:** Yingcheng Liu; Peiqi Wang; Sebastian Diaz; Esra Abaci Turk; Benjamin Billot; P. Ellen Grant; Polina Golland
>
> **摘要:** Analyzing fetal body motion and shape is paramount in prenatal diagnostics and monitoring. Existing methods for fetal MRI analysis mainly rely on anatomical keypoints or volumetric body segmentations. Keypoints simplify body structure to facilitate motion analysis, but may ignore important details of full-body shape. Body segmentations capture complete shape information but complicate temporal analysis due to large non-local fetal movements. To address these limitations, we construct a 3D articulated statistical fetal body model based on the Skinned Multi-Person Linear Model (SMPL). Our algorithm iteratively estimates body pose in the image space and body shape in the canonical pose space. This approach improves robustness to MRI motion artifacts and intensity distortions, and reduces the impact of incomplete surface observations due to challenging fetal poses. We train our model on segmentations and keypoints derived from $19,816$ MRI volumes across $53$ subjects. Our model captures body shape and motion across time series and provides intuitive visualization. Furthermore, it enables automated anthropometric measurements traditionally difficult to obtain from segmentations and keypoints. When tested on unseen fetal body shapes, our method yields a surface alignment error of $3.2$ mm for $3$ mm MRI voxel size. To our knowledge, this represents the first 3D articulated statistical fetal body model, paving the way for enhanced fetal motion and shape analysis in prenatal diagnostics. The code is available at https://github.com/MedicalVisionGroup/fetal-smpl .
>
---
#### [replaced 049] Creating a Historical Migration Dataset from Finnish Church Records, 1800-1920
- **分类: cs.CV; I.4.6, J.5**

- **链接: [http://arxiv.org/pdf/2506.07960v2](http://arxiv.org/pdf/2506.07960v2)**

> **作者:** Ari Vesalainen; Jenna Kanerva; Aida Nitsch; Kiia Korsu; Ilari Larkiola; Laura Ruotsalainen; Filip Ginter
>
> **摘要:** This article presents a large-scale effort to create a structured dataset of internal migration in Finland between 1800 and 1920 using digitized church moving records. These records, maintained by Evangelical-Lutheran parishes, document the migration of individuals and families and offer a valuable source for studying historical demographic patterns. The dataset includes over six million entries extracted from approximately 200,000 images of handwritten migration records. The data extraction process was automated using a deep learning pipeline that included layout analysis, table detection, cell classification, and handwriting recognition. The complete pipeline was applied to all images, resulting in a structured dataset suitable for research. The dataset can be used to study internal migration, urbanization, and family migration, and the spread of disease in preindustrial Finland. A case study from the Elim\"aki parish shows how local migration histories can be reconstructed. The work demonstrates how large volumes of handwritten archival material can be transformed into structured data to support historical and demographic research.
>
---
#### [replaced 050] Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.23114v4](http://arxiv.org/pdf/2410.23114v4)**

> **作者:** Junjie Wu; Tsz Ting Chung; Kai Chen; Dit-Yan Yeung
>
> **备注:** Accepted by TMLR 2025. Project Page: https://kaichen1998.github.io/projects/tri-he/
>
> **摘要:** Despite the outstanding performance in vision-language reasoning, Large Vision-Language Models (LVLMs) might generate hallucinated contents that do not exist in the given image. Most existing LVLM hallucination benchmarks are constrained to evaluate the object-related hallucinations. However, the potential hallucination on the relations between two objects, i.e., relation hallucination, still lacks investigation. To remedy that, we design a unified framework to measure the object and relation hallucination in LVLMs simultaneously. The core idea of our framework is to evaluate hallucinations via (object, relation, object) triplets extracted from LVLMs' responses, making it easily generalizable to different vision-language tasks. Based on our framework, we further introduce Tri-HE, a novel Triplet-level Hallucination Evaluation benchmark which can be used to study both object and relation hallucination at the same time. With comprehensive evaluations on Tri-HE, we observe that the relation hallucination issue is even more serious than object hallucination among existing LVLMs, highlighting a previously neglected problem towards reliable LVLMs. Moreover, based on our findings, we design a simple training-free approach that effectively mitigates hallucinations for LVLMs. Our dataset and code for the reproduction of our experiments are available publicly at https://github.com/wujunjie1998/Tri-HE.
>
---
#### [replaced 051] A Brain Tumor Segmentation Method Based on CLIP and 3D U-Net with Cross-Modal Semantic Guidance and Multi-Level Feature Fusion
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.09966v2](http://arxiv.org/pdf/2507.09966v2)**

> **作者:** Mingda Zhang
>
> **备注:** 13 pages,6 figures
>
> **摘要:** Precise segmentation of brain tumors from magnetic resonance imaging (MRI) is essential for neuro-oncology diagnosis and treatment planning. Despite advances in deep learning methods, automatic segmentation remains challenging due to tumor morphological heterogeneity and complex three-dimensional spatial relationships. Current techniques primarily rely on visual features extracted from MRI sequences while underutilizing semantic knowledge embedded in medical reports. This research presents a multi-level fusion architecture that integrates pixel-level, feature-level, and semantic-level information, facilitating comprehensive processing from low-level data to high-level concepts. The semantic-level fusion pathway combines the semantic understanding capabilities of Contrastive Language-Image Pre-training (CLIP) models with the spatial feature extraction advantages of 3D U-Net through three mechanisms: 3D-2D semantic bridging, cross-modal semantic guidance, and semantic-based attention mechanisms. Experimental validation on the BraTS 2020 dataset demonstrates that the proposed model achieves an overall Dice coefficient of 0.8567, representing a 4.8% improvement compared to traditional 3D U-Net, with a 7.3% Dice coefficient increase in the clinically important enhancing tumor (ET) region.
>
---
#### [replaced 052] Depth-Sequence Transformer (DST) for Segment-Specific ICA Calcification Mapping on Non-Contrast CT
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08214v2](http://arxiv.org/pdf/2507.08214v2)**

> **作者:** Xiangjian Hou; Ebru Yaman Akcicek; Xin Wang; Kazem Hashemizadeh; Scott Mcnally; Chun Yuan; Xiaodong Ma
>
> **摘要:** While total intracranial carotid artery calcification (ICAC) volume is an established stroke biomarker, growing evidence shows this aggregate metric ignores the critical influence of plaque location, since calcification in different segments carries distinct prognostic and procedural risks. However, a finer-grained, segment-specific quantification has remained technically infeasible. Conventional 3D models are forced to process downsampled volumes or isolated patches, sacrificing the global context required to resolve anatomical ambiguity and render reliable landmark localization. To overcome this, we reformulate the 3D challenge as a \textbf{Parallel Probabilistic Landmark Localization} task along the 1D axial dimension. We propose the \textbf{Depth-Sequence Transformer (DST)}, a framework that processes full-resolution CT volumes as sequences of 2D slices, learning to predict $N=6$ independent probability distributions that pinpoint key anatomical landmarks. Our DST framework demonstrates exceptional accuracy and robustness. Evaluated on a 100-patient clinical cohort with rigorous 5-fold cross-validation, it achieves a Mean Absolute Error (MAE) of \textbf{0.1 slices}, with \textbf{96\%} of predictions falling within a $\pm1$ slice tolerance. Furthermore, to validate its architectural power, the DST backbone establishes the best result on the public Clean-CC-CCII classification benchmark under an end-to-end evaluation protocol. Our work delivers the first practical tool for automated segment-specific ICAC analysis. The proposed framework provides a foundation for further studies on the role of location-specific biomarkers in diagnosis, prognosis, and procedural planning.
>
---
#### [replaced 053] DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19263v3](http://arxiv.org/pdf/2503.19263v3)**

> **作者:** Fucai Ke; Vijay Kumar B G; Xingjian Leng; Zhixi Cai; Zaid Khan; Weiqing Wang; Pari Delir Haghighi; Hamid Rezatofighi; Manmohan Chandraker
>
> **备注:** ICCV 2025
>
> **摘要:** Visual reasoning (VR), which is crucial in many fields for enabling human-like visual understanding, remains highly challenging. Recently, compositional visual reasoning approaches, which leverage the reasoning abilities of large language models (LLMs) with integrated tools to solve problems, have shown promise as more effective strategies than end-to-end VR methods. However, these approaches face limitations, as frozen LLMs lack tool awareness in VR, leading to performance bottlenecks. While leveraging LLMs for reasoning is widely used in other domains, they are not directly applicable to VR due to limited training data, imperfect tools that introduce errors and reduce data collection efficiency in VR, and challenging in fine-tuning on noisy workflows. To address these challenges, we propose DWIM: i) Discrepancy-aware training Workflow generation, which assesses tool usage and extracts more viable workflows for training; and ii) Instruct-Masking fine-tuning, which guides the model to only clone effective actions, enabling the generation of more practical solutions. Our experiments demonstrate that DWIM achieves state-of-the-art performance across various VR tasks, exhibiting strong generalization on multiple widely-used datasets.
>
---
#### [replaced 054] A Progressive Image Restoration Network for High-order Degradation Imaging in Remote Sensing
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.07195v2](http://arxiv.org/pdf/2412.07195v2)**

> **作者:** Yujie Feng; Yin Yang; Xiaohong Fan; Zhengpeng Zhang; Lijing Bu; Jianping Zhang
>
> **备注:** 17 pages, Accepted to Transactions on Geoscience and Remote Sensing (TGRS), July 16, 2025
>
> **摘要:** Recently, deep learning methods have gained remarkable achievements in the field of image restoration for remote sensing (RS). However, most existing RS image restoration methods focus mainly on conventional first-order degradation models, which may not effectively capture the imaging mechanisms of remote sensing images. Furthermore, many RS image restoration approaches that use deep learning are often criticized for their lacks of architecture transparency and model interpretability. To address these problems, we propose a novel progressive restoration network for high-order degradation imaging (HDI-PRNet), to progressively restore different image degradation. HDI-PRNet is developed based on the theoretical framework of degradation imaging, also Markov properties of the high-order degradation process and Maximum a posteriori (MAP) estimation, offering the benefit of mathematical interpretability within the unfolding network. The framework is composed of three main components: a module for image denoising that relies on proximal mapping prior learning, a module for image deblurring that integrates Neumann series expansion with dual-domain degradation learning, and a module for super-resolution. Extensive experiments demonstrate that our method achieves superior performance on both synthetic and real remote sensing images.
>
---
#### [replaced 055] Color Image Set Recognition Based on Quaternionic Grassmannians
- **分类: cs.CV; math.AG**

- **链接: [http://arxiv.org/pdf/2505.23629v2](http://arxiv.org/pdf/2505.23629v2)**

> **作者:** Xiang Xiang Wang; Tin-Yau Tam
>
> **摘要:** We propose a new method for recognizing color image sets using quaternionic Grassmannians, which use the power of quaternions to capture color information and represent each color image set as a point on the quaternionic Grassmannian. We provide a direct formula to calculate the shortest distance between two points on the quaternionic Grassmannian, and use this distance to build a new classification framework. Experiments on the ETH-80 benchmark dataset and and the Highway Traffic video dataset show that our method achieves good recognition results. We also discuss some limitations in stability and suggest ways the method can be improved in the future.
>
---
#### [replaced 056] Exploring the Collaborative Advantage of Low-level Information on Generalizable AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00463v2](http://arxiv.org/pdf/2504.00463v2)**

> **作者:** Ziyin Zhou; Ke Sun; Zhongxi Chen; Xianming Lin; Yunpeng Luo; Ke Yan; Shouhong Ding; Xiaoshuai Sun
>
> **摘要:** Existing state-of-the-art AI-Generated image detection methods mostly consider extracting low-level information from RGB images to help improve the generalization of AI-Generated image detection, such as noise patterns. However, these methods often consider only a single type of low-level information, which may lead to suboptimal generalization. Through empirical analysis, we have discovered a key insight: different low-level information often exhibits generalization capabilities for different types of forgeries. Furthermore, we found that simple fusion strategies are insufficient to leverage the detection advantages of each low-level and high-level information for various forgery types. Therefore, we propose the Adaptive Low-level Experts Injection (ALEI) framework. Our approach introduces Lora Experts, enabling the backbone network, which is trained with high-level semantic RGB images, to accept and learn knowledge from different low-level information. We utilize a cross-attention method to adaptively fuse these features at intermediate layers. To prevent the backbone network from losing the modeling capabilities of different low-level features during the later stages of modeling, we developed a Low-level Information Adapter that interacts with the features extracted by the backbone network. Finally, we propose Dynamic Feature Selection, which dynamically selects the most suitable features for detecting the current image to maximize generalization detection capability. Extensive experiments demonstrate that our method, finetuned on only four categories of mainstream ProGAN data, performs excellently and achieves state-of-the-art results on multiple datasets containing unseen GAN and Diffusion methods.
>
---
#### [replaced 057] (Almost) Free Modality Stitching of Foundation Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.10015v3](http://arxiv.org/pdf/2507.10015v3)**

> **作者:** Jaisidh Singh; Diganta Misra; Boris Knyazev; Antonio Orvieto
>
> **备注:** Pre-print
>
> **摘要:** Foundation multi-modal models are often designed by stitching of multiple existing pretrained uni-modal models: for example, an image classifier with an text model. This stitching process is performed by training a connector module that aims to align the representation spaces of these uni-modal models towards a multi-modal objective. However, given the complexity of training such connectors on large scale web-based datasets coupled with the ever-increasing number of available pretrained uni-modal models, the task of uni-modal models selection and subsequent connector module training becomes computationally demanding. To address this under-studied critical problem, we propose Hypernetwork Model Alignment (Hyma), a novel all-in-one solution for optimal uni-modal model selection and connector training by leveraging hypernetworks. Specifically, our framework utilizes the parameter prediction capability of a hypernetwork to obtain jointly trained connector modules for $N \times M$ combinations of uni-modal models. In our experiments, Hyma reduces the cost of searching for the best performing uni-modal model pair by $10\times$, while matching the ranking and trained connector performance obtained via grid search across a suite of diverse multi-modal benchmarks.
>
---
