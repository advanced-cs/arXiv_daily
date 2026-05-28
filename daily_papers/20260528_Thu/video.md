# 计算机视觉 cs.CV

- **最新发布 133 篇**

- **更新 109 篇**

## 最新发布

#### [new 001] A Road-Conditioned Traffic Movie Prediction Network with Spatiotemporal and Structure-Consistent Learning
- **分类: cs.CV**

- **简介: 该论文属于交通预测任务，旨在解决城市交通地图准确预测问题。提出RCSNet模型，结合道路结构进行时空学习，提升预测精度与结构一致性。**

- **链接: [https://arxiv.org/pdf/2605.27884](https://arxiv.org/pdf/2605.27884)**

> **作者:** Joshua Kofi Asamoah; Blessing Agyei Kyem; Armstrong Aboah
>
> **备注:** 22 pages (double column), 7 Tables, 11 Figures
>
> **摘要:** City-wide traffic forecasting is important for congestion management, route guidance, and intelligent transportation systems, but accurate prediction remains challenging when future traffic must be generated as spatial maps over an entire urban network. Existing traffic movie prediction methods have improved frame-level accuracy, yet many still treat forecasting mainly as image reconstruction. This can produce traffic maps that are numerically close to the ground truth but weakly constrained by road layout, connectivity, travel direction, and congestion propagation, especially in cross-city settings where both traffic behavior and road structure change. To address this limitation, this study proposes RCSNet, a road-conditioned spatiotemporal network that reformulates traffic movie prediction as topology-guided future-state generation. RCSNet extracts road-aware representations from static road maps, models multi-horizon traffic dynamics from historical observations, aligns directional traffic features with local road structure, and progressively generates future traffic maps for improved temporal consistency. A structure-consistent learning objective further encourages predictions to remain accurate, road-aligned, and spatially stable. Experiments across multiple cities show that RCSNet improves both forecasting accuracy and structural consistency. In same-city forecasting on Berlin, Antwerp, and Moscow, RCSNet reduces average MAE, MSE, and RMSE by 11.5%, 10.0%, and 5.1%, respectively, compared with the closest baseline. In cross-city testing on unseen Chicago and Bangkok, it reduces RMSE by 10.6% and 10.5% without target-city fine-tuning. Additional horizon-wise, road-structure, explainability, statistical, and efficiency analyses show that RCSNet produces more accurate, transferable, road-aligned, and computationally efficient traffic forecasts.
>
---
#### [new 002] Transfer learning RGB models to hyperspectral images with trainable tensor decompositions
- **分类: cs.CV**

- **简介: 该论文属于跨模态迁移学习任务，解决RGB模型在 hyperspectral 图像上的应用问题。通过可训练张量分解，保留图像和空间信息，提升模型适应性。**

- **链接: [https://arxiv.org/pdf/2605.28331](https://arxiv.org/pdf/2605.28331)**

> **作者:** Mariette Schönfeld; Laurens Devos; Wannes Meert; Hendrik Blockeel
>
> **摘要:** Transfer learning makes it possible to use large vision networks on a variety of domains, by specializing their models' general filters to new tasks. However, these networks assume the input images to have 3 input channels, making them incompatible with multi- or hyperspectral images. Current approaches that mitigate this incompatibility sacrifice information in either the image, or the model. This work proposes a novel approach that preserves the image and spatial information present in the model by using partially trainable tensor decompositions. We create such decompositions of pretrained convolutional filters, separating the filters into spatial and spectral components. The spectral components are then replaced with trainable components of higher channel dimensionality. This creates hyperspectral filters that can specialize to new datasets, while retaining the spatial patterns of the original filter. Experiments on a variety of hyperspectral datasets show that our approach is more accurate and robust than other hyperspectral transfer learning methods.
>
---
#### [new 003] GEM: Generative Supervision Helps Embodied Intelligence
- **分类: cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决 embodied VLMs 语义与物理知识脱节的问题。通过引入深度图生成任务，提升模型的物理操作能力。**

- **链接: [https://arxiv.org/pdf/2605.28548](https://arxiv.org/pdf/2605.28548)**

> **作者:** Ruowen Zhao; Bangguo Li; Zuyan Liu; Yinan Liang; Junliang Ye; Fangfu Liu; Diankun Wu; Zhengyi Wang; Xumin Yu; Yongming Rao; Han Hu; Jun Zhu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Embodied Vision-Language Models (VLMs) have demonstrated impressive performance and generalization in robotics, particularly within Vision-Language-Action frameworks. However, a significant gap remains between the high-level semantic focus of standard text-guided pre-training paradigms and the low-level spatial and physical knowledge critical for execution in embodied environments. In this paper, we introduce GEM, a Generative-supervised Embodied vision-language Model designed to bridge this divide. We propose integrating a depth map generation task directly into the VLM pre-training phase. By training this generative objective jointly with the main model, we observe substantial improvements in embodied intelligence, significantly enhancing both semantic understanding and physical operation capabilities. To support this paradigm, we curate and release GEM-4M, a comprehensive large-scale dataset featuring a mixture of grounding, reasoning, and planning data paired with high-quality depth supervision. Extensive experiments demonstrate that GEM achieves state-of-the-art results across diverse embodied benchmarks. Furthermore, our deployed action model, GEM-VLA, exhibits vastly superior task execution abilities in both simulation environments and real-world evaluations. Code, models, and datasets are available at this https URL
>
---
#### [new 004] Clinical Validation of the Melanoscope AI Mobile Dermoscopy Clinical Decision Support System
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于皮肤癌检测任务，旨在解决 dermatologist 缺乏和模型可解释性问题。开发了可解释的深度学习模型和患者分诊算法，并在俄罗斯进行了临床验证。**

- **链接: [https://arxiv.org/pdf/2605.27561](https://arxiv.org/pdf/2605.27561)**

> **作者:** Elena Sergeevna Kozachok; Sergey Sergeevich Seregin
>
> **备注:** 24 pages, 6 figures, 5 tables, 21 references
>
> **摘要:** Introduction. Early detection of malignant skin lesions is critical for prognosis, yet dermatologist shortages in Russian regions limit screening coverage. Mobile dermoscopy clinical decision support systems (CDSS) offer a promising approach, with model interpretability and standardised patient routing remaining key barriers to adoption. Aim. To develop a quantitative interpretability assessment method for cascade deep learning models and a three-zone patient routing algorithm, and to conduct a preliminary single-centre prospective clinical validation of the Melanoscope AI CDSS in Russian outpatient practice. Material and methods. Two-stage cascade classification of dermoscopic images; attention map visualisation (attention rollout for ViT and Swin; Grad-CAM for ConvNeXt and EfficientNetV2); quantitative IoU-based agreement assessment between activation maps and expert annotations; prospective single-centre validation across four "Melanoma Day" sessions (Orel, Russia, June 2025 - April 2026). Results. On 176 patients: agreement with expert assessment 88.6%; no false negatives among 5 malignant lesions (95% CI: 47.8-100.0%); specificity 88.3%. Three melanomas and two basal cell carcinomas were histologically confirmed; six dysplastic naevi placed under follow-up. Mean IoU (n=180): ViT - 0.69; Swin - 0.64; ConvNeXt - 0.53; EfficientNetV2 - 0.51. Routing thresholds: P<0.15 / 0.15-0.50 / >=0.50. Conclusion. No false negatives were observed; specificity was 88.3%, supporting screening use. The integrated cascade classification, attention map visualisation with IoU assessment, and three-zone routing provide reproducible, interpretable clinical decision support adaptable to varying resource levels.
>
---
#### [new 005] Sketch2Motion: Text-driven 2D Sketch to 3D Animation via Diffusion-guided Skeleton Optimization
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出Sketch2Motion，解决从2D手绘草图生成3D动画的任务，通过扩散引导骨架优化实现文本驱动的运动合成。**

- **链接: [https://arxiv.org/pdf/2605.28394](https://arxiv.org/pdf/2605.28394)**

> **作者:** Gaurav Rai; Ojaswa Sharma
>
> **摘要:** Animation of 2D hand-drawn sketches provides an effective medium for visual communication. However, these sketches pose challenges, particularly in handling occlusions and accurately mapping motion. While 3D animation naturally addresses these challenges, estimating 3D motion remains a very complex task. Recent approaches to converting 2D sketches to 3D animations have mainly focused on specific types of motion, such as bipedal movements and facial expressions. We propose Sketch2Motion, a diffusion-guided framework for skeleton-based motion synthesis that combines classical character animation pipelines with deep generative priors. Our method represents motion using skeletal transformations, which are propagated to mesh deformations via linear blend skinning. To guide the resulting animation toward realistic and semantically meaningful motion, we integrate a text-to-video diffusion model via motion-aware score-distillation sampling (MoSDS), enabling optimization without paired motion data. Additionally, we apply physics-inspired smoothness, topological, and contact constraints to stabilize optimization and preserve motion plausibility. Further, we integrate a spring-mass simulator to introduce secondary motion effects. The proposed framework is generalized, fully differentiable, modular, and compatible with biped, quadruped, and non-living articulated characters. Experiments demonstrate that our approach produces temporally coherent, text-aligned animations that outperform baseline motion transfer methods that lack generative priors or explicit physical constraints. We will make our code and dataset publicly available.
>
---
#### [new 006] AndroidDaily: A Verifiable Benchmark for Mobile GUI Agents on Real-World Closed-Source Applications
- **分类: cs.CV; cs.SE**

- **简介: 该论文提出AndroidDaily基准和GRADE评估方法，解决真实封闭应用中GUI代理的验证难题。任务属于移动GUI代理评估，旨在提升实际应用场景下的自动验证能力。**

- **链接: [https://arxiv.org/pdf/2605.27761](https://arxiv.org/pdf/2605.27761)**

> **作者:** Yifan Sui; Xin Huang; Hongbing Li; Fang Xu; Jiahe Lv; Haolong Yan; Yeqing Shen; Litao Liu; Zhimin Fan; Ziyang Meng; Jia Wang; Junbo Qi; Kaijun Tan; Zheng Ge; Xiangyu Zhang; Daxin Jiang; Osamu Yoshie
>
> **备注:** 11 pages, 6 figures. Preprint
>
> **摘要:** The rapid development of GUI foundation models and mobile GUI agents has spurred numerous evaluation benchmarks, yet most rely on simulated environments or open-source applications, leaving real-world closed-source applications largely unevaluated. The core difficulty is that closed-source applications do not expose internal states, making traditional automatic verification inapplicable. To bridge this gap, we introduce AndroidDaily, a large-scale benchmark comprising 350 realistic daily-use tasks across 94 high-frequency Android applications spanning transportation, shopping, local services, entertainment, content creation, social media, and everyday utilities. To enable automatic and verifiable assessment in these opaque environments, we propose Guideline-grounded Reviewer for Automatic Diagnostic Evaluation (GRADE), a process-aware evaluator built on a three-tiered system of observable external guidelines: operational obligations, output quality, and negative constraints. GRADE tracks the agent's visual trajectory against these criteria and produces step-level diagnostic judgments, turning long-horizon, open-ended mobile interactions into verifiable evaluation without relying on hidden internal states. Experiments show that GRADE achieves 87.37\% agreement with human evaluators. The strongest model reaches a 62.0\% success rate on AndroidDaily, highlighting a substantial gap between current reasoning capabilities and practical execution in realistic mobile workflows.
>
---
#### [new 007] Con-DSO: Learning Short-Horizon Consistency Priors for RGB-D Direct Sparse Odometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉里程计任务，解决RGB-D直接稀疏里程计在复杂环境中的一致性问题。通过预测光流和深度一致性不确定性，提升跟踪鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.27952](https://arxiv.org/pdf/2605.27952)**

> **作者:** Haolan Zhang; Thanh Nguyen Canh; Chenghao Li; Ziyan Gao; Xiongwen Jiang; Nak Young Chong
>
> **备注:** Submitted
>
> **摘要:** Visual odometry (VO) is a fundamental component in robotics and augmented reality. RGB-D direct VO benefits from metric depth measurements, but it can degrade in challenging environments, where dynamic objects, occlusions, illumination changes, and unreliable depth violate the short-horizon photometric and depth-geometric consistency assumptions used by direct alignment. Existing approaches mitigate these issues through semantic filtering, explicit occlusion reasoning, illumination adaptation, or hand-crafted geometric criteria, but often rely on external modules or fixed assumptions tailored to individual failure modes, limiting their flexibility and ability to handle diverse challenges in a unified manner. In this work, we propose Con-DSO, a consistency-aware RGB-D direct sparse odometry framework that predicts dense photometric and depth-geometric consistency uncertainty from temporally adjacent RGB-D frame pairs. The consistency network is trained using flow-guided photometric errors and projective depth-consistency errors, allowing consistency violations to be represented as pixel-level uncertainty. These pairwise uncertainty predictions are converted into a host-side quality prior for keyframe-based tracking. The prior is then applied to VO through quality-aware support-pixel selection and decoupled photometric-geometric weighting during pose estimation, enabling continuous attenuation of unreliable observations rather than hard rejection or threshold-based gating. Experiments on five public RGB-D benchmarks show substantial gains over direct RGB-D VO baselines, with over 20\% absolute trajectory error reduction on ICL-NUIM and 50\%--80\% reductions on RGB-D Scenes V2, TUM/Bonn Dynamic, and OpenLORIS sequences.
>
---
#### [new 008] LV-OSD: Language-Vision-Complementary Open-Set Object Detection
- **分类: cs.CV**

- **简介: 该论文提出LV-OSD任务，解决通过文本和图像提示进行开放集目标检测的问题。设计双分支框架LVDor与TPDW模块，实现多模态提示对齐，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.28271](https://arxiv.org/pdf/2605.28271)**

> **作者:** Yupeng Zhang; Ruize Han; Wei Feng; Song Wang; Liang Wan
>
> **摘要:** Object detection is an important task in computer vision, which aims to detect the objects of interest. through the given category list or query images. In this work, we propose a new problem of language-visual-complementary open-set object detection (LV-OSD), i.e., using the flexible text-based and/or image-based prompts to specify the desired object categories. This setting is more common and practical in real-world applications. For this purpose, we design a dual-branch detection framework, LVDor, which can simultaneously accept both text and image prompts. Specifically, we first build the Multi-modal Prompts (MPr) containing various text descriptions and image samples for each category. Subsequently, to bridge the semantic gap among the input image, text prompts, and image prompts, we design a Target-guided Prompt Dynamic Weighting (TPDW) module. Guided by the prior information of the target image, this module dynamically produces the text and image prompts that best align with the target semantics, achieving precise alignment and effectively reducing the discrepancy between the two modalities, thereby accommodating the LV-OSD setting. We also propose a simple Prompt Random Masking (PRM) mechanism during training to simulate the arbitrary combination of text and/or image prompts in testing. Extensive experimental results verify our problem formulation's reasonability and our method's effectiveness. Prompts and code will be released publicly.
>
---
#### [new 009] OphIn-500K: Curating Web-Scale Visual Instructions for Scaling Ophthalmic Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学多模态语言模型任务，旨在解决眼科领域数据稀缺问题。通过构建大规模眼科指令数据集OphIn-500K，提升模型的临床理解和对话能力。**

- **链接: [https://arxiv.org/pdf/2605.27916](https://arxiv.org/pdf/2605.27916)**

> **作者:** Xuanzhao Dong; Wenhui Zhu; Xiwen Chen; Hao Wang; Xin Li; Yujian Xiong; Jiajun Cheng; Jingjing Wang; Xiaobing Yu; Haiyu Wu; Shao Tang; Zhipeng Wang; Langechuan Liu; Shan Lin; Oana Dumitrascu; Yalin Wang
>
> **摘要:** The advancement of general medical Multimodal Large Language Models (MLLMs) has shown great potential for building conversational assistants to support clinical diagnosis. However, their adaptation to highly specialized domains such as ophthalmology remains underexplored, primarily due to the scarcity of large-scale, domain-specific instruction-tuning data. Existing ophthalmic datasets for conversational agents are often limited in scale and largely rely on images from established public benchmarks, limiting the scalability of ophthalmic MLLMs and their ability to capture real-world clinical complexity. To address this gap, we propose $\textbf{OphIn-Engine}$, an ophthalmology-specific instruction data curation pipeline that constructs high-quality instruction data from open-access ophthalmology web-scale videos. The pipeline integrates multimodal transcription for extracting image-transcript pairs, visual cue separation and scoring for identifying clinically relevant visual descriptions, and instruction synthesis with quality control for generating accurate and diverse clinical dialogues. Using this engine, we introduce $\textbf{OphIn-500K}$, a large-scale multimodal ophthalmology instruction-tuning dataset containing over 500,000 instruction instances and more than 151,000 unique images from over 29,000 video clips, formatted as visual question answering (VQA), multi-turn conversational interactions, and chain-of-thought (CoT) reasoning. Built upon this dataset, we further develop $\textbf{OphIn-VL}$, an ophthalmology-specific MLLM with advanced visual understanding and conversational capabilities. Comprehensive experiments and case studies demonstrate that OphIn-VL achieves superior performance compared with state-of-the-art general medical and domain-specific MLLMs.
>
---
#### [new 010] Revisiting Change Detection Methods for their Application to Serac Fall Time-Lapse Monitoring
- **分类: cs.CV; cs.AI**

- **简介: 论文属于变化检测任务，旨在解决时间序列图像中体积变化的自动检测问题。针对传感器部署成本高、盲区多的问题，研究利用时差相机数据，提出新数据集并评估不同方法的有效性。**

- **链接: [https://arxiv.org/pdf/2605.28100](https://arxiv.org/pdf/2605.28100)**

> **作者:** Arthur Dérédel; Carlos Crispim-Junior; Pierre Lemaire; Johan Berthet; Laure Tougne Rodet
>
> **备注:** Preprint, 19 pages, 8 figures
>
> **摘要:** In an era where climate change aggravates environmental uncertainties, the identification and detection of event precursors are becoming crucial to mitigate the impacts of disastrous natural hazards. While classical sensors such as interferometric lasers or seismometers are reliable, their widespread deployment is often hindered by logistical and economic barriers, leaving numerous blind spots. Time-lapse cameras, which already provide cost-effective, high-resolution visual context to such sensors, present a promising alternative. However, processing their output automatically faces significant challenges, notably linked to extreme shape and lighting variations. Overcoming those issues is essential to deploy them at large-scale as a monitoring tool. This paper introduces a novel sub-task of change detection, namely volumetric change detection, applied to time-lapse cameras and slope instabilities. We conduct a comprehensive review of state-of-the-art change detection methods and related tasks, analyze their core components and assess their applicability to this context. To that end, we introduce the new dataset SeracFallDet, which contains serac fall annotations and has been thoroughly annotated to meet the latter demand. Through generalization experiments, we demonstrate that dense and semi-dense feature matching, although not trained specifically for this task, exhibit robust performance. Alternatively, supervised approaches struggle with data scarcity and annotation imbalance. This suggests that hybrid methods may offer a path forward by leveraging the strengths of both tasks. These findings highlight the potential of feature matching techniques and the need for further innovation to overcome the challenges of real-world deployment in environmental monitoring.
>
---
#### [new 011] Representation-Conditioned Diffusion Models for Guided Training Data Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉学习任务，旨在解决数据稀缺问题。通过生成合成数据提升模型性能，使用基于表示的扩散模型生成高质量图像，增强数据集效果。**

- **链接: [https://arxiv.org/pdf/2605.27495](https://arxiv.org/pdf/2605.27495)**

> **作者:** Nithesh Chandher Karthikeyan; Jonas Unger; Gabriel Eilertsen
>
> **摘要:** Data availability remains a critical bottleneck in many deep learning applications. Large-scale datasets are often expensive to collect, curate and annotate, which can limit the scalability and applicability of supervised learning methods. In this work, we evaluate the classification performance of models trained on synthetic image datasets produced by generative deep learning. In particular, we use latent diffusion models conditioned on learned representations from DINOv2, DINOv3, and CLIP. Our results demonstrates that this representation-conditioned formulation significantly outperforms class-conditioned generation by a large margin (+10.76 p.p. top-1 accuracy on ImageNet100), by improving sample quality and mode coverage. Furthermore, by scaling the size of the synthetic dataset, we are able to outperform a classifier trained on the real data (+2.0 p.p top-1 accuracy). We also demonstrate how generated images can be used for augmentation purposes, outperforming classical augmentation methods, and how the conditioning space can be used for sample filtering to further improve training value. Collectively, these findings highlight that representation-conditioned diffusion models provide a promising approach for augmenting, complementing, or potentially replacing real-world datasets in large-scale visual learning tasks.
>
---
#### [new 012] Bridging the Sampling Distribution Shift in Radio Map Estimation: A Trajectory-Aware Paradigm
- **分类: cs.CV**

- **简介: 该论文属于无线传感任务，解决UAV测量中采样分布不匹配的问题。通过引入轨迹感知采样方法，提升无线电图估计的准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.28234](https://arxiv.org/pdf/2605.28234)**

> **作者:** Feng Qiu; Zheng Fang; Shuhang Zhang; Kangjun Liu; Longkun Zou; Jing Liu; Ke Chen
>
> **摘要:** Learning-based radio map estimation (RME) plays a critical role in UAV-assisted wireless sensing, enabling tasks such as coverage prediction and network optimization. Most current methods assume an independently and identically distributed (i.i.d.) training and testing setting based on random sampling. However, practical UAV measurements are collected sequentially along feasible trajectories, resulting in highly structured and spatially correlated patterns. This mismatch introduces a sampling distribution shift that increases the intrinsic difficulty of spatial field recovery and compromises the generalization of models trained under i.i.d. assumptions. To mitigate this issue, we propose a trajectory-aware training paradigm based on Stochastic-Triggered Trajectory-Based Sampling (ST-TBS), which preserves trajectory continuity while introducing sampling variability. Moreover, from a statistical perspective, we show that trajectory-based sampling reduces spatial diversity and increases information redundancy compared to random sampling. Extensive experiments on the RadioMapSeer and SpectrumNet datasets demonstrate that models trained with random sampling suffer significant performance degradation under trajectory-based observations, with RMSE increasing from 0.0391 to 0.2632 on SpectrumNet. Conversely, our proposed ST-TBS method effectively reduces the RMSE to 0.0571. These results highlight the necessity of aligning training and deployment sampling distributions for reliable RME.
>
---
#### [new 013] SAM-Enhanced Segmentation on Road Datasets: Balancing Critical Classes in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于语义分割任务，解决自动驾驶中缺乏像素级标注的问题。通过SAM生成密集标注，提升模型在极端类别不平衡下的表现。**

- **链接: [https://arxiv.org/pdf/2605.28136](https://arxiv.org/pdf/2605.28136)**

> **作者:** Toomas Tahves; Mauro Bellone; Junyi Gu; Raivo Sell
>
> **摘要:** Dense semantic segmentation is essential for autonomous driving, yet many multi-modal datasets lack pixel-level annotations. The Zenseact Open Dataset (ZOD) provides rich multi-sensor data but only bounding-box labels, limiting its use for segmentation research. Our primary contribution is a Segment Anything Model (SAM)-based annotation pipeline that produces dense, pixel-level annotations for ZOD by converting bounding boxes into semantic masks. In this pilot study, we process over 100,000 frames and manually curate a 2,300-frame subset (36% acceptance rate) to establish a reliable baseline. Using these annotations, we evaluate transformer-based CLFT and CNN-based DeepLabV3+ architectures across diverse weather conditions, achieving up to 48.1% mIoU with CLFT-Hybrid. To address extreme class imbalance, where pedestrians, cyclists, and signs constitute less than 1% of pixels, we explore specialized models targeting rare classes. We further validate the pipeline on the Iseauto autonomous-vehicle platform, achieving 77.5% mIoU, and show that SAM-derived representations transfer effectively across sensor configurations via bidirectional transfer learning. All code and annotations are released to support reproducible research.
>
---
#### [new 014] Compositional Text-to-Image Generation Via Region-aware Bimodal Direct Preference Optimization
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂组合提示生成准确图像的问题。通过提出BiDPO框架，结合偏好数据和区域引导，提升模型对复杂文本的生成能力。**

- **链接: [https://arxiv.org/pdf/2605.28615](https://arxiv.org/pdf/2605.28615)**

> **作者:** Zhuohan Liu; Wujian Peng; Yitong Chen; Zuxuan Wu
>
> **摘要:** Despite the rapid progress of text-to-image (T2I) models, generating images that accurately reflect complex compositional prompts (covering attribute bindings, object relationships, counting) still remains challenging. To address this, we propose BiDPO, a framework to enhance T2I model's capability of compositional text-to-image generation. We begin by introducing an carefully designed pipeline to construct a large-scale preference dataset, BiComp, with strictly quality control. Then, we extend Diffusion DPO to jointly optimize image and text preferences, which is shown to greatly effective in improving the models to follow complex text prompt in generation. To further enhance the models for fine-grained alignment, we employ a region-level guidance method to focus on regions relevant to compositional concepts. Experimental results demonstrate that our BiDPO substantially improves compositional fidelity, consistently outperforming prior methods across multiple benchmarks. Our approach highlights the potential of preference-based fine-tuning for complex text-to-image tasks, offering a flexible and scalable alternative to existing techniques.
>
---
#### [new 015] EgoRelight: Egocentric Human Capture and Illumination Recovery for Relightable and Photoreal Avatar Rendering
- **分类: cs.CV**

- **简介: 该论文属于虚拟人像渲染任务，解决从单目HMD中捕获人体动作、生成可重光照的逼真形象并估计环境光照问题。工作包括提出感知模块、神经外观模型和逆渲染过程。**

- **链接: [https://arxiv.org/pdf/2605.28401](https://arxiv.org/pdf/2605.28401)**

> **作者:** Jianchun Chen; Yinda Zhang; Rohit Pandey; Thabo Beeler; Marc Habermann; Christian Theobalt
>
> **摘要:** Mixed Reality (MR) headsets promise a future of immersive telepresence where virtual humans blend indistinguishably into real or virtual surroundings. Achieving this vision requires a method for capturing a user's motion, estimating appearance under novel lighting, and understanding the environment - all from the constrained viewpoint of a head-mounted display (HMD). Existing approaches treat these as isolated problems: they either focus on driving avatars with baked-in lighting or rely on studio setups for relighting. In this paper, we present EgoRelight, a holistic framework for egocentric telepresence that simultaneously captures full-body human performance, synthesizes photorealistic and relightable appearance, and estimates high dynamic range (HDR) environment maps from a single HMD. First, to ensure motion and surface reconstruction, we propose an egocentric perception module that leverages stereo down-facing cameras to extract dense depth maps, which serve as geometric control signals to drive a mesh-based avatar. Second, we introduce a novel neural appearance model that learns to synthesize view-dependent specular and view-independent diffuse shading separately. By employing a specialized ray-sampling strategy, our model generalizes to unseen illumination without relying on restrictive analytical BRDF priors. Third, we enable seamless avatar integration into the physical world via a test-time inverse rendering process, which recovers an HDR environment map by matching the pre-trained avatar's appearance to live egocentric camera observations. We demonstrate our system through a social telepresence application, where remote users are coherently relit according to their physical environment. Extensive experiments show that our components and the integrated system significantly outperform state-of-the-art baselines in geometric accuracy and rendering as well as relighting fidelity.
>
---
#### [new 016] Mags-RL: Wearing Multimodal LLMs a Magnifying Glass via Agentic Reinforcement Learning For Complex Scene Reasoning
- **分类: cs.CV**

- **简介: 该论文提出Mags-RL框架，解决MLLM在复杂场景中视觉推理不足的问题。通过引入强化学习的超分辨率代理，提升细节识别能力，实现更准确的推理。**

- **链接: [https://arxiv.org/pdf/2605.27960](https://arxiv.org/pdf/2605.27960)**

> **作者:** Xuanzhao Dong; Wenhui Zhu; Peijie Qiu; Xiwen Chen; Xiaobing Yu; Xin Li; Zhipeng Wang; Shao Tang; Gen Li; Yujian Xiong; Hao Wang; Yanxi Chen; Prayag Tiwari; Yalin Wang
>
> **摘要:** Despite their popularity and success, Multimodal Large Language Models (MLLMs) often struggle to interpret images accurately, which limits their reasoning capability in complex scenarios (e.g., high object density and complex background clutter). Prior work mainly addresses this limitation by incorporating explicit visual cues like bounding boxes that require extra annotations. In addition, the resulting low-resolution crops often miss fine-grained details that MLLMs require for accurate reasoning. Therefore, we propose Mags-RL, an Agentic Reinforcement Learning (RL) framework that equips MLLMs with an external super-resolution "magnifying glass" agent for high-resolution fine-grained inspection. Specifically, the model performs two-round reasoning: in the first round, it generates an initial rationale and autonomously identifies regions of interest without relying on additional annotations; in the second round, it invokes a super-resolution agent to crop and upscale those regions, then revisits and verifies its earlier reasoning to produce the final answer. We also introduce a novel curriculum learning strategy that enables data-efficient RL training, needing as few as only 40 training samples to achieve reasonable performance. Experiments on VSR, TallyQA, and GQA subsets show its superior performance against recent strong competing methods, demonstrating high-quality reasoning with precise visual grounding. Code and weights will be released soon.
>
---
#### [new 017] FLORO: A Multimodal Geospatial Foundation Model for Ecological Remote Sensing Across Sensors and Scales
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FLORO，一个用于生态遥感的多模态地理基础模型，解决传感器和尺度差异带来的表征迁移问题。通过预训练实现跨任务、跨传感器的稳定性能。**

- **链接: [https://arxiv.org/pdf/2605.28174](https://arxiv.org/pdf/2605.28174)**

> **作者:** Jorge L. Rodriguez; Victor Angulo Morales; Areej Alwahas; Mariana Elias Lara; Fida Mohammad Thoker; Kasper Johansen; Bernard Ghanem; Fernando T. Maestre; Matthew F. McCabe
>
> **备注:** 29 pages, 9 figures
>
> **摘要:** Foundation models offer a promising route to transferable remote sensing representations, but many current approaches depend on very large pretraining datasets and fixed sensor configurations, limiting their suitability for ecological and environmental applications, where observations often vary across platforms, spatial and spectral resolutions, and available modalities. We introduce FLORO, a multimodal geospatial foundation model designed to learn transferable representations from a small but highly diverse remote sensing corpus. FLORO is pretrained using masked autoencoding on a heterogeneous combination of Sentinel-1, Sentinel-2, SkySAT imagery, elevation, and UAV-derived data. To accommodate sensor variability, FLORO incorporates availability-aware inputs that indicate which spectral bands and auxiliary modalities are present in each sample, enabling a unified input space across heterogeneous sensor configurations. We evaluated FLORO on the PANGAEA benchmark under a frozen-encoder protocol across scene classification, segmentation, and regression tasks. Despite being pretrained on a smaller corpus than competing foundation models, FLORO achieved strong and stable transfer across optical, optical-SAR, and optical-elevation benchmarks spanning medium-resolution satellite, airborne, and ultra-high-resolution UAV imagery. FLORO obtained the second-best average segmentation performance across six PANGAEA benchmarks, trailing only a recently introduced foundation model pretrained on over two orders of magnitude more images, remained competitive on scene classification, and was robust in regression tasks, while qualitative results showed improved preservation of spatial structure in flood, urban, biomass, and canopy-height prediction settings. In a separate controlled experiment on EuroSAT-MS, geo-positional encoding further improved classification relative to absolute positional encoding.
>
---
#### [new 018] Beyond Surrogate Gradients: Fully Differentiable Token Pruning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型优化任务，解决冗余视觉令牌带来的计算成本过高问题。提出DiffPrune，通过连续控制实现可微剪枝，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2605.28051](https://arxiv.org/pdf/2605.28051)**

> **作者:** Landi He; Mingde Yao; Shawn Young; Lijian Xu
>
> **摘要:** Visual token pruning reduces the computational cost of Vision-Language Models (VLMs) by removing redundant visual tokens. Existing methods typically rely on Gumbel-Softmax to approximate discrete selection during training. However, the optimization is driven by surrogate gradients rather than the true selection process, leading to unreliable learning of token importance. In this paper, we propose DiffPrune, which reformulates pruning as continuous control of token information instead of discrete selection learning. Specifically, we introduce an Information Throttler that modulates each token using variance-preserving noise conditioned on importance scores, where higher scores induce less information suppression during training. This design directly operates on token representations, naturally providing a fully differentiable optimization path for learning token importance. At inference, tokens are removed via hard thresholding on the learned scores. Across ten VLM benchmarks, DiffPrune retains 96.5% of full-model accuracy while accelerating LLM prefill by 2.85x, with only 0.69 ms of inference overhead.
>
---
#### [new 019] Bound-Constrained Sparse Representation for Electrical Impedance Tomography
- **分类: cs.CV**

- **简介: 该论文属于电气阻抗断层成像（EIT）任务，旨在提高导电性估计的准确性。通过引入约束稀疏表示框架，解决传统方法依赖显式正则化的问题，提升图像质量和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.28392](https://arxiv.org/pdf/2605.28392)**

> **作者:** Chun Zhang; Dong Liu
>
> **摘要:** This study proposes a bound-constrained sparse representation (BC-SR) framework for electrical impedance tomography (EIT), aimed at improving conductivity estimation without explicit regularization. BC-SR adopts a representation-driven strategy, generating conductivity from low-dimensional latent variables via an implicit composite parameterization. Structural priors are embedded using a truncated graph-Laplacian basis, while a bound-preserving nonlinear mapping enforces admissible conductivity ranges and improves conditioning through implicit gradient modulation. The approach ensures robust convergence, even under noisy or incomplete data. Extensive validation on 2D/3D simulations, tank experiments, and in-vivo lung data shows that BC-SR improves physical consistency and structural fidelity, offering enhanced robustness compared to traditional methods. Additionally, BC-SR enables 3D time-difference EIT reconstruction, offering improved spatial resolution and a more coherent representation of 3D conductivity distributions, particularly for in-vivo lung data. This suggests potential for improved performance in EIT, particularly in clinical applications for respiratory monitoring.
>
---
#### [new 020] Mining Multi-Modality Spatio-Temporal Cues for Video Important Person Identification
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出视频重要人物识别任务，解决因忽略时序信息导致的临时重要性偏差问题。构建了Temporal-VIP数据集，并提出VIP-Net框架进行多模态时空特征融合与人物排序。**

- **链接: [https://arxiv.org/pdf/2605.28604](https://arxiv.org/pdf/2605.28604)**

> **作者:** Xiao Wang; Minglei Yang; Bin Yang; Wenke Huang; Zheng Wang; Xin Xu; Mang Ye
>
> **摘要:** Identifying key individuals in video scenes is essential for applications such as automated video editing and intelligent surveillance. Current methods primarily focus on static images and immediate visual cues, overlooking the rich spatio-temporal information in videos. This leads to the phenomenon of Temporal Importance Shift (TIS), wherein individuals deemed significant in early frames may be demoted as the entire temporal context is considered. To address this, we introduce the Video Important Person (VIP) identification task, aimed at automatically identifying the most influential individuals in videos while providing textual rationales. We present Temporal-VIP, a large-scale rationale-annotated dataset consisting of 9,249 video segments across 11 categories with aligned importance rationales. To mitigate TIS, we develop the VIP-Net framework, which includes a Social Cue Encoder (SCE) for extracting multi-modal spatio-temporal cues, a Temporal Importance Rectifier (TIR) for hierarchical cue fusion and cross-modal alignment, and VIP Inference for ranking individuals. Experimental results show that VIP-Net achieves 67.3% accuracy, significantly outperforming state-of-the-art models (37.5%-53.9%) and yielding a mean rationale similarity of 0.63 to ground truth through feature-guided LLM refinement. The dataset and code are available at this https URL.
>
---
#### [new 021] Asynchronous Remote Sensing Time-Series Fusion for Cloud Removal and Anytime Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于遥感时间序列融合任务，旨在解决云覆盖导致的S2数据缺失问题。通过AGFlow模型，实现异步S1/S2数据融合，提升云去除和任意时间点重建效果。**

- **链接: [https://arxiv.org/pdf/2605.27726](https://arxiv.org/pdf/2605.27726)**

> **作者:** Forouzan Fallah; Chia Yu Hsu; Wenwen Li; Anna Liljedahl; Yezhou Yang
>
> **备注:** CVPR 2026 MORSE Workshop
>
> **摘要:** Frequent cloud cover severely limits the usability of Sentinel-2 (S2) optical time series for Earth surface monitoring. Sentinel-1 (S1) SAR provides all-weather complementary observations, but practical S1/S2 fusion remains difficult because acquisitions are irregular and asynchronous. Many existing approaches assume temporally aligned inputs (or require external nearest-date matching) and typically restore only observed timestamps, limiting reconstruction under long gaps and preventing on-demand synthesis. We propose AGFlow (Time Aligned Generative Flow Matching), a spatiotemporal flow-matching model for S1/S2 cloud removal and time-series reconstruction with three capabilities: (1) timestamp-conditioned internal alignment that fuses asynchronous S1 and cloudy S2 observations without preprocessing-based pairing; (2) spatiotemporal, context-aware denoising that models spatial structure jointly with temporal dynamics (rather than independent per-pixel time series); and (3) anytime querying, enabling generation of cloud-free S2 frames at both observed and user-specified timestamps within the monitoring window. We evaluate on the RESTORE-DiT benchmark protocol with quantitative metrics, qualitative comparisons, and component ablations. AGFlow notably improves fully missing-frame reconstruction (MAE and RMSE reduce by 16-19% over RESTORE-DiT) and provides reliable reconstructions under persistent gaps, while also yielding competitive cloud removal performance and flexible temporal querying for downstream tasks such as dense vegetation monitoring.
>
---
#### [new 022] Qwen-Image-Bench: From Generation to Creation in Text-to-Image Evaluation
- **分类: cs.CV**

- **简介: 该论文提出Qwen-Image-Bench，解决文本到图像生成模型评估不足的问题。通过构建多维度评价体系，提升对真实性和创造力的评估能力。**

- **链接: [https://arxiv.org/pdf/2605.28091](https://arxiv.org/pdf/2605.28091)**

> **作者:** Niantong Li; Guangzheng Hu; Weixu Qiao; Ying Ba; Qichen Hong; Shijun Shen; Jinlin Wang; Fan Zhou; Jianye Kang; Xin Shang; Ziyi He; Wei Wang; Dalin Li; Jiahao Li; Jie Zhang; Kaiyuan Gao; Kun Yan; Lihan Jiang; Ningyuan Tang; Shengming Yin; Tianhe Wu; Xiao Xu; Xiaoyue Chen; Yuxiang Chen; Yan Shu; Yanran Zhang; Yilei Chen; Yixian Xu; Zekai Zhang; Zhendong Wang; Zihao Liu; Zikai Zhou; Hongzhu Shi; Yi Wang; Bing Zhao; Hu Wei; Lin Qu; Chenfei Wu
>
> **摘要:** Text-to-Image generation has evolved from basic image synthesis into a frequently used core capability in professional creative workflows, where simple text-image alignment can no longer satisfy users' pressing demands for faithful real-world reconstruction and genuine creative expression. Existing benchmarks, however, remain anchored in these foundational criteria and do not yet capture the nuanced capabilities that matter in authentic artistic practice, making it difficult to reliably distinguish state-of-the-art T2I models. To address the gap, we introduce Qwen-Image-Bench, a creator-centric benchmark co-designed with professional artists and grounded in real-world creation scenarios. Qwen-Image-Bench enriches conventional evaluation with two application-driven dimensions: Real-world Fidelity and Creative Generation. Drawing on the staged reasoning inherent in professional artistic workflows, we organize these five pillars into a top-down hierarchical taxonomy that further decomposes into 23 second-level sub-capabilities and 56 third-level verifiable rubrics. To ensure broad coverage, we curate 1000 stratified prompts with each prompt jointly exercising more than four fine-grained facets across multiple pillars. We train a unified judge model Q-Judger based on Qwen3.6-27B, supervised by 80 professional annotators from global art academies under blind labeling and triple-review protocols, that scores every image across all 56 verifiable facets, producing fine-grained, rubric-grounded, and fully attributable diagnostics rather than a single opaque score. Empirically, Qwen-Image-Bench reliably distinguishes leading T2I models, achieving the greatest separation on the two application-driven dimensions of Real-world Fidelity and Creative Generation where existing benchmarks provide little insight, while also providing a trustworthy optimization signal for production-level T2I development.
>
---
#### [new 023] MangaFlow: An End-to-End Agentic Framework for Controllable Story to Manga Generation
- **分类: cs.CV**

- **简介: 该论文提出MangaFlow，解决漫画生成中布局控制与跨面板一致性问题。通过分解生成流程，实现可控的长篇漫画创作。**

- **链接: [https://arxiv.org/pdf/2605.28173](https://arxiv.org/pdf/2605.28173)**

> **作者:** Muyao Wang; Zeke Xie; Yanhao Chen; Lixin Xiu; Hideki Nakayama
>
> **摘要:** End-to-end manga generation is a structured visual storytelling task that requires story decomposition, recurring character and scene grounding, page layout design, panel rendering, page composition, and lettering. However, existing generative models often perform direct page synthesis, entangling these factors in a single visual output and limiting precise control over layout geometry, visual references, and cross-panel consistency. To address these limitations, we propose MangaFlow, an agentic framework for controllable long-form manga generation that decomposes manga creation into planning, grounding, layout construction, reference-conditioned rendering, composition, and text placement. By treating layout and visual references as explicit intermediate variables, MangaFlow enables both simple text-to-manga generation and more precise user-controlled manga creation. This design exposes layout, visual assets, and lettering as editable intermediate controls for refining panel geometry, references, and text placement. To support long-form consistency, MangaFlow introduces a story section memory that links section descriptions with corresponding character, scene, and object references for reuse across panels. We further present a meta-benchmark for evaluating layout controllability, visual consistency, and generation quality. Experiments show that MangaFlow improves layout adherence and cross-panel consistency over direct generation baselines while supporting flexible human control.
>
---
#### [new 024] JECA^2: Judgment-Explanation Consistent Adversarial Attack against Forensic Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于对抗攻击任务，旨在解决伪造视觉-语言模型判断与解释不一致的问题。提出JECA^2方法，同时调整视觉归因和文本解释，实现判断与解释的一致性攻击。**

- **链接: [https://arxiv.org/pdf/2605.28609](https://arxiv.org/pdf/2605.28609)**

> **作者:** Jiachen Qian
>
> **备注:** 37 pages, 6 figures. Includes supplementary material
>
> **摘要:** Forensic vision-language models (VLMs) have recently been developed to detect image tampering and provide natural-language explanations. However, their robustness against adversarial manipulation remains underexplored. Existing adversarial attacks typically aim to flip the model's binary judgment, while the accompanying explanation may still reveal forensic cues and contradict the attacked judgment. In this paper, we study judgment-explanation consistent adversarial attacks against forensic VLMs and propose JECA^2, a controlled white-box red-team diagnostic that jointly redirects visual attribution and aligns textual explanations with the target judgment. On the visual side, JECA^2 uses Grad-CAM-guided perturbations to divert attribution from tampered regions toward benign regions. On the textual side, it optimizes prompt embeddings toward authenticity-affirming semantics under a token-proximity constraint. Experiments on forensic VLM benchmarks show that JECA^2 achieves higher attack success and automated judgment-explanation consistency than implemented baselines under white-box threat settings, while transfer to closed-source VLMs remains measurable but limited. Our results highlight a consistency failure mode in explanation-based forensic VLMs and motivate future robustness evaluation beyond binary detection accuracy.
>
---
#### [new 025] Tensor Memory: Fixed-Size Recurrent State for Long-Horizon Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Tensor Memory，解决长序列Transformer中记忆容量受限和缺乏显式空间状态的问题，通过固定大小的3D记忆张量提升视频理解能力。**

- **链接: [https://arxiv.org/pdf/2605.27686](https://arxiv.org/pdf/2605.27686)**

> **作者:** Kabir Swain; Sijie Han; Daniel Karl I. Weidele; Mauro Martino; Antonio Torralba
>
> **摘要:** Transformers process images and videos by flattening space and time into long token sequences. While attention and KV caching preserve past features, their memory grows with sequence length and they lack an explicit, persistent spatial state, making long-horizon video understanding and occlusion-sensitive reasoning difficult. We propose Tensor Memory, a lightweight module that augments Transformer blocks with a fixed-size recurrent 3D memory tensor: tokens write into a voxel grid via a differentiable soft write that deposits content as a Gaussian-weighted volume around a predicted continuous 3D location, the memory is updated with an efficient local interaction operator and gated recurrent dynamics, and tokens read back context via continuous sampling with gated residual fusion. Because the memory tensor has a constant size, Tensor Memory decouples state capacity from input length while preserving a spatial inductive bias. We evaluate the module on standard language, image, and video benchmarks and on a controlled toy diagnostic suite designed to isolate when persistent state is beneficial; it integrates with standard Transformer training pipelines and can be attached to or removed from existing blocks without other architectural changes.
>
---
#### [new 026] VidPrism: Heterogeneous Mixture of Experts for Image-to-Video Transfer
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，旨在解决图像到视频的迁移学习问题。提出VidPrism框架，通过异构专家系统提升时空特征学习效率。**

- **链接: [https://arxiv.org/pdf/2605.28229](https://arxiv.org/pdf/2605.28229)**

> **作者:** Rui Lin; Chuanming Wang; Huadong Ma
>
> **备注:** CVPR2026 camera ready
>
> **摘要:** With the rapid development of pre-training technologies, adapting large-scale Vision-Language Models (VLMs) for video understanding \emph{\ie} image-to-video transfer learning has become a dominant paradigm. To achieve superior performance, it raises as an effective strategy among recent advances to employ Mixture-of-Experts (MoE) to enhance VLMs' temporal modeling capabilities. However, conventional MoE designs suffer from expert homogenization, where all experts act as identical generalists, inefficiently learning spatio-temporal features from undifferentiated video streams. To overcome this problem, we propose VidPrism, a novel heterogeneous temporal Mixture-of-Experts framework. VidPrism pioneers a division of labor by deploying functionally specialized experts, each assuming a role ranging from spatial understanding to temporal modeling. To feed these specialists appropriately, we introduce a content-aware, multi-rate sampling module that dynamically generates streams ranging from semantically rich to motion-focused representations, providing specialized inputs for experts. Furthermore, a dynamic, bidirectional fusion mechanism enables synergistic information exchange between these pathways, leading to a comprehensive video representation. Extensive experiments on various video recognition benchmarks demonstrate that VidPrism achieves state-of-the-art performance and effectively fosters expert specialization. Our source code is available at \href{this https URL}{this https URL}.
>
---
#### [new 027] D$^2$Turb: Depth-Aware Simulation and Decoupled Learning for Single-Frame Atmospheric Turbulence Mitigation
- **分类: cs.CV**

- **简介: 该论文属于单帧大气湍流校正任务，解决模糊与几何失真耦合问题。提出D$^2$Turb框架，通过深度感知仿真和解耦恢复实现纹理去模糊与几何校正。**

- **链接: [https://arxiv.org/pdf/2605.27460](https://arxiv.org/pdf/2605.27460)**

> **作者:** Zixiao Hu; Tianyu Li; Guoqing Wang; Wei Li; Guoguo Xin; Xun Liu; Peng Wang
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Single-frame atmospheric turbulence mitigation is inherently ill-posed due to spatially varying blur coupled with non-rigid geometric distortion. Existing end-to-end approaches trained on flat-field simulations often struggle to balance texture recovery with geometric rectification. To overcome this limitation, we propose D$^2$Turb, a unified framework that bridges physics-grounded simulation with explicitly decoupled restoration. First, we introduce a Depth-Aware Turbulence Synthesis protocol that incorporates scene depth into the phase-to-space formulation. This generates physically consistent, depth-dependent degradations and provides a crucial intermediate tilt supervision signal for disentangled learning. Building upon this simulation engine, D$^2$Turb decomposes restoration into two interactive stages: texture deblurring and geometric rectification. The texture deblurring stage employs a deblurring backbone to recover fine-grained details while preserving geometric distortion for the subsequent rectification stage. To mitigate the information fragmentation commonly observed in cascaded designs, we further propose an Adaptive Structural Prior Injection (ASPI) mechanism that dynamically transfers deep structural representations from the deblurring module to guide dense flow prediction for spatial unwarping. Extensive experiments demonstrate that D$^2$Turb achieves state-of-the-art performance on both synthetic and real-world datasets, with consistent improvements in both texture recovery and geometric fidelity. Our code and pre-trained models are publicly available at this https URL.
>
---
#### [new 028] Evaluating the Feasibility of Inferring Dietary Behavior Change Receptivity from Egocentric Images of Eating Environment
- **分类: cs.CV**

- **简介: 该论文属于行为识别任务，旨在通过吃东西的自中心图像推断饮食行为改变的接受度。研究使用视觉信息和模型分析，以解决自我报告评估不及时的问题。**

- **链接: [https://arxiv.org/pdf/2605.27950](https://arxiv.org/pdf/2605.27950)**

> **作者:** Long Li; Yuning Huang; Heather A. Eicher-Miller; J.Graham Thomas; Fengqing Zhu; Edward Sazonov
>
> **摘要:** Accurately assessing dietary behavior change receptivity is essential for designing effective just-in-time adaptive interventions (JITAIs) that promote healthier eating habits. However, self-report-based assessment of behavior change receptivity is sparse and delayed, limiting its practical use in continuous monitoring. To explore whether passive sensing may help address this challenge, this study conducts a pilot investigation of inferring participants' self-reported behavior change receptivity from egocentric eating images collected by a wearable camera. We use pilot data obtained from free-living eating episodes using the Automatic Ingestion Monitor v2 (AIM-2). The data included egocentric image sequences captured during eating and paired with responses to questions assessing specific dimensions of behavior change receptivity (awareness, interaction capability, and motivation). To examine whether visual information contained any relevancy to these responses, we evaluated a transfer-learning-assisted framework that combines a pre-trained Contrastive Language-Image Pre-Training (CLIP) vision encoder with a lightweight transformer classifier. The model processes eating episode image sequences to extract potential semantic and temporal cues related to behavior change receptivity. Preliminary experimental results show promising improvements over simple baseline models for behavior change receptivity indicators. These early findings suggest that egocentric eating episode images may contain cues related to dietary behavior change receptivity, and warrant further investigation with larger and more comprehensive datasets.
>
---
#### [new 029] Not All NVFP4 QAT Recipes Are Equal: How Architecture and Scale Shape Model Quality for Anomaly Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究FP4量化感知训练（QAT）对异常分割任务的影响，分析架构与规模对模型质量的作用，提出Swin Transformer在不同规模下均具鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.27616](https://arxiv.org/pdf/2605.27616)**

> **作者:** Zijian Du; Oleg Rybakov
>
> **摘要:** Real-time anomaly segmentation demands both high recall and efficient low-precision inference. We study the three-way interaction of model architecture, model scale, and FP4 quantization-aware training (QAT) recipe on a recall-critical brain tumor segmentation task, evaluating multiple architectures, scales, and QAT recipes under a unified protocol. We find that architecture choice has the largest impact on quantization robustness, with attention-based architectures showing remarkable resilience to recipe choice while CNN degrades under gradient-quantizing recipes at larger scales. At low capacity, FP4 can discretize softmax attention, but advanced QAT recipes prevent this collapse. At larger scales, advanced recipes mitigate gradient quantization noise that degrades CNN quality. Five-fold patient-level cross-validation confirms these findings are robust to data partition. Our results show that the Swin Transformer is robust to QAT recipe choice across all scales, making it the recommended architecture for FP4-quantized anomaly segmentation.
>
---
#### [new 030] Category-Level 3D Correspondence in Camera Space via Morphable Object Priors
- **分类: cs.CV**

- **简介: 该论文属于3D物体对应任务，旨在通过学习形态先验实现无监督的类别级3D对应。工作包括构建数据集HouseCorr3D和提出Morpheus方法。**

- **链接: [https://arxiv.org/pdf/2605.28257](https://arxiv.org/pdf/2605.28257)**

> **作者:** Leonhard Sommer; Artur Jesslen; Basavaraj Sunagad; Adam Kortylewski
>
> **备注:** 14 pages, 4 figures. Data and code are publicly available at this https URL
>
> **摘要:** Understanding 3D objects from images is fundamental to robotics and AR/VR applications. While recent work has made progress in category-level pose estimation, current representations fail to capture the fine-grained semantics needed for reasoning about object parts, functions, and interactions. In this work, we study category-level 3D correspondence in camera space -- predicting, from a single image, 3D locations that remain consistent across instances within a category -- and show that it can emerge without explicit correspondence supervision by learning a shared morphable object prior. To enable research in this direction, we introduce HouseCorr3D, the first large-scale benchmark for monocular category-level 3D correspondence with 178k images across 50 household object categories, 280 unique instances, and 3D keypoint annotations directly on CAD models. Crucially, HouseCorr3D provides amodal correspondence labels for occluded regions and explicit symmetry annotations, addressing key limitations of existing datasets. We further propose Morpheus, a method that learns morphable category-level shape priors by disentangling canonical shape, deformation, and object pose. Through this shared canonical grounding, semantically meaningful 3D correspondences in camera space emerge implicitly. These emerging 3D correspondences set a new state of the art on HouseCorr3D, demonstrating that semantic 3D object understanding can arise without direct correspondence supervision. Data and code are publicly available at this https URL.
>
---
#### [new 031] CuriosAI Submission to the CASTLE Challenge at EgoVis 2026
- **分类: cs.CV**

- **简介: 该论文针对CASTLE挑战任务，解决多视角自指视频中的多项选择问题。通过两种方法处理视频数据，提升问答准确率。**

- **链接: [https://arxiv.org/pdf/2605.27800](https://arxiv.org/pdf/2605.27800)**

> **作者:** Yuto Kanda; Hayato Tanoue; Takayuki Hori
>
> **备注:** The 4th place solution for the CASTLE Challenge at the CVPR EgoVis Workshop 2026
>
> **摘要:** CASTLE 2026 asks 185 multiple-choice questions over 600+ hours of synchronized multi-view egocentric video. We explore two approaches on top of a shared multimodal preprocessing layer, including per-person timelines, speaker-resolved transcripts, and multi-VLM caption ensembles. Approach A, SVA: Search-Verify-Answer, is a three-stage pipeline that hierarchically narrows to a primary window, verifies sub-windows with a VLM under four anti-confabulation rules, and fuses evidence with an LLM judge under an evidence-priority hierarchy. Approach B, TMKG: Temporal-Multimodal-Knowledge-Graph, is the contrast: it builds a temporal multimodal knowledge graph, locates a primary cell via graph search, and produces the final answer with a single grounded VLM. SVA reaches a leaderboard accuracy of 0.50 and is our final challenge submission; TMKG reaches 0.35.
>
---
#### [new 032] SeeGroup: Multi-Layer Depth Estimation of Transparent Surfaces via Self-Determined Grouping
- **分类: cs.CV**

- **简介: 该论文属于多层深度估计任务，旨在解决透明物体多层深度预测问题。提出SeeGroup方法，通过自适应分组避免预定义策略，提升准确率。**

- **链接: [https://arxiv.org/pdf/2605.28735](https://arxiv.org/pdf/2605.28735)**

> **作者:** Hongyu Wen; Jia Deng
>
> **摘要:** Transparent objects are common in daily life, and it is important to understand their multilayer depth, including the transparent surface and the objects behind it. Existing methods for multilayer depth typically extend single-layer prediction. They define layers by the front-to-back ordering of 3D points and predict the layers sequentially. However, as layered geometry can admit multiple valid groupings of 3D points into layers, a predefined grouping strategy is inherently restrictive. In this work, we propose SeeGroup, a multi-layer depth estimation method that avoids imposing a predefined grouping and allows the model itself to adaptively assign surfaces to depth maps. We formulate per-pixel multi-layer depth as a point process, treating depth layers as unordered events along each camera ray. This induces a permutation-invariant likelihood over the observed depth layers, yielding a loss that naturally supports arbitrary layer groupings. Experiments demonstrate that our method significantly advances the state of the art of multi-layer depth estimation, improving quadruplet relative depth accuracy on LayeredDepth benchmark from 61.34% to 70.09%. Code is available at this https URL.
>
---
#### [new 033] VITAL: Visual-Semantic Dual Supervision for Enhanced and Interpretable Latent Reasoning in Medical MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗多模态大模型任务，解决latent reasoning的可解释性和效率问题。提出VITAL框架，通过视觉-语义双监督提升推理效果并实现解释性。**

- **链接: [https://arxiv.org/pdf/2605.28422](https://arxiv.org/pdf/2605.28422)**

> **作者:** Qiaoru Li; Shaotian Liang; Jintao Chen; Haoran Sun; Yuxiang Cai; Jianwei Yin; Yankai Jiang
>
> **摘要:** Latent reasoning enables reasoning over continuous hidden states rather than explicit tokens, avoiding the language bottleneck and inference overhead of chain-of-thought for medical VQA. However, existing methods suffer from modality collapse, insufficient visual supervision, and train-inference mismatch. Moreover, their opaque latent states offer no interpretability, which is critical in clinical applications. We propose VITAL, a latent-space reasoning framework for medical MLLMs with visual-semantic dual supervision: an auxiliary text decoder reconstructs reasoning chains from latent states, while a visual projector regresses ROI features from a frozen, independent medical vision encoder. Both modules are discarded at inference with zero overhead, yet can be re-attached post-hoc for dual interpretability, providing textual and visual explanations of the reasoning process without sacrificing efficiency. We construct a 61K dataset spanning 9 imaging modalities, exceeding prior medical visual latent reasoning datasets by an order of magnitude. Experiments on 7 benchmarks show that VITAL consistently and substantially outperforms the backbone, all latent reasoning baselines, and medical MLLMs trained on far larger data, achieving state-of-the-art results competitive with trillion-parameter proprietary models.
>
---
#### [new 034] Toward Semantic-Agnostic and Shape-Aware Vision-Language Segmentation Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言分割任务，旨在解决模型依赖语义类别而忽视形状等视觉属性的问题。提出SANSA方法，通过非语义文本提示微调模型，提升视觉推理能力。**

- **链接: [https://arxiv.org/pdf/2605.28348](https://arxiv.org/pdf/2605.28348)**

> **作者:** Corentin Seutin; Mohamed Amine Ettaki; Michaël Clément; Pierrick Coupé; Rémi Giraud
>
> **备注:** Accepted at the 2026 IEEE International Conference on Image Processing (ICIP 2026)
>
> **摘要:** Vision-language segmentation models have recently achieved strong performance by leveraging high-level semantic object categories expressed in natural language. However, this semantic dependence limits their ability to reason about intrinsic visual properties such as shape, geometry, or texture, which are essential in many real-world applications. In this work, we introduce Semantic-Agnostic aNd Shape-Aware (SANSA) segmentation, a new paradigm that requires segmentation models to operate solely from non-semantic textual descriptions. To this end, we propose two strategies to generate SANSA segmentation prompts based on either dictionary constraints or example guidance, both generating semantic-agnostic textual descriptions. These prompts are then used to finetune segmentation models under semantic-agnostic supervision. Experiments show that finetuning on SANSA prompts yields up to a 20% mIoU improvement on this new segmentation task, compared to pretrained state-of-the-art models, while maintaining strong performance on standard semantic prompts. These results highlight the importance of low- and mid-level visual reasoning for improving the generalization and controllability of vision-language segmentation models.
>
---
#### [new 035] Learning to Label: A Reinforced Self-Evolving Framework for Semi-supervised Referring Expression Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督指代表达分割任务，解决标注数据有限和伪标签不可靠的问题。提出L2L框架，通过强化学习优化伪标签和分割模型。**

- **链接: [https://arxiv.org/pdf/2605.28239](https://arxiv.org/pdf/2605.28239)**

> **作者:** Runlong Cao; Ying Zang; Chuanwei Zhou; Tianrun Chen; Tong Zhang; Zhen Cui; Chunyan Xu
>
> **备注:** 24 pages, 13 figures
>
> **摘要:** Semi-supervised referring expression segmentation (SS-RES) aims to achieve precise pixel-level language grounding under limited annotation, yet suffers from limited supervision and unreliable pseudo-labels when exploiting unlabeled image-text pairs. In this work, we propose Learning to Label, a reinforced self-evolving framework (L2L) that casts pseudo-label construction as a learnable decision-making process. To build foundational understanding, we leverage a multimodal large language model to extract semantic-spatial priors, which are instantiated as initial soft segmentation proposals and elevated, together with textual cues, into learnable guidance signals that condition a hierarchical segmentation network. To ensure stable learning, reinforced pseudo-label selection is formulated as an exploratory decision process that adaptively rewards high-utility pixel-level supervision based on multimodal priors and model predictions. This reinforced self-evolving loop enables joint optimization of the segmentation model and pseudo-labels, progressively enhancing label reliability under sparse supervision. Extensive experiments on RefCOCO, RefCOCO+, and RefCOCOg demonstrate improvements over existing methods, validating its effectiveness and generalization.
>
---
#### [new 036] Intra-YOLO: A Small Object Detection Model for Caries and Molar-Incisor Hypomineralization in Intraoral Photography Based on Transfer Learning with Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像识别任务，旨在解决口腔内小病灶（龋齿和牙釉质发育不全）的检测问题，采用迁移学习与强化学习方法提升检测精度。**

- **链接: [https://arxiv.org/pdf/2605.28157](https://arxiv.org/pdf/2605.28157)**

> **作者:** Po-Lun Chwang; Po-Yu Chang; Wen-Liang Lin; Tung-Sheng Wu; Min-Ching Wang; Yun-Chien Cheng
>
> **摘要:** This study developed a computer-aided diagnosis (CAD) system for detecting caries and molar-incisor hypomineralization (MIH) in intraoral photographs. These lesions share similar appearances, making clinical differentiation challenging, especially given their small size and variability in imaging conditions.
>
---
#### [new 037] MORI-Seg: Learning Morphological Geometry for Instance Segmentation without Instance Annotations
- **分类: cs.CV**

- **简介: 该论文属于实例分割任务，解决无实例标注时的分割问题。通过学习形态几何表示，从语义掩码中分解出独立实例，提升分割精度和定量分析可靠性。**

- **链接: [https://arxiv.org/pdf/2605.28261](https://arxiv.org/pdf/2605.28261)**

> **作者:** Leiyue Zhao; Tianyu Shi; Daniel Reisenbuchler; Xinzi He; Junchao Zhu; Tianyuan Yao; Yuechen Yang; Yanfan Zhu; Junlin Guo; Gelei Xu; Haichun Yang; Yuankai Huo; Mert R. Sabuncu; Yihe Yang; Ruining Deng
>
> **摘要:** Instance-level quantification of kidney functional units is essential for morphometric analysis, yet most publicly available pathology datasets provide only semantic segmentation annotations, where adjacent structures of the same class are merged into single regions. This prevents reliable instance-level analysis and limits downstream quantitative studies. Existing heuristic post-processing methods often yield suboptimal instance separation, particularly in crowded and adherent regions, while deep learning-based instance segmentation approaches typically require intensive instance-level annotations that are costly and labor-intensive to obtain. We propose MORI-Seg, a deep learning framework that enables instance segmentation without requiring instance-level annotations. Instead of heuristic splitting or instance supervision, MORI-Seg learns morphology-aware geometric representations directly from semantic masks by jointly modeling object-centric distance fields and boundary-band representations to encode interior structure and contact interfaces. A class-conditioned feature disentanglement module further promotes intra-instance coherence and inter-instance separation. Under semantic-only supervision, MORI-Seg decomposes connected semantic regions into distinct instance masks in an end-to-end manner. Experiments demonstrate improved instance separation accuracy and more reliable morphometric quantification compared with classical post-processing pipelines and representative semantic-to-instance learning approaches. The official implementation is publicly available at this https URL.
>
---
#### [new 038] SEMAGIC: Learning Semantically Consistent Deformable 3D Representations from In-the-Wild Images
- **分类: cs.CV**

- **简介: 该论文提出SEMAGIC，解决单视角图像中学习语义一致的可变形3D表示问题。通过引入语义一致性损失和顶点条件变形，提升模型在语义对应任务上的表现。**

- **链接: [https://arxiv.org/pdf/2605.27938](https://arxiv.org/pdf/2605.27938)**

> **作者:** Sky Cen; Wufei Ma; Guofeng Zhang; Alan Yuille; Adam Kortylewski
>
> **摘要:** Learning deformable 3D object models from single-view in-the-wild images has enabled impressive 3D shape reconstruction without supervision. However, it remains unclear whether these models capture the semantic structure required for downstream tasks. We find that existing deformable reconstruction approaches, despite producing visually plausible geometry, yield unstable correspondences across instances and perform poorly on semantic correspondence benchmarks. We introduce SEMAGIC, a framework for learning semantically consistent deformable 3D representations from single-view in-the-wild images. Rather than treating reconstruction as the end goal, SEMAGIC uses deformable modeling as a mechanism to discover category-level correspondences. Each category is represented by a canonical template mesh and a learned deformation field, functioning similarly to an autoencoder that reconstructs instance geometry from image features, enabling vertices to maintain consistent semantic meaning across instances. Semantic consistency is enforced during training through (i) a feature-level consistency loss aligning semantic features between canonical and deformed meshes, and (ii) vertex-index-conditioned deformation that preserves semantic correspondence across instances. By explicitly coupling geometric deformation with semantic alignment, SEMAGIC produces representations that maintain stable part correspondences across intra-category variation. Experiments demonstrate that SEMAGIC improves semantic correspondence of deformable models by +14.7 PCK@0.1 on SPair-71k, establishing deformable models as effective semantic 3D representations.
>
---
#### [new 039] Automated Estimation of Impact Time, Impact Location, and Shuttlecock Speed in Badminton Smashes Using Event Cameras
- **分类: cs.CV**

- **简介: 该论文属于运动分析任务，旨在解决badminton smash性能评估问题。通过事件相机自动估计击球时间、位置和球速，提升测量效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.28011](https://arxiv.org/pdf/2605.28011)**

> **作者:** Yudai Washida; Yuto Kase; Kai Ishibe; Ryoma Yasuda; Sakiko Hashimoto
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Quantifying impact phenomena in badminton smashes is important for evaluating both athletic performance and equipment; however, conventional measurement systems involve trade-offs between temporal resolution, data efficiency, and preparation effort. This study proposes a measurement method using two synchronized event cameras to automatically estimate impact time, impact location on the racket face, and post-impact shuttlecock speed in an integrated manner within the same trial. The swing interval was detected from event rate statistics, impact time was estimated from the shuttlecock trajectory inflection in the lateral-view event data, impact location was determined by ellipse fitting to the racket face in the rear-view event image, and shuttlecock speed was calculated in the sagittal plane. To validate the proposed method, Bland-Altman analysis was performed against a high-speed camera-based reference method using 125 smash trials from five players. Impact time and shuttlecock speed were estimated in all 124 analyzable trials, and impact location was estimated in 93.5% (116/124). The bias (95% CI) for impact time, medio-lateral impact location, longitudinal impact location, and shuttlecock speed were 1.84 ms (1.45 to 2.23), 3.45 mm (2.18 to 4.72), -1.92 mm (-2.97 to -0.88), and -1.00 m/s (-2.46 to 0.46), respectively. No proportional bias was observed for any metric. These results suggest that the proposed method can serve as a useful tool for integrated assessment of badminton smash performance and equipment in practical settings.
>
---
#### [new 040] Structure over Pixels: Learning Variable-Length Visual Programs
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出STROP，一种学习可变长度视觉程序的离散视觉分词器，解决结构描述与序列长度自适应问题，通过优化长度头提升场景结构表示。**

- **链接: [https://arxiv.org/pdf/2605.27696](https://arxiv.org/pdf/2605.27696)**

> **作者:** Piotr Wyrwiński; Kacper Dobek; Krzysztof Krawiec
>
> **摘要:** Discrete visual tokenizers translate images into ordered sequences of codes, providing a natural representation for structural description of scenes. Yet existing adaptive tokenizers either require post-hoc search or select among a discrete set of pre-trained rates, rather than learning a continuous per-image sequence length coupled to the model and scene, and they typically train against pixel reconstruction, emphasizing texture rather than structure. We propose STROP, a discrete visual tokenizer architecture that forms structural scene representations and simultaneously learns how long an image's visual program should be. Using a four-phase curriculum supervised by local rate--distortion probes against frozen DINOv3 features, STROP optimizes a dedicated length head that estimates the active prefix length in a single forward pass. By bypassing pixel-level reconstruction gradients, the codebook is shaped entirely by the quality of higher-level latent representations. Program length grows with scene complexity, and signs of compositional structure emerge both in downstream dense-prediction transfer and in direct inspection of the learned code vocabulary.
>
---
#### [new 041] Dual-branch Distilled Transformer for Efficient Asymmetric UAV Tracking
- **分类: cs.CV**

- **简介: 该论文属于无人机目标跟踪任务，旨在解决轻量化模型性能下降的问题。通过双分支知识蒸馏策略提升学生模型的特征表达和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.28018](https://arxiv.org/pdf/2605.28018)**

> **作者:** Hongtao Yang; Bineng Zhong; Qihua Liang; Yaozong Zheng; Xiantao Hu; Yuanliang Xue; Shuxiang Song
>
> **备注:** CVPR2026 Highlight
>
> **摘要:** Given the real-time demands of UAV tracking, many methods simplify the backbone to reduce computation, but this often weakens feature representation and degrades performance in complex scenarios. To alleviate this issue, we propose EATrack, an efficient and asymmetric UAV tracking framework centered around a teacher-guided dual-branch distillation strategy that enhances the feature expressiveness of the lightweight student model. Specifically, EATrack investigates two complementary perspectives of knowledge transfer: spatially focused feature-level distillation that compensates for weakened representations by guiding the student to learn strong target representations, and prediction-level distillation that enhances spatial localization by learning the teacher's capability for accurate target localization. Furthermore, to enhance robustness against appearance variations, we introduce a fine-grained target-aware distillation strategy that selectively transfers the teacher's target modeling capacity to the student. A temporal adaptation module is incorporated at inference to enhance robustness over time. Experiments on five UAV benchmarks demonstrate that EATrack achieves a favorable balance between accuracy and speed. Code: this https URL
>
---
#### [new 042] HarmoVid: Relightful Video Portrait Harmonization
- **分类: cs.CV**

- **简介: 该论文属于视频人像光照调和任务，解决视频光照不一致问题。通过引入去闪烁模型和不对称掩码技术，提升视频的时序一致性与光照自然性。**

- **链接: [https://arxiv.org/pdf/2605.28811](https://arxiv.org/pdf/2605.28811)**

> **作者:** Jun Myeong Choi; Jae Shin Yoon; Luchao Qi; Roni Sengupta; Joon-Young Lee
>
> **备注:** CVPR 2026
>
> **摘要:** We present a method for harmonizing the lighting of a foreground video to match a target background scene, adjusting shadows, color tone, and illumination intensity (relightful harmonization). Unlike images, acquiring labeled data for videos, where identical motions are recorded under different lighting conditions, is practically infeasible and non-scalable. While one way to create such paired data is to apply existing image-based harmonization models frame by frame to a video, the resulting outputs often suffer from significant temporal jitters. We overcome this problem by introducing a novel lighting deflickering model that can stabilize the global and local lighting flickering artifacts. Our video diffusion model learns from these upgraded deflickered data with a volume of real and synthetic videos to generate high-quality video harmonization results. We further propose an asymmetric alpha mask conditioning technique to learn the clean boundaries from real videos. Experiments demonstrate that our model achieves strong temporal coherence, naturalness, cleaner boundaries, and physically meaningful lighting behavior, while maintaining strong relighting expressiveness compared to prior image-based and video-based harmonization methods.
>
---
#### [new 043] Rethinking Video-Language Model from the Language Input Perspective
- **分类: cs.CV**

- **简介: 该论文属于视频-文本对齐任务，解决传统VLM依赖预定义文本的问题。通过生成正负文本、属性推理和自加权损失，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27920](https://arxiv.org/pdf/2605.27920)**

> **作者:** Xiang Fang; Wanlong Fang; Changshuo Wang; Xiaoye Qu; Daizong Liu
>
> **备注:** Published in AAAI 2026
>
> **摘要:** Driven by the wave of large language models, Video-Language Models (VLMs) have become a significant yet challenging technology to bridge the gap between videos and texts. Although previous VLM works have made significant progress, almost all of them implicitly assume that all the texts are predefined by the specific template. In real-world applications, such a strict assumption is impossible to satisfy since 1) predefining all the texts is extremely time-consuming and labor-intensive. 2) these predefined text inputs are too restrictive and user-unfriendly, limiting their applications. It is observed that given a video input, texts with similar semantics but different templates lead to various performances. To this end, in this paper, we propose a novel plug-and-play framework for various VLM-based methods to fully bridge videos and texts. Specifically, we first generate positive and negative texts from the original ones to target specific text components. Then, we propose an attribute-based text reasoning strategy to mine fine-grained textual semantics of generated texts. Finally, we utilize videos as guidance to conduct cross-modal bridging by designing a self-weighted loss. Extensive experiments show that the proposed method can serve as the plug-and-play module to effectively improve the performance of state-of-the-art VLMs.
>
---
#### [new 044] Bounded-Compute Multimodal Regression for Product-Rating Prediction
- **分类: cs.CV**

- **简介: 该论文属于多模态回归任务，旨在解决产品评分预测问题。针对严格时延限制，提出一种有限计算量的模型，采用特征回归替代文本生成，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2605.27737](https://arxiv.org/pdf/2605.27737)**

> **作者:** William Leach; Ru He; Sizhuo Ma; Yizhen Jia; Min Cao; Jian Wang; Rick Cao
>
> **备注:** Accepted to the LoViF Workshop at CVPR 2026. 8 pages, 2 figures
>
> **摘要:** Vision-language models (VLMs) are increasingly attractive for multimodal quality assessment, but their default reliance on autoregressive text generation and dynamic visual processing is poorly matched to scalar regression under strict latency budgets. We present a bounded-compute adaptation of SmolVLM2-256M-Video-Instruct for product-rating prediction in the LoViF 2026 Efficient VLM challenge. Motivated by recent multimodal engagement-prediction results showing that feature-based regression can outperform token-based score generation, we replace the language-modeling head with a lightweight two-layer MLP fed by pooled decoder states, and we enforce deterministic inputs through fixed 384x384 images and truncated metadata. Across controlled ablations, static global image processing slightly outperforms dynamic tiling, and scaling from 100K to 16M training examples substantially improves validation correlation. Under the official held-out evaluation, our 228M-parameter model achieves 0.39 PLCC and 0.40 CES, providing a strong and reproducible baseline for resource-constrained multimodal regression.
>
---
#### [new 045] What-If World: A Causal Benchmark for General World Models in Embodied Scenarios
- **分类: cs.CV**

- **简介: 该论文属于因果推理任务，旨在评估世界模型在具身场景中的物理一致性。通过构建319个提示对，测试模型对物理变量变化的响应能力，发现现有模型表现不佳，表明其在支持动作条件模拟方面仍有不足。**

- **链接: [https://arxiv.org/pdf/2605.27589](https://arxiv.org/pdf/2605.27589)**

> **作者:** Kunlin Cai; Rui Song; Jinghuai Zhang; Kaiyuan Zhang; Pranav Bodapati; Alicia Yu; Fnu Suya; Mohammad Rostami; Jiaqi Ma; Yuan Tian
>
> **备注:** 38 pages, World Model Benchmark
>
> **摘要:** Video generation models are increasingly used as world simulators for tasks like driving and robotic manipulation. What matters in these settings is not whether a single video looks right, but whether the model's output changes when its input changes. We test this by giving a model two prompts describing the same scene with one physical detail varied, and checking whether the two videos diverge the way physics predicts. The wording difference between the prompts is small by design, since only one variable is changed, but the correct physical difference is not. A model that misses this can still produce two videos that each look plausible individually, and existing benchmarks score videos one at a time and cannot detect this failure. We introduce What-If World, 319 such prompt pairs built on real frames from nuScenes and DROID, organized by a taxonomy of six physical variables shared across driving and manipulation. Each pair is scored with APEO, a four-part rubric checking whether each video follows its prompt (Adherence), is physically consistent (Physics), preserves the shared scene (Environment), and ends in the correct difference (Outcome). Across nine state-of-the-art models, no system exceeds 52% on the paired score, and open-source models cluster near 28%. Every model tested fails on a large fraction of causal interventions, indicating substantial room before these models can reliably support action-conditioned simulation or model-based planning. Where models do score well, performance appears to track the visual prominence of the intervention rather than the tractability of its underlying physics. Some visually subtle interventions score as low as 14.2%, while visually pronounced ones reach 40.4%.
>
---
#### [new 046] Internally Referenced Low-Light Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光照图像增强任务，旨在解决自监督方法中缺乏外部参考导致的光照、纹理和噪声难以解耦的问题。通过内部参考机制提升图像质量。**

- **链接: [https://arxiv.org/pdf/2605.28605](https://arxiv.org/pdf/2605.28605)**

> **作者:** Peiyuan He; Hainuo Wang; Hengxing Liu; Mingjia Li; Xiaojie Guo
>
> **摘要:** Self-supervised low-light image enhancement (LLIE) is highly appealing as it eliminates the reliance on external paired data. However, the lack of external references causes networks to struggle with decoupling entangled illumination, delicate textures, and amplified noise. To resolve this challenge, we propose an Internally Referenced LLIE framework that extracts reliable physical and structural references from the degraded input image itself. First, we introduce a local exposure-simulated scheme to extract a low-frequency pseudo ground-truth. This serves as an internal physical reference to guide global illumination estimation and correct color casts. Second, we propose a dual-domain preservation strategy with spatial and spectral constraints to construct internal structural references. Specifically, an Illumination-Aligned Perceptual loss preserves global structures under illumination shifts, while a Shift-Invariant Spectral Correlation loss captures fine-grained local structures and suppresses high-frequency noise. Finally, we propose a Gain-Adaptive Feature Modulation (GAFM) mechanism to address highly spatially-variant residual noise. By transforming the self-estimated illumination map into an internal spatial gain prior, GAFM dynamically guides a blind-spot network for spatially-aware denoising. Extensive experiments demonstrate that our method achieves state-of-the-art performance, delivering superior noise suppression and textural fidelity. Code will be publicly released at this https URL.
>
---
#### [new 047] Mahalanobis PatchCore: Covariance-Aware and Streaming-Compatible Industrial Anomaly Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于工业视觉异常检测任务，解决正常样本多、缺陷样本少且难以获取的问题。提出Mahalanobis PatchCore，通过协方差感知和流式处理提升检测精度并降低内存占用。**

- **链接: [https://arxiv.org/pdf/2605.27748](https://arxiv.org/pdf/2605.27748)**

> **作者:** Niccolò Ferrari; Oligert Osmani; Evelina Lamma
>
> **备注:** 57 pages, 7 figures
>
> **摘要:** Industrial visual anomaly detection is usually one-class: normal images are abundant, while defects are rare, heterogeneous, and often unavailable during system design. PatchCore-style retrieval suits this setting because it scores test images from a memory bank of normal patch features, but the standard Euclidean geometry ignores feature correlations and its offline construction materialises the full patch pool before subsampling. We introduce Mahalanobis PatchCore, a covariance-aware, streaming-compatible extension of PatchCore. Its artificial intelligence contribution is a retrieval detector that estimates a regularised covariance model in reduced feature space and whitens embeddings, so Euclidean nearest-neighbour search after transformation implements Mahalanobis retrieval. A bounded-memory, re-iterable training pipeline builds the memory bank without storing all normal patches at once, using incremental dimensionality reduction, online covariance estimation, and streaming aggregation. The engineering application is automated industrial inspection, where visual anomaly detection must remain accurate under practical memory limits. We evaluate the method on a public 15-category industrial anomaly-detection benchmark and three industrial datasets covering blow-fill-seal strip-ampoule meniscus inspection, amber-glass-ampoule bottom inspection, and lyophilised-cake vial inspection. Mahalanobis PatchCore preserves most offline PatchCore image-level performance on the public benchmark while reducing peak memory from 5.41 to 2.78 GB, and improves the selected industrial mean image area under the receiver operating characteristic curve from 0.981 to 0.986.
>
---
#### [new 048] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning
- **分类: cs.CV**

- **简介: 该论文属于联邦学习任务，旨在解决FL中全局适应与泛化难以平衡的问题。提出FedDTL框架，通过解耦编码器和两阶段微调提升性能。**

- **链接: [https://arxiv.org/pdf/2605.27900](https://arxiv.org/pdf/2605.27900)**

> **作者:** Yuting Ma; Lechao Cheng; Xiaohua Xu
>
> **备注:** This work has been accepted by ICML 2026
>
> **摘要:** Federated Learning (FL) with pre-trained Vision-Language Models (VLMs) has emerged as a promising paradigm for various downstream tasks. By leveraging its strong representations, recent studies improve task adaptation under insufficient local data while preserving generalization. However, these methods emphasize fully local optimization with simple parameter aggregation,which can amplify inter-client optimization inconsistency and intra-client over-specialization under heterogeneous and full-data FL settings, making it difficult to balance global task adaptation and generalization. To address these challenges, we propose FedDTL, a novel federated VLM framework that decouples the image encoder and text encoder across clients and the server. Through decoupled encoder training with server-client modality alignment, FedDTL promotes coherent global semantic update and reduces inter-client optimization inconsistency, improving global task this http URL further mitigate intra-client over-specialization,we introduce a two-stage local fine-tuning, where a supervised fine-tuning stage enables rapid and reliable warm-start, followed by a reinforcement learning stage that enhances generalization. Extensive experiments on multiple benchmarks, including label skew and feature shift, demonstrate that FedDTL achieves an effective balance between global task adaptation and generalization under various FL data distributions in both few-shot and full-data regimes.
>
---
#### [new 049] Bayesian Gated Non-Negative Contrastive Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自监督表示学习任务，旨在解决对比学习中特征纠缠问题。提出BayesNCL方法，通过概率门控机制提升语义可解释性。**

- **链接: [https://arxiv.org/pdf/2605.28441](https://arxiv.org/pdf/2605.28441)**

> **作者:** Peng Cui; Jiahao Zhang; Lijie Hu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While Contrastive Learning (CL) has revolutionized self-supervised representation learning, its latent representations remain highly entangled and opaque, limiting their interpretability in safety-critical applications. We identify that a fundamental cause of this entanglement is the reliance on deterministic similarity measures, which treat all feature dimensions equally. In compositional scenes, this creates an Optimization Conflict: common background features, such as, "blue sky", are encouraged to align in positive pairs but simultaneously repelled in negative pairs, causing gradient oscillations that hinder precise semantic disentanglement. To address this, we propose BayesNCL (Bayesian Gated Non-Negative Contrastive Learning). Unlike standard approaches, BayesNCL introduces a probabilistic gating mechanism that dynamically filters out task-irrelevant, high-frequency common features while selectively retaining discriminative semantics. By formalizing feature selection as a variational inference problem with a sparse Bernoulli prior, our method effectively resolves the optimization conflict. Empirical experimental results on Imagenet-100 demonstrate that BayesNCL achieves a remarkable 142.1% improvement in semantic consistency compared to state-of-the-art baselines, yielding highly interpretable representations without compromising downstream task performance. Code is available at this https URL.
>
---
#### [new 050] Adaptive Temporal Gating of Longitudinal Magnetic Resonance Imaging for Alzheimer's Prediction
- **分类: cs.CV**

- **简介: 该论文属于阿尔茨海默病预测任务，解决MCI向AD转化的早期预测问题。提出TAF-Net模型，融合纵向MRI数据提升预测性能。**

- **链接: [https://arxiv.org/pdf/2605.28397](https://arxiv.org/pdf/2605.28397)**

> **作者:** Alireza Moayedikia; Sara Fin; Alicia Troncoso Lora; Uffe Kock Wiil
>
> **摘要:** Predicting conversion from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) is critical for early intervention. Current deep learning paradigms predominantly rely on cross-sectional structural MRI, neglecting prognostic value in patient-specific anatomical trajectories. We introduce the Temporal Adaptive Fusion Network (TAF-Net), a hybrid CNN-Transformer architecture that models paired longitudinal 3D MRI scans. Central to TAF-Net is a Temporal Fusion Module governed by an Adaptive Temporal Gate, which learns patient-specific weightings to synthesize three spatiotemporal representations: explicit structural change, region-to-region temporal cross-attention, and bilateral feature concatenation. Evaluated on the Alzheimer's Disease Neuroimaging Initiative cohort for three-year MCI-to-AD conversion prediction, TAF-Net achieved the highest discriminative performance among all evaluated methods using only structural MRI, significantly outperforming the strongest baseline and approaching multimodal methods requiring PET, CSF, or genetic data. The architecture exhibited exceptional data efficiency, matching baseline performance with a fraction of training data. Ablation studies demonstrate that longitudinal fusion improves discrimination while reducing predictive variance by 48% compared to single-timepoint evaluation. Interpretability analyses reveal spatial attention aligned with established AD pathology in the medial temporal lobe and ventricles, while the gating mechanism prioritizes explicit volumetric change with strong positive correlation to conversion risk.
>
---
#### [new 051] Stay Fair! Ensuring Group Fairness in Diffusion Models Across Guidance Scales
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于扩散模型公平性任务，解决引导尺度调整导致的群体不公平问题。通过分解偏差来源，提出StayFair算法，确保不同引导尺度下的公平性。**

- **链接: [https://arxiv.org/pdf/2605.28036](https://arxiv.org/pdf/2605.28036)**

> **作者:** Myeongsoo Kim; Eunji Kim; Minwoo Chae; Sangwoo Mo
>
> **备注:** 28 pages, 18 figures
>
> **摘要:** Diffusion models steer conditional generation with a tunable guidance scale to trade off prompt alignment and diversity. However, existing debiasing techniques are optimized for a single scale, degrading fairness when users adjust this parameter. We trace this behavior to a previously overlooked source by decomposing total bias into two components: a model bias and a guidance bias. While prior work primarily targets the former, we show that the guidance bias grows monotonically with the guidance scale, eventually dominating the high-guidance regimes users prefer. To address this, we extend Strong Demographic Parity to guidance and derive a condition under which the target distribution retains its group ratio across guidance scales. We propose StayFair, which leverages this condition to design fair guidance algorithms in both regimes. For classifier guidance, it equalizes the classifier's output distributions across groups; for classifier-free guidance, it shifts the null embedding by a prompt-dependent offset. Because StayFair modifies only the guidance step, it is orthogonal to model debiasing and can be layered onto existing fair diffusion models to extend their fairness across guidance scales. Across class-conditional and text-to-image generation, StayFair decouples fairness from the guidance scale without sacrificing image quality.
>
---
#### [new 052] SSR3D-LLM: Structured Spatial Reasoning via Latent Steps for Fine-Grained Grounding in Unified 3D-LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D物体定位任务，解决统一3D-LLM在细粒度查询中的定位问题。通过结构化空间推理和逐步排序提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.28490](https://arxiv.org/pdf/2605.28490)**

> **作者:** Jiawei Li; Ziyi Liu; Weijie Shi; Long Chen; Jiajie Xu; Xiaofang Zhou
>
> **摘要:** 3D object grounding localizes referred objects in a 3D scene from natural language. Unified instance-centric 3D-LLMs aim to solve grounding together with dialog, QA, and captioning, yet many rely on a single pointer-style grounding decision that compresses a relational instruction into one selection. This is brittle for fine-grained queries where multiple same-class candidates must be ruled out by context objects and spatial relations. We propose Structured Spatial Reasoning 3D-LLM (SSR3D-LLM), a structured grounding interface for unified 3D-LLMs. Given fixed Mask3D object proposals, the LLM writes a sequence of latent spatial reasoning steps and memory tokens from the query, and a geometry-aware scorer reads these latent steps in order to refine candidate rankings step by step with step-length masking. The latent steps are learned from standard benchmark target supervision with auxiliary referential-cue supervision during training, while inference uses only the input query and Mask3D proposals. Across ReferIt3D, ScanRefer, and Multi3DRef, SSR3D-LLM achieves the strongest results among unified 3D-LLM baselines, with substantial gains over the single-pointer QPG baseline on fine-grained grounding and consistent improvements over prior unified 3D-LLMs, while preserving the default language-task route.
>
---
#### [new 053] Bias Leaves a Gradient Trail: Label-Free Bias Identification via Gradient Probes on Concept Decompositions
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉模型偏见分析任务，解决部署后模型中隐含偏见的识别问题。通过分析梯度和概念分解，无需标签即可发现决策相关的虚假特征，并提升模型公平性。**

- **链接: [https://arxiv.org/pdf/2605.28780](https://arxiv.org/pdf/2605.28780)**

> **作者:** Thomas Vitry; Kieran Edgeworth; Stefan Wermter; Jae Hee Lee
>
> **备注:** Accepted to the 49th German Conference on Artificial Intelligence (KI2026)
>
> **摘要:** Vision classifiers can exploit spurious correlations, achieving high in-distribution accuracy yet failing under distribution shift. Existing approaches to bias mitigation and analysis often depend on curated datasets, spurious-attribute or group labels, or retraining, which may be infeasible once a model is deployed or the relevant bias is unknown. We present a bias-label-free, post-hoc method for identifying spurious concepts in frozen vision models, relying only on standard class labels from a held-out audit dataset. For each target class, we collect patches from inputs predicted as that class and apply non-negative matrix factorization to intermediate activations to obtain a bank of interpretable concept vectors. Candidate concepts are then ranked with a bias estimator derived from their interaction with backpropagated gradients on misclassified examples: bias concepts tend to get activated when correcting false negatives and suppressed when correcting false positives. On Colored MNIST and Waterbirds the method recovers concepts aligned with the known spurious cue, and on CelebA it surfaces decision-relevant directions that only partially coincide with the annotated gender attribute; suppressing the top-ranked concepts at inference time improves worst-group accuracy by up to 17.9 percentage points on Waterbirds and 10.4 on CelebA without any retraining or parameter updates. Our method identifies decision-relevant spurious directions that need not coincide with annotated ones, providing both an interpretable auditing tool and an actionable debiasing handle for frozen vision models. Code is available at this https URL.
>
---
#### [new 054] ROVER: Routing Object-Centric Visual Evidence for Grounded Multi-Image Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多图像推理任务，旨在解决视觉证据整合效率与全局理解不足的问题。提出ROVER模型，通过对象中心的视觉证据路由提升多图像推理性能。**

- **链接: [https://arxiv.org/pdf/2605.27959](https://arxiv.org/pdf/2605.27959)**

> **作者:** Guannan Lv; Ren Nie; Hongjian Dou
>
> **摘要:** Multimodal Large Language Models (MLLMs) have increasingly localized and interleaved visual evidence for deliberative reasoning. Grounding-based approaches typically focus on regions of interest (RoIs) by injecting cropped image patches or RoI-specific features into the reasoning context. However, such designs can weaken holistic scene understanding and inter-object relations, while incurring decoding costs that scale with the number and size of RoIs. Alternatively, adaptive visual feature selection often requires fine-grained supervision or complex heuristics. To address these limitations, we propose ROVER (Routing Object-centric Visual Evidence for grounded multi-image Reasoning), a lightweight, learnable plugin for efficient global visual evidence routing. Upon each object grounding prediction, ROVER injects a step-specific token triplet to synergistically: (i) aggregate the ongoing reasoning context, (ii) distill intra-image cues into a visual working space via object-centric differential attention, and (iii) route and integrate history-aware evidence across objects and images within this space for subsequent reasoning. We integrate ROVER into Qwen2.5-VL-7B and develop an interleaved SFT-to-GRPO training pipeline. Strictly adhering to the original datasets and evaluation protocols, our method achieves the best performance on MM-GCoT (+4.8% answer accuracy, +14.6% grounding accuracy) and VideoEspresso (+8.6% answer accuracy). The VideoEspresso-trained model demonstrates strong transferability, outperforming the base model by +4.7% on average across diverse benchmarks.
>
---
#### [new 055] Inpainting-Style Conditional Diffusion for Multivariable Time Series Forecasting
- **分类: cs.CV**

- **简介: 该论文属于时间序列预测任务，解决多变量太阳能功率预测问题。通过将时间序列转化为图像并引入扩散模型，实现未来数据的修复与预测。**

- **链接: [https://arxiv.org/pdf/2605.28324](https://arxiv.org/pdf/2605.28324)**

> **作者:** Kourosh Kiani; S.M. Muyeen
>
> **摘要:** In this paper, we propose a novel conditional diffusion-based framework for multivariable time-series solar power forecasting. The proposed method reformulates temporal PV data as structured two-dimensional representations (images) using a sliding-window patch construction, enabling the application of Denoising Diffusion Probabilistic Models (DDPM) within a unified spatiotemporal learning paradigm. A key contribution of this work is the formulation of solar forecasting as an inpainting problem, where future time steps are treated as missing regions to be reconstructed. This is achieved through a mask-based conditional diffusion mechanism, in which historical observations are preserved as conditioning context while the target (future) region is progressively corrupted and subsequently recovered via reverse diffusion. The model learns to generate coherent future sequences conditioned on observed data, effectively performing time-series inpainting. To fully utilize all available features and ensure compatibility with U-Net architectural constraints, a zero-padding strategy is introduced to construct fixed-size inputs. The model is trained using a supervised denoising objective to predict injected noise, enabling accurate iterative reconstruction during the reverse process. Extensive experiments conducted on benchmark PV dataset, including GEFCom2014, demonstrate that the proposed approach achieves high forecasting accuracy, particularly for short-term horizons. The results highlight the effectiveness of integrating diffusion-based generative modeling with an inpainting formulation for robust, flexible, and high-fidelity solar power forecasting.
>
---
#### [new 056] SIGMA: Bridging Structural and Distributional Gaps for Vision Foundation Model Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视觉基础模型适应任务，旨在解决参数高效微调中的结构与分布差异问题。提出SIGMA方法，通过多粒度融合和语义调制实现高效适配。**

- **链接: [https://arxiv.org/pdf/2605.27893](https://arxiv.org/pdf/2605.27893)**

> **作者:** Lingyu Xiong; Jinjin Shi; Xuran Xu; Cong Luo; Runyu Shi; Ying Huang
>
> **摘要:** Vision Foundation Models (VFMs) have demonstrated impressive representational capabilities. However, adapting them to downstream tasks via full fine-tuning incurs prohibitive computational and storage overhead. Parameter-Efficient Fine-Tuning (PEFT) has emerged as a compelling alternative, aiming to achieve performance parity with full fine-tuning at minimal training costs. Nonetheless, applying PEFT to VFMs for dense prediction tasks remains challenging due to the structural and distributional gaps. To bridge these gaps, we propose \textbf{S}cale-\textbf{I}ntegrated \textbf{G}lobal \textbf{M}odulation \textbf{A}dapter (\textbf{SIGMA}), a novel lightweight PEFT method, which consists of two modules: scale-adaptive fusion and semantic modulation. Specifically, the scale-adaptive fusion module is utilized to bridge structural gaps by enhancing the extraction of multi-granularity visual information. Furthermore, SIGMA introduces semantic modulation on the fusion features to perform global feature alignment to further eliminate the distribution gap. This design facilitates unified spatial and distributional adaptation, requiring only 1.72\% trainable parameters relative to the VFM backbone. Comprehensive experiments across various downstream dense tasks and multiple VFM backbones demonstrate that SIGMA achieves consistent and superior performance over state-of-the-art PEFT methods.
>
---
#### [new 057] Ω-QVLA: Robust Quantization for Vision-Language-Action Models via Composite Rotation and Per-step Scaling
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Omega-QVLA，解决VLA模型部署成本高的问题，通过统一量化语言和动作模块，实现高效压缩与稳定性能。**

- **链接: [https://arxiv.org/pdf/2605.28803](https://arxiv.org/pdf/2605.28803)**

> **作者:** Xinyu Wang; Mingze Li; Sicheng Lyu; Dongxiu Liu; Kaicheng Yang; Ziyu Zhao; Yufei Cui; Xiao-Wen Chang; Peng Lu
>
> **摘要:** Vision-Language-Action (VLA) models unify perception, reasoning, and control within a single policy, yet their multi-billion-parameter backbones and diffusion-based action heads make on-device deployment prohibitively expensive. Prior quantization efforts offer only partial solutions, compressing the LLM backbone while leaving the DiT action head at full precision, or resorting to mixed-precision schemes, driven by the belief that uniformly quantizing the action head is inherently unstable. We challenge this assumption with Omega-QVLA, the first training-free post-training quantization framework that compresses both the language backbone and the entire diffusion action head of a VLA model to a uniform W4A4 precision, eliminating the need for mixed-precision allocation. Omega-QVLA combines a composite SVD-Hadamard rotation that equalizes per-channel weight energy while diffusing residual activation outliers with per-step DiT activation scaling quantization that absorbs dynamic-range drift across denoising steps. On LIBERO, Omega-QVLA compresses Pi 0.5 and GR00T N1.5 to W4A4 with 98.0% and 87.8% task success rates, matching or exceeding their FP16 references of 97.1% and 87.0%, while reducing the static memory footprint by 71.3%. Real-world manipulation experiments further confirm smooth, accurate manipulation where prior methods fail. Code is available at this https URL.
>
---
#### [new 058] Do We Really Need Quantum Machine Learning?: A Multidimensional Empirical Study
- **分类: cs.CV; cs.AI; cs.LG; quant-ph**

- **简介: 该论文属于图像识别任务，旨在比较量子与经典机器学习模型的性能。通过实验分析不同模型在准确率、运行时间等维度的表现，探讨量子模型是否更具优势。**

- **链接: [https://arxiv.org/pdf/2605.27923](https://arxiv.org/pdf/2605.27923)**

> **作者:** Sudip Vhaduri; Ryan Gammon; Sayanton Dibbo
>
> **摘要:** The rapid growth of computer vision and increasingly complex image recognition tasks has exposed fundamental computational limitations of classical machine learning models, motivating the exploration of quantum computing as an emerging new paradigm. This paper presents a comprehensive benchmarking study of classical and quantum machine learning models for image recognition on the MNIST handwritten digit dataset, evaluating both traditional models, a Classical Support Vector Machine (CSVM) and a Quantum Support Vector Machine (QSVM), and deep neural network models, a Classical Convolutional Neural Network (CCNN) and a Quantum Convolutional Neural Network (QCNN), across four performance dimensions: classification accuracy, computational runtime, parameter count, and memory requirements. Experiments are conducted as functions of both feature dimensionality and sample size, and across CPU and GPU execution environments, providing a controlled, multidimensional comparison to address gaps in prior work. For the SVM-based models, QSVM consistently outperforms CSVM in accuracy, reaching $\sim$ 0.90 versus $\sim$ 0.85 at 1,000 samples, with a higher computational cost. A feature count of 10 qubits and a sample size in the range of 200 -- 500 emerge as practical operating points that balance accuracy and runtime. For the neural network models, CCNN and QCNN achieve comparable classification accuracy, both exceeding 0.96 at 64 features and 60,000 samples, yet QCNN offers substantially superior parameter and memory efficiency, requiring $\sim$ 94\% fewer parameters and $\sim$ 75\% less memory than CCNN at higher feature counts, while incurring higher runtime. Across both model families, quantum models consistently outperform classical models by greater margins in accuracy as feature dimensionality or sample size increases.
>
---
#### [new 059] VCap: Hypergeometric Rewards for Weak-to-Strong Visual Captioning
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于视觉问答任务，解决 captioning 中事实一致性验证问题。提出 VCap 奖励机制，通过对比参考 caption 与视觉信号，提升生成质量与准确性。**

- **链接: [https://arxiv.org/pdf/2605.28023](https://arxiv.org/pdf/2605.28023)**

> **作者:** Xingyu Lu; Jinpeng Wang; Yi-Fan Zhang; Yankai Yang; Yancheng Long; Yiyang Fan; Xuanyu Zheng; Haonan Fan; Kaiyu Jiang; Tianke Zhang; Changyi Liu; Bin Wen; Fan Yang; Tingting Gao; Han Li; Chun Yuan
>
> **备注:** 28 pages, 8 figures
>
> **摘要:** Visual captioning requires models to capture visual content faithfully while minimizing both omission and hallucination. As the dominant paradigm for captioning, MLLMs have achieved strong performance through scaling and high-quality data. Recently, RL has emerged as a key route to driving MLLMs toward higher precision and broader coverage, however, existing reward designs for captioning fail to provide fine-grained and reliable signals for factual verification, limiting their effectiveness. To address this, we propose VCap, a Witness-Adjudicator reward that pairs the reference caption (a witness) with the visual signal (an adjudicator). By explicitly verifying factual consistency between the reference and policy-generated captions grounded in the visual signal, VCap delivers a reward signal with hypergeometric-distribution-level precision for caption quality verification. This design enables effective learning even from imperfect references, facilitating weak-to-strong generalization in RL training. In our experiments, an 8B model trained with VCap outperforms open- and closed-source SOTA models on multiple image and video captioning benchmarks. Human evaluation further confirms its strong alignment with factual correctness. Additionally, VCap improves MLLM perceptual capability, generalizes across tasks, and surpasses best-of-N distillation, challenging prior assumptions about RLVR.
>
---
#### [new 060] DebFilter: Eradicating Biases Stashed in Value
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型中因文本嵌入和社会语义偏见导致的生成结果偏差问题。提出DebFilter框架，在推理阶段通过调整注意力值来减轻偏见，无需训练。**

- **链接: [https://arxiv.org/pdf/2605.28167](https://arxiv.org/pdf/2605.28167)**

> **作者:** Seung Hyuk Lee; Songkuk Kim
>
> **备注:** 8 pages, 7 figures, supplementary material included, CVPR 2026
>
> **摘要:** Text-to-image diffusion models, which are theoretically equivalent to score-based generative models, generate images through a multi-step denoising process guided by text embeddings extracted from pretrained vision-language models such as CLIP. However, these text embeddings inherently encode social and semantic biases -- such as those related to gender and age -- that are subsequently propagated and amplified through the guidance mechanism, along with the model's training on large-scale datasets that are imbalanced with respect to these bias-related concepts, often leading to skewed outputs in text-to-image generation. We propose DebFilter, a lightweight and training-free framework for mitigating such biases in text-to-image diffusion models. Observing that the model's error prediction at each denoising step is primarily influenced by cross-attention dynamics, we introduce a bias-correction strategy that adjusts the value components within cross-attention. Specifically, we apply a fixed offset to the slice of guidance embedding, effectively steering the semantic direction of cross-attention values toward unbiased representations. This adjustment reconfigures the score landscape to produce balanced outputs while maintaining alignment with the intended text semantics. Unlike prior approaches that rely on fine-tuning or retraining, DebFilter operates entirely at inference time, requiring no additional data or model updates. Our results demonstrate that this method effectively mitigates social biases in generated images, offering an efficient and scalable pathway toward fairer and more inclusive text-to-image generation.
>
---
#### [new 061] AdaMerge: Salience-Aware Adaptive Token Merging for Training-Free Acceleration of Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉Transformer加速任务，解决token合并中的信息丢失问题。提出AdaMerge框架，通过显著性加权和自适应合并强度提升性能。**

- **链接: [https://arxiv.org/pdf/2605.27465](https://arxiv.org/pdf/2605.27465)**

> **作者:** Semi Lee; Hyejin Go; Hyesong Choi
>
> **备注:** 11 pages, 3 figures, 5 tables. Submitted to NeurIPS 2026
>
> **摘要:** The quadratic cost of self-attention in Vision Transformers (ViTs) constitutes a fundamental bottleneck for practical deployment, motivating a vibrant line of research on token reduction. Among existing approaches, token merging (ToMe) has emerged as an elegant training-free solution; yet its design rests on an unspoken premise of token equality, which contravenes the well-documented non-uniformity of self-attention and leads to information loss in high-salience tokens under aggressive compression. We address this limitation with AdaMerge, a token-merging framework based on two complementary mechanisms. First, salience-weighted similarity leverages column-wise feature-affinity centrality as a token-importance proxy and incorporates the resulting salience scores into the bipartite matching score, ensuring that pivotal tokens contribute more strongly to the merged representation. Second, adaptive merging intensity uses pre-computed layer-wise similarity statistics to dynamically modulate the per-layer reduction count in accordance with input-specific redundancy. On ImageNet-1k with ViT-B/16, AdaMerge consistently outperforms ToMe, PiToMe, and DSM across all FLOPs-matched regimes. The accuracy gap widens monotonically with compression: at the 13.4G FLOPs operating point, AdaMerge sustains a Top-1 degradation of only -1.06%, compared to -1.45% for PiToMe and -4.62% for DSM. To our knowledge, AdaMerge is the first to combine salience-weighted similarity and adaptive per-layer reduction into a single training-free token merging framework, advancing the accuracy-FLOPs Pareto frontier of ViT acceleration.
>
---
#### [new 062] Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于无监督异常检测任务，旨在解决仅用正常样本检测图像中细微异常的问题。提出ANoCo方法，通过图拉普拉斯能量优化衡量异常程度，无需训练且计算高效。**

- **链接: [https://arxiv.org/pdf/2605.28428](https://arxiv.org/pdf/2605.28428)**

> **作者:** Jungwook Seo; Minjeong Kim; Younkwan Lee; Seungho Shin; Sungyong Baik
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Detecting subtle visual anomalies in images remains challenging, particularly when only normal samples are available a priori. Such unsupervised anomaly detection is typically solved by measuring feature similarity of a query patch to a memory of normal patches. However, similarity alone does not reveal how strongly a query patch violates the structure of the normal feature manifold. We propose a training-free Laplacian graph energy optimization formulation, named ANoCo that scores Anomaly by the cost of Non-Conformity of a query patch to align with a fixed normal manifold. For each query patch, we construct a bipartite query to normal graph weighted by cosine affinity, explicitly removing query-query and normal-normal edges to prevent evidence dilution. We formulate anomaly scoring as a convex Laplacian energy with anchored normal nodes, and solve in closed form. In particular, we do not use the optimized features themselves-the anomaly score is the magnitude of the update required to satisfy normality constraints, reframing the graph Laplacian as a non-conformity operator rather than a smoothing prior. The proposed method introduces no learnable parameters, message passing, or sampling, and has complexity comparable to a single linear solve. Across standard benchmarks, it delivers strong image-level AUROC, stable localization maps, and improved robustness over prior methods, demonstrating the effectiveness of using optimization-induced feature drift as anomaly measure.
>
---
#### [new 063] VLA-Hijack: A Transferable Patch Attack against Vision-Language-Action Models via Visual Proprioception Hijacking
- **分类: cs.CV**

- **简介: 该论文属于安全攻击任务，针对视觉-语言-动作模型的对抗性补丁攻击问题，提出VLA-Hijack方法提升攻击的迁移性。**

- **链接: [https://arxiv.org/pdf/2605.28083](https://arxiv.org/pdf/2605.28083)**

> **作者:** Jiyuan Fu; Kaixun Jiang; Jingkai Jia; Zhaoyu Chen; Xueyao Chen; Lingyi Hong; Shuyong Gao; Chenzhi Tan; Dingkang Yang; Wenqiang Zhang
>
> **摘要:** While Vision-Language-Action (VLA) models have emerged as powerful generalist policies, their severe vulnerability to adversarial patches significantly hinders their deployment in safety-critical domains. Moreover, existing patch attacks primarily focus on white-box settings, heavily overfitting to the specific action output space of the target model, which results in poor cross-architecture transferability. To overcome this limitation, we propose VLA-Hijack, a unified adversarial framework that breaks the transferability bottleneck by exploiting a fundamental vulnerability identified in this work: before planning any motion, a VLA model must first use visual information to locate its own robotic arm within the environment. Targeting this shared visual self-localization process, our approach concurrently optimizes Attention-Guided Proprioceptive Suppression to inhibit the real robotic arm's features, and Multimodal Proprioceptive Injection to establish the patch as a surrogate "phantom embodiment". By alternating between semantic concept anchoring and visual prototype projection, VLA-Hijack effectively severs the semantic relationship between the agent's true embodiment and its control policy. Extensive experiments across diverse architectures (OpenVLA, UniVLA, and CronusVLA) demonstrate that VLA-Hijack achieves superior optimization efficiency in white-box settings and sets a new SOTA for cross-architecture and cross-domain black-box transferability.
>
---
#### [new 064] CogPortrait: Fine-Grained Eye-Region Control in Portrait Animation via Hierarchical Agent Planning
- **分类: cs.CV**

- **简介: 该论文提出CogPortrait，解决肖像动画中眼区精细控制问题，通过两阶段框架实现高精度眼区与头部动作控制。**

- **链接: [https://arxiv.org/pdf/2605.28056](https://arxiv.org/pdf/2605.28056)**

> **作者:** He Feng; Yongjia Ma; Donglin Di; Lei Fan; Tonghua Su
>
> **摘要:** Portrait animation methods have achieved substantial visual quality and lip synchronization, but fine-grained manipulation of the eye region still faces a trade-off between input granularity and motion accuracy. Existing methods using emotion labels or coarse text prompts are insufficient for describing subtle ocular dynamics, whereas approaches based on Action Units or driving videos provide higher fidelity at the cost of a heavier input burden. These limitations are still restrictive for beyond-emotion states (e.g., thinking) and drowsiness. In light of the above, we propose CogPortrait, a two-stage framework that generates portrait animations from high-level labels. In the first stage, three chain-of-thought Multimodal Large Language Models (MLLMs) agents compile high-level labels into facial keypoints through temporal event planning, prototype retrieval, and composition from a real-behavior library, and semantic-physiological constraint enforcement. In the second stage, a DiT-based video generation backbone synthesizes the final animation conditioned on the keypoints, reference portrait, audio, and text prompt, enhanced by a dynamic classifier-free guidance strategy with eye-region-aware reweighting and KTO-based refinement for boundary cases. We further introduce the EMH benchmark covering diverse emotions and beyond-emotion categories with two AU-level metrics for evaluating fine-grained eye-region and head-motion control. Extensive experiments on HDTF and the EMH benchmark demonstrate that CogPortrait achieves more precise eye-region control than existing methods while maintaining supe- rior visual quality and identity consistency
>
---
#### [new 065] PointQ-Bench: Benchmarking Diagnostic and Interpretable Point Cloud Quality Assessment
- **分类: cs.CV**

- **简介: 该论文提出PointQ-Bench，用于点云质量评估，解决现有方法仅预测分数而缺乏诊断与解释的问题。工作包括构建数据集和评估协议，支持缺陷诊断与质量描述任务。**

- **链接: [https://arxiv.org/pdf/2605.28241](https://arxiv.org/pdf/2605.28241)**

> **作者:** Duanchu Wang; Cheng Li; Junjie Yang; Jing Huang; Zihang Cheng; Zhi Gao; ZhuBohong; Di Wang
>
> **摘要:** Point cloud quality plays a critical role in 3D acquisition, reconstruction, rendering, and perception, yet existing point cloud quality assessment (PCQA) research remains largely centered on scalar score prediction. In practical inspection scenarios, quality assessment often involves identifying defects, characterizing dominant issue types, assessing downstream usability, and providing evidence-supported descriptions, which are not explicitly evaluated by current benchmarks. We introduce PointQ-Bench, a benchmark designed to extend PCQA from scalar scoring toward comprehensive quality understanding. PointQ-Bench consists of 3,083 point clouds spanning authentic scans, simulated distortions, and AI-generated content, covering eight major issue types. Each sample is annotated with mean opinion scores (MOS), quality levels, issue tags, expert-grounded descriptions, and 12,332 question-answer pairs. The benchmark supports three perception-oriented tasks: anomaly sensing, defect diagnosis, and usability grading, as well as a cognition-oriented task of open-ended quality reporting. To evaluate free-form quality descriptions, we further propose SSFRQ-5D, a five-dimensional evaluation protocol validated through human-AI agreement analysis. Extensive experiments on 14 vision-language models and traditional PCQA baselines reveal a consistent perception-diagnosis gap: while current models exhibit emerging abilities in coarse defect perception, they struggle with grounded diagnosis and quality calibration. Strong 2D MLLMs generally outperform existing 3D VLMs, and the benefit of additional views or point-level inputs is non-uniform, varying across tasks, data sources, and models, particularly under boundary-ambiguous conditions. Overall, PointQ-Bench provides a diagnostic testbed for advancing reliable and interpretable point cloud quality understanding.
>
---
#### [new 066] CLEAR-NeRF: Collinearity and Local-region Enhanced Accurate 3D Reconstruction in Unbounded Scenes
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，解决复杂场景下光照和姿态变化带来的精度问题。提出CLEAR-NeRF方法，提升无界场景的重建准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.28125](https://arxiv.org/pdf/2605.28125)**

> **作者:** Vladislav Polianskii; Elijs Dima; Isabel Salmerón Marazuela; Gergő László Nagy; Sigurdur Sverrisson; Volodya Grancharov
>
> **摘要:** Many real-world 3D reconstruction applications demand photorealism and metric accuracy across unbounded, complex scenes with challenging lighting and imperfect captures that current Neural Radiance Field (NeRF) pipelines only partly satisfy. This study adapts NeRF-based 3D reconstruction to multi-region of interest unbounded scenes to improve robustness to lighting and pose variation while enforcing metric accuracy suitable for digital-twin applications. Our approach introduces (i) automated local region localization/detection and reconstruction to seamlessly prioritize areas of interest without proliferating submodules, (ii) collinearity-enforcing ray sampling to learn smooth planar and curved surfaces, (iii) depth-localized neighborhood point extraction to suppress surface artifacts, and (iv) geometry-relevant color aggregation to mitigate lighting- and pose-caused variations. Results indicate superior performance of the proposed pipeline over the baseline NeRF models and established Structure from Motion (SfM) - Multi-View Stereo (MVS) solutions.
>
---
#### [new 067] AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于CLIP-based类增量学习任务，解决模型在新增类别时的灾难性遗忘问题。提出AREA方法，通过属性提取与聚合稳定模型性能。**

- **链接: [https://arxiv.org/pdf/2605.28809](https://arxiv.org/pdf/2605.28809)**

> **作者:** Zhen-Hao Xie; Yu-Cheng Shi; Da-Wei Zhou
>
> **备注:** Accepted to ICML 2026. Code is available at this https URL
>
> **摘要:** Class-Incremental Learning (CIL) is important in building real-world learning systems. In CLIP-based CIL, the model performs classification by comparing similarity between visual and textual embeddings obtained from template prompts, e.g., ``a photo of a [CLASS]''. This seemingly monolithic matching process can be decomposed into two conceptually distinct stages: attribute extraction and attribute aggregation. For example, a model may recognize cat using attributes such as fur texture and whiskers. When learning a new class like car, the model must extract additional attributes like wheels and adjust how they are aggregated in the shared representation space. However, since only data from the current task is available, incremental updates can bias both attribute extraction and aggregation toward new classes, leading to catastrophic forgetting. Therefore, we propose AREA for attribute extraction and aggregation in CLIP-based CIL. To stabilize extraction, we anchor class-level visual and textual attributes on the hyperspherical embedding space via principal geodesic analysis. To stabilize aggregation, we learn lightweight task-specific experts with scoring and residual refinement, regularized by a variational information bottleneck objective. During inference, we perform routing over task attribute manifolds via optimal transport for more concise prediction. Experiments show that AREA consistently outperforms SOTA methods. Code is available at this https URL.
>
---
#### [new 068] From Kellgren-Lawrence to Calcium Pyrophosphate Crystal Deposition: A Soft-Labelling Framework for Knee Osteoarthritis Assessmen
- **分类: cs.CV**

- **简介: 该论文属于膝骨关节炎评估任务，旨在解决传统方法无法准确反映KL和CPPD评分的有序不确定性和非对称关系的问题。通过引入软标签框架提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.28176](https://arxiv.org/pdf/2605.28176)**

> **作者:** Francisco Bérchez-Moreno; Riccardo Rosati; Maria Chiara Fiorentino; Víctor M. Vargas; Edoardo Cipolletta; Emilio Filippucci; Luca Romeo; Pedro A. Gutiérrez; César Hervás-Martínez
>
> **摘要:** Background and objective. Conventional Deep Learning (DL) approaches for Knee Osteoarthritis (KOA) grading rely on one-hot labels, which fail to capture both the ordinal uncertainty of Kellgren--Lawrence (KL) and Calcium Pyrophosphate Deposition Disease (CPPD) severity scores and the asymmetric relationship between the two scales observed in clinical practice. Methods. We retrospectively collected 2172 knee X-ray images, including 968 radiographs jointly annotated for KL and CPPD severity. An ordinal DL framework based on soft-labelling was developed for both tasks, replacing one-hot targets with unimodal probability distributions centred on the annotated grade. Four formulations were investigated: binomial, beta, triangular, and exponential. Results. All soft-labelling strategies consistently outperformed the nominal baseline. For CPPD grading, the triangular formulation achieved the highest Quadratic Weighted Kappa (QWK) and the lowest Mean Absolute Error (MAE) (QWK = 0.796; MAE = 0.438), while the beta formulation yielded the most balanced class-wise performance considering Average MAE (AMAE) and Maximum MAE (MMAE) across classes (AMAE = 0.458; MMAE = 0.573). For KL grading, the beta-based approach provided the best overall performance, achieving the highest QWK together with the lowest MAE and class-wise errors (QWK = 0.777; MAE = 0.529; AMAE = 0.523; MMAE = 0.775). Statistical analysis demonstrated significant improvements over conventional one-hot supervision (p < 0.001).
>
---
#### [new 069] Structure-Guided Visual Perturbation Neutralization for LVLMs
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型的防御任务，解决对抗扰动问题。提出SIGN框架，通过结构引导实现高效且轻量的扰动中和。**

- **链接: [https://arxiv.org/pdf/2605.27927](https://arxiv.org/pdf/2605.27927)**

> **作者:** Yuanhe Zhang; Xueting Wang; YanBin Ren; Haoran Gao; Xinhan Zheng; Zhenhong Zhou; Fanyu Meng; Li Sun; Sen Su
>
> **摘要:** Image inputs enable Large Vision Language Models (LVLMs) to perceive fine-grained visual information, but also introduce a pixel-level attack surface through which adversarial perturbations can elicit unsafe model behaviors. However, most existing defenses are designed for traditional computer vision settings and thus often overlook the cross-modal alignment required by LVLMs, leading to degraded performance. Meanwhile, the limited defenses tailored to LVLMs often require substantial image modifications and introduce considerable computational overhead, thereby compromising inference quality and efficiency. To address these limitations, we propose Structure-Induced Guided Neutralization (SIGN), a lightweight, plug-and-play defense framework that improves LVLM compatibility via Prior Structural Extraction and achieves efficient perturbation suppression via Dynamic Guided Neutralization. Extensive experiments show that SIGN achieves over 87\% defense success rate with only 0.5\% pixel modification and 0.16 seconds per image, while nearly preserving original visual representations and benign task performance. Our work offers a lightweight alternative to defenses that require costly model training and highlights the potential of exploiting a vision encoder for efficient adversarial protection. Our code is open source on this https URL.
>
---
#### [new 070] MeniOmni: A Structured Multimodal Benchmark for Holistic Meniscus Injury Assessment
- **分类: cs.CV**

- **简介: 该论文提出MeniOmni，一个用于半月板损伤评估的多模态基准，解决临床诊断中整合影像与患者信息的问题。工作包括构建数据集、定义评估任务及引入新评价指标。**

- **链接: [https://arxiv.org/pdf/2605.28161](https://arxiv.org/pdf/2605.28161)**

> **作者:** Shurui Xu; Siqi Yang; Weiping Ding; Hui Wang; Mengzhen Fan; Yuyu Sun; Shuyan Li
>
> **备注:** Accepted by IEEE International Conference on Multimedia and Expo (ICME) 2026 (Oral Presentation)
>
> **摘要:** Clinical diagnosis of meniscus injuries requires radiologists to integrate volumetric MRI evidence with patient context (e.g., sex, age, BMI) and to produce structured diagnostic reports. Existing knee MRI benchmarks are typically unimodal and rely on coarse labels, limiting their ability to evaluate holistic clinical reasoning. We introduce MeniOmni, a structured multimodal benchmark for meniscus injury assessment, consisting of 746 multi-center MRI studies with tri-planar volumetric inputs, Clinical Priors, and expert-annotated clinical text. MeniOmni supports two tasks: (1) fine-grained Stoller severity grading and (2) diagnostic report generation. We further propose risk-aware ordinal evaluation and a semantic consistency metric (Meni-Score) to better reflect clinical relevance. Baseline experiments show that incorporating Clinical Priors improves grading performance and reduces severe errors, highlighting the value of multimodal context for safer assessment. Code and data are available at this https URL.
>
---
#### [new 071] Pattern Recognition Tasks with Personalized Federated Learning
- **分类: cs.CV**

- **简介: 该论文研究个性化联邦学习在模式识别任务中的应用，旨在提升模型准确性与隐私保护。通过比较七种算法，评估其在多个数据集上的表现，找出最优方案。**

- **链接: [https://arxiv.org/pdf/2605.27816](https://arxiv.org/pdf/2605.27816)**

> **作者:** Md. Arifur Rahman; Isha Das; Mushfiqur Rahman Abir; B. M. Taslimul Haque; Abdullah Al Noman; Abir Ahmed; Md. Jakir Hossen
>
> **备注:** Comprehensive comparative analysis of 7 Personalized Federated Learning algorithms across MNIST, SignMNIST, and Digit5 datasets. The paper presents detailed methodology, workflow architecture, experimental evaluation, and privacy-preserving AI analysis for distributed intelligent systems, secure collaborative learning, and critical infrastructure applications
>
> **摘要:** Personalized Federated Learning (PFL) constitutes a novel paradigm that tailors Machine Learning (ML) models to individual clients, thereby furnishing personalized model updates whilst upholding stringent data privacy principles. Diverging from conventional standard Federated Learning (FL) approaches, PFL adapts models to distinct client data distributions, engendering heightened levels of accuracy, customization, and data security, all while minimizing communication overhead. This methodology proves particularly salient in contexts marked by pattern recognition tasks reliant upon heterogeneous data sources and underpinned by paramount privacy apprehensions. In the present research endeavor, this article undertake a comprehensive comparative analysis of seven distinct PFL algorithms deployed across three diverse datasets, namely MNIST, SignMNIST, and Digit5. The overarching objective entails ascertaining the preeminent PFL algorithm, within the framework of pattern recognition tasks, through a rigorous evaluation anchored in metrics encompassing Accuracy, Precision, Recall, and F1 Score. Concurrently, an in-depth scrutiny of these PFL algorithms is conducted, elucidating their operative workflows, advantages, and limitations. Through empirical investigation, the findings evince that APPLE, FedGC, and FedProto emerge as stalwart contenders, consistently furnishing superior performance across the spectrum of assessed datasets, while acknowledging the contextual specificity of alternative algorithms and the potential for iterative refinement to realize optimal outcomes.
>
---
#### [new 072] When Think-with-Image Meets Safety: What Determines Multimodal Jailbreak Robustness?
- **分类: cs.CV; cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于安全评估任务，研究多模态模型的对抗攻击鲁棒性。通过实验比较不同设计模式，发现显式图像工具交互能有效提升安全性。**

- **链接: [https://arxiv.org/pdf/2605.27932](https://arxiv.org/pdf/2605.27932)**

> **作者:** Yuan Tian; Bing Hu; Fang Wu; Xiaomin Li; Binghang Lu; Neil Zhenqiang Gong
>
> **备注:** 17 pages, 6 figures, 7 tables
>
> **摘要:** Think-with-image reasoning is emerging as a new inference paradigm for large vision-language models, but its safety implications remain poorly understood. Existing systems already span multiple process designs, including direct response generation, text-only prior turn, visual-state manipulation, and explicit external image-tool invocation. In this paper, we ask which of these evaluated paradigms improves multimodal jailbreak robustness, and why. Across multiple vision-language models, explicit image-tool interaction yields the lowest attack success rates in our experiments, reducing jailbreak success by around 30% relative on average across the evaluated models. This finding is initially surprising: ASR remains low even when the returned image-tool output is manually overridden or itself unsafe-looking, but returns near direct-answering levels under text-only prior turn controls. These results indicate that the lower ASR is not explained by benign returned-image semantics or by the textual image-tool trace alone. To explain the pattern, we introduce an image-tool safety vector framework that models image-tool invocation as a residual shift in hidden representations toward a safety-relevant direction. Representation-level analyses and activation interventions support this account. Overall, our results suggest that explicit image-tool interaction is a promising design pattern for improving jailbreak robustness, while also motivating pipeline-specific safety evaluation.
>
---
#### [new 073] From Affect to Complex Behavior: Advancing Multimodal Human-Centered AI at the 10th ABAW Workshop & Competition
- **分类: cs.CV**

- **简介: 本文属于多模态人本AI研究，聚焦于真实环境下情感与行为分析。解决情感估计、行为识别等复杂任务，通过竞赛与论文展示最新方法与数据集。**

- **链接: [https://arxiv.org/pdf/2605.27451](https://arxiv.org/pdf/2605.27451)**

> **作者:** Dimitrios Kollias; Panagiotis Tzirakis; Alan Cowen; Stefanos Zafeiriou; Irene Kotsia; Eric Granger; Marco Pedersoli; Simon Bacon; Jens Madsen; Soufiane Belharbi; Muhammad Haseeb Aslam; Chunchang Shao; Guanyu Hu
>
> **备注:** accepted at CVPR 2026
>
> **摘要:** The 10th Affective & Behavior Analysis in-the-Wild (ABAW) Workshop and Competition, held at CVPR 2026, continues to advance research on modelling, analysis, understanding of human affect and behavior in real-world, unconstrained environments. The workshop maintains its dual structure, comprising both a competition and a paper track. The ABAW Competition introduces a diverse set of challenges targeting key aspects of affective and behavioral understanding, including continuous affect (valence-arousal) estimation, discrete affect (expression and action unit) recognition, as well as more complex behavior analysis tasks, such as emotional mimicry intensity estimation, ambivalence/hesitancy recognition and fine-grained violence detection. These challenges are built upon large-scale in-the-wild datasets, providing comprehensive benchmarks for state-of-the-art approaches. In parallel, the paper track presents a wide range of contributions spanning pose, motion & behavior estimation, affect modelling & multimodal learning, benchmarks, datasets & evaluation protocols, fairness, robustness & deployment. Overall, the 10th ABAW Workshop and Competition continues to serve as a key platform for benchmarking, collaboration and innovation, shaping the development of next-generation multimodal, human-centered AI systems.
>
---
#### [new 074] SA4Depth: Consistent Pose-Depth Scale Alignment for Self-Supervised Monocular Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，解决深度与位姿尺度不一致的问题。通过改进位姿网络，提升深度预测的一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.28477](https://arxiv.org/pdf/2605.28477)**

> **作者:** Changxuan Li; Nadine Berner; Nassir Navab; Federico Tombari; Stefano Gasperini
>
> **备注:** Accepted by IEEE RA-L 2026
>
> **摘要:** Self-supervised depth estimation from monocular sequences relies on the joint learning of a depth and a pose network. Despite abundant research done to improve the depth network, efforts on the pose remain limited. In this context, even when depth is estimated up to scale, we highlight the importance of the alignment between the scene scales estimated by the pose and depth nets. Then, we introduce SA4Depth, an approach to improve this alignment and boost the depth predictions while keeping the inference time unchanged. Our proposed method uses the depth estimated during training to reproject learnable visual features across consecutive frames and refine the pose estimates by reducing feature alignment residuals. With our method, the estimated scene scales by the separate depth and pose networks are aligned, and the prediction scale consistency is improved across different sequences. Our differentiable refinement integrates seamlessly into existing self-supervised pipelines and substantially improves their depth estimates. We demonstrate this with extensive experiments both outdoors and indoors on KITTI, Cityscapes, and NYUv2. Additionally, results on KITTI Odometry confirm the effectiveness of our pose refinement. Our code is available at this https URL .
>
---
#### [new 075] Self-Prophetic Decoding to Unlock Visual Search in LVLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉搜索任务，解决LVLMs在多步骤推理中的能力退化和干扰问题。提出SeProD框架，通过自预言采样实现连贯推理。**

- **链接: [https://arxiv.org/pdf/2605.28741](https://arxiv.org/pdf/2605.28741)**

> **作者:** Zhendong He; Qiyuan Dai; Guanbin Li; Liang Lin; Sibei Yang
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) are rapidly evolving toward true multimodal reasoning, with visual search representing a concrete instantiation of the thinking-with-images paradigm. However, LVLM visual search faces two key challenges: incompatibility among intrinsic capabilities after post-training, and interference in long multi-step reasoning contexts. To address these, we identify two novel insights. First, self-regulation between pre- and post-training LVLMs leverages the intrinsic single-step capabilities of the pre-training model to mitigate capability deterioration and long-context interference. Second, probability-based prophetic sampling, replacing naive prompting, provides a probabilistic interface where the pre-training model acts as a prophet and the post-training model selectively accepts prophetic tokens under its output distribution, preserving coherent multi-step reasoning. Building on these insights, we introduce SeProD, a self-prophetic decoding framework that leverages intrinsic single-step capabilities to enable coherent multi-step reasoning in a training-free, plug-and-play manner. Experiments show that SeProD consistently improves multiple visual-search LVLMs across all 12 splits of 4 visual search benchmarks, as well as across general VQA benchmarks, without added computational overhead, thanks to its parallel prophetic acceptance mechanism.
>
---
#### [new 076] Beyond Motion Primitives: Behavioral Activity Recognition from Head-Mounted IMU
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于行为识别任务，旨在突破传统运动基元限制，实现更高级别的行为识别。通过构建数据集和提出分层模型，提升AR设备的上下文感知能力。**

- **链接: [https://arxiv.org/pdf/2605.27464](https://arxiv.org/pdf/2605.27464)**

> **作者:** Chung-Ta Huang; Leopold Das; Jeffrey Zhou; Faizaan Siddique; Julia Seungjoo Baek; Serena Liu; Andrew Rusli; Todd Y. Zhou; Freddy Yu; Sinclair Hansen; Ziling Hu; Arnav Sharma; Mengyu Wang
>
> **摘要:** AR smart glasses need continuous behavioral context to offer proactive assistance, yet their most practical always-on sensor, the head-mounted Inertial Measurement Unit (IMU), detects only motion primitives such as walking or standing. We push beyond motion primitives to behavioral-level recognition, defining five categories that balance AR application need with sensor observability. To this end, we construct a 160K-sample Ego4D dataset with a four-tier quality assurance framework spanning 8 activity scenarios, and propose HiT-HAR, a 703K-parameter hierarchical model that outperforms prior head-mounted IMU models on five-class action and eight-class scenario recognition. We further map the observability frontier of head-mounted IMU through per-class separability analysis, identifying which behavioral categories are reliably observable (Locomotion), which benefit from temporal context (Object Transfer, Task Operation), and where scenario-dependent signal overlap poses remaining challenges. Our results indicate that architectural choices exploiting temporal context and scenario structure outperform simply scaling model size. The code and dataset are publicly available at this https URL.
>
---
#### [new 077] Janus-LoRA: A Balanced Low-Rank Adaptation for Continual Learning
- **分类: cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决LoRA方法中因更新不正交导致的灾难性遗忘问题，提出Janus-LoRA框架以平衡稳定性和可塑性。**

- **链接: [https://arxiv.org/pdf/2605.28495](https://arxiv.org/pdf/2605.28495)**

> **作者:** Cheng Chen; Pengpeng Zeng; Yuyu Guo; Lianli Gao; Hengtao Shen; Jingkuan Song
>
> **备注:** 9pages, International Conference on Machine Learning
>
> **摘要:** Low-Rank Adaptation (LoRA) has emerged as a promising paradigm for Continual Learning. It independently updates its low-rank factors ($A$ and $B$), creating a composite update to the full weight matrix through their interaction. To prevent catastrophic forgetting, this update should remain orthogonal to the task-specific subspace that contains previously learned knowledge. However, we identify that this composite update systematically violates this orthogonality, reintroducing interference and undermining stability. Furthermore, naively enforcing this orthogonality compromises plasticity, disrupting the delicate stability-plasticity trade-off. To resolve these issues, we propose \textbf{Janus-LoRA}, a framework that restores this balance through two novel components. Specifically, we first introduce Gradient Rectification, a closed-form solution that mathematically decouples LoRA's factor updates, enforcing orthogonality against the historical knowledge subspace identified by an efficient Online Estimation. Next, to enhance plasticity, we introduce a Decoupled Margin Loss that promotes feature-level separation by pushing new feature representations away from old ones, thus creating distinct, low-interference regions for new learning. Comprehensive experiments on challenging benchmarks demonstrate that by harmonizing parameter-level orthogonality with feature-level separation, Janus-LoRA achieves a superior balance and establishes new state-of-the-art performance.
>
---
#### [new 078] ABot-OCR Technical Report
- **分类: cs.CV**

- **简介: 该论文提出ABot-OCR，解决文档图像转Markdown的端到端任务，通过结构约束强化学习提升准确性与格式规范性。**

- **链接: [https://arxiv.org/pdf/2605.27978](https://arxiv.org/pdf/2605.27978)**

> **作者:** Kaitao Jiang; Ruiyan Gong; Xiaolong Cheng; Kangning Niu; Tianlun Li; Mu Xu
>
> **备注:** 21 pages, 11 figures, technical report
>
> **摘要:** We introduce ABot-OCR, an end-to-end vision-language model that transcribes a page image directly into clean Markdown in a single forward pass. By doing so, our approach completely eliminates the need for brittle modular orchestration. To maximize parsing fidelity, we develop a dedicated data engine to provide large-scale, structurally consistent supervision. Furthermore, we propose Decoupled Heterogeneous Document Optimization, a structure-constrained reinforcement learning method that sharpens textual accuracy and strictly enforces markup well-formedness beyond supervised fine-tuning alone. Extensive evaluations demonstrate the superior performance of our framework. On the OmniDocBench v1.5 and v1.6 benchmarks, ABot-OCR achieves state-of-the-art scores of 92.81 and 93.30 among all end-to-end systems, substantially narrowing the performance gap relative to strong pipeline baselines. Finally, comprehensive multilingual text recognition across ten diverse languages further confirms the robust generalizability of ABot-OCR.
>
---
#### [new 079] REVEAL: Reference-Grounded Reasoning for Multimodal Manipulation Detection
- **分类: cs.CV**

- **简介: 该论文属于多模态篡改检测任务，旨在识别伪造的图像-文本对并定位篡改区域。提出REVEAL框架，通过参考证据进行比较验证，提升检测效果和领域适应性。**

- **链接: [https://arxiv.org/pdf/2605.28459](https://arxiv.org/pdf/2605.28459)**

> **作者:** Jun Zhou; Bingwen Hu; Yaxiong Wang; Zhedong Zheng; Yongzhen Wang; Yuchen Zhang; Ping Liu
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Multimodal manipulation detection aims to simultaneously identify forged image--text pairs and localize tampered regions, yet existing methods typically rely on memorizing isolated artifacts and struggle with imperceptible manipulation traces or domain shifts. Inspired by human comparative reasoning, we reformulate this task as a reference-grounded verification problem, where authenticity is assessed by comparing a query against retrieved authentic evidence. We propose REVEAL Reference-Enabled Verification for Evidence Analysis and Localization), a framework explicitly designed for this comparative paradigm. To support this paradigm, we construct a large-scale reference library comprising 170K authentic news image--text pairs featuring over 40K public figures. Technically, REVEAL employs a difference-aware fusion mechanism to capture fine-grained discrepancies between the query and retrieved evidence. Furthermore, we introduce a task-decoupled Mixture-of-Experts (MoE) architecture to jointly execute instance-level detection and fine-grained grounding, effectively mitigating optimization conflicts between these heterogeneous objectives. Extensive experiments demonstrate that REVEAL significantly outperforms state-of-the-art methods, and notably enables \emph{training-free domain adaptation} by simply updating the reference library, offering a robust and practical solution for detecting evolving misinformation. Code is available at this https URL.
>
---
#### [new 080] DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing
- **分类: cs.CV**

- **简介: 该论文属于实时音频驱动角色控制任务，解决音频条件突变下的稳定运动生成问题。提出DiscoForcing框架，结合因果音乐编码与扩散强制模型，提升音频-运动对齐与长期一致性。**

- **链接: [https://arxiv.org/pdf/2605.28491](https://arxiv.org/pdf/2605.28491)**

> **作者:** Kaiyang Ji; Bingsheng Qian; Binghuan Wu; Kangyi Chen; Ye Shi; Jingya Wang
>
> **备注:** accepted by ICML 2026
>
> **摘要:** We study real-time audio-responsive character control as a deployment-faithful problem: strictly causal, bounded-latency streaming that must generate coherent full-body motion at interactive frame rates while the audio condition can change abruptly, including tempo shifts, drops, or user edits. Prior music-to-motion systems are largely optimized for offline generation with global context, and degrade in streaming rollouts where conditioning history becomes stale or unreliable. We introduce DiscoForcing, a streaming audio-driven diffusion framework that combines a causal music encoder that captures rhythmic structure and phase dynamics with a diffusion-forcing sequence model trained under heterogeneous noise levels across the temporal horizon. Building on this, we design a hybrid temporal schedule and a history-guided streaming sampler to explicitly trade off responsiveness against long-horizon consistency under non-stationary audio. Implemented in an end-to-end real-time interactive system with online avatar playback and humanoid deployment workflows, DiscoForcing delivers more stable long-horizon rollouts and sharper audio-motion alignment than prior baselines under matched causality and latency constraints while maintaining real-time throughput.
>
---
#### [new 081] Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players
- **分类: cs.CV**

- **简介: 该论文属于多智能体交互视频生成任务，解决多智能体世界建模中的可扩展性和一致性问题。提出Simplex Rotary Agent Encoding和Sparse Hub Attention，提升视频质量与交互性能。**

- **链接: [https://arxiv.org/pdf/2605.28816](https://arxiv.org/pdf/2605.28816)**

> **作者:** Fangfu Liu; Kai He; Tianchang Shen; Tianshi Cao; Sanja Fidler; Yueqi Duan; Jun Gao; Igor Gilitschenski; Zian Wang; Xuanchi Ren
>
> **备注:** Project Page: this https URL
>
> **摘要:** World models for interactive video generation have largely focused on single-agent settings, where future observations are generated from a single control signal. However, many generated environments require multi-agent interaction: multiple players, robots, or embodied agents act simultaneously within a shared space. Scaling world models to such settings requires a principled multi-agent design: agents should remain independently controllable, permutation-symmetric, and support efficient inference while maintaining consistency across time and perspectives. In this paper, we present our generative multi-agent world model for interactive simulation. It introduces Simplex Rotary Agent Encoding, a parameter-free extension of 3D RoPE that represents agents as vertices of a regular simplex in rotary angle space. This gives each agent a distinct phase while making all agents permutation-equivalent, enabling scalable agent identity without learned per-slot identities or a fixed agent ordering. To avoid dense all-to-all attention across agents, we further propose Sparse Hub Attention, where learnable hub tokens mediate token interaction across agents, reducing cross-agent attention cost from quadratic to linear in the number of agents. For real-time rollout, we distill a full-context diffusion teacher into a causal student that generates temporal blocks sequentially with KV caching, enabling action-responsive generation at 24 FPS. Experiments in multiplayer virtual environments show that our model improves video fidelity, action controllability, and inter-agent consistency over slot-based and dense-attention baselines, while generalizing from two to four players without additional training.
>
---
#### [new 082] Deformable Gaussian Occupancy: Decoupling Rigid and Nonrigid Motion with Factorized Distillation
- **分类: cs.CV**

- **简介: 该论文属于动态3D环境建模任务，解决弱监督下非刚性运动捕捉问题。提出DeGO框架，分离刚性和非刚性运动，提升场景理解精度。**

- **链接: [https://arxiv.org/pdf/2605.28587](https://arxiv.org/pdf/2605.28587)**

> **作者:** Yang Gao; Wuyang Li; Po-Chien Luan; Alexandre Alahi
>
> **备注:** CVPR 2026
>
> **摘要:** Understanding dynamic 3D environments is essential for safe autonomous driving, particularly when reasoning about human-centric, nonrigid agents. However, existing weakly supervised occupancy prediction frameworks predominantly assume rigid-body motion and rely on simple frame-to-frame offsets, limiting their ability to capture fine-grained deformations and maintain temporal coherence. To address this issue, we propose DeGO, a deformable Gaussian occupancy framework that unifies decoupled Gaussian deformation with factorized 4D foundation-model distillation. DeGO disentangles rigid and nonrigid motion, enabling each Gaussian primitive to evolve through both deformation and offset-based updates. In parallel, a factorized 4D distillation strategy transfers cross-camera and cross-frame knowledge from the VGGT foundation model, producing foundation-aligned features that enhance temporal consistency. Experiments on the Occ3D-NuScenes benchmark demonstrate that our method achieves state-of-the-art performance under weak supervision, delivering 13.5% gains on human-centric instances and 10.9% overall improvements. These results highlight the effectiveness of deformation-aware and foundation-guided occupancy modeling for dynamic scene understanding. The code is publicly available: this https URL
>
---
#### [new 083] A novel ordinal multi-view aggregation scheme for oak defoliation
- **分类: cs.CV**

- **简介: 该论文属于森林健康评估任务，旨在解决树冠落叶程度的有序分类问题。通过多视角卷积神经网络集成方法提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2605.28151](https://arxiv.org/pdf/2605.28151)**

> **作者:** Francisco Bérchez-Moreno; Ricardo Enrique Hernández-Lambraño; David Guijo-Rubio; Víctor Manuel Vargas; Francisco José Ruiz-Gómez; Juan Carlos Fernández; Pablo González-Moreno
>
> **摘要:** Forest decline driven by climate and biotic stressors threatens ecosystem functioning, making accurate monitoring of tree health essential. In this work, we address tree defoliation estimation as an ordinal classification problem using ground-level imagery. We propose a novel multi-view ensemble framework that aggregates predictions from Convolutional Neural Networks (CNNs) trained on different perspectives of individual trees (north, south, and crown). This approach leverages complementary visual information while preserving modelling consistency through a homogeneous ensemble design. A comprehensive evaluation is conducted by comparing multiple ordinal classification methods and analysing the contribution of each view and their combinations. Results show that modelling the ordinal structure of defoliation levels improves performance over nominal approaches, while the proposed multi-view ensemble consistently outperforms single-view and pairwise configurations. In particular, the three-view ensemble achieves the most robust and accurate predictions across all evaluation metrics. These findings highlight the potential of combining Deep Learning (DL), Ordinal Classification (OC), and multi-view aggregation for scalable, consistent, and objective forest health assessment in complex ecosystems such as Mediterranean dehesas.
>
---
#### [new 084] Which Pretraining Paradigm Better Serves Spatial Intelligence? An Empirical Comparison of Vision-Language and Video Generation Models
- **分类: cs.CV**

- **简介: 该论文属于视觉表示学习任务，比较VLM与VGM在空间智能中的表现。研究解决哪种预训练方案更优的问题，通过实验分析两者在语义标签、实例分组和3D预测上的优势。**

- **链接: [https://arxiv.org/pdf/2605.28132](https://arxiv.org/pdf/2605.28132)**

> **作者:** Haozhan Shen; Tiancheng Zhao; Kangjia Zhao; Jianwei Yin
>
> **备注:** Code is here: \href{this https URL}{this https URL}
>
> **摘要:** Spatial intelligence requires visual representations that capture both semantic objects and geometric structure in the physical world. To support this, two major pre-training schemes are now widely used as foundation backbones: Vision-Language Models (VLMs), which use language supervision to align visual observations with semantic concepts, and Video Generation Models (VGMs), which learn from temporally evolving visual worlds. However, it still remains unclear which pre-training scheme provides a better representation substrate for spatial intelligence. In this paper, we present the first systematic frozen-feature probing study of VLMs and VGMs across three representative axes of spatial intelligence: semantic tagging, instance grouping, and 3D geometry prediction. Using the lightweight probe, our framework enables a controlled comparison of what information is already encoded in frozen representations from two model families. Experimental results reveal a clear complementarity: VLMs are stronger at semantic tagging and instance grouping, while VGMs provide more accessible signals for dense geometry and camera motion. Moreover, a naive fusion of the two already yields a representation that excels at both geometry and semantics, suggesting a promising direction for building stronger spatial-intelligence backbones by effectively integrating features from both model families. Our code is available at \href{this https URL}{this https URL}.
>
---
#### [new 085] A Patient-Specific Pulmonary Arterial Tree Digital Twin to Extract Pulmonary Embolism Biomarkers
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决肺栓塞生物标志物自动提取问题。通过构建患者特异性肺动脉树数字孪生，实现栓塞负荷和分布的快速精准评估。**

- **链接: [https://arxiv.org/pdf/2605.28217](https://arxiv.org/pdf/2605.28217)**

> **作者:** Morgane des Ligneris; Nathan Painchaud; Allan Serva; Laurent Bertoletti; Pierre Croisille; Carole Frindel; Odyssée Merveille
>
> **备注:** 11 pages + 2 pages of supplementary materials. Submitted to special issue of JBHI
>
> **摘要:** Pulmonary embolism, the obstruction of a pulmonary artery by a blood clot, is one of the leading causes of acute cardiovascular syndrome. In clinical practice, therapeutic decisions after diagnosis via computed tomography pulmonary angiography rely on risk stratification, which categorizes 30-day mortality risk into three categories. This stratification depends on the right-to-left ventricular diameter ratio and blood levels of two cardiac enzymes. However, blood biomarkers are not always available in emergency settings, and manual calculation of established severity scores - such as Qanadli and Mastora - is time-consuming and rarely performed in clinical routine practice. This study introduces an automated pipeline that models a directed graph representation of the pulmonary arterial tree, labeling its hierarchical structure and characterizing pulmonary embolism. The pipeline derives image-based biomarkers, including local artery-level features (morphological information, hierarchical position, clot volume, and resulting obstruction) and global patient-level biomarkers such as automatically calculated severity scores (Qanadli and Mastora) and the total embolic volume distribution by lobes and hierarchical levels. Using artificial-intelligence-generated binary masks of arteries, emboli, lungs, and lobes, it creates a patient digital twin of the arterial structure. Validation of the pipeline through comparison to an existing pipeline, anatomical expectations, and manual severity score calculations demonstrates the pipeline's ability to automatically generate anatomically accurate digital twins and severity scores with strong agreement. This supports the potential of these image-derived biomarkers to automatically provide rapid, precise information on thrombotic burden and spatial clot distribution.
>
---
#### [new 086] Hallucination Behavior in Multimodal LLMs Across Agricultural Image Interpretation and Generation Tasks
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多模态大语言模型在农业图像解释与生成任务中的幻觉行为，旨在解决模型输出与现实不符的问题。通过分析图像到文本和文本到图像两种任务，评估模型的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.27595](https://arxiv.org/pdf/2605.27595)**

> **作者:** Partho Ghose; Al Bashir; Prem Raj; Azlan Zahid
>
> **摘要:** Large Language Models (LLMs) are being rapidly adopted in agricultural imaging applications, ranging from crop interpretation to synthetic field image generation. However, these models frequently exhibit hallucinations outputs that appear confident yet deviate from biological or environmental reality potentially leading to misinformed agronomic insights. This study investigates such hallucinations in two complementary directions: image-to-text, where LLMs interpret crop or field imagery to describe conditions such as biotic and abiotic stresses, and text-to-image, where models generate synthetic agricultural scenes based on descriptive prompts. We examine errors involving biological inconsistency, contextual inaccuracy, and agronomic implausibility, evaluating the outputs under domain-informed criteria across multiple imaging modalities. Our analysis identifies recurring hallucination patterns within both interpretive and generative tasks. In image interpretation, LLMs (e.g., Gemma, LLAVA, Qwen, and MiniCPM) achieved modest zero-shot accuracy (63 to 75 percent), whereas few-shot prompting improved performance up to 86.8 percent, exhibiting false detections and missed infections, indicating residual hallucination effects. In text-to-image tasks, advanced models such as GPT-5 and Gemini 2.5 Flash generate up to 91 percent biologically inconsistent scenes under relaxed prompt constraints, revealing fundamental weaknesses in current LLMs. This systematic assessment of visual reasoning and generation offers critical insights toward enhancing the reliability and trustworthiness of LLM-based agricultural imaging platforms.
>
---
#### [new 087] From Pixels to Words -- Towards Native One-Vision Models at Scale
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决多图像和视频理解中的像素级对齐问题。提出NEO-ov模型，实现端到端的跨帧和像素-词对应，无需外部模块。**

- **链接: [https://arxiv.org/pdf/2605.28820](https://arxiv.org/pdf/2605.28820)**

> **作者:** Haiwen Diao; Jiahao Wang; Penghao Wu; Yuhao Dong; Yuwei Niu; Yue Zhu; Zhongang Cai; Weichen Fan; Linjun Dai; Silei Wu; Xuanyu Zheng; Mingxuan Li; Yuanhan Zhang; Bo Li; Hanming Deng; Huchuan Lu; Quan Wang; Lei Yang; Lewei Lu; Dahua Lin; Ziwei Liu
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Current vision-language models (VLMs) typically stitch together separate image encoders and language decoders via multi-stage alignment, a modular framework that inevitably fragments pixel-level signals across frames and scatters early pixel-word interactions. In parallel, native VLMs, despite impressive performance on single images, remain largely unexplored in multi-image, video understanding, and spatial intelligence. Hence, we introduce NEO-ov, a native foundation model that learns cross-frame and pixel-word correspondence end-to-end, without any external encoders, auxiliary adapters, or post-hoc fusion. By eliminating module boundaries entirely, NEO-ov enables fine-grained and unified spatiotemporal modeling to emerge natively inside the model. Notably, NEO-ov largely narrows the gap to modular counterparts while excelling at fine-grained visual perception, validating that native "one-vision" architectures are not only feasible but competitive at scale. Beyond empirical performance, we unveil systematic architectural analyses and detailed training recipes to facilitate subsequent native multimodal modeling. Our code and models are publicly available at: this https URL.
>
---
#### [new 088] Reflective Dialogue between Teacher and Solver Agents for Video Question Answering
- **分类: cs.CV**

- **简介: 该论文属于视频问答任务，解决小样本下模型适应问题。通过构建教师与求解器的对话，实现推理阶段的上下文注入，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27885](https://arxiv.org/pdf/2605.27885)**

> **作者:** Takuya Murakawa; Toru Tamaki
>
> **备注:** Yhis paper serves as the technical report for the 1st Cross-Domain EgoCross Challenge @ EgoVis Workshop, CVPR 2026
>
> **摘要:** Various approaches have been proposed to adapt Vision-Language Models (VLMs) to specialized domains for Video Question Answering, including fine-tuning and in-context learning. However, acquiring task-specific knowledge at the inference phase from only a small labeled support set without fine-tuning remains a challenge. In this paper, we propose a method that achieves adaptation solely through inference-time context injection. Our method first constructs a Reflective Dialogue (RD) -- a multi-turn conversation between two agents, in which Teacher poses each support question and delivers correctness feedback, and Solver answers and provides visual grounding explanations (or reflections) for both correct and incorrect answers. This dialogue history is then used as context at the inference phase. Experiments on the EgoCross benchmark demonstrate that our method outperforms both a baseline zero-shot setting and a standard in-context learning approach that passes support set examples directly, achieving 3rd place in the Open-source Track of the 1st Cross-Domain EgoCross Challenge at the CVPR 2026 EgoVis Workshop, for which this paper also serves as a technical report.
>
---
#### [new 089] Bridging the Generalization Gap in Adverse Weather Segmentation: A Training Recipe Perspective
- **分类: cs.CV**

- **简介: 该论文属于户外场景语义分割任务，解决恶劣天气下模型泛化能力不足的问题。通过优化训练策略提升模型在不同天气条件下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.27962](https://arxiv.org/pdf/2605.27962)**

> **作者:** Cong Xu; Pu Luo; Yumei Li; Boyou Xue
>
> **摘要:** This paper describes our approach for the 8th UG2+ Workshop (CVPR 2026) Track~2, which targets semantic segmentation of outdoor scenes degraded by five weather conditions: blur, darkness, snow, haze, and glare. A central challenge we observe is a severe generalization gap -- models that perform well on the validation set often collapse on the test set. For instance, SegFormer-B5 drops 16.1 mIoU points from validation to test, suggesting that model capacity alone is insufficient for robustness. We investigate whether a carefully designed training recipe, rather than architectural complexity, can address this gap. Starting from a pre-trained SegMAN-S backbone, we systematically study the effects of domain-adaptive fine-tuning, multi-source data mixing, scene-balanced sampling, and synthetic degradation augmentation. Our final system achieves 59.9\% mIoU on the official test set while maintaining a validation-test gap of only 6.5 points -- less than half that of larger models. We analyze negative results from architectural modifications, loss function variants, and model scaling to provide practical insights for weather-robust segmentation under limited data.
>
---
#### [new 090] Can Segmentation Models Understand the World? Towards Proactive Affordance Reasoning via Visual Chain-of-Thought
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SegWorld，解决意图级指令下的场景分割问题。通过多层级视觉链式推理，模型在执行分割前主动理解场景，提升意图导向的分割效果。**

- **链接: [https://arxiv.org/pdf/2605.27764](https://arxiv.org/pdf/2605.27764)**

> **作者:** Yuchen Guo; Junli Gong; Hongmin Cai; Yiu-ming Cheung; Weifeng Su
>
> **摘要:** Recent segmentation models couple large language models (LLMs) with mask decoders to ground complex language expressions into masks, yet their instructions remain target-referential: they describe, constrain, or imply the region to be segmented. However, in real-world embodied interaction, human instructions are often at the intent-level, which includes the desired outcome without naming the region that enables it. To bridge this gap, we introduce SegWorld, where the model reasons about the scene through a multi-level visual chain-of-thought (CoT) before committing to a mask. Before receiving any instructions, it proactively observes the scene, describing visible objects and inferring plausible events they may support. Given an instruction, it continues the chain: from the object relevant to the intent, through the action that satisfies it, to the physical interaction site, the object part that affords the action. We formalize SegWorld as probabilistic inference, in which proactive observation supplies a linguistic scene context that improves mask prediction when instructions are given at the level of intent. We construct an intent-to-part benchmark for evaluating affordance-bearing part segmentation from high-level goals. Experiments show SegWorld matches instruction-driven baselines on target-referential instructions and improves substantially on intent-level ones.
>
---
#### [new 091] BiasEdit: A Training-Free Bias-Detect-and-Edit Framework for Learning Fair Visual Classifiers
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出BiasEdit框架，用于检测并编辑视觉分类中的数据偏见，解决Web数据带来的不公平问题。通过自动检测和编辑偏见属性，生成去偏数据集，提升模型公平性。**

- **链接: [https://arxiv.org/pdf/2605.28450](https://arxiv.org/pdf/2605.28450)**

> **作者:** Jungwook Seo; Yoonsik Park; Changmin Lee; Sungyong Baik
>
> **备注:** Accepted to The Web Conference 2026 (formerly WWW) as an Oral presentation
>
> **摘要:** Visual data from the Web power image classifiers, which often underpin many web services, such as recommendation and content moderation. However, the raw Web data often contain spurious correlations and social biases, and neural networks are known for their tendency to learn biases present in data. This can reinforce unfairness in web services and the web data, leading to a vicious cycle. In the context of image classification, networks learn bias attributes for a specific class when a majority of images contain the same attribute only for a given class. Hence, training a fair and debiased classifier from a biased dataset demands handling an imbalanced problem between a majority of images with bias attributes (bias-aligned samples) and a minority without (bias-conflict samples). In this work, we introduce BiasEdit, a modular framework that automatically detects bias attributes from the original dataset and edits them to construct a debiased dataset. Specifically, BiasEdit first detects unknown bias attributes via statistical dependence and mutual information analysis of visual-linguistic representations, and then explicitly edits those attributes using text-guided image editing to generate realistic bias-conflict samples. Unlike prior works that assume known bias attributes or relies on synthetic mixing, our method operates without manual annotations and can leverage off-the-shelf vision-language and editing models. BiasEdit addresses a fundamental challenge in Web-sourced visual AI, mitigating dataset-induced bias and achieving state-of-the-art debiasing performance even when training data are fully biased.
>
---
#### [new 092] SmartDirector: Keyframe-Conditioned Cinematic Video Generation with Narrative Pacing Control
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在提升视频叙事质量。解决现有方法依赖稀疏条件信号导致叙事控制不足的问题，提出SmartDirector框架，通过多关键帧增强叙事结构与节奏控制。**

- **链接: [https://arxiv.org/pdf/2605.27891](https://arxiv.org/pdf/2605.27891)**

> **作者:** Zhida Zhang; Jie Ma; Zhan Peng; Haoxue Wu; Yang Han; Jun Liang; Jie Cao; Jing Li
>
> **摘要:** The narrative quality of a video fundamentally determines its perceptual value. Although existing video generation methods can produce visually appealing content, they predominantly rely on sparse conditioning signals such as text prompts or first/last frames, which limits precise control over narrative structure and temporal pacing. In this paper, we propose SmartDirector, a framework that enhances the narrative capacity of video generation models through multiple keyframes. SmartDirector supports flexible generation scenarios including single-shot generation, multi-shot narrative synthesis, and video extension. The framework operates in two stages: Director-Gen generates a low-resolution video conditioned on the provided keyframes, and Director-SR refines the output by exploiting high-resolution keyframes as semantic anchors to recover fine-grained details. To enable robust multi-keyframe training, we construct a data pipeline that curates single-shot and multi-shot sequences from movies. Extensive experiments demonstrate that SmartDirector substantially outperforms existing state-of-the-art approaches. We will release the code to facilitate further research.
>
---
#### [new 093] ST-ColoNet: Spatio-Temporal Colon Segment Recognition via Hybrid Attention and Edge-Guided Feature Learning
- **分类: cs.CV**

- **简介: 该论文针对结肠段识别任务，解决现有方法未充分利用时间信息的问题，提出ST-ColoNet框架，结合时空注意力与边缘引导特征学习，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2605.28119](https://arxiv.org/pdf/2605.28119)**

> **作者:** Ziyi Wang; Zhengjie Zhang; Jingsheng Gao; Dahong Qian; Suncheng Xiang
>
> **摘要:** Colo-segment recognition in colonoscopy videos is a key requirement for many downstream tasks, but existing automatic recognition methods only use colonoscopy images without fully exploiting the use of temporal information, leading to poor performance. Additionally, relevant public video-based datasets are in scarcity. To tackle this problem, we curate and release a labeled dataset specifically for the task of colo-segment recognition. In addition, we propose a two-stage deep learning-based framework, Colo-Segment Recognition via SpatioTemporal Network (ST-ColoNet), for the task of colo-segment recognition from colonoscopy videos which includes the Colorlaus module that uses metric learning to optimize edge-mediated spatial feature extraction, as well as the Full-Temp module which combines three self-attention patterns to better approximate full self-attention on long colonoscopy sequences and optimize temporal feature aggregation. Through extensive ablation experiments, we show that our framework is capable of achieving state-of-the-art performance on the task of colo-segment recognition, achieving an accuracy of 81.0% and F1-score of 70.7%, which is a tremendous improvement over state-of-the-art methods.
>
---
#### [new 094] Towards Unified Vision-Language Models with Incomplete Multi-Modal Inputs
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决视频-语言模型在输入不完整时的性能问题。提出统一模型以处理不完整多模态数据，提升模型鲁棒性与适用性。**

- **链接: [https://arxiv.org/pdf/2605.27894](https://arxiv.org/pdf/2605.27894)**

> **作者:** Xiang Fang; Wanlong Fang; Changshuo Wang; Keke Tang; Daizong Liu; Siyi Wang; Wei Ji
>
> **备注:** Published in AAAI 2026
>
> **摘要:** Video-Language Models (VLMs) have demonstrated impressive multi-modal reasoning capabilities across diverse computer vision applications. However, these VLMs are task-specific and assume that both video and language inputs are complete. However, real-world VLM applications might face challenges due to deactivated sensors (e.g., cameras are unavailable due to data privacy), yielding modality-incomplete data and leading to inconsistency between training and testing data. While straightforward incomplete input can boast training generalization-ability and lead to training failure, its potential risks to VLMs regarding safety and trustworthiness have been largely neglected. To this end, we make the first attempt to propose a unified incomplete video-language model to process the incomplete multi-modal inputs. Extensive experimental results show that our method can serve as a plug-and-play module for previous works to improve their performance in various multi-modal tasks.
>
---
#### [new 095] ForestHG-Trace: Traceable Long-Horizon Ecological Reasoning over Large-Scale Forest Scenes
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于生态推理任务，解决大尺度森林场景下的多步骤生态分析问题。通过构建生态超图和LLM引导的代理，实现可追踪的长期推理与验证。**

- **链接: [https://arxiv.org/pdf/2605.27590](https://arxiv.org/pdf/2605.27590)**

> **作者:** Zihang Cheng; Duanchu Wang; Cheng Li; Jing Huang; Huanzhao Fu; Di Wang
>
> **备注:** 14 pages, 5 figures, 4 tables
>
> **摘要:** Remote sensing question answering (RS-QA) often requires more than direct semantic prediction, especially in large-scale forest scenes where ecological analysis involves multi-step filtering, numerical aggregation, neighborhood reasoning, and verifiable evidence. We introduce ForestHG-Trace, a framework for traceable long-horizon ecological reasoning over forest environments. It represents multimodal NEON forest scenes as ecological hypergraphs, where tree instances, spatial units, semantic groups, and neighborhood relations support higher-order reasoning beyond pairwise scene graphs. An LLM-guided agent then invokes deterministic tools for reading, filtering, expansion, aggregation, comparison, and auditing, producing replayable execution traces and compact evidence records rather than only free-form answers. We further construct ForestTraceQA, an executable benchmark for evaluating ecological QA across diverse task types and reasoning depths. Experiments show that ForestHG-Trace substantially improves answer accuracy and execution faithfulness over single-step baselines and scene-graph agents, while highlighting execution depth as the main bottleneck for long-horizon ecological QA.
>
---
#### [new 096] EntroAD: Structural Entropy-Guided Prompt Adaptation for Zero-Shot Anomaly Detection
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于零样本异常检测任务，旨在解决跨领域异常模式差异大的问题。提出EntroAD框架，通过结构熵引导的动态路由和双分支提示适配，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.28630](https://arxiv.org/pdf/2605.28630)**

> **作者:** Xinyu Zhao; Qingyun Sun; Jiayi Luo; Jianxin Li
>
> **摘要:** Zero-Shot Anomaly Detection (ZSAD) aims to detect anomalies in unseen domains without target-domain adaptation. Recent CLIP-based methods have shown promising performance by leveraging prompt learning and visual-text alignment. However, most existing approaches rely on a single adaptation pathway, which may be insufficient for heterogeneous anomaly patterns across domains. In practice, anomalies exhibit vastly different characteristics, ranging from salient, localized structural disruptions to subtle, diffuse, and irregular variations. To address this challenge, we propose EntroAD, a structural entropy-guided zero-shot anomaly detection framework. Unlike previous methods, EntroAD introduces a dynamic routing mechanism to process different types of anomalies with specialized adaptation strategies. Specifically, we estimate patch-level structural entropy from self-attention-induced patch relations and use it as a proxy for relational uncertainty to guide anomaly-aware token routing. Based on this routing signal, we construct anomaly-aware routed tokens to better capture anomaly cues with different structural characteristics. We further introduce a confidence-aware dual-branch prompt adaptation module to stabilize visual-text alignment while preserving CLIP's transferable prior. Extensive experiments on 10 industrial and medical benchmarks show that EntroAD achieves state-of-the-art performance in challenging cross-dataset ZSAD settings.
>
---
#### [new 097] OSP-Next: Efficient High-Quality Video Generation with Sparse Sequence Parallelism, HiF8 Quantization, and Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决高效高质量视频生成问题。提出OSP-Next模型，结合稀疏注意力、并行计算、量化和强化学习，提升生成效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.28691](https://arxiv.org/pdf/2605.28691)**

> **作者:** Yunyang Ge; Xianyi He; Zezhong Zhang; Bin Lin; Bin Zhu; Xinhua Cheng; Li Yuan
>
> **摘要:** Diffusion Transformers achieve strong video generation quality, but the quadratic cost of full attention limits efficiency. We introduce OSP-Next, an efficient text-to-video generation model that integrates sparse attention, parallelism, quantization, and reinforcement learning. OSP-Next uses a hybrid full-sparse attention architecture, where the sparse component is implemented with Skiparse-2D Attention. This fixed-pattern mechanism applies token-wise and group-wise sparse attention along spatial dimensions, leveraging locality while maintaining native compatibility with FlashAttention kernels. Based on the local equivalence of rearrangement in Skiparse-2D Attention, we further propose Sparse Sequence Parallelism (SSP), which partitions subsequences across ranks and switches sparse patterns through a single All-to-All communication. Compared with Ulysses Sequence Parallelism (SP), SSP provides a native parallel strategy for sparse attention and reduces communication volume by 75%. OSP-Next also incorporates HiF8 quantization to enable stable joint training with 8-bit quantization and sparse fine-tuning, and applies Mix-GRPO post-training to improve the performance of the sparse model. Experiments show that OSP-Next achieves a VBench total score of 83.73%, surpassing the Wan2.1 baseline. Under the 5-second 720P and 5-second 768P settings, OSP-Next achieves up to 1.64$\times$ single-GPU speedup and over 1.52$\times$ eight-GPU speedup on NVIDIA H200 GPUs. In addition, with only a 0.4% drop in VBench total score, OSP-Next-HiF8 achieves 1.69$\times$ and 2.27$\times$ speedups under the two settings on a single Ascend 950PR, demonstrating the efficiency and performance of OSP-Next across hardware platforms.
>
---
#### [new 098] SIGMA: Semantic-Difference Instruction-Grounding Mask Annotator for Text-Driven Image Manipulation Localization
- **分类: cs.CV**

- **简介: 该论文提出SIGMA，解决文本驱动图像编辑的定位问题，通过语义差异和指令引导生成像素级掩码，提升图像操作定位模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27924](https://arxiv.org/pdf/2605.27924)**

> **作者:** Peiyu Zhuang; Jianquan Yang; Haodong Li; Zhuoying Cai; Ruitao Xie; Jishen Zeng; Baoying Chen; Jiwu Huang; Xiaochun Cao
>
> **摘要:** Text-driven image editing has advanced rapidly, but reliably localizing these manipulations requires image manipulation localization (IML) models trained on large pixel-annotated datasets, and there is still no low-cost way to obtain such training data at scale. We observe that these data already exist in disguise: public editing datasets contain millions of structurally identical (original, edited) pairs to IML training samples, lacking only pixel-level masks. Recovering these masks automatically is non-trivial: pixel differencing is overwhelmed by diffusion-induced perturbations across all pixels, and instruction-only grounding localizes only what the prompt describes, missing unintended editor side-effects. We propose SIGMA (Semantic-difference Instruction-Grounding Mask Annotator), which performs semantic-feature differencing in a vision foundation backbone and injects an instruction-derived spatial prior into this visual stream via bidirectional cross-modal refinement, amplifying the difference signal at intended-edit regions when the editor faithfully realizes user intent. SIGMA is trained in two complementary stages: Stage I supervises on inpainting masks; Stage II closes the diffusion-domain shift via VAE-roundtrip noise calibration, EMA self-training, and an edit-noise disentanglement loss. SIGMA outperforms existing automatic mask generators on five benchmarks (+12.20% F1, +11.16% IoU). When applied to public editing corpora, it produces a ~1.1M IML training set that improves six diverse detectors by +18.34% F1 across five datasets, turning previously unused editing data into a model-agnostic supervisory resource for IML. We'll release the full codebase as soon as the paper is accepted.
>
---
#### [new 099] Every9D-21M: Large-Scale Real-World 9D Canonicalization of Everyday Objects
- **分类: cs.CV**

- **简介: 该论文提出Every9D-21M数据集，解决真实世界9D姿态估计问题，通过大规模实拍图像和跨实例对齐方法，提升姿态估计性能。**

- **链接: [https://arxiv.org/pdf/2605.28270](https://arxiv.org/pdf/2605.28270)**

> **作者:** Leonhard Sommer; Emil Akopyan; Adam Kortylewski
>
> **摘要:** Estimating the 9D pose of everyday objects from a single real-world image remains challenging. This is largely due to the lack of large-scale supervision. Most existing datasets either rely heavily on synthetic renderings or provide limited coverage of real-world objects: the largest real-world 9D pose dataset to date contains only 17K annotated objects across 9 categories. We address this gap with Every9D-21M, a dataset of 9D pose annotations for 21.8M real-world images from 109K object- centric videos spanning 700 everyday object categories - two orders of magnitude larger than prior real-world 9D pose benchmarks in both image and category count. To achieve this scale, we leverage object-centric videos by reconstructing object- level point clouds via multi-view geometry and aligning similar instances into a shared canonical coordinate frame. Canonical poses are manually annotated for only a small set of reference objects (fewer than 0.01% of all images) and propagated to the remaining instances via cross-instance alignment. All propagated canonical poses are then verified from multiple viewpoints. We further introduce cross-category orientation rules that induce category-level symmetries, enabling symmetry-aware evaluation. Beyond establishing dedicated training and evaluation splits as a benchmark for 9D pose foundation models, we show that training on Every9D-21M improves performance on ImageNet3D and PASCAL3D+, and generalizes to HANDAL substantially better than training on ImageNet3D. Data and code are available at this https URL.
>
---
#### [new 100] No Safe Dose: How Training Data Drives Unsafe Image Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本生成图像任务，研究训练数据对模型输出安全性的影响。通过控制数据集中的不安全图像比例，验证其对生成结果安全性的直接驱动作用。**

- **链接: [https://arxiv.org/pdf/2605.28137](https://arxiv.org/pdf/2605.28137)**

> **作者:** Felix Friedrich; Lukas Helff; Niharika Hegde; Patrick Schramowski; Kristian Kersting
>
> **摘要:** Text-to-image models trained on large-scale data often inevitably ingest unsafe content. While some people observe input-output amplifications, it remains unclear whether and how training data composition directly drives model output safety or by other factors. We shed light on this question by isolating this variable: we train the same text-to-image model on datasets that differ \emph{only} in their fraction of unsafe images (0\% to 9.6\%), across several dataset scales (100K to 8M). Then we generate images with the resulting models, and evaluate them with four independent safety classifiers. Output unsafety rises monotonically from 16.6\% at 0\% contamination to 25.5\% at 5\%. A factorial design reveals that the \emph{proportion}, not the absolute count, of unsafe training images is the operative variable. The 16.6\% irreducible baseline at zero contamination implicates the other components, e.g. frozen text encoder, as a residual safety risk -- confirmed by a text encoder ablation showing that SafeCLIP reduces this floor to 9.6\%, while the dose-response effect persists across all three encoders tested. Critically, no quality degradation in terms of FID, CLIPscore and ImageReward accompanies safety filtering. These results establish that data curation and text encoder safety are complementary and independently effective interventions. At the same time, the remaining level of unsafety poses questions for future research about emerging capabilities and compositionality.
>
---
#### [new 101] Resolution-free neural surrogates for geometric parameterization and mapping with spatially varying fields
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文属于几何参数化与映射任务，解决空间变化场的计算问题。提出一种无需固定网格的神经代理模型，高效预测映射位置。**

- **链接: [https://arxiv.org/pdf/2605.28551](https://arxiv.org/pdf/2605.28551)**

> **作者:** Yanwen Huang; Lok Ming Lui; Gary P. T. Choi
>
> **摘要:** Many imaging problems require computing spatial transformations induced by spatially varying intensity, feature, or density fields. Canonical examples include distortion correction, deformable image registration, atlas-based segmentation, and deformation-driven image analysis. These tasks can be formulated as geometric mapping problems in which the transformation is constrained to preserve local structure, control boundary behavior, or regulate angular distortion. Such formulations typically lead to variational models, diffusion processes, or elliptic partial differential equations. However, repeatedly solving high-resolution systems becomes computationally expensive when the underlying parameter fields vary across instances. In this work, we propose a resolution-free neural surrogate for geometric parameterization and mapping problems. Given a spatially varying parameter field $p:\Omega\to\mathbb{R}^m$ and query locations $\{x_i\}_{i=1}^N\subset\Omega$, the model predicts mapped locations $\{u(x_i)\}_{i=1}^N$ on arbitrary structured or unstructured point sets. To avoid dependence on a fixed grid, we use a multi-resolution geometric encoding strategy that conditions the network on coordinate-augmented samples of the parameter field. The model is trained without labeled solution data by enforcing geometry-aware constraints derived from variational energies, diffusion-based density equalization, and quasi-conformal theory. Experimental results on quasi-conformal mapping and density-equalizing mapping problems are presented to demonstrate the effectiveness of our proposed method.
>
---
#### [new 102] Generic Interpretation Approach for Transformer Models Incorporating Heterogenous Attention Structures
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型解释任务，旨在解决具有异构注意力结构的Transformer模型的可解释性问题。提出了一种解释方法，并通过实验分析其工作机制。**

- **链接: [https://arxiv.org/pdf/2605.27458](https://arxiv.org/pdf/2605.27458)**

> **作者:** Yongjin Cui; Xiaohui Fan; Huajun Chen
>
> **摘要:** Transformer has significantly propelled the development of artificial intelligence, and certainly the development of agents as well. We categorize attention structures of Transformer into two types based on the source of the input information: homogenous and heterogenous attention structures. Heterogenous attention structures, with co-attention as a typical example, process information from different sources. Heterogenous attention structure is the foundation for Transformer models to achieve more complex functions and integrate more modal information. Whether for research purposes or policy requirements, the interpretation of Transformer models with heterogenous attention structures is an important task. The fusion of information from different sources brings new challenges. Our work mainly includes two parts: method and experimentation. In terms of method, we propose an interpretation method for Transformer models with heterogenous attention structures. In terms of experimentation, based on our experimental analysis paradigm, we interpret the operating mechanisms of representative models, conduct semantic interpretation and logical interpretation.
>
---
#### [new 103] A Multiscale Kinetic Framework for Image Segmentation: From Particle Systems to Continuum Models
- **分类: cs.CV; nlin.AO**

- **简介: 该论文属于图像分割任务，旨在解决复杂图像的准确分割问题。通过构建多尺度动力学框架，将图像视为粒子系统，结合位置与特征空间的相互作用，提出一种高效分割方法。**

- **链接: [https://arxiv.org/pdf/2605.28619](https://arxiv.org/pdf/2605.28619)**

> **作者:** Horacio Tettamanti; Giulia Guicciardi; Mattia Zanella
>
> **备注:** 26 pages, 34 figures
>
> **摘要:** In this work, we present a multiscale kinetic framework for consensus-based image segmentation. By interpreting an image as a system of interacting particles, each pixel is characterised by its spatial position and an internal feature encoding color information. We introduce a coupled interaction scheme governing the evolution of particles in both position and feature spaces, from which we derive a kinetic formulation for the particle density in the space-feature domain combining transport, aggregation, and diffusion effects. Furthermore, through a suitable scaling, we obtain a first-order macroscopic model describing the evolution of the fraction of pixels carrying information on the fraction of pixels having a certain feature. Based on this reduced-complexity model, we present a data-oriented approach where we make use of particle-based optimisation techniques for the accurate segmentation of images. Numerical tests show the effectiveness of the proposed framework and its robustness under different noise conditions.
>
---
#### [new 104] Fine-Tuning Vision-Language Models for Understanding Current Damage and Scoring Priority with Quality Guard Agent
- **分类: cs.CV**

- **简介: 该论文属于桥梁损伤评估任务，旨在解决人工评估一致性差和工程师老龄化问题。通过微调视觉语言模型自动识别损伤并评分，提升评估效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.27452](https://arxiv.org/pdf/2605.27452)**

> **作者:** Takato Yasuno
>
> **备注:** 23 pages, 11 figures, 13 tables
>
> **摘要:** Bridge inspection in Japan requires mandatory visual assessments every five years, yet qualitative damage ratings (levels a-e) assigned by different engineers exhibit significant inter-rater variability -- a critical barrier to consistent infrastructure management. The aging of skilled engineers further threatens inspection capacity. This paper presents a methodology for automating bridge damage understanding and repair priority scoring using fine-tuned Vision-Language Models (VLMs). We fine-tune LLaVA-1.5-7B with QLoRA on up to 4,000 paired bridge damage images and inspection text records, then evaluate on a fixed test set of 800 images. The model outputs natural language descriptions identifying structural members and damage patterns, from which a rule-based scoring engine calculates a five-level repair priority index. A progressive training study (1k/2k/3k/4k samples) reveals that 2k training samples achieve near-optimal validation loss in only 2.9 hours of training; beyond 2k, validation loss improves by no more than 0.2% per doubling of training samples, exhibiting clear diminishing returns. Furthermore, semantic similarity on the held-out test set peaks at 3k (0.6909) and degrades at 4k (0.6739), indicating that quality-curated mid-scale data outperforms larger but noisier corpora. Inference optimization combining this http URL() and batch processing (batch_size=8) achieves 10.06 seconds per image -- a 70.2% reduction over the unoptimized baseline. Our approach contributes to data governance in bridge inspection, reduces inter-rater variability, and provides AI-assisted triage to augment expert engineers in inspection workflows. Furthermore, we introduce a two-stage Quality Guard using a fine-tuned Swallow-8B SLM to reject low-quality VLM outputs before priority scoring, preventing spurious scores from damaged or unrecognised images.
>
---
#### [new 105] Diffusion-Based Ukrainian Handwritten Text Generation with Cross-Domain Style Transfer
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于手写文本生成任务，旨在解决非拉丁语系（如乌克兰语）的风格迁移问题。通过构建数据集并改进模型，实现跨语言和跨时代的手写文本生成。**

- **链接: [https://arxiv.org/pdf/2605.27487](https://arxiv.org/pdf/2605.27487)**

> **作者:** Andrii Ahitoliev; Pavlo Berezin
>
> **备注:** 16 pages, 7 figures. Submitted to ICTERI 2026
>
> **摘要:** Handwritten text generation (HTG) conditioned on writer style has been widely studied for Latin scripts, but remains underexplored for low-resource and non-Latin writing systems, leaving open how well existing models generalise beyond the Latin domain. Cyrillic, particularly Ukrainian, lacks both large-scale writer-labeled datasets and empirical evidence of such generalisation. To address this gap, we construct a Ukrainian handwritten word dataset of 126,177 images from 308 writers using connected-component segmentation, quality filtering, and targeted oversampling of underrepresented Ukrainian characters. We retrain DiffusionPen, a MobileNetV2 triplet-loss style encoder with a CANINE-conditioned latent diffusion U-Net, on this dataset without architectural modification, testing direct transfer from Latin to Cyrillic. We evaluate cross-domain style transfer in three settings: cross-lingual transfer from IAM English samples, zero-shot transfer to an early 20th-century Ukrainian manuscript, and few-shot imitation of contemporary writers. The model produces legible, style-consistent word images, indicating that few-shot latent diffusion models generalize beyond the Latin-script domain. We release the dataset, trained models, and evaluation protocol as a reproducible benchmark for writer-aware Cyrillic HTG, providing a foundation for extending stylized HTG to other underrepresented writing systems.
>
---
#### [new 106] Enhancing Ultra-low-field MRI with Segmentation-guided Adversarial Learning
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于ULF MRI增强任务，旨在提升低场强MRI图像质量。通过分割引导的对抗学习方法，结合CycleGAN和T-REX模型，生成高质量类似高场强MRI图像。**

- **链接: [https://arxiv.org/pdf/2605.28016](https://arxiv.org/pdf/2605.28016)**

> **作者:** James Grover; Andrew Phair; Michael Ferraro; David E.J. Waddington
>
> **摘要:** Ultra-low-field (ULF) MRI offers portable and low-cost imaging but suffers from poor image quality. To address this, we present our submission to the 2025 ULF Enhancement Challenge (ULF-EnC), where the goal is to synthesise high-field-like MRIs from 64 mT scans. Our pipeline enhances ULF MRI through a combination of anatomical conditioning and model ensembling. We first generate tissue segmentation priors using a Swin UNETR trained solely on challenge-provided data. These priors condition two independent enhancement networks - a CycleGAN and a transformer-based residual enhancement model (T-REX) - each trained to synthesise 3 T-like MRIs. Outputs from both models are combined using a weighted average. Our approach produces enhanced MRIs that were comparable to high-field scans both quantitatively and qualitatively.
>
---
#### [new 107] Proprio: Latent Self-Scoring and Inference-Time Refinement for Physically Plausible Video Generation
- **分类: cs.CV**

- **简介: 该论文提出Proprio框架，解决视频生成中物理合理性不足的问题。通过自评估和优化，提升生成视频的物理可信度。**

- **链接: [https://arxiv.org/pdf/2605.28230](https://arxiv.org/pdf/2605.28230)**

> **作者:** Mariam Hassan; Kaouther Messaoud; Wuyang Li; Alexandre Alahi
>
> **摘要:** Modern video generative models produce visually impressive results, yet frequently violate basic physical principles. We propose Proprio, a training-free framework that enables a frozen video generator to assess and improve the physical plausibility of its own outputs. Inspired by proprioception, the biological sense of one's own movement, Proprio treats the model's flow residual under controlled latent perturbations as a self-scoring signal. Samples that are better explained by the generator's learned dynamics induce smaller and more stable residuals. We aggregate this signal across timesteps and perturbations, focus it on motion-relevant regions with a dynamic spatiotemporal mask, and use it for best-of-N search, gradient-based self-refinement, or both. Across text-to-video and image-to-video benchmarks, Proprio consistently improves physical plausibility, outperforming VLM-based scoring, and external world-model baselines in several settings. With TurboWan2.2, Proprio improves Physics-IQ from 32.2 to 37.5 (+16.5%) and VideoPhy2-hard physical commonsense from 45.6 to 55.0 (+20.6%). Human evaluation further shows that raters prefer Proprio-selected or refined videos for physical plausibility in roughly two-thirds of comparisons. These results suggest that frozen video generators contain actionable internal signals for evaluating and improving the physical plausibility of their own outputs.
>
---
#### [new 108] Residualized Temporal Sparse Autoencoders for Interpreting Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像生成领域，解决扩散模型中激活轨迹的解释问题。通过引入残差时间稀疏自编码器，捕捉非线性结构，提升对扩散过程的理解与分析。**

- **链接: [https://arxiv.org/pdf/2605.27813](https://arxiv.org/pdf/2605.27813)**

> **作者:** Calvin Yeung; Prathyush Poduval; Ali Zakeri; Zhuowen Zou; Mohsen Imani
>
> **摘要:** Text-to-image diffusion models generate images through an iterative denoising process, so internal neural layers produce trajectories of activations rather than single static representations. Sparse autoencoders (SAEs) have recently been used to decompose diffusion activations into interpretable feature directions, but most approaches analyze activations at individual timesteps or condition on time rather than learning directly from full activation trajectories. In this work, we introduce residualized temporal SAEs for diffusion activation trajectories. We collect activations across denoising time, fit linear predictors between neighboring timesteps, and represent each trajectory using an initial activation together with residual components not explained by these linear dynamics. Training an SAE on this residualized representation encourages sparse latents to capture structure beyond what is linearly predictable. The residualized decoder directions can be mapped back into activation space, allowing each latent to be analyzed as a feature trajectory over denoising time. Through reconstruction and ablation studies, spatiotemporal feature analysis, and qualitative steering experiments on Stable Diffusion~1.5, we show that residualized temporal SAEs provide a useful framework for studying temporally structured diffusion activations.
>
---
#### [new 109] DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出DriveWAM，用于自动驾驶的世界-动作建模任务，解决如何有效融合视频生成与动作决策的问题。通过视频扩散模型生成动作序列，提升长期规划性能。**

- **链接: [https://arxiv.org/pdf/2605.28544](https://arxiv.org/pdf/2605.28544)**

> **作者:** Chen Shi; Jinrui Xu; Shaoshuai Shi; Kehua Sheng; Bo Zhang; Li Jiang
>
> **摘要:** Pretrained foundation models have become an important basis for end-to-end autonomous driving. In contrast to vision-language models pretrained primarily on static image-text pairs, video generative models capture temporal dynamics and motion priors that are naturally suited for driving. We present DriveWAM, a driving world-action model that adapts a pretrained video diffusion transformer into an autoregressive video-action policy. DriveWAM organizes video and action streams into a unified temporal token sequence and trains them under a joint flow-matching objective, preserving the pretrained video-generation architecture while adapting its large-scale video priors to action generation. To incorporate high-level scene understanding, we introduce scene-evolving driving guidance, where a frozen VLM produces chunk-specific semantic intent to guide video-action generation. To keep long-horizon rollout bounded, we further introduce selective KV memory, which maintains bounded modality-aware video and action memory pools through relevance-redundancy cache selection at inference time. Experiments on NAVSIM and the PhysicalAI-Autonomous-Vehicles benchmark show that DriveWAM achieves strong planning performance, and a data-scaling study from 4k to 100k driving clips further confirms the scaling potential of world-action modeling for end-to-end autonomous driving.
>
---
#### [new 110] A self-supervised learning approach to deep filter banks for texture recognition
- **分类: cs.CV**

- **简介: 该论文属于纹理识别任务，旨在解决数据量少的问题。提出使用自监督预训练的卷积自编码器结合深度滤波器，提升识别效果且降低计算负担。**

- **链接: [https://arxiv.org/pdf/2605.27843](https://arxiv.org/pdf/2605.27843)**

> **作者:** Joao B. Florindo; Lucas O.Lyra; Antonio E. Fabris
>
> **摘要:** An important challenge in texture recognition is the limited amount of data for training frequently found in real-world applications. In computer vision in general, a successful strategy to mitigate this issue is the use of a pretraining stage where the neural network learns to identify relations between parts of the data in a self-supervised manner. A well-established framework in this direction is masked autoencoder. Nevertheless, these models usually rely on computationally intensive architectures, such as vision transformers. In the particular case of texture images, most of the relevant information is compacted within a delimited area around each pixel, which suggests that capturing long-range dependence via the attention mechanism may be unnecessary. Based on that assumption, here we propose a framework where the pretraining model is a convolutional autoencoder. To leverage the rich information conveyed by texture patterns, we employ deep filters coupled with Fisher vector pooling. In this way, we improve the performance of texture recognition without adding significant computational burden. Our approach is compared with several state-of-the-art methods in different texture databases, confirming its potential both in terms of classification accuracy and computational complexity.
>
---
#### [new 111] Personal Visual Memory from Explicit and Implicit Evidence
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于个性化AI长期记忆任务，解决现有系统忽视视觉中隐含用户信息的问题。提出VisualMem架构，融合视觉与文本记忆，提升个性化记忆效果。**

- **链接: [https://arxiv.org/pdf/2605.28806](https://arxiv.org/pdf/2605.28806)**

> **作者:** Viet Nguyen; Thao Nguyen; Vishal M. Patel; Yuheng Li
>
> **备注:** Project Page: this https URL
>
> **摘要:** Long-term memory is increasingly important for personalized AI agents, yet existing benchmarks and methods remain largely text-centric. Even when images are included, the user-specific information needed for later questions is typically recoverable from text alone, and most memory systems reduce image turns to generic captions. Yet images often carry personal information that text rarely states -- both explicit evidence, such as recurring user-associated entities, and implicit evidence, such as latent user facts inferred from visual or multimodal cues. We introduce a benchmark for personal visual memory that targets both forms of evidence, and propose VisualMem, a hybrid visual--text architecture that augments a text-memory backend with a structured personal visual memory module. Rather than collapsing images into captions, VisualMem uses conversational context to resolve identity, ownership, and durable user facts. Experiments show that VisualMem substantially outperforms prior memory systems on our benchmark while remaining competitive on standard text-memory benchmarks, indicating that personal visual memory is a distinct and important component of long-term memory for personalized AI agents.
>
---
#### [new 112] EchoAvatar: Real-time Generative Avatar Animation from Audio Streams
- **分类: cs.CV**

- **简介: 该论文属于语音驱动的3D角色动画生成任务，旨在实现实时、高质量的全身运动合成。针对现有方法在实时性和跨域适应性上的不足，提出一种统一的流式架构，结合强化学习与语义控制，提升生成效果与灵活性。**

- **链接: [https://arxiv.org/pdf/2605.28272](https://arxiv.org/pdf/2605.28272)**

> **作者:** Bohong Chen; Yumeng Li; Yinglin Xu; Youyi Zheng; Yanlin Weng; Kun Zhou
>
> **备注:** SIGGRAPH 2026; Project Page: this https URL
>
> **摘要:** Real-time synthesis of high-fidelity 3D character motion from audio is a pivotal component for next-generation interactive avatars and virtual assistants. However, most existing approaches are limited to offline processing of complete audio sequences or are constrained to specific domains, rarely handling both speech and music effectively. In this paper, we introduce a novel framework designed to generate continuous, coherent full-body motion from streaming speech and music with low latency. Central to our approach is a unified streaming architecture capable of synthesizing continuous motion from incremental audio inputs. We employ a robust training strategy that enforces strong audio dependency, allowing the model to seamlessly generalize across conversational speech and rhythmic music without requiring explicit domain labels or mode switching. Additionally, we explored Reinforcement Learning to refine the quality of online generation. Furthermore, we bridge reactive animation with intent-driven behavior via a tool-call interface that allows upstream Large Language Models to inject explicit semantic control. By combining this controllability with stream audio-driven synthesis, our framework serves as a plug-and-play solution for transforming voice agents into interactive humanoid avatars. Extensive experiments demonstrate that our method outperforms state-of-the-art realtime baselines in motion quality and synchronization while maintaining the flexibility required for live deployment. Our code, pre-trained models, and videos are available at this https URL.
>
---
#### [new 113] EventShiftFlow: Towards Hardware-efficient FPGA-based Flow Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于运动估计任务，旨在解决事件相机在FPGA上高效实现流估计的问题。提出一种基于固定时间窗和1位网格的并行速度估计算法，无需浮点运算和迭代优化。**

- **链接: [https://arxiv.org/pdf/2605.28312](https://arxiv.org/pdf/2605.28312)**

> **作者:** Arianna Alonso Bizzi; Fernando Cladera; C. J. Taylor
>
> **备注:** 10 pages, 5 figures. Accepted to the IEEE ICRA 2026 Workshop on Challenges and Opportunities of Neuromorphic Field Robotics and Automation
>
> **摘要:** Event-based vision sensors offer asynchronous, high-temporal-resolution measurements that are attractive for low-latency robotic perception, but many event-based motion estimation methods are computationally intensive and difficult to map to FPGA hardware. We present a streaming velocity estimator that discretizes asynchronous events into fixed-duration time bins, constructs a 1-bit spatial occupancy grid, and evaluates multiple velocity hypotheses in parallel using only fixed-width integer logic - shift registers, counters, comparators, and small LUT-mapped multiplies - with no dividers and no DSP blocks. It requires no frame reconstruction, no floating-point arithmetic, and no iterative optimization. The method deliberately trades dense sub-pixel optical flow for a sparse, quantized velocity estimate at each active pixel, suited to low-latency tasks such as reactive obstacle avoidance on size-, weight-, and power-constrained platforms. On noisy synthetic data with known ground-truth velocities, the method recovers both magnitude and direction, with magnitude estimates being most challenged when objects of different velocities intersect. On a real event-camera sequence, directional accuracy reaches 99.5% across all four evaluated motion segments, with performance remaining robust across occupancy densities in the 10-40% range. We characterize the algorithm's density-dependent behavior, present a parameter sensitivity analysis, show that the proposed datapath requires less than 2 kB of storage, and implement a single-axis prototype on a low-cost Xilinx Artix-7.
>
---
#### [new 114] OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于多模态验证任务，旨在提升基础模型的可靠性和可解释性。通过符号化元验证和解耦强化学习，解决传统验证方法依赖模型奖励和效果不佳的问题。**

- **链接: [https://arxiv.org/pdf/2605.28805](https://arxiv.org/pdf/2605.28805)**

> **作者:** Xinchen Zhang; Bowei Liu; Jiale Liu; Chufan Shi; Yizhen Zhang; Junhong Liu; Youliang Zhang; Zhiheng Li; Yujiu Yang; Ling Yang
>
> **备注:** ICML 2026. Project: this https URL
>
> **摘要:** Visual outcomes are increasingly central to multimodal large language models, making reliable and fine-grained verification essential for scaling generalist foundation models. In this work, we investigate multimodal meta-verification, which leverages verifier-generated rationales rather than decision-only signals, and explore how to effectively incorporate meta-verification feedback into multimodal verifier training. We identify two key findings. First, symbolic verifier outputs (e.g., bounding boxes) outperform textual explanations as meta-verification rationales, enabling efficient rule-based reinforcement learning rewards while avoiding reliance on model-based rewards from auxiliary judge models. Second, decoupling reinforcement learning objectives for binary judgment and meta-verification substantially outperforms joint reward optimization, due to intrinsic differences in output structure and learning dynamics. Based on these insights, we train OmniVerifier-M1, a generalist visual verifier leveraging symbolic meta-verification and decoupled reinforcement learning. OmniVerifier-M1 provides robust verification and fine-grained error localization, and further enables M1-TTS, a verifier-driven agentic generation system achieving dynamic region-level self-correction. This approach paves the way for more reliable, interpretable, and fine-grained multimodal verification, supporting safer and more controllable foundation model deployment.
>
---
#### [new 115] OralAgent: Integrating Reasoning, Tools, and Knowledge for Interactive Dental Image Analysis
- **分类: cs.CL; cs.CV; cs.MA**

- **简介: 该论文提出OralAgent，解决口腔影像分析中多模态任务整合问题，融合推理、工具与知识，提升临床实用性。**

- **链接: [https://arxiv.org/pdf/2605.27378](https://arxiv.org/pdf/2605.27378)**

> **作者:** Jing Hao; Siyuan Dai; Yongxin Zhang; Yuci Liang; Jiamin Wu; Jiahao Bao; Yuxuan Fan; Zanting Ye; Yanpeng Sun; Xinyu Zhang; Ming Hu; Liang Zhan; James Kit Hon Tsoi; Linlin Shen; Junjun He; Kuo Feng Hung
>
> **备注:** 14 pages, 7 figures, 6 tables
>
> **摘要:** Dental image analysis plays a pivotal role in supporting accurate diagnosis and treatment planning in oral healthcare. Although recent advances have produced dental AI models for specific tasks and individual imaging modalities, their isolated designs limit practical use in real-world clinical workflows. In this paper, we present OralAgent, the first dental-specialized AI agent that unifies multimodal reasoning, tool-based decision-making, and knowledge-grounded retrieval within an end-to-end automated framework. It integrates 22 visual analysis tools and 368 widely-used classical dental textbooks, enabling autonomous reasoning, planning, tool use, knowledge retrieval, and multi-step workflow execution. Furthermore, we introduce OralCorpus, a large-scale, high-quality bilingual textual resource containing 134.8M tokens curated for dental retrieval-augmented generation (RAG). To evaluate models' multidisciplinary dental knowledge, we construct OralQA-ZH, a Chinese multiple-choice question benchmark consisting of 798 items across eleven oral subspecialties. Extensive experiments demonstrate that OralAgent achieves state-of-the-art performance on the MMOral-Uni, MMOral-OPG, and OralQA-ZH benchmarks, highlighting its effectiveness, interpretability, and adaptability in real-world clinical settings. The code and models are publicly available at this https URL.
>
---
#### [new 116] Reading or Guessing? Visual Grounding Failures of Vision-Language Models for OCR in Ancient Greek Editions
- **分类: cs.CL; cs.AI; cs.CV; cs.DL**

- **简介: 该论文属于OCR任务，研究VLM在古希腊文本识别中的视觉 grounding 问题。通过对比实验与图像扰动分析，揭示模型依赖语言先验而非视觉证据的现象，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2605.27750](https://arxiv.org/pdf/2605.27750)**

> **作者:** Antonia Karamolegkou; Nicolas Angleraud; Benoît Sagot; Thibault Clérice
>
> **摘要:** Recent work has shown that Vision-Language Models (VLMs) used for optical character recognition (OCR) can generate plausible but visually unsupported text, suggesting reliance on language priors. Comparing open-weight VLMs with traditional OCR baselines on low-resource Ancient Greek critical editions, we show that VLM errors often remain fluent even when wrong, producing plausible Greek substitutions where traditional engines produce local recognition noise. To analyze visual evidence during decoding, we introduce controlled image perturbations and token-level grounding measures based on conditional versus image-free decoding distributions. Under character-level perturbations, VLMs diverge sharply from the perturbed ground truth while traditional OCR remains comparatively faithful; however, token-level analysis shows that prior reliance is model-specific: in an OCR-specialist model, fluent lexical errors are produced with little reliance on the image, whereas general-purpose VLMs remain conditioned on the visual input even when wrong. Decode-time interventions fail to reliably restore grounding, while post-OCR language-model correction improves several systems only by repairing text after generation. Our results extend prior evidence of OCR language-prior reliance to low-resource historical documents and a broader set of models, showing that fluent output is not necessarily visually grounded and motivating interpretability-driven evaluation beyond aggregate accuracy.
>
---
#### [new 117] Explicit Critic Guidance for Aligning Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于扩散模型对齐任务，解决非可导目标下的信用分配和稳定优化问题。提出一种状态对齐的actor-critic框架，利用扩散模型自身作为价值函数，提升训练稳定性与生成质量。**

- **链接: [https://arxiv.org/pdf/2605.27736](https://arxiv.org/pdf/2605.27736)**

> **作者:** Zhengyang Liang; Qihang Zhang; Ceyuan Yang
>
> **摘要:** Online reinforcement learning is becoming increasingly important for aligning diffusion models with non-differentiable objectives. However, existing methods still face limitations in assigning fine-grained credit along denoising trajectories and in realizing stable value-based optimization. We propose a state-aligned latent actor-critic framework for diffusion post-training, in which the diffusion model serves as its own timestep-conditioned value function and predicts values directly on noisy latent states. This enables trajectory-level PPO training, supports stable actor-critic optimization with simple conditioning and value pretraining strategies, and naturally allows the learned critic to be reused for inference-time steering. We further extend the framework to multi-reward optimization, where joint training with complementary rewards helps alleviate reward hacking. Across both UNet- and DiT-based backbones, our method consistently outperforms prior group-relative RL and actor-critic baselines on single-reward and multi-reward benchmarks, while test-time steering provides additional gains in generation quality.
>
---
#### [new 118] Geometry-Correct Diffusion Posterior Sampling with Denoiser-Pullback Curvature Guidance and Manifold-Aligned Damping
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于图像重建任务，解决扩散采样中的数据一致性不稳定问题。通过引入阻尼高斯-牛顿校正和曲率引导，提升采样效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.27990](https://arxiv.org/pdf/2605.27990)**

> **作者:** Seunghyeok Shin; Minwoo Kim; Dabin Kim; Hongki Lim
>
> **备注:** Code: this https URL
>
> **摘要:** Diffusion posterior sampling conditions diffusion priors on measurements, but data-consistency updates are typically scaled by hand-tuned guidance weights and can destabilize sampling under stiff, operator-dependent curvature. We replace scalar guidance with a per-noise-level damped Gauss--Newton correction computed in diffusion-state coordinates. The correction pulls likelihood gradients back through the denoiser, uses a one-sided curvature model that avoids forward denoiser Jacobians, and applies diffusion-calibrated rank-one damping aligned with the denoiser residual. Each correction is solved with matrix-free GMRES using automatic differentiation, and sampling proceeds with a variance-preserving Langevin transition with a closed-form drift/noise split. On FFHQ and ImageNet across inverse problems, it achieves competitive PSNR/SSIM/LPIPS while running markedly faster than most of the compared baselines; on accelerated MRI reconstruction, it achieves the best PSNR/SSIM among the compared baselines.
>
---
#### [new 119] Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Uni-LaViRA，解决跨任务、跨机器人的具身导航问题，通过语言-视觉-机器人动作统一翻译实现零样本泛化。**

- **链接: [https://arxiv.org/pdf/2605.27582](https://arxiv.org/pdf/2605.27582)**

> **作者:** Hongyu Ding; Sizhuo Zhang; Ziming Xu; Jinwen Guo; Hongxiu Liu; Xingzhi Cheng; Zixuan Chen; Haifei Qi; Duo Wang; Hao Xu; Jieqi Shi; Yifan Zhang; Jing Huo; Jian Cheng; Yang Gao; Jiebo Luo
>
> **备注:** Project page: this https URL
>
> **摘要:** Embodied navigation requires an agent to map language and visual observations to a stream of spatial actions that drive a real robot through environments it has never seen. The dominant approach has been to scale vision-language-action (VLA) foundation models on ever-larger collections of robot trajectories. This paper argues that, for navigation specifically, generality can be obtained structurally, not only through data scale. The underlying decision structure of navigation reduces to a single Language-Vision-Robot Actions Translation. The language action emits semantic-level directional command and the vision action emits a pixel-level visual target. Both outputs lie inside the natural output manifold of pretrained multimodal large language models (MLLMs), so the task can be reasoned about by an agent rather than learned from robot data. Therefore, we present Uni-LaViRA, a unified agentic architecture that extends the same insight to four task families (VLN-CE, ObjectNav, EQA, and Aerial-VLN) and to four heterogeneous real robots (Wheeled, Quadruped, Humanoid robot, and a self-built UAV) in a zero-shot manner. Two agent-loop mechanisms make this unification practical. TODO List Memory (TDM) rewrites a structured checklist of pending sub-goals at every step, reciting the unfinished items back into the agent's most recent attention window. Second Chance Backtrack (SCB) rolls the robot back to the pre-error state and conditions the agent's next plan on the failed sub-trajectory, turning single-pass navigation into a self-correcting process. With zero training effort, Uni-LaViRA reaches 60.7% SR on VLN-CE R2R, 51.3% on VLN-CE RxR, 77.7% on HM3D-v2, 60.0% on HM3D-OVON, 54.7% on MP3D-EQA, and 40.0% on OpenUAV, matching or even surpassing recent training navigation foundation models that consume millions of samples and thousands of GPU-hours.
>
---
#### [new 120] ClothTransformer: Unified Latent-Space Transformers for Scalable Cloth Simulation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于物理模拟任务，旨在解决 cloth simulation 的挑战。提出 ClothTransformer 框架，通过 Transformer 实现统一、可扩展的模拟，提升精度并处理多种场景。**

- **链接: [https://arxiv.org/pdf/2605.27852](https://arxiv.org/pdf/2605.27852)**

> **作者:** Yu Zhang; Yidi Shao; Wenqi Ouyang; Yushi Lan; Zhexin Liang; Chengrui Wu; Xudong Xu; Xingang Pan
>
> **摘要:** Unified and scalable Transformers have recently achieved remarkable success in modeling diverse phenomena traditionally associated with computer graphics, such as 3D visual effects, rendering processes, and motion in videos. In this work, we take a step further by investigating whether modern Transformer techniques can tackle the challenging task of cloth simulation. To this end, we present ClothTransformer, a framework that reformulates cloth simulation as autoregressive sequence modeling in a learned latent space. Existing neural cloth simulators are largely specialized to single scenarios, intrinsically coupled to the mesh discretization, and lack robust collision handling. Our approach addresses these limitations through three contributions: (1) a unified Transformer architecture that handles diverse scenarios -- body-driven garments, robotic manipulation, and free-fall collisions -- under a single model and achieves approximately $4$--$9{\times}$ lower error than prior state-of-the-art methods across all scenarios; (2) a scalable latent-space formulation that compresses arbitrary-resolution meshes into a fixed-size set of latent tokens, making temporal dynamics computation independent of mesh resolution; and (3) a diverse-scenario high-fidelity penetration-free dataset of ${\sim}$493.4k frames spanning all three settings, which enables a differentiable Continuous Collision Detection (CCD) module to suppress penetration artifacts.
>
---
#### [new 121] Benchmarking Ultrasound Foundation Models for Fetal Plane Classification
- **分类: eess.IV; cs.CV; cs.LG; eess.SP; stat.AP**

- **简介: 该论文属于胎儿平面分类任务，旨在解决超声图像解读依赖操作者的问题。通过对比不同基础模型的性能，评估其在有限标注数据下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.27796](https://arxiv.org/pdf/2605.27796)**

> **作者:** Leya Barrientos; Yuexi Du; Nicha C. Dvornek
>
> **摘要:** Ultrasound is widely used in obstetric care due to its safety, accessibility, and real-time imaging. However, interpretation remains operator-dependent and susceptible to noise and artifacts. Deep learning models have shown strong performance to solve these problem, but they typically require large annotated datasets that are difficult to obtain in clinical ultrasound. Foundation models (FMs) offer an alternative, using a large number of ultrasound images to learn transferable representations that can generalize with limited labeled data. This work presents a comprehensive benchmark of ultrasound-specific FMs for fetal plane classification. We evaluated four ultrasound FMs (USFM, MOFO, UltraSAM, FetalCLIP) against two CNN baselines (ResNet50, EfficientNet-V2) and a ViT (DINOv3) pretrained on natural images. We trained all models under two complementary settings: full fine-tuning and linear probing with a frozen encoder. All models were trained using 5-fold patient-level cross-validation on a Spanish fetal ultrasound dataset and tested on both in-domain data and an external African cohort to assess cross-population generalization. We found that FetalCLIP achieved the best results in the linear probing setting (F1 = 0.9261 for in-domain, F1 = 0.9731 for out-of-domain), while USFM performed best in the full fine-tuning setting (F1 = 0.9476 for in-domain, F1 = 0.9515 for out-of-domain). MOFO and UltraSAM degraded most in both settings, underperforming natural image pretrained models in some cases. These findings highlight how the choice of pretrained model strongly affects fetal plane classification performance, since different pretraining objectives lead to different levels of transferability.
>
---
#### [new 122] GUI Agents for Continual Game Generation
- **分类: cs.SE; cs.AI; cs.CV; cs.HC**

- **简介: 该论文属于游戏生成任务，解决生成可玩游戏的问题。通过引入GUI代理作为评估者和测试者，提升游戏生成质量，实验表明方法有效。**

- **链接: [https://arxiv.org/pdf/2605.28258](https://arxiv.org/pdf/2605.28258)**

> **作者:** Yixu Huang; Bo Li; Na Li; Zhe Wang; Kaijie Chen; Haonan Ge; Qingyi Si; Yuanzhe Shen; Ruihan Yang; Guangjing Wang; Hongcheng Guo
>
> **摘要:** Generating a game is not the same as making one that can be played. Despite advances in code generation, existing approaches treat game generation as one-shot translation from prompt to artifact, leaving interaction-level failures undetected. We argue that evaluating and improving game generation requires a player, and study two roles for graphical user interface (GUI) agents in this process: (1) as an objective evaluator, for which we introduce PlaytestArena, a new evaluation environment that pairs 200 browser-based game generation tasks across eight genres with rubrics of expected in-play behaviors, adjudicated by a GUI agent that loads each build in a browser and plays it; and (2) as a subjective playtester, for which we propose Play2Code, where a game agent and a GUI agent operate in a sustained loop with shared memory, turning game generation into a dialogue between coding and playing. Our experiments show that even frontier models struggle to generate playable games directly, while Play2Code achieves a 66.8\% rubric pass-rate, improving over single-pass and agentic-coding baselines by 37.1 and 14.6 points respectively. Further analysis shows that GUI playtester feedback is more traceable than a human report, yet idiosyncratic in ways reminiscent of human testers, establishing game playtesting as a critical testbed for interactive code generation. Our project website is available at this https URL.
>
---
#### [new 123] POINav: Benchmarking and Enhancing Final-Meters Arrival in Real-World Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决真实场景中精准到达POI的“最后米”问题。构建了POINav-Bench基准和POINav-Dataset，并提出脑-行动框架提升导航精度。**

- **链接: [https://arxiv.org/pdf/2605.28237](https://arxiv.org/pdf/2605.28237)**

> **作者:** Ruiyan Gong; Meisheng Zhang; Yuxiang Zhao; Mingchao Sun; Yanfen Shen; Zedong Chu; Zhining Gu; Wei Guo; Xiaolong Cheng; Qiming Li; Kangning Niu; Yanqing Zhu; Xiaolong Wu; Tianlun Li; Mu Xu
>
> **备注:** 25 pages, 9 figures
>
> **摘要:** Real-world navigation is fundamentally driven by Points of Interest (POIs), yet reaching a precise POI remains a critical "final-meters" challenge. Existing Vision-Language Navigation (VLN) benchmarks of POI-goal navigation often suffer from coarse granularity or significant sim-to-real gaps due to generated scene. To bridge this gap, we present POINav-Bench, the first benchmark designed for closed-loop evaluation of real-world POI-goal navigation. It comprises 11 commercial areas reconstructed from real-world captures using 3D Gaussian Splatting (3DGS), covering 126,398 $m^{2}$ in total and spanning 163 distinct POIs. With traversability-aware annotations and reference trajectories, POINav-Bench enables high-fidelity evaluation of navigation agents in realistic, POI-rich real-world environments. Building on this, we propose the POINav Brain-Action Framework where a Brain module performs POI-grounded reasoning to guide an Action module in predicting continuous waypoints for real-world execution. We further curate the POINav-Dataset, containing 70K real-world signage-entrance pairs. Experiments show that our framework provides a viable path toward refining real-world POI-goal navigation.
>
---
#### [new 124] Disentangling Adversarial Prompts: A Semantic-Graph Defense for Robust LLM Security
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属于LLM安全任务，解决 adversarial prompts 带来的威胁。提出APD框架，通过语义分解、图分类和轻量分类器，有效识别并中和恶意提示。**

- **链接: [https://arxiv.org/pdf/2605.27823](https://arxiv.org/pdf/2605.27823)**

> **作者:** Xiang Fang; Wanlong Fang
>
> **备注:** Published in AAAI 2026
>
> **摘要:** Large Language Models (LLMs) are increasingly vulnerable to adversarial prompts that exploit semantic ambiguities to bypass safety mechanisms, resulting in harmful or inappropriate outputs. Such attacks, including jailbreaking and prompt injection, pose significant risks to the integrity and availability of LLMs in security-critical applications. This paper proposes the Adversarial Prompt Disentanglement (APD) framework, a novel defense mechanism that proactively identifies and neutralizes malicious components in input prompts before they are processed by the LLM. The APD framework integrates three key innovations: (1) a mutual information-based semantic decomposition method to isolate adversarial and benign prompt components, ensuring statistical independence; (2) a graph-based intent classification approach that leverages spectral analysis to detect malicious patterns in prompt semantics; and (3) a lightweight transformer-based classifier trained on real-world datasets of toxic and jailbreaking prompts, enabling efficient and accurate adversarial intent detection. Evaluated on diverse datasets containing adversarial prompts, APD demonstrates superior robustness, reducing harmful output generation by over 85\% while maintaining negligible impact on model performance. The framework's computational efficiency supports real-time deployment, making it a practical solution for securing LLMs. Our work addresses critical challenges in machine learning security on novel attacks and integrity methods for ML systems, and offers a scalable, ethically grounded defense against prompt-based adversarial threats.
>
---
#### [new 125] Turning Video Models into Generalist Robot Policies
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决跨身体结构的通用机器人策略问题。通过解耦视频规划与逆动力学模型，实现高效、可扩展的零样本控制。**

- **链接: [https://arxiv.org/pdf/2605.27817](https://arxiv.org/pdf/2605.27817)**

> **作者:** Sizhe Lester Li; Evan Kim; Xingjian Bai; Tong Zhao; Tao Pang; Max Simchowitz; Vincent Sitzmann
>
> **备注:** project page: this https URL
>
> **摘要:** Video generative models have emerged as a promising robotics backbone, capable of generating videos that depict the completion of complex tasks across embodiments and environments. Recent work proposes robot foundation models that jointly predict future observations and actions by finetuning video models with action-labeled data. In this paper, we test the limits of an alternative approach: leave the video planner as-is while training an embodiment-specific inverse dynamics model (IDM). This decoupling offers several natural benefits: the video planner remains embodiment-agnostic, different video models can be interchanged easily without re-training the IDM, and the IDM can be independently trained with readily available self-play data. We present a closed-loop, video-to-action policy that combines an action-free video world model with a carefully-designed IDM based on the robot embodiment Jacobian. We demonstrate that our IDM design is both data-efficient and scalable to high-dimensional action spaces. Our policy, which we coin the Video-to-Embodied Robot Action Model (VERA), achieves strong performance across simulated and real-world benchmarks, including zero-shot Panda arm manipulation and 16-DoF Allegro-hand dexterous cube re-orientation. The same video planner can be used across multiple embodiments by pairing it with different embodiment-specific IDMs. Our results show that decoupled video planning plus faithful video-to-action translation is a viable alternative route towards zero-shot, cross-embodiment, and generalizable robot control. More results are available on our project website: this https URL.
>
---
#### [new 126] NL-MambaXCT: Self-Supervised Nested-Learning Mamba for Nomex Honeycomb X-ray CT Defect Classification
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于XCT缺陷分类任务，解决人工标注依赖和数据不足问题。提出NL-MambaXCT框架，结合自监督和嵌套学习，提升分类精度。**

- **链接: [https://arxiv.org/pdf/2605.27454](https://arxiv.org/pdf/2605.27454)**

> **作者:** Ghaleb Aldoboni; Lobna Nassar; Fakhri Karray; Reem Alshamsi
>
> **摘要:** X-ray computed tomography (XCT) is widely used for non-destructive testing of Nomex honeycomb structures in aerospace manufacturing, but industrial inspection still relies heavily on manual interpretation and supervised models trained on limited labeled data. This work introduces NL-MambaXCT, a Mamba-based framework that combines self-supervised masked image modelling with a Nested Learning (NL) formulation for automated, label-efficient defect classification from production XCT slices. The backbone is a four-stage 2D encoder with RegNet convolutional blocks in the early stages and Mamba-based sequence mixing with attention in the deeper stages. It is pretrained by masked image modelling on 19,961 unlabeled industrial XCT slices and fine-tuned on 2,000 relabeled Nomex XCT slices split by production order. NL is instantiated through two-timescale parameter dynamics: selected projections maintain slow exponential-moving-average traces alongside fast weights, while a deep-momentum optimizer introduces an additional slow parameter-update trajectory. On the held-out test set, the MIM-pretrained NL-MambaXCT model achieves 96.91% accuracy and 96.8% macro F1, outperforming CNN, attention, and single-timescale Mamba baselines by 3.11--10.31 percentage points in accuracy. The results suggest that combining masked self-supervision with NL-style fast/ slow learning dynamics is a promising strategy for robust defect classification in Nomex honeycomb XCT inspection.
>
---
#### [new 127] Deep Learning Strain Estimation: Is Physics-Based Simulation the Solution?
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决心肌应变估计的准确性问题。通过引入基于物理的仿真与深度学习结合的方法，提升区域应变的估计精度。**

- **链接: [https://arxiv.org/pdf/2605.28697](https://arxiv.org/pdf/2605.28697)**

> **作者:** Thierry Judge; Nicolas Duchateau; Andreas Østvik; Khuram Faraz; Anders Austlid Taskén; Sigve Karlsen; Thor Edvardsen; Harald Brunvand; Md Abulkalam Azad; Havard Dalen; Bjørnar Grenne; Gabriel Kiss; Pierre-Yves Courand; Lasse Lovstakken; Pierre-Marc Jodoin; Olivier Bernard
>
> **备注:** 10 pages
>
> **摘要:** Speckle tracking echocardiography (STE) is the clinical standard for myocardial strain estimation. Despite good performance on global strain (GLS), its accuracy for regional strain remains limited, even though this biomarker is highly relevant for early diagnosis and the characterization of subtle abnormalities. from clinical data. Deep learning is a promising alternative, but its development is constrained by the lack of reliable motion references. Existing solutions rely either on STE-derived labels or on simulations generated by physics-based models, but these synthetic sequences still have limited realism compared with clinical this http URL this paper, we propose a novel simulation strategy that incorporates speckle decorrelation measures from real videos and uses an iterative refinement process to improve the motion realism in the simulations. We created an open-source photorealistic dataset of 1,478 videos with reference motion, which was used to train an echocardiographic motion estimation algorithm. The proposed method achieves unmatched performance on global and regional strain, notably reaching a GLS variability of 1.42% in an inter-expert setting compared to 1.78% for the clinical reference.
>
---
#### [new 128] On the Equivariant Learning of the $Q$-tensor Order Parameter
- **分类: cond-mat.soft; cs.CV; cs.LG**

- **简介: 该论文属于机器学习任务，旨在预测二维Q-张量有序参数。通过构建等变神经网络，解决对称性约束下的预测问题，并验证其在不同旋转对称性下的性能优势。**

- **链接: [https://arxiv.org/pdf/2605.27679](https://arxiv.org/pdf/2605.27679)**

> **作者:** Julia Navarro; Mark Wilkinson
>
> **备注:** 15 pages (excluding 7-page appendix); 6 figures
>
> **摘要:** We construct and evaluate group-equivariant neural networks for the prediction of the two-dimensional $Q$-tensor order parameter of nematic liquid crystals from synthetically generated microscopic textures. Seven architectures, equivariant to cyclic groups $C_k$ of order $k$ for $k=4,\,8,\,16,\,32,\,64,\,128,\, 256$, are built using a combination of weight-sharing constraints, equivariant activations and regularization techniques. To do this, we construct rotation-like permutation matrix groups with elements $\varrho_{C_k}(g)$ that act on row-wise vectorized images, thereby approximating a $\frac{2\pi}{k}$ rotation of the circular subdomain on square images. We show that all seven equivariant models satisfy the $Q$-tensor equivariance constraint to within single-precision floating point accuracy. Comparing against approximate parameter-matched non-equivariant benchmarks, with and without data augmentation, we find that the equivariant models consistently achieve lower errors and generalize more robustly to unseen defect configurations. Performance increases with group order, suggesting that the incorporation of finer rotational symmetry leads to lower errors.
>
---
#### [new 129] Diffusion Large Language Models for Visual Speech Recognition
- **分类: cs.AI; cs.CV; eess.AS**

- **简介: 该论文属于视觉语音识别任务，解决传统方法因上下文不足导致的误判问题。提出DLLM-VSR框架，通过扩散模型实现灵活解码，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2605.28456](https://arxiv.org/pdf/2605.28456)**

> **作者:** Jeong Hun Yeo; Chae Won Kim; Hyeongseop Rha; Yong Man Ro
>
> **备注:** Code: this https URL
>
> **摘要:** Existing Visual Speech Recognition (VSR) systems commonly rely on left-to-right autoregressive decoding, which can force premature decisions on visually ambiguous tokens before sufficient context is available. We propose DLLM-VSR, to the best of our knowledge, the first Diffusion Large Language Model (DLLM)-based VSR framework, formulating transcription as iterative masked denoising with flexible-order decoding. With confidence-based unmasking, DLLM-VSR commits high-confidence positions early and uses the committed tokens as bidirectional context to refine ambiguous ones. To adapt DLLMs to VSR, we introduce a two-stage masked-denoising training strategy that separates visual-to-text content alignment from length modeling. We further observe a performance gap with oracle-length decoding, which assumes access to the true transcript length, indicating that reducing target-length uncertainty can improve DLLM-based VSR. To reduce this gap, we develop length-guided candidate decoding, which uses video duration to construct plausible transcript-length hypotheses, decodes under multiple hypotheses, and reranks candidates using length plausibility and decoding confidence. The proposed method achieves a state-of-the-art WER of 19.5\% on LRS3 using only its labeled training data.
>
---
#### [new 130] Self-Supervised Online Robot-Agnostic Traversability Estimation for Open-World Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决开放环境中路径可行驶性估计问题。提出COTRATE框架，实现在线学习与知识迁移，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2605.28442](https://arxiv.org/pdf/2605.28442)**

> **作者:** Julia Hindel; Simon Bultmann; Houman Masnavi; Daniele Cattaneo; Abhinav Valada
>
> **备注:** 14 pages, 16 Figures
>
> **摘要:** Self-supervised online traversability estimation enables robots to continuously learn from unlabeled open-world experiences and adapt their navigation behavior toward safe and efficient trajectories. Existing approaches either rely on handcrafted proprioceptive traversability scores, limiting robot-agnosticism, or cluster prior data, preventing online learning. Moreover, many continual learning methods incur substantial memory and computational costs, hindering onboard deployment. We introduce COTRATE, an online learning framework for continuous traversability estimation from multimodal, unlabeled robot experience. Our method first infers robust traversability scores using a robot-agnostic, learning-based online terrain assessment module operating on proprioceptiveand inertial signals. These scores then supervise a visual traversability network through a novel alignment loss that associates visual embeddings with online terrain this http URL mitigate forgetting during continual learning with minimal overhead, we propose a diversity-aware feature selection strategythat preserves performance using a compact replay memory. We further show that the learned traversability representation supports knowledge transfer across different robot platforms with different locomotion kinematics. We evaluate COTRATE on a dataset of \approx 50,000 images collected with two robotic platforms across 11 outdoor terrains, and benchmark it on navigation tasks in three representative outdoor environments. We make the dataset, code, and trained models publicly available.
>
---
#### [new 131] The Abstraction Gap in Vision-Language Causal Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言因果推理任务，旨在解决评估中难以区分语言流畅性与真实因果推理的问题。通过提出双探针方法和CAGE基准，量化并分析了模型的抽象差距。**

- **链接: [https://arxiv.org/pdf/2605.28779](https://arxiv.org/pdf/2605.28779)**

> **作者:** Chinh Hoang; Mohammad Rashedul Hasan
>
> **摘要:** Vision-language models (VLMs) generate fluent causal explanations, but current evaluations cannot distinguish linguistic plausibility from faithful causal reasoning. We introduce a dual-probe methodology that isolates these properties. The Text-Only Probe measures linguistic quality. The Chain-Text Probe requires models to first generate explicit causal chains. The Abstraction Gap (AG) metric quantifies the normalized performance difference. Evaluating eight VLMs on CAGE (Causal Abstraction Gap Evaluation), a benchmark of 49,500 questions across 5,500 images spanning Pearl's causal hierarchy, we find seven models exhibit AG exceeding 0.50 with text scores of 6--8 but chain scores below 2.5. Fine-tuning on 45,000 chain-annotated examples fails to close the gap. However, one model achieves near-zero AG. The capability exists within current VLM architectures and depends on pretraining and architectural choices. CAGE provides a diagnostic tool for assessing faithful causal reasoning in VLMs.
>
---
#### [new 132] Comparative Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition: Robustness, Efficiency, and Clinical Utility
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于序列模式识别任务，比较LNN与LSTM在时间动态建模中的表现，解决传统RNN在连续时间处理上的不足。**

- **链接: [https://arxiv.org/pdf/2605.27467](https://arxiv.org/pdf/2605.27467)**

> **作者:** Ye Kyaw Thu; Thazin Myint Oo; Thepchai Supnithi
>
> **备注:** 9 pages, 7 figures, 6 tables, The conference paper will appear in Proceedings of JCSSE 2026
>
> **摘要:** Traditional Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) units operate on discrete time steps, often failing to capture the fluid temporal dynamics of real-world physical processes. Liquid Neural Networks (LNNs), specifically Closed-form Continuous-time (CfC) networks, address this by modeling the hidden state evolution as a continuous differential equation. In this paper, we conduct a comprehensive benchmarking study across four distinct sequential modalities: neuromorphic event-based data (N-MNIST), stroke-based drawing (QuickDraw), visual handwriting (IAM), and physiological time-series (PhysioNet Sepsis-3). Furthermore, we perform a rigorous stress test using temporal dropout to evaluate model robustness against missing data. Our findings reveal that LNNs consistently provide superior parameter efficiency and significantly higher robustness in natively temporal domains and clinical environments where data sparsity is prevalent. This extended preprint provides additional background on related datasets and the LNN theoretical lineage, supplemented with a detailed appendix documenting our full implementation and experimental settings.
>
---
#### [new 133] RE-TRIANGLE: Does TRIANGLE Enable Multimodal Alignment Beyond Cosine Similarity in Retrieval?
- **分类: cs.IR; cs.AI; cs.CV**

- **简介: 该论文属于多模态检索任务，旨在解决传统方法在模态间对齐上的几何盲点问题。通过改进的TRIANGLE框架，优化模态三元组对齐，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.27436](https://arxiv.org/pdf/2605.27436)**

> **作者:** Arijit Ghosh; Aritra Bandyopadhyay; Chiranjeev Bindra; Jingfen Qiao
>
> **摘要:** Multimodal alignment is critical for bridging the semantic gap in information retrieval. However, traditional pairwise strategies introduce a geometric blind spot: while they align anchor modalities (e.g., text) with others, they lack constraints to enforce mutual consistency between peripheral modalities (e.g., video and audio). The TRIANGLE framework addresses this by minimizing the area of modality triplets on a hypersphere to enforce holistic alignment. In this reproducibility study, we verify the robustness of this geometric objective for retrieval tasks. We confirm that TRIANGLE outperforms pairwise baselines in zero-shot settings, achieving Recall@1 gains of up to +8.7 points, though benefits are domain-dependent. However, we fail to reproduce the reported learning-from-scratch results. Analysis using a synthetic toy dataset attributes this to instability when jointly optimizing geometric alignment with Data-Text Matching (DTM) loss. Furthermore, we find that cosine regularization primarily stabilizes text-to-video retrieval, and fine-tuning with domain supervision amplifies geometric benefits but reduces cross-dataset generalization. Our findings support the efficacy of geometric alignment while highlighting critical optimization sensitivities. Code available at this https URL.
>
---
## 更新

#### [replaced 001] PrecisionCUA: Iterative Visual Refinement for Pixel-Precise Cursor Grounding in Code Editors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.13019](https://arxiv.org/pdf/2604.13019)**

> **作者:** Himangi Mittal; Gaurav Mittal; Nelson Daniel Troncoso; Yu Hu
>
> **摘要:** Computer Use Agents (CUAs) fundamentally rely on graphical user interface (GUI) grounding to translate language instructions into executable screen actions, but editing-level grounding in dense coding interfaces (such as VS Code and Cursor), where sub-pixel accuracy is required to interact with dense IDE elements, remains underexplored. Existing approaches typically rely on single-shot coordinate prediction, which lacks a mechanism for error correction and often fails in high-density interfaces. In this technical report, we conduct an empirical study of pixel-precise cursor localization in coding environments. Instead of a single-step execution, our agent engages in an iterative refinement process, utilizing visual feedback from previous attempts to reach the target element. This closed-loop grounding mechanism allows the agent to self-correct displacement errors and adapt to dynamic UI changes. We evaluate our approach across Claude, Qwen, and GPT on a suite of complex coding benchmarks, demonstrating that multi-turn refinement significantly outperforms state-of-the-art single-shot models in both click precision and overall task success rate. Our results suggest that iterative visual reasoning is a critical component for the next generation of reliable software engineering agents. Code: this https URL.
>
---
#### [replaced 002] The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.17737](https://arxiv.org/pdf/2601.17737)**

> **作者:** Chenyu Mu; Xin He; Qu Yang; Wanshun Chen; Jiadi Yao; Huang Liu; Zihao Yi; Bo Zhao; Xingyu Chen; Ruotian Ma; Fanghua Ye; Erkun Yang; Cheng Deng; Zhaopeng Tu; Xiaolong Li; Linus
>
> **摘要:** Recent advances in video generation have produced models capable of synthesizing stunning visual content from simple text prompts. However, these models struggle to generate long-form, coherent narratives from high-level concepts like dialogue, revealing a ``semantic gap'' between a creative idea and its cinematic execution. To bridge this gap, we introduce a novel, end-to-end agentic framework for dialogue-to-cinematic-video generation. Central to our framework is ScripterAgent, a model trained to translate coarse dialogue into a fine-grained, executable cinematic script. To enable this, we construct ScriptBench, a new large-scale benchmark with rich multimodal context, annotated via an expert-guided pipeline. The generated script then guides DirectorAgent, which orchestrates state-of-the-art video models using a cross-scene continuous generation strategy to ensure long-horizon coherence. Our comprehensive evaluation, featuring an AI-powered CriticAgent and a new Visual-Script Alignment (VSA) metric, shows our framework significantly improves script faithfulness and temporal fidelity across all tested video models. Furthermore, our analysis uncovers a crucial trade-off in current SOTA models between visual spectacle and strict script adherence, providing valuable insights for the future of automated filmmaking.
>
---
#### [replaced 003] Cross-Modal Action Recognition in Egocentric Video Using Mamba: Integrating RGB and Hand Skeleton Streams via CLS Token Fusion Strategies
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.24302](https://arxiv.org/pdf/2605.24302)**

> **作者:** Juan Ignacio Bustos Gorostegui; Maria Elena Buemi
>
> **备注:** 4 pages , 2 figures , Egovis2026 , CVPR2026
>
> **摘要:** Egocentric action recognition is a challenging task due to erratic camera motion, frequent hand occlusion, and the difficulty of maintaining consistent visual representations over time. In this work, we propose a cross-modal architecture that combines RGB video and temporal hand skeleton data within a unified Mamba-based framework, exploiting the linear time complexity of State Space Models (SSMs). Our architecture consists of three components: a VideoMamba module for visual feature extraction, a skeleton encoder built on a stack of Mamba blocks, and a fusion module that integrates both modalities into a single representation. A central contribution of this work is the design and evaluation of four Class (CLS) token mixing strategies for multimodal fusion: Naive, Average, Weighted and Context-based. These strategies differ in how the pretrained unimodal CLS tokens, which role is to act as information sinks concentrating learned representations, are leveraged to initialize the mixed CLS token used for final classification. We evaluate all strategies on the H2O dataset. Experimental results show that the Average strategy achieves the best performance, yielding gains of over 10% Top-1 accuracy in the Tiny configuration and 2% in the Small configuration over the VideoMamba baseline.
>
---
#### [replaced 004] A Survey on Event-based Optical Marker Systems
- **分类: cs.RO; cs.CV**

- **简介: 本文综述事件驱动的光学标记系统，属于机器感知任务，解决传统视觉系统在动态环境中的局限性，通过分析其异步特性与鲁棒性，探讨其在目标跟踪、位姿估计等领域的应用。**

- **链接: [https://arxiv.org/pdf/2504.20736](https://arxiv.org/pdf/2504.20736)**

> **作者:** Nafiseh Jabbari Tofighi; Maxime Robic; Fabio Morbidi; Pascal Vasseur
>
> **备注:** 11 pages, 6 figures, 2 table
>
> **摘要:** The advent of event-based cameras, with their low latency, high dynamic range, and reduced power consumption, marked a turning point in machine perception and robotic vision. In~particular, the combination of these neuromorphic sensors with widely-available passive or active optical markers (e.g. AprilTags, arrays of blinking LEDs), has recently opened up a new field of opportunities. This survey paper provides a comprehensive review of Event-Based Optical Marker Systems (EBOMS). We~analyze the underlying principles and technologies on which these systems are based, with a special focus on their asynchronous operation and robustness against challenging lighting conditions. We also describe the most relevant applications of EBOMS, including object detection and tracking, pose estimation, and optical communication. The article concludes with a discussion of possible future research directions in this rapidly-emerging and multidisciplinary area.
>
---
#### [replaced 005] WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22096](https://arxiv.org/pdf/2602.22096)**

> **作者:** Wenhua Wu; Huai Guan; Zhe Liu; Hesheng Wang
>
> **摘要:** Editable high-fidelity 4D scenes are crucial for autonomous driving, as they can be applied to end-to-end training and closed-loop simulation. However, existing reconstruction methods are primarily limited to replicating observed scenes and lack the capability for diverse weather simulation. While image-level weather editing methods tend to introduce scene artifacts and offer poor controllability over the weather effects. To address these limitations, we propose \textbf{WeatherCity}, a novel framework for 4D urban scene reconstruction and weather editing. Specifically, we leverage a text-guided image editing model to achieve flexible editing of image weather backgrounds. To tackle the challenge of multi-weather modeling, we introduce a novel weather Gaussian representation based on shared scene features and dedicated weather-specific decoders. This representation is further enhanced with a content consistency optimization, ensuring coherent modeling across different weather conditions. Additionally, we design a physics-driven model that simulates dynamic weather effects through particles and motion patterns. Extensive experiments on multiple datasets and various scenes demonstrate that WeatherCity achieves flexible controllability, high fidelity, and temporal consistency in 4D reconstruction and weather editing. Our framework not only enables fine-grained control over weather conditions (e.g., light rain and heavy snow) but also supports object-level manipulation within the scene. Codes are released at this https URL.
>
---
#### [replaced 006] Polygon-mamba: Retinal vessel segmentation using polygon scanning mamba and space-frequency collaborative attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.10581](https://arxiv.org/pdf/2605.10581)**

> **作者:** Yuanyuan Peng; Wen Li
>
> **摘要:** Retinal vessel segmentation is crucial for diagnosis and assessment of ocular diseases. Notably, segmentation of small retinal vessels has been consistently recognized as a challenging and complex task. To tackle this challenge, we design a hybrid CNN-Mamba fusion network that integrates polygon scanning mamba and space-frequency collaborative attention mechanism for the detection of small vessels. Considering that the traditional mamba architecture with horizontal-vertical scanning may compromise the topological integrity of target structures and result in local discontinuities in small retinal vessels, we present a polygon scanning visual state space model (PS-VSS) to identify small vessel structural features by multi-layer reverse scanning way. Which effectively preserves pixels connectivity, thereby substantially mitigating the loss of information pertaining to small vessels. Furthermore, as we all known that the spatial domain prioritizes positional and structural information, while the frequency domain emphasizes global perception and local detail components, a space-frequency collaborative attention mechanism (SFCAM) is introduced within the skip connection to extract efficient features from the spatial and frequency domains. This strategy empowers the model to dynamically enhance the key features while effectively suppressing clutters. To assess the efficacy of our model, it was tested on three publicly available datasets: DRIVE, STARE, and CHASE_DB1. Compared to manual annotations, our model demonstrated F1 scores of 0.8283, 0.8282, and 0.8251, Area Under Curve (AUC) values of 0.9806, 0.9840, and 0.9866, and Sensitivity (SE) values of of 0.8268, 0.8314, and 0.8484 across three datasets, respectively. The effectiveness of our model was validated through both visual inspection and quantitative analysis.
>
---
#### [replaced 007] AI-T2I: Aggregating-and-Isolating Cross-Attention to Diffusion Models for Text-to-Image Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.25763](https://arxiv.org/pdf/2605.25763)**

> **作者:** Shipeng Cao; Biao Qian; Haipeng Liu; Yang Wang; Meng Wang
>
> **备注:** Accepted by IEEE Transactions on Multimedia (2026). 13 pages, 15 figures
>
> **摘要:** Text-to-image synthesis has made significant progress, benefiting from the strong generative capabilities of diffusion models. However, these models struggle to achieve precise text-to-image alignment within cross-attention maps during the denoising process. Existing works primarily focus on inter-subject-token activations (i.e., cross-attention scores) overlap for different subjects, overlooking the intra-subject-token activations scattering issue for identical subjects. In this paper, we propose an Aggregating-and-Isolating cross-attention approach to diffusion models for Text-to-Image synthesis, dubbed AI-T2I. Technically, to address the scattering issue, we devise an aggregation loss to identify and consolidate the scattered intra-token activations, which implicitly helps mitigate the potential overlap issue. Upon that, an isolation loss is further introduced to push the inter-token activations apart, thus fulfilling precise text-to-image alignment. Extensive experiments on various benchmarks demonstrate the superiority of AI-T2I over the state-of-the-art works for text-to-image synthesis. Furthermore, our AI-T2I exhibits excellent generalization across other tasks, e.g., controllable layout generation and personalized generation. Our code is available at this https URL.
>
---
#### [replaced 008] VesselSim: learning 3D blood vessel segmentation without expert annotations
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.26277](https://arxiv.org/pdf/2605.26277)**

> **作者:** Erin Rainville; Melissa Ananian; Tristan Mirolla; Hassan Rivaz; Yiming Xiao
>
> **备注:** This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution will be published as part of the MICCAI 2026 proceedings in October
>
> **摘要:** Blood vessel segmentation is a core task in medical image analysis for the care of vascular diseases and surgical planning, yet the challenges of providing expert vascular annotations pose a major obstacle for the progress of related deep learning techniques. To address this, we propose VesselSim, a two-stage framework for universal 3D blood vessel segmentation that eliminates the need for real annotated data during training. First, we introduce a stochastic, geometry-driven vascular simulation framework that models recursive branching, curvature-controlled growth, and collision-aware topology, followed by domain-randomized intensity synthesis to generate 16,500 anatomically plausible 3D angiographic volumes. Second, a 3D U-Net is trained solely on this synthetic data. To bridge the domain gap from synthetic to real images at inference time, we introduce a test-time adaptation strategy via a self-supervised mask reconstruction decoder, enabling adaptation to unseen clinical scans without prior domain knowledge. We evaluate VesselSim in a zero-shot setting on multiple real-world datasets spanning MR and CT across several anatomical regions, including the brain and kidneys. Despite being trained exclusively on synthetic data, VesselSim achieves performance competitive with state-of-the-art vascular segmentation foundation models. These findings suggest that learning vessel geometry from synthetic tubular structures is effective for robust cross-domain generalization, substantially reducing the reliance on acquired medical imaging data and more importantly, expert annotations.
>
---
#### [replaced 009] The Forensic Cost of Watermark Removal: From Dedicated Attacks to Image Editing
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.25491](https://arxiv.org/pdf/2604.25491)**

> **作者:** Gautier Evennou; Ewa Kijak
>
> **备注:** v1:The Forensic Cost of Watermark Removal, accepted at IH&MMSEC 2026, Special Session "Watermarking Across the Lifecycle of Generative Models". v2: extended version, under review
>
> **摘要:** Current watermark removal methods are evaluated on two axes: attack success rate and perceptual quality. We show this is insufficient. While state-of-the-art attacks successfully degrade the watermark signal without visible distortion, they leave distinct statistical artifacts that betray the removal attempt. We name this overlooked axis Watermark Removal Detection (WRD) and demonstrate that a modern classifier trained on these artifacts achieves state-of-the-art detection rates at $10^{-3}$ FPR across every removal method tested. No existing attack accounts for this forensic leakage. We benchmark leading watermarking schemes against standard removal pipelines under the extended evaluation triple of attack success, perceptual quality, and forensic detectability, and find that no current method balances all three. Our results establish forensic stealthiness as a necessary requirement for watermark removal.
>
---
#### [replaced 010] PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.17354](https://arxiv.org/pdf/2601.17354)**

> **作者:** Wenzhi Guo; Guangchi Fang; Shu Yang; Bing Wang
>
> **摘要:** While 3D Gaussian Splatting (3DGS) enables real-time rendering, its training demands workstation-level compute and memory, making mobile deployment impractical under minute-scale time budgets and limited peak memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high-fidelity reconstruction. PocketGS resolves the fundamental tension between training efficiency, memory compactness, and modeling quality through three co-designed operators: $\mathcal{G}$ builds geometry-faithful point-cloud priors; $\mathcal{I}$ injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and $\mathcal{T}$ unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Extensive experiments demonstrate that PocketGS outperforms the powerful mainstream workstation 3DGS baseline under mobile budgets, delivering high-quality reconstructions and enabling a fully on-device, practical capture-to-rendering workflow.
>
---
#### [replaced 011] MMTABREAL: Real-World Benchmark for Multimodal Table Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.21771](https://arxiv.org/pdf/2505.21771)**

> **作者:** Prasham Titiya; Jainil Trivedi; Chitta Baral; Vivek Gupta
>
> **摘要:** Multimodal tables i.e. tabular layouts interleaved with charts, maps, icons, and color encodings are ubiquitous in real applications yet remain difficult for Multimodal Large Language Models (MLLMs). Despite advances in text and image understanding, systematic evaluation of table-centric multimodal reasoning is limited. We introduce MMTABREAL, a MultiModal Table Benchmark, human-curated suite of 500 real-world tables paired with 4,021 question-answer pairs. MMTABREAL spans four question types, five reasoning categories, and eight structural archetypes. Evaluations of state-of-the-art models reveal substantial gaps, especially in visual grounding, spatial alignment, and multi-step inference, with 20-40% performance drops relative to existing benchmarks. These results highlight the need for architectures that more tightly fuse vision with tabular structure and support explicit numeric/logical operations. MMTABREAL is released for evaluation only, providing a rigorous, reproducible testbed that reflects the linguistic, structural, and reasoning complexity of real-world multimodal tables.
>
---
#### [replaced 012] Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态大模型推理任务，旨在解决模态对齐和训练可扩展性问题。提出D2I框架，通过格式化奖励提升推理能力，无需额外标注或复杂奖励。**

- **链接: [https://arxiv.org/pdf/2507.06999](https://arxiv.org/pdf/2507.06999)**

> **作者:** Yahan Yu; Yuyang Dong; Masafumi Oyamada
>
> **备注:** 22 pages, 24 figures
>
> **摘要:** Reasoning is essential for large language models (LLMs), especially in complex tasks such as mathematical problem solving. However, multimodal reasoning still faces challenges in modality alignment and training scalability, as many existing methods rely on additional annotations or complex rule-based rewards. To address these issues, we propose the Deliberate-to-Intuitive reasoning framework (D2I), which improves the understanding and reasoning abilities of multimodal LLMs (MLLMs) without extra annotations or complex rewards. During training, D2I uses deliberate reasoning strategies supervised only by rule-based format rewards to enhance modality alignment. During inference, it shifts to intuitive reasoning by removing these explicit strategies, allowing the model to implicitly apply the acquired abilities in its responses. D2I outperforms baselines on both in-domain and out-of-domain benchmarks, highlighting the effectiveness of format rewards in fostering transferable multimodal reasoning skills and suggesting the benefit of decoupling training-time reasoning depth from test-time response flexibility.
>
---
#### [replaced 013] Segment to Focus: Guiding Latent Action Models in the Presence of Distractors
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02259](https://arxiv.org/pdf/2602.02259)**

> **作者:** Marcus Fechner; Hamza Adnan; Constantin C. Lüth; Matthew T. Jackson; Alexey Zakharov; J. Marius Zöllner
>
> **摘要:** Latent action models (LAMs) offer a promising path to pre-training embodied agents on large amounts of action-free video. They infer latent actions between consecutive observations that can later be decoded to ground-truth actions using a small number of labels. However, recent work has shown that this recipe fails in the presence of action-correlated visual distractors common in real-world video, such as dynamic backgrounds, camera shake, or other moving objects. In these scenarios, the standard reconstruction objective drives latent actions to encode exogenous motion instead of agent-controlled dynamics, resulting in policies that underperform when fine-tuned. We observe, however, that endogenous and exogenous factors are typically spatially separated in pixel space: control-relevant change is concentrated on the agent, while distractor motion occurs elsewhere. We exploit this observation by restricting the reconstruction objective to agent pixels, forcing latent actions to explain agent-controlled dynamics rather than exogenous ones. We call this method MaskLAM; it obtains the agent mask zero-shot from off-the-shelf segmentation foundation models (e.g., SAM) and requires no architectural changes, auxiliary losses, or action labels during pre-training. Across two continuous-control benchmarks (Distracting Control Suite, Distracting Meta-World), MaskLAM reduces normalized linear-probe MSE by up to $3.51\times$ and improves normalized return by up to $4.97\times$ over LAPO, while narrowing the gap to LAOM-Labels, which relies on ground-truth action supervision.
>
---
#### [replaced 014] CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher On-Policy Distillation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.25378](https://arxiv.org/pdf/2605.25378)**

> **作者:** Fangtai Wu; Hailong Guo; Shijie Huang; Jiayi Song; Yubo Huang; Mushui Liu; Zhao Wang; Yunlong Yu; Jiaming Liu; Ruihua Huang
>
> **摘要:** Customized image editing aims to equip pre-trained diffusion models with specific visual effects using limited paired data, typically via Low-Rank Adaptation (LoRA). As the number of desired effects grows, storing and dynamically loading numerous these effect LoRAs significantly increases deployment overhead. Furthermore, current pipelines typically cascade these effect LoRAs with acceleration modules for fast generation, which triggers severe parameter interference and results in concept bleeding and style degradation. We propose CollectionLoRA, a multi-teacher on-policy distillation framework capable of distilling the concepts of up to 50 different effect LoRAs along with few-step generation capabilities into a single LoRA. This fundamentally resolves the feature interference issue and significantly reduces deployment costs. Specifically, the method introduces (i) a Probabilistic Dual-Stream Routing mechanism that enables the model to randomly switch between data sources during training, effectively enhancing its generalization in unseen scenarios; (ii) an Asymmetric Orthogonal Prompting strategy to achieve concept isolation within the prompt space; (iii) a Coarse-to-Fine Distillation Objective to mitigate the distribution gap between the teacher and student models. Extensive evaluations show that CollectionLoRA distills all customized effects and few-step generation into a single LoRA, reducing deployment overhead while achieving concept fidelity comparable to or better than independently trained teacher models. Code: this https URL
>
---
#### [replaced 015] When Eyes Betray AI: Social Gaze Consistency as a Semantic Cue for AI-Generated Image Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.27348](https://arxiv.org/pdf/2605.27348)**

> **作者:** Jihyeon Kim; Sohee Kim; Soosan Lee; Souhwan Jung; James Matthew Rehg; Hyesong Choi
>
> **备注:** 23 pages, 2 figures, 17 tables
>
> **摘要:** Recent generative models have largely closed the gap on low-level artifacts - pixel fingerprints, frequency anomalies, upsampling traces - particularly in person-centric and partial-edit settings where the manipulated region is small and surrounded by photometrically authentic content. We introduce Social Gaze Consistency, a high-level semantic cue defined as the mutual coherence of gaze direction, head-eye alignment, and pupil placement between interacting individuals, and show that it constitutes a previously underutilized detection axis orthogonal to existing low-level paradigms. We instantiate this insight through three coupled mechanisms: (i) a controlled diagnostic dataset with region-specific perturbations of gaze-consistent imagery, where strict pair-level grouping forecloses generator-fingerprint memorization as an optimization-time shortcut rather than relying on augmentation; (ii) Block-Compositional Caption Supervision, which holds a single 5-block reasoning skeleton invariant across 1,250 macro-combined captions, decoupling reasoning consistency from surface diversity; (iii) Cross-architecture validation showing the same supervision improves a vision-language backbone (FakeVLM) by +3.7 pp on the COCOAI Interaction subset (balanced accuracy 67.8 -> 71.5) and +1.3 pp on the COCOAI Person subset (83.0 -> 84.3), with consistent gains on a vision-only backbone (Effort), evidencing a backbone-agnostic cue. Real- and fake-class recalls rise simultaneously, ruling out a "predict-all-fake" artifact. A four-step mechanistic account - paired-edit shortcut blocking, hard-to-easy difficulty transfer, CLIP prior preservation, and diffusion-family shared spectral weakness in periocular structure - explains why training on a single inpainter (FLUX.1-Fill) transfers to multi-generator suites. We will release the code upon acceptance to facilitate reproducibility.
>
---
#### [replaced 016] DODO: Discrete OCR Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.16872](https://arxiv.org/pdf/2602.16872)**

> **作者:** Sean Man; Gilad Deutch; Roy Ganz; Roi Ronen; Shahar Tsiper; Shai Mazor; Niv Nayman
>
> **摘要:** Optical Character Recognition (OCR) is a fundamental task for digitizing information, serving as a critical bridge between visual data and textual understanding. While modern Vision-Language Models (VLM) have achieved high accuracy in this domain, they predominantly rely on autoregressive decoding, which becomes computationally expensive and slow for long documents as it requires a sequential forward pass for every generated token. We identify a key opportunity to overcome this bottleneck: unlike open-ended generation, OCR is a highly deterministic task where the visual input strictly dictates a unique output sequence, theoretically enabling efficient, parallel decoding via diffusion models. However, we show that existing masked diffusion models fail to harness this potential; those introduce structural instabilities that are benign in flexible tasks, like captioning, but catastrophic for the rigid, exact-match requirements of OCR. To bridge this gap, we introduce DODO, the first VLM to utilize block discrete diffusion and unlock its speedup potential for OCR. By decomposing generation into blocks, DODO mitigates the synchronization errors of global diffusion. Empirically, our method achieves near state-of-the-art accuracy while enabling up to 5x faster inference compared to autoregressive baselines.
>
---
#### [replaced 017] Paris 2.0: A Decentralized Diffusion Model for Video Generation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.26064](https://arxiv.org/pdf/2605.26064)**

> **作者:** Ali Rouzbayani; Bidhan Roy; Marcos Villagra; Zhiying Jiang
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** We present Paris 2.0, the first video generation model pre-trained through decentralized computation. Its training recipe builds upon Paris 1.0 (arXiv:2510.03434), the first ever open-weight Decentralized Diffusion Model (DDM), which showed that image generation can be trained without a monolithic GPU cluster. However, temporally coherent video generation had remained an open problem under decentralized training, and Paris 2.0 closes it. In low-resolution text-to-video training, against a monolithic model trained on the same data under a matched total compute budget, Paris 2.0 cuts Frechet Video Distance (FVD) from 561.04 to 279.01, a ~2.0x improvement, and lifts CLIP text-video similarity and aesthetic score.
>
---
#### [replaced 018] VideoCanvas: Unified Video Completion from Arbitrary Spatiotemporal Patches via In-Context Conditioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.08555](https://arxiv.org/pdf/2510.08555)**

> **作者:** Minghong Cai; Qiulin Wang; Zongli Ye; Wenze Liu; Quande Liu; Weicai Ye; Xintao Wang; Pengfei Wan; Kun Gai; Xiangyu Yue
>
> **备注:** Project page: this https URL
>
> **摘要:** Existing controllable video generation methods are typically designed for rigid, task-specific settings, such as first-frame image-to-video, inpainting, or interpolation, treating spatio-temporal control as a set of isolated problems. We formalize a unified task, arbitrary spatio-temporal video completion, where a model generates a coherent video from user-specified patches placed at any spatial location and timestamp. However, realizing such a unified framework within modern latent video diffusion models is non-trivial: causal video VAEs compress multiple frames into a single latent slot, making frame-level conditioning fundamentally ill-posed, and directly feeding sparsely populated, zero-padded video inputs into the VAE leads to severe out-of-distribution artifacts. To address these challenges, we propose VideoCanvas, a simple yet effective framework that adapts the In-Context Conditioning paradigm to arbitrary spatio-temporal completion without modifying or retraining the VAE. Our key idea is a hybrid conditioning strategy that decouples spatial and temporal control: spatially, we encode zero-padded full-frame canvases in image mode to keep VAE inputs in-distribution, and temporally we use Temporal RoPE Interpolation to assign each condition a continuous fractional index in the latent sequence for precise frame-level alignment. To evaluate this capability, we develop VideoCanvasBench, the first benchmark for arbitrary spatio-temporal video completion, covering both intra-scene fidelity and inter-scene creativity. Extensive experiments demonstrate that VideoCanvas achieves state-of-the-art performance across a diverse range of video generation tasks under a single, unified framework.
>
---
#### [replaced 019] Privacy Protection Against Personalized Text-to-Image Synthesis via Cross-image Consistency Constraints
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.12747](https://arxiv.org/pdf/2504.12747)**

> **作者:** Guanyu Wang; Kailong Wang; Yihao Huang; Mingyi Zhou; Geguang Pu; Li Li
>
> **摘要:** The rapid advancement of diffusion models and personalization techniques has made it possible to recreate individual portraits from just a few publicly available images. While such capabilities empower various creative applications, they also introduce serious privacy concerns, as adversaries can exploit them to generate highly realistic impersonations. To counter these threats, anti-personalization methods have been proposed, which add adversarial perturbations to published images to disrupt the training of personalization models. However, existing approaches largely overlook the intrinsic multi-image nature of personalization and instead adopt a naive strategy of applying perturbations independently, as commonly done in single-image settings. This neglects the opportunity to leverage inter-image relationships for stronger privacy protection. Therefore, we advocate for a group-level perspective on privacy protection against personalization. Specifically, we introduce Cross-image Anti-Personalization (CAP), a novel framework that enhances resistance to personalization by enforcing style consistency across perturbed images. Furthermore, we develop a dynamic ratio adjustment strategy that adaptively balances the impact of the consistency loss throughout the attack iterations. Extensive experiments on the classical CelebHQ and VGGFace2 benchmarks show that CAP substantially improves existing methods.
>
---
#### [replaced 020] IRPO: Boosting Image Restoration via Post-training GRPO
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00814](https://arxiv.org/pdf/2512.00814)**

> **作者:** Haoxuan Xu; Yi Liu; Tianfu Li; Ruolin Shen; Boyuan Jiang; Jinlong Peng; Donghao Luo; Xiaobin Hu; Shuicheng Yan; Haoang Li
>
> **摘要:** Post-training has become effective for high-level generation, but its role in low-level vision remains underexplored. Existing image restoration methods often rely on fixed pixel-wise fitting to ground-truth images, which can lead to over-smoothing and weak generalization. We propose IRPO, a GRPO-based post-training framework for deterministic restoration models. IRPO is built around two axes: data formulation and reward modeling. For data formulation, we select the 30% underperforming samples from the pre-training stage, which improves both accuracy and training efficiency. For reward modeling, we combine fidelity-oriented and quality-aware feedback with three components: a General Reward for structural fidelity, an Expert Reward that uses a Vision-Language Model as a coarse visual-quality judge, and a Restoration Reward for task-specific low-level cues. Experiments on six in-domain and five out-of-domain (OOD) benchmarks show that IRPO improves the AdaIR baseline by 0.93 dB on in-domain tasks and 3.43 dB on OOD settings. Our code can be shown in this https URL.
>
---
#### [replaced 021] NCSAM Noise-Compensated Sharpness-Aware Minimization for Noisy Label Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19947](https://arxiv.org/pdf/2601.19947)**

> **作者:** Jiayu Xu; Junbiao Pang
>
> **备注:** 11 pages, 1 figure, 8 tables. Major revision of v1: revised PAC-Bayesian theoretical analysis, clarified the NCSAM formulation, added appendix derivations, reorganized experiments and ablations, updated related work, citations, writing, and author list
>
> **摘要:** Learning from Noisy Labels (LNL) remains a fundamental challenge in deep learning because real-world datasets often contain corrupted annotations. Most existing methods rely on label correction or sample selection mechanisms. In contrast, we study LNL from an optimization perspective by establishing a theoretical connection between label noise and the flatness-seeking behavior of Sharpness-Aware Minimization (SAM). Based on this analysis, we propose Noise-Compensated Sharpness-Aware Minimization (NCSAM), which uses a noise-compensated perturbation to counteract the optimization bias induced by noisy labels. By correcting distorted SAM perturbations, NCSAM mitigates the memorization of noisy labels during training while preserving the simplicity of optimization-based learning. Experiments on synthetic and real-world noisy-label benchmarks show that NCSAM consistently improves over SAM-based optimization baselines and remains competitive with representative noisy-label learning methods.
>
---
#### [replaced 022] Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12887](https://arxiv.org/pdf/2512.12887)**

> **作者:** Han Liu; Bogdan Georgescu; Yanbo Zhang; Youngjin Yoo; Michael Baumgartner; Riqiang Gao; Jianing Wang; Gengyan Zhao; Eli Gibson; Dorin Comaniciu; Sasa Grbic
>
> **备注:** 1st Place in VLM3D Challenge
>
> **摘要:** 3D medical image classification is essential for modern clinical workflows. Medical foundation models (FMs) have emerged as a promising approach for scaling to new tasks, yet current research suffers from three critical pitfalls: data-regime bias, suboptimal adaptation, and insufficient task coverage. In this paper, we address these pitfalls and introduce AnyMC3D, a scalable 3D classifier adapted from 2D FMs. Our method scales efficiently to new tasks by adding only lightweight plugins (about 1M parameters per task) on top of a single frozen backbone. This versatile framework also supports multi-view inputs, auxiliary pixel-level supervision, and interpretable heatmap generation. We establish a comprehensive benchmark of 12 tasks covering diverse pathologies, anatomies, and modalities, and systematically analyze state-of-the-art 3D classification techniques. Our analysis reveals key insights: (1) effective adaptation is essential to unlock FM potential, (2) general-purpose FMs can match medical-specific FMs if properly adapted, and (3) 2D-based methods surpass 3D architectures for 3D classification. For the first time, we demonstrate the feasibility of achieving state-of-the-art performance across diverse applications using a single scalable framework (including 1st place in the VLM3D challenge), eliminating the need for separate task-specific models.
>
---
#### [replaced 023] JLT: Clean-Latent Prediction in Latent Diffusion Transformers
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.27102](https://arxiv.org/pdf/2605.27102)**

> **作者:** Funing Fu; Tenghui Wang; Guanyu Zhou; Junyong Cen; Qichao Zhu
>
> **摘要:** Flow matching with clean-data prediction has shown that regressing the clean point can exploit low-dimensional structure more effectively than predicting an ambient noised quantity. We ask whether this principle remains useful after images are mapped into a learned latent space, where compression has already removed much of the raw pixel variability. We introduce JLT, a 130M latent diffusion Transformer over frozen FLUX.2 VAE codes, and compare clean-latent prediction with a matched velocity-prediction DiT under the same representation, backbone, and training settings. Although the three variables x, epsilon, and v are linearly convertible for a fixed corruption time, a local Gaussian analysis shows that velocity regression inherits an isotropic target-covariance floor and amplifies low-variance latent directions, while clean prediction damps them. On ImageNet 256 x 256, JLT-B/1 obtains FID-50K 2.50 with classifier-free guidance, with a large matched-target gap over velocity prediction. These results suggest that prediction targets in latent diffusion are representation-dependent geometric choices, rather than interchangeable algebraic parameterizations.
>
---
#### [replaced 024] Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [https://arxiv.org/pdf/2503.02857](https://arxiv.org/pdf/2503.02857)**

> **作者:** Nuria Alina Chandra; Hannah Lee; Ryan Murtfeldt; Lin Qiu; Arnab Karmakar; Emmanuel Tanumihardja; Kevin Farhat; Ben Caffee; Changyeon Lee; Jongwook Choi; Sejin Paik; Aerin Kim; Oren Etzioni
>
> **摘要:** In the age of increasingly realistic generative AI, robust deepfake detection is essential for mitigating fraud and disinformation. While many deepfake detectors report high accuracy on academic datasets, we show that these academic benchmarks are out of date and not representative of real-world deepfakes. We introduce Deepfake-Eval-2024, a new deepfake detection benchmark consisting of in-the-wild deepfakes collected from social media and deepfake detection platform users in 2024. Deepfake-Eval-2024 consists of 45 hours of videos, 56.5 hours of audio, and 1,975 images, encompassing the latest manipulation technologies. The benchmark contains diverse media content from 88 different websites in 52 different languages. We find that the performance of open-source state-of-the-art deepfake detection models drops precipitously when evaluated on Deepfake-Eval-2024, with AUC decreasing by 50% for video, 48% for audio, and 45% for image models compared to previous benchmarks. We also evaluate commercial deepfake detection models and models finetuned on Deepfake-Eval-2024, and find that they have superior performance to off-the-shelf open-source models, but do not yet reach the accuracy of deepfake forensic analysts. The dataset is available at this https URL.
>
---
#### [replaced 025] KG-ViP: Bridging Knowledge Grounding and Visual Perception in Multi-modal LLMs for Visual Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11632](https://arxiv.org/pdf/2601.11632)**

> **作者:** Zhiyang Li; Ao Ke; Yukun Cao; Xike Xie
>
> **摘要:** Multi-modal Large Language Models (MLLMs) for Visual Question Answering (VQA) often suffer from dual limitations: knowledge hallucination and insufficient fine-grained visual perception. Crucially, we identify that commonsense graphs and scene graphs provide precisely complementary solutions to these respective deficiencies by providing rich external knowledge and capturing fine-grained visual details. However, prior works typically treat them in isolation, overlooking their synergistic potential. To bridge this gap, we propose KG-ViP, a unified framework that empowers MLLMs by fusing scene graphs and commonsense graphs. The core of the KG-ViP framework is a novel retrieval-and-fusion pipeline that utilizes the query as a semantic bridge to progressively integrate both graphs, synthesizing a unified structured context that facilitates reliable multi-modal reasoning. Extensive experiments on FVQA 2.0+ and MVQA benchmarks demonstrate that KG-ViP significantly outperforms existing VQA methods.
>
---
#### [replaced 026] ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型任务，旨在解决视觉处理计算开销大的问题。提出ViCA架构，通过稀疏跨注意力减少视觉计算，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.07574](https://arxiv.org/pdf/2602.07574)**

> **作者:** Wenjie Liu; Hao Wu; Xin Qiu; Xudong Wang; Yingqi Fan; Yihan Zhang; Anhao Zhao; Yunpu Ma; Xiaoyu Shen
>
> **摘要:** Modern multimodal large language models (MLLMs) adopt a unified self-attention design that processes visual and textual tokens at every Transformer layer, incurring substantial computational overhead. In this work, we revisit the necessity of such dense visual processing and show that projected visual embeddings are already well-aligned with the language space, while effective vision-language interaction occurs in only a small subset of layers. Based on these insights, we propose ViCA (Vision-only Cross-Attention), a minimal MLLM architecture in which visual tokens bypass all self-attention and feed-forward layers, interacting with text solely through sparse cross-attention at selected layers. Extensive evaluations across three MLLM backbones, nine multimodal benchmarks, and 26 pruning-based baselines show that ViCA preserves 98% of baseline accuracy while reducing visual-side computation to 4%, consistently achieving superior performance-efficiency trade-offs. Moreover, ViCA provides a regular, hardware-friendly inference pipeline that yields over 3.5x speedup in single-batch inference and over 10x speedup in multi-batch inference, reducing visual grounding to near-zero overhead compared with text-only LLMs. It is also orthogonal to token pruning methods and can be seamlessly combined for further efficiency gains. Our code is available at this https URL.
>
---
#### [replaced 027] Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于表格推理任务，解决LVLM在复杂表格结构下的理解与推理问题。提出DiSCo和Table-GLS框架，实现结构与内容解耦及结构引导推理，无需外部工具和大量标注。**

- **链接: [https://arxiv.org/pdf/2602.03491](https://arxiv.org/pdf/2602.03491)**

> **作者:** Yingjie Zhu; Xuefeng Bai; Kehai Chen; Yang Xiang; Youcheng Pan; Xiaoqiang Zhou; Min Zhang
>
> **备注:** Accepted as a Spotlight Paper at ICML 2026
>
> **摘要:** Reasoning over table images remains challenging for Large Vision-Language Models (LVLMs) due to complex layouts and tightly coupled structure-content information. Existing solutions often depend on expensive supervised training, reinforcement learning, or external tools, limiting efficiency and scalability. This work addresses a key question: how to adapt LVLMs to table reasoning with minimal annotation and no external tools? Specifically, we first introduce DiSCo, a Disentangled Structure-Content alignment framework that explicitly separates structural abstraction from semantic grounding during multimodal alignment, efficiently adapting LVLMs to tables structures. Building on DiSCo, we further present Table-GLS, a Global-to-Local Structure-guided reasoning framework that performs table reasoning via structured exploration and evidence-grounded inference. Extensive experiments across diverse benchmarks demonstrate that our framework efficiently enhances LVLM's table understanding and reasoning capabilities, particularly generalizing to unseen table structures. Our data and code are available at this https URL.
>
---
#### [replaced 028] LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.19729](https://arxiv.org/pdf/2605.19729)**

> **作者:** Hyunsoo Han; Sangyeop Yeo; Jaejun Yoo
>
> **备注:** Project page: this https URL , 15 pages, 11 figure, 9 tables, To appear in CVPR 2026
>
> **摘要:** We demonstrate that in knowledge distillation for diffusion models, the teacher network's highly complex denoising process - stemming from its substantially larger capacity - poses a significant challenge for the student model to faithfully mimic. To address this problem, we propose a coarse-to-fine distillation framework with LInear FiTtingbased distillation (LIFT) and Piecewise Local Adaptive Coefficient Estimation (PLACE). First, LIFT decomposes the objective into a "coarse" alignment and a "fine" refinement. The student is then trained on coarse alignment before proceeding to hard refinement. Second, PLACE extends LIFT to address spatially non-uniform errors by partitioning outputs into error-based groups, providing locally adaptive guidance. Our experiments show that LIFT and PLACE is effective across diffusion spaces (image/latent), backbones (U-Net/DiT), tasks (unconditional/conditional), datasets, and even extends to flow-based models such as MMDiT (SD3). Furthermore, under extreme compression with a 1.3M-parameter student (only 1.6% of the teacher), conventional KD fails to provide sufficient guidance for stable training, with FID scores often degrading to 50-200+, but our method remains stably convergent and achieves an FID of 15.73.
>
---
#### [replaced 029] Garment Particles: A 2D--3D Symmetric Garment Representation for Generation and Editing
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2605.26391](https://arxiv.org/pdf/2605.26391)**

> **作者:** Kiyohiro Nakayama; I-Chao Shen; Ruofan Liu; Yiming Wang; Gordon Wetzstein; Takeo Igarashi
>
> **摘要:** Practical garment design spans two modes: intuitive creation from high-level intent, such as a reference image or text description, and complex low-level editing across 2D sewing patterns and 3D draped geometry, which requires professional training to navigate their complex interdependencies. Yet existing frameworks address only part of this challenge, offering either garment generation from casual inputs or direct editing on sewing patterns. To support both ends of the spectrum, we propose Garment Particles, a 5D point-cloud representation that jointly encodes 2D sewing patterns and 3D geometry. This representation enables Garment Particles Flow (GPF), a rectified flow framework that supports intuitive generation from high-level inputs (text, images, sketches) and various editing operations on 2D sewing patterns and 3D geometries via diffusion posterior sampling. Finally, we introduce Particles-to-Pattern Flow that converts generated garment particles into curved-based patterns for simulation. We validate our model's generation ability on multiple datasets, achieving state-of-the-art garment generation results against competitive baselines. Our model also enables many garment editing scenarios, including garment interpolation, sewing pattern editing, point-cloud- and silhouette-conditioned garment generation. Our project website is at this https URL .
>
---
#### [replaced 030] MMRad-22K: A Structured Multimodal Evidence Dataset for Chest X-ray Report Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.12843](https://arxiv.org/pdf/2602.12843)**

> **作者:** Yichen Zhao; Zelin Peng; Fenghe Tang; Piao Yang; Yu Huang; Wei Shen
>
> **摘要:** Chest X-ray (CXR) reporting follows a region-based clinical workflow in which radiologists inspect anatomical regions and integrate localized findings into a final report. However, existing resources for CXR report generation provide these supervision signals in fragmented forms. We introduce MMRad-22K, a dataset that organizes regional textual observations, anatomical grounding coordinates, localized image evidence, and report targets into structured multimodal evidence units for CXR report generation. To motivate this formulation, we first compare different evidence formats for report generation and find that structured multimodal evidence is generally more useful than text-only or bounding box-based evidence. We then adapt a unified LVLM backbone using MMRad-22K and show that adaptation with multimodal evidence outperforms both textual-evidence adaptation and end-to-end adaptation on language and clinically oriented metrics. Under the same evaluation protocol, the adapted model also reaches a performance level comparable to several open-source LVLM references. Together, these results support MMRad-22K as a practical structured multimodal resource for training and evaluating CXR report generation aligned with clinical reading workflows.
>
---
#### [replaced 031] Event-based Motion & Appearance Fusion for 6D Object Pose Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08264](https://arxiv.org/pdf/2603.08264)**

> **作者:** Zhichao Li; Chiara Bartolozzi; Lorenzo Natale; Arren Glover
>
> **摘要:** Object pose tracking is a fundamental and essential task for robotics to perform tasks in the home and industrial settings. The most commonly used sensors to do so are RGB-D cameras, which can hit limitations in highly dynamic environments due to motion blur and frame-rate constraints. Event cameras have remarkable features such as high temporal resolution and low latency, which make them a potentially ideal vision sensors for object pose tracking at high speed. Even so, there are still only few works on 6D pose tracking with event cameras. In this work, we take advantage of the high temporal resolution and propose a method that uses both a propagation step fused with a pose correction strategy. Specifically, we use 6D object velocity obtained from event-based optical flow for pose propagation, after which, a template-based local pose correction module is utilized for pose correction. Our learning-free method has comparable performance to the state-of-the-art algorithms, and in some cases out performs them for fast-moving objects. The results indicate the potential for using event cameras in highly-dynamic scenarios where the use of deep network approaches are limited by low update rates.
>
---
#### [replaced 032] Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型在装配图与视频对齐任务中的研究，旨在解决2D装配图与实际视频间的 depiction gap。通过构建IKEA-Bench数据集并评估多种模型，发现视觉编码是提升对齐效果的关键。**

- **链接: [https://arxiv.org/pdf/2604.00913](https://arxiv.org/pdf/2604.00913)**

> **作者:** Zhuchenyang Liu; Yao Zhang; Yu Xiao
>
> **摘要:** 2D assembly diagrams are often abstract and hard to follow, creating a need for intelligent assistants that can monitor progress, detect errors, and provide step-by-step guidance. In mixed reality settings, such systems must recognize completed and ongoing steps from the camera feed and align them with the diagram instructions. Vision Language Models (VLMs) show promise for this task, but face a depiction gap because assembly diagrams and video frames share few visual features. To systematically assess this gap, we construct IKEA-Bench, a benchmark of 1,623 questions across 6 task types on 29 IKEA furniture products, and evaluate 19 VLMs (2B-38B) under three alignment strategies. Our key findings: (1) assembly instruction understanding is recoverable via text, but text simultaneously degrades diagram-to-video alignment; (2) architecture family predicts alignment accuracy more strongly than parameter count; (3) video understanding remains a hard bottleneck unaffected by strategy. A three-level mechanistic analysis further reveals that diagrams and video occupy disjoint ViT subspaces, and that adding text shifts models from visual to text-driven reasoning. These results identify visual encoding as the primary target for improving cross-depiction robustness. Project page: this https URL
>
---
#### [replaced 033] STAMBRIDGE: Spectral-Temporal Amplitude-aware Mid-Feature Bridge for EEG Visual Decoding
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2605.23137](https://arxiv.org/pdf/2605.23137)**

> **作者:** Jiahe Meng; Weiming Zeng; Yueyang Li; Bo Chai; Hongjie Yan; Zhiguo Zhang; Wai Ting Siok; Nizhuan Wang
>
> **摘要:** Electroencephalography (EEG) visual decoding remains challenging due to the modality gap between low-SNR neural signals and highly structured vision--language spaces, making direct cross-modal alignment unstable. To address this, we propose STAMBRIDGE, a versatile two-stage framework that sequentially tackles feature conditioning and cross-modal alignment. First, we introduce a Spectral-Temporal Amplitude-aware Modulation (STAM) to extract well-conditioned EEG representations. By replacing hard frequency masking with amplitude-derived soft channel weighting and multi-scale temporal convolutions, STAM explicitly preserves frequency-aware transients while reducing the risk of time-domain ringing artifacts. Building upon these robust neural features, we further introduce a model-agnostic Mid-Feature Semantic Bridge (MFSB) that constructs a regularized intermediate space through directed cross-modal interactions, enabling staged distillation and more stable semantic alignment. Experiments on the THINGS-EEG benchmark show competitive 200-way zero-shot retrieval performance, with 34.50\% Top-1 and 65.95\% Top-5 accuracy. In addition, embeddings learned by STAMBRIDGE produce semantically coherent image reconstructions with a diffusion model, demonstrating robust EEG-to-vision semantic alignment. The code is available at: this https URL.
>
---
#### [replaced 034] DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.02417](https://arxiv.org/pdf/2605.02417)**

> **作者:** Desong Yang; Mang Ye
>
> **备注:** ICML 2026. Project page: this https URL
>
> **摘要:** With recent advancements in large-scale pre-trained text-to-image (T2I) models, training-free image editing methods have demonstrated remarkable success. Typically, these methods involve adding noise to a clean image via an inversion process, followed by separate denoising steps for the reconstruction and editing paths during the forward process. However, since the reconstruction path is approximated using noisy latents from mismatched timesteps, existing methods inevitably suffer from accumulated drift, which fundamentally limits reconstruction fidelity. To address this challenge, we systematically analyze the inversion process within the flow transformer and propose DirectEdit, a simple yet effective editing method that eliminates the inherent reconstruction error without introducing additional neural function evaluations (NFEs). Unlike most prior works that attempt to rectify the inversion path, DirectEdit focuses on directly aligning the forward paths, enabling precise reconstruction and reliable feature sharing. Furthermore, we introduce a preservation mechanism based on attention feature injection and multi-branch mask-guided noise blending, which effectively balances fidelity and editability. Extensive experiments across diverse scenarios demonstrate that DirectEdit achieves efficient and accurate image editing, delivering superior performance that outperforms state-of-the-art methods. Code and examples are available at this https URL.
>
---
#### [replaced 035] Encoder-Free Human Motion Understanding via Structured Motion Descriptions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.21668](https://arxiv.org/pdf/2604.21668)**

> **作者:** Yao Zhang; Zhuchenyang Liu; Thomas Ploetz; Yu Xiao
>
> **摘要:** The world knowledge and reasoning capabilities of text-based large language models (LLMs) are advancing rapidly, yet current approaches to human motion understanding, including motion question answering and captioning, have not fully exploited these capabilities. Existing LLM-based methods typically learn motion-language alignment through dedicated encoders that project motion features into the LLM's embedding space, remaining constrained by cross-modal representation and alignment. Inspired by biomechanical analysis, where joint angles and body-part kinematics have long served as a precise descriptive language for human movement, we propose \textbf{Structured Motion Description (SMD)}, a rule-based, deterministic approach that converts joint position sequences into structured natural language descriptions of joint angles, body part movements, and global trajectory. By representing motion as text, SMD enables LLMs to apply their pretrained knowledge of body parts, spatial directions, and movement semantics directly to motion reasoning, without requiring learned encoders or alignment modules. We show that this approach goes beyond state-of-the-art results on both motion question answering (66.7\% on BABEL-QA, 90.1\% on HuMMan-QA) and motion captioning (R@1 of 0.584, CIDEr of 53.16 on HumanML3D), surpassing all prior methods. SMD additionally offers practical benefits: the same text input works across different LLMs with only lightweight LoRA adaptation (validated on 8 LLMs from 6 model families), and its human-readable representation enables interpretable attention analysis over motion descriptions. Code, data, and pretrained LoRA adapters are available at this https URL.
>
---
#### [replaced 036] In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models
- **分类: cs.AI; cs.CL; cs.CV; cs.NE**

- **简介: 该论文属于AI生成任务，旨在探索AI在开放性创作中的表现。通过复制Picbreeder系统，用视觉语言模型替代人类用户，研究其生成内容的差异与影响因素。**

- **链接: [https://arxiv.org/pdf/2605.23908](https://arxiv.org/pdf/2605.23908)**

> **作者:** Sam Earle; Kai Arulkumaran; Andrew Dai; Akarsh Kumar; Julian Togelius; Sebastian Risi
>
> **备注:** 26 pages, 21 figures, to be published at GECCO 2026
>
> **摘要:** We are in the midst of large-scale industrial and academic efforts to automate the processes of scientific, technological and creative production through AI-driven assistants. Historically, a fundamental property of these processes in their human form has been their open-endedness: their capacity for generating a seemingly endless supply of novel and meaningful new forms. Do artificial agents have any capacity for such fruitful unguided discovery? To answer this question, we turn to Picbreeder, the canonical exemplar of human-driven open-ended search, in which users collaboratively generated a diverse library of images through interactive evolution of small neural networks. We replicate Picbreeder, replacing human users with frontier Vision Language Models (VLMs). We observe clear qualitative differences between the output of our system and the historical human baseline, and attempt to characterize them using metrics of phylogenetic complexity and visual and semantic salience and novelty. In an effort to identify some of the causal factors contributing these differences, we study the addition of exploratory noise to the agents' selection process, of behavioral diversity between agents, and of narrative momentum in the form of memory of past actions. We make our code available at this https URL.
>
---
#### [replaced 037] RMPL: Relation-aware Multi-task Progressive Learning with Stage-wise Training for Multimedia Event Extraction
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多媒体事件抽取任务，解决数据稀缺下事件与论元识别问题。提出RMPL框架，结合多任务和阶段训练，提升跨模态事件表示学习效果。**

- **链接: [https://arxiv.org/pdf/2602.13748](https://arxiv.org/pdf/2602.13748)**

> **作者:** Yongkang Jin; Jianwen Luo; Jingjing Wang; Jianmin Yao; Yu Hong
>
> **备注:** Accepted by ACM ICMR 2026
>
> **摘要:** Multimedia Event Extraction (MEE) aims to identify events and their arguments from documents that contain both text and images. It requires grounding event semantics across different modalities. Progress in MEE is limited by the lack of annotated training data. M2E2 is the only established benchmark, but it provides annotations only for evaluation. This makes direct supervised training impractical. Existing methods mainly rely on cross-modal alignment or inference-time prompting with Vision--Language Models (VLMs). These approaches do not explicitly learn structured event representations and often produce weak argument grounding in multimodal settings. To address these limitations, we propose RMPL, a Relation-aware Multi-task Progressive Learning framework for MEE under low-resource conditions. RMPL incorporates heterogeneous supervision from unimodal event extraction and multimedia relation extraction with stage-wise training. The model is first trained with a unified schema to learn shared event-centric representations across modalities. It is then fine-tuned for event mention identification and argument role extraction using mixed textual and visual data. Experiments on the M2E2 benchmark with multiple VLMs show consistent improvements across different modality settings.
>
---
#### [replaced 038] From Clinical Intent to Clinical Model: Autonomous Coding-Agents for Clinician-driven AI Development
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.17110](https://arxiv.org/pdf/2604.17110)**

> **作者:** Zihao Zhao; Frederik Hauke; Juliana De Castilhos; Mathis Bode; Jakob Nikolas Kather; Sven Nebelung; Daniel Truhn
>
> **备注:** Code is available at this https URL
>
> **摘要:** Developing AI models that are useful in clinical practice, requires efficient collaboration between clinicians and AI developers. This poses a practical challenge: clinicians must repeatedly communicate and refine their requirements with AI developers before those requirements can be translated into executable model development. This iterative process is time-consuming, and even after repeated discussion, misalignment may still exist because the two sides do not fully share each other's expertise. Coding agents may help close this gap. They can write and refine code on their own, and they carry working knowledge of both medicine and AI to understand commands formulated by both medical experts and developers. We present a prototype that lets clinicians drive AI development directly. A clinician describes the task in plain language, and the system turns the description into a working pipeline, refines it through repeated experiments together with the clinician, and returns a model that meets the stated clinical objective. Across five clinical tasks, the system reliably produces models that matched the clinician's request and reached competitive performance. Most notably, on chest radiographs the system sharply reduced the model's reliance on chest drains, a well-known shortcut for pneumothorax classification, from 60% to 31% on one dataset and from 50% to 18% on another. Our results suggest that coding agents can shift clinical AI development toward a more clinician-driven mode, allowing domain experts to shape models directly instead of relaying requirements through specialized AI teams.
>
---
#### [replaced 039] CPPO: Contrastive Perception Policy Optimization for VLM Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.00501](https://arxiv.org/pdf/2601.00501)**

> **作者:** Ahmad Rezaei; Mohsen Gholami; Saeed Ranjbar Alvar; Kevin Cannons; Mohammad Asiful Hossain; Zhou Weimin; Yong Zhang; Mohammad Akbari
>
> **摘要:** We introduce CPPO, a Contrastive Perception Policy Optimization method for finetuning vision--language models (VLMs). Reliable perception is a core requirement for VLM-based agents that must reason and act in open-ended environments: faulty visual grounding cascades directly into faulty actions, hallucinated tool calls, and unsafe decisions. While reinforcement learning (RL) has significantly improved reasoning in language models, extending these advances to multimodal agents requires improving both perception and reasoning. Prior works address this challenge mainly through explicit perception rewards, which often require extra LLM judges, ground-truth annotations, or forced separation of perception from reasoning. CPPO addresses this limitation in a self-supervised manner by extending the RL objective with a Contrastive Perception Loss (CPL) that provides a direct learning signal for visual grounding. The contrastive objective encourages the model to become more sensitive to input visual information. To apply this signal effectively, CPPO identifies perception tokens using an entropy-shift mechanism in the model's output distributions under perturbed images and applies the contrastive loss selectively to those tokens during training. Experiments show that CPPO surpasses prior methods while avoiding extra models, making training more efficient and scalable, and yielding policies that are better suited to perception-critical agentic tasks.
>
---
#### [replaced 040] FasterVAR: Plug-and-Play Acceleration for Visual Autoregressive Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.16483](https://arxiv.org/pdf/2512.16483)**

> **作者:** Senmao Li; Kai Wang; Salman Khan; Fahad Shahbaz Khan; Jian Yang; Yaxing Wang
>
> **备注:** Accepted at ICML2026
>
> **摘要:** Visual Autoregressive (VAR) modeling departs from the next-token prediction paradigm of traditional Autoregressive (AR) models through next-scale prediction, enabling high-quality image generation. However, the VAR paradigm suffers from sharply increased computational complexity and running time at large-scale steps. Although existing acceleration methods reduce runtime for large-scale steps, but rely on manual step selection and overlook the varying importance of different stages in the generation process. To address this challenge, we present FasterVAR, a systematic study and plug-and-play acceleration framework for VAR models. Our analysis shows that early steps are critical for preserving semantic and structural consistency and should remain intact,while later steps mainly refine details and can be pruned or approximated for acceleration. Building on these insights, FasterVAR introduces a plug-and-play acceleration strategy that exploits semantic irrelevance and low-rank properties in late-stage computations, without requiring additional training. Our proposed FasterVAR achieves up to 3.4x speedup with almost no performance loss. consistently outperforming existing acceleration this http URL results highlight stage-aware design as a powerful principle for efficient visual autoregressive image generation.
>
---
#### [replaced 041] TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization
- **分类: cs.CV; cs.PF**

- **链接: [https://arxiv.org/pdf/2605.20150](https://arxiv.org/pdf/2605.20150)**

> **作者:** Chonghao Zhong; Linfeng Shi; Hua Chen; Tiecheng Sun; Hao Zhao; Binhang Yuan; Chaojian Li
>
> **备注:** Accepted to ICML 2026 as Spotlight. Website: this https URL
>
> **摘要:** Training 3D Gaussian Splatting (3DGS) at billion-primitive scale is fundamentally memory-bound: each Gaussian primitive carries a large attribute vector, and the aggregate parameter table quickly exceeds GPU capacity, limiting prior systems to tens of millions of Gaussians on commodity single-GPU hardware. We observe that 3DGS training is inherently sparse and trajectory-conditioned: each iteration activates only the Gaussians visible from the current camera batch, so GPU memory can serve as a working-set cache rather than a persistent parameter store. Building on this insight, we introduce TideGS, an out-of-core training framework that manages parameters across an SSD-CPU-GPU hierarchy via three synergistic techniques: block-virtualized geometry for SSD-aligned spatial locality, a hierarchical asynchronous pipeline to overlap I/O with computation, and trajectory-adaptive differential streaming that transfers only incremental working-set deltas between iterations. Experiments show that TideGS enables training with over one billion Gaussians on a single 24 GB GPU while achieving the best reconstruction quality among evaluated single-GPU baselines on large-scale scenes, scaling beyond prior out-of-core baselines (e.g., approximately 100M Gaussians) and standard in-memory training (e.g., approximately 11M Gaussians).
>
---
#### [replaced 042] Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.15523](https://arxiv.org/pdf/2605.15523)**

> **作者:** Hongxi Li; Tong Wang; Chengjing Wu; Tianbao Liu; Jiangtao Yao; Xiaochao Qu; Xinxiao Wu; Luoqi Liu; Ting Liu
>
> **备注:** ICML 2026
>
> **摘要:** Scene text editing aims to modify text in a target region of an image while preserving surrounding background style and texture. Existing methods rely solely on image background information while neglecting the visual details of target regions, which discards stylistic features in the original text and essentially degrades the task to text rendering. Moreover, the conditions imposed by pre-trained glyph encoder limit the scope of editable text. To address these issues, this paper proposes a self-prompting scene text editing method that constructs style and glyph prompts directly from the original image, without introducing additional style or glyph encoders. We employ a two-stage training strategy: the diffusion transformer is first trained on large-scale self-supervised data and then refined using a small set of paired images. By leveraging the in-context learning capability of the Multi-Modal Diffusion Transformer (MM-DiT), it achieves open-vocabulary and style-consistent text editing. Experimental results on various languages demonstrate that our method achieves the state-of-the-art performance in both text accuracy and style consistency. Our project page: this http URL.
>
---
#### [replaced 043] Artemis: Structured Visual Reasoning for Perception Policy Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01988](https://arxiv.org/pdf/2512.01988)**

> **作者:** Wei Tang; Yanpeng Sun; Shan Zhang; Weihao Bo; Xiaofan Li; Piotr Koniusz; Wei Li; Na Zhao; Zechao Li
>
> **摘要:** Recent reinforcement-learning frameworks for visual perception policy usually incorporate intermediate reasoning chains expressed in natural language. Empirical observations indicate that such purely linguistic intermediate reasoning often reduces performance on perception tasks. We argue that the core issue lies not in reasoning per se but in the form of reasoning: while these chains perform semantic reasoning in an unstructured linguistic space, \textbf{visual perception requires reasoning in a spatial and object-centric space}. In response, we introduce \textbf{Artemis}, a perception-policy learning method that performs structured visual reasoning, where each intermediate step is represented as a (label, bounding-box) pair capturing a verifiable visual state. This design enables explicit tracking of intermediate states, direct supervision for proposal quality, and avoids ambiguity introduced by language-based reasoning. Building upon verifiable and spatially grounded reasoning chains, Artemis provides a unified architecture for diverse perceptual tasks, without requiring the task-specific designs relied upon by prior perceptual policy models. Trained using grounding and detection sampeles in natural image domains, Artemis generalizes to counting and geometric perception tasks. At its core, a spatially grounded, object-centric chain rule provides a principled foundation for scalable and general perceptual policies.
>
---
#### [replaced 044] LUVE : Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.11564](https://arxiv.org/pdf/2602.11564)**

> **作者:** Chen Zhao; Jiawei Chen; Hongyu Li; Zhuoliang Kang; Shilin Lu; Xiaoming Wei; Kai Zhang; Jian Yang; Ying Tai
>
> **备注:** ICML 2026
>
> **摘要:** Recent advances in video diffusion models have significantly improved visual quality, yet ultra-high-resolution (UHR) video generation remains a formidable challenge due to the compounded difficulties of motion modeling, semantic planning, and detail synthesis. To address these limitations, we propose \textbf{LUVE}, a \textbf{L}atent-cascaded \textbf{U}HR \textbf{V}ideo generation framework built upon dual frequency \textbf{E}xperts. LUVE employs a three-stage architecture comprising low-resolution motion generation for motion-consistent latent synthesis, video latent upsampling that performs resolution upsampling directly in the latent space to mitigate memory and computational overhead, and high-resolution content refinement that integrates low-frequency and high-frequency experts to jointly enhance semantic coherence and fine-grained detail generation. Extensive experiments demonstrate that our LUVE achieves superior photorealism and content fidelity in UHR video generation, and comprehensive ablation studies further validate the effectiveness of each component. The project is available at \href{this https URL}{this https URL}.
>
---
#### [replaced 045] EpiAgent: An Agent-Centric System for Ancient Inscription Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.09367](https://arxiv.org/pdf/2604.09367)**

> **作者:** Shipeng Zhu; Ang Chen; Na Nie; Pengfei Fang; Min-Ling Zhang; Hui Xue
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Ancient inscriptions, as repositories of cultural memory, have suffered from centuries of environmental and human-induced degradation. Restoring their intertwined visual and textual integrity poses one of the most demanding challenges in digital heritage preservation. However, existing AI-based approaches often rely on rigid pipelines, struggling to generalize across such complex and heterogeneous real-world degradations. Inspired by the skill-coordinated workflow of human epigraphers, we propose EpiAgent, an agent-centric system that formulates inscription restoration as a hierarchical planning problem. Following an Observe-Conceive-Execute-Reevaluate paradigm, an LLM-based central planner orchestrates collaboration among multimodal analysis, historical experience, specialized restoration tools, and iterative self-refinement. This agent-centric coordination enables a flexible and adaptive restoration process beyond conventional single-pass methods. Across real-world degraded inscriptions, EpiAgent achieves superior restoration quality and stronger generalization compared to existing methods. Our work marks an important step toward expert-level agent-driven restoration of cultural heritage. The code is available at this https URL.
>
---
#### [replaced 046] HyperBones: Realtime Bone-driven Neural Garment Simulation with Hypernetwork Conditioning
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2605.20460](https://arxiv.org/pdf/2605.20460)**

> **作者:** Astitva Srivastava; Hsiao-Yu Chen; Ryan Goldade; Philipp Herholz; Zhongshi Jiang; Gene Wei-Chin Lin; Lingchen Yang; Nikolaos Sarafianos; Tuur Stuyck; Doug Roble; Avinash Sharma; Egor Larionov
>
> **摘要:** Recent advances in garment simulation have brought high-quality results closer to real-time performance. Physics-based simulators can produce accurate motion, but remain too computationally expensive for interactive applications. In contrast, linear blend skinning is efficient, but cannot capture the complex dynamics of loose-fitting garments, often leading to unrealistic motion and visual artifacts. Neural methods offer a promising alternative, yet they still struggle to animate loose clothing plausibly under strict runtime constraints. We present a fast and physically plausible approach for dynamic garment simulation. Our method trains a reduced-space neural dynamics simulator composed of independent coarse- and fine-level components. At the coarse level, the garment is driven by a set of virtual bones integrated with a lightweight neural network. Fine-scale wrinkle details are then recovered using a trained convolutional neural map. By decoupling identity-specific computation from real-time neural integration, our architecture maintains high performance while supporting diverse body shapes and motions. We further introduce an effective physics-supervision scheme that enables accurate results without relying on an external simulator. Experiments show that our method produces physically plausible garment dynamics, generalizes across a range of motions and body shapes, and supports a fixed set of garments. Our simulator runs at 300+ FPS on a commodity GPU, making it suitable for real-time applications.
>
---
#### [replaced 047] Prototyping an End-to-End Multi-Modal Tiny-CNN for Cardiovascular Sensor Patches
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18668](https://arxiv.org/pdf/2510.18668)**

> **作者:** Mustafa Fuad Rifet Ibrahim; Tunc Alkanat; Felix Manthey; Maurice Meijer; Alexander Schlaefer; Peer Stelldinger
>
> **备注:** 11 pages, 2 figures. Extended version of our 2024 IEEE PerCom paper, with direct on-device energy measurements, a BLE communication benchmark, architecture comparisons, and an extended evaluation. Submitted to Biomedical Signal Processing and Control; Fixed typos
>
> **摘要:** The vast majority of cardiovascular diseases may be preventable if early signs and risk factors are detected. Cardiovascular monitoring with body-worn sensor devices like sensor patches allows for the detection of such signs while preserving the freedom and comfort of patients. However, the analysis of the sensor data must be robust, reliable, efficient, and highly accurate. Deep learning methods can automate data interpretation, reducing the workload of clinicians. In this work, we analyze the feasibility of applying deep learning models to the classification of synchronized electrocardiogram (ECG) and phonocardiogram (PCG) recordings on resource-constrained medical edge devices. We propose a convolutional neural network with early fusion of data to solve a binary classification problem. The model is trained and validated on the synchronized ECG and PCG recordings from the Physionet Challenge 2016 dataset. Our approach reduces memory footprint and compute cost by approximately three orders of magnitude compared with the state-of-the-art while maintaining competitive accuracy. We further demonstrate the applicability of the proposed model on medical edge devices by measuring its energy consumption on a microcontroller equipped with a neural processing unit (NPU) and benchmarking the energy of Bluetooth Low Energy (BLE) communication on a representative BLE evaluation kit across a range of payload sizes. The comparison confirms that on-device inference can be more energy efficient than continuous data streaming.
>
---
#### [replaced 048] Super-Resolved Canopy Height Mapping from Sentinel-2 Time Series Using Airborne LiDAR HD Reference Data across Metropolitan France
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.11524](https://arxiv.org/pdf/2512.11524)**

> **作者:** Ekaterina Kalinicheva; Florian Helen; Stéphane Mermoz; Florian Mouret; Milena Planells
>
> **摘要:** Fine-scale forest monitoring is essential for understanding canopy structure and its dynamics, which are key indicators of carbon stocks, biodiversity, and forest health. Deep learning is particularly effective for this task, as it integrates spectral, temporal, and spatial signals that jointly reflect the canopy structure. To address this need, we introduce THREASURE-Net, a novel end-to-end framework for Tree Height Regression And Super-Resolution. The model is trained on Sentinel-2 time series using reference height metrics derived from LiDAR HD data at multiple spatial resolutions over Metropolitan France to produce annual height maps. We evaluate three model variants, producing tree-height predictions at 2.5 m, 5 m, and 10 m resolution. THREASURE-Net does not rely on any pretrained model nor on reference very high resolution optical imagery to train its super-resolution module; instead, it learns solely from LiDAR-derived height information. Our approach outperforms existing state-of-the-art methods based on Sentinel data and is competitive with methods based on very high resolution imagery. It can be deployed to generate high-precision annual canopy-height maps, achieving mean absolute errors of 2.63 m, 2.70 m, and 2.88 m at 2.5 m, 5 m, and 10 m resolution, respectively. These results highlight the potential of THREASURE-Net for scalable and cost-effective structural monitoring of temperate forests using only freely available satellite data. The source code for THREASURE-Net is available at: this https URL.
>
---
#### [replaced 049] Smoothing Slot Attention Iterations and Recurrences
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05417](https://arxiv.org/pdf/2508.05417)**

> **作者:** Rongzhen Zhao; Wenyan Yang; Juho Kannala; Joni Pajarinen
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Slot Attention (SA) lies at the heart of mainstream Object-Centric Learning (OCL). Image features can be aggregated into object-level representations by SA \textit{iteratively} refining cold-start query slots. For video, such aggregation proceeds by SA \textit{recurrently} shared across frames, with queries cold-started on the first frame while transitioned from the previous frame's slots thereafter. However, cold-start queries lack sample-specific cues thus hindering precise aggregation on image or video's first frame; Non-first frames' queries are already sample-specific thus requiring aggregation transforms different from the first frame. We address these issues with our \textit{SmoothSA}: (1) To smooth SA iterations on image or video's first frame, we \textit{preheat} cold-start queries with rich input-feature information, by a tiny module self-distilled inside OCL; (2) To smooth SA recurrences across video's first and non-first frames, we \textit{differentiate} the homogeneous aggregation transforms by using full and single iterations respectively. Comprehensive experiments on object discovery, recognition and visual reasoning validate our method's effectiveness. Further visual analyses illuminate the underline mechanisms. Our \textit{source code}, \textit{model checkpoints} and \textit{training logs} are provided on this https URL.
>
---
#### [replaced 050] LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出LocateAnything，解决视觉-语言定位与检测任务中的效率与精度问题。通过并行框解码技术提升解码速度和定位准确性，并构建大规模数据集增强模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27365](https://arxiv.org/pdf/2605.27365)**

> **作者:** Shihao Wang; Shilong Liu; Yuanguo Kuang; Xinyu Wei; Yangzhou Liu; Zhiqi Li; Yunze Man; Guo Chen; Andrew Tao; Guilin Liu; Jan Kautz; Lei Zhang; Zhiding Yu
>
> **备注:** fix github link
>
> **摘要:** Vision-language models (VLMs) commonly formulate visual grounding and detection as a coordinate-token generation problem, serializing each 2D box into multiple 1D tokens that are learned and decoded largely independently. This token-by-token decoding mismatches the coupled structure of box geometry and creates a practical inference bottleneck due to strictly sequential generation. We introduce LocateAnything, a unified generative grounding and detection framework based on Parallel Box Decoding (PBD). By decoding geometric elements such as bounding boxes and points as atomic units in a single step, LocateAnything preserves intra-box geometric coherence and unlocks substantial parallelism. We show that PBD improves both decoding throughput and localization accuracy. We further develop a scalable data engine and curate LocateAnything-Data, a large-scale dataset with more than 138 million training samples, substantially increasing data diversity for high-precision localization. Extensive evaluations show that LocateAnything advances the speed-accuracy frontier, achieving significantly higher decoding throughput while improving high-IoU localization quality across diverse benchmarks. The results highlight the complementary benefits of Parallel Box Decoding and large-scale training data in enabling efficient and precise unified visual grounding and detection.
>
---
#### [replaced 051] SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.21666](https://arxiv.org/pdf/2601.21666)**

> **作者:** Ahmed Y. Radwan; Christos Emmanouilidis; Hina Tabassum; Deval Pandya; Shaina Raza
>
> **摘要:** Multimodal Large Language Models (MLLMs) are a major focus of recent AI research. However, most prior work focuses on static image understanding, while their ability to process sequential audio-video data remains underexplored. This gap highlights the need for a high-quality benchmark to systematically evaluate MLLM performance in a real-world setting. We introduce SONIC-O1, a comprehensive, fully human-verified benchmark of 60 hours (231 clips) spanning 13 real-world conversational domains with 4,958 annotations and demographic metadata. SONIC-O1 evaluates three capabilities: open-ended summarization, multiple-choice question (MCQ) answering, and temporal localization with supporting rationales (reasoning). Across closed- and open-source models, we find that the MCQ accuracy shows the smallest gap between model families, but the best closed-source model outperforms the best open-source model by 22.6% on temporal localization. We further observe accuracy gaps of up to 21.4% on temporal localization across demographic groups, indicating persistent disparities in model behaviour. SONIC-O1 provides an open evaluation suite for temporally grounded and demographically robust multimodal understanding. SONIC-O1 is publicly available for research: Project page (this https URL), Dataset (this https URL), GitHub (this https URL), Leaderboard (this https URL).
>
---
#### [replaced 052] Beyond Interpretability: When, Why, and How Sparse Autoencoders Enable Label-Free Visual Steering
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.01247](https://arxiv.org/pdf/2506.01247)**

> **作者:** Gerasimos Chatzoudis; Zhuowei Li; Gemma E. Moran; Hao Wang; Dimitris N. Metaxas
>
> **摘要:** Sparse Autoencoders (SAEs) are increasingly used to interpret foundation models, but their role as an actionable intervention space remains less understood, especially in vision. We study whether sparse visual features can be used not only for post-hoc analysis, but also to steer frozen vision-language models. We introduce Visual Sparse Steering (VS2), a label-free method that trains a top-$k$ SAE on unlabeled activations from a frozen CLIP image encoder and, at test time, constructs an interpretable steering vector by amplifying the input's active sparse features and decoding the induced change. We show that this procedure admits a closed-form decomposition as centroid-deviation steering: each input is moved along its deviation from the SAE-learned centroid. The residual term is controlled exactly by the SAE's per-sample reconstruction error, measured by FVU, yielding an FVU-based residual bound and motivating a reliability gate that falls back to zero-shot CLIP when SAE reconstruction is unreliable. With target-domain SAEs trained on unlabeled CLIP image-encoder activations, VS2 improves zero-shot accuracy across nine image-classification datasets, achieving gains up to $+4.12\%$ with less than $0.1\%$ additional inference compute. Finally, a controlled upper-bound study, VS2++, shows that selective amplification of sparse features can yield gains up to $+21.44\%$, exposing a reconstruction-vs-task saliency gap: features salient for reconstruction need not align with features useful for downstream prediction.
>
---
#### [replaced 053] Many Dialects, Many Languages, One Cultural Lens: Evaluating Multilingual VLMs for Bengali Culture Understanding Across Historically Linked Languages and Regional Dialects
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型评估任务，旨在解决 Bengali 文化在多语言和方言中的理解问题。构建了 BanglaVerse 基准，评估 VLM 在不同语言和方言中的文化理解能力。**

- **链接: [https://arxiv.org/pdf/2603.21165](https://arxiv.org/pdf/2603.21165)**

> **作者:** Nurul Labib Sayeedi; Md. Faiyaz Abdullah Sayeedi; Shubhashis Roy Dipta; Rubaya Tabassum; Ariful Ekraj Hridoy; Mehraj Mahmood; Mahbub E Sobhani; Md. Tarek Hasan; Swakkhar Shatabda
>
> **备注:** this https URL
>
> **摘要:** Bangla culture is richly expressed through region, dialect, history, food, politics, media, and everyday visual life, yet it remains underrepresented in multimodal evaluation. To address this gap, we introduce BanglaVerse, a culturally grounded benchmark for evaluating multilingual vision-language models (VLMs) on Bengali culture across historically linked languages and regional dialects. Built from 1,152 manually curated images across nine domains, the benchmark supports visual question answering and captioning, and is expanded into four languages and five Bangla dialects, yielding ~32.2K artifacts. Our experiments show that evaluating only standard Bangla overestimates true model capability: performance drops under dialectal variation, especially for caption generation, while historically linked languages such as Hindi and Urdu retain some cultural meaning but remain weaker for structured reasoning. Across domains, the main bottleneck is missing cultural knowledge rather than visual grounding alone, with knowledge-intensive categories. These findings position BanglaVerse as a more realistic test bed for measuring culturally grounded multimodal understanding under linguistic variation.
>
---
#### [replaced 054] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CogVLA，解决视觉-语言-动作模型的效率与性能问题，通过指令驱动路由和稀疏化提升效果，适用于机器人任务。**

- **链接: [https://arxiv.org/pdf/2508.21046](https://arxiv.org/pdf/2508.21046)**

> **作者:** Wei Li; Renshan Zhang; Rui Shao; Jie He; Liqiang Nie
>
> **备注:** Accepted to NeurIPS 2025, Project Page: this https URL
>
> **摘要:** Recent Vision-Language-Action (VLA) models built on pre-trained Vision-Language Models (VLMs) require extensive post-training, resulting in high computational overhead that limits scalability and this http URL propose CogVLA, a Cognition-Aligned Vision-Language-Action framework that leverages instruction-driven routing and sparsification to improve both efficiency and performance. CogVLA draws inspiration from human multimodal coordination and introduces a 3-stage progressive architecture. 1) Encoder-FiLM based Aggregation Routing (EFA-Routing) injects instruction information into the vision encoder to selectively aggregate and compress dual-stream visual tokens, forming a instruction-aware latent representation. 2) Building upon this compact visual encoding, LLM-FiLM based Pruning Routing (LFP-Routing) introduces action intent into the language model by pruning instruction-irrelevant visually grounded tokens, thereby achieving token-level sparsity. 3) To ensure that compressed perception inputs can still support accurate and coherent action generation, we introduce V-L-A Coupled Attention (CAtten), which combines causal vision-language attention with bidirectional action parallel decoding. Extensive experiments on the LIBERO benchmark and real-world robotic tasks demonstrate that CogVLA achieves state-of-the-art performance with success rates of 97.4% and 70.0%, respectively, while reducing training costs by 2.5-fold and decreasing inference latency by 2.8-fold compared to OpenVLA. CogVLA is open-sourced and publicly available at this https URL.
>
---
#### [replaced 055] Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，解决细粒度视觉理解问题。提出Vision-OPD框架，通过自蒸馏提升模型对关键视觉证据的聚焦能力。**

- **链接: [https://arxiv.org/pdf/2605.18740](https://arxiv.org/pdf/2605.18740)**

> **作者:** Qianhao Yuan; Jie Lou; Xing Yu; Hongyu Lin; Le Sun; Xianpei Han; Yaojie Lu
>
> **备注:** Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) still struggle with fine-grained visual understanding, where answers often depend on small but decisive evidence in the full image. We observe a regional-to-global perception gap: the same MLLM answers fine-grained questions more accurately when conditioned on evidence-centered crops than on the corresponding full images, suggesting that many failures stem from difficulty to focus on relevant evidence rather than insufficient local recognition ability. Motivated by this observation, we propose Vision-OPD (Vision On-Policy Distillation), a regional-to-global self-distillation framework that transfers the model's own privileged regional perception to its full-image policy. Vision-OPD instantiates two conditional policies from the same MLLM: a crop-conditioned teacher and a full-image-conditioned student. The student generates on-policy rollouts, and Vision-OPD minimizes token-level divergence between the teacher and student next-token distributions along these rollouts. This enables the model to internalize the benefit of visual zooming without external teacher models, ground-truth labels, reward verifiers, or inference-time tool use. Experiments on multiple fine-grained visual understanding benchmarks show that Vision-OPD models achieve competitive or superior performance against much larger open-source, closed-source, and "Thinking-with-Images" agentic models.
>
---
#### [replaced 056] SAFE-Diff: Scale-Aware Attention and Feature-Dispersive Diffusion with Uncertainty Estimation for Contrast-Enhanced Breast MRI Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.25767](https://arxiv.org/pdf/2605.25767)**

> **作者:** Tianyu Zhang; Xinglong Liang; Jarek van Dijk; Luyi Han; Chunyao Lu; Antonio Portaluri; Xinghe Xie; Yaofei Duan; Nika Rasoolzadeh; Xin Wang; Yuan Gao; Muzhen He; Yue Sun; Jonas Teuwen; Tao Tan; Ritse Mann
>
> **备注:** Early accepted by MICCAI 2026
>
> **摘要:** Synthesizing high fidelity contrast enhanced MRI is clinically valuable for safer and more efficient breast cancer screening, yet remains challenging due to complex lesion textures and heterogeneous enhancement patterns.
>
---
#### [replaced 057] IAR2: Improving Autoregressive Visual Generation with Semantic-Detail Associated Token Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.06928](https://arxiv.org/pdf/2510.06928)**

> **作者:** Ran Yi; Teng Hu; Zihan Su; Jiangning Zhang; Lizhuang Ma
>
> **摘要:** Autoregressive models have emerged as a powerful paradigm for visual content creation, but often overlook the intrinsic structural properties of visual data. Our prior work, IAR, initiated a direction to address this by reorganizing the visual codebook based on embedding similarity, thereby improving generation robustness. However, it is constrained by the rigidity of pre-trained codebooks and the inaccuracies of hard, uniform clustering. To overcome these limitations, we propose IAR2, an advanced autoregressive framework that enables a hierarchical semantic-detail synthesis process. At the core of IAR2 is a novel Semantic-Detail Associated Dual Codebook, which decouples image representations into a semantic codebook for global semantic information and a detail codebook for fine-grained refinements. It expands the quantization capacity from a linear to a polynomial scale, significantly enhancing expressiveness. To accommodate this dual representation, we propose a Semantic-Detail Autoregressive Prediction scheme coupled with a Local-Context Enhanced Autoregressive Head, which performs hierarchical prediction-first the semantic token, then the detail token-while leveraging a local context window to enhance spatial coherence. Furthermore, for conditional generation, we introduce a Progressive Attention-Guided Adaptive CFG mechanism that dynamically modulates the guidance scale for each token based on its relevance to the condition and its temporal position in the generation sequence, improving conditional alignment without sacrificing realism. Extensive experiments demonstrate that IAR2 sets a new state-of-the-art for autoregressive image generation, achieving a FID of 1.50 on ImageNet. Our model not only surpasses previous methods in performance but also demonstrates superior computational efficiency, highlighting the effectiveness of our structured, coarse-to-fine generation strategy.
>
---
#### [replaced 058] Text-Only Data Synthesis for Vision Language Model Training
- **分类: cs.AI; cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2503.22655](https://arxiv.org/pdf/2503.22655)**

> **作者:** Xiaomin Yu; Wenjie Zhang; Ziyue Qiao; Chengwei Qin; Hui Xiong
>
> **摘要:** Training vision-language models (VLMs) typically requires large-scale, high-quality image-text pairs, but collecting or synthesizing such data is costly. In contrast, text data is abundant and inexpensive, prompting the question: can high-quality multimodal training data be synthesized purely from text? To tackle this, we propose a cross-integrated three-stage multimodal data synthesis framework, which generates two datasets: Unicorn-1.2M and Unicorn-471K-Instruction. In Stage 1: Diverse Caption Data Synthesis, we construct 1.2M semantically diverse high-quality captions by expanding sparse caption seeds using large language models (LLMs). In Stage 2: Instruction-Tuning Data Generation, we further process 471K captions into multi-turn instruction-tuning tasks to support complex reasoning. Finally, in Stage 3: Modality Representation Transfer, these textual captions representations are transformed into visual representations, resulting in diverse synthetic image representations. This three-stage process enables us to construct Unicorn-1.2M for pretraining and Unicorn-471K-Instruction for instruction-tuning, without relying on real images. By eliminating the dependency on real images while maintaining data quality and diversity, our framework offers a cost-effective and scalable solution for VLMs training.
>
---
#### [replaced 059] Case-Aware Medical Image Classification with Multimodal Knowledge Graphs and Reliability-Guided Refinement
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.22547](https://arxiv.org/pdf/2605.22547)**

> **作者:** Yiming Xu; Yixuan Liu; Yuhang Zhang; Ling Zheng; Yihan Wang; Qi Song
>
> **摘要:** Deep learning has brought significant progress to medical image classification, yet most existing methods still rely on isolated visual evidence and cannot effectively leverage similar cases or external knowledge. In clinical practice, diagnosis is typically supported by similar historical cases and their associated symptoms. To explicitly model this evidence-based diagnostic process, we propose a case-aware reasoning framework driven by multimodal knowledge graphs for medical image classification. Specifically, we construct a case-aware multimodal knowledge graph as a structured diagnostic memory, where diseases, images, and symptoms are hierarchically organized. Given an input image, our method adaptively retrieves similar cases from this memory and extracts their corresponding case-centered subgraphs. We further introduce a knowledge propagation and injection mechanism, in which an image-centric Graph Attention Network aggregates heterogeneous semantics into case-based features, followed by a bidirectional cross-modal attention mechanism that injects these features into visual representations for cross-modal alignment. To mitigate noisy retrieval, we design a confidence-calibrated decision refinement scheme that estimates the reliability of each retrieved case by jointly considering prediction confidence and sample similarity, and reweights its contribution to the final prediction, providing interpretable case-level evidence. Extensive experiments on multiple medical imaging datasets demonstrate that our approach consistently outperforms strong baselines, while ablation and qualitative analyses validate its effectiveness and interpretability. The code is available at this https URL.
>
---
#### [replaced 060] FrequencyCT: Frequency Domain Self-supervised Low-dose CT Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.10583](https://arxiv.org/pdf/2605.10583)**

> **作者:** Guoquan Wei; Liu Shi; Chong Chen; Qiegen Liu
>
> **摘要:** Despite extensive research on computed tomography (CT) denoising, few studies exploit projection-domain data characteristics to mitigate noise correlation. To bridge this gap, this work proposes FrequencyCT, the first zero-shot self-supervised method for pseudo-sample generation in the frequency domain for low-dose CT denoising. Specifically, by exploiting the distinct frequency-domain distributions of noise and true signal, a regional low-frequency anchoring technique is proposed. Applying phase-preserving noise and mask perturbations to the high-frequency region generates pseudo-samples for self-supervision. Driven by the exponential correlation between noise variance of noisy projections and the underlying true signal, consistent data truncation is applied to the generated samples to stabilize optimization gradients. Evaluation results on multiple public and real datasets confirm the clinical application potential of this research, which provides an innovative perspective for the field of denoising. The code is available at: this https URL.
>
---
#### [replaced 061] Chirpy3D: Part-Aware Multi-View Diffusion for Creative Fine-Grained Object Generation
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2501.04144](https://arxiv.org/pdf/2501.04144)**

> **作者:** Kam Woh Ng; Jing Yang; Jia Wei Sii; Chee Seng Chan; Jiankang Deng; Yi-Zhe Song; Tao Xiang; Xiatian Zhu
>
> **备注:** 20 pages. Code at this https URL
>
> **摘要:** Understanding and generating the fine-grained structure of objects -- such as birds with species-specific beaks, wings, and tails -- is a long-standing challenge in computer vision. We propose Chirpy3D, a part-aware multi-view diffusion framework that learns a hierarchical part latent space from unposed 2D images, using only off-the-shelf 2D part segmentation masks as spatial guidance -- without requiring any 3D data, camera poses, or manual part annotations. This latent space enables intuitive part-level swapping, interpolation, and zero-shot composition. A self-supervised feature consistency loss further encourages structural alignment across views, allowing coherent generation even with hybrid or unseen part combinations. Our core contribution is the controllable part-aware latent space and multi-view diffusion model. Downstream 3D generation is supported via any differentiable renderer such as NeRF but is orthogonal to the main framework, making Chirpy3D a flexible foundation for creative object generation in the absence of structured 3D data. Code is released at this https URL.
>
---
#### [replaced 062] HiRQA: Hierarchical Ranking and Quality Alignment for Opinion-Unaware Image Quality Assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.15130](https://arxiv.org/pdf/2508.15130)**

> **作者:** Vaishnav Ramesh; Haining Wang; Md Jahidul Islam
>
> **备注:** Accepted for publication in Machine Vision and Applications
>
> **摘要:** Despite significant progress in no-reference image quality assessment (NR-IQA), dataset biases and reliance on subjective labels continue to hinder their generalization performance. We propose HiRQA (Hierarchical Ranking and Quality Alignment), a self-supervised, opinion-unaware framework that offers a hierarchical, quality-aware embedding through a combination of ranking and contrastive learning. Unlike prior approaches that depend on pristine references or auxiliary modalities at inference time, HiRQA predicts quality scores using only the input image. We introduce a novel higher-order ranking loss that supervises quality predictions through relational ordering across distortion pairs, along with an embedding distance loss that enforces consistency between feature distances and perceptual differences. A training-time contrastive alignment loss, guided by structured textual prompts, further enhances the learned representation. Trained only on synthetic image distortions, HiRQA generalizes to authentic degradations, as demonstrated through comprehensive evaluations on various unseen distortions such as lens flare, haze, motion blur, and low-light conditions. For real-time deployment, we introduce HiRQA-S, a lightweight variant with an inference time of only 3.5 ms per image. Extensive experiments across synthetic and authentic benchmarks validate HiRQA's competitive performance, strong generalization ability, and scalability. The HiRQA model and inference pipeline are available at: this https URL.
>
---
#### [replaced 063] Alterbute: Editing Intrinsic Attributes of Objects in Images
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.10714](https://arxiv.org/pdf/2601.10714)**

> **作者:** Tal Reiss; Daniel Winter; Matan Cohen; Alex Rav-Acha; Yael Pritch; Ariel Shamir; Yedid Hoshen
>
> **备注:** ICML 2026. Project page is available at this https URL
>
> **摘要:** We introduce Alterbute, a diffusion-based method for editing an object's intrinsic attributes in an image. We allow changing color, texture, material, and even the shape of an object, while preserving its perceived identity and scene context. Existing approaches either rely on unsupervised priors that often fail to preserve identity or use overly restrictive supervision that prevents meaningful intrinsic variations. Our method relies on: (i) a relaxed training objective that allows the model to change both intrinsic and extrinsic attributes conditioned on an identity reference image, a textual prompt describing the target intrinsic attributes, and a background image and object mask defining the extrinsic context. At inference, we restrict extrinsic changes by reusing the original background and object mask, thereby ensuring that only the desired intrinsic attributes are altered; (ii) Visual Named Entities (VNEs) - fine-grained visual identity categories (e.g., ''Porsche 911 Carrera'') that group objects sharing identity-defining features while allowing variation in intrinsic attributes. We use a vision-language model to automatically extract VNE labels and intrinsic attribute descriptions from a large public image dataset, enabling scalable, identity-preserving supervision. Alterbute outperforms existing methods on identity-preserving object intrinsic attribute editing.
>
---
#### [replaced 064] Semantic Robustness Probing via Inpainting: An Interactive Tool for Safety-Critical Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.27155](https://arxiv.org/pdf/2605.27155)**

> **作者:** Nico Steckhan; Krutarth Prajapati; Weija Shao; Silvia Vock
>
> **摘要:** Testing object detectors in safety-critical domains requires semantically meaningful probes beyond pixel-level corruptions. We present SemProbe, a tool for semantic robustness probing: users upload deployment images, create masks manually or automatically, select operational design domain-derived factors (or custom prompts), and run diffusion-based controlled inpainting. The system supports batch jobs, parallel seed/workflow variations, and configurable generation parameters. After each output, model inference runs automatically and displays annotated before/after comparisons with performance deltas. All probes are logged as structured artifacts, enabling traceable robustness evidence aligned with safety evaluation workflows. We demonstrate \textsc{SemProbe} on hand detection for dimension saws, targeting factors from insurance-oriented test criteria.
>
---
#### [replaced 065] Noise Scheduling as Information-Guided Allocation in Diffusion Training
- **分类: cs.LG; cs.AI; cs.CV; cs.IT**

- **链接: [https://arxiv.org/pdf/2602.18647](https://arxiv.org/pdf/2602.18647)**

> **作者:** Gabriel Raya; Bac Nguyen; Georgios Batzolis; Yuhta Takida; Dejan Stancevic; Naoki Murata; Chieh-Hsin Lai; Yuki Mitsufuji; Luca Ambrogioni
>
> **摘要:** We introduce InfoNoise, an online adaptive noise schedule for diffusion training that reallocates optimization effort toward noise levels where denoising is most informative. Together with loss weighting, a noise schedule induces an effective allocation across denoising problems, often fixed before informative noise levels are known. InfoNoise makes this allocation data-adaptive by estimating a conditional-entropy-rate profile from denoising losses during training, without auxiliary models or offline search. Through I--MMSE, this profile identifies where noisy observations rapidly reduce uncertainty about the clean sample and guides adaptation of the training noise distribution. It changes only this distribution, keeping the objective, weighting, and parameterization fixed. On image benchmarks, where schedules have been extensively tuned, InfoNoise matches or slightly exceeds strong baselines and can reach the same quality with fewer updates. On representation, sequence, and modality shifts, including DNA and language generation, InfoNoise improves over fixed and adaptive baselines and reaches target quality with up to $3\times$ less training compute. These results establish the conditional-entropy-rate profile as the data-dependent target for noise schedule design and make online adaptation a practical alternative to manual schedule search.
>
---
#### [replaced 066] Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决VLM在声称进行视觉复核时是否真的“看见”问题。通过图像替换实验，发现模型多为“说”而非“看”，揭示其视觉理解的局限性。**

- **链接: [https://arxiv.org/pdf/2605.15864](https://arxiv.org/pdf/2605.15864)**

> **作者:** Chufan Shi; Cheng Yang; Yaokang Wu; Linghao Jin; Bo Shui; Taylor Berg-Kirkpatrick; Xuezhe Ma
>
> **备注:** ICML 2026 Oral
>
> **摘要:** Vision-Language Models (VLMs) often produce self-reflective statements like "let me check the figure again" during reasoning. Do such statements trigger genuine visual re-examination, or are they merely learned textual patterns? We investigate this via VisualSwap, an image-swap probing framework: after a model reasons over an image, we replace it with a visually similar but semantically different one and test whether the model notices. We introduce VS-Bench, 800 image pairs curated from MathVista, MathVerse, MathVision, and MMMU-Pro. Experiments on Qwen3-VL, Kimi-VL, and ERNIE-VL reveal a striking failure: models overwhelmingly miss the swap, with accuracy dropping by up to 60%. Counterintuitively, thinking models are nearly 3x more vulnerable than their instructed counterparts, and scaling offers no mitigation. Multi-turn user instructions restore visual grounding, but self-generated reflective statements during continuous generation do not. Attention analysis explains why: user instructions substantially elevate attention to visual tokens, whereas self-reflection does not. Current VLMs tend to say rather than actually see when claiming to perform visual re-examination. Our code and dataset are available at the project page: this https URL
>
---
#### [replaced 067] Xiaomi Auto World Model: A Joint World Model Integrating Reconstruction and Generation for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.18137](https://arxiv.org/pdf/2605.18137)**

> **作者:** Lijun Zhou; Hongcheng Luo; Zhenxin Zhu; Cheng Chi; Mingfei Tu; Kaixin Xiong; Lei Gong; Zhanqian Wu; Zehan Zhang; Fangzhen Li; Hao Li; Yingying Shen; Jiale He; Haohui Zhu; Shan Zhao; Kai Wang; Zhiwei Zhan; Yuechuan Pu; Kaiyuan Tan; Ruiling Yang; Xianqi Wang; Tianyi Yan; Jiawei Zhou; Lei Zhang; Jingyang Zhao; Xi Zhou; Chitian Sun; Chenming Wu; Jiong Deng; Hongwei Xie; Ming Lu; Kun Ma; Long Chen; Guang Chen; Hangjun Ye; Bing Wang; Haiyang Sun
>
> **摘要:** This report presents a unified technical system addressing the two core capabilities of world models for autonomous driving: world representation and world generation. For world representation, we propose WorldRec, a feed-forward reconstruction architecture driven by sparse scene queries. WorldRec initializes structured queries in 3D space, leveraging them to aggregate cross-view, cross-temporal features, thereby naturally enforcing spatial consistency across frames and yielding compact yet high-fidelity 3D Gaussian scene representations. For world generation, we propose WorldGen, a two-stage training framework of bidirectional pretraining followed by causal fine-tuning through three progressive stages (Teacher Forcing, ODE distillation, and DMD), enabling high-quality online causal video generation in as few as 4 denoising steps. Building on both modules, we further introduce the JWM, which deeply integrates WorldRec and WorldGen to achieve synergistic gains in generation stability, cross-frame consistency, and visual fidelity, providing a solid foundation for closed-loop simulation, data synthesis, and end-to-end training in autonomous driving.
>
---
#### [replaced 068] Object-Centric Vision Token Pruning for Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.20439](https://arxiv.org/pdf/2511.20439)**

> **作者:** Guangyuan Li; Rongzhen Zhao; Jinhong Deng; Yanbo Wang; Joni Pajarinen
>
> **摘要:** In Vision Language Models (VLMs), vision tokens are quantity-heavy yet information-dispersed compared with language tokens, thus consume too much unnecessary computation. Pruning redundant vision tokens for high VLM inference efficiency has been continuously studied but all existing methods resort to indirect and non-guaranteed ways. We propose OC-VTP, a direct and guaranteed approach to select the most representative vision tokens for high-efficiency yet accuracy-preserving VLM inference. Our OC-VTP requires merely light-weight pre-training of a small object-centric vision token pruner, which can then be inserted into existing VLMs, without fine-tuning of any models on any datasets. It is gauranteed that the most representative vision tokens are kept by minimizing the error in reconstructing the original unpruned tokens from the selected ones. Across any vision pruning ratios, i.e., inference efficiency, our OC-VTP consistently helps mainstream VLMs to preserve the highest inference accuracy. Our pruning also demonstrates interesting interpretability. Our codes are available at this https URL.
>
---
#### [replaced 069] An analytic theory of convolutional neural network inverse problems solvers
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.10334](https://arxiv.org/pdf/2601.10334)**

> **作者:** Minh Hai Nguyen; Quoc Bao Do; Edouard Pauwels; Pierre Weiss
>
> **摘要:** Supervised convolutional neural networks (CNNs) are widely used to solve imaging inverse problems, achieving state-of-the-art performance in numerous applications. However, despite their empirical success, these methods are poorly understood from a theoretical perspective and often treated as black boxes. To bridge this gap, we analyze trained neural networks through the lens of the Minimum Mean Square Error (MMSE) estimator, incorporating functional constraints that capture two fundamental inductive biases of CNNs: translation equivariance and locality via finite receptive fields. Under the empirical training distribution, we derive an analytic, interpretable, and tractable formula for this constrained variant, termed Local-Equivariant MMSE (LE-MMSE). Through extensive numerical experiments across various inverse problems (denoising, inpainting, deconvolution), datasets (FFHQ, CIFAR-10, FashionMNIST), and architectures (U-Net, ResNet, PatchMLP), we demonstrate that our theory matches the neural networks outputs (PSNR $\gtrsim25$dB). Furthermore, we provide insights into the differences between \emph{physics-aware} and \emph{physics-agnostic} estimators, the impact of high-density regions in the training (patch) distribution, and the influence of other factors (dataset size, patch size, etc).
>
---
#### [replaced 070] RASR: Retrieval-Augmented Super Resolution for Practical Reference-based Image Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09449](https://arxiv.org/pdf/2508.09449)**

> **作者:** Jiaqi Yan; Shuning Xu; Xiangyu Chen; Dell Zhang; Jiantao Zhou; Jie Tang; Gangshan Wu; Jie Liu
>
> **备注:** Accepted at ISCAS 2026
>
> **摘要:** Reference-based Super Resolution (RefSR) improves upon Single Image Super Resolution (SISR) by leveraging high-quality reference images to enhance texture fidelity and visual realism. However, a critical limitation of existing RefSR approaches is their reliance on manually curated target-reference image pairs, which severely constrains their practicality in real-world scenarios. To overcome this, we introduce Retrieval-Augmented Super Resolution (RASR), a new and practical RefSR paradigm that automatically retrieves semantically relevant high-resolution images from a reference database given only a low-quality input. This enables scalable and flexible RefSR in realistic use cases, such as enhancing mobile photos taken in environments like zoos or museums, where category-specific reference data (e.g., animals, artworks) can be readily collected or pre-curated. To facilitate research in this direction, we construct RASR-Flickr30, the first benchmark dataset designed for RASR. Unlike prior datasets with fixed target-reference pairs, RASR-Flickr30 provides per-category reference databases to support open-world retrieval. We further propose RASRNet, a strong baseline that combines a semantic reference retriever with a diffusion-based RefSR generator. It retrieves relevant references based on semantic similarity and employs a diffusion-based generator enhanced with semantic conditioning. Experiments on RASR-Flickr30 demonstrate that RASRNet consistently improves over SISR baselines, achieving +0.38 dB PSNR and -0.0131 LPIPS, while generating more realistic textures. These findings highlight retrieval augmentation as a promising direction to bridge the gap between academic RefSR research and real-world applicability.
>
---
#### [replaced 071] Explaining Digital Pathology Models via Clustering Activations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14558](https://arxiv.org/pdf/2511.14558)**

> **作者:** Adam Bajger; Jan Obdržálek; Vojtěch Kůr; Rudolf Nenutil; Petr Holub; Vít Musil; Tomáš Brázdil
>
> **摘要:** We present a clustering-based explainability technique for digital pathology models based on convolutional neural networks. Unlike commonly used methods based on saliency maps, such as occlusion, GradCAM, or relevance propagation, which highlight regions that contribute the most to the prediction for a single slide, our method shows the global behaviour of the model under consideration, while also providing more fine-grained information. The result clusters can be visualised not only to understand the model, but also to increase confidence in its operation, leading to faster adoption in clinical practice. We also evaluate the performance of our technique on an existing model for detecting prostate cancer, demonstrating its usefulness.
>
---
#### [replaced 072] COTTA: Context-Aware Transfer Adaptation for Trajectory Prediction in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.00402](https://arxiv.org/pdf/2604.00402)**

> **作者:** Seohyoung Park; Jaeyeol Lim; Seoyoung Ju; Kyeonghun Kim; Nam-Joon Kim; Hyuk-Jae Lee
>
> **备注:** 4 pages, 2 figures. Accepted at ICEIC 2026
>
> **摘要:** Developing robust models to accurately predict the trajectories of surrounding agents is fundamental to autonomous driving safety. However, most public datasets, such as the Waymo Open Motion Dataset and Argoverse, are collected in Western road environments and do not reflect the unique traffic patterns, infrastructure, and driving behaviors of other regions, including South Korea. This domain discrepancy leads to performance degradation when state-of-the-art models trained on Western data are deployed in different geographic contexts. In this work, we investigate the adaptability of Query-Centric Trajectory Prediction (QCNet) when transferred from U.S.-based data to Korean road environments. Using a Korean autonomous driving dataset, we compare four training strategies: zero-shot transfer, training from scratch, full fine-tuning, and encoder freezing. Experimental results demonstrate that leveraging pretrained knowledge significantly improves prediction performance. Specifically, selectively fine-tuning the decoder while freezing the encoder yields the best trade-off between accuracy and training efficiency, reducing prediction error by over 66% compared to training from scratch. This study provides practical insights into effective transfer learning strategies for deploying trajectory prediction models in new geographic domains.
>
---
#### [replaced 073] Feedforward 3D Editing Learns from Semantic-Part Transformation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.27351](https://arxiv.org/pdf/2605.27351)**

> **作者:** Jiawei Weng; Saining Zhang; Zhenxin Diao; Peishuo Li; Henghaofan Zhang; Junhao Chen; Hao Zhao
>
> **备注:** 31 pages, 22 figures. Project Page: this https URL
>
> **摘要:** 3D editing is a fundamental capability for scalable 3D content creation. While image editing has rapidly evolved toward large-scale feedforward generative paradigms, 3D AI generation remains dominated by training-free editing pipelines. A central challenge of feedforward 3D editing lies in the lack of high-quality paired supervision. Editable 3D assets require simultaneous preservation of geometry, multi-view consistency, structural coherence, and localized edit controllability. Existing 3D editing datasets often rely on independently generated assets, image-mediated reconstruction or narrow edit taxonomies, leading to inaccurate localization, weak preservation, blurred edit boundaries, and limited semantic consistency. In this work, we introduce a new perspective: scalable feedforward 3D editing should be learned from semantic-part transformations. Based on this insight, we propose Pxform, a high-quality 3D editing dataset with over 100K consistent before/after editing pairs across seven edit types. Instead of treating objects as unstructured shapes, our pipeline grounds edits directly in semantic 3D parts. Built upon Pxform, we further propose PartFlow, a feedforward 3D editing network that injects source-aware latent control into pretrained 3D generative priors. PartFlow introduces mask-aware velocity preservation and render-space consistency supervision to jointly improve edit fidelity and source preservation, while requiring no 3D edit mask during inference. Extensive experiments demonstrate that high-quality semantic-part supervision substantially improves scalable 3D editing, enabling PartFlow to achieve state-of-the-art performance on both geometric and appearance editing benchmarks.
>
---
#### [replaced 074] Where Detectors Fail: Probing Generative Space for Generalizable AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.24906](https://arxiv.org/pdf/2605.24906)**

> **作者:** Zijie Cao; Weijie Tu; Yao Xiao; Weijian Deng; Liang Lin; Pengxu Wei
>
> **备注:** Accepted by ICML2026
>
> **摘要:** Detecting AI-generated images (AIGI) remains challenging because detectors often fail to generalize to unseen generators. Although existing methods are trained on large datasets, their performance still degrades when generation settings change, indicating that data scale alone is insufficient and that limited coverage of generative variations during training is a key factor. Studies on generative model editing show that small changes in internal representations can produce diverse and meaningful image variations, many of which are not explored under standard sampling. Leveraging this insight, we propose PROBE (Probing Robustness via Boundary Exploration), a framework that improves detector generalization by actively exploring challenging regions of the generative process. Instead of treating the generator as a fixed data source, PROBE uses the detector as a critic to steer the generator through manifold-level modifications, producing realistic samples that are difficult to classify. These samples expose failure cases that are uncommon under standard data sampling strategies and are used to refine the detector. Experimental results across multiple benchmarks indicate that PROBE enhances generalization to unseen generators, resulting in more generalizable AIGI detection performance. Code and models are available at this https URL
>
---
#### [replaced 075] Automatic Pruning Discovery for Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15390](https://arxiv.org/pdf/2511.15390)**

> **作者:** Haidong Kang; Lihong Lin; Enneng Yang; Hongning Dai; Hao Wang
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Large language models (LLMs) have achieved remarkable performance on a wide range of tasks, hindering real-world deployment due to their massive size. Existing pruning methods (e.g., Wanda) tailored for LLMs rely heavily on manual design pruning algorithms, thereby leading to huge labor costs and requires expert knowledge. Furthermore, we are the first to identify the serious outlier value issue behind dramatic performance degradation under high pruning ratios that are caused by uniform sparsity, raising an additional concern about how to design adaptive pruning sparsity ideal for LLMs. Can LLMs prune by themselves? In this work, we introduce an affirmative answer by proposing a novel pruning method called AutoPrune, which first overcomes expert knowledge limits by leveraging LLMs to design optimal pruning algorithm for themselves automatically without any expert knowledge. Specifically, to mitigate the black-box nature of LLMs, we propose a Graph-driven Chain-of-Thought (GCoT) to optimize prompts, significantly enhancing the reasoning process in learning the pruning algorithm and enabling us to generate pruning algorithms with superior performance and interpretability in the next generation. Finally, grounded in insights of outlier value issue, we introduce Skew-aware Dynamic Sparsity Allocation (SDSA) to overcome the outlier value issue, mitigating performance degradation under high pruning ratios. We conduct extensive experiments on mainstream LLMs benchmarks, demonstrating the superiority of AutoPrune, which consistently excels state-of-the-art competitors.
>
---
#### [replaced 076] MVI-Bench: A Comprehensive Benchmark for Evaluating Robustness to Misleading Visual Inputs in LVLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14159](https://arxiv.org/pdf/2511.14159)**

> **作者:** Huiyi Chen; Jiawei Peng; Dehai Min; Changchang Sun; Kaijie Chen; Yan Yan; Xu Yang; Lu Cheng
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Evaluating the robustness of Large Vision-Language Models (LVLMs) is essential for their continued development and responsible deployment in real-world applications. However, existing robustness benchmarks typically focus on hallucination or misleading textual inputs, while largely overlooking the equally critical challenge posed by misleading visual inputs in assessing visual understanding. To fill this important gap, we introduce MVI-Bench, the first comprehensive benchmark specially designed for evaluating how Misleading Visual Inputs undermine the robustness of LVLMs. Grounded in fundamental visual primitives, the design of MVI-Bench centers on three hierarchical levels of misleading visual inputs: Visual Concept, Visual Attribute, and Visual Relationship. Using this taxonomy, we curate six representative categories and compile 1,248 expertly annotated VQA instances. To facilitate fine-grained robustness evaluation, we further introduce MVI-Sensitivity, a novel metric that characterizes LVLM robustness at a granular level. Empirical results across 18 state-of-the-art LVLMs uncover pronounced vulnerabilities to misleading visual inputs, and our in-depth analyses on MVI-Bench provide actionable insights that can guide the development of more reliable and robust LVLMs. The benchmark and codebase can be accessed at this https URL.
>
---
#### [replaced 077] XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.22726](https://arxiv.org/pdf/2506.22726)**

> **作者:** Yu Zhang; Xi Zhang; Hualin Zhou; Xinyuan Chen; Shang Gao; Hong Jia; Jianfei Yang; Yuankai Qi; Tao Gu
>
> **备注:** Accepted at ICML2026
>
> **摘要:** Deep learning for human sensing on edge systems presents significant potential for smart applications. However, its training and development are hindered by the limited availability of sensor data and resource constraints of edge systems. While transferring pre-trained models to different sensing applications is promising, existing methods often require extensive sensor data and computational resources, resulting in high costs and limited transferability. In this paper, we propose XTransfer, a first-of-its-kind method enabling modality-agnostic, few-shot model transfer with resource-efficient design. XTransfer flexibly uses pre-trained models and transfers knowledge across different modalities by (i) model repairing that safely mitigates modality shift by adapting pre-trained layers with only few sensor data, and (ii) layer recombining that efficiently searches and recombines layers of interest from source models in a layer-wise manner to restructure models. We benchmark various baselines across diverse human sensing datasets spanning different modalities. The results show that XTransfer achieves state-of-the-art performance while significantly reducing the costs of sensor data collection, model training, and edge deployment.
>
---
#### [replaced 078] Occlusion-Aware Physics-Semantic Keyframe Selection for Robust Video Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.23192](https://arxiv.org/pdf/2605.23192)**

> **作者:** Lin Liu; Zhihan Xiao; Haohang Xu; Rong Cong; Zhibo Zhang; Xiaopeng Zhang; Qi Tian
>
> **摘要:** Video editing has recently achieved remarkable progress with diffusion-based generative models, enabling diverse object-level manipulations from natural language instructions. However, existing methods often struggle under occlusion, viewpoint changes, and fast object motion, where unreliable visual observations lead to inaccurate localization, temporal flickering, and inconsistent edits. In this work, we identify the absence of reliable visual anchors as a fundamental bottleneck in occlusion-robust video editing. To address this issue, we propose an occlusion-aware physics-semantic keyframe selection framework that automatically identifies an optimal anchor frame for downstream editing. Specifically, our method evaluates candidate frames from three complementary perspectives: structural completeness for avoiding truncated observations, cycle-consistent tracking stability for measuring physical reliability, and vision-language-based attribute visibility for ensuring semantic clarity. The selected keyframe is then propagated through bidirectional tracking to generate dense spatiotemporal masks, which are used as auxiliary supervision for a diffusion-based video editing backbone. By transforming occlusion handling from explicit reconstruction into reliable anchor selection, our framework enables precise and temporally consistent editing without requiring manual annotations. Extensive experiments on challenging video editing benchmarks demonstrate the effectiveness and high-quality performance of our method.
>
---
#### [replaced 079] Hierarchical Relation-augmented Representation Generalization for Few-shot Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.10079](https://arxiv.org/pdf/2504.10079)**

> **作者:** Hongyu Qu; Ling Xing; Jiachao Zhang; Rui Yan; Yazhou Yao; Xiangbo Shu
>
> **摘要:** Few-shot action recognition (FSAR) aims to recognize novel action categories with few exemplars. Existing methods typically learn frame-level representations for each video by designing inter-frame temporal modeling strategies or inter-video interaction at the coarse video-level granularity. However, they treat each episode task in isolation and neglect fine-grained temporal relation modeling between videos, thus failing to capture shared fine-grained temporal patterns across videos and reuse temporal knowledge from historical tasks. In light of this, we propose HR2G-shot, a Hierarchical Relation-augmented Representation Generalization framework for FSAR, which unifies three types of relation modeling (inter-frame, inter-video, and inter-task) to learn task-specific temporal patterns from a holistic view. Going beyond conducting inter-frame temporal interactions, we further devise two components to respectively explore inter-video and inter-task relationships: i) Inter-video Semantic Correlation (ISC) performs cross-video frame-level interactions in a fine-grained manner, thereby capturing task-specific query features and enhancing both intra-class consistency and inter-class separability; ii) Inter-task Knowledge Transfer (IKT) retrieves and aggregates relevant temporal knowledge from the bank, which stores diverse temporal patterns from historical episode tasks. Extensive experiments on five benchmarks show that HR2G-shot outperforms current top-leading FSAR methods.
>
---
#### [replaced 080] MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MVP-LAM模型，用于学习具有动作信息的潜在动作表示，解决多视角下潜在动作监督不足的问题，提升动作预测与下游任务性能。**

- **链接: [https://arxiv.org/pdf/2602.03668](https://arxiv.org/pdf/2602.03668)**

> **作者:** Jung Min Lee; Dohyeok Lee; Seokhun Ju; Taehyun Cho; Jin Woo Koo; Li Zhao; Sangwoo Hong; Jungwoo Lee
>
> **摘要:** Latent actions learned from diverse human videos serve as pseudo-labels for vision-language-action (VLA) pretraining, but provide effective supervision only if they remain informative about the underlying ground-truth actions. For effective supervision, latent actions should contain information about the underlying actions even though they are inaccessible. We propose Multi-ViewPoint Latent Action Moel (MVP-LAM), which learns latent actions that are highly informative about ground-truth actions from multi-view videos. MVP-LAM trains latent actions with a cross-viewpoint reconstruction objective, so that a latent action from one view must explain the future in another view, reducing reliance on viewpoint-specific cues. On Bridge V2, MVP-LAM produces more action-centric latent actions, achieving higher mutual information with ground-truth actions and improved action prediction, including under out-of-distribution evaluation. Finally, pretraining VLAs with MVP-LAM latent actions improves downstream manipulation performance on various benchmarks. The code and trained checkpoints are available at this https URL.
>
---
#### [replaced 081] Accelerating Diffusion Sampling via Exploiting Local Transition Coherence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.09675](https://arxiv.org/pdf/2503.09675)**

> **作者:** Shangwen Zhu; Han Zhang; Zhantao Yang; Qianyu Peng; Zhao Pu; Huangji Wang; Fan Cheng
>
> **摘要:** Text-based diffusion models have made significant breakthroughs in generating high-quality images and videos from textual descriptions. However, the lengthy sampling time of the denoising process remains a significant bottleneck in practical applications. Previous methods either ignore the statistical relationships between adjacent steps or rely on attention or feature similarity between them, which often only works with specific network structures. To address this issue, we discover a new statistical relationship in the transition operator between adjacent steps, focusing on the relationship of the outputs from the network. This relationship does not impose any requirements on the network structure. Based on this observation, we propose a novel training-free acceleration method called LTC-Accel, which uses the identified relationship to estimate the current transition operator based on adjacent steps. Due to no specific assumptions regarding the network structure, LTC-Accel is applicable to almost all diffusion-based methods and orthogonal to almost all existing acceleration techniques, making it easy to combine with them. Experimental results demonstrate that LTC-Accel significantly speeds up sampling in text-to-image and text-to-video synthesis while maintaining competitive sample quality. Specifically, LTC-Accel achieves a speedup of 1.67-fold in Stable Diffusion v2 and a speedup of 1.55-fold in video generation models. When combined with distillation models, LTC-Accel achieves a remarkable 10-fold speedup in video generation, allowing real-time generation of more than 16FPS.
>
---
#### [replaced 082] Neural Image Space Tessellation efect
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23754](https://arxiv.org/pdf/2602.23754)**

> **作者:** Youyang Du; Junqiu Zhu; Zheng Zeng; Lu Wang; Lingqi Yan
>
> **摘要:** We present Neural Image Space Tessellation effect (NIST), a lightweight screen-space post-processing approach for reducing the faceted silhouettes of low-poly renderings. Instead of tessellating primitives, creating new geometry, or modifying the underlying mesh, NIST uses the low-poly rendering result together with simple auxiliary G-buffer attributes to learn geometry-guided smoothing of object contours in image space. At its core, NIST first deforms image-space contours implicitly and then learns to reassign appearance in the whole image-space, including the deformed regions, preserving texture continuity and avoiding seam artifacts. Experiments show that NIST reduces visually apparent geometric faceting and produces smooth, coherent silhouettes close to tessellation-based smoothing references, with a nearly constant per-frame cost in our tested settings. To the best of our knowledge, NIST is the first work to move the solution of low-poly silhouette faceting from the pre-rendering geometry stage to a post-rendering screen-space stage.
>
---
#### [replaced 083] RelaxFlow: Text-Driven Amodal 3D Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.05425](https://arxiv.org/pdf/2603.05425)**

> **作者:** Jiayin Zhu; Guoji Fu; Xiaolu Liu; Qiyuan He; Yicong Li; Angela Yao
>
> **备注:** Accepted as a spotlight presentation at ICML 2026. Code: this https URL
>
> **摘要:** Image-to-3D generation faces inherent semantic ambiguity under occlusion, where partial observation alone is often insufficient to determine object category. In this work, we formalize text-driven amodal 3D generation, where text prompts steer the completion of unseen regions while strictly preserving input observation. Crucially, we identify that these objectives demand distinct control granularities: rigid control for the observation versus relaxed structural control for the prompt. To this end, we propose RelaxFlow, a training-free dual-branch framework that decouples control granularity via a Multi-Prior Consensus Module and a Relaxation Mechanism. Theoretically, we prove that our relaxation is equivalent to applying a low-pass filter on the generative vector field, which suppresses high-frequency instance details to isolate geometric structure that accommodates the observation. To facilitate evaluation, we introduce two diagnostic benchmarks, ExtremeOcc-3D and AmbiSem-3D. Extensive experiments demonstrate that RelaxFlow successfully steers the generation of unseen regions to match the prompt intent without compromising visual fidelity.
>
---
#### [replaced 084] UDM-GRPO: Stable and Efficient Group Relative Policy Optimization for Uniform Discrete Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.18518](https://arxiv.org/pdf/2604.18518)**

> **作者:** Jiaqi Wang; Haoge Deng; Ting Pan; Yang Liu; Chengyuan Wang; Fan Zhang; Yonggang Qi; Xinlong Wang
>
> **备注:** UDM-GRPO is accepted by ICML 2026 (Spotlight). Code is available at this https URL
>
> **摘要:** Uniform Discrete Diffusion Model (UDM) has recently emerged as a promising paradigm for discrete generative modeling; however, its integration with reinforcement learning remains largely unexplored. We observe that naively applying GRPO to UDM leads to training instability and marginal performance gains. To address this, we propose UDM-GRPO, the first framework to integrate UDM with RL. Our method is guided by two key insights: (i) treating the final clean sample as the action provides more accurate and stable optimization signals; and (ii) reconstructing trajectories via the diffusion forward process better aligns probability paths with the pretraining distribution. Additionally, we introduce two strategies, Reduced-Step and CFG-Free, to further improve training efficiency. UDM-GRPO significantly improves base model performance across multiple T2I tasks. Notably, GenEval accuracy improves from $69\%$ to $96\%$ and PickScore increases from $20.46$ to $23.81$, achieving state-of-the-art performance in both continuous and discrete settings. On the OCR benchmark, accuracy rises from $8\%$ to $57\%$, further validating the generalization ability of our method. Code is available at this https URL.
>
---
#### [replaced 085] MAVEN A Multi-Agent Framework for Multicultural Text-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.16716](https://arxiv.org/pdf/2605.16716)**

> **作者:** Shuowei Li; Yuming Zhao; Parth Bhalerao; Oana Ignat
>
> **备注:** [14] pages, [6] figures, [11] tables, appendix included. Preprint
>
> **摘要:** Text-to-video (T2V) generation has rapidly progressed in visual fidelity, yet its ability to faithfully represent multiple cultures within a single prompt remains underexplored. We introduce MAVEN, a multi-agent prompt refinement framework designed to improve cultural fidelity in both mono-cultural and cross-cultural T2V generation. MAVEN decomposes prompts into person, action, and location dimensions, handled by specialized agents operating in parallel or sequentially. To support systematic evaluation, we contribute a new benchmark of 243 culturally grounded prompts and 972 corresponding videos, spanning three cultures (Chinese, American, Romanian), three action categories, and both mono-cultural and cross-cultural scenarios. Evaluations combining CLIP-based metrics, VLM-as-judge assessments, and videoquality measures show that multi-agent refinement, particularly parallel specialization, significantly improves cultural relevance while preserving visual quality and temporal consistency. The dataset and code are available athttps://github.com/AIM-SCU/CRAFT
>
---
#### [replaced 086] ObjFiller3D: Scaling 3D Object Inpainting to Dense Multi-View Consistency
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.18271](https://arxiv.org/pdf/2508.18271)**

> **作者:** Haitang Feng; Xinkai Chen; Jie Liu; Jie Tang; Gangshan Wu; Beiqi Chen; Jianhuang Lai; Guangcong Wang
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** 3D object inpainting is commonly achieved via multi-view 2D image completion, yet independently inpainted views often suffer from cross-view inconsistencies, leading to blurred textures, geometric discontinuities, and visual artifacts in the reconstructed 3D objects. To overcome these limitations, we propose ObjFiller-3D, a novel method designed for the completion and editing of high-quality and consistent 3D objects. Instead of relying on sparse-view editing or per-view 2D inpainting, our method jointly optimizes a sequence of densely sampled views along a $360^\circ$ trajectory, enabling global coherence across viewpoints. We design a new framework with three complementary components: a Temporal-Driven Generative Encoder for modeling dense-view dependencies, a Semantic-Aware Completion Encoder for object-level inpainting, and a Cycle-Consistent 3D Encoder that enforces global coherence through a closed-loop formulation. Our framework also supports reference-guided 3D inpainting, allowing fine-grained control over appearance. Extensive experiments on diverse datasets demonstrate that ObjFiller-3D significantly outperforms prior methods, achieving higher reconstruction fidelity (PSNR 26.6 vs.\ 15.9 of NeRFiller) and perceptual quality (LPIPS 0.19 vs.\ 0.25 of Instant3dit), while reducing reconstruction time from over 40 minutes to under 10 minutes. These results highlight the effectiveness and practical potential of our approach for real-world 3D editing applications. Project page: this https URL Code: this https URL .
>
---
#### [replaced 087] One-Step Generative Modeling via Wasserstein Gradient Flows
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2605.11755](https://arxiv.org/pdf/2605.11755)**

> **作者:** Jiaqi Han; Puheng Li; Qiushan Guo; Renyuan Xu; Stefano Ermon; Emmanuel J. Candès
>
> **备注:** 40 pages, 14 figures
>
> **摘要:** Diffusion models and flow-based methods have shown impressive generative capability, especially for images, but their sampling is expensive because it requires many iterative updates. We introduce W-Flow, a framework for training a generator that transforms samples from a simple reference distribution into samples from a target data distribution in a single step. This is achieved in two steps: we first define an evolution from the reference distribution to the target distribution through a Wasserstein gradient flow that minimizes an energy functional; second, we train a static neural generator to compress this evolution into one-step generation. We instantiate the energy functional with the Sinkhorn divergence, which yields an efficient optimal-transport-based update rule that captures global distributional discrepancy and improves coverage of the target distribution. We further prove that the finite-sample training dynamics converge to the continuous-time distributional dynamics under suitable assumptions. Empirically, W-Flow sets a new state of the art for one-step ImageNet 256$\times$256 generation, achieving 1.29 FID, with improved mode coverage and domain transfer. Compared to multi-step diffusion models with similar FID scores, our method yields approximately 100$\times$ faster sampling. These results show that Wasserstein gradient flows provide a principled and effective foundation for fast and high-fidelity generative modeling.
>
---
#### [replaced 088] FEA-SLT: A Gloss-Free End-to-End Framework for Facial-Expression-Aware Sign Language Translation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于手语翻译任务，解决现有方法忽视面部表情导致语义模糊的问题。提出FEA-SLT框架，融合面部动态与手部动作，提升翻译准确性。**

- **链接: [https://arxiv.org/pdf/2601.03549](https://arxiv.org/pdf/2601.03549)**

> **作者:** Guobin Tu; Di Weng
>
> **摘要:** Sign Language Translation (SLT) is a challenging cross-modal task requiring joint modeling of manual articulations and non-manual signals. Existing gloss-free SLT methods effectively capture gestural dynamics but often underutilize facial expressions, which play crucial grammatical and disambiguating roles. This limitation can cause semantic degradation when distinct concepts share similar manual configurations. To address this issue, we propose FEA-SLT (**F**acial-**E**xpression-**A**ware **S**ign **L**anguage **T**ranslation), a gloss-free end-to-end framework that uses facial dynamics as semantic anchors for resolving manual ambiguity. FEA-SLT employs a domain-transferred facial encoder to extract expression-sensitive representations and integrates them with manual features through a linguistically constrained *Facial-Expression-Aware Fusion* (FEAF) module. FEAF captures reciprocal dependencies between manual and facial channels via bidirectional modulation, enhancing syntactic fidelity. Experiments on PHOENIX14T and CSL-Daily show that FEA-SLT achieves state-of-the-art BLEU performance among gloss-free methods, while targeted analyses confirm improved translation of facial-sensitive utterances. Code is available at [this https URL](this https URL).
>
---
#### [replaced 089] MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2502.17832](https://arxiv.org/pdf/2502.17832)**

> **作者:** Hyeonjeong Ha; Qiusi Zhan; Jeonghwan Kim; Dimitrios Bralios; Saikrishna Sanniboina; Nanyun Peng; Kai-Wei Chang; Daniel Kang; Heng Ji
>
> **备注:** Code is available at this https URL
>
> **摘要:** Retrieval-augmented generation (RAG) has become a common practice in multimodal large language models (MLLM) to enhance factual grounding and reduce hallucination. Yet, its reliance on retrieval exposes MLLMs to knowledge poisoning attacks, in which adversaries deliberately inject malicious multimodal content into external knowledge bases to steer models toward generating incorrect or even harmful responses. We present MM-PoisonRAG, a framework to systematically study the vulnerability of multimodal RAG under knowledge poisoning. Specifically, we design two novel attack strategies: Localized Poisoning Attack (LPA), which implants targeted, query-specific multimodal misinformation to manipulate outputs toward attacker-controlled responses, and Globalized Poisoning Attack (GPA), which uses a single, untargeted adversarial injection to broadly corrupt reasoning and collapse generation quality across all queries. Extensive experiments on diverse tasks, multimodal RAG components, and attacker access levels reveal severe vulnerabilities: LPA achieves up to 56% attack success rate even under restricted access, and transfers effectively across four different retrievers without re-optimizing the adversaries. GPA completely disrupts model generation to 0% accuracy with just one poisoned content. Moreover, both LPA and GPA bypass existing defenses, underscoring the fragility of multimodal RAG and establishing MM-PoisonRAG as a foundation for future research on securing RAG frameworks against multimodal knowledge poisoning.
>
---
#### [replaced 090] Manboformer: Learning Gaussian Representations via Spatial-temporal Attention Mechanism
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.04863](https://arxiv.org/pdf/2503.04863)**

> **作者:** Ziyue Zhao; Qining Qi; Jianfa Ma
>
> **备注:** After careful self-check, we found several unnoticed deficiencies and incomplete discussions in this manuscript. To ensure the rigor and accuracy of academic results, we decide to withdraw this preprint. A refined, complete, and rigorous version will be submitted soon
>
> **摘要:** Compared with voxel-based grid prediction, in the field of 3D semantic occupation prediction for autonomous driving, GaussianFormer proposed using 3D Gaussian to describe scenes with sparse 3D semantic Gaussian based on objects is another scheme with lower memory requirements. Each 3D Gaussian function represents a flexible region of interest and its semantic features, which are iteratively refined by the attention mechanism. In the experiment, it is found that the Gaussian function required by this method is larger than the query resolution of the original dense grid network, resulting in impaired performance. Therefore, we consider optimizing GaussianFormer by using unused temporal information. We learn the Spatial-Temporal Self-attention Mechanism from the previous grid-given occupation network and improve it to GaussianFormer. The experiment was conducted with the NuScenes dataset, and the experiment is currently underway.
>
---
#### [replaced 091] Not All Pixels Are Equal: Pixel-wise Meta-Learning for Medical Segmentation with Noisy Labels
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18894](https://arxiv.org/pdf/2511.18894)**

> **作者:** Chenyu Mu; Guihai Chen; Xun Yang; Erkun Yang; Cheng Deng
>
> **摘要:** Medical image segmentation is crucial for clinical applications, but it is frequently disrupted by noisy annotations and ambiguous anatomical boundaries, limiting its application in real-world scenarios. Existing methods often directly adapt noisy label learning techniques designed for instance classification, overlooking the pixel-wise heterogeneity in medical segmentation with its spatially and anatomically varying difficulties. Consequently, global assumptions or simple confidence metrics fail to address these local variations, leaving boundary ambiguities unresolved. To address this issue, we propose MetaDCSeg, a robust framework that dynamically learns optimal pixel-wise weights to suppress the influence of noisy labels while preserving reliable annotations. By explicitly modeling boundary uncertainty through a Dynamic Center Distance (DCD) mechanism, our approach utilizes weighted feature distances for foreground, background, and boundary centers, directing the model's attention toward hard-to-segment pixels near ambiguous boundaries. This strategy enables more precise handling of structural boundaries, which are often overlooked by existing methods, and significantly enhances segmentation performance. Extensive experiments across four benchmark datasets with varying noise levels demonstrate that MetaDCSeg outperforms existing state-of-the-art methods.
>
---
#### [replaced 092] ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2605.13517](https://arxiv.org/pdf/2605.13517)**

> **作者:** Jaeyung Kim; YoungJoon Yoo
>
> **备注:** To appear in Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Vector Quantized Variational Autoencoder (VQ-VAE) has become a fundamental framework for learning discrete representations in image modeling. However, VQ-VAE models must tokenize entire images using a finite set of codebook vectors, and this capacity limitation restricts their ability to capture rich and diverse representations. In this paper, we propose ArcCosine Additive Margin VQ-VAE (ArcVQ-VAE), a novel vector quantization framework that introduces a spherical angular-margin prior (SAMP) for the codebook of a conventional VQ-VAE. The proposed SAMP consists of Ball-Bounded Norm Regularization, which constrains all codebook vectors within a time-dependent Euclidean ball, and ArcCosine Additive Margin Loss, which encourages greater angular separability among latent vectors. This formulation promotes more discriminative and uniformly dispersed latent representations within the constrained space, thereby improving effective latent-space coverage and leading to improved codebook utilization. Experimental results on standard image reconstruction and generation tasks show that ArcVQ-VAE achieves competitive performance against baseline models in terms of reconstruction accuracy, representation diversity, and sample quality. The code is available at: this https URL
>
---
#### [replaced 093] An Empirical Study on Variance-based MC Dropout Uncertainty-Error Correlation in 2D Brain Tumor Segmentation
- **分类: cs.LG; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2510.15541](https://arxiv.org/pdf/2510.15541)**

> **作者:** Saumya B
>
> **备注:** v2: Updated title and framing to clarify that findings are specific to variance-based uncertainty estimation via MC Dropout, not MC Dropout broadly. Minor textual improvements throughout. Code and results available at this https URL
>
> **摘要:** Accurate brain tumor segmentation from MRI is vital for diagnosis and treatment planning. Although Monte Carlo (MC) Dropout is widely used to estimate model uncertainty, the effectiveness of variance-based uncertainty - computed as pixel-wise variance across stochastic forward passes - in identifying segmentation errors, particularly near tumor boundaries, remains insufficiently studied. This study empirically examines the relationship between variance-based MC Dropout uncertainty and segmentation error in 2D brain tumor MRI segmentation using a U-Net trained under four augmentation settings: none, horizontal flip, rotation, and scaling. Uncertainty was estimated as the pixel-wise variance across 50 stochastic forward passes and correlated with pixel-wise errors using Pearson and Spearman coefficients. Results show weak global correlations (r ~ 0.30-0.38) and negligible boundary correlations (|r| < 0.05). Although differences across augmentations were statistically significant (p < 0.001), they lacked practical relevance. These findings suggest that variance-based MC Dropout uncertainty provides limited cues for global and boundary error localization, and that the choice of uncertainty representation critically affects the utility of MC Dropout in medical image segmentation. Alternative representations such as predictive entropy or mutual information may better capture segmentation errors, particularly at boundaries.
>
---
#### [replaced 094] Anatomy-Slot: Unsupervised Anatomical Factorization for Homologous Bilateral Reasoning in Retinal Diagnosis
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.12929](https://arxiv.org/pdf/2605.12929)**

> **作者:** Yingzhe Ma; Xiao Yang; Yuguo Yin; Zheyu Wang
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Retinal diagnosis is inherently bilateral: clinicians compare homologous structures across eyes (e.g., optic disc asymmetry), yet most deep models operate on monocular representations. We investigate whether explicit structural correspondence improves diagnosis, and propose Anatomy-Slot to operationalize this hypothesis. Anatomy-Slot introduces an unsupervised anatomical bottleneck by decomposing patch tokens into a set of emergent, structurally-coherent slots that correspond to anatomical regions, then aligning these slots across eyes via bidirectional cross-attention. On ODIR-5K with $n=10$ seeds, the method improves AUC by $4.2$ points over a matched ViT-L baseline (95% CIs; Wilcoxon signed-rank test, $W=0$, $p=0.002$). Pairing disruption and stress testing under Gaussian noise provide controlled tests of correspondence dependence and robustness under corruption. We further report quantitative optic disc grounding on REFUGE and cross-attention localization analysis. Beyond the reported gains, these results indicate that object-centric anatomical correspondence offers a principled path toward interpretable diagnostic systems aligned with clinical bilateral comparison.
>
---
#### [replaced 095] MSCGC-KAN: Multi-scale Causal Graph Convolution and Kolmogorov-Arnold Feature Mapping for EEG Emotion Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.26624](https://arxiv.org/pdf/2605.26624)**

> **作者:** Haoliang Gong; Qingshan She; Jiale Xu; Yunyan Gao; Xugang Xi
>
> **摘要:** Electroencephalogram (EEG)-based emotion recognition is an important affective computing task, and recent EEG foundation models provide useful generic representations for downstream adaptation. However, under the fine-tuning setting, three limitations remain prominent: insufficient modeling of multi-scale emotional dynamics, inadequate exploitation of inter-channel functional connectivity, and the limited expressive power of simple linear classification heads. To address these issues, this paper proposes a new EEG emotion recognition method, termed MSCGC-KAN, which introduces a structured task head composed of multi-scale causal graph convolution and Kolmogorov--Arnold feature mapping. Built on a pre-trained CBraMod backbone, MSCGC-KAN enhances downstream adaptation by jointly strengthening multi-scale temporal modeling, learnable inter-channel connectivity modeling, and nonlinear discriminative mapping within a compact task-specific head. This design preserves the representation advantage of the foundation model while making the classifier more sensitive to emotion-related spatiotemporal patterns. Extensive experiments are conducted on the public FACED and SEED-VII datasets. The proposed method achieves a balanced accuracy of 60.66\%, a Cohen's Kappa of 0.5525, and a weighted F1-score of 60.40\% on FACED, and obtains 33.27\%, 0.2223, and 33.64\%, respectively, on SEED-VII. Compared with the CBraMod+Linear baseline, the balanced accuracy is improved by 5.91 and 2.03 percentage points on the two datasets, respectively. These results indicate that structured task-head design is an effective way to improve EEG emotion recognition when fine-tuning pre-trained EEG models.
>
---
#### [replaced 096] Next-Scale Autoregressive Models for Text-to-Motion Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.03799](https://arxiv.org/pdf/2604.03799)**

> **作者:** Zhiwei Zheng; Shibo Jin; Lingjie Liu; Mingmin Zhao
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Autoregressive (AR) models offer stable and efficient training, but standard next-token prediction is not well aligned with the temporal structure required for text-conditioned motion generation. We introduce MoScale, a next-scale AR framework that generates motion hierarchically from coarse to fine temporal resolutions. By providing global semantics at the coarsest scale and refining them progressively, MoScale establishes a causal hierarchy better suited for long-range motion structure. To improve robustness under limited text-motion data, we further incorporate cross-scale hierarchical refinement for improving per-scale initial predictions and in-scale temporal refinement for selective bidirectional re-prediction. MoScale achieves SOTA text-to-motion performance with high training efficiency, scales effectively with model size, and generalizes zero-shot to diverse motion generation and editing tasks.
>
---
#### [replaced 097] NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval
- **分类: cs.IR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.12824](https://arxiv.org/pdf/2603.12824)**

> **作者:** Zhuchenyang Liu; Yao Zhang; Yu Xiao
>
> **摘要:** Vision-Language Model (VLM) based retrievers have advanced visual document retrieval (VDR) to impressive quality. They require the same multi-billion parameter encoder for both document indexing and query encoding, incurring high latency and GPU dependence even for plain-text queries. We observe that this design is unnecessarily symmetric: documents are visually complex and demand strong visual understanding, whereas queries are just short text strings. NanoVDR exploits this query--document asymmetry by decoupling the two encoding paths: a frozen 2B VLM teacher indexes documents offline, while a distilled text-only student as small as 69M parameters encodes queries at inference. The key design choice is the distillation objective. Through systematic comparison of six objectives across three backbones and 22 ViDoRe benchmark datasets, we find that pointwise cosine alignment on query text consistently outperforms ranking-based and contrastive alternatives, while requiring only pre-cached teacher query embeddings and no document processing during training. Furthermore, we identify cross-lingual transfer as the primary performance bottleneck, and resolve it cheaply by augmenting training data with machine-translated queries. The resulting NanoVDR-S-Multi (DistilBERT, 69M) retains 95.1\% of teacher quality and outperforms DSE-Qwen2 (2B) on v2 and v3 with 32$\times$ fewer parameters and 50$\times$ lower CPU query latency, at a total training cost under 13 GPU-hours.
>
---
#### [replaced 098] Beyond External Monitors: Enhancing Transparency of Large Language Models for Easier Monitoring
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于模型透明性研究，旨在提升大语言模型的可监控性。针对现有方法无法准确反映模型思维过程的问题，提出TELLME方法，增强模型透明度并检测不当行为。**

- **链接: [https://arxiv.org/pdf/2502.05242](https://arxiv.org/pdf/2502.05242)**

> **作者:** Guanxu Chen; Jing Shao; Tao Luo; Lijie Hu; Qihao Lin; Dongrui Liu
>
> **备注:** 28 pages,8 figures,15 tables
>
> **摘要:** Large language models (LLMs) are becoming increasingly capable, but the mechanisms of their thinking and decision-making processes remain unclear. Chain-of-thoughts (CoTs) have been commonly utilized to externalize LLMs' thinking, but this strategy fails to accurately reflect LLMs' thinking process. Techniques based on LLMs' hidden representations provide an inner perspective to improve the monitorability of their latent thinking. However, previous methods only try to develop external modules instead of making LLMs themselves easier to monitor. In this paper, we propose a novel method, TELLME, improving the transparency of LLMs and helping monitors identify unsuitable and sensitive behaviors. Furthermore, we showcase the effectiveness of TELLME on detoxification tasks, where LLMs achieve consistent improvement among multimodal test sets, distinct architectures, and varying parameter scales. We further analyze TELLME's improvement on LLMs' generalization ability from both optimal transport theory and empirical perspectives.
>
---
#### [replaced 099] Are Large Pre-trained Vision Language Models Effective Construction Safety Inspectors?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11011](https://arxiv.org/pdf/2508.11011)**

> **作者:** Xuezheng Chen; Zhengbo Zou
>
> **摘要:** Construction safety inspections typically involve a human inspector identifying safety concerns on-site. With the rise of powerful Vision Language Models (VLMs), researchers are exploring their use for tasks such as detecting safety rule violations from on-site images. However, there is a lack of open datasets to comprehensively evaluate and further fine-tune VLMs in construction safety inspection. Current applications of VLMs use small, supervised datasets, limiting their applicability in tasks they are not directly trained for. In this paper, we propose the ConstructionSite 10k, featuring 10,000 construction site images with annotations for three inter-connected tasks, including image captioning, safety rule violation visual question answering (VQA), and construction element visual grounding. Our subsequent evaluation of current state-of-the-art large pre-trained VLMs shows notable generalization abilities in zero-shot and few-shot settings, while additional training is needed to make them applicable to actual construction sites. This dataset allows researchers to train and evaluate their own VLMs with new architectures and techniques, providing a valuable benchmark for construction safety inspection.
>
---
#### [replaced 100] Enhancing Trustworthy GUI Grounding via Self-Critiqued Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.27266](https://arxiv.org/pdf/2510.27266)**

> **作者:** Shaojie Zhang; Pei Fu; Ruoceng Zhang; Jiahui Yang; Anan Du; Xiuwen Xi; Shaokang Wang; Ying Huang; Bin Qin; Zhenbo Luo; Jian Luan
>
> **摘要:** Autonomous graphical user interface (GUI) agents rely on accurate GUI grounding, which maps language instructions to on-screen coordinates, to execute user commands. However, current models, whether trained via supervised fine-tuning (SFT) or reinforcement learning (RL), often provide confidence signals that are poorly aligned with actual grounding correctness, leading to overconfident and unreliable predictions. To address this, we propose HyperClick, a novel framework that enhances trustworthy GUI grounding through self-critiqued reinforcement learning (SCRL). HyperClick combines a correctness reward and a confidence alignment reward, training the policy model to output both a click prediction and an explicit confidence estimate. This approach jointly optimizes grounding accuracy and confidence reliability through confidence-based self-assessment. Extensive experiments on challenging benchmarks show that HyperClick maintains strong grounding performance while providing better-aligned confidence estimates. By exposing uncertainty alongside GUI actions, HyperClick supports confidence-based abstention in GUI automation. Code will be released here.
>
---
#### [replaced 101] Guaranteed Optimal Compositional Explanations for Neurons
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.20934](https://arxiv.org/pdf/2511.20934)**

> **作者:** Biagio La Rosa; Leilani H. Gilpin
>
> **备注:** Accepted at ICML 2026 (Oral), 43 pages, 10 figures
>
> **摘要:** Compositional explanations are a family of methods that aim to describe the spatial alignment between neurons' receptive field activations and concepts through logical rules, typically computed via a search over all possible concept combinations. Since computing the spatial alignment over the entire state space is computationally infeasible, the literature commonly adopts assumptions related to the structure of the combinations and beam search to restrict the state space. However, beam search cannot provide any theoretical guarantees of optimality, and it remains unclear how close current explanations are to the true optimum. In this theoretical paper, we address this gap by introducing the first framework for computing guaranteed optimal compositional explanations over the entire state space spanned by the adopted assumptions. Specifically, we propose: (i) a decomposition that identifies the factors influencing the spatial alignment, (ii) a heuristic to estimate the alignment at any stage of the search, and (iii) the first algorithm that can compute optimal compositional explanations in a time comparable to exhaustive beam search. Using this framework, we demonstrate that 10-40% of explanations previously obtained with beam search are suboptimal when overlapping concepts are involved. Finally, we evaluate a beam-search variant guided by our proposed decomposition and heuristic, showing that it matches or improves runtime over prior methods while offering greater flexibility in hyperparameters and computational resources.
>
---
#### [replaced 102] The Point, the Vision and the Text: Does Point Cloud Boost Spatial Reasoning of Large Language Models? A Bias-Controlled Study
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.04540](https://arxiv.org/pdf/2504.04540)**

> **作者:** Weichen Zhang; Ruiying Peng; Xin Zeng; Jianjie Fang; Ziyou Wang; Kaiyuan Li; Heng Dong; Wei Li; Chen Gao; Xin Wang; Xinlei Chen; Yong Li
>
> **摘要:** 3D Large Language Models (LLMs) leveraging spatial information in point clouds for 3D spatial reasoning attract great attention. Despite some promising results, the advantages of point clouds over other modalities remain unclear. Moreover, existing 3D benchmarks are insufficient for fairly evaluating the ability of multimodal LLMs to comprehend spatial concepts. To address these challenges, we introduce ScanReQA, a 3D spatial reasoning benchmark encompassing text, vision, and point cloud modalities. We then evaluate the performance of text, 2D, and 3D LLMs on the benchmark to compare the effectiveness of different modalities in understanding spatial concepts. Furthermore, we analyze the reasoning mechanisms behind 3D LLMs using point clouds. Our findings reveal that: 1) binary spatial reasoning remains challenging for current 3D LLMs, 2) MLLMs based on point cloud and visual modalities demonstrate stronger spatial reasoning capabilities than LLMs, and 3) 3D LLMs exhibit the attention sink phenomenon similar to that in 2D LLMs, impairing spatial reasoning. We think these conclusions can help the next step of 3D LLMs and also offer insights for foundation models in other modalities. We release datasets and codes in the project page: this https URL.
>
---
#### [replaced 103] On the Intrinsic Limits of Transformer Image Embeddings in Non-Solvable Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.CC**

- **链接: [https://arxiv.org/pdf/2601.03048](https://arxiv.org/pdf/2601.03048)**

> **作者:** Siyi Lyu; Quan Liu; Feng Yan
>
> **摘要:** Vision Transformers (ViTs) excel in semantic recognition but exhibit systematic failures in spatial reasoning tasks such as mental rotation. While often attributed to data scale, this work argues that the limitation arises from the intrinsic circuit complexity of the architecture. By formalizing spatial understanding as learning a Group Homomorphism Problem -- where latent embeddings preserve the algebraic structure of physical transformations acting on images -- we identify a fundamental computational bottleneck. Specifically, for non-solvable groups (e.g., $\mathrm{SO}(3)$), maintaining such structure-preserving embeddings is lowerbounded by the Word Problem, which is $\mathsf{NC^1}$-complete. In contrast, constant-depth ViTs with polynomial precision are strictly bounded by the complexity class $\mathsf{TC^0}$. Under the standard conjecture $\mathsf{TC^0} \subsetneq \mathsf{NC^1}$, a complexity boundary emerges: constant-depth architectures lack the logical depth required to capture non-solvable spatial structures in a single forward pass. To empirically validate this theoretical gap, we propose the Latent Space Algebra (LSA) benchmark, which reveals a significant degradation in ViT representations as the compositional depth of non-solvable tasks increases.
>
---
#### [replaced 104] VEOcc: Voxel-Centric Online Semantic Occupancy Prediction For Embodied Scene Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.25059](https://arxiv.org/pdf/2605.25059)**

> **作者:** Ruoyu Wang; Yong Liu; Sheng Tao; Yuhang Lin; Yukai Ma
>
> **摘要:** Crucial for autonomous exploration, online 3D occupancy prediction and mapping incrementally constructs dense spatial representations on the fly. However, recent Gaussian-centric methods struggle with structural boundary fidelity and rely heavily on predefined scene-size priors, fundamentally limiting their operational efficiency. In this work, we present VEOcc, a voxel-centric framework formulated as a recursive perception-and-assimilation paradigm. By eliminating the need for initial scale estimation, VEOcc enables highly streamlined, open-ended map expansion. Furthermore, to robustly aggregate noisy temporal observations within the discrete voxel space, we propose a Spatio-Temporal-Aware Online Update Strategy. It integrates Cross-Temporal Logit Aggregation (TLA) for temporal consistency, Reliability-Aware Confidence Modulation (RCM) for spatial uncertainty calibration, and Confidence-Driven Incremental State Update (CSU) for robust global state assimilation. % Extensive experiments on Occ-ScanNet and EmbodiedOcc-ScanNet demonstrate that VEOcc establishes new state-of-the-art performance in both local and embodied settings, providing an accurate and efficient solution for real-world exploration. Extensive experiments on Occ-ScanNet and EmbodiedOcc-ScanNet demonstrate that VEOcc establishes new state-of-the-art performance in both local and embodied settings. Notably, zero-shot evaluations on self-collected video sequences further confirm its robust out-of-distribution generalization capability in completely unseen real-world environments. Ultimately, our framework provides an accurate and highly efficient solution for autonomous exploration. Code and supplementary visualizations are available on our project page: this https URL.
>
---
#### [replaced 105] Unified Panoramic Geometry Estimation via Multi-View Foundation Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2605.26368](https://arxiv.org/pdf/2605.26368)**

> **作者:** Vukasin Bozic; Isidora Slavkovic; Dominik Narnhofer; Nando Metzger; Denis Rozumny; Konrad Schindler; Nikolai Kalischek
>
> **摘要:** Geometry estimation from perspective images has greatly advanced, maturing to the point where off-the-shelf foundation models are able to reconstruct 3D scene structure not only from multi-view imagery, but even from a single view. A natural extension is 3D reconstruction from panoramas, with the exciting prospect of recovering a full 360-degree scene from a single panoramic image. In this work, we introduce PaGeR (Panoramic Geometry Reconstruction), a framework to lift powerful 3D foundation models designed for perspective imagery to the panorama domain. Our strategy is to start from a pre-trained transformer for 3D reconstruction and turn it into a unified high-performance model that predicts scale-invariant depth, metric depth, surface normals, and sky masks from both perspective and omnidirectional images, in a single forward pass. By keeping architectural changes to a minimum and mixing perspective and panoramic images during training, PaGeR retains the rich 3D prior of the underlying foundation model while learning to also estimate geometrically consistent 360-degree scenes from single panoramas. We extensively test our method in both indoor and outdoor environments and find that it delivers state-of-the-art performance and excellent zero-shot performance across a wide range of scenes. Code, data and models are available $\href{this https URL}{\text{here}}$.
>
---
#### [replaced 106] OmniEgo-R$^2$: A Routed Reasoning Framework for the 1st Cross-Domain EgoCross Challenge at CVPR 2026
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.24481](https://arxiv.org/pdf/2605.24481)**

> **作者:** Zixu Li; Zhiwei Chen; Zhiheng Fu; Wenbo Wang; Yupeng Hu; Weili Guan; Liqiang Nie
>
> **备注:** Technical Report for the 1st Cross-Domain EgoCross Challenge at CVPR 2026
>
> **摘要:** The 1st Cross-Domain EgoCross Challenge at EgoVis, CVPR 2026 evaluates whether multimodal large language models can reason over egocentric videos across surgery, industry, extreme sports, and animal perspective. We achieved second place in both the Source-Limited and Open-Source tracks. In this report, we formulate EgoCross as a robust cross-domain embodied video reasoning problem rather than a simple multiple-choice visual question answering task. We identify three key challenges: (C1) temporal boundary ambiguity, where critical state transitions are sparsely sampled and often occur between frames; (C2) cross-domain semantic granularity mismatch, where the same capability requires different domain-specific visual grammar; and (C3) decision instability under close options, where long multimodal reasoning can select unsupported distractors or produce malformed outputs. To address them, we propose OmniEgo-R$^2$ (Omnidomain Egocentric Routed Reasoning), a unified routed reasoning pipeline consisting of temporal-evidence normalization, domain-agnostic capability routing, structured perception--dynamics--decision reasoning, boundary-aware option verification, and defensive answer calibration. OmniEgo-R$^2$ uses the Qwen3-VL-4B-SFT checkpoints on each EgoCross domain as the visual-language backbone, and wraps them with lightweight test-time reasoning and parsing programs. Our final submissions obtain 66.35% overall accuracy in the Source-Limited track and 66.77% in the Open-Source track, ranking second in both leaderboards. The codes are available on this https URL
>
---
#### [replaced 107] LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.20497](https://arxiv.org/pdf/2602.20497)**

> **作者:** Peiliang Cai; Jiacheng Liu; Haowen Xu; Xinyu Wang; Chang Zou; Linfeng Zhang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Diffusion models have achieved remarkable success in image and video generation tasks. However, the high computational demands of Diffusion Transformers (DiTs) pose a significant challenge to their practical deployment. While feature caching is a promising acceleration strategy, existing methods based on simple reusing or training-free forecasting struggle to adapt to the complex, stage-dependent dynamics of the diffusion process, often resulting in quality degradation and failing to maintain consistency with the standard denoising process. To address this, we propose a LEarnable Stage-Aware (LESA) predictor framework based on two-stage training. Our approach leverages a Kolmogorov-Arnold Network (KAN) to accurately learn temporal feature mappings from data. We further introduce a multi-stage, multi-expert architecture that assigns specialized predictors to different noise-level stages, enabling more precise and robust feature forecasting. Extensive experiments show our method achieves significant acceleration while maintaining high-fidelity generation. Experiments demonstrate 5.00x acceleration on FLUX.1-dev with minimal quality degradation (1.0% drop), 6.25x speedup on Qwen-Image with a 20.2% quality improvement over the previous SOTA (TaylorSeer), and 5.00x acceleration on HunyuanVideo with a 24.7% PSNR improvement over TaylorSeer. State-of-the-art performance on both text-to-image and text-to-video synthesis validates the effectiveness and generalization capability of our training-based framework across different models. Our code is available at this https URL.
>
---
#### [replaced 108] Bridging the Pose-Semantic Gap: A Cascade Framework for Text-Based Person Anomaly Search
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2604.23282](https://arxiv.org/pdf/2604.23282)**

> **作者:** Zequn Xie; Guijin Luo; Chuxin Wang; Sihang Cai; Tao Jin; Zhou Zhao; Yixuan Tang
>
> **备注:** Accepted to ACL 2026.10 pages, 5 figures
>
> **摘要:** Text-based person anomaly search retrieves specific behavioral events from surveillance archives using natural-language queries. Although recent pose-aware methods align geometric structures well, they face a fundamental Pose-Semantic Gap: semantically different actions can share similar skeletal geometries. While Multimodal Large Language Models (MLLMs) can reduce this ambiguity, using them for large-scale retrieval is computationally prohibitive. We propose the Structure-Semantic Decoupled Cascade (SSDC) framework, which decouples retrieval into two stages: (1) Structure-Aware Coarse Retrieval, where a lightweight model quickly filters candidates by skeletal similarity ; and (2) Detective Squad Interaction, a multi-agent semantic verification module. The squad consists of a Detective for fast binary filtering, an Analyst for evidence extraction, and a Writer for semantic synthesis. Finally, we re-rank candidates by fusing the synthesized captions with structural priors. Experiments on the PAB benchmark show that SSDC achieves state-of-the-art performance by balancing efficiency and semantic reasoning.
>
---
#### [replaced 109] Semantic-Enriched Latent Visual Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2605.19342](https://arxiv.org/pdf/2605.19342)**

> **作者:** Tianrun Xu; Yue Sun; Qixun Wang; Jingyi Lu; Yuan Wang; Tianren Zhang; Longteng Guo; Fengyun Rao; Jing Lyu; Feng Chen; Jing Liu
>
> **摘要:** Multimodal latent-space reasoning aims to replace explicit thinking with images by performing visual reasoning directly in a compact latent space. However, existing approaches largely rely on visual supervision and produce latent representations that lack sufficient semantic richness, limiting their ability to support diverse region-level reasoning tasks. In this work, we introduce Semantic-Enriched Latent Visual Reasoning (SLVR), a two-stage learning framework that enriches latent representations with attribute-level visual semantics and aligns them with diverse reasoning objectives. In the first stage, SLVR learns semantically enriched region-centric latents under fine-grained attribute supervision. In the second stage, we design Multi-query Group Relative Policy Optimization (M-GRPO) to align latent representations across multiple queries grounded in the same region. To support this framework, we construct SLV-Set, comprising approximately 400K region-level attribute annotations and 800K multi-query question answering samples, and introduce SV-QA, a benchmark that evaluates latent reasoning under semantic variation. Experiments demonstrate that SLVR improves the robustness and semantic consistency of latent visual reasoning compared to existing baselines.
>
---
