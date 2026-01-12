# 计算机视觉 cs.CV

- **最新发布 62 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] STResNet & STYOLO : A New Family of Compact Classification and Object Detection Models for MCUs
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出STResNet和STYOLO，解决嵌入式设备上模型精度与效率的平衡问题，优化图像分类和目标检测模型。**

- **链接: [https://arxiv.org/pdf/2601.05364v1](https://arxiv.org/pdf/2601.05364v1)**

> **作者:** Sudhakar Sah; Ravish Kumar
>
> **备注:** 9 pages, 1 figure
>
> **摘要:** Recent advancements in lightweight neural networks have significantly improved the efficiency of deploying deep learning models on edge hardware. However, most existing architectures still trade accuracy for latency, which limits their applicability on microcontroller and neural processing unit based devices. In this work, we introduce two new model families, STResNet for image classification and STYOLO for object detection, jointly optimized for accuracy, efficiency, and memory footprint on resource constrained platforms. The proposed STResNet series, ranging from Nano to Tiny variants, achieves competitive ImageNet 1K accuracy within a four million parameter budget. Specifically, STResNetMilli attains 70.0 percent Top 1 accuracy with only three million parameters, outperforming MobileNetV1 and ShuffleNetV2 at comparable computational complexity. For object detection, STYOLOMicro and STYOLOMilli achieve 30.5 percent and 33.6 percent mean average precision, respectively, on the MS COCO dataset, surpassing YOLOv5n and YOLOX Nano in both accuracy and efficiency. Furthermore, when STResNetMilli is used as a backbone with the Ultralytics training environment.
>
---
#### [new 002] Sketch&Patch++: Efficient Structure-Aware 3D Gaussian Representation
- **分类: cs.CV; cs.GR; cs.MM; eess.IV**

- **简介: 该论文提出Sketch&Patch++，用于3D高斯表示的结构感知优化，解决高效存储与渲染问题。通过分类高斯为结构和区域特征，提升压缩效率与视觉质量。**

- **链接: [https://arxiv.org/pdf/2601.05394v1](https://arxiv.org/pdf/2601.05394v1)**

> **作者:** Yuang Shi; Simone Gasparini; Géraldine Morin; Wei Tsang Ooi
>
> **摘要:** We observe that Gaussians exhibit distinct roles and characteristics analogous to traditional artistic techniques -- like how artists first sketch outlines before filling in broader areas with color, some Gaussians capture high-frequency features such as edges and contours, while others represent broader, smoother regions analogous to brush strokes that add volume and depth. Based on this observation, we propose a hybrid representation that categorizes Gaussians into (i) Sketch Gaussians, which represent high-frequency, boundary-defining features, and (ii) Patch Gaussians, which cover low-frequency, smooth regions. This semantic separation naturally enables layered progressive streaming, where the compact Sketch Gaussians establish the structural skeleton before Patch Gaussians incrementally refine volumetric detail. In this work, we extend our previous method to arbitrary 3D scenes by proposing a novel hierarchical adaptive categorization framework that operates directly on the 3DGS representation. Our approach employs multi-criteria density-based clustering, combined with adaptive quality-driven refinement. This method eliminates dependency on external 3D line primitives while ensuring optimal parametric encoding effectiveness. Our comprehensive evaluation across diverse scenes, including both man-made and natural environments, demonstrates that our method achieves up to 1.74 dB improvement in PSNR, 6.7% in SSIM, and 41.4% in LPIPS at equivalent model sizes compared to uniform pruning baselines. For indoor scenes, our method can maintain visual quality with only 0.5\% of the original model size. This structure-aware representation enables efficient storage, adaptive streaming, and rendering of high-fidelity 3D content across bandwidth-constrained networks and resource-limited devices.
>
---
#### [new 003] Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geolocalization
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像地理定位任务，旨在提升模型通过视觉线索预测图像拍摄位置的能力。提出一种结合地图的强化代理方法，通过两阶段优化提高定位精度。**

- **链接: [https://arxiv.org/pdf/2601.05432v1](https://arxiv.org/pdf/2601.05432v1)**

> **作者:** Yuxiang Ji; Yong Wang; Ziyu Ma; Yiming Hu; Hailang Huang; Xuecai Hu; Guanhua Chen; Liaoni Wu; Xiangxiang Chu
>
> **摘要:** The image geolocalization task aims to predict the location where an image was taken anywhere on Earth using visual clues. Existing large vision-language model (LVLM) approaches leverage world knowledge, chain-of-thought reasoning, and agentic capabilities, but overlook a common strategy used by humans -- using maps. In this work, we first equip the model \textit{Thinking with Map} ability and formulate it as an agent-in-the-map loop. We develop a two-stage optimization scheme for it, including agentic reinforcement learning (RL) followed by parallel test-time scaling (TTS). The RL strengthens the agentic capability of model to improve sampling efficiency, and the parallel TTS enables the model to explore multiple candidate paths before making the final prediction, which is crucial for geolocalization. To evaluate our method on up-to-date and in-the-wild images, we further present MAPBench, a comprehensive geolocalization training and evaluation benchmark composed entirely of real-world images. Experimental results show that our method outperforms existing open- and closed-source models on most metrics, specifically improving Acc@500m from 8.0\% to 22.1\% compared to \textit{Gemini-3-Pro} with Google Search/Map grounded mode.
>
---
#### [new 004] What's Left Unsaid? Detecting and Correcting Misleading Omissions in Multimodal News Previews
- **分类: cs.CV; cs.SI**

- **简介: 该论文属于多模态内容检测任务，旨在解决新闻预览中因遗漏关键信息导致的误导问题。通过构建基准和提出OMGuard模型，提升误导性检测与修正效果。**

- **链接: [https://arxiv.org/pdf/2601.05563v1](https://arxiv.org/pdf/2601.05563v1)**

> **作者:** Fanxiao Li; Jiaying Wu; Tingchao Fu; Dayang Li; Herun Wan; Wei Zhou; Min-Yen Kan
>
> **摘要:** Even when factually correct, social-media news previews (image-headline pairs) can induce interpretation drift: by selectively omitting crucial context, they lead readers to form judgments that diverge from what the full article conveys. This covert harm is harder to detect than explicit misinformation yet remains underexplored. To address this gap, we develop a multi-stage pipeline that disentangles and simulates preview-based versus context-based understanding, enabling construction of the MM-Misleading benchmark. Using this benchmark, we systematically evaluate open-source LVLMs and uncover pronounced blind spots to omission-based misleadingness detection. We further propose OMGuard, which integrates (1) Interpretation-Aware Fine-Tuning, which used to improve multimodal misleadingness detection and (2) Rationale-Guided Misleading Content Correction, which uses explicit rationales to guide headline rewriting and reduce misleading impressions. Experiments show that OMGuard lifts an 8B model's detection accuracy to match a 235B LVLM and delivers markedly stronger end-to-end correction. Further analysis reveals that misleadingness typically stems from local narrative shifts (e.g., missing background) rather than global frame changes, and identifies image-driven scenarios where text-only correction fails, highlighting the necessity of visual interventions.
>
---
#### [new 005] Ensemble of radiomics and ConvNeXt for breast cancer diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于乳腺癌诊断任务，旨在提升早期检测效果。通过结合影像组学与深度学习，采用集成方法提高诊断准确性。**

- **链接: [https://arxiv.org/pdf/2601.05373v1](https://arxiv.org/pdf/2601.05373v1)**

> **作者:** Jorge Alberto Garza-Abdala; Gerardo Alejandro Fumagal-González; Beatriz A. Bosques-Palomo; Mario Alexis Monsivais Molina; Daly Avedano; Servando Cardona-Huerta; José Gerardo Tamez-Pena
>
> **备注:** Accepted and presented at the IEEE International Symposium on Computer-Based Medical Systems (CBMS) 2025
>
> **摘要:** Early diagnosis of breast cancer is crucial for improving survival rates. Radiomics and deep learning (DL) have shown significant potential in assisting radiologists with early cancer detection. This paper aims to critically assess the performance of radiomics, DL, and ensemble techniques in detecting cancer from screening mammograms. Two independent datasets were used: the RSNA 2023 Breast Cancer Detection Challenge (11,913 patients) and a Mexican cohort from the TecSalud dataset (19,400 patients). The ConvNeXtV1-small DL model was trained on the RSNA dataset and validated on the TecSalud dataset, while radiomics models were developed using the TecSalud dataset and validated with a leave-one-year-out approach. The ensemble method consistently combined and calibrated predictions using the same methodology. Results showed that the ensemble approach achieved the highest area under the curve (AUC) of 0.87, compared to 0.83 for ConvNeXtV1-small and 0.80 for radiomics. In conclusion, ensemble methods combining DL and radiomics predictions significantly enhance breast cancer diagnosis from mammograms.
>
---
#### [new 006] MoGen: A Unified Collaborative Framework for Controllable Multi-Object Image Generation
- **分类: cs.CV**

- **简介: 该论文属于多物体图像生成任务，旨在解决语义与图像区域对齐不精准、对象数量不一致的问题。提出MoGen框架，通过RSA和AMG模块实现精确控制与灵活生成。**

- **链接: [https://arxiv.org/pdf/2601.05546v1](https://arxiv.org/pdf/2601.05546v1)**

> **作者:** Yanfeng Li; Yue Sun; Keren Fu; Sio-Kei Im; Xiaoming Liu; Guangtao Zhai; Xiaohong Liu; Tao Tan
>
> **摘要:** Existing multi-object image generation methods face difficulties in achieving precise alignment between localized image generation regions and their corresponding semantics based on language descriptions, frequently resulting in inconsistent object quantities and attribute aliasing. To mitigate this limitation, mainstream approaches typically rely on external control signals to explicitly constrain the spatial layout, local semantic and visual attributes of images. However, this strong dependency makes the input format rigid, rendering it incompatible with the heterogeneous resource conditions of users and diverse constraint requirements. To address these challenges, we propose MoGen, a user-friendly multi-object image generation method. First, we design a Regional Semantic Anchor (RSA) module that precisely anchors phrase units in language descriptions to their corresponding image regions during the generation process, enabling text-to-image generation that follows quantity specifications for multiple objects. Building upon this foundation, we further introduce an Adaptive Multi-modal Guidance (AMG) module, which adaptively parses and integrates various combinations of multi-source control signals to formulate corresponding structured intent. This intent subsequently guides selective constraints on scene layouts and object attributes, achieving dynamic fine-grained control. Experimental results demonstrate that MoGen significantly outperforms existing methods in generation quality, quantity consistency, and fine-grained control, while exhibiting superior accessibility and control flexibility. Code is available at: https://github.com/Tear-kitty/MoGen/tree/master.
>
---
#### [new 007] Multi-task Cross-modal Learning for Chest X-ray Image Retrieval
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文属于医学图像检索任务，旨在提升胸部X光图像与文本的跨模态匹配效果。通过多任务学习优化BiomedCLIP模型，增强其在临床场景中的表现。**

- **链接: [https://arxiv.org/pdf/2601.05399v1](https://arxiv.org/pdf/2601.05399v1)**

> **作者:** Zhaohui Liang; Sivaramakrishnan Rajaraman; Niccolo Marini; Zhiyun Xue; Sameer Antani
>
> **摘要:** CLIP and BiomedCLIP are examples of vision-language foundation models and offer strong cross-modal embeddings; however, they are not optimized for fine-grained medical retrieval tasks, such as retrieving clinically relevant radiology reports using chest X-ray (CXR) image queries. To address this shortcoming, we propose a multi-task learning framework to fine-tune BiomedCLIP and evaluate improvements to CXR image-text retrieval. Using BiomedCLIP as the backbone, we incorporate a lightweight MLP projector head trained with a multi-task composite loss function that includes: (1) a binary cross-entropy loss to distinguish normal from abnormal CXR studies, (2) a supervised contrastive loss to reinforce intra-class consistency, and (3) a CLIP loss to maintain cross-modal alignment. Experimental results demonstrate that the fine-tuned model achieves more balanced and clinically meaningful performance across both image-to-text and text-to-image retrieval tasks compared to the pretrained BiomedCLIP and general-purpose CLIP models. Furthermore, t-SNE visualizations reveal clearer semantic clustering of normal and abnormal cases, demonstrating the model's enhanced diagnostic sensitivity. These findings highlight the value of domain-adaptive, multi-task learning for advancing cross-modal retrieval in biomedical applications.
>
---
#### [new 008] MMViR: A Multi-Modal and Multi-Granularity Representation for Long-range Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MMViR，解决长视频理解任务中的多模态表示问题，通过多粒度结构提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2601.05495v1](https://arxiv.org/pdf/2601.05495v1)**

> **作者:** Zizhong Li; Haopeng Zhang; Jiawei Zhang
>
> **备注:** 13 pages, 11 figures
>
> **摘要:** Long videos, ranging from minutes to hours, present significant challenges for current Multi-modal Large Language Models (MLLMs) due to their complex events, diverse scenes, and long-range dependencies. Direct encoding of such videos is computationally too expensive, while simple video-to-text conversion often results in redundant or fragmented content. To address these limitations, we introduce MMViR, a novel multi-modal, multi-grained structured representation for long video understanding. MMViR identifies key turning points to segment the video and constructs a three-level description that couples global narratives with fine-grained visual details. This design supports efficient query-based retrieval and generalizes well across various scenarios. Extensive evaluations across three tasks, including QA, summarization, and retrieval, show that MMViR outperforms the prior strongest method, achieving a 19.67% improvement in hour-long video understanding while reducing processing latency to 45.4% of the original.
>
---
#### [new 009] FlyPose: Towards Robust Human Pose Estimation From Aerial Views
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人体姿态估计任务，解决从无人机视角准确检测和估计人体姿态的问题。通过多数据集训练提升模型性能，并发布新数据集FlyPose-104。**

- **链接: [https://arxiv.org/pdf/2601.05747v1](https://arxiv.org/pdf/2601.05747v1)**

> **作者:** Hassaan Farooq; Marvin Brenner; Peter St\ütz
>
> **备注:** 11 pages, 9 figures, IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly deployed in close proximity to humans for applications such as parcel delivery, traffic monitoring, disaster response and infrastructure inspections. Ensuring safe and reliable operation in these human-populated environments demands accurate perception of human poses and actions from an aerial viewpoint. This perspective challenges existing methods with low resolution, steep viewing angles and (self-)occlusion, especially if the application demands realtime feasibile models. We train and deploy FlyPose, a lightweight top-down human pose estimation pipeline for aerial imagery. Through multi-dataset training, we achieve an average improvement of 6.8 mAP in person detection across the test-sets of Manipal-UAV, VisDrone, HIT-UAV as well as our custom dataset. For 2D human pose estimation we report an improvement of 16.3 mAP on the challenging UAV-Human dataset. FlyPose runs with an inference latency of ~20 milliseconds including preprocessing on a Jetson Orin AGX Developer Kit and is deployed onboard a quadrotor UAV during flight experiments. We also publish FlyPose-104, a small but challenging aerial human pose estimation dataset, that includes manual annotations from difficult aerial perspectives: https://github.com/farooqhassaan/FlyPose.
>
---
#### [new 010] TAGRPO: Boosting GRPO on Image-to-Video Generation with Direct Trajectory Alignment
- **分类: cs.CV**

- **简介: 该论文属于图像到视频生成任务，解决GRPO在I2V中效果不佳的问题。提出TAGRPO框架，通过直接轨迹对齐提升奖励效果。**

- **链接: [https://arxiv.org/pdf/2601.05729v1](https://arxiv.org/pdf/2601.05729v1)**

> **作者:** Jin Wang; Jianxiang Lu; Guangzheng Xu; Comi Chen; Haoyu Yang; Linqing Wang; Peng Chen; Mingtao Chen; Zhichao Hu; Longhuang Wu; Shuai Shao; Qinglin Lu; Ping Luo
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Recent studies have demonstrated the efficacy of integrating Group Relative Policy Optimization (GRPO) into flow matching models, particularly for text-to-image and text-to-video generation. However, we find that directly applying these techniques to image-to-video (I2V) models often fails to yield consistent reward improvements. To address this limitation, we present TAGRPO, a robust post-training framework for I2V models inspired by contrastive learning. Our approach is grounded in the observation that rollout videos generated from identical initial noise provide superior guidance for optimization. Leveraging this insight, we propose a novel GRPO loss applied to intermediate latents, encouraging direct alignment with high-reward trajectories while maximizing distance from low-reward counterparts. Furthermore, we introduce a memory bank for rollout videos to enhance diversity and reduce computational overhead. Despite its simplicity, TAGRPO achieves significant improvements over DanceGRPO in I2V generation.
>
---
#### [new 011] SketchVL: Policy Optimization via Fine-Grained Credit Assignment for Chart Understanding and More
- **分类: cs.CV**

- **简介: 该论文提出SketchVL模型，解决图表理解中因信用分配不精准导致的强化学习难题。通过FinePO算法实现细粒度奖励分配，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2601.05688v1](https://arxiv.org/pdf/2601.05688v1)**

> **作者:** Muye Huang; Lingling Zhang; Yifei Li; Yaqiang Wu; Jun Liu
>
> **摘要:** Charts are high-density visual carriers of complex data and medium for information extraction and analysis. Due to the need for precise and complex visual reasoning, automated chart understanding poses a significant challenge to existing Multimodal Large Language Models (MLLMs). Many MLLMs trained with reinforcement learning (RL) face the challenge of credit assignment. Their advantage estimation, typically performed at the trajectory level, cannot distinguish between correct and incorrect reasoning steps within a single generated response. To address this limitation, we introduce SketchVL, a novel MLLM that optimized with FinePO, a new RL algorithm designed for fine-grained credit assignment within each trajectory. SketchVL's methodology involves drawing its intermediate reasoning steps as markers on the image and feeding the annotated image back to itself, creating a robust, multi-step reasoning process. During training, the FinePO algorithm leverages a Fine-grained Process Reward Model (FinePRM) to score each drawing action within a trajectory, thereby precisely assigning credit for each step. This mechanism allows FinePO to more strongly reward correct tokens when a trajectory is globally successful, and more heavily penalize incorrect tokens when the trajectory is globally suboptimal, thus achieving fine-grained reinforcement signals. Experiments show that SketchVL learns to align its step-level behavior with the FinePRM, achieving an average performance gain of 7.23\% over its base model across chart datasets, natural image datasets, and mathematics, providing a promising new direction for training powerful reasoning models.
>
---
#### [new 012] Adapting Vision Transformers to Ultra-High Resolution Semantic Segmentation with Relay Tokens
- **分类: cs.CV**

- **简介: 该论文属于图像语义分割任务，解决超高清图像分割中丢失全局上下文或细节的问题。通过引入可学习的中继令牌，在局部与全局尺度间传递特征，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.05927v1](https://arxiv.org/pdf/2601.05927v1)**

> **作者:** Yohann Perron; Vladyslav Sydorov; Christophe Pottier; Loic Landrieu
>
> **备注:** 13 pages +3 pages of suppmat
>
> **摘要:** Current approaches for segmenting ultra high resolution images either slide a window, thereby discarding global context, or downsample and lose fine detail. We propose a simple yet effective method that brings explicit multi scale reasoning to vision transformers, simultaneously preserving local details and global awareness. Concretely, we process each image in parallel at a local scale (high resolution, small crops) and a global scale (low resolution, large crops), and aggregate and propagate features between the two branches with a small set of learnable relay tokens. The design plugs directly into standard transformer backbones (eg ViT and Swin) and adds fewer than 2 % parameters. Extensive experiments on three ultra high resolution segmentation benchmarks, Archaeoscape, URUR, and Gleason, and on the conventional Cityscapes dataset show consistent gains, with up to 15 % relative mIoU improvement. Code and pretrained models are available at https://archaeoscape.ai/work/relay-tokens/ .
>
---
#### [new 013] Deepfake detectors are DUMB: A benchmark to assess adversarial training robustness under transferability constraints
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于深度伪造检测任务，研究对抗训练在真实场景下的鲁棒性。通过实验分析不同检测器和攻击方法，在跨数据集情况下的性能变化，揭示对抗训练的局限性。**

- **链接: [https://arxiv.org/pdf/2601.05986v1](https://arxiv.org/pdf/2601.05986v1)**

> **作者:** Adrian Serrano; Erwan Umlil; Ronan Thomas
>
> **备注:** 10 pages, four tables, one figure
>
> **摘要:** Deepfake detection systems deployed in real-world environments are subject to adversaries capable of crafting imperceptible perturbations that degrade model performance. While adversarial training is a widely adopted defense, its effectiveness under realistic conditions -- where attackers operate with limited knowledge and mismatched data distributions - remains underexplored. In this work, we extend the DUMB -- Dataset soUrces, Model architecture and Balance - and DUMBer methodology to deepfake detection. We evaluate detectors robustness against adversarial attacks under transferability constraints and cross-dataset configuration to extract real-world insights. Our study spans five state-of-the-art detectors (RECCE, SRM, XCeption, UCF, SPSL), three attacks (PGD, FGSM, FPBA), and two datasets (FaceForensics++ and Celeb-DF-V2). We analyze both attacker and defender perspectives mapping results to mismatch scenarios. Experiments show that adversarial training strategies reinforce robustness in the in-distribution cases but can also degrade it under cross-dataset configuration depending on the strategy adopted. These findings highlight the need for case-aware defense strategies in real-world applications exposed to adversarial attacks.
>
---
#### [new 014] Kidney Cancer Detection Using 3D-Based Latent Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决3D肾脏异常检测问题。通过结合扩散模型和生成对抗网络，实现弱监督下的异常检测，提升标注效率。**

- **链接: [https://arxiv.org/pdf/2601.05852v1](https://arxiv.org/pdf/2601.05852v1)**

> **作者:** Jen Dusseljee; Sarah de Boer; Alessa Hering
>
> **备注:** 8 pages, 2 figures. This paper has been accepted at Bildverarbeitung für die Medizin (BVM) 2026
>
> **摘要:** In this work, we present a novel latent diffusion-based pipeline for 3D kidney anomaly detection on contrast-enhanced abdominal CT. The method combines Denoising Diffusion Probabilistic Models (DDPMs), Denoising Diffusion Implicit Models (DDIMs), and Vector-Quantized Generative Adversarial Networks (VQ-GANs). Unlike prior slice-wise approaches, our method operates directly on an image volume and leverages weak supervision with only case-level pseudo-labels. We benchmark our approach against state-of-the-art supervised segmentation and detection models. This study demonstrates the feasibility and promise of 3D latent diffusion for weakly supervised anomaly detection. While the current results do not yet match supervised baselines, they reveal key directions for improving reconstruction fidelity and lesion localization. Our findings provide an important step toward annotation-efficient, generative modeling of complex abdominal anatomy.
>
---
#### [new 015] SceneFoundry: Generating Interactive Infinite 3D Worlds
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出SceneFoundry，用于生成可交互的大型3D场景，解决真实环境生成难题。属于环境生成任务，旨在为机器人学习提供物理真实的虚拟环境。**

- **链接: [https://arxiv.org/pdf/2601.05810v1](https://arxiv.org/pdf/2601.05810v1)**

> **作者:** ChunTeng Chen; YiChen Hsu; YiWen Liu; WeiFang Sun; TsaiChing Ni; ChunYi Lee; Min Sun; YuanFu Yang
>
> **备注:** 15 pages
>
> **摘要:** The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research.
>
---
#### [new 016] MOSAIC-GS: Monocular Scene Reconstruction via Advanced Initialization for Complex Dynamic Environments
- **分类: cs.CV**

- **简介: 该论文属于动态场景重建任务，解决单目视频中几何与时间一致性难题。通过多几何线索和轨迹建模，实现高效且高质量的重建。**

- **链接: [https://arxiv.org/pdf/2601.05368v1](https://arxiv.org/pdf/2601.05368v1)**

> **作者:** Svitlana Morkva; Maximum Wilder-Smith; Michael Oechsle; Alessio Tonioni; Marco Hutter; Vaishakh Patil
>
> **摘要:** We present MOSAIC-GS, a novel, fully explicit, and computationally efficient approach for high-fidelity dynamic scene reconstruction from monocular videos using Gaussian Splatting. Monocular reconstruction is inherently ill-posed due to the lack of sufficient multiview constraints, making accurate recovery of object geometry and temporal coherence particularly challenging. To address this, we leverage multiple geometric cues, such as depth, optical flow, dynamic object segmentation, and point tracking. Combined with rigidity-based motion constraints, these cues allow us to estimate preliminary 3D scene dynamics during an initialization stage. Recovering scene dynamics prior to the photometric optimization reduces reliance on motion inference from visual appearance alone, which is often ambiguous in monocular settings. To enable compact representations, fast training, and real-time rendering while supporting non-rigid deformations, the scene is decomposed into static and dynamic components. Each Gaussian in the dynamic part of the scene is assigned a trajectory represented as time-dependent Poly-Fourier curve for parameter-efficient motion encoding. We demonstrate that MOSAIC-GS achieves substantially faster optimization and rendering compared to existing methods, while maintaining reconstruction quality on par with state-of-the-art approaches across standard monocular dynamic scene benchmarks.
>
---
#### [new 017] Coding the Visual World: From Image to Simulation Using Vision Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉理解任务，旨在探究视觉语言模型能否从图像生成模拟代码。研究通过Im2Sim方法测试模型对复杂系统的建模能力，发现其具备高层次理解但细节处理有限。**

- **链接: [https://arxiv.org/pdf/2601.05344v1](https://arxiv.org/pdf/2601.05344v1)**

> **作者:** Sagi Eppel
>
> **摘要:** The ability to construct mental models of the world is a central aspect of understanding. Similarly, visual understanding can be viewed as the ability to construct a representative model of the system depicted in an image. This work explores the capacity of Vision Language Models (VLMs) to recognize and simulate the systems and mechanisms depicted in images using the Im2Sim methodology. The VLM is given a natural image of a real-world system (e.g., cities, clouds, vegetation) and is tasked with describing the system and writing code that simulates and generates it. This generative code is then executed to produce a synthetic image, which is compared against the original. This approach is tested on various complex emergent systems, ranging from physical systems (waves, lights, clouds) to vegetation, cities, materials, and geological formations. Through analysis of the models and images generated by the VLMs, we examine their understanding of the systems in images. The results show that leading VLMs (GPT, Gemini) demonstrate the capacity to understand and model complex, multi-component systems across multiple layers of abstraction and a wide range of domains. At the same time, the VLMs exhibit limited ability to replicate fine details and low-level arrangements of patterns in the image. These findings reveal an interesting asymmetry: VLMs combine high-level, deep visual understanding of images with limited perception of fine details.
>
---
#### [new 018] GaussianSwap: Animatable Video Face Swapping with 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出GaussianSwap，属于视频人脸交换任务，解决传统方法生成的面部无法动画化的问题。通过3D高斯点云构建可操控的面部角色，实现高质量、连贯的面部交换。**

- **链接: [https://arxiv.org/pdf/2601.05511v1](https://arxiv.org/pdf/2601.05511v1)**

> **作者:** Xuan Cheng; Jiahao Rao; Chengyang Li; Wenhao Wang; Weilin Chen; Lvqing Yang
>
> **摘要:** We introduce GaussianSwap, a novel video face swapping framework that constructs a 3D Gaussian Splatting based face avatar from a target video while transferring identity from a source image to the avatar. Conventional video swapping frameworks are limited to generating facial representations in pixel-based formats. The resulting swapped faces exist merely as a set of unstructured pixels without any capacity for animation or interactive manipulation. Our work introduces a paradigm shift from conventional pixel-based video generation to the creation of high-fidelity avatar with swapped faces. The framework first preprocesses target video to extract FLAME parameters, camera poses and segmentation masks, and then rigs 3D Gaussian splats to the FLAME model across frames, enabling dynamic facial control. To ensure identity preserving, we propose an compound identity embedding constructed from three state-of-the-art face recognition models for avatar finetuning. Finally, we render the face-swapped avatar on the background frames to obtain the face-swapped video. Experimental results demonstrate that GaussianSwap achieves superior identity preservation, visual clarity and temporal consistency, while enabling previously unattainable interactive applications.
>
---
#### [new 019] Enabling Stroke-Level Structural Analysis of Hieroglyphic Scripts without Language-Specific Priors
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于字符结构分析任务，旨在解决LLMs对象形文字结构感知不足的问题。提出HieroSA框架，自动提取字符笔画结构，无需语言先验知识。**

- **链接: [https://arxiv.org/pdf/2601.05508v1](https://arxiv.org/pdf/2601.05508v1)**

> **作者:** Fuwen Luo; Zihao Wan; Ziyue Wang; Yaluo Liu; Pau Tong Lin Xu; Xuanjia Qiao; Xiaolong Wang; Peng Li; Yang Liu
>
> **摘要:** Hieroglyphs, as logographic writing systems, encode rich semantic and cultural information within their internal structural composition. Yet, current advanced Large Language Models (LLMs) and Multimodal LLMs (MLLMs) usually remain structurally blind to this information. LLMs process characters as textual tokens, while MLLMs additionally view them as raw pixel grids. Both fall short to model the underlying logic of character strokes. Furthermore, existing structural analysis methods are often script-specific and labor-intensive. In this paper, we propose Hieroglyphic Stroke Analyzer (HieroSA), a novel and generalizable framework that enables MLLMs to automatically derive stroke-level structures from character bitmaps without handcrafted data. It transforms modern logographic and ancient hieroglyphs character images into explicit, interpretable line-segment representations in a normalized coordinate space, allowing for cross-lingual generalization. Extensive experiments demonstrate that HieroSA effectively captures character-internal structures and semantics, bypassing the need for language-specific priors. Experimental results highlight the potential of our work as a graphematics analysis tool for a deeper understanding of hieroglyphic scripts. View our code at https://github.com/THUNLP-MT/HieroSA.
>
---
#### [new 020] SAS-VPReID: A Scale-Adaptive Framework with Shape Priors for Video-based Person Re-Identification at Extreme Far Distances
- **分类: cs.CV**

- **简介: 该论文属于视频行人重识别任务，旨在解决极端远距离下因分辨率低、视角变化大和噪声干扰导致的识别难题。提出SAS-VPReID框架，包含三个模块提升特征表达与时空建模能力。**

- **链接: [https://arxiv.org/pdf/2601.05535v1](https://arxiv.org/pdf/2601.05535v1)**

> **作者:** Qiwei Yang; Pingping Zhang; Yuhao Wang; Zijing Gong
>
> **备注:** Accepted by WACV2026 VReID-XFD Workshop. Our final framework ranks the first on the VReID-XFD challenge leaderboard
>
> **摘要:** Video-based Person Re-IDentification (VPReID) aims to retrieve the same person from videos captured by non-overlapping cameras. At extreme far distances, VPReID is highly challenging due to severe resolution degradation, drastic viewpoint variation and inevitable appearance noise. To address these issues, we propose a Scale-Adaptive framework with Shape Priors for VPReID, named SAS-VPReID. The framework is built upon three complementary modules. First, we deploy a Memory-Enhanced Visual Backbone (MEVB) to extract discriminative feature representations, which leverages the CLIP vision encoder and multi-proxy memory. Second, we propose a Multi-Granularity Temporal Modeling (MGTM) to construct sequences at multiple temporal granularities and adaptively emphasize motion cues across scales. Third, we incorporate Prior-Regularized Shape Dynamics (PRSD) to capture body structure dynamics. With these modules, our framework can obtain more discriminative feature representations. Experiments on the VReID-XFD benchmark demonstrate the effectiveness of each module and our final framework ranks the first on the VReID-XFD challenge leaderboard. The source code is available at https://github.com/YangQiWei3/SAS-VPReID.
>
---
#### [new 021] Context-Aware Decoding for Faithful Vision-Language Generation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言生成任务，旨在解决大模型中的幻觉问题。通过分析生成动态，提出一种无需训练的缓解方法，提升生成内容的视觉一致性。**

- **链接: [https://arxiv.org/pdf/2601.05939v1](https://arxiv.org/pdf/2601.05939v1)**

> **作者:** Mehrdad Fazli; Bowen Wei; Ziwei Zhu
>
> **摘要:** Hallucinations, generating responses inconsistent with the visual input, remain a critical limitation of large vision-language models (LVLMs), especially in open-ended tasks such as image captioning and visual reasoning. In this work, we probe the layer-wise generation dynamics that drive hallucinations and propose a training-free mitigation strategy. Employing the Logit Lens, we examine how LVLMs construct next-token distributions across decoder layers, uncovering a pronounced commitment-depth gap: truthful tokens accumulate probability mass on their final candidates earlier than hallucinatory ones. Drawing on this discovery, we introduce Context Embedding Injection (CEI), a lightweight method that harnesses the hidden state of the last input token-the context embedding-as a grounding signal to maintain visual fidelity throughout decoding and curb hallucinations. Evaluated on the CHAIR, AMBER, and MMHal-Bench benchmarks (with a maximum token length of 512), CEI outperforms state-of-the-art baselines across three LVLMs, with its dynamic variant yielding the lowest overall hallucination rates. By integrating novel mechanistic insights with a scalable intervention, this work advances the mitigation of hallucinations in LVLMs.
>
---
#### [new 022] ROAP: A Reading-Order and Attention-Prior Pipeline for Optimizing Layout Transformers in Key Information Extraction
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文档理解任务，旨在解决Layout Transformers在关键信息提取中的阅读顺序缺失和视觉干扰问题。提出ROAP管道，优化注意力分布，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.05470v1](https://arxiv.org/pdf/2601.05470v1)**

> **作者:** Tingwei Xie; Jinxin He; Yonghong Song
>
> **备注:** 10 pages, 4 figures, 4 tables
>
> **摘要:** The efficacy of Multimodal Transformers in visually-rich document understanding (VrDU) is critically constrained by two inherent limitations: the lack of explicit modeling for logical reading order and the interference of visual tokens that dilutes attention on textual semantics. To address these challenges, this paper presents ROAP, a lightweight and architecture-agnostic pipeline designed to optimize attention distributions in Layout Transformers without altering their pre-trained backbones. The proposed pipeline first employs an Adaptive-XY-Gap (AXG-Tree) to robustly extract hierarchical reading sequences from complex layouts. These sequences are then integrated into the attention mechanism via a Reading-Order-Aware Relative Position Bias (RO-RPB). Furthermore, a Textual-Token Sub-block Attention Prior (TT-Prior) is introduced to adaptively suppress visual noise and enhance fine-grained text-text interactions. Extensive experiments on the FUNSD and CORD benchmarks demonstrate that ROAP consistently improves the performance of representative backbones, including LayoutLMv3 and GeoLayoutLM. These findings confirm that explicitly modeling reading logic and regulating modality interference are critical for robust document understanding, offering a scalable solution for complex layout analysis. The implementation code will be released at https://github.com/KevinYuLei/ROAP.
>
---
#### [new 023] Performance of a Deep Learning-Based Segmentation Model for Pancreatic Tumors on Public Endoscopic Ultrasound Datasets
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在提升胰腺肿瘤在EUS图像中的自动分割效果。通过构建基于Vision Transformer的模型，解决人工分割主观性问题，并验证其性能。**

- **链接: [https://arxiv.org/pdf/2601.05937v1](https://arxiv.org/pdf/2601.05937v1)**

> **作者:** Pankaj Gupta; Priya Mudgil; Niharika Dutta; Kartik Bose; Nitish Kumar; Anupam Kumar; Jimil Shah; Vaneet Jearth; Jayanta Samanta; Vishal Sharma; Harshal Mandavdhare; Surinder Rana; Saroj K Sinha; Usha Dutta
>
> **摘要:** Background: Pancreatic cancer is one of the most aggressive cancers, with poor survival rates. Endoscopic ultrasound (EUS) is a key diagnostic modality, but its effectiveness is constrained by operator subjectivity. This study evaluates a Vision Transformer-based deep learning segmentation model for pancreatic tumors. Methods: A segmentation model using the USFM framework with a Vision Transformer backbone was trained and validated with 17,367 EUS images (from two public datasets) in 5-fold cross-validation. The model was tested on an independent dataset of 350 EUS images from another public dataset, manually segmented by radiologists. Preprocessing included grayscale conversion, cropping, and resizing to 512x512 pixels. Metrics included Dice similarity coefficient (DSC), intersection over union (IoU), sensitivity, specificity, and accuracy. Results: In 5-fold cross-validation, the model achieved a mean DSC of 0.651 +/- 0.738, IoU of 0.579 +/- 0.658, sensitivity of 69.8%, specificity of 98.8%, and accuracy of 97.5%. For the external validation set, the model achieved a DSC of 0.657 (95% CI: 0.634-0.769), IoU of 0.614 (95% CI: 0.590-0.689), sensitivity of 71.8%, and specificity of 97.7%. Results were consistent, but 9.7% of cases exhibited erroneous multiple predictions. Conclusions: The Vision Transformer-based model demonstrated strong performance for pancreatic tumor segmentation in EUS images. However, dataset heterogeneity and limited external validation highlight the need for further refinement, standardization, and prospective studies.
>
---
#### [new 024] Boosting Latent Diffusion Models via Disentangled Representation Alignment
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决LDMs与VAEs在表示学习上的不匹配问题。提出Send-VAE，通过语义解耦提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.05823v1](https://arxiv.org/pdf/2601.05823v1)**

> **作者:** John Page; Xuesong Niu; Kai Wu; Kun Gai
>
> **摘要:** Latent Diffusion Models (LDMs) generate high-quality images by operating in a compressed latent space, typically obtained through image tokenizers such as Variational Autoencoders (VAEs). In pursuit of a generation-friendly VAE, recent studies have explored leveraging Vision Foundation Models (VFMs) as representation alignment targets for VAEs, mirroring the approach commonly adopted for LDMs. Although this yields certain performance gains, using the same alignment target for both VAEs and LDMs overlooks their fundamentally different representational requirements. We advocate that while LDMs benefit from latents retaining high-level semantic concepts, VAEs should excel in semantic disentanglement, enabling encoding of attribute-level information in a structured way. To address this, we propose the Semantic disentangled VAE (Send-VAE), explicitly optimized for disentangled representation learning through aligning its latent space with the semantic hierarchy of pre-trained VFMs. Our approach employs a non-linear mapper network to transform VAE latents, aligning them with VFMs to bridge the gap between attribute-level disentanglement and high-level semantics, facilitating effective guidance for VAE learning. We evaluate semantic disentanglement via linear probing on attribute prediction tasks, showing strong correlation with improved generation performance. Finally, using Send-VAE, we train flow-based transformers SiTs; experiments show Send-VAE significantly speeds up training and achieves a state-of-the-art FID of 1.21 and 1.75 with and without classifier-free guidance on ImageNet 256x256.
>
---
#### [new 025] One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视觉异常检测任务，旨在解决开放场景下的零/少样本异常检测问题。提出UniADet框架，通过简化模型结构和解耦任务，实现高效、通用的异常检测。**

- **链接: [https://arxiv.org/pdf/2601.05552v1](https://arxiv.org/pdf/2601.05552v1)**

> **作者:** Bin-Bin Gao; Chengjie Wang
>
> **备注:** 20 pages, 5 figures, 34 tabels
>
> **摘要:** Universal visual anomaly detection (AD) aims to identify anomaly images and segment anomaly regions towards open and dynamic scenarios, following zero- and few-shot paradigms without any dataset-specific fine-tuning. We have witnessed significant progress in widely use of visual-language foundational models in recent approaches. However, current methods often struggle with complex prompt engineering, elaborate adaptation modules, and challenging training strategies, ultimately limiting their flexibility and generality. To address these issues, this paper rethinks the fundamental mechanism behind visual-language models for AD and presents an embarrassingly simple, general, and effective framework for Universal vision Anomaly Detection (UniADet). Specifically, we first find language encoder is used to derive decision weights for anomaly classification and segmentation, and then demonstrate that it is unnecessary for universal AD. Second, we propose an embarrassingly simple method to completely decouple classification and segmentation, and decouple cross-level features, i.e., learning independent weights for different tasks and hierarchical features. UniADet is highly simple (learning only decoupled weights), parameter-efficient (only 0.002M learnable parameters), general (adapting a variety of foundation models), and effective (surpassing state-of-the-art zero-/few-shot by a large margin and even full-shot AD methods for the first time) on 14 real-world AD benchmarks covering both industrial and medical domains. We will make the code and model of UniADet available at https://github.com/gaobb/UniADet.
>
---
#### [new 026] Learning Geometric Invariance for Gait Recognition
- **分类: cs.CV**

- **简介: 该论文属于行为识别任务，旨在解决跨视角和跨服装的步态识别问题。通过引入几何变换建模，提出RRS-Gait框架实现几何不变性，从而提升识别性能。**

- **链接: [https://arxiv.org/pdf/2601.05604v1](https://arxiv.org/pdf/2601.05604v1)**

> **作者:** Zengbin Wang; Junjie Li; Saihui Hou; Xu Liu; Chunshui Cao; Yongzhen Huang; Muyi Sun; Siye Wang; Man Zhang
>
> **摘要:** The goal of gait recognition is to extract identity-invariant features of an individual under various gait conditions, e.g., cross-view and cross-clothing. Most gait models strive to implicitly learn the common traits across different gait conditions in a data-driven manner to pull different gait conditions closer for recognition. However, relatively few studies have explicitly explored the inherent relations between different gait conditions. For this purpose, we attempt to establish connections among different gait conditions and propose a new perspective to achieve gait recognition: variations in different gait conditions can be approximately viewed as a combination of geometric transformations. In this case, all we need is to determine the types of geometric transformations and achieve geometric invariance, then identity invariance naturally follows. As an initial attempt, we explore three common geometric transformations (i.e., Reflect, Rotate, and Scale) and design a $\mathcal{R}$eflect-$\mathcal{R}$otate-$\mathcal{S}$cale invariance learning framework, named ${\mathcal{RRS}}$-Gait. Specifically, it first flexibly adjusts the convolution kernel based on the specific geometric transformations to achieve approximate feature equivariance. Then these three equivariant-aware features are respectively fed into a global pooling operation for final invariance-aware learning. Extensive experiments on four popular gait datasets (Gait3D, GREW, CCPG, SUSTech1K) show superior performance across various gait conditions.
>
---
#### [new 027] VIB-Probe: Detecting and Mitigating Hallucinations in Vision-Language Models via Variational Information Bottleneck
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型的 hallucination 检测与缓解任务，旨在解决生成文本与视觉内容不符的问题。通过 VIB-Probe 框架，利用信息瓶颈理论提取关键特征并干预注意力头以提升生成真实性。**

- **链接: [https://arxiv.org/pdf/2601.05547v1](https://arxiv.org/pdf/2601.05547v1)**

> **作者:** Feiran Zhang; Yixin Wu; Zhenghua Wang; Xiaohua Wang; Changze Lv; Xuanjing Huang; Xiaoqing Zheng
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable progress in multimodal tasks, but remain susceptible to hallucinations, where generated text deviates from the underlying visual content. Existing hallucination detection methods primarily rely on output logits or external verification tools, often overlooking their internal mechanisms. In this work, we investigate the outputs of internal attention heads, postulating that specific heads carry the primary signals for truthful generation.However, directly probing these high-dimensional states is challenging due to the entanglement of visual-linguistic syntax and noise. To address this, we propose VIB-Probe, a novel hallucination detection and mitigation framework leveraging the Variational Information Bottleneck (VIB) theory. Our method extracts discriminative patterns across layers and heads while filtering out semantic nuisances through the information bottleneck principle. Furthermore, by leveraging the gradients of our VIB probe, we identify attention heads with strong causal influence on hallucinations and introduce an inference-time intervention strategy for hallucination mitigation. Extensive experiments across diverse benchmarks demonstrate that VIB-Probe significantly outperforms existing baselines in both settings. Our code will be made publicly available.
>
---
#### [new 028] Multi-Image Super Resolution Framework for Detection and Analysis of Plant Roots
- **分类: cs.CV; cs.ET**

- **简介: 该论文属于图像超分辨率任务，旨在解决地下植物根系成像质量差的问题。通过多视角图像和深度学习方法提升根系细节与清晰度，实现更准确的根部性状分析。**

- **链接: [https://arxiv.org/pdf/2601.05482v1](https://arxiv.org/pdf/2601.05482v1)**

> **作者:** Shubham Agarwal; Ofek Nourian; Michael Sidorov; Sharon Chemweno; Ofer Hadar; Naftali Lazarovitch; Jhonathan E. Ephrath
>
> **摘要:** Understanding plant root systems is critical for advancing research in soil-plant interactions, nutrient uptake, and overall plant health. However, accurate imaging of roots in subterranean environments remains a persistent challenge due to adverse conditions such as occlusion, varying soil moisture, and inherently low contrast, which limit the effectiveness of conventional vision-based approaches. In this work, we propose a novel underground imaging system that captures multiple overlapping views of plant roots and integrates a deep learning-based Multi-Image Super Resolution (MISR) framework designed to enhance root visibility and detail. To train and evaluate our approach, we construct a synthetic dataset that simulates realistic underground imaging scenarios, incorporating key environmental factors that affect image quality. Our proposed MISR algorithm leverages spatial redundancy across views to reconstruct high-resolution images with improved structural fidelity and visual clarity. Quantitative evaluations show that our approach outperforms state-of-the-art super resolution baselines, achieving a 2.3 percent reduction in BRISQUE, indicating improved image quality with the same CLIP-IQA score, thereby enabling enhanced phenotypic analysis of root systems. This, in turn, facilitates accurate estimation of critical root traits, including root hair count and root hair density. The proposed framework presents a promising direction for robust automatic underground plant root imaging and trait quantification for agricultural and ecological research.
>
---
#### [new 029] Bi-Orthogonal Factor Decomposition for Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉Transformer分析任务，旨在理解注意力机制如何交换信息。通过BFD方法分解位置与内容因素，揭示注意力交互机制。**

- **链接: [https://arxiv.org/pdf/2601.05328v1](https://arxiv.org/pdf/2601.05328v1)**

> **作者:** Fenil R. Doshi; Thomas Fel; Talia Konkle; George Alvarez
>
> **摘要:** Self-attention is the central computational primitive of Vision Transformers, yet we lack a principled understanding of what information attention mechanisms exchange between tokens. Attention maps describe where weight mass concentrates; they do not reveal whether queries and keys trade position, content, or both. We introduce Bi-orthogonal Factor Decomposition (BFD), a two-stage analytical framework: first, an ANOVA-based decomposition statistically disentangles token activations into orthogonal positional and content factors; second, SVD of the query-key interaction matrix QK^T exposes bi-orthogonal modes that reveal how these factors mediate communication. After validating proper isolation of position and content, we apply BFD to state-of-the-art vision models and uncover three phenomena.(i) Attention operates primarily through content. Content-content interactions dominate attention energy, followed by content-position coupling. DINOv2 allocates more energy to content-position than supervised models and distributes computation across a richer mode spectrum. (ii) Attention mechanisms exhibit specialization: heads differentiate into content-content, content-position, and position-position operators, while singular modes within heads show analogous specialization. (iii) DINOv2's superior holistic shape processing emerges from intermediate layers that simultaneously preserve positional structure while contextually enriching semantic content. Overall, BFD exposes how tokens interact through attention and which informational factors - positional or semantic - mediate their communication, yielding practical insights into vision transformer mechanisms.
>
---
#### [new 030] GS-DMSR: Dynamic Sensitive Multi-scale Manifold Enhancement for Accelerated High-Quality 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D动态场景重建任务，旨在解决模型收敛速度与渲染质量的平衡问题。提出GS-DMSR方法，通过自适应优化和多尺度增强提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2601.05584v1](https://arxiv.org/pdf/2601.05584v1)**

> **作者:** Nengbo Lu; Minghua Pan; Shaohua Sun; Yizhou Liang
>
> **摘要:** In the field of 3D dynamic scene reconstruction, how to balance model convergence rate and rendering quality has long been a critical challenge that urgently needs to be addressed, particularly in high-precision modeling of scenes with complex dynamic motions. To tackle this issue, this study proposes the GS-DMSR method. By quantitatively analyzing the dynamic evolution process of Gaussian attributes, this mechanism achieves adaptive gradient focusing, enabling it to dynamically identify significant differences in the motion states of Gaussian models. It then applies differentiated optimization strategies to Gaussian models with varying degrees of significance, thereby significantly improving the model convergence rate. Additionally, this research integrates a multi-scale manifold enhancement module, which leverages the collaborative optimization of an implicit nonlinear decoder and an explicit deformation field to enhance the modeling efficiency for complex deformation scenes. Experimental results demonstrate that this method achieves a frame rate of up to 96 FPS on synthetic datasets, while effectively reducing both storage overhead and training time.Our code and data are available at https://anonymous.4open.science/r/GS-DMSR-2212.
>
---
#### [new 031] EdgeLDR: Quaternion Low-Displacement Rank Neural Networks for Edge-Efficient Deep Learning
- **分类: cs.CV**

- **简介: 该论文属于边缘高效深度学习任务，旨在解决神经网络在边缘设备上的内存和计算效率问题。通过引入Quaternion Low-Displacement Rank结构，实现参数压缩与快速计算。**

- **链接: [https://arxiv.org/pdf/2601.05379v1](https://arxiv.org/pdf/2601.05379v1)**

> **作者:** Vladimir Frants; Sos Agaian; Karen Panetta
>
> **摘要:** Deploying deep neural networks on edge devices is often limited by the memory traffic and compute cost of dense linear operators. While quaternion neural networks improve parameter efficiency by coupling multiple channels through Hamilton products, they typically retain unstructured dense weights; conversely, structured matrices enable fast computation but are usually applied in the real domain. This paper introduces EdgeLDR, a practical framework for quaternion block-circulant linear and convolutional layers that combines quaternion channel mixing with block-circulant parameter structure and enables FFT-based evaluation through the complex adjoint representation. We present reference implementations of EdgeLDR layers and compare FFT-based computation against a naive spatial-domain realization of quaternion circulant products. FFT evaluation yields large empirical speedups over the naive implementation and keeps latency stable as block size increases, making larger compression factors computationally viable. We further integrate EdgeLDR layers into compact CNN and Transformer backbones and evaluate accuracy-compression trade-offs on 32x32 RGB classification (CIFAR-10/100, SVHN) and hyperspectral image classification (Houston 2013, Pavia University), reporting parameter counts and CPU/GPU latency. The results show that EdgeLDR layers provide significant compression with competitive accuracy.
>
---
#### [new 032] LayerGS: Decomposition and Inpainting of Layered 3D Human Avatars via 2D Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文属于3D人体重建任务，旨在解决多层人体模型分解与修复问题。通过2D高斯溅射和扩散模型实现准确的层分解与渲染，提升虚拟试穿效果。**

- **链接: [https://arxiv.org/pdf/2601.05853v1](https://arxiv.org/pdf/2601.05853v1)**

> **作者:** Yinghan Xu; John Dingliana
>
> **摘要:** We propose a novel framework for decomposing arbitrarily posed humans into animatable multi-layered 3D human avatars, separating the body and garments. Conventional single-layer reconstruction methods lock clothing to one identity, while prior multi-layer approaches struggle with occluded regions. We overcome both limitations by encoding each layer as a set of 2D Gaussians for accurate geometry and photorealistic rendering, and inpainting hidden regions with a pretrained 2D diffusion model via score-distillation sampling (SDS). Our three-stage training strategy first reconstructs the coarse canonical garment via single-layer reconstruction, followed by multi-layer training to jointly recover the inner-layer body and outer-layer garment details. Experiments on two 3D human benchmark datasets (4D-Dress, Thuman2.0) show that our approach achieves better rendering quality and layer decomposition and recomposition than the previous state-of-the-art, enabling realistic virtual try-on under novel viewpoints and poses, and advancing practical creation of high-fidelity 3D human assets for immersive applications. Our code is available at https://github.com/RockyXu66/LayerGS
>
---
#### [new 033] FeatureSLAM: Feature-enriched 3D gaussian splatting SLAM in real time
- **分类: cs.CV**

- **简介: 该论文属于实时SLAM任务，解决传统SLAM语义不足的问题。通过融合3D高斯泼溅和特征渲染，提升跟踪与建图精度，实现高效语义映射。**

- **链接: [https://arxiv.org/pdf/2601.05738v1](https://arxiv.org/pdf/2601.05738v1)**

> **作者:** Christopher Thirgood; Oscar Mendez; Erin Ling; Jon Storey; Simon Hadfield
>
> **摘要:** We present a real-time tracking SLAM system that unifies efficient camera tracking with photorealistic feature-enriched mapping using 3D Gaussian Splatting (3DGS). Our main contribution is integrating dense feature rasterization into the novel-view synthesis, aligned with a visual foundation model. This yields strong semantics, going beyond basic RGB-D input, aiding both tracking and mapping accuracy. Unlike previous semantic SLAM approaches (which embed pre-defined class labels) FeatureSLAM enables entirely new downstream tasks via free-viewpoint, open-set segmentation. Across standard benchmarks, our method achieves real-time tracking, on par with state-of-the-art systems while improving tracking stability and map fidelity without prohibitive compute. Quantitatively, we obtain 9\% lower pose error and 8\% higher mapping accuracy compared to recent fixed-set SLAM baselines. Our results confirm that real-time feature-embedded SLAM, is not only valuable for enabling new downstream applications. It also improves the performance of the underlying tracking and mapping subsystems, providing semantic and language masking results that are on-par with offline 3DGS models, alongside state-of-the-art tracking, depth and RGB rendering.
>
---
#### [new 034] Compressing image encoders via latent distillation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像压缩任务，旨在解决深度学习模型在硬件受限环境中的部署问题。通过简化知识蒸馏策略，压缩编码器，提升轻量化效果。**

- **链接: [https://arxiv.org/pdf/2601.05639v1](https://arxiv.org/pdf/2601.05639v1)**

> **作者:** Caroline Mazini Rodrigues; Nicolas Keriven; Thomas Maugey
>
> **摘要:** Deep learning models for image compression often face practical limitations in hardware-constrained applications. Although these models achieve high-quality reconstructions, they are typically complex, heavyweight, and require substantial training data and computational resources. We propose a methodology to partially compress these networks by reducing the size of their encoders. Our approach uses a simplified knowledge distillation strategy to approximate the latent space of the original models with less data and shorter training, yielding lightweight encoders from heavyweight ones. We evaluate the resulting lightweight encoders across two different architectures on the image compression task. Experiments show that our method preserves reconstruction quality and statistical fidelity better than training lightweight encoders with the original loss, making it practical for resource-limited environments.
>
---
#### [new 035] WaveRNet: Wavelet-Guided Frequency Learning for Multi-Source Domain-Generalized Retinal Vessel Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决多源域泛化下的视网膜血管分割问题。针对光照和对比度变化导致的性能下降，提出WaveRNet框架，融合小波频域信息与SAM，提升分割精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.05942v1](https://arxiv.org/pdf/2601.05942v1)**

> **作者:** Chanchan Wang; Yuanfang Wang; Qing Xu; Guanxin Chen
>
> **摘要:** Domain-generalized retinal vessel segmentation is critical for automated ophthalmic diagnosis, yet faces significant challenges from domain shift induced by non-uniform illumination and varying contrast, compounded by the difficulty of preserving fine vessel structures. While the Segment Anything Model (SAM) exhibits remarkable zero-shot capabilities, existing SAM-based methods rely on simple adapter fine-tuning while overlooking frequency-domain information that encodes domain-invariant features, resulting in degraded generalization under illumination and contrast variations. Furthermore, SAM's direct upsampling inevitably loses fine vessel details. To address these limitations, we propose WaveRNet, a wavelet-guided frequency learning framework for robust multi-source domain-generalized retinal vessel segmentation. Specifically, we devise a Spectral-guided Domain Modulator (SDM) that integrates wavelet decomposition with learnable domain tokens, enabling the separation of illumination-robust low-frequency structures from high-frequency vessel boundaries while facilitating domain-specific feature generation. Furthermore, we introduce a Frequency-Adaptive Domain Fusion (FADF) module that performs intelligent test-time domain selection through wavelet-based frequency similarity and soft-weighted fusion. Finally, we present a Hierarchical Mask-Prompt Refiner (HMPR) that overcomes SAM's upsampling limitation through coarse-to-fine refinement with long-range dependency modeling. Extensive experiments under the Leave-One-Domain-Out protocol on four public retinal datasets demonstrate that WaveRNet achieves state-of-the-art generalization performance. The source code is available at https://github.com/Chanchan-Wang/WaveRNet.
>
---
#### [new 036] ViTNT-FIQA: Training-Free Face Image Quality Assessment with Vision Transformers
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人脸图像质量评估任务，旨在无需训练的情况下评估图像质量。提出ViTNT-FIQA方法，通过分析ViT中间块的特征稳定性来判断图像质量。**

- **链接: [https://arxiv.org/pdf/2601.05741v1](https://arxiv.org/pdf/2601.05741v1)**

> **作者:** Guray Ozgur; Eduarda Caldeira; Tahar Chettaoui; Jan Niklas Kolf; Marco Huber; Naser Damer; Fadi Boutros
>
> **备注:** Accepted at WACV Workshops
>
> **摘要:** Face Image Quality Assessment (FIQA) is essential for reliable face recognition systems. Current approaches primarily exploit only final-layer representations, while training-free methods require multiple forward passes or backpropagation. We propose ViTNT-FIQA, a training-free approach that measures the stability of patch embedding evolution across intermediate Vision Transformer (ViT) blocks. We demonstrate that high-quality face images exhibit stable feature refinement trajectories across blocks, while degraded images show erratic transformations. Our method computes Euclidean distances between L2-normalized patch embeddings from consecutive transformer blocks and aggregates them into image-level quality scores. We empirically validate this correlation on a quality-labeled synthetic dataset with controlled degradation levels. Unlike existing training-free approaches, ViTNT-FIQA requires only a single forward pass without backpropagation or architectural modifications. Through extensive evaluation on eight benchmarks (LFW, AgeDB-30, CFP-FP, CALFW, Adience, CPLFW, XQLFW, IJB-C), we show that ViTNT-FIQA achieves competitive performance with state-of-the-art methods while maintaining computational efficiency and immediate applicability to any pre-trained ViT-based face recognition model.
>
---
#### [new 037] DIFF-MF: A Difference-Driven Channel-Spatial State Space Model for Multi-Modal Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于多模态图像融合任务，旨在解决现有方法在保留细节与热源显著性之间的平衡问题。提出DIFF-MF模型，通过特征差异引导融合，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2601.05538v1](https://arxiv.org/pdf/2601.05538v1)**

> **作者:** Yiming Sun; Zifan Ye; Qinghua Hu; Pengfei Zhu
>
> **摘要:** Multi-modal image fusion aims to integrate complementary information from multiple source images to produce high-quality fused images with enriched content. Although existing approaches based on state space model have achieved satisfied performance with high computational efficiency, they tend to either over-prioritize infrared intensity at the cost of visible details, or conversely, preserve visible structure while diminishing thermal target salience. To overcome these challenges, we propose DIFF-MF, a novel difference-driven channel-spatial state space model for multi-modal image fusion. Our approach leverages feature discrepancy maps between modalities to guide feature extraction, followed by a fusion process across both channel and spatial dimensions. In the channel dimension, a channel-exchange module enhances channel-wise interaction through cross-attention dual state space modeling, enabling adaptive feature reweighting. In the spatial dimension, a spatial-exchange module employs cross-modal state space scanning to achieve comprehensive spatial fusion. By efficiently capturing global dependencies while maintaining linear computational complexity, DIFF-MF effectively integrates complementary multi-modal features. Experimental results on the driving scenarios and low-altitude UAV datasets demonstrate that our method outperforms existing approaches in both visual quality and quantitative evaluation.
>
---
#### [new 038] Generalizable and Adaptive Continual Learning Framework for AI-generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决检测模型泛化能力差和适应性不足的问题。通过提出三阶段持续学习框架，提升模型对新型生成模型的适应能力。**

- **链接: [https://arxiv.org/pdf/2601.05580v1](https://arxiv.org/pdf/2601.05580v1)**

> **作者:** Hanyi Wang; Jun Lan; Yaoyu Kang; Huijia Zhu; Weiqiang Wang; Zhuosheng Zhang; Shilin Wang
>
> **备注:** Accepted by TMM 2025
>
> **摘要:** The malicious misuse and widespread dissemination of AI-generated images pose a significant threat to the authenticity of online information. Current detection methods often struggle to generalize to unseen generative models, and the rapid evolution of generative techniques continuously exacerbates this challenge. Without adaptability, detection models risk becoming ineffective in real-world applications. To address this critical issue, we propose a novel three-stage domain continual learning framework designed for continuous adaptation to evolving generative models. In the first stage, we employ a strategic parameter-efficient fine-tuning approach to develop a transferable offline detection model with strong generalization capabilities. Building upon this foundation, the second stage integrates unseen data streams into a continual learning process. To efficiently learn from limited samples of novel generated models and mitigate overfitting, we design a data augmentation chain with progressively increasing complexity. Furthermore, we leverage the Kronecker-Factored Approximate Curvature (K-FAC) method to approximate the Hessian and alleviate catastrophic forgetting. Finally, the third stage utilizes a linear interpolation strategy based on Linear Mode Connectivity, effectively capturing commonalities across diverse generative models and further enhancing overall performance. We establish a comprehensive benchmark of 27 generative models, including GANs, deepfakes, and diffusion models, chronologically structured up to August 2024 to simulate real-world scenarios. Extensive experiments demonstrate that our initial offline detectors surpass the leading baseline by +5.51% in terms of mean average precision. Our continual learning strategy achieves an average accuracy of 92.20%, outperforming state-of-the-art methods.
>
---
#### [new 039] SceneAlign: Aligning Multimodal Reasoning to Scene Graphs in Complex Visual Scenes
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态推理任务，旨在解决复杂视觉场景中推理不准确的问题。通过构建结构化干预，提升模型对视觉信息的精准理解与推理能力。**

- **链接: [https://arxiv.org/pdf/2601.05600v1](https://arxiv.org/pdf/2601.05600v1)**

> **作者:** Chuhan Wang; Xintong Li; Jennifer Yuntong Zhang; Junda Wu; Chengkai Huang; Lina Yao; Julian McAuley; Jingbo Shang
>
> **备注:** Preprint
>
> **摘要:** Multimodal large language models often struggle with faithful reasoning in complex visual scenes, where intricate entities and relations require precise visual grounding at each step. This reasoning unfaithfulness frequently manifests as hallucinated entities, mis-grounded relations, skipped steps, and over-specified reasoning. Existing preference-based approaches, typically relying on textual perturbations or answer-conditioned rationales, fail to address this challenge as they allow models to exploit language priors to bypass visual grounding. To address this, we propose SceneAlign, a framework that leverages scene graphs as structured visual information to perform controllable structural interventions. By identifying reasoning-critical nodes and perturbing them through four targeted strategies that mimic typical grounding failures, SceneAlign constructs hard negative rationales that remain linguistically plausible but are grounded in inaccurate visual facts. These contrastive pairs are used in Direct Preference Optimization to steer models toward fine-grained, structure-faithful reasoning. Across seven visual reasoning benchmarks, SceneAlign consistently improves answer accuracy and reasoning faithfulness, highlighting the effectiveness of grounding-aware alignment for multimodal reasoning.
>
---
#### [new 040] Quantifying and Inducing Shape Bias in CNNs via Max-Pool Dilation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解决CNN在形状数据上的纹理偏差问题。通过量化数据集的形状-纹理平衡，并调整最大池化 dilation 来增强形状偏好，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.05599v1](https://arxiv.org/pdf/2601.05599v1)**

> **作者:** Takito Sawada; Akinori Iwata; Masahiro Okuda
>
> **备注:** Accepted to IEVC 2026. 4 pages, 1 figure, 3 tables
>
> **摘要:** Convolutional Neural Networks (CNNs) are known to exhibit a strong texture bias, favoring local patterns over global shape information--a tendency inherent to their convolutional architecture. While this bias is beneficial for texture-rich natural images, it often degrades performance on shape-dominant data such as illustrations and sketches. Although prior work has proposed shape-biased models to mitigate this issue, these approaches lack a quantitative metric for identifying which datasets would actually benefit from such modifications. To address this gap, we propose a data-driven metric that quantifies the shape-texture balance of a dataset by computing the Structural Similarity Index (SSIM) between each image's luminance channel and its L0-smoothed counterpart. Building on this metric, we further introduce a computationally efficient adaptation method that promotes shape bias by modifying the dilation of max-pooling operations while keeping convolutional weights frozen. Experimental results show that this approach consistently improves classification accuracy on shape-dominant datasets, particularly in low-data regimes where full fine-tuning is impractical, requiring training only the final classification layer.
>
---
#### [new 041] Adaptive Conditional Contrast-Agnostic Deformable Image Registration with Uncertainty Estimation
- **分类: cs.CV**

- **简介: 该论文属于图像配准任务，解决多对比度图像配准泛化能力差的问题。提出AC-CAR框架，实现任意对比度的准确配准与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2601.05981v1](https://arxiv.org/pdf/2601.05981v1)**

> **作者:** Yinsong Wang; Xinzhe Luo; Siyi Du; Chen Qin
>
> **备注:** Accepted by ieee transactions on Medical Imaging
>
> **摘要:** Deformable multi-contrast image registration is a challenging yet crucial task due to the complex, non-linear intensity relationships across different imaging contrasts. Conventional registration methods typically rely on iterative optimization of the deformation field, which is time-consuming. Although recent learning-based approaches enable fast and accurate registration during inference, their generalizability remains limited to the specific contrasts observed during training. In this work, we propose an adaptive conditional contrast-agnostic deformable image registration framework (AC-CAR) based on a random convolution-based contrast augmentation scheme. AC-CAR can generalize to arbitrary imaging contrasts without observing them during training. To encourage contrast-invariant feature learning, we propose an adaptive conditional feature modulator (ACFM) that adaptively modulates the features and the contrast-invariant latent regularization to enforce the consistency of the learned feature across different imaging contrasts. Additionally, we enable our framework to provide contrast-agnostic registration uncertainty by integrating a variance network that leverages the contrast-agnostic registration encoder to improve the trustworthiness and reliability of AC-CAR. Experimental results demonstrate that AC-CAR outperforms baseline methods in registration accuracy and exhibits superior generalization to unseen imaging contrasts. Code is available at https://github.com/Yinsong0510/AC-CAR.
>
---
#### [new 042] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督医学图像分割任务，旨在解决标注数据不足及数据流间交互问题。提出BCSI框架，包含SSP、CR和BCI机制，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2601.05855v1](https://arxiv.org/pdf/2601.05855v1)**

> **作者:** Kaiwen Huang; Yizhe Zhang; Yi Zhou; Tianyang Xu; Tao Zhou
>
> **备注:** Accepted to AAAI 2026. Code at: https://github.com/taozh2017/BCSI
>
> **摘要:** Semi-supervised medical image segmentation is an effective method for addressing scenarios with limited labeled data. Existing methods mainly rely on frameworks such as mean teacher and dual-stream consistency learning. These approaches often face issues like error accumulation and model structural complexity, while also neglecting the interaction between labeled and unlabeled data streams. To overcome these challenges, we propose a Bidirectional Channel-selective Semantic Interaction~(BCSI) framework for semi-supervised medical image segmentation. First, we propose a Semantic-Spatial Perturbation~(SSP) mechanism, which disturbs the data using two strong augmentation operations and leverages unsupervised learning with pseudo-labels from weak augmentations. Additionally, we employ consistency on the predictions from the two strong augmentations to further improve model stability and robustness. Second, to reduce noise during the interaction between labeled and unlabeled data, we propose a Channel-selective Router~(CR) component, which dynamically selects the most relevant channels for information exchange. This mechanism ensures that only highly relevant features are activated, minimizing unnecessary interference. Finally, the Bidirectional Channel-wise Interaction~(BCI) strategy is employed to supplement additional semantic information and enhance the representation of important channels. Experimental results on multiple benchmarking 3D medical datasets demonstrate that the proposed method outperforms existing semi-supervised approaches.
>
---
#### [new 043] Hippocampal Atrophy Patterns Across the Alzheimer's Disease Spectrum: A Voxel-Based Morphometry Analysis
- **分类: cs.CV**

- **简介: 该论文属于神经影像分析任务，旨在研究阿尔茨海默病谱系中海马萎缩模式，通过 voxel-based morphometry 分析 MRI 数据，探讨疾病进展与生物标志物关系。**

- **链接: [https://arxiv.org/pdf/2601.05494v1](https://arxiv.org/pdf/2601.05494v1)**

> **作者:** Trishna Niraula
>
> **备注:** 8 pages, 7 figures, 6 tables
>
> **摘要:** Alzheimer's disease (AD) and mild cognitive impairment (MCI) are associated with progressive gray matter loss, particularly in medial temporal structures. In this study, CAT12/SPM12 voxel-based morphometry was applied to baseline T1-weighted MRI scans from 249 ADNI participants (CN = 90, MCI = 129, AD = 30). Gray matter volume was analyzed using a general linear model, with the diagnostic group as primary predictor and age and total intracranial volume as covariates. Statistical maps were thresholded at p < 0.001 (voxelwise) and corrected for multiple comparisons at the cluster level using family-wise error (FWE) correction (p < 0.05). Significant hippocampal atrophy was observed in AD relative to CN and MCI (Cohen's d = 2.03 and 1.61, respectively). Hippocampal volume demonstrated moderate predictive value for conversion from MCI to AD (AUC = 0.66). Stratification by APOE4 status did not reveal significant genetic effects on cross-sectional hippocampal volume. These results support medial temporal degeneration as a key feature of AD progression and provide insights into predictive biomarkers and genetic influences.
>
---
#### [new 044] Goal Force: Teaching Video Models To Accomplish Physics-Conditioned Goals
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视频生成任务，旨在解决视频模型目标定义困难的问题。通过引入力向量和动态过程作为目标，训练模型实现物理条件下的目标达成。**

- **链接: [https://arxiv.org/pdf/2601.05848v1](https://arxiv.org/pdf/2601.05848v1)**

> **作者:** Nate Gillman; Yinghua Zhou; Zitian Tang; Evan Luo; Arjan Chakravarthy; Daksh Aggarwal; Michael Freeman; Charles Herrmann; Chen Sun
>
> **备注:** Code and interactive demos at https://goal-force.github.io/
>
> **摘要:** Recent advancements in video generation have enabled the development of ``world models'' capable of simulating potential futures for robotics and planning. However, specifying precise goals for these models remains a challenge; text instructions are often too abstract to capture physical nuances, while target images are frequently infeasible to specify for dynamic tasks. To address this, we introduce Goal Force, a novel framework that allows users to define goals via explicit force vectors and intermediate dynamics, mirroring how humans conceptualize physical tasks. We train a video generation model on a curated dataset of synthetic causal primitives-such as elastic collisions and falling dominos-teaching it to propagate forces through time and space. Despite being trained on simple physics data, our model exhibits remarkable zero-shot generalization to complex, real-world scenarios, including tool manipulation and multi-object causal chains. Our results suggest that by grounding video generation in fundamental physical interactions, models can emerge as implicit neural physics simulators, enabling precise, physics-aware planning without reliance on external engines. We release all datasets, code, model weights, and interactive video demos at our project page.
>
---
#### [new 045] Phase4DFD: Multi-Domain Phase-Aware Attention for Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决传统方法忽视相位信息的问题。通过引入相位感知的注意力机制，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.05861v1](https://arxiv.org/pdf/2601.05861v1)**

> **作者:** Zhen-Xin Lin; Shang-Kuan Chen
>
> **备注:** 15 pages, 3 figures, conference
>
> **摘要:** Recent deepfake detection methods have increasingly explored frequency domain representations to reveal manipulation artifacts that are difficult to detect in the spatial domain. However, most existing approaches rely primarily on spectral magnitude, implicitly under exploring the role of phase information. In this work, we propose Phase4DFD, a phase aware frequency domain deepfake detection framework that explicitly models phase magnitude interactions via a learnable attention mechanism. Our approach augments standard RGB input with Fast Fourier Transform (FFT) magnitude and local binary pattern (LBP) representations to expose subtle synthesis artifacts that remain indistinguishable under spatial analysis alone. Crucially, we introduce an input level phase aware attention module that uses phase discontinuities commonly introduced by synthetic generation to guide the model toward frequency patterns that are most indicative of manipulation before backbone feature extraction. The attended multi domain representation is processed by an efficient BNext M backbone, with optional channel spatial attention applied for semantic feature refinement. Extensive experiments on the CIFAKE and DFFD datasets demonstrate that our proposed model Phase4DFD outperforms state of the art spatial and frequency-based detectors while maintaining low computational overhead. Comprehensive ablation studies further confirm that explicit phase modeling provides complementary and non-redundant information beyond magnitude-only frequency representations.
>
---
#### [new 046] Rotate Your Character: Revisiting Video Diffusion Models for High-Quality 3D Character Generation
- **分类: cs.CV**

- **简介: 该论文属于3D角色生成任务，旨在解决单图生成高质量3D角色的问题。提出RCM框架，实现复杂姿态角色的统一视角合成与高分辨率视频生成。**

- **链接: [https://arxiv.org/pdf/2601.05722v1](https://arxiv.org/pdf/2601.05722v1)**

> **作者:** Jin Wang; Jianxiang Lu; Comi Chen; Guangzheng Xu; Haoyu Yang; Peng Chen; Na Zhang; Yifan Xu; Longhuang Wu; Shuai Shao; Qinglin Lu; Ping Luo
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Generating high-quality 3D characters from single images remains a significant challenge in digital content creation, particularly due to complex body poses and self-occlusion. In this paper, we present RCM (Rotate your Character Model), an advanced image-to-video diffusion framework tailored for high-quality novel view synthesis (NVS) and 3D character generation. Compared to existing diffusion-based approaches, RCM offers several key advantages: (1) transferring characters with any complex poses into a canonical pose, enabling consistent novel view synthesis across the entire viewing orbit, (2) high-resolution orbital video generation at 1024x1024 resolution, (3) controllable observation positions given different initial camera poses, and (4) multi-view conditioning supporting up to 4 input images, accommodating diverse user scenarios. Extensive experiments demonstrate that RCM outperforms state-of-the-art methods in both novel view synthesis and 3D generation quality.
>
---
#### [new 047] VideoAR: Autoregressive Video Generation via Next-Frame & Scale Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VideoAR，一种基于自回归的视频生成框架，解决计算复杂和难以扩展的问题。通过多尺度帧预测和时序建模，提升生成效率与一致性。**

- **链接: [https://arxiv.org/pdf/2601.05966v1](https://arxiv.org/pdf/2601.05966v1)**

> **作者:** Longbin Ji; Xiaoxiong Liu; Junyuan Shang; Shuohuan Wang; Yu Sun; Hua Wu; Haifeng Wang
>
> **摘要:** Recent advances in video generation have been dominated by diffusion and flow-matching models, which produce high-quality results but remain computationally intensive and difficult to scale. In this work, we introduce VideoAR, the first large-scale Visual Autoregressive (VAR) framework for video generation that combines multi-scale next-frame prediction with autoregressive modeling. VideoAR disentangles spatial and temporal dependencies by integrating intra-frame VAR modeling with causal next-frame prediction, supported by a 3D multi-scale tokenizer that efficiently encodes spatio-temporal dynamics. To improve long-term consistency, we propose Multi-scale Temporal RoPE, Cross-Frame Error Correction, and Random Frame Mask, which collectively mitigate error propagation and stabilize temporal coherence. Our multi-stage pretraining pipeline progressively aligns spatial and temporal learning across increasing resolutions and durations. Empirically, VideoAR achieves new state-of-the-art results among autoregressive models, improving FVD on UCF-101 from 99.5 to 88.6 while reducing inference steps by over 10x, and reaching a VBench score of 81.74-competitive with diffusion-based models an order of magnitude larger. These results demonstrate that VideoAR narrows the performance gap between autoregressive and diffusion paradigms, offering a scalable, efficient, and temporally consistent foundation for future video generation research.
>
---
#### [new 048] Semi-Supervised Facial Expression Recognition based on Dynamic Threshold and Negative Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于面部表情识别任务，旨在解决标注数据成本高的问题。通过动态阈值和选择性负学习，有效利用标签和未标签数据，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2601.05556v1](https://arxiv.org/pdf/2601.05556v1)**

> **作者:** Zhongpeng Cai; Jun Yu; Wei Xu; Tianyu Liu; Jianqing Sun; Jiaen Liang
>
> **摘要:** Facial expression recognition is a key task in human-computer interaction and affective computing. However, acquiring a large amount of labeled facial expression data is often costly. Therefore, it is particularly important to design a semi-supervised facial expression recognition algorithm that makes full use of both labeled and unlabeled data. In this paper, we propose a semi-supervised facial expression recognition algorithm based on Dynamic Threshold Adjustment (DTA) and Selective Negative Learning (SNL). Initially, we designed strategies for local attention enhancement and random dropout of feature maps during feature extraction, which strengthen the representation of local features while ensuring the model does not overfit to any specific local area. Furthermore, this study introduces a dynamic thresholding method to adapt to the requirements of the semi-supervised learning framework for facial expression recognition tasks, and through a selective negative learning strategy, it fully utilizes unlabeled samples with low confidence by mining useful expression information from complementary labels, achieving impressive results. We have achieved state-of-the-art performance on the RAF-DB and AffectNet datasets. Our method surpasses fully supervised methods even without using the entire dataset, which proves the effectiveness of our approach.
>
---
#### [new 049] LatentVLA: Efficient Vision-Language Models for Autonomous Driving via Latent Action Prediction
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，解决罕见场景下模型性能不足的问题。通过无语言标注的潜在动作预测，提升模型效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.05611v1](https://arxiv.org/pdf/2601.05611v1)**

> **作者:** Chengen Xie; Bin Sun; Tianyu Li; Junjie Wu; Zhihui Hao; XianPeng Lang; Hongyang Li
>
> **摘要:** End-to-end autonomous driving models trained on largescale datasets perform well in common scenarios but struggle with rare, long-tail situations due to limited scenario diversity. Recent Vision-Language-Action (VLA) models leverage broad knowledge from pre-trained visionlanguage models to address this limitation, yet face critical challenges: (1) numerical imprecision in trajectory prediction due to discrete tokenization, (2) heavy reliance on language annotations that introduce linguistic bias and annotation burden, and (3) computational inefficiency from multi-step chain-of-thought reasoning hinders real-time deployment. We propose LatentVLA, a novel framework that employs self-supervised latent action prediction to train VLA models without language annotations, eliminating linguistic bias while learning rich driving representations from unlabeled trajectory data. Through knowledge distillation, LatentVLA transfers the generalization capabilities of VLA models to efficient vision-based networks, achieving both robust performance and real-time efficiency. LatentVLA establishes a new state-of-the-art on the NAVSIM benchmark with a PDMS score of 92.4 and demonstrates strong zeroshot generalization on the nuScenes benchmark.
>
---
#### [new 050] Adaptive Disentangled Representation Learning for Incomplete Multi-View Multi-Label Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多视图多标签分类任务，解决特征缺失和标注不全的问题。提出ADRL方法，通过特征补全、表示解耦和标签建模提升性能。**

- **链接: [https://arxiv.org/pdf/2601.05785v1](https://arxiv.org/pdf/2601.05785v1)**

> **作者:** Quanjiang Li; Zhiming Liu; Tianxiang Xu; Tingjin Luo; Chenping Hou
>
> **摘要:** Multi-view multi-label learning frequently suffers from simultaneous feature absence and incomplete annotations, due to challenges in data acquisition and cost-intensive supervision. To tackle the complex yet highly practical problem while overcoming the existing limitations of feature recovery, representation disentanglement, and label semantics modeling, we propose an Adaptive Disentangled Representation Learning method (ADRL). ADRL achieves robust view completion by propagating feature-level affinity across modalities with neighborhood awareness, and reinforces reconstruction effectiveness by leveraging a stochastic masking strategy. Through disseminating category-level association across label distributions, ADRL refines distribution parameters for capturing interdependent label prototypes. Besides, we formulate a mutual-information-based objective to promote consistency among shared representations and suppress information overlap between view-specific representation and other modalities. Theoretically, we derive the tractable bounds to train the dual-channel network. Moreover, ADRL performs prototype-specific feature selection by enabling independent interactions between label embeddings and view representations, accompanied by the generation of pseudo-labels for each category. The structural characteristics of the pseudo-label space are then exploited to guide a discriminative trade-off during view fusion. Finally, extensive experiments on public datasets and real-world applications demonstrate the superior performance of ADRL.
>
---
#### [new 051] Towards Generalized Multi-Image Editing for Unified Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于多图像编辑任务，旨在解决统一多模态模型在跨图像引用细节时的视觉一致性和歧义问题。提出可扩展的多图像编辑框架，通过潜在分离器和正弦索引编码提升模型的准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.05572v1](https://arxiv.org/pdf/2601.05572v1)**

> **作者:** Pengcheng Xu; Peng Tang; Donghao Luo; Xiaobin Hu; Weichu Cui; Qingdong He; Zhennan Chen; Jiangning Zhang; Charles Ling; Boyu Wang
>
> **备注:** Project page: https://github.com/Pengchengpcx/MIE-UMM
>
> **摘要:** Unified Multimodal Models (UMMs) integrate multimodal understanding and generation, yet they are limited to maintaining visual consistency and disambiguating visual cues when referencing details across multiple input images. In this work, we propose a scalable multi-image editing framework for UMMs that explicitly distinguishes image identities and generalizes to variable input counts. Algorithmically, we introduce two innovations: 1) The learnable latent separators explicitly differentiate each reference image in the latent space, enabling accurate and disentangled conditioning. 2) The sinusoidal index encoding assigns visual tokens from the same image a continuous sinusoidal index embedding, which provides explicit image identity while allowing generalization and extrapolation on a variable number of inputs. To facilitate training and evaluation, we establish a high-fidelity benchmark using an inverse dataset construction methodology to guarantee artifact-free, achievable outputs. Experiments show clear improvements in semantic consistency, visual fidelity, and cross-image integration over prior baselines on diverse multi-image editing tasks, validating our advantages on consistency and generalization ability.
>
---
#### [new 052] Orient Anything V2: Unifying Orientation and Rotation Understanding
- **分类: cs.CV**

- **简介: 该论文提出Orient Anything V2，解决3D物体方向与旋转理解问题，通过创新方法提升方向估计和旋转预测性能。**

- **链接: [https://arxiv.org/pdf/2601.05573v1](https://arxiv.org/pdf/2601.05573v1)**

> **作者:** Zehan Wang; Ziang Zhang; Jiayang Xu; Jialei Wang; Tianyu Pang; Chao Du; HengShuang Zhao; Zhou Zhao
>
> **备注:** NeurIPS 2025 Spotlight, Repo: https://github.com/SpatialVision/Orient-Anything-V2
>
> **摘要:** This work presents Orient Anything V2, an enhanced foundation model for unified understanding of object 3D orientation and rotation from single or paired images. Building upon Orient Anything V1, which defines orientation via a single unique front face, V2 extends this capability to handle objects with diverse rotational symmetries and directly estimate relative rotations. These improvements are enabled by four key innovations: 1) Scalable 3D assets synthesized by generative models, ensuring broad category coverage and balanced data distribution; 2) An efficient, model-in-the-loop annotation system that robustly identifies 0 to N valid front faces for each object; 3) A symmetry-aware, periodic distribution fitting objective that captures all plausible front-facing orientations, effectively modeling object rotational symmetry; 4) A multi-frame architecture that directly predicts relative object rotations. Extensive experiments show that Orient Anything V2 achieves state-of-the-art zero-shot performance on orientation estimation, 6DoF pose estimation, and object symmetry recognition across 11 widely used benchmarks. The model demonstrates strong generalization, significantly broadening the applicability of orientation estimation in diverse downstream tasks.
>
---
#### [new 053] TAPM-Net: Trajectory-Aware Perturbation Modeling for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决信号弱、背景杂等问题。提出TAPM-Net，通过建模特征扰动轨迹提升检测性能。**

- **链接: [https://arxiv.org/pdf/2601.05446v1](https://arxiv.org/pdf/2601.05446v1)**

> **作者:** Hongyang Xie; Hongyang He; Victor Sanchez
>
> **备注:** Published in BMVC 2025 see: https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_709/paper.pdf. Conference version. 12 pages, 6 figures, 4 tables. Author-prepared version
>
> **摘要:** Infrared small target detection (ISTD) remains a long-standing challenge due to weak signal contrast, limited spatial extent, and cluttered backgrounds. Despite performance improvements from convolutional neural networks (CNNs) and Vision Transformers (ViTs), current models lack a mechanism to trace how small targets trigger directional, layer-wise perturbations in the feature space, which is an essential cue for distinguishing signal from structured noise in infrared scenes. To address this limitation, we propose the Trajectory-Aware Mamba Propagation Network (TAPM-Net), which explicitly models the spatial diffusion behavior of target-induced feature disturbances. TAPM-Net is built upon two novel components: a Perturbation-guided Path Module (PGM) and a Trajectory-Aware State Block (TASB). The PGM constructs perturbation energy fields from multi-level features and extracts gradient-following feature trajectories that reflect the directionality of local responses. The resulting feature trajectories are fed into the TASB, a Mamba-based state-space unit that models dynamic propagation along each trajectory while incorporating velocity-constrained diffusion and semantically aligned feature fusion from word-level and sentence-level embeddings. Unlike existing attention-based methods, TAPM-Net enables anisotropic, context-sensitive state transitions along spatial trajectories while maintaining global coherence at low computational cost. Experiments on NUAA-SIRST and IRSTD-1K demonstrate that TAPM-Net achieves state-of-the-art performance in ISTD.
>
---
#### [new 054] SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，解决VLM在复杂场景中缺乏驾驶特定时空理解的问题。提出SGDrive框架，通过场景-代理-目标层次结构提升轨迹规划能力。**

- **链接: [https://arxiv.org/pdf/2601.05640v1](https://arxiv.org/pdf/2601.05640v1)**

> **作者:** Jingyu Li; Junjie Wu; Dongnan Hu; Xiangkai Huang; Bin Sun; Zhihui Hao; Xianpeng Lang; Xiatian Zhu; Li Zhang
>
> **摘要:** Recent end-to-end autonomous driving approaches have leveraged Vision-Language Models (VLMs) to enhance planning capabilities in complex driving scenarios. However, VLMs are inherently trained as generalist models, lacking specialized understanding of driving-specific reasoning in 3D space and time. When applied to autonomous driving, these models struggle to establish structured spatial-temporal representations that capture geometric relationships, scene context, and motion patterns critical for safe trajectory planning. To address these limitations, we propose SGDrive, a novel framework that explicitly structures the VLM's representation learning around driving-specific knowledge hierarchies. Built upon a pre-trained VLM backbone, SGDrive decomposes driving understanding into a scene-agent-goal hierarchy that mirrors human driving cognition: drivers first perceive the overall environment (scene context), then attend to safety-critical agents and their behaviors, and finally formulate short-term goals before executing actions. This hierarchical decomposition provides the structured spatial-temporal representation that generalist VLMs lack, integrating multi-level information into a compact yet comprehensive format for trajectory planning. Extensive experiments on the NAVSIM benchmark demonstrate that SGDrive achieves state-of-the-art performance among camera-only methods on both PDMS and EPDMS, validating the effectiveness of hierarchical knowledge structuring for adapting generalist VLMs to autonomous driving.
>
---
#### [new 055] Prompt-Free SAM-Based Multi-Task Framework for Breast Ultrasound Lesion Segmentation and Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于乳腺超声病变分割与分类任务，旨在解决低对比度、噪声和形态多样性带来的挑战。通过SAM的嵌入进行多任务学习，提升分割与诊断精度。**

- **链接: [https://arxiv.org/pdf/2601.05498v1](https://arxiv.org/pdf/2601.05498v1)**

> **作者:** Samuel E. Johnny; Bernes L. Atabonfack; Israel Alagbe; Assane Gueye
>
> **摘要:** Accurate tumor segmentation and classification in breast ultrasound (BUS) imaging remain challenging due to low contrast, speckle noise, and diverse lesion morphology. This study presents a multi-task deep learning framework that jointly performs lesion segmentation and diagnostic classification using embeddings from the Segment Anything Model (SAM) vision encoder. Unlike prompt-based SAM variants, our approach employs a prompt-free, fully supervised adaptation where high-dimensional SAM features are decoded through either a lightweight convolutional head or a UNet-inspired decoder for pixel-wise segmentation. The classification branch is enhanced via mask-guided attention, allowing the model to focus on lesion-relevant features while suppressing background artifacts. Experiments on the PRECISE 2025 breast ultrasound dataset, split per class into 80 percent training and 20 percent testing, show that the proposed method achieves a Dice Similarity Coefficient (DSC) of 0.887 and an accuracy of 92.3 percent, ranking among the top entries on the PRECISE challenge leaderboard. These results demonstrate that SAM-based representations, when coupled with segmentation-guided learning, significantly improve both lesion delineation and diagnostic prediction in breast ultrasound imaging.
>
---
#### [new 056] GeoSurDepth: Spatial Geometry-Consistent Self-Supervised Depth Estimation for Surround-View Cameras
- **分类: cs.CV**

- **简介: 该论文属于多视角深度估计任务，旨在提升自动驾驶中周围摄像头的深度预测精度。通过引入几何一致性约束和自监督学习，解决单视角重建不足的问题。**

- **链接: [https://arxiv.org/pdf/2601.05839v1](https://arxiv.org/pdf/2601.05839v1)**

> **作者:** Weimin Liu; Wenjun Wang; Joshua H. Meng
>
> **摘要:** Accurate surround-view depth estimation provides a competitive alternative to laser-based sensors and is essential for 3D scene understanding in autonomous driving. While prior studies have proposed various approaches that primarily focus on enforcing cross-view constraints at the photometric level, few explicitly exploit the rich geometric structure inherent in both monocular and surround-view setting. In this work, we propose GeoSurDepth, a framework that leverages geometry consistency as the primary cue for surround-view depth estimation. Concretely, we utilize foundation models as a pseudo geometry prior and feature representation enhancement tool to guide the network to maintain surface normal consistency in spatial 3D space and regularize object- and texture-consistent depth estimation in 2D. In addition, we introduce a novel view synthesis pipeline where 2D-3D lifting is achieved with dense depth reconstructed via spatial warping, encouraging additional photometric supervision across temporal, spatial, and spatial-temporal contexts, and compensating for the limitations of single-view image reconstruction. Finally, a newly-proposed adaptive joint motion learning strategy enables the network to adaptively emphasize informative spatial geometry cues for improved motion reasoning. Extensive experiments on DDAD and nuScenes demonstrate that GeoSurDepth achieves state-of-the-art performance, validating the effectiveness of our approach. Our framework highlights the importance of exploiting geometry coherence and consistency for robust self-supervised multi-view depth estimation.
>
---
#### [new 057] AGDC: Autoregressive Generation of Variable-Length Sequences with Joint Discrete and Continuous Spaces
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出AGDC框架，解决高精度混合序列生成问题，通过联合建模离散与连续值，提升半导体设计等领域的生成质量。**

- **链接: [https://arxiv.org/pdf/2601.05680v1](https://arxiv.org/pdf/2601.05680v1)**

> **作者:** Yeonsang Shin; Insoo Kim; Bongkeun Kim; Keonwoo Bae; Bohyung Han
>
> **摘要:** Transformer-based autoregressive models excel in data generation but are inherently constrained by their reliance on discretized tokens, which limits their ability to represent continuous values with high precision. We analyze the scalability limitations of existing discretization-based approaches for generating hybrid discrete-continuous sequences, particularly in high-precision domains such as semiconductor circuit designs, where precision loss can lead to functional failure. To address the challenge, we propose AGDC, a novel unified framework that jointly models discrete and continuous values for variable-length sequences. AGDC employs a hybrid approach that combines categorical prediction for discrete values with diffusion-based modeling for continuous values, incorporating two key technical components: an end-of-sequence (EOS) logit adjustment mechanism that uses an MLP to dynamically adjust EOS token logits based on sequence context, and a length regularization term integrated into the loss function. Additionally, we present ContLayNet, a large-scale benchmark comprising 334K high-precision semiconductor layout samples with specialized evaluation metrics that capture functional correctness where precision errors significantly impact performance. Experiments on semiconductor layouts (ContLayNet), graphic layouts, and SVGs demonstrate AGDC's superior performance in generating high-fidelity hybrid vector representations compared to discretization-based and fixed-schema baselines, achieving scalable high-precision generation across diverse domains.
>
---
#### [new 058] PII-VisBench: Evaluating Personally Identifiable Information Safety in Vision Language Models Along a Continuum of Visibility
- **分类: cs.AI; cs.CL; cs.CR; cs.CV**

- **简介: 该论文属于隐私安全任务，旨在评估视觉语言模型中个人身份信息泄露问题。通过构建基准测试，分析不同在线可见度下的隐私保护效果，揭示模型在不同情况下的表现差异。**

- **链接: [https://arxiv.org/pdf/2601.05739v1](https://arxiv.org/pdf/2601.05739v1)**

> **作者:** G M Shahariar; Zabir Al Nazi; Md Olid Hasan Bhuiyan; Zhouxing Shi
>
> **摘要:** Vision Language Models (VLMs) are increasingly integrated into privacy-critical domains, yet existing evaluations of personally identifiable information (PII) leakage largely treat privacy as a static extraction task and ignore how a subject's online presence--the volume of their data available online--influences privacy alignment. We introduce PII-VisBench, a novel benchmark containing 4000 unique probes designed to evaluate VLM safety through the continuum of online presence. The benchmark stratifies 200 subjects into four visibility categories: high, medium, low, and zero--based on the extent and nature of their information available online. We evaluate 18 open-source VLMs (0.3B-32B) based on two key metrics: percentage of PII probing queries refused (Refusal Rate) and the fraction of non-refusal responses flagged for containing PII (Conditional PII Disclosure Rate). Across models, we observe a consistent pattern: refusals increase and PII disclosures decrease (9.10% high to 5.34% low) as subject visibility drops. We identify that models are more likely to disclose PII for high-visibility subjects, alongside substantial model-family heterogeneity and PII-type disparities. Finally, paraphrasing and jailbreak-style prompts expose attack and model-dependent failures, motivating visibility-aware safety evaluation and training interventions.
>
---
#### [new 059] Naiad: Novel Agentic Intelligent Autonomous System for Inland Water Monitoring
- **分类: cs.AI; cs.CL; cs.CV; cs.IR**

- **简介: 该论文提出NAIAD系统，用于内河水质监测任务，解决传统方法孤立处理问题的不足，通过整合AI与工具实现全面分析。**

- **链接: [https://arxiv.org/pdf/2601.05256v1](https://arxiv.org/pdf/2601.05256v1)**

> **作者:** Eirini Baltzi; Tilemachos Moumouris; Athena Psalta; Vasileios Tsironis; Konstantinos Karantzalos
>
> **摘要:** Inland water monitoring is vital for safeguarding public health and ecosystems, enabling timely interventions to mitigate risks. Existing methods often address isolated sub-problems such as cyanobacteria, chlorophyll, or other quality indicators separately. NAIAD introduces an agentic AI assistant that leverages Large Language Models (LLMs) and external analytical tools to deliver a holistic solution for inland water monitoring using Earth Observation (EO) data. Designed for both experts and non-experts, NAIAD provides a single-prompt interface that translates natural-language queries into actionable insights. Through Retrieval-Augmented Generation (RAG), LLM reasoning, external tool orchestration, computational graph execution, and agentic reflection, it retrieves and synthesizes knowledge from curated sources to produce tailored reports. The system integrates diverse tools for weather data, Sentinel-2 imagery, remote-sensing index computation (e.g., NDCI), chlorophyll-a estimation, and established platforms such as CyFi. Performance is evaluated using correctness and relevancy metrics, achieving over 77% and 85% respectively on a dedicated benchmark covering multiple user-expertise levels. Preliminary results show strong adaptability and robustness across query types. An ablation study on LLM backbones further highlights Gemma 3 (27B) and Qwen 2.5 (14B) as offering the best balance between computational efficiency and reasoning performance.
>
---
#### [new 060] Continual Learning of Achieving Forgetting-free and Positive Knowledge Transfer
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决灾难性遗忘和正向/反向知识迁移问题。提出ETCL方法，实现无遗忘和正向知识迁移。**

- **链接: [https://arxiv.org/pdf/2601.05623v1](https://arxiv.org/pdf/2601.05623v1)**

> **作者:** Zhi Wang; Zhongbin Wu; Yanni Li; Bing Liu; Guangxi Li; Yuping Wang
>
> **摘要:** Existing research on continual learning (CL) of a sequence of tasks focuses mainly on dealing with catastrophic forgetting (CF) to balance the learning plasticity of new tasks and the memory stability of old tasks. However, an ideal CL agent should not only be able to overcome CF, but also encourage positive forward and backward knowledge transfer (KT), i.e., using the learned knowledge from previous tasks for the new task learning (namely FKT), and improving the previous tasks' performance with the knowledge of the new task (namely BKT). To this end, this paper first models CL as an optimization problem in which each sequential learning task aims to achieve its optimal performance under the constraint that both FKT and BKT should be positive. It then proposes a novel Enhanced Task Continual Learning (ETCL) method, which achieves forgetting-free and positive KT. Furthermore, the bounds that can lead to negative FKT and BKT are estimated theoretically. Based on the bounds, a new strategy for online task similarity detection is also proposed to facilitate positive KT. To overcome CF, ETCL learns a set of task-specific binary masks to isolate a sparse sub-network for each task while preserving the performance of a dense network for the task. At the beginning of a new task learning, ETCL tries to align the new task's gradient with that of the sub-network of the previous most similar task to ensure positive FKT. By using a new bi-objective optimization strategy and an orthogonal gradient projection method, ETCL updates only the weights of previous similar tasks at the classification layer to achieve positive BKT. Extensive evaluations demonstrate that the proposed ETCL markedly outperforms strong baselines on dissimilar, similar, and mixed task sequences.
>
---
#### [new 061] Studying Illustrations in Manuscripts: An Efficient Deep-Learning Approach
- **分类: cs.IR; cs.CV; cs.LG**

- **简介: 该论文属于图像识别任务，旨在解决大规模手稿中插图检测与分析的问题。通过AI技术实现插图的自动检测、提取和描述，提升历史研究的效率与精度。**

- **链接: [https://arxiv.org/pdf/2601.05269v1](https://arxiv.org/pdf/2601.05269v1)**

> **作者:** Yoav Evron; Michal Bar-Asher Siegal; Michael Fire
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** The recent Artificial Intelligence (AI) revolution has opened transformative possibilities for the humanities, particularly in unlocking the visual content embedded in historical manuscripts. While digital archives now offer unprecedented access to these materials, the ability to systematically study illustrations at a large scale remains challenging. Our study presents a fast and scalable AI approach for detecting, extracting, and describing illustrations in digitized manuscripts. Focusing on collections like the Vatican Library, our system enables efficient visual analysis across millions of pages. Our pipeline consists of three stages: (1) a fine-tuned image classification model filters out text-only pages; (2) an efficient object detection model identifies and crops illustrations; and (3) a multimodal image captioning model generates concise, human-readable descriptions. These are stored in a searchable database, allowing scholars to retrieve relevant visual materials through keyword queries. By harnessing the power of recent AI advancements, we enable large-scale visual research that was previously impractical, empowering scholars in historical studies, art history, and cultural heritage to explore visual motifs, artistic styles, and cross-cultural influences with new precision and speed. Applying our pipeline to over three million digitized manuscript pages, we automatically identified and extracted more than 200,000 unique illustrations. This scale of processing in under 0.06 seconds per page, dramatically outperforms traditional segmentation techniques in both efficiency and accessibility for visual scholarship. Our work demonstrates how cutting-edge AI tools can profoundly reshape scholarly workflows and open new avenues for multidisciplinary research in the age of digital manuscripts.
>
---
#### [new 062] Router-Suggest: Dynamic Routing for Multimodal Auto-Completion in Visually-Grounded Dialogs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出MAC任务，解决多模态自动补全问题，通过Router-Suggest框架动态选择模型，提升效率与用户满意度。**

- **链接: [https://arxiv.org/pdf/2601.05851v1](https://arxiv.org/pdf/2601.05851v1)**

> **作者:** Sandeep Mishra; Devichand Budagam; Anubhab Mandal; Bishal Santra; Pawan Goyal; Manish Gupta
>
> **备注:** Accepted to EACL 2026 Industry Track, 12 pages, 6 figures
>
> **摘要:** Real-time multimodal auto-completion is essential for digital assistants, chatbots, design tools, and healthcare consultations, where user inputs rely on shared visual context. We introduce Multimodal Auto-Completion (MAC), a task that predicts upcoming characters in live chats using partially typed text and visual cues. Unlike traditional text-only auto-completion (TAC), MAC grounds predictions in multimodal context to better capture user intent. To enable this task, we adapt MMDialog and ImageChat to create benchmark datasets. We evaluate leading vision-language models (VLMs) against strong textual baselines, highlighting trade-offs in accuracy and efficiency. We present Router-Suggest, a router framework that dynamically selects between textual models and VLMs based on dialog context, along with a lightweight variant for resource-constrained environments. Router-Suggest achieves a 2.3x to 10x speedup over the best-performing VLM. A user study shows that VLMs significantly excel over textual models on user satisfaction, notably saving user typing effort and improving the quality of completions in multi-turn conversations. These findings underscore the need for multimodal context in auto-completions, leading to smarter, user-aware assistants.
>
---
## 更新

#### [replaced 001] CAST-LUT: Tokenizer-Guided HSV Look-Up Tables for Purple Flare Removal
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06764v2](https://arxiv.org/pdf/2511.06764v2)**

> **作者:** Pu Wang; Shuning Sun; Jialang Lu; Chen Wu; Zhihua Zhang; Youshan Zhang; Chenggang Shan; Dianjie Lu; Guijuan Zhang; Zhuoran Zheng
>
> **摘要:** Purple flare, a diffuse chromatic aberration artifact commonly found around highlight areas, severely degrades the tone transition and color of the image. Existing traditional methods are based on hand-crafted features, which lack flexibility and rely entirely on fixed priors, while the scarcity of paired training data critically hampers deep learning. To address this issue, we propose a novel network built upon decoupled HSV Look-Up Tables (LUTs). The method aims to simplify color correction by adjusting the Hue (H), Saturation (S), and Value (V) components independently. This approach resolves the inherent color coupling problems in traditional methods. Our model adopts a two-stage architecture: First, a Chroma-Aware Spectral Tokenizer (CAST) converts the input image from RGB space to HSV space and independently encodes the Hue (H) and Value (V) channels into a set of semantic tokens describing the Purple flare status; second, the HSV-LUT module takes these tokens as input and dynamically generates independent correction curves (1D-LUTs) for the three channels H, S, and V. To effectively train and validate our model, we built the first large-scale purple flare dataset with diverse scenes. We also proposed new metrics and a loss function specifically designed for this task. Extensive experiments demonstrate that our model not only significantly outperforms existing methods in visual effects but also achieves state-of-the-art performance on all quantitative metrics.
>
---
#### [replaced 002] RobustFormer: Noise-Robust Pre-training for images and videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.13040v2](https://arxiv.org/pdf/2411.13040v2)**

> **作者:** Ashish Bastola; Nishant Luitel; Hao Wang; Danda Pani Paudel; Roshani Poudel; Abolfazl Razi
>
> **备注:** 13 pages
>
> **摘要:** While deep learning-based models like transformers, have revolutionized time-series and vision tasks, they remain highly susceptible to noise and often overfit on noisy patterns rather than robust features. This issue is exacerbated in vision transformers, which rely on pixel-level details that can easily be corrupt. To address this, we leverage the discrete wavelet transform (DWT) for its ability to decompose into multi-resolution layers, isolating noise primarily in the high frequency domain while preserving essential low-frequency information for resilient feature learning. Conventional DWT-based methods, however, struggle with computational inefficiencies due to the requirement for a subsequent inverse discrete wavelet transform (IDWT) step. In this work, we introduce RobustFormer, a novel framework that enables noise-robust masked autoencoder (MAE) pre-training for both images and videos by using DWT for efficient downsampling, eliminating the need for expensive IDWT reconstruction and simplifying the attention mechanism to focus on noise-resilient multi-scale representations. To our knowledge, RobustFormer is the first DWT-based method fully compatible with video inputs and MAE-style pre-training. Extensive experiments on noisy image and video datasets demonstrate that our approach achieves up to 8% increase in Top-1 classification accuracy under severe noise conditions in Imagenet-C and up to 2.7% in Imagenet-P standard benchmarks compared to the baseline and up to 13% higher Top-1 accuracy on UCF-101 under severe custom noise perturbations while maintaining similar accuracy scores for clean datasets. We also observe the reduction of computation complexity by up to 4.4% through IDWT removal compared to VideoMAE baseline without any performance drop.
>
---
#### [replaced 003] SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM与神经渲染交叉任务，旨在解决现有数据集无法涵盖两者挑战的问题。作者构建了SLAM&Render数据集，包含多模态传感器数据和精确运动信息，以评估相关方法。**

- **链接: [https://arxiv.org/pdf/2504.13713v5](https://arxiv.org/pdf/2504.13713v5)**

> **作者:** Samuel Cerezo; Gaetano Meli; Tomás Berriel Martins; Kirill Safronov; Javier Civera
>
> **备注:** 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
>
> **摘要:** Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
>
---
#### [replaced 004] TRec: Learning Hand-Object Interactions through 2D Point Track Motion
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.03667v3](https://arxiv.org/pdf/2601.03667v3)**

> **作者:** Dennis Holzmann; Sven Wachsmuth
>
> **备注:** submitted to ICPR 2026
>
> **摘要:** We present a novel approach for hand-object action recognition that leverages 2D point tracks as an additional motion cue. While most existing methods rely on RGB appearance, human pose estimation, or their combination, our work demonstrates that tracking randomly sampled image points across video frames can substantially improve recognition accuracy. Unlike prior approaches, we do not detect hands, objects, or interaction regions. Instead, we employ CoTracker to follow a set of randomly initialized points through each video and use the resulting trajectories, together with the corresponding image frames, as input to a Transformer-based recognition model. Surprisingly, our method achieves notable gains even when only the initial frame and the point tracks are provided, without incorporating the full video sequence. Experimental results confirm that integrating 2D point tracks consistently enhances performance compared to the same model trained without motion information, highlighting their potential as a lightweight yet effective representation for hand-object action understanding.
>
---
#### [replaced 005] Pyramidal Adaptive Cross-Gating for Multimodal Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18291v2](https://arxiv.org/pdf/2512.18291v2)**

> **作者:** Zidong Gu; Shoufu Tian
>
> **备注:** 17 pages, 6 figures, submitted to Image and Vision Computing
>
> **摘要:** Object detection in aerial imagery is a critical task in applications such as UAV reconnaissance. Although existing methods have extensively explored feature interaction between different modalities, they commonly rely on simple fusion strategies for feature aggregation. This introduces two critical flaws: it is prone to cross-modal noise and disrupts the hierarchical structure of the feature pyramid, thereby impairing the fine-grained detection of small objects. To address this challenge, we propose the Pyramidal Adaptive Cross-Gating Network (PACGNet), an architecture designed to perform deep fusion within the backbone. To this end, we design two core components: the Symmetrical Cross-Gating (SCG) module and the Pyramidal Feature-aware Multimodal Gating (PFMG) module. The SCG module employs a bidirectional, symmetrical "horizontal" gating mechanism to selectively absorb complementary information, suppress noise, and preserve the semantic integrity of each modality. The PFMG module reconstructs the feature hierarchy via a progressive hierarchical gating mechanism. This leverages the detailed features from a preceding, higher-resolution level to guide the fusion at the current, lower-resolution level, effectively preserving fine-grained details as features propagate. Through evaluations conducted on the DroneVehicle and VEDAI datasets, our PACGNet sets a new state-of-the-art benchmark, with mAP50 scores reaching 82.2% and 82.1% respectively.
>
---
#### [replaced 006] LightFormer: A lightweight and efficient decoder for remote sensing image segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.10834v3](https://arxiv.org/pdf/2504.10834v3)**

> **作者:** Sihang Chen; Lijun Yun; Ze Liu; JianFeng Zhu; Jie Chen; Hui Wang; Yueping Nie
>
> **备注:** The manuscript was submitted without obtaining the consent of the other co-authors. We therefore request the withdrawal of the manuscript
>
> **摘要:** Deep learning techniques have achieved remarkable success in the semantic segmentation of remote sensing images and in land-use change detection. Nevertheless, their real-time deployment on edge platforms remains constrained by decoder complexity. Herein, we introduce LightFormer, a lightweight decoder for time-critical tasks that involve unstructured targets, such as disaster assessment, unmanned aerial vehicle search-and-rescue, and cultural heritage monitoring. LightFormer employs a feature-fusion and refinement module built on channel processing and a learnable gating mechanism to aggregate multi-scale, multi-range information efficiently, which drastically curtails model complexity. Furthermore, we propose a spatial information selection module (SISM) that integrates long-range attention with a detail preservation branch to capture spatial dependencies across multiple scales, thereby substantially improving the recognition of unstructured targets in complex scenes. On the ISPRS Vaihingen benchmark, LightFormer attains 99.9% of GLFFNet's mIoU (83.9% vs. 84.0%) while requiring only 14.7% of its FLOPs and 15.9% of its parameters, thus achieving an excellent accuracy-efficiency trade-off. Consistent results on LoveDA, ISPRS Potsdam, RescueNet, and FloodNet further demonstrate its robustness and superior perception of unstructured objects. These findings highlight LightFormer as a practical solution for remote sensing applications where both computational economy and high-precision segmentation are imperative.
>
---
#### [replaced 007] RxnBench: A Multimodal Benchmark for Evaluating Large Language Models on Chemical Reaction Understanding from Scientific Literature
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.23565v3](https://arxiv.org/pdf/2512.23565v3)**

> **作者:** Hanzheng Li; Xi Fang; Yixuan Li; Chaozheng Huang; Junjie Wang; Xi Wang; Hongzhe Bai; Bojun Hao; Shenyu Lin; Huiqi Liang; Linfeng Zhang; Guolin Ke
>
> **摘要:** The integration of Multimodal Large Language Models (MLLMs) into chemistry promises to revolutionize scientific discovery, yet their ability to comprehend the dense, graphical language of reactions within authentic literature remains underexplored. Here, we introduce RxnBench, a multi-tiered benchmark designed to rigorously evaluate MLLMs on chemical reaction understanding from scientific PDFs. RxnBench comprises two tasks: Single-Figure QA (SF-QA), which tests fine-grained visual perception and mechanistic reasoning using 1,525 questions derived from 305 curated reaction schemes, and Full-Document QA (FD-QA), which challenges models to synthesize information from 108 articles, requiring cross-modal integration of text, schemes, and tables. Our evaluation of MLLMs reveals a critical capability gap: while models excel at extracting explicit text, they struggle with deep chemical logic and precise structural recognition. Notably, models with inference-time reasoning significantly outperform standard architectures, yet none achieve 50\% accuracy on FD-QA. These findings underscore the urgent need for domain-specific visual encoders and stronger reasoning engines to advance autonomous AI chemists.
>
---
#### [replaced 008] AtomThink: Multimodal Slow Thinking with Atomic Step Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2411.11930v5](https://arxiv.org/pdf/2411.11930v5)**

> **作者:** Kun Xiang; Zhili Liu; Terry Jingchen Zhang; Yinya Huang; Yunshuang Nie; Kaixin Cai; Yiyang Yin; Runhui Huang; Hanhui Li; Yihan Zeng; Yu-Jie Yuan; Jianhua Han; Lanqing Hong; Hang Xu; Xiaodan Liang
>
> **备注:** TPAMI accepted
>
> **摘要:** In this paper, we address the challenging task of multimodal reasoning by incorporating the notion of ``slow thinking'' into multimodal large language models (MLLMs). Our core idea is that models can learn to adaptively use different levels of reasoning to tackle questions of varying complexity. We propose a novel paradigm of Self-structured Chain of Thought (SCoT), which consists of minimal semantic atomic steps. Unlike existing methods that rely on structured templates or free-form paradigms, our method not only generates flexible CoT structures for various complex tasks but also mitigates the phenomenon of overthinking for easier tasks. To introduce structured reasoning into visual cognition, we design a novel AtomThink framework with four key modules: (i) a data engine to generate high-quality multimodal reasoning paths; (ii) a supervised fine-tuning (SFT) process with serialized inference data; (iii) a policy-guided multi-turn inference method; and (iv) an atomic capability metric to evaluate the single-step utilization rate. Extensive experiments demonstrate that the proposed AtomThink significantly improves the performance of baseline MLLMs, achieving more than 10\% average accuracy gains on MathVista and MathVerse. Compared to state-of-the-art structured CoT approaches, our method not only achieves higher accuracy but also improves data utilization by 5 $\times$ and boosts inference efficiency by 85.3\%. Our code is publicly available at https://github.com/Kun-Xiang/AtomThink.
>
---
#### [replaced 009] ReVision: Refining Video Diffusion with Explicit 3D Motion Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.21855v2](https://arxiv.org/pdf/2504.21855v2)**

> **作者:** Qihao Liu; Ju He; Qihang Yu; Liang-Chieh Chen; Alan Yuille
>
> **备注:** TMLR camera-ready version. Project Page: https://revision-video.github.io/
>
> **摘要:** In recent years, video generation has seen significant advancements. However, challenges still persist in generating complex motions and interactions. To address these challenges, we introduce ReVision, a plug-and-play framework that explicitly integrates parameterized 3D model knowledge into a pretrained conditional video generation model, significantly enhancing its ability to generate high-quality videos with complex motion and interactions. Specifically, ReVision consists of three stages. First, a video diffusion model is used to generate a coarse video. Next, we extract a set of 2D and 3D features from the coarse video to construct a 3D object-centric representation, which is then refined by our proposed parameterized motion prior model to produce an accurate 3D motion sequence. Finally, this refined motion sequence is fed back into the same video diffusion model as additional conditioning, enabling the generation of motion-consistent videos, even in scenarios involving complex actions and interactions. We validate the effectiveness of our approach on Stable Video Diffusion, where ReVision significantly improves motion fidelity and coherence. Remarkably, with only 1.5B parameters, it even outperforms a state-of-the-art video generation model with over 13B parameters on complex video generation by a substantial margin. Our results suggest that, by incorporating 3D motion knowledge, even a relatively small video diffusion model can generate complex motions and interactions with greater realism and controllability, offering a promising solution for physically plausible video generation.
>
---
#### [replaced 010] Higher-Order Domain Generalization in Magnetic Resonance-Based Assessment of Alzheimer's Disease
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.01485v2](https://arxiv.org/pdf/2601.01485v2)**

> **作者:** Zobia Batool; Diala Lteif; Vijaya B. Kolachalama; Huseyin Ozkan; Erchan Aptoula
>
> **摘要:** Despite progress in deep learning for Alzheimer's disease (AD) diagnostics, models trained on structural magnetic resonance imaging (sMRI) often do not perform well when applied to new cohorts due to domain shifts from varying scanners, protocols and patient demographics. AD, the primary driver of dementia, manifests through progressive cognitive and neuroanatomical changes like atrophy and ventricular expansion, making robust, generalizable classification essential for real-world use. While convolutional neural networks and transformers have advanced feature extraction via attention and fusion techniques, single-domain generalization (SDG) remains underexplored yet critical, given the fragmented nature of AD datasets. To bridge this gap, we introduce Extended MixStyle (EM), a framework for blending higher-order feature moments (skewness and kurtosis) to mimic diverse distributional variations. Trained on sMRI data from the National Alzheimer's Coordinating Center (NACC; n=4,647) to differentiate persons with normal cognition (NC) from those with mild cognitive impairment (MCI) or AD and tested on three unseen cohorts (total n=3,126), EM yields enhanced cross-domain performance, improving macro-F1 on average by 2.4 percentage points over state-of-the-art SDG benchmarks, underscoring its promise for invariant, reliable AD detection in heterogeneous real-world settings. The source code will be made available upon acceptance at https://github.com/zobia111/Extended-Mixstyle.
>
---
#### [replaced 011] InnerGS: Internal Scenes Reconstruction and Segmentation via Factorized 3D Gaussian Splatting
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13287v2](https://arxiv.org/pdf/2508.13287v2)**

> **作者:** Shuxin Liang; Yihan Xiao; Wenlu Tang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently gained popularity for efficient scene rendering by representing scenes as explicit sets of anisotropic 3D Gaussians. However, most existing work focuses primarily on modeling external surfaces. In this work, we target the reconstruction of internal scenes, which is crucial for applications that require a deep understanding of an object's interior. By directly modeling a continuous volumetric density through the inner 3D Gaussian distribution, our model effectively reconstructs smooth and detailed internal structures from sparse sliced data. Beyond high-fidelity reconstruction, we further demonstrate the framework's potential for downstream tasks such as segmentation. By integrating language features, we extend our approach to enable text-guided segmentation of medical scenes via natural language queries. Our approach eliminates the need for camera poses, is plug-and-play, and is inherently compatible with any data modalities. We provide cuda implementation at: https://github.com/Shuxin-Liang/InnerGS.
>
---
#### [replaced 012] From See to Shield: ML-Assisted Fine-Grained Access Control for Visual Data
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.19418v2](https://arxiv.org/pdf/2510.19418v2)**

> **作者:** Mete Harun Akcay; Buse Gul Atli; Siddharth Prakash Rao; Alexandros Bakas
>
> **备注:** 11 pages, 4 figures, 6 tables. In submission
>
> **摘要:** As the volume of stored data continues to grow, identifying and protecting sensitive information within large repositories becomes increasingly challenging, especially when shared with multiple users with different roles and permissions. This work presents a system architecture for trusted data sharing with policy-driven access control, enabling selective protection of sensitive regions while maintaining scalability. The proposed architecture integrates four core modules that combine automated detection of sensitive regions, post-correction, key management, and access control. Sensitive regions are secured using a hybrid scheme that employs symmetric encryption for efficiency and Attribute-Based Encryption for policy enforcement. The system supports efficient key distribution and isolates key storage to strengthen overall security. To demonstrate its applicability, we evaluate the system on visual datasets, where Privacy-Sensitive Objects in images are automatically detected, reassessed, and selectively encrypted prior to sharing in a data repository. Experimental results show that our system provides effective PSO detection, increases macro-averaged F1 score (5%) and mean Average Precision (10%), and maintains an average policy-enforced decryption time of less than 1 second per image. These results demonstrate the effectiveness, efficiency and scalability of our proposed solution for fine-grained access control.
>
---
#### [replaced 013] AURASeg: Attention Guided Upsampling with Residual Boundary-Assistive Refinement for Drivable-Area Segmentation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AURASeg，用于道路区域分割任务，解决边界精度不足和特征表示有限的问题，通过引入RBRM和APUD模块提升分割效果。**

- **链接: [https://arxiv.org/pdf/2510.21536v2](https://arxiv.org/pdf/2510.21536v2)**

> **作者:** Narendhiran Vijayakumar; Sridevi. M
>
> **备注:** 6 pages, 4 figures, 4 tables
>
> **摘要:** Free space ground segmentation is essential to navigate autonomous robots, recognize drivable zones, and traverse efficiently. Fine-grained features remain challenging for existing segmentation models, particularly for robots in indoor and structured environments. These difficulties arise from ineffective multi-scale processing, suboptimal boundary refinement, and limited feature representation. To address this, we propose Attention-Guided Upsampling with Residual Boundary-Assistive Refinement (AURASeg), a ground-plane semantic segmentation framework designed to improve border precision while preserving strong region accuracy. Built on a ResNet-50 backbone, AURASeg introduces (i) a Residual Border Refinement Module (RBRM) that enhances edge delineation through boundary-assistive feature refinement, and (ii) Attention Progressive Upsampling Decoder (APUD) blocks that progressively fuse multi-level features during decoding. Additionally, we integrate a (iii) lightweight ASPPLite module to capture multi-scale context with minimal overhead. Extensive experiments on CARL-D, the Ground Mobile Robot Perception (GMRP) dataset, and a custom Gazebo indoor dataset show that AURASeg consistently outperforms strong baselines, with notable gains in boundary metrics. Finally, we demonstrate real-time deployment on a Kobuki TurtleBot, validating practical usability. The code is available at https://github.com/Narendhiranv04/AURASeg
>
---
#### [replaced 014] Sprint: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21986v2](https://arxiv.org/pdf/2510.21986v2)**

> **作者:** Dogyun Park; Moayed Haji-Ali; Yanyu Li; Willi Menapace; Sergey Tulyakov; Hyunwoo J. Kim; Aliaksandr Siarohin; Anil Kag
>
> **摘要:** Diffusion Transformers (DiTs) deliver state-of-the-art generative performance but their quadratic training cost with sequence length makes large-scale pretraining prohibitively expensive. Token dropping can reduce training cost, yet naïve strategies degrade representations, and existing methods are either parameter-heavy or fail at high drop ratios. We present SPRINT, Sparse--Dense Residual Fusion for Efficient Diffusion Transformers, a simple method that enables aggressive token dropping (up to 75%) while preserving quality. SPRINT leverages the complementary roles of shallow and deep layers: early layers process all tokens to capture local detail, deeper layers operate on a sparse subset to cut computation, and their outputs are fused through residual connections. Training follows a two-stage schedule: long masked pre-training for efficiency followed by short full-token fine-tuning to close the train--inference gap. On ImageNet-1K 256x256, SPRINT achieves 9.8x training savings with comparable FID/FDD, and at inference, its Path-Drop Guidance (PDG) nearly halves FLOPs while improving quality. These results establish SPRINT as a simple, effective, and general solution for efficient DiT training.
>
---
#### [replaced 015] Low-Latency Event-Based Velocimetry for Quadrotor Control in a Narrow Pipe
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于无人机控制任务，旨在解决狭窄管道中悬停时的气动干扰问题。通过实时流场测量与神经网络估计扰动，实现闭环控制，提升飞行稳定性。**

- **链接: [https://arxiv.org/pdf/2507.15444v2](https://arxiv.org/pdf/2507.15444v2)**

> **作者:** Leonard Bauersfeld; Davide Scaramuzza
>
> **备注:** 19 pages
>
> **摘要:** Autonomous quadrotor flight in confined spaces such as pipes and tunnels presents significant challenges due to unsteady, self-induced aerodynamic disturbances. Very recent advances have enabled flight in such conditions, but they either rely on constant motion through the pipe to mitigate airflow recirculation effects or suffer from limited stability during hovering. In this work, we present the first closed-loop control system for quadrotors for hovering in narrow pipes that leverages real-time flow field measurements. We develop a low-latency, event-based smoke velocimetry method that estimates local airflow at high temporal resolution. This flow information is used by a disturbance estimator based on a recurrent convolutional neural network, which infers force and torque disturbances in real time. The estimated disturbances are integrated into a learning-based controller trained via reinforcement learning. The flow-feedback control proves particularly effective during lateral translation maneuvers in the pipe cross-section. There, the real-time disturbance information enables the controller to effectively counteract transient aerodynamic effects, thereby preventing collisions with the pipe wall. To the best of our knowledge, this work represents the first demonstration of an aerial robot with closed-loop control informed by real-time flow field measurements. This opens new directions for research on flight in aerodynamically complex environments. In addition, our work also sheds light on the characteristic flow structures that emerge during flight in narrow, circular pipes, providing new insights at the intersection of robotics and fluid dynamics.
>
---
#### [replaced 016] Hallucination Score: Towards Mitigating Hallucinations in Generative Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.14367v2](https://arxiv.org/pdf/2507.14367v2)**

> **作者:** Weiming Ren; Raghav Goyal; Zhiming Hu; Tristan Ty Aumentado-Armstrong; Iqbal Mohomed; Alex Levinshtein
>
> **备注:** 31 pages, 21 figures, and 10 tables
>
> **摘要:** Generative super-resolution (GSR) currently sets the state-of-the-art in terms of perceptual image quality, overcoming the "regression-to-the-mean" blur of prior non-generative models. However, from a human perspective, such models do not fully conform to the optimal balance between quality and fidelity. Instead, a different class of artifacts, in which generated details fail to perceptually match the low resolution image (LRI) or ground-truth image (GTI), is a critical but under-studied issue in GSR, limiting its practical deployment. In this work, we focus on measuring, analyzing, and mitigating these artifacts (i.e., "hallucinations"). We observe that hallucinations are not well-characterized with existing image metrics or quality models, as they are orthogonal to both exact fidelity and no-reference quality. Instead, we take advantage of multimodal large language models (MLLMs) by constructing a prompt that assesses hallucinatory visual elements and generates a "Hallucination Score" (HS). We find that HS is closely aligned with human evaluations, and also provides complementary insights to prior image metrics used for super-resolution (SR) models. Finally, we propose a few efficient HS proxies and demonstrate how diffusion-based GSR models can be fine-tuned to mitigate hallucinations, leveraging HS proxies as differentiable reward functions.
>
---
#### [replaced 017] Controlled Automatic Task-Specific Synthetic Data Generation for Hallucination Detection
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于 hallucination 检测任务，旨在生成高质量合成数据以提升检测效果。通过两步生成-选择流程，生成与真实文本风格一致的 hallucination 数据，增强检测器的泛化能力。**

- **链接: [https://arxiv.org/pdf/2410.12278v2](https://arxiv.org/pdf/2410.12278v2)**

> **作者:** Yong Xie; Karan Aggarwal; Aitzaz Ahmad; Stephen Lau
>
> **备注:** 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (ACM KDD 2024). Accepted by Workshop on Evaluation and Trustworthiness of Generative AI Models
>
> **摘要:** We present a novel approach to automatically generate non-trivial task-specific synthetic datasets for hallucination detection. Our approach features a two-step generation-selection pipeline, using hallucination pattern guidance and a language style alignment during generation. Hallucination pattern guidance leverages the most important task-specific hallucination patterns while language style alignment aligns the style of the synthetic dataset with benchmark text. To obtain robust supervised detectors from synthetic datasets, we also adopt a data mixture strategy to improve performance robustness and generalization. Our results on three datasets show that our generated hallucination text is more closely aligned with non-hallucinated text versus baselines, to train hallucination detectors with better generalization. Our hallucination detectors trained on synthetic datasets outperform in-context-learning (ICL)-based detectors by a large margin of 32%. Our extensive experiments confirm the benefits of our approach with cross-task and cross-generator generalization. Our data-mixture-based training further improves the generalization and robustness of hallucination detection.
>
---
#### [replaced 018] From Preoperative CT to Postmastoidectomy Mesh Construction: Mastoidectomy Shape Prediction for Cochlear Implant Surgery
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.04405v2](https://arxiv.org/pdf/2601.04405v2)**

> **作者:** Yike Zhang; Eduardo Davalos; Dingjie Su; Ange Lou; Jack Noble
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2505.18368
>
> **摘要:** Cochlear Implant (CI) surgery treats severe hearing loss by inserting an electrode array into the cochlea to stimulate the auditory nerve. An important step in this procedure is mastoidectomy, which removes part of the mastoid region of the temporal bone to provide surgical access. Accurate mastoidectomy shape prediction from preoperative imaging improves pre-surgical planning, reduces risks, and enhances surgical outcomes. Despite its importance, there are limited deep-learning-based studies regarding this topic due to the challenges of acquiring ground-truth labels. We address this gap by investigating self-supervised and weakly-supervised learning models to predict the mastoidectomy region without human annotations. We propose a hybrid self-supervised and weakly-supervised learning framework to predict the mastoidectomy region directly from preoperative CT scans, where the mastoid remains intact. Our hybrid method achieves a mean Dice score of 0.72 when predicting the complex and boundary-less mastoidectomy shape, surpassing state-of-the-art approaches and demonstrating strong performance. The method provides groundwork for constructing 3D postmastoidectomy surfaces directly from the corresponding preoperative CT scans. To our knowledge, this is the first work that integrating self-supervised and weakly-supervised learning for mastoidectomy shape prediction, offering a robust and efficient solution for CI surgical planning while leveraging 3D T-distribution loss in weakly-supervised medical imaging.
>
---
#### [replaced 019] Adaptive aggregation of Monte Carlo augmented decomposed filters for efficient group-equivariant convolutional neural network
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2305.10110v4](https://arxiv.org/pdf/2305.10110v4)**

> **作者:** Wenzhao Zhao; Barbara D. Wichtmann; Steffen Albert; Angelika Maurer; Frank G. Zöllner; Jürgen Hesser
>
> **摘要:** Group-equivariant convolutional neural networks (G-CNN) heavily rely on parameter sharing to increase CNN's data efficiency and performance. However, the parameter-sharing strategy greatly increases the computational burden for each added parameter, which hampers its application to deep neural network models. In this paper, we address these problems by proposing a non-parameter-sharing approach for group equivariant neural networks. The proposed methods adaptively aggregate a diverse range of filters by a weighted sum of stochastically augmented decomposed filters. We give theoretical proof about how the group equivariance can be achieved by our methods. Our method applies to both continuous and discrete groups, where the augmentation is implemented using Monte Carlo sampling and bootstrap resampling, respectively. Our methods also serve as an efficient extension of standard CNN. The experiments show that our method outperforms parameter-sharing group equivariant networks and enhances the performance of standard CNNs in image classification and denoising tasks, by using suitable filter bases to build efficient lightweight networks. The code will be available at https://github.com/ZhaoWenzhao/MCG_CNN.
>
---
#### [replaced 020] Video Generation Models Are Good Latent Reward Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21541v3](https://arxiv.org/pdf/2511.21541v3)**

> **作者:** Xiaoyue Mi; Wenqing Yu; Jiesong Lian; Shibo Jie; Ruizhe Zhong; Zijun Liu; Guozhen Zhang; Zixiang Zhou; Zhiyong Xu; Yuan Zhou; Qinglin Lu; Fan Tang
>
> **摘要:** Reward feedback learning (ReFL) has proven effective for aligning image generation with human preferences. However, its extension to video generation faces significant challenges. Existing video reward models rely on vision-language models designed for pixel-space inputs, confining ReFL optimization to near-complete denoising steps after computationally expensive VAE decoding. This pixel-space approach incurs substantial memory overhead and increased training time, and its late-stage optimization lacks early-stage supervision, refining only visual quality rather than fundamental motion dynamics and structural coherence. In this work, we show that pre-trained video generation models are naturally suited for reward modeling in the noisy latent space, as they are explicitly designed to process noisy latent representations at arbitrary timesteps and inherently preserve temporal information through their sequential modeling capabilities. Accordingly, we propose Process Reward Feedback Learning~(PRFL), a framework that conducts preference optimization entirely in latent space, enabling efficient gradient backpropagation throughout the full denoising chain without VAE decoding. Extensive experiments demonstrate that PRFL significantly improves alignment with human preferences, while achieving substantial reductions in memory consumption and training time compared to RGB ReFL.
>
---
#### [replaced 021] ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.20234v5](https://arxiv.org/pdf/2509.20234v5)**

> **作者:** Tom Burgert; Oliver Stoll; Paolo Rota; Begüm Demir
>
> **备注:** Accepted at NeurIPS 2025 (oral)
>
> **摘要:** The hypothesis that Convolutional Neural Networks (CNNs) are inherently texture-biased has shaped much of the discourse on feature use in deep learning. We revisit this hypothesis by examining limitations in the cue-conflict experiment by Geirhos et al. To address these limitations, we propose a domain-agnostic framework that quantifies feature reliance through systematic suppression of shape, texture, and color cues, avoiding the confounds of forced-choice conflicts. By evaluating humans and neural networks under controlled suppression conditions, we find that CNNs are not inherently texture-biased but predominantly rely on local shape features. Nonetheless, this reliance can be substantially mitigated through modern training strategies or architectures (ConvNeXt, ViTs). We further extend the analysis across computer vision, medical imaging, and remote sensing, revealing that reliance patterns differ systematically: computer vision models prioritize shape, medical imaging models emphasize color, and remote sensing models exhibit a stronger reliance on texture. Code is available at https://github.com/tomburgert/feature-reliance.
>
---
#### [replaced 022] PixelArena: A benchmark for Pixel-Precision Visual Intelligence
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.16303v2](https://arxiv.org/pdf/2512.16303v2)**

> **作者:** Feng Liang; Sizhe Cheng; Chenqi Yi; Yong Wang
>
> **备注:** 8 pages, 11 figures, project page: https://pixelarena.reify.ing/project
>
> **摘要:** Omni-modal models that have multimodal input and output are emerging. However, benchmarking their multimodal generation, especially in image generation, is challenging due to the subtleties of human preferences and model biases. Many image generation benchmarks focus on aesthetics instead of the fine-grained generation capabilities of these models, failing to evaluate their visual intelligence with objective metrics. In PixelArena, we propose using semantic segmentation tasks to objectively examine their fine-grained generative intelligence with pixel precision. With our benchmark and experiments, we find the latest Gemini 3 Pro Image has emergent image generation capabilities that generate semantic masks with high fidelity under zero-shot settings, showcasing visual intelligence unseen before and true generalization in new image generation tasks. We further investigate its results, compare them qualitatively and quantitatively with those of other models, and present failure cases. The findings not only signal exciting progress in the field but also provide insights into future research related to dataset development, omni-modal model development, and the design of metrics.
>
---
#### [replaced 023] Transferability of Adversarial Attacks in Video-based MLLMs: A Cross-modal Image-to-Video Approach
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.01042v4](https://arxiv.org/pdf/2501.01042v4)**

> **作者:** Linhao Huang; Xue Jiang; Zhiqiang Wang; Wentao Mo; Xi Xiao; Yong-Jie Yin; Bo Han; Feng Zheng
>
> **摘要:** Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models - a common and practical real-world scenario - remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal large language model (I-MLLM) as a surrogate model to craft adversarial video samples. Multimodal interactions and spatiotemporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. Additionally, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as a surrogate model) achieve competitive performance, with average attack success rate (AASR) of 57.98% on MSVD-QA and 58.26% on MSRVTT-QA for Zero-Shot VideoQA tasks, respectively.
>
---
#### [replaced 024] Reflect3r: Single-View 3D Stereo Reconstruction Aided by Mirror Reflections
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.20607v2](https://arxiv.org/pdf/2509.20607v2)**

> **作者:** Jing Wu; Zirui Wang; Iro Laina; Victor Adrian Prisacariu
>
> **备注:** 3DV 2026. Code and Data Available at https://jingwu2121.github.io/reflect3r/
>
> **摘要:** Mirror reflections are common in everyday environments and can provide stereo information within a single capture, as the real and reflected virtual views are visible simultaneously. We exploit this property by treating the reflection as an auxiliary view and designing a transformation that constructs a physically valid virtual camera, allowing direct pixel-domain generation of the virtual view while adhering to the real-world imaging process. This enables a multi-view stereo setup from a single image, simplifying the imaging process, making it compatible with powerful feed-forward reconstruction models for generalizable and robust 3D reconstruction. To further exploit the geometric symmetry introduced by mirrors, we propose a symmetric-aware loss to refine pose estimation. Our framework also naturally extends to dynamic scenes, where each frame contains a mirror reflection, enabling efficient per-frame geometry recovery. For quantitative evaluation, we provide a fully customizable synthetic dataset of 16 Blender scenes, each with ground-truth point clouds and camera poses. Extensive experiments on real-world data and synthetic data are conducted to illustrate the effectiveness of our method.
>
---
#### [replaced 025] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.09527v2](https://arxiv.org/pdf/2503.09527v2)**

> **作者:** Peng Chen; Pi Bu; Yingyao Wang; Xinyi Wang; Ziming Wang; Jie Guo; Yingxiu Zhao; Qi Zhu; Jun Song; Siran Yang; Jiamang Wang; Bo Zheng
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent advances in Vision-Language-Action models (VLAs) have expanded the capabilities of embodied intelligence. However, significant challenges remain in real-time decision-making in complex 3D environments, which demand second-level responses, high-resolution perception, and tactical reasoning under dynamic conditions. To advance the field, we introduce CombatVLA, an efficient VLA model optimized for combat tasks in 3D action role-playing games(ARPGs). Specifically, our CombatVLA is a 3B model trained on video-action pairs collected by an action tracker, where the data is formatted as action-of-thought (AoT) sequences. Thereafter, CombatVLA seamlessly integrates into an action execution framework, allowing efficient inference through our truncated AoT strategy. Experimental results demonstrate that CombatVLA not only outperforms all existing models on the combat understanding benchmark but also achieves a 50-fold acceleration in game combat. Moreover, it has a higher task success rate than human players. We will open-source all resources, including the action tracker, dataset, benchmark, model weights, training code, and the implementation of the framework at https://combatvla.github.io/.
>
---
#### [replaced 026] Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12766v3](https://arxiv.org/pdf/2506.12766v3)**

> **作者:** Ruojing Li; Wei An; Yingqian Wang; Xinyi Ying; Yimian Dai; Longguang Wang; Miao Li; Yulan Guo; Li Liu
>
> **摘要:** Infrared small target (IRST) detection is challenging in simultaneously achieving precise, robust, and efficient performance due to extremely dim targets and strong interference. Current learning-based methods attempt to leverage ``more" information from both the spatial and the short-term temporal domains, but suffer from unreliable performance under complex conditions while incurring computational redundancy. In this paper, we explore the ``more essential" information from a more crucial domain for the detection. Through theoretical analysis, we reveal that the global temporal saliency and correlation information in the temporal profile demonstrate significant superiority in distinguishing target signals from other signals. To investigate whether such superiority is preferentially leveraged by well-trained networks, we built the first prediction attribution tool in this field and verified the importance of the temporal profile information. Inspired by the above conclusions, we remodel the IRST detection task as a one-dimensional signal anomaly detection task, and propose an efficient deep temporal probe network (DeepPro) that only performs calculations in the time dimension for IRST detection. We conducted extensive experiments to fully validate the effectiveness of our method. The experimental results are exciting, as our DeepPro outperforms existing state-of-the-art IRST detection methods on widely-used benchmarks with extremely high efficiency, and achieves a significant improvement on dim targets and in complex scenarios. We provide a new modeling domain, a new insight, a new method, and a new performance, which can promote the development of IRST detection. Codes are available at https://github.com/TinaLRJ/DeepPro.
>
---
#### [replaced 027] Efficient Bayesian Computation Using Plug-and-Play Priors for Poisson Inverse Problems
- **分类: stat.CO; cs.CV; math.NA; stat.ML**

- **链接: [https://arxiv.org/pdf/2503.16222v2](https://arxiv.org/pdf/2503.16222v2)**

> **作者:** Teresa Klatzer; Savvas Melidonis; Marcelo Pereyra; Konstantinos C. Zygalakis
>
> **备注:** 35 pages, 19 figures
>
> **摘要:** This paper studies plug-and-play (PnP) Langevin sampling strategies for Bayesian inference in low-photon Poisson imaging problems, a challenging class of problems with significant applications in astronomy, medicine, and biology. PnP Langevin sampling offers a powerful framework for Bayesian image restoration, enabling accurate point estimation as well as advanced inference tasks, including uncertainty quantification and visualization analyses, and empirical Bayesian inference for automatic model parameter tuning. Herein, we leverage and adapt recent developments in this framework to tackle challenging imaging problems involving weakly informative Poisson data. Existing PnP Langevin algorithms are not well-suited for low-photon Poisson imaging due to high solution uncertainty and poor regularity properties, such as exploding gradients and non-negativity constraints. To address these challenges, we explore two strategies for extending Langevin PnP sampling to Poisson imaging models: (i) an accelerated PnP Langevin method that incorporates boundary reflections and a Poisson likelihood approximation and (ii) a mirror sampling algorithm that leverages a Riemannian geometry to handle the constraints and the poor regularity of the likelihood without approximations. The effectiveness of these approaches is evaluated and contrasted through extensive numerical experiments and comparisons with state-of-the-art methods. The source code accompanying this paper is available at https://github.com/freyyia/pnp-langevin-poisson.
>
---
#### [replaced 028] Causality-Aware Temporal Projection for Video Understanding in Video-LLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.01804v2](https://arxiv.org/pdf/2601.01804v2)**

> **作者:** Zhengjian Kang; Qi Chen; Rui Liu; Kangtong Mo; Xingyu Zhang; Xiaoyu Deng; Ye Zhang
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Recent Video Large Language Models (Video-LLMs) have shown strong multimodal reasoning capabilities, yet remain challenged by video understanding tasks that require consistent temporal ordering and causal coherence. Many parameter-efficient Video-LLMs rely on unconstrained bidirectional projectors to model inter-frame interactions, which can blur temporal ordering by allowing later frames to influence earlier representations, without explicit architectural mechanisms to respect the directional nature of video reasoning. To address this limitation, we propose V-CORE, a parameter-efficient framework that introduces explicit temporal ordering constraints for video understanding. V-CORE consists of two key components: (1) Learnable Spatial Aggregation (LSA), which adaptively selects salient spatial tokens to reduce redundancy, and (2) a Causality-Aware Temporal Projector (CATP), which enforces structured unidirectional information flow via block-causal attention and a terminal dynamic summary token acting as a causal sink. This design preserves intra-frame spatial interactions while ensuring that temporal information is aggregated in a strictly ordered manner. With 4-bit QLoRA and a frozen LLM backbone, V-CORE can be trained efficiently on a single consumer GPU. Experiments show that V-CORE achieves strong performance on the challenging NExT-QA benchmark, reaching 61.2% accuracy, and remains competitive across MSVD-QA, MSRVTT-QA, and TGIF-QA, with gains concentrated in temporal and causal reasoning subcategories (+3.5% and +5.2% respectively), directly validating the importance of explicit temporal ordering constraints.
>
---
#### [replaced 029] Language as Prior, Vision as Calibration: Metric Scale Recovery for Monocular Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.01457v3](https://arxiv.org/pdf/2601.01457v3)**

> **作者:** Mingxia Zhan; Li Zhang; Beibei Wang; Yingjie Wang; Zenglin Shi
>
> **摘要:** Relative-depth foundation models transfer well, yet monocular metric depth remains ill-posed due to unidentifiable global scale and heightened domain-shift sensitivity. Under a frozen-backbone calibration setting, we recover metric depth via an image-specific affine transform in inverse depth and train only lightweight calibration heads while keeping the relative-depth backbone and the CLIP text encoder fixed. Since captions provide coarse but noisy scale cues that vary with phrasing and missing objects, we use language to predict an uncertainty-aware envelope that bounds feasible calibration parameters in an unconstrained space, rather than committing to a text-only point estimate. We then use pooled multi-scale frozen visual features to select an image-specific calibration within this envelope. During training, a closed-form least-squares oracle in inverse depth provides per-image supervision for learning the envelope and the selected calibration. Experiments on NYUv2 and KITTI improve in-domain accuracy, while zero-shot transfer to SUN-RGBD and DDAD demonstrates improved robustness over strong language-only baselines.
>
---
#### [replaced 030] seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.03176v3](https://arxiv.org/pdf/2505.03176v3)**

> **作者:** Hafez Ghaemi; Eilif Muller; Shahab Bakhtiari
>
> **摘要:** Joint-embedding self-supervised learning (SSL) commonly relies on transformations such as data augmentation and masking to learn visual representations, a task achieved by enforcing invariance or equivariance with respect to these transformations applied to two views of an image. This dominant two-view paradigm in SSL often limits the flexibility of learned representations for downstream adaptation by creating performance trade-offs between high-level invariance-demanding tasks such as image classification and more fine-grained equivariance-related tasks. In this work, we propose \emph{seq-JEPA}, a world modeling framework that introduces architectural inductive biases into joint-embedding predictive architectures to resolve this trade-off. Without relying on dual equivariance predictors or loss terms, seq-JEPA simultaneously learns two architecturally separate representations for equivariance- and invariance-demanding tasks. To do so, our model processes short sequences of different views (observations) of inputs. Each encoded view is concatenated with an embedding of the relative transformation (action) that produces the next observation in the sequence. These view-action pairs are passed through a transformer encoder that outputs an aggregate representation. A predictor head then conditions this aggregate representation on the upcoming action to predict the representation of the next observation. Empirically, seq-JEPA demonstrates strong performance on both equivariance- and invariance-demanding downstream tasks without sacrificing one for the other. Furthermore, it excels at tasks that inherently require aggregating a sequence of observations, such as path integration across actions and predictive learning across eye movements.
>
---
#### [replaced 031] PsOCR: Benchmarking Large Multimodal Models for Optical Character Recognition in Low-resource Pashto Language
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.10055v2](https://arxiv.org/pdf/2505.10055v2)**

> **作者:** Ijazul Haq; Yingjie Zhang; Irfan Ali Khan
>
> **摘要:** This paper evaluates the performance of Large Multimodal Models (LMMs) on Optical Character Recognition (OCR) in the low-resource Pashto language. Natural Language Processing (NLP) in Pashto faces several challenges due to the cursive nature of its script and a scarcity of structured datasets. To address this, we developed a synthetic Pashto OCR dataset, PsOCR, consisting of one million images annotated with bounding boxes at word, line, and document levels, suitable for training and evaluating models based on different architectures, including Convolutional Neural Networks (CNNs) and Transformers. PsOCR covers variations across 1,000 unique font families, colors, image sizes, and layouts. A benchmark subset of 10K images was selected to evaluate the performance of several LMMs, including seven open-source models: DeepSeek's Janus, InternVL, MiniCPM, Florence, and Qwen (3B and 7B), and four closed-source models: GPT-4o, Gemini, Claude, and Grok. Experimental results demonstrate that Gemini achieves the best performance among all models, whereas among open-source models, Qwen-7B stands out. This work provides an insightful assessment of the capabilities and limitations of current LMMs for OCR tasks in Pashto and establishes a foundation for further research not only in Pashto OCR but also for other similar scripts such as Arabic, Persian, and Urdu. PsOCR is available at https://github.com/zirak-ai/PashtoOCR.
>
---
#### [replaced 032] 3D-WAG: Hierarchical Wavelet-Guided Autoregressive Generation for High-Fidelity 3D Shapes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.19037v3](https://arxiv.org/pdf/2411.19037v3)**

> **作者:** Tejaswini Medi; Arianna Rampini; Pradyumna Reddy; Pradeep Kumar Jayaraman; Margret Keuper
>
> **摘要:** Autoregressive (AR) models have achieved remarkable success in natural language and image generation, but their application to 3D shape modeling remains largely unexplored. Unlike diffusion models, AR models enable more efficient and controllable generation with faster inference times, making them especially suitable for data-intensive domains. Traditional 3D generative models using AR approaches often rely on ``next-token" predictions at the voxel or point level. While effective for certain applications, these methods can be restrictive and computationally expensive when dealing with large-scale 3D data. To tackle these challenges, we introduce 3D-WAG, an AR model for 3D implicit distance fields that can perform unconditional shape generation, class-conditioned and also text-conditioned shape generation. Our key idea is to encode shapes as multi-scale wavelet token maps and use a Transformer to predict the ``next higher-resolution token map" in an autoregressive manner. By redefining 3D AR generation task as ``next-scale" prediction, we reduce the computational cost of generation compared to traditional ``next-token" prediction models, while preserving essential geometric details of 3D shapes in a more structured and hierarchical manner. We evaluate 3D-WAG to showcase its benefit by quantitative and qualitative comparisons with state-of-the-art methods on widely used benchmarks. Our results show 3D-WAG achieves superior performance in key metrics like Coverage and MMD, generating high-fidelity 3D shapes that closely match the real data distribution.
>
---
#### [replaced 033] Infrared-Assisted Single-Stage Framework for Joint Restoration and Fusion of Visible and Infrared Images under Hazy Conditions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.12586v3](https://arxiv.org/pdf/2411.12586v3)**

> **作者:** Huafeng Li; Jiaqi Fang; Yafei Zhang; Yu Liu
>
> **备注:** Accepted by Pattern Recognition
>
> **摘要:** Infrared and visible (IR-VIS) image fusion has gained significant attention for its broad application value. However, existing methods often neglect the complementary role of infrared image in restoring visible image features under hazy conditions. To address this, we propose a joint learning framework that utilizes infrared image for the restoration and fusion of hazy IR-VIS images. To mitigate the adverse effects of feature diversity between IR-VIS images, we introduce a prompt generation mechanism that regulates modality-specific feature incompatibility. This creates a prompt selection matrix from non-shared image information, followed by prompt embeddings generated from a prompt pool. These embeddings help generate candidate features for dehazing. We further design an infrared-assisted feature restoration mechanism that selects candidate features based on haze density, enabling simultaneous restoration and fusion within a single-stage framework. To enhance fusion quality, we construct a multi-stage prompt embedding fusion module that leverages feature supplementation from the prompt generation module. Our method effectively fuses IR-VIS images while removing haze, yielding clear, haze-free fusion results. In contrast to two-stage methods that dehaze and then fuse, our approach enables collaborative training in a single-stage framework, making the model relatively lightweight and suitable for practical deployment. Experimental results validate its effectiveness and demonstrate advantages over existing methods. The source code of the paper is available at \href{https://github.com/fangjiaqi0909/IASSF}{\textcolor{blue}{https://github.com/fangjiaqi0909/IASSF
>
---
#### [replaced 034] ThinkRL-Edit: Thinking in Reinforcement Learning for Reasoning-Centric Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.03467v2](https://arxiv.org/pdf/2601.03467v2)**

> **作者:** Hengjia Li; Liming Jiang; Qing Yan; Yizhi Song; Hao Kang; Zichuan Liu; Xin Lu; Boxi Wu; Deng Cai
>
> **摘要:** Instruction-driven image editing with unified multimodal generative models has advanced rapidly, yet their underlying visual reasoning remains limited, leading to suboptimal performance on reasoning-centric edits. Reinforcement learning (RL) has been investigated for improving the quality of image editing, but it faces three key challenges: (1) limited reasoning exploration confined to denoising stochasticity, (2) biased reward fusion, and (3) unstable VLM-based instruction rewards. In this work, we propose ThinkRL-Edit, a reasoning-centric RL framework that decouples visual reasoning from image synthesis and expands reasoning exploration beyond denoising. To the end, we introduce Chain-of-Thought (CoT)-based reasoning sampling with planning and reflection stages prior to generation in online sampling, compelling the model to explore multiple semantic hypotheses and validate their plausibility before committing to a visual outcome. To avoid the failures of weighted aggregation, we propose an unbiased chain preference grouping strategy across multiple reward dimensions. Moreover, we replace interval-based VLM scores with a binary checklist, yielding more precise, lower-variance, and interpretable rewards for complex reasoning. Experiments show our method significantly outperforms prior work on reasoning-centric image editing, producing instruction-faithful, visually coherent, and semantically grounded edits.
>
---
#### [replaced 035] Grasp the Graph (GtG) 2.0: Ensemble of Graph Neural Networks for High-Precision Grasp Pose Detection in Clutter
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人抓取任务，解决杂乱环境中高精度抓取位姿检测问题。提出GtG 2.0框架，利用图神经网络集成方法提升抓取性能。**

- **链接: [https://arxiv.org/pdf/2505.02664v2](https://arxiv.org/pdf/2505.02664v2)**

> **作者:** Ali Rashidi Moghadam; Sayedmohammadreza Rastegari; Mehdi Tale Masouleh; Ahmad Kalhor
>
> **备注:** 20 pages
>
> **摘要:** Grasp pose detection in cluttered, real-world environments remains a significant challenge due to noisy and incomplete sensory data combined with complex object geometries. This paper introduces Grasp the Graph 2.0 (GtG 2.0) method, a lightweight yet highly effective hypothesis-and-test robotics grasping framework which leverages an ensemble of Graph Neural Networks for efficient geometric reasoning from point cloud data. Building on the success of GtG 1.0, which demonstrated the potential of Graph Neural Networks for grasp detection but was limited by assumptions of complete, noise-free point clouds and 4-Dof grasping, GtG 2.0 employs a conventional Grasp Pose Generator to efficiently produce 7-Dof grasp candidates. Candidates are assessed with an ensemble Graph Neural Network model which includes points within the gripper jaws (inside points) and surrounding contextual points (outside points). This improved representation boosts grasp detection performance over previous methods using the same generator. GtG 2.0 shows up to a 35% improvement in Average Precision on the GraspNet-1Billion benchmark compared to hypothesis-and-test and Graph Neural Network-based methods, ranking it among the top three frameworks. Experiments with a 3-Dof Delta Parallel robot and Kinect-v1 camera show a success rate of 91% and a clutter completion rate of 100%, demonstrating its flexibility and reliability.
>
---
#### [replaced 036] Normalized Conditional Mutual Information Surrogate Loss for Deep Neural Classifiers
- **分类: cs.LG; cs.AI; cs.CV; cs.IT**

- **链接: [https://arxiv.org/pdf/2601.02543v2](https://arxiv.org/pdf/2601.02543v2)**

> **作者:** Linfeng Ye; Zhixiang Chi; Konstantinos N. Plataniotis; En-hui Yang
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** In this paper, we propose a novel information theoretic surrogate loss; normalized conditional mutual information (NCMI); as a drop in alternative to the de facto cross-entropy (CE) for training deep neural network (DNN) based classifiers. We first observe that the model's NCMI is inversely proportional to its accuracy. Building on this insight, we introduce an alternating algorithm to efficiently minimize the NCMI. Across image recognition and whole-slide imaging (WSI) subtyping benchmarks, NCMI-trained models surpass state of the art losses by substantial margins at a computational cost comparable to that of CE. Notably, on ImageNet, NCMI yields a 2.77% top-1 accuracy improvement with ResNet-50 comparing to the CE; on CAMELYON-17, replacing CE with NCMI improves the macro-F1 by 8.6% over the strongest baseline. Gains are consistent across various architectures and batch sizes, suggesting that NCMI is a practical and competitive alternative to CE.
>
---
#### [replaced 037] Dense 3D Displacement Estimation for Landslide Monitoring via Fusion of TLS Point Clouds and Embedded RGB Images
- **分类: cs.CV; cs.RO; eess.IV; physics.geo-ph**

- **简介: 该论文属于滑坡监测任务，解决传统方法难以获得高精度3D位移估计的问题，通过融合TLS点云与RGB图像，提出一种分层粗到细的位移估计方法。**

- **链接: [https://arxiv.org/pdf/2506.16265v2](https://arxiv.org/pdf/2506.16265v2)**

> **作者:** Zhaoyi Wang; Jemil Avers Butt; Shengyu Huang; Tomislav Medic; Andreas Wieser
>
> **备注:** Published in the International Journal of Applied Earth Observation and Geoinformation. 25 pages, 19 figures
>
> **摘要:** Landslide monitoring is essential for understanding geohazards and mitigating associated risks. Existing point cloud-based methods, however, typically rely on either geometric or radiometric information and often yield sparse or non-3D displacement estimates. In this paper, we propose a hierarchical partitioning-based coarse-to-fine approach that integrates 3D point clouds and co-registered RGB images to estimate dense 3D displacement vector fields. Patch-level matches are constructed using both 3D geometry and 2D image features, refined via geometric consistency checks, and followed by rigid transformation estimation per match. Experimental results on two real-world landslide datasets demonstrate that the proposed method produces 3D displacement estimates with high spatial coverage (79% and 97%) and accuracy. Deviations in displacement magnitude with respect to external measurements (total station or GNSS observations) are 0.15 m and 0.25 m on the two datasets, respectively, and only 0.07 m and 0.20 m compared to manually derived references, all below the mean scan resolutions (0.08 m and 0.30 m). Compared with the state-of-the-art method F2S3, the proposed approach improves spatial coverage while maintaining comparable accuracy. The proposed approach offers a practical and adaptable solution for TLS-based landslide monitoring and is extensible to other types of point clouds and monitoring tasks. The example data and source code are publicly available at https://github.com/gseg-ethz/fusion4landslide.
>
---
#### [replaced 038] Co-Training Vision Language Models for Remote Sensing Multi-task Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21272v2](https://arxiv.org/pdf/2511.21272v2)**

> **作者:** Qingyun Li; Shuran Ma; Junwei Luo; Yi Yu; Yue Zhou; Fengxiang Wang; Xudong Lu; Xiaoxing Wang; Xin He; Yushi Chen; Xue Yang
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** With Transformers achieving outstanding performance on individual remote sensing (RS) tasks, we are now approaching the realization of a unified model that excels across multiple tasks through multi-task learning (MTL). Compared to single-task approaches, MTL methods offer improved generalization, enhanced scalability, and greater practical applicability. Recently, vision language models (VLMs) have achieved promising results in RS image understanding, grounding, and ultra-high-resolution (UHR) image reasoning, respectively. Moreover, the unified text-based interface demonstrates significant potential for MTL. Hence, in this work, we present RSCoVLM, a simple yet flexible VLM baseline for RS MTL. Firstly, we create the data curation engine, including data acquisition, offline processing and integrating, as well as online loading and weighting. This data engine effectively addresses complex RS data enviroment and generates flexible vision-language conversations. Furthermore, we propose a unified dynamic-resolution strategy to address the diverse image scales inherent in RS imagery. For UHR images, we introduce the Zoom-in Chain mechanism together with its corresponding dataset, LRS-VQA-Zoom. The strategies are flexible and effectively mitigate the computational burdens. Additionally, we significantly enhance the model's object detection capability and propose a novel evaluation protocol that ensures fair comparison between VLMs and conventional detection models. Extensive experiments demonstrate that RSCoVLM achieves state-of-the-art performance across diverse tasks, outperforming existing RS VLMs and even rivaling specialized expert models. All the training and evaluating tools, model weights, and datasets have been fully open-sourced to support reproducibility. We expect that this baseline will promote further progress toward general-purpose RS models.
>
---
#### [replaced 039] A Novel Patch-Based TDA Approach for Computed Tomography
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.12108v2](https://arxiv.org/pdf/2512.12108v2)**

> **作者:** Dashti A. Ali; Aras T. Asaad; Jacob J. Peoples; Mohammad Hamghalam; Alex Robins; Mane Piliposyan; Richard K. G. Do; Natalie Gangai; Yun S. Chun; Ahmad Bashir Barekzai; Jayasree Chakraborty; Hala Khasawneh; Camila Vilela; Natally Horvat; João Miranda; Alice C. Wei; Amber L. Simpson
>
> **摘要:** The development of machine learning (ML) models based on computed tomography (CT) imaging modality has been a major focus of recent research in the medical imaging domain. Incorporating robust feature engineering approach can highly improve the performance of these models. Topological data analysis (TDA), a recent development based on the mathematical field of algebraic topology, mainly focuses on the data from a topological perspective, extracting deeper insight and higher dimensional structures from the data. Persistent homology (PH), a fundamental tool in the area of TDA, can extract topological features such as connected components, cycles and voids from the data. A popular approach to construct PH from 3D CT images is to utilize the 3D cubical complex filtration, a method adapted for grid-structured data. However, this approach may not always yield the best performance and can suffer from computational complexity with higher resolution CT images. This study introduces a novel patch-based PH construction approach tailored for volumetric medical imaging data, in particular CT modality. A wide range of experiments has been conducted on several datasets of 3D CT images to comprehensively analyze the performance of the proposed method with various parameters and benchmark it against the 3D cubical complex algorithm. Our results highlight the dominance of the patch-based TDA approach in terms of both classification performance and time-efficiency. The proposed approach outperformed the cubical complex method, achieving average improvement of 10.38%, 6.94%, 2.06%, 11.58%, and 8.51% in accuracy, AUC, sensitivity, specificity, and F1 score, respectively, across all datasets. Finally, we provide a convenient python package, Patch-TDA, to facilitate the utilization of the proposed approach.
>
---
#### [replaced 040] Multimodal Interpretation of Remote Sensing Images: Dynamic Resolution Input Strategy and Multi-scale Vision-Language Alignment Mechanism
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23243v2](https://arxiv.org/pdf/2512.23243v2)**

> **作者:** Siyu Zhang; Lianlei Shan; Runhe Qiu
>
> **摘要:** Multimodal fusion of remote sensing images serves as a core technology for overcoming the limitations of single-source data and improving the accuracy of surface information extraction, which exhibits significant application value in fields such as environmental monitoring and urban planning. To address the deficiencies of existing methods, including the failure of fixed resolutions to balance efficiency and detail, as well as the lack of semantic hierarchy in single-scale alignment, this study proposes a Vision-language Model (VLM) framework integrated with two key innovations: the Dynamic Resolution Input Strategy (DRIS) and the Multi-scale Vision-language Alignment Mechanism (MS-VLAM).Specifically, the DRIS adopts a coarse-to-fine approach to adaptively allocate computational resources according to the complexity of image content, thereby preserving key fine-grained features while reducing redundant computational overhead. The MS-VLAM constructs a three-tier alignment mechanism covering object, local-region and global levels, which systematically captures cross-modal semantic consistency and alleviates issues of semantic misalignment and granularity imbalance.Experimental results on the RS-GPT4V dataset demonstrate that the proposed framework significantly improves the accuracy of semantic understanding and computational efficiency in tasks including image captioning and cross-modal retrieval. Compared with conventional methods, it achieves superior performance in evaluation metrics such as BLEU-4 and CIDEr for image captioning, as well as R@10 for cross-modal retrieval. This technical framework provides a novel approach for constructing efficient and robust multimodal remote sensing systems, laying a theoretical foundation and offering technical guidance for the engineering application of intelligent remote sensing interpretation.
>
---
#### [replaced 041] e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出e5-omni，解决多模态嵌入中的对齐问题，通过显式对齐提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.03666v2](https://arxiv.org/pdf/2601.03666v2)**

> **作者:** Haonan Chen; Sicheng Gao; Radu Timofte; Tetsuya Sakai; Zhicheng Dou
>
> **备注:** https://huggingface.co/Haon-Chen/e5-omni-7B
>
> **摘要:** Modern information systems often involve different types of items, e.g., a text query, an image, a video clip, or an audio segment. This motivates omni-modal embedding models that map heterogeneous modalities into a shared space for direct comparison. However, most recent omni-modal embeddings still rely heavily on implicit alignment inherited from pretrained vision-language model (VLM) backbones. In practice, this causes three common issues: (i) similarity logits have modality-dependent sharpness, so scores are not on a consistent scale; (ii) in-batch negatives become less effective over time because mixed-modality batches create an imbalanced hardness distribution; as a result, many negatives quickly become trivial and contribute little gradient; and (iii) embeddings across modalities show mismatched first- and second-order statistics, which makes rankings less stable. To tackle these problems, we propose e5-omni, a lightweight explicit alignment recipe that adapts off-the-shelf VLMs into robust omni-modal embedding models. e5-omni combines three simple components: (1) modality-aware temperature calibration to align similarity scales, (2) a controllable negative curriculum with debiasing to focus on confusing negatives while reducing the impact of false negatives, and (3) batch whitening with covariance regularization to better match cross-modal geometry in the shared embedding space. Experiments on MMEB-V2 and AudioCaps show consistent gains over strong bi-modal and omni-modal baselines, and the same recipe also transfers well to other VLM backbones. We release our model checkpoint at https://huggingface.co/Haon-Chen/e5-omni-7B.
>
---
#### [replaced 042] 360DVO: Deep Visual Odometry for Monocular 360-Degree Camera
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.02309v2](https://arxiv.org/pdf/2601.02309v2)**

> **作者:** Xiaopeng Guo; Yinzhe Xu; Huajian Huang; Sai-Kit Yeung
>
> **备注:** 12 pages. Received by RA-L
>
> **摘要:** Monocular omnidirectional visual odometry (OVO) systems leverage 360-degree cameras to overcome field-of-view limitations of perspective VO systems. However, existing methods, reliant on handcrafted features or photometric objectives, often lack robustness in challenging scenarios, such as aggressive motion and varying illumination. To address this, we present 360DVO, the first deep learning-based OVO framework. Our approach introduces a distortion-aware spherical feature extractor (DAS-Feat) that adaptively learns distortion-resistant features from 360-degree images. These sparse feature patches are then used to establish constraints for effective pose estimation within a novel omnidirectional differentiable bundle adjustment (ODBA) module. To facilitate evaluation in realistic settings, we also contribute a new real-world OVO benchmark. Extensive experiments on this benchmark and public synthetic datasets (TartanAir V2 and 360VO) demonstrate that 360DVO surpasses state-of-the-art baselines (including 360VO and OpenVSLAM), improving robustness by 50% and accuracy by 37.5%. Homepage: https://chris1004336379.github.io/360DVO-homepage
>
---
#### [replaced 043] Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于文档理解任务，旨在解决传统方法在结构丢失和上下文建模上的不足，通过多模态检索增强生成技术实现更全面的文档智能。**

- **链接: [https://arxiv.org/pdf/2510.15253v2](https://arxiv.org/pdf/2510.15253v2)**

> **作者:** Sensen Gao; Shanshan Zhao; Xu Jiang; Lunhao Duan; Yong Xien Chng; Qing-Guo Chen; Weihua Luo; Kaifu Zhang; Jia-Wang Bian; Mingming Gong
>
> **摘要:** Document understanding is critical for applications from financial analysis to scientific discovery. Current approaches, whether OCR-based pipelines feeding Large Language Models (LLMs) or native Multimodal LLMs (MLLMs), face key limitations: the former loses structural detail, while the latter struggles with context modeling. Retrieval-Augmented Generation (RAG) helps ground models in external data, but documents' multimodal nature, i.e., combining text, tables, charts, and layout, demands a more advanced paradigm: Multimodal RAG. This approach enables holistic retrieval and reasoning across all modalities, unlocking comprehensive document intelligence. Recognizing its importance, this paper presents a systematic survey of Multimodal RAG for document understanding. We propose a taxonomy based on domain, retrieval modality, and granularity, and review advances involving graph structures and agentic frameworks. We also summarize key datasets, benchmarks, applications and industry deployment, and highlight open challenges in efficiency, fine-grained representation, and robustness, providing a roadmap for future progress in document AI.
>
---
#### [replaced 044] CoV: Chain-of-View Prompting for Spatial Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.05172v2](https://arxiv.org/pdf/2601.05172v2)**

> **作者:** Haoyu Zhao; Akide Liu; Zeyu Zhang; Weijie Wang; Feng Chen; Ruihan Zhu; Gholamreza Haffari; Bohan Zhuang
>
> **备注:** Code link https://github.com/ziplab/CoV
>
> **摘要:** Embodied question answering (EQA) in 3D environments often requires collecting context that is distributed across multiple viewpoints and partially occluded. However, most recent vision--language models (VLMs) are constrained to a fixed and finite set of input views, which limits their ability to acquire question-relevant context at inference time and hinders complex spatial reasoning. We propose Chain-of-View (CoV) prompting, a training-free, test-time reasoning framework that transforms a VLM into an active viewpoint reasoner through a coarse-to-fine exploration process. CoV first employs a View Selection agent to filter redundant frames and identify question-aligned anchor views. It then performs fine-grained view adjustment by interleaving iterative reasoning with discrete camera actions, obtaining new observations from the underlying 3D scene representation until sufficient context is gathered or a step budget is reached. We evaluate CoV on OpenEQA across four mainstream VLMs and obtain an average +11.56% improvement in LLM-Match, with a maximum gain of +13.62% on Qwen3-VL-Flash. CoV further exhibits test-time scaling: increasing the minimum action budget yields an additional +2.51% average improvement, peaking at +3.73% on Gemini-2.5-Flash. On ScanQA and SQA3D, CoV delivers strong performance (e.g., 116 CIDEr / 31.9 EM@1 on ScanQA and 51.1 EM@1 on SQA3D). Overall, these results suggest that question-aligned view selection coupled with open-view search is an effective, model-agnostic strategy for improving spatial reasoning in 3D EQA without additional training. Code is available on https://github.com/ziplab/CoV .
>
---
#### [replaced 045] Subject-driven Video Generation via Disentangled Identity and Motion
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2504.17816v2](https://arxiv.org/pdf/2504.17816v2)**

> **作者:** Daneul Kim; Jingxu Zhang; Wonjoon Jin; Sunghyun Cho; Qi Dai; Jaesik Park; Chong Luo
>
> **备注:** [v2 updated] Project Page : https://carpedkm.github.io/projects/disentangled_sub/index.html
>
> **摘要:** We propose to train a subject-driven customized video generation model through decoupling the subject-specific learning from temporal dynamics in zero-shot without additional tuning. A traditional method for video customization that is tuning-free often relies on large, annotated video datasets, which are computationally expensive and require extensive annotation. In contrast to the previous approach, we introduce the use of an image customization dataset directly on training video customization models, factorizing the video customization into two folds: (1) identity injection through image customization dataset and (2) temporal modeling preservation with a small set of unannotated videos through the image-to-video training method. Additionally, we employ random image token dropping with randomized image initialization during image-to-video fine-tuning to mitigate the copy-and-paste issue. To further enhance learning, we introduce stochastic switching during joint optimization of subject-specific and temporal features, mitigating catastrophic forgetting. Our method achieves strong subject consistency and scalability, outperforming existing video customization models in zero-shot settings, demonstrating the effectiveness of our framework.
>
---
#### [replaced 046] AttriCtrl: Fine-Grained Control of Aesthetic Attribute Intensity in Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02151v2](https://arxiv.org/pdf/2508.02151v2)**

> **作者:** Die Chen; Zhongjie Duan; Zhiwen Li; Cen Chen; Daoyuan Chen; Yaliang Li; Yingda Chen
>
> **摘要:** Diffusion models have recently become the dominant paradigm for image generation, yet existing systems struggle to interpret and follow numeric instructions for adjusting semantic attributes. In real-world creative scenarios, especially when precise control over aesthetic attributes is required, current methods fail to provide such controllability. This limitation partly arises from the subjective and context-dependent nature of aesthetic judgments, but more fundamentally stems from the fact that current text encoders are designed for discrete tokens rather than continuous values. Meanwhile, efforts on aesthetic alignment, often leveraging reinforcement learning, direct preference optimization, or architectural modifications, primarily align models with a global notion of human preference. While these approaches improve user experience, they overlook the multifaceted and compositional nature of aesthetics, underscoring the need for explicit disentanglement and independent control of aesthetic attributes. To address this gap, we introduce AttriCtrl, a lightweight framework for continuous aesthetic intensity control in diffusion models. It first defines relevant aesthetic attributes, then quantifies them through a hybrid strategy that maps both concrete and abstract dimensions onto a unified $[0,1]$ scale. A plug-and-play value encoder is then used to transform user-specified values into model-interpretable embeddings for controllable generation. Experiments show that AttriCtrl achieves accurate and continuous control over both single and multiple aesthetic attributes, significantly enhancing personalization and diversity. Crucially, it is implemented as a lightweight adapter while keeping the diffusion model frozen, ensuring seamless integration with existing frameworks such as ControlNet at negligible computational cost.
>
---
#### [replaced 047] DYRECT Computed Tomography: DYnamic Reconstruction of Events on a Continuous Timescale
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.00065v2](https://arxiv.org/pdf/2412.00065v2)**

> **作者:** Wannes Goethals; Tom Bultreys; Steffen Berg; Matthieu N. Boone; Jan Aelterman
>
> **备注:** 13 pages, 10 figures, article. Submitted to IEEE Transactions on Computational Imaging 23/10/2024 - Accepted 18/04/2025 - Published 01/05/2025
>
> **摘要:** Time-resolved high-resolution X-ray Computed Tomography (4D $μ$CT) is an imaging technique that offers insight into the evolution of dynamic processes inside materials that are opaque to visible light. Conventional tomographic reconstruction techniques are based on recording a sequence of 3D images that represent the sample state at different moments in time. This frame-based approach limits the temporal resolution compared to dynamic radiography experiments due to the time needed to make CT scans. Moreover, it leads to an inflation of the amount of data and thus to costly post-processing computations to quantify the dynamic behaviour from the sequence of time frames, hereby often ignoring the temporal correlations of the sample structure. Our proposed 4D $μ$CT reconstruction technique, named DYRECT, estimates individual attenuation evolution profiles for each position in the sample. This leads to a novel memory-efficient event-based representation of the sample, using as little as three image volumes: its initial attenuation, its final attenuation and the transition times. This third volume represents local events on a continuous timescale instead of the discrete global time frames. We propose a method to iteratively reconstruct the transition times and the attenuation volumes. The dynamic reconstruction technique was validated on synthetic ground truth data and experimental data, and was found to effectively pinpoint the transition times in the synthetic dataset with a time resolution corresponding to less than a tenth of the amount of projections required to reconstruct traditional $μ$CT time frames.
>
---
#### [replaced 048] Neural-Driven Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.05397v3](https://arxiv.org/pdf/2507.05397v3)**

> **作者:** Pengfei Zhou; Jie Xia; Xiaopeng Peng; Wangbo Zhao; Zilong Ye; Zekai Li; Suorong Yang; Jiadong Pan; Yuanxiang Chen; Ziqiao Wang; Kai Wang; Qian Zheng; Hao Jin; Xiaojun Chang; Gang Pan; Shurong Dong; Kaipeng Zhang; Yang You
>
> **备注:** 22 pages, 14 figures
>
> **摘要:** Traditional image editing typically relies on manual prompting, making it labor-intensive and inaccessible to individuals with limited motor control or language abilities. Leveraging recent advances in brain-computer interfaces (BCIs) and generative models, we propose LoongX, a hands-free image editing approach driven by multimodal neurophysiological signals. LoongX utilizes state-of-the-art diffusion models trained on a comprehensive dataset of 23,928 image editing pairs, each paired with synchronized electroencephalography (EEG), functional near-infrared spectroscopy (fNIRS), photoplethysmography (PPG), and head motion signals that capture user intent. To effectively address the heterogeneity of these signals, LoongX integrates two key modules. The cross-scale state space (CS3) module encodes informative modality-specific features. The dynamic gated fusion (DGF) module further aggregates these features into a unified latent space, which is then aligned with edit semantics via fine-tuning on a diffusion transformer (DiT). Additionally, we pre-train the encoders using contrastive learning to align cognitive states with semantic intentions from embedded natural language. Extensive experiments demonstrate that LoongX achieves performance comparable to text-driven methods (CLIP-I: 0.6605 vs. 0.6558; DINO: 0.4812 vs. 0.4636) and outperforms them when neural signals are combined with speech (CLIP-T: 0.2588 vs. 0.2549). These results highlight the promise of neural-driven generative models in enabling accessible, intuitive image editing and open new directions for cognitive-driven creative technologies. The code and dataset are released on the project website: https://loongx1.github.io.
>
---
#### [replaced 049] Solving Inverse Problems in Stochastic Self-Organizing Systems through Invariant Representations
- **分类: nlin.AO; cond-mat.dis-nn; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.11796v2](https://arxiv.org/pdf/2506.11796v2)**

> **作者:** Elias Najarro; Nicolas Bessone; Sebastian Risi
>
> **备注:** Preprint. Under review
>
> **摘要:** Self-organizing systems demonstrate how simple local rules can generate complex stochastic patterns. Many natural systems rely on such dynamics, making self-organization central to understanding natural complexity. A fundamental challenge in modeling such systems is solving the inverse problem: finding the unknown causal parameters from macroscopic observations. This task becomes particularly difficult when observations have a strong stochastic component, yielding diverse yet equivalent patterns. Traditional inverse methods fail in this setting, as pixel-wise metrics cannot capture feature similarities between variable outcomes. In this work, we introduce a novel inverse modeling method specifically designed to handle stochasticity in the observable space, leveraging the capacity of visual embeddings to produce robust representations that capture perceptual invariances. By mapping the pattern representations onto an invariant embedding space, we can effectively recover unknown causal parameters without the need for handcrafted objective functions or heuristics. We evaluate the method on three self-organizing systems: a physical, a biological, and a social one; namely, a reaction-diffusion system, a model of embryonic development, and an agent-based model of social segregation. We show that the method reliably recovers parameters despite stochasticity in the pattern outcomes. We further apply the method to real biological patterns, highlighting its potential as a tool for both theorists and experimentalists to investigate the dynamics underlying complex stochastic pattern formation.
>
---
#### [replaced 050] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12409v3](https://arxiv.org/pdf/2506.12409v3)**

> **作者:** Ziwei Liu; Borui Kang; Wei Li; Hangjie Yuan; Yanbing Yang; Wenbin Li; Yifan Zhu; Tao Feng; Jun Luo
>
> **备注:** Published in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Vision-Language Continual Learning (VLCL) has attracted significant research attention for its robust capabilities, and the adoption of Parameter-Efficient Fine-Tuning (PEFT) strategies is enabling these models to achieve competitive performance with substantially reduced resource consumption. However, dominated First-Order (FO) optimization is prone to trap models in suboptimal local minima, especially in limited exploration subspace within PEFT. To overcome this challenge, this paper pioneers a systematic exploration of adopting Zeroth-Order (ZO) optimization for PEFT-based VLCL. We first identify the incompatibility of naive full-ZO adoption in VLCL due to optimization process instability. We then investigate the application of ZO optimization from a modality branch-wise to a fine-grained layer-wise across various training units to identify an optimal strategy. Besides, a key theoretical insight reveals that vision modality exhibit higher variance than language counterparts in VLCL during the ZO optimization process, and we propose a modality-aware ZO strategy, which adopts gradient sign normalization in ZO and constrains vision modality perturbation to further improve performance. Benefiting from the adoption of ZO optimization, PEFT-based VLCL fulfills better ability to escape local minima during the optimization process, extensive experiments on four benchmarks demonstrate that our method achieves state-of-the-art results.
>
---
#### [replaced 051] SOVABench: A Vehicle Surveillance Action Retrieval Benchmark for Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.04824v2](https://arxiv.org/pdf/2601.04824v2)**

> **作者:** Oriol Rabasseda; Zenjie Li; Kamal Nasrollahi; Sergio Escalera
>
> **备注:** This work has been accepted at Real World Surveillance: Applications and Challenges, 6th (in WACV Workshops)
>
> **摘要:** Automatic identification of events and recurrent behavior analysis are critical for video surveillance. However, most existing content-based video retrieval benchmarks focus on scene-level similarity and do not evaluate the action discrimination required in surveillance. To address this gap, we introduce SOVABench (Surveillance Opposite Vehicle Actions Benchmark), a real-world retrieval benchmark built from surveillance footage and centered on vehicle-related actions. SOVABench defines two evaluation protocols (inter-pair and intra-pair) to assess cross-action discrimination and temporal direction understanding. Although action distinctions are generally intuitive for human observers, our experiments show that they remain challenging for state-of-the-art vision and multimodal models. Leveraging the visual reasoning and instruction-following capabilities of Multimodal Large Language Models (MLLMs), we present a training-free framework for producing interpretable embeddings from MLLM-generated descriptions for both images and videos. The framework achieves strong performance on SOVABench as well as on several spatial and counting benchmarks where contrastive Vision-Language Models often fail. The code, annotations, and instructions to construct the benchmark are publicly available.
>
---
