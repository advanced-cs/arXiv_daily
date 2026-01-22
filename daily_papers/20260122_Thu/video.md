# 计算机视觉 cs.CV

- **最新发布 97 篇**

- **更新 52 篇**

## 最新发布

#### [new 001] Enhancing Few-Shot Out-of-Distribution Detection via the Refinement of Foreground and Background
- **分类: cs.CV**

- **简介: 该论文属于少样本分布外检测任务，针对现有方法在背景抑制和前景混淆上的不足，提出一个包含分解、自适应抑制和修正的框架，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2601.15065v1](https://arxiv.org/pdf/2601.15065v1)**

> **作者:** Tianyu Li; Songyue Cai; Zongqian Wu; Ping Hu; Xiaofeng Zhu
>
> **摘要:** CLIP-based foreground-background (FG-BG) decomposition methods have demonstrated remarkable effectiveness in improving few-shot out-of-distribution (OOD) detection performance. However, existing approaches still suffer from several limitations. For background regions obtained from decomposition, existing methods adopt a uniform suppression strategy for all patches, overlooking the varying contributions of different patches to the prediction. For foreground regions, existing methods fail to adequately consider that some local patches may exhibit appearance or semantic similarity to other classes, which may mislead the training process. To address these issues, we propose a new plug-and-play framework. This framework consists of three core components: (1) a Foreground-Background Decomposition module, which follows previous FG-BG methods to separate an image into foreground and background regions; (2) an Adaptive Background Suppression module, which adaptively weights patch classification entropy; and (3) a Confusable Foreground Rectification module, which identifies and rectifies confusable foreground patches. Extensive experimental results demonstrate that the proposed plug-and-play framework significantly improves the performance of existing FG-BG decomposition methods. Code is available at: https://github.com/lounwb/FoBoR.
>
---
#### [new 002] SpatialMem: Unified 3D Memory with Metric Anchoring and Fast Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SpatialMem，属于室内场景理解任务，旨在统一3D几何、语义与语言信息，解决环境建模与高效检索问题。通过构建层次化记忆结构实现快速查询与空间推理。**

- **链接: [https://arxiv.org/pdf/2601.14895v1](https://arxiv.org/pdf/2601.14895v1)**

> **作者:** Xinyi Zheng; Yunze Liu; Chi-Hao Wu; Fan Zhang; Hao Zheng; Wenqi Zhou; Walterio W. Mayol-Cuevas; Junxiao Shen
>
> **摘要:** We present SpatialMem, a memory-centric system that unifies 3D geometry, semantics, and language into a single, queryable representation. Starting from casually captured egocentric RGB video, SpatialMem reconstructs metrically scaled indoor environments, detects structural 3D anchors (walls, doors, windows) as the first-layer scaffold, and populates a hierarchical memory with open-vocabulary object nodes -- linking evidence patches, visual embeddings, and two-layer textual descriptions to 3D coordinates -- for compact storage and fast retrieval. This design enables interpretable reasoning over spatial relations (e.g., distance, direction, visibility) and supports downstream tasks such as language-guided navigation and object retrieval without specialized sensors. Experiments across three real-life indoor scenes demonstrate that SpatialMem maintains strong anchor-description-level navigation completion and hierarchical retrieval accuracy under increasing clutter and occlusion, offering an efficient and extensible framework for embodied spatial intelligence.
>
---
#### [new 003] LaVR: Scene Latent Conditioned Generative Video Trajectory Re-Rendering using Large 4D Reconstruction Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频重渲染任务，旨在解决单目视频生成新视角画面时的几何偏差问题。通过利用大4D重建模型的潜在空间信息，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.14674v1](https://arxiv.org/pdf/2601.14674v1)**

> **作者:** Mingyang Xie; Numair Khan; Tianfu Wang; Naina Dhingra; Seonghyeon Nam; Haitao Yang; Zhuo Hui; Christopher Metzler; Andrea Vedaldi; Hamed Pirsiavash; Lei Luo
>
> **摘要:** Given a monocular video, the goal of video re-rendering is to generate views of the scene from a novel camera trajectory. Existing methods face two distinct challenges. Geometrically unconditioned models lack spatial awareness, leading to drift and deformation under viewpoint changes. On the other hand, geometrically-conditioned models depend on estimated depth and explicit reconstruction, making them susceptible to depth inaccuracies and calibration errors. We propose to address these challenges by using the implicit geometric knowledge embedded in the latent space of a large 4D reconstruction model to condition the video generation process. These latents capture scene structure in a continuous space without explicit reconstruction. Therefore, they provide a flexible representation that allows the pretrained diffusion prior to regularize errors more effectively. By jointly conditioning on these latents and source camera poses, we demonstrate that our model achieves state-of-the-art results on the video re-rendering task. Project webpage is https://lavr-4d-scene-rerender.github.io/
>
---
#### [new 004] FeedbackSTS-Det: Sparse Frames-Based Spatio-Temporal Semantic Feedback Network for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文属于红外小目标检测任务，旨在解决复杂背景下的低信杂比、动态干扰和特征不明显问题。提出FeedbackSTS-Det网络，通过时空语义反馈机制提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.14690v1](https://arxiv.org/pdf/2601.14690v1)**

> **作者:** Yian Huang; Qing Qin; Aji Mao; Xiangyu Qiu; Liang Xu; Xian Zhang; Zhenming Peng
>
> **备注:** Submitted to Journal IEEE Transactions on Geoscience and Remote Sensing
>
> **摘要:** Infrared small target detection (ISTD) under complex backgrounds remains a critical yet challenging task, primarily due to the extremely low signal-to-clutter ratio, persistent dynamic interference, and the lack of distinct target features. While multi-frame detection methods leverages temporal cues to improve upon single-frame approaches, existing methods still struggle with inefficient long-range dependency modeling and insufficient robustness. To overcome these issues, we propose a novel scheme for ISTD, realized through a sparse frames-based spatio-temporal semantic feedback network named FeedbackSTS-Det. The core of our approach is a novel spatio-temporal semantic feedback strategy with a closed-loop semantic association mechanism, which consists of paired forward and backward refinement modules that work cooperatively across the encoder and decoder. Moreover, both modules incorporate an embedded sparse semantic module (SSM), which performs structured sparse temporal modeling to capture long-range dependencies with low computational cost. This integrated design facilitates robust implicit inter-frame registration and continuous semantic refinement, effectively suppressing false alarms. Furthermore, our overall procedure maintains a consistent training-inference pipeline, which ensures reliable performance transfer and increases model robustness. Extensive experiments on multiple benchmark datasets confirm the effectiveness of FeedbackSTS-Det. Code and models are available at: https://github.com/IDIP-Lab/FeedbackSTS-Det.
>
---
#### [new 005] Large-Scale Multidimensional Knowledge Profiling of Scientific Literature
- **分类: cs.CV**

- **简介: 该论文属于科学文献分析任务，旨在解决传统工具无法深入理解研究主题演化的问题。通过构建多维知识图谱，分析论文内容，揭示研究趋势与变化。**

- **链接: [https://arxiv.org/pdf/2601.15170v1](https://arxiv.org/pdf/2601.15170v1)**

> **作者:** Zhucun Xue; Jiangning Zhang; Juntao Jiang; Jinzhuo Liu; Haoyang He; Teng Hu; Xiaobin Hu; Guangming Yao; Yi Yuan; Yong Liu
>
> **备注:** Code and dataset: https://github.com/xzc-zju/Profiling_Scientific_Literature
>
> **摘要:** The rapid expansion of research across machine learning, vision, and language has produced a volume of publications that is increasingly difficult to synthesize. Traditional bibliometric tools rely mainly on metadata and offer limited visibility into the semantic content of papers, making it hard to track how research themes evolve over time or how different areas influence one another. To obtain a clearer picture of recent developments, we compile a unified corpus of more than 100,000 papers from 22 major conferences between 2020 and 2025 and construct a multidimensional profiling pipeline to organize and analyze their textual content. By combining topic clustering, LLM-assisted parsing, and structured retrieval, we derive a comprehensive representation of research activity that supports the study of topic lifecycles, methodological transitions, dataset and model usage patterns, and institutional research directions. Our analysis highlights several notable shifts, including the growth of safety, multimodal reasoning, and agent-oriented studies, as well as the gradual stabilization of areas such as neural machine translation and graph-based methods. These findings provide an evidence-based view of how AI research is evolving and offer a resource for understanding broader trends and identifying emerging directions. Code and dataset: https://github.com/xzc-zju/Profiling_Scientific_Literature
>
---
#### [new 006] Tracing 3D Anatomy in 2D Strokes: A Multi-Stage Projection Driven Approach to Cervical Spine Fracture Identification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决颈椎骨折的自动检测问题。通过2D投影与多阶段分割方法，实现3D颈椎结构的高效识别与骨折分析。**

- **链接: [https://arxiv.org/pdf/2601.15235v1](https://arxiv.org/pdf/2601.15235v1)**

> **作者:** Fabi Nahian Madhurja; Rusab Sarmun; Muhammad E. H. Chowdhury; Adam Mushtak; Israa Al-Hashimi; Sohaib Bassam Zoghoul
>
> **摘要:** Cervical spine fractures are critical medical conditions requiring precise and efficient detection for effective clinical management. This study explores the viability of 2D projection-based vertebra segmentation for vertebra-level fracture detection in 3D CT volumes, presenting an end-to-end pipeline for automated analysis of cervical vertebrae (C1-C7). By approximating a 3D volume through optimized 2D axial, sagittal, and coronal projections, regions of interest are identified using the YOLOv8 model from all views and combined to approximate the 3D cervical spine area, achieving a 3D mIoU of 94.45 percent. This projection-based localization strategy reduces computational complexity compared to traditional 3D segmentation methods while maintaining high performance. It is followed by a DenseNet121-Unet-based multi-label segmentation leveraging variance- and energy-based projections, achieving a Dice score of 87.86 percent. Strategic approximation of 3D vertebral masks from these 2D segmentation masks enables the extraction of individual vertebra volumes. The volumes are analyzed for fractures using an ensemble of 2.5D Spatio-Sequential models incorporating both raw slices and projections per vertebra for complementary evaluation. This ensemble achieves vertebra-level and patient-level F1 scores of 68.15 and 82.26, and ROC-AUC scores of 91.62 and 83.04, respectively. We further validate our approach through an explainability study that provides saliency map visualizations highlighting anatomical regions relevant for diagnosis, and an interobserver variability analysis comparing our model's performance with expert radiologists, demonstrating competitive results.
>
---
#### [new 007] LookBench: A Live and Holistic Open Benchmark for Fashion Image Retrieval
- **分类: cs.CV**

- **简介: 该论文提出LookBench，一个用于时尚图像检索的实时、全面基准。旨在解决真实电商环境中图像检索的挑战，包含真实和生成数据，评估模型性能并推动技术进步。**

- **链接: [https://arxiv.org/pdf/2601.14706v1](https://arxiv.org/pdf/2601.14706v1)**

> **作者:** Chao Gao; Siqiao Xue; Yimin Peng; Jiwen Fu; Tingyi Gu; Shanshan Li; Fan Zhou
>
> **备注:** The first two authors contributed equally to this work. Project site: https://serendipityoneinc.github.io/look-bench-page/
>
> **摘要:** In this paper, we present LookBench (We use the term "look" to reflect retrieval that mirrors how people shop -- finding the exact item, a close substitute, or a visually consistent alternative.), a live, holistic and challenging benchmark for fashion image retrieval in real e-commerce settings. LookBench includes both recent product images sourced from live websites and AI-generated fashion images, reflecting contemporary trends and use cases. Each test sample is time-stamped and we intend to update the benchmark periodically, enabling contamination-aware evaluation aligned with declared training cutoffs. Grounded in our fine-grained attribute taxonomy, LookBench covers single-item and outfit-level retrieval across. Our experiments reveal that LookBench poses a significant challenge on strong baselines, with many models achieving below $60\%$ Recall@1. Our proprietary model achieves the best performance on LookBench, and we release an open-source counterpart that ranks second, with both models attaining state-of-the-art results on legacy Fashion200K evaluations. LookBench is designed to be updated semi-annually with new test samples and progressively harder task variants, providing a durable measure of progress. We publicly release our leaderboard, dataset, evaluation code, and trained models.
>
---
#### [new 008] Transfer Learning from One Cancer to Another via Deep Learning Domain Adaptation
- **分类: cs.CV; cs.AI; cs.LG; cs.NE; q-bio.TO**

- **简介: 该论文属于跨域分类任务，旨在解决癌症类型间模型泛化不足的问题。通过域适应方法提升模型在未标注数据上的表现。**

- **链接: [https://arxiv.org/pdf/2601.14678v1](https://arxiv.org/pdf/2601.14678v1)**

> **作者:** Justin Cheung; Samuel Savine; Calvin Nguyen; Lin Lu; Alhassan S. Yasin
>
> **备注:** 8 pages, 6 figures, 3 table
>
> **摘要:** Supervised deep learning models often achieve excellent performance within their training distribution but struggle to generalize beyond it. In cancer histopathology, for example, a convolutional neural network (CNN) may classify cancer severity accurately for cancer types represented in its training data, yet fail on related but unseen types. Although adenocarcinomas from different organs share morphological features that might support limited cross-domain generalization, addressing domain shift directly is necessary for robust performance. Domain adaptation offers a way to transfer knowledge from labeled data in one cancer type to unlabeled data in another, helping mitigate the scarcity of annotated medical images. This work evaluates cross-domain classification performance among lung, colon, breast, and kidney adenocarcinomas. A ResNet50 trained on any single adenocarcinoma achieves over 98% accuracy on its own domain but shows minimal generalization to others. Ensembling multiple supervised models does not resolve this limitation. In contrast, converting the ResNet50 into a domain adversarial neural network (DANN) substantially improves performance on unlabeled target domains. A DANN trained on labeled breast and colon data and adapted to unlabeled lung data reaches 95.56% accuracy. We also examine the impact of stain normalization on domain adaptation. Its effects vary by target domain: for lung, accuracy drops from 95.56% to 66.60%, while for breast and colon targets, stain normalization boosts accuracy from 49.22% to 81.29% and from 78.48% to 83.36%, respectively. Finally, using Integrated Gradients reveals that DANNs consistently attribute importance to biologically meaningful regions such as densely packed nuclei, indicating that the model learns clinically relevant features and can apply them to unlabeled cancer types.
>
---
#### [new 009] READ-Net: Clarifying Emotional Ambiguity via Adaptive Feature Recalibration for Audio-Visual Depression Detection
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音频-视觉抑郁检测任务，旨在解决情感模糊性问题。通过自适应特征重新校准（AFR）方法，提升抑郁相关信号的识别准确性。**

- **链接: [https://arxiv.org/pdf/2601.14651v1](https://arxiv.org/pdf/2601.14651v1)**

> **作者:** Chenglizhao Chen; Boze Li; Mengke Song; Dehao Feng; Xinyu Liu; Shanchen Pang; Jufeng Yang; Hui Yu
>
> **备注:** 12 pages
>
> **摘要:** Depression is a severe global mental health issue that impairs daily functioning and overall quality of life. Although recent audio-visual approaches have improved automatic depression detection, methods that ignore emotional cues often fail to capture subtle depressive signals hidden within emotional expressions. Conversely, those incorporating emotions frequently confuse transient emotional expressions with stable depressive symptoms in feature representations, a phenomenon termed \emph{Emotional Ambiguity}, thereby leading to detection errors. To address this critical issue, we propose READ-Net, the first audio-visual depression detection framework explicitly designed to resolve Emotional Ambiguity through Adaptive Feature Recalibration (AFR). The core insight of AFR is to dynamically adjust the weights of emotional features to enhance depression-related signals. Rather than merely overlooking or naively combining emotional information, READ-Net innovatively identifies and preserves depressive-relevant cues within emotional features, while adaptively filtering out irrelevant emotional noise. This recalibration strategy significantly clarifies feature representations, and effectively mitigates the persistent challenge of emotional interference. Additionally, READ-Net can be easily integrated into existing frameworks for improved performance. Extensive evaluations on three publicly available datasets show that READ-Net outperforms state-of-the-art methods, with average gains of 4.55\% in accuracy and 1.26\% in F1-score, demonstrating its robustness to emotional disturbances and improving audio-visual depression detection.
>
---
#### [new 010] RegFreeNet: A Registration-Free Network for CBCT-based 3D Dental Implant Planning
- **分类: cs.CV**

- **简介: 该论文属于3D牙科种植体定位任务，解决传统方法依赖配准和数据配对的问题。通过掩码技术实现无需配准的训练，提出RegFreeNet网络和ImplantFairy数据集。**

- **链接: [https://arxiv.org/pdf/2601.14703v1](https://arxiv.org/pdf/2601.14703v1)**

> **作者:** Xinquan Yang; Xuguang Li; Mianjie Zheng; Xuefen Liu; Kun Tang; Kian Ming Lim; He Meng; Jianfeng Ren; Linlin Shen
>
> **摘要:** As the commercial surgical guide design software usually does not support the export of implant position for pre-implantation data, existing methods have to scan the post-implantation data and map the implant to pre-implantation space to get the label of implant position for training. Such a process is time-consuming and heavily relies on the accuracy of registration algorithm. Moreover, not all hospitals have paired CBCT data, limitting the construction of multi-center dataset. Inspired by the way dentists determine the implant position based on the neighboring tooth texture, we found that even if the implant area is masked, it will not affect the determination of the implant position. Therefore, we propose to mask the implants in the post-implantation data so that any CBCT containing the implants can be used as training data. This paradigm enables us to discard the registration process and makes it possible to construct a large-scale multi-center implant dataset. On this basis, we proposes ImplantFairy, a comprehensive, publicly accessible dental implant dataset with voxel-level 3D annotations of 1622 CBCT data. Furthermore, according to the area variation characteristics of the tooth's spatial structure and the slope information of the implant, we designed a slope-aware implant position prediction network. Specifically, a neighboring distance perception (NDP) module is designed to adaptively extract tooth area variation features, and an implant slope prediction branch assists the network in learning more robust features through additional implant supervision information. Extensive experiments conducted on ImplantFairy and two public dataset demonstrate that the proposed RegFreeNet achieves the state-of-the-art performance.
>
---
#### [new 011] StableWorld: Towards Stable and Consistent Long Interactive Video Generation
- **分类: cs.CV**

- **简介: 该论文属于交互式视频生成任务，旨在解决长期交互中的稳定性与时间一致性问题。通过提出动态帧剔除机制，提升生成视频的稳定性和连贯性。**

- **链接: [https://arxiv.org/pdf/2601.15281v1](https://arxiv.org/pdf/2601.15281v1)**

> **作者:** Ying Yang; Zhengyao Lv; Tianlin Pan; Haofan Wang; Binxin Yang; Hubery Yin; Chen Li; Ziwei Liu; Chenyang Si
>
> **备注:** 17 pages, 21 figures,
>
> **摘要:** In this paper, we explore the overlooked challenge of stability and temporal consistency in interactive video generation, which synthesizes dynamic and controllable video worlds through interactive behaviors such as camera movements and text prompts. Despite remarkable progress in world modeling, current methods still suffer from severe instability and temporal degradation, often leading to spatial drift and scene collapse during long-horizon interactions. To better understand this issue, we initially investigate the underlying causes of instability and identify that the major source of error accumulation originates from the same scene, where generated frames gradually deviate from the initial clean state and propagate errors to subsequent frames. Building upon this observation, we propose a simple yet effective method, \textbf{StableWorld}, a Dynamic Frame Eviction Mechanism. By continuously filtering out degraded frames while retaining geometrically consistent ones, StableWorld effectively prevents cumulative drift at its source, leading to more stable and temporal consistency of interactive generation. Promising results on multiple interactive video models, \eg, Matrix-Game, Open-Oasis, and Hunyuan-GameCraft, demonstrate that StableWorld is model-agnostic and can be applied to different interactive video generation frameworks to substantially improve stability, temporal consistency, and generalization across diverse interactive scenarios.
>
---
#### [new 012] DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration
- **分类: cs.CV**

- **简介: 该论文提出DrivIng数据集，解决自动驾驶感知数据不足问题。包含多模态传感器数据与高保真数字孪生，支持复杂场景测试与仿真验证。**

- **链接: [https://arxiv.org/pdf/2601.15260v1](https://arxiv.org/pdf/2601.15260v1)**

> **作者:** Dominik Rößle; Xujun Xie; Adithya Mohan; Venkatesh Thirugnana Sambandham; Daniel Cremers; Torsten Schön
>
> **备注:** Accepted to the IEEE Intelligent Vehicles Symposium 2026. For code and dataset, see https://github.com/cvims/DrivIng
>
> **摘要:** Perception is a cornerstone of autonomous driving, enabling vehicles to understand their surroundings and make safe, reliable decisions. Developing robust perception algorithms requires large-scale, high-quality datasets that cover diverse driving conditions and support thorough evaluation. Existing datasets often lack a high-fidelity digital twin, limiting systematic testing, edge-case simulation, sensor modification, and sim-to-real evaluations. To address this gap, we present DrivIng, a large-scale multimodal dataset with a complete geo-referenced digital twin of a ~18 km route spanning urban, suburban, and highway segments. Our dataset provides continuous recordings from six RGB cameras, one LiDAR, and high-precision ADMA-based localization, captured across day, dusk, and night. All sequences are annotated at 10 Hz with 3D bounding boxes and track IDs across 12 classes, yielding ~1.2 million annotated instances. Alongside the benefits of a digital twin, DrivIng enables a 1-to-1 transfer of real traffic into simulation, preserving agent interactions while enabling realistic and flexible scenario testing. To support reproducible research and robust validation, we benchmark DrivIng with state-of-the-art perception models and publicly release the dataset, digital twin, HD map, and codebase.
>
---
#### [new 013] Mirai: Autoregressive Visual Generation Needs Foresight
- **分类: cs.CV**

- **简介: 该论文属于视觉生成任务，旨在解决AR模型全局一致性差、收敛慢的问题。通过引入未来信息（foresight）提升生成质量与速度。**

- **链接: [https://arxiv.org/pdf/2601.14671v1](https://arxiv.org/pdf/2601.14671v1)**

> **作者:** Yonghao Yu; Lang Huang; Zerun Wang; Runyi Li; Toshihiko Yamasaki
>
> **摘要:** Autoregressive (AR) visual generators model images as sequences of discrete tokens and are trained with next token likelihood. This strict causality supervision optimizes each step only by its immediate next token, which diminishes global coherence and slows convergence. We ask whether foresight, training signals that originate from later tokens, can help AR visual generation. We conduct a series of controlled diagnostics along the injection level, foresight layout, and foresight source axes, unveiling a key insight: aligning foresight to AR models' internal representation on the 2D image grids improves causality modeling. We formulate this insight with Mirai (meaning "future" in Japanese), a general framework that injects future information into AR training with no architecture change and no extra inference overhead: Mirai-E uses explicit foresight from multiple future positions of unidirectional representations, whereas Mirai-I leverages implicit foresight from matched bidirectional representations. Extensive experiments show that Mirai significantly accelerates convergence and improves generation quality. For instance, Mirai can speed up LlamaGen-B's convergence by up to 10$\times$ and reduce the generation FID from 5.34 to 4.34 on the ImageNet class-condition image generation benchmark. Our study highlights that visual autoregressive models need foresight.
>
---
#### [new 014] POTR: Post-Training 3DGS Compression
- **分类: cs.CV**

- **简介: 该论文属于3D场景压缩任务，解决3DGS存储需求高的问题。提出POTR方法，通过剪枝和重新计算光照系数实现高效压缩与加速。**

- **链接: [https://arxiv.org/pdf/2601.14821v1](https://arxiv.org/pdf/2601.14821v1)**

> **作者:** Bert Ramlot; Martijn Courteaux; Peter Lambert; Glenn Van Wallendael
>
> **备注:** 15 pages, 12 figures. Submitted to IEEE TCSVT, under review
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as a promising contender to Neural Radiance Fields (NeRF) in 3D scene reconstruction and real-time novel view synthesis. 3DGS outperforms NeRF in training and inference speed but has substantially higher storage requirements. To remedy this downside, we propose POTR, a post-training 3DGS codec built on two novel techniques. First, POTR introduces a novel pruning approach that uses a modified 3DGS rasterizer to efficiently calculate every splat's individual removal effect simultaneously. This technique results in 2-4x fewer splats than other post-training pruning techniques and as a result also significantly accelerates inference with experiments demonstrating 1.5-2x faster inference than other compressed models. Second, we propose a novel method to recompute lighting coefficients, significantly reducing their entropy without using any form of training. Our fast and highly parallel approach especially increases AC lighting coefficient sparsity, with experiments demonstrating increases from 70% to 97%, with minimal loss in quality. Finally, we extend POTR with a simple fine-tuning scheme to further enhance pruning, inference, and rate-distortion performance. Experiments demonstrate that POTR, even without fine-tuning, consistently outperforms all other post-training compression techniques in both rate-distortion performance and inference speed.
>
---
#### [new 015] UniRoute: Unified Routing Mixture-of-Experts for Modality-Adaptive Remote Sensing Change Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，解决多模态适应性问题。提出UniRoute框架，通过路由机制实现特征融合与差异计算，提升不同模态下的检测性能。**

- **链接: [https://arxiv.org/pdf/2601.14797v1](https://arxiv.org/pdf/2601.14797v1)**

> **作者:** Qingling Shu; Sibao Chen; Wei Lu; Zhihui You; Chengzhuang Liu
>
> **摘要:** Current remote sensing change detection (CD) methods mainly rely on specialized models, which limits the scalability toward modality-adaptive Earth observation. For homogeneous CD, precise boundary delineation relies on fine-grained spatial cues and local pixel interactions, whereas heterogeneous CD instead requires broader contextual information to suppress speckle noise and geometric distortions. Moreover, difference operator (e.g., subtraction) works well for aligned homogeneous images but introduces artifacts in cross-modal or geometrically misaligned scenarios. Across different modality settings, specialized models based on static backbones or fixed difference operations often prove insufficient. To address this challenge, we propose UniRoute, a unified framework for modality-adaptive learning by reformulating feature extraction and fusion as conditional routing problems. We introduce an Adaptive Receptive Field Routing MoE (AR2-MoE) module to disentangle local spatial details from global semantic context, and a Modality-Aware Difference Routing MoE (MDR-MoE) module to adaptively select the most suitable fusion primitive at each pixel. In addition, we propose a Consistency-Aware Self-Distillation (CASD) strategy that stabilizes unified training under data-scarce heterogeneous settings by enforcing multi-level consistency. Extensive experiments on five public datasets demonstrate that UniRoute achieves strong overall performance, with a favorable accuracy-efficiency trade-off under a unified deployment setting.
>
---
#### [new 016] Enhancing Text-to-Image Generation via End-Edge Collaborative Hybrid Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决高分辨率图像生成中的延迟与质量平衡问题。提出端边协同的混合超分辨率方法，提升生成效率与图像质量。**

- **链接: [https://arxiv.org/pdf/2601.14741v1](https://arxiv.org/pdf/2601.14741v1)**

> **作者:** Chongbin Yi; Yuxin Liang; Ziqi Zhou; Peng Yang
>
> **备注:** Accpeted by ICC 2026
>
> **摘要:** Artificial Intelligence-Generated Content (AIGC) has made significant strides, with high-resolution text-to-image (T2I) generation becoming increasingly critical for improving users' Quality of Experience (QoE). Although resource-constrained edge computing adequately supports fast low-resolution T2I generations, achieving high-resolution output still faces the challenge of ensuring image fidelity at the cost of latency. To address this, we first investigate the performance of super-resolution (SR) methods for image enhancement, confirming a fundamental trade-off that lightweight learning-based SR struggles to recover fine details, while diffusion-based SR achieves higher fidelity at a substantial computational cost. Motivated by these observations, we propose an end-edge collaborative generation-enhancement framework. Upon receiving a T2I generation task, the system first generates a low-resolution image based on adaptively selected denoising steps and super-resolution scales at the edge side, which is then partitioned into patches and processed by a region-aware hybrid SR policy. This policy applies a diffusion-based SR model to foreground patches for detail recovery and a lightweight learning-based SR model to background patches for efficient upscaling, ultimately stitching the enhanced ones into the high-resolution image. Experiments show that our system reduces service latency by 33% compared with baselines while maintaining competitive image quality.
>
---
#### [new 017] The Pictorial Cortex: Zero-Shot Cross-Subject fMRI-to-Image Reconstruction via Compositional Latent Modeling
- **分类: cs.CV**

- **简介: 该论文属于fMRI到图像的重建任务，解决跨被试零样本重建问题。通过构建统一数据集和提出PictorialCortex模型，实现跨被试的视觉体验重建。**

- **链接: [https://arxiv.org/pdf/2601.15071v1](https://arxiv.org/pdf/2601.15071v1)**

> **作者:** Jingyang Huo; Yikai Wang; Yanwei Fu; Jianfeng Feng
>
> **摘要:** Decoding visual experiences from human brain activity remains a central challenge at the intersection of neuroscience, neuroimaging, and artificial intelligence. A critical obstacle is the inherent variability of cortical responses: neural activity elicited by the same visual stimulus differs across individuals and trials due to anatomical, functional, cognitive, and experimental factors, making fMRI-to-image reconstruction non-injective. In this paper, we tackle a challenging yet practically meaningful problem: zero-shot cross-subject fMRI-to-image reconstruction, where the visual experience of a previously unseen individual must be reconstructed without subject-specific training. To enable principled evaluation, we present a unified cortical-surface dataset -- UniCortex-fMRI, assembled from multiple visual-stimulus fMRI datasets to provide broad coverage of subjects and stimuli. Our UniCortex-fMRI is particularly processed by standardized data formats to make it possible to explore this possibility in the zero-shot scenario of cross-subject fMRI-to-image reconstruction. To tackle the modeling challenge, we propose PictorialCortex, which models fMRI activity using a compositional latent formulation that structures stimulus-driven representations under subject-, dataset-, and trial-related variability. PictorialCortex operates in a universal cortical latent space and implements this formulation through a latent factorization-composition module, reinforced by paired factorization and re-factorizing consistency regularization. During inference, surrogate latents synthesized under multiple seen-subject conditions are aggregated to guide diffusion-based image synthesis for unseen subjects. Extensive experiments show that PictorialCortex improves zero-shot cross-subject visual reconstruction, highlighting the benefits of compositional latent modeling and multi-dataset training.
>
---
#### [new 018] Safeguarding Facial Identity against Diffusion-based Face Swapping via Cascading Pathway Disruption
- **分类: cs.CV**

- **简介: 该论文属于人脸交换防御任务，旨在解决扩散模型带来的身份安全问题。通过注入扰动破坏关键路径，实现有效防御。**

- **链接: [https://arxiv.org/pdf/2601.14738v1](https://arxiv.org/pdf/2601.14738v1)**

> **作者:** Liqin Wang; Qianyue Hu; Wei Lu; Xiangyang Luo
>
> **摘要:** The rapid evolution of diffusion models has democratized face swapping but also raises concerns about privacy and identity security. Existing proactive defenses, often adapted from image editing attacks, prove ineffective in this context. We attribute this failure to an oversight of the structural resilience and the unique static conditional guidance mechanism inherent in face swapping systems. To address this, we propose VoidFace, a systemic defense method that views face swapping as a coupled identity pathway. By injecting perturbations at critical bottlenecks, VoidFace induces cascading disruption throughout the pipeline. Specifically, we first introduce localization disruption and identity erasure to degrade physical regression and semantic embeddings, thereby impairing the accurate modeling of the source face. We then intervene in the generative domain by decoupling attention mechanisms to sever identity injection, and corrupting intermediate diffusion features to prevent the reconstruction of source identity. To ensure visual imperceptibility, we perform adversarial search in the latent manifold, guided by a perceptual adaptive strategy to balance attack potency with image quality. Extensive experiments show that VoidFace outperforms existing defenses across various diffusion-based swapping models, while producing adversarial faces with superior visual quality.
>
---
#### [new 019] Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，旨在提升模型对对抗攻击的鲁棒性。针对现有方法忽略上下文语义关系的问题，提出EroSeg-AT框架，通过生成更有效的对抗样本增强训练效果。**

- **链接: [https://arxiv.org/pdf/2601.14950v1](https://arxiv.org/pdf/2601.14950v1)**

> **作者:** Yufei Song; Ziqi Zhou; Menghao Deng; Yifan Hu; Shengshan Hu; Minghui Li; Leo Yu Zhang
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Existing segmentation models exhibit significant vulnerability to adversarial attacks.To improve robustness, adversarial training incorporates adversarial examples into model training. However, existing attack methods consider only global semantic information and ignore contextual semantic relationships within the samples, limiting the effectiveness of adversarial training. To address this issue, we propose EroSeg-AT, a vulnerability-aware adversarial training framework that leverages EroSeg to generate adversarial examples. EroSeg first selects sensitive pixels based on pixel-level confidence and then progressively propagates perturbations to higher-confidence pixels, effectively disrupting the semantic consistency of the samples. Experimental results show that, compared to existing methods, our approach significantly improves attack effectiveness and enhances model robustness under adversarial training.
>
---
#### [new 020] LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding
- **分类: cs.CV**

- **简介: 该论文提出LiViBench，一个针对互动直播视频的多模态基准，解决现有评测数据集缺乏互动性的问题。工作包括构建数据集、设计标注流程和改进模型以提升对直播视频的理解能力。**

- **链接: [https://arxiv.org/pdf/2601.15016v1](https://arxiv.org/pdf/2601.15016v1)**

> **作者:** Xiaodong Wang; Langling Huang; Zhirong Wu; Xu Zhao; Teng Xu; Xuhong Xia; Peixi Peng
>
> **备注:** AAAI 2026 Main Track
>
> **摘要:** The development of multimodal large language models (MLLMs) has advanced general video understanding. However, existing video evaluation benchmarks primarily focus on non-interactive videos, such as movies and recordings. To fill this gap, this paper proposes the first omnimodal benchmark for interactive livestream videos, LiViBench. It features a diverse set of 24 tasks, highlighting the perceptual, reasoning, and livestream-specific challenges. To efficiently construct the dataset, we design a standardized semi-automatic annotation workflow that incorporates the human-in-the-loop at multiple stages. The workflow leverages multiple MLLMs to form a multi-agent system for comprehensive video description and uses a seed-question-driven method to construct high-quality annotations. All interactive videos in the benchmark include audio, speech, and real-time comments modalities. To enhance models' understanding of interactive videos, we design tailored two-stage instruction-tuning and propose a Video-to-Comment Retrieval (VCR) module to improve the model's ability to utilize real-time comments. Based on these advancements, we develop LiVi-LLM-7B, an MLLM with enhanced knowledge of interactive livestreams. Experiments show that our model outperforms larger open-source models with up to 72B parameters, narrows the gap with leading proprietary models on LiViBench, and achieves enhanced performance on general video benchmarks, including VideoMME, LongVideoBench, MLVU, and VideoEval-Pro.
>
---
#### [new 021] Forest-Chat: Adapting Vision-Language Agents for Interactive Forest Change Analysis
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出Forest-Chat，用于森林变化分析的视觉语言代理系统，解决复杂森林动态的像素级变化检测与语义解释问题。**

- **链接: [https://arxiv.org/pdf/2601.14637v1](https://arxiv.org/pdf/2601.14637v1)**

> **作者:** James Brock; Ce Zhang; Nantheera Anantrasirichai
>
> **备注:** 22 pages, 8 figures, 7 tables, Submitted to Ecological Informatics
>
> **摘要:** The increasing availability of high-resolution satellite imagery, together with advances in deep learning, creates new opportunities for enhancing forest monitoring workflows. Two central challenges in this domain are pixel-level change detection and semantic change interpretation, particularly for complex forest dynamics. While large language models (LLMs) are increasingly adopted for data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored, especially beyond urban environments. We introduce Forest-Chat, an LLM-driven agent designed for integrated forest change analysis. The proposed framework enables natural language querying and supports multiple RSICI tasks, including change detection, change captioning, object counting, deforestation percentage estimation, and change reasoning. Forest-Chat builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration, and incorporates zero-shot change detection via a foundation change detection model together with an interactive point-prompt interface to support fine-grained user guidance. To facilitate adaptation and evaluation in forest environments, we introduce the Forest-Change dataset, comprising bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated through a combination of human annotation and rule-based methods. Experimental results demonstrate that Forest-Chat achieves strong performance on Forest-Change and on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI, for joint change detection and captioning, highlighting the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and analytical efficiency in forest change analysis.
>
---
#### [new 022] A comprehensive overview of deep learning models for object detection from videos/images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测任务，旨在解决视频和图像中对象检测的准确性和鲁棒性问题。通过综述深度学习模型，分析其架构、数据处理及应用挑战。**

- **链接: [https://arxiv.org/pdf/2601.14677v1](https://arxiv.org/pdf/2601.14677v1)**

> **作者:** Sukana Zulfqar; Sadia Saeed; M. Azam Zia; Anjum Ali; Faisal Mehmood; Abid Ali
>
> **备注:** N/A
>
> **摘要:** Object detection in video and image surveillance is a well-established yet rapidly evolving task, strongly influenced by recent deep learning advancements. This review summarises modern techniques by examining architectural innovations, generative model integration, and the use of temporal information to enhance robustness and accuracy. Unlike earlier surveys, it classifies methods based on core architectures, data processing strategies, and surveillance specific challenges such as dynamic environments, occlusions, lighting variations, and real-time requirements. The primary goal is to evaluate the current effectiveness of semantic object detection, while secondary aims include analysing deep learning models and their practical applications. The review covers CNN-based detectors, GAN-assisted approaches, and temporal fusion methods, highlighting how generative models support tasks such as reconstructing missing frames, reducing occlusions, and normalising illumination. It also outlines preprocessing pipelines, feature extraction progress, benchmarking datasets, and comparative evaluations. Finally, emerging trends in low-latency, efficient, and spatiotemporal learning approaches are identified for future research.
>
---
#### [new 023] FlowSSC: Universal Generative Monocular Semantic Scene Completion via One-Step Latent Diffusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于语义场景补全任务，解决单目图像中遮挡区域的3D语义生成问题。提出FlowSSC框架，通过单步扩散实现高效高质生成。**

- **链接: [https://arxiv.org/pdf/2601.15250v1](https://arxiv.org/pdf/2601.15250v1)**

> **作者:** Zichen Xi; Hao-Xiang Chen; Nan Xue; Hongyu Yan; Qi-Yuan Feng; Levent Burak Kara; Joaquim Jorge; Qun-Ce Xu
>
> **备注:** Under Review
>
> **摘要:** Semantic Scene Completion (SSC) from monocular RGB images is a fundamental yet challenging task due to the inherent ambiguity of inferring occluded 3D geometry from a single view. While feed-forward methods have made progress, they often struggle to generate plausible details in occluded regions and preserve the fundamental spatial relationships of objects. Such accurate generative reasoning capability for the entire 3D space is critical in real-world applications. In this paper, we present FlowSSC, the first generative framework applied directly to monocular semantic scene completion. FlowSSC treats the SSC task as a conditional generation problem and can seamlessly integrate with existing feed-forward SSC methods to significantly boost their performance. To achieve real-time inference without compromising quality, we introduce Shortcut Flow-matching that operates in a compact triplane latent space. Unlike standard diffusion models that require hundreds of steps, our method utilizes a shortcut mechanism to achieve high-fidelity generation in a single step, enabling practical deployment in autonomous systems. Extensive experiments on SemanticKITTI demonstrate that FlowSSC achieves state-of-the-art performance, significantly outperforming existing baselines.
>
---
#### [new 024] LFS: Learnable Frame Selector for Event-Aware and Temporally Diverse Video Captioning
- **分类: cs.CV**

- **简介: 该论文属于视频描述生成任务，旨在解决统一采样导致的事件分布不均问题。提出LFS模型，通过学习选择时间多样且事件相关的帧，提升描述质量。**

- **链接: [https://arxiv.org/pdf/2601.14594v1](https://arxiv.org/pdf/2601.14594v1)**

> **作者:** Lianying Chao; Linfeng Yin; Peiyu Ren; Yifan Jiang; Qiaoyu Ren; Dingcheng Shan; Jing-cheng Pang; Sijie Wu; Xubin Li; Kai Zhang
>
> **摘要:** Video captioning models convert frames into visual tokens and generate descriptions with large language models (LLMs). Since encoding all frames is prohibitively expensive, uniform sampling is the default choice, but it enforces equal temporal coverage while ignoring the uneven events distribution. This motivates a Learnable Frame Selector (LFS) that selects temporally diverse and event-relevant frames. LFS explicitly models temporal importance to balance temporal diversity and event relevance, and employs a stratified strategy to ensure temporal coverage while avoiding clustering. Crucially, LFS leverages caption feedback from frozen video-LLMs to learn frame selection that directly optimizes downstream caption quality. Additionally, we identify the gap between existing benchmark and human's cognition. Thus, we introduce ICH-CC built from carefully designed questions by annotators that reflect human-consistent understanding of video. Experiments indicate that LFS consistently improves detailed video captioning across two representative community benchmarks and ICH-CC, achieving up to 2.0% gains on VDC and over 4% gains on ICH-CC. Moreover, we observe that enhanced captions with LFS leads to improved performance on video question answering. Overall, LFS provides an effective and easy-to-integrate solution for detailed video captioning.
>
---
#### [new 025] SpatialV2A: Visual-Guided High-fidelity Spatial Audio Generation
- **分类: cs.CV**

- **简介: 该论文属于视频到音频生成任务，旨在解决现有方法在空间感知和沉浸感上的不足。通过构建首个支持空间感知的视频-双耳音频数据集，并提出视觉引导的空间音频生成框架，提升音频的空间真实性和层次感。**

- **链接: [https://arxiv.org/pdf/2601.15017v1](https://arxiv.org/pdf/2601.15017v1)**

> **作者:** Yanan Wang; Linjie Ren; Zihao Li; Junyi Wang; Tian Gan
>
> **摘要:** While video-to-audio generation has achieved remarkable progress in semantic and temporal alignment, most existing studies focus solely on these aspects, paying limited attention to the spatial perception and immersive quality of the synthesized audio. This limitation stems largely from current models' reliance on mono audio datasets, which lack the binaural spatial information needed to learn visual-to-spatial audio mappings. To address this gap, we introduce two key contributions: we construct BinauralVGGSound, the first large-scale video-binaural audio dataset designed to support spatially aware video-to-audio generation; and we propose a end-to-end spatial audio generation framework guided by visual cues, which explicitly models spatial features. Our framework incorporates a visual-guided audio spatialization module that ensures the generated audio exhibits realistic spatial attributes and layered spatial depth while maintaining semantic and temporal alignment. Experiments show that our approach substantially outperforms state-of-the-art models in spatial fidelity and delivers a more immersive auditory experience, without sacrificing temporal or semantic consistency. All datasets, code, and model checkpoints will be publicly released to facilitate future research.
>
---
#### [new 026] Real-Time Wildfire Localization on the NASA Autonomous Modular Sensor using Deep Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于 wildfire detection 任务，旨在解决高海拔火灾实时定位问题。通过构建多光谱数据集并训练深度学习模型，实现高效准确的火灾区域分割与定位。**

- **链接: [https://arxiv.org/pdf/2601.14475v1](https://arxiv.org/pdf/2601.14475v1)**

> **作者:** Yajvan Ravan; Aref Malek; Chester Dolph; Nikhil Behari
>
> **备注:** 16 pages, 9 figures, published at AIAA SciTech 2026
>
> **摘要:** High-altitude, multi-spectral, aerial imagery is scarce and expensive to acquire, yet it is necessary for algorithmic advances and application of machine learning models to high-impact problems such as wildfire detection. We introduce a human-annotated dataset from the NASA Autonomous Modular Sensor (AMS) using 12-channel, medium to high altitude (3 - 50 km) aerial wildfire images similar to those used in current US wildfire missions. Our dataset combines spectral data from 12 different channels, including infrared (IR), short-wave IR (SWIR), and thermal. We take imagery from 20 wildfire missions and randomly sample small patches to generate over 4000 images with high variability, including occlusions by smoke/clouds, easily-confused false positives, and nighttime imagery. We demonstrate results from a deep-learning model to automate the human-intensive process of fire perimeter determination. We train two deep neural networks, one for image classification and the other for pixel-level segmentation. The networks are combined into a unique real-time segmentation model to efficiently localize active wildfire on an incoming image feed. Our model achieves 96% classification accuracy, 74% Intersection-over-Union(IoU), and 84% recall surpassing past methods, including models trained on satellite data and classical color-rule algorithms. By leveraging a multi-spectral dataset, our model is able to detect active wildfire at nighttime and behind clouds, while distinguishing between false positives. We find that data from the SWIR, IR, and thermal bands is the most important to distinguish fire perimeters. Our code and dataset can be found here: https://github.com/nasa/Autonomous-Modular-Sensor-Wildfire-Segmentation/tree/main and https://drive.google.com/drive/folders/1-u4vs9rqwkwgdeeeoUhftCxrfe_4QPTn?=usp=drive_link
>
---
#### [new 027] GutenOCR: A Grounded Vision-Language Front-End for Documents
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出GutenOCR，属于文档OCR任务，解决传统OCR在文本定位与语义理解上的不足，通过微调视觉语言模型实现更精准的文本检测与查询。**

- **链接: [https://arxiv.org/pdf/2601.14490v1](https://arxiv.org/pdf/2601.14490v1)**

> **作者:** Hunter Heidenreich; Ben Elliott; Olivia Dinica; Yosheb Getachew
>
> **摘要:** GutenOCR is a family of grounded OCR front-ends obtained by fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B. The resulting single-checkpoint vision-language models expose reading, detection, and grounding through a unified, prompt-based interface. Trained on business documents, scientific articles, and synthetic grounding data, the models support full-page and localized reading with line- and paragraph-level bounding boxes and conditional ``where is x?'' queries. We introduce a grounded OCR evaluation protocol and show that GutenOCR-7B more than doubles the composite grounded OCR score of its Qwen2.5-VL-7B backbone on 10.5K held-out business and scientific pages (0.40 to 0.82). On Fox and OmniDocBench v1.5, our approach substantially improves region- and line-level OCR as well as text-detection recall, but reveals trade-offs in page-level linearization, color-guided OCR, and formula-heavy layouts.
>
---
#### [new 028] RayRoPE: Projective Ray Positional Encoding for Multi-view Attention
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多视角Transformer中的位置编码问题，提出RayRoPE方法，解决几何感知、SE(3)不变性和不确定性下的位置编码难题，提升新视角合成和立体深度估计效果。**

- **链接: [https://arxiv.org/pdf/2601.15275v1](https://arxiv.org/pdf/2601.15275v1)**

> **作者:** Yu Wu; Minsik Jeon; Jen-Hao Rick Chang; Oncel Tuzel; Shubham Tulsiani
>
> **备注:** Project page: https://rayrope.github.io/
>
> **摘要:** We study positional encodings for multi-view transformers that process tokens from a set of posed input images, and seek a mechanism that encodes patches uniquely, allows SE(3)-invariant attention with multi-frequency similarity, and can be adaptive to the geometry of the underlying scene. We find that prior (absolute or relative) encoding schemes for multi-view attention do not meet the above desiderata, and present RayRoPE to address this gap. RayRoPE represents patch positions based on associated rays but leverages a predicted point along the ray instead of the direction for a geometry-aware encoding. To achieve SE(3) invariance, RayRoPE computes query-frame projective coordinates for computing multi-frequency similarity. Lastly, as the 'predicted' 3D point along a ray may not be precise, RayRoPE presents a mechanism to analytically compute the expected position encoding under uncertainty. We validate RayRoPE on the tasks of novel-view synthesis and stereo depth estimation and show that it consistently improves over alternate position encoding schemes (e.g. 15% relative improvement on LPIPS in CO3D). We also show that RayRoPE can seamlessly incorporate RGB-D input, resulting in even larger gains over alternatives that cannot positionally encode this information.
>
---
#### [new 029] SimD3: A Synthetic drone Dataset with Payload and Bird Distractor Modeling for Robust Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决无人机检测中数据不足和干扰物相似的问题。提出SimD3数据集，包含多样化无人机和鸟类干扰物，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.14742v1](https://arxiv.org/pdf/2601.14742v1)**

> **作者:** Ami Pandat; Kanyala Muvva; Punna Rajasekhar; Gopika Vinod; Rohit Shukla
>
> **摘要:** Reliable drone detection is challenging due to limited annotated real-world data, large appearance variability, and the presence of visually similar distractors such as birds. To address these challenges, this paper introduces SimD3, a large-scale high-fidelity synthetic dataset designed for robust drone detection in complex aerial environments. Unlike existing synthetic drone datasets, SimD3 explicitly models drones with heterogeneous payloads, incorporates multiple bird species as realistic distractors, and leverages diverse Unreal Engine 5 environments with controlled weather, lighting, and flight trajectories captured using a 360 six-camera rig. Using SimD3, we conduct an extensive experimental evaluation within the YOLOv5 detection framework, including an attention-enhanced variant termed Yolov5m+C3b, where standard bottleneck-based C3 blocks are replaced with C3b modules. Models are evaluated on synthetic data, combined synthetic and real data, and multiple unseen real-world benchmarks to assess robustness and generalization. Experimental results show that SimD3 provides effective supervision for small-object drone detection and that Yolov5m+C3b consistently outperforms the baseline across in-domain and cross-dataset evaluations. These findings highlight the utility of SimD3 for training and benchmarking robust drone detection models under diverse and challenging conditions.
>
---
#### [new 030] Diffusion Epistemic Uncertainty with Asymmetric Learning for Diffusion-Generated Image Detection
- **分类: cs.CV; stat.ML**

- **简介: 该论文属于图像检测任务，旨在提升扩散生成图像的检测性能。针对重建误差中不确定性的影响，提出DEUA框架，结合Epistemic不确定性估计与非对称损失函数，增强检测效果。**

- **链接: [https://arxiv.org/pdf/2601.14625v1](https://arxiv.org/pdf/2601.14625v1)**

> **作者:** Yingsong Huang; Hui Guo; Jing Huang; Bing Bai; Qi Xiong
>
> **摘要:** The rapid progress of diffusion models highlights the growing need for detecting generated images. Previous research demonstrates that incorporating diffusion-based measurements, such as reconstruction error, can enhance the generalizability of detectors. However, ignoring the differing impacts of aleatoric and epistemic uncertainty on reconstruction error can undermine detection performance. Aleatoric uncertainty, arising from inherent data noise, creates ambiguity that impedes accurate detection of generated images. As it reflects random variations within the data (e.g., noise in natural textures), it does not help distinguish generated images. In contrast, epistemic uncertainty, which represents the model's lack of knowledge about unfamiliar patterns, supports detection. In this paper, we propose a novel framework, Diffusion Epistemic Uncertainty with Asymmetric Learning~(DEUA), for detecting diffusion-generated images. We introduce Diffusion Epistemic Uncertainty~(DEU) estimation via the Laplace approximation to assess the proximity of data to the manifold of diffusion-generated samples. Additionally, an asymmetric loss function is introduced to train a balanced classifier with larger margins, further enhancing generalizability. Extensive experiments on large-scale benchmarks validate the state-of-the-art performance of our method.
>
---
#### [new 031] BBoxMaskPose v2: Expanding Mutual Conditioning to 3D
- **分类: cs.CV**

- **简介: 该论文属于人体姿态估计任务，解决拥挤场景下2D/3D姿态估计问题。提出BMPv2模型，结合概率与掩码条件，提升姿态精度，尤其在OCHuman数据集上表现优异。**

- **链接: [https://arxiv.org/pdf/2601.15200v1](https://arxiv.org/pdf/2601.15200v1)**

> **作者:** Miroslav Purkrabek; Constantin Kolomiiets; Jiri Matas
>
> **备注:** GitHub repository: https://github.com/MiraPurkrabek/BBoxMaskPose/
>
> **摘要:** Most 2D human pose estimation benchmarks are nearly saturated, with the exception of crowded scenes. We introduce PMPose, a top-down 2D pose estimator that incorporates the probabilistic formulation and the mask-conditioning. PMPose improves crowded pose estimation without sacrificing performance on standard scenes. Building on this, we present BBoxMaskPose v2 (BMPv2) integrating PMPose and an enhanced SAM-based mask refinement module. BMPv2 surpasses state-of-the-art by 1.5 average precision (AP) points on COCO and 6 AP points on OCHuman, becoming the first method to exceed 50 AP on OCHuman. We demonstrate that BMP's 2D prompting of 3D model improves 3D pose estimation in crowded scenes and that advances in 2D pose quality directly benefit 3D estimation. Results on the new OCHuman-Pose dataset show that multi-person performance is more affected by pose prediction accuracy than by detection. The code, models, and data are available on https://MiraPurkrabek.github.io/BBox-Mask-Pose/.
>
---
#### [new 032] Anatomically Guided Latent Diffusion for Brain MRI Progression Modeling
- **分类: cs.CV**

- **简介: 该论文属于脑部MRI进展建模任务，旨在解决神经退行性疾病中结构变化预测的问题。提出AG-LDM模型，通过解剖引导实现更准确、一致的影像生成与进展模拟。**

- **链接: [https://arxiv.org/pdf/2601.14584v1](https://arxiv.org/pdf/2601.14584v1)**

> **作者:** Cheng Wan; Bahram Jafrasteh; Ehsan Adeli; Miaomiao Zhang; Qingyu Zhao
>
> **备注:** 10 pages, 5 figures, 3 tables
>
> **摘要:** Accurately modeling longitudinal brain MRI progression is crucial for understanding neurodegenerative diseases and predicting individualized structural changes. Existing state-of-the-art approaches, such as Brain Latent Progression (BrLP), often use multi-stage training pipelines with auxiliary conditioning modules but suffer from architectural complexity, suboptimal use of conditional clinical covariates, and limited guarantees of anatomical consistency. We propose Anatomically Guided Latent Diffusion Model (AG-LDM), a segmentation-guided framework that enforces anatomically consistent progression while substantially simplifying the training pipeline. AG-LDM conditions latent diffusion by directly fusing baseline anatomy, noisy follow-up states, and clinical covariates at the input level, a strategy that avoids auxiliary control networks by learning a unified, end-to-end model that represents both anatomy and progression. A lightweight 3D tissue segmentation model (WarpSeg) provides explicit anatomical supervision during both autoencoder fine-tuning and diffusion model training, ensuring consistent brain tissue boundaries and morphometric fidelity. Experiments on 31,713 ADNI longitudinal pairs and zero-shot evaluation on OASIS-3 demonstrate that AG-LDM matches or surpasses more complex diffusion models, achieving state-of-the-art image quality and 15-20\% reduction in volumetric errors in generated images. AG-LDM also exhibits markedly stronger utilization of temporal and clinical covariates (up to 31.5x higher sensitivity than BrLP) and generates biologically plausible counterfactual trajectories, accurately capturing hallmarks of Alzheimer's progression such as limbic atrophy and ventricular expansion. These results highlight AG-LDM as an efficient, anatomically grounded framework for reliable brain MRI progression modeling.
>
---
#### [new 033] DeepMoLM: Leveraging Visual and Geometric Structural Information for Molecule-Text Modeling
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文提出DeepMoLM，解决分子图像与文本生成任务中的3D结构建模问题，结合视觉与几何信息提升生成质量。**

- **链接: [https://arxiv.org/pdf/2601.14732v1](https://arxiv.org/pdf/2601.14732v1)**

> **作者:** Jing Lan; Hexiao Ding; Hongzhao Chen; Yufeng Jiang; Nga-Chun Ng; Gwing Kei Yip; Gerald W. Y. Cheng; Yunlin Mao; Jing Cai; Liang-ting Lin; Jung Sun Yoo
>
> **备注:** Under review
>
> **摘要:** AI models for drug discovery and chemical literature mining must interpret molecular images and generate outputs consistent with 3D geometry and stereochemistry. Most molecular language models rely on strings or graphs, while vision-language models often miss stereochemical details and struggle to map continuous 3D structures into discrete tokens. We propose DeepMoLM: Deep Molecular Language M odeling, a dual-view framework that grounds high-resolution molecular images in geometric invariants derived from molecular conformations. DeepMoLM preserves high-frequency evidence from 1024 $\times$ 1024 inputs, encodes conformer neighborhoods as discrete Extended 3-Dimensional Fingerprints, and fuses visual and geometric streams with cross-attention, enabling physically grounded generation without atom coordinates. DeepMoLM improves PubChem captioning with a 12.3% relative METEOR gain over the strongest generalist baseline while staying competitive with specialist methods. It produces valid numeric outputs for all property queries and attains MAE 13.64 g/mol on Molecular Weight and 37.89 on Complexity in the specialist setting. On ChEBI-20 description generation from images, it exceeds generalist baselines and matches state-of-the-art vision-language models. Code is available at https://github.com/1anj/DeepMoLM.
>
---
#### [new 034] ScenDi: 3D-to-2D Scene Diffusion Cascades for Urban Generation
- **分类: cs.CV**

- **简介: 该论文属于城市场景生成任务，旨在解决3D城市场景生成中细节不足与相机控制困难的问题。通过结合3D和2D扩散模型，提升生成场景的 realism 和可控性。**

- **链接: [https://arxiv.org/pdf/2601.15221v1](https://arxiv.org/pdf/2601.15221v1)**

> **作者:** Hanlei Guo; Jiahao Shao; Xinya Chen; Xiyang Tan; Sheng Miao; Yujun Shen; Yiyi Liao
>
> **摘要:** Recent advancements in 3D object generation using diffusion models have achieved remarkable success, but generating realistic 3D urban scenes remains challenging. Existing methods relying solely on 3D diffusion models tend to suffer a degradation in appearance details, while those utilizing only 2D diffusion models typically compromise camera controllability. To overcome this limitation, we propose ScenDi, a method for urban scene generation that integrates both 3D and 2D diffusion models. We first train a 3D latent diffusion model to generate 3D Gaussians, enabling the rendering of images at a relatively low resolution. To enable controllable synthesis, this 3DGS generation process can be optionally conditioned by specifying inputs such as 3d bounding boxes, road maps, or text prompts. Then, we train a 2D video diffusion model to enhance appearance details conditioned on rendered images from the 3D Gaussians. By leveraging the coarse 3D scene as guidance for 2D video diffusion, ScenDi generates desired scenes based on input conditions and successfully adheres to accurate camera trajectories. Experiments on two challenging real-world datasets, Waymo and KITTI-360, demonstrate the effectiveness of our approach.
>
---
#### [new 035] Vision-Based Natural Language Scene Understanding for Autonomous Driving: An Extended Dataset and a New Model for Traffic Scene Description Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于交通场景理解任务，旨在通过单目图像生成自然语言描述。提出新模型和数据集，解决场景描述生成问题。**

- **链接: [https://arxiv.org/pdf/2601.14438v1](https://arxiv.org/pdf/2601.14438v1)**

> **作者:** Danial Sadrian Zadeh; Otman A. Basir; Behzad Moshiri
>
> **备注:** Under review at Computer Vision and Image Understanding (submitted July 25, 2025)
>
> **摘要:** Traffic scene understanding is essential for enabling autonomous vehicles to accurately perceive and interpret their environment, thereby ensuring safe navigation. This paper presents a novel framework that transforms a single frontal-view camera image into a concise natural language description, effectively capturing spatial layouts, semantic relationships, and driving-relevant cues. The proposed model leverages a hybrid attention mechanism to enhance spatial and semantic feature extraction and integrates these features to generate contextually rich and detailed scene descriptions. To address the limited availability of specialized datasets in this domain, a new dataset derived from the BDD100K dataset has been developed, with comprehensive guidelines provided for its construction. Furthermore, the study offers an in-depth discussion of relevant evaluation metrics, identifying the most appropriate measures for this task. Extensive quantitative evaluations using metrics such as CIDEr and SPICE, complemented by human judgment assessments, demonstrate that the proposed model achieves strong performance and effectively fulfills its intended objectives on the newly developed dataset.
>
---
#### [new 036] BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文研究promptable分割模型在真实场景下的鲁棒性问题，提出BREPS方法生成对抗性边界框以评估模型稳定性。任务属于模型鲁棒性评估。**

- **链接: [https://arxiv.org/pdf/2601.15123v1](https://arxiv.org/pdf/2601.15123v1)**

> **作者:** Andrey Moskalenko; Danil Kuznetsov; Irina Dudko; Anastasiia Iasakova; Nikita Boldyrev; Denis Shepelev; Andrei Spiridonov; Andrey Kuznetsov; Vlad Shakhuro
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Promptable segmentation models such as SAM have established a powerful paradigm, enabling strong generalization to unseen objects and domains with minimal user input, including points, bounding boxes, and text prompts. Among these, bounding boxes stand out as particularly effective, often outperforming points while significantly reducing annotation costs. However, current training and evaluation protocols typically rely on synthetic prompts generated through simple heuristics, offering limited insight into real-world robustness. In this paper, we investigate the robustness of promptable segmentation models to natural variations in bounding box prompts. First, we conduct a controlled user study and collect thousands of real bounding box annotations. Our analysis reveals substantial variability in segmentation quality across users for the same model and instance, indicating that SAM-like models are highly sensitive to natural prompt noise. Then, since exhaustive testing of all possible user inputs is computationally prohibitive, we reformulate robustness evaluation as a white-box optimization problem over the bounding box prompt space. We introduce BREPS, a method for generating adversarial bounding boxes that minimize or maximize segmentation error while adhering to naturalness constraints. Finally, we benchmark state-of-the-art models across 10 datasets, spanning everyday scenes to medical imaging. Code - https://github.com/emb-ai/BREPS.
>
---
#### [new 037] TempViz: On the Evaluation of Temporal Knowledge in Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文属于文本到图像生成任务，旨在评估模型中的时间知识。研究提出TempViz数据集，分析五种模型在五个时间类别中的表现，发现其时间理解能力较弱，需进一步研究。**

- **链接: [https://arxiv.org/pdf/2601.14951v1](https://arxiv.org/pdf/2601.14951v1)**

> **作者:** Carolin Holtermann; Nina Krebs; Anne Lauscher
>
> **摘要:** Time alters the visual appearance of entities in our world, like objects, places, and animals. Thus, for accurately generating contextually-relevant images, knowledge and reasoning about time can be crucial (e.g., for generating a landscape in spring vs. in winter). Yet, although substantial work exists on understanding and improving temporal knowledge in natural language processing, research on how temporal phenomena appear and are handled in text-to-image (T2I) models remains scarce. We address this gap with TempViz, the first data set to holistically evaluate temporal knowledge in image generation, consisting of 7.9k prompts and more than 600 reference images. Using TempViz, we study the capabilities of five T2I models across five temporal knowledge categories. Human evaluation shows that temporal competence is generally weak, with no model exceeding 75% accuracy across categories. Towards larger-scale studies, we also examine automated evaluation methods, comparing several established approaches against human judgments. However, none of these approaches provides a reliable assessment of temporal cues - further indicating the pressing need for future research on temporal knowledge in T2I.
>
---
#### [new 038] CityCube: Benchmarking Cross-view Spatial Reasoning on Vision-Language Models in Urban Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型的跨视角空间推理任务，旨在解决城市环境中空间理解不足的问题。作者构建了CityCube基准，评估VLMs在多种视角下的表现。**

- **链接: [https://arxiv.org/pdf/2601.14339v1](https://arxiv.org/pdf/2601.14339v1)**

> **作者:** Haotian Xu; Yue Hu; Zhengqiu Zhu; Chen Gao; Ziyou Wang; Junreng Rao; Wenhao Lu; Weishi Li; Quanjun Yin; Yong Li
>
> **摘要:** Cross-view spatial reasoning is essential for embodied AI, underpinning spatial understanding, mental simulation and planning in complex environments. Existing benchmarks primarily emphasize indoor or street settings, overlooking the unique challenges of open-ended urban spaces characterized by rich semantics, complex geometries, and view variations. To address this, we introduce CityCube, a systematic benchmark designed to probe cross-view reasoning capabilities of current VLMs in urban settings. CityCube integrates four viewpoint dynamics to mimic camera movements and spans a wide spectrum of perspectives from multiple platforms, e.g., vehicles, drones and satellites. For a comprehensive assessment, it features 5,022 meticulously annotated multi-view QA pairs categorized into five cognitive dimensions and three spatial relation expressions. A comprehensive evaluation of 33 VLMs reveals a significant performance disparity with humans: even large-scale models struggle to exceed 54.1% accuracy, remaining 34.2% below human performance. By contrast, small-scale fine-tuned VLMs achieve over 60.0% accuracy, highlighting the necessity of our benchmark. Further analyses indicate the task correlations and fundamental cognitive disparity between VLMs and human-like reasoning.
>
---
#### [new 039] Breaking the accuracy-resource dilemma: a lightweight adaptive video inference enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频推理任务，旨在解决模型精度与资源消耗的矛盾。通过引入模糊控制器，动态调整模型规模，提升推理效率与效果。**

- **链接: [https://arxiv.org/pdf/2601.14568v1](https://arxiv.org/pdf/2601.14568v1)**

> **作者:** Wei Ma; Shaowu Chen; Junjie Ye; Peichang Zhang; Lei Huang
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Existing video inference (VI) enhancement methods typically aim to improve performance by scaling up model sizes and employing sophisticated network architectures. While these approaches demonstrated state-of-the-art performance, they often overlooked the trade-off of resource efficiency and inference effectiveness, leading to inefficient resource utilization and suboptimal inference performance. To address this problem, a fuzzy controller (FC-r) is developed based on key system parameters and inference-related metrics. Guided by the FC-r, a VI enhancement framework is proposed, where the spatiotemporal correlation of targets across adjacent video frames is leveraged. Given the real-time resource conditions of the target device, the framework can dynamically switch between models of varying scales during VI. Experimental results demonstrate that the proposed method effectively achieves a balance between resource utilization and inference performance.
>
---
#### [new 040] GAT-NeRF: Geometry-Aware-Transformer Enhanced Neural Radiance Fields for High-Fidelity 4D Facial Avatars
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于4D面部动画重建任务，解决单目视频中高保真面部细节捕捉问题。提出GAT-NeRF框架，结合Transformer与NeRF，提升动态面部特征建模能力。**

- **链接: [https://arxiv.org/pdf/2601.14875v1](https://arxiv.org/pdf/2601.14875v1)**

> **作者:** Zhe Chang; Haodong Jin; Ying Sun; Yan Song; Hui Yu
>
> **摘要:** High-fidelity 4D dynamic facial avatar reconstruction from monocular video is a critical yet challenging task, driven by increasing demands for immersive virtual human applications. While Neural Radiance Fields (NeRF) have advanced scene representation, their capacity to capture high-frequency facial details, such as dynamic wrinkles and subtle textures from information-constrained monocular streams, requires significant enhancement. To tackle this challenge, we propose a novel hybrid neural radiance field framework, called Geometry-Aware-Transformer Enhanced NeRF (GAT-NeRF) for high-fidelity and controllable 4D facial avatar reconstruction, which integrates the Transformer mechanism into the NeRF pipeline. GAT-NeRF synergistically combines a coordinate-aligned Multilayer Perceptron (MLP) with a lightweight Transformer module, termed as Geometry-Aware-Transformer (GAT) due to its processing of multi-modal inputs containing explicit geometric priors. The GAT module is enabled by fusing multi-modal input features, including 3D spatial coordinates, 3D Morphable Model (3DMM) expression parameters, and learnable latent codes to effectively learn and enhance feature representations pertinent to fine-grained geometry. The Transformer's effective feature learning capabilities are leveraged to significantly augment the modeling of complex local facial patterns like dynamic wrinkles and acne scars. Comprehensive experiments unequivocally demonstrate GAT-NeRF's state-of-the-art performance in visual fidelity and high-frequency detail recovery, forging new pathways for creating realistic dynamic digital humans for multimedia applications.
>
---
#### [new 041] Graph Recognition via Subgraph Prediction
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉图识别任务，旨在解决图像中图形结构识别困难的问题。提出GraSP方法，通过子图预测实现跨任务的通用图识别。**

- **链接: [https://arxiv.org/pdf/2601.15133v1](https://arxiv.org/pdf/2601.15133v1)**

> **作者:** André Eberhard; Gerhard Neumann; Pascal Friederich
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Despite tremendous improvements in tasks such as image classification, object detection, and segmentation, the recognition of visual relationships, commonly modeled as the extraction of a graph from an image, remains a challenging task. We believe that this mainly stems from the fact that there is no canonical way to approach the visual graph recognition task. Most existing solutions are specific to a problem and cannot be transferred between different contexts out-of-the box, even though the conceptual problem remains the same. With broad applicability and simplicity in mind, in this paper we develop a method, \textbf{Gra}ph Recognition via \textbf{S}ubgraph \textbf{P}rediction (\textbf{GraSP}), for recognizing graphs in images. We show across several synthetic benchmarks and one real-world application that our method works with a set of diverse types of graphs and their drawings, and can be transferred between tasks without task-specific modifications, paving the way to a more unified framework for visual graph recognition.
>
---
#### [new 042] APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping
- **分类: cs.CV**

- **简介: 该论文属于人脸交换任务，解决真实标签缺失导致的属性保留与身份转移难题。提出APPLE框架，通过伪标签和属性感知机制提升效果。**

- **链接: [https://arxiv.org/pdf/2601.15288v1](https://arxiv.org/pdf/2601.15288v1)**

> **作者:** Jiwon Kang; Yeji Choi; JoungBin Lee; Wooseok Jang; Jinhyeok Choi; Taekeun Kang; Yongjae Park; Myungin Kim; Seungryong Kim
>
> **备注:** Project Page: https://cvlab-kaist.github.io/APPLE/
>
> **摘要:** Face swapping aims to transfer the identity of a source face onto a target face while preserving target-specific attributes such as pose, expression, lighting, skin tone, and makeup. However, since real ground truth for face swapping is unavailable, achieving both accurate identity transfer and high-quality attribute preservation remains challenging. In addition, recent diffusion-based approaches attempt to improve visual fidelity through conditional inpainting on masked target images, but the masked condition removes crucial appearance cues of target, resulting in plausible yet misaligned attributes. To address these limitations, we propose APPLE (Attribute-Preserving Pseudo-Labeling), a diffusion-based teacher-student framework that enhances attribute fidelity through attribute-aware pseudo-label supervision. We reformulate face swapping as a conditional deblurring task to more faithfully preserve target-specific attributes such as lighting, skin tone, and makeup. In addition, we introduce an attribute-aware inversion scheme to further improve detailed attribute preservation. Through an elaborate attribute-preserving design for teacher learning, APPLE produces high-quality pseudo triplets that explicitly provide the student with direct face-swapping supervision. Overall, APPLE achieves state-of-the-art performance in terms of attribute preservation and identity transfer, producing more photorealistic and target-faithful results.
>
---
#### [new 043] Does medical specialization of VLMs enhance discriminative power?: A comprehensive investigation through feature distribution analysis
- **分类: cs.CV**

- **简介: 该论文属于医学视觉语言模型研究，旨在探讨医疗专用化是否提升特征区分能力。通过分析特征分布，比较医疗与非医疗模型，发现文本编码器优化更为关键。**

- **链接: [https://arxiv.org/pdf/2601.14774v1](https://arxiv.org/pdf/2601.14774v1)**

> **作者:** Keita Takeda; Tomoya Sakai
>
> **备注:** A short version paper of this research has been accepted for The IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** This study investigates the feature representations produced by publicly available open source medical vision-language models (VLMs). While medical VLMs are expected to capture diagnostically relevant features, their learned representations remain underexplored, and standard evaluations like classification accuracy do not fully reveal if they acquire truly discriminative, lesion-specific features. Understanding these representations is crucial for revealing medical image structures and improving downstream tasks in medical image analysis. This study aims to investigate the feature distributions learned by medical VLMs and evaluate the impact of medical specialization. We analyze the feature distribution of multiple image modalities extracted by some representative medical VLMs across lesion classification datasets on multiple modalities. These distributions were compared them with non-medical VLMs to assess the domain-specific medical training. Our experiments showed that medical VLMs can extract discriminative features that are effective for medical classification tasks. Moreover, it was found that non-medical VLMs with recent improvement with contextual enrichment such as LLM2CLIP produce more refined feature representations. Our results imply that enhancing text encoder is more crucial than training intensively on medical images when developing medical VLMs. Notably, non-medical models are particularly vulnerable to biases introduced by overlaied text strings on images. These findings underscore the need for careful consideration on model selection according to downstream tasks besides potential risks in inference due to background biases such as textual information in images.
>
---
#### [new 044] LURE: Latent Space Unblocking for Multi-Concept Reawakening in Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于概念重激活任务，旨在解决扩散模型中擦除概念仍被唤醒的问题。提出LURE方法，通过重构潜在空间和优化采样轨迹实现多概念高保真重激活。**

- **链接: [https://arxiv.org/pdf/2601.14330v1](https://arxiv.org/pdf/2601.14330v1)**

> **作者:** Mengyu Sun; Ziyuan Yang; Andrew Beng Jin Teoh; Junxu Liu; Haibo Hu; Yi Zhang
>
> **摘要:** Concept erasure aims to suppress sensitive content in diffusion models, but recent studies show that erased concepts can still be reawakened, revealing vulnerabilities in erasure methods. Existing reawakening methods mainly rely on prompt-level optimization to manipulate sampling trajectories, neglecting other generative factors, which limits a comprehensive understanding of the underlying dynamics. In this paper, we model the generation process as an implicit function to enable a comprehensive theoretical analysis of multiple factors, including text conditions, model parameters, and latent states. We theoretically show that perturbing each factor can reawaken erased concepts. Building on this insight, we propose a novel concept reawakening method: Latent space Unblocking for concept REawakening (LURE), which reawakens erased concepts by reconstructing the latent space and guiding the sampling trajectory. Specifically, our semantic re-binding mechanism reconstructs the latent space by aligning denoising predictions with target distributions to reestablish severed text-visual associations. However, in multi-concept scenarios, naive reconstruction can cause gradient conflicts and feature entanglement. To address this, we introduce Gradient Field Orthogonalization, which enforces feature orthogonality to prevent mutual interference. Additionally, our Latent Semantic Identification-Guided Sampling (LSIS) ensures stability of the reawakening process via posterior density verification. Extensive experiments demonstrate that LURE enables simultaneous, high-fidelity reawakening of multiple erased concepts across diverse erasure tasks and methods.
>
---
#### [new 045] Large-Scale Label Quality Assessment for Medical Segmentation via a Vision-Language Judge and Synthetic Data
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于医学分割任务，解决标签质量不一致问题。提出SegAE模型，自动评估142个解剖结构的标签质量，提升数据效率与训练性能。**

- **链接: [https://arxiv.org/pdf/2601.14406v1](https://arxiv.org/pdf/2601.14406v1)**

> **作者:** Yixiong Chen; Zongwei Zhou; Wenxuan Li; Alan Yuille
>
> **备注:** ISBI 2026 accepted
>
> **摘要:** Large-scale medical segmentation datasets often combine manual and pseudo-labels of uneven quality, which can compromise training and evaluation. Low-quality labels may hamper performance and make the model training less robust. To address this issue, we propose SegAE (Segmentation Assessment Engine), a lightweight vision-language model (VLM) that automatically predicts label quality across 142 anatomical structures. Trained on over four million image-label pairs with quality scores, SegAE achieves a high correlation coefficient of 0.902 with ground-truth Dice similarity and evaluates a 3D mask in 0.06s. SegAE shows several practical benefits: (I) Our analysis reveals widespread low-quality labeling across public datasets; (II) SegAE improves data efficiency and training performance in active and semi-supervised learning, reducing dataset annotation cost by one-third and quality-checking time by 70% per label. This tool provides a simple and effective solution for quality control in large-scale medical segmentation datasets. The dataset, model weights, and codes are released at https://github.com/Schuture/SegAE.
>
---
#### [new 046] Differential Privacy Image Generation with Reconstruction Loss and Noise Injection Using an Error Feedback SGD
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于隐私保护生成任务，旨在解决数据隐私与实用性的平衡问题。提出一种结合误差反馈SGD、重建损失和噪声注入的框架，提升图像生成质量与隐私保障。**

- **链接: [https://arxiv.org/pdf/2601.15061v1](https://arxiv.org/pdf/2601.15061v1)**

> **作者:** Qiwei Ma; Jun Zhang
>
> **摘要:** Traditional data masking techniques such as anonymization cannot achieve the expected privacy protection while ensuring data utility for privacy-preserving machine learning. Synthetic data plays an increasingly important role as it generates a large number of training samples and prevents information leakage in real data. The existing methods suffer from the repeating trade-off processes between privacy and utility. We propose a novel framework for differential privacy generation, which employs an Error Feedback Stochastic Gradient Descent(EFSGD) method and introduces a reconstruction loss and noise injection mechanism into the training process. We generate images with higher quality and usability under the same privacy budget as the related work. Extensive experiments demonstrate the effectiveness and generalization of our proposed framework for both grayscale and RGB images. We achieve state-of-the-art results over almost all metrics on three benchmarks: MNIST, Fashion-MNIST, and CelebA.
>
---
#### [new 047] Gaussian Based Adaptive Multi-Modal 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D语义占用预测任务，解决自动驾驶中长尾安全挑战。通过融合相机与LiDAR数据，提出一种基于高斯的自适应多模态模型，提升预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.14448v1](https://arxiv.org/pdf/2601.14448v1)**

> **作者:** A. Enes Doruk
>
> **备注:** Master Thesis
>
> **摘要:** The sparse object detection paradigm shift towards dense 3D semantic occupancy prediction is necessary for dealing with long-tail safety challenges for autonomous vehicles. Nonetheless, the current voxelization methods commonly suffer from excessive computation complexity demands, where the fusion process is brittle, static, and breaks down under dynamic environmental settings. To this end, this research work enhances a novel Gaussian-based adaptive camera-LiDAR multimodal 3D occupancy prediction model that seamlessly bridges the semantic strengths of camera modality with the geometric strengths of LiDAR modality through a memory-efficient 3D Gaussian model. The proposed solution has four key components: (1) LiDAR Depth Feature Aggregation (LDFA), where depth-wise deformable sampling is employed for dealing with geometric sparsity, (2) Entropy-Based Feature Smoothing, where cross-entropy is employed for handling domain-specific noise, (3) Adaptive Camera-LiDAR Fusion, where dynamic recalibration of sensor outputs is performed based on model outputs, and (4) Gauss-Mamba Head that uses Selective State Space Models for global context decoding that enjoys linear computation complexity.
>
---
#### [new 048] Multimodal system for skin cancer detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于皮肤癌检测任务，旨在解决传统方法依赖专业设备的问题。通过融合照片图像与元数据，构建多模态系统提升检测准确性。**

- **链接: [https://arxiv.org/pdf/2601.14822v1](https://arxiv.org/pdf/2601.14822v1)**

> **作者:** Volodymyr Sydorskyi; Igor Krashenyi; Oleksii Yakubenko
>
> **备注:** Accepted to System research and information technologies
>
> **摘要:** Melanoma detection is vital for early diagnosis and effective treatment. While deep learning models on dermoscopic images have shown promise, they require specialized equipment, limiting their use in broader clinical settings. This study introduces a multi-modal melanoma detection system using conventional photo images, making it more accessible and versatile. Our system integrates image data with tabular metadata, such as patient demographics and lesion characteristics, to improve detection accuracy. It employs a multi-modal neural network combining image and metadata processing and supports a two-step model for cases with or without metadata. A three-stage pipeline further refines predictions by boosting algorithms and enhancing performance. To address the challenges of a highly imbalanced dataset, specific techniques were implemented to ensure robust training. An ablation study evaluated recent vision architectures, boosting algorithms, and loss functions, achieving a peak Partial ROC AUC of 0.18068 (0.2 maximum) and top-15 retrieval sensitivity of 0.78371. Results demonstrate that integrating photo images with metadata in a structured, multi-stage pipeline yields significant performance improvements. This system advances melanoma detection by providing a scalable, equipment-independent solution suitable for diverse healthcare environments, bridging the gap between specialized and general clinical practices.
>
---
#### [new 049] UBATrack: Spatio-Temporal State Space Model for General Multi-Modal Tracking
- **分类: cs.CV**

- **简介: 该论文属于多模态目标跟踪任务，解决现有方法忽视时空线索的问题。提出UBATrack框架，结合时空Mamba适配器和动态多模态特征混合模块，提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2601.14799v1](https://arxiv.org/pdf/2601.14799v1)**

> **作者:** Qihua Liang; Liang Chen; Yaozong Zheng; Jian Nong; Zhiyi Mo; Bineng Zhong
>
> **摘要:** Multi-modal object tracking has attracted considerable attention by integrating multiple complementary inputs (e.g., thermal, depth, and event data) to achieve outstanding performance. Although current general-purpose multi-modal trackers primarily unify various modal tracking tasks (i.e., RGB-Thermal infrared, RGB-Depth or RGB-Event tracking) through prompt learning, they still overlook the effective capture of spatio-temporal cues. In this work, we introduce a novel multi-modal tracking framework based on a mamba-style state space model, termed UBATrack. Our UBATrack comprises two simple yet effective modules: a Spatio-temporal Mamba Adapter (STMA) and a Dynamic Multi-modal Feature Mixer. The former leverages Mamba's long-sequence modeling capability to jointly model cross-modal dependencies and spatio-temporal visual cues in an adapter-tuning manner. The latter further enhances multi-modal representation capacity across multiple feature dimensions to improve tracking robustness. In this way, UBATrack eliminates the need for costly full-parameter fine-tuning, thereby improving the training efficiency of multi-modal tracking algorithms. Experiments show that UBATrack outperforms state-of-the-art methods on RGB-T, RGB-D, and RGB-E tracking benchmarks, achieving outstanding results on the LasHeR, RGBT234, RGBT210, DepthTrack, VOT-RGBD22, and VisEvent datasets.
>
---
#### [new 050] MTFlow: Time-Conditioned Flow Matching for Microtubule Segmentation in Noisy Microscopy Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于微管分割任务，解决噪声显微图像中微管结构分割难题。提出MTFlow模型，通过时间条件流匹配实现更精确的分割。**

- **链接: [https://arxiv.org/pdf/2601.14841v1](https://arxiv.org/pdf/2601.14841v1)**

> **作者:** Sidi Mohamed Sid El Moctar; Achraf Ait Laydi; Yousef El Mourabit; Hélène Bouvrais
>
> **备注:** Accepted for presentation at ISBI 2026
>
> **摘要:** Microtubules are cytoskeletal filaments that play essential roles in many cellular processes and are key therapeutic targets in several diseases. Accurate segmentation of microtubule networks is critical for studying their organization and dynamics but remains challenging due to filament curvature, dense crossings, and image noise. We present MTFlow, a novel time-conditioned flow-matching model for microtubule segmentation. Unlike conventional U-Net variants that predict masks in a single pass, MTFlow learns vector fields that iteratively transport noisy masks toward the ground truth, enabling interpretable, trajectory-based refinement. Our architecture combines a U-Net backbone with temporal embeddings, allowing the model to capture the dynamics of uncertainty resolution along filament boundaries. We trained and evaluated MTFlow on synthetic and real microtubule datasets and assessed its generalization capability on public biomedical datasets of curvilinear structures such as retinal blood vessels and nerves. MTFlow achieves competitive segmentation accuracy comparable to state-of-the-art models, offering a powerful and time-efficient tool for filamentous structure analysis with more precise annotations than manual or semi-automatic approaches.
>
---
#### [new 051] Symmetry Informative and Agnostic Feature Disentanglement for 3D Shapes
- **分类: cs.CV**

- **简介: 该论文属于3D形状分析任务，旨在解决对称信息与语义信息融合不足的问题。提出一种同时具备对称信息和对称无关的特征解耦方法，并改进特征鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.14804v1](https://arxiv.org/pdf/2601.14804v1)**

> **作者:** Tobias Weißberg; Weikang Wang; Paul Roetzer; Nafie El Amrani; Florian Bernard
>
> **备注:** Accepted at 3DV 2026
>
> **摘要:** Shape descriptors, i.e., per-vertex features of 3D meshes or point clouds, are fundamental to shape analysis. Historically, various handcrafted geometry-aware descriptors and feature refinement techniques have been proposed. Recently, several studies have initiated a new research direction by leveraging features from image foundation models to create semantics-aware descriptors, demonstrating advantages across tasks like shape matching, editing, and segmentation. Symmetry, another key concept in shape analysis, has also attracted increasing attention. Consequently, constructing symmetry-aware shape descriptors is a natural progression. Although the recent method $χ$ (Wang et al., 2025) successfully extracted symmetry-informative features from semantic-aware descriptors, its features are only one-dimensional, neglecting other valuable semantic information. Furthermore, the extracted symmetry-informative feature is usually noisy and yields small misclassified patches. To address these gaps, we propose a feature disentanglement approach which is simultaneously symmetry informative and symmetry agnostic. Further, we propose a feature refinement technique to improve the robustness of predicted symmetry informative features. Extensive experiments, including intrinsic symmetry detection, left/right classification, and shape matching, demonstrate the effectiveness of our proposed framework compared to various state-of-the-art methods, both qualitatively and quantitatively.
>
---
#### [new 052] U-Harmony: Enhancing Joint Training for Segmentation Models with Universal Harmonization
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，解决多机构数据异质性带来的模型泛化与领域知识丢失问题。提出U-Harmony方法，实现跨域特征统一与模态适应。**

- **链接: [https://arxiv.org/pdf/2601.14605v1](https://arxiv.org/pdf/2601.14605v1)**

> **作者:** Weiwei Ma; Xiaobing Yu; Peijie Qiu; Jin Yang; Pan Xiao; Xiaoqi Zhao; Xiaofeng Liu; Tomo Miyazaki; Shinichiro Omachi; Yongsong Huang
>
> **摘要:** In clinical practice, medical segmentation datasets are often limited and heterogeneous, with variations in modalities, protocols, and anatomical targets across institutions. Existing deep learning models struggle to jointly learn from such diverse data, often sacrificing either generalization or domain-specific knowledge. To overcome these challenges, we propose a joint training method called Universal Harmonization (U-Harmony), which can be integrated into deep learning-based architectures with a domain-gated head, enabling a single segmentation model to learn from heterogeneous datasets simultaneously. By integrating U-Harmony, our approach sequentially normalizes and then denormalizes feature distributions to mitigate domain-specific variations while preserving original dataset-specific knowledge. More appealingly, our framework also supports universal modality adaptation, allowing the seamless learning of new imaging modalities and anatomical classes. Extensive experiments on cross-institutional brain lesion datasets demonstrate the effectiveness of our approach, establishing a new benchmark for robust and adaptable 3D medical image segmentation models in real-world clinical settings.
>
---
#### [new 053] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，解决 streaming 视频实时处理中的性能与内存问题。提出 HERMES 架构，通过层级 KV 缓存实现高效准确的视频流理解。**

- **链接: [https://arxiv.org/pdf/2601.14724v1](https://arxiv.org/pdf/2601.14724v1)**

> **作者:** Haowei Zhang; Shudong Yang; Jinlan Fu; See-Kiong Ng; Xipeng Qiu
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated significant improvement in offline video understanding. However, extending these capabilities to streaming video inputs, remains challenging, as existing models struggle to simultaneously maintain stable understanding performance, real-time responses, and low GPU memory overhead. To address this challenge, we propose HERMES, a novel training-free architecture for real-time and accurate understanding of video streams. Based on a mechanistic attention investigation, we conceptualize KV cache as a hierarchical memory framework that encapsulates video information across multiple granularities. During inference, HERMES reuses a compact KV cache, enabling efficient streaming understanding under resource constraints. Notably, HERMES requires no auxiliary computations upon the arrival of user queries, thereby guaranteeing real-time responses for continuous video stream interactions, which achieves 10$\times$ faster TTFT compared to prior SOTA. Even when reducing video tokens by up to 68% compared with uniform sampling, HERMES achieves superior or comparable accuracy across all benchmarks, with up to 11.4% gains on streaming datasets.
>
---
#### [new 054] 3D Space as a Scratchpad for Editable Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决图像生成中空间关系不准确的问题。通过引入3D空间作为可编辑的思维板，提升图像生成的几何和语义一致性。**

- **链接: [https://arxiv.org/pdf/2601.14602v1](https://arxiv.org/pdf/2601.14602v1)**

> **作者:** Oindrila Saha; Vojtech Krs; Radomir Mech; Subhransu Maji; Matheus Gadelha; Kevin Blackburn-Matzen
>
> **摘要:** Recent progress in large language models (LLMs) has shown that reasoning improves when intermediate thoughts are externalized into explicit workspaces, such as chain-of-thought traces or tool-augmented reasoning. Yet, visual language models (VLMs) lack an analogous mechanism for spatial reasoning, limiting their ability to generate images that accurately reflect geometric relations, object identities, and compositional intent. We introduce the concept of a spatial scratchpad -- a 3D reasoning substrate that bridges linguistic intent and image synthesis. Given a text prompt, our framework parses subjects and background elements, instantiates them as editable 3D meshes, and employs agentic scene planning for placement, orientation, and viewpoint selection. The resulting 3D arrangement is rendered back into the image domain with identity-preserving cues, enabling the VLM to generate spatially consistent and visually coherent outputs. Unlike prior 2D layout-based methods, our approach supports intuitive 3D edits that propagate reliably into final images. Empirically, it achieves a 32% improvement in text alignment on GenAI-Bench, demonstrating the benefit of explicit 3D reasoning for precise, controllable image generation. Our results highlight a new paradigm for vision-language models that deliberate not only in language, but also in space. Code and visualizations at https://oindrilasaha.github.io/3DScratchpad/
>
---
#### [new 055] Scribble-Supervised Medical Image Segmentation with Dynamic Teacher Switching and Hierarchical Consistency
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决弱监督下标注稀疏导致的边界学习困难问题。提出SDT-Net框架，通过动态教师切换和层次一致性机制提升分割精度。**

- **链接: [https://arxiv.org/pdf/2601.14563v1](https://arxiv.org/pdf/2601.14563v1)**

> **作者:** Thanh-Huy Nguyen; Hoang-Loc Cao; Dat T. Chung; Mai-Anh Vu; Thanh-Minh Nguyen; Minh Le; Phat K. Huynh; Ulas Bagci
>
> **摘要:** Scribble-supervised methods have emerged to mitigate the prohibitive annotation burden in medical image segmentation. However, the inherent sparsity of these annotations introduces significant ambiguity, which results in noisy pseudo-label propagation and hinders the learning of robust anatomical boundaries. To address this challenge, we propose SDT-Net, a novel dual-teacher, single-student framework designed to maximize supervision quality from these weak signals. Our method features a Dynamic Teacher Switching (DTS) module to adaptively select the most reliable teacher. This selected teacher then guides the student via two synergistic mechanisms: high-confidence pseudo-labels, refined by a Pick Reliable Pixels (PRP) mechanism, and multi-level feature alignment, enforced by a Hierarchical Consistency (HiCo) module. Extensive experiments on the ACDC and MSCMRseg datasets demonstrate that SDT-Net achieves state-of-the-art performance, producing more accurate and anatomically plausible segmentation.
>
---
#### [new 056] Reconstruction-Anchored Diffusion Model for Text-to-Motion Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到动作生成任务，旨在解决运动扩散模型的表征差距和误差传播问题。提出RAM模型，通过运动潜在空间和重构误差引导提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.14788v1](https://arxiv.org/pdf/2601.14788v1)**

> **作者:** Yifei Liu; Changxing Ding; Ling Guo; Huaiguang Jiang; Qiong Cao
>
> **摘要:** Diffusion models have seen widespread adoption for text-driven human motion generation and related tasks due to their impressive generative capabilities and flexibility. However, current motion diffusion models face two major limitations: a representational gap caused by pre-trained text encoders that lack motion-specific information, and error propagation during the iterative denoising process. This paper introduces Reconstruction-Anchored Diffusion Model (RAM) to address these challenges. First, RAM leverages a motion latent space as intermediate supervision for text-to-motion generation. To this end, RAM co-trains a motion reconstruction branch with two key objective functions: self-regularization to enhance the discrimination of the motion space and motion-centric latent alignment to enable accurate mapping from text to the motion latent space. Second, we propose Reconstructive Error Guidance (REG), a testing-stage guidance mechanism that exploits the diffusion model's inherent self-correction ability to mitigate error propagation. At each denoising step, REG uses the motion reconstruction branch to reconstruct the previous estimate, reproducing the prior error patterns. By amplifying the residual between the current prediction and the reconstructed estimate, REG highlights the improvements in the current prediction. Extensive experiments demonstrate that RAM achieves significant improvements and state-of-the-art performance. Our code will be released.
>
---
#### [new 057] M2I2HA: A Multi-modal Object Detection Method Based on Intra- and Inter-Modal Hypergraph Attention
- **分类: cs.CV**

- **简介: 该论文属于多模态目标检测任务，旨在解决跨模态对齐与信息提取难题。提出M2I2HA网络，通过超图注意力机制实现多模态特征融合与增强。**

- **链接: [https://arxiv.org/pdf/2601.14776v1](https://arxiv.org/pdf/2601.14776v1)**

> **作者:** Xiaofan Yang; Yubin Liu; Wei Pan; Guoqing Chu; Junming Zhang; Jie Zhao; Zhuoqi Man; Xuanming Cao
>
> **备注:** 43 pages, 13 figures
>
> **摘要:** Recent advances in multi-modal detection have significantly improved detection accuracy in challenging environments (e.g., low light, overexposure). By integrating RGB with modalities such as thermal and depth, multi-modal fusion increases data redundancy and system robustness. However, significant challenges remain in effectively extracting task-relevant information both within and across modalities, as well as in achieving precise cross-modal alignment. While CNNs excel at feature extraction, they are limited by constrained receptive fields, strong inductive biases, and difficulty in capturing long-range dependencies. Transformer-based models offer global context but suffer from quadratic computational complexity and are confined to pairwise correlation modeling. Mamba and other State Space Models (SSMs), on the other hand, are hindered by their sequential scanning mechanism, which flattens 2D spatial structures into 1D sequences, disrupting topological relationships and limiting the modeling of complex higher-order dependencies. To address these issues, we propose a multi-modal perception network based on hypergraph theory called M2I2HA. Our architecture includes an Intra-Hypergraph Enhancement module to capture global many-to-many high-order relationships within each modality, and an Inter-Hypergraph Fusion module to align, enhance, and fuse cross-modal features by bridging configuration and spatial gaps between data sources. We further introduce a M2-FullPAD module to enable adaptive multi-level fusion of multi-modal enhanced features within the network, meanwhile enhancing data distribution and flow across the architecture. Extensive object detection experiments on multiple public datasets against baselines demonstrate that M2I2HA achieves state-of-the-art performance in multi-modal object detection tasks.
>
---
#### [new 058] Learning Consistent Taxonomic Classification through Hierarchical Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉分类任务，旨在解决VLM在层级分类中一致性不足的问题。通过两阶段框架VL-Taxon提升分类准确性和层级一致性。**

- **链接: [https://arxiv.org/pdf/2601.14610v1](https://arxiv.org/pdf/2601.14610v1)**

> **作者:** Zhenghong Li; Kecheng Zheng; Haibin Ling
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** While Vision-Language Models (VLMs) excel at visual understanding, they often fail to grasp hierarchical knowledge. This leads to common errors where VLMs misclassify coarser taxonomic levels even when correctly identifying the most specific level (leaf level). Existing approaches largely overlook this issue by failing to model hierarchical reasoning. To address this gap, we propose VL-Taxon, a two-stage, hierarchy-based reasoning framework designed to improve both leaf-level accuracy and hierarchical consistency in taxonomic classification. The first stage employs a top-down process to enhance leaf-level classification accuracy. The second stage then leverages this accurate leaf-level output to ensure consistency throughout the entire taxonomic hierarchy. Each stage is initially trained with supervised fine-tuning to instill taxonomy knowledge, followed by reinforcement learning to refine the model's reasoning and generalization capabilities. Extensive experiments reveal a remarkable result: our VL-Taxon framework, implemented on the Qwen2.5-VL-7B model, outperforms its original 72B counterpart by over 10% in both leaf-level and hierarchical consistency accuracy on average on the iNaturalist-2021 dataset. Notably, this significant gain was achieved by fine-tuning on just a small subset of data, without relying on any examples generated by other VLMs.
>
---
#### [new 059] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于动作生成任务，旨在解决传统方法在姿态和时间控制上的不足。提出SOS脚本和SOSControl框架，实现更精确的运动控制与生成。**

- **链接: [https://arxiv.org/pdf/2601.14258v1](https://arxiv.org/pdf/2601.14258v1)**

> **作者:** Ho Yin Au; Junkun Jiang; Jie Chen
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Traditional text-to-motion frameworks often lack precise control, and existing approaches based on joint keyframe locations provide only positional guidance, making it challenging and unintuitive to specify body part orientations and motion timing. To address these limitations, we introduce the Salient Orientation Symbolic (SOS) script, a programmable symbolic framework for specifying body part orientations and motion timing at keyframes. We further propose an automatic SOS extraction pipeline that employs temporally-constrained agglomerative clustering for frame saliency detection and a Saliency-based Masking Scheme (SMS) to generate sparse, interpretable SOS scripts directly from motion data. Moreover, we present the SOSControl framework, which treats the available orientation symbols in the sparse SOS script as salient and prioritizes satisfying these constraints during motion generation. By incorporating SMS-based data augmentation and gradient-based iterative optimization, the framework enhances alignment with user-specified constraints. Additionally, it employs a ControlNet-based ACTOR-PAE Decoder to ensure smooth and natural motion outputs. Extensive experiments demonstrate that the SOS extraction pipeline generates human-interpretable scripts with symbolic annotations at salient keyframes, while the SOSControl framework outperforms existing baselines in motion quality, controllability, and generalizability with respect to motion timing and body part orientation control.
>
---
#### [new 060] Unified Multi-Dataset Training for TBPS
- **分类: cs.CV**

- **简介: 该论文属于文本行人检索任务，旨在解决多数据集训练效果不佳的问题。提出Scale-TBPS方法，通过统一数据集和可扩展身份学习，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.14978v1](https://arxiv.org/pdf/2601.14978v1)**

> **作者:** Nilanjana Chatterjee; Sidharatha Garg; A V Subramanyam; Brejesh Lall
>
> **摘要:** Text-Based Person Search (TBPS) has seen significant progress with vision-language models (VLMs), yet it remains constrained by limited training data and the fact that VLMs are not inherently pre-trained for pedestrian-centric recognition. Existing TBPS methods therefore rely on dataset-centric fine-tuning to handle distribution shift, resulting in multiple independently trained models for different datasets. While synthetic data can increase the scale needed to fine-tune VLMs, it does not eliminate dataset-specific adaptation. This motivates a fundamental question: can we train a single unified TBPS model across multiple datasets? We show that naive joint training over all datasets remains sub-optimal because current training paradigms do not scale to a large number of unique person identities and are vulnerable to noisy image-text pairs. To address these challenges, we propose Scale-TBPS with two contributions: (i) a noise-aware unified dataset curation strategy that cohesively merges diverse TBPS datasets; and (ii) a scalable discriminative identity learning framework that remains effective under a large number of unique identities. Extensive experiments on CUHK-PEDES, ICFG-PEDES, RSTPReid, IIITD-20K, and UFine6926 demonstrate that a single Scale-TBPS model outperforms dataset-centric optimized models and naive joint training.
>
---
#### [new 061] Intelligent Power Grid Design Review via Active Perception-Enabled Multimodal Large Language Models
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于电力工程设计审查任务，旨在解决高分辨率图纸中设计错误识别困难的问题。通过三阶段框架，利用多模态大语言模型提升审查准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2601.14261v1](https://arxiv.org/pdf/2601.14261v1)**

> **作者:** Taoliang Tan; Chengwei Ma; Zhen Tian; Zhao Lin; Dongdong Li; Si Shi
>
> **摘要:** The intelligent review of power grid engineering design drawings is crucial for power system safety. However, current automated systems struggle with ultra-high-resolution drawings due to high computational demands, information loss, and a lack of holistic semantic understanding for design error identification. This paper proposes a novel three-stage framework for intelligent power grid drawing review, driven by pre-trained Multimodal Large Language Models (MLLMs) through advanced prompt engineering. Mimicking the human expert review process, the first stage leverages an MLLM for global semantic understanding to intelligently propose domain-specific semantic regions from a low-resolution overview. The second stage then performs high-resolution, fine-grained recognition within these proposed regions, acquiring detailed information with associated confidence scores. In the final stage, a comprehensive decision-making module integrates these confidence-aware results to accurately diagnose design errors and provide a reliability assessment. Preliminary results on real-world power grid drawings demonstrate our approach significantly enhances MLLM's ability to grasp macroscopic semantic information and pinpoint design errors, showing improved defect discovery accuracy and greater reliability in review judgments compared to traditional passive MLLM inference. This research offers a novel, prompt-driven paradigm for intelligent and reliable power grid drawing review.
>
---
#### [new 062] Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network
- **分类: cs.CV**

- **简介: 该论文属于服装模拟任务，解决传统方法计算成本高和GNN跨分辨率泛化差的问题。提出Pb4U-GNet框架，通过动态传播深度和几何感知更新提升模型适应不同分辨率的能力。**

- **链接: [https://arxiv.org/pdf/2601.15110v1](https://arxiv.org/pdf/2601.15110v1)**

> **作者:** Aoran Liu; Kun Hu; Clinton Ansun Mo; Qiuxia Wu; Wenxiong Kang; Zhiyong Wang
>
> **备注:** Camera-ready version accepted at AAAI 2026
>
> **摘要:** Garment simulation is fundamental to various applications in computer vision and graphics, from virtual try-on to digital human modelling. However, conventional physics-based methods remain computationally expensive, hindering their application in time-sensitive scenarios. While graph neural networks (GNNs) offer promising acceleration, existing approaches exhibit poor cross-resolution generalisation, demonstrating significant performance degradation on higher-resolution meshes beyond the training distribution. This stems from two key factors: (1) existing GNNs employ fixed message-passing depth that fails to adapt information aggregation to mesh density variation, and (2) vertex-wise displacement magnitudes are inherently resolution-dependent in garment simulation. To address these issues, we introduce Propagation-before-Update Graph Network (Pb4U-GNet), a resolution-adaptive framework that decouples message propagation from feature updates. Pb4U-GNet incorporates two key mechanisms: (1) dynamic propagation depth control, adjusting message-passing iterations based on mesh resolution, and (2) geometry-aware update scaling, which scales predictions according to local mesh characteristics. Extensive experiments show that even trained solely on low-resolution meshes, Pb4U-GNet exhibits strong generalisability across diverse mesh resolutions, addressing a fundamental challenge in neural garment simulation.
>
---
#### [new 063] Synthetic Data Augmentation for Multi-Task Chinese Porcelain Classification: A Stable Diffusion Approach
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多任务中国瓷器分类任务，旨在解决考古数据稀缺问题。通过Stable Diffusion生成合成数据增强真实数据，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.14791v1](https://arxiv.org/pdf/2601.14791v1)**

> **作者:** Ziyao Ling; Silvia Mirri; Paola Salomoni; Giovanni Delnevo
>
> **摘要:** The scarcity of training data presents a fundamental challenge in applying deep learning to archaeological artifact classification, particularly for the rare types of Chinese porcelain. This study investigates whether synthetic images generated through Stable Diffusion with Low-Rank Adaptation (LoRA) can effectively augment limited real datasets for multi-task CNN-based porcelain classification. Using MobileNetV3 with transfer learning, we conducted controlled experiments comparing models trained on pure real data against those trained on mixed real-synthetic datasets (95:5 and 90:10 ratios) across four classification tasks: dynasty, glaze, kiln and type identification. Results demonstrate task-specific benefits: type classification showed the most substantial improvement (5.5\% F1-macro increase with 90:10 ratio), while dynasty and kiln tasks exhibited modest gains (3-4\%), suggesting that synthetic augmentation effectiveness depends on the alignment between generated features and task-relevant visual signatures. Our work contributes practical guidelines for deploying generative AI in archaeological research, demonstrating both the potential and limitations of synthetic data when archaeological authenticity must be balanced with data diversity.
>
---
#### [new 064] PROGRESSLM: Towards Progress Reasoning in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型的任务进度推理问题，提出Progress-Bench基准和ProgressLM-45K数据集，探索两种推理方法，发现多数模型在任务进度估计上表现不佳。**

- **链接: [https://arxiv.org/pdf/2601.15224v1](https://arxiv.org/pdf/2601.15224v1)**

> **作者:** Jianshu Zhang; Chengxuan Qian; Haosen Sun; Haoran Lu; Dingcheng Wang; Letian Xue; Han Liu
>
> **备注:** Website: https://progresslm.github.io/ProgressLM/
>
> **摘要:** Estimating task progress requires reasoning over long-horizon dynamics rather than recognizing static visual content. While modern Vision-Language Models (VLMs) excel at describing what is visible, it remains unclear whether they can infer how far a task has progressed from partial observations. To this end, we introduce Progress-Bench, a benchmark for systematically evaluating progress reasoning in VLMs. Beyond benchmarking, we further explore a human-inspired two-stage progress reasoning paradigm through both training-free prompting and training-based approach based on curated dataset ProgressLM-45K. Experiments on 14 VLMs show that most models are not yet ready for task progress estimation, exhibiting sensitivity to demonstration modality and viewpoint changes, as well as poor handling of unanswerable cases. While training-free prompting that enforces structured progress reasoning yields limited and model-dependent gains, the training-based ProgressLM-3B achieves consistent improvements even at a small model scale, despite being trained on a task set fully disjoint from the evaluation tasks. Further analyses reveal characteristic error patterns and clarify when and why progress reasoning succeeds or fails.
>
---
#### [new 065] Walk through Paintings: Egocentric World Models from Internet Priors
- **分类: cs.CV**

- **简介: 该论文提出EgoWM，将预训练视频模型转化为动作条件世界模型，解决可控未来预测问题，适用于不同机器人系统，提升物理一致性与推理效率。**

- **链接: [https://arxiv.org/pdf/2601.15284v1](https://arxiv.org/pdf/2601.15284v1)**

> **作者:** Anurag Bagchi; Zhipeng Bao; Homanga Bharadhwaj; Yu-Xiong Wang; Pavel Tokmakov; Martial Hebert
>
> **摘要:** What if a video generation model could not only imagine a plausible future, but the correct one, accurately reflecting how the world changes with each action? We address this question by presenting the Egocentric World Model (EgoWM), a simple, architecture-agnostic method that transforms any pretrained video diffusion model into an action-conditioned world model, enabling controllable future prediction. Rather than training from scratch, we repurpose the rich world priors of Internet-scale video models and inject motor commands through lightweight conditioning layers. This allows the model to follow actions faithfully while preserving realism and strong generalization. Our approach scales naturally across embodiments and action spaces, ranging from 3-DoF mobile robots to 25-DoF humanoids, where predicting egocentric joint-angle-driven dynamics is substantially more challenging. The model produces coherent rollouts for both navigation and manipulation tasks, requiring only modest fine-tuning. To evaluate physical correctness independently of visual appearance, we introduce the Structural Consistency Score (SCS), which measures whether stable scene elements evolve consistently with the provided actions. EgoWM improves SCS by up to 80 percent over prior state-of-the-art navigation world models, while achieving up to six times lower inference latency and robust generalization to unseen environments, including navigation inside paintings.
>
---
#### [new 066] Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning
- **分类: cs.CV**

- **简介: 该论文属于 hateful video detection 任务，旨在解决检测仇恨视频的可靠性与可解释性问题。提出 MARS 框架，通过多阶段对抗推理实现无需训练的准确检测与可理解的解释。**

- **链接: [https://arxiv.org/pdf/2601.15115v1](https://arxiv.org/pdf/2601.15115v1)**

> **作者:** Shuonan Yang; Yuchen Zhang; Zeyu Fu
>
> **备注:** Accepted at ICASSP 2026. \c{opyright} 2026 IEEE. This is the author accepted manuscript. The final published version will be available via IEEE Xplore
>
> **摘要:** Hateful videos pose serious risks by amplifying discrimination, inciting violence, and undermining online safety. Existing training-based hateful video detection methods are constrained by limited training data and lack of interpretability, while directly prompting large vision-language models often struggle to deliver reliable hate detection. To address these challenges, this paper introduces MARS, a training-free Multi-stage Adversarial ReaSoning framework that enables reliable and interpretable hateful content detection. MARS begins with the objective description of video content, establishing a neutral foundation for subsequent analysis. Building on this, it develops evidence-based reasoning that supports potential hateful interpretations, while in parallel incorporating counter-evidence reasoning to capture plausible non-hateful perspectives. Finally, these perspectives are synthesized into a conclusive and explainable decision. Extensive evaluation on two real-world datasets shows that MARS achieves up to 10% improvement under certain backbones and settings compared to other training-free approaches and outperforms state-of-the-art training-based methods on one dataset. In addition, MARS produces human-understandable justifications, thereby supporting compliance oversight and enhancing the transparency of content moderation workflows. The code is available at https://github.com/Multimodal-Intelligence-Lab-MIL/MARS.
>
---
#### [new 067] Iterative Refinement Improves Compositional Image Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂提示下的生成问题。通过迭代精炼策略，提升生成图像的准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2601.15286v1](https://arxiv.org/pdf/2601.15286v1)**

> **作者:** Shantanu Jaiswal; Mihir Prabhudesai; Nikash Bhardwaj; Zheyang Qin; Amir Zadeh; Chuan Li; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project webpage: https://iterative-img-gen.github.io/
>
> **摘要:** Text-to-image (T2I) models have achieved remarkable progress, yet they continue to struggle with complex prompts that require simultaneously handling multiple objects, relations, and attributes. Existing inference-time strategies, such as parallel sampling with verifiers or simply increasing denoising steps, can improve prompt alignment but remain inadequate for richly compositional settings where many constraints must be satisfied. Inspired by the success of chain-of-thought reasoning in large language models, we propose an iterative test-time strategy in which a T2I model progressively refines its generations across multiple steps, guided by feedback from a vision-language model as the critic in the loop. Our approach is simple, requires no external tools or priors, and can be flexibly applied to a wide range of image generators and vision-language models. Empirically, we demonstrate consistent gains on image generation across benchmarks: a 16.9% improvement in all-correct rate on ConceptMix (k=7), a 13.8% improvement on T2I-CompBench (3D-Spatial category) and a 12.5% improvement on Visual Jenga scene decomposition compared to compute-matched parallel sampling. Beyond quantitative gains, iterative refinement produces more faithful generations by decomposing complex prompts into sequential corrections, with human evaluators preferring our method 58.7% of the time over 41.3% for the parallel baseline. Together, these findings highlight iterative self-correction as a broadly applicable principle for compositional image generation. Results and visualizations are available at https://iterative-img-gen.github.io/
>
---
#### [new 068] PAS-Mamba: Phase-Amplitude-Spatial State Space Model for MRI Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于MRI重建任务，旨在解决频率域信息建模不足的问题。通过分离相位与幅度，结合空间特征，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2601.14530v1](https://arxiv.org/pdf/2601.14530v1)**

> **作者:** Xiaoyan Kui; Zijie Fan; Zexin Ji; Qinsong Li; Hao Xu; Weixin Si; Haodong Xu; Beiji Zou
>
> **摘要:** Joint feature modeling in both the spatial and frequency domains has become a mainstream approach in MRI reconstruction. However, existing methods generally treat the frequency domain as a whole, neglecting the differences in the information carried by its internal components. According to Fourier transform theory, phase and amplitude represent different types of information in the image. Our spectrum swapping experiments show that magnitude mainly reflects pixel-level intensity, while phase predominantly governs image structure. To prevent interference between phase and magnitude feature learning caused by unified frequency-domain modeling, we propose the Phase-Amplitude-Spatial State Space Model (PAS-Mamba) for MRI Reconstruction, a framework that decouples phase and magnitude modeling in the frequency domain and combines it with image-domain features for better reconstruction. In the image domain, LocalMamba preserves spatial locality to sharpen fine anatomical details. In frequency domain, we disentangle amplitude and phase into two specialized branches to avoid representational coupling. To respect the concentric geometry of frequency information, we propose Circular Frequency Domain Scanning (CFDS) to serialize features from low to high frequencies. Finally, a Dual-Domain Complementary Fusion Module (DDCFM) adaptively fuses amplitude phase representations and enables bidirectional exchange between frequency and image domains, delivering superior reconstruction. Extensive experiments on the IXI and fastMRI knee datasets show that PAS-Mamba consistently outperforms state of the art reconstruction methods.
>
---
#### [new 069] A Computer Vision Hybrid Approach: CNN and Transformer Models for Accurate Alzheimer's Detection from Brain MRI Scans
- **分类: cs.CV**

- **简介: 该论文属于阿尔茨海默病分类任务，旨在提高MRI扫描的准确诊断。通过比较CNN、Transformer及混合模型，提出Evan_V2混合模型，显著提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.15202v1](https://arxiv.org/pdf/2601.15202v1)**

> **作者:** Md Mahmudul Hoque; Shuvo Karmaker; Md. Hadi Al-Amin; Md Modabberul Islam; Jisun Junayed; Farha Ulfat Mahi
>
> **摘要:** Early and accurate classification of Alzheimers disease (AD) from brain MRI scans is essential for timely clinical intervention and improved patient outcomes. This study presents a comprehensive comparative analysis of five CNN architectures (EfficientNetB0, ResNet50, DenseNet201, MobileNetV3, VGG16), five Transformer-based models (ViT, ConvTransformer, PatchTransformer, MLP-Mixer, SimpleTransformer), and a proposed hybrid model named Evan_V2. All models were evaluated on a four-class AD classification task comprising Mild Dementia, Moderate Dementia, Non-Demented, and Very Mild Dementia categories. Experimental findings show that CNN architectures consistently achieved strong performance, with ResNet50 attaining 98.83% accuracy. Transformer models demonstrated competitive generalization capabilities, with ViT achieving the highest accuracy among them at 95.38%. However, individual Transformer variants exhibited greater class-specific instability. The proposed Evan_V2 hybrid model, which integrates outputs from ten CNN and Transformer architectures through feature-level fusion, achieved the best overall performance with 99.99% accuracy, 0.9989 F1-score, and 0.9968 ROC AUC. Confusion matrix analysis further confirmed that Evan_V2 substantially reduced misclassification across all dementia stages, outperforming every standalone model. These findings highlight the potential of hybrid ensemble strategies in producing highly reliable and clinically meaningful diagnostic tools for Alzheimers disease classification.
>
---
#### [new 070] LocBAM: Advancing 3D Patch-Based Image Segmentation by Integrating Location Contex
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像分割任务，解决patch方法忽略位置信息的问题。提出LocBAM注意力机制，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2601.14802v1](https://arxiv.org/pdf/2601.14802v1)**

> **作者:** Donnate Hooft; Stefan M. Fischer; Cosmin Bercea; Jan C. Peeken; Julia A. Schnabel
>
> **备注:** Accepted at ISBI 2026
>
> **摘要:** Patch-based methods are widely used in 3D medical image segmentation to address memory constraints in processing high-resolution volumetric data. However, these approaches often neglect the patch's location within the global volume, which can limit segmentation performance when anatomical context is important. In this paper, we investigate the role of location context in patch-based 3D segmentation and propose a novel attention mechanism, LocBAM, that explicitly processes spatial information. Experiments on BTCV, AMOS22, and KiTS23 demonstrate that incorporating location context stabilizes training and improves segmentation performance, particularly under low patch-to-volume coverage where global context is missing. Furthermore, LocBAM consistently outperforms classical coordinate encoding via CoordConv. Code is publicly available at https://github.com/compai-lab/2026-ISBI-hooft
>
---
#### [new 071] FunCineForge: A Unified Dataset Toolkit and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FunCineForge，解决电影配音任务中的数据不足和模型性能问题，构建了首个中文配音数据集，并设计了多场景配音模型。**

- **链接: [https://arxiv.org/pdf/2601.14777v1](https://arxiv.org/pdf/2601.14777v1)**

> **作者:** Jiaxuan Liu; Yang Xiang; Han Zhao; Xiangang Li; Zhenhua Ling
>
> **摘要:** Movie dubbing is the task of synthesizing speech from scripts conditioned on video scenes, requiring accurate lip sync, faithful timbre transfer, and proper modeling of character identity and emotion. However, existing methods face two major limitations: (1) high-quality multimodal dubbing datasets are limited in scale, suffer from high word error rates, contain sparse annotations, rely on costly manual labeling, and are restricted to monologue scenes, all of which hinder effective model training; (2) existing dubbing models rely solely on the lip region to learn audio-visual alignment, which limits their applicability to complex live-action cinematic scenes, and exhibit suboptimal performance in lip sync, speech quality, and emotional expressiveness. To address these issues, we propose FunCineForge, which comprises an end-to-end production pipeline for large-scale dubbing datasets and an MLLM-based dubbing model designed for diverse cinematic scenes. Using the pipeline, we construct the first Chinese television dubbing dataset with rich annotations, and demonstrate the high quality of these data. Experiments across monologue, narration, dialogue, and multi-speaker scenes show that our dubbing model consistently outperforms SOTA methods in audio quality, lip sync, timbre transfer, and instruction following. Code and demos are available at https://anonymous.4open.science/w/FunCineForge.
>
---
#### [new 072] Three-dimensional visualization of X-ray micro-CT with large-scale datasets: Efficiency and accuracy for real-time interaction
- **分类: cs.CV**

- **简介: 该论文属于三维可视化任务，解决大尺度数据下精度与效率的平衡问题，分析CT重建和体渲染方法，提出高效准确的缺陷检测方案。**

- **链接: [https://arxiv.org/pdf/2601.15098v1](https://arxiv.org/pdf/2601.15098v1)**

> **作者:** Yipeng Yin; Rao Yao; Qingying Li; Dazhong Wang; Hong Zhou; Zhijun Fang; Jianing Chen; Longjie Qian; Mingyue Wu
>
> **备注:** Page1-37
>
> **摘要:** As Micro-CT technology continues to refine its characterization of material microstructures, industrial CT ultra-precision inspection is generating increasingly large datasets, necessitating solutions to the trade-off between accuracy and efficiency in the 3D characterization of defects during ultra-precise detection. This article provides a unique perspective on recent advances in accurate and efficient 3D visualization using Micro-CT, tracing its evolution from medical imaging to industrial non-destructive testing (NDT). Among the numerous CT reconstruction and volume rendering methods, this article selectively reviews and analyzes approaches that balance accuracy and efficiency, offering a comprehensive analysis to help researchers quickly grasp highly efficient and accurate 3D reconstruction methods for microscopic features. By comparing the principles of computed tomography with advancements in microstructural technology, this article examines the evolution of CT reconstruction algorithms from analytical methods to deep learning techniques, as well as improvements in volume rendering algorithms, acceleration, and data reduction. Additionally, it explores advanced lighting models for high-accuracy, photorealistic, and efficient volume rendering. Furthermore, this article envisions potential directions in CT reconstruction and volume rendering. It aims to guide future research in quickly selecting efficient and precise methods and developing new ideas and approaches for real-time online monitoring of internal material defects through virtual-physical interaction, for applying digital twin model to structural health monitoring (SHM).
>
---
#### [new 073] XD-MAP: Cross-Modal Domain Adaptation using Semantic Parametric Mapping
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文提出XD-MAP方法，解决跨模态领域自适应问题，将图像数据知识迁移至LiDAR，无需人工标注，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2601.14477v1](https://arxiv.org/pdf/2601.14477v1)**

> **作者:** Frank Bieder; Hendrik Königshof; Haohao Hu; Fabian Immel; Yinzhe Shen; Jan-Hendrik Pauls; Christoph Stiller
>
> **摘要:** Until open-world foundation models match the performance of specialized approaches, the effectiveness of deep learning models remains heavily dependent on dataset availability. Training data must align not only with the target object categories but also with the sensor characteristics and modalities. To bridge the gap between available datasets and deployment domains, domain adaptation strategies are widely used. In this work, we propose a novel approach to transferring sensor-specific knowledge from an image dataset to LiDAR, an entirely different sensing domain. Our method XD-MAP leverages detections from a neural network on camera images to create a semantic parametric map. The map elements are modeled to produce pseudo labels in the target domain without any manual annotation effort. Unlike previous domain transfer approaches, our method does not require direct overlap between sensors and enables extending the angular perception range from a front-view camera to a full 360 view. On our large-scale road feature dataset, XD-MAP outperforms single shot baseline approaches by +19.5 mIoU for 2D semantic segmentation, +19.5 PQth for 2D panoptic segmentation, and +32.3 mIoU in 3D semantic segmentation. The results demonstrate the effectiveness of our approach achieving strong performance on LiDAR data without any manual labeling.
>
---
#### [new 074] Towards Understanding Best Practices for Quantization of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究多模态大模型的量化方法，探讨不同量化策略对模型性能的影响，旨在提升模型部署效率。**

- **链接: [https://arxiv.org/pdf/2601.15287v1](https://arxiv.org/pdf/2601.15287v1)**

> **作者:** Gautom Das; Vincent La; Ethan Lau; Abhinav Shrivastava; Matthew Gwilliam
>
> **备注:** 15 pages, 12 figures, 1 table
>
> **摘要:** Large language models (LLMs) deliver impressive results for a variety of tasks, but state-of-the-art systems require fast GPUs with large amounts of memory. To reduce both the memory and latency of these systems, practitioners quantize their learned parameters, typically at half precision. A growing body of research focuses on preserving the model performance with more aggressive bit widths, and some work has been done to apply these strategies to other models, like vision transformers. In our study we investigate how a variety of quantization methods, including state-of-the-art GPTQ and AWQ, can be applied effectively to multimodal pipelines comprised of vision models, language models, and their connectors. We address how performance on captioning, retrieval, and question answering can be affected by bit width, quantization method, and which portion of the pipeline the quantization is used for. Results reveal that ViT and LLM exhibit comparable importance in model performance, despite significant differences in parameter size, and that lower-bit quantization of the LLM achieves high accuracy at reduced bits per weight (bpw). These findings provide practical insights for efficient deployment of MLLMs and highlight the value of exploration for understanding component sensitivities in multimodal models. Our code is available at https://github.com/gautomdas/mmq.
>
---
#### [new 075] From Volumes to Slices: Computationally Efficient Contrastive Learning for Sequential Abdominal CT Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决标注数据不足的问题。通过2D-VoCo方法进行自监督预训练，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.14593v1](https://arxiv.org/pdf/2601.14593v1)**

> **作者:** Po-Kai Chiu; Hung-Hsuan Chen
>
> **摘要:** The requirement for expert annotations limits the effectiveness of deep learning for medical image analysis. Although 3D self-supervised methods like volume contrast learning (VoCo) are powerful and partially address the labeling scarcity issue, their high computational cost and memory consumption are barriers. We propose 2D-VoCo, an efficient adaptation of the VoCo framework for slice-level self-supervised pre-training that learns spatial-semantic features from unlabeled 2D CT slices via contrastive learning. The pre-trained CNN backbone is then integrated into a CNN-LSTM architecture to classify multi-organ injuries. In the RSNA 2023 Abdominal Trauma dataset, 2D-VoCo pre-training significantly improves mAP, precision, recall, and RSNA score over training from scratch. Our framework provides a practical method to reduce the dependency on labeled data and enhance model performance in clinical CT analysis. We release the code for reproducibility. https://github.com/tkz05/2D-VoCo-CT-Classifier
>
---
#### [new 076] Federated Transformer-GNN for Privacy-Preserving Brain Tumor Localization with Modality-Level Explainability
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于脑肿瘤定位任务，解决多机构数据隐私问题。提出联邦学习框架，结合Transformer-GNN模型，在保护数据隐私的前提下提升模型性能，并通过注意力机制实现模态级解释性。**

- **链接: [https://arxiv.org/pdf/2601.15042v1](https://arxiv.org/pdf/2601.15042v1)**

> **作者:** Andrea Protani; Riccardo Taiello; Marc Molina Van Den Bosch; Luigi Serio
>
> **摘要:** Deep learning models for brain tumor analysis require large and diverse datasets that are often siloed across healthcare institutions due to privacy regulations. We present a federated learning framework for brain tumor localization that enables multi-institutional collaboration without sharing sensitive patient data. Our method extends a hybrid Transformer-Graph Neural Network architecture derived from prior decoder-free supervoxel GNNs and is deployed within CAFEIN\textsuperscript{\textregistered}, CERN's federated learning platform designed for healthcare environments. We provide an explainability analysis through Transformer attention mechanisms that reveals which MRI modalities drive the model predictions. Experiments on the BraTS dataset demonstrate a key finding: while isolated training on individual client data triggers early stopping well before reaching full training capacity, federated learning enables continued model improvement by leveraging distributed data, ultimately matching centralized performance. This result provides strong justification for federated learning when dealing with complex tasks and high-dimensional input data, as aggregating knowledge from multiple institutions significantly benefits the learning process. Our explainability analysis, validated through rigorous statistical testing on the full test set (paired t-tests with Bonferroni correction), reveals that deeper network layers significantly increase attention to T2 and FLAIR modalities ($p<0.001$, Cohen's $d$=1.50), aligning with clinical practice.
>
---
#### [new 077] Context Patch Fusion With Class Token Enhancement for Weakly Supervised Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于弱监督语义分割任务，旨在解决因依赖图像级标签导致的语义模糊和分割精度低的问题。提出CPF-CTE框架，通过上下文融合与类标记增强提升分割效果。**

- **链接: [https://arxiv.org/pdf/2601.14718v1](https://arxiv.org/pdf/2601.14718v1)**

> **作者:** Yiyang Fu; Hui Li; Wangyu Wu
>
> **摘要:** Weakly Supervised Semantic Segmentation (WSSS), which relies only on image-level labels, has attracted significant attention for its cost-effectiveness and scalability. Existing methods mainly enhance inter-class distinctions and employ data augmentation to mitigate semantic ambiguity and reduce spurious activations. However, they often neglect the complex contextual dependencies among image patches, resulting in incomplete local representations and limited segmentation accuracy. To address these issues, we propose the Context Patch Fusion with Class Token Enhancement (CPF-CTE) framework, which exploits contextual relations among patches to enrich feature representations and improve segmentation. At its core, the Contextual-Fusion Bidirectional Long Short-Term Memory (CF-BiLSTM) module captures spatial dependencies between patches and enables bidirectional information flow, yielding a more comprehensive understanding of spatial correlations. This strengthens feature learning and segmentation robustness. Moreover, we introduce learnable class tokens that dynamically encode and refine class-specific semantics, enhancing discriminative capability. By effectively integrating spatial and semantic cues, CPF-CTE produces richer and more accurate representations of image content. Extensive experiments on PASCAL VOC 2012 and MS COCO 2014 validate that CPF-CTE consistently surpasses prior WSSS methods.
>
---
#### [new 078] ReinPath: A Multimodal Reinforcement Learning Approach for Pathology
- **分类: cs.CV**

- **简介: 该论文属于病理学与人工智能交叉任务，旨在提升病理分析的可解释性。针对现有方法可解释性不足的问题，提出一种多模态强化学习方法，构建高质量数据集并设计语义奖励策略，以增强模型推理能力。**

- **链接: [https://arxiv.org/pdf/2601.14757v1](https://arxiv.org/pdf/2601.14757v1)**

> **作者:** Kangcheng Zhou; Jun Jiang; Qing Zhang; Shuang Zheng; Qingli Li; Shugong Xu
>
> **摘要:** Interpretability is significant in computational pathology, leading to the development of multimodal information integration from histopathological image and corresponding text data.However, existing multimodal methods have limited interpretability due to the lack of high-quality dataset that support explicit reasoning and inference and simple reasoning process.To address the above problems, we introduce a novel multimodal pathology large language model with strong reasoning capabilities.To improve the generation of accurate and contextually relevant textual descriptions, we design a semantic reward strategy integrated with group relative policy optimization.We construct a high-quality pathology visual question answering (VQA) dataset, specifically designed to support complex reasoning tasks.Comprehensive experiments conducted on this dataset demonstrate that our method outperforms state-of-the-art methods, even when trained with only 20% of the data.Our method also achieves comparable performance on downstream zero-shot image classification task compared with CLIP.
>
---
#### [new 079] Rethinking Video Generation Model for the Embodied World
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文聚焦视频生成任务，旨在提升机器人视频的物理真实性。针对数据不足和评估缺失的问题，提出RBench基准和RoVid-X数据集，推动 embodied AI 发展。**

- **链接: [https://arxiv.org/pdf/2601.15282v1](https://arxiv.org/pdf/2601.15282v1)**

> **作者:** Yufan Deng; Zilin Pan; Hongyu Zhang; Xiaojie Li; Ruoqing Hu; Yufei Ding; Yiming Zou; Yan Zeng; Daquan Zhou
>
> **备注:** Github: https://github.com/DAGroup-PKU/ReVidgen/ Project website: https://dagroup-pku.github.io/ReVidgen.github.io/
>
> **摘要:** Video generation models have significantly advanced embodied intelligence, unlocking new possibilities for generating diverse robot data that capture perception, reasoning, and action in the physical world. However, synthesizing high-quality videos that accurately reflect real-world robotic interactions remains challenging, and the lack of a standardized benchmark limits fair comparisons and progress. To address this gap, we introduce a comprehensive robotics benchmark, RBench, designed to evaluate robot-oriented video generation across five task domains and four distinct embodiments. It assesses both task-level correctness and visual fidelity through reproducible sub-metrics, including structural consistency, physical plausibility, and action completeness. Evaluation of 25 representative models highlights significant deficiencies in generating physically realistic robot behaviors. Furthermore, the benchmark achieves a Spearman correlation coefficient of 0.96 with human evaluations, validating its effectiveness. While RBench provides the necessary lens to identify these deficiencies, achieving physical realism requires moving beyond evaluation to address the critical shortage of high-quality training data. Driven by these insights, we introduce a refined four-stage data pipeline, resulting in RoVid-X, the largest open-source robotic dataset for video generation with 4 million annotated video clips, covering thousands of tasks and enriched with comprehensive physical property annotations. Collectively, this synergistic ecosystem of evaluation and data establishes a robust foundation for rigorous assessment and scalable training of video models, accelerating the evolution of embodied AI toward general intelligence.
>
---
#### [new 080] Using Multi-Instance Learning to Identify Unique Polyps in Colon Capsule Endoscopy Images
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决结肠胶囊内镜中独特息肉识别的问题。通过多实例学习和注意力机制提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.14771v1](https://arxiv.org/pdf/2601.14771v1)**

> **作者:** Puneet Sharma; Kristian Dalsbø Hindberg; Eibe Frank; Benedicte Schelde-Olesen; Ulrik Deding
>
> **备注:** 19 pages
>
> **摘要:** Identifying unique polyps in colon capsule endoscopy (CCE) images is a critical yet challenging task for medical personnel due to the large volume of images, the cognitive load it creates for clinicians, and the ambiguity in labeling specific frames. This paper formulates this problem as a multi-instance learning (MIL) task, where a query polyp image is compared with a target bag of images to determine uniqueness. We employ a multi-instance verification (MIV) framework that incorporates attention mechanisms, such as variance-excited multi-head attention (VEMA) and distance-based attention (DBA), to enhance the model's ability to extract meaningful representations. Additionally, we investigate the impact of self-supervised learning using SimCLR to generate robust embeddings. Experimental results on a dataset of 1912 polyps from 754 patients demonstrate that attention mechanisms significantly improve performance, with DBA L1 achieving the highest test accuracy of 86.26\% and a test AUC of 0.928 using a ConvNeXt backbone with SimCLR pretraining. This study underscores the potential of MIL and self-supervised learning in advancing automated analysis of Colon Capsule Endoscopy images, with implications for broader medical imaging applications.
>
---
#### [new 081] A Cloud-Based Cross-Modal Transformer for Emotion Recognition and Adaptive Human-Computer Interaction
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于情感识别任务，旨在解决单模态系统在真实环境中的鲁棒性不足问题。提出一种基于云的跨模态Transformer框架，融合多模态数据并提升情感识别效果与响应速度。**

- **链接: [https://arxiv.org/pdf/2601.14259v1](https://arxiv.org/pdf/2601.14259v1)**

> **作者:** Ziwen Zhong; Zhitao Shu; Yue Zhao
>
> **摘要:** Emotion recognition is a fundamental component of next-generation human-computer interaction (HCI), enabling machines to perceive, understand, and respond to users' affective states. However, existing systems often rely on single-modality analysis such as facial expressions, speech tone, or textual sentiment, resulting in limited robustness and poor generalization in real-world environments. To address these challenges, this study proposes a Cloud-Based Cross-Modal Transformer (CMT) framework for multimodal emotion recognition and adaptive human-computer interaction. The proposed model integrates visual, auditory, and textual signals using pretrained encoders (Vision Transformer, Wav2Vec2, and BERT) and employs a cross-modal attention mechanism to capture complex interdependencies among heterogeneous features. By leveraging cloud computing infrastructure with distributed training on Kubernetes and TensorFlow Serving, the system enables scalable, low-latency emotion recognition for large-scale user interactions. Experiments conducted on benchmark datasets including IEMOCAP, MELD, and AffectNet demonstrate that the CMT achieves state-of-the-art performance, improving the F1-score by 3.0 percent and reducing cross-entropy loss by 12.9 percent compared to strong multimodal baselines. Additionally, cloud deployment evaluations show an average response latency of 128 ms, representing a 35 percent reduction compared with conventional transformer-based fusion systems. These results confirm that the proposed framework enables efficient, real-time emotion recognition and adaptive feedback in applications such as intelligent customer service, virtual tutoring systems, and affective computing interfaces, marking an important step toward cloud-native affective computing and emotionally intelligent interactive systems.
>
---
#### [new 082] LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于室内场景光照编辑任务，旨在解决单视角多视图下光照分解与重编辑问题。通过生成式光照分解模型和多视角光照协调，实现对光源的独立控制与实时交互。**

- **链接: [https://arxiv.org/pdf/2601.15283v1](https://arxiv.org/pdf/2601.15283v1)**

> **作者:** Ruofan Liang; Norman Müller; Ethan Weber; Duncan Zauss; Nandita Vijaykumar; Peter Kontschieder; Christian Richardt
>
> **备注:** Project page: https://luxremix.github.io
>
> **摘要:** We present a novel approach for interactive light editing in indoor scenes from a single multi-view scene capture. Our method leverages a generative image-based light decomposition model that factorizes complex indoor scene illumination into its constituent light sources. This factorization enables independent manipulation of individual light sources, specifically allowing control over their state (on/off), chromaticity, and intensity. We further introduce multi-view lighting harmonization to ensure consistent propagation of the lighting decomposition across all scene views. This is integrated into a relightable 3D Gaussian splatting representation, providing real-time interactive control over the individual light sources. Our results demonstrate highly photorealistic lighting decomposition and relighting outcomes across diverse indoor scenes. We evaluate our method on both synthetic and real-world datasets and provide a quantitative and qualitative comparison to state-of-the-art techniques. For video results and interactive demos, see https://luxremix.github.io.
>
---
#### [new 083] Towards Holistic Modeling for Video Frame Interpolation with Auto-regressive Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于视频帧插值任务，旨在解决传统方法导致的时间不一致和运动伪影问题。提出LDF-VFI框架，采用自回归扩散Transformer实现全局时序一致性。**

- **链接: [https://arxiv.org/pdf/2601.14959v1](https://arxiv.org/pdf/2601.14959v1)**

> **作者:** Xinyu Peng; Han Li; Yuyang Huang; Ziyang Zheng; Yaoming Wang; Xin Chen; Wenrui Dai; Chenglin Li; Junni Zou; Hongkai Xiong
>
> **摘要:** Existing video frame interpolation (VFI) methods often adopt a frame-centric approach, processing videos as independent short segments (e.g., triplets), which leads to temporal inconsistencies and motion artifacts. To overcome this, we propose a holistic, video-centric paradigm named \textbf{L}ocal \textbf{D}iffusion \textbf{F}orcing for \textbf{V}ideo \textbf{F}rame \textbf{I}nterpolation (LDF-VFI). Our framework is built upon an auto-regressive diffusion transformer that models the entire video sequence to ensure long-range temporal coherence. To mitigate error accumulation inherent in auto-regressive generation, we introduce a novel skip-concatenate sampling strategy that effectively maintains temporal stability. Furthermore, LDF-VFI incorporates sparse, local attention and tiled VAE encoding, a combination that not only enables efficient processing of long sequences but also allows generalization to arbitrary spatial resolutions (e.g., 4K) at inference without retraining. An enhanced conditional VAE decoder, which leverages multi-scale features from the input video, further improves reconstruction fidelity. Empirically, LDF-VFI achieves state-of-the-art performance on challenging long-sequence benchmarks, demonstrating superior per-frame quality and temporal consistency, especially in scenes with large motion. The source code is available at https://github.com/xypeng9903/LDF-VFI.
>
---
#### [new 084] Deep Leakage with Generative Flow Matching Denoiser
- **分类: cs.CV**

- **简介: 该论文属于联邦学习安全任务，旨在解决深度泄露攻击问题。通过引入生成流匹配先验，提升数据重建质量与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.15049v1](https://arxiv.org/pdf/2601.15049v1)**

> **作者:** Isaac Baglin; Xiatian Zhu; Simon Hadfield
>
> **摘要:** Federated Learning (FL) has emerged as a powerful paradigm for decentralized model training, yet it remains vulnerable to deep leakage (DL) attacks that reconstruct private client data from shared model updates. While prior DL methods have demonstrated varying levels of success, they often suffer from instability, limited fidelity, or poor robustness under realistic FL settings. We introduce a new DL attack that integrates a generative Flow Matching (FM) prior into the reconstruction process. By guiding optimization toward the distribution of realistic images (represented by a flow matching foundation model), our method enhances reconstruction fidelity without requiring knowledge of the private data. Extensive experiments on multiple datasets and target models demonstrate that our approach consistently outperforms state-of-the-art attacks across pixel-level, perceptual, and feature-based similarity metrics. Crucially, the method remains effective across different training epochs, larger client batch sizes, and under common defenses such as noise injection, clipping, and sparsification. Our findings call for the development of new defense strategies that explicitly account for adversaries equipped with powerful generative priors.
>
---
#### [new 085] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出RoT框架，将文本推理过程转化为图像，解决LLM推理可解释性差的问题，实现更高效的推理。**

- **链接: [https://arxiv.org/pdf/2601.14750v1](https://arxiv.org/pdf/2601.14750v1)**

> **作者:** Yifan Wang; Shiyu Li; Peiming Li; Xiaochen Yang; Yang Tang; Zheng Wei
>
> **摘要:** Chain-of-Thought (CoT) prompting has achieved remarkable success in unlocking the reasoning capabilities of Large Language Models (LLMs). Although CoT prompting enhances reasoning, its verbosity imposes substantial computational overhead. Recent works often focus exclusively on outcome alignment and lack supervision on the intermediate reasoning process. These deficiencies obscure the analyzability of the latent reasoning chain. To address these challenges, we introduce Render-of-Thought (RoT), the first framework to reify the reasoning chain by rendering textual steps into images, making the latent rationale explicit and traceable. Specifically, we leverage the vision encoders of existing Vision Language Models (VLMs) as semantic anchors to align the vision embeddings with the textual space. This design ensures plug-and-play implementation without incurring additional pre-training overhead. Extensive experiments on mathematical and logical reasoning benchmarks demonstrate that our method achieves 3-4x token compression and substantial inference acceleration compared to explicit CoT. Furthermore, it maintains competitive performance against other methods, validating the feasibility of this paradigm. Our code is available at https://github.com/TencentBAC/RoT
>
---
#### [new 086] Mixture-of-Experts Models in Vision: Routing, Optimization, and Generalization
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究视觉任务中的Mixture-of-Experts模型，解决其性能、利用效率和泛化能力问题，通过实验对比不同架构的性能与效率。**

- **链接: [https://arxiv.org/pdf/2601.15021v1](https://arxiv.org/pdf/2601.15021v1)**

> **作者:** Adam Rokah; Daniel Veress; Caleb Caulk; Sourav Sharan
>
> **备注:** 7 pages, 8 figures. Code available at: https://github.com/moe-project-uu/mixture-of-experts-project
>
> **摘要:** Mixture-of-Experts (MoE) architectures enable conditional computation by routing inputs to multiple expert subnetworks and are often motivated as a mechanism for scaling large language models. In this project, we instead study MoE behavior in an image classification setting, focusing on predictive performance, expert utilization, and generalization. We compare dense, SoftMoE, and SparseMoE classifier heads on the CIFAR10 dataset under comparable model capacity. Both MoE variants achieve slightly higher validation accuracy than the dense baseline while maintaining balanced expert utilization through regularization, avoiding expert collapse. To analyze generalization, we compute Hessian-based sharpness metrics at convergence, including the largest eigenvalue and trace of the loss Hessian, evaluated on both training and test data. We find that SoftMoE exhibits higher sharpness by these metrics, while Dense and SparseMoE lie in a similar curvature regime, despite all models achieving comparable generalization performance. Complementary loss surface perturbation analyses reveal qualitative differences in non-local behavior under finite parameter perturbations between dense and MoE models, which help contextualize curvature-based measurements without directly explaining validation accuracy. We further evaluate empirical inference efficiency and show that naively implemented conditional routing does not yield inference speedups on modern hardware at this scale, highlighting the gap between theoretical and realized efficiency in sparse MoE models.
>
---
#### [new 087] SpooFL: Spoofing Federated Learning
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 该论文属于联邦学习安全任务，解决数据泄露问题。提出SpooFL，通过生成无关合成数据欺骗攻击者，防止信息泄露同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2601.15055v1](https://arxiv.org/pdf/2601.15055v1)**

> **作者:** Isaac Baglin; Xiatian Zhu; Simon Hadfield
>
> **摘要:** Traditional defenses against Deep Leakage (DL) attacks in Federated Learning (FL) primarily focus on obfuscation, introducing noise, transformations or encryption to degrade an attacker's ability to reconstruct private data. While effective to some extent, these methods often still leak high-level information such as class distributions or feature representations, and are frequently broken by increasingly powerful denoising attacks. We propose a fundamentally different perspective on FL defense: framing it as a spoofing problem.We introduce SpooFL (Figure 1), a spoofing-based defense that deceives attackers into believing they have recovered the true training data, while actually providing convincing but entirely synthetic samples from an unrelated task. Unlike prior synthetic-data defenses that share classes or distributions with the private data and thus still leak semantic information, SpooFL uses a state-of-the-art generative model trained on an external dataset with no class overlap. As a result, attackers are misled into recovering plausible yet completely irrelevant samples, preventing meaningful data leakage while preserving FL training integrity. We implement the first example of such a spoofing defense, and evaluate our method against state-of-the-art DL defenses and demonstrate that it successfully misdirects attackers without compromising model performance significantly.
>
---
#### [new 088] DeepFedNAS: A Unified Framework for Principled, Hardware-Aware, and Predictor-Free Federated Neural Architecture Search
- **分类: cs.LG; cs.CV; cs.DC**

- **简介: 该论文提出DeepFedNAS，解决联邦神经网络架构搜索中的效率与性能问题，通过两阶段框架提升模型精度和搜索速度。**

- **链接: [https://arxiv.org/pdf/2601.15127v1](https://arxiv.org/pdf/2601.15127v1)**

> **作者:** Bostan Khan; Masoud Daneshtalab
>
> **备注:** This paper significantly extends the preliminary work accepted at ESANN 2026. Source Code: https://github.com/bostankhan6/DeepFedNAS
>
> **摘要:** Federated Neural Architecture Search (FedNAS) aims to automate model design for privacy-preserving Federated Learning (FL) but currently faces two critical bottlenecks: unguided supernet training that yields suboptimal models, and costly multi-hour pipelines for post-training subnet discovery. We introduce DeepFedNAS, a novel, two-phase framework underpinned by a principled, multi-objective fitness function that synthesizes mathematical network design with architectural heuristics. Enabled by a re-engineered supernet, DeepFedNAS introduces Federated Pareto Optimal Supernet Training, which leverages a pre-computed Pareto-optimal cache of high-fitness architectures as an intelligent curriculum to optimize shared supernet weights. Subsequently, its Predictor-Free Search Method eliminates the need for costly accuracy surrogates by utilizing this fitness function as a direct, zero-cost proxy for accuracy, enabling on-demand subnet discovery in mere seconds. DeepFedNAS achieves state-of-the-art accuracy (e.g., up to 1.21% absolute improvement on CIFAR-100), superior parameter and communication efficiency, and a substantial ~61x speedup in total post-training search pipeline time. By reducing the pipeline from over 20 hours to approximately 20 minutes (including initial cache generation) and enabling 20-second individual subnet searches, DeepFedNAS makes hardware-aware FL deployments instantaneous and practical. The complete source code and experimental scripts are available at: https://github.com/bostankhan6/DeepFedNAS
>
---
#### [new 089] Filtered 2D Contour-Based Reconstruction of 3D STL Model from CT-DICOM Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D模型重建任务，旨在解决从CT-DICOM图像中提取的2D轮廓数据存在噪声和偏差的问题，通过过滤2D轮廓数据来提高重建的3D STL模型的几何精度。**

- **链接: [https://arxiv.org/pdf/2601.14997v1](https://arxiv.org/pdf/2601.14997v1)**

> **作者:** K. Punnam Chandar; Y. Ravi Kumar
>
> **备注:** 8 pages, 18 figures
>
> **摘要:** Reconstructing a 3D Stereo-lithography (STL) Model from 2D Contours of scanned structure in Digital Imaging and Communication in Medicine (DICOM) images is crucial to understand the geometry and deformity. Computed Tomography (CT) images are processed to enhance the contrast, reduce the noise followed by smoothing. The processed CT images are segmented using thresholding technique. 2D contour data points are extracted from segmented CT images and are used to construct 3D STL Models. The 2D contour data points may contain outliers as a result of segmentation of low resolution images and the geometry of the constructed 3D structure deviate from the actual. To cope with the imperfections in segmentation process, in this work we propose to use filtered 2D contour data points to reconstruct 3D STL Model. The filtered 2D contour points of each image are delaunay triangulated and joined layer-by-layer to reconstruct the 3D STL model. The 3D STL Model reconstruction is verified on i) 2D Data points of basic shapes and ii) Region of Interest (ROI) of human pelvic bone and are presented as case studies. The 3D STL model constructed from 2D contour data points of ROI of segmented pelvic bone with and without filtering are presented. The 3D STL model reconstructed from filtered 2D data points improved the geometry of model compared to the model reconstructed without filtering 2D data points.
>
---
#### [new 090] ZENITH: Automated Gradient Norm Informed Stochastic Optimization
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出ZENITH优化器，用于自动调整深度学习中的学习率，解决传统方法计算开销大、兼容性差的问题。任务为模型训练优化。**

- **链接: [https://arxiv.org/pdf/2601.15212v1](https://arxiv.org/pdf/2601.15212v1)**

> **作者:** Dhrubo Saha
>
> **摘要:** Training deep computer vision models requires manual oversight or hyperparameter tuning of the learning rate (LR) schedule. While existing adaptive optimizers schedule the LR automatically, they suffer from computational and memory overhead, incompatibility with regularization, and suboptimal LR choices. In this work, we introduce the ZENITH (Zero-overhead Evolution using Norm-Informed Training History) optimizer, which adapts the LR using the temporal evolution of the gradient norm. Image classification experiments spanning 6 CNN architectures and 6 benchmarks demonstrate that ZENITH achieves higher test accuracy in lower wall-clock time than baselines. It also yielded superior mAP in object detection, keypoint detection, and instance segmentation on MS COCO using the R-CNN family of models. Furthermore, its compatibility with regularization enables even better generalization.
>
---
#### [new 091] Self-Supervised Score-Based Despeckling for SAR Imagery via Log-Domain Transformation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于SAR图像去噪任务，解决乘性Gamma分布噪声问题。通过log域变换和自监督得分模型，实现高效去噪。**

- **链接: [https://arxiv.org/pdf/2601.14334v1](https://arxiv.org/pdf/2601.14334v1)**

> **作者:** Junhyuk Heo
>
> **摘要:** The speckle noise inherent in Synthetic Aperture Radar (SAR) imagery significantly degrades image quality and complicates subsequent analysis. Given that SAR speckle is multiplicative and Gamma-distributed, effectively despeckling SAR imagery remains challenging. This paper introduces a novel self-supervised framework for SAR image despeckling based on score-based generative models operating in the transformed log domain. We first transform the data into the log-domain and then convert the speckle noise residuals into an approximately additive Gaussian distribution. This step enables the application of score-based models, which are trained in the transformed domain using a self-supervised objective. This objective allows our model to learn the clean underlying signal by training on further corrupted versions of the input data itself. Consequently, our method exhibits significantly shorter inference times compared to many existing self-supervised techniques, offering a robust and practical solution for SAR image restoration.
>
---
#### [new 092] Unsupervised Deformable Image Registration with Local-Global Attention and Image Decomposition
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决传统方法计算量大、泛化能力差的问题。提出LGANet++框架，结合局部全局注意力和特征融合，提升配准精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.14337v1](https://arxiv.org/pdf/2601.14337v1)**

> **作者:** Zhengyong Huang; Xingwen Sun; Xuting Chang; Ning Jiang; Yao Wang; Jianfei Sun; Hongbin Han; Yao Sui
>
> **摘要:** Deformable image registration is a critical technology in medical image analysis, with broad applications in clinical practice such as disease diagnosis, multi-modal fusion, and surgical navigation. Traditional methods often rely on iterative optimization, which is computationally intensive and lacks generalizability. Recent advances in deep learning have introduced attention-based mechanisms that improve feature alignment, yet accurately registering regions with high anatomical variability remains challenging. In this study, we proposed a novel unsupervised deformable image registration framework, LGANet++, which employs a novel local-global attention mechanism integrated with a unique technique for feature interaction and fusion to enhance registration accuracy, robustness, and generalizability. We evaluated our approach using five publicly available datasets, representing three distinct registration scenarios: cross-patient, cross-time, and cross-modal CT-MR registration. The results demonstrated that our approach consistently outperforms several state-of-the-art registration methods, improving registration accuracy by 1.39% in cross-patient registration, 0.71% in cross-time registration, and 6.12% in cross-modal CT-MR registration tasks. These results underscore the potential of LGANet++ to support clinical workflows requiring reliable and efficient image registration. The source code is available at https://github.com/huangzyong/LGANet-Registration.
>
---
#### [new 093] AutoDriDM: An Explainable Benchmark for Decision-Making of Vision-Language Models in Autonomous Driving
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于自主驾驶中的决策任务，旨在解决现有基准对感知与决策评估不均衡的问题。提出AutoDriDM基准，评估视觉语言模型的决策能力，并分析其推理过程。**

- **链接: [https://arxiv.org/pdf/2601.14702v1](https://arxiv.org/pdf/2601.14702v1)**

> **作者:** Zecong Tang; Zixu Wang; Yifei Wang; Weitong Lian; Tianjian Gao; Haoran Li; Tengju Ru; Lingyi Meng; Zhejun Cui; Yichen Zhu; Qi Kang; Kaixuan Wang; Yu Zhang
>
> **备注:** 23 pages. Submitted to ACL ARR 2026 January
>
> **摘要:** Autonomous driving is a highly challenging domain that requires reliable perception and safe decision-making in complex scenarios. Recent vision-language models (VLMs) demonstrate reasoning and generalization abilities, opening new possibilities for autonomous driving; however, existing benchmarks and metrics overemphasize perceptual competence and fail to adequately assess decision-making processes. In this work, we present AutoDriDM, a decision-centric, progressive benchmark with 6,650 questions across three dimensions - Object, Scene, and Decision. We evaluate mainstream VLMs to delineate the perception-to-decision capability boundary in autonomous driving, and our correlation analysis reveals weak alignment between perception and decision-making performance. We further conduct explainability analyses of models' reasoning processes, identifying key failure modes such as logical reasoning errors, and introduce an analyzer model to automate large-scale annotation. AutoDriDM bridges the gap between perception-centered and decision-centered evaluation, providing guidance toward safer and more reliable VLMs for real-world autonomous driving.
>
---
#### [new 094] Partial Decoder Attention Network with Contour-weighted Loss Function for Data-Imbalance Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，针对数据不平衡问题提出一种带轮廓加权损失的分割方法，提升小结构和罕见结构的分割效果。**

- **链接: [https://arxiv.org/pdf/2601.14338v1](https://arxiv.org/pdf/2601.14338v1)**

> **作者:** Zhengyong Huang; Ning Jiang; Xingwen Sun; Lihua Zhang; Peng Chen; Jens Domke; Yao Sui
>
> **摘要:** Image segmentation is pivotal in medical image analysis, facilitating clinical diagnosis, treatment planning, and disease evaluation. Deep learning has significantly advanced automatic segmentation methodologies by providing superior modeling capability for complex structures and fine-grained anatomical regions. However, medical images often suffer from data imbalance issues, such as large volume disparities among organs or tissues, and uneven sample distributions across different anatomical structures. This imbalance tends to bias the model toward larger organs or more frequently represented structures, while overlooking smaller or less represented structures, thereby affecting the segmentation accuracy and robustness. To address these challenges, we proposed a novel contour-weighted segmentation approach, which improves the model's capability to represent small and underrepresented structures. We developed PDANet, a lightweight and efficient segmentation network based on a partial decoder mechanism. We evaluated our method using three prominent public datasets. The experimental results show that our methodology excelled in three distinct tasks: segmenting multiple abdominal organs, brain tumors, and pelvic bone fragments with injuries. It consistently outperformed nine state-of-the-art methods. Moreover, the proposed contour-weighted strategy improved segmentation for other comparison methods across the three datasets, yielding average enhancements in Dice scores of 2.32%, 1.67%, and 3.60%, respectively. These results demonstrate that our contour-weighted segmentation method surpassed current leading approaches in both accuracy and robustness. As a model-independent strategy, it can seamlessly fit various segmentation frameworks, enhancing their performance. This flexibility highlighted its practical importance and potential for broad use in medical image analysis.
>
---
#### [new 095] BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在新指令和复杂任务中泛化能力差的问题。通过引入贝叶斯分解和潜在动作查询，提升语言引导的行动策略。**

- **链接: [https://arxiv.org/pdf/2601.15197v1](https://arxiv.org/pdf/2601.15197v1)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
#### [new 096] Vision Models for Medical Imaging: A Hybrid Approach for PCOS Detection from Ultrasound Scans
- **分类: eess.IV; cs.CV**

- **简介: 论文属于医学图像分析任务，旨在解决PCOS的准确检测问题。通过构建混合模型，提升超声图像中PCOS的诊断准确性。**

- **链接: [https://arxiv.org/pdf/2601.15119v1](https://arxiv.org/pdf/2601.15119v1)**

> **作者:** Md Mahmudul Hoque; Md Mehedi Hassain; Muntakimur Rahaman; Md. Towhidul Islam; Shaista Rani; Md Sharif Mollah
>
> **摘要:** Polycystic Ovary Syndrome (PCOS) is the most familiar endocrine illness in women of reproductive age. Many Bangladeshi women suffer from PCOS disease in their older age. The aim of our research is to identify effective vision-based medical image analysis techniques and evaluate hybrid models for the accurate detection of PCOS. We introduced two novel hybrid models combining convolutional and transformer-based approaches. The training and testing data were organized into two categories: "infected" (PCOS-positive) and "noninfected" (healthy ovaries). In the initial stage, our first hybrid model, 'DenConST' (integrating DenseNet121, Swin Transformer, and ConvNeXt), achieved 85.69% accuracy. The final optimized model, 'DenConREST' (incorporating Swin Transformer, ConvNeXt, DenseNet121, ResNet18, and EfficientNetV2), demonstrated superior performance with 98.23% accuracy. Among all evaluated models, DenConREST showed the best performance. This research highlights an efficient solution for PCOS detection from ultrasound images, significantly improving diagnostic accuracy while reducing detection errors.
>
---
#### [new 097] ExPrIS: Knowledge-Level Expectations as Priors for Object Interpretation from Sensor Data
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人场景理解任务，旨在解决对象识别中语义不一致的问题。通过构建3D语义场景图并融合先验知识，提升对象解释的鲁棒性与一致性。**

- **链接: [https://arxiv.org/pdf/2601.15025v1](https://arxiv.org/pdf/2601.15025v1)**

> **作者:** Marian Renz; Martin Günther; Felix Igelbrink; Oscar Lima; Martin Atzmueller
>
> **备注:** This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this article is published in KI - Künstliche Intelligenz, and is available online at https://doi.org/10.1007/s13218-026-00901-7
>
> **摘要:** While deep learning has significantly advanced robotic object recognition, purely data-driven approaches often lack semantic consistency and fail to leverage valuable, pre-existing knowledge about the environment. This report presents the ExPrIS project, which addresses this challenge by investigating how knowledge-level expectations can serve as to improve object interpretation from sensor data. Our approach is based on the incremental construction of a 3D Semantic Scene Graph (3DSSG). We integrate expectations from two sources: contextual priors from past observations and semantic knowledge from external graphs like ConceptNet. These are embedded into a heterogeneous Graph Neural Network (GNN) to create an expectation-biased inference process. This method moves beyond static, frame-by-frame analysis to enhance the robustness and consistency of scene understanding over time. The report details this architecture, its evaluation, and outlines its planned integration on a mobile robotic platform.
>
---
## 更新

#### [replaced 001] Image class translation: visual inspection of class-specific hypotheticals and classification based on translation distance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.08973v2](https://arxiv.org/pdf/2408.08973v2)**

> **作者:** Mikyla K. Bowen; Jesse W. Wilson
>
> **备注:** 27 pages, 20 figures, submitted revision to SPIE J. Medical Imaging
>
> **摘要:** Purpose: A major barrier to the implementation of artificial intelligence for medical applications is the lack of explainability and high confidence for incorrect decisions, specifically with out-of-domain samples. We propose a generalization of image translation networks for image classification and demonstrate their potential as a more interpretable alternative to conventional black-box classifiers. Approach: We train an image2image network to translate an input image to class-specific hypotheticals, and then compare these with the input, both visually and quantitatively. Translation distances, i.e., the degree of alteration needed to conform to one class or another, are examined for clusters and trends, and used as simple low-dimensional feature vectors for classification. Results: On melanoma/benign dermoscopy images, a translation distance classifier achieved 80% accuracy using only a 2-dimensional feature space (versus 85% for a conventional CNN using a ~62,000-dimensional feature space). Visual inspection of rendered images revealed dataset biases, such as scalebars, vignetting, and pale background pigmentation in melanomas. Image distributions in translation distance space revealed a natural separation along the lines of dermatologist decision to biopsy, rather than between malignant and benign. On bone marrow cytology images, translation distance classifiers outperformed a conventional CNN in both 3-class (92% accuracy vs 89% for CNN) and 6-class (90% vs 86% for CNN) scenarios. Conclusions: This proof-of-concept shows the potential for image2image networks to go beyond artistic/stylistic changes and to expose dataset biases, perform dimension reduction and dataset visualization, and in some cases, potentially outperform conventional end-to-end CNN classifiers.
>
---
#### [replaced 002] HiT: History-Injection Transformers for Onboard Continuous Flood Change Detection
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.13751v2](https://arxiv.org/pdf/2601.13751v2)**

> **作者:** Daniel Kyselica; Jonáš Herec; Oliver Kutis; Rado Pitoňák
>
> **备注:** 19 pages, 9 figures, submitted to conference
>
> **摘要:** Natural disaster monitoring through continuous satellite observation requires processing multi-temporal data under strict operational constraints. This paper addresses flood detection, a critical application for hazard management, by developing an onboard change detection system that operates within the memory and computational limits of small satellites. We propose History Injection mechanism for Transformer models (HiT), that maintains historical context from previous observations while reducing data storage by over 99\% of original image size. Moreover, testing on the STTORM-CD flood dataset confirms that the HiT mechanism within the Prithvi-tiny foundation model maintains detection accuracy compared to the bitemporal baseline. The proposed HiT-Prithvi model achieved 43 FPS on Jetson Orin Nano, a representative onboard hardware used in nanosats. This work establishes a practical framework for satellite-based continuous monitoring of natural disasters, supporting real-time hazard assessment without dependency on ground-based processing infrastructure. Architecture as well as model checkpoints is available at https://github.com/zaitra/HiT-change-detection
>
---
#### [replaced 003] Interleaved Latent Visual Reasoning with Selective Perceptual Modeling
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，解决MLLMs中视觉反馈计算成本高与感知建模不足的问题。提出ILVR框架，结合动态状态演化与精确感知建模。**

- **链接: [https://arxiv.org/pdf/2512.05665v3](https://arxiv.org/pdf/2512.05665v3)**

> **作者:** Shuai Dong; Siyuan Wang; Xingyu Liu; Chenglin Li; Haowen Hou; Zhongyu Wei
>
> **备注:** 18 pages, 11 figures. Code available at https://github.com/XD111ds/ILVR
>
> **摘要:** Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet faces limitations: methods either fail to capture intermediate state evolution due to single-step, non-interleaved structures, or sacrifice precise perceptual modeling by over-compressing features. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. Specifically, we employ a self-supervision strategy where a momentum teacher model selectively distills relevant features from ground-truth intermediate images into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning. The code is available at https://github.com/XD111ds/ILVR.
>
---
#### [replaced 004] T2T-VICL: Unlocking the Boundaries of Cross-Task Visual In-Context Learning via Implicit Text-Driven VLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.16107v2](https://arxiv.org/pdf/2511.16107v2)**

> **作者:** Shao-Jun Xia; Huixin Zhang; Zhengzhong Tu
>
> **摘要:** In large language models (LLM), in-context learning (ICL) refers to performing new tasks by conditioning on small demonstrations provided in the input context. Recent advances in visual in-context learning (VICL) demonstrate promising capabilities for solving downstream tasks by unified vision-language models (VLMs). When the visual prompt and the target images originate from different visual tasks, can VLMs still enable VICL? In the paper, we propose a fully collaborative pipeline, i.e. T2T-VICL, for VLMs to investigate the potential of cross-task VICL. Fundamentally, we design a mechanism to generate and select text prompts that best implicitly describe the differences between two distinct low-level vision tasks, and construct the first cross-task VICL dataset. Building upon this, we propose a novel inference framework that combines perceptual score-based reasoning with traditional evaluation metrics to perform cross-task VICL. Our approach achieves top-tier results across twelve cross-task scenarios and second-tier performance in nine additional scenarios, unlocking the boundaries of cross-task VICL within VLMs.
>
---
#### [replaced 005] Encoding Emotion Through Self-Supervised Eye Movement Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.12534v2](https://arxiv.org/pdf/2601.12534v2)**

> **作者:** Marcus Ma; Jordan Prescott; Emily Zhou; Tiantian Feng; Kleanthis Avramidis; Gabor Mihaly Toth; Shrikanth Narayanan
>
> **摘要:** The relationship between emotional expression and eye movement is well-documented, with literature establishing gaze patterns are reliable indicators of emotion. However, most studies utilize specialized, high-resolution eye-tracking equipment, limiting the potential reach of findings. We investigate how eye movement can be used to predict multimodal markers of emotional expression from naturalistic, low-resolution videos. We utilize a collection of video interviews from the USC Shoah Foundation's Visual History Archive with Holocaust survivors as they recount their experiences in the Auschwitz concentration camp. Inspired by pretraining methods on language models, we develop a novel gaze detection model that uses self-supervised eye movement reconstruction that can effectively leverage unlabeled video. We use this model's encoder embeddings to fine-tune models on two downstream tasks related to emotional expression. The first is aligning eye movement with directional emotion estimates from speech. The second task is using eye gaze as a predictor of three momentary manifestations of emotional behaviors: laughing, crying/sobbing, and sighing. We find our new model is predictive of emotion outcomes and observe a positive correlation between pretraining performance and emotion processing performance for both experiments. We conclude self-supervised eye movement reconstruction is an effective method for encoding the affective signal they carry.
>
---
#### [replaced 006] Semantic Image Synthesis via Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2207.00050v4](https://arxiv.org/pdf/2207.00050v4)**

> **作者:** Wengang Zhou; Weilun Wang; Jianmin Bao; Dongdong Chen; Dong Chen; Lu Yuan; Houqiang Li
>
> **摘要:** Denoising Diffusion Probabilistic Models (DDPMs) have achieved remarkable success in various image generation tasks compared with Generative Adversarial Nets (GANs). Recent work on semantic image synthesis mainly follows the de facto GAN-based approaches, which may lead to unsatisfactory quality or diversity of generated images. In this paper, we propose a novel framework based on DDPM for semantic image synthesis. Unlike previous conditional diffusion model directly feeds the semantic layout and noisy image as input to a U-Net structure, which may not fully leverage the information in the input semantic mask, our framework processes semantic layout and noisy image differently. It feeds noisy image to the encoder of the U-Net structure while the semantic layout to the decoder by multi-layer spatially-adaptive normalization operators. To further improve the generation quality and semantic interpretability in semantic image synthesis, we introduce the classifier-free guidance sampling strategy, which acknowledge the scores of an unconditional model for sampling process. Extensive experiments on four benchmark datasets demonstrate the effectiveness of our proposed method, achieving state-of-the-art performance in terms of fidelity (FID) and diversity (LPIPS). Our code and pretrained models are available at https://github.com/WeilunWang/semantic-diffusion-model.
>
---
#### [replaced 007] $\mathrm{D}^\mathrm{3}$-Predictor: Noise-Free Deterministic Diffusion for Dense Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.07062v4](https://arxiv.org/pdf/2512.07062v4)**

> **作者:** Changliang Xia; Chengyou Jia; Minnan Luo; Zhuohang Dang; Xin Shen; Bowen Ping
>
> **摘要:** Although diffusion models with strong visual priors have emerged as powerful dense prediction backbones, they overlook a core limitation: the stochastic noise at the core of diffusion sampling is inherently misaligned with dense prediction that requires a deterministic mapping from image to geometry. In this paper, we show that this stochastic noise corrupts fine-grained spatial cues and pushes the model toward timestep-specific noise objectives, consequently destroying meaningful geometric structure mappings. To address this, we introduce $\mathrm{D}^\mathrm{3}$-Predictor, a noise-free deterministic diffusion-based dense prediction model built by reformulating a pretrained diffusion model without stochasticity noise. Instead of relying on noisy inputs to leverage diffusion priors, $\mathrm{D}^\mathrm{3}$-Predictor views the pretrained diffusion network as an ensemble of timestep-dependent visual experts and self-supervisedly aggregates their heterogeneous priors into a single, clean, and complete geometric prior. Meanwhile, we utilize task-specific supervision to seamlessly adapt this noise-free prior to dense prediction tasks. Extensive experiments on various dense prediction tasks demonstrate that $\mathrm{D}^\mathrm{3}$-Predictor achieves competitive or state-of-the-art performance in diverse scenarios. In addition, it requires less than half the training data previously used and efficiently performs inference in a single step. Our code, data, and checkpoints are publicly available at https://x-gengroup.github.io/HomePage_D3-Predictor/.
>
---
#### [replaced 008] Debate-Enhanced Pseudo Labeling and Frequency-Aware Progressive Debiasing for Weakly-Supervised Camouflaged Object Detection with Scribble Annotations
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.20260v4](https://arxiv.org/pdf/2512.20260v4)**

> **作者:** Jiawei Ge; Jiuxin Cao; Xinyi Li; Xuelin Zhu; Chang Liu; Bo Liu; Chen Feng; Ioannis Patras
>
> **摘要:** Weakly-Supervised Camouflaged Object Detection (WSCOD) aims to locate and segment objects that are visually concealed within their surrounding scenes, relying solely on sparse supervision such as scribble annotations. Despite recent progress, existing WSCOD methods still lag far behind fully supervised ones due to two major limitations: (1) the pseudo masks generated by general-purpose segmentation models (e.g., SAM) and filtered via rules are often unreliable, as these models lack the task-specific semantic understanding required for effective pseudo labeling in COD; and (2) the neglect of inherent annotation bias in scribbles, which hinders the model from capturing the global structure of camouflaged objects. To overcome these challenges, we propose ${D}^{3}$ETOR, a two-stage WSCOD framework consisting of Debate-Enhanced Pseudo Labeling and Frequency-Aware Progressive Debiasing. In the first stage, we introduce an adaptive entropy-driven point sampling method and a multi-agent debate mechanism to enhance the capability of SAM for COD, improving the interpretability and precision of pseudo masks. In the second stage, we design FADeNet, which progressively fuses multi-level frequency-aware features to balance global semantic understanding with local detail modeling, while dynamically reweighting supervision strength across regions to alleviate scribble bias. By jointly exploiting the supervision signals from both the pseudo masks and scribble semantics, ${D}^{3}$ETOR significantly narrows the gap between weakly and fully supervised COD, achieving state-of-the-art performance on multiple benchmarks.
>
---
#### [replaced 009] OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于机器人感知任务，旨在评估不同光照条件下开放语义映射的性能。提出OSMa-Bench框架，通过新数据集和场景图方法分析模型的语义准确性和结构理解能力。**

- **链接: [https://arxiv.org/pdf/2503.10331v3](https://arxiv.org/pdf/2503.10331v3)**

> **作者:** Maxim Popov; Regina Kurkova; Mikhail Iumanov; Jaafar Mahmoud; Sergey Kolyubin
>
> **备注:** Project page: https://be2rlab.github.io/OSMa-Bench/
>
> **摘要:** Open Semantic Mapping (OSM) is a key technology in robotic perception, combining semantic segmentation and SLAM techniques. This paper introduces a dynamically configurable and highly automated LLM/LVLM-powered pipeline for evaluating OSM solutions called OSMa-Bench (Open Semantic Mapping Benchmark). The study focuses on evaluating state-of-the-art semantic mapping algorithms under varying indoor lighting conditions, a critical challenge in indoor environments. We introduce a novel dataset with simulated RGB-D sequences and ground truth 3D reconstructions, facilitating the rigorous analysis of mapping performance across different lighting conditions. Through experiments on leading models such as ConceptGraphs, BBQ, and OpenScene, we evaluate the semantic fidelity of object recognition and segmentation. Additionally, we introduce a Scene Graph evaluation method to analyze the ability of models to interpret semantic structure. The results provide insights into the robustness of these models, forming future research directions for developing resilient and adaptable robotic systems. Project page is available at https://be2rlab.github.io/OSMa-Bench/.
>
---
#### [replaced 010] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2503.10860v2](https://arxiv.org/pdf/2503.10860v2)**

> **作者:** Avinash Paliwal; Xilong Zhou; Wei Ye; Jinhui Xiong; Rakesh Ranjan; Nima Khademi Kalantari
>
> **备注:** ICCV 2025, Project page: https://people.engr.tamu.edu/nimak/Papers/RI3D, Code: https://github.com/avinashpaliwal/RI3D
>
> **摘要:** In this paper, we propose RI3D, a novel 3DGS-based approach that harnesses the power of diffusion models to reconstruct high-quality novel views given a sparse set of input images. Our key contribution is separating the view synthesis process into two tasks of reconstructing visible regions and hallucinating missing regions, and introducing two personalized diffusion models, each tailored to one of these tasks. Specifically, one model ('repair') takes a rendered image as input and predicts the corresponding high-quality image, which in turn is used as a pseudo ground truth image to constrain the optimization. The other model ('inpainting') primarily focuses on hallucinating details in unobserved areas. To integrate these models effectively, we introduce a two-stage optimization strategy: the first stage reconstructs visible areas using the repair model, and the second stage reconstructs missing regions with the inpainting model while ensuring coherence through further optimization. Moreover, we augment the optimization with a novel Gaussian initialization method that obtains per-image depth by combining 3D-consistent and smooth depth with highly detailed relative depth. We demonstrate that by separating the process into two tasks and addressing them with the repair and inpainting models, we produce results with detailed textures in both visible and missing regions that outperform state-of-the-art approaches on a diverse set of scenes with extremely sparse inputs.
>
---
#### [replaced 011] Benchmarking the Influence of Pre-training on Explanation Performance in MR Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2306.12150v2](https://arxiv.org/pdf/2306.12150v2)**

> **作者:** Marta Oliveira; Rick Wilming; Benedict Clark; Céline Budding; Fabian Eitel; Kerstin Ritter; Stefan Haufe
>
> **备注:** Under review
>
> **摘要:** Convolutional Neural Networks (CNNs) are frequently and successfully used in medical prediction tasks. They are often used in combination with transfer learning, leading to improved performance when training data for the task are scarce. The resulting models are highly complex and typically do not provide any insight into their predictive mechanisms, motivating the field of "explainable" artificial intelligence (XAI). However, previous studies have rarely quantitatively evaluated the "explanation performance" of XAI methods against ground-truth data, and transfer learning and its influence on objective measures of explanation performance has not been investigated. Here, we propose a benchmark dataset that allows for quantifying explanation performance in a realistic magnetic resonance imaging (MRI) classification task. We employ this benchmark to understand the influence of transfer learning on the quality of explanations. Experimental results show that popular XAI methods applied to the same underlying model differ vastly in performance, even when considering only correctly classified examples. We further observe that explanation performance strongly depends on the task used for pre-training and the number of CNN layers pre-trained. These results hold after correcting for a substantial correlation between explanation and classification performance.
>
---
#### [replaced 012] Coding the Visual World: From Image to Simulation Using Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.05344v2](https://arxiv.org/pdf/2601.05344v2)**

> **作者:** Sagi Eppel
>
> **摘要:** The ability to construct mental models of the world is a central aspect of understanding. Similarly, visual understanding can be viewed as the ability to construct a representative model of the system depicted in an image. This work explores the capacity of Vision Language Models (VLMs) to recognize and simulate the systems and mechanisms depicted in images using the Im2Sim methodology. The VLM is given a natural image of a real-world system (e.g., cities, clouds, vegetation) and is tasked with describing the system and writing code that simulates and generates it. This generative code is then executed to produce a synthetic image, which is compared against the original. This approach is tested on various complex emergent systems, ranging from physical systems (waves, lights, clouds) to vegetation, cities, materials, and geological formations. Through analysis of the models and images generated by the VLMs, we examine their understanding of the systems in images. The results show that leading VLMs (GPT, Gemini) have the ability to understand and model complex, multi-component systems across multiple layers of abstraction and a wide range of domains. At the same time, the VLMs exhibit limited ability to replicate fine details and low-level arrangements of patterns in the image. These findings reveal an interesting asymmetry: VLMs combine high-level, deep visual understanding of images with limited perception of fine details.
>
---
#### [replaced 013] Hyperphantasia: A Benchmark for Evaluating the Mental Visualization Capabilities of Multimodal LLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.11932v2](https://arxiv.org/pdf/2507.11932v2)**

> **作者:** Mohammad Shahab Sepehri; Berk Tinaz; Zalan Fabian; Mahdi Soltanolkotabi
>
> **摘要:** Mental visualization, the ability to construct and manipulate visual representations internally, is a core component of human cognition and plays a vital role in tasks involving reasoning, prediction, and abstraction. Despite the rapid progress of Multimodal Large Language Models (MLLMs), current benchmarks primarily assess passive visual perception, offering limited insight into the more active capability of internally constructing visual patterns to support problem solving. Yet mental visualization is a critical cognitive skill in humans, supporting abilities such as spatial navigation, predicting physical trajectories, and solving complex visual problems through imaginative simulation. To bridge this gap, we introduce Hyperphantasia, a synthetic benchmark designed to evaluate the mental visualization abilities of MLLMs through four carefully constructed puzzles. Each puzzle is procedurally generated and presented at three difficulty levels, enabling controlled analysis of model performance across increasing complexity. Our comprehensive evaluation of state-of-the-art models reveals a substantial gap between the performance of humans and MLLMs. Additionally, we explore the potential of reinforcement learning to improve visual simulation capabilities. Our findings suggest that while some models exhibit partial competence in recognizing visual patterns, robust mental visualization remains an open challenge for current MLLMs.
>
---
#### [replaced 014] BirdsEye-RU: A Dataset For Detecting Faces from Overhead Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.12533v2](https://arxiv.org/pdf/2601.12533v2)**

> **作者:** Md. Ahanaf Arif Khan; Ariful Islam; Sangeeta Biswas; Md. Iqbal Aziz Khan; Subrata Pramanik; Sanjoy Kumar Chakravarty; Bimal Kumar Pramanik
>
> **摘要:** Detecting faces in overhead images remains a significant challenge due to extreme scale variations and environmental clutter. To address this, we created the BirdsEye-RU dataset, a comprehensive collection of 2,978 images containing over eight thousand annotated faces. This dataset is specifically designed to capture small and distant faces across diverse environments, containing both drone images and smartphone-captured images from high altitude. We present a detailed description of the BirdsEye-RU dataset in this paper. We made our dataset freely available to the public, and it can be accessed at https://www.kaggle.com/datasets/mdahanafarifkhan/birdseye-ru.
>
---
#### [replaced 015] AI-generated data contamination erodes pathological variability and diagnostic reliability
- **分类: cs.CY; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于医疗AI领域，研究AI生成数据对病理多样性与诊断可靠性的影响。工作包括分析合成数据、发现模型偏差，并评估缓解策略。**

- **链接: [https://arxiv.org/pdf/2601.12946v2](https://arxiv.org/pdf/2601.12946v2)**

> **作者:** Hongyu He; Shaowen Xiang; Ye Zhang; Yingtao Zhu; Jin Zhang; Hao Deng; Emily Alsentzer; Qingyu Chen; Kun-Hsing Yu; Andrew Marshall; Tingting Chen; Srinivas Anumasa; Daniel Ebner; Dean Ho; Kee Yuan Ngiam; Ching-Yu Cheng; Dianbo Liu
>
> **备注:** *Corresponding author: Dianbo Liu (dianbo@nus.edu.sg)
>
> **摘要:** Generative artificial intelligence (AI) is rapidly populating medical records with synthetic content, creating a feedback loop where future models are increasingly at risk of training on uncurated AI-generated data. However, the clinical consequences of this AI-generated data contamination remain unexplored. Here, we show that in the absence of mandatory human verification, this self-referential cycle drives a rapid erosion of pathological variability and diagnostic reliability. By analysing more than 800,000 synthetic data points across clinical text generation, vision-language reporting, and medical image synthesis, we find that models progressively converge toward generic phenotypes regardless of the model architecture. Specifically, rare but critical findings, including pneumothorax and effusions, vanish from the synthetic content generated by AI models, while demographic representations skew heavily toward middle-aged male phenotypes. Crucially, this degradation is masked by false diagnostic confidence; models continue to issue reassuring reports while failing to detect life-threatening pathology, with false reassurance rates tripling to 40%. Blinded physician evaluation confirms that this decoupling of confidence and accuracy renders AI-generated documentation clinically useless after just two generations. We systematically evaluate three mitigation strategies, finding that while synthetic volume scaling fails to prevent collapse, mixing real data with quality-aware filtering effectively preserves diversity. Ultimately, our results suggest that without policy-mandated human oversight, the deployment of generative AI threatens to degrade the very healthcare data ecosystems it relies upon.
>
---
#### [replaced 016] When Are Two Scores Better Than One? Investigating Ensembles of Diffusion Models
- **分类: cs.LG; cs.CV; math.ST; stat.ME; stat.ML**

- **链接: [https://arxiv.org/pdf/2601.11444v2](https://arxiv.org/pdf/2601.11444v2)**

> **作者:** Raphaël Razafindralambo; Rémy Sun; Frédéric Precioso; Damien Garreau; Pierre-Alexandre Mattei
>
> **备注:** Accepted at Transactions on Machine Learning Research (reviewed on OpenReview: https://openreview.net/forum?id=4iRx9b0Csu). Code: https://github.com/rarazafin/score_diffusion_ensemble
>
> **摘要:** Diffusion models now generate high-quality, diverse samples, with an increasing focus on more powerful models. Although ensembling is a well-known way to improve supervised models, its application to unconditional score-based diffusion models remains largely unexplored. In this work we investigate whether it provides tangible benefits for generative modelling. We find that while ensembling the scores generally improves the score-matching loss and model likelihood, it fails to consistently enhance perceptual quality metrics such as FID on image datasets. We confirm this observation across a breadth of aggregation rules using Deep Ensembles, Monte Carlo Dropout, on CIFAR-10 and FFHQ. We attempt to explain this discrepancy by investigating possible explanations, such as the link between score estimation and image quality. We also look into tabular data through random forests, and find that one aggregation strategy outperforms the others. Finally, we provide theoretical insights into the summing of score models, which shed light not only on ensembling but also on several model composition techniques (e.g. guidance).
>
---
#### [replaced 017] A Dynamic Prognostic Prediction Method for Colorectal Cancer Liver Metastasis
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2505.03123v2](https://arxiv.org/pdf/2505.03123v2)**

> **作者:** Wei Yang; Yiran Zhu; Yan su; Zesheng Li; Chengchang Pan; Honggang Qi
>
> **摘要:** Colorectal cancer liver metastasis (CRLM) exhibits high postoperative recurrence and pronounced prognostic heterogeneity, challenging individualized management. Existing prognostic approaches often rely on static representations from a single postoperative snapshot, and fail to jointly capture tumor spatial distribution, longitudinal disease dynamics, and multimodal clinical information, limiting predictive accuracy. We propose DyPro, a deep learning framework that infers postoperative latent trajectories via residual dynamic evolution. Starting from an initial patient representation, DyPro generates a 12-step sequence of trajectory snapshots through autoregressive residual updates and integrates them to predict recurrence and survival outcomes. On the MSKCC CRLM dataset, DyPro achieves strong discrimination under repeated stratified 5-fold cross-validation, reaching a C-index of 0.755 for OS and 0.714 for DFS, with OS AUC@1y of 0.920 and OS IBS of 0.143. DyPro provides quantitative risk cues to support adjuvant therapy planning and follow-up scheduling.
>
---
#### [replaced 018] LRR-Bench: Left, Right or Rotate? Vision-Language models Still Struggle With Spatial Understanding Tasks
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.20174v2](https://arxiv.org/pdf/2507.20174v2)**

> **作者:** Fei Kong
>
> **摘要:** Real-world applications, such as autonomous driving and humanoid robot manipulation, require precise spatial perception. However, it remains underexplored how Vision-Language Models (VLMs) recognize spatial relationships and perceive spatial movement. In this work, we introduce a spatial evaluation pipeline and construct a corresponding benchmark. Specifically, we categorize spatial understanding into two main types: absolute spatial understanding, which involves querying the absolute spatial position (e.g., left, right) of an object within an image, and 3D spatial understanding, which includes movement and rotation. Notably, our dataset is entirely synthetic, enabling the generation of test samples at a low cost while also preventing dataset contamination. We conduct experiments on multiple state-of-the-art VLMs and observe that there is significant room for improvement in their spatial understanding abilities. Explicitly, in our experiments, humans achieve near-perfect performance on all tasks, whereas current VLMs attain human-level performance only on the two simplest tasks. For the remaining tasks, the performance of VLMs is distinctly lower than that of humans. In fact, the best-performing Vision-Language Models even achieve near-zero scores on multiple tasks. The dataset and code are available on https://github.com/kong13661/LRR-Bench.
>
---
#### [replaced 019] Karhunen-Loève Expansion-Based Residual Anomaly Map for Resource-Efficient Glioma MRI Segmentation
- **分类: q-bio.QM; cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2601.11833v2](https://arxiv.org/pdf/2601.11833v2)**

> **作者:** Anthony Hur
>
> **摘要:** Accurate segmentation of brain tumors is essential for clinical diagnosis and treatment planning. Deep learning is currently the state-of-the-art for brain tumor segmentation, yet it requires either large datasets or extensive computational resources that are inaccessible in most areas. This makes the problem increasingly difficult: state-of-the-art models use thousands of training cases and vast computational power, where performance drops sharply when either is limited. The top performer in the Brats GLI 2023 competition relied on supercomputers trained on over 92,000 augmented MRI scans using an AMD EPYC 7402 CPU, six NVIDIA RTX 6000 GPUs (48GB VRAM each), and 1024GB of RAM over multiple weeks. To address this, the Karhunen--Loève Expansion (KLE) was implemented as a feature extraction step on downsampled, z-score normalized MRI volumes. Each 240$\times$240$\times$155 multi-modal scan is reduced to four $48^3$ channels and compressed into 32 KL coefficients. The resulting approximate reconstruction enables a residual-based anomaly map, which is upsampled and added as a fifth channel to a compact 3D U-Net. All experiments were run on a consumer workstation (AMD Ryzen 5 7600X CPU, RTX 4060Ti (8GB VRAM), and 64GB RAM while using far fewer training cases. This model achieves post-processed Dice scores of 0.929 (WT), 0.856 (TC), and 0.821 (ET), with HD95 distances of 2.93, 6.78, and 10.35 voxels. These results are significantly better than the winning BraTS 2023 methodology for HD95 distances and WT dice scores. This demonstrates that a KLE-based residual anomaly map can dramatically reduce computational cost and data requirements while retaining state-of-the-art performance.
>
---
#### [replaced 020] Training-Free In-Context Forensic Chain for Image Manipulation Detection and Localization
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2510.10111v3](https://arxiv.org/pdf/2510.10111v3)**

> **作者:** Rui Chen; Bin Liu; Changtao Miao; Xinghao Wang; Yi Li; Tao Gong; Qi Chu; Nenghai Yu
>
> **备注:** This version was uploaded in error and contains misleading information found in an early draft. The manuscript requires extensive and long-term revisions
>
> **摘要:** Advances in image tampering pose serious security threats, underscoring the need for effective image manipulation localization (IML). While supervised IML achieves strong performance, it depends on costly pixel-level annotations. Existing weakly supervised or training-free alternatives often underperform and lack interpretability. We propose the In-Context Forensic Chain (ICFC), a training-free framework that leverages multi-modal large language models (MLLMs) for interpretable IML tasks. ICFC integrates an objectified rule construction with adaptive filtering to build a reliable knowledge base and a multi-step progressive reasoning pipeline that mirrors expert forensic workflows from coarse proposals to fine-grained forensics results. This design enables systematic exploitation of MLLM reasoning for image-level classification, pixel-level localization, and text-level interpretability. Across multiple benchmarks, ICFC not only surpasses state-of-the-art training-free methods but also achieves competitive or superior performance compared to weakly and fully supervised approaches.
>
---
#### [replaced 021] Sora as a World Model? A Complete Survey on Text-to-Video Generation
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2403.05131v3](https://arxiv.org/pdf/2403.05131v3)**

> **作者:** Fachrina Dewi Puspitasari; Chaoning Zhang; Joseph Cho; Adnan Haider; Noor Ul Eman; Omer Amin; Alexis Mankowski; Muhammad Umair; Jingyao Zheng; Sheng Zheng; Lik-Hang Lee; Caiyan Qin; Tae-Ho Kim; Choong Seon Hong; Yang Yang; Heng Tao Shen
>
> **备注:** First complete survey on Text-to-Video Generation from World Model perspective, 35 pages
>
> **摘要:** The evolution of video generation from text, from animating MNIST to simulating the world with Sora, has progressed at a breakneck speed. Here, we systematically discuss how far text-to-video generation technology supports essential requirements in world modeling. We curate 250+ studies on text-based video synthesis and world modeling. We then observe that recent models increasingly support spatial, action, and strategic intelligences in world modeling through adherence to completeness, consistency, invention, as well as human interaction and control. We conclude that text-to-video generation is adept at world modeling, although homework in several aspects, such as the diversity-consistency trade-offs, remains to be addressed.
>
---
#### [replaced 022] Can Synthetic Images Serve as Effective and Efficient Class Prototypes?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17160v2](https://arxiv.org/pdf/2512.17160v2)**

> **作者:** Dianxing Shi; Dingjie Fu; Yuqiao Liu; Jun Wang
>
> **备注:** Accepted by IEEE ICASSP2026
>
> **摘要:** Vision-Language Models (VLMs) have shown strong performance in zero-shot image classification tasks. However, existing methods, including Contrastive Language-Image Pre-training (CLIP), all rely on annotated text-to-image pairs for aligning visual and textual modalities. This dependency introduces substantial cost and accuracy requirement in preparing high-quality datasets. At the same time, processing data from two modes also requires dual-tower encoders for most models, which also hinders their lightweight. To address these limitations, we introduce a ``Contrastive Language-Image Pre-training via Large-Language-Model-based Generation (LGCLIP)" framework. LGCLIP leverages a Large Language Model (LLM) to generate class-specific prompts that guide a diffusion model in synthesizing reference images. Afterwards these generated images serve as visual prototypes, and the visual features of real images are extracted and compared with the visual features of these prototypes to achieve comparative prediction. By optimizing prompt generation through the LLM and employing only a visual encoder, LGCLIP remains lightweight and efficient. Crucially, our framework requires only class labels as input during whole experimental procedure, eliminating the need for manually annotated image-text pairs and extra pre-processing. Experimental results validate the feasibility and efficiency of LGCLIP, demonstrating great performance in zero-shot classification tasks and establishing a novel paradigm for classification.
>
---
#### [replaced 023] Chain-of-Thought Compression Should Not Be Blind: V-Skip for Efficient Multimodal Reasoning via Dual-Path Anchoring
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决CoT推理延迟过高问题。通过V-Skip方法，结合视觉锚点优化，实现高效压缩，提升推理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2601.13879v2](https://arxiv.org/pdf/2601.13879v2)**

> **作者:** Dongxu Zhang; Yiding Sun; Cheng Tan; Wenbiao Yan; Ning Yang; Jihua Zhu; Haijun Zhang
>
> **摘要:** While Chain-of-Thought (CoT) reasoning significantly enhances the performance of Multimodal Large Language Models (MLLMs), its autoregressive nature incurs prohibitive latency constraints. Current efforts to mitigate this via token compression often fail by blindly applying text-centric metrics to multimodal contexts. We identify a critical failure mode termed Visual Amnesia, where linguistically redundant tokens are erroneously pruned, leading to hallucinations. To address this, we introduce V-Skip that reformulates token pruning as a Visual-Anchored Information Bottleneck (VA-IB) optimization problem. V-Skip employs a dual-path gating mechanism that weighs token importance through both linguistic surprisal and cross-modal attention flow, effectively rescuing visually salient anchors. Extensive experiments on Qwen2-VL and Llama-3.2 families demonstrate that V-Skip achieves a $2.9\times$ speedup with negligible accuracy loss. Specifically, it preserves fine-grained visual details, outperforming other baselines over 30\% on the DocVQA.
>
---
#### [replaced 024] SUG-Occ: An Explicit Semantics and Uncertainty Guided Sparse Learning Framework for Real-Time 3D Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11396v2](https://arxiv.org/pdf/2601.11396v2)**

> **作者:** Hanlin Wu; Pengfei Lin; Ehsan Javanmardi; Nanren Bao; Bo Qian; Hao Si; Manabu Tsukada
>
> **摘要:** As autonomous driving moves toward full scene understanding, 3D semantic occupancy prediction has emerged as a crucial perception task, offering voxel-level semantics beyond traditional detection and segmentation paradigms. However, such a refined representation for scene understanding incurs prohibitive computation and memory overhead, posing a major barrier to practical real-time deployment. To address this, we propose SUG-Occ, an explicit Semantics and Uncertainty Guided Sparse Learning Enabled 3D Occupancy Prediction Framework, which exploits the inherent sparsity of 3D scenes to reduce redundant computation while maintaining geometric and semantic completeness. Specifically, we first utilize semantic and uncertainty priors to suppress projections from free space during view transformation while employing an explicit unsigned distance encoding to enhance geometric consistency, producing a structurally consistent sparse 3D representation. Secondly, we design an cascade sparse completion module via hyper cross sparse convolution and generative upsampling to enable efficiently coarse-to-fine reasoning. Finally, we devise an object contextual representation (OCR) based mask decoder that aggregates global semantic context from sparse features and refines voxel-wise predictions via lightweight query-context interactions, avoiding expensive attention operations over volumetric features. Extensive experiments on SemanticKITTI benchmark demonstrate that the proposed approach outperforms the baselines, achieving a 7.34/% improvement in accuracy and a 57.8\% gain in efficiency.
>
---
#### [replaced 025] Smudged Fingerprints: A Systematic Evaluation of the Robustness of AI Image Fingerprints
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.11771v2](https://arxiv.org/pdf/2512.11771v2)**

> **作者:** Kai Yao; Marc Juarez
>
> **备注:** This work has been accepted for publication in the 4th IEEE Conference on Secure and Trustworthy Machine Learning (IEEE SaTML 2026). The final version will be available on IEEE Xplore
>
> **摘要:** Model fingerprint detection has shown promise to trace the provenance of AI-generated images in forensic applications. However, despite the inherent adversarial nature of these applications, existing evaluations rarely consider adversarial settings. We present the first systematic security evaluation of these techniques, formalizing threat models that encompass both white- and black-box access and two attack goals: fingerprint removal, which erases identifying traces to evade attribution, and fingerprint forgery, which seeks to cause misattribution to a target model. We implement five attack strategies and evaluate 14 representative fingerprinting methods across RGB, frequency, and learned-feature domains on 12 state-of-the-art image generators. Our experiments reveal a pronounced gap between clean and adversarial performance. Removal attacks are highly effective, often achieving success rates above 80% in white-box settings and over 50% under black-box access. While forgery is more challenging than removal, its success varies significantly across targeted models. We also observe a utility-robustness trade-off: accurate attribution methods are often vulnerable to attacks and, although some techniques are robust in specific settings, none achieves robustness and accuracy across all evaluated threat models. These findings highlight the need for techniques that balance robustness and accuracy, and we identify the most promising approaches toward this goal. Code available at: https://github.com/kaikaiyao/SmudgedFingerprints.
>
---
#### [replaced 026] NumGrad-Pull: Numerical Gradient Guided Tri-plane Representation for Surface Reconstruction from Point Clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.17392v2](https://arxiv.org/pdf/2411.17392v2)**

> **作者:** Ruikai Cui; Binzhu Xie; Shi Qiu; Jiawei Liu; Saeed Anwar; Nick Barnes
>
> **备注:** under review
>
> **摘要:** Reconstructing continuous surfaces from unoriented and unordered 3D points is a fundamental challenge in computer vision and graphics. Recent advancements address this problem by training neural signed distance functions to pull 3D location queries to their closest points on a surface, following the predicted signed distances and the analytical gradients computed by the network. In this paper, we introduce NumGrad-Pull, leveraging the representation capability of tri-plane structures to accelerate the learning of signed distance functions and enhance the fidelity of local details in surface reconstruction. To further improve the training stability of grid-based tri-planes, we propose to exploit numerical gradients, replacing conventional analytical computations. Additionally, we present a progressive plane expansion strategy to facilitate faster signed distance function convergence and design a data sampling strategy to mitigate reconstruction artifacts. Our extensive experiments across a variety of benchmarks demonstrate the effectiveness and robustness of our approach. Code is available at https://github.com/CuiRuikai/NumGrad-Pull
>
---
#### [replaced 027] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
- **分类: cs.CV; cs.AI; cs.LG; q-bio.NC**

- **链接: [https://arxiv.org/pdf/2510.24709v2](https://arxiv.org/pdf/2510.24709v2)**

> **作者:** Yihao Li; Saeed Salehi; Lyle Ungar; Konrad P. Kording
>
> **备注:** Accepted as a Spotlight at NeurIPS 2025
>
> **摘要:** Object binding, the brain's ability to bind the many features that collectively represent an object into a coherent whole, is central to human cognition. It groups low-level perceptual features into high-level object representations, stores those objects efficiently and compositionally in memory, and supports human reasoning about individual object instances. While prior work often imposes object-centric attention (e.g., Slot Attention) explicitly to probe these benefits, it remains unclear whether this ability naturally emerges in pre-trained Vision Transformers (ViTs). Intuitively, they could: recognizing which patches belong to the same object should be useful for downstream prediction and thus guide attention. Motivated by the quadratic nature of self-attention, we hypothesize that ViTs represent whether two patches belong to the same object, a property we term IsSameObject. We decode IsSameObject from patch embeddings across ViT layers using a quadratic similarity probe, which reaches over 90% accuracy. Crucially, this object-binding capability emerges reliably in DINO, CLIP, and ImageNet-supervised ViTs, but is markedly weaker in MAE, suggesting that binding is not a trivial architectural artifact, but an ability acquired through specific pretraining objectives. We further discover that IsSameObject is encoded in a low-dimensional subspace on top of object features, and that this signal actively guides attention. Ablating IsSameObject from model activations degrades downstream performance and works against the learning objective, implying that emergent object binding naturally serves the pretraining objective. Our findings challenge the view that ViTs lack object binding and highlight how symbolic knowledge of "which parts belong together" emerges naturally in a connectionist system.
>
---
#### [replaced 028] VIAFormer: Voxel-Image Alignment Transformer for High-Fidelity Voxel Refinement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.13664v2](https://arxiv.org/pdf/2601.13664v2)**

> **作者:** Tiancheng Fang; Bowen Pan; Lingxi Chen; Jiangjing Lyu; Chengfei Lyu; Chaoyue Niu; Fan Wu
>
> **摘要:** We propose VIAFormer, a Voxel-Image Alignment Transformer model designed for Multi-view Conditioned Voxel Refinement--the task of repairing incomplete noisy voxels using calibrated multi-view images as guidance. Its effectiveness stems from a synergistic design: an Image Index that provides explicit 3D spatial grounding for 2D image tokens, a Correctional Flow objective that learns a direct voxel-refinement trajectory, and a Hybrid Stream Transformer that enables robust cross-modal fusion. Experiments show that VIAFormer establishes a new state of the art in correcting both severe synthetic corruptions and realistic artifacts on the voxel shape obtained from powerful Vision Foundation Models. Beyond benchmarking, we demonstrate VIAFormer as a practical and reliable bridge in real-world 3D creation pipelines, paving the way for voxel-based methods to thrive in large-model, big-data wave.
>
---
#### [replaced 029] Intrinsic Dimensionality as a Model-Free Measure of Class Imbalance
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10475v2](https://arxiv.org/pdf/2511.10475v2)**

> **作者:** Çağrı Eser; Zeynep Sonat Baltacı; Emre Akbaş; Sinan Kalkan
>
> **备注:** 22 pages, 14 figures, Accepted to Neurocomputing
>
> **摘要:** Imbalance in classification tasks is commonly quantified by the cardinalities of examples across classes. This, however, disregards the presence of redundant examples and inherent differences in the learning difficulties of classes. Alternatively, one can use complex measures such as training loss and uncertainty, which, however, depend on training a machine learning model. Our paper proposes using data Intrinsic Dimensionality (ID) as an easy-to-compute, model-free measure of imbalance that can be seamlessly incorporated into various imbalance mitigation methods. Our results across five different datasets with a diverse range of imbalance ratios show that ID consistently outperforms cardinality-based re-weighting and re-sampling techniques used in the literature. Moreover, we show that combining ID with cardinality can further improve performance. Our code and models are available at https://github.com/cagries/IDIM.
>
---
#### [replaced 030] Weakly-supervised segmentation using inherently-explainable classification models and their application to brain tumour classification
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2206.05148v3](https://arxiv.org/pdf/2206.05148v3)**

> **作者:** Soumick Chatterjee; Hadya Yassin; Florian Dubost; Andreas Nürnberger; Oliver Speck
>
> **摘要:** Deep learning has demonstrated significant potential in medical imaging; however, the opacity of "black-box" models hinders clinical trust, while segmentation tasks typically necessitate labourious, hard-to-obtain pixel-wise annotations. To address these challenges simultaneously, this paper introduces a framework for three inherently explainable classifiers (GP-UNet, GP-ShuffleUNet, and GP-ReconResNet). By integrating a global pooling mechanism, these networks generate localisation heatmaps that directly influence classification decisions, offering inherent interpretability without relying on potentially unreliable post-hoc methods. These heatmaps are subsequently thresholded to achieve weakly-supervised segmentation, requiring only image-level classification labels for training. Validated on two datasets for multi-class brain tumour classification, the proposed models achieved a peak F1-score of 0.93. For the weakly-supervised segmentation task, a median Dice score of 0.728 (95% CI 0.715-0.739) was recorded. Notably, on a subset of tumour-only images, the best model achieved an accuracy of 98.7%, outperforming state-of-the-art glioma grading binary classifiers. Furthermore, comparative Precision-Recall analysis validated the framework's robustness against severe class imbalance, establishing a direct correlation between diagnostic confidence and segmentation fidelity. These results demonstrate that the proposed framework successfully combines high diagnostic accuracy with essential transparency, offering a promising direction for trustworthy clinical decision support. Code is available on GitHub: https://github.com/soumickmj/GPModels
>
---
#### [replaced 031] A Constraint Programming Model for the Super-Agile Earth Observation Satellite Imaging Scheduling Problem
- **分类: eess.SY; cs.CV; math.OC**

- **链接: [https://arxiv.org/pdf/2601.11967v2](https://arxiv.org/pdf/2601.11967v2)**

> **作者:** Margarida Caleiras; Samuel Moniz; Paulo Jorge Nascimento
>
> **备注:** 12 pages, 4 figures, To be published in the Proceedings of the International Conference on Operations Research and Enterprise Systems (ICORES 2026)
>
> **摘要:** As the dependence on satellite imaging continues to grow, modern satellites have become increasingly agile, with the new generation, namely super-agile Earth observation satellites (SAEOS), providing unprecedented imaging flexibility. The highly dynamic capabilities of these satellites introduce additional challenges to the scheduling of observation tasks, as existing approaches for conventional agile satellites do not account for variable observation durations and multiple imaging directions. Although some efforts have been made in this regard, the SAEOS imaging scheduling problem (SAEOS-ISP) remains largely unexplored, and no exact approaches have yet been proposed. In this context, this study presents the first exact Constraint Programming formulation for the SAEOS-ISP, considering flexible observation windows, multiple pointing directions and sequence-dependent transition times across multiple satellites. Computational experiments on a newly generated benchmark set demonstrate that the model can be solved efficiently and within very short computational times. Moreover, the results also show that the proposed approach has the potential to achieve higher computational performance compared to the non-exact approaches that are currently considered state-of-the-art.
>
---
#### [replaced 032] Human detectors are surprisingly powerful reward models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.14037v2](https://arxiv.org/pdf/2601.14037v2)**

> **作者:** Kumar Ashutosh; XuDong Wang; Xi Yin; Kristen Grauman; Adam Polyak; Ishan Misra; Rohit Girdhar
>
> **备注:** Technical report
>
> **摘要:** Video generation models have recently achieved impressive visual fidelity and temporal coherence. Yet, they continue to struggle with complex, non-rigid motions, especially when synthesizing humans performing dynamic actions such as sports, dance, etc. Generated videos often exhibit missing or extra limbs, distorted poses, or physically implausible actions. In this work, we propose a remarkably simple reward model, HuDA, to quantify and improve the human motion in generated videos. HuDA integrates human detection confidence for appearance quality, and a temporal prompt alignment score to capture motion realism. We show this simple reward function that leverages off-the-shelf models without any additional training, outperforms specialized models finetuned with manually annotated data. Using HuDA for Group Reward Policy Optimization (GRPO) post-training of video models, we significantly enhance video generation, especially when generating complex human motions, outperforming state-of-the-art models like Wan 2.1, with win-rate of 73%. Finally, we demonstrate that HuDA improves generation quality beyond just humans, for instance, significantly improving generation of animal videos and human-object interactions.
>
---
#### [replaced 033] Geo-Registration of Terrestrial LiDAR Point Clouds with Satellite Images without GNSS
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.05999v4](https://arxiv.org/pdf/2507.05999v4)**

> **作者:** Xinyu Wang; Muhammad Ibrahim; Haitian Wang; Atif Mansoor; Xiuping Jia; Ajmal Mian
>
> **备注:** Submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. Under reviewing now
>
> **摘要:** Accurate geo-registration of LiDAR point clouds remains a significant challenge in urban environments where Global Navigation Satellite System (GNSS) signals are denied or degraded. Existing methods typically rely on real-time GNSS and Inertial Measurement Unit (IMU) data, which require pre-calibration and assume stable signals. However, this assumption often fails in dense cities, resulting in localization errors. To address this, we propose a structured post-hoc geo-registration method that accurately aligns LiDAR point clouds with satellite images. The proposed approach targets point cloud datasets where reliable GNSS information is unavailable or degraded, enabling city-scale geo-registration as a post-processing solution. Our method uses a pre-trained Point Transformer to segment road points, then extracts road skeletons and intersections from the point cloud and the satellite image. Global alignment is achieved through rigid transformation using corresponding intersection points, followed by local non-rigid refinement with radial basis function (RBF) interpolation. Elevation discrepancies are corrected using terrain data from the Shuttle Radar Topography Mission (SRTM). To evaluate geo-registration accuracy, we measure the absolute distances between the roads extracted from the two modalities. Our method is validated on the KITTI benchmark and a newly collected dataset of Perth, Western Australia. On KITTI, our method achieves a mean planimetric alignment error of 0.69m, corresponding to a 50% reduction in global geo-registration bias compared to the raw KITTI annotations. On Perth dataset, it achieves a mean planimetric error of 2.17m from GNSS values extracted from Google Maps, corresponding to 57.4% improvement over rigid alignment. Elevation correlation factor improved by 30.5% (KITTI) and 55.8% (Perth).
>
---
#### [replaced 034] Radially Distorted Homographies, Revisited
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.21190v2](https://arxiv.org/pdf/2508.21190v2)**

> **作者:** Mårten Wadenbäck; Marcus Valtonen Örnhag; Johan Edstedt
>
> **摘要:** Homographies are among the most prevalent transformations occurring in geometric computer vision and projective geometry, and homography estimation is consequently a crucial step in a wide assortment of computer vision tasks. When working with real images, which are often afflicted with geometric distortions caused by the camera lens, it may be necessary to determine both the homography and the lens distortion-particularly the radial component, called radial distortion-simultaneously to obtain anything resembling useful estimates. When considering a homography with radial distortion between two images, there are three conceptually distinct configurations for the radial distortion; (i) distortion in only one image, (ii) identical distortion in the two images, and (iii) independent distortion in the two images. While these cases have been addressed separately in the past, the present paper provides a novel and unified approach to solve all three cases. We demonstrate how the proposed approach can be used to construct new fast, stable, and accurate minimal solvers for radially distorted homographies. In all three cases, our proposed solvers are faster than the existing state-of-the-art solvers while maintaining similar accuracy. The solvers are tested on well-established benchmarks including images taken with fisheye cameras. A reference implementation of the proposed solvers is made available as part of HomLib (https://github.com/marcusvaltonen/HomLib).
>
---
#### [replaced 035] Improving Artifact Robustness for CT Deep Learning Models Without Labeled Artifact Images via Domain Adaptation
- **分类: cs.CV; q-bio.TO**

- **链接: [https://arxiv.org/pdf/2510.06584v2](https://arxiv.org/pdf/2510.06584v2)**

> **作者:** Justin Cheung; Samuel Savine; Calvin Nguyen; Lin Lu; Alhassan S. Yasin
>
> **备注:** 8 pages, 12 figures, 1 table
>
> **摘要:** If a CT scanner introduces a new artifact not present in the training labels, the model may misclassify the images. Although modern CT scanners include design features which mitigate these artifacts, unanticipated or difficult-to-mitigate artifacts can still appear in practice. The direct solution of labeling images from this new distribution can be costly. As a more accessible alternative, this study evaluates domain adaptation as an approach for training models that maintain classification performance despite new artifacts, even without corresponding labels. We simulate ring artifacts from detector gain error in sinogram space and evaluate domain adversarial neural networks (DANN) against baseline and augmentation-based approaches on the OrganAMNIST abdominal CT dataset. We simulate the absence of labels from an unseen distribution via masking in the loss function and selectively detaching unlabeled instances from the computational graph. Our results demonstrate that baseline models trained only on clean images fail to generalize to images with ring artifacts, and traditional augmentation with other distortion types provides no improvement on unseen artifact domains. In contrast, the DANN approach improves classification accuracy on ring artifact images using only unlabeled artifact data during training, demonstrating the viability of domain adaptation for artifact robustness. The domain-adapted model achieved a classification accuracy of 77.4% on ring artifact test data, 38.7% higher than a baseline model only trained on images with no artifact. These findings provide empirical evidence that domain adaptation can effectively address distribution shift in medical imaging without requiring expensive expert labeling of new artifact distributions, suggesting promise for deployment in clinical settings where novel artifacts may emerge.
>
---
#### [replaced 036] Trustworthy Longitudinal Brain MRI Completion: A Deformation-Based Approach with KAN-Enhanced Diffusion Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.09572v2](https://arxiv.org/pdf/2601.09572v2)**

> **作者:** Tianli Tao; Ziyang Wang; Delong Yang; Han Zhang; Le Zhang
>
> **摘要:** Longitudinal brain MRI is essential for lifespan study, yet high attrition rates often lead to missing data, complicating analysis. Deep generative models have been explored, but most rely solely on image intensity, leading to two key limitations: 1) the fidelity or trustworthiness of the generated brain images are limited, making downstream studies questionable; 2) the usage flexibility is restricted due to fixed guidance rooted in the model structure, restricting full ability to versatile application scenarios. To address these challenges, we introduce DF-DiffCom, a Kolmogorov-Arnold Networks (KAN)-enhanced diffusion model that smartly leverages deformation fields for trustworthy longitudinal brain image completion. Trained on OASIS-3, DF-DiffCom outperforms state-of-the-art methods, improving PSNR by 5.6% and SSIM by 0.12. More importantly, its modality-agnostic nature allows smooth extension to varied MRI modalities, even to attribute maps such as brain tissue segmentation results.
>
---
#### [replaced 037] A Training-Free Guess What Vision Language Model from Snippets to Open-Vocabulary Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11910v2](https://arxiv.org/pdf/2601.11910v2)**

> **作者:** Guiying Zhu; Bowen Yang; Yin Zhuang; Tong Zhang; Guanqun Wang; Zhihao Che; He Chen; Lianlin Li
>
> **摘要:** Open-Vocabulary Object Detection (OVOD) aims to develop the capability to detect anything. Although myriads of large-scale pre-training efforts have built versatile foundation models that exhibit impressive zero-shot capabilities to facilitate OVOD, the necessity of creating a universal understanding for any object cognition according to already pretrained foundation models is usually overlooked. Therefore, in this paper, a training-free Guess What Vision Language Model, called GW-VLM, is proposed to form a universal understanding paradigm based on our carefully designed Multi-Scale Visual Language Searching (MS-VLS) coupled with Contextual Concept Prompt (CCP) for OVOD. This approach can engage a pre-trained Vision Language Model (VLM) and a Large Language Model (LLM) in the game of "guess what". Wherein, MS-VLS leverages multi-scale visual-language soft-alignment for VLM to generate snippets from the results of class-agnostic object detection, while CCP can form the concept of flow referring to MS-VLS and then make LLM understand snippets for OVOD. Finally, the extensive experiments are carried out on natural and remote sensing datasets, including COCO val, Pascal VOC, DIOR, and NWPU-10, and the results indicate that our proposed GW-VLM can achieve superior OVOD performance compared to the-state-of-the-art methods without any training step.
>
---
#### [replaced 038] PanoDreamer: Optimization-Based Single Image to 360 3D Scene With Diffusion
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2412.04827v3](https://arxiv.org/pdf/2412.04827v3)**

> **作者:** Avinash Paliwal; Xilong Zhou; Andrii Tsarov; Nima Khademi Kalantari
>
> **备注:** SIGGRAPH Asia 2025, Project page: https://people.engr.tamu.edu/nimak/Papers/PanoDreamer, Code: https://github.com/avinashpaliwal/PanoDreamer
>
> **摘要:** In this paper, we present PanoDreamer, a novel method for producing a coherent 360° 3D scene from a single input image. Unlike existing methods that generate the scene sequentially, we frame the problem as single-image panorama and depth estimation. Once the coherent panoramic image and its corresponding depth are obtained, the scene can be reconstructed by inpainting the small occluded regions and projecting them into 3D space. Our key contribution is formulating single-image panorama and depth estimation as two optimization tasks and introducing alternating minimization strategies to effectively solve their objectives. We demonstrate that our approach outperforms existing techniques in single-image 360° 3D scene reconstruction in terms of consistency and overall quality.
>
---
#### [replaced 039] KBE-DME: Dynamic Multimodal Evaluation via Knowledge Enhanced Benchmark Evolution
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态模型评估任务，解决静态基准数据污染和饱和问题。提出KBE框架，通过知识增强实现动态基准演化，提升评估可靠性与全面性。**

- **链接: [https://arxiv.org/pdf/2510.21182v2](https://arxiv.org/pdf/2510.21182v2)**

> **作者:** Junzhe Zhang; Huixuan Zhang; Xiaojun Wan
>
> **摘要:** The rapid progress of multimodal large language models (MLLMs) calls for more reliable evaluation protocols. Existing static benchmarks suffer from the potential risk of data contamination and saturation, leading to inflated or misleading performance evaluations. To address these issues, we first apply Graph formulation to represent a static or dynamic VQA sample. With the formulation, we propose Knowledge-enhanced Benchmark Evolution(KBE), a dynamic multimodal evaluation framework. KBE first analyzes the original static benchmark, then expands it by integrating multimodal knowledge, transforming the static benchmark into a controllable, dynamic evolving version. Crucially, KBE can both reconstruct questions by Re-selecting visual information in the original image and expand existing questions with external textual knowledge. It enables difficulty-controllable evaluation by adjusting the degree of question exploration. Extensive experiments demonstrate that KBE alleviates the risk of data contamination, data saturation, and provides a more comprehensive assessment of MLLM capabilities.
>
---
#### [replaced 040] Omni-AVSR: Towards Unified Multimodal Speech Recognition with Large Language Models
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 论文提出Omni-AVSR，解决多模态语音识别任务中的模型独立训练与资源消耗问题，通过统一框架实现ASR、VSR和AVSR的高效联合训练与推理。**

- **链接: [https://arxiv.org/pdf/2511.07253v2](https://arxiv.org/pdf/2511.07253v2)**

> **作者:** Umberto Cappellazzo; Xubo Liu; Pingchuan Ma; Stavros Petridis; Maja Pantic
>
> **备注:** Accepted to IEEE ICASSP 2026 (camera-ready version). Project website (code and model weights): https://umbertocappellazzo.github.io/Omni-AVSR/
>
> **摘要:** Large language models (LLMs) have recently achieved impressive results in speech recognition across multiple modalities, including Auditory Speech Recognition (ASR), Visual Speech Recognition (VSR), and Audio-Visual Speech Recognition (AVSR). Despite this progress, current LLM-based approaches typically address each task independently, training separate models that raise computational and deployment resource use while missing potential cross-task synergies. They also rely on fixed-rate token compression, which restricts flexibility in balancing accuracy with efficiency. These limitations highlight the need for a unified framework that can support ASR, VSR, and AVSR while enabling elastic inference. To this end, we present Omni-AVSR, a unified audio-visual LLM that combines efficient multi-granularity training with parameter-efficient adaptation. Specifically, we adapt the matryoshka representation learning paradigm to efficiently train across multiple audio and visual granularities, reducing its inherent training resource use. Furthermore, we explore three LoRA-based strategies for adapting the backbone LLM, balancing shared and task-specific specialization. Experiments on LRS2 and LRS3 show that Omni-AVSR achieves comparable or superior accuracy to state-of-the-art baselines while training a single model at substantially lower training and deployment resource use. The model also remains robust under acoustic noise, and we analyze its scaling behavior as LLM size increases, providing insights into the trade-off between performance and efficiency.
>
---
#### [replaced 041] Unified Text-Image Generation with Weakness-Targeted Post-Training
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.04339v2](https://arxiv.org/pdf/2601.04339v2)**

> **作者:** Jiahui Chen; Philippe Hansen-Estruch; Xiaochuang Han; Yushi Hu; Emily Dinan; Amita Kamath; Michal Drozdzal; Reyhane Askari-Hemmat; Luke Zettlemoyer; Marjan Ghazvininejad
>
> **摘要:** Unified multimodal generation architectures that jointly produce text and images have recently emerged as a promising direction for text-to-image (T2I) synthesis. However, many existing systems rely on explicit modality switching, generating reasoning text before switching manually to image generation. This separate, sequential inference process limits cross-modal coupling and prohibits automatic multimodal generation. This work explores post-training to achieve fully unified text-image generation, where models autonomously transition from textual reasoning to visual synthesis within a single inference process. We examine the impact of joint text-image generation on T2I performance and the relative importance of each modality during post-training. We additionally explore different post-training data strategies, showing that a targeted dataset addressing specific limitations achieves superior results compared to broad image-caption corpora or benchmark-aligned data. Using offline, reward-weighted post-training with fully self-generated synthetic data, our approach enables improvements in multimodal image generation across four diverse T2I benchmarks, demonstrating the effectiveness of reward-weighting both modalities and strategically designed post-training data.
>
---
#### [replaced 042] Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.07301v2](https://arxiv.org/pdf/2511.07301v2)**

> **作者:** Huizai Yao; Sicheng Zhao; Pengteng Li; Yi Cui; Shuo Lu; Weiyu Guo; Yunfan Lu; Yijie Xu; Hui Xiong
>
> **备注:** Accepted to AAAI 2026. Extended version with full Appendix
>
> **摘要:** Source-Free Object Detection (SFOD) aims to adapt a source-pretrained object detector to a target domain without access to source data. However, existing SFOD methods predominantly rely on internal knowledge from the source model, which limits their capacity to generalize across domains and often results in biased pseudo-labels, thereby hindering both transferability and discriminability. In contrast, Vision Foundation Models (VFMs), pretrained on massive and diverse data, exhibit strong perception capabilities and broad generalization, yet their potential remains largely untapped in the SFOD setting. In this paper, we propose a novel SFOD framework that leverages VFMs as external knowledge sources to jointly enhance feature alignment and label quality. Specifically, we design three VFM-based modules: (1) Patch-weighted Global Feature Alignment (PGFA) distills global features from VFMs using patch-similarity-based weighting to enhance global feature transferability; (2) Prototype-based Instance Feature Alignment (PIFA) performs instance-level contrastive learning guided by momentum-updated VFM prototypes; and (3) Dual-source Enhanced Pseudo-label Fusion (DEPF) fuses predictions from detection VFMs and teacher models via an entropy-aware strategy to yield more reliable supervision. Extensive experiments on six benchmarks demonstrate that our method achieves state-of-the-art SFOD performance, validating the effectiveness of integrating VFMs to simultaneously improve transferability and discriminability.
>
---
#### [replaced 043] Extendable Generalization Self-Supervised Diffusion for Low-Dose CT Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.23885v3](https://arxiv.org/pdf/2509.23885v3)**

> **作者:** Guoquan Wei; Liu Shi; Zekun Zhou; Mohan Li; Cunfeng Wei; Wenzhe Shan; Qiegen Liu
>
> **摘要:** Current methods based on deep learning for self-supervised low-dose CT (LDCT) reconstruction, while reducing the dependence on paired data, face the problem of significantly decreased generalization when training with single-dose data and extending to other doses. To enable dose-extensive generalization using only single-dose projection data for training, this work proposes a novel method of Extendable GENeraLization self-supervised Diffusion (EGenDiff) for low-dose CT reconstruction. Specifically, a contextual subdata self-enhancing similarity strategy is designed to provide an initial prior for the subsequent progress. During training, the initial prior is used to combine knowledge distillation with a deep combination of latent diffusion models for optimizing image details. On the stage of inference, the pixel-wise self-correcting fusion technique is proposed for data fidelity enhancement, resulting in extensive generalization of higher and lower doses or even unseen doses. EGenDiff requires only LDCT projection data for training and testing. Comprehensive evaluation on benchmark datasets, clinical data, photon counting CT data, and across all three anatomical planes (transverse, coronal, and sagittal) demonstrates that EGenDiff enables extendable generalization multi-dose, yielding reconstructions that consistently outperform leading existing methods.
>
---
#### [replaced 044] Registration-Free Monitoring of Unstructured Point Cloud Data via Intrinsic Geometrical Properties
- **分类: cs.CV; cs.LG; stat.ME; stat.ML**

- **链接: [https://arxiv.org/pdf/2511.05623v2](https://arxiv.org/pdf/2511.05623v2)**

> **作者:** Mariafrancesca Patalano; Giovanna Capizzi; Kamran Paynabar
>
> **备注:** Code available at https://github.com/franci2312/RFM
>
> **摘要:** Modern sensing technologies have enabled the collection of unstructured point cloud data (PCD) of varying sizes, which are used to monitor the geometric accuracy of 3D objects. PCD are widely applied in advanced manufacturing processes, including additive, subtractive, and hybrid manufacturing. To ensure the consistency of analysis and avoid false alarms, preprocessing steps such as registration and mesh reconstruction are commonly applied prior to monitoring. However, these steps are error-prone, time-consuming and may introduce artifacts, potentially affecting monitoring outcomes. In this paper, we present a novel registration-free approach for monitoring PCD of complex shapes, eliminating the need for both registration and mesh reconstruction. Our proposal consists of two alternative feature learning methods and a common monitoring scheme designed to handle hundreds of features. Feature learning methods leverage intrinsic geometric properties of the shape, captured via the Laplacian and geodesic distances. In the monitoring scheme, thresholding techniques are used to further select intrinsic features most indicative of potential out-of-control conditions. Numerical experiments and case studies highlight the effectiveness of the proposed approach in identifying different types of defects.
>
---
#### [replaced 045] Difference Decomposition Networks for Infrared Small Target Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03470v3](https://arxiv.org/pdf/2512.03470v3)**

> **作者:** Chen Hu; Mingyu Zhou; Shuai Yuan; Hongbo Hu; Zhenming Peng; Tian Pu; Xiyin Li
>
> **摘要:** Infrared small target detection (ISTD) faces two major challenges: a lack of discernible target texture and severe background clutter, which results in the background obscuring the target. To enhance targets and suppress backgrounds, we propose the Basis Decomposition Module (BDM) as an extensible and lightweight module based on basis decomposition, which decomposes a complex feature into several basis features and enhances certain information while eliminating redundancy. Extending BDM leads to a series of modules, including the Spatial Difference Decomposition Module (SD$^\mathrm{2}$M), Spatial Difference Decomposition Downsampling Module (SD$^\mathrm{3}$M), and Temporal Difference Decomposition Module (TD$^\mathrm{2}$M). Based on these modules, we develop the Spatial Difference Decomposition Network (SD$^\mathrm{2}$Net) for single-frame ISTD (SISTD) and the Spatiotemporal Difference Decomposition Network (STD$^\mathrm{2}$Net) for multi-frame ISTD (MISTD). SD$^\mathrm{2}$Net integrates SD$^\mathrm{2}$M and SD$^\mathrm{3}$M within an adapted U-shaped architecture. We employ TD$^\mathrm{2}$M to introduce motion information, which transforms SD$^\mathrm{2}$Net into STD$^\mathrm{2}$Net. Extensive experiments on SISTD and MISTD datasets demonstrate state-of-the-art (SOTA) performance. On the SISTD task, SD$^\mathrm{2}$Net performs well compared to most established networks. On the MISTD datasets, STD$^\mathrm{2}$Net achieves a mIoU of 87.68\%, outperforming SD$^\mathrm{2}$Net, which achieves a mIoU of 64.97\%. Our codes are available: https://github.com/greekinRoma/IRSTD_HC_Platform.
>
---
#### [replaced 046] ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction With Fewer Primitives
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06810v2](https://arxiv.org/pdf/2511.06810v2)**

> **作者:** Bartłomiej Baranowski; Stefano Esposito; Patricia Gschoßmann; Anpei Chen; Andreas Geiger
>
> **摘要:** 3D Gaussian Splatting (3DGS) achieves state-of-the-art image quality and real-time performance in novel view synthesis but often suffers from a suboptimal spatial distribution of primitives. This issue stems from cloning-based densification, which propagates Gaussians along existing geometry, limiting exploration and requiring many primitives to adequately cover the scene. We present ConeGS, an image-space-informed densification framework that is independent of existing scene geometry state. ConeGS first creates a fast Instant Neural Graphics Primitives (iNGP) reconstruction as a geometric proxy to estimate per-pixel depth. During the subsequent 3DGS optimization, it identifies high-error pixels and inserts new Gaussians along the corresponding viewing cones at the predicted depth values, initializing their size according to the cone diameter. A pre-activation opacity penalty rapidly removes redundant Gaussians, while a primitive budgeting strategy controls the total number of primitives, either by a fixed budget or by adapting to scene complexity, ensuring high reconstruction quality. Experiments show that ConeGS consistently enhances reconstruction quality and rendering performance across Gaussian budgets, with especially strong gains under tight primitive constraints where efficient placement is crucial.
>
---
#### [replaced 047] A Multi-Stage Augmented Multimodal Interaction Network for Quantifying Fish Feeding Intensity Using Feeding Image, Audio and Water Wave
- **分类: cs.CV; cs.AI; cs.ET**

- **链接: [https://arxiv.org/pdf/2506.14170v2](https://arxiv.org/pdf/2506.14170v2)**

> **作者:** Shulong Zhang; Mingyuan Yao; Jiayin Zhao; Daoliang Li; Yingyi Chen; Haihua Wang
>
> **摘要:** In recirculating aquaculture systems, accurate and effective assessment of fish feeding intensity is crucial for reducing feed costs and calculating optimal feeding times. However, current studies have limitations in modality selection, feature extraction and fusion, and co-inference for decision making, which restrict further improvement in the accuracy, applicability and reliability of multimodal fusion models. To address this problem, this study proposes a Multi-stage Augmented Multimodal Interaction Network (MAINet) for quantifying fish feeding intensity. Firstly, a general feature extraction framework is proposed to efficiently extract feature information from input image, audio and water wave datas. Second, an Auxiliary-modality Reinforcement Primary-modality Mechanism (ARPM) is designed for inter-modal interaction and generate enhanced features, which consists of a Channel Attention Fusion Network (CAFN) and a Dual-mode Attention Fusion Network (DAFN). Finally, an Evidence Reasoning (ER) rule is introduced to fuse the output results of each modality and make decisions, thereby completing the quantification of fish feeding intensity. The experimental results show that the constructed MAINet reaches 96.76%, 96.78%, 96.79% and 96.79% in accuracy, precision, recall and F1-Score respectively, and its performance is significantly higher than the comparison models. Compared with models that adopt single-modality, dual-modality fusion and different decision-making fusion methods, it also has obvious advantages. Meanwhile, the ablation experiments further verified the key role of the proposed improvement strategy in improving the robustness and feature utilization efficiency of model, which can effectively improve the accuracy of the quantitative results of fish feeding intensity. The dataset is available at: https://huggingface.co/datasets/ShulongZhang/Multimodal_Fish_Feeding_Intensity.
>
---
#### [replaced 048] RealX3D: A Physically-Degraded 3D Benchmark for Multi-view Visual Restoration and Reconstruction
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2512.23437v2](https://arxiv.org/pdf/2512.23437v2)**

> **作者:** Shuhong Liu; Chenyu Bao; Ziteng Cui; Yun Liu; Xuangeng Chu; Lin Gu; Marcos V. Conde; Ryo Umagami; Tomohiro Hashimoto; Zijian Hu; Tianhan Xu; Yuan Gan; Yusuke Kurose; Tatsuya Harada
>
> **摘要:** We introduce RealX3D, a real-capture benchmark for multi-view visual restoration and 3D reconstruction under diverse physical degradations. RealX3D groups corruptions into four families, including illumination, scattering, occlusion, and blurring, and captures each at multiple severity levels using a unified acquisition protocol that yields pixel-aligned LQ/GT views. Each scene includes high-resolution capture, RAW images, and dense laser scans, from which we derive world-scale meshes and metric depth. Benchmarking a broad range of optimization-based and feed-forward methods shows substantial degradation in reconstruction quality under physical corruptions, underscoring the fragility of current multi-view pipelines in real-world challenging environments.
>
---
#### [replaced 049] Unlocking Generalization in Polyp Segmentation with DINO Self-Attention "keys"
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13376v2](https://arxiv.org/pdf/2512.13376v2)**

> **作者:** Carla Monteiro; Valentina Corbetta; Regina Beets-Tan; Luís F. Teixeira; Wilson Silva
>
> **备注:** We have found a bug in our codebase. The DINO vision encoder was not properly frozen, therefore the results and claims are not fully valid. We are working on new results
>
> **摘要:** Automatic polyp segmentation is crucial for improving the clinical identification of colorectal cancer (CRC). While Deep Learning (DL) techniques have been extensively researched for this problem, current methods frequently struggle with generalization, particularly in data-constrained or challenging settings. Moreover, many existing polyp segmentation methods rely on complex, task-specific architectures. To address these limitations, we present a framework that leverages the intrinsic robustness of DINO self-attention "key" features for robust segmentation. Unlike traditional methods that extract tokens from the deepest layers of the Vision Transformer (ViT), our approach leverages the key features of the self-attention module with a simple convolutional decoder to predict polyp masks, resulting in enhanced performance and better generalizability. We validate our approach using a multi-center dataset under two rigorous protocols: Domain Generalization (DG) and Extreme Single Domain Generalization (ESDG). Our results, supported by a comprehensive statistical analysis, demonstrate that this pipeline achieves state-of-the-art (SOTA) performance, significantly enhancing generalization, particularly in data-scarce and challenging scenarios. While avoiding a polyp-specific architecture, we surpass well-established models like nnU-Net and UM-Net. Additionally, we provide a systematic benchmark of the DINO framework's evolution, quantifying the specific impact of architectural advancements on downstream polyp segmentation performance.
>
---
#### [replaced 050] CRAFT: Continuous Reasoning and Agentic Feedback Tuning for Multimodal Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20362v2](https://arxiv.org/pdf/2512.20362v2)**

> **作者:** V. Kovalev; A. Kuvshinov; A. Buzovkin; D. Pokidov; D. Timonin
>
> **备注:** 37 pages, 42 figures
>
> **摘要:** Recent work has shown that inference-time reasoning and reflection can improve text-to-image generation without retraining. However, existing approaches often rely on implicit, holistic critiques or unconstrained prompt rewrites, making their behavior difficult to interpret, control, or stop reliably. In contrast, large language models have benefited from explicit, structured forms of **thinking** based on verification, targeted correction, and early stopping. We introduce CRAFT (Continuous Reasoning and Agentic Feedback Tuning), a training-free and model-agnostic framework for multimodal image generation. CRAFT transforms a user prompt into a set of explicit, dependency-structured visual constraints, verifies generated images using a vision-language model, and performs targeted prompt updates only when specific constraints are violated. This iterative process includes an explicit stopping criterion, resulting in an interpretable and controllable inference-time refinement loop. Across multiple model families and challenging benchmarks, CRAFT consistently improves compositional accuracy, text rendering, and preference-based evaluations, with particularly strong gains for lightweight generators. Importantly, these improvements incur only a negligible inference-time overhead, allowing smaller or cheaper models to approach the quality of substantially more expensive systems. Our results suggest that explicitly structured, constraint-driven inference-time reasoning is a key ingredient for improving the reliability of multimodal generative models.
>
---
#### [replaced 051] RiskCueBench: Benchmarking Anticipatory Reasoning from Early Risk Cues in Video-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出RiskCueBench基准，用于评估视频语言模型从早期风险线索中进行预见性推理的能力，旨在解决实时风险预测中的挑战。**

- **链接: [https://arxiv.org/pdf/2601.03369v2](https://arxiv.org/pdf/2601.03369v2)**

> **作者:** Sha Luo; Yogesh Prabhu; Timothy Ossowski; Kaiping Chen; Junjie Hu
>
> **备注:** *updated author email in this version
>
> **摘要:** With the rapid growth of video centered social media, the ability to anticipate risky events from visual data is a promising direction for ensuring public safety and preventing real world accidents. Prior work has extensively studied supervised video risk assessment across domains such as driving, protests, and natural disasters. However, many existing datasets provide models with access to the full video sequence, including the accident itself, which substantially reduces the difficulty of the task. To better reflect real world conditions, we introduce a new video understanding benchmark RiskCueBench in which videos are carefully annotated to identify a risk signal clip, defined as the earliest moment that indicates a potential safety concern. Experimental results reveal a significant gap in current systems ability to interpret evolving situations and anticipate future risky events from early visual signals, highlighting important challenges for deploying video risk prediction models in practice.
>
---
#### [replaced 052] GAIA: A Global, Multi-modal, Multi-scale Vision-Language Dataset for Remote Sensing Image Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.09598v2](https://arxiv.org/pdf/2502.09598v2)**

> **作者:** Angelos Zavras; Dimitrios Michail; Xiao Xiang Zhu; Begüm Demir; Ioannis Papoutsis
>
> **备注:** 26 pages, 14 figures
>
> **摘要:** Existing Vision-Language Models (VLMs) are predominantly trained on web-scraped, noisy image-text data, exhibiting limited exposure to the specialized domain of RS. This deficiency results in poor performance on RS-specific tasks, as commonly used datasets often lack detailed, scientifically accurate textual descriptions and instead emphasize solely on attributes like date and location. To bridge this critical gap, we introduce GAIA, a novel dataset designed for multi-scale, multi-sensor, and multi-modal RS image analysis. GAIA comprises of 201,005 meticulously curated RS image-text pairs, representing a diverse range of RS modalities associated to different spatial resolutions. Unlike existing vision-language datasets in RS, GAIA specifically focuses on capturing a diverse range of RS applications, providing unique information about environmental changes, natural disasters, and various other dynamic phenomena. The dataset provides a spatially and temporally balanced distribution, spanning across the globe, covering the last 25 years with a balanced temporal distribution of observations. GAIA's construction involved a two-stage process: (1) targeted web-scraping of images and accompanying text from reputable RS-related sources, and (2) generation of five high-quality, scientifically grounded synthetic captions for each image using carefully crafted prompts that leverage the advanced vision-language capabilities of GPT-4o. Our extensive experiments, including fine-tuning of CLIP and BLIP2 models, demonstrate that GAIA significantly improves performance on RS image classification, cross-modal retrieval and image captioning tasks. We make our dataset, automated processing framework and fine-tuned model weights publicly available on our project's GitHub repository: https://github.com/Orion-AI-Lab/GAIA.
>
---
