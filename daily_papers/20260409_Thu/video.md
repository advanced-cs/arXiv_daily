# 计算机视觉 cs.CV

- **最新发布 127 篇**

- **更新 90 篇**

## 最新发布

#### [new 001] LiveStre4m: Feed-Forward Live Streaming of Novel Views from Unposed Multi-View Video
- **分类: cs.CV**

- **简介: 该论文属于实时新颖视角合成任务，解决无标定多视角视频直播中的动态场景表示问题。提出LiveStre4m方法，实现快速、稳定的实时新视角视频流。**

- **链接: [https://arxiv.org/pdf/2604.06740](https://arxiv.org/pdf/2604.06740)**

> **作者:** Pedro Quesado; Erkut Akdag; Yasaman Kashefbahrami; Willem Menu; Egor Bondarev
>
> **摘要:** Live-streaming Novel View Synthesis (NVS) from unposed multi-view video remains an open challenge in a wide range of applications. Existing methods for dynamic scene representation typically require ground-truth camera parameters and involve lengthy optimizations ($\approx 2.67$s), which makes them unsuitable for live streaming scenarios. To address this issue, we propose a novel viewpoint video live-streaming method (LiveStre4m), a feed-forward model for real-time NVS from unposed sparse multi-view inputs. LiveStre4m introduces a multi-view vision transformer for keyframe 3D scene reconstruction coupled with a diffusion-transformer interpolation module that ensures temporal consistency and stable streaming. In addition, a Camera Pose Predictor module is proposed to efficiently estimate both poses and intrinsics directly from RGB images, removing the reliance on known camera calibration information. Our approach enables temporally consistent novel-view video streaming in real-time using as few as two synchronized unposed input streams. LiveStre4m attains an average reconstruction time of $ 0.07$s per-frame at $ 1024 \times 768$ resolution, outperforming the optimization-based dynamic scene representation methods by orders of magnitude in runtime. These results demonstrate that LiveStre4m makes real-time NVS streaming feasible in practical settings, marking a substantial step toward deployable live novel-view synthesis systems. Code available at: this https URL
>
---
#### [new 002] USCNet: Transformer-Based Multimodal Fusion with Segmentation Guidance for Urolithiasis Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决肾结石术前快速分类问题。通过融合CT图像与电子病历数据，提出USCNet模型实现精准分类。**

- **链接: [https://arxiv.org/pdf/2604.07141](https://arxiv.org/pdf/2604.07141)**

> **作者:** Changmiao Wang; Songqi Zhang; Yongquan Zhang; Yifei Wang; Liya Liu; Nannan Li; Xingzhi Li; Jiexin Pan; Yi Jiang; Xiang Wan; Hai Wang; Ahmed Elazab
>
> **备注:** Accepted by IEEE Journal of Biomedical and Health Informatics. Early Access
>
> **摘要:** Kidney stone disease ranks among the most prevalent conditions in urology, and understanding the composition of these stones is essential for creating personalized treatment plans and preventing recurrence. Current methods for analyzing kidney stones depend on postoperative specimens, which prevents rapid classification before surgery. To overcome this limitation, we introduce a new approach called the Urinary Stone Segmentation and Classification Network (USCNet). This innovative method allows for precise preoperative classification of kidney stones by integrating Computed Tomography (CT) images with clinical data from Electronic Health Records (EHR). USCNet employs a Transformer-based multimodal fusion framework with CT-EHR attention and segmentation-guided attention modules for accurate classification. Moreover, a dynamic loss function is introduced to effectively balance the dual objectives of segmentation and classification. Experiments on an in-house kidney stone dataset show that USCNet demonstrates outstanding performance across all evaluation metrics, with its classification efficacy significantly surpassing existing mainstream methods. This study presents a promising solution for the precise preoperative classification of kidney stones, offering substantial clinical benefits. The source code has been made publicly available: this https URL.
>
---
#### [new 003] Energy-Regularized Spatial Masking: A Novel Approach to Enhancing Robustness and Interpretability in Vision Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉模型优化任务，旨在提升模型的鲁棒性和可解释性。通过引入能量正则化空间掩码机制，自动发现信息密度平衡，实现高效特征选择与结构化遮挡鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.06893](https://arxiv.org/pdf/2604.06893)**

> **作者:** Tom Devynck Bilal Faye Djamel Bouchaffra Nadjib Lazaar Hanane Azzag Mustapha Lebbah
>
> **摘要:** Deep convolutional neural networks achieve remarkable performance by exhaustively processing dense spatial feature maps, yet this brute-force strategy introduces significant computational redundancy and encourages reliance on spurious background correlations. As a result, modern vision models remain brittle and difficult to interpret. We propose Energy-Regularized Spatial Masking (ERSM), a novel framework that reformulates feature selection as a differentiable energy minimization problem. By embedding a lightweight Energy-Mask Layer inside standard convolutional backbones, each visual token is assigned a scalar energy composed of two competing forces: an intrinsic Unary importance cost and a Pairwise spatial coherence penalty. Unlike prior pruning methods that enforce rigid sparsity budgets or rely on heuristic importance scores, ERSM allows the network to autonomously discover an optimal information-density equilibrium tailored to each input. We validate ERSM on convolutional architectures and demonstrate that it produces emergent sparsity, improved robustness to structured occlusion, and highly interpretable spatial masks, while preserving classification accuracy. Furthermore, we show that the learned energy ranking significantly outperforms magnitude-based pruning in deletion-based robustness tests, revealing ERSM as an intrinsic denoising mechanism that isolates semantic object regions without pixel-level supervision.
>
---
#### [new 004] VGGT-SLAM++
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出VGGT-SLAM++，属于视觉SLAM任务，解决大场景下定位与建图的精度和效率问题，通过融合VGGT和DEM实现高精度、低漂移的实时定位与地图构建。**

- **链接: [https://arxiv.org/pdf/2604.06830](https://arxiv.org/pdf/2604.06830)**

> **作者:** Avilasha Mandal; Rajesh Kumar; Sudarshan Sunil Harithas; Chetan Arora
>
> **备注:** 8 pages (main paper) + supplementary material. Accepted at CVPR 2026 Workshop (VOCVALC)
>
> **摘要:** We introduce VGGT-SLAM++, a complete visual SLAM system that leverages the geometry-rich outputs of the Visual Geometry Grounded Transformer (VGGT). The system comprises a visual odometry (front-end) fusing the VGGT feed-forward transformer and a Sim(3) solution, a Digital Elevation Map (DEM)-based graph construction module, and a back-end that jointly enable accurate large-scale mapping with bounded memory. While prior transformer-based SLAM pipelines such as VGGT-SLAM rely primarily on sparse loop closures or global Sim(3) manifold constraints - allowing short-horizon pose drift - VGGT-SLAM++ restores high-cadence local bundle adjustment (LBA) through a spatially corrective back-end. For each VGGT submap, we construct a dense planar-canonical DEM, partition it into patches, and compute their DINOv2 embeddings to integrate the submap into a covisibility graph. Spatial neighbors are retrieved using a Visual Place Recognition (VPR) module within the covisibility window, triggering frequent local optimization that stabilizes trajectories. Across standard SLAM benchmarks, VGGT-SLAM++ achieves state-of-the-art accuracy, substantially reducing short-term drift, accelerating graph convergence, and maintaining global consistency with compact DEM tiles and sublinear retrieval.
>
---
#### [new 005] RePL: Pseudo-label Refinement for Semi-supervised LiDAR Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于LiDAR语义分割任务，针对半监督学习中伪标签噪声导致的误差传播和确认偏差问题，提出RePL框架通过掩码重建提升伪标签质量。**

- **链接: [https://arxiv.org/pdf/2604.06825](https://arxiv.org/pdf/2604.06825)**

> **作者:** Donghyeon Kwon; Taegyu Park; Suha Kwak
>
> **摘要:** Semi-supervised learning for LiDAR semantic segmentation often suffers from error propagation and confirmation bias caused by noisy pseudo-labels. To tackle this chronic issue, we introduce RePL, a novel framework that enhances pseudo-label quality by identifying and correcting potential errors in pseudo-labels through masked reconstruction, along with a dedicated training strategy. We also provide a theoretical analysis demonstrating the condition under which the pseudo-label refinement is beneficial, and empirically confirm that the condition is mild and clearly met by RePL. Extensive evaluations on the nuScenes-lidarseg and SemanticKITTI datasets show that RePL improves pseudo-label quality a lot and, as a result, achieves the state of the art in LiDAR semantic segmentation.
>
---
#### [new 006] Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training
- **分类: cs.CV**

- **简介: 该论文提出Mem3R，解决长序列3D重建中的时间一致性问题。通过混合记忆设计，提升性能并减少模型规模，适用于机器人和增强现实。**

- **链接: [https://arxiv.org/pdf/2604.07279](https://arxiv.org/pdf/2604.07279)**

> **作者:** Changkun Liu; Jiezhi Yang; Zeman Li; Yuan Deng; Jiancong Guo; Luca Ballan
>
> **备注:** Project page: this https URL
>
> **摘要:** Streaming 3D perception is well suited to robotics and augmented reality, where long visual streams must be processed efficiently and consistently. Recent recurrent models offer a promising solution by maintaining fixed-size states and enabling linear-time inference, but they often suffer from drift accumulation and temporal forgetting over long sequences due to the limited capacity of compressed latent memories. We propose Mem3R, a streaming 3D reconstruction model with a hybrid memory design that decouples camera tracking from geometric mapping to improve temporal consistency over long sequences. For camera tracking, Mem3R employs an implicit fast-weight memory implemented as a lightweight Multi-Layer Perceptron updated via Test-Time Training. For geometric mapping, Mem3R maintains an explicit token-based fixed-size state. Compared with CUT3R, this design not only significantly improves long-sequence performance but also reduces the model size from 793M to 644M parameters. Mem3R supports existing improved plug-and-play state update strategies developed for CUT3R. Specifically, integrating it with TTT3R decreases Absolute Trajectory Error by up to 39% over the base implementation on 500 to 1000 frame sequences. The resulting improvements also extend to other downstream tasks, including video depth estimation and 3D reconstruction, while preserving constant GPU memory usage and comparable inference throughput. Project page: this https URL
>
---
#### [new 007] RASR: Retrieval-Augmented Semantic Reasoning for Fake News Video Detection
- **分类: cs.CV**

- **简介: 该论文属于虚假新闻视频检测任务，旨在解决现有方法缺乏跨实例语义关联和领域知识引导的问题。提出RASR框架，结合语义检索与多模态推理，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.06687](https://arxiv.org/pdf/2604.06687)**

> **作者:** Hui Li; Peien Ding; Jun Li; Guoqi Ma; Zhanyu Liu; Ge Xu; Junfeng Yao; Jinsong Su
>
> **备注:** 10 pages,5 figures
>
> **摘要:** Multimodal fake news video detection is a crucial research direction for maintaining the credibility of online information. Existing studies primarily verify content authenticity by constructing multimodal feature fusion representations or utilizing pre-trained language models to analyze video-text consistency. However, these methods still face the following limitations: (1) lacking cross-instance global semantic correlations, making it difficult to effectively utilize historical associative evidence to verify the current video; (2) semantic discrepancies across domains hinder the transfer of general knowledge, lacking the guidance of domain-specific expert knowledge. To this end, we propose a novel Retrieval-Augmented Semantic Reasoning (RASR) framework. First, a Cross-instance Semantic Parser and Retriever (CSPR) deconstructs the video into high-level semantic primitives and retrieves relevant associative evidence from a dynamic memory bank. Subsequently, a Domain-Guided Multimodal Reasoning (DGMP) module incorporates domain priors to drive an expert multimodal large language model in generating domain-aware, in-depth analysis reports. Finally, a Multi-View Feature Decoupling and Fusion (MVDFF) module integrates multi-dimensional features through an adaptive gating mechanism to achieve robust authenticity determination. Extensive experiments on the FakeSV and FakeTT datasets demonstrate that RASR significantly outperforms state-of-the-art baselines, achieves superior cross-domain generalization, and improves the overall detection accuracy by up to 0.93%.
>
---
#### [new 008] Variational Feature Compression for Model-Specific Representations
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于隐私保护任务，旨在防止输入数据被未经授权的模型滥用。通过构建特征压缩框架，抑制跨模型迁移，同时保持目标分类器的准确率。**

- **链接: [https://arxiv.org/pdf/2604.06644](https://arxiv.org/pdf/2604.06644)**

> **作者:** Zinan Guo; Zihan Wang; Chuan Yan; Liuhuo Wan; Ethan Ma; Guangdong Bai
>
> **摘要:** As deep learning inference is increasingly deployed in shared and cloud-based settings, a growing concern is input repurposing, in which data submitted for one task is reused by unauthorized models for another. Existing privacy defenses largely focus on restricting data access, but provide limited control over what downstream uses a released representation can still support. We propose a feature extraction framework that suppresses cross-model transfer while preserving accuracy for a designated classifier. The framework employs a variational latent bottleneck, trained with a task-driven cross-entropy objective and KL regularization, but without any pixel-level reconstruction loss, to encode inputs into a compact latent space. A dynamic binary mask, computed from per-dimension KL divergence and gradient-based saliency with respect to the frozen target model, suppresses latent dimensions that are uninformative for the intended task. Because saliency computation requires gradient access, the encoder is trained in a white-box setting, whereas inference requires only a forward pass through the frozen target model. On CIFAR-100, the processed representations retain strong utility for the designated classifier while reducing the accuracy of all unintended classifiers to below 2%, yielding a suppression ratio exceeding 45 times relative to unintended models. Preliminary experiments on CIFAR-10, Tiny ImageNet, and Pascal VOC provide exploratory evidence that the approach extends across task settings, although further evaluation is needed to assess robustness against adaptive adversaries.
>
---
#### [new 009] LiftFormer: Lifting and Frame Theory Based Monocular Depth Estimation Using Depth and Edge Oriented Subspace Representation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于单目深度估计任务，旨在解决从单张图像中准确预测深度图的问题。提出LiftFormer模型，通过构建深度和边缘感知的子空间提升深度预测效果。**

- **链接: [https://arxiv.org/pdf/2604.06576](https://arxiv.org/pdf/2604.06576)**

> **作者:** Shuai Li; Huibin Bai; Yanbo Gao; Chong Lv; Hui Yuan; Chuankun Li; Wei Hua; Tian Xie
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** Monocular depth estimation (MDE) has attracted increasing interest in the past few years, owing to its important role in 3D vision. MDE is the estimation of a depth map from a monocular image/video to represent the 3D structure of a scene, which is a highly ill-posed problem. To solve this problem, in this paper, we propose a LiftFormer based on lifting theory topology, for constructing an intermediate subspace that bridges the image color features and depth values, and a subspace that enhances the depth prediction around edges. MDE is formulated by transforming the depth value prediction problem into depth-oriented geometric representation (DGR) subspace feature representation, thus bridging the learning from color values to geometric depth values. A DGR subspace is constructed based on frame theory by using linearly dependent vectors in accordance with depth bins to provide a redundant and robust representation. The image spatial features are transformed into the DGR subspace, where these features correspond directly to the depth values. Moreover, considering that edges usually present sharp changes in a depth map and tend to be erroneously predicted, an edge-aware representation (ER) subspace is constructed, where depth features are transformed and further used to enhance the local features around edges. The experimental results demonstrate that our LiftFormer achieves state-of-the-art performance on widely used datasets, and an ablation study validates the effectiveness of both proposed lifting modules in our LiftFormer.
>
---
#### [new 010] DesigNet: Learning to Draw Vector Graphics as Designers Do
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出DesigNet，解决SVG生成中的连续性和对齐问题。通过引入可微模块，提升生成结果的可编辑性，适用于专业设计流程。**

- **链接: [https://arxiv.org/pdf/2604.06494](https://arxiv.org/pdf/2604.06494)**

> **作者:** Tomas Guija-Valiente; Iago Suárez
>
> **摘要:** AI-driven content generation has made remarkable progress in recent years. However, neural networks and human designers operate in fundamentally different ways, making collaboration between them challenging. We address this gap for Scalable Vector Graphics (SVG) by equipping neural networks with tools commonly used by designers, such as axis alignment and explicit continuity control at command junctions. We introduce DesigNet, a hierarchical Transformer-VAE that operates directly on SVG sequences with a continuous command parameterization. Our main contributions are two differentiable modules: a continuity self-refinement module that predicts $C^0$, $G^1$, and $C^1$ continuity for each curve point and enforces it by modifying Bézier control points, and an alignment self-refinement module with snapping capabilities for horizontal or vertical lines. DesigNet produces editable outlines and achieves competitive results against state-of-the-art methods, with notably higher accuracy in continuity and alignment. These properties ensure the outputs are easier to refine and integrate into professional design workflows. Source Code: this https URL.
>
---
#### [new 011] IQ-LUT: interpolated and quantized LUT for efficient image super-resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，解决LUT存储瓶颈问题。通过插值、量化和残差学习，提出IQ-LUT，显著减少存储并提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.07000](https://arxiv.org/pdf/2604.07000)**

> **作者:** Yuxuan Zhang; Zhikai Dong; Xinning Chai; Xiangyun Zhou; Yi Xu; Zhengxue Cheng; Li Song
>
> **摘要:** Lookup table (LUT) methods demonstrate considerable potential in accelerating image super-resolution inference. However, pursuing higher image quality through larger receptive fields and bit-depth triggers exponential growth in the LUT's index space, creating a storage bottleneck that limits deployment on resource-constrained devices. We introduce IQ-LUT, which achieves a reduction in LUT size while simultaneously enhancing super-resolution quality. First, we integrate interpolation and quantization into the single-input, multiple-output ECNN, which dramatically reduces the index space and thereby the overall LUT size. Second, the integration of residual learning mitigates the dependence on LUT bit-depth, which facilitates training stability and prioritizes the reconstruction of fine-grained details for superior visual quality. Finally, guided by knowledge distillation, our non-uniform quantization process optimizes the quantization levels, thereby reducing storage while also compensating for quantization loss. Extensive benchmarking demonstrates our approach substantially reduces storage costs (by up to 50x compared to ECNN) while achieving superior super-resolution quality.
>
---
#### [new 012] Multi-modal user interface control detection using cross-attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于UI控件检测任务，旨在解决视觉模糊和设计多样性带来的检测难题。通过融合视觉与文本信息，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.06934](https://arxiv.org/pdf/2604.06934)**

> **作者:** Milad Moradi; Ke Yan; David Colwell; Matthias Samwald; Rhona Asgari
>
> **摘要:** Detecting user interface (UI) controls from software screenshots is a critical task for automated testing, accessibility, and software analytics, yet it remains challenging due to visual ambiguities, design variability, and the lack of contextual cues in pixel-only approaches. In this paper, we introduce a novel multi-modal extension of YOLOv5 that integrates GPT-generated textual descriptions of UI images into the detection pipeline through cross-attention modules. By aligning visual features with semantic information derived from text embeddings, our model enables more robust and context-aware UI control detection. We evaluate the proposed framework on a large dataset of over 16,000 annotated UI screenshots spanning 23 control classes. Extensive experiments compare three fusion strategies, i.e. element-wise addition, weighted sum, and convolutional fusion, demonstrating consistent improvements over the baseline YOLOv5 model. Among these, convolutional fusion achieved the strongest performance, with significant gains in detecting semantically complex or visually ambiguous classes. These results establish that combining visual and textual modalities can substantially enhance UI element detection, particularly in edge cases where visual information alone is insufficient. Our findings open promising opportunities for more reliable and intelligent tools in software testing, accessibility support, and UI analytics, setting the stage for future research on efficient, robust, and generalizable multi-modal detection systems.
>
---
#### [new 013] Controllable Generative Video Compression
- **分类: cs.CV**

- **简介: 该论文属于视频压缩任务，旨在解决感知与保真度之间的矛盾。通过引入可控生成框架，结合关键帧和密集控制先验，提升视频压缩的细节还原能力和视觉质量。**

- **链接: [https://arxiv.org/pdf/2604.06655](https://arxiv.org/pdf/2604.06655)**

> **作者:** Ding Ding; Daowen Li; Ying Chen; Yixin Gao; Ruixiao Dong; Kai Li; Li Li
>
> **摘要:** Perceptual video compression adopts generative video modeling to improve perceptual realism but frequently sacrifices signal fidelity, diverging from the goal of video compression to faithfully reproduce visual signal. To alleviate the dilemma between perception and fidelity, in this paper we propose Controllable Generative Video Compression (CGVC) paradigm to faithfully generate details guided by multiple visual conditions. Under the paradigm, representative keyframes of the scene are coded and used to provide structural priors for non-keyframe generation. Dense per-frame control prior is additionally coded to better preserve finer structure and semantics of each non-keyframe. Guided by these priors, non-keyframes are reconstructed by controllable video generation model with temporal and content consistency. Furthermore, to accurately recover color information of the video, we develop a color-distance-guided keyframe selection algorithm to adaptively choose keyframes. Experimental results show CGVC outperforms previous perceptual video compression method in terms of both signal fidelity and perceptual quality.
>
---
#### [new 014] Grounded Forcing: Bridging Time-Independent Semantics and Proximal Dynamics in Autoregressive Video Synthesis
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决自回归视频合成中的语义遗忘、视觉漂移和可控性丢失问题。提出Grounded Forcing框架，通过三个机制提升长期一致性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.06939](https://arxiv.org/pdf/2604.06939)**

> **作者:** Jintao Chen; Chengyu Bai; Junjun hu; Xinda Xue; Mu Xu
>
> **摘要:** Autoregressive video synthesis offers a promising pathway for infinite-horizon generation but is fundamentally hindered by three intertwined challenges: semantic forgetting from context limitations, visual drift due to positional extrapolation, and controllability loss during interactive instruction switching. Current methods often tackle these issues in isolation, limiting long-term coherence. We introduce Grounded Forcing, a novel framework that bridges time-independent semantics and proximal dynamics through three interlocking mechanisms. First, to address semantic forgetting, we propose a Dual Memory KV Cache that decouples local temporal dynamics from global semantic anchors, ensuring long-term semantic coherence and identity stability. Second, to suppress visual drift, we design Dual-Reference RoPE Injection, which confines positional embeddings within the training manifold while rendering global semantics time-invariant. Third, to resolve controllability issues, we develop Asymmetric Proximity Recache, which facilitates smooth semantic inheritance during prompt transitions via proximity-weighted cache updates. These components operate synergistically to tether the generative process to stable semantic cores while accommodating flexible local dynamics. Extensive experiments demonstrate that Grounded Forcing significantly enhances long-range consistency and visual stability, establishing a robust foundation for interactive long-form video synthesis.
>
---
#### [new 015] From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决3D高保真表面重建问题。通过引入可学习法向量和适应性衰减，提升表面提取精度，生成完整封闭网格。**

- **链接: [https://arxiv.org/pdf/2604.07337](https://arxiv.org/pdf/2604.07337)**

> **作者:** Diego Gomez; Antoine Guédon; Nissim Maruani; Bingchen Gong; Maks Ovsjanikov
>
> **备注:** Our project page is available in this http URL
>
> **摘要:** 3D Gaussian Splatting (3DGS) has revolutionized fast novel view synthesis, yet its opacity-based formulation makes surface extraction fundamentally difficult. Unlike implicit methods built on Signed Distance Fields or occupancy, 3DGS lacks a global geometric field, forcing existing approaches to resort to heuristics such as TSDF fusion of blended depth maps. Inspired by the Objects as Volumes framework, we derive a principled occupancy field for Gaussian Splatting and show how it can be used to extract highly accurate watertight meshes of complex scenes. Our key contribution is to introduce a learnable oriented normal at each Gaussian element and to define an adapted attenuation formulation, which leads to closed-form expressions for both the normal and occupancy fields at arbitrary locations in space. We further introduce a novel consistency loss and a dedicated densification strategy to enforce Gaussians to wrap the entire surface by closing geometric holes, ensuring a complete shell of oriented primitives. We modify the differentiable rasterizer to output depth as an isosurface of our continuous model, and introduce Primal Adaptive Meshing for Region-of-Interest meshing at arbitrary resolution. We additionally expose fundamental biases in standard surface evaluation protocols and propose two more rigorous alternatives. Overall, our method Gaussian Wrapping sets a new state-of-the-art on DTU and Tanks and Temples, producing complete, watertight meshes at a fraction of the size of concurrent work-recovering thin structures such as the notoriously elusive bicycle spokes.
>
---
#### [new 016] Accuracy Improvement of Semi-Supervised Segmentation Using Supervised ClassMix and Sup-Unsup Feature Discriminator
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于语义分割任务，旨在提升半监督学习的准确性。针对伪标签不准确和数据质量差异问题，提出结合有标签图像的类标签与无标签图像进行混合，并增强模型对无标签图像预测的一致性。**

- **链接: [https://arxiv.org/pdf/2604.07122](https://arxiv.org/pdf/2604.07122)**

> **作者:** Takahiro Mano; Reiji Saito; Kazuhiro Hotta
>
> **摘要:** In semantic segmentation, the creation of pixel-level labels for training data incurs significant costs. To address this problem, semi-supervised learning, which utilizes a small number of labeled images alongside unlabeled images to enhance the performance, has gained attention. A conventional semi-supervised learning method, ClassMix, pastes class labels predicted from unlabeled images onto other images. However, since ClassMix performs operations using pseudo-labels obtained from unlabeled images, there is a risk of handling inaccurate labels. Additionally, there is a gap in data quality between labeled and unlabeled images, which can impact the feature maps. This study addresses these two issues. First, we propose a method where class labels from labeled images, along with the corresponding image regions, are pasted onto unlabeled images and their pseudo-labeled images. Second, we introduce a method that trains the model to make predictions on unlabeled images more similar to those on labeled images. Experiments on the Chase and COVID-19 datasets demonstrated an average improvement of 2.07% in mIoU compared to conventional semi-supervised learning methods.
>
---
#### [new 017] From Static to Interactive: Adapting Visual in-Context Learners for User-Driven Tasks
- **分类: cs.CV**

- **简介: 该论文属于视觉上下文学习任务，旨在解决静态模型无法有效响应用户交互的问题。通过改进DeLVM方法，实现用户驱动的交互式视觉任务。**

- **链接: [https://arxiv.org/pdf/2604.06748](https://arxiv.org/pdf/2604.06748)**

> **作者:** Carlos Schmidt; Simon Reiß
>
> **摘要:** Visual in-context learning models are designed to adapt to new tasks by leveraging a set of example input-output pairs, enabling rapid generalization without task-specific fine-tuning. However, these models operate in a fundamentally static paradigm: while they can adapt to new tasks, they lack any mechanism to incorporate user-provided guidance signals such as scribbles, clicks, or bounding boxes to steer or refine the prediction process. This limitation is particularly restrictive in real-world applications, where users want to actively guide model predictions, e.g., by highlighting the target object for segmentation, indicating a region which should be visually altered, or isolating a specific person in a complex scene to run targeted pose estimation. In this work, we propose a simple method to transform static visual in-context learners, particularly the DeLVM approach, into highly controllable, user-driven systems, i.e., Interactive DeLVM, enabling seamless interaction through natural visual cues such as scribbles, clicks, or drawing boxes. Specifically, by encoding interactions directly into the example input-output pairs, we keep the philosophy of visual in-context learning intact: enabling users to prompt models with unseen interactions without fine-tuning and empowering them to dynamically steer model predictions with personalized interactions. Our experiments demonstrate that SOTA visual in-context learning models fail to effectively leverage interaction cues, often ignoring user guidance entirely. In contrast, our method excels in controllable, user-guided scenarios, achieving improvements of $+7.95%$ IoU for interactive segmentation, $+2.46$ PSNR for directed super-resolution, and $-3.14%$ LPIPS for interactive object removal. With this, our work bridges the gap between rigid static task adaptation and fluid interactivity for user-centric visual in-context learning.
>
---
#### [new 018] CloudMamba: An Uncertainty-Guided Dual-Scale Mamba Network for Cloud Detection in Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文属于遥感图像云检测任务，旨在解决单阶段方法在薄云和边界细节上的不确定性问题。提出CloudMamba框架，结合双尺度Mamba网络与不确定性引导策略，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.06844](https://arxiv.org/pdf/2604.06844)**

> **作者:** Jiajun Yang; Keyan Chen; Zhengxia Zou; Zhenwei Shi
>
> **摘要:** Cloud detection in remote sensing imagery is a fundamental, critical, and highly challenging problem. Existing deep learning-based cloud detection methods generally formulate it as a single-stage pixel-wise binary segmentation task with one forward pass. However, such single-stage approaches exhibit ambiguity and uncertainty in thin-cloud regions and struggle to accurately handle fragmented clouds and boundary details. In this paper, we propose a novel deep learning framework termed CloudMamba. To address the ambiguity in thin-cloud regions, we introduce an uncertainty-guided two-stage cloud detection strategy. An embedded uncertainty estimation module is proposed to automatically quantify the confidence of thin-cloud segmentation, and a second-stage refinement segmentation is introduced to improve the accuracy in low-confidence hard regions. To better handle fragmented clouds and fine-grained boundary details, we design a dual-scale Mamba network based on a CNN-Mamba hybrid architecture. Compared with Transformer-based models with quadratic computational complexity, the proposed method maintains linear computational complexity while effectively capturing both large-scale structural characteristics and small-scale boundary details of clouds, enabling accurate delineation of overall cloud morphology and precise boundary segmentation. Extensive experiments conducted on the GF1_WHU and Levir_CS public datasets demonstrate that the proposed method outperforms existing approaches across multiple segmentation accuracy metrics, while offering high efficiency and process transparency. Our code is available at this https URL.
>
---
#### [new 019] TeaLeafVision: An Explainable and Robust Deep Learning Framework for Tea Leaf Disease Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类任务，旨在解决茶叶病害识别问题。通过改进CNN模型并引入可解释技术，提升分类准确性和可靠性，应用于实际农业检测。**

- **链接: [https://arxiv.org/pdf/2604.07182](https://arxiv.org/pdf/2604.07182)**

> **作者:** Rafi Ahamed; Sidratul Moon Nafsin; Md Abir Rahman; Tasnia Tarannum Roza; Munaia Jannat Easha; Abu Raihan
>
> **摘要:** As the worlds second most consumed beverage after water, tea is not just a cultural staple but a global economic force of profound scale and influence. More than a mere drink, it represents a quiet negotiation between nature, culture, and the human desire for a moment of reflection. So, the precise identification and detection of tea leaf disease is crucial. With this goal, we have evaluated several Convolutional Neural Networks (CNN) models, among them three shows noticeable performance including DenseNet201, MobileNetV2, InceptionV3 on the teaLeafBD dataset. teaLeafBD dataset contains seven classes, six disease classes and one healthy class, collected under various field conditions reflecting real world challenges. Among the CNN models, DenseNet201 has achieved the highest test accuracy of 99%. In order to enhance the model reliability and interpretability, we have implemented Gradient weighted Class Activation Mapping (Grad CAM), occlusion sensitivity analysis and adversarial training techniques to increase the noise resistance of the model. Finally, we have developed a prototype in order to leverage the models capabilities on real life agriculture. This paper illustrates the deep learning models capabilities to classify the disease in real life tea leaf disease detection and management.
>
---
#### [new 020] DISSECT: Diagnosing Where Vision Ends and Language Priors Begin in Scientific VLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型（VLM）诊断任务，旨在解决模型在视觉理解与推理间的整合瓶颈。通过构建基准测试DISSECT，分析模型在不同输入模式下的表现，揭示其语言先验依赖与视觉推理能力的差距。**

- **链接: [https://arxiv.org/pdf/2604.06250](https://arxiv.org/pdf/2604.06250)**

> **作者:** Dikshant Kukreja; Kshitij Sah; Karan Goyal; Mukesh Mohania; Vikram Goyal
>
> **摘要:** When asked to describe a molecular diagram, a Vision-Language Model correctly identifies ``a benzene ring with an -OH group.'' When asked to reason about the same image, it answers incorrectly. The model can see but it cannot think about what it sees. We term this the perception-integration gap: a failure where visual information is successfully extracted but lost during downstream reasoning, invisible to single-configuration benchmarks that conflate perception with integration under one accuracy number. To systematically expose such failures, we introduce DISSECT, a 12,000-question diagnostic benchmark spanning Chemistry (7,000) and Biology (5,000). Every question is evaluated under five input modes -- Vision+Text, Text-Only, Vision-Only, Human Oracle, and a novel Model Oracle in which the VLM first verbalizes the image and then reasons from its own description -- yielding diagnostic gaps that decompose performance into language-prior exploitation, visual extraction, perception fidelity, and integration effectiveness. Evaluating 18~VLMs, we find that: (1) Chemistry exhibits substantially lower language-prior exploitability than Biology, confirming molecular visual content as a harder test of genuine visual reasoning; (2) Open-source models consistently score higher when reasoning from their own verbalized descriptions than from raw images, exposing a systematic integration bottleneck; and (3) Closed-source models show no such gap, indicating that bridging perception and integration is the frontier separating open-source from closed-source multimodal capability. The Model Oracle protocol is both model and benchmark agnostic, applicable post-hoc to any VLM evaluation to diagnose integration failures.
>
---
#### [new 021] Predicting Alzheimer's disease progression using rs-fMRI and a history-aware graph neural network
- **分类: cs.CV**

- **简介: 该论文属于疾病进展预测任务，旨在预测阿尔茨海默病患者认知阶段的转换。通过结合rs-fMRI数据和历史感知图神经网络，提高预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.06469](https://arxiv.org/pdf/2604.06469)**

> **作者:** Mahdi Moghaddami; Mohammad-Reza Siadat; Austin Toma; Connor Laming; Huirong Fu
>
> **备注:** Proc. SPIE 13926, Medical Imaging 2026: Computer-Aided Diagnosis, 1392604
>
> **摘要:** Alzheimer's disease (AD) is a neurodegenerative disorder that affects more than seven million people in the United States alone. AD currently has no cure, but there are ways to potentially slow its progression if caught early enough. In this study, we propose a graph neural network (GNN)-based model for predicting whether a subject will transition to a more severe stage of cognitive impairment at their next clinical visit. We consider three stages of cognitive impairment in order of severity: cognitively normal (CN), mild cognitive impairment (MCI), and AD. We use functional connectivity graphs derived from resting-state functional magnetic resonance imaging (rs-fMRI) scans of 303 subjects, each with a different number of visits. Our GNN-based model incorporates a recurrent neural network (RNN) block, enabling it to process data from the subject's entire visit history. It can also work with irregular time gaps between visits by incorporating visit distance information into our input features. Our model demonstrates robust predictive performance, even with missing visits in the subjects' visit histories. It achieves an accuracy of 82.9%, with an especially impressive accuracy of 68.8% on CN to MCI conversions - a task that poses a substantial challenge in the field. Our results highlight the effectiveness of rs-fMRI in predicting the onset of MCI or AD and, in conjunction with other modalities, could offer a viable method for enabling timely interventions to slow the progression of cognitive impairment.
>
---
#### [new 022] PhyEdit: Towards Real-World Object Manipulation via Physically-Grounded Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决现实物体操作中的物理准确性问题。通过引入3D几何模拟，提出PhyEdit框架，提升对象缩放与定位的精确性。**

- **链接: [https://arxiv.org/pdf/2604.07230](https://arxiv.org/pdf/2604.07230)**

> **作者:** Ruihang Xu; Dewei Zhou; Xiaolong Shen; Fan Ma; Yi Yang
>
> **摘要:** Achieving physically accurate object manipulation in image editing is essential for its potential applications in interactive world models. However, existing visual generative models often fail at precise spatial manipulation, resulting in incorrect scaling and positioning of objects. This limitation primarily stems from the lack of explicit mechanisms to incorporate 3D geometry and perspective projection. To achieve accurate manipulation, we develop PhyEdit, an image editing framework that leverages explicit geometric simulation as contextual 3D-aware visual guidance. By combining this plug-and-play 3D prior with joint 2D--3D supervision, our method effectively improves physical accuracy and manipulation consistency. To support this method and evaluate performance, we present a real-world dataset, RealManip-10K, for 3D-aware object manipulation featuring paired images and depth annotations. We also propose ManipEval, a benchmark with multi-dimensional metrics to evaluate 3D spatial control and geometric consistency. Extensive experiments show that our approach outperforms existing methods, including strong closed-source models, in both 3D geometric accuracy and manipulation consistency.
>
---
#### [new 023] VAMAE: Vessel-Aware Masked Autoencoders for OCT Angiography
- **分类: cs.CV**

- **简介: 该论文提出VAMAE，用于OCTA图像的自监督预训练，解决血管结构稀疏和拓扑约束问题，通过关注血管区域和多目标重建提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.06583](https://arxiv.org/pdf/2604.06583)**

> **作者:** Ilerioluwakiiye Abolade; Prince Mireku; Kelechi Chibundu; Peace Ododo; Emmanuel Idoko; Promise Omoigui; Solomon Odelola
>
> **备注:** 8 pages, 5 figures. Accepted at ICPR 2026
>
> **摘要:** Optical coherence tomography angiography (OCTA) provides non-invasive visualization of retinal microvasculature, but learning robust representations remains challenging due to sparse vessel structures and strong topological constraints. Many existing self-supervised learning approaches, including masked autoencoders, are primarily designed for dense natural images and rely on uniform masking and pixel-level reconstruction, which may inadequately capture vascular geometry. We propose VAMAE, a vessel-aware masked autoencoding framework for self-supervised pretraining on OCTA images. The approach incorporates anatomically informed masking that emphasizes vessel-rich regions using vesselness and skeleton-based cues, encouraging the model to focus on vascular connectivity and branching patterns. In addition, the pretraining objective includes reconstructing multiple complementary targets, enabling the model to capture appearance, structural, and topological information. We evaluate the proposed pretraining strategy on the OCTA-500 benchmark for several vessel segmentation tasks under varying levels of supervision. The results indicate that vessel-aware masking and multi-target reconstruction provide consistent improvements over standard masked autoencoding baselines, particularly in limited-label settings, suggesting the potential of geometry-aware self-supervised learning for OCTA analysis.
>
---
#### [new 024] MoRight: Motion Control Done Right
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出MoRight，解决视频生成中的运动控制问题，实现对象与相机运动的解耦及因果关系建模，提升视频生成质量与交互性。**

- **链接: [https://arxiv.org/pdf/2604.07348](https://arxiv.org/pdf/2604.07348)**

> **作者:** Shaowei Liu; Xuanchi Ren; Tianchang Shen; Huan Ling; Saurabh Gupta; Shenlong Wang; Sanja Fidler; Jun Gao
>
> **备注:** Project Page: this https URL
>
> **摘要:** Generating motion-controlled videos--where user-specified actions drive physically plausible scene dynamics under freely chosen viewpoints--demands two capabilities: (1) disentangled motion control, allowing users to separately control the object motion and adjust camera viewpoint; and (2) motion causality, ensuring that user-driven actions trigger coherent reactions from other objects rather than merely displacing pixels. Existing methods fall short on both fronts: they entangle camera and object motion into a single tracking signal and treat motion as kinematic displacement without modeling causal relationships between object motion. We introduce MoRight, a unified framework that addresses both limitations through disentangled motion modeling. Object motion is specified in a canonical static-view and transferred to an arbitrary target camera viewpoint via temporal cross-view attention, enabling disentangled camera and object control. We further decompose motion into active (user-driven) and passive (consequence) components, training the model to learn motion causality from data. At inference, users can either supply active motion and MoRight predicts consequences (forward reasoning), or specify desired passive outcomes and MoRight recovers plausible driving actions (inverse reasoning), all while freely adjusting the camera viewpoint. Experiments on three benchmarks demonstrate state-of-the-art performance in generation quality, motion controllability, and interaction awareness.
>
---
#### [new 025] DietDelta: A Vision-Language Approach for Dietary Assessment via Before-and-After Images
- **分类: cs.CV; cs.AI; cs.MM; eess.IV**

- **简介: 该论文属于饮食评估任务，旨在解决单张图像无法准确判断实际摄入量的问题。通过对比进食前后图像，利用视觉-语言方法进行食物级营养分析。**

- **链接: [https://arxiv.org/pdf/2604.06352](https://arxiv.org/pdf/2604.06352)**

> **作者:** Gautham Vinod; Siddeshwar Raghavan; Bruce Coburn; Fengqing Zhu
>
> **摘要:** Accurate dietary assessment is critical for precision nutrition, yet most image-based methods rely on a single pre-consumption image and provide only coarse, meal-level estimates. These approaches cannot determine what was actually consumed and often require restrictive inputs such as depth sensing, multi-view imagery, or explicit segmentation. In this paper, we propose a simple vision-language framework for food-item-level nutritional analysis using paired before-and-after eating images. Instead of relying on rigid segmentation masks, our method leverages natural language prompts to localize specific food items and estimate their weight directly from a single RGB image. We further estimate food consumption by predicting weight differences between paired images using a two-stage training strategy. We evaluate our method on three publicly available datasets and demonstrate consistent improvements over existing approaches, establishing a strong baseline for before-and-after dietary image analysis.
>
---
#### [new 026] Vision-Language Model-Guided Deep Unrolling Enables Personalized, Fast MRI
- **分类: cs.CV**

- **简介: 该论文提出PASS框架，解决MRI成像速度慢与临床任务适应性差的问题。通过结合视觉语言模型与深度去卷积网络，实现个性化、快速且高质量的MRI重建。**

- **链接: [https://arxiv.org/pdf/2604.06849](https://arxiv.org/pdf/2604.06849)**

> **作者:** Fangmao Ju; Yuzhu He; Zhiwen Xue; Chunfeng Lian; Jianhua Ma
>
> **摘要:** Magnetic Resonance Imaging (MRI) is a cornerstone in medicine and healthcare but suffers from long acquisition times. Traditional accelerated MRI methods optimize for generic image quality, lacking adaptability for specific clinical tasks. To address this, we introduce PASS (Personalized, Anomaly-aware Sampling and reconStruction), an intelligent MRI framework that leverages a Vision-Language Model (VLM) to guide a deep unrolling network for task-oriented, fast imaging. PASS dynamically personalizes the imaging pipeline through three core contributions: (1) a deep unrolled reconstruction network derived from a physics-based MRI model; (2) a sampling module that generates patient-specific $k$-space trajectories; and (3) an anomaly-aware prior, extracted from a pretrained VLM, which steers both sampling and reconstruction toward clinically relevant regions. By integrating the high-level clinical reasoning of a VLM with an interpretable, physics-aware network, PASS achieves superior image quality across diverse anatomies, contrasts, anomalies, and acceleration factors. This enhancement directly translates to improvements in downstream diagnostic tasks, including fine-grained anomaly detection, localization, and diagnosis.
>
---
#### [new 027] Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于病理图像分类任务，解决MIL框架中实例路由不均衡问题，提出ROAM方法通过最优传输和图正则化实现专家平衡利用。**

- **链接: [https://arxiv.org/pdf/2604.07298](https://arxiv.org/pdf/2604.07298)**

> **作者:** Xin Tian; Jiuliu Lu; Ephraim Tsalik; Bart Wanders; Colleen Knoth; Julian Knight
>
> **备注:** 10 pages, 2 figures, 2 tables
>
> **摘要:** Multiple Instance Learning (MIL) is the dominant framework for gigapixel whole-slide image (WSI) classification in computational pathology. However, current MIL aggregators route all instances through a shared pathway, constraining their capacity to specialise across the pathological heterogeneity inherent in each slide. Mixture-of-Experts (MoE) methods offer a natural remedy by partitioning instances across specialised expert subnetworks; yet unconstrained softmax routing may yield highly imbalanced utilisation, where one or a few experts absorb most routing mass, collapsing the mixture back to a near-single-pathway solution. To address these limitations, we propose ROAM (Region-graph OptimAl-transport Mixture-of-experts), a spatially aware MoE-MIL aggregator that routes region tokens to expert poolers via capacity-constrained entropic optimal transport, promoting balanced expert utilisation by construction. ROAM operates on spatial region tokens, obtained by compressing dense patch bags into spatially binned units that align routing with local tissue neighbourhoods and introduces two key mechanisms: (i) region-to-expert assignment formulated as entropic optimal transport (Sinkhorn) with explicit per slide capacity marginals, enforcing balanced expert utilisation without auxiliary load-balancing losses; and (ii) graph-regularised Sinkhorn iterations that diffuse routing assignments over the spatial region graph, encouraging neighbouring regions to coherently route to the same experts. Evaluated on four WSI benchmarks with frozen foundation-model patch embeddings, ROAM achieves performance competitive against strong MIL and MoE baselines, and on NSCLC generalisation (TCGA-CPTAC) reaches external AUC 0.845 +- 0.019.
>
---
#### [new 028] Video-guided Machine Translation with Global Video Context
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频引导的机器翻译任务，旨在解决长视频中全局语义上下文捕捉不足的问题。通过构建相关视频片段的语义集合，并利用注意力机制提升翻译效果。**

- **链接: [https://arxiv.org/pdf/2604.06789](https://arxiv.org/pdf/2604.06789)**

> **作者:** Jian Chen; JinZe Lv; Zi Long; XiangHua Fu
>
> **摘要:** Video-guided Multimodal Translation (VMT) has advanced significantly in recent years. However, most existing methods rely on locally aligned video segments paired one-to-one with subtitles, limiting their ability to capture global narrative context across multiple segments in long videos. To overcome this limitation, we propose a globally video-guided multimodal translation framework that leverages a pretrained semantic encoder and vector database-based subtitle retrieval to construct a context set of video segments closely related to the target subtitle semantics. An attention mechanism is employed to focus on highly relevant visual content, while preserving the remaining video features to retain broader contextual information. Furthermore, we design a region-aware cross-modal attention mechanism to enhance semantic alignment during translation. Experiments on a large-scale documentary translation dataset demonstrate that our method significantly outperforms baseline models, highlighting its effectiveness in long-video scenarios.
>
---
#### [new 029] Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于数据修剪任务，旨在解决标签噪声下动态修剪效果下降的问题。提出AlignPrune模块，通过损失轨迹对齐提高噪声样本识别准确性，提升修剪效果。**

- **链接: [https://arxiv.org/pdf/2604.07306](https://arxiv.org/pdf/2604.07306)**

> **作者:** Huaiyuan Qin; Muli Yang; Gabriel James Goenawan; Kai Wang; Zheng Wang; Peng Hu; Xi Peng; Hongyuan Zhu
>
> **备注:** Published in CVPR 2026 Findings
>
> **摘要:** Existing dynamic data pruning methods often fail under noisy-label settings, as they typically rely on per-sample loss as the ranking criterion. This could mistakenly lead to preserving noisy samples due to their high loss values, resulting in significant performance drop. To address this, we propose AlignPrune, a noise-robust module designed to enhance the reliability of dynamic pruning under label noise. Specifically, AlignPrune introduces the Dynamic Alignment Score (DAS), which is a loss-trajectory-based criterion that enables more accurate identification of noisy samples, thereby improving pruning effectiveness. As a simple yet effective plug-and-play module, AlignPrune can be seamlessly integrated into state-of-the-art dynamic pruning frameworks, consistently outperforming them without modifying either the model architecture or the training pipeline. Extensive experiments on five widely-used benchmarks across various noise types and pruning ratios demonstrate the effectiveness of AlignPrune, boosting accuracy by up to 6.3\% over state-of-the-art baselines. Our results offer a generalizable solution for pruning under noisy data, encouraging further exploration of learning in real-world scenarios. Code is available at: this https URL.
>
---
#### [new 030] Generative Phomosaic with Structure-Aligned and Personalized Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，解决传统 photomosaic 结构不一致和多样性不足的问题。通过扩散模型生成瓷砖图像，实现结构对齐与个性化风格。**

- **链接: [https://arxiv.org/pdf/2604.06989](https://arxiv.org/pdf/2604.06989)**

> **作者:** Jaeyoung Chung; Hyunjin Son; Kyoung Mu Lee
>
> **备注:** Project page: this https URL
>
> **摘要:** We present the first generative approach to photomosaic creation. Traditional photomosaic methods rely on a large number of tile images and color-based matching, which limits both diversity and structural consistency. Our generative photomosaic framework synthesizes tile images using diffusion-based generation conditioned on reference images. A low-frequency conditioned diffusion mechanism aligns global structure while preserving prompt-driven details. This generative formulation enables photomosaic composition that is both semantically expressive and structurally coherent, effectively overcoming the fundamental limitations of matching-based approaches. By leveraging few-shot personalized diffusion, our model is able to produce user-specific or stylistically consistent tiles without requiring an extensive collection of images.
>
---
#### [new 031] A Utility-preserving De-identification Pipeline for Cross-hospital Radiology Data Sharing
- **分类: cs.CV**

- **简介: 该论文属于医疗数据隐私保护任务，旨在解决跨医院放射数据共享中的隐私与实用性问题。提出UPDP管道，在去除隐私信息的同时保留病理特征，确保数据可用性与安全性。**

- **链接: [https://arxiv.org/pdf/2604.07128](https://arxiv.org/pdf/2604.07128)**

> **作者:** Chenhao Liu; Zelin Wen; Yan Tong; Junjie Zhu; Xinyu Tian; Yuchi Liu; Ashu Gupta; Syed M. S. Islam; Tom Gedeon; Yue Yao
>
> **摘要:** Large-scale radiology data are critical for developing robust medical AI systems. However, sharing such data across hospitals remains heavily constrained by privacy concerns. Existing de-identification research in radiology mainly focus on removing identifiable information to enable compliant data release. Yet whether de-identified radiology data can still preserve sufficient utility for large-scale vision-language model training and cross-hospital transfer remains underexplored. In this paper, we introduce a utility-preserving de-identification pipeline (UPDP) for cross-hospital radiology data sharing. Specifically, we compile a blacklist of privacy-sensitive terms and a whitelist of pathology-related terms. For radiology images, we use a generative filtering mechanism that synthesis a privacy-filtered and pathology-reserved counterparts of the original images. These synthetic image counterparts, together with ID-filtered reports, can then be securely shared across hospitals for downstream model development and evaluation. Experiments on public chest X-ray benchmarks demonstrate that our method effectively removes privacy-sensitive information while preserving diagnostically relevant pathology cues. Models trained on the de-identified data maintain competitive diagnostic accuracy compared with those trained on the original data, while exhibiting a marked decline in identity-related accuracy, confirming effective privacy protection. In the cross-hospital setting, we further show that de-identified data can be combined with local data to yield better performance.
>
---
#### [new 032] DOC-GS: Dual-Domain Observation and Calibration for Reliable Sparse-View Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决稀疏视角下3DGS的过拟合与伪影问题。提出DOC-GS框架，通过双域观测与校准提升高斯分布可靠性。**

- **链接: [https://arxiv.org/pdf/2604.06739](https://arxiv.org/pdf/2604.06739)**

> **作者:** Hantang Li; Qiang Zhu; Xiandong Meng; Debin Zhao; Xiaopeng Fan
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Sparse-view reconstruction with 3D Gaussian Splatting (3DGS) is fundamentally ill-posed due to insufficient geometric supervision, often leading to severe overfitting and the emergence of structural distortions and translucent haze-like artifacts. While existing approaches attempt to alleviate this issue via dropout-based regularization, they are largely heuristic and lack a unified understanding of artifact formation. In this paper, we revisit sparse-view 3DGS reconstruction from a new perspective and identify the core challenge as the unobservability of Gaussian primitive reliability. Unreliable Gaussians are insufficiently constrained during optimization and accumulate as haze-like degradations in rendered images. Motivated by this observation, we propose a unified Dual-domain Observation and Calibration (DOC-GS) framework that models and corrects Gaussian reliability through the synergy of optimization-domain inductive bias and observation-domain evidence. Specifically, in the optimization domain, we characterize Gaussian reliability by the degree to which each primitive is constrained during training, and instantiate this signal via a Continuous Depth-Guided Dropout (CDGD) strategy, where the dropout probability serves as an explicit proxy for primitive reliability. This imposes a smooth depth-aware inductive bias to suppress weakly constrained Gaussians and improve optimization stability. In the observation domain, we establish a connection between floater artifacts and atmospheric scattering, and leverage the Dark Channel Prior (DCP) as a structural consistency cue to identify and accumulate anomalous regions. Based on cross-view aggregated evidence, we further design a reliability-driven geometric pruning strategy to remove low-confidence Gaussians.
>
---
#### [new 033] Insights from Visual Cognition: Understanding Human Action Dynamics with Overall Glance and Refined Gaze Transformer
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频中运动和长距离依赖捕捉不足的问题。提出OG-ReG Transformer，结合全局与局部注意力机制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.06783](https://arxiv.org/pdf/2604.06783)**

> **作者:** Bohao Xing; Deng Li; Rong Gao; Xin Liu; Heikki Kälviäinen
>
> **摘要:** Recently, Transformer has made significant progress in various vision tasks. To balance computation and efficiency in video tasks, recent works heavily rely on factorized or window-based self-attention. However, these approaches split spatiotemporal correlations between regions of interest in videos, limiting the models' ability to capture motion and long-range dependencies. In this paper, we argue that, similar to the human visual system, the importance of temporal and spatial information varies across different time scales, and attention is allocated sparsely over time through glance and gaze behavior. Is equal consideration of time and space crucial for success in video tasks? Motivated by this understanding, we propose a dual-path network called the Overall Glance and Refined Gaze (OG-ReG) Transformer. The Glance path extracts coarse-grained overall spatiotemporal information, while the Gaze path supplements the Glance path by providing local details. Our model achieves state-of-the-art results on the Kinetics-400, Something-Something v2, and Diving-48, demonstrating its competitive performance. The code will be available at this https URL.
>
---
#### [new 034] URMF: Uncertainty-aware Robust Multimodal Fusion for Multimodal Sarcasm Detection
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于多模态讽刺检测任务，解决多模态数据可靠性不一导致的融合噪声问题。提出URMF框架，通过不确定性建模提升融合鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.06728](https://arxiv.org/pdf/2604.06728)**

> **作者:** Zhenyu Wang; Weichen Cheng; Weijia Li; Junjie Mou; Zongyou Zhao; Guoying Zhang
>
> **摘要:** Multimodal sarcasm detection (MSD) aims to identify sarcastic intent from semantic incongruity between text and image. Although recent methods have improved MSD through cross-modal interaction and incongruity reasoning, they often assume that all modalities are equally reliable. In real-world social media, however, textual content may be ambiguous and visual content may be weakly relevant or even irrelevant, causing deterministic fusion to introduce noisy evidence and weaken robust reasoning. To address this issue, we propose Uncertainty-aware Robust Multimodal Fusion (URMF), a unified framework that explicitly models modality reliability during interaction and fusion. URMF first employs multi-head cross-attention to inject visual evidence into textual representations, followed by multi-head self-attention in the fused semantic space to enhance incongruity-aware reasoning. It then performs unified unimodal aleatoric uncertainty modeling over text, image, and interaction-aware latent representations by parameterizing each modality as a learnable Gaussian posterior. The estimated uncertainty is further used to dynamically regulate modality contributions during fusion, suppressing unreliable modalities and yielding a more robust joint representation. In addition, we design a joint training objective integrating task supervision, modality prior regularization, cross-modal distribution alignment, and uncertainty-driven self-sampling contrastive learning. Experiments on public MSD benchmarks show that URMF consistently outperforms strong unimodal, multimodal, and MLLM-based baselines, demonstrating the effectiveness of uncertainty-aware fusion for improving both accuracy and robustness.
>
---
#### [new 035] TC-AE: Unlocking Token Capacity for Deep Compression Autoencoders
- **分类: cs.CV**

- **简介: 该论文提出TC-AE，解决视觉生成中高压缩下的重建与生成性能问题。通过优化token空间，提升压缩效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2604.07340](https://arxiv.org/pdf/2604.07340)**

> **作者:** Teng Li; Ziyuan Huang; Cong Chen; Yangfu Li; Yuanhuiyi Lyu; Dandan Zheng; Chunhua Shen; Jun Zhang
>
> **摘要:** We propose TC-AE, a ViT-based architecture for deep compression autoencoders. Existing methods commonly increase the channel number of latent representations to maintain reconstruction quality under high compression ratios. However, this strategy often leads to latent representation collapse, which degrades generative performance. Instead of relying on increasingly complex architectures or multi-stage training schemes, TC-AE addresses this challenge from the perspective of the token space, the key bridge between pixels and image latents, through two complementary innovations: Firstly, we study token number scaling by adjusting the patch size in ViT under a fixed latent budget, and identify aggressive token-to-latent compression as the key factor that limits effective scaling. To address this issue, we decompose token-to-latent compression into two stages, reducing structural information loss and enabling effective token number scaling for generation. Secondly, to further mitigate latent representation collapse, we enhance the semantic structure of image tokens via joint self-supervised training, leading to more generative-friendly latents. With these designs, TC-AE achieves substantially improved reconstruction and generative performance under deep compression. We hope our research will advance ViT-based tokenizer for visual generation.
>
---
#### [new 036] Distilling Photon-Counting CT into Routine Chest CT through Clinically Validated Degradation Modeling
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在将高质Photon-counting CT图像转化为常规CT图像，解决临床可用性不足问题。通过模拟退化过程，提升常规CT质量。**

- **链接: [https://arxiv.org/pdf/2604.07329](https://arxiv.org/pdf/2604.07329)**

> **作者:** Junqi Liu; Xinze Zhou; Wenxuan Li; Scott Ye; Arkadiusz Sitek; Xiaofeng Yang; Yucheng Tang; Daguang Xu; Kai Ding; Kang Wang; Yang Yang; Alan L. Yuille; Zongwei Zhou
>
> **摘要:** Photon-counting CT (PCCT) provides superior image quality with higher spatial resolution and lower noise compared to conventional energy-integrating CT (EICT), but its limited clinical availability restricts large-scale research and clinical deployment. To bridge this gap, we propose SUMI, a simulated degradation-to-enhancement method that learns to reverse realistic acquisition artifacts in low-quality EICT by leveraging high-quality PCCT as reference. Our central insight is to explicitly model realistic acquisition degradations, transforming PCCT into clinically plausible lower-quality counterparts and learning to invert this process. The simulated degradations were validated for clinical realism by board-certified radiologists, enabling faithful supervision without requiring paired acquisitions at scale. As outcomes of this technical contribution, we: (1) train a latent diffusion model on 1,046 PCCTs, using an autoencoder first pre-trained on both these PCCTs and 405,379 EICTs from 145 hospitals to extract general CT latent features that we release for reuse in other generative medical imaging tasks; (2) construct a large-scale dataset of over 17,316 publicly available EICTs enhanced to PCCT-like quality, with radiologist-validated voxel-wise annotations of airway trees, arteries, veins, lungs, and lobes; and (3) demonstrate substantial improvements: across external data, SUMI outperforms state-of-the-art image translation methods by 15% in SSIM and 20% in PSNR, improves radiologist-rated clinical utility in reader studies, and enhances downstream top-ranking lesion detection performance, increasing sensitivity by up to 15% and F1 score by up to 10%. Our results suggest that emerging imaging advances can be systematically distilled into routine EICT using limited high-quality scans as reference.
>
---
#### [new 037] Holistic Optimal Label Selection for Robust Prompt Learning under Partial Labels
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于弱监督下的提示学习任务，解决部分标签下的标签歧义问题。提出HopS方法，结合局部密度和全局最优传输策略，实现鲁棒的标签选择。**

- **链接: [https://arxiv.org/pdf/2604.06614](https://arxiv.org/pdf/2604.06614)**

> **作者:** Yaqi Zhao; Haoliang Sun; Yating Wang; Yongshun Gong; Yilong Yin
>
> **摘要:** Prompt learning has gained significant attention as a parameter-efficient approach for adapting large pre-trained vision-language models to downstream tasks. However, when only partial labels are available, its performance is often limited by label ambiguity and insufficient supervisory information. To address this issue, we propose Holistic Optimal Label Selection (HopS), leveraging the generalization ability of pre-trained feature encoders through two complementary strategies. First, we design a local density-based filter that selects the top frequent labels from the nearest neighbors' candidate sets and uses the softmax scores to identify the most plausible label, capturing structural regularities in the feature space. Second, we introduce a global selection objective based on optimal transport that maps the uniform sampling distribution to the candidate label distributions across a batch. By minimizing the expected transport cost, it can determine the most likely label assignments. These two strategies work together to provide robust label selection from both local and global perspectives. Extensive experiments on eight benchmark datasets show that HopS consistently improves performance under partial supervision and outperforms all baselines. Those results highlight the merit of holistic label selection and offer a practical solution for prompt learning in weakly supervised settings.
>
---
#### [new 038] SurFITR: A Dataset for Surveillance Image Forgery Detection and Localisation
- **分类: cs.CV; cs.AI; cs.MM; eess.IV**

- **简介: 该论文提出SurFITR数据集，用于监控图像篡改检测与定位。针对现有模型在监控场景下表现不佳的问题，通过多模态生成方法构建高质量篡改图像，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.07101](https://arxiv.org/pdf/2604.07101)**

> **作者:** Qizhou Wang; Guansong Pang; Christopher Leckie
>
> **摘要:** We present the Surveillance Forgery Image Test Range (SurFITR), a dataset for surveillance-style image forgery detection and localisation, in response to recent advances in open-access image generation models that raise concerns about falsifying visual evidence. Existing forgery models, trained on datasets with full-image synthesis or large manipulated regions in object-centric images, struggle to generalise to surveillance scenarios. This is because tampering in surveillance imagery is typically localised and subtle, occurring in scenes with varied viewpoints, small or occluded subjects, and lower visual quality. To address this gap, SurFITR provides a large collection of forensically valuable imagery generated via a multimodal LLM-powered pipeline, enabling semantically aware, fine-grained editing across diverse surveillance scenes. It contains over 137k tampered images with varying resolutions and edit types, generated using multiple image editing models. Extensive experiments show that existing detectors degrade significantly on SurFITR, while training on SurFITR yields substantial improvements in both in-domain and cross-domain performance. SurFITR is publicly available on GitHub.
>
---
#### [new 039] GenLCA: 3D Diffusion for Full-Body Avatars from In-the-Wild Videos
- **分类: cs.CV**

- **简介: 论文提出GenLCA，用于从视频生成高保真全身虚拟人。解决从2D数据生成3D Avatar的问题，通过扩散模型和可见性感知训练策略实现高质量生成与编辑。**

- **链接: [https://arxiv.org/pdf/2604.07273](https://arxiv.org/pdf/2604.07273)**

> **作者:** Yiqian Wu; Rawal Khirodkar; Egor Zakharov; Timur Bagautdinov; Lei Xiao; Zhaoen Su; Shunsuke Saito; Xiaogang Jin; Junxuan Li
>
> **摘要:** We present GenLCA, a diffusion-based generative model for generating and editing photorealistic full-body avatars from text and image inputs. The generated avatars are faithful to the inputs, while supporting high-fidelity facial and full-body animations. The core idea is a novel paradigm that enables training a full-body 3D diffusion model from partially observable 2D data, allowing the training dataset to scale to millions of real-world videos. This scalability contributes to the superior photorealism and generalizability of GenLCA. Specifically, we scale up the dataset by repurposing a pretrained feed-forward avatar reconstruction model as an animatable 3D tokenizer, which encodes unstructured video frames into structured 3D tokens. However, most real-world videos only provide partial observations of body parts, resulting in excessive blurring or transparency artifacts in the 3D tokens. To address this, we propose a novel visibility-aware diffusion training strategy that replaces invalid regions with learnable tokens and computes losses only over valid regions. We then train a flow-based diffusion model on the token dataset, inherently maintaining the photorealism and animatability provided by the pretrained avatar reconstruction model. Our approach effectively enables the use of large-scale real-world video data to train a diffusion model natively in 3D. We demonstrate the efficacy of our method through diverse and high-fidelity generation and editing results, outperforming existing solutions by a large margin. The project page is available at this https URL.
>
---
#### [new 040] Compression as an Adversarial Amplifier Through Decision Space Reduction
- **分类: cs.CV**

- **简介: 该论文研究图像压缩对对抗鲁棒性的影响，属于安全与机器学习领域。它揭示压缩会放大对抗攻击效果，因信息丢失导致分类边界收缩。**

- **链接: [https://arxiv.org/pdf/2604.06954](https://arxiv.org/pdf/2604.06954)**

> **作者:** Lewis Evans; Harkrishan Jandu; Zihan Ye; Yang Lu; Shreyank N Gowda
>
> **摘要:** Image compression is a ubiquitous component of modern visual pipelines, routinely applied by social media platforms and resource-constrained systems prior to inference. Despite its prevalence, the impact of compression on adversarial robustness remains poorly understood. We study a previously unexplored adversarial setting in which attacks are applied directly in compressed representations, and show that compression can act as an adversarial amplifier for deep image classifiers. Under identical nominal perturbation budgets, compression-aware attacks are substantially more effective than their pixel-space counterparts. We attribute this effect to decision space reduction, whereby compression induces a non-invertible, information-losing transformation that contracts classification margins and increases sensitivity to perturbations. Extensive experiments across standard benchmarks and architectures support our analysis and reveal a critical vulnerability in compression-in-the-loop deployment settings. Code will be released.
>
---
#### [new 041] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，解决正常样本定义模糊的问题。提出新场景和评估指标，并引入RePaste方法提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.07097](https://arxiv.org/pdf/2604.07097)**

> **作者:** Reiji Saito; Satoshi Kamiya; Kazuhiro Hotta
>
> **备注:** Accepted by CVPR 2026 Workshop
>
> **摘要:** In conventional anomaly detection, training data consist of only normal samples. However, in real-world scenarios, the definition of a normal sample is often ambiguous. For example, there are cases where a sample has small scratches or stains but is still acceptable for practical usage. On the other hand, higher precision is required when manufacturing equipment is upgraded. In such cases, normal samples may include small scratches, tiny dust particles, or a foreign object that we would prefer to classify as an anomaly. Such cases frequently occur in industrial settings, yet they have not been discussed until now. Thus, we propose novel scenarios and an evaluation metric to accommodate specification changes in real-world applications. Furthermore, to address the ambiguity of normal samples, we propose the RePaste, which enhances learning by re-pasting regions with high anomaly scores from the previous step into the input for the next step. On our scenarios using the MVTec AD benchmark, RePaste achieved the state-of-the-art performance with respect to the proposed evaluation metric, while maintaining high AUROC and PRO scores. Code: this https URL
>
---
#### [new 042] Fast Spatial Memory with Elastic Test-Time Training
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文提出一种名为Fast Spatial Memory（FSM）的模型，用于解决长序列3D/4D重建中的灾难性遗忘和过拟合问题，通过弹性测试时训练实现高效多块适应。**

- **链接: [https://arxiv.org/pdf/2604.07350](https://arxiv.org/pdf/2604.07350)**

> **作者:** Ziqiao Ma; Xueyang Yu; Haoyu Zhen; Yuncong Yang; Joyce Chai; Chuang Gan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Large Chunk Test-Time Training (LaCT) has shown strong performance on long-context 3D reconstruction, but its fully plastic inference-time updates remain vulnerable to catastrophic forgetting and overfitting. As a result, LaCT is typically instantiated with a single large chunk spanning the full input sequence, falling short of the broader goal of handling arbitrarily long sequences in a single pass. We propose Elastic Test-Time Training inspired by elastic weight consolidation, that stabilizes LaCT fast-weight updates with a Fisher-weighted elastic prior around a maintained anchor state. The anchor evolves as an exponential moving average of past fast weights to balance stability and plasticity. Based on this updated architecture, we introduce Fast Spatial Memory (FSM), an efficient and scalable model for 4D reconstruction that learns spatiotemporal representations from long observation sequences and renders novel view-time combinations. We pre-trained FSM on large-scale curated 3D/4D data to capture the dynamics and semantics of complex spatial environments. Extensive experiments show that FSM supports fast adaptation over long sequences and delivers high-quality 3D/4D reconstruction with smaller chunks and mitigating the camera-interpolation shortcut. Overall, we hope to advance LaCT beyond the bounded single-chunk setting toward robust multi-chunk adaptation, a necessary step for generalization to genuinely longer sequences, while substantially alleviating the activation-memory bottleneck.
>
---
#### [new 043] Synthetic Dataset Generation for Partially Observed Indoor Objects
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决真实数据获取成本高的问题。通过虚拟扫描生成合成数据集，包含部分点云和完整几何信息，用于训练和评估相关算法。**

- **链接: [https://arxiv.org/pdf/2604.07010](https://arxiv.org/pdf/2604.07010)**

> **作者:** Jelle Vermandere; Maarten Bassier; Maarten Vergauwen
>
> **摘要:** Learning-based methods for 3D scene reconstruction and object completion require large datasets containing partial scans paired with complete ground-truth geometry. However, acquiring such datasets using real-world scanning systems is costly and time-consuming, particularly when accurate ground truth for occluded regions is required. In this work, we present a virtual scanning framework implemented in Unity for generating realistic synthetic 3D scan datasets. The proposed system simulates the behaviour of real-world scanners using configurable parameters such as scan resolution, measurement range, and distance-dependent noise. Instead of directly sampling mesh surfaces, the framework performs ray-based scanning from virtual viewpoints, enabling realistic modelling of sensor visibility and occlusion effects. In addition, panoramic images captured at the scanner location are used to assign colours to the resulting point clouds. To support scalable dataset creation, the scanner is integrated with a procedural indoor scene generation pipeline that automatically produces diverse room layouts and furniture arrangements. Using this system, we introduce the \textit{V-Scan} dataset, which contains synthetic indoor scans together with object-level partial point clouds, voxel-based occlusion grids, and complete ground-truth geometry. The resulting dataset provides valuable supervision for training and evaluating learning-based methods for scene reconstruction and object completion.
>
---
#### [new 044] CSA-Graphs: A Privacy-Preserving Structural Dataset for Child Sexual Abuse Research
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解决CSAI分类中的数据隐私问题。通过构建CSA-Graphs结构化数据集，以图形式保留上下文信息，同时去除显性内容，支持安全研究。**

- **链接: [https://arxiv.org/pdf/2604.07132](https://arxiv.org/pdf/2604.07132)**

> **作者:** Carlos Caetano; Camila Laranjeira; Clara Ernesto; Artur Barros; João Macedo; Leo S. F. Ribeiro; Jefersson A. dos Santos; Sandra Avila
>
> **备注:** Conference on Computer Vision and Pattern Recognition (CVPR 2026), in the Workshop on Computer Vision for Children (CV4CHL)
>
> **摘要:** Child Sexual Abuse Imagery (CSAI) classification is an important yet challenging problem for computer vision research due to the strict legal and ethical restrictions that prevent the public sharing of CSAI datasets. This limitation hinders reproducibility and slows progress in developing automated methods. In this work, we introduce CSA-Graphs, a privacy-preserving structural dataset. Instead of releasing the original images, we provide structural representations that remove explicit visual content while preserving contextual information. CSA-Graphs includes two complementary graph-based modalities: scene graphs describing object relationships and skeleton graphs encoding human pose. Experiments show that both representations retain useful information for classifying CSAI, and that combining them further improves performance. This dataset enables broader research on computer vision methods for child safety while respecting legal and ethical constraints.
>
---
#### [new 045] Appear2Meaning: A Cross-Cultural Benchmark for Structured Cultural Metadata Inference from Images
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于跨文化结构化文化元数据推理任务，旨在解决从图像中提取文化信息的问题。研究构建了基准数据集并评估了视觉语言模型的表现。**

- **链接: [https://arxiv.org/pdf/2604.07338](https://arxiv.org/pdf/2604.07338)**

> **作者:** Yuechen Jiang; Enze Zhang; Md Mohsinul Kabir; Qianqian Xie; Stavroula Golfomitsou; Konstantinos Arvanitis; Sophia Ananiadou
>
> **摘要:** Recent advances in vision-language models (VLMs) have improved image captioning for cultural heritage. However, inferring structured cultural metadata (e.g., creator, origin, period) from visual input remains underexplored. We introduce a multi-category, cross-cultural benchmark for this task and evaluate VLMs using an LLM-as-Judge framework that measures semantic alignment with reference annotations. To assess cultural reasoning, we report exact-match, partial-match, and attribute-level accuracy across cultural regions. Results show that models capture fragmented signals and exhibit substantial performance variation across cultures and metadata types, leading to inconsistent and weakly grounded predictions. These findings highlight the limitations of current VLMs in structured cultural metadata inference beyond visual perception.
>
---
#### [new 046] Bridging MRI and PET physiology: Untangling complementarity through orthogonal representations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态影像分析任务，旨在解决模态间信息共享与特异性区分的问题。通过子空间分解框架，分离MRI与PET信号的互补成分，提升对生理特征的理解。**

- **链接: [https://arxiv.org/pdf/2604.07154](https://arxiv.org/pdf/2604.07154)**

> **作者:** Sonja Adomeit; Kartikay Tehlan; Lukas Förner; Katharina Weisser; Helen Scholtiseek; David Kaufmann; Julie Steinestel; Constantin Lapa; Thomas Kröncke; Thomas Wendler
>
> **备注:** The code is available at this https URL
>
> **摘要:** Multimodal imaging analysis often relies on joint latent representations, yet these approaches rarely define what information is shared versus modality-specific. Clarifying this distinction is clinically relevant, as it delineates the irreducible contribution of each modality and informs rational acquisition strategies. We propose a subspace decomposition framework that reframes multimodal fusion as a problem of orthogonal subspace separation rather than translation. We decompose Prostate-Specific Membrane Antigen (PSMA) PET uptake into an MRI-explainable physiological envelope and an orthogonal residual reflecting signal components not expressible within the MRI feature manifold. Using multiparametric MRI, we train an intensity-based, non-spatial implicit neural representation (INR) to map MRI feature vectors to PET uptake. We introduce a projection-based regularization using singular value decomposition to penalize residual components lying within the span of the MRI feature manifold. This enforces mathematical orthogonality between tissue-level physiological properties (structure, diffusion, perfusion) and intracellular PSMA expression. Tested on 13 prostate cancer patients, the model demonstrates that residual components spanned by MRI features are absorbed into the learned envelope, while the orthogonal residual is largest in tumour regions. This indicates that PSMA PET contains signal components not recoverable from MRI-derived physiological descriptors. The resulting decomposition provides a structured characterization of modality complementarity grounded in representation geometry rather than image translation.
>
---
#### [new 047] NTIRE 2026 Challenge on Bitstream-Corrupted Video Restoration: Methods and Results
- **分类: cs.CV**

- **简介: 该论文属于视频修复任务，旨在解决比特流损坏视频的恢复问题。通过构建基准数据集和评估协议，对比分析了多种修复方法，总结了技术趋势与挑战。**

- **链接: [https://arxiv.org/pdf/2604.06945](https://arxiv.org/pdf/2604.06945)**

> **作者:** Wenbin Zou; Tianyi Li; Kejun Wu; Huiping Zhuang; Zongwei Wu; Zhuyun Zhou; Radu Timofte; Kim-Hui Yap; Lap-Pui Chau; Yi Wang; Shiqi Zhou; Xiaodi Shi; Yuxiang Chen; Yilian Zhong; Shibo Yin; Yushun Fang; Xilei Zhu; Yahui Wang; Chen Lu; Zhitao Wang; Lifa Ha; Hengyu Man; Xiaopeng Fan; Priyansh Singh; Sidharth; Krrish Dev; Soham Kakkar; Vinit Jakhetiya; Ovais Iqbal Shah; Wei Zhou; Linfeng Li; Qi Xu; Zhenyang Liu; Kepeng Xu; Tong Qiao; Jiachen Tu; Guoyi Xu; Yaoxin Jiang; Jiajia Liu; Yaokun Shi
>
> **备注:** 15 pages, 8 figures, 1 table, CVPRW2026 NTIRE Challenge Report
>
> **摘要:** This paper reports on the NTIRE 2026 Challenge on Bitstream-Corrupted Video Restoration (BSCVR). The challenge aims to advance research on recovering visually coherent videos from corrupted bitstreams, whose decoding often produces severe spatial-temporal artifacts and content distortion. Built upon recent progress in bitstream-corrupted video recovery, the challenge provides a common benchmark for evaluating restoration methods under realistic corruption settings. We describe the dataset, evaluation protocol, and participating methods, and summarize the final results and main technical trends. The challenge highlights the difficulty of this emerging task and provides useful insights for future research on robust video restoration under practical bitstream corruption.
>
---
#### [new 048] No-reference based automatic parameter optimization for iterative reconstruction using a novel search space aware crow search algorithm
- **分类: cs.CV**

- **简介: 该论文属于图像重建任务，旨在解决迭代重建中参数调优困难的问题。提出一种无需参考的自动参数优化框架，采用改进的乌鸦搜索算法提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.06246](https://arxiv.org/pdf/2604.06246)**

> **作者:** Poorya MohammadiNasab; Ander Biguri; Philipp Steininger; Peter Keuschnigg; Lukas Lamminger; Agnieszka Lach; S M Ragib Shahriar Islam; Anna Breger; Clemens Karner; Carola-Bibiane Schönlieb; Wolfgang Birkfellner; Sepideh Hatamikia
>
> **摘要:** Iterative reconstruction technique's ability to reduce radiation exposure by using fewer projections has attracted significant attention. However, these methods typically require a precise tuning of several hyperparameters, which can have a major impact on reconstruction quality. Manually setting these parameters is time-consuming and increases the workload for human operators. In this paper, we introduce a novel fully automatic parameter optimization framework that can be applied to a wide range of Cone-beam computed tomography (CBCT) iterative reconstruction algorithms to determine optimal parameters without requiring a reference reconstruction. The proposed method incorporates a modified crow search algorithm (CSA) featuring a superior set-dependent local search mechanism, a search-space-aware global search strategy, and an objective-driven balance between local and global search. Additionally, to ensure an effective initial population, we propose a chaotic diagonal linear uniform initialization scheme that accelerates algorithm convergence. The performance of the proposed framework was evaluated on three imaging machines and four real datasets, as well as three different iterative reconstruction methods with the highest number of tunable parameters, representing the most challenging senario. The results indicate that the proposed method could outperform manual settings and CSA, with an 4.19% improvement in average fitness and 4.89% and 3.82% improvements on CHILL@UK and RPI_AXIS, respectively, which are two benchmark no-reference learning-based quality metrics. In addition, the qualitative results clearly show the superiority of the proposed method by maintaining fine details sharply. The overall performance of the proposed framework across different comparison scenarios demonstrates its effectiveness and robustness across all cases.
>
---
#### [new 049] Evidence-Based Actor-Verifier Reasoning for Echocardiographic Agents
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决 echocardiographic 数据智能分析中的可靠性与可解释性问题。提出 EchoTrust 框架，通过 Actor-Verifier 结构提升决策可信度。**

- **链接: [https://arxiv.org/pdf/2604.06347](https://arxiv.org/pdf/2604.06347)**

> **作者:** Peng Huang; Yiming Wang; Yineng Chen; Liangqiao Gui; Hui Guo; Bo Peng; Shu Hu; Xi Wu; Tsao Connie; Hongtu Zhu; Balakrishnan Prabhakaran; Xin Wang
>
> **备注:** cvprw 2026(AIMS)
>
> **摘要:** Echocardiography plays an important role in the screening and diagnosis of cardiovascular diseases. However, automated intelligent analysis of echocardiographic data remains challenging due to complex cardiac dynamics and strong view heterogeneity. In recent years, visual language models (VLM) have opened a new avenue for building ultrasound understanding systems for clinical decision support. Nevertheless, most existing methods formulate this task as a direct mapping from video and question to answer, making them vulnerable to template shortcuts and spurious explanations. To address these issues, we propose EchoTrust, an evidence-driven Actor-Verifier framework for trustworthy reasoning in echocardiography VLM-based agents. EchoTrust produces a structured intermediate representation that is subsequently analyzed by distinct roles, enabling more reliable and interpretable decision-making for high-stakes clinical applications.
>
---
#### [new 050] FlowExtract: Procedural Knowledge Extraction from Maintenance Flowcharts
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于流程图知识提取任务，旨在解决维护流程图中程序知识难以被现代系统利用的问题。工作包括构建FlowExtract系统，分离节点检测与连接重建，提升边提取效果。**

- **链接: [https://arxiv.org/pdf/2604.06770](https://arxiv.org/pdf/2604.06770)**

> **作者:** Guillermo Gil de Avalle; Laura Maruster; Eric Sloot; Christos Emmanouilidis
>
> **摘要:** Maintenance procedures in manufacturing facilities are often documented as flowcharts in static PDFs or scanned images. They encode procedural knowledge essential for asset lifecycle management, yet inaccessible to modern operator support systems. Vision-language models, the dominant paradigm for image understanding, struggle to reconstruct connection topology from such diagrams. We present FlowExtract, a pipeline for extracting directed graphs from ISO 5807-standardized flowcharts. The system separates element detection from connectivity reconstruction, using YOLOv8 and EasyOCR for standard domain-aligned node detection and text extraction, combined with a novel edge detection method that analyzes arrowhead orientations and traces connecting lines backward to source nodes. Evaluated on industrial troubleshooting guides, FlowExtract achieves very high node detection and substantially outperforms vision-language model baselines on edge extraction, offering organizations a practical path toward queryable procedural knowledge representations. The implementation is available athttps://github.com/guille-gil/FlowExtract.
>
---
#### [new 051] Auditing Demographic Bias in Facial Landmark Detection for Fair Human-Robot Interaction
- **分类: cs.CV**

- **简介: 该论文属于人脸关键点检测任务，旨在解决公平性问题。通过审计年龄、性别和种族偏差，发现性能差异主要由视觉因素引起，而非人口统计属性。**

- **链接: [https://arxiv.org/pdf/2604.06961](https://arxiv.org/pdf/2604.06961)**

> **作者:** Pablo Parte; Roberto Valle; José M. Buenaposada; Luis Baumela
>
> **摘要:** Fairness in human-robot interaction critically depends on the reliability of the perceptual models that enable robots to interpret human behavior. While demographic biases have been widely studied in high-level facial analysis tasks, their presence in facial landmark detection remains unexplored. In this paper, we conduct a systematic audit of demographic bias in this task, analyzing the age, gender and race biases. To this end we introduce a controlled statistical methodology to disentangle demographic effects from confounding visual factors. Evaluations of a standard representative model demonstrate that confounding visual factors, particularly head pose and image resolution, heavily outweigh the impact of demographic attributes. Notably, after accounting for these confounders, we show that performance disparities across gender and race vanish. However, we identify a statistically significant age-related effect, with higher biases observed for older individuals. This shows that fairness issues can emerge even in low-level vision components and can propagate through the HRI pipeline, disproportionately affecting vulnerable populations. We argue that auditing and correcting such biases is a necessary step toward trustworthy and equitable robot perception systems.
>
---
#### [new 052] Multiple Domain Generalization Using Category Information Independent of Domain Differences
- **分类: cs.CV**

- **简介: 该论文属于领域泛化任务，旨在解决模型在不同领域数据上性能下降的问题。通过分离类别信息与领域差异，结合量子向量缩小领域差距，提升分割准确性。**

- **链接: [https://arxiv.org/pdf/2604.07175](https://arxiv.org/pdf/2604.07175)**

> **作者:** Reiji Saito; Kazuhiro Hotta
>
> **摘要:** Domain generalization is a technique aimed at enabling models to maintain high accuracy when applied to new environments or datasets (unseen domains) that differ from the datasets used in training. Generally, the accuracy of models trained on a specific dataset (source domain) often decreases significantly when evaluated on different datasets (target domain). This issue arises due to differences in domains caused by varying environmental conditions such as imaging equipment and staining methods. Therefore, we undertook two initiatives to perform segmentation that does not depend on domain differences. We propose a method that separates category information independent of domain differences from the information specific to the source domain. By using information independent of domain differences, our method enables learning the segmentation targets (e.g., blood vessels and cell nuclei). Although we extract independent information of domain differences, this cannot completely bridge the domain gap between training and test data. Therefore, we absorb the domain gap using the quantum vectors in Stochastically Quantized Variational AutoEncoder (SQ-VAE). In experiments, we evaluated our method on datasets for vascular segmentation and cell nucleus segmentation. Our methods improved the accuracy compared to conventional methods.
>
---
#### [new 053] POS-ISP: Pipeline Optimization at the Sequence Level for Task-aware ISP
- **分类: cs.CV**

- **简介: 该论文属于图像信号处理（ISP）任务，解决模块序列与参数联合优化问题。提出POS-ISP框架，通过序列级强化学习一次性预测模块序列和参数，提升性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2604.06938](https://arxiv.org/pdf/2604.06938)**

> **作者:** Jiyun Won; Heemin Yang; Woohyeok Kim; Jungseul Ok; Sunghyun Cho
>
> **摘要:** Recent work has explored optimizing image signal processing (ISP) pipelines for various tasks by composing predefined modules and adapting them to task-specific objectives. However, jointly optimizing module sequences and parameters remains challenging. Existing approaches rely on neural architecture search (NAS) or step-wise reinforcement learning (RL), but NAS suffers from a training-inference mismatch, while step-wise RL leads to unstable training and high computational overhead due to stage-wise decision-making. We propose POS-ISP, a sequence-level RL framework that formulates modular ISP optimization as a global sequence prediction problem. Our method predicts the entire module sequence and its parameters in a single forward pass and optimizes the pipeline using a terminal task reward, eliminating the need for intermediate supervision and redundant executions. Experiments across multiple downstream tasks show that POS-ISP improves task performance while reducing computational cost, highlighting sequence-level optimization as a stable and efficient paradigm for task-aware ISP. The project page is available at this https URL
>
---
#### [new 054] Continual Visual Anomaly Detection on the Edge: Benchmark and Efficient Solutions
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉异常检测任务，解决边缘计算与持续学习下的高效检测问题。提出基准测试和轻量模型Tiny-Dinomaly，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.06435](https://arxiv.org/pdf/2604.06435)**

> **作者:** Manuel Barusco; Francesco Borsatti; David Petrovic; Davide Dalle Pezze; Gian Antonio Susto
>
> **摘要:** Visual Anomaly Detection (VAD) is a critical task for many applications including industrial inspection and healthcare. While VAD has been extensively studied, two key challenges remain largely unaddressed in conjunction: edge deployment, where computational resources are severely constrained, and continual learning, where models must adapt to evolving data distributions without forgetting previously acquired knowledge. Our benchmark provides guidance for the selection of the optimal backbone and VAD method under joint efficiency and adaptability constraints, characterizing the trade-offs between memory footprint, inference cost, and detection performance. Studying these challenges in isolation is insufficient, as methods designed for one setting make assumptions that break down when the other constraint is simultaneously imposed. In this work, we propose the first comprehensive benchmark for VAD on the edge in the continual learning scenario, evaluating seven VAD models across three lightweight backbone architectures. Furthermore, we propose Tiny-Dinomaly, a lightweight adaptation of the Dinomaly model built on the DINO foundation model that achieves 13x smaller memory footprint and 20x lower computational cost while improving Pixel F1 by 5 percentage points. Finally, we introduce targeted modifications to PatchCore and PaDiM to improve their efficiency in the continual learning setting.
>
---
#### [new 055] Evolution of Video Generative Foundations
- **分类: cs.CV**

- **简介: 该论文属于视频生成领域，旨在全面梳理技术演进，解决现有研究视角狭窄的问题。工作包括系统回顾技术发展，分析原理与优劣，并探讨多模态趋势。**

- **链接: [https://arxiv.org/pdf/2604.06339](https://arxiv.org/pdf/2604.06339)**

> **作者:** Teng Hu; Jiangning Zhang; Hongrui Huang; Ran Yi; Zihan Su; Jieyu Weng; Zhucun Xue; Lizhuang Ma; Ming-Hsuan Yang; Dacheng Tao
>
> **摘要:** The rapid advancement of Artificial Intelligence Generated Content (AIGC) has revolutionized video generation, enabling systems ranging from proprietary pioneers like OpenAI's Sora, Google's Veo3, and Bytedance's Seedance to powerful open-source contenders like Wan and HunyuanVideo to synthesize temporally coherent and semantically rich videos. These advancements pave the way for building "world models" that simulate real-world dynamics, with applications spanning entertainment, education, and virtual reality. However, existing reviews on video generation often focus on narrow technical fields, e.g., Generative Adversarial Networks (GAN) and diffusion models, or specific tasks (e. g., video editing), lacking a comprehensive perspective on the field's evolution, especially regarding Auto-Regressive (AR) models and integration of multimodal information. To address these gaps, this survey firstly provides a systematic review of the development of video generation technology, tracing its evolution from early GANs to dominant diffusion models, and further to emerging AR-based and multimodal techniques. We conduct an in-depth analysis of the foundational principles, key advancements, and comparative strengths/limitations. Then, we explore emerging trends in multimodal video generation, emphasizing the integration of diverse data types to enhance contextual awareness. Finally, by bridging historical developments and contemporary innovations, this survey offers insights to guide future research in video generation and its applications, including virtual/augmented reality, personalized education, autonomous driving simulations, digital entertainment, and advanced world models, in this rapidly evolving field. For more details, please refer to the project at this https URL.
>
---
#### [new 056] Improving Local Feature Matching by Entropy-inspired Scale Adaptability and Flow-endowed Local Consistency
- **分类: cs.CV**

- **简介: 该论文属于图像匹配任务，解决尺度差异和局部一致性问题。提出尺度感知模块和流优化方法，提升匹配精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.06713](https://arxiv.org/pdf/2604.06713)**

> **作者:** Ke Jin; Jiming Chen; Qi Ye
>
> **摘要:** Recent semi-dense image matching methods have achieved remarkable success, but two long-standing issues still impair their performance. At the coarse stage, the over-exclusion issue of their mutual nearest neighbor (MNN) matching layer makes them struggle to handle cases with scale difference between images. To this end, we comprehensively revisit the matching mechanism and make a key observation that the hint concealed in the score matrix can be exploited to indicate the scale ratio. Based on this, we propose a scale-aware matching module which is exceptionally effective but introduces negligible overhead. At the fine stage, we point out that existing methods neglect the local consistency of final matches, which undermines their robustness. To this end, rather than independently predicting the correspondence for each source pixel, we reformulate the fine stage as a cascaded flow refinement problem and introduce a novel gradient loss to encourage local consistency of the flow field. Extensive experiments demonstrate that our novel matching pipeline, with these proposed modifications, achieves robust and accurate matching performance on downstream tasks.
>
---
#### [new 057] Specializing Large Models for Oracle Bone Script Interpretation via Component-Grounded Multimodal Knowledge Augmentation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于古文字解读任务，旨在解决Oracle Bone Script的解释缺口问题。通过构建多模态知识增强框架，提升字符解析的准确性和细致度。**

- **链接: [https://arxiv.org/pdf/2604.06711](https://arxiv.org/pdf/2604.06711)**

> **作者:** Jianing Zhang; Runan Li; Honglin Pang; Ding Xia; Zhou Zhu; Qian Zhang; Chuntao Li; Xi Yang
>
> **摘要:** Deciphering ancient Chinese Oracle Bone Script (OBS) is a challenging task that offers insights into the beliefs, systems, and culture of the ancient era. Existing approaches treat decipherment as a closed-set image recognition problem, which fails to bridge the ``interpretation gap'': while individual characters are often unique and rare, they are composed of a limited set of recurring, pictographic components that carry transferable semantic meanings. To leverage this structural logic, we propose an agent-driven Vision-Language Model (VLM) framework that integrates a VLM for precise visual grounding with an LLM-based agent to automate a reasoning chain of component identification, graph-based knowledge retrieval, and relationship inference for linguistically accurate interpretation. To support this, we also introduce OB-Radix, an expert-annotated dataset providing structural and semantic data absent from prior corpora, comprising 1,022 character images (934 unique characters) and 1,853 fine-grained component images across 478 distinct components with verified explanations. By evaluating our system across three benchmarks of different tasks, we demonstrate that our framework yields more detailed and precise decipherments compared to baseline methods.
>
---
#### [new 058] WeatherRemover: All-in-one Adverse Weather Removal with Multi-scale Feature Map Compression
- **分类: cs.CV**

- **简介: 该论文属于图像增强任务，旨在解决多种恶劣天气导致的图像模糊、遮挡和亮度低的问题。提出WeatherRemover模型，结合多尺度特征压缩与轻量结构，实现高效且高质量的天气去除。**

- **链接: [https://arxiv.org/pdf/2604.06623](https://arxiv.org/pdf/2604.06623)**

> **作者:** Weikai Qu; Sijun Liang; Cheng Pan; Zikuan Yang; Guanchi Zhou; Xianjun Fu; Bo Liu; Changmiao Wang; Ahmed Elazab
>
> **备注:** Accepted by IEEE Transactions on Artificial Intelligence
>
> **摘要:** Photographs taken in adverse weather conditions often suffer from blurriness, occlusion, and low brightness due to interference from rain, snow, and fog. These issues can significantly hinder the performance of subsequent computer vision tasks, making the removal of weather effects a crucial step in image enhancement. Existing methods primarily target specific weather conditions, with only a few capable of handling multiple weather scenarios. However, mainstream approaches often overlook performance considerations, resulting in large parameter sizes, long inference times, and high memory costs. In this study, we introduce the WeatherRemover model, designed to enhance the restoration of images affected by various weather conditions while balancing performance. Our model adopts a UNet-like structure with a gating mechanism and a multi-scale pyramid vision Transformer. It employs channel-wise attention derived from convolutional neural networks to optimize feature extraction, while linear spatial reduction helps curtail the computational demands of attention. The gating mechanisms, strategically placed within the feed-forward and downsampling phases, refine the processing of information by selectively addressing redundancy and mitigating its influence on learning. This approach facilitates the adaptive selection of essential data, ensuring superior restoration and maximizing efficiency. Additionally, our lightweight model achieves an optimal balance between restoration quality, parameter efficiency, computational overhead, and memory usage, distinguishing it from other multi-weather models, thereby meeting practical application demands effectively. The source code is available at this https URL.
>
---
#### [new 059] Towards Robust Content Watermarking Against Removal and Forgery Attacks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于内容水印任务，旨在解决生成内容的版权保护问题。针对水印被移除和伪造的问题，提出一种新的水印方法ISTS，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.06662](https://arxiv.org/pdf/2604.06662)**

> **作者:** Yifan Zhu; Yihan Wang; Xiao-Shan Gao
>
> **备注:** 14 pages, 5 figures, CVPR 2026 Findings
>
> **摘要:** Generated contents have raised serious concerns about copyright protection, image provenance, and credit attribution. A potential solution for these problems is watermarking. Recently, content watermarking for text-to-image diffusion models has been studied extensively for its effective detection utility and robustness. However, these watermarking techniques are vulnerable to potential adversarial attacks, such as removal attacks and forgery attacks. In this paper, we build a novel watermarking paradigm called Instance-Specific watermarking with Two-Sided detection (ISTS) to resist removal and forgery attacks. Specifically, we introduce a strategy that dynamically controls the injection time and watermarking patterns based on the semantics of users' prompts. Furthermore, we propose a new two-sided detection approach to enhance robustness in watermark detection. Experiments have demonstrated the superiority of our watermarking against removal and forgery attacks.
>
---
#### [new 060] RefineAnything: Multimodal Region-Specific Refinement for Perfect Local Details
- **分类: cs.CV**

- **简介: 该论文提出RefineAnything，解决图像局部细节修复问题。针对用户指定区域，实现精准细节恢复且不改变背景。通过区域聚焦策略与边界损失提升效果。**

- **链接: [https://arxiv.org/pdf/2604.06870](https://arxiv.org/pdf/2604.06870)**

> **作者:** Dewei Zhou; You Li; Zongxin Yang; Yi Yang
>
> **备注:** 18 pages
>
> **摘要:** We introduce region-specific image refinement as a dedicated problem setting: given an input image and a user-specified region (e.g., a scribble mask or a bounding box), the goal is to restore fine-grained details while keeping all non-edited pixels strictly unchanged. Despite rapid progress in image generation, modern models still frequently suffer from local detail collapse (e.g., distorted text, logos, and thin structures). Existing instruction-driven editing models emphasize coarse-grained semantic edits and often either overlook subtle local defects or inadvertently change the background, especially when the region of interest occupies only a small portion of a fixed-resolution input. We present RefineAnything, a multimodal diffusion-based refinement model that supports both reference-based and reference-free refinement. Building on a counter-intuitive observation that crop-and-resize can substantially improve local reconstruction under a fixed VAE input resolution, we propose Focus-and-Refine, a region-focused refinement-and-paste-back strategy that improves refinement effectiveness and efficiency by reallocating the resolution budget to the target region, while a blended-mask paste-back guarantees strict background preservation. We further introduce a boundary-aware Boundary Consistency Loss to reduce seam artifacts and improve paste-back naturalness. To support this new setting, we construct Refine-30K (20K reference-based and 10K reference-free samples) and introduce RefineEval, a benchmark that evaluates both edited-region fidelity and background consistency. On RefineEval, RefineAnything achieves strong improvements over competitive baselines and near-perfect background preservation, establishing a practical solution for high-precision local refinement. Project Page: this https URL.
>
---
#### [new 061] Q-Zoom: Query-Aware Adaptive Perception for Efficient Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Q-Zoom，解决MLLMs在高分辨率输入下的效率问题，通过自适应感知提升推理速度与精度。**

- **链接: [https://arxiv.org/pdf/2604.06912](https://arxiv.org/pdf/2604.06912)**

> **作者:** Yuheng Shi; Xiaohuan Pei; Linfeng Wen; Minjing Dong; Chang Xu
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** MLLMs require high-resolution visual inputs for fine-grained tasks like document understanding and dense scene perception. However, current global resolution scaling paradigms indiscriminately flood the quadratic self-attention mechanism with visually redundant tokens, severely bottlenecking inference throughput while ignoring spatial sparsity and query intent. To overcome this, we propose Q-Zoom, a query-aware adaptive high-resolution perception framework that operates in an efficient coarse-to-fine manner. First, a lightweight Dynamic Gating Network safely bypasses high-resolution processing when coarse global features suffice. Second, for queries demanding fine-grained perception, a Self-Distilled Region Proposal Network (SD-RPN) precisely localizes the task-relevant Region-of-Interest (RoI) directly from intermediate feature spaces. To optimize these modules efficiently, the gating network uses a consistency-aware generation strategy to derive deterministic routing labels, while the SD-RPN employs a fully self-supervised distillation paradigm. A continuous spatio-temporal alignment scheme and targeted fine-tuning then seamlessly fuse the dense local RoI with the coarse global layout. Extensive experiments demonstrate that Q-Zoom establishes a dominant Pareto frontier. Using Qwen2.5-VL-7B as a primary testbed, Q-Zoom accelerates inference by 2.52 times on Document & OCR benchmarks and 4.39 times in High-Resolution scenarios while matching the baseline's peak accuracy. Furthermore, when configured for maximum perceptual fidelity, Q-Zoom surpasses the baseline's peak performance by 1.1% and 8.1% on these respective benchmarks. These robust improvements transfer seamlessly to Qwen3-VL, LLaVA, and emerging RL-based thinking-with-image models. Project page is available at this https URL.
>
---
#### [new 062] VersaVogue: Visual Expert Orchestration and Preference Alignment for Unified Fashion Synthesis
- **分类: cs.CV**

- **简介: 该论文提出VersaVogue，解决时尚图像生成中服装设计与虚拟试穿分离、多条件控制困难的问题。通过引入特征路由注意力模块和偏好优化管道，提升生成质量与可控性。**

- **链接: [https://arxiv.org/pdf/2604.07210](https://arxiv.org/pdf/2604.07210)**

> **作者:** Jian Yu; Fei Shen; Cong Wang; Yi Xin; Si Shen; Xiaoyu Du; Jinhui Tang
>
> **摘要:** Diffusion models have driven remarkable advancements in fashion image generation, yet prior works usually treat garment generation and virtual dressing as separate problems, limiting their flexibility in real-world fashion workflows. Moreover, fashion image synthesis under multi-source heterogeneous conditions remains challenging, as existing methods typically rely on simple feature concatenation or static layer-wise injection, which often causes attribute entanglement and semantic interference. To address these issues, we propose VersaVogue, a unified framework for multi-condition controllable fashion synthesis that jointly supports garment generation and virtual dressing, corresponding to the design and showcase stages of the fashion lifecycle. Specifically, we introduce a trait-routing attention (TA) module that leverages a mixture-of-experts mechanism to dynamically route condition features to the most compatible experts and generative layers, enabling disentangled injection of visual attributes such as texture, shape, and color. To further improve realism and controllability, we develop an automated multi-perspective preference optimization (MPO) pipeline that constructs preference data without human annotation or task-specific reward models. By combining evaluators of content fidelity, textual alignment, and perceptual quality, MPO identifies reliable preference pairs, which are then used to optimize the model via direct preference optimization (DPO). Extensive experiments on both garment generation and virtual dressing benchmarks demonstrate that VersaVogue consistently outperforms existing methods in visual fidelity, semantic consistency, and fine-grained controllability.
>
---
#### [new 063] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于图像分类任务，旨在提升视觉基础模型的可解释性。通过DINO-QPM方法，将复杂特征转化为可解释的表示，同时保持高分类精度。**

- **链接: [https://arxiv.org/pdf/2604.07166](https://arxiv.org/pdf/2604.07166)**

> **作者:** Robert Zimmermann; Thomas Norrenbrock; Bodo Rosenhahn
>
> **备注:** Accepted to the 5th Explainable AI for Computer Vision (XAI4CV) Workshop at CVPR 2026
>
> **摘要:** Although visual foundation models like DINOv2 provide state-of-the-art performance as feature extractors, their complex, high-dimensional representations create substantial hurdles for interpretability. This work proposes DINO-QPM, which converts these powerful but entangled features into contrastive, class-independent representations that are interpretable by humans. DINO-QPM is a lightweight interpretability adapter that pursues globally interpretable image classification, adapting the Quadratic Programming Enhanced Model (QPM) to operate on strictly frozen DINO backbones. While classification with visual foundation models typically relies on the \texttt{CLS} token, we deliberately diverge from this standard. By leveraging average-pooling, we directly connect the patch embeddings to the model's features and therefore enable spatial localisation of DINO-QPM's globally interpretable features within the input space. Furthermore, we apply a sparsity loss to minimise spatial scatter and background noise, ensuring that explanations are grounded in relevant object parts. With DINO-QPM we make the level of interpretability of QPM available as an adapter while exceeding the accuracy of DINOv2 linear probe. Evaluated through an introduced Plausibility metric and other interpretability metrics, extensive experiments demonstrate that DINO-QPM is superior to other applicable methods for frozen visual foundation models in both classification accuracy and explanation quality.
>
---
#### [new 064] Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning
- **分类: cs.CV**

- **简介: 该论文属于声源定位任务，旨在解决复杂场景下定位效果不佳的问题。提出一种无需训练的框架，利用多模态大语言模型进行生成、分析和优化，提升定位准确性。**

- **链接: [https://arxiv.org/pdf/2604.06824](https://arxiv.org/pdf/2604.06824)**

> **作者:** Subin Park; Jung Uk Kim
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Sound source localization task aims to identify the locations of sound-emitting objects by leveraging correlations between audio and visual modalities. Most existing SSL methods rely on contrastive learning-based feature matching, but lack explicit reasoning and verification, limiting their effectiveness in complex acoustic scenes. Inspired by human meta-cognitive processes, we propose a training-free SSL framework that exploits the intrinsic reasoning capabilities of Multimodal Large Language Models (MLLMs). Our Generation-Analysis-Refinement (GAR) pipeline consists of three stages: Generation produces initial bounding boxes and audio classifications; Analysis quantifies Audio-Visual Consistency via open-set role tagging and anchor voting; and Refinement applies adaptive gating to prevent unnecessary adjustments. Extensive experiments on single-source and multi-source benchmarks demonstrate competitive performance. The source code is available at this https URL.
>
---
#### [new 065] Time-driven Survival Analysis from FDG-PET/CT in Non-Small Cell Lung Cancer
- **分类: cs.CV**

- **简介: 该论文属于生存分析任务，旨在预测非小细胞肺癌患者的总生存期。通过结合影像和时间数据，构建深度学习模型提升预测效果。**

- **链接: [https://arxiv.org/pdf/2604.06885](https://arxiv.org/pdf/2604.06885)**

> **作者:** Sambit Tarai; Ashish Chauhan; Elin Lundström; Johan Öfverstedt; Therese Sjöholm; Veronica Sanchez Rodriguez; Håkan Ahlström; Joel Kullberg
>
> **备注:** Under review
>
> **摘要:** Purpose: Automated medical image-based prediction of clinical outcomes, such as overall survival (OS), has great potential in improving patient prognostics and personalized treatment planning. We developed a deep regression framework using tissue-wise FDG-PET/CT projections as input, along with a temporal input representing a scalar time horizon (in days) to predict OS in patients with Non-Small Cell Lung Cancer (NSCLC). Methods: The proposed framework employed a ResNet-50 backbone to process input images and generate corresponding image embeddings. The embeddings were then combined with temporal data to produce OS probabilities as a function of time, effectively parameterizing the predictions based on time. The overall framework was developed using the U-CAN cohort (n = 556) and evaluated by comparing with a baseline method on the test set (n = 292). The baseline utilized the ResNet-50 architecture, processing only the images as input and providing OS predictions at pre-specified intervals, such as 2- or 5-year. Results: The incorporation of temporal data with image embeddings demonstrated an advantage in predicting OS, outperforming the baseline method with an improvement in AUC of 4.3%. The proposed model using clinical + IDP features achieved strong performance, and an ensemble of imaging and clinical + IDP models achieved the best overall performance (0.788), highlighting the complementary value of multimodal inputs. The proposed method also enabled risk stratification of patients into distinct categories (high vs low risk). Heat maps from the saliency analysis highlighted tumor regions as key structures for the prediction. Conclusion: Our method provided an automated framework for predicting OS as a function of time and demonstrates the potential of combining imaging and tabular data for improved survival prediction.
>
---
#### [new 066] Assessing the Added Value of Onboard Earth Observation Processing with the IRIDE HEO Service Segment
- **分类: cs.CV; cs.AI; cs.AR; cs.ET**

- **简介: 论文探讨了星上遥感处理对提升地球观测服务性能的价值，针对地面处理的延迟和带宽限制问题，提出IRIDE HEO系统，通过星上预分类支持更及时、精细的灾害响应与土地管理。**

- **链接: [https://arxiv.org/pdf/2604.07120](https://arxiv.org/pdf/2604.07120)**

> **作者:** Parampuneet Kaur Thind; Charles Mwangi; Giovanni Varetto; Lorenzo Sarti; Andrea Papa; Andrea Taramelli
>
> **摘要:** Current operational Earth Observation (EO) services, including the Copernicus Emergency Management Service (CEMS), the European Forest Fire Information System (EFFIS), and the Copernicus Land Monitoring Service (CLMS), rely primarily on ground-based processing pipelines. While these systems provide mature large-scale information products, they remain constrained by downlink latency, bandwidth limitations, and limited capability for autonomous observation prioritisation. The International Report for an Innovative Defence of Earth (IRIDE) programme is a national Earth observation initiative led by the Italian government to support public authorities through timely, objective information derived from spaceborne data. Rather than a single constellation, IRIDE is designed as a constellation of constellations, integrating heterogeneous sensing technologies within a unified service-oriented architecture. Within this framework, Hawk for Earth Observation (HEO) enables onboard generation of data products, allowing information extraction earlier in the processing chain. This paper examines the limitations of ground-only architectures and evaluates the added value of onboard processing at the operational service level. The IRIDE burnt-area mapping service is used as a representative case study to demonstrate how onboard intelligence can support higher spatial detail (sub-three-metre ground sampling distance), smaller detectable events (minimum mapping unit of three hectares), and improved system responsiveness. Rather than replacing existing Copernicus services, the IRIDE HEO capability is positioned as a complementary layer providing image-driven pre-classification to support downstream emergency and land-management workflows. This work highlights the operational value of onboard intelligence for emerging low-latency EO service architectures.
>
---
#### [new 067] SCT-MOT: Enhancing Air-to-Air Multiple UAVs Tracking with Swarm-Coupled Motion and Trajectory Guidance
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，解决空中无人机群跟踪中的轨迹不连贯和身份混淆问题。提出SCT-MOT框架，融合群体运动建模与轨迹引导特征融合，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.06883](https://arxiv.org/pdf/2604.06883)**

> **作者:** Zhaochen Chu; Tao Song; Ren Jin; Shaoming He; Defu Lin; Siqing Cheng
>
> **备注:** 17 pages, 7 figures. Under review at IEEE Transactions on Aerospace and Electronic Systems (TAES). This work has been submitted to the IEEE for possible publication
>
> **摘要:** Air-to-air tracking of swarm UAVs presents significant challenges due to the complex nonlinear group motion and weak visual cues for small objects, which often cause detection failures, trajectory fragmentation, and identity switches. Although existing methods have attempted to improve performance by incorporating trajectory prediction, they model each object independently, neglecting the swarm-level motion dependencies. Their limited integration between motion prediction and appearance representation also weakens the spatio-temporal consistency required for tracking in visually ambiguous and cluttered environments, making it difficult to maintain coherent trajectories and reliable associations. To address these challenges, we propose SCT-MOT, a tracking framework that integrates Swarm-Coupled motion modeling and Trajectory-guided feature fusion. First, we develop a Swarm Motion-Aware Trajectory Prediction (SMTP) module jointly models historical trajectories and posture-aware appearance features from a swarm-level perspective, enabling more accurate forecasting of the nonlinear, coupled group trajectories. Second, we design a Trajectory-Guided Spatio-Temporal Feature Fusion (TG-STFF) module aligns predicted positions with historical visual cues and deeply integrates them with current frame features, enhancing temporal consistency and spatial discriminability for weak objects. Extensive experiments on three public air-to-air swarm UAV tracking datasets, including AIRMOT, MOT-FLY, and UAVSwarm, demonstrate that SMTP achieves more accurate trajectory forecasts and yields a 1.21\% IDF1 improvement over the state-of-the-art trajectory prediction module EqMotion when integrated into the same MOT framework. Overall, our SCT-MOT consistently achieves superior accuracy and robustness compared to state-of-the-art trackers across multiple metrics under complex swarm scenarios.
>
---
#### [new 068] Non-identifiability of Explanations from Model Behavior in Deep Networks of Image Authenticity Judgments
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究深度网络在图像真实性判断任务中是否能生成可识别的解释。通过测试不同模型的解释一致性，发现尽管模型预测准确，但解释不可靠，表明其机制与人类认知不同。**

- **链接: [https://arxiv.org/pdf/2604.07254](https://arxiv.org/pdf/2604.07254)**

> **作者:** Icaro Re Depaolini; Uri Hasson
>
> **摘要:** Deep neural networks can predict human judgments, but this does not imply that they rely on human-like information or reveal the cues underlying those judgments. Prior work has addressed this issue using attribution heatmaps, but their explanatory value in itself depends on robustness. Here we tested the robustness of such explanations by evaluating whether models that predict human authenticity ratings also produce consistent explanations within and across architectures. We fit lightweight regression heads to multiple frozen pretrained vision models and generated attribution maps using Grad-CAM, LIME, and multiscale pixel masking. Several architectures predicted ratings well, reaching about 80% of the noise ceiling. VGG models achieved this by tracking image quality rather than authenticity-specific variance, limiting the relevance of their attributions. Among the remaining models, attribution maps were generally stable across random seeds within an architecture, especially for EfficientNetB3 and Barlow Twins, and consistency was higher for images judged as more authentic. Crucially, agreement in attribution across architectures was weak even when predictive performance was similar. To address this, we combined models in ensembles, which improved prediction of human authenticity judgments and enabled image-level attribution via pixel masking. We conclude that while deep networks can predict human authenticity judgments well, they do not produce identifiable explanations for those judgments. More broadly, our findings suggest that post hoc explanations from successful models of behavior should be treated as weak evidence for cognitive mechanism.
>
---
#### [new 069] MTA-Agent: An Open Recipe for Multimodal Deep Search Agents
- **分类: cs.CV**

- **简介: 该论文提出MTA-Agent，解决多模态深度搜索中的复杂推理问题，通过构建高质量多跳数据集提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.06376](https://arxiv.org/pdf/2604.06376)**

> **作者:** Xiangyu Peng; Can Qin; An Yan; Xinyi Yang; Zeyuan Chen; Ran Xu; Chien-Sheng Wu
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated strong capabilities in visual understanding, yet they remain limited in complex, multi-step reasoning that requires deep searching and integrating visual evidence with external knowledge. In this work, we address this challenge by constructing high-quality, verified multi-hop vision-language training data for multimodal deep-search agents. We propose a Multi-hop Tool-Augmented Agent for Evidence-based QA Synthesis (MTA-Agent), which automatically selects tools and their parameters to retrieve and validate evidence from both visual and textual sources and generates structured multi-hop question-answer trajectories. Starting from diverse VQA seed datasets, our pipeline produces a large-scale training dataset, MTA-Vision-DeepSearch, containing 21K high-quality multi-hop examples. The data is filtered through a multi-stage verification process to ensure factual consistency and answer uniqueness. Using MTA-Vision-DeepSearch, a 32B open-source multimodal search agent achieves state-of-the-art performance, reaching an average of 54.63\% across six challenging benchmarks, outperforming GPT-5 (51.86\%), Gemini-2.5-Pro (50.98\%), and Gemini-3-Pro (54.46\%) under the same tool settings. We further show that training on our data improves both reasoning depth and tool-use behavior, increasing the average number of steps from 2.27 to 4.28, and leading to more systematic and persistent search strategies. Additionally, we demonstrate that training can be performed without real-time tool calls by replaying cached interactions, significantly reducing training cost. Importantly, we present MTA-Agent as a fully open recipe for multimodal deep search: we release the entire dataset, training trajectories, and implementation details to enable reproducibility and future research on open multimodal search agents.
>
---
#### [new 070] Exploring 6D Object Pose Estimation with Deformation
- **分类: cs.CV**

- **简介: 该论文属于6D物体位姿估计任务，解决物体变形带来的位姿估计难题。提出DeSOPE数据集，包含26类物体的多形态3D扫描与大量标注数据，用于研究变形物体的位姿估计方法。**

- **链接: [https://arxiv.org/pdf/2604.06720](https://arxiv.org/pdf/2604.06720)**

> **作者:** Zhiqiang Liu; Rui Song; Duanmu Chuangqi; Jiaojiao Li; David Ferstl; Yinlin Hu
>
> **摘要:** We present DeSOPE, a large-scale dataset for 6DoF deformed objects. Most 6D object pose methods assume rigid or articulated objects, an assumption that fails in practice as objects deviate from their canonical shapes due to wear, impact, or deformation. To model this, we introduce the DeSOPE dataset, which features high-fidelity 3D scans of 26 common object categories, each captured in one canonical state and three deformed configurations, with accurate 3D registration to the canonical mesh. Additionally, it features an RGB-D dataset with 133K frames across diverse scenarios and 665K pose annotations produced via a semi-automatic pipeline. We begin by annotating 2D masks for each instance, then compute initial poses using an object pose method, refine them through an object-level SLAM system, and finally perform manual verification to produce the final annotations. We evaluate several object pose methods and find that performance drops sharply with increasing deformation, suggesting that robust handling of such deformations is critical for practical applications. The project page and dataset are available at this https URL}{this https URL.
>
---
#### [new 071] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation
- **分类: cs.CV**

- **简介: 该论文属于内容安全任务，研究如何通过对抗性走私攻击绕过MLLM的内容审核，分析攻击路径并提出防御方法。**

- **链接: [https://arxiv.org/pdf/2604.06950](https://arxiv.org/pdf/2604.06950)**

> **作者:** Zhiheng Li; Zongyang Ma; Yuntong Pan; Ziqi Zhang; Xiaolei Lv; Bo Li; Jun Gao; Jianing Zhang; Chunfeng Yuan; Bing Li; Weiming Hu
>
> **备注:** Accepted to ACL 2026. 19 pages, 6 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly being deployed as automated content moderators. Within this landscape, we uncover a critical threat: Adversarial Smuggling Attacks. Unlike adversarial perturbations (for misclassification) and adversarial jailbreaks (for harmful output generation), adversarial smuggling exploits the Human-AI capability gap. It encodes harmful content into human-readable visual formats that remain AI-unreadable, thereby evading automated detection and enabling the dissemination of harmful content. We classify smuggling attacks into two pathways: (1) Perceptual Blindness, disrupting text recognition; and (2) Reasoning Blockade, inhibiting semantic understanding despite successful text recognition. To evaluate this threat, we constructed SmuggleBench, the first comprehensive benchmark comprising 1,700 adversarial smuggling attack instances. Evaluations on SmuggleBench reveal that both proprietary (e.g., GPT-5) and open-source (e.g., Qwen3-VL) state-of-the-art models are vulnerable to this threat, producing Attack Success Rates (ASR) exceeding 90%. By analyzing the vulnerability through the lenses of perception and reasoning, we identify three root causes: the limited capabilities of vision encoders, the robustness gap in OCR, and the scarcity of domain-specific adversarial examples. We conduct a preliminary exploration of mitigation strategies, investigating the potential of test-time scaling (via CoT) and adversarial training (via SFT) to mitigate this threat. Our code is publicly available at this https URL.
>
---
#### [new 072] GPAFormer: Graph-guided Patch Aggregation Transformer for Efficient 3D Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像分割任务，旨在解决多器官、多模态分割中的精度与效率平衡问题。提出GPAFormer网络，结合两个核心模块提升分割效果与计算效率。**

- **链接: [https://arxiv.org/pdf/2604.06658](https://arxiv.org/pdf/2604.06658)**

> **作者:** Chung-Ming Lo; I-Yun Liu; Wei-Yang Lin
>
> **摘要:** Deep learning has been widely applied to 3D medical image segmentation tasks. However, due to the diversity of imaging modalities, the high-dimensional nature of the data, and the heterogeneity of anatomical structures, achieving both segmentation accuracy and computational efficiency in multi-organ segmentation remains a challenge. This study proposed GPAFormer, a lightweight network architecture specifically designed for 3D medical image segmentation, emphasizing efficiency while keeping high accuracy. GPAFormer incorporated two core modules: the multi-scale attention-guided stacked aggregation (MASA) and the mutual-aware patch graph aggregator (MPGA). MASA utilized three parallel paths with different receptive fields, combined through planar aggregation, to enhance the network's capability in handling structures of varying sizes. MPGA employed a graph-guided approach to dynamically aggregate regions with similar feature distributions based on inter-patch feature similarity and spatial adjacency, thereby improving the discrimination of both internal and boundary structures of organs. Experiments were performed on public whole-body CT and MRI datasets including BTCV, Synapse, ACDC, and BraTS. Compared to the existed 3D segmentation networkd, GPAFormer using only 1.81 M parameters achieved overall highest DSC on BTCV (75.70%), Synapse (81.20%), ACDC (89.32%), and BraTS (82.74%). Using consumer level GPU, the inference time for one validation case of BTCV spent less than one second. The results demonstrated that GPAFormer balanced accuracy and efficiency in multi-organ, multi-modality 3D segmentation tasks across various clinical scenarios especially for resource-constrained and time-sensitive clinical environments.
>
---
#### [new 073] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于联邦学习任务，旨在解决域偏移问题。通过构建领域感知的原型，提升模型在不同域间的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.06795](https://arxiv.org/pdf/2604.06795)**

> **作者:** Huy Q. Le; Loc X. Nguyen; Yu Qiao; Seong Tae Kim; Eui-Nam Huh; Choong Seon Hong
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Federated Learning (FL) enables decentralized model training across multiple clients without exposing private data, making it ideal for privacy-sensitive applications. However, in real-world FL scenarios, clients often hold data from distinct domains, leading to severe domain shift and degraded global model performance. To address this, prototype learning has been emerged as a promising solution, which leverages class-wise feature representations. Yet, existing methods face two key limitations: (1) Existing prototype-based FL methods typically construct a $\textit{single global prototype}$ per class by aggregating local prototypes from all clients without preserving domain information. (2) Current feature-prototype alignment is $\textit{domain-agnostic}$, forcing clients to align with global prototypes regardless of domain origin. To address these challenges, we propose Federated Domain-Aware Prototypes (FedDAP) to construct domain-specific global prototypes by aggregating local client prototypes within the same domain using a similarity-weighted fusion mechanism. These global domain-specific prototypes are then used to guide local training by aligning local features with prototypes from the same domain, while encouraging separation from prototypes of different domains. This dual alignment enhances domain-specific learning at the local level and enables the global model to generalize across diverse domains. Finally, we conduct extensive experiments on three different datasets: DomainNet, Office-10, and PACS to demonstrate the effectiveness of our proposed framework to address the domain shift challenges. The code is available at this https URL.
>
---
#### [new 074] Location Is All You Need: Continuous Spatiotemporal Neural Representations of Earth Observation Data
- **分类: cs.CV**

- **简介: 该论文提出LIANet，用于建模地球观测数据的连续时空神经表示，解决卫星图像重建与下游任务适应问题，无需原始数据即可进行微调。**

- **链接: [https://arxiv.org/pdf/2604.07092](https://arxiv.org/pdf/2604.07092)**

> **作者:** Mojgan Madadikhaljan; Jonathan Prexl; Isabelle Wittmann; Conrad M Albrecht; Michael Schmitt
>
> **摘要:** In this work, we present LIANet (Location Is All You Need Network), a coordinate-based neural representation that models multi-temporal spaceborne Earth observation (EO) data for a given region of interest as a continuous spatiotemporal neural field. Given only spatial and temporal coordinates, LIANet reconstructs the corresponding satellite imagery. Once pretrained, this neural representation can be adapted to various EO downstream tasks, such as semantic segmentation or pixel-wise regression, importantly, without requiring access to the original satellite data. LIANet intends to serve as a user-friendly alternative to Geospatial Foundation Models (GFMs) by eliminating the overhead of data access and preprocessing for end-users and enabling fine-tuning solely based on labels. We demonstrate the pretraining of LIANet across target areas of varying sizes and show that fine-tuning it for downstream tasks achieves competitive performance compared to training from scratch or using established GFMs. The source code and datasets are publicly available at this https URL.
>
---
#### [new 075] Canopy Tree Height Estimation Using Quantile Regression: Modeling and Evaluating Uncertainty in Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于树高估计任务，旨在解决传统方法缺乏不确定性评估的问题。通过量化回归改进模型，提升风险敏感场景下的适用性。**

- **链接: [https://arxiv.org/pdf/2604.06988](https://arxiv.org/pdf/2604.06988)**

> **作者:** Karsten Schrödter; Jan Pauls; Fabian Gieseke
>
> **备注:** Accepted to AISTATS 2026
>
> **摘要:** Accurate tree height estimation is vital for ecological monitoring and biomass assessment. We apply quantile regression to existing tree height estimation models based on satellite data to incorporate uncertainty quantification. Most current approaches for tree height estimation rely on point predictions, which limits their applicability in risk-sensitive scenarios. In this work, we show that, with minor modifications of a given prediction head, existing models can be adapted to provide statistically calibrated uncertainty estimates via quantile regression. Furthermore, we demonstrate how our results correlate with known challenges in remote sensing (e.g., terrain complexity, vegetation heterogeneity), indicating that the model is less confident in more challenging conditions.
>
---
#### [new 076] Balancing Efficiency and Restoration: Lightweight Mamba-Based Model for CT Metal Artifact Reduction
- **分类: cs.CV**

- **简介: 该论文属于CT图像金属伪影去除任务，解决现有方法结构破坏、依赖sinogram数据及效率不平衡问题，提出MARMamba模型实现高效准确的伪影消除。**

- **链接: [https://arxiv.org/pdf/2604.06622](https://arxiv.org/pdf/2604.06622)**

> **作者:** Weikai Qu; Sijun Liang; Xianfeng Li; Cheng Pan; An Yan; Ahmed Elazab; Shanzhou Niu; Dong Zeng; Xiang Wan; Changmiao Wang
>
> **备注:** Accepted by IEEE Transactions on Radiation and Plasma Medical Sciences
>
> **摘要:** In computed tomography imaging, metal implants frequently generate severe artifacts that compromise image quality and hinder diagnostic accuracy. There are three main challenges in the existing methods: the deterioration of organ and tissue structures, dependence on sinogram data, and an imbalance between resource use and restoration efficiency. Addressing these issues, we introduce MARMamba, which effectively eliminates artifacts caused by metals of different sizes while maintaining the integrity of the original anatomical structures of the image. Furthermore, this model only focuses on CT images affected by metal artifacts, thus negating the requirement for additional input data. The model is a streamlined UNet architecture, which incorporates multi-scale Mamba (MS-Mamba) as its core module. Within MS-Mamba, a flip mamba block captures comprehensive contextual information by analyzing images from multiple orientations. Subsequently, the average maximum feed-forward network integrates critical features with average features to suppress the artifacts. This combination allows MARMamba to reduce artifacts efficiently. The experimental results demonstrate that our model excels in reducing metal artifacts, offering distinct advantages over other models. It also strikes an optimal balance between computational demands, memory usage, and the number of parameters, highlighting its practical utility in the real world. The code of the presented model is available at: this https URL.
>
---
#### [new 077] ModuSeg: Decoupling Object Discovery and Semantic Retrieval for Training-Free Weakly Supervised Segmentation
- **分类: cs.CV**

- **简介: 该论文属于弱监督语义分割任务，旨在解决模型依赖稀疏区域、难以处理伪标签噪声的问题。提出ModuSeg框架，通过解耦目标发现与语义分配，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.07021](https://arxiv.org/pdf/2604.07021)**

> **作者:** Qingze He; Fagui Liu; Dengke Zhang; Qingmao Wei; Quan Tang
>
> **摘要:** Weakly supervised semantic segmentation aims to achieve pixel-level predictions using image-level labels. Existing methods typically entangle semantic recognition and object localization, which often leads models to focus exclusively on sparse discriminative regions. Although foundation models show immense potential, many approaches still follow the tightly coupled optimization paradigm, struggling to effectively alleviate pseudo-label noise and often relying on time-consuming multi-stage retraining or unstable end-to-end joint optimization. To address the above challenges, we present ModuSeg, a training-free weakly supervised semantic segmentation framework centered on explicitly decoupling object discovery and semantic assignment. Specifically, we integrate a general mask proposer to extract geometric proposals with reliable boundaries, while leveraging semantic foundation models to construct an offline feature bank, transforming segmentation into a non-parametric feature retrieval process. Furthermore, we propose semantic boundary purification and soft-masked feature aggregation strategies to effectively mitigate boundary ambiguity and quantization errors, thereby extracting high-quality category prototypes. Extensive experiments demonstrate that the proposed decoupled architecture better preserves fine boundaries without parameter fine-tuning and achieves highly competitive performance on standard benchmark datasets. Code is available at this https URL.
>
---
#### [new 078] Not all tokens contribute equally to diffusion learning
- **分类: cs.CV**

- **简介: 该论文针对文本到视频生成任务，解决模型在推理中忽略语义重要标记的问题。提出DARE框架，通过分布去偏和空间对齐提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.07026](https://arxiv.org/pdf/2604.07026)**

> **作者:** Guoqing Zhang; Lu Shi; Wanru Xu; Linna Zhang; Sen Wang; Fangfang Wang; Yigang Cen
>
> **摘要:** With the rapid development of conditional diffusion models, significant progress has been made in text-to-video generation. However, we observe that these models often neglect semantically important tokens during inference, leading to biased or incomplete generations under classifier-free guidance. We attribute this issue to two key factors: distributional bias caused by the long-tailed token frequency in training data, and spatial misalignment in cross-attention where semantically important tokens are overshadowed by less informative ones. To address these issues, we propose Distribution-Aware Rectification and Spatial Ensemble (DARE), a unified framework that improves semantic guidance in diffusion models from the perspectives of distributional debiasing and spatial consistency. First, we introduce Distribution-Rectified Classifier-Free Guidance (DR-CFG), which regularizes the training process by dynamically suppressing dominant tokens with low semantic density, encouraging the model to better capture underrepresented semantic cues and learn a more balanced conditional distribution. This design mitigates the risk of the model distribution overfitting to tokens with low semantic density. Second, we propose Spatial Representation Alignment (SRA), which adaptively reweights cross-attention maps according to token importance and enforces representation consistency, enabling semantically important tokens to exert stronger spatial guidance during generation. This mechanism effectively prevents low semantic-density tokens from dominating the attention allocation, thereby avoiding the dilution of the spatial and distributional guidance provided by high semantic-density tokens. Extensive experiments on multiple benchmark datasets demonstrate that DARE consistently improves generation fidelity and semantic alignment, achieving significant gains over existing approaches.
>
---
#### [new 079] Walk the Talk: Bridging the Reasoning-Action Gap for Thinking with Images via Multimodal Agentic Policy Optimization
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决模型文本推理与视觉动作不一致的问题。通过引入MAPO方法，增强文本描述与视觉内容的对齐，提升多模态推理能力。**

- **链接: [https://arxiv.org/pdf/2604.06777](https://arxiv.org/pdf/2604.06777)**

> **作者:** Wenhao Yang; Yu Xia; Jinlong Huang; Shiyin Lu; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang; Yuchen Zhou; Xiaobo Xia; Yuanyu Wan; Lijun Zhang; Tat-Seng Chua
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs) have incentivized models to ``think with images'' by actively invoking visual tools during multi-turn reasoning. The common Reinforcement Learning (RL) practice of relying on outcome-based rewards ignores the fact that textual plausibility often masks executive failure, meaning that models may exhibit intuitive textual reasoning while executing imprecise or irrelevant visual actions within their agentic reasoning trajectories. This reasoning-action discrepancy introduces noise that accumulates throughout the multi-turn reasoning process, severely degrading the model's multimodal reasoning capabilities and potentially leading to training collapse. In this paper, we introduce Multimodal Agentic Policy Optimization (MAPO), bridging the gap between textual reasoning and visual actions generated by models within their Multimodal Chain-of-Thought (MCoT). Specifically, MAPO mandates the model to generate explicit textual descriptions for the visual content obtained via tool usage. We then employ a novel advantage estimation that couples the semantic alignment between these descriptions and the actual observations with the task reward. Theoretical findings are provided to justify the rationale behind MAPO, which inherently reduces the variance of gradients, and extensive experiments demonstrate that our method achieves superior performance across multiple visual reasoning benchmarks.
>
---
#### [new 080] Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，解决超远距离物体检测难题。针对高精度检测需求，提出Telescope模型，提升远距离检测效果。**

- **链接: [https://arxiv.org/pdf/2604.06332](https://arxiv.org/pdf/2604.06332)**

> **作者:** Parker Ewen; Dmitriy Rivkin; Mario Bijelic; Felix Heide
>
> **备注:** Project website: this https URL
>
> **摘要:** Autonomous highway driving, especially for long-haul heavy trucks, requires detecting objects at long ranges beyond 500 meters to satisfy braking distance requirements at high speeds. At long distances, vehicles and other critical objects occupy only a few pixels in high-resolution images, causing state-of-the-art object detectors to fail. This challenge is compounded by the limited effective range of commercially available LiDAR sensors, which fall short of ultra-long range thresholds because of quadratic loss of resolution with distance, making image-based detection the most practically scalable solution given commercially available sensor constraints. We introduce Telescope, a two-stage detection model designed for ultra-long range autonomous driving. Alongside a powerful detection backbone, this model contains a novel re-sampling layer and image transformation to address the fundamental challenges of detecting small, distant objects. Telescope achieves $76\%$ relative improvement in mAP in ultra-long range detection compared to state-of-the-art methods (improving from an absolute mAP of 0.185 to 0.326 at distances beyond 250 meters), requires minimal computational overhead, and maintains strong performance across all detection ranges.
>
---
#### [new 081] PRISM: Rethinking Scattered Atmosphere Reconstruction as a Unified Understanding and Generation Model for Real-world Dehazing
- **分类: cs.CV**

- **简介: 该论文属于真实场景去雾任务，解决非均匀雾霾分布与缺乏配对数据的问题。提出PSAR框架，联合重建清晰场景与散射变量，提升去雾效果。**

- **链接: [https://arxiv.org/pdf/2604.07048](https://arxiv.org/pdf/2604.07048)**

> **作者:** Chengyu Fang; Chunming He; Yuelin Zhang; Chubin Chen; Chenyang Zhu; Longxiang Tang; Xiu Li
>
> **备注:** 24 Pages, 7 Figures
>
> **摘要:** Real-world image dehazing (RID) aims to remove haze induced degradation from real scenes. This task remains challenging due to non-uniform haze distribution, spatially varying illumination from multiple light sources, and the scarcity of paired real hazy-clean data. In PRISM, we propose Proximal Scattered Atmosphere Reconstruction (PSAR), a physically structured framework that jointly reconstructs the clear scene and scattering variables under the atmospheric scattering model, thereby improving reliability in complex regions and mixed-light conditions. To bridge the synthetic-to-real gap, we design an online non-uniform haze synthesis pipeline and a Selective Self-distillation Adaptation scheme for unpaired real-world scenarios, which enables the model to selectively learn from high-quality perceptual targets while leveraging its intrinsic scattering understanding to audit residual haze and guide self-refinement. Extensive experiments on real-world benchmarks demonstrate that PRISM achieves state-of-the-art performance on RID tasks.
>
---
#### [new 082] EventFace: Event-Based Face Recognition via Structure-Driven Spatiotemporal Modeling
- **分类: cs.CV**

- **简介: 该论文属于事件相机下的面部识别任务，解决传统RGB系统在事件流中无法依赖稳定光度特征的问题。通过结构驱动的时空建模，提出EventFace框架，提升识别性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.06782](https://arxiv.org/pdf/2604.06782)**

> **作者:** Qingguo Meng; Xingbo Dong; Zhe Jin; Massimo Tistarelli
>
> **摘要:** Event cameras offer a promising sensing modality for face recognition due to their inherent advantages in illumination robustness and privacy-friendliness. However, because event streams lack the stable photometric appearance relied upon by conventional RGB-based face recognition systems, we argue that event-based face recognition should model structure-driven spatiotemporal identity representations shaped by rigid facial motion and individual facial geometry. Since dedicated datasets for event-based face recognition remain lacking, we construct EFace, a small-scale event-based face dataset captured under rigid facial motion. To learn effectively from this limited event data, we further propose EventFace, a framework for event-based face recognition that integrates spatial structure and temporal dynamics for identity modeling. Specifically, we employ Low-Rank Adaptation (LoRA) to transfer structural facial priors from pretrained RGB face models to the event domain, thereby establishing a reliable spatial basis for identity modeling. Building on this foundation, we further introduce a Motion Prompt Encoder (MPE) to explicitly encode temporal features and a Spatiotemporal Modulator (STM) to fuse them with spatial features, thereby enhancing the representation of identity-relevant event patterns. Extensive experiments demonstrate that EventFace achieves the best performance among the evaluated baselines, with a Rank-1 identification rate of 94.19% and an equal error rate (EER) of 5.35%. Results further indicate that EventFace exhibits stronger robustness under degraded illumination than the competing methods. In addition, the learned representations exhibit reduced template reconstructability.
>
---
#### [new 083] AnchorSplat: Feed-Forward 3D Gaussian SplattingWith 3D Geometric Priors
- **分类: cs.CV**

- **简介: 该论文提出AnchorSplat，解决3D场景重建任务中的效率与精度问题。通过引入3D几何先验的锚点对齐高斯表示，减少高斯数量，提升计算效率和重建质量。**

- **链接: [https://arxiv.org/pdf/2604.07053](https://arxiv.org/pdf/2604.07053)**

> **作者:** Xiaoxue Zhang; Xiaoxu Zheng; Yixuan Yin; Tiao Zhao; Kaihua Tang; Michael Bi Mi; Zhan Xu; Dave Zhenyu Chen
>
> **摘要:** Recent feed-forward Gaussian reconstruction models adopt a pixel-aligned formulation that maps each 2D pixel to a 3D Gaussian, entangling Gaussian representations tightly with the input images. In this paper, we propose AnchorSplat, a novel feed-forward 3DGS framework for scene-level reconstruction that represents the scene directly in 3D space. AnchorSplat introduces an anchor-aligned Gaussian representation guided by 3D geometric priors (e.g., sparse point clouds, voxels, or RGB-D point clouds), enabling a more geometry-aware renderable 3D Gaussians that is independent of image resolution and number of views. This design substantially reduces the number of required Gaussians, improving computational efficiency while enhancing reconstruction fidelity. Beyond the anchor-aligned design, we utilize a Gaussian Refiner to adjust the intermediate Gaussiansy via merely a few forward passes. Experiments on the ScanNet++ v2 NVS benchmark demonstrate the SOTA performance, outperforming previous methods with more view-consistent and substantially fewer Gaussian primitives.
>
---
#### [new 084] CAAP: Capture-Aware Adversarial Patch Attacks on Palmprint Recognition Models
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于生物特征识别安全任务，旨在解决深度掌纹识别系统对物理可实现攻击的脆弱性问题。提出CAAP框架，通过捕获感知的对抗补丁进行攻击，验证了系统的安全性缺陷。**

- **链接: [https://arxiv.org/pdf/2604.06987](https://arxiv.org/pdf/2604.06987)**

> **作者:** Renyang Liu; Jiale Li; Jie Zhang; Cong Wu; Xiaojun Jia; Shuxin Li; Wei Zhou; Kwok-Yan Lam; See-kiong Ng
>
> **摘要:** Palmprint recognition is deployed in security-critical applications, including access control and palm-based payment, due to its contactless acquisition and highly discriminative ridge-and-crease textures. However, the robustness of deep palmprint recognition systems against physically realizable attacks remains insufficiently understood. Existing studies are largely confined to the digital setting and do not adequately account for the texture-dominant nature of palmprint recognition or the distortions introduced during physical acquisition. To address this gap, we propose CAAP, a capture-aware adversarial patch framework for palmprint recognition. CAAP learns a universal patch that can be reused across inputs while remaining effective under realistic acquisition variation. To match the structural characteristics of palmprints, the framework adopts a cross-shaped patch topology, which enlarges spatial coverage under a fixed pixel budget and more effectively disrupts long-range texture continuity. CAAP further integrates three modules: ASIT for input-conditioned patch rendering, RaS for stochastic capture-aware simulation, and MS-DIFE for feature-level identity-disruptive guidance. We evaluate CAAP on the Tongji, IITD, and AISEC datasets against generic CNN backbones and palmprint-specific recognition models. Experiments show that CAAP achieves strong untargeted and targeted attack performance with favorable cross-model and cross-dataset transferability. The results further show that, although adversarial training can partially reduce the attack success rate, substantial residual vulnerability remains. These findings indicate that deep palmprint recognition systems remain vulnerable to physically realizable, capture-aware adversarial patch attacks, underscoring the need for more effective defenses in practice. Code available at this https URL.
>
---
#### [new 085] Hybrid ResNet-1D-BiGRU with Multi-Head Attention for Cyberattack Detection in Industrial IoT Environments
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于工业物联网入侵检测任务，旨在解决传统方法在实时性和准确率上的不足。通过融合ResNet-1D、BiGRU和多头注意力机制，提升特征提取与权重分配效果，实现高效精准的攻击检测。**

- **链接: [https://arxiv.org/pdf/2604.06481](https://arxiv.org/pdf/2604.06481)**

> **作者:** Afrah Gueriani; Hamza Kheddar; Ahmed Cherif Mazari
>
> **摘要:** This study introduces a hybrid deep learning model for intrusion detection in Industrial IoT (IIoT) systems, combining ResNet-1D, BiGRU, and Multi-Head Attention (MHA) for effective spatial-temporal feature extraction and attention-based feature weighting. To address class imbalance, SMOTE was applied during training on the EdgeHoTset dataset. The model achieved 98.71% accuracy, a loss of 0.0417%, and low inference latency (0.0001 sec /instance), demonstrating strong real-time capability. To assess generalizability, the model was also tested on the CICIoV2024 dataset, where it reached 99.99% accuracy and F1-score, with a loss of 0.0028, 0 % FPR, and 0.00014 sec/instance inference time. Across all metrics and datasets, the proposed model outperformed existing methods, confirming its robustness and effectiveness for real-time IoT intrusion detection.
>
---
#### [new 086] INSPATIO-WORLD: A Real-Time 4D World Simulator via Spatiotemporal Autoregressive Modeling
- **分类: cs.CV**

- **简介: 该论文提出INSPATIO-WORLD，解决实时4D场景模拟中的空间一致性与交互精度问题。通过STAR架构和JDMD方法，实现从单视频生成高保真动态场景。**

- **链接: [https://arxiv.org/pdf/2604.07209](https://arxiv.org/pdf/2604.07209)**

> **作者:** InSpatio Team; Donghui Shen; Guofeng Zhang; Haomin Liu; Haoyu Ji; Hujun Bao; Hongjia Zhai; Jialin Liu; Jing Guo; Nan Wang; Siji Pan; Weihong Pan; Weijian Xie; Xianbin Liu; Xiaojun Xiang; Xiaoyu Zhang; Xinyu Chen; Yifu Wang; Yipeng Chen; Zhenzhou Fan; Zhewen Le; Zhichao Ye; Ziqiang Zhao
>
> **摘要:** Building world models with spatial consistency and real-time interactivity remains a fundamental challenge in computer vision. Current video generation paradigms often struggle with a lack of spatial persistence and insufficient visual realism, making it difficult to support seamless navigation in complex environments. To address these challenges, we propose INSPATIO-WORLD, a novel real-time framework capable of recovering and generating high-fidelity, dynamic interactive scenes from a single reference video. At the core of our approach is a Spatiotemporal Autoregressive (STAR) architecture, which enables consistent and controllable scene evolution through two tightly coupled components: Implicit Spatiotemporal Cache aggregates reference and historical observations into a latent world representation, ensuring global consistency during long-horizon navigation; Explicit Spatial Constraint Module enforces geometric structure and translates user interactions into precise and physically plausible camera trajectories. Furthermore, we introduce Joint Distribution Matching Distillation (JDMD). By using real-world data distributions as a regularizing guide, JDMD effectively overcomes the fidelity degradation typically caused by over-reliance on synthetic data. Extensive experiments demonstrate that INSPATIO-WORLD significantly outperforms existing state-of-the-art (SOTA) models in spatial consistency and interaction precision, ranking first among real-time interactive methods on the WorldScore-Dynamic benchmark, and establishing a practical pipeline for navigating 4D environments reconstructed from monocular videos.
>
---
#### [new 087] PhysHead: Simulation-Ready Gaussian Head Avatars
- **分类: cs.CV**

- **简介: 该论文提出PhysHead，解决头像中头发动态模拟问题，通过3D高斯层和物理引擎实现真实头发运动，提升数字头像表现力。**

- **链接: [https://arxiv.org/pdf/2604.06467](https://arxiv.org/pdf/2604.06467)**

> **作者:** Berna Kabadayi; Vanessa Sklyarova; Wojciech Zielonka; Justus Thies; Gerard Pons-Moll
>
> **备注:** Project Page: see this https URL Youtube Video: see this https URL Accepted to CVPR 2026
>
> **摘要:** Realistic digital avatars require expressive and dynamic hair motion; however, most existing head avatar methods assume rigid hair movement. These methods often fail to disentangle hair from the head, representing it as a simple outer shell and failing to capture its natural volumetric behavior. In this paper, we address these limitations by introducing PhysHead, a hybrid representation for animatable head avatars with realistic hair dynamics learned from multi-view video. At the core is a 3D Gaussian-based layered representation of the head. Our approach combines a 3D parametric mesh for the head with strand-based hair, which can be directly simulated using physics engines. For the appearance model, we employ Gaussian primitives attached to both the head mesh and hair segments. This representation enables the creation of photorealistic head avatars with dynamic hair behavior, such as wind-blown motion, overcoming the constraints of rigid hair in existing methods. However, these animation capabilities also require new training schemes. In particular, we propose the use of VLM-based models to generate appearance of regions that are occluded in the dynamic training sequences. In quantitative and qualitative studies, we demonstrate the capabilities of the proposed model and compare it with existing baselines. We show that our method can synthesize physically plausible hair motion besides expression and camera control.
>
---
#### [new 088] MorphDistill: Distilling Unified Morphological Knowledge from Pathology Foundation Models for Colorectal Cancer Survival Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于结直肠癌生存预测任务，旨在解决现有模型忽视器官特异性特征的问题。通过多阶段知识蒸馏方法，构建高效病理特征编码器，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2604.06390](https://arxiv.org/pdf/2604.06390)**

> **作者:** Hikmat Khan; Usama Sajjad; Metin N. Gurcan; Anil Parwani; Wendy L. Frankel; Wei Chen; Muhammad Khalid Khan Niazi
>
> **摘要:** Background: Colorectal cancer (CRC) remains a leading cause of cancer-related mortality worldwide. Accurate survival prediction is essential for treatment stratification, yet existing pathology foundation models often overlook organ-specific features critical for CRC prognostication. Methods: We propose MorphDistill, a two-stage framework that distills complementary knowledge from multiple pathology foundation models into a compact CRC-specific encoder. In Stage I, a student encoder is trained using dimension-agnostic multi-teacher relational distillation with supervised contrastive regularization on large-scale colorectal datasets. This preserves inter-sample relationships from ten foundation models without explicit feature alignment. In Stage II, the encoder extracts patch-level features from whole-slide images, which are aggregated via attention-based multiple instance learning to predict five-year survival. Results: On the Alliance/CALGB 89803 cohort (n=424, stage III CRC), MorphDistill achieves an AUC of 0.68 (SD 0.08), an approximately 8% relative improvement over the strongest baseline (AUC 0.63). It also attains a C-index of 0.661 and a hazard ratio of 2.52 (95% CI: 1.73-3.65), outperforming all baselines. On an external TCGA cohort (n=562), it achieves a C-index of 0.628, demonstrating strong generalization across datasets and robustness across clinical subgroups. Conclusion: MorphDistill enables task-specific representation learning by integrating knowledge from multiple foundation models into a unified encoder. This approach provides an efficient strategy for prognostic modeling in computational pathology, with potential for broader oncology applications. Further validation across additional cohorts and disease stages is warranted.
>
---
#### [new 089] Learning to Search: A Decision-Based Agent for Knowledge-Based Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文属于知识增强的视觉问答任务，解决传统方法在检索与推理分离、适应性差的问题。提出基于决策的搜索代理，通过多步决策优化检索与回答流程。**

- **链接: [https://arxiv.org/pdf/2604.07146](https://arxiv.org/pdf/2604.07146)**

> **作者:** Zhuohong Chen; Zhenxian Wu; Yunyao Yu; Hangrui Xu; Zirui Liao; Zhifang Liu; Xiangwen Deng; Pen Jiao; Haoqian Wang
>
> **摘要:** Knowledge-based visual question answering (KB-VQA) requires vision-language models to understand images and use external knowledge, especially for rare entities and long-tail facts. Most existing retrieval-augmented generation (RAG) methods adopt a fixed pipeline that sequentially retrieves information, filters it, and then produces an answer. Such a design makes it difficult to adapt to diverse question types. Moreover, it separates retrieval from reasoning, making it hard for the model to decide when to search, how to refine queries, or when to stop. As a result, the retrieved evidence is often poorly aligned with the question. To address these limitations, we reformulate KB-VQA as a search-agent problem and model the solving process as a multi-step decision-making procedure. At each step, the agent selects one of four actions-Answer, Image Retrieval, Text Retrieval, and Caption-based on its current information state. We further design an automated pipeline to collect multi-step trajectories that record the agent's reasoning process, tool usage, and intermediate decisions. These trajectories are then used as supervision for fine-tuning. Experiments on InfoSeek and E-VQA demonstrate that our method achieves state-of-the-art performance, consistently outperforming prior baselines and confirming the effectiveness of our framework.
>
---
#### [new 090] VDPP: Video Depth Post-Processing for Speed and Scalability
- **分类: cs.CV**

- **简介: 该论文属于视频深度估计任务，解决传统方法适应性差、效率低的问题。提出VDPP框架，通过几何精修提升速度与精度，实现高效实时部署。**

- **链接: [https://arxiv.org/pdf/2604.06665](https://arxiv.org/pdf/2604.06665)**

> **作者:** Daewon Yoon; Injun Baek; Sangyu Han; Yearim Kim; Nojun Kwak
>
> **备注:** 8 pages, 6 figures. Accepted to CVPR 2024 Workshop. Project page: this https URL
>
> **摘要:** Video depth estimation is essential for providing 3D scene structure in applications ranging from autonomous driving to mixed reality. Current end-to-end video depth models have established state-of-the-art performance. Although current end-to-end (E2E) models have achieved state-of-the-art performance, they function as tightly coupled systems that suffer from a significant adaptation lag whenever superior single-image depth estimators are released. To mitigate this issue, post-processing methods such as NVDS offer a modular plug-and-play alternative to incorporate any evolving image depth model without retraining. However, existing post-processing methods still struggle to match the efficiency and practicality of E2E systems due to limited speed, accuracy, and RGB reliance. In this work, we revitalize the role of post-processing by proposing VDPP (Video Depth Post-Processing), a framework that improves the speed and accuracy of post-processing methods for video depth estimation. By shifting the paradigm from computationally expensive scene reconstruction to targeted geometric refinement, VDPP operates purely on geometric refinements in low-resolution space. This design achieves exceptional speed (>43.5 FPS on NVIDIA Jetson Orin Nano) while matching the temporal coherence of E2E systems, with dense residual learning driving geometric representations rather than full reconstructions. Furthermore, our VDPP's RGB-free architecture ensures true scalability, enabling immediate integration with any evolving image depth model. Our results demonstrate that VDPP provides a superior balance of speed, accuracy, and memory efficiency, making it the most practical solution for real-time edge deployment. Our project page is at this https URL
>
---
#### [new 091] Are Face Embeddings Compatible Across Deep Neural Network Models?
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究不同深度神经网络模型在人脸嵌入上的兼容性问题，分析其嵌入空间的几何结构，探索通过线性变换实现模型间对齐的可能性。任务属于人脸识别与模型互操作性研究。**

- **链接: [https://arxiv.org/pdf/2604.07282](https://arxiv.org/pdf/2604.07282)**

> **作者:** Fizza Rubab; Yiying Tong; Arun Ross
>
> **摘要:** Automated face recognition has made rapid strides over the past decade due to the unprecedented rise of deep neural network (DNN) models that can be trained for domain-specific tasks. At the same time, foundation models that are pretrained on broad vision or vision-language tasks have shown impressive generalization across diverse domains, including biometrics. This raises an important question: Do different DNN models--both domain-specific and foundation models--encode facial identity in similar ways, despite being trained on different datasets, loss functions, and architectures? In this regard, we directly analyze the geometric structure of embedding spaces imputed by different DNN models. Treating embeddings of face images as point clouds, we study whether simple affine transformations can align face representations of one model with another. Our findings reveal surprising cross-model compatibility: low-capacity linear mappings substantially improve cross-model face recognition over unaligned baselines for both face identification and verification tasks. Alignment patterns generalize across datasets and vary systematically across model families, indicating representational convergence in facial identity encoding. These findings have implications for model interoperability, ensemble design, and biometric template security.
>
---
#### [new 092] HQF-Net: A Hybrid Quantum-Classical Multi-Scale Fusion Network for Remote Sensing Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感图像语义分割任务，旨在提升模型对细粒度空间细节和高层语义的捕捉能力。提出HQF-Net，结合量子增强模块与多尺度融合机制，有效提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.06715](https://arxiv.org/pdf/2604.06715)**

> **作者:** Md Aminur Hossain; Ayush V. Patel; Siddhant Gole; Sanjay K. Singh; Biplab Banerjee
>
> **备注:** 17 pages
>
> **摘要:** Remote sensing semantic segmentation requires models that can jointly capture fine spatial details and high-level semantic context across complex scenes. While classical encoder-decoder architectures such as U-Net remain strong baselines, they often struggle to fully exploit global semantics and structured feature interactions. In this work, we propose HQF-Net, a hybrid quantum-classical multi-scale fusion network for remote sensing image segmentation. HQF-Net integrates multi-scale semantic guidance from a frozen DINOv3 ViT-L/16 backbone with a customized U-Net architecture through a Deformable Multiscale Cross-Attention Fusion (DMCAF) module. To enhance feature refinement, the framework further introduces quantum-enhanced skip connections (QSkip) and a Quantum bottleneck with Mixture-of-Experts (QMoE), which combines complementary local, global, and directional quantum circuits within an adaptive routing mechanism. Experiments on three remote sensing benchmarks show consistent improvements with the proposed design. HQF-Net achieves 0.8568 mIoU and 96.87% overall accuracy on this http URL, 71.82% mIoU on OpenEarthMap, and 55.28% mIoU with 99.37% overall accuracy on SeasoNet. An architectural ablation study further confirms the contribution of each major component. These results show that structured hybrid quantum-classical feature processing is a promising direction for improving remote sensing semantic segmentation under near-term quantum constraints.
>
---
#### [new 093] Visual prompting reimagined: The power of the Activation Prompts
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉模型微调任务，旨在解决视觉提示（VP）性能不足的问题。通过引入激活提示（AP），提升模型效率与准确率。**

- **链接: [https://arxiv.org/pdf/2604.06440](https://arxiv.org/pdf/2604.06440)**

> **作者:** Yihua Zhang; Hongkang Li; Yuguang Yao; Aochuan Chen; Shuai Zhang; Pin-Yu Chen; Meng Wang; Sijia Liu
>
> **备注:** AISTATS 2026
>
> **摘要:** Visual prompting (VP) has emerged as a popular method to repurpose pretrained vision models for adaptation to downstream tasks. Unlike conventional model fine-tuning techniques, VP introduces a universal perturbation directly into the input data to facilitate task-specific fine-tuning rather than modifying model parameters. However, there exists a noticeable performance gap between VP and conventional fine-tuning methods, highlighting an unexplored realm in theory and practice to understand and advance the input-level VP to reduce its current performance gap. Towards this end, we introduce a generalized concept, termed activation prompt (AP), which extends the scope of the input-level VP by enabling universal perturbations to be applied to activation maps within the intermediate layers of the model. By using AP to revisit the problem of VP and employing it as an analytical tool, we demonstrate the intrinsic limitations of VP in both performance and efficiency, revealing why input-level prompting may lack effectiveness compared to AP, which exhibits a model-dependent layer preference. We show that AP is closely related to normalization tuning in convolutional neural networks and vision transformers, although each model type has distinct layer preferences for prompting. We also theoretically elucidate the rationale behind such a preference by analyzing global features across layers. Through extensive experiments across 29 datasets and various model architectures, we provide a comprehensive performance analysis of AP, comparing it with VP and parameter-efficient fine-tuning baselines. Our results demonstrate AP's superiority in both accuracy and efficiency, considering factors such as time, parameters, memory usage, and throughput.
>
---
#### [new 094] How Well Do Vision-Language Models Understand Sequential Driving Scenes? A Sensitivity Study
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，研究VLM在顺序驾驶场景中的表现。通过构建VENUSS框架，分析输入配置对模型性能的影响，揭示其在动态理解上的不足。**

- **链接: [https://arxiv.org/pdf/2604.06750](https://arxiv.org/pdf/2604.06750)**

> **作者:** Roberto Brusnicki; Mattia Piccinini; Johannes Betz
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Vision-Language Models (VLMs) are increasingly proposed for autonomous driving tasks, yet their performance on sequential driving scenes remains poorly characterized, particularly regarding how input configurations affect their capabilities. We introduce VENUSS (VLM Evaluation oN Understanding Sequential Scenes), a framework for systematic sensitivity analysis of VLM performance on sequential driving scenes, establishing baselines for future research. Building upon existing datasets, VENUSS extracts temporal sequences from driving videos, and generates structured evaluations across custom categories. By comparing 25+ existing VLMs across 2,600+ scenarios, we reveal how even top models achieve only 57% accuracy, not matching human performance in similar constraints (65%) and exposing significant capability gaps. Our analysis shows that VLMs excel with static object detection but struggle with understanding the vehicle dynamics and temporal relations. VENUSS offers the first systematic sensitivity analysis of VLMs focused on how input image configurations - resolution, frame count, temporal intervals, spatial layouts, and presentation modes - affect performance on sequential driving scenes. Supplementary material available at this https URL
>
---
#### [new 095] Enhancing MLLM Spatial Understanding via Active 3D Scene Exploration for Multi-Perspective Reasoning
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型的3D空间理解任务，旨在解决MLLM在复杂3D空间推理中的不足。通过引入视觉思维链和3D重建，提升模型的空间感知能力。**

- **链接: [https://arxiv.org/pdf/2604.06725](https://arxiv.org/pdf/2604.06725)**

> **作者:** Jiahua Chen; Qihong Tang; Weinong Wang; Qi Fan
>
> **摘要:** Although Multimodal Large Language Models have achieved remarkable progress, they still struggle with complex 3D spatial reasoning due to the reliance on 2D visual priors. Existing approaches typically mitigate this limitation either through computationally expensive post-training procedures on limited 3D datasets or through rigid tool-calling mechanisms that lack explicit geometric understanding and viewpoint flexibility. To address these challenges, we propose a \textit{training-free} framework that introduces a Visual Chain-of-Thought mechanism grounded in explicit 3D reconstruction. The proposed pipeline first reconstructs a high-fidelity 3D mesh from a single image using MLLM-guided keyword extraction and mask generation at multiple granularities. Subsequently, the framework leverages an external knowledge base to iteratively compute optimal camera extrinsic parameters and synthesize novel views, thereby emulating human perspective-taking. Extensive experiments demonstrate that the proposed approach significantly enhances spatial comprehension. Specifically, the framework outperforms specialized spatial models and general-purpose MLLMs, including \textit{GPT-5.2} and \textit{Gemini-2.5-Flash}, on major benchmarks such as 3DSRBench and Rel3D.
>
---
#### [new 096] CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale
- **分类: cs.CV**

- **简介: 该论文提出CraterBench-R，解决行星尺度下的陨石坑实例级检索问题，通过自监督ViT和实例-令牌聚合提升效率与准确率。**

- **链接: [https://arxiv.org/pdf/2604.06245](https://arxiv.org/pdf/2604.06245)**

> **作者:** Jichao Fang; Lei Zhang; Michael Phillips; Wei Luo
>
> **备注:** Accepted at the EarthVision 2026 Workshop at CVPR 2026
>
> **摘要:** Impact craters are a cornerstone of planetary surface analysis. However, while most deep learning pipelines treat craters solely as a detection problem, critical scientific workflows such as catalog deduplication, cross-observation matching, and morphological analog discovery are inherently retrieval tasks. To address this, we formulate crater analysis as an instance-level image retrieval problem and introduce CraterBench-R, a curated benchmark featuring about 25,000 crater identities with multi-scale gallery views and manually verified queries spanning diverse scales and contexts. Our baseline evaluations across various architectures reveal that self-supervised Vision Transformers (ViTs), particularly those with in-domain pretraining, dominate the task, outperforming generic models with significantly more parameters. Furthermore, we demonstrate that retaining multiple ViT patch tokens for late-interaction matching dramatically improves accuracy over standard single-vector pooling. However, storing all tokens per image is operationally inefficient at a planetary scale. To close this efficiency gap, we propose instance-token aggregation, a scalable, training-free method that selects K seed tokens, assigns the remaining tokens to these seeds via cosine similarity, and aggregates each cluster into a single representative token. This approach yields substantial gains: at K=16, aggregation improves mAP by 17.9 points over raw token selection, and at K=64, it matches the accuracy of using all 196 tokens with significantly less storage. Finally, we demonstrate that a practical two-stage pipeline, with single-vector shortlisting followed by instance-token reranking, recovers 89-94% of the full late-interaction accuracy while searching only a small candidate set. The benchmark is publicly available at this http URL.
>
---
#### [new 097] MAR-GRPO: Stabilized GRPO for AR-diffusion Hybrid Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决AR-diffusion混合框架中的训练不稳定问题。提出MAR-GRPO框架，通过多轨迹优化和不确定性筛选提升生成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.06966](https://arxiv.org/pdf/2604.06966)**

> **作者:** Xiaoxiao Ma; Jiachen Lei; Tianfei Ren; Jie Huang; Siming Fu; Aiming Hao; Jiahong Wu; Xiangxiang Chu; Feng Zhao
>
> **摘要:** Reinforcement learning (RL) has been successfully applied to autoregressive (AR) and diffusion models. However, extending RL to hybrid AR-diffusion frameworks remains challenging due to interleaved inference and noisy log-probability estimation. In this work, we study masked autoregressive models (MAR) and show that the diffusion head plays a critical role in training dynamics, often introducing noisy gradients that lead to instability and early performance saturation. To address this issue, we propose a stabilized RL framework for MAR. We introduce multi-trajectory expectation (MTE), which estimates the optimization direction by averaging over multiple diffusion trajectories, thereby reducing diffusion-induced gradient noise. To avoid over-smoothing, we further estimate token-wise uncertainty from multiple trajectories and apply multi-trajectory optimization only to the top-k% uncertain tokens. In addition, we introduce a consistency-aware token selection strategy that filters out AR tokens that are less aligned with the final generated content. Extensive experiments across multiple benchmarks demonstrate that our method consistently improves visual quality, training stability, and spatial structure understanding over baseline GRPO and pre-RL models. Code is available at: this https URL.
>
---
#### [new 098] Energy-based Tissue Manifolds for Longitudinal Multiparametric MRI Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于能量的几何框架，用于纵向多参数MRI分析。任务是无需分割或分类，通过能量函数描述组织变化，解决长期影像评估问题。**

- **链接: [https://arxiv.org/pdf/2604.07180](https://arxiv.org/pdf/2604.07180)**

> **作者:** Kartikay Tehlan; Lukas Förner; Nico Schmutzenhofer; Michael Frühwald; Matthias Wagner; Nassir Navab; Thomas Wendler
>
> **备注:** The code is available at this https URL
>
> **摘要:** We propose a geometric framework for longitudinal multi-parametric MRI analysis based on patient-specific energy modelling in sequence space. Rather than operating on images with spatial networks, each voxel is represented by its multi-sequence intensity vector ($T1$, $T1c$, $T2$, FLAIR, ADC), and a compact implicit neural representation is trained via denoising score matching to learn an energy function $E_{\theta}(\mathbf{u})$ over $\mathbb{R}^d$ from a single baseline scan. The learned energy landscape provides a differential-geometric description of tissue regimes without segmentation labels. Local minima define tissue basins, gradient magnitude reflects proximity to regime boundaries, and Laplacian curvature characterises local constraint structure. Importantly, this baseline energy manifold is treated as a fixed geometric reference: it encodes the set of contrast combinations observed at diagnosis and is not retrained at follow-up. Longitudinal assessment is therefore formulated as evaluation of subsequent scans relative to this baseline geometry. Rather than comparing anatomical segmentations, we analyse how the distribution of MRI sequence vectors evolves under the baseline energy function. In a paediatric case with later recurrence, follow-up scans show progressive deviation in energy and directional displacement in sequence space toward the baseline tumour-associated regime before clear radiological reappearance. In a case with stable disease, voxel distributions remain confined to established low-energy basins without systematic drift. The presented cases serve as proof-of-concept that patient-specific energy manifolds can function as geometric reference systems for longitudinal mpMRI analysis without explicit segmentation or supervised classification, providing a foundation for further investigation of manifold-based tissue-at-risk tracking in neuro-oncology.
>
---
#### [new 099] Geo-EVS: Geometry-Conditioned Extrapolative View Synthesis for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于视图合成任务，解决自动驾驶中相机依赖和外轨迹性能下降问题。提出Geo-EVS框架，通过几何约束和缺陷训练提升稀疏视图质量和3D检测效果。**

- **链接: [https://arxiv.org/pdf/2604.07250](https://arxiv.org/pdf/2604.07250)**

> **作者:** Yatong Lan; Rongkui Tang; Lei He
>
> **摘要:** Extrapolative novel view synthesis can reduce camera-rig dependency in autonomous driving by generating standardized virtual views from heterogeneous sensors. Existing methods degrade outside recorded trajectories because extrapolated poses provide weak geometric support and no dense target-view supervision. The key is to explicitly expose the model to out-of-trajectory condition defects during training. We propose Geo-EVS, a geometry-conditioned framework under sparse supervision. Geo-EVS has two components. Geometry-Aware Reprojection (GAR) uses fine-tuned VGGT to reconstruct colored point clouds and reproject them to observed and virtual target poses, producing geometric condition maps. This design unifies the reprojection path between training and inference. Artifact-Guided Latent Diffusion (AGLD) injects reprojection-derived artifact masks during training so the model learns to recover structure under missing support. For evaluation, we use a LiDAR-Projected Sparse-Reference (LPSR) protocol when dense extrapolated-view ground truth is unavailable. On Waymo, Geo-EVS improves sparse-view synthesis quality and geometric accuracy, especially in high-angle and low-coverage settings. It also improves downstream 3D detection.
>
---
#### [new 100] Physical Adversarial Attacks on AI Surveillance Systems:Detection, Tracking, and Visible--Infrared Evasion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI安防系统安全研究，解决物理对抗攻击问题。通过分析时间持续性、传感模态等关键因素，探讨攻击有效性与系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.06865](https://arxiv.org/pdf/2604.06865)**

> **作者:** Miguel A.DelaCruz; Patricia Mae Santos; Rafael T.Navarro
>
> **摘要:** Physical adversarial attacks are increasingly studied in settings that resemble deployed surveillance systems rather than isolated image benchmarks. In these settings, person detection, multi-object tracking, visible--infrared sensing, and the practical form of the attack carrier all matter at once. This changes how the literature should be read. A perturbation that suppresses a detector in one frame may have limited practical effect if identity is recovered over time; an RGB-only result may say little about night-time systems that rely on visible and thermal inputs together; and a conspicuous patch can imply a different threat model from a wearable or selectively activated carrier. This paper reviews physical attacks from that surveillance-oriented viewpoint. Rather than attempting a complete catalogue of all physical attacks in computer vision, we focus on the technical questions that become central in surveillance: temporal persistence, sensing modality, carrier realism, and system-level objective. We organize prior work through a four-part taxonomy and discuss how recent results on multi-object tracking, dual-modal visible--infrared evasion, and controllable clothing reflect a broader change in the field. We also summarize evaluation practices and unresolved gaps, including distance robustness, camera-pipeline variation, identity-level metrics, and activation-aware testing. The resulting picture is that surveillance robustness cannot be judged reliably from isolated per-frame benchmarks alone; it has to be examined as a system problem unfolding over time, across sensors, and under realistic physical deployment constraints.
>
---
#### [new 101] FlowInOne:Unifying Multimodal Generation as Image-in, Image-out Flow Matching
- **分类: cs.CV**

- **简介: 该论文提出FlowInOne，将多模态生成统一为视觉流匹配任务，解决跨模态对齐与任务分离问题，通过视觉提示实现图像生成与编辑。**

- **链接: [https://arxiv.org/pdf/2604.06757](https://arxiv.org/pdf/2604.06757)**

> **作者:** Junchao Yi; Rui Zhao; Jiahao Tang; Weixian Lei; Linjie Li; Qisheng Su; Zhengyuan Yang; Lijuan Wang; Xiaofeng Zhu; Alex Jinpeng Wang
>
> **摘要:** Multimodal generation has long been dominated by text-driven pipelines where language dictates vision but cannot reason or create within it. We challenge this paradigm by asking whether all modalities, including textual descriptions, spatial layouts, and editing instructions, can be unified into a single visual representation. We present FlowInOne, a framework that reformulates multimodal generation as a purely visual flow, converting all inputs into visual prompts and enabling a clean image-in, image-out pipeline governed by a single flow matching model. This vision-centric formulation naturally eliminates cross-modal alignment bottlenecks, noise scheduling, and task-specific architectural branches, unifying text-to-image generation, layout-guided editing, and visual instruction following under one coherent paradigm. To support this, we introduce VisPrompt-5M, a large-scale dataset of 5 million visual prompt pairs spanning diverse tasks including physics-aware force dynamics and trajectory prediction, alongside VP-Bench, a rigorously curated benchmark assessing instruction faithfulness, spatial precision, visual realism, and content consistency. Extensive experiments demonstrate that FlowInOne achieves state-of-the-art performance across all unified generation tasks, surpassing both open-source models and competitive commercial systems, establishing a new foundation for fully vision-centric generative modeling where perception and creation coexist within a single continuous visual space.
>
---
#### [new 102] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于联邦学习任务，旨在解决系统和统计异构性问题。提出SubFLOT框架，通过最优传输和自适应正则化实现高效个性化模型压缩。**

- **链接: [https://arxiv.org/pdf/2604.06631](https://arxiv.org/pdf/2604.06631)**

> **作者:** Zheng Jiang; Nan He; Yiming Chen; Lifeng Sun
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Federated Learning (FL) enables collaborative model training while preserving data privacy, but its practical deployment is hampered by system and statistical heterogeneity. While federated network pruning offers a path to mitigate these issues, existing methods face a critical dilemma: server-side pruning lacks personalization, whereas client-side pruning is computationally prohibitive for resource-constrained devices. Furthermore, the pruning process itself induces significant parametric divergence among heterogeneous submodels, destabilizing training and hindering global convergence. To address these challenges, we propose SubFLOT, a novel framework for server-side personalized federated pruning. SubFLOT introduces an Optimal Transport-enhanced Pruning (OTP) module that treats historical client models as proxies for local data distributions, formulating the pruning task as a Wasserstein distance minimization problem to generate customized submodels without accessing raw data. Concurrently, to counteract parametric divergence, our Scaling-based Adaptive Regularization (SAR) module adaptively penalizes a submodel's deviation from the global model, with the penalty's strength scaled by the client's pruning rate. Comprehensive experiments demonstrate that SubFLOT consistently and substantially outperforms state-of-the-art methods, underscoring its potential for deploying efficient and personalized models on resource-constrained edge devices.
>
---
#### [new 103] MedRoute: RL-Based Dynamic Specialist Routing in Multi-Agent Medical Diagnosis
- **分类: eess.IV; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于医疗诊断任务，解决多专科协作诊断问题。提出MedRoute框架，通过动态选择专家模型提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2604.06180](https://arxiv.org/pdf/2604.06180)**

> **作者:** Ashmal Vayani; Parth Parag Kulkarni; Joseph Fioresi; Song Wang; Mubarak Shah
>
> **摘要:** Medical diagnosis using Large Multimodal Models (LMMs) has gained increasing attention due to capability of these models in providing precise diagnoses. These models generally combine medical questions with visual inputs to generate diagnoses or treatments. However, they are often overly general and unsuitable under the wide range of medical conditions in real-world healthcare. In clinical practice, diagnosis is performed by multiple specialists, each contributing domain-specific expertise. To emulate this process, a potential solution is to deploy a dynamic multi-agent LMM framework, where each agent functions as a medical specialist. Current approaches in this emerging area, typically relying on static or predefined selection of various specialists, cannot be adapted to the changing practical scenario. In this paper, we propose MedRoute, a flexible and dynamic multi-agent framework that comprises of a collaborative system of specialist LMM agents. Furthermore, we add a General Practitioner with an RL-trained router for dynamic specialist selection, and a Moderator that produces the final decision. In this way, our framework closely mirrors real clinical workflows. Extensive evaluations on text and image-based medical datasets demonstrate improved diagnostic accuracy, outperforming the state-of-the-art baselines. Our work lays a strong foundation for future research. Code and models are available at this https URL.
>
---
#### [new 104] ProofSketcher: Hybrid LLM + Lightweight Proof Checker for Reliable Math/Logic Reasoning
- **分类: cs.AI; cs.CE; cs.CV; cs.LG**

- **简介: 该论文提出一种混合系统，结合大语言模型与轻量证明检查器，解决数学逻辑推理中的可靠性问题。通过生成类型化证明草图并扩展为明确的证明义务，提升推理的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.06401](https://arxiv.org/pdf/2604.06401)**

> **作者:** Kranthi Kommuru; Kunal Khanvilkar; Gaurav Parekh
>
> **摘要:** The large language models (LLMs) might produce a persuasive argument within mathematical and logical fields, although such argument often includes some minor missteps, including the entire omission of side conditions, invalid inference patterns, or appeals to a lemma that cannot be derived logically out of the context being discussed. These omissions are infamously hard to notice solely out of the text, as even the misconstrued construction still may seem mostly accurate. Conversely, interactive theorem provers like Lean and Coq have rigorous reliability by ensuring that syntactic and semantic statements only accept statements that can pass all the syntactic and semantic steps in the program which is a small trusted kernel of the language type-checks with. Despite the fact that this technique provides strong guarantees, it comes at quite a heavy price: the evidence must be completely formalized, and the evidence user or a auxiliary search program must provide an avalanche of low-level information. This paper presents a hybrid pipeline where an LLM generates a typed proof sketch in a compact DSL and a lightweight trusted kernel expands the sketch into explicit proof obligations.
>
---
#### [new 105] Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言模型安全任务，旨在解决恶意提示导致的不安全内容生成问题。提出HyPE和HyPS方法，实现高效、可解释的有害提示检测与净化。**

- **链接: [https://arxiv.org/pdf/2604.06285](https://arxiv.org/pdf/2604.06285)**

> **作者:** Igor Maljkovic; Maria Rosaria Briglia; Iacopo Masi; Antonio Emanuele Cinà; Fabio Roli
>
> **备注:** Paper accepted at ICLR 2026. Webpage available at: this https URL
>
> **摘要:** Vision-Language Models (VLMs) have become essential for tasks such as image synthesis, captioning, and retrieval by aligning textual and visual information in a shared embedding space. Yet, this flexibility also makes them vulnerable to malicious prompts designed to produce unsafe content, raising critical safety concerns. Existing defenses either rely on blacklist filters, which are easily circumvented, or on heavy classifier-based systems, both of which are costly and fragile under embedding-level attacks. We address these challenges with two complementary components: Hyperbolic Prompt Espial (HyPE) and Hyperbolic Prompt Sanitization (HyPS). HyPE is a lightweight anomaly detector that leverages the structured geometry of hyperbolic space to model benign prompts and detect harmful ones as outliers. HyPS builds on this detection by applying explainable attribution methods to identify and selectively modify harmful words, neutralizing unsafe intent while preserving the original semantics of user prompts. Through extensive experiments across multiple datasets and adversarial scenarios, we prove that our framework consistently outperforms prior defenses in both detection accuracy and robustness. Together, HyPE and HyPS offer an efficient, interpretable, and resilient approach to safeguarding VLMs against malicious prompt misuse.
>
---
#### [new 106] TurPy: a physics-based and differentiable optical turbulence simulator for algorithmic development and system optimization
- **分类: physics.optics; cs.CV**

- **简介: 该论文提出TurPy，一个基于物理的可微光学湍流模拟器，用于算法开发与系统优化。解决湍流对光传播影响的高精度模拟问题，通过参数化方法支持多种传播环境，提升光学系统设计效率。**

- **链接: [https://arxiv.org/pdf/2604.07248](https://arxiv.org/pdf/2604.07248)**

> **作者:** Joseph L. Greene; Alfred Moore; Iris Ochoa; Emily Kwan; Patrick Marano; Christopher R. Valenta
>
> **备注:** 19 pages, 7 figures, 1 table. Presented at 2026 SPIE DS Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques, and Applications IV
>
> **摘要:** Developing optical systems for free-space applications requires simulation tools that accurately capture turbulence-induced wavefront distortions and support gradient-based optimization. Here we introduce TurPy, a GPU-accelerated, fully differentiable wave optics turbulence simulator to bridge high fidelity simulation with end-to-end optical system design. TurPy incorporates subharmonic phase screen generation, autoregressive temporal evolution, and an automated screen placement routine balancing Fourier aliasing constraints and weak-turbulence approximations into a unified, user-ready framework. Because TurPy's phase screen generation is parameterized through a media-specific power spectral density, the framework extends to atmospheric, oceanic, and biological propagation environments with minimal modification. We validate TurPy against established atmospheric turbulence theory by matching 2nd order Gaussian beam broadening and 4th order plane wave scintillation to closed-form models with 98% accuracy across weak to strong turbulence regimes, requiring only the medium's refractive index structure constant and power spectral density as inputs. To demonstrate TurPy as a gradient-based training platform, we optimize a dual-domain diffractive deep neural network (D2NN) in a two-mask dual-domain architecture to recover a Gaussian beam from a weakly turbulent path and achieving over 20x reduction in scintillation relative to an uncompensated receiver in simulation. TurPy is released as an open-source package to support synthetic data generation, turbulence-informed algorithm development, and the end-to-end design of optical platforms operating in turbulent environments.
>
---
#### [new 107] Structural Regularities of Cinema SDR-to-HDR Mapping in a Controlled Mastering Workflow: A Pixel-wise Case Study on ASC StEM2
- **分类: eess.IV; cs.CV**

- **简介: 该论文研究电影SDR到HDR映射的结构规律，分析ASC StEM2数据集中的像素级特征，旨在为HDR转换提供定量基准与方法参考。**

- **链接: [https://arxiv.org/pdf/2604.06276](https://arxiv.org/pdf/2604.06276)**

> **作者:** Xin Zhang; Xiaoyi Chen
>
> **备注:** 15 pages, 6 figures. Empirical case study on cinema SDR-to-HDR mapping using ASC StEM2
>
> **摘要:** We present an empirical case study of cinema SDR-to-HDR mapping using ASC StEM2, a rare common-source dataset containing EXR scene-referred images and matched SDR/HDR cinema release masters from the same ACES-based mastering workflow. Based on pixel-wise statistics over all 18,580 frames of the test film, we construct a three-domain comparison involving EXR source data, SDR release masters, and HDR release masters to characterize their luminance and color structural relationships within this controlled workflow. In the luminance dimension, SDR and HDR masters exhibit a highly stable global monotonic correspondence, with geometric structure remaining largely consistent overall; sparse and structured deviations appear in self-luminous highlights and specific material regions. In the color dimension, the two masters remain largely consistent in hue, with saturation exhibiting a redistribution pattern of shadow suppression, midtone expansion, and highlight convergence. Using EXR as a scene-referred anchor, we further define a pixel-level decision map that operationally separates EXR-closer recovery regions from content-adaptive adjustment regions. Under this operational definition, 82.4% of sampled image regions are classified as EXR-closer recovery, while the remainder require localized adaptive adjustment. Rather than claiming a universal law for all cinema mastering pipelines, the study provides an interpretable quantitative baseline for structure-aware SDR-to-HDR analysis and for designing learning-based models under shared-source mastering conditions.
>
---
#### [new 108] Adaptive Differential Privacy for Federated Medical Image Segmentation Across Diverse Modalities
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决联邦学习中隐私与模型性能的平衡问题。提出ADP-FL框架，动态调整隐私机制，提升分割精度和稳定性。**

- **链接: [https://arxiv.org/pdf/2604.06518](https://arxiv.org/pdf/2604.06518)**

> **作者:** Puja Saha; Eranga Ukwatta
>
> **备注:** 10 pages, 8 figures. Accepted in SPIE Medical Imaging 2026. Recipient of CAD Best Paper Award: 1st Place, and Robert F. Wagner All-Conference Best Paper Award: Finalist
>
> **摘要:** Large volumes of medical data remain underutilized because centralizing distributed data is often infeasible due to strict privacy regulations and institutional constraints. In addition, models trained in centralized settings frequently fail to generalize across clinical sites because of heterogeneity in imaging protocols and continuously evolving data distributions arising from differences in scanners, acquisition parameters, and patient populations. Federated learning offers a promising solution by enabling collaborative model training without sharing raw data. However, incorporating differential privacy into federated learning, while essential for privacy guarantees, often leads to degraded accuracy, unstable convergence, and reduced generalization. In this work, we propose an adaptive differentially private federated learning (ADP-FL) framework for medical image segmentation that dynamically adjusts privacy mechanisms to better balance the privacy-utility trade-off. The proposed approach stabilizes training, significantly improves Dice scores and segmentation boundary quality, and maintains rigorous privacy guarantees. We evaluated ADP-FL across diverse imaging modalities and segmentation tasks, including skin lesion segmentation in dermoscopic images, kidney tumor segmentation in 3D CT scans, and brain tumor segmentation in multi-parametric MRI. Compared with conventional federated learning and standard differentially private federated learning, ADP-FL consistently achieves higher accuracy, improved boundary delineation, faster convergence, and greater training stability, with performance approaching that of non-private federated learning under the same privacy budgets. These results demonstrate the practical viability of ADP-FL for high-performance, privacy-preserving medical image segmentation in real-world federated settings.
>
---
#### [new 109] Bi-Level Optimization for Single Domain Generalization
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究单域泛化（SDG）任务，旨在提升模型在无目标数据情况下跨域的泛化能力。提出BiSDG框架，通过双层优化分离任务学习与领域建模，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.06349](https://arxiv.org/pdf/2604.06349)**

> **作者:** Marzi Heidari; Hanping Zhang; Hao Yan; Yuhong Guo
>
> **备注:** CVPR Findings Track, 2026
>
> **摘要:** Generalizing from a single labeled source domain to unseen target domains, without access to any target data during training, remains a fundamental challenge in robust machine learning. We address this underexplored setting, known as Single Domain Generalization (SDG), by proposing BiSDG, a bi-level optimization framework that explicitly decouples task learning from domain modeling. BiSDG simulates distribution shifts through surrogate domains constructed via label-preserving transformations of the source data. To capture domain-specific context, we propose a domain prompt encoder that generates lightweight modulation signals to produce augmenting features via feature-wise linear modulation. The learning process is formulated as a bi-level optimization problem: the inner objective optimizes task performance under fixed prompts, while the outer objective maximizes generalization across the surrogate domains by updating the domain prompt encoder. We further develop a practical gradient approximation scheme that enables efficient bi-level training without second-order derivatives. Extensive experiments on various SGD benchmarks demonstrate that BiSDG consistently outperforms prior methods, setting new state-of-the-art performance in the SDG setting.
>
---
#### [new 110] Euclid Quick Data Release (Q1). AgileLens: A scalable CNN-based pipeline for strong gravitational lens identification
- **分类: astro-ph.GA; cs.CV**

- **简介: 该论文属于强引力透镜识别任务，旨在高效筛选Euclid Q1数据中的透镜系统。通过构建CNN管道，提升识别准确率与效率。**

- **链接: [https://arxiv.org/pdf/2604.06648](https://arxiv.org/pdf/2604.06648)**

> **作者:** Euclid Collaboration; X. Xu; R. Chen; T. Li; A. R. Cooray; S. Schuldt; J. A. Acevedo Barroso; D. Stern; D. Scott; M. Meneghetti; G. Despali; J. Chopra; Y. Cao; M. Cheng; J. Buda; J. Zhang; J. Furumizo; R. Valencia; Z. Jiang; C. Tortora; N. E. P. Lines; T. E. Collett; S. Fotopoulou; A. Galan; A. Manjón-García; R. Gavazzi; L. Iwamoto; S. Kruk; M. Millon; P. Nugent; C. Saulder; D. Sluse; J. Wilde; M. Walmsley; F. Courbin; R. B. Metcalf; B. Altieri; A. Amara; S. Andreon; N. Auricchio; C. Baccigalupi; M. Baldi; A. Balestra; S. Bardelli; P. Battaglia; R. Bender; A. Biviano; E. Branchini; M. Brescia; S. Camera; V. Capobianco; C. Carbone; V. F. Cardone; J. Carretero; S. Casas; M. Castellano; G. Castignani; S. Cavuoti; A. Cimatti; C. Colodro-Conde; G. Congedo; C. J. Conselice; L. Conversi; Y. Copin; H. M. Courtois; M. Cropper; A. Da Silva; H. Degaudenzi; G. De Lucia; C. Dolding; H. Dole; F. Dubath; X. Dupac; S. Dusini; S. Escoffier; M. Farina; R. Farinelli; S. Farrens; S. Ferriol; F. Finelli; P. Fosalba; M. Frailis; E. Franceschi; M. Fumana; S. Galeotta; K. George; W. Gillard; B. Gillis; C. Giocoli; P. Gómez-Alvarez; J. Gracia-Carpio; A. Grazian; F. Grupp; S. V. H. Haugan; W. Holmes; F. Hormuth; A. Hornstrup; K. Jahnke; M. Jhabvala; B. Joachimi
>
> **备注:** 30 pages, 16 figures
>
> **摘要:** We present an end-to-end, iterative pipeline for efficient identification of strong galaxy--galaxy lensing systems, applied to the Euclid Q1 imaging data. Starting from VIS catalogues, we reject point sources, apply a magnitude cut (I$_E$ $\leq$ 24) on deflectors, and run a pixel-level artefact/noise filter to build 96 $\times$ 96 pix cutouts; VIS+NISP colour composites are constructed with a VIS-anchored luminance scheme that preserves VIS morphology and NISP colour contrast. A VIS-only seed classifier supplies clear positives and typical impostors, from which we curate a morphology-balanced negative set and augment scarce positives. Among the six CNNs studied initially, a modified VGG16 (GlobalAveragePooling + 256/128 dense layers with the last nine layers trainable) performs best; the training set grows from 27 seed lenses (augmented to 1809) plus 2000 negatives to a colour dataset of 30,686 images. After three rounds of iterative fine-tuning, human grading of the top 4000 candidates ranked by the final model yields 441 Grade A/B candidate lensing systems, including 311 overlapping with the existing Q1 strong-lens catalogue, and 130 additional A/B candidates (9 As and 121 Bs) not previously reported. Independently, the model recovers 740 out of 905 (81.8%) candidate Q1 lenses within its top 20,000 predictions, considering off-centred samples. Candidates span I$_E$ $\simeq$ 17--24 AB mag (median 21.3 AB mag) and are redder in Y$_E$--H$_E$ than the parent population, consistent with massive early-type deflectors. Each training iteration required a week for a small team, and the approach easily scales to future Euclid releases; future work will calibrate the selection function via lens injection, extend recall through uncertainty-aware active learning, explore multi-scale or attention-based neural networks with fast post-hoc vetters that incorporate lens models into the classification.
>
---
#### [new 111] 4D Vessel Reconstruction for Benchtop Thrombectomy Analysis
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文属于医学影像分析任务，旨在解决血栓切除术中血管运动测量不足的问题。通过4D重建技术，实现血管位移和应力的定量评估。**

- **链接: [https://arxiv.org/pdf/2604.06671](https://arxiv.org/pdf/2604.06671)**

> **作者:** Ethan Nguyen; Javier Carmona; Arisa Matsuzaki; Naoki Kaneko; Katsushi Arisaka
>
> **备注:** 20 pages, 10 figures, 1 table, supplementary material (3 tables, 3 figures, and 11 videos). Project page: this https URL
>
> **摘要:** Introduction: Mechanical thrombectomy can cause vessel deformation and procedure-related injury. Benchtop models are widely used for device testing, but time-resolved, full-field 3D vessel-motion measurements remain limited. Methods: We developed a nine-camera, low-cost multi-view workflow for benchtop thrombectomy in silicone middle cerebral artery phantoms (2160p, 20 fps). Multi-view videos were calibrated, segmented, and reconstructed with 4D Gaussian Splatting. Reconstructed point clouds were converted to fixed-connectivity edge graphs for region-of-interest (ROI) displacement tracking and a relative surface-based stress proxy. Stress-proxy values were derived from edge stretch using a Neo-Hookean mapping and reported as comparative surface metrics. A synthetic Blender pipeline with known deformation provided geometric and temporal validation. Results: In synthetic bulk translation, the stress proxy remained near zero for most edges (median $\approx$ 0 MPa; 90th percentile 0.028 MPa), with sparse outliers. In synthetic pulling (1-5 mm), reconstruction showed close geometric and temporal agreement with ground truth, with symmetric Chamfer distance of 1.714-1.815 mm and precision of 0.964-0.972 at $\tau = 1$ mm. In preliminary benchtop comparative trials (one trial per condition), cervical aspiration catheter placement showed higher max-median ROI displacement and stress-proxy values than internal carotid artery terminus placement. Conclusion: The proposed protocol provides standardized, time-resolved surface kinematics and comparative relative displacement and stress proxy measurements for thrombectomy benchtop studies. The framework supports condition-to-condition comparisons and methods validation, while remaining distinct from absolute wall-stress estimation. Implementation code and example data are available at this https URL
>
---
#### [new 112] When to Call an Apple Red: Humans Follow Introspective Rules, VLMs Don't
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于AI可信部署研究，探讨VLMs与人类在颜色判断上的差异。通过构建GCA数据集，分析模型是否遵循自身推理规则，发现模型常违背自身规则，而人类则更忠实。**

- **链接: [https://arxiv.org/pdf/2604.06422](https://arxiv.org/pdf/2604.06422)**

> **作者:** Jonathan Nemitz; Carsten Eickhoff; Junyi Jessy Li; Kyle Mahowald; Michal Golovanevsky; William Rudman
>
> **摘要:** Understanding when Vision-Language Models (VLMs) will behave unexpectedly, whether models can reliably predict their own behavior, and if models adhere to their introspective reasoning are central challenges for trustworthy deployment. To study this, we introduce the Graded Color Attribution (GCA) dataset, a controlled benchmark designed to elicit decision rules and evaluate participant faithfulness to these rules. GCA consists of line drawings that vary pixel-level color coverage across three conditions: world-knowledge recolorings, counterfactual recolorings, and shapes with no color priors. Using GCA, both VLMs and human participants establish a threshold: the minimum percentage of pixels of a given color an object must have to receive that color label. We then compare these rules with their subsequent color attribution decisions. Our findings reveal that models systematically violate their own introspective rules. For example, GPT-5-mini violates its stated introspection rules in nearly 60\% of cases on objects with strong color priors. Human participants remain faithful to their stated rules, with any apparent violations being explained by a well-documented tendency to overestimate color coverage. In contrast, we find that VLMs are excellent estimators of color coverage, yet blatantly contradict their own reasoning in their final responses. Across all models and strategies for eliciting introspective rules, world-knowledge priors systematically degrade faithfulness in ways that do not mirror human cognition. Our findings challenge the view that VLM reasoning failures are difficulty-driven and suggest that VLM introspective self-knowledge is miscalibrated, with direct implications for high-stakes deployment.
>
---
#### [new 113] XR-CareerAssist: An Immersive Platform for Personalised Career Guidance Leveraging Extended Reality and Multimodal AI
- **分类: cs.CE; cs.AI; cs.CV; cs.CY; cs.ET**

- **简介: 论文提出XR-CareerAssist，融合XR与AI技术，解决传统职业指导平台互动性差、缺乏个性化的问题，通过多模态交互提供沉浸式职业建议。**

- **链接: [https://arxiv.org/pdf/2604.06901](https://arxiv.org/pdf/2604.06901)**

> **作者:** N.D. Tantaroudas; A.J. McCracken; I. Karachalios; E. Papatheou; V. Pastrikakis
>
> **备注:** 21
>
> **摘要:** Conventional career guidance platforms rely on static, text-driven interfaces that struggle to engage users or deliver personalised, evidence-based insights. Although Computer-Assisted Career Guidance Systems have evolved since the 1960s, they remain limited in interactivity and pay little attention to the narrative dimensions of career development. We introduce XR-CareerAssist, a platform that unifies Extended Reality (XR) with several Artificial Intelligence (AI) modules to deliver immersive, multilingual career guidance. The system integrates Automatic Speech Recognition for voice-driven interaction, Neural Machine Translation across English, Greek, French, and Italian, a Langchain-based conversational Training Assistant for personalised dialogue, a BLIP-based Vision-Language model for career visualisations, and AWS Polly Text-to-Speech delivered through an interactive 3D avatar. Career trajectories are rendered as dynamic Sankey diagrams derived from a repository of more than 100,000 anonymised professional profiles. The application was built in Unity for Meta Quest 3, with backend services hosted on AWS. A pilot evaluation at the University of Exeter with 23 participants returned 95.6% speech recognition accuracy, 78.3% overall user satisfaction, and 91.3% favourable ratings for system responsiveness, with feedback informing subsequent improvements to motion comfort, audio clarity, and text legibility. XR-CareerAssist demonstrates how the fusion of XR and AI can produce more engaging, accessible, and effective career development tools, with the integration of five AI modules within a single immersive environment yielding a multimodal interaction experience that distinguishes it from existing career guidance platforms.
>
---
#### [new 114] A Noise Constrained Diffusion (NC-Diffusion) Framework for High Fidelity Image Compression
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决扩散模型压缩中因噪声导致的重建偏差问题。提出NC-Diffusion框架，通过约束噪声提升压缩质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.06568](https://arxiv.org/pdf/2604.06568)**

> **作者:** Zhenyu Du; Yanbo Gao; Shuai Li; Yiyang Li; Hui Yuan; Mao Ye
>
> **备注:** Accepted by IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
>
> **摘要:** With the great success of diffusion models in image generation, diffusion-based image compression is attracting increasing interests. However, due to the random noise introduced in the diffusion learning, they usually produce reconstructions with deviation from the original images, leading to suboptimal compression results. To address this problem, in this paper, we propose a Noise Constrained Diffusion (NC-Diffusion) framework for high fidelity image compression. Unlike existing diffusion-based compression methods that add random Gaussian noise and direct the noise into the image space, the proposed NC-Diffusion formulates the quantization noise originally added in the learned image compression as the noise in the forward process of diffusion. Then a noise constrained diffusion process is constructed from the ground-truth image to the initial compression result generated with quantization noise. The NC-Diffusion overcomes the problem of noise mismatch between compression and diffusion, significantly improving the inference efficiency. In addition, an adaptive frequency-domain filtering module is developed to enhance the skip connections in the U-Net based diffusion architecture, in order to enhance high-frequency details. Moreover, a zero-shot sample-guided enhancement method is designed to further improve the fidelity of the image. Experiments on multiple benchmark datasets demonstrate that our method can achieve the best performance compared with existing methods.
>
---
#### [new 115] FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于文本到图像生成的对齐任务，解决大规模扩散模型训练效率低的问题。通过FP4量化与BF16优化结合，提升训练速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.06916](https://arxiv.org/pdf/2604.06916)**

> **作者:** Yitong Li; Junsong Chen; Shuchen Xue; Pengcuo Zeren; Siyuan Fu; Dinghao Yang; Yangyang Tang; Junjie Bai; Ping Luo; Song Han; Enze Xie
>
> **摘要:** Reinforcement-Learning-based post-training has recently emerged as a promising paradigm for aligning text-to-image diffusion models with human preferences. In recent studies, increasing the rollout group size yields pronounced performance improvements, indicating substantial room for further alignment gains. However, scaling rollouts on large-scale foundational diffusion models (e.g., FLUX.1-12B) imposes a heavy computational burden. To alleviate this bottleneck, we explore the integration of FP4 quantization into Diffusion RL rollouts. Yet, we identify that naive quantized pipelines inherently introduce risks of performance degradation. To overcome this dilemma between efficiency and training integrity, we propose Sol-RL (Speed-of-light RL), a novel FP4-empowered Two-stage Reinforcement Learning framework. First, we utilize high-throughput NVFP4 rollouts to generate a massive candidate pool and extract a highly contrastive subset. Second, we regenerate these selected samples in BF16 precision and optimize the policy exclusively on them. By decoupling candidate exploration from policy optimization, Sol-RL integrates the algorithmic mechanisms of rollout scaling with the system-level throughput gains of NVFP4. This synergistic algorithm-hardware design effectively accelerates the rollout phase while reserving high-fidelity samples for optimization. We empirically demonstrate that our framework maintains the training integrity of BF16 precision pipeline while fully exploiting the throughput gains enabled by FP4 arithmetic. Extensive experiments across SANA, FLUX.1, and SD3.5-L substantiate that our approach delivers superior alignment performance across multiple metrics while accelerating training convergence by up to $4.64\times$, unlocking the power of massive rollout scaling at a fraction of the cost.
>
---
#### [new 116] Drifting Fields are not Conservative
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型领域，研究漂移场是否为保守场。工作表明漂移场通常不可表示为标量梯度，提出新归一化方法恢复保守性，但实际收益有限，建议使用简单损失函数训练模型。**

- **链接: [https://arxiv.org/pdf/2604.06333](https://arxiv.org/pdf/2604.06333)**

> **作者:** Leonard Franz; Sebastian Hoffmann; Georg Martius
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Drifting models generate high-quality samples in a single forward pass by transporting generated samples toward the data distribution using a vector valued drift field. We investigate whether this procedure is equivalent to optimizing a scalar loss and find that, in general, it is not: drift fields are not conservative - they cannot be written as the gradient of any scalar potential. We identify the position-dependent normalization as the source of non-conservatism. The Gaussian kernel is the unique exception where the normalization is harmless and the drift field is exactly the gradient of a scalar function. Generalizing this, we propose an alternative normalization via a related kernel (the sharp kernel) which restores conservatism for any radial kernel, yielding well-defined loss functions for training drifting models. While we identify that the drifting field matching objective is strictly more general than loss minimization, as it can implement non-conservative transport fields that no scalar loss can reproduce, we observe that practical gains obtained utilizing this flexibility are minimal. We thus propose to train drifting models with the conceptually simpler formulations utilizing loss functions.
>
---
#### [new 117] KITE: Keyframe-Indexed Tokenized Evidence for VLM-Based Robot Failure Analysis
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出KITE，用于机器人故障分析的视觉-语言模型前端。解决长视频处理与故障解释问题，通过关键帧和布局信息生成可解释的提示。**

- **链接: [https://arxiv.org/pdf/2604.07034](https://arxiv.org/pdf/2604.07034)**

> **作者:** Mehdi Hosseinzadeh; King Hang Wong; Feras Dayoub
>
> **备注:** ICRA 2026; Project page: this https URL
>
> **摘要:** We present KITE, a training-free, keyframe-anchored, layout-grounded front-end that converts long robot-execution videos into compact, interpretable tokenized evidence for vision-language models (VLMs). KITE distills each trajectory into a small set of motion-salient keyframes with open-vocabulary detections and pairs each keyframe with a schematic bird's-eye-view (BEV) representation that encodes relative object layout, axes, timestamps, and detection confidence. These visual cues are serialized with robot-profile and scene-context tokens into a unified prompt, allowing the same front-end to support failure detection, identification, localization, explanation, and correction with an off-the-shelf VLM. On the RoboFAC benchmark, KITE with Qwen2.5-VL substantially improves over vanilla Qwen2.5-VL in the training-free setting, with especially large gains on simulation failure detection, identification, and localization, while remaining competitive with a RoboFAC-tuned baseline. A small QLoRA fine-tune further improves explanation and correction quality. We also report qualitative results on real dual-arm robots, demonstrating the practical applicability of KITE as a structured and interpretable front-end for robot failure analysis. Code and models are released on our project page: this https URL
>
---
#### [new 118] SE-Enhanced ViT and BiLSTM-Based Intrusion Detection for Secure IIoT and IoMT Environments
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文属于入侵检测任务，旨在提高IIoT和IoMT环境中的安全检测精度与效率，通过SE-ViT-BiLSTM模型实现更准确的威胁识别。**

- **链接: [https://arxiv.org/pdf/2604.06254](https://arxiv.org/pdf/2604.06254)**

> **作者:** Afrah Gueriani; Hamza Kheddar; Ahmed Cherif Mazari; Seref Sagiroglu; Onur Ceran
>
> **摘要:** With the rapid growth of interconnected devices in Industrial and Medical Internet of Things (IIoT and MIoT) ecosystems, ensuring timely and accurate detection of cyber threats has become a critical challenge. This study presents an advanced intrusion detection framework based on a hybrid Squeeze-and-Excitation Attention Vision Transformer-Bidirectional Long Short-Term Memory (SE ViT-BiLSTM) architecture. In this design, the traditional multi-head attention mechanism of the Vision Transformer is replaced with Squeeze-and-Excitation attention, and integrated with BiLSTM layers to enhance detection accuracy and computational efficiency. The proposed model was trained and evaluated on two real-world benchmark datasets; EdgeIIoT and CICIoMT2024; both before and after data balancing using the Synthetic Minority Over-sampling Technique (SMOTE) and RandomOverSampler. Experimental results demonstrate that the SE ViT-BiLSTM model outperforms existing approaches across multiple metrics. Before balancing, the model achieved accuracies of 99.11% (FPR: 0.0013%, latency: 0.00032 sec/inst) on EdgeIIoT and 96.10% (FPR: 0.0036%, latency: 0.00053 sec/inst) on CICIoMT2024. After balancing, performance further improved, reaching 99.33% accuracy with 0.00035 sec/inst latency on EdgeIIoT and 98.16% accuracy with 0.00014 sec/inst latency on CICIoMT2024.
>
---
#### [new 119] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment
- **分类: cs.IR; cs.CV**

- **简介: 该论文属于多模态到文本的检索任务，旨在解决多模态查询与文本语料不匹配的问题。通过构建BRIDGE系统，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.07201](https://arxiv.org/pdf/2604.07201)**

> **作者:** Mohamed Darwish Mounis; Mohamed Mahmoud; Shaimaa Sedek; Mahmoud Abdalla; Mahmoud SalahEldin Kasem; Abdelrahman Abdallah; Hyun-Soo Kang
>
> **备注:** Accepted at CVPR 2026 Workshop GRAIL-V
>
> **摘要:** Multimodal retrieval systems struggle to resolve image-text queries against text-only corpora: the best vision-language encoder achieves only 27.6 nDCG@10 on MM-BRIGHT, underperforming strong text-only retrievers. We argue the bottleneck is not the retriever but the query -- raw multimodal queries entangle visual descriptions, conversational noise, and retrieval intent in ways that systematically degrade embedding similarity. We present \textbf{BRIDGE}, a two-component system that resolves this mismatch without multimodal encoders. \textbf{FORGE} (\textbf{F}ocused Retrieval Query Generato\textbf{r}) is a query alignment model trained via reinforcement learning, which distills noisy multimodal queries into compact, retrieval-optimized search strings. \textbf{LENS} (\textbf{L}anguage-\textbf{E}nhanced \textbf{N}eural \textbf{S}earch) is a reasoning-enhanced dense retriever fine-tuned on reasoning-intensive retrieval data to handle the intent-rich queries FORGE produces. Evaluated on MM-BRIGHT (2,803 queries, 29 domains), BRIDGE achieves \textbf{29.7} nDCG@10, surpassing all multimodal encoder baselines including Nomic-Vision (27.6). When FORGE is applied as a plug-and-play aligner on top of Nomic-Vision, the combined system reaches \textbf{33.3} nDCG@10 -- exceeding the best text-only retriever (32.2) -- demonstrating that \textit{query alignment} is the key bottleneck in multimodal-to-text retrieval. this https URL
>
---
#### [new 120] Steering the Verifiability of Multimodal AI Hallucinations
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI可信性研究任务，旨在解决多模态AI幻觉的可验证性控制问题。通过构建数据集并提出干预方法，实现对不同类型幻觉的精细调控。**

- **链接: [https://arxiv.org/pdf/2604.06714](https://arxiv.org/pdf/2604.06714)**

> **作者:** Jianhong Pang; Ruoxi Cheng; Ziyi Ye; Xingjun Ma; Zuxuan Wu; Xuanjing Huang; Yu-Gang Jiang
>
> **摘要:** AI applications driven by multimodal large language models (MLLMs) are prone to hallucinations and pose considerable risks to human users. Crucially, such hallucinations are not equally problematic: some hallucination contents could be detected by human users(i.e., obvious hallucinations), while others are often missed or require more verification effort(i.e., elusive hallucinations). This indicates that multimodal AI hallucinations vary significantly in their verifiability. Yet, little research has explored how to control this property for AI applications with diverse security and usability demands. To address this gap, we construct a dataset from 4,470 human responses to AI-generated hallucinations and categorize these hallucinations into obvious and elusive types based on their verifiability by human users. Further, we propose an activation-space intervention method that learns separate probes for obvious and elusive hallucinations. We reveal that obvious and elusive hallucinations elicit different intervention probes, allowing for fine-grained control over the model's verifiability. Empirical results demonstrate the efficacy of this approach and show that targeted interventions yield superior performance in regulating corresponding verifiability. Moreover, simply mixing these interventions enables flexible control over the verifiability required for different scenarios.
>
---
#### [new 121] Enhanced Self-Supervised Multi-Image Super-Resolution for Camera Array Images
- **分类: physics.optics; cs.CV**

- **简介: 该论文属于多图像超分辨率任务，旨在解决传统方法在复杂退化和遮挡下的恢复难题。提出一种结合多到单与多到多的自监督学习框架及双Transformer网络，提升细节恢复能力。**

- **链接: [https://arxiv.org/pdf/2604.06816](https://arxiv.org/pdf/2604.06816)**

> **作者:** Yating Chen; Feng Huang; Xianyu Wu; Jing Wu; Ying Shen
>
> **摘要:** Conventional multi-image super-resolution (MISR) methods, such as burst and video SR, rely on sequential frames from a single camera. Consequently, they suffer from complex image degradation and severe occlusion, increasing the difficulty of accurate image restoration. In contrast, multi-aperture camera-array imaging captures spatially distributed views with sampling offsets forming a stable disk-like distribution, which enhances the non-redundancy of observed data. Existing MISR algorithms fail to fully exploit these unique properties. Supervised MISR methods tend to overfit the degradation patterns in training data, and current self-supervised learning (SSL) techniques struggle to recover fine-grained details. To address these issues, this paper thoroughly investigates the strengths, limitations and applicability boundaries of multi-image-to-single-image (Multi-to-Single) and multi-image-to-multi-image (Multi-to-Multi) SSL methods. We propose the Multi-to-Single-Guided Multi-to-Multi SSL framework that combines the advantages of Multi-to-Single and Multi-to-Multi to generate visually appealing and high-fidelity images rich in texture details. The Multi-to-Single-Guided Multi-to-Multi SSL framework provides a new paradigm for integrating deep neural network with classical physics-based variational methods. To enhance the ability of MISR network to recover high-frequency details from aliased artifacts, this paper proposes a novel camera-array SR network called dual Transformer suitable for SSL. Experiments on synthetic and real-world datasets demonstrate the superiority of the proposed method.
>
---
#### [new 122] BATON: A Multimodal Benchmark for Bidirectional Automation Transition Observation in Naturalistic Driving
- **分类: cs.HC; cs.CV; cs.MM**

- **简介: 该论文提出BATON数据集，用于研究驾驶自动化控制交接的多模态感知。任务是解决控制交接预测问题，通过融合视频、CAN信号等多源数据提升预测性能。**

- **链接: [https://arxiv.org/pdf/2604.07263](https://arxiv.org/pdf/2604.07263)**

> **作者:** Yuhang Wang; Yiyao Xu; Chaoyun Yang; Lingyao Li; Jingran Sun; Hao Zhou
>
> **摘要:** Existing driving automation (DA) systems on production vehicles rely on human drivers to decide when to engage DA while requiring them to remain continuously attentive and ready to intervene. This design demands substantial situational judgment and imposes significant cognitive load, leading to steep learning curves, suboptimal user experience, and safety risks from both over-reliance and delayed takeover. Predicting when drivers hand over control to DA and when they take it back is therefore critical for designing proactive, context-aware HMI, yet existing datasets rarely capture the multimodal context, including road scene, driver state, vehicle dynamics, and route environment. To fill this gap, we introduce BATON, a large-scale naturalistic dataset capturing real-world DA usage across 127 drivers, and 136.6 hours of driving. The dataset synchronizes front-view video, in-cabin video, decoded CAN bus signals, radar-based lead-vehicle interaction, and GPS-derived route context, forming a closed-loop multimodal record around each control transition. We define three benchmark tasks: driving action understanding, handover prediction, and takeover prediction, and evaluate baselines spanning sequence models, classical classifiers, and zero-shot VLMs. Results show that visual input alone is insufficient for reliable transition prediction: front-view video captures road context but not driver state, while in-cabin video reflects driver readiness but not the external scene. Incorporating CAN and route-context signals substantially improves performance over video-only settings, indicating strong complementarity across modalities. We further find takeover events develop more gradually and benefit from longer prediction horizons, whereas handover events depend more on immediate contextual cues, revealing an asymmetry with direct implications for HMI design in assisted driving systems.
>
---
#### [new 123] CWRNN-INVR: A Coupled WarpRNN based Implicit Neural Video Representation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于视频表示与压缩任务，旨在解决传统方法在结构与细节信息表达上的不足。提出CWRNN-INVR框架，结合神经网络与残差网格，提升视频重建效果。**

- **链接: [https://arxiv.org/pdf/2604.06564](https://arxiv.org/pdf/2604.06564)**

> **作者:** Yiyang Li; Yanbo Gao; Shuai Li; Zhenyu Du; Jinglin Zhang; Hui Yuan; Mao Ye; Xingyu Gao
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** Implicit Neural Video Representation (INVR) has emerged as a novel approach for video representation and compression, using learnable grids and neural networks. Existing methods focus on developing new grid structures efficient for latent representation and neural network architectures with large representation capability, lacking the study on their roles in video representation. In this paper, the difference between INVR based on neural network and INVR based on grid is first investigated from the perspective of video information composition to specify their own advantages, i.e., neural network for general structure while grid for specific detail. Accordingly, an INVR based on mixed neural network and residual grid framework is proposed, where the neural network is used to represent the regular and structured information and the residual grid is used to represent the remaining irregular information in a video. A Coupled WarpRNN-based multi-scale motion representation and compensation module is specifically designed to explicitly represent the regular and structured information, thus terming our method as CWRNN-INVR. For the irregular information, a mixed residual grid is learned where the irregular appearance and motion information are represented together. The mixed residual grid can be combined with the coupled WarpRNN in a way that allows for network reuse. Experiments show that our method achieves the best reconstruction results compared with the existing methods, with an average PSNR of 33.73 dB on the UVG dataset under the 3M model and outperforms existing INVR methods in other downstream tasks. The code can be found at this https URL}{this https URL.
>
---
#### [new 124] MAE-SAM2: Mask Autoencoder-Enhanced SAM2 for Clinical Retinal Vascular Leakage Segmentation
- **分类: q-bio.TO; cs.CV; eess.IV**

- **简介: 该论文提出MAE-SAM2模型，用于临床视网膜血管渗漏分割任务。针对数据量小、渗漏区域小且密集的问题，结合自监督学习和掩码自编码器，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2509.10554](https://arxiv.org/pdf/2509.10554)**

> **作者:** Xin Xing; Irmak Karaca; Amir Akhavanrezayat; Samira Badrloo; Quan Dong Nguyen; Mahadevan Subramaniam
>
> **摘要:** We propose MAE-SAM2, a novel foundation model for retinal vascular leakage segmentation on fluorescein angiography images. Due to the small size and dense distribution of the leakage areas, along with the limited availability of labeled clinical data, this presents a significant challenge for segmentation tasks. Our approach integrates a Self-Supervised learning (SSL) strategy, Masked Autoencoder (MAE), with SAM2. In our implementation, we explore different loss functions and conclude a task-specific combined loss. Extensive experiments and ablation studies demonstrate that MAE-SAM2 outperforms several state-of-the-art models, achieving the highest Dice score and Intersection-over-Union (IoU). Compared to the original SAM2, our model achieves a $5\%$ performance improvement, highlighting the promise of foundation models with self-supervised pretraining in clinical imaging tasks.
>
---
#### [new 125] An RTK-SLAM Dataset for Absolute Accuracy Evaluation in GNSS-Degraded Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决RTK-SLAM系统在GNSS受限环境下绝对精度评估问题。通过构建独立于GNSS的基准数据集，揭示SE(3)对齐带来的误差低估现象。**

- **链接: [https://arxiv.org/pdf/2604.07151](https://arxiv.org/pdf/2604.07151)**

> **作者:** Wei Zhang; Vincent Ress; David Skuddis; Uwe Soergel; Norbert Haala
>
> **备注:** Accepted by ISPRS congress 2026
>
> **摘要:** RTK-SLAM systems integrate simultaneous localization and mapping (SLAM) with real-time kinematic (RTK) GNSS positioning, promising both relative consistency and globally referenced coordinates for efficient georeferenced surveying. A critical and underappreciated issue is that the standard evaluation metric, Absolute Trajectory Error (ATE), first fits an optimal rigid-body transformation between the estimated trajectory and reference before computing errors. This so-called SE(3) alignment absorbs global drift and systematic errors, making trajectories appear more accurate than they are in practice, and is unsuitable for evaluating the global accuracy of RTK-SLAM. We present a geodetically referenced dataset and evaluation methodology that expose this gap. A key design principle is that the RTK receiver is used solely as a system input, while ground truth is established independently via a geodetic total station. This separation is absent from all existing datasets, where GNSS typically serves as (part of) the ground truth. The dataset is collected with a handheld RTK-SLAM device, comprising two scenes. We evaluate LiDAR-inertial, visual-inertial, and LiDAR-visual-inertial RTK-SLAM systems alongside standalone RTK, reporting direct global accuracy and SE(3)-aligned relative accuracy to make the gap explicit. Results show that SE(3) alignment can underestimate absolute positioning error by up to 76\%. RTK-SLAM achieves centimeter-level absolute accuracy in open-sky conditions and maintains decimeter-level global accuracy indoors, where standalone RTK degrades to tens of meters. The dataset, calibration files, and evaluation scripts are publicly available at this https URL.
>
---
#### [new 126] Towards foundation-style models for energy-frontier heterogeneous neutrino detectors via self-supervised pre-training
- **分类: hep-ex; cs.CV**

- **简介: 该论文属于粒子物理数据分析任务，旨在解决高能中微子探测中标签数据稀缺与复杂事件解释问题。通过自监督预训练学习可复用的表征，提升中微子识别与参数回归性能。**

- **链接: [https://arxiv.org/pdf/2604.07037](https://arxiv.org/pdf/2604.07037)**

> **作者:** Saúl Alonso-Monsalve; Fabio Cufino; Umut Kose; Anna Mascellani; André Rubbia
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Accelerator-based neutrino physics is entering an energy-frontier regime in which interactions reach the TeV scale and produce exceptionally dense, overlapping detector signatures. In this regime, event interpretation becomes impractical for conventional reconstruction approaches, particularly when labelled data are scarce and the analysis spans diverse downstream objectives. We present a sparse ViT framework for learning reusable representations from heterogeneous detector data. Self-supervised pre-training combines masked autoencoder reconstruction with relational voxel-level objectives for hierarchy, ghost and particle identification, and the resulting shared encoder is then jointly fine-tuned across classification and regression tasks. Evaluated on simulated events from the proposed FASERCal concept at the LHC, we find that pre-training consistently improves neutrino flavour and charm-quark identification, momentum regression, and vertex reconstruction over training from scratch, with the addition of relational objectives yielding further gains in the most topologically complex channels. Interpretability analyses further show that pre-training yields a more structured latent space, while detector-subsystem ablations recover physically plausible channel-dependent roles for the heterogeneous inputs. A data-efficiency study shows that, with roughly $10^3$ labelled events, the pre-trained encoder already matches the flavour-classification performance of a randomly initialised model trained on an order of magnitude more data. The learned representations also transfer effectively to publicly available benchmarks spanning different detector technologies and energy scales, matching or exceeding published baselines. These results support self-supervised pre-training on multimodal detector data as a scalable route towards reusable representations for neutrino and particle-detector analysis.
>
---
#### [new 127] RoSHI: A Versatile Robot-oriented Suit for Human Data In-the-Wild
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RoSHI系统，用于在真实环境中获取人体3D姿态和形状数据。解决人机交互数据采集的便携性、鲁棒性和全局一致性问题。融合IMU与视觉传感器，提升数据质量，适用于机器人学习。**

- **链接: [https://arxiv.org/pdf/2604.07331](https://arxiv.org/pdf/2604.07331)**

> **作者:** Wenjing Margaret Mao; Jefferson Ng; Luyang Hu; Daniel Gehrig; Antonio Loquercio
>
> **备注:** 8 pages, 4 figures. *Equal contribution by first three authors. Project webpage: this https URL
>
> **摘要:** Scaling up robot learning will likely require human data containing rich and long-horizon interactions in the wild. Existing approaches for collecting such data trade off portability, robustness to occlusion, and global consistency. We introduce RoSHI, a hybrid wearable that fuses low-cost sparse IMUs with the Project Aria glasses to estimate the full 3D pose and body shape of the wearer in a metric global coordinate frame from egocentric perception. This system is motivated by the complementarity of the two sensors: IMUs provide robustness to occlusions and high-speed motions, while egocentric SLAM anchors long-horizon motion and stabilizes upper body pose. We collect a dataset of agile activities to evaluate RoSHI. On this dataset, we generally outperform other egocentric baselines and perform comparably to a state-of-the-art exocentric baseline (SAM3D). Finally, we demonstrate that the motion data recorded from our system are suitable for real-world humanoid policy learning. For videos, data and more, visit the project webpage: this https URL
>
---
## 更新

#### [replaced 001] Robust Mesh Saliency Ground Truth Acquisition in VR via View Cone Sampling and Manifold Diffusion
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2601.02721](https://arxiv.org/pdf/2601.02721)**

> **作者:** Guoquan Zheng; Jie Hao; Huiyu Duan; Long Tang; Shuo Yang; Yucheng Zhu; Yongming Han; Liang Yuan; Patrick Le Callet; Guangtao Zhai
>
> **摘要:** As the complexity of 3D digital content grows exponentially, understanding human visual attention is critical for optimizing rendering and processing resources. Therefore, reliable 3D mesh saliency ground truth (GT) is essential for human-centric visual modeling in virtual reality (VR). However, existing VR eye-tracking frameworks are fundamentally bottlenecked by their underlying acquisition and generation mechanisms. The reliance on zero-area single ray sampling (SRS) fails to capture contextual features, leading to severe texture aliasing and discontinuous saliency signals. And the conventional application of Euclidean smoothing propagates saliency across disconnected physical gaps, resulting in semantic confusion on complex 3D manifolds. This paper proposes a robust framework to address these limitations. We first introduce a view cone sampling (VCS) strategy, which simulates the human foveal receptive field via Gaussian-distributed ray bundles to improve sampling robustness for complex topologies. Furthermore, a hybrid Manifold-Euclidean constrained diffusion (HCD) algorithm is developed, fusing manifold geodesic constraints with Euclidean scales to ensure topologically-consistent saliency propagation. We demonstrate the improvement in performance over baseline methods and the benefits for downstream tasks through subjective experiments and qualitative and quantitative methods. By mitigating "topological short-circuits" and aliasing, our framework provides a high-fidelity 3D attention acquisition paradigm that aligns with natural human perception, offering a more accurate and robust baseline for 3D mesh saliency research.
>
---
#### [replaced 002] Temporal Inversion for Learning Interval Change in Chest X-Rays
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.04563](https://arxiv.org/pdf/2604.04563)**

> **作者:** Hanbin Ko; Kyungmin Jeon; Doowoong Choi; Chang Min Park
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Recent advances in vision--language pretraining have enabled strong medical foundation models, yet most analyze radiographs in isolation, overlooking the key clinical task of comparing prior and current images to assess interval change. For chest radiographs (CXRs), capturing interval change is essential, as radiologists must evaluate not only the static appearance of findings but also how they evolve over time. We introduce TILA (Temporal Inversion-aware Learning and Alignment), a simple yet effective framework that uses temporal inversion, reversing image pairs, as a supervisory signal to enhance the sensitivity of existing temporal vision-language models to directional change. TILA integrates inversion-aware objectives across pretraining, fine-tuning, and inference, complementing conventional appearance modeling with explicit learning of temporal order. We also propose a unified evaluation protocol to assess order sensitivity and consistency under temporal inversion, and introduce MS-CXR-Tretrieval, a retrieval evaluation set constructed through a general protocol that can be applied to any temporal CXR dataset. Experiments on public datasets and real-world hospital cohorts demonstrate that TILA consistently improves progression classification and temporal embedding alignment when applied to multiple existing architectures.
>
---
#### [replaced 003] CylinderDepth: Cylindrical Spatial Attention for Multi-View Consistent Self-Supervised Surround Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16428](https://arxiv.org/pdf/2511.16428)**

> **作者:** Samer Abualhanud; Christian Grannemann; Max Mehltretter
>
> **备注:** Accepted at 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
>
> **摘要:** Self-supervised surround-view depth estimation enables dense, low-cost 3D perception with a 360° field of view from multiple minimally overlapping images. Yet, most existing methods suffer from depth estimates that are inconsistent across overlapping images. To address this limitation, we propose a novel geometry-guided method for calibrated, time-synchronized multi-camera rigs that predicts dense metric depth. Our approach targets two main sources of inconsistency: the limited receptive field in border regions of single-image depth estimation, and the difficulty of correspondence matching. We mitigate these two issues by extending the receptive field across views and restricting cross-view attention to a small neighborhood. To this end, we establish the neighborhood relationships between images by mapping the image-specific feature positions onto a shared cylinder. Based on the cylindrical positions, we apply an explicit spatial attention mechanism, with non-learned weighting, that aggregates features across images according to their distances on the cylinder. The modulated features are then decoded into a depth map for each view. Evaluated on the DDAD and nuScenes datasets, our method improves both cross-view depth consistency and overall depth accuracy compared with state-of-the-art approaches. Code is available at this https URL.
>
---
#### [replaced 004] SonoSelect: Efficient Ultrasound Perception via Active Probe Exploration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.05933](https://arxiv.org/pdf/2604.05933)**

> **作者:** Yixin Zhang; Yunzhong Hou; Longqi Li; Zhenyue Qin; Yang Liu; Yue Yao
>
> **备注:** Withdrawn due to incorrect institutional affiliation information. We need sufficient time to confirm the proper designations with the respective institutions before making the work public again
>
> **摘要:** Ultrasound perception typically requires multiple scan views through probe movement to reduce diagnostic ambiguity, mitigate acoustic occlusions, and improve anatomical coverage. However, not all probe views are equally informative. Exhaustively acquiring a large number of views can introduce substantial redundancy, increase scanning and processing costs. To address this, we define an active view exploration task for ultrasound and propose SonoSelect, an ultrasound-specific method that adaptively guides probe movement based on current observations. Specifically, we cast ultrasound active view exploration as a sequential decision-making problem. Each new 2D ultrasound view is fused into a 3D spatial memory of the observed anatomy, which guides the next probe position. On top of this formulation, we propose an ultrasound-specific objective that favors probe movements with greater organ coverage, lower reconstruction uncertainty, and less redundant scanning. Experiments on the ultrasound simulator show that SonoSelect achieves promising multi-view organ classification accuracy using only 2 out of N views. Furthermore, for a more difficult kidney cyst detection task, it reaches 54.56% kidney coverage and 35.13% cyst coverage, with short trajectories consistently centered on the target cyst.
>
---
#### [replaced 005] UniDAC: Universal Metric Depth Estimation for Any Camera
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27105](https://arxiv.org/pdf/2603.27105)**

> **作者:** Girish Chandar Ganesan; Yuliang Guo; Liu Ren; Xiaoming Liu
>
> **摘要:** Monocular metric depth estimation (MMDE) is a core challenge in computer vision, playing a pivotal role in real-world applications that demand accurate spatial understanding. Although prior works have shown promising zero-shot performance in MMDE, they often struggle with generalization across diverse camera types, such as fisheye and $360^\circ$ cameras. Recent advances have addressed this through unified camera representations or canonical representation spaces, but they require either including large-FoV camera data during training or separately trained models for different domains. We propose UniDAC, an MMDE framework that presents universal robustness in all domains and generalizes across diverse cameras using a single model. We achieve this by decoupling metric depth estimation into relative depth prediction and spatially varying scale estimation, enabling robust performance across different domains. We propose a lightweight Depth-Guided Scale Estimation module that upsamples a coarse scale map to high resolution using the relative depth map as guidance to account for local scale variations. Furthermore, we introduce RoPE-$\phi$, a distortion-aware positional embedding that respects the spatial warping in Equi-Rectangular Projections (ERP) via latitude-aware weighting. UniDAC achieves state of the art (SoTA) in cross-camera generalization by consistently outperforming prior methods across all datasets.
>
---
#### [replaced 006] PlaneCycle: Training-Free 2D-to-3D Lifting of Foundation Models Without Adapters
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.04165](https://arxiv.org/pdf/2603.04165)**

> **作者:** Yinghong Yu; Guangyuan Li; Jiancheng Yang
>
> **摘要:** Large-scale 2D foundation models exhibit strong transferable representations, yet extending them to 3D volumetric data typically requires retraining, adapters, or architectural redesign. We introduce PlaneCycle, a training-free, adapter-free operator for architecture-agnostic 2D-to-3D lifting of foundation models. PlaneCycle reuses the original pretrained 2D backbone by cyclically distributing spatial aggregation across orthogonal HW, DW, and DH planes throughout network depth, enabling progressive 3D fusion while preserving pretrained inductive biases. The method introduces no additional parameters and is applicable to arbitrary 2D networks. Using pretrained DINOv3 models, we evaluate PlaneCycle on six 3D classification and three 3D segmentation benchmarks. Without any training, the lifted models exhibit intrinsic 3D fusion capability and, under linear probing, outperform slice-wise 2D baselines and strong 3D counterparts, approaching the performance of fully trained models. With full fine-tuning, PlaneCycle matches standard 3D architectures, highlighting its potential as a seamless and practical 2D-to-3D lifting operator. These results demonstrate that 3D capability can be unlocked from pretrained 2D foundation models without structural modification or retraining. Code is available at this https URL.
>
---
#### [replaced 007] Focus on What Really Matters in Low-Altitude Governance: A Management-Centric Multi-Modal Benchmark with Implicitly Coordinated Vision-Language Reasoning Framework
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19640](https://arxiv.org/pdf/2601.19640)**

> **作者:** Hao Chang; Zhihui Wang; Lingxiang Wu; Wei An; Boyang Li; Zaiping Lin; Weidong Sheng; Jinqiao Wang
>
> **摘要:** Low-altitude vision systems are becoming a critical infrastructure for smart city governance. However, existing object-centric perception paradigms and loosely coupled vision-language pipelines are still difficult to support management-oriented anomaly understanding required in real-world urban governance. To bridge this gap, we introduce GovLA-10K, the first management-oriented multi-modal benchmark for low-altitude intelligence, along with GovLA-Reasoner, a unified vision-language reasoning framework tailored for governance-aware aerial perception. Unlike existing studies that aim to exhaustively annotate all visible objects, GovLA-10K is deliberately designed around functionally salient targets that directly correspond to practical management needs, and further provides actionable management suggestions grounded in these observations. To effectively coordinate the fine-grained visual grounding with high-level contextual language reasoning, GovLA-Reasoner introduces an efficient Spatially-aware Grounding Adapter (SGA) that implicitly coordinates discriminative representation sharing between the visual detector and the large language model (LLM). Different from existing adapters that primarily focus on global embedding alignment, our SGA is specifically designed to compress and aggregate multi-stream grounding-aware representations, thereby preserving fine-grained spatial cues while enabling their effective integration into the language reasoning process. Extensive experiments indicate that our GovLA-Reasoner effectively improves performance while avoiding the need of fine-tuning for any task-specific individual components. We believe our work offers a new perspective and foundation for future studies on management-aware low-altitude vision-language systems. The code and dataset will be publicly released after further organization.
>
---
#### [replaced 008] Learning Spatial-Preserving Hierarchical Representations for Digital Pathology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.09333](https://arxiv.org/pdf/2406.09333)**

> **作者:** Weiyi Wu; Xingjian Diao; Chunhui Zhang; Chongyang Gao; Xinwen Xu; Siting Li; Jiang Gui
>
> **摘要:** Whole slide images (WSIs) pose fundamental computational challenges due to their gigapixel resolution and the sparse distribution of informative regions. Existing approaches often treat image patches independently or reshape them in ways that distort spatial context, thereby obscuring the hierarchical pyramid representations intrinsic to WSIs. We introduce Sparse Pyramid Attention Networks (SPAN), a hierarchical framework that preserves spatial relationships while allocating computation to informative regions. SPAN constructs multi-scale representations directly from single-scale inputs, enabling precise hierarchical modeling of WSI data. We demonstrate SPAN's versatility through two variants: SPAN-MIL for slide classification and SPAN-UNet for segmentation. Comprehensive evaluations across multiple public datasets show that SPAN effectively captures hierarchical structure and contextual relationships. Our results provide clear evidence that architectural inductive biases and hierarchical representations enhance both slide-level and patch-level performance. By addressing key computational challenges in WSI analysis, SPAN provides an effective framework for computational pathology and demonstrates important design principles for large-scale medical image analysis.
>
---
#### [replaced 009] PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking
- **分类: cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.21540](https://arxiv.org/pdf/2507.21540)**

> **作者:** Quanchen Zou; Zonghao Ying; Moyang Chen; Wenzhuo Xu; Yisong Xiao; Yakai Li; Deyue Zhang; Dongdong Yang; Zhao Liu; Xiangzheng Zhang
>
> **备注:** This version is withdrawn to consolidate the submission under the corresponding author's primary account. The most recent and maintained version of this work can be found at arXiv:2603.09246
>
> **摘要:** The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.
>
---
#### [replaced 010] AHCQ-SAM: Toward Accurate and Hardware-Compatible Post-Training Segment Anything Model Quantization
- **分类: cs.CV; cs.AR; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.03088](https://arxiv.org/pdf/2503.03088)**

> **作者:** Wenlun Zhang; Yunshan Zhong; Weiqi Yan; Shengchuan Zhang; Shimpei Ando; Kentaro Yoshioka
>
> **备注:** Update to AHCQ-SAM
>
> **摘要:** The Segment Anything Model (SAM) has revolutionized image and video segmentation with its powerful zero-shot capabilities. However, its massive parameter scale and high computational demands hinder efficient deployment on resource-constrained edge devices. While Post-Training Quantization (PTQ) offers a practical solution, existing methods still fail to handle four critical quantization challenges: (1) ill-conditioned weights; (2) skewed and long-tailed post-GELU activations; (3) pronounced inter-channel variance in linear projections; and (4) exponentially scaled and heterogeneous attention scores. To mitigate these bottlenecks, we propose AHCQ-SAM, an accurate and hardware-compatible PTQ framework featuring four synergistic components: (1) Activation-aware Condition Number Reduction (ACNR), which regularizes weight matrices via a proximal point algorithm to suppress ill-conditioning; (2) Hybrid Log-Uniform Quantization (HLUQ), which combines power-of-two and uniform quantizers to capture skewed post-GELU activations; (3) Channel-Aware Grouping (CAG), which clusters channels with homogeneous statistics to achieve high accuracy with minimal hardware overhead; and (4) Logarithmic Nonlinear Quantization (LNQ), which utilizes logarithmic transformations to adaptively adjust quantization resolution for exponential and heterogeneous attention scores. Experimental results demonstrate that AHCQ-SAM outperforms current methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN on the COCO dataset. Furthermore, we establish a PTQ benchmark for SAM2, where AHCQ-SAM yields a 14.01% improvement in J&F for 4-bit SAM2-Tiny on the SA-V Test dataset. Finally, FPGA-based implementation validates the practical utility of AHCQ-SAM, delivering a 7.12x speedup and a 6.62x power efficiency improvement over the floating-point baseline.
>
---
#### [replaced 011] CHiQPM: Calibrated Hierarchical Interpretable Image Classification
- **分类: cs.LG; cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.20779](https://arxiv.org/pdf/2511.20779)**

> **作者:** Thomas Norrenbrock; Timo Kaiser; Sovan Biswas; Neslihan Kose; Ramesh Manuvinakurike; Bodo Rosenhahn
>
> **备注:** Accepted to NeurIPS 2025, updated version with correction
>
> **摘要:** Globally interpretable models are a promising approach for trustworthy AI in safety-critical domains. Alongside global explanations, detailed local explanations are a crucial complement to effectively support human experts during inference. This work proposes the Calibrated Hierarchical QPM (CHiQPM) which offers uniquely comprehensive global and local interpretability, paving the way for human-AI complementarity. CHiQPM achieves superior global interpretability by contrastively explaining the majority of classes and offers novel hierarchical explanations that are more similar to how humans reason and can be traversed to offer a built-in interpretable Conformal prediction (CP) method. Our comprehensive evaluation shows that CHiQPM achieves state-of-the-art accuracy as a point predictor, maintaining 99% accuracy of non-interpretable models. This demonstrates a substantial improvement, where interpretability is incorporated without sacrificing overall accuracy. Furthermore, its calibrated set prediction is competitively efficient to other CP methods, while providing interpretable predictions of coherent sets along its hierarchical explanation.
>
---
#### [replaced 012] SpatialEdit: Benchmarking Fine-Grained Image Spatial Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04911](https://arxiv.org/pdf/2604.04911)**

> **作者:** Yicheng Xiao; Wenhu Zhang; Lin Song; Yukang Chen; Wenbo Li; Nan Jiang; Tianhe Ren; Haokun Lin; Wei Huang; Haoyang Huang; Xiu Li; Nan Duan; Xiaojuan Qi
>
> **备注:** Code: this https URL
>
> **摘要:** Image spatial editing performs geometry-driven transformations, allowing precise control over object layout and camera viewpoints. Current models are insufficient for fine-grained spatial manipulations, motivating a dedicated assessment suite. Our contributions are listed: (i) We introduce SpatialEdit-Bench, a complete benchmark that evaluates spatial editing by jointly measuring perceptual plausibility and geometric fidelity via viewpoint reconstruction and framing analysis. (ii) To address the data bottleneck for scalable training, we construct SpatialEdit-500k, a synthetic dataset generated with a controllable Blender pipeline that renders objects across diverse backgrounds and systematic camera trajectories, providing precise ground-truth transformations for both object- and camera-centric operations. (iii) Building on this data, we develop SpatialEdit-16B, a baseline model for fine-grained spatial editing. Our method achieves competitive performance on general editing while substantially outperforming prior methods on spatial manipulation tasks. All resources will be made public at this https URL.
>
---
#### [replaced 013] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.19365](https://arxiv.org/pdf/2511.19365)**

> **作者:** Zehong Ma; Longhui Wei; Shuai Wang; Shiliang Zhang; Qi Tian
>
> **备注:** Accepted to CVPR2026. Project Page: this https URL. Code Repository: this https URL
>
> **摘要:** Pixel diffusion aims to generate images directly in pixel space in an end-to-end fashion. This approach avoids the limitations of VAE in the two-stage latent diffusion, offering higher model capacity. Existing pixel diffusion models suffer from slow training and inference, as they usually model both high-frequency signals and low-frequency semantics within a single diffusion transformer (DiT). To pursue a more efficient pixel diffusion paradigm, we propose the frequency-DeCoupled pixel diffusion framework. With the intuition to decouple the generation of high and low frequency components, we leverage a lightweight pixel decoder to generate high-frequency details conditioned on semantic guidance from the DiT. This thus frees the DiT to specialize in modeling low-frequency semantics. In addition, we introduce a frequency-aware flow-matching loss that emphasizes visually salient frequencies while suppressing insignificant ones. Extensive experiments show that DeCo achieves superior performance among pixel diffusion models, attaining FID of 1.62 (256x256) and 2.22 (512x512) on ImageNet, closing the gap with latent diffusion methods. Furthermore, our pretrained text-to-image model achieves a leading overall score of 0.86 on GenEval in system-level comparison. Codes are publicly available at this https URL.
>
---
#### [replaced 014] How to Embed Matters: Evaluation of EO Embedding Design Choices
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10658](https://arxiv.org/pdf/2603.10658)**

> **作者:** Luis Gilch; Isabelle Wittmann; Maximilian Nitsche; Johannes Jakubik; Arne Ewald; Thomas Brunschwiler
>
> **摘要:** Earth observation (EO) missions produce petabytes of multispectral imagery, increasingly analyzed using large Geospatial Foundation Models (GeoFMs). Alongside end-to-end adaptation, workflows make growing use of intermediate representations as task-agnostic embeddings, enabling models to compute representations once and reuse them across downstream tasks. Consequently, when GeoFMs act as feature extractors, decisions about how representations are obtained, aggregated, and combined affect downstream performance and pipeline scalability. Understanding these trade-offs is essential for scalable embedding-based EO workflows, where compact embeddings can replace raw data while remaining broadly useful. We present a systematic analysis of embedding design in GeoFM-based EO workflows. Leveraging NeuCo-Bench, we study how backbone architecture, pretraining strategy, representation depth, spatial aggregation, and representation combination influence EO task performance. We demonstrate the usability of GeoFM embeddings by aggregating them into fixed-size representations more than 500x smaller than the raw input data. Across models, we find consistent trends: transformer backbones with mean pooling provide strong default embeddings, intermediate ResNet layers can outperform final layers, self-supervised objectives exhibit task-specific strengths, and combining embeddings from different objectives often improves robustness.
>
---
#### [replaced 015] VisionClaw: Always-On AI Agents through Smart Glasses
- **分类: cs.HC; cs.AI; cs.CV; cs.LG; cs.MA**

- **链接: [https://arxiv.org/pdf/2604.03486](https://arxiv.org/pdf/2604.03486)**

> **作者:** Xiaoan Liu; DaeHo Lee; Eric J Gonzalez; Mar Gonzalez-Franco; Ryo Suzuki
>
> **备注:** 17 pages, 11 figures, plus appendix
>
> **摘要:** We present VisionClaw, an always-on wearable AI agent that integrates live egocentric perception with agentic task execution. Running on Meta Ray-Ban smart glasses, VisionClaw continuously perceives real-world context and enables in-situ, speech-driven action initiation and delegation via OpenClaw AI agents. Therefore, users can directly execute tasks through the smart glasses, such as adding real-world objects to an Amazon cart, generating notes from physical documents, receiving meeting briefings on the go, creating events from posters, or controlling IoT devices. We evaluate VisionClaw through a controlled laboratory study (N=12) and a longitudinal deployment study (N=5). Results show that integrating perception and execution enables faster task completion and reduces interaction overhead compared to non-always-on and non-agent baselines. Beyond performance gains, deployment findings reveal a shift in interaction: tasks are initiated opportunistically during ongoing activities, and execution is increasingly delegated rather than manually controlled. These results suggest a new paradigm for wearable AI agents, where perception and action are continuously coupled to support situated, hands-free interaction.
>
---
#### [replaced 016] UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于GUI代理任务，解决推理设计、奖励无效和视觉噪声问题。提出UI-AGILE框架，改进训练与推理方法，提升接地精度。**

- **链接: [https://arxiv.org/pdf/2507.22025](https://arxiv.org/pdf/2507.22025)**

> **作者:** Shuquan Lian; Yuhang Wu; Jia Ma; Yifan Ding; Zihan Song; Bingqi Chen; Xiawu Zheng; Hui Li; Rongrong Ji
>
> **摘要:** The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE for enhancing GUI agents at both training and inference. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a continuous reward function to incentivize high-precision grounding; 2) a ``Simple Thinking'' reward to balance planning with speed and grounding accuracy; and 3) a cropping-based resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present decomposed grounding with selection to dramatically improve grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art grounding performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2 while it also exhibits strong general agent capabilities. For instance, using both our training and inference enhancement methods brings 23\% grounding accuracy improvement over the best baseline on ScreenSpot-Pro. We provide the code in this https URL.
>
---
#### [replaced 017] PyFi: Toward Pyramid-like Financial Image Understanding for VLMs via Adversarial Agents
- **分类: q-fin.CP; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14735](https://arxiv.org/pdf/2512.14735)**

> **作者:** Yuqun Zhang; Yuxuan Zhao; Sijia Chen
>
> **摘要:** This paper proposes PyFi, a novel framework for pyramid-like financial image understanding that enables vision language models (VLMs) to reason through question chains in a progressive, simple-to-complex manner. At the core of PyFi is PyFi-600K, a dataset comprising 600K financial question-answer pairs organized into a reasoning pyramid: questions at the base require only basic perception, while those toward the apex demand increasing levels of capability in financial visual understanding and expertise. This data is scalable because it is synthesized without human annotations, using PyFi-adv, a multi-agent adversarial mechanism under the Monte Carlo Tree Search (MCTS) paradigm, in which, for each image, a challenger agent competes with a solver agent by generating question chains that progressively probe deeper capability levels in financial visual reasoning. Leveraging this dataset, we present fine-grained, hierarchical, and comprehensive evaluations of advanced VLMs in the financial domain. Moreover, fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B on the pyramid-structured question chains enables these models to answer complex financial questions by decomposing them into sub-questions with gradually increasing reasoning demands, yielding average accuracy improvements of 19.52% and 8.06%, respectively, on the dataset. All resources of code, dataset and models are available at: this https URL .
>
---
#### [replaced 018] SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts
- **分类: cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2511.22490](https://arxiv.org/pdf/2511.22490)**

> **作者:** Shun Inadumi; Shohei Tanaka; Tosho Hirasawa; Atsushi Hashimoto; Koichiro Yoshino; Yoshitaka Ushiku
>
> **备注:** CVPR2026 Findings
>
> **摘要:** As the number of scientific papers continues to grow, there is a demand for approaches that can effectively convey research findings, with posters serving as a key medium for presenting paper contents. Poster layouts determine how effectively research is communicated and understood, highlighting their growing importance. In particular, a gap remains in understanding how papers correspond to the layouts that present them, which calls for datasets with paired annotations at scale. To bridge this gap, we introduce SciPostGen, a large-scale dataset for understanding and generating poster layouts from scientific papers. Our analyses based on SciPostGen show that paper structures are associated with the number of layout elements in posters. Based on this insight, we explore a framework, Retrieval-Augmented Poster Layout Generation, which retrieves layouts consistent with a given paper and uses them as guidance for layout generation. We conducted experiments under two conditions: with and without layout constraints typically specified by poster creators. The results show that the retriever estimates layouts aligned with paper structures, and our framework generates layouts that also satisfy given constraints. The dataset and code are publicly available at this https URL.
>
---
#### [replaced 019] AdaptMMBench: Benchmarking Adaptive Multimodal Reasoning for Mode Selection and Reasoning Process
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02676](https://arxiv.org/pdf/2602.02676)**

> **作者:** Xintong Zhang; Xiaowen Zhang; Jingrong Wu; Zhi Gao; Shilin Yan; Zhenxin Diao; Kunpeng Gao; Xuanyan Chen; Yuwei Wu; Yunde Jia; Qing Li
>
> **摘要:** Adaptive multimodal reasoning has emerged as a promising frontier in Vision-Language Models (VLMs), aiming to dynamically modulate between tool-augmented visual reasoning and text reasoning to enhance both effectiveness and efficiency. However, existing evaluations rely on static difficulty labels and simplistic metrics, which fail to capture the dynamic nature of difficulty relative to varying model capacities. Consequently, they obscure the distinction between adaptive mode selection and general performance while neglecting fine-grained process analyses. In this paper, we propose AdaptMMBench, a comprehensive benchmark for adaptive multimodal reasoning across five domains: real-world, OCR, GUI, knowledge, and math, encompassing both direct perception and complex reasoning tasks. AdaptMMBench utilizes a Matthews Correlation Coefficient (MCC) metric to evaluate the selection rationality of different reasoning modes, isolating this meta-cognition ability by dynamically identifying task difficulties based on models' capability boundaries. Moreover, AdaptMMBench facilitates multi-dimensional process evaluation across key step coverage, tool effectiveness, and computational efficiency. Our evaluation reveals that while adaptive mode selection scales with model capacity, it notably decouples from final accuracy. Conversely, key step coverage aligns with performance, though tool effectiveness remains highly inconsistent across model architectures.
>
---
#### [replaced 020] Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2603.26167](https://arxiv.org/pdf/2603.26167)**

> **作者:** Yi Zhang; Hongbo Huang; Liang-Jie Zhang
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** Diffusion models generate high-quality images but pose serious risks like copyright violation and disinformation. Watermarking is a key defense for tracing and authenticating AI-generated content. However, existing methods rely on threshold-based detection, which only supports fuzzy matching and cannot recover structured watermark data bit-exactly, making them unsuitable for offline verification or applications requiring lossless metadata (e.g., licensing instructions). To address this problem, in this paper, we propose Gaussian Shannon, a watermarking framework that treats the diffusion process as a noisy communication channel and enables both robust tracing and exact bit recovery. Our method embeds watermarks in the initial Gaussian noise without fine-tuning or quality loss. We identify two types of channel interference, namely local bit flips and global stochastic distortions, and design a cascaded defense combining error-correcting codes and majority voting. This ensures reliable end-to-end transmission of semantic payloads. Experiments across three Stable Diffusion variants and seven perturbation types show that Gaussian Shannon achieves state-of-the-art bit-level accuracy while maintaining a high true positive rate, enabling trustworthy rights attribution in real-world deployment. The source code have been made available at: this https URL
>
---
#### [replaced 021] Reading Between the Pixels: An Inscriptive Jailbreak Attack on Text-to-Image Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.05853](https://arxiv.org/pdf/2604.05853)**

> **作者:** Zonghao Ying; Haowen Dai; Lianyu Hu; Zonglei Jing; Quanchen Zou; Yaodong Yang; Aishan Liu; Xianglong Liu
>
> **备注:** Withdrawn for extensive revisions and inclusion of new experimental results
>
> **摘要:** Modern text-to-image (T2I) models can now render legible, paragraph-length text, enabling a fundamentally new class of misuse. We identify and formalize the inscriptive jailbreak, where an adversary coerces a T2I system into generating images containing harmful textual payloads (e.g., fraudulent documents) embedded within visually benign scenes. Unlike traditional depictive jailbreaks that elicit visually objectionable imagery, inscriptive attacks weaponize the text-rendering capability itself. Because existing jailbreak techniques are designed for coarse visual manipulation, they struggle to bypass multi-stage safety filters while maintaining character-level fidelity. To expose this vulnerability, we propose Etch, a black-box attack framework that decomposes the adversarial prompt into three functionally orthogonal layers: semantic camouflage, visual-spatial anchoring, and typographic encoding. This decomposition reduces joint optimization over the full prompt space to tractable sub-problems, which are iteratively refined through a zero-order loop. In this process, a vision-language model critiques each generated image, localizes failures to specific layers, and prescribes targeted revisions. Extensive evaluations across 7 models on the 2 benchmarks demonstrate that Etch achieves an average attack success rate of 65.57% (peaking at 91.00%), significantly outperforming existing baselines. Our results reveal a critical blind spot in current T2I safety alignments and underscore the urgent need for typography-aware defense multimodal mechanisms.
>
---
#### [replaced 022] Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Splatblox，用于户外机器人导航任务，解决复杂地形中障碍物与可通行区域的识别问题。通过融合RGB图像和LiDAR数据，构建具有语义信息的ESDF，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2511.18525](https://arxiv.org/pdf/2511.18525)**

> **作者:** Samarth Chopra; Jing Liang; Gershom Seneviratne; Yonghan Lee; Jaehoon Choi; Jianyu An; Stephen Cheng; Dinesh Manocha
>
> **摘要:** We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: this https URL
>
---
#### [replaced 023] Pistachio: Towards Synthetic, Balanced, and Long-Form Video Anomaly Benchmarks
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.19474](https://arxiv.org/pdf/2511.19474)**

> **作者:** Jie Li; Hongyi Cai; Mingkang Dong; Muxin Pu; Shan You; Fei Wang; Tao Huang
>
> **备注:** this https URL
>
> **摘要:** Automatically detecting abnormal events in videos is crucial for modern autonomous systems, yet existing Video Anomaly Detection (VAD) benchmarks lack the scene diversity, balanced anomaly coverage, and temporal complexity needed to reliably assess real-world performance. Meanwhile, the community is increasingly moving toward Video Anomaly Understanding (VAU), which requires deeper semantic and causal reasoning but remains difficult to benchmark due to the heavy manual annotation effort it demands. In this paper, we introduce Pistachio, a new VAD/VAU benchmark constructed entirely through a controlled, generation-based pipeline. By leveraging recent advances in video generation models, Pistachio provides precise control over scenes, anomaly types, and temporal narratives, effectively eliminating the biases and limitations of Internet-collected datasets. Our pipeline integrates scene-conditioned anomaly assignment, multi-step storyline generation, and a temporally consistent long-form synthesis strategy that produces coherent 41-second videos with minimal human intervention. Extensive experiments demonstrate the scale, diversity, and complexity of Pistachio, revealing new challenges for existing methods and motivating future research on dynamic and multi-event anomaly understanding.
>
---
#### [replaced 024] STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction
- **分类: cs.CV; cs.GR; eess.IV**

- **链接: [https://arxiv.org/pdf/2603.20284](https://arxiv.org/pdf/2603.20284)**

> **作者:** Runze Wang; Yuxuan Song; Youcheng Cai; Ligang Liu
>
> **备注:** 10 pages, 6 figures. Accepted by CVPR 2026. This version includes supplementary material
>
> **摘要:** Online 3D reconstruction from streaming inputs requires both long-term temporal consistency and efficient memory usage. Although causal variants of VGGT address this challenge through a key-value (KV) cache mechanism, the cache grows linearly with the stream length, creating a major memory bottleneck. Under limited memory budgets, early cache eviction significantly degrades reconstruction quality and temporal consistency. In this work, we observe that attention in causal transformers for 3D reconstruction exhibits intrinsic spatio-temporal sparsity. Based on this insight, we propose STAC, a Spatio-Temporally Aware Cache Compression framework for streaming 3D reconstruction with large causal transformers. STAC consists of three key components: (1) a Working Temporal Token Caching mechanism that preserves long-term informative tokens using decayed cumulative attention scores; (2) a Long-term Spatial Token Caching scheme that compresses spatially redundant tokens into voxel-aligned representations for memory-efficient storage; and (3) a Chunk-based Multi-frame Optimization strategy that jointly processes consecutive frames to improve temporal coherence and GPU efficiency. Extensive experiments show that STAC achieves state-of-the-art reconstruction quality while reducing memory consumption by nearly 10x and accelerating inference by 4x, substantially improving the scalability of real-time 3D reconstruction in streaming settings.
>
---
#### [replaced 025] SoftHGNN: Soft Hypergraph Neural Networks for General Visual Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15325](https://arxiv.org/pdf/2505.15325)**

> **作者:** Mengqi Lei; Yihong Wu; Siqi Li; Xinhu Zheng; Juan Wang; Shaoyi Du; Yue Gao
>
> **备注:** This paper has been accepted by the International Journal of Computer Vision (IJCV)
>
> **摘要:** Visual recognition relies on understanding the semantics of image tokens and their complex interactions. Mainstream self-attention methods, while effective at modeling global pair-wise relations, fail to capture high-order associations inherent in real-world scenes and often suffer from redundant computation. Hypergraphs extend conventional graphs by modeling high-order interactions and offer a promising framework for addressing these limitations. However, existing hypergraph neural networks typically rely on static and hard hyperedge assignments, which lead to redundant hyperedges and overlooking the continuity of visual semantics. In this work, we present Soft Hypergraph Neural Networks (SoftHGNN), a lightweight plug-and-play hypergraph computation method for late-stage semantic reasoning in existing vision pipelines. Our SoftHGNN introduces the concept of soft hyperedges, where each vertex is associated with hyperedges via continuous and differentiable participation weights rather than hard binary assignments. These weights are produced by measuring similarities between vertex features and a small set of learnable hyperedge prototypes, yielding input-adaptive and semantically rich soft hyperedges. Using soft hyperedges as the medium for message aggregation and dissemination, SoftHGNN enriches feature representations with high-order contextual associations. To further enhance efficiency when scaling up the number of soft hyperedges, we incorporate a sparse hyperedge selection mechanism that activates only the top-k important hyperedges, along with a load-balancing regularizer to ensure adequate and balanced hyperedge utilization. Experimental results across three tasks on five datasets demonstrate that SoftHGNN efficiently captures high-order associations in visual scenes, achieving significant performance improvements. The code is available at: this https URL.
>
---
#### [replaced 026] CodecFlow: Codec-Guided End-to-End Optimization for Streaming Video Analytics
- **分类: cs.DC; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.06036](https://arxiv.org/pdf/2604.06036)**

> **作者:** Yulin Zou; Yan Chen; Wenyan Chen; JooYoung Park; Shivaraman Nitin; Luo Tao; Francisco Romero; Dmitrii Ustiugov
>
> **备注:** 18 pages, 34 figures
>
> **摘要:** Video streaming analytics is a crucial workload for vision-language model serving, but the high cost of multimodal inference limits scalability. Prior systems reduce inference cost by exploiting temporal and spatial redundancy in video streams, but they target either the vision transformer (ViT) or the LLM with a limited view, leaving end-to-end opportunities untapped. Moreover, existing methods incur significant overhead to identify redundancy, either through offline profiling and training or costly online computation, making them ill-suited for dynamic real-time streams. We present CodecFlow, a codec-guided streaming video analytics system built on a key observation that video codecs already extract the temporal and spatial structure of each stream as a byproduct of compression. CodecFlow treats this codec metadata as a low-cost runtime signal to unify optimization across video decoding, visual processing, and LLM prefilling, with transmission reduction as an inherent benefit of operating directly on compressed bitstreams. This drives codec-guided patch pruning before ViT encoding and selective key-value cache refresh during LLM prefilling, both of which are fully online and do not require offline training. Experiments show that CodecFlow achieves up to 3x throughput improvement and up to 87% GPU compute reduction over state-of-the-art baselines, while maintaining competitive accuracy with only 0-8% F1 drop.
>
---
#### [replaced 027] Can VLMs Unlock Semantic Anomaly Detection? A Framework for Structured Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中的异常检测任务，旨在解决罕见语义异常检测难题。提出SAVANT框架，通过语义一致性验证提升VLM的检测性能，并实现高效数据标注与模型优化。**

- **链接: [https://arxiv.org/pdf/2510.18034](https://arxiv.org/pdf/2510.18034)**

> **作者:** Roberto Brusnicki; David Pop; Yuan Gao; Mattia Piccinini; Johannes Betz
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Autonomous driving systems remain critically vulnerable to the long-tail of rare, out-of-distribution semantic anomalies. While VLMs have emerged as promising tools for perception, their application in anomaly detection remains largely restricted to prompting proprietary models - limiting reliability, reproducibility, and deployment feasibility. To address this gap, we introduce SAVANT (Semantic Anomaly Verification/Analysis Toolkit), a novel model-agnostic reasoning framework that reformulates anomaly detection as a layered semantic consistency verification. By applying SAVANT's two-phase pipeline - structured scene description extraction and multi-modal evaluation - existing VLMs achieve significantly higher scores in detecting anomalous driving scenarios from input images. Our approach replaces ad hoc prompting with semantic-aware reasoning, transforming VLM-based detection into a principled decomposition across four semantic domains. We show that across a balanced set of real-world driving scenarios, applying SAVANT improves VLM's absolute recall by approximately 18.5% compared to prompting baselines. Moreover, this gain enables reliable large-scale annotation: leveraging the best proprietary model within our framework, we automatically labeled around 10,000 real-world images with high confidence. We use the resulting high-quality dataset to fine-tune a 7B open-source model (Qwen2.5-VL) to perform single-shot anomaly detection, achieving 90.8% recall and 93.8% accuracy - surpassing all models evaluated while enabling local deployment at near-zero cost. By coupling structured semantic reasoning with scalable data curation, SAVANT provides a practical solution to data scarcity in semantic anomaly detection for autonomous systems. Supplementary material: this https URL
>
---
#### [replaced 028] B-MoE: A Body-Part-Aware Mixture-of-Experts "All Parts Matter" Approach to Micro-Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24245](https://arxiv.org/pdf/2603.24245)**

> **作者:** Nishit Poddar; Aglind Reka; Diana-Laura Borza; Snehashis Majhi; Michal Balazia; Abhijit Das; Francois Bremond
>
> **摘要:** Micro-actions, fleeting and low-amplitude motions, such as glances, nods, or minor posture shifts, carry rich social meaning but remain difficult for current action recognition models to recognize due to their subtlety, short duration, and high inter-class ambiguity. In this paper, we introduce B-MoE, a Body-part-aware Mixture-of-Experts framework designed to explicitly model the structured nature of human motion. In B-MoE, each expert specializes in a distinct body region (head, body, upper limbs, lower limbs), and is based on the lightweight Macro-Micro Motion Encoder (M3E) that captures long-range contextual structure and fine-grained local motion. A cross-attention routing mechanism learns inter-region relationships and dynamically selects the most informative regions for each micro-action. B-MoE uses a dual-stream encoder that fuses these region-specific semantic cues with global motion features to jointly capture spatially localized cues and temporally subtle variations that characterize micro-actions. Experiments on three challenging benchmarks (MA-52, SocialGesture, and MPII-GroupInteraction) show consistent state-of-theart gains, with improvements in ambiguous, underrepresented, and low amplitude classes.
>
---
#### [replaced 029] Drift-AR: Single-Step Visual Autoregressive Generation via Anti-Symmetric Drifting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.28049](https://arxiv.org/pdf/2603.28049)**

> **作者:** Zhen Zou; Xiaoxiao Ma; Mingde Yao; Jie Huang; LinJiang Huang; Feng Zhao
>
> **摘要:** Autoregressive (AR)-Diffusion hybrid paradigms combine AR's structured semantic modeling with diffusion's high-fidelity synthesis, yet suffer from a dual speed bottleneck: the sequential AR stage and the iterative multi-step denoising of the diffusion vision decode stage. Existing methods address each in isolation without a unified principle design. We observe that the per-position \emph{prediction entropy} of continuous-space AR models naturally encodes spatially varying generation uncertainty, which simultaneously governing draft prediction quality in the AR stage and reflecting the corrective effort required by vision decoding stage, which is not fully explored before. Since entropy is inherently tied to both bottlenecks, it serves as a natural unifying signal for joint acceleration. In this work, we propose \textbf{Drift-AR}, which leverages entropy signal to accelerate both stages: 1) for AR acceleration, we introduce Entropy-Informed Speculative Decoding that align draft-target entropy distributions via a causal-normalized entropy loss, resolving the entropy mismatch that causes excessive draft rejection; 2) for visual decoder acceleration, we reinterpret entropy as the \emph{physical variance} of the initial state for an anti-symmetric drifting field -- high-entropy positions activate stronger drift toward the data manifold while low-entropy positions yield vanishing drift -- enabling single-step (1-NFE) decoding without iterative denoising or distillation. Moreover, both stages share the same entropy signal, which is computed once with no extra cost. Experiments on MAR, TransDiff, and NextStep-1 demonstrate 3.8-5.5$\times$ speedup with genuine 1-NFE decoding, matching or surpassing original quality. Code will be available at this https URL.
>
---
#### [replaced 030] Generating Attribution Reports for Manipulated Facial Images: A Dataset and Baseline
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2412.19685](https://arxiv.org/pdf/2412.19685)**

> **作者:** Jingchun Lian; Lingyu Liu; Yaxiong Wang; Yujiao Wu; Lianwei Wu; Li Zhu; Zhedong Zheng
>
> **摘要:** Existing facial forgery detection methods typically focus on binary classification or pixel-level localization, providing little semantic insight into the nature of the manipulation. To address this, we introduce Forgery Attribution Report Generation, a new multimodal task that jointly localizes forged regions ("Where") and generates natural language explanations grounded in the editing process ("Why"). This dual-focus approach goes beyond traditional forensics, providing a comprehensive understanding of the manipulation. To enable research in this domain, we present Multi-Modal Tamper Tracing (MMTT), a large-scale dataset of 152,217 samples, each with a process-derived ground-truth mask and a human-authored textual description, ensuring high annotation precision and linguistic richness. We further propose ForgeryTalker, a unified end-to-end framework that integrates vision and language via a shared encoder (image encoder + Q-former) and dual decoders for mask and text generation, enabling coherent cross-modal reasoning. Experiments show that ForgeryTalker achieves competitive performance on both report generation and forgery localization subtasks, i.e., 59.3 CIDEr and 73.67 IoU, respectively, establishing a baseline for explainable multimedia forensics. Dataset and code will be released to foster future research.
>
---
#### [replaced 031] AgriPath: A Systematic Exploration of Architectural Trade-offs for Crop Disease Classification
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.13354](https://arxiv.org/pdf/2603.13354)**

> **作者:** Hamza Mooraj; George Pantazopoulos; Alessandro Suglia
>
> **备注:** 11 pages main text, 24 pages total including references and appendix. 6 figures, 14 tables. Code and dataset will be released upon publication
>
> **摘要:** Reliable crop disease detection requires models that perform consistently across diverse acquisition conditions, yet existing evaluations often focus on single architectural families or lab-generated datasets. This work presents a systematic empirical comparison of three model paradigms for fine-grained crop disease classification: Convolutional Neural Networks (CNNs), contrastive Vision-Language Models (VLMs), and generative VLMs. To enable controlled analysis of domain effects, we introduce AgriPath-LF16, a benchmark of 111k images spanning 16 crops and 41 diseases with explicit separation between laboratory and field imagery, alongside a balanced 30k subset for standardised training and evaluation. We train and evaluate all models under unified protocols across full, lab-only, and field-only training regimes using macro-F1 and Parse Success Rate (PSR) to account for generative reliability (i.e., output parsability measured via PSR). The results reveal distinct performance profiles: CNNs achieve the highest accuracy on in-domain imagery but exhibit pronounced degradation under domain shift; contrastive VLMs provide a robust and parameter-efficient alternative with competitive cross-domain performance; generative VLMs demonstrate the strongest resilience to distributional variation, albeit with additional failure modes stemming from free-text generation. These findings highlight that architectural choice should be guided by deployment context rather than aggregate performance alone.
>
---
#### [replaced 032] SleepNet and DreamNet: Enriching and Reconstructing Representations for Consolidated Visual Classification
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2409.01633](https://arxiv.org/pdf/2409.01633)**

> **作者:** Mingze Ni; Wei Liu
>
> **摘要:** An effective integration of rich feature representations with robust classification mechanisms remains a key challenge in visual understanding tasks. This study introduces two novel deep learning models, SleepNet and DreamNet, which are designed to improve representation utilization through feature enrichment and reconstruction strategies. SleepNet integrates supervised learning with representations obtained from pre-trained encoders, leading to stronger and more robust feature learning. Building on this foundation, DreamNet incorporates pre-trained encoder decoder frameworks to reconstruct hidden states, allowing deeper consolidation and refinement of visual representations. Our experiments show that our models consistently achieve superior performance compared with existing state-of-the-art methods, demonstrating the effectiveness of the proposed enrichment and reconstruction approaches.
>
---
#### [replaced 033] EarthEmbeddingExplorer: A Web Application for Cross-Modal Retrieval of Global Satellite Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29441](https://arxiv.org/pdf/2603.29441)**

> **作者:** Yijie Zheng; Weijie Wu; Bingyue Wu; Long Zhao; Guoqing Li; Mikolaj Czerkawski; Konstantin Klemmer
>
> **备注:** ICLR 2026 Workshop ML4RS Tutorial Track (oral)
>
> **摘要:** While the Earth observation community has witnessed a surge in high-impact foundation models and global Earth embedding datasets, a significant barrier remains in translating these academic assets into freely accessible tools. This tutorial introduces EarthEmbeddingExplorer, an interactive web application designed to bridge this gap, transforming static research artifacts into dynamic, practical workflows for discovery. We will provide a comprehensive hands-on guide to the system, detailing its cloud-native software architecture, demonstrating cross-modal queries (natural language, visual, and geolocation), and showcasing how to derive scientific insights from retrieval results. By democratizing access to precomputed Earth embeddings, this tutorial empowers researchers to seamlessly transition from state-of-the-art models and data archives to real-world application and analysis. The web application is available at this https URL.
>
---
#### [replaced 034] AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于图像目标导航任务，解决精确定位问题。通过几何查询方法，实现高精度位姿恢复，提升导航成功率和定位精度。**

- **链接: [https://arxiv.org/pdf/2604.05351](https://arxiv.org/pdf/2604.05351)**

> **作者:** Yijie Deng; Shuaihang Yuan; Yi Fang
>
> **摘要:** Image Goal Navigation (ImageNav) is evaluated by a coarse success criterion, the agent must stop within 1m of the target, which is sufficient for finding objects but falls short for downstream tasks such as grasping that require precise positioning. We introduce AnyImageNav, a training-free system that pushes ImageNav toward this more demanding setting. Our key insight is that the goal image can be treated as a geometric query: any photo of an object, a hallway, or a room corner can be registered to the agent's observations via dense pixel-level correspondences, enabling recovery of the exact 6-DoF camera pose. Our method realizes this through a semantic-to-geometric cascade: a semantic relevance signal guides exploration and acts as a proximity gate, invoking a 3D multi-view foundation model only when the current view is highly relevant to the goal image; the model then self-certifies its registration in a loop for an accurate recovered pose. Our method sets state-of-the-art navigation success rates on Gibson (93.1%) and HM3D (82.6%), and achieves pose recovery that prior methods do not provide: a position error of 0.27m and heading error of 3.41 degrees on Gibson, and 0.21m / 1.23 degrees on HM3D, a 5-10x improvement over adapted baselines.
>
---
#### [replaced 035] A Dynamic Prognostic Prediction Method for Colorectal Cancer Liver Metastasis
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2505.03123](https://arxiv.org/pdf/2505.03123)**

> **作者:** Wei Yang; Yiran Zhu; Yan su; Zesheng Li; Chengchang Pan; Honggang Qi
>
> **备注:** Accepted to IEEE International Conference on Multimedia and Expo (ICME) 2026
>
> **摘要:** Colorectal cancer liver metastasis (CRLM) exhibits high postoperative recurrence and pronounced prognostic heterogeneity, challenging individualized management. Existing prognostic approaches often rely on static representations from a single postoperative snapshot, and fail to jointly capture tumor spatial distribution, longitudinal disease dynamics, and multimodal clinical information, limiting predictive accuracy. We propose DyPro, a deep learning framework that infers postoperative latent trajectories via residual dynamic evolution. Starting from an initial patient representation, DyPro generates a 12-step sequence of trajectory snapshots through autoregressive residual updates and integrates them to predict recurrence and survival outcomes. On the MSKCC CRLM dataset, DyPro achieves strong discrimination under repeated stratified 5-fold cross-validation, reaching a C-index of 0.755 for OS and 0.714 for DFS, with OS AUC@1y of 0.920 and OS IBS of 0.143. DyPro provides quantitative risk cues to support adjuvant therapy planning and follow-up scheduling.
>
---
#### [replaced 036] Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.06676](https://arxiv.org/pdf/2602.06676)**

> **作者:** Bo Du; Xiaochen Ma; Xuekang Zhu; Zhe Yang; Chaogun Niu; Chenfan Qu; Mingqi Fang; Zhenming Wang; Jingjing Liu; Jian Liu; Ji-Zhe Zhou
>
> **摘要:** Fake Image Detection (FID), aiming at unified detection across four image forensic subdomains, is critical in real-world forensic scenarios. Compared with ensemble approaches, monolithic FID models are theoretically more promising, but to date, consistently yield inferior performance in practice. In this work, by discovering the ``heterogeneous phenomenon'', which is the intrinsic distinctness of artifacts across subdomains, we diagnose the cause of this underperformance for the first time: the collapse of the artifact feature space driven by such phenomenon. The core challenge for developing a practical monolithic FID model thus boils down to the ``unified-yet-discriminative" reconstruction of the artifact feature space. To address this paradoxical challenge, we hypothesize that high-level semantics can serve as a structural prior for the reconstruction, and further propose Semantic-Induced Constrained Adaptation (SICA), the first monolithic FID paradigm. Extensive experiments on our OpenMMSec dataset demonstrate that SICA outperforms 15 state-of-the-art methods and reconstructs the target unified-yet-discriminative artifact feature space in a near-orthogonal manner, thus firmly validating our hypothesis. The code and dataset are available at:https: //github.com/scu-zjz/SICA_OpenMMSec.
>
---
#### [replaced 037] MozzaVID: Mozzarella Volumetric Image Dataset
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2412.04880](https://arxiv.org/pdf/2412.04880)**

> **作者:** Pawel Tomasz Pieta; Peter Winkel Rasmussen; Anders Bjorholm Dahl; Jeppe Revall Frisvad; Siavash Arjomand Bigdeli; Carsten Gundlach; Anders Nymark Christensen
>
> **备注:** Accepted at MetaFood (CVPR 2026 Workshop)
>
> **摘要:** Influenced by the complexity of volumetric imaging, there is a shortage of established datasets useful for benchmarking volumetric deep-learning models. As a consequence, new and existing models are not easily comparable, limiting the development of architectures optimized specifically for volumetric data. To counteract this trend, we introduce MozzaVID -- a large, clean, and versatile volumetric classification dataset. Our dataset contains X-ray computed tomography (CT) images of mozzarella microstructure and enables the classification of 25 cheese types and 149 cheese samples. We provide data in three different resolutions, resulting in three dataset instances containing from 591 to 37,824 images. While targeted for developing general-purpose volumetric algorithms, the dataset also facilitates investigating the properties of mozzarella microstructure. The complex and disordered nature of food structures brings a unique challenge, where a choice of appropriate imaging method, scale, and sample size is not trivial. With this dataset, we aim to address these complexities, contributing to more robust structural analysis models and a deeper understanding of food structure. The dataset can be explored through: this https URL
>
---
#### [replaced 038] Erased, But Not Forgotten: Erased Rectified Flow Transformers Still Remain Unsafe Under Concept Attack
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00635](https://arxiv.org/pdf/2510.00635)**

> **作者:** Nanxiang Jiang; Zhaoxin Fan; Enhan Kang; Daiheng Gao; Yun Zhou; Yanxia Chang; Zheng Zhu; Yeying Jin; Wenjun Wu
>
> **摘要:** Recent advances in text-to-image (T2I) diffusion models have enabled impressive generative capabilities, but they also raise significant safety concerns due to the potential to produce harmful or undesirable content. While concept erasure has been explored as a mitigation strategy, most existing approaches and corresponding attack evaluations are tailored to Stable Diffusion (SD) and exhibit limited effectiveness when transferred to next-generation rectified flow transformers such as Flux. In this work, we present ReFlux, the first concept attack method specifically designed to assess the robustness of concept erasure in the latest rectified flow-based T2I framework. Our approach is motivated by the observation that existing concept erasure techniques, when applied to Flux, fundamentally rely on a phenomenon known as attention localization. Building on this insight, we propose a simple yet effective attack strategy that specifically targets this property. At its core, a reverse-attention optimization strategy is introduced to effectively reactivate suppressed signals while stabilizing attention. This is further reinforced by a velocity-guided dynamic that enhances the robustness of concept reactivation by steering the flow matching process, and a consistency-preserving objective that maintains the global layout and preserves unrelated content. Extensive experiments consistently demonstrate the effectiveness and efficiency of the proposed attack method, establishing a reliable benchmark for evaluating the robustness of concept erasure strategies in rectified flow transformers.
>
---
#### [replaced 039] Toward Memory-Aided World Models: Benchmarking via Spatial Consistency
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.22976](https://arxiv.org/pdf/2505.22976)**

> **作者:** Kewei Lian; Shaofei Cai; Yilun Du; Yitao Liang
>
> **备注:** V2: add details in appendix
>
> **摘要:** The ability to simulate the world in a spatially consistent manner is a crucial requirements for effective world models. Such a model enables high-quality visual generation, and also ensures the reliability of world models for downstream tasks such as simulation and planning. Designing a memory module is a crucial component for addressing spatial consistency: such a model must not only retain long-horizon observational information, but also enables the construction of explicit or implicit internal spatial representations. However, there are no dataset designed to promote the development of memory modules by explicitly enforcing spatial consistency constraints. Furthermore, most existing benchmarks primarily emphasize visual coherence or generation quality, neglecting the requirement of long-range spatial consistency. To bridge this gap, we construct a dataset and corresponding benchmark by sampling 150 distinct locations within the open-world environment of Minecraft, collecting about 250 hours (20 million frames) of loop-based navigation videos with actions. Our dataset follows a curriculum design of sequence lengths, allowing models to learn spatial consistency on increasingly complex navigation trajectories. Furthermore, our data collection pipeline is easily extensible to new Minecraft environments and modules. Four representative world model baselines are evaluated on our benchmark. Dataset, benchmark, and code are open-sourced to support future research.
>
---
#### [replaced 040] Unsupervised Source-Free Ranking of Biomedical Segmentation Models Under Distribution Shift
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.00450](https://arxiv.org/pdf/2503.00450)**

> **作者:** Joshua Talks; Kevin Marchesini; Luca Lumetti; Federico Bolelli; Anna Kreshuk
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Model reuse offers a solution to the challenges of segmentation in biomedical imaging, where high data annotation costs remain a major bottleneck for deep learning. However, although many pretrained models are released through challenges, model zoos, and repositories, selecting the most suitable model for a new dataset remains difficult due to the lack of reliable model ranking methods. We introduce the first black-box-compatible framework for unsupervised and source-free ranking of semantic and instance segmentation models based on the consistency of predictions under perturbations. While ranking methods have been studied for classification and a few segmentation-related approaches exist, most target related tasks such as transferability estimation or model validation and typically rely on labelled data, feature-space access, or specific training assumptions. In contrast, our method directly addresses the repository setting and applies to both semantic and instance segmentation, for zero-shot reuse or after unsupervised domain adaptation. We evaluate the approach across a wide range of biomedical segmentation tasks in both 2D and 3D imaging, showing that our estimated rankings strongly correlate with true target-domain model performance rankings.
>
---
#### [replaced 041] Rendering Multi-Human and Multi-Object with 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.02996](https://arxiv.org/pdf/2604.02996)**

> **作者:** Weiquan Wang; Jun Xiao; Feifei Shao; Yi Yang; Yueting Zhuang; Long Chen
>
> **备注:** 8 pages, 4 figures, accepted by ICRA 2026
>
> **摘要:** Reconstructing dynamic scenes with multiple interacting humans and objects from sparse-view inputs is a critical yet challenging task, essential for creating high-fidelity digital twins for robotics and VR/AR. This problem, which we term Multi-Human Multi-Object (MHMO) rendering, presents two significant obstacles: achieving view-consistent representations for individual instances under severe mutual occlusion, and explicitly modeling the complex and combinatorial dependencies that arise from their interactions. To overcome these challenges, we propose MM-GS, a novel hierarchical framework built upon 3D Gaussian Splatting. Our method first employs a Per-Instance Multi-View Fusion module to establish a robust and consistent representation for each instance by aggregating visual information across all available views. Subsequently, a Scene-Level Instance Interaction module operates on a global scene graph to reason about relationships between all participants, refining their attributes to capture subtle interaction effects. Extensive experiments on challenging datasets demonstrate that our method significantly outperforms strong baselines, producing state-of-the-art results with high-fidelity details and plausible inter-instance contacts.
>
---
#### [replaced 042] Gym-V: A Unified Vision Environment System for Agentic Vision Research
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.15432](https://arxiv.org/pdf/2603.15432)**

> **作者:** Fanqing Meng; Lingxiao Du; Jiawei Gu; Jiaqi Liao; Linjie Li; Zijian Wu; Xiangyan Liu; Ziqi Zhao; Mengkang Hu; Zichen Liu; Jiaheng Zhang; Michael Qizhe Shieh
>
> **摘要:** As agentic systems increasingly rely on reinforcement learning from verifiable rewards, standardized ``gym'' infrastructure has become essential for rapid iteration, reproducibility, and fair comparison. Vision agents lack such infrastructure, limiting systematic study of what drives their learning and where current models fall short. We introduce \textbf{Gym-V}, a unified platform of 179 procedurally generated visual environments across 10 domains with controllable difficulty, enabling controlled experiments that were previously infeasible across fragmented toolkits. Using it, we find that observation scaffolding is more decisive for training success than the choice of RL algorithm, with captions and game rules determining whether learning succeeds at all. Cross-domain transfer experiments further show that training on diverse task categories generalizes broadly while narrow training can cause negative transfer, with multi-turn interaction amplifying all of these effects. Gym-V is released as a convenient foundation for training environments and evaluation toolkits, aiming to accelerate future research on agentic VLMs.
>
---
#### [replaced 043] ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.17812](https://arxiv.org/pdf/2603.17812)**

> **作者:** Dmitriy Rivkin; Parker Ewen; Lili Gao; Julian Ost; Stefanie Walz; Rasika Kangutkar; Mario Bijelic; Felix Heide
>
> **备注:** Project website: this https URL
>
> **摘要:** Recent video diffusion models achieve high-quality generation through recurrent frame processing where each frame generation depends on previous frames. However, this recurrent mechanism means that training such models in the pixel domain incurs prohibitive memory costs, as activations accumulate across the entire video sequence. This fundamental limitation also makes fine-tuning these models with pixel-wise losses computationally intractable for long or high-resolution videos. This paper introduces ChopGrad, a truncated backpropagation scheme for video decoding, limiting gradient computation to local frame windows while maintaining global consistency. We provide a theoretical analysis of this approximation and show that it enables efficient fine-tuning with frame-wise losses. ChopGrad reduces training memory from scaling linearly with the number of video frames (full backpropagation) to constant memory, and compares favorably to existing state-of-the-art video diffusion models across a suite of conditional video generation tasks with pixel-wise losses, including video super-resolution, video inpainting, video enhancement of neural-rendered scenes, and controlled driving video generation.
>
---
#### [replaced 044] From Synthetic Data to Real Restorations: Diffusion Model for Patient-specific Dental Crown Completion
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.26588](https://arxiv.org/pdf/2603.26588)**

> **作者:** Dávid Pukanec; Tibor Kubík; Michal Španěl
>
> **备注:** VISAPP 2026 Conference / CVPR Workshop GenRecon3D
>
> **摘要:** We present ToothCraft, a diffusion-based model for the contextual generation of tooth crowns, trained on artificially created incomplete teeth. Building upon recent advancements in conditioned diffusion models for 3D shapes, we developed a model capable of an automated tooth crown completion conditioned on local anatomical context. To address the lack of training data for this task, we designed an augmentation pipeline that generates incomplete tooth geometries from a publicly available dataset of complete dental arches (3DS, ODD). By synthesising a diverse set of training examples, our approach enables robust learning across a wide spectrum of tooth defects. Experimental results demonstrate the strong capability of our model to reconstruct complete tooth crowns, achieving an intersection over union (IoU) of 81.8% and a Chamfer Distance (CD) of 0.00034 on synthetically damaged testing restorations. Our experiments demonstrate that the model can be applied directly to real-world cases, effectively filling in incomplete teeth, while generated crowns show minimal intersection with the opposing dentition, thus reducing the risk of occlusal interference. Access to the code, model weights, and dataset information will be available at: this https URL
>
---
#### [replaced 045] FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11680](https://arxiv.org/pdf/2601.11680)**

> **作者:** Zheng Zhang; Hao Tang; Yingying Hu; Zhanli Hu; Jing Qin
>
> **备注:** Accepted for oral presentation at AAAI 2026
>
> **摘要:** Low-count positron emission tomography (PET) reconstruction is a challenging inverse problem due to severe degradations arising from Poisson noise, photon scarcity, and attenuation correction errors. Existing deep learning methods typically address these in the spatial domain with an undifferentiated optimization objective, making it difficult to disentangle overlapping artifacts and limiting correction effectiveness. In this work, we perform a Fourier-domain analysis and reveal that these degradations are spectrally separable: Poisson noise and photon scarcity cause high-frequency phase perturbations, while attenuation errors suppress low-frequency amplitude components. Leveraging this insight, we propose FourierPET, a Fourier-based unrolled reconstruction framework grounded in the Alternating Direction Method of Multipliers. It consists of three tailored modules: a spectral consistency module that enforces global frequency alignment to maintain data fidelity, an amplitude-phase correction module that decouples and compensates for high-frequency phase distortions and low-frequency amplitude suppression, and a dual adjustment module that accelerates convergence during iterative reconstruction. Extensive experiments demonstrate that FourierPET achieves state-of-the-art performance with significantly fewer parameters, while offering enhanced interpretability through frequency-aware correction.
>
---
#### [replaced 046] VHOI: Controllable Video Generation of Human-Object Interactions from Sparse Trajectories via Motion Densification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.09646](https://arxiv.org/pdf/2512.09646)**

> **作者:** Wanyue Zhang; Lin Geng Foo; Thabo Beeler; Rishabh Dabral; Christian Theobalt
>
> **摘要:** Synthesizing realistic human-object interactions (HOI) in video is challenging due to the complex, instance-specific interaction dynamics of both humans and objects. Incorporating controllability in video generation further adds to the complexity. Existing controllable video generation approaches face a trade-off: sparse controls like keypoint trajectories are easy to specify but lack instance-awareness, while dense signals such as optical flow, depths or 3D meshes are informative but costly to obtain. We propose VHOI, a two-stage framework that first densifies sparse trajectories into HOI mask sequences, and then fine-tunes a video diffusion model conditioned on these dense masks. We introduce a novel HOI-aware motion representation that uses color encodings to distinguish not only human and object motion, but also body-part-specific dynamics. This design incorporates a human prior into the conditioning signal and strengthens the model's ability to understand and generate realistic HOI dynamics. Experiments demonstrate state-of-the-art results in controllable HOI video generation. VHOI is not limited to interaction-only scenarios and can also generate full human navigation leading up to object interactions in an end-to-end manner. Project page: this https URL.
>
---
#### [replaced 047] Countering the Over-Reliance Trap: Mitigating Object Hallucination for LVLMs via a Self-Validation Framework
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.22451](https://arxiv.org/pdf/2601.22451)**

> **作者:** Shiyu Liu; Xinyi Wen; Zhibin Lan; Ante Wang; Jinsong Su
>
> **备注:** Code is available at this https URL
>
> **摘要:** Despite progress in Large Vision Language Models (LVLMs), object hallucination remains a critical issue in image captioning task, where models generate descriptions of non-existent objects, compromising their reliability. Previous work attributes this to LVLMs' over-reliance on language priors and attempts to mitigate it through logits calibration. However, they still lack a thorough analysis of the over-reliance. To gain a deeper understanding of over-reliance, we conduct a series of preliminary experiments, indicating that as the generation length increases, LVLMs' over-reliance on language priors leads to inflated probability of hallucinated object tokens, consequently exacerbating object hallucination. To circumvent this issue, we propose Language-Prior-Free Verification to enable LVLMs to faithfully verify the confidence of object existence. Based on this, we propose a novel training-free Self-Validation Framework to counter the over-reliance trap. It first validates objects' existence in sampled candidate captions and further mitigates object hallucination via caption selection or aggregation. Experiment results demonstrate that our framework mitigates object hallucination significantly in image captioning task (e.g., 65.6% improvement on CHAIRI metric with LLaVA-v1.5-7B), surpassing the previous SOTA methods. This result highlights a novel path towards mitigating hallucination by unlocking the inherent potential within LVLMs themselves.
>
---
#### [replaced 048] Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态检索增强生成任务，解决图像查询中因背景干扰导致的重排序偏差问题。提出Region-R1框架，通过区域选择提升重排序效果。**

- **链接: [https://arxiv.org/pdf/2604.05268](https://arxiv.org/pdf/2604.05268)**

> **作者:** Chan-Wei Hu; Zhengzhong Tu
>
> **备注:** 12 pages, 4 figures, accepted to ACL 2026 Findings, code available at this https URL
>
> **摘要:** Multi-modal retrieval-augmented generation (MM-RAG) relies heavily on re-rankers to surface the most relevant evidence for image-question queries. However, standard re-rankers typically process the full query image as a global embedding, making them susceptible to visual distractors (e.g., background clutter) that skew similarity scores. We propose Region-R1, a query-side region cropping framework that formulates region selection as a decision-making problem during re-ranking, allowing the system to learn to retain the full image or focus only on a question-relevant region before scoring the retrieved candidates. Region-R1 learns a policy with a novel region-aware group relative policy optimization (r-GRPO) to dynamically crop a discriminative region. Across two challenging benchmarks, E-VQA and InfoSeek, Region-R1 delivers consistent gains, achieving state-of-the-art performances by increasing conditional Recall@1 by up to 20%. These results show the great promise of query-side adaptation as a simple but effective way to strengthen MM-RAG re-ranking.
>
---
#### [replaced 049] BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.21105](https://arxiv.org/pdf/2602.21105)**

> **作者:** Jiaxing Yu; Dongyang Ren; Hangyu Xu; Zhouyuxiao Yang; Yuanqi Li; Jie Guo; Zhengkang Zhou; Yanwen Guo
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** The boundary representation (B-Rep) models a 3D solid as its explicit boundaries: trimmed corners, edges, and faces. Recovering B-Rep representation from unstructured data is a challenging and valuable task of computer vision and graphics. Recent advances in deep learning have greatly improved the recovery of 3D shape geometry, but still depend on dense and clean point clouds and struggle to generalize to novel shapes. We propose B-Rep Gaussian Splatting (BrepGaussian), a novel framework that learns 3D parametric representations from 2D images. We employ a Gaussian Splatting renderer with learnable features, followed by a specific fitting strategy. To disentangle geometry reconstruction and feature learning, we introduce a two-stage learning framework that first captures geometry and edges and then refines patch features to achieve clean geometry and coherent instance representations. Extensive experiments demonstrate the superior performance of our approach to state-of-the-art methods.
>
---
#### [replaced 050] SpatialMosaic: A Multiview VLM Dataset for Partial Visibility
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23365](https://arxiv.org/pdf/2512.23365)**

> **作者:** Kanghee Lee; Injae Lee; Minseok Kwak; Jungi Hong; Kwonyoung Ryu; Jaesik Park
>
> **摘要:** The rapid progress of Multimodal Large Language Models (MLLMs) has unlocked the potential for enhanced 3D scene understanding and spatial reasoning. A recent line of work explores learning spatial reasoning directly from multi-view images, enabling MLLMs to understand 3D scenes without explicit 3D reconstructions. Nevertheless, key challenges that frequently arise in real-world environments, such as partial visibility, occlusion, and low-overlap conditions that require spatial reasoning from fragmented visual cues, remain under-explored. To address these limitations, we propose a scalable multi-view data generation and annotation pipeline that constructs realistic spatial reasoning QAs, resulting in SpatialMosaic, a comprehensive instruction-tuning dataset featuring 2M QA pairs. We further introduce SpatialMosaic-Bench, a challenging benchmark for evaluating multi-view spatial reasoning under complex and diverse scenarios, consisting of 1M QA pairs across 6 tasks. Our proposed dataset spans both indoor and outdoor scenes, enabling comprehensive evaluation in diverse real-world scenarios. In addition, we introduce a new baseline for multi-view settings, SpatialMosaicVLM, a hybrid framework that integrates 3D reconstruction models as geometry encoders within VLMs for robust spatial reasoning. Extensive experiments demonstrate that our proposed dataset effectively enhances spatial reasoning under challenging multi-view conditions, validating the effectiveness of our data generation pipeline in constructing realistic and challenging QAs. Code and dataset will be available soon.
>
---
#### [replaced 051] Looking Beyond the Obvious: A Survey on Abstract Concept Recognition for Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.20765](https://arxiv.org/pdf/2508.20765)**

> **作者:** Gowreesh Mago; Pascal Mettes; Stevan Rudinac
>
> **备注:** Accepted at IJCV
>
> **摘要:** The automatic understanding of video content is advancing rapidly. Empowered by deeper neural networks and large datasets, machines are increasingly capable of understanding what is concretely visible in video frames, whether it be objects, actions, events, or scenes. In comparison, humans retain a unique ability to also look beyond concrete entities and recognize abstract concepts like justice, freedom, and togetherness. Abstract concept recognition forms a crucial open challenge in video understanding, where reasoning on multiple semantic levels based on contextual information is key. In this paper, we argue that the recent advances in foundation models make for an ideal setting to address abstract concept understanding in videos. Automated understanding of high-level abstract concepts is imperative as it enables models to be more aligned with human reasoning and values. In this survey, we study different tasks and datasets used to understand abstract concepts in video content. We observe that, periodically and over a long period, researchers have attempted to solve these tasks, making the best use of the tools available at their disposal. We advocate that drawing on decades of community experience will help us shed light on this important open grand challenge and avoid ``re-inventing the wheel'' as we start revisiting it in the era of multi-modal foundation models.
>
---
#### [replaced 052] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.05743](https://arxiv.org/pdf/2604.05743)**

> **作者:** Amit Vaisman; Gal Pomerants; Raz Lapid
>
> **备注:** Accepted at AIGENS @ CVPR 2026
>
> **摘要:** Modern image compression methods are typically optimized for the rate--distortion--perception trade-off, whereas their robustness to bit-level corruption is rarely examined. We show that diffusion-based compressors built on the Reverse Channel Coding (RCC) paradigm are substantially more robust to bit flips than classical and learned codecs. We further introduce a more robust variant of Turbo-DDCM that significantly improves robustness while only minimally affecting the rate--distortion--perception trade-off. Our findings suggest that RCC-based compression can yield more resilient compressed representations, potentially reducing reliance on error-correcting codes in highly noisy environments.
>
---
#### [replaced 053] REVEAL: Reasoning-Enhanced Forensic Evidence Analysis for Explainable AI-Generated Image Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.23158](https://arxiv.org/pdf/2511.23158)**

> **作者:** Huangsen Cao; Qin Mei; Zhiheng Li; Yuxi Li; Zhan Meng; Ying Zhang; Chen Li; Zhimeng Zhang; Xin Ding; Yongwei Wang; Jing Lyu; Fei Wu
>
> **摘要:** The rapid progress of visual generative models has made AI-generated images increasingly difficult to distinguish from authentic ones, posing growing risks to social trust and information integrity. This motivates detectors that are not only accurate but also forensically explainable. While recent multimodal approaches improve interpretability, many rely on post-hoc rationalizations or coarse visual cues, without constructing verifiable chains of evidence, thus often leading to poor generalization. We introduce REVEAL-Bench, a reasoning-enhanced multimodal benchmark for AI-generated image forensics, structured around explicit chains of forensic evidence derived from lightweight expert models and consolidated into step-by-step chain-of-evidence traces. Based on this benchmark, we propose REVEAL (\underline{R}easoning-\underline{e}nhanced Forensic E\underline{v}id\underline{e}nce \underline{A}na\underline{l}ysis), an explainable forensic framework trained with expert-grounded reinforcement learning. Our reward design jointly promotes detection accuracy, evidence-grounded reasoning stability, and explanation faithfulness. Extensive experiments demonstrate significantly improved cross-domain generalization and more faithful explanations to baseline detectors. All data and codes will be released.
>
---
#### [replaced 054] Think in Strokes, Not Pixels: Process-Driven Image Generation via Interleaved Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04746](https://arxiv.org/pdf/2604.04746)**

> **作者:** Lei Zhang; Junjiao Tian; Zhipeng Fan; Kunpeng Li; Jialiang Wang; Weifeng Chen; Markos Georgopoulos; Felix Juefei-Xu; Yuxiang Bao; Julian McAuley; Manling Li; Zecheng He
>
> **摘要:** Humans paint images incrementally: they plan a global layout, sketch a coarse draft, inspect, and refine details, and most importantly, each step is grounded in the evolving visual states. However, can unified multimodal models trained on text-image interleaved datasets also imagine the chain of intermediate states? In this paper, we introduce process-driven image generation, a multi-step paradigm that decomposes synthesis into an interleaved reasoning trajectory of thoughts and actions. Rather than generating images in a single step, our approach unfolds across multiple iterations, each consisting of 4 stages: textual planning, visual drafting, textual reflection, and visual refinement. The textual reasoning explicitly conditions how the visual state should evolve, while the generated visual intermediate in turn constrains and grounds the next round of textual reasoning. A core challenge of process-driven generation stems from the ambiguity of intermediate states: how can models evaluate each partially-complete image? We address this through dense, step-wise supervision that maintains two complementary constraints: for the visual intermediate states, we enforce the spatial and semantic consistency; for the textual intermediate states, we preserve the prior visual knowledge while enabling the model to identify and correct prompt-violating elements. This makes the generation process explicit, interpretable, and directly supervisable. To validate proposed method, we conduct experiments under various text-to-image generation benchmarks.
>
---
#### [replaced 055] STRADAViT: Towards a Foundational Model for Radio Astronomy through Self-Supervised Transfer
- **分类: astro-ph.IM; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29660](https://arxiv.org/pdf/2603.29660)**

> **作者:** Andrea DeMarco; Ian Fenech Conti; Hayley Camilleri; Ardiana Bushi; Simone Riggi
>
> **备注:** 19 pages
>
> **摘要:** Next-generation radio astronomy surveys are delivering millions of resolved sources, but robust and scalable morphology analysis remains difficult across heterogeneous telescopes and imaging pipelines. We present STRADAViT, a self-supervised Vision Transformer continued-pretraining framework for learning transferable encoders from radio astronomy imagery. The framework combines mixed-survey data curation, radio astronomy-aware training-view generation, and a ViT-MAE-initialized encoder family with optional register tokens. It supports reconstruction-only, contrastive-only, and two-stage branches. Our pretraining dataset comprises radio astronomy cutouts drawn from four complementary sources. We evaluate transfer with linear probing and fine-tuning on three morphology benchmarks spanning binary and multi-class settings. Relative to the ViT-MAE initialization used for continued pretraining, the best two-stage models improve Macro-F1 in all reported linear-probe settings and in two of three fine-tuning settings, with the largest gain on RGZ DR1. Relative to DINOv2, gains are selective rather than universal: the best two-stage models achieve higher mean Macro-F1 than the strongest DINOv2 baseline on LoTSS DR2 and RGZ DR1 under linear probing, and on MiraBest and RGZ DR1 under fine-tuning. A targeted DINOv2 initialization ablation further indicates that the adaptation recipe is not specific to the ViT-MAE starting point and that, under the same recipe. The ViT-MAE-based STRADAViT checkpoint is retained as the released checkpoint because it combines competitive transfer with substantially lower token count and downstream cost than the DINOv2-based alternative. These results indicate that radio astronomy-aware view generation and staged continued pretraining can provide a stronger domain-adapted starting point than off-the-shelf ViT checkpoints for radio astronomy transfer.
>
---
#### [replaced 056] Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.01204](https://arxiv.org/pdf/2604.01204)**

> **作者:** Jorge Condor; Nicolas Moenne-Loccoz; Merlin Nimier-David; Piotr Didyk; Zan Gojcic; Qi Wu
>
> **摘要:** Primitive-based methods such as 3D Gaussian Splatting have recently become the state-of-the-art for novel-view synthesis and related reconstruction tasks. Compared to neural fields, these representations are more flexible, adaptive, and scale better to large scenes. However, the limited expressivity of individual primitives makes modeling high-frequency detail challenging. We introduce Neural Harmonic Textures, a neural representation approach that anchors latent feature vectors on a virtual scaffold surrounding each primitive. These features are interpolated within the primitive at ray intersection points. Inspired by Fourier analysis, we apply periodic activations to the interpolated features, turning alpha blending into a weighted sum of harmonic components. The resulting signal is then decoded in a single deferred pass using a small neural network, significantly reducing computational cost. Neural Harmonic Textures yield state-of-the-art results in real-time novel view synthesis while bridging the gap between primitive- and neural-field-based reconstruction. Our method integrates seamlessly into existing primitive-based pipelines such as 3DGUT, Triangle Splatting, and 2DGS. We further demonstrate its generality with applications to 2D image fitting and semantic reconstruction.
>
---
#### [replaced 057] dMLLM-TTS: Self-Verified and Efficient Test-Time Scaling for Diffusion Multi-Modal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19433](https://arxiv.org/pdf/2512.19433)**

> **作者:** Yi Xin; Siqi Luo; Tianxiang Xu; Qi Qin; Haoxing Chen; Kaiwen Zhu; Zhiwei Zhang; Yangfan He; Rongchao Zhang; Jinbin Bai; Shuo Cao; Bin Fu; Junjun He; Yihao Liu; Yuewen Cao; Xiaohong Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** Diffusion Multi-modal Large Language Models (dMLLMs) have recently emerged as a novel architecture unifying image generation and understanding. However, developing effective and efficient Test-Time Scaling (TTS) methods to unlock their full generative potential remains an underexplored challenge. To address this, we propose dMLLM-TTS, a novel framework operating on two complementary scaling axes: (1) trajectory exploration scaling to enhance the diversity of generated hypotheses, and (2) iterative refinement scaling for stable generation. Conventional TTS approaches typically perform linear search across these two dimensions, incurring substantial computational costs of O(NT) and requiring an external verifier for best-of-N selection. To overcome these limitations, we propose two innovations. First, we design an efficient hierarchical search algorithm with O(N+T) complexity that adaptively expands and prunes sampling trajectories. Second, we introduce a self-verified feedback mechanism that leverages the dMLLMs' intrinsic image understanding capabilities to assess text-image alignment, eliminating the need for external verifier. Extensive experiments on the GenEval benchmark across three representative dMLLMs (e.g., Lumina-DiMOO, MMaDA, Muddit) show that our framework substantially improves generation quality while achieving up to 6x greater efficiency than linear search. Project page: this https URL.
>
---
#### [replaced 058] Asking like Socrates: Socrates helps VLMs understand remote sensing images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.22396](https://arxiv.org/pdf/2511.22396)**

> **作者:** Run Shao; Ziyu Li; Zhaoyang Zhang; Linrui Xu; Xinran He; Hongyuan Yuan; Bolei He; Yongxing Dai; Yiming Yan; Yijun Chen; Wang Guo; Haifeng Li
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Recent multimodal reasoning models, inspired by DeepSeek-R1, have significantly advanced vision-language systems. However, in remote sensing (RS) tasks, we observe widespread pseudo reasoning: models narrate the process of reasoning rather than genuinely reason toward the correct answer based on visual evidence. We attribute this to the Glance Effect, where a single, coarse perception of large-scale RS imagery results in incomplete understanding and reasoning based on linguistic self-consistency instead of visual evidence. To address this, we propose RS-EoT (Remote Sensing Evidence-of-Thought), a language-driven, iterative visual evidence-seeking paradigm. To instill this paradigm, we propose SocraticAgent, a self-play multi-agent system that synthesizes reasoning traces via alternating cycles of reasoning and visual inspection. To enhance and generalize these patterns, we propose a two-stage progressive RL strategy: first, RL on fine-grained Grounding tasks to enhance RS-EoT capabilities, followed by RL on RS VQA to generalize to broader understanding scenarios. Experiments show RS-EoT achieves state-of-the-art performance on multiple RS VQA and grounding benchmarks. Analyses reveal clear iterative cycles of reasoning and evidence seeking, confirming RS-EoT mitigates the Glance Effect and enables genuine evidence-grounded reasoning. Our code, data, and models are available at this https URL
>
---
#### [replaced 059] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.06581](https://arxiv.org/pdf/2512.06581)**

> **作者:** Yuhao Su; Anwesa Choudhuri; Zhongpai Gao; Benjamin Planche; Van Nguyen Nguyen; Meng Zheng; Yuhan Shen; Arun Innanje; Terrence Chen; Ehsan Elhamifar; Ziyan Wu
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Large vision-language models struggle with medical video understanding, where spatial precision, temporal reasoning, and clinical semantics are critical. To address this, we first introduce \textbf{MedVidBench}, a large-scale benchmark of 531,850 video-instruction pairs across 8 medical sources spanning video, segment, and frame-level tasks, curated through a rigorous quality assurance pipeline with expert-guided prompting and dual-model validation. While supervised fine-tuning on MedVidBench yields noticeable gains, standard Reinforcement Learning (RL) fails due to imbalanced reward scales across datasets, which destabilizes optimization and leads to training collapse. To overcome this, we introduce \textbf{MedGRPO}, a novel RL framework for balanced multi-dataset training with two key innovations: (1) \emph{cross-dataset reward normalization} that maps each dataset's median performance to a common reward value, ensuring fair optimization regardless of difficulty, and (2) a \emph{medical LLM judge} that evaluates caption quality on five clinical dimensions through comparative similarity scoring. Supervised fine-tuning Qwen2.5-VL-7B on MedVidBench outperforms GPT-4.1 and Gemini-2.5-Flash across all tasks, while MedGRPO further improves the SFT baseline on grounding and captioning. Our work establishes a foundational benchmark and training methodology for advancing medical video understanding with VLMs. Our project website is available at: this https URL.
>
---
#### [replaced 060] MSG Score: Automated Video Verification for Reliable Multi-Scene Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2411.19121](https://arxiv.org/pdf/2411.19121)**

> **作者:** Daewon Yoon; Hyeongseok Lee; Wonsik Shin; Sangyu Han; Nojun Kwak
>
> **备注:** 8 pages, 5 figures, 1 table, Accepted AAAI 2026 CVM workshop
>
> **摘要:** While text-to-video diffusion models have advanced significantly, creating coherent long-form content remains unreliable due to stochastic sampling artifacts. This necessitates generating multiple candidates, yet verifying them creates a severe bottleneck; manual review is unscalable, and existing automated metrics lack the adaptability and speed required for runtime monitoring. Another critical issue is the trade-off between evaluation quality and run-time performance: metrics that best capture human-like judgment are often too slow to support iterative generation. These challenges, originating from the lack of an effective evaluation, motivate our work toward a novel solution. To address this, we propose a scalable automated verification framework for long-form video. First, we introduce the MSG(Multi-Scene Generation) score, a hierarchical attention-based metric that adaptively evaluates narrative and visual consistency. This serves as the core verifier within our CGS (Candidate Generation and Selection) framework, which automatically identifies and filters high-quality outputs. Furthermore, we introduce Implicit Insight Distillation (IID) to resolve the trade-off between evaluation reliability and inference speed, distilling complex metric insights into a lightweight student model. Our approach offers the first comprehensive solution for reliable and scalable long-form video production.
>
---
#### [replaced 061] TopoMaskV3: 3D Mask Head with Dense Offset and Height Predictions for Road Topology Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01558](https://arxiv.org/pdf/2603.01558)**

> **作者:** Muhammet Esat Kalfaoglu; Halil Ibrahim Ozturk; Ozsel Kilinc; Alptekin Temizel
>
> **备注:** Accepted to CVPR 2026 Workshops (AUTOPILOT 2026): 3rd Workshop on Autonomous Understanding Through Open-world Perception and Integrated Language Models for On-road Tasks
>
> **摘要:** Mask-based paradigms for road topology understanding, such as TopoMaskV2, offer a complementary alternative to query-based methods by generating centerlines via a dense rasterized intermediate representation. However, prior work was limited to 2D predictions and suffered from severe discretization artifacts, necessitating fusion with parametric heads. We introduce TopoMaskV3, which advances this pipeline into a robust, standalone 3D predictor via two novel dense prediction heads: a dense offset field for sub-grid discretization correction within the existing BEV resolution, and a dense height map for direct 3D estimation. Beyond the architecture, we are the first to address geographic data leakage in road topology evaluation by introducing (1) geographically distinct splits to prevent memorization and ensure fair generalization, and (2) a long-range (+/-100 m) benchmark. TopoMaskV3 achieves state-of-the-art 28.5 OLS on this geographically disjoint benchmark, surpassing all prior methods. Our analysis shows that the mask representation is more robust to geographic overfitting than Bezier, while LiDAR fusion is most beneficial at long range and exhibits larger relative gains on the overlapping original split, suggesting overlap-induced memorization effects.
>
---
#### [replaced 062] Less is More: Data-Efficient Adaptation for Controllable Text-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.17844](https://arxiv.org/pdf/2511.17844)**

> **作者:** Shihan Cheng; Nilesh Kulkarni; David Hyde; Dmitriy Smirnov
>
> **摘要:** Fine-tuning large-scale text-to-video diffusion models to add new generative controls, such as those over physical camera parameters (e.g., shutter speed or aperture), typically requires vast, high-fidelity datasets that are difficult to acquire. In this work, we propose a data-efficient fine-tuning strategy that learns these controls from sparse, low-quality synthetic data. We show that not only does fine-tuning on such simple data enable the desired controls, it actually yields superior results to models fine-tuned on photorealistic "real" data. Beyond demonstrating these results, we provide a framework that justifies this phenomenon both intuitively and quantitatively.
>
---
#### [replaced 063] DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2306.14685](https://arxiv.org/pdf/2306.14685)**

> **作者:** Ximing Xing; Chuang Wang; Haitao Zhou; Jing Zhang; Qian Yu; Dong Xu
>
> **备注:** Accepted by NeurIPS 2023. Project page: this https URL
>
> **摘要:** We demonstrate that pre-trained text-to-image diffusion models, despite being trained on raster images, possess a remarkable capacity to guide vector sketch synthesis. In this paper, we introduce DiffSketcher, a novel algorithm for generating vectorized free-hand sketches directly from natural language prompts. Our method optimizes a set of Bézier curves via an extended Score Distillation Sampling (SDS) loss, successfully bridging a raster-level diffusion prior with a parametric vector generator. To further accelerate the generation process, we propose a stroke initialization strategy driven by the diffusion model's intrinsic attention maps. Results show that DiffSketcher produces sketches across varying levels of abstraction while maintaining the structural integrity and essential visual details of the subject. Experiments confirm that our approach yields superior perceptual quality and controllability over existing methods. The code and demo are available at this https URL
>
---
#### [replaced 064] From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2512.07527](https://arxiv.org/pdf/2512.07527)**

> **作者:** Fei Yu; Yu Liu; Luyang Tang; Mingchao Sun; Zengye Ge; Rui Bu; Yuchao Jin; Haisen Zhao; He Sun; Yangyan Li; Mu Xu; Wenzheng Chen; Baoquan Chen
>
> **备注:** Accepted by CVPR 2026 Findings. Project page: this https URL
>
> **摘要:** City-scale 3D reconstruction from satellite imagery presents the challenge of extreme viewpoint extrapolation, where our goal is to synthesize ground-level novel views from sparse orbital images with minimal parallax. This requires inferring nearly $90^\circ$ viewpoint gaps from image sources with severely foreshortened facades and flawed textures, causing state-of-the-art reconstruction engines such as NeRF and 3DGS to fail. To address this problem, we propose two design choices tailored for city structures and satellite inputs. First, we model city geometry as a 2.5D height map, implemented as a Z-monotonic signed distance field (SDF) that matches urban building layouts from top-down viewpoints. This stabilizes geometry optimization under sparse, off-nadir satellite views and yields a watertight mesh with crisp roofs and clean, vertically extruded facades. Second, we paint the mesh appearance from satellite images via differentiable rendering techniques. While the satellite inputs may contain long-range, blurry captures, we further train a generative texture restoration network to enhance the appearance, recovering high-frequency, plausible texture details from degraded inputs. Our method's scalability and robustness are demonstrated through extensive experiments on large-scale urban reconstruction. For example, in our teaser figure, we reconstruct a $4\,\mathrm{km}^2$ real-world region from only a few satellite images, achieving state-of-the-art performance in synthesizing photorealistic ground views. The resulting models are not only visually compelling but also serve as high-fidelity, application-ready assets for downstream tasks like urban planning and simulation. Project page can be found at this https URL.
>
---
#### [replaced 065] A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.03604](https://arxiv.org/pdf/2602.03604)**

> **作者:** Basile Terver; Randall Balestriero; Megi Dervishi; David Fan; Quentin Garrido; Tushar Nagarajan; Koustuv Sinha; Wancong Zhang; Mike Rabbat; Yann LeCun; Amir Bar
>
> **备注:** v2: clarify confusion in definition of JEPAs vs. regularization-based JEPAs v3: Camera-ready of ICLR world models workshop, fixed formatting and ViT config / results
>
> **摘要:** We present EB-JEPA, an open-source library for learning representations and world models using Joint-Embedding Predictive Architectures (JEPAs). JEPAs learn to predict in representation space rather than pixel space, avoiding the pitfalls of generative modeling while capturing semantically meaningful features suitable for downstream tasks. Our library provides modular, self-contained implementations that illustrate how representation learning techniques developed for image-level self-supervised learning can transfer to video, where temporal dynamics add complexity, and ultimately to action-conditioned world models, where the model must additionally learn to predict the effects of control inputs. Each example is designed for single-GPU training within a few hours, making energy-based self-supervised learning accessible for research and education. We provide ablations of JEA components on CIFAR-10. Probing these representations yields 91% accuracy, indicating that the model learns useful features. Extending to video, we include a multi-step prediction example on Moving MNIST that demonstrates how the same principles scale to temporal modeling. Finally, we show how these representations can drive action-conditioned world models, achieving a 97% planning success rate on the Two Rooms navigation task. Comprehensive ablations reveal the critical importance of each regularization component for preventing representation collapse. Code is available at this https URL.
>
---
#### [replaced 066] Caption-Matching: A Multimodal Approach for Cross-Domain Image Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.15152](https://arxiv.org/pdf/2403.15152)**

> **作者:** Lucas Iijima; Nikolaos Giakoumoglou; Tania Stathaki
>
> **摘要:** Cross-Domain Image Retrieval (CDIR) is a challenging task in computer vision, aiming to match images across different visual domains such as sketches, paintings, and photographs. Existing CDIR methods rely either on supervised learning with labeled cross-domain correspondences or on methods that require training or fine-tuning on target datasets, often struggling with substantial domain gaps and limited generalization to unseen domains. This paper introduces a novel CDIR approach that incorporates textual context by leveraging publicly available pre-trained vision-language models. Our method, Caption-Matching (CM), uses generated image captions as a domain-agnostic intermediate representation, enabling effective cross-domain similarity computation without the need for labeled data or further training. We evaluate our method on standard CDIR benchmark datasets, demonstrating state-of-the-art performance in plug-and-play settings with consistent improvements on Office-Home and DomainNet over previous methods. We also demonstrate our method's effectiveness on a dataset of AI-generated images from Midjourney, showcasing its ability to handle complex, multi-domain queries.
>
---
#### [replaced 067] Efficient Image-to-Image Schrödinger Bridge for CT Field of View Extension
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11211](https://arxiv.org/pdf/2508.11211)**

> **作者:** Zhenhao Li; Song Ni; Long Yang; Xiaojie Yin; Haijun Yu; Jiazhou Wang; Hongbin Han; Weigang Hu; Yixing Huang
>
> **备注:** 10 pages
>
> **摘要:** Computed tomography (CT) is a cornerstone imaging modality for non-invasive, high-resolution visualization of internal anatomical structures. However, when the scanned object exceeds the scanner's field of view (FOV), projection data are truncated, resulting in incomplete reconstructions and pronounced artifacts near FOV boundaries. Conventional reconstruction algorithms struggle to recover accurate anatomy from such data, limiting clinical reliability. Deep learning approaches have been explored for FOV extension, with diffusion generative models representing the latest advances in image synthesis. Yet, conventional diffusion models are computationally demanding and slow at inference due to their iterative sampling process. To address these limitations, we propose an efficient CT FOV extension framework based on the image-to-image Schrödinger Bridge (I$^2$SB) diffusion model. Unlike traditional diffusion models that synthesize images from pure Gaussian noise, I$^2$SB learns a direct stochastic mapping between paired limited-FOV and extended-FOV images. This direct correspondence yields a more interpretable and traceable generative process, enhancing anatomical consistency and structural fidelity in reconstructions. I$^2$SB achieves superior quantitative performance, with root-mean-square error (RMSE) values of 49.8 HU on simulated noisy data and 152.0 HU on real data, outperforming state-of-the-art diffusion models such as conditional denoising diffusion probabilistic models (cDDPM) and patch-based diffusion methods. Moreover, its one-step inference enables reconstruction in just 0.19 s per 2D slice, representing over a 700-fold speedup compared to cDDPM (135 s) and surpassing DiffusionGAN (0.58 s), the second fastest. This combination of accuracy and efficiency indicates that I$^2$SB has potential for real-time or clinical deployment.
>
---
#### [replaced 068] Retrievals Can Be Detrimental: A Contrastive Backdoor Attack Paradigm on Retrieval-Augmented Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.13340](https://arxiv.org/pdf/2501.13340)**

> **作者:** Hao Fang; Xiaohang Sui; Hongyao Yu; Kuofeng Gao; Jiawei Kong; Sijin Yu; Bin Chen; Shu-Tao Xia
>
> **备注:** Accepted by ACL-2026
>
> **摘要:** Diffusion models (DMs) have recently demonstrated remarkable generation capability. However, their training generally requires huge computational resources and large-scale datasets. To solve these, recent studies empower DMs with the advanced Retrieval-Augmented Generation (RAG) technique and propose retrieval-augmented diffusion models (RDMs). By incorporating rich knowledge from an auxiliary database, RAG enhances diffusion models' generation and generalization ability while significantly reducing model parameters. Despite the great success, RAG may introduce novel security issues that warrant further investigation. In this paper, we reveal that the RDM is susceptible to backdoor attacks by proposing a multimodal contrastive attack approach named BadRDM. Our framework fully considers RAG's characteristics and is devised to manipulate the retrieved items for given text triggers, thereby further controlling the generated contents. Specifically, we first insert a tiny portion of images into the retrieval database as target toxicity surrogates. Subsequently, a malicious variant of contrastive learning is adopted to inject backdoors into the retriever, which builds shortcuts from triggers to the toxicity surrogates. Furthermore, we enhance the attacks through novel entropy-based selection and generative augmentation strategies that can derive better toxicity surrogates. Extensive experiments on two mainstream tasks demonstrate the proposed BadRDM achieves outstanding attack effects while preserving the model's benign utility.
>
---
#### [replaced 069] RASALoRE: Region Aware Spatial Attention with Location-based Random Embeddings for Weakly Supervised Anomaly Detection in Brain MRI Scans
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.08052](https://arxiv.org/pdf/2510.08052)**

> **作者:** Bheeshm Sharma; Karthikeyan Jaganathan; Balamurugan Palaniappan
>
> **备注:** Accepted at the 36th British Machine Vision Conference (BMVC-2025)
>
> **摘要:** Weakly Supervised Anomaly detection (WSAD) in brain MRI scans is an important challenge useful to obtain quick and accurate detection of brain anomalies when precise pixel-level anomaly annotations are unavailable and only weak labels (e.g., slice-level) are available. In this work, we propose RASALoRE: Region Aware Spatial Attention with Location-based Random Embeddings, a novel two-stage WSAD framework. In the first stage, we introduce a Discriminative Dual Prompt Tuning (DDPT) mechanism that generates high-quality pseudo weak masks based on slice-level labels, serving as coarse localization cues. In the second stage, we propose a segmentation network with a region-aware spatial attention mechanism that relies on fixed location-based random embeddings. This design enables the model to effectively focus on anomalous regions. Our approach achieves state-of-the-art anomaly detection performance, significantly outperforming existing WSAD methods while utilizing less than 8 million parameters. Extensive evaluations on the BraTS20, BraTS21, BraTS23, and MSD datasets demonstrate a substantial performance improvement coupled with a significant reduction in computational complexity. Code is available at: this https URL.
>
---
#### [replaced 070] V$^{2}$-SAM: Marrying SAM2 with Multi-Prompt Experts for Cross-View Object Correspondence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20886](https://arxiv.org/pdf/2511.20886)**

> **作者:** Jiancheng Pan; Runze Wang; Tianwen Qian; Mohammad Mahdi; Yanwei Fu; Xiangyang Xue; Xiaomeng Huang; Luc Van Gool; Danda Pani Paudel; Yuqian Fu
>
> **备注:** 19 pages
>
> **摘要:** Cross-view object correspondence, exemplified by the representative task of ego-exo object correspondence, aims to establish consistent associations of the same object across different viewpoints (e.g., egocentric and exocentric). This task poses significant challenges due to drastic viewpoint and appearance variations, making existing segmentation models, such as SAM2, difficult to apply directly. To address this, we present V2-SAM, a unified cross-view object correspondence framework that adapts SAM2 from single-view segmentation to cross-view correspondence through two complementary prompt generators. Specifically, the Cross-View Anchor Prompt Generator (V2-Anchor), built upon DINOv3 features, establishes geometry-aware correspondences and, for the first time, enables coordinate-based prompting for SAM2 in cross-view scenarios, while the Cross-View Visual Prompt Generator (V2-Visual) enhances appearance-guided cues via a novel visual prompt matcher that aligns ego-exo representations from both feature and structural perspectives. To effectively exploit the strengths of both prompts, we further adopt a multi-expert design and introduce a Post-hoc Cyclic Consistency Selector (PCCS) that adaptively selects the most reliable expert based on cyclic consistency. Extensive experiments validate the effectiveness of V2-SAM, achieving new state-of-the-art performance on Ego-Exo4D (ego-exo object correspondence), DAVIS-2017 (video object tracking), and HANDAL-X (robotic-ready cross-view correspondence).
>
---
#### [replaced 071] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.05584](https://arxiv.org/pdf/2604.05584)**

> **作者:** Pengcheng Weng; Yanyu Qian; Yangxin Xu; Fei Wang
>
> **备注:** Accepted by CVPR 2026 Workshop On Any-to-Any Multimodal Learning
>
> **摘要:** Robust multimodal human sensing must overcome the critical challenge of missing modalities. Two principal barriers are the Representation Gap between heterogeneous data and the Contamination Effect from low-quality modalities. These barriers are causally linked, as the corruption introduced by contamination fundamentally impedes the reduction of representation disparities. In this paper, we propose PTA, a novel "Purify-then-Align" framework that solves this causal dependency through a synergistic integration of meta-learning and knowledge diffusion. To purify the knowledge source, PTA first employs a meta-learning-driven weighting mechanism that dynamically learns to down-weight the influence of noisy, low-contributing modalities. Subsequently, to align different modalities, PTA introduces a diffusion-based knowledge distillation paradigm in which an information-rich clean teacher, formed from this purified consensus, refines the features of each student modality. The ultimate payoff of this "Purify-then-Align" strategy is the creation of exceptionally powerful single-modality encoders imbued with cross-modal knowledge. Comprehensive experiments on the large-scale MM-Fi and XRF55 datasets, under pronounced Representation Gap and Contamination Effect, demonstrate that PTA achieves state-of-the-art performance and significantly improves the robustness of single-modality models in diverse missing-modality scenarios.
>
---
#### [replaced 072] FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15599](https://arxiv.org/pdf/2512.15599)**

> **作者:** Tobias Kirschstein; Simon Giebenhain; Matthias Nießner
>
> **备注:** Accepted to CVPR 2026, Project website: this https URL , Video: this https URL
>
> **摘要:** We introduce FlexAvatar, a method for creating high-quality and complete 3D head avatars from a single image. A core challenge lies in the limited availability of multi-view data and the tendency of monocular training to yield incomplete 3D head reconstructions. We identify the root cause of this issue as the entanglement between driving signal and target viewpoint when learning from monocular videos. To address this, we propose a transformer-based 3D portrait animation model with learnable data source tokens, so-called bias sinks, which enables unified training across monocular and multi-view datasets. This design leverages the strengths of both data sources during inference: strong generalization from monocular data and full 3D completeness from multi-view supervision. Furthermore, our training procedure yields a smooth latent avatar space that facilitates identity interpolation and flexible fitting to an arbitrary number of input observations. In extensive evaluations on single-view, few-shot, and monocular avatar creation tasks, we verify the efficacy of FlexAvatar. Many existing methods struggle with view extrapolation while FlexAvatar generates complete 3D head avatars with realistic facial animations. Website: this https URL
>
---
#### [replaced 073] Positive-First Most Ambiguous: A Simple Active Learning Criterion for Interactive Retrieval of Rare Categories
- **分类: cs.CV; cs.HC; cs.IR**

- **链接: [https://arxiv.org/pdf/2603.24480](https://arxiv.org/pdf/2603.24480)**

> **作者:** Kawtar Zaher; Olivier Buisson; Alexis Joly
>
> **备注:** CVPRW 2026 - The 13th Workshop on Fine-Grained Visual Categorization (FGVC13)
>
> **摘要:** Real-world fine-grained visual retrieval often requires discovering a rare concept from large unlabeled collections with minimal supervision. This is especially critical in biodiversity monitoring, ecological studies, and long-tailed visual domains, where the target may represent only a tiny fraction of the data, creating highly imbalanced binary problems. Interactive retrieval with relevance feedback offers a practical solution: starting from a small query, the system selects candidates for binary user annotation and iteratively refines a lightweight classifier. While Active Learning (AL) is commonly used to guide selection, conventional AL assumes symmetric class priors and large annotation budgets, limiting effectiveness in imbalanced, low-budget, low-latency settings. We introduce Positive-First Most Ambiguous (PF-MA), a simple yet effective AL criterion that explicitly addresses the class imbalance asymmetry: it prioritizes near-boundary samples while favoring likely positives, enabling rapid discovery of subtle visual categories while maintaining informativeness. Unlike standard methods that oversample negatives, PF-MA consistently returns small batches with a high proportion of relevant samples, improving early retrieval and user satisfaction. To capture retrieval diversity, we also propose a class coverage metric that measures how well selected positives span the visual variability of the target class. Experiments on long-tailed datasets, including fine-grained botanical data, demonstrate that PF-MA consistently outperforms strong baselines in both coverage and classifier performance, across varying class sizes and descriptors. Our results highlight that aligning AL with the asymmetric and user-centric objectives of interactive fine-grained retrieval enables simple yet powerful solutions for retrieving rare and visually subtle categories in realistic human-in-the-loop settings.
>
---
#### [replaced 074] Exploring Conditions for Diffusion models in Robotic Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决预训练视觉表示在控制任务中适应性不足的问题。通过引入可学习的提示，提升视觉表示的动态适应性，从而提高控制性能。**

- **链接: [https://arxiv.org/pdf/2510.15510](https://arxiv.org/pdf/2510.15510)**

> **作者:** Heeseong Shin; Byeongho Heo; Dongyoon Han; Seungryong Kim; Taekyung Kim
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** While pre-trained visual representations have significantly advanced imitation learning, they are often task-agnostic as they remain frozen during policy learning. In this work, we explore leveraging pre-trained text-to-image diffusion models to obtain task-adaptive visual representations for robotic control, without fine-tuning the model itself. However, we find that naively applying textual conditions - a successful strategy in other vision domains - yields minimal or even negative gains in control tasks. We attribute this to the domain gap between the diffusion model's training data and robotic control environments, leading us to argue for conditions that consider the specific, dynamic visual information required for control. To this end, we propose ORCA, which introduces learnable task prompts that adapt to the control environment and visual prompts that capture fine-grained, frame-specific details. Through facilitating task-adaptive representations with our newly devised conditions, our approach achieves state-of-the-art performance on various robotic control benchmarks, significantly surpassing prior methods.
>
---
#### [replaced 075] Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.03647](https://arxiv.org/pdf/2604.03647)**

> **作者:** Yunyao Yu; Zhengxian Wu; Zhuohong Chen; Hangrui Xu; Zirui Liao; Xiangwen Deng; Zhifang Liu; Senyuan Shi; Haoqian Wang
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** In the unsupervised self-evolution of Multimodal Large Language Models, the quality of feedback signals during post-training is pivotal for stable and effective learning. However, existing self-evolution methods predominantly rely on majority voting to select the most frequent output as the pseudo-golden answer, which may stem from the model's intrinsic biases rather than guaranteeing the objective correctness of the reasoning paths. To counteract the degradation, we propose Continuous Softened Retracing reSampling (CSRS) in MLLM self-evolution. Specifically, we introduce a Retracing Re-inference Mechanism (RRM) that the model re-inferences from anchor points to expand the exploration of long-tail reasoning paths. Simultaneously, we propose Softened Frequency Reward (SFR), which replaces binary rewards with continuous signals, calibrating reward based on the answers' frequency across sampled reasoning sets. Furthermore, incorporated with Visual Semantic Perturbation (VSP), CSRS ensures the model prioritizes mathematical logic over visual superficiality. Experimental results demonstrate that CSRS significantly enhances the reasoning performance of Qwen2.5-VL-7B on benchmarks such as MathVision. We achieve state-of-the-art (SOTA) results in unsupervised self-evolution on geometric tasks. Our code is avaible at this https URL.
>
---
#### [replaced 076] SimpleProc: Fully Procedural Synthetic Data from Simple Rules for Multi-View Stereo
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04925](https://arxiv.org/pdf/2604.04925)**

> **作者:** Zeyu Ma; Alexander Raistrick; Jia Deng
>
> **摘要:** In this paper, we explore the design space of procedural rules for multi-view stereo (MVS). We demonstrate that we can generate effective training data using SimpleProc: a new, fully procedural generator driven by a very small set of rules using Non-Uniform Rational Basis Splines (NURBS), as well as basic displacement and texture patterns. At a modest scale of 8,000 images, our approach achieves superior results compared to manually curated images (at the same scale) sourced from games and real-world objects. When scaled to 352,000 images, our method yields performance comparable to--and in several benchmarks, exceeding--models trained on over 692,000 manually curated images. The source code and the data are available at this https URL.
>
---
#### [replaced 077] Motion Focus Recognition in Fast-Moving Egocentric Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.07154](https://arxiv.org/pdf/2601.07154)**

> **作者:** Si-En Hong; James Tribble; Alexander Lake; Hao Wang; Chaoyi Zhou; Ashish Bastola; Siyu Huang; Eisa Chaudhary; Brian Canada; Ismahan Arslan-Ari; Abolfazl Razi
>
> **摘要:** From Vision-Language-Action (VLA) systems to robotics, existing egocentric datasets primarily focus on action recognition tasks, while largely overlooking the inherent role of motion analysis in sports and other fast-movement scenarios. To bridge this gap, we propose a real-time motion focus recognition method that estimates the subject's locomotion intention from any egocentric video. We leverage the foundation model for camera pose estimation and introduce system-level optimizations to enable efficient and scalable inference. Evaluated on a collected egocentric action dataset, our method achieves real-time performance with manageable memory consumption through a sliding batch inference strategy. This work makes motion-centric analysis practical for edge deployment and offers a complementary perspective to existing egocentric studies on sports and fast-movement activities.
>
---
#### [replaced 078] Spatial-Conditioned Reasoning in Long-Egocentric Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.18100](https://arxiv.org/pdf/2601.18100)**

> **作者:** James Tribble; Hao Wang; Si-En Hong; Chaoyi Zhou; Ashish Bastola; Siyu Huang; Abolfazl Razi
>
> **摘要:** Long-horizon egocentric video presents significant challenges for visual navigation due to viewpoint drift and the absence of persistent geometric context. Although recent vision-language models perform well on image and short-video reasoning, their spatial reasoning capability in long egocentric sequences remains limited. In this work, we study how explicit spatial signals influence VLM-based video understanding without modifying model architectures or inference procedures. We introduce Sanpo-D, a fine-grained re-annotation of the Google Sanpo dataset, and benchmark multiple VLMs on navigation-oriented spatial queries. To examine input-level inductive bias, we further fuse depth maps with RGB frames and evaluate their impact on spatial reasoning. Our results reveal a trade-off between general-purpose accuracy and spatial specialization, showing that depth-aware and spatially grounded representations can improve performance on safety-critical tasks such as pedestrian and obstruction detection.
>
---
#### [replaced 079] SDesc3D: Towards Layout-Aware 3D Indoor Scene Generation from Short Descriptions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.01972](https://arxiv.org/pdf/2604.01972)**

> **作者:** Jie Feng; Jiawei Shen; Junjia Huang; Junpeng Zhang; Mingtao Feng; Weisheng Dong; Guanbin Li
>
> **摘要:** 3D indoor scene generation conditioned on short textual descriptions provides a promising avenue for interactive 3D environment construction without the need for labor-intensive layout specification. Despite recent progress in text-conditioned 3D scene generation, existing works suffer from poor physical plausibility and insufficient detail richness in such semantic condensation cases, largely due to their reliance on explicit semantic cues about compositional objects and their spatial relationships. This limitation highlights the need for enhanced 3D reasoning capabilities, particularly in terms of prior integration and spatial anchoring. Motivated by this, we propose SDesc3D, a short-text conditioned 3D indoor scene generation framework, that leverages multi-view structural priors and regional functionality implications to enable 3D layout reasoning under sparse textual guidance. Specifically, we introduce a Multi-view scene prior augmentation that enriches underspecified textual inputs with aggregated multi-view structural knowledge, shifting from inaccessible semantic relation cues to multi-view relational prior aggregation. Building on this, we design a Functionality-aware layout grounding, employing regional functionality grounding for implicit spatial anchors and conducting hierarchical layout reasoning to enhance scene organization and semantic plausibility. Furthermore, an Iterative reflection-rectification scheme is employed for progressive structural plausibility refinement via self-rectification. Extensive experiments show that our method outperforms existing approaches on short-text conditioned 3D indoor scene generation. Code will be publicly available.
>
---
#### [replaced 080] D-Garment: Physically Grounded Latent Diffusion for Dynamic Garment Deformations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.03468](https://arxiv.org/pdf/2504.03468)**

> **作者:** Antoine Dumoulin; Adnane Boukhayma; Laurence Boissieux; Bharath Bhushan Damodaran; Pierre Hellier; Stefanie Wuhrer
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** We present a method to dynamically deform 3D garments, in the form of a 3D polygon mesh, based on body shape, motion, and physical cloth material properties. Considering physical cloth properties allows to learn a physically grounded model, with the advantage of being more accurate in terms of physically inspired metrics such as strain or curvature. Existing work studies pose-dependent garment modeling to generate garment deformations from example data, and possibly data-driven dynamic cloth simulation to generate realistic garments in motion. We propose D-Garment, a learning-based approach trained on new data generated with a physics-based simulator. Compared to prior work, our 3D generative model learns garment deformations conditioned by physical material properties, which allows to model loose cloth geometry, especially for large deformations and dynamic wrinkles driven by body motion. Furthermore, the model can be efficiently fitted to observations captured using vision sensors such as 3D point clouds. We leverage the capability of diffusion models to learn flexible and powerful generative priors by modeling the 3D garment in a 2D parameter space independently from the mesh resolution. This representation allows to learn a template-specific latent diffusion model. This allows to condition global and local geometry with body and cloth material information. We quantitatively and qualitatively evaluate D-Garment on both simulations and data captured with a multi-view acquisition platform. Compared to recent baselines, our method is more realistic and accurate in terms of shape similarity and physical validity metrics. Code and data are available for research purposes at this https URL
>
---
#### [replaced 081] Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.01986](https://arxiv.org/pdf/2509.01986)**

> **作者:** Ziyun Zeng; David Junhao Zhang; Wei Li; Mike Zheng Shou
>
> **备注:** ICLR 2026 Camera Ready Version; Add more discussions and fix typos
>
> **摘要:** In recent years, integrating multimodal understanding and generation into a single unified model has emerged as a promising paradigm. While this approach achieves strong results in text-to-image (T2I) generation, it still struggles with precise image editing. We attribute this limitation to an imbalanced division of responsibilities. The understanding module primarily functions as a translator that encodes user instructions into semantic conditions, while the generation module must simultaneously act as designer and painter, inferring the original layout, identifying the target editing region, and rendering the new content. This imbalance is counterintuitive because the understanding module is typically trained with several times more data on complex reasoning tasks than the generation module. To address this issue, we introduce Draw-In-Mind (DIM), a dataset comprising two complementary subsets: (i) DIM-T2I, containing 14M long-context image-text pairs to enhance complex instruction comprehension; and (ii) DIM-Edit, consisting of 233K chain-of-thought imaginations generated by GPT-4o, serving as explicit design blueprints for image edits. We connect a frozen Qwen2.5-VL-3B with a trainable SANA1.5-1.6B via a lightweight two-layer MLP, and train it on the proposed DIM dataset, resulting in DIM-4.6B-T2I/Edit. Despite its modest parameter scale, DIM-4.6B-Edit achieves SOTA or competitive performance on the ImgEdit and GEdit-Bench benchmarks, outperforming much larger models such as UniWorld-V1 and Step1X-Edit. These findings demonstrate that explicitly assigning the design responsibility to the understanding module provides significant benefits for image editing. Our dataset and models are available at this https URL.
>
---
#### [replaced 082] Machine Unlearning in the Era of Quantum Machine Learning: An Empirical Study
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19253](https://arxiv.org/pdf/2512.19253)**

> **作者:** Carla Crivoi; Radu Tudor Ionescu
>
> **备注:** Accepted at ICPR 2026
>
> **摘要:** We present the first empirical study of machine unlearning (MU) in hybrid quantum-classical neural networks. While MU has been extensively explored in classical deep learning, its behavior within variational quantum circuits (VQCs) and quantum-augmented architectures remains largely unexplored. First, we adapt a broad suite of unlearning methods to quantum settings, including gradient-based, distillation-based, regularization-based and certified techniques. Second, we introduce two new unlearning strategies tailored to hybrid models. Experiments across Iris, MNIST, and Fashion-MNIST, under both subset removal and full-class deletion, reveal that quantum models can support effective unlearning, but outcomes depend strongly on circuit depth, entanglement structure, and task complexity. Shallow VQCs display high intrinsic stability with minimal memorization, whereas deeper hybrid models exhibit stronger trade-offs between utility, forgetting strength, and alignment with retrain oracle. We find that certain methods, e.g. EU-k, LCA, and Certified Unlearning, consistently provide the best balance across metrics. These findings establish baseline empirical insights into quantum machine unlearning and highlight the need for quantum-aware algorithms and theoretical guarantees, as quantum machine learning systems continue to expand in scale and capability. We publicly release our code at: this https URL.
>
---
#### [replaced 083] PCSR: Pseudo-label Consistency-Guided Sample Refinement for Noisy Correspondence Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.15623](https://arxiv.org/pdf/2509.15623)**

> **作者:** Zhuoyao Liu; Yang Liu; Wentao Feng; Shudong Huang
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Cross-modal retrieval aims to align different modalities via semantic similarity. However, existing methods often assume that image-text pairs are perfectly aligned, overlooking Noisy Correspondences in real data. These misaligned pairs misguide similarity learning and degrade retrieval performance. Previous methods often rely on coarse-grained categorizations that simply divide data into clean and noisy samples, overlooking the intrinsic diversity within noisy instances. Moreover, they typically apply uniform training strategies regardless of sample characteristics, resulting in suboptimal sample utilization for model optimization. To address the above challenges, we introduce a novel framework, called Pseudo-label Consistency-Guided Sample Refinement (PCSR), which enhances correspondence reliability by explicitly dividing samples based on pseudo-label consistency. Specifically, we first employ a confidence-based estimation to distinguish clean and noisy pairs, then refine the noisy pairs via pseudo-label consistency to uncover structurally distinct subsets. We further proposed a Pseudo-label Consistency Score (PCS) to quantify prediction stability, enabling the separation of ambiguous and refinable samples within noisy pairs. Accordingly, we adopt Adaptive Pair Optimization (APO), where ambiguous samples are optimized with robust loss functions and refinable ones are enhanced via text replacement during training. Extensive experiments on CC152K, MS-COCO and Flickr30K validate the effectiveness of our method in improving retrieval robustness under noisy supervision.
>
---
#### [replaced 084] Toward Personalized Darts Training: A Data-Driven Framework Based on Skeleton-Based Biomechanical Analysis and Motion Modeling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.01130](https://arxiv.org/pdf/2604.01130)**

> **作者:** Zhantao Chen; Dongyi He; Jin Fang; Xi Chen; Yishuo Liu; Xiaozhen Zhong; Xuejun Hu
>
> **摘要:** As sports training becomes more data-driven, traditional dart coaching based mainly on experience and visual observation is increasingly inadequate for high-precision, goal-oriented movements. Although prior studies have highlighted the importance of release parameters, joint motion, and coordination in dart throwing, most quantitative methods still focus on local variables, single-release metrics, or static template matching. These approaches offer limited support for personalized training and often overlook useful movement variability. This paper presents a data-driven dart training assistance system. The system creates a closed-loop framework spanning motion capture, feature modeling, and personalized feedback. Dart-throwing data were collected in markerless conditions using a Kinect 2.0 depth sensor and an optical camera. Eighteen kinematic features were extracted from four biomechanical dimensions: three-link coordination, release velocity, multi-joint angular configuration, and postural stability. Two modules were developed: a personalized optimal throwing trajectory model that combines historical high-quality samples with the minimum jerk criterion, and a motion deviation diagnosis and recommendation model based on z-scores and hierarchical logic. A total of 2,396 throwing samples from professional and non-professional athletes were collected. Results show that the system generates smooth personalized reference trajectories consistent with natural human movement. Case studies indicate that it can detect poor trunk stability, abnormal elbow displacement, and imbalanced velocity control, then provide targeted recommendations. The framework shifts dart evaluation from deviation from a uniform standard to deviation from an individual's optimal control range, improving personalization and interpretability for darts training and other high-precision target sports.
>
---
#### [replaced 085] Linearized Coupling Flow with Shortcut Constraints for One-Step Face Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.03648](https://arxiv.org/pdf/2603.03648)**

> **作者:** Xiaohui Sun; Hanlin Wu
>
> **摘要:** Face restoration can be formulated as a continuous-time transformation between image distributions via Flow Matching (FM). However, standard FM typically employs independent coupling, ignoring the statistical correlation between low-quality (LQ) and high-quality (HQ) data. This leads to intersecting trajectories and high velocity-field curvature, requiring multi-step integration. We propose Shortcut-constrained Coupling Flow for Face Restoration (SCFlowFR) to address these challenges. By establishing a data-dependent coupling, we explicitly model the LQ-HQ dependency to minimize path crossovers and promote near-linear probability flow. Furthermore, we employ a conditional mean estimator to refine the source distribution's anchor, effectively tightening the transport cost and stabilizing the velocity field. To ensure stable one-step inference, a shortcut constraint is introduced to supervise average velocities over arbitrary intervals, mitigating discretization bias in large-step updates. SCFlowFR achieves state-of-the-art one-step restoration, providing a superior trade-off between perceptual fidelity and computational efficiency.
>
---
#### [replaced 086] LoFT: Parameter-Efficient Fine-Tuning for Long-tailed Semi-Supervised Learning in Open-World Scenarios
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09926](https://arxiv.org/pdf/2509.09926)**

> **作者:** Zhiyuan Huang; Jiahao Chen; Bing Su
>
> **摘要:** Long-tailed semi-supervised learning (LTSSL) presents a formidable challenge where models must overcome the scarcity of tail samples while mitigating the noise from unreliable pseudo-labels. Most prior LTSSL methods are designed to train models from scratch, which often leads to issues such as overconfidence and low-quality pseudo-labels. To address this problem, we first theoretically prove that utilizing a foundation model significantly reduces the hypothesis complexity, which tightens the generalization bound and in turn minimizes the Balanced Posterior Error (BPE). Furthermore, we demonstrate that the feature compactness of foundation models strictly compresses the acceptance region for outliers, providing a geometric guarantee for robustness. Motivated by these theoretical insights, we extend LTSSL into the foundation model fine-tuning paradigm and propose a novel framework: LoFT (Long-tailed semi-supervised learning via parameter-efficient Fine-Tuning). Furthermore, we explore a more practical setting by investigating semi-supervised learning under open-world conditions, where the unlabeled data may include out-of-distribution (OOD) this http URL handle this problem, we propose LoFT-OW (LoFT under Open-World scenarios) to improve the discriminative ability. Experimental results on multiple benchmarks demonstrate that our method achieves superior performance. Code is available: this https URL
>
---
#### [replaced 087] AugLift: Depth-Aware Input Reparameterization Improves Domain Generalization in 2D-to-3D Pose Lifting
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.07112](https://arxiv.org/pdf/2508.07112)**

> **作者:** Nikolai Warner; Wenjin Zhang; Hamid Badiozamani; Irfan Essa; Apaar Sadhwani
>
> **备注:** Preprint. Under review
>
> **摘要:** Lifting-based 3D human pose estimation infers 3D joints from 2D keypoints but generalizes poorly because $(x,y)$ coordinates alone are an ill-posed, sparse representation that discards geometric information modern foundation models can recover. We propose \emph{AugLift}, which changes the representation format of lifting from 2D coordinates to a 6D geometric descriptor via two modules: (1) an \emph{Uncertainty-Aware Depth Descriptor} (UADD) -- a compact tuple $(c, d, d_{\min}, d_{\max})$ extracted from a confidence-scaled neighborhood of an off-the-shelf monocular depth map -- and (2) a scale normalization component that handles train/test distance shifts. AugLift requires no new sensors, no new data collection, and no architectural changes beyond widening the input layer; because it operates at the representation level, it is composable with any lifting architecture or domain generalization technique. In the detection setting, AugLift reduces cross-dataset MPJPE by $10.1$% on average across four datasets and four lifting architectures while improving in-distribution accuracy by $4.0$%; post-hoc analysis shows gains concentrate on novel poses and occluded joints. In the ground-truth 2D setting, combining AugLift with PoseAug's differentiable domain generalization achieves state-of-the-art cross-dataset performance ($62.4$\,mm on 3DHP, $92.6$\,mm on 3DPW; $14.5$% and $22.2$% over PoseAug), demonstrating that foundation-model depth provides genuine geometric signal complementary to explicit 3D augmentation. Code will be made publicly available.
>
---
#### [replaced 088] AstraNav-World: World Model for Foresight Control and Consistency
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21714](https://arxiv.org/pdf/2512.21714)**

> **作者:** Jintao Chen; Junjun Hu; Haochen Bai; Minghua Luo; Xinda Xue; Botao Ren; Chengyu Bai; Shichao Xie; Ziyi Chen; Fei Liu; Zedong Chu; Xiaolong Wu; Mu Xu; Shanghang Zhang
>
> **摘要:** Embodied navigation in open, dynamic environments demands accurate foresight of how the world will evolve and how actions will unfold over time. We propose AstraNav-World, an end-to-end world model that jointly reasons about future visual states and action sequences within a unified probabilistic framework. Our framework integrates a diffusion-based video generator with a vision-language policy, enabling synchronized rollouts where predicted scenes and planned actions are updated simultaneously. Training optimizes two complementary objectives: generating action-conditioned multi-step visual predictions and deriving trajectories conditioned on those predicted visuals. This bidirectional constraint makes visual predictions executable and keeps decisions grounded in physically consistent, task-relevant futures, mitigating cumulative errors common in decoupled "envision-then-plan" pipelines. Experiments across diverse embodied navigation benchmarks show improved trajectory accuracy and higher success rates. Ablations confirm the necessity of tight vision-action coupling and unified training, with either branch removal degrading both prediction quality and policy reliability. In real-world testing, AstraNav-World demonstrated exceptional zero-shot capabilities, adapting to previously unseen scenarios without any real-world fine-tuning. These results suggest that AstraNav-World captures transferable spatial understanding and planning-relevant navigation dynamics, rather than merely overfitting to simulation-specific data distribution. Overall, by unifying foresight vision and control within a single generative model, we move closer to reliable, interpretable, and general-purpose embodied agents that operate robustly in open-ended real-world settings.
>
---
#### [replaced 089] Free-Grained Hierarchical Visual Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.14737](https://arxiv.org/pdf/2510.14737)**

> **作者:** Seulki Park; Zilin Wang; Stella X. Yu
>
> **备注:** Accepted to CVPR 2026. 29 pages
>
> **摘要:** Hierarchical image recognition seeks to predict class labels along a semantic taxonomy, from broad categories to specific ones, typically under the tidy assumption that every training image is fully annotated along its taxonomy path. Reality is messier: A distant bird may be labeled only bird, while a clear close-up may justify bald eagle. We introduce free-grain training, where labels may appear at any level of the taxonomy and models must learn consistent hierarchical predictions from incomplete, mixed-granularity supervision. We build benchmark datasets with varying label granularity and show that existing hierarchical methods deteriorate sharply in this setting. To make up for missing supervision, we propose two simple solutions: One adds broad text-based supervision that captures visual attributes, and the other treats missing labels at specific taxonomy levels as a semi-supervised learning problem. We also study free-grained inference, where the model chooses how deep to predict, returning a reliable coarse label when a fine-grained one is uncertain. Together, our task, datasets, and methods move hierarchical recognition closer to the way labels arise in the real world.
>
---
#### [replaced 090] A Robust 3D Registration Method via Simultaneous Inlier Identification and Model Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2008.01574](https://arxiv.org/pdf/2008.01574)**

> **作者:** Xianyun Qian; Fei Wen; Peilin Liu
>
> **摘要:** Robust 3D registration is a fundamental problem in computer vision and robotics, where the goal is to estimate the geometric transformation between two sets of measurements in the presence of noise, mismatches, and extreme outlier contamination. Existing robust registration methods are mainly built on either maximum consensus (MC) estimators, which first identify inliers and then estimate the transformation, or M-estimators, which directly optimize a robust objective. In this work, we revisit a truncated-loss based formulation for simultaneous inlier identification and model estimation (SIME) and study it in the context of 3D registration. We show that, compared with MC-based robust fitting, SIME can achieve a lower fitting residual because it incorporates residual magnitudes into the inlier selection process. To solve the resulting nonconvex problem, we develop an alternating minimization (AM) algorithm, and further propose an AM method embedded with semidefinite relaxation (SDR) to alleviate the difficulty caused by the binary inlier variables. We instantiate the proposed framework for 3D rotation search and rigid point-set registration using quaternion-based formulations. Experimental results on both simulated and real-world registration tasks demonstrate that the proposed methods compare favorably with strong baseline solvers, especially in challenging cases with high noise levels and many outliers.
>
---
