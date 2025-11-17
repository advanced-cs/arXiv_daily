# 计算机视觉 cs.CV

- **最新发布 135 篇**

- **更新 91 篇**

## 最新发布

#### [new 001] YCB-Ev SD: Synthetic event-vision dataset for 6DoF object pose estimation
- **分类: cs.CV**

- **简介: 论文提出YCB-Ev SD合成数据集，解决事件视觉中6DoF物体姿态估计缺乏资源的问题。该数据集包含50,000个34ms事件序列，基于PBR场景生成。实验表明，时间表面线性衰减和双通道极性编码性能最优。**

- **链接: [https://arxiv.org/pdf/2511.11344v1](https://arxiv.org/pdf/2511.11344v1)**

> **作者:** Pavel Rojtberg; Julius Kühn
>
> **摘要:** We introduce YCB-Ev SD, a synthetic dataset of event-camera data at standard definition (SD) resolution for 6DoF object pose estimation. While synthetic data has become fundamental in frame-based computer vision, event-based vision lacks comparable comprehensive resources. Addressing this gap, we present 50,000 event sequences of 34 ms duration each, synthesized from Physically Based Rendering (PBR) scenes of YCB-Video objects following the Benchmark for 6D Object Pose (BOP) methodology. Our generation framework employs simulated linear camera motion to ensure complete scene coverage, including background activity. Through systematic evaluation of event representations for CNN-based inference, we demonstrate that time-surfaces with linear decay and dual-channel polarity encoding achieve superior pose estimation performance, outperforming exponential decay and single-channel alternatives by significant margins. Our analysis reveals that polarity information contributes most substantially to performance gains, while linear temporal encoding preserves critical motion information more effectively than exponential decay. The dataset is provided in a structured format with both raw event streams and precomputed optimal representations to facilitate immediate research use and reproducible benchmarking. The dataset is publicly available at https://huggingface.co/datasets/paroj/ycbev_sd.
>
---
#### [new 002] DEFT-LLM: Disentangled Expert Feature Tuning for Micro-Expression Recognition
- **分类: cs.CV; cs.HC**

- **简介: 论文聚焦微表情识别任务，解决静态-动态线索纠缠和文本-运动语义鸿沟问题。提出DEFT-LLM，通过Uni-MER数据集和三专家架构实现运动语义对齐，提升精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.10948v1](https://arxiv.org/pdf/2511.10948v1)**

> **作者:** Ren Zhang; Huilai Li; Chao qi; Guoliang Xu; Tianyu Zhou; Wei wei; Jianqin Yin
>
> **摘要:** Micro expression recognition (MER) is crucial for inferring genuine emotion. Applying a multimodal large language model (MLLM) to this task enables spatio-temporal analysis of facial motion and provides interpretable descriptions. However, there are still two core challenges: (1) The entanglement of static appearance and dynamic motion cues prevents the model from focusing on subtle motion; (2) Textual labels in existing MER datasets do not fully correspond to underlying facial muscle movements, creating a semantic gap between text supervision and physical motion. To address these issues, we propose DEFT-LLM, which achieves motion semantic alignment by multi-expert disentanglement. We first introduce Uni-MER, a motion-driven instruction dataset designed to align text with local facial motion. Its construction leverages dual constraints from optical flow and Action Unit (AU) labels to ensure spatio-temporal consistency and reasonable correspondence to the movements. We then design an architecture with three experts to decouple facial dynamics into independent and interpretable representations (structure, dynamic textures, and motion-semantics). By integrating the instruction-aligned knowledge from Uni-MER into DEFT-LLM, our method injects effective physical priors for micro expressions while also leveraging the cross modal reasoning ability of large language models, thus enabling precise capture of subtle emotional cues. Experiments on multiple challenging MER benchmarks demonstrate state-of-the-art performance, as well as a particular advantage in interpretable modeling of local facial motion.
>
---
#### [new 003] Toward Gaze Target Detection of Young Autistic Children
- **分类: cs.CV; cs.AI**

- **简介: 该论文解决自闭症儿童注视目标检测任务中的类不平衡问题（因儿童减少注视人脸）。研究者构建了首个AGT数据集，提出社会感知粗到精（SACF）框架，利用双路径架构和上下文感知模块，显著提升人脸注视检测性能。**

- **链接: [https://arxiv.org/pdf/2511.11244v1](https://arxiv.org/pdf/2511.11244v1)**

> **作者:** Shijian Deng; Erin E. Kosloski; Siva Sai Nagender Vasireddy; Jia Li; Randi Sierra Sherwood; Feroz Mohamed Hatha; Siddhi Patel; Pamela R Rollins; Yapeng Tian
>
> **备注:** AAAI 2026 Artificial Intelligence for Social Impact Track
>
> **摘要:** The automatic detection of gaze targets in autistic children through artificial intelligence can be impactful, especially for those who lack access to a sufficient number of professionals to improve their quality of life. This paper introduces a new, real-world AI application for gaze target detection in autistic children, which predicts a child's point of gaze from an activity image. This task is foundational for building automated systems that can measure joint attention-a core challenge in Autism Spectrum Disorder (ASD). To facilitate the study of this challenging application, we collected the first-ever Autism Gaze Target (AGT) dataset. We further propose a novel Socially Aware Coarse-to-Fine (SACF) gaze detection framework that explicitly leverages the social context of a scene to overcome the class imbalance common in autism datasets-a consequence of autistic children's tendency to show reduced gaze to faces. It utilizes a two-pathway architecture with expert models specialized in social and non-social gaze, guided by a context-awareness gate module. The results of our comprehensive experiments demonstrate that our framework achieves new state-of-the-art performance for gaze target detection in this population, significantly outperforming existing methods, especially on the critical minority class of face-directed gaze.
>
---
#### [new 004] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对4D Flow MRI超分辨率任务，解决现有物理信息神经网络训练慢、需为每个患者单独训练的问题。提出PINGS-X框架，通过轴对齐高斯表示和归一化溅射，显著减少训练时间并提升精度。**

- **链接: [https://arxiv.org/pdf/2511.11048v1](https://arxiv.org/pdf/2511.11048v1)**

> **作者:** Sun Jo; Seok Young Hong; JinHyun Kim; Seungmin Kang; Ahjin Choi; Don-Gwan An; Simon Song; Je Hyeong Hong
>
> **备注:** Accepted at AAAI 2026. Supplementary material included after references. 27 pages, 21 figures, 11 tables
>
> **摘要:** 4D flow magnetic resonance imaging (MRI) is a reliable, non-invasive approach for estimating blood flow velocities, vital for cardiovascular diagnostics. Unlike conventional MRI focused on anatomical structures, 4D flow MRI requires high spatiotemporal resolution for early detection of critical conditions such as stenosis or aneurysms. However, achieving such resolution typically results in prolonged scan times, creating a trade-off between acquisition speed and prediction accuracy. Recent studies have leveraged physics-informed neural networks (PINNs) for super-resolution of MRI data, but their practical applicability is limited as the prohibitively slow training process must be performed for each patient. To overcome this limitation, we propose PINGS-X, a novel framework modeling high-resolution flow velocities using axes-aligned spatiotemporal Gaussian representations. Inspired by the effectiveness of 3D Gaussian splatting (3DGS) in novel view synthesis, PINGS-X extends this concept through several non-trivial novel innovations: (i) normalized Gaussian splatting with a formal convergence guarantee, (ii) axes-aligned Gaussians that simplify training for high-dimensional data while preserving accuracy and the convergence guarantee, and (iii) a Gaussian merging procedure to prevent degenerate solutions and boost computational efficiency. Experimental results on computational fluid dynamics (CFD) and real 4D flow MRI datasets demonstrate that PINGS-X substantially reduces training time while achieving superior super-resolution accuracy. Our code and datasets are available at https://github.com/SpatialAILab/PINGS-X.
>
---
#### [new 005] VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: VisMem框架解决视觉语言模型在复杂任务中的视觉处理瓶颈问题，通过动态潜在视觉记忆（短时感知保留与长时语义巩固），显著提升视觉理解、推理和生成性能，平均提升11.8%。**

- **链接: [https://arxiv.org/pdf/2511.11007v1](https://arxiv.org/pdf/2511.11007v1)**

> **作者:** Xinlei Yu; Chengming Xu; Guibin Zhang; Zhangquan Chen; Yudong Zhang; Yongbo He; Peng-Tao Jiang; Jiangning Zhang; Xiaobin Hu; Shuicheng Yan
>
> **摘要:** Despite the remarkable success of Vision-Language Models (VLMs), their performance on a range of complex visual tasks is often hindered by a "visual processing bottleneck": a propensity to lose grounding in visual evidence and exhibit a deficit in contextualized visual experience during prolonged generation. Drawing inspiration from human cognitive memory theory, which distinguishes short-term visually-dominant memory and long-term semantically-dominant memory, we propose VisMem, a cognitively-aligned framework that equips VLMs with dynamic latent vision memories, a short-term module for fine-grained perceptual retention and a long-term module for abstract semantic consolidation. These memories are seamlessly invoked during inference, allowing VLMs to maintain both perceptual fidelity and semantic consistency across thinking and generation. Extensive experiments across diverse visual benchmarks for understanding, reasoning, and generation reveal that VisMem delivers a significant average performance boost of 11.8% relative to the vanilla model and outperforms all counterparts, establishing a new paradigm for latent-space memory enhancement. The code will be available: https://github.com/YU-deep/VisMem.git.
>
---
#### [new 006] WEAVE: Unleashing and Benchmarking the In-context Interleaved Comprehension and Generation
- **分类: cs.CV**

- **简介: WEAVE论文解决多模态模型在图像理解、编辑与生成中的多轮上下文依赖问题，提出WEAVE-100k数据集（10万样本）和WEAVEBench基准（100任务），评估模型视觉记忆与推理能力。**

- **链接: [https://arxiv.org/pdf/2511.11434v1](https://arxiv.org/pdf/2511.11434v1)**

> **作者:** Wei Chow; Jiachun Pan; Yongyuan Liang; Mingze Zhou; Xue Song; Liyu Jia; Saining Zhang; Siliang Tang; Juncheng Li; Fengda Zhang; Weijia Wu; Hanwang Zhang; Tat-Seng Chua
>
> **摘要:** Recent advances in unified multimodal models (UMMs) have enabled impressive progress in visual comprehension and generation. However, existing datasets and benchmarks focus primarily on single-turn interactions, failing to capture the multi-turn, context-dependent nature of real-world image creation and editing. To address this gap, we present WEAVE, the first suite for in-context interleaved cross-modality comprehension and generation. Our suite consists of two complementary parts. WEAVE-100k is a large-scale dataset of 100K interleaved samples spanning over 370K dialogue turns and 500K images, covering comprehension, editing, and generation tasks that require reasoning over historical context. WEAVEBench is a human-annotated benchmark with 100 tasks based on 480 images, featuring a hybrid VLM judger evaluation framework based on both the reference image and the combination of the original image with editing instructions that assesses models' abilities in multi-turn generation, visual memory, and world-knowledge reasoning across diverse domains. Experiments demonstrate that training on WEAVE-100k enables vision comprehension, image editing, and comprehension-generation collaboration capabilities. Furthermore, it facilitates UMMs to develop emergent visual-memory capabilities, while extensive evaluations on WEAVEBench expose the persistent limitations and challenges of current approaches in multi-turn, context-aware image generation and editing. We believe WEAVE provides a view and foundation for studying in-context interleaved comprehension and generation for multi-modal community.
>
---
#### [new 007] Questioning the Stability of Visual Question Answering
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究视觉问答模型（VQA）的稳定性，针对微小语义扰动（如像素偏移、文本重写）进行系统评估。发现现代VLMs高度敏感，稳定样本更可能正确，并提出用小模型预测大模型正确性。**

- **链接: [https://arxiv.org/pdf/2511.11206v1](https://arxiv.org/pdf/2511.11206v1)**

> **作者:** Amir Rosenfeld; Neta Glazer; Ethan Fetaya
>
> **摘要:** Visual Language Models (VLMs) have achieved remarkable progress, yet their reliability under small, meaning-preserving input changes remains poorly understood. We present the first large-scale, systematic study of VLM robustness to benign visual and textual perturbations: pixel-level shifts, light geometric transformations, padded rescaling, paraphrasing, and multilingual rewrites that do not alter the underlying semantics of an image-question pair. Across a broad set of models and datasets, we find that modern VLMs are highly sensitive to such minor perturbations: a substantial fraction of samples change their predicted answer under at least one visual or textual modification. We characterize how this instability varies across perturbation types, question categories, and models, revealing that even state-of-the-art systems (e.g., GPT-4o, Gemini 2.0 Flash) frequently fail under shifts as small as a few pixels or harmless rephrasings. We further show that sample-level stability serves as a strong indicator of correctness: stable samples are consistently far more likely to be answered correctly. Leveraging this, we demonstrate that the stability patterns of small, accessible open-source models can be used to predict the correctness of much larger closed-source models with high precision. Our findings expose a fundamental fragility in current VLMs and highlight the need for robustness evaluations that go beyond adversarial perturbations, focusing instead on invariances that models should reliably uphold.
>
---
#### [new 008] Sat2RealCity: Geometry-Aware and Appearance-Controllable 3D Urban Generation from Satellite Imagery
- **分类: cs.CV**

- **简介: 论文提出Sat2RealCity，解决3D城市生成中依赖大尺度3D资产和缺乏真实外观的问题。通过OSM空间先验、外观可控建模及MLLM语义引导，实现几何准确、外观真实的卫星图像驱动城市生成。**

- **链接: [https://arxiv.org/pdf/2511.11470v1](https://arxiv.org/pdf/2511.11470v1)**

> **作者:** Yijie Kang; Xinliang Wang; Zhenyu Wu; Yifeng Shi; Hailong Zhu
>
> **摘要:** Recent advances in generative modeling have substantially enhanced 3D urban generation, enabling applications in digital twins, virtual cities, and large-scale simulations. However, existing methods face two key challenges: (1) the need for large-scale 3D city assets for supervised training, which are difficult and costly to obtain, and (2) reliance on semantic or height maps, which are used exclusively for generating buildings in virtual worlds and lack connection to real-world appearance, limiting the realism and generalizability of generated cities. To address these limitations, we propose Sat2RealCity, a geometry-aware and appearance-controllable framework for 3D urban generation from real-world satellite imagery. Unlike previous city-level generation methods, Sat2RealCity builds generation upon individual building entities, enabling the use of rich priors and pretrained knowledge from 3D object generation while substantially reducing dependence on large-scale 3D city assets. Specifically, (1) we introduce the OSM-based spatial priors strategy to achieve interpretable geometric generation from spatial topology to building instances; (2) we design an appearance-guided controllable modeling mechanism for fine-grained appearance realism and style control; and (3) we construct an MLLM-powered semantic-guided generation pipeline, bridging semantic interpretation and geometric reconstruction. Extensive quantitative and qualitative experiments demonstrate that Sat2RealCity significantly surpasses existing baselines in structural consistency and appearance realism, establishing a strong foundation for real-world aligned 3D urban content creation. The code will be released soon.
>
---
#### [new 009] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对CIL任务，解决CLIP应用中额外模块引发的遗忘和模态融合不足问题。提出BOFA框架，仅在CLIP桥接层适应，通过正交低秩融合防止遗忘，并用跨模态混合原型提升性能。**

- **链接: [https://arxiv.org/pdf/2511.11421v1](https://arxiv.org/pdf/2511.11421v1)**

> **作者:** Lan Li; Tao Hu; Da-Wei Zhou; Han-Jia Ye; De-Chuan Zhan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Class-Incremental Learning (CIL) aims to continually learn new categories without forgetting previously acquired knowledge. Vision-language models such as CLIP offer strong transferable representations via multi-modal supervision, making them promising for CIL. However, applying CLIP to CIL poses two major challenges: (1) adapting to downstream tasks often requires additional learnable modules, increasing model complexity and susceptibility to forgetting; and (2) while multi-modal representations offer complementary strengths, existing methods have yet to fully realize their potential in effectively integrating visual and textual modalities. To address these issues, we propose BOFA (Bridge-layer Orthogonal Fusion for Adaptation), a novel framework for CIL. BOFA confines all model adaptation exclusively to CLIP's existing cross-modal bridge-layer, thereby adding no extra parameters or inference cost. To prevent forgetting within this layer, it leverages Orthogonal Low-Rank Fusion, a mechanism that constrains parameter updates to a low-rank ``safe subspace" mathematically constructed to be orthogonal to past task features. This ensures stable knowledge accumulation without data replay. Furthermore, BOFA employs a cross-modal hybrid prototype that synergizes stable textual prototypes with visual counterparts derived from our stably adapted bridge-layer, enhancing classification performance. Extensive experiments on standard benchmarks show that BOFA achieves superior accuracy and efficiency compared to existing methods.
>
---
#### [new 010] CATS-V2V: A Real-World Vehicle-to-Vehicle Cooperative Perception Dataset with Complex Adverse Traffic Scenarios
- **分类: cs.CV**

- **简介: 论文提出CATS-V2V数据集，解决V2V协同感知在复杂恶劣交通场景中的数据缺失问题。数据集覆盖10种天气/光照条件、10个地点，含60K帧LiDAR点云、1.26M多视角图像及时间一致的3D标注，支持高精度协同感知研究。**

- **链接: [https://arxiv.org/pdf/2511.11168v1](https://arxiv.org/pdf/2511.11168v1)**

> **作者:** Hangyu Li; Bofeng Cao; Zhaohui Liang; Wuzhen Li; Juyoung Oh; Yuxuan Chen; Shixiao Liang; Hang Zhou; Chengyuan Ma; Jiaxi Liu; Zheng Li; Peng Zhang; KeKe Long; Maolin Liu; Jackson Jiang; Chunlei Yu; Shengxiang Liu; Hongkai Yu; Xiaopeng Li
>
> **摘要:** Vehicle-to-Vehicle (V2V) cooperative perception has great potential to enhance autonomous driving performance by overcoming perception limitations in complex adverse traffic scenarios (CATS). Meanwhile, data serves as the fundamental infrastructure for modern autonomous driving AI. However, due to stringent data collection requirements, existing datasets focus primarily on ordinary traffic scenarios, constraining the benefits of cooperative perception. To address this challenge, we introduce CATS-V2V, the first-of-its-kind real-world dataset for V2V cooperative perception under complex adverse traffic scenarios. The dataset was collected by two hardware time-synchronized vehicles, covering 10 weather and lighting conditions across 10 diverse locations. The 100-clip dataset includes 60K frames of 10 Hz LiDAR point clouds and 1.26M multi-view 30 Hz camera images, along with 750K anonymized yet high-precision RTK-fixed GNSS and IMU records. Correspondingly, we provide time-consistent 3D bounding box annotations for objects, as well as static scenes to construct a 4D BEV representation. On this basis, we propose a target-based temporal alignment method, ensuring that all objects are precisely aligned across all sensor modalities. We hope that CATS-V2V, the largest-scale, most supportive, and highest-quality dataset of its kind to date, will benefit the autonomous driving community in related tasks.
>
---
#### [new 011] Draft and Refine with Visual Experts
- **分类: cs.CV**

- **简介: 解决LVLMs幻觉问题，提出Draft and Refine (DnR)框架。基于视觉利用度度量量化模型对视觉证据依赖，引导视觉专家反馈优化响应，无需重训。在VQA和图像描述任务中提升准确率，减少幻觉。**

- **链接: [https://arxiv.org/pdf/2511.11005v1](https://arxiv.org/pdf/2511.11005v1)**

> **作者:** Sungheon Jeong; Ryozo Masukawa; Jihong Park; Sanggeon Yun; Wenjun Huang; Hanning Chen; Mahdi Imani; Mohsen Imani
>
> **摘要:** While recent Large Vision-Language Models (LVLMs) exhibit strong multimodal reasoning abilities, they often produce ungrounded or hallucinated responses because they rely too heavily on linguistic priors instead of visual evidence. This limitation highlights the absence of a quantitative measure of how much these models actually use visual information during reasoning. We propose Draft and Refine (DnR), an agent framework driven by a question-conditioned utilization metric. The metric quantifies the model's reliance on visual evidence by first constructing a query-conditioned relevance map to localize question-specific cues and then measuring dependence through relevance-guided probabilistic masking. Guided by this metric, the DnR agent refines its initial draft using targeted feedback from external visual experts. Each expert's output (such as boxes or masks) is rendered as visual cues on the image, and the model is re-queried to select the response that yields the largest improvement in utilization. This process strengthens visual grounding without retraining or architectural changes. Experiments across VQA and captioning benchmarks show consistent accuracy gains and reduced hallucination, demonstrating that measuring visual utilization provides a principled path toward more interpretable and evidence-driven multimodal agent systems.
>
---
#### [new 012] Hi-DREAM: Brain Inspired Hierarchical Diffusion for fMRI Reconstruction via ROI Encoder and visuAl Mapping
- **分类: cs.CV; cs.HC**

- **简介: Hi-DREAM解决fMRI重建任务中忽略大脑层次处理的问题。提出脑启发框架，用ROI适配器将fMRI分组为多尺度皮层金字塔，通过深度匹配ControlNet注入尺度提示，实现高效可解释图像重建。**

- **链接: [https://arxiv.org/pdf/2511.11437v1](https://arxiv.org/pdf/2511.11437v1)**

> **作者:** Guowei Zhang; Yun Zhao; Moein Khajehnejad; Adeel Razi; Levin Kuhlmann
>
> **摘要:** Mapping human brain activity to natural images offers a new window into vision and cognition, yet current diffusion-based decoders face a core difficulty: most condition directly on fMRI features without analyzing how visual information is organized across the cortex. This overlooks the brain's hierarchical processing and blurs the roles of early, middle, and late visual areas. We propose Hi-DREAM, a brain-inspired conditional diffusion framework that makes the cortical organization explicit. A region-of-interest (ROI) adapter groups fMRI into early/mid/late streams and converts them into a multi-scale cortical pyramid aligned with the U-Net depth (shallow scales preserve layout and edges; deeper scales emphasize objects and semantics). A lightweight, depth-matched ControlNet injects these scale-specific hints during denoising. The result is an efficient and interpretable decoder in which each signal plays a brain-like role, allowing the model not only to reconstruct images but also to illuminate functional contributions of different visual areas. Experiments on the Natural Scenes Dataset (NSD) show that Hi-DREAM attains state-of-the-art performance on high-level semantic metrics while maintaining competitive low-level fidelity. These findings suggest that structuring conditioning by cortical hierarchy is a powerful alternative to purely data-driven embeddings and provides a useful lens for studying the visual cortex.
>
---
#### [new 013] A Mathematical Framework for AI Singularity: Conditions, Bounds, and Control of Recursive Improvement
- **分类: cs.CV**

- **简介: 论文提出AI奇点的数学框架，解决能力无界增长的条件问题。基于资源约束和物理限制，定义服务包络与临界边界，推导可测试决策规则及安全控制（如功率上限），实现对无界增长的实证验证与预防。**

- **链接: [https://arxiv.org/pdf/2511.10668v1](https://arxiv.org/pdf/2511.10668v1)**

> **作者:** Akbar Anbar Jafari; Cagri Ozcinar; Gholamreza Anbarjafari
>
> **备注:** 41 pages
>
> **摘要:** AI systems improve by drawing on more compute, data, energy, and better training methods. This paper asks a precise, testable version of the "runaway growth" question: under what measurable conditions could capability escalate without bound in finite time, and under what conditions can that be ruled out? We develop an analytic framework for recursive self-improvement that links capability growth to resource build-out and deployment policies. Physical and information-theoretic limits from power, bandwidth, and memory define a service envelope that caps instantaneous improvement. An endogenous growth model couples capital to compute, data, and energy and defines a critical boundary separating superlinear from subcritical regimes. We derive decision rules that map observable series (facility power, IO bandwidth, training throughput, benchmark losses, and spending) into yes/no certificates for runaway versus nonsingular behavior. The framework yields falsifiable tests based on how fast improvement accelerates relative to its current level, and it provides safety controls that are directly implementable in practice, such as power caps, throughput throttling, and evaluation gates. Analytical case studies cover capped-power, saturating-data, and investment-amplified settings, illustrating when the envelope binds and when it does not. The approach is simulation-free and grounded in measurements engineers already collect. Limitations include dependence on the chosen capability metric and on regularity diagnostics; future work will address stochastic dynamics, multi-agent competition, and abrupt architectural shifts. Overall, the results replace speculation with testable conditions and deployable controls for certifying or precluding an AI singularity.
>
---
#### [new 014] Explainable Deep Convolutional Multi-Type Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出MultiTypeFCDD框架，用于可解释的多类型异常检测。解决现有方法无法区分异常类型（如裂缝vs划痕）及需为每类训练单独模型的问题。通过图像级标签生成多通道热图，单框架支持多类别，参数少、推理快，性能媲美复杂模型，适用于资源受限场景。**

- **链接: [https://arxiv.org/pdf/2511.11165v1](https://arxiv.org/pdf/2511.11165v1)**

> **作者:** Alex George; Lyudmila Mihaylova; Sean Anderson
>
> **摘要:** Most explainable anomaly detection methods often identify anomalies but lack the capability to differentiate the type of anomaly. Furthermore, they often require the costly training and maintenance of separate models for each object category. The lack of specificity is a significant research gap, as identifying the type of anomaly (e.g., "Crack" vs. "Scratch") is crucial for accurate diagnosis that facilitates cost-saving operational decisions across diverse application domains. While some recent large-scale Vision-Language Models (VLMs) have begun to address this, they are computationally intensive and memory-heavy, restricting their use in real-time or embedded systems. We propose MultiTypeFCDD, a simple and lightweight convolutional framework designed as a practical alternative for explainable multi-type anomaly detection. MultiTypeFCDD uses only image-level labels to learn and produce multi-channel heatmaps, where each channel is trained to correspond to a specific anomaly type. The model functions as a single, unified framework capable of differentiating anomaly types across multiple object categories, eliminating the need to train and manage separate models for each object category. We evaluated our proposed method on the Real-IAD dataset and it delivers results competitive with state-of-the-art complex models at significantly reduced parametric load and inference times. This makes it a highly practical and viable solution for real-world applications where computational resources are tightly constrained.
>
---
#### [new 015] Expert Consensus-based Video-Based Assessment Tool for Workflow Analysis in Minimally Invasive Colorectal Surgery: Development and Validation of ColoWorkflow
- **分类: cs.CV**

- **简介: 论文开发了ColoWorkflow工具，首个基于共识的视频评估工具用于微创结直肠手术工作流程分析。通过德尔菲过程确立10个通用阶段和34个步骤，应用于54个手术视频，实现良好适用性（Cohen's K=0.71）和可靠性。解决手术标准化与质量改进问题。**

- **链接: [https://arxiv.org/pdf/2511.10766v1](https://arxiv.org/pdf/2511.10766v1)**

> **作者:** Pooja P Jain; Pietro Mascagni; Giuseppe Massimiani; Nabani Banik; Marta Goglia; Lorenzo Arboit; Britty Baby; Andrea Balla; Ludovica Baldari; Gianfranco Silecchia; Claudio Fiorillo; CompSurg Colorectal Experts Group; Sergio Alfieri; Salvador Morales-Conde; Deborah S Keller; Luigi Boni; Nicolas Padoy
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Minimally invasive colorectal surgery is characterized by procedural variability, a difficult learning curve, and complications that impact quality and outcomes. Video-based assessment (VBA) offers an opportunity to generate data-driven insights to reduce variability, optimize training, and improve surgical performance. However, existing tools for workflow analysis remain difficult to standardize and implement. This study aims to develop and validate a VBA tool for workflow analysis across minimally invasive colorectal procedures. A Delphi process was conducted to achieve consensus on generalizable workflow descriptors. The resulting framework informed the development of a new VBA tool, ColoWorkflow. Independent raters then applied ColoWorkflow to a multicentre video dataset of laparoscopic and robotic colorectal surgery (CRS). Applicability and inter-rater reliability were evaluated. Consensus was achieved for 10 procedure-agnostic phases and 34 procedure-specific steps describing CRS workflows. ColoWorkflow was developed and applied to 54 colorectal operative videos (left and right hemicolectomies, sigmoid and rectosigmoid resections, and total proctocolectomies) from five centres. The tool demonstrated broad applicability, with all but one label utilized. Inter-rater reliability was moderate, with mean Cohen's K of 0.71 for phases and 0.66 for steps. Most discrepancies arose at phase transitions and step boundary definitions. ColoWorkflow is the first consensus-based, validated VBA tool for comprehensive workflow analysis in minimally invasive CRS. It establishes a reproducible framework for video-based performance assessment, enabling benchmarking across institutions and supporting the development of artificial intelligence-driven workflow recognition. Its adoption may standardize training, accelerate competency acquisition, and advance data-informed surgical quality improvement.
>
---
#### [new 016] CountSteer: Steering Attention for Object Counting in Diffusion Models
- **分类: cs.CV**

- **简介: 论文针对文本到图像扩散模型在对象计数任务中无法准确遵循数字指令的问题，提出CountSteer方法。该方法在推理时引导交叉注意力隐藏状态，无需训练即可提升计数准确性约4%，同时保持视觉质量。**

- **链接: [https://arxiv.org/pdf/2511.11253v1](https://arxiv.org/pdf/2511.11253v1)**

> **作者:** Hyemin Boo; Hyoryung Kim; Myungjin Lee; Seunghyeon Lee; Jiyoung Lee; Jang-Hwan Choi; Hyunsoo Cho
>
> **备注:** Accepted to AAAI 2026 Workshop on Shaping Responsible Synthetic Data in the Era of Foundation Models (RSD)
>
> **摘要:** Text-to-image diffusion models generate realistic and coherent images but often fail to follow numerical instructions in text, revealing a gap between language and visual representation. Interestingly, we found that these models are not entirely blind to numbers-they are implicitly aware of their own counting accuracy, as their internal signals shift in consistent ways depending on whether the output meets the specified count. This observation suggests that the model already encodes a latent notion of numerical correctness, which can be harnessed to guide generation more precisely. Building on this intuition, we introduce CountSteer, a training-free method that improves generation of specified object counts by steering the model's cross-attention hidden states during inference. In our experiments, CountSteer improved object-count accuracy by about 4% without compromising visual quality, demonstrating a simple yet effective step toward more controllable and semantically reliable text-to-image generation.
>
---
#### [new 017] Beyond Flatlands: Unlocking Spatial Intelligence by Decoupling 3D Reasoning from Numerical Regression
- **分类: cs.CV**

- **简介: 该论文解决视觉语言模型在3D空间推理中的双重瓶颈问题，提出GEODE架构解耦3D推理与数值生成，通过DRM和DRH模块实现高效空间推理。**

- **链接: [https://arxiv.org/pdf/2511.11239v1](https://arxiv.org/pdf/2511.11239v1)**

> **作者:** Zhongbin Guo; Jiahe Liu; Yushan Li; Wenyu Gao; Zhen Yang; Chenzhi Li; Xinyue Zhang; Ping Jian
>
> **摘要:** Existing Vision Language Models (VLMs) architecturally rooted in "flatland" perception, fundamentally struggle to comprehend real-world 3D spatial intelligence. This failure stems from a dual-bottleneck: input-stage conflict between computationally exorbitant geometric-aware encoders and superficial 2D-only features, and output-stage misalignment where discrete tokenizers are structurally incapable of producing precise, continuous numerical values. To break this impasse, we introduce GEODE (Geometric-Output and Decoupled-Input Engine), a novel architecture that resolves this dual-bottleneck by decoupling 3D reasoning from numerical generation. GEODE augments main VLM with two specialized, plug-and-play modules: Decoupled Rationale Module (DRM) that acts as spatial co-processor, aligning explicit 3D data with 2D visual features via cross-attention and distilling spatial Chain-of-Thought (CoT) logic into injectable Rationale Tokens; and Direct Regression Head (DRH), an "Embedding-as-Value" paradigm which routes specialized control tokens to a lightweight MLP for precise, continuous regression of scalars and 3D bounding boxes. The synergy of these modules allows our 1.5B parameter model to function as a high-level semantic dispatcher, achieving state-of-the-art spatial reasoning performance that rivals 7B+ models.
>
---
#### [new 018] VIDEOP2R: Video Understanding from Perception to Reasoning
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出VideoP2R框架，解决视频理解中RFT扩展到LVLMs的挑战。通过将感知与推理建模为独立过程：开发process-aware CoT数据集并设计PA-GRPO算法。在6/7视频基准测试中达SotA性能。**

- **链接: [https://arxiv.org/pdf/2511.11113v1](https://arxiv.org/pdf/2511.11113v1)**

> **作者:** Yifan Jiang; Yueying Wang; Rui Zhao; Toufiq Parag; Zhimin Chen; Zhenyu Liao; Jayakrishnan Unnikrishnan
>
> **摘要:** Reinforcement fine-tuning (RFT), a two-stage framework consisting of supervised fine-tuning (SFT) and reinforcement learning (RL) has shown promising results on improving reasoning ability of large language models (LLMs). Yet extending RFT to large video language models (LVLMs) remains challenging. We propose VideoP2R, a novel process-aware video RFT framework that enhances video reasoning by modeling perception and reasoning as distinct processes. In the SFT stage, we develop a three-step pipeline to generate VideoP2R-CoT-162K, a high-quality, process-aware chain-of-thought (CoT) dataset for perception and reasoning. In the RL stage, we introduce a novel process-aware group relative policy optimization (PA-GRPO) algorithm that supplies separate rewards for perception and reasoning. Extensive experiments show that VideoP2R achieves state-of-the-art (SotA) performance on six out of seven video reasoning and understanding benchmarks. Ablation studies further confirm the effectiveness of our process-aware modeling and PA-GRPO and demonstrate that model's perception output is information-sufficient for downstream reasoning.
>
---
#### [new 019] PROMISE: Prompt-Attentive Hierarchical Contrastive Learning for Robust Cross-Modal Representation with Missing Modalities
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多模态表示学习中模态缺失问题，提出PROMISE框架，通过提示-注意力分层对比学习动态生成稳健表示，有效弥合完整与缺失数据的表示差距。**

- **链接: [https://arxiv.org/pdf/2511.10997v1](https://arxiv.org/pdf/2511.10997v1)**

> **作者:** Jiajun Chen; Sai Cheng; Yutao Yuan; Yirui Zhang; Haitao Yuan; Peng Peng; Yi Zhong
>
> **备注:** Accepted by AAAI'2026 Main Conference
>
> **摘要:** Multimodal models integrating natural language and visual information have substantially improved generalization of representation models. However, their effectiveness significantly declines in real-world situations where certain modalities are missing or unavailable. This degradation primarily stems from inconsistent representation learning between complete multimodal data and incomplete modality scenarios. Existing approaches typically address missing modalities through relatively simplistic generation methods, yet these approaches fail to adequately preserve cross-modal consistency, leading to suboptimal performance. To overcome this limitation, we propose a novel multimodal framework named PROMISE, a PROMpting-Attentive HIerarchical ContraStive LEarning approach designed explicitly for robust cross-modal representation under conditions of missing modalities. Specifically, PROMISE innovatively incorporates multimodal prompt learning into a hierarchical contrastive learning framework, equipped with a specially designed prompt-attention mechanism. This mechanism dynamically generates robust and consistent representations for scenarios where particular modalities are absent, thereby effectively bridging the representational gap between complete and incomplete data. Extensive experiments conducted on benchmark datasets, along with comprehensive ablation studies, clearly demonstrate the superior performance of PROMISE compared to current state-of-the-art multimodal methods.
>
---
#### [new 020] RealisticDreamer: Guidance Score Distillation for Few-shot Gaussian Splatting
- **分类: cs.CV**

- **简介: 论文针对稀疏视图下3D高斯泼溅的过拟合问题，提出Guidance Score Distillation (GSD)框架。利用预训练视频扩散模型提取多视图一致性先验，通过深度扭曲和语义特征引导确保几何与相机姿态一致，实验显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.11213v1](https://arxiv.org/pdf/2511.11213v1)**

> **作者:** Ruocheng Wu; Haolan He; Yufei Wang; Zhihao Li; Bihan Wen
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently gained great attention in the 3D scene representation for its high-quality real-time rendering capabilities. However, when the input comprises sparse training views, 3DGS is prone to overfitting, primarily due to the lack of intermediate-view supervision. Inspired by the recent success of Video Diffusion Models (VDM), we propose a framework called Guidance Score Distillation (GSD) to extract the rich multi-view consistency priors from pretrained VDMs. Building on the insights from Score Distillation Sampling (SDS), GSD supervises rendered images from multiple neighboring views, guiding the Gaussian splatting representation towards the generative direction of VDM. However, the generative direction often involves object motion and random camera trajectories, making it challenging for direct supervision in the optimization process. To address this problem, we introduce an unified guidance form to correct the noise prediction result of VDM. Specifically, we incorporate both a depth warp guidance based on real depth maps and a guidance based on semantic image features, ensuring that the score update direction from VDM aligns with the correct camera pose and accurate geometry. Experimental results show that our method outperforms existing approaches across multiple datasets.
>
---
#### [new 021] Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval
- **分类: cs.CV**

- **简介: 论文提出H²ARN解决文本-3D检索中的层次表示崩溃和冗余导致的显著性稀释问题。通过双曲空间嵌入保留层次结构，设计层次排序损失和贡献感知聚合模块，增强判别能力。**

- **链接: [https://arxiv.org/pdf/2511.11045v1](https://arxiv.org/pdf/2511.11045v1)**

> **作者:** Wenrui Li; Yidan Lu; Yeyu Chai; Rui Zhao; Hengyu Man; Xiaopeng Fan
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** With the daily influx of 3D data on the internet, text-3D retrieval has gained increasing attention. However, current methods face two major challenges: Hierarchy Representation Collapse (HRC) and Redundancy-Induced Saliency Dilution (RISD). HRC compresses abstract-to-specific and whole-to-part hierarchies in Euclidean embeddings, while RISD averages noisy fragments, obscuring critical semantic cues and diminishing the model's ability to distinguish hard negatives. To address these challenges, we introduce the Hyperbolic Hierarchical Alignment Reasoning Network (H$^{2}$ARN) for text-3D retrieval. H$^{2}$ARN embeds both text and 3D data in a Lorentz-model hyperbolic space, where exponential volume growth inherently preserves hierarchical distances. A hierarchical ordering loss constructs a shrinking entailment cone around each text vector, ensuring that the matched 3D instance falls within the cone, while an instance-level contrastive loss jointly enforces separation from non-matching samples. To tackle RISD, we propose a contribution-aware hyperbolic aggregation module that leverages Lorentzian distance to assess the relevance of each local feature and applies contribution-weighted aggregation guided by hyperbolic geometry, enhancing discriminative regions while suppressing redundancy without additional supervision. We also release the expanded T3DR-HIT v2 benchmark, which contains 8,935 text-to-3D pairs, 2.6 times the original size, covering both fine-grained cultural artefacts and complex indoor scenes. Our codes are available at https://github.com/liwrui/H2ARN.
>
---
#### [new 022] D-GAP: Improving Out-of-Domain Robustness via Dataset-Agnostic and Gradient-Guided Augmentation in Amplitude and Pixel Spaces
- **分类: cs.CV; cs.AI**

- **简介: 论文解决计算机视觉域外鲁棒性问题。针对域变化导致性能下降，提出D-GAP：基于梯度引导在幅度和像素空间自适应增强，减少频率学习偏差并恢复细节，显著提升OOD性能。**

- **链接: [https://arxiv.org/pdf/2511.11286v1](https://arxiv.org/pdf/2511.11286v1)**

> **作者:** Ruoqi Wang; Haitao Wang; Shaojie Guo; Qiong Luo
>
> **摘要:** Out-of-domain (OOD) robustness is challenging to achieve in real-world computer vision applications, where shifts in image background, style, and acquisition instruments always degrade model performance. Generic augmentations show inconsistent gains under such shifts, whereas dataset-specific augmentations require expert knowledge and prior analysis. Moreover, prior studies show that neural networks adapt poorly to domain shifts because they exhibit a learning bias to domain-specific frequency components. Perturbing frequency values can mitigate such bias but overlooks pixel-level details, leading to suboptimal performance. To address these problems, we propose D-GAP (Dataset-agnostic and Gradient-guided augmentation in Amplitude and Pixel spaces), improving OOD robustness by introducing targeted augmentation in both the amplitude space (frequency space) and pixel space. Unlike conventional handcrafted augmentations, D-GAP computes sensitivity maps in the frequency space from task gradients, which reflect how strongly the model responds to different frequency components, and uses the maps to adaptively interpolate amplitudes between source and target samples. This way, D-GAP reduces the learning bias in frequency space, while a complementary pixel-space blending procedure restores fine spatial details. Extensive experiments on four real-world datasets and three domain-adaptation benchmarks show that D-GAP consistently outperforms both generic and dataset-specific augmentations, improving average OOD performance by +5.3% on real-world datasets and +1.8% on benchmark datasets.
>
---
#### [new 023] EmbryoDiff: A Conditional Diffusion Framework with Multi-Focal Feature Fusion for Fine-Grained Embryo Developmental Stage Recognition
- **分类: cs.CV**

- **简介: 论文提出EmbryoDiff，用于细粒度胚胎发育阶段识别任务。解决现有方法忽略发育分布先验及单焦点信息导致的遮挡问题，通过扩散框架融合多焦点特征并设计混合语义-边界条件块，实现82.8%准确率。**

- **链接: [https://arxiv.org/pdf/2511.11027v1](https://arxiv.org/pdf/2511.11027v1)**

> **作者:** Yong Sun; Zhengjie Zhang; Junyu Shi; Zhiyuan Zhang; Lijiang Liu; Qiang Nie
>
> **摘要:** Identification of fine-grained embryo developmental stages during In Vitro Fertilization (IVF) is crucial for assessing embryo viability. Although recent deep learning methods have achieved promising accuracy, existing discriminative models fail to utilize the distributional prior of embryonic development to improve accuracy. Moreover, their reliance on single-focal information leads to incomplete embryonic representations, making them susceptible to feature ambiguity under cell occlusions. To address these limitations, we propose EmbryoDiff, a two-stage diffusion-based framework that formulates the task as a conditional sequence denoising process. Specifically, we first train and freeze a frame-level encoder to extract robust multi-focal features. In the second stage, we introduce a Multi-Focal Feature Fusion Strategy that aggregates information across focal planes to construct a 3D-aware morphological representation, effectively alleviating ambiguities arising from cell occlusions. Building on this fused representation, we derive complementary semantic and boundary cues and design a Hybrid Semantic-Boundary Condition Block to inject them into the diffusion-based denoising process, enabling accurate embryonic stage classification. Extensive experiments on two benchmark datasets show that our method achieves state-of-the-art results. Notably, with only a single denoising step, our model obtains the best average test performance, reaching 82.8% and 81.3% accuracy on the two datasets, respectively.
>
---
#### [new 024] YOLO-Drone: An Efficient Object Detection Approach Using the GhostHead Network for Drone Images
- **分类: cs.CV**

- **简介: 论文针对无人机高空图像物体识别困难问题，提出YOLO-Drone模型。基于YOLOv11n，引入GhostHead网络改进Head结构，显著提升Precision、Recall、F1和mAP（各增0.4-0.6%），推理速度更快，优于YOLOv8/9/10。**

- **链接: [https://arxiv.org/pdf/2511.10905v1](https://arxiv.org/pdf/2511.10905v1)**

> **作者:** Hyun-Ki Jung
>
> **备注:** Preprint version. Accepted for publication in the Journal of Information Systems Engineering and Management
>
> **摘要:** Object detection using images or videos captured by drones is a promising technology with significant potential across various industries. However, a major challenge is that drone images are typically taken from high altitudes, making object identification difficult. This paper proposes an effective solution to address this issue. The base model used in the experiments is YOLOv11, the latest object detection model, with a specific implementation based on YOLOv11n. The experimental data were sourced from the widely used and reliable VisDrone dataset, a standard benchmark in drone-based object detection. This paper introduces an enhancement to the Head network of the YOLOv11 algorithm, called the GhostHead Network. The model incorporating this improvement is named YOLO-Drone. Experimental results demonstrate that YOLO-Drone achieves significant improvements in key detection accuracy metrics, including Precision, Recall, F1-Score, and mAP (0.5), compared to the original YOLOv11. Specifically, the proposed model recorded a 0.4% increase in Precision, a 0.6% increase in Recall, a 0.5% increase in F1-Score, and a 0.5% increase in mAP (0.5). Additionally, the Inference Speed metric, which measures image processing speed, also showed a notable improvement. These results indicate that YOLO-Drone is a high-performance model with enhanced accuracy and speed compared to YOLOv11. To further validate its reliability, comparative experiments were conducted against other high-performance object detection models, including YOLOv8, YOLOv9, and YOLOv10. The results confirmed that the proposed model outperformed YOLOv8 by 0.1% in mAP (0.5) and surpassed YOLOv9 and YOLOv10 by 0.3% and 0.6%, respectively.
>
---
#### [new 025] NP-LoRA: Null Space Projection Unifies Subject and Style in LoRA Fusion
- **分类: cs.CV**

- **简介: 该论文解决LoRA融合中的结构性干扰问题，提出NP-LoRA：通过零空间投影强制子空间分离，防止主方向重叠；引入软投影机制平滑控制主体保真度与风格一致性，显著提升融合质量。**

- **链接: [https://arxiv.org/pdf/2511.11051v1](https://arxiv.org/pdf/2511.11051v1)**

> **作者:** Chuheng Chen; Xiaofei Zhou; Geyuan Zhang; Yong Huang
>
> **摘要:** Low-Rank Adaptation (LoRA) fusion has emerged as a key technique for reusing and composing learned subject and style representations for controllable generation without costly retraining. However, existing methods rely on weight-based merging, where one LoRA often dominates the other, leading to interference and degraded fidelity. This interference is structural: separately trained LoRAs occupy low-rank high-dimensional subspaces, leading to non-orthogonal and overlapping representations. In this work, we analyze the internal structure of LoRAs and find their generative behavior is dominated by a few principal directions in the low-rank subspace, which should remain free from interference during fusion. To achieve this, we propose Null Space Projection LoRA (NP-LoRA), a projection-based framework for LoRA fusion that enforces subspace separation to prevent structural interference among principal directions. Specifically, we first extract principal style directions via singular value decomposition (SVD) and then project the subject LoRA into its orthogonal null space. Furthermore, we introduce a soft projection mechanism that enables smooth control over the trade-off between subject fidelity and style consistency. Experiments show NP-LoRA consistently improves fusion quality over strong baselines (e.g., DINO and CLIP-based metrics, with human and LLM preference scores), and applies broadly across backbones and LoRA pairs without retraining.
>
---
#### [new 026] Text-guided Weakly Supervised Framework for Dynamic Facial Expression Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对动态面部表情识别（DFER）任务，解决视频单标签标注问题。提出TG-DFER框架，采用文本引导弱监督，结合视觉语言模型提供语义引导，引入视觉提示对齐文本与视觉特征，设计多粒度时间网络捕捉时间动态。**

- **链接: [https://arxiv.org/pdf/2511.10958v1](https://arxiv.org/pdf/2511.10958v1)**

> **作者:** Gunho Jung; Heejo Kong; Seong-Whan Lee
>
> **摘要:** Dynamic facial expression recognition (DFER) aims to identify emotional states by modeling the temporal changes in facial movements across video sequences. A key challenge in DFER is the many-to-one labeling problem, where a video composed of numerous frames is assigned a single emotion label. A common strategy to mitigate this issue is to formulate DFER as a Multiple Instance Learning (MIL) problem. However, MIL-based approaches inherently suffer from the visual diversity of emotional expressions and the complexity of temporal dynamics. To address this challenge, we propose TG-DFER, a text-guided weakly supervised framework that enhances MIL-based DFER by incorporating semantic guidance and coherent temporal modeling. We incorporate a vision-language pre-trained (VLP) model is integrated to provide semantic guidance through fine-grained textual descriptions of emotional context. Furthermore, we introduce visual prompts, which align enriched textual emotion labels with visual instance features, enabling fine-grained reasoning and frame-level relevance estimation. In addition, a multi-grained temporal network is designed to jointly capture short-term facial dynamics and long-range emotional flow, ensuring coherent affective understanding across time. Extensive results demonstrate that TG-DFER achieves improved generalization, interpretability, and temporal sensitivity under weak supervision.
>
---
#### [new 027] SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices
- **分类: cs.CV; cs.AI; cs.DC**

- **简介: 论文针对物联网极弱设备AI推理卸载任务，解决资源受限和网络不可靠导致的传输效率低问题。提出SemanticNN，通过语义级错误容忍和压缩编码，使特征传输量减少56.82-344.83倍，同时保持高精度。**

- **链接: [https://arxiv.org/pdf/2511.11038v1](https://arxiv.org/pdf/2511.11038v1)**

> **作者:** Jiaming Huang; Yi Gao; Fuchang Pan; Renjie Li; Wei Dong
>
> **摘要:** With the rapid growth of the Internet of Things (IoT), integrating artificial intelligence (AI) on extremely weak embedded devices has garnered significant attention, enabling improved real-time performance and enhanced data privacy. However, the resource limitations of such devices and unreliable network conditions necessitate error-resilient device-edge collaboration systems. Traditional approaches focus on bit-level transmission correctness, which can be inefficient under dynamic channel conditions. In contrast, we propose SemanticNN, a semantic codec that tolerates bit-level errors in pursuit of semantic-level correctness, enabling compressive and resilient collaborative inference offloading under strict computational and communication constraints. It incorporates a Bit Error Rate (BER)-aware decoder that adapts to dynamic channel conditions and a Soft Quantization (SQ)-based encoder to learn compact representations. Building on this architecture, we introduce Feature-augmentation Learning, a novel training strategy that enhances offloading efficiency. To address encoder-decoder capability mismatches from asymmetric resources, we propose XAI-based Asymmetry Compensation to enhance decoding semantic fidelity. We conduct extensive experiments on STM32 using three models and six datasets across image classification and object detection tasks. Experimental results demonstrate that, under varying transmission error rates, SemanticNN significantly reduces feature transmission volume by 56.82-344.83x while maintaining superior inference accuracy.
>
---
#### [new 028] DINOv3 as a Frozen Encoder for CRPS-Oriented Probabilistic Rainfall Nowcasting
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于DINOv3冻结编码器的概率性降雨临近预报方法，通过视频投影器映射至离散经验CDF并端到端优化CRPS。在Weather4Cast 2025上CRPS达3.5102，较最佳3D-UNET提升26%。**

- **链接: [https://arxiv.org/pdf/2511.10894v1](https://arxiv.org/pdf/2511.10894v1)**

> **作者:** Luciano Araujo Dourado Filho; Almir Moreira da Silva Neto; Anthony Miyaguchi; Rodrigo Pereira David; Rodrigo Tripodi Calumby; Lukáš Picek
>
> **摘要:** This paper proposes a competitive and computationally efficient approach to probabilistic rainfall nowcasting. A video projector (V-JEPA Vision Transformer) associated to a lightweight probabilistic head is attached to a pre-trained satellite vision encoder (DINOv3\text{-}SAT493M) to map encoder tokens into a discrete empirical CDF (eCDF) over 4-hour accumulated rainfall. The projector-head is optimized end-to-end over the Continuous Ranked Probability Score (CRPS). As an alternative, 3D-UNET baselines trained with an aggregate Rank Probability Score and a per-pixel Gamma-Hurdle objective are used. On the Weather4Cast 2025 benchmark, the proposed method achieved a promising performance, with a CRPS of 3.5102 (CRPS), which represents $\approx$26\% in effectiveness gain against the best 3D-UNET.
>
---
#### [new 029] Algorithms Trained on Normal Chest X-rays Can Predict Health Insurance Types
- **分类: cs.CV; cs.AI**

- **简介: 论文研究医疗AI公平性任务，解决模型能从正常胸部X光预测健康保险类型（社会经济代理）的问题。工作包括训练DenseNet121等模型在MIMIC-CXR-JPG和CheXpert数据集上，获AUC 0.67-0.68；分析信号弥散于上胸部，证明模型内化社会不平等痕迹，挑战医学图像中性假设，强调公平AI需探究数据社会指纹。**

- **链接: [https://arxiv.org/pdf/2511.11030v1](https://arxiv.org/pdf/2511.11030v1)**

> **作者:** Chi-Yu Chen; Rawan Abulibdeh; Arash Asgari; Leo Anthony Celi; Deirdre Goode; Hassan Hamidi; Laleh Seyyed-Kalantari; Po-Chih Kuo; Ned McCague; Thomas Sounack
>
> **备注:** Submitting to MIDL 2026
>
> **摘要:** Artificial intelligence is revealing what medicine never intended to encode. Deep vision models, trained on chest X-rays, can now detect not only disease but also invisible traces of social inequality. In this study, we show that state-of-the-art architectures (DenseNet121, SwinV2-B, MedMamba) can predict a patient's health insurance type, a strong proxy for socioeconomic status, from normal chest X-rays with significant accuracy (AUC around 0.67 on MIMIC-CXR-JPG, 0.68 on CheXpert). The signal persists even when age, race, and sex are controlled for, and remains detectable when the model is trained exclusively on a single racial group. Patch-based occlusion reveals that the signal is diffuse rather than localized, embedded in the upper and mid-thoracic regions. This suggests that deep networks may be internalizing subtle traces of clinical environments, equipment differences, or care pathways; learning socioeconomic segregation itself. These findings challenge the assumption that medical images are neutral biological data. By uncovering how models perceive and exploit these hidden social signatures, this work reframes fairness in medical AI: the goal is no longer only to balance datasets or adjust thresholds, but to interrogate and disentangle the social fingerprints embedded in clinical data itself.
>
---
#### [new 030] Geospatial Chain of Thought Reasoning for Enhanced Visual Question Answering on Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文针对卫星图像视觉问答（VQA）任务，解决现有模型缺乏结构化地理空间推理的问题。提出整合链式思维（CoT）与直接偏好优化（DPO）的框架，实验显示准确率提升34.9%，增强气候应用决策支持。**

- **链接: [https://arxiv.org/pdf/2511.11198v1](https://arxiv.org/pdf/2511.11198v1)**

> **作者:** Shambhavi Shanker; Manikandan Padmanaban; Jagabondhu Hazra
>
> **摘要:** Geospatial chain of thought (CoT) reasoning is essential for advancing Visual Question Answering (VQA) on satellite imagery, particularly in climate related applications such as disaster monitoring, infrastructure risk assessment, urban resilience planning, and policy support. Existing VQA models enable scalable interpretation of remote sensing data but often lack the structured reasoning required for complex geospatial queries. We propose a VQA framework that integrates CoT reasoning with Direct Preference Optimization (DPO) to improve interpretability, robustness, and accuracy. By generating intermediate rationales, the model better handles tasks involving detection, classification, spatial relations, and comparative analysis, which are critical for reliable decision support in high stakes climate domains. Experiments show that CoT supervision improves accuracy by 34.9\% over direct baselines, while DPO yields additional gains in accuracy and reasoning quality. The resulting system advances VQA for multispectral Earth observation by enabling richer geospatial reasoning and more effective climate use cases.
>
---
#### [new 031] Fast Data Attribution for Text-to-Image Models
- **分类: cs.CV; cs.LG**

- **简介: 论文提出快速数据归属方法，用于文本到图像模型，解决现有方法计算昂贵问题。通过知识蒸馏至特征嵌入空间，结合高效检索，实现秒级响应，速度提升2500-400000倍。**

- **链接: [https://arxiv.org/pdf/2511.10721v1](https://arxiv.org/pdf/2511.10721v1)**

> **作者:** Sheng-Yu Wang; Aaron Hertzmann; Alexei A Efros; Richard Zhang; Jun-Yan Zhu
>
> **备注:** NeurIPS 2025 camera ready. Project page: https://peterwang512.github.io/FastGDA
>
> **摘要:** Data attribution for text-to-image models aims to identify the training images that most significantly influenced a generated output. Existing attribution methods involve considerable computational resources for each query, making them impractical for real-world applications. We propose a novel approach for scalable and efficient data attribution. Our key idea is to distill a slow, unlearning-based attribution method to a feature embedding space for efficient retrieval of highly influential training images. During deployment, combined with efficient indexing and search methods, our method successfully finds highly influential images without running expensive attribution algorithms. We show extensive results on both medium-scale models trained on MSCOCO and large-scale Stable Diffusion models trained on LAION, demonstrating that our method can achieve better or competitive performance in a few seconds, faster than existing methods by 2,500x - 400,000x. Our work represents a meaningful step towards the large-scale application of data attribution methods on real-world models such as Stable Diffusion.
>
---
#### [new 032] One-to-N Backdoor Attack in 3D Point Cloud via Spherical Trigger
- **分类: cs.CV**

- **简介: 该论文提出3D点云的one-to-N后门攻击框架，解决现有攻击仅支持一对一目标的问题。利用球形触发器作为参数空间，实现单触发器编码多目标类别，实验验证攻击成功率100%且保持干净数据准确率。**

- **链接: [https://arxiv.org/pdf/2511.11210v1](https://arxiv.org/pdf/2511.11210v1)**

> **作者:** Dongmei Shan; Wei Lian; Chongxia Wang
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Backdoor attacks represent a critical threat to deep learning systems, particularly in safety-sensitive 3D domains such as autonomous driving and robotics. However, existing backdoor attacks for 3D point clouds have been limited to a rigid one-to-one paradigm. To address this, we present the first one-to-N backdoor framework for 3D vision, based on a novel, configurable spherical trigger. Our key insight is to leverage the spatial properties of spheres as a parameter space, allowing a single trigger design to encode multiple target classes. We establish a theoretical foundation for one-to-N backdoor attacks in 3D, demonstrating that poisoned models can map distinct trigger configurations to different target labels. Experimental results systematically validate this conclusion across multiple datasets and model architectures, achieving high attack success rates (up to 100\%) while maintaining accuracy on clean data. This work establishes a crucial benchmark for multi-target threats in 3D vision and provides the foundational understanding needed to secure future 3D-driven intelligent systems.
>
---
#### [new 033] The Persistence of Cultural Memory: Investigating Multimodal Iconicity in Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究扩散模型中的文化记忆问题，聚焦多模态标志性。解决泛化与记忆的模糊性，提出识别与实现的评估框架。评估5模型在767个文化参考上，发现模型常复制视觉结构，文化对齐受文本独特性等因素影响。**

- **链接: [https://arxiv.org/pdf/2511.11435v1](https://arxiv.org/pdf/2511.11435v1)**

> **作者:** Maria-Teresa De Rosa Palmini; Eva Cetinic
>
> **摘要:** Our work addresses the ambiguity between generalization and memorization in text-to-image diffusion models, focusing on a specific case we term multimodal iconicity. This refers to instances where images and texts evoke culturally shared associations, such as when a title recalls a familiar artwork or film scene. While prior research on memorization and unlearning emphasizes forgetting, we examine what is remembered and how, focusing on the balance between recognizing cultural references and reproducing them. We introduce an evaluation framework that separates recognition, whether a model identifies a reference, from realization, how it depicts it through replication or reinterpretation, quantified through measures capturing both dimensions. By evaluating five diffusion models across 767 Wikidata-derived cultural references spanning static and dynamic imagery, we show that our framework distinguishes replication from transformation more effectively than existing similarity-based methods. To assess linguistic sensitivity, we conduct prompt perturbation experiments using synonym substitutions and literal image descriptions, finding that models often reproduce iconic visual structures even when textual cues are altered. Finally, our analysis shows that cultural alignment correlates not only with training data frequency, but also textual uniqueness, reference popularity, and creation date. Our work reveals that the value of diffusion models lies not only in what they reproduce but in how they transform and recontextualize cultural knowledge, advancing evaluation beyond simple text-image matching toward richer contextual understanding.
>
---
#### [new 034] Comprehension of Multilingual Expressions Referring to Target Objects in Visual Inputs
- **分类: cs.CV**

- **简介: 该论文解决多语言Referring Expression Comprehension (REC)问题，构建10语言统一数据集（8M表达，177k图像），提出注意力锚定神经架构，多语言评估达86.9% IoU@50，实现跨语言一致性能。**

- **链接: [https://arxiv.org/pdf/2511.11427v1](https://arxiv.org/pdf/2511.11427v1)**

> **作者:** Francisco Nogueira; Alexandre Bernardino; Bruno Martins
>
> **摘要:** Referring Expression Comprehension (REC) requires models to localize objects in images based on natural language descriptions. Research on the area remains predominantly English-centric, despite increasing global deployment demands. This work addresses multilingual REC through two main contributions. First, we construct a unified multilingual dataset spanning 10 languages, by systematically expanding 12 existing English REC benchmarks through machine translation and context-based translation enhancement. The resulting dataset comprises approximately 8 million multilingual referring expressions across 177,620 images, with 336,882 annotated objects. Second, we introduce an attention-anchored neural architecture that uses multilingual SigLIP2 encoders. Our attention-based approach generates coarse spatial anchors from attention distributions, which are subsequently refined through learned residuals. Experimental evaluation demonstrates competitive performance on standard benchmarks, e.g. achieving 86.9% accuracy at IoU@50 on RefCOCO aggregate multilingual evaluation, compared to an English-only result of 91.3%. Multilingual evaluation shows consistent capabilities across languages, establishing the practical feasibility of multilingual visual grounding systems. The dataset and model are available at $\href{https://multilingual.franreno.com}{multilingual.franreno.com}$.
>
---
#### [new 035] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation
- **分类: cs.CV**

- **简介: 论文提出EmoVid数据集，解决视频生成中情感维度缺失问题。数据集包含动画、电影等情感标注视频，分析视觉特征与情感关联。基于此，开发情感条件视频生成技术，微调Wan2.1模型，显著提升文本/图像到视频生成质量。**

- **链接: [https://arxiv.org/pdf/2511.11002v1](https://arxiv.org/pdf/2511.11002v1)**

> **作者:** Zongyang Qiu; Bingyuan Wang; Xingbei Chen; Yingqing He; Zeyu Wang
>
> **备注:** 15 pages, 12 figures. Accepted as an Oral presentation at AAAI 2026. For code and dataset, see https://zane-zyqiu.github.io/EmoVid
>
> **摘要:** Emotion plays a pivotal role in video-based expression, but existing video generation systems predominantly focus on low-level visual metrics while neglecting affective dimensions. Although emotion analysis has made progress in the visual domain, the video community lacks dedicated resources to bridge emotion understanding with generative tasks, particularly for stylized and non-realistic contexts. To address this gap, we introduce EmoVid, the first multimodal, emotion-annotated video dataset specifically designed for creative media, which includes cartoon animations, movie clips, and animated stickers. Each video is annotated with emotion labels, visual attributes (brightness, colorfulness, hue), and text captions. Through systematic analysis, we uncover spatial and temporal patterns linking visual features to emotional perceptions across diverse video forms. Building on these insights, we develop an emotion-conditioned video generation technique by fine-tuning the Wan2.1 model. The results show a significant improvement in both quantitative metrics and the visual quality of generated videos for text-to-video and image-to-video tasks. EmoVid establishes a new benchmark for affective video computing. Our work not only offers valuable insights into visual emotion analysis in artistically styled videos, but also provides practical methods for enhancing emotional expression in video generation.
>
---
#### [new 036] Binary Verification for Zero-Shot Vision
- **分类: cs.CV; cs.AI**

- **简介: 论文提出训练-free二元验证工作流程用于零样本视觉，通过量化将开放查询转为多选题，再二值化为真假问题确定答案。在引用表达定位、空间推理等任务上显著提升准确率，避免任务特定训练。**

- **链接: [https://arxiv.org/pdf/2511.10983v1](https://arxiv.org/pdf/2511.10983v1)**

> **作者:** Jeffrey Liu; Rongbin Hu
>
> **摘要:** We propose a training-free, binary verification workflow for zero-shot vision with off-the-shelf VLMs. It comprises two steps: (i) quantization, which turns the open-ended query into a multiple-choice question (MCQ) with a small, explicit list of unambiguous candidates; and (ii) binarization, which asks one True/False question per candidate and resolves deterministically: if exactly one is True, select it; otherwise, revert to an MCQ over the remaining plausible candidates. We evaluate the workflow on referring expression grounding (REC), spatial reasoning (Spatial-Map, Spatial-Grid, Spatial-Maze), and BLINK-Jigsaw. Relative to answering open-ended queries directly, quantization to MCQ yields large gains, and True/False binarization provides a consistent additional boost. Across all tasks, the same workflow produces significant improvements, indicating generality. Our theory formalizes how open-ended vision queries can be quantized to MCQs and further binarized into True/False verifications, establishing a hardness ladder. A simple analysis explains why Boolean resolution boosts accuracy. Together, these components yield a simple and unified workflow that emphasizes inference-time design over task-specific training. It offers a practical, drop-in path to stronger zero-shot vision with today's VLMs.
>
---
#### [new 037] PhaseWin Search Framework Enable Efficient Object-Level Interpretation
- **分类: cs.CV**

- **简介: 论文提出PhaseWin框架解决对象级解释效率问题。通过分阶段粗到精搜索，将计算复杂度降至近线性，仅用20%计算量实现95%忠实度，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10914v1](https://arxiv.org/pdf/2511.10914v1)**

> **作者:** Zihan Gu; Ruoyu Chen; Junchi Zhang; Yue Hu; Hua Zhang; Xiaochun Cao
>
> **摘要:** Attribution is essential for interpreting object-level foundation models. Recent methods based on submodular subset selection have achieved high faithfulness, but their efficiency limitations hinder practical deployment in real-world scenarios. To address this, we propose PhaseWin, a novel phase-window search algorithm that enables faithful region attribution with near-linear complexity. PhaseWin replaces traditional quadratic-cost greedy selection with a phased coarse-to-fine search, combining adaptive pruning, windowed fine-grained selection, and dynamic supervision mechanisms to closely approximate greedy behavior while dramatically reducing model evaluations. Theoretically, PhaseWin retains near-greedy approximation guarantees under mild monotone submodular assumptions. Empirically, PhaseWin achieves over 95% of greedy attribution faithfulness using only 20% of the computational budget, and consistently outperforms other attribution baselines across object detection and visual grounding tasks with Grounding DINO and Florence-2. PhaseWin establishes a new state of the art in scalable, high-faithfulness attribution for object-level multimodal models.
>
---
#### [new 038] ImAgent: A Unified Multimodal Agent Framework for Test-Time Scalable Image Generation
- **分类: cs.CV; cs.AI**

- **简介: ImAgent解决文本到图像生成的随机性和提示不一致问题。它提出训练-free统一多模态代理框架，整合推理、生成与自评估，通过动态交互提升图像质量，实现高效测试时扩展。**

- **链接: [https://arxiv.org/pdf/2511.11483v1](https://arxiv.org/pdf/2511.11483v1)**

> **作者:** Kaishen Wang; Ruibo Chen; Tong Zheng; Heng Huang
>
> **备注:** 12 pages, 5 tables, 6 figures
>
> **摘要:** Recent text-to-image (T2I) models have made remarkable progress in generating visually realistic and semantically coherent images. However, they still suffer from randomness and inconsistency with the given prompts, particularly when textual descriptions are vague or underspecified. Existing approaches, such as prompt rewriting, best-of-N sampling, and self-refinement, can mitigate these issues but usually require additional modules and operate independently, hindering test-time scaling efficiency and increasing computational overhead. In this paper, we introduce ImAgent, a training-free unified multimodal agent that integrates reasoning, generation, and self-evaluation within a single framework for efficient test-time scaling. Guided by a policy controller, multiple generation actions dynamically interact and self-organize to enhance image fidelity and semantic alignment without relying on external models. Extensive experiments on image generation and editing tasks demonstrate that ImAgent consistently improves over the backbone and even surpasses other strong baselines where the backbone model fails, highlighting the potential of unified multimodal agents for adaptive and efficient image generation under test-time scaling.
>
---
#### [new 039] Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Image Segmentation
- **分类: cs.CV**

- **简介: 论文针对联邦医疗图像分割中的特征异质性问题（由不同扫描仪或协议导致），提出FedBCS方法。通过频域自适应风格重校准和上下文感知双层原型对齐，桥接特征表示差距，解决多级上下文缺失和层级风格偏差累积问题。**

- **链接: [https://arxiv.org/pdf/2511.10945v1](https://arxiv.org/pdf/2511.10945v1)**

> **作者:** Xingyue Zhao; Wenke Huang; Xingguang Wang; Haoyu Zhao; Linghao Zhuang; Anwen Jiang; Guancheng Wan; Mang Ye
>
> **备注:** Accepted at AAAI-26
>
> **摘要:** Federated learning enables multiple medical institutions to train a global model without sharing data, yet feature heterogeneity from diverse scanners or protocols remains a major challenge. Many existing works attempt to address this issue by leveraging model representations (e.g., mean feature vectors) to correct local training; however, they often face two key limitations: 1) Incomplete Contextual Representation Learning: Current approaches primarily focus on final-layer features, overlooking critical multi-level cues and thus diluting essential context for accurate segmentation. 2) Layerwise Style Bias Accumulation: Although utilizing representations can partially align global features, these methods neglect domain-specific biases within intermediate layers, allowing style discrepancies to build up and reduce model robustness. To address these challenges, we propose FedBCS to bridge feature representation gaps via domain-invariant contextual prototypes alignment. Specifically, we introduce a frequency-domain adaptive style recalibration into prototype construction that not only decouples content-style representations but also learns optimal style parameters, enabling more robust domain-invariant prototypes. Furthermore, we design a context-aware dual-level prototype alignment method that extracts domain-invariant prototypes from different layers of both encoder and decoder and fuses them with contextual information for finer-grained representation alignment. Extensive experiments on two public datasets demonstrate that our method exhibits remarkable performance.
>
---
#### [new 040] Out-of-Distribution Detection with Positive and Negative Prompt Supervision Using Large Language Models
- **分类: cs.CV**

- **简介: 该论文解决OOD检测中负提示引入非ID特征导致性能下降的问题，提出正负提示监督方法：LLMs初始化优化提示，正提示聚焦类内特征，负提示突出类别边界；通过图架构聚合语义知识增强视觉检测器性能。**

- **链接: [https://arxiv.org/pdf/2511.10923v1](https://arxiv.org/pdf/2511.10923v1)**

> **作者:** Zhixia He; Chen Zhao; Minglai Shao; Xintao Wu; Xujiang Zhao; Dong Li; Qin Tian; Linlin Yu
>
> **摘要:** Out-of-distribution (OOD) detection is committed to delineating the classification boundaries between in-distribution (ID) and OOD images. Recent advances in vision-language models (VLMs) have demonstrated remarkable OOD detection performance by integrating both visual and textual modalities. In this context, negative prompts are introduced to emphasize the dissimilarity between image features and prompt content. However, these prompts often include a broad range of non-ID features, which may result in suboptimal outcomes due to the capture of overlapping or misleading information. To address this issue, we propose Positive and Negative Prompt Supervision, which encourages negative prompts to capture inter-class features and transfers this semantic knowledge to the visual modality to enhance OOD detection performance. Our method begins with class-specific positive and negative prompts initialized by large language models (LLMs). These prompts are subsequently optimized, with positive prompts focusing on features within each class, while negative prompts highlight features around category boundaries. Additionally, a graph-based architecture is employed to aggregate semantic-aware supervision from the optimized prompt representations and propagate it to the visual branch, thereby enhancing the performance of the energy-based OOD detector. Extensive experiments on two benchmarks, CIFAR-100 and ImageNet-1K, across eight OOD datasets and five different LLMs, demonstrate that our method outperforms state-of-the-art baselines.
>
---
#### [new 041] Discovering Meaningful Units with Visually Grounded Semantics from Image Captions
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉-语言模型的细粒度理解任务，解决现有方法仅对齐图像块与标记而忽略标记分组的问题。提出模型分组图像描述标记，使语言表示匹配图像对象，提升细粒度理解，发现的标记组与可接地短语高度相似。**

- **链接: [https://arxiv.org/pdf/2511.11262v1](https://arxiv.org/pdf/2511.11262v1)**

> **作者:** Melika Behjati; James Henderson
>
> **摘要:** Fine-grained knowledge is crucial for vision-language models to obtain a better understanding of the real world. While there has been work trying to acquire this kind of knowledge in the space of vision and language, it has mostly focused on aligning the image patches with the tokens on the language side. However, image patches do not have any meaning to the human eye, and individual tokens do not necessarily carry groundable information in the image. It is groups of tokens which describe different aspects of the scene. In this work, we propose a model which groups the caption tokens as part of its architecture in order to capture a fine-grained representation of the language. We expect our representations to be at the level of objects present in the image, and therefore align our representations with the output of an image encoder trained to discover objects. We show that by learning to group the tokens, the vision-language model has a better fine-grained understanding of vision and language. In addition, the token groups that our model discovers are highly similar to groundable phrases in text, both qualitatively and quantitatively.
>
---
#### [new 042] Computationally-efficient deep learning models for nowcasting of precipitation: A solution for the Weather4cast 2025 challenge
- **分类: cs.CV**

- **简介: 该论文针对Weather4Cast 2025挑战赛，提出基于ConvGRU的迁移学习框架用于降水短临预报。输入SEVIRI红外数据，采用两阶段训练预测降雨率，累积降雨任务获第二名，事件预测表现与基线相当。**

- **链接: [https://arxiv.org/pdf/2511.11197v1](https://arxiv.org/pdf/2511.11197v1)**

> **作者:** Anushree Bhuskute; Kaushik Gopalan; Jeet Shah
>
> **摘要:** This study presents a transfer-learning framework based on Convolutional Gated Recurrent Units (ConvGRU) for short-term rainfall prediction in the Weather4Cast 2025 competition. A single SEVIRI infrared channel (10.8 μm wavelength) is used as input, which consists of four observations over a one-hour period. A two-stage training strategy is applied to generate rainfall estimates up to four hours ahead. In the first stage, ConvGRU is trained to forecast the brightness temperatures from SEVIRI, enabling the model to capture relevant spatiotemporal patterns. In the second stage, an empirically derived nonlinear transformation maps the predicted fields to OPERA-compatible rainfall rates. For the event-prediction task, the transformed rainfall forecasts are processed using 3D event detection followed by spatiotemporal feature extraction to identify and characterize precipitation events. Our submission achieved 2nd place in the cumulative rainfall task. Further, the same model was used out-of-the-box for the event prediction task, and resulted in similar scores as the baseline model to the competition.
>
---
#### [new 043] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization
- **分类: cs.CV**

- **简介: ERMoE解决MoE架构的路由不稳定和负载不平衡问题。通过将专家重参数化到正交特征基，用输入与专家基的余弦相似度作为路由分数，实现稳定利用和可解释专业化，无需平衡损失。在ImageNet、跨模态检索和脑龄预测中显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.10971v1](https://arxiv.org/pdf/2511.10971v1)**

> **作者:** Anzhe Cheng; Shukai Duan; Shixuan Li; Chenzhong Yin; Mingxi Cheng; Heng Ping; Tamoghna Chattopadhyay; Sophia I Thomopoulos; Shahin Nazarian; Paul Thompson; Paul Bogdan
>
> **摘要:** Mixture-of-Experts (MoE) architectures expand model capacity by sparsely activating experts but face two core challenges: misalignment between router logits and each expert's internal structure leads to unstable routing and expert underutilization, and load imbalances create straggler bottlenecks. Standard solutions, such as auxiliary load-balancing losses, can reduce load disparities but often weaken expert specialization and hurt downstream performance. To address these issues, we propose ERMoE, a sparse MoE transformer that reparameterizes each expert in a learned orthonormal eigenbasis and replaces learned gating logits with an "Eigenbasis Score", defined as the cosine similarity between input features and an expert's basis. This content-aware routing ties token assignments directly to experts' representation spaces, stabilizing utilization and promoting interpretable specialization without sacrificing sparsity. Crucially, ERMoE removes the need for explicit balancing losses and avoids the interfering gradients they introduce. We show that ERMoE achieves state-of-the-art accuracy on ImageNet classification and cross-modal image-text retrieval benchmarks (e.g., COCO, Flickr30K), while naturally producing flatter expert load distributions. Moreover, a 3D MRI variant (ERMoE-ba) improves brain age prediction accuracy by more than 7\% and yields anatomically interpretable expert specializations. ERMoE thus introduces a new architectural principle for sparse expert models that directly addresses routing instabilities and enables improved performance with scalable, interpretable specialization.
>
---
#### [new 044] Parameter-Efficient MoE LoRA for Few-Shot Multi-Style Editing
- **分类: cs.CV**

- **简介: 该论文针对少样本多风格图像编辑任务，提出参数高效的MoE LoRA框架。通过风格特定与共享路由机制实现多风格联合微调，自动优化秩分配并集成对抗学习与流匹配，显著减少参数量且性能超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.11236v1](https://arxiv.org/pdf/2511.11236v1)**

> **作者:** Cong Cao; Yujie Xu; Xiaodong Xu
>
> **摘要:** In recent years, image editing has garnered growing attention. However, general image editing models often fail to produce satisfactory results when confronted with new styles. The challenge lies in how to effectively fine-tune general image editing models to new styles using only a limited amount of paired data. To address this issue, this paper proposes a novel few-shot style editing framework. For this task, we construct a benchmark dataset that encompasses five distinct styles. Correspondingly, we propose a parameter-efficient multi-style Mixture-of-Experts Low-Rank Adaptation (MoE LoRA) with style-specific and style-shared routing mechanisms for jointly fine-tuning multiple styles. The style-specific routing ensures that different styles do not interfere with one another, while the style-shared routing adaptively allocates shared MoE LoRAs to learn common patterns. Our MoE LoRA can automatically determine the optimal ranks for each layer through a novel metric-guided approach that estimates the importance score of each single-rank component. Additionally, we explore the optimal location to insert LoRA within the Diffusion in Transformer (DiT) model and integrate adversarial learning and flow matching to guide the diffusion training process. Experimental results demonstrate that our proposed method outperforms existing state-of-the-art approaches with significantly fewer LoRA parameters.
>
---
#### [new 045] Disentangling Emotional Bases and Transient Fluctuations: A Low-Rank Sparse Decomposition Approach for Video Affective Analysis
- **分类: cs.CV**

- **简介: 论文聚焦视频情感分析任务，解决情感动态复杂导致的模型不稳定与表示退化问题。提出LSEF框架，基于低秩稀疏分解原理，有效分离情感基础（长期基调）与瞬时波动（短期变化），显著提升分析鲁棒性与动态判别力。**

- **链接: [https://arxiv.org/pdf/2511.11406v1](https://arxiv.org/pdf/2511.11406v1)**

> **作者:** Feng-Qi Cui; Jinyang Huang; Ziyu Jia; Xinyu Li; Xin Yan; Xiaokang Zhou; Meng Wang
>
> **摘要:** Video-based Affective Computing (VAC), vital for emotion analysis and human-computer interaction, suffers from model instability and representational degradation due to complex emotional dynamics. Since the meaning of different emotional fluctuations may differ under different emotional contexts, the core limitation is the lack of a hierarchical structural mechanism to disentangle distinct affective components, i.e., emotional bases (the long-term emotional tone), and transient fluctuations (the short-term emotional fluctuations). To address this, we propose the Low-Rank Sparse Emotion Understanding Framework (LSEF), a unified model grounded in the Low-Rank Sparse Principle, which theoretically reframes affective dynamics as a hierarchical low-rank sparse compositional process. LSEF employs three plug-and-play modules, i.e., the Stability Encoding Module (SEM) captures low-rank emotional bases; the Dynamic Decoupling Module (DDM) isolates sparse transient signals; and the Consistency Integration Module (CIM) reconstructs multi-scale stability and reactivity coherence. This framework is optimized by a Rank Aware Optimization (RAO) strategy that adaptively balances gradient smoothness and sensitivity. Extensive experiments across multiple datasets confirm that LSEF significantly enhances robustness and dynamic discrimination, which further validates the effectiveness and generality of hierarchical low-rank sparse modeling for understanding affective dynamics.
>
---
#### [new 046] SP-Guard: Selective Prompt-adaptive Guidance for Safe Text-to-Image Generation
- **分类: cs.CV; cs.CY**

- **简介: SP-Guard解决文本到图像生成的安全问题，通过估计提示有害性并应用选择性引导掩码，仅针对不安全区域进行引导，提升安全性同时最小化内容改变。**

- **链接: [https://arxiv.org/pdf/2511.11014v1](https://arxiv.org/pdf/2511.11014v1)**

> **作者:** Sumin Yu; Taesup Moon
>
> **备注:** Accepted for presentation at TRUST-AI Workshop, ECAI 2025. Proceedings to appear in CEUR-WS
>
> **摘要:** While diffusion-based T2I models have achieved remarkable image generation quality, they also enable easy creation of harmful content, raising social concerns and highlighting the need for safer generation. Existing inference-time guiding methods lack both adaptivity--adjusting guidance strength based on the prompt--and selectivity--targeting only unsafe regions of the image. Our method, SP-Guard, addresses these limitations by estimating prompt harmfulness and applying a selective guidance mask to guide only unsafe areas. Experiments show that SP-Guard generates safer images than existing methods while minimizing unintended content alteration. Beyond improving safety, our findings highlight the importance of transparency and controllability in image generation.
>
---
#### [new 047] MeCaMIL: Causality-Aware Multiple Instance Learning for Fair and Interpretable Whole Slide Image Diagnosis
- **分类: cs.CV**

- **简介: 论文提出MeCaMIL框架，解决WSI诊断中MIL方法的因果可解释性不足和公平性问题。通过结构化因果图建模人口统计混杂因素，利用do-calculus分离疾病信号，显著提升诊断性能（ACC 0.939+）和公平性（人口差异方差降65%+）。**

- **链接: [https://arxiv.org/pdf/2511.11004v1](https://arxiv.org/pdf/2511.11004v1)**

> **作者:** Yiran Song; Yikai Zhang; Shuang Zhou; Guojun Xiong; Xiaofeng Yang; Nian Wang; Fenglong Ma; Rui Zhang; Mingquan Lin
>
> **备注:** 15page,5 figures,8 tables
>
> **摘要:** Multiple instance learning (MIL) has emerged as the dominant paradigm for whole slide image (WSI) analysis in computational pathology, achieving strong diagnostic performance through patch-level feature aggregation. However, existing MIL methods face critical limitations: (1) they rely on attention mechanisms that lack causal interpretability, and (2) they fail to integrate patient demographics (age, gender, race), leading to fairness concerns across diverse populations. These shortcomings hinder clinical translation, where algorithmic bias can exacerbate health disparities. We introduce \textbf{MeCaMIL}, a causality-aware MIL framework that explicitly models demographic confounders through structured causal graphs. Unlike prior approaches treating demographics as auxiliary features, MeCaMIL employs principled causal inference -- leveraging do-calculus and collider structures -- to disentangle disease-relevant signals from spurious demographic correlations. Extensive evaluation on three benchmarks demonstrates state-of-the-art performance across CAMELYON16 (ACC/AUC/F1: 0.939/0.983/0.946), TCGA-Lung (0.935/0.979/0.931), and TCGA-Multi (0.977/0.993/0.970, five cancer types). Critically, MeCaMIL achieves superior fairness -- demographic disparity variance drops by over 65% relative reduction on average across attributes, with notable improvements for underserved populations. The framework generalizes to survival prediction (mean C-index: 0.653, +0.017 over best baseline across five cancer types). Ablation studies confirm causal graph structure is essential -- alternative designs yield 0.048 lower accuracy and 4.2x times worse fairness. These results establish MeCaMIL as a principled framework for fair, interpretable, and clinically actionable AI in digital pathology. Code will be released upon acceptance.
>
---
#### [new 048] Unsupervised Segmentation of Micro-CT Scans of Polyurethane Structures By Combining Hidden-Markov-Random Fields and a U-Net
- **分类: cs.CV**

- **简介: 论文提出HMRF-UNet模型，结合隐马尔可夫随机场与U-Net，解决无监督分割中精度低、需大量标注数据的问题。该模型实现Polyurethane泡沫μCT图像的高精度无监督分割，并通过预训练策略显著减少标注数据需求。**

- **链接: [https://arxiv.org/pdf/2511.11378v1](https://arxiv.org/pdf/2511.11378v1)**

> **作者:** Julian Grolig; Lars Griem; Michael Selzer; Hans-Ulrich Kauczor; Simon M. F. Triphan; Britta Nestler; Arnd Koeppe
>
> **摘要:** Extracting digital material representations from images is a necessary prerequisite for a quantitative analysis of material properties. Different segmentation approaches have been extensively studied in the past to achieve this task, but were often lacking accuracy or speed. With the advent of machine learning, supervised convolutional neural networks (CNNs) have achieved state-of-the-art performance for different segmentation tasks. However, these models are often trained in a supervised manner, which requires large labeled datasets. Unsupervised approaches do not require ground-truth data for learning, but suffer from long segmentation times and often worse segmentation accuracy. Hidden Markov Random Fields (HMRF) are an unsupervised segmentation approach that incorporates concepts of neighborhood and class distributions. We present a method that integrates HMRF theory and CNN segmentation, leveraging the advantages of both areas: unsupervised learning and fast segmentation times. We investigate the contribution of different neighborhood terms and components for the unsupervised HMRF loss. We demonstrate that the HMRF-UNet enables high segmentation accuracy without ground truth on a Micro-Computed Tomography ($μ$CT) image dataset of Polyurethane (PU) foam structures. Finally, we propose and demonstrate a pre-training strategy that considerably reduces the required amount of ground-truth data when training a segmentation model.
>
---
#### [new 049] Facial Expression Recognition with YOLOv11 and YOLOv12: A Comparative Study
- **分类: cs.CV**

- **简介: 论文研究面部表情识别任务，比较YOLOv11n和YOLOv12n在FER2013和KDEF数据集上的性能。YOLOv12n在KDEF上mAP 95.6最优，YOLOv11n在FER2013精度65.2更高，揭示敏感性与精度的权衡，适用于实时资源受限AI应用。**

- **链接: [https://arxiv.org/pdf/2511.10940v1](https://arxiv.org/pdf/2511.10940v1)**

> **作者:** Umma Aymon; Nur Shazwani Kamarudin; Ahmad Fakhri Ab. Nasir
>
> **备注:** IEEE Conference Proceedings for the 2025 IEEE 9th International Conference on Software Engineering & Computer Systems (ICSECS)
>
> **摘要:** Facial Expression Recognition remains a challenging task, especially in unconstrained, real-world environments. This study investigates the performance of two lightweight models, YOLOv11n and YOLOv12n, which are the nano variants of the latest official YOLO series, within a unified detection and classification framework for FER. Two benchmark classification datasets, FER2013 and KDEF, are converted into object detection format and model performance is evaluated using mAP 0.5, precision, recall, and confusion matrices. Results show that YOLOv12n achieves the highest overall performance on the clean KDEF dataset with a mAP 0.5 of 95.6, and also outperforms YOLOv11n on the FER2013 dataset in terms of mAP 63.8, reflecting stronger sensitivity to varied expressions. In contrast, YOLOv11n demonstrates higher precision 65.2 on FER2013, indicating fewer false positives and better reliability in noisy, real-world conditions. On FER2013, both models show more confusion between visually similar expressions, while clearer class separation is observed on the cleaner KDEF dataset. These findings underscore the trade-off between sensitivity and precision, illustrating how lightweight YOLO models can effectively balance performance and efficiency. The results demonstrate adaptability across both controlled and real-world conditions, establishing these models as strong candidates for real-time, resource-constrained emotion-aware AI applications.
>
---
#### [new 050] SUPER Decoder Block for Reconstruction-Aware U-Net Variants
- **分类: cs.CV**

- **简介: 论文针对U-Net变体在图像逆问题（如裂缝分割和去噪）中因信息损失导致高频率细节恢复不足的问题，提出SUPER Decoder Block。它利用小波完美重建特性选择性抑制冗余特征，作为即插即用解码器提升表示丰富性，实验验证在裂缝分割和图像去噪中有效。**

- **链接: [https://arxiv.org/pdf/2511.11015v1](https://arxiv.org/pdf/2511.11015v1)**

> **作者:** Siheon Joo; Hongjo Kim
>
> **备注:** 8 pages. Under review
>
> **摘要:** Skip-connected encoder-decoder architectures (U-Net variants) are widely adopted for inverse problems but still suffer from information loss, limiting recovery of fine high-frequency details. We present Selectively Suppressed Perfect Reconstruction (SUPER), which exploits the perfect reconstruction (PR) property of wavelets to prevent information degradation while selectively suppressing (SS) redundant features. Free from rigid framelet constraints, SUPER serves as a plug-and-play decoder block for diverse U-Net variants, eliminating their intrinsic reconstruction bottlenecks and enhancing representational richness. Experiments across diverse crack benchmarks, including state-of-the-art (SOTA) models, demonstrate the structural potential of the proposed SUPER Decoder Block. Maintaining comparable computational cost, SUPER enriches representational diversity through increased parameterization. In small-scale in-domain experiments on the CrackVision12K dataset, SUPER markedly improves thin-crack segmentation performance, particularly for cracks narrower than 4 px, underscoring its advantage in high-frequency dominant settings. In smartphone image denoising on SIDD, where low-frequency components prevail, SUPER still achieves a moderate gain in PSNR, confirming its robustness across low- and high-frequency regimes. These results validate its plug-and-play generality across U-Net variants, achieving high-frequency fidelity and global coherence within a unified, reconstruction-aware framework.
>
---
#### [new 051] Bridging Hidden States in Vision-Language Models
- **分类: cs.CV**

- **简介: 论文解决视觉-语言模型隐藏状态对齐问题。现有方法融合不充分，未利用模态特定结构。提出BRIDGE模块：通过轻量级交叉注意力层直接对齐视觉与文本隐藏状态，保持编码器非因果并解耦生成。在检索、VQA等任务上优于现有模型。**

- **链接: [https://arxiv.org/pdf/2511.11526v1](https://arxiv.org/pdf/2511.11526v1)**

> **作者:** Benjamin Fein-Ashley; Jacob Fein-Ashley
>
> **摘要:** Vision-Language Models (VLMs) are a new family of models that align image content with natural language. Existing approaches typically fuse either (a) early: by mixing tokens/features inside the encoders, or (b) late: by comparing pooled embeddings. Many methods also tie fusion to an autoregressive decoder. However, the hidden states of both modalities already carry rich, modality-specific structure (spatial layout in vision; syntax and semantics in text), so directly aligning these states is a natural way to match what the two modalities "think". We propose a lightweight fusion module: a few cross-only, bidirectional attention layers placed near the top of both encoders. Each layer projects the vision and text encoder hidden-state sequences into a shared space, attends across modalities, and sends gated residual updates back, with simple stabilizers to improve alignment. The encoders remain non-causal and strong for understanding, while generation stays cleanly decoupled via an optional decoder. Across standard retrieval, VQA, and visual reasoning benchmarks, BRIDGE outperforms comparable VLMs while preserving the bi-encoder efficiency of contrastive models. We make our code publicly available at https://github.com/jfeinashley/BRIDGE.
>
---
#### [new 052] LiteAttention: A Temporal Sparse Attention for Diffusion Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视频生成扩散Transformer的注意力二次复杂度问题，提出LiteAttention方法。利用注意力稀疏模式的时间相干性，通过早期标记非关键块并传播跳过决策，实现高效稀疏注意力计算，显著加速模型且无质量损失。**

- **链接: [https://arxiv.org/pdf/2511.11062v1](https://arxiv.org/pdf/2511.11062v1)**

> **作者:** Dor Shmilovich; Tony Wu; Aviad Dahan; Yuval Domb
>
> **摘要:** Diffusion Transformers, particularly for video generation, achieve remarkable quality but suffer from quadratic attention complexity, leading to prohibitive latency. Existing acceleration methods face a fundamental trade-off: dynamically estimating sparse attention patterns at each denoising step incurs high computational overhead and estimation errors, while static sparsity patterns remain fixed and often suboptimal throughout denoising. We identify a key structural property of diffusion attention, namely, its sparsity patterns exhibit strong temporal coherence across denoising steps. Tiles deemed non-essential at step $t$ typically remain so at step $t+δ$. Leveraging this observation, we introduce LiteAttention, a method that exploits temporal coherence to enable evolutionary computation skips across the denoising sequence. By marking non-essential tiles early and propagating skip decisions forward, LiteAttention eliminates redundant attention computations without repeated profiling overheads, combining the adaptivity of dynamic methods with the efficiency of static ones. We implement a highly optimized LiteAttention kernel on top of FlashAttention and demonstrate substantial speedups on production video diffusion models, with no degradation in quality. The code and implementation details will be publicly released.
>
---
#### [new 053] VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation
- **分类: cs.CV; cs.LG**

- **简介: VoxTell是视觉-语言模型，用于3D医学图像分割任务。它解决自由文本提示驱动的通用分割问题，通过多阶段视觉-语言融合实现文本到3D掩码的映射，支持零样本跨模态分割和临床语言处理。**

- **链接: [https://arxiv.org/pdf/2511.11450v1](https://arxiv.org/pdf/2511.11450v1)**

> **作者:** Maximilian Rokuss; Moritz Langenberg; Yannick Kirchhoff; Fabian Isensee; Benjamin Hamm; Constantin Ulrich; Sebastian Regnery; Lukas Bauer; Efthimios Katsigiannopulos; Tobias Norajitra; Klaus Maier-Hein
>
> **摘要:** We introduce VoxTell, a vision-language model for text-prompted volumetric medical image segmentation. It maps free-form descriptions, from single words to full clinical sentences, to 3D masks. Trained on 62K+ CT, MRI, and PET volumes spanning over 1K anatomical and pathological classes, VoxTell uses multi-stage vision-language fusion across decoder layers to align textual and visual features at multiple scales. It achieves state-of-the-art zero-shot performance across modalities on unseen datasets, excelling on familiar concepts while generalizing to related unseen classes. Extensive experiments further demonstrate strong cross-modality transfer, robustness to linguistic variations and clinical language, as well as accurate instance-specific segmentation from real-world text. Code is available at: https://www.github.com/MIC-DKFZ/VoxTell
>
---
#### [new 054] Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Short-Window Sliding Learning框架解决实时暴力检测问题。将视频分1-2秒片段，用LLM自动标注构建细粒度数据集，保留时间连续性。实验在RWF-2000（95.25%）和UCF-Crime（83.25%）上验证高效实时性。**

- **链接: [https://arxiv.org/pdf/2511.10866v1](https://arxiv.org/pdf/2511.10866v1)**

> **作者:** Seoik Jung; Taekyung Song; Yangro Lee; Sungjun Lee
>
> **备注:** 5 pages, 2 figures. Accepted paper for the IEIE (Institute of Electronics and Information Engineers) Fall Conference 2025. Presentation on Nov 27, 2025
>
> **摘要:** This paper proposes a Short-Window Sliding Learning framework for real-time violence detection in CCTV footages. Unlike conventional long-video training approaches, the proposed method divides videos into 1-2 second clips and applies Large Language Model (LLM)-based auto-caption labeling to construct fine-grained datasets. Each short clip fully utilizes all frames to preserve temporal continuity, enabling precise recognition of rapid violent events. Experiments demonstrate that the proposed method achieves 95.25\% accuracy on RWF-2000 and significantly improves performance on long videos (UCF-Crime: 83.25\%), confirming its strong generalization and real-time applicability in intelligent surveillance systems.
>
---
#### [new 055] Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos
- **分类: cs.CV**

- **简介: 论文解决多视角动态场景重建中视频时间不同步问题。提出粗到细时间对齐模块，估计并补偿相机时间偏移，实现子帧精度对齐。该方法可无缝集成到4DGS框架，显著提升异步视频重建质量。**

- **链接: [https://arxiv.org/pdf/2511.11175v1](https://arxiv.org/pdf/2511.11175v1)**

> **作者:** Zhixin Xu; Hengyu Zhou; Yuan Liu; Wenhan Xue; Hao Pan; Wenping Wang; Bin Wang
>
> **备注:** AAAI 2026
>
> **摘要:** Multi-view video reconstruction plays a vital role in computer vision, enabling applications in film production, virtual reality, and motion analysis. While recent advances such as 4D Gaussian Splatting (4DGS) have demonstrated impressive capabilities in dynamic scene reconstruction, they typically rely on the assumption that input video streams are temporally synchronized. However, in real-world scenarios, this assumption often fails due to factors like camera trigger delays or independent recording setups, leading to temporal misalignment across views and reduced reconstruction quality. To address this challenge, a novel temporal alignment strategy is proposed for high-quality 4DGS reconstruction from unsynchronized multi-view videos. Our method features a coarse-to-fine alignment module that estimates and compensates for each camera's time shift. The method first determines a coarse, frame-level offset and then refines it to achieve sub-frame accuracy. This strategy can be integrated as a readily integrable module into existing 4DGS frameworks, enhancing their robustness when handling asynchronous data. Experiments show that our approach effectively processes temporally misaligned videos and significantly enhances baseline methods.
>
---
#### [new 056] PAS : Prelim Attention Score for Detecting Object Hallucinations in Large Vision--Language Models
- **分类: cs.CV; cs.AI**

- **简介: 论文针对大型视觉语言模型的对象幻觉问题，发现模型常依赖先前生成的prelim tokens而非图像。提出Prelim Attention Score (PAS)，基于注意力权重的轻量级信号，无需训练即可实时检测幻觉，性能领先。**

- **链接: [https://arxiv.org/pdf/2511.11502v1](https://arxiv.org/pdf/2511.11502v1)**

> **作者:** Nhat Hoang-Xuan; Minh Vu; My T. Thai; Manish Bhattarai
>
> **摘要:** Large vision-language models (LVLMs) are powerful, yet they remain unreliable due to object hallucinations. In this work, we show that in many hallucinatory predictions the LVLM effectively ignores the image and instead relies on previously generated output (prelim) tokens to infer new objects. We quantify this behavior via the mutual information between the image and the predicted object conditioned on the prelim, demonstrating that weak image dependence strongly correlates with hallucination. Building on this finding, we introduce the Prelim Attention Score (PAS), a lightweight, training-free signal computed from attention weights over prelim tokens. PAS requires no additional forward passes and can be computed on the fly during inference. Exploiting this previously overlooked signal, PAS achieves state-of-the-art object-hallucination detection across multiple models and datasets, enabling real-time filtering and intervention.
>
---
#### [new 057] SplineSplat: 3D Ray Tracing for Higher-Quality Tomography
- **分类: cs.CV; eess.IV; eess.SP**

- **简介: 论文提出SplineSplat方法，利用B样条表示3D体积并结合神经网络优化射线追踪算法，解决断层成像重建质量低的问题，实现高精度投影计算，无需正则化，显著优于传统体素方法。**

- **链接: [https://arxiv.org/pdf/2511.11078v1](https://arxiv.org/pdf/2511.11078v1)**

> **作者:** Youssef Haouchat; Sepand Kashani; Aleix Boquet-Pujadas; Philippe Thévenaz; Michael Unser
>
> **摘要:** We propose a method to efficiently compute tomographic projections of a 3D volume represented by a linear combination of shifted B-splines. To do so, we propose a ray-tracing algorithm that computes 3D line integrals with arbitrary projection geometries. One of the components of our algorithm is a neural network that computes the contribution of the basis functions efficiently. In our experiments, we consider well-posed cases where the data are sufficient for accurate reconstruction without the need for regularization. We achieve higher reconstruction quality than traditional voxel-based methods.
>
---
#### [new 058] S2D-ALIGN: Shallow-to-Deep Auxiliary Learning for Anatomically-Grounded Radiology Report Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对放射学报告生成任务，解决现有方法仅依赖实例级对齐导致解剖学基础缺失的问题。提出S2D-ALIGN，采用浅到深辅助学习策略：从粗略图像-报告配对逐步引入参考报告和关键短语，利用记忆适配器整合多级指导，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.11066v1](https://arxiv.org/pdf/2511.11066v1)**

> **作者:** Jiechao Gao; Chang Liu; Yuangang Li
>
> **摘要:** Radiology Report Generation (RRG) aims to automatically generate diagnostic reports from radiology images. To achieve this, existing methods have leveraged the powerful cross-modal generation capabilities of Multimodal Large Language Models (MLLMs), primarily focusing on optimizing cross-modal alignment between radiographs and reports through Supervised Fine-Tuning (SFT). However, by only performing instance-level alignment with the image-text pairs, the standard SFT paradigm fails to establish anatomically-grounded alignment, where the templated nature of reports often leads to sub-optimal generation quality. To address this, we propose \textsc{S2D-Align}, a novel SFT paradigm that establishes anatomically-grounded alignment by leveraging auxiliary signals of varying granularities. \textsc{S2D-Align} implements a shallow-to-deep strategy, progressively enriching the alignment process: it begins with the coarse radiograph-report pairing, then introduces reference reports for instance-level guidance, and ultimately utilizes key phrases to ground the generation in specific anatomical details. To bridge the different alignment stages, we introduce a memory-based adapter that empowers feature sharing, thereby integrating coarse and fine-grained guidance. For evaluation, we conduct experiments on the public \textsc{MIMIC-CXR} and \textsc{IU X-Ray} benchmarks, where \textsc{S2D-Align} achieves state-of-the-art performance compared to existing methods. Ablation studies validate the effectiveness of our multi-stage, auxiliary-guided approach, highlighting a promising direction for enhancing grounding capabilities in complex, multi-modal generation tasks.
>
---
#### [new 059] DocSLM: A Small Vision-Language Model for Long Multimodal Document Understanding
- **分类: cs.CV**

- **简介: DocSLM针对长多模态文档理解任务，解决LVLMs内存高、不适用于边缘设备的问题。提出小型模型，通过分层多模态压缩器和流式回避机制，减少82%视觉令牌、75%参数和71%延迟，性能匹配或超越SOTA。**

- **链接: [https://arxiv.org/pdf/2511.11313v1](https://arxiv.org/pdf/2511.11313v1)**

> **作者:** Tanveer Hannan; Dimitrios Mallios; Parth Pathak; Faegheh Sardari; Thomas Seidl; Gedas Bertasius; Mohsen Fayyaz; Sunando Sengupta
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated strong multimodal reasoning capabilities on long and complex documents. However, their high memory footprint makes them impractical for deployment on resource-constrained edge devices. We present DocSLM, an efficient Small Vision-Language Model designed for long-document understanding under constrained memory resources. DocSLM incorporates a Hierarchical Multimodal Compressor that jointly encodes visual, textual, and layout information from each page into a fixed-length sequence, greatly reducing memory consumption while preserving both local and global semantics. To enable scalable processing over arbitrarily long inputs, we introduce a Streaming Abstention mechanism that operates on document segments sequentially and filters low-confidence responses using an entropy-based uncertainty calibrator. Across multiple long multimodal document benchmarks, DocSLM matches or surpasses state-of-the-art methods while using 82\% fewer visual tokens, 75\% fewer parameters, and 71\% lower latency, delivering reliable multimodal document understanding on lightweight edge devices. Code is available in the supplementary material.
>
---
#### [new 060] Multimodal Posterior Sampling-based Uncertainty in PD-L1 Segmentation from H&E Images
- **分类: cs.CV; q-bio.QM**

- **简介: 论文提出nnUNet-B框架，用于H&E图像中PD-L1表达分割。通过多模态后验采样实现贝叶斯分割和不确定性估计，解决IHC方法资源密集问题。在肺鳞癌数据集上，Dice 0.805，IoU 0.709，提供像素级不确定性图并与分割误差强相关。**

- **链接: [https://arxiv.org/pdf/2511.11486v1](https://arxiv.org/pdf/2511.11486v1)**

> **作者:** Roman Kinakh; Gonzalo R. Ríos-Muñoz; Arrate Muñoz-Barrutia
>
> **备注:** Preprint (pre-review). Accepted for publication in Lecture Notes in Bioinformatics (Springer, 2025). The final authenticated version will be available on SpringerLink once published
>
> **摘要:** Accurate assessment of PD-L1 expression is critical for guiding immunotherapy, yet current immunohistochemistry (IHC) based methods are resource-intensive. We present nnUNet-B: a Bayesian segmentation framework that infers PD-L1 expression directly from H&E-stained histology images using Multimodal Posterior Sampling (MPS). Built upon nnUNet-v2, our method samples diverse model checkpoints during cyclic training to approximate the posterior, enabling both accurate segmentation and epistemic uncertainty estimation via entropy and standard deviation. Evaluated on a dataset of lung squamous cell carcinoma, our approach achieves competitive performance against established baselines with mean Dice Score and mean IoU of 0.805 and 0.709, respectively, while providing pixel-wise uncertainty maps. Uncertainty estimates show strong correlation with segmentation error, though calibration remains imperfect. These results suggest that uncertainty-aware H&E-based PD-L1 prediction is a promising step toward scalable, interpretable biomarker assessment in clinical workflows.
>
---
#### [new 061] GraphPilot: Grounded Scene Graph Conditioning for Language-Based Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文针对语言驱动自动驾驶任务，解决现有模型缺乏关系监督的问题。提出GraphPilot方法，通过结构化场景图条件化训练，显著提升驾驶性能（LMDrive提升15.6%），无需测试时输入场景图。**

- **链接: [https://arxiv.org/pdf/2511.11266v1](https://arxiv.org/pdf/2511.11266v1)**

> **作者:** Fabian Schmidt; Markus Enzweiler; Abhinav Valada
>
> **摘要:** Vision-language models have recently emerged as promising planners for autonomous driving, where success hinges on topology-aware reasoning over spatial structure and dynamic interactions from multimodal input. However, existing models are typically trained without supervision that explicitly encodes these relational dependencies, limiting their ability to infer how agents and other traffic entities influence one another from raw sensor data. In this work, we bridge this gap with a novel model-agnostic method that conditions language-based driving models on structured relational context in the form of traffic scene graphs. We serialize scene graphs at various abstraction levels and formats, and incorporate them into the models via structured prompt templates, enabling a systematic analysis of when and how relational supervision is most beneficial. Extensive evaluations on the public LangAuto benchmark show that scene graph conditioning of state-of-the-art approaches yields large and persistent improvement in driving performance. Notably, we observe up to a 15.6\% increase in driving score for LMDrive and 17.5\% for BEVDriver, indicating that models can better internalize and ground relational priors through scene graph-conditioned training, even without requiring scene graph input at test-time. Code, fine-tuned models, and our scene graph dataset are publicly available at https://github.com/iis-esslingen/GraphPilot.
>
---
#### [new 062] Rethinking Efficient Mixture-of-Experts for Remote Sensing Modality-Missing Classification
- **分类: cs.CV**

- **简介: 该论文针对远程传感多模态分类中的模态缺失问题，提出MaMOL框架，通过双路由机制实现参数高效适应，显著提升鲁棒性并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2511.11460v1](https://arxiv.org/pdf/2511.11460v1)**

> **作者:** Qinghao Gao; Jianhai Qu; Yunsong Li; Weiqiang Dong
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Multimodal classification in remote sensing often suffers from missing modalities caused by environmental interference, sensor failures, or atmospheric effects, which severely degrade classification performance. Existing two-stage adaptation methods are computationally expensive and assume complete multimodal data during training, limiting their generalization to real-world incompleteness. To overcome these issues, we propose a Missing-aware Mixture-of-Loras (MaMOL) framework that reformulates modality missing as a multi-task learning problem. MaMOL introduces a dual-routing mechanism: a task-oriented dynamic router that adaptively activates experts for different missing patterns, and a modality-specific-shared static router that maintains stable cross-modal knowledge sharing. Unlike prior methods that train separate networks for each missing configuration, MaMOL achieves parameter-efficient adaptation via lightweight expert updates and shared expert reuse. Experiments on multiple remote sensing benchmarks demonstrate superior robustness and generalization under varying missing rates, with minimal computational overhead. Moreover, transfer experiments on natural image datasets validate its scalability and cross-domain applicability, highlighting MaMOL as a general and efficient solution for incomplete multimodal learning.
>
---
#### [new 063] OT-ALD: Aligning Latent Distributions with Optimal Transport for Accelerated Image-to-Image Translation
- **分类: cs.CV; cs.AI**

- **简介: OT-ALD解决图像到图像翻译效率低和潜在分布不匹配问题。通过最优传输对齐源域与目标域分布，作为反向扩散起点，提升采样效率20.29%并降低FID分数2.6。**

- **链接: [https://arxiv.org/pdf/2511.11162v1](https://arxiv.org/pdf/2511.11162v1)**

> **作者:** Zhanpeng Wang; Shuting Cao; Yuhang Lu; Yuhan Li; Na Lei; Zhongxuan Luo
>
> **摘要:** The Dual Diffusion Implicit Bridge (DDIB) is an emerging image-to-image (I2I) translation method that preserves cycle consistency while achieving strong flexibility. It links two independently trained diffusion models (DMs) in the source and target domains by first adding noise to a source image to obtain a latent code, then denoising it in the target domain to generate the translated image. However, this method faces two key challenges: (1) low translation efficiency, and (2) translation trajectory deviations caused by mismatched latent distributions. To address these issues, we propose a novel I2I translation framework, OT-ALD, grounded in optimal transport (OT) theory, which retains the strengths of DDIB-based approach. Specifically, we compute an OT map from the latent distribution of the source domain to that of the target domain, and use the mapped distribution as the starting point for the reverse diffusion process in the target domain. Our error analysis confirms that OT-ALD eliminates latent distribution mismatches. Moreover, OT-ALD effectively balances faster image translation with improved image quality. Experiments on four translation tasks across three high-resolution datasets show that OT-ALD improves sampling efficiency by 20.29% and reduces the FID score by 2.6 on average compared to the top-performing baseline models.
>
---
#### [new 064] MAFM^3: Modular Adaptation of Foundation Models for Multi-Modal Medical AI
- **分类: cs.CV**

- **简介: 论文提出MAFM³框架，解决医学影像数据稀缺导致的多任务多模态模型适应难题。通过轻量级模块组件，使单个基础模型灵活扩展至不同任务与模态。实验证明在胸部CT预后和分割任务中性能提升，加入PET后Dice分数提高5%。**

- **链接: [https://arxiv.org/pdf/2511.11212v1](https://arxiv.org/pdf/2511.11212v1)**

> **作者:** Mohammad Areeb Qazi; Munachiso S Nwadike; Ibrahim Almakky; Mohammad Yaqub; Numan Saeed
>
> **备注:** 2 figures, 3 tables
>
> **摘要:** Foundational models are trained on extensive datasets to capture the general trends of a domain. However, in medical imaging, the scarcity of data makes pre-training for every domain, modality, or task challenging. Instead of building separate models, we propose MAFM^3 (Modular Adaptation of Foundation Models for Multi-Modal Medical AI), a framework that enables a single foundation model to expand into diverse domains, tasks, and modalities through lightweight modular components. These components serve as specialized skill sets that allow the system to flexibly activate the appropriate capability at the inference time, depending on the input type or clinical objective. Unlike conventional adaptation methods that treat each new task or modality in isolation, MAFM^3 provides a unified and expandable framework for efficient multitask and multimodality adaptation. Empirically, we validate our approach by adapting a chest CT foundation model initially trained for classification into prognosis and segmentation modules. Our results show improved performance on both tasks. Furthermore, by incorporating PET scans, MAFM^3 achieved an improvement in the Dice score 5% compared to the respective baselines. These findings establish that foundation models, when equipped with modular components, are not inherently constrained to their initial training scope but can evolve into multitask, multimodality systems for medical imaging. The code implementation of this work can be found at https://github.com/Areeb2735/CTscan_prognosis_VLM
>
---
#### [new 065] Rethinking Autoregressive Models for Lossless Image Compression via Hierarchical Parallelism and Progressive Adaptation
- **分类: cs.CV**

- **简介: 论文针对无损图像压缩任务，解决自回归模型计算成本高、不实用的问题。提出HPAC框架，通过分层并行和渐进适应（含CSI加速与AFC优化），实现高效压缩，达到新SOTA，参数少且编码速度快。**

- **链接: [https://arxiv.org/pdf/2511.10991v1](https://arxiv.org/pdf/2511.10991v1)**

> **作者:** Daxin Li; Yuanchao Bai; Kai Wang; Wenbo Zhao; Junjun Jiang; Xianming Liu
>
> **备注:** 15 pages
>
> **摘要:** Autoregressive (AR) models, the theoretical performance benchmark for learned lossless image compression, are often dismissed as impractical due to prohibitive computational cost. This work re-thinks this paradigm, introducing a framework built on hierarchical parallelism and progressive adaptation that re-establishes pure autoregression as a top-performing and practical solution. Our approach is embodied in the Hierarchical Parallel Autoregressive ConvNet (HPAC), an ultra-lightweight pre-trained model using a hierarchical factorized structure and content-aware convolutional gating to efficiently capture spatial dependencies. We introduce two key optimizations for practicality: Cache-then-Select Inference (CSI), which accelerates coding by eliminating redundant computations, and Adaptive Focus Coding (AFC), which efficiently extends the framework to high bit-depth images. Building on this efficient foundation, our progressive adaptation strategy is realized by Spatially-Aware Rate-Guided Progressive Fine-tuning (SARP-FT). This instance-level strategy fine-tunes the model for each test image by optimizing low-rank adapters on progressively larger, spatially-continuous regions selected via estimated information density. Experiments on diverse datasets (natural, satellite, medical) validate that our method achieves new state-of-the-art compression. Notably, our approach sets a new benchmark in learned lossless compression, showing a carefully designed AR framework can offer significant gains over existing methods with a small parameter count and competitive coding speeds.
>
---
#### [new 066] MPCGNet: A Multiscale Feature Extraction and Progressive Feature Aggregation Network Using Coupling Gates for Polyp Segmentation
- **分类: cs.CV**

- **简介: 论文提出MPCGNet解决息肉分割中的小息肉易漏、边界模糊和图像噪声问题。通过耦合门模块（CGMFE、WCAD、DFA）提取特征、抑制噪声并聚合特征，mDice分数提升2.20%和0.68%。**

- **链接: [https://arxiv.org/pdf/2511.11032v1](https://arxiv.org/pdf/2511.11032v1)**

> **作者:** Wei Wang; Feng Jiang; Xin Wang
>
> **备注:** 8 pages, 4 figures,3 tables. This paper has been accepted by IJCNN 2025 but not published
>
> **摘要:** Automatic segmentation methods of polyps is crucial for assisting doctors in colorectal polyp screening and cancer diagnosis. Despite the progress made by existing methods, polyp segmentation faces several challenges: (1) small-sized polyps are prone to being missed during identification, (2) the boundaries between polyps and the surrounding environment are often ambiguous, (3) noise in colonoscopy images, caused by uneven lighting and other factors, affects segmentation results. To address these challenges, this paper introduces coupling gates as components in specific modules to filter noise and perform feature importance selection. Three modules are proposed: the coupling gates multiscale feature extraction (CGMFE) module, which effectively extracts local features and suppresses noise; the windows cross attention (WCAD) decoder module, which restores details after capturing the precise location of polyps; and the decoder feature aggregation (DFA) module, which progressively aggregates features, further extracts them, and performs feature importance selection to reduce the loss of small-sized polyps. Experimental results demonstrate that MPCGNet outperforms recent networks, with mDice scores 2.20% and 0.68% higher than the second-best network on the ETIS-LaribPolypDB and CVC-ColonDB datasets, respectively.
>
---
#### [new 067] Abstract 3D Perception for Spatial Intelligence in Vision-Language Models
- **分类: cs.CV**

- **简介: 论文针对VLMs在3D空间任务中的不足，因2D训练导致3D信息检索效率低。提出SandboxVLM框架，通过抽象边界框编码几何结构，设计四阶段3D感知管道，显著提升空间智能，零样本下SAT Real任务提升8.3%。**

- **链接: [https://arxiv.org/pdf/2511.10946v1](https://arxiv.org/pdf/2511.10946v1)**

> **作者:** Yifan Liu; Fangneng Zhan; Kaichen Zhou; Yilun Du; Paul Pu Liang; Hanspeter Pfister
>
> **摘要:** Vision-language models (VLMs) struggle with 3D-related tasks such as spatial cognition and physical understanding, which are crucial for real-world applications like robotics and embodied agents. We attribute this to a modality gap between the 3D tasks and the 2D training of VLM, which led to inefficient retrieval of 3D information from 2D input. To bridge this gap, we introduce SandboxVLM, a simple yet effective framework that leverages abstract bounding boxes to encode geometric structure and physical kinematics for VLM. Specifically, we design a 3D Sandbox reconstruction and perception pipeline comprising four stages: generating multi-view priors with abstract control, proxy elevation, multi-view voting and clustering, and 3D-aware reasoning. Evaluated in zero-shot settings across multiple benchmarks and VLM backbones, our approach consistently improves spatial intelligence, achieving an 8.3\% gain on SAT Real compared with baseline methods for instance. These results demonstrate that equipping VLMs with a 3D abstraction substantially enhances their 3D reasoning ability without additional training, suggesting new possibilities for general-purpose embodied intelligence.
>
---
#### [new 068] OpenUS: A Fully Open-Source Foundation Model for Ultrasound Image Analysis via Self-Adaptive Masked Contrastive Learning
- **分类: cs.CV**

- **简介: OpenUS是首个开源超声基础模型，解决操作者依赖和标注不足问题。采用自适应掩码对比学习与视觉Mamba，构建308K图像数据集，支持高效下游任务微调。**

- **链接: [https://arxiv.org/pdf/2511.11510v1](https://arxiv.org/pdf/2511.11510v1)**

> **作者:** Xiaoyu Zheng; Xu Chen; Awais Rauf; Qifan Fu; Benedetta Monosi; Felice Rivellese; Myles J. Lewis; Shaogang Gong; Gregory Slabaugh
>
> **摘要:** Ultrasound (US) is one of the most widely used medical imaging modalities, thanks to its low cost, portability, real-time feedback, and absence of ionizing radiation. However, US image interpretation remains highly operator-dependent and varies significantly across anatomical regions, acquisition protocols, and device types. These variations, along with unique challenges such as speckle, low contrast, and limited standardized annotations, hinder the development of generalizable, label-efficient ultrasound AI models. In this paper, we propose OpenUS, the first reproducible, open-source ultrasound foundation model built on a large collection of public data. OpenUS employs a vision Mamba backbone, capturing both local and global long-range dependencies across the image. To extract rich features during pre-training, we introduce a novel self-adaptive masking framework that combines contrastive learning with masked image modeling. This strategy integrates the teacher's attention map with student reconstruction loss, adaptively refining clinically-relevant masking to enhance pre-training effectiveness. OpenUS also applies a dynamic learning schedule to progressively adjust the difficulty of the pre-training process. To develop the foundation model, we compile the largest to-date public ultrasound dataset comprising over 308K images from 42 publicly available datasets, covering diverse anatomical regions, institutions, imaging devices, and disease types. Our pre-trained OpenUS model can be easily adapted to specific downstream tasks by serving as a backbone for label-efficient fine-tuning. Code is available at https://github.com/XZheng0427/OpenUS.
>
---
#### [new 069] Hindsight Distillation Reasoning with Knowledge Encouragement Preference for Knowledge-based Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文针对KBVQA任务，解决现有方法推理隐式问题。提出HinD框架，通过Hindsight Distillation生成显式推理步骤和知识，并用KEPO优化知识正确性。实验在OK-VQA和A-OKVQA上验证，无需外部API或知识。**

- **链接: [https://arxiv.org/pdf/2511.11132v1](https://arxiv.org/pdf/2511.11132v1)**

> **作者:** Yu Zhao; Ying Zhang; Xuhui Sui; Baohang Zhou; Li Shen; Dacheng Tao
>
> **摘要:** Knowledge-based Visual Question Answering (KBVQA) necessitates external knowledge incorporation beyond cross-modal understanding. Existing KBVQA methods either utilize implicit knowledge in multimodal large language models (MLLMs) via in-context learning or explicit knowledge via retrieval augmented generation. However, their reasoning processes remain implicit, without explicit multi-step trajectories from MLLMs. To address this gap, we provide a Hindsight Distilled Reasoning (HinD) framework with Knowledge Encouragement Preference Optimization (KEPO), designed to elicit and harness internal knowledge reasoning ability in MLLMs. First, to tackle the reasoning supervision problem, we propose to emphasize the hindsight wisdom of MLLM by prompting a frozen 7B-size MLLM to complete the reasoning process between the question and its ground truth answer, constructing Hindsight-Zero training data. Then we self-distill Hindsight-Zero into Chain-of-Thought (CoT) Generator and Knowledge Generator, enabling the generation of sequential steps and discrete facts. Secondly, to tackle the misalignment between knowledge correctness and confidence, we optimize the Knowledge Generator with KEPO, preferring under-confident but helpful knowledge over the over-confident but unhelpful one. The generated CoT and sampled knowledge are then exploited for answer prediction. Experiments on OK-VQA and A-OKVQA validate the effectiveness of HinD, showing that HinD with elicited reasoning from 7B-size MLLM achieves superior performance without commercial model APIs or outside knowledge.
>
---
#### [new 070] Arcee: Differentiable Recurrent State Chain for Generative Vision Modeling with Mamba SSMs
- **分类: cs.CV**

- **简介: 论文提出Arcee用于生成式视觉建模，解决Mamba在视觉任务中丢弃跨块状态的问题。通过重用终端状态空间表示作为初始条件，构建可微分边界图实现端到端训练，显著提升生成质量（CelebA-HQ上FID降低5.4倍）。**

- **链接: [https://arxiv.org/pdf/2511.11243v1](https://arxiv.org/pdf/2511.11243v1)**

> **作者:** Jitesh Chavan; Rohit Lal; Anand Kamat; Mengjia Xu
>
> **摘要:** State-space models (SSMs), Mamba in particular, are increasingly adopted for long-context sequence modeling, providing linear-time aggregation via an input-dependent, causal selective-scan operation. Along this line, recent "Mamba-for-vision" variants largely explore multiple scan orders to relax strict causality for non-sequential signals (e.g., images). Rather than preserving cross-block memory, the conventional formulation of the selective-scan operation in Mamba reinitializes each block's state-space dynamics from zero, discarding the terminal state-space representation (SSR) from the previous block. Arcee, a cross-block recurrent state chain, reuses each block's terminal state-space representation as the initial condition for the next block. Handoff across blocks is constructed as a differentiable boundary map whose Jacobian enables end-to-end gradient flow across terminal boundaries. Key to practicality, Arcee is compatible with all prior "vision-mamba" variants, parameter-free, and incurs constant, negligible cost. As a modeling perspective, we view terminal SSR as a mild directional prior induced by a causal pass over the input, rather than an estimator of the non-sequential signal itself. To quantify the impact, for unconditional generation on CelebA-HQ (256$\times$256) with Flow Matching, Arcee reduces FID$\downarrow$ from $82.81$ to $15.33$ ($5.4\times$ lower) on a single scan-order Zigzag Mamba baseline. Efficient CUDA kernels and training code will be released to support rigorous and reproducible research.
>
---
#### [new 071] Accelerating Controllable Generation via Hybrid-grained Cache
- **分类: cs.CV; cs.MM**

- **简介: 论文针对可控视觉生成效率低的问题，提出混合粒度缓存（HGC）方法。通过块级粗粒度缓存和提示级细粒度缓存策略，动态跳过冗余计算，显著降低计算成本（如COCO-Stuff上MACs减少63%），同时保持语义质量。**

- **链接: [https://arxiv.org/pdf/2511.11031v1](https://arxiv.org/pdf/2511.11031v1)**

> **作者:** Lin Liu; Huixia Ben; Shuo Wang; Jinda Lu; Junxiang Qiu; Shengeng Tang; Yanbin Hao
>
> **摘要:** Controllable generative models have been widely used to improve the realism of synthetic visual content. However, such models must handle control conditions and content generation computational requirements, resulting in generally low generation efficiency. To address this issue, we propose a Hybrid-Grained Cache (HGC) approach that reduces computational overhead by adopting cache strategies with different granularities at different computational stages. Specifically, (1) we use a coarse-grained cache (block-level) based on feature reuse to dynamically bypass redundant computations in encoder-decoder blocks between each step of model reasoning. (2) We design a fine-grained cache (prompt-level) that acts within a module, where the fine-grained cache reuses cross-attention maps within consecutive reasoning steps and extends them to the corresponding module computations of adjacent steps. These caches of different granularities can be seamlessly integrated into each computational link of the controllable generation process. We verify the effectiveness of HGC on four benchmark datasets, especially its advantages in balancing generation efficiency and visual quality. For example, on the COCO-Stuff segmentation benchmark, our HGC significantly reduces the computational cost (MACs) by 63% (from 18.22T to 6.70T), while keeping the loss of semantic fidelity (quantized performance degradation) within 1.5%.
>
---
#### [new 072] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大语言模型的视觉概念移除问题，提出AUVIC框架。通过对抗性扰动实现目标视觉概念的精确遗忘，避免影响相关实体。构建VCUBench基准，实验表明高效移除目标概念且性能损失小。**

- **链接: [https://arxiv.org/pdf/2511.11299v1](https://arxiv.org/pdf/2511.11299v1)**

> **作者:** Haokun Chen; Jianing Li; Yao Zhang; Jinhe Bi; Yan Xia; Jindong Gu; Volker Tresp
>
> **备注:** AAAI 2026. Code: https://github.com/HaokunChen245/AUVIC
>
> **摘要:** Multimodal Large Language Models (MLLMs) achieve impressive performance once optimized on massive datasets. Such datasets often contain sensitive or copyrighted content, raising significant data privacy concerns. Regulatory frameworks mandating the 'right to be forgotten' drive the need for machine unlearning. This technique allows for the removal of target data without resource-consuming retraining. However, while well-studied for text, visual concept unlearning in MLLMs remains underexplored. A primary challenge is precisely removing a target visual concept without disrupting model performance on related entities. To address this, we introduce AUVIC, a novel visual concept unlearning framework for MLLMs. AUVIC applies adversarial perturbations to enable precise forgetting. This approach effectively isolates the target concept while avoiding unintended effects on similar entities. To evaluate our method, we construct VCUBench. It is the first benchmark designed to assess visual concept unlearning in group contexts. Experimental results demonstrate that AUVIC achieves state-of-the-art target forgetting rates while incurs minimal performance degradation on non-target concepts.
>
---
#### [new 073] Φeat: Physically-Grounded Feature Representation
- **分类: cs.CV**

- **简介: 论文提出Φeat，解决自监督特征纠缠物理因素（如几何、光照）的问题。通过自监督对比学习，利用材料在不同形状光照下的图像，构建物理基础特征表示，提升材料识别等任务性能。**

- **链接: [https://arxiv.org/pdf/2511.11270v1](https://arxiv.org/pdf/2511.11270v1)**

> **作者:** Giuseppe Vecchio; Adrien Kaiser; Rouffet Romain; Rosalie Martin; Elena Garces; Tamy Boubekeur
>
> **摘要:** Foundation models have emerged as effective backbones for many vision tasks. However, current self-supervised features entangle high-level semantics with low-level physical factors, such as geometry and illumination, hindering their use in tasks requiring explicit physical reasoning. In this paper, we introduce $Φ$eat, a novel physically-grounded visual backbone that encourages a representation sensitive to material identity, including reflectance cues and geometric mesostructure. Our key idea is to employ a pretraining strategy that contrasts spatial crops and physical augmentations of the same material under varying shapes and lighting conditions. While similar data have been used in high-end supervised tasks such as intrinsic decomposition or material estimation, we demonstrate that a pure self-supervised training strategy, without explicit labels, already provides a strong prior for tasks requiring robust features invariant to external physical factors. We evaluate the learned representations through feature similarity analysis and material selection, showing that $Φ$eat captures physically-grounded structure beyond semantic grouping. These findings highlight the promise of unsupervised physical feature learning as a foundation for physics-aware perception in vision and graphics. These findings highlight the promise of unsupervised physical feature learning as a foundation for physics-aware perception in vision and graphics.
>
---
#### [new 074] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出VP-Bench基准，解决MLLMs对视觉提示（如边界框）理解能力的系统评估问题。通过两阶段框架：Stage 1测试30k VP感知，Stage 2评估下游任务效果，评估28个模型并分析属性、问题排列及模型规模影响。**

- **链接: [https://arxiv.org/pdf/2511.11438v1](https://arxiv.org/pdf/2511.11438v1)**

> **作者:** Mingjie Xu; Jinpeng Chen; Yuzhi Zhao; Jason Chun Lok Li; Yue Qiu; Zekang Du; Mengyang Wu; Pingping Zhang; Kun Li; Hongzheng Yang; Wenao Ma; Jiaheng Wei; Qinbin Li; Kangcheng Liu; Wenqiang Lei
>
> **备注:** This is the extended version of the paper accepted at AAAI 2026, which includes all technical appendices and additional experimental details
>
> **摘要:** Multimodal large language models (MLLMs) have enabled a wide range of advanced vision-language applications, including fine-grained object recognition and contextual understanding. When querying specific regions or objects in an image, human users naturally use "visual prompts" (VPs), such as bounding boxes, to provide reference. However, no existing benchmark systematically evaluates the ability of MLLMs to interpret such VPs. This gap leaves it unclear whether current MLLMs can effectively recognize VPs, an intuitive prompting method for humans, and use them to solve problems. To address this limitation, we introduce VP-Bench, a benchmark for assessing MLLMs' capability in VP perception and utilization. VP-Bench employs a two-stage evaluation framework: Stage 1 examines models' ability to perceive VPs in natural scenes, using 30k visualized prompts spanning eight shapes and 355 attribute combinations. Stage 2 investigates the impact of VPs on downstream tasks, measuring their effectiveness in real-world problem-solving scenarios. Using VP-Bench, we evaluate 28 MLLMs, including proprietary systems (e.g., GPT-4o) and open-source models (e.g., InternVL3 and Qwen2.5-VL), and provide a comprehensive analysis of factors that affect VP understanding, such as variations in VP attributes, question arrangement, and model scale. VP-Bench establishes a new reference framework for studying how MLLMs comprehend and resolve grounded referring questions.
>
---
#### [new 075] AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AirCopBench，首个多无人机协同感知与推理基准，解决MLLMs在复杂降级感知条件下协同任务评估缺失问题。包含14.6k+问题，覆盖四个任务维度，评估显示最佳模型落后人类24.38%，证实模拟到真实转移可行性。**

- **链接: [https://arxiv.org/pdf/2511.11025v1](https://arxiv.org/pdf/2511.11025v1)**

> **作者:** Jirong Zha; Yuxuan Fan; Tianyu Zhang; Geng Chen; Yingfeng Chen; Chen Gao; Xinlei Chen
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown promise in single-agent vision tasks, yet benchmarks for evaluating multi-agent collaborative perception remain scarce. This gap is critical, as multi-drone systems provide enhanced coverage, robustness, and collaboration compared to single-sensor setups. Existing multi-image benchmarks mainly target basic perception tasks using high-quality single-agent images, thus failing to evaluate MLLMs in more complex, egocentric collaborative scenarios, especially under real-world degraded perception conditions.To address these challenges, we introduce AirCopBench, the first comprehensive benchmark designed to evaluate MLLMs in embodied aerial collaborative perception under challenging perceptual conditions. AirCopBench includes 14.6k+ questions derived from both simulator and real-world data, spanning four key task dimensions: Scene Understanding, Object Understanding, Perception Assessment, and Collaborative Decision, across 14 task types. We construct the benchmark using data from challenging degraded-perception scenarios with annotated collaborative events, generating large-scale questions through model-, rule-, and human-based methods under rigorous quality control. Evaluations on 40 MLLMs show significant performance gaps in collaborative perception tasks, with the best model trailing humans by 24.38% on average and exhibiting inconsistent results across tasks. Fine-tuning experiments further confirm the feasibility of sim-to-real transfer in aerial collaborative perception and reasoning.
>
---
#### [new 076] From Synthetic Scenes to Real Performance: Enhancing Spatial Reasoning in VLMs
- **分类: cs.CV; cs.CL**

- **简介: 论文针对VLMs空间推理任务中的数据偏差问题，提出通过自动构建平衡合成数据集（采样对象属性）并微调模型。实验表明，该方法显著提升真实世界性能（如COCO），优于传统真实数据微调。**

- **链接: [https://arxiv.org/pdf/2511.11440v1](https://arxiv.org/pdf/2511.11440v1)**

> **作者:** Massimo Rizzoli; Simone Alghisi; Seyed Mahed Mousavi; Giuseppe Riccardi
>
> **摘要:** Fine-tuning Vision-Language Models (VLMs) is a common strategy to improve performance following an ad-hoc data collection and annotation of real-world scenes. However, this process is often prone to biases, errors, and distribution imbalance, resulting in overfitting and imbalanced performance. Although a few studies have tried to address this problem by generating synthetic data, they lacked control over distribution bias and annotation quality. To address these challenges, we redesign the fine-tuning process in two ways. First, we control the generation of data and its annotations, ensuring it is free from bias, distribution imbalance, and annotation errors. We automatically construct the dataset by comprehensively sampling objects' attributes, including color, shape, size, and position within the scene. Secondly, using this annotated dataset, we fine-tune state-of-the-art VLMs and assess performance transferability to real-world data on the absolute position task. We conduct exhaustive evaluations on both synthetic and real-world benchmarks. Our experiments reveal two key findings: 1) fine-tuning on balanced synthetic data yields uniform performance across the visual scene and mitigates common biases; and 2) fine-tuning on synthetic stimuli significantly improves performance on real-world data (COCO), outperforming models fine-tuned in the matched setting.
>
---
#### [new 077] CVChess: A Deep Learning Framework for Converting Chessboard Images to Forsyth-Edwards Notation
- **分类: cs.CV; cs.LG**

- **简介: CVChess提出深度学习框架，解决物理棋盘图像转FEN问题。采用CNN残差网络进行13类棋子识别（6白、6黑、空），经预处理、分割和分类，基于ChessReD数据集训练，使物理棋局接入在线引擎获取最优走法。**

- **链接: [https://arxiv.org/pdf/2511.11522v1](https://arxiv.org/pdf/2511.11522v1)**

> **作者:** Luthira Abeykoon; Ved Patel; Gawthaman Senthilvelan; Darshan Kasundra
>
> **摘要:** Chess has experienced a large increase in viewership since the pandemic, driven largely by the accessibility of online learning platforms. However, no equivalent assistance exists for physical chess games, creating a divide between analog and digital chess experiences. This paper presents CVChess, a deep learning framework for converting chessboard images to Forsyth-Edwards Notation (FEN), which is later input into online chess engines to provide you with the best next move. Our approach employs a convolutional neural network (CNN) with residual layers to perform piece recognition from smartphone camera images. The system processes RGB images of a physical chess board through a multistep process: image preprocessing using the Hough Line Transform for edge detection, projective transform to achieve a top-down board alignment, segmentation into 64 individual squares, and piece classification into 13 classes (6 unique white pieces, 6 unique black pieces and an empty square) using the residual CNN. Residual connections help retain low-level visual features while enabling deeper feature extraction, improving accuracy and stability during training. We train and evaluate our model using the Chess Recognition Dataset (ChessReD), containing 10,800 annotated smartphone images captured under diverse lighting conditions and angles. The resulting classifications are encoded as an FEN string, which can be fed into a chess engine to generate the most optimal move
>
---
#### [new 078] Accuracy-Preserving CNN Pruning Method under Limited Data Availability
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出基于LRP的CNN剪枝方法，在数据有限场景下实现高剪枝率同时保持准确率，解决现有方法因准确率下降导致的实用性问题。**

- **链接: [https://arxiv.org/pdf/2511.10861v1](https://arxiv.org/pdf/2511.10861v1)**

> **作者:** Daisuke Yasui; Toshitaka Matsuki; Hiroshi Sato
>
> **摘要:** Convolutional Neural Networks (CNNs) are widely used in image recognition and have succeeded in various domains. CNN models have become larger-scale to improve accuracy and generalization performance. Research has been conducted on compressing pre-trained models for specific target applications in environments with limited computing resources. Among model compression techniques, methods using Layer-wise Relevance Propagation (LRP), an explainable AI technique, have shown promise by achieving high pruning rates while preserving accuracy, even without fine-tuning. Because these methods do not require fine-tuning, they are suited to scenarios with limited data. However, existing LRP-based pruning approaches still suffer from significant accuracy degradation, limiting their practical usability. This study proposes a pruning method that achieves a higher pruning rate while preserving better model accuracy. Our approach to pruning with a small amount of data has achieved pruning that preserves accuracy better than existing methods.
>
---
#### [new 079] CrossMed: A Multimodal Cross-Task Benchmark for Compositional Generalization in Medical Imaging
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CrossMed基准，评估医疗多模态LLM的组合泛化能力（模态-解剖-任务）。统一四个数据集为20,200个VQA实例，测试模型在未见组合下的性能，显示LLM在zero-overlap条件下准确率骤降，验证了基准挑战性并证明LLM在组合泛化上的优势。**

- **链接: [https://arxiv.org/pdf/2511.11034v1](https://arxiv.org/pdf/2511.11034v1)**

> **作者:** Pooja Singh; Siddhant Ujjain; Tapan Kumar Gandhi; Sandeep Kumar
>
> **摘要:** Recent advances in multimodal large language models have enabled unified processing of visual and textual inputs, offering promising applications in general-purpose medical AI. However, their ability to generalize compositionally across unseen combinations of imaging modality, anatomy, and task type remains underexplored. We introduce CrossMed, a benchmark designed to evaluate compositional generalization (CG) in medical multimodal LLMs using a structured Modality-Anatomy-Task (MAT) schema. CrossMed reformulates four public datasets, CheXpert (X-ray classification), SIIM-ACR (X-ray segmentation), BraTS 2020 (MRI classification and segmentation), and MosMedData (CT classification) into a unified visual question answering (VQA) format, resulting in 20,200 multiple-choice QA instances. We evaluate two open-source multimodal LLMs, LLaVA-Vicuna-7B and Qwen2-VL-7B, on both Related and Unrelated MAT splits, as well as a zero-overlap setting where test triplets share no Modality, Anatomy, or Task with the training data. Models trained on Related splits achieve 83.2 percent classification accuracy and 0.75 segmentation cIoU, while performance drops significantly under Unrelated and zero-overlap conditions, demonstrating the benchmark difficulty. We also show cross-task transfer, where segmentation performance improves by 7 percent cIoU even when trained using classification-only data. Traditional models (ResNet-50 and U-Net) show modest gains, confirming the broad utility of the MAT framework, while multimodal LLMs uniquely excel at compositional generalization. CrossMed provides a rigorous testbed for evaluating zero-shot, cross-task, and modality-agnostic generalization in medical vision-language models.
>
---
#### [new 080] Evaluating Latent Generative Paradigms for High-Fidelity 3D Shape Completion from a Single Depth Image
- **分类: cs.CV**

- **简介: 论文针对3D形状补全任务，评估扩散模型与自回归Transformer在单个深度图像下的性能。结果表明，扩散模型在连续潜空间表现最优，自回归模型在相同离散潜空间可匹敌或超越扩散模型。**

- **链接: [https://arxiv.org/pdf/2511.11074v1](https://arxiv.org/pdf/2511.11074v1)**

> **作者:** Matthias Humt; Ulrich Hillenbrand; Rudolph Triebel
>
> **备注:** 17 pages, 4 figures, 19 tables
>
> **摘要:** While generative models have seen significant adoption across a wide range of data modalities, including 3D data, a consensus on which model is best suited for which task has yet to be reached. Further, conditional information such as text and images to steer the generation process are frequently employed, whereas others, like partial 3D data, have not been thoroughly evaluated. In this work, we compare two of the most promising generative models--Denoising Diffusion Probabilistic Models and Autoregressive Causal Transformers--which we adapt for the tasks of generative shape modeling and completion. We conduct a thorough quantitative evaluation and comparison of both tasks, including a baseline discriminative model and an extensive ablation study. Our results show that (1) the diffusion model with continuous latents outperforms both the discriminative model and the autoregressive approach and delivers state-of-the-art performance on multi-modal shape completion from a single, noisy depth image under realistic conditions and (2) when compared on the same discrete latent space, the autoregressive model can match or exceed diffusion performance on these tasks.
>
---
#### [new 081] Shrinking the Teacher: An Adaptive Teaching Paradigm for Asymmetric EEG-Vision Alignment
- **分类: cs.CV**

- **简介: 论文解决EEG-视觉跨模态对齐中的不对称问题（Fidelity Gap和Semantic Gap），提出自适应教学范式，使视觉模态动态缩小知识结构匹配EEG。通过ShrinkAdapter模块实现，零样本脑到图像检索准确率达60.2%，超越SOTA 9.8%。**

- **链接: [https://arxiv.org/pdf/2511.11422v1](https://arxiv.org/pdf/2511.11422v1)**

> **作者:** Lukun Wu; Jie Li; Ziqi Ren; Kaifan Zhang; Xinbo Gao
>
> **备注:** 21pages,12 figures,published to AAAI 2026
>
> **摘要:** Decoding visual features from EEG signals is a central challenge in neuroscience, with cross-modal alignment as the dominant approach. We argue that the relationship between visual and brain modalities is fundamentally asymmetric, characterized by two critical gaps: a Fidelity Gap (stemming from EEG's inherent noise and signal degradation, vs. vision's high-fidelity features) and a Semantic Gap (arising from EEG's shallow conceptual representation, vs. vision's rich semantic depth). Previous methods often overlook this asymmetry, forcing alignment between the two modalities as if they were equal partners and thereby leading to poor generalization. To address this, we propose the adaptive teaching paradigm. This paradigm empowers the ``teacher" modality (vision) to dynamically shrink and adjust its knowledge structure under task guidance, tailoring its semantically dense features to match the ``student" modality (EEG)'s capacity. We implement this paradigm with the ShrinkAdapter, a simple yet effective module featuring a residual-free design and a bottleneck structure. Through extensive experiments, we validate the underlying rationale and effectiveness of our paradigm. Our method achieves a top-1 accuracy of 60.2\% on the zero-shot brain-to-image retrieval task, surpassing previous state-of-the-art methods by a margin of 9.8\%. Our work introduces a new perspective for asymmetric alignment: the teacher must shrink and adapt to bridge the vision-brain gap.
>
---
#### [new 082] Detection of Bark Beetle Attacks using Hyperspectral PRISMA Data and Few-Shot Learning
- **分类: cs.CV**

- **简介: 论文属于森林健康监测任务，解决松树皮甲虫侵袭检测问题。采用PRISMA高光谱数据，提出基于对比学习的少样本学习方法：预训练1D CNN提取特征，结合SVM回归估计每像素健康、受攻击和死亡树木比例。实验表明优于PRISMA原始波段和Sentinel-2数据。**

- **链接: [https://arxiv.org/pdf/2511.11096v1](https://arxiv.org/pdf/2511.11096v1)**

> **作者:** Mattia Ferrari; Giancarlo Papitto; Giorgio Deligios; Lorenzo Bruzzone
>
> **备注:** 5 pages, 3 figures, accepted at IGARSS conference 3-8 August 2025 Brisbane, Australia
>
> **摘要:** Bark beetle infestations represent a serious challenge for maintaining the health of coniferous forests. This paper proposes a few-shot learning approach leveraging contrastive learning to detect bark beetle infestations using satellite PRISMA hyperspectral data. The methodology is based on a contrastive learning framework to pre-train a one-dimensional CNN encoder, enabling the extraction of robust feature representations from hyperspectral data. These extracted features are subsequently utilized as input to support vector regression estimators, one for each class, trained on few labeled samples to estimate the proportions of healthy, attacked by bark beetle, and dead trees for each pixel. Experiments on the area of study in the Dolomites show that our method outperforms the use of original PRISMA spectral bands and of Sentinel-2 data. The results indicate that PRISMA hyperspectral data combined with few-shot learning offers significant advantages for forest health monitoring.
>
---
#### [new 083] CareCom: Generative Image Composition with Calibrated Reference Features
- **分类: cs.CV**

- **简介: 该论文解决图像合成任务中细节保留与前景姿势调整难题。提出多参考生成模型，通过校准前景参考图像的全局和局部特征，使其与背景兼容，提升合成质量。**

- **链接: [https://arxiv.org/pdf/2511.11060v1](https://arxiv.org/pdf/2511.11060v1)**

> **作者:** Jiaxuan Chen; Bo Zhang; Qingdong He; Jinlong Peng; Li Niu
>
> **摘要:** Image composition aims to seamlessly insert foreground object into background. Despite the huge progress in generative image composition, the existing methods are still struggling with simultaneous detail preservation and foreground pose/view adjustment. To address this issue, we extend the existing generative composition model to multi-reference version, which allows using arbitrary number of foreground reference images. Furthermore, we propose to calibrate the global and local features of foreground reference images to make them compatible with the background information. The calibrated reference features can supplement the original reference features with useful global and local information of proper pose/view. Extensive experiments on MVImgNet and MureCom demonstrate that the generative model can greatly benefit from the calibrated reference features.
>
---
#### [new 084] Reverberation: Learning the Latencies Before Forecasting Trajectories
- **分类: cs.CV**

- **简介: 论文聚焦轨迹预测任务，解决现有方法忽略代理响应延迟的问题。提出Reverberation (Rev)模型，利用两个可学习核模拟延迟偏好，实现可控轨迹预测，实验验证于行人和车辆数据集。**

- **链接: [https://arxiv.org/pdf/2511.11164v1](https://arxiv.org/pdf/2511.11164v1)**

> **作者:** Conghao Wong; Ziqian Zou; Beihao Xia; Xinge You
>
> **摘要:** Bridging the past to the future, connecting agents both spatially and temporally, lies at the core of the trajectory prediction task. Despite great efforts, it remains challenging to explicitly learn and predict latencies, the temporal delays with which agents respond to different trajectory-changing events and adjust their future paths, whether on their own or interactively. Different agents may exhibit distinct latency preferences for noticing, processing, and reacting to any specific trajectory-changing event. The lack of consideration of such latencies may undermine the causal continuity of the forecasting system and also lead to implausible or unintended trajectories. Inspired by the reverberation curves in acoustics, we propose a new reverberation transform and the corresponding Reverberation (short for Rev) trajectory prediction model, which simulates and predicts different latency preferences of each agent as well as their stochasticity by using two explicit and learnable reverberation kernels, allowing for the controllable trajectory prediction based on these forecasted latencies. Experiments on multiple datasets, whether pedestrians or vehicles, demonstrate that Rev achieves competitive accuracy while revealing interpretable latency dynamics across agents and scenarios. Qualitative analyses further verify the properties of the proposed reverberation transform, highlighting its potential as a general latency modeling approach.
>
---
#### [new 085] Heterogeneous Complementary Distillation
- **分类: cs.CV**

- **简介: 该论文解决异构架构知识蒸馏问题（如ViT到ResNet18）。传统方法因特征差异效果差，且计算成本高。提出Heterogeneous Complementary Distillation (HCD)，融合互补特征，通过CFM模块和子logit解耦蒸馏优化对齐，结合正交性损失提升多样性，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10942v1](https://arxiv.org/pdf/2511.10942v1)**

> **作者:** Liuchi Xu; Hao Zheng; Lu Wang; Lisheng Xu; Jun Cheng
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Knowledge distillation (KD)transfers the dark knowledge from a complex teacher to a compact student. However, heterogeneous architecture distillation, such as Vision Transformer (ViT) to ResNet18, faces challenges due to differences in spatial feature representations.Traditional KD methods are mostly designed for homogeneous architectures and hence struggle to effectively address the disparity. Although heterogeneous KD approaches have been developed recently to solve these issues, they often incur high computational costs and complex designs, or overly rely on logit alignment, which limits their ability to leverage the complementary features. To overcome these limitations, we propose Heterogeneous Complementary Distillation (HCD),a simple yet effective framework that integrates complementary teacher and student features to align representations in shared logits.These logits are decomposed and constrained to facilitate diverse knowledge transfer to the student. Specifically, HCD processes the student's intermediate features through convolutional projector and adaptive pooling, concatenates them with teacher's feature from the penultimate layer and then maps them via the Complementary Feature Mapper (CFM) module, comprising fully connected layer,to produce shared logits.We further introduce Sub-logit Decoupled Distillation (SDD) that partitions the shared logits into n sub-logits, which are fused with teacher's logits to rectify classification.To ensure sub-logit diversity and reduce redundant knowledge transfer, we propose an Orthogonality Loss (OL).By preserving student-specific strengths and leveraging teacher knowledge,HCD enhances robustness and generalization in students.Extensive experiments on the CIFAR-100, Fine-grained (e.g., CUB200)and ImageNet-1K datasets demonstrate that HCD outperforms state-of-the-art KD methods,establishing it as an effective solution for heterogeneous KD.
>
---
#### [new 086] Stroke Modeling Enables Vectorized Character Generation with Large Vectorized Glyph Model
- **分类: cs.CV**

- **简介: 该论文提出Large Vectorized Glyph Model (LVGM)，解决中文矢量字符高效生成问题。通过笔画建模，将笔画编码为嵌入，微调LLM预测下一笔画，实现从有限笔画生成完整字符、词语。发布90万+中文SVG数据集，实验验证模型有效性。**

- **链接: [https://arxiv.org/pdf/2511.11119v1](https://arxiv.org/pdf/2511.11119v1)**

> **作者:** Xinyue Zhang; Haolong Li; Jiawei Ma; Chen Ye
>
> **摘要:** Vectorized glyphs are widely used in poster design, network animation, art display, and various other fields due to their scalability and flexibility. In typography, they are often seen as special sequences composed of ordered strokes. This concept extends to the token sequence prediction abilities of large language models (LLMs), enabling vectorized character generation through stroke modeling. In this paper, we propose a novel Large Vectorized Glyph Model (LVGM) designed to generate vectorized Chinese glyphs by predicting the next stroke. Initially, we encode strokes into discrete latent variables called stroke embeddings. Subsequently, we train our LVGM via fine-tuning DeepSeek LLM by predicting the next stroke embedding. With limited strokes given, it can generate complete characters, semantically elegant words, and even unseen verses in vectorized form. Moreover, we release a new large-scale Chinese SVG dataset containing 907,267 samples based on strokes for dynamically vectorized glyph generation. Experimental results show that our model has scaling behaviors on data scales. Our generated vectorized glyphs have been validated by experts and relevant individuals.
>
---
#### [new 087] MCN-CL: Multimodal Cross-Attention Network and Contrastive Learning for Multimodal Emotion Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦多模态情感识别任务，解决类别不平衡、动态面部动作单元建模复杂性和模态异质性问题。提出MCN-CL模型，利用三重查询机制和硬负样本挖掘策略，显著提升性能，加权F1分数在IEMOCAP和MELD数据集上分别提高3.42%和5.73%。**

- **链接: [https://arxiv.org/pdf/2511.10892v1](https://arxiv.org/pdf/2511.10892v1)**

> **作者:** Feng Li; Ke Wu; Yongwei Li
>
> **备注:** Accepted by 32nd International Conference on MultiMedia Modeling (MMM 2026)
>
> **摘要:** Multimodal emotion recognition plays a key role in many domains, including mental health monitoring, educational interaction, and human-computer interaction. However, existing methods often face three major challenges: unbalanced category distribution, the complexity of dynamic facial action unit time modeling, and the difficulty of feature fusion due to modal heterogeneity. With the explosive growth of multimodal data in social media scenarios, the need for building an efficient cross-modal fusion framework for emotion recognition is becoming increasingly urgent. To this end, this paper proposes Multimodal Cross-Attention Network and Contrastive Learning (MCN-CL) for multimodal emotion recognition. It uses a triple query mechanism and hard negative mining strategy to remove feature redundancy while preserving important emotional cues, effectively addressing the issues of modal heterogeneity and category imbalance. Experiment results on the IEMOCAP and MELD datasets show that our proposed method outperforms state-of-the-art approaches, with Weighted F1 scores improving by 3.42% and 5.73%, respectively.
>
---
#### [new 088] GFT: Graph Feature Tuning for Efficient Point Cloud Analysis
- **分类: cs.CV**

- **简介: 该论文针对点云分析任务，解决参数高效微调（PEFT）中可训练参数过多问题。提出Graph Feature Tuning (GFT)，通过轻量图卷积网络学习动态图特征，经跳过连接和交叉注意力传递至深层，显著减少参数量，同时保持物体分类与分割性能。**

- **链接: [https://arxiv.org/pdf/2511.10799v1](https://arxiv.org/pdf/2511.10799v1)**

> **作者:** Manish Dhakal; Venkat R. Dasari; Raj Sunderraman; Yi Ding
>
> **备注:** WACV 2026
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) significantly reduces computational and memory costs by updating only a small subset of the model's parameters, enabling faster adaptation to new tasks with minimal loss in performance. Previous studies have introduced PEFTs tailored for point cloud data, as general approaches are suboptimal. To further reduce the number of trainable parameters, we propose a point-cloud-specific PEFT, termed Graph Features Tuning (GFT), which learns a dynamic graph from initial tokenized inputs of the transformer using a lightweight graph convolution network and passes these graph features to deeper layers via skip connections and efficient cross-attention modules. Extensive experiments on object classification and segmentation tasks show that GFT operates in the same domain, rivalling existing methods, while reducing the trainable parameters. Code is at https://github.com/manishdhakal/GFT.
>
---
#### [new 089] Machine-Learning Based Detection of Coronary Artery Calcification Using Synthetic Chest X-Rays
- **分类: cs.CV**

- **简介: 论文解决冠状动脉钙化（CAC）检测问题，因CT成本高、CXRs缺乏可靠标签。通过数字化重建射线（DRRs）生成合成CXRs作为训练数据，优化轻量级CNN模型与超分辨率策略，实现AUC 0.754，优于现有CXR研究。**

- **链接: [https://arxiv.org/pdf/2511.11093v1](https://arxiv.org/pdf/2511.11093v1)**

> **作者:** Dylan Saeed; Ramtin Gharleghi; Susann Bier; Sonit Singh
>
> **备注:** 10 pages, 5 figures. Under review for MIDL 2026
>
> **摘要:** Coronary artery calcification (CAC) is a strong predictor of cardiovascular events, with CT-based Agatston scoring widely regarded as the clinical gold standard. However, CT is costly and impractical for large-scale screening, while chest X-rays (CXRs) are inexpensive but lack reliable ground truth labels, constraining deep learning development. Digitally reconstructed radiographs (DRRs) offer a scalable alternative by projecting CT volumes into CXR-like images while inheriting precise labels. In this work, we provide the first systematic evaluation of DRRs as a surrogate training domain for CAC detection. Using 667 CT scans from the COCA dataset, we generate synthetic DRRs and assess model capacity, super-resolution fidelity enhancement, preprocessing, and training strategies. Lightweight CNNs trained from scratch outperform large pretrained networks; pairing super-resolution with contrast enhancement yields significant gains; and curriculum learning stabilises training under weak supervision. Our best configuration achieves a mean AUC of 0.754, comparable to or exceeding prior CXR-based studies. These results establish DRRs as a scalable, label-rich foundation for CAC detection, while laying the foundation for future transfer learning and domain adaptation to real CXRs.
>
---
#### [new 090] Q-Doc: Benchmarking Document Image Quality Assessment Capabilities in Multi-modal Large Language Models
- **分类: cs.CV**

- **简介: 论文提出Q-Doc基准，系统评估多模态大语言模型在文档图像质量评估（DIQA）中的能力。通过粗、中、细粒度三层次测试，发现模型存在评分不一致、失真误判等问题，但链式思维提示显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.11410v1](https://arxiv.org/pdf/2511.11410v1)**

> **作者:** Jiaxi Huang; Dongxu Wu; Hanwei Zhu; Lingyu Zhu; Jun Xing; Xu Wang; Baoliang Chen
>
> **摘要:** The rapid advancement of Multi-modal Large Language Models (MLLMs) has expanded their capabilities beyond high-level vision tasks. Nevertheless, their potential for Document Image Quality Assessment (DIQA) remains underexplored. To bridge this gap, we propose Q-Doc, a three-tiered evaluation framework for systematically probing DIQA capabilities of MLLMs at coarse, middle, and fine granularity levels. a) At the coarse level, we instruct MLLMs to assign quality scores to document images and analyze their correlation with Quality Annotations. b) At the middle level, we design distortion-type identification tasks, including single-choice and multi-choice tests for multi-distortion scenarios. c) At the fine level, we introduce distortion-severity assessment where MLLMs classify distortion intensity against human-annotated references. Our evaluation demonstrates that while MLLMs possess nascent DIQA abilities, they exhibit critical limitations: inconsistent scoring, distortion misidentification, and severity misjudgment. Significantly, we show that Chain-of-Thought (CoT) prompting substantially enhances performance across all levels. Our work provides a benchmark for DIQA capabilities in MLLMs, revealing pronounced deficiencies in their quality perception and promising pathways for enhancement. The benchmark and code are publicly available at: https://github.com/cydxf/Q-Doc.
>
---
#### [new 091] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?
- **分类: cs.CV**

- **简介: 论文探究多模态嵌入模型中的位置偏差问题，聚焦图像-文本检索任务。发现文本编码器偏好输入开头，图像编码器偏好开头和结尾；偏差源于位置编码、训练损失等因素。**

- **链接: [https://arxiv.org/pdf/2511.11216v1](https://arxiv.org/pdf/2511.11216v1)**

> **作者:** Kebin Wu; Fatima Albreiki
>
> **备注:** accepted to AAAI 2026 main track
>
> **摘要:** Positional bias - where models overemphasize certain positions regardless of content - has been shown to negatively impact model performance across various tasks. While recent research has extensively examined positional bias in text generation models, its presence and effects in representation models remain underexplored. Even less is known about such biases in multimodal models. In this work, we investigate positional bias in multimodal representation models, specifically in the context of image-text retrieval. We begin by distinguishing between context importance and positional bias, and then assess the presence and extent of positional bias across different models and datasets. Our experiments demonstrate that positional bias is prevalent in multimodal models, but manifests differently across modalities: text encoders tend to exhibit bias toward the beginning of the input, whereas image encoders show bias at both the beginning and end. Furthermore, we find that this bias arises from, or is amplified by, a combination of factors, including the positional encoding scheme, training loss, context importance, and the nature of using image-text pairs in multimodal training.
>
---
#### [new 092] 6D Strawberry Pose Estimation: Real-time and Edge AI Solutions Using Purely Synthetic Training Data
- **分类: cs.CV; cs.RO**

- **简介: 该论文解决草莓6D姿态估计问题，利用纯合成数据训练YOLOX-6D-Pose模型，实现边缘设备实时推理。在RTX 3090和Jetson Orin Nano上验证，精度高，但未成熟草莓检测需优化。**

- **链接: [https://arxiv.org/pdf/2511.11307v1](https://arxiv.org/pdf/2511.11307v1)**

> **作者:** Saptarshi Neil Sinha; Julius Kühn; Mika Silvan Goschke; Michael Weinmann
>
> **摘要:** Automated and selective harvesting of fruits has become an important area of research, particularly due to challenges such as high costs and a shortage of seasonal labor in advanced economies. This paper focuses on 6D pose estimation of strawberries using purely synthetic data generated through a procedural pipeline for photorealistic rendering. We employ the YOLOX-6D-Pose algorithm, a single-shot approach that leverages the YOLOX backbone, known for its balance between speed and accuracy, and its support for edge inference. To address the lacking availability of training data, we introduce a robust and flexible pipeline for generating synthetic strawberry data from various 3D models via a procedural Blender pipeline, where we focus on enhancing the realism of the synthesized data in comparison to previous work to make it a valuable resource for training pose estimation algorithms. Quantitative evaluations indicate that our models achieve comparable accuracy on both the NVIDIA RTX 3090 and Jetson Orin Nano across several ADD-S metrics, with the RTX 3090 demonstrating superior processing speed. However, the Jetson Orin Nano is particularly suited for resource-constrained environments, making it an excellent choice for deployment in agricultural robotics. Qualitative assessments further confirm the model's performance, demonstrating its capability to accurately infer the poses of ripe and partially ripe strawberries, while facing challenges in detecting unripe specimens. This suggests opportunities for future improvements, especially in enhancing detection capabilities for unripe strawberries (if desired) by exploring variations in color. Furthermore, the methodology presented could be adapted easily for other fruits such as apples, peaches, and plums, thereby expanding its applicability and impact in the field of agricultural automation.
>
---
#### [new 093] A Space-Time Transformer for Precipitation Forecasting
- **分类: cs.CV**

- **简介: 论文提出SaTformer空间-时间Transformer用于降水预报，解决传统数值模型计算昂贵、短时预报性能差的问题。通过视频Transformer架构和数据处理技术（重写为分类问题、类加权损失），在NeurIPS 2025挑战赛中夺冠。**

- **链接: [https://arxiv.org/pdf/2511.11090v1](https://arxiv.org/pdf/2511.11090v1)**

> **作者:** Levi Harris; Tianlong Chen
>
> **摘要:** Meteorological agencies around the world rely on real-time flood guidance to issue live-saving advisories and warnings. For decades traditional numerical weather prediction (NWP) models have been state-of-the-art for precipitation forecasting. However, physically-parameterized models suffer from a few core limitations: first, solving PDEs to resolve atmospheric dynamics is computationally demanding, and second, these methods degrade in performance at nowcasting timescales (i.e., 0-4 hour lead-times). Motivated by these shortcomings, recent work proposes AI-weather prediction (AI-WP) alternatives that learn to emulate analysis data with neural networks. While these data-driven approaches have enjoyed enormous success across diverse spatial and temporal resolutions, applications of video-understanding architectures for weather forecasting remain underexplored. To address these gaps, we propose SaTformer: a video transformer built on full space-time attention that skillfully forecasts extreme precipitation from satellite radiances. Along with our novel architecture, we introduce techniques to tame long-tailed precipitation datasets. Namely, we reformulate precipitation regression into a classification problem, and employ a class-weighted loss to address label imbalances. Our model scored first place on the NeurIPS Weather4Cast 2025 Cumulative Rainfall challenge. Code and model weights are available: https://github.com/leharris3/satformer
>
---
#### [new 094] From Retinal Pixels to Patients: Evolution of Deep Learning Research in Diabetic Retinopathy Screening
- **分类: cs.CV; cs.AI**

- **简介: 该论文系统综述2016-2025年深度学习在糖尿病视网膜病变筛查的研究，整合50+研究和20+数据集。分析自监督学习、联邦训练等方法，解决类别不平衡、标签稀缺等挑战，提出可重复、隐私保护的临床部署实践议程。**

- **链接: [https://arxiv.org/pdf/2511.11065v1](https://arxiv.org/pdf/2511.11065v1)**

> **作者:** Muskaan Chopra; Lorenz Sparrenberg; Armin Berger; Sarthak Khanna; Jan H. Terheyden; Rafet Sifa
>
> **备注:** Accepted in IEEE BigData 2025
>
> **摘要:** Diabetic Retinopathy (DR) remains a leading cause of preventable blindness, with early detection critical for reducing vision loss worldwide. Over the past decade, deep learning has transformed DR screening, progressing from early convolutional neural networks trained on private datasets to advanced pipelines addressing class imbalance, label scarcity, domain shift, and interpretability. This survey provides the first systematic synthesis of DR research spanning 2016-2025, consolidating results from 50+ studies and over 20 datasets. We critically examine methodological advances, including self- and semi-supervised learning, domain generalization, federated training, and hybrid neuro-symbolic models, alongside evaluation protocols, reporting standards, and reproducibility challenges. Benchmark tables contextualize performance across datasets, while discussion highlights open gaps in multi-center validation and clinical trust. By linking technical progress with translational barriers, this work outlines a practical agenda for reproducible, privacy-preserving, and clinically deployable DR AI. Beyond DR, many of the surveyed innovations extend broadly to medical imaging at scale.
>
---
#### [new 095] Coordinative Learning with Ordinal and Relational Priors for Volumetric Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文聚焦体积医学图像分割任务，解决现有方法因硬二元阈值忽略连续解剖相似性及全局方向不一致的问题。提出CORAL框架，通过对比排序和序数目标协调学习局部与全局解剖结构，实现最优分割性能。**

- **链接: [https://arxiv.org/pdf/2511.11276v1](https://arxiv.org/pdf/2511.11276v1)**

> **作者:** Haoyi Wang
>
> **摘要:** Volumetric medical image segmentation presents unique challenges due to the inherent anatomical structure and limited availability of annotations. While recent methods have shown promise by contrasting spatial relationships between slices, they rely on hard binary thresholds to define positive and negative samples, thereby discarding valuable continuous information about anatomical similarity. Moreover, these methods overlook the global directional consistency of anatomical progression, resulting in distorted feature spaces that fail to capture the canonical anatomical manifold shared across patients. To address these limitations, we propose Coordinative Ordinal-Relational Anatomical Learning (CORAL) to capture both local and global structure in volumetric images. First, CORAL employs a contrastive ranking objective to leverage continuous anatomical similarity, ensuring relational feature distances between slices are proportional to their anatomical position differences. In addition, CORAL incorporates an ordinal objective to enforce global directional consistency, aligning the learned feature distribution with the canonical anatomical progression across patients. Learning these inter-slice relationships produces anatomically informed representations that benefit the downstream segmentation task. Through this coordinative learning framework, CORAL achieves state-of-the-art performance on benchmark datasets under limited-annotation settings while learning representations with meaningful anatomical structure. Code is available at https://github.com/haoyiwang25/CORAL.
>
---
#### [new 096] Viper-F1: Fast and Fine-Grained Multimodal Understanding with Cross-Modal State-Space Modulation
- **分类: cs.CV**

- **简介: 该论文针对多模态视觉-语言理解任务，解决现有模型计算成本高（Transformer二次复杂度）和细粒度视觉区域捕捉不足的问题。提出Viper-F1，用状态空间动力学替代注意力机制，并引入Token-Grid Correlation Module实现轻量级视觉-文本关联，提升效率与细粒度精度。**

- **链接: [https://arxiv.org/pdf/2511.11177v1](https://arxiv.org/pdf/2511.11177v1)**

> **作者:** Quoc-Huy Trinh; Mustapha Abdullahi; Do Duy Hung Trinh; Bo Zhao; Debesh Jha
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have enabled impressive progress in vision-language understanding, yet their high computational cost limits deployment in resource-constrained scenarios such as robotic manipulation, personal assistants, and smart cameras. Most existing methods rely on Transformer-based cross-attention, whose quadratic complexity hinders efficiency. Moreover, small vision-language models often struggle to precisely capture fine-grained, task-relevant visual regions, leading to degraded performance on fine-grained reasoning tasks that limit their effectiveness in the real world. To address these issues, we introduce Viper-F1, a Hybrid State-Space Vision-Language Model that replaces attention with efficient Liquid State-Space Dynamics. To further enhance visual grounding, we propose a Token-Grid Correlation Module, which computes lightweight correlations between text tokens and image patches and modulates the state-space dynamics via FiLM conditioning. This enables the model to selectively emphasize visual regions relevant to the textual prompt while maintaining linear-time inference. Experimental results across multiple benchmarks demonstrate that Viper-F1 achieves accurate, fine-grained understanding with significantly improved efficiency.
>
---
#### [new 097] MicroVQA++: High-Quality Microscopy Reasoning Dataset with Weakly Supervised Graphs for Multimodal Large Language Model
- **分类: cs.CV**

- **简介: 论文提出MicroVQA++，解决生物医学显微镜推理中高质量数据稀缺问题。通过三阶段流程：专家验证图-文对、HiCQA-Graph过滤不一致样本、MLLM生成问题并人工筛选，构建高质量VQA数据集，使4B规模MLLM达到先进性能。**

- **链接: [https://arxiv.org/pdf/2511.11407v1](https://arxiv.org/pdf/2511.11407v1)**

> **作者:** Manyu Li; Ruian He; Chenxi Ma; Weimin Tan; Bo Yan
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Multimodal Large Language Models are increasingly applied to biomedical imaging, yet scientific reasoning for microscopy remains limited by the scarcity of large-scale, high-quality training data. We introduce MicroVQA++, a three-stage, large-scale and high-quality microscopy VQA corpus derived from the BIOMEDICA archive. Stage one bootstraps supervision from expert-validated figure-caption pairs sourced from peer-reviewed articles. Stage two applies HiCQA-Graph, a novel heterogeneous graph over images, captions, and QAs that fuses NLI-based textual entailment, CLIP-based vision-language alignment, and agent signals to identify and filter inconsistent samples. Stage three uses a MultiModal Large Language Model (MLLM) agent to generate multiple-choice questions (MCQ) followed by human screening. The resulting release comprises a large training split and a human-checked test split whose Bloom's level hard-sample distribution exceeds the MicroVQA benchmark. Our work delivers (i) a quality-controlled dataset that couples expert literature with graph-based filtering and human refinement; (ii) HiCQA-Graph, the first graph that jointly models (image, caption, QA) for cross-modal consistency filtering; (iii) evidence that careful data construction enables 4B-scale MLLMs to reach competitive microscopy reasoning performance (e.g., GPT-5) and achieve state-of-the-art performance among open-source MLLMs. Code and dataset will be released after the review process concludes.
>
---
#### [new 098] Semantic VLM Dataset for Safe Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 论文提出CAR-Scenes数据集，用于自动驾驶视觉语言模型（VLMs）的训练与评估。标注5,192张图像，覆盖28类别、350+属性，支持场景级可解释理解与风险感知分析。提供属性共现图、基线模型及分析工具，推动安全自动驾驶研究。**

- **链接: [https://arxiv.org/pdf/2511.10701v1](https://arxiv.org/pdf/2511.10701v1)**

> **作者:** Yuankai He; Weisong Shi
>
> **备注:** 8 pages, 6 figures, 7 tables
>
> **摘要:** CAR-Scenes is a frame-level dataset for autonomous driving that enables training and evaluation of vision-language models (VLMs) for interpretable, scene-level understanding. We annotate 5,192 images drawn from Argoverse 1, Cityscapes, KITTI, and nuScenes using a 28-key category/sub-category knowledge base covering environment, road geometry, background-vehicle behavior, ego-vehicle behavior, vulnerable road users, sensor states, and a discrete severity scale (1-10), totaling 350+ leaf attributes. Labels are produced by a GPT-4o-assisted vision-language pipeline with human-in-the-loop verification; we release the exact prompts, post-processing rules, and per-field baseline model performance. CAR-Scenes also provides attribute co-occurrence graphs and JSONL records that support semantic retrieval, dataset triage, and risk-aware scenario mining across sources. To calibrate task difficulty, we include reproducible, non-benchmark baselines, notably a LoRA-tuned Qwen2-VL-2B with deterministic decoding, evaluated via scalar accuracy, micro-averaged F1 for list attributes, and severity MAE/RMSE on a fixed validation split. We publicly release the annotation and analysis scripts, including graph construction and evaluation scripts, to enable explainable, data-centric workflows for future intelligent vehicles. Dataset: https://github.com/Croquembouche/CAR-Scenes
>
---
#### [new 099] Frequency-Aware Vision-Language Multimodality Generalization Network for Remote Sensing Image Classification
- **分类: cs.CV**

- **简介: 论文提出FVMGN解决遥感图像分类多模态泛化问题，克服数据异质性和缺乏模态专用语言先验。通过频率感知模块、多模态解耦及特征对齐，提升跨场景泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.10774v1](https://arxiv.org/pdf/2511.10774v1)**

> **作者:** Junjie Zhang; Feng Zhao; Hanqiang Liu; Jun Yu
>
> **摘要:** The booming remote sensing (RS) technology is giving rise to a novel multimodality generalization task, which requires the model to overcome data heterogeneity while possessing powerful cross-scene generalization ability. Moreover, most vision-language models (VLMs) usually describe surface materials in RS images using universal texts, lacking proprietary linguistic prior knowledge specific to different RS vision modalities. In this work, we formalize RS multimodality generalization (RSMG) as a learning paradigm, and propose a frequency-aware vision-language multimodality generalization network (FVMGN) for RS image classification. Specifically, a diffusion-based training-test-time augmentation (DTAug) strategy is designed to reconstruct multimodal land-cover distributions, enriching input information for FVMGN. Following that, to overcome multimodal heterogeneity, a multimodal wavelet disentanglement (MWDis) module is developed to learn cross-domain invariant features by resampling low and high frequency components in the frequency domain. Considering the characteristics of RS vision modalities, shared and proprietary class texts is designed as linguistic inputs for the transformer-based text encoder to extract diverse text features. For multimodal vision inputs, a spatial-frequency-aware image encoder (SFIE) is constructed to realize local-global feature reconstruction and representation. Finally, a multiscale spatial-frequency feature alignment (MSFFA) module is suggested to construct a unified semantic space, ensuring refined multiscale alignment of different text and vision features in spatial and frequency domains. Extensive experiments show that FVMGN has the excellent multimodality generalization ability compared with state-of-the-art (SOTA) methods.
>
---
#### [new 100] PAS: A Training-Free Stabilizer for Temporal Encoding in Video LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视频大语言模型的时序编码不稳定性问题，提出PAS机制。它解决帧时间微小变化导致注意力翻转的缺陷，通过多头相位偏移聚合实现无需训练的稳定，显著提升视频理解性能。**

- **链接: [https://arxiv.org/pdf/2511.10979v1](https://arxiv.org/pdf/2511.10979v1)**

> **作者:** Bowen Sun; Yujun Cai; Ming-Hsuan Yang; Hang Wu; Yiwei Wang
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Video LLMs suffer from temporal inconsistency: small shifts in frame timing can flip attention and suppress relevant frames. We trace this instability to the common extension of Rotary Position Embeddings to video through multimodal RoPE. The induced inverse Fourier time kernel exhibits frame-scale ripples that multiply adjacent frames by different factors, which perturbs attention that should otherwise be governed by the raw query key inner product. We present Phase Aggregated Smoothing (PAS), a simple, training-free mechanism that applies small opposed phase offsets across heads and then aggregates their outputs. PAS preserves the per-head spectrum magnitude, while the aggregation effectively smooths the temporal kernel and reduces phase sensitivity without changing the positional encoding structure. Our analysis shows that the RoPE rotated logit can be approximated as a content dot product scaled by a time kernel; smoothing this kernel yields Lipschitz stability of attention to small temporal shifts; multi phase averaging attenuates high frequency ripples while preserving per-head spectra under Nyquist-valid sampling. Experiments on multiple video understanding benchmarks under matched token budgets show consistent improvements with negligible computational overhead. PAS provides a plug and play upgrade for robust temporal encoding in Video LLMs.
>
---
#### [new 101] DoReMi: A Domain-Representation Mixture Framework for Generalizable 3D Understanding
- **分类: cs.CV**

- **简介: 论文提出DoReMi框架解决3D理解跨域泛化问题。针对多源点云异质性导致的负迁移，采用Mixture-of-Experts结构融合域感知专家与统一表示分支，通过动态路由和熵控分配实现协同学习，在ScanNet和S3DIS上显著提升mIoU性能。**

- **链接: [https://arxiv.org/pdf/2511.11232v1](https://arxiv.org/pdf/2511.11232v1)**

> **作者:** Mingwei Xing; Xinliang Wang; Yifeng Shi
>
> **摘要:** The generalization of 3D deep learning across multiple domains remains limited by the limited scale of existing datasets and the high heterogeneity of multi-source point clouds. Point clouds collected from different sensors (e.g., LiDAR scans and mesh-derived point clouds) exhibit substantial discrepancies in density and noise distribution, resulting in negative transfer during multi-domain fusion. Most existing approaches focus exclusively on either domain-aware or domain-general features, overlooking the potential synergy between them. To address this, we propose DoReMi (Domain-Representation Mixture), a Mixture-of-Experts (MoE) framework that jointly models Domain-aware Experts branch and a unified Representation branch to enable cooperative learning between specialized and generalizable knowledge. DoReMi dynamically activates domain-aware expert branch via Domain-Guided Spatial Routing (DSR) for context-aware expert selection and employs Entropy-Controlled Dynamic Allocation (EDA) for stable and efficient expert utilization, thereby adaptively modeling diverse domain distributions. Complemented by a frozen unified representation branch pretrained through robust multi-attribute self-supervised learning, DoReMi preserves cross-domain geometric and structural priors while maintaining global consistency. We evaluate DoReMi across multiple 3D understanding benchmarks. Notably, DoReMi achieves 80.1% mIoU on ScanNet Val and 77.2% mIoU on S3DIS, demonstrating competitive or superior performance compared to existing approaches, and showing strong potential as a foundation framework for future 3D understanding research. The code will be released soon.
>
---
#### [new 102] A Comparison of Lightweight Deep Learning Models for Particulate-Matter Nowcasting in the Indian Subcontinent & Surrounding Regions
- **分类: cs.CV**

- **简介: 论文属于Weather4Cast~2025污染nowcasting任务，解决印度次大陆PM1/2.5/10的6小时实时预报问题。提出轻量级深度学习模型，基于CAMS分析数据训练评估，显著提升精度与推理效率。**

- **链接: [https://arxiv.org/pdf/2511.11185v1](https://arxiv.org/pdf/2511.11185v1)**

> **作者:** Ansh Kushwaha; Kaushik Gopalan
>
> **摘要:** This paper is a submission for the Weather4Cast~2025 complementary Pollution Task and presents an efficient framework for 6-hour lead-time nowcasting of PM$_1$, PM$_{2.5}$, and PM$_{10}$ across the Indian subcontinent and surrounding regions. The proposed approach leverages analysis fields from the Copernicus Atmosphere Monitoring Service (CAMS) Global Atmospheric Composition Forecasts at 0.4 degree resolution. A 256x256 spatial region, covering 28.4S-73.6N and 32E-134.0E, is used as the model input, while predictions are generated for the central 128x128 area spanning 2.8S-48N and 57.6E-108.4E, ensuring an India-centric forecast domain with sufficient synoptic-scale context. Models are trained on CAMS analyses from 2021-2023 using a shuffled 90/10 split and independently evaluated on 2024 data. Three lightweight parameter-specific architectures are developed to improve accuracy, minimize systematic bias, and enable rapid inference. Evaluation using RMSE, MAE, Bias, and SSIM demonstrates substantial performance gains over the Aurora foundation model, underscoring the effectiveness of compact & specialized deep learning models for short-range forecasts on limited spatial domains.
>
---
#### [new 103] Preserving Cross-Modal Consistency for CLIP-based Class-Incremental Learning
- **分类: cs.CV**

- **简介: 论文解决CLIP类增量学习中的分类器偏差与分布漂移问题，提出DMC两阶段框架解耦视觉编码器与文本软提示优化，并引入DMC-OT的最优传输校准策略，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.10974v1](https://arxiv.org/pdf/2511.10974v1)**

> **作者:** Haoran Chen; Houze Xu; Micah Goldblum; Daoguo Dong; Zuxuan Wu
>
> **摘要:** Class-incremental learning (CIL) enables models to continuously learn new categories from sequential tasks without forgetting previously acquired knowledge. While recent advances in vision-language models such as CLIP have demonstrated strong generalization across domains, extending them to continual settings remains challenging. In particular, learning task-specific soft prompts for newly introduced classes often leads to severe classifier bias, as the text prototypes overfit to recent categories when prior data are unavailable. In this paper, we propose DMC, a simple yet effective two-stage framework for CLIP-based CIL that decouples the adaptation of the vision encoder and the optimization of textual soft prompts. Each stage is trained with the other frozen, allowing one modality to act as a stable semantic anchor for the other to preserve cross-modal alignment. Furthermore, current CLIP-based CIL approaches typically store class-wise Gaussian statistics for generative replay, yet they overlook the distributional drift that arises when the vision encoder is updated over time. To address this issue, we introduce DMC-OT, an enhanced version of DMC that incorporates an optimal-transport guided calibration strategy to align memory statistics across evolving encoders, along with a task-specific prompting design that enhances inter-task separability. Extensive experiments on CIFAR-100, Imagenet-R, CUB-200, and UCF-101 demonstrate that both DMC and DMC-OT achieve state-of-the-art performance, with DMC-OT further improving accuracy by an average of 1.80%.
>
---
#### [new 104] Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids
- **分类: cs.CV; cs.RO**

- **简介: 论文针对透明液体3D几何与体积估计难题，提出Phys-Liquid数据集（含97,200个模拟图像及3D网格），覆盖多场景动态行为。通过四阶段重建管道验证，显著提升精度，助力机器人精准液体操作任务。**

- **链接: [https://arxiv.org/pdf/2511.11077v1](https://arxiv.org/pdf/2511.11077v1)**

> **作者:** Ke Ma; Yizhou Fang; Jean-Baptiste Weibel; Shuai Tan; Xinggang Wang; Yang Xiao; Yi Fang; Tian Xia
>
> **备注:** 14 pages, 19 figures. Accepted as an oral paper at AAAI-26 (Main Technical Track). Code and dataset: https://github.com/dualtransparency/Phys-Liquid-AAAI Project page: https://dualtransparency.github.io/Phys-Liquid/
>
> **摘要:** Estimating the geometric and volumetric properties of transparent deformable liquids is challenging due to optical complexities and dynamic surface deformations induced by container movements. Autonomous robots performing precise liquid manipulation tasks, such as dispensing, aspiration, and mixing, must handle containers in ways that inevitably induce these deformations, complicating accurate liquid state assessment. Current datasets lack comprehensive physics-informed simulation data representing realistic liquid behaviors under diverse dynamic scenarios. To bridge this gap, we introduce Phys-Liquid, a physics-informed dataset comprising 97,200 simulation images and corresponding 3D meshes, capturing liquid dynamics across multiple laboratory scenes, lighting conditions, liquid colors, and container rotations. To validate the realism and effectiveness of Phys-Liquid, we propose a four-stage reconstruction and estimation pipeline involving liquid segmentation, multi-view mask generation, 3D mesh reconstruction, and real-world scaling. Experimental results demonstrate improved accuracy and consistency in reconstructing liquid geometry and volume, outperforming existing benchmarks. The dataset and associated validation methods facilitate future advancements in transparent liquid perception tasks. The dataset and code are available at https://dualtransparency.github.io/Phys-Liquid/.
>
---
#### [new 105] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image
- **分类: cs.CV**

- **简介: 论文提出RTGaze，解决眼神重定向任务中3D一致性差、效率低的问题。通过神经渲染和面部几何先验，实现实时（0.06秒/图像）高质量重定向，效率比SOTA快800倍。**

- **链接: [https://arxiv.org/pdf/2511.11289v1](https://arxiv.org/pdf/2511.11289v1)**

> **作者:** Hengfei Wang; Zhongqun Zhang; Yihua Cheng; Hyung Jin Chang
>
> **备注:** AAAI 2026
>
> **摘要:** Gaze redirection methods aim to generate realistic human face images with controllable eye movement. However, recent methods often struggle with 3D consistency, efficiency, or quality, limiting their practical applications. In this work, we propose RTGaze, a real-time and high-quality gaze redirection method. Our approach learns a gaze-controllable facial representation from face images and gaze prompts, then decodes this representation via neural rendering for gaze redirection. Additionally, we distill face geometric priors from a pretrained 3D portrait generator to enhance generation quality. We evaluate RTGaze both qualitatively and quantitatively, demonstrating state-of-the-art performance in efficiency, redirection accuracy, and image quality across multiple datasets. Our system achieves real-time, 3D-aware gaze redirection with a feedforward network (~0.06 sec/image), making it 800x faster than the previous state-of-the-art 3D-aware methods.
>
---
#### [new 106] CLUE: Controllable Latent space of Unprompted Embeddings for Diversity Management in Text-to-Image Synthesis
- **分类: cs.CV**

- **简介: CLUE用于文本到图像合成的多样性管理，解决医学等数据稀缺领域的挑战。它通过Style Encoder和KL散度实现无需额外数据的连续潜在空间表示，在耳炎数据集上FID降至9.30，F1提升至83.21%。**

- **链接: [https://arxiv.org/pdf/2511.10993v1](https://arxiv.org/pdf/2511.10993v1)**

> **作者:** Keunwoo Park; Jihye Chae; Joong Ho Ahn; Jihoon Kweon
>
> **摘要:** Text-to-image synthesis models require the ability to generate diverse images while maintaining stability. To overcome this challenge, a number of methods have been proposed, including the collection of prompt-image datasets and the integration of additional data modalities during training. Although these methods have shown promising results in general domains, they face limitations when applied to specialized fields such as medicine, where only limited types and insufficient amounts of data are available. We present CLUE (Controllable Latent space of Unprompted Embeddings), a generative model framework that achieves diverse generation while maintaining stability through fixed-format prompts without requiring any additional data. Based on the Stable Diffusion architecture, CLUE employs a Style Encoder that processes images and prompts to generate style embeddings, which are subsequently fed into a new second attention layer of the U-Net architecture. Through Kullback-Leibler divergence, the latent space achieves continuous representation of image features within Gaussian regions, independent of prompts. Performance was assessed on otitis media dataset. CLUE reduced FID to 9.30 (vs. 46.81) and improved recall to 70.29% (vs. 49.60%). A classifier trained on synthetic-only data at 1000% scale achieved an F1 score of 83.21% (vs. 73.83%). Combining synthetic data with equal amounts of real data achieved an F1 score of 94.76%, higher than when using only real data. On an external dataset, synthetic-only training achieved an F1 score of 76.77% (vs. 60.61%) at 1000% scale. The combined approach achieved an F1 score of 85.78%, higher than when using only the internal dataset. These results demonstrate that CLUE enables diverse yet stable image generation from limited datasets and serves as an effective data augmentation method for domain-specific applications.
>
---
#### [new 107] DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding
- **分类: cs.CV; cs.CL**

- **简介: DocLens解决长视觉文档理解中的证据定位难题，提出工具增强多智能体框架，通过导航和采样-裁决机制精准定位细节，与Gemini-2.5-Pro结合实现SOTA性能，超越人类专家。**

- **链接: [https://arxiv.org/pdf/2511.11552v1](https://arxiv.org/pdf/2511.11552v1)**

> **作者:** Dawei Zhu; Rui Meng; Jiefeng Chen; Sujian Li; Tomas Pfister; Jinsung Yoon
>
> **摘要:** Comprehending long visual documents, where information is distributed across extensive pages of text and visual elements, is a critical but challenging task for modern Vision-Language Models (VLMs). Existing approaches falter on a fundamental challenge: evidence localization. They struggle to retrieve relevant pages and overlook fine-grained details within visual elements, leading to limited performance and model hallucination. To address this, we propose DocLens, a tool-augmented multi-agent framework that effectively ``zooms in'' on evidence like a lens. It first navigates from the full document to specific visual elements on relevant pages, then employs a sampling-adjudication mechanism to generate a single, reliable answer. Paired with Gemini-2.5-Pro, DocLens achieves state-of-the-art performance on MMLongBench-Doc and FinRAGBench-V, surpassing even human experts. The framework's superiority is particularly evident on vision-centric and unanswerable queries, demonstrating the power of its enhanced localization capabilities.
>
---
#### [new 108] Benchmarking Visual LLMs Resilience to Unanswerable Questions on Visually Rich Documents
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉丰富文档VQA任务，研究VLLMs检测无法回答问题的鲁棒性，提出VRD-UQA基准，通过自动生成相关错误问题评估12模型性能，分析错误类型与知识注入策略。**

- **链接: [https://arxiv.org/pdf/2511.11468v1](https://arxiv.org/pdf/2511.11468v1)**

> **作者:** Davide Napolitano; Luca Cagliero; Fabrizio Battiloro
>
> **摘要:** The evolution of Visual Large Language Models (VLLMs) has revolutionized the automatic understanding of Visually Rich Documents (VRDs), which contain both textual and visual elements. Although VLLMs excel in Visual Question Answering (VQA) on multi-page VRDs, their ability to detect unanswerable questions is still an open research question. Our research delves into the robustness of the VLLMs to plausible yet unanswerable questions, i.e., questions that appear valid but cannot be answered due to subtle corruptions caused by swaps between related concepts or plausible question formulations. Corruptions are generated by replacing the original natural language entities with other ones of the same type, belonging to different document elements, and in different layout positions or pages of the related document. To this end, we present VRD-UQA (VISUALLY RICH DOCUMENT UNANSWERABLE QUESTION ANSWERING), a benchmark for evaluating VLLMs' resilience to plausible yet unanswerable questions across multiple dimensions. It automatically alters the questions of existing VQA datasets consisting of multi-page VRDs, verifies their unanswerability using a VLLM-as-a-judge approach, and then thoroughly evaluates VLLMs' performance. Experiments, run on 12 models, analyze: (1) The VLLMs' accuracy in detecting unanswerable questions at both page and document levels; (2) The effect of different types of corruption (NLP entity, document element, layout); (3) The effectiveness of different knowledge injection strategies based on in-context learning (OCR, multi-page selection, or the possibility of unanswerability). Our findings reveal VLLMs' limitations and demonstrate that VRD-UQA can serve as an evaluation framework for developing resilient document VQA systems.
>
---
#### [new 109] LARM: A Large Articulated-Object Reconstruction Model
- **分类: cs.CV**

- **简介: LARM提出统一前馈框架，解决3D可动物体重建中稀疏输入下的几何、纹理和关节重建问题，通过Transformer架构实现高保真输出，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.11563v1](https://arxiv.org/pdf/2511.11563v1)**

> **作者:** Sylvia Yuan; Ruoxi Shi; Xinyue Wei; Xiaoshuai Zhang; Hao Su; Minghua Liu
>
> **备注:** project page: https://sylviayuan-sy.github.io/larm-site/
>
> **摘要:** Modeling 3D articulated objects with realistic geometry, textures, and kinematics is essential for a wide range of applications. However, existing optimization-based reconstruction methods often require dense multi-view inputs and expensive per-instance optimization, limiting their scalability. Recent feedforward approaches offer faster alternatives but frequently produce coarse geometry, lack texture reconstruction, and rely on brittle, complex multi-stage pipelines. We introduce LARM, a unified feedforward framework that reconstructs 3D articulated objects from sparse-view images by jointly recovering detailed geometry, realistic textures, and accurate joint structures. LARM extends LVSM a recent novel view synthesis (NVS) approach for static 3D objects into the articulated setting by jointly reasoning over camera pose and articulation variation using a transformer-based architecture, enabling scalable and accurate novel view synthesis. In addition, LARM generates auxiliary outputs such as depth maps and part masks to facilitate explicit 3D mesh extraction and joint estimation. Our pipeline eliminates the need for dense supervision and supports high-fidelity reconstruction across diverse object categories. Extensive experiments demonstrate that LARM outperforms state-of-the-art methods in both novel view and state synthesis as well as 3D articulated object reconstruction, generating high-quality meshes that closely adhere to the input images. project page: https://sylviayuan-sy.github.io/larm-site/
>
---
#### [new 110] Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对VQA任务中AI系统置信度过度自信问题，提出AlignVQA多代理辩论框架。通过专业化代理交互精炼答案并生成校准置信度，引入可微分校准损失AlignCal优化代理。实验证实显著降低校准误差。**

- **链接: [https://arxiv.org/pdf/2511.11169v1](https://arxiv.org/pdf/2511.11169v1)**

> **作者:** Ayush Pandey; Jai Bardhan; Ishita Jain; Ramya S Hebbalaguppe; Rohan Raju Dhanakshirur; Lovekesh Vig
>
> **备注:** 17 pages, 6 figures, 5 tables. Accepted to Special Track on AI Alignment, AAAI 2026. Project Page- https://refine-align.github.io/
>
> **摘要:** In the context of Visual Question Answering (VQA) and Agentic AI, calibration refers to how closely an AI system's confidence in its answers reflects their actual correctness. This aspect becomes especially important when such systems operate autonomously and must make decisions under visual uncertainty. While modern VQA systems, powered by advanced vision-language models (VLMs), are increasingly used in high-stakes domains like medical diagnostics and autonomous navigation due to their improved accuracy, the reliability of their confidence estimates remains under-examined. Particularly, these systems often produce overconfident responses. To address this, we introduce AlignVQA, a debate-based multi-agent framework, in which diverse specialized VLM -- each following distinct prompting strategies -- generate candidate answers and then engage in two-stage interaction: generalist agents critique, refine and aggregate these proposals. This debate process yields confidence estimates that more accurately reflect the model's true predictive performance. We find that more calibrated specialized agents produce better aligned confidences. Furthermore, we introduce a novel differentiable calibration-aware loss function called aligncal designed to fine-tune the specialized agents by minimizing an upper bound on the calibration error. This objective explicitly improves the fidelity of each agent's confidence estimates. Empirical results across multiple benchmark VQA datasets substantiate the efficacy of our approach, demonstrating substantial reductions in calibration discrepancies. Furthermore, we propose a novel differentiable calibration-aware loss to fine-tune the specialized agents and improve the quality of their individual confidence estimates based on minimising upper bound calibration error.
>
---
#### [new 111] Language-Guided Graph Representation Learning for Video Summarization
- **分类: cs.CV**

- **简介: 该论文针对视频摘要任务，解决全局依赖捕捉、多模态定制及时间-语义邻近性问题。提出LGRLN：视频图生成器构建结构化图保留时序，双阈值图卷积区分语义帧，语言引导模块生成定制摘要，性能提升且推理时间减87.8%。**

- **链接: [https://arxiv.org/pdf/2511.10953v1](https://arxiv.org/pdf/2511.10953v1)**

> **作者:** Wenrui Li; Wei Han; Hengyu Man; Wangmeng Zuo; Xiaopeng Fan; Yonghong Tian
>
> **备注:** Accepted by IEEE TPAMI
>
> **摘要:** With the rapid growth of video content on social media, video summarization has become a crucial task in multimedia processing. However, existing methods face challenges in capturing global dependencies in video content and accommodating multimodal user customization. Moreover, temporal proximity between video frames does not always correspond to semantic proximity. To tackle these challenges, we propose a novel Language-guided Graph Representation Learning Network (LGRLN) for video summarization. Specifically, we introduce a video graph generator that converts video frames into a structured graph to preserve temporal order and contextual dependencies. By constructing forward, backward and undirected graphs, the video graph generator effectively preserves the sequentiality and contextual relationships of video content. We designed an intra-graph relational reasoning module with a dual-threshold graph convolution mechanism, which distinguishes semantically relevant frames from irrelevant ones between nodes. Additionally, our proposed language-guided cross-modal embedding module generates video summaries with specific textual descriptions. We model the summary generation output as a mixture of Bernoulli distribution and solve it with the EM algorithm. Experimental results show that our method outperforms existing approaches across multiple benchmarks. Moreover, we proposed LGRLN reduces inference time and model parameters by 87.8% and 91.7%, respectively. Our codes and pre-trained models are available at https://github.com/liwrui/LGRLN.
>
---
#### [new 112] SimuFreeMark: A Noise-Simulation-Free Robust Watermarking Against Image Editing
- **分类: cs.CV**

- **简介: 该论文提出SimuFreeMark，解决图像水印中现有方法依赖噪声模拟训练导致泛化不足的问题。利用图像低频分量稳定性，将水印嵌入深度特征空间，避免噪声模拟，有效抵抗图像编辑攻击，性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.11295v1](https://arxiv.org/pdf/2511.11295v1)**

> **作者:** Yichao Tang; Mingyang Li; Di Miao; Sheng Li; Zhenxing Qian; Xinpeng Zhang
>
> **摘要:** The advancement of artificial intelligence generated content (AIGC) has created a pressing need for robust image watermarking that can withstand both conventional signal processing and novel semantic editing attacks. Current deep learning-based methods rely on training with hand-crafted noise simulation layers, which inherently limit their generalization to unforeseen distortions. In this work, we propose $\textbf{SimuFreeMark}$, a noise-$\underline{\text{simu}}$lation-$\underline{\text{free}}$ water$\underline{\text{mark}}$ing framework that circumvents this limitation by exploiting the inherent stability of image low-frequency components. We first systematically establish that low-frequency components exhibit significant robustness against a wide range of attacks. Building on this foundation, SimuFreeMark embeds watermarks directly into the deep feature space of the low-frequency components, leveraging a pre-trained variational autoencoder (VAE) to bind the watermark with structurally stable image representations. This design completely eliminates the need for noise simulation during training. Extensive experiments demonstrate that SimuFreeMark outperforms state-of-the-art methods across a wide range of conventional and semantic attacks, while maintaining superior visual quality.
>
---
#### [new 113] 3D Gaussian and Diffusion-Based Gaze Redirection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对眼动重定向任务，解决3DGS模型渲染细微连续眼动变化不足的问题。提出DiT-Gaze框架，结合扩散Transformer、弱监督和正交性约束，显著提升重定向质量，误差降低4.1%。**

- **链接: [https://arxiv.org/pdf/2511.11231v1](https://arxiv.org/pdf/2511.11231v1)**

> **作者:** Abiram Panchalingam; Indu Bodala; Stuart Middleton
>
> **摘要:** High-fidelity gaze redirection is critical for generating augmented data to improve the generalization of gaze estimators. 3D Gaussian Splatting (3DGS) models like GazeGaussian represent the state-of-the-art but can struggle with rendering subtle, continuous gaze shifts. In this paper, we propose DiT-Gaze, a framework that enhances 3D gaze redirection models using a novel combination of Diffusion Transformer (DiT), weak supervision across gaze angles, and an orthogonality constraint loss. DiT allows higher-fidelity image synthesis, while our weak supervision strategy using synthetically generated intermediate gaze angles provides a smooth manifold of gaze directions during training. The orthogonality constraint loss mathematically enforces the disentanglement of internal representations for gaze, head pose, and expression. Comprehensive experiments show that DiT-Gaze sets a new state-of-the-art in both perceptual quality and redirection accuracy, reducing the state-of-the-art gaze error by 4.1% to 6.353 degrees, providing a superior method for creating synthetic training data. Our code and models will be made available for the research community to benchmark against.
>
---
#### [new 114] Toward Generalized Detection of Synthetic Media: Limitations, Challenges, and the Path to Multimodal Solutions
- **分类: cs.CV; cs.NE**

- **简介: 该论文属于合成媒体检测任务，旨在解决现有检测模型泛化性差、多模态处理无效的问题。通过回顾24篇研究，分析局限与挑战，提出基于多模态深度学习的未来解决方案。**

- **链接: [https://arxiv.org/pdf/2511.11116v1](https://arxiv.org/pdf/2511.11116v1)**

> **作者:** Redwan Hussain; Mizanur Rahman; Prithwiraj Bhattacharjee
>
> **备注:** 10 Pages, 4 figures, 1 table, 7th International Conference on Trends in Computational and Cognitive Engineering(TCCE-2025)
>
> **摘要:** Artificial intelligence (AI) in media has advanced rapidly over the last decade. The introduction of Generative Adversarial Networks (GANs) improved the quality of photorealistic image generation. Diffusion models later brought a new era of generative media. These advances made it difficult to separate real and synthetic content. The rise of deepfakes demonstrated how these tools could be misused to spread misinformation, political conspiracies, privacy violations, and fraud. For this reason, many detection models have been developed. They often use deep learning methods such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). These models search for visual, spatial, or temporal anomalies. However, such approaches often fail to generalize across unseen data and struggle with content from different models. In addition, existing approaches are ineffective in multimodal data and highly modified content. This study reviews twenty-four recent works on AI-generated media detection. Each study was examined individually to identify its contributions and weaknesses, respectively. The review then summarizes the common limitations and key challenges faced by current approaches. Based on this analysis, a research direction is suggested with a focus on multimodal deep learning models. Such models have the potential to provide more robust and generalized detection. It offers future researchers a clear starting point for building stronger defenses against harmful synthetic media.
>
---
#### [new 115] Free3D: 3D Human Motion Emerges from Single-View 2D Supervision
- **分类: cs.CV**

- **简介: 论文提出Free3D，解决3D人体运动生成中依赖3D监督导致泛化能力差的问题。通过2D监督训练，引入ML-RQ和3D-free正则化，生成高质量3D运动，无需3D标注。**

- **链接: [https://arxiv.org/pdf/2511.11368v1](https://arxiv.org/pdf/2511.11368v1)**

> **作者:** Sheng Liu; Yuanzhi Liang; Sidan Du
>
> **摘要:** Recent 3D human motion generation models demonstrate remarkable reconstruction accuracy yet struggle to generalize beyond training distributions. This limitation arises partly from the use of precise 3D supervision, which encourages models to fit fixed coordinate patterns instead of learning the essential 3D structure and motion semantic cues required for robust generalization.To overcome this limitation, we propose Free3D, a framework that synthesizes realistic 3D motions without any 3D motion annotations. Free3D introduces a Motion-Lifting Residual Quantized VAE (ML-RQ) that maps 2D motion sequences into 3D-consistent latent spaces, and a suite of 3D-free regularization objectives enforcing view consistency, orientation coherence, and physical plausibility. Trained entirely on 2D motion data, Free3D generates diverse, temporally coherent, and semantically aligned 3D motions, achieving performance comparable to or even surpassing fully 3D-supervised counterparts. These results suggest that relaxing explicit 3D supervision encourages stronger structural reasoning and generalization, offering a scalable and data-efficient paradigm for 3D motion generation.
>
---
#### [new 116] Unsupervised Robust Domain Adaptation: Paradigm, Theory and Algorithm
- **分类: cs.LG; cs.CV**

- **简介: 论文提出无监督鲁棒域适应（URDA）范式，解决UDA中对抗鲁棒性缺失问题。推导泛化边界理论，设计DART算法（两步训练），确保迁移能力和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.11009v1](https://arxiv.org/pdf/2511.11009v1)**

> **作者:** Fuxiang Huang; Xiaowei Fu; Shiyu Ye; Lina Ma; Wen Li; Xinbo Gao; David Zhang; Lei Zhang
>
> **备注:** To appear in IJCV
>
> **摘要:** Unsupervised domain adaptation (UDA) aims to transfer knowledge from a label-rich source domain to an unlabeled target domain by addressing domain shifts. Most UDA approaches emphasize transfer ability, but often overlook robustness against adversarial attacks. Although vanilla adversarial training (VAT) improves the robustness of deep neural networks, it has little effect on UDA. This paper focuses on answering three key questions: 1) Why does VAT, known for its defensive effectiveness, fail in the UDA paradigm? 2) What is the generalization bound theory under attacks and how does it evolve from classical UDA theory? 3) How can we implement a robustification training procedure without complex modifications? Specifically, we explore and reveal the inherent entanglement challenge in general UDA+VAT paradigm, and propose an unsupervised robust domain adaptation (URDA) paradigm. We further derive the generalization bound theory of the URDA paradigm so that it can resist adversarial noise and domain shift. To the best of our knowledge, this is the first time to establish the URDA paradigm and theory. We further introduce a simple, novel yet effective URDA algorithm called Disentangled Adversarial Robustness Training (DART), a two-step training procedure that ensures both transferability and robustness. DART first pre-trains an arbitrary UDA model, and then applies an instantaneous robustification post-training step via disentangled distillation.Experiments on four benchmark datasets with/without attacks show that DART effectively enhances robustness while maintaining domain adaptability, and validate the URDA paradigm and theory.
>
---
#### [new 117] Boosting Neural Video Representation via Online Structural Reparameterization
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文针对视频压缩中的神经视频表示（NVR）任务，解决模型复杂度高和容量不足导致的性能瓶颈问题。提出Online-RepNeRV框架，通过在线结构重参数化（ERB块）增强模型容量，训练后动态融合参数使额外开销仅限编码阶段，实验获0.37-2.7 dB PSNR提升，保持高效解码。**

- **链接: [https://arxiv.org/pdf/2511.11071v1](https://arxiv.org/pdf/2511.11071v1)**

> **作者:** Ziyi Li; Qingyu Mao; Shuai Liu; Qilei Li; Fanyang Meng; Yongsheng Liang
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Neural Video Representation~(NVR) is a promising paradigm for video compression, showing great potential in improving video storage and transmission efficiency. While recent advances have made efforts in architectural refinements to improve representational capability, these methods typically involve complex designs, which may incur increased computational overhead and lack the flexibility to integrate into other frameworks. Moreover, the inherent limitation in model capacity restricts the expressiveness of NVR networks, resulting in a performance bottleneck. To overcome these limitations, we propose Online-RepNeRV, a NVR framework based on online structural reparameterization. Specifically, we propose a universal reparameterization block named ERB, which incorporates multiple parallel convolutional paths to enhance the model capacity. To mitigate the overhead, an online reparameterization strategy is adopted to dynamically fuse the parameters during training, and the multi-branch structure is equivalently converted into a single-branch structure after training. As a result, the additional computational and parameter complexity is confined to the encoding stage, without affecting the decoding efficiency. Extensive experiments on mainstream video datasets demonstrate that our method achieves an average PSNR gain of 0.37-2.7 dB over baseline methods, while maintaining comparable training time and decoding speed.
>
---
#### [new 118] From Attention to Frequency: Integration of Vision Transformer and FFT-ReLU for Enhanced Image Deblurring
- **分类: eess.IV; cs.CV**

- **简介: 该论文聚焦图像去模糊任务，解决现有CNNs和ViTs在复杂模糊下性能不足、计算成本高的问题。提出双域架构，整合Vision Transformer与FFT-ReLU模块，通过空间注意力和频率稀疏性结合，显著提升PSNR、SSIM等指标。**

- **链接: [https://arxiv.org/pdf/2511.10806v1](https://arxiv.org/pdf/2511.10806v1)**

> **作者:** Syed Mumtahin Mahmud; Mahdi Mohd Hossain Noki; Prothito Shovon Majumder; Abdul Mohaimen Al Radi; Md. Haider Ali; Md. Mosaddek Khan
>
> **摘要:** Image deblurring is vital in computer vision, aiming to recover sharp images from blurry ones caused by motion or camera shake. While deep learning approaches such as CNNs and Vision Transformers (ViTs) have advanced this field, they often struggle with complex or high-resolution blur and computational demands. We propose a new dual-domain architecture that unifies Vision Transformers with a frequency-domain FFT-ReLU module, explicitly bridging spatial attention modeling and frequency sparsity. In this structure, the ViT backbone captures local and global dependencies, while the FFT-ReLU component enforces frequency-domain sparsity to suppress blur-related artifacts and preserve fine details. Extensive experiments on benchmark datasets demonstrate that this architecture achieves superior PSNR, SSIM, and perceptual quality compared to state-of-the-art models. Both quantitative metrics, qualitative comparisons, and human preference evaluations confirm its effectiveness, establishing a practical and generalizable paradigm for real-world image restoration.
>
---
#### [new 119] Enhancing Meme Emotion Understanding with Multi-Level Modality Enhancement and Dual-Stage Modal Fusion
- **分类: cs.CL; cs.CV**

- **简介: 论文聚焦Meme Emotion Understanding (MEU)任务，解决细粒度多模态融合不足和隐含意义挖掘问题。提出MemoDetector框架：通过MLLMs文本增强提取隐含信息，设计双阶段模态融合（浅层与深层），在MET-MEME和MOOD数据集上F1提升4.3%和3.4%。**

- **链接: [https://arxiv.org/pdf/2511.11126v1](https://arxiv.org/pdf/2511.11126v1)**

> **作者:** Yi Shi; Wenlong Meng; Zhenyuan Guo; Chengkun Wei; Wenzhi Chen
>
> **摘要:** With the rapid rise of social media and Internet culture, memes have become a popular medium for expressing emotional tendencies. This has sparked growing interest in Meme Emotion Understanding (MEU), which aims to classify the emotional intent behind memes by leveraging their multimodal contents. While existing efforts have achieved promising results, two major challenges remain: (1) a lack of fine-grained multimodal fusion strategies, and (2) insufficient mining of memes' implicit meanings and background knowledge. To address these challenges, we propose MemoDetector, a novel framework for advancing MEU. First, we introduce a four-step textual enhancement module that utilizes the rich knowledge and reasoning capabilities of Multimodal Large Language Models (MLLMs) to progressively infer and extract implicit and contextual insights from memes. These enhanced texts significantly enrich the original meme contents and provide valuable guidance for downstream classification. Next, we design a dual-stage modal fusion strategy: the first stage performs shallow fusion on raw meme image and text, while the second stage deeply integrates the enhanced visual and textual features. This hierarchical fusion enables the model to better capture nuanced cross-modal emotional cues. Experiments on two datasets, MET-MEME and MOOD, demonstrate that our method consistently outperforms state-of-the-art baselines. Specifically, MemoDetector improves F1 scores by 4.3\% on MET-MEME and 3.4\% on MOOD. Further ablation studies and in-depth analyses validate the effectiveness and robustness of our approach, highlighting its strong potential for advancing MEU. Our code is available at https://github.com/singing-cat/MemoDetector.
>
---
#### [new 120] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective
- **分类: cs.RO; cs.CV**

- **简介: 论文解决机器人抓取中对象级记忆推理问题，针对非马尔可夫环境中VLA模型因物体历史缺失导致决策失效的挑战。提出LIBERO-Mem任务套件和Embodied-SlotSSM框架，通过槽位中心建模实现时间可扩展的动作预测。**

- **链接: [https://arxiv.org/pdf/2511.11478v1](https://arxiv.org/pdf/2511.11478v1)**

> **作者:** Nhat Chung; Taisei Hanyu; Toan Nguyen; Huy Le; Frederick Bumgarner; Duy Minh Ho Nguyen; Khoa Vo; Kashu Yamazaki; Chase Rainwater; Tung Kieu; Anh Nguyen; Ngan Le
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** As embodied agents operate in increasingly complex environments, the ability to perceive, track, and reason about individual object instances over time becomes essential, especially in tasks requiring sequenced interactions with visually similar objects. In these non-Markovian settings, key decision cues are often hidden in object-specific histories rather than the current scene. Without persistent memory of prior interactions (what has been interacted with, where it has been, or how it has changed) visuomotor policies may fail, repeat past actions, or overlook completed ones. To surface this challenge, we introduce LIBERO-Mem, a non-Markovian task suite for stress-testing robotic manipulation under object-level partial observability. It combines short- and long-horizon object tracking with temporally sequenced subgoals, requiring reasoning beyond the current frame. However, vision-language-action (VLA) models often struggle in such settings, with token scaling quickly becoming intractable even for tasks spanning just a few hundred frames. We propose Embodied-SlotSSM, a slot-centric VLA framework built for temporal scalability. It maintains spatio-temporally consistent slot identities and leverages them through two mechanisms: (1) slot-state-space modeling for reconstructing short-term history, and (2) a relational encoder to align the input tokens with action decoding. Together, these components enable temporally grounded, context-aware action prediction. Experiments show Embodied-SlotSSM's baseline performance on LIBERO-Mem and general tasks, offering a scalable solution for non-Markovian reasoning in object-centric robotic policies.
>
---
#### [new 121] Attentive Feature Aggregation or: How Policies Learn to Stop Worrying about Robustness and Attend to Task-Relevant Visual Cues
- **分类: cs.RO; cs.CV**

- **简介: 论文针对视觉运动策略训练中预训练视觉表示导致的鲁棒性问题，提出Attentive Feature Aggregation (AFA)机制，学习关注任务相关视觉线索忽略干扰，显著提升扰动场景性能，无需额外数据增强。**

- **链接: [https://arxiv.org/pdf/2511.10762v1](https://arxiv.org/pdf/2511.10762v1)**

> **作者:** Nikolaos Tsagkas; Andreas Sochopoulos; Duolikun Danier; Sethu Vijayakumar; Alexandros Kouris; Oisin Mac Aodha; Chris Xiaoxuan Lu
>
> **备注:** This paper stems from a split of our earlier work "When Pre-trained Visual Representations Fall Short: Limitations in Visuo-Motor Robot Learning." While "The Temporal Trap" replaces the original and focuses on temporal entanglement, this companion study examines policy robustness and task-relevant visual cue selection
>
> **摘要:** The adoption of pre-trained visual representations (PVRs), leveraging features from large-scale vision models, has become a popular paradigm for training visuomotor policies. However, these powerful representations can encode a broad range of task-irrelevant scene information, making the resulting trained policies vulnerable to out-of-domain visual changes and distractors. In this work we address visuomotor policy feature pooling as a solution to the observed lack of robustness in perturbed scenes. We achieve this via Attentive Feature Aggregation (AFA), a lightweight, trainable pooling mechanism that learns to naturally attend to task-relevant visual cues, ignoring even semantically rich scene distractors. Through extensive experiments in both simulation and the real world, we demonstrate that policies trained with AFA significantly outperform standard pooling approaches in the presence of visual perturbations, without requiring expensive dataset augmentation or fine-tuning of the PVR. Our findings show that ignoring extraneous visual information is a crucial step towards deploying robust and generalisable visuomotor policies. Project Page: tsagkas.github.io/afa
>
---
#### [new 122] MOON Embedding: Multimodal Representation Learning for E-commerce Search Advertising
- **分类: cs.IR; cs.AI; cs.CV; cs.LG**

- **简介: 论文提出MOON系统，解决电子商务搜索广告中多模态表示学习与下游任务错配问题。通过三阶段训练范式和交换率定义，优化图像搜索召回率，实现CTR预测提升20%。工作涵盖数据处理、训练策略等四维度迭代优化。**

- **链接: [https://arxiv.org/pdf/2511.11305v1](https://arxiv.org/pdf/2511.11305v1)**

> **作者:** Chenghan Fu; Daoze Zhang; Yukang Lin; Zhanheng Nie; Xiang Zhang; Jianyu Liu; Yueran Liu; Wanxian Guan; Pengjie Wang; Jian Xu; Bo Zheng
>
> **备注:** 31 pages, 12 figures
>
> **摘要:** We introduce MOON, our comprehensive set of sustainable iterative practices for multimodal representation learning for e-commerce applications. MOON has already been fully deployed across all stages of Taobao search advertising system, including retrieval, relevance, ranking, and so on. The performance gains are particularly significant on click-through rate (CTR) prediction task, which achieves an overall +20.00% online CTR improvement. Over the past three years, this project has delivered the largest improvement on CTR prediction task and undergone five full-scale iterations. Throughout the exploration and iteration of our MOON, we have accumulated valuable insights and practical experience that we believe will benefit the research community. MOON contains a three-stage training paradigm of "Pretraining, Post-training, and Application", allowing effective integration of multimodal representations with downstream tasks. Notably, to bridge the misalignment between the objectives of multimodal representation learning and downstream training, we define the exchange rate to quantify how effectively improvements in an intermediate metric can translate into downstream gains. Through this analysis, we identify the image-based search recall as a critical intermediate metric guiding the optimization of multimodal models. Over three years and five iterations, MOON has evolved along four critical dimensions: data processing, training strategy, model architecture, and downstream application. The lessons and insights gained through the iterative improvements will also be shared. As part of our exploration into scaling effects in the e-commerce field, we further conduct a systematic study of the scaling laws governing multimodal representation learning, examining multiple factors such as the number of training tokens, negative samples, and the length of user behavior sequences.
>
---
#### [new 123] Deep Learning-Enhanced Analysis for Delineating Anticoagulant Essay Efficacy Using Phase Microscopy
- **分类: physics.optics; cs.CV**

- **简介: 该论文属于生物医学分析任务，解决血液凝固导致诊断不准确的问题。通过深度学习增强的数字全息显微镜框架，无标记比较EDTA与KFeOx-NPs抗凝效果，发现KFeOx-NPs有效防止凝固且不改变红细胞形态。**

- **链接: [https://arxiv.org/pdf/2511.11158v1](https://arxiv.org/pdf/2511.11158v1)**

> **作者:** S. Shrivastava; M. Rathor; D. Yenurkar; S. K. Chaubey; S. Mukherjee; R. K. Singh
>
> **摘要:** The coagulation of blood after it is drawn from the body poses a significant challenge for hematological analysis, potentially leading to inaccurate test results and altered cellular characteristics, compromising diagnostic reliability. This paper presents a deep learning-enhanced framework for delineating anticoagulant efficacy ex vivo using Digital Holographic Microscopy (DHM). We demonstrate a label-free, non-invasive approach for analyzing human blood samples, capable of accurate cell counting and morphological estimation. A DHM with an automated image processing and deep learning pipeline is built for morphological analysis of the blood cells under two different anti-coagulation agents, e.g. conventional EDTA and novel potassium ferric oxalate nanoparticles (KFeOx-NPs). This enables automated high-throughput screening of cells and estimation of blood coagulation rates when samples are treated with different anticoagulants. Results indicated that KFeOx-NPs prevented human blood coagulation without altering the cellular morphology of red blood cells (RBCs), whereas EDTA incubation caused notable changes within 6 hours of incubation. The system allows for quantitative analysis of coagulation dynamics by assessing parameters like cell clustering and morphology over time in these prepared samples, offering insights into the comparative efficacy and effects of anticoagulants outside the body.
>
---
#### [new 124] AV-Dialog: Spoken Dialogue Models with Audio-Visual Input
- **分类: cs.CL; cs.AI; cs.CV; cs.MM; cs.SD**

- **简介: AV-Dialog解决嘈杂多说话人环境下的语音对话问题。它提出首个多模态框架，融合音频与视觉输入实现目标说话者跟踪、轮次预测及响应生成。通过多任务训练，该模型在干扰下显著减少转录错误，提升对话质量与自然流畅度。**

- **链接: [https://arxiv.org/pdf/2511.11124v1](https://arxiv.org/pdf/2511.11124v1)**

> **作者:** Tuochao Chen; Bandhav Veluri; Hongyu Gong; Shyamnath Gollakota
>
> **摘要:** Dialogue models falter in noisy, multi-speaker environments, often producing irrelevant responses and awkward turn-taking. We present AV-Dialog, the first multimodal dialog framework that uses both audio and visual cues to track the target speaker, predict turn-taking, and generate coherent responses. By combining acoustic tokenization with multi-task, multi-stage training on monadic, synthetic, and real audio-visual dialogue datasets, AV-Dialog achieves robust streaming transcription, semantically grounded turn-boundary detection and accurate responses, resulting in a natural conversational flow. Experiments show that AV-Dialog outperforms audio-only models under interference, reducing transcription errors, improving turn-taking prediction, and enhancing human-rated dialogue quality. These results highlight the power of seeing as well as hearing for speaker-aware interaction, paving the way for {spoken} dialogue agents that perform {robustly} in real-world, noisy environments.
>
---
#### [new 125] AccKV: Towards Efficient Audio-Video LLMs Inference via Adaptive-Focusing and Cross-Calibration KV Cache Optimization
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 论文针对音频-视频大语言模型推理效率问题，提出AccKV框架。解决KV缓存过大导致的模态混淆与性能下降，通过自适应聚焦和交叉校准优化缓存，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2511.11106v1](https://arxiv.org/pdf/2511.11106v1)**

> **作者:** Zhonghua Jiang; Kui Chen; Kunxi Li; Keting Yin; Yiyun Zhou; Zhaode Wang; Chengfei Lv; Shengyu Zhang
>
> **摘要:** Recent advancements in Audio-Video Large Language Models (AV-LLMs) have enhanced their capabilities in tasks like audio-visual question answering and multimodal dialog systems. Video and audio introduce an extended temporal dimension, resulting in a larger key-value (KV) cache compared to static image embedding. A naive optimization strategy is to selectively focus on and retain KV caches of audio or video based on task. However, in the experiment, we observed that the attention of AV-LLMs to various modalities in the high layers is not strictly dependent on the task. In higher layers, the attention of AV-LLMs shifts more towards the video modality. In addition, we also found that directly integrating temporal KV of audio and spatial-temporal KV of video may lead to information confusion and significant performance degradation of AV-LLMs. If audio and video are processed indiscriminately, it may also lead to excessive compression or reservation of a certain modality, thereby disrupting the alignment between modalities. To address these challenges, we propose AccKV, an Adaptive-Focusing and Cross-Calibration KV cache optimization framework designed specifically for efficient AV-LLMs inference. Our method is based on layer adaptive focusing technology, selectively focusing on key modalities according to the characteristics of different layers, and enhances the recognition of heavy hitter tokens through attention redistribution. In addition, we propose a Cross-Calibration technique that first integrates inefficient KV caches within the audio and video modalities, and then aligns low-priority modalities with high-priority modalities to selectively evict KV cache of low-priority modalities. The experimental results show that AccKV can significantly improve the computational efficiency of AV-LLMs while maintaining accuracy.
>
---
#### [new 126] Collaborative Representation Learning for Alignment of Tactile, Language, and Vision Modalities
- **分类: cs.RO; cs.CV**

- **简介: 论文提出TLV-CoRe方法解决触觉-语言-视觉模态对齐任务。针对触觉传感器不标准化导致的冗余特征和跨模态交互不足，引入Sensor-Aware Modulator统一触觉特征、Unified Bridging Adapter增强三模态交互，并设计RSS评估框架。实验表明其显著提升跨模态对齐效果。**

- **链接: [https://arxiv.org/pdf/2511.11512v1](https://arxiv.org/pdf/2511.11512v1)**

> **作者:** Yiyun Zhou; Mingjing Xu; Jingwei Shi; Quanjiang Li; Jingyuan Chen
>
> **摘要:** Tactile sensing offers rich and complementary information to vision and language, enabling robots to perceive fine-grained object properties. However, existing tactile sensors lack standardization, leading to redundant features that hinder cross-sensor generalization. Moreover, existing methods fail to fully integrate the intermediate communication among tactile, language, and vision modalities. To address this, we propose TLV-CoRe, a CLIP-based Tactile-Language-Vision Collaborative Representation learning method. TLV-CoRe introduces a Sensor-Aware Modulator to unify tactile features across different sensors and employs tactile-irrelevant decoupled learning to disentangle irrelevant tactile features. Additionally, a Unified Bridging Adapter is introduced to enhance tri-modal interaction within the shared representation space. To fairly evaluate the effectiveness of tactile models, we further propose the RSS evaluation framework, focusing on Robustness, Synergy, and Stability across different methods. Experimental results demonstrate that TLV-CoRe significantly improves sensor-agnostic representation learning and cross-modal alignment, offering a new direction for multimodal tactile representation.
>
---
#### [new 127] CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文提出CLIPPan，解决无监督遥感图像融合中的分辨率域适应问题。通过微调CLIP作为语言监督器，利用文本提示引导融合过程，无需真实标签，实现在真实数据集上的新SOTA。**

- **链接: [https://arxiv.org/pdf/2511.10896v1](https://arxiv.org/pdf/2511.10896v1)**

> **作者:** Lihua Jian; Jiabo Liu; Shaowu Wu; Lihui Chen
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Despite remarkable advancements in supervised pansharpening neural networks, these methods face domain adaptation challenges of resolution due to the intrinsic disparity between simulated reduced-resolution training data and real-world full-resolution scenarios.To bridge this gap, we propose an unsupervised pansharpening framework, CLIPPan, that enables model training at full resolution directly by taking CLIP, a visual-language model, as a supervisor. However, directly applying CLIP to supervise pansharpening remains challenging due to its inherent bias toward natural images and limited understanding of pansharpening tasks. Therefore, we first introduce a lightweight fine-tuning pipeline that adapts CLIP to recognize low-resolution multispectral, panchromatic, and high-resolution multispectral images, as well as to understand the pansharpening process. Then, building on the adapted CLIP, we formulate a novel \textit{loss integrating semantic language constraints}, which aligns image-level fusion transitions with protocol-aligned textual prompts (e.g., Wald's or Khan's descriptions), thus enabling CLIPPan to use language as a powerful supervisory signal and guide fusion learning without ground truth. Extensive experiments demonstrate that CLIPPan consistently improves spectral and spatial fidelity across various pansharpening backbones on real-world datasets, setting a new state of the art for unsupervised full-resolution pansharpening.
>
---
#### [new 128] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出MoCo-INR，无监督心脏MRI重建方法，解决高加速成像中图像质量差和依赖真实数据的问题。通过整合隐式神经表示与运动补偿框架，实现准确运动分解和高质量重建，在模拟和真实数据验证。**

- **链接: [https://arxiv.org/pdf/2511.11436v1](https://arxiv.org/pdf/2511.11436v1)**

> **作者:** Xuanyu Tian; Lixuan Chen; Qing Wu; Xiao Wang; Jie Feng; Yuyao Zhang; Hongjiang Wei
>
> **备注:** Accepted by AAAI-26
>
> **摘要:** Cardiac magnetic resonance (CMR) imaging is widely used to characterize cardiac morphology and function. To accelerate CMR imaging, various methods have been proposed to recover high-quality spatiotemporal CMR images from highly undersampled k-t space data. However, current CMR reconstruction techniques either fail to achieve satisfactory image quality or are restricted by the scarcity of ground truth data, leading to limited applicability in clinical scenarios. In this work, we proposed MoCo-INR, a new unsupervised method that integrates implicit neural representations (INR) with the conventional motion-compensated (MoCo) framework. Using explicit motion modeling and the continuous prior of INRs, MoCo-INR can produce accurate cardiac motion decomposition and high-quality CMR reconstruction. Furthermore, we introduce a new INR network architecture tailored to the CMR problem, which significantly stabilizes model optimization. Experiments on retrospective (simulated) datasets demonstrate the superiority of MoCo-INR over state-of-the-art methods, achieving fast convergence and fine-detailed reconstructions at ultra-high acceleration factors (e.g., 20x in VISTA sampling). Additionally, evaluations on prospective (real-acquired) free-breathing CMR scans highlight the clinical practicality of MoCo-INR for real-time imaging. Several ablation studies further confirm the effectiveness of the critical components of MoCo-INR.
>
---
#### [new 129] Synergy vs. Noise: Performance-Guided Multimodal Fusion For Biochemical Recurrence-Free Survival in Prostate Cancer
- **分类: q-bio.QM; cs.CV; cs.LG; eess.IV**

- **简介: 该论文研究前列腺癌生化复发预测任务，解决多模态融合中模态质量影响性能的问题。通过实证分析前列腺癌数据（组织病理学、放射学、临床），证明仅整合高性能模态提升预测，整合低性能模态引入噪声，强调性能导向的融合策略。**

- **链接: [https://arxiv.org/pdf/2511.11452v1](https://arxiv.org/pdf/2511.11452v1)**

> **作者:** Seth Alain Chang; Muhammad Mueez Amjad; Noorul Wahab; Ethar Alzaid; Nasir Rajpoot; Adam Shephard
>
> **备注:** 5 pages, 1 figure, 4 tables
>
> **摘要:** Multimodal deep learning (MDL) has emerged as a transformative approach in computational pathology. By integrating complementary information from multiple data sources, MDL models have demonstrated superior predictive performance across diverse clinical tasks compared to unimodal models. However, the assumption that combining modalities inherently improves performance remains largely unexamined. We hypothesise that multimodal gains depend critically on the predictive quality of individual modalities, and that integrating weak modalities may introduce noise rather than complementary information. We test this hypothesis on a prostate cancer dataset with histopathology, radiology, and clinical data to predict time-to-biochemical recurrence. Our results confirm that combining high-performing modalities yield superior performance compared to unimodal approaches. However, integrating a poor-performing modality with other higher-performing modalities degrades predictive accuracy. These findings demonstrate that multimodal benefit requires selective, performance-guided integration rather than indiscriminate modality combination, with implications for MDL design across computational pathology and medical imaging.
>
---
#### [new 130] Grounded Visual Factualization: Factual Anchor-Based Finetuning for Enhancing MLLM Factual Consistency
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对MLLM视觉幻觉问题，提出GVF微调方法，通过事实锚点机制提升视觉事实一致性，在VHTest上显著优于基线，同时保持通用性能。**

- **链接: [https://arxiv.org/pdf/2511.10671v1](https://arxiv.org/pdf/2511.10671v1)**

> **作者:** Filippo Morbiato; Luca Romano; Alessandro Persona
>
> **摘要:** Visual hallucination, where Multimodal Large Language Models fabricate details inconsistent with image content, critically undermines their reliability. Existing fine-tuning methods offer limited improvement, failing to deeply intervene in factual reasoning. This paper introduces Grounded Visual Factualization (GVF) Finetuning, a novel approach to systematically enhance MLLM visual factual consistency. GVF integrates explicit factual signals via three core mechanisms: Factual Anchor Data Augmentation, enriching training data with structured factual anchors and counter-factual prompts; Fact-Aware Instruction Tuning, embedding these cues into explicit instructions; and a Factual Consistency Loss function, specifically penalizing factual inaccuracies. Evaluated on LLaVA-1.5-13B, GVF Finetuning significantly outperforms standard fine-tuning on the VHTest benchmark for both Open-Ended Question (OEQ) and Yes/No Question (YNQ) formats. Crucially, GVF maintains or even slightly improves performance on general multimodal benchmarks like MME and POPE, demonstrating effective mitigation of visual hallucinations without compromising general understanding and reasoning abilities.
>
---
#### [new 131] From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究可控模型合并任务，解决现有方法离线优化计算成本高、复杂度指数增长的问题。提出基于表示的闭式方法，通过线性变换直接校正模型表示，单步生成帕累托最优模型，复杂度线性于任务数，显著降低计算成本。**

- **链接: [https://arxiv.org/pdf/2511.10943v1](https://arxiv.org/pdf/2511.10943v1)**

> **作者:** Jialin Wu; Jian Yang; Handing Wang; Jiajun Wen; Zhiyong Yu
>
> **备注:** Accepted by AAAI 2026, Extended Version
>
> **摘要:** Model merging combines expert models for multitask performance but faces challenges from parameter interference. This has sparked recent interest in controllable model merging, giving users the ability to explicitly balance performance trade-offs. Existing approaches employ a compile-then-query paradigm, performing a costly offline multi-objective optimization to enable fast, preference-aware model generation. This offline stage typically involves iterative search or dedicated training, with complexity that grows exponentially with the number of tasks. To overcome these limitations, we shift the perspective from parameter-space optimization to a direct correction of the model's final representation. Our approach models this correction as an optimal linear transformation, yielding a closed-form solution that replaces the entire offline optimization process with a single-step, architecture-agnostic computation. This solution directly incorporates user preferences, allowing a Pareto-optimal model to be generated on-the-fly with complexity that scales linearly with the number of tasks. Experimental results show our method generates a superior Pareto front with more precise preference alignment and drastically reduced computational cost.
>
---
#### [new 132] Low-Bit, High-Fidelity: Optimal Transport Quantization for Flow Matching
- **分类: cs.LG; cs.CV**

- **简介: 论文解决Flow Matching生成模型高精度参数部署难题，提出基于最优传输的量化方法，最小化2-Wasserstein距离。实验证明在2-3位/参数下保持生成质量与潜空间稳定，优于均匀、分段和对数量化，适用于边缘AI。**

- **链接: [https://arxiv.org/pdf/2511.11418v1](https://arxiv.org/pdf/2511.11418v1)**

> **作者:** Dara Varam; Diaa A. Abuhani; Imran Zualkernan; Raghad AlDamani; Lujain Khalil
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Flow Matching (FM) generative models offer efficient simulation-free training and deterministic sampling, but their practical deployment is challenged by high-precision parameter requirements. We adapt optimal transport (OT)-based post-training quantization to FM models, minimizing the 2-Wasserstein distance between quantized and original weights, and systematically compare its effectiveness against uniform, piecewise, and logarithmic quantization schemes. Our theoretical analysis provides upper bounds on generative degradation under quantization, and empirical results across five benchmark datasets of varying complexity show that OT-based quantization preserves both visual generation quality and latent space stability down to 2-3 bits per parameter, where alternative methods fail. This establishes OT-based quantization as a principled, effective approach to compress FM generative models for edge and embedded AI applications.
>
---
#### [new 133] DualVision ArthroNav: Investigating Opportunities to Enhance Localization and Reconstruction in Image-based Arthroscopy Navigation via External Cameras
- **分类: eess.IV; cs.CV; cs.RO**

- **简介: 论文提出DualVision ArthroNav系统，用于关节镜手术导航任务。解决纯视觉导航的漂移与尺度模糊问题，通过集成外部相机提供稳定定位，单目关节镜实现场景重建。实验验证平均轨迹误差1.09mm，注册误差2.16mm。**

- **链接: [https://arxiv.org/pdf/2511.10699v1](https://arxiv.org/pdf/2511.10699v1)**

> **作者:** Hongchao Shu; Lalithkumar Seenivasan; Mingxu Liu; Yunseo Hwang; Yu-Chun Ku; Jonathan Knopf; Alejandro Martin-Gomez; Mehran Armand; Mathias Unberath
>
> **摘要:** Arthroscopic procedures can greatly benefit from navigation systems that enhance spatial awareness, depth perception, and field of view. However, existing optical tracking solutions impose strict workspace constraints and disrupt surgical workflow. Vision-based alternatives, though less invasive, often rely solely on the monocular arthroscope camera, making them prone to drift, scale ambiguity, and sensitivity to rapid motion or occlusion. We propose DualVision ArthroNav, a multi-camera arthroscopy navigation system that integrates an external camera rigidly mounted on the arthroscope. The external camera provides stable visual odometry and absolute localization, while the monocular arthroscope video enables dense scene reconstruction. By combining these complementary views, our system resolves the scale ambiguity and long-term drift inherent in monocular SLAM and ensures robust relocalization. Experiments demonstrate that our system effectively compensates for calibration errors, achieving an average absolute trajectory error of 1.09 mm. The reconstructed scenes reach an average target registration error of 2.16 mm, with high visual fidelity (SSIM = 0.69, PSNR = 22.19). These results indicate that our system provides a practical and cost-efficient solution for arthroscopic navigation, bridging the gap between optical tracking and purely vision-based systems, and paving the way toward clinically deployable, fully vision-based arthroscopic guidance.
>
---
#### [new 134] Large-scale modality-invariant foundation models for brain MRI analysis: Application to lesion segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对脑部MRI病变分割任务，解决现有自监督学习框架不适用于多模态MRI的问题。提出模态不变表示学习方法，实验表明病变分割主要受益于保留细粒度模态特定特征而非跨模态对齐。**

- **链接: [https://arxiv.org/pdf/2511.11311v1](https://arxiv.org/pdf/2511.11311v1)**

> **作者:** Petros Koutsouvelis; Matej Gazda; Leroy Volmer; Sina Amirrajab; Kamil Barbierik; Branislav Setlak; Jakub Gazda; Peter Drotar
>
> **备注:** Submitted to IEEE ISBI 2026
>
> **摘要:** The field of computer vision is undergoing a paradigm shift toward large-scale foundation model pre-training via self-supervised learning (SSL). Leveraging large volumes of unlabeled brain MRI data, such models can learn anatomical priors that improve few-shot performance in diverse neuroimaging tasks. However, most SSL frameworks are tailored to natural images, and their adaptation to capture multi-modal MRI information remains underexplored. This work proposes a modality-invariant representation learning setup and evaluates its effectiveness in stroke and epilepsy lesion segmentation, following large-scale pre-training. Experimental results suggest that despite successful cross-modality alignment, lesion segmentation primarily benefits from preserving fine-grained modality-specific features. Model checkpoints and code are made publicly available.
>
---
#### [new 135] LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对长尾分类任务，解决PEFT方法在尾类优化时牺牲头类性能的问题。提出LT-Soups框架：第一阶段平均平衡子集微调模型减少头类偏差；第二阶段微调分类器恢复头类准确率。在多个数据集上实现更优的头尾性能平衡。**

- **链接: [https://arxiv.org/pdf/2511.10683v1](https://arxiv.org/pdf/2511.10683v1)**

> **作者:** Masih Aminbeidokhti; Subhankar Roy; Eric Granger; Elisa Ricci; Marco Pedersoli
>
> **备注:** Neurips 2025
>
> **摘要:** Real-world datasets typically exhibit long-tailed (LT) distributions, where a few head classes dominate and many tail classes are severely underrepresented. While recent work shows that parameter-efficient fine-tuning (PEFT) methods like LoRA and AdaptFormer preserve tail-class performance on foundation models such as CLIP, we find that they do so at the cost of head-class accuracy. We identify the head-tail ratio, the proportion of head to tail classes, as a crucial but overlooked factor influencing this trade-off. Through controlled experiments on CIFAR100 with varying imbalance ratio ($ρ$) and head-tail ratio ($η$), we show that PEFT excels in tail-heavy scenarios but degrades in more balanced and head-heavy distributions. To overcome these limitations, we propose LT-Soups, a two-stage model soups framework designed to generalize across diverse LT regimes. In the first stage, LT-Soups averages models fine-tuned on balanced subsets to reduce head-class bias; in the second, it fine-tunes only the classifier on the full dataset to restore head-class accuracy. Experiments across six benchmark datasets show that LT-Soups achieves superior trade-offs compared to both PEFT and traditional model soups across a wide range of imbalance regimes.
>
---
## 更新

#### [replaced 001] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09347v2](https://arxiv.org/pdf/2511.09347v2)**

> **作者:** Jiangyong Yu; Changyong Shu; Sifan Zhou; Zichen Yu; Xing Hu; Yan Chen; Dawei Yang
>
> **备注:** I made an operational error. I intended to update the paper with Identifier arXiv:2502.15488, not submit a new paper with a different identifier. Therefore, I would like to withdraw the current submission and resubmit an updated version for Identifier arXiv:2502.15488
>
> **摘要:** Camera-based multi-view 3D detection is crucial for autonomous driving. PETR and its variants (PETRs) excel in benchmarks but face deployment challenges due to high computational cost and memory footprint. Quantization is an effective technique for compressing deep neural networks by reducing the bit width of weights and activations. However, directly applying existing quantization methods to PETRs leads to severe accuracy degradation. This issue primarily arises from two key challenges: (1) significant magnitude disparity between multi-modal features-specifically, image features and camera-ray positional embeddings (PE), and (2) the inefficiency and approximation error of quantizing non-linear operators, which commonly rely on hardware-unfriendly computations. In this paper, we propose FQ-PETR, a fully quantized framework for PETRs, featuring three key innovations: (1) Quantization-Friendly LiDAR-ray Position Embedding (QFPE): Replacing multi-point sampling with LiDAR-prior-guided single-point sampling and anchor-based embedding eliminates problematic non-linearities (e.g., inverse-sigmoid) and aligns PE scale with image features, preserving accuracy. (2) Dual-Lookup Table (DULUT): This algorithm approximates complex non-linear functions using two cascaded linear LUTs, achieving high fidelity with minimal entries and no specialized hardware. (3) Quantization After Numerical Stabilization (QANS): Performing quantization after softmax numerical stabilization mitigates attention distortion from large inputs. On PETRs (e.g. PETR, StreamPETR, PETRv2, MV2d), FQ-PETR under W8A8 achieves near-floating-point accuracy (1% degradation) while reducing latency by up to 75%, significantly outperforming existing PTQ and QAT baselines.
>
---
#### [replaced 002] ORIC: Benchmarking Object Recognition under Contextual Incongruity in Large Vision-Language Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.15695v2](https://arxiv.org/pdf/2509.15695v2)**

> **作者:** Zhaoyang Li; Zhan Ling; Yuchen Zhou; Litian Gong; Erdem Bıyık; Hao Su
>
> **摘要:** Large Vision-Language Models (LVLMs) excel at captioning, visual question answering, and robotics by combining vision and language, yet they often miss obvious objects or hallucinate nonexistent ones in atypical scenes. We examine these failures through the lens of uncertainty, focusing on contextual incongruity, where objects appear unexpectedly or fail to appear in expected contexts, and show that such cases increase recognition difficulty for state-of-the-art LVLMs. To study this regime, we introduce the Object Recognition in Incongruous Context (ORIC) framework, which constructs incongruous object-context pairs through two complementary strategies: (1) LLM-guided sampling to identify hard-to-recognize objects present in the image and (2) CLIP-guided sampling to mine plausible but absent ones. Applied to MSCOCO, ORIC produces ORIC-Bench and ORIC-style training data. Evaluating 18 LVLMs and 2 open-vocabulary detectors reveals substantial performance drops and bias patterns under incongruous contexts. Fine-tuning Qwen3-VL-8B-Instruct with Visual Reinforcement Fine-Tuning on 600 ORIC-style samples improves results on ORIC-Bench, AMBER, and HallusionBench. Overall, we show that contextual incongruity is a key source of uncertainty and provide tools for more reliable LVLMs. The code is available at https://github.com/ZhaoyangLi-1/ORIC.
>
---
#### [replaced 003] Enhanced Structured Lasso Pruning with Class-wise Information
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.09125v3](https://arxiv.org/pdf/2502.09125v3)**

> **作者:** Xiang Liu; Mingchen Li; Xia Li; Leigang Qu; Guangsu Wang; Zifan Peng; Yijun Song; Zemin Liu; Linshan Jiang; Jialin Li
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Modern applications require lightweight neural network models. Most existing neural network pruning methods focus on removing unimportant filters; however, these may result in the loss of statistical information after pruning due to failing to consider the class-wise information. In this paper, we employ the structured lasso from the perspective of utilizing precise class-wise information for model pruning with the help of Information Bottleneck theory, which guides us to ensure the retention of statistical information before and after pruning. With these techniques, we propose two novel adaptive network pruning schemes in parallel: sparse graph-structured lasso pruning with Information Bottleneck (sGLP-IB) and sparse tree-guided lasso pruning with Information Bottleneck (sTLP-IB). The key component is that we prune the model filters utilizing sGLP-IB and sTLP-IB with more precise structured class-wise relatedness. Compared to multiple state-of-the-art methods, our approaches achieve the best performance across three datasets and six model structures on extensive experiments. For example, with the VGG16 model based on the CIFAR-10 dataset, we can reduce the parameters by 85%, decrease the FLOPs by 61%, and maintain an accuracy of 94.10% (0.14% better than the original). For large-scale ImageNet, we can reduce the parameters by 55% while keeping the accuracy at 76.12% (only drop 0.03%) using the ResNet architecture. In summary, we succeed in reducing the model size and computational resource usage while maintaining the effectiveness of accuracy.
>
---
#### [replaced 004] Adaptive Cache Enhancement for Test-Time Adaptation of Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.07570v2](https://arxiv.org/pdf/2508.07570v2)**

> **作者:** Khanh-Binh Nguyen; Phuoc-Nguyen Bui; Hyunseung Choo; Duc Thanh Nguyen
>
> **备注:** 12 pages, Under review
>
> **摘要:** Vision-language models (VLMs) exhibit remarkable zero-shot generalization but suffer performance degradation under distribution shifts in downstream tasks, particularly in the absence of labeled data. Test-Time Adaptation (TTA) addresses this challenge by enabling online optimization of VLMs during inference, eliminating the need for annotated data. Cache-based TTA methods exploit historical knowledge by maintaining a dynamic memory cache of low-entropy or high-confidence samples, promoting efficient adaptation to out-of-distribution data. Nevertheless, these methods face two critical challenges: (1) unreliable confidence metrics under significant distribution shifts, resulting in error accumulation within the cache and degraded adaptation performance; and (2) rigid decision boundaries that fail to accommodate substantial distributional variations, leading to suboptimal predictions. To overcome these limitations, we introduce the Adaptive Cache Enhancement (ACE) framework, which constructs a robust cache by selectively storing high-confidence or low-entropy image embeddings per class, guided by dynamic, class-specific thresholds initialized from zero-shot statistics and iteratively refined using an exponential moving average and exploration-augmented updates. This approach enables adaptive, class-wise decision boundaries, ensuring robust and accurate predictions across diverse visual distributions. Extensive experiments on 15 diverse benchmark datasets demonstrate that ACE achieves state-of-the-art performance, delivering superior robustness and generalization compared to existing TTA methods in challenging out-of-distribution scenarios.
>
---
#### [replaced 005] Fractured Glass, Failing Cameras: Simulating Physics-Based Adversarial Samples for Autonomous Driving Systems
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [https://arxiv.org/pdf/2405.15033v3](https://arxiv.org/pdf/2405.15033v3)**

> **作者:** Manav Prabhakar; Jwalandhar Girnar; Arpan Kusari
>
> **备注:** Accepted to AAAI
>
> **摘要:** While much research has recently focused on generating physics-based adversarial samples, a critical yet often overlooked category originates from physical failures within on-board cameras-components essential to the perception systems of autonomous vehicles. Camera failures, whether due to external stresses causing hardware breakdown or internal component faults, can directly jeopardize the safety and reliability of autonomous driving systems. Firstly, we motivate the study using two separate real-world experiments to showcase that indeed glass failures would cause the detection based neural network models to fail. Secondly, we develop a simulation-based study using the physical process of the glass breakage to create perturbed scenarios, representing a realistic class of physics-based adversarial samples. Using a finite element model (FEM)-based approach, we generate surface cracks on the camera image by applying a stress field defined by particles within a triangular mesh. Lastly, we use physically-based rendering (PBR) techniques to provide realistic visualizations of these physically plausible fractures. To assess the safety implications, we apply the simulated broken glass effects as image filters to two autonomous driving datasets- KITTI and BDD100K- as well as the large-scale image detection dataset MS-COCO. We then evaluate detection failure rates for critical object classes using CNN-based object detection models (YOLOv8 and Faster R-CNN) and a transformer-based architecture with Pyramid Vision Transformers. To further investigate the distributional impact of these visual distortions, we compute the Kullback-Leibler (K-L) divergence between three distinct data distributions, applying various broken glass filters to a custom dataset (captured through a cracked windshield), as well as the KITTI and Kaggle cats and dogs datasets.
>
---
#### [replaced 006] AI Assisted AR Assembly: Object Recognition and Computer Vision for Augmented Reality Assisted Assembly
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.05394v2](https://arxiv.org/pdf/2511.05394v2)**

> **作者:** Alexander Htet Kyaw; Haotian Ma; Sasa Zivkovic; Jenny Sabin
>
> **备注:** Accepted to the Association for Computing Machinery (ACM) Symposium on Computational Fabrication (SCF '25)
>
> **摘要:** We present an AI-assisted Augmented Reality assembly workflow that uses deep learning-based object recognition to identify different assembly components and display step-by-step instructions. For each assembly step, the system displays a bounding box around the corresponding components in the physical space, and where the component should be placed. By connecting assembly instructions with the real-time location of relevant components, the system eliminates the need for manual searching, sorting, or labeling of different components before each assembly. To demonstrate the feasibility of using object recognition for AR-assisted assembly, we highlight a case study involving the assembly of LEGO sculptures.
>
---
#### [replaced 007] Duplex-GS: Proxy-Guided Weighted Blending for Real-Time Order-Independent Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03180v2](https://arxiv.org/pdf/2508.03180v2)**

> **作者:** Weihang Liu; Yuke Li; Yuxuan Li; Jingyi Yu; Xin Lou
>
> **备注:** submitted to TCSVT
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated remarkable rendering fidelity and efficiency. However, these methods still rely on computationally expensive sequential alpha-blending operations, resulting in significant overhead, particularly on resource-constrained platforms. In this paper, we propose Duplex-GS, a dual-hierarchy framework that integrates proxy Gaussian representations with order-independent rendering techniques to achieve photorealistic results while sustaining real-time performance. To mitigate the overhead caused by view-adaptive radix sort, we introduce cell proxies for local Gaussians management and propose cell search rasterization for further acceleration. By seamlessly combining our framework with Order-Independent Transparency (OIT), we develop a physically inspired weighted sum rendering technique that simultaneously eliminates "popping" and "transparency" artifacts, yielding substantial improvements in both accuracy and efficiency. Extensive experiments on a variety of real-world datasets demonstrate the robustness of our method across diverse scenarios, including multi-scale training views and large-scale environments. Our results validate the advantages of the OIT rendering paradigm in Gaussian Splatting, achieving high-quality rendering with an impressive 1.5 to 4 speedup over existing OIT based Gaussian Splatting approaches and 52.2% to 86.9% reduction of the radix sort overhead without quality degradation.
>
---
#### [replaced 008] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.15488v3](https://arxiv.org/pdf/2502.15488v3)**

> **作者:** Jiangyong Yu; Changyong Shu; Sifan Zhou; Zichen Yu; Xing Hu; Yan Chen; Dawei Yang
>
> **备注:** This paper is acceptted by AAAI 2026
>
> **摘要:** Camera-based multi-view 3D detection is crucial for autonomous driving. PETR and its variants (PETRs) excel in benchmarks but face deployment challenges due to high computational cost and memory footprint. Quantization is an effective technique for compressing deep neural networks by reducing the bit width of weights and activations. However, directly applying existing quantization methods to PETRs leads to severe accuracy degradation. This issue primarily arises from two key challenges: (1) significant magnitude disparity between multi-modal features-specifically, image features and camera-ray positional embeddings (PE), and (2) the inefficiency and approximation error of quantizing non-linear operators, which commonly rely on hardware-unfriendly computations. In this paper, we propose FQ-PETR, a fully quantized framework for PETRs, featuring three key innovations: (1) Quantization-Friendly LiDAR-ray Position Embedding (QFPE): Replacing multi-point sampling with LiDAR-prior-guided single-point sampling and anchor-based embedding eliminates problematic non-linearities (e.g., inverse-sigmoid) and aligns PE scale with image features, preserving accuracy. (2) Dual-Lookup Table (DULUT): This algorithm approximates complex non-linear functions using two cascaded linear LUTs, achieving high fidelity with minimal entries and no specialized hardware. (3) Quantization After Numerical Stabilization (QANS): Performing quantization after softmax numerical stabilization mitigates attention distortion from large inputs. On PETRs (e.g. PETR, StreamPETR, PETRv2, MV2d), FQ-PETR under W8A8 achieves near-floating-point accuracy (1% degradation) while reducing latency by up to 75%, significantly outperforming existing PTQ and QAT baselines.
>
---
#### [replaced 009] OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10560v2](https://arxiv.org/pdf/2511.10560v2)**

> **作者:** Haosong Peng; Hao Li; Yalun Dai; Yushi Lan; Yihang Luo; Tianyu Qi; Zhengshen Zhang; Yufeng Zhan; Junfei Zhang; Wenchao Xu; Ziwei Liu
>
> **备注:** Project Page: https://livioni.github.io/OmniVGGT-official/
>
> **摘要:** General 3D foundation models have started to lead the trend of unifying diverse vision tasks, yet most assume RGB-only inputs and ignore readily available geometric cues (e.g., camera intrinsics, poses, and depth maps). To address this issue, we introduce OmniVGGT, a novel framework that can effectively benefit from an arbitrary number of auxiliary geometric modalities during both training and inference. In our framework, a GeoAdapter is proposed to encode depth and camera intrinsics/extrinsics into a spatial foundation model. It employs zero-initialized convolutions to progressively inject geometric information without disrupting the foundation model's representation space. This design ensures stable optimization with negligible overhead, maintaining inference speed comparable to VGGT even with multiple additional inputs. Additionally, a stochastic multimodal fusion regimen is proposed, which randomly samples modality subsets per instance during training. This enables an arbitrary number of modality inputs during testing and promotes learning robust spatial representations instead of overfitting to auxiliary cues. Comprehensive experiments on monocular/multi-view depth estimation, multi-view stereo, and camera pose estimation demonstrate that OmniVGGT outperforms prior methods with auxiliary inputs and achieves state-of-the-art results even with RGB-only input. To further highlight its practical utility, we integrated OmniVGGT into vision-language-action (VLA) models. The enhanced VLA model by OmniVGGT not only outperforms the vanilla point-cloud-based baseline on mainstream benchmarks, but also effectively leverages accessible auxiliary inputs to achieve consistent gains on robotic tasks.
>
---
#### [replaced 010] NOCTIS: Novel Object Cyclic Threshold based Instance Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.01463v3](https://arxiv.org/pdf/2507.01463v3)**

> **作者:** Max Gandyra; Alessandro Santonicola; Michael Beetz
>
> **备注:** 9 pages, 3 figures, 5 tables, CVPR 2026 preprint
>
> **摘要:** Instance segmentation of novel objects instances in RGB images, given some example images for each object, is a well known problem in computer vision. Designing a model general enough to be employed for all kinds of novel objects without (re-) training has proven to be a difficult task. To handle this, we present a new training-free framework, called: Novel Object Cyclic Threshold based Instance Segmentation (NOCTIS). NOCTIS integrates two pre-trained models: Grounded-SAM 2 for object proposals with precise bounding boxes and corresponding segmentation masks; and DINOv2 for robust class and patch embeddings, due to its zero-shot capabilities. Internally, the proposal-object matching is realized by determining an object matching score based on the similarity of the class embeddings and the average maximum similarity of the patch embeddings with a new cyclic thresholding (CT) mechanism that mitigates unstable matches caused by repetitive textures or visually similar patterns. Beyond CT, NOCTIS introduces: (i) an appearance score that is unaffected by object selection bias; (ii) the usage of the average confidence of the proposals' bounding box and mask as a scoring component; and (iii) an RGB-only pipeline that performs even better than RGB-D ones. We empirically show that NOCTIS, without further training/fine tuning, outperforms the best RGB and RGB-D methods regarding the mean AP score on the seven core datasets of the BOP 2023 challenge for the "Model-based 2D segmentation of unseen objects" task.
>
---
#### [replaced 011] CSGaze: Context-aware Social Gaze Prediction
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.05955v2](https://arxiv.org/pdf/2511.05955v2)**

> **作者:** Surbhi Madan; Shreya Ghosh; Ramanathan Subramanian; Abhinav Dhall; Tom Gedeon
>
> **摘要:** A person's gaze offers valuable insights into their focus of attention, level of social engagement, and confidence. In this work, we investigate how contextual cues combined with visual scene and facial information can be effectively utilized to predict and interpret social gaze patterns during conversational interactions. We introduce CSGaze, a context aware multimodal approach that leverages facial, scene information as complementary inputs to enhance social gaze pattern prediction from multi-person images. The model also incorporates a fine-grained attention mechanism centered on the principal speaker, which helps in better modeling social gaze dynamics. Experimental results show that CSGaze performs competitively with state-of-the-art methods on GP-Static, UCO-LAEO and AVA-LAEO. Our findings highlight the role of contextual cues in improving social gaze prediction. Additionally, we provide initial explainability through generated attention scores, offering insights into the model's decision-making process. We also demonstrate our model's generalizability by testing our model on open set datasets that demonstrating its robustness across diverse scenarios.
>
---
#### [replaced 012] Active Contour Models Driven by Hyperbolic Mean Curvature Flow for Image Segmentation
- **分类: cs.CV; math.AP**

- **链接: [https://arxiv.org/pdf/2506.06712v2](https://arxiv.org/pdf/2506.06712v2)**

> **作者:** Saiyu Hu; Chunlei He; Jianfeng Zhang; Dexing Kong; Shoujun Huang
>
> **摘要:** Parabolic mean curvature flow-driven active contour models (PMCF-ACMs) are widely used for image segmentation, yet they suffer severe degradation under high-intensity noise because gradient-descent evolutions exhibit the well-known zig-zag phenomenon. To overcome this drawback, we propose hyperbolic mean curvature flow-driven ACMs (HMCF-ACMs). This novel framework incorporates an adjustable acceleration field to autonomously regulate curve evolution smoothness, providing dual degrees of freedom for adaptive selection of both initial contours and velocity fields. We rigorously prove that HMCF-ACMs are normal flows and establish their numerical equivalence to wave equations through a level set formulation with signed distance functions. An efficient numerical scheme combining spectral discretization and optimized temporal integration is developed to solve the governing equations, and its stability condition is derived through Fourier analysis. Extensive experiments on natural and medical images validate that HMCF-ACMs achieve superior performance under high-noise conditions, demonstrating reduced parameter sensitivity, enhanced noise robustness, and improved segmentation accuracy compared to PMCF-ACMs.
>
---
#### [replaced 013] DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2305.19112v2](https://arxiv.org/pdf/2305.19112v2)**

> **作者:** Ibrahim Ethem Hamamci; Sezgin Er; Omer Faruk Durugol; Gulsade Rabia Cakmak; Ezequiel de la Rosa; Enis Simsar; Atif Emre Yuksel; Sadullah Gultekin; Serife Damla Ozdemir; Kaiyuan Yang; Mehmet Berke Isler; Mustafa Salih Gucez; Shenxiao Mei; Chenglong Ma; Feihong Shen; Kaidi Shen; Huikai Wu; Han Wu; Lanzhuju Mei; Zhiming Cui; Niels van Nistelrooij; Khalid El Ghoul; Steven Kempers; Tong Xi; Shankeeth Vinayahalingam; Kyoungyeon Choi; Jaewon Shin; Eunyi Lyou; Lanshan He; Yusheng Liu; Lisheng Wang; Tudor Dascalu; Shaqayeq Ramezanzade; Azam Bakhshandeh; Lars Bjørndal; Bulat Ibragimov; Hongwei Bran Li; Sarthak Pati; Bernd Stadlinger; Albert Mehl; Mehmet Kemal Ozdemir; Mustafa Gundogar; Bjoern Menze
>
> **摘要:** Panoramic X-rays are frequently used in dentistry for treatment planning, but their interpretation can be both time-consuming and prone to error. Artificial intelligence (AI) has the potential to aid in the analysis of these X-rays, thereby improving the accuracy of dental diagnoses and treatment plans. Nevertheless, designing automated algorithms for this purpose poses significant challenges, mainly due to the scarcity of annotated data and variations in anatomical structure. To address these issues, we organized the Dental Enumeration and Diagnosis on Panoramic X-rays Challenge (DENTEX) in association with the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) in 2023. This challenge aims to promote the development of algorithms for multi-label detection of abnormal teeth, using three types of hierarchically annotated data: partially annotated quadrant data, partially annotated quadrant-enumeration data, and fully annotated quadrant-enumeration-diagnosis data, inclusive of four different diagnoses. In this paper, we present a comprehensive analysis of the methods and results from the challenge. Our findings reveal that top performers succeeded through diverse, specialized strategies, from segmentation-guided pipelines to highly-engineered single-stage detectors, using advanced Transformer and diffusion models. These strategies significantly outperformed traditional approaches, particularly for the challenging tasks of tooth enumeration and subtle disease classification. By dissecting the architectural choices that drove success, this paper provides key insights for future development of AI-powered tools that can offer more precise and efficient diagnosis and treatment planning in dentistry. The evaluation code and datasets can be accessed at https://github.com/ibrahimethemhamamci/DENTEX
>
---
#### [replaced 014] Physics informed Transformer-VAE for biophysical parameter estimation: PROSAIL model inversion in Sentinel-2 imagery
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.10387v2](https://arxiv.org/pdf/2511.10387v2)**

> **作者:** Prince Mensah; Pelumi Victor Aderinto; Ibrahim Salihu Yusuf; Arnu Pretorius
>
> **备注:** My co-authors say some specific changes has to be made first
>
> **摘要:** Accurate retrieval of vegetation biophysical variables from satellite imagery is crucial for ecosystem monitoring and agricultural management. In this work, we propose a physics-informed Transformer-VAE architecture to invert the PROSAIL radiative transfer model for simultaneous estimation of key canopy parameters from Sentinel-2 data. Unlike previous hybrid approaches that require real satellite images for self-supevised training. Our model is trained exclusively on simulated data, yet achieves performance on par with state-of-the-art methods that utilize real imagery. The Transformer-VAE incorporates the PROSAIL model as a differentiable physical decoder, ensuring that inferred latent variables correspond to physically plausible leaf and canopy properties. We demonstrate retrieval of leaf area index (LAI) and canopy chlorophyll content (CCC) on real-world field datasets (FRM4Veg and BelSAR) with accuracy comparable to models trained with real Sentinel-2 data. Our method requires no in-situ labels or calibration on real images, offering a cost-effective and self-supervised solution for global vegetation monitoring. The proposed approach illustrates how integrating physical models with advanced deep networks can improve the inversion of RTMs, opening new prospects for large-scale, physically-constrained remote sensing of vegetation traits.
>
---
#### [replaced 015] Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2510.16926v3](https://arxiv.org/pdf/2510.16926v3)**

> **作者:** Chenxu Li; Zhicai Wang; Yuan Sheng; Xingyu Zhu; Yanbin Hao; Xiang Wang
>
> **备注:** 23 pages
>
> **摘要:** Multimodal Large Language Models (MLLMs) increasingly support dynamic image resolutions. However, current evaluation paradigms primarily assess semantic performance, overlooking the critical question of resolution robustness - whether performance remains stable across varying input resolutions. To address this gap, we introduce \textbf{Res-Bench}, a comprehensive benchmark comprising 14,400 samples across 12 resolution levels and six core capability dimensions. We designed a novel evaluation framework that goes beyond traditional accuracy metrics to capture performance stability. This framework introduces multiple robustness metrics: Spearman's correlation for assessing resolution-performance trends, and Absolute/Relative Continuous Error (ACE/RCE) for measuring performance volatility. Using these metrics, we conducted a large-scale evaluation of leading MLLMs. Our analysis encompasses: (1) model-centric and task-centric robustness examination, (2) investigation of preprocessing strategies including padding and super-resolution, and (3) exploration of fine-tuning for stability enhancement.
>
---
#### [replaced 016] MRT: Learning Compact Representations with Mixed RWKV-Transformer for Extreme Image Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06717v2](https://arxiv.org/pdf/2511.06717v2)**

> **作者:** Han Liu; Hengyu Man; Xingtao Wang; Wenrui Li; Debin Zhao
>
> **摘要:** Recent advances in extreme image compression have revealed that mapping pixel data into highly compact latent representations can significantly improve coding efficiency. However, most existing methods compress images into 2-D latent spaces via convolutional neural networks (CNNs) or Swin Transformers, which tend to retain substantial spatial redundancy, thereby limiting overall compression performance. In this paper, we propose a novel Mixed RWKV-Transformer (MRT) architecture that encodes images into more compact 1-D latent representations by synergistically integrating the complementary strengths of linear-attention-based RWKV and self-attention-based Transformer models. Specifically, MRT partitions each image into fixed-size windows, utilizing RWKV modules to capture global dependencies across windows and Transformer blocks to model local redundancies within each window. The hierarchical attention mechanism enables more efficient and compact representation learning in the 1-D domain. To further enhance compression efficiency, we introduce a dedicated RWKV Compression Model (RCM) tailored to the structure characteristics of the intermediate 1-D latent features in MRT. Extensive experiments on standard image compression benchmarks validate the effectiveness of our approach. The proposed MRT framework consistently achieves superior reconstruction quality at bitrates below 0.02 bits per pixel (bpp). Quantitative results based on the DISTS metric show that MRT significantly outperforms the state-of-the-art 2-D architecture GLC, achieving bitrate savings of 43.75%, 30.59% on the Kodak and CLIC2020 test datasets, respectively.
>
---
#### [replaced 017] Contrastive Integrated Gradients: A Feature Attribution-Based Method for Explaining Whole Slide Image Classification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.08464v2](https://arxiv.org/pdf/2511.08464v2)**

> **作者:** Anh Mai Vu; Tuan L. Vo; Ngoc Lam Quang Bui; Nam Nguyen Le Binh; Akash Awasthi; Huy Quoc Vo; Thanh-Huy Nguyen; Zhu Han; Chandra Mohan; Hien Van Nguyen
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Interpretability is essential in Whole Slide Image (WSI) analysis for computational pathology, where understanding model predictions helps build trust in AI-assisted diagnostics. While Integrated Gradients (IG) and related attribution methods have shown promise, applying them directly to WSIs introduces challenges due to their high-resolution nature. These methods capture model decision patterns but may overlook class-discriminative signals that are crucial for distinguishing between tumor subtypes. In this work, we introduce Contrastive Integrated Gradients (CIG), a novel attribution method that enhances interpretability by computing contrastive gradients in logit space. First, CIG highlights class-discriminative regions by comparing feature importance relative to a reference class, offering sharper differentiation between tumor and non-tumor areas. Second, CIG satisfies the axioms of integrated attribution, ensuring consistency and theoretical soundness. Third, we propose two attribution quality metrics, MIL-AIC and MIL-SIC, which measure how predictive information and model confidence evolve with access to salient regions, particularly under weak supervision. We validate CIG across three datasets spanning distinct cancer types: CAMELYON16 (breast cancer metastasis in lymph nodes), TCGA-RCC (renal cell carcinoma), and TCGA-Lung (lung cancer). Experimental results demonstrate that CIG yields more informative attributions both quantitatively, using MIL-AIC and MIL-SIC, and qualitatively, through visualizations that align closely with ground truth tumor regions, underscoring its potential for interpretable and trustworthy WSI-based diagnostics
>
---
#### [replaced 018] Adaptive LiDAR Scanning: Harnessing Temporal Cues for Efficient 3D Object Detection via Multi-Modal Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01562v2](https://arxiv.org/pdf/2508.01562v2)**

> **作者:** Sara Shoouri; Morteza Tavakoli Taba; Hun-Seok Kim
>
> **备注:** Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026
>
> **摘要:** Multi-sensor fusion using LiDAR and RGB cameras significantly enhances 3D object detection task. However, conventional LiDAR sensors perform dense, stateless scans, ignoring the strong temporal continuity in real-world scenes. This leads to substantial sensing redundancy and excessive power consumption, limiting their practicality on resource-constrained platforms. To address this inefficiency, we propose a predictive, history-aware adaptive scanning framework that anticipates informative regions of interest (ROI) based on past observations. Our approach introduces a lightweight predictor network that distills historical spatial and temporal contexts into refined query embeddings. These embeddings guide a differentiable Mask Generator network, which leverages Gumbel-Softmax sampling to produce binary masks identifying critical ROIs for the upcoming frame. Our method significantly reduces unnecessary data acquisition by concentrating dense LiDAR scanning only within these ROIs and sparsely sampling elsewhere. Experiments on nuScenes and Lyft benchmarks demonstrate that our adaptive scanning strategy reduces LiDAR energy consumption by over 65% while maintaining competitive or even superior 3D object detection performance compared to traditional LiDAR-camera fusion methods with dense LiDAR scanning.
>
---
#### [replaced 019] UI2Code^N: A Visual Language Model for Test-Time Scalable Interactive UI-to-Code Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08195v2](https://arxiv.org/pdf/2511.08195v2)**

> **作者:** Zhen Yang; Wenyi Hong; Mingde Xu; Xinyue Fan; Weihan Wang; Jiele Cheng; Xiaotao Gu; Jie Tang
>
> **备注:** 24 pages
>
> **摘要:** User interface (UI) programming is a core yet highly complex part of modern software development. Recent advances in visual language models (VLMs) highlight the potential of automatic UI coding, but current approaches face two key limitations: multimodal coding capabilities remain underdeveloped, and single-turn paradigms make little use of iterative visual feedback. We address these challenges with an interactive UI-to-code paradigm that better reflects real-world workflows and raises the upper bound of achievable performance. Under this paradigm, we present UI2Code$^\text{N}$, a visual language model trained through staged pretraining, fine-tuning, and reinforcement learning to achieve foundational improvements in multimodal coding. The model unifies three key capabilities: UI-to-code generation, UI editing, and UI polishing. We further explore test-time scaling for interactive generation, enabling systematic use of multi-turn feedback. Experiments on UI-to-code and UI polishing benchmarks show that UI2Code$^\text{N}$ establishes a new state of the art among open-source models and achieves performance comparable to leading closed-source models such as Claude-4-Sonnet and GPT-5. Our code and models are available at https://github.com/zai-org/UI2Code_N.
>
---
#### [replaced 020] StreamDiT: Real-Time Streaming Text-to-Video Generation
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.03745v3](https://arxiv.org/pdf/2507.03745v3)**

> **作者:** Akio Kodaira; Tingbo Hou; Ji Hou; Markos Georgopoulos; Felix Juefei-Xu; Masayoshi Tomizuka; Yue Zhao
>
> **摘要:** Recently, great progress has been achieved in text-to-video (T2V) generation by scaling transformer-based diffusion models to billions of parameters, which can generate high-quality videos. However, existing models typically produce only short clips offline, restricting their use cases in interactive and real-time applications. This paper addresses these challenges by proposing StreamDiT, a streaming video generation model. StreamDiT training is based on flow matching by adding a moving buffer. We design mixed training with different partitioning schemes of buffered frames to boost both content consistency and visual quality. StreamDiT modeling is based on adaLN DiT with varying time embedding and window attention. To practice the proposed method, we train a StreamDiT model with 4B parameters. In addition, we propose a multistep distillation method tailored for StreamDiT. Sampling distillation is performed in each segment of a chosen partitioning scheme. After distillation, the total number of function evaluations (NFEs) is reduced to the number of chunks in a buffer. Finally, our distilled model reaches real-time performance at 16 FPS on one GPU, which can generate video streams at 512p resolution. We evaluate our method through both quantitative metrics and human evaluation. Our model enables real-time applications, e.g. streaming generation, interactive generation, and video-to-video. We provide video results and more examples in our project website: https://cumulo-autumn.github.io/StreamDiT/
>
---
#### [replaced 021] UHKD: A Unified Framework for Heterogeneous Knowledge Distillation via Frequency-Domain Representations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.24116v2](https://arxiv.org/pdf/2510.24116v2)**

> **作者:** Fengming Yu; Haiwei Pan; Kejia Zhang; Jian Guan; Haiying Jiang
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Knowledge distillation (KD) is an effective model compression technique that transfers knowledge from a high-performance teacher to a lightweight student, reducing computational and storage costs while maintaining competitive accuracy. However, most existing KD methods are tailored for homogeneous models and perform poorly in heterogeneous settings, particularly when intermediate features are involved. Semantic discrepancies across architectures hinder effective use of intermediate representations from the teacher model, while prior heterogeneous KD studies mainly focus on the logits space, underutilizing rich semantic information in intermediate layers. To address this, Unified Heterogeneous Knowledge Distillation (UHKD) is proposed, a framework that leverages intermediate features in the frequency domain for cross-architecture transfer. Frequency-domain representations are leveraged to capture global semantic knowledge and mitigate representational discrepancies between heterogeneous teacher-student pairs. Specifically, a Feature Transformation Module (FTM) generates compact frequency-domain representations of teacher features, while a learnable Feature Alignment Module (FAM) projects student features and aligns them via multi-level matching. Training is guided by a joint objective combining mean squared error on intermediate features with Kullback-Leibler divergence on logits. Extensive experiments on CIFAR-100 and ImageNet-1K demonstrate the effectiveness of the proposed approach, achieving maximum gains of 5.59% and 0.83% over the latest heterogeneous distillation method on the two datasets, respectively. Code will be released soon.
>
---
#### [replaced 022] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.10376v2](https://arxiv.org/pdf/2511.10376v2)**

> **作者:** Xun Huang; Shijia Zhao; Yunxiang Wang; Xin Lu; Wanfa Zhang; Rongsheng Qu; Weixin Li; Yunhong Wang; Chenglu Wen
>
> **备注:** 10 pages
>
> **摘要:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relation
>
---
#### [replaced 023] MS-Occ: Multi-Stage LiDAR-Camera Fusion for 3D Semantic Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.15888v2](https://arxiv.org/pdf/2504.15888v2)**

> **作者:** Zhiqiang Wei; Lianqing Zheng; Jianan Liu; Tao Huang; Qing-Long Han; Wenwen Zhang; Fengdeng Zhang
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Accurate 3D semantic occupancy perception is essential for autonomous driving in complex environments with diverse and irregular objects. While vision-centric methods suffer from geometric inaccuracies, LiDAR-based approaches often lack rich semantic information. To address these limitations, MS-Occ, a novel multi-stage LiDAR-camera fusion framework which includes middle-stage fusion and late-stage fusion, is proposed, integrating LiDAR's geometric fidelity with camera-based semantic richness via hierarchical cross-modal fusion. The framework introduces innovations at two critical stages: (1) In the middle-stage feature fusion, the Gaussian-Geo module leverages Gaussian kernel rendering on sparse LiDAR depth maps to enhance 2D image features with dense geometric priors, and the Semantic-Aware module enriches LiDAR voxels with semantic context via deformable cross-attention; (2) In the late-stage voxel fusion, the Adaptive Fusion (AF) module dynamically balances voxel features across modalities, while the High Classification Confidence Voxel Fusion (HCCVF) module resolves semantic inconsistencies using self-attention-based refinement. Experiments on two large-scale benchmarks demonstrate state-of-the-art performance. On nuScenes-OpenOccupancy, MS-Occ achieves an Intersection over Union (IoU) of 32.1% and a mean IoU (mIoU) of 25.3%, surpassing the state-of-the-art by +0.7% IoU and +2.4% mIoU. Furthermore, on the SemanticKITTI benchmark, our method achieves a new state-of-the-art mIoU of 24.08%, robustly validating its generalization capabilities.Ablation studies further confirm the effectiveness of each individual module, highlighting substantial improvements in the perception of small objects and reinforcing the practical value of MS-Occ for safety-critical autonomous driving scenarios.
>
---
#### [replaced 024] Concept Retrieval -- What and How?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.07058v3](https://arxiv.org/pdf/2510.07058v3)**

> **作者:** Ori Nizan; Oren Shrout; Ayellet Tal
>
> **摘要:** A concept may reflect either a concrete or abstract idea. Given an input image, this paper seeks to retrieve other images that share its central concepts, capturing aspects of the underlying narrative. This goes beyond conventional retrieval or clustering methods, which emphasize visual or semantic similarity. We formally define the problem, outline key requirements, and introduce appropriate evaluation metrics. We propose a novel approach grounded in two key observations: (1) While each neighbor in the embedding space typically shares at least one concept with the query, not all neighbors necessarily share the same concept with one another. (2) Modeling this neighborhood with a bimodal Gaussian distribution uncovers meaningful structure that facilitates concept identification. Qualitative, quantitative, and human evaluations confirm the effectiveness of our approach. See the package on PyPI: https://pypi.org/project/coret/
>
---
#### [replaced 025] A COCO-Formatted Instance-Level Dataset for Plasmodium Falciparum Detection in Giemsa-Stained Blood Smears
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.18483v2](https://arxiv.org/pdf/2507.18483v2)**

> **作者:** Frauke Wilm; Luis Carlos Rivera Monroy; Mathias Öttl; Lukas Mürdter; Leonid Mill; Andreas Maier
>
> **备注:** 7 pages, 4 figures, 2 tables, accepted at MICCAI 2025 Open Data
>
> **摘要:** Accurate detection of Plasmodium falciparum in Giemsa-stained blood smears is an essential component of reliable malaria diagnosis, especially in developing countries. Deep learning-based object detection methods have demonstrated strong potential for automated Malaria diagnosis, but their adoption is limited by the scarcity of datasets with detailed instance-level annotations. In this work, we present an enhanced version of the publicly available NIH malaria dataset, with detailed bounding box annotations in COCO format to support object detection training. We validated the revised annotations by training a Faster R-CNN model to detect infected and non-infected red blood cells, as well as white blood cells. Cross-validation on the original dataset yielded F1 scores of up to 0.88 for infected cell detection. These results underscore the importance of annotation volume and consistency, and demonstrate that automated annotation refinement combined with targeted manual correction can produce training data of sufficient quality for robust detection performance. The updated annotations set is publicly available via Zenodo: https://doi.org/10.5281/zenodo.17514694
>
---
#### [replaced 026] Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.03404v2](https://arxiv.org/pdf/2508.03404v2)**

> **作者:** Xinlei Yu; Chengming Xu; Zhangquan Chen; Yudong Zhang; Shilin Lu; Cheng Yang; Jiangning Zhang; Shuicheng Yan; Xiaobin Hu
>
> **摘要:** The dominant paradigm of monolithic scaling in Vision-Language Models (VLMs) is failing for understanding and reasoning in documents, yielding diminishing returns as it struggles with the inherent need of this domain for document-based procedural reasoning, cognitive complexity, and factual accuracy. To this end, we introduce MACT, a Multi-Agent Collaboration framework with agent-wise adaptive Test-time scaling that pioneers a paradigm shift to procedural scaling, adapting dynamically to the functional entities of visual documents understanding and reasoning. MACT decomposes the visual document processing flow into four specialized agents, i.e., planning, execution, judgment, and answer, to resolve cognitive overload and introduce a critical self-correction loop for factual grounding. This collaborative architecture is amplified by an agent-wise adaptive test-time scaling strategy that intelligently allocates computational resources based on the complexity and redundancy of each functionality. Evaluated on multiple visual document understanding benchmarks, MACT achieves superior performance with a smaller parameter scale, adapting effectively to various document scenarios without compromising its general or mathematical reasoning capabilities. The three variants of MACT consistently attain top-three average performance rankings, with average performance enhancements of 9.9-11.5% over the base models. The source code will be released publicly.
>
---
#### [replaced 027] FlowLensing: Simulating Gravitational Lensing with Flow Matching
- **分类: astro-ph.IM; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.07878v3](https://arxiv.org/pdf/2510.07878v3)**

> **作者:** Hamees Sayed; Pranath Reddy; Michael W. Toomey; Sergei Gleyzer
>
> **备注:** 6 pages, 2 figures, 3 tables
>
> **摘要:** Gravitational lensing is one of the most powerful probes of dark matter, yet creating high-fidelity lensed images at scale remains a bottleneck. Existing tools rely on ray-tracing or forward-modeling pipelines that, while precise, are prohibitively slow. We introduce FlowLensing, a Diffusion Transformer-based compact and efficient flow-matching model for strong gravitational lensing simulation. FlowLensing operates in both discrete and continuous regimes, handling classes such as different dark matter models as well as continuous model parameters ensuring physical consistency. By enabling scalable simulations, our model can advance dark matter studies, specifically for probing dark matter substructure in cosmological surveys. We find that our model achieves a speedup of over 200$\times$ compared to classical simulators for intensive dark matter models, with high fidelity and low inference latency. FlowLensing enables rapid, scalable, and physically consistent image synthesis, offering a practical alternative to traditional forward-modeling pipelines.
>
---
#### [replaced 028] Adaptive Pareto-Optimal Token Merging for Edge Transformer Models in Semantic Communication
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2509.09168v2](https://arxiv.org/pdf/2509.09168v2)**

> **作者:** Omar Erak; Omar Alhussein; Hatem Abou-Zeid; Mehdi Bennis
>
> **备注:** Accepted for presentation in IEEE Globecom 2025
>
> **摘要:** Large-scale transformer models have emerged as a powerful tool for semantic communication systems, enabling edge devices to extract rich representations for robust inference across noisy wireless channels. However, their substantial computational demands remain a major barrier to practical deployment in resource-constrained 6G networks. In this paper, we present a training-free framework for adaptive token merging in pretrained vision transformers to jointly reduce inference time and transmission resource usage. We formulate the selection of per-layer merging proportions as a multi-objective optimization problem to balance accuracy and computational cost. We employ Gaussian process-based Bayesian optimization to construct a Pareto frontier of optimal configurations, enabling flexible runtime adaptation to dynamic application requirements and channel conditions. Extensive experiments demonstrate that our method consistently outperforms other baselines and achieves significant reductions in floating-point operations while maintaining competitive accuracy across a wide range of signal-to-noise ratio (SNR) conditions. Additional results highlight the effectiveness of adaptive policies that adjust merging aggressiveness in response to channel quality, providing a practical mechanism to trade off latency and semantic fidelity on demand. These findings establish a scalable and efficient approach for deploying transformer-based semantic communication in future edge intelligence systems.
>
---
#### [replaced 029] Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.08015v2](https://arxiv.org/pdf/2511.08015v2)**

> **作者:** Jian Wang; Lijun He; Yixing Yong; Haixia Bi; Fan Li
>
> **备注:** Accepted by the AAAI 2026 (Main Track)
>
> **摘要:** Modern autonomous driving (AD) systems leverage 3D object detection to perceive foreground objects in 3D environments for subsequent prediction and planning. Visual 3D detection based on RGB cameras provides a cost-effective solution compared to the LiDAR paradigm. While achieving promising detection accuracy, current deep neural network-based models remain highly susceptible to adversarial examples. The underlying safety concerns motivate us to investigate realistic adversarial attacks in AD scenarios. Previous work has demonstrated the feasibility of placing adversarial posters on the road surface to induce hallucinations in the detector. However, the unnatural appearance of the posters makes them easily noticeable by humans, and their fixed content can be readily targeted and defended. To address these limitations, we propose the AdvRoad to generate diverse road-style adversarial posters. The adversaries have naturalistic appearances resembling the road surface while compromising the detector to perceive non-existent objects at the attack locations. We employ a two-stage approach, termed Road-Style Adversary Generation and Scenario-Associated Adaptation, to maximize the attack effectiveness on the input scene while ensuring the natural appearance of the poster, allowing the attack to be carried out stealthily without drawing human attention. Extensive experiments show that AdvRoad generalizes well to different detectors, scenes, and spoofing locations. Moreover, physical attacks further demonstrate the practical threats in real-world environments.
>
---
#### [replaced 030] Concept-as-Tree: A Controllable Synthetic Data Framework Makes Stronger Personalized VLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.12999v3](https://arxiv.org/pdf/2503.12999v3)**

> **作者:** Ruichuan An; Kai Zeng; Ming Lu; Sihan Yang; Renrui Zhang; Huitong Ji; Hao Liang; Wentao Zhang
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated exceptional performance in various multi-modal tasks. Recently, there has been an increasing interest in improving the personalization capabilities of VLMs. To better integrate user-provided concepts into VLMs, many methods use positive and negative samples to fine-tune these models. However, the scarcity of user-provided positive samples and the low quality of retrieved negative samples pose challenges for existing techniques. To reveal the relationship between sample and model performance, we systematically investigate the amount and diversity impact of positive and negative samples (easy and hard) on VLM personalization tasks. Based on the detailed analysis, we introduce Concept-as-Tree (CaT), which represents a concept as a tree structure, thereby enabling the data generation of positive and negative samples with varying difficulty and diversity, and can be easily extended to multi-concept scenarios. With a well-designed data filtering strategy, our CaT framework can ensure the quality of generated data, constituting a powerful pipeline. We perform thorough experiments with various VLM personalization baselines to assess the effectiveness of the pipeline, alleviating the lack of positive samples and the low quality of negative samples. Our results demonstrate that CaT equipped with the proposed data filter significantly enhances the capabilities of VLMs across personalization benchmarks. To the best of our knowledge, this work is the first controllable synthetic data pipeline for VLM personalization. The code will be released.
>
---
#### [replaced 031] FreDFT: Frequency Domain Fusion Transformer for Visible-Infrared Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10046v2](https://arxiv.org/pdf/2511.10046v2)**

> **作者:** Wencong Wu; Xiuwei Zhang; Hanlin Yin; Shun Dai; Hongxi Zhang; Yanning Zhang
>
> **摘要:** Visible-infrared object detection has gained sufficient attention due to its detection performance in low light, fog, and rain conditions. However, visible and infrared modalities captured by different sensors exist the information imbalance problem in complex scenarios, which can cause inadequate cross-modal fusion, resulting in degraded detection performance. \textcolor{red}{Furthermore, most existing methods use transformers in the spatial domain to capture complementary features, ignoring the advantages of developing frequency domain transformers to mine complementary information.} To solve these weaknesses, we propose a frequency domain fusion transformer, called FreDFT, for visible-infrared object detection. The proposed approach employs a novel multimodal frequency domain attention (MFDA) to mine complementary information between modalities and a frequency domain feed-forward layer (FDFFL) via a mixed-scale frequency feature fusion strategy is designed to better enhance multimodal features. To eliminate the imbalance of multimodal information, a cross-modal global modeling module (CGMM) is constructed to perform pixel-wise inter-modal feature interaction in a spatial and channel manner. Moreover, a local feature enhancement module (LFEM) is developed to strengthen multimodal local feature representation and promote multimodal feature fusion by using various convolution layers and applying a channel shuffle. Extensive experimental results have verified that our proposed FreDFT achieves excellent performance on multiple public datasets compared with other state-of-the-art methods. The code of our FreDFT is linked at https://github.com/WenCongWu/FreDFT.
>
---
#### [replaced 032] NeuS-QA: Grounding Long-Form Video Understanding in Temporal Logic and Neuro-Symbolic Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.18041v2](https://arxiv.org/pdf/2509.18041v2)**

> **作者:** Sahil Shah; S P Sharan; Harsh Goel; Minkyu Choi; Mustafa Munir; Manvik Pasula; Radu Marculescu; Sandeep Chinchali
>
> **摘要:** While vision-language models (VLMs) excel at tasks involving single images or short videos, they still struggle with Long Video Question Answering (LVQA) due to its demand for complex multi-step temporal reasoning. Vanilla approaches, which simply sample frames uniformly and feed them to a VLM along with the question, incur significant token overhead. This forces aggressive downsampling of long videos, causing models to miss fine-grained visual structure, subtle event transitions, and key temporal cues. Recent works attempt to overcome these limitations through heuristic approaches; however, they lack explicit mechanisms for encoding temporal relationships and fail to provide any formal guarantees that the sampled context actually encodes the compositional or causal logic required by the question. To address these foundational gaps, we introduce NeuS-QA, a training-free, plug-and-play neuro-symbolic pipeline for LVQA. NeuS-QA first translates a natural language question into a logic specification that models the temporal relationship between frame-level events. Next, we construct a video automaton to model the video's frame-by-frame event progression, and finally employ model checking to compare the automaton against the specification to identify all video segments that satisfy the question's logical requirements. Only these logic-verified segments are submitted to the VLM, thus improving interpretability, reducing hallucinations, and enabling compositional reasoning without modifying or fine-tuning the model. Experiments on the LongVideoBench and CinePile LVQA benchmarks show that NeuS-QA significantly improves performance by over 10%, particularly on questions involving event ordering, causality, and multi-step reasoning. We open-source our code at https://utaustin-swarmlab.github.io/NeuS-QA/.
>
---
#### [replaced 033] A filtering scheme for confocal laser endomicroscopy (CLE)-video sequences for self-supervised learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.00098v2](https://arxiv.org/pdf/2511.00098v2)**

> **作者:** Nils Porsche; Flurin Müller-Diesing; Sweta Banerjee; Miguel Goncalves; Marc Aubreville
>
> **摘要:** Confocal laser endomicroscopy (CLE) is a non-invasive, real-time imaging modality that can be used for in-situ, in-vivo imaging and the microstructural analysis of mucous structures. The diagnosis using CLE is, however, complicated by images being hard to interpret for non-experienced physicians. Utilizing machine learning as an augmentative tool would hence be beneficial, but is complicated by the shortage of histopathology-correlated CLE imaging sequences with respect to the plurality of patterns in this domain, leading to overfitting of machine learning models. To overcome this, self-supervised learning (SSL) can be employed on larger unlabeled datasets. CLE is a video-based modality with high inter-frame correlation, leading to a non-stratified data distribution for SSL training. In this work, we propose a filter functionality on CLE video sequences to reduce the dataset redundancy in SSL training and improve SSL training convergence and training efficiency. We use four state-of-the-art baseline networks and a SSL teacher-student network with a vision transformer small backbone for the evaluation. These networks were evaluated on downstream tasks for a sinonasal tumor dataset and a squamous cell carcinoma of the skin dataset. On both datasets, we found the highest test accuracy on the filtered SSL-pretrained model, with 67.48% and 73.52%, both considerably outperforming their non-SSL baselines. Our results show that SSL is an effective method for CLE pretraining. Further, we show that our proposed CLE video filter can be utilized to improve training efficiency in self-supervised scenarios, resulting in a reduction of 67% in training time.
>
---
#### [replaced 034] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01603v3](https://arxiv.org/pdf/2508.01603v3)**

> **作者:** Yiheng Li; Zichang Tan; Zhen Lei; Xu Zhou; Yang Yang
>
> **备注:** under review, codes: https://github.com/liyih/IAPL
>
> **摘要:** In AI-generated image detection, current cutting-edge methods typically adapt pre-trained foundation models through partial-parameter fine-tuning. However, these approaches often struggle to generalize to forgeries from unseen generators, as the fine-tuned models capture only limited patterns from training data and fail to reflect the evolving traits of new ones. To overcome this limitation, we propose Image-Adaptive Prompt Learning (IAPL), a novel paradigm that dynamically adjusts the prompts fed into the encoder according to each testing image, rather than fixing them after training. This design significantly enhances robustness and adaptability to diverse forged images. The dynamic prompts integrate conditional information with test-time adaptive tokens through a lightweight learnable scaling factor. The conditional information is produced by a Conditional Information Learner, which leverages CNN-based feature extractors to model both forgery-specific and general conditions. The test-time adaptive tokens are optimized during inference on a single sample by enforcing prediction consistency across multiple views, ensuring that the parameters align with the current image. For the final decision, the optimal input with the highest prediction confidence is selected. Extensive experiments show that IAPL achieves state-of-the-art performance, with mean accuracies of 95.61% and 96.7% on the widely used UniversalFakeDetect and GenImage datasets, respectively. Codes and weights will be released on https://github.com/liyih/IAPL.
>
---
#### [replaced 035] An Empirical Study on Improving SimCLR's Nonlinear Projection Head using Pretrained Autoencoder Embeddings
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2408.14514v2](https://arxiv.org/pdf/2408.14514v2)**

> **作者:** Andreas Schliebitz; Heiko Tapken; Martin Atzmueller
>
> **备注:** 7 pages, 1 figure, accepted for publication (ICTAI 2025)
>
> **摘要:** This paper focuses on improving the effectiveness of the standard 2-layer MLP projection head featured in the SimCLR framework through the use of pretrained autoencoder embeddings. Given a contrastive learning task with a largely unlabeled image classification dataset, we first train a shallow autoencoder architecture and extract its compressed representations contained in the encoder's embedding layer. After freezing the weights within this pretrained layer, we use it as a drop-in replacement for the input layer of SimCLR's default projector. Additionally, we also apply further architectural changes to the projector by decreasing its width and changing its activation function. The different projection heads are then used to contrastively train and evaluate a feature extractor following the SimCLR protocol. Our experiments indicate that using a pretrained autoencoder embedding in the projector can not only increase classification accuracy by up to 2.9% or 1.7% on average, but can also significantly decrease the dimensionality of the projection space. Our results also suggest, that using the sigmoid and tanh activation functions within the projector can outperform ReLU in terms of peak and average classification accuracy. All experiments involving our pretrained projectors are conducted with frozen embeddings, since our test results indicate an advantage compared to using their non-frozen counterparts.
>
---
#### [replaced 036] Synthetic Object Compositions for Scalable and Accurate Learning in Detection, Segmentation, and Grounding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.09110v2](https://arxiv.org/pdf/2510.09110v2)**

> **作者:** Weikai Huang; Jieyu Zhang; Taoyang Jia; Chenhao Zheng; Ziqi Gao; Jae Sung Park; Winson Han; Ranjay Krishna
>
> **备注:** Project website: https://github.com/weikaih04/Synthetic-Detection-Segmentation-Grounding-Data
>
> **摘要:** Visual grouping -- operationalized through tasks such as instance segmentation, visual grounding, and object detection -- enables applications ranging from robotic perception to photo editing. These fundamental problems in computer vision are powered by large-scale, painstakingly annotated datasets. Despite their impact, these datasets are costly to build, biased in coverage, and difficult to scale. Synthetic datasets offer a promising alternative but struggle with flexibility, accuracy, and compositional diversity. We introduce Synthetic Object Compositions (SOC), an accurate and scalable data synthesis pipeline via a novel object-centric composition strategy. It composes high-quality synthetic object segments into new images using 3D geometric layout augmentation and camera configuration augmentation with generative harmonization and mask-area-weighted blending, yielding accurate and diverse masks, boxes, and referring expressions. Models trained on just 100K of our synthetic images outperform those trained on larger real datasets (GRIT 20M, V3Det 200K) and synthetic pipelines (Copy-Paste, X-Paste, SynGround, SegGen) by +24-36% -- achieving +10.9 AP on LVIS and +8.4 NAcc on gRefCOCO. Beyond the general open-vocabulary setup, SOC also enables controllable dataset construction for different use cases and boosts performance in both low-data and closed-vocabulary scenarios. Augmenting LVIS and COCO with synthetic object segments delivers strong performance across different real-data scales and yields even greater improvements under extremely limited real-data conditions, including +6.59 AP on a 1% COCO data setup. Furthermore, this controllability enables targeted data generation for intra-class referring, a diagnostic grounding task we propose that requires fine-grained attribute discrimination.
>
---
#### [replaced 037] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.10634v2](https://arxiv.org/pdf/2506.10634v2)**

> **作者:** Francisco Caetano; Christiaan Viviers; Peter H. N. De With; Fons van der Sommen
>
> **备注:** AAAI 2026
>
> **摘要:** Flow Matching has emerged as a powerful framework for learning continuous transformations between distributions, enabling high-fidelity generative modeling. This work introduces Symmetrical Flow Matching (SymmFlow), a new formulation that unifies semantic segmentation, classification, and image generation within a single model. Using a symmetric learning objective, SymmFlow models forward and reverse transformations jointly, ensuring bi-directional consistency, while preserving sufficient entropy for generative diversity. A new training objective is introduced to explicitly retain semantic information across flows, featuring efficient sampling while preserving semantic structure, allowing for one-step segmentation and classification without iterative refinement. Unlike previous approaches that impose strict one-to-one mapping between masks and images, SymmFlow generalizes to flexible conditioning, supporting both pixel-level and image-level class labels. Experimental results on various benchmarks demonstrate that SymmFlow achieves state-of-the-art performance on semantic image synthesis, obtaining FID scores of 11.9 on CelebAMask-HQ and 7.0 on COCO-Stuff with only 25 inference steps. Additionally, it delivers competitive results on semantic segmentation and shows promising capabilities in classification tasks.
>
---
#### [replaced 038] RodEpil: A Video Dataset of Laboratory Rodents for Seizure Detection and Benchmark Evaluation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10431v2](https://arxiv.org/pdf/2511.10431v2)**

> **作者:** Daniele Perlo; Vladimir Despotovic; Selma Boudissa; Sang-Yoon Kim; Petr V. Nazarov; Yanrong Zhang; Max Wintermark; Olivier Keunen
>
> **摘要:** We introduce a curated video dataset of laboratory rodents for automatic detection of convulsive events. The dataset contains short (10~s) top-down and side-view video clips of individual rodents, labeled at clip level as normal activity or seizure. It includes 10,101 negative samples and 2,952 positive samples collected from 19 subjects. We describe the data curation, annotation protocol and preprocessing pipeline, and report baseline experiments using a transformer-based video classifier (TimeSformer). Experiments employ five-fold cross-validation with strict subject-wise partitioning to prevent data leakage (no subject appears in more than one fold). Results show that the TimeSformer architecture enables discrimination between seizure and normal activity with an average F1-score of 97%. The dataset and baseline code are publicly released to support reproducible research on non-invasive, video-based monitoring in preclinical epilepsy research. RodEpil Dataset access - DOI: 10.5281/zenodo.17601357
>
---
#### [replaced 039] GreatSplicing: A Semantically Rich Splicing Dataset
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2310.10070v3](https://arxiv.org/pdf/2310.10070v3)**

> **作者:** Jiaming Liang; Yuwan Xue; Haowei Liu; Zhenqi Dai; Yu Liao; Rui Wang; Weihao Jiang; Yaping Liu; Zhikun Chen; Guoxiao Liu; Bo Liu; Xiuli Bi
>
> **备注:** This version updates the author list and author order, and incorporates changes to the content
>
> **摘要:** In existing splicing forgery datasets, the insufficient semantic variety of spliced regions causes trained detection models to overfit semantic features rather than learn genuine splicing traces. Meanwhile, the lack of a reasonable benchmark dataset has led to inconsistent experimental settings across existing detection methods. To address these issues, we propose GreatSplicing, a manually created, large-scale, high-quality splicing dataset. GreatSplicing comprises 5,000 spliced images and covers spliced regions across 335 distinct semantic categories, enabling detection models to learn splicing traces more effectively. Empirical results show that detection models trained on GreatSplicing achieve low misidentification rates and stronger cross-dataset generalization compared to existing datasets. GreatSplicing is now publicly available for research purposes at the following link.
>
---
#### [replaced 040] Efficient Bayer-Domain Video Computer Vision with Fast Motion Estimation and Learned Perception Residual
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05990v3](https://arxiv.org/pdf/2508.05990v3)**

> **作者:** Haichao Wang; Jiangtao Wen; Yuxing Han
>
> **摘要:** Video computer vision systems face substantial computational burdens arising from two fundamental challenges: eliminating unnecessary processing and reducing temporal redundancy in back-end inference while maintaining accuracy with minimal extra computation. To address these issues, we propose an efficient video computer vision framework that jointly optimizes both the front end and back end of the pipeline. On the front end, we remove the traditional image signal processor (ISP) and feed Bayer raw measurements directly into Bayer-domain vision models, avoiding costly human-oriented ISP operations. On the back end, we introduce a fast and highly parallel motion estimation algorithm that extracts inter-frame temporal correspondence to avoid redundant computation. To mitigate artifacts caused by motion inaccuracies, we further employ lightweight perception residual networks that directly learn perception-level residuals and refine the propagated features. Experiments across multiple models and tasks demonstrate that our system achieves substantial acceleration with only minor performance degradation.
>
---
#### [replaced 041] Enhancing Video Inpainting with Aligned Frame Interval Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21461v2](https://arxiv.org/pdf/2510.21461v2)**

> **作者:** Ming Xie; Junqiu Yu; Qiaole Dong; Xiangyang Xue; Yanwei Fu
>
> **备注:** 15 pages
>
> **摘要:** Recent image-to-video (I2V) based video inpainting methods have made significant strides by leveraging single-image priors and modeling temporal consistency across masked frames. Nevertheless, these methods suffer from severe content degradation within video chunks. Furthermore, the absence of a robust frame alignment scheme compromises intra-chunk and inter-chunk spatiotemporal stability, resulting in insufficient control over the entire video. To address these limitations, we propose VidPivot, a novel framework that decouples video inpainting into two sub-tasks: multi-frame consistent image inpainting and masked area motion propagation. Our approach introduces frame interval priors as spatiotemporal cues to guide the inpainting process. To enhance cross-frame coherence, we design a FrameProp Module that implements a frame content propagation strategy, diffusing reference frame content into subsequent frames via a splicing mechanism. Additionally, a dedicated context controller encodes these coherent frame priors into the I2V generative backbone, effectively serving as soft constrain to suppress content distortion during generation. Extensive evaluations demonstrate that VidPivot achieves competitive performance across diverse benchmarks and generalizes well to different video inpainting scenarios.
>
---
#### [replaced 042] MOSABench: Multi-Object Sentiment Analysis Benchmark for Evaluating Multimodal Large Language Models Understanding of Complex Image
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2412.00060v2](https://arxiv.org/pdf/2412.00060v2)**

> **作者:** Shezheng Song; Chengxiang He; Shan Zhao; Chengyu Wang; Qian Wan; Tianwei Yan; Meng Wang
>
> **摘要:** Multimodal large language models (MLLMs) have shown remarkable progress in high-level semantic tasks such as visual question answering, image captioning, and emotion recognition. However, despite advancements, there remains a lack of standardized benchmarks for evaluating MLLMs performance in multi-object sentiment analysis, a key task in semantic understanding. To address this gap, we introduce MOSABench, a novel evaluation dataset designed specifically for multi-object sentiment analysis. MOSABench includes approximately 1,000 images with multiple objects, requiring MLLMs to independently assess the sentiment of each object, thereby reflecting real-world complexities. Key innovations in MOSABench include distance-based target annotation, post-processing for evaluation to standardize outputs, and an improved scoring mechanism. Our experiments reveal notable limitations in current MLLMs: while some models, like mPLUG-owl and Qwen-VL2, demonstrate effective attention to sentiment-relevant features, others exhibit scattered focus and performance declines, especially as the spatial distance between objects increases. This research underscores the need for MLLMs to enhance accuracy in complex, multi-object sentiment analysis tasks and establishes MOSABench as a foundational tool for advancing sentiment analysis capabilities in MLLMs.
>
---
#### [replaced 043] Leveraging NTPs for Efficient Hallucination Detection in VLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.20379v2](https://arxiv.org/pdf/2509.20379v2)**

> **作者:** Ofir Azachi; Kfir Eliyahu; Eyal El Ani; Rom Himelstein; Roi Reichart; Yuval Pinter; Nitay Calderon
>
> **备注:** Accepted to The First Workshop on Confabulation, Hallucinations, & Overgeneration in Multilingual & Precision-critical Setting - AACL-IJCNLP2025
>
> **摘要:** Hallucinations of vision-language models (VLMs), which are misalignments between visual content and generated text, undermine the reliability of VLMs. One common approach for detecting them employs the same VLM, or a different one, to assess generated outputs. This process is computationally intensive and increases model latency. In this paper, we explore an efficient on-the-fly method for hallucination detection by training traditional ML models over signals based on the VLM's next-token probabilities (NTPs). NTPs provide a direct quantification of model uncertainty. We hypothesize that high uncertainty (i.e., a low NTP value) is strongly associated with hallucinations. To test this, we introduce a dataset of 1,400 human-annotated statements derived from VLM-generated content, each labeled as hallucinated or not, and use it to test our NTP-based lightweight method. Our results demonstrate that NTP-based features are valuable predictors of hallucinations, enabling fast and simple ML models to achieve performance comparable to that of strong VLMs. Furthermore, augmenting these NTPs with linguistic NTPs, computed by feeding only the generated text back into the VLM, enhances hallucination detection performance. Finally, integrating hallucination prediction scores from VLMs into the NTP-based models led to better performance than using either VLMs or NTPs alone. We hope this study paves the way for simple, lightweight solutions that enhance the reliability of VLMs.
>
---
#### [replaced 044] Axis-Aligned Document Dewarping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.15000v2](https://arxiv.org/pdf/2507.15000v2)**

> **作者:** Chaoyun Wang; I-Chao Shen; Takeo Igarashi; Caigui Jiang
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Document dewarping is crucial for many applications. However, existing learning-based methods rely heavily on supervised regression with annotated data without fully leveraging the inherent geometric properties of physical documents. Our key insight is that a well-dewarped document is defined by its axis-aligned feature lines. This property aligns with the inherent axis-aligned nature of the discrete grid geometry in planar documents. Harnessing this property, we introduce three synergistic contributions: for the training phase, we propose an axis-aligned geometric constraint to enhance document dewarping; for the inference phase, we propose an axis alignment preprocessing strategy to reduce the dewarping difficulty; and for the evaluation phase, we introduce a new metric, Axis-Aligned Distortion (AAD), that not only incorporates geometric meaning and aligns with human visual perception but also demonstrates greater robustness. As a result, our method achieves state-of-the-art performance on multiple existing benchmarks, improving the AAD metric by 18.2% to 34.5%. The code is publicly available at https://github.com/chaoyunwang/AADD.
>
---
#### [replaced 045] TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2508.19257v3](https://arxiv.org/pdf/2508.19257v3)**

> **作者:** Chenghao Liu; Jiachen Zhang; Chengxuan Li; Zhimu Zhou; Shixin Wu; Songfang Huang; Huiling Duan
>
> **备注:** Accepted to AAAI 2026. Camera-ready version
>
> **摘要:** Vision-Language-Action (VLA) models process visual inputs independently at each timestep, discarding valuable temporal information inherent in robotic manipulation tasks. This frame-by-frame processing makes models vulnerable to visual noise while ignoring the substantial coherence between consecutive frames in manipulation sequences. We propose Temporal Token Fusion (TTF), a training-free approach that intelligently integrates historical and current visual representations to enhance VLA inference quality. Our method employs dual-dimension detection combining efficient grayscale pixel difference analysis with attention-based semantic relevance assessment, enabling selective temporal token fusion through hard fusion strategies and keyframe anchoring to prevent error accumulation. Comprehensive experiments across LIBERO, SimplerEnv, and real robot tasks demonstrate consistent improvements: 4.0 percentage points average on LIBERO (72.4\% vs 68.4\% baseline), cross-environment validation on SimplerEnv (4.8\% relative improvement), and 8.7\% relative improvement on real robot tasks. Our approach proves model-agnostic, working across OpenVLA and VLA-Cache architectures. Notably, TTF reveals that selective Query matrix reuse in attention mechanisms enhances rather than compromises performance, suggesting promising directions for direct KQV matrix reuse strategies that achieve computational acceleration while improving task success rates.
>
---
#### [replaced 046] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09977v2](https://arxiv.org/pdf/2511.09977v2)**

> **作者:** Yongdeuk Seo; Hyun-seok Min; Sungchul Choi
>
> **备注:** Accepted to AAAI 2026 Workshop (Artificial Intelligence with Biased or Scarce Data)
>
> **摘要:** Scene Text Editing (STE) is the task of modifying text content in an image while preserving its visual style, such as font, color, and background. While recent diffusion-based approaches have shown improvements in visual quality, key limitations remain: lack of support for low-resource languages, domain gap between synthetic and real data, and the absence of appropriate metrics for evaluating text style preservation. To address these challenges, we propose STELLAR (Scene Text Editor for Low-resource LAnguages and Real-world data). STELLAR enables reliable multilingual editing through a language-adaptive glyph encoder and a multi-stage training strategy that first pre-trains on synthetic data and then fine-tunes on real images. We also construct a new dataset, STIPLAR(Scene Text Image Pairs of Low-resource lAnguages and Real-world data), for training and evaluation. Furthermore, we propose Text Appearance Similarity (TAS), a novel metric that assesses style preservation by independently measuring font, color, and background similarity, enabling robust evaluation even without ground truth. Experimental results demonstrate that STELLAR outperforms state-of-the-art models in visual consistency and recognition accuracy, achieving an average TAS improvement of 2.2% across languages over the baselines.
>
---
#### [replaced 047] FlexPara: Flexible Neural Surface Parameterization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.19210v2](https://arxiv.org/pdf/2504.19210v2)**

> **作者:** Yuming Zhao; Qijian Zhang; Junhui Hou; Jiazhi Xia; Wenping Wang; Ying He
>
> **摘要:** Surface parameterization is a fundamental geometry processing task, laying the foundations for the visual presentation of 3D assets and numerous downstream shape analysis scenarios. Conventional parameterization approaches demand high-quality mesh triangulation and are restricted to certain simple topologies unless additional surface cutting and decomposition are provided. In practice, the optimal configurations (e.g., type of parameterization domains, distribution of cutting seams, number of mapping charts) may vary drastically with different surface structures and task characteristics, thus requiring more flexible and controllable processing pipelines. To this end, this paper introduces FlexPara, an unsupervised neural optimization framework to achieve both global and multi-chart surface parameterizations by establishing point-wise mappings between 3D surface points and adaptively-deformed 2D UV coordinates. We ingeniously design and combine a series of geometrically-interpretable sub-networks, with specific functionalities of cutting, deforming, unwrapping, and wrapping, to construct a bi-directional cycle mapping framework for global parameterization without the need for manually specified cutting seams. Furthermore, we construct a multi-chart parameterization framework with adaptively-learned chart assignment. Extensive experiments demonstrate the universality, superiority, and inspiring potential of our neural surface parameterization paradigm. The code will be publicly available at https://github.com/AidenZhao/FlexPara
>
---
#### [replaced 048] SGLP: A Similarity Guided Fast Layer Partition Pruning for Compressing Large Deep Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2410.14720v2](https://arxiv.org/pdf/2410.14720v2)**

> **作者:** Yuqi Li; Yao Lu; Junhao Dong; Zeyu Dong; Chuanguang Yang; Xin Yin; Yihao Chen; Jianping Gou; Yingli Tian; Tingwen Huang
>
> **备注:** 16 pages
>
> **摘要:** Layer pruning has emerged as a potent approach to remove redundant layers in the pre-trained network on the purpose of reducing network size and improve computational efficiency. However, existing layer pruning methods mostly overlook the intrinsic connections and inter-dependencies between different layers within complicated deep neural networks. This oversight can result in pruned models that do not preserve the essential characteristics of the pre-trained network as effectively as desired. To address these limitations, we propose a Similarity-Guided Layer Partition (SGLP) Pruning, a novel pruning framework that exploits representation similarity to guide efficient and informed layer removal for compressing large deep models. Our method begins by employing Centered Kernel Alignment (CKA) to quantify representational similarity between layers, uncovering structural patterns within the network. We then apply Fisher Optimal Segmentation on the similarity matrix to partition the network into semantically coherent layer segments. This segmentation allows pruning decisions to respect layer interdependencies and preserve essential knowledge. Within each segment, we introduce a fine-tuning-free importance evaluation using GradNorm, identifying and removing redundant layers in a targeted, segment-wise manner. Experimental results on both image classification tasks and large language models (LLMs) demonstrate that our proposed SGLP outperforms the state-of-the-art methods in accuracy and efficiency. Our approach achieves significant model compression with minimal performance degradation, making it well-suited for deployment in resource-limited environments.
>
---
#### [replaced 049] Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2311.17663v3](https://arxiv.org/pdf/2311.17663v3)**

> **作者:** Junyi Ma; Xieyuanli Chen; Jiawei Huang; Jingyi Xu; Zhen Luo; Jintao Xu; Weihao Gu; Rui Ai; Hesheng Wang
>
> **备注:** Accepted to CVPR 2024
>
> **摘要:** Understanding how the surrounding environment changes is crucial for performing downstream tasks safely and reliably in autonomous driving applications. Recent occupancy estimation techniques using only camera images as input can provide dense occupancy representations of large-scale scenes based on the current observation. However, they are mostly limited to representing the current 3D space and do not consider the future state of surrounding objects along the time axis. To extend camera-only occupancy estimation into spatiotemporal prediction, we propose Cam4DOcc, a new benchmark for camera-only 4D occupancy forecasting, evaluating the surrounding scene changes in a near future. We build our benchmark based on multiple publicly available datasets, including nuScenes, nuScenes-Occupancy, and Lyft-Level5, which provides sequential occupancy states of general movable and static objects, as well as their 3D backward centripetal flow. To establish this benchmark for future research with comprehensive comparisons, we introduce four baseline types from diverse camera-based perception and prediction implementations, including a static-world occupancy model, voxelization of point cloud prediction, 2D-3D instance-based prediction, and our proposed novel end-to-end 4D occupancy forecasting network. Furthermore, the standardized evaluation protocol for preset multiple tasks is also provided to compare the performance of all the proposed baselines on present and future occupancy estimation with respect to objects of interest in autonomous driving scenarios. The dataset and our implementation of all four baselines in the proposed Cam4DOcc benchmark will be released here: https://github.com/haomo-ai/Cam4DOcc.
>
---
#### [replaced 050] Zero-Shot Temporal Interaction Localization for Egocentric Videos
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2506.03662v4](https://arxiv.org/pdf/2506.03662v4)**

> **作者:** Erhang Zhang; Junyi Ma; Yin-Dong Zheng; Yixuan Zhou; Hesheng Wang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We have released our code and relevant data as open-source at https://github.com/IRMVLab/EgoLoc.
>
---
#### [replaced 051] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06328v2](https://arxiv.org/pdf/2511.06328v2)**

> **作者:** Dingkang Yang; Mingcheng Li; Xuecheng Wu; Zhaoyu Chen; Kaixun Jiang; Keliang Liu; Peng Zhai; Lihua Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multimodal Sentiment Analysis (MSA) aims to predict sentiment from language, acoustic, and visual data in videos. However, imbalanced unimodal performance often leads to suboptimal fused representations. Existing approaches typically adopt fixed primary modality strategies to maximize dominant modality advantages, yet fail to adapt to dynamic variations in modality importance across different samples. Moreover, non-language modalities suffer from sequential redundancy and noise, degrading model performance when they serve as primary inputs. To address these issues, this paper proposes a modality optimization and dynamic primary modality selection framework (MODS). First, a Graph-based Dynamic Sequence Compressor (GDC) is constructed, which employs capsule networks and graph convolution to reduce sequential redundancy in acoustic/visual modalities. Then, we develop a sample-adaptive Primary Modality Selector (MSelector) for dynamic dominance determination. Finally, a Primary-modality-Centric Cross-Attention (PCCA) module is designed to enhance dominant modalities while facilitating cross-modal interaction. Extensive experiments on four benchmark datasets demonstrate that MODS outperforms state-of-the-art methods, achieving superior performance by effectively balancing modality contributions and eliminating redundant noise.
>
---
#### [replaced 052] OccamVTS: Distilling Vision Models to 1% Parameters for Time Series Forecasting
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01727v2](https://arxiv.org/pdf/2508.01727v2)**

> **作者:** Sisuo Lyu; Siru Zhong; Weilin Ruan; Qingxiang Liu; Qingsong Wen; Hui Xiong; Yuxuan Liang
>
> **摘要:** Time series forecasting is fundamental to diverse applications, with recent approaches leverage large vision models (LVMs) to capture temporal patterns through visual representations. We reveal that while vision models enhance forecasting performance, 99% of their parameters are unnecessary for time series tasks. Through cross-modal analysis, we find that time series align with low-level textural features but not high-level semantics, which can impair forecasting accuracy. We propose OccamVTS, a knowledge distillation framework that extracts only the essential 1% of predictive information from LVMs into lightweight networks. Using pre-trained LVMs as privileged teachers, OccamVTS employs pyramid-style feature alignment combined with correlation and feature distillation to transfer beneficial patterns while filtering out semantic noise. Counterintuitively, this aggressive parameter reduction improves accuracy by eliminating overfitting to irrelevant visual features while preserving essential temporal patterns. Extensive experiments across multiple benchmark datasets demonstrate that OccamVTS consistently achieves state-of-the-art performance with only 1% of the original parameters, particularly excelling in few-shot and zero-shot scenarios.
>
---
#### [replaced 053] Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.04370v5](https://arxiv.org/pdf/2405.04370v5)**

> **作者:** Junyi Ma; Jingyi Xu; Xieyuanli Chen; Hesheng Wang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Understanding how humans would behave during hand-object interaction is vital for applications in service robot manipulation and extended reality. To achieve this, some recent works have been proposed to simultaneously forecast hand trajectories and object affordances on human egocentric videos. The joint prediction serves as a comprehensive representation of future hand-object interactions in 2D space, indicating potential human motion and motivation. However, the existing approaches mostly adopt the autoregressive paradigm for unidirectional prediction, which lacks mutual constraints within the holistic future sequence, and accumulates errors along the time axis. Meanwhile, these works basically overlook the effect of camera egomotion on first-person view predictions. To address these limitations, we propose a novel diffusion-based interaction prediction method, namely Diff-IP2D, to forecast future hand trajectories and object affordances concurrently in an iterative non-autoregressive manner. We transform the sequential 2D images into latent feature space and design a denoising diffusion model to predict future latent interaction features conditioned on past ones. Motion features are further integrated into the conditional denoising process to enable Diff-IP2D aware of the camera wearer's dynamics for more accurate interaction prediction. Extensive experiments demonstrate that our method significantly outperforms the state-of-the-art baselines on both the off-the-shelf metrics and our newly proposed evaluation protocol. This highlights the efficacy of leveraging a generative paradigm for 2D hand-object interaction prediction. The code of Diff-IP2D is released as open source at https://github.com/IRMVLab/Diff-IP2D.
>
---
#### [replaced 054] TEyeD: Over 20 million real-world eye images with Pupil, Eyelid, and Iris 2D and 3D Segmentations, 2D and 3D Landmarks, 3D Eyeball, Gaze Vector, and Eye Movement Types
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2102.02115v4](https://arxiv.org/pdf/2102.02115v4)**

> **作者:** Wolfgang Fuhl; Gjergji Kasneci; Enkelejda Kasneci
>
> **备注:** Download: https://es-cloud.cs.uni-tuebingen.de/d/8e2ab8c3fdd444e1a135/?p=%2FTEyeDS&mode=list
>
> **摘要:** We present TEyeD, the world's largest unified public data set of eye images taken with head-mounted devices. TEyeD was acquired with seven different head-mounted eye trackers. Among them, two eye trackers were integrated into virtual reality (VR) or augmented reality (AR) devices. The images in TEyeD were obtained from various tasks, including car rides, simulator rides, outdoor sports activities, and daily indoor activities. The data set includes 2D and 3D landmarks, semantic segmentation, 3D eyeball annotation and the gaze vector and eye movement types for all images. Landmarks and semantic segmentation are provided for the pupil, iris and eyelids. Video lengths vary from a few minutes to several hours. With more than 20 million carefully annotated images, TEyeD provides a unique, coherent resource and a valuable foundation for advancing research in the field of computer vision, eye tracking and gaze estimation in modern VR and AR applications. Download: https://es-cloud.cs.uni-tuebingen.de/d/8e2ab8c3fdd444e1a135/?p=%2FTEyeDS&mode=list
>
---
#### [replaced 055] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.23318v4](https://arxiv.org/pdf/2507.23318v4)**

> **作者:** Jiajun Cao; Qizhe Zhang; Peidong Jia; Xuhui Zhao; Bo Lan; Xiaoan Zhang; Zhuo Li; Xiaobao Wei; Sixiang Chen; Liyun Li; Xianming Liu; Ming Lu; Yang Wang; Shanghang Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated significant potential in complex scene understanding and action reasoning, leading to their increasing adoption in end-to-end autonomous driving systems. However, the long visual tokens of VLA models greatly increase computational costs. Current visual token pruning methods in Vision-Language Models (VLM) rely on either visual token similarity or visual-text attention, but both have shown poor performance in autonomous driving scenarios. Given that human drivers concentrate on relevant foreground areas while driving, we assert that retaining visual tokens containing this foreground information is essential for effective decision-making. Inspired by this, we propose FastDriveVLA, a novel reconstruction-based vision token pruning framework designed specifically for autonomous driving. FastDriveVLA includes a plug-and-play visual token pruner called ReconPruner, which prioritizes foreground information through MAE-style pixel reconstruction. A novel adversarial foreground-background reconstruction strategy is designed to train ReconPruner for the visual encoder of VLA models. Once trained, ReconPruner can be seamlessly applied to different VLA models with the same visual encoder without retraining. To train ReconPruner, we also introduce a large-scale dataset called nuScenes-FG, consisting of 241K image-mask pairs with annotated foreground regions. Our approach achieves state-of-the-art results on the nuScenes open-loop planning benchmark across different pruning ratios.
>
---
#### [replaced 056] Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval
- **分类: cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.10202v2](https://arxiv.org/pdf/2506.10202v2)**

> **作者:** Shubhashis Roy Dipta; Francis Ferraro
>
> **备注:** Accepted in IJCNLP-AACL 2025 (also presented in MAGMAR 2025 at ACL 2025)
>
> **摘要:** Recent approaches have shown impressive proficiency in extracting and leveraging parametric knowledge from Large-Language Models (LLMs) and Vision-Language Models (VLMs). In this work, we consider how we can improve the identification and retrieval of videos related to complex real-world events by automatically extracting latent parametric knowledge about those events. We present Q2E: a Query-to-Event decomposition method for zero-shot multilingual text-to-video retrieval, adaptable across datasets, domains, LLMs, or VLMs. Our approach demonstrates that we can enhance the understanding of otherwise overly simplified human queries by decomposing the query using the knowledge embedded in LLMs and VLMs. We additionally show how to apply our approach to both visual and speech-based inputs. To combine this varied multimodal knowledge, we adopt entropy-based fusion scoring for zero-shot fusion. Through evaluations on two diverse datasets and multiple retrieval metrics, we demonstrate that Q2E outperforms several state-of-the-art baselines. Our evaluation also shows that integrating audio information can significantly improve text-to-video retrieval. We have released code and data for future research.
>
---
#### [replaced 057] MADiff: Motion-Aware Mamba Diffusion Models for Hand Trajectory Prediction on Egocentric Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.02638v2](https://arxiv.org/pdf/2409.02638v2)**

> **作者:** Junyi Ma; Xieyuanli Chen; Wentao Bao; Jingyi Xu; Hesheng Wang
>
> **备注:** Accepted to TPAMI 2025
>
> **摘要:** Understanding human intentions and actions through egocentric videos is important on the path to embodied artificial intelligence. As a branch of egocentric vision techniques, hand trajectory prediction plays a vital role in comprehending human motion patterns, benefiting downstream tasks in extended reality and robot manipulation. However, capturing high-level human intentions consistent with reasonable temporal causality is challenging when only egocentric videos are available. This difficulty is exacerbated under camera egomotion interference and the absence of affordance labels to explicitly guide the optimization of hand waypoint distribution. In this work, we propose a novel hand trajectory prediction method dubbed MADiff, which forecasts future hand waypoints with diffusion models. The devised denoising operation in the latent space is achieved by our proposed motion-aware Mamba, where the camera wearer's egomotion is integrated to achieve motion-driven selective scan (MDSS). To discern the relationship between hands and scenarios without explicit affordance supervision, we leverage a foundation model that fuses visual and language features to capture high-level semantics from video clips. Comprehensive experiments conducted on five public datasets with the existing and our proposed new evaluation metrics demonstrate that MADiff predicts comparably reasonable hand trajectories compared to the state-of-the-art baselines, and achieves real-time performance. We will release our code and pretrained models of MADiff at the project page: https://irmvlab.github.io/madiff.github.io.
>
---
#### [replaced 058] RiverScope: High-Resolution River Masking Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.02451v2](https://arxiv.org/pdf/2509.02451v2)**

> **作者:** Rangel Daroya; Taylor Rowley; Jonathan Flores; Elisa Friedmann; Fiona Bennitt; Heejin An; Travis Simmons; Marissa Jean Hughes; Camryn L Kluetmeier; Solomon Kica; J. Daniel Vélez; Sarah E. Esenther; Thomas E. Howard; Yanqi Ye; Audrey Turcotte; Colin Gleason; Subhransu Maji
>
> **摘要:** Surface water dynamics play a critical role in Earth's climate system, influencing ecosystems, agriculture, disaster resilience, and sustainable development. Yet monitoring rivers and surface water at fine spatial and temporal scales remains challenging -- especially for narrow or sediment-rich rivers that are poorly captured by low-resolution satellite data. To address this, we introduce RiverScope, a high-resolution dataset developed through collaboration between computer science and hydrology experts. RiverScope comprises 1,145 high-resolution images (covering 2,577 square kilometers) with expert-labeled river and surface water masks, requiring over 100 hours of manual annotation. Each image is co-registered with Sentinel-2, SWOT, and the SWOT River Database (SWORD), enabling the evaluation of cost-accuracy trade-offs across sensors -- a key consideration for operational water monitoring. We also establish the first global, high-resolution benchmark for river width estimation, achieving a median error of 7.2 meters -- significantly outperforming existing satellite-derived methods. We extensively evaluate deep networks across multiple architectures (e.g., CNNs and transformers), pretraining strategies (e.g., supervised and self-supervised), and training datasets (e.g., ImageNet and satellite imagery). Our best-performing models combine the benefits of transfer learning with the use of all the multispectral PlanetScope channels via learned adaptors. RiverScope provides a valuable resource for fine-scale and multi-sensor hydrological modeling, supporting climate adaptation and sustainable water management.
>
---
#### [replaced 059] iTrace: Click-Based Gaze Visualization on the Apple Vision Pro
- **分类: cs.HC; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12268v2](https://arxiv.org/pdf/2508.12268v2)**

> **作者:** Esra Mehmedova; Santiago Berrezueta-Guzman; Stefan Wagner
>
> **备注:** Paper submitted to review
>
> **摘要:** The Apple Vision Pro is equipped with accurate eye-tracking capabilities, yet the privacy restrictions on the device prevent direct access to continuous user gaze data. This study introduces iTrace, a novel application that overcomes these limitations through click-based gaze extraction techniques, including manual methods like a pinch gesture, and automatic approaches utilizing dwell control or a gaming controller. We developed a system with a client-server architecture that captures the gaze coordinates and transforms them into dynamic heatmaps for video and spatial eye tracking. The system can generate individual and averaged heatmaps, enabling analysis of personal and collective attention patterns. To demonstrate its effectiveness and evaluate the usability and performance, a study was conducted with two groups of 10 participants, each testing different clicking methods. The 8BitDo controller achieved higher average data collection rates at 14.22 clicks/s compared to 0.45 clicks/s with dwell control, enabling significantly denser heatmap visualizations. The resulting heatmaps reveal distinct attention patterns, including concentrated focus in lecture videos and broader scanning during problem-solving tasks. By allowing dynamic attention visualization while maintaining a high gaze precision of 91 %, iTrace demonstrates strong potential for a wide range of applications in educational content engagement, environmental design evaluation, marketing analysis, and clinical cognitive assessment. Despite the current gaze data restrictions on the Apple Vision Pro, we encourage developers to use iTrace only in research settings.
>
---
#### [replaced 060] First-Order Error Matters: Accurate Compensation for Quantized Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.11017v2](https://arxiv.org/pdf/2507.11017v2)**

> **作者:** Xingyu Zheng; Haotong Qin; Yuye Li; Haoran Chu; Jiakai Wang; Jinyang Guo; Michele Magno; Xianglong Liu
>
> **备注:** Accepted by AAAI 2026. The code is available at https://github.com/Xingyu-Zheng/FOEM
>
> **摘要:** Post-training quantization (PTQ) offers an efficient approach to compressing large language models (LLMs), significantly reducing memory access and computational costs. Existing compensation-based weight calibration methods often rely on a second-order Taylor expansion to model quantization error, under the assumption that the first-order term is negligible in well-trained full-precision models. However, we reveal that the progressive compensation process introduces accumulated first-order deviations between latent weights and their full-precision counterparts, making this assumption fundamentally flawed. To address this, we propose FOEM, a novel PTQ method that explicitly incorporates first-order gradient terms to improve quantization error compensation. FOEM approximates gradients by performing a first-order Taylor expansion around the pre-quantization weights. This yields an approximation based on the difference between latent and full-precision weights as well as the Hessian matrix. When substituted into the theoretical solution, the formulation eliminates the need to explicitly compute the Hessian, thereby avoiding the high computational cost and limited generalization of backpropagation-based gradient methods. This design introduces only minimal additional computational overhead. Extensive experiments across a wide range of models and benchmarks demonstrate that FOEM consistently outperforms the classical GPTQ method. In 3-bit weight-only quantization, FOEM reduces the perplexity of Llama3-8B by 17.3% and increases the 5-shot MMLU accuracy from 53.8% achieved by GPTAQ to 56.1%. Moreover, FOEM can be seamlessly combined with advanced techniques such as SpinQuant, delivering additional gains under the challenging W4A4KV4 setting and further narrowing the performance gap with full-precision baselines, surpassing existing state-of-the-art methods.
>
---
#### [replaced 061] Unifying Segment Anything in Microscopy with Vision-Language Knowledge
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.10769v2](https://arxiv.org/pdf/2505.10769v2)**

> **作者:** Manyu Li; Ruian He; Zixian Zhang; Chenxi Ma; Weimin Tan; Bo Yan
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Accurate segmentation of regions of interest in biomedical images holds substantial value in image analysis. Although several foundation models for biomedical segmentation have currently achieved excellent performance on certain datasets, they typically demonstrate sub-optimal performance on unseen domain data. We owe the deficiency to lack of vision-language knowledge before segmentation. Multimodal Large Language Models (MLLMs) bring outstanding understanding and reasoning capabilities to multimodal tasks, which inspires us to leverage MLLMs to inject Vision-Language Knowledge (VLK), thereby enabling vision models to demonstrate superior generalization capabilities on cross-domain datasets. In this paper, we propose a novel framework that seamlessly uses MLLMs to guide SAM in learning microscopy cross-domain data, unifying Segment Anything in Microscopy, named uLLSAM. Specifically, we propose the Vision-Language Semantic Alignment (VLSA) module, which injects VLK into Segment Anything Model (SAM). We find that after SAM receives global VLK prompts, its performance improves significantly, but there are deficiencies in boundary contour perception. Therefore, we further propose Semantic Boundary Regularization (SBR) to regularize SAM. Our method achieves performance improvements of 11.8% in SA across 9 in-domain microscopy datasets, achieving state-of-the-art performance. Our method also demonstrates improvements of 9.2% in SA across 10 out-of-domain datasets, exhibiting strong generalization capabilities. Code is available at https://github.com/ieellee/uLLSAM.
>
---
#### [replaced 062] Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination
- **分类: cs.LG; cs.CV; math.OC**

- **链接: [https://arxiv.org/pdf/2311.02960v4](https://arxiv.org/pdf/2311.02960v4)**

> **作者:** Peng Wang; Xiao Li; Can Yaras; Zhihui Zhu; Laura Balzano; Wei Hu; Qing Qu
>
> **备注:** This paper has been accepted for publication in the Journal of Machine Learning Research
>
> **摘要:** Over the past decade, deep learning has proven to be a highly effective tool for learning meaningful features from raw data. However, it remains an open question how deep networks perform hierarchical feature learning across layers. In this work, we attempt to unveil this mystery by investigating the structures of intermediate features. Motivated by our empirical findings that linear layers mimic the roles of deep layers in nonlinear networks for feature learning, we explore how deep linear networks transform input data into output by investigating the output (i.e., features) of each layer after training in the context of multi-class classification problems. Toward this goal, we first define metrics to measure within-class compression and between-class discrimination of intermediate features, respectively. Through theoretical analysis of these two metrics, we show that the evolution of features follows a simple and quantitative pattern from shallow to deep layers when the input data is nearly orthogonal and the network weights are minimum-norm, balanced, and approximate low-rank: Each layer of the linear network progressively compresses within-class features at a geometric rate and discriminates between-class features at a linear rate with respect to the number of layers that data have passed through. To the best of our knowledge, this is the first quantitative characterization of feature evolution in hierarchical representations of deep linear networks. Empirically, our extensive experiments not only validate our theoretical results numerically but also reveal a similar pattern in deep nonlinear networks which aligns well with recent empirical studies. Moreover, we demonstrate the practical implications of our results in transfer learning. Our code is available at https://github.com/Heimine/PNC_DLN.
>
---
#### [replaced 063] The Temporal Trap: Entanglement in Pre-Trained Visual Representations for Visuomotor Policy Learning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2502.03270v3](https://arxiv.org/pdf/2502.03270v3)**

> **作者:** Nikolaos Tsagkas; Andreas Sochopoulos; Duolikun Danier; Chris Xiaoxuan Lu; Oisin Mac Aodha
>
> **备注:** This submission replaces our earlier work "When Pre-trained Visual Representations Fall Short: Limitations in Visuo-Motor Robot Learning." The original paper was split into two studies; this version focuses on temporal entanglement in pre-trained visual representations. The companion paper is "Attentive Feature Aggregation."
>
> **摘要:** The integration of pre-trained visual representations (PVRs) has significantly advanced visuomotor policy learning. However, effectively leveraging these models remains a challenge. We identify temporal entanglement as a critical, inherent issue when using these time-invariant models in sequential decision-making tasks. This entanglement arises because PVRs, optimised for static image understanding, struggle to represent the temporal dependencies crucial for visuomotor control. In this work, we quantify the impact of temporal entanglement, demonstrating a strong correlation between a policy's success rate and the ability of its latent space to capture task-progression cues. Based on these insights, we propose a simple, yet effective disentanglement baseline designed to mitigate temporal entanglement. Our empirical results show that traditional methods aimed at enriching features with temporal components are insufficient on their own, highlighting the necessity of explicitly addressing temporal disentanglement for robust visuomotor policy learning.
>
---
#### [replaced 064] LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10004v2](https://arxiv.org/pdf/2511.10004v2)**

> **作者:** Minjun Kim; Jaeri Lee; Jongjin Kim; Jeongin Yun; Yongmo Kwon; U Kang
>
> **备注:** AAAI 2026
>
> **摘要:** How can we accurately quantize a pre-trained Vision Transformer model? Quantization algorithms compress Vision Transformers (ViTs) into low-bit formats, reducing memory and computation demands with minimal accuracy degradation. However, existing methods rely on uniform precision, ignoring the diverse sensitivity of ViT components to quantization. Metric-based Mixed Precision Quantization (MPQ) is a promising alternative, but previous MPQ methods for ViTs suffer from three major limitations: 1) coarse granularity, 2) mismatch in metric scale across component types, and 3) quantization-unaware bit allocation. In this paper, we propose LampQ (Layer-wise Mixed Precision Quantization for Vision Transformers), an accurate metric-based MPQ method for ViTs to overcome these limitations. LampQ performs layer-wise quantization to achieve both fine-grained control and efficient acceleration, incorporating a type-aware Fisher-based metric to measure sensitivity. Then, LampQ assigns bit-widths optimally through integer linear programming and further updates them iteratively. Extensive experiments show that LampQ provides the state-of-the-art performance in quantizing ViTs pre-trained on various tasks such as image classification, object detection, and zero-shot quantization.
>
---
#### [replaced 065] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2411.16657v4](https://arxiv.org/pdf/2411.16657v4)**

> **作者:** Zun Wang; Jialu Li; Han Lin; Jaehong Yoon; Mohit Bansal
>
> **备注:** AAAI 2026, Project website: https://zunwang1.github.io/DreamRunner
>
> **摘要:** Storytelling video generation (SVG) aims to produce coherent and visually rich multi-scene videos that follow a structured narrative. Existing methods primarily employ LLM for high-level planning to decompose a story into scene-level descriptions, which are then independently generated and stitched together. However, these approaches struggle with generating high-quality videos aligned with the complex single-scene description, as visualizing such complex description involves coherent composition of multiple characters and events, complex motion synthesis and multi-character customization. To address these challenges, we propose DREAMRUNNER, a novel story-to-video generation method: First, we structure the input script using a large language model (LLM) to facilitate both coarse-grained scene planning as well as fine-grained object-level layout planning. Next, DREAMRUNNER presents retrieval-augmented test-time adaptation to capture target motion priors for objects in each scene, supporting diverse motion customization based on retrieved videos, thus facilitating the generation of new videos with complex, scripted motions. Lastly, we propose a novel spatial-temporal region-based 3D attention and prior injection module SR3AI for fine-grained object-motion binding and frame-by-frame spatial-temporal semantic control. We compare DREAMRUNNER with various SVG baselines, demonstrating state-of-the-art performance in character consistency, text alignment, and smooth transitions. Additionally, DREAMRUNNER exhibits strong fine-grained condition-following ability in compositional text-to-video generation, significantly outperforming baselines on T2V-ComBench. Finally, we validate DREAMRUNNER's robust ability to generate multi-object interactions with qualitative examples.
>
---
#### [replaced 066] Latent Motion Profiling for Annotation-free Cardiac Phase Detection in Adult and Fetal Echocardiography Videos
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.05154v2](https://arxiv.org/pdf/2507.05154v2)**

> **作者:** Yingyu Yang; Qianye Yang; Kangning Cui; Can Peng; Elena D'Alberti; Netzahualcoyotl Hernandez-Cruz; Olga Patey; Aris T. Papageorghiou; J. Alison Noble
>
> **摘要:** The identification of cardiac phase is an essential step for analysis and diagnosis of cardiac function. Automatic methods, especially data-driven methods for cardiac phase detection, typically require extensive annotations, which is time-consuming and labor-intensive. In this paper, we present an unsupervised framework for end-diastole (ED) and end-systole (ES) detection through self-supervised learning of latent cardiac motion trajectories from 4-chamber-view echocardiography videos. Our method eliminates the need for manual annotations, including ED and ES indices, segmentation, or volumetric measurements, by training a reconstruction model to encode interpretable spatiotemporal motion patterns. Evaluated on the EchoNet-Dynamic benchmark, the approach achieves mean absolute error (MAE) of 3 frames (58.3 ms) for ED and 2 frames (38.8 ms) for ES detection, matching state-of-the-art supervised methods. Extended to fetal echocardiography, the model demonstrates robust performance with MAE 1.46 frames (20.7 ms) for ED and 1.74 frames (25.3 ms) for ES, despite the fact that the fetal heart model is built using non-standardized heart views due to fetal heart positioning variability. Our results demonstrate the potential of the proposed latent motion trajectory strategy for cardiac phase detection in adult and fetal echocardiography. This work advances unsupervised cardiac motion analysis, offering a scalable solution for clinical populations lacking annotated data. Code will be released at https://github.com/YingyuYyy/CardiacPhase.
>
---
#### [replaced 067] Walk Before You Dance: High-fidelity and Editable Dance Synthesis via Generative Masked Motion Prior
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.04634v2](https://arxiv.org/pdf/2504.04634v2)**

> **作者:** Foram N Shah; Parshwa Shah; Muhammad Usama Saleem; Ekkasit Pinyoanuntapong; Pu Wang; Hongfei Xue; Ahmed Helmy
>
> **摘要:** Recent advances in dance generation have enabled the automatic synthesis of 3D dance motions. However, existing methods still face significant challenges in simultaneously achieving high realism, precise dance-music synchronization, diverse motion expression, and physical plausibility. To address these limitations, we propose a novel approach that leverages a generative masked text-to-motion model as a distribution prior to learn a probabilistic mapping from diverse guidance signals, including music, genre, and pose, into high-quality dance motion sequences. Our framework also supports semantic motion editing, such as motion inpainting and body part modification. Specifically, we introduce a multi-tower masked motion model that integrates a text-conditioned masked motion backbone with two parallel, modality-specific branches: a music-guidance tower and a pose-guidance tower. The model is trained using synchronized and progressive masked training, which allows effective infusion of the pretrained text-to-motion prior into the dance synthesis process while enabling each guidance branch to optimize independently through its own loss function, mitigating gradient interference. During inference, we introduce classifier-free logits guidance and pose-guided token optimization to strengthen the influence of music, genre, and pose signals. Extensive experiments demonstrate that our method sets a new state of the art in dance generation, significantly advancing both the quality and editability over existing approaches. Project Page available at https://foram-s1.github.io/DanceMosaic/
>
---
#### [replaced 068] MoPE: Mixture of Prompt Experts for Parameter-Efficient and Scalable Multimodal Fusion
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2403.10568v4](https://arxiv.org/pdf/2403.10568v4)**

> **作者:** Ruixiang Jiang; Lingbo Liu; Changwen Chen
>
> **备注:** Accepted to IEEE TMM
>
> **摘要:** Despite the demonstrated parameter efficiency of prompt-based fusion, its limited adaptivity and expressiveness hinder its effectiveness for multimodal applications at scale. In this paper, we present the first comprehensive study addressing these limitations. Our key motivation is to ``divide and conquer'' the vanilla prompt, traditionally shared across all instances, by generating instance-specific prompts. Specifically, we propose the Mixture of Prompt Experts (MoPE), a framework that significantly enhances prompt adaptivity and expressiveness by dynamically generating instance-specific prompts. MoPE leverages multimodal pairings as additional evidence, allowing the model to adaptively select optimal prompts tailored to each individual instance. Unlike traditional prompt-fusion methods, which encounter scalability bottlenecks when optimizing long unified prompts, MoPE maintains fixed prompt length while effectively scaling the number of specialized experts. Moreover, we investigate regularization terms to encourage expert specialization, resulting in highly adaptive and interpretable prompting. MoPE fundamentally changes the scaling dynamic, unlocking greater expressiveness and adaptability to complex multimodal relationships, enabling the model to selectively attend to task-relevant sub-sequences based on instance-specific multimodal input. Extensive experiments across six multimodal datasets spanning four modalities demonstrate state-of-the-art performance for multimodal fusion, matching or surpassing the performance of fine-tuning while requiring only 0.8% of the trainable parameters. Code is available: https://github.com/songrise/MoPE.
>
---
#### [replaced 069] YOLO-SAT: A Data-based and Model-based Enhanced YOLOv12 Model for Desert Waste Detection and Classification
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.03888v2](https://arxiv.org/pdf/2511.03888v2)**

> **作者:** Abdulmumin Sa'ad; Sulaimon Oyeniyi Adebayo
>
> **备注:** 8 pages
>
> **摘要:** The global waste crisis is escalating, with solid waste generation expected to increase tremendously in the coming years. Traditional waste collection methods, particularly in remote or harsh environments like deserts, are labor-intensive, inefficient, and often hazardous. Recent advances in computer vision and deep learning have opened the door to automated waste detection systems, yet most research focuses on urban environments and recyclable materials, overlooking organic and hazardous waste and underexplored terrains such as deserts. In this work, we propose YOLO-SAT, an enhanced real-time object detection framework based on a pruned, lightweight version of YOLOv12 integrated with Self-Adversarial Training (SAT) and specialized data augmentation strategies. Using the DroneTrashNet dataset, we demonstrate significant improvements in precision, recall, and mean average precision (mAP), while achieving low latency and compact model size suitable for deployment on resource-constrained aerial drones. Benchmarking YOLO-SAT against state-of-the-art lightweight YOLO variants further highlights its optimal balance of accuracy and efficiency. Our results validate the effectiveness of combining data-centric and model-centric enhancements for robust, real-time waste detection in desert environments.
>
---
#### [replaced 070] Adaptive Parametric Activation: Unifying and Generalising Activation Functions Across Tasks
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2407.08567v3](https://arxiv.org/pdf/2407.08567v3)**

> **作者:** Konstantinos Panagiotis Alexandridis; Jiankang Deng; Anh Nguyen; Shan Luo
>
> **备注:** Version 2: 19 pages, 7 figures, 13 Tables. Extension of the ECCV2024 oral paper arXiv:2407.08567v2
>
> **摘要:** The activation function plays a crucial role in model optimisation, yet the optimal choice remains unclear. For example, the Sigmoid activation is the de-facto activation in balanced classification tasks, however, in imbalanced classification, it proves inappropriate due to bias towards frequent classes. In this work, we delve deeper in this phenomenon by performing a comprehensive statistical analysis in the classification and intermediate layers of both balanced and imbalanced networks and we empirically show that aligning the activation function with the data distribution, enhances the performance in both balanced and imbalanced tasks. To this end, we propose the Adaptive Parametric Activation (APA) function, a novel and versatile activation function that unifies most common activation functions under a single formula. APA can be applied in both intermediate layers and attention layers, significantly outperforming the state-of-the-art on several imbalanced benchmarks such as ImageNet-LT, iNaturalist2018, Places-LT, CIFAR100-LT and LVIS. Also, we extend APA to a plethora of other tasks such as classification, detection, visual instruction following tasks, image generation and next-text-token prediction benchmarks. APA increases the performance in multiple benchmarks across various model architectures. The code is available at https://github.com/kostas1515/AGLU.
>
---
#### [replaced 071] Generative AI in Map-Making: A Technical Exploration and Its Implications for Cartographers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.18959v2](https://arxiv.org/pdf/2508.18959v2)**

> **作者:** Claudio Affolter; Sidi Wu; Yizi Chen; Lorenz Hurni
>
> **摘要:** Traditional map-making relies heavily on Geographic Information Systems (GIS), requiring domain expertise and being time-consuming, especially for repetitive tasks. Recent advances in generative AI (GenAI), particularly image diffusion models, offer new opportunities for automating and democratizing the map-making process. However, these models struggle with accurate map creation due to limited control over spatial composition and semantic layout. To address this, we integrate vector data to guide map generation in different styles, specified by the textual prompts. Our model is the first to generate accurate maps in controlled styles, and we have integrated it into a web application to improve its usability and accessibility. We conducted a user study with professional cartographers to assess the fidelity of generated maps, the usability of the web application, and the implications of ever-emerging GenAI in map-making. The findings have suggested the potential of our developed application and, more generally, the GenAI models in helping both non-expert users and professionals in creating maps more efficiently. We have also outlined further technical improvements and emphasized the new role of cartographers to advance the paradigm of AI-assisted map-making. The code and pre-trained models are available at https://github.com/claudaff/generative-ai-mapmaking/.
>
---
#### [replaced 072] Curing Semantic Drift: A Dynamic Approach to Grounding Generation in Large Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21509v3](https://arxiv.org/pdf/2506.21509v3)**

> **作者:** Jiahe Chen; Jiaying He; Qiyuan Chen; Qian Shao; Jiahe Ying; Hongxia Xu; Jintai Chen; Jianwei Zheng; Jian Wu
>
> **摘要:** Large Vision-Language Models (LVLMs) face a tug-of-war between powerful linguistic priors and visual evidence, often leading to ``semantic drift'' -- the progressive detachment from visual input that we identify as the root cause of hallucination. While several existing training-free decoding strategies have achieved considerable success, they still suffer from inherent limitations. Many are computationally prohibitive, requiring multiple forward passes through the entire LVLM, while others rely on indirect, heuristic-based proxies that are unreliable correlates for a direct semantic conflict. We propose \textbf{D}ynamic \textbf{L}ogits \textbf{C}alibration (DLC), a novel training-free framework that is the first to cure semantic drift in a direct, dynamic, and efficient manner. At each decoding step, DLC introduces a real-time visual referee that performs a dual-aspect visual alignment check: (1) it assesses the intrinsic visual relevance of a candidate token and (2) its contextual visual coherence. By dynamically balancing these two checks and evaluating them against an adaptive baseline, DLC surgically modulates the output logits to favor grounded tokens. Extensive experiments show DLC significantly outperforms existing methods in mitigating hallucinations while, crucially, maintaining high inference efficiency by avoiding costly multiple LVLM forward passes. Our work presents a powerful and practical solution for building more reliable and visually-grounded LVLMs. Code will be released on https://github.com/JiaheChen2002/DLC.
>
---
#### [replaced 073] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.11777v2](https://arxiv.org/pdf/2506.11777v2)**

> **作者:** Divyanshu Mishra; Mohammadreza Salehi; Pramit Saha; Olga Patey; Aris T. Papageorghiou; Yuki M. Asano; J. Alison Noble
>
> **摘要:** Self-supervised learning (SSL) has achieved major advances in natural images and video understanding, but challenges remain in domains like echocardiography (heart ultrasound) due to subtle anatomical structures, complex temporal dynamics, and the current lack of domain-specific pre-trained models. Existing SSL approaches such as contrastive, masked modeling, and clustering-based methods struggle with high intersample similarity, sensitivity to low PSNR inputs common in ultrasound, or aggressive augmentations that distort clinically relevant features. We present DISCOVR (Distilled Image Supervision for Cross Modal Video Representation), a self-supervised dual branch framework for cardiac ultrasound video representation learning. DISCOVR combines a clustering-based video encoder that models temporal dynamics with an online image encoder that extracts fine-grained spatial semantics. These branches are connected through a semantic cluster distillation loss that transfers anatomical knowledge from the evolving image encoder to the video encoder, enabling temporally coherent representations enriched with fine-grained semantic understanding.Evaluated on six echocardiography datasets spanning fetal, pediatric, and adult populations, DISCOVR outperforms both specialized video anomaly detection methods and state-of-the-art video-SSL baselines in zero-shot and linear probing setups,achieving superior segmentation transfer and strong downstream performance on clinically relevant tasks such as LVEF prediction. Code available at: https://github.com/mdivyanshu97/DISCOVR
>
---
#### [replaced 074] Quantifying the Limits of Segmentation Foundation Models: Modeling Challenges in Segmenting Tree-Like and Low-Contrast Objects
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2412.04243v3](https://arxiv.org/pdf/2412.04243v3)**

> **作者:** Yixin Zhang; Nicholas Konz; Kevin Kramer; Maciej A. Mazurowski
>
> **备注:** Accepted at WACV 2026. Code: https://github.com/mazurowski-lab/SAMFailureMetrics
>
> **摘要:** Image segmentation foundation models (SFMs) like Segment Anything Model (SAM) have achieved impressive zero-shot and interactive segmentation across diverse domains. However, they struggle to segment objects with certain structures, particularly those with dense, tree-like morphology and low textural contrast from their surroundings. These failure modes are crucial for understanding the limitations of SFMs in real-world applications. To systematically study this issue, we introduce interpretable metrics quantifying object tree-likeness and textural separability. On carefully controlled synthetic experiments and real-world datasets, we show that SFM performance (\eg, SAM, SAM 2, HQ-SAM) noticeably correlates with these factors. We attribute these failures to SFMs misinterpreting local structure as global texture, resulting in over-segmentation or difficulty distinguishing objects from similar backgrounds. Notably, targeted fine-tuning fails to resolve this issue, indicating a fundamental limitation. Our study provides the first quantitative framework for modeling the behavior of SFMs on challenging structures, offering interpretable insights into their segmentation capabilities.
>
---
#### [replaced 075] MeshMosaic: Scaling Artist Mesh Generation via Local-to-Global Assembly
- **分类: cs.GR; cs.CG; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.19995v2](https://arxiv.org/pdf/2509.19995v2)**

> **作者:** Rui Xu; Tianyang Xue; Qiujie Dong; Le Wan; Zhe Zhu; Peng Li; Zhiyang Dou; Cheng Lin; Shiqing Xin; Yuan Liu; Wenping Wang; Taku Komura
>
> **备注:** Project is available at: https://xrvitd.github.io/MeshMosaic/index.html
>
> **摘要:** Scaling artist-designed meshes to high triangle numbers remains challenging for autoregressive generative models. Existing transformer-based methods suffer from long-sequence bottlenecks and limited quantization resolution, primarily due to the large number of tokens required and constrained quantization granularity. These issues prevent faithful reproduction of fine geometric details and structured density patterns. We introduce MeshMosaic, a novel local-to-global framework for artist mesh generation that scales to over 100K triangles--substantially surpassing prior methods, which typically handle only around 8K faces. MeshMosaic first segments shapes into patches, generating each patch autoregressively and leveraging shared boundary conditions to promote coherence, symmetry, and seamless connectivity between neighboring regions. This strategy enhances scalability to high-resolution meshes by quantizing patches individually, resulting in more symmetrical and organized mesh density and structure. Extensive experiments across multiple public datasets demonstrate that MeshMosaic significantly outperforms state-of-the-art methods in both geometric fidelity and user preference, supporting superior detail representation and practical mesh generation for real-world applications.
>
---
#### [replaced 076] MILD: Multi-Layer Diffusion Strategy for Complex and Precise Multi-IP Aware Human Erasing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06543v2](https://arxiv.org/pdf/2508.06543v2)**

> **作者:** Jinghan Yu; Junhao Xiao; Zhiyuan Ma; Yue Ma; Kaiqi Liu; Yuhan Wang; Daizong Liu; Xianghao Meng; Jianjun Li
>
> **摘要:** Recent years have witnessed the success of diffusion models in image customization tasks. However, existing mask-guided human erasing methods still struggle in complex scenarios such as human-human occlusion, human-object entanglement, and human-background interference, mainly due to the lack of large-scale multi-instance datasets and effective spatial decoupling to separate foreground from background. To bridge these gaps, we curate the MILD dataset capturing diverse poses, occlusions, and complex multi-instance interactions. We then define the Cross-Domain Attention Gap (CAG), an attention-gap metric to quantify semantic leakage. On top of these, we propose Multi-Layer Diffusion (MILD), which decomposes the generation process into independent denoising pathways, enabling separate reconstruction of each foreground instance and the background. To enhance human-centric understanding, we introduce Human Morphology Guidance, a plug-and-play module that incorporates pose, parsing, and spatial relationships into the diffusion process to improve structural awareness and restoration quality. Additionally, we present Spatially-Modulated Attention, an adaptive mechanism that leverages spatial mask priors to modulate attention across semantic regions, further widening the CAG to effectively minimize boundary artifacts and mitigate semantic leakage. Experiments show that MILD significantly outperforms existing methods. Datasets and code are publicly available at: https://mild-multi-layer-diffusion.github.io/.
>
---
#### [replaced 077] Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.07375v2](https://arxiv.org/pdf/2504.07375v2)**

> **作者:** Junyi Ma; Wentao Bao; Jingyi Xu; Guanzhong Sun; Xieyuanli Chen; Hesheng Wang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Predicting hand motion is critical for understanding human intentions and bridging the action space between human movements and robot manipulations. Existing hand trajectory prediction (HTP) methods forecast the future hand waypoints in 3D space conditioned on past egocentric observations. However, such models are only designed to accommodate 2D egocentric video inputs. There is a lack of awareness of multimodal environmental information from both 2D and 3D observations, hindering the further improvement of 3D HTP performance. In addition, these models overlook the synergy between hand movements and headset camera egomotion, either predicting hand trajectories in isolation or encoding egomotion only from past frames. To address these limitations, we propose novel diffusion models (MMTwin) for multimodal 3D hand trajectory prediction. MMTwin is designed to absorb multimodal information as input encompassing 2D RGB images, 3D point clouds, past hand waypoints, and text prompt. Besides, two latent diffusion models, the egomotion diffusion and the HTP diffusion as twins, are integrated into MMTwin to predict camera egomotion and future hand trajectories concurrently. We propose a novel hybrid Mamba-Transformer module as the denoising model of the HTP diffusion to better fuse multimodal features. The experimental results on three publicly available datasets and our self-recorded data demonstrate that our proposed MMTwin can predict plausible future 3D hand trajectories compared to the state-of-the-art baselines, and generalizes well to unseen environments. The code and pretrained models have been released at https://github.com/IRMVLab/MMTwin.
>
---
#### [replaced 078] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.06271v2](https://arxiv.org/pdf/2506.06271v2)**

> **作者:** Jonathan Schmidt; Simon Giebenhain; Matthias Niessner
>
> **备注:** NeurIPS 2025, Project Page: see https://jonathsch.github.io/becominglit/ , YouTube Video: see https://youtu.be/xPyeIqKdszA
>
> **摘要:** We introduce BecomingLit, a novel method for reconstructing relightable, high-resolution head avatars that can be rendered from novel viewpoints at interactive rates. Therefore, we propose a new low-cost light stage capture setup, tailored specifically towards capturing faces. Using this setup, we collect a novel dataset consisting of diverse multi-view sequences of numerous subjects under varying illumination conditions and facial expressions. By leveraging our new dataset, we introduce a new relightable avatar representation based on 3D Gaussian primitives that we animate with a parametric head model and an expression-dependent dynamics module. We propose a new hybrid neural shading approach, combining a neural diffuse BRDF with an analytical specular term. Our method reconstructs disentangled materials from our dynamic light stage recordings and enables all-frequency relighting of our avatars with both point lights and environment maps. In addition, our avatars can easily be animated and controlled from monocular videos. We validate our approach in extensive experiments on our dataset, where we consistently outperform existing state-of-the-art methods in relighting and reenactment by a significant margin.
>
---
#### [replaced 079] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.14137v2](https://arxiv.org/pdf/2504.14137v2)**

> **作者:** Hangyu Liu; Bo Peng; Pengxiang Ding; Donglin Wang
>
> **备注:** AAAI-26 (Oral)
>
> **摘要:** Compared to single-target adversarial attacks, multi-target attacks have garnered significant attention due to their ability to generate adversarial images for multiple target classes simultaneously. However, existing generative approaches for multi-target attacks primarily encode target labels into one-dimensional tensors, leading to a loss of fine-grained visual information and overfitting to model-specific features during noise generation. To address this gap, we first identify and validate that the semantic feature quality and quantity are critical factors affecting the transferability of targeted attacks: 1) Feature quality refers to the structural and detailed completeness of the implanted target features, as deficiencies may result in the loss of key discriminative information; 2) Feature quantity refers to the spatial sufficiency of the implanted target features, as inadequacy limits the victim model's attention to this feature. Based on these findings, we propose the 2D Tensor-Guided Adversarial Fusion (TGAF) framework, which leverages the powerful generative capabilities of diffusion models to encode target labels into two-dimensional semantic tensors for guiding adversarial noise generation. Additionally, we design a novel masking strategy tailored for the training process, ensuring that parts of the generated noise retain complete semantic information about the target class. Extensive experiments demonstrate that TGAF consistently surpasses state-of-the-art methods across various settings.
>
---
#### [replaced 080] Medverse: A Universal Model for Full-Resolution 3D Medical Image Segmentation, Transformation and Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09232v2](https://arxiv.org/pdf/2509.09232v2)**

> **作者:** Jiesi Hu; Jianfeng Cao; Yanwu Yang; Chenfei Ye; Yixuan Zhang; Hanyang Peng; Ting Ma
>
> **摘要:** In-context learning (ICL) offers a promising paradigm for universal medical image analysis, enabling models to perform diverse image processing tasks without retraining. However, current ICL models for medical imaging remain limited in two critical aspects: they cannot simultaneously achieve high-fidelity predictions and global anatomical understanding, and there is no unified model trained across diverse medical imaging tasks (e.g., segmentation and enhancement) and anatomical regions. As a result, the full potential of ICL in medical imaging remains underexplored. Thus, we present \textbf{Medverse}, a universal ICL model for 3D medical imaging, trained on 22 datasets covering diverse tasks in universal image segmentation, transformation, and enhancement across multiple organs, imaging modalities, and clinical centers. Medverse employs a next-scale autoregressive in-context learning framework that progressively refines predictions from coarse to fine, generating consistent, full-resolution volumetric outputs and enabling multi-scale anatomical awareness. We further propose a blockwise cross-attention module that facilitates long-range interactions between context and target inputs while preserving computational efficiency through spatial sparsity. Medverse is extensively evaluated on a broad collection of held-out datasets covering previously unseen clinical centers, organs, species, and imaging modalities. Results demonstrate that Medverse substantially outperforms existing ICL baselines and establishes a novel paradigm for in-context learning. Code and model weights will be made publicly available. Our model are publicly available at https://github.com/jiesihu/Medverse.
>
---
#### [replaced 081] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.07334v4](https://arxiv.org/pdf/2503.07334v4)**

> **作者:** Xing Xie; Jiawei Liu; Ziyue Lin; Huijie Fan; Zhi Han; Yandong Tang; Liangqiong Qu
>
> **备注:** Accepted by AAAI 2026 Oral
>
> **摘要:** We present Autoregressive Representation Alignment (ARRA), a new training framework that unlocks global-coherent text-to-image generation in autoregressive LLMs without architectural modifications. Different from prior works that require complex architectural redesigns, ARRA aligns LLM's hidden states with visual representations from external visual foundational models via a global visual alignment loss and a hybrid token, <HYBNEXT>. This token enforces dual constraints: local next-token prediction and global semantic distillation, enabling LLMs to implicitly learn spatial and contextual coherence while retaining their original autoregressive paradigm. Extensive experiments validate ARRA's plug-and-play versatility. When training T2I LLMs from scratch, ARRA reduces FID by 16.6% (ImageNet), 12.0% (LAION-COCO) for autoregressive LLMs like LlamaGen, without modifying original architecture and inference mechanism. For training from text-generation-only LLMs, ARRA reduces FID by 25.5% (MIMIC-CXR), 8.8% (DeepEyeNet) for advanced LLMs like Chameleon. For domain adaptation, ARRA aligns general-purpose LLMs with specialized models (e.g., BioMedCLIP), achieving an 18.6% FID reduction over direct fine-tuning on medical imaging (MIMIC-CXR). These results demonstrate that training objective redesign, rather than architectural modifications, can resolve cross-modal global coherence challenges. ARRA offers a complementary paradigm for advancing autoregressive models. The code is available at https://github.com/HKU-HealthAI/ARRA.
>
---
#### [replaced 082] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.16533v2](https://arxiv.org/pdf/2505.16533v2)**

> **作者:** Jiacong Chen; Qingyu Mao; Youneng Bao; Xiandong Meng; Fanyang Meng; Ronggang Wang; Yongsheng Liang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a high-fidelity and efficient paradigm for online free-viewpoint video (FVV) reconstruction, offering viewers rapid responsiveness and immersive experiences. However, existing online methods face challenge in prohibitive storage requirements primarily due to point-wise modeling that fails to exploit the motion properties. To address this limitation, we propose a novel Compact Gaussian Streaming (ComGS) framework, leveraging the locality and consistency of motion in dynamic scene, that models object-consistent Gaussian point motion through keypoint-driven motion representation. By transmitting only the keypoint attributes, this framework provides a more storage-efficient solution. Specifically, we first identify a sparse set of motion-sensitive keypoints localized within motion regions using a viewspace gradient difference strategy. Equipped with these keypoints, we propose an adaptive motion-driven mechanism that predicts a spatial influence field for propagating keypoint motion to neighboring Gaussian points with similar motion. Moreover, ComGS adopts an error-aware correction strategy for key frame reconstruction that selectively refines erroneous regions and mitigates error accumulation without unnecessary overhead. Overall, ComGS achieves a remarkable storage reduction of over 159 X compared to 3DGStream and 14 X compared to the SOTA method QUEEN, while maintaining competitive visual fidelity and rendering speed.
>
---
#### [replaced 083] Self-Diffusion Driven Blind Imaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.27439v2](https://arxiv.org/pdf/2510.27439v2)**

> **作者:** Yanlong Yang; Guanxiong Luo
>
> **摘要:** Optical imaging systems are inherently imperfect due to diffraction limits, lens manufacturing tolerances, assembly misalignment, and other physical constraints. In addition, unavoidable camera shake and object motion further introduce non-ideal degradations during acquisition. These aberrations and motion-induced variations are typically unknown, difficult to measure, and costly to model or calibrate in practice. Blind inverse problems offer a promising direction by jointly estimating both the latent image and the unknown degradation kernel. However, existing approaches often suffer from convergence instability, limited prior expressiveness, and sensitivity to hyperparameters. Inspired by recent advances in self-diffusion, we propose DeblurSDI, a zero-shot, self-supervised blind imaging framework that requires no pre-training. DeblurSDI formulates blind image recovery as an iterative reverse self-diffusion process that begins from pure noise and progressively refines both the sharp image and the blur kernel. Extensive experiments on combined optical aberrations and motion blur demonstrate that DeblurSDI consistently outperforms other methods by a substantial margin.
>
---
#### [replaced 084] EIDSeg: A Pixel-Level Semantic Segmentation Dataset for Post-Earthquake Damage Assessment from Social Media Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06456v2](https://arxiv.org/pdf/2511.06456v2)**

> **作者:** Huili Huang; Chengeng Liu; Danrong Zhang; Shail Patel; Anastasiya Masalava; Sagar Sadak; Parisa Babolhavaeji; WeiHong Low; Max Mahdi Roozbahani; J. David Frost
>
> **备注:** Camera-Ready for AAAI-AISI26
>
> **摘要:** Rapid post-earthquake damage assessment is crucial for rescue and resource planning. Still, existing remote sensing methods depend on costly aerial images, expert labeling, and produce only binary damage maps for early-stage evaluation. Although ground-level images from social networks provide a valuable source to fill this gap, a large pixel-level annotated dataset for this task is still unavailable. We introduce EIDSeg, the first large-scale semantic segmentation dataset specifically for post-earthquake social media imagery. The dataset comprises 3,266 images from nine major earthquakes (2008-2023), annotated across five classes of infrastructure damage: Undamaged Building, Damaged Building, Destroyed Building, Undamaged Road, and Damaged Road. We propose a practical three-phase cross-disciplinary annotation protocol with labeling guidelines that enables consistent segmentation by non-expert annotators, achieving over 70% inter-annotator agreement. We benchmark several state-of-the-art segmentation models, identifying Encoder-only Mask Transformer (EoMT) as the top-performing method with a Mean Intersection over Union (mIoU) of 80.8%. By unlocking social networks' rich ground-level perspective, our work paves the way for a faster, finer-grained damage assessment in the post-earthquake scenario.
>
---
#### [replaced 085] Explicit Multimodal Graph Modeling for Human-Object Interaction Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12554v2](https://arxiv.org/pdf/2509.12554v2)**

> **作者:** Wenxuan Ji; Haichao Shi; Xiao-Yu Zhang
>
> **摘要:** Transformer-based methods have recently become the prevailing approach for Human-Object Interaction (HOI) detection. However, the Transformer architecture does not explicitly model the relational structures inherent in HOI detection, which impedes the recognition of interactions. In contrast, Graph Neural Networks (GNNs) are inherently better suited for this task, as they explicitly model the relationships between human-object pairs. Therefore, in this paper, we propose \textbf{M}ultimodal \textbf{G}raph \textbf{N}etwork \textbf{M}odeling (MGNM) that leverages GNN-based relational structures to enhance HOI detection. Specifically, we design a multimodal graph network framework that explicitly models the HOI task in a four-stage graph structure. Furthermore, we introduce a multi-level feature interaction mechanism within our graph network. This mechanism leverages multi-level visual and language features to enhance information propagation across human-object pairs. Consequently, our proposed MGNM achieves state-of-the-art (SOTA) performance on two widely used benchmarks: HICO-DET and V-COCO. Moreover, when integrated with a more advanced object detector, our method demonstrates a significant performance gain and maintains an effective balance between rare and non-rare classes.
>
---
#### [replaced 086] Towards Formalizing Spuriousness of Biased Datasets Using Partial Information Decomposition
- **分类: cs.LG; cs.AI; cs.CV; cs.CY; cs.IT**

- **链接: [https://arxiv.org/pdf/2407.00482v2](https://arxiv.org/pdf/2407.00482v2)**

> **作者:** Barproda Halder; Faisal Hamman; Pasan Dissanayake; Qiuyi Zhang; Ilia Sucholutsky; Sanghamitra Dutta
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Spuriousness arises when there is an association between two or more variables in a dataset that are not causally related. In this work, we propose an explainability framework to preemptively disentangle the nature of such spurious associations in a dataset before model training. We leverage a body of work in information theory called Partial Information Decomposition (PID) to decompose the total information about the target into four non-negative quantities, namely unique information (in core and spurious features, respectively), redundant information, and synergistic information. Our framework helps anticipate when the core or spurious feature is indispensable, when either suffices, and when both are jointly needed for an optimal classifier trained on the dataset. Next, we leverage this decomposition to propose a novel measure of the spuriousness of a dataset. We arrive at this measure systematically by examining several candidate measures, and demonstrating what they capture and miss through intuitive canonical examples and counterexamples. Our framework Spurious Disentangler consists of segmentation, dimensionality reduction, and estimation modules, with capabilities to specifically handle high-dimensional image data efficiently. Finally, we also perform empirical evaluation to demonstrate the trends of unique, redundant, and synergistic information, as well as our proposed spuriousness measure across $6$ benchmark datasets under various experimental settings. We observe an agreement between our preemptive measure of dataset spuriousness and post-training model generalization metrics such as worst-group accuracy, further supporting our proposition. The code is available at https://github.com/Barproda/spuriousness-disentangler.
>
---
#### [replaced 087] Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Network
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.08628v2](https://arxiv.org/pdf/2511.08628v2)**

> **作者:** Xuan Yu; Tianyang Xu
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Grassmannian manifold offers a powerful carrier for geometric representation learning by modelling high-dimensional data as low-dimensional subspaces. However, existing approaches predominantly rely on static single-subspace representations, neglecting the dynamic interplay between multiple subspaces critical for capturing complex geometric structures. To address this limitation, we propose a topology-driven multi-subspace fusion framework that enables adaptive subspace collaboration on the Grassmannian. Our solution introduces two key innovations: (1) Inspired by the Kolmogorov-Arnold representation theorem, an adaptive multi-subspace modelling mechanism is proposed that dynamically selects and weights task-relevant subspaces via topological convergence analysis, and (2) a multi-subspace interaction block that fuses heterogeneous geometric representations through Fréchet mean optimisation on the manifold. Theoretically, we establish the convergence guarantees of adaptive subspaces under a projection metric topology, ensuring stable gradient-based optimisation. Practically, we integrate Riemannian batch normalisation and mutual information regularisation to enhance discriminability and robustness. Extensive experiments on 3D action recognition (HDM05, FPHA), EEG classification (MAMEM-SSVEPII), and graph tasks demonstrate state-of-the-art performance. Our work not only advances geometric deep learning but also successfully adapts the proven multi-channel interaction philosophy of Euclidean networks to non-Euclidean domains, achieving superior discriminability and interpretability.
>
---
#### [replaced 088] Hierarchical Mixing Architecture for Low-light RAW Image Enhancement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.15497v2](https://arxiv.org/pdf/2510.15497v2)**

> **作者:** Xianmin Chen; Peiliang Huang; Longfei Han; Dingwen Zhang; Junwei Han
>
> **摘要:** With the rapid development of deep learning, low-light RAW image enhancement (LLRIE) has achieved remarkable progress. However, the challenge that how to simultaneously achieve strong enhancement quality and high efficiency still remains. Leveraging the inherent efficiency of Channel Attention and Mamba, we introduce a Hierarchical Mixing Architecture (HiMA), a hybrid LLRIE framework built upon two core modules. Specifically, we introduce Large Scale Block (LSB) for upper layers and Small Scale Block (SSB) for lower layers that reduce the parameters while improve the performance. Based on this framework, we also introduce a novel Local Distribution Adjustment (LoDA) module that adaptively aligns local feature statistics in a content-aware manner by learning to adjust regional luminance and contrast distributions. Moreover, to alleviate the domain ambiguity commonly observed in existing LLRIE pipelines, we design a Multi-Prior Fusion (MPF) module that leverages three complementary priors extracted from the first stage of the hybrid architecture to maintain domain consistency. Extensive experiments on multiple public benchmarks demonstrate that our approach outperforms state-of-the-art methods, delivering superior performance with fewer parameters. Code is available at https://github.com/Cynicarlos/HiMA.
>
---
#### [replaced 089] SIMS-V: Simulated Instruction-Tuning for Spatial Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.04668v2](https://arxiv.org/pdf/2511.04668v2)**

> **作者:** Ellis Brown; Arijit Ray; Ranjay Krishna; Ross Girshick; Rob Fergus; Saining Xie
>
> **备注:** Project page: https://ellisbrown.github.io/sims-v
>
> **摘要:** Despite impressive high-level video comprehension, multimodal language models struggle with spatial reasoning across time and space. While current spatial training approaches rely on real-world video data, obtaining diverse footage with precise spatial annotations remains a bottleneck. To alleviate this bottleneck, we present SIMS-V -- a systematic data-generation framework that leverages the privileged information of 3D simulators to create spatially-rich video training data for multimodal language models. Using this framework, we investigate which properties of simulated data drive effective real-world transfer through systematic ablations of question types, mixes, and scales. We identify a minimal set of three question categories (metric measurement, perspective-dependent reasoning, and temporal tracking) that prove most effective for developing transferable spatial intelligence, outperforming comprehensive coverage despite using fewer question types. These insights enable highly efficient training: our 7B-parameter video LLM fine-tuned on just 25K simulated examples outperforms the larger 72B baseline and achieves competitive performance with proprietary models on rigorous real-world spatial reasoning benchmarks. Our approach demonstrates robust generalization, maintaining performance on general video understanding while showing substantial improvements on embodied and real-world spatial tasks.
>
---
#### [replaced 090] LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/2511.08544v3](https://arxiv.org/pdf/2511.08544v3)**

> **作者:** Randall Balestriero; Yann LeCun
>
> **摘要:** Learning manipulable representations of the world and its dynamics is central to AI. Joint-Embedding Predictive Architectures (JEPAs) offer a promising blueprint, but lack of practical guidance and theory has led to ad-hoc R&D. We present a comprehensive theory of JEPAs and instantiate it in {\bf LeJEPA}, a lean, scalable, and theoretically grounded training objective. First, we identify the isotropic Gaussian as the optimal distribution that JEPAs' embeddings should follow to minimize downstream prediction risk. Second, we introduce a novel objective--{\bf Sketched Isotropic Gaussian Regularization} (SIGReg)--to constrain embeddings to reach that ideal distribution. Combining the JEPA predictive loss with SIGReg yields LeJEPA with numerous theoretical and practical benefits: (i) single trade-off hyperparameter, (ii) linear time and memory complexity, (iii) stability across hyper-parameters, architectures (ResNets, ViTs, ConvNets) and domains, (iv) heuristics-free, e.g., no stop-gradient, no teacher-student, no hyper-parameter schedulers, and (v) distributed training-friendly implementation requiring only $\approx$50 lines of code. Our empirical validation covers 10+ datasets, 60+ architectures, all with varying scales and domains. As an example, using imagenet-1k for pretraining and linear evaluation with frozen backbone, LeJEPA reaches 79\% with a ViT-H/14. We hope that the simplicity and theory-friendly ecosystem offered by LeJEPA will reestablish self-supervised pre-training as a core pillar of AI research (\href{https://github.com/rbalestr-lab/lejepa}{GitHub repo}).
>
---
#### [replaced 091] FlashI2V: Fourier-Guided Latent Shifting Prevents Conditional Image Leakage in Image-to-Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25187v2](https://arxiv.org/pdf/2509.25187v2)**

> **作者:** Yunyang Ge; Xinhua Cheng; Chengshu Zhao; Xianyi He; Shenghai Yuan; Bin Lin; Bin Zhu; Li Yuan
>
> **摘要:** In Image-to-Video (I2V) generation, a video is created using an input image as the first-frame condition. Existing I2V methods concatenate the full information of the conditional image with noisy latents to achieve high fidelity. However, the denoisers in these methods tend to shortcut the conditional image, which is known as conditional image leakage, leading to performance degradation issues such as slow motion and color inconsistency. In this work, we further clarify that conditional image leakage leads to overfitting to in-domain data and decreases the performance in out-of-domain scenarios. Moreover, we introduce Fourier-Guided Latent Shifting I2V, named FlashI2V, to prevent conditional image leakage. Concretely, FlashI2V consists of: (1) Latent Shifting. We modify the source and target distributions of flow matching by subtracting the conditional image information from the noisy latents, thereby incorporating the condition implicitly. (2) Fourier Guidance. We use high-frequency magnitude features obtained by the Fourier Transform to accelerate convergence and enable the adjustment of detail levels in the generated video. Experimental results show that our method effectively overcomes conditional image leakage and achieves the best generalization and performance on out-of-domain data among various I2V paradigms. With only 1.3B parameters, FlashI2V achieves a dynamic degree score of 53.01 on Vbench-I2V, surpassing CogVideoX1.5-5B-I2V and Wan2.1-I2V-14B-480P. Project page: https://pku-yuangroup.github.io/FlashI2V/
>
---
