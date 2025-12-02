# 计算机视觉 cs.CV

- **最新发布 279 篇**

- **更新 164 篇**

## 最新发布

#### [new 001] Rethinking Lung Cancer Screening: AI Nodule Detection and Diagnosis Outperforms Radiologists, Leading Models, and Standards Beyond Size and Growth
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文聚焦于肺癌筛查任务，旨在解决传统方法依赖结节大小和生长速度导致诊断延迟的问题。提出一种新型AI系统，直接在低剂量CT上实现结节检测与恶性判断，通过集成浅层深度学习与特征模型，在大规模数据上训练，显著优于放射科医生及现有标准与模型，尤其在早期癌和慢生长结节中表现突出。**

- **链接: [https://arxiv.org/pdf/2512.00281v1](https://arxiv.org/pdf/2512.00281v1)**

> **作者:** Sylvain Bodard; Pierre Baudot; Benjamin Renoust; Charles Voyton; Gwendoline De Bie; Ezequiel Geremia; Van-Khoa Le; Danny Francis; Pierre-Henri Siot; Yousra Haddou; Vincent Bobin; Jean-Christophe Brisset; Carey C. Thomson; Valerie Bourdes; Benoit Huet
>
> **备注:** 25 pages, 8 figures, with supplementary information containing 11 figures
>
> **摘要:** Early detection of malignant lung nodules is critical, but its dependence on size and growth in screening inherently delays diagnosis. We present an AI system that redefines lung cancer screening by performing both detection and malignancy diagnosis directly at the nodule level on low-dose CT scans. To address limitations in dataset scale and explainability, we designed an ensemble of shallow deep learning and feature-based specialized models. Trained and evaluated on 25,709 scans with 69,449 annotated nodules, the system outperforms radiologists, Lung-RADS, and leading AI models (Sybil, Brock, Google, Kaggle). It achieves an area under the receiver operating characteristic curve (AUC) of 0.98 internally and 0.945 on an independent cohort. With 0.5 false positives per scan at 99.3\% sensitivity, it addresses key barriers to AI adoption. Critically, it outperforms radiologists across all nodule sizes and stages, excelling in stage 1 cancers, and all growth-based metrics, including the least accurate: Volume-Doubling Time. It also surpasses radiologists by up to one year in diagnosing indeterminate and slow-growing nodules.
>
---
#### [new 002] VSRD++: Autolabeling for 3D Object Detection via Instance-Aware Volumetric Silhouette Rendering
- **分类: cs.CV**

- **简介: 该论文针对单目3D目标检测中依赖大量3D标注的问题，提出VSRD++框架。通过基于神经场的体素轮廓渲染实现实例感知的弱监督自动标注，分解SDF优化边界框，并引入速度与置信度建模动态物体，最终用伪标签训练检测器，在KITTI-360上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.01178v1](https://arxiv.org/pdf/2512.01178v1)**

> **作者:** Zihua Liu; Hiroki Sakuma; Masatoshi Okutomi
>
> **备注:** arXiv admin note: text overlap with arXiv:2404.00149
>
> **摘要:** Monocular 3D object detection is a fundamental yet challenging task in 3D scene understanding. Existing approaches heavily depend on supervised learning with extensive 3D annotations, which are often acquired from LiDAR point clouds through labor-intensive labeling processes. To tackle this problem, we propose VSRD++, a novel weakly supervised framework for monocular 3D object detection that eliminates the reliance on 3D annotations and leverages neural-field-based volumetric rendering with weak 2D supervision. VSRD++ consists of a two-stage pipeline: multi-view 3D autolabeling and subsequent monocular 3D detector training. In the multi-view autolabeling stage, object surfaces are represented as signed distance fields (SDFs) and rendered as instance masks via the proposed instance-aware volumetric silhouette rendering. To optimize 3D bounding boxes, we decompose each instance's SDF into a cuboid SDF and a residual distance field (RDF) that captures deviations from the cuboid. To address the geometry inconsistency commonly observed in volume rendering methods applied to dynamic objects, we model the dynamic objects by including velocity into bounding box attributes as well as assigning confidence to each pseudo-label. Moreover, we also employ a 3D attribute initialization module to initialize the dynamic bounding box parameters. In the monocular 3D object detection phase, the optimized 3D bounding boxes serve as pseudo labels for training monocular 3D object detectors. Extensive experiments on the KITTI-360 dataset demonstrate that VSRD++ significantly outperforms existing weakly supervised approaches for monocular 3D object detection on both static and dynamic scenes. Code is available at https://github.com/Magicboomliu/VSRD_plus_plus
>
---
#### [new 003] Multi-GRPO: Multi-Group Advantage Estimation for Text-to-Image Generation with Tree-Based Trajectories and Multiple Rewards
- **分类: cs.CV**

- **简介: 该论文针对文本生成图像（T2I）中的对齐问题，解决GRPO方法存在的共享信用分配和奖励混合缺陷。提出Multi-GRPO框架，通过树状轨迹实现时间分组以精准评估早期去噪步骤，并基于奖励分组独立计算优势，解耦多目标冲突。在新构建的OCR-Color-10数据集上验证了其优越的稳定性与对齐性能。**

- **链接: [https://arxiv.org/pdf/2512.00743v1](https://arxiv.org/pdf/2512.00743v1)**

> **作者:** Qiang Lyu; Zicong Chen; Chongxiao Wang; Haolin Shi; Shibo Gao; Ran Piao; Youwei Zeng; Jianlou Si; Fei Ding; Jing Li; Chun Pong Lau; Weiqiang Wang
>
> **备注:** 20 pages, 15 figures
>
> **摘要:** Recently, Group Relative Policy Optimization (GRPO) has shown promising potential for aligning text-to-image (T2I) models, yet existing GRPO-based methods suffer from two critical limitations. (1) \textit{Shared credit assignment}: trajectory-level advantages derived from group-normalized sparse terminal rewards are uniformly applied across timesteps, failing to accurately estimate the potential of early denoising steps with vast exploration spaces. (2) \textit{Reward-mixing}: predefined weights for combining multi-objective rewards (e.g., text accuracy, visual quality, text color)--which have mismatched scales and variances--lead to unstable gradients and conflicting updates. To address these issues, we propose \textbf{Multi-GRPO}, a multi-group advantage estimation framework with two orthogonal grouping mechanisms. For better credit assignment, we introduce tree-based trajectories inspired by Monte Carlo Tree Search: branching trajectories at selected early denoising steps naturally forms \emph{temporal groups}, enabling accurate advantage estimation for early steps via descendant leaves while amortizing computation through shared prefixes. For multi-objective optimization, we introduce \emph{reward-based grouping} to compute advantages for each reward function \textit{independently} before aggregation, disentangling conflicting signals. To facilitate evaluation of multiple objective alignment, we curate \textit{OCR-Color-10}, a visual text rendering dataset with explicit color constraints. Across the single-reward \textit{PickScore-25k} and multi-objective \textit{OCR-Color-10} benchmarks, Multi-GRPO achieves superior stability and alignment performance, effectively balancing conflicting objectives. Code will be publicly available at \href{https://github.com/fikry102/Multi-GRPO}{https://github.com/fikry102/Multi-GRPO}.
>
---
#### [new 004] Terrain Sensing with Smartphone Structured Light: 2D Dynamic Time Warping for Grid Pattern Matching
- **分类: cs.CV**

- **简介: 该论文针对移动机器人在不平地形中感知微小起伏的难题，提出基于智能手机的结构光系统。通过设计拓扑约束的2D-DTW算法，实现对投影网格在透视畸变和遮挡下的鲁棒匹配，以重建局部地形。工作包括系统设计与2D-DTW算法创新，适用于资源受限平台，兼具地形感知与通用网格匹配能力。**

- **链接: [https://arxiv.org/pdf/2512.00514v1](https://arxiv.org/pdf/2512.00514v1)**

> **作者:** Tanaka Nobuaki
>
> **摘要:** Low-cost mobile rovers often operate on uneven terrain where small bumps or tilts are difficult to perceive visually but can significantly affect locomotion stability. To address this problem, we explore a smartphone-based structured-light system that projects a grid pattern onto the ground and reconstructs local terrain unevenness from a single handheld device. The system is inspired by face-recognition projectors, but adapted for ground sensing. A key technical challenge is robustly matching the projected grid with its deformed observation under perspective distortion and partial occlusion. Conventional one-dimensional dynamic time warping (1D-DTW) is not directly applicable to such two-dimensional grid patterns. We therefore propose a topology-constrained two-dimensional dynamic time warping (2D-DTW) algorithm that performs column-wise alignment under a global grid consistency constraint. The proposed method is designed to be simple enough to run on resource limited platforms while preserving the grid structure required for accurate triangulation. We demonstrate that our 2D-DTW formulation can be used not only for terrain sensing but also as a general tool for matching structured grid patterns in image processing scenarios. This paper describes the overall system design as well as the 2D-DTW extension that emerged from this application.
>
---
#### [new 005] DenseScan: Advancing 3D Scene Understanding with 2D Dense Annotation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DenseScan，一个基于多视图2D图像和多模态大模型的3D场景理解数据集。针对现有数据缺乏细粒度语义标注的问题，通过自动化管道实现密集场景元素描述与情景化问答生成，提升3D场景的语义丰富性，推动视觉-语言任务发展。**

- **链接: [https://arxiv.org/pdf/2512.00226v1](https://arxiv.org/pdf/2512.00226v1)**

> **作者:** Zirui Wang; Tao Zhang
>
> **备注:** Workshop on Space in Vision, Language, and Embodied AI at NeurIPS 2025
>
> **摘要:** 3D understanding is a key capability for real-world AI assistance. High-quality data plays an important role in driving the development of the 3D understanding community. Current 3D scene understanding datasets often provide geometric and instance-level information, yet they lack the rich semantic annotations necessary for nuanced visual-language tasks.In this work, we introduce DenseScan, a novel dataset with detailed multi-level descriptions generated by an automated pipeline leveraging multi-view 2D images and multimodal large language models (MLLMs). Our approach enables dense captioning of scene elements, ensuring comprehensive object-level descriptions that capture context-sensitive details. Furthermore, we extend these annotations through scenario-based question generation, producing high-level queries that integrate object properties, spatial relationships, and scene context. By coupling geometric detail with semantic richness, DenseScan broadens the range of downstream tasks, from detailed visual-language navigation to interactive question answering. Experimental results demonstrate that our method significantly enhances object-level understanding and question-answering performance in 3D environments compared to traditional annotation pipelines. We release both the annotated dataset and our annotation pipeline to facilitate future research and applications in robotics, augmented reality, and beyond. Through DenseScan, we aim to catalyze new avenues in 3D scene understanding, allowing researchers and practitioners to tackle the complexities of real-world environments with richer, more contextually aware annotations.
>
---
#### [new 006] M4-BLIP: Advancing Multi-Modal Media Manipulation Detection through Face-Enhanced Local Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态媒体伪造检测任务，聚焦于局部区域（尤其是面部）的伪造问题。提出M4-BLIP框架，结合BLIP-2提取局部特征与面部先验知识，通过融合模块增强全局与局部信息，提升检测精度，并集成LLM提高结果可解释性。**

- **链接: [https://arxiv.org/pdf/2512.01214v1](https://arxiv.org/pdf/2512.01214v1)**

> **作者:** Hang Wu; Ke Sun; Jiayi Ji; Xiaoshuai Sun; Rongrong Ji
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** In the contemporary digital landscape, multi-modal media manipulation has emerged as a significant societal threat, impacting the reliability and integrity of information dissemination. Current detection methodologies in this domain often overlook the crucial aspect of localized information, despite the fact that manipulations frequently occur in specific areas, particularly in facial regions. In response to this critical observation, we propose the M4-BLIP framework. This innovative framework utilizes the BLIP-2 model, renowned for its ability to extract local features, as the cornerstone for feature extraction. Complementing this, we incorporate local facial information as prior knowledge. A specially designed alignment and fusion module within M4-BLIP meticulously integrates these local and global features, creating a harmonious blend that enhances detection accuracy. Furthermore, our approach seamlessly integrates with Large Language Models (LLM), significantly improving the interpretability of the detection outcomes. Extensive quantitative and visualization experiments validate the effectiveness of our framework against the state-of-the-art competitors.
>
---
#### [new 007] Accelerating Streaming Video Large Language Models via Hierarchical Token Compression
- **分类: cs.CV**

- **简介: 该论文针对流式视频大模型（VideoLLM）实时部署中因密集视觉令牌导致的高计算开销问题，提出一种可插拔的分层令牌压缩框架STC。通过缓存相似帧特征（STC-Cacher）和压缩冗余令牌（STC-Pruner），显著降低视觉编码与语言模型预填充阶段的延迟，实现99%准确率下24.5%和45.3%的加速。**

- **链接: [https://arxiv.org/pdf/2512.00891v1](https://arxiv.org/pdf/2512.00891v1)**

> **作者:** Yiyu Wang; Xuyang Liu; Xiyan Gui; Xinying Lin; Boxue Yang; Chenfei Liao; Tailai Chen; Linfeng Zhang
>
> **备注:** Code is avaliable at \url{https://github.com/lern-to-write/STC}
>
> **摘要:** Streaming Video Large Language Models (VideoLLMs) have demonstrated impressive performance across various video understanding tasks, but they face significant challenges in real-time deployment due to the high computational cost of processing dense visual tokens from continuous video streams. In streaming video scenarios, the primary bottleneck lies in the Vision Transformer (ViT) encoding stage, where redundant processing of temporally similar frames leads to inefficiency. Additionally, inflated token sequences during LLM pre-filling further exacerbate latency and memory overhead. To address these challenges, we propose \textbf{S}treaming \textbf{T}oken \textbf{C}ompression (\textbf{STC}), a plug-and-play hierarchical framework that seamlessly integrates into existing streaming VideoLLMs, optimizing both ViT encoding and LLM pre-filling stages to accelerate processing. STC introduces two token-level accelerators: \textbf{STC-Cacher}, which reduces ViT encoding overhead by caching and reusing features from temporally similar frames, and \textbf{STC-Pruner}, which compresses the visual token sequence before it enters the LLM, preserving only the most salient tokens based on both spatial and temporal relevance. Extensive experiments on four baseline streaming VideoLLMs across five benchmarks demonstrate that STC outperforms other compression methods. Notably, STC retains up to \textbf{99\%} of accuracy on the ReKV framework while reducing ViT encoding latency and LLM pre-filling latency by \textbf{24.5\%} and \textbf{45.3\%}.
>
---
#### [new 008] AlignVid: Training-Free Attention Scaling for Semantic Fidelity in Text-Guided Image-to-Video Generation
- **分类: cs.CV**

- **简介: 该论文针对文本引导图像生成视频（TI2V）中的语义偏差问题，提出训练-free的AlignVid框架。通过注意力缩放调制与引导调度，提升对细粒度提示的遵循能力，尤其在对象增删改场景下改善语义保真度，并引入OmitI2V数据集评估该问题。**

- **链接: [https://arxiv.org/pdf/2512.01334v1](https://arxiv.org/pdf/2512.01334v1)**

> **作者:** Yexin Liu; Wen-Jie Shu; Zile Huang; Haoze Zheng; Yueze Wang; Manyuan Zhang; Ser-Nam Lim; Harry Yang
>
> **摘要:** Text-guided image-to-video (TI2V) generation has recently achieved remarkable progress, particularly in maintaining subject consistency and temporal coherence. However, existing methods still struggle to adhere to fine-grained prompt semantics, especially when prompts entail substantial transformations of the input image (e.g., object addition, deletion, or modification), a shortcoming we term semantic negligence. In a pilot study, we find that applying a Gaussian blur to the input image improves semantic adherence. Analyzing attention maps, we observe clearer foreground-background separation. From an energy perspective, this corresponds to a lower-entropy cross-attention distribution. Motivated by this, we introduce AlignVid, a training-free framework with two components: (i) Attention Scaling Modulation (ASM), which directly reweights attention via lightweight Q or K scaling, and (ii) Guidance Scheduling (GS), which applies ASM selectively across transformer blocks and denoising steps to reduce visual quality degradation. This minimal intervention improves prompt adherence while limiting aesthetic degradation. In addition, we introduce OmitI2V to evaluate semantic negligence in TI2V generation, comprising 367 human-annotated samples that span addition, deletion, and modification scenarios. Extensive experiments demonstrate that AlignVid can enhance semantic fidelity.
>
---
#### [new 009] IRPO: Boosting Image Restoration via Post-training GRPO
- **分类: cs.CV**

- **简介: 该论文针对图像恢复任务，解决现有方法过度平滑、泛化性差的问题。提出基于GRPO的后训练框架IRPO，通过筛选低表现样本优化数据，并设计三重奖励机制平衡精度与感知质量，显著提升恢复效果，在多个基准上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00814v1](https://arxiv.org/pdf/2512.00814v1)**

> **作者:** Haoxuan Xu. Yi Liu; Boyuan Jiang; Jinlong Peng; Donghao Luo; Xiaobin Hu; Shuicheng Yan; Haoang Li
>
> **摘要:** Recent advances in post-training paradigms have achieved remarkable success in high-level generation tasks, yet their potential for low-level vision remains rarely explored. Existing image restoration (IR) methods rely on pixel-level hard-fitting to ground-truth images, struggling with over-smoothing and poor generalization. To address these limitations, we propose IRPO, a low-level GRPO-based post-training paradigm that systematically explores both data formulation and reward modeling. We first explore a data formulation principle for low-level post-training paradigm, in which selecting underperforming samples from the pre-training stage yields optimal performance and improved efficiency. Furthermore, we model a reward-level criteria system that balances objective accuracy and human perceptual preference through three complementary components: a General Reward for structural fidelity, an Expert Reward leveraging Qwen-VL for perceptual alignment, and a Restoration Reward for task-specific low-level quality. Comprehensive experiments on six in-domain and five out-of-domain (OOD) low-level benchmarks demonstrate that IRPO achieves state-of-the-art results across diverse degradation types, surpassing the AdaIR baseline by 0.83 dB on in-domain tasks and 3.43 dB on OOD settings. Our code can be shown in https://github.com/HaoxuanXU1024/IRPO.
>
---
#### [new 010] Image Generation as a Visual Planner for Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究视觉规划在机器人操作中的应用，旨在解决传统视频生成模型依赖大量特定数据且泛化能力差的问题。作者提出利用预训练图像生成模型（如扩散模型）通过轻量微调（LoRA）实现文本或轨迹条件下的视频生成，使其能作为无需复杂时序建模的视觉规划器，有效生成符合指令的连贯机器人操作视频。**

- **链接: [https://arxiv.org/pdf/2512.00532v1](https://arxiv.org/pdf/2512.00532v1)**

> **作者:** Ye Pang
>
> **备注:** 11 pages 9 figures Under review at CVPR 2026
>
> **摘要:** Generating realistic robotic manipulation videos is an important step toward unifying perception, planning, and action in embodied agents. While existing video diffusion models require large domain-specific datasets and struggle to generalize, recent image generation models trained on language-image corpora exhibit strong compositionality, including the ability to synthesize temporally coherent grid images. This suggests a latent capacity for video-like generation even without explicit temporal modeling. We explore whether such models can serve as visual planners for robots when lightly adapted using LoRA finetuning. We propose a two-part framework that includes: (1) text-conditioned generation, which uses a language instruction and the first frame, and (2) trajectory-conditioned generation, which uses a 2D trajectory overlay and the same initial frame. Experiments on the Jaco Play dataset, Bridge V2, and the RT1 dataset show that both modes produce smooth, coherent robot videos aligned with their respective conditions. Our findings indicate that pretrained image generators encode transferable temporal priors and can function as video-like robotic planners under minimal supervision. Code is released at \href{https://github.com/pangye202264690373/Image-Generation-as-a-Visual-Planner-for-Robotic-Manipulation}{https://github.com/pangye202264690373/Image-Generation-as-a-Visual-Planner-for-Robotic-Manipulation}.
>
---
#### [new 011] Affordance-First Decomposition for Continual Learning in Video-Language Understanding
- **分类: cs.CV**

- **简介: 该论文针对视频-语言理解中的持续学习问题，提出Affordance-First Decomposition（AFD）框架。通过将视频映射为稳定不变的语义特征（ affordance tokens），实现交互中心的稳定基底；同时设计轻量级、查询驱动的调度器，仅在必要时动态适应并扩展容量。采用仅问题重放策略，在保持隐私与内存受限的前提下，有效缓解遗忘，显著提升模型在域和时间增量场景下的性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00694v1](https://arxiv.org/pdf/2512.00694v1)**

> **作者:** Mengzhu Xu; Hanzhi Liu; Ningkang Peng; Qianyu Chen; Canran Xiao
>
> **备注:** Under review
>
> **摘要:** Continual learning for video--language understanding is increasingly important as models face non-stationary data, domains, and query styles, yet prevailing solutions blur what should stay stable versus what should adapt, rely on static routing/capacity, or require replaying past videos. We aim to explicitly specify where stability lives and where plasticity should be focused under realistic memory and privacy constraints. We introduce Affordance-First Decomposition (AFD): videos are mapped to slowly varying affordance tokens that form a shared, time-aligned substrate, while a lightweight, query-routed, conflict-aware scheduler concentrates adaptation and grows capacity only when needed. The substrate is stabilized via weak alignment and teacher consistency, and training uses question-only replay. AFD achieves state-of-the-art across protocols: 51.6% average accuracy with -1.8% forgetting on domain-incremental VideoQA, ViLCo R@1@0.5 of 29.6% (MQ) and 20.7% (NLQ) with 18.4% stAP@0.25 (VQ), and 39.5% accuracy with -1.6% forgetting on time-incremental iVQA. Overall, AFD offers an explicit, interpretable split between a stable interaction-centered substrate and targeted adaptation.
>
---
#### [new 012] Rice-VL: Evaluating Vision-Language Models for Cultural Understanding Across ASEAN Countries
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型（VLMs）的西方中心偏见问题，提出RICE-VL基准，评估其在11个东盟国家的文化理解能力。通过2.8万+跨文化VQA样本和1000组图像标注数据，结合SEA-LAVE指标，揭示了模型在低资源国家及抽象文化领域的性能短板，推动更具包容性的多文化AI发展。**

- **链接: [https://arxiv.org/pdf/2512.01419v1](https://arxiv.org/pdf/2512.01419v1)**

> **作者:** Tushar Pranav; Eshan Pandey; Austria Lyka Diane Bala; Aman Chadha; Indriyati Atmosukarto; Donny Soh Cheng Lock
>
> **备注:** 14 pages
>
> **摘要:** Vision-Language Models (VLMs) excel in multimodal tasks but often exhibit Western-centric biases, limiting their effectiveness in culturally diverse regions like Southeast Asia (SEA). To address this, we introduce RICE-VL, a novel benchmark evaluating VLM cultural understanding across 11 ASEAN countries. RICE-VL includes over 28,000 human-curated Visual Question Answering (VQA) samples -- covering True or False, Fill-in-the-Blank, and open-ended formats -- and 1,000 image-bounding box pairs for Visual Grounding, annotated by culturally informed experts across 14 sub-ground categories. We propose SEA-LAVE, an extension of the LAVE metric, assessing textual accuracy, cultural alignment, and country identification. Evaluations of six open- and closed-source VLMs reveal significant performance gaps in low-resource countries and abstract cultural domains. The Visual Grounding task tests models' ability to localize culturally significant elements in complex scenes, probing spatial and contextual accuracy. RICE-VL exposes limitations in VLMs' cultural comprehension and highlights the need for inclusive model development to better serve diverse global populations.
>
---
#### [new 013] UniDiff: Parameter-Efficient Adaptation of Diffusion Models for Land Cover Classification with Multi-Modal Remotely Sensed Imagery and Sparse Annotations
- **分类: cs.CV**

- **简介: 该论文针对多模态遥感影像分类中标注数据稀缺的问题，提出UniDiff框架。通过参数高效微调与伪RGB锚定，将ImageNet预训练扩散模型适配至高光谱与SAR等异构模态，实现仅用目标域数据的无监督适应，有效缓解标注瓶颈并提升多模态融合性能。**

- **链接: [https://arxiv.org/pdf/2512.00261v1](https://arxiv.org/pdf/2512.00261v1)**

> **作者:** Yuzhen Hu; Saurabh Prasad
>
> **备注:** Camera-ready for WACV 2026
>
> **摘要:** Sparse annotations fundamentally constrain multimodal remote sensing: even recent state-of-the-art supervised methods such as MSFMamba are limited by the availability of labeled data, restricting their practical deployment despite architectural advances. ImageNet-pretrained models provide rich visual representations, but adapting them to heterogeneous modalities such as hyperspectral imaging (HSI) and synthetic aperture radar (SAR) without large labeled datasets remains challenging. We propose UniDiff, a parameter-efficient framework that adapts a single ImageNet-pretrained diffusion model to multiple sensing modalities using only target-domain data. UniDiff combines FiLM-based timestep-modality conditioning, parameter-efficient adaptation of approximately 5% of parameters, and pseudo-RGB anchoring to preserve pre-trained representations and prevent catastrophic forgetting. This design enables effective feature extraction from remote sensing data under sparse annotations. Our results with two established multi-modal benchmarking datasets demonstrate that unsupervised adaptation of a pre-trained diffusion model effectively mitigates annotation constraints and achieves effective fusion of multi-modal remotely sensed data.
>
---
#### [new 014] PhyDetEx: Detecting and Explaining the Physical Plausibility of T2V Models
- **分类: cs.CV**

- **简介: 该论文针对文本到视频（T2V）模型生成内容的物理合理性问题，构建了PID数据集，并提出轻量级微调方法，使视觉语言模型（VLM）能检测并解释视频中的物理不一致现象。研究通过PhyDetEx评估主流T2V模型，发现其对物理规律理解仍存显著不足，尤其开源模型表现较差。**

- **链接: [https://arxiv.org/pdf/2512.01843v1](https://arxiv.org/pdf/2512.01843v1)**

> **作者:** Zeqing Wang; Keze Wang; Lei Zhang
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Driven by the growing capacity and training scale, Text-to-Video (T2V) generation models have recently achieved substantial progress in video quality, length, and instruction-following capability. However, whether these models can understand physics and generate physically plausible videos remains a question. While Vision-Language Models (VLMs) have been widely used as general-purpose evaluators in various applications, they struggle to identify the physically impossible content from generated videos. To investigate this issue, we construct a \textbf{PID} (\textbf{P}hysical \textbf{I}mplausibility \textbf{D}etection) dataset, which consists of a \textit{test split} of 500 manually annotated videos and a \textit{train split} of 2,588 paired videos, where each implausible video is generated by carefully rewriting the caption of its corresponding real-world video to induce T2V models producing physically implausible content. With the constructed dataset, we introduce a lightweight fine-tuning approach, enabling VLMs to not only detect physically implausible events but also generate textual explanations on the violated physical principles. Taking the fine-tuned VLM as a physical plausibility detector and explainer, namely \textbf{PhyDetEx}, we benchmark a series of state-of-the-art T2V models to assess their adherence to physical laws. Our findings show that although recent T2V models have made notable progress toward generating physically plausible content, understanding and adhering to physical laws remains a challenging issue, especially for open-source models. Our dataset, training code, and checkpoints are available at \href{https://github.com/Zeqing-Wang/PhyDetEx}{https://github.com/Zeqing-Wang/PhyDetEx}.
>
---
#### [new 015] Towards aligned body representations in vision models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉模型中的身体表征问题，旨在探索机器学习模型是否能生成类人粗粒度的物理空间表征。通过将人类心理实验转化为语义分割任务，对比不同规模模型的表现，发现小模型更倾向于形成类似人类的粗略身体表征，而大模型则趋向过度细节化。研究揭示了计算资源限制对表征结构的影响，为理解大脑物理推理机制提供了新路径。**

- **链接: [https://arxiv.org/pdf/2512.00365v1](https://arxiv.org/pdf/2512.00365v1)**

> **作者:** Andrey Gizdov; Andrea Procopio; Yichen Li; Daniel Harari; Tomer Ullman
>
> **备注:** Andrea Procopio and Andrey Gizdov have equal contributions
>
> **摘要:** Human physical reasoning relies on internal "body" representations - coarse, volumetric approximations that capture an object's extent and support intuitive predictions about motion and physics. While psychophysical evidence suggests humans use such coarse representations, their internal structure remains largely unknown. Here we test whether vision models trained for segmentation develop comparable representations. We adapt a psychophysical experiment conducted with 50 human participants to a semantic segmentation task and test a family of seven segmentation networks, varying in size. We find that smaller models naturally form human-like coarse body representations, whereas larger models tend toward overly detailed, fine-grain encodings. Our results demonstrate that coarse representations can emerge under limited computational resources, and that machine representations can provide a scalable path toward understanding the structure of physical reasoning in the brain.
>
---
#### [new 016] TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models
- **分类: cs.CV**

- **简介: 该论文提出TUNA，一种原生统一多模态模型，通过级联VAE编码器与表示编码器构建统一视觉表征空间，实现图像与视频的端到端理解与生成。解决了传统模型中多模态表示解耦导致的格式不匹配问题，提升了任务性能，并证明联合训练可促进理解与生成任务协同优化。**

- **链接: [https://arxiv.org/pdf/2512.02014v1](https://arxiv.org/pdf/2512.02014v1)**

> **作者:** Zhiheng Liu; Weiming Ren; Haozhe Liu; Zijian Zhou; Shoufa Chen; Haonan Qiu; Xiaoke Huang; Zhaochong An; Fanny Yang; Aditya Patel; Viktar Atliha; Tony Ng; Xiao Han; Chuyan Zhu; Chenyang Zhang; Ding Liu; Juan-Manuel Perez-Rua; Sen He; Jürgen Schmidhuber; Wenhu Chen; Ping Luo; Wei Liu; Tao Xiang; Jonas Schult; Yuren Cong
>
> **备注:** Project page: https://tuna-ai.org/
>
> **摘要:** Unified multimodal models (UMMs) aim to jointly perform multimodal understanding and generation within a single framework. We present TUNA, a native UMM that builds a unified continuous visual representation by cascading a VAE encoder with a representation encoder. This unified representation space allows end-to-end processing of images and videos for both understanding and generation tasks. Compared to prior UMMs with decoupled representations, TUNA's unified visual space avoids representation format mismatches introduced by separate encoders, outperforming decoupled alternatives in both understanding and generation. Moreover, we observe that stronger pretrained representation encoders consistently yield better performance across all multimodal tasks, highlighting the importance of the representation encoder. Finally, in this unified setting, jointly training on both understanding and generation data allows the two tasks to benefit from each other rather than interfere. Our extensive experiments on multimodal understanding and generation benchmarks show that TUNA achieves state-of-the-art results in image and video understanding, image and video generation, and image editing, demonstrating the effectiveness and scalability of its unified representation design.
>
---
#### [new 017] ViT$^3$: Unlocking Test-Time Training in Vision
- **分类: cs.CV**

- **简介: 该论文针对视觉序列建模中的测试时训练（TTT）技术，系统研究其设计要素，提出ViT³模型。旨在解决视觉TTT中内模块与训练策略缺乏明确指导的问题，通过六项实践洞察构建高效、线性复杂度的纯TTT架构，在多项视觉任务上表现优异，推动了该领域发展。**

- **链接: [https://arxiv.org/pdf/2512.01643v1](https://arxiv.org/pdf/2512.01643v1)**

> **作者:** Dongchen Han; Yining Li; Tianyu Li; Zixuan Cao; Ziming Wang; Jun Song; Yu Cheng; Bo Zheng; Gao Huang
>
> **摘要:** Test-Time Training (TTT) has recently emerged as a promising direction for efficient sequence modeling. TTT reformulates attention operation as an online learning problem, constructing a compact inner model from key-value pairs at test time. This reformulation opens a rich and flexible design space while achieving linear computational complexity. However, crafting a powerful visual TTT design remains challenging: fundamental choices for the inner module and inner training lack comprehensive understanding and practical guidelines. To bridge this critical gap, in this paper, we present a systematic empirical study of TTT designs for visual sequence modeling. From a series of experiments and analyses, we distill six practical insights that establish design principles for effective visual TTT and illuminate paths for future improvement. These findings culminate in the Vision Test-Time Training (ViT$^3$) model, a pure TTT architecture that achieves linear complexity and parallelizable computation. We evaluate ViT$^3$ across diverse visual tasks, including image classification, image generation, object detection, and semantic segmentation. Results show that ViT$^3$ consistently matches or outperforms advanced linear-complexity models (e.g., Mamba and linear attention variants) and effectively narrows the gap to highly optimized vision Transformers. We hope this study and the ViT$^3$ baseline can facilitate future work on visual TTT models. Code is available at https://github.com/LeapLabTHU/ViTTT.
>
---
#### [new 018] TGSFormer: Scalable Temporal Gaussian Splatting for Embodied Semantic Scene Completion
- **分类: cs.CV**

- **简介: 该论文针对具身3D语义场景补全任务，解决现有方法在无界场景中因随机初始化导致冗余、扩展性差的问题。提出TGSFormer框架，通过持久化高斯记忆与双时序编码实现高效时序融合，结合置信度感知体素融合，显著减少参数量，提升精度与长期一致性。**

- **链接: [https://arxiv.org/pdf/2512.00300v1](https://arxiv.org/pdf/2512.00300v1)**

> **作者:** Rui Qian; Haozhi Cao; Tianchen Deng; Tianxin Hu; Weixiang Guo; Shenghai Yuan; Lihua Xie
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Embodied 3D Semantic Scene Completion (SSC) infers dense geometry and semantics from continuous egocentric observations. Most existing Gaussian-based methods rely on random initialization of many primitives within predefined spatial bounds, resulting in redundancy and poor scalability to unbounded scenes. Recent depth-guided approach alleviates this issue but remains local, suffering from latency and memory overhead as scale increases. To overcome these challenges, we propose TGSFormer, a scalable Temporal Gaussian Splatting framework for embodied SSC. It maintains a persistent Gaussian memory for temporal prediction, without relying on image coherence or frame caches. For temporal fusion, a Dual Temporal Encoder jointly processes current and historical Gaussian features through confidence-aware cross-attention. Subsequently, a Confidence-aware Voxel Fusion module merges overlapping primitives into voxel-aligned representations, regulating density and maintaining compactness. Extensive experiments demonstrate that TGSFormer achieves state-of-the-art results on both local and embodied SSC benchmarks, offering superior accuracy and scalability with significantly fewer primitives while maintaining consistent long-term scene integrity. The code will be released upon acceptance.
>
---
#### [new 019] Mammo-FM: Breast-specific foundational model for Integrated Mammographic Diagnosis, Prognosis, and Reporting
- **分类: cs.CV**

- **简介: 该论文提出Mammo-FM，首个乳腺影像专用基础模型，解决乳腺癌诊断、定位、报告生成与风险预测的整合问题。基于大规模多中心数据预训练，实现图像与文本对齐，提升可解释性与临床可用性，在多项任务中优于通用模型，验证了领域专属基础模型的有效性。**

- **链接: [https://arxiv.org/pdf/2512.00198v1](https://arxiv.org/pdf/2512.00198v1)**

> **作者:** Shantanu Ghosh; Vedant Parthesh Joshi; Rayan Syed; Aya Kassem; Abhishek Varshney; Payel Basak; Weicheng Dai; Judy Wawira Gichoya; Hari M. Trivedi; Imon Banerjee; Shyam Visweswaran; Clare B. Poynton; Kayhan Batmanghelich
>
> **摘要:** Breast cancer is one of the leading causes of death among women worldwide. We introduce Mammo-FM, the first foundation model specifically for mammography, pretrained on the largest and most diverse dataset to date - 140,677 patients (821,326 mammograms) across four U.S. institutions. Mammo-FM provides a unified foundation for core clinical tasks in breast imaging, including cancer diagnosis, pathology localization, structured report generation, and cancer risk prognosis within a single framework. Its alignment between images and text enables both visual and textual interpretability, improving transparency and clinical auditability, which are essential for real-world adoption. We rigorously evaluate Mammo-FM across diagnosis, prognosis, and report-generation tasks in in- and out-of-distribution datasets. Despite operating on native-resolution mammograms and using only one-third of the parameters of state-of-the-art generalist FMs, Mammo-FM consistently outperforms them across multiple public and private benchmarks. These results highlight the efficiency and value of domain-specific foundation models designed around the full spectrum of tasks within a clinical domain and emphasize the importance of rigorous, domain-aligned evaluation.
>
---
#### [new 020] Efficient Edge-Compatible CNN for Speckle-Based Material Recognition in Laser Cutting Systems
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文针对激光切割中材料识别难题，提出一种轻量级专用CNN模型，用于基于激光散斑的材料分类。解决了传统方法参数多、计算量大、部署难的问题，在59类材料上达95.05%准确率，仅341k参数，可实时部署于边缘设备，支持高效精准的切割参数自适应。**

- **链接: [https://arxiv.org/pdf/2512.00179v1](https://arxiv.org/pdf/2512.00179v1)**

> **作者:** Mohamed Abdallah Salem; Nourhan Zein Diab
>
> **备注:** Copyright 2025 IEEE. This is the author's version of the work that has been Accepted for publication in the Proceedings of the 2025 IEEE The 35th International Conference on Computer Theory and Applications (ICCTA 2025). Final published version will be available on IEEE Xplore
>
> **摘要:** Accurate material recognition is critical for safe and effective laser cutting, as misidentification can lead to poor cut quality, machine damage, or the release of hazardous fumes. Laser speckle sensing has recently emerged as a low-cost and non-destructive modality for material classification; however, prior work has either relied on computationally expensive backbone networks or addressed only limited subsets of materials. In this study, A lightweight convolutional neural network (CNN) tailored for speckle patterns is proposed, designed to minimize parameters while maintaining high discriminative power. Using the complete SensiCut dataset of 59 material classes spanning woods, acrylics, composites, textiles, metals, and paper-based products, the proposed model achieves 95.05% test accuracy, with macro and weighted F1-scores of 0.951. The network contains only 341k trainable parameters (~1.3 MB) -- over 70X fewer than ResNet-50 -- and achieves an inference speed of 295 images per second, enabling deployment on Raspberry Pi and Jetson-class devices. Furthermore, when materials are regrouped into nine and five practical families, recall exceeds 98% and approaches 100%, directly supporting power and speed preset selection in laser cutters. These results demonstrate that compact, domain-specific CNNs can outperform large backbones for speckle-based material classification, advancing the feasibility of material-aware, edge-deployable laser cutting systems.
>
---
#### [new 021] What about gravity in video generation? Post-Training Newton's Laws with Verifiable Rewards
- **分类: cs.CV**

- **简介: 该论文针对视频生成中物理规律缺失的问题，提出基于可验证奖励的后训练框架NewtonRewards。通过光学流和外观特征作为速度与质量的可测量代理，引入牛顿运动约束与质量守恒奖励，提升生成视频的物理合理性与运动一致性，显著改善了视觉与物理指标。**

- **链接: [https://arxiv.org/pdf/2512.00425v1](https://arxiv.org/pdf/2512.00425v1)**

> **作者:** Minh-Quan Le; Yuanzhi Zhu; Vicky Kalogeiton; Dimitris Samaras
>
> **备注:** Project page: https://cvlab-stonybrook.github.io/NewtonRewards
>
> **摘要:** Recent video diffusion models can synthesize visually compelling clips, yet often violate basic physical laws-objects float, accelerations drift, and collisions behave inconsistently-revealing a persistent gap between visual realism and physical realism. We propose $\texttt{NewtonRewards}$, the first physics-grounded post-training framework for video generation based on $\textit{verifiable rewards}$. Instead of relying on human or VLM feedback, $\texttt{NewtonRewards}$ extracts $\textit{measurable proxies}$ from generated videos using frozen utility models: optical flow serves as a proxy for velocity, while high-level appearance features serve as a proxy for mass. These proxies enable explicit enforcement of Newtonian structure through two complementary rewards: a Newtonian kinematic constraint enforcing constant-acceleration dynamics, and a mass conservation reward preventing trivial, degenerate solutions. We evaluate $\texttt{NewtonRewards}$ on five Newtonian Motion Primitives (free fall, horizontal/parabolic throw, and ramp sliding down/up) using our newly constructed large-scale benchmark, $\texttt{NewtonBench-60K}$. Across all primitives in visual and physics metrics, $\texttt{NewtonRewards}$ consistently improves physical plausibility, motion smoothness, and temporal coherence over prior post-training methods. It further maintains strong performance under out-of-distribution shifts in height, speed, and friction. Our results show that physics-grounded verifiable rewards offer a scalable path toward physics-aware video generation.
>
---
#### [new 022] Deep Unsupervised Anomaly Detection in Brain Imaging: Large-Scale Benchmarking and Bias Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦脑部MRI图像的无监督异常检测任务，旨在解决现有方法评估碎片化、数据异质性及算法偏差问题。研究构建了大规模多中心基准，系统评估多种算法性能，发现重建类方法效果更优但存在扫描仪、年龄、性别等系统性偏差，强调需改进模型鲁棒性与公平性以推动临床应用。**

- **链接: [https://arxiv.org/pdf/2512.01534v1](https://arxiv.org/pdf/2512.01534v1)**

> **作者:** Alexander Frotscher; Christian F. Baumgartner; Thomas Wolfers
>
> **摘要:** Deep unsupervised anomaly detection in brain magnetic resonance imaging offers a promising route to identify pathological deviations without requiring lesion-specific annotations. Yet, fragmented evaluations, heterogeneous datasets, and inconsistent metrics have hindered progress toward clinical translation. Here, we present a large-scale, multi-center benchmark of deep unsupervised anomaly detection for brain imaging. The training cohort comprised 2,976 T1 and 2,972 T2-weighted scans from healthy individuals across six scanners, with ages ranging from 6 to 89 years. Validation used 92 scans to tune hyperparameters and estimate unbiased thresholds. Testing encompassed 2,221 T1w and 1,262 T2w scans spanning healthy datasets and diverse clinical cohorts. Across all algorithms, the Dice-based segmentation performance varied between 0.03 and 0.65, indicating substantial variability. To assess robustness, we systematically evaluated the impact of different scanners, lesion types and sizes, as well as demographics (age, sex). Reconstruction-based methods, particularly diffusion-inspired approaches, achieved the strongest lesion segmentation performance, while feature-based methods showed greater robustness under distributional shifts. However, systematic biases, such as scanner-related effects, were observed for the majority of algorithms, including that small and low-contrast lesions were missed more often, and that false positives varied with age and sex. Increasing healthy training data yields only modest gains, underscoring that current unsupervised anomaly detection frameworks are limited algorithmically rather than by data availability. Our benchmark establishes a transparent foundation for future research and highlights priorities for clinical translation, including image native pretraining, principled deviation measures, fairness-aware modeling, and robust domain adaptation.
>
---
#### [new 023] MV-TAP: Tracking Any Point in Multi-View Videos
- **分类: cs.CV**

- **简介: 该论文提出MV-TAP，一种用于多视角视频中任意点跟踪的新方法。针对多视角动态场景下轨迹不完整、不准确的问题，利用相机几何与跨视角注意力机制融合多视图时空信息，提升轨迹估计的完整性和可靠性。构建了大规模合成与真实数据集，实验表明其显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.02006v1](https://arxiv.org/pdf/2512.02006v1)**

> **作者:** Jahyeok Koo; Inès Hyeonsu Kim; Mungyeom Kim; Junghyun Park; Seohyun Park; Jaeyeong Kim; Jung Yi; Seokju Cho; Seungryong Kim
>
> **备注:** Project Page: https://cvlab-kaist.github.io/MV-TAP/
>
> **摘要:** Multi-view camera systems enable rich observations of complex real-world scenes, and understanding dynamic objects in multi-view settings has become central to various applications. In this work, we present MV-TAP, a novel point tracker that tracks points across multi-view videos of dynamic scenes by leveraging cross-view information. MV-TAP utilizes camera geometry and a cross-view attention mechanism to aggregate spatio-temporal information across views, enabling more complete and reliable trajectory estimation in multi-view videos. To support this task, we construct a large-scale synthetic training dataset and real-world evaluation sets tailored for multi-view tracking. Extensive experiments demonstrate that MV-TAP outperforms existing point-tracking methods on challenging benchmarks, establishing an effective baseline for advancing research in multi-view point tracking.
>
---
#### [new 024] LISA-3D: Lifting Language-Image Segmentation to 3D via Multi-View Consistency
- **分类: cs.CV**

- **简介: 该论文提出LISA-3D，解决文本驱动3D重建中多视角一致性难题。通过在LISA模型中引入几何感知LoRA，结合冻结的SAM-3D重构器，利用RGB-D序列和可微重投影损失实现跨视角一致的掩码生成，无需额外3D标注，显著提升语言到3D的准确性。**

- **链接: [https://arxiv.org/pdf/2512.01008v1](https://arxiv.org/pdf/2512.01008v1)**

> **作者:** Zhongbin Guo; Jiahe Liu; Wenyu Gao; Yushan Li; Chengzhi Li; Ping Jian
>
> **摘要:** Text-driven 3D reconstruction demands a mask generator that simultaneously understands open-vocabulary instructions and remains consistent across viewpoints. We present LISA-3D, a two-stage framework that lifts language-image segmentation into 3D by retrofitting the instruction-following model LISA with geometry-aware Low-Rank Adaptation (LoRA) layers and reusing a frozen SAM-3D reconstructor. During training we exploit off-the-shelf RGB-D sequences and their camera poses to build a differentiable reprojection loss that enforces cross-view agreement without requiring any additional 3D-text supervision. The resulting masks are concatenated with RGB images to form RGBA prompts for SAM-3D, which outputs Gaussian splats or textured meshes without retraining. Across ScanRefer and Nr3D, LISA-3D improves language-to-3D accuracy by up to +15.6 points over single-view baselines while adapting only 11.6M parameters. The system is modular, data-efficient, and supports zero-shot deployment on unseen categories, providing a practical recipe for language-guided 3D content creation. Our code will be available at https://github.com/binisalegend/LISA-3D.
>
---
#### [new 025] USB: Unified Synthetic Brain Framework for Bidirectional Pathology-Healthy Generation and Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出USB框架，解决脑影像中病理与健康图像配对数据稀缺问题。通过统一的双向生成与编辑机制，实现病理与健康脑图像的联合建模与一致性编辑，提升生成质量与解剖一致性。**

- **链接: [https://arxiv.org/pdf/2512.00269v1](https://arxiv.org/pdf/2512.00269v1)**

> **作者:** Jun Wang; Peirong Liu
>
> **备注:** 16 pages, 17 figures
>
> **摘要:** Understanding the relationship between pathological and healthy brain structures is fundamental to neuroimaging, connecting disease diagnosis and detection with modeling, prediction, and treatment planning. However, paired pathological-healthy data are extremely difficult to obtain, as they rely on pre- and post-treatment imaging, constrained by clinical outcomes and longitudinal data availability. Consequently, most existing brain image generation and editing methods focus on visual quality yet remain domain-specific, treating pathological and healthy image modeling independently. We introduce USB (Unified Synthetic Brain), the first end-to-end framework that unifies bidirectional generation and editing of pathological and healthy brain images. USB models the joint distribution of lesions and brain anatomy through a paired diffusion mechanism and achieves both pathological and healthy image generation. A consistency guidance algorithm further preserves anatomical consistency and lesion correspondence during bidirectional pathology-healthy editing. Extensive experiments on six public brain MRI datasets including healthy controls, stroke, and Alzheimer's patients, demonstrate USB's ability to produce diverse and realistic results. By establishing the first unified benchmark for brain image generation and editing, USB opens opportunities for scalable dataset creation and robust neuroimaging analysis. Code is available at https://github.com/jhuldr/USB.
>
---
#### [new 026] ProvRain: Rain-Adaptive Denoising and Vehicle Detection via MobileNet-UNet and Faster R-CNN
- **分类: cs.CV**

- **简介: 该论文针对夜间雨天环境下车辆检测的噪声干扰问题，提出ProvRain管道。通过轻量级MobileNet-U-Net进行自适应去噪，并结合Faster R-CNN实现精准检测。利用混合数据与课程学习提升鲁棒性，在雨夜场景中显著提升检测准确率与召回率，同时优化图像质量。**

- **链接: [https://arxiv.org/pdf/2512.00073v1](https://arxiv.org/pdf/2512.00073v1)**

> **作者:** Aswinkumar Varathakumaran; Nirmala Paramanandham
>
> **摘要:** Provident vehicle detection has a lot of scope in the detection of vehicle during night time. The extraction of features other than the headlamps of vehicles allows us to detect oncoming vehicles before they appear directly on the camera. However, it faces multiple issues especially in the field of night vision, where a lot of noise caused due to weather conditions such as rain or snow as well as camera conditions. This paper focuses on creating a pipeline aimed at dealing with such noise while at the same time maintaining the accuracy of provident vehicular detection. The pipeline in this paper, ProvRain, uses a lightweight MobileNet-U-Net architecture tuned to generalize to robust weather conditions by using the concept of curricula training. A mix of synthetic as well as available data from the PVDN dataset is used for this. This pipeline is compared to the base Faster RCNN architecture trained on the PVDN dataset to see how much the addition of a denoising architecture helps increase the detection model's performance in rainy conditions. The system boasts an 8.94\% increase in accuracy and a 10.25\% increase in recall in the detection of vehicles in rainy night time frames. Similarly, the custom MobileNet-U-Net architecture that was trained also shows a 10-15\% improvement in PSNR, a 5-6\% increase in SSIM, and upto a 67\% reduction in perceptual error (LPIPS) compared to other transformer approaches.
>
---
#### [new 027] Data-Centric Visual Development for Self-Driving Labs
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自驱动实验室中因数据稀缺导致的视觉模型训练难题，聚焦移液操作的精确检测。提出融合真实与虚拟数据的混合生成策略，通过人机协作采集真实数据，结合条件生成图像扩充数据集，实现类平衡，显著提升泡泡检测模型精度与效率。**

- **链接: [https://arxiv.org/pdf/2512.02018v1](https://arxiv.org/pdf/2512.02018v1)**

> **作者:** Anbang Liu; Guanzhong Hu; Jiayi Wang; Ping Guo; Han Liu
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Self-driving laboratories offer a promising path toward reducing the labor-intensive, time-consuming, and often irreproducible workflows in the biological sciences. Yet their stringent precision requirements demand highly robust models whose training relies on large amounts of annotated data. However, this kind of data is difficult to obtain in routine practice, especially negative samples. In this work, we focus on pipetting, the most critical and precision sensitive action in SDLs. To overcome the scarcity of training data, we build a hybrid pipeline that fuses real and virtual data generation. The real track adopts a human-in-the-loop scheme that couples automated acquisition with selective human verification to maximize accuracy with minimal effort. The virtual track augments the real data using reference-conditioned, prompt-guided image generation, which is further screened and validated for reliability. Together, these two tracks yield a class-balanced dataset that enables robust bubble detection training. On a held-out real test set, a model trained entirely on automatically acquired real images reaches 99.6% accuracy, and mixing real and generated data during training sustains 99.4% accuracy while reducing collection and review load. Our approach offers a scalable and cost-effective strategy for supplying visual feedback data to SDL workflows and provides a practical solution to data scarcity in rare event detection and broader vision tasks.
>
---
#### [new 028] TokenPure: Watermark Removal through Tokenized Appearance and Structural Guidance
- **分类: cs.CV**

- **简介: 该论文提出TokenPure，一种基于扩散Transformer的数字水印去除框架。针对水印去除中内容一致性与去噪彻底性的矛盾，通过视觉与结构令牌联合条件生成，实现高保真无水印图像重建，显著提升去水印效果与视觉质量。**

- **链接: [https://arxiv.org/pdf/2512.01314v1](https://arxiv.org/pdf/2512.01314v1)**

> **作者:** Pei Yang; Yepeng Liu; Kelly Peng; Yuan Gao; Yiren Song
>
> **摘要:** In the digital economy era, digital watermarking serves as a critical basis for ownership proof of massive replicable content, including AI-generated and other virtual assets. Designing robust watermarks capable of withstanding various attacks and processing operations is even more paramount. We introduce TokenPure, a novel Diffusion Transformer-based framework designed for effective and consistent watermark removal. TokenPure solves the trade-off between thorough watermark destruction and content consistency by leveraging token-based conditional reconstruction. It reframes the task as conditional generation, entirely bypassing the initial watermark-carrying noise. We achieve this by decomposing the watermarked image into two complementary token sets: visual tokens for texture and structural tokens for geometry. These tokens jointly condition the diffusion process, enabling the framework to synthesize watermark-free images with fine-grained consistency and structural integrity. Comprehensive experiments show that TokenPure achieves state-of-the-art watermark removal and reconstruction fidelity, substantially outperforming existing baselines in both perceptual quality and consistency.
>
---
#### [new 029] Script: Graph-Structured and Query-Conditioned Semantic Token Pruning for Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型中视觉令牌过多导致的内存与延迟问题，提出无需微调的Script方法。通过图结构冗余剔除与查询相关语义保留双重模块，实现高效精准的令牌剪枝，在14个基准上显著提升效率与精度，最高提速6.8倍、降低10倍计算量。**

- **链接: [https://arxiv.org/pdf/2512.01949v1](https://arxiv.org/pdf/2512.01949v1)**

> **作者:** Zhongyu Yang; Dannong Xu; Wei Pang; Yingfang Yuan
>
> **备注:** Published in Transactions on Machine Learning Research, Project in https://01yzzyu.github.io/script.github.io/
>
> **摘要:** The rapid growth of visual tokens in multimodal large language models (MLLMs) leads to excessive memory consumption and inference latency, especially when handling high-resolution images and videos. Token pruning is a technique used to mitigate this issue by removing redundancy, but existing methods often ignore relevance to the user query or suffer from the limitations of attention mechanisms, reducing their adaptability and effectiveness. To address these challenges, we propose Script, a plug-and-play pruning method that requires no retraining and generalizes across diverse MLLMs. Script comprises two modules: a graph-structured pruning module that removes visually redundant tokens, and a query-conditioned semantic pruning module that preserves query-relevant visual information. Together, they enhance performance on multimodal tasks. Experiments on fourteen benchmarks across image and video understanding tasks show that Script consistently achieves higher model efficiency and predictive accuracy compared to existing pruning methods. On LLaVA-NeXT-7B, it achieves up to 6.8x prefill speedup and 10x FLOP reduction, while retaining 96.88% of the original performance.
>
---
#### [new 030] Generative Action Tell-Tales: Assessing Human Motion in Synthesized Videos
- **分类: cs.CV**

- **简介: 该论文针对视频生成中人类动作真实性的评估难题，提出基于真实动作潜空间的生成式动作判别方法。通过融合骨骼几何与外观特征，构建动作合理性表征，量化生成视频与真实动作分布的距离。在自建多维度基准上显著优于现有方法，更贴近人类感知，推动视频生成质量评估发展。**

- **链接: [https://arxiv.org/pdf/2512.01803v1](https://arxiv.org/pdf/2512.01803v1)**

> **作者:** Xavier Thomas; Youngsun Lim; Ananya Srinivasan; Audrey Zheng; Deepti Ghadiyaram
>
> **摘要:** Despite rapid advances in video generative models, robust metrics for evaluating visual and temporal correctness of complex human actions remain elusive. Critically, existing pure-vision encoders and Multimodal Large Language Models (MLLMs) are strongly appearance-biased, lack temporal understanding, and thus struggle to discern intricate motion dynamics and anatomical implausibilities in generated videos. We tackle this gap by introducing a novel evaluation metric derived from a learned latent space of real-world human actions. Our method first captures the nuances, constraints, and temporal smoothness of real-world motion by fusing appearance-agnostic human skeletal geometry features with appearance-based features. We posit that this combined feature space provides a robust representation of action plausibility. Given a generated video, our metric quantifies its action quality by measuring the distance between its underlying representations and this learned real-world action distribution. For rigorous validation, we develop a new multi-faceted benchmark specifically designed to probe temporally challenging aspects of human action fidelity. Through extensive experiments, we show that our metric achieves substantial improvement of more than 68% compared to existing state-of-the-art methods on our benchmark, performs competitively on established external benchmarks, and has a stronger correlation with human perception. Our in-depth analysis reveals critical limitations in current video generative models and establishes a new standard for advanced research in video generation.
>
---
#### [new 031] HeartFormer: Semantic-Aware Dual-Structure Transformers for 3D Four-Chamber Cardiac Point Cloud Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出HeartFormer，用于从2D cine MRI重建3D四腔心脏点云，解决传统MRI无法提供完整三维结构的问题。创新性地设计双结构注意力网络，融合语义与几何先验，实现高精度多类别点云补全，并构建首个大规模公开数据集HeartCompv1，推动该领域发展。**

- **链接: [https://arxiv.org/pdf/2512.00264v1](https://arxiv.org/pdf/2512.00264v1)**

> **作者:** Zhengda Ma; Abhirup Banerjee
>
> **摘要:** We present the first geometric deep learning framework based on point cloud representation for 3D four-chamber cardiac reconstruction from cine MRI data. This work addresses a long-standing limitation in conventional cine MRI, which typically provides only 2D slice images of the heart, thereby restricting a comprehensive understanding of cardiac morphology and physiological mechanisms in both healthy and pathological conditions. To overcome this, we propose \textbf{HeartFormer}, a novel point cloud completion network that extends traditional single-class point cloud completion to the multi-class. HeartFormer consists of two key components: a Semantic-Aware Dual-Structure Transformer Network (SA-DSTNet) and a Semantic-Aware Geometry Feature Refinement Transformer Network (SA-GFRTNet). SA-DSTNet generates an initial coarse point cloud with both global geometry features and substructure geometry features. Guided by these semantic-geometry representations, SA-GFRTNet progressively refines the coarse output, effectively leveraging both global and substructure geometric priors to produce high-fidelity and geometrically consistent reconstructions. We further construct \textbf{HeartCompv1}, the first publicly available large-scale dataset with 17,000 high-resolution 3D multi-class cardiac meshes and point-clouds, to establish a general benchmark for this emerging research direction. Extensive cross-domain experiments on HeartCompv1 and UK Biobank demonstrate that HeartFormer achieves robust, accurate, and generalizable performance, consistently surpassing state-of-the-art (SOTA) methods. Code and dataset will be released upon acceptance at: https://github.com/10Darren/HeartFormer.
>
---
#### [new 032] TinyViT: Field Deployable Transformer Pipeline for Solar Panel Surface Fault and Severity Screening
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对太阳能板表面故障检测与严重程度评估任务，提出TinyViT轻量级框架，仅用普通可见光图像实现七类故障分类与严重度回归。通过融合Transformer分割、谱空间特征工程与集成回归，克服多模态设备依赖，提升部署经济性与可扩展性，推动光伏健康监测的普适化应用。**

- **链接: [https://arxiv.org/pdf/2512.00117v1](https://arxiv.org/pdf/2512.00117v1)**

> **作者:** Ishwaryah Pandiarajan; Mohamed Mansoor Roomi Sindha; Uma Maheswari Pandyan; Sharafia N
>
> **备注:** 3pages, 2figures,ICGVIP 2025
>
> **摘要:** Sustained operation of solar photovoltaic assets hinges on accurate detection and prioritization of surface faults across vast, geographically distributed modules. While multi modal imaging strategies are popular, they introduce logistical and economic barriers for routine farm level deployment. This work demonstrates that deep learning and classical machine learning may be judiciously combined to achieve robust surface anomaly categorization and severity estimation from planar visible band imagery alone. We introduce TinyViT which is a compact pipeline integrating Transformer based segmentation, spectral-spatial feature engineering, and ensemble regression. The system ingests consumer grade color camera mosaics of PV panels, classifies seven nuanced surface faults, and generates actionable severity grades for maintenance triage. By eliminating reliance on electroluminescence or IR sensors, our method enables affordable, scalable upkeep for resource limited installations, and advances the state of solar health monitoring toward universal field accessibility. Experiments on real public world datasets validate both classification and regression sub modules, achieving accuracy and interpretability competitive with specialized approaches.
>
---
#### [new 033] Odometry Without Correspondence from Inertially Constrained Ruled Surfaces
- **分类: cs.CV**

- **简介: 该论文针对视觉里程计中点对应关系计算昂贵且不准确的问题，提出一种无需点对点匹配的新方法。利用相机运动时直线在图像中形成的惯性约束的直纹面，结合IMU数据降低解空间维度，仅通过点线关联实现3D场景重建与位姿估计。**

- **链接: [https://arxiv.org/pdf/2512.00327v1](https://arxiv.org/pdf/2512.00327v1)**

> **作者:** Chenqi Zhu; Levi Burner; Yiannis Aloimonos
>
> **备注:** 14 pages, 13 figures, 5 tables
>
> **摘要:** Visual odometry techniques typically rely on feature extraction from a sequence of images and subsequent computation of optical flow. This point-to-point correspondence between two consecutive frames can be costly to compute and suffers from varying accuracy, which affects the odometry estimate's quality. Attempts have been made to bypass the difficulties originating from the correspondence problem by adopting line features and fusing other sensors (event camera, IMU) to improve performance, many of which still heavily rely on correspondence. If the camera observes a straight line as it moves, the image of the line sweeps a smooth surface in image-space time. It is a ruled surface and analyzing its shape gives information about odometry. Further, its estimation requires only differentially computed updates from point-to-line associations. Inspired by event cameras' propensity for edge detection, this research presents a novel algorithm to reconstruct 3D scenes and visual odometry from these ruled surfaces. By constraining the surfaces with the inertia measurements from an onboard IMU sensor, the dimensionality of the solution space is greatly reduced.
>
---
#### [new 034] Deep Filament Extraction for 3D Concrete Printing
- **分类: cs.CV**

- **简介: 该论文针对3D混凝土打印中连续沉积丝线（filament）的几何质量控制问题，提出一种自动化质量检测方法。研究解决丝线形状与精度评估难题，构建了不依赖特定传感器的通用工作流程，适用于新鲜或硬化材料的在线及离线检测，提升打印结构的可靠性与精度。**

- **链接: [https://arxiv.org/pdf/2512.00091v1](https://arxiv.org/pdf/2512.00091v1)**

> **作者:** Karam Mawas; Mehdi Maboudi; Pedro Achanccaray; Markus Gerke
>
> **摘要:** The architecture, engineering and construction (AEC) industry is constantly evolving to meet the demand for sustainable and effective design and construction of the built environment. In the literature, two primary deposition techniques for large-scale 3D concrete printing (3DCP) have been described, namely extrusion-based (Contour Crafting-CC) and shotcrete 3D printing (SC3DP) methods. The deposition methods use a digitally controlled nozzle to print material layer by layer. The continuous flow of concrete material used to create the printed structure is called a filament or layer. As these filaments are the essential structure defining the printed object, the filaments' geometry quality control is crucial. This paper presents an automated procedure for quality control (QC) of filaments in extrusion-based and SC3DP printing methods. The paper also describes a workflow that is independent of the sensor used for data acquisition, such as a camera, a structured light system (SLS) or a terrestrial laser scanner (TLS). This method can be used with materials in either the fresh or cured state. Thus, it can be used for online and post-printing QC.
>
---
#### [new 035] Dual-Projection Fusion for Accurate Upright Panorama Generation in Robotic Vision
- **分类: cs.CV**

- **简介: 该论文针对机器人视觉中非正立全景图问题，提出双流角度感知网络，通过融合等距圆柱投影与立方体投影特征，联合估计相机倾角并生成正立全景图。引入高频增强、循环填充与通道注意力机制，提升360°连续性与几何精度。在SUN360和M3D数据集上验证了方法有效性。**

- **链接: [https://arxiv.org/pdf/2512.00911v1](https://arxiv.org/pdf/2512.00911v1)**

> **作者:** Yuhao Shan; Qianyi Yuan; Jingguo Liu; Shigang Li; Jianfeng Li; Tong Chen
>
> **摘要:** Panoramic cameras, capable of capturing a 360-degree field of view, are crucial in robotic vision, particularly in environments with sparse features. However, non-upright panoramas due to unstable robot postures hinder downstream tasks. Traditional IMU-based correction methods suffer from drift and external disturbances, while vision-based approaches offer a promising alternative. This study presents a dual-stream angle-aware generation network that jointly estimates camera inclination angles and reconstructs upright panoramic images. The network comprises a CNN branch that extracts local geometric structures from equirectangular projections and a ViT branch that captures global contextual cues from cubemap projections. These are integrated through a dual-projection adaptive fusion module that aligns spatial features across both domains. To further enhance performance, we introduce a high-frequency enhancement block, circular padding, and channel attention mechanisms to preserve 360° continuity and improve geometric sensitivity. Experiments on the SUN360 and M3D datasets demonstrate that our method outperforms existing approaches in both inclination estimation and upright panorama generation. Ablation studies further validate the contribution of each module and highlight the synergy between the two tasks. The code and related datasets can be found at: https://github.com/YuhaoShine/DualProjectionFusion.
>
---
#### [new 036] TabletopGen: Instance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image
- **分类: cs.CV**

- **简介: 该论文提出TabletopGen，一个无需训练的全自动框架，用于从文本或单图生成高保真、可交互的3D桌面场景。针对现有方法在小尺度密集布局与复杂空间关系建模上的不足，通过实例分割、3D重建与分阶段姿态对齐，实现精准场景合成，显著提升视觉质量、布局准确性和物理合理性。**

- **链接: [https://arxiv.org/pdf/2512.01204v1](https://arxiv.org/pdf/2512.01204v1)**

> **作者:** Ziqian Wang; Yonghao He; Licheng Yang; Wei Zou; Hongxuan Ma; Liu Liu; Wei Sui; Yuxin Guo; Hu Su
>
> **备注:** Project page: https://d-robotics-ai-lab.github.io/TabletopGen.project/
>
> **摘要:** Generating high-fidelity, physically interactive 3D simulated tabletop scenes is essential for embodied AI--especially for robotic manipulation policy learning and data synthesis. However, current text- or image-driven 3D scene generation methods mainly focus on large-scale scenes, struggling to capture the high-density layouts and complex spatial relations that characterize tabletop scenes. To address these challenges, we propose TabletopGen, a training-free, fully automatic framework that generates diverse, instance-level interactive 3D tabletop scenes. TabletopGen accepts a reference image as input, which can be synthesized by a text-to-image model to enhance scene diversity. We then perform instance segmentation and completion on the reference to obtain per-instance images. Each instance is reconstructed into a 3D model followed by canonical coordinate alignment. The aligned 3D models then undergo pose and scale estimation before being assembled into a collision-free, simulation-ready tabletop scene. A key component of our framework is a novel pose and scale alignment approach that decouples the complex spatial reasoning into two stages: a Differentiable Rotation Optimizer for precise rotation recovery and a Top-view Spatial Alignment mechanism for robust translation and scale estimation, enabling accurate 3D reconstruction from 2D reference. Extensive experiments and user studies show that TabletopGen achieves state-of-the-art performance, markedly surpassing existing methods in visual fidelity, layout accuracy, and physical plausibility, capable of generating realistic tabletop scenes with rich stylistic and spatial diversity. Our code will be publicly available.
>
---
#### [new 037] CircleFlow: Flow-Guided Camera Blur Estimation using a Circle Grid Target
- **分类: cs.CV**

- **简介: 该论文提出CircleFlow，用于高精度相机模糊点扩散函数（PSF）估计。针对传统方法在模糊退化建模中的模糊性与病态性难题，利用圆网格靶标编码空间变化的PSF，结合光流引导边缘定位与隐式神经表示，实现图像与PSF联合优化，显著提升估计准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00796v1](https://arxiv.org/pdf/2512.00796v1)**

> **作者:** Jiajian He; Enjie Hu; Shiqi Chen; Tianchen Qiu; Huajun Feng; Zhihai Xu; Yueting Chen
>
> **摘要:** The point spread function (PSF) serves as a fundamental descriptor linking the real-world scene to the captured signal, manifesting as camera blur. Accurate PSF estimation is crucial for both optical characterization and computational vision, yet remains challenging due to the inherent ambiguity and the ill-posed nature of intensity-based deconvolution. We introduce CircleFlow, a high-fidelity PSF estimation framework that employs flow-guided edge localization for precise blur characterization. CircleFlow begins with a structured capture that encodes locally anisotropic and spatially varying PSFs by imaging a circle grid target, while leveraging the target's binary luminance prior to decouple image and kernel estimation. The latent sharp image is then reconstructed through subpixel alignment of an initialized binary structure guided by optical flow, whereas the PSF is modeled as an energy-constrained implicit neural representation. Both components are jointly optimized within a demosaicing-aware differentiable framework, ensuring physically consistent and robust PSF estimation enabled by accurate edge localization. Extensive experiments on simulated and real-world data demonstrate that CircleFlow achieves state-of-the-art accuracy and reliability, validating its effectiveness for practical PSF calibration.
>
---
#### [new 038] Silhouette-based Gait Foundation Model
- **分类: cs.CV**

- **简介: 该论文提出FoundationGait，首个可扩展的自监督步态基础模型，解决步态识别中模型规模小、泛化能力差的问题。通过大规模预训练（超200万步态序列），实现跨数据集、任务和模态的零样本迁移，显著提升步态识别、健康分析等任务性能，推动步态理解向通用化发展。**

- **链接: [https://arxiv.org/pdf/2512.00691v1](https://arxiv.org/pdf/2512.00691v1)**

> **作者:** Dingqiang Ye; Chao Fan; Kartik Narayan; Bingzhe Wu; Chengwen Luo; Jianqiang Li; Vishal M. Patel
>
> **摘要:** Gait patterns play a critical role in human identification and healthcare analytics, yet current progress remains constrained by small, narrowly designed models that fail to scale or generalize. Building a unified gait foundation model requires addressing two longstanding barriers: (a) Scalability. Why have gait models historically failed to follow scaling laws? (b) Generalization. Can one model serve the diverse gait tasks that have traditionally been studied in isolation? We introduce FoundationGait, the first scalable, self-supervised pretraining framework for gait understanding. Its largest version has nearly 0.13 billion parameters and is pretrained on 12 public gait datasets comprising over 2 million walking sequences. Extensive experiments demonstrate that FoundationGait, with or without fine-tuning, performs robustly across a wide spectrum of gait datasets, conditions, tasks (e.g., human identification, scoliosis screening, depression prediction, and attribute estimation), and even input modality. Notably, it achieves 48.0% zero-shot rank-1 accuracy on the challenging in-the-wild Gait3D dataset (1,000 test subjects) and 64.5% on the largest in-the-lab OU-MVLP dataset (5,000+ test subjects), setting a new milestone in robust gait recognition. Coming code and model: https://github.com/ShiqiYu/OpenGait.
>
---
#### [new 039] PhotoFramer: Multi-modal Image Composition Instruction
- **分类: cs.CV**

- **简介: 该论文提出PhotoFramer，一种多模态图像构图指导框架。针对普通用户难以拍出构图良好照片的问题，通过自然语言描述与生成示例图像，提供可操作的构图建议。研究构建了包含平移、缩放、视角变化的多模态数据集，并基于此训练联合处理文本与图像的模型，显著提升构图质量。**

- **链接: [https://arxiv.org/pdf/2512.00993v1](https://arxiv.org/pdf/2512.00993v1)**

> **作者:** Zhiyuan You; Ke Wang; He Zhang; Xin Cai; Jinjin Gu; Tianfan Xue; Chao Dong; Zhoutong Zhang
>
> **摘要:** Composition matters during the photo-taking process, yet many casual users struggle to frame well-composed images. To provide composition guidance, we introduce PhotoFramer, a multi-modal composition instruction framework. Given a poorly composed image, PhotoFramer first describes how to improve the composition in natural language and then generates a well-composed example image. To train such a model, we curate a large-scale dataset. Inspired by how humans take photos, we organize composition guidance into a hierarchy of sub-tasks: shift, zoom-in, and view-change tasks. Shift and zoom-in data are sampled from existing cropping datasets, while view-change data are obtained via a two-stage pipeline. First, we sample pairs with varying viewpoints from multi-view datasets, and train a degradation model to transform well-composed photos into poorly composed ones. Second, we apply this degradation model to expert-taken photos to synthesize poor images to form training pairs. Using this dataset, we finetune a model that jointly processes and generates both text and images, enabling actionable textual guidance with illustrative examples. Extensive experiments demonstrate that textual instructions effectively steer image composition, and coupling them with exemplars yields consistent improvements over exemplar-only baselines. PhotoFramer offers a practical step toward composition assistants that make expert photographic priors accessible to everyday users. Codes, model weights, and datasets have been released in https://zhiyuanyou.github.io/photoframer.
>
---
#### [new 040] StyleYourSmile: Cross-Domain Face Retargeting Without Paired Multi-Style Data
- **分类: cs.CV**

- **简介: 该论文提出StyleYourSmile，解决跨域人脸重定向中缺乏配对多风格数据的问题。通过双编码器与高效数据增强，实现身份与风格的解耦控制，利用扩散模型在无配对数据下完成跨域表情迁移，显著提升身份保真度与迁移质量。**

- **链接: [https://arxiv.org/pdf/2512.01895v1](https://arxiv.org/pdf/2512.01895v1)**

> **作者:** Avirup Dey; Vinay Namboodiri
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** Cross-domain face retargeting requires disentangled control over identity, expressions, and domain-specific stylistic attributes. Existing methods, typically trained on real-world faces, either fail to generalize across domains, need test-time optimizations, or require fine-tuning with carefully curated multi-style datasets to achieve domain-invariant identity representations. In this work, we introduce \textit{StyleYourSmile}, a novel one-shot cross-domain face retargeting method that eliminates the need for curated multi-style paired data. We propose an efficient data augmentation strategy alongside a dual-encoder framework, for extracting domain-invariant identity cues and capturing domain-specific stylistic variations. Leveraging these disentangled control signals, we condition a diffusion model to retarget facial expressions across domains. Extensive experiments demonstrate that \textit{StyleYourSmile} achieves superior identity preservation and retargeting fidelity across a wide range of visual domains.
>
---
#### [new 041] Feed-Forward 3D Gaussian Splatting Compression with Long-Context Modeling
- **分类: cs.CV**

- **简介: 该论文针对3D高斯点云压缩任务，解决现有方法难以建模长程空间依赖的问题。提出基于Morton编码的大规模上下文结构与细粒度自回归熵模型，结合注意力机制的变换编码器，有效捕捉远距离相关性，实现20倍压缩率，性能优于现有通用压缩方法。**

- **链接: [https://arxiv.org/pdf/2512.00877v1](https://arxiv.org/pdf/2512.00877v1)**

> **作者:** Zhening Liu; Rui Song; Yushi Huang; Yingdong Hu; Xinjie Zhang; Jiawei Shao; Zehong Lin; Jun Zhang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a revolutionary 3D representation. However, its substantial data size poses a major barrier to widespread adoption. While feed-forward 3DGS compression offers a practical alternative to costly per-scene per-train compressors, existing methods struggle to model long-range spatial dependencies, due to the limited receptive field of transform coding networks and the inadequate context capacity in entropy models. In this work, we propose a novel feed-forward 3DGS compression framework that effectively models long-range correlations to enable highly compact and generalizable 3D representations. Central to our approach is a large-scale context structure that comprises thousands of Gaussians based on Morton serialization. We then design a fine-grained space-channel auto-regressive entropy model to fully leverage this expansive context. Furthermore, we develop an attention-based transform coding model to extract informative latent priors by aggregating features from a wide range of neighboring Gaussians. Our method yields a $20\times$ compression ratio for 3DGS in a feed-forward inference and achieves state-of-the-art performance among generalizable codecs.
>
---
#### [new 042] TAP-CT: 3D Task-Agnostic Pretraining of Computed Tomography Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TAP-CT，一种用于3D CT影像的通用预训练方法，旨在解决医学影像中基础模型依赖特定任务、需大量微调的问题。通过改进ViT与DINOv2以适配体数据，实现大规模自监督预训练，获得强泛化能力的冻结特征表示，支持低资源下游任务。**

- **链接: [https://arxiv.org/pdf/2512.00872v1](https://arxiv.org/pdf/2512.00872v1)**

> **作者:** Tim Veenboer; George Yiasemis; Eric Marcus; Vivien Van Veldhuizen; Cees G. M. Snoek; Jonas Teuwen; Kevin B. W. Groot Lipman
>
> **备注:** 22 pages, 4 figures, 8 tables
>
> **摘要:** Existing foundation models (FMs) in the medical domain often require extensive fine-tuning or rely on training resource-intensive decoders, while many existing encoders are pretrained with objectives biased toward specific tasks. This illustrates a need for a strong, task-agnostic foundation model that requires minimal fine-tuning beyond feature extraction. In this work, we introduce a suite of task-agnostic pretraining of CT foundation models (TAP-CT): a simple yet effective adaptation of Vision Transformers (ViTs) and DINOv2 for volumetric data, enabling scalable self-supervised pretraining directly on 3D CT volumes. Our approach incorporates targeted modifications to patch embeddings, positional encodings, and volumetric augmentations, making the architecture depth-aware while preserving the simplicity of the underlying architectures. We show that large-scale 3D pretraining on an extensive in-house CT dataset (105K volumes) yields stable, robust frozen representations that generalize strongly across downstream tasks. To promote transparency and reproducibility, and to establish a powerful, low-resource baseline for future research in medical imaging, we will release all pretrained models, experimental configurations, and downstream benchmark code at https://huggingface.co/fomofo/tap-ct-b-3d.
>
---
#### [new 043] \textit{ViRectify}: A Challenging Benchmark for Video Reasoning Correction with Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出ViRectify，一个用于视频推理纠错的基准，旨在评估多模态大模型在动态感知、科学推理和具身决策中的细粒度纠错能力。通过构建3万+实例数据集与轨迹证据驱动的纠正框架，揭示模型纠错偏差，推动反射学习。**

- **链接: [https://arxiv.org/pdf/2512.01424v1](https://arxiv.org/pdf/2512.01424v1)**

> **作者:** Xusen Hei; Jiali Chen; Jinyu Yang; Mengchen Zhao; Yi Cai
>
> **备注:** 22 pages, 11 figures
>
> **摘要:** As multimodal large language models (MLLMs) frequently exhibit errors in complex video reasoning scenarios, correcting these errors is critical for uncovering their weaknesses and improving performance. However, existing benchmarks lack systematic evaluation of MLLMs' ability to identify and correct these video reasoning errors. To bridge this gap, we propose \textit{ViRectify}, a comprehensive benchmark to evaluate their fine-grained correction capability. Through an AI-assisted annotation pipeline with human verification, we construct a dataset of over 30\textit{K} instances spanning dynamic perception, scientific reasoning, and embodied decision-making domains. In \textit{ViRectify}, we challenge MLLMs to perform step-wise error identification and generate rationales with key video evidence grounding. In addition, we further propose the trajectory evidence-driven correction framework, comprising step-wise error trajectory and reward modeling on visual evidence-grounded correction. It encourages the model to explicitly concentrate on error propagation and key timestamps for correction. Extensive evaluation across 16 advanced MLLMs demonstrates that our \textit{ViRectify} serves as a challenging testbed, where GPT-5 achieves only 31.94\% correction accuracy. Our framework enables a Qwen2.5-VL-7B to consistently outperform the variants of 72B on \textit{ViRectify}, showing the effectiveness of our approach. Further analysis uncovers systematic asymmetries in error correction across models, and our dataset is also a valuable data resource to perform reflection learning. We believe \textit{ViRectify} provides a new direction for comprehensively evaluating the advanced MLLMs in video reasoning.
>
---
#### [new 044] Semantic-aware Random Convolution and Source Matching for Domain Generalization in Medical Image Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对医学图像分割中的单源域泛化问题，提出SRCSM方法。通过语义感知的随机卷积增强源域多样性，并在测试时对目标域图像进行强度映射以缩小域差距，实现无需目标域数据的跨模态、跨中心泛化，显著提升分割性能，达到新基准水平。**

- **链接: [https://arxiv.org/pdf/2512.01510v1](https://arxiv.org/pdf/2512.01510v1)**

> **作者:** Franz Thaler; Martin Urschler; Mateusz Kozinski; Matthias AF Gsell; Gernot Plank; Darko Stern
>
> **备注:** Preprint submitted to Computer Methods and Programs in Biomedicine (currently under revision)
>
> **摘要:** We tackle the challenging problem of single-source domain generalization (DG) for medical image segmentation. To this end, we aim for training a network on one domain (e.g., CT) and directly apply it to a different domain (e.g., MR) without adapting the model and without requiring images or annotations from the new domain during training. We propose a novel method for promoting DG when training deep segmentation networks, which we call SRCSM. During training, our method diversifies the source domain through semantic-aware random convolution, where different regions of a source image are augmented differently, based on their annotation labels. At test-time, we complement the randomization of the training domain via mapping the intensity of target domain images, making them similar to source domain data. We perform a comprehensive evaluation on a variety of cross-modality and cross-center generalization settings for abdominal, whole-heart and prostate segmentation, where we outperform previous DG techniques in a vast majority of experiments. Additionally, we also investigate our method when training on whole-heart CT or MR data and testing on the diastolic and systolic phase of cine MR data captured with different scanner hardware, where we make a step towards closing the domain gap in this even more challenging setting. Overall, our evaluation shows that SRCSM can be considered a new state-of-the-art in DG for medical image segmentation and, moreover, even achieves a segmentation performance that matches the performance of the in-domain baseline in several settings.
>
---
#### [new 045] SARL: Spatially-Aware Self-Supervised Representation Learning for Visuo-Tactile Perception
- **分类: cs.CV**

- **简介: 该论文针对接触丰富的机器人操作任务，解决视觉-触觉融合数据中传统自监督学习忽略空间结构的问题。提出SARL框架，通过三个地图级损失保持特征的空间一致性，提升几何敏感任务性能，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.01908v1](https://arxiv.org/pdf/2512.01908v1)**

> **作者:** Gurmeher Khurana; Lan Wei; Dandan Zhang
>
> **摘要:** Contact-rich robotic manipulation requires representations that encode local geometry. Vision provides global context but lacks direct measurements of properties such as texture and hardness, whereas touch supplies these cues. Modern visuo-tactile sensors capture both modalities in a single fused image, yielding intrinsically aligned inputs that are well suited to manipulation tasks requiring visual and tactile information. Most self-supervised learning (SSL) frameworks, however, compress feature maps into a global vector, discarding spatial structure and misaligning with the needs of manipulation. To address this, we propose SARL, a spatially-aware SSL framework that augments the Bootstrap Your Own Latent (BYOL) architecture with three map-level objectives, including Saliency Alignment (SAL), Patch-Prototype Distribution Alignment (PPDA), and Region Affinity Matching (RAM), to keep attentional focus, part composition, and geometric relations consistent across views. These losses act on intermediate feature maps, complementing the global objective. SARL consistently outperforms nine SSL baselines across six downstream tasks with fused visual-tactile data. On the geometry-sensitive edge-pose regression task, SARL achieves a Mean Absolute Error (MAE) of 0.3955, a 30% relative improvement over the next-best SSL method (0.5682 MAE) and approaching the supervised upper bound. These findings indicate that, for fused visual-tactile data, the most effective signal is structured spatial equivariance, in which features vary predictably with object geometry, which enables more capable robotic perception.
>
---
#### [new 046] Quantum-Inspired Spectral Geometry for Neural Operator Equivalence and Structured Pruning
- **分类: cs.CV**

- **简介: 该论文针对资源受限环境下多模态模型的算子冗余与异构硬件适配问题，提出量子启发的谱几何框架，通过奇异值谱在布洛赫超球上的表示，建立谱距离与函数等价性的严格关联，并据此设计度量驱动的结构化剪枝方法，实现跨模态、跨架构算子替换与高效压缩。**

- **链接: [https://arxiv.org/pdf/2512.00880v1](https://arxiv.org/pdf/2512.00880v1)**

> **作者:** Haijian Shao; Wei Liu; Xing Deng
>
> **备注:** 6 pages, 1 figure, preliminary version; concepts and simulation experiments only
>
> **摘要:** The rapid growth of multimodal intelligence on resource-constrained and heterogeneous domestic hardware exposes critical bottlenecks: multimodal feature heterogeneity, real-time requirements in dynamic scenarios, and hardware-specific operator redundancy. This work introduces a quantum-inspired geometric framework for neural operators that represents each operator by its normalized singular value spectrum on the Bloch hypersphere. We prove a tight spectral-to-functional equivalence theorem showing that vanishing Fubini--Study/Wasserstein-2 distance implies provable functional closeness, establishing the first rigorous foundation for cross-modal and cross-architecture operator substitutability. Based on this metric, we propose Quantum Metric-Driven Functional Redundancy Graphs (QM-FRG) and one-shot structured pruning. Controlled simulation validates the superiority of the proposed metric over magnitude and random baselines. An extensive experimental validation on large-scale multimodal transformers and domestic heterogeneous hardware (Huawei Ascend, Cambricon MLU, Kunlunxin) hardware is deferred to an extended journal version currently in preparation.
>
---
#### [new 047] XAI-Driven Skin Disease Classification: Leveraging GANs to Augment ResNet-50 Performance
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对皮肤病变多分类诊断中的数据不平衡与深度学习模型“黑箱”问题，提出基于DCGAN数据增强与微调ResNet-50的CAD系统，并结合LIME、SHAP实现可解释性。有效提升分类准确率（92.50%）与可解释性，推动临床安全应用。**

- **链接: [https://arxiv.org/pdf/2512.00626v1](https://arxiv.org/pdf/2512.00626v1)**

> **作者:** Kim Gerard A. Villanueva; Priyanka Kumar
>
> **摘要:** Accurate and timely diagnosis of multi-class skin lesions is hampered by subjective methods, inherent data imbalance in datasets like HAM10000, and the "black box" nature of Deep Learning (DL) models. This study proposes a trustworthy and highly accurate Computer-Aided Diagnosis (CAD) system to overcome these limitations. The approach utilizes Deep Convolutional Generative Adversarial Networks (DCGANs) for per class data augmentation to resolve the critical class imbalance problem. A fine-tuned ResNet-50 classifier is then trained on the augmented dataset to classify seven skin disease categories. Crucially, LIME and SHAP Explainable AI (XAI) techniques are integrated to provide transparency by confirming that predictions are based on clinically relevant features like irregular morphology. The system achieved a high overall Accuracy of 92.50 % and a Macro-AUC of 98.82 %, successfully outperforming various prior benchmarked architectures. This work successfully validates a verifiable framework that combines high performance with the essential clinical interpretability required for safe diagnostic deployment. Future research should prioritize enhancing discrimination for critical categories, such as Melanoma NOS (F1-Score is 0.8602).
>
---
#### [new 048] SwiftVLA: Unlocking Spatiotemporal Dynamics for Lightweight VLA Models at Minimal Overhead
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对轻量级视觉-语言-动作（VLA）模型在时空推理能力不足的问题，提出SwiftVLA架构。通过4D视觉几何变换器与时间缓存提取4D特征，引入融合令牌增强多模态表示，并采用掩码重建策略使模型学习有效4D表征，最终实现高效推理。**

- **链接: [https://arxiv.org/pdf/2512.00903v1](https://arxiv.org/pdf/2512.00903v1)**

> **作者:** Chaojun Ni; Cheng Chen; Xiaofeng Wang; Zheng Zhu; Wenzhao Zheng; Boyuan Wang; Tianrun Chen; Guosheng Zhao; Haoyun Li; Zhehao Dong; Qiang Zhang; Yun Ye; Yang Wang; Guan Huang; Wenjun Mei
>
> **摘要:** Vision-Language-Action (VLA) models built on pretrained Vision-Language Models (VLMs) show strong potential but are limited in practicality due to their large parameter counts. To mitigate this issue, using a lightweight VLM has been explored, but it compromises spatiotemporal reasoning. Although some methods suggest that incorporating additional 3D inputs can help, they usually rely on large VLMs to fuse 3D and 2D inputs and still lack temporal understanding. Therefore, we propose SwiftVLA, an architecture that enhances a compact model with 4D understanding while preserving design efficiency. Specifically, our approach features a pretrained 4D visual geometry transformer with a temporal cache that extracts 4D features from 2D images. Then, to enhance the VLM's ability to exploit both 2D images and 4D features, we introduce Fusion Tokens, a set of learnable tokens trained with a future prediction objective to generate unified representations for action generation. Finally, we introduce a mask-and-reconstruct strategy that masks 4D inputs to the VLM and trains the VLA to reconstruct them, enabling the VLM to learn effective 4D representations and allowing the 4D branch to be dropped at inference with minimal performance loss. Experiments in real and simulated environments show that SwiftVLA outperforms lightweight baselines and rivals VLAs up to 7 times larger, achieving comparable performance on edge devices while being 18 times faster and reducing memory footprint by 12 times.
>
---
#### [new 049] S$^2$-MLLM: Boosting Spatial Reasoning Capability of MLLMs for 3D Visual Grounding with Structural Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D视觉定位任务，解决多模态大模型在缺乏3D结构感知下的空间推理能力不足问题。提出S²-MLLM框架，通过隐式结构引导与结构增强模块，利用3D重建的结构意识实现高效空间理解，无需依赖低效的点云渲染，在多个数据集上实现更优性能。**

- **链接: [https://arxiv.org/pdf/2512.01223v1](https://arxiv.org/pdf/2512.01223v1)**

> **作者:** Beining Xu; Siting Zhu; Zhao Jin; Junxian Li; Hesheng Wang
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** 3D Visual Grounding (3DVG) focuses on locating objects in 3D scenes based on natural language descriptions, serving as a fundamental task for embodied AI and robotics. Recent advances in Multi-modal Large Language Models (MLLMs) have motivated research into extending them to 3DVG. However, MLLMs primarily process 2D visual inputs and struggle with understanding 3D spatial structure of scenes solely from these limited perspectives. Existing methods mainly utilize viewpoint-dependent rendering of reconstructed point clouds to provide explicit structural guidance for MLLMs in 3DVG tasks, leading to inefficiency and limited spatial reasoning. To address this issue, we propose S$^2$-MLLM, an efficient framework that enhances spatial reasoning in MLLMs through implicit spatial reasoning. We introduce a spatial guidance strategy that leverages the structure awareness of feed-forward 3D reconstruction. By acquiring 3D structural understanding during training, our model can implicitly reason about 3D scenes without relying on inefficient point cloud reconstruction. Moreover, we propose a structure-enhanced module (SE), which first employs intra-view and inter-view attention mechanisms to capture dependencies within views and correspondences across views. The module further integrates multi-level position encoding to associate visual representations with spatial positions and viewpoint information, enabling more accurate structural understanding. Extensive experiments demonstrate that S$^2$-MLLM unifies superior performance, generalization, and efficiency, achieving significant performance over existing methods across the ScanRefer, Nr3D, and Sr3D datasets. Code will be available upon acceptance.
>
---
#### [new 050] ForamDeepSlice: A High-Accuracy Deep Learning Framework for Foraminifera Species Classification from 2D Micro-CT Slices
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对微古生物分类中人工识别效率低的问题，提出ForamDeepSlice框架，利用2D微CT切片实现12种有孔虫的高精度自动分类。通过构建高质量数据集与基于Specimen-level划分的训练流程，结合CNN集成模型，实现95.64%测试准确率，并开发交互式可视化工具支持实时分类与3D匹配。**

- **链接: [https://arxiv.org/pdf/2512.00912v1](https://arxiv.org/pdf/2512.00912v1)**

> **作者:** Abdelghafour Halimi; Ali Alibrahim; Didier Barradas-Bautista; Ronell Sicat; Abdulkader M. Afifi
>
> **摘要:** This study presents a comprehensive deep learning pipeline for the automated classification of 12 foraminifera species using 2D micro-CT slices derived from 3D scans. We curated a scientifically rigorous dataset comprising 97 micro-CT scanned specimens across 27 species, selecting 12 species with sufficient representation for robust machine learning. To ensure methodological integrity and prevent data leakage, we employed specimen-level data splitting, resulting in 109,617 high-quality 2D slices (44,103 for training, 14,046 for validation, and 51,468 for testing). We evaluated seven state-of-the-art 2D convolutional neural network (CNN) architectures using transfer learning. Our final ensemble model, combining ConvNeXt-Large and EfficientNetV2-Small, achieved a test accuracy of 95.64%, with a top-3 accuracy of 99.6% and an area under the ROC curve (AUC) of 0.998 across all species. To facilitate practical deployment, we developed an interactive advanced dashboard that supports real-time slice classification and 3D slice matching using advanced similarity metrics, including SSIM, NCC, and the Dice coefficient. This work establishes new benchmarks for AI-assisted micropaleontological identification and provides a fully reproducible framework for foraminifera classification research, bridging the gap between deep learning and applied geosciences.
>
---
#### [new 051] ViscNet: Vision-Based In-line Viscometry for Fluid Mixing Process
- **分类: cs.CV**

- **简介: 该论文提出ViscNet，一种基于视觉的在线粘度测量方法，旨在解决传统粘度计侵入性强、依赖实验室环境的问题。通过分析光通过动态自由表面时背景图案的光学畸变，实现非接触式粘度估计，结合不确定性量化提升可靠性，适用于复杂工业过程监控。**

- **链接: [https://arxiv.org/pdf/2512.01268v1](https://arxiv.org/pdf/2512.01268v1)**

> **作者:** Jongwon Sohn; Juhyeon Moon; Hyunjoon Jung; Jaewook Nam
>
> **摘要:** Viscosity measurement is essential for process monitoring and autonomous laboratory operation, yet conventional viscometers remain invasive and require controlled laboratory environments that differ substantially from real process conditions. We present a computer-vision-based viscometer that infers viscosity by exploiting how a fixed background pattern becomes optically distorted as light refracts through the mixing-driven, continuously deforming free surface. Under diverse lighting conditions, the system achieves a mean absolute error of 0.113 in log m2 s^-1 units for regression and reaches up to 81% accuracy in viscosity-class prediction. Although performance declines for classes with closely clustered viscosity values, a multi-pattern strategy improves robustness by providing enriched visual cues. To ensure sensor reliability, we incorporate uncertainty quantification, enabling viscosity predictions with confidence estimates. This stand-off viscometer offers a practical, automation-ready alternative to existing viscometry methods.
>
---
#### [new 052] Physical ID-Transfer Attacks against Multi-Object Tracking via Adversarial Trajectory
- **分类: cs.CV**

- **简介: 该论文针对多目标跟踪（MOT）系统，提出一种物理域的在线ID转移攻击方法AdvTraj。通过生成对抗性轨迹，使攻击者身份转移至目标物体，干扰跟踪系统的关联逻辑。工作包括仿真验证、攻击模式分析及可执行的通用对抗行为设计，揭示了SOTA MOT系统在对象关联阶段的脆弱性。**

- **链接: [https://arxiv.org/pdf/2512.01934v1](https://arxiv.org/pdf/2512.01934v1)**

> **作者:** Chenyi Wang; Yanmao Man; Raymond Muller; Ming Li; Z. Berkay Celik; Ryan Gerdes; Jonathan Petit
>
> **备注:** Accepted to Annual Computer Security Applications Conference (ACSAC) 2024
>
> **摘要:** Multi-Object Tracking (MOT) is a critical task in computer vision, with applications ranging from surveillance systems to autonomous driving. However, threats to MOT algorithms have yet been widely studied. In particular, incorrect association between the tracked objects and their assigned IDs can lead to severe consequences, such as wrong trajectory predictions. Previous attacks against MOT either focused on hijacking the trackers of individual objects, or manipulating the tracker IDs in MOT by attacking the integrated object detection (OD) module in the digital domain, which are model-specific, non-robust, and only able to affect specific samples in offline datasets. In this paper, we present AdvTraj, the first online and physical ID-manipulation attack against tracking-by-detection MOT, in which an attacker uses adversarial trajectories to transfer its ID to a targeted object to confuse the tracking system, without attacking OD. Our simulation results in CARLA show that AdvTraj can fool ID assignments with 100% success rate in various scenarios for white-box attacks against SORT, which also have high attack transferability (up to 93% attack success rate) against state-of-the-art (SOTA) MOT algorithms due to their common design principles. We characterize the patterns of trajectories generated by AdvTraj and propose two universal adversarial maneuvers that can be performed by a human walker/driver in daily scenarios. Our work reveals under-explored weaknesses in the object association phase of SOTA MOT systems, and provides insights into enhancing the robustness of such systems.
>
---
#### [new 053] Generalized Medical Phrase Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出广义医学短语定位（GMPG）任务，解决传统系统仅支持单区域定位且无法处理非诊断、多区域或不可定位短语的问题。作者构建MedGrounder模型，采用两阶段训练，实现零样本迁移，有效定位多区域及非地面短语，减少人工标注依赖，并可与报告生成器组合生成带标注报告。**

- **链接: [https://arxiv.org/pdf/2512.01085v1](https://arxiv.org/pdf/2512.01085v1)**

> **作者:** Wenjun Zhang; Shekhar S. Chandra; Aaron Nicolson
>
> **摘要:** Medical phrase grounding (MPG) maps textual descriptions of radiological findings to corresponding image regions. These grounded reports are easier to interpret, especially for non-experts. Existing MPG systems mostly follow the referring expression comprehension (REC) paradigm and return exactly one bounding box per phrase. Real reports often violate this assumption. They contain multi-region findings, non-diagnostic text, and non-groundable phrases, such as negations or descriptions of normal anatomy. Motivated by this, we reformulate the task as generalised medical phrase grounding (GMPG), where each sentence is mapped to zero, one, or multiple scored regions. To realise this formulation, we introduce the first GMPG model: MedGrounder. We adopted a two-stage training regime: pre-training on report sentence--anatomy box alignment datasets and fine-tuning on report sentence--human annotated box datasets. Experiments on PadChest-GR and MS-CXR show that MedGrounder achieves strong zero-shot transfer and outperforms REC-style and grounded report generation baselines on multi-region and non-groundable phrases, while using far fewer human box annotations. Finally, we show that MedGrounder can be composed with existing report generators to produce grounded reports without retraining the generator.
>
---
#### [new 054] CAR-Net: A Cascade Refinement Network for Rotational Motion Deblurring under Angle Information Uncertainty
- **分类: cs.CV**

- **简介: 该论文针对旋转运动模糊图像的去模糊任务，解决在模糊角度信息不准确的半盲场景下的去模糊难题。提出CAR-Net架构，通过频域反演初始化，并设计级联精炼模块逐步修正残差，提升细节恢复效果；可选角度检测模块支持端到端训练，增强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00700v1](https://arxiv.org/pdf/2512.00700v1)**

> **作者:** Ka Chung Lai; Ahmet Cetinkaya
>
> **备注:** Accepted to AAIML 2026
>
> **摘要:** We propose a new neural network architecture called CAR-net (CAscade Refinement Network) to deblur images that are subject to rotational motion blur. Our architecture is specifically designed for the semi-blind scenarios where only noisy information of the rotational motion blur angle is available. The core of our approach is progressive refinement process that starts with an initial deblurred estimate obtained from frequency-domain inversion; A series of refinement stages take the current deblurred image to predict and apply residual correction to the current estimate, progressively suppressing artifacts and restoring fine details. To handle parameter uncertainty, our architecture accommodates an optional angle detection module which can be trained end-to-end with refinement modules. We provide a detailed description of our architecture and illustrate its efficiency through experiments using both synthetic and real-life images. Our code and model as well as the links to the datasets are available at https://github.com/tony123105/CAR-Net
>
---
#### [new 055] Cross-Domain Validation of a Resection-Trained Self-Supervised Model on Multicentre Mesothelioma Biopsies
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决小样本活检中恶性间皮瘤亚型分类与预后预测难题。研究提出一种基于切除标本训练的自监督模型，经跨中心验证，可有效应用于活检样本，精准识别形态特征，实现亚型分类与生存预测，推动AI在真实临床场景中的应用。**

- **链接: [https://arxiv.org/pdf/2512.01681v1](https://arxiv.org/pdf/2512.01681v1)**

> **作者:** Farzaneh Seyedshahi; Francesca Damiola; Sylvie Lantuejoul; Ke Yuan; John Le Quesne
>
> **摘要:** Accurate subtype classification and outcome prediction in mesothelioma are essential for guiding therapy and patient care. Most computational pathology models are trained on large tissue images from resection specimens, limiting their use in real-world settings where small biopsies are common. We show that a self-supervised encoder trained on resection tissue can be applied to biopsy material, capturing meaningful morphological patterns. Using these patterns, the model can predict patient survival and classify tumor subtypes. This approach demonstrates the potential of AI-driven tools to support diagnosis and treatment planning in mesothelioma.
>
---
#### [new 056] Exploring Automated Recognition of Instructional Activity and Discourse from Multimodal Classroom Data
- **分类: cs.CV**

- **简介: 该论文聚焦于课堂多模态数据中的教学活动与话语自动识别任务，旨在解决传统人工标注耗时费力、难以规模化的问题。研究构建了视频与文本双通道分析框架，采用微调模型与上下文增强策略，在大规模标注数据上实现高精度识别，验证了自动化课堂分析的可行性。**

- **链接: [https://arxiv.org/pdf/2512.00087v1](https://arxiv.org/pdf/2512.00087v1)**

> **作者:** Ivo Bueno; Ruikun Hou; Babette Bühler; Tim Fütterer; James Drimalla; Jonathan Kyle Foster; Peter Youngs; Peter Gerjets; Ulrich Trautwein; Enkelejda Kasneci
>
> **备注:** This article has been accepted for publication in the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Observation of classroom interactions can provide concrete feedback to teachers, but current methods rely on manual annotation, which is resource-intensive and hard to scale. This work explores AI-driven analysis of classroom recordings, focusing on multimodal instructional activity and discourse recognition as a foundation for actionable feedback. Using a densely annotated dataset of 164 hours of video and 68 lesson transcripts, we design parallel, modality-specific pipelines. For video, we evaluate zero-shot multimodal LLMs, fine-tuned vision-language models, and self-supervised video transformers on 24 activity labels. For transcripts, we fine-tune a transformer-based classifier with contextualized inputs and compare it against prompting-based LLMs on 19 discourse labels. To handle class imbalance and multi-label complexity, we apply per-label thresholding, context windows, and imbalance-aware loss functions. The results show that fine-tuned models consistently outperform prompting-based approaches, achieving macro-F1 scores of 0.577 for video and 0.460 for transcripts. These results demonstrate the feasibility of automated classroom analysis and establish a foundation for scalable teacher feedback systems.
>
---
#### [new 057] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval
- **分类: cs.CV**

- **简介: 该论文针对视频中基于自然语言查询的片段检索任务，解决传统方法在细粒度对齐与不确定性估计上的不足。提出DEMR框架，通过反射翻转融合块与查询重建增强跨模态对齐和文本敏感性，并引入几何正则化实现不确定性的自适应调整，显著提升检索的准确性、鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.00953v1](https://arxiv.org/pdf/2512.00953v1)**

> **作者:** Haojian Huang; Kaijing Ma; Jin Chen; Haodong Chen; Zhou Wu; Xianghao Zang; Han Fang; Chao Ban; Hao Sun; Mulin Chen; Zhongjiang He
>
> **备注:** Accepted by AAAI 2026, 10 pages, 9 figures, 5 tables
>
> **摘要:** In the domain of moment retrieval, accurately identifying temporal segments within videos based on natural language queries remains challenging. Traditional methods often employ pre-trained models that struggle with fine-grained information and deterministic reasoning, leading to difficulties in aligning with complex or ambiguous moments. To overcome these limitations, we explore Deep Evidential Regression (DER) to construct a vanilla Evidential baseline. However, this approach encounters two major issues: the inability to effectively handle modality imbalance and the structural differences in DER's heuristic uncertainty regularizer, which adversely affect uncertainty estimation. This misalignment results in high uncertainty being incorrectly associated with accurate samples rather than challenging ones. Our observations indicate that existing methods lack the adaptability required for complex video scenarios. In response, we propose Debiased Evidential Learning for Moment Retrieval (DEMR), a novel framework that incorporates a Reflective Flipped Fusion (RFF) block for cross-modal alignment and a query reconstruction task to enhance text sensitivity, thereby reducing bias in uncertainty estimation. Additionally, we introduce a Geom-regularizer to refine uncertainty predictions, enabling adaptive alignment with difficult moments and improving retrieval accuracy. Extensive testing on standard datasets and debiased datasets ActivityNet-CD and Charades-CD demonstrates significant enhancements in effectiveness, robustness, and interpretability, positioning our approach as a promising solution for temporal-semantic robustness in moment retrieval. The code is publicly available at https://github.com/KaijingOfficial/DEMR.
>
---
#### [new 058] Bridging the Scale Gap: Balanced Tiny and General Object Detection in Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文针对遥感图像中微小与一般尺度目标检测不平衡问题，提出ScaleBridge-Det框架。通过自适应路由的多专家注意力模块和密度引导查询分配机制，实现多尺度特征融合与资源动态分配，显著提升对密集微小目标和大目标的联合检测性能，首次构建面向微小目标的大规模检测模型。**

- **链接: [https://arxiv.org/pdf/2512.01665v1](https://arxiv.org/pdf/2512.01665v1)**

> **作者:** Zhicheng Zhao; Yin Huang; Lingma Sun; Chenglong Li; Jin Tang
>
> **摘要:** Tiny object detection in remote sensing imagery has attracted significant research interest in recent years. Despite recent progress, achieving balanced detection performance across diverse object scales remains a formidable challenge, particularly in scenarios where dense tiny objects and large objects coexist. Although large foundation models have revolutionized general vision tasks, their application to tiny object detection remains unexplored due to the extreme scale variation and density distribution inherent to remote sensing imagery. To bridge this scale gap, we propose ScaleBridge-Det, to the best of our knowledge, the first large detection framework designed for tiny objects, which could achieve balanced performance across diverse scales through scale-adaptive expert routing and density-guided query allocation. Specifically, we introduce a Routing-Enhanced Mixture Attention (REM) module that dynamically selects and fuses scale-specific expert features via adaptive routing to address the tendency of standard MoE models to favor dominant scales. REM generates complementary and discriminative multi-scale representations suitable for both tiny and large objects. Furthermore, we present a Density-Guided Dynamic Query (DGQ) module that predicts object density to adaptively adjust query positions and numbers, enabling efficient resource allocation for objects of varying scales. The proposed framework allows ScaleBridge-Det to simultaneously optimize performance for both dense tiny and general objects without trade-offs. Extensive experiments on benchmark and cross-domain datasets demonstrate that ScaleBridge-Det achieves state-of-the-art performance on AI-TOD-V2 and DTOD, while exhibiting superior cross-domain robustness on VisDrone.
>
---
#### [new 059] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出StreamGaze，首个评估多模态大模型在流式视频中利用眼动信号进行时序推理与主动理解的基准。针对现有工作缺乏对眼动引导推理的评测，构建了融合眼动轨迹的时空对齐问答数据集，揭示当前模型在眼动感知、意图预测上的显著不足，并提供深入分析以指导未来研究。**

- **链接: [https://arxiv.org/pdf/2512.01707v1](https://arxiv.org/pdf/2512.01707v1)**

> **作者:** Daeun Lee; Subhojyoti Mukherjee; Branislav Kveton; Ryan A. Rossi; Viet Dac Lai; Seunghyun Yoon; Trung Bui; Franck Dernoncourt; Mohit Bansal
>
> **备注:** Project page: https://streamgaze.github.io/
>
> **摘要:** Streaming video understanding requires models not only to process temporally incoming frames, but also to anticipate user intention for realistic applications like AR glasses. While prior streaming benchmarks evaluate temporal reasoning, none measure whether MLLMs can interpret or leverage human gaze signals within a streaming setting. To fill this gap, we introduce StreamGaze, the first benchmark designed to evaluate how effectively MLLMs use gaze for temporal and proactive reasoning in streaming videos. StreamGaze introduces gaze-guided past, present, and proactive tasks that comprehensively evaluate streaming video understanding. These tasks assess whether models can use real-time gaze to follow shifting attention and infer user intentions from only past and currently observed frames. To build StreamGaze, we develop a gaze-video QA generation pipeline that aligns egocentric videos with raw gaze trajectories via fixation extraction, region-specific visual prompting, and scanpath construction. This pipeline produces spatio-temporally grounded QA pairs that closely reflect human perceptual dynamics. Across all StreamGaze tasks, we observe substantial performance gaps between state-of-the-art MLLMs and human performance, revealing fundamental limitations in gaze-based temporal reasoning, intention modeling, and proactive prediction. We further provide detailed analyses of gaze-prompting strategies, reasoning behaviors, and task-specific failure modes, offering deeper insight into why current MLLMs struggle and what capabilities future models must develop. All data and code will be publicly released to support continued research in gaze-guided streaming video understanding.
>
---
#### [new 060] DL-CapsNet: A Deep and Light Capsule Network
- **分类: cs.CV**

- **简介: 该论文提出一种深度轻量级胶囊网络DL-CapsNet，用于图像分类任务。针对传统CapsNet参数多、计算复杂的问题，设计多层胶囊结构与胶囊汇总层，降低参数量，提升训练与推理速度，同时保持高准确率，有效处理高类别复杂数据。**

- **链接: [https://arxiv.org/pdf/2512.00061v1](https://arxiv.org/pdf/2512.00061v1)**

> **作者:** Pouya Shiri; Amirali Baniasadi
>
> **摘要:** Capsule Network (CapsNet) is among the promising classifiers and a possible successor of the classifiers built based on Convolutional Neural Network (CNN). CapsNet is more accurate than CNNs in detecting images with overlapping categories and those with applied affine transformations. In this work, we propose a deep variant of CapsNet consisting of several capsule layers. In addition, we design the Capsule Summarization layer to reduce the complexity by reducing the number of parameters. DL-CapsNet, while being highly accurate, employs a small number of parameters and delivers faster training and inference. DL-CapsNet can process complex datasets with a high number of categories.
>
---
#### [new 061] Structured Context Learning for Generic Event Boundary Detection
- **分类: cs.CV**

- **简介: 该论文针对通用事件边界检测任务，解决视频中事件边界难以准确识别的问题。提出结构化上下文学习方法，通过结构化序列划分（SPoS）提供有序上下文，结合相似性计算与轻量卷积网络，实现端到端训练。方法高效灵活，优于现有模型。**

- **链接: [https://arxiv.org/pdf/2512.00475v1](https://arxiv.org/pdf/2512.00475v1)**

> **作者:** Xin Gu; Congcong Li; Xinyao Wang; Dexiang Hong; Libo Zhang; Tiejian Luo; Longyin Wen; Heng Fan
>
> **摘要:** Generic Event Boundary Detection (GEBD) aims to identify moments in videos that humans perceive as event boundaries. This paper proposes a novel method for addressing this task, called Structured Context Learning, which introduces the Structured Partition of Sequence (SPoS) to provide a structured context for learning temporal information. Our approach is end-to-end trainable and flexible, not restricted to specific temporal models like GRU, LSTM, and Transformers. This flexibility enables our method to achieve a better speed-accuracy trade-off. Specifically, we apply SPoS to partition the input frame sequence and provide a structured context for the subsequent temporal model. Notably, SPoS's overall computational complexity is linear with respect to the video length. We next calculate group similarities to capture differences between frames, and a lightweight fully convolutional network is utilized to determine the event boundaries based on the grouped similarity maps. To remedy the ambiguities of boundary annotations, we adapt the Gaussian kernel to preprocess the ground-truth event boundaries. Our proposed method has been extensively evaluated on the challenging Kinetics-GEBD, TAPOS, and shot transition detection datasets, demonstrating its superiority over existing state-of-the-art methods.
>
---
#### [new 062] Look, Recite, Then Answer: Enhancing VLM Performance via Self-Generated Knowledge Hints
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型在精准农业等专业领域因“推理驱动幻觉”导致的性能瓶颈，提出“看、复述、回答”框架。通过自生成知识提示，将视觉描述转化为可激活模型内部专家知识的查询，实现可控知识检索，有效缓解幻觉问题，在AgroBench上显著提升作物识别准确率。**

- **链接: [https://arxiv.org/pdf/2512.00882v1](https://arxiv.org/pdf/2512.00882v1)**

> **作者:** Xisheng Feng
>
> **摘要:** Vision-Language Models (VLMs) exhibit significant performance plateaus in specialized domains like precision agriculture, primarily due to "Reasoning-Driven Hallucination" where linguistic priors override visual perception. A key bottleneck is the "Modality Gap": visual embeddings fail to reliably activate the fine-grained expert knowledge already encoded in model parameters. We propose "Look, Recite, Then Answer," a parameter-efficient framework that enhances VLMs via self-generated knowledge hints while keeping backbone models frozen. The framework decouples inference into three stages: (1) Look generates objective visual descriptions and candidate sets; (2) Recite employs a lightweight 1.7B router to transform visual cues into targeted queries that trigger candidate-specific parametric knowledge; (3) Answer performs parallel evidence alignment between descriptions and recited knowledge to select the most consistent label. On AgroBench, our method achieves state-of-the-art results, improving Weed Identification accuracy by 23.6% over Qwen-VL and surpassing GPT-4o without external search overhead. This modular design mitigates hallucinations by transforming passive perception into active, controllable knowledge retrieval
>
---
#### [new 063] ChronosObserver: Taming 4D World with Hyperspace Diffusion Sampling
- **分类: cs.CV**

- **简介: 该论文针对3D一致且时序同步的多视角视频生成任务，解决现有方法在泛化性与可扩展性上的局限。提出无需训练的ChronosObserver方法，通过世界状态超空间表示时空约束，并引导多视角扩散采样轨迹同步，实现高质量4D世界建模。**

- **链接: [https://arxiv.org/pdf/2512.01481v1](https://arxiv.org/pdf/2512.01481v1)**

> **作者:** Qisen Wang; Yifan Zhao; Peisen Shen; Jialu Li; Jia Li
>
> **摘要:** Although prevailing camera-controlled video generation models can produce cinematic results, lifting them directly to the generation of 3D-consistent and high-fidelity time-synchronized multi-view videos remains challenging, which is a pivotal capability for taming 4D worlds. Some works resort to data augmentation or test-time optimization, but these strategies are constrained by limited model generalization and scalability issues. To this end, we propose ChronosObserver, a training-free method including World State Hyperspace to represent the spatiotemporal constraints of a 4D world scene, and Hyperspace Guided Sampling to synchronize the diffusion sampling trajectories of multiple views using the hyperspace. Experimental results demonstrate that our method achieves high-fidelity and 3D-consistent time-synchronized multi-view videos generation without training or fine-tuning for diffusion models.
>
---
#### [new 064] TeleViT1.0: Teleconnection-aware Vision Transformers for Subseasonal to Seasonal Wildfire Pattern Forecasts
- **分类: cs.CV**

- **简介: 该论文针对子季节至季节尺度野火预测难题，提出TeleViT模型，融合局部火情驱动因子、全球场和遥相关指数，通过异构令牌化与多尺度融合，提升长期预测性能。实验表明其在多时滞下优于基线模型，尤其在非洲草原等稳定火情区表现优异。**

- **链接: [https://arxiv.org/pdf/2512.00089v1](https://arxiv.org/pdf/2512.00089v1)**

> **作者:** Ioannis Prapas; Nikolaos Papadopoulos; Nikolaos-Ioannis Bountos; Dimitrios Michail; Gustau Camps-Valls; Ioannis Papoutsis
>
> **备注:** Under review
>
> **摘要:** Forecasting wildfires weeks to months in advance is difficult, yet crucial for planning fuel treatments and allocating resources. While short-term predictions typically rely on local weather conditions, long-term forecasting requires accounting for the Earth's interconnectedness, including global patterns and teleconnections. We introduce TeleViT, a Teleconnection-aware Vision Transformer that integrates (i) fine-scale local fire drivers, (ii) coarsened global fields, and (iii) teleconnection indices. This multi-scale fusion is achieved through an asymmetric tokenization strategy that produces heterogeneous tokens processed jointly by a transformer encoder, followed by a decoder that preserves spatial structure by mapping local tokens to their corresponding prediction patches. Using the global SeasFire dataset (2001-2021, 8-day resolution), TeleViT improves AUPRC performance over U-Net++, ViT, and climatology across all lead times, including horizons up to four months. At zero lead, TeleViT with indices and global inputs reaches AUPRC 0.630 (ViT 0.617, U-Net 0.620), at 16x8day lead (around 4 months), TeleViT variants using global input maintain 0.601-0.603 (ViT 0.582, U-Net 0.578), while surpassing the climatology (0.572) at all lead times. Regional results show the highest skill in seasonally consistent fire regimes, such as African savannas, and lower skill in boreal and arid regions. Attention and attribution analyses indicate that predictions rely mainly on local tokens, with global fields and indices contributing coarse contextual information. These findings suggest that architectures explicitly encoding large-scale Earth-system context can extend wildfire predictability on subseasonal-to-seasonal timescales.
>
---
#### [new 065] Thinking with Drafts: Speculative Temporal Reasoning for Efficient Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文针对长视频理解中的效率瓶颈问题，提出SpecTemp框架。通过轻量级草案模型与强大目标模型协同，实现快速帧筛选与精准推理，提升推理效率。构建了双层标注数据集，实验表明其在保持高精度的同时显著加速推理。**

- **链接: [https://arxiv.org/pdf/2512.00805v1](https://arxiv.org/pdf/2512.00805v1)**

> **作者:** Pengfei Hu; Meng Cao; Yingyao Wang; Yi Wang; Jiahua Dong; Jun Song; Yu Cheng; Bo Zheng; Xiaodan Liang
>
> **摘要:** Long video understanding is essential for human-like intelligence, enabling coherent perception and reasoning over extended temporal contexts. While the emerging thinking-with-frames paradigm, which alternates between global temporal reasoning and local frame examination, has advanced the reasoning capabilities of video multi-modal large language models (MLLMs), it suffers from a significant efficiency bottleneck due to the progressively growing and redundant multi-modal context. To address this, we propose SpecTemp, a reinforcement learning-based Speculative Temporal reasoning framework that decouples temporal perception from reasoning via a cooperative dual-model design. In SpecTemp, a lightweight draft MLLM rapidly explores and proposes salient frames from densely sampled temporal regions, while a powerful target MLLM focuses on temporal reasoning and verifies the draft's proposals, iteratively refining its attention until convergence. This design mirrors the collaborative pathways of the human brain, balancing efficiency with accuracy. To support training, we construct the SpecTemp-80K dataset, featuring synchronized dual-level annotations for coarse evidence spans and fine-grained frame-level evidence. Experiments across multiple video understanding benchmarks demonstrate that SpecTemp not only maintains competitive accuracy but also significantly accelerates inference compared with existing thinking-with-frames methods.
>
---
#### [new 066] FOD-S2R: A FOD Dataset for Sim2Real Transfer Learning based Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对航空燃油箱内异物检测（FOD）难题，提出首个面向仿真到现实迁移学习的FOD-S2R数据集。通过融合真实与合成图像，系统评估合成数据对提升模型在封闭空间中检测性能的作用，验证了其在减少标注依赖、增强泛化能力方面的有效性，为自动化FOD检测提供了有力支持。**

- **链接: [https://arxiv.org/pdf/2512.01315v1](https://arxiv.org/pdf/2512.01315v1)**

> **作者:** Ashish Vashist; Qiranul Saadiyean; Suresh Sundaram; Chandra Sekhar Seelamantula
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** Foreign Object Debris (FOD) within aircraft fuel tanks presents critical safety hazards including fuel contamination, system malfunctions, and increased maintenance costs. Despite the severity of these risks, there is a notable lack of dedicated datasets for the complex, enclosed environments found inside fuel tanks. To bridge this gap, we present a novel dataset, FOD-S2R, composed of real and synthetic images of the FOD within a simulated aircraft fuel tank. Unlike existing datasets that focus on external or open-air environments, our dataset is the first to systematically evaluate the effectiveness of synthetic data in enhancing the real-world FOD detection performance in confined, closed structures. The real-world subset consists of 3,114 high-resolution HD images captured in a controlled fuel tank replica, while the synthetic subset includes 3,137 images generated using Unreal Engine. The dataset is composed of various Field of views (FOV), object distances, lighting conditions, color, and object size. Prior research has demonstrated that synthetic data can reduce reliance on extensive real-world annotations and improve the generalizability of vision models. Thus, we benchmark several state-of-the-art object detection models and demonstrate that introducing synthetic data improves the detection accuracy and generalization to real-world conditions. These experiments demonstrate the effectiveness of synthetic data in enhancing the model performance and narrowing the Sim2Real gap, providing a valuable foundation for developing automated FOD detection systems for aviation maintenance.
>
---
#### [new 067] Evaluating SAM2 for Video Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文研究视频语义分割（VSS）任务，旨在解决SAM2模型在复杂场景下实现高精度、时序一致的多对象分割难题。通过两种融合策略：一是结合SAM2边界精准性与分割网络优化；二是利用特征向量分类并融合结果，验证了SAM2在提升VSS性能上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.01774v1](https://arxiv.org/pdf/2512.01774v1)**

> **作者:** Syed Hesham Syed Ariff; Yun Liu; Guolei Sun; Jing Yang; Henghui Ding; Xue Geng; Xudong Jiang
>
> **备注:** 17 pages, 3 figures and 7 tables
>
> **摘要:** The Segmentation Anything Model 2 (SAM2) has proven to be a powerful foundation model for promptable visual object segmentation in both images and videos, capable of storing object-aware memories and transferring them temporally through memory blocks. While SAM2 excels in video object segmentation by providing dense segmentation masks based on prompts, extending it to dense Video Semantic Segmentation (VSS) poses challenges due to the need for spatial accuracy, temporal consistency, and the ability to track multiple objects with complex boundaries and varying scales. This paper explores the extension of SAM2 for VSS, focusing on two primary approaches and highlighting firsthand observations and common challenges faced during this process. The first approach involves using SAM2 to extract unique objects as masks from a given image, with a segmentation network employed in parallel to generate and refine initial predictions. The second approach utilizes the predicted masks to extract unique feature vectors, which are then fed into a simple network for classification. The resulting classifications and masks are subsequently combined to produce the final segmentation. Our experiments suggest that leveraging SAM2 enhances overall performance in VSS, primarily due to its precise predictions of object boundaries.
>
---
#### [new 068] Envision: Benchmarking Unified Understanding & Generation for Causal World Process Insights
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态模型在动态过程理解与生成上的不足，提出Envision基准任务，聚焦因果事件进展的文本到多图像生成。通过1000个四阶段跨领域提示和Envision-Score评估指标，揭示现有模型在时空一致性与世界知识内化上的缺陷，强调从静态图像转向动态序列生成的重要性。**

- **链接: [https://arxiv.org/pdf/2512.01816v1](https://arxiv.org/pdf/2512.01816v1)**

> **作者:** Juanxi Tian; Siyuan Li; Conghui He; Lijun Wu; Cheng Tan
>
> **备注:** 35 pages, 12 figures, 10 tables
>
> **摘要:** Current multimodal models aim to transcend the limitations of single-modality representations by unifying understanding and generation, often using text-to-image (T2I) tasks to calibrate semantic consistency. However, their reliance on static, single-image generation in training and evaluation leads to overfitting to static pattern matching and semantic fusion, while fundamentally hindering their ability to model dynamic processes that unfold over time. To address these constraints, we propose Envision-a causal event progression benchmark for chained text-to-multi-image generation. Grounded in world knowledge and structured by spatiotemporal causality, it reorganizes existing evaluation dimensions and includes 1,000 four-stage prompts spanning six scientific and humanities domains. To transition evaluation from single images to sequential frames and assess whether models truly internalize world knowledge while adhering to causal-temporal constraints, we introduce Envision-Score, a holistic metric integrating multi-dimensional consistency, physicality, and aesthetics. Comprehensive evaluation of 15 models (10 specialized T2I models, 5 unified models) uncovers: specialized T2I models demonstrate proficiency in aesthetic rendering yet lack intrinsic world knowledge. Unified multimodal models bridge this gap, consistently outperforming specialized counterparts in causal narrative coherence. However, even these unified architectures remain subordinate to closed-source models and struggle to overcome the core challenge of spatiotemporal consistency. This demonstrates that a focus on causally-isolated single images impedes multi-frame reasoning and generation, promoting static pattern matching over dynamic world modeling-ultimately limiting world knowledge internalization, generation.
>
---
#### [new 069] Generative Editing in the Joint Vision-Language Space for Zero-Shot Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文针对零样本组合图像检索（ZS-CIR）任务，解决文本与视觉模态间对齐难题。提出Fusion-Diff框架，通过联合视觉-语言空间中的多模态融合编辑与轻量级Control-Adapter，实现高效生成式编辑，在20万合成数据上微调即达先进性能，显著提升检索准确率与模型可解释性。**

- **链接: [https://arxiv.org/pdf/2512.01636v1](https://arxiv.org/pdf/2512.01636v1)**

> **作者:** Xin Wang; Haipeng Zhang; Mang Li; Zhaohui Xia; Yueguo Chen; Yu Zhang; Chunyu Wei
>
> **摘要:** Composed Image Retrieval (CIR) enables fine-grained visual search by combining a reference image with a textual modification. While supervised CIR methods achieve high accuracy, their reliance on costly triplet annotations motivates zero-shot solutions. The core challenge in zero-shot CIR (ZS-CIR) stems from a fundamental dilemma: existing text-centric or diffusion-based approaches struggle to effectively bridge the vision-language modality gap. To address this, we propose Fusion-Diff, a novel generative editing framework with high effectiveness and data efficiency designed for multimodal alignment. First, it introduces a multimodal fusion feature editing strategy within a joint vision-language (VL) space, substantially narrowing the modality gap. Second, to maximize data efficiency, the framework incorporates a lightweight Control-Adapter, enabling state-of-the-art performance through fine-tuning on only a limited-scale synthetic dataset of 200K samples. Extensive experiments on standard CIR benchmarks (CIRR, FashionIQ, and CIRCO) demonstrate that Fusion-Diff significantly outperforms prior zero-shot approaches. We further enhance the interpretability of our model by visualizing the fused multimodal representations.
>
---
#### [new 070] Handwritten Text Recognition for Low Resource Languages
- **分类: cs.CV**

- **简介: 该论文针对低资源语言（如印地语、乌尔都语）的段落级手写文本识别难题，提出BharatOCR模型。采用视觉变压器与Transformer解码器结合，引入预训练语言模型提升识别准确率与流畅性，实现无需显式分行的隐式线分割。在自建及公开数据集上取得领先性能。**

- **链接: [https://arxiv.org/pdf/2512.01348v1](https://arxiv.org/pdf/2512.01348v1)**

> **作者:** Sayantan Dey; Alireza Alaei; Partha Pratim Roy
>
> **备注:** 21 Pages
>
> **摘要:** Despite considerable progress in handwritten text recognition, paragraph-level handwritten text recognition, especially in low-resource languages, such as Hindi, Urdu and similar scripts, remains a challenging problem. These languages, often lacking comprehensive linguistic resources, require special attention to develop robust systems for accurate optical character recognition (OCR). This paper introduces BharatOCR, a novel segmentation-free paragraph-level handwritten Hindi and Urdu text recognition. We propose a ViT-Transformer Decoder-LM architecture for handwritten text recognition, where a Vision Transformer (ViT) extracts visual features, a Transformer decoder generates text sequences, and a pre-trained language model (LM) refines the output to improve accuracy, fluency, and coherence. Our model utilizes a Data-efficient Image Transformer (DeiT) model proposed for masked image modeling in this research work. In addition, we adopt a RoBERTa architecture optimized for masked language modeling (MLM) to enhance the linguistic comprehension and generative capabilities of the proposed model. The transformer decoder generates text sequences from visual embeddings. This model is designed to iteratively process a paragraph image line by line, called implicit line segmentation. The proposed model was evaluated using our custom dataset ('Parimal Urdu') and ('Parimal Hindi'), introduced in this research work, as well as two public datasets. The proposed model achieved benchmark results in the NUST-UHWR, PUCIT-OUHL, and Parimal-Urdu datasets, achieving character recognition rates of 96.24%, 92.05%, and 94.80%, respectively. The model also provided benchmark results using the Hindi dataset achieving a character recognition rate of 80.64%. The results obtained from our proposed model indicated that it outperformed several state-of-the-art Urdu text recognition methods.
>
---
#### [new 071] MM-DETR: An Efficient Multimodal Detection Transformer with Mamba-Driven Dual-Granularity Fusion and Frequency-Aware Modality Adapters
- **分类: cs.CV**

- **简介: 该论文针对多模态遥感目标检测中融合效率与参数冗余问题，提出轻量级MM-DETR框架。通过Mamba驱动的双粒度融合与频域感知适配器，实现高效跨模态建模与模态特异性特征提取，显著提升检测性能并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2512.00363v1](https://arxiv.org/pdf/2512.00363v1)**

> **作者:** Jianhong Han; Yupei Wang; Yuan Zhang; Liang Chen
>
> **备注:** Manuscript submitted to IEEE Transactions on Geoscience and Remote Sensing
>
> **摘要:** Multimodal remote sensing object detection aims to achieve more accurate and robust perception under challenging conditions by fusing complementary information from different modalities. However, existing approaches that rely on attention-based or deformable convolution fusion blocks still struggle to balance performance and lightweight design. Beyond fusion complexity, extracting modality features with shared backbones yields suboptimal representations due to insufficient modality-specific modeling, whereas dual-stream architectures nearly double the parameter count, ultimately limiting practical deployment. To this end, we propose MM-DETR, a lightweight and efficient framework for multimodal object detection. Specifically, we propose a Mamba-based dual granularity fusion encoder that reformulates global interaction as channel-wise dynamic gating and leverages a 1D selective scan for efficient cross-modal modeling with linear complexity. Following this design, we further reinterpret multimodal fusion as a modality completion problem. A region-aware 2D selective scanning completion branch is introduced to recover modality-specific cues, supporting fine-grained fusion along a bidirectional pyramid pathway with minimal overhead. To further reduce parameter redundancy while retaining strong feature extraction capability, a lightweight frequency-aware modality adapter is inserted into the shared backbone. This adapter employs a spatial-frequency co-expert structure to capture modality-specific cues, while a pixel-wise router dynamically balances expert contributions for efficient spatial-frequency fusion. Extensive experiments conducted on four multimodal benchmark datasets demonstrate the effectiveness and generalization capability of the proposed method.
>
---
#### [new 072] Multi-modal On-Device Learning for Monocular Depth Estimation on Ultra-low-power MCUs
- **分类: cs.CV**

- **简介: 该论文针对超低功耗物联网设备上单目深度估计的域偏移问题，提出一种多模态在设备端学习方法。通过激活低分辨率深度传感器获取伪标签，仅用3千样本在MCU上实现17.8分钟内快速微调，内存降至1.2MB，精度损失仅2%。首次实现在真实场景中高效、低功耗的设备端自适应深度估计。**

- **链接: [https://arxiv.org/pdf/2512.00086v1](https://arxiv.org/pdf/2512.00086v1)**

> **作者:** Davide Nadalini; Manuele Rusci; Elia Cereda; Luca Benini; Francesco Conti; Daniele Palossi
>
> **备注:** 14 pages, 9 figures, 3 tables. Associated open-source release available at: https://github.com/dnadalini/ondevice_learning_for_monocular_depth_estimation
>
> **摘要:** Monocular depth estimation (MDE) plays a crucial role in enabling spatially-aware applications in Ultra-low-power (ULP) Internet-of-Things (IoT) platforms. However, the limited number of parameters of Deep Neural Networks for the MDE task, designed for IoT nodes, results in severe accuracy drops when the sensor data observed in the field shifts significantly from the training dataset. To address this domain shift problem, we present a multi-modal On-Device Learning (ODL) technique, deployed on an IoT device integrating a Greenwaves GAP9 MicroController Unit (MCU), a 80 mW monocular camera and a 8 x 8 pixel depth sensor, consuming $\approx$300mW. In its normal operation, this setup feeds a tiny 107 k-parameter $μ$PyD-Net model with monocular images for inference. The depth sensor, usually deactivated to minimize energy consumption, is only activated alongside the camera to collect pseudo-labels when the system is placed in a new environment. Then, the fine-tuning task is performed entirely on the MCU, using the new data. To optimize our backpropagation-based on-device training, we introduce a novel memory-driven sparse update scheme, which minimizes the fine-tuning memory to 1.2 MB, 2.2x less than a full update, while preserving accuracy (i.e., only 2% and 1.5% drops on the KITTI and NYUv2 datasets). Our in-field tests demonstrate, for the first time, that ODL for MDE can be performed in 17.8 minutes on the IoT node, reducing the root mean squared error from 4.9 to 0.6m with only 3 k self-labeled samples, collected in a real-life deployment scenario.
>
---
#### [new 073] OpenREAD: Reinforced Open-Ended Reasoing for End-to-End Autonomous Driving with LLM-as-Critic
- **分类: cs.CV**

- **简介: 该论文提出OpenREAD框架，针对自动驾驶中推理泛化弱与开放场景奖励难量化问题，构建大规模思维链数据，利用Qwen3 LLM作为评议员实现端到端强化微调，提升从高阶推理到低阶规划的全流程性能，推动知识驱动自动驾驶发展。**

- **链接: [https://arxiv.org/pdf/2512.01830v1](https://arxiv.org/pdf/2512.01830v1)**

> **作者:** Songyan Zhang; Wenhui Huang; Zhan Chen; Chua Jiahao Collister; Qihang Huang; Chen Lv
>
> **摘要:** Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks.
>
---
#### [new 074] Gaussian Swaying: Surface-Based Framework for Aerodynamic Simulation with 3D Gaussians
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出Gaussian Swaying框架，用于高效模拟自然物体在风中的动态变形。针对传统方法需网格或粒子表示的局限，利用3D高斯连续建模表面，统一模拟与渲染，实现精细且高效的气动交互，显著提升仿真真实感与计算效率。**

- **链接: [https://arxiv.org/pdf/2512.01306v1](https://arxiv.org/pdf/2512.01306v1)**

> **作者:** Hongru Yan; Xiang Zhang; Zeyuan Chen; Fangyin Wei; Zhuowen Tu
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Branches swaying in the breeze, flags rippling in the wind, and boats rocking on the water all show how aerodynamics shape natural motion -- an effect crucial for realism in vision and graphics. In this paper, we present Gaussian Swaying, a surface-based framework for aerodynamic simulation using 3D Gaussians. Unlike mesh-based methods that require costly meshing, or particle-based approaches that rely on discrete positional data, Gaussian Swaying models surfaces continuously with 3D Gaussians, enabling efficient and fine-grained aerodynamic interaction. Our framework unifies simulation and rendering on the same representation: Gaussian patches, which support force computation for dynamics while simultaneously providing normals for lightweight shading. Comprehensive experiments on both synthetic and real-world datasets across multiple metrics demonstrate that Gaussian Swaying achieves state-of-the-art performance and efficiency, offering a scalable approach for realistic aerodynamic scene simulation.
>
---
#### [new 075] TalkingPose: Efficient Face and Gesture Animation with Feedback-guided Diffusion Model
- **分类: cs.CV**

- **简介: 该论文针对长时序、高保真人物上半身动画生成任务，解决现有扩散模型在长时间生成中缺乏时序一致性与计算资源受限的问题。提出TalkingPose框架，通过反馈引导的扩散机制实现无限制时长的稳定动画生成，无需额外训练，同时构建大规模基准数据集。**

- **链接: [https://arxiv.org/pdf/2512.00909v1](https://arxiv.org/pdf/2512.00909v1)**

> **作者:** Alireza Javanmardi; Pragati Jaiswal; Tewodros Amberbir Habtegebrial; Christen Millerdurai; Shaoxiang Wang; Alain Pagani; Didier Stricker
>
> **备注:** WACV 2026, Project page available at https://dfki-av.github.io/TalkingPose
>
> **摘要:** Recent advancements in diffusion models have significantly improved the realism and generalizability of character-driven animation, enabling the synthesis of high-quality motion from just a single RGB image and a set of driving poses. Nevertheless, generating temporally coherent long-form content remains challenging. Existing approaches are constrained by computational and memory limitations, as they are typically trained on short video segments, thus performing effectively only over limited frame lengths and hindering their potential for extended coherent generation. To address these constraints, we propose TalkingPose, a novel diffusion-based framework specifically designed for producing long-form, temporally consistent human upper-body animations. TalkingPose leverages driving frames to precisely capture expressive facial and hand movements, transferring these seamlessly to a target actor through a stable diffusion backbone. To ensure continuous motion and enhance temporal coherence, we introduce a feedback-driven mechanism built upon image-based diffusion models. Notably, this mechanism does not incur additional computational costs or require secondary training stages, enabling the generation of animations with unlimited duration. Additionally, we introduce a comprehensive, large-scale dataset to serve as a new benchmark for human upper-body animation.
>
---
#### [new 076] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition
- **分类: cs.CV**

- **简介: 该论文针对表格识别（TR）任务，解决标注数据稀缺导致开源模型性能落后的问题。提出TRivia方法，通过自监督学习让视觉语言模型从无标签表格图像中自主学习，利用问答反馈机制实现闭环优化，无需人工标注。构建了性能领先的TRivia-3B模型并开源。**

- **链接: [https://arxiv.org/pdf/2512.01248v1](https://arxiv.org/pdf/2512.01248v1)**

> **作者:** Junyuan Zhang; Bin Wang; Qintong Zhang; Fan Wu; Zichen Wen; Jialin Lu; Junjie Shan; Ziqi Zhao; Shuya Yang; Ziling Wang; Ziyang Miao; Huaping Zhong; Yuhang Zang; Xiaoyi Dong; Ka-Ho Chow; Conghui He
>
> **摘要:** Table recognition (TR) aims to transform table images into semi-structured representations such as HTML or Markdown. As a core component of document parsing, TR has long relied on supervised learning, with recent efforts dominated by fine-tuning vision-language models (VLMs) using labeled data. While VLMs have brought TR to the next level, pushing performance further demands large-scale labeled data that is costly to obtain. Consequently, although proprietary models have continuously pushed the performance boundary, open-source models, often trained with limited resources and, in practice, the only viable option for many due to privacy regulations, still lag far behind. To bridge this gap, we introduce TRivia, a self-supervised fine-tuning method that enables pretrained VLMs to learn TR directly from unlabeled table images in the wild. Built upon Group Relative Policy Optimization, TRivia automatically identifies unlabeled samples that most effectively facilitate learning and eliminates the need for human annotations through a question-answering-based reward mechanism. An attention-guided module generates diverse questions for each table image, and the ability to interpret the recognition results and answer them correctly provides feedback to optimize the TR model. This closed-loop process allows the TR model to autonomously learn to recognize, structure, and reason over tables without labeled data. Leveraging this pipeline, we present TRivia-3B, an open-sourced, compact, and state-of-the-art TR model that surpasses existing systems (e.g., Gemini 2.5 Pro, MinerU2.5) on three popular benchmarks. Model and code are released at: https://github.com/opendatalab/TRivia
>
---
#### [new 077] A variational method for curve extraction with curvature-dependent energies
- **分类: cs.CV**

- **简介: 该论文针对图像中1D结构自动提取任务，提出一种基于变分法的曲线提取方法。通过能量离散化与向量场分解定理，构建双层优化模型，实现无监督曲线生成；进一步引入曲率相关能量，将曲线提升至位置-方向空间，结合子黎曼或芬斯勒度量，增强对复杂结构的建模能力。**

- **链接: [https://arxiv.org/pdf/2512.01494v1](https://arxiv.org/pdf/2512.01494v1)**

> **作者:** Majid Arthaud; Antonin Chambolle; Vincent Duval
>
> **摘要:** We introduce a variational approach for extracting curves between a list of possible endpoints, based on the discretization of an energy and Smirnov's decomposition theorem for vector fields. It is used to design a bi-level minimization approach to automatically extract curves and 1D structures from an image, which is mostly unsupervised. We extend then the method to curvature-dependent energies, using a now classical lifting of the curves in the space of positions and orientations equipped with an appropriate sub-Riemanian or Finslerian metric.
>
---
#### [new 078] Satellite to Street : Disaster Impact Estimator
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对灾后损毁评估中人工判读效率低、难以规模化的问题，提出Satellite-to-Street：灾害影响估测器框架。采用改进的双输入U-Net结构，融合前后灾卫星图像，结合类别感知损失函数，提升对细微结构变化与严重损毁区域的检测能力，实现像素级损毁图生成，支持高效、数据驱动的应急响应决策。**

- **链接: [https://arxiv.org/pdf/2512.00065v1](https://arxiv.org/pdf/2512.00065v1)**

> **作者:** Sreesritha Sai; Sai Venkata Suma Sreeja; Deepthi; Nikhil
>
> **备注:** 11 pages,9 figures
>
> **摘要:** Accurate post-disaster damage assessment is of high importance for prioritizing emergency response; however, manual interpretation of satellite imagery is slow, subjective, and hard to scale. While deep-learning models for image segmentation, such as U-Net-based baselines and change-detection models, are useful baselines, they often struggle with subtle structural variations and severe class imbalance, yielding poor detection of highly damaged regions. The present work proposes a deep-learning framework that jointly processes pre- and post-disaster satellite images to obtain fine-grained pixel-level damage maps: Satellite-to-Street: Disaster Impact Estimator. The model uses a modified dual-input U-Net architecture with enhanced feature fusion to capture both the local structural changes as well as the broader contextual cues. Class-aware weighted loss functions are integrated in order to handle the dominance of undamaged pixels in real disaster datasets, thus enhancing sensitivity toward major and destroyed categories. Experimentation on publicly available disaster datasets shows improved localization and classification of structural damage when compared to traditional segmentation and baseline change-detection models. The resulting damage maps provide a rapid and consistent assessment mechanism to support and not replace expert decision-making, thus allowing more efficient, data-driven disaster management.
>
---
#### [new 079] mmPred: Radar-based Human Motion Prediction in the Dark
- **分类: cs.CV**

- **简介: 该论文提出mmPred，首个基于毫米波雷达的人体运动预测框架。针对雷达信号噪声大、关节检测不一致的问题，设计双域表示与全局骨骼关系Transformer，提升预测精度。有效解决光照敏感与隐私问题，推动雷达在医疗、救援等场景应用。**

- **链接: [https://arxiv.org/pdf/2512.00345v1](https://arxiv.org/pdf/2512.00345v1)**

> **作者:** Junqiao Fan; Haocong Rao; Jiarui Zhang; Jianfei Yang; Lihua Xie
>
> **备注:** This paper is accepted by AAAI-2026
>
> **摘要:** Existing Human Motion Prediction (HMP) methods based on RGB-D cameras are sensitive to lighting conditions and raise privacy concerns, limiting their real-world applications such as firefighting and healthcare. Motivated by the robustness and privacy-preserving nature of millimeter-wave (mmWave) radar, this work introduces radar as a novel sensing modality for HMP, for the first time. Nevertheless, radar signals often suffer from specular reflections and multipath effects, resulting in noisy and temporally inconsistent measurements, such as body-part miss-detection. To address these radar-specific artifacts, we propose mmPred, the first diffusion-based framework tailored for radar-based HMP. mmPred introduces a dual-domain historical motion representation to guide the generation process, combining a Time-domain Pose Refinement (TPR) branch for learning fine-grained details and a Frequency-domain Dominant Motion (FDM) branch for capturing global motion trends and suppressing frame-level inconsistency. Furthermore, we design a Global Skeleton-relational Transformer (GST) as the diffusion backbone to model global inter-joint cooperation, enabling corrupted joints to dynamically aggregate information from others. Extensive experiments show that mmPred achieves state-of-the-art performance, outperforming existing methods by 8.6% on mmBody and 22% on mm-Fi.
>
---
#### [new 080] SocialFusion: Addressing Social Degradation in Pre-trained Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对预训练视觉语言模型在社会感知任务中表现不佳的问题，提出“社会退化”现象源于视觉编码器对社会信息表征能力的削弱。为此，提出SocialFusion框架，通过轻量级连接融合冻结视觉编码器与语言模型，实现多社会任务正向迁移，显著提升综合性能，推动更社会-aware的模型训练。**

- **链接: [https://arxiv.org/pdf/2512.01148v1](https://arxiv.org/pdf/2512.01148v1)**

> **作者:** Hamza Tahboub; Weiyan Shi; Gang Hua; Huaizu Jiang
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** Understanding social interactions from visual cues is a fundamental challenge for a socially competent AI. While powerful pre-trained vision-language models (VLMs) have shown remarkable general capabilities, they surprisingly struggle to unify and learn multiple social perception tasks simultaneously, often exhibiting negative transfer. We identify that this negative transfer stems from a critical issue we term "social degradation," whereby the general visual-linguistic pre-training process of VLMs impairs the visual encoder's ability to represent nuanced social information. We investigate this behavior further under two lenses: decodability through linear representation probing and compatibility through gradient conflict analysis, revealing that both play a role in the degradation, especially the former, which is significantly compromised in the VLM pre-training process. To address these issues, we propose SocialFusion, a unified framework that learns a minimal connection between a frozen visual encoder and a language model. Compared with existing VLMs, it exhibits positive transfer across all five social tasks, leveraging synergies between them to enhance overall performance and achieves comparable performance to task-specific state-of-the-art models on various benchmarks. Our findings suggest that current VLM pre-training strategies may be detrimental to acquiring general social competence and highlight the need for more socially-aware training paradigms.
>
---
#### [new 081] NeuroVolve: Evolving Visual Stimuli toward Programmable Neural Objectives
- **分类: cs.CV**

- **简介: 该论文提出NeuroVolve框架，旨在通过优化视觉语言模型嵌入空间中的神经目标函数，实现脑引导的图像生成。解决如何精准激活或抑制特定脑区以生成符合神经目标的视觉刺激问题。工作包括验证单区域选择性、合成多区域协同/拮抗的语义场景，并揭示神经表征的语义演化轨迹，支持个性化脑驱动图像合成与可解释的神经机制研究。**

- **链接: [https://arxiv.org/pdf/2512.00557v1](https://arxiv.org/pdf/2512.00557v1)**

> **作者:** Haomiao Chen; Keith W Jamison; Mert R. Sabuncu; Amy Kuceyeski
>
> **摘要:** What visual information is encoded in individual brain regions, and how do distributed patterns combine to create their neural representations? Prior work has used generative models to replicate known category selectivity in isolated regions (e.g., faces in FFA), but these approaches offer limited insight into how regions interact during complex, naturalistic vision. We introduce NeuroVolve, a generative framework that provides brain-guided image synthesis via optimization of a neural objective function in the embedding space of a pretrained vision-language model. Images are generated under the guidance of a programmable neural objective, i.e., activating or deactivating single regions or multiple regions together. NeuroVolve is validated by recovering known selectivity for individual brain regions, while expanding to synthesize coherent scenes that satisfy complex, multi-region constraints. By tracking optimization steps, it reveals semantic trajectories through embedding space, unifying brain-guided image editing and preferred stimulus generation in a single process. We show that NeuroVolve can generate both low-level and semantic feature-specific stimuli for single ROIs, as well as stimuli aligned to curated neural objectives. These include co-activation and decorrelation between regions, exposing cooperative and antagonistic tuning relationships. Notably, the framework captures subject-specific preferences, supporting personalized brain-driven synthesis and offering interpretable constraints for mapping, analyzing, and probing neural representations of visual information.
>
---
#### [new 082] Optimizing LVLMs with On-Policy Data for Effective Hallucination Mitigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对大视觉语言模型（LVLM）中的幻觉问题，提出基于在线数据的优化方法。通过构建幻觉分类器保证标注质量，设计动态重加权的迭代直接偏好优化算法，有效利用在线数据，显著降低幻觉率，使开源模型性能超越GPT-4V。**

- **链接: [https://arxiv.org/pdf/2512.00706v1](https://arxiv.org/pdf/2512.00706v1)**

> **作者:** Chengzhi Yu; Yifan Xu; Yifan Chen; Wenyi Zhang
>
> **摘要:** Recently, large vision-language models (LVLMs) have risen to be a promising approach for multimodal tasks. However, principled hallucination mitigation remains a critical challenge.In this work, we first analyze the data generation process in LVLM hallucination mitigation and affirm that on-policy data significantly outperforms off-policy data, which thus calls for efficient and reliable preference annotation of on-policy data. We then point out that, existing annotation methods introduce additional hallucination in training samples, which may enhance the model's hallucination patterns, to address this problem, we propose training a hallucination classifier giving binary annotations, which guarantee clean chosen samples for the subsequent alignment. To further harness of the power of on-policy data, we design a robust iterative direct preference optimization (DPO) algorithm adopting a dynamic sample reweighting scheme. We conduct comprehensive experiments on three benchmarks with comparison to 8 state-of-the-art baselines. In particular, our approach reduces the hallucination rate of LLaVA-1.5-7B on MMHalBench by 50.8% and the average hallucination rate on Object HalBench by 79.5%; more significantly, our method fully taps into the potential of open-source models, enabling LLaVA-1.5-13B to even surpass the performance of GPT-4V.
>
---
#### [new 083] Words into World: A Task-Adaptive Agent for Language-Guided Spatial Retrieval in AR
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文提出一种面向增强现实（AR）的语言引导空间检索任务，解决传统AR系统难以理解开放词汇自然语言查询的问题。通过融合多模态大模型与具身视觉模型，构建可自适应的任务代理，实现从简单识别到复杂关系推理的动态响应，并生成米级精度3D锚点，支持人机协同优化。**

- **链接: [https://arxiv.org/pdf/2512.00294v1](https://arxiv.org/pdf/2512.00294v1)**

> **作者:** Lixing Guo; Tobias Höllerer
>
> **摘要:** Traditional augmented reality (AR) systems predominantly rely on fixed class detectors or fiducial markers, limiting their ability to interpret complex, open-vocabulary natural language queries. We present a modular AR agent system that integrates multimodal large language models (MLLMs) with grounded vision models to enable relational reasoning in space and language-conditioned spatial retrieval in physical environments. Our adaptive task agent coordinates MLLMs and coordinate-aware perception tools to address varying query complexities, ranging from simple object identification to multi-object relational reasoning, while returning meter-accurate 3D anchors. It constructs dynamic AR scene graphs encoding nine typed relations (spatial, structural-semantic, causal-functional), enabling MLLMs to understand not just what objects exist, but how they relate and interact in 3D space. Through task-adaptive region-of-interest highlighting and contextual spatial retrieval, the system guides human attention to information-dense areas while supporting human-in-the-loop refinement. The agent dynamically invokes coordinate-aware tools for complex queries-selection, measurement, comparison, and actuation-grounding language understanding in physical operations. The modular architecture supports plug-and-use vision-language models without retraining, establishing AR agents as intermediaries that augment MLLMs with real-world spatial intelligence for interactive scene understanding. We also introduce GroundedAR-Bench, an evaluation framework for language-driven real world localization and relation grounding across diverse environments.
>
---
#### [new 084] Diffusion-Based Synthetic Brightfield Microscopy Images for Enhanced Single Cell Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对明亮场显微镜下单细胞检测因数据稀缺和标注困难导致的深度学习模型性能受限问题，提出基于U-Net扩散模型生成合成图像。通过混合真实与合成数据训练目标检测模型，显著提升检测精度，且生成图像具备高真实性，有效减少对人工标注的依赖。**

- **链接: [https://arxiv.org/pdf/2512.00078v1](https://arxiv.org/pdf/2512.00078v1)**

> **作者:** Mario de Jesus da Graca; Jörg Dahlkemper; Peer Stelldinger
>
> **摘要:** Accurate single cell detection in brightfield microscopy is crucial for biological research, yet data scarcity and annotation bottlenecks limit the progress of deep learning methods. We investigate the use of unconditional models to generate synthetic brightfield microscopy images and evaluate their impact on object detection performance. A U-Net based diffusion model was trained and used to create datasets with varying ratios of synthetic and real images. Experiments with YOLOv8, YOLOv9 and RT-DETR reveal that training with synthetic data can achieve improved detection accuracies (at minimal costs). A human expert survey demonstrates the high realism of generated images, with experts not capable to distinguish them from real microscopy images (accuracy 50%). Our findings suggest that diffusion-based synthetic data generation is a promising avenue for augmenting real datasets in microscopy image analysis, reducing the reliance on extensive manual annotation and potentially improving the robustness of cell detection models.
>
---
#### [new 085] Weakly Supervised Continuous Micro-Expression Intensity Estimation Using Temporal Deep Neural Network
- **分类: cs.CV**

- **简介: 该论文聚焦连续微表情强度估计任务，解决缺乏帧级标注导致的监督困难问题。提出仅需时序弱标签（起始、峰值、结束）的统一框架，通过三角先验生成伪强度轨迹，并结合ResNet18与双向GRU进行时序回归，实现无需帧级标注的精准强度预测。**

- **链接: [https://arxiv.org/pdf/2512.01145v1](https://arxiv.org/pdf/2512.01145v1)**

> **作者:** Riyadh Mohammed Almushrafy
>
> **摘要:** Micro-facial expressions are brief and involuntary facial movements that reflect genuine emotional states. While most prior work focuses on classifying discrete micro-expression categories, far fewer studies address the continuous evolution of intensity over time. Progress in this direction is limited by the lack of frame-level intensity labels, which makes fully supervised regression impractical. We propose a unified framework for continuous micro-expression intensity estimation using only weak temporal labels (onset, apex, offset). A simple triangular prior converts sparse temporal landmarks into dense pseudo-intensity trajectories, and a lightweight temporal regression model that combines a ResNet18 encoder with a bidirectional GRU predicts frame-wise intensity directly from image sequences. The method requires no frame-level annotation effort and is applied consistently across datasets through a single preprocessing and temporal alignment pipeline. Experiments on SAMM and CASME II show strong temporal agreement with the pseudo-intensity trajectories. On SAMM, the model reaches a Spearman correlation of 0.9014 and a Kendall correlation of 0.7999, outperforming a frame-wise baseline. On CASME II, it achieves up to 0.9116 and 0.8168, respectively, when trained without the apex-ranking term. Ablation studies confirm that temporal modeling and structured pseudo labels are central to capturing the rise-apex-fall dynamics of micro-facial movements. To our knowledge, this is the first unified approach for continuous micro-expression intensity estimation using only sparse temporal annotations.
>
---
#### [new 086] LAHNet: Local Attentive Hashing Network for Point Cloud Registration
- **分类: cs.CV**

- **简介: 该论文针对点云配准任务，解决现有方法局部感受野不足导致特征区分性弱的问题。提出LAHNet，通过局部注意力机制与局部敏感哈希实现非重叠窗口划分，结合跨窗口策略扩展感受野，并设计交互变压器增强配对点云重叠区域特征交互，提升配准精度。**

- **链接: [https://arxiv.org/pdf/2512.00927v1](https://arxiv.org/pdf/2512.00927v1)**

> **作者:** Wentao Qu; Xiaoshui Huang; Liang Xiao
>
> **摘要:** Most existing learning-based point cloud descriptors for point cloud registration focus on perceiving local information of point clouds to generate distinctive features. However, a reasonable and broader receptive field is essential for enhancing feature distinctiveness. In this paper, we propose a Local Attentive Hashing Network for point cloud registration, called LAHNet, which introduces a local attention mechanism with the inductive bias of locality of convolution-like operators into point cloud descriptors. Specifically, a Group Transformer is designed to capture reasonable long-range context between points. This employs a linear neighborhood search strategy, Locality-Sensitive Hashing, enabling uniformly partitioning point clouds into non-overlapping windows. Meanwhile, an efficient cross-window strategy is adopted to further expand the reasonable feature receptive field. Furthermore, building on this effective windowing strategy, we propose an Interaction Transformer to enhance the feature interactions of the overlap regions within point cloud pairs. This computes an overlap matrix to match overlap regions between point cloud pairs by representing each window as a global signal. Extensive results demonstrate that LAHNet can learn robust and distinctive features, achieving significant registration results on real-world indoor and outdoor benchmarks.
>
---
#### [new 087] TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对时序视觉语言模型（VLMs）在预测中依赖静态特征偏差的问题，提出TRoVe方法，自动识别导致错误的静态特征。通过量化特征对错误的影响及模型依赖程度，实现在101个模型上的精准检测，性能优于基线28.6%，并验证其在提升模型性能中的价值。**

- **链接: [https://arxiv.org/pdf/2512.01048v1](https://arxiv.org/pdf/2512.01048v1)**

> **作者:** Maya Varma; Jean-Benoit Delbrouck; Sophie Ostmeier; Akshay Chaudhari; Curtis Langlotz
>
> **备注:** NeurIPS 2025
>
> **摘要:** Vision-language models (VLMs) have made great strides in addressing temporal understanding tasks, which involve characterizing visual changes across a sequence of images. However, recent works have suggested that when making predictions, VLMs may rely on static feature biases, such as background or object features, rather than dynamic visual changes. Static feature biases are a type of shortcut and can contribute to systematic prediction errors on downstream tasks; as a result, identifying and characterizing error-inducing static feature biases is critical prior to real-world model deployment. In this work, we introduce TRoVe, an automated approach for discovering error-inducing static feature biases learned by temporal VLMs. Given a trained VLM and an annotated validation dataset associated with a downstream classification task, TRoVe extracts candidate static features from the dataset and scores each feature by (i) the effect of the feature on classification errors as well as (ii) the extent to which the VLM relies on the feature when making predictions. In order to quantitatively evaluate TRoVe, we introduce an evaluation framework consisting of 101 trained temporal VLMs paired with ground-truth annotations for learned static feature biases. We use this framework to demonstrate that TRoVe can accurately identify error-inducing static feature biases in VLMs, achieving a 28.6% improvement over the closest baseline. Finally, we apply TRoVe to 7 off-the-shelf VLMs and 2 temporal understanding tasks, surfacing previously-unknown static feature biases and demonstrating that knowledge of learned biases can aid in improving model performance at test time. Our code is available at https://github.com/Stanford-AIMI/TRoVe.
>
---
#### [new 088] TrajDiff: End-to-end Autonomous Driving without Perception Annotation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对端到端自动驾驶中感知标注成本高的问题，提出TrajDiff框架，无需感知标注即可从原始传感器数据生成合理轨迹。通过轨迹导向的BEV编码器和扩散变压器，直接生成多样且合理的驾驶轨迹，实现全注释自由的规划，显著提升性能并验证了数据规模收益。**

- **链接: [https://arxiv.org/pdf/2512.00723v1](https://arxiv.org/pdf/2512.00723v1)**

> **作者:** Xingtai Gui; Jianbo Zhao; Wencheng Han; Jikai Wang; Jiahao Gong; Feiyang Tan; Cheng-zhong Xu; Jianbing Shen
>
> **摘要:** End-to-end autonomous driving systems directly generate driving policies from raw sensor inputs. While these systems can extract effective environmental features for planning, relying on auxiliary perception tasks, developing perception annotation-free planning paradigms has become increasingly critical due to the high cost of manual perception annotation. In this work, we propose TrajDiff, a Trajectory-oriented BEV Conditioned Diffusion framework that establishes a fully perception annotation-free generative method for end-to-end autonomous driving. TrajDiff requires only raw sensor inputs and future trajectory, constructing Gaussian BEV heatmap targets that inherently capture driving modalities. We design a simple yet effective trajectory-oriented BEV encoder to extract the TrajBEV feature without perceptual supervision. Furthermore, we introduce Trajectory-oriented BEV Diffusion Transformer (TB-DiT), which leverages ego-state information and the predicted TrajBEV features to directly generate diverse yet plausible trajectories, eliminating the need for handcrafted motion priors. Beyond architectural innovations, TrajDiff enables exploration of data scaling benefits in the annotation-free setting. Evaluated on the NAVSIM benchmark, TrajDiff achieves 87.5 PDMS, establishing state-of-the-art performance among all annotation-free methods. With data scaling, it further improves to 88.5 PDMS, which is comparable to advanced perception-based approaches. Our code and model will be made publicly available.
>
---
#### [new 089] Visual Sync: Multi-Camera Synchronization via Cross-View Object Motion
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对多摄像头视频同步难题，提出VisualSync框架。通过利用跨视角运动点的对极几何约束，结合3D重建与特征匹配，联合优化以毫米级精度估计时间偏移。解决了无标定、非同步视频在复杂场景下的自动同步问题。**

- **链接: [https://arxiv.org/pdf/2512.02017v1](https://arxiv.org/pdf/2512.02017v1)**

> **作者:** Shaowei Liu; David Yifan Yao; Saurabh Gupta; Shenlong Wang
>
> **备注:** Accepted to NeurIPS 2025. Project page: https://stevenlsw.github.io/visualsync/
>
> **摘要:** Today, people can easily record memorable moments, ranging from concerts, sports events, lectures, family gatherings, and birthday parties with multiple consumer cameras. However, synchronizing these cross-camera streams remains challenging. Existing methods assume controlled settings, specific targets, manual correction, or costly hardware. We present VisualSync, an optimization framework based on multi-view dynamics that aligns unposed, unsynchronized videos at millisecond accuracy. Our key insight is that any moving 3D point, when co-visible in two cameras, obeys epipolar constraints once properly synchronized. To exploit this, VisualSync leverages off-the-shelf 3D reconstruction, feature matching, and dense tracking to extract tracklets, relative poses, and cross-view correspondences. It then jointly minimizes the epipolar error to estimate each camera's time offset. Experiments on four diverse, challenging datasets show that VisualSync outperforms baseline methods, achieving an median synchronization error below 50 ms.
>
---
#### [new 090] Cross-Temporal 3D Gaussian Splatting for Sparse-View Guided Scene Update
- **分类: cs.CV**

- **简介: 该论文提出一种跨时序3D高斯点云渲染方法，解决稀疏视图下长期场景更新难题。通过相机位姿对齐、变化区域识别与渐进式优化，实现基于历史先验的高效跨时间场景重建与版本管理，支持非连续采集与过去场景恢复，显著提升重建质量与数据效率。**

- **链接: [https://arxiv.org/pdf/2512.00534v1](https://arxiv.org/pdf/2512.00534v1)**

> **作者:** Zeyuan An; Yanghang Xiao; Zhiying Leng; Frederick W. B. Li; Xiaohui Liang
>
> **备注:** AAAI2026 accepted
>
> **摘要:** Maintaining consistent 3D scene representations over time is a significant challenge in computer vision. Updating 3D scenes from sparse-view observations is crucial for various real-world applications, including urban planning, disaster assessment, and historical site preservation, where dense scans are often unavailable or impractical. In this paper, we propose Cross-Temporal 3D Gaussian Splatting (Cross-Temporal 3DGS), a novel framework for efficiently reconstructing and updating 3D scenes across different time periods, using sparse images and previously captured scene priors. Our approach comprises three stages: 1) Cross-temporal camera alignment for estimating and aligning camera poses across different timestamps; 2) Interference-based confidence initialization to identify unchanged regions between timestamps, thereby guiding updates; and 3) Progressive cross-temporal optimization, which iteratively integrates historical prior information into the 3D scene to enhance reconstruction quality. Our method supports non-continuous capture, enabling not only updates using new sparse views to refine existing scenes, but also recovering past scenes from limited data with the help of current captures. Furthermore, we demonstrate the potential of this approach to achieve temporal changes using only sparse images, which can later be reconstructed into detailed 3D representations as needed. Experimental results show significant improvements over baseline methods in reconstruction quality and data efficiency, making this approach a promising solution for scene versioning, cross-temporal digital twins, and long-term spatial documentation.
>
---
#### [new 091] Better, Stronger, Faster: Tackling the Trilemma in MLLM-based Segmentation with Simultaneous Textual Mask Prediction
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLM）图像分割中的“三难困境”——兼顾对话能力、分割精度与推理速度。提出STAMP模型，采用并行化全掩码预测机制，实现文本生成与掩码预测解耦，在单次前向传播中完成高质量分割，显著提升性能与速度，同时保持强对话能力。**

- **链接: [https://arxiv.org/pdf/2512.00395v1](https://arxiv.org/pdf/2512.00395v1)**

> **作者:** Jiazhen Liu; Mingkuan Feng; Long Chen
>
> **摘要:** Integrating segmentation into Multimodal Large Language Models (MLLMs) presents a core trilemma: simultaneously preserving dialogue ability, achieving high segmentation performance, and ensuring fast inference. Prevailing paradigms are forced into a compromise. Embedding prediction methods introduce a conflicting pixel-level objective that degrades the MLLM's general dialogue abilities. The alternative, next-token prediction, reframes segmentation as an autoregressive task, which preserves dialogue but forces a trade-off between poor segmentation performance with sparse outputs or prohibitive inference speeds with rich ones. We resolve this trilemma with all-mask prediction, a novel paradigm that decouples autoregressive dialogue generation from non-autoregressive mask prediction. We present STAMP: Simultaneous Textual All-Mask Prediction, an MLLM that embodies this paradigm. After generating a textual response, STAMP predicts an entire segmentation mask in a single forward pass by treating it as a parallel "fill-in-the-blank" task over image patches. This design maintains the MLLM's dialogue ability by avoiding conflicting objectives, enables high segmentation performance by leveraging rich, bidirectional spatial context for all mask tokens, and achieves exceptional speed. Extensive experiments show that STAMP significantly outperforms state-of-the-art methods across multiple segmentation benchmarks, providing a solution that excels in dialogue, segmentation, and speed without compromise.
>
---
#### [new 092] Multilingual Training-Free Remote Sensing Image Captioning
- **分类: cs.CV**

- **简介: 该论文针对遥感图像描述任务，解决现有方法依赖大量标注数据且仅支持英文的问题。提出无需训练的多语言生成方法，基于检索增强提示，结合图重排序提升内容连贯性，实现跨语言零样本生成，显著提升多语言泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.00887v1](https://arxiv.org/pdf/2512.00887v1)**

> **作者:** Carlos Rebelo; Gil Rocha; João Daniel Silva; Bruno Martins
>
> **摘要:** Remote sensing image captioning has advanced rapidly through encoder--decoder models, although the reliance on large annotated datasets and the focus on English restricts global applicability. To address these limitations, we propose the first training-free multilingual approach, based on retrieval-augmented prompting. For a given aerial image, we employ a domain-adapted SigLIP2 encoder to retrieve related captions and few-shot examples from a datastore, which are then provided to a language model. We explore two variants: an image-blind setup, where a multilingual Large Language Model (LLM) generates the caption from textual prompts alone, and an image-aware setup, where a Vision--Language Model (VLM) jointly processes the prompt and the input image. To improve the coherence of the retrieved content, we introduce a graph-based re-ranking strategy using PageRank on a graph of images and captions. Experiments on four benchmark datasets across ten languages demonstrate that our approach is competitive with fully supervised English-only systems and generalizes to other languages. Results also highlight the importance of re-ranking with PageRank, yielding up to 35% improvements in performance metrics. Additionally, it was observed that while VLMs tend to generate visually grounded but lexically diverse captions, LLMs can achieve stronger BLEU and CIDEr scores. Lastly, directly generating captions in the target language consistently outperforms other translation-based strategies. Overall, our work delivers one of the first systematic evaluations of multilingual, training-free captioning for remote sensing imagery, advancing toward more inclusive and scalable multimodal Earth observation systems.
>
---
#### [new 093] AirSim360: A Panoramic Simulation Platform within Drone View
- **分类: cs.CV**

- **简介: 该论文提出AirSim360，一个面向无人机视角的全景仿真平台，旨在解决360°空间理解缺乏大规模多样化数据的问题。通过像素级标注、交互式行人建模与自动路径生成，实现4D全景场景模拟，支持导航等任务。已收集超6万张全景样本，平台将开源。**

- **链接: [https://arxiv.org/pdf/2512.02009v1](https://arxiv.org/pdf/2512.02009v1)**

> **作者:** Xian Ge; Yuling Pan; Yuhang Zhang; Xiang Li; Weijun Zhang; Dizhe Zhang; Zhaoliang Wan; Xin Lin; Xiangkai Zhang; Juntao Liang; Jason Li; Wenjie Jiang; Bo Du; Ming-Hsuan Yang; Lu Qi
>
> **备注:** Project Website: https://insta360-research-team.github.io/AirSim360-website/
>
> **摘要:** The field of 360-degree omnidirectional understanding has been receiving increasing attention for advancing spatial intelligence. However, the lack of large-scale and diverse data remains a major limitation. In this work, we propose AirSim360, a simulation platform for omnidirectional data from aerial viewpoints, enabling wide-ranging scene sampling with drones. Specifically, AirSim360 focuses on three key aspects: a render-aligned data and labeling paradigm for pixel-level geometric, semantic, and entity-level understanding; an interactive pedestrian-aware system for modeling human behavior; and an automated trajectory generation paradigm to support navigation tasks. Furthermore, we collect more than 60K panoramic samples and conduct extensive experiments across various tasks to demonstrate the effectiveness of our simulator. Unlike existing simulators, our work is the first to systematically model the 4D real world under an omnidirectional setting. The entire platform, including the toolkit, plugins, and collected datasets, will be made publicly available at https://insta360-research-team.github.io/AirSim360-website.
>
---
#### [new 094] EGG-Fusion: Efficient 3D Reconstruction with Geometry-aware Gaussian Surfel on the Fly
- **分类: cs.CV**

- **简介: 该论文针对实时3D重建中计算效率低与传感器噪声敏感导致几何精度差的问题，提出EGG-Fusion系统。通过几何感知的可微高斯表面贴片映射与基于信息滤波的融合方法，实现高精度、实时（24 FPS）重建，在Replica和ScanNet++上误差达0.6cm，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.01296v1](https://arxiv.org/pdf/2512.01296v1)**

> **作者:** Xiaokun Pan; Zhenzhe Li; Zhichao Ye; Hongjia Zhai; Guofeng Zhang
>
> **备注:** SIGGRAPH ASIA 2025
>
> **摘要:** Real-time 3D reconstruction is a fundamental task in computer graphics. Recently, differentiable-rendering-based SLAM system has demonstrated significant potential, enabling photorealistic scene rendering through learnable scene representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Current differentiable rendering methods face dual challenges in real-time computation and sensor noise sensitivity, leading to degraded geometric fidelity in scene reconstruction and limited practicality. To address these challenges, we propose a novel real-time system EGG-Fusion, featuring robust sparse-to-dense camera tracking and a geometry-aware Gaussian surfel mapping module, introducing an information filter-based fusion method that explicitly accounts for sensor noise to achieve high-precision surface reconstruction. The proposed differentiable Gaussian surfel mapping effectively models multi-view consistent surfaces while enabling efficient parameter optimization. Extensive experimental results demonstrate that the proposed system achieves a surface reconstruction error of 0.6\textit{cm} on standardized benchmark datasets including Replica and ScanNet++, representing over 20\% improvement in accuracy compared to state-of-the-art (SOTA) GS-based methods. Notably, the system maintains real-time processing capabilities at 24 FPS, establishing it as one of the most accurate differentiable-rendering-based real-time reconstruction systems. Project Page: https://zju3dv.github.io/eggfusion/
>
---
#### [new 095] Doppler-Enhanced Deep Learning: Improving Thyroid Nodule Segmentation with YOLOv5 Instance Segmentation
- **分类: cs.CV; cs.AI; cs.CE; cs.LG; cs.PF**

- **简介: 该论文针对甲状腺结节自动分割任务，提出基于YOLOv5的实例分割方法。通过对比不同模型在含与不含多普勒超声图像的数据集上的表现，发现引入多普勒信息可显著提升分割精度，最高达Dice 91%、mAP 0.87，验证了其在临床辅助诊断中的潜力。**

- **链接: [https://arxiv.org/pdf/2512.00639v1](https://arxiv.org/pdf/2512.00639v1)**

> **作者:** Mahmoud El Hussieni
>
> **摘要:** The increasing prevalence of thyroid cancer globally has led to the development of various computer-aided detection methods. Accurate segmentation of thyroid nodules is a critical first step in the development of AI-assisted clinical decision support systems. This study focuses on instance segmentation of thyroid nodules using YOLOv5 algorithms on ultrasound images. We evaluated multiple YOLOv5 variants (Nano, Small, Medium, Large, and XLarge) across two dataset versions, with and without doppler images. The YOLOv5-Large algorithm achieved the highest performance with a dice score of 91\% and mAP of 0.87 on the dataset including doppler images. Notably, our results demonstrate that doppler images, typically excluded by physicians, can significantly improve segmentation performance. The YOLOv5-Small model achieved 79\% dice score when doppler images were excluded, while including them improved performance across all model variants. These findings suggest that instance segmentation with YOLOv5 provides an effective real-time approach for thyroid nodule detection, with potential clinical applications in automated diagnostic systems.
>
---
#### [new 096] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards
- **分类: cs.CV**

- **简介: 该论文聚焦多主体个性化图像生成任务，针对现有模型在多主体场景下一致性差、文本控制弱的问题，提出基于成对主体一致性的强化学习框架与可扩展数据生成管道，通过高质量多主体数据与奖励机制提升生成效果。**

- **链接: [https://arxiv.org/pdf/2512.01236v1](https://arxiv.org/pdf/2512.01236v1)**

> **作者:** Shulei Wang; Longhui Wei; Xin He; Jianbo Ouyang; Hui Lu; Zhou Zhao; Qi Tian
>
> **摘要:** Personalized generation models for a single subject have demonstrated remarkable effectiveness, highlighting their significant potential. However, when extended to multiple subjects, existing models often exhibit degraded performance, particularly in maintaining subject consistency and adhering to textual prompts. We attribute these limitations to the absence of high-quality multi-subject datasets and refined post-training strategies. To address these challenges, we propose a scalable multi-subject data generation pipeline that leverages powerful single-subject generation models to construct diverse and high-quality multi-subject training data. Through this dataset, we first enable single-subject personalization models to acquire knowledge of synthesizing multi-image and multi-subject scenarios. Furthermore, to enhance both subject consistency and text controllability, we design a set of Pairwise Subject-Consistency Rewards and general-purpose rewards, which are incorporated into a refined reinforcement learning stage. To comprehensively evaluate multi-subject personalization, we introduce a new benchmark that assesses model performance using seven subsets across three dimensions. Extensive experiments demonstrate the effectiveness of our approach in advancing multi-subject personalized image generation. Github Link: https://github.com/wang-shulei/PSR
>
---
#### [new 097] HanDyVQA: A Video QA Benchmark for Fine-Grained Hand-Object Interaction Dynamics
- **分类: cs.CV**

- **简介: 该论文提出HanDyVQA，一个细粒度视频问答基准，聚焦手物交互动态。针对现有数据集在时空细节和部分级变化上的不足，构建包含六类问题的11.1K QA对及10.3K分割掩码，评估模型对动作、状态变化等细粒度理解能力。实验表明当前模型性能远低于人类，揭示空间关系、运动与部件几何理解仍是挑战。**

- **链接: [https://arxiv.org/pdf/2512.00885v1](https://arxiv.org/pdf/2512.00885v1)**

> **作者:** Masatoshi Tateno; Gido Kato; Hirokatsu Kataoka; Yoichi Sato; Takuma Yagi
>
> **备注:** Project page: https://masatate.github.io/HanDyVQA-project-page/
>
> **摘要:** Hand-object interaction (HOI) inherently involves dynamics where human manipulations produce distinct spatio-temporal effects on objects. However, existing semantic HOI benchmarks focused either on manipulation or on the resulting effects at a coarse level, lacking fine-grained spatio-temporal reasoning to capture the underlying dynamics in HOI. We introduce HanDyVQA, a fine-grained video question-answering benchmark that comprehensively covers both the manipulation and effect aspects of HOI. HanDyVQA comprises six complementary question types (Action, Process, Objects, Location, State Change, and Object Parts), totalling 11.1K multiple-choice QA pairs. Collected QA pairs recognizing manipulation styles, hand/object motions, and part-level state changes. HanDyVQA also includes 10.3K segmentation masks for Objects and Object Parts questions, enabling the evaluation of object/part-level reasoning in video object segmentation. We evaluated recent video foundation models on our benchmark and found that even the best-performing model, Gemini-2.5-Pro, reached only 73% average accuracy, which is far from human performance (97%). Further analysis shows the remaining challenges in spatial relationship, motion, and part-level geometric understanding. We also found that integrating explicit HOI-related cues into visual features improves performance, offering insights for developing future models with a deeper understanding of HOI dynamics.
>
---
#### [new 098] Textured Geometry Evaluation: Perceptual 3D Textured Shape Metric via 3D Latent-Geometry Network
- **分类: cs.CV**

- **简介: 该论文针对3D纹理模型的保真度评估问题，提出无需渲染的TGE方法，联合几何与颜色信息评估真实世界失真下的3D网格质量。通过构建人类标注的真实失真数据集，解决了现有方法在视角敏感、域不匹配上的缺陷，显著提升评价准确性。**

- **链接: [https://arxiv.org/pdf/2512.01380v1](https://arxiv.org/pdf/2512.01380v1)**

> **作者:** Tianyu Luan; Xuelu Feng; Zixin Zhu; Phani Nuney; Sheng Liu; Xuan Gong; David Doermann; Chunming Qiao; Junsong Yuan
>
> **备注:** Accepted by AAAI26
>
> **摘要:** Textured high-fidelity 3D models are crucial for games, AR/VR, and film, but human-aligned evaluation methods still fall behind despite recent advances in 3D reconstruction and generation. Existing metrics, such as Chamfer Distance, often fail to align with how humans evaluate the fidelity of 3D shapes. Recent learning-based metrics attempt to improve this by relying on rendered images and 2D image quality metrics. However, these approaches face limitations due to incomplete structural coverage and sensitivity to viewpoint choices. Moreover, most methods are trained on synthetic distortions, which differ significantly from real-world distortions, resulting in a domain gap. To address these challenges, we propose a new fidelity evaluation method that is based directly on 3D meshes with texture, without relying on rendering. Our method, named Textured Geometry Evaluation TGE, jointly uses the geometry and color information to calculate the fidelity of the input textured mesh with comparison to a reference colored shape. To train and evaluate our metric, we design a human-annotated dataset with real-world distortions. Experiments show that TGE outperforms rendering-based and geometry-only methods on real-world distortion dataset.
>
---
#### [new 099] CourtMotion: Learning Event-Driven Motion Representations from Skeletal Data for Basketball
- **分类: cs.CV; cs.MA**

- **简介: 该论文提出CourtMotion，用于篮球比赛中事件与动作的分析与预测。针对传统方法仅依赖位置数据、忽略动作语义的问题，通过图神经网络捕捉骨骼运动细节，并结合Transformer建模球员交互，引入事件投影头关联动作与比赛事件。在NBA数据上显著提升轨迹预测与多项篮球分析任务性能。**

- **链接: [https://arxiv.org/pdf/2512.01478v1](https://arxiv.org/pdf/2512.01478v1)**

> **作者:** Omer Sela; Michael Chertok; Lior Wolf
>
> **摘要:** This paper presents CourtMotion, a spatiotemporal modeling framework for analyzing and predicting game events and plays as they develop in professional basketball. Anticipating basketball events requires understanding both physical motion patterns and their semantic significance in the context of the game. Traditional approaches that use only player positions fail to capture crucial indicators such as body orientation, defensive stance, or shooting preparation motions. Our two-stage approach first processes skeletal tracking data through Graph Neural Networks to capture nuanced motion patterns, then employs a Transformer architecture with specialized attention mechanisms to model player interactions. We introduce event projection heads that explicitly connect player movements to basketball events like passes, shots, and steals, training the model to associate physical motion patterns with their tactical purposes. Experiments on NBA tracking data demonstrate significant improvements over position-only baselines: 35% reduction in trajectory prediction error compared to state-of-the-art position-based models and consistent performance gains across key basketball analytics tasks. The resulting pretrained model serves as a powerful foundation for multiple downstream tasks, with pick detection, shot taker identification, assist prediction, shot location classification, and shot type recognition demonstrating substantial improvements over existing methods.
>
---
#### [new 100] PointNet4D: A Lightweight 4D Point Cloud Video Backbone for Online and Offline Perception in Robotic Applications
- **分类: cs.CV**

- **简介: 该论文针对机器人感知中动态4D点云视频处理难题，提出轻量级PointNet4D框架。通过混合Mamba-Transformer时序融合与4DMAP预训练策略，实现高效在线/离线推理，显著提升时序建模能力，在9项任务中表现优异，推动了4D感知在机器人应用中的落地。**

- **链接: [https://arxiv.org/pdf/2512.01383v1](https://arxiv.org/pdf/2512.01383v1)**

> **作者:** Yunze Liu; Zifan Wang; Peiran Wu; Jiayang Ao
>
> **备注:** Accepted by WACV2026
>
> **摘要:** Understanding dynamic 4D environments-3D space evolving over time-is critical for robotic and interactive systems. These applications demand systems that can process streaming point cloud video in real-time, often under resource constraints, while also benefiting from past and present observations when available. However, current 4D backbone networks rely heavily on spatiotemporal convolutions and Transformers, which are often computationally intensive and poorly suited to real-time applications. We propose PointNet4D, a lightweight 4D backbone optimized for both online and offline settings. At its core is a Hybrid Mamba-Transformer temporal fusion block, which integrates the efficient state-space modeling of Mamba and the bidirectional modeling power of Transformers. This enables PointNet4D to handle variable-length online sequences efficiently across different deployment scenarios. To enhance temporal understanding, we introduce 4DMAP, a frame-wise masked auto-regressive pretraining strategy that captures motion cues across frames. Our extensive evaluations across 9 tasks on 7 datasets, demonstrating consistent improvements across diverse domains. We further demonstrate PointNet4D's utility by building two robotic application systems: 4D Diffusion Policy and 4D Imitation Learning, achieving substantial gains on the RoboTwin and HandoverSim benchmarks.
>
---
#### [new 101] SatireDecoder: Visual Cascaded Decoupling for Enhancing Satirical Image Comprehension
- **分类: cs.CV**

- **简介: 该论文针对视觉讽刺图像理解任务，解决现有模型难以融合局部与全局语义、易产生误解和幻觉的问题。提出无需训练的SatireDecoder框架，通过多智能体视觉级联解耦生成细粒度语义表示，并结合不确定性引导的思维链推理，提升理解准确率，减少错误。**

- **链接: [https://arxiv.org/pdf/2512.00582v1](https://arxiv.org/pdf/2512.00582v1)**

> **作者:** Yue Jiang; Haiwei Xue; Minghao Han; Mingcheng Li; Xiaolu Hou; Dingkang Yang; Lihua Zhang; Xu Zheng
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Satire, a form of artistic expression combining humor with implicit critique, holds significant social value by illuminating societal issues. Despite its cultural and societal significance, satire comprehension, particularly in purely visual forms, remains a challenging task for current vision-language models. This task requires not only detecting satire but also deciphering its nuanced meaning and identifying the implicated entities. Existing models often fail to effectively integrate local entity relationships with global context, leading to misinterpretation, comprehension biases, and hallucinations. To address these limitations, we propose SatireDecoder, a training-free framework designed to enhance satirical image comprehension. Our approach proposes a multi-agent system performing visual cascaded decoupling to decompose images into fine-grained local and global semantic representations. In addition, we introduce a chain-of-thought reasoning strategy guided by uncertainty analysis, which breaks down the complex satire comprehension process into sequential subtasks with minimized uncertainty. Our method significantly improves interpretive accuracy while reducing hallucinations. Experimental results validate that SatireDecoder outperforms existing baselines in comprehending visual satire, offering a promising direction for vision-language reasoning in nuanced, high-level semantic tasks.
>
---
#### [new 102] VideoScoop: A Non-Traditional Domain-Independent Framework For Video Analysis
- **分类: cs.CV; cs.DB**

- **简介: 该论文提出VideoScoop框架，解决视频情境分析（VSA）中通用性差、依赖人工或定制算法的问题。通过提取视频内容并用关系模型与图模型表示，支持连续查询和复杂情境检测，采用参数化模板实现跨领域泛化，实验验证其在助老、监控、安防三领域的有效性。**

- **链接: [https://arxiv.org/pdf/2512.01769v1](https://arxiv.org/pdf/2512.01769v1)**

> **作者:** Hafsa Billah
>
> **备注:** This is a report submitted as part of PhD proposal defense of Hafsa Billah
>
> **摘要:** Automatically understanding video contents is important for several applications in Civic Monitoring (CM), general Surveillance (SL), Assisted Living (AL), etc. Decades of Image and Video Analysis (IVA) research have advanced tasks such as content extraction (e.g., object recognition and tracking). Identifying meaningful activities or situations (e.g., two objects coming closer) remains difficult and cannot be achieved by content extraction alone. Currently, Video Situation Analysis (VSA) is done manually with a human in the loop, which is error-prone and labor-intensive, or through custom algorithms designed for specific video types or situations. These algorithms are not general-purpose and require a new algorithm/software for each new situation or video from a new domain. This report proposes a general-purpose VSA framework that overcomes the above limitations. Video contents are extracted once using state-of-the-art Video Content Extraction technologies. They are represented using two alternative models -- the extended relational model (R++) and graph models. When represented using R++, the extracted contents can be used as data streams, enabling Continuous Query Processing via the proposed Continuous Query Language for Video Analysis. The graph models complement this by enabling the detection of situations that are difficult or impossible to detect using the relational model alone. Existing graph algorithms and newly developed algorithms support a wide variety of situation detection. To support domain independence, primitive situation variants across domains are identified and expressed as parameterized templates. Extensive experiments were conducted across several interesting situations from three domains -- AL, CM, and SL-- to evaluate the accuracy, efficiency, and robustness of the proposed approach using a dataset of videos of varying lengths from these domains.
>
---
#### [new 103] Scaling Down to Scale Up: Towards Operationally-Efficient and Deployable Clinical Models via Cross-Modal Low-Rank Adaptation for Medical Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对医学影像中大型视觉语言模型部署难的问题，提出MedCT-VLM框架，通过低秩适配（LoRA）实现参数高效微调。在18种胸部病理的零样本分类任务中，显著提升性能，验证了其在医疗场景下高效、可部署的应用潜力。**

- **链接: [https://arxiv.org/pdf/2512.00597v1](https://arxiv.org/pdf/2512.00597v1)**

> **作者:** Thuraya Alzubaidi; Farhad R. Nezami; Muzammil Behzad
>
> **摘要:** Foundation models trained via vision-language pretraining have demonstrated strong zero-shot capabilities across diverse image domains, yet their application to volumetric medical imaging remains limited. We introduce MedCT-VLM: Medical CT Vision-Language Model, a parameter-efficient vision-language framework designed to adapt large-scale CT foundation models for downstream clinical tasks. MedCT-VLM uses a parameter-efficient approach to adapt CT-CLIP, a contrastive vision-language model trained on 25,692 chest CT volumes, for multi-label pathology classification using Low-Rank Adaptation (LoRA). Rather than fine-tuning the model's 440 M parameters directly, we insert low-rank decomposition matrices into attention layers of both vision and text encoders, training only 1.67M parameters (0.38\% of total). We evaluate on zero-shot classification across 18 thoracic pathologies, where the model must align CT embeddings with unseen text prompts at inference without task-specific training. LoRA fine-tuning improves mean AUROC from 61.3\% to 68.9\% (+7.6 pp), accuracy from 67.2\% to 73.6\% (+6.4 pp), and macro-F1 from 32.1\% to 36.9\% (+4.8 pp). These results demonstrate that parameter-efficient methods can effectively transfer large-scale pretraining to downstream medical imaging tasks, particularly for zero-shot scenarios where labeled data is scarce.
>
---
#### [new 104] POLARIS: Projection-Orthogonal Least Squares for Robust and Adaptive Inversion in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对扩散模型中图像重建时的噪声近似误差问题，提出POLARIS方法。通过将引导尺度ω设为逐步可变，从根源上减少误差积累，实现鲁棒自适应反演。仅需一行代码即可显著提升反演质量，有效改善图像编辑与修复任务的精度。**

- **链接: [https://arxiv.org/pdf/2512.00369v1](https://arxiv.org/pdf/2512.00369v1)**

> **作者:** Wenshuo Chen; Haosen Li; Shaofeng Liang; Lei Wang; Haozhe Jia; Kaishen Yuan; Jieming Wu; Bowen Tian; Yutao Yue
>
> **摘要:** The Inversion-Denoising Paradigm, which is based on diffusion models, excels in diverse image editing and restoration tasks. We revisit its mechanism and reveal a critical, overlooked factor in reconstruction degradation: the approximate noise error. This error stems from approximating the noise at step t with the prediction at step t-1, resulting in severe error accumulation throughout the inversion process. We introduce Projection-Orthogonal Least Squares for Robust and Adaptive Inversion (POLARIS), which reformulates inversion from an error-compensation problem into an error-origin problem. Rather than optimizing embeddings or latent codes to offset accumulated drift, POLARIS treats the guidance scale ω as a step-wise variable and derives a mathematically grounded formula to minimize inversion error at each step. Remarkably, POLARIS improves inversion latent quality with just one line of code. With negligible performance overhead, it substantially mitigates noise approximation errors and consistently improves the accuracy of downstream tasks.
>
---
#### [new 105] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对工业场景中大量未标注视频数据难以用于视觉-语言-动作（VLA）模型预训练的问题，提出一种端到端无监督框架。通过轻量级运动编码器与基于“潜在动作能量”的动作分割算法，自动提取语义一致的动作原型，生成可直接用于VLA预训练的结构化数据。**

- **链接: [https://arxiv.org/pdf/2511.21428v1](https://arxiv.org/pdf/2511.21428v1)**

> **作者:** Jiajie Zhang; Sören Schwertfeger; Alexander Kleiner
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** We present a novel unsupervised framework to unlock vast unlabeled human demonstration data from continuous industrial video streams for Vision-Language-Action (VLA) model pre-training. Our method first trains a lightweight motion tokenizer to encode motion dynamics, then employs an unsupervised action segmenter leveraging a novel "Latent Action Energy" metric to discover and segment semantically coherent action primitives. The pipeline outputs both segmented video clips and their corresponding latent action sequences, providing structured data directly suitable for VLA pre-training. Evaluations on public benchmarks and a proprietary electric motor assembly dataset demonstrate effective segmentation of key tasks performed by humans at workstations. Further clustering and quantitative assessment via a Vision-Language Model confirm the semantic coherence of the discovered action primitives. To our knowledge, this is the first fully automated end-to-end system for extracting and organizing VLA pre-training data from unstructured industrial videos, offering a scalable solution for embodied AI integration in manufacturing.
>
---
#### [new 106] COACH: Collaborative Agents for Contextual Highlighting - A Multi-Agent Framework for Sports Video Analysis
- **分类: cs.CV**

- **简介: 该论文提出COACH框架，解决体育视频分析中时序上下文理解难、泛化性差、可解释性低的问题。通过多智能体系统，将不同分析任务分解为可重构的“认知工具”，实现从局部动作到全局策略的灵活推理，支持短时问答与长时摘要生成，提升系统的适应性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.01853v1](https://arxiv.org/pdf/2512.01853v1)**

> **作者:** Tsz-To Wong; Ching-Chun Huang; Hong-Han Shuai
>
> **备注:** Accepted by AAAI 2026 Workshop LaMAS
>
> **摘要:** Intelligent sports video analysis demands a comprehensive understanding of temporal context, from micro-level actions to macro-level game strategies. Existing end-to-end models often struggle with this temporal hierarchy, offering solutions that lack generalization, incur high development costs for new tasks, and suffer from poor interpretability. To overcome these limitations, we propose a reconfigurable Multi-Agent System (MAS) as a foundational framework for sports video understanding. In our system, each agent functions as a distinct "cognitive tool" specializing in a specific aspect of analysis. The system's architecture is not confined to a single temporal dimension or task. By leveraging iterative invocation and flexible composition of these agents, our framework can construct adaptive pipelines for both short-term analytic reasoning (e.g., Rally QA) and long-term generative summarization (e.g., match summaries). We demonstrate the adaptability of this framework using two representative tasks in badminton analysis, showcasing its ability to bridge fine-grained event detection and global semantic organization. This work presents a paradigm shift towards a flexible, scalable, and interpretable system for robust, cross-task sports video intelligence.The project homepage is available at https://aiden1020.github.io/COACH-project-page
>
---
#### [new 107] PolarGS: Polarimetric Cues for Ambiguity-Free Gaussian Splatting with Accurate Geometry Recovery
- **分类: cs.CV**

- **简介: 该论文针对3D高斯点云渲染中反射与无纹理区域的几何模糊问题，提出PolarGS方法。通过引入偏振信息（DoLP、A/DoLP），设计偏振引导的光度校正与增强的高斯稠密化机制，提升重建精度与一致性，实现无歧义的几何恢复。**

- **链接: [https://arxiv.org/pdf/2512.00794v1](https://arxiv.org/pdf/2512.00794v1)**

> **作者:** Bo Guo; Sijia Wen; Yifan Zhao; Jia Li; Zhiming Zheng
>
> **摘要:** Recent advances in surface reconstruction for 3D Gaussian Splatting (3DGS) have enabled remarkable geometric accuracy. However, their performance degrades in photometrically ambiguous regions such as reflective and textureless surfaces, where unreliable cues disrupt photometric consistency and hinder accurate geometry estimation. Reflected light is often partially polarized in a manner that reveals surface orientation, making polarization an optic complement to photometric cues in resolving such ambiguities. Therefore, we propose PolarGS, an optics-aware extension of RGB-based 3DGS that leverages polarization as an optical prior to resolve photometric ambiguities and enhance reconstruction accuracy. Specifically, we introduce two complementary modules: a polarization-guided photometric correction strategy, which ensures photometric consistency by identifying reflective regions via the Degree of Linear Polarization (DoLP) and refining reflective Gaussians with Color Refinement Maps; and a polarization-enhanced Gaussian densification mechanism for textureless area geometry recovery, which integrates both Angle and Degree of Linear Polarization (A/DoLP) into a PatchMatch-based depth completion process. This enables the back-projection and fusion of new Gaussians, leading to more complete reconstruction. PolarGS is framework-agnostic and achieves superior geometric accuracy compared to state-of-the-art methods.
>
---
#### [new 108] PEFT-DML: Parameter-Efficient Fine-Tuning Deep Metric Learning for Robust Multi-Modal 3D Object Detection in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶中多模态3D目标检测的鲁棒性问题，提出PEFT-DML框架。通过参数高效微调技术将多种传感器数据映射到共享空间，提升对传感器失效和环境变化的适应能力，显著增强检测精度与训练效率。**

- **链接: [https://arxiv.org/pdf/2512.00060v1](https://arxiv.org/pdf/2512.00060v1)**

> **作者:** Abdolazim Rezaei; Mehdi Sookhak
>
> **摘要:** This study introduces PEFT-DML, a parameter-efficient deep metric learning framework for robust multi-modal 3D object detection in autonomous driving. Unlike conventional models that assume fixed sensor availability, PEFT-DML maps diverse modalities (LiDAR, radar, camera, IMU, GNSS) into a shared latent space, enabling reliable detection even under sensor dropout or unseen modality class combinations. By integrating Low-Rank Adaptation (LoRA) and adapter layers, PEFT-DML achieves significant training efficiency while enhancing robustness to fast motion, weather variability, and domain shifts. Experiments on benchmarks nuScenes demonstrate superior accuracy.
>
---
#### [new 109] Dynamic-eDiTor: Training-Free Text-Driven 4D Scene Editing with Multimodal Diffusion Transformer
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Dynamic-eDiTor，一种无需训练的文本驱动4D场景编辑框架。针对现有方法在多视角与时间上一致性差的问题，利用多模态扩散变换器与4DGS，通过时空子网格注意力和上下文令牌传播，实现无缝、全局一致的动态场景编辑，直接优化预训练4DGS模型。**

- **链接: [https://arxiv.org/pdf/2512.00677v1](https://arxiv.org/pdf/2512.00677v1)**

> **作者:** Dong In Lee; Hyungjun Doh; Seunggeun Chi; Runlin Duan; Sangpil Kim; Karthik Ramani
>
> **备注:** 4D Scene Editing
>
> **摘要:** Recent progress in 4D representations, such as Dynamic NeRF and 4D Gaussian Splatting (4DGS), has enabled dynamic 4D scene reconstruction. However, text-driven 4D scene editing remains under-explored due to the challenge of ensuring both multi-view and temporal consistency across space and time during editing. Existing studies rely on 2D diffusion models that edit frames independently, often causing motion distortion, geometric drift, and incomplete editing. We introduce Dynamic-eDiTor, a training-free text-driven 4D editing framework leveraging Multimodal Diffusion Transformer (MM-DiT) and 4DGS. This mechanism consists of Spatio-Temporal Sub-Grid Attention (STGA) for locally consistent cross-view and temporal fusion, and Context Token Propagation (CTP) for global propagation via token inheritance and optical-flow-guided token replacement. Together, these components allow Dynamic-eDiTor to perform seamless, globally consistent multi-view video without additional training and directly optimize pre-trained source 4DGS. Extensive experiments on multi-view video dataset DyNeRF demonstrate that our method achieves superior editing fidelity and both multi-view and temporal consistency prior approaches. Project page for results and code: https://di-lee.github.io/dynamic-eDiTor/
>
---
#### [new 110] MDiff4STR: Mask Diffusion Model for Scene Text Recognition
- **分类: cs.CV**

- **简介: 该论文将掩码扩散模型（MDM）引入场景文本识别（STR）任务，针对其精度低于自回归模型（ARM）的问题，提出MDiff4STR。通过六种噪声策略对齐训练与推理，并引入令牌替换噪声机制，缓解过自信预测。实验表明，该方法在多种场景下超越SOTA ARM模型，兼具高精度与快速推理。**

- **链接: [https://arxiv.org/pdf/2512.01422v1](https://arxiv.org/pdf/2512.01422v1)**

> **作者:** Yongkun Du; Miaomiao Zhao; Songlin Fan; Zhineng Chen; Caiyan Jia; Yu-Gang Jiang
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Mask Diffusion Models (MDMs) have recently emerged as a promising alternative to auto-regressive models (ARMs) for vision-language tasks, owing to their flexible balance of efficiency and accuracy. In this paper, for the first time, we introduce MDMs into the Scene Text Recognition (STR) task. We show that vanilla MDM lags behind ARMs in terms of accuracy, although it improves recognition efficiency. To bridge this gap, we propose MDiff4STR, a Mask Diffusion model enhanced with two key improvement strategies tailored for STR. Specifically, we identify two key challenges in applying MDMs to STR: noising gap between training and inference, and overconfident predictions during inference. Both significantly hinder the performance of MDMs. To mitigate the first issue, we develop six noising strategies that better align training with inference behavior. For the second, we propose a token-replacement noise mechanism that provides a non-mask noise type, encouraging the model to reconsider and revise overly confident but incorrect predictions. We conduct extensive evaluations of MDiff4STR on both standard and challenging STR benchmarks, covering diverse scenarios including irregular, artistic, occluded, and Chinese text, as well as whether the use of pretraining. Across these settings, MDiff4STR consistently outperforms popular STR models, surpassing state-of-the-art ARMs in accuracy, while maintaining fast inference with only three denoising steps. Code: https://github.com/Topdu/OpenOCR.
>
---
#### [new 111] TransientTrack: Advanced Multi-Object Tracking and Classification of Cancer Cells with Transient Fluorescent Signals
- **分类: cs.CV; q-bio.CB; q-bio.QM**

- **简介: 该论文提出TransientTrack，一种基于深度学习的多目标跟踪与分类框架，用于处理具有瞬态荧光信号的多通道显微视频。针对传统方法无法检测细胞分裂与死亡的问题，该方法通过Transformer网络和多阶段匹配，实现细胞轨迹完整追踪，并结合卡尔曼滤波补全缺失轨迹，提升在复杂动态下的跟踪性能。**

- **链接: [https://arxiv.org/pdf/2512.01885v1](https://arxiv.org/pdf/2512.01885v1)**

> **作者:** Florian Bürger; Martim Dias Gomes; Nica Gutu; Adrián E. Granada; Noémie Moreau; Katarzyna Bozek
>
> **备注:** 13 pages, 7 figures, 2 tables. This work has been submitted to IEEE Transactions on Medical Imaging
>
> **摘要:** Tracking cells in time-lapse videos is an essential technique for monitoring cell population dynamics at a single-cell level. Current methods for cell tracking are developed on videos with mostly single, constant signals and do not detect pivotal events such as cell death. Here, we present TransientTrack, a deep learning-based framework for cell tracking in multi-channel microscopy video data with transient fluorescent signals that fluctuate over time following processes such as the circadian rhythm of cells. By identifying key cellular events - mitosis (cell division) and apoptosis (cell death) our method allows us to build complete trajectories, including cell lineage information. TransientTrack is lightweight and performs matching on cell detection embeddings directly, without the need for quantification of tracking-specific cell features. Furthermore, our approach integrates Transformer Networks, multi-stage matching using all detection boxes, and the interpolation of missing tracklets with the Kalman Filter. This unified framework achieves strong performance across diverse conditions, effectively tracking cells and capturing cell division and death. We demonstrate the use of TransientTrack in an analysis of the efficacy of a chemotherapeutic drug at a single-cell level. The proposed framework could further advance quantitative studies of cancer cell dynamics, enabling detailed characterization of treatment response and resistance mechanisms. The code is available at https://github.com/bozeklab/TransientTrack.
>
---
#### [new 112] Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对单目视频中人-物交互（HOI）的4D重建难题，提出4DHOISolver优化框架，利用稀疏人工标注的接触点实现高效、可扩展的高保真重建。构建了包含144类物体和103种动作的大规模Open4DHOI数据集，并验证了其在强化学习运动模仿中的有效性，揭示了自动接触预测仍是未解挑战。**

- **链接: [https://arxiv.org/pdf/2512.00960v1](https://arxiv.org/pdf/2512.00960v1)**

> **作者:** Boran Wen; Ye Lu; Keyan Wan; Sirui Wang; Jiahong Zhou; Junxuan Liang; Xinpeng Liu; Bang Xiao; Dingbang Huang; Ruiyang Liu; Yong-Lu Li
>
> **摘要:** Generalized robots must learn from diverse, large-scale human-object interactions (HOI) to operate robustly in the real world. Monocular internet videos offer a nearly limitless and readily available source of data, capturing an unparalleled diversity of human activities, objects, and environments. However, accurately and scalably extracting 4D interaction data from these in-the-wild videos remains a significant and unsolved challenge. Thus, in this work, we introduce 4DHOISolver, a novel and efficient optimization framework that constrains the ill-posed 4D HOI reconstruction problem by leveraging sparse, human-in-the-loop contact point annotations, while maintaining high spatio-temporal coherence and physical plausibility. Leveraging this framework, we introduce Open4DHOI, a new large-scale 4D HOI dataset featuring a diverse catalog of 144 object types and 103 actions. Furthermore, we demonstrate the effectiveness of our reconstructions by enabling an RL-based agent to imitate the recovered motions. However, a comprehensive benchmark of existing 3D foundation models indicates that automatically predicting precise human-object contact correspondences remains an unsolved problem, underscoring the immediate necessity of our human-in-the-loop strategy while posing an open challenge to the community. Data and code will be publicly available at https://wenboran2002.github.io/open4dhoi/
>
---
#### [new 113] Generative Video Motion Editing with 3D Point Tracks
- **分类: cs.CV**

- **简介: 该论文针对视频中相机与物体运动难以精确编辑的问题，提出一种基于3D点轨迹的视频生成框架。通过条件化生成模型，利用源视频与配对3D点轨迹实现联合运动编辑，有效保留时空一致性并处理遮挡，支持复杂运动变换，提升视频编辑的精度与创作自由度。**

- **链接: [https://arxiv.org/pdf/2512.02015v1](https://arxiv.org/pdf/2512.02015v1)**

> **作者:** Yao-Chih Lee; Zhoutong Zhang; Jiahui Huang; Jui-Hsien Wang; Joon-Young Lee; Jia-Bin Huang; Eli Shechtman; Zhengqi Li
>
> **备注:** Project page: https://edit-by-track.github.io
>
> **摘要:** Camera and object motions are central to a video's narrative. However, precisely editing these captured motions remains a significant challenge, especially under complex object movements. Current motion-controlled image-to-video (I2V) approaches often lack full-scene context for consistent video editing, while video-to-video (V2V) methods provide viewpoint changes or basic object translation, but offer limited control over fine-grained object motion. We present a track-conditioned V2V framework that enables joint editing of camera and object motion. We achieve this by conditioning a video generation model on a source video and paired 3D point tracks representing source and target motions. These 3D tracks establish sparse correspondences that transfer rich context from the source video to new motions while preserving spatiotemporal coherence. Crucially, compared to 2D tracks, 3D tracks provide explicit depth cues, allowing the model to resolve depth order and handle occlusions for precise motion editing. Trained in two stages on synthetic and real data, our model supports diverse motion edits, including joint camera/object manipulation, motion transfer, and non-rigid deformation, unlocking new creative potential in video editing.
>
---
#### [new 114] HiconAgent: History Context-aware Policy Optimization for GUI Agents
- **分类: cs.CV**

- **简介: 该论文针对GUI导航中历史信息利用效率低的问题，提出HiconAgent模型。通过动态上下文采样与锚点引导压缩机制，实现高效历史信息选择与利用，在保持性能的同时显著降低计算开销。**

- **链接: [https://arxiv.org/pdf/2512.01763v1](https://arxiv.org/pdf/2512.01763v1)**

> **作者:** Xurui Zhou; Gongwei Chen; Yuquan Xie; Zaijing Li; Kaiwen Zhou; Shuai Wang; Shuo Yang; Zhuotao Tian; Rui Shao
>
> **摘要:** Graphical User Interface (GUI) agents require effective use of historical context to perform sequential navigation tasks. While incorporating past actions and observations can improve decision making, naive use of full history leads to excessive computational overhead and distraction from irrelevant information. To address this, we introduce HiconAgent, a GUI agent trained with History Context-aware Policy Optimization (HCPO) for efficient and effective utilization of historical information. HCPO optimizes history usage in both sampling and policy updates through two complementary components: (1) Dynamic Context Sampling (DCS) presents the agent with variable length histories during sampling, enabling adaptive use of the most relevant context; (2) Anchor-guided History Compression (AHC) refines the policy update phase with a dual branch strategy where the compressed branch removes history observations while keeping history actions as information flow anchors. The compressed and uncompressed branches are coupled through a history-enhanced alignment loss to enforce consistent history usage while maintaining efficiency. Experiments on mainstream GUI navigation benchmarks demonstrate strong performance. Despite being smaller, HiconAgent-3B outperforms GUI-R1-7B by +8.46 percent grounding accuracy and +11.32 percent step success rate on GUI-Odyssey, while achieving comparable results on AndroidControl and AITW with up to 2.47x computational speedup and 60 percent FLOPs reduction.
>
---
#### [new 115] Neural Discrete Representation Learning for Sparse-View CBCT Reconstruction: From Algorithm Design to Prospective Multicenter Clinical Evaluation
- **分类: cs.CV**

- **简介: 该论文针对低剂量锥形束CT（CBCT）重建中图像质量下降的问题，提出DeepPriorCBCT框架，通过三阶段深度学习实现仅用1/6辐射剂量的诊断级重建。基于多中心大样本数据训练与前瞻性临床试验验证，结果表明其图像质量与标准方法相当，显著降低辐射风险。**

- **链接: [https://arxiv.org/pdf/2512.00873v1](https://arxiv.org/pdf/2512.00873v1)**

> **作者:** Haoshen Wang; Lei Chen; Wei-Hua Zhang; Linxia Wu; Yong Luo; Zengmao Wang; Yuan Xiong; Chengcheng Zhu; Wenjuan Tang; Xueyi Zhang; Wei Zhou; Xuhua Duan; Lefei Zhang; Gao-Jun Teng; Bo Du; Huangxuan Zhao
>
> **摘要:** Cone beam computed tomography (CBCT)-guided puncture has become an established approach for diagnosing and treating early- to mid-stage thoracic tumours, yet the associated radiation exposure substantially elevates the risk of secondary malignancies. Although multiple low-dose CBCT strategies have been introduced, none have undergone validation using large-scale multicenter retrospective datasets, and prospective clinical evaluation remains lacking. Here, we propose DeepPriorCBCT - a three-stage deep learning framework that achieves diagnostic-grade reconstruction using only one-sixth of the conventional radiation dose. 4102 patients with 8675 CBCT scans from 12 centers were included to develop and validate DeepPriorCBCT. Additionally, a prospective cross-over trial (Registry number: NCT07035977) which recruited 138 patients scheduled for percutaneous thoracic puncture was conducted to assess the model's clinical applicability. Assessment by 11 physicians confirmed that reconstructed images were indistinguishable from original scans. Moreover, diagnostic performance and overall image quality were comparable to those generated by standard reconstruction algorithms. In the prospective trial, five radiologists reported no significant differences in image quality or lesion assessment between DeepPriorCBCT and the clinical standard (all P>0.05). Likewise, 25 interventionalists expressed no preference between model-based and full-sampling images for surgical guidance (Kappa<0.2). Radiation exposure with DeepPriorCBCT was reduced to approximately one-sixth of that with the conventional approach, and collectively, the findings confirm that it enables high-quality CBCT reconstruction under sparse sampling conditions while markedly decreasing intraoperative radiation risk.
>
---
#### [new 116] OpenBox: Annotate Any Bounding Boxes in 3D
- **分类: cs.CV**

- **简介: 该论文提出OpenBox，解决3D目标检测中标注成本高、难以识别未见物体的问题。通过两阶段自动标注流程，利用2D视觉大模型对齐图像与点云，按刚性与运动状态自适应生成高质量3D边界框，无需自训练，显著提升标注精度与效率。**

- **链接: [https://arxiv.org/pdf/2512.01352v1](https://arxiv.org/pdf/2512.01352v1)**

> **作者:** In-Jae Lee; Mungyeom Kim; Kwonyoung Ryu; Pierre Musacchio; Jaesik Park
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Unsupervised and open-vocabulary 3D object detection has recently gained attention, particularly in autonomous driving, where reducing annotation costs and recognizing unseen objects are critical for both safety and scalability. However, most existing approaches uniformly annotate 3D bounding boxes, ignore objects' physical states, and require multiple self-training iterations for annotation refinement, resulting in suboptimal quality and substantial computational overhead. To address these challenges, we propose OpenBox, a two-stage automatic annotation pipeline that leverages a 2D vision foundation model. In the first stage, OpenBox associates instance-level cues from 2D images processed by a vision foundation model with the corresponding 3D point clouds via cross-modal instance alignment. In the second stage, it categorizes instances by rigidity and motion state, then generates adaptive bounding boxes with class-specific size statistics. As a result, OpenBox produces high-quality 3D bounding box annotations without requiring self-training. Experiments on the Waymo Open Dataset, the Lyft Level 5 Perception dataset, and the nuScenes dataset demonstrate improved accuracy and efficiency over baselines.
>
---
#### [new 117] Local and Global Context-and-Object-part-Aware Superpixel-based Data Augmentation for Deep Visual Recognition
- **分类: cs.CV**

- **简介: 该论文针对深度视觉识别中的数据增强问题，提出LGCOAMix方法。它通过超像素级的局部与全局上下文感知，实现更精准的图像混合，保留物体局部特征，避免标签不一致问题，提升分类与弱监督目标定位性能。**

- **链接: [https://arxiv.org/pdf/2512.00130v1](https://arxiv.org/pdf/2512.00130v1)**

> **作者:** Fadi Dornaika; Danyang Sun
>
> **摘要:** Cutmix-based data augmentation, which uses a cut-and-paste strategy, has shown remarkable generalization capabilities in deep learning. However, existing methods primarily consider global semantics with image-level constraints, which excessively reduces attention to the discriminative local context of the class and leads to a performance improvement bottleneck. Moreover, existing methods for generating augmented samples usually involve cutting and pasting rectangular or square regions, resulting in a loss of object part information. To mitigate the problem of inconsistency between the augmented image and the generated mixed label, existing methods usually require double forward propagation or rely on an external pre-trained network for object centering, which is inefficient. To overcome the above limitations, we propose LGCOAMix, an efficient context-aware and object-part-aware superpixel-based grid blending method for data augmentation. To the best of our knowledge, this is the first time that a label mixing strategy using a superpixel attention approach has been proposed for cutmix-based data augmentation. It is the first instance of learning local features from discriminative superpixel-wise regions and cross-image superpixel contrasts. Extensive experiments on various benchmark datasets show that LGCOAMix outperforms state-of-the-art cutmix-based data augmentation methods on classification tasks, {and weakly supervised object location on CUB200-2011.} We have demonstrated the effectiveness of LGCOAMix not only for CNN networks, but also for Transformer networks. Source codes are available at https://github.com/DanielaPlusPlus/LGCOAMix.
>
---
#### [new 118] GrndCtrl: Grounding World Models via Self-Supervised Reward Alignment
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对视频世界模型缺乏几何接地的问题，提出GrndCtrl框架，通过自监督奖励对齐实现物理可验证结构的建模。利用姿态循环一致性、深度重投影等多奖励机制，基于GRPO优化，提升模型在户外环境中的空间一致性和导航稳定性，解决生成模型与真实物理世界脱节的难题。**

- **链接: [https://arxiv.org/pdf/2512.01952v1](https://arxiv.org/pdf/2512.01952v1)**

> **作者:** Haoyang He; Jay Patrikar; Dong-Ki Kim; Max Smith; Daniel McGann; Ali-akbar Agha-mohammadi; Shayegan Omidshafiei; Sebastian Scherer
>
> **摘要:** Recent advances in video world modeling have enabled large-scale generative models to simulate embodied environments with high visual fidelity, providing strong priors for prediction, planning, and control. Yet, despite their realism, these models often lack geometric grounding, limiting their use in navigation tasks that require spatial coherence and long-horizon stability. We introduce Reinforcement Learning with World Grounding (RLWG), a self-supervised post-training framework that aligns pretrained world models with a physically verifiable structure through geometric and perceptual rewards. Analogous to reinforcement learning from verifiable feedback (RLVR) in language models, RLWG can use multiple rewards that measure pose cycle-consistency, depth reprojection, and temporal coherence. We instantiate this framework with GrndCtrl, a reward-aligned adaptation method based on Group Relative Policy Optimization (GRPO), yielding world models that maintain stable trajectories, consistent geometry, and reliable rollouts for embodied navigation. Like post-training alignment in large language models, GrndCtrl leverages verifiable rewards to bridge generative pretraining and grounded behavior, achieving superior spatial coherence and navigation stability over supervised fine-tuning in outdoor environments.
>
---
#### [new 119] MambaScope: Coarse-to-Fine Scoping for Efficient Vision Mamba
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉Mamba模型计算效率低的问题，提出Coarse-to-Fine Vision Mamba（CF-ViM）框架。通过动态调整输入图像的处理分辨率，简单图像粗处理，复杂区域细处理，实现高效推理。实验表明，该方法在ImageNet上兼顾精度与效率，优于现有技术。**

- **链接: [https://arxiv.org/pdf/2512.00647v1](https://arxiv.org/pdf/2512.00647v1)**

> **作者:** Shanhui Liu; Rui Xu; Yunke Wang
>
> **摘要:** Vision Mamba has emerged as a promising and efficient alternative to Vision Transformers, yet its efficiency remains fundamentally constrained by the number of input tokens. Existing token reduction approaches typically adopt token pruning or merging to reduce computation. However, they inherently lead to information loss, as they discard or compress token representations. This problem is exacerbated when applied uniformly to fine-grained token representations across all images, regardless of visual complexity. We observe that not all inputs require fine-grained processing. Simple images can be effectively handled at coarse resolution, while only complex ones may warrant refinement. Based on this insight, we propose \textit{Coarse-to-Fine Vision Mamba (CF-ViM)}, an adaptive framework for efficient inference. CF-ViM first performs coarse-grained inference by dividing the input image into large patches, significantly reducing the token length and computation. When the model's prediction confidence is low, selected regions are re-processed at a finer resolution to recover critical visual details with minimal additional cost. This dynamic resolution assignment strategy allows CF-ViM to allocate computation adaptively according to image complexity, ensuring efficient processing without compromising essential visual information. Experiments on ImageNet demonstrate that CF-ViM outperforms both the baseline Vision Mamba and state-of-the-art token reduction techniques in terms of accuracy and efficiency.
>
---
#### [new 120] SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SPARK框架，解决从单张RGB图像生成可模拟的关节式3D物体的问题。利用视觉语言模型提取粗略URDF参数并生成部件参考图，结合扩散Transformer生成一致的部件与整体形状，并通过可微正向运动学与渲染优化关节参数，实现物理一致、可直接用于机器人操作等下游任务的高质量资产生成。**

- **链接: [https://arxiv.org/pdf/2512.01629v1](https://arxiv.org/pdf/2512.01629v1)**

> **作者:** Yumeng He; Ying Jiang; Jiayin Lu; Yin Yang; Chenfanfu Jiang
>
> **摘要:** Articulated 3D objects are critical for embodied AI, robotics, and interactive scene understanding, yet creating simulation-ready assets remains labor-intensive and requires expert modeling of part hierarchies and motion structures. We introduce SPARK, a framework for reconstructing physically consistent, kinematic part-level articulated objects from a single RGB image. Given an input image, we first leverage VLMs to extract coarse URDF parameters and generate part-level reference images. We then integrate the part-image guidance and the inferred structure graph into a generative diffusion transformer to synthesize consistent part and complete shapes of articulated objects. To further refine the URDF parameters, we incorporate differentiable forward kinematics and differentiable rendering to optimize joint types, axes, and origins under VLM-generated open-state supervision. Extensive experiments show that SPARK produces high-quality, simulation-ready articulated assets across diverse categories, enabling downstream applications such as robotic manipulation and interaction modeling.
>
---
#### [new 121] Recognizing Pneumonia in Real-World Chest X-rays with a Classifier Trained with Images Synthetically Generated by Nano Banana
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决真实胸片数据稀缺问题。研究利用Google的Nano Banana生成合成胸片训练肺炎识别模型，在两个真实数据集上分别取得0.923和0.824的AUROC，验证了合成数据在医疗AI中的可行性。**

- **链接: [https://arxiv.org/pdf/2512.00428v1](https://arxiv.org/pdf/2512.00428v1)**

> **作者:** Jiachuan Peng; Kyle Lam; Jianing Qiu
>
> **备注:** 9 pages
>
> **摘要:** We trained a classifier with synthetic chest X-ray (CXR) images generated by Nano Banana, the latest AI model for image generation and editing, released by Google. When directly applied to real-world CXRs having only been trained with synthetic data, the classifier achieved an AUROC of 0.923 (95% CI: 0.919 - 0.927), and an AUPR of 0.900 (95% CI: 0.894 - 0.907) in recognizing pneumonia in the 2018 RSNA Pneumonia Detection dataset (14,863 CXRs), and an AUROC of 0.824 (95% CI: 0.810 - 0.836), and an AUPR of 0.913 (95% CI: 0.904 - 0.922) in the Chest X-Ray dataset (5,856 CXRs). These external validation results on real-world data demonstrate the feasibility of this approach and suggest potential for synthetic data in medical AI development. Nonetheless, several limitations remain at present, including challenges in prompt design for controlling the diversity of synthetic CXR data and the requirement for post-processing to ensure alignment with real-world data. However, the growing sophistication and accessibility of medical intelligence will necessitate substantial validation, regulatory approval, and ethical oversight prior to clinical translation.
>
---
#### [new 122] HIMOSA: Efficient Remote Sensing Image Super-Resolution with Hierarchical Mixture of Sparse Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感图像超分辨率任务，解决模型性能与计算效率之间的权衡问题。提出HIMOSA框架，通过内容感知稀疏注意力和分层窗口扩展，利用遥感图像冗余与多尺度模式，在保持轻量化的同时实现高效高精度重建。**

- **链接: [https://arxiv.org/pdf/2512.00275v1](https://arxiv.org/pdf/2512.00275v1)**

> **作者:** Yi Liu; Yi Wan; Xinyi Liu; Qiong Wu; Panwang Xia; Xuejun Huang; Yongjun Zhang
>
> **摘要:** In remote sensing applications, such as disaster detection and response, real-time efficiency and model lightweighting are of critical importance. Consequently, existing remote sensing image super-resolution methods often face a trade-off between model performance and computational efficiency. In this paper, we propose a lightweight super-resolution framework for remote sensing imagery, named HIMOSA. Specifically, HIMOSA leverages the inherent redundancy in remote sensing imagery and introduces a content-aware sparse attention mechanism, enabling the model to achieve fast inference while maintaining strong reconstruction performance. Furthermore, to effectively leverage the multi-scale repetitive patterns found in remote sensing imagery, we introduce a hierarchical window expansion and reduce the computational complexity by adjusting the sparsity of the attention. Extensive experiments on multiple remote sensing datasets demonstrate that our method achieves state-of-the-art performance while maintaining computational efficiency.
>
---
#### [new 123] Exploring Diagnostic Prompting Approach for Multimodal LLM-based Visual Complexity Assessment: A Case Study of Amazon Search Result Pages
- **分类: cs.CV**

- **简介: 该论文研究多模态大模型（MLLM）在亚马逊搜索结果页视觉复杂度评估中的应用，旨在提升其与人类判断的一致性。通过对比诊断提示与传统格式塔原则提示，发现诊断提示显著提升预测性能（F1-score提升858%），但绝对效果仍有限。研究揭示了模型与人类在视觉设计元素与内容相似性关注上的差异，指出当前在产品相似性和色彩感知方面仍存在挑战。**

- **链接: [https://arxiv.org/pdf/2512.00082v1](https://arxiv.org/pdf/2512.00082v1)**

> **作者:** Divendar Murtadak; Yoon Kim; Trilokya Akula
>
> **备注:** 9 pages, 4 figures, 9 tables. Study on diagnostic prompting for multimodal LLM-based visual complexity assessment of Amazon search result pages
>
> **摘要:** This study investigates whether diagnostic prompting can improve Multimodal Large Language Model (MLLM) reliability for visual complexity assessment of Amazon Search Results Pages (SRP). We compare diagnostic prompting with standard gestalt principles-based prompting using 200 Amazon SRP pages and human expert annotations. Diagnostic prompting showed notable improvements in predicting human complexity judgments, with F1-score increasing from 0.031 to 0.297 (+858\% relative improvement), though absolute performance remains modest (Cohen's $κ$ = 0.071). The decision tree revealed that models prioritize visual design elements (badge clutter: 38.6\% importance) while humans emphasize content similarity, suggesting partial alignment in reasoning patterns. Failure case analysis reveals persistent challenges in MLLM visual perception, particularly for product similarity and color intensity assessment. Our findings indicate that diagnostic prompting represents a promising initial step toward human-aligned MLLM-based evaluation, though failure cases with consistent human-MLLM disagreement require continued research and refinement in prompting approaches with larger ground truth datasets for reliable practical deployment.
>
---
#### [new 124] Depth Matching Method Based on ShapeDTW for Oil-Based Mud Imager
- **分类: cs.CV; physics.geo-ph**

- **简介: 该论文针对油基泥浆井下成像中上下垫块图像因深度错位导致的对齐问题，提出基于ShapeDTW的深度匹配方法。通过融合一维HOG与原始信号作为形状描述子，构建形态敏感距离矩阵，实现复杂纹理、偏移和缩放场景下的精确对齐，具备良好可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.01611v1](https://arxiv.org/pdf/2512.01611v1)**

> **作者:** Fengfeng Li; Zhou Feng; Hongliang Wu; Hao Zhang; Han Tian; Peng Liu; Lixin Yuan
>
> **摘要:** In well logging operations using the oil-based mud (OBM) microresistivity imager, which employs an interleaved design with upper and lower pad sets, depth misalignment issues persist between the pad images even after velocity correction. This paper presents a depth matching method for borehole images based on the Shape Dynamic Time Warping (ShapeDTW) algorithm. The method extracts local shape features to construct a morphologically sensitive distance matrix, better preserving structural similarity between sequences during alignment. We implement this by employing a combined feature set of the one-dimensional Histogram of Oriented Gradients (HOG1D) and the original signal as the shape descriptor. Field test examples demonstrate that our method achieves precise alignment for images with complex textures, depth shifts, or local scaling. Furthermore, it provides a flexible framework for feature extension, allowing the integration of other descriptors tailored to specific geological features.
>
---
#### [new 125] Reversible Inversion for Training-Free Exemplar-guided Image Editing
- **分类: cs.CV**

- **简介: 该论文针对训练-free 的示例引导图像编辑（EIE）任务，解决标准反演方法在质量与效率上的不足。提出可逆反演（ReInversion），通过双阶段去噪并结合掩码引导的区域编辑策略，实现高效、高质量的图像编辑，显著降低计算开销。**

- **链接: [https://arxiv.org/pdf/2512.01382v1](https://arxiv.org/pdf/2512.01382v1)**

> **作者:** Yuke Li; Lianli Gao; Ji Zhang; Pengpeng Zeng; Lichuan Xiang; Hongkai Wen; Heng Tao Shen; Jingkuan Song
>
> **摘要:** Exemplar-guided Image Editing (EIE) aims to modify a source image according to a visual reference. Existing approaches often require large-scale pre-training to learn relationships between the source and reference images, incurring high computational costs. As a training-free alternative, inversion techniques can be used to map the source image into a latent space for manipulation. However, our empirical study reveals that standard inversion is sub-optimal for EIE, leading to poor quality and inefficiency. To tackle this challenge, we introduce \textbf{Reversible Inversion ({ReInversion})} for effective and efficient EIE. Specifically, ReInversion operates as a two-stage denoising process, which is first conditioned on the source image and subsequently on the reference. Besides, we introduce a Mask-Guided Selective Denoising (MSD) strategy to constrain edits to target regions, preserving the structural consistency of the background. Both qualitative and quantitative comparisons demonstrate that our ReInversion method achieves state-of-the-art EIE performance with the lowest computational overhead.
>
---
#### [new 126] EZ-SP: Fast and Lightweight Superpoint-Based 3D Segmentation
- **分类: cs.CV**

- **简介: 该论文针对3D语义分割中超点（superpoint）生成效率低的问题，提出EZ-SP：一种轻量级、全GPU可学习的超点划分算法。相比传统方法，其速度提升13倍，模型仅需<2MB显存，支持实时推理，在多个数据集上达到顶尖精度，显著优于点云基SOTA模型。**

- **链接: [https://arxiv.org/pdf/2512.00385v1](https://arxiv.org/pdf/2512.00385v1)**

> **作者:** Louis Geist; Loic Landrieu; Damien Robert
>
> **摘要:** Superpoint-based pipelines provide an efficient alternative to point- or voxel-based 3D semantic segmentation, but are often bottlenecked by their CPU-bound partition step. We propose a learnable, fully GPU partitioning algorithm that generates geometrically and semantically coherent superpoints 13$\times$ faster than prior methods. Our module is compact (under 60k parameters), trains in under 20 minutes with a differentiable surrogate loss, and requires no handcrafted features. Combine with a lightweight superpoint classifier, the full pipeline fits in $<$2 MB of VRAM, scales to multi-million-point scenes, and supports real-time inference. With 72$\times$ faster inference and 120$\times$ fewer parameters, EZ-SP matches the accuracy of point-based SOTA models across three domains: indoor scans (S3DIS), autonomous driving (KITTI-360), and aerial LiDAR (DALES). Code and pretrained models are accessible at github.com/drprojects/superpoint_transformer.
>
---
#### [new 127] Graph-Attention Network with Adversarial Domain Alignment for Robust Cross-Domain Facial Expression Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对跨域面部表情识别（CD-FER）中的域偏移问题，提出GAT-ADA框架。结合ResNet-50与批次级图注意力网络建模样本间关系，利用对抗性域对齐与统计对齐方法减少域差异，实现无监督域适应，在多个目标域上显著提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2512.00641v1](https://arxiv.org/pdf/2512.00641v1)**

> **作者:** Razieh Ghaedi; AmirReza BabaAhmadi; Reyer Zwiggelaar; Xinqi Fan; Nashid Alam
>
> **备注:** 17 pages, 5 figures. Accepted at the 17th Asian Conference on Machine Learning (ACML 2025), Taipei, Taiwan, December 9-12, 2025
>
> **摘要:** Cross-domain facial expression recognition (CD-FER) remains difficult due to severe domain shift between training and deployment data. We propose Graph-Attention Network with Adversarial Domain Alignment (GAT-ADA), a hybrid framework that couples a ResNet-50 as backbone with a batch-level Graph Attention Network (GAT) to model inter-sample relations under shift. Each mini-batch is cast as a sparse ring graph so that attention aggregates cross-sample cues that are informative for adaptation. To align distributions, GAT-ADA combines adversarial learning via a Gradient Reversal Layer (GRL) with statistical alignment using CORAL and MMD. GAT-ADA is evaluated under a standard unsupervised domain adaptation protocol: training on one labeled source (RAF-DB) and adapting to multiple unlabeled targets (CK+, JAFFE, SFEW 2.0, FER2013, and ExpW). GAT-ADA attains 74.39% mean cross-domain accuracy. On RAF-DB to FER2013, it reaches 98.0% accuracy, corresponding to approximately a 36-point improvement over the best baseline we re-implemented with the same backbone and preprocessing.
>
---
#### [new 128] EvalTalker: Learning to Evaluate Real-Portrait-Driven Multi-Subject Talking Humans
- **分类: cs.CV**

- **简介: 该论文针对多主体语音驱动说话人生成（Multi-Talker）质量下降问题，构建首个大规模评估数据集THQA-MT，提出EvalTalker质量评估框架，实现对全局质量、身份一致性和多模态同步性的精准感知，显著提升客观评估与主观感受的相关性。**

- **链接: [https://arxiv.org/pdf/2512.01340v1](https://arxiv.org/pdf/2512.01340v1)**

> **作者:** Yingjie Zhou; Xilei Zhu; Siyu Ren; Ziyi Zhao; Ziwen Wang; Farong Wen; Yu Zhou; Jiezhang Cao; Xiongkuo Min; Fengjiao Chen; Xiaoyu Li; Xuezhi Cao; Guangtao Zhai; Xiaohong Liu
>
> **摘要:** Speech-driven Talking Human (TH) generation, commonly known as "Talker," currently faces limitations in multi-subject driving capabilities. Extending this paradigm to "Multi-Talker," capable of animating multiple subjects simultaneously, introduces richer interactivity and stronger immersion in audiovisual communication. However, current Multi-Talkers still exhibit noticeable quality degradation caused by technical limitations, resulting in suboptimal user experiences. To address this challenge, we construct THQA-MT, the first large-scale Multi-Talker-generated Talking Human Quality Assessment dataset, consisting of 5,492 Multi-Talker-generated THs (MTHs) from 15 representative Multi-Talkers using 400 real portraits collected online. Through subjective experiments, we analyze perceptual discrepancies among different Multi-Talkers and identify 12 common types of distortion. Furthermore, we introduce EvalTalker, a novel TH quality assessment framework. This framework possesses the ability to perceive global quality, human characteristics, and identity consistency, while integrating Qwen-Sync to perceive multimodal synchrony. Experimental results demonstrate that EvalTalker achieves superior correlation with subjective scores, providing a robust foundation for future research on high-quality Multi-Talker generation and evaluation.
>
---
#### [new 129] Learned Image Compression for Earth Observation: Implications for Downstream Segmentation Tasks
- **分类: cs.CV**

- **简介: 该论文研究遥感图像压缩对下游分割任务的影响。针对卫星遥感数据量大、传输存储困难的问题，比较传统压缩（JPEG 2000）与学习型压缩（DMGL）在火灾、云层、建筑检测任务中的表现。结果表明，学习型压缩在多通道光学影像上显著优于传统方法，但对小规模单通道红外数据优势不明显；且联合优化未提升性能。**

- **链接: [https://arxiv.org/pdf/2512.01788v1](https://arxiv.org/pdf/2512.01788v1)**

> **作者:** Christian Mollière; Iker Cumplido; Marco Zeulner; Lukas Liesenhoff; Matthias Schubert; Julia Gottfriedsen
>
> **摘要:** The rapid growth of data from satellite-based Earth observation (EO) systems poses significant challenges in data transmission and storage. We evaluate the potential of task-specific learned compression algorithms in this context to reduce data volumes while retaining crucial information. In detail, we compare traditional compression (JPEG 2000) versus a learned compression approach (Discretized Mixed Gaussian Likelihood) on three EO segmentation tasks: Fire, cloud, and building detection. Learned compression notably outperforms JPEG 2000 for large-scale, multi-channel optical imagery in both reconstruction quality (PSNR) and segmentation accuracy. However, traditional codecs remain competitive on smaller, single-channel thermal infrared datasets due to limited data and architectural constraints. Additionally, joint end-to-end optimization of compression and segmentation models does not improve performance over standalone optimization.
>
---
#### [new 130] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark
- **分类: cs.CV**

- **简介: 该论文针对低光视频实例分割（VIS）难题，提出ELVIS框架。针对缺乏真实低光数据与现有方法不鲁棒的问题，构建了无监督合成低光视频管道，设计VDP-Net与增强解码器，有效分离退化与内容特征，显著提升模型在低光条件下的性能。**

- **链接: [https://arxiv.org/pdf/2512.01495v1](https://arxiv.org/pdf/2512.01495v1)**

> **作者:** Joanne Lin; Ruirui Lin; Yini Li; David Bull; Nantheera Anantrasirichai
>
> **摘要:** Video instance segmentation (VIS) for low-light content remains highly challenging for both humans and machines alike, due to adverse imaging conditions including noise, blur and low-contrast. The lack of large-scale annotated datasets and the limitations of current synthetic pipelines, particularly in modeling temporal degradations, further hinder progress. Moreover, existing VIS methods are not robust to the degradations found in low-light videos and, as a result, perform poorly even when finetuned on low-light data. In this paper, we introduce \textbf{ELVIS} (\textbf{E}nhance \textbf{L}ow-light for \textbf{V}ideo \textbf{I}nstance \textbf{S}egmentation), a novel framework that enables effective domain adaptation of state-of-the-art VIS models to low-light scenarios. ELVIS comprises an unsupervised synthetic low-light video pipeline that models both spatial and temporal degradations, a calibration-free degradation profile synthesis network (VDP-Net) and an enhancement decoder head that disentangles degradations from content features. ELVIS improves performances by up to \textbf{+3.7AP} on the synthetic low-light YouTube-VIS 2019 dataset. Code will be released upon acceptance.
>
---
#### [new 131] Hierarchical Semantic Alignment for Image Clustering
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对图像聚类任务，解决外部语义知识中名词模糊性导致的语义表示失真问题。提出无训练的层次语义对齐方法CAE，融合标题级描述与名词级概念，通过最优传输对齐图像特征与多粒度文本语义，构建更判别性的语义空间，显著提升聚类性能。**

- **链接: [https://arxiv.org/pdf/2512.00904v1](https://arxiv.org/pdf/2512.00904v1)**

> **作者:** Xingyu Zhu; Beier Zhu; Yunfan Li; Junfeng Fang; Shuo Wang; Kesen Zhao; Hanwang Zhang
>
> **备注:** AAAI 2026
>
> **摘要:** Image clustering is a classic problem in computer vision, which categorizes images into different groups. Recent studies utilize nouns as external semantic knowledge to improve clus- tering performance. However, these methods often overlook the inherent ambiguity of nouns, which can distort semantic representations and degrade clustering quality. To address this issue, we propose a hierarChical semAntic alignmEnt method for image clustering, dubbed CAE, which improves cluster- ing performance in a training-free manner. In our approach, we incorporate two complementary types of textual seman- tics: caption-level descriptions, which convey fine-grained attributes of image content, and noun-level concepts, which represent high-level object categories. We first select relevant nouns from WordNet and descriptions from caption datasets to construct a semantic space aligned with image features. Then, we align image features with selected nouns and captions via optimal transport to obtain a more discriminative semantic space. Finally, we combine the enhanced semantic and image features to perform clustering. Extensive experiments across 8 datasets demonstrate the effectiveness of our method, notably surpassing the state-of-the-art training-free approach with a 4.2% improvement in accuracy and a 2.9% improvement in adjusted rand index (ARI) on the ImageNet-1K dataset.
>
---
#### [new 132] RS-ISRefiner: Towards Better Adapting Vision Foundation Models for Interactive Segmentation of Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文针对遥感图像交互式分割任务，解决自然图像方法在遥感场景下泛化性差、标注效率低的问题。提出RS-ISRefiner框架，通过适配器微调保留视觉基础模型通用性，结合混合注意力机制与历史交互优化，提升边界精度与迭代稳定性，在多个遥感数据集上实现更优的分割性能与交互效率。**

- **链接: [https://arxiv.org/pdf/2512.00718v1](https://arxiv.org/pdf/2512.00718v1)**

> **作者:** Deliang Wang; Peng Liu
>
> **摘要:** Interactive image segmentation(IIS) plays a critical role in generating precise annotations for remote sensing imagery, where objects often exhibit scale variations, irregular boundaries and complex backgrounds. However, existing IIS methods, primarily designed for natural images, struggle to generalize to remote sensing domains due to limited annotated data and computational overhead. To address these challenges, we proposed RS-ISRefiner, a novel click-based IIS framework tailored for remote sensing images. The framework employs an adapter-based tuning strategy that preserves the general representations of Vision Foundation Models while enabling efficient learning of remote sensing-specific spatial and boundary characteristics. A hybrid attention mechanism integrating convolutional local modeling with Transformer-based global reasoning enhances robustness against scale diversity and scene complexity. Furthermore, an improved probability map modulation scheme effectively incorporates historical user interactions, yielding more stable iterative refinement and higher boundary fidelity. Comprehensive experiments on six remote sensing datasets, including iSAID, ISPRS Potsdam, SandBar, NWPU, LoveDA Urban and WHUBuilding, demonstrate that RS-ISRefiner consistently outperforms state-of-the-art IIS methods in terms of segmentation accuracy, efficiency and interaction cost. These results confirm the effectiveness and generalizability of our framework, making it highly suitable for high-quality instance segmentation in practical remote sensing scenarios.
>
---
#### [new 133] Joint Multi-scale Gated Transformer and Prior-guided Convolutional Network for Learned Image Compression
- **分类: cs.CV**

- **简介: 该论文针对学习型图像压缩任务，解决传统神经网络在局部与非局部特征提取上的不足。提出PGConv增强局部特征，MGT提升多尺度非局部建模能力，并构建MGTPCN框架，在性能与复杂度间取得更好平衡。**

- **链接: [https://arxiv.org/pdf/2512.00744v1](https://arxiv.org/pdf/2512.00744v1)**

> **作者:** Zhengxin Chen; Xiaohai He; Tingrong Zhang; Shuhua Xiong; Chao Ren
>
> **摘要:** Recently, learned image compression methods have made remarkable achievements, some of which have outperformed the traditional image codec VVC. The advantages of learned image compression methods over traditional image codecs can be largely attributed to their powerful nonlinear transform coding. Convolutional layers and shifted window transformer (Swin-T) blocks are the basic units of neural networks, and their representation capabilities play an important role in nonlinear transform coding. In this paper, to improve the ability of the vanilla convolution to extract local features, we propose a novel prior-guided convolution (PGConv), where asymmetric convolutions (AConvs) and difference convolutions (DConvs) are introduced to strengthen skeleton elements and extract high-frequency information, respectively. A re-parameterization strategy is also used to reduce the computational complexity of PGConv. Moreover, to improve the ability of the Swin-T block to extract non-local features, we propose a novel multi-scale gated transformer (MGT), where dilated window-based multi-head self-attention blocks with different dilation rates and depth-wise convolution layers with different kernel sizes are used to extract multi-scale features, and a gate mechanism is introduced to enhance non-linearity. Finally, we propose a novel joint Multi-scale Gated Transformer and Prior-guided Convolutional Network (MGTPCN) for learned image compression. Experimental results show that our MGTPCN surpasses state-of-the-art algorithms with a better trade-off between performance and complexity.
>
---
#### [new 134] BlinkBud: Detecting Hazards from Behind via Sampled Monocular 3D Detection on a Single Earbud
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 论文提出BlinkBud系统，利用单个耳塞和手机实现对用户后方危险物体的实时3D检测。针对耳塞低功耗与高精度追踪难题，设计基于强化学习的采样策略与卡尔曼滤波轨迹估计，结合头部姿态校正，显著提升追踪精度并降低功耗，有效解决行人与骑行者后方盲区安全隐患。**

- **链接: [https://arxiv.org/pdf/2512.01366v1](https://arxiv.org/pdf/2512.01366v1)**

> **作者:** Yunzhe Li; Jiajun Yan; Yuzhou Wei; Kechen Liu; Yize Zhao; Chong Zhang; Hongzi Zhu; Li Lu; Shan Chang; Minyi Guo
>
> **备注:** This is the author-accepted version of the paper published in Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), Vol. 9, No. 4, Article 191, 2025. Final published version: https://doi.org/10.1145/3770707
>
> **摘要:** Failing to be aware of speeding vehicles approaching from behind poses a huge threat to the road safety of pedestrians and cyclists. In this paper, we propose BlinkBud, which utilizes a single earbud and a paired phone to online detect hazardous objects approaching from behind of a user. The core idea is to accurately track visually identified objects utilizing a small number of sampled camera images taken from the earbud. To minimize the power consumption of the earbud and the phone while guaranteeing the best tracking accuracy, a novel 3D object tracking algorithm is devised, integrating both a Kalman filter based trajectory estimation scheme and an optimal image sampling strategy based on reinforcement learning. Moreover, the impact of constant user head movements on the tracking accuracy is significantly eliminated by leveraging the estimated pitch and yaw angles to correct the object depth estimation and align the camera coordinate system to the user's body coordinate system, respectively. We implement a prototype BlinkBud system and conduct extensive real-world experiments. Results show that BlinkBud is lightweight with ultra-low mean power consumptions of 29.8 mW and 702.6 mW on the earbud and smartphone, respectively, and can accurately detect hazards with a low average false positive ratio (FPR) and false negative ratio (FNR) of 4.90% and 1.47%, respectively.
>
---
#### [new 135] Hybrid Synthetic Data Generation with Domain Randomization Enables Zero-Shot Vision-Based Part Inspection Under Extreme Class Imbalance
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对工业零件视觉检测中缺陷样本稀少导致的严重类别不平衡问题，提出一种融合仿真渲染、领域随机化与真实背景合成的混合生成框架。通过零样本学习实现无需人工标注的高质量检测与分类，显著提升模型在极端不平衡数据下的性能。**

- **链接: [https://arxiv.org/pdf/2512.00125v1](https://arxiv.org/pdf/2512.00125v1)**

> **作者:** Ruo-Syuan Mei; Sixian Jia; Guangze Li; Soo Yeon Lee; Brian Musser; William Keller; Sreten Zakula; Jorge Arinez; Chenhui Shao
>
> **备注:** Submitted to the NAMRC 54
>
> **摘要:** Machine learning, particularly deep learning, is transforming industrial quality inspection. Yet, training robust machine learning models typically requires large volumes of high-quality labeled data, which are expensive, time-consuming, and labor-intensive to obtain in manufacturing. Moreover, defective samples are intrinsically rare, leading to severe class imbalance that degrades model performance. These data constraints hinder the widespread adoption of machine learning-based quality inspection methods in real production environments. Synthetic data generation (SDG) offers a promising solution by enabling the creation of large, balanced, and fully annotated datasets in an efficient, cost-effective, and scalable manner. This paper presents a hybrid SDG framework that integrates simulation-based rendering, domain randomization, and real background compositing to enable zero-shot learning for computer vision-based industrial part inspection without manual annotation. The SDG pipeline generates 12,960 labeled images in one hour by varying part geometry, lighting, and surface properties, and then compositing synthetic parts onto real image backgrounds. A two-stage architecture utilizing a YOLOv8n backbone for object detection and MobileNetV3-small for quality classification is trained exclusively on synthetic data and evaluated on 300 real industrial parts. The proposed approach achieves an mAP@0.5 of 0.995 for detection, 96% classification accuracy, and 90.1% balanced accuracy. Comparative evaluation against few-shot real-data baseline approaches demonstrates significant improvement. The proposed SDG-based approach achieves 90-91% balanced accuracy under severe class imbalance, while the baselines reach only 50% accuracy. These results demonstrate that the proposed method enables annotation-free, scalable, and robust quality inspection for real-world manufacturing applications.
>
---
#### [new 136] SceneProp: Combining Neural Network and Markov Random Field for Scene-Graph Grounding
- **分类: cs.CV**

- **简介: 该论文聚焦场景图接地任务，解决复杂关系查询定位不准的问题。针对现有方法在查询复杂度增加时性能下降的缺陷，提出SceneProp，将问题建模为马尔可夫随机场的MAP推理，通过可微信念传播实现全局优化，显著提升准确率，且精度随查询复杂度增加而提高。**

- **链接: [https://arxiv.org/pdf/2512.00936v1](https://arxiv.org/pdf/2512.00936v1)**

> **作者:** Keita Otani; Tatsuya Harada
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Grounding complex, compositional visual queries with multiple objects and relationships is a fundamental challenge for vision-language models. While standard phrase grounding methods excel at localizing single objects, they lack the structural inductive bias to parse intricate relational descriptions, often failing as queries become more descriptive. To address this structural deficit, we focus on scene-graph grounding, a powerful but less-explored formulation where the query is an explicit graph of objects and their relationships. However, existing methods for this task also struggle, paradoxically showing decreased performance as the query graph grows -- failing to leverage the very information that should make grounding easier. We introduce SceneProp, a novel method that resolves this issue by reformulating scene-graph grounding as a Maximum a Posteriori (MAP) inference problem in a Markov Random Field (MRF). By performing global inference over the entire query graph, SceneProp finds the optimal assignment of image regions to nodes that jointly satisfies all constraints. This is achieved within an end-to-end framework via a differentiable implementation of the Belief Propagation algorithm. Experiments on four benchmarks show that our dedicated focus on the scene-graph grounding formulation allows SceneProp to significantly outperform prior work. Critically, its accuracy consistently improves with the size and complexity of the query graph, demonstrating for the first time that more relational context can, and should, lead to better grounding. Codes are available at https://github.com/keitaotani/SceneProp.
>
---
#### [new 137] MasHeNe: A Benchmark for Head and Neck CT Mass Segmentation using Window-Enhanced Mamba with Frequency-Domain Integration
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对头颈部非恶性占位性病变分割数据匮乏问题，构建了首个包含肿瘤与囊肿的3,779张增强CT像素级标注数据集MasHeNe。提出WEMF模型，融合窗宽增强与多频域注意力，实现高效分割。在该基准上取得优异性能，推动头颈部质量分割研究。**

- **链接: [https://arxiv.org/pdf/2512.01563v1](https://arxiv.org/pdf/2512.01563v1)**

> **作者:** Thao Thi Phuong Dao; Tan-Cong Nguyen; Nguyen Chi Thanh; Truong Hoang Viet; Trong-Le Do; Mai-Khiem Tran; Minh-Khoi Pham; Trung-Nghia Le; Minh-Triet Tran; Thanh Dinh Le
>
> **备注:** The 14th International Symposium on Information and Communication Technology Conference SoICT 2025
>
> **摘要:** Head and neck masses are space-occupying lesions that can compress the airway and esophagus and may affect nerves and blood vessels. Available public datasets primarily focus on malignant lesions and often overlook other space-occupying conditions in this region. To address this gap, we introduce MasHeNe, an initial dataset of 3,779 contrast-enhanced CT slices that includes both tumors and cysts with pixel-level annotations. We also establish a benchmark using standard segmentation baselines and report common metrics to enable fair comparison. In addition, we propose the Windowing-Enhanced Mamba with Frequency integration (WEMF) model. WEMF applies tri-window enhancement to enrich the input appearance before feature extraction. It further uses multi-frequency attention to fuse information across skip connections within a U-shaped Mamba backbone. On MasHeNe, WEMF attains the best performance among evaluated methods, with a Dice of 70.45%, IoU of 66.89%, NSD of 72.33%, and HD95 of 5.12 mm. This model indicates stable and strong results on this challenging task. MasHeNe provides a benchmark for head-and-neck mass segmentation beyond malignancy-only datasets. The observed error patterns also suggest that this task remains challenging and requires further research. Our dataset and code are available at https://github.com/drthaodao3101/MasHeNe.git.
>
---
#### [new 138] ReactionMamba: Generating Short &Long Human Reaction Sequences
- **分类: cs.CV**

- **简介: 该论文提出ReactionMamba，用于生成短时简单与长时复杂3D人类反应动作序列。针对现有方法在长序列生成效率与一致性上的不足，结合运动VAE与Mamba状态空间模型，实现高效、逼真且多样化的动作生成，在多个数据集上表现优异，显著提升推理速度。**

- **链接: [https://arxiv.org/pdf/2512.00208v1](https://arxiv.org/pdf/2512.00208v1)**

> **作者:** Hajra Anwar Beg; Baptiste Chopin; Hao Tang; Mohamed Daoudi
>
> **摘要:** We present ReactionMamba, a novel framework for generating long 3D human reaction motions. Reaction-Mamba integrates a motion VAE for efficient motion encoding with Mamba-based state-space models to decode temporally consistent reactions. This design enables ReactionMamba to generate both short sequences of simple motions and long sequences of complex motions, such as dance and martial arts. We evaluate ReactionMamba on three datasets--NTU120-AS, Lindy Hop, and InterX--and demonstrate competitive performance in terms of realism, diversity, and long-sequence generation compared to previous methods, including InterFormer, ReMoS, and Ready-to-React, while achieving substantial improvements in inference speed.
>
---
#### [new 139] Artemis: Structured Visual Reasoning for Perception Policy Learning
- **分类: cs.CV**

- **简介: 该论文针对视觉感知策略学习中语言推理导致性能下降的问题，提出Artemis框架。通过结构化提案式推理（标签+边界框），将中间步骤锚定于空间与对象中心的表示，提升可验证性与监督精度，显著改善定位、检测、计数及几何感知任务表现，并在多模态基准上展现强泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.01988v1](https://arxiv.org/pdf/2512.01988v1)**

> **作者:** Wei Tang; Yanpeng Sun; Shan Zhang; Xiaofan Li; Piotr Koniusz; Wei Li; Na Zhao; Zechao Li
>
> **摘要:** Recent reinforcement-learning frameworks for visual perception policy have begun to incorporate intermediate reasoning chains expressed in natural language. Empirical observations indicate that such purely linguistic intermediate reasoning often reduces performance on perception tasks. We argue that the core issue lies not in reasoning per se but in the form of reasoning: while these chains perform semantic reasoning in an unstructured linguistic space, visual perception requires reasoning in a spatial and object-centric space. In response, we introduce Artemis, a perception-policy learning framework that performs structured proposal-based reasoning, where each intermediate step is represented as a (label, bounding-box) pair capturing a verifiable visual state. This design enables explicit tracking of intermediate states, direct supervision for proposal quality, and avoids ambiguity introduced by language-based reasoning. Artemis is built on Qwen2.5-VL-3B, achieves strong performance on grounding and detection task and exhibits substantial generalization to counting and geometric-perception tasks. The consistent improvements across these diverse settings confirm that aligning reasoning with spatial representations enhances perception-policy learning. Owing to its strengthened visual reasoning, Artemis also achieves competitive performance on general MLLM benchmarks, illustrating that spatially grounded reasoning provides a principled route toward scalable and general perception policies.
>
---
#### [new 140] ResDiT: Evoking the Intrinsic Resolution Scalability in Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文针对扩散Transformer（DiT）在高分辨率图像生成中出现的布局坍缩和纹理失真问题，提出无需训练的ResDiT方法。通过修正位置编码的外推误差并引入局部增强机制，有效提升生成图像的分辨率与细节质量，实现高效、高质量的高分辨率图像合成。**

- **链接: [https://arxiv.org/pdf/2512.01426v1](https://arxiv.org/pdf/2512.01426v1)**

> **作者:** Yiyang Ma; Feng Zhou; Xuedan Yin; Pu Cao; Yonghao Dang; Jianqin Yin
>
> **备注:** 8 pages
>
> **摘要:** Leveraging pre-trained Diffusion Transformers (DiTs) for high-resolution (HR) image synthesis often leads to spatial layout collapse and degraded texture fidelity. Prior work mitigates these issues with complex pipelines that first perform a base-resolution (i.e., training-resolution) denoising process to guide HR generation. We instead explore the intrinsic generative mechanisms of DiTs and propose ResDiT, a training-free method that scales resolution efficiently. We identify the core factor governing spatial layout, position embeddings (PEs), and show that the original PEs encode incorrect positional information when extrapolated to HR, which triggers layout collapse. To address this, we introduce a PE scaling technique that rectifies positional encoding under resolution changes. To further remedy low-fidelity details, we develop a local-enhancement mechanism grounded in base-resolution local attention. We design a patch-level fusion module that aggregates global and local cues, together with a Gaussian-weighted splicing strategy that eliminates grid artifacts. Comprehensive evaluations demonstrate that ResDiT consistently delivers high-fidelity, high-resolution image synthesis and integrates seamlessly with downstream tasks, including spatially controlled generation.
>
---
#### [new 141] Closing the Approximation Gap of Partial AUC Optimization: A Tale of Two Formulations
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对部分AUC（PAUC）优化中的近似误差与可扩展性问题，提出两种实例级极小极大重构方法，实现渐近无偏且线性复杂度的优化，显著降低计算开销并提升精度，为不平衡数据下的分类评估提供了高效可靠的新方案。**

- **链接: [https://arxiv.org/pdf/2512.01213v1](https://arxiv.org/pdf/2512.01213v1)**

> **作者:** Yangbangyan Jiang; Qianqian Xu; Huiyang Shao; Zhiyong Yang; Shilong Bao; Xiaochun Cao; Qingming Huang
>
> **摘要:** As a variant of the Area Under the ROC Curve (AUC), the partial AUC (PAUC) focuses on a specific range of false positive rate (FPR) and/or true positive rate (TPR) in the ROC curve. It is a pivotal evaluation metric in real-world scenarios with both class imbalance and decision constraints. However, selecting instances within these constrained intervals during its calculation is NP-hard, and thus typically requires approximation techniques for practical resolution. Despite the progress made in PAUC optimization over the last few years, most existing methods still suffer from uncontrollable approximation errors or a limited scalability when optimizing the approximate PAUC objectives. In this paper, we close the approximation gap of PAUC optimization by presenting two simple instance-wise minimax reformulations: one with an asymptotically vanishing gap, the other with the unbiasedness at the cost of more variables. Our key idea is to first establish an equivalent instance-wise problem to lower the time complexity, simplify the complicated sample selection procedure by threshold learning, and then apply different smoothing techniques. Equipped with an efficient solver, the resulting algorithms enjoy a linear per-iteration computational complexity w.r.t. the sample size and a convergence rate of $O(ε^{-1/3})$ for typical one-way and two-way PAUCs. Moreover, we provide a tight generalization bound of our minimax reformulations. The result explicitly demonstrates the impact of the TPR/FPR constraints $α$/$β$ on the generalization and exhibits a sharp order of $\tilde{O}(α^{-1}\n_+^{-1} + β^{-1}\n_-^{-1})$. Finally, extensive experiments on several benchmark datasets validate the strength of our proposed methods.
>
---
#### [new 142] Objects in Generated Videos Are Slower Than They Appear: Models Suffer Sub-Earth Gravity and Don't Know Galileo's Principle...for now
- **分类: cs.CV**

- **简介: 该论文研究视频生成模型对重力的表征问题，旨在评估其是否具备物理世界建模能力。发现模型生成物体下落过慢，且违反伽利略等效原理。通过设计无量纲双物体测试协议，排除尺度干扰，证实模型存在物理偏差。进一步提出轻量级适配器，仅用100个样本微调，显著提升重力感知，实现零样本泛化。**

- **链接: [https://arxiv.org/pdf/2512.02016v1](https://arxiv.org/pdf/2512.02016v1)**

> **作者:** Varun Varma Thozhiyoor; Shivam Tripathi; Venkatesh Babu Radhakrishnan; Anand Bhattad
>
> **备注:** https://gravity-eval.github.io/
>
> **摘要:** Video generators are increasingly evaluated as potential world models, which requires them to encode and understand physical laws. We investigate their representation of a fundamental law: gravity. Out-of-the-box video generators consistently generate objects falling at an effectively slower acceleration. However, these physical tests are often confounded by ambiguous metric scale. We first investigate if observed physical errors are artifacts of these ambiguities (e.g., incorrect frame rate assumptions). We find that even temporal rescaling cannot correct the high-variance gravity artifacts. To rigorously isolate the underlying physical representation from these confounds, we introduce a unit-free, two-object protocol that tests the timing ratio $t_1^2/t_2^2 = h_1/h_2$, a relationship independent of $g$, focal length, and scale. This relative test reveals violations of Galileo's equivalence principle. We then demonstrate that this physical gap can be partially mitigated with targeted specialization. A lightweight low-rank adaptor fine-tuned on only 100 single-ball clips raises $g_{\mathrm{eff}}$ from $1.81\,\mathrm{m/s^2}$ to $6.43\,\mathrm{m/s^2}$ (reaching $65\%$ of terrestrial gravity). This specialist adaptor also generalizes zero-shot to two-ball drops and inclined planes, offering initial evidence that specific physical laws can be corrected with minimal data.
>
---
#### [new 143] Learning Visual Affordance from Audio
- **分类: cs.CV**

- **简介: 该论文提出音频-视觉可操作性定位（AV-AG）任务，旨在通过动作音频定位物体交互区域。针对现有方法依赖文本或视频带来的模糊与遮挡问题，构建首个包含音视频与像素级标注的AV-AG数据集，并提出AVAGFormer模型，实现跨模态融合与端到端掩码预测，显著提升性能，验证了音频在可操作性理解中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02005v1](https://arxiv.org/pdf/2512.02005v1)**

> **作者:** Lidong Lu; Guo Chen; Zhu Wei; Yicheng Liu; Tong Lu
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** We introduce Audio-Visual Affordance Grounding (AV-AG), a new task that segments object interaction regions from action sounds. Unlike existing approaches that rely on textual instructions or demonstration videos, which often limited by ambiguity or occlusion, audio provides real-time, semantically rich, and visually independent cues for affordance grounding, enabling more intuitive understanding of interaction regions. To support this task, we construct the first AV-AG dataset, comprising a large collection of action sounds, object images, and pixel-level affordance annotations. The dataset also includes an unseen subset to evaluate zero-shot generalization. Furthermore, we propose AVAGFormer, a model equipped with a semantic-conditioned cross-modal mixer and a dual-head decoder that effectively fuses audio and visual signals for mask prediction. Experiments show that AVAGFormer achieves state-of-the-art performance on AV-AG, surpassing baselines from related tasks. Comprehensive analyses highlight the distinctions between AV-AG and AVS, the benefits of end-to-end modeling, and the contribution of each component. Code and dataset have been released on https://jscslld.github.io/AVAGFormer/.
>
---
#### [new 144] SpriteHand: Real-Time Versatile Hand-Object Interaction with Autoregressive Video Generation
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出SpriteHand，一个基于自回归视频生成的实时手物交互系统，旨在解决复杂非刚体（如柔性织物、弹性材料）与手部交互建模难题。通过输入静态物体图像和手部动作视频，模型实时生成高保真、物理合理的交互效果，支持多种物体类型与连续动态交互。**

- **链接: [https://arxiv.org/pdf/2512.01960v1](https://arxiv.org/pdf/2512.01960v1)**

> **作者:** Zisu Li; Hengye Lyu; Jiaxin Shi; Yufeng Zeng; Mingming Fan; Hanwang Zhang; Chen Liang
>
> **摘要:** Modeling and synthesizing complex hand-object interactions remains a significant challenge, even for state-of-the-art physics engines. Conventional simulation-based approaches rely on explicitly defined rigid object models and pre-scripted hand gestures, making them inadequate for capturing dynamic interactions with non-rigid or articulated entities such as deformable fabrics, elastic materials, hinge-based structures, furry surfaces, or even living creatures. In this paper, we present SpriteHand, an autoregressive video generation framework for real-time synthesis of versatile hand-object interaction videos across a wide range of object types and motion patterns. SpriteHand takes as input a static object image and a video stream in which the hands are imagined to interact with the virtual object embedded in a real-world scene, and generates corresponding hand-object interaction effects in real time. Our model employs a causal inference architecture for autoregressive generation and leverages a hybrid post-training approach to enhance visual realism and temporal coherence. Our 1.3B model supports real-time streaming generation at around 18 FPS and 640x368 resolution, with an approximate 150 ms latency on a single NVIDIA RTX 5090 GPU, and more than a minute of continuous output. Experiments demonstrate superior visual quality, physical plausibility, and interaction fidelity compared to both generative and engine-based baselines.
>
---
#### [new 145] Assimilation Matters: Model-level Backdoor Detection in Vision-Language Pretrained Models
- **分类: cs.CV**

- **简介: 该论文针对视觉语言预训练模型（VLPs）的后门检测问题，提出AMDET框架。无需先验知识，通过分析文本编码器中触发词的注意力集中现象，利用梯度反演识别隐含特征，区分真实后门与自然相似特征，实现高效高精度检测。**

- **链接: [https://arxiv.org/pdf/2512.00343v1](https://arxiv.org/pdf/2512.00343v1)**

> **作者:** Zhongqi Wang; Jie Zhang; Shiguang Shan; Xilin Chen
>
> **摘要:** Vision-language pretrained models (VLPs) such as CLIP have achieved remarkable success, but are also highly vulnerable to backdoor attacks. Given a model fine-tuned by an untrusted third party, determining whether the model has been injected with a backdoor is a critical and challenging problem. Existing detection methods usually rely on prior knowledge of training dataset, backdoor triggers and targets, or downstream classifiers, which may be impractical for real-world applications. To address this, To address this challenge, we introduce Assimilation Matters in DETection (AMDET), a novel model-level detection framework that operates without any such prior knowledge. Specifically, we first reveal the feature assimilation property in backdoored text encoders: the representations of all tokens within a backdoor sample exhibit a high similarity. Further analysis attributes this effect to the concentration of attention weights on the trigger token. Leveraging this insight, AMDET scans a model by performing gradient-based inversion on token embeddings to recover implicit features that capable of activating backdoor behaviors. Furthermore, we identify the natural backdoor feature in the OpenAI's official CLIP model, which are not intentionally injected but still exhibit backdoor-like behaviors. We then filter them out from real injected backdoor by analyzing their loss landscapes. Extensive experiments on 3,600 backdoored and benign-finetuned models with two attack paradigms and three VLP model structures show that AMDET detects backdoors with an F1 score of 89.90%. Besides, it achieves one complete detection in approximately 5 minutes on a RTX 4090 GPU and exhibits strong robustness against adaptive attacks. Code is available at: https://github.com/Robin-WZQ/AMDET
>
---
#### [new 146] QuantumCanvas: A Multimodal Benchmark for Visual Learning of Atomic Interactions
- **分类: cs.CV; cond-mat.mtrl-sci; quant-ph**

- **简介: 该论文提出QuantumCanvas，一个用于原子相互作用视觉学习的大规模多模态基准。针对现有模型缺乏物理可迁移性的问题，通过18种属性与十通道图像表征原子对量子交互，实现可解释的视觉-数值联合建模。在多个任务中验证了其有效性，提升了下游任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.01519v1](https://arxiv.org/pdf/2512.01519v1)**

> **作者:** Can Polat; Erchin Serpedin; Mustafa Kurban; Hasan Kurban
>
> **摘要:** Despite rapid advances in molecular and materials machine learning, most models still lack physical transferability: they fit correlations across whole molecules or crystals rather than learning the quantum interactions between atomic pairs. Yet bonding, charge redistribution, orbital hybridization, and electronic coupling all emerge from these two-body interactions that define local quantum fields in many-body systems. We introduce QuantumCanvas, a large-scale multimodal benchmark that treats two-body quantum systems as foundational units of matter. The dataset spans 2,850 element-element pairs, each annotated with 18 electronic, thermodynamic, and geometric properties and paired with ten-channel image representations derived from l- and m-resolved orbital densities, angular field transforms, co-occupancy maps, and charge-density projections. These physically grounded images encode spatial, angular, and electrostatic symmetries without explicit coordinates, providing an interpretable visual modality for quantum learning. Benchmarking eight architectures across 18 targets, we report mean absolute errors of 0.201 eV on energy gap using GATv2, 0.265 eV on HOMO and 0.274 eV on LUMO using EGNN. For energy-related quantities, DimeNet attains 2.27 eV total-energy MAE and 0.132 eV repulsive-energy MAE, while a multimodal fusion model achieves a 2.15 eV Mermin free-energy MAE. Pretraining on QuantumCanvas further improves convergence stability and generalization when fine-tuned on larger datasets such as QM9, MD17, and CrysMTM. By unifying orbital physics with vision-based representation learning, QuantumCanvas provides a principled and interpretable basis for learning transferable quantum interactions through coupled visual and numerical modalities. Dataset and model implementations are available at https://github.com/KurbanIntelligenceLab/QuantumCanvas.
>
---
#### [new 147] Toward Content-based Indexing and Retrieval of Head and Neck CT with Abscess Segmentation
- **分类: cs.CV**

- **简介: 该论文针对头颈部脓肿的精准影像诊断问题，提出AbscessHeNe数据集，包含4926张增强CT切片及像素级标注。旨在支持脓肿分割与内容检索，推动深度学习模型在临床决策中的应用。**

- **链接: [https://arxiv.org/pdf/2512.01589v1](https://arxiv.org/pdf/2512.01589v1)**

> **作者:** Thao Thi Phuong Dao; Tan-Cong Nguyen; Trong-Le Do; Truong Hoang Viet; Nguyen Chi Thanh; Huynh Nguyen Thuan; Do Vo Cong Nguyen; Minh-Khoi Pham; Mai-Khiem Tran; Viet-Tham Huynh; Trong-Thuan Nguyen; Trung-Nghia Le; Vo Thanh Toan; Tam V. Nguyen; Minh-Triet Tran; Thanh Dinh Le
>
> **备注:** The 2025 IEEE International Conference on Content-Based Multimedia Indexing (IEEE CBMI)
>
> **摘要:** Abscesses in the head and neck represent an acute infectious process that can potentially lead to sepsis or mortality if not diagnosed and managed promptly. Accurate detection and delineation of these lesions on imaging are essential for diagnosis, treatment planning, and surgical intervention. In this study, we introduce AbscessHeNe, a curated and comprehensively annotated dataset comprising 4,926 contrast-enhanced CT slices with clinically confirmed head and neck abscesses. The dataset is designed to facilitate the development of robust semantic segmentation models that can accurately delineate abscess boundaries and evaluate deep neck space involvement, thereby supporting informed clinical decision-making. To establish performance baselines, we evaluate several state-of-the-art segmentation architectures, including CNN, Transformer, and Mamba-based models. The highest-performing model achieved a Dice Similarity Coefficient of 0.39, Intersection-over-Union of 0.27, and Normalized Surface Distance of 0.67, indicating the challenges of this task and the need for further research. Beyond segmentation, AbscessHeNe is structured for future applications in content-based multimedia indexing and case-based retrieval. Each CT scan is linked with pixel-level annotations and clinical metadata, providing a foundation for building intelligent retrieval systems and supporting knowledge-driven clinical workflows. The dataset will be made publicly available at https://github.com/drthaodao3101/AbscessHeNe.git.
>
---
#### [new 148] SAM3-UNet: Simplified Adaptation of Segment Anything Model 3
- **分类: cs.CV**

- **简介: 该论文提出SAM3-UNet，用于低成本适配Segment Anything Model 3（SAM3）至下游任务。针对SAM3参数量大、训练成本高的问题，设计轻量级图像编码器、高效适配器与U-Net式解码器，实现参数高效微调。实验表明其在镜像检测与显著物检测任务中表现更优，且训练仅需<6GB GPU内存。**

- **链接: [https://arxiv.org/pdf/2512.01789v1](https://arxiv.org/pdf/2512.01789v1)**

> **作者:** Xinyu Xiong; Zihuang Wu; Lei Lu; Yufa Xia
>
> **备注:** Technical Report
>
> **摘要:** In this paper, we introduce SAM3-UNet, a simplified variant of Segment Anything Model 3 (SAM3), designed to adapt SAM3 for downstream tasks at a low cost. Our SAM3-UNet consists of three components: a SAM3 image encoder, a simple adapter for parameter-efficient fine-tuning, and a lightweight U-Net-style decoder. Preliminary experiments on multiple tasks, such as mirror detection and salient object detection, demonstrate that the proposed SAM3-UNet outperforms the prior SAM2-UNet and other state-of-the-art methods, while requiring less than 6 GB of GPU memory during training with a batch size of 12. The code is publicly available at https://github.com/WZH0120/SAM3-UNet.
>
---
#### [new 149] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design
- **分类: cs.CV**

- **简介: 该论文针对工业设计中3D形状生成的物理真实性问题，提出PhysGen框架。通过融合物理约束的流匹配模型与联合编码形状与物理信息的SP-VAE，实现基于物理原理的3D形状生成，提升生成结果在工程性能上的合理性与真实感。**

- **链接: [https://arxiv.org/pdf/2512.00422v1](https://arxiv.org/pdf/2512.00422v1)**

> **作者:** Yingxuan You; Chen Zhao; Hantao Zhang; Mingda Xu; Pascal Fua
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Existing generative models for 3D shapes can synthesize high-fidelity and visually plausible shapes. For certain classes of shapes that have undergone an engineering design process, the realism of the shape is tightly coupled with the underlying physical properties, e.g., aerodynamic efficiency for automobiles. Since existing methods lack knowledge of such physics, they are unable to use this knowledge to enhance the realism of shape generation. Motivated by this, we propose a unified physics-based 3D shape generation pipeline, with a focus on industrial design applications. Specifically, we introduce a new flow matching model with explicit physical guidance, consisting of an alternating update process. We iteratively perform a velocity-based update and a physics-based refinement, progressively adjusting the latent code to align with the desired 3D shapes and physical properties. We further strengthen physical validity by incorporating a physics-aware regularization term into the velocity-based update step. To support such physics-guided updates, we build a shape-and-physics variational autoencoder (SP-VAE) that jointly encodes shape and physics information into a unified latent space. The experiments on three benchmarks show that this synergistic formulation improves shape realism beyond mere visual plausibility.
>
---
#### [new 150] MVAD : A Comprehensive Multimodal Video-Audio Dataset for AIGC Detection
- **分类: cs.CV**

- **简介: 该论文针对AI生成视频音频内容的检测难题，提出首个综合性多模态数据集MVAD。旨在解决现有数据集多局限于视觉或面部伪造、缺乏真实多模态场景的问题。研究构建了涵盖多种风格、内容与伪造模式的高质量多模态数据，支持更可靠的AIGC检测系统开发。**

- **链接: [https://arxiv.org/pdf/2512.00336v1](https://arxiv.org/pdf/2512.00336v1)**

> **作者:** Mengxue Hu; Yunfeng Diao; Changtao Miao; Jianshu Li; Zhe Li; Joey Tianyi Zhou
>
> **备注:** 7 pages,2 figures
>
> **摘要:** The rapid advancement of AI-generated multimodal video-audio content has raised significant concerns regarding information security and content authenticity. Existing synthetic video datasets predominantly focus on the visual modality alone, while the few incorporating audio are largely confined to facial deepfakes--a limitation that fails to address the expanding landscape of general multimodal AI-generated content and substantially impedes the development of trustworthy detection systems. To bridge this critical gap, we introduce the Multimodal Video-Audio Dataset (MVAD), the first comprehensive dataset specifically designed for detecting AI-generated multimodal video-audio content. Our dataset exhibits three key characteristics: (1) genuine multimodality with samples generated according to three realistic video-audio forgery patterns; (2) high perceptual quality achieved through diverse state-of-the-art generative models; and (3) comprehensive diversity spanning realistic and anime visual styles, four content categories (humans, animals, objects, and scenes), and four video-audio multimodal data types. Our dataset will be available at https://github.com/HuMengXue0104/MVAD.
>
---
#### [new 151] Diffusion Model in Latent Space for Medical Image Segmentation Task
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学图像分割任务，解决传统方法无法捕捉不确定性的问题。提出MedSegLatDiff框架，结合VAE与潜空间扩散模型，实现高效多假设分割与置信度图生成，提升分割的可靠性与临床适用性。**

- **链接: [https://arxiv.org/pdf/2512.01292v1](https://arxiv.org/pdf/2512.01292v1)**

> **作者:** Huynh Trinh Ngoc; Toan Nguyen Hai; Ba Luong Son; Long Tran Quoc
>
> **摘要:** Medical image segmentation is crucial for clinical diagnosis and treatment planning. Traditional methods typically produce a single segmentation mask, failing to capture inherent uncertainty. Recent generative models enable the creation of multiple plausible masks per image, mimicking the collaborative interpretation of several clinicians. However, these approaches remain computationally heavy. We propose MedSegLatDiff, a diffusion based framework that combines a variational autoencoder (VAE) with a latent diffusion model for efficient medical image segmentation. The VAE compresses the input into a low dimensional latent space, reducing noise and accelerating training, while the diffusion process operates directly in this compact representation. We further replace the conventional MSE loss with weighted cross entropy in the VAE mask reconstruction path to better preserve tiny structures such as small nodules. MedSegLatDiff is evaluated on ISIC-2018 (skin lesions), CVC-Clinic (polyps), and LIDC-IDRI (lung nodules). It achieves state of the art or highly competitive Dice and IoU scores while simultaneously generating diverse segmentation hypotheses and confidence maps. This provides enhanced interpretability and reliability compared to deterministic baselines, making the model particularly suitable for clinical deployment.
>
---
#### [new 152] Seeing the Wind from a Falling Leaf
- **分类: cs.CV**

- **简介: 该论文旨在从视频中恢复不可见的物理力（如风场），解决视觉与物理建模之间的鸿沟。提出端到端可微分逆图形框架，联合建模物体几何、物理属性与交互，通过反向传播实现力场的隐式推断。在合成与真实场景验证，可用于物理驱动的视频生成与编辑。**

- **链接: [https://arxiv.org/pdf/2512.00762v1](https://arxiv.org/pdf/2512.00762v1)**

> **作者:** Zhiyuan Gao; Jiageng Mao; Hong-Xing Yu; Haozhe Lou; Emily Yue-Ting Jia; Jernej Barbic; Jiajun Wu; Yue Wang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** A longstanding goal in computer vision is to model motions from videos, while the representations behind motions, i.e. the invisible physical interactions that cause objects to deform and move, remain largely unexplored. In this paper, we study how to recover the invisible forces from visual observations, e.g., estimating the wind field by observing a leaf falling to the ground. Our key innovation is an end-to-end differentiable inverse graphics framework, which jointly models object geometry, physical properties, and interactions directly from videos. Through backpropagation, our approach enables the recovery of force representations from object motions. We validate our method on both synthetic and real-world scenarios, and the results demonstrate its ability to infer plausible force fields from videos. Furthermore, we show the potential applications of our approach, including physics-based video generation and editing. We hope our approach sheds light on understanding and modeling the physical process behind pixels, bridging the gap between vision and physics. Please check more video results in our \href{https://chaoren2357.github.io/seeingthewind/}{project page}.
>
---
#### [new 153] DreamingComics: A Story Visualization Pipeline via Subject and Layout Customized Generation using Video Models
- **分类: cs.CV**

- **简介: 该论文针对故事可视化中角色定位不准、风格不一致的问题，提出DreamingComics框架。基于视频扩散变换模型，引入区域感知位置编码（RegionalRoPE）与掩码条件损失，实现布局可控的图像生成，并结合LLM生成漫画式布局，显著提升角色一致性与风格相似性。**

- **链接: [https://arxiv.org/pdf/2512.01686v1](https://arxiv.org/pdf/2512.01686v1)**

> **作者:** Patrick Kwon; Chen Chen
>
> **摘要:** Current story visualization methods tend to position subjects solely by text and face challenges in maintaining artistic consistency. To address these limitations, we introduce DreamingComics, a layout-aware story visualization framework. We build upon a pretrained video diffusion-transformer (DiT) model, leveraging its spatiotemporal priors to enhance identity and style consistency. For layout-based position control, we propose RegionalRoPE, a region-aware positional encoding scheme that re-indexes embeddings based on the target layout. Additionally, we introduce a masked condition loss to further constrain each subject's visual features to their designated region. To infer layouts from natural language scripts, we integrate an LLM-based layout generator trained to produce comic-style layouts, enabling flexible and controllable layout conditioning. We present a comprehensive evaluation of our approach, showing a 29.2% increase in character consistency and a 36.2% increase in style similarity compared to previous methods, while displaying high spatial accuracy. Our project page is available at https://yj7082126.github.io/dreamingcomics/
>
---
#### [new 154] EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出EAG3R框架，解决动态、极端光照下RGB-only方法在3D几何估计中的性能瓶颈。通过融合异步事件流与图像，引入增强模块与事件感知融合机制，并设计事件引导的光度一致性损失，实现无需夜间数据重训练的鲁棒重建，在深度估计、位姿跟踪等任务上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00771v1](https://arxiv.org/pdf/2512.00771v1)**

> **作者:** Xiaoshan Wu; Yifei Yu; Xiaoyang Lyu; Yihua Huang; Bo Wang; Baoheng Zhang; Zhongrui Wang; Xiaojuan Qi
>
> **备注:** Accepted at NeurIPS 2025 (spotlight)
>
> **摘要:** Robust 3D geometry estimation from videos is critical for applications such as autonomous navigation, SLAM, and 3D scene reconstruction. Recent methods like DUSt3R demonstrate that regressing dense pointmaps from image pairs enables accurate and efficient pose-free reconstruction. However, existing RGB-only approaches struggle under real-world conditions involving dynamic objects and extreme illumination, due to the inherent limitations of conventional cameras. In this paper, we propose EAG3R, a novel geometry estimation framework that augments pointmap-based reconstruction with asynchronous event streams. Built upon the MonST3R backbone, EAG3R introduces two key innovations: (1) a retinex-inspired image enhancement module and a lightweight event adapter with SNR-aware fusion mechanism that adaptively combines RGB and event features based on local reliability; and (2) a novel event-based photometric consistency loss that reinforces spatiotemporal coherence during global optimization. Our method enables robust geometry estimation in challenging dynamic low-light scenes without requiring retraining on night-time data. Extensive experiments demonstrate that EAG3R significantly outperforms state-of-the-art RGB-only baselines across monocular depth estimation, camera pose tracking, and dynamic reconstruction tasks.
>
---
#### [new 155] AFRAgent : An Adaptive Feature Renormalization Based High Resolution Aware GUI agent
- **分类: cs.CV**

- **简介: 该论文针对移动GUI自动化任务，解决视觉语言模型因低分辨率特征导致的组件识别与动作决策不准问题。提出AFRAgent，基于instruct-BLIP架构，引入自适应特征重归一化技术，增强图像嵌入表征，提升高分辨率细节融合能力，实现更小模型尺寸与更优性能，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00846v1](https://arxiv.org/pdf/2512.00846v1)**

> **作者:** Neeraj Anand; Rishabh Jain; Sohan Patnaik; Balaji Krishnamurthy; Mausoom Sarkar
>
> **备注:** Accepted at WACV 2026 Conference
>
> **摘要:** There is a growing demand for mobile user interface (UI) automation, driven by its broad applications across industries. With the advent of visual language models (VLMs), GUI automation has progressed from generating text-based instructions for humans to autonomously executing tasks, thus optimizing automation workflows. Recent approaches leverage VLMs for this problem due to their ability to 1) process on-screen content directly, 2) remain independent of device-specific APIs by utilizing human actions (e.g., clicks, typing), and 3) apply real-world contextual knowledge for task understanding. However, these models often have trouble accurately identifying widgets and determining actions due to limited spatial information in vision encoder features. Additionally, top-performing models are often large, requiring extensive training and resulting in inference delays. In this work, we introduce AFRAgent, an instruct-BLIP-based multimodal architecture that achieves superior performance in GUI automation while being less than one-fourth the size of its nearest competitor. To enhance image embeddings in the large language model (LLM) pipeline, we propose an adaptive feature renormalization-based (a token-level affine transformation) technique that effectively enriches low-resolution image embeddings and fuses high-resolution details. We evaluate AFRAgent on Meta-GUI and AITW benchmarks, establishing a new state-of-the-art baseline for smartphone automation.
>
---
#### [new 156] Probabilistic Modeling of Multi-rater Medical Image Segmentation for Diversity and Personalization
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多标注者医学图像分割任务，解决数据不确定性与专家间差异问题。提出ProSeg模型，通过引入两个潜在变量建模专家偏好与边界模糊性，利用变分推断生成兼具多样性和个性化特征的分割结果，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.00748v1](https://arxiv.org/pdf/2512.00748v1)**

> **作者:** Ke Liu; Shangde Gao; Yichao Fu; Shangqi Gao; Chunhua Shen
>
> **摘要:** Medical image segmentation is inherently influenced by data uncertainty, arising from ambiguous boundaries in medical scans and inter-observer variability in diagnosis. To address this challenge, previous works formulated the multi-rater medical image segmentation task, where multiple experts provide separate annotations for each image. However, existing models are typically constrained to either generate diverse segmentation that lacks expert specificity or to produce personalized outputs that merely replicate individual annotators. We propose Probabilistic modeling of multi-rater medical image Segmentation (ProSeg) that simultaneously enables both diversification and personalization. Specifically, we introduce two latent variables to model expert annotation preferences and image boundary ambiguity. Their conditional probabilistic distributions are then obtained through variational inference, allowing segmentation outputs to be generated by sampling from these distributions. Extensive experiments on both the nasopharyngeal carcinoma dataset (NPC) and the lung nodule dataset (LIDC-IDRI) demonstrate that our ProSeg achieves a new state-of-the-art performance, providing segmentation results that are both diverse and expert-personalized. Code can be found in https://github.com/AI4MOL/ProSeg.
>
---
#### [new 157] Deep Learning-Based Computer Vision Models for Early Cancer Detection Using Multimodal Medical Imaging and Radiogenomic Integration Frameworks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析中的早期癌症检测任务，旨在通过深度学习模型从多模态影像（如MRI、CT、PET等）中自动识别微小病变。研究提出融合影像与基因组数据的放射基因组框架，实现无创预测肿瘤分子特征与治疗反应，提升早期诊断精度与个性化诊疗水平。**

- **链接: [https://arxiv.org/pdf/2512.00714v1](https://arxiv.org/pdf/2512.00714v1)**

> **作者:** Emmanuella Avwerosuoghene Oghenekaro
>
> **摘要:** Early cancer detection remains one of the most critical challenges in modern healthcare, where delayed diagnosis significantly reduces survival outcomes. Recent advancements in artificial intelligence, particularly deep learning, have enabled transformative progress in medical imaging analysis. Deep learning-based computer vision models, such as convolutional neural networks (CNNs), transformers, and hybrid attention architectures, can automatically extract complex spatial, morphological, and temporal patterns from multimodal imaging data including MRI, CT, PET, mammography, histopathology, and ultrasound. These models surpass traditional radiological assessment by identifying subtle tissue abnormalities and tumor microenvironment variations invisible to the human eye. At a broader scale, the integration of multimodal imaging with radiogenomics linking quantitative imaging features with genomics, transcriptomics, and epigenetic biomarkers has introduced a new paradigm for personalized oncology. This radiogenomic fusion allows the prediction of tumor genotype, immune response, molecular subtypes, and treatment resistance without invasive biopsies.
>
---
#### [new 158] Parameter Reduction Improves Vision Transformers: A Comparative Study of Sharing and Width Reduction
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文研究视觉Transformer的参数效率问题，针对ViT-B/16在ImageNet-1K上的过参数化现象，提出两种参数缩减策略：分组共享MLP权重和减半隐藏维度。实验表明，两者均在减少32.7%参数的同时提升或保持精度，并显著改善训练稳定性，验证了参数分配与架构约束作为有效归纳偏置的重要性。**

- **链接: [https://arxiv.org/pdf/2512.01059v1](https://arxiv.org/pdf/2512.01059v1)**

> **作者:** Anantha Padmanaban Krishna Kumar
>
> **备注:** 7 pages total (6 pages main text, 1 page references), 1 figures, 2 tables. Code available at https://github.com/AnanthaPadmanaban-KrishnaKumar/parameter-efficient-vit-mlps
>
> **摘要:** Although scaling laws and many empirical results suggest that increasing the size of Vision Transformers often improves performance, model accuracy and training behavior are not always monotonically increasing with scale. Focusing on ViT-B/16 trained on ImageNet-1K, we study two simple parameter-reduction strategies applied to the MLP blocks, each removing 32.7\% of the baseline parameters. Our \emph{GroupedMLP} variant shares MLP weights between adjacent transformer blocks and achieves 81.47\% top-1 accuracy while maintaining the baseline computational cost. Our \emph{ShallowMLP} variant halves the MLP hidden dimension and reaches 81.25\% top-1 accuracy with a 38\% increase in inference throughput. Both models outperform the 86.6M-parameter baseline (81.05\%) and exhibit substantially improved training stability, reducing peak-to-final accuracy degradation from 0.47\% to the range 0.03\% to 0.06\%. These results suggest that, for ViT-B/16 on ImageNet-1K with a standard training recipe, the model operates in an overparameterized regime in which MLP capacity can be reduced without harming performance and can even slightly improve it. More broadly, our findings suggest that architectural constraints such as parameter sharing and reduced width may act as useful inductive biases, and highlight the importance of how parameters are allocated when designing Vision Transformers. All code is available at: https://github.com/AnanthaPadmanaban-KrishnaKumar/parameter-efficient-vit-mlps.
>
---
#### [new 159] Smol-GS: Compact Representations for Abstract 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出Smol-GS，针对3D高斯点云渲染的存储与计算开销问题，通过递归体素层次结构与点级抽象特征编码，实现高效紧凑的3D场景表示。在保持高质量渲染的同时，大幅压缩数据量，支持下游任务如导航与场景理解。**

- **链接: [https://arxiv.org/pdf/2512.00850v1](https://arxiv.org/pdf/2512.00850v1)**

> **作者:** Haishan Wang; Mohammad Hassan Vali; Arno Solin
>
> **摘要:** We present Smol-GS, a novel method for learning compact representations for 3D Gaussian Splatting (3DGS). Our approach learns highly efficient encodings in 3D space that integrate both spatial and semantic information. The model captures the coordinates of the splats through a recursive voxel hierarchy, while splat-wise features store abstracted cues, including color, opacity, transformation, and material properties. This design allows the model to compress 3D scenes by orders of magnitude without loss of flexibility. Smol-GS achieves state-of-the-art compression on standard benchmarks while maintaining high rendering quality. Beyond visual fidelity, the discrete representations could potentially serve as a foundation for downstream tasks such as navigation, planning, and broader 3D scene understanding.
>
---
#### [new 160] PanFlow: Decoupled Motion Control for Panoramic Video Generation
- **分类: cs.CV**

- **简介: 该论文针对全景视频生成中运动控制不精准、复杂大运动难以处理的问题，提出PanFlow方法。通过解耦相机旋转与光流，利用球面特性实现精确运动控制，并引入球面噪声扭曲提升环路一致性。构建了大规模带姿态与光流标注的数据集，验证了在运动迁移与视频编辑中的优越性。**

- **链接: [https://arxiv.org/pdf/2512.00832v1](https://arxiv.org/pdf/2512.00832v1)**

> **作者:** Cheng Zhang; Hanwen Liang; Donny Y. Chen; Qianyi Wu; Konstantinos N. Plataniotis; Camilo Cruz Gambardella; Jianfei Cai
>
> **备注:** Accepted by AAAI. Code: https://github.com/chengzhag/PanFlow
>
> **摘要:** Panoramic video generation has attracted growing attention due to its applications in virtual reality and immersive media. However, existing methods lack explicit motion control and struggle to generate scenes with large and complex motions. We propose PanFlow, a novel approach that exploits the spherical nature of panoramas to decouple the highly dynamic camera rotation from the input optical flow condition, enabling more precise control over large and dynamic motions. We further introduce a spherical noise warping strategy to promote loop consistency in motion across panorama boundaries. To support effective training, we curate a large-scale, motion-rich panoramic video dataset with frame-level pose and flow annotations. We also showcase the effectiveness of our method in various applications, including motion transfer and video editing. Extensive experiments demonstrate that PanFlow significantly outperforms prior methods in motion fidelity, visual quality, and temporal coherence. Our code, dataset, and models are available at https://github.com/chengzhag/PanFlow.
>
---
#### [new 161] IVCR-200K: A Large-Scale Multi-turn Dialogue Benchmark for Interactive Video Corpus Retrieval
- **分类: cs.CV**

- **简介: 该论文提出交互式视频语料检索（IVCR）任务，针对传统单向检索无法满足用户个性化与动态需求的问题。构建了多轮、双语、多模态的IVCR-200K数据集，并提出基于多模态大模型的交互框架，支持自然对话与可解释检索，提升检索的交互性与实用性。**

- **链接: [https://arxiv.org/pdf/2512.01312v1](https://arxiv.org/pdf/2512.01312v1)**

> **作者:** Ning Han; Yawen Zeng; Shaohua Long; Chengqing Li; Sijie Yang; Dun Tan; Jianfeng Dong; Jingjing Chen
>
> **备注:** Accepted by SIGIR2025
>
> **摘要:** In recent years, significant developments have been made in both video retrieval and video moment retrieval tasks, which respectively retrieve complete videos or moments for a given text query. These advancements have greatly improved user satisfaction during the search process. However, previous work has failed to establish meaningful "interaction" between the retrieval system and the user, and its one-way retrieval paradigm can no longer fully meet the personalization and dynamic needs of at least 80.8\% of users. In this paper, we introduce the Interactive Video Corpus Retrieval (IVCR) task, a more realistic setting that enables multi-turn, conversational, and realistic interactions between the user and the retrieval system. To facilitate research on this challenging task, we introduce IVCR-200K, a high-quality, bilingual, multi-turn, conversational, and abstract semantic dataset that supports video retrieval and even moment retrieval. Furthermore, we propose a comprehensive framework based on multi-modal large language models (MLLMs) to help users interact in several modes with more explainable solutions. The extensive experiments demonstrate the effectiveness of our dataset and framework.
>
---
#### [new 162] Realistic Handwritten Multi-Digit Writer (MDW) Number Recognition Challenges
- **分类: cs.CV; cs.LG**

- **简介: 该论文聚焦于真实场景下的多数字手写识别任务，针对传统孤立数字分类在实际应用中表现不佳的问题，利用NIST数据集中的作者信息构建更真实的多数字写作者（MDW）基准数据集。通过引入任务特定评估指标，推动发展能利用书写者先验知识的改进方法，以提升真实场景下的识别性能。**

- **链接: [https://arxiv.org/pdf/2512.00676v1](https://arxiv.org/pdf/2512.00676v1)**

> **作者:** Kiri L. Wagstaff
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Isolated digit classification has served as a motivating problem for decades of machine learning research. In real settings, numbers often occur as multiple digits, all written by the same person. Examples include ZIP Codes, handwritten check amounts, and appointment times. In this work, we leverage knowledge about the writers of NIST digit images to create more realistic benchmark multi-digit writer (MDW) data sets. As expected, we find that classifiers may perform well on isolated digits yet do poorly on multi-digit number recognition. If we want to solve real number recognition problems, additional advances are needed. The MDW benchmarks come with task-specific performance metrics that go beyond typical error calculations to more closely align with real-world impact. They also create opportunities to develop methods that can leverage task-specific knowledge to improve performance well beyond that of individual digit classification methods.
>
---
#### [new 163] Med-VCD: Mitigating Hallucination for Medical Large Vision Language Models through Visual Contrastive Decoding
- **分类: cs.CV**

- **简介: 该论文针对医疗视觉语言模型（LVLMs）易产生看似合理实则错误的幻觉问题，提出Med-VCD方法。通过动态筛选视觉相关词元，实现高效、无延迟的对比解码，在不增加推理时间的前提下，显著提升医疗图像问答与报告生成任务中的事实准确率与抗幻觉能力。**

- **链接: [https://arxiv.org/pdf/2512.01922v1](https://arxiv.org/pdf/2512.01922v1)**

> **作者:** Zahra Mahdavi; Zahra Khodakaramimaghsoud; Hooman Khaloo; Sina Bakhshandeh Taleshani; Erfan Hashemi; Javad Mirzapour Kaleybar; Omid Nejati Manzari
>
> **摘要:** Large vision-language models (LVLMs) are now central to healthcare applications such as medical visual question answering and imaging report generation. Yet, these models remain vulnerable to hallucination outputs that appear plausible but are in fact incorrect. In the natural image domain, several decoding strategies have been proposed to mitigate hallucinations by reinforcing visual evidence, but most rely on secondary decoding or rollback procedures that substantially slow inference. Moreover, existing solutions are often domain-specific and may introduce misalignment between modalities or between generated and ground-truth content. We introduce Med-VCD, a sparse visual-contrastive decoding method that mitigates hallucinations in medical LVLMs without the time overhead of secondary decoding. Med-VCD incorporates a novel token-sparsification strategy that selects visually informed tokens on the fly, trimming redundancy while retaining critical visual context and thus balancing efficiency with reliability. Evaluations on eight medical datasets, spanning ophthalmology, radiology, and pathology tasks in visual question answering, report generation, and dedicated hallucination benchmarks, show that Med-VCD raises factual accuracy by an average of 13\% and improves hallucination accuracy by 6\% relative to baseline medical LVLMs.
>
---
#### [new 164] AutocleanEEG ICVision: Automated ICA Artifact Classification Using Vision-Language AI
- **分类: cs.CV; cs.LG; eess.IV; q-bio.QM**

- **简介: 该论文提出ICVision，一种基于视觉-语言AI的自动脑电图ICA伪迹分类系统。针对传统方法依赖手工特征、缺乏可解释性的问题，利用GPT-4 Vision直接解析EEG可视化数据，实现专家级伪迹分类与自然语言解释，提升分类准确率与临床可操作性。**

- **链接: [https://arxiv.org/pdf/2512.00194v1](https://arxiv.org/pdf/2512.00194v1)**

> **作者:** Zag ElSayed; Grace Westerkamp; Gavin Gammoh; Yanchen Liu; Peyton Siekierski; Craig Erickson; Ernest Pedapati
>
> **备注:** 6 pages, 8 figures
>
> **摘要:** We introduce EEG Autoclean Vision Language AI (ICVision) a first-of-its-kind system that emulates expert-level EEG ICA component classification through AI-agent vision and natural language reasoning. Unlike conventional classifiers such as ICLabel, which rely on handcrafted features, ICVision directly interprets ICA dashboard visualizations topography, time series, power spectra, and ERP plots, using a multimodal large language model (GPT-4 Vision). This allows the AI to see and explain EEG components the way trained neurologists do, making it the first scientific implementation of AI-agent visual cognition in neurophysiology. ICVision classifies each component into one of six canonical categories (brain, eye, heart, muscle, channel noise, and other noise), returning both a confidence score and a human-like explanation. Evaluated on 3,168 ICA components from 124 EEG datasets, ICVision achieved k = 0.677 agreement with expert consensus, surpassing MNE ICLabel, while also preserving clinically relevant brain signals in ambiguous cases. Over 97% of its outputs were rated as interpretable and actionable by expert reviewers. As a core module of the open-source EEG Autoclean platform, ICVision signals a paradigm shift in scientific AI, where models do not just classify, but see, reason, and communicate. It opens the door to globally scalable, explainable, and reproducible EEG workflows, marking the emergence of AI agents capable of expert-level visual decision-making in brain science and beyond.
>
---
#### [new 165] Generative Adversarial Gumbel MCTS for Abstract Visual Composition Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究抽象视觉构图生成任务，旨在解决几何约束下基于文本目标的离散结构组合难题。提出结合蒙特卡洛树搜索与视觉语言模型的生成对抗框架，通过约束引导搜索和对抗性奖励优化，提升生成结果的可行性与语义一致性，在拼图任务中优于扩散与自回归模型。**

- **链接: [https://arxiv.org/pdf/2512.01242v1](https://arxiv.org/pdf/2512.01242v1)**

> **作者:** Zirui Zhao; Boye Niu; David Hsu; Wee Sun Lee
>
> **摘要:** We study abstract visual composition, in which identity is primarily determined by the spatial configuration and relations among a small set of geometric primitives (e.g., parts, symmetry, topology). They are invariant primarily to texture and photorealistic detail. Composing such structures from fixed components under geometric constraints and vague goal specification (such as text) is non-trivial due to combinatorial placement choices, limited data, and discrete feasibility (overlap-free, allowable orientations), which create a sparse solution manifold ill-suited to purely statistical pixel-space generators. We propose a constraint-guided framework that combines explicit geometric reasoning with neural semantics. An AlphaGo-style search enforces feasibility, while a fine-tuned vision-language model scores semantic alignment as reward signals. Our algorithm uses a policy network as a heuristic in Monte-Carlo Tree Search and fine-tunes the network via search-generated plans. Inspired by the Generative Adversarial Network, we use the generated instances for adversarial reward refinement. Over time, the generation should approach the actual data more closely when the reward model cannot distinguish between generated instances and ground-truth. In the Tangram Assembly task, our approach yields higher validity and semantic fidelity than diffusion and auto-regressive baselines, especially as constraints tighten.
>
---
#### [new 166] Automatic Pith Detection in Tree Cross-Section Images Using Deep Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对树木横截面图像中髓心自动检测任务，解决人工检测效率低、易出错的问题。通过对比多种深度学习模型，在自建数据集上评估性能，采用数据增强提升泛化能力，并通过NMS优化Mask R-CNN，验证了模型在真实场景下的适用性。**

- **链接: [https://arxiv.org/pdf/2512.00625v1](https://arxiv.org/pdf/2512.00625v1)**

> **作者:** Tzu-I Liao; Mahmoud Fakhry; Jibin Yesudas Varghese
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Pith detection in tree cross-sections is essential for forestry and wood quality analysis but remains a manual, error-prone task. This study evaluates deep learning models -- YOLOv9, U-Net, Swin Transformer, DeepLabV3, and Mask R-CNN -- to automate the process efficiently. A dataset of 582 labeled images was dynamically augmented to improve generalization. Swin Transformer achieved the highest accuracy (0.94), excelling in fine segmentation. YOLOv9 performed well for bounding box detection but struggled with boundary precision. U-Net was effective for structured patterns, while DeepLabV3 captured multi-scale features with slight boundary imprecision. Mask R-CNN initially underperformed due to overlapping detections, but applying Non-Maximum Suppression (NMS) improved its IoU from 0.45 to 0.80. Generalizability was next tested using an oak dataset of 11 images from Oregon State University's Tree Ring Lab. Additionally, for exploratory analysis purposes, an additional dataset of 64 labeled tree cross-sections was used to train the worst-performing model to see if this would improve its performance generalizing to the unseen oak dataset. Key challenges included tensor mismatches and boundary inconsistencies, addressed through hyperparameter tuning and augmentation. Our results highlight deep learning's potential for tree cross-section pith detection, with model choice depending on dataset characteristics and application needs.
>
---
#### [new 167] DEJIMA: A Novel Large-scale Japanese Dataset for Image Captioning and Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文针对日本视觉语言（V&L）领域缺乏高质量大规模数据的问题，提出DEJIMA管道，构建了388万对图像-文本数据集（DEJIMA-Cap与DEJIMA-VQA）。通过自动化采集、去重、目标检测与LLM精炼，实现高日语自然度与文化相关性，显著提升模型在日文多模态任务上的性能。**

- **链接: [https://arxiv.org/pdf/2512.00773v1](https://arxiv.org/pdf/2512.00773v1)**

> **作者:** Toshiki Katsube; Taiga Fukuhara; Kenichiro Ando; Yusuke Mukuta; Kohei Uehara; Tatsuya Harada
>
> **摘要:** This work addresses the scarcity of high-quality, large-scale resources for Japanese Vision-and-Language (V&L) modeling. We present a scalable and reproducible pipeline that integrates large-scale web collection with rigorous filtering/deduplication, object-detection-driven evidence extraction, and Large Language Model (LLM)-based refinement under grounding constraints. Using this pipeline, we build two resources: an image-caption dataset (DEJIMA-Cap) and a VQA dataset (DEJIMA-VQA), each containing 3.88M image-text pairs, far exceeding the size of existing Japanese V&L datasets. Human evaluations demonstrate that DEJIMA achieves substantially higher Japaneseness and linguistic naturalness than datasets constructed via translation or manual annotation, while maintaining factual correctness at a level comparable to human-annotated corpora. Quantitative analyses of image feature distributions further confirm that DEJIMA broadly covers diverse visual domains characteristic of Japan, complementing its linguistic and cultural representativeness. Models trained on DEJIMA exhibit consistent improvements across multiple Japanese multimodal benchmarks, confirming that culturally grounded, large-scale resources play a key role in enhancing model performance. All data sources and modules in our pipeline are licensed for commercial use, and we publicly release the resulting dataset and metadata to encourage further research and industrial applications in Japanese V&L modeling.
>
---
#### [new 168] MOTION: ML-Assisted On-Device Low-Latency Motion Recognition
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文针对嵌入式设备上低延迟手势识别任务，解决医疗监测中误报与响应慢的问题。利用三轴加速度计数据，结合AutoML提取特征，训练轻量级模型。在WeBe Band设备上实现高效实时识别，神经网络表现最优，适用于安全快速的医疗监控场景。**

- **链接: [https://arxiv.org/pdf/2512.00008v1](https://arxiv.org/pdf/2512.00008v1)**

> **作者:** Veeramani Pugazhenthi; Wei-Hsiang Chu; Junwei Lu; Jadyn N. Miyahira; Soheil Salehi
>
> **摘要:** The use of tiny devices capable of low-latency gesture recognition is gaining momentum in everyday human-computer interaction and especially in medical monitoring fields. Embedded solutions such as fall detection, rehabilitation tracking, and patient supervision require fast and efficient tracking of movements while avoiding unwanted false alarms. This study presents an efficient solution on how to build very efficient motion-based models only using triaxial accelerometer sensors. We explore the capability of the AutoML pipelines to extract the most important features from the data segments. This approach also involves training multiple lightweight machine learning algorithms using the extracted features. We use WeBe Band, a multi-sensor wearable device that is equipped with a powerful enough MCU to effectively perform gesture recognition entirely on the device. Of the models explored, we found that the neural network provided the best balance between accuracy, latency, and memory use. Our results also demonstrate that reliable real-time gesture recognition can be achieved in WeBe Band, with great potential for real-time medical monitoring solutions that require a secure and fast response time.
>
---
#### [new 169] SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation
- **分类: cs.CV**

- **简介: 该论文针对CLIP-based弱监督语义分割中目标区域与背景的过激活问题，提出语义与空间校正（SSR）方法。通过跨模态原型对齐增强语义区分性，利用超像素引导校正抑制非目标区域干扰，有效提升分割精度，在PASCAL VOC和MS COCO上分别达到79.5%和50.6% mIoU。**

- **链接: [https://arxiv.org/pdf/2512.01701v1](https://arxiv.org/pdf/2512.01701v1)**

> **作者:** Xiuli Bi; Die Xiao; Junchao Fan; Bin Xiao
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** In recent years, Contrastive Language-Image Pretraining (CLIP) has been widely applied to Weakly Supervised Semantic Segmentation (WSSS) tasks due to its powerful cross-modal semantic understanding capabilities. This paper proposes a novel Semantic and Spatial Rectification (SSR) method to address the limitations of existing CLIP-based weakly supervised semantic segmentation approaches: over-activation in non-target foreground regions and background areas. Specifically, at the semantic level, the Cross-Modal Prototype Alignment (CMPA) establishes a contrastive learning mechanism to enforce feature space alignment across modalities, reducing inter-class overlap while enhancing semantic correlations, to rectify over-activation in non-target foreground regions effectively; at the spatial level, the Superpixel-Guided Correction (SGC) leverages superpixel-based spatial priors to precisely filter out interference from non-target regions during affinity propagation, significantly rectifying background over-activation. Extensive experiments on the PASCAL VOC and MS COCO datasets demonstrate that our method outperforms all single-stage approaches, as well as more complex multi-stage approaches, achieving mIoU scores of 79.5% and 50.6%, respectively.
>
---
#### [new 170] Real-Time On-the-Go Annotation Framework Using YOLO for Automated Dataset Generation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对农业等领域实时对象检测中数据标注效率低的问题，提出基于YOLO的边缘设备实时标注框架。通过在图像采集时即时标注，显著减少数据准备时间，验证了预训练与单类标注配置在收敛性、性能和鲁棒性上的优势。**

- **链接: [https://arxiv.org/pdf/2512.01165v1](https://arxiv.org/pdf/2512.01165v1)**

> **作者:** Mohamed Abdallah Salem; Ahmed Harb Rabia
>
> **备注:** Copyright 2025 IEEE. This is the author's version of the work that has been accepted for publication in Proceedings of the 5. Interdisciplinary Conference on Electrics and Computer (INTCEC 2025) 15-16 September 2025, Chicago-USA. The final version of record is available at: https://doi.org/10.1109/INTCEC65580.2025.11256048
>
> **摘要:** Efficient and accurate annotation of datasets remains a significant challenge for deploying object detection models such as You Only Look Once (YOLO) in real-world applications, particularly in agriculture where rapid decision-making is critical. Traditional annotation techniques are labor-intensive, requiring extensive manual labeling post data collection. This paper presents a novel real-time annotation approach leveraging YOLO models deployed on edge devices, enabling immediate labeling during image capture. To comprehensively evaluate the efficiency and accuracy of our proposed system, we conducted an extensive comparative analysis using three prominent YOLO architectures (YOLOv5, YOLOv8, YOLOv12) under various configurations: single-class versus multi-class annotation and pretrained versus scratch-based training. Our analysis includes detailed statistical tests and learning dynamics, demonstrating significant advantages of pretrained and single-class configurations in terms of model convergence, performance, and robustness. Results strongly validate the feasibility and effectiveness of our real-time annotation framework, highlighting its capability to drastically reduce dataset preparation time while maintaining high annotation quality.
>
---
#### [new 171] The Outline of Deception: Physical Adversarial Attacks on Traffic Signs Using Edge Patches
- **分类: cs.CV**

- **简介: 该论文针对智能驾驶中交通标志的物理对抗攻击问题，提出一种隐蔽性强的边缘贴纸攻击方法TESP-Attack。通过实例分割生成边缘对齐掩码，结合U-Net生成贴纸，并利用颜色、纹理与频域约束实现视觉隐匿，有效提升攻击成功率与跨模型迁移性，在真实场景下稳定表现。**

- **链接: [https://arxiv.org/pdf/2512.00765v1](https://arxiv.org/pdf/2512.00765v1)**

> **作者:** Haojie Jia; Te Hu; Haowen Li; Long Jin; Chongshi Xin; Yuchi Yao; Jiarui Xiao
>
> **摘要:** Intelligent driving systems are vulnerable to physical adversarial attacks on traffic signs. These attacks can cause misclassification, leading to erroneous driving decisions that compromise road safety. Moreover, within V2X networks, such misinterpretations can propagate, inducing cascading failures that disrupt overall traffic flow and system stability. However, a key limitation of current physical attacks is their lack of stealth. Most methods apply perturbations to central regions of the sign, resulting in visually salient patterns that are easily detectable by human observers, thereby limiting their real-world practicality. This study proposes TESP-Attack, a novel stealth-aware adversarial patch method for traffic sign classification. Based on the observation that human visual attention primarily focuses on the central regions of traffic signs, we employ instance segmentation to generate edge-aligned masks that conform to the shape characteristics of the signs. A U-Net generator is utilized to craft adversarial patches, which are then optimized through color and texture constraints along with frequency domain analysis to achieve seamless integration with the background environment, resulting in highly effective visual concealment. The proposed method demonstrates outstanding attack success rates across traffic sign classification models with varied architectures, achieving over 90% under limited query budgets. It also exhibits strong cross-model transferability and maintains robust real-world performance that remains stable under varying angles and distances.
>
---
#### [new 172] SplatFont3D: Structure-Aware Text-to-3D Artistic Font Generation with Part-Level Style Control
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出SplatFont3D，解决3D艺术字体生成中结构约束强、需细粒度部件风格控制的问题。基于3D高斯泼溅，通过Glyph2Cloud模块生成初始点云，并结合2D扩散模型优化，实现文本到3D艺术字体的高效生成与部件级风格调控。**

- **链接: [https://arxiv.org/pdf/2512.00413v1](https://arxiv.org/pdf/2512.00413v1)**

> **作者:** Ji Gan; Lingxu Chen; Jiaxu Leng; Xinbo Gao
>
> **摘要:** Artistic font generation (AFG) can assist human designers in creating innovative artistic fonts. However, most previous studies primarily focus on 2D artistic fonts in flat design, leaving personalized 3D-AFG largely underexplored. 3D-AFG not only enables applications in immersive 3D environments such as video games and animations, but also may enhance 2D-AFG by rendering 2D fonts of novel views. Moreover, unlike general 3D objects, 3D fonts exhibit precise semantics with strong structural constraints and also demand fine-grained part-level style control. To address these challenges, we propose SplatFont3D, a novel structure-aware text-to-3D AFG framework with 3D Gaussian splatting, which enables the creation of 3D artistic fonts from diverse style text prompts with precise part-level style control. Specifically, we first introduce a Glyph2Cloud module, which progressively enhances both the shapes and styles of 2D glyphs (or components) and produces their corresponding 3D point clouds for Gaussian initialization. The initialized 3D Gaussians are further optimized through interaction with a pretrained 2D diffusion model using score distillation sampling. To enable part-level control, we present a dynamic component assignment strategy that exploits the geometric priors of 3D Gaussians to partition components, while alleviating drift-induced entanglement during 3D Gaussian optimization. Our SplatFont3D provides more explicit and effective part-level style control than NeRF, attaining faster rendering efficiency. Experiments show that our SplatFont3D outperforms existing 3D models for 3D-AFG in style-text consistency, visual quality, and rendering efficiency.
>
---
#### [new 173] CC-FMO: Camera-Conditioned Zero-Shot Single Image to 3D Scene Generation with Foundation Model Orchestration
- **分类: cs.CV**

- **简介: 该论文提出CC-FMO，解决单图生成3D场景中实例一致性与空间协调性不足的问题。通过相机条件化姿态估计与混合生成架构，实现零样本、高保真、相机对齐的3D场景生成，显著提升场景整体一致性与细节质量。**

- **链接: [https://arxiv.org/pdf/2512.00493v1](https://arxiv.org/pdf/2512.00493v1)**

> **作者:** Boshi Tang; Henry Zheng; Rui Huang; Gao Huang
>
> **摘要:** High-quality 3D scene generation from a single image is crucial for AR/VR and embodied AI applications. Early approaches struggle to generalize due to reliance on specialized models trained on curated small datasets. While recent advancements in large-scale 3D foundation models have significantly enhanced instance-level generation, coherent scene generation remains a challenge, where performance is limited by inaccurate per-object pose estimations and spatial inconsistency. To this end, this paper introduces CC-FMO, a zero-shot, camera-conditioned pipeline for single-image to 3D scene generation that jointly conforms to the object layout in input image and preserves instance fidelity. CC-FMO employs a hybrid instance generator that combines semantics-aware vector-set representation with detail-rich structured latent representation, yielding object geometries that are both semantically plausible and high-quality. Furthermore, CC-FMO enables the application of foundational pose estimation models in the scene generation task via a simple yet effective camera-conditioned scale-solving algorithm, to enforce scene-level coherence. Extensive experiments demonstrate that CC-FMO consistently generates high-fidelity camera-aligned compositional scenes, outperforming all state-of-the-art methods.
>
---
#### [new 174] Pore-scale Image Patch Dataset and A Comparative Evaluation of Pore-scale Facial Features
- **分类: cs.CV**

- **简介: 该论文针对面部弱纹理区域特征匹配难题，构建了PorePatch pore-scale图像补丁数据集，提出DMCE框架实现高质量数据生成。通过在该数据集上评估SOTA深度学习描述子，发现其在匹配任务中显著优于传统方法，但在3D重建中优势不明显，揭示深度学习在该领域仍存局限。**

- **链接: [https://arxiv.org/pdf/2512.00381v1](https://arxiv.org/pdf/2512.00381v1)**

> **作者:** Dong Li; HuaLiang Lin; JiaYu Li
>
> **摘要:** The weak-texture nature of facial skin regions presents significant challenges for local descriptor matching in applications such as facial motion analysis and 3D face reconstruction. Although deep learning-based descriptors have demonstrated superior performance to traditional hand-crafted descriptors in many applications, the scarcity of pore-scale image patch datasets has hindered their further development in the facial domain. In this paper, we propose the PorePatch dataset, a high-quality pore-scale image patch dataset, and establish a rational evaluation benchmark. We introduce a Data-Model Co-Evolution (DMCE) framework to generate a progressively refined, high-quality dataset from high-resolution facial images. We then train existing SOTA models on our dataset and conduct extensive experiments. Our results show that the SOTA model achieves a FPR95 value of 1.91% on the matching task, outperforming PSIFT (22.41%) by a margin of 20.5%. However, its advantage is diminished in the 3D reconstruction task, where its overall performance is not significantly better than that of traditional descriptors. This indicates that deep learning descriptors still have limitations in addressing the challenges of facial weak-texture regions, and much work remains to be done in this field.
>
---
#### [new 175] Comparative Analysis of Vision Transformer, Convolutional, and Hybrid Architectures for Mental Health Classification Using Actigraphy-Derived Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究基于可穿戴设备的活动记录图像在精神健康分类中的应用，旨在区分抑郁症、精神分裂症和健康人群。通过将腕部活动数据转为图像，对比VGG16、ViT-B/16与CoAtNet-Tiny三种模型性能，发现混合架构CoAtNet-Tiny表现最稳定准确，尤其对少数类识别效果更优。**

- **链接: [https://arxiv.org/pdf/2512.00103v1](https://arxiv.org/pdf/2512.00103v1)**

> **作者:** Ifeanyi Okala
>
> **摘要:** This work examines how three different image-based methods, VGG16, ViT-B/16, and CoAtNet-Tiny, perform in identifying depression, schizophrenia, and healthy controls using daily actigraphy records. Wrist-worn activity signals from the Psykose and Depresjon datasets were converted into 30 by 48 images and evaluated through a three-fold subject-wise split. Although all methods fitted the training data well, their behaviour on unseen data differed. VGG16 improved steadily but often settled at lower accuracy. ViT-B/16 reached strong results in some runs, but its performance shifted noticeably from fold to fold. CoAtNet-Tiny stood out as the most reliable, recording the highest average accuracy and the most stable curves across folds. It also produced the strongest precision, recall, and F1-scores, particularly for the underrepresented depression and schizophrenia classes. Overall, the findings indicate that CoAtNet-Tiny performed most consistently on the actigraphy images, while VGG16 and ViT-B/16 yielded mixed results. These observations suggest that certain hybrid designs may be especially suited for mental-health work that relies on actigraphy-derived images.
>
---
#### [new 176] MM-ACT: Learn from Multimodal Parallel Generation to Act
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出MM-ACT，一种统一的视觉-语言-动作模型，旨在解决机器人通用策略中语义理解与环境交互的难题。通过多模态并行生成与共享上下文学习，提升任务规划与动作预测能力，在仿真与真实机器人上均取得优异性能。**

- **链接: [https://arxiv.org/pdf/2512.00975v1](https://arxiv.org/pdf/2512.00975v1)**

> **作者:** Haotian Liang; Xinyi Chen; Bin Wang; Mingkang Chen; Yitian Liu; Yuhao Zhang; Zanxin Chen; Tianshuo Yang; Yilun Chen; Jiangmiao Pang; Dong Liu; Xiaokang Yang; Yao Mu; Wenqi Shao; Ping Luo
>
> **备注:** 17 pages
>
> **摘要:** A generalist robotic policy needs both semantic understanding for task planning and the ability to interact with the environment through predictive capabilities. To tackle this, we present MM-ACT, a unified Vision-Language-Action (VLA) model that integrates text, image, and action in shared token space and performs generation across all three modalities. MM-ACT adopts a re-mask parallel decoding strategy for text and image generation, and employs a one-step parallel decoding strategy for action generation to improve efficiency. We introduce Context-Shared Multimodal Learning, a unified training paradigm that supervises generation in all three modalities from a shared context, enhancing action generation through cross-modal learning. Experiments were conducted on the LIBERO simulation and Franka real-robot setups as well as RoboTwin2.0 to assess in-domain and out-of-domain performances respectively. Our approach achieves a success rate of 96.3% on LIBERO, 72.0% across three tasks of real Franka, and 52.38% across eight bimanual tasks of RoboTwin2.0 with an additional gain of 9.25% from cross-modal learning. We release our codes, models and data at https://github.com/HHYHRHY/MM-ACT.
>
---
#### [new 177] Charts Are Not Images: On the Challenges of Scientific Chart Editing
- **分类: cs.CV**

- **简介: 该论文针对科学图表编辑任务，指出现有生成模型因将图表视为像素图像而失效。提出FigEdit基准，涵盖3万+样本与五类复杂编辑任务，揭示传统模型在结构化变换上的不足，并强调需发展理解数据与视觉语义的结构感知模型。**

- **链接: [https://arxiv.org/pdf/2512.00752v1](https://arxiv.org/pdf/2512.00752v1)**

> **作者:** Shawn Li; Ryan Rossi; Sungchul Kim; Sunav Choudhary; Franck Dernoncourt; Puneet Mathur; Zhengzhong Tu; Yue Zhao
>
> **摘要:** Generative models, such as diffusion and autoregressive approaches, have demonstrated impressive capabilities in editing natural images. However, applying these tools to scientific charts rests on a flawed assumption: a chart is not merely an arrangement of pixels but a visual representation of structured data governed by a graphical grammar. Consequently, chart editing is not a pixel-manipulation task but a structured transformation problem. To address this fundamental mismatch, we introduce \textit{FigEdit}, a large-scale benchmark for scientific figure editing comprising over 30,000 samples. Grounded in real-world data, our benchmark is distinguished by its diversity, covering 10 distinct chart types and a rich vocabulary of complex editing instructions. The benchmark is organized into five distinct and progressively challenging tasks: single edits, multi edits, conversational edits, visual-guidance-based edits, and style transfer. Our evaluation of a range of state-of-the-art models on this benchmark reveals their poor performance on scientific figures, as they consistently fail to handle the underlying structured transformations required for valid edits. Furthermore, our analysis indicates that traditional evaluation metrics (e.g., SSIM, PSNR) have limitations in capturing the semantic correctness of chart edits. Our benchmark demonstrates the profound limitations of pixel-level manipulation and provides a robust foundation for developing and evaluating future structure-aware models. By releasing \textit{FigEdit} (https://github.com/adobe-research/figure-editing), we aim to enable systematic progress in structure-aware figure editing, provide a common ground for fair comparison, and encourage future research on models that understand both the visual and semantic layers of scientific charts.
>
---
#### [new 178] Language-Guided Open-World Anomaly Segmentation
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中的开放世界异常分割任务，解决未知物体难以标注与识别的问题。提出Clipomaly方法，基于CLIP实现零样本推理，动态扩展词汇表，无需训练即可分割并命名未知异常，兼具性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.01427v1](https://arxiv.org/pdf/2512.01427v1)**

> **作者:** Klara Reichard; Nikolas Brasch; Nassir Navab; Federico Tombari
>
> **摘要:** Open-world and anomaly segmentation methods seek to enable autonomous driving systems to detect and segment both known and unknown objects in real-world scenes. However, existing methods do not assign semantically meaningful labels to unknown regions, and distinguishing and learning representations for unknown classes remains difficult. While open-vocabulary segmentation methods show promise in generalizing to novel classes, they require a fixed inference vocabulary and thus cannot be directly applied to anomaly segmentation where unknown classes are unconstrained. We propose Clipomaly, the first CLIP-based open-world and anomaly segmentation method for autonomous driving. Our zero-shot approach requires no anomaly-specific training data and leverages CLIP's shared image-text embedding space to both segment unknown objects and assign human-interpretable names to them. Unlike open-vocabulary methods, our model dynamically extends its vocabulary at inference time without retraining, enabling robust detection and naming of anomalies beyond common class definitions such as those in Cityscapes. Clipomaly achieves state-of-the-art performance on established anomaly segmentation benchmarks while providing interpretability and flexibility essential for practical deployment.
>
---
#### [new 179] Describe Anything Anywhere At Any Moment
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DAAAM框架，解决大尺度、实时4D场景理解中语义描述与几何定位的平衡问题。通过优化前端加速局部描述生成，构建分层4D场景图，实现高精度时空记忆。在NaVQA和SG3D基准上显著提升问答与任务接地性能，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00565v1](https://arxiv.org/pdf/2512.00565v1)**

> **作者:** Nicolas Gorlo; Lukas Schmid; Luca Carlone
>
> **备注:** 14 pages, 5 figures, 6 tables
>
> **摘要:** Computer vision and robotics applications ranging from augmented reality to robot autonomy in large-scale environments require spatio-temporal memory frameworks that capture both geometric structure for accurate language-grounding as well as semantic detail. Existing methods face a tradeoff, where producing rich open-vocabulary descriptions comes at the expense of real-time performance when these descriptions have to be grounded in 3D. To address these challenges, we propose Describe Anything, Anywhere, at Any Moment (DAAAM), a novel spatio-temporal memory framework for large-scale and real-time 4D scene understanding. DAAAM introduces a novel optimization-based frontend to infer detailed semantic descriptions from localized captioning models, such as the Describe Anything Model (DAM), leveraging batch processing to speed up inference by an order of magnitude for online processing. It leverages such semantic understanding to build a hierarchical 4D scene graph (SG), which acts as an effective globally spatially and temporally consistent memory representation. DAAAM constructs 4D SGs with detailed, geometrically grounded descriptions while maintaining real-time performance. We show that DAAAM's 4D SG interfaces well with a tool-calling agent for inference and reasoning. We thoroughly evaluate DAAAM in the complex task of spatio-temporal question answering on the NaVQA benchmark and show its generalization capabilities for sequential task grounding on the SG3D benchmark. We further curate an extended OC-NaVQA benchmark for large-scale and long-time evaluations. DAAAM achieves state-of-the-art results in both tasks, improving OC-NaVQA question accuracy by 53.6%, position errors by 21.9%, temporal errors by 21.6%, and SG3D task grounding accuracy by 27.8% over the most competitive baselines, respectively. We release our data and code open-source.
>
---
#### [new 180] CausalAffect: Causal Discovery for Facial Affective Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对面部情感理解中的因果关系建模问题，提出CausalAffect框架，首次实现从数据中自动发现AU间及AU与表情间的因果结构。通过双层极性方向感知因果层次与反事实干预机制，无需标注数据或先验知识，揭示心理可解释的因果关系，提升表情识别与AU检测性能。**

- **链接: [https://arxiv.org/pdf/2512.00456v1](https://arxiv.org/pdf/2512.00456v1)**

> **作者:** Guanyu Hu; Tangzheng Lian; Dimitrios Kollias; Oya Celiktutan; Xinyu Yang
>
> **摘要:** Understanding human affect from facial behavior requires not only accurate recognition but also structured reasoning over the latent dependencies that drive muscle activations and their expressive outcomes. Although Action Units (AUs) have long served as the foundation of affective computing, existing approaches rarely address how to infer psychologically plausible causal relations between AUs and expressions directly from data. We propose CausalAffect, the first framework for causal graph discovery in facial affect analysis. CausalAffect models AU-AU and AU-Expression dependencies through a two-level polarity and direction aware causal hierarchy that integrates population-level regularities with sample-adaptive structures. A feature-level counterfactual intervention mechanism further enforces true causal effects while suppressing spurious correlations. Crucially, our approach requires neither jointly annotated datasets nor handcrafted causal priors, yet it recovers causal structures consistent with established psychological theories while revealing novel inhibitory and previously uncharacterized dependencies. Extensive experiments across six benchmarks demonstrate that CausalAffect advances the state of the art in both AU detection and expression recognition, establishing a principled connection between causal discovery and interpretable facial behavior. All trained models and source code will be released upon acceptance.
>
---
#### [new 181] nnMobileNet++: Towards Efficient Hybrid Networks for Retinal Image Analysis
- **分类: cs.CV**

- **简介: 该论文针对视网膜图像分析任务，解决纯卷积网络难以捕捉长距离依赖和不规则病灶问题。提出nnMobileNet++，融合动态蛇形卷积、分阶段变压器模块与预训练策略，提升特征提取与全局建模能力，实现高精度且低计算成本的视网膜图像分析。**

- **链接: [https://arxiv.org/pdf/2512.01273v1](https://arxiv.org/pdf/2512.01273v1)**

> **作者:** Xin Li; Wenhui Zhu; Xuanzhao Dong; Hao Wang; Yujian Xiong; Oana Dumitrascu; Yalin Wang
>
> **摘要:** Retinal imaging is a critical, non-invasive modality for the early detection and monitoring of ocular and systemic diseases. Deep learning, particularly convolutional neural networks (CNNs), has significant progress in automated retinal analysis, supporting tasks such as fundus image classification, lesion detection, and vessel segmentation. As a representative lightweight network, nnMobileNet has demonstrated strong performance across multiple retinal benchmarks while remaining computationally efficient. However, purely convolutional architectures inherently struggle to capture long-range dependencies and model the irregular lesions and elongated vascular patterns that characterize on retinal images, despite the critical importance of vascular features for reliable clinical diagnosis. To further advance this line of work and extend the original vision of nnMobileNet, we propose nnMobileNet++, a hybrid architecture that progressively bridges convolutional and transformer representations. The framework integrates three key components: (i) dynamic snake convolution for boundary-aware feature extraction, (ii) stage-specific transformer blocks introduced after the second down-sampling stage for global context modeling, and (iii) retinal image pretraining to improve generalization. Experiments on multiple public retinal datasets for classification, together with ablation studies, demonstrate that nnMobileNet++ achieves state-of-the-art or highly competitive accuracy while maintaining low computational cost, underscoring its potential as a lightweight yet effective framework for retinal image analysis.
>
---
#### [new 182] WiseEdit: Benchmarking Cognition- and Creativity-Informed Image Editing
- **分类: cs.CV**

- **简介: 该论文提出WiseEdit，一个面向认知与创意驱动图像编辑的综合性基准。针对现有评估体系覆盖不足的问题，构建包含感知、理解、想象三阶段及三类知识的1220个测试用例，系统评测模型在复杂任务中的认知推理与创造性生成能力，揭示当前顶尖模型的局限性。**

- **链接: [https://arxiv.org/pdf/2512.00387v1](https://arxiv.org/pdf/2512.00387v1)**

> **作者:** Kaihang Pan; Weile Chen; Haiyi Qiu; Qifan Yu; Wendong Bu; Zehan Wang; Yun Zhu; Juncheng Li; Siliang Tang
>
> **备注:** 32 pages, 20 figures. Project Page: https://qnancy.github.io/wiseedit_project_page/
>
> **摘要:** Recent image editing models boast next-level intelligent capabilities, facilitating cognition- and creativity-informed image editing. Yet, existing benchmarks provide too narrow a scope for evaluation, failing to holistically assess these advanced abilities. To address this, we introduce WiseEdit, a knowledge-intensive benchmark for comprehensive evaluation of cognition- and creativity-informed image editing, featuring deep task depth and broad knowledge breadth. Drawing an analogy to human cognitive creation, WiseEdit decomposes image editing into three cascaded steps, i.e., Awareness, Interpretation, and Imagination, each corresponding to a task that poses a challenge for models to complete at the specific step. It also encompasses complex tasks, where none of the three steps can be finished easily. Furthermore, WiseEdit incorporates three fundamental types of knowledge: Declarative, Procedural, and Metacognitive knowledge. Ultimately, WiseEdit comprises 1,220 test cases, objectively revealing the limitations of SoTA image editing models in knowledge-based cognitive reasoning and creative composition capabilities. The benchmark, evaluation code, and the generated images of each model will be made publicly available soon. Project Page: https://qnancy.github.io/wiseedit_project_page/.
>
---
#### [new 183] Binary-Gaussian: Compact and Progressive Representation for 3D Gaussian Segmentation
- **分类: cs.CV**

- **简介: 该论文针对3D高斯点云分割中类别特征冗余、细粒度分割难、前景背景混淆等问题，提出二值化编码压缩特征表示，设计渐进式训练策略与透明度微调机制，显著降低内存占用并提升分割精度。**

- **链接: [https://arxiv.org/pdf/2512.00944v1](https://arxiv.org/pdf/2512.00944v1)**

> **作者:** An Yang; Chenyu Liu; Jun Du; Jianqing Gao; Jia Pan; Jinshui Hu; Baocai Yin; Bing Yin; Cong Liu
>
> **摘要:** 3D Gaussian Splatting (3D-GS) has emerged as an efficient 3D representation and a promising foundation for semantic tasks like segmentation. However, existing 3D-GS-based segmentation methods typically rely on high-dimensional category features, which introduce substantial memory overhead. Moreover, fine-grained segmentation remains challenging due to label space congestion and the lack of stable multi-granularity control mechanisms. To address these limitations, we propose a coarse-to-fine binary encoding scheme for per-Gaussian category representation, which compresses each feature into a single integer via the binary-to-decimal mapping, drastically reducing memory usage. We further design a progressive training strategy that decomposes panoptic segmentation into a series of independent sub-tasks, reducing inter-class conflicts and thereby enhancing fine-grained segmentation capability. Additionally, we fine-tune opacity during segmentation training to address the incompatibility between photometric rendering and semantic segmentation, which often leads to foreground-background confusion. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art segmentation performance while significantly reducing memory consumption and accelerating inference.
>
---
#### [new 184] FR-TTS: Test-Time Scaling for NTP-based Image Generation with Effective Filling-based Reward Signal
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对NTP图像生成中的测试时缩放（TTS）难题，提出基于填充的奖励机制（FR）。通过估计中间序列的未来轨迹，提升中间样本奖励与最终图像奖励的相关性，从而更准确指导样本筛选。进一步设计FR-TTS，结合动态多样性奖励，实现高质量图像生成。**

- **链接: [https://arxiv.org/pdf/2512.00438v1](https://arxiv.org/pdf/2512.00438v1)**

> **作者:** Hang Xu; Linjiang Huang; Feng Zhao
>
> **摘要:** Test-time scaling (TTS) has become a prevalent technique in image generation, significantly boosting output quality by expanding the number of parallel samples and filtering them using pre-trained reward models. However, applying this powerful methodology to the next-token prediction (NTP) paradigm remains challenging. The primary obstacle is the low correlation between the reward of an image decoded from an intermediate token sequence and the reward of the fully generated image. Consequently, these incomplete intermediate representations prove to be poor indicators for guiding the pruning direction, a limitation that stems from their inherent incompleteness in scale or semantic content. To effectively address this critical issue, we introduce the Filling-Based Reward (FR). This novel design estimates the approximate future trajectory of an intermediate sample by finding and applying a reasonable filling scheme to complete the sequence. Both the correlation coefficient between rewards of intermediate samples and final samples, as well as multiple intrinsic signals like token confidence, indicate that the FR provides an excellent and reliable metric for accurately evaluating the quality of intermediate samples. Building upon this foundation, we propose FR-TTS, a sophisticated scaling strategy. FR-TTS efficiently searches for good filling schemes and incorporates a diversity reward with a dynamic weighting schedule to achieve a balanced and comprehensive evaluation of intermediate samples. We experimentally validate the superiority of FR-TTS over multiple established benchmarks and various reward models. Code is available at \href{https://github.com/xuhang07/FR-TTS}{https://github.com/xuhang07/FR-TTS}.
>
---
#### [new 185] FreqEdit: Preserving High-Frequency Features for Robust Multi-Turn Image Editing
- **分类: cs.CV**

- **简介: 该论文针对多轮自然语言图像编辑中细节丢失的问题，提出无需训练的FreqEdit框架。通过高频特征注入、自适应控制与路径补偿机制，有效保留图像细节，提升编辑稳定性与准确性，显著改善多轮编辑质量。**

- **链接: [https://arxiv.org/pdf/2512.01755v1](https://arxiv.org/pdf/2512.01755v1)**

> **作者:** Yucheng Liao; Jiajun Liang; Kaiqian Cui; Baoquan Zhao; Haoran Xie; Wei Liu; Qing Li; Xudong Mao
>
> **摘要:** Instruction-based image editing through natural language has emerged as a powerful paradigm for intuitive visual manipulation. While recent models achieve impressive results on single edits, they suffer from severe quality degradation under multi-turn editing. Through systematic analysis, we identify progressive loss of high-frequency information as the primary cause of this quality degradation. We present FreqEdit, a training-free framework that enables stable editing across 10+ consecutive iterations. Our approach comprises three synergistic components: (1) high-frequency feature injection from reference velocity fields to preserve fine-grained details, (2) an adaptive injection strategy that spatially modulates injection strength for precise region-specific control, and (3) a path compensation mechanism that periodically recalibrates the editing trajectory to prevent over-constraint. Extensive experiments demonstrate that FreqEdit achieves superior performance in both identity preservation and instruction following compared to seven state-of-the-art baselines.
>
---
#### [new 186] Robust Rigid and Non-Rigid Medical Image Registration Using Learnable Edge Kernels
- **分类: cs.CV**

- **简介: 该论文研究医学图像配准任务，针对多模态图像间对比度差异、形变等挑战，提出基于可学习边缘核的刚性与非刚性配准方法。通过预设边缘核并随机扰动，学习优化边缘特征提取，提升结构匹配精度。在多个数据集上验证，性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.01771v1](https://arxiv.org/pdf/2512.01771v1)**

> **作者:** Ahsan Raza Siyal; Markus Haltmeier; Ruth Steiger; Malik Galijasevic; Elke Ruth Gizewski; Astrid Ellen Grams
>
> **摘要:** Medical image registration is crucial for various clinical and research applications including disease diagnosis or treatment planning which require alignment of images from different modalities, time points, or subjects. Traditional registration techniques often struggle with challenges such as contrast differences, spatial distortions, and modality-specific variations. To address these limitations, we propose a method that integrates learnable edge kernels with learning-based rigid and non-rigid registration techniques. Unlike conventional layers that learn all features without specific bias, our approach begins with a predefined edge detection kernel, which is then perturbed with random noise. These kernels are learned during training to extract optimal edge features tailored to the task. This adaptive edge detection enhances the registration process by capturing diverse structural features critical in medical imaging. To provide clearer insight into the contribution of each component in our design, we introduce four variant models for rigid registration and four variant models for non-rigid registration. We evaluated our approach using a dataset provided by the Medical University across three setups: rigid registration without skull removal, with skull removal, and non-rigid registration. Additionally, we assessed performance on two publicly available datasets. Across all experiments, our method consistently outperformed state-of-the-art techniques, demonstrating its potential to improve multi-modal image alignment and anatomical structure analysis.
>
---
#### [new 187] KM-ViPE: Online Tightly Coupled Vision-Language-Geometry Fusion for Open-Vocabulary Semantic SLAM
- **分类: cs.CV**

- **简介: 该论文提出KM-ViPE，一种面向动态环境的实时开集语义SLAM系统，解决单目相机在未校准条件下实现鲁棒定位与语义建图的问题。通过紧耦合视觉-语言-几何特征，融合深度特征与语言嵌入，实现在线、无需深度传感器的开集语义建图，适用于自主机器人与AR/VR场景。**

- **链接: [https://arxiv.org/pdf/2512.01889v1](https://arxiv.org/pdf/2512.01889v1)**

> **作者:** Zaid Nasser; Mikhail Iumanov; Tianhao Li; Maxim Popov; Jaafar Mahmoud; Malik Mohrat; Ilya Obrubov; Ekaterina Derevyanka; Ivan Sosin; Sergey Kolyubin
>
> **摘要:** We present KM-ViPE (Knowledge Mapping Video Pose Engine), a real-time open-vocabulary SLAM framework for uncalibrated monocular cameras in dynamic environments. Unlike systems requiring depth sensors and offline calibration, KM-ViPE operates directly on raw RGB streams, making it ideal for ego-centric applications and harvesting internet-scale video data for training. KM-ViPE tightly couples DINO visual features with geometric constraints through a high-level features based adaptive robust kernel that handles both moving objects and movable static objects (e.g., moving furniture in ego-centric views). The system performs simultaneous online localization and open-vocabulary semantic mapping by fusing geometric and deep visual features aligned with language embeddings. Our results are competitive with state-of-the-art approaches, while existing solutions either operate offline, need depth data and/or odometry estimation, or lack dynamic scene robustness. KM-ViPE benefits from internet-scale training and uniquely combines online operation, uncalibrated monocular input, and robust handling of dynamic scenes, which makes it a good fit for autonomous robotics and AR/VR applications and advances practical spatial intelligence capabilities for embodied AI.
>
---
#### [new 188] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文针对生成式数据蒸馏任务，解决现有方法忽视实例级特征与类内差异的问题。提出基于最优传输（OT）的几何对齐框架，通过OT引导的扩散采样、标签-图像对齐软重标注和OT-basedlogit匹配，实现全局与实例级分布精准对齐，显著提升蒸馏性能。**

- **链接: [https://arxiv.org/pdf/2512.00308v1](https://arxiv.org/pdf/2512.00308v1)**

> **作者:** Xiao Cui; Yulei Qin; Wengang Zhou; Hongsheng Li; Houqiang Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Dataset distillation seeks to synthesize a compact distilled dataset, enabling models trained on it to achieve performance comparable to models trained on the full dataset. Recent methods for large-scale datasets focus on matching global distributional statistics (e.g., mean and variance), but overlook critical instance-level characteristics and intraclass variations, leading to suboptimal generalization. We address this limitation by reformulating dataset distillation as an Optimal Transport (OT) distance minimization problem, enabling fine-grained alignment at both global and instance levels throughout the pipeline. OT offers a geometrically faithful framework for distribution matching. It effectively preserves local modes, intra-class patterns, and fine-grained variations that characterize the geometry of complex, high-dimensional distributions. Our method comprises three components tailored for preserving distributional geometry: (1) OT-guided diffusion sampling, which aligns latent distributions of real and distilled images; (2) label-image-aligned soft relabeling, which adapts label distributions based on the complexity of distilled image distributions; and (3) OT-based logit matching, which aligns the output of student models with soft-label distributions. Extensive experiments across diverse architectures and large-scale datasets demonstrate that our method consistently outperforms state-of-the-art approaches in an efficient manner, achieving at least 4% accuracy improvement under IPC=10 settings for each architecture on ImageNet-1K.
>
---
#### [new 189] Diffusion Fuzzy System: Fuzzy Rule Guided Latent Multi-Path Diffusion Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对扩散模型在处理异质图像特征时的性能瓶颈，提出一种基于模糊规则引导的多路径扩散系统（DFS）。通过分路径学习不同特征类型、规则链动态协调路径及模糊隶属度压缩潜空间，有效提升生成质量与训练效率，解决了多路径协作低效与计算成本高的问题。**

- **链接: [https://arxiv.org/pdf/2512.01533v1](https://arxiv.org/pdf/2512.01533v1)**

> **作者:** Hailong Yang; Te Zhang; Kup-sze Choi; Zhaohong Deng
>
> **摘要:** Diffusion models have emerged as a leading technique for generating images due to their ability to create high-resolution and realistic images. Despite their strong performance, diffusion models still struggle in managing image collections with significant feature differences. They often fail to capture complex features and produce conflicting results. Research has attempted to address this issue by learning different regions of an image through multiple diffusion paths and then combining them. However, this approach leads to inefficient coordination among multiple paths and high computational costs. To tackle these issues, this paper presents a Diffusion Fuzzy System (DFS), a latent-space multi-path diffusion model guided by fuzzy rules. DFS offers several advantages. First, unlike traditional multi-path diffusion methods, DFS uses multiple diffusion paths, each dedicated to learning a specific class of image features. By assigning each path to a different feature type, DFS overcomes the limitations of multi-path models in capturing heterogeneous image features. Second, DFS employs rule-chain-based reasoning to dynamically steer the diffusion process and enable efficient coordination among multiple paths. Finally, DFS introduces a fuzzy membership-based latent-space compression mechanism to reduce the computational costs of multi-path diffusion effectively. We tested our method on three public datasets: LSUN Bedroom, LSUN Church, and MS COCO. The results show that DFS achieves more stable training and faster convergence than existing single-path and multi-path diffusion models. Additionally, DFS surpasses baseline models in both image quality and alignment between text and images, and also shows improved accuracy when comparing generated images to target references.
>
---
#### [new 190] Integrating Skeleton Based Representations for Robust Yoga Pose Classification Using Deep Learning Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对瑜伽姿势识别任务，解决因姿势错误导致受伤的问题。提出新数据集Yoga-16，系统比较三种深度模型在原始图像与两种骨架表示下的表现，发现骨架输入显著提升准确率（最高96.09%），并用Grad-CAM提供模型可解释性分析。**

- **链接: [https://arxiv.org/pdf/2512.00572v1](https://arxiv.org/pdf/2512.00572v1)**

> **作者:** Mohammed Mohiuddin; Syed Mohammod Minhaz Hossain; Sumaiya Khanam; Prionkar Barua; Aparup Barua; MD Tamim Hossain
>
> **摘要:** Yoga is a popular form of exercise worldwide due to its spiritual and physical health benefits, but incorrect postures can lead to injuries. Automated yoga pose classification has therefore gained importance to reduce reliance on expert practitioners. While human pose keypoint extraction models have shown high potential in action recognition, systematic benchmarking for yoga pose recognition remains limited, as prior works often focus solely on raw images or a single pose extraction model. In this study, we introduce a curated dataset, 'Yoga-16', which addresses limitations of existing datasets, and systematically evaluate three deep learning architectures (VGG16, ResNet50, and Xception) using three input modalities (direct images, MediaPipe Pose skeleton images, and YOLOv8 Pose skeleton images). Our experiments demonstrate that skeleton-based representations outperform raw image inputs, with the highest accuracy of 96.09% achieved by VGG16 with MediaPipe Pose skeleton input. Additionally, we provide interpretability analysis using Grad-CAM, offering insights into model decision-making for yoga pose classification with cross validation analysis.
>
---
#### [new 191] SGDiff: Scene Graph Guided Diffusion Model for Image Collaborative SegCaptioning
- **分类: cs.CV**

- **简介: 该论文提出“图像协同分割与描述”新任务，旨在用简单提示（如框选物体）生成多样化的（描述, 分割掩码）对。针对提示信息少、需同时生成语义一致的多组结果的问题，提出SGDiff模型，通过场景图引导的扩散机制与对比学习，实现高效、精准的跨模态对齐，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2512.01975v1](https://arxiv.org/pdf/2512.01975v1)**

> **作者:** Xu Zhang; Jin Yuan; Hanwang Zhang; Guojin Zhong; Yongsheng Zang; Jiacheng Lin; Zhiyong Li
>
> **备注:** Accept by AAAI-2025
>
> **摘要:** Controllable image semantic understanding tasks, such as captioning or segmentation, necessitate users to input a prompt (e.g., text or bounding boxes) to predict a unique outcome, presenting challenges such as high-cost prompt input or limited information output. This paper introduces a new task ``Image Collaborative Segmentation and Captioning'' (SegCaptioning), which aims to translate a straightforward prompt, like a bounding box around an object, into diverse semantic interpretations represented by (caption, masks) pairs, allowing flexible result selection by users. This task poses significant challenges, including accurately capturing a user's intention from a minimal prompt while simultaneously predicting multiple semantically aligned caption words and masks. Technically, we propose a novel Scene Graph Guided Diffusion Model that leverages structured scene graph features for correlated mask-caption prediction. Initially, we introduce a Prompt-Centric Scene Graph Adaptor to map a user's prompt to a scene graph, effectively capturing his intention. Subsequently, we employ a diffusion process incorporating a Scene Graph Guided Bimodal Transformer to predict correlated caption-mask pairs by uncovering intricate correlations between them. To ensure accurate alignment, we design a Multi-Entities Contrastive Learning loss to explicitly align visual and textual entities by considering inter-modal similarity, resulting in well-aligned caption-mask pairs. Extensive experiments conducted on two datasets demonstrate that SGDiff achieves superior performance in SegCaptioning, yielding promising results for both captioning and segmentation tasks with minimal prompt input.
>
---
#### [new 192] RecruitView: A Multimodal Dataset for Predicting Personality and Interview Performance for Human Resources Applications
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RecruitView数据集与CRMF模型，解决人力资源中基于多模态行为数据预测人格与面试表现的难题。通过几何深度学习，融合双曲、球面与欧氏流形建模，实现高效精准评估，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00450v1](https://arxiv.org/pdf/2512.00450v1)**

> **作者:** Amit Kumar Gupta; Farhan Sheth; Hammad Shaikh; Dheeraj Kumar; Angkul Puniya; Deepak Panwar; Sandeep Chaurasia; Priya Mathur
>
> **备注:** 20 pages, 10 figures, 10 tables
>
> **摘要:** Automated personality and soft skill assessment from multimodal behavioral data remains challenging due to limited datasets and methods that fail to capture geometric structure inherent in human traits. We introduce RecruitView, a dataset of 2,011 naturalistic video interview clips from 300+ participants with 27,000 pairwise comparative judgments across 12 dimensions: Big Five personality traits, overall personality score, and six interview performance metrics. To leverage this data, we propose Cross-Modal Regression with Manifold Fusion (CRMF), a geometric deep learning framework that explicitly models behavioral representations across hyperbolic, spherical, and Euclidean manifolds. CRMF employs geometry-specific expert networks to capture hierarchical trait structures, directional behavioral patterns, and continuous performance variations simultaneously. An adaptive routing mechanism dynamically weights expert contributions based on input characteristics. Through principled tangent space fusion, CRMF achieves superior performance while training 40-50% fewer trainable parameters than large multimodal models. Extensive experiments demonstrate that CRMF substantially outperforms the selected baselines, achieving up to 11.4% improvement in Spearman correlation and 6.0% in concordance index. Our RecruitView dataset is publicly available at https://huggingface.co/datasets/AI4A-lab/RecruitView
>
---
#### [new 193] Recovering Origin Destination Flows from Bus CCTV: Early Results from Nairobi and Kigali
- **分类: cs.CV**

- **简介: 该论文针对撒哈拉以南非洲公交系统缺乏可靠客流数据的问题，利用车载CCTV，结合目标检测、跟踪、重识别与时间戳技术，构建了从视频中恢复乘客起讫点（OD）流量的初步框架。在低密度光照条件下效果良好，但在拥挤、光线变化等真实场景下性能显著下降，揭示了现有方法的局限性。**

- **链接: [https://arxiv.org/pdf/2512.00424v1](https://arxiv.org/pdf/2512.00424v1)**

> **作者:** Nthenya Kyatha; Jay Taneja
>
> **摘要:** Public transport in sub-Saharan Africa (SSA) often operates in overcrowded conditions where existing automated systems fail to capture reliable passenger flow data. Leveraging onboard CCTV already deployed for security, we present a baseline pipeline that combines YOLOv12 detection, BotSORT tracking, OSNet embeddings, OCR-based timestamping, and telematics-based stop classification to recover bus origin--destination (OD) flows. On annotated CCTV segments from Nairobi and Kigali buses, the system attains high counting accuracy under low-density, well-lit conditions (recall $\approx$95\%, precision $\approx$91\%, F1 $\approx$93\%). It produces OD matrices that closely match manual tallies. Under realistic stressors such as overcrowding, color-to-monochrome shifts, posture variation, and non-standard door use, performance degrades sharply (e.g., $\sim$40\% undercount in peak-hour boarding and a $\sim$17 percentage-point drop in recall for monochrome segments), revealing deployment-specific failure modes and motivating more robust, deployment-focused Re-ID methods for SSA transit.
>
---
#### [new 194] GRASP: Guided Residual Adapters with Sample-wise Partitioning
- **分类: cs.CV**

- **简介: 该论文针对文本生成图像模型在长尾数据（如医学影像）中罕见病灶生成质量差的问题，提出GRASP方法。通过样本级聚类减少梯度冲突，引入残差适配器实现高效微调，显著提升稀有类别生成质量与多样性，适用于医疗数据增强与跨领域泛化。**

- **链接: [https://arxiv.org/pdf/2512.01675v1](https://arxiv.org/pdf/2512.01675v1)**

> **作者:** Felix Nützel; Mischa Dombrowski; Bernhard Kainz
>
> **备注:** 10 pages, 4 figures, 6 tables
>
> **摘要:** Recent advances in text-to-image diffusion models enable high-fidelity generation across diverse prompts. However, these models falter in long-tail settings, such as medical imaging, where rare pathologies comprise a small fraction of data. This results in mode collapse: tail-class outputs lack quality and diversity, undermining the goal of synthetic data augmentation for underrepresented conditions. We pinpoint gradient conflicts between frequent head and rare tail classes as the primary culprit, a factor unaddressed by existing sampling or conditioning methods that mainly steer inference without altering the learned distribution. To resolve this, we propose GRASP: Guided Residual Adapters with Sample-wise Partitioning. GRASP uses external priors to statically partition samples into clusters that minimize intra-group gradient clashes. It then fine-tunes pre-trained models by injecting cluster-specific residual adapters into transformer feedforward layers, bypassing learned gating for stability and efficiency. On the long-tail MIMIC-CXR-LT dataset, GRASP yields superior FID and diversity metrics, especially for rare classes, outperforming baselines like vanilla fine-tuning and Mixture of Experts variants. Downstream classification on NIH-CXR-LT improves considerably for tail labels. Generalization to ImageNet-LT confirms broad applicability. Our method is lightweight, scalable, and readily integrates with diffusion pipelines.
>
---
#### [new 195] ART-ASyn: Anatomy-aware Realistic Texture-based Anomaly Synthesis Framework for Chest X-Rays
- **分类: cs.CV**

- **简介: 该论文针对胸部X光片异常检测任务，解决现有合成异常方法生成不真实、忽略解剖结构的问题。提出ART-ASyn框架，基于纹理增强与渐进二值阈值分割（PBTSeg）生成解剖一致的肺部阴影异常，提供精确掩码实现显式监督，并在零样本场景下验证了其泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.00310v1](https://arxiv.org/pdf/2512.00310v1)**

> **作者:** Qinyi Cao; Jianan Fan; Weidong Cai
>
> **备注:** Accepted in WACV2026
>
> **摘要:** Unsupervised anomaly detection aims to identify anomalies without pixel-level annotations. Synthetic anomaly-based methods exhibit a unique capacity to introduce controllable irregularities with known masks, enabling explicit supervision during training. However, existing methods often produce synthetic anomalies that are visually distinct from real pathological patterns and ignore anatomical structure. This paper presents a novel Anatomy-aware Realistic Texture-based Anomaly Synthesis framework (ART-ASyn) for chest X-rays that generates realistic and anatomically consistent lung opacity related anomalies using texture-based augmentation guided by our proposed Progressive Binary Thresholding Segmentation method (PBTSeg) for lung segmentation. The generated paired samples of synthetic anomalies and their corresponding precise pixel-level anomaly mask for each normal sample enable explicit segmentation supervision. In contrast to prior work limited to one-class classification, ART-ASyn is further evaluated for zero-shot anomaly segmentation, demonstrating generalizability on an unseen dataset without target-domain annotations. Code availability is available at https://github.com/angelacao-hub/ART-ASyn.
>
---
#### [new 196] RoleMotion: A Large-Scale Dataset towards Robust Scene-Specific Role-Playing Motion Synthesis with Fine-grained Descriptions
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RoleMotion，一个面向场景化角色扮演的大型人体动作数据集，解决现有数据集功能缺失、标注粗略、质量不一的问题。通过构建25个场景、110个角色、500+行为及高质量动作序列与细粒度文本描述，支持文本驱动的全身动作生成任务，验证了其在动作合成中的有效性与实用性。**

- **链接: [https://arxiv.org/pdf/2512.01582v1](https://arxiv.org/pdf/2512.01582v1)**

> **作者:** Junran Peng; Yiheng Huang; Silei Shen; Zeji Wei; Jingwei Yang; Baojie Wang; Yonghao He; Chuanchen Luo; Man Zhang; Xucheng Yin; Wei Sui
>
> **摘要:** In this paper, we introduce RoleMotion, a large-scale human motion dataset that encompasses a wealth of role-playing and functional motion data tailored to fit various specific scenes. Existing text datasets are mainly constructed decentrally as amalgamation of assorted subsets that their data are nonfunctional and isolated to work together to cover social activities in various scenes. Also, the quality of motion data is inconsistent, and textual annotation lacks fine-grained details in these datasets. In contrast, RoleMotion is meticulously designed and collected with a particular focus on scenes and roles. The dataset features 25 classic scenes, 110 functional roles, over 500 behaviors, and 10296 high-quality human motion sequences of body and hands, annotated with 27831 fine-grained text descriptions. We build an evaluator stronger than existing counterparts, prove its reliability, and evaluate various text-to-motion methods on our dataset. Finally, we explore the interplay of motion generation of body and hands. Experimental results demonstrate the high-quality and functionality of our dataset on text-driven whole-body generation.
>
---
#### [new 197] A Fast and Efficient Modern BERT based Text-Conditioned Diffusion Model for Medical Image Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出FastTextDiff，一种基于ModernBERT的快速高效文本条件扩散模型，用于医学图像分割。针对标注成本高、依赖密集标签的问题，利用临床文本注释增强图像语义表征，通过跨模态注意力提升分割性能，实现更高效、准确的弱监督分割。**

- **链接: [https://arxiv.org/pdf/2512.00084v1](https://arxiv.org/pdf/2512.00084v1)**

> **作者:** Venkata Siddharth Dhara; Pawan Kumar
>
> **备注:** 15 pages, 3 figures, Accepted in Slide 3 10th International Conference on Computer Vision & Image Processing (CVIP 2026)
>
> **摘要:** In recent times, denoising diffusion probabilistic models (DPMs) have proven effective for medical image generation and denoising, and as representation learners for downstream segmentation. However, segmentation performance is limited by the need for dense pixel-wise labels, which are expensive, time-consuming, and require expert knowledge. We propose FastTextDiff, a label-efficient diffusion-based segmentation model that integrates medical text annotations to enhance semantic representations. Our approach uses ModernBERT, a transformer capable of processing long clinical notes, to tightly link textual annotations with semantic content in medical images. Trained on MIMIC-III and MIMIC-IV, ModernBERT encodes clinical knowledge that guides cross-modal attention between visual and textual features. This study validates ModernBERT as a fast, scalable alternative to Clinical BioBERT in diffusion-based segmentation pipelines and highlights the promise of multi-modal techniques for medical image analysis. By replacing Clinical BioBERT with ModernBERT, FastTextDiff benefits from FlashAttention 2, an alternating attention mechanism, and a 2-trillion-token corpus, improving both segmentation accuracy and training efficiency over traditional diffusion-based models.
>
---
#### [new 198] Learning Eigenstructures of Unstructured Data Manifolds
- **分类: cs.CV**

- **简介: 该论文提出一种无需显式构造算子的谱基学习框架，直接从无结构数据中学习流形的谱表示。针对传统方法依赖人工选型、离散化与求解器的问题，通过神经网络最小化重构误差，联合学习谱基、度量密度及特征值，实现对任意维度数据的统一建模，为高维几何处理提供数据驱动的新范式。**

- **链接: [https://arxiv.org/pdf/2512.01103v1](https://arxiv.org/pdf/2512.01103v1)**

> **作者:** Roy Velich; Arkadi Piven; David Bensaïd; Daniel Cremers; Thomas Dagès; Ron Kimmel
>
> **摘要:** We introduce a novel framework that directly learns a spectral basis for shape and manifold analysis from unstructured data, eliminating the need for traditional operator selection, discretization, and eigensolvers. Grounded in optimal-approximation theory, we train a network to decompose an implicit approximation operator by minimizing the reconstruction error in the learned basis over a chosen distribution of probe functions. For suitable distributions, they can be seen as an approximation of the Laplacian operator and its eigendecomposition, which are fundamental in geometry processing. Furthermore, our method recovers in a unified manner not only the spectral basis, but also the implicit metric's sampling density and the eigenvalues of the underlying operator. Notably, our unsupervised method makes no assumption on the data manifold, such as meshing or manifold dimensionality, allowing it to scale to arbitrary datasets of any dimension. On point clouds lying on surfaces in 3D and high-dimensional image manifolds, our approach yields meaningful spectral bases, that can resemble those of the Laplacian, without explicit construction of an operator. By replacing the traditional operator selection, construction, and eigendecomposition with a learning-based approach, our framework offers a principled, data-driven alternative to conventional pipelines. This opens new possibilities in geometry processing for unstructured data, particularly in high-dimensional spaces.
>
---
#### [new 199] Supervised Contrastive Machine Unlearning of Background Bias in Sonar Image Classification with Fine-Grained Explainable AI
- **分类: cs.CV**

- **简介: 该论文针对声呐图像分类中模型过度依赖海底背景特征导致泛化能力差的问题，提出融合对比未学习与可解释AI的框架。通过目标对比未学习模块减少背景偏差，并利用改进的LIME生成精准归因，提升模型鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.01291v1](https://arxiv.org/pdf/2512.01291v1)**

> **作者:** Kamal Basha S; Athira Nambiar
>
> **备注:** Accepted to CVIP 2025
>
> **摘要:** Acoustic sonar image analysis plays a critical role in object detection and classification, with applications in both civilian and defense domains. Despite the availability of real and synthetic datasets, existing AI models that achieve high accuracy often over-rely on seafloor features, leading to poor generalization. To mitigate this issue, we propose a novel framework that integrates two key modules: (i) a Targeted Contrastive Unlearning (TCU) module, which extends the traditional triplet loss to reduce seafloor-induced background bias and improve generalization, and (ii) the Unlearn to Explain Sonar Framework (UESF), which provides visual insights into what the model has deliberately forgotten while adapting the LIME explainer to generate more faithful and localized attributions for unlearning evaluation. Extensive experiments across both real and synthetic sonar datasets validate our approach, demonstrating significant improvements in unlearning effectiveness, model robustness, and interpretability.
>
---
#### [new 200] Provenance-Driven Reliable Semantic Medical Image Vector Reconstruction via Lightweight Blockchain-Verified Latent Fingerprints
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医疗图像因噪声、损坏或篡改导致AI诊断不可靠的问题，提出一种基于轻量区块链的语义感知重建方法。通过融合高层语义嵌入与混合U-Net架构，提升重建结构准确性；利用无标度图设计的区块链记录溯源信息，保障过程可验证性。实验表明，该方法在结构一致性和可信度上优于现有技术。**

- **链接: [https://arxiv.org/pdf/2512.00999v1](https://arxiv.org/pdf/2512.00999v1)**

> **作者:** Mohsin Rasheed; Abdullah Al-Mamun
>
> **摘要:** Medical imaging is essential for clinical diagnosis, yet real-world data frequently suffers from corruption, noise, and potential tampering, challenging the reliability of AI-assisted interpretation. Conventional reconstruction techniques prioritize pixel-level recovery and may produce visually plausible outputs while compromising anatomical fidelity, an issue that can directly impact clinical outcomes. We propose a semantic-aware medical image reconstruction framework that integrates high-level latent embeddings with a hybrid U-Net architecture to preserve clinically relevant structures during restoration. To ensure trust and accountability, we incorporate a lightweight blockchain-based provenance layer using scale-free graph design, enabling verifiable recording of each reconstruction event without imposing significant overhead. Extensive evaluation across multiple datasets and corruption types demonstrates improved structural consistency, restoration accuracy, and provenance integrity compared with existing approaches. By uniting semantic-guided reconstruction with secure traceability, our solution advances dependable AI for medical imaging, enhancing both diagnostic confidence and regulatory compliance in healthcare environments.
>
---
#### [new 201] SRAM: Shape-Realism Alignment Metric for No Reference 3D Shape Evaluation
- **分类: cs.CV**

- **简介: 该论文针对无参考3D形状真实感评估任务，提出SRAM度量方法。通过将网格形状编码至语言空间，利用大语言模型实现真实感评估，设计专门解码器对齐人类感知。构建了无需真值的RealismGrading数据集，验证了方法在跨物体场景下的高相关性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.01373v1](https://arxiv.org/pdf/2512.01373v1)**

> **作者:** Sheng Liu; Tianyu Luan; Phani Nuney; Xuelu Feng; Junsong Yuan
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** 3D generation and reconstruction techniques have been widely used in computer games, film, and other content creation areas. As the application grows, there is a growing demand for 3D shapes that look truly realistic. Traditional evaluation methods rely on a ground truth to measure mesh fidelity. However, in many practical cases, a shape's realism does not depend on having a ground truth reference. In this work, we propose a Shape-Realism Alignment Metric that leverages a large language model (LLM) as a bridge between mesh shape information and realism evaluation. To achieve this, we adopt a mesh encoding approach that converts 3D shapes into the language token space. A dedicated realism decoder is designed to align the language model's output with human perception of realism. Additionally, we introduce a new dataset, RealismGrading, which provides human-annotated realism scores without the need for ground truth shapes. Our dataset includes shapes generated by 16 different algorithms on over a dozen objects, making it more representative of practical 3D shape distributions. We validate our metric's performance and generalizability through k-fold cross-validation across different objects. Experimental results show that our metric correlates well with human perceptions and outperforms existing methods, and has good generalizability.
>
---
#### [new 202] S2AM3D: Scale-controllable Part Segmentation of 3D Point Cloud
- **分类: cs.CV**

- **简介: 该论文针对3D点云部件分割中模型泛化差与视图不一致问题，提出S2AM3D方法。通过3D一致性监督融合2D分割先验，设计点一致编码器与尺度感知提示解码器，实现细粒度可控分割。构建超大规模高质量数据集，显著提升模型性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00995v1](https://arxiv.org/pdf/2512.00995v1)**

> **作者:** Han Su; Tianyu Huang; Zichen Wan; Xiaohe Wu; Wangmeng Zuo
>
> **摘要:** Part-level point cloud segmentation has recently attracted significant attention in 3D computer vision. Nevertheless, existing research is constrained by two major challenges: native 3D models lack generalization due to data scarcity, while introducing 2D pre-trained knowledge often leads to inconsistent segmentation results across different views. To address these challenges, we propose S2AM3D, which incorporates 2D segmentation priors with 3D consistent supervision. We design a point-consistent part encoder that aggregates multi-view 2D features through native 3D contrastive learning, producing globally consistent point features. A scale-aware prompt decoder is then proposed to enable real-time adjustment of segmentation granularity via continuous scale signals. Simultaneously, we introduce a large-scale, high-quality part-level point cloud dataset with more than 100k samples, providing ample supervision signals for model training. Extensive experiments demonstrate that S2AM3D achieves leading performance across multiple evaluation settings, exhibiting exceptional robustness and controllability when handling complex structures and parts with significant size variations.
>
---
#### [new 203] Relightable Holoported Characters: Capturing and Relighting Dynamic Human Performance from Sparse Views
- **分类: cs.CV**

- **简介: 该论文提出Relightable Holoported Characters（RHC），解决从稀疏视角视频中实现人物动态全身自由视图渲染与光照重演的问题。通过基于Transformer的RelightNet，单次前向传播即可预测新光照下的外观，利用物理启发特征和3D高斯点云实现高效渲染，显著提升视觉质量和光照还原度。**

- **链接: [https://arxiv.org/pdf/2512.00255v1](https://arxiv.org/pdf/2512.00255v1)**

> **作者:** Kunwar Maheep Singh; Jianchun Chen; Vladislav Golyanik; Stephan J. Garbin; Thabo Beeler; Rishabh Dabral; Marc Habermann; Christian Theobalt
>
> **摘要:** We present Relightable Holoported Characters (RHC), a novel person-specific method for free-view rendering and relighting of full-body and highly dynamic humans solely observed from sparse-view RGB videos at inference. In contrast to classical one-light-at-a-time (OLAT)-based human relighting, our transformer-based RelightNet predicts relit appearance within a single network pass, avoiding costly OLAT-basis capture and generation. For training such a model, we introduce a new capture strategy and dataset recorded in a multi-view lightstage, where we alternate frames lit by random environment maps with uniformly lit tracking frames, simultaneously enabling accurate motion tracking and diverse illumination as well as dynamics coverage. Inspired by the rendering equation, we derive physics-informed features that encode geometry, albedo, shading, and the virtual camera view from a coarse human mesh proxy and the input views. Our RelightNet then takes these features as input and cross-attends them with a novel lighting condition, and regresses the relit appearance in the form of texel-aligned 3D Gaussian splats attached to the coarse mesh proxy. Consequently, our RelightNet implicitly learns to efficiently compute the rendering equation for novel lighting conditions within a single feed-forward pass. Experiments demonstrate our method's superior visual fidelity and lighting reproduction compared to state-of-the-art approaches. Project page: https://vcai.mpi-inf.mpg.de/projects/RHC/
>
---
#### [new 204] SemImage: Semantic Image Representation for Text, a Novel Framework for Embedding Disentangled Linguistic Features
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出SemImage，将文本转为二维语义图像，用HSV向量编码主题、情感和强度，通过多任务学习实现特征解耦。利用动态边界行突出段落转换，结合CNN进行文档分类，提升准确率与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.00088v1](https://arxiv.org/pdf/2512.00088v1)**

> **作者:** Mohammad Zare
>
> **摘要:** We propose SemImage, a novel method for representing a text document as a two-dimensional semantic image to be processed by convolutional neural networks (CNNs). In a SemImage, each word is represented as a pixel in a 2D image: rows correspond to sentences and an additional boundary row is inserted between sentences to mark semantic transitions. Each pixel is not a typical RGB value but a vector in a disentangled HSV color space, encoding different linguistic features: the Hue with two components H_cos and H_sin to account for circularity encodes the topic, Saturation encodes the sentiment, and Value encodes intensity or certainty. We enforce this disentanglement via a multi-task learning framework: a ColorMapper network maps each word embedding to the HSV space, and auxiliary supervision is applied to the Hue and Saturation channels to predict topic and sentiment labels, alongside the main task objective. The insertion of dynamically computed boundary rows between sentences yields sharp visual boundaries in the image when consecutive sentences are semantically dissimilar, effectively making paragraph breaks salient. We integrate SemImage with standard 2D CNNs (e.g., ResNet) for document classification. Experiments on multi-label datasets (with both topic and sentiment annotations) and single-label benchmarks demonstrate that SemImage can achieve competitive or better accuracy than strong text classification baselines (including BERT and hierarchical attention networks) while offering enhanced interpretability. An ablation study confirms the importance of the multi-channel HSV representation and the dynamic boundary rows. Finally, we present visualizations of SemImage that qualitatively reveal clear patterns corresponding to topic shifts and sentiment changes in the generated image, suggesting that our representation makes these linguistic features visible to both humans and machines.
>
---
#### [new 205] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文针对真实图像超分辨率（Real-ISR）任务，解决扩散模型因低频偏置和“先低后高”层级导致高频细节重建不足的问题。提出FRAMER框架，通过频域对齐的自蒸馏与自适应调制，利用最终层特征指导中间层，结合内外对比损失优化低频全局结构与高频局部细节，显著提升重建质量。**

- **链接: [https://arxiv.org/pdf/2512.01390v1](https://arxiv.org/pdf/2512.01390v1)**

> **作者:** Seungho Choi; Jeahun Sung; Jihyong Oh
>
> **备注:** Comments: Please visit our project page at https://cmlab-korea.github.io/FRAMER/
>
> **摘要:** Real-image super-resolution (Real-ISR) seeks to recover HR images from LR inputs with mixed, unknown degradations. While diffusion models surpass GANs in perceptual quality, they under-reconstruct high-frequency (HF) details due to a low-frequency (LF) bias and a depth-wise "low-first, high-later" hierarchy. We introduce FRAMER, a plug-and-play training scheme that exploits diffusion priors without changing the backbone or inference. At each denoising step, the final-layer feature map teaches all intermediate layers. Teacher and student feature maps are decomposed into LF/HF bands via FFT masks to align supervision with the model's internal frequency hierarchy. For LF, an Intra Contrastive Loss (IntraCL) stabilizes globally shared structure. For HF, an Inter Contrastive Loss (InterCL) sharpens instance-specific details using random-layer and in-batch negatives. Two adaptive modulators, Frequency-based Adaptive Weight (FAW) and Frequency-based Alignment Modulation (FAM), reweight per-layer LF/HF signals and gate distillation by current similarity. Across U-Net and DiT backbones (e.g., Stable Diffusion 2, 3), FRAMER consistently improves PSNR/SSIM and perceptual metrics (LPIPS, NIQE, MANIQA, MUSIQ). Ablations validate the final-layer teacher and random-layer negatives.
>
---
#### [new 206] OmniFD: A Unified Model for Versatile Face Forgery Detection
- **分类: cs.CV**

- **简介: 该论文提出OmniFD，统一检测人脸伪造的四类任务：图像/视频分类、空间与时间定位。针对现有方法任务分离导致冗余的问题，设计共享编码器与跨任务交互模块，实现多任务协同学习，提升精度与效率，显著减少参数与训练时间。**

- **链接: [https://arxiv.org/pdf/2512.01128v1](https://arxiv.org/pdf/2512.01128v1)**

> **作者:** Haotian Liu; Haoyu Chen; Chenhui Pan; You Hu; Guoying Zhao; Xiaobai Li
>
> **摘要:** Face forgery detection encompasses multiple critical tasks, including identifying forged images and videos and localizing manipulated regions and temporal segments. Current approaches typically employ task-specific models with independent architectures, leading to computational redundancy and ignoring potential correlations across related tasks. We introduce OmniFD, a unified framework that jointly addresses four core face forgery detection tasks within a single model, i.e., image and video classification, spatial localization, and temporal localization. Our architecture consists of three principal components: (1) a shared Swin Transformer encoder that extracts unified 4D spatiotemporal representations from both images and video inputs, (2) a cross-task interaction module with learnable queries that dynamically captures inter-task dependencies through attention-based reasoning, and (3) lightweight decoding heads that transform refined representations into corresponding predictions for all FFD tasks. Extensive experiments demonstrate OmniFD's advantage over task-specific models. Its unified design leverages multi-task learning to capture generalized representations across tasks, especially enabling fine-grained knowledge transfer that facilitates other tasks. For example, video classification accuracy improves by 4.63% when image data are incorporated. Furthermore, by unifying images, videos and the four tasks within one framework, OmniFD achieves superior performance across diverse benchmarks with high efficiency and scalability, e.g., reducing 63% model parameters and 50% training time. It establishes a practical and generalizable solution for comprehensive face forgery detection in real-world applications. The source code is made available at https://github.com/haotianll/OmniFD.
>
---
#### [new 207] Learning What Helps: Task-Aligned Context Selection for Vision Tasks
- **分类: cs.CV**

- **简介: 该论文针对视觉任务中上下文选择不精准的问题，提出任务对齐的上下文选择（TACS）框架。通过联合训练选择器与任务模型，利用梯度与强化学习结合的方式，使上下文选取直接优化任务性能，提升模型在细粒度识别、医学图像分类与分割等任务中的表现，尤其在数据有限时效果显著。**

- **链接: [https://arxiv.org/pdf/2512.00489v1](https://arxiv.org/pdf/2512.00489v1)**

> **作者:** Jingyu Guo; Emir Konuk; Fredrik Strand; Christos Matsoukas; Kevin Smith
>
> **摘要:** Humans often resolve visual uncertainty by comparing an image with relevant examples, but ViTs lack the ability to identify which examples would improve their predictions. We present Task-Aligned Context Selection (TACS), a framework that learns to select paired examples which truly improve task performance rather than those that merely appear similar. TACS jointly trains a selector network with the task model through a hybrid optimization scheme combining gradient-based supervision and reinforcement learning, making retrieval part of the learning objective. By aligning selection with task rewards, TACS enables discriminative models to discover which contextual examples genuinely help. Across 18 datasets covering fine-grained recognition, medical image classification, and medical image segmentation, TACS consistently outperforms similarity-based retrieval, particularly in challenging or data-limited settings.
>
---
#### [new 208] Improved Mean Flows: On the Challenges of Fastforward Generative Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对快速生成模型（Fastforward Generative Models）中的训练目标不稳与引导机制僵化问题，提出改进的MeanFlow（iMF）方法。通过重定义损失为对瞬时速度的回归，并引入上下文条件化引导，实现单次函数评估下ImageNet 256×256上1.72 FID，显著优于已有方法。**

- **链接: [https://arxiv.org/pdf/2512.02012v1](https://arxiv.org/pdf/2512.02012v1)**

> **作者:** Zhengyang Geng; Yiyang Lu; Zongze Wu; Eli Shechtman; J. Zico Kolter; Kaiming He
>
> **备注:** Technical report
>
> **摘要:** MeanFlow (MF) has recently been established as a framework for one-step generative modeling. However, its ``fastforward'' nature introduces key challenges in both the training objective and the guidance mechanism. First, the original MF's training target depends not only on the underlying ground-truth fields but also on the network itself. To address this issue, we recast the objective as a loss on the instantaneous velocity $v$, re-parameterized by a network that predicts the average velocity $u$. Our reformulation yields a more standard regression problem and improves the training stability. Second, the original MF fixes the classifier-free guidance scale during training, which sacrifices flexibility. We tackle this issue by formulating guidance as explicit conditioning variables, thereby retaining flexibility at test time. The diverse conditions are processed through in-context conditioning, which reduces model size and benefits performance. Overall, our $\textbf{improved MeanFlow}$ ($\textbf{iMF}$) method, trained entirely from scratch, achieves $\textbf{1.72}$ FID with a single function evaluation (1-NFE) on ImageNet 256$\times$256. iMF substantially outperforms prior methods of this kind and closes the gap with multi-step methods while using no distillation. We hope our work will further advance fastforward generative modeling as a stand-alone paradigm.
>
---
#### [new 209] CauSight: Learning to Supersense for Visual Causal Discovery
- **分类: cs.CV**

- **简介: 该论文提出视觉因果发现任务，旨在让AI理解视觉场景中实体间的因果关系。为此构建了VCG-32K数据集，并提出CauSight模型，通过因果感知推理与强化学习实现精准因果推断，性能超越GPT-4.1，显著提升视觉因果理解能力。**

- **链接: [https://arxiv.org/pdf/2512.01827v1](https://arxiv.org/pdf/2512.01827v1)**

> **作者:** Yize Zhang; Meiqi Chen; Sirui Chen; Bo Peng; Yanxi Zhang; Tianyu Li; Chaochao Lu
>
> **备注:** project page: https://github.com/OpenCausaLab/CauSight
>
> **摘要:** Causal thinking enables humans to understand not just what is seen, but why it happens. To replicate this capability in modern AI systems, we introduce the task of visual causal discovery. It requires models to infer cause-and-effect relations among visual entities across diverse scenarios instead of merely perceiving their presence. To this end, we first construct the Visual Causal Graph dataset (VCG-32K), a large-scale collection of over 32,000 images annotated with entity-level causal graphs, and further develop CauSight, a novel vision-language model to perform visual causal discovery through causally aware reasoning. Our training recipe integrates three components: (1) training data curation from VCG-32K, (2) Tree-of-Causal-Thought (ToCT) for synthesizing reasoning trajectories, and (3) reinforcement learning with a designed causal reward to refine the reasoning policy. Experiments show that CauSight outperforms GPT-4.1 on visual causal discovery, achieving over a threefold performance boost (21% absolute gain). Our code, model, and dataset are fully open-sourced at project page: https://github.com/OpenCausaLab/CauSight.
>
---
#### [new 210] SMamDiff: Spatial Mamba for Stochastic Human Motion Prediction
- **分类: cs.CV**

- **简介: 该论文针对智能机器人场景下的人体运动预测任务，解决单阶段扩散模型中时空一致性不足的问题。提出SMamDiff模型，通过残差-DCT编码减少位置冗余，利用有序关节处理的时空Mamba模块建模长程依赖，实现高效、高保真的概率化运动预测。**

- **链接: [https://arxiv.org/pdf/2512.00355v1](https://arxiv.org/pdf/2512.00355v1)**

> **作者:** Junqiao Fan; Pengfei Liu; Haocong Rao
>
> **摘要:** With intelligent room-side sensing and service robots widely deployed, human motion prediction (HMP) is essential for safe, proactive assistance. However, many existing HMP methods either produce a single, deterministic forecast that ignores uncertainty or rely on probabilistic models that sacrifice kinematic plausibility. Diffusion models improve the accuracy-diversity trade-off but often depend on multi-stage pipelines that are costly for edge deployment. This work focuses on how to ensure spatial-temporal coherence within a single-stage diffusion model for HMP. We introduce SMamDiff, a Spatial Mamba-based Diffusion model with two novel designs: (i) a residual-DCT motion encoding that subtracts the last observed pose before a temporal DCT, reducing the first DC component ($f=0$) dominance and highlighting informative higher-frequency cues so the model learns how joints move rather than where they are; and (ii) a stickman-drawing spatial-mamba module that processes joints in an ordered, joint-by-joint manner, making later joints condition on earlier ones to induce long-range, cross-joint dependencies. On Human3.6M and HumanEva, these coherence mechanisms deliver state-of-the-art results among single-stage probabilistic HMP methods while using less latency and memory than multi-stage diffusion baselines.
>
---
#### [new 211] InternVideo-Next: Towards General Video Foundation Models without Video-Text Supervision
- **分类: cs.CV**

- **简介: 该论文针对视频表征学习任务，解决现有视频预训练模型依赖有噪声文本监督的问题。提出InternVideo-Next，通过解耦编码器-预测器-解码器架构，结合条件扩散解码与两阶段预训练，实现无文本监督下的语义一致且细节保真的视频表征学习，显著提升通用视频理解性能。**

- **链接: [https://arxiv.org/pdf/2512.01342v1](https://arxiv.org/pdf/2512.01342v1)**

> **作者:** Chenting Wang; Yuhan Zhu; Yicheng Xu; Jiange Yang; Ziang Yan; Yali Wang; Yi Wang; Limin Wang
>
> **摘要:** Large-scale video-text pretraining achieves strong performance but depends on noisy, synthetic captions with limited semantic coverage, often overlooking implicit world knowledge such as object motion, 3D geometry, and physical cues. In contrast, masked video modeling (MVM) directly exploits spatiotemporal structures but trails text-supervised methods on general tasks. We find this gap arises from overlooked architectural issues: pixel-level reconstruction struggles with convergence and its low-level requirement often conflicts with semantics, while latent prediction often encourages shortcut learning. To address these, we disentangle the traditional encoder-decoder design into an Encoder-Predictor-Decoder (EPD) framework, where the predictor acts as a latent world model, and propose InternVideo-Next, a two-stage pretraining scheme that builds a semantically consistent yet detail-preserving latent space for this world model. First, conventional linear decoder in pixel MVM enforces the predictor output latent to be linearly projected to, thus separable in pixel space, causing the conflict with semantic abstraction. Our Stage 1 proposes a conditional diffusion decoder and injects reliable image-level semantic priors to enhance semantics and convergence, thus bridging pixel-level fidelity with high-level semantic abstraction. Stage 2 further learns world knowledge by predicting frozen Stage 1 targets within this space, mitigating shortcut learning. Trained on public, unlabeled videos, InternVideo-Next achieves state-of-the-art results across benchmarks and provides a scalable path toward general video representation learning.
>
---
#### [new 212] DB-KAUNet: An Adaptive Dual Branch Kolmogorov-Arnold UNet for Retinal Vessel Segmentation
- **分类: cs.CV**

- **简介: 该论文针对视网膜血管分割任务，解决传统CNN难以捕捉长距离依赖和非线性关系的问题。提出自适应双分支Kolmogorov-Arnold UNet（DB-KAUNet），融合CNN与Transformer路径，通过交叉分支通道交互、空间特征增强及几何自适应融合模块，提升特征表达与分割精度。**

- **链接: [https://arxiv.org/pdf/2512.01657v1](https://arxiv.org/pdf/2512.01657v1)**

> **作者:** Hongyu Xu; Panpan Meng; Meng Wang; Dayu Hu; Liming Liang; Xiaoqi Sheng
>
> **摘要:** Accurate segmentation of retinal vessels is crucial for the clinical diagnosis of numerous ophthalmic and systemic diseases. However, traditional Convolutional Neural Network (CNN) methods exhibit inherent limitations, struggling to capture long-range dependencies and complex nonlinear relationships. To address the above limitations, an Adaptive Dual Branch Kolmogorov-Arnold UNet (DB-KAUNet) is proposed for retinal vessel segmentation. In DB-KAUNet, we design a Heterogeneous Dual-Branch Encoder (HDBE) that features parallel CNN and Transformer pathways. The HDBE strategically interleaves standard CNN and Transformer blocks with novel KANConv and KAT blocks, enabling the model to form a comprehensive feature representation. To optimize feature processing, we integrate several critical components into the HDBE. First, a Cross-Branch Channel Interaction (CCI) module is embedded to facilitate efficient interaction of channel features between the parallel pathways. Second, an attention-based Spatial Feature Enhancement (SFE) module is employed to enhance spatial features and fuse the outputs from both branches. Building upon the SFE module, an advanced Spatial Feature Enhancement with Geometrically Adaptive Fusion (SFE-GAF) module is subsequently developed. In the SFE-GAF module, adaptive sampling is utilized to focus on true vessel morphology precisely. The adaptive process strengthens salient vascular features while significantly reducing background noise and computational overhead. Extensive experiments on the DRIVE, STARE, and CHASE_DB1 datasets validate that DB-KAUNet achieves leading segmentation performance and demonstrates exceptional robustness.
>
---
#### [new 213] Asset-Driven Sematic Reconstruction of Dynamic Scene with Multi-Human-Object Interactions
- **分类: cs.CV**

- **简介: 该论文针对多人类-多物体动态场景的3D重建任务，解决单目视角下因复杂交互、频繁遮挡导致的结构不一致问题。提出融合3D生成模型、语义驱动变形与高斯溅射优化的混合方法，实现多视角与时间上一致的高质量几何重建，在HOI-M3数据集上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00547v1](https://arxiv.org/pdf/2512.00547v1)**

> **作者:** Sandika Biswas; Qianyi Wu; Biplab Banerjee; Hamid Rezatofighi
>
> **摘要:** Real-world human-built environments are highly dynamic, involving multiple humans and their complex interactions with surrounding objects. While 3D geometry modeling of such scenes is crucial for applications like AR/VR, gaming, and embodied AI, it remains underexplored due to challenges like diverse motion patterns and frequent occlusions. Beyond novel view rendering, 3D Gaussian Splatting (GS) has demonstrated remarkable progress in producing detailed, high-quality surface geometry with fast optimization of the underlying structure. However, very few GS-based methods address multihuman, multiobject scenarios, primarily due to the above-mentioned inherent challenges. In a monocular setup, these challenges are further amplified, as maintaining structural consistency under severe occlusion becomes difficult when the scene is optimized solely based on GS-based rendering loss. To tackle the challenges of such a multihuman, multiobject dynamic scene, we propose a hybrid approach that effectively combines the advantages of 1) 3D generative models for generating high-fidelity meshes of the scene elements, 2) Semantic-aware deformation, \ie rigid transformation of the rigid objects and LBS-based deformation of the humans, and mapping of the deformed high-fidelity meshes in the dynamic scene, and 3) GS-based optimization of the individual elements for further refining their alignments in the scene. Such a hybrid approach helps maintain the object structures even under severe occlusion and can produce multiview and temporally consistent geometry. We choose HOI-M3 for evaluation, as, to the best of our knowledge, this is the only dataset featuring multihuman, multiobject interactions in a dynamic scene. Our method outperforms the state-of-the-art method in producing better surface reconstruction of such scenes.
>
---
#### [new 214] Conceptual Evaluation of Deep Visual Stereo Odometry for the MARWIN Radiation Monitoring Robot in Accelerator Tunnels
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究机器人在加速器隧道中的自主定位问题，针对传统方法在低纹理、复杂环境下的局限性，提出采用基于深度视觉立体里程计（DVSO）的方案。通过纯视觉方式估计位姿与深度，结合3D几何约束提升精度，探索其在辐射监测机器人MARWIN上的应用潜力，以实现更灵活、低成本的自主导航。**

- **链接: [https://arxiv.org/pdf/2512.00080v1](https://arxiv.org/pdf/2512.00080v1)**

> **作者:** André Dehne; Juri Zach; Peer Stelldinger
>
> **摘要:** The MARWIN robot operates at the European XFEL to perform autonomous radiation monitoring in long, monotonous accelerator tunnels where conventional localization approaches struggle. Its current navigation concept combines lidar-based edge detection, wheel/lidar odometry with periodic QR-code referencing, and fuzzy control of wall distance, rotation, and longitudinal position. While robust in predefined sections, this design lacks flexibility for unknown geometries and obstacles. This paper explores deep visual stereo odometry (DVSO) with 3D-geometric constraints as a focused alternative. DVSO is purely vision-based, leveraging stereo disparity, optical flow, and self-supervised learning to jointly estimate depth and ego-motion without labeled data. For global consistency, DVSO can subsequently be fused with absolute references (e.g., landmarks) or other sensors. We provide a conceptual evaluation for accelerator tunnel environments, using the European XFEL as a case study. Expected benefits include reduced scale drift via stereo, low-cost sensing, and scalable data collection, while challenges remain in low-texture surfaces, lighting variability, computational load, and robustness under radiation. The paper defines a research agenda toward enabling MARWIN to navigate more autonomously in constrained, safety-critical infrastructures.
>
---
#### [new 215] FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention
- **分类: cs.CV**

- **简介: 该论文针对多视角3D重建中视觉几何变压器（VGGT）因自注意力机制导致的计算复杂度高、难以扩展的问题，提出FlashVGGT。通过压缩每帧空间信息为紧凑描述符，采用跨注意力机制降低计算开销，并引入分块递归在线推理，实现高效长序列处理，显著提升速度与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.01540v1](https://arxiv.org/pdf/2512.01540v1)**

> **作者:** Zipeng Wang; Dan Xu
>
> **摘要:** 3D reconstruction from multi-view images is a core challenge in computer vision. Recently, feed-forward methods have emerged as efficient and robust alternatives to traditional per-scene optimization techniques. Among them, state-of-the-art models like the Visual Geometry Grounding Transformer (VGGT) leverage full self-attention over all image tokens to capture global relationships. However, this approach suffers from poor scalability due to the quadratic complexity of self-attention and the large number of tokens generated in long image sequences. In this work, we introduce FlashVGGT, an efficient alternative that addresses this bottleneck through a descriptor-based attention mechanism. Instead of applying dense global attention across all tokens, FlashVGGT compresses spatial information from each frame into a compact set of descriptor tokens. Global attention is then computed as cross-attention between the full set of image tokens and this smaller descriptor set, significantly reducing computational overhead. Moreover, the compactness of the descriptors enables online inference over long sequences via a chunk-recursive mechanism that reuses cached descriptors from previous chunks. Experimental results show that FlashVGGT achieves reconstruction accuracy competitive with VGGT while reducing inference time to just 9.3% of VGGT for 1,000 images, and scaling efficiently to sequences exceeding 3,000 images. Our project page is available at https://wzpscott.github.io/flashvggt_page/.
>
---
#### [new 216] DCText: Scheduled Attention Masking for Visual Text Generation via Divide-and-Conquer Strategy
- **分类: cs.CV**

- **简介: 该论文针对文本生成图像中长文本或多重文本渲染不准确的问题，提出DCText方法。通过分治策略分解文本并分配至指定区域，结合注意力掩码与局部噪声初始化，在不增加计算成本下提升文本准确性与图像一致性，实现高效高精度视觉文本生成。**

- **链接: [https://arxiv.org/pdf/2512.01302v1](https://arxiv.org/pdf/2512.01302v1)**

> **作者:** Jaewoo Song; Jooyoung Choi; Kanghyun Baek; Sangyub Lee; Daemin Park; Sungroh Yoon
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Despite recent text-to-image models achieving highfidelity text rendering, they still struggle with long or multiple texts due to diluted global attention. We propose DCText, a training-free visual text generation method that adopts a divide-and-conquer strategy, leveraging the reliable short-text generation of Multi-Modal Diffusion Transformers. Our method first decomposes a prompt by extracting and dividing the target text, then assigns each to a designated region. To accurately render each segment within their regions while preserving overall image coherence, we introduce two attention masks - Text-Focus and Context-Expansion - applied sequentially during denoising. Additionally, Localized Noise Initialization further improves text accuracy and region alignment without increasing computational cost. Extensive experiments on single- and multisentence benchmarks show that DCText achieves the best text accuracy without compromising image quality while also delivering the lowest generation latency.
>
---
#### [new 217] Closing the Gap: Data-Centric Fine-Tuning of Vision Language Models for the Standardized Exam Questions
- **分类: cs.CV; cs.AI; cs.CL; cs.CY**

- **简介: 该论文聚焦于视觉语言模型在标准化考试题中的多模态推理任务，旨在提升开放权重模型性能。通过构建161.4百万词元的高质量多模态数据集并优化推理语法，实现78.6%准确率，仅比闭源模型低1.0%，验证了数据驱动方法在提升模型表现中的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.00042v1](https://arxiv.org/pdf/2512.00042v1)**

> **作者:** Egemen Sert; Şeyda Ertekin
>
> **摘要:** Multimodal reasoning has become a cornerstone of modern AI research. Standardized exam questions offer a uniquely rigorous testbed for such reasoning, providing structured visual contexts and verifiable answers. While recent progress has largely focused on algorithmic advances such as reinforcement learning (e.g., GRPO, DPO), the data centric foundations of vision language reasoning remain less explored. We show that supervised fine-tuning (SFT) with high-quality data can rival proprietary approaches. To this end, we compile a 161.4 million token multimodal dataset combining textbook question-solution pairs, curriculum aligned diagrams, and contextual materials, and fine-tune Qwen-2.5VL-32B using an optimized reasoning syntax (QMSA). The resulting model achieves 78.6% accuracy, only 1.0% below Gemini 2.0 Flash, on our newly released benchmark YKSUniform, which standardizes 1,854 multimodal exam questions across 309 curriculum topics. Our results reveal that data composition and representational syntax play a decisive role in multimodal reasoning. This work establishes a data centric framework for advancing open weight vision language models, demonstrating that carefully curated and curriculum-grounded multimodal data can elevate supervised fine-tuning to near state-of-the-art performance.
>
---
#### [new 218] Accelerating Inference of Masked Image Generators via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文针对掩码图像生成模型推理慢的问题，提出Speed-RL方法。通过将加速问题建模为强化学习任务，结合质量与速度奖励，微调预训练模型，实现3倍加速且保持高质量图像生成。**

- **链接: [https://arxiv.org/pdf/2512.01094v1](https://arxiv.org/pdf/2512.01094v1)**

> **作者:** Pranav Subbaraman; Shufan Li; Siyan Zhao; Aditya Grover
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Masked Generative Models (MGM)s demonstrate strong capabilities in generating high-fidelity images. However, they need many sampling steps to create high-quality generations, resulting in slow inference speed. In this work, we propose Speed-RL, a novel paradigm for accelerating a pretrained MGMs to generate high-quality images in fewer steps. Unlike conventional distillation methods which formulate the acceleration problem as a distribution matching problem, where a few-step student model is trained to match the distribution generated by a many-step teacher model, we consider this problem as a reinforcement learning problem. Since the goal of acceleration is to generate high quality images in fewer steps, we can combine a quality reward with a speed reward and finetune the base model using reinforcement learning with the combined reward as the optimization target. Through extensive experiments, we show that the proposed method was able to accelerate the base model by a factor of 3x while maintaining comparable image quality.
>
---
#### [new 219] FastAnimate: Towards Learnable Template Construction and Pose Deformation for Fast 3D Human Avatar Animation
- **分类: cs.CV**

- **简介: 该论文针对3D人体动画中的模板构建与姿态变形问题，提出FastAnimate框架。通过U-Net实现快速解耦纹理与姿态的模板生成，解决传统方法依赖复杂骨骼且易产伪影的问题；并引入数据驱动优化提升形变结构完整性，有效缓解LBS导致的形变失真，实现高效高质量动画生成。**

- **链接: [https://arxiv.org/pdf/2512.01444v1](https://arxiv.org/pdf/2512.01444v1)**

> **作者:** Jian Shu; Nanjie Yao; Gangjian Zhang; Junlong Ren; Yu Feng; Hao Wang
>
> **备注:** 9 pages,4 figures
>
> **摘要:** 3D human avatar animation aims at transforming a human avatar from an arbitrary initial pose to a specified target pose using deformation algorithms. Existing approaches typically divide this task into two stages: canonical template construction and target pose deformation. However, current template construction methods demand extensive skeletal rigging and often produce artifacts for specific poses. Moreover, target pose deformation suffers from structural distortions caused by Linear Blend Skinning (LBS), which significantly undermines animation realism. To address these problems, we propose a unified learning-based framework to address both challenges in two phases. For the former phase, to overcome the inefficiencies and artifacts during template construction, we leverage a U-Net architecture that decouples texture and pose information in a feed-forward process, enabling fast generation of a human template. For the latter phase, we propose a data-driven refinement technique that enhances structural integrity. Extensive experiments show that our model delivers consistent performance across diverse poses with an optimal balance between efficiency and quality,surpassing state-of-the-art (SOTA) methods.
>
---
#### [new 220] Low-Bitrate Video Compression through Semantic-Conditioned Diffusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DiSCo框架，针对超低码率下传统视频编码因追求像素精度而产生严重伪影的问题，通过语义条件扩散模型，将视频分解为文本、退化视频和可选草图/姿态三类紧凑模态，利用生成先验重建高质量视频。创新点包括时序前向填充、令牌交错和模态专用编码，显著提升感知质量，性能优于基线2-10倍。**

- **链接: [https://arxiv.org/pdf/2512.00408v1](https://arxiv.org/pdf/2512.00408v1)**

> **作者:** Lingdong Wang; Guan-Ming Su; Divya Kothandaraman; Tsung-Wei Huang; Mohammad Hajiesmaili; Ramesh K. Sitaraman
>
> **摘要:** Traditional video codecs optimized for pixel fidelity collapse at ultra-low bitrates and produce severe artifacts. This failure arises from a fundamental misalignment between pixel accuracy and human perception. We propose a semantic video compression framework named DiSCo that transmits only the most meaningful information while relying on generative priors for detail synthesis. The source video is decomposed into three compact modalities: a textual description, a spatiotemporally degraded video, and optional sketches or poses that respectively capture semantic, appearance, and motion cues. A conditional video diffusion model then reconstructs high-quality, temporally coherent videos from these compact representations. Temporal forward filling, token interleaving, and modality-specific codecs are proposed to improve multimodal generation and modality compactness. Experiments show that our method outperforms baseline semantic and traditional codecs by 2-10X on perceptual metrics at low bitrates.
>
---
#### [new 221] SAIDO: Generalizable Detection of AI-Generated Images via Scene-Aware and Importance-Guided Dynamic Optimization in Continual Learning
- **分类: cs.CV**

- **简介: 该论文针对AI生成图像检测中的泛化能力不足问题，提出SAIDO框架。通过场景感知的专家模块动态识别新场景，结合重要性引导的动态优化机制，提升模型在持续学习中对新型生成方法和内容的适应性，有效缓解灾难性遗忘，显著提升检测准确率与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.00539v1](https://arxiv.org/pdf/2512.00539v1)**

> **作者:** Yongkang Hu; Yu Cheng; Yushuo Zhang; Yuan Xie; Zhaoxia Yin
>
> **备注:** 17 pages, 19 figures
>
> **摘要:** The widespread misuse of image generation technologies has raised security concerns, driving the development of AI-generated image detection methods. However, generalization has become a key challenge and open problem: existing approaches struggle to adapt to emerging generative methods and content types in real-world scenarios. To address this issue, we propose a Scene-Aware and Importance-Guided Dynamic Optimization detection framework with continual learning (SAIDO). Specifically, we design Scene-Awareness-Based Expert Module (SAEM) that dynamically identifies and incorporates new scenes using VLLMs. For each scene, independent expert modules are dynamically allocated, enabling the framework to capture scene-specific forgery features better and enhance cross-scene generalization. To mitigate catastrophic forgetting when learning from multiple image generative methods, we introduce Importance-Guided Dynamic Optimization Mechanism (IDOM), which optimizes each neuron through an importance-guided gradient projection strategy, thereby achieving an effective balance between model plasticity and stability. Extensive experiments on continual learning tasks demonstrate that our method outperforms the current SOTA method in both stability and plasticity, achieving 44.22\% and 40.57\% relative reductions in average detection error rate and forgetting rate, respectively. On open-world datasets, it improves the average detection accuracy by 9.47\% compared to the current SOTA method.
>
---
#### [new 222] Optimizing Stroke Risk Prediction: A Machine Learning Pipeline Combining ROS-Balanced Ensembles and XAI
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医疗预测任务，旨在解决卒中风险早期预测的准确性与可解释性问题。通过构建融合ROS数据增强与多模型集成的机器学习框架，结合XAI技术，实现高精度（99.09%）且可解释的卒中风险预测，识别出年龄、高血压和血糖为关键影响因素。**

- **链接: [https://arxiv.org/pdf/2512.01333v1](https://arxiv.org/pdf/2512.01333v1)**

> **作者:** A S M Ahsanul Sarkar Akib; Raduana Khawla; Abdul Hasib
>
> **摘要:** Stroke is a major cause of death and permanent impairment, making it a major worldwide health concern. For prompt intervention and successful preventative tactics, early risk assessment is essential. To address this challenge, we used ensemble modeling and explainable AI (XAI) techniques to create an interpretable machine learning framework for stroke risk prediction. A thorough evaluation of 10 different machine learning models using 5-fold cross-validation across several datasets was part of our all-inclusive strategy, which also included feature engineering and data pretreatment (using Random Over-Sampling (ROS) to solve class imbalance). Our optimized ensemble model (Random Forest + ExtraTrees + XGBoost) performed exceptionally well, obtaining a strong 99.09% accuracy on the Stroke Prediction Dataset (SPD). We improved the model's transparency and clinical applicability by identifying three important clinical variables using LIME-based interpretability analysis: age, hypertension, and glucose levels. Through early prediction, this study highlights how combining ensemble learning with explainable AI (XAI) can deliver highly accurate and interpretable stroke risk assessment. By enabling data-driven prevention and personalized clinical decisions, our framework has the potential to transform stroke prediction and cardiovascular risk management.
>
---
#### [new 223] Adapter Shield: A Unified Framework with Built-in Authentication for Preventing Unauthorized Zero-Shot Image-to-Image Generation
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对零样本图像生成带来的身份与风格盗用风险，提出Adapter Shield框架。通过可逆加密机制与对抗性扰动，实现对输入图像的认证保护，仅授权用户可正常生成，未经授权则输出失真。有效防范未经授权的图像复制与模仿，兼顾安全性与访问控制。**

- **链接: [https://arxiv.org/pdf/2512.00075v1](https://arxiv.org/pdf/2512.00075v1)**

> **作者:** Jun Jia; Hongyi Miao; Yingjie Zhou; Wangqiu Zhou; Jianbo Zhang; Linhan Cao; Dandan Zhu; Hua Yang; Xiongkuo Min; Wei Sun; Guangtao Zhai
>
> **摘要:** With the rapid progress in diffusion models, image synthesis has advanced to the stage of zero-shot image-to-image generation, where high-fidelity replication of facial identities or artistic styles can be achieved using just one portrait or artwork, without modifying any model weights. Although these techniques significantly enhance creative possibilities, they also pose substantial risks related to intellectual property violations, including unauthorized identity cloning and stylistic imitation. To counter such threats, this work presents Adapter Shield, the first universal and authentication-integrated solution aimed at defending personal images from misuse in zero-shot generation scenarios. We first investigate how current zero-shot methods employ image encoders to extract embeddings from input images, which are subsequently fed into the UNet of diffusion models through cross-attention layers. Inspired by this mechanism, we construct a reversible encryption system that maps original embeddings into distinct encrypted representations according to different secret keys. The authorized users can restore the authentic embeddings via a decryption module and the correct key, enabling normal usage for authorized generation tasks. For protection purposes, we design a multi-target adversarial perturbation method that actively shifts the original embeddings toward designated encrypted patterns. Consequently, protected images are embedded with a defensive layer that ensures unauthorized users can only produce distorted or encrypted outputs. Extensive evaluations demonstrate that our method surpasses existing state-of-the-art defenses in blocking unauthorized zero-shot image synthesis, while supporting flexible and secure access control for verified users.
>
---
#### [new 224] Open-world Hand-Object Interaction Video Generation Based on Structure and Contact-aware Representation
- **分类: cs.CV**

- **简介: 该论文聚焦于开放世界手物交互视频生成任务，旨在解决现有方法在2D与3D表示间难以兼顾可扩展性与交互真实性的难题。提出一种无需3D标注的结构与接触感知表示，结合共享-专精联合生成框架，有效建模接触、遮挡与整体结构，提升视频物理真实性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.01677v1](https://arxiv.org/pdf/2512.01677v1)**

> **作者:** Haodong Yan; Hang Yu; Zhide Zhong; Weilin Yuan; Xin Gong; Zehang Luo; Chengxi Heyu; Junfeng Li; Wenxuan Song; Shunbo Zhou; Haoang Li
>
> **摘要:** Generating realistic hand-object interactions (HOI) videos is a significant challenge due to the difficulty of modeling physical constraints (e.g., contact and occlusion between hands and manipulated objects). Current methods utilize HOI representation as an auxiliary generative objective to guide video synthesis. However, there is a dilemma between 2D and 3D representations that cannot simultaneously guarantee scalability and interaction fidelity. To address this limitation, we propose a structure and contact-aware representation that captures hand-object contact, hand-object occlusion, and holistic structure context without 3D annotations. This interaction-oriented and scalable supervision signal enables the model to learn fine-grained interaction physics and generalize to open-world scenarios. To fully exploit the proposed representation, we introduce a joint-generation paradigm with a share-and-specialization strategy that generates interaction-oriented representations and videos. Extensive experiments demonstrate that our method outperforms state-of-the-art methods on two real-world datasets in generating physics-realistic and temporally coherent HOI videos. Furthermore, our approach exhibits strong generalization to challenging open-world scenarios, highlighting the benefit of our scalable design. Our project page is https://hgzn258.github.io/SCAR/.
>
---
#### [new 225] Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D点云配准任务，提出基于流匹配的端到端方法，将配准视为条件生成过程，直接生成对齐点云。通过学习点级速度场与测试时刚性约束，实现高效、高精度的单对及多视图配准，尤其在低重叠场景下表现优异，支持多模态、跨尺度应用。**

- **链接: [https://arxiv.org/pdf/2512.01850v1](https://arxiv.org/pdf/2512.01850v1)**

> **作者:** Yue Pan; Tao Sun; Liyuan Zhu; Lucas Nunes; Iro Armeni; Jens Behley; Cyrill Stachniss
>
> **备注:** 22 pages
>
> **摘要:** Point cloud registration aligns multiple unposed point clouds into a common frame, and is a core step for 3D reconstruction and robot localization. In this work, we cast registration as conditional generation: a learned continuous, point-wise velocity field transports noisy points to a registered scene, from which the pose of each view is recovered. Unlike previous methods that conduct correspondence matching to estimate the transformation between a pair of point clouds and then optimize the pairwise transformations to realize multi-view registration, our model directly generates the registered point cloud. With a lightweight local feature extractor and test-time rigidity enforcement, our approach achieves state-of-the-art results on pairwise and multi-view registration benchmarks, particularly with low overlap, and generalizes across scales and sensor modalities. It further supports downstream tasks including relocalization, multi-robot SLAM, and multi-session map merging. Source code available at: https://github.com/PRBonn/RAP.
>
---
#### [new 226] Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model
- **分类: cs.CV**

- **简介: 该论文针对单图像几何密集预测任务，解决因外观模糊和2D-3D映射非单射导致的病态问题。提出Lotus-2框架，通过两阶段确定性流程，利用预训练扩散模型的世界先验，实现稳定、精确的深度与法线估计，仅用5.9万样本即达新标杆性能。**

- **链接: [https://arxiv.org/pdf/2512.01030v1](https://arxiv.org/pdf/2512.01030v1)**

> **作者:** Jing He; Haodong Li; Mingzhi Sheng; Ying-Cong Chen
>
> **备注:** Work done at the Hong Kong University of Science and Technology (Guangzhou). Project page: https://lotus-2.github.io/. 15 Pages, 12 Figures, 3 Tables
>
> **摘要:** Recovering pixel-wise geometric properties from a single image is fundamentally ill-posed due to appearance ambiguity and non-injective mappings between 2D observations and 3D structures. While discriminative regression models achieve strong performance through large-scale supervision, their success is bounded by the scale, quality and diversity of available data and limited physical reasoning. Recent diffusion models exhibit powerful world priors that encode geometry and semantics learned from massive image-text data, yet directly reusing their stochastic generative formulation is suboptimal for deterministic geometric inference: the former is optimized for diverse and high-fidelity image generation, whereas the latter requires stable and accurate predictions. In this work, we propose Lotus-2, a two-stage deterministic framework for stable, accurate and fine-grained geometric dense prediction, aiming to provide an optimal adaption protocol to fully exploit the pre-trained generative priors. Specifically, in the first stage, the core predictor employs a single-step deterministic formulation with a clean-data objective and a lightweight local continuity module (LCM) to generate globally coherent structures without grid artifacts. In the second stage, the detail sharpener performs a constrained multi-step rectified-flow refinement within the manifold defined by the core predictor, enhancing fine-grained geometry through noise-free deterministic flow matching. Using only 59K training samples, less than 1% of existing large-scale datasets, Lotus-2 establishes new state-of-the-art results in monocular depth estimation and highly competitive surface normal prediction. These results demonstrate that diffusion models can serve as deterministic world priors, enabling high-quality geometric reasoning beyond traditional discriminative and generative paradigms.
>
---
#### [new 227] Lost in Distortion: Uncovering the Domain Gap Between Computer Vision and Brain Imaging - A Study on Pretraining for Age Prediction
- **分类: cs.CV**

- **简介: 该论文研究脑影像预训练中数据质量对年龄预测任务的影响。针对神经影像数据质量参差不齐的问题，通过在不同质量水平数据上预训练模型，评估其下游性能，揭示低质数据的双面作用。研究强调需结合临床标准进行领域感知的数据筛选，以构建可靠、可泛化的脑影像基础模型。**

- **链接: [https://arxiv.org/pdf/2512.01310v1](https://arxiv.org/pdf/2512.01310v1)**

> **作者:** Yanteng Zhang; Songheng Li; Zeyu Shen; Qizhen Lan; Lipei Zhang; Yang Liu; Vince Calhoun
>
> **摘要:** Large-scale brain imaging datasets provide unprecedented opportunities for developing domain foundation models through pretraining. However, unlike natural image datasets in computer vision, these neuroimaging data often exhibit high heterogeneity in quality, ranging from well-structured scans to severely distorted or incomplete brain volumes. This raises a fundamental question: can noise or low-quality scans contribute meaningfully to pretraining, or do they instead hinder model learning? In this study, we systematically explore the role of data quality level in pretraining and its impact on downstream tasks. Specifically, we perform pretraining on datasets with different quality levels and perform fine-tuning for brain age prediction on external cohorts. Our results show significant performance differences across quality levels, revealing both opportunities and limitations. We further discuss the gap between computer vision practices and clinical neuroimaging standards, emphasizing the necessity of domain-aware curation to ensure trusted and generalizable domain-specific foundation models.
>
---
#### [new 228] DPAC: Distribution-Preserving Adversarial Control for Diffusion Sampling
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对扩散模型中对抗引导采样导致的样本质量下降问题，提出DPAC方法。通过最小化路径空间KL散度（path-KL），建立控制能量与生成质量的理论联系，设计基于得分几何正交投影的对抗控制策略，有效抑制分布漂移，提升图像质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.01153v1](https://arxiv.org/pdf/2512.01153v1)**

> **作者:** Han-Jin Lee; Han-Ju Lee; Jin-Seong Kim; Seok-Hwan Choi
>
> **摘要:** Adversarially guided diffusion sampling often achieves the target class, but sample quality degrades as deviations between the adversarially controlled and nominal trajectories accumulate. We formalize this degradation as a path-space Kullback-Leibler divergence(path-KL) between controlled and nominal (uncontrolled) diffusion processes, thereby showing via Girsanov's theorem that it exactly equals the control energy. Building on this stochastic optimal control (SOC) view, we theoretically establish that minimizing this path-KL simultaneously tightens upper bounds on both the 2-Wasserstein distance and Fréchet Inception Distance (FID), revealing a principled connection between adversarial control energy and perceptual fidelity. From a variational perspective, we derive a first-order optimality condition for the control: among all directions that yield the same classification gain, the component tangent to iso-(log-)density surfaces (i.e., orthogonal to the score) minimizes path-KL, whereas the normal component directly increases distributional drift. This leads to DPAC (Distribution-Preserving Adversarial Control), a diffusion guidance rule that projects adversarial gradients onto the tangent space defined by the generative score geometry. We further show that in discrete solvers, the tangent projection cancels the O(Δt) leading error term in the Wasserstein distance, achieving an O(Δt^2) quality gap; moreover, it remains second-order robust to score or metric approximation. Empirical studies on ImageNet-100 validate the theoretical predictions, confirming that DPAC achieves lower FID and estimated path-KL at matched attack success rates.
>
---
#### [new 229] PAI-Bench: A Comprehensive Benchmark For Physical AI
- **分类: cs.CV**

- **简介: 该论文提出PAI-Bench，一个评估物理人工智能感知与预测能力的综合性基准，涵盖视频生成、条件视频生成和视频理解任务。针对当前多模态大模型在物理合理性与因果推理上的不足，构建2808个真实场景案例，量化评估模型表现，揭示现有系统在动态一致性与预测能力方面的短板，为未来研究提供方向。**

- **链接: [https://arxiv.org/pdf/2512.01989v1](https://arxiv.org/pdf/2512.01989v1)**

> **作者:** Fengzhe Zhou; Jiannan Huang; Jialuo Li; Deva Ramanan; Humphrey Shi
>
> **摘要:** Physical AI aims to develop models that can perceive and predict real-world dynamics; yet, the extent to which current multi-modal large language models and video generative models support these abilities is insufficiently understood. We introduce Physical AI Bench (PAI-Bench), a unified and comprehensive benchmark that evaluates perception and prediction capabilities across video generation, conditional video generation, and video understanding, comprising 2,808 real-world cases with task-aligned metrics designed to capture physical plausibility and domain-specific reasoning. Our study provides a systematic assessment of recent models and shows that video generative models, despite strong visual fidelity, often struggle to maintain physically coherent dynamics, while multi-modal large language models exhibit limited performance in forecasting and causal interpretation. These observations suggest that current systems are still at an early stage in handling the perceptual and predictive demands of Physical AI. In summary, PAI-Bench establishes a realistic foundation for evaluating Physical AI and highlights key gaps that future systems must address.
>
---
#### [new 230] Seeing through Imagination: Learning Scene Geometry via Implicit Spatial World Modeling
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型空间推理能力不足的问题，提出MILO框架，通过隐式世界建模与视觉生成器实现符号推理与感知经验的结合。引入相对位置编码RePE，并构建GeoGen数据集，显著提升模型对3D场景结构的理解能力。**

- **链接: [https://arxiv.org/pdf/2512.01821v1](https://arxiv.org/pdf/2512.01821v1)**

> **作者:** Meng Cao; Haokun Lin; Haoyuan Li; Haoran Tang; Rongtao Xu; Dong An; Xue Liu; Ian Reid; Xiaodan Liang
>
> **摘要:** Spatial reasoning, the ability to understand and interpret the 3D structure of the world, is a critical yet underdeveloped capability in Multimodal Large Language Models (MLLMs). Current methods predominantly rely on verbal descriptive tuning, which suffers from visual illiteracy, i.e., they learn spatial concepts through textual symbols alone, devoid of connection to their visual manifestations. To bridge this gap, this paper introduces MILO, an Implicit spatIaL wOrld modeling paradigm that simulates human-like spatial imagination. MILO integrates a visual generator to provide geometry-aware feedback, thereby implicitly grounding the MLLM's symbolic reasoning in perceptual experience. Complementing this paradigm, we propose RePE (Relative Positional Encoding), a novel encoding scheme that captures relative camera-pose transformations, offering superior performance over absolute coordinate systems. To support the training, we construct GeoGen, a large-scale Geometry-aware Generative dataset with approximately 2,241 videos and 67,827 observation-action-outcome triplets. Experiments demonstrate that our approach significantly enhances spatial reasoning capabilities across multiple baselines and benchmarks, offering a more holistic understanding of 3D space.
>
---
#### [new 231] CycliST: A Video Language Model Benchmark for Reasoning on Cyclical State Transitions
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出CycliST基准，用于评估视频语言模型在周期性状态转换上的推理能力。针对现有模型在时空认知与周期模式理解上的不足，构建了分层难度的合成视频数据集，揭示了主流模型在识别周期运动、时间依赖属性及量化分析方面的局限性，推动视觉推理模型发展。**

- **链接: [https://arxiv.org/pdf/2512.01095v1](https://arxiv.org/pdf/2512.01095v1)**

> **作者:** Simon Kohaut; Daniel Ochs; Shun Zhang; Benedict Flade; Julian Eggert; Kristian Kersting; Devendra Singh Dhami
>
> **摘要:** We present CycliST, a novel benchmark dataset designed to evaluate Video Language Models (VLM) on their ability for textual reasoning over cyclical state transitions. CycliST captures fundamental aspects of real-world processes by generating synthetic, richly structured video sequences featuring periodic patterns in object motion and visual attributes. CycliST employs a tiered evaluation system that progressively increases difficulty through variations in the number of cyclic objects, scene clutter, and lighting conditions, challenging state-of-the-art models on their spatio-temporal cognition. We conduct extensive experiments with current state-of-the-art VLMs, both open-source and proprietary, and reveal their limitations in generalizing to cyclical dynamics such as linear and orbital motion, as well as time-dependent changes in visual attributes like color and scale. Our results demonstrate that present-day VLMs struggle to reliably detect and exploit cyclic patterns, lack a notion of temporal understanding, and are unable to extract quantitative insights from scenes, such as the number of objects in motion, highlighting a significant technical gap that needs to be addressed. More specifically, we find no single model consistently leads in performance: neither size nor architecture correlates strongly with outcomes, and no model succeeds equally well across all tasks. By providing a targeted challenge and a comprehensive evaluation framework, CycliST paves the way for visual reasoning models that surpass the state-of-the-art in understanding periodic patterns.
>
---
#### [new 232] Structural Prognostic Event Modeling for Multimodal Cancer Survival Analysis
- **分类: cs.CV**

- **简介: 该论文针对多模态癌症生存分析任务，解决高维复杂数据中稀疏、未标注的预后事件难以捕捉的问题。提出SlotSPE框架，通过槽注意力压缩多模态数据为可解释的结构化事件表示，有效建模模态内/间交互，提升预测性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.01116v1](https://arxiv.org/pdf/2512.01116v1)**

> **作者:** Yilan Zhang; Li Nanbo; Changchun Yang; Jürgen Schmidhuber; Xin Gao
>
> **备注:** 37 pages, 14 Figures
>
> **摘要:** The integration of histology images and gene profiles has shown great promise for improving survival prediction in cancer. However, current approaches often struggle to model intra- and inter-modal interactions efficiently and effectively due to the high dimensionality and complexity of the inputs. A major challenge is capturing critical prognostic events that, though few, underlie the complexity of the observed inputs and largely determine patient outcomes. These events, manifested as high-level structural signals such as spatial histologic patterns or pathway co-activations, are typically sparse, patient-specific, and unannotated, making them inherently difficult to uncover. To address this, we propose SlotSPE, a slot-based framework for structural prognostic event modeling. Specifically, inspired by the principle of factorial coding, we compress each patient's multimodal inputs into compact, modality-specific sets of mutually distinctive slots using slot attention. By leveraging these slot representations as encodings for prognostic events, our framework enables both efficient and effective modeling of complex intra- and inter-modal interactions, while also facilitating seamless incorporation of biological priors that enhance prognostic relevance. Extensive experiments on ten cancer benchmarks show that SlotSPE outperforms existing methods in 8 out of 10 cohorts, achieving an overall improvement of 2.9%. It remains robust under missing genomic data and delivers markedly improved interpretability through structured event decomposition.
>
---
#### [new 233] Analysis of Incursive Breast Cancer in Mammograms Using YOLO, Explainability, and Domain Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对乳腺癌检测中深度学习模型对非乳腺影像（如CT、MRI）等分布外（OOD）输入的误检问题，提出融合YOLO与ResNet50的联合框架。通过余弦相似度构建域内图库实现OOD过滤，提升系统可靠性，同时保持高检测精度（mAP@0.5: 0.947）和可解释性。**

- **链接: [https://arxiv.org/pdf/2512.00129v1](https://arxiv.org/pdf/2512.00129v1)**

> **作者:** Jayan Adhikari; Prativa Joshi; Susish Baral
>
> **摘要:** Deep learning models for breast cancer detection from mammographic images have significant reliability problems when presented with Out-of-Distribution (OOD) inputs such as other imaging modalities (CT, MRI, X-ray) or equipment variations, leading to unreliable detection and misdiagnosis. The current research mitigates the fundamental OOD issue through a comprehensive approach integrating ResNet50-based OOD filtering with YOLO architectures (YOLOv8, YOLOv11, YOLOv12) for accurate detection of breast cancer. Our strategy establishes an in-domain gallery via cosine similarity to rigidly reject non-mammographic inputs prior to processing, ensuring that only domain-associated images supply the detection pipeline. The OOD detection component achieves 99.77\% general accuracy with immaculate 100\% accuracy on OOD test sets, effectively eliminating irrelevant imaging modalities. ResNet50 was selected as the optimum backbone after 12 CNN architecture searches. The joint framework unites OOD robustness with high detection performance (mAP@0.5: 0.947) and enhanced interpretability through Grad-CAM visualizations. Experimental validation establishes that OOD filtering significantly improves system reliability by preventing false alarms on out-of-distribution inputs while maintaining higher detection accuracy on mammographic data. The present study offers a fundamental foundation for the deployment of reliable AI-based breast cancer detection systems in diverse clinical environments with inherent data heterogeneity.
>
---
#### [new 234] THCRL: Trusted Hierarchical Contrastive Representation Learning for Multi-View Clustering
- **分类: cs.CV**

- **简介: 该论文针对多视图聚类中的不可信融合问题，提出THCRL框架。通过深度对称层次融合模块消除视图噪声，并引入基于平均K近邻的对比学习模块，增强同簇样本表示一致性，提升融合可信度，显著优化聚类性能。**

- **链接: [https://arxiv.org/pdf/2512.00368v1](https://arxiv.org/pdf/2512.00368v1)**

> **作者:** Jian Zhu
>
> **摘要:** Multi-View Clustering (MVC) has garnered increasing attention in recent years. It is capable of partitioning data samples into distinct groups by learning a consensus representation. However, a significant challenge remains: the problem of untrustworthy fusion. This problem primarily arises from two key factors: 1) Existing methods often ignore the presence of inherent noise within individual views; 2) In traditional MVC methods using Contrastive Learning (CL), similarity computations typically rely on different views of the same instance, while neglecting the structural information from nearest neighbors within the same cluster. Consequently, this leads to the wrong direction for multi-view fusion. To address this problem, we present a novel Trusted Hierarchical Contrastive Representation Learning (THCRL). It consists of two key modules. Specifically, we propose the Deep Symmetry Hierarchical Fusion (DSHF) module, which leverages the UNet architecture integrated with multiple denoising mechanisms to achieve trustworthy fusion of multi-view data. Furthermore, we present the Average K-Nearest Neighbors Contrastive Learning (AKCL) module to align the fused representation with the view-specific representation. Unlike conventional strategies, AKCL enhances representation similarity among samples belonging to the same cluster, rather than merely focusing on the same sample across views, thereby reinforcing the confidence of the fused representation. Extensive experiments demonstrate that THCRL achieves the state-of-the-art performance in deep MVC tasks.
>
---
#### [new 235] TBT-Former: Learning Temporal Boundary Distributions for Action Localization
- **分类: cs.CV**

- **简介: 该论文针对视频中动作定位（TAL）任务，解决边界模糊与多尺度信息融合难题。提出TBT-Former模型，通过增强Transformer骨干、引入跨尺度FPN结构及新型边界分布回归头，提升定位精度与不确定性建模能力，在THUMOS14、EPIC-Kitchens等数据集上取得领先性能。**

- **链接: [https://arxiv.org/pdf/2512.01298v1](https://arxiv.org/pdf/2512.01298v1)**

> **作者:** Thisara Rathnayaka; Uthayasanker Thayasivam
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Temporal Action Localization (TAL) remains a fundamental challenge in video understanding, aiming to identify the start time, end time, and category of all action instances within untrimmed videos. While recent single-stage, anchor-free models like ActionFormer have set a high standard by leveraging Transformers for temporal reasoning, they often struggle with two persistent issues: the precise localization of actions with ambiguous or "fuzzy" temporal boundaries and the effective fusion of multi-scale contextual information. In this paper, we introduce the Temporal Boundary Transformer (TBT-Former), a new architecture that directly addresses these limitations. TBT-Former enhances the strong ActionFormer baseline with three core contributions: (1) a higher-capacity scaled Transformer backbone with an increased number of attention heads and an expanded Multi-Layer Perceptron (MLP) dimension for more powerful temporal feature extraction; (2) a cross-scale feature pyramid network (FPN) that integrates a top-down pathway with lateral connections, enabling richer fusion of high-level semantics and low-level temporal details; and (3) a novel boundary distribution regression head. Inspired by the principles of Generalized Focal Loss (GFL), this new head recasts the challenging task of boundary regression as a more flexible probability distribution learning problem, allowing the model to explicitly represent and reason about boundary uncertainty. Within the paradigm of Transformer-based architectures, TBT-Former advances the formidable benchmark set by its predecessors, establishing a new level of performance on the highly competitive THUMOS14 and EPIC-Kitchens 100 datasets, while remaining competitive on the large-scale ActivityNet-1.3. Our code is available at https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects/tree/main/projects/210536K-Multi-Modal-Learning_Video-Understanding
>
---
#### [new 236] Rethinking Intracranial Aneurysm Vessel Segmentation: A Perspective from Computational Fluid Dynamics Applications
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对颅内动脉瘤血管分割任务，解决现有方法忽视后续流体动力学分析适用性的问题。构建首个多中心3D MRA数据集IAVS，包含641例图像与标注，并建立两阶段分割框架及标准化CFD适用性评估体系，推动分割结果向临床应用转化。**

- **链接: [https://arxiv.org/pdf/2512.01319v1](https://arxiv.org/pdf/2512.01319v1)**

> **作者:** Feiyang Xiao; Yichi Zhang; Xigui Li; Yuanye Zhou; Chen Jiang; Xin Guo; Limei Han; Yuxin Li; Fengping Zhu; Yuan Cheng
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** The precise segmentation of intracranial aneurysms and their parent vessels (IA-Vessel) is a critical step for hemodynamic analyses, which mainly depends on computational fluid dynamics (CFD). However, current segmentation methods predominantly focus on image-based evaluation metrics, often neglecting their practical effectiveness in subsequent CFD applications. To address this deficiency, we present the Intracranial Aneurysm Vessel Segmentation (IAVS) dataset, the first comprehensive, multi-center collection comprising 641 3D MRA images with 587 annotations of aneurysms and IA-Vessels. In addition to image-mask pairs, IAVS dataset includes detailed hemodynamic analysis outcomes, addressing the limitations of existing datasets that neglect topological integrity and CFD applicability. To facilitate the development and evaluation of clinically relevant techniques, we construct two evaluation benchmarks including global localization of aneurysms (Stage I) and fine-grained segmentation of IA-Vessel (Stage II) and develop a simple and effective two-stage framework, which can be used as a out-of-the-box method and strong baseline. For comprehensive evaluation of applicability of segmentation results, we establish a standardized CFD applicability evaluation system that enables the automated and consistent conversion of segmentation masks into CFD models, offering an applicability-focused assessment of segmentation outcomes. The dataset, code, and model will be public available at https://github.com/AbsoluteResonance/IAVS.
>
---
#### [new 237] RealGen: Photorealistic Text-to-Image Generation via Detector-Guided Rewards
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像生成中真实感不足的问题，提出RealGen框架。通过引入检测器奖励机制与GRPO算法，优化生成图像的细节与真实感，解决AI生成图像常见的人工痕迹问题。同时构建RealBench评估基准，实现无需人工参与的真实感评测。**

- **链接: [https://arxiv.org/pdf/2512.00473v1](https://arxiv.org/pdf/2512.00473v1)**

> **作者:** Junyan Ye; Leiqi Zhu; Yuncheng Guo; Dongzhi Jiang; Zilong Huang; Yifan Zhang; Zhiyuan Yan; Haohuan Fu; Conghui He; Weijia Li
>
> **摘要:** With the continuous advancement of image generation technology, advanced models such as GPT-Image-1 and Qwen-Image have achieved remarkable text-to-image consistency and world knowledge However, these models still fall short in photorealistic image generation. Even on simple T2I tasks, they tend to produce " fake" images with distinct AI artifacts, often characterized by "overly smooth skin" and "oily facial sheens". To recapture the original goal of "indistinguishable-from-reality" generation, we propose RealGen, a photorealistic text-to-image framework. RealGen integrates an LLM component for prompt optimization and a diffusion model for realistic image generation. Inspired by adversarial generation, RealGen introduces a "Detector Reward" mechanism, which quantifies artifacts and assesses realism using both semantic-level and feature-level synthetic image detectors. We leverage this reward signal with the GRPO algorithm to optimize the entire generation pipeline, significantly enhancing image realism and detail. Furthermore, we propose RealBench, an automated evaluation benchmark employing Detector-Scoring and Arena-Scoring. It enables human-free photorealism assessment, yielding results that are more accurate and aligned with real user experience. Experiments demonstrate that RealGen significantly outperforms general models like GPT-Image-1 and Qwen-Image, as well as specialized photorealistic models like FLUX-Krea, in terms of realism, detail, and aesthetics. The code is available at https://github.com/yejy53/RealGen.
>
---
#### [new 238] Fast, Robust, Permutation-and-Sign Invariant SO(3) Pattern Alignment
- **分类: cs.RO; cs.CG; cs.CV**

- **简介: 该论文解决旋转集合在SO(3)上的无对应对齐问题，针对时间不同步、异常值及轴约定未知的挑战。通过将旋转分解为球面上的变换基向量，利用快速鲁棒匹配器对齐，并设计置换与符号不变的包装器，实现线性复杂度下的精确对齐，显著提升速度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00659v1](https://arxiv.org/pdf/2512.00659v1)**

> **作者:** Anik Sarker; Alan T. Asbeck
>
> **摘要:** We address the correspondence-free alignment of two rotation sets on \(SO(3)\), a core task in calibration and registration that is often impeded by missing time alignment, outliers, and unknown axis conventions. Our key idea is to decompose each rotation into its \emph{Transformed Basis Vectors} (TBVs)-three unit vectors on \(S^2\)-and align the resulting spherical point sets per axis using fast, robust matchers (SPMC, FRS, and a hybrid). To handle axis relabels and sign flips, we introduce a \emph{Permutation-and-Sign Invariant} (PASI) wrapper that enumerates the 24 proper signed permutations, scores them via summed correlations, and fuses the per-axis estimates into a single rotation by projection/Karcher mean. The overall complexity remains linear in the number of rotations (\(\mathcal{O}(n)\)), contrasting with \(\mathcal{O}(N_r^3\log N_r)\) for spherical/\(SO(3)\) correlation. Experiments on EuRoC Machine Hall simulations (axis-consistent) and the ETH Hand-Eye benchmark (\texttt{robot\_arm\_real}) (axis-ambiguous) show that our methods are accurate, 6-60x faster than traditional methods, and robust under extreme outlier ratios (up to 90\%), all without correspondence search.
>
---
#### [new 239] RealAppliance: Let High-fidelity Appliance Assets Controllable and Workable as Aligned Real Manuals
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对家电仿真与真实手册不一致导致的模拟-现实差距问题，构建了100个高保真、机制完整的家电资产数据集RealAppliance，并提出RealAppliance-Bench基准。旨在评估多模态大模型与具身操作规划模型在家电操作任务中的表现，推动家电操控研究发展。**

- **链接: [https://arxiv.org/pdf/2512.00287v1](https://arxiv.org/pdf/2512.00287v1)**

> **作者:** Yuzheng Gao; Yuxing Long; Lei Kang; Yuchong Guo; Ziyan Yu; Shangqing Mao; Jiyao Zhang; Ruihai Wu; Dongjiang Li; Hui Shen; Hao Dong
>
> **摘要:** Existing appliance assets suffer from poor rendering, incomplete mechanisms, and misalignment with manuals, leading to simulation-reality gaps that hinder appliance manipulation development. In this work, we introduce the RealAppliance dataset, comprising 100 high-fidelity appliances with complete physical, electronic mechanisms, and program logic aligned with their manuals. Based on these assets, we propose the RealAppliance-Bench benchmark, which evaluates multimodal large language models and embodied manipulation planning models across key tasks in appliance manipulation planning: manual page retrieval, appliance part grounding, open-loop manipulation planning, and closed-loop planning adjustment. Our analysis of model performances on RealAppliance-Bench provides insights for advancing appliance manipulation research
>
---
#### [new 240] ICD-Net: Inertial Covariance Displacement Network for Drone Visual-Inertial SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对无人机视觉-惯性SLAM中传感器误差与环境挑战导致的精度下降问题，提出ICD-Net框架。通过神经网络直接从原始惯性数据学习位移估计及不确定性，将其作为残差约束融入VINS-Fusion优化，有效提升轨迹精度与系统鲁棒性，尤其在视觉退化时表现优异。**

- **链接: [https://arxiv.org/pdf/2512.00037v1](https://arxiv.org/pdf/2512.00037v1)**

> **作者:** Tali Orlev Shapira; Itzik Klein
>
> **摘要:** Visual-inertial SLAM systems often exhibit suboptimal performance due to multiple confounding factors including imperfect sensor calibration, noisy measurements, rapid motion dynamics, low illumination, and the inherent limitations of traditional inertial navigation integration methods. These issues are particularly problematic in drone applications where robust and accurate state estimation is critical for safe autonomous operation. In this work, we present ICD-Net, a novel framework that enhances visual-inertial SLAM performance by learning to process raw inertial measurements and generating displacement estimates with associated uncertainty quantification. Rather than relying on analytical inertial sensor models that struggle with real-world sensor imperfections, our method directly extracts displacement maps from sensor data while simultaneously predicting measurement covariances that reflect estimation confidence. We integrate ICD-Net outputs as additional residual constraints into the VINS-Fusion optimization framework, where the predicted uncertainties appropriately weight the neural network contributions relative to traditional visual and inertial terms. The learned displacement constraints provide complementary information that compensates for various error sources in the SLAM pipeline. Our approach can be used under both normal operating conditions and in situations of camera inconsistency or visual degradation. Experimental evaluation on challenging high-speed drone sequences demonstrated that our approach significantly improved trajectory estimation accuracy compared to standard VINS-Fusion, with more than 38% improvement in mean APE and uncertainty estimates proving crucial for maintaining system robustness. Our method shows that neural network enhancement can effectively address multiple sources of SLAM degradation while maintaining real-time performance requirements.
>
---
#### [new 241] Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中3D视觉预训练方法性能不足的问题，提出AFRO框架。它通过生成式扩散过程建模状态预测，联合学习前后向动态以捕捉因果转移结构，避免显式几何重建与动作监督。采用特征差分与逆一致性监督提升特征质量。实验表明，AFRO在16个仿真和4个真实任务中显著提升操作成功率，具备良好可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.00074v1](https://arxiv.org/pdf/2512.00074v1)**

> **作者:** Qiwei Liang; Boyang Cai; Minghao Lai; Sitong Zhuang; Tao Lin; Yan Qin; Yixuan Ye; Jiaming Liang; Renjing Xu
>
> **摘要:** Despite strong results on recognition and segmentation, current 3D visual pre-training methods often underperform on robotic manipulation. We attribute this gap to two factors: the lack of state-action-state dynamics modeling and the unnecessary redundancy of explicit geometric reconstruction. We introduce AFRO, a self-supervised framework that learns dynamics-aware 3D representations without action or reconstruction supervision. AFRO casts state prediction as a generative diffusion process and jointly models forward and inverse dynamics in a shared latent space to capture causal transition structure. To prevent feature leakage in action learning, we employ feature differencing and inverse-consistency supervision, improving the quality and stability of visual features. When combined with Diffusion Policy, AFRO substantially increases manipulation success rates across 16 simulated and 4 real-world tasks, outperforming existing pre-training approaches. The framework also scales favorably with data volume and task complexity. Qualitative visualizations indicate that AFRO learns semantically rich, discriminative features, offering an effective pre-training solution for 3D representation learning in robotics. Project page: https://kolakivy.github.io/AFRO/
>
---
#### [new 242] A Survey on Improving Human Robot Collaboration through Vision-and-Language Navigation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC**

- **简介: 该论文聚焦视觉-语言导航（VLN）任务，旨在提升人机协作中的多机器人协同能力。针对当前模型在双向沟通、歧义消解与协作决策方面的不足，系统综述了近200篇相关研究，提出需引入主动澄清、实时反馈与动态角色分配机制，以推动高效、可扩展的智能协作系统发展。**

- **链接: [https://arxiv.org/pdf/2512.00027v1](https://arxiv.org/pdf/2512.00027v1)**

> **作者:** Nivedan Yakolli; Avinash Gautam; Abhijit Das; Yuankai Qi; Virendra Singh Shekhawat
>
> **摘要:** Vision-and-Language Navigation (VLN) is a multi-modal, cooperative task requiring agents to interpret human instructions, navigate 3D environments, and communicate effectively under ambiguity. This paper presents a comprehensive review of recent VLN advancements in robotics and outlines promising directions to improve multi-robot coordination. Despite progress, current models struggle with bidirectional communication, ambiguity resolution, and collaborative decision-making in the multi-agent systems. We review approximately 200 relevant articles to provide an in-depth understanding of the current landscape. Through this survey, we aim to provide a thorough resource that inspires further research at the intersection of VLN and robotics. We advocate that the future VLN systems should support proactive clarification, real-time feedback, and contextual reasoning through advanced natural language understanding (NLU) techniques. Additionally, decentralized decision-making frameworks with dynamic role assignment are essential for scalable, efficient multi-robot collaboration. These innovations can significantly enhance human-robot interaction (HRI) and enable real-world deployment in domains such as healthcare, logistics, and disaster response.
>
---
#### [new 243] Arcadia: Toward a Full-Lifecycle Framework for Embodied Lifelong Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Arcadia框架，解决具身智能体终身学习的生命周期问题。针对单一阶段优化难以持续改进与泛化的问题，构建四阶段闭环：自主数据采集、生成式场景重建、共享表示学习、仿真评估演化。通过紧密耦合实现持续进化与真实迁移，推动具身智能向通用化发展。**

- **链接: [https://arxiv.org/pdf/2512.00076v1](https://arxiv.org/pdf/2512.00076v1)**

> **作者:** Minghe Gao; Juncheng Li; Yuze Lin; Xuqi Liu; Jiaming Ji; Xiaoran Pan; Zihan Xu; Xian Li; Mingjie Li; Wei Ji; Rong Wei; Rui Tang; Qizhou Wang; Kai Shen; Jun Xiao; Qi Wu; Siliang Tang; Yueting Zhuang
>
> **摘要:** We contend that embodied learning is fundamentally a lifecycle problem rather than a single-stage optimization. Systems that optimize only one link (data collection, simulation, learning, or deployment) rarely sustain improvement or generalize beyond narrow settings. We introduce Arcadia, a closed-loop framework that operationalizes embodied lifelong learning by tightly coupling four stages: (1) Self-evolving exploration and grounding for autonomous data acquisition in physical environments, (2) Generative scene reconstruction and augmentation for realistic and extensible scene creation, (3) a Shared embodied representation architecture that unifies navigation and manipulation within a single multimodal backbone, and (4) Sim-from-real evaluation and evolution that closes the feedback loop through simulation-based adaptation. This coupling is non-decomposable: removing any stage breaks the improvement loop and reverts to one-shot training. Arcadia delivers consistent gains on navigation and manipulation benchmarks and transfers robustly to physical robots, indicating that a tightly coupled lifecycle: continuous real-world data acquisition, generative simulation update, and shared-representation learning, supports lifelong improvement and end-to-end generalization. We release standardized interfaces enabling reproducible evaluation and cross-model comparison in reusable environments, positioning Arcadia as a scalable foundation for general-purpose embodied agents.
>
---
#### [new 244] TIE: A Training-Inversion-Exclusion Framework for Visually Interpretable and Uncertainty-Guided Out-of-Distribution Detection
- **分类: cs.LG; cs.CV; eess.IV; stat.ML**

- **简介: 该论文提出TIE框架，解决深度模型对分布外（OOD）样本识别能力弱、预测过度自信的问题。通过训练-反演-排除的闭环机制，引入“垃圾类”并生成可解释的异常原型，实现统一的不确定性估计与OOD检测，无需外部数据，在多个基准上达到近乎零误报率。**

- **链接: [https://arxiv.org/pdf/2512.00229v1](https://arxiv.org/pdf/2512.00229v1)**

> **作者:** Pirzada Suhail; Rehna Afroz; Amit Sethi
>
> **摘要:** Deep neural networks often struggle to recognize when an input lies outside their training experience, leading to unreliable and overconfident predictions. Building dependable machine learning systems therefore requires methods that can both estimate predictive \textit{uncertainty} and detect \textit{out-of-distribution (OOD)} samples in a unified manner. In this paper, we propose \textbf{TIE: a Training--Inversion--Exclusion} framework for visually interpretable and uncertainty-guided anomaly detection that jointly addresses these challenges through iterative refinement. TIE extends a standard $n$-class classifier to an $(n+1)$-class model by introducing a garbage class initialized with Gaussian noise to represent outlier inputs. Within each epoch, TIE performs a closed-loop process of \textit{training, inversion, and exclusion}, where highly uncertain inverted samples reconstructed from the just-trained classifier are excluded into the garbage class. Over successive iterations, the inverted samples transition from noisy artifacts into visually coherent class prototypes, providing transparent insight into how the model organizes its learned manifolds. During inference, TIE rejects OOD inputs by either directly mapping them to the garbage class or producing low-confidence, uncertain misclassifications within the in-distribution classes that are easily separable, all without relying on external OOD datasets. A comprehensive threshold-based evaluation using multiple OOD metrics and performance measures such as \textit{AUROC}, \textit{AUPR}, and \textit{FPR@95\%TPR} demonstrates that TIE offers a unified and interpretable framework for robust anomaly detection and calibrated uncertainty estimation (UE) achieving near-perfect OOD detection with \textbf{\(\!\approx\!\) 0 FPR@95\%TPR} when trained on MNIST or FashionMNIST and tested against diverse unseen datasets.
>
---
#### [new 245] RoaD: Rollouts as Demonstrations for Closed-Loop Supervised Fine-Tuning of Autonomous Driving Policies
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对自动驾驶政策在闭环部署时因协变量偏移导致的误差累积问题，提出RoaD方法。通过利用策略自身生成的闭环轨迹作为示范数据，结合专家引导提升轨迹质量，实现高效闭环微调。实验表明，RoaD在仿真环境中显著提升驾驶性能并减少碰撞，且所需数据远少于强化学习。**

- **链接: [https://arxiv.org/pdf/2512.01993v1](https://arxiv.org/pdf/2512.01993v1)**

> **作者:** Guillermo Garcia-Cobo; Maximilian Igl; Peter Karkus; Zhejun Zhang; Michael Watson; Yuxiao Chen; Boris Ivanovic; Marco Pavone
>
> **备注:** Preprint
>
> **摘要:** Autonomous driving policies are typically trained via open-loop behavior cloning of human demonstrations. However, such policies suffer from covariate shift when deployed in closed loop, leading to compounding errors. We introduce Rollouts as Demonstrations (RoaD), a simple and efficient method to mitigate covariate shift by leveraging the policy's own closed-loop rollouts as additional training data. During rollout generation, RoaD incorporates expert guidance to bias trajectories toward high-quality behavior, producing informative yet realistic demonstrations for fine-tuning. This approach enables robust closed-loop adaptation with orders of magnitude less data than reinforcement learning, and avoids restrictive assumptions of prior closed-loop supervised fine-tuning (CL-SFT) methods, allowing broader applications domains including end-to-end driving. We demonstrate the effectiveness of RoaD on WOSAC, a large-scale traffic simulation benchmark, where it performs similar or better than the prior CL-SFT method; and in AlpaSim, a high-fidelity neural reconstruction-based simulator for end-to-end driving, where it improves driving score by 41\% and reduces collisions by 54\%.
>
---
#### [new 246] First On-Orbit Demonstration of a Geospatial Foundation Model
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究地球观测中资源受限空间硬件上的人工智能部署问题。针对大型地理空间基础模型（GeoFM）难以在轨运行的挑战，提出紧凑型视觉变换器架构，通过模型压缩与领域适应，在保持性能的同时实现轻量化。首次在国际空间站成功验证了其在轨推理能力，为星载AI应用提供了可行路径。**

- **链接: [https://arxiv.org/pdf/2512.01181v1](https://arxiv.org/pdf/2512.01181v1)**

> **作者:** Andrew Du; Roberto Del Prete; Alejandro Mousist; Nick Manser; Fabrice Marre; Andrew Barton; Carl Seubert; Gabriele Meoni; Tat-Jun Chin
>
> **摘要:** Geospatial foundation models (GeoFMs) promise broad generalisation capacity for Earth observation (EO) tasks, particularly under data-limited conditions. However, their large size poses a barrier to deployment on resource-constrained space hardware. To address this, we present compact variants of a Vision Transformer (ViT)-based GeoFM that preserve downstream task performance while enabling onboard execution. Evaluation across five downstream tasks and validation in two representative flight environments show that model compression and domain adaptation are critical to reducing size and resource demands while maintaining high performance under operational conditions. We further demonstrate reliable on-orbit inference with the IMAGIN-e payload aboard the International Space Station. These results establish a pathway from large GeoFMs to flight-ready, resource-efficient deployments, expanding the feasibility of onboard AI for EO missions.
>
---
#### [new 247] Learning from Watching: Scalable Extraction of Manipulation Trajectories from Human Videos
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人学习中数据采集成本高的问题，提出利用互联网人类操作视频提取密集的操纵关键点轨迹。通过结合大模型视频理解与点追踪技术，实现对任务相关关键点的全流程精准跟踪，提升数据规模与利用效率，推动更高效、可扩展的机器人学习。**

- **链接: [https://arxiv.org/pdf/2512.00024v1](https://arxiv.org/pdf/2512.00024v1)**

> **作者:** X. Hu; G. Ye
>
> **备注:** Accepted to RSS 2025 Workshop
>
> **摘要:** Collecting high-quality data for training large-scale robotic models typically relies on real robot platforms, which is labor-intensive and costly, whether via teleoperation or scripted demonstrations. To scale data collection, many researchers have turned to leveraging human manipulation videos available online. However, current methods predominantly focus on hand detection or object pose estimation, failing to fully exploit the rich interaction cues embedded in these videos. In this work, we propose a novel approach that combines large foundation models for video understanding with point tracking techniques to extract dense trajectories of all task-relevant keypoints during manipulation. This enables more comprehensive utilization of Internet-scale human demonstration videos. Experimental results demonstrate that our method can accurately track keypoints throughout the entire manipulation process, paving the way for more scalable and data-efficient robot learning.
>
---
#### [new 248] Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉引导的人形机器人开锁任务，解决模拟到现实的零样本迁移问题。提出教师-学生自举框架，结合分阶段重置探索与GRPO微调，实现仅用RGB图像的端到端控制，在多种门类型上超越人类操作者31.7%的任务效率。**

- **链接: [https://arxiv.org/pdf/2512.01061v1](https://arxiv.org/pdf/2512.01061v1)**

> **作者:** Haoru Xue; Tairan He; Zi Wang; Qingwei Ben; Wenli Xiao; Zhengyi Luo; Xingye Da; Fernando Castañeda; Guanya Shi; Shankar Sastry; Linxi "Jim" Fan; Yuke Zhu
>
> **备注:** https://doorman-humanoid.github.io/
>
> **摘要:** Recent progress in GPU-accelerated, photorealistic simulation has opened a scalable data-generation path for robot learning, where massive physics and visual randomization allow policies to generalize beyond curated environments. Building on these advances, we develop a teacher-student-bootstrap learning framework for vision-based humanoid loco-manipulation, using articulated-object interaction as a representative high-difficulty benchmark. Our approach introduces a staged-reset exploration strategy that stabilizes long-horizon privileged-policy training, and a GRPO-based fine-tuning procedure that mitigates partial observability and improves closed-loop consistency in sim-to-real RL. Trained entirely on simulation data, the resulting policy achieves robust zero-shot performance across diverse door types and outperforms human teleoperators by up to 31.7% in task completion time under the same whole-body control stack. This represents the first humanoid sim-to-real policy capable of diverse articulated loco-manipulation using pure RGB perception.
>
---
#### [new 249] Ternary-Input Binary-Weight CNN Accelerator Design for Miniature Object Classification System with Query-Driven Spatial DVS
- **分类: cs.AR; cs.CV; eess.IV**

- **简介: 该论文针对微型成像系统中内存与功耗受限问题，提出一种基于空间DVS的三值输入二值权重CNN加速器。通过像素共享实现时空模式切换，结合量化设计，显著降低计算与存储开销，实现在28nm工艺下440ms推理、1.6mW功耗，性能提升7.3倍。**

- **链接: [https://arxiv.org/pdf/2512.00138v1](https://arxiv.org/pdf/2512.00138v1)**

> **作者:** Yuyang Li; Swasthik Muloor; Jack Laudati; Nickolas Dematteis; Yidam Park; Hana Kim; Nathan Chang; Inhee Lee
>
> **备注:** 6 pages.12 figures & 2 table
>
> **摘要:** Miniature imaging systems are essential for space-constrained applications but are limited by memory and power constraints. While machine learning can reduce data size by extracting key features, its high energy demands often exceed the capacity of small batteries. This paper presents a CNN hardware accelerator optimized for object classification in miniature imaging systems. It processes data from a spatial Dynamic Vision Sensor (DVS), reconfigurable to a temporal DVS via pixel sharing, minimizing sensor area. By using ternary DVS outputs and a ternary-input, binary-weight neural network, the design reduces computation and memory needs. Fabricated in 28 nm CMOS, the accelerator cuts data size by 81% and MAC operations by 27%. It achieves 440 ms inference time at just 1.6 mW power consumption, improving the Figure-of-Merit (FoM) by 7.3x over prior CNN accelerators for miniature systems.
>
---
#### [new 250] Art2Music: Generating Music for Art Images with Multi-modal Feeling Alignment
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文提出Art2Music，解决艺术图像与音乐间的感知自然、情感对齐的跨模态生成问题。构建ArtiCaps数据集，设计轻量级框架，通过融合图像与文本特征生成高质量音乐，显著提升音色保真度与情感一致性，适用于互动艺术与个性化声景。**

- **链接: [https://arxiv.org/pdf/2512.00120v1](https://arxiv.org/pdf/2512.00120v1)**

> **作者:** Jiaying Hong; Ting Zhu; Thanet Markchom; Huizhi Liang
>
> **摘要:** With the rise of AI-generated content (AIGC), generating perceptually natural and feeling-aligned music from multimodal inputs has become a central challenge. Existing approaches often rely on explicit emotion labels that require costly annotation, underscoring the need for more flexible feeling-aligned methods. To support multimodal music generation, we construct ArtiCaps, a pseudo feeling-aligned image-music-text dataset created by semantically matching descriptions from ArtEmis and MusicCaps. We further propose Art2Music, a lightweight cross-modal framework that synthesizes music from artistic images and user comments. In the first stage, images and text are encoded with OpenCLIP and fused using a gated residual module; the fused representation is decoded by a bidirectional LSTM into Mel-spectrograms with a frequency-weighted L1 loss to enhance high-frequency fidelity. In the second stage, a fine-tuned HiFi-GAN vocoder reconstructs high-quality audio waveforms. Experiments on ArtiCaps show clear improvements in Mel-Cepstral Distortion, Frechet Audio Distance, Log-Spectral Distance, and cosine similarity. A small LLM-based rating study further verifies consistent cross-modal feeling alignment and offers interpretable explanations of matches and mismatches across modalities. These results demonstrate improved perceptual naturalness, spectral fidelity, and semantic consistency. Art2Music also maintains robust performance with only 50k training samples, providing a scalable solution for feeling-aligned creative audio generation in interactive art, personalized soundscapes, and digital art exhibitions.
>
---
#### [new 251] Audio-Visual World Models: Towards Multisensory Imagination in Sight and Sound
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出首个音频-视觉世界模型（AVWM）框架，解决多模态环境建模中视听同步与任务奖励预测问题。构建AVW-4k数据集，设计AV-CDiT模型实现高保真视听动态预测，显著提升连续导航任务性能。**

- **链接: [https://arxiv.org/pdf/2512.00883v1](https://arxiv.org/pdf/2512.00883v1)**

> **作者:** Jiahua Wang; Shannan Yan; Leqi Zheng; Jialong Wu; Yaoxin Mao
>
> **摘要:** World models simulate environmental dynamics to enable agents to plan and reason about future states. While existing approaches have primarily focused on visual observations, real-world perception inherently involves multiple sensory modalities. Audio provides crucial spatial and temporal cues such as sound source localization and acoustic scene properties, yet its integration into world models remains largely unexplored. No prior work has formally defined what constitutes an audio-visual world model or how to jointly capture binaural spatial audio and visual dynamics under precise action control with task reward prediction. This work presents the first formal framework for Audio-Visual World Models (AVWM), formulating multimodal environment simulation as a partially observable Markov decision process with synchronized audio-visual observations, fine-grained actions, and task rewards. To address the lack of suitable training data, we construct AVW-4k, a dataset comprising 30 hours of binaural audio-visual trajectories with action annotations and reward signals across 76 indoor environments. We propose AV-CDiT, an Audio-Visual Conditional Diffusion Transformer with a novel modality expert architecture that balances visual and auditory learning, optimized through a three-stage training strategy for effective multimodal integration. Extensive experiments demonstrate that AV-CDiT achieves high-fidelity multimodal prediction across visual and auditory modalities with reward. Furthermore, we validate its practical utility in continuous audio-visual navigation tasks, where AVWM significantly enhances the agent's performance.
>
---
#### [new 252] NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对长时序复杂语言指令下的具身导航任务，解决现有模型在未知环境中的长期规划与预测能力不足问题。提出NavForesee，一个统一视觉-语言世界模型，融合显式语言规划与隐式时空预测，实现层级规划与双时间尺度导航预测，通过感知-规划/预测-行动的闭环提升导航性能。**

- **链接: [https://arxiv.org/pdf/2512.01550v1](https://arxiv.org/pdf/2512.01550v1)**

> **作者:** Fei Liu; Shichao Xie; Minghua Luo; Zedong Chu; Junjun Hu; Xiaolong Wu; Mu Xu
>
> **摘要:** Embodied navigation for long-horizon tasks, guided by complex natural language instructions, remains a formidable challenge in artificial intelligence. Existing agents often struggle with robust long-term planning about unseen environments, leading to high failure rates. To address these limitations, we introduce NavForesee, a novel Vision-Language Model (VLM) that unifies high-level language planning and predictive world model imagination within a single, unified framework. Our approach empowers a single VLM to concurrently perform planning and predictive foresight. Conditioned on the full instruction and historical observations, the model is trained to understand the navigation instructions by decomposing the task, tracking its progress, and formulating the subsequent sub-goal. Simultaneously, it functions as a generative world model, providing crucial foresight by predicting short-term environmental dynamics and long-term navigation milestones. The VLM's structured plan guides its targeted prediction, while the imagined future provides rich context to inform the navigation actions, creating a powerful internal feedback loop of perception-planning/prediction-action. We demonstrate through extensive experiments on the R2R-CE and RxR-CE benchmark that NavForesee achieves highly competitive performance in complex scenarios. Our work highlights the immense potential of fusing explicit language planning with implicit spatiotemporal prediction, paving the way for more intelligent and capable embodied agents.
>
---
#### [new 253] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对具身智能中的视觉-运动策略学习问题，解决生成式策略数据效率低、采样慢的难题。提出EfficientFlow框架，通过引入等变性提升数据效率，并设计加速正则化策略，实现高效训练与快速推理，在有限数据下取得优异性能。**

- **链接: [https://arxiv.org/pdf/2512.02020v1](https://arxiv.org/pdf/2512.02020v1)**

> **作者:** Jianlei Chang; Ruofeng Mei; Wei Ke; Xiangyu Xu
>
> **备注:** Accepted by AAAI 2026. Project Page: https://efficientflow.github.io/
>
> **摘要:** Generative modeling has recently shown remarkable promise for visuomotor policy learning, enabling flexible and expressive control across diverse embodied AI tasks. However, existing generative policies often struggle with data inefficiency, requiring large-scale demonstrations, and sampling inefficiency, incurring slow action generation during inference. We introduce EfficientFlow, a unified framework for efficient embodied AI with flow-based policy learning. To enhance data efficiency, we bring equivariance into flow matching. We theoretically prove that when using an isotropic Gaussian prior and an equivariant velocity prediction network, the resulting action distribution remains equivariant, leading to improved generalization and substantially reduced data demands. To accelerate sampling, we propose a novel acceleration regularization strategy. As direct computation of acceleration is intractable for marginal flow trajectories, we derive a novel surrogate loss that enables stable and scalable training using only conditional trajectories. Across a wide range of robotic manipulation benchmarks, the proposed algorithm achieves competitive or superior performance under limited data while offering dramatically faster inference. These results highlight EfficientFlow as a powerful and efficient paradigm for high-performance embodied AI.
>
---
#### [new 254] Coarse-to-Fine Non-Rigid Registration for Side-Scan Sonar Mosaicking
- **分类: physics.geo-ph; cs.CV**

- **简介: 该论文针对侧扫声呐拼接中的非刚性形变问题，提出一种粗到精的分层非刚性配准框架。通过薄板样条初始化、超像素分割与预训练SynthMorph网络，实现全局与局部变形的精准融合，有效解决复杂形变建模与稀疏纹理下的过拟合问题，显著提升拼接精度与平滑性。**

- **链接: [https://arxiv.org/pdf/2512.00052v1](https://arxiv.org/pdf/2512.00052v1)**

> **作者:** Can Lei; Nuno Gracias; Rafael Garcia; Hayat Rajani; Huigang Wang
>
> **摘要:** Side-scan sonar mosaicking plays a crucial role in large-scale seabed mapping but is challenged by complex non-linear, spatially varying distortions due to diverse sonar acquisition conditions. Existing rigid or affine registration methods fail to model such complex deformations, whereas traditional non-rigid techniques tend to overfit and lack robustness in sparse-texture sonar data. To address these challenges, we propose a coarse-to-fine hierarchical non-rigid registration framework tailored for large-scale side-scan sonar images. Our method begins with a global Thin Plate Spline initialization from sparse correspondences, followed by superpixel-guided segmentation that partitions the image into structurally consistent patches preserving terrain integrity. Each patch is then refined by a pretrained SynthMorph network in an unsupervised manner, enabling dense and flexible alignment without task-specific training. Finally, a fusion strategy integrates both global and local deformations into a smooth, unified deformation field. Extensive quantitative and visual evaluations demonstrate that our approach significantly outperforms state-of-the-art rigid, classical non-rigid, and learning-based methods in accuracy, structural consistency, and deformation smoothness on the challenging sonar dataset.
>
---
#### [new 255] Forget Less, Retain More: A Lightweight Regularizer for Rehearsal-Based Continual Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对持续学习中的灾难性遗忘问题，提出一种轻量级信息最大化（IM）正则化器。它基于预期标签分布，无需类别信息，可无缝集成到各类回放式持续学习方法中，有效减少遗忘、加速收敛，且在图像与视频数据上均表现优异，具有高效、通用的特性。**

- **链接: [https://arxiv.org/pdf/2512.01818v1](https://arxiv.org/pdf/2512.01818v1)**

> **作者:** Lama Alssum; Hasan Abed Al Kader Hammoud; Motasem Alfarra; Juan C Leon Alcazar; Bernard Ghanem
>
> **摘要:** Deep neural networks suffer from catastrophic forgetting, where performance on previous tasks degrades after training on a new task. This issue arises due to the model's tendency to overwrite previously acquired knowledge with new information. We present a novel approach to address this challenge, focusing on the intersection of memory-based methods and regularization approaches. We formulate a regularization strategy, termed Information Maximization (IM) regularizer, for memory-based continual learning methods, which is based exclusively on the expected label distribution, thus making it class-agnostic. As a consequence, IM regularizer can be directly integrated into various rehearsal-based continual learning methods, reducing forgetting and favoring faster convergence. Our empirical validation shows that, across datasets and regardless of the number of tasks, our proposed regularization strategy consistently improves baseline performance at the expense of a minimal computational overhead. The lightweight nature of IM ensures that it remains a practical and scalable solution, making it applicable to real-world continual learning scenarios where efficiency is paramount. Finally, we demonstrate the data-agnostic nature of our regularizer by applying it to video data, which presents additional challenges due to its temporal structure and higher memory requirements. Despite the significant domain gap, our experiments show that IM regularizer also improves the performance of video continual learning methods.
>
---
#### [new 256] VISTAv2: World Imagination for Indoor Vision-and-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉语言导航（VLN）任务，解决现有方法缺乏在线动作条件预测与显式规划价值的问题。提出VISTAv2，通过条件扩散Transformer生成动作相关的未来视图，结合指令引导融合为在线价值图，提升导航的鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.00041v1](https://arxiv.org/pdf/2512.00041v1)**

> **作者:** Yanjia Huang; Xianshun Jiang; Xiangbo Gao; Mingyang Wu; Zhengzhong Tu
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to follow language instructions while acting in continuous real-world spaces. Prior image imagination based VLN work shows benefits for discrete panoramas but lacks online, action-conditioned predictions and does not produce explicit planning values; moreover, many methods replace the planner with long-horizon objectives that are brittle and slow. To bridge this gap, we propose VISTAv2, a generative world model that rolls out egocentric future views conditioned on past observations, candidate action sequences, and instructions, and projects them into an online value map for planning. Unlike prior approaches, VISTAv2 does not replace the planner. The online value map is fused at score level with the base objective, providing reachability and risk-aware guidance. Concretely, we employ an action-aware Conditional Diffusion Transformer video predictor to synthesize short-horizon futures, align them with the natural language instruction via a vision-language scorer, and fuse multiple rollouts in a differentiable imagination-to-value head to output an imagined egocentric value map. For efficiency, rollouts occur in VAE latent space with a distilled sampler and sparse decoding, enabling inference on a single consumer GPU. Evaluated on MP3D and RoboTHOR, VISTAv2 improves over strong baselines, and ablations show that action-conditioned imagination, instruction-guided value fusion, and the online value-map planner are all critical, suggesting that VISTAv2 offers a practical and interpretable route to robust VLN.
>
---
#### [new 257] MILE: A Mechanically Isomorphic Exoskeleton Data Collection System with Fingertip Visuotactile Sensing for Dexterous Manipulation
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文针对灵巧操作中高质量数据缺失问题，提出MILE系统，实现人体手与机器人手的机械同构，通过无畸变运动映射和高分辨率指尖触觉视觉传感，高效采集多模态数据，显著提升操作成功率，推动模仿学习在精细操作中的应用。**

- **链接: [https://arxiv.org/pdf/2512.00324v1](https://arxiv.org/pdf/2512.00324v1)**

> **作者:** Jinda Du; Jieji Ren; Qiaojun Yu; Ningbin Zhang; Yu Deng; Xingyu Wei; Yufei Liu; Guoying Gu; Xiangyang Zhu
>
> **摘要:** Imitation learning provides a promising approach to dexterous hand manipulation, but its effectiveness is limited by the lack of large-scale, high-fidelity data. Existing data-collection pipelines suffer from inaccurate motion retargeting, low data-collection efficiency, and missing high-resolution fingertip tactile sensing. We address this gap with MILE, a mechanically isomorphic teleoperation and data-collection system co-designed from human hand to exoskeleton to robotic hand. The exoskeleton is anthropometrically derived from the human hand, and the robotic hand preserves one-to-one joint-position isomorphism, eliminating nonlinear retargeting and enabling precise, natural control. The exoskeleton achieves a multi-joint mean absolute angular error below one degree, while the robotic hand integrates compact fingertip visuotactile modules that provide high-resolution tactile observations. Built on this retargeting-free interface, we teleoperate complex, contact-rich in-hand manipulation and efficiently collect a multimodal dataset comprising high-resolution fingertip visuotactile signals, RGB-D images, and joint positions. The teleoperation pipeline achieves a mean success rate improvement of 64%. Incorporating fingertip tactile observations further increases the success rate by an average of 25% over the vision-only baseline, validating the fidelity and utility of the dataset. Further details are available at: https://sites.google.com/view/mile-system.
>
---
#### [new 258] FOM-Nav: Frontier-Object Maps for Object Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对未知环境中寻物导航任务，解决现有方法在长期记忆与语义信息不足的问题。提出FOM-Nav框架，通过在线构建融合空间前沿与细粒度物体信息的前沿-物体地图，结合视觉语言模型实现高层目标预测与高效路径规划，显著提升导航效率，在多个基准上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2512.01009v1](https://arxiv.org/pdf/2512.01009v1)**

> **作者:** Thomas Chabal; Shizhe Chen; Jean Ponce; Cordelia Schmid
>
> **备注:** Project page: https://www.di.ens.fr/willow/research/fom-nav/
>
> **摘要:** This paper addresses the Object Goal Navigation problem, where a robot must efficiently find a target object in an unknown environment. Existing implicit memory-based methods struggle with long-term memory retention and planning, while explicit map-based approaches lack rich semantic information. To address these challenges, we propose FOM-Nav, a modular framework that enhances exploration efficiency through Frontier-Object Maps and vision-language models. Our Frontier-Object Maps are built online and jointly encode spatial frontiers and fine-grained object information. Using this representation, a vision-language model performs multimodal scene understanding and high-level goal prediction, which is executed by a low-level planner for efficient trajectory generation. To train FOM-Nav, we automatically construct large-scale navigation datasets from real-world scanned environments. Extensive experiments validate the effectiveness of our model design and constructed dataset. FOM-Nav achieves state-of-the-art performance on the MP3D and HM3D benchmarks, particularly in navigation efficiency metric SPL, and yields promising results on a real robot.
>
---
#### [new 259] HMARK: Radioactive Multi-Bit Semantic-Latent Watermarking for Diffusion Models
- **分类: cs.CR; cs.CV**

- **简介: 该论文提出HMARK，一种用于扩散模型的多比特语义-隐空间放射性水印方案。针对生成模型训练中版权数据滥用问题，通过在语义隐空间嵌入不可见且鲁棒的水印，实现所有权追踪。实验表明其具备高检测精度与抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2512.00094v1](https://arxiv.org/pdf/2512.00094v1)**

> **作者:** Kexin Li; Guozhen Ding; Ilya Grishchenko; David Lie
>
> **摘要:** Modern generative diffusion models rely on vast training datasets, often including images with uncertain ownership or usage rights. Radioactive watermarks -- marks that transfer to a model's outputs -- can help detect when such unauthorized data has been used for training. Moreover, aside from being radioactive, an effective watermark for protecting images from unauthorized training also needs to meet other existing requirements, such as imperceptibility, robustness, and multi-bit capacity. To overcome these challenges, we propose HMARK, a novel multi-bit watermarking scheme, which encodes ownership information as secret bits in the semantic-latent space (h-space) for image diffusion models. By leveraging the interpretability and semantic significance of h-space, ensuring that watermark signals correspond to meaningful semantic attributes, the watermarks embedded by HMARK exhibit radioactivity, robustness to distortions, and minimal impact on perceptual quality. Experimental results demonstrate that HMARK achieves 98.57% watermark detection accuracy, 95.07% bit-level recovery accuracy, 100% recall rate, and 1.0 AUC on images produced by the downstream adversarial model finetuned with LoRA on watermarked data across various types of distortions.
>
---
#### [new 260] Sign Language Recognition using Bidirectional Reservoir Computing
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手语识别任务，旨在解决深度学习模型计算资源消耗大、不适用于边缘设备的问题。提出基于MediaPipe与双向储备池计算（BRC）的高效识别方法，利用手部关节点特征，通过双向处理捕捉时序依赖，实现9秒训练时间与57.71%准确率，显著优于传统深度学习方法。**

- **链接: [https://arxiv.org/pdf/2512.00777v1](https://arxiv.org/pdf/2512.00777v1)**

> **作者:** Nitin Kumar Singh; Arie Rachmad Syulistyo; Yuichiro Tanaka; Hakaru Tamukoh
>
> **摘要:** Sign language recognition (SLR) facilitates communication between deaf and hearing individuals. Deep learning is widely used to develop SLR-based systems; however, it is computationally intensive and requires substantial computational resources, making it unsuitable for resource-constrained devices. To address this, we propose an efficient sign language recognition system using MediaPipe and an echo state network (ESN)-based bidirectional reservoir computing (BRC) architecture. MediaPipe extracts hand joint coordinates, which serve as inputs to the ESN-based BRC architecture. The BRC processes these features in both forward and backward directions, efficiently capturing temporal dependencies. The resulting states of BRC are concatenated to form a robust representation for classification. We evaluated our method on the Word-Level American Sign Language (WLASL) video dataset, achieving a competitive accuracy of 57.71% and a significantly lower training time of only 9 seconds, in contrast to the 55 minutes and $38$ seconds required by the deep learning-based Bi-GRU approach. Consequently, the BRC-based SLR system is well-suited for edge devices.
>
---
#### [new 261] Estimation of Kinematic Motion from Dashcam Footage
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究基于行车记录仪视频估计车辆运动参数的任务，旨在评估其预测车速、航向角及前车相对位置与速度的准确性。通过同步车载CAN数据与摄像头视频，构建神经网络模型，并提供开源工具与方法供他人复现数据采集与实验。**

- **链接: [https://arxiv.org/pdf/2512.01104v1](https://arxiv.org/pdf/2512.01104v1)**

> **作者:** Evelyn Zhang; Alex Richardson; Jonathan Sprinkle
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** The goal of this paper is to explore the accuracy of dashcam footage to predict the actual kinematic motion of a car-like vehicle. Our approach uses ground truth information from the vehicle's on-board data stream, through the controller area network, and a time-synchronized dashboard camera, mounted to a consumer-grade vehicle, for 18 hours of footage and driving. The contributions of the paper include neural network models that allow us to quantify the accuracy of predicting the vehicle speed and yaw, as well as the presence of a lead vehicle, and its relative distance and speed. In addition, the paper describes how other researchers can gather their own data to perform similar experiments, using open-source tools and off-the-shelf technology.
>
---
#### [new 262] Time-Series at the Edge: Tiny Separable CNNs for Wearable Gait Detection and Optimal Sensor Placement
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **简介: 该论文研究可穿戴设备上帕金森病步态检测任务，针对资源受限边缘设备，提出超轻量1D可分离CNN模型，替代传统阈值法，在极低参数量下实现高精度步态识别，优化传感器位置选择，满足实时性与能效要求。**

- **链接: [https://arxiv.org/pdf/2512.00396v1](https://arxiv.org/pdf/2512.00396v1)**

> **作者:** Andrea Procopio; Marco Esposito; Sara Raggiunto; Andrey Gizdov; Alberto Belli; Paola Pierleoni
>
> **摘要:** We study on-device time-series analysis for gait detection in Parkinson's disease (PD) from short windows of triaxial acceleration, targeting resource-constrained wearables and edge nodes. We compare magnitude thresholding to three 1D CNNs for time-series analysis: a literature baseline (separable convolutions) and two ultra-light models - one purely separable and one with residual connections. Using the BioStampRC21 dataset, 2 s windows at 30 Hz, and subject-independent leave-one-subject-out (LOSO) validation on 16 PwPD with chest-worn IMUs, our residual separable model (Model 2, 533 params) attains PR-AUC = 94.5%, F1 = 91.2%, MCC = 89.4%, matching or surpassing the baseline (5,552 params; PR-AUC = 93.7%, F1 = 90.5%, MCC = 88.5%) with approximately 10x fewer parameters. The smallest model (Model 1, 305 params) reaches PR-AUC = 94.0%, F1 = 91.0%, MCC = 89.1%. Thresholding obtains high recall (89.0%) but low precision (76.5%), yielding many false positives and high inter-subject variance. Sensor-position analysis (train-on-all) shows chest and thighs are most reliable; forearms degrade precision/recall due to non-gait arm motion; naive fusion of all sites does not outperform the best single site. Both compact CNNs execute within tight memory/latency budgets on STM32-class MCUs (sub-10 ms on low-power boards), enabling on-sensor gating of transmission/storage. Overall, ultra-light separable CNNs provide a superior accuracy-efficiency-generalization trade-off to fixed thresholds for wearable PD gait detection and underscore the value of tailored time-series models for edge deployment.
>
---
#### [new 263] A Comprehensive Survey on Surgical Digital Twin
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文综述手术数字孪生（SDT）技术，聚焦多模态数据融合、实时计算与临床落地挑战。提出分类体系，梳理核心进展，对比架构设计，揭示验证、安全、数据治理等关键问题，提出可信、标准化的SDT研究路线，推动从实验室走向临床应用。**

- **链接: [https://arxiv.org/pdf/2512.00019v1](https://arxiv.org/pdf/2512.00019v1)**

> **作者:** Afsah Sharaf Khan; Falong Fan; Doohwan DH Kim; Abdurrahman Alshareef; Dong Chen; Justin Kim; Ernest Carter; Bo Liu; Jerzy W. Rozenblit; Bernard Zeigler
>
> **摘要:** With the accelerating availability of multimodal surgical data and real-time computation, Surgical Digital Twins (SDTs) have emerged as virtual counterparts that mirror, predict, and inform decisions across pre-, intra-, and postoperative care. Despite promising demonstrations, SDTs face persistent challenges: fusing heterogeneous imaging, kinematics, and physiology under strict latency budgets; balancing model fidelity with computational efficiency; ensuring robustness, interpretability, and calibrated uncertainty; and achieving interoperability, privacy, and regulatory compliance in clinical environments. This survey offers a critical, structured review of SDTs. We clarify terminology and scope, propose a taxonomy by purpose, model fidelity, and data sources, and synthesize state-of-the-art achievements in deformable registration and tracking, real-time simulation and co-simulation, AR/VR guidance, edge-cloud orchestration, and AI for scene understanding and prediction. We contrast non-robotic twins with robot-in-the-loop architectures for shared control and autonomy, and identify open problems in validation and benchmarking, safety assurance and human factors, lifecycle "digital thread" integration, and scalable data governance. We conclude with a research agenda toward trustworthy, standards-aligned SDTs that deliver measurable clinical benefit. By unifying vocabulary, organizing capabilities, and highlighting gaps, this work aims to guide SDT design and deployment and catalyze translation from laboratory prototypes to routine surgical care.
>
---
#### [new 264] Revisiting Direct Encoding: Learnable Temporal Dynamics for Static Image Spiking Neural Networks
- **分类: cs.NE; cs.CV**

- **简介: 该论文研究静态图像输入下脉冲神经网络（SNN）的时序建模问题。针对直接编码因重复输入导致时序信息退化的问题，提出可学习的时序编码机制，通过自适应相位偏移引入有效时序变化，揭示性能差异源于卷积可学习性与梯度替代而非编码方式本身。**

- **链接: [https://arxiv.org/pdf/2512.01687v1](https://arxiv.org/pdf/2512.01687v1)**

> **作者:** Huaxu He
>
> **摘要:** Handling static images that lack inherent temporal dynamics remains a fundamental challenge for spiking neural networks (SNNs). In directly trained SNNs, static inputs are typically repeated across time steps, causing the temporal dimension to collapse into a rate like representation and preventing meaningful temporal modeling. This work revisits the reported performance gap between direct and rate based encodings and shows that it primarily stems from convolutional learnability and surrogate gradient formulations rather than the encoding schemes themselves. To illustrate this mechanism level clarification, we introduce a minimal learnable temporal encoding that adds adaptive phase shifts to induce meaningful temporal variation from static inputs.
>
---
#### [new 265] Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics
- **分类: hep-ex; cs.CV**

- **简介: 该论文针对高能物理中液氩时间投影室（LArTPC）的粒子重建任务，提出Panda模型。它通过自蒸馏学习可复用的传感器级表征，无需大量标注数据，显著提升标签效率与重建质量，仅用千分之一标签即超越此前最先进方法，并实现高性能粒子识别。**

- **链接: [https://arxiv.org/pdf/2512.01324v1](https://arxiv.org/pdf/2512.01324v1)**

> **作者:** Samuel Young; Kazuhiro Terao
>
> **备注:** 23 pages, 15 figures, preprint. Project page at https://youngsm.com/panda/
>
> **摘要:** Liquid argon time projection chambers (LArTPCs) provide dense, high-fidelity 3D measurements of particle interactions and underpin current and future neutrino and rare-event experiments. Physics reconstruction typically relies on complex detector-specific pipelines that use tens of hand-engineered pattern recognition algorithms or cascades of task-specific neural networks that require extensive, labeled simulation that requires a careful, time-consuming calibration process. We introduce \textbf{Panda}, a model that learns reusable sensor-level representations directly from raw unlabeled LArTPC data. Panda couples a hierarchical sparse 3D encoder with a multi-view, prototype-based self-distillation objective. On a simulated dataset, Panda substantially improves label efficiency and reconstruction quality, beating the previous state-of-the-art semantic segmentation model with 1,000$\times$ fewer labels. We also show that a single set-prediction head 1/20th the size of the backbone with no physical priors trained on frozen outputs from Panda can result in particle identification that is comparable with state-of-the-art (SOTA) reconstruction tools. Full fine-tuning further improves performance across all tasks.
>
---
#### [new 266] Disentangling Progress in Medical Image Registration: Beyond Trend-Driven Architectures towards Domain-Specific Strategies
- **分类: eess.IV; cs.CV**

- **简介: 该论文研究学习型医学图像配准任务，旨在厘清通用架构趋势与领域特定设计的贡献。通过模块化框架对比分析，发现领域先验设计显著优于流行计算模块，推动性能提升。研究提出可插拔基准平台，倡导聚焦领域知识而非盲目追随架构潮流。**

- **链接: [https://arxiv.org/pdf/2512.01913v1](https://arxiv.org/pdf/2512.01913v1)**

> **作者:** Bailiang Jian; Jiazhen Pan; Rohit Jena; Morteza Ghahremani; Hongwei Bran Li; Daniel Rueckert; Christian Wachinger; Benedikt Wiestler
>
> **备注:** Submitted to Medical Image Analysis. Journal Extension of arXiv:2407.19274
>
> **摘要:** Medical image registration drives quantitative analysis across organs, modalities, and patient populations. Recent deep learning methods often combine low-level "trend-driven" computational blocks from computer vision, such as large-kernel CNNs, Transformers, and state-space models, with high-level registration-specific designs like motion pyramids, correlation layers, and iterative refinement. Yet, their relative contributions remain unclear and entangled. This raises a central question: should future advances in registration focus on importing generic architectural trends or on refining domain-specific design principles? Through a modular framework spanning brain, lung, cardiac, and abdominal registration, we systematically disentangle the influence of these two paradigms. Our evaluation reveals that low-level "trend-driven" computational blocks offer only marginal or inconsistent gains, while high-level registration-specific designs consistently deliver more accurate, smoother, and more robust deformations. These domain priors significantly elevate the performance of a standard U-Net baseline, far more than variants incorporating "trend-driven" blocks, achieving an average relative improvement of $\sim3\%$. All models and experiments are released within a transparent, modular benchmark that enables plug-and-play comparison for new architectures and registration tasks (https://github.com/BailiangJ/rethink-reg). This dynamic and extensible platform establishes a common ground for reproducible and fair evaluation, inviting the community to isolate genuine methodological contributions from domain priors. Our findings advocate a shift in research emphasis: from following architectural trends to embracing domain-specific design principles as the true drivers of progress in learning-based medical image registration.
>
---
#### [new 267] Foundation Models for Trajectory Planning in Autonomous Driving: A Review of Progress and Open Challenges
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦自动驾驶中的轨迹规划任务，针对传统方法依赖手工设计的局限，综述基于多模态基础模型的新范式。系统梳理37种方法，提出统一分类框架，分析其架构、优势与挑战，并评估开源情况，为研究者提供参考。**

- **链接: [https://arxiv.org/pdf/2512.00021v1](https://arxiv.org/pdf/2512.00021v1)**

> **作者:** Kemal Oksuz; Alexandru Buburuzan; Anthony Knittel; Yuhan Yao; Puneet K. Dokania
>
> **备注:** Under review
>
> **摘要:** The emergence of multi-modal foundation models has markedly transformed the technology for autonomous driving, shifting away from conventional and mostly hand-crafted design choices towards unified, foundation-model-based approaches, capable of directly inferring motion trajectories from raw sensory inputs. This new class of methods can also incorporate natural language as an additional modality, with Vision-Language-Action (VLA) models serving as a representative example. In this review, we provide a comprehensive examination of such methods through a unifying taxonomy to critically evaluate their architectural design choices, methodological strengths, and their inherent capabilities and limitations. Our survey covers 37 recently proposed approaches that span the landscape of trajectory planning with foundation models. Furthermore, we assess these approaches with respect to the openness of their source code and datasets, offering valuable information to practitioners and researchers. We provide an accompanying webpage that catalogs the methods based on our taxonomy, available at: https://github.com/fiveai/FMs-for-driving-trajectories
>
---
#### [new 268] MoLT: Mixture of Layer-Wise Tokens for Efficient Audio-Visual Learning
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文针对音频-视觉学习中的参数与内存效率问题，提出MoLT框架。通过在深层网络中并行提取和融合层间令牌，实现轻量化适配，有效避免早期层误差传播，提升性能。实验表明，MoLT在多个跨模态任务上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00115v1](https://arxiv.org/pdf/2512.00115v1)**

> **作者:** Kyeongha Rho; Hyeongkeun Lee; Jae Won Cho; Joon Son Chung
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** In this paper, we propose Mixture of Layer-Wise Tokens (MoLT), a parameter- and memory-efficient adaptation framework for audio-visual learning. The key idea of MoLT is to replace conventional, computationally heavy sequential adaptation at every transformer layer with a parallel, lightweight scheme that extracts and fuses layer-wise tokens only from the late layers. We adopt two types of adapters to distill modality-specific information and cross-modal interaction into compact latent tokens in a layer-wise manner. A token fusion module then dynamically fuses these layer-wise tokens by taking into account their relative significance. To prevent the redundancy of latent tokens, we apply an orthogonality regularization between latent tokens during training. Through the systematic analysis of the position of adaptation in the pre-trained transformers, we extract latent tokens only from the late layers of the transformers. This strategic adaptation approach avoids error propagation from the volatile early-layer features, thereby maximizing the adaptation performance while maintaining parameter and memory efficiency. Through extensive experiments, we demonstrate that MoLT outperforms existing methods on diverse audio-visual benchmarks, including Audio-Visual Question Answering, Audio-Visual Segmentation, and Audio-Visual Event Localization.
>
---
#### [new 269] Stay Unique, Stay Efficient: Preserving Model Personality in Multi-Task Merging
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对多任务模型合并中任务特异性信息丢失导致性能下降的问题，提出DTS框架。通过奇异值分解与分组阈值缩放，有效保留任务特征，仅需1%额外存储。扩展版本实现无数据融合，提升未见任务泛化能力。实验表明其优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.01461v1](https://arxiv.org/pdf/2512.01461v1)**

> **作者:** Kuangpu Guo; Yuhe Ding; Jian Liang; Zilei Wang; Ran He
>
> **摘要:** Model merging has emerged as a promising paradigm for enabling multi-task capabilities without additional training. However, existing methods often experience substantial performance degradation compared with individually fine-tuned models, even on similar tasks, underscoring the need to preserve task-specific information. This paper proposes Decomposition, Thresholding, and Scaling (DTS), an approximation-based personalized merging framework that preserves task-specific information with minimal storage overhead. DTS first applies singular value decomposition to the task-specific information and retains only a small subset of singular values and vectors. It then introduces a novel thresholding strategy that partitions singular vector elements into groups and assigns a scaling factor to each group. To enable generalization to unseen tasks, we further extend DTS with a variant that fuses task-specific information in a data-free manner based on the semantic similarity of task characteristics. Extensive experiments demonstrate that DTS consistently outperforms state-of-the-art baselines while requiring only 1\% additional storage per task. Furthermore, experiments on unseen tasks show that the DTS variant achieves significantly better generalization performance. Our code is available at https://github.com/krumpguo/DTS.
>
---
#### [new 270] REM: Evaluating LLM Embodied Spatial Reasoning through Multi-Frame Trajectories
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对多模态大模型在具身空间推理上的不足，提出REM基准，通过可控3D环境评估模型在多帧轨迹下的物体恒常性、空间关系与数量追踪能力。研究揭示当前模型在中等复杂度任务中可靠性下降，旨在推动具身空间理解能力的提升。**

- **链接: [https://arxiv.org/pdf/2512.00736v1](https://arxiv.org/pdf/2512.00736v1)**

> **作者:** Jacob Thompson; Emiliano Garcia-Lopez; Yonatan Bisk
>
> **摘要:** Humans build viewpoint-independent cognitive maps through navigation, enabling intuitive reasoning about object permanence and spatial relations. We argue that multimodal large language models (MLLMs), despite extensive video training, lack this fundamental spatial reasoning capability, a critical limitation for embodied applications. To demonstrate these limitations and drive research, we introduce REM (Reasoning over Embodied Multi-Frame Trajectories), a benchmark using controllable 3D environments for long-horizon embodied spatial reasoning. REM systematically evaluates key aspects like object permanence/distinction, spatial relationships, and numerical tracking across dynamic embodied viewpoints. Our evaluation shows that the best-performing current models exhibit promising overall performance, but become increasingly unreliable at even moderate complexity levels easily handled by humans. These findings highlight challenges MLLMs face in developing robust spatial representations from sequential visual input. Consequently, REM provides targeted metrics and diagnostics to foster improved spatial understanding in future models.
>
---
#### [new 271] Med-CMR: A Fine-Grained Benchmark Integrating Visual Evidence and Clinical Logic for Medical Complex Multimodal Reasoning
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出Med-CMR，一个用于评估医疗多模态大模型复杂推理能力的细粒度基准。针对现有评估缺乏系统性与临床真实性的问题，构建涵盖11个器官系统、12种影像模态的高质量VQA数据集，通过分解视觉理解与多步推理能力，揭示模型在罕见场景下的脆弱性，为医疗AI提供严谨的评测标准。**

- **链接: [https://arxiv.org/pdf/2512.00818v1](https://arxiv.org/pdf/2512.00818v1)**

> **作者:** Haozhen Gong; Xiaozhong Ji; Yuansen Liu; Wenbin Wu; Xiaoxiao Yan; Jingjing Liu; Kai Wu; Jiazhen Pan; Bailiang Jian; Jiangning Zhang; Xiaobin Hu; Hongwei Bran Li
>
> **摘要:** MLLMs MLLMs are beginning to appear in clinical workflows, but their ability to perform complex medical reasoning remains unclear. We present Med-CMR, a fine-grained Medical Complex Multimodal Reasoning benchmark. Med-CMR distinguishes from existing counterparts by three core features: 1) Systematic capability decomposition, splitting medical multimodal reasoning into fine-grained visual understanding and multi-step reasoning to enable targeted evaluation; 2) Challenging task design, with visual understanding across three key dimensions (small-object detection, fine-detail discrimination, spatial understanding) and reasoning covering four clinically relevant scenarios (temporal prediction, causal reasoning, long-tail generalization, multi-source integration); 3) Broad, high-quality data coverage, comprising 20,653 Visual Question Answering (VQA) pairs spanning 11 organ systems and 12 imaging modalities, validated via a rigorous two-stage (human expert + model-assisted) review to ensure clinical authenticity. We evaluate 18 state-of-the-art MLLMs with Med-CMR, revealing GPT-5 as the top-performing commercial model: 57.81 accuracy on multiple-choice questions (MCQs) and a 48.70 open-ended score, outperforming Gemini 2.5 Pro (49.87 MCQ accuracy, 45.98 open-ended score) and leading open-source model Qwen3-VL-235B-A22B (49.34 MCQ accuracy, 42.62 open-ended score). However, specialized medical MLLMs do not reliably outperform strong general models, and long-tail generalization emerges as the dominant failure mode. Med-CMR thus provides a stress test for visual-reasoning integration and rare-case robustness in medical MLLMs, and a rigorous yardstick for future clinical systems.
>
---
#### [new 272] MedCondDiff: Lightweight, Robust, Semantically Guided Diffusion for Medical Image Segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MedCondDiff，一种轻量、鲁棒的语义引导扩散模型，用于多器官医学图像分割。通过金字塔视觉变压器提取语义先验，指导去噪过程，提升分割精度与效率，显著降低推理时间和显存占用，适用于多模态医学图像。**

- **链接: [https://arxiv.org/pdf/2512.00350v1](https://arxiv.org/pdf/2512.00350v1)**

> **作者:** Ruirui Huang; Jiacheng Li
>
> **摘要:** We introduce MedCondDiff, a diffusion-based framework for multi-organ medical image segmentation that is efficient and anatomically grounded. The model conditions the denoising process on semantic priors extracted by a Pyramid Vision Transformer (PVT) backbone, yielding a semantically guided and lightweight diffusion architecture. This design improves robustness while reducing both inference time and VRAM usage compared to conventional diffusion models. Experiments on multi-organ, multi-modality datasets demonstrate that MedCondDiff delivers competitive performance across anatomical regions and imaging modalities, underscoring the potential of semantically guided diffusion models as an effective class of architectures for medical imaging tasks.
>
---
#### [new 273] TagSplat: Topology-Aware Gaussian Splatting for Dynamic Mesh Modeling and Tracking
- **分类: cs.GR; cs.CV**

- **简介: 该论文针对动态网格建模与跟踪任务，解决现有4D重建方法难以生成拓扑一致高质量网格的问题。提出基于高斯点阵的拓扑感知框架，通过显式编码空间连通性，实现拓扑一致的稀疏化与修剪，并结合时序正则化与可微网格光栅化，提升重建精度与跟踪能力。**

- **链接: [https://arxiv.org/pdf/2512.01329v1](https://arxiv.org/pdf/2512.01329v1)**

> **作者:** Hanzhi Guo; Dongdong Weng; Mo Su; Yixiao Chen; Xiaonuo Dongye; Chenyu Xu
>
> **摘要:** Topology-consistent dynamic model sequences are essential for applications such as animation and model editing. However, existing 4D reconstruction methods face challenges in generating high-quality topology-consistent meshes. To address this, we propose a topology-aware dynamic reconstruction framework based on Gaussian Splatting. We introduce a Gaussian topological structure that explicitly encodes spatial connectivity. This structure enables topology-aware densification and pruning, preserving the manifold consistency of the Gaussian representation. Temporal regularization terms further ensure topological coherence over time, while differentiable mesh rasterization improves mesh quality. Experimental results demonstrate that our method reconstructs topology-consistent mesh sequences with significantly higher accuracy than existing approaches. Moreover, the resulting meshes enable precise 3D keypoint tracking. Project page: https://haza628.github.io/tagSplat/
>
---
#### [new 274] Chain-of-Ground: Improving GUI Grounding via Iterative Reasoning and Reference Feedback
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对GUI接地任务中模型对微小或相似目标定位不准、现实布局模糊等问题，提出无需训练的多步迭代推理框架Chain-of-Ground。通过引导模型逐步反思与修正假设，提升定位精度与可解释性，在ScreenSpot Pro和TPanel UI数据集上显著优于基线，验证了结构化迭代优化的有效性。**

- **链接: [https://arxiv.org/pdf/2512.01979v1](https://arxiv.org/pdf/2512.01979v1)**

> **作者:** Aiden Yiliu Li; Bizhi Yu; Daoan Lei; Tianhe Ren; Shilong Liu
>
> **摘要:** GUI grounding aims to align natural language instructions with precise regions in complex user interfaces. Advanced multimodal large language models show strong ability in visual GUI grounding but still struggle with small or visually similar targets and ambiguity in real world layouts. These limitations arise from limited grounding capacity and from underuse of existing reasoning potential. We present Chain of Ground CoG a training free multi step grounding framework that uses multimodal large language models for iterative visual reasoning and refinement. Instead of direct prediction the model progressively reflects and adjusts its hypotheses leading to more accurate and interpretable localization. Our approach achieves 68.4 accuracy on the ScreenSpot Pro benchmark an improvement of 4.8 points. To measure real world generalization we introduce TPanel UI a dataset of 420 labeled industrial control panels with visual distortions such as blur and masking. On TPanel UI Chain of Ground improves over the strong baseline Qwen3 VL 235B by 6.9 points showing the effectiveness of multi step training free grounding across real world and digital interfaces. These results highlight a direction for unlocking grounding potential through structured iterative refinement instead of additional training.
>
---
#### [new 275] Open-Set Domain Adaptation Under Background Distribution Shift: Challenges and A Provably Efficient Solution
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对开放集域适应中的背景分布漂移问题，提出一种可证明高效的方法\ours{}。解决已知类别分布变化下新类识别的挑战，理论证明其在新类可分离条件下有效，并通过实证验证其在图像与文本数据上的优越性能，揭示了新类规模对效果的影响。**

- **链接: [https://arxiv.org/pdf/2512.01152v1](https://arxiv.org/pdf/2512.01152v1)**

> **作者:** Shravan Chaudhari; Yoav Wald; Suchi Saria
>
> **摘要:** As we deploy machine learning systems in the real world, a core challenge is to maintain a model that is performant even as the data shifts. Such shifts can take many forms: new classes may emerge that were absent during training, a problem known as open-set recognition, and the distribution of known categories may change. Guarantees on open-set recognition are mostly derived under the assumption that the distribution of known classes, which we call \emph{the background distribution}, is fixed. In this paper we develop \ours{}, a method that is guaranteed to solve open-set recognition even in the challenging case where the background distribution shifts. We prove that the method works under benign assumptions that the novel class is separable from the non-novel classes, and provide theoretical guarantees that it outperforms a representative baseline in a simplified overparameterized setting. We develop techniques to make \ours{} scalable and robust, and perform comprehensive empirical evaluations on image and text data. The results show that \ours{} significantly outperforms existing open-set recognition methods under background shift. Moreover, we provide new insights into how factors such as the size of the novel class influences performance, an aspect that has not been extensively explored in prior work.
>
---
#### [new 276] Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究扩散模型中的混合专家（Diffusion MoE）架构，旨在提升训练效率与性能。针对现有工作过度关注路由机制而忽视架构配置的问题，提出一系列关键设计因素，并通过系统实验验证其有效性，实现了在同等或更少激活参数下超越强基线的高效训练方案。**

- **链接: [https://arxiv.org/pdf/2512.01252v1](https://arxiv.org/pdf/2512.01252v1)**

> **作者:** Yahui Liu; Yang Yue; Jingyuan Zhang; Chenxi Sun; Yang Zhou; Wencong Zeng; Ruiming Tang; Guorui Zhou
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Recent efforts on Diffusion Mixture-of-Experts (MoE) models have primarily focused on developing more sophisticated routing mechanisms. However, we observe that the underlying architectural configuration space remains markedly under-explored. Inspired by the MoE design paradigms established in large language models (LLMs), we identify a set of crucial architectural factors for building effective Diffusion MoE models--including DeepSeek-style expert modules, alternative intermediate widths, varying expert counts, and enhanced attention positional encodings. Our systematic study reveals that carefully tuning these configurations is essential for unlocking the full potential of Diffusion MoE models, often yielding gains that exceed those achieved by routing innovations alone. Through extensive experiments, we present novel architectures that can be efficiently applied to both latent and pixel-space diffusion frameworks, which provide a practical and efficient training recipe that enables Diffusion MoE models to surpass strong baselines while using equal or fewer activated parameters. All code and models are publicly available at: https://github.com/yhlleo/EfficientMoE.
>
---
#### [new 277] SelfAI: Building a Self-Training AI System with LLM Agents
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出SelfAI，一个用于自主科学发现的多智能体系统，解决现有框架在领域泛化、实时交互和探索终止机制上的不足。通过用户代理、认知代理与实验管理器协同，实现高效超参数优化与人机协作，提升搜索效率与多样性，在多个领域验证了优越性。**

- **链接: [https://arxiv.org/pdf/2512.00403v1](https://arxiv.org/pdf/2512.00403v1)**

> **作者:** Xiao Wu; Ting-Zhu Huang; Liang-Jian Deng; Xiaobing Yu; Yu Zhong; Shangqi Deng; Ufaq Khan; Jianghao Wu; Xiaofeng Liu; Imran Razzak; Xiaojun Chang; Yutong Xie
>
> **摘要:** Recent work on autonomous scientific discovery has leveraged LLM-based agents to integrate problem specification, experiment planning, and execution into end-to-end systems. However, these frameworks are often confined to narrow application domains, offer limited real-time interaction with researchers, and lack principled mechanisms for determining when to halt exploration, resulting in inefficiencies, reproducibility challenges, and under-utilized human expertise. To address these gaps, we propose \textit{SelfAI}, a general multi-agent platform that combines a User Agent for translating high-level research objectives into standardized experimental configurations, a Cognitive Agent powered by LLMs with optimal stopping criteria to iteratively refine hyperparameter searches, and an Experiment Manager responsible for orchestrating parallel, fault-tolerant training workflows across heterogeneous hardware while maintaining a structured knowledge base for continuous feedback. We further introduce two novel evaluation metrics, Score and $\text{AUP}_D$, to quantify discovery efficiency and search diversity. Across regression, NLP, computer vision, scientific computing, medical imaging, and drug discovery benchmarks, SelfAI consistently achieves strong performance and reduces redundant trials compared to classical Bayesian optimization and LLM-based baselines, while enabling seamless interaction with human researchers.
>
---
#### [new 278] InnoGym: Benchmarking the Innovation Potential of AI Agents
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出InnoGym，首个系统评估AI代理创新潜力的基准与框架。针对现有评测仅关注正确性、忽视方法多样性的不足，引入性能提升与新颖性双指标，涵盖18个真实领域任务，并提供可复现的执行环境。实验揭示当前代理虽具创意但缺乏鲁棒性，凸显创新与有效性间的差距。**

- **链接: [https://arxiv.org/pdf/2512.01822v1](https://arxiv.org/pdf/2512.01822v1)**

> **作者:** Jintian Zhang; Kewei Xu; Jingsheng Zheng; Zhuoyun Yu; Yuqi Zhu; Yujie Luo; Lanning Wei; Shuofei Qiao; Lun Du; Da Zheng; Shumin Deng; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** LLMs and Agents have achieved impressive progress in code generation, mathematical reasoning, and scientific discovery. However, existing benchmarks primarily measure correctness, overlooking the diversity of methods behind solutions. True innovation depends not only on producing correct answers but also on the originality of the approach. We present InnoGym, the first benchmark and framework designed to systematically evaluate the innovation potential of AI agents. InnoGym introduces two complementary metrics: performance gain, which measures improvement over the best-known solutions, and novelty, which captures methodological differences from prior approaches. The benchmark includes 18 carefully curated tasks from real-world engineering and scientific domains, each standardized through resource filtering, evaluator validation, and solution collection. In addition, we provide iGym, a unified execution environment for reproducible and long-horizon evaluations. Extensive experiments show that while some agents produce novel approaches, their lack of robustness limits performance gains. These results highlight a key gap between creativity and effectiveness, underscoring the need for benchmarks that evaluate both.
>
---
#### [new 279] Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中故障检测难题，提出基于视觉语言模型的Guardian系统。通过自动合成失败数据，构建新基准，提升故障检测精度与泛化能力，显著改善机器人任务成功率。**

- **链接: [https://arxiv.org/pdf/2512.01946v1](https://arxiv.org/pdf/2512.01946v1)**

> **作者:** Paul Pacaud; Ricardo Garcia; Shizhe Chen; Cordelia Schmid
>
> **备注:** 9 pages, 9 figures, 6 tables
>
> **摘要:** Robust robotic manipulation requires reliable failure detection and recovery. Although current Vision-Language Models (VLMs) show promise, their accuracy and generalization are limited by the scarcity of failure data. To address this data gap, we propose an automatic robot failure synthesis approach that procedurally perturbs successful trajectories to generate diverse planning and execution failures. This method produces not only binary classification labels but also fine-grained failure categories and step-by-step reasoning traces in both simulation and the real world. With it, we construct three new failure detection benchmarks: RLBench-Fail, BridgeDataV2-Fail, and UR5-Fail, substantially expanding the diversity and scale of existing failure datasets. We then train Guardian, a VLM with multi-view images for detailed failure reasoning and detection. Guardian achieves state-of-the-art performance on both existing and newly introduced benchmarks. It also effectively improves task success rates when integrated into a state-of-the-art manipulation system in simulation and real robots, demonstrating the impact of our generated failure data.
>
---
## 更新

#### [replaced 001] CraftSVG: Multi-Object Text-to-SVG Synthesis via Layout Guided Diffusion
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2404.00412v3](https://arxiv.org/pdf/2404.00412v3)**

> **作者:** Ayan Banerjee; Nityanand Mathur; Josep Llados; Umapada Pal; Anjan Dutta
>
> **摘要:** Generating VectorArt from text prompts is a challenging vision task, requiring diverse yet realistic depictions of the seen as well as unseen entities. However, existing research has been mostly limited to the generation of single objects, rather than comprehensive scenes comprising multiple elements. In response, this work introduces SVGCraft, a novel end-to-end framework for the creation of vector graphics depicting entire scenes from textual descriptions. Utilizing a pre-trained LLM for layout generation from text prompts, this framework introduces a technique for producing masked latents in specified bounding boxes for accurate object placement. It introduces a fusion mechanism for integrating attention maps and employs a diffusion U-Net for coherent composition, speeding up the drawing process. The resulting SVG is optimized using a pre-trained encoder and LPIPS loss with opacity modulation to maximize similarity. Additionally, this work explores the potential of primitive shapes in facilitating canvas completion in constrained environments. Through both qualitative and quantitative assessments, SVGCraft is demonstrated to surpass prior works in abstraction, recognizability, and detail, as evidenced by its performance metrics (CLIP-T: 0.4563, Cosine Similarity: 0.6342, Confusion: 0.66, Aesthetic: 6.7832). The code will be available at https://github.com/ayanban011/SVGCraft.
>
---
#### [replaced 002] A Comprehensive Survey on World Models for Embodied AI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.16732v2](https://arxiv.org/pdf/2510.16732v2)**

> **作者:** Xinqing Li; Xin He; Le Zhang; Min Wu; Xiaoli Li; Yun Liu
>
> **备注:** https://github.com/Li-Zn-H/AwesomeWorldModels
>
> **摘要:** Embodied AI requires agents that perceive, act, and anticipate how actions reshape future world states. World models serve as internal simulators that capture environment dynamics, enabling forward and counterfactual rollouts to support perception, prediction, and decision making. This survey presents a unified framework for world models in embodied AI. Specifically, we formalize the problem setting and learning objectives, and propose a three-axis taxonomy encompassing: (1) Functionality, Decision-Coupled vs. General-Purpose; (2) Temporal Modeling, Sequential Simulation and Inference vs. Global Difference Prediction; (3) Spatial Representation, Global Latent Vector, Token Feature Sequence, Spatial Latent Grid, and Decomposed Rendering Representation. We systematize data resources and metrics across robotics, autonomous driving, and general video settings, covering pixel prediction quality, state-level understanding, and task performance. Furthermore, we offer a quantitative comparison of state-of-the-art models and distill key open challenges, including the scarcity of unified datasets and the need for evaluation metrics that assess physical consistency over pixel fidelity, the trade-off between model performance and the computational efficiency required for real-time control, and the core modeling difficulty of achieving long-horizon temporal consistency while mitigating error accumulation. Finally, we maintain a curated bibliography at https://github.com/Li-Zn-H/AwesomeWorldModels.
>
---
#### [replaced 003] Learning to Hear by Seeing: It's Time for Vision Language Models to Understand Artistic Emotion from Sight and Sound
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12077v2](https://arxiv.org/pdf/2511.12077v2)**

> **作者:** Dengming Zhang; Weitao You; Jingxiong Li; Weishen Lin; Wenda Shi; Xue Zhao; Heda Zuo; Junxian Wu; Lingyun Sun
>
> **摘要:** Emotion understanding is critical for making Large Language Models (LLMs) more general, reliable, and aligned with humans. Art conveys emotion through the joint design of visual and auditory elements, yet most prior work is human-centered or single-modality, overlooking the emotion intentionally expressed by the artwork. Meanwhile, current Audio-Visual Language Models (AVLMs) typically require large-scale audio pretraining to endow Visual Language Models (VLMs) with hearing, which limits scalability. We present Vision Anchored Audio-Visual Emotion LLM (VAEmotionLLM), a two-stage framework that teaches a VLM to hear by seeing with limited audio pretraining and to understand emotion across modalities. In Stage 1, Vision-Guided Audio Alignment (VG-Align) distills the frozen visual pathway into a new audio pathway by aligning next-token distributions of the shared LLM on synchronized audio-video clips, enabling hearing without a large audio dataset. In Stage 2, a lightweight Cross-Modal Emotion Adapter (EmoAdapter), composed of the Emotion Enhancer and the Emotion Supervisor, injects emotion-sensitive residuals and applies emotion supervision to enhance cross-modal emotion understanding. We also construct ArtEmoBenchmark, an art-centric emotion benchmark that evaluates content and emotion understanding under audio-only, visual-only, and audio-visual inputs. VAEmotionLLM achieves state-of-the-art results on ArtEmoBenchmark, outperforming audio-only, visual-only, and audio-visual baselines. Ablations show that the proposed components are complementary.
>
---
#### [replaced 004] 3D MedDiffusion: A 3D Medical Latent Diffusion Model for Controllable and High-quality Medical Image Generation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.13059v2](https://arxiv.org/pdf/2412.13059v2)**

> **作者:** Haoshen Wang; Zhentao Liu; Kaicong Sun; Xiaodong Wang; Dinggang Shen; Zhiming Cui
>
> **摘要:** The generation of medical images presents significant challenges due to their high-resolution and three-dimensional nature. Existing methods often yield suboptimal performance in generating high-quality 3D medical images, and there is currently no universal generative framework for medical imaging. In this paper, we introduce a 3D Medical Latent Diffusion (3D MedDiffusion) model for controllable, high-quality 3D medical image generation. 3D MedDiffusion incorporates a novel, highly efficient Patch-Volume Autoencoder that compresses medical images into latent space through patch-wise encoding and recovers back into image space through volume-wise decoding. Additionally, we design a new noise estimator to capture both local details and global structural information during diffusion denoising process. 3D MedDiffusion can generate fine-detailed, high-resolution images (up to 512x512x512) and effectively adapt to various downstream tasks as it is trained on large-scale datasets covering CT and MRI modalities and different anatomical regions (from head to leg). Experimental results demonstrate that 3D MedDiffusion surpasses state-of-the-art methods in generative quality and exhibits strong generalizability across tasks such as sparse-view CT reconstruction, fast MRI reconstruction, and data augmentation for segmentation and classification. Source code and checkpoints are available at https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion.
>
---
#### [replaced 005] LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10209v2](https://arxiv.org/pdf/2511.10209v2)**

> **作者:** Wenzhe He; Xiaojun Chen; Ruiqi Wang; Ruihui Li; Huilong Pi; Jiapeng Zhang; Zhuo Tang; Kenli Li
>
> **备注:** 18 pages, 13 figures, Accepted to AAAI 2026
>
> **摘要:** 3D LiDAR scene completion from point clouds is a fundamental component of perception systems in autonomous vehicles. Previous methods have predominantly employed diffusion models for high-fidelity reconstruction. However, their multi-step iterative sampling incurs significant computational overhead, limiting its real-time applicability. To address this, we propose LiNeXt-a lightweight, non-diffusion network optimized for rapid and accurate point cloud completion. Specifically, LiNeXt first applies the Noise-to-Coarse (N2C) Module to denoise the input noisy point cloud in a single pass, thereby obviating the multi-step iterative sampling of diffusion-based methods. The Refine Module then takes the coarse point cloud and its intermediate features from the N2C Module to perform more precise refinement, further enhancing structural completeness. Furthermore, we observe that LiDAR point clouds exhibit a distance-dependent spatial distribution, being densely sampled at proximal ranges and sparsely sampled at distal ranges. Accordingly, we propose the Distance-aware Selected Repeat strategy to generate a more uniformly distributed noisy point cloud. On the SemanticKITTI dataset, LiNeXt achieves a 199.8x speedup in inference, reduces Chamfer Distance by 50.7%, and uses only 6.1% of the parameters compared with LiDiff. These results demonstrate the superior efficiency and effectiveness of LiNeXt for real-time scene completion.
>
---
#### [replaced 006] Zero-shot Denoising via Neural Compression: Theoretical and algorithmic framework
- **分类: eess.IV; cs.CV; cs.IT**

- **链接: [https://arxiv.org/pdf/2506.12693v2](https://arxiv.org/pdf/2506.12693v2)**

> **作者:** Ali Zafari; Xi Chen; Shirin Jalali
>
> **摘要:** Zero-shot denoising aims to denoise observations without access to training samples or clean reference images. This setting is particularly relevant in practical imaging scenarios involving specialized domains such as medical imaging or biology. In this work, we propose the Zero-Shot Neural Compression Denoiser (ZS-NCD), a novel denoising framework based on neural compression. ZS-NCD treats a neural compression network as an untrained model, optimized directly on patches extracted from a single noisy image. The final reconstruction is then obtained by aggregating the outputs of the trained model over overlapping patches. Thanks to the built-in entropy constraints of compression architectures, our method naturally avoids overfitting and does not require manual regularization or early stopping. Through extensive experiments, we show that ZS-NCD achieves state-of-the-art performance among zero-shot denoisers for both Gaussian and Poisson noise, and generalizes well to both natural and non-natural images. Additionally, we provide new finite-sample theoretical results that characterize upper bounds on the achievable reconstruction error of general maximum-likelihood compression-based denoisers. These results further establish the theoretical foundations of compression-based denoising. Our code is available at: https://github.com/Computational-Imaging-RU/ZS-NCDenoiser.
>
---
#### [replaced 007] B2N3D: Progressive Learning from Binary to N-ary Relationships for 3D Object Grounding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.10194v2](https://arxiv.org/pdf/2510.10194v2)**

> **作者:** Feng Xiao; Hongbin Xu; Hai Ci; Wenxiong Kang
>
> **摘要:** Localizing 3D objects using natural language is essential for robotic scene understanding. The descriptions often involve multiple spatial relationships to distinguish similar objects, making 3D-language alignment difficult. Current methods only model relationships for pairwise objects, ignoring the global perceptual significance of n-ary combinations in multi-modal relational understanding. To address this, we propose a novel progressive relational learning framework for 3D object grounding. We extend relational learning from binary to n-ary to identify visual relations that match the referential description globally. Given the absence of specific annotations for referred objects in the training data, we design a grouped supervision loss to facilitate n-ary relational learning. In the scene graph created with n-ary relationships, we use a multi-modal network with hybrid attention mechanisms to further localize the target within the n-ary combinations. Experiments and ablation studies on the ReferIt3D and ScanRefer benchmarks demonstrate that our method outperforms the state-of-the-art, and proves the advantages of the n-ary relational perception in 3D localization.
>
---
#### [replaced 008] Ensuring Force Safety in Vision-Guided Robotic Manipulation via Implicit Tactile Calibration
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉引导机器人操作中因缺乏触觉感知导致的力安全问题，提出SafeDiff框架。通过隐式触觉校准实时优化状态规划，生成安全动作轨迹。构建了大规模仿真数据集SafeDoorManip50k，实验证明可有效降低开门时的有害力。**

- **链接: [https://arxiv.org/pdf/2412.10349v2](https://arxiv.org/pdf/2412.10349v2)**

> **作者:** Lai Wei; Jiahua Ma; Yibo Hu; Ruimao Zhang
>
> **备注:** Website URL: see https://i-am-future.github.io/safediff/
>
> **摘要:** In dynamic environments, robots often encounter constrained movement trajectories when manipulating objects with specific properties, such as doors. Therefore, applying the appropriate force is crucial to prevent damage to both the robots and the objects. However, current vision-guided robot state generation methods often falter in this regard, as they lack the integration of tactile perception. To tackle this issue, this paper introduces a novel state diffusion framework termed SafeDiff. It generates a prospective state sequence from the current robot state and visual context observation while incorporating real-time tactile feedback to refine the sequence. As far as we know, this is the first study specifically focused on ensuring force safety in robotic manipulation. It significantly enhances the rationality of state planning, and the safe action trajectory is derived from inverse dynamics based on this refined planning. In practice, unlike previous approaches that concatenate visual and tactile data to generate future robot state sequences, our method employs tactile data as a calibration signal to adjust the robot's state within the state space implicitly. Additionally, we've developed a large-scale simulation dataset called SafeDoorManip50k, offering extensive multimodal data to train and evaluate the proposed method. Extensive experiments show that our visual-tactile model substantially mitigates the risk of harmful forces in the door opening, across both simulated and real-world settings.
>
---
#### [replaced 009] MAMMA: Markerless & Automatic Multi-Person Motion Action Capture
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.13040v3](https://arxiv.org/pdf/2506.13040v3)**

> **作者:** Hanz Cuevas-Velasquez; Anastasios Yiannakidis; Soyong Shin; Giorgio Becherini; Markus Höschle; Joachim Tesch; Taylor Obersat; Tsvetelina Alexiadis; Eni Halilaj; Michael J. Black
>
> **摘要:** We present MAMMA, a markerless motion-capture pipeline that accurately recovers SMPL-X parameters from multi-view video of two-person interaction sequences. Traditional motion-capture systems rely on physical markers. Although they offer high accuracy, their requirements of specialized hardware, manual marker placement, and extensive post-processing make them costly and time-consuming. Recent learning-based methods attempt to overcome these limitations, but most are designed for single-person capture, rely on sparse keypoints, or struggle with occlusions and physical interactions. In this work, we introduce a method that predicts dense 2D contact-aware surface landmarks conditioned on segmentation masks, enabling person-specific correspondence estimation even under heavy occlusion. We employ a novel architecture that exploits learnable queries for each landmark. We demonstrate that our approach can handle complex person--person interaction and offers greater accuracy than existing methods. To train our network, we construct a large, synthetic multi-view dataset combining human motions from diverse sources, including extreme poses, hand motions, and close interactions. Our dataset yields high-variability synthetic sequences with rich body contact and occlusion, and includes SMPL-X ground-truth annotations with dense 2D landmarks. The result is a system capable of capturing human motion without the need for markers. Our approach offers competitive reconstruction quality compared to commercial marker-based motion-capture solutions, without the extensive manual cleanup. Finally, we address the absence of common benchmarks for dense-landmark prediction and markerless motion capture by introducing two evaluation settings built from real multi-view sequences. We will release our dataset, benchmark, method, training code, and pre-trained model weights for research purposes.
>
---
#### [replaced 010] PRIMA: Multi-Image Vision-Language Models for Reasoning Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2412.15209v2](https://arxiv.org/pdf/2412.15209v2)**

> **作者:** Muntasir Wahed; Kiet A. Nguyen; Adheesh Sunil Juvekar; Xinzhuo Li; Xiaona Zhou; Vedant Shah; Tianjiao Yu; Pinar Yanardag; Ismini Lourentzou
>
> **备注:** Project page: https://plan-lab.github.io/prima
>
> **摘要:** Despite significant advancements in Large Vision-Language Models (LVLMs)' capabilities, existing pixel-grounding models operate in single-image settings, limiting their ability to perform detailed, fine-grained comparisons across multiple images. Conversely, current multi-image understanding models lack pixel-level grounding. Our work addresses this gap by introducing the task of multi-image pixel-grounded reasoning alongside PRIMA, an LVLM that integrates pixel-level grounding with robust multi-image reasoning to produce contextually rich, pixel-grounded explanations. Central to PRIMA is SQuARE, a vision module that injects cross-image relational context into compact query-based visual tokens before fusing them with the language backbone. To support training and evaluation, we curate M4SEG, a new multi-image reasoning segmentation benchmark consisting of $\sim$744K question-answer pairs that require fine-grained visual understanding across multiple images. PRIMA outperforms state-of-the-art baselines with $7.83\%$ and $11.25\%$ improvements in Recall and S-IoU, respectively. Ablation studies further demonstrate the effectiveness of the proposed SQuARE module in capturing cross-image relationships.
>
---
#### [replaced 011] Multigranular Evaluation for Brain Visual Decoding
- **分类: cs.CV; cs.AI; eess.IV; q-bio.NC**

- **链接: [https://arxiv.org/pdf/2507.07993v2](https://arxiv.org/pdf/2507.07993v2)**

> **作者:** Weihao Xia; Cengiz Oztireli
>
> **备注:** AAAI 2026 (Oral). Code: https://github.com/weihaox/BASIC
>
> **摘要:** Existing evaluation protocols for brain visual decoding predominantly rely on coarse metrics that obscure inter-model differences, lack neuroscientific foundation, and fail to capture fine-grained visual distinctions. To address these limitations, we introduce BASIC, a unified, multigranular evaluation framework that jointly quantifies structural fidelity, inferential alignment, and contextual coherence between decoded and ground-truth images. For the structural level, we introduce a hierarchical suite of segmentation-based metrics, including foreground, semantic, instance, and component masks, anchored in granularity-aware correspondence across mask structures. For the semantic level, we extract structured scene representations encompassing objects, attributes, and relationships using multimodal large language models, enabling detailed, scalable, and context-rich comparisons with ground-truth stimuli. We benchmark a diverse set of visual decoding methods across multiple stimulus-neuroimaging datasets within this unified evaluation framework. Together, these criteria provide a more discriminative, interpretable, and comprehensive foundation for evaluating brain visual decoding methods.
>
---
#### [replaced 012] Rank Matters: Understanding and Defending Model Inversion Attacks via Low-Rank Feature Filtering
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2410.05814v3](https://arxiv.org/pdf/2410.05814v3)**

> **作者:** Hongyao Yu; Yixiang Qiu; Hao Fang; Tianqu Zhuang; Bin Chen; Sijin Yu; Bin Wang; Shu-Tao Xia; Ke Xu
>
> **备注:** KDD 2026 Accept
>
> **摘要:** Model Inversion Attacks (MIAs) pose a significant threat to data privacy by reconstructing sensitive training samples from the knowledge embedded in trained machine learning models. Despite recent progress in enhancing the effectiveness of MIAs across diverse settings, defense strategies have lagged behind -- struggling to balance model utility with robustness against increasingly sophisticated attacks. In this work, we propose the ideal inversion error to measure the privacy leakage, and our theoretical and empirical investigations reveals that higher-rank features are inherently more prone to privacy leakage. Motivated by this insight, we propose a lightweight and effective defense strategy based on low-rank feature filtering, which explicitly reduces the attack surface by constraining the dimension of intermediate representations. Extensive experiments across various model architectures and datasets demonstrate that our method consistently outperforms existing defenses, achieving state-of-the-art performance against a wide range of MIAs. Notably, our approach remains effective even in challenging regimes involving high-resolution data and high-capacity models, where prior defenses fail to provide adequate protection.
>
---
#### [replaced 013] Rethinking Efficient Mixture-of-Experts for Remote Sensing Modality-Missing Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11460v2](https://arxiv.org/pdf/2511.11460v2)**

> **作者:** Qinghao Gao; Jiahui Qu; Yunsong Li; Wenqian Dong
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Multimodal classification in remote sensing often suffers from missing modalities caused by environmental interference, sensor failures, or atmospheric effects, which severely degrade classification performance. Existing two-stage adaptation methods are computationally expensive and assume complete multimodal data during training, limiting their generalization to real-world incompleteness. To overcome these issues, we propose a Missing-aware Mixture-of-Loras (MaMOL) framework that reformulates modality missing as a multi-task learning problem. MaMOL introduces a dual-routing mechanism: a task-oriented dynamic router that adaptively activates experts for different missing patterns, and a modality-specific-shared static router that maintains stable cross-modal knowledge sharing. Unlike prior methods that train separate networks for each missing configuration, MaMOL achieves parameter-efficient adaptation via lightweight expert updates and shared expert reuse. Experiments on multiple remote sensing benchmarks demonstrate superior robustness and generalization under varying missing rates, with minimal computational overhead. Moreover, transfer experiments on natural image datasets validate its scalability and cross-domain applicability, highlighting MaMOL as a general and efficient solution for incomplete multimodal learning.
>
---
#### [replaced 014] GigaWorld-0: World Models as Data Engine to Empower Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GigaWorld-0，一个用于具身智能的统一世界模型数据引擎。针对真实交互数据稀缺与训练成本高的问题，构建视频与3D生成协同的合成数据系统，实现高保真、物理可信、可控的具身交互数据生成。通过高效训练框架支持大规模训练，使VLA模型在无真实数据参与下显著提升机器人任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19861v2](https://arxiv.org/pdf/2511.19861v2)**

> **作者:** GigaWorld Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jiagang Zhu; Kerui Li; Mengyuan Xu; Qiuping Deng; Siting Wang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yankai Wang; Yu Cao; Yifan Chang; Yuan Xu; Yun Ye; Yang Wang; Yukun Zhou; Zhengyuan Zhang; Zhehao Dong; Zheng Zhu
>
> **备注:** Project Page: https://giga-world-0.github.io/
>
> **摘要:** World models are emerging as a foundational paradigm for scalable, data-efficient embodied AI. In this work, we present GigaWorld-0, a unified world model framework designed explicitly as a data engine for Vision-Language-Action (VLA) learning. GigaWorld-0 integrates two synergistic components: GigaWorld-0-Video, which leverages large-scale video generation to produce diverse, texture-rich, and temporally coherent embodied sequences under fine-grained control of appearance, camera viewpoint, and action semantics; and GigaWorld-0-3D, which combines 3D generative modeling, 3D Gaussian Splatting reconstruction, physically differentiable system identification, and executable motion planning to ensure geometric consistency and physical realism. Their joint optimization enables the scalable synthesis of embodied interaction data that is visually compelling, spatially coherent, physically plausible, and instruction-aligned. Training at scale is made feasible through our efficient GigaTrain framework, which exploits FP8-precision and sparse attention to drastically reduce memory and compute requirements. We conduct comprehensive evaluations showing that GigaWorld-0 generates high-quality, diverse, and controllable data across multiple dimensions. Critically, VLA model (e.g., GigaBrain-0) trained on GigaWorld-0-generated data achieve strong real-world performance, significantly improving generalization and task success on physical robots without any real-world interaction during training.
>
---
#### [replaced 015] SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言-动作模型在机器人操作中依赖专家示范、奖励稀疏的问题，提出自参照策略优化（SRPO）。通过利用模型自身生成的成功轨迹作为参考，结合世界模型的潜在表示，为失败轨迹赋予渐进式奖励，实现高效无监督强化学习。在LIBERO基准上，200步内达99.2%成功率，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15605v2](https://arxiv.org/pdf/2511.15605v2)**

> **作者:** Senyu Fei; Siyin Wang; Li Ji; Ao Li; Shiduo Zhang; Liming Liu; Jinlong Hou; Jingjing Gong; Xianzhong Zhao; Xipeng Qiu
>
> **摘要:** Vision-Language-Action (VLA) models excel in robotic manipulation but are constrained by their heavy reliance on expert demonstrations, leading to demonstration bias and limiting performance. Reinforcement learning (RL) is a vital post-training strategy to overcome these limits, yet current VLA-RL methods, including group-based optimization approaches, are crippled by severe reward sparsity. Relying on binary success indicators wastes valuable information in failed trajectories, resulting in low training efficiency. To solve this, we propose Self-Referential Policy Optimization (SRPO), a novel VLA-RL framework. SRPO eliminates the need for external demonstrations or manual reward engineering by leveraging the model's own successful trajectories, generated within the current training batch, as a self-reference. This allows us to assign a progress-wise reward to failed attempts. A core innovation is the use of latent world representations to measure behavioral progress robustly. Instead of relying on raw pixels or requiring domain-specific fine-tuning, we utilize the compressed, transferable encodings from a world model's latent space. These representations naturally capture progress patterns across environments, enabling accurate, generalized trajectory comparison. Empirical evaluations on the LIBERO benchmark demonstrate SRPO's efficiency and effectiveness. Starting from a supervised baseline with 48.9% success, SRPO achieves a new state-of-the-art success rate of 99.2% in just 200 RL steps, representing a 103% relative improvement without any extra supervision. Furthermore, SRPO shows substantial robustness, achieving a 167% performance improvement on the LIBERO-Plus benchmark.
>
---
#### [replaced 016] Securing the Skies: A Comprehensive Survey on Anti-UAV Methods, Benchmarking, and Future Directions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文聚焦反无人机（anti-UAV）任务，针对无人机带来的安全威胁，系统综述了检测、分类与跟踪技术。研究涵盖多模态传感器与前沿方法，评估主流方案并指出实时性、隐蔽探测及集群应对等挑战，提出未来研究方向以推动智能防御体系发展。**

- **链接: [https://arxiv.org/pdf/2504.11967v3](https://arxiv.org/pdf/2504.11967v3)**

> **作者:** Yifei Dong; Fengyi Wu; Sanjian Zhang; Guangyu Chen; Yuzhi Hu; Masumi Yano; Jingdong Sun; Siyu Huang; Feng Liu; Qi Dai; Zhi-Qi Cheng
>
> **备注:** Best Paper, Accepted at CVPR Workshop Anti-UAV 2025. 16 pages
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are indispensable for infrastructure inspection, surveillance, and related tasks, yet they also introduce critical security challenges. This survey provides a wide-ranging examination of the anti-UAV domain, centering on three core objectives-classification, detection, and tracking-while detailing emerging methodologies such as diffusion-based data synthesis, multi-modal fusion, vision-language modeling, self-supervised learning, and reinforcement learning. We systematically evaluate state-of-the-art solutions across both single-modality and multi-sensor pipelines (spanning RGB, infrared, audio, radar, and RF) and discuss large-scale as well as adversarially oriented benchmarks. Our analysis reveals persistent gaps in real-time performance, stealth detection, and swarm-based scenarios, underscoring pressing needs for robust, adaptive anti-UAV systems. By highlighting open research directions, we aim to foster innovation and guide the development of next-generation defense strategies in an era marked by the extensive use of UAVs.
>
---
#### [replaced 017] PETAR: Localized Findings Generation with Mask-Aware Vision-Language Modeling for PET Automated Reporting
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.27680v2](https://arxiv.org/pdf/2510.27680v2)**

> **作者:** Danyal Maqbool; Changhee Lee; Zachary Huemann; Samuel D. Church; Matthew E. Larson; Scott B. Perlman; Tomas A. Romero; Joshua D. Warner; Meghan Lubner; Xin Tie; Jameson Merkow; Junjie Hu; Steve Y. Cho; Tyler J. Bradshaw
>
> **摘要:** Generating automated reports for 3D positron emission tomography (PET) is an important and challenging task in medical imaging. PET plays a vital role in oncology, but automating report generation is difficult due to the complexity of whole-body 3D volumes, the wide range of potential clinical findings, and the limited availability of annotated datasets. To address these challenges, we introduce PETARSeg-11K, the first large-scale, publicly available dataset that provides lesion-level correspondence between 3D PET/CT volumes and free-text radiological findings. It comprises 11,356 lesion descriptions paired with 3D segmentations. Second, we propose PETAR-4B, a 3D vision-language model designed for mask-aware, spatially grounded PET/CT reporting. PETAR-4B jointly encodes PET, CT, and 3D lesion segmentation masks, using a 3D focal prompt to capture fine-grained details of lesions that normally comprise less than 0.1% of the volume. Evaluations using automated metrics show PETAR-4B substantially outperforming all 2D and 3D baselines. A human study involving five physicians -- the first of its kind for automated PET reporting -- confirms the model's clinical utility and establishes correlations between automated metrics and expert judgment. This work provides a foundational dataset and a novel architecture, advancing 3D medical vision-language understanding in PET.
>
---
#### [replaced 018] Off the Planckian Locus: Using 2D Chromaticity to Improve In-Camera Color
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17133v2](https://arxiv.org/pdf/2511.17133v2)**

> **作者:** SaiKiran Tedla; Joshua E. Little; Hakki Can Karaimer; Michael S. Brown
>
> **备注:** Project page: https://cst-mlp.github.io
>
> **摘要:** Traditional in-camera colorimetric mapping relies on correlated color temperature (CCT)-based interpolation between pre-calibrated transforms optimized for Planckian illuminants such as CIE A and D65. However, modern lighting technologies such as LEDs can deviate substantially from the Planckian locus, exposing the limitations of relying on conventional one-dimensional CCT for illumination characterization. This paper demonstrates that transitioning from 1D CCT (on the Planckian locus) to a 2D chromaticity space (off the Planckian locus) improves colorimetric accuracy across various mapping approaches. In addition, we replace conventional CCT interpolation with a lightweight multi-layer perceptron (MLP) that leverages 2D chromaticity features for robust colorimetric mapping under non-Planckian illuminants. A lightbox-based calibration procedure incorporating representative LED sources is used to train our MLP. Validated across diverse LED lighting, our method reduces angular reproduction error by 22% on average in LED-lit scenes, maintains backward compatibility with traditional illuminants, accommodates multi-illuminant scenes, and supports real-time in-camera deployment with negligible additional computational cost.
>
---
#### [replaced 019] Rendering-Aware Reinforcement Learning for Vector Graphics Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.20793v2](https://arxiv.org/pdf/2505.20793v2)**

> **作者:** Juan A. Rodriguez; Haotian Zhang; Abhay Puri; Aarash Feizi; Rishav Pramanik; Pascal Wichmann; Arnab Mondal; Mohammad Reza Samsami; Rabiul Awal; Perouz Taslakian; Spandana Gella; Sai Rajeswar; David Vazquez; Christopher Pal; Marco Pedersoli
>
> **摘要:** Scalable Vector Graphics (SVG) offer a powerful format for representing visual designs as interpretable code. Recent advances in vision-language models (VLMs) have enabled high-quality SVG generation by framing the problem as a code generation task and leveraging large-scale pretraining. VLMs are particularly suitable for this task as they capture both global semantics and fine-grained visual patterns, while transferring knowledge across vision, natural language, and code domains. However, existing VLM approaches often struggle to produce faithful and efficient SVGs because they never observe the rendered images during training. Although differentiable rendering for autoregressive SVG code generation remains unavailable, rendered outputs can still be compared to original inputs, enabling evaluative feedback suitable for reinforcement learning (RL). We introduce RLRF (Reinforcement Learning from Rendering Feedback), an RL method that enhances SVG generation in autoregressive VLMs by leveraging feedback from rendered SVG outputs. Given an input image, the model generates SVG roll-outs that are rendered and compared to the original image to compute a reward. This visual fidelity feedback guides the model toward producing more accurate, efficient, and semantically coherent SVGs. RLRF significantly outperforms supervised fine-tuning, addressing common failure modes and enabling precise, high-quality SVG generation with strong structural understanding and generalization.
>
---
#### [replaced 020] Face-MakeUpV2: Facial Consistency Learning for Controllable Text-to-Image Generation
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [https://arxiv.org/pdf/2510.21775v2](https://arxiv.org/pdf/2510.21775v2)**

> **作者:** Dawei Dai; Yinxiu Zhou; Chenghang Li; Guolai Jiang; Chengfang Zhang
>
> **备注:** Some errors in the critical data presented in Table 1 and Table 2
>
> **摘要:** In facial image generation, current text-to-image models often suffer from facial attribute leakage and insufficient physical consistency when responding to local semantic instructions. In this study, we propose Face-MakeUpV2, a facial image generation model that aims to maintain the consistency of face ID and physical characteristics with the reference image. First, we constructed a large-scale dataset FaceCaptionMask-1M comprising approximately one million image-text-masks pairs that provide precise spatial supervision for the local semantic instructions. Second, we employed a general text-to-image pretrained model as the backbone and introduced two complementary facial information injection channels: a 3D facial rendering channel to incorporate the physical characteristics of the image and a global facial feature channel. Third, we formulated two optimization objectives for the supervised learning of our model: semantic alignment in the model's embedding space to mitigate the attribute leakage problem and perceptual loss on facial images to preserve ID consistency. Extensive experiments demonstrated that our Face-MakeUpV2 achieves best overall performance in terms of preserving face ID and maintaining physical consistency of the reference images. These results highlight the practical potential of Face-MakeUpV2 for reliable and controllable facial editing in diverse applications.
>
---
#### [replaced 021] GFT: Graph Feature Tuning for Efficient Point Cloud Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10799v2](https://arxiv.org/pdf/2511.10799v2)**

> **作者:** Manish Dhakal; Venkat R. Dasari; Rajshekhar Sunderraman; Yi Ding
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) significantly reduces computational and memory costs by updating only a small subset of the model's parameters, enabling faster adaptation to new tasks with minimal loss in performance. Previous studies have introduced PEFTs tailored for point cloud data, as general approaches are suboptimal. To further reduce the number of trainable parameters, we propose a point-cloud-specific PEFT, termed Graph Features Tuning (GFT), which learns a dynamic graph from initial tokenized inputs of the transformer using a lightweight graph convolution network and passes these graph features to deeper layers via skip connections and efficient cross-attention modules. Extensive experiments on object classification and segmentation tasks show that GFT operates in the same domain, rivalling existing methods, while reducing the trainable parameters. Code is available at https://github.com/manishdhakal/GFT.
>
---
#### [replaced 022] Benchmarking machine learning models for multi-class state recognition in double quantum dot data
- **分类: cs.CV; cond-mat.mes-hall; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.22451v2](https://arxiv.org/pdf/2511.22451v2)**

> **作者:** Valeria Díaz Moreno; Ryan P Khalili; Daniel Schug; Patrick J. Walsh; Justyna P. Zwolak
>
> **备注:** 12 pages, 4 figures, 2 tables
>
> **摘要:** Semiconductor quantum dots (QDs) are a leading platform for scalable quantum processors. However, scaling to large arrays requires reliable, automated tuning strategies for devices' bootstrapping, calibration, and operation, with many tuning aspects depending on accurately identifying QD device states from charge-stability diagrams (CSDs). In this work, we present a comprehensive benchmarking study of four modern machine learning (ML) architectures for multi-class state recognition in double-QD CSDs. We evaluate their performance across different data budgets and normalization schemes using both synthetic and experimental data. We find that the more resource-intensive models -- U-Nets and visual transformers (ViTs) -- achieve the highest MSE score (defined as $1-\mathrm{MSE}$) on synthetic data (over $0.98$) but fail to generalize to experimental data. MDNs are the most computationally efficient and exhibit highly stable training, but with substantially lower peak performance. CNNs offer the most favorable trade-off on experimental CSDs, achieving strong accuracy with two orders of magnitude fewer parameters than the U-Nets and ViTs. Normalization plays a nontrivial role: min-max scaling generally yields higher MSE scores but less stable convergence, whereas z-score normalization produces more predictable training dynamics but at reduced accuracy for most models. Overall, our study shows that CNNs with min-max normalization are a practical approach for QD CSDs.
>
---
#### [replaced 023] Have We Scene It All? Scene Graph-Aware Deep Point Cloud Compression
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D点云在多机器人系统中传输效率低的问题，提出基于语义场景图的深度压缩框架。通过语义分割点云并利用FiLM增强编码器，实现高效压缩与高保真重建，支持下游任务，显著提升压缩率与系统性能。**

- **链接: [https://arxiv.org/pdf/2510.08512v2](https://arxiv.org/pdf/2510.08512v2)**

> **作者:** Nikolaos Stathoulopoulos; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** Please cite published version. 8 pages, 6 figures
>
> **摘要:** Efficient transmission of 3D point cloud data is critical for advanced perception in centralized and decentralized multi-agent robotic systems, especially nowadays with the growing reliance on edge and cloud-based processing. However, the large and complex nature of point clouds creates challenges under bandwidth constraints and intermittent connectivity, often degrading system performance. We propose a deep compression framework based on semantic scene graphs. The method decomposes point clouds into semantically coherent patches and encodes them into compact latent representations with semantic-aware encoders conditioned by Feature-wise Linear Modulation (FiLM). A folding-based decoder, guided by latent features and graph node attributes, enables structurally accurate reconstruction. Experiments on the SemanticKITTI and nuScenes datasets show that the framework achieves state-of-the-art compression rates, reducing data size by up to 98% while preserving both structural and semantic fidelity. In addition, it supports downstream applications such as multi-robot pose graph optimization and map merging, achieving trajectory accuracy and map alignment comparable to those obtained with raw LiDAR scans.
>
---
#### [replaced 024] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.21192v2](https://arxiv.org/pdf/2511.21192v2)**

> **作者:** Hui Lu; Yi Yu; Yiming Yang; Chenyu Yi; Qixin Zhang; Bingquan Shen; Alex C. Kot; Xudong Jiang
>
> **摘要:** Vision-Language-Action (VLA) models are vulnerable to adversarial attacks, yet universal and transferable attacks remain underexplored, as most existing patches overfit to a single model and fail in black-box settings. To address this gap, we present a systematic study of universal, transferable adversarial patches against VLA-driven robots under unknown architectures, finetuned variants, and sim-to-real shifts. We introduce UPA-RFAS (Universal Patch Attack via Robust Feature, Attention, and Semantics), a unified framework that learns a single physical patch in a shared feature space while promoting cross-model transfer. UPA-RFAS combines (i) a feature-space objective with an $\ell_1$ deviation prior and repulsive InfoNCE loss to induce transferable representation shifts, (ii) a robustness-augmented two-phase min-max procedure where an inner loop learns invisible sample-wise perturbations and an outer loop optimizes the universal patch against this hardened neighborhood, and (iii) two VLA-specific losses: Patch Attention Dominance to hijack text$\to$vision attention and Patch Semantic Misalignment to induce image-text mismatch without labels. Experiments across diverse VLA models, manipulation suites, and physical executions show that UPA-RFAS consistently transfers across models, tasks, and viewpoints, exposing a practical patch-based attack surface and establishing a strong baseline for future defenses.
>
---
#### [replaced 025] Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.20518v3](https://arxiv.org/pdf/2504.20518v3)**

> **作者:** Zhongqi Wang; Jie Zhang; Shiguang Shan; Xilin Chen
>
> **备注:** Accepted by TPAMI
>
> **摘要:** Recent studies have revealed that text-to-image diffusion models are vulnerable to backdoor attacks, where attackers implant stealthy textual triggers to manipulate model outputs. Previous backdoor detection methods primarily focus on the static features of backdoor samples. However, a vital property of diffusion models is their inherent dynamism. This study introduces a novel backdoor detection perspective named Dynamic Attention Analysis (DAA), showing that these dynamic characteristics serve as better indicators for backdoor detection. Specifically, by examining the dynamic evolution of cross-attention maps, we observe that backdoor samples exhibit distinct feature evolution patterns at the $<$EOS$>$ token compared to benign samples. To quantify these dynamic anomalies, we first introduce DAA-I, which treats the tokens' attention maps as spatially independent and measures dynamic feature using the Frobenius norm. Furthermore, to better capture the interactions between attention maps and refine the feature, we propose a dynamical system-based approach, referred to as DAA-S. This model formulates the spatial correlations among attention maps using a graph-based state equation and we theoretically analyze the global asymptotic stability of this method. Extensive experiments across six representative backdoor attack scenarios demonstrate that our approach significantly surpasses existing detection methods, achieving an average F1 Score of 79.27% and an AUC of 86.27%. The code is available at https://github.com/Robin-WZQ/DAA.
>
---
#### [replaced 026] Fast Multi-view Consistent 3D Editing with Video Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23172v2](https://arxiv.org/pdf/2511.23172v2)**

> **作者:** Liyi Chen; Ruihuang Li; Guowen Zhang; Pengfei Wang; Lei Zhang
>
> **备注:** accepted by AAAI2026
>
> **摘要:** Text-driven 3D editing enables user-friendly 3D object or scene editing with text instructions. Due to the lack of multi-view consistency priors, existing methods typically resort to employing 2D generation or editing models to process each view individually, followed by iterative 2D-3D-2D updating. However, these methods are not only time-consuming but also prone to over-smoothed results because the different editing signals gathered from different views are averaged during the iterative process. In this paper, we propose generative Video Prior based 3D Editing (ViP3DE) to employ the temporal consistency priors from pre-trained video generation models for multi-view consistent 3D editing in a single forward pass. Our key insight is to condition the video generation model on a single edited view to generate other consistent edited views for 3D updating directly, thereby bypassing the iterative editing paradigm. Since 3D updating requires edited views to be paired with specific camera poses, we propose motion-preserved noise blending for the video model to generate edited views at predefined camera poses. In addition, we introduce geometry-aware denoising to further enhance multi-view consistency by integrating 3D geometric priors into video models. Extensive experiments demonstrate that our proposed ViP3DE can achieve high-quality 3D editing results even within a single forward pass, significantly outperforming existing methods in both editing quality and speed.
>
---
#### [replaced 027] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.17515v4](https://arxiv.org/pdf/2411.17515v4)**

> **作者:** Yijia Hong; Yuan-Chen Guo; Ran Yi; Yulong Chen; Yan-Pei Cao; Lizhuang Ma
>
> **备注:** https://hyj542682306.github.io/SuperMat/
>
> **摘要:** Decomposing physically-based materials from images into their constituent properties remains challenging, particularly when maintaining both computational efficiency and physical consistency. While recent diffusion-based approaches have shown promise, they face substantial computational overhead due to multiple denoising steps and separate models for different material properties. We present SuperMat, a single-step framework that achieves high-quality material decomposition with one-step inference. This enables end-to-end training with perceptual and re-render losses while decomposing albedo, metallic, and roughness maps at millisecond-scale speeds. We further extend our framework to 3D objects through a UV refinement network, enabling consistent material estimation across viewpoints while maintaining efficiency. Experiments demonstrate that SuperMat achieves state-of-the-art PBR material decomposition quality while reducing inference time from seconds to milliseconds per image, and completes PBR material estimation for 3D objects in approximately 3 seconds. The project page is at https://hyj542682306.github.io/SuperMat/.
>
---
#### [replaced 028] 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08015v2](https://arxiv.org/pdf/2506.08015v2)**

> **作者:** Zhen Xu; Zhengqin Li; Zhao Dong; Xiaowei Zhou; Richard Newcombe; Zhaoyang Lv
>
> **备注:** NeurIPS 2025 (Spotlight); Project Page: https://4dgt.github.io
>
> **摘要:** We propose 4DGT, a 4D Gaussian-based Transformer model for dynamic scene reconstruction, trained entirely on real-world monocular posed videos. Using 4D Gaussian as an inductive bias, 4DGT unifies static and dynamic components, enabling the modeling of complex, time-varying environments with varying object lifespans. We proposed a novel density control strategy in training, which enables our 4DGT to handle longer space-time input and remain efficient rendering at runtime. Our model processes 64 consecutive posed frames in a rolling-window fashion, predicting consistent 4D Gaussians in the scene. Unlike optimization-based methods, 4DGT performs purely feed-forward inference, reducing reconstruction time from hours to seconds and scaling effectively to long video sequences. Trained only on large-scale monocular posed video datasets, 4DGT can outperform prior Gaussian-based networks significantly in real-world videos and achieve on-par accuracy with optimization-based methods on cross-domain videos. Project page: https://4dgt.github.io
>
---
#### [replaced 029] M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.23728v2](https://arxiv.org/pdf/2509.23728v2)**

> **作者:** Yiheng Zhang; Zhuojiang Cai; Mingdao Wang; Meitong Guo; Tianxiao Li; Li Lin; Yuwang Wang
>
> **备注:** https://graphic-kiliani.github.io/M3DLayout/
>
> **摘要:** In text-driven 3D scene generation, object layout serves as a crucial intermediate representation that bridges high-level language instructions with detailed geometric output. It not only provides a structural blueprint for ensuring physical plausibility but also supports semantic controllability and interactive editing. However, the learning capabilities of current 3D indoor layout generation models are constrained by the limited scale, diversity, and annotation quality of existing datasets. To address this, we introduce M3DLayout, a large-scale, multi-source dataset for 3D indoor layout generation. M3DLayout comprises 21,367 layouts and over 433k object instances, integrating three distinct sources: real-world scans, professional CAD designs, and procedurally generated scenes. Each layout is paired with detailed structured text describing global scene summaries, relational placements of large furniture, and fine-grained arrangements of smaller items. This diverse and richly annotated resource enables models to learn complex spatial and semantic patterns across a wide variety of indoor environments. To assess the potential of M3DLayout, we establish a benchmark using both a text-conditioned diffusion model and a text-conditioned autoregressive model. Experimental results demonstrate that our dataset provides a solid foundation for training layout generation models. Its multi-source composition enhances diversity, notably through the Inf3DLayout subset which provides rich small-object information, enabling the generation of more complex and detailed scenes. We hope that M3DLayout can serve as a valuable resource for advancing research in text-driven 3D scene synthesis. All dataset and code will be made public upon acceptance.
>
---
#### [replaced 030] Deep Learning-Based Multiclass Classification of Oral Lesions with Stratified Augmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21582v2](https://arxiv.org/pdf/2511.21582v2)**

> **作者:** Joy Naoum; Revana Salama; Ali Hamdi
>
> **备注:** Technical error in the experimentation and we will resubmit it again
>
> **摘要:** Oral cancer is highly common across the globe and is mostly diagnosed during the later stages due to the close visual similarity to benign, precancerous, and malignant lesions in the oral cavity. Implementing computer aided diagnosis systems early on has the potential to greatly improve clinical outcomes. This research intends to use deep learning to build a multiclass classifier for sixteen different oral lesions. To overcome the challenges of limited and imbalanced datasets, the proposed technique combines stratified data splitting and advanced data augmentation and oversampling to perform the classification. The experimental results, which achieved 83.33 percent accuracy, 89.12 percent precision, and 77.31 percent recall, demonstrate the superiority of the suggested model over state of the art methods now in use. The suggested model effectively conveys the effectiveness of oversampling and augmentation strategies in situations where the minority class classification performance is noteworthy. As a first step toward trustworthy computer aided diagnostic systems for the early detection of oral cancer in clinical settings, the suggested framework shows promise.
>
---
#### [replaced 031] HiGFA: Hierarchical Guidance for Fine-grained Data Augmentation with Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12547v4](https://arxiv.org/pdf/2511.12547v4)**

> **作者:** Zhiguang Lu; Qianqian Xu; Peisong Wen; Siran Dai; Qingming Huang
>
> **摘要:** Generative diffusion models show promise for data augmentation. However, applying them to fine-grained tasks presents a significant challenge: ensuring synthetic images accurately capture the subtle, category-defining features critical for high fidelity. Standard approaches, such as text-based Classifier-Free Guidance (CFG), often lack the required specificity, potentially generating misleading examples that degrade fine-grained classifier performance. To address this, we propose Hierarchically Guided Fine-grained Augmentation (HiGFA). HiGFA leverages the temporal dynamics of the diffusion sampling process. It employs strong text and transformed contour guidance with fixed strengths in the early-to-mid sampling stages to establish overall scene, style, and structure. In the final sampling stages, HiGFA activates a specialized fine-grained classifier guidance and dynamically modulates the strength of all guidance signals based on prediction confidence. This hierarchical, confidence-driven orchestration enables HiGFA to generate diverse yet faithful synthetic images by intelligently balancing global structure formation with precise detail refinement. Experiments on several FGVC datasets demonstrate the effectiveness of HiGFA.
>
---
#### [replaced 032] Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.19418v2](https://arxiv.org/pdf/2511.19418v2)**

> **作者:** Yiming Qin; Bomin Wei; Jiaxin Ge; Konstantinos Kallidromitis; Stephanie Fu; Trevor Darrell; XuDong Wang
>
> **备注:** Project page: https://wakalsprojectpage.github.io/covt-website/
>
> **摘要:** Vision-Language Models (VLMs) excel at reasoning in linguistic space but struggle with perceptual understanding that requires dense visual perception, e.g., spatial reasoning and geometric awareness. This limitation stems from the fact that current VLMs have limited mechanisms to capture dense visual information across spatial dimensions. We introduce Chain-of-Visual-Thought (COVT), a framework that enables VLMs to reason not only in words but also through continuous visual tokens-compact latent representations that encode rich perceptual cues. Within a small budget of roughly 20 tokens, COVT distills knowledge from lightweight vision experts, capturing complementary properties such as 2D appearance, 3D geometry, spatial layout, and edge structure. During training, the VLM with COVT autoregressively predicts these visual tokens to reconstruct dense supervision signals (e.g., depth, segmentation, edges, and DINO features). At inference, the model reasons directly in the continuous visual token space, preserving efficiency while optionally decoding dense predictions for interpretability. Evaluated across more than ten diverse perception benchmarks, including CV-Bench, MMVP, RealWorldQA, MMStar, WorldMedQA, and HRBench, integrating COVT into strong VLMs such as Qwen2.5-VL and LLaVA consistently improves performance by 3% to 16% and demonstrates that compact continuous visual thinking enables more precise, grounded, and interpretable multimodal intelligence.
>
---
#### [replaced 033] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.19622v2](https://arxiv.org/pdf/2510.19622v2)**

> **作者:** Zhengxuan Wei; Jiajin Tang; Sibei Yang
>
> **备注:** This work is accepted by ICCV 2025
>
> **摘要:** Existing Moment Retrieval methods face three critical bottlenecks: (1) data scarcity forces models into shallow keyword-feature associations; (2) boundary ambiguity in transition regions between adjacent events; (3) insufficient discrimination of fine-grained semantics (e.g., distinguishing ``kicking" vs. ``throwing" a ball). In this paper, we propose a zero-external-dependency Augmented Moment Retrieval framework, AMR, designed to overcome local optima caused by insufficient data annotations and the lack of robust boundary and semantic discrimination capabilities. AMR is built upon two key insights: (1) it resolves ambiguous boundary information and semantic confusion in existing annotations without additional data (avoiding costly manual labeling), and (2) it preserves boundary and semantic discriminative capabilities enhanced by training while generalizing to real-world scenarios, significantly improving performance. Furthermore, we propose a two-stage training framework with cold-start and distillation adaptation. The cold-start stage employs curriculum learning on augmented data to build foundational boundary/semantic awareness. The distillation stage introduces dual query sets: Original Queries maintain DETR-based localization using frozen Base Queries from the cold-start model, while Active Queries dynamically adapt to real-data distributions. A cross-stage distillation loss enforces consistency between Original and Base Queries, preventing knowledge forgetting while enabling real-world generalization. Experiments on multiple benchmarks show that AMR achieves improved performance over prior state-of-the-art approaches.
>
---
#### [replaced 034] 3EED: Ground Everything Everywhere in 3D
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出3EED，一个大规模多平台3D视觉语言定位基准，涵盖车、无人机、四足机器人采集的RGB与LiDAR数据。针对现有数据集规模小、场景单一、平台受限的问题，构建超12.8万物体、2.2万标注表达的数据集，设计高效标注流程与跨平台对齐方法，推动开放世界中可泛化的3D语言感知研究。**

- **链接: [https://arxiv.org/pdf/2511.01755v2](https://arxiv.org/pdf/2511.01755v2)**

> **作者:** Rong Li; Yuhao Dong; Tianshuai Hu; Ao Liang; Youquan Liu; Dongyue Lu; Liang Pan; Lingdong Kong; Junwei Liang; Ziwei Liu
>
> **备注:** NeurIPS 2025 DB Track; 38 pages, 17 figures, 10 tables; Project Page at https://project-3eed.github.io/
>
> **摘要:** Visual grounding in 3D is the key for embodied agents to localize language-referred objects in open-world environments. However, existing benchmarks are limited to indoor focus, single-platform constraints, and small scale. We introduce 3EED, a multi-platform, multi-modal 3D grounding benchmark featuring RGB and LiDAR data from vehicle, drone, and quadruped platforms. We provide over 128,000 objects and 22,000 validated referring expressions across diverse outdoor scenes -- 10x larger than existing datasets. We develop a scalable annotation pipeline combining vision-language model prompting with human verification to ensure high-quality spatial grounding. To support cross-platform learning, we propose platform-aware normalization and cross-modal alignment techniques, and establish benchmark protocols for in-domain and cross-platform evaluations. Our findings reveal significant performance gaps, highlighting the challenges and opportunities of generalizable 3D grounding. The 3EED dataset and benchmark toolkit are released to advance future research in language-driven 3D embodied perception.
>
---
#### [replaced 035] Hierarchical Semi-Supervised Active Learning for Remote Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18058v2](https://arxiv.org/pdf/2511.18058v2)**

> **作者:** Wei Huang; Zhitong Xiong; Chenying Liu; Xiao Xiang Zhu
>
> **备注:** Under review
>
> **摘要:** The performance of deep learning models in remote sensing (RS) strongly depends on the availability of high-quality labeled data. However, collecting large-scale annotations is costly and time-consuming, while vast amounts of unlabeled imagery remain underutilized. To address this challenge, we propose a Hierarchical Semi-Supervised Active Learning (HSSAL) framework that integrates semi-supervised learning (SSL) and a novel hierarchical active learning (HAL) in a closed iterative loop. In each iteration, SSL refines the model using both labeled data through supervised learning and unlabeled data via weak-to-strong self-training, improving feature representation and uncertainty estimation. Guided by the refined representations and uncertainty cues of unlabeled samples, HAL then conducts sample querying through a progressive clustering strategy, selecting the most informative instances that jointly satisfy the criteria of scalability, diversity, and uncertainty. This hierarchical process ensures both efficiency and representativeness in sample selection. Extensive experiments on three benchmark RS scene classification datasets, including UCM, AID, and NWPU-RESISC45, demonstrate that HSSAL consistently outperforms SSL- or AL-only baselines. Remarkably, with only 8%, 4%, and 2% labeled training data on UCM, AID, and NWPU-RESISC45, respectively, HSSAL achieves over 95% of fully-supervised accuracy, highlighting its superior label efficiency through informativeness exploitation of unlabeled data. Our code will be publicly available.
>
---
#### [replaced 036] TTSnap: Test-Time Scaling of Diffusion Models via Noise-Aware Pruning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22242v2](https://arxiv.org/pdf/2511.22242v2)**

> **作者:** Qingtao Yu; Changlin Song; Minghao Sun; Zhengyang Yu; Vinay Kumar Verma; Soumya Roy; Sumit Negi; Hongdong Li; Dylan Campbell
>
> **摘要:** A prominent approach to test-time scaling for text-to-image diffusion models formulates the problem as a search over multiple noise seeds, selecting the one that maximizes a certain image-reward function. The effectiveness of this strategy heavily depends on the number and diversity of noise seeds explored. However, verifying each candidate is computationally expensive, because each must be fully denoised before a reward can be computed. This severely limits the number of samples that can be explored under a fixed budget. We propose test-time scaling with noise-aware pruning (TTSnap), a framework that prunes low-quality candidates without fully denoising them. The key challenge is that reward models are learned in the clean image domain, and the ranking of rewards predicted for intermediate estimates are often inconsistent with those predicted for clean images. To overcome this, we train noise-aware reward models via self-distillation to align the reward for intermediate estimates with that of the final clean images. To stabilize learning across different noise levels, we adopt a curriculum training strategy that progressively shifts the data domain from clean images to noise images. In addition, we introduce a new metric that measures reward alignment and computational budget utilization. Experiments demonstrate that our approach improves performance by over 16\% compared with existing methods, enabling more efficient and effective test-time scaling. It also provides orthogonal gains when combined with post-training techniques and local test-time optimization. Code: https://github.com/TerrysLearning/TTSnap/.
>
---
#### [replaced 037] CLIP-Free, Label-Free, Zero-Shot Concept Bottleneck Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.10981v3](https://arxiv.org/pdf/2503.10981v3)**

> **作者:** Fawaz Sammani; Jonas Fischer; Nikos Deligiannis
>
> **摘要:** Concept Bottleneck Models (CBMs) map dense, high-dimensional feature representations into a set of human-interpretable concepts which are then combined linearly to make a prediction. However, modern CBMs rely on the CLIP model to establish a mapping from dense feature representations to textual concepts, and it remains unclear how to design CBMs for models other than CLIP. Methods that do not use CLIP instead require manual, labor intensive annotation to associate feature representations with concepts. Furthermore, all CBMs necessitate training a linear classifier to map the extracted concepts to class labels. In this work, we lift all three limitations simultaneously by proposing a method that converts any frozen visual classifier into a CBM without requiring image-concept labels (label-free), without relying on the CLIP model (CLIP-free), and by deriving the linear classifier in a zero-shot manner. Our method is formulated by aligning the original classifier's distribution (over discrete class indices) with its corresponding vision-language counterpart distribution derived from textual class names, while preserving the classifier's performance. The approach requires no ground-truth image-class annotations, and is highly data-efficient and preserves the classifier's reasoning process. Applied and tested on over 40 visual classifiers, our resulting CLIP-free, zero-shot CBM sets a new state of the art, surpassing even supervised CLIP-based CBMs. Finally, we also show that our method can be used for zero-shot image captioning, outperforming existing methods based on CLIP, and achieving state of the art results.
>
---
#### [replaced 038] Adversarial Exploitation of Data Diversity Improves Visual Localization
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对视觉定位任务中绝对姿态回归（APR）方法泛化能力差的问题，提出利用外观多样性提升鲁棒性。通过将2D图像转为带外观与去模糊特性的3D高斯点云，合成多样化训练数据，并设计双分支对抗训练框架，有效缩小了仿真到真实场景的差距，显著降低定位误差，尤其在复杂动态和光照变化场景下表现优异。**

- **链接: [https://arxiv.org/pdf/2412.00138v2](https://arxiv.org/pdf/2412.00138v2)**

> **作者:** Sihang Li; Siqi Tan; Bowen Chang; Jing Zhang; Chen Feng; Yiming Li
>
> **备注:** 24 pages, 22 figures
>
> **摘要:** Visual localization, which estimates a camera's pose within a known scene, is a fundamental capability for autonomous systems. While absolute pose regression (APR) methods have shown promise for efficient inference, they often struggle with generalization. Recent approaches attempt to address this through data augmentation with varied viewpoints, yet they overlook a critical factor: appearance diversity. In this work, we identify appearance variation as the key to robust localization. Specifically, we first lift real 2D images into 3D Gaussian Splats with varying appearance and deblurring ability, enabling the synthesis of diverse training data that varies not just in poses but also in environmental conditions such as lighting and weather. To fully unleash the potential of the appearance-diverse data, we build a two-branch joint training pipeline with an adversarial discriminator to bridge the syn-to-real gap. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art methods, reducing translation and rotation errors by 50\% and 41\% on indoor datasets, and 38\% and 44\% on outdoor datasets. Most notably, our method shows remarkable robustness in dynamic driving scenarios under varying weather conditions and in day-to-night scenarios, where previous APR methods fail. Project Page: https://ai4ce.github.io/RAP/
>
---
#### [replaced 039] Multivariate Gaussian Representation Learning for Medical Action Evaluation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10060v2](https://arxiv.org/pdf/2511.10060v2)**

> **作者:** Luming Yang; Haoxian Liu; Siqing Li; Alper Yilmaz
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Fine-grained action evaluation in medical vision faces unique challenges due to the unavailability of comprehensive datasets, stringent precision requirements, and insufficient spatiotemporal dynamic modeling of very rapid actions. To support development and evaluation, we introduce CPREval-6k, a multi-view, multi-label medical action benchmark containing 6,372 expert-annotated videos with 22 clinical labels. Using this dataset, we present GaussMedAct, a multivariate Gaussian encoding framework, to advance medical motion analysis through adaptive spatiotemporal representation learning. Multivariate Gaussian Representation projects the joint motions to a temporally scaled multi-dimensional space, and decomposes actions into adaptive 3D Gaussians that serve as tokens. These tokens preserve motion semantics through anisotropic covariance modeling while maintaining robustness to spatiotemporal noise. Hybrid Spatial Encoding, employing a Cartesian and Vector dual-stream strategy, effectively utilizes skeletal information in the form of joint and bone features. The proposed method achieves 92.1% Top-1 accuracy with real-time inference on the benchmark, outperforming baseline by +5.9% accuracy with only 10% FLOPs. Cross-dataset experiments confirm the superiority of our method in robustness.
>
---
#### [replaced 040] Towards Efficient and Accurate Spiking Neural Networks via Adaptive Bit Allocation
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.23717v4](https://arxiv.org/pdf/2506.23717v4)**

> **作者:** Xingting Yao; Qinghao Hu; Fei Zhou; Tielong Liu; Gang Li; Peisong Wang; Jian Cheng
>
> **备注:** Neural Networks, In press
>
> **摘要:** Multi-bit spiking neural networks (SNNs) have recently become a heated research spot, pursuing energy-efficient and high-accurate AI. However, with more bits involved, the associated memory and computation demands escalate to the point where the performance improvements become disproportionate. Based on the insight that different layers demonstrate different importance and extra bits could be wasted and interfering, this paper presents an adaptive bit allocation strategy for direct-trained SNNs, achieving fine-grained layer-wise allocation of memory and computation resources. Thus, SNN's efficiency and accuracy can be improved. Specifically, we parametrize the temporal lengths and the bit widths of weights and spikes, and make them learnable and controllable through gradients. To address the challenges caused by changeable bit widths and temporal lengths, we propose the refined spiking neuron, which can handle different temporal lengths, enable the derivation of gradients for temporal lengths, and suit spike quantization better. In addition, we theoretically formulate the step-size mismatch problem of learnable bit widths, which may incur severe quantization errors to SNN, and accordingly propose the step-size renewal mechanism to alleviate this issue. Experiments on various datasets, including the static CIFAR and ImageNet datasets and the dynamic CIFAR-DVS and DVS-GESTURE datasets, demonstrate that our methods can reduce the overall memory and computation cost while achieving higher accuracy. Particularly, our SEWResNet-34 can achieve a 2.69\% accuracy gain and 4.16$\times$ lower bit budgets over the advanced baseline work on ImageNet. This work is open-sourced at \href{https://github.com/Ikarosy/Towards-Efficient-and-Accurate-Spiking-Neural-Networks-via-Adaptive-Bit-Allocation}{this link}.
>
---
#### [replaced 041] PRISM-Bench: A Benchmark of Puzzle-Based Visual Tasks with CoT Error Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23594v4](https://arxiv.org/pdf/2510.23594v4)**

> **作者:** Yusu Qian; Cheng Wan; Chao Jia; Yinfei Yang; Qingyu Zhao; Zhe Gan
>
> **备注:** This paper's first error detection task's ground truth data contains hallucination introduced by gpt and needs to be withdrawn
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable progress on vision-language tasks, yet their reasoning processes remain sometimes unreliable. We introduce PRISM-Bench, a benchmark of puzzle-based visual challenges designed to evaluate not only whether models can solve problems, but how their reasoning unfolds. Unlike prior evaluations that measure only final-answer accuracy, PRISM-Bench introduces a diagnostic task: given a visual puzzle and a step-by-step chain-of-thought (CoT) containing exactly one error, models must identify the first incorrect step. This setting enables fine-grained assessment of logical consistency, error detection, and visual reasoning. The puzzles in PRISM-Bench require multi-step symbolic, geometric, and analogical reasoning, resisting shortcuts based on superficial pattern matching. Evaluations across state-of-the-art MLLMs reveal a persistent gap between fluent generation and faithful reasoning: models that produce plausible CoTs often fail to locate simple logical faults. By disentangling answer generation from reasoning verification, PRISM-Bench offers a sharper lens on multimodal reasoning competence and underscores the need for diagnostic evaluation protocols in the development of trustworthy MLLMs.
>
---
#### [replaced 042] DMC$^3$: Dual-Modal Counterfactual Contrastive Construction for Egocentric Video Question Answering
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2510.20285v2](https://arxiv.org/pdf/2510.20285v2)**

> **作者:** Jiayi Zou; Chaofan Chen; Bing-Kun Bao; Changsheng Xu
>
> **摘要:** Egocentric Video Question Answering (Egocentric VideoQA) plays an important role in egocentric video understanding, which refers to answering questions based on first-person videos. Although existing methods have made progress through the paradigm of pre-training and fine-tuning, they ignore the unique challenges posed by the first-person perspective, such as understanding multiple events and recognizing hand-object interactions. To deal with these challenges, we propose a Dual-Modal Counterfactual Contrastive Construction (DMC$^3$) framework, which contains an egocentric videoqa baseline, a counterfactual sample construction module and a counterfactual sample-involved contrastive optimization. Specifically, We first develop a counterfactual sample construction module to generate positive and negative samples for textual and visual modalities through event description paraphrasing and core interaction mining, respectively. Then, We feed these samples together with the original samples into the baseline. Finally, in the counterfactual sample-involved contrastive optimization module, we apply contrastive loss to minimize the distance between the original sample features and the positive sample features, while maximizing the distance from the negative samples. Experiments show that our method achieve 52.51\% and 46.04\% on the \textit{normal} and \textit{indirect} splits of EgoTaskQA, and 13.2\% on QAEGO4D, both reaching the state-of-the-art performance.
>
---
#### [replaced 043] Structure is Supervision: Multiview Masked Autoencoders for Radiology
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.22294v2](https://arxiv.org/pdf/2511.22294v2)**

> **作者:** Sonia Laguna; Andrea Agostini; Alain Ryser; Samuel Ruiperez-Campillo; Irene Cannistraci; Moritz Vandenhirtz; Stephan Mandt; Nicolas Deperrois; Farhad Nooralahzadeh; Michael Krauthammer; Thomas M. Sutter; Julia E. Vogt
>
> **摘要:** Building robust medical machine learning systems requires pretraining strategies that exploit the intrinsic structure present in clinical data. We introduce Multiview Masked Autoencoder (MVMAE), a self-supervised framework that leverages the natural multi-view organization of radiology studies to learn view-invariant and disease-relevant representations. MVMAE combines masked image reconstruction with cross-view alignment, transforming clinical redundancy across projections into a powerful self-supervisory signal. We further extend this approach with MVMAE-V2T, which incorporates radiology reports as an auxiliary text-based learning signal to enhance semantic grounding while preserving fully vision-based inference. Evaluated on a downstream disease classification task on three large-scale public datasets, MIMIC-CXR, CheXpert, and PadChest, MVMAE consistently outperforms supervised and vision-language baselines. Furthermore, MVMAE-V2T provides additional gains, particularly in low-label regimes where structured textual supervision is most beneficial. Together, these results establish the importance of structural and textual supervision as complementary paths toward scalable, clinically grounded medical foundation models.
>
---
#### [replaced 044] Manual-PA: Learning 3D Part Assembly from Instruction Diagrams
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.18011v2](https://arxiv.org/pdf/2411.18011v2)**

> **作者:** Jiahao Zhang; Anoop Cherian; Cristian Rodriguez; Weijian Deng; Stephen Gould
>
> **备注:** Accepted to ICCV'25
>
> **摘要:** Assembling furniture amounts to solving the discrete-continuous optimization task of selecting the furniture parts to assemble and estimating their connecting poses in a physically realistic manner. The problem is hampered by its combinatorially large yet sparse solution space thus making learning to assemble a challenging task for current machine learning models. In this paper, we attempt to solve this task by leveraging the assembly instructions provided in diagrammatic manuals that typically accompany the furniture parts. Our key insight is to use the cues in these diagrams to split the problem into discrete and continuous phases. Specifically, we present Manual-PA, a transformer-based instruction Manual-guided 3D Part Assembly framework that learns to semantically align 3D parts with their illustrations in the manuals using a contrastive learning backbone towards predicting the assembly order and infers the 6D pose of each part via relating it to the final furniture depicted in the manual. To validate the efficacy of our method, we conduct experiments on the benchmark PartNet dataset. Our results show that using the diagrams and the order of the parts lead to significant improvements in assembly performance against the state of the art. Further, Manual-PA demonstrates strong generalization to real-world IKEA furniture assembly on the IKEA-Manual dataset.
>
---
#### [replaced 045] GARF: Learning Generalizable 3D Reassembly for Real-World Fractures
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.05400v2](https://arxiv.org/pdf/2504.05400v2)**

> **作者:** Sihang Li; Zeyu Jiang; Grace Chen; Chenyang Xu; Siqi Tan; Xue Wang; Irving Fang; Kristof Zyskowski; Shannon P. McPherron; Radu Iovita; Chen Feng; Jing Zhang
>
> **备注:** 18 pages. Project Page https://ai4ce.github.io/GARF/
>
> **摘要:** 3D reassembly is a challenging spatial intelligence task with broad applications across scientific domains. While large-scale synthetic datasets have fueled promising learning-based approaches, their generalizability to different domains is limited. Critically, it remains uncertain whether models trained on synthetic datasets can generalize to real-world fractures where breakage patterns are more complex. To bridge this gap, we propose GARF, a generalizable 3D reassembly framework for real-world fractures. GARF leverages fracture-aware pretraining to learn fracture features from individual fragments, with flow matching enabling precise 6-DoF alignments. At inference time, we introduce one-step preassembly, improving robustness to unseen objects and varying numbers of fractures. In collaboration with archaeologists, paleoanthropologists, and ornithologists, we curate Fractura, a diverse dataset for vision and learning communities, featuring real-world fracture types across ceramics, bones, eggshells, and lithics. Comprehensive experiments have shown our approach consistently outperforms state-of-the-art methods on both synthetic and real-world datasets, achieving 82.87\% lower rotation error and 25.15\% higher part accuracy. This sheds light on training on synthetic data to advance real-world 3D puzzle solving, demonstrating its strong generalization across unseen object shapes and diverse fracture types. GARF's code, data and demo are available at https://ai4ce.github.io/GARF/.
>
---
#### [replaced 046] FreeSwim: Revisiting Sliding-Window Attention Mechanisms for Training-Free Ultra-High-Resolution Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14712v2](https://arxiv.org/pdf/2511.14712v2)**

> **作者:** Yunfeng Wu; Jiayi Song; Zhenxiong Tan; Zihao He; Songhua Liu
>
> **备注:** 23 pages, 14 figures
>
> **摘要:** The quadratic time and memory complexity of the attention mechanism in modern Transformer based video generators makes end-to-end training for ultra high resolution videos prohibitively expensive. Motivated by this limitation, we introduce a training-free approach that leverages video Diffusion Transformers pretrained at their native scale to synthesize higher resolution videos without any additional training or adaptation. At the core of our method lies an inward sliding window attention mechanism, which originates from a key observation: maintaining each query token's training scale receptive field is crucial for preserving visual fidelity and detail. However, naive local window attention, unfortunately, often leads to repetitive content and exhibits a lack of global coherence in the generated results. To overcome this challenge, we devise a dual-path pipeline that backs up window attention with a novel cross-attention override strategy, enabling the semantic content produced by local attention to be guided by another branch with a full receptive field and, therefore, ensuring holistic consistency. Furthermore, to improve efficiency, we incorporate a cross-attention caching strategy for this branch to avoid the frequent computation of full 3D attention. Extensive experiments demonstrate that our method delivers ultra-high-resolution videos with fine-grained visual details and high efficiency in a training-free paradigm. Meanwhile, it achieves superior performance on VBench, even compared to training-based alternatives, with competitive or improved efficiency. Codes are available at: https://github.com/WillWu111/FreeSwim
>
---
#### [replaced 047] AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对自动驾驶中视觉-语言-动作模型的决策可解释性与动作合理性问题，提出AutoDrive-R²框架。通过构建含自省的思维链数据集和基于物理约束的强化学习策略，增强模型推理与自我反思能力，提升轨迹规划的逻辑性与真实性，在nuScenes和Waymo数据集上实现先进性能。**

- **链接: [https://arxiv.org/pdf/2509.01944v2](https://arxiv.org/pdf/2509.01944v2)**

> **作者:** Zhenlong Yuan; Chengxuan Qian; Jing Tang; Rui Chen; Zijian Song; Lei Sun; Xiangxiang Chu; Yujun Cai; Dapeng Zhang; Shuo Li
>
> **摘要:** Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.
>
---
#### [replaced 048] Capturing Context-Aware Route Choice Semantics for Trajectory Representation Learning
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.14819v2](https://arxiv.org/pdf/2510.14819v2)**

> **作者:** Ji Cao; Yu Wang; Tongya Zheng; Jie Song; Qinghong Guo; Zujie Ren; Canghong Jin; Gang Chen; Mingli Song
>
> **摘要:** Trajectory representation learning (TRL) aims to encode raw trajectory data into low-dimensional embeddings for downstream tasks such as travel time estimation, mobility prediction, and trajectory similarity analysis. From a behavioral perspective, a trajectory reflects a sequence of route choices within an urban environment. However, most existing TRL methods ignore this underlying decision-making process and instead treat trajectories as static, passive spatiotemporal sequences, thereby limiting the semantic richness of the learned representations. To bridge this gap, we propose CORE, a TRL framework that integrates context-aware route choice semantics into trajectory embeddings. CORE first incorporates a multi-granular Environment Perception Module, which leverages large language models (LLMs) to distill environmental semantics from point of interest (POI) distributions, thereby constructing a context-enriched road network. Building upon this backbone, CORE employs a Route Choice Encoder with a mixture-of-experts (MoE) architecture, which captures route choice patterns by jointly leveraging the context-enriched road network and navigational factors. Finally, a Transformer encoder aggregates the route-choice-aware representations into a global trajectory embedding. Extensive experiments on 4 real-world datasets across 6 downstream tasks demonstrate that CORE consistently outperforms 12 state-of-the-art TRL methods, achieving an average improvement of 9.79% over the best-performing baseline. Our code is available at https://github.com/caoji2001/CORE.
>
---
#### [replaced 049] InsightDrive: Insight Scene Representation for End-to-End Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.13047v2](https://arxiv.org/pdf/2503.13047v2)**

> **作者:** Ruiqi Song; Xianda Guo; Yanlun Peng; Qinggong Wei; Hangbin Wu; Long Chen
>
> **摘要:** Conventional end-to-end autonomous driving methods often rely on explicit global scene representations, which typically consist of 3D object detection, online mapping, and motion prediction. In contrast, human drivers selectively attend to task-relevant regions and implicitly reason over the broader traffic context. Motivated by this observation, we introduce a lightweight end-to-end autonomous driving framework, InsightDrive. Unlike approaches that directly embed large language models (LLMs), InsightDrive introduces an Insight scene representation that jointly models attention-centric explicit scene representation and reasoning-centric implicit scene representation, so that scene understanding aligns more closely with human cognitive patterns for trajectory planning. To this end, we employ Chain-of-Thought (CoT) instructions to model human driving cognition and design a task-level Mixture-of-Experts (MoE) adapter that injects this knowledge into the autonomous driving model at negligible parameter cost. We further condition the planner on both explicit and implicit scene representations and employ a diffusion-based generative policy, which produces robust trajectory predictions and decisions. The overall framework establishes a knowledge distillation pipeline that transfers human driving knowledge to LLMs and subsequently to onboard models. Extensive experiments on the nuScenes and Navsim benchmarks demonstrate that InsightDrive achieves significant improvements over conventional scene representation approaches.
>
---
#### [replaced 050] Dynamic Prompt Generation for Interactive 3D Medical Image Segmentation Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.03189v2](https://arxiv.org/pdf/2510.03189v2)**

> **作者:** Tidiane Camaret Ndir; Alexander Pfefferle; Robin Tibor Schirrmeister
>
> **摘要:** Interactive 3D biomedical image segmentation requires efficient models that can iteratively refine predictions based on user prompts. Current foundation models either lack volumetric awareness or suffer from limited interactive capabilities. We propose a training strategy that combines dynamic volumetric prompt generation with content-aware adaptive cropping to optimize the use of the image encoder. Our method simulates realistic user interaction patterns during training while addressing the computational challenges of learning from sequential refinement feedback on a single GPU. For efficient training, we initialize our network using the publicly available weights from the nnInteractive segmentation model. Evaluation on the \textbf{Foundation Models for Interactive 3D Biomedical Image Segmentation} competition demonstrates strong performance with an average final Dice score of 0.6385, normalized surface distance of 0.6614, and area-under-the-curve metrics of 2.4799 (Dice) and 2.5671 (NSD).
>
---
#### [replaced 051] Class-Conditional Distribution Balancing for Group Robust Classification
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.17314v3](https://arxiv.org/pdf/2504.17314v3)**

> **作者:** Miaoyun Zhao; Chenrong Li; Qiang Zhang
>
> **摘要:** Spurious correlations that lead models to correct predictions for the wrong reasons pose a critical challenge for robust real-world generalization. Existing research attributes this issue to group imbalance and addresses it by maximizing group-balanced or worst-group accuracy, which heavily relies on expensive bias annotations. A compromise approach involves predicting bias information using extensively pretrained foundation models, which requires large-scale data and becomes impractical for resource-limited rare domains. To address these challenges, we offer a novel perspective by reframing the spurious correlations as imbalances or mismatches in class-conditional distributions, and propose a simple yet effective robust learning method that eliminates the need for both bias annotations and predictions. With the goal of maximizing the conditional entropy (uncertainty) of the label given spurious factors, our method leverages a sample reweighting strategy to achieve class-conditional distribution balancing, which automatically highlights minority groups and classes, effectively dismantling spurious correlations and producing a debiased data distribution for classification. Extensive experiments and analysis demonstrate that our approach consistently delivers state-of-the-art performance, rivaling methods that rely on bias supervision.
>
---
#### [replaced 052] RDTF: Resource-efficient Dual-mask Training Framework for Multi-frame Animated Sticker Generation
- **分类: cs.MM; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.17735v2](https://arxiv.org/pdf/2503.17735v2)**

> **作者:** Zhiqiang Yuan; Ting Zhang; Peixiang Luo; Ying Deng; Jiapei Zhang; Zexi Jia; Jinchao Zhang; Jie Zhou
>
> **备注:** Submitted to TMM
>
> **摘要:** Recently, significant advancements have been achieved in video generation technology, but applying it to resource-constrained downstream tasks like multi-frame animated sticker generation (ASG) characterized by low frame rates, abstract semantics, and long tail frame length distribution-remains challenging. Parameter-efficient fine-tuning (PEFT) techniques (e.g., Adapter, LoRA) for large pre-trained models suffer from insufficient fitting ability and source-domain knowledge interference. In this paper, we propose Resource-Efficient Dual-Mask Training Framework (RDTF), a dedicated solution for multi-frame ASG task under resource constraints. We argue that training a compact model from scratch with million-level samples outperforms PEFT on large models, with RDTF realizing this via three core designs: 1) a Discrete Frame Generation Network (DFGN) optimized for low-frame-rate ASG, ensuring parameter efficiency; 2) a dual-mask based data utilization strategy to enhance the availability and diversity of limited data; 3) a difficulty-adaptive curriculum learning method that decomposes sample entropy into static and adaptive components, enabling easy-to-difficult training convergence. To provide high-quality data support for RDTFs training from scratch, we construct VSD2M-a million-level multi-modal animated sticker dataset with rich annotations (static and animated stickers, action-focused text descriptions)-filling the gap of dedicated animated data for ASG task. Experiments demonstrate that RDTF is quantitatively and qualitatively superior to state-of-the-art PEFT methods (e.g., I2V-Adapter, SimDA) on ASG tasks, verifying the feasibility of our framework under resource constraints.
>
---
#### [replaced 053] MOON: Generative MLLM-based Multimodal Representation Learning for E-commerce Product Understanding
- **分类: cs.CV; cs.AI; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.11999v4](https://arxiv.org/pdf/2508.11999v4)**

> **作者:** Daoze Zhang; Chenghan Fu; Zhanheng Nie; Jianyu Liu; Wanxian Guan; Yuan Gao; Jun Song; Pengjie Wang; Jian Xu; Bo Zheng
>
> **备注:** Accepted by WSDM 2026. 11 pages, 9 figures
>
> **摘要:** With the rapid advancement of e-commerce, exploring general representations rather than task-specific ones has attracted increasing research attention. For product understanding, although existing discriminative dual-flow architectures drive progress in this field, they inherently struggle to model the many-to-one alignment between multiple images and texts of products. Therefore, we argue that generative Multimodal Large Language Models (MLLMs) hold significant potential for improving product representation learning. Nevertheless, achieving this goal still remains non-trivial due to several key challenges: the lack of multimodal and aspect-aware modeling modules in typical LLMs; the common presence of background noise in product images; and the absence of a standard benchmark for evaluation. To address these issues, we propose the first generative MLLM-based model named MOON for product representation learning. Our method (1) employs a guided Mixture-of-Experts (MoE) module for targeted modeling of multimodal and aspect-specific product content; (2) effectively detects core semantic regions in product images to mitigate the distraction and interference caused by background noise; and (3) introduces the specialized negative sampling strategy to increase the difficulty and diversity of negative samples. In addition, we release a large-scale multimodal benchmark MBE for various product understanding tasks. Experimentally, our model demonstrates competitive zero-shot performance on both our benchmark and the public dataset, showcasing strong generalization across various downstream tasks, including cross-modal retrieval, product classification, and attribute prediction. Furthermore, the case study and visualization illustrate the effectiveness of MOON for product understanding.
>
---
#### [replaced 054] A Cross-Hierarchical Difference Feature Fusion Network Based on Multiscale Encoder-Decoder for Hyperspectral Change Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16988v2](https://arxiv.org/pdf/2509.16988v2)**

> **作者:** Mingshuai Sheng; Bhatti Uzair Aslam; Junfeng Zhang; Siling Feng; Yonis Gulzar
>
> **摘要:** Hyperspectral change detection (HCD) is one of the core applications of remote sensing images, holding significant research value in fields like environmental monitoring and disaster assessment. However, existing methods often suffer from incomplete capture of multiscale spatial-spectral features and insufficient fusion of differential feature information. To address these challenges, this paper proposes a Cross-Hierarchical Differential Feature Fusion Network (CHDFFN) based on a multiscale encoder-decoder. Firstly, a multiscale feature extraction subnetwork is designed, taking the customized encoder-decoder as the backbone, combined with residual connections and the proposed dual-core channel-spatial attention module to achieve multi-level extraction and initial integration of spatial-spectral features. The encoder embeds convolutional blocks with different receptive field sizes to capture multiscale representations from shallow details to deep semantics. The decoder fuses the encoder's output via skip connections to gradually restore spatial resolution while suppressing background noise and redundancy. To enhance the model's ability to capture differential features between bi-temporal hyperspectral images, a spatial-spectral change feature learning module is designed to learn hierarchical change representations. Additionally, an adaptive high-level feature fusion module is proposed, dynamically balancing the contribution of hierarchical differential features by adaptively assigning weights, which effectively strengthens the model's capability to characterize complex change patterns. Finally, experiments on four public hyperspectral datasets show that compared with some state-of-the-art methods, the average maximum improvements of OA, KC, and F1 are 4.61%, 19.79%, and 18.90% respectively, verifying the model's effectiveness.
>
---
#### [replaced 055] Interpreting ResNet-based CLIP via Neuron-Attention Decomposition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.19943v3](https://arxiv.org/pdf/2509.19943v3)**

> **作者:** Edmund Bu; Yossi Gandelsman
>
> **备注:** Accepted at NeurIPS 2025 Workshop on Mechanistic Interpretability. Project page: https://edmundbu.github.io/clip-neur-attn/
>
> **摘要:** We present a novel technique for interpreting the neurons in CLIP-ResNet by decomposing their contributions to the output into individual computation paths. More specifically, we analyze all pairwise combinations of neurons and the following attention heads of CLIP's attention-pooling layer. We find that these neuron-head pairs can be approximated by a single direction in CLIP-ResNet's image-text embedding space. Leveraging this insight, we interpret each neuron-head pair by associating it with text. Additionally, we find that only a sparse set of the neuron-head pairs have a significant contribution to the output value, and that some neuron-head pairs, while polysemantic, represent sub-concepts of their corresponding neurons. We use these observations for two applications. First, we employ the pairs for training-free semantic segmentation, outperforming previous methods for CLIP-ResNet. Second, we utilize the contributions of neuron-head pairs to monitor dataset distribution shifts. Our results demonstrate that examining individual computation paths in neural networks uncovers interpretable units, and that such units can be utilized for downstream tasks.
>
---
#### [replaced 056] MBMamba: When Memory Buffer Meets Mamba for Structure-Aware Image Deblurring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12346v2](https://arxiv.org/pdf/2508.12346v2)**

> **作者:** Hu Gao; Xiaoning Lei; Xichen Xu; Depeng Dang; Lizhuang Ma
>
> **摘要:** The Mamba architecture has emerged as a promising alternative to CNNs and Transformers for image deblurring. However, its flatten-and-scan strategy often results in local pixel forgetting and channel redundancy, limiting its ability to effectively aggregate 2D spatial information. Although existing methods mitigate this by modifying the scan strategy or incorporating local feature modules, it increase computational complexity and hinder real-time performance. In this paper, we propose a structure-aware image deblurring network without changing the original Mamba architecture. Specifically, we design a memory buffer mechanism to preserve historical information for later fusion, enabling reliable modeling of relevance between adjacent features. Additionally, we introduce an Ising-inspired regularization loss that simulates the energy minimization of the physical system's "mutual attraction" between pixels, helping to maintain image structure and coherence. Building on this, we develop MBMamba. Experimental results show that our method outperforms state-of-the-art approaches on widely used benchmarks.
>
---
#### [replaced 057] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16024v2](https://arxiv.org/pdf/2511.16024v2)**

> **作者:** Xiao He; Zhijun Tu; Kun Cheng; Mingrui Zhu; Jie Hu; Nannan Wang; Xinbo Gao
>
> **备注:** 16 pages, Accepted by AAAI 2026, v2: corrected typos
>
> **摘要:** The demonstrated success of sparsely-gated Mixture-of-Experts (MoE) architectures, exemplified by models such as DeepSeek and Grok, has motivated researchers to investigate their adaptation to diverse domains. In real-world image super-resolution (Real-ISR), existing approaches mainly rely on fine-tuning pre-trained diffusion models through Low-Rank Adaptation (LoRA) module to reconstruct high-resolution (HR) images. However, these dense Real-ISR models are limited in their ability to adaptively capture the heterogeneous characteristics of complex real-world degraded samples or enable knowledge sharing between inputs under equivalent computational budgets. To address this, we investigate the integration of sparse MoE into Real-ISR and propose a Mixture-of-Ranks (MoR) architecture for single-step image super-resolution. We introduce a fine-grained expert partitioning strategy that treats each rank in LoRA as an independent expert. This design enables flexible knowledge recombination while isolating fixed-position ranks as shared experts to preserve common-sense features and minimize routing redundancy. Furthermore, we develop a degradation estimation module leveraging CLIP embeddings and predefined positive-negative text pairs to compute relative degradation scores, dynamically guiding expert activation. To better accommodate varying sample complexities, we incorporate zero-expert slots and propose a degradation-aware load-balancing loss, which dynamically adjusts the number of active experts based on degradation severity, ensuring optimal computational resource allocation. Comprehensive experiments validate our framework's effectiveness and state-of-the-art performance.
>
---
#### [replaced 058] CountSteer: Steering Attention for Object Counting in Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11253v3](https://arxiv.org/pdf/2511.11253v3)**

> **作者:** Hyemin Boo; Hyoryung Kim; Myungjin Lee; Seunghyeon Lee; Jiyoung Lee; Jang-Hwan Choi; Hyunsoo Cho
>
> **备注:** Accepted to AAAI 2026 Workshop on Shaping Responsible Synthetic Data in the Era of Foundation Models (RSD)
>
> **摘要:** Text-to-image diffusion models generate realistic and coherent images but often fail to follow numerical instructions in text, revealing a gap between language and visual representation. Interestingly, we found that these models are not entirely blind to numbers-they are implicitly aware of their own counting accuracy, as their internal signals shift in consistent ways depending on whether the output meets the specified count. This observation suggests that the model already encodes a latent notion of numerical correctness, which can be harnessed to guide generation more precisely. Building on this intuition, we introduce CountSteer, a training-free method that improves generation of specified object counts by steering the model's cross-attention hidden states during inference. In our experiments, CountSteer improved object-count accuracy by about 4% without compromising visual quality, demonstrating a simple yet effective step toward more controllable and semantically reliable text-to-image generation.
>
---
#### [replaced 059] MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对AI在非语言社交理解上的不足，提出基于哑剧视频的MimeQA数据集，构建非语言社交推理任务。通过806个标注问答对，评估视频大模型发现其非语言理解能力差（准确率20-30%），远低于人类（86%），揭示模型过度依赖文本提示、难以捕捉细微非语言互动的问题。**

- **链接: [https://arxiv.org/pdf/2502.16671v3](https://arxiv.org/pdf/2502.16671v3)**

> **作者:** Hengzhi Li; Megan Tjandrasuwita; Yi R. Fung; Armando Solar-Lezama; Paul Pu Liang
>
> **备注:** NeurIPS 2025 Datasets and Benchmarks
>
> **摘要:** As AI becomes more closely integrated with peoples' daily activities, socially intelligent AI that can understand and interact seamlessly with humans in daily lives is increasingly important. However, current works in AI social reasoning all rely on language-only or language-dominant approaches to benchmark and training models, resulting in systems that are improving in verbal communication but struggle with nonverbal social understanding. To address this limitation, we tap into a novel data source rich in nonverbal social interactions -- mime videos. Mimes refer to the art of expression through gesture and movement without spoken words, which presents unique challenges and opportunities in interpreting nonverbal social communication. We contribute a new dataset called MimeQA, obtained by sourcing ~8 hours of videos clips from YouTube and developing a comprehensive video question-answering benchmark comprising 806 carefully annotated and verified question-answer pairs, designed to probe nonverbal social reasoning capabilities. Using MimeQA, we evaluate state-of-the-art video large language models (VideoLLMs) and find that they achieve low accuracy, generally ranging from 20-30%, while humans score 86%. Our analysis reveals that VideoLLMs often fail to ground imagined objects and over-rely on the text prompt while ignoring subtle nonverbal interactions. We hope to inspire future work in AI models that embody true social intelligence capable of interpreting non-verbal human interactions.
>
---
#### [replaced 060] Modeling Rapid Contextual Learning in the Visual Cortex with Fast-Weight Deep Autoencoder Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04988v2](https://arxiv.org/pdf/2508.04988v2)**

> **作者:** Yue Li; Weifan Wang; Tai Sing Lee
>
> **摘要:** Recent neurophysiological studies have revealed that the early visual cortex can rapidly learn global image context, as evidenced by a sparsification of population responses and a reduction in mean activity when exposed to familiar versus novel image contexts. This phenomenon has been attributed primarily to local recurrent interactions, rather than changes in feedforward or feedback pathways, supported by both empirical findings and circuit-level modeling. Recurrent neural circuits capable of simulating these effects have been shown to reshape the geometry of neural manifolds, enhancing robustness and invariance to irrelevant variations. In this study, we employ a Vision Transformer (ViT)-based autoencoder to investigate, from a functional perspective, how familiarity training can induce sensitivity to global context in the early layers of a deep neural network. We hypothesize that rapid learning operates via fast weights, which encode transient or short-term memory traces, and we explore the use of Low-Rank Adaptation (LoRA) to implement such fast weights within each Transformer layer. Our results show that (1) The proposed ViT-based autoencoder's self-attention circuit performs a manifold transform similar to a neural circuit model of the familiarity effect. (2) Familiarity training aligns latent representations in early layers with those in the top layer that contains global context information. (3) Familiarity training broadens the self-attention scope within the remembered image context. (4) These effects are significantly amplified by LoRA-based fast weights. Together, these findings suggest that familiarity training introduces global sensitivity to earlier layers in a hierarchical network, and that a hybrid fast-and-slow weight architecture may provide a viable computational model for studying rapid global context learning in the brain.
>
---
#### [replaced 061] CaliTex: Geometry-Calibrated Attention for View-Coherent 3D Texture Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21309v2](https://arxiv.org/pdf/2511.21309v2)**

> **作者:** Chenyu Liu; Hongze Chen; Jingzhi Bao; Lingting Zhu; Runze Zhang; Weikai Chen; Zeyu Hu; Yingda Yin; Keyang Luo; Xin Wang
>
> **摘要:** Despite major advances brought by diffusion-based models, current 3D texture generation systems remain hindered by cross-view inconsistency -- textures that appear convincing from one viewpoint often fail to align across others. We find that this issue arises from attention ambiguity, where unstructured full attention is applied indiscriminately across tokens and modalities, causing geometric confusion and unstable appearance-structure coupling. To address this, we introduce CaliTex, a framework of geometry-calibrated attention that explicitly aligns attention with 3D structure. It introduces two modules: Part-Aligned Attention that enforces spatial alignment across semantically matched parts, and Condition-Routed Attention which routes appearance information through geometry-conditioned pathways to maintain spatial fidelity. Coupled with a two-stage diffusion transformer, CaliTex makes geometric coherence an inherent behavior of the network rather than a byproduct of optimization. Empirically, CaliTex produces seamless and view-consistent textures and outperforms both open-source and commercial baselines.
>
---
#### [replaced 062] Fine-grained Image Retrieval via Dual-Vision Adaptation
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2506.16273v4](https://arxiv.org/pdf/2506.16273v4)**

> **作者:** Xin Jiang; Meiqi Cao; Hao Tang; Fei Shen; Zechao Li
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Fine-Grained Image Retrieval~(FGIR) faces challenges in learning discriminative visual representations to retrieve images with similar fine-grained features. Current leading FGIR solutions typically follow two regimes: enforce pairwise similarity constraints in the semantic embedding space, or incorporate a localization sub-network to fine-tune the entire model. However, such two regimes tend to overfit the training data while forgetting the knowledge gained from large-scale pre-training, thus reducing their generalization ability. In this paper, we propose a Dual-Vision Adaptation (DVA) approach for FGIR, which guides the frozen pre-trained model to perform FGIR through collaborative sample and feature adaptation. Specifically, we design Object-Perceptual Adaptation, which modifies input samples to help the pre-trained model perceive critical objects and elements within objects that are helpful for category prediction. Meanwhile, we propose In-Context Adaptation, which introduces a small set of parameters for feature adaptation without modifying the pre-trained parameters. This makes the FGIR task using these adjusted features closer to the task solved during the pre-training. Additionally, to balance retrieval efficiency and performance, we propose Discrimination Perception Transfer to transfer the discriminative knowledge in the object-perceptual adaptation to the image encoder using the knowledge distillation mechanism. Extensive experiments show that DVA has fewer learnable parameters and performs well on three in-distribution and three out-of-distribution fine-grained datasets.
>
---
#### [replaced 063] OmniSVG: A Unified Scalable Vector Graphics Generation Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.06263v3](https://arxiv.org/pdf/2504.06263v3)**

> **作者:** Yiying Yang; Wei Cheng; Sijin Chen; Xianfang Zeng; Fukun Yin; Jiaxu Zhang; Liao Wang; Gang Yu; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** 20 pages; Project Page: https://omnisvg.github.io/
>
> **摘要:** Scalable Vector Graphics (SVG) is an important image format widely adopted in graphic design because of their resolution independence and editability. The study of generating high-quality SVG has continuously drawn attention from both designers and researchers in the AIGC community. However, existing methods either produces unstructured outputs with huge computational cost or is limited to generating monochrome icons of over-simplified structures. To produce high-quality and complex SVG, we propose OmniSVG, a unified framework that leverages pre-trained Vision-Language Models (VLMs) for end-to-end multimodal SVG generation. By parameterizing SVG commands and coordinates into discrete tokens, OmniSVG decouples structural logic from low-level geometry for efficient training while maintaining the expressiveness of complex SVG structure. To further advance the development of SVG synthesis, we introduce MMSVG-2M, a multimodal dataset with two million richly annotated SVG assets, along with a standardized evaluation protocol for conditional SVG generation tasks. Extensive experiments show that OmniSVG outperforms existing methods and demonstrates its potential for integration into professional SVG design workflows.
>
---
#### [replaced 064] AgriPotential: A Novel Multi-Spectral and Multi-Temporal Remote Sensing Dataset for Agricultural Potentials
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2506.11740v3](https://arxiv.org/pdf/2506.11740v3)**

> **作者:** Mohammad El Sakka; Caroline De Pourtales; Lotfi Chaari; Josiane Mothe
>
> **备注:** Accepted at CBMI 2025
>
> **摘要:** Remote sensing has emerged as a critical tool for large-scale Earth monitoring and land management. In this paper, we introduce AgriPotential, a novel benchmark dataset composed of Sentinel-2 satellite imagery captured over multiple months. The dataset provides pixel-level annotations of agricultural potentials for three major crop types - viticulture, market gardening, and field crops - across five ordinal classes. AgriPotential supports a broad range of machine learning tasks, including ordinal regression, multi-label classification, and spatio-temporal modeling. The data cover diverse areas in Southern France, offering rich spectral information. AgriPotential is the first public dataset designed specifically for agricultural potential prediction, aiming to improve data-driven approaches to sustainable land use planning. The dataset and the code are freely accessible at: https://zenodo.org/records/15551829
>
---
#### [replaced 065] MRI Super-Resolution with Deep Learning: A Comprehensive Survey
- **分类: eess.IV; cs.AI; cs.CV; eess.SP**

- **链接: [https://arxiv.org/pdf/2511.16854v2](https://arxiv.org/pdf/2511.16854v2)**

> **作者:** Mohammad Khateri; Serge Vasylechko; Morteza Ghahremani; Liam Timms; Deniz Kocanaogullari; Simon K. Warfield; Camilo Jaimes; Davood Karimi; Alejandra Sierra; Jussi Tohka; Sila Kurugol; Onur Afacan
>
> **备注:** 41 pages
>
> **摘要:** High-resolution (HR) magnetic resonance imaging (MRI) is crucial for many clinical and research applications. However, achieving it remains costly and constrained by technical trade-offs and experimental limitations. Super-resolution (SR) presents a promising computational approach to overcome these challenges by generating HR images from more affordable low-resolution (LR) scans, potentially improving diagnostic accuracy and efficiency without requiring additional hardware. This survey reviews recent advances in MRI SR techniques, with a focus on deep learning (DL) approaches. It examines DL-based MRI SR methods from the perspectives of computer vision, computational imaging, inverse problems, and MR physics, covering theoretical foundations, architectural designs, learning strategies, benchmark datasets, and performance metrics. We propose a systematic taxonomy to categorize these methods and present an in-depth study of both established and emerging SR techniques applicable to MRI, considering unique challenges in clinical and research contexts. We also highlight open challenges and directions that the community needs to address. Additionally, we provide a collection of essential open-access resources, tools, and tutorials, available on our GitHub: https://github.com/mkhateri/Awesome-MRI-Super-Resolution. IEEE keywords: MRI, Super-Resolution, Deep Learning, Computational Imaging, Inverse Problem, Survey.
>
---
#### [replaced 066] TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21526v2](https://arxiv.org/pdf/2509.21526v2)**

> **作者:** Hongyang He; Xinyuan Song; Yangfan He; Zeyu Zhang; Yanshu Li; Haochen You; Lifan Sun; Wenqiao Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We introduce TRiCo, a novel triadic game-theoretic co-training framework that rethinks the structure of semi-supervised learning by incorporating a teacher, two students, and an adversarial generator into a unified training paradigm. Unlike existing co-training or teacher-student approaches, TRiCo formulates SSL as a structured interaction among three roles: (i) two student classifiers trained on frozen, complementary representations, (ii) a meta-learned teacher that adaptively regulates pseudo-label selection and loss balancing via validation-based feedback, and (iii) a non-parametric generator that perturbs embeddings to uncover decision boundary weaknesses. Pseudo-labels are selected based on mutual information rather than confidence, providing a more robust measure of epistemic uncertainty. This triadic interaction is formalized as a Stackelberg game, where the teacher leads strategy optimization and students follow under adversarial perturbations. By addressing key limitations in existing SSL frameworks, such as static view interactions, unreliable pseudo-labels, and lack of hard sample modeling, TRiCo provides a principled and generalizable solution. Extensive experiments on CIFAR-10, SVHN, STL-10, and ImageNet demonstrate that TRiCo consistently achieves state-of-the-art performance in low-label regimes, while remaining architecture-agnostic and compatible with frozen vision backbones.Code:https://github.com/HoHongYeung/NeurIPS25-TRiCo.
>
---
#### [replaced 067] 3D Motion Perception of Binocular Vision Target with PID-CNN
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.20332v2](https://arxiv.org/pdf/2511.20332v2)**

> **作者:** Jiazhao Shi; Pan Pan; Haotian Shi
>
> **备注:** 7 pages, 9 figures, 2 tables. The codes of this article have been released at: https://github.com/ShiJZ123/PID-CNN
>
> **摘要:** This article trained a network for perceiving three-dimensional motion information of binocular vision target, which can provide real-time three-dimensional coordinate, velocity, and acceleration, and has a basic spatiotemporal perception capability. Understood the ability of neural networks to fit nonlinear problems from the perspective of PID. Considered a single-layer neural network as using a second-order difference equation and a nonlinearity to describe a local problem. Multilayer networks gradually transform the raw representation to the desired representation through multiple such combinations. Analysed some reference principles for designing neural networks. Designed a relatively small PID convolutional neural network, with a total of 17 layers and 413 thousand parameters. Implemented a simple but practical feature reuse method by concatenation and pooling. The network was trained and tested using the simulated randomly moving ball datasets, and the experimental results showed that the prediction accuracy was close to the upper limit that the input image resolution can represent. Analysed the experimental results and errors, as well as the existing shortcomings and possible directions for improvement. Finally, discussed the advantages of high-dimensional convolution in improving computational efficiency and feature space utilization. As well as the potential advantages of using PID information to implement memory and attention mechanisms.
>
---
#### [replaced 068] PointNSP: Autoregressive 3D Point Cloud Generation with Next-Scale Level-of-Detail Prediction
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.08594v3](https://arxiv.org/pdf/2503.08594v3)**

> **作者:** Ziqiao Meng; Qichao Wang; Zhiyang Dou; Zixing Song; Zhipeng Zhou; Irwin King; Peilin Zhao
>
> **备注:** 24 pages; Previously this version appeared as arXiv:2510.05613 which was submitted as a new work by accident
>
> **摘要:** Autoregressive point cloud generation has long lagged behind diffusion-based approaches in quality. The performance gap stems from the fact that autoregressive models impose an artificial ordering on inherently unordered point sets, forcing shape generation to proceed as a sequence of local predictions. This sequential bias emphasizes short-range continuity but undermines the model's capacity to capture long-range dependencies, hindering its ability to enforce global structural properties such as symmetry, consistent topology, and large-scale geometric regularities. Inspired by the level-of-detail (LOD) principle in shape modeling, we propose PointNSP, a coarse-to-fine generative framework that preserves global shape structure at low resolutions and progressively refines fine-grained geometry at higher scales through a next-scale prediction paradigm. This multi-scale factorization aligns the autoregressive objective with the permutation-invariant nature of point sets, enabling rich intra-scale interactions while avoiding brittle fixed orderings. Experiments on ShapeNet show that PointNSP establishes state-of-the-art (SOTA) generation quality for the first time within the autoregressive paradigm. In addition, it surpasses strong diffusion-based baselines in parameter, training, and inference efficiency. Finally, in dense generation with 8,192 points, PointNSP's advantages become even more pronounced, underscoring its scalability potential.
>
---
#### [replaced 069] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.24365v2](https://arxiv.org/pdf/2509.24365v2)**

> **作者:** Jitai Hao; Hao Liu; Xinyan Xiao; Qiang Huang; Jun Yu
>
> **摘要:** Unified Multimodal Models (UMMs) built on shared autoregressive (AR) transformers are attractive for their architectural simplicity. However, we identify a critical limitation: when trained on multimodal inputs, modality-shared transformers suffer from severe gradient conflicts between vision and text, particularly in shallow and deep layers. We trace this issue to the fundamentally different low-level statistical properties of images and text, while noting that conflicts diminish in middle layers where representations become more abstract and semantically aligned. To overcome this challenge, we propose Uni-X, a two-end-separated, middle-shared architecture. Uni-X dedicates its initial and final layers to modality-specific processing, while maintaining shared parameters in the middle layers for high-level semantic fusion. This X-shaped design not only eliminates gradient conflicts at both ends but also further alleviates residual conflicts in the shared layers. Extensive experiments validate the effectiveness of Uni-X. Under identical training conditions, Uni-X achieves superior training efficiency compared to strong baselines. When scaled to 3B parameters with larger training data, Uni-X matches or surpasses 7B AR-based UMMs, achieving a GenEval score of 82 for image generation alongside strong performance in text and vision understanding tasks. These results establish Uni-X as a parameter-efficient and scalable foundation for future unified multimodal modeling. Our code is available at https://github.com/CURRENTF/Uni-X
>
---
#### [replaced 070] Pushing the Limits of Sparsity: A Bag of Tricks for Extreme Pruning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.13545v4](https://arxiv.org/pdf/2411.13545v4)**

> **作者:** Andy Li; Aiden Durrant; Milan Markovic; Tianjin Huang; Souvik Kundu; Tianlong Chen; Lu Yin; Georgios Leontidis
>
> **备注:** V4: moderate revisions and overall improvements for journal camera ready submission
>
> **摘要:** Pruning of deep neural networks has been an effective technique for reducing model size while preserving most of the performance of dense networks, crucial for deploying models on memory and power-constrained devices. While recent sparse learning methods have shown promising performance up to moderate sparsity levels such as 95% and 98%, accuracy quickly deteriorates when pushing sparsities to extreme levels due to unique challenges such as fragile gradient flow. In this work, we explore network performance beyond the commonly studied sparsities, and develop techniques that encourage stable training without accuracy collapse even at extreme sparsities, including 99.90%, 99.95\% and 99.99% on ResNet architectures. We propose three complementary techniques that enhance sparse training through different mechanisms: 1) Dynamic ReLU phasing, where DyReLU initially allows for richer parameter exploration before being gradually replaced by standard ReLU, 2) weight sharing which reuses parameters within a residual layer while maintaining the same number of learnable parameters, and 3) cyclic sparsity, where both sparsity levels and sparsity patterns evolve dynamically throughout training to better encourage parameter exploration. We evaluate our method, which we term Extreme Adaptive Sparse Training (EAST) at extreme sparsities using ResNet-34 and ResNet-50 on CIFAR-10, CIFAR-100, and ImageNet, achieving competitive or improved performance compared to existing methods, with notable gains at extreme sparsity levels.
>
---
#### [replaced 071] Vision Language Models are Biased
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23941v3](https://arxiv.org/pdf/2505.23941v3)**

> **作者:** An Vo; Khai-Nguyen Nguyen; Mohammad Reza Taesiri; Vy Tuong Dang; Anh Totti Nguyen; Daeyoung Kim
>
> **备注:** Code and qualitative examples are available at: vlmsarebiased.github.io
>
> **摘要:** Large language models (LLMs) memorize a vast amount of prior knowledge from the Internet that helps them on downstream tasks but also may notoriously sway their outputs towards wrong or biased answers. In this work, we test how the knowledge about popular subjects hurt the accuracy of vision language models (VLMs) on standard, objective visual tasks of counting and identification. We find that state-of-the-art VLMs are strongly biased (e.g., unable to recognize the 4th stripe has been added to a 3-stripe Adidas logo) scoring an average of 17.05% accuracy in counting (e.g., counting stripes in an Adidas-like logo) across 7 diverse domains from animals, logos, chess, board games, optical illusions, to patterned grids. Removing image backgrounds nearly doubles accuracy (21.09 percentage points), revealing that contextual visual cues trigger these biased responses. Further analysis of VLMs' reasoning patterns shows that counting accuracy initially rises with thinking tokens, reaching ~40%, before declining with excessive reasoning. Our work presents an interesting failure mode in VLMs and a human-supervised automated framework for testing VLM biases. Code and data are available at: vlmsarebiased.github.io.
>
---
#### [replaced 072] Multivariate Variational Autoencoder
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07472v2](https://arxiv.org/pdf/2511.07472v2)**

> **作者:** Mehmet Can Yavuz
>
> **摘要:** Learning latent representations that are simultaneously expressive, geometrically well-structured, and reliably calibrated remains a central challenge for Variational Autoencoders (VAEs). Standard VAEs typically assume a diagonal Gaussian posterior, which simplifies optimization but rules out correlated uncertainty and often yields entangled or redundant latent dimensions. We introduce the Multivariate Variational Autoencoder (MVAE), a tractable full-covariance extension of the VAE that augments the encoder with sample-specific diagonal scales and a global coupling matrix. This induces a multivariate Gaussian posterior of the form $N(μ_φ(x), C \operatorname{diag}(σ_φ^2(x)) C^\top)$, enabling correlated latent factors while preserving a closed-form KL divergence and a simple reparameterization path. Beyond likelihood, we propose a multi-criterion evaluation protocol that jointly assesses reconstruction quality (MSE, ELBO), downstream discrimination (linear probes), probabilistic calibration (NLL, Brier, ECE), and unsupervised structure (NMI, ARI). Across Larochelle-style MNIST variants, Fashion-MNIST, and CIFAR-10/100, MVAE consistently matches or outperforms diagonal-covariance VAEs of comparable capacity, with particularly notable gains in calibration and clustering metrics at both low and high latent dimensions. Qualitative analyses further show smoother, more semantically coherent latent traversals and sharper reconstructions. All code, dataset splits, and evaluation utilities are released to facilitate reproducible comparison and future extensions of multivariate posterior models.
>
---
#### [replaced 073] Proxy-Tuning: Tailoring Multimodal Autoregressive Models for Subject-Driven Image Generation
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2503.10125v2](https://arxiv.org/pdf/2503.10125v2)**

> **作者:** Yi Wu; Shengju Qian; Lingting Zhu; Lei Liu; Wandi Qiao; Ziqiang Li; Lequan Yu; Bin Li
>
> **摘要:** Multimodal autoregressive (AR) models, based on next-token prediction and transformer architecture, have demonstrated remarkable capabilities in various multimodal tasks including text-to-image (T2I) generation. Despite their strong performance in general T2I tasks, our research reveals that these models initially struggle with subject-driven image generation compared to dominant diffusion models. To address this limitation, we introduce Proxy-Tuning, leveraging diffusion models to enhance AR models' capabilities in subject-specific image generation. Our method reveals a striking weak-to-strong phenomenon: fine-tuned AR models consistently outperform their diffusion model supervisors in both subject fidelity and prompt adherence. We analyze this performance shift and identify scenarios where AR models excel, particularly in multi-subject compositions and contextual understanding. This work not only demonstrates impressive results in subject-driven AR image generation, but also unveils the potential of weak-to-strong generalization in the image generation domain, contributing to a deeper understanding of different architectures' strengths and limitations.
>
---
#### [replaced 074] DenoiseGS: Gaussian Reconstruction Model for Burst Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22939v2](https://arxiv.org/pdf/2511.22939v2)**

> **作者:** Yongsen Cheng; Yuanhao Cai; Yulun Zhang
>
> **备注:** Update Abstract
>
> **摘要:** Burst denoising methods are crucial for enhancing images captured on handheld devices, but they often struggle with large motion or suffer from prohibitive computational costs. In this paper, we propose DenoiseGS, the first framework to leverage the efficiency of 3D Gaussian Splatting for burst denoising. Our approach addresses two key challenges when applying feedforward Gaussian reconsturction model to noisy inputs: the degradation of Gaussian point clouds and the loss of fine details. To this end, we propose a Gaussian self-consistency (GSC) loss, which regularizes the geometry predicted from noisy inputs with high-quality Gaussian point clouds. These point clouds are generated from clean inputs by the same model that we are training, thereby alleviating potential bias or domain gaps. Additionally, we introduce a log-weighted frequency (LWF) loss to strengthen supervision within the spectral domain, effectively preserving fine-grained details. The LWF loss adaptively weights frequency discrepancies in a logarithmic manner, emphasizing challenging high-frequency details. Extensive experiments demonstrate that DenoiseGS significantly exceeds the state-of-the-art NeRF-based methods on both burst denoising and novel view synthesis under noisy conditions, while achieving 250$\times$ faster inference speed. Code and models are released at https://github.com/yscheng04/DenoiseGS.
>
---
#### [replaced 075] HSR-KAN: Efficient Hyperspectral Image Super-Resolution via Kolmogorov-Arnold Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.06705v3](https://arxiv.org/pdf/2409.06705v3)**

> **作者:** Baisong Li; Xingwang Wang; Haixiao Xu
>
> **摘要:** Hyperspectral images (HSIs) have great potential in various visual tasks due to their rich spectral information. However, obtaining high-resolution hyperspectral images remains challenging due to limitations of physical imaging. Inspired by Kolmogorov-Arnold Networks (KANs), we propose an efficient HSI super-resolution (HSI-SR) model to fuse a low-resolution HSI (LR-HSI) and a high-resolution multispectral image (HR-MSI), yielding a high-resolution HSI (HR-HSI). To achieve the effective integration of spatial information from HR-MSI, we design a fusion module based on KANs, called KAN-Fusion. Further inspired by the channel attention mechanism, we design a spectral channel attention module called KAN Channel Attention Block (KAN-CAB) for post-fusion feature extraction. As a channel attention module integrated with KANs, KAN-CAB not only enhances the fine-grained adjustment ability of deep networks, enabling networks to accurately simulate details of spectral sequences and spatial textures, but also effectively avoid Curse of Dimensionality. Extensive experiments show that, compared to current state-of-the-art HSI-SR methods, proposed HSR-KAN achieves the best performance in terms of both qualitative and quantitative assessments. Our code is available at: https://github.com/Baisonm-Li/HSR-KAN.
>
---
#### [replaced 076] MoEGCL: Mixture of Ego-Graphs Contrastive Representation Learning for Multi-View Clustering
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.05876v3](https://arxiv.org/pdf/2511.05876v3)**

> **作者:** Jian Zhu; Xin Zou; Jun Sun; Cheng Luo; Lei Liu; Lingfang Zeng; Ning Zhang; Bian Wu; Chang Tang; Lirong Dai
>
> **摘要:** In recent years, the advancement of Graph Neural Networks (GNNs) has significantly propelled progress in Multi-View Clustering (MVC). However, existing methods face the problem of coarse-grained graph fusion. Specifically, current approaches typically generate a separate graph structure for each view and then perform weighted fusion of graph structures at the view level, which is a relatively rough strategy. To address this limitation, we present a novel Mixture of Ego-Graphs Contrastive Representation Learning (MoEGCL). It mainly consists of two modules. In particular, we propose an innovative Mixture of Ego-Graphs Fusion (MoEGF), which constructs ego graphs and utilizes a Mixture-of-Experts network to implement fine-grained fusion of ego graphs at the sample level, rather than the conventional view-level fusion. Additionally, we present the Ego Graph Contrastive Learning (EGCL) module to align the fused representation with the view-specific representation. The EGCL module enhances the representation similarity of samples from the same cluster, not merely from the same sample, further boosting fine-grained graph representation. Extensive experiments demonstrate that MoEGCL achieves state-of-the-art results in deep multi-view clustering tasks. The source code is publicly available at https://github.com/HackerHyper/MoEGCL.
>
---
#### [replaced 077] VITA: Vision-to-Action Flow Matching Policy
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VITA，一种视觉到动作的无噪声、无条件流匹配策略框架。针对传统方法需反复引入视觉信息导致效率低的问题，VITA直接从视觉表征映射到动作潜空间，通过动作自编码器对齐维度并防止潜空间坍塌，实现更快推理与更优性能，在仿真与真实机器人任务中均表现优异。**

- **链接: [https://arxiv.org/pdf/2507.13231v3](https://arxiv.org/pdf/2507.13231v3)**

> **作者:** Dechen Gao; Boqi Zhao; Andrew Lee; Ian Chuang; Hanchu Zhou; Hang Wang; Zhe Zhao; Junshan Zhang; Iman Soltani
>
> **备注:** Project page: https://ucd-dare.github.io/VITA/ Code: https://github.com/ucd-dare/VITA
>
> **摘要:** Conventional flow matching and diffusion-based policies sample through iterative denoising from standard noise distributions (e.g., Gaussian), and require conditioning modules to repeatedly incorporate visual information during the generative process, incurring substantial time and memory overhead. To reduce the complexity, we develop VITA(VIsion-To-Action policy), a noise-free and conditioning-free flow matching policy learning framework that directly flows from visual representations to latent actions. Since the source of the flow is visually grounded, VITA eliminates the need of visual conditioning during generation. As expected, bridging vision and action is challenging, because actions are lower-dimensional, less structured, and sparser than visual representations; moreover, flow matching requires the source and target to have the same dimensionality. To overcome this, we introduce an action autoencoder that maps raw actions into a structured latent space aligned with visual latents, trained jointly with flow matching. To further prevent latent space collapse, we propose flow latent decoding, which anchors the latent generation process by backpropagating the action reconstruction loss through the flow matching ODE (ordinary differential equation) solving steps. We evaluate VITA on 9 simulation and 5 real-world tasks from ALOHA and Robomimic. VITA achieves 1.5x-2x faster inference compared to conventional methods with conditioning modules, while outperforming or matching state-of-the-art policies. Codes, datasets, and demos are available at our project page: https://ucd-dare.github.io/VITA/.
>
---
#### [replaced 078] Prediction of Distant Metastasis in Head and Neck Cancer Patients Using Tumor and Peritumoral Multi-Modal Deep Learning
- **分类: q-bio.QM; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.20469v2](https://arxiv.org/pdf/2508.20469v2)**

> **作者:** Nuo Tong; Changhao Liu; Zizhao Tang; Feifan Sun; Yingping Li; Shuiping Gou; Mei Shi
>
> **备注:** 23 pages, 6 figures, 7 tables. Nuo Tong and Changhao Liu contributed equally. Corresponding Authors: Shuiping Gou and Mei Shi
>
> **摘要:** Although the combined treatment of surgery, radiotherapy, chemotherapy, and emerging target therapy has significantly improved the outcomes of patients with head and neck cancer, distant metastasis remains the leading cause of treatment failure. In this study, we propose a deep learning-based multimodal framework integrating CT imaging, radiomics, and clinical data to predict metastasis risk in HNSCC. A total of 1497 patients were retrospectively analyzed. Tumor and organ masks were generated from pretreatment CT scans, from which a 3D Swin Transformer extracted deep imaging features, while 1562 radiomics features were reduced to 36 via correlation filtering and random forest selection. Clinical data (age, sex, smoking, and alcohol status) were encoded and fused with imaging features, and the multimodal representation was fed into a fully connected network for prediction. Five-fold cross-validation was used to assess performance via AUC, accuracy, sensitivity, and specificity. The multimodal model outperformed all single-modality baselines. The deep learning module alone achieved an AUC of 0.715, whereas multimodal fusion significantly improved performance (AUC = 0.803, ACC = 0.752, SEN = 0.730, SPE = 0.758). Stratified analyses confirmed good generalizability across tumor subtypes. Ablation experiments demonstrated complementary contributions from each modality, and the 3D Swin Transformer provided more robust representations than conventional architectures. This multimodal deep learning model enables accurate, non-invasive metastasis prediction in HNSCC and shows strong potential for individualized treatment planning.
>
---
#### [replaced 079] RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出RealWebAssist基准，针对长周期网页助手任务中真实用户指令的复杂性问题。旨在评估AI在多步骤、跨网站交互中理解模糊指令、追踪用户意图与状态的能力。通过收集真实用户序列指令数据集，揭示当前模型在语义理解与图形界面操作上的显著不足。**

- **链接: [https://arxiv.org/pdf/2504.10445v2](https://arxiv.org/pdf/2504.10445v2)**

> **作者:** Suyu Ye; Haojun Shi; Darren Shih; Hyokun Yun; Tanya Roosta; Tianmin Shu
>
> **备注:** Project Website: https://scai.cs.jhu.edu/projects/RealWebAssist/ Code: https://github.com/SCAI-JHU/RealWebAssist
>
> **摘要:** To achieve successful assistance with long-horizon web-based tasks, AI agents must be able to sequentially follow real-world user instructions over a long period. Unlike existing web-based agent benchmarks, sequential instruction following in the real world poses significant challenges beyond performing a single, clearly defined task. For instance, real-world human instructions can be ambiguous, require different levels of AI assistance, and may evolve over time, reflecting changes in the user's mental state. To address this gap, we introduce RealWebAssist, a novel benchmark designed to evaluate sequential instruction-following in realistic scenarios involving long-horizon interactions with the web, visual GUI grounding, and understanding ambiguous real-world user instructions. RealWebAssist includes a dataset of sequential instructions collected from real-world human users. Each user instructs a web-based assistant to perform a series of tasks on multiple websites. A successful agent must reason about the true intent behind each instruction, keep track of the mental state of the user, understand user-specific routines, and ground the intended tasks to actions on the correct GUI elements. Our experimental results show that state-of-the-art models struggle to understand and ground user instructions, posing critical challenges in following real-world user instructions for long-horizon web assistance.
>
---
#### [replaced 080] SynPlay: Large-Scale Synthetic Human Data with Real-World Diversity for Aerial-View Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.11814v2](https://arxiv.org/pdf/2408.11814v2)**

> **作者:** Jinsub Yim; Hyungtae Lee; Sungmin Eum; Yi-Ting Shen; Yan Zhang; Heesung Kwon; Shuvra S. Bhattacharyya
>
> **备注:** Project Page: https://synplaydataset.github.io/
>
> **摘要:** We introduce SynPlay, a large-scale synthetic human dataset purpose-built for advancing multi-perspective human localization, with a predominant focus on aerial-view perception. SynPlay departs from traditional synthetic datasets by addressing a critical but underexplored challenge: localizing humans in aerial scenes where subjects often occupy only tens of pixels in the image. In such scenarios, fine-grained details like facial features or textures become irrelevant, shifting the burden of recognition to human motion, behavior, and interactions. To meet this need, SynPlay implements a novel rule-guided motion generation framework that combines real-world motion capture with motion evolution graphs. This design enables human actions to evolve dynamically through high-level game rules rather than predefined scripts, resulting in effectively uncountable motion variations. Unlike existing synthetic datasets-which either focus on static visual traits or reuse a limited set of mocap-driven actions-SynPlay captures a wide spectrum of spontaneous behaviors, including complex interactions that naturally emerge from unscripted gameplay scenarios. SynPlay also introduces an extensive multi-camera setup that spans UAVs at random altitudes, CCTVs, and a freely roaming UGV, achieving true near-to-far perspective coverage in a single dataset. The majority of instances are captured from aerial viewpoints at varying scales, directly supporting the development of models for long-range human analysis-a setting where existing datasets fall short. Our data contains over 73k images and 6.5M human instances, with detailed annotations for detection, segmentation, and keypoint tasks. Extensive experiments demonstrate that training with SynPlay significantly improves human localization performance, especially in few-shot and data-scarce scenarios.
>
---
#### [replaced 081] LORE: Lagrangian-Optimized Robust Embeddings for Visual Encoders
- **分类: cs.LG; cs.AI; cs.CV; math.OC**

- **链接: [https://arxiv.org/pdf/2505.18884v2](https://arxiv.org/pdf/2505.18884v2)**

> **作者:** Borna Khodabandeh; Amirabbas Afzali; Amirhossein Afsharrad; Seyed Shahabeddin Mousavi; Sanjay Lall; Sajjad Amini; Seyed-Mohsen Moosavi-Dezfooli
>
> **摘要:** Visual encoders have become fundamental components in modern computer vision pipelines. However, ensuring robustness against adversarial perturbations remains a critical challenge. Recent efforts have explored both supervised and unsupervised adversarial fine-tuning strategies. We identify two key limitations in these approaches: (i) they often suffer from instability, especially during the early stages of fine-tuning, resulting in suboptimal convergence and degraded performance on clean data, and (ii) they exhibit a suboptimal trade-off between robustness and clean data accuracy, hindering the simultaneous optimization of both objectives. To overcome these challenges, we propose Lagrangian-Optimized Robust Embeddings (LORE), a novel unsupervised adversarial fine-tuning framework. LORE utilizes constrained optimization, which offers a principled approach to balancing competing goals, such as improving robustness while preserving nominal performance. By enforcing embedding-space proximity constraints, LORE effectively maintains clean data performance throughout adversarial fine-tuning. Extensive experiments show that LORE significantly improves zero-shot adversarial robustness with minimal degradation in clean data accuracy. Furthermore, we demonstrate the effectiveness of the adversarially fine-tuned CLIP image encoder in out-of-distribution generalization and enhancing the interpretability of image embeddings.
>
---
#### [replaced 082] Harnessing Diffusion-Generated Synthetic Images for Fair Image Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08711v2](https://arxiv.org/pdf/2511.08711v2)**

> **作者:** Abhipsa Basu; Aviral Gupta; Abhijnya Bhat; R. Venkatesh Babu
>
> **备注:** Accepted to AAAI AISI Track, 2026
>
> **摘要:** Image classification systems often inherit biases from uneven group representation in training data. For example, in face datasets for hair color classification, blond hair may be disproportionately associated with females, reinforcing stereotypes. A recent approach leverages the Stable Diffusion model to generate balanced training data, but these models often struggle to preserve the original data distribution. In this work, we explore multiple diffusion-finetuning techniques, e.g., LoRA and DreamBooth, to generate images that more accurately represent each training group by learning directly from their samples. Additionally, in order to prevent a single DreamBooth model from being overwhelmed by excessive intra-group variations, we explore a technique of clustering images within each group and train a DreamBooth model per cluster. These models are then used to generate group-balanced data for pretraining, followed by fine-tuning on real data. Experiments on multiple benchmarks demonstrate that the studied finetuning approaches outperform vanilla Stable Diffusion on average and achieve results comparable to SOTA debiasing techniques like Group-DRO, while surpassing them as the dataset bias severity increases.
>
---
#### [replaced 083] Can World Simulators Reason? Gen-ViRe: A Generative Visual Reasoning Benchmark
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13853v2](https://arxiv.org/pdf/2511.13853v2)**

> **作者:** Xinxin Liu; Zhaopan Xu; Ming Li; Kai Wang; Yong Jae Lee; Yuzhang Shang
>
> **备注:** 10 pages
>
> **摘要:** While Chain-of-Thought (CoT) prompting enables sophisticated symbolic reasoning in LLMs, it remains confined to discrete text and cannot simulate the continuous, physics-governed dynamics of the real world. Recent video generation models have emerged as potential world simulators through Chain-of-Frames (CoF) reasoning -- materializing thought as frame-by-frame visual sequences, with each frame representing a physically-grounded reasoning step. Despite compelling demonstrations, a challenge persists: existing benchmarks, focusing on fidelity or alignment, do not assess CoF reasoning and thus cannot measure core cognitive abilities in multi-step planning, algorithmic logic, or abstract pattern extrapolation. This evaluation void prevents systematic understanding of model capabilities and principled guidance for improvement. We introduce Gen-ViRe (Generative Visual Reasoning Benchmark), a framework grounded in cognitive science and real-world AI applications, which decomposes CoF reasoning into six cognitive dimensions -- from perceptual logic to abstract planning -- and 24 subtasks. Through multi-source data curation, minimal prompting protocols, and hybrid VLM-assisted evaluation with detailed criteria, Gen-ViRe delivers the first quantitative assessment of video models as reasoners. Our experiments on SOTA systems reveal substantial discrepancies between impressive visual quality and actual reasoning depth, establishing baselines and diagnostic tools to advance genuine world simulators.
>
---
#### [replaced 084] AVFakeBench: A Comprehensive Audio-Video Forgery Detection Benchmark for AV-LMMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21251v2](https://arxiv.org/pdf/2511.21251v2)**

> **作者:** Shuhan Xia; Peipei Li; Xuannan Liu; Dongsen Zhang; Xinyu Guo; Zekun Li
>
> **备注:** The experimental results in this paper have been further improved and updated; the baseline results do not match existing results, therefore the paper needs to be retracted
>
> **摘要:** The threat of Audio-Video (AV) forgery is rapidly evolving beyond human-centric deepfakes to include more diverse manipulations across complex natural scenes. However, existing benchmarks are still confined to DeepFake-based forgeries and single-granularity annotations, thus failing to capture the diversity and complexity of real-world forgery scenarios. To address this, we introduce AVFakeBench, the first comprehensive audio-video forgery detection benchmark that spans rich forgery semantics across both human subject and general subject. AVFakeBench comprises 12K carefully curated audio-video questions, covering seven forgery types and four levels of annotations. To ensure high-quality and diverse forgeries, we propose a multi-stage hybrid forgery framework that integrates proprietary models for task planning with expert generative models for precise manipulation. The benchmark establishes a multi-task evaluation framework covering binary judgment, forgery types classification, forgery detail selection, and explanatory reasoning. We evaluate 11 Audio-Video Large Language Models (AV-LMMs) and 2 prevalent detection methods on AVFakeBench, demonstrating the potential of AV-LMMs as emerging forgery detectors while revealing their notable weaknesses in fine-grained perception and reasoning.
>
---
#### [replaced 085] TPCNet: Triple physical constraints for Low-light Image Enhancement
- **分类: cs.CV; physics.optics**

- **链接: [https://arxiv.org/pdf/2511.22052v2](https://arxiv.org/pdf/2511.22052v2)**

> **作者:** Jing-Yi Shi; Ming-Fei Li; Ling-An Wu
>
> **摘要:** Low-light image enhancement is an essential computer vision task to improve image contrast and to decrease the effects of color bias and noise. Many existing interpretable deep-learning algorithms exploit the Retinex theory as the basis of model design. However, previous Retinex-based algorithms, that consider reflected objects as ideal Lambertian ignore specular reflection in the modeling process and construct the physical constraints in image space, limiting generalization of the model. To address this issue, we preserve the specular reflection coefficient and reformulate the original physical constraints in the imaging process based on the Kubelka-Munk theory, thereby constructing constraint relationship between illumination, reflection, and detection, the so-called triple physical constraints (TPCs)theory. Based on this theory, the physical constraints are constructed in the feature space of the model to obtain the TPC network (TPCNet). Comprehensive quantitative and qualitative benchmark and ablation experiments confirm that these constraints effectively improve the performance metrics and visual quality without introducing new parameters, and demonstrate that our TPCNet outperforms other state-of-the-art methods on 10 datasets.
>
---
#### [replaced 086] SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.20405v2](https://arxiv.org/pdf/2504.20405v2)**

> **作者:** Sahil Sethi; Sai Reddy; Mansi Sakarvadia; Jordan Serotte; Darlington Nwaudo; Nicholas Maassen; Lewis Shi
>
> **备注:** This version of the article has been accepted for publication at Nature Partner Journal (NPJ) Artificial Intelligence, but is not the Version of Record and does not reflect post-acceptance improvements or any corrections. The Version of Record is available online at: http://dx.doi.org/10.1038/s44387-025-00043-5
>
> **摘要:** Deep learning has shown strong performance in musculoskeletal imaging, but prior work has largely targeted conditions where diagnosis is relatively straightforward. More challenging problems remain underexplored, such as detecting Bankart lesions (anterior-inferior glenoid labral tears) on standard MRIs. These lesions are difficult to diagnose due to subtle imaging features, often necessitating invasive MRI arthrograms (MRAs). We introduce ScopeMRI, the first publicly available, expert-annotated dataset for shoulder pathologies, and present a deep learning framework for Bankart lesion detection on both standard MRIs and MRAs. ScopeMRI contains shoulder MRIs from patients who underwent arthroscopy, providing ground-truth labels from intraoperative findings, the diagnostic gold standard. Separate models were trained for MRIs and MRAs using CNN- and transformer-based architectures, with predictions ensembled across multiple imaging planes. Our models achieved radiologist-level performance, with accuracy on standard MRIs surpassing radiologists interpreting MRAs. External validation on independent hospital data demonstrated initial generalizability across imaging protocols. By releasing ScopeMRI and a modular codebase for training and evaluation, we aim to accelerate research in musculoskeletal imaging and foster development of datasets and models that address clinically challenging diagnostic tasks.
>
---
#### [replaced 087] Contrastive Forward-Forward: A Training Algorithm of Vision Transformer
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2502.00571v2](https://arxiv.org/pdf/2502.00571v2)**

> **作者:** Hossein Aghagolzadeh; Mehdi Ezoji
>
> **备注:** Updated text and added pointer to the published Neural Networks version
>
> **摘要:** Although backpropagation is widely accepted as a training algorithm for artificial neural networks, researchers are always looking for inspiration from the brain to find ways with potentially better performance. Forward-Forward is a novel training algorithm that is more similar to what occurs in the brain, although there is a significant performance gap compared to backpropagation. In the Forward-Forward algorithm, the loss functions are placed after each layer, and the updating of a layer is done using two local forward passes and one local backward pass. Forward-Forward is in its early stages and has been designed and evaluated on simple multi-layer perceptron networks to solve image classification tasks. In this work, we have extended the use of this algorithm to a more complex and modern network, namely the Vision Transformer. Inspired by insights from contrastive learning, we have attempted to revise this algorithm, leading to the introduction of Contrastive Forward-Forward. Experimental results show that our proposed algorithm performs significantly better than the baseline Forward-Forward leading to an increase of up to 10% in accuracy and accelerating the convergence speed by 5 to 20 times. Furthermore, if we take Cross Entropy as the baseline loss function in backpropagation, it will be demonstrated that the proposed modifications to the baseline Forward-Forward reduce its performance gap compared to backpropagation on Vision Transformer, and even outperforms it in certain conditions, such as inaccurate supervision.
>
---
#### [replaced 088] Self-Supervised One-Step Diffusion Refinement for Snapshot Compressive Imaging
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2409.07417v2](https://arxiv.org/pdf/2409.07417v2)**

> **作者:** Shaoguang Huang; Yunzhen Wang; Haijin Zeng; Hongyu Chen; Hongyan Zhang
>
> **摘要:** Snapshot compressive imaging (SCI) captures multispectral images (MSIs) using a single coded two-dimensional (2-D) measurement, but reconstructing high-fidelity MSIs from these compressed inputs remains a fundamentally ill-posed challenge. While diffusion-based reconstruction methods have recently raised the bar for quality, they face critical limitations: a lack of large-scale MSI training data, adverse domain shifts from RGB-pretrained models, and inference inefficiencies due to multi-step sampling. These drawbacks restrict their practicality in real-world applications. In contrast to existing methods, which either follow costly iterative refinement or adapt subspace-based embeddings for diffusion models (e.g. DiffSCI, PSR-SCI), we introduce a fundamentally different paradigm: a self-supervised One-Step Diffusion (OSD) framework specifically designed for SCI. The key novelty lies in using a single-step diffusion refiner to correct an initial reconstruction, eliminating iterative denoising entirely while preserving generative quality. Moreover, we adopt a self-supervised equivariant learning strategy to train both the predictor and refiner directly from raw 2-D measurements, enabling generalization to unseen domains without the need for ground-truth MSI. To further address the challenge of limited MSI data, we design a band-selection-driven distillation strategy that transfers core generative priors from large-scale RGB datasets, effectively bridging the domain gap. Extensive experiments confirm that our approach sets a new benchmark, yielding PSNR gains of 3.44 dB, 1.61 dB, and 0.28 dB on the Harvard, NTIRE, and ICVL datasets, respectively, while reducing reconstruction time by 97.5%. This remarkable improvement in efficiency and adaptability makes our method a significant advancement in SCI reconstruction, combining both accuracy and practicality for real-world deployment.
>
---
#### [replaced 089] DualCamCtrl: Dual-Branch Diffusion Model for Geometry-Aware Camera-Controlled Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23127v2](https://arxiv.org/pdf/2511.23127v2)**

> **作者:** Hongfei Zhang; Kanghao Chen; Zixin Zhang; Harold Haodong Chen; Yuanhuiyi Lyu; Yuqi Zhang; Shuai Yang; Kun Zhou; Yingcong Chen
>
> **摘要:** This paper presents DualCamCtrl, a novel end-to-end diffusion model for camera-controlled video generation. Recent works have advanced this field by representing camera poses as ray-based conditions, yet they often lack sufficient scene understanding and geometric awareness. DualCamCtrl specifically targets this limitation by introducing a dual-branch framework that mutually generates camera-consistent RGB and depth sequences. To harmonize these two modalities, we further propose the Semantic Guided Mutual Alignment (SIGMA) mechanism, which performs RGB-depth fusion in a semantics-guided and mutually reinforced manner. These designs collectively enable DualCamCtrl to better disentangle appearance and geometry modeling, generating videos that more faithfully adhere to the specified camera trajectories. Additionally, we analyze and reveal the distinct influence of depth and camera poses across denoising stages and further demonstrate that early and late stages play complementary roles in forming global structure and refining local details. Extensive experiments demonstrate that DualCamCtrl achieves more consistent camera-controlled video generation, with over 40\% reduction in camera motion errors compared with prior methods. Our project page: https://soyouthinkyoucantell.github.io/dualcamctrl-page/
>
---
#### [replaced 090] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2501.05264v4](https://arxiv.org/pdf/2501.05264v4)**

> **作者:** Mengshi Qi; Jiaxuan Peng; Xianlin Zhang; Huadong Ma
>
> **摘要:** 3D human pose estimation (3D HPE) has emerged as a prominent research topic, particularly in the realm of RGB-based methods. However, the use of RGB images is often limited by issues such as occlusion and privacy constraints. Consequently, multi-modal sensing, which leverages non-intrusive sensors, is gaining increasing attention. Nevertheless, multi-modal 3D HPE still faces challenges, including modality imbalance. In this work, we introduce a novel balanced multi-modal learning method for 3D HPE, which harnesses the power of RGB, LiDAR, mmWave, and WiFi. Specifically, we propose a Shapley value-based contribution algorithm to assess the contribution of each modality and detect modality imbalance. To address this imbalance, we design a modality learning regulation strategy that decelerates the learning process during the early stages of training. We conduct extensive experiments on the widely adopted multi-modal dataset, MM-Fi, demonstrating the superiority of our approach in enhancing 3D pose estimation under complex conditions. We will release our codes soon.
>
---
#### [replaced 091] DynaStride: Dynamic Stride Windowing with MMCoT for Instructional Multi-Scene Captioning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.23907v2](https://arxiv.org/pdf/2510.23907v2)**

> **作者:** Eddison Pham; Prisha Priyadarshini; Adrian Maliackel; Kanishk Bandi; Cristian Meo; Kevin Zhu
>
> **备注:** 16 pages, 15 figures, 5 Tables, Accepted at NeurIPS 7HVU Workshop, Accepted at AAAI AI4ED Workshop
>
> **摘要:** Scene-level captioning in instructional videos can enhance learning by requiring an understanding of both visual cues and temporal structure. By aligning visual cues with textual guidance, this understanding supports procedural learning and multimodal reasoning, providing a richer context for skill acquisition. However, captions that fail to capture this structure may lack coherence and quality, which can create confusion and undermine the video's educational intent. To address this gap, we introduce DynaStride, a pipeline to generate coherent, scene-level captions without requiring manual scene segmentation. Using the YouCookII dataset's scene annotations, DynaStride performs adaptive frame sampling and multimodal windowing to capture key transitions within each scene. It then employs a multimodal chain-of-thought process to produce multiple action-object pairs, which are refined and fused using a dynamic stride window selection algorithm that adaptively balances temporal context and redundancy. The final scene-level caption integrates visual semantics and temporal reasoning in a single instructional caption. Empirical evaluations against strong baselines, including VLLaMA3 and GPT-4o, demonstrate consistent gains on both N-gram-based metrics (BLEU, METEOR) and semantic similarity measures (BERTScore, CLIPScore). Qualitative analyses further show that DynaStride produces captions that are more temporally coherent and informative, suggesting a promising direction for improving AI-powered instructional content generation.
>
---
#### [replaced 092] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.12263v2](https://arxiv.org/pdf/2511.12263v2)**

> **作者:** Jingyao Li; Jingyun Wang; Molin Tan; Haochen Wang; Cilin Yan; Likun Shi; Jiayin Cai; Xiaolong Jiang; Yao Hu
>
> **备注:** Accepted to AAAI 2026 (main track). For code and data, see https://github.com/chuntianli666/CrossVid
>
> **摘要:** Cross-Video Reasoning (CVR) presents a significant challenge in video understanding, which requires simultaneous understanding of multiple videos to aggregate and compare information across groups of videos. Most existing video understanding benchmarks focus on single-video analysis, failing to assess the ability of multimodal large language models (MLLMs) to simultaneously reason over various videos. Recent benchmarks evaluate MLLMs' capabilities on multi-view videos that capture different perspectives of the same scene. However, their limited tasks hinder a thorough assessment of MLLMs in diverse real-world CVR scenarios. To this end, we introduce CrossVid, the first benchmark designed to comprehensively evaluate MLLMs' spatial-temporal reasoning ability in cross-video contexts. Firstly, CrossVid encompasses a wide spectrum of hierarchical tasks, comprising four high-level dimensions and ten specific tasks, thereby closely reflecting the complex and varied nature of real-world video understanding. Secondly, CrossVid provides 5,331 videos, along with 9,015 challenging question-answering pairs, spanning single-choice, multiple-choice, and open-ended question formats. Through extensive experiments on various open-source and closed-source MLLMs, we observe that Gemini-2.5-Pro performs best on CrossVid, achieving an average accuracy of 50.4%. Notably, our in-depth case study demonstrates that most current MLLMs struggle with CVR tasks, primarily due to their inability to integrate or compare evidence distributed across multiple videos for reasoning. These insights highlight the potential of CrossVid to guide future advancements in enhancing MLLMs' CVR capabilities.
>
---
#### [replaced 093] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.12170v2](https://arxiv.org/pdf/2511.12170v2)**

> **作者:** Wang Luo; Di Wu; Hengyuan Na; Yinlin Zhu; Miao Hu; Guocong Quan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Point cloud completion aims to reconstruct complete 3D shapes from partial observations, which is a challenging problem due to severe occlusions and missing geometry. Despite recent advances in multimodal techniques that leverage complementary RGB images to compensate for missing geometry, most methods still follow a Completion-by-Inpainting paradigm, synthesizing missing structures from fused latent features. We empirically show that this paradigm often results in structural inconsistencies and topological artifacts due to limited geometric and semantic constraints. To address this, we rethink the task and propose a more robust paradigm, termed Completion-by-Correction, which begins with a topologically complete shape prior generated by a pretrained image-to-3D model and performs feature-space correction to align it with the partial observation. This paradigm shifts completion from unconstrained synthesis to guided refinement, enabling structurally consistent and observation-aligned reconstruction. Building upon this paradigm, we introduce PGNet, a multi-stage framework that conducts dual-feature encoding to ground the generative prior, synthesizes a coarse yet structurally aligned scaffold, and progressively refines geometric details via hierarchical correction. Experiments on the ShapeNetViPC dataset demonstrate the superiority of PGNet over state-of-the-art baselines in terms of average Chamfer Distance (-23.5%) and F-score (+7.1%).
>
---
#### [replaced 094] Video Anomaly Detection with Semantics-Aware Information Bottleneck
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.02535v3](https://arxiv.org/pdf/2506.02535v3)**

> **作者:** Juntong Li; Lingwei Dang; Qingxin Xiao; Shishuo Shang; Jiajia Cheng; Haomin Wu; Yun Hao; Qingyao Wu
>
> **摘要:** Semi-supervised video anomaly detection methods face two critical challenges: (1) Strong generalization blurs the boundary between normal and abnormal patterns. Although existing approaches attempt to alleviate this issue using memory modules, their rigid prototype-matching process limits adaptability to diverse scenarios; (2) Relying solely on low-level appearance and motion cues makes it difficult to perceive high-level semantic anomalies in complex scenes. To address these limitations, we propose SIB-VAD, a novel framework based on adaptive information bottleneck filtering and semantic-aware enhancement. We propose the Sparse Feature Filtering Module (SFFM) to replace traditional memory modules. It compresses normal features directly into a low-dimensional manifold based on the information bottleneck principle and uses an adaptive routing mechanism to dynamically select the most suitable normal bottleneck subspace. Trained only on normal data, SFFMs only learn normal low-dimensional manifolds, while abnormal features deviate and are effectively filtered. Unlike memory modules, SFFM directly removes abnormal information and adaptively handles scene variations. To improve semantic awareness, we further design a multimodal prediction framework that jointly models appearance, motion, and semantics. Through multimodal consistency constraints and joint error computation, it achieves more robust VAD performance. Experimental results validate the effectiveness of our feature filtering paradigm based on semantics-aware information bottleneck. Project page at https://qzfm.github.io/sib_vad_project_page/
>
---
#### [replaced 095] VoQA: Visual-only Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.14227v2](https://arxiv.org/pdf/2505.14227v2)**

> **作者:** Jianing An; Luyang Jiang; Jie Luo; Wenjun Wu; Lei Huang
>
> **备注:** 21 pages
>
> **摘要:** Visual understanding requires interpreting both natural scenes and the textual information that appears within them, motivating tasks such as Visual Question Answering (VQA). However, current VQA benchmarks overlook scenarios with visually embedded questions, whereas advanced agents should be able to see the question without separate text input as humans. We introduce Visual-only Question Answering (VoQA), where both the scene and the question appear within a single image, requiring models to perceive and reason purely through vision. This setting supports more realistic visual understanding and interaction in scenarios where questions or instructions are embedded directly in the visual scene. Evaluations under pure visual-only zero-shot, prompt-guided and OCR-assisted settings show that current models exhibit a clear performance drop compared to traditional VQA. To address this, we investigate question-alignment fine-tuning strategies designed to guide models toward interpreting the visual question prior to reasoning. Leveraging VoQA dataset together with these strategies yields robust vision-only reasoning while preserving cross-task generalization to traditional VQA, reflecting the complementary visual and textual reasoning capabilities fostered through VoQA training. The code and data are publicly available.
>
---
#### [replaced 096] Flow Equivariant Recurrent Neural Networks
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.14793v2](https://arxiv.org/pdf/2507.14793v2)**

> **作者:** T. Anderson Keller
>
> **备注:** NeurIPS '25, Spotlight
>
> **摘要:** Data arrives at our senses as a continuous stream, smoothly transforming from one instant to the next. These smooth transformations can be viewed as continuous symmetries of the environment that we inhabit, defining equivalence relations between stimuli over time. In machine learning, neural network architectures that respect symmetries of their data are called equivariant and have provable benefits in terms of generalization ability and sample efficiency. To date, however, equivariance has been considered only for static transformations and feed-forward networks, limiting its applicability to sequence models, such as recurrent neural networks (RNNs), and corresponding time-parameterized sequence transformations. In this work, we extend equivariant network theory to this regime of 'flows' -- one-parameter Lie subgroups capturing natural transformations over time, such as visual motion. We begin by showing that standard RNNs are generally not flow equivariant: their hidden states fail to transform in a geometrically structured manner for moving stimuli. We then show how flow equivariance can be introduced, and demonstrate that these models significantly outperform their non-equivariant counterparts in terms of training speed, length generalization, and velocity generalization, on both next step prediction and sequence classification. We present this work as a first step towards building sequence models that respect the time-parameterized symmetries which govern the world around us.
>
---
#### [replaced 097] VeriSciQA: An Auto-Verified Dataset for Scientific Visual Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19899v2](https://arxiv.org/pdf/2511.19899v2)**

> **作者:** Yuyi Li; Daoyuan Chen; Zhen Wang; Yutong Lu; Yaliang Li
>
> **摘要:** Large Vision-Language Models (LVLMs) show promise for scientific applications, yet open-source models still struggle with Scientific Visual Question Answering (SVQA), namely answering questions about figures from scientific papers. A key bottleneck lies in the lack of public, large-scale, high-quality SVQA datasets. Although recent work uses LVLMs to synthesize data at scale, we identify systematic errors in their resulting QA pairs, stemming from LVLMs' inherent limitations and information asymmetry between figures and text. To address these challenges, we propose a verification-centric Generate-then-Verify framework that first generates QA pairs with figure-associated textual context, then applies cross-modal consistency checks against figures along with auxiliary filters to eliminate erroneous pairs. We instantiate this framework to curate VeriSciQA, a dataset of 20,351 QA pairs spanning 20 scientific domains and 12 figure types. VeriSciQA poses a challenging benchmark for open-source models, with a substantial accuracy gap between the leading open-source models (64%) and a proprietary model (82%). Moreover, models fine-tuned on VeriSciQA achieve consistent improvements on SVQA benchmarks, with performance gains that scale with data size and surpass models trained on existing datasets. Human evaluation further validates the superior correctness of VeriSciQA. Together, these evidences demonstrate that continued data expansion by our scalable framework can further advance SVQA capability in the open-source community.
>
---
#### [replaced 098] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.16177v2](https://arxiv.org/pdf/2503.16177v2)**

> **作者:** Shiyong Liu; Xiao Tang; Zhihao Li; Yingfan He; Chongjie Ye; Jianzhuang Liu; Binxiao Huang; Shunbo Zhou; Xiaofei Wu
>
> **备注:** Accepted to ICCV 2025. Project website: https://occlugaussian.github.io
>
> **摘要:** In large-scale scene reconstruction using 3D Gaussian splatting, it is common to partition the scene into multiple smaller regions and reconstruct them individually. However, existing division methods are occlusion-agnostic, meaning that each region may contain areas with severe occlusions. As a result, the cameras within those regions are less correlated, leading to a low average contribution to the overall reconstruction. In this paper, we propose an occlusion-aware scene division strategy that clusters training cameras based on their positions and co-visibilities to acquire multiple regions. Cameras in such regions exhibit stronger correlations and a higher average contribution, facilitating high-quality scene reconstruction. We further propose a region-based rendering technique to accelerate large scene rendering, which culls Gaussians invisible to the region where the viewpoint is located. Such a technique significantly speeds up the rendering without compromising quality. Extensive experiments on multiple large scenes show that our method achieves superior reconstruction results with faster rendering speed compared to existing state-of-the-art approaches. Project page: https://occlugaussian.github.io.
>
---
#### [replaced 099] IMSE: Efficient U-Net-based Speech Enhancement using Inception Depthwise Convolution and Amplitude-Aware Linear Attention
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文针对资源受限设备上的语音增强任务，提出IMSE模型。针对现有方法MUSE的效率瓶颈，创新性地引入幅度感知线性注意力（MALA）和因式分解深度卷积（IDConv），在减少16.8%参数量的同时保持优异语音质量，实现了轻量化与高性能的平衡。**

- **链接: [https://arxiv.org/pdf/2511.14515v2](https://arxiv.org/pdf/2511.14515v2)**

> **作者:** Xinxin Tang; Bin Qin; Yufang Li
>
> **摘要:** Achieving a balance between lightweight design and high performance remains a significant challenge for speech enhancement (SE) tasks on resource-constrained devices. Existing state-of-the-art methods, such as MUSE, have established a strong baseline with only 0.51M parameters by introducing a Multi-path Enhanced Taylor (MET) transformer and Deformable Embedding (DE). However, an in-depth analysis reveals that MUSE still suffers from efficiency bottlenecks: the MET module relies on a complex "approximate-compensate" mechanism to mitigate the limitations of Taylor-expansion-based attention, while the offset calculation for deformable embedding introduces additional computational burden. This paper proposes IMSE, a systematically optimized and ultra-lightweight network. We introduce two core innovations: 1) Replacing the MET module with Amplitude-Aware Linear Attention (MALA). MALA fundamentally rectifies the "amplitude-ignoring" problem in linear attention by explicitly preserving the norm information of query vectors in the attention calculation, achieving efficient global modeling without an auxiliary compensation branch. 2) Replacing the DE module with Inception Depthwise Convolution (IDConv). IDConv borrows the Inception concept, decomposing large-kernel operations into efficient parallel branches (square, horizontal, and vertical strips), thereby capturing spectrogram features with extremely low parameter redundancy. Extensive experiments on the VoiceBank+DEMAND dataset demonstrate that, compared to the MUSE baseline, IMSE significantly reduces the parameter count by 16.8\% (from 0.513M to 0.427M) while achieving competitive performance comparable to the state-of-the-art on the PESQ metric (3.373). This study sets a new benchmark for the trade-off between model size and speech quality in ultra-lightweight speech enhancement.
>
---
#### [replaced 100] Dynamic Multimodal Prototype Learning in Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.03657v2](https://arxiv.org/pdf/2507.03657v2)**

> **作者:** Xingyu Zhu; Shuo Wang; Beier Zhu; Miaoge Li; Yunfan Li; Junfeng Fang; Zhicai Wang; Dongsheng Wang; Hanwang Zhang
>
> **备注:** ICCV 2025
>
> **摘要:** With the increasing attention to pre-trained vision-language models (VLMs), \eg, CLIP, substantial efforts have been devoted to many downstream tasks, especially in test-time adaptation (TTA). However, previous works focus on learning prototypes only in the textual modality while overlooking the ambiguous semantics in class names. These ambiguities lead to textual prototypes that are insufficient to capture visual concepts, resulting in limited performance. To address this issue, we introduce \textbf{ProtoMM}, a training-free framework that constructs multimodal prototypes to adapt VLMs during the test time. By viewing the prototype as a discrete distribution over the textual descriptions and visual particles, ProtoMM has the ability to combine the multimodal features for comprehensive prototype learning. More importantly, the visual particles are dynamically updated as the testing stream flows. This allows our multimodal prototypes to continually learn from the data, enhancing their generalizability in unseen scenarios. In addition, we quantify the importance of the prototypes and test images by formulating their semantic distance as an optimal transport problem. Extensive experiments on 15 zero-shot benchmarks demonstrate the effectiveness of our method, achieving a 1.03\% average accuracy improvement over state-of-the-art methods on ImageNet and its variant datasets.
>
---
#### [replaced 101] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对信息密集型图像的视觉问答任务，解决复杂布局中关键线索定位难、多跳推理效率低的问题。提出无需训练的Speculative Verdict框架，通过轻量级草稿专家生成多样推理路径，由强模型综合并筛选高共识路径，实现高效准确的答案生成。**

- **链接: [https://arxiv.org/pdf/2510.20812v2](https://arxiv.org/pdf/2510.20812v2)**

> **作者:** Yuhan Liu; Lianhui Qin; Shengjie Wang
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable progress in multimodal understanding, yet they struggle when reasoning over information-intensive images that densely interleave textual annotations with fine-grained graphical elements. The main challenges lie in precisely localizing critical cues in dense layouts and multi-hop reasoning to integrate dispersed evidence. We propose Speculative Verdict (SV), a training-free framework inspired by speculative decoding that combines multiple lightweight draft experts with a large verdict model. In the draft stage, small VLMs act as draft experts to generate reasoning paths that provide diverse localization candidates; in the verdict stage, a strong VLM synthesizes these paths to produce the final answer, minimizing computational cost while recovering correct answers. To further improve efficiency and accuracy, SV introduces a consensus expert selection mechanism that forwards only high-agreement reasoning paths to the verdict. Empirically, SV achieves consistent gains on challenging information-intensive and high-resolution visual question answering benchmarks, including InfographicVQA, ChartMuseum, ChartQAPro, and HR-Bench 4K. By synthesizing correct insights from multiple partially accurate reasoning paths, SV achieves both error correction and cost-efficiency compared to large proprietary models or training pipelines. Code is available at https://github.com/Tinaliu0123/speculative-verdict.
>
---
#### [replaced 102] GuideGen: A Text-Guided Framework for Paired Full-torso Anatomy and CT Volume Generation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2403.07247v3](https://arxiv.org/pdf/2403.07247v3)**

> **作者:** Linrui Dai; Rongzhao Zhang; Yongrui Yu; Xiaofan Zhang
>
> **备注:** accepted as AAAI 2026 poster
>
> **摘要:** The recently emerging conditional diffusion models seem promising for mitigating the labor and expenses in building large 3D medical imaging datasets. However, previous studies on 3D CT generation primarily focus on specific organs characterized by a local structure and fixed contrast and have yet to fully capitalize on the benefits of both semantic and textual conditions. In this paper, we present GuideGen, a controllable framework based on easily-acquired text prompts to generate anatomical masks and corresponding CT volumes for the entire torso-from chest to pelvis. Our approach includes three core components: a text-conditional semantic synthesizer for creating realistic full-torso anatomies; an anatomy-aware high-dynamic-range (HDR) autoencoder for high-fidelity feature extraction across varying intensity levels; and a latent feature generator that ensures alignment between CT images, anatomical semantics and input prompts. Combined, these components enable data synthesis for segmentation tasks from only textual instructions. To train and evaluate GuideGen, we compile a multi-modality cancer imaging dataset with paired CT and clinical descriptions from 12 public TCIA datasets and one private real-world dataset. Comprehensive evaluations across generation quality, cross-modality alignment, and data usability on multi-organ and tumor segmentation tasks demonstrate GuideGen's superiority over existing CT generation methods. Relevant materials are available at https://github.com/OvO1111/GuideGen.
>
---
#### [replaced 103] Efficient Generative Adversarial Networks for Color Document Image Enhancement and Binarization Using Multi-scale Feature Extraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.04231v2](https://arxiv.org/pdf/2407.04231v2)**

> **作者:** Rui-Yang Ju; KokSheik Wong; Jen-Shiun Chiang
>
> **备注:** Accepted to APSIPA ASC 2025
>
> **摘要:** The outcome of text recognition for degraded color documents is often unsatisfactory due to interference from various contaminants. To extract information more efficiently for text recognition, document image enhancement and binarization are often employed as preliminary steps in document analysis. Training independent generative adversarial networks (GANs) for each color channel can generate images where shadows and noise are effectively removed, which subsequently allows for efficient text information extraction. However, employing multiple GANs for different color channels requires long training and inference times. To reduce both the training and inference times of these preliminary steps, we propose an efficient method based on multi-scale feature extraction, which incorporates Haar wavelet transformation and normalization to process document images before submitting them to GANs for training. Experiment results show that our proposed method significantly reduces both the training and inference times while maintaining comparable performances when benchmarked against the state-of-the-art methods. In the best case scenario, a reduction of 10% and 26% are observed for training and inference times, respectively, while maintaining the model performance at 73.79 of Average-Score metric. The implementation of this work is available at https://github.com/RuiyangJu/Efficient_Document_Image_Binarization.
>
---
#### [replaced 104] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.07710v3](https://arxiv.org/pdf/2511.07710v3)**

> **作者:** Jiale Liu; Haoming Zhou; Yishu Liu; Bingzhi Chen; Yuncheng Jiang
>
> **备注:** 10 pages, 6 figures, accepted by AAAI 2026
>
> **摘要:** Fine-grained image-text alignment is a pivotal challenge in multimodal learning, underpinning key applications such as visual question answering, image captioning, and vision-language navigation. Unlike global alignment, fine-grained alignment requires precise correspondence between localized visual regions and textual tokens, often hindered by noisy attention mechanisms and oversimplified modeling of cross-modal relationships. In this work, we identify two fundamental limitations of existing approaches: the lack of robust intra-modal mechanisms to assess the significance of visual and textual tokens, leading to poor generalization in complex scenes; and the absence of fine-grained uncertainty modeling, which fails to capture the one-to-many and many-to-one nature of region-word correspondences. To address these issues, we propose a unified approach that incorporates significance-aware and granularity-aware modeling and region-level uncertainty modeling. Our method leverages modality-specific biases to identify salient features without relying on brittle cross-modal attention, and represents region features as a mixture of Gaussian distributions to capture fine-grained uncertainty. Extensive experiments on Flickr30K and MS-COCO demonstrate that our approach achieves state-of-the-art performance across various backbone architectures, significantly enhancing the robustness and interpretability of fine-grained image-text alignment.
>
---
#### [replaced 105] BrainPuzzle: Hybrid Physics and Data-Driven Reconstruction for Transcranial Ultrasound Tomography
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20029v2](https://arxiv.org/pdf/2510.20029v2)**

> **作者:** Shengyu Chen; Shihang Feng; Yi Luo; Xiaowei Jia; Youzuo Lin
>
> **备注:** 13 pages
>
> **摘要:** Ultrasound brain imaging remains challenging due to the large difference in sound speed between the skull and brain tissues and the difficulty of coupling large probes to the skull. This work aims to achieve quantitative transcranial ultrasound by reconstructing an accurate speed-of-sound (SoS) map of the brain. Traditional physics-based full-waveform inversion (FWI) is limited by weak signals caused by skull-induced attenuation, mode conversion, and phase aberration, as well as incomplete spatial coverage since full-aperture arrays are clinically impractical. In contrast, purely data-driven methods that learn directly from raw ultrasound data often fail to model the complex nonlinear and nonlocal wave propagation through bone, leading to anatomically plausible but quantitatively biased SoS maps under low signal-to-noise and sparse-aperture conditions. To address these issues, we propose BrainPuzzle, a hybrid two-stage framework that combines physical modeling with machine learning. In the first stage, reverse time migration (time-reversal acoustics) is applied to multi-angle acquisitions to produce migration fragments that preserve structural details even under low SNR. In the second stage, a transformer-based super-resolution encoder-decoder with a graph-based attention unit (GAU) fuses these fragments into a coherent and quantitatively accurate SoS image. A partial-array acquisition strategy using a movable low-count transducer set improves feasibility and coupling, while the hybrid algorithm compensates for the missing aperture. Experiments on two synthetic datasets show that BrainPuzzle achieves superior SoS reconstruction accuracy and image completeness, demonstrating its potential for advancing quantitative ultrasound brain imaging.
>
---
#### [replaced 106] WorldScore: A Unified Evaluation Benchmark for World Generation
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.00983v2](https://arxiv.org/pdf/2504.00983v2)**

> **作者:** Haoyi Duan; Hong-Xing Yu; Sirui Chen; Li Fei-Fei; Jiajun Wu
>
> **备注:** ICCV 2025. Project website: https://haoyi-duan.github.io/WorldScore/ The first two authors contributed equally
>
> **摘要:** We introduce the WorldScore benchmark, the first unified benchmark for world generation. We decompose world generation into a sequence of next-scene generation tasks with explicit camera trajectory-based layout specifications, enabling unified evaluation of diverse approaches from 3D and 4D scene generation to video generation models. The WorldScore benchmark encompasses a curated dataset of 3,000 test examples that span diverse worlds: static and dynamic, indoor and outdoor, photorealistic and stylized. The WorldScore metrics evaluate generated worlds through three key aspects: controllability, quality, and dynamics. Through extensive evaluation of 19 representative models, including both open-source and closed-source ones, we reveal key insights and challenges for each category of models. Our dataset, evaluation code, and leaderboard can be found at https://haoyi-duan.github.io/WorldScore/
>
---
#### [replaced 107] PANDA -- Patch And Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2511.09791v2](https://arxiv.org/pdf/2511.09791v2)**

> **作者:** Siddeshwar Raghavan; Jiangpeng He; Fengqing Zhu
>
> **备注:** Accepted in AAAI 2026 Main Technical Track
>
> **摘要:** Exemplar-Free Continual Learning (EFCL) restricts the storage of previous task data and is highly susceptible to catastrophic forgetting. While pre-trained models (PTMs) are increasingly leveraged for EFCL, existing methods often overlook the inherent imbalance of real-world data distributions. We discovered that real-world data streams commonly exhibit dual-level imbalances, dataset-level distributions combined with extreme or reversed skews within individual tasks, creating both intra-task and inter-task disparities that hinder effective learning and generalization. To address these challenges, we propose PANDA, a Patch-and-Distribution-Aware Augmentation framework that integrates seamlessly with existing PTM-based EFCL methods. PANDA amplifies low-frequency classes by using a CLIP encoder to identify representative regions and transplanting those into frequent-class samples within each task. Furthermore, PANDA incorporates an adaptive balancing strategy that leverages prior task distributions to smooth inter-task imbalances, reducing the overall gap between average samples across tasks and enabling fairer learning with frozen PTMs. Extensive experiments and ablation studies demonstrate PANDA's capability to work with existing PTM-based CL methods, improving accuracy and reducing catastrophic forgetting.
>
---
#### [replaced 108] Hallo4: High-Fidelity Dynamic Portrait Animation via Direct Preference Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23525v4](https://arxiv.org/pdf/2505.23525v4)**

> **作者:** Jiahao Cui; Yan Chen; Mingwang Xu; Hanlin Shang; Yuxuan Chen; Yun Zhan; Zilong Dong; Yao Yao; Jingdong Wang; Siyu Zhu
>
> **摘要:** Generating highly dynamic and photorealistic portrait animations driven by audio and skeletal motion remains challenging due to the need for precise lip synchronization, natural facial expressions, and high-fidelity body motion dynamics. We propose a human-preference-aligned diffusion framework that addresses these challenges through two key innovations. First, we introduce direct preference optimization tailored for human-centric animation, leveraging a curated dataset of human preferences to align generated outputs with perceptual metrics for portrait motion-video alignment and naturalness of expression. Second, the proposed temporal motion modulation resolves spatiotemporal resolution mismatches by reshaping motion conditions into dimensionally aligned latent features through temporal channel redistribution and proportional feature expansion, preserving the fidelity of high-frequency motion details in diffusion-based synthesis. The proposed mechanism is complementary to existing UNet and DiT-based portrait diffusion approaches, and experiments demonstrate obvious improvements in lip-audio synchronization, expression vividness, body motion coherence over baseline methods, alongside notable gains in human preference metrics. Our model and source code can be found at: https://github.com/fudan-generative-vision/hallo4.
>
---
#### [replaced 109] FedHK-MVFC: Federated Heat Kernel Multi-View Clustering
- **分类: cs.LG; cs.CV; cs.DC; math.AG**

- **链接: [https://arxiv.org/pdf/2509.15844v2](https://arxiv.org/pdf/2509.15844v2)**

> **作者:** Kristina P. Sinaga
>
> **备注:** 53 pages, 11 figures, and 9 tables
>
> **摘要:** In the realm of distributed artificial intelligence (AI) and privacy-focused medical applications, this paper proposes a multi-view clustering framework that links quantum field theory with federated healthcare analytics. The method uses heat kernel coefficients from spectral analysis to convert Euclidean distances into geometry-aware similarity measures that capture the structure of diverse medical data. The framework is presented through the heat kernel distance (HKD) transformation, which has convergence guarantees. Two algorithms have been developed: The first, Heat Kernel-Enhanced Multi-View Fuzzy Clustering (HK-MVFC), is used for central analysis. The second, Federated Heat Kernel Multi-View Fuzzy Clustering (FedHK-MVFC), is used for secure, privacy-preserving learning across hospitals. FedHK-MVFC uses differential privacy and secure aggregation to enable HIPAA-compliant collaboration. Tests on synthetic cardiovascular patient datasets demonstrate increased clustering accuracy, reduced communication, and retained efficiency compared to centralized methods. After being validated on 10,000 synthetic patient records across two hospitals, the methods proved useful for collaborative phenotyping involving electrocardiogram (ECG) data, cardiac imaging data, and behavioral data. The proposed methods' theoretical contributions include update rules with proven convergence, adaptive view weighting, and privacy-preserving protocols. These contributions establish a new standard for geometry-aware federated learning in healthcare, translating advanced mathematics into practical solutions for analyzing sensitive medical data while ensuring rigor and clinical relevance.
>
---
#### [replaced 110] Pistachio: Towards Synthetic, Balanced, and Long-Form Video Anomaly Benchmarks
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.19474v3](https://arxiv.org/pdf/2511.19474v3)**

> **作者:** Jie Li; Hongyi Cai; Mingkang Dong; Muxin Pu; Shan You; Fei Wang; Tao Huang
>
> **摘要:** Automatically detecting abnormal events in videos is crucial for modern autonomous systems, yet existing Video Anomaly Detection (VAD) benchmarks lack the scene diversity, balanced anomaly coverage, and temporal complexity needed to reliably assess real-world performance. Meanwhile, the community is increasingly moving toward Video Anomaly Understanding (VAU), which requires deeper semantic and causal reasoning but remains difficult to benchmark due to the heavy manual annotation effort it demands. In this paper, we introduce Pistachio, a new VAD/VAU benchmark constructed entirely through a controlled, generation-based pipeline. By leveraging recent advances in video generation models, Pistachio provides precise control over scenes, anomaly types, and temporal narratives, effectively eliminating the biases and limitations of Internet-collected datasets. Our pipeline integrates scene-conditioned anomaly assignment, multi-step storyline generation, and a temporally consistent long-form synthesis strategy that produces coherent 41-second videos with minimal human intervention. Extensive experiments demonstrate the scale, diversity, and complexity of Pistachio, revealing new challenges for existing methods and motivating future research on dynamic and multi-event anomaly understanding.
>
---
#### [replaced 111] VIVAT: Virtuous Improving VAE Training through Artifact Mitigation
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2506.07863v2](https://arxiv.org/pdf/2506.07863v2)**

> **作者:** Lev Novitskiy; Viacheslav Vasilev; Maria Kovaleva; Vladimir Arkhipkin; Denis Dimitrov
>
> **摘要:** Variational Autoencoders (VAEs) remain a cornerstone of generative computer vision, yet their training is often plagued by artifacts that degrade reconstruction and generation quality. This paper introduces VIVAT, a systematic approach to mitigating common artifacts in KL-VAE training without requiring radical architectural changes. We present a detailed taxonomy of five prevalent artifacts - color shift, grid patterns, blur, corner and droplet artifacts - and analyze their root causes. Through straightforward modifications, including adjustments to loss weights, padding strategies, and the integration of Spatially Conditional Normalization, we demonstrate significant improvements in VAE performance. Our method achieves state-of-the-art results in image reconstruction metrics (PSNR and SSIM) across multiple benchmarks and enhances text-to-image generation quality, as evidenced by superior CLIP scores. By preserving the simplicity of the KL-VAE framework while addressing its practical challenges, VIVAT offers actionable insights for researchers and practitioners aiming to optimize VAE training.
>
---
#### [replaced 112] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-domain Few-shot Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12200v2](https://arxiv.org/pdf/2511.12200v2)**

> **作者:** Sujun Sun; Haowen Gu; Cheng Xie; Yanxu Ren; Mingwu Ren; Haofeng Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Cross-domain Few-shot Segmentation (CD-FSS) aims to segment novel classes from target domains that are not involved in training and have significantly different data distributions from the source domain, using only a few annotated samples, and recent years have witnessed significant progress on this task. However, existing CD-FSS methods primarily focus on style gaps between source and target domains while ignoring segmentation granularity gaps, resulting in insufficient semantic discriminability for novel classes in target domains. Therefore, we propose a Hierarchical Semantic Learning (HSL) framework to tackle this problem. Specifically, we introduce a Dual Style Randomization (DSR) module and a Hierarchical Semantic Mining (HSM) module to learn hierarchical semantic features, thereby enhancing the model's ability to recognize semantics at varying granularities. DSR simulates target domain data with diverse foreground-background style differences and overall style variations through foreground and global style randomization respectively, while HSM leverages multi-scale superpixels to guide the model to mine intra-class consistency and inter-class distinction at different granularities. Additionally, we also propose a Prototype Confidence-modulated Thresholding (PCMT) module to mitigate segmentation ambiguity when foreground and background are excessively similar. Extensive experiments are conducted on four popular target domain datasets, and the results demonstrate that our method achieves state-of-the-art performance.
>
---
#### [replaced 113] DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2305.13625v4](https://arxiv.org/pdf/2305.13625v4)**

> **作者:** Jiang Liu; Chun Pong Lau; Zhongliang Guo; Yuxiang Guo; Zhaoyang Wang; Rama Chellappa
>
> **备注:** Code is at https://github.com/joellliu/DiffProtect/
>
> **摘要:** The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.
>
---
#### [replaced 114] Sketch-guided Cage-based 3D Gaussian Splatting Deformation
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2411.12168v3](https://arxiv.org/pdf/2411.12168v3)**

> **作者:** Tianhao Xie; Noam Aigerman; Eugene Belilovsky; Tiberiu Popa
>
> **备注:** 10 pages, 9 figures, accepted at WACV 26, project page: https://tianhaoxie.github.io/project/gs_deform/
>
> **摘要:** 3D Gaussian Splatting (GS) is one of the most promising novel 3D representations that has received great interest in computer graphics and computer vision. While various systems have introduced editing capabilities for 3D GS, such as those guided by text prompts, fine-grained control over deformation remains an open challenge. In this work, we present a novel sketch-guided 3D GS deformation system that allows users to intuitively modify the geometry of a 3D GS model by drawing a silhouette sketch from a single viewpoint. Our approach introduces a new deformation method that combines cage-based deformations with a variant of Neural Jacobian Fields, enabling precise, fine-grained control. Additionally, it leverages large-scale 2D diffusion priors and ControlNet to ensure the generated deformations are semantically plausible. Through a series of experiments, we demonstrate the effectiveness of our method and showcase its ability to animate static 3D GS models as one of its key applications.
>
---
#### [replaced 115] Improving Partially Observed Trajectories Forecasting by Target-driven Self-Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.16767v2](https://arxiv.org/pdf/2501.16767v2)**

> **作者:** Peng Shu; Pengfei Zhu; Mengshi Qi; Liang Liu
>
> **摘要:** Accurate prediction of future trajectories of traffic agents is essential for ensuring safe autonomous driving. However, partially observed trajectories can significantly degrade the performance of even state-of-the-art models. Previous approaches often rely on knowledge distillation to transfer features from fully observed trajectories to partially observed ones. This involves firstly training a fully observed model and then using a distillation process to create the final model. While effective, they require multi-stage training, making the training process very expensive. Moreover, knowledge distillation can lead to a performance degradation of the model. In this paper, we introduce a Target-drivenSelf-Distillation method (TSD) for motion forecasting. Our method leverages predicted accurate targets to guide the model in making predictions under partial observation conditions. By employing self-distillation, the model learns from the feature distributions of both fully observed and partially observed trajectories during a single end-to-end training process. This enhances the model's ability to predict motion accurately in both fully observed and partially observed scenarios. We evaluate our method on multiple datasets and state-of-the-art motion forecasting models. Extensive experimental results demonstrate that our approach achieves significant performance improvements in both settings. To facilitate further research, we will release our code and model checkpoints.
>
---
#### [replaced 116] TempoMaster: Efficient Long Video Generation via Next-Frame-Rate Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12578v2](https://arxiv.org/pdf/2511.12578v2)**

> **作者:** Yukuo Ma; Cong Liu; Junke Wang; Junqi Liu; Haibin Huang; Zuxuan Wu; Chi Zhang; Xuelong Li
>
> **备注:** for more information, see https://scottykma.github.io/tempomaster-gitpage/
>
> **摘要:** We present TempoMaster, a novel framework that formulates long video generation as next-frame-rate prediction. Specifically, we first generate a low-frame-rate clip that serves as a coarse blueprint of the entire video sequence, and then progressively increase the frame rate to refine visual details and motion continuity. During generation, TempoMaster employs bidirectional attention within each frame-rate level while performing autoregression across frame rates, thus achieving long-range temporal coherence while enabling efficient and parallel synthesis. Extensive experiments demonstrate that TempoMaster establishes a new state-of-the-art in long video generation, excelling in both visual and temporal quality.
>
---
#### [replaced 117] ICAS: Detecting Training Data from Autoregressive Image Generative Models
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2507.05068v2](https://arxiv.org/pdf/2507.05068v2)**

> **作者:** Hongyao Yu; Yixiang Qiu; Yiheng Yang; Hao Fang; Tianqu Zhuang; Jiaxin Hong; Bin Chen; Hao Wu; Shu-Tao Xia
>
> **备注:** ACM MM 2025
>
> **摘要:** Autoregressive image generation has witnessed rapid advancements, with prominent models such as scale-wise visual auto-regression pushing the boundaries of visual synthesis. However, these developments also raise significant concerns regarding data privacy and copyright. In response, training data detection has emerged as a critical task for identifying unauthorized data usage in model training. To better understand the vulnerability of autoregressive image generative models to such detection, we conduct the first study applying membership inference to this domain. Our approach comprises two key components: implicit classification and an adaptive score aggregation strategy. First, we compute the implicit token-wise classification score within the query image. Then we propose an adaptive score aggregation strategy to acquire a final score, which places greater emphasis on the tokens with lower scores. A higher final score indicates that the sample is more likely to be involved in the training set. To validate the effectiveness of our method, we adapt existing detection algorithms originally designed for LLMs to visual autoregressive models. Extensive experiments demonstrate the superiority of our method in both class-conditional and text-to-image scenarios. Moreover, our approach exhibits strong robustness and generalization under various data transformations. Furthermore, sufficient experiments suggest two novel key findings: (1) A linear scaling law on membership inference, exposing the vulnerability of large foundation models. (2) Training data from scale-wise visual autoregressive models is easier to detect than other autoregressive paradigms. Our code is available at https://github.com/Chrisqcwx/ImageAR-MIA.
>
---
#### [replaced 118] WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.18151v2](https://arxiv.org/pdf/2505.18151v2)**

> **作者:** Zizhang Li; Hong-Xing Yu; Wei Liu; Yin Yang; Charles Herrmann; Gordon Wetzstein; Jiajun Wu
>
> **备注:** ICCV 2025 (Highlight). The first two authors contributed equally. Project website: https://kyleleey.github.io/WonderPlay/
>
> **摘要:** WonderPlay is a novel framework integrating physics simulation with video generation for generating action-conditioned dynamic 3D scenes from a single image. While prior works are restricted to rigid body or simple elastic dynamics, WonderPlay features a hybrid generative simulator to synthesize a wide range of 3D dynamics. The hybrid generative simulator first uses a physics solver to simulate coarse 3D dynamics, which subsequently conditions a video generator to produce a video with finer, more realistic motion. The generated video is then used to update the simulated dynamic 3D scene, closing the loop between the physics solver and the video generator. This approach enables intuitive user control to be combined with the accurate dynamics of physics-based simulators and the expressivity of diffusion-based video generators. Experimental results demonstrate that WonderPlay enables users to interact with various scenes of diverse content, including cloth, sand, snow, liquid, smoke, elastic, and rigid bodies -- all using a single image input. Code will be made public. Project website: https://kyleleey.github.io/WonderPlay/
>
---
#### [replaced 119] Beyond Subspace Isolation: Many-to-Many Transformer for Light Field Image Super-resolution
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2401.00740v4](https://arxiv.org/pdf/2401.00740v4)**

> **作者:** Zeke Zexi Hu; Xiaoming Chen; Vera Yuk Ying Chung; Yiran Shen
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** The effective extraction of spatial-angular features plays a crucial role in light field image super-resolution (LFSR) tasks, and the introduction of convolution and Transformers leads to significant improvement in this area. Nevertheless, due to the large 4D data volume of light field images, many existing methods opted to decompose the data into a number of lower-dimensional subspaces and perform Transformers in each sub-space individually. As a side effect, these methods inadvertently restrict the self-attention mechanisms to a One-to-One scheme accessing only a limited subset of LF data, explicitly preventing comprehensive optimization on all spatial and angular cues. In this paper, we identify this limitation as subspace isolation and introduce a novel Many-to-Many Transformer (M2MT) to address it. M2MT aggregates angular information in the spatial subspace before performing the self-attention mechanism. It enables complete access to all information across all sub-aperture images (SAIs) in a light field image. Consequently, M2MT is enabled to comprehensively capture long-range correlation dependencies. With M2MT as the foundational component, we develop a simple yet effective M2MT network for LFSR. Our experimental results demonstrate that M2MT achieves state-of-the-art performance across various public datasets, and it offers a favorable balance between model performance and efficiency, yielding higher-quality LFSR results with substantially lower demand for memory and computation. We further conduct in-depth analysis using local attribution maps (LAM) to obtain visual interpretability, and the results validate that M2MT is empowered with a truly non-local context in both spatial and angular subspaces to mitigate subspace isolation and acquire effective spatial-angular representation.
>
---
#### [replaced 120] One-to-All Animation: Alignment-Free Character Animation and Image Pose Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22940v2](https://arxiv.org/pdf/2511.22940v2)**

> **作者:** Shijun Shi; Jing Xu; Zhihang Li; Chunli Peng; Xiaoda Yang; Lijing Lu; Kai Hu; Jiangning Zhang
>
> **备注:** Project Page:https://ssj9596.github.io/one-to-all-animation-project/
>
> **摘要:** Recent advances in diffusion models have greatly improved pose-driven character animation. However, existing methods are limited to spatially aligned reference-pose pairs with matched skeletal structures. Handling reference-pose misalignment remains unsolved. To address this, we present One-to-All Animation, a unified framework for high-fidelity character animation and image pose transfer for references with arbitrary layouts. First, to handle spatially misaligned reference, we reformulate training as a self-supervised outpainting task that transforms diverse-layout reference into a unified occluded-input format. Second, to process partially visible reference, we design a reference extractor for comprehensive identity feature extraction. Further, we integrate hybrid reference fusion attention to handle varying resolutions and dynamic sequence lengths. Finally, from the perspective of generation quality, we introduce identity-robust pose control that decouples appearance from skeletal structure to mitigate pose overfitting, and a token replace strategy for coherent long-video generation. Extensive experiments show that our method outperforms existing approaches. The code and model are available at https://github.com/ssj9596/One-to-All-Animation.
>
---
#### [replaced 121] Prompt-OT: An Optimal Transport Regularization Paradigm for Knowledge Preservation in Vision-Language Model Adaptation
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文针对视觉-语言模型（VLM）在下游任务适应中出现的过拟合与零样本泛化能力下降问题，提出基于最优传输（OT）的提示学习框架Prompt-OT。通过约束视觉与文本特征分布的结构一致性，实现知识保留与性能提升，在多个评估场景中优于现有方法。**

- **链接: [https://arxiv.org/pdf/2503.08906v3](https://arxiv.org/pdf/2503.08906v3)**

> **作者:** Xiwen Chen; Wenhui Zhu; Peijie Qiu; Hao Wang; Huayu Li; Haiyu Wu; Aristeidis Sotiras; Yalin Wang; Abolfazl Razi
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Vision-language models (VLMs) such as CLIP demonstrate strong performance but struggle when adapted to downstream tasks. Prompt learning has emerged as an efficient and effective strategy to adapt VLMs while preserving their pre-trained knowledge. However, existing methods still lead to overfitting and degrade zero-shot generalization. To address this challenge, we propose an optimal transport (OT)-guided prompt learning framework that mitigates forgetting by preserving the structural consistency of feature distributions between pre-trained and fine-tuned models. Unlike conventional point-wise constraints, OT naturally captures cross-instance relationships and expands the feasible parameter space for prompt tuning, allowing a better trade-off between adaptation and generalization. Our approach enforces joint constraints on both vision and text representations, ensuring a holistic feature alignment. Extensive experiments on benchmark datasets demonstrate that our simple yet effective method can outperform existing prompt learning strategies in base-to-novel generalization, cross-dataset evaluation, and domain generalization without additional augmentation or ensemble techniques. The code is available at https://github.com/ChongQingNoSubway/Prompt-OT
>
---
#### [replaced 122] MoH: Multi-Head Attention as Mixture-of-Head Attention
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2410.11842v3](https://arxiv.org/pdf/2410.11842v3)**

> **作者:** Peng Jin; Bo Zhu; Li Yuan; Shuicheng Yan
>
> **备注:** Accepted by ICML 2025, code: https://github.com/SkyworkAI/MoH
>
> **摘要:** In this work, we upgrade the multi-head attention mechanism, the core of the Transformer model, to improve efficiency while maintaining or surpassing the previous accuracy level. We show that multi-head attention can be expressed in the summation form. Drawing on the insight that not all attention heads hold equal significance, we propose Mixture-of-Head attention (MoH), a new architecture that treats attention heads as experts in the Mixture-of-Experts (MoE) mechanism. MoH has two significant advantages: First, MoH enables each token to select the appropriate attention heads, enhancing inference efficiency without compromising accuracy or increasing the number of parameters. Second, MoH replaces the standard summation in multi-head attention with a weighted summation, introducing flexibility to the attention mechanism and unlocking extra performance potential. Extensive experiments on ViT, DiT, and LLMs demonstrate that MoH outperforms multi-head attention by using only 50%-90% of the attention heads. Moreover, we demonstrate that pre-trained multi-head attention models, such as LLaMA3-8B, can be further continue-tuned into our MoH models. Notably, MoH-LLaMA3-8B achieves an average accuracy of 64.0% across 14 benchmarks, outperforming LLaMA3-8B by 2.4% by utilizing only 75% of the attention heads. We believe the proposed MoH is a promising alternative to multi-head attention and provides a strong foundation for developing advanced and efficient attention-based models.
>
---
#### [replaced 123] Blind Inverse Problem Solving Made Easy by Text-to-Image Latent Diffusion
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2412.00557v2](https://arxiv.org/pdf/2412.00557v2)**

> **作者:** Michail Dontas; Yutong He; Naoki Murata; Yuki Mitsufuji; J. Zico Kolter; Ruslan Salakhutdinov
>
> **摘要:** This paper considers blind inverse image restoration, the task of predicting a target image from a degraded source when the degradation (i.e. the forward operator) is unknown. Existing solutions typically rely on restrictive assumptions such as operator linearity, curated training data or narrow image distributions limiting their practicality. We introduce LADiBI, a training-free method leveraging large-scale text-to-image diffusion to solve diverse blind inverse problems with minimal assumptions. Within a Bayesian framework, LADiBI uses text prompts to jointly encode priors for both target images and operators, unlocking unprecedented flexibility compared to existing methods. Additionally, we propose a novel diffusion posterior sampling algorithm that combines strategic operator initialization with iterative refinement of image and operator parameters, eliminating the need for highly constrained operator forms. Experiments show that LADiBI effectively handles both linear and challenging nonlinear image restoration problems across various image distributions, all without task-specific assumptions or retraining.
>
---
#### [replaced 124] STORM: Segment, Track, and Object Re-Localization from a Single Image
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09771v2](https://arxiv.org/pdf/2511.09771v2)**

> **作者:** Yu Deng; Teng Cao; Hikaru Shindo; Jiahong Xue; Quentin Delfosse; Kristian Kersting
>
> **摘要:** Accurate 6D pose estimation and tracking are fundamental capabilities for physical AI systems such as robots. However, existing approaches typically require a pre-defined 3D model of the target and rely on a manually annotated segmentation mask in the first frame, which is labor-intensive and leads to reduced performance when faced with occlusions or rapid movement. To address these limitations, we propose STORM (Segment, Track, and Object Re-localization from a single iMage), an open-source robust real-time 6D pose estimation system that requires no manual annotation. STORM employs a novel three-stage pipeline combining vision-language understanding with feature matching: contextual object descriptions guide localization, self-cross-attention mechanisms identify candidate regions, and produce precise masks and 3D models for accurate pose estimation. Another key innovation is our automatic re-registration mechanism that detects tracking failures through feature similarity monitoring and recovers from severe occlusions or rapid motion. STORM achieves state-of-the-art accuracy on challenging industrial datasets featuring multi-object occlusions, high-speed motion, and varying illumination, while operating at real-time speeds without additional training. This annotation-free approach significantly reduces deployment overhead, providing a practical solution for modern applications, such as flexible manufacturing and intelligent quality control.
>
---
#### [replaced 125] Full-scale Representation Guided Network for Retinal Vessel Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2501.18921v3](https://arxiv.org/pdf/2501.18921v3)**

> **作者:** Sunyong Seo; Sangwook Yoo; Huisu Yoon
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** The U-Net architecture and its variants have remained state-of-the-art (SOTA) for retinal vessel segmentation over the past decade. In this study, we introduce a Full-Scale Guided Network (FSG-Net), where a novel feature representation module using modernized convolution blocks effectively captures full-scale structural information, while a guided convolution block subsequently refines this information. Specifically, we introduce an attention-guided filter within the guided convolution block, leveraging its similarity to unsharp masking to enhance fine vascular structures. Passing full-scale information to the attention block facilitates the generation of more contextually relevant attention maps, which are then passed to the attention-guided filter, providing further refinement to the segmentation performance. The structure preceding the guided convolution block can be replaced by any U-Net variant, ensuring flexibility and scalability across various segmentation tasks. For a fair comparison, we re-implemented recent studies available in public repositories to evaluate their scalability and reproducibility. Our experiments demonstrate that, despite its compact architecture, FSG-Net delivers performance competitive with SOTA methods across multiple public datasets. Ablation studies further demonstrate that each proposed component meaningfully contributes to this competitive performance. Our code is available on https://github.com/ZombaSY/FSG-Net-pytorch.
>
---
#### [replaced 126] UniFucGrasp: Human-Hand-Inspired Unified Functional Grasp Annotation Strategy and Dataset for Diverse Dexterous Hands
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 该论文针对灵巧手抓取数据集缺乏功能性标注的问题，提出UniFucGrasp策略与多手功能抓取数据集。基于人体手部生物力学与几何力闭合原理，实现低成本、高效的功能性抓取标注，提升抓取准确率与跨手适应性，有效缓解标注成本高与泛化难问题。**

- **链接: [https://arxiv.org/pdf/2508.03339v2](https://arxiv.org/pdf/2508.03339v2)**

> **作者:** Haoran Lin; Wenrui Chen; Xianchi Chen; Fan Yang; Qiang Diao; Wenxin Xie; Sijie Wu; Kailun Yang; Maojun Li; Yaonan Wang
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L). The project page is at https://haochen611.github.io/UFG
>
> **摘要:** Dexterous grasp datasets are vital for embodied intelligence, but mostly emphasize grasp stability, ignoring functional grasps needed for tasks like opening bottle caps or holding cup handles. Most rely on bulky, costly, and hard-to-control high-DOF Shadow Hands. Inspired by the human hand's underactuated mechanism, we establish UniFucGrasp, a universal functional grasp annotation strategy and dataset for multiple dexterous hand types. Based on biomimicry, it maps natural human motions to diverse hand structures and uses geometry-based force closure to ensure functional, stable, human-like grasps. This method supports low-cost, efficient collection of diverse, high-quality functional grasps. Finally, we establish the first multi-hand functional grasp dataset and provide a synthesis model to validate its effectiveness. Experiments on the UFG dataset, IsaacSim, and complex robotic tasks show that our method improves functional manipulation accuracy and grasp stability, demonstrates improved adaptability across multiple robotic hands, helping to alleviate annotation cost and generalization challenges in dexterous grasping. The project page is at https://haochen611.github.io/UFG.
>
---
#### [replaced 127] A Minimal Subset Approach for Informed Keyframe Sampling in Large-Scale SLAM
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对大规模LiDAR SLAM中的关键帧采样问题，提出一种基于最小子集的在线采样方法（MSA），旨在减少冗余、保留关键信息。通过在特征空间中构建姿态图，有效降低误检率与计算开销，提升定位精度与系统效率，无需人工调参。**

- **链接: [https://arxiv.org/pdf/2501.01791v3](https://arxiv.org/pdf/2501.01791v3)**

> **作者:** Nikolaos Stathoulopoulos; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** Please cite the published version. 8 pages, 9 figures
>
> **摘要:** Typical LiDAR SLAM architectures feature a front-end for odometry estimation and a back-end for refining and optimizing the trajectory and map, commonly through loop closures. However, loop closure detection in large-scale missions presents significant computational challenges due to the need to identify, verify, and process numerous candidate pairs for pose graph optimization. Keyframe sampling bridges the front-end and back-end by selecting frames for storing and processing during global optimization. This article proposes an online keyframe sampling approach that constructs the pose graph using the most impactful keyframes for loop closure. We introduce the Minimal Subset Approach (MSA), which optimizes two key objectives: redundancy minimization and information preservation, implemented within a sliding window framework. By operating in the feature space rather than 3-D space, MSA efficiently reduces redundant keyframes while retaining essential information. Evaluations on diverse public datasets show that the proposed approach outperforms naive methods in reducing false positive rates in place recognition, while delivering superior ATE and RPE in metric localization, without the need for manual parameter tuning. Additionally, MSA demonstrates efficiency and scalability by reducing memory usage and computational overhead during loop closure detection and pose graph optimization.
>
---
#### [replaced 128] Robust Phase-Shifting Profilometry for Arbitrary Motion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.10009v2](https://arxiv.org/pdf/2507.10009v2)**

> **作者:** Geyou Zhang; Kai Liu; Ao Li; Ce Zhu
>
> **摘要:** Phase-shifting profilometry (PSP) enables high-accuracy 3D reconstruction but remains highly susceptible to object motion. Although numerous studies have explored compensation for motion-induced errors, residual inaccuracies still persist, particularly in complex motion scenarios. In this paper, we propose a robust phase-shifting profilometry for arbitrary motion (RPSP-AM), including six-degrees-of-freedom (6-DoF) motion (translation and rotation in any direction), non-rigid deformations, and multi-target movements, achieving high-fidelity motion-error-free 3D reconstruction. We categorize motion errors into two components: 1) ghosting artifacts induced by image misalignment, and 2) ripple-like distortions induced by phase deviation. To eliminate the ghosting artifacts, we perform pixel-wise image alignment based on dense optical flow tracking. To correct ripple-like distortions, we propose a high-accuracy, low-complexity image-sequential binomial self-compensation (I-BSC) method, which performs a summation of the homogeneous fringe images weighted by binomial coefficients, exponentially reducing the ripple-like distortions with a competitive computational speed compared with the traditional four-step phase-shifting method. Extensive experimental results demonstrate that, under challenging conditions such as 6-DoF motion, non-rigid deformations, and multi-target movements, the proposed RPSP-AM outperforms state-of-the-art (SoTA) methods in compensating for both ghosting artifacts and ripple-like distortions. Our approach extends the applicability of PSP to arbitrary motion scenarios, endowing it with potential for widespread adoption in fields such as robotics, industrial inspection, and medical reconstruction.
>
---
#### [replaced 129] GBT-SAM: A Parameter-Efficient Depth-Aware Model for Generalizable Brain tumour Segmentation on mp-MRI
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.04325v4](https://arxiv.org/pdf/2503.04325v4)**

> **作者:** Cecilia Diana-Albelda; Roberto Alcover-Couso; Álvaro García-Martín; Jesus Bescos; Marcos Escudero-Viñolo
>
> **摘要:** Gliomas are aggressive brain tumors that require accurate imaging-based diagnosis, with segmentation playing a critical role in evaluating morphology and treatment decisions. Manual delineation of gliomas is time-consuming and prone to variability, motivating the use of deep learning to improve consistency and alleviate clinical workload. However, existing methods often fail to fully exploit the information available in multi-parametric MRI (mp-MRI), particularly inter-slice contextual features, and typically require considerable computational resources while lacking robustness across tumor type variations. We present GBT-SAM, a parameter-efficient deep learning framework that adapts the Segment Anything Model (SAM), a large-scale vision model, to volumetric mp-MRI data. GBT-SAM reduces input complexity by selecting fewer than 2.6\% of slices per scan while incorporating all four MRI modalities, preserving essential tumor-related information with minimal cost. Furthermore, our model is trained by a two-step fine-tuning strategy that incorporates a depth-aware module to capture inter-slice correlations and lightweight adaptation layers, resulting in just 6.5M trainable parameters, which is the lowest among SAM-based approaches. GBT-SAM achieves a Dice Score of 93.54 on the BraTS Adult Glioma dataset and demonstrates robust performance on Meningioma, Pediatric Glioma, and Sub-Saharan Glioma datasets. These results highlight GBT-SAM's potential as a computationally efficient and domain-robust framework for brain tumor segmentation using mp-MRI. Our code and models are available at https://github.com/vpulab/med-sam-brain .
>
---
#### [replaced 130] SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02261v2](https://arxiv.org/pdf/2508.02261v2)**

> **作者:** Rui Qian; Haozhi Cao; Tianchen Deng; Shenghai Yuan; Lihua Xie
>
> **摘要:** Monocular 3D Semantic Scene Completion (SSC) is a challenging yet promising task that aims to infer dense geometric and semantic descriptions of a scene from a single image. While recent object-centric paradigms significantly improve efficiency by leveraging flexible 3D Gaussian primitives, they still rely heavily on a large number of randomly initialized primitives, which inevitably leads to 1) inefficient primitive initialization and 2) outlier primitives that introduce erroneous artifacts. In this paper, we propose SplatSSC, a novel framework that resolves these limitations with a depth-guided initialization strategy and a principled Gaussian aggregator. Instead of random initialization, SplatSSC utilizes a dedicated depth branch composed of a Group-wise Multi-scale Fusion (GMF) module, which integrates multi-scale image and depth features to generate a sparse yet representative set of initial Gaussian primitives. To mitigate noise from outlier primitives, we develop the Decoupled Gaussian Aggregator (DGA), which enhances robustness by decomposing geometric and semantic predictions during the Gaussian-to-voxel splatting process. Complemented with a specialized Probability Scale Loss, our method achieves state-of-the-art performance on the Occ-ScanNet dataset, outperforming prior approaches by over 6.3% in IoU and 4.1% in mIoU, while reducing both latency and memory cost by more than 9.3%.
>
---
#### [replaced 131] U-FaceBP: Uncertainty-aware Bayesian Ensemble Deep Learning for Face Video-based Blood Pressure Measurement
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2412.10679v2](https://arxiv.org/pdf/2412.10679v2)**

> **作者:** Yusuke Akamatsu; Akinori F. Ebihara; Terumi Umematsu
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Blood pressure (BP) measurement is crucial for daily health assessment. Remote photoplethysmography (rPPG), which extracts pulse waves from face videos captured by a camera, has the potential to enable convenient BP measurement without specialized medical devices. However, there are various uncertainties in BP estimation using rPPG, leading to limited estimation performance and reliability. In this paper, we propose U-FaceBP, an uncertainty-aware Bayesian ensemble deep learning method for face video-based BP measurement. U-FaceBP models aleatoric and epistemic uncertainties in face video-based BP estimation with a Bayesian neural network (BNN). Additionally, we design U-FaceBP as an ensemble method, estimating BP from rPPG signals, PPG signals derived from face videos, and face images using multiple BNNs. Large-scale experiments on two datasets involving 1197 subjects from diverse racial groups demonstrate that U-FaceBP outperforms state-of-the-art BP estimation methods. Furthermore, we show that the uncertainty estimates provided by U-FaceBP are informative and useful for guiding modality fusion, assessing prediction reliability, and analyzing performance across racial groups.
>
---
#### [replaced 132] HARMONY: Hidden Activation Representations and Model Output-Aware Uncertainty Estimation for Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.22171v2](https://arxiv.org/pdf/2510.22171v2)**

> **作者:** Erum Mushtaq; Zalan Fabian; Yavuz Faruk Bakman; Anil Ramakrishna; Mahdi Soltanolkotabi; Salman Avestimehr
>
> **摘要:** Uncertainty Estimation (UE) plays a central role in quantifying the reliability of model outputs and reducing unsafe generations via selective prediction. In this regard, most existing probability-based UE approaches rely on predefined functions, aggregating token probabilities into a single UE score using heuristics such as length-normalization. However, these methods often fail to capture the complex relationships between generated tokens and struggle to identify biased probabilities often influenced by \textbf{language priors}. Another line of research uses hidden representations of the model and trains simple MLP architectures to predict uncertainty. However, such functions often lose the intricate \textbf{ inter-token dependencies}. While prior works show that hidden representations encode multimodal alignment signals, our work demonstrates that how these signals are processed has a significant impact on the UE performance. To effectively leverage these signals to identify inter-token dependencies, and vision-text alignment, we propose \textbf{HARMONY} (Hidden Activation Representations and Model Output-Aware Uncertainty Estimation for Vision-Language Models), a novel UE framework that integrates generated tokens ('text'), model's uncertainty score at the output ('MaxProb'), and its internal belief on the visual understanding of the image and the generated token (captured by 'hidden representations') at token level via appropriate input mapping design and suitable architecture choice. Our experimental experiments across two open-ended VQA benchmarks (A-OKVQA, and VizWiz) and four state-of-the-art VLMs (LLaVA-7B, LLaVA-13B, InstructBLIP, and Qwen-VL) show that HARMONY consistently matches or surpasses existing approaches, achieving up to 5\% improvement in AUROC and 9\% in PRR.
>
---
#### [replaced 133] Hybrid Swin Attention Networks for Simultaneously Low-Dose PET and CT Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.06591v5](https://arxiv.org/pdf/2509.06591v5)**

> **作者:** Yichao Liu; Hengzhi Xue; YueYang Teng; Junwen Guo
>
> **摘要:** Low-dose computed tomography (LDCT) and positron emission tomography (PET) have emerged as safer alternatives to conventional imaging modalities by significantly reducing radiation exposure. However, this reduction often results in increased noise and artifacts, which can compromise diagnostic accuracy. Consequently, denoising for LDCT/PET has become a vital area of research aimed at enhancing image quality while maintaining radiation safety. In this study, we introduce a novel Hybrid Swin Attention Network (HSANet), which incorporates Efficient Global Attention (EGA) modules and a hybrid upsampling module. The EGA modules enhance both spatial and channel-wise interaction, improving the network's capacity to capture relevant features, while the hybrid upsampling module mitigates the risk of overfitting to noise. We validate the proposed approach using a publicly available LDCT/PET dataset. Experimental results demonstrate that HSANet achieves superior denoising performance compared to existing methods, while maintaining a lightweight model size suitable for deployment on GPUs with standard memory configurations. This makes our approach highly practical for real-world clinical applications.
>
---
#### [replaced 134] Beyond Randomness: Understand the Order of the Noise in Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07756v2](https://arxiv.org/pdf/2511.07756v2)**

> **作者:** Song Yan; Min Li; Bi Xinliang; Jian Yang; Yusen Zhang; Guanye Xiong; Yunwei Lan; Tao Zhang; Wei Zhai; Zheng-Jun Zha
>
> **摘要:** In text-driven content generation (T2C) diffusion model, semantic of generated content is mostly attributed to the process of text embedding and attention mechanism interaction. The initial noise of the generation process is typically characterized as a random element that contributes to the diversity of the generated content. Contrary to this view, this paper reveals that beneath the random surface of noise lies strong analyzable patterns. Specifically, this paper first conducts a comprehensive analysis of the impact of random noise on the model's generation. We found that noise not only contains rich semantic information, but also allows for the erasure of unwanted semantics from it in an extremely simple way based on information theory, and using the equivalence between the generation process of diffusion model and semantic injection to inject semantics into the cleaned noise. Then, we mathematically decipher these observations and propose a simple but efficient training-free and universal two-step "Semantic Erasure-Injection" process to modulate the initial noise in T2C diffusion model. Experimental results demonstrate that our method is consistently effective across various T2C models based on both DiT and UNet architectures and presents a novel perspective for optimizing the generation of diffusion model, providing a universal tool for consistent generation.
>
---
#### [replaced 135] Enhancing OCR for Sino-Vietnamese Language Processing via Fine-tuned PaddleOCRv5
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对古越南汉喃文（Han-Nom）文本的光学字符识别难题，提出对PaddleOCRv5进行微调。通过在古籍手稿数据上重训练识别模块，显著提升在模糊、非标准字形等复杂条件下的识别准确率（37.5%→50.0%），并开发交互式演示系统，助力历史文献数字化与跨语言研究。**

- **链接: [https://arxiv.org/pdf/2510.04003v2](https://arxiv.org/pdf/2510.04003v2)**

> **作者:** Minh Hoang Nguyen; Su Nguyen Thiet
>
> **备注:** Short Paper: 7 pages, 8 figures, 3 tables
>
> **摘要:** Recognizing and processing Classical Chinese (Han-Nom) texts play a vital role in digitizing Vietnamese historical documents and enabling cross-lingual semantic research. However, existing OCR systems struggle with degraded scans, non-standard glyphs, and handwriting variations common in ancient sources. In this work, we propose a fine-tuning approach for PaddleOCRv5 to improve character recognition on Han-Nom texts. We retrain the text recognition module using a curated subset of ancient Vietnamese Chinese manuscripts, supported by a full training pipeline covering preprocessing, LMDB conversion, evaluation, and visualization. Experimental results show a significant improvement over the base model, with exact accuracy increasing from 37.5 percent to 50.0 percent, particularly under noisy image conditions. Furthermore, we develop an interactive demo that visually compares pre- and post-fine-tuning recognition results, facilitating downstream applications such as Han-Vietnamese semantic alignment, machine translation, and historical linguistics research. The demo is available at https://huggingface.co/spaces/MinhDS/Fine-tuned-PaddleOCRv5
>
---
#### [replaced 136] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22643v3](https://arxiv.org/pdf/2505.22643v3)**

> **作者:** Dekai Zhu; Yixuan Hu; Youquan Liu; Dongyue Lu; Lingdong Kong; Slobodan Ilic
>
> **备注:** NeurIPS 2025; 24 pages, 10 figures, 9 tables; Code at https://github.com/worldbench/SPIRAL
>
> **摘要:** Leveraging recent diffusion models, LiDAR-based large-scale 3D scene generation has achieved great success. While recent voxel-based approaches can generate both geometric structures and semantic labels, existing range-view methods are limited to producing unlabeled LiDAR scenes. Relying on pretrained segmentation models to predict the semantic maps often results in suboptimal cross-modal consistency. To address this limitation while preserving the advantages of range-view representations, such as computational efficiency and simplified network design, we propose Spiral, a novel range-view LiDAR diffusion model that simultaneously generates depth, reflectance images, and semantic maps. Furthermore, we introduce novel semantic-aware metrics to evaluate the quality of the generated labeled range-view data. Experiments on the SemanticKITTI and nuScenes datasets demonstrate that Spiral achieves state-of-the-art performance with the smallest parameter size, outperforming two-step methods that combine the generative and segmentation models. Additionally, we validate that range images generated by Spiral can be effectively used for synthetic data augmentation in the downstream segmentation training, significantly reducing the labeling effort on LiDAR data.
>
---
#### [replaced 137] Fusion or Confusion? Assessing the impact of visible-thermal image fusion for automated wildlife detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22768v2](https://arxiv.org/pdf/2511.22768v2)**

> **作者:** Camille Dionne-Pierre; Samuel Foucher; Jérôme Théau; Jérôme Lemaître; Patrick Charbonneau; Maxime Brousseau; Mathieu Varin
>
> **备注:** 19 pages, 9 figures, submitted to Remote Sensing in Ecology and Conservation
>
> **摘要:** Efficient wildlife monitoring methods are necessary for biodiversity conservation and management. The combination of remote sensing, aerial imagery and deep learning offer promising opportunities to renew or improve existing survey methods. The complementary use of visible (VIS) and thermal infrared (TIR) imagery can add information compared to a single-source image and improve results in an automated detection context. However, the alignment and fusion process can be challenging, especially since visible and thermal images usually have different fields of view (FOV) and spatial resolutions. This research presents a case study on the great blue heron (Ardea herodias) to evaluate the performances of synchronous aerial VIS and TIR imagery to automatically detect individuals and nests using a YOLO11n model. Two VIS-TIR fusion methods were tested and compared: an early fusion approach and a late fusion approach, to determine if the addition of the TIR image gives any added value compared to a VIS-only model. VIS and TIR images were automatically aligned using a deep learning model. A principal component analysis fusion method was applied to VIS-TIR image pairs to form the early fusion dataset. A classification and regression tree was used to process the late fusion dataset, based on the detection from the VIS-only and TIR-only trained models. Across all classes, both late and early fusion improved the F1 score compared to the VIS-only model. For the main class, occupied nest, the late fusion improved the F1 score from 90.2 (VIS-only) to 93.0%. This model was also able to identify false positives from both sources with 90% recall. Although fusion methods seem to give better results, this approach comes with a limiting TIR FOV and alignment constraints that eliminate data. Using an aircraft-mounted very high-resolution visible sensor could be an interesting option for operationalizing surveys.
>
---
#### [replaced 138] DynamicTree: Interactive Real Tree Animation via Sparse Voxel Spectrum
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.22213v2](https://arxiv.org/pdf/2510.22213v2)**

> **作者:** Yaokun Li; Lihe Ding; Xiao Chen; Guang Tan; Tianfan Xue
>
> **备注:** Project Page: https://dynamictree-dev.github.io/DynamicTree.github.io/
>
> **摘要:** Generating dynamic and interactive 3D trees has wide applications in virtual reality, games, and world simulation. However, existing methods still face various challenges in generating structurally consistent and realistic 4D motion for complex real trees. In this paper, we propose DynamicTree, the first framework that can generate long-term, interactive 3D motion for 3DGS reconstructions of real trees. Unlike prior optimization-based methods, our approach generates dynamics in a fast feed-forward manner. The key success of our approach is the use of a compact sparse voxel spectrum to represent the tree movement. Given a 3D tree from Gaussian Splatting reconstruction, our pipeline first generates mesh motion using the sparse voxel spectrum and then binds Gaussians to deform the mesh. Additionally, the proposed sparse voxel spectrum can also serve as a basis for fast modal analysis under external forces, allowing real-time interactive responses. To train our model, we also introduce 4DTree, the first large-scale synthetic 4D tree dataset containing 8,786 animated tree meshes with 100-frame motion sequences. Extensive experiments demonstrate that our method achieves realistic and responsive tree animations, significantly outperforming existing approaches in both visual quality and computational efficiency.
>
---
#### [replaced 139] Learning to Generate Rigid Body Interactions with Video Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.02284v2](https://arxiv.org/pdf/2510.02284v2)**

> **作者:** David Romero; Ariana Bermudez; Hao Li; Fabio Pizzati; Ivan Laptev
>
> **摘要:** Recent video generation models have achieved remarkable progress and are now deployed in film, social media production, and advertising. Beyond their creative potential, such models also hold promise as world simulators for robotics and embodied decision making. Despite strong advances, however, current approaches still struggle to generate physically plausible object interactions and lack object-level control mechanisms. To address these limitations, we introduce KineMask, an approach for video generation that enables realistic rigid body control, interactions, and effects. Given a single image and a specified object velocity, our method generates videos with inferred motions and future object interactions. We propose a two-stage training strategy that gradually removes future motion supervision via object masks. Using this strategy we train video diffusion models (VDMs) on synthetic scenes of simple interactions and demonstrate significant improvements of object interactions in real scenes. Furthermore, KineMask integrates low-level motion control with high-level textual conditioning via predicted scene descriptions, leading to support for synthesis of complex dynamical phenomena. Our experiments show that KineMask achieves strong improvements over recent models of comparable size. Ablation studies further highlight the complementary roles of low- and high-level conditioning in VDMs. Our code, model, and data will be made publicly available. Project Page: https://daromog.github.io/KineMask/
>
---
#### [replaced 140] PRISM: Diversifying Dataset Distillation by Decoupling Architectural Priors
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09905v2](https://arxiv.org/pdf/2511.09905v2)**

> **作者:** Brian B. Moser; Shalini Sarode; Federico Raue; Stanislav Frolov; Krzysztof Adamkiewicz; Arundhati Shanbhag; Joachim Folz; Tobias C. Nauen; Andreas Dengel
>
> **摘要:** Dataset distillation (DD) promises compact yet faithful synthetic data, but existing approaches often inherit the inductive bias of a single teacher model. As dataset size increases, this bias drives generation toward overly smooth, homogeneous samples, reducing intra-class diversity and limiting generalization. We present PRISM (PRIors from diverse Source Models), a framework that disentangles architectural priors during synthesis. PRISM decouples the logit-matching and regularization objectives, supervising them with different teacher architectures: a primary model for logits and a stochastic subset for batch-normalization (BN) alignment. On ImageNet-1K, PRISM consistently and reproducibly outperforms single-teacher methods (e.g., SRe2L) and recent multi-teacher variants (e.g., G-VBSM) at low- and mid-IPC regimes. The generated data also show significantly richer intra-class diversity, as reflected by a notable drop in cosine similarity between features. We further analyze teacher selection strategies (pre- vs. intra-distillation) and introduce a scalable cross-class batch formation scheme for fast parallel synthesis. Code will be released after the review period.
>
---
#### [replaced 141] Generating Fit Check Videos with a Handheld Camera
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23886v2](https://arxiv.org/pdf/2505.23886v2)**

> **作者:** Bowei Chen; Brian Curless; Ira Kemelmacher-Shlizerman; Steven M. Seitz
>
> **摘要:** Self-captured full-body videos are popular, but most deployments require mounted cameras, carefully-framed shots, and repeated practice. We propose a more convenient solution that enables full-body video capture using handheld mobile devices. Our approach takes as input two static photos (front and back) of you in a mirror, along with an IMU motion reference that you perform while holding your mobile phone, and synthesizes a realistic video of you performing a similar target motion. We enable rendering into a new scene, with consistent illumination and shadows. We propose a novel video diffusion-based model to achieve this. Specifically, we propose a parameter-free frame generation strategy and a multi-reference attention mechanism to effectively integrate appearance information from both the front and back selfies into the video diffusion model. Further, we introduce an image-based fine-tuning strategy to enhance frame sharpness and improve shadows and reflections generation for more realistic human-scene composition.
>
---
#### [replaced 142] SizeGS: Size-aware Compression of 3D Gaussian Splatting via Mixed Integer Programming
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2412.05808v2](https://arxiv.org/pdf/2412.05808v2)**

> **作者:** Shuzhao Xie; Jiahang Liu; Weixiang Zhang; Shijia Ge; Sicheng Pan; Chen Tang; Yunpeng Bai; Cong Zhang; Xiaoyi Fan; Zhi Wang
>
> **备注:** Automatically compressing 3DGS into the desired file size while maximizing the visual quality
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have greatly improved 3D reconstruction. However, its substantial data size poses a significant challenge for transmission and storage. While many compression techniques have been proposed, they fail to efficiently adapt to fluctuating network bandwidth, leading to resource wastage. We address this issue from the perspective of size-aware compression, where we aim to compress 3DGS to a desired size by quickly searching for suitable hyperparameters. Through a measurement study, we identify key hyperparameters that affect the size -- namely, the reserve ratio of Gaussians and bit-width settings for Gaussian attributes. Then, we formulate this hyperparameter optimization problem as a mixed-integer nonlinear programming (MINLP) problem, with the goal of maximizing visual quality while respecting the size budget constraint. To solve the MINLP, we decouple this problem into two parts: discretely sampling the reserve ratio and determining the bit-width settings using integer linear programming (ILP). To solve the ILP more quickly and accurately, we design a quality loss estimator and a calibrated size estimator, as well as implement a CUDA kernel. Extensive experiments on multiple 3DGS variants demonstrate that our method achieves state-of-the-art performance in post-training compression. Furthermore, our method can achieve comparable quality to leading training-required methods after fine-tuning.
>
---
#### [replaced 143] iMontage: Unified, Versatile, Highly Dynamic Many-to-many Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20635v2](https://arxiv.org/pdf/2511.20635v2)**

> **作者:** Zhoujie Fu; Xianfang Zeng; Jinghong Lan; Xinyao Liao; Cheng Chen; Junyi Chen; Jiacheng Wei; Wei Cheng; Shiyu Liu; Yunuo Chen; Gang Yu; Guosheng Lin
>
> **备注:** Our homepage: https://kr1sjfu.github.io/iMontage-web/
>
> **摘要:** Pre-trained video models learn powerful priors for generating high-quality, temporally coherent content. While these models excel at temporal coherence, their dynamics are often constrained by the continuous nature of their training data. We hypothesize that by injecting the rich and unconstrained content diversity from image data into this coherent temporal framework, we can generate image sets that feature both natural transitions and a far more expansive dynamic range. To this end, we introduce iMontage, a unified framework designed to repurpose a powerful video model into an all-in-one image generator. The framework consumes and produces variable-length image sets, unifying a wide array of image generation and editing tasks. To achieve this, we propose an elegant and minimally invasive adaptation strategy, complemented by a tailored data curation process and training paradigm. This approach allows the model to acquire broad image manipulation capabilities without corrupting its invaluable original motion priors. iMontage excels across several mainstream many-in-many-out tasks, not only maintaining strong cross-image contextual consistency but also generating scenes with extraordinary dynamics that surpass conventional scopes. Find our homepage at: https://kr1sjfu.github.io/iMontage-web/.
>
---
#### [replaced 144] HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶仿真中视图切换不真实、几何不一致的问题，提出HybridWorldSim框架，融合神经重建与生成模型，实现高保真、可控的动态场景仿真。构建MIRROR数据集，支持多样化的城市环境测试。实验表明其显著优于现有方法，为自动驾驶研发提供可靠仿真平台。**

- **链接: [https://arxiv.org/pdf/2511.22187v2](https://arxiv.org/pdf/2511.22187v2)**

> **作者:** Qiang Li; Yingwenqi Jiang; Tuoxi Li; Duyu Chen; Xiang Feng; Yucheng Ao; Shangyue Liu; Xingchen Yu; Youcheng Cai; Yumeng Liu; Yuexin Ma; Xin Hu; Li Liu; Yu Zhang; Linkun Xu; Bingtao Gao; Xueyuan Wang; Shuchang Zhou; Xianming Liu; Ligang Liu
>
> **摘要:** Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.
>
---
#### [replaced 145] Benchmarking pig detection and tracking under diverse and challenging conditions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.16639v2](https://arxiv.org/pdf/2507.16639v2)**

> **作者:** Jonathan Henrich; Christian Post; Maximilian Zilke; Parth Shiroya; Emma Chanut; Amir Mollazadeh Yamchi; Ramin Yahyapour; Thomas Kneib; Imke Traulsen
>
> **备注:** 16 pages, 6 figures and 8 tables
>
> **摘要:** To ensure animal welfare and effective management in pig farming, monitoring individual behavior is a crucial prerequisite. While monitoring tasks have traditionally been carried out manually, advances in machine learning have made it possible to collect individualized information in an increasingly automated way. Central to these methods is the localization of animals across space (object detection) and time (multi-object tracking). Despite extensive research of these two tasks in pig farming, a systematic benchmarking study has not yet been conducted. In this work, we address this gap by curating two datasets: PigDetect for object detection and PigTrack for multi-object tracking. The datasets are based on diverse image and video material from realistic barn conditions, and include challenging scenarios such as occlusions or bad visibility. For object detection, we show that challenging training images improve detection performance beyond what is achievable with randomly sampled images alone. Comparing different approaches, we found that state-of-the-art models offer substantial improvements in detection quality over real-time alternatives. For multi-object tracking, we observed that SORT-based methods achieve superior detection performance compared to end-to-end trainable models. However, end-to-end models show better association performance, suggesting they could become strong alternatives in the future. We also investigate characteristic failure cases of end-to-end models, providing guidance for future improvements. The detection and tracking models trained on our datasets perform well in unseen pens, suggesting good generalization capabilities. This highlights the importance of high-quality training data. The datasets and research code are made publicly available to facilitate reproducibility, re-use and further development.
>
---
#### [replaced 146] PowerCLIP: Powerset Alignment for Contrastive Pre-Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23170v2](https://arxiv.org/pdf/2511.23170v2)**

> **作者:** Masaki Kawamura; Nakamasa Inoue; Rintaro Yanagi; Hirokatsu Kataoka; Rio Yokota
>
> **摘要:** Contrastive vision-language pre-training frameworks such as CLIP have demonstrated impressive zero-shot performance across a range of vision-language tasks. Recent studies have shown that aligning individual text tokens with specific image patches or regions enhances fine-grained compositional understanding. However, it remains challenging to capture compositional semantics that span multiple image regions. To address this limitation, we propose PowerCLIP, a novel contrastive pre-training framework enhanced by powerset alignment, which exhaustively optimizes region-to-phrase alignments by minimizing the loss defined between powersets of image regions and textual parse trees. Since the naive powerset construction incurs exponential computational cost due to the combinatorial explosion in the number of region subsets, we introduce efficient non-linear aggregators (NLAs) that reduce complexity from O(2^M) to O(M) with respect to the number of regions M, while approximating the exact loss value with arbitrary precision. Our extensive experiments demonstrate that PowerCLIP outperforms state-of-the-art methods in zero-shot classification and retrieval tasks, underscoring the compositionality and robustness of our approach. Our code will be made publicly available.
>
---
#### [replaced 147] ReasonEdit: Towards Reasoning-Enhanced Image Editing Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22625v2](https://arxiv.org/pdf/2511.22625v2)**

> **作者:** Fukun Yin; Shiyu Liu; Yucheng Han; Zhibo Wang; Peng Xing; Rui Wang; Wei Cheng; Yingming Wang; Aojie Li; Zixin Yin; Pengtao Chen; Xiangyu Zhang; Daxin Jiang; Xianfang Zeng; Gang Yu
>
> **备注:** code: https://github.com/stepfun-ai/Step1X-Edit
>
> **摘要:** Recent advances in image editing models have shown remarkable progress. A common architectural design couples a multimodal large language model (MLLM) encoder with a diffusion decoder, as seen in systems such as Step1X-Edit and Qwen-Image-Edit, where the MLLM encodes both the reference image and the instruction but remains frozen during training. In this work, we demonstrate that unlocking the reasoning capabilities of MLLM can further push the boundaries of editing models. Specifically, we explore two reasoning mechanisms, thinking and reflection, which enhance instruction understanding and editing accuracy. Based on that, our proposed framework enables image editing in a thinking-editing-reflection loop: the thinking mechanism leverages the world knowledge of MLLM to interpret abstract instructions, while the reflection reviews editing results, automatically corrects unintended manipulations, and identifies the stopping round. Extensive experiments demonstrate that our reasoning approach achieves significant performance gains, with improvements of ImgEdit (+4.3%), GEdit (+4.7%), and Kris (+8.2%) when initializing our DiT from the Step1X-Edit (ReasonEdit-S), and also outperforms previous open-source methods on both GEdit and Kris when integrated with Qwen-Image-Edit (ReasonEdit-Q).
>
---
#### [replaced 148] Adaptive Plane Reformatting for 4D Flow MRI using Deep Reinforcement Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.00727v2](https://arxiv.org/pdf/2506.00727v2)**

> **作者:** Javier Bisbal; Julio Sotelo; Maria I Valdés; Pablo Irarrazaval; Marcelo E Andia; Julio García; José Rodriguez-Palomarez; Francesca Raimondi; Cristián Tejos; Sergio Uribe
>
> **摘要:** Background and Objective: Plane reformatting for four-dimensional phase contrast MRI (4D flow MRI) is time-consuming and prone to inter-observer variability, which limits fast cardiovascular flow assessment. Deep reinforcement learning (DRL) trains agents to iteratively adjust plane position and orientation, enabling accurate plane reformatting without the need for detailed landmarks, making it suitable for images with limited contrast and resolution such as 4D flow MRI. However, current DRL methods assume that test volumes share the same spatial alignment as the training data, limiting generalization across scanners and institutions. To address this limitation, we introduce AdaPR (Adaptive Plane Reformatting), a DRL framework that uses a local coordinate system to navigate volumes with arbitrary positions and orientations. Methods: We implemented AdaPR using the Asynchronous Advantage Actor-Critic (A3C) algorithm and validated it on 88 4D flow MRI datasets acquired from multiple vendors, including patients with congenital heart disease. Results: AdaPR achieved a mean angular error of 6.32 +/- 4.15 degrees and a distance error of 3.40 +/- 2.75 mm, outperforming global-coordinate DRL methods and alternative non-DRL methods. AdaPR maintained consistent accuracy under different volume orientations and positions. Flow measurements from AdaPR planes showed no significant differences compared to two manual observers, with excellent correlation (R^2 = 0.972 and R^2 = 0.968), comparable to inter-observer agreement (R^2 = 0.969). Conclusion: AdaPR provides robust, orientation-independent plane reformatting for 4D flow MRI, achieving flow quantification comparable to expert observers. Its adaptability across datasets and scanners makes it a promising candidate for medical imaging applications beyond 4D flow MRI.
>
---
#### [replaced 149] MMIF-AMIN: Adaptive Loss-Driven Multi-Scale Invertible Dense Network for Multimodal Medical Image Fusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.08679v2](https://arxiv.org/pdf/2508.08679v2)**

> **作者:** Tao Luo; Weihua Xu
>
> **备注:** This manuscript is withdrawn to allow for substantial expansion and restructuring. Based on recent research progress, we plan to add Generalization experiment and reorganize the manuscript structure to improve readability and logical flow. Thank you for your understanding and support
>
> **摘要:** Multimodal medical image fusion (MMIF) aims to integrate images from different modalities to produce a comprehensive image that enhances medical diagnosis by accurately depicting organ structures, tissue textures, and metabolic information. Capturing both the unique and complementary information across multiple modalities simultaneously is a key research challenge in MMIF. To address this challenge, this paper proposes a novel image fusion method, MMIF-AMIN, which features a new architecture that can effectively extract these unique and complementary features. Specifically, an Invertible Dense Network (IDN) is employed for lossless feature extraction from individual modalities. To extract complementary information between modalities, a Multi-scale Complementary Feature Extraction Module (MCFEM) is designed, which incorporates a hybrid attention mechanism, convolutional layers of varying sizes, and Transformers. An adaptive loss function is introduced to guide model learning, addressing the limitations of traditional manually-designed loss functions and enhancing the depth of data mining. Extensive experiments demonstrate that MMIF-AMIN outperforms nine state-of-the-art MMIF methods, delivering superior results in both quantitative and qualitative analyses. Ablation experiments confirm the effectiveness of each component of the proposed method. Additionally, extending MMIF-AMIN to other image fusion tasks also achieves promising performance.
>
---
#### [replaced 150] HiMo: High-Speed Objects Motion Compensation in Point Clouds
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中LiDAR点云因高速动态物体引起的运动畸变问题，提出HiMo框架，通过改进的场景流估计实现非自车运动补偿。提出SeFlow++实现实时高精度场景流估计，并引入新评估指标验证效果，显著提升点云几何一致性和下游任务性能。**

- **链接: [https://arxiv.org/pdf/2503.00803v3](https://arxiv.org/pdf/2503.00803v3)**

> **作者:** Qingwen Zhang; Ajinkya Khoche; Yi Yang; Li Ling; Sina Sharif Mansouri; Olov Andersson; Patric Jensfelt
>
> **备注:** 15 pages, 13 figures, Published in Transactions on Robotics (Volume 41)
>
> **摘要:** LiDAR point cloud is essential for autonomous vehicles, but motion distortions from dynamic objects degrade the data quality. While previous work has considered distortions caused by ego motion, distortions caused by other moving objects remain largely overlooked, leading to errors in object shape and position. This distortion is particularly pronounced in high-speed environments such as highways and in multi-LiDAR configurations, a common setup for heavy vehicles. To address this challenge, we introduce HiMo, a pipeline that repurposes scene flow estimation for non-ego motion compensation, correcting the representation of dynamic objects in point clouds. During the development of HiMo, we observed that existing self-supervised scene flow estimators often produce degenerate or inconsistent estimates under high-speed distortion. We further propose SeFlow++, a real-time scene flow estimator that achieves state-of-the-art performance on both scene flow and motion compensation. Since well-established motion distortion metrics are absent in the literature, we introduce two evaluation metrics: compensation accuracy at a point level and shape similarity of objects. We validate HiMo through extensive experiments on Argoverse 2, ZOD, and a newly collected real-world dataset featuring highway driving and multi-LiDAR-equipped heavy vehicles. Our findings show that HiMo improves the geometric consistency and visual fidelity of dynamic objects in LiDAR point clouds, benefiting downstream tasks such as semantic segmentation and 3D detection. See https://kin-zhang.github.io/HiMo for more details.
>
---
#### [replaced 151] Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究统一多模态模型中理解与生成的关系，针对“理解是否真正指导生成”这一核心问题，构建了去耦合的评估框架UniSandbox及合成数据集。通过实验证明存在理解-生成差距，提出通过显式链式思维（CoT）和自训练实现隐式推理，并揭示查询架构的潜在CoT特性促进知识迁移。**

- **链接: [https://arxiv.org/pdf/2511.20561v2](https://arxiv.org/pdf/2511.20561v2)**

> **作者:** Yuwei Niu; Weiyang Jin; Jiaqi Liao; Chaoran Feng; Peng Jin; Bin Lin; Zongjian Li; Bin Zhu; Weihao Yu; Li Yuan
>
> **摘要:** Recent years have witnessed significant progress in Unified Multimodal Models, yet a fundamental question remains: Does understanding truly inform generation? To investigate this, we introduce UniSandbox, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: reasoning generation and knowledge transfer. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned knowledge, and also discover that query-based architectures inherently exhibit latent CoT-like properties that affect this transfer. UniSandbox provides preliminary insights for designing future unified architectures and training strategies that truly bridge the gap between understanding and generation. Code and data are available at https://github.com/PKU-YuanGroup/UniSandBox
>
---
#### [replaced 152] AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦无人机场景下的指代多目标跟踪（RMOT）任务，旨在解决地面场景下视角局限、难以实现广域感知与路径规划的问题。提出首个大规模无人机RMOT基准数据集AerialMind，并开发COALA标注框架降低人工成本；同时提出HETrack方法，通过视觉-语言协同学习提升空中场景理解能力。**

- **链接: [https://arxiv.org/pdf/2511.21053v2](https://arxiv.org/pdf/2511.21053v2)**

> **作者:** Chenglizhao Chen; Shaofeng Liang; Runwei Guan; Xiaolou Sun; Haocheng Zhao; Haiyun Jiang; Tao Huang; Henghui Ding; Qing-Long Han
>
> **备注:** AAAI 2026
>
> **摘要:** Referring Multi-Object Tracking (RMOT) aims to achieve precise object detection and tracking through natural language instructions, representing a fundamental capability for intelligent robotic systems. However, current RMOT research remains mostly confined to ground-level scenarios, which constrains their ability to capture broad-scale scene contexts and perform comprehensive tracking and path planning. In contrast, Unmanned Aerial Vehicles (UAVs) leverage their expansive aerial perspectives and superior maneuverability to enable wide-area surveillance. Moreover, UAVs have emerged as critical platforms for Embodied Intelligence, which has given rise to an unprecedented demand for intelligent aerial systems capable of natural language interaction. To this end, we introduce AerialMind, the first large-scale RMOT benchmark in UAV scenarios, which aims to bridge this research gap. To facilitate its construction, we develop an innovative semi-automated collaborative agent-based labeling assistant (COALA) framework that significantly reduces labor costs while maintaining annotation quality. Furthermore, we propose HawkEyeTrack (HETrack), a novel method that collaboratively enhances vision-language representation learning and improves the perception of UAV scenarios. Comprehensive experiments validated the challenging nature of our dataset and the effectiveness of our method.
>
---
#### [replaced 153] Physics-Informed Image Restoration via Progressive PDE Integration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06244v2](https://arxiv.org/pdf/2511.06244v2)**

> **作者:** Shamika Likhite; Santiago López-Tapia; Aggelos K. Katsaggelos
>
> **摘要:** Motion blur, caused by relative movement between camera and scene during exposure, significantly degrades image quality and impairs downstream computer vision tasks such as object detection, tracking, and recognition in dynamic environments. While deep learning-based motion deblurring methods have achieved remarkable progress, existing approaches face fundamental challenges in capturing the long-range spatial dependencies inherent in motion blur patterns. Traditional convolutional methods rely on limited receptive fields and require extremely deep networks to model global spatial relationships. These limitations motivate the need for alternative approaches that incorporate physical priors to guide feature evolution during restoration. In this paper, we propose a progressive training framework that integrates physics-informed PDE dynamics into state-of-the-art restoration architectures. By leveraging advection-diffusion equations to model feature evolution, our approach naturally captures the directional flow characteristics of motion blur while enabling principled global spatial modeling. Our PDE-enhanced deblurring models achieve superior restoration quality with minimal overhead, adding only approximately 1\% to inference GMACs while providing consistent improvements in perceptual quality across multiple state-of-the-art architectures. Comprehensive experiments on standard motion deblurring benchmarks demonstrate that our physics-informed approach improves PSNR and SSIM significantly across four diverse architectures, including FFTformer, NAFNet, Restormer, and Stripformer. These results validate that incorporating mathematical physics principles through PDE-based global layers can enhance deep learning-based image restoration, establishing a promising direction for physics-informed neural network design in computer vision applications.
>
---
#### [replaced 154] Towards Fast and Scalable Normal Integration using Continuous Components
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.11508v2](https://arxiv.org/pdf/2510.11508v2)**

> **作者:** Francesco Milano; Jen Jen Chung; Lionel Ott; Roland Siegwart
>
> **备注:** Accepted by the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026, first round. Camera-ready version. 17 pages, 9 figures, 6 tables. Code is available at https://github.com/francescomilano172/normal_integration_continuous_components
>
> **摘要:** Surface normal integration is a fundamental problem in computer vision, dealing with the objective of reconstructing a surface from its corresponding normal map. Existing approaches require an iterative global optimization to jointly estimate the depth of each pixel, which scales poorly to larger normal maps. In this paper, we address this problem by recasting normal integration as the estimation of relative scales of continuous components. By constraining pixels belonging to the same component to jointly vary their scale, we drastically reduce the number of optimization variables. Our framework includes a heuristic to accurately estimate continuous components from the start, a strategy to rebalance optimization terms, and a technique to iteratively merge components to further reduce the size of the problem. Our method achieves state-of-the-art results on the standard normal integration benchmark in as little as a few seconds and achieves one-order-of-magnitude speedup over pixel-level approaches on large-resolution normal maps.
>
---
#### [replaced 155] Hi-EF: Benchmarking Emotion Forecasting in Human-interaction
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2407.16406v2](https://arxiv.org/pdf/2407.16406v2)**

> **作者:** Haoran Wang; Xinji Mai; Zeng Tao; Junxiong Lin; Xuan Tong; Ivy Pan; Shaoqi Yan; Yan Wang; Shuyong Gao
>
> **摘要:** Affective Forecasting is an psychology task that involves predicting an individual's future emotional responses, often hampered by reliance on external factors leading to inaccuracies, and typically remains at a qualitative analysis stage. To address these challenges, we narrows the scope of Affective Forecasting by introducing the concept of Human-interaction-based Emotion Forecasting (EF). This task is set within the context of a two-party interaction, positing that an individual's emotions are significantly influenced by their interaction partner's emotional expressions and informational cues. This dynamic provides a structured perspective for exploring the patterns of emotional change, thereby enhancing the feasibility of emotion forecasting.
>
---
#### [replaced 156] 3-Tracer: A Tri-level Temporal-Aware Framework for Audio Forgery Detection and Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21237v2](https://arxiv.org/pdf/2511.21237v2)**

> **作者:** Shuhan Xia; Xuannan Liu; Xing Cui; Peipei Li
>
> **备注:** The experimental results in this paper have been further improved and updated; the baseline results do not match existing results, therefore the paper needs to be retracted
>
> **摘要:** Recently, partial audio forgery has emerged as a new form of audio manipulation. Attackers selectively modify partial but semantically critical frames while preserving the overall perceptual authenticity, making such forgeries particularly difficult to detect. Existing methods focus on independently detecting whether a single frame is forged, lacking the hierarchical structure to capture both transient and sustained anomalies across different temporal levels. To address these limitations, We identify three key levels relevant to partial audio forgery detection and present T3-Tracer, the first framework that jointly analyzes audio at the frame, segment, and audio levels to comprehensively detect forgery traces. T3-Tracer consists of two complementary core modules: the Frame-Audio Feature Aggregation Module (FA-FAM) and the Segment-level Multi-Scale Discrepancy-Aware Module (SMDAM). FA-FAM is designed to detect the authenticity of each audio frame. It combines both frame-level and audio-level temporal information to detect intra-frame forgery cues and global semantic inconsistencies. To further refine and correct frame detection, we introduce SMDAM to detect forgery boundaries at the segment level. It adopts a dual-branch architecture that jointly models frame features and inter-frame differences across multi-scale temporal windows, effectively identifying abrupt anomalies that appeared on the forged boundaries. Extensive experiments conducted on three challenging datasets demonstrate that our approach achieves state-of-the-art performance.
>
---
#### [replaced 157] Explainable Deep Convolutional Multi-Type Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11165v2](https://arxiv.org/pdf/2511.11165v2)**

> **作者:** Alex George; Lyudmila Mihaylova; Sean Anderson
>
> **摘要:** Explainable anomaly detection methods often have the capability to identify and spatially localise anomalies within an image but lack the capability to differentiate the type of anomaly. Furthermore, they often require the costly training and maintenance of separate models for each object category. The lack of specificity is a significant research gap because identifying the type of anomaly (e.g., "Crack" vs. "Scratch") is crucial for accurate diagnosis that facilitates cost-saving operational decisions across diverse application domains. While some recent large-scale Vision-Language Models (VLMs) have begun to address this, they are computationally intensive and memory-heavy, restricting their use in real-time or embedded systems. We propose MultiTypeFCDD, a simple and lightweight convolutional framework designed as a practical alternative for explainable multi-type anomaly detection. MultiTypeFCDD uses only image-level labels to learn and produce multi-channel heatmaps, where each channel is trained to correspond to a specific anomaly type. The model functions as a single, unified framework capable of differentiating anomaly types across multiple object categories, eliminating the need to train and manage separate models for each object category. We evaluated our proposed method on the Real-IAD dataset and it delivers competitive results (96.4% I-AUROC) at just over 1% the size of state-of-the-art VLM models used for similar tasks. This makes it a highly practical and viable solution for real-world applications where computational resources are tightly constrained.
>
---
#### [replaced 158] HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22544v3](https://arxiv.org/pdf/2509.22544v3)**

> **作者:** Mohammad Mahdi Hemmatyar; Mahdi Jafari; Mohammad Amin Yousefi; Mohammad Reza Nemati; Mobin Azadani; Hamid Reza Rastad; Amirmohammad Akbari
>
> **备注:** 25 pages, 1 figure
>
> **摘要:** Video anomaly detection (VAD) is crucial for intelligent surveillance, but a significant challenge lies in identifying complex anomalies, which are events defined by intricate relationships and temporal dependencies among multiple entities rather than by isolated actions. While self-supervised learning (SSL) methods effectively model low-level spatiotemporal patterns, they often struggle to grasp the semantic meaning of these interactions. Conversely, large language models (LLMs) offer powerful contextual reasoning but are computationally expensive for frame-by-frame analysis and lack fine-grained spatial localization. We introduce HyCoVAD, Hybrid Complex Video Anomaly Detection, a hybrid SSL-LLM model that combines a multi-task SSL temporal analyzer with LLM validator. The SSL module is built upon an nnFormer backbone which is a transformer-based model for image segmentation. It is trained with multiple proxy tasks, learns from video frames to identify those suspected of anomaly. The selected frames are then forwarded to the LLM, which enriches the analysis with semantic context by applying structured, rule-based reasoning to validate the presence of anomalies. Experiments on the challenging ComplexVAD dataset show that HyCoVAD achieves a 72.5% frame-level AUC, outperforming existing baselines by 12.5% while reducing LLM computation. We release our interaction anomaly taxonomy, adaptive thresholding protocol, and code to facilitate future research in complex VAD scenarios.
>
---
#### [replaced 159] Cross-Cancer Knowledge Transfer in WSI-based Prognosis Prediction
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13482v3](https://arxiv.org/pdf/2508.13482v3)**

> **作者:** Pei Liu; Luping Ji; Jiaxiang Gou; Xiangxiang Zeng
>
> **备注:** 24 pages (11 figures and 10 tables)
>
> **摘要:** Whole-Slide Image (WSI) is an important tool for estimating cancer prognosis. Current studies generally follow a conventional cancer-specific paradigm in which each cancer corresponds to a single model. However, this paradigm naturally struggles to scale to rare tumors and cannot leverage knowledge from other cancers. While multi-task learning frameworks have been explored recently, they often place high demands on computational resources and require extensive training on ultra-large, multi-cancer WSI datasets. To this end, this paper shifts the paradigm to knowledge transfer and presents the first preliminary yet systematic study on cross-cancer prognosis knowledge transfer in WSIs, called CROPKT. It comprises three major parts. (1) We curate a large dataset (UNI2-h-DSS) with 26 cancers and use it to measure the transferability of WSI-based prognostic knowledge across different cancers (including rare tumors). (2) Beyond a simple evaluation merely for benchmarking, we design a range of experiments to gain deeper insights into the underlying mechanism behind transferability. (3) We further show the utility of cross-cancer knowledge transfer, by proposing a routing-based baseline approach (ROUPKT) that could often efficiently utilize the knowledge transferred from off-the-shelf models of other cancers. CROPKT could serve as an inception that lays the foundation for this nascent paradigm, i.e., WSI-based prognosis prediction with cross-cancer knowledge transfer. Our source code is available at https://github.com/liupei101/CROPKT.
>
---
#### [replaced 160] Continuous Perception Matters: Diagnosing Temporal Integration Failures in Multimodal Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.07867v2](https://arxiv.org/pdf/2408.07867v2)**

> **作者:** Zeyu Wang; Zhenzhen Weng; Serena Yeung-Levy
>
> **摘要:** Continuous perception, the ability to integrate visual observations over time in a continuous stream fashion, is essential for robust real-world understanding, yet remains largely untested in current multimodal models. We introduce CP-Bench, a minimal and fully controlled benchmark designed to isolate this capability using an extremely simple task: counting identical cubes in a synthetic scene while the camera moves and only reveals subsets of objects at any moment. Despite the simplicity of the setting, we find that state-of-the-art open-source and commercial models, including Qwen-3-VL, InternVL3, GPT-5, and Gemini-3-Pro, fail dramatically. A static-camera control variant confirms that the failure arises not from object recognition but from an inability to accumulate evidence across time. Further experiments show that neither higher sampling FPS, perception- or spatial-enhanced models, nor finetuning with additional videos leads to meaningful cross-temporal generalization. Our results reveal a fundamental limitation in modern multimodal architectures and training paradigms. CP-Bench provides a simple yet powerful diagnostic tool and establishes a clean testbed for developing models capable of genuine time-consistent visual reasoning.
>
---
#### [replaced 161] Global-to-local image quality assessment in optical microscopy via fast and robust deep learning predictions
- **分类: cs.CV; physics.data-an; q-bio.QM**

- **链接: [https://arxiv.org/pdf/2510.04859v2](https://arxiv.org/pdf/2510.04859v2)**

> **作者:** Elena Corbetta; Thomas Bocklitz
>
> **备注:** 16 pages, 6 figures. μDeepIQA is publicly available at https://git.photonicdata.science/elena.corbetta/udeepiqa
>
> **摘要:** Optical microscopy is one of the most widely used techniques in research studies for life sciences and biomedicine. These applications require reliable experimental pipelines to extract valuable knowledge from the measured samples and must be supported by image quality assessment (IQA) to ensure correct processing and analysis of the image data. IQA methods are implemented with variable complexity. However, while most quality metrics have a straightforward implementation, they might be time consuming and computationally expensive when evaluating a large dataset. In addition, quality metrics are often designed for well-defined image features and may be unstable for images out of the ideal domain. To overcome these limitations, recent works have proposed deep learning-based IQA methods, which can provide superior performance, increased generalizability and fast prediction. Our method, named $\mathrmμ$DeepIQA, is inspired by previous studies and applies a deep convolutional neural network designed for IQA on natural images to optical microscopy measurements. We retrained the same architecture to predict individual quality metrics and global quality scores for optical microscopy data. The resulting models provide fast and stable predictions of image quality by generalizing quality estimation even outside the ideal range of standard methods. In addition, $\mathrmμ$DeepIQA provides patch-wise prediction of image quality and can be used to visualize spatially varying quality in a single image. Our study demonstrates that optical microscopy-based studies can benefit from the generalizability of deep learning models due to their stable performance in the presence of outliers, the ability to assess small image patches, and rapid predictions.
>
---
#### [replaced 162] AttnRegDeepLab: A Two-Stage Decoupled Framework for Interpretable Embryo Fragmentation Grading
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.18454v2](https://arxiv.org/pdf/2511.18454v2)**

> **作者:** Ming-Jhe Lee
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Embryo fragmentation is a morphological indicator critical for evaluating developmental potential in In Vitro Fertilization (IVF). However, manual grading is subjective and inefficient, while existing deep learning solutions often lack clinical explainability or suffer from accumulated errors in segmentation area estimation. To address these issues, this study proposes AttnRegDeepLab (Attention-Guided Regression DeepLab), a framework characterized by dual-branch Multi-Task Learning (MTL). A vanilla DeepLabV3+ decoder is modified by integrating Attention Gates into its skip connections, explicitly suppressing cytoplasmic noise to preserve contour details. Furthermore, a Multi-Scale Regression Head is introduced with a Feature Injection mechanism to propagate global grading priors into the segmentation task, rectifying systematic quantification errors. A 2-stage decoupled training strategy is proposed to address the gradient conflict in MTL. Also, a range-based loss is designed to leverage weakly labeled data. Our method achieves robust grading precision while maintaining excellent segmentation accuracy (Dice coefficient =0.729), in contrast to the end-to-end counterpart that might minimize grading error at the expense of contour integrity. This work provides a clinically interpretable solution that balances visual fidelity and quantitative precision.
>
---
#### [replaced 163] PIF-Net: Ill-Posed Prior Guided Multispectral and Hyperspectral Image Fusion via Invertible Mamba and Fusion-Aware LoRA
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.00453v2](https://arxiv.org/pdf/2508.00453v2)**

> **作者:** Baisong Li; Xingwang Wang; Haixiao Xu
>
> **摘要:** The goal of multispectral and hyperspectral image fusion (MHIF) is to generate high-quality images that simultaneously possess rich spectral information and fine spatial details. However, due to the inherent trade-off between spectral and spatial information and the limited availability of observations, this task is fundamentally ill-posed. Previous studies have not effectively addressed the ill-posed nature caused by data misalignment. To tackle this challenge, we propose a fusion framework named PIF-Net, which explicitly incorporates ill-posed priors to effectively fuse multispectral images and hyperspectral images. To balance global spectral modeling with computational efficiency, we design a method based on an invertible Mamba architecture that maintains information consistency during feature transformation and fusion, ensuring stable gradient flow and process reversibility. Furthermore, we introduce a novel fusion module called the Fusion-Aware Low-Rank Adaptation module, which dynamically calibrates spectral and spatial features while keeping the model lightweight. Extensive experiments on multiple benchmark datasets demonstrate that PIF-Net achieves significantly better image restoration performance than current state-of-the-art methods while maintaining model efficiency.
>
---
#### [replaced 164] Speech Audio Generation from dynamic MRI via a Knowledge Enhanced Conditional Variational Autoencoder
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于语音生成任务，旨在从动态MRI序列中恢复受损语音。针对MRI采集环境导致的数据丢失与噪声问题，提出知识增强的条件变分自编码器（KE-CVAE），通过无标签数据增强与变分推理提升生成质量，实现高保真语音重建，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2503.06588v2](https://arxiv.org/pdf/2503.06588v2)**

> **作者:** Yaxuan Li; Han Jiang; Yifei Ma; Shihua Qin; Jonghye Woo; Fangxu Xing
>
> **摘要:** Dynamic Magnetic Resonance Imaging (MRI) of the vocal tract has become an increasingly adopted imaging modality for speech motor studies. Beyond image signals, systematic data loss, noise pollution, and audio file corruption can occur due to the unpredictability of the MRI acquisition environment. In such cases, generating audio from images is critical for data recovery in both clinical and research applications. However, this remains challenging due to hardware constraints, acoustic interference, and data corruption. Existing solutions, such as denoising and multi-stage synthesis methods, face limitations in audio fidelity and generalizability. To address these challenges, we propose a Knowledge Enhanced Conditional Variational Autoencoder (KE-CVAE), a novel two-step "knowledge enhancement + variational inference" framework for generating speech audio signals from cine dynamic MRI sequences. This approach introduces two key innovations: (1) integration of unlabeled MRI data for knowledge enhancement, and (2) a variational inference architecture to improve generative modeling capacity. To the best of our knowledge, this is one of the first attempts at synthesizing speech audio directly from dynamic MRI video sequences. The proposed method was trained and evaluated on an open-source dynamic vocal tract MRI dataset recorded during speech. Experimental results demonstrate its effectiveness in generating natural speech waveforms while addressing MRI-specific acoustic challenges, outperforming conventional deep learning-based synthesis approaches.
>
---
