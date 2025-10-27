# 计算机视觉 cs.CV

- **最新发布 88 篇**

- **更新 101 篇**

## 最新发布

#### [new 001] Head Pursuit: Probing Attention Specialization in Multimodal Transformers
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究多模态Transformer中注意力头的语义专一性，通过信号处理视角分析中间激活，提出可解释的头级概念探测方法。工作包括头的重要性排序与精准编辑（仅改1%头），实现对生成内容中特定概念的可控抑制或增强，在问答、毒性检测、图像分类与描述等任务中验证有效性，揭示了注意力层的可解释与可操控结构。**

- **链接: [http://arxiv.org/pdf/2510.21518v1](http://arxiv.org/pdf/2510.21518v1)**

> **作者:** Lorenzo Basile; Valentino Maiorca; Diego Doimo; Francesco Locatello; Alberto Cazzaniga
>
> **备注:** Accepted at NeurIPS 2025 (spotlight)
>
> **摘要:** Language and vision-language models have shown impressive performance across a wide range of tasks, but their internal mechanisms remain only partly understood. In this work, we study how individual attention heads in text-generative models specialize in specific semantic or visual attributes. Building on an established interpretability method, we reinterpret the practice of probing intermediate activations with the final decoding layer through the lens of signal processing. This lets us analyze multiple samples in a principled way and rank attention heads based on their relevance to target concepts. Our results show consistent patterns of specialization at the head level across both unimodal and multimodal transformers. Remarkably, we find that editing as few as 1% of the heads, selected using our method, can reliably suppress or enhance targeted concepts in the model output. We validate our approach on language tasks such as question answering and toxicity mitigation, as well as vision-language tasks including image classification and captioning. Our findings highlight an interpretable and controllable structure within attention layers, offering simple tools for understanding and editing large-scale generative models.
>
---
#### [new 002] ArtiLatent: Realistic Articulated 3D Object Generation via Structured Latents
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出ArtiLatent，用于生成具有精细几何、准确关节运动和真实外观的可动3D物体。通过联合建模部件几何与关节属性，在统一潜空间中实现多样且物理合理的采样，并引入考虑关节状态的高斯解码器，提升不同姿态下的视觉真实感。任务为可动3D物体生成，解决现有方法在几何一致性与外观真实性上的不足。**

- **链接: [http://arxiv.org/pdf/2510.21432v1](http://arxiv.org/pdf/2510.21432v1)**

> **作者:** Honghua Chen; Yushi Lan; Yongwei Chen; Xingang Pan
>
> **备注:** accepted to SIGGRAPH Asia; Project page: https://chenhonghua.github.io/MyProjects/ArtiLatent/
>
> **摘要:** We propose ArtiLatent, a generative framework that synthesizes human-made 3D objects with fine-grained geometry, accurate articulation, and realistic appearance. Our approach jointly models part geometry and articulation dynamics by embedding sparse voxel representations and associated articulation properties, including joint type, axis, origin, range, and part category, into a unified latent space via a variational autoencoder. A latent diffusion model is then trained over this space to enable diverse yet physically plausible sampling. To reconstruct photorealistic 3D shapes, we introduce an articulation-aware Gaussian decoder that accounts for articulation-dependent visibility changes (e.g., revealing the interior of a drawer when opened). By conditioning appearance decoding on articulation state, our method assigns plausible texture features to regions that are typically occluded in static poses, significantly improving visual realism across articulation configurations. Extensive experiments on furniture-like objects from PartNet-Mobility and ACD datasets demonstrate that ArtiLatent outperforms existing approaches in geometric consistency and appearance fidelity. Our framework provides a scalable solution for articulated 3D object synthesis and manipulation.
>
---
#### [new 003] HistRetinex: Optimizing Retinex model in Histogram Domain for Efficient Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文针对低光图像增强任务，解决传统Retinex方法计算效率低的问题。提出基于直方图域的HistRetinex模型，通过构建两层优化框架，实现快速迭代求解，显著提升处理速度，同时保持优异增强效果。**

- **链接: [http://arxiv.org/pdf/2510.21100v1](http://arxiv.org/pdf/2510.21100v1)**

> **作者:** Jingtian Zhao; Xueli Xie; Jianxiang Xi; Xiaogang Yang; Haoxuan Sun
>
> **备注:** Currently, this manuscript has been rejected by TIP and is undergoing revisions. The reviewers noted that the paper contains some innovative aspects, but identified issues in the experimental and algorithmic sections
>
> **摘要:** Retinex-based low-light image enhancement methods are widely used due to their excellent performance. However, most of them are time-consuming for large-sized images. This paper extends the Retinex model from the spatial domain to the histogram domain, and proposes a novel histogram-based Retinex model for fast low-light image enhancement, named HistRetinex. Firstly, we define the histogram location matrix and the histogram count matrix, which establish the relationship among histograms of the illumination, reflectance and the low-light image. Secondly, based on the prior information and the histogram-based Retinex model, we construct a novel two-level optimization model. Through solving the optimization model, we give the iterative formulas of the illumination histogram and the reflectance histogram, respectively. Finally, we enhance the low-light image through matching its histogram with the one provided by HistRetinex. Experimental results demonstrate that the HistRetinex outperforms existing enhancement methods in both visibility and performance metrics, while executing 1.86 seconds on 1000*664 resolution images, achieving a minimum time saving of 6.67 seconds.
>
---
#### [new 004] Long-tailed Species Recognition in the NACTI Wildlife Dataset
- **分类: cs.CV**

- **简介: 该论文针对野生动物图像中长尾分布问题，提出改进的长尾识别方法。在NACTI数据集上，通过优化损失函数与正则化策略，显著提升分类准确率至99.40%，并验证了模型在域偏移下的更强泛化能力。研究强调了调度器与先进损失函数的协同作用，并公开全部代码与权重以保障可复现性。**

- **链接: [http://arxiv.org/pdf/2510.21657v1](http://arxiv.org/pdf/2510.21657v1)**

> **作者:** Zehua Liu; Tilo Burghardt
>
> **摘要:** As most ''in the wild'' data collections of the natural world, the North America Camera Trap Images (NACTI) dataset shows severe long-tailed class imbalance, noting that the largest 'Head' class alone covers >50% of the 3.7M images in the corpus. Building on the PyTorch Wildlife model, we present a systematic study of Long-Tail Recognition methodologies for species recognition on the NACTI dataset covering experiments on various LTR loss functions plus LTR-sensitive regularisation. Our best configuration achieves 99.40% Top-1 accuracy on our NACTI test data split, substantially improving over a 95.51% baseline using standard cross-entropy with Adam. This also improves on previously reported top performance in MLWIC2 at 96.8% albeit using partly unpublished (potentially different) partitioning, optimiser, and evaluation protocols. To evaluate domain shifts (e.g. night-time captures, occlusion, motion-blur) towards other datasets we construct a Reduced-Bias Test set from the ENA-Detection dataset where our experimentally optimised long-tail enhanced model achieves leading 52.55% accuracy (up from 51.20% with WCE loss), demonstrating stronger generalisation capabilities under distribution shift. We document the consistent improvements of LTR-enhancing scheduler choices in this NACTI wildlife domain, particularly when in tandem with state-of-the-art LTR losses. We finally discuss qualitative and quantitative shortcomings that LTR methods cannot sufficiently address, including catastrophic breakdown for 'Tail' classes under severe domain shift. For maximum reproducibility we publish all dataset splits, key code, and full network weights.
>
---
#### [new 005] 3DReasonKnee: Advancing Grounded Reasoning in Medical Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出3DReasonKnee，首个面向3D医学影像的接地推理数据集，解决当前视觉语言模型在3D医学图像中难以精准定位与逐步推理的问题。通过7,970例膝关节MRI构建494k高质量五元组数据，包含诊断问题、3D定位框、临床推理链与严重程度评估，建立基准测试平台，推动多模态医疗AI向临床可解释的三维决策发展。**

- **链接: [http://arxiv.org/pdf/2510.20967v1](http://arxiv.org/pdf/2510.20967v1)**

> **作者:** Sraavya Sambara; Sung Eun Kim; Xiaoman Zhang; Luyang Luo; Shreya Johri; Mohammed Baharoon; Du Hyun Ro; Pranav Rajpurkar
>
> **摘要:** Current Vision-Language Models (VLMs) struggle to ground anatomical regions in 3D medical images and reason about them in a step-by-step manner, a key requirement of real-world diagnostic assessment. This ability is essential for aligning model outputs with the diagnostic workflows clinicians use in practice, enabling trustworthy clinician-AI collaboration. Existing 3D datasets provide localization labels, but none support this "grounded reasoning" ability. To address this gap, we introduce 3DReasonKnee, the first 3D grounded reasoning dataset for medical images, which provides 494k high-quality quintuples derived from 7,970 3D knee MRI volumes. Each quintuple includes: (1) the 3D MRI volume, (2) a diagnostic question targeting a specific anatomical region (3) a 3D bounding box localizing the relevant anatomical structures, (4) clinician-generated diagnostic reasoning steps that explicitly detail the 3D reasoning process, and (5) structured severity assessments for the relevant anatomical region. The creation and validation of 3DReasonKnee, involving over 450 hours of expert clinician time for manually segmenting MRIs and generating reasoning chains, ensures its superior quality and clinical relevance. We establish ReasonKnee-Bench to evaluate localization and diagnostic accuracy, providing insight into VLM ability to perform grounding and severity assessment across anatomical regions and diagnostic inquiries. We benchmark five state-of-the-art VLMs, providing baseline performance for ReasonKnee-Bench. By providing this unique resource of expert-annotated 3D reasoning pathways, 3DReasonKnee serves as a repository of orthopedic surgeons' diagnostic expertise and offers a vital testbed for advancing multimodal medical AI systems towards 3D, clinically aligned, localized decision-making capabilities. The dataset can be found in: https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee
>
---
#### [new 006] Video-As-Prompt: Unified Semantic Control for Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Video-As-Prompt（VAP），解决视频生成中统一、泛化性强的语义控制难题。通过将参考视频作为语义提示，利用可插拔的MoT专家引导冻结的DiT模型，实现零样本泛化。构建了包含10万+对数据的VAP-Data数据集，显著提升开源方法性能，推动通用可控视频生成发展。**

- **链接: [http://arxiv.org/pdf/2510.20888v1](http://arxiv.org/pdf/2510.20888v1)**

> **作者:** Yuxuan Bian; Xin Chen; Zenan Li; Tiancheng Zhi; Shen Sang; Linjie Luo; Qiang Xu
>
> **备注:** Website: https://bytedance.github.io/Video-As-Prompt
>
> **摘要:** Unified, generalizable semantic control in video generation remains a critical open challenge. Existing methods either introduce artifacts by enforcing inappropriate pixel-wise priors from structure-based controls, or rely on non-generalizable, condition-specific finetuning or task-specific architectures. We introduce Video-As-Prompt (VAP), a new paradigm that reframes this problem as in-context generation. VAP leverages a reference video as a direct semantic prompt, guiding a frozen Video Diffusion Transformer (DiT) via a plug-and-play Mixture-of-Transformers (MoT) expert. This architecture prevents catastrophic forgetting and is guided by a temporally biased position embedding that eliminates spurious mapping priors for robust context retrieval. To power this approach and catalyze future research, we built VAP-Data, the largest dataset for semantic-controlled video generation with over 100K paired videos across 100 semantic conditions. As a single unified model, VAP sets a new state-of-the-art for open-source methods, achieving a 38.7% user preference rate that rivals leading condition-specific commercial models. VAP's strong zero-shot generalization and support for various downstream applications mark a significant advance toward general-purpose, controllable video generation.
>
---
#### [new 007] WorldGrow: Generating Infinite 3D World
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出WorldGrow，用于生成无限扩展的3D世界。针对现有方法在几何一致性、可扩展性和场景级生成上的局限，提出基于预训练3D模型的分层框架，结合高质量场景块数据、上下文感知的块补全与粗到细生成策略，实现大尺度、结构一致且逼真的3D场景合成。**

- **链接: [http://arxiv.org/pdf/2510.21682v1](http://arxiv.org/pdf/2510.21682v1)**

> **作者:** Sikuang Li; Chen Yang; Jiemin Fang; Taoran Yi; Jia Lu; Jiazhong Cen; Lingxi Xie; Wei Shen; Qi Tian
>
> **备注:** Project page: https://world-grow.github.io/ Code: https://github.com/world-grow/WorldGrow
>
> **摘要:** We tackle the challenge of generating the infinitely extendable 3D world -- large, continuous environments with coherent geometry and realistic appearance. Existing methods face key challenges: 2D-lifting approaches suffer from geometric and appearance inconsistencies across views, 3D implicit representations are hard to scale up, and current 3D foundation models are mostly object-centric, limiting their applicability to scene-level generation. Our key insight is leveraging strong generation priors from pre-trained 3D models for structured scene block generation. To this end, we propose WorldGrow, a hierarchical framework for unbounded 3D scene synthesis. Our method features three core components: (1) a data curation pipeline that extracts high-quality scene blocks for training, making the 3D structured latent representations suitable for scene generation; (2) a 3D block inpainting mechanism that enables context-aware scene extension; and (3) a coarse-to-fine generation strategy that ensures both global layout plausibility and local geometric/textural fidelity. Evaluated on the large-scale 3D-FRONT dataset, WorldGrow achieves SOTA performance in geometry reconstruction, while uniquely supporting infinite scene generation with photorealistic and structurally consistent outputs. These results highlight its capability for constructing large-scale virtual environments and potential for building future world models.
>
---
#### [new 008] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文针对在线视频异常检测（Online VAD）难题，提出MoniTor框架。利用预训练视觉语言模型，结合记忆化评分队列与类LSTM预测机制，实现无需训练的实时异常识别，有效捕捉时序依赖并动态更新异常判断，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.21449v1](http://arxiv.org/pdf/2510.21449v1)**

> **作者:** Shengtian Yang; Yue Feng; Yingshi Liu; Jingrou Zhang; Jie Qin
>
> **备注:** Accepted to NeurIPS 2025. The first two authors hold equal contributions
>
> **摘要:** Video Anomaly Detection (VAD) aims to locate unusual activities or behaviors within videos. Recently, offline VAD has garnered substantial research attention, which has been invigorated by the progress in large language models (LLMs) and vision-language models (VLMs), offering the potential for a more nuanced understanding of anomalies. However, online VAD has seldom received attention due to real-time constraints and computational intensity. In this paper, we introduce a novel Memory-based online scoring queue scheme for Training-free VAD (MoniTor), to address the inherent complexities in online VAD. Specifically, MoniTor applies a streaming input to VLMs, leveraging the capabilities of pre-trained large-scale models. To capture temporal dependencies more effectively, we incorporate a novel prediction mechanism inspired by Long Short-Term Memory (LSTM) networks. This ensures the model can effectively model past states and leverage previous predictions to identify anomalous behaviors. Thereby, it better understands the current frame. Moreover, we design a scoring queue and an anomaly prior to dynamically store recent scores and cover all anomalies in the monitoring scenario, providing guidance for LLMs to distinguish between normal and abnormal behaviors over time. We evaluate MoniTor on two large datasets (i.e., UCF-Crime and XD-Violence) containing various surveillance and real-world scenarios. The results demonstrate that MoniTor outperforms state-of-the-art methods and is competitive with weakly supervised methods without training. Code is available at https://github.com/YsTvT/MoniTor.
>
---
#### [new 009] Generative Point Tracking with Flow Matching
- **分类: cs.CV**

- **简介: 该论文提出生成式点追踪框架GenPT，解决现有方法在遮挡等不确定场景下仅能回归单一轨迹、忽略多模态问题。通过流匹配训练，结合窗口依赖先验与坐标专用方差调度，实现多模态轨迹建模，并在推理时采用基于置信度的最优优先搜索提升精度。在多个基准上实现对遮挡点的领先追踪性能。**

- **链接: [http://arxiv.org/pdf/2510.20951v1](http://arxiv.org/pdf/2510.20951v1)**

> **作者:** Mattie Tesfaldet; Adam W. Harley; Konstantinos G. Derpanis; Derek Nowrouzezahrai; Christopher Pal
>
> **备注:** Project page: https://mtesfaldet.net/genpt_projpage/
>
> **摘要:** Tracking a point through a video can be a challenging task due to uncertainty arising from visual obfuscations, such as appearance changes and occlusions. Although current state-of-the-art discriminative models excel in regressing long-term point trajectory estimates -- even through occlusions -- they are limited to regressing to a mean (or mode) in the presence of uncertainty, and fail to capture multi-modality. To overcome this limitation, we introduce Generative Point Tracker (GenPT), a generative framework for modelling multi-modal trajectories. GenPT is trained with a novel flow matching formulation that combines the iterative refinement of discriminative trackers, a window-dependent prior for cross-window consistency, and a variance schedule tuned specifically for point coordinates. We show how our model's generative capabilities can be leveraged to improve point trajectory estimates by utilizing a best-first search strategy on generated samples during inference, guided by the model's own confidence of its predictions. Empirically, we evaluate GenPT against the current state of the art on the standard PointOdyssey, Dynamic Replica, and TAP-Vid benchmarks. Further, we introduce a TAP-Vid variant with additional occlusions to assess occluded point tracking performance and highlight our model's ability to capture multi-modality. GenPT is capable of capturing the multi-modality in point trajectories, which translates to state-of-the-art tracking accuracy on occluded points, while maintaining competitive tracking accuracy on visible points compared to extant discriminative point trackers.
>
---
#### [new 010] An Automatic Detection Method for Hematoma Features in Placental Abruption Ultrasound Images Based on Few-Shot Learning
- **分类: cs.CV; cs.NE**

- **简介: 该论文针对胎盘早剥超声图像中血肿特征自动检测任务，解决传统诊断依赖经验、存在主观偏差的问题。提出基于小样本学习的EH-YOLOv11n模型，融合波变换与坐标卷积增强特征提取，引入级联分组注意力机制提升定位精度，在低样本下实现78%检测准确率，显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.21495v1](http://arxiv.org/pdf/2510.21495v1)**

> **作者:** Xiaoqing Liu; Jitai Han; Hua Yan; Peng Li; Sida Tang; Ying Li; Kaiwen Zhang; Min Yu
>
> **摘要:** Placental abruption is a severe complication during pregnancy, and its early accurate diagnosis is crucial for ensuring maternal and fetal safety. Traditional ultrasound diagnostic methods heavily rely on physician experience, leading to issues such as subjective bias and diagnostic inconsistencies. This paper proposes an improved model, EH-YOLOv11n (Enhanced Hemorrhage-YOLOv11n), based on small-sample learning, aiming to achieve automatic detection of hematoma features in placental ultrasound images. The model enhances performance through multidimensional optimization: it integrates wavelet convolution and coordinate convolution to strengthen frequency and spatial feature extraction; incorporates a cascaded group attention mechanism to suppress ultrasound artifacts and occlusion interference, thereby improving bounding box localization accuracy. Experimental results demonstrate a detection accuracy of 78%, representing a 2.5% improvement over YOLOv11n and a 13.7% increase over YOLOv8. The model exhibits significant superiority in precision-recall curves, confidence scores, and occlusion scenarios. Combining high accuracy with real-time processing, this model provides a reliable solution for computer-aided diagnosis of placental abruption, holding significant clinical application value.
>
---
#### [new 011] Topology Sculptor, Shape Refiner: Discrete Diffusion Model for High-Fidelity 3D Meshes Generation
- **分类: cs.CV**

- **简介: 该论文提出TSSR，一种基于离散扩散模型的3D网格生成方法，旨在高效生成高保真艺术风格网格。针对传统序列模型效率低、难以兼顾全局与局部细节的问题，提出分阶段训练与混合推理、改进的双向注意力架构及连接损失，实现并行生成与高质量输出，可生成高达10,000面的精细网格。**

- **链接: [http://arxiv.org/pdf/2510.21264v1](http://arxiv.org/pdf/2510.21264v1)**

> **作者:** Kaiyu Song; Hanjiang Lai; Yaqing Zhang; Chuangjian Cai; Yan Pan Kun Yue; Jian Yin
>
> **摘要:** In this paper, we introduce Topology Sculptor, Shape Refiner (TSSR), a novel method for generating high-quality, artist-style 3D meshes based on Discrete Diffusion Models (DDMs). Our primary motivation for TSSR is to achieve highly accurate token prediction while enabling parallel generation, a significant advantage over sequential autoregressive methods. By allowing TSSR to "see" all mesh tokens concurrently, we unlock a new level of efficiency and control. We leverage this parallel generation capability through three key innovations: 1) Decoupled Training and Hybrid Inference, which distinctly separates the DDM-based generation into a topology sculpting stage and a subsequent shape refinement stage. This strategic decoupling enables TSSR to effectively capture both intricate local topology and overarching global shape. 2) An Improved Hourglass Architecture, featuring bidirectional attention enriched by face-vertex-sequence level Rotational Positional Embeddings (RoPE), thereby capturing richer contextual information across the mesh structure. 3) A novel Connection Loss, which acts as a topological constraint to further enhance the realism and fidelity of the generated meshes. Extensive experiments on complex datasets demonstrate that TSSR generates high-quality 3D artist-style meshes, capable of achieving up to 10,000 faces at a remarkable spatial resolution of $1024^3$. The code will be released at: https://github.com/psky1111/Tencent-TSSR.
>
---
#### [new 012] BachVid: Training-Free Video Generation with Consistent Background and Character
- **分类: cs.CV**

- **简介: 该论文提出BachVid，一个无需训练且不依赖参考图像的视频生成方法，解决多视频中角色与背景一致性难题。通过分析DiT的注意力机制，利用中间特征缓存实现身份视频的特征复用，确保生成视频在前景和背景上的一致性。**

- **链接: [http://arxiv.org/pdf/2510.21696v1](http://arxiv.org/pdf/2510.21696v1)**

> **作者:** Han Yan; Xibin Song; Yifu Wang; Hongdong Li; Pan Ji; Chao Ma
>
> **备注:** Project page: https://wolfball.github.io/bachvid
>
> **摘要:** Diffusion Transformers (DiTs) have recently driven significant progress in text-to-video (T2V) generation. However, generating multiple videos with consistent characters and backgrounds remains a significant challenge. Existing methods typically rely on reference images or extensive training, and often only address character consistency, leaving background consistency to image-to-video models. We introduce BachVid, the first training-free method that achieves consistent video generation without needing any reference images. Our approach is based on a systematic analysis of DiT's attention mechanism and intermediate features, revealing its ability to extract foreground masks and identify matching points during the denoising process. Our method leverages this finding by first generating an identity video and caching the intermediate variables, and then inject these cached variables into corresponding positions in newly generated videos, ensuring both foreground and background consistency across multiple videos. Experimental results demonstrate that BachVid achieves robust consistency in generated videos without requiring additional training, offering a novel and efficient solution for consistent video generation without relying on reference images or additional training.
>
---
#### [new 013] Bridging the gap to real-world language-grounded visual concept learning
- **分类: cs.CV**

- **简介: 该论文聚焦语言接地的视觉概念学习任务，旨在解决现有方法仅限于预定义简单属性（如颜色、形状）且依赖合成数据的问题。作者提出一种可扩展框架，通过通用提示策略自动发现真实场景中的多样视觉概念轴，并利用无额外参数的概念编码器实现概念绑定与独立操控，显著提升真实世界复杂概念的编辑与组合泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.21412v1](http://arxiv.org/pdf/2510.21412v1)**

> **作者:** Whie Jung; Semin Kim; Junee Kim; Seunghoon Hong
>
> **摘要:** Human intelligence effortlessly interprets visual scenes along a rich spectrum of semantic dimensions. However, existing approaches to language-grounded visual concept learning are limited to a few predefined primitive axes, such as color and shape, and are typically explored in synthetic datasets. In this work, we propose a scalable framework that adaptively identifies image-related concept axes and grounds visual concepts along these axes in real-world scenes. Leveraging a pretrained vision-language model and our universal prompting strategy, our framework identifies a diverse image-related axes without any prior knowledge. Our universal concept encoder adaptively binds visual features to the discovered axes without introducing additional model parameters for each concept. To ground visual concepts along the discovered axes, we optimize a compositional anchoring objective, which ensures that each axis can be independently manipulated without affecting others. We demonstrate the effectiveness of our framework on subsets of ImageNet, CelebA-HQ, and AFHQ, showcasing superior editing capabilities across diverse real-world concepts that are too varied to be manually predefined. Our method also exhibits strong compositional generalization, outperforming existing visual concept learning and text-based editing methods. The code is available at https://github.com/whieya/Language-grounded-VCL.
>
---
#### [new 014] Focal Modulation and Bidirectional Feature Fusion Network for Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医学图像分割任务，解决复杂边界与多尺度结构分割困难的问题。提出FM-BFF-Net，融合卷积与Transformer优势，引入焦点调制注意力和双向特征融合模块，增强全局上下文感知与跨尺度特征交互，显著提升分割精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.20933v1](http://arxiv.org/pdf/2510.20933v1)**

> **作者:** Moin Safdar; Shahzaib Iqbal; Mehwish Mehmood; Mubeen Ghafoor; Tariq M. Khan; Imran Razzak
>
> **摘要:** Medical image segmentation is essential for clinical applications such as disease diagnosis, treatment planning, and disease development monitoring because it provides precise morphological and spatial information on anatomical structures that directly influence treatment decisions. Convolutional neural networks significantly impact image segmentation; however, since convolution operations are local, capturing global contextual information and long-range dependencies is still challenging. Their capacity to precisely segment structures with complicated borders and a variety of sizes is impacted by this restriction. Since transformers use self-attention methods to capture global context and long-range dependencies efficiently, integrating transformer-based architecture with CNNs is a feasible approach to overcoming these challenges. To address these challenges, we propose the Focal Modulation and Bidirectional Feature Fusion Network for Medical Image Segmentation, referred to as FM-BFF-Net in the remainder of this paper. The network combines convolutional and transformer components, employs a focal modulation attention mechanism to refine context awareness, and introduces a bidirectional feature fusion module that enables efficient interaction between encoder and decoder representations across scales. Through this design, FM-BFF-Net enhances boundary precision and robustness to variations in lesion size, shape, and contrast. Extensive experiments on eight publicly available datasets, including polyp detection, skin lesion segmentation, and ultrasound imaging, show that FM-BFF-Net consistently surpasses recent state-of-the-art methods in Jaccard index and Dice coefficient, confirming its effectiveness and adaptability for diverse medical imaging scenarios.
>
---
#### [new 015] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文提出FineRS框架，针对多模态大模型在高分辨率图像中难以精准识别与分割极小物体的问题，设计了粗到细的两阶段强化学习方法。通过全局语义探索与局部感知精炼，联合实现细粒度推理与像素级分割，并构建新数据集验证性能，显著提升小物体定位与理解能力。**

- **链接: [http://arxiv.org/pdf/2510.21311v1](http://arxiv.org/pdf/2510.21311v1)**

> **作者:** Lu Zhang; Jiazuo Yu; Haomiao Xiong; Ping Hu; Yunzhi Zhuge; Huchuan Lu; You He
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Multi-modal Large Language Models (MLLMs) have shown remarkable capabilities across a wide range of vision-language tasks. However, due to the restricted input resolutions, MLLMs face significant challenges in precisely understanding and localizing visual details in high-resolution images -- particularly when dealing with extra-small objects embedded in cluttered contexts. To address this issue, we propose \textsc{FineRS}, a two-stage MLLM-based reinforcement learning framework for jointly reasoning and segmenting extremely small objects within high-resolution scenes. \textsc{FineRS} adopts a coarse-to-fine pipeline comprising Global Semantic Exploration (GSE) and Localized Perceptual Refinement (LPR). Specifically, GSE performs instruction-guided reasoning to generate a textural response and a coarse target region, while LPR refines this region to produce an accurate bounding box and segmentation mask. To couple the two stages, we introduce a locate-informed retrospective reward, where LPR's outputs are used to optimize GSE for more robust coarse region exploration. % Additionally, we present \textsc{FineRS}-4k, a new dataset for evaluating MLLMs on attribute-level reasoning and pixel-level segmentation on subtle, small-scale targets in complex high-resolution scenes. Experimental results on \textsc{FineRS}-4k and public datasets demonstrate that our method consistently outperforms state-of-the-art MLLM-based approaches on both instruction-guided segmentation and visual reasoning tasks.
>
---
#### [new 016] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出VESSA，一种基于多视角物体中心视频的自监督适配方法，用于视觉基础模型在分布偏移和标签稀缺场景下的无标注适应。通过自蒸馏与参数高效微调，提升模型对新域的鲁棒性，显著改善下游分类性能。**

- **链接: [http://arxiv.org/pdf/2510.20994v1](http://arxiv.org/pdf/2510.20994v1)**

> **作者:** Jesimon Barreto; Carlos Caetano; André Araujo; William Robson Schwartz
>
> **备注:** Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Foundation models have advanced computer vision by enabling strong performance across diverse tasks through large-scale pretraining and supervised fine-tuning. However, they may underperform in domains with distribution shifts and scarce labels, where supervised fine-tuning may be infeasible. While continued self-supervised learning for model adaptation is common for generative language models, this strategy has not proven effective for vision-centric encoder models. To address this challenge, we introduce a novel formulation of self-supervised fine-tuning for vision foundation models, where the model is adapted to a new domain without requiring annotations, leveraging only short multi-view object-centric videos. Our method is referred to as VESSA: Video-based objEct-centric Self-Supervised Adaptation for visual foundation models. VESSA's training technique is based on a self-distillation paradigm, where it is critical to carefully tune prediction heads and deploy parameter-efficient adaptation techniques - otherwise, the model may quickly forget its pretrained knowledge and reach a degraded state. VESSA benefits significantly from multi-view object observations sourced from different frames in an object-centric video, efficiently learning robustness to varied capture conditions, without the need of annotations. Through comprehensive experiments with 3 vision foundation models on 2 datasets, VESSA demonstrates consistent improvements in downstream classification tasks, compared to the base models and previous adaptation methods. Code is publicly available at https://github.com/jesimonbarreto/VESSA.
>
---
#### [new 017] Restore Text First, Enhance Image Later: Two-Stage Scene Text Image Super-Resolution with Glyph Structure Guidance
- **分类: cs.CV**

- **简介: 该论文针对场景文本图像超分辨率任务，解决现有方法在提升图像质量时损害文本可读性的难题。提出两阶段框架TIGER，先恢复精确字形结构，再以此引导图像增强，实现文本清晰与图像真实性的统一。同时构建了超大缩放比的UltraZoom-ST数据集。**

- **链接: [http://arxiv.org/pdf/2510.21590v1](http://arxiv.org/pdf/2510.21590v1)**

> **作者:** Minxing Luo; Linlong Fan; Wang Qiushi; Ge Wu; Yiyan Luo; Yuhang Yu; Jinwei Chen; Yaxing Wang; Qingnan Fan; Jian Yang
>
> **摘要:** Current generative super-resolution methods show strong performance on natural images but distort text, creating a fundamental trade-off between image quality and textual readability. To address this, we introduce \textbf{TIGER} (\textbf{T}ext-\textbf{I}mage \textbf{G}uided sup\textbf{E}r-\textbf{R}esolution), a novel two-stage framework that breaks this trade-off through a \textit{"text-first, image-later"} paradigm. \textbf{TIGER} explicitly decouples glyph restoration from image enhancement: it first reconstructs precise text structures and then uses them to guide subsequent full-image super-resolution. This glyph-to-image guidance ensures both high fidelity and visual consistency. To support comprehensive training and evaluation, we also contribute the \textbf{UltraZoom-ST} (UltraZoom-Scene Text), the first scene text dataset with extreme zoom (\textbf{$\times$14.29}). Extensive experiments show that \textbf{TIGER} achieves \textbf{state-of-the-art} performance, enhancing readability while preserving overall image quality.
>
---
#### [new 018] Automated interictal epileptic spike detection from simple and noisy annotations in MEG data
- **分类: cs.CV**

- **简介: 该论文针对药物难治性癫痫中脑磁图（MEG）数据的间歇性癫痫尖峰自动检测任务，解决手动标注耗时易错、现有自动化方法依赖大量标注或对非典型数据不鲁棒的问题。提出基于ANN和CNN的轻量模型，结合交互式机器学习策略，在仅需时间标记和单专家标注的现实临床数据上实现高效检测，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.21596v1](http://arxiv.org/pdf/2510.21596v1)**

> **作者:** Pauline Mouches; Julien Jung; Armand Demasson; Agnès Guinard; Romain Bouet; Rosalie Marchal; Romain Quentin
>
> **备注:** 17 pages, 7 Figures
>
> **摘要:** In drug-resistant epilepsy, presurgical evaluation of epilepsy can be considered. Magnetoencephalography (MEG) has been shown to be an effective exam to inform the localization of the epileptogenic zone through the localization of interictal epileptic spikes. Manual detection of these pathological biomarkers remains a fastidious and error-prone task due to the high dimensionality of MEG recordings, and interrater agreement has been reported to be only moderate. Current automated methods are unsuitable for clinical practice, either requiring extensively annotated data or lacking robustness on non-typical data. In this work, we demonstrate that deep learning models can be used for detecting interictal spikes in MEG recordings, even when only temporal and single-expert annotations are available, which represents real-world clinical practice. We propose two model architectures: a feature-based artificial neural network (ANN) and a convolutional neural network (CNN), trained on a database of 59 patients, and evaluated against a state-of-the-art model to classify short time windows of signal. In addition, we employ an interactive machine learning strategy to iteratively improve our data annotation quality using intermediary model outputs. Both proposed models outperform the state-of-the-art model (F1-scores: CNN=0.46, ANN=0.44) when tested on 10 holdout test patients. The interactive machine learning strategy demonstrates that our models are robust to noisy annotations. Overall, results highlight the robustness of models with simple architectures when analyzing complex and imperfectly annotated data. Our method of interactive machine learning offers great potential for faster data annotation, while our models represent useful and efficient tools for automated interictal spikes detection.
>
---
#### [new 019] Group Inertial Poser: Multi-Person Pose and Global Translation from Sparse Inertial Sensors and Ultra-Wideband Ranging
- **分类: cs.CV; cs.AI; cs.GR; cs.HC; 68T07, 68T45, 68U01; I.2; I.3; I.4; I.5**

- **简介: 该论文属于多人体态追踪任务，旨在解决稀疏IMU仅能提供相对运动、难以估计全局位置的问题。通过融合超宽带（UWB）测距信息与惯性数据，提出Group Inertial Poser方法，实现多人精确3D姿态与全球轨迹估计，并构建首个双人IMU+UWB数据集，显著提升精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21654v1](http://arxiv.org/pdf/2510.21654v1)**

> **作者:** Ying Xue; Jiaxi Jiang; Rayan Armani; Dominik Hollidt; Yi-Chi Liao; Christian Holz
>
> **备注:** Accepted by ICCV 2025, Code: https://github.com/eth-siplab/GroupInertialPoser
>
> **摘要:** Tracking human full-body motion using sparse wearable inertial measurement units (IMUs) overcomes the limitations of occlusion and instrumentation of the environment inherent in vision-based approaches. However, purely IMU-based tracking compromises translation estimates and accurate relative positioning between individuals, as inertial cues are inherently self-referential and provide no direct spatial reference for others. In this paper, we present a novel approach for robustly estimating body poses and global translation for multiple individuals by leveraging the distances between sparse wearable sensors - both on each individual and across multiple individuals. Our method Group Inertial Poser estimates these absolute distances between pairs of sensors from ultra-wideband ranging (UWB) and fuses them with inertial observations as input into structured state-space models to integrate temporal motion patterns for precise 3D pose estimation. Our novel two-step optimization further leverages the estimated distances for accurately tracking people's global trajectories through the world. We also introduce GIP-DB, the first IMU+UWB dataset for two-person tracking, which comprises 200 minutes of motion recordings from 14 participants. In our evaluation, Group Inertial Poser outperforms previous state-of-the-art methods in accuracy and robustness across synthetic and real-world data, showing the promise of IMU+UWB-based multi-human motion capture in the wild. Code, models, dataset: https://github.com/eth-siplab/GroupInertialPoser
>
---
#### [new 020] Towards Physics-informed Spatial Intelligence with Human Priors: An Autonomous Driving Pilot Study
- **分类: cs.CV**

- **简介: 该论文针对基础模型中空间智能难以有效集成与评估的问题，提出基于网格的时空结构表示SIG，显式编码场景布局与物理先验。通过构建SIGBench基准，实现对自动驾驶场景下机器与人类空间感知能力的量化评估，显著提升模型在少样本学习中的空间推理性能。**

- **链接: [http://arxiv.org/pdf/2510.21160v1](http://arxiv.org/pdf/2510.21160v1)**

> **作者:** Guanlin Wu; Boyan Su; Yang Zhao; Pu Wang; Yichen Lin; Hao Frank Yang
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** How to integrate and verify spatial intelligence in foundation models remains an open challenge. Current practice often proxies Visual-Spatial Intelligence (VSI) with purely textual prompts and VQA-style scoring, which obscures geometry, invites linguistic shortcuts, and weakens attribution to genuinely spatial skills. We introduce Spatial Intelligence Grid (SIG): a structured, grid-based schema that explicitly encodes object layouts, inter-object relations, and physically grounded priors. As a complementary channel to text, SIG provides a faithful, compositional representation of scene structure for foundation-model reasoning. Building on SIG, we derive SIG-informed evaluation metrics that quantify a model's intrinsic VSI, which separates spatial capability from language priors. In few-shot in-context learning with state-of-the-art multimodal LLMs (e.g. GPT- and Gemini-family models), SIG yields consistently larger, more stable, and more comprehensive gains across all VSI metrics compared to VQA-only representations, indicating its promise as a data-labeling and training schema for learning VSI. We also release SIGBench, a benchmark of 1.4K driving frames annotated with ground-truth SIG labels and human gaze traces, supporting both grid-based machine VSI tasks and attention-driven, human-like VSI tasks in autonomous-driving scenarios.
>
---
#### [new 021] Epipolar Geometry Improves Video Generation Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决现有模型产生的视频存在几何不一致、运动不稳定和视觉伪影等问题。作者引入双视图对极几何约束，通过偏好优化实现非端到端的几何约束强化，提升空间一致性，同时保持视觉质量，实现更真实的3D场景生成。**

- **链接: [http://arxiv.org/pdf/2510.21615v1](http://arxiv.org/pdf/2510.21615v1)**

> **作者:** Orest Kupyn; Fabian Manhardt; Federico Tombari; Christian Rupprecht
>
> **摘要:** Video generation models have progressed tremendously through large latent diffusion transformers trained with rectified flow techniques. Yet these models still struggle with geometric inconsistencies, unstable motion, and visual artifacts that break the illusion of realistic 3D scenes. 3D-consistent video generation could significantly impact numerous downstream applications in generation and reconstruction tasks. We explore how epipolar geometry constraints improve modern video diffusion models. Despite massive training data, these models fail to capture fundamental geometric principles underlying visual content. We align diffusion models using pairwise epipolar geometry constraints via preference-based optimization, directly addressing unstable camera trajectories and geometric artifacts through mathematically principled geometric enforcement. Our approach efficiently enforces geometric principles without requiring end-to-end differentiability. Evaluation demonstrates that classical geometric constraints provide more stable optimization signals than modern learned metrics, which produce noisy targets that compromise alignment quality. Training on static scenes with dynamic cameras ensures high-quality measurements while the model generalizes effectively to diverse dynamic content. By bridging data-driven deep learning with classical geometric computer vision, we present a practical method for generating spatially consistent videos without compromising visual quality.
>
---
#### [new 022] Depth-Supervised Fusion Network for Seamless-Free Image Stitching
- **分类: cs.CV**

- **简介: 该论文针对多视角图像拼接中的视差导致的鬼影和错位问题，提出一种深度一致性约束的无缝拼接方法。通过多阶段对齐与全局深度正则化提升对齐精度，结合图论低代价计算确定最优拼接缝，并引入重参数化策略提升效率，实现高质量无缝拼接。**

- **链接: [http://arxiv.org/pdf/2510.21396v1](http://arxiv.org/pdf/2510.21396v1)**

> **作者:** Zhiying Jiang; Ruhao Yan; Zengxi Zhang; Bowei Zhang; Jinyuan Liu
>
> **备注:** Accepted to Neurips 2025
>
> **摘要:** Image stitching synthesizes images captured from multiple perspectives into a single image with a broader field of view. The significant variations in object depth often lead to large parallax, resulting in ghosting and misalignment in the stitched results. To address this, we propose a depth-consistency-constrained seamless-free image stitching method. First, to tackle the multi-view alignment difficulties caused by parallax, a multi-stage mechanism combined with global depth regularization constraints is developed to enhance the alignment accuracy of the same apparent target across different depth ranges. Second, during the multi-view image fusion process, an optimal stitching seam is determined through graph-based low-cost computation, and a soft-seam region is diffused to precisely locate transition areas, thereby effectively mitigating alignment errors induced by parallax and achieving natural and seamless stitching results. Furthermore, considering the computational overhead in the shift regression process, a reparameterization strategy is incorporated to optimize the structural design, significantly improving algorithm efficiency while maintaining optimal performance. Extensive experiments demonstrate the superior performance of the proposed method against the existing methods. Code is available at https://github.com/DLUT-YRH/DSFN.
>
---
#### [new 023] CT-CLIP: A Multi-modal Fusion Framework for Robust Apple Leaf Disease Recognition in Complex Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对复杂果园环境下苹果叶病多形态、背景干扰等问题，提出CT-CLIP框架，融合CNN局部特征与Vision Transformer全局结构信息，通过自适应融合模块和图文对比学习，提升少样本条件下的疾病识别准确率。**

- **链接: [http://arxiv.org/pdf/2510.21346v1](http://arxiv.org/pdf/2510.21346v1)**

> **作者:** Lemin Liu; Fangchao Hu; Honghua Jiang; Yaru Chen; Limin Liu; Yongliang Qiao
>
> **摘要:** In complex orchard environments, the phenotypic heterogeneity of different apple leaf diseases, characterized by significant variation among lesions, poses a challenge to traditional multi-scale feature fusion methods. These methods only integrate multi-layer features extracted by convolutional neural networks (CNNs) and fail to adequately account for the relationships between local and global features. Therefore, this study proposes a multi-branch recognition framework named CNN-Transformer-CLIP (CT-CLIP). The framework synergistically employs a CNN to extract local lesion detail features and a Vision Transformer to capture global structural relationships. An Adaptive Feature Fusion Module (AFFM) then dynamically fuses these features, achieving optimal coupling of local and global information and effectively addressing the diversity in lesion morphology and distribution. Additionally, to mitigate interference from complex backgrounds and significantly enhance recognition accuracy under few-shot conditions, this study proposes a multimodal image-text learning approach. By leveraging pre-trained CLIP weights, it achieves deep alignment between visual features and disease semantic descriptions. Experimental results show that CT-CLIP achieves accuracies of 97.38% and 96.12% on a publicly available apple disease and a self-built dataset, outperforming several baseline methods. The proposed CT-CLIP demonstrates strong capabilities in recognizing agricultural diseases, significantly enhances identification accuracy under complex environmental conditions, provides an innovative and practical solution for automated disease recognition in agricultural applications.
>
---
#### [new 024] Gaze-VLM:Bridging Gaze and VLMs through Attention Regularization for Egocentric Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自指视角下行为理解任务，提出Gaze-VLM框架，通过注意力正则化融合人类眼动数据，提升视觉语言模型在细粒度未来事件预测与当前活动理解中的性能。仅训练时使用眼动信息，增强模型对关键区域的关注能力，显著提升预测准确率。**

- **链接: [http://arxiv.org/pdf/2510.21356v1](http://arxiv.org/pdf/2510.21356v1)**

> **作者:** Anupam Pani; Yanchao Yang
>
> **摘要:** Eye gaze offers valuable cues about attention, short-term intent, and future actions, making it a powerful signal for modeling egocentric behavior. In this work, we propose a gaze-regularized framework that enhances VLMs for two key egocentric understanding tasks: fine-grained future event prediction and current activity understanding. Unlike prior approaches that rely solely on visual inputs or use gaze as an auxiliary input signal , our method uses gaze only during training. We introduce a gaze-regularized attention mechanism that aligns model focus with human visual gaze. This design is flexible and modular, allowing it to generalize across multiple VLM architectures that utilize attention. Experimental results show that our approach improves semantic prediction scores by up to 11 for future event prediction and around 7 for current activity understanding, compared to the corresponding baseline models trained without gaze regularization. These results highlight the value of gaze-guided training in improving the accuracy and robustness of egocentric VLMs. Overall, this work establishes a foundation for using human gaze to enhance the predictive capabilities of VLMs in real-world scenarios like assistive robots and human-machine collaboration. Code and additional information is available at: https://github.com/anupampani/Gaze-VLM
>
---
#### [new 025] Controllable-LPMoE: Adapting to Challenging Object Segmentation via Dynamic Local Priors from Mixture-of-Experts
- **分类: cs.CV**

- **简介: 该论文针对对象分割任务，提出Controllable-LPMoE方法，通过动态生成局部先验实现高效微调。解决全参数微调计算开销大、可训练提示缺乏语义先验的问题。设计轻量级混合局部先验提取器与双向交互适配器，以少量可训练参数增强模型对细粒度特征的感知能力。**

- **链接: [http://arxiv.org/pdf/2510.21114v1](http://arxiv.org/pdf/2510.21114v1)**

> **作者:** Yanguang Sun; Jiawei Lian; Jian Yang; Lei Luo
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Large-scale foundation models provide powerful feature representations for downstream object segmentation tasks. However, when adapted to specific tasks through the full-parameter fine-tuning, the enormous parameters being updated often results in significant computational overhead, creating a bottleneck in training efficiency. Although existing methods attempt to fine-tune frozen models by directly embedding trainable prompts, these prompts lack inherent semantic priors, limiting the adaptability of large-scale models. In this paper, we propose a novel dynamic priors-based fine-tuning paradigm with fewer trainable parameters, dubbed Controllable-LPMoE, which adaptively modulates frozen foundation models by dynamically controlling local priors to enhance fine-grained perception for specific segmentation tasks. More specifically, we construct a lightweight dynamic mixed local priors extractor that captures diverse local priors from input images through heterogeneous convolutions while employing a gating network to dynamically output expert priors required for the subsequent fine-tuning. Furthermore, we design a bi-directional interaction adapter that employs cosine-aligned deformable attention and channel-oriented adaptive scale enhancement to interact and restructure between frozen and trainable features, achieving efficient fine-tuning. Extensive experiments validate the superiority of our \href{https://github.com/CSYSI/Controllable-LPMoE} {Controllable-LPMoE} approach, demonstrating excellent segmentation performance compared to 31 state-of-the-art (SOTA) methods and adaptability to multiple binary object segmentation tasks.
>
---
#### [new 026] TerraGen: A Unified Multi-Task Layout Generation Framework for Remote Sensing Data Augmentation
- **分类: cs.CV**

- **简介: 该论文提出TerraGen，一个统一的多任务遥感图像生成框架，解决现有数据增强方法任务孤立、忽略地理空间约束的问题。通过统一编码边界框与分割掩码，结合多尺度注入与加权损失，实现可控的遥感图像合成。构建了首个大规模多任务遥感布局数据集，实验表明其生成质量高，可显著提升下游任务性能。**

- **链接: [http://arxiv.org/pdf/2510.21391v1](http://arxiv.org/pdf/2510.21391v1)**

> **作者:** Datao Tang; Hao Wang; Yudeng Xin; Hui Qiao; Dongsheng Jiang; Yin Li; Zhiheng Yu; Xiangyong Cao
>
> **摘要:** Remote sensing vision tasks require extensive labeled data across multiple, interconnected domains. However, current generative data augmentation frameworks are task-isolated, i.e., each vision task requires training an independent generative model, and ignores the modeling of geographical information and spatial constraints. To address these issues, we propose \textbf{TerraGen}, a unified layout-to-image generation framework that enables flexible, spatially controllable synthesis of remote sensing imagery for various high-level vision tasks, e.g., detection, segmentation, and extraction. Specifically, TerraGen introduces a geographic-spatial layout encoder that unifies bounding box and segmentation mask inputs, combined with a multi-scale injection scheme and mask-weighted loss to explicitly encode spatial constraints, from global structures to fine details. Also, we construct the first large-scale multi-task remote sensing layout generation dataset containing 45k images and establish a standardized evaluation protocol for this task. Experimental results show that our TerraGen can achieve the best generation image quality across diverse tasks. Additionally, TerraGen can be used as a universal data-augmentation generator, enhancing downstream task performance significantly and demonstrating robust cross-task generalisation in both full-data and few-shot scenarios.
>
---
#### [new 027] Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent
- **分类: cs.CV**

- **简介: 该论文针对视觉模型依赖特定视觉属性的问题，提出一种基于自反思智能体的自动化检测框架。通过迭代生成与验证假设，实现对模型决策依据的自我评估与优化，有效识别模型在真实场景中的潜在依赖关系，提升模型可解释性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21704v1](http://arxiv.org/pdf/2510.21704v1)**

> **作者:** Christy Li; Josep Lopez Camuñas; Jake Thomas Touchet; Jacob Andreas; Agata Lapedriza; Antonio Torralba; Tamar Rott Shaham
>
> **备注:** 32 pages, 10 figures, Neurips 2025
>
> **摘要:** When a vision model performs image recognition, which visual attributes drive its predictions? Detecting unintended reliance on specific visual features is critical for ensuring model robustness, preventing overfitting, and avoiding spurious correlations. We introduce an automated framework for detecting such dependencies in trained vision models. At the core of our method is a self-reflective agent that systematically generates and tests hypotheses about visual attributes that a model may rely on. This process is iterative: the agent refines its hypotheses based on experimental outcomes and uses a self-evaluation protocol to assess whether its findings accurately explain model behavior. When inconsistencies arise, the agent self-reflects over its findings and triggers a new cycle of experimentation. We evaluate our approach on a novel benchmark of 130 models designed to exhibit diverse visual attribute dependencies across 18 categories. Our results show that the agent's performance consistently improves with self-reflection, with a significant performance increase over non-reflective baselines. We further demonstrate that the agent identifies real-world visual attribute dependencies in state-of-the-art models, including CLIP's vision encoder and the YOLOv8 object detector.
>
---
#### [new 028] Knowledge-Driven Vision-Language Model for Plexus Detection in Hirschsprung's Disease
- **分类: cs.CV**

- **简介: 该论文针对先天性巨结肠症中肌间神经丛的病理图像分类任务，提出一种融合专家知识的视觉语言模型。通过引入医学文本概念引导模型学习，提升分类准确性与临床可解释性，显著优于传统CNN模型。**

- **链接: [http://arxiv.org/pdf/2510.21083v1](http://arxiv.org/pdf/2510.21083v1)**

> **作者:** Youssef Megahed; Atallah Madi; Dina El Demellawy; Adrian D. C. Chan
>
> **备注:** Accepted into the ICAAI 2025 - The 9th International Conference on Advances in Artificial Intelligence
>
> **摘要:** Hirschsprung's disease is defined as the congenital absence of ganglion cells in some segment(s) of the colon. The muscle cannot make coordinated movements to propel stool in that section, most commonly leading to obstruction. The diagnosis and treatment for this disease require a clear identification of different region(s) of the myenteric plexus, where ganglion cells should be present, on the microscopic view of the tissue slide. While deep learning approaches, such as Convolutional Neural Networks, have performed very well in this task, they are often treated as black boxes, with minimal understanding gained from them, and may not conform to how a physician makes decisions. In this study, we propose a novel framework that integrates expert-derived textual concepts into a Contrastive Language-Image Pre-training-based vision-language model to guide plexus classification. Using prompts derived from expert sources (e.g., medical textbooks and papers) generated by large language models and reviewed by our team before being encoded with QuiltNet, our approach aligns clinically relevant semantic cues with visual features. Experimental results show that the proposed model demonstrated superior discriminative capability across different classification metrics as it outperformed CNN-based models, including VGG-19, ResNet-18, and ResNet-50; achieving an accuracy of 83.9%, a precision of 86.6%, and a specificity of 87.6%. These findings highlight the potential of multi-modal learning in histopathology and underscore the value of incorporating expert knowledge for more clinically relevant model outputs.
>
---
#### [new 029] Deep learning-based automated damage detection in concrete structures using images from earthquake events
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉中的图像识别任务，旨在自动化检测地震后混凝土结构的损伤。针对灾后结构完整性评估难题，研究构建了新数据集，基于YOLOv11模型实现钢筋暴露、裂缝与剥落的检测，并融合多模型形成混合框架，实现损伤等级的自动判定。**

- **链接: [http://arxiv.org/pdf/2510.21063v1](http://arxiv.org/pdf/2510.21063v1)**

> **作者:** Abdullah Turer; Yongsheng Bai; Halil Sezen; Alper Yilmaz
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Timely assessment of integrity of structures after seismic events is crucial for public safety and emergency response. This study focuses on assessing the structural damage conditions using deep learning methods to detect exposed steel reinforcement in concrete buildings and bridges after large earthquakes. Steel bars are typically exposed after concrete spalling or large flexural or shear cracks. The amount and distribution of exposed steel reinforcement is an indication of structural damage and degradation. To automatically detect exposed steel bars, new datasets of images collected after the 2023 Turkey Earthquakes were labeled to represent a wide variety of damaged concrete structures. The proposed method builds upon a deep learning framework, enhanced with fine-tuning, data augmentation, and testing on public datasets. An automated classification framework is developed that can be used to identify inside/outside buildings and structural components. Then, a YOLOv11 (You Only Look Once) model is trained to detect cracking and spalling damage and exposed bars. Another YOLO model is finetuned to distinguish different categories of structural damage levels. All these trained models are used to create a hybrid framework to automatically and reliably determine the damage levels from input images. This research demonstrates that rapid and automated damage detection following disasters is achievable across diverse damage contexts by utilizing image data collection, annotation, and deep learning approaches.
>
---
#### [new 030] SafetyPairs: Isolating Safety Critical Image Features with Counterfactual Image Generation
- **分类: cs.CV**

- **简介: 该论文提出SafetyPairs框架，用于生成仅在安全相关特征上不同的图像对，以精确区分图像安全性的细微差异。针对现有数据集标签粗粒度的问题，通过可控图像编辑生成反事实样本，构建细粒度安全基准，提升模型对安全特征的识别能力，并可用于训练更高效的轻量级安全检测模型。**

- **链接: [http://arxiv.org/pdf/2510.21120v1](http://arxiv.org/pdf/2510.21120v1)**

> **作者:** Alec Helbling; Shruti Palaskar; Kundan Krishna; Polo Chau; Leon Gatys; Joseph Yitan Cheng
>
> **摘要:** What exactly makes a particular image unsafe? Systematically differentiating between benign and problematic images is a challenging problem, as subtle changes to an image, such as an insulting gesture or symbol, can drastically alter its safety implications. However, existing image safety datasets are coarse and ambiguous, offering only broad safety labels without isolating the specific features that drive these differences. We introduce SafetyPairs, a scalable framework for generating counterfactual pairs of images, that differ only in the features relevant to the given safety policy, thus flipping their safety label. By leveraging image editing models, we make targeted changes to images that alter their safety labels while leaving safety-irrelevant details unchanged. Using SafetyPairs, we construct a new safety benchmark, which serves as a powerful source of evaluation data that highlights weaknesses in vision-language models' abilities to distinguish between subtly different images. Beyond evaluation, we find our pipeline serves as an effective data augmentation strategy that improves the sample efficiency of training lightweight guard models. We release a benchmark containing over 3,020 SafetyPair images spanning a diverse taxonomy of 9 safety categories, providing the first systematic resource for studying fine-grained image safety distinctions.
>
---
#### [new 031] ZING-3D: Zero-shot Incremental 3D Scene Graphs via Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ZING-3D框架，解决3D场景图生成中缺乏零样本识别、增量更新与3D几何接地的问题。通过视觉语言模型生成2D语义场景图，并结合深度信息实现3D空间对齐，支持开放词汇、增量更新与空间关系建模，适用于机器人等具身应用。**

- **链接: [http://arxiv.org/pdf/2510.21069v1](http://arxiv.org/pdf/2510.21069v1)**

> **作者:** Pranav Saxena; Jimmy Chiun
>
> **摘要:** Understanding and reasoning about complex 3D environments requires structured scene representations that capture not only objects but also their semantic and spatial relationships. While recent works on 3D scene graph generation have leveraged pretrained VLMs without task-specific fine-tuning, they are largely confined to single-view settings, fail to support incremental updates as new observations arrive and lack explicit geometric grounding in 3D space, all of which are essential for embodied scenarios. In this paper, we propose, ZING-3D, a framework that leverages the vast knowledge of pretrained foundation models to enable open-vocabulary recognition and generate a rich semantic representation of the scene in a zero-shot manner while also enabling incremental updates and geometric grounding in 3D space, making it suitable for downstream robotics applications. Our approach leverages VLM reasoning to generate a rich 2D scene graph, which is grounded in 3D using depth information. Nodes represent open-vocabulary objects with features, 3D locations, and semantic context, while edges capture spatial and semantic relations with inter-object distances. Our experiments on scenes from the Replica and HM3D dataset show that ZING-3D is effective at capturing spatial and relational knowledge without the need of task-specific training.
>
---
#### [new 032] Preventing Shortcuts in Adapter Training via Providing the Shortcuts
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像生成中适配器训练时属性纠缠问题，提出通过引入辅助模块（如ControlNet）显式处理干扰因素（如姿态、光照），使适配器专注学习目标属性。方法在训练时“提供”短路路径以消除冗余关联，推理时移除辅助模块，提升生成质量与提示遵循度，实现更优的属性解耦。**

- **链接: [http://arxiv.org/pdf/2510.20887v1](http://arxiv.org/pdf/2510.20887v1)**

> **作者:** Anujraaj Argo Goyal; Guocheng Gordon Qian; Huseyin Coskun; Aarush Gupta; Himmy Tam; Daniil Ostashev; Ju Hu; Dhritiman Sagar; Sergey Tulyakov; Kfir Aberman; Kuan-Chieh Jackson Wang
>
> **备注:** Accepted to NeurIPS 2025, webpage: https://snap-research.github.io/shortcut-rerouting/
>
> **摘要:** Adapter-based training has emerged as a key mechanism for extending the capabilities of powerful foundation image generators, enabling personalized and stylized text-to-image synthesis. These adapters are typically trained to capture a specific target attribute, such as subject identity, using single-image reconstruction objectives. However, because the input image inevitably contains a mixture of visual factors, adapters are prone to entangle the target attribute with incidental ones, such as pose, expression, and lighting. This spurious correlation problem limits generalization and obstructs the model's ability to adhere to the input text prompt. In this work, we uncover a simple yet effective solution: provide the very shortcuts we wish to eliminate during adapter training. In Shortcut-Rerouted Adapter Training, confounding factors are routed through auxiliary modules, such as ControlNet or LoRA, eliminating the incentive for the adapter to internalize them. The auxiliary modules are then removed during inference. When applied to tasks like facial and full-body identity injection, our approach improves generation quality, diversity, and prompt adherence. These results point to a general design principle in the era of large models: when seeking disentangled representations, the most effective path may be to establish shortcuts for what should NOT be learned.
>
---
#### [new 033] MATrack: Efficient Multiscale Adaptive Tracker for Real-Time Nighttime UAV Operations
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对夜间无人机跟踪任务，解决低光下视觉退化、背景杂乱和视角变化导致的跟踪漂移问题。提出MATrack系统，通过多尺度融合、自适应关键令牌门和夜间模板校准三模块协同，提升特征一致性与跟踪稳定性，在UAVDark135上显著优于SOTA方法，实现81 FPS实时性能。**

- **链接: [http://arxiv.org/pdf/2510.21586v1](http://arxiv.org/pdf/2510.21586v1)**

> **作者:** Xuzhao Li; Xuchen Li; Shiyu Hu
>
> **备注:** Preprint, Under Review
>
> **摘要:** Nighttime UAV tracking faces significant challenges in real-world robotics operations. Low-light conditions not only limit visual perception capabilities, but cluttered backgrounds and frequent viewpoint changes also cause existing trackers to drift or fail during deployment. To address these difficulties, researchers have proposed solutions based on low-light enhancement and domain adaptation. However, these methods still have notable shortcomings in actual UAV systems: low-light enhancement often introduces visual artifacts, domain adaptation methods are computationally expensive and existing lightweight designs struggle to fully leverage dynamic object information. Based on an in-depth analysis of these key issues, we propose MATrack-a multiscale adaptive system designed specifically for nighttime UAV tracking. MATrack tackles the main technical challenges of nighttime tracking through the collaborative work of three core modules: Multiscale Hierarchy Blende (MHB) enhances feature consistency between static and dynamic templates. Adaptive Key Token Gate accurately identifies object information within complex backgrounds. Nighttime Template Calibrator (NTC) ensures stable tracking performance over long sequences. Extensive experiments show that MATrack achieves a significant performance improvement. On the UAVDark135 benchmark, its precision, normalized precision and AUC surpass state-of-the-art (SOTA) methods by 5.9%, 5.4% and 4.2% respectively, while maintaining a real-time processing speed of 81 FPS. Further tests on a real-world UAV platform validate the system's reliability, demonstrating that MATrack can provide stable and effective nighttime UAV tracking support for critical robotics applications such as nighttime search and rescue and border patrol.
>
---
#### [new 034] GranViT: A Fine-Grained Vision Model With Autoregressive Perception For MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大模型中视觉编码器缺乏细粒度区域分析的问题，提出GranViT模型。通过构建大规模细粒度标注数据集Gran-29M，结合区域级自回归预训练与自蒸馏机制，提升视觉编码器的局部感知与语义对齐能力，显著增强模型在细粒度识别、视觉问答和OCR理解等任务上的性能。**

- **链接: [http://arxiv.org/pdf/2510.21501v1](http://arxiv.org/pdf/2510.21501v1)**

> **作者:** Guanghao Zheng; Bowen Shi; Mingxing Xu; Ruoyu Sun; Peisen Zhao; Zhibo Zhang; Wenrui Dai; Junni Zou; Hongkai Xiong; Xiaopeng Zhang; Qi Tian
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** Vision encoders are indispensable for allowing impressive performance of Multi-modal Large Language Models (MLLMs) in vision language tasks such as visual question answering and reasoning. However, existing vision encoders focus on global image representations but overlook fine-grained regional analysis. They are limited in fine grained perception due to the scarcity of fine grained annotated data and the lack of a fine grained pre-training paradigm. In this paper, we propose GranViT, a novel Vision Transformer that integrates fine-grained feature extraction with semantic alignment to Large Language Models (LLMs) via region level autoregressive training. We first construct Gran-29M, a dataset comprising 2million natural and OCR images paired with over 180 million high-quality region-level annotations, to enable large scale fine grained pretraining. Consequently, we develop a pretraining-adaptation framework along with a self distillation mechanism to train fine-grained GranViT on Gran-29M. We sufficiently exploit the fine-grained annotations from Gran-29M to resort to bounding-box-to-caption regression to enhance localized visual representation of the vision encoder in the pretraining and caption-to-bounding-box regression to improve vision feature utilization and localization for LLM in the adaptation. We further incorporate a self distillation mechanism that imposes explicit localization constraints on the vision encoder to strengthen its regional reasoning capability. Extensive experiments show that GranViT surpasses existing vision encoders and attains strong transferability to varying LLMs. Remarkably, it achieves state-of-the-art results on fine-grained recognition, multimodal VQA, and OCR understanding.
>
---
#### [new 035] 3rd Place Solution to Large-scale Fine-grained Food Recognition
- **分类: cs.CV**

- **简介: 该论文针对大规模细粒度食物识别任务，提出结合Arcface与Circle损失的模型优化方法。通过精心调参与模型集成，提升识别精度，在Kaggle竞赛中获第三名。**

- **链接: [http://arxiv.org/pdf/2510.21199v1](http://arxiv.org/pdf/2510.21199v1)**

> **作者:** Yang Zhong; Yifan Yao; Tong Luo; Youcai Zhang; Yaqian Li
>
> **摘要:** Food analysis is becoming a hot topic in health area, in which fine-grained food recognition task plays an important role. In this paper, we describe the details of our solution to the LargeFineFoodAI-ICCV Workshop-Recognition challenge held on Kaggle. We find a proper combination of Arcface loss[1] and Circle loss[9] can bring improvement to the performance. With Arcface and the combined loss, model was trained with carefully tuned configurations and ensembled to get the final results. Our solution won the 3rd place in the competition.
>
---
#### [new 036] A Dynamic Knowledge Distillation Method Based on the Gompertz Curve
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对知识蒸馏中学生模型认知能力动态变化未被充分建模的问题，提出基于Gompertz曲线的动态知识蒸馏框架Gompertz-CNN。通过引入Gompertz曲线调节蒸馏损失权重，模拟学习过程的阶段性特征，并结合Wasserstein距离与梯度匹配，实现更高效的特征对齐。在CIFAR数据集上显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.21649v1](http://arxiv.org/pdf/2510.21649v1)**

> **作者:** Han Yang; Guangjun Qin
>
> **备注:** 15 pages, 2 figures
>
> **摘要:** This paper introduces a novel dynamic knowledge distillation framework, Gompertz-CNN, which integrates the Gompertz growth model into the training process to address the limitations of traditional knowledge distillation. Conventional methods often fail to capture the evolving cognitive capacity of student models, leading to suboptimal knowledge transfer. To overcome this, we propose a stage-aware distillation strategy that dynamically adjusts the weight of distillation loss based on the Gompertz curve, reflecting the student's learning progression: slow initial growth, rapid mid-phase improvement, and late-stage saturation. Our framework incorporates Wasserstein distance to measure feature-level discrepancies and gradient matching to align backward propagation behaviors between teacher and student models. These components are unified under a multi-loss objective, where the Gompertz curve modulates the influence of distillation losses over time. Extensive experiments on CIFAR-10 and CIFAR-100 using various teacher-student architectures (e.g., ResNet50 and MobileNet_v2) demonstrate that Gompertz-CNN consistently outperforms traditional distillation methods, achieving up to 8% and 4% accuracy gains on CIFAR-10 and CIFAR-100, respectively.
>
---
#### [new 037] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations
- **分类: cs.CV**

- **简介: 该论文研究文本到图像生成中的条件引导问题，针对现有分类器自由引导（CFG）方法理论不统一、效率低的问题，提出基于前瞻固定点迭代的Foresight Guidance（FSG），通过早期多步长迭代优化生成质量与效率，实验证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.21512v1](http://arxiv.org/pdf/2510.21512v1)**

> **作者:** Kaibo Wang; Jianda Mao; Tong Wu; Yang Xiang
>
> **备注:** Accepted at NeurIPS 2025 (Spotlight)
>
> **摘要:** Classifier-Free Guidance (CFG) is an essential component of text-to-image diffusion models, and understanding and advancing its operational mechanisms remains a central focus of research. Existing approaches stem from divergent theoretical interpretations, thereby limiting the design space and obscuring key design choices. To address this, we propose a unified perspective that reframes conditional guidance as fixed point iterations, seeking to identify a golden path where latents produce consistent outputs under both conditional and unconditional generation. We demonstrate that CFG and its variants constitute a special case of single-step short-interval iteration, which is theoretically proven to exhibit inefficiency. To this end, we introduce Foresight Guidance (FSG), which prioritizes solving longer-interval subproblems in early diffusion stages with increased iterations. Extensive experiments across diverse datasets and model architectures validate the superiority of FSG over state-of-the-art methods in both image quality and computational efficiency. Our work offers novel perspectives for conditional guidance and unlocks the potential of adaptive design.
>
---
#### [new 038] GRAP-MOT: Unsupervised Graph-based Position Weighted Person Multi-camera Multi-object Tracking in a Highly Congested Space
- **分类: cs.CV**

- **简介: 该论文针对高密度封闭场景下的多人多摄像头跟踪（MOT）任务，提出GRAP-MOT方法。通过图加权与位置估计，实现无监督在线身份更新，有效解决遮挡问题。实验表明其优于现有方法，并建议使用IDF1评估。**

- **链接: [http://arxiv.org/pdf/2510.21482v1](http://arxiv.org/pdf/2510.21482v1)**

> **作者:** Marek Socha; Michał Marczyk; Aleksander Kempski; Michał Cogiel; Paweł Foszner; Radosław Zawiski; Michał Staniszewski
>
> **备注:** 13 pages, 5 figures, 8 tables
>
> **摘要:** GRAP-MOT is a new approach for solving the person MOT problem dedicated to videos of closed areas with overlapping multi-camera views, where person occlusion frequently occurs. Our novel graph-weighted solution updates a person's identification label online based on tracks and the person's characteristic features. To find the best solution, we deeply investigated all elements of the MOT process, including feature extraction, tracking, and community search. Furthermore, GRAP-MOT is equipped with a person's position estimation module, which gives additional key information to the MOT method, ensuring better results than methods without position data. We tested GRAP-MOT on recordings acquired in a closed-area model and on publicly available real datasets that fulfil the requirement of a highly congested space, showing the superiority of our proposition. Finally, we analyzed existing metrics used to compare MOT algorithms and concluded that IDF1 is more adequate than MOTA in such comparisons. We made our code, along with the acquired dataset, publicly available.
>
---
#### [new 039] VidSplice: Towards Coherent Video Inpainting via Explicit Spaced Frame Guidance
- **分类: cs.CV**

- **简介: 该论文聚焦视频修复任务，针对现有方法在严重内容缺失下时空不一致的问题，提出VidSplice框架。通过引入间隔帧先验，分离图像修复与运动传播，设计协同拼接模块与上下文控制器，增强前后景对齐与运动稳定性，显著提升生成质量与可控性。**

- **链接: [http://arxiv.org/pdf/2510.21461v1](http://arxiv.org/pdf/2510.21461v1)**

> **作者:** Ming Xie; Junqiu Yu; Qiaole Dong; Xiangyang Xue; Yanwei Fu
>
> **备注:** 19 pages
>
> **摘要:** Recent video inpainting methods often employ image-to-video (I2V) priors to model temporal consistency across masked frames. While effective in moderate cases, these methods struggle under severe content degradation and tend to overlook spatiotemporal stability, resulting in insufficient control over the latter parts of the video. To address these limitations, we decouple video inpainting into two sub-tasks: multi-frame consistent image inpainting and masked area motion propagation. We propose VidSplice, a novel framework that introduces spaced-frame priors to guide the inpainting process with spatiotemporal cues. To enhance spatial coherence, we design a CoSpliced Module to perform first-frame propagation strategy that diffuses the initial frame content into subsequent reference frames through a splicing mechanism. Additionally, we introduce a delicate context controller module that encodes coherent priors after frame duplication and injects the spliced video into the I2V generative backbone, effectively constraining content distortion during generation. Extensive evaluations demonstrate that VidSplice achieves competitive performance across diverse video inpainting scenarios. Moreover, its design significantly improves both foreground alignment and motion stability, outperforming existing approaches.
>
---
#### [new 040] Modest-Align: Data-Efficient Alignment for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文针对视觉-语言模型在低资源下的跨模态对齐问题，提出轻量级的Modest-Align框架。通过随机扰动与嵌入平滑缓解过自信问题，提升噪声数据下的性能。实验表明，其在极少数据和计算资源下超越现有方法，适用于真实低资源场景。**

- **链接: [http://arxiv.org/pdf/2510.21606v1](http://arxiv.org/pdf/2510.21606v1)**

> **作者:** Jiaxiang Liu; Yuan Wang; Jiawei Du; Joey Tianyi Zhou; Mingkun Xu; Zuozhu Liu
>
> **摘要:** Cross-modal alignment aims to map heterogeneous modalities into a shared latent space, as exemplified by models like CLIP, which benefit from large-scale image-text pretraining for strong recognition capabilities. However, when operating in resource-constrained settings with limited or low-quality data, these models often suffer from overconfidence and degraded performance due to the prevalence of ambiguous or weakly correlated image-text pairs. Current contrastive learning approaches, which rely on single positive pairs, further exacerbate this issue by reinforcing overconfidence on uncertain samples. To address these challenges, we propose Modest-Align, a lightweight alignment framework designed for robustness and efficiency. Our approach leverages two complementary strategies -- Random Perturbation, which introduces controlled noise to simulate uncertainty, and Embedding Smoothing, which calibrates similarity distributions in the embedding space. These mechanisms collectively reduce overconfidence and improve performance on noisy or weakly aligned samples. Extensive experiments across multiple benchmark datasets demonstrate that Modest-Align outperforms state-of-the-art methods in retrieval tasks, achieving competitive results with over 100x less training data and 600x less GPU time than CLIP. Our method offers a practical and scalable solution for cross-modal alignment in real-world, low-resource scenarios.
>
---
#### [new 041] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning
- **分类: cs.CV**

- **简介: 该论文针对点云数据跨域学习中因数据稀缺导致的性能下降问题，提出DAP-MAE方法。通过设计异构域适配器与域特征生成器，在预训练中自适应融合多域知识，微调时增强特征表达，显著提升分类与表情识别等任务性能。**

- **链接: [http://arxiv.org/pdf/2510.21635v1](http://arxiv.org/pdf/2510.21635v1)**

> **作者:** Ziqi Gao; Qiufu Li; Linlin Shen
>
> **备注:** 14 pages, 7 figures, conference
>
> **摘要:** Compared to 2D data, the scale of point cloud data in different domains available for training, is quite limited. Researchers have been trying to combine these data of different domains for masked autoencoder (MAE) pre-training to leverage such a data scarcity issue. However, the prior knowledge learned from mixed domains may not align well with the downstream 3D point cloud analysis tasks, leading to degraded performance. To address such an issue, we propose the Domain-Adaptive Point Cloud Masked Autoencoder (DAP-MAE), an MAE pre-training method, to adaptively integrate the knowledge of cross-domain datasets for general point cloud analysis. In DAP-MAE, we design a heterogeneous domain adapter that utilizes an adaptation mode during pre-training, enabling the model to comprehensively learn information from point clouds across different domains, while employing a fusion mode in the fine-tuning to enhance point cloud features. Meanwhile, DAP-MAE incorporates a domain feature generator to guide the adaptation of point cloud features to various downstream tasks. With only one pre-training, DAP-MAE achieves excellent performance across four different point cloud analysis tasks, reaching 95.18% in object classification on ScanObjectNN and 88.45% in facial expression recognition on Bosphorus.
>
---
#### [new 042] NoisyGRPO: Incentivizing Multimodal CoT Reasoning via Noise Injection and Bayesian Estimation
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型在链式思维（CoT）推理中泛化能力弱的问题，提出NoisyGRPO框架。通过注入高斯噪声增强视觉探索，并基于贝叶斯推断建模优势估计，提升小规模模型的推理鲁棒性与泛化性能。**

- **链接: [http://arxiv.org/pdf/2510.21122v1](http://arxiv.org/pdf/2510.21122v1)**

> **作者:** Longtian Qiu; Shan Ning; Jiaxuan Sun; Xuming He
>
> **备注:** Accepted by Neurips2025, Project page at at https://artanic30.github.io/project_pages/NoisyGRPO/
>
> **摘要:** Reinforcement learning (RL) has shown promise in enhancing the general Chain-of-Thought (CoT) reasoning capabilities of multimodal large language models (MLLMs). However, when applied to improve general CoT reasoning, existing RL frameworks often struggle to generalize beyond the training distribution. To address this, we propose NoisyGRPO, a systematic multimodal RL framework that introduces controllable noise into visual inputs for enhanced exploration and explicitly models the advantage estimation process via a Bayesian framework. Specifically, NoisyGRPO improves RL training by: (1) \textbf{Noise-Injected Exploration Policy}: Perturbing visual inputs with Gaussian noise to encourage exploration across a wider range of visual scenarios; and (2) \textbf{Bayesian Advantage Estimation}: Formulating advantage estimation as a principled Bayesian inference problem, where the injected noise level serves as a prior and the observed trajectory reward as the likelihood. This Bayesian modeling fuses both sources of information to compute a robust posterior estimate of trajectory advantage, effectively guiding MLLMs to prefer visually grounded trajectories over noisy ones. Experiments on standard CoT quality, general capability, and hallucination benchmarks demonstrate that NoisyGRPO substantially improves generalization and robustness, especially in RL settings with small-scale MLLMs such as Qwen2.5-VL 3B. The project page is available at \href{https://artanic30.github.io/project_pages/NoisyGRPO/}{\texttt{https://artanic30.github.io/project\_pages/NoisyGRPO}}.
>
---
#### [new 043] KBE-DME: Dynamic Multimodal Evaluation via Knowledge Enhanced Benchmark Evolution
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型评估中静态基准的数据污染与饱和问题，提出知识增强的动态基准演化框架KBE。通过图结构建模视觉问答样本，融合多模态知识动态扩展并重构问题，实现可控难度的持续评估，提升评估可靠性与全面性。**

- **链接: [http://arxiv.org/pdf/2510.21182v1](http://arxiv.org/pdf/2510.21182v1)**

> **作者:** Junzhe Zhang; Huixuan Zhang; Xiaojun Wan
>
> **备注:** submitting to ICLR2026
>
> **摘要:** The rapid progress of multimodal large language models (MLLMs) calls for more reliable evaluation protocols. Existing static benchmarks suffer from the potential risk of data contamination and saturation, leading to inflated or misleading performance evaluations. To address these issues, we first apply Graph formulation to represent a static or dynamic VQA sample. With the formulation, we propose Knowledge-enhanced Benchmark Evolution(KBE), a dynamic multimodal evaluation framework. KBE first analyzes the original static benchmark, then expands it by integrating multimodal knowledge, transforming the static benchmark into a controllable, dynamic evolving version. Crucially, KBE can both reconstruct questions by Re-selecting visual information in the original image and expand existing questions with external textual knowledge. It enables difficulty-controllable evaluation by adjusting the degree of question exploration. Extensive experiments demonstrate that KBE alleviates the risk of data contamination, data saturation, and provides a more comprehensive assessment of MLLM capabilities.
>
---
#### [new 044] Visual Diffusion Models are Geometric Solvers
- **分类: cs.CV; cs.LG**

- **简介: 该论文将视觉扩散模型用于几何问题求解，通过像素空间中的图像生成直接推理几何结构。针对内接正方形、斯坦纳树和简单多边形等难题，训练标准扩散模型从噪声生成近似解，实现无需专用架构的通用几何求解。**

- **链接: [http://arxiv.org/pdf/2510.21697v1](http://arxiv.org/pdf/2510.21697v1)**

> **作者:** Nir Goren; Shai Yehezkel; Omer Dahary; Andrey Voynov; Or Patashnik; Daniel Cohen-Or
>
> **备注:** Project page: https://kariander1.github.io/visual-geo-solver/
>
> **摘要:** In this paper we show that visual diffusion models can serve as effective geometric solvers: they can directly reason about geometric problems by working in pixel space. We first demonstrate this on the Inscribed Square Problem, a long-standing problem in geometry that asks whether every Jordan curve contains four points forming a square. We then extend the approach to two other well-known hard geometric problems: the Steiner Tree Problem and the Simple Polygon Problem. Our method treats each problem instance as an image and trains a standard visual diffusion model that transforms Gaussian noise into an image representing a valid approximate solution that closely matches the exact one. The model learns to transform noisy geometric structures into correct configurations, effectively recasting geometric reasoning as image generation. Unlike prior work that necessitates specialized architectures and domain-specific adaptations when applying diffusion to parametric geometric representations, we employ a standard visual diffusion model that operates on the visual representation of the problem. This simplicity highlights a surprising bridge between generative modeling and geometric problem solving. Beyond the specific problems studied here, our results point toward a broader paradigm: operating in image space provides a general and practical framework for approximating notoriously hard problems, and opens the door to tackling a far wider class of challenging geometric tasks.
>
---
#### [new 045] OpenHype: Hyperbolic Embeddings for Hierarchical Open-Vocabulary Radiance Fields
- **分类: cs.CV**

- **简介: 该论文提出OpenHype，用于建模3D场景的层次结构。针对现有方法在隐式表示中难以捕捉多尺度层级关系、依赖封闭离散层级或需多次渲染的问题，提出基于连续双曲空间的嵌入方法，利用双曲几何自然表达层次结构，实现高效、平滑的层次遍历与开放词汇场景理解。**

- **链接: [http://arxiv.org/pdf/2510.21441v1](http://arxiv.org/pdf/2510.21441v1)**

> **作者:** Lisa Weijler; Sebastian Koch; Fabio Poiesi; Timo Ropinski; Pedro Hermosilla
>
> **摘要:** Modeling the inherent hierarchical structure of 3D objects and 3D scenes is highly desirable, as it enables a more holistic understanding of environments for autonomous agents. Accomplishing this with implicit representations, such as Neural Radiance Fields, remains an unexplored challenge. Existing methods that explicitly model hierarchical structures often face significant limitations: they either require multiple rendering passes to capture embeddings at different levels of granularity, significantly increasing inference time, or rely on predefined, closed-set discrete hierarchies that generalize poorly to the diverse and nuanced structures encountered by agents in the real world. To address these challenges, we propose OpenHype, a novel approach that represents scene hierarchies using a continuous hyperbolic latent space. By leveraging the properties of hyperbolic geometry, OpenHype naturally encodes multi-scale relationships and enables smooth traversal of hierarchies through geodesic paths in latent space. Our method outperforms state-of-the-art approaches on standard benchmarks, demonstrating superior efficiency and adaptability in 3D scene understanding.
>
---
#### [new 046] Blockwise Flow Matching: Improving Flow Matching Models For Efficient High-Quality Generation
- **分类: cs.CV**

- **简介: 该论文针对流匹配模型生成效率低、难以捕捉时序特征的问题，提出块状流匹配（BFM）框架。通过将生成轨迹分段并设计专用速度块，结合语义特征引导与轻量残差近似，显著提升推理效率与生成质量，在ImageNet上实现2.1x–4.9x加速。**

- **链接: [http://arxiv.org/pdf/2510.21167v1](http://arxiv.org/pdf/2510.21167v1)**

> **作者:** Dogyun Park; Taehoon Lee; Minseok Joo; Hyunwoo J. Kim
>
> **摘要:** Recently, Flow Matching models have pushed the boundaries of high-fidelity data generation across a wide range of domains. It typically employs a single large network to learn the entire generative trajectory from noise to data. Despite their effectiveness, this design struggles to capture distinct signal characteristics across timesteps simultaneously and incurs substantial inference costs due to the iterative evaluation of the entire model. To address these limitations, we propose Blockwise Flow Matching (BFM), a novel framework that partitions the generative trajectory into multiple temporal segments, each modeled by smaller but specialized velocity blocks. This blockwise design enables each block to specialize effectively in its designated interval, improving inference efficiency and sample quality. To further enhance generation fidelity, we introduce a Semantic Feature Guidance module that explicitly conditions velocity blocks on semantically rich features aligned with pretrained representations. Additionally, we propose a lightweight Feature Residual Approximation strategy that preserves semantic quality while significantly reducing inference cost. Extensive experiments on ImageNet 256x256 demonstrate that BFM establishes a substantially improved Pareto frontier over existing Flow Matching methods, achieving 2.1x to 4.9x accelerations in inference complexity at comparable generation performance. Code is available at https://github.com/mlvlab/BFM.
>
---
#### [new 047] Foundation Models in Dermatopathology: Skin Tissue Classification
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于皮肤组织病理图像分类任务，旨在解决全切片图像（WSI）自动化分类难题。研究采用UNI和Virchow2基础模型提取局部特征，通过均值聚合生成整体图像特征，结合多种机器学习模型进行分类，验证了Virchow2在性能上的优势，提升了诊断效率与准确性。**

- **链接: [http://arxiv.org/pdf/2510.21664v1](http://arxiv.org/pdf/2510.21664v1)**

> **作者:** Riya Gupta; Yiwei Zong; Dennis H. Murphree
>
> **摘要:** The rapid generation of whole-slide images (WSIs) in dermatopathology necessitates automated methods for efficient processing and accurate classification. This study evaluates the performance of two foundation models, UNI and Virchow2, as feature extractors for classifying WSIs into three diagnostic categories: melanocytic, basaloid, and squamous lesions. Patch-level embeddings were aggregated into slide-level features using a mean-aggregation strategy and subsequently used to train multiple machine learning classifiers, including logistic regression, gradient-boosted trees, and random forest models. Performance was assessed using precision, recall, true positive rate, false positive rate, and the area under the receiver operating characteristic curve (AUROC) on the test set. Results demonstrate that patch-level features extracted using Virchow2 outperformed those extracted via UNI across most slide-level classifiers, with logistic regression achieving the highest accuracy (90%) for Virchow2, though the difference was not statistically significant. The study also explored data augmentation techniques and image normalization to enhance model robustness and generalizability. The mean-aggregation approach provided reliable slide-level feature representations. All experimental results and metrics were tracked and visualized using WandB.ai, facilitating reproducibility and interpretability. This research highlights the potential of foundation models for automated WSI classification, providing a scalable and effective approach for dermatopathological diagnosis while paving the way for future advancements in slide-level representation learning.
>
---
#### [new 048] Urban 3D Change Detection Using LiDAR Sensor for HD Map Maintenance and Smart Mobility
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对城市级3D变化检测任务，解决LiDAR数据在高精地图维护中因配准误差、遮挡和对象分裂/合并导致的误检问题。提出基于多分辨率NDT与点到平面ICP的对齐方法，结合不确定性感知的局部检测水平，通过语义与实例分割及类约束分配实现精准关联，支持分块处理与实例级决策，显著提升检测精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21112v1](http://arxiv.org/pdf/2510.21112v1)**

> **作者:** Hezam Albagami; Haitian Wang; Xinyu Wang; Muhammad Ibrahim; Zainy M. Malakan; Abdullah M. Alqamdi; Mohammed H. Alghamdi; Ajmal Mian
>
> **摘要:** High-definition 3D city maps underpin smart transportation, digital twins, and autonomous driving, where object level change detection across bi temporal LiDAR enables HD map maintenance, construction monitoring, and reliable localization. Classical DSM differencing and image based methods are sensitive to small vertical bias, ground slope, and viewpoint mismatch and yield cellwise outputs without object identity. Point based neural models and voxel encodings demand large memory, assume near perfect pre alignment, degrade thin structures, and seldom enforce class consistent association, which leaves split or merge cases unresolved and ignores uncertainty. We propose an object centric, uncertainty aware pipeline for city scale LiDAR that aligns epochs with multi resolution NDT followed by point to plane ICP, normalizes height, and derives a per location level of detection from registration covariance and surface roughness to calibrate decisions and suppress spurious changes. Geometry only proxies seed cross epoch associations that are refined by semantic and instance segmentation and a class constrained bipartite assignment with augmented dummies to handle splits and merges while preserving per class counts. Tiled processing bounds memory without eroding narrow ground changes, and instance level decisions combine 3D overlap, normal direction displacement, and height and volume differences with a histogram distance, all gated by the local level of detection to remain stable under partial overlap and sampling variation. On 15 representative Subiaco blocks the method attains 95.2% accuracy, 90.4% mF1, and 82.6% mIoU, exceeding Triplet KPConv by 0.2 percentage points in accuracy, 0.2 in mF1, and 0.8 in mIoU, with the largest gain on Decreased where IoU reaches 74.8% and improves by 7.6 points.
>
---
#### [new 049] Foley Control: Aligning a Frozen Latent Text-to-Audio Model to Video
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出Foley Control，用于视频引导的音效生成任务。针对现有方法需重训练模型、参数多的问题，提出冻结预训练音视频模型，仅添加轻量级交叉注意力桥接，实现高效同步。通过视频控制音频时序与细节，保留文本语义控制，支持模块灵活替换，显著减少参数并提升实用性。**

- **链接: [http://arxiv.org/pdf/2510.21581v1](http://arxiv.org/pdf/2510.21581v1)**

> **作者:** Ciara Rowles; Varun Jampani; Simon Donné; Shimon Vainer; Julian Parker; Zach Evans
>
> **备注:** Project Page: https://stability-ai.github.io/foleycontrol.github.io/
>
> **摘要:** Foley Control is a lightweight approach to video-guided Foley that keeps pretrained single-modality models frozen and learns only a small cross-attention bridge between them. We connect V-JEPA2 video embeddings to a frozen Stable Audio Open DiT text-to-audio (T2A) model by inserting compact video cross-attention after the model's existing text cross-attention, so prompts set global semantics while video refines timing and local dynamics. The frozen backbones retain strong marginals (video; audio given text) and the bridge learns the audio-video dependency needed for synchronization -- without retraining the audio prior. To cut memory and stabilize training, we pool video tokens before conditioning. On curated video-audio benchmarks, Foley Control delivers competitive temporal and semantic alignment with far fewer trainable parameters than recent multi-modal systems, while preserving prompt-driven controllability and production-friendly modularity (swap/upgrade encoders or the T2A backbone without end-to-end retraining). Although we focus on Video-to-Foley, the same bridge design can potentially extend to other audio modalities (e.g., speech).
>
---
#### [new 050] CXR-LanIC: Language-Grounded Interpretable Classifier for Chest X-Ray Diagnosis
- **分类: cs.CV**

- **简介: 该论文针对胸部X光诊断中深度学习模型黑箱问题，提出CXR-LanIC框架。通过在特定诊断任务上训练稀疏自编码器，从医学影像中提取5000个可解释的视觉模式，实现高精度诊断与透明化解释，使模型决策可验证、可理解，推动医疗AI临床落地。**

- **链接: [http://arxiv.org/pdf/2510.21464v1](http://arxiv.org/pdf/2510.21464v1)**

> **作者:** Yiming Tang; Wenjia Zhong; Rushi Shah; Dianbo Liu
>
> **摘要:** Deep learning models have achieved remarkable accuracy in chest X-ray diagnosis, yet their widespread clinical adoption remains limited by the black-box nature of their predictions. Clinicians require transparent, verifiable explanations to trust automated diagnoses and identify potential failure modes. We introduce CXR-LanIC (Language-Grounded Interpretable Classifier for Chest X-rays), a novel framework that addresses this interpretability challenge through task-aligned pattern discovery. Our approach trains transcoder-based sparse autoencoders on a BiomedCLIP diagnostic classifier to decompose medical image representations into interpretable visual patterns. By training an ensemble of 100 transcoders on multimodal embeddings from the MIMIC-CXR dataset, we discover approximately 5,000 monosemantic patterns spanning cardiac, pulmonary, pleural, structural, device, and artifact categories. Each pattern exhibits consistent activation behavior across images sharing specific radiological features, enabling transparent attribution where predictions decompose into 20-50 interpretable patterns with verifiable activation galleries. CXR-LanIC achieves competitive diagnostic accuracy on five key findings while providing the foundation for natural language explanations through planned large multimodal model annotation. Our key innovation lies in extracting interpretable features from a classifier trained on specific diagnostic objectives rather than general-purpose embeddings, ensuring discovered patterns are directly relevant to clinical decision-making, demonstrating that medical AI systems can be both accurate and interpretable, supporting safer clinical deployment through transparent, clinically grounded explanations.
>
---
#### [new 051] 3rd Place Solution to ICCV LargeFineFoodAI Retrieval
- **分类: cs.CV**

- **简介: 该论文针对大型精细食物图像检索任务，提出融合ArcFace与Circle损失的模型，结合TTA、集成学习与新型扩散k-互惠重排序方法，提升特征表示与检索精度，最终在公开与私有榜单上分别取得0.81219与0.81191的mAP@100成绩。**

- **链接: [http://arxiv.org/pdf/2510.21198v1](http://arxiv.org/pdf/2510.21198v1)**

> **作者:** Yang Zhong; Zhiming Wang; Zhaoyang Li; Jinyu Ma; Xiang Li
>
> **摘要:** This paper introduces the 3rd place solution to the ICCV LargeFineFoodAI Retrieval Competition on Kaggle. Four basic models are independently trained with the weighted sum of ArcFace and Circle loss, then TTA and Ensemble are successively applied to improve feature representation ability. In addition, a new reranking method for retrieval is proposed based on diffusion and k-reciprocal reranking. Finally, our method scored 0.81219 and 0.81191 mAP@100 on the public and private leaderboard, respectively.
>
---
#### [new 052] ITC-RWKV: Interactive Tissue-Cell Modeling with Recurrent Key-Value Aggregation for Histopathological Subtyping
- **分类: cs.CV**

- **简介: 该论文针对病理图像细粒度分型任务，解决现有模型忽视细胞级特征的问题。提出双流架构，结合组织宏观特征与细胞聚合表示，设计线性复杂度的递归键值聚合模块及双向交互机制，有效建模细胞间依赖与组织-细胞互作，显著提升亚型分类性能。**

- **链接: [http://arxiv.org/pdf/2510.21479v1](http://arxiv.org/pdf/2510.21479v1)**

> **作者:** Yating Huang; Qijun Yang; Lintao Xiang; Hujun Yin
>
> **备注:** Accept by BMVC 2025
>
> **摘要:** Accurate interpretation of histopathological images demands integration of information across spatial and semantic scales, from nuclear morphology and cellular textures to global tissue organization and disease-specific patterns. Although recent foundation models in pathology have shown strong capabilities in capturing global tissue context, their omission of cell-level feature modeling remains a key limitation for fine-grained tasks such as cancer subtype classification. To address this, we propose a dual-stream architecture that models the interplay between macroscale tissue features and aggregated cellular representations. To efficiently aggregate information from large cell sets, we propose a receptance-weighted key-value aggregation model, a recurrent transformer that captures inter-cell dependencies with linear complexity. Furthermore, we introduce a bidirectional tissue-cell interaction module to enable mutual attention between localized cellular cues and their surrounding tissue environment. Experiments on four histopathological subtype classification benchmarks show that the proposed method outperforms existing models, demonstrating the critical role of cell-level aggregation and tissue-cell interaction in fine-grained computational pathology.
>
---
#### [new 053] Sample By Step, Optimize By Chunk: Chunk-Level GRPO For Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像生成中的GRPO方法，解决其优势归因不准和忽略生成时序动态的问题。提出分块级优化的Chunk-GRPO，将连续生成步骤分组为语义连贯的“块”，在块级别进行策略优化，并引入加权采样策略，显著提升图像质量和偏好对齐效果。**

- **链接: [http://arxiv.org/pdf/2510.21583v1](http://arxiv.org/pdf/2510.21583v1)**

> **作者:** Yifu Luo; Penghui Du; Bo Li; Sinan Du; Tiantian Zhang; Yongzhe Chang; Kai Wu; Kun Gai; Xueqian Wang
>
> **备注:** 11 pages, preprint
>
> **摘要:** Group Relative Policy Optimization (GRPO) has shown strong potential for flow-matching-based text-to-image (T2I) generation, but it faces two key limitations: inaccurate advantage attribution, and the neglect of temporal dynamics of generation. In this work, we argue that shifting the optimization paradigm from the step level to the chunk level can effectively alleviate these issues. Building on this idea, we propose Chunk-GRPO, the first chunk-level GRPO-based approach for T2I generation. The insight is to group consecutive steps into coherent 'chunk's that capture the intrinsic temporal dynamics of flow matching, and to optimize policies at the chunk level. In addition, we introduce an optional weighted sampling strategy to further enhance performance. Extensive experiments show that ChunkGRPO achieves superior results in both preference alignment and image quality, highlighting the promise of chunk-level optimization for GRPO-based methods.
>
---
#### [new 054] PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments
- **分类: cs.CV**

- **简介: 该论文提出主动视觉推理（AVR）任务，解决多模态大模型在部分可观测物理环境中因信息不全而推理失效的问题。通过构建CLEVR-AVR基准与AVR-152k数据集，训练出PhysVLM-AVR模型，实现基于交互的动态信息获取与整合，显著提升在复杂环境中的推理能力。**

- **链接: [http://arxiv.org/pdf/2510.21111v1](http://arxiv.org/pdf/2510.21111v1)**

> **作者:** Weijie Zhou; Xuantang Xiong; Yi Peng; Manli Tao; Chaoyang Zhao; Honghui Dong; Ming Tang; Jinqiao Wang
>
> **备注:** 39th Conference on Neural Information Processing Systemss (NeurIPS 2025)
>
> **摘要:** Visual reasoning in multimodal large language models (MLLMs) has primarily been studied in static, fully observable settings, limiting their effectiveness in real-world environments where information is often incomplete due to occlusion or limited field of view. Humans, in contrast, actively explore and interact with their environment-moving, examining, and manipulating objects-to gather information through a closed-loop process integrating perception, reasoning, and action. Inspired by this human capability, we introduce the Active Visual Reasoning (AVR) task, extending visual reasoning to partially observable, interactive environments. AVR necessitates agents to: (1) actively acquire information via sequential physical actions, (2) integrate observations across multiple steps for coherent reasoning, and (3) dynamically adjust decisions based on evolving visual feedback. To rigorously evaluate AVR, we introduce CLEVR-AVR, a simulation benchmark featuring multi-round interactive environments designed to assess both reasoning correctness and information-gathering efficiency. We present AVR-152k, a large-scale dataset that offers rich Chain-of-Thought (CoT) annotations detailing iterative reasoning for uncertainty identification, action-conditioned information gain prediction, and information-maximizing action selection, crucial for training agents in a higher-order Markov Decision Process. Building on this, we develop PhysVLM-AVR, an MLLM achieving state-of-the-art performance on CLEVR-AVR, embodied reasoning (OpenEQA, RoboVQA), and passive visual reasoning (GeoMath, Geometry30K). Our analysis also reveals that current embodied MLLMs, despite detecting information incompleteness, struggle to actively acquire and integrate new information through interaction, highlighting a fundamental gap in active reasoning capabilities.
>
---
#### [new 055] Anisotropic Pooling for LUT-realizable CNN Image Restoration
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对LUT实现的CNN图像修复任务，解决因平均池化不适应各向异性结构导致的性能瓶颈。提出各向异性池化方法，包括广义中值池化及可学习的定向加权机制，有效提升修复质量与效率。**

- **链接: [http://arxiv.org/pdf/2510.21437v1](http://arxiv.org/pdf/2510.21437v1)**

> **作者:** Xi Zhang; Xiaolin Wu
>
> **摘要:** Table look-up realization of image restoration CNNs has the potential of achieving competitive image quality while being much faster and resource frugal than the straightforward CNN implementation. The main technical challenge facing the LUT-based CNN algorithm designers is to manage the table size without overly restricting the receptive field. The prevailing strategy is to reuse the table for small pixel patches of different orientations (apparently assuming a degree of isotropy) and then fuse the look-up results. The fusion is currently done by average pooling, which we find being ill suited to anisotropic signal structures. To alleviate the problem, we investigate and discuss anisotropic pooling methods to replace naive averaging for improving the performance of the current LUT-realizable CNN restoration methods. First, we introduce the method of generalized median pooling which leads to measurable gains over average pooling. We then extend this idea by learning data-dependent pooling coefficients for each orientation, so that they can adaptively weigh the contributions of differently oriented pixel patches. Experimental results on various restoration benchmarks show that our anisotropic pooling strategy yields both perceptually and numerically superior results compared to existing LUT-realizable CNN methods.
>
---
#### [new 056] Self-Supervised Learning of Synapse Types from EM Images
- **分类: cs.CV**

- **简介: 该论文提出一种自监督学习方法，用于从电子显微镜图像中分类突触类型。任务是无监督地识别突触亚型，解决传统方法需预先指定类别数和标注数据的问题。基于同一神经元内突触更相似的假设，利用数据内在结构进行聚类，无需先验知识，可自动发现突触类型并提供结构化的真值参考。**

- **链接: [http://arxiv.org/pdf/2510.21663v1](http://arxiv.org/pdf/2510.21663v1)**

> **作者:** Aarav Shetty; Gary B Huang
>
> **摘要:** Separating synapses into different classes based on their appearance in EM images has many applications in biology. Examples may include assigning a neurotransmitter to a particular class, or separating synapses whose strength can be modulated from those whose strength is fixed. Traditionally, this has been done in a supervised manner, giving the classification algorithm examples of the different classes. Here we instead separate synapses into classes based only on the observation that nearby synapses in the same neuron are likely more similar than synapses chosen randomly from different cells. We apply our methodology to data from {\it Drosophila}. Our approach has the advantage that the number of synapse types does not need to be known in advance. It may also provide a principled way to select ground-truth that spans the range of synapse structure.
>
---
#### [new 057] WaveSeg: Enhancing Segmentation Precision via High-Frequency Prior and Mamba-Driven Spectrum Decomposition
- **分类: cs.CV**

- **简介: 该论文针对语义分割中细节与语义平衡问题，提出WaveSeg框架。通过引入高频先验与基于Mamba的谱分解注意力机制，在空间与小波域联合优化特征，提升边界精度与语义完整性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.21079v1](http://arxiv.org/pdf/2510.21079v1)**

> **作者:** Guoan Xu; Yang Xiao; Wenjing Jia; Guangwei Gao; Guo-Jun Qi; Chia-Wen Lin
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** While recent semantic segmentation networks heavily rely on powerful pretrained encoders, most employ simplistic decoders, leading to suboptimal trade-offs between semantic context and fine-grained detail preservation. To address this, we propose a novel decoder architecture, WaveSeg, which jointly optimizes feature refinement in spatial and wavelet domains. Specifically, high-frequency components are first learned from input images as explicit priors to reinforce boundary details at early stages. A multi-scale fusion mechanism, Dual Domain Operation (DDO), is then applied, and the novel Spectrum Decomposition Attention (SDA) block is proposed, which is developed to leverage Mamba's linear-complexity long-range modeling to enhance high-frequency structural details. Meanwhile, reparameterized convolutions are applied to preserve low-frequency semantic integrity in the wavelet domain. Finally, a residual-guided fusion integrates multi-scale features with boundary-aware representations at native resolution, producing semantically and structurally rich feature maps. Extensive experiments on standard benchmarks demonstrate that WaveSeg, leveraging wavelet-domain frequency prior with Mamba-based attention, consistently outperforms state-of-the-art approaches both quantitatively and qualitatively, achieving efficient and precise segmentation.
>
---
#### [new 058] BioDet: Boosting Industrial Object Detection with Image Preprocessing Strategies
- **分类: cs.CV**

- **简介: 该论文针对工业场景下6D姿态估计中的对象检测瓶颈问题，提出BioDet框架。通过低光照增强与基于大模型的背景去除，降低域偏移与误检，提升未见物体的2D检测精度，显著改善下游姿态估计效果，且推理开销极小。**

- **链接: [http://arxiv.org/pdf/2510.21000v1](http://arxiv.org/pdf/2510.21000v1)**

> **作者:** Jiaqi Hu; Hongli Xu; Junwen Huang; Peter KT Yu; Slobodan Ilic; Benjamin Busam
>
> **备注:** 8 pages, accepted by ICCV 2025 R6D
>
> **摘要:** Accurate 6D pose estimation is essential for robotic manipulation in industrial environments. Existing pipelines typically rely on off-the-shelf object detectors followed by cropping and pose refinement, but their performance degrades under challenging conditions such as clutter, poor lighting, and complex backgrounds, making detection the critical bottleneck. In this work, we introduce a standardized and plug-in pipeline for 2D detection of unseen objects in industrial settings. Based on current SOTA baselines, our approach reduces domain shift and background artifacts through low-light image enhancement and background removal guided by open-vocabulary detection with foundation models. This design suppresses the false positives prevalent in raw SAM outputs, yielding more reliable detections for downstream pose estimation. Extensive experiments on real-world industrial bin-picking benchmarks from BOP demonstrate that our method significantly boosts detection accuracy while incurring negligible inference overhead, showing the effectiveness and practicality of the proposed method.
>
---
#### [new 059] On Thin Ice: Towards Explainable Conservation Monitoring via Attribution and Perturbations
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对生态监测中黑箱模型可信度低的问题，提出结合梯度与扰动解释方法（如HiResCAM、LIME）提升目标检测模型的可解释性。基于冰川湾国家公园航拍图像，训练Faster R-CNN检测海豹，通过定位精度、忠实性与诊断能力评估解释效果，发现模型误将黑冰和岩石识别为海豹，并提出数据优化建议，推动可审计的保护决策工具发展。**

- **链接: [http://arxiv.org/pdf/2510.21689v1](http://arxiv.org/pdf/2510.21689v1)**

> **作者:** Jiayi Zhou; Günel Aghakishiyeva; Saagar Arya; Julian Dale; James David Poling; Holly R. Houliston; Jamie N. Womble; Gregory D. Larsen; David W. Johnston; Brinnae Bent
>
> **备注:** NeurIPS Imageomics Workshop 2025
>
> **摘要:** Computer vision can accelerate ecological research and conservation monitoring, yet adoption in ecology lags in part because of a lack of trust in black-box neural-network-based models. We seek to address this challenge by applying post-hoc explanations to provide evidence for predictions and document limitations that are important to field deployment. Using aerial imagery from Glacier Bay National Park, we train a Faster R-CNN to detect pinnipeds (harbor seals) and generate explanations via gradient-based class activation mapping (HiResCAM, LayerCAM), local interpretable model-agnostic explanations (LIME), and perturbation-based explanations. We assess explanations along three axes relevant to field use: (i) localization fidelity: whether high-attribution regions coincide with the animal rather than background context; (ii) faithfulness: whether deletion/insertion tests produce changes in detector confidence; and (iii) diagnostic utility: whether explanations reveal systematic failure modes. Explanations concentrate on seal torsos and contours rather than surrounding ice/rock, and removal of the seals reduces detection confidence, providing model-evidence for true positives. The analysis also uncovers recurrent error sources, including confusion between seals and black ice and rocks. We translate these findings into actionable next steps for model development, including more targeted data curation and augmentation. By pairing object detection with post-hoc explainability, we can move beyond "black-box" predictions toward auditable, decision-supporting tools for conservation monitoring.
>
---
#### [new 060] Dynamic Semantic-Aware Correlation Modeling for UAV Tracking
- **分类: cs.CV**

- **简介: 该论文针对无人机跟踪任务，解决现有方法因缺乏语义感知导致定位不准的问题。提出动态语义相关性建模框架，通过动态语义相关性生成器增强模板与搜索区域的语义关联，提升在运动模糊、快速运动等挑战下的精度与鲁棒性，并设计剪枝策略实现速度与精度的灵活权衡。**

- **链接: [http://arxiv.org/pdf/2510.21351v1](http://arxiv.org/pdf/2510.21351v1)**

> **作者:** Xinyu Zhou; Tongxin Pan; Lingyi Hong; Pinxue Guo; Haijing Guo; Zhaoyu Chen; Kaixun Jiang; Wenqiang Zhang
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** UAV tracking can be widely applied in scenarios such as disaster rescue, environmental monitoring, and logistics transportation. However, existing UAV tracking methods predominantly emphasize speed and lack exploration in semantic awareness, which hinders the search region from extracting accurate localization information from the template. The limitation results in suboptimal performance under typical UAV tracking challenges such as camera motion, fast motion, and low resolution, etc. To address this issue, we propose a dynamic semantic aware correlation modeling tracking framework. The core of our framework is a Dynamic Semantic Relevance Generator, which, in combination with the correlation map from the Transformer, explore semantic relevance. The approach enhances the search region's ability to extract important information from the template, improving accuracy and robustness under the aforementioned challenges. Additionally, to enhance the tracking speed, we design a pruning method for the proposed framework. Therefore, we present multiple model variants that achieve trade-offs between speed and accuracy, enabling flexible deployment according to the available computational resources. Experimental results validate the effectiveness of our method, achieving competitive performance on multiple UAV tracking datasets. The code is available at https://github.com/zxyyxzz/DSATrack.
>
---
#### [new 061] TokenCLIP: Token-wise Prompt Learning for Zero-shot Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出TokenCLIP，用于零样本异常检测。针对现有方法依赖单一文本空间导致异常语义捕捉不精准的问题，提出基于最优传输的词元级动态对齐机制，将视觉词元自适应分配至正交子空间组合，实现细粒度异常学习。**

- **链接: [http://arxiv.org/pdf/2510.21171v1](http://arxiv.org/pdf/2510.21171v1)**

> **作者:** Qihang Zhou; Binbin Gao; Guansong Pang; Xin Wang; Jiming Chen; Shibo He
>
> **摘要:** Adapting CLIP for anomaly detection on unseen objects has shown strong potential in a zero-shot manner. However, existing methods typically rely on a single textual space to align with visual semantics across diverse objects and domains. The indiscriminate alignment hinders the model from accurately capturing varied anomaly semantics. We propose TokenCLIP, a token-wise adaptation framework that enables dynamic alignment between visual and learnable textual spaces for fine-grained anomaly learning. Rather than mapping all visual tokens to a single, token-agnostic textual space, TokenCLIP aligns each token with a customized textual subspace that represents its visual characteristics. Explicitly assigning a unique learnable textual space to each token is computationally intractable and prone to insufficient optimization. We instead expand the token-agnostic textual space into a set of orthogonal subspaces, and then dynamically assign each token to a subspace combination guided by semantic affinity, which jointly supports customized and efficient token-wise adaptation. To this end, we formulate dynamic alignment as an optimal transport problem, where all visual tokens in an image are transported to textual subspaces based on semantic similarity. The transport constraints of OT ensure sufficient optimization across subspaces and encourage them to focus on different semantics. Solving the problem yields a transport plan that adaptively assigns each token to semantically relevant subspaces. A top-k masking is then applied to sparsify the plan and specialize subspaces for distinct visual regions. Extensive experiments demonstrate the superiority of TokenCLIP.
>
---
#### [new 062] Why Registration Quality Matters: Enhancing sCT Synthesis with IMPACT-Based Registration
- **分类: cs.CV**

- **简介: 该论文聚焦于医学图像合成中的sCT生成任务，旨在提升合成质量。针对注册质量对模型性能的影响，提出基于IMPACT的特征匹配注册方法，相比传统互信息注册更优，显著增强解剖一致性与结构保真度，有效缓解注册偏差问题，推动模型泛化能力提升。**

- **链接: [http://arxiv.org/pdf/2510.21358v1](http://arxiv.org/pdf/2510.21358v1)**

> **作者:** Valentin Boussot; Cédric Hémon; Jean-Claude Nunes; Jean-Louis Dillenseger
>
> **备注:** Paper for the SynthRAD2025 challenge, Team BreizhCT
>
> **摘要:** We participated in the SynthRAD2025 challenge (Tasks 1 and 2) with a unified pipeline for synthetic CT (sCT) generation from MRI and CBCT, implemented using the KonfAI framework. Our model is a 2.5D U-Net++ with a ResNet-34 encoder, trained jointly across anatomical regions and fine-tuned per region. The loss function combined pixel-wise L1 loss with IMPACT-Synth, a perceptual loss derived from SAM and TotalSegmentator to enhance structural fidelity. Training was performed using AdamW (initial learning rate = 0.001, halved every 25k steps) on patch-based, normalized, body-masked inputs (320x320 for MRI, 256x256 for CBCT), with random flipping as the only augmentation. No post-processing was applied. Final predictions leveraged test-time augmentation and five-fold ensembling. The best model was selected based on validation MAE. Two registration strategies were evaluated: (i) Elastix with mutual information, consistent with the challenge pipeline, and (ii) IMPACT, a feature-based similarity metric leveraging pretrained segmentation networks. On the local test sets, IMPACT-based registration achieved more accurate and anatomically consistent alignments than mutual-information-based registration, resulting in improved sCT synthesis with lower MAE and more realistic anatomical structures. On the public validation set, however, models trained with Elastix-aligned data achieved higher scores, reflecting a registration bias favoring alignment strategies consistent with the evaluation pipeline. This highlights how registration errors can propagate into supervised learning, influencing both training and evaluation, and potentially inflating performance metrics at the expense of anatomical fidelity. By promoting anatomically consistent alignment, IMPACT helps mitigate this bias and supports the development of more robust and generalizable sCT synthesis models.
>
---
#### [new 063] VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言模型（VLM）中多模态表示对齐的可解释性问题，提出VL-SAE稀疏自编码器。通过统一概念集映射神经元与语义概念，实现对对齐机制的解释，并在概念层面增强对齐，提升零样本分类与抗幻觉性能。**

- **链接: [http://arxiv.org/pdf/2510.21323v1](http://arxiv.org/pdf/2510.21323v1)**

> **作者:** Shufan Shen; Junshu Sun; Qingming Huang; Shuhui Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** The alignment of vision-language representations endows current Vision-Language Models (VLMs) with strong multi-modal reasoning capabilities. However, the interpretability of the alignment component remains uninvestigated due to the difficulty in mapping the semantics of multi-modal representations into a unified concept set. To address this problem, we propose VL-SAE, a sparse autoencoder that encodes vision-language representations into its hidden activations. Each neuron in its hidden layer correlates to a concept represented by semantically similar images and texts, thereby interpreting these representations with a unified concept set. To establish the neuron-concept correlation, we encourage semantically similar representations to exhibit consistent neuron activations during self-supervised training. First, to measure the semantic similarity of multi-modal representations, we perform their alignment in an explicit form based on cosine similarity. Second, we construct the VL-SAE with a distance-based encoder and two modality-specific decoders to ensure the activation consistency of semantically similar representations. Experiments across multiple VLMs (e.g., CLIP, LLaVA) demonstrate the superior capability of VL-SAE in interpreting and enhancing the vision-language alignment. For interpretation, the alignment between vision and language representations can be understood by comparing their semantics with concepts. For enhancement, the alignment can be strengthened by aligning vision-language representations at the concept level, contributing to performance improvements in downstream tasks, including zero-shot image classification and hallucination elimination. Codes are available at https://github.com/ssfgunner/VL-SAE.
>
---
#### [new 064] S3OD: Towards Generalizable Salient Object Detection with Synthetic Data
- **分类: cs.CV**

- **简介: 该论文聚焦于显著性物体检测任务，针对标注数据稀缺与跨数据集泛化差的问题，提出S3OD方法。通过大规模合成数据生成与模糊感知架构，利用多模态扩散模型生成13.9万张高分辨率图像，并设计多掩码解码器处理检测不确定性。仅用合成数据训练的模型即实现跨数据集性能显著提升。**

- **链接: [http://arxiv.org/pdf/2510.21605v1](http://arxiv.org/pdf/2510.21605v1)**

> **作者:** Orest Kupyn; Hirokatsu Kataoka; Christian Rupprecht
>
> **摘要:** Salient object detection exemplifies data-bounded tasks where expensive pixel-precise annotations force separate model training for related subtasks like DIS and HR-SOD. We present a method that dramatically improves generalization through large-scale synthetic data generation and ambiguity-aware architecture. We introduce S3OD, a dataset of over 139,000 high-resolution images created through our multi-modal diffusion pipeline that extracts labels from diffusion and DINO-v3 features. The iterative generation framework prioritizes challenging categories based on model performance. We propose a streamlined multi-mask decoder that naturally handles the inherent ambiguity in salient object detection by predicting multiple valid interpretations. Models trained solely on synthetic data achieve 20-50% error reduction in cross-dataset generalization, while fine-tuned versions reach state-of-the-art performance across DIS and HR-SOD benchmarks.
>
---
#### [new 065] PhysWorld: From Real Videos to World Models of Deformable Objects via Physics-Aware Demonstration Synthesis
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出PhysWorld框架，旨在从有限真实视频中学习可变形物体的物理一致动态模型。针对数据稀缺与物理一致性难题，利用物理模拟器构建数字孪生，通过局部属性扰动生成多样化演示，训练轻量级图神经网络世界模型，并用真实视频优化物理参数，实现高效精准的未来预测与新交互泛化。**

- **链接: [http://arxiv.org/pdf/2510.21447v1](http://arxiv.org/pdf/2510.21447v1)**

> **作者:** Yu Yang; Zhilu Zhang; Xiang Zhang; Yihan Zeng; Hui Li; Wangmeng Zuo
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Interactive world models that simulate object dynamics are crucial for robotics, VR, and AR. However, it remains a significant challenge to learn physics-consistent dynamics models from limited real-world video data, especially for deformable objects with spatially-varying physical properties. To overcome the challenge of data scarcity, we propose PhysWorld, a novel framework that utilizes a simulator to synthesize physically plausible and diverse demonstrations to learn efficient world models. Specifically, we first construct a physics-consistent digital twin within MPM simulator via constitutive model selection and global-to-local optimization of physical properties. Subsequently, we apply part-aware perturbations to the physical properties and generate various motion patterns for the digital twin, synthesizing extensive and diverse demonstrations. Finally, using these demonstrations, we train a lightweight GNN-based world model that is embedded with physical properties. The real video can be used to further refine the physical properties. PhysWorld achieves accurate and fast future predictions for various deformable objects, and also generalizes well to novel interactions. Experiments show that PhysWorld has competitive performance while enabling inference speeds 47 times faster than the recent state-of-the-art method, i.e., PhysTwin.
>
---
#### [new 066] Towards Physically Executable 3D Gaussian for Embodied Navigation
- **分类: cs.CV**

- **简介: 该论文针对视觉语言导航（VLN）中3D高斯溅射（3DGS）缺乏语义与物理可执行性的问题，提出SAGE-3D框架，通过对象级语义标注和物理碰撞建模，实现语义与物理对齐的可执行3D环境。构建了InteriorGS数据集与SAGE-Bench基准，显著提升导航性能。**

- **链接: [http://arxiv.org/pdf/2510.21307v1](http://arxiv.org/pdf/2510.21307v1)**

> **作者:** Bingchen Miao; Rong Wei; Zhiqi Ge; Xiaoquan sun; Shiqi Gao; Jingzhe Zhu; Renhan Wang; Siliang Tang; Jun Xiao; Rui Tang; Juncheng Li
>
> **备注:** Download link of InteriorGS: https://huggingface.co/datasets/spatialverse/InteriorGS
>
> **摘要:** 3D Gaussian Splatting (3DGS), a 3D representation method with photorealistic real-time rendering capabilities, is regarded as an effective tool for narrowing the sim-to-real gap. However, it lacks fine-grained semantics and physical executability for Visual-Language Navigation (VLN). To address this, we propose SAGE-3D (Semantically and Physically Aligned Gaussian Environments for 3D Navigation), a new paradigm that upgrades 3DGS into an executable, semantically and physically aligned environment. It comprises two components: (1) Object-Centric Semantic Grounding, which adds object-level fine-grained annotations to 3DGS; and (2) Physics-Aware Execution Jointing, which embeds collision objects into 3DGS and constructs rich physical interfaces. We release InteriorGS, containing 1K object-annotated 3DGS indoor scene data, and introduce SAGE-Bench, the first 3DGS-based VLN benchmark with 2M VLN data. Experiments show that 3DGS scene data is more difficult to converge, while exhibiting strong generalizability, improving baseline performance by 31% on the VLN-CE Unseen task. The data and code will be available soon.
>
---
#### [new 067] Thermal Polarimetric Multi-view Stereo
- **分类: cs.CV**

- **简介: 该论文提出一种基于热偏振的多视角三维重建方法，旨在克服光照与材质依赖问题。通过建立通用偏振观测理论，利用长波红外偏振成像消除可见光偏振分析的歧义，实现对透明、半透明及非均质物体的精细三维形状重建，显著优于现有技术。**

- **链接: [http://arxiv.org/pdf/2510.20972v1](http://arxiv.org/pdf/2510.20972v1)**

> **作者:** Takahiro Kushida; Kenichiro Tanaka
>
> **备注:** ICCV 2025
>
> **摘要:** This paper introduces a novel method for detailed 3D shape reconstruction utilizing thermal polarization cues. Unlike state-of-the-art methods, the proposed approach is independent of illumination and material properties. In this paper, we formulate a general theory of polarization observation and show that long-wave infrared (LWIR) polarimetric imaging is free from the ambiguities that affect visible polarization analyses. Subsequently, we propose a method for recovering detailed 3D shapes using multi-view thermal polarimetric images. Experimental results demonstrate that our approach effectively reconstructs fine details in transparent, translucent, and heterogeneous objects, outperforming existing techniques.
>
---
#### [new 068] Morphologically Intelligent Perturbation Prediction with FORM
- **分类: cs.CV**

- **简介: 该论文提出FORM框架，解决细胞三维形态在扰动下的预测难题。针对现有模型局限于二维表示的问题，构建基于多通道VQGAN的形态编码器与扩散轨迹模块，实现3D细胞结构的生成与条件模拟，并引入MorphoEval评估体系，推动高分辨率虚拟细胞建模。**

- **链接: [http://arxiv.org/pdf/2510.21337v1](http://arxiv.org/pdf/2510.21337v1)**

> **作者:** Reed Naidoo; Matt De Vries; Olga Fourkioti; Vicky Bousgouni; Mar Arias-Garcia; Maria Portillo-Malumbres; Chris Bakal
>
> **摘要:** Understanding how cells respond to external stimuli is a central challenge in biomedical research and drug development. Current computational frameworks for modelling cellular responses remain restricted to two-dimensional representations, limiting their capacity to capture the complexity of cell morphology under perturbation. This dimensional constraint poses a critical bottleneck for the development of accurate virtual cell models. Here, we present FORM, a machine learning framework for predicting perturbation-induced changes in three-dimensional cellular structure. FORM consists of two components: a morphology encoder, trained end-to-end via a novel multi-channel VQGAN to learn compact 3D representations of cell shape, and a diffusion-based perturbation trajectory module that captures how morphology evolves across perturbation conditions. Trained on a large-scale dataset of over 65,000 multi-fluorescence 3D cell volumes spanning diverse chemical and genetic perturbations, FORM supports both unconditional morphology synthesis and conditional simulation of perturbed cell states. Beyond generation, FORM can predict downstream signalling activity, simulate combinatorial perturbation effects, and model morphodynamic transitions between states of unseen perturbations. To evaluate performance, we introduce MorphoEval, a benchmarking suite that quantifies perturbation-induced morphological changes in structural, statistical, and biological dimensions. Together, FORM and MorphoEval work toward the realisation of the 3D virtual cell by linking morphology, perturbation, and function through high-resolution predictive simulation.
>
---
#### [new 069] Improved Training Technique for Shortcut Models
- **分类: cs.CV**

- **简介: 该论文针对生成模型中的快捷模型（shortcut models）存在的五大问题：引导累积缺陷、引导僵化、频率偏差、自一致性冲突和曲线生成轨迹。提出iSM统一框架，通过内在引导、多级小波损失、缩放最优传输和双EMA策略，显著提升生成质量与稳定性，在ImageNet上实现更优的FID得分。**

- **链接: [http://arxiv.org/pdf/2510.21250v1](http://arxiv.org/pdf/2510.21250v1)**

> **作者:** Anh Nguyen; Viet Nguyen; Duc Vu; Trung Dao; Chi Tran; Toan Tran; Anh Tran
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Shortcut models represent a promising, non-adversarial paradigm for generative modeling, uniquely supporting one-step, few-step, and multi-step sampling from a single trained network. However, their widespread adoption has been stymied by critical performance bottlenecks. This paper tackles the five core issues that held shortcut models back: (1) the hidden flaw of compounding guidance, which we are the first to formalize, causing severe image artifacts; (2) inflexible fixed guidance that restricts inference-time control; (3) a pervasive frequency bias driven by a reliance on low-level distances in the direct domain, which biases reconstructions toward low frequencies; (4) divergent self-consistency arising from a conflict with EMA training; and (5) curvy flow trajectories that impede convergence. To address these challenges, we introduce iSM, a unified training framework that systematically resolves each limitation. Our framework is built on four key improvements: Intrinsic Guidance provides explicit, dynamic control over guidance strength, resolving both compounding guidance and inflexibility. A Multi-Level Wavelet Loss mitigates frequency bias to restore high-frequency details. Scaling Optimal Transport (sOT) reduces training variance and learns straighter, more stable generative paths. Finally, a Twin EMA strategy reconciles training stability with self-consistency. Extensive experiments on ImageNet 256 x 256 demonstrate that our approach yields substantial FID improvements over baseline shortcut models across one-step, few-step, and multi-step generation, making shortcut models a viable and competitive class of generative models.
>
---
#### [new 070] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence
- **分类: cs.CV**

- **简介: 该论文提出多模态未剪辑视频检索任务与基准MUVR，解决长视频平台中基于多模态查询的精准检索问题。构建包含六级视觉对应关系的53K视频数据集，支持文本、标签、掩码等多模态查询，评估模型在未剪辑视频中的匹配与重排序能力。**

- **链接: [http://arxiv.org/pdf/2510.21406v1](http://arxiv.org/pdf/2510.21406v1)**

> **作者:** Yue Feng; Jinwei Hu; Qijia Lu; Jiawei Niu; Li Tan; Shuo Yuan; Ziyi Yan; Yizhen Jia; Qingzhi He; Shiping Ge; Ethan Q. Chen; Wentong Li; Limin Wang; Jie Qin
>
> **备注:** Accepted to NeurIPS 2025 D&B Track
>
> **摘要:** We propose the Multi-modal Untrimmed Video Retrieval task, along with a new benchmark (MUVR) to advance video retrieval for long-video platforms. MUVR aims to retrieve untrimmed videos containing relevant segments using multi-modal queries. It has the following features: 1) Practical retrieval paradigm: MUVR supports video-centric multi-modal queries, expressing fine-grained retrieval needs through long text descriptions, video tag prompts, and mask prompts. It adopts a one-to-many retrieval paradigm and focuses on untrimmed videos, tailored for long-video platform applications. 2) Multi-level visual correspondence: To cover common video categories (e.g., news, travel, dance) and precisely define retrieval matching criteria, we construct multi-level visual correspondence based on core video content (e.g., news events, travel locations, dance moves) which users are interested in and want to retrieve. It covers six levels: copy, event, scene, instance, action, and others. 3) Comprehensive evaluation criteria: We develop 3 versions of MUVR (i.e., Base, Filter, QA). MUVR-Base/Filter evaluates retrieval models, while MUVR-QA assesses MLLMs in a question-answering format. We also propose a Reranking Score to evaluate the reranking ability of MLLMs. MUVR consists of 53K untrimmed videos from the video platform Bilibili, with 1,050 multi-modal queries and 84K matches. Extensive evaluations of 3 state-of-the-art video retrieval models, 6 image-based VLMs, and 10 MLLMs are conducted. MUVR reveals the limitations of retrieval methods in processing untrimmed videos and multi-modal queries, as well as MLLMs in multi-video understanding and reranking. Our code and benchmark is available at https://github.com/debby-0527/MUVR.
>
---
#### [new 071] BADiff: Bandwidth Adaptive Diffusion Model
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出BADiff，一种基于实时带宽自适应的扩散模型。针对传统模型在低带宽下因固定步数生成导致质量损失与计算浪费的问题，通过条件化质量嵌入实现端到端训练，使模型能根据带宽动态调整去噪过程，支持早期停止并保持感知质量。**

- **链接: [http://arxiv.org/pdf/2510.21366v1](http://arxiv.org/pdf/2510.21366v1)**

> **作者:** Xi Zhang; Hanwei Zhu; Yan Zhong; Jiamang Wang; Weisi Lin
>
> **备注:** NeurIPS 2025 Poster
>
> **摘要:** In this work, we propose a novel framework to enable diffusion models to adapt their generation quality based on real-time network bandwidth constraints. Traditional diffusion models produce high-fidelity images by performing a fixed number of denoising steps, regardless of downstream transmission limitations. However, in practical cloud-to-device scenarios, limited bandwidth often necessitates heavy compression, leading to loss of fine textures and wasted computation. To address this, we introduce a joint end-to-end training strategy where the diffusion model is conditioned on a target quality level derived from the available bandwidth. During training, the model learns to adaptively modulate the denoising process, enabling early-stop sampling that maintains perceptual quality appropriate to the target transmission condition. Our method requires minimal architectural changes and leverages a lightweight quality embedding to guide the denoising trajectory. Experimental results demonstrate that our approach significantly improves the visual fidelity of bandwidth-adapted generations compared to naive early-stopping, offering a promising solution for efficient image delivery in bandwidth-constrained environments. Code is available at: https://github.com/xzhang9308/BADiff.
>
---
#### [new 072] Digital Contrast CT Pulmonary Angiography Synthesis from Non-contrast CT for Pulmonary Vascular Disease
- **分类: cs.CV**

- **简介: 该论文提出一种基于CycleGAN的级联生成器，从非增强CT（NCCT）合成数字对比增强CT肺血管造影（DCCTPA），以解决传统CTPA依赖对比剂带来的肾毒性与过敏风险。通过410对数据训练与验证，模型在结构保持与血管增强上表现优异，并显著提升肺血管分割与量化精度。**

- **链接: [http://arxiv.org/pdf/2510.21140v1](http://arxiv.org/pdf/2510.21140v1)**

> **作者:** Ying Ming; Yue Lin; Longfei Zhao; Gengwan Li; Zuopeng Tan; Bing Li; Sheng Xie; Wei Song; Qiqi Xu
>
> **摘要:** Computed Tomography Pulmonary Angiography (CTPA) is the reference standard for diagnosing pulmonary vascular diseases such as Pulmonary Embolism (PE) and Chronic Thromboembolic Pulmonary Hypertension (CTEPH). However, its reliance on iodinated contrast agents poses risks including nephrotoxicity and allergic reactions, particularly in high-risk patients. This study proposes a method to generate Digital Contrast CTPA (DCCTPA) from Non-Contrast CT (NCCT) scans using a cascaded synthesizer based on Cycle-Consistent Generative Adversarial Networks (CycleGAN). Totally retrospective 410 paired CTPA and NCCT scans were obtained from three centers. The model was trained and validated internally on 249 paired images. Extra dataset that comprising 161 paired images was as test set for model generalization evaluation and downstream clinical tasks validation. Compared with state-of-the-art (SOTA) methods, the proposed method achieved the best comprehensive performance by evaluating quantitative metrics (For validation, MAE: 156.28, PSNR: 20.71 and SSIM: 0.98; For test, MAE: 165.12, PSNR: 20.27 and SSIM: 0.98) and qualitative visualization, demonstrating valid vessel enhancement, superior image fidelity and structural preservation. The approach was further applied to downstream tasks of pulmonary vessel segmentation and vascular quantification. On the test set, the average Dice, clDice, and clRecall of artery and vein pulmonary segmentation was 0.70, 0.71, 0.73 and 0.70, 0.72, 0.75 respectively, all markedly improved compared with NCCT inputs.\@ Inter-class Correlation Coefficient (ICC) for vessel volume between DCCTPA and CTPA was significantly better than that between NCCT and CTPA (Average ICC : 0.81 vs 0.70), indicating effective vascular enhancement in DCCTPA, especially for small vessels.
>
---
#### [new 073] Lightweight Classifier for Detecting Intracranial Hemorrhage in Ultrasound Data
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对创伤性脑损伤后颅内出血的快速检测问题，提出基于超声组织脉动成像的轻量级分类模型。通过PCA降维与多种机器学习算法对比，实现98.0%准确率，验证了便携式超声在资源受限场景下辅助诊断的可行性。**

- **链接: [http://arxiv.org/pdf/2510.20857v1](http://arxiv.org/pdf/2510.20857v1)**

> **作者:** Phat Tran; Enbai Kuang; Fred Xu
>
> **摘要:** Intracranial hemorrhage (ICH) secondary to Traumatic Brain Injury (TBI) represents a critical diagnostic challenge, with approximately 64,000 TBI-related deaths annually in the United States. Current diagnostic modalities including Computed Tomography (CT) and Magnetic Resonance Imaging (MRI) have significant limitations: high cost, limited availability, and infrastructure dependence, particularly in resource-constrained environments. This study investigates machine learning approaches for automated ICH detection using Ultrasound Tissue Pulsatility Imaging (TPI), a portable technique measuring tissue displacement from hemodynamic forces during cardiac cycles. We analyze ultrasound TPI signals comprising 30 temporal frames per cardiac cycle with recording angle information, collected from TBI patients with CT-confirmed ground truth labels. Our preprocessing pipeline employs z-score normalization and Principal Component Analysis (PCA) for dimensionality reduction, retaining components explaining 95% of cumulative variance. We systematically evaluate multiple classification algorithms spanning probabilistic, kernel-based, neural network, and ensemble learning approaches across three feature representations: original 31-dimensional space, reduced subset, and PCA-transformed space. Results demonstrate that PCA transformation substantially improves classifier performance, with ensemble methods achieving 98.0% accuracy and F1-score of 0.890, effectively balancing precision and recall despite class imbalance. These findings establish the feasibility of machine learning-based ICH detection in TBI patients using portable ultrasound devices, with applications in emergency medicine, rural healthcare, and military settings where traditional imaging is unavailable.
>
---
#### [new 074] Physics-Informed Deep Learning for Improved Input Function Estimation in Motion-Blurred Dynamic [${}^{18}$F]FDG PET Images
- **分类: q-bio.QM; cs.CV**

- **简介: 该论文针对小鼠[18F]FDG动态PET图像中因运动导致的模糊问题，提出一种物理信息深度学习模型（PIDLIF），用于直接从图像预测动脉输入函数（AIF）。通过引入基于两室模型的生理约束损失，提升模型在严重图像退化下的鲁棒性，实现更准确、非侵入式的AIF估计。**

- **链接: [http://arxiv.org/pdf/2510.21281v1](http://arxiv.org/pdf/2510.21281v1)**

> **作者:** Christian Salomonsen; Kristoffer K. Wickstrøm; Samuel Kuttner; Elisabeth Wetzer
>
> **备注:** 12 pages, 4 figures, 1 table. Preprint: Accepted to PRIME @ MICCAI 2025. This is the submitted (pre-review) version (url: https://openreview.net/forum?id=twg1nba5ep)
>
> **摘要:** Kinetic modeling enables \textit{in vivo} quantification of tracer uptake and glucose metabolism in [${}^{18}$F]Fluorodeoxyglucose ([${}^{18}$F]FDG) dynamic positron emission tomography (dPET) imaging of mice. However, kinetic modeling requires the accurate determination of the arterial input function (AIF) during imaging, which is time-consuming and invasive. Recent studies have shown the efficacy of using deep learning to directly predict the input function, surpassing established methods such as the image-derived input function (IDIF). In this work, we trained a physics-informed deep learning-based input function prediction model (PIDLIF) to estimate the AIF directly from the PET images, incorporating a kinetic modeling loss during training. The proposed method uses a two-tissue compartment model over two regions, the myocardium and brain of the mice, and is trained on a dataset of 70 [${}^{18}$F]FDG dPET images of mice accompanied by the measured AIF during imaging. The proposed method had comparable performance to the network without a physics-informed loss, and when sudden movement causing blurring in the images was simulated, the PIDLIF model maintained high performance in severe cases of image degradation. The proposed physics-informed method exhibits an improved robustness that is promoted by physically constraining the problem, enforcing consistency for out-of-distribution samples. In conclusion, the PIDLIF model offers insight into the effects of leveraging physiological distribution mechanics in mice to guide a deep learning-based AIF prediction network in images with severe degradation as a result of blurring due to movement during imaging.
>
---
#### [new 075] REMONI: An Autonomous System Integrating Wearables and Multimodal Large Language Models for Enhanced Remote Health Monitoring
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出REMONI系统，融合可穿戴设备、IoT与多模态大语言模型，实现远程健康监测。针对传统监测系统人机交互不足的问题，系统自动采集生理与视觉数据，通过异常检测与自然语言处理，实时分析患者状态与情绪，并支持医护人员通过自然语言交互获取信息，提升监测效率与医疗体验。**

- **链接: [http://arxiv.org/pdf/2510.21445v1](http://arxiv.org/pdf/2510.21445v1)**

> **作者:** Thanh Cong Ho; Farah Kharrat; Abderrazek Abid; Fakhri Karray
>
> **摘要:** With the widespread adoption of wearable devices in our daily lives, the demand and appeal for remote patient monitoring have significantly increased. Most research in this field has concentrated on collecting sensor data, visualizing it, and analyzing it to detect anomalies in specific diseases such as diabetes, heart disease and depression. However, this domain has a notable gap in the aspect of human-machine interaction. This paper proposes REMONI, an autonomous REmote health MONItoring system that integrates multimodal large language models (MLLMs), the Internet of Things (IoT), and wearable devices. The system automatically and continuously collects vital signs, accelerometer data from a special wearable (such as a smartwatch), and visual data in patient video clips collected from cameras. This data is processed by an anomaly detection module, which includes a fall detection model and algorithms to identify and alert caregivers of the patient's emergency conditions. A distinctive feature of our proposed system is the natural language processing component, developed with MLLMs capable of detecting and recognizing a patient's activity and emotion while responding to healthcare worker's inquiries. Additionally, prompt engineering is employed to integrate all patient information seamlessly. As a result, doctors and nurses can access real-time vital signs and the patient's current state and mood by interacting with an intelligent agent through a user-friendly web application. Our experiments demonstrate that our system is implementable and scalable for real-life scenarios, potentially reducing the workload of medical professionals and healthcare costs. A full-fledged prototype illustrating the functionalities of the system has been developed and being tested to demonstrate the robustness of its various capabilities.
>
---
#### [new 076] Sparser Block-Sparse Attention via Token Permutation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对大语言模型长序列处理中的计算效率问题，提出基于令牌置换的稀疏块注意力机制（PBS-Attn）。通过优化注意力模式的块级稀疏性，提升预填充阶段的计算效率，实现高达2.75倍的加速，同时保持高模型精度。**

- **链接: [http://arxiv.org/pdf/2510.21270v1](http://arxiv.org/pdf/2510.21270v1)**

> **作者:** Xinghao Wang; Pengyu Wang; Dong Zhang; Chenkun Tan; Shaojun Zhou; Zhaoxiang Liu; Shiguo Lian; Fangxu Liu; Kai Song; Xipeng Qiu
>
> **摘要:** Scaling the context length of large language models (LLMs) offers significant benefits but is computationally expensive. This expense stems primarily from the self-attention mechanism, whose $O(N^2)$ complexity with respect to sequence length presents a major bottleneck for both memory and latency. Fortunately, the attention matrix is often sparse, particularly for long sequences, suggesting an opportunity for optimization. Block-sparse attention has emerged as a promising solution that partitions sequences into blocks and skips computation for a subset of these blocks. However, the effectiveness of this method is highly dependent on the underlying attention patterns, which can lead to sub-optimal block-level sparsity. For instance, important key tokens for queries within a single block may be scattered across numerous other blocks, leading to computational redundancy. In this work, we propose Permuted Block-Sparse Attention (\textbf{PBS-Attn}), a plug-and-play method that leverages the permutation properties of attention to increase block-level sparsity and enhance the computational efficiency of LLM prefilling. We conduct comprehensive experiments on challenging real-world long-context datasets, demonstrating that PBS-Attn consistently outperforms existing block-sparse attention methods in model accuracy and closely matches the full attention baseline. Powered by our custom permuted-FlashAttention kernels, PBS-Attn achieves an end-to-end speedup of up to $2.75\times$ in long-context prefilling, confirming its practical viability. Code available at https://github.com/xinghaow99/pbs-attn
>
---
#### [new 077] Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文面向机器人灵巧操作任务，提出基于真实人类活动视频的视觉-语言-动作（VLA）模型预训练方法。通过自动化分析无标注的视角视频，生成带3D手部运动与语言描述的高质量数据，构建100万条任务数据集。模型在零样本和微调后均表现出色，显著提升机器人对新物体的泛化能力，验证了大规模预训练的有效性。**

- **链接: [http://arxiv.org/pdf/2510.21571v1](http://arxiv.org/pdf/2510.21571v1)**

> **作者:** Qixiu Li; Yu Deng; Yaobo Liang; Lin Luo; Lei Zhou; Chengtang Yao; Lingqi Zeng; Zhiyuan Feng; Huizhi Liang; Sicheng Xu; Yizhong Zhang; Xi Chen; Hao Chen; Lily Sun; Dong Chen; Jiaolong Yang; Baining Guo
>
> **备注:** Project page: https://microsoft.github.io/VITRA/
>
> **摘要:** This paper presents a novel approach for pretraining robotic manipulation Vision-Language-Action (VLA) models using a large corpus of unscripted real-life video recordings of human hand activities. Treating human hand as dexterous robot end-effector, we show that "in-the-wild" egocentric human videos without any annotations can be transformed into data formats fully aligned with existing robotic V-L-A training data in terms of task granularity and labels. This is achieved by the development of a fully-automated holistic human activity analysis approach for arbitrary human hand videos. This approach can generate atomic-level hand activity segments and their language descriptions, each accompanied with framewise 3D hand motion and camera motion. We process a large volume of egocentric videos and create a hand-VLA training dataset containing 1M episodes and 26M frames. This training data covers a wide range of objects and concepts, dexterous manipulation tasks, and environment variations in real life, vastly exceeding the coverage of existing robot data. We design a dexterous hand VLA model architecture and pretrain the model on this dataset. The model exhibits strong zero-shot capabilities on completely unseen real-world observations. Additionally, fine-tuning it on a small amount of real robot action data significantly improves task success rates and generalization to novel objects in real robotic experiments. We also demonstrate the appealing scaling behavior of the model's task performance with respect to pretraining data scale. We believe this work lays a solid foundation for scalable VLA pretraining, advancing robots toward truly generalizable embodied intelligence.
>
---
#### [new 078] Disentangled Representation Learning via Modular Compositional Bias
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于表示学习任务，旨在解决现有解耦表示学习方法因依赖特定目标或架构而难以适应新因子的问题。提出一种模块化组合偏差，通过因子特异性混合策略与双目标优化，实现属性、对象及二者的联合解耦，无需修改模型结构或目标函数。**

- **链接: [http://arxiv.org/pdf/2510.21402v1](http://arxiv.org/pdf/2510.21402v1)**

> **作者:** Whie Jung; Dong Hoon Lee; Seunghoon Hong
>
> **摘要:** Recent disentangled representation learning (DRL) methods heavily rely on factor specific strategies-either learning objectives for attributes or model architectures for objects-to embed inductive biases. Such divergent approaches result in significant overhead when novel factors of variation do not align with prior assumptions, such as statistical independence or spatial exclusivity, or when multiple factors coexist, as practitioners must redesign architectures or objectives. To address this, we propose a compositional bias, a modular inductive bias decoupled from both objectives and architectures. Our key insight is that different factors obey distinct recombination rules in the data distribution: global attributes are mutually exclusive, e.g., a face has one nose, while objects share a common support (any subset of objects can co-exist). We therefore randomly remix latents according to factor-specific rules, i.e., a mixing strategy, and force the encoder to discover whichever factor structure the mixing strategy reflects through two complementary objectives: (i) a prior loss that ensures every remix decodes into a realistic image, and (ii) the compositional consistency loss introduced by Wiedemer et al. (arXiv:2310.05327), which aligns each composite image with its corresponding composite latent. Under this general framework, simply adjusting the mixing strategy enables disentanglement of attributes, objects, and even both, without modifying the objectives or architectures. Extensive experiments demonstrate that our method shows competitive performance in both attribute and object disentanglement, and uniquely achieves joint disentanglement of global style and objects. Code is available at https://github.com/whieya/Compositional-DRL.
>
---
#### [new 079] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文针对文本生成图像模型中存在的性别、种族等社会偏见问题，提出一种无需重训练的后处理去偏框架FairImagen。通过公平主成分分析投影提示嵌入，减少群体特异性信息，同时保持语义一致性，实现多维度去偏，显著提升公平性且适度保留图像质量与提示保真度。**

- **链接: [http://arxiv.org/pdf/2510.21363v1](http://arxiv.org/pdf/2510.21363v1)**

> **作者:** Zihao Fu; Ryan Brown; Shun Shao; Kai Rawal; Eoin Delaney; Chris Russell
>
> **备注:** Neurips 2025
>
> **摘要:** Text-to-image diffusion models, such as Stable Diffusion, have demonstrated remarkable capabilities in generating high-quality and diverse images from natural language prompts. However, recent studies reveal that these models often replicate and amplify societal biases, particularly along demographic attributes like gender and race. In this paper, we introduce FairImagen (https://github.com/fuzihaofzh/FairImagen), a post-hoc debiasing framework that operates on prompt embeddings to mitigate such biases without retraining or modifying the underlying diffusion model. Our method integrates Fair Principal Component Analysis to project CLIP-based input embeddings into a subspace that minimizes group-specific information while preserving semantic content. We further enhance debiasing effectiveness through empirical noise injection and propose a unified cross-demographic projection method that enables simultaneous debiasing across multiple demographic attributes. Extensive experiments across gender, race, and intersectional settings demonstrate that FairImagen significantly improves fairness with a moderate trade-off in image quality and prompt fidelity. Our framework outperforms existing post-hoc methods and offers a simple, scalable, and model-agnostic solution for equitable text-to-image generation.
>
---
#### [new 080] This EEG Looks Like These EEGs: Interpretable Interictal Epileptiform Discharge Detection With ProtoEEG-kNN
- **分类: q-bio.NC; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对癫痫间期痫样放电（IED）检测任务，解决现有机器学习模型不可解释的问题。提出ProtoEEG-kNN模型，基于原型的kNN方法，通过对比待测EEG与训练集中相似样本，提供形态与空间分布双重可视化解释，实现高精度且可解释的IED检测。**

- **链接: [http://arxiv.org/pdf/2510.20846v1](http://arxiv.org/pdf/2510.20846v1)**

> **作者:** Dennis Tang; Jon Donnelly; Alina Jade Barnett; Lesia Semenova; Jin Jing; Peter Hadar; Ioannis Karakis; Olga Selioutski; Kehan Zhao; M. Brandon Westover; Cynthia Rudin
>
> **备注:** MICCAI 2025
>
> **摘要:** The presence of interictal epileptiform discharges (IEDs) in electroencephalogram (EEG) recordings is a critical biomarker of epilepsy. Even trained neurologists find detecting IEDs difficult, leading many practitioners to turn to machine learning for help. While existing machine learning algorithms can achieve strong accuracy on this task, most models are uninterpretable and cannot justify their conclusions. Absent the ability to understand model reasoning, doctors cannot leverage their expertise to identify incorrect model predictions and intervene accordingly. To improve the human-model interaction, we introduce ProtoEEG-kNN, an inherently interpretable model that follows a simple case-based reasoning process. ProtoEEG-kNN reasons by comparing an EEG to similar EEGs from the training set and visually demonstrates its reasoning both in terms of IED morphology (shape) and spatial distribution (location). We show that ProtoEEG-kNN can achieve state-of-the-art accuracy in IED detection while providing explanations that experts prefer over existing approaches.
>
---
#### [new 081] More Than Memory Savings: Zeroth-Order Optimization Mitigates Forgetting in Continual Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究持续学习中的遗忘问题，提出零阶优化（ZO）可缓解遗忘并提升稳定性，但牺牲塑性。为此，作者设计ZO-FC方法，仅对适配器模块使用ZO优化，保留分类器的梯度更新，实现稳定与适应性的平衡，显著降低内存开销，适用于设备端持续学习。**

- **链接: [http://arxiv.org/pdf/2510.21019v1](http://arxiv.org/pdf/2510.21019v1)**

> **作者:** Wanhao Yu; Zheng Wang; Shuteng Niu; Sen Lin; Li Yang
>
> **摘要:** Zeroth-order (ZO) optimization has gained attention as a memory-efficient alternative to first-order (FO) methods, particularly in settings where gradient computation is expensive or even impractical. Beyond its memory efficiency, in this work, we investigate ZO optimization for continual learning (CL) as a novel approach to address the plasticity-stability-efficiency trilemma. Through theoretical analysis and empirical evidence, we show that ZO optimization naturally leads to flatter loss landscapes, which in turn reduce forgetting in CL. However, this stability comes at a cost of plasticity: due to its imprecise gradient estimates and slower convergence, ZO optimization tends to be less effective than FO in acquiring new task-specific knowledge, particularly under constrained training budgets. To better understand this trade-off, we conduct a holistic evaluation of ZO optimization applied to various existing CL methods. Our findings reveal that ZO optimization enhances stability but often undermines plasticity, particularly when used with learnable classifiers. Motivated by this insight, we propose ZO-FC, a simple but effective approach that applies ZO optimization to a single adapter-based PEFT module with FO optimized classifier. This design leverages the stability benefits of ZO while preserving the adaptability of FO updates with negligible memory overhead. Experiments demonstrate that ZO-FC achieves an effective balance between stability and plasticity, offering a practical and memory-efficient solution for on-device CL.
>
---
#### [new 082] Buffer layers for Test-Time Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对测试时适应（TTA）中依赖归一化层导致的稳定性差与遗忘问题，提出基于缓冲层（Buffer layer）的新范式。通过引入模块化缓冲层，在不修改预训练模型的前提下实现稳定适应，有效缓解域偏移并防止灾难性遗忘，可无缝集成至多数现有TTA框架，显著提升性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21271v1](http://arxiv.org/pdf/2510.21271v1)**

> **作者:** Hyeongyu Kim; Geonhui Han; Dosik Hwang
>
> **备注:** NeurIPS 2025
>
> **摘要:** In recent advancements in Test Time Adaptation (TTA), most existing methodologies focus on updating normalization layers to adapt to the test domain. However, the reliance on normalization-based adaptation presents key challenges. First, normalization layers such as Batch Normalization (BN) are highly sensitive to small batch sizes, leading to unstable and inaccurate statistics. Moreover, normalization-based adaptation is inherently constrained by the structure of the pre-trained model, as it relies on training-time statistics that may not generalize well to unseen domains. These issues limit the effectiveness of normalization-based TTA approaches, especially under significant domain shift. In this paper, we introduce a novel paradigm based on the concept of a Buffer layer, which addresses the fundamental limitations of normalization layer updates. Unlike existing methods that modify the core parameters of the model, our approach preserves the integrity of the pre-trained backbone, inherently mitigating the risk of catastrophic forgetting during online adaptation. Through comprehensive experimentation, we demonstrate that our approach not only outperforms traditional methods in mitigating domain shift and enhancing model robustness, but also exhibits strong resilience to forgetting. Furthermore, our Buffer layer is modular and can be seamlessly integrated into nearly all existing TTA frameworks, resulting in consistent performance improvements across various architectures. These findings validate the effectiveness and versatility of the proposed solution in real-world domain adaptation scenarios. The code is available at https://github.com/hyeongyu-kim/Buffer_TTA.
>
---
#### [new 083] Vision Language Models for Dynamic Human Activity Recognition in Healthcare Settings
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文聚焦于医疗场景下动态人类活动识别任务，针对传统深度学习模型局限及视觉语言模型（VLM）输出难评估的问题，构建了描述性字幕数据集并提出综合评估方法。实验表明，VLM在准确率上可媲美甚至超越主流模型，为智能医疗系统中VLM的应用提供了有力基准。**

- **链接: [http://arxiv.org/pdf/2510.21424v1](http://arxiv.org/pdf/2510.21424v1)**

> **作者:** Abderrazek Abid; Thanh-Cong Ho; Fakhri Karray
>
> **摘要:** As generative AI continues to evolve, Vision Language Models (VLMs) have emerged as promising tools in various healthcare applications. One area that remains relatively underexplored is their use in human activity recognition (HAR) for remote health monitoring. VLMs offer notable strengths, including greater flexibility and the ability to overcome some of the constraints of traditional deep learning models. However, a key challenge in applying VLMs to HAR lies in the difficulty of evaluating their dynamic and often non-deterministic outputs. To address this gap, we introduce a descriptive caption data set and propose comprehensive evaluation methods to evaluate VLMs in HAR. Through comparative experiments with state-of-the-art deep learning models, our findings demonstrate that VLMs achieve comparable performance and, in some cases, even surpass conventional approaches in terms of accuracy. This work contributes a strong benchmark and opens new possibilities for the integration of VLMs into intelligent healthcare systems.
>
---
#### [new 084] Seed3D 1.0: From Images to High-Fidelity Simulation-Ready 3D Assets
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出Seed3D 1.0，一种从单张图像生成高保真、可直接用于物理引擎的3D资产的模型。针对传统方法在内容多样性与物理准确性间的权衡问题，该工作实现了高效、自动化的仿真资产生成，支持物体及场景级构建，推动了可扩展物理模拟环境的发展。**

- **链接: [http://arxiv.org/pdf/2510.19944v1](http://arxiv.org/pdf/2510.19944v1)**

> **作者:** Jiashi Feng; Xiu Li; Jing Lin; Jiahang Liu; Gaohong Liu; Weiqiang Lou; Su Ma; Guang Shi; Qinlong Wang; Jun Wang; Zhongcong Xu; Xuanyu Yi; Zihao Yu; Jianfeng Zhang; Yifan Zhu; Rui Chen; Jinxin Chi; Zixian Du; Li Han; Lixin Huang; Kaihua Jiang; Yuhan Li; Guan Luo; Shuguang Wang; Qianyi Wu; Fan Yang; Junyang Zhang; Xuanmeng Zhang
>
> **备注:** Seed3D 1.0 Technical Report; Official Page on https://seed.bytedance.com/seed3d
>
> **摘要:** Developing embodied AI agents requires scalable training environments that balance content diversity with physics accuracy. World simulators provide such environments but face distinct limitations: video-based methods generate diverse content but lack real-time physics feedback for interactive learning, while physics-based engines provide accurate dynamics but face scalability limitations from costly manual asset creation. We present Seed3D 1.0, a foundation model that generates simulation-ready 3D assets from single images, addressing the scalability challenge while maintaining physics rigor. Unlike existing 3D generation models, our system produces assets with accurate geometry, well-aligned textures, and realistic physically-based materials. These assets can be directly integrated into physics engines with minimal configuration, enabling deployment in robotic manipulation and simulation training. Beyond individual objects, the system scales to complete scene generation through assembling objects into coherent environments. By enabling scalable simulation-ready content creation, Seed3D 1.0 provides a foundation for advancing physics-based world simulators. Seed3D 1.0 is now available on https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?modelId=doubao-seed3d-1-0-250928&tab=Gen3D
>
---
#### [new 085] An Experimental Study of Trojan Vulnerabilities in UAV Autonomous Landing
- **分类: cs.CR; cs.AI; cs.CV; cs.RO**

- **简介: 该论文研究无人机自主着陆系统中基于深度学习模型的后门攻击漏洞。针对卷积神经网络，通过在训练数据中嵌入隐蔽触发器，导致模型在特定条件下失效。作者构建了定制数据集与评估框架，验证了攻击有效性，并揭示了城市空中交通系统的安全风险，为提升系统鲁棒性提供基础。**

- **链接: [http://arxiv.org/pdf/2510.20932v1](http://arxiv.org/pdf/2510.20932v1)**

> **作者:** Reza Ahmari; Ahmad Mohammadi; Vahid Hemmati; Mohammed Mynuddin; Mahmoud Nabil Mahmoud; Parham Kebria; Abdollah Homaifar; Mehrdad Saif
>
> **备注:** 6 pages
>
> **摘要:** This study investigates the vulnerabilities of autonomous navigation and landing systems in Urban Air Mobility (UAM) vehicles. Specifically, it focuses on Trojan attacks that target deep learning models, such as Convolutional Neural Networks (CNNs). Trojan attacks work by embedding covert triggers within a model's training data. These triggers cause specific failures under certain conditions, while the model continues to perform normally in other situations. We assessed the vulnerability of Urban Autonomous Aerial Vehicles (UAAVs) using the DroNet framework. Our experiments showed a significant drop in accuracy, from 96.4% on clean data to 73.3% on data triggered by Trojan attacks. To conduct this study, we collected a custom dataset and trained models to simulate real-world conditions. We also developed an evaluation framework designed to identify Trojan-infected models. This work demonstrates the potential security risks posed by Trojan attacks and lays the groundwork for future research on enhancing the resilience of UAM systems.
>
---
#### [new 086] AURASeg: Attention Guided Upsampling with Residual Boundary-Assistive Refinement for Drivable-Area Segmentation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对自动驾驶与机器人导航中的可行驶区域分割任务，解决现有模型在边界精度与多尺度特征融合上的不足。提出AURASeg模型，结合残差边界精修模块与注意力引导的渐进上采样解码器，提升边缘识别准确率与整体分割性能，在保持实时性的同时显著优化mIoU与F1指标。**

- **链接: [http://arxiv.org/pdf/2510.21536v1](http://arxiv.org/pdf/2510.21536v1)**

> **作者:** Narendhiran Vijayakumar; Sridevi. M
>
> **备注:** 10 pages, 5 figures, 4 tables
>
> **摘要:** Free space ground segmentation is essential to navigate robots and autonomous vehicles, recognize drivable zones, and traverse efficiently. Fine-grained features remain challenging for existing segmentation models, particularly for robots in indoor and structured environments. These difficulties arise from ineffective multi-scale processing, suboptimal boundary refinement, and limited feature representation. In order to overcome these limitations, we propose Attention-Guided Upsampling with Residual Boundary-Assistive Refinement (AURASeg), a ground-plane semantic segmentation model that maintains high segmentation accuracy while improving border precision. Our method uses CSP-Darknet backbone by adding a Residual Border Refinement Module (RBRM) for accurate edge delineation and an Attention Progressive Upsampling Decoder (APUD) for strong feature integration. We also incorporate a lightweight Atrous Spatial Pyramid Pooling (ASPP-Lite) module to ensure multi-scale context extraction without compromising real-time performance. The proposed model beats benchmark segmentation architectures in mIoU and F1 metrics when tested on the Ground Mobile Robot Perception (GMRP) Dataset and a custom Gazebo indoor dataset. Our approach achieves an improvement in mean Intersection-over-Union (mIoU) of +1.26% and segmentation precision of +1.65% compared to state-of-the-art models. These results show that our technique is feasible for autonomous perception in both indoor and outdoor environments, enabling precise border refinement with minimal effect on inference speed.
>
---
#### [new 087] Eye-Tracking as a Tool to Quantify the Effects of CAD Display on Radiologists' Interpretation of Chest Radiographs
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在研究计算机辅助检测（CAD）中边界框（BB）显示对放射科医生读片行为的影响。通过眼动追踪技术，对比有无BB时的视觉搜索特征，发现BB延长解读时间、增加注视停留时间与视线路径长度，但缩短首次聚焦病变时间，验证了眼动追踪在量化阅读行为变化中的可行性。**

- **链接: [http://arxiv.org/pdf/2510.20864v1](http://arxiv.org/pdf/2510.20864v1)**

> **作者:** Daisuke Matsumoto; Tomohiro Kikuchi; Yusuke Takagi; Soichiro Kojima; Ryoma Kobayashi; Daiju Ueda; Kohei Yamamoto; Sho Kawabe; Harushi Mori
>
> **摘要:** Rationale and Objectives: Computer-aided detection systems for chest radiographs are widely used, and concurrent reader displays, such as bounding-box (BB) highlights, may influence the reading process. This pilot study used eye tracking to conduct a preliminary experiment to quantify which aspects of visual search were affected. Materials and Methods: We sampled 180 chest radiographs from the VinDR-CXR dataset: 120 with solitary pulmonary nodules or masses and 60 without. The BBs were configured to yield an overall display sensitivity and specificity of 80%. Three radiologists (with 11, 5, and 1 years of experience, respectively) interpreted each case twice - once with BBs visible and once without - after a washout of >= 2 weeks. Eye movements were recorded using an EyeTech VT3 Mini. Metrics included interpretation time, time to first fixation on the lesion, lesion dwell time, total gaze-path length, and lung-field coverage ratio. Outcomes were modeled using a linear mixed model, with reading condition as a fixed effect and case and reader as random intercepts. The primary analysis was restricted to true positives (n=96). Results: Concurrent BB display prolonged interpretation time by 4.9 s (p<0.001) and increased lesion dwell time by 1.3 s (p<0.001). Total gaze-path length increased by 2,076 pixels (p<0.001), and lung-field coverage ratio increased by 10.5% (p<0.001). Time to first fixation on the lesion was reduced by 1.3 s (p<0.001). Conclusion: Eye tracking captured measurable alterations in search behavior associated with concurrent BB displays during chest radiograph interpretation. These findings support the feasibility of this approach and highlight the need for larger studies to confirm effects and explore implications across modalities and clinical contexts.
>
---
#### [new 088] Efficient Meningioma Tumor Segmentation Using Ensemble Learning
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文针对脑膜瘤分割任务，旨在解决传统方法耗时且深度学习模型计算成本高的问题。提出基于三种网络架构的集成学习方法，在有限硬件下仅用20轮训练即实现高效准确分割，显著降低训练需求并提升性能。**

- **链接: [http://arxiv.org/pdf/2510.21040v1](http://arxiv.org/pdf/2510.21040v1)**

> **作者:** Mohammad Mahdi Danesh Pajouh; Sara Saeedi
>
> **备注:** 2nd Place Winner in the BraTS 2025 MICCAI Challenge (Task 2: Meningioma Tumor Segmentation)
>
> **摘要:** Meningiomas represent the most prevalent form of primary brain tumors, comprising nearly one-third of all diagnosed cases. Accurate delineation of these tumors from MRI scans is crucial for guiding treatment strategies, yet remains a challenging and time-consuming task in clinical practice. Recent developments in deep learning have accelerated progress in automated tumor segmentation; however, many advanced techniques are hindered by heavy computational demands and long training schedules, making them less accessible for researchers and clinicians working with limited hardware. In this work, we propose a novel ensemble-based segmentation approach that combines three distinct architectures: (1) a baseline SegResNet model, (2) an attention-augmented SegResNet with concatenative skip connections, and (3) a dual-decoder U-Net enhanced with attention-gated skip connections (DDUNet). The ensemble aims to leverage architectural diversity to improve robustness and accuracy while significantly reducing training demands. Each baseline model was trained for only 20 epochs and Evaluated on the BraTS-MEN 2025 dataset. The proposed ensemble model achieved competitive performance, with average Lesion-Wise Dice scores of 77.30%, 76.37% and 73.9% on test dataset for Enhancing Tumor (ET), Tumor Core (TC) and Whole Tumor (WT) respectively. These results highlight the effectiveness of ensemble learning for brain tumor segmentation, even under limited hardware constraints. Our proposed method provides a practical and accessible tool for aiding the diagnosis of meningioma, with potential impact in both clinical and research settings.
>
---
## 更新

#### [replaced 001] FORLA: Federated Object-centric Representation Learning with Slot Attention
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.02964v2](http://arxiv.org/pdf/2506.02964v2)**

> **作者:** Guiqiu Liao; Matjaz Jogan; Eric Eaton; Daniel A. Hashimoto
>
> **备注:** Accepted by Neurips2025
>
> **摘要:** Learning efficient visual representations across heterogeneous unlabeled datasets remains a central challenge in federated learning. Effective federated representations require features that are jointly informative across clients while disentangling domain-specific factors without supervision. We introduce FORLA, a novel framework for federated object-centric representation learning and feature adaptation across clients using unsupervised slot attention. At the core of our method is a shared feature adapter, trained collaboratively across clients to adapt features from foundation models, and a shared slot attention module that learns to reconstruct the adapted features. To optimize this adapter, we design a two-branch student-teacher architecture. In each client, a student decoder learns to reconstruct full features from foundation models, while a teacher decoder reconstructs their adapted, low-dimensional counterpart. The shared slot attention module bridges cross-domain learning by aligning object-level representations across clients. Experiments in multiple real-world datasets show that our framework not only outperforms centralized baselines on object discovery but also learns a compact, universal representation that generalizes well across domains. This work highlights federated slot attention as an effective tool for scalable, unsupervised visual representation learning from cross-domain data with distributed concepts.
>
---
#### [replaced 002] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18303v2](http://arxiv.org/pdf/2411.18303v2)**

> **作者:** Wenjie Zhuo; Fan Ma; Hehe Fan
>
> **摘要:** We present InfiniDreamer, a novel framework for arbitrarily long human motion generation. InfiniDreamer addresses the limitations of current motion generation methods, which are typically restricted to short sequences due to the lack of long motion training data. To achieve this, we first generate sub-motions corresponding to each textual description and then assemble them into a coarse, extended sequence using randomly initialized transition segments. We then introduce an optimization-based method called Segment Score Distillation (SSD) to refine the entire long motion sequence. SSD is designed to utilize an existing motion prior, which is trained only on short clips, in a training-free manner. Specifically, SSD iteratively refines overlapping short segments sampled from the coarsely extended long motion sequence, progressively aligning them with the pre-trained motion diffusion prior. This process ensures local coherence within each segment, while the refined transitions between segments maintain global consistency across the entire sequence. Extensive qualitative and quantitative experiments validate the superiority of our framework, showcasing its ability to generate coherent, contextually aware motion sequences of arbitrary length.
>
---
#### [replaced 003] Iterative Tool Usage Exploration for Multimodal Agents via Step-wise Preference Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21561v4](http://arxiv.org/pdf/2504.21561v4)**

> **作者:** Pengxiang Li; Zhi Gao; Bofei Zhang; Yapeng Mi; Xiaojian Ma; Chenrui Shi; Tao Yuan; Yuwei Wu; Yunde Jia; Song-Chun Zhu; Qing Li
>
> **备注:** 24 pages
>
> **摘要:** Multimodal agents, which integrate a controller e.g., a vision language model) with external tools, have demonstrated remarkable capabilities in tackling complex multimodal tasks. Existing approaches for training these agents, both supervised fine-tuning and reinforcement learning, depend on extensive human-annotated task-answer pairs and tool trajectories. However, for complex multimodal tasks, such annotations are prohibitively expensive or impractical to obtain. In this paper, we propose an iterative tool usage exploration method for multimodal agents without any pre-collected data, namely SPORT, via step-wise preference optimization to refine the trajectories of tool usage. Our method enables multimodal agents to autonomously discover effective tool usage strategies through self-exploration and optimization, eliminating the bottleneck of human annotation. SPORT has four iterative components: task synthesis, step sampling, step verification, and preference tuning. We first synthesize multimodal tasks using language models. Then, we introduce a novel trajectory exploration scheme, where step sampling and step verification are executed alternately to solve synthesized tasks. In step sampling, the agent tries different tools and obtains corresponding results. In step verification, we employ a verifier to provide AI feedback to construct step-wise preference data. The data is subsequently used to update the controller for tool usage through preference tuning, producing a SPORT agent. By interacting with real environments, the SPORT agent gradually evolves into a more refined and capable system. Evaluation in the GTA and GAIA benchmarks shows that the SPORT agent achieves 6.41% and 3.64% improvements, underscoring the generalization and effectiveness introduced by our method. The project page is https://SPORT-Agents.github.io.
>
---
#### [replaced 004] Rethinking Driving World Model as Synthetic Data Generator for Perception Tasks
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.19195v2](http://arxiv.org/pdf/2510.19195v2)**

> **作者:** Kai Zeng; Zhanqian Wu; Kaixin Xiong; Xiaobao Wei; Xiangyu Guo; Zhenxin Zhu; Kalok Ho; Lijun Zhou; Bohan Zeng; Ming Lu; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Wentao Zhang
>
> **摘要:** Recent advancements in driving world models enable controllable generation of high-quality RGB videos or multimodal videos. Existing methods primarily focus on metrics related to generation quality and controllability. However, they often overlook the evaluation of downstream perception tasks, which are $\mathbf{really\ crucial}$ for the performance of autonomous driving. Existing methods usually leverage a training strategy that first pretrains on synthetic data and finetunes on real data, resulting in twice the epochs compared to the baseline (real data only). When we double the epochs in the baseline, the benefit of synthetic data becomes negligible. To thoroughly demonstrate the benefit of synthetic data, we introduce Dream4Drive, a novel synthetic data generation framework designed for enhancing the downstream perception tasks. Dream4Drive first decomposes the input video into several 3D-aware guidance maps and subsequently renders the 3D assets onto these guidance maps. Finally, the driving world model is fine-tuned to produce the edited, multi-view photorealistic videos, which can be used to train the downstream perception models. Dream4Drive enables unprecedented flexibility in generating multi-view corner cases at scale, significantly boosting corner case perception in autonomous driving. To facilitate future research, we also contribute a large-scale 3D asset dataset named DriveObj3D, covering the typical categories in driving scenarios and enabling diverse 3D-aware video editing. We conduct comprehensive experiments to show that Dream4Drive can effectively boost the performance of downstream perception models under various training epochs. Page: https://wm-research.github.io/Dream4Drive/ GitHub Link: https://github.com/wm-research/Dream4Drive
>
---
#### [replaced 005] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17982v4](http://arxiv.org/pdf/2505.17982v4)**

> **作者:** Bryan Wong; Jong Woo Kim; Huazhu Fu; Mun Yong Yi
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Vision-language models (VLMs) have recently been integrated into multiple instance learning (MIL) frameworks to address the challenge of few-shot, weakly supervised classification of whole slide images (WSIs). A key trend involves leveraging multi-scale information to better represent hierarchical tissue structures. However, existing methods often face two key limitations: (1) insufficient modeling of interactions within the same modalities across scales (e.g., 5x and 20x) and (2) inadequate alignment between visual and textual modalities on the same scale. To address these gaps, we propose HiVE-MIL, a hierarchical vision-language framework that constructs a unified graph consisting of (1) parent-child links between coarse (5x) and fine (20x) visual/textual nodes to capture hierarchical relationships, and (2) heterogeneous intra-scale edges linking visual and textual nodes on the same scale. To further enhance semantic consistency, HiVE-MIL incorporates a two-stage, text-guided dynamic filtering mechanism that removes weakly correlated patch-text pairs, and introduces a hierarchical contrastive loss to align textual semantics across scales. Extensive experiments on TCGA breast, lung, and kidney cancer datasets demonstrate that HiVE-MIL consistently outperforms both traditional MIL and recent VLM-based MIL approaches, achieving gains of up to 4.1% in macro F1 under 16-shot settings. Our results demonstrate the value of jointly modeling hierarchical structure and multimodal alignment for efficient and scalable learning from limited pathology data. The code is available at https://github.com/bryanwong17/HiVE-MIL.
>
---
#### [replaced 006] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.04297v2](http://arxiv.org/pdf/2508.04297v2)**

> **作者:** Yaopeng Lou; Liao Shen; Tianqi Liu; Jiaqi Li; Zihao Huang; Huiqiang Sun; Zhiguo Cao
>
> **备注:** This work is accepted by ICCV 2025
>
> **摘要:** We present Multi-Baseline Gaussian Splatting (MuGS), a generalized feed-forward approach for novel view synthesis that effectively handles diverse baseline settings, including sparse input views with both small and large baselines. Specifically, we integrate features from Multi-View Stereo (MVS) and Monocular Depth Estimation (MDE) to enhance feature representations for generalizable reconstruction. Next, We propose a projection-and-sampling mechanism for deep depth fusion, which constructs a fine probability volume to guide the regression of the feature map. Furthermore, We introduce a reference-view loss to improve geometry and optimization efficiency. We leverage 3D Gaussian representations to accelerate training and inference time while enhancing rendering quality. MuGS achieves state-of-the-art performance across multiple baseline settings and diverse scenarios ranging from simple objects (DTU) to complex indoor and outdoor scenes (RealEstate10K). We also demonstrate promising zero-shot performance on the LLFF and Mip-NeRF 360 datasets. Code is available at https://github.com/EuclidLou/MuGS.
>
---
#### [replaced 007] PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09237v2](http://arxiv.org/pdf/2506.09237v2)**

> **作者:** Mojtaba Nafez; Amirhossein Koochakian; Arad Maleki; Jafar Habibi; Mohammad Hossein Rohban
>
> **备注:** Accepted to the Conference on Computer Vision and Pattern Recognition (CVPR) 2025
>
> **摘要:** Anomaly Detection (AD) and Anomaly Localization (AL) are crucial in fields that demand high reliability, such as medical imaging and industrial monitoring. However, current AD and AL approaches are often susceptible to adversarial attacks due to limitations in training data, which typically include only normal, unlabeled samples. This study introduces PatchGuard, an adversarially robust AD and AL method that incorporates pseudo anomalies with localization masks within a Vision Transformer (ViT)-based architecture to address these vulnerabilities. We begin by examining the essential properties of pseudo anomalies, and follow it by providing theoretical insights into the attention mechanisms required to enhance the adversarial robustness of AD and AL systems. We then present our approach, which leverages Foreground-Aware Pseudo-Anomalies to overcome the deficiencies of previous anomaly-aware methods. Our method incorporates these crafted pseudo-anomaly samples into a ViT-based framework, with adversarial training guided by a novel loss function designed to improve model robustness, as supported by our theoretical analysis. Experimental results on well-established industrial and medical datasets demonstrate that PatchGuard significantly outperforms previous methods in adversarial settings, achieving performance gains of $53.2\%$ in AD and $68.5\%$ in AL, while also maintaining competitive accuracy in non-adversarial settings. The code repository is available at https://github.com/rohban-lab/PatchGuard .
>
---
#### [replaced 008] Multi-Atlas Brain Network Classification through Consistency Distillation and Complementary Information Fusion
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.08228v2](http://arxiv.org/pdf/2410.08228v2)**

> **作者:** Jiaxing Xu; Mengcheng Lan; Xia Dong; Kai He; Wei Zhang; Qingtian Bian; Yiping Ke
>
> **摘要:** In the realm of neuroscience, identifying distinctive patterns associated with neurological disorders via brain networks is crucial. Resting-state functional magnetic resonance imaging (fMRI) serves as a primary tool for mapping these networks by correlating blood-oxygen-level-dependent (BOLD) signals across different brain regions, defined as regions of interest (ROIs). Constructing these brain networks involves using atlases to parcellate the brain into ROIs based on various hypotheses of brain division. However, there is no standard atlas for brain network classification, leading to limitations in detecting abnormalities in disorders. Some recent methods have proposed utilizing multiple atlases, but they neglect consistency across atlases and lack ROI-level information exchange. To tackle these limitations, we propose an Atlas-Integrated Distillation and Fusion network (AIDFusion) to improve brain network classification using fMRI data. AIDFusion addresses the challenge of utilizing multiple atlases by employing a disentangle Transformer to filter out inconsistent atlas-specific information and distill distinguishable connections across atlases. It also incorporates subject- and population-level consistency constraints to enhance cross-atlas consistency. Additionally, AIDFusion employs an inter-atlas message-passing mechanism to fuse complementary information across brain regions. Experimental results on four datasets of different diseases demonstrate the effectiveness and efficiency of AIDFusion compared to state-of-the-art methods. A case study illustrates AIDFusion extract patterns that are both interpretable and consistent with established neuroscience findings.
>
---
#### [replaced 009] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14674v2](http://arxiv.org/pdf/2506.14674v2)**

> **作者:** Ling Li; Yao Zhou; Yuxuan Liang; Fugee Tsung; Jiaheng Wei
>
> **备注:** NeurIPS 2025
>
> **摘要:** Previous methods for image geo-localization have typically treated the task as either classification or retrieval, often relying on black-box decisions that lack interpretability. The rise of large vision-language models (LVLMs) has enabled a rethinking of geo-localization as a reasoning-driven task grounded in visual cues. However, two major challenges persist. On the data side, existing reasoning-focused datasets are primarily based on street-view imagery, offering limited scene diversity and constrained viewpoints. On the modeling side, current approaches predominantly rely on supervised fine-tuning, which yields only marginal improvements in reasoning capabilities. To address these challenges, we propose a novel pipeline that constructs a reasoning-oriented geo-localization dataset, MP16-Reason, using diverse social media images. We introduce GLOBE, Group-relative policy optimization for Localizability assessment and Optimized visual-cue reasoning, yielding Bi-objective geo-Enhancement for the VLM in recognition and reasoning. GLOBE incorporates task-specific rewards that jointly enhance localizability assessment, visual-cue reasoning, and geolocation accuracy. Both qualitative and quantitative results demonstrate that GLOBE outperforms state-of-the-art open-source LVLMs on geo-localization tasks, particularly in diverse visual scenes, while also generating more insightful and interpretable reasoning trajectories. The data and code are available at https://github.com/lingli1996/GLOBE.
>
---
#### [replaced 010] Point Cloud Synthesis Using Inner Product Transforms
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.18987v4](http://arxiv.org/pdf/2410.18987v4)**

> **作者:** Ernst Röell; Bastian Rieck
>
> **备注:** Accepted at the 39th Conference on Neural Information Processing Systems (NeurIPS) 2025. Our code is available at https://github.com/aidos-lab/inner-product-transforms
>
> **摘要:** Point cloud synthesis, i.e. the generation of novel point clouds from an input distribution, remains a challenging task, for which numerous complex machine learning models have been devised. We develop a novel method that encodes geometrical-topological characteristics of point clouds using inner products, leading to a highly-efficient point cloud representation with provable expressivity properties. Integrated into deep learning models, our encoding exhibits high quality in typical tasks like reconstruction, generation, and interpolation, with inference times orders of magnitude faster than existing methods.
>
---
#### [replaced 011] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22535v3](http://arxiv.org/pdf/2505.22535v3)**

> **作者:** Mohamad Hakam Shams Eddin; Yikui Zhang; Stefan Kollet; Juergen Gall
>
> **备注:** Accepted at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Main paper 10 pages, Appendix 54 pages
>
> **摘要:** Recent deep learning approaches for river discharge forecasting have improved the accuracy and efficiency in flood forecasting, enabling more reliable early warning systems for risk management. Nevertheless, existing deep learning approaches in hydrology remain largely confined to local-scale applications and do not leverage the inherent spatial connections of bodies of water. Thus, there is a strong need for new deep learning methodologies that are capable of modeling spatio-temporal relations to improve river discharge and flood forecasting for scientific and operational applications. To address this, we present RiverMamba, a novel deep learning model that is pretrained with long-term reanalysis data and that can forecast global river discharge and floods on a $0.05^\circ$ grid up to $7$ days lead time, which is of high relevance in early warning. To achieve this, RiverMamba leverages efficient Mamba blocks that enable the model to capture spatio-temporal relations in very large river networks and enhance its forecast capability for longer lead times. The forecast blocks integrate ECMWF HRES meteorological forecasts, while accounting for their inaccuracies through spatio-temporal modeling. Our analysis demonstrates that RiverMamba provides reliable predictions of river discharge across various flood return periods, including extreme floods, and lead times, surpassing both AI- and physics-based models. The source code and datasets are publicly available at the project page https://hakamshams.github.io/RiverMamba.
>
---
#### [replaced 012] Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15966v3](http://arxiv.org/pdf/2505.15966v3)**

> **作者:** Haozhe Wang; Alex Su; Weiming Ren; Fangzhen Lin; Wenhu Chen
>
> **备注:** Project Page: https://tiger-ai-lab.github.io/Pixel-Reasoner/, Hands-on Demo: https://huggingface.co/spaces/TIGER-Lab/Pixel-Reasoner
>
> **摘要:** Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks. Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework.
>
---
#### [replaced 013] AngleRoCL: Angle-Robust Concept Learning for Physically View-Invariant T2I Adversarial Patches
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09538v2](http://arxiv.org/pdf/2506.09538v2)**

> **作者:** Wenjun Ji; Yuxiang Fu; Luyang Ying; Deng-Ping Fan; Yuyi Wang; Ming-Ming Cheng; Ivor Tsang; Qing Guo
>
> **摘要:** Cutting-edge works have demonstrated that text-to-image (T2I) diffusion models can generate adversarial patches that mislead state-of-the-art object detectors in the physical world, revealing detectors' vulnerabilities and risks. However, these methods neglect the T2I patches' attack effectiveness when observed from different views in the physical world (i.e., angle robustness of the T2I adversarial patches). In this paper, we study the angle robustness of T2I adversarial patches comprehensively, revealing their angle-robust issues, demonstrating that texts affect the angle robustness of generated patches significantly, and task-specific linguistic instructions fail to enhance the angle robustness. Motivated by the studies, we introduce Angle-Robust Concept Learning (AngleRoCL), a simple and flexible approach that learns a generalizable concept (i.e., text embeddings in implementation) representing the capability of generating angle-robust patches. The learned concept can be incorporated into textual prompts and guides T2I models to generate patches with their attack effectiveness inherently resistant to viewpoint variations. Through extensive simulation and physical-world experiments on five SOTA detectors across multiple views, we demonstrate that AngleRoCL significantly enhances the angle robustness of T2I adversarial patches compared to baseline methods. Our patches maintain high attack success rates even under challenging viewing conditions, with over 50% average relative improvement in attack effectiveness across multiple angles. This research advances the understanding of physically angle-robust patches and provides insights into the relationship between textual concepts and physical properties in T2I-generated contents. We released our code at https://github.com/tsingqguo/anglerocl.
>
---
#### [replaced 014] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.15566v3](http://arxiv.org/pdf/2509.15566v3)**

> **作者:** Shaojie Zhang; Ruoceng Zhang; Pei Fu; Shaokang Wang; Jiahui Yang; Xin Du; Shiqi Cui; Bin Qin; Ying Huang; Zhenbo Luo; Jian Luan
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** In the field of AI-driven human-GUI interaction automation, while rapid advances in multimodal large language models and reinforcement fine-tuning techniques have yielded remarkable progress, a fundamental challenge persists: their interaction logic significantly deviates from natural human-GUI communication patterns. To fill this gap, we propose "Blink-Think-Link" (BTL), a brain-inspired framework for human-GUI interaction that mimics the human cognitive process between users and graphical interfaces. The system decomposes interactions into three biologically plausible phases: (1) Blink - rapid detection and attention to relevant screen areas, analogous to saccadic eye movements; (2) Think - higher-level reasoning and decision-making, mirroring cognitive planning; and (3) Link - generation of executable commands for precise motor control, emulating human action selection mechanisms. Additionally, we introduce two key technical innovations for the BTL framework: (1) Blink Data Generation - an automated annotation pipeline specifically optimized for blink data, and (2) BTL Reward -- the first rule-based reward mechanism that enables reinforcement learning driven by both process and outcome. Building upon this framework, we develop a GUI agent model named BTL-UI, which demonstrates competitive performance across both static GUI understanding and dynamic interaction tasks in comprehensive benchmarks. These results provide conclusive empirical validation of the framework's efficacy in developing advanced GUI Agents.
>
---
#### [replaced 015] Dataset Condensation with Color Compensation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.01139v3](http://arxiv.org/pdf/2508.01139v3)**

> **作者:** Huyu Wu; Duo Su; Junjie Hou; Guang Li
>
> **备注:** Accepted in TMLR
>
> **摘要:** Dataset condensation always faces a constitutive trade-off: balancing performance and fidelity under extreme compression. Existing methods struggle with two bottlenecks: image-level selection methods (Coreset Selection, Dataset Quantization) suffer from inefficiency condensation, while pixel-level optimization (Dataset Distillation) introduces semantic distortion due to over-parameterization. With empirical observations, we find that a critical problem in dataset condensation is the oversight of color's dual role as an information carrier and a basic semantic representation unit. We argue that improving the colorfulness of condensed images is beneficial for representation learning. Motivated by this, we propose DC3: a Dataset Condensation framework with Color Compensation. After a calibrated selection strategy, DC3 utilizes the latent diffusion model to enhance the color diversity of an image rather than creating a brand-new one. Extensive experiments demonstrate the superior performance and generalization of DC3 that outperforms SOTA methods across multiple benchmarks. To the best of our knowledge, besides focusing on downstream tasks, DC3 is the first research to fine-tune pre-trained diffusion models with condensed datasets. The Frechet Inception Distance (FID) and Inception Score (IS) results prove that training networks with our high-quality datasets is feasible without model collapse or other degradation issues. Code and generated data are available at https://github.com/528why/Dataset-Condensation-with-Color-Compensation.
>
---
#### [replaced 016] Operational Change Detection for Geographical Information: Overview and Challenges
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.14109v2](http://arxiv.org/pdf/2503.14109v2)**

> **作者:** Nicolas Gonthier
>
> **备注:** Preprint under review
>
> **摘要:** Rapid evolution of territories due to climate change and human impact requires prompt and effective updates to geospatial databases maintained by the National Mapping Agency. This paper presents a comprehensive overview of change detection methods tailored for the operational updating of large-scale geographic databases. This review first outlines the fundamental definition of change, emphasizing its multifaceted nature, from temporal to semantic characterization. It categorizes automatic change detection methods into four main families: rule-based, statistical, machine learning, and simulation methods. The strengths, limitations, and applicability of every family are discussed in the context of various input data. Then, key applications for National Mapping Agencies are identified, particularly the optimization of geospatial database updating, change-based phenomena, and dynamics monitoring. Finally, the paper highlights the current challenges for leveraging change detection such as the variability of change definition, the missing of relevant large-scale datasets, the diversity of input data, the unstudied no-change detection, the human in the loop integration and the operational constraints. The discussion underscores the necessity for ongoing innovation in change detection techniques to address the future needs of geographic information systems for national mapping agencies.
>
---
#### [replaced 017] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02064v3](http://arxiv.org/pdf/2505.02064v3)**

> **作者:** Shuhang Xun; Sicheng Tao; Jungang Li; Yibo Shi; Zhixin Lin; Zhanhui Zhu; Yibo Yan; Hanqian Li; Linghao Zhang; Shikang Wang; Yixin Liu; Hanbo Zhang; Ying Ma; Xuming Hu
>
> **备注:** Accepted by NeurIPS 2025 Datasets and Benchmarks Track;
>
> **摘要:** Multimodal Large Language Models (MLLMs) increasingly excel at perception, understanding, and reasoning. However, current benchmarks inadequately evaluate their ability to perform these tasks continuously in dynamic, real-world environments. To bridge this gap, we introduce RTV-Bench, a fine-grained benchmark for MLLM real-time video analysis. RTV-Bench uses three key principles: (1) Multi-Timestamp Question Answering (MTQA), where answers evolve with scene changes; (2) Hierarchical Question Structure, combining basic and advanced queries; and (3) Multi-dimensional Evaluation, assessing the ability of continuous perception, understanding, and reasoning. RTV-Bench contains 552 diverse videos (167.2 hours) and 4,631 high-quality QA pairs. We evaluated leading MLLMs, including proprietary (GPT-4o, Gemini 2.0), open-source offline (Qwen2.5-VL, VideoLLaMA3), and open-source real-time (VITA-1.5, InternLM-XComposer2.5-OmniLive) models. Experiment results show open-source real-time models largely outperform offline ones but still trail top proprietary models. Our analysis also reveals that larger model size or higher frame sampling rates do not significantly boost RTV-Bench performance, sometimes causing slight decreases. This underscores the need for better model architectures optimized for video stream processing and long sequences to advance real-time video analysis with MLLMs. Our benchmark toolkit is available at: https://github.com/LJungang/RTV-Bench.
>
---
#### [replaced 018] Photorealistic Inpainting for Perturbation-based Explanations in Ecological Monitoring
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.03317v2](http://arxiv.org/pdf/2510.03317v2)**

> **作者:** Günel Aghakishiyeva; Jiayi Zhou; Saagar Arya; Julian Dale; James David Poling; Holly R. Houliston; Jamie N. Womble; Gregory D. Larsen; David W. Johnston; Brinnae Bent
>
> **备注:** NeurIPS 2025 Imageomics Workshop
>
> **摘要:** Ecological monitoring is increasingly automated by vision models, yet opaque predictions limit trust and field adoption. We present an inpainting-guided, perturbation-based explanation technique that produces photorealistic, mask-localized edits that preserve scene context. Unlike masking or blurring, these edits stay in-distribution and reveal which fine-grained morphological cues drive predictions in tasks such as species recognition and trait attribution. We demonstrate the approach on a YOLOv9 detector fine-tuned for harbor seal detection in Glacier Bay drone imagery, using Segment-Anything-Model-refined masks to support two interventions: (i) object removal/replacement (e.g., replacing seals with plausible ice/water or boats) and (ii) background replacement with original animals composited onto new scenes. Explanations are assessed by re-scoring perturbed images (flip rate, confidence drop) and by expert review for ecological plausibility and interpretability. The resulting explanations localize diagnostic structures, avoid deletion artifacts common to traditional perturbations, and yield domain-relevant insights that support expert validation and more trustworthy deployment of AI in ecology.
>
---
#### [replaced 019] Knot So Simple: A Minimalistic Environment for Spatial Reasoning
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.18028v2](http://arxiv.org/pdf/2505.18028v2)**

> **作者:** Zizhao Chen; Yoav Artzi
>
> **摘要:** We propose KnotGym, an interactive environment for complex, spatial reasoning and manipulation. KnotGym includes goal-oriented rope manipulation tasks with varying levels of complexity, all requiring acting from pure image observations. Tasks are defined along a clear and quantifiable axis of complexity based on the number of knot crossings, creating a natural generalization test. KnotGym has a simple observation space, allowing for scalable development, yet it highlights core challenges in integrating acute perception, spatial reasoning, and grounded manipulation. We evaluate methods of different classes, including model-based RL, model-predictive control, and chain-of-thought reasoning, and illustrate the challenges KnotGym presents. KnotGym is available at https://github.com/lil-lab/knotgym.
>
---
#### [replaced 020] Metropolis-Hastings Sampling for 3D Gaussian Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12945v2](http://arxiv.org/pdf/2506.12945v2)**

> **作者:** Hyunjin Kim; Haebeom Jung; Jaesik Park
>
> **备注:** NeurIPS 2025. Project Page: https://hjhyunjinkim.github.io/MH-3DGS
>
> **摘要:** We propose an adaptive sampling framework for 3D Gaussian Splatting (3DGS) that leverages comprehensive multi-view photometric error signals within a unified Metropolis-Hastings approach. Vanilla 3DGS heavily relies on heuristic-based density-control mechanisms (e.g., cloning, splitting, and pruning), which can lead to redundant computations or premature removal of beneficial Gaussians. Our framework overcomes these limitations by reformulating densification and pruning as a probabilistic sampling process, dynamically inserting and relocating Gaussians based on aggregated multi-view errors and opacity scores. Guided by Bayesian acceptance tests derived from these error-based importance scores, our method substantially reduces reliance on heuristics, offers greater flexibility, and adaptively infers Gaussian distributions without requiring predefined scene complexity. Experiments on benchmark datasets, including Mip-NeRF360, Tanks and Temples and Deep Blending, show that our approach reduces the number of Gaussians needed, achieving faster convergence while matching or modestly surpassing the view-synthesis quality of state-of-the-art models.
>
---
#### [replaced 021] OmniNWM: Omniscient Driving Navigation World Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.18313v3](http://arxiv.org/pdf/2510.18313v3)**

> **作者:** Bohan Li; Zhuang Ma; Dalong Du; Baorui Peng; Zhujin Liang; Zhenqiang Liu; Chao Ma; Yueming Jin; Hao Zhao; Wenjun Zeng; Xin Jin
>
> **备注:** https://arlo0o.github.io/OmniNWM/
>
> **摘要:** Autonomous driving world models are expected to work effectively across three core dimensions: state, action, and reward. Existing models, however, are typically restricted to limited state modalities, short video sequences, imprecise action control, and a lack of reward awareness. In this paper, we introduce OmniNWM, an omniscient panoramic navigation world model that addresses all three dimensions within a unified framework. For state, OmniNWM jointly generates panoramic videos of RGB, semantics, metric depth, and 3D occupancy. A flexible forcing strategy enables high-quality long-horizon auto-regressive generation. For action, we introduce a normalized panoramic Plucker ray-map representation that encodes input trajectories into pixel-level signals, enabling highly precise and generalizable control over panoramic video generation. Regarding reward, we move beyond learning reward functions with external image-based models: instead, we leverage the generated 3D occupancy to directly define rule-based dense rewards for driving compliance and safety. Extensive experiments demonstrate that OmniNWM achieves state-of-the-art performance in video generation, control accuracy, and long-horizon stability, while providing a reliable closed-loop evaluation framework through occupancy-grounded rewards. Project page is available at https://arlo0o.github.io/OmniNWM/.
>
---
#### [replaced 022] Grids Often Outperform Implicit Neural Representation at Compressing Dense Signals
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11139v2](http://arxiv.org/pdf/2506.11139v2)**

> **作者:** Namhoon Kim; Sara Fridovich-Keil
>
> **备注:** Our analysis are available at https://github.com/voilalab/INR-benchmark
>
> **摘要:** Implicit Neural Representations (INRs) have recently shown impressive results, but their fundamental capacity, implicit biases, and scaling behavior remain poorly understood. We investigate the performance of diverse INRs across a suite of 2D and 3D real and synthetic signals with varying effective bandwidth, as well as both overfitting and generalization tasks including tomography, super-resolution, and denoising. By stratifying performance according to model size as well as signal type and bandwidth, our results shed light on how different INR and grid representations allocate their capacity. We find that, for most tasks and signals, a simple regularized grid with interpolation trains faster and to higher quality than any INR with the same number of parameters. We also find limited settings--namely fitting binary signals such as shape contours--where INRs outperform grids, to guide future development and use of INRs towards the most advantageous applications.
>
---
#### [replaced 023] Guided MRI Reconstruction via Schrödinger Bridge
- **分类: eess.IV; cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2411.14269v2](http://arxiv.org/pdf/2411.14269v2)**

> **作者:** Yue Wang; Yuanbiao Yang; Zhuo-xu Cui; Tian Zhou; Bingsheng Huang; Hairong Zheng; Dong Liang; Yanjie Zhu
>
> **摘要:** Magnetic Resonance Imaging (MRI) is an inherently multi-contrast modality, where cross-contrast priors can be exploited to improve image reconstruction from undersampled data. Recently, diffusion models have shown remarkable performance in MRI reconstruction. However, they still struggle to effectively utilize such priors, mainly because existing methods rely on feature-level fusion in image or latent spaces, which lacks explicit structural correspondence and thus leads to suboptimal performance. To address this issue, we propose $\mathbf{I}^2$SB-Inversion, a multi-contrast guided reconstruction framework based on the Schr\"odinger Bridge (SB). The proposed method performs pixel-wise translation between paired contrasts, providing explicit structural constraints between the guidance and target images. Furthermore, an Inversion strategy is introduced to correct inter-modality misalignment, which often occurs in guided reconstruction, thereby mitigating artifacts and improving reconstruction accuracy. Experiments on paired T1- and T2-weighted datasets demonstrate that $\mathbf{I}^2$SB-Inversion achieves a high acceleration factor of up to 14.4 and consistently outperforms existing methods in both quantitative and qualitative evaluations.
>
---
#### [replaced 024] MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2306.13394v5](http://arxiv.org/pdf/2306.13394v5)**

> **作者:** Chaoyou Fu; Peixian Chen; Yunhang Shen; Yulei Qin; Mengdan Zhang; Xu Lin; Jinrui Yang; Xiawu Zheng; Ke Li; Xing Sun; Yunsheng Wu; Rongrong Ji; Caifeng Shan; Ran He
>
> **备注:** NeurIPS DB 2025 Spotlight, Project Page: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation
>
> **摘要:** Multimodal Large Language Model (MLLM) relies on the powerful LLM to perform multimodal tasks, showing amazing emergent abilities in recent studies, such as writing poems based on an image. However, it is difficult for these case studies to fully reflect the performance of MLLM, lacking a comprehensive evaluation. In this paper, we fill in this blank, presenting the first comprehensive MLLM Evaluation benchmark MME. It measures both perception and cognition abilities on a total of 14 subtasks. In order to avoid data leakage that may arise from direct use of public datasets for evaluation, the annotations of instruction-answer pairs are all manually designed. The concise instruction design allows us to fairly compare MLLMs, instead of struggling in prompt engineering. Besides, with such an instruction, we can also easily carry out quantitative statistics. A total of 30 advanced MLLMs are comprehensively evaluated on our MME, which not only suggests that existing MLLMs still have a large room for improvement, but also reveals the potential directions for the subsequent model optimization. The data are released at the project page https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation.
>
---
#### [replaced 025] RigAnything: Template-Free Autoregressive Rigging for Diverse 3D Assets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.09615v2](http://arxiv.org/pdf/2502.09615v2)**

> **作者:** Isabella Liu; Zhan Xu; Wang Yifan; Hao Tan; Zexiang Xu; Xiaolong Wang; Hao Su; Zifan Shi
>
> **备注:** SIGGRAPH TOG 2025, Project page: https://www.liuisabella.com/RigAnything
>
> **摘要:** We present RigAnything, a novel autoregressive transformer-based model, which makes 3D assets rig-ready by probabilistically generating joints and skeleton topologies and assigning skinning weights in a template-free manner. Unlike most existing auto-rigging methods, which rely on predefined skeleton templates and are limited to specific categories like humanoid, RigAnything approaches the rigging problem in an autoregressive manner, iteratively predicting the next joint based on the global input shape and the previous prediction. While autoregressive models are typically used to generate sequential data, RigAnything extends its application to effectively learn and represent skeletons, which are inherently tree structures. To achieve this, we organize the joints in a breadth-first search (BFS) order, enabling the skeleton to be defined as a sequence of 3D locations and the parent index. Furthermore, our model improves the accuracy of position prediction by leveraging diffusion modeling, ensuring precise and consistent placement of joints within the hierarchy. This formulation allows the autoregressive model to efficiently capture both spatial and hierarchical relationships within the skeleton. Trained end-to-end on both RigNet and Objaverse datasets, RigAnything demonstrates state-of-the-art performance across diverse object types, including humanoids, quadrupeds, marine creatures, insects, and many more, surpassing prior methods in quality, robustness, generalizability, and efficiency. It achieves significantly faster performance than existing auto-rigging methods, completing rigging in under a few seconds per shape. Please check our website for more details: https://www.liuisabella.com/RigAnything
>
---
#### [replaced 026] MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.15548v4](http://arxiv.org/pdf/2509.15548v4)**

> **作者:** Deming Li; Kaiwen Jiang; Yutao Tang; Ravi Ramamoorthi; Rama Chellappa; Cheng Peng
>
> **摘要:** In-the-wild photo collections often contain limited volumes of imagery and exhibit multiple appearances, e.g., taken at different times of day or seasons, posing significant challenges to scene reconstruction and novel view synthesis. Although recent adaptations of Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have improved in these areas, they tend to oversmooth and are prone to overfitting. In this paper, we present MS-GS, a novel framework designed with Multi-appearance capabilities in Sparse-view scenarios using 3DGS. To address the lack of support due to sparse initializations, our approach is built on the geometric priors elicited from monocular depth estimations. The key lies in extracting and utilizing local semantic regions with a Structure-from-Motion (SfM) points anchored algorithm for reliable alignment and geometry cues. Then, to introduce multi-view constraints, we propose a series of geometry-guided supervision steps at virtual views in pixel and feature levels to encourage 3D consistency and reduce overfitting. We also introduce a dataset and an in-the-wild experiment setting to set up more realistic benchmarks. We demonstrate that MS-GS achieves photorealistic renderings under various challenging sparse-view and multi-appearance conditions, and outperforms existing approaches significantly across different datasets.
>
---
#### [replaced 027] CLASP: Adaptive Spectral Clustering for Unsupervised Per-Image Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.25016v2](http://arxiv.org/pdf/2509.25016v2)**

> **作者:** Max Curie; Paulo da Costa
>
> **摘要:** We introduce CLASP (Clustering via Adaptive Spectral Processing), a lightweight framework for unsupervised image segmentation that operates without any labeled data or finetuning. CLASP first extracts per patch features using a self supervised ViT encoder (DINO); then, it builds an affinity matrix and applies spectral clustering. To avoid manual tuning, we select the segment count automatically with a eigengap silhouette search, and we sharpen the boundaries with a fully connected DenseCRF. Despite its simplicity and training free nature, CLASP attains competitive mIoU and pixel accuracy on COCO Stuff and ADE20K, matching recent unsupervised baselines. The zero training design makes CLASP a strong, easily reproducible baseline for large unannotated corpora especially common in digital advertising and marketing workflows such as brand safety screening, creative asset curation, and social media content moderation
>
---
#### [replaced 028] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06363v2](http://arxiv.org/pdf/2507.06363v2)**

> **作者:** Szymon Płotka; Gizem Mert; Maciej Chrabaszcz; Ewa Szczurek; Arkadiusz Sitek
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** In recent years, artificial intelligence has significantly advanced medical image segmentation. Nonetheless, challenges remain, including efficient 3D medical image processing across diverse modalities and handling data variability. In this work, we introduce Hierarchical Soft Mixture-of-Experts (HoME), a two-level token-routing layer for efficient long-context modeling, specifically designed for 3D medical image segmentation. Built on the Mamba Selective State Space Model (SSM) backbone, HoME enhances sequential modeling through adaptive expert routing. In the first level, a Soft Mixture-of-Experts (SMoE) layer partitions input sequences into local groups, routing tokens to specialized per-group experts for localized feature extraction. The second level aggregates these outputs through a global SMoE layer, enabling cross-group information fusion and global context refinement. This hierarchical design, combining local expert routing with global expert refinement, enhances generalizability and segmentation performance, surpassing state-of-the-art results across datasets from the three most widely used 3D medical imaging modalities and varying data qualities. The code is publicly available at https://github.com/gmum/MambaHoME.
>
---
#### [replaced 029] ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23356v3](http://arxiv.org/pdf/2503.23356v3)**

> **作者:** Linfeng Tang; Yeda Wang; Zhanchuan Cai; Junjun Jiang; Jiayi Ma
>
> **备注:** Accepted to NeurIPS 2025. The code is available at https://github.com/Linfeng-Tang/ControlFusion
>
> **摘要:** Current image fusion methods struggle to address the composite degradations encountered in real-world imaging scenarios and lack the flexibility to accommodate user-specific requirements. In response to these challenges, we propose a controllable image fusion framework with language-vision prompts, termed ControlFusion, which adaptively neutralizes composite degradations. On the one hand, we develop a degraded imaging model that integrates physical imaging mechanisms, including the Retinex theory and atmospheric scattering principle, to simulate composite degradations, thereby providing potential for addressing real-world complex degradations from the data level. On the other hand, we devise a prompt-modulated restoration and fusion network that dynamically enhances features with degradation prompts, enabling our method to accommodate composite degradation of varying levels. Specifically, considering individual variations in quality perception of users, we incorporate a text encoder to embed user-specified degradation types and severity levels as degradation prompts. We also design a spatial-frequency collaborative visual adapter that autonomously perceives degradations in source images, thus eliminating the complete dependence on user instructions. Extensive experiments demonstrate that ControlFusion outperforms SOTA fusion methods in fusion quality and degradation handling, particularly in countering real-world and compound degradations with various levels. The source code is publicly available at https://github.com/Linfeng-Tang/ControlFusion.
>
---
#### [replaced 030] An Evaluation of DUSt3R/MASt3R/VGGT 3D Reconstruction on Photogrammetric Aerial Blocks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.14798v2](http://arxiv.org/pdf/2507.14798v2)**

> **作者:** Xinyi Wu; Steven Landgraf; Markus Ulrich; Rongjun Qin
>
> **备注:** 23 pages, 7 figures, this manuscript has been submitted to Geo-spatial Information Science for consideration
>
> **摘要:** State-of-the-art 3D computer vision algorithms continue to advance in handling sparse, unordered image sets. Recently developed foundational models for 3D reconstruction, such as Dense and Unconstrained Stereo 3D Reconstruction (DUSt3R), Matching and Stereo 3D Reconstruction (MASt3R), and Visual Geometry Grounded Transformer (VGGT), have attracted attention due to their ability to handle very sparse image overlaps. Evaluating DUSt3R/MASt3R/VGGT on typical aerial images matters, as these models may handle extremely low image overlaps, stereo occlusions, and textureless regions. For redundant collections, they can accelerate 3D reconstruction by using extremely sparsified image sets. Despite tests on various computer vision benchmarks, their potential on photogrammetric aerial blocks remains unexplored. This paper conducts a comprehensive evaluation of the pre-trained DUSt3R/MASt3R/VGGT models on the aerial blocks of the UseGeo dataset for pose estimation and dense 3D reconstruction. Results show these methods can accurately reconstruct dense point clouds from very sparse image sets (fewer than 10 images, up to 518 pixels resolution), with completeness gains up to +50% over COLMAP. VGGT also demonstrates higher computational efficiency, scalability, and more reliable camera pose estimation. However, all exhibit limitations with high-resolution images and large sets, as pose reliability declines with more images and geometric complexity. These findings suggest transformer-based methods cannot fully replace traditional SfM and MVS, but offer promise as complementary approaches, especially in challenging, low-resolution, and sparse scenarios.
>
---
#### [replaced 031] Robust Residual Finite Scalar Quantization for Neural Compression
- **分类: eess.IV; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.15860v2](http://arxiv.org/pdf/2508.15860v2)**

> **作者:** Xiaoxu Zhu; Jiakui Li; Ken Zheng; Guiping Zhong; Huimeng Wang; Shiyin Kang; Dahua Lin
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Finite Scalar Quantization (FSQ) offers simplified training but suffers from residual magnitude decay in multi-stage settings, where subsequent stages receive exponentially weaker signals. We propose Robust Residual Finite Scalar Quantization (RFSQ), addressing this fundamental limitation through two novel conditioning strategies: learnable scaling factors and invertible layer normalization. Our experiments across audio and image modalities demonstrate RFSQ's effectiveness and generalizability. In audio reconstruction at 24 bits/frame, RFSQ-LayerNorm achieves 3.646 DNSMOS, a 3.6% improvement over state-of-the-art RVQ (3.518). On ImageNet, RFSQ achieves 0.102 L1 loss and 0.100 perceptual loss, with LayerNorm providing 9.7% L1 improvement and 17.4% perceptual improvement over unconditioned variants. The LayerNorm strategy consistently outperforms alternatives by maintaining normalized input statistics across stages, effectively preventing exponential magnitude decay that limits naive residual approaches. RFSQ combines FSQ's simplicity with multi-stage quantization's representational power, establishing a new standard for neural compression across diverse modalities.
>
---
#### [replaced 032] Breaking the Batch Barrier (B3) of Contrastive Learning via Smart Batch Mining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11293v2](http://arxiv.org/pdf/2505.11293v2)**

> **作者:** Raghuveer Thirukovalluru; Rui Meng; Ye Liu; Karthikeyan K; Mingyi Su; Ping Nie; Semih Yavuz; Yingbo Zhou; Wenhu Chen; Bhuwan Dhingra
>
> **备注:** 17 pages, 4 figures
>
> **摘要:** Contrastive learning (CL) is a prevalent technique for training embedding models, which pulls semantically similar examples (positives) closer in the representation space while pushing dissimilar ones (negatives) further apart. A key source of negatives are 'in-batch' examples, i.e., positives from other examples in the batch. Effectiveness of such models is hence strongly influenced by the size and quality of training batches. In this work, we propose 'Breaking the Batch Barrier' (B3), a novel batch construction strategy designed to curate high-quality batches for CL. Our approach begins by using a pretrained teacher embedding model to rank all examples in the dataset, from which a sparse similarity graph is constructed. A community detection algorithm is then applied to this graph to identify clusters of examples that serve as strong negatives for one another. The clusters are then used to construct batches that are rich in in-batch negatives. Empirical results on the MMEB multimodal embedding benchmark (36 tasks) demonstrate that our method sets a new state of the art, outperforming previous best methods by +1.3 and +2.9 points at the 7B and 2B model scales, respectively. Notably, models trained with B3 surpass existing state-of-the-art results even with a batch size as small as 64, which is 4-16x smaller than that required by other methods. Moreover, experiments show that B3 generalizes well across domains and tasks, maintaining strong performance even when trained with considerably weaker teachers.
>
---
#### [replaced 033] Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.19618v2](http://arxiv.org/pdf/2510.19618v2)**

> **作者:** Junfei Zhou; Penglin Dai; Quanmin Wei; Bingyi Liu; Xiao Wu; Jianping Wang
>
> **备注:** 19 pages, 10 figures, accepted to NeurIPS 2025
>
> **摘要:** Multi-agent collaboration enhances the perception capabilities of individual agents through information sharing. However, in real-world applications, differences in sensors and models across heterogeneous agents inevitably lead to domain gaps during collaboration. Existing approaches based on adaptation and reconstruction fail to support pragmatic heterogeneous collaboration due to two key limitations: (1) Intrusive retraining of the encoder or core modules disrupts the established semantic consistency among agents; and (2) accommodating new agents incurs high computational costs, limiting scalability. To address these challenges, we present a novel Generative Communication mechanism (GenComm) that facilitates seamless perception across heterogeneous multi-agent systems through feature generation, without altering the original network, and employs lightweight numerical alignment of spatial information to efficiently integrate new agents at minimal cost. Specifically, a tailored Deformable Message Extractor is designed to extract spatial message for each collaborator, which is then transmitted in place of intermediate features. The Spatial-Aware Feature Generator, utilizing a conditional diffusion model, generates features aligned with the ego agent's semantic space while preserving the spatial information of the collaborators. These generated features are further refined by a Channel Enhancer before fusion. Experiments conducted on the OPV2V-H, DAIR-V2X and V2X-Real datasets demonstrate that GenComm outperforms existing state-of-the-art methods, achieving an 81\% reduction in both computational cost and parameter count when incorporating new agents. Our code is available at https://github.com/jeffreychou777/GenComm.
>
---
#### [replaced 034] The Narrow Gate: Localized Image-Text Communication in Native Multimodal Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06646v3](http://arxiv.org/pdf/2412.06646v3)**

> **作者:** Alessandro Serra; Francesco Ortu; Emanuele Panizon; Lucrezia Valeriani; Lorenzo Basile; Alessio Ansuini; Diego Doimo; Alberto Cazzaniga
>
> **摘要:** Recent advances in multimodal training have significantly improved the integration of image understanding and generation within a unified model. This study investigates how vision-language models (VLMs) handle image-understanding tasks, focusing on how visual information is processed and transferred to the textual domain. We compare native multimodal VLMs, models trained from scratch on multimodal data to generate both text and images, and non-native multimodal VLMs, models adapted from pre-trained large language models or capable of generating only text, highlighting key differences in information flow. We find that in native multimodal VLMs, image and text embeddings are more separated within the residual stream. Moreover, VLMs differ in how visual information reaches text: non-native multimodal VLMs exhibit a distributed communication pattern, where information is exchanged through multiple image tokens, whereas models trained natively for joint image and text generation tend to rely on a single post-image token that acts as a narrow gate for visual information. We show that ablating this single token significantly deteriorates image-understanding performance, whereas targeted, token-level interventions reliably steer image semantics and downstream text with fine-grained control.
>
---
#### [replaced 035] Progressive Data Dropout: An Embarrassingly Simple Approach to Faster Training
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22342v3](http://arxiv.org/pdf/2505.22342v3)**

> **作者:** Shriram M Sathiyanarayanan; Xinyue Hao; Shihao Hou; Yang Lu; Laura Sevilla-Lara; Anurag Arnab; Shreyank N Gowda
>
> **摘要:** The success of the machine learning field has reliably depended on training on large datasets. While effective, this trend comes at an extraordinary cost. This is due to two deeply intertwined factors: the size of models and the size of datasets. While promising research efforts focus on reducing the size of models, the other half of the equation remains fairly mysterious. Indeed, it is surprising that the standard approach to training remains to iterate over and over, uniformly sampling the training dataset. In this paper we explore a series of alternative training paradigms that leverage insights from hard-data-mining and dropout, simple enough to implement and use that can become the new training standard. The proposed Progressive Data Dropout reduces the number of effective epochs to as little as 12.4% of the baseline. This savings actually do not come at any cost for accuracy. Surprisingly, the proposed method improves accuracy by up to 4.82%. Our approach requires no changes to model architecture or optimizer, and can be applied across standard training pipelines, thus posing an excellent opportunity for wide adoption. Code can be found here: https://github.com/bazyagami/LearningWithRevision
>
---
#### [replaced 036] Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06485v2](http://arxiv.org/pdf/2507.06485v2)**

> **作者:** Ziyang Wang; Jaehong Yoon; Shoubin Yu; Md Mohaiminul Islam; Gedas Bertasius; Mohit Bansal
>
> **备注:** EMNLP 2025. The first two authors contributed equally. Project page: https://sites.google.com/cs.unc.edu/videorts2025/
>
> **摘要:** Despite advances in reinforcement learning (RL)-based video reasoning with large language models (LLMs), data collection and fine-tuning remain significant challenges. These methods often rely on large-scale supervised fine-tuning (SFT) with extensive video data and long Chain-of-Thought (CoT) annotations, making them costly and hard to scale. To address this, we present Video-RTS, a new approach to improve video reasoning capability with drastically improved data efficiency by combining data-efficient RL with a video-adaptive test-time scaling (TTS) strategy. Building on observations about the data scaling, we skip the resource-intensive SFT step and employ efficient pure-RL training with output-based rewards, requiring no additional annotations or extensive fine-tuning. Furthermore, to utilize computational resources more efficiently, we introduce a sparse-to-dense video TTS strategy that improves inference by iteratively adding frames based on output consistency. We validate our approach on multiple video reasoning benchmarks, showing that Video-RTS surpasses existing video reasoning models by 2.4% in accuracy using only 3.6% training samples. Specifically, Video-RTS achieves a 4.2% improvement on Video-Holmes, a recent and challenging video reasoning benchmark. Notably, our pure RL training and adaptive video TTS offer complementary strengths, enabling Video-RTS's strong reasoning performance.
>
---
#### [replaced 037] On the Influence of Shape, Texture and Color for Learning Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.14878v2](http://arxiv.org/pdf/2410.14878v2)**

> **作者:** Annika Mütze; Natalie Grabowsky; Edgar Heinert; Matthias Rottmann; Hanno Gottschalk
>
> **备注:** Accepted at the 28th European Conference on Artificial Intelligence
>
> **摘要:** Recent research has investigated the shape and texture biases of pre-trained deep neural networks (DNNs) in image classification. Those works test how much a trained DNN relies on specific image cues like texture. The present study shifts the focus to understanding the cue influence during training, analyzing what DNNs can learn from shape, texture, and color cues in absence of the others; investigating their individual and combined influence on the learning success. We analyze these cue influences at multiple levels by decomposing datasets into cue-specific versions. Addressing semantic segmentation, we learn the given task from these reduced cue datasets, creating cue experts. Early fusion of cues is performed by constructing appropriate datasets. This is complemented by a late fusion of experts which allows us to study cue influence location-dependent on pixel level. Experiments on Cityscapes, PASCAL Context, and a synthetic CARLA dataset show that while no single cue dominates, the shape + color expert predominantly improves the prediction of small objects and border pixels. The cue performance order is consistent for the tested convolutional and transformer architecture, indicating similar cue extraction capabilities, although pre-trained transformers are said to be more biased towards shape than convolutional neural networks.
>
---
#### [replaced 038] PLD: A Choice-Theoretic List-Wise Knowledge Distillation
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2506.12542v3](http://arxiv.org/pdf/2506.12542v3)**

> **作者:** Ejafa Bassam; Dawei Zhu; Kaigui Bian
>
> **摘要:** Knowledge distillation is a model compression technique in which a compact "student" network is trained to replicate the predictive behavior of a larger "teacher" network. In logit-based knowledge distillation, it has become the de facto approach to augment cross-entropy with a distillation term. Typically, this term is either a KL divergence that matches marginal probabilities or a correlation-based loss that captures intra- and inter-class relationships. In every case, it acts as an additional term to cross-entropy. This term has its own weight, which must be carefully tuned. In this paper, we adopt a choice-theoretic perspective and recast knowledge distillation under the Plackett-Luce model by interpreting teacher logits as "worth" scores. We introduce "Plackett-Luce Distillation (PLD)", a weighted list-wise ranking loss. In PLD, the teacher model transfers knowledge of its full ranking of classes, weighting each ranked choice by its own confidence. PLD directly optimizes a single "teacher-optimal" ranking. The true label is placed first, followed by the remaining classes in descending teacher confidence. This process yields a convex and translation-invariant surrogate that subsumes weighted cross-entropy. Empirically, across CIFAR-100, ImageNet-1K, and MS-COCO, PLD achieves consistent gains across diverse architectures and distillation objectives, including divergence-based, correlation-based, and feature-based methods, in both homogeneous and heterogeneous teacher-student pairs.
>
---
#### [replaced 039] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.17054v2](http://arxiv.org/pdf/2508.17054v2)**

> **作者:** Qingwen Zhang; Xiaomeng Zhu; Yushan Zhang; Yixi Cai; Olov Andersson; Patric Jensfelt
>
> **备注:** NeurIPS 2025 Spotlight, 18 pages (10 main pages + 8 supp materail), 11 figures, code at https://github.com/Kin-Zhang/DeltaFlow
>
> **摘要:** Previous dominant methods for scene flow estimation focus mainly on input from two consecutive frames, neglecting valuable information in the temporal domain. While recent trends shift towards multi-frame reasoning, they suffer from rapidly escalating computational costs as the number of frames grows. To leverage temporal information more efficiently, we propose DeltaFlow ($\Delta$Flow), a lightweight 3D framework that captures motion cues via a $\Delta$ scheme, extracting temporal features with minimal computational cost, regardless of the number of frames. Additionally, scene flow estimation faces challenges such as imbalanced object class distributions and motion inconsistency. To tackle these issues, we introduce a Category-Balanced Loss to enhance learning across underrepresented classes and an Instance Consistency Loss to enforce coherent object motion, improving flow accuracy. Extensive evaluations on the Argoverse 2, Waymo and nuScenes datasets show that $\Delta$Flow achieves state-of-the-art performance with up to 22% lower error and $2\times$ faster inference compared to the next-best multi-frame supervised method, while also demonstrating a strong cross-domain generalization ability. The code is open-sourced at https://github.com/Kin-Zhang/DeltaFlow along with trained model weights.
>
---
#### [replaced 040] Spiking Neural Networks Need High Frequency Information
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18608v5](http://arxiv.org/pdf/2505.18608v5)**

> **作者:** Yuetong Fang; Deming Zhou; Ziqing Wang; Hongwei Ren; ZeCui Zeng; Lusong Li; Shibo Zhou; Renjing Xu
>
> **摘要:** Spiking Neural Networks promise brain-inspired and energy-efficient computation by transmitting information through binary (0/1) spikes. Yet, their performance still lags behind that of artificial neural networks, often assumed to result from information loss caused by sparse and binary activations. In this work, we challenge this long-standing assumption and reveal a previously overlooked frequency bias: spiking neurons inherently suppress high-frequency components and preferentially propagate low-frequency information. This frequency-domain imbalance, we argue, is the root cause of degraded feature representation in SNNs. Empirically, on Spiking Transformers, adopting Avg-Pooling (low-pass) for token mixing lowers performance to 76.73% on Cifar-100, whereas replacing it with Max-Pool (high-pass) pushes the top-1 accuracy to 79.12%. Accordingly, we introduce Max-Former that restores high-frequency signals through two frequency-enhancing operators: (1) extra Max-Pool in patch embedding, and (2) Depth-Wise Convolution in place of self-attention. Notably, Max-Former attains 82.39% top-1 accuracy on ImageNet using only 63.99M parameters, surpassing Spikformer (74.81%, 66.34M) by +7.58%. Extending our insight beyond transformers, our Max-ResNet-18 achieves state-of-the-art performance on convolution-based benchmarks: 97.17% on CIFAR-10 and 83.06% on CIFAR-100. We hope this simple yet effective solution inspires future research to explore the distinctive nature of spiking neural networks. Code is available: https://github.com/bic-L/MaxFormer.
>
---
#### [replaced 041] WCCNet: Wavelet-context Cooperative Network for Efficient Multispectral Pedestrian Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2308.01042v2](http://arxiv.org/pdf/2308.01042v2)**

> **作者:** Xingjian Wang; Li Chai; Jiming Chen; Zhiguo Shi
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Multispectral pedestrian detection achieves better visibility in challenging conditions and thus is essential to autonomous driving, for which both the accuracy and computational cost are of paramount importance. Most existing approaches treat RGB and infrared modalities equally. They typically adopt two symmetrical backbones for multimodal feature extraction, which ignore the substantial differences between modalities and bring great difficulty for the reduction of the computational cost as well as effective crossmodal fusion. In this work, we propose a novel and efficient framework named Wavelet-context Cooperative Network (WCCNet) that is able to differentially extract complementary features of different spectra with lower computational complexity, and further fuse these diverse features based on their spatially relevant crossmodal semantics. In particular, WCCNet simultaneously explore wavelet context and RGB textures within a cooperative dual-stream backbone, which is composed of adaptive discrete wavelet transform (ADWT) layers and heavyweight neural layers. The ADWT layers extract frequency components for infrared modality, while neural layers handle RGB modality features. Since ADWT layers are lightweight and extract complementary features, this cooperative structure not only significantly reduces the computational complexity, but also facilitates the subsequent crossmodal fusion. To further fuse these infrared and RGB features with significant semantic differences, we elaborately design the crossmodal rearranging fusion module (CMRF), which can mitigate spatial misalignment and merge semantically complementary features in spatially-related local regions to amplify the crossmodal reciprocal information. Experimental results on KAIST and FLIR benchmarks indicate that WCCNet outperforms state-of-the-art methods with considerable efficiency and competitive accuracy.
>
---
#### [replaced 042] LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.13626v2](http://arxiv.org/pdf/2510.13626v2)**

> **作者:** Senyu Fei; Siyin Wang; Junhao Shi; Zihao Dai; Jikun Cai; Pengfang Qian; Li Ji; Xinzhe He; Shiduo Zhang; Zhaoye Fei; Jinlan Fu; Jingjing Gong; Xipeng Qiu
>
> **摘要:** Visual-Language-Action (VLA) models report impressive success rates on robotic manipulation benchmarks, yet these results may mask fundamental weaknesses in robustness. We perform a systematic vulnerability analysis by introducing controlled perturbations across seven dimensions: objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures and sensor noise. We comprehensively analyzed multiple state-of-the-art models and revealed consistent brittleness beneath apparent competence. Our analysis exposes critical weaknesses: models exhibit extreme sensitivity to perturbation factors, including camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations. Surprisingly, models are largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely. Our findings challenge the assumption that high benchmark scores equate to true competency and highlight the need for evaluation practices that assess reliability under realistic variation.
>
---
#### [replaced 043] UniTok: A Unified Tokenizer for Visual Generation and Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.20321v3](http://arxiv.org/pdf/2502.20321v3)**

> **作者:** Chuofan Ma; Yi Jiang; Junfeng Wu; Jihan Yang; Xin Yu; Zehuan Yuan; Bingyue Peng; Xiaojuan Qi
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Visual generative and understanding models typically rely on distinct tokenizers to process images, presenting a key challenge for unifying them within a single framework. Recent studies attempt to address this by connecting the training of VQVAE (for autoregressive generation) and CLIP (for understanding) to build a unified tokenizer. However, directly combining these training objectives has been observed to cause severe loss conflicts. In this paper, we show that reconstruction and semantic supervision do not inherently conflict. Instead, the underlying bottleneck stems from limited representational capacity of discrete token space. Building on these insights, we introduce UniTok, a unified tokenizer featuring a novel multi-codebook quantization mechanism that effectively scales up the vocabulary size and bottleneck dimension. In terms of final performance, UniTok sets a new record of 0.38 rFID and 78.6% zero-shot accuracy on ImageNet. Besides, UniTok can be seamlessly integrated into MLLMs to unlock native visual generation capability, without compromising the understanding performance. Additionally, we show that UniTok favors cfg-free generation, reducing gFID from 14.6 to 2.5 on ImageNet 256$\times$256 benchmark. GitHub: https://github.com/FoundationVision/UniTok.
>
---
#### [replaced 044] EditInfinity: Image Editing with Binary-Quantized Generative Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.20217v2](http://arxiv.org/pdf/2510.20217v2)**

> **作者:** Jiahuan Wang; Yuxin Chen; Jun Yu; Guangming Lu; Wenjie Pei
>
> **备注:** 28 pages, 13 figures, accepted by The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Adapting pretrained diffusion-based generative models for text-driven image editing with negligible tuning overhead has demonstrated remarkable potential. A classical adaptation paradigm, as followed by these methods, first infers the generative trajectory inversely for a given source image by image inversion, then performs image editing along the inferred trajectory guided by the target text prompts. However, the performance of image editing is heavily limited by the approximation errors introduced during image inversion by diffusion models, which arise from the absence of exact supervision in the intermediate generative steps. To circumvent this issue, we investigate the parameter-efficient adaptation of VQ-based generative models for image editing, and leverage their inherent characteristic that the exact intermediate quantized representations of a source image are attainable, enabling more effective supervision for precise image inversion. Specifically, we propose \emph{EditInfinity}, which adapts \emph{Infinity}, a binary-quantized generative model, for image editing. We propose an efficient yet effective image inversion mechanism that integrates text prompting rectification and image style preservation, enabling precise image inversion. Furthermore, we devise a holistic smoothing strategy which allows our \emph{EditInfinity} to perform image editing with high fidelity to source images and precise semantic alignment to the text prompts. Extensive experiments on the PIE-Bench benchmark across "add", "change", and "delete" editing operations, demonstrate the superior performance of our model compared to state-of-the-art diffusion-based baselines. Code available at: https://github.com/yx-chen-ust/EditInfinity.
>
---
#### [replaced 045] Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06771v3](http://arxiv.org/pdf/2412.06771v3)**

> **作者:** Meera Hahn; Wenjun Zeng; Nithish Kannen; Rich Galt; Kartikeya Badola; Been Kim; Zi Wang
>
> **摘要:** User prompts for generative AI models are often underspecified, leading to a misalignment between the user intent and models' understanding. As a result, users commonly have to painstakingly refine their prompts. We study this alignment problem in text-to-image (T2I) generation and propose a prototype for proactive T2I agents equipped with an interface to (1) actively ask clarification questions when uncertain, and (2) present their uncertainty about user intent as an understandable and editable belief graph. We build simple prototypes for such agents and propose a new scalable and automated evaluation approach using two agents, one with a ground truth intent (an image) while the other tries to ask as few questions as possible to align with the ground truth. We experiment over three image-text datasets: ImageInWords (Garg et al., 2024), COCO (Lin et al., 2014) and DesignBench, a benchmark we curated with strong artistic and design elements. Experiments over the three datasets demonstrate the proposed T2I agents' ability to ask informative questions and elicit crucial information to achieve successful alignment with at least 2 times higher VQAScore (Lin et al., 2024) than the standard T2I generation. Moreover, we conducted human studies and observed that at least 90% of human subjects found these agents and their belief graphs helpful for their T2I workflow, highlighting the effectiveness of our approach. Code and DesignBench can be found at https://github.com/google-deepmind/proactive_t2i_agents.
>
---
#### [replaced 046] IPFormer: Visual 3D Panoptic Scene Completion with Context-Adaptive Instance Proposals
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20671v2](http://arxiv.org/pdf/2506.20671v2)**

> **作者:** Markus Gross; Aya Fahmy; Danit Niwattananan; Dominik Muhle; Rui Song; Daniel Cremers; Henri Meeß
>
> **摘要:** Semantic Scene Completion (SSC) has emerged as a pivotal approach for jointly learning scene geometry and semantics, enabling downstream applications such as navigation in mobile robotics. The recent generalization to Panoptic Scene Completion (PSC) advances the SSC domain by integrating instance-level information, thereby enhancing object-level sensitivity in scene understanding. While PSC was introduced using LiDAR modality, methods based on camera images remain largely unexplored. Moreover, recent Transformer-based approaches utilize a fixed set of learned queries to reconstruct objects within the scene volume. Although these queries are typically updated with image context during training, they remain static at test time, limiting their ability to dynamically adapt specifically to the observed scene. To overcome these limitations, we propose IPFormer, the first method that leverages context-adaptive instance proposals at train and test time to address vision-based 3D Panoptic Scene Completion. Specifically, IPFormer adaptively initializes these queries as panoptic instance proposals derived from image context and further refines them through attention-based encoding and decoding to reason about semantic instance-voxel relationships. Extensive experimental results show that our approach achieves state-of-the-art in-domain performance, exhibits superior zero-shot generalization on out-of-domain data, and achieves a runtime reduction exceeding 14x. These results highlight our introduction of context-adaptive instance proposals as a pioneering effort in addressing vision-based 3D Panoptic Scene Completion.
>
---
#### [replaced 047] A Geometric Approach to Steerable Convolutions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.18813v2](http://arxiv.org/pdf/2510.18813v2)**

> **作者:** Soumyabrata Kundu; Risi Kondor
>
> **摘要:** In contrast to the somewhat abstract, group theoretical approach adopted by many papers, our work provides a new and more intuitive derivation of steerable convolutional neural networks in $d$ dimensions. This derivation is based on geometric arguments and fundamental principles of pattern matching. We offer an intuitive explanation for the appearance of the Clebsch--Gordan decomposition and spherical harmonic basis functions. Furthermore, we suggest a novel way to construct steerable convolution layers using interpolation kernels that improve upon existing implementation, and offer greater robustness to noisy data.
>
---
#### [replaced 048] SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12448v3](http://arxiv.org/pdf/2505.12448v3)**

> **作者:** Yang Liu; Ming Ma; Xiaomin Yu; Pengxiang Ding; Han Zhao; Mingyang Sun; Siteng Huang; Donglin Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Despite impressive advancements in Visual-Language Models (VLMs) for multi-modal tasks, their reliance on RGB inputs limits precise spatial understanding. Existing methods for integrating spatial cues, such as point clouds or depth, either require specialized sensors or fail to effectively exploit depth information for higher-order reasoning. To this end, we propose a novel Spatial Sense and Reasoning method, dubbed SSR, a novel framework that transforms raw depth data into structured, interpretable textual rationales. These textual rationales serve as meaningful intermediate representations to significantly enhance spatial reasoning capabilities. Additionally, we leverage knowledge distillation to compress the generated rationales into compact latent embeddings, which facilitate resource-efficient and plug-and-play integration into existing VLMs without retraining. To enable comprehensive evaluation, we introduce a new dataset named SSR-CoT, a million-scale visual-language reasoning dataset enriched with intermediate spatial reasoning annotations, and present SSRBench, a comprehensive multi-task benchmark. Extensive experiments on multiple benchmarks demonstrate SSR substantially improves depth utilization and enhances spatial reasoning, thereby advancing VLMs toward more human-like multi-modal understanding. Project page: https://yliu-cs.github.io/SSR.
>
---
#### [replaced 049] E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.12753v2](http://arxiv.org/pdf/2510.12753v2)**

> **作者:** Wenpu Li; Bangyan Liao; Yi Zhou; Qi Xu; Pian Wan; Peidong Liu
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems(NeurIPS 2025)
>
> **摘要:** The estimation of optical flow and 6-DoF ego-motion, two fundamental tasks in 3D vision, has typically been addressed independently. For neuromorphic vision (e.g., event cameras), however, the lack of robust data association makes solving the two problems separately an ill-posed challenge, especially in the absence of supervision via ground truth. Existing works mitigate this ill-posedness by either enforcing the smoothness of the flow field via an explicit variational regularizer or leveraging explicit structure-and-motion priors in the parametrization to improve event alignment. The former notably introduces bias in results and computational overhead, while the latter, which parametrizes the optical flow in terms of the scene depth and the camera motion, often converges to suboptimal local minima. To address these issues, we propose an unsupervised framework that jointly optimizes egomotion and optical flow via implicit spatial-temporal and geometric regularization. First, by modeling camera's egomotion as a continuous spline and optical flow as an implicit neural representation, our method inherently embeds spatial-temporal coherence through inductive biases. Second, we incorporate structure-and-motion priors through differential geometric constraints, bypassing explicit depth estimation while maintaining rigorous geometric consistency. As a result, our framework (called E-MoFlow) unifies egomotion and optical flow estimation via implicit regularization under a fully unsupervised paradigm. Experiments demonstrate its versatility to general 6-DoF motion scenarios, achieving state-of-the-art performance among unsupervised methods and competitive even with supervised approaches.
>
---
#### [replaced 050] Diffusing DeBias: Synthetic Bias Amplification for Model Debiasing
- **分类: cs.LG; cs.CV; I.4; I.5**

- **链接: [http://arxiv.org/pdf/2502.09564v5](http://arxiv.org/pdf/2502.09564v5)**

> **作者:** Massimiliano Ciranni; Vito Paolo Pastore; Roberto Di Via; Enzo Tartaglione; Francesca Odone; Vittorio Murino
>
> **备注:** 18 Pages, 9 Figures; Accepted at NeurIPS2025 (Poster)
>
> **摘要:** Deep learning model effectiveness in classification tasks is often challenged by the quality and quantity of training data whenever they are affected by strong spurious correlations between specific attributes and target labels. This results in a form of bias affecting training data, which typically leads to unrecoverable weak generalization in prediction. This paper aims at facing this problem by leveraging bias amplification with generated synthetic data: we introduce Diffusing DeBias (DDB), a novel approach acting as a plug-in for common methods of unsupervised model debiasing exploiting the inherent bias-learning tendency of diffusion models in data generation. Specifically, our approach adopts conditional diffusion models to generate synthetic bias-aligned images, which replace the original training set for learning an effective bias amplifier model that we subsequently incorporate into an end-to-end and a two-step unsupervised debiasing approach. By tackling the fundamental issue of bias-conflicting training samples memorization in learning auxiliary models, typical of this type of techniques, our proposed method beats current state-of-the-art in multiple benchmark datasets, demonstrating its potential as a versatile and effective tool for tackling bias in deep learning models. Code is available at https://github.com/Malga-Vision/DiffusingDeBias
>
---
#### [replaced 051] AugGen: Synthetic Augmentation using Diffusion Models Can Improve Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11544v3](http://arxiv.org/pdf/2503.11544v3)**

> **作者:** Parsa Rahimi; Damien Teney; Sebastien Marcel
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** The increasing reliance on large-scale datasets in machine learning poses significant privacy and ethical challenges, particularly in sensitive domains such as face recognition. Synthetic data generation offers a promising alternative; however, most existing methods depend heavily on external datasets or pre-trained models, increasing complexity and resource demands. In this paper, we introduce AugGen, a self-contained synthetic augmentation technique. AugGen strategically samples from a class-conditional generative model trained exclusively on the target FR dataset, eliminating the need for external resources. Evaluated across 8 FR benchmarks, including IJB-C and IJB-B, our method achieves 1-12% performance improvements, outperforming models trained solely on real data and surpassing state-of-the-art synthetic data generation approaches, while using less real data. Notably, these gains often exceed those from architectural enhancements, underscoring the value of synthetic augmentation in data-limited scenarios. Our findings demonstrate that carefully integrated synthetic data can both mitigate privacy constraints and substantially enhance recognition performance. Paper website: https://parsa-ra.github.io/auggen/.
>
---
#### [replaced 052] Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21955v2](http://arxiv.org/pdf/2505.21955v2)**

> **作者:** Insu Lee; Wooje Park; Jaeyun Jang; Minyoung Noh; Kyuhong Shim; Byonghyo Shim
>
> **备注:** Accepted to NeurIPS 2025 (Spotlight)
>
> **摘要:** Large vision-language models (LVLMs) are increasingly deployed in interactive applications such as virtual and augmented reality, where a first-person (egocentric) view captured by head-mounted cameras serves as key input. While this view offers fine-grained cues about user attention and hand-object interactions, its narrow field of view and lack of global context often lead to failures on spatially or contextually demanding queries. To address this, we introduce a framework that augments egocentric inputs with third-person (exocentric) views, providing complementary information such as global scene layout and object visibility to LVLMs. We present E3VQA, the first benchmark for multi-view question answering with 4K high-quality question-answer pairs grounded in synchronized ego-exo image pairs. Additionally, we propose M3CoT, a training-free prompting technique that constructs a unified scene representation by integrating scene graphs from three complementary perspectives. M3CoT enables LVLMs to reason more effectively across views, yielding consistent performance gains (4.84% for GPT-4o and 5.94% for Gemini 2.0 Flash) over a recent CoT baseline. Our extensive evaluation reveals key strengths and limitations of LVLMs in multi-view reasoning and highlights the value of leveraging both egocentric and exocentric inputs. The dataset and source code are available at https://github.com/Leeinsu1/Towards-Comprehensive-Scene-Understanding.
>
---
#### [replaced 053] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.19385v5](http://arxiv.org/pdf/2503.19385v5)**

> **作者:** Jaihoon Kim; Taehoon Yoon; Jisung Hwang; Minhyuk Sung
>
> **备注:** Project page: https://flow-inference-time-scaling.github.io/ (NeurIPS 2025)
>
> **摘要:** We propose an inference-time scaling approach for pretrained flow models. Recently, inference-time scaling has gained significant attention in LLMs and diffusion models, improving sample quality or better aligning outputs with user preferences by leveraging additional computation. For diffusion models, particle sampling has allowed more efficient scaling due to the stochasticity at intermediate denoising steps. On the contrary, while flow models have gained popularity as an alternative to diffusion models--offering faster generation and high-quality outputs in state-of-the-art image and video generative models--efficient inference-time scaling methods used for diffusion models cannot be directly applied due to their deterministic generative process. To enable efficient inference-time scaling for flow models, we propose three key ideas: 1) SDE-based generation, enabling particle sampling in flow models, 2) Interpolant conversion, broadening the search space and enhancing sample diversity, and 3) Rollover Budget Forcing (RBF), an adaptive allocation of computational resources across timesteps to maximize budget utilization. Our experiments show that SDE-based generation, particularly variance-preserving (VP) interpolant-based generation, improves the performance of particle sampling methods for inference-time scaling in flow models. Additionally, we demonstrate that RBF with VP-SDE achieves the best performance, outperforming all previous inference-time scaling approaches.
>
---
#### [replaced 054] MELLM: Exploring LLM-Powered Micro-Expression Understanding Enhanced by Subtle Motion Perception
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07007v2](http://arxiv.org/pdf/2505.07007v2)**

> **作者:** Sirui Zhao; Zhengye Zhang; Shifeng Liu; Xinglong Mao; Shukang Yin; Chaoyou Fu; Tong Xu; Enhong Chen
>
> **摘要:** Micro-expressions (MEs), brief and low-intensity facial movements revealing concealed emotions, are crucial for affective computing. Despite notable progress in ME recognition, existing methods are largely confined to discrete emotion classification, lacking the capacity for comprehensive ME Understanding (MEU), particularly in interpreting subtle facial dynamics and underlying emotional cues. While Multimodal Large Language Models (MLLMs) offer potential for MEU with their advanced reasoning abilities, they still struggle to perceive such subtle facial affective behaviors. To bridge this gap, we propose a ME Large Language Model (MELLM) that integrates optical flow-based sensitivity to subtle facial motions with the powerful inference ability of LLMs. Specifically, an iterative, warping-based optical-flow estimator, named MEFlowNet, is introduced to precisely capture facial micro-movements. For its training and evaluation, we construct MEFlowDataset, a large-scale optical-flow dataset with 54,611 onset-apex image pairs spanning diverse identities and subtle facial motions. Subsequently, we design a Flow-Guided Micro-Expression Understanding paradigm. Under this framework, the optical flow signals extracted by MEFlowNet are leveraged to build MEU-Instruct, an instruction-tuning dataset for MEU. MELLM is then fine-tuned on MEU-Instruct, enabling it to translate subtle motion patterns into human-readable descriptions and generate corresponding emotional inferences. Experiments demonstrate that MEFlowNet significantly outperforms existing optical flow methods in facial and ME-flow estimation, while MELLM achieves state-of-the-art accuracy and generalization across multiple ME benchmarks. To the best of our knowledge, this work presents two key contributions: MEFlowNet as the first dedicated ME flow estimator, and MELLM as the first LLM tailored for MEU.
>
---
#### [replaced 055] CAR-Flow: Condition-Aware Reparameterization Aligns Source and Target for Better Flow Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.19300v2](http://arxiv.org/pdf/2509.19300v2)**

> **作者:** Chen Chen; Pengsheng Guo; Liangchen Song; Jiasen Lu; Rui Qian; Xinze Wang; Tsu-Jui Fu; Wei Liu; Yinfei Yang; Alex Schwing
>
> **摘要:** Conditional generative modeling aims to learn a conditional data distribution from samples containing data-condition pairs. For this, diffusion and flow-based methods have attained compelling results. These methods use a learned (flow) model to transport an initial standard Gaussian noise that ignores the condition to the conditional data distribution. The model is hence required to learn both mass transport and conditional injection. To ease the demand on the model, we propose Condition-Aware Reparameterization for Flow Matching (CAR-Flow) -- a lightweight, learned shift that conditions the source, the target, or both distributions. By relocating these distributions, CAR-Flow shortens the probability path the model must learn, leading to faster training in practice. On low-dimensional synthetic data, we visualize and quantify the effects of CAR-Flow. On higher-dimensional natural image data (ImageNet-256), equipping SiT-XL/2 with CAR-Flow reduces FID from 2.07 to 1.68, while introducing less than 0.6% additional parameters.
>
---
#### [replaced 056] Video-Skill-CoT: Skill-based Chain-of-Thoughts for Domain-Adaptive Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03525v2](http://arxiv.org/pdf/2506.03525v2)**

> **作者:** Daeun Lee; Jaehong Yoon; Jaemin Cho; Mohit Bansal
>
> **备注:** Project website: https://video-skill-cot.github.io/
>
> **摘要:** Recent advances in Chain-of-Thought (CoT) reasoning have improved complex video understanding, but existing methods often struggle to adapt to domain-specific skills (e.g., event detection, spatial relation understanding, emotion understanding) over various video content. To address this, we propose Video-Skill-CoT (a.k.a. Video-SKoT), a framework that automatically constructs and leverages skill-aware CoT supervisions for domain-adaptive video reasoning. First, we construct skill-based CoT annotations: we extract domain-relevant reasoning skills from training questions, cluster them into a shared skill taxonomy, and create detailed multi-step CoT rationale tailored to each video-question pair for training. Second, we introduce a skill-specific expert learning framework. Each expert module specializes in a subset of reasoning skills and is trained with lightweight adapters using the collected CoT supervision. We demonstrate the effectiveness of the proposed approach on three video understanding benchmarks, where Video-SKoT consistently outperforms strong baselines. We also provide in-depth analyses on comparing different CoT annotation pipelines and learned skills over multiple video domains.
>
---
#### [replaced 057] HAVT-IVD: Heterogeneity-Aware Cross-Modal Network for Audio-Visual Surveillance: Idling Vehicles Detection With Multichannel Audio and Multiscale Visual Cues
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.16102v2](http://arxiv.org/pdf/2504.16102v2)**

> **作者:** Xiwen Li; Xiaoya Tang; Tolga Tasdizen
>
> **摘要:** Idling vehicle detection (IVD) uses surveillance video and multichannel audio to localize and classify vehicles in the last frame as moving, idling, or engine-off in pick-up zones. IVD faces three challenges: (i) modality heterogeneity between visual cues and audio patterns; (ii) large box scale variation requiring multi-resolution detection; and (iii) training instability due to coupled detection heads. The previous end-to-end (E2E) model with simple CBAM-based bi-modal attention fails to handle these issues and often misses vehicles. We propose HAVT-IVD, a heterogeneity-aware network with a visual feature pyramid and decoupled heads. Experiments show HAVT-IVD improves mAP by 7.66 over the disjoint baseline and 9.42 over the E2E baseline.
>
---
#### [replaced 058] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18812v2](http://arxiv.org/pdf/2505.18812v2)**

> **作者:** Ye Sun; Hao Zhang; Henghui Ding; Tiehua Zhang; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Achieving fine-grained spatio-temporal understanding in videos remains a major challenge for current Video Large Multimodal Models (Video LMMs). Addressing this challenge requires mastering two core capabilities: video referring understanding, which captures the semantics of video regions, and video grounding, which segments object regions based on natural language descriptions. However, most existing approaches tackle these tasks in isolation, limiting progress toward unified, referentially grounded video interaction. We identify a key bottleneck in the lack of high-quality, unified video instruction data and a comprehensive benchmark for evaluating referentially grounded video chat. To address these challenges, we contribute in three core aspects: dataset, model, and benchmark. First, we introduce SAMA-239K, a large-scale dataset comprising 15K videos specifically curated to enable joint learning of video referring understanding, grounding, and multi-turn video chat. Second, we propose the SAMA model, which incorporates a versatile spatio-temporal context aggregator and a Segment Anything Model to jointly enhance fine-grained video comprehension and precise grounding capabilities. Finally, we establish SAMA-Bench, a meticulously designed benchmark consisting of 5,067 questions from 522 videos, to comprehensively evaluate the integrated capabilities of Video LMMs in multi-turn, spatio-temporal referring understanding and grounded dialogue. Extensive experiments and benchmarking results show that SAMA not only achieves strong performance on SAMA-Bench but also sets a new state-of-the-art on general grounding benchmarks, while maintaining highly competitive performance on standard visual understanding benchmarks.
>
---
#### [replaced 059] Seeing the Arrow of Time in Large Multimodal Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03340v2](http://arxiv.org/pdf/2506.03340v2)**

> **作者:** Zihui Xue; Mi Luo; Kristen Grauman
>
> **备注:** Accepted by NeurIPS 2025, Project website: https://vision.cs.utexas.edu/projects/SeeAoT
>
> **摘要:** The Arrow of Time (AoT)-time's irreversible flow shaping physical events-is fundamental to video comprehension, yet remains a significant challenge for modern large multimodal models (LMMs). Current LMMs struggle to perceive and utilize temporal directionality in video when responding to language queries, obstructing deeper temporal understanding. We tackle this deficiency by first providing a critical analysis of existing benchmarks and models. We then introduce ArrowRL, a reinforcement learning (RL)-based training strategy with an innovative reverse reward that instills AoT awareness by encouraging divergent video interpretations between forward and reversed visual frames. For rigorous evaluation, we additionally develop AoTBench, a new multi-faceted benchmark probing temporally challenging questions. Experiments show ArrowRL greatly advances temporal perception: it not only achieves substantial improvements on our challenging AoTBench but also demonstrably boosts performance on standard video question answering (VQA) benchmarks (with peak accuracy gains reaching over 20% and 10% respectively). This validates ArrowRL's effectiveness and highlights the critical need for dedicated AoT understanding in LMMs.
>
---
#### [replaced 060] ScoreMix: Synthetic Data Generation by Score Composition in Diffusion Models Improves Recognition
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.10226v2](http://arxiv.org/pdf/2506.10226v2)**

> **作者:** Parsa Rahimi; Sebastien Marcel
>
> **备注:** Extended version of ICMLw25 Oral
>
> **摘要:** Synthetic data generation is increasingly used in machine learning for training and data augmentation. Yet, current strategies often rely on external foundation models or datasets, whose usage is restricted in many scenarios due to policy or legal constraints. We propose ScoreMix, a self-contained synthetic generation method to produce hard synthetic samples for recognition tasks by leveraging the score compositionality of diffusion models. The approach mixes class-conditioned scores along reverse diffusion trajectories, yielding domain-specific data augmentation without external resources. We systematically study class-selection strategies and find that mixing classes distant in the discriminator's embedding space yields larger gains, providing up to 3% additional average improvement, compared to selection based on proximity. Interestingly, we observe that condition and embedding spaces are largely uncorrelated under standard alignment metrics, and the generator's condition space has a negligible effect on downstream performance. Across 8 public face recognition benchmarks, ScoreMix improves accuracy by up to 7 percentage points, without hyperparameter search, highlighting both robustness and practicality. Our method provides a simple yet effective way to maximize discriminator performance using only the available dataset, without reliance on third-party resources. Paper website: https://parsa-ra.github.io/scoremix/.
>
---
#### [replaced 061] Total Generalized Variation of the Normal Vector Field and Applications to Mesh Denoising
- **分类: cs.CV; math.DG; math.OC**

- **链接: [http://arxiv.org/pdf/2507.13530v2](http://arxiv.org/pdf/2507.13530v2)**

> **作者:** Lukas Baumgärtner; Ronny Bergmann; Roland Herzog; Stephan Schmidt; Manuel Weiß
>
> **摘要:** We propose a novel formulation for the second-order total generalized variation (TGV) of the normal vector on an oriented, triangular mesh embedded in $\R^3$. The normal vector is considered as a manifold-valued function, taking values on the unit sphere. Our formulation extends previous discrete TGV models for piecewise constant scalar data that utilize a Raviart-Thomas function space. To extend this formulation to the manifold setting, a tailor-made tangential Raviart-Thomas type finite element space is constructed in this work. The new regularizer is compared to existing methods in mesh denoising experiments.
>
---
#### [replaced 062] Size and Smoothness Aware Adaptive Focal Loss for Small Tumor Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.09828v2](http://arxiv.org/pdf/2407.09828v2)**

> **作者:** Md Rakibul Islam; Riad Hassan; Abdullah Nazib; Kien Nguyen; Clinton Fookes; Md Zahidul Islam
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Deep learning has achieved remarkable accuracy in medical image segmentation, particularly for larger structures with well-defined boundaries. However, its effectiveness can be challenged by factors such as irregular object shapes and edges, non-smooth surfaces, small target areas, etc. which complicate the ability of networks to grasp the intricate and diverse nature of anatomical regions. In response to these challenges, we propose an Adaptive Focal Loss (A-FL) that takes both object boundary smoothness and size into account, with the goal to improve segmentation performance in intricate anatomical regions. The proposed A-FL dynamically adjusts itself based on an object's surface smoothness, size, and the class balancing parameter based on the ratio of targeted area and background. We evaluated the performance of the A-FL on the PICAI 2022 and BraTS 2018 datasets. In the PICAI 2022 dataset, the A-FL achieved an Intersection over Union (IoU) score of 0.696 and a Dice Similarity Coefficient (DSC) of 0.769, outperforming the regular Focal Loss (FL) by 5.5% and 5.4% respectively. It also surpassed the best baseline by 2.0% and 1.2%. In the BraTS 2018 dataset, A-FL achieved an IoU score of 0.883 and a DSC score of 0.931. Our ablation experiments also show that the proposed A-FL surpasses conventional losses (this includes Dice Loss, Focal Loss, and their hybrid variants) by large margin in IoU, DSC, and other metrics. The code is available at https://github.com/rakibuliuict/AFL-CIBM.git.
>
---
#### [replaced 063] Adaptive Non-uniform Timestep Sampling for Accelerating Diffusion Model Training
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.09998v2](http://arxiv.org/pdf/2411.09998v2)**

> **作者:** Myunsoo Kim; Donghyeon Ki; Seong-Woong Shim; Byung-Jun Lee
>
> **摘要:** As a highly expressive generative model, diffusion models have demonstrated exceptional success across various domains, including image generation, natural language processing, and combinatorial optimization. However, as data distributions grow more complex, training these models to convergence becomes increasingly computationally intensive. While diffusion models are typically trained using uniform timestep sampling, our research shows that the variance in stochastic gradients varies significantly across timesteps, with high-variance timesteps becoming bottlenecks that hinder faster convergence. To address this issue, we introduce a non-uniform timestep sampling method that prioritizes these more critical timesteps. Our method tracks the impact of gradient updates on the objective for each timestep, adaptively selecting those most likely to minimize the objective effectively. Experimental results demonstrate that this approach not only accelerates the training process, but also leads to improved performance at convergence. Furthermore, our method shows robust performance across various datasets, scheduling strategies, and diffusion architectures, outperforming previously proposed timestep sampling and weighting heuristics that lack this degree of robustness.
>
---
#### [replaced 064] Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive Regression
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.03674v2](http://arxiv.org/pdf/2501.03674v2)**

> **作者:** Mengshi Qi; Hao Ye; Jiaxuan Peng; Huadong Ma
>
> **摘要:** Action Quality Assessment (AQA), which aims at automatic and fair evaluation of athletic performance, has gained increasing attention in recent years. However, athletes are often in rapid movement and the corresponding visual appearance variances are subtle, making it challenging to capture fine-grained pose differences and leading to poor estimation performance. Furthermore, most common AQA tasks, such as diving in sports, are usually divided into multiple sub-actions, each of which contains different durations. However, existing methods focus on segmenting the video into fixed frames, which disrupts the temporal continuity of sub-actions resulting in unavoidable prediction errors. To address these challenges, we propose a novel action quality assessment method through hierarchically pose-guided multi-stage contrastive regression. Firstly, we introduce a multi-scale dynamic visual-skeleton encoder to capture fine-grained spatio-temporal visual and skeletal features. Then, a procedure segmentation network is introduced to separate different sub-actions and obtain segmented features. Afterwards, the segmented visual and skeletal features are both fed into a multi-modal fusion module as physics structural priors, to guide the model in learning refined activity similarities and variances. Finally, a multi-stage contrastive learning regression approach is employed to learn discriminative representations and output prediction results. In addition, we introduce a newly-annotated FineDiving-Pose Dataset to improve the current low-quality human pose labels. In experiments, the results on FineDiving and MTL-AQA datasets demonstrate the effectiveness and superiority of our proposed approach. Our source code and dataset are available at https://github.com/Lumos0507/HP-MCoRe.
>
---
#### [replaced 065] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.02489v2](http://arxiv.org/pdf/2506.02489v2)**

> **作者:** Tao Zhong; Jonah Buchanan; Christine Allen-Blanchette
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** We propose a new approach to vision-based dexterous grasp translation, which aims to transfer grasp intent across robotic hands with differing morphologies. Given a visual observation of a source hand grasping an object, our goal is to synthesize a functionally equivalent grasp for a target hand without requiring paired demonstrations or hand-specific simulations. We frame this problem as a stochastic transport between grasp distributions using the Schr\"odinger Bridge formalism. Our method learns to map between source and target latent grasp spaces via score and flow matching, conditioned on visual observations. To guide this translation, we introduce physics-informed cost functions that encode alignment in base pose, contact maps, wrench space, and manipulability. Experiments across diverse hand-object pairs demonstrate our approach generates stable, physically grounded grasps with strong generalization. This work enables semantic grasp transfer for heterogeneous manipulators and bridges vision-based grasping with probabilistic generative modeling. Additional details at https://grasp2grasp.github.io/
>
---
#### [replaced 066] Frame In-N-Out: Unbounded Controllable Image-to-Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21491v2](http://arxiv.org/pdf/2505.21491v2)**

> **作者:** Boyang Wang; Xuweiyi Chen; Matheus Gadelha; Zezhou Cheng
>
> **摘要:** Controllability, temporal coherence, and detail synthesis remain the most critical challenges in video generation. In this paper, we focus on a commonly used yet underexplored cinematic technique known as Frame In and Frame Out. Specifically, starting from image-to-video generation, users can control the objects in the image to naturally leave the scene or provide breaking new identity references to enter the scene, guided by a user-specified motion trajectory. To support this task, we introduce a new dataset that is curated semi-automatically, an efficient identity-preserving motion-controllable video Diffusion Transformer architecture, and a comprehensive evaluation protocol targeting this task. Our evaluation shows that our proposed approach significantly outperforms existing baselines.
>
---
#### [replaced 067] Text-conditioned State Space Model For Domain-generalized Change Detection Visual Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.08974v3](http://arxiv.org/pdf/2508.08974v3)**

> **作者:** Elman Ghazaei; Erchan Aptoula
>
> **摘要:** The Earth's surface is constantly changing, and detecting these changes provides valuable insights that benefit various aspects of human society. While traditional change detection methods have been employed to detect changes from bi-temporal images, these approaches typically require expert knowledge for accurate interpretation. To enable broader and more flexible access to change information by non-expert users, the task of Change Detection Visual Question Answering (CDVQA) has been introduced. However, existing CDVQA methods have been developed under the assumption that training and testing datasets share similar distributions. This assumption does not hold in real-world applications, where domain shifts often occur. In this paper, the CDVQA task is revisited with a focus on addressing domain shift. To this end, a new multi-modal and multi-domain dataset, BrightVQA, is introduced to facilitate domain generalization research in CDVQA. Furthermore, a novel state space model, termed Text-Conditioned State Space Model (TCSSM), is proposed. The TCSSM framework is designed to leverage both bi-temporal imagery and geo-disaster-related textual information in an unified manner to extract domain-invariant features across domains. Input-dependent parameters existing in TCSSM are dynamically predicted by using both bi-temporal images and geo-disaster-related description, thereby facilitating the alignment between bi-temporal visual data and the associated textual descriptions. Extensive experiments are conducted to evaluate the proposed method against state-of-the-art models, and superior performance is consistently demonstrated. The code and dataset will be made publicly available upon acceptance at https://github.com/Elman295/TCSSM.
>
---
#### [replaced 068] Approximating Signed Distance Fields of Implicit Surfaces with Sparse Ellipsoidal Radial Basis Function Networks
- **分类: cs.GR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02350v3](http://arxiv.org/pdf/2505.02350v3)**

> **作者:** Bobo Lian; Dandan Wang; Chenjian Wu; Minxin Chen
>
> **摘要:** Accurate and compact representation of signed distance functions (SDFs) of implicit surfaces is crucial for efficient storage, computation, and downstream processing of 3D geometry. In this work, we propose a general learning method for approximating precomputed SDF fields of implicit surfaces by a relatively small number of ellipsoidal radial basis functions (ERBFs). The SDF values could be computed from various sources, including point clouds, triangle meshes, analytical expressions, pretrained neural networks, etc. Given SDF values on spatial grid points, our method approximates the SDF using as few ERBFs as possible, achieving a compact representation while preserving the geometric shape of the corresponding implicit surface. To balance sparsity and approximation precision, we introduce a dynamic multi-objective optimization strategy, which adaptively incorporates regularization to enforce sparsity and jointly optimizes the weights, centers, shapes, and orientations of the ERBFs. For computational efficiency, a nearest-neighbor-based data structure restricts computations to points near each kernel center, and CUDA-based parallelism further accelerates the optimization. Furthermore, a hierarchical refinement strategy based on SDF spatial grid points progressively incorporates coarse-to-fine samples for parameter initialization and optimization, improving convergence and training efficiency. Extensive experiments on multiple benchmark datasets demonstrate that our method can represent SDF fields with significantly fewer parameters than existing sparse implicit representation approaches, achieving better accuracy, robustness, and computational efficiency. The corresponding executable program is publicly available at https://github.com/lianbobo/SE-RBFNet.git
>
---
#### [replaced 069] ViTime: Foundation Model for Time Series Forecasting Powered by Vision Intelligence
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.07311v4](http://arxiv.org/pdf/2407.07311v4)**

> **作者:** Luoxiao Yang; Yun Wang; Xinqi Fan; Israel Cohen; Jingdong Chen; Zijun Zhang
>
> **摘要:** Time series forecasting (TSF) possesses great practical values in various fields, including power and energy, transportation, etc. TSF methods have been studied based on knowledge from classical statistics to modern deep learning. Yet, all of them were developed based on one fundamental concept, the numerical data fitting. Thus, the models developed have long been known to be problem-specific and lacking application generalizability. Practitioners expect a TSF foundation model that serves TSF tasks in different applications. The central question is then how to develop such a TSF foundation model. This paper offers one pioneering study in the TSF foundation model development method and proposes a vision intelligence-powered framework, ViTime, for the first time. ViTime fundamentally shifts TSF from numerical fitting to operations based on a binary image-based time series metric space and naturally supports both point and probabilistic forecasting. We also provide rigorous theoretical analyses of ViTime, including quantization-induced system error bounds and principled strategies for optimal parameter selection. Furthermore, we propose RealTS, an innovative synthesis algorithm generating diverse and realistic training samples, effectively enriching the training data and significantly enhancing model generalizability. Extensive experiments demonstrate ViTime's state-of-the-art performance. In zero-shot scenarios, ViTime outperforms TimesFM by 9-15\%. With just 10\% fine-tuning data, ViTime surpasses both leading foundation models and fully-supervised benchmarks, a gap that widens with 100\% fine-tuning. ViTime also exhibits exceptional robustness, effectively handling missing data and outperforming TimesFM by 20-30\% under various data perturbations, validating the power of its visual space data operation paradigm.
>
---
#### [replaced 070] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.01957v4](http://arxiv.org/pdf/2501.01957v4)**

> **作者:** Chaoyou Fu; Haojia Lin; Xiong Wang; Yi-Fan Zhang; Yunhang Shen; Xiaoyu Liu; Haoyu Cao; Zuwei Long; Heting Gao; Ke Li; Long Ma; Xiawu Zheng; Rongrong Ji; Xing Sun; Caifeng Shan; Ran He
>
> **备注:** NeurIPS 2025 Spotlight, Code 2.4K Stars: https://github.com/VITA-MLLM/VITA
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) have typically focused on integrating visual and textual modalities, with less emphasis placed on the role of speech in enhancing interaction. However, speech plays a crucial role in multimodal dialogue systems, and implementing high-performance in both vision and speech tasks remains a significant challenge due to the fundamental modality differences. In this paper, we propose a carefully designed multi-stage training methodology that progressively trains LLM to understand both visual and speech information, ultimately enabling fluent vision and speech interaction. Our approach not only preserves strong vision-language capacity, but also enables efficient speech-to-speech dialogue capabilities without separate ASR and TTS modules, significantly accelerating multimodal end-to-end response speed. By comparing our method against state-of-the-art counterparts across benchmarks for image, video, and speech tasks, we demonstrate that our model is equipped with both strong visual and speech capabilities, making near real-time vision and speech interaction. Code has been released at https://github.com/VITA-MLLM/VITA.
>
---
#### [replaced 071] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI models in Sound Localization
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.11217v2](http://arxiv.org/pdf/2505.11217v2)**

> **作者:** Yanhao Jia; Ji Xie; S Jivaganesh; Hao Li; Xu Wu; Mengmi Zhang
>
> **备注:** NeurIPS 2025, Spotlight
>
> **摘要:** Imagine hearing a dog bark and turning toward the sound only to see a parked car, while the real, silent dog sits elsewhere. Such sensory conflicts test perception, yet humans reliably resolve them by prioritizing sound over misleading visuals. Despite advances in multimodal AI integrating vision and audio, little is known about how these systems handle cross-modal conflicts or whether they favor one modality. In this study, we systematically examine modality bias and conflict resolution in AI sound localization. We assess leading multimodal models and benchmark them against human performance in psychophysics experiments across six audiovisual conditions, including congruent, conflicting, and absent cues. Humans consistently outperform AI, demonstrating superior resilience to conflicting or missing visuals by relying on auditory information. In contrast, AI models often default to visual input, degrading performance to near chance levels. To address this, we propose a neuroscience-inspired model, EchoPin, which uses a stereo audio-image dataset generated via 3D simulations. Even with limited training data, EchoPin surpasses existing benchmarks. Notably, it also mirrors human-like horizontal localization bias favoring left-right precision-likely due to the stereo audio structure reflecting human ear placement. These findings underscore how sensory input quality and system architecture shape multimodal representation accuracy.
>
---
#### [replaced 072] Rectified Point Flow: Generic Point Cloud Pose Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.05282v2](http://arxiv.org/pdf/2506.05282v2)**

> **作者:** Tao Sun; Liyuan Zhu; Shengyu Huang; Shuran Song; Iro Armeni
>
> **备注:** NeurIPS 2025 Camera-ready. Project page: https://rectified-pointflow.github.io/
>
> **摘要:** We introduce Rectified Point Flow, a unified parameterization that formulates pairwise point cloud registration and multi-part shape assembly as a single conditional generative problem. Given unposed point clouds, our method learns a continuous point-wise velocity field that transports noisy points toward their target positions, from which part poses are recovered. In contrast to prior work that regresses part-wise poses with ad-hoc symmetry handling, our method intrinsically learns assembly symmetries without symmetry labels. Together with a self-supervised encoder focused on overlapping points, our method achieves a new state-of-the-art performance on six benchmarks spanning pairwise registration and shape assembly. Notably, our unified formulation enables effective joint training on diverse datasets, facilitating the learning of shared geometric priors and consequently boosting accuracy. Project page: https://rectified-pointflow.github.io/.
>
---
#### [replaced 073] SPAN: Continuous Modeling of Suspicion Progression for Temporal Intention Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.20189v2](http://arxiv.org/pdf/2510.20189v2)**

> **作者:** Xinyi Hu; Yuran Wang; Ruixu Zhang; Yue Li; Wenxuan Liu; Zheng Wang
>
> **摘要:** Temporal Intention Localization (TIL) is crucial for video surveillance, focusing on identifying varying levels of suspicious intentions to improve security monitoring. However, existing discrete classification methods fail to capture the continuous nature of suspicious intentions, limiting early intervention and explainability. In this paper, we propose the Suspicion Progression Analysis Network (SPAN), which shifts from discrete classification to continuous regression, enabling the capture of fluctuating and evolving suspicious intentions. We reveal that suspicion exhibits long-term dependencies and cumulative effects, similar to Temporal Point Process (TPP) theory. Based on these insights, we define a suspicion score formula that models continuous changes while accounting for temporal characteristics. We also introduce Suspicion Coefficient Modulation, which adjusts suspicion coefficients using multimodal information to reflect the varying impacts of suspicious actions. Additionally, the Concept-Anchored Mapping method is proposed to link suspicious actions to predefined intention concepts, offering insights into both the actions and their potential underlying intentions. Extensive experiments on the HAI dataset show that SPAN significantly outperforms existing methods, reducing MSE by 19.8% and improving average mAP by 1.78%. Notably, SPAN achieves a 2.74% mAP gain in low-frequency cases, demonstrating its superior ability to capture subtle behavioral changes. Compared to discrete classification systems, our continuous suspicion modeling approach enables earlier detection and proactive intervention, greatly enhancing system explainability and practical utility in security applications.
>
---
#### [replaced 074] LEGNet: A Lightweight Edge-Gaussian Network for Low-Quality Remote Sensing Image Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14012v3](http://arxiv.org/pdf/2503.14012v3)**

> **作者:** Wei Lu; Si-Bao Chen; Hui-Dong Li; Qing-Ling Shu; Chris H. Q. Ding; Jin Tang; Bin Luo
>
> **备注:** 19 pages, 9 figures. Accepted by ICCV 2025 Workshop
>
> **摘要:** Remote sensing object detection (RSOD) often suffers from degradations such as low spatial resolution, sensor noise, motion blur, and adverse illumination. These factors diminish feature distinctiveness, leading to ambiguous object representations and inadequate foreground-background separation. Existing RSOD methods exhibit limitations in robust detection of low-quality objects. To address these pressing challenges, we introduce LEGNet, a lightweight backbone network featuring a novel Edge-Gaussian Aggregation (EGA) module specifically engineered to enhance feature representation derived from low-quality remote sensing images. EGA module integrates: (a) orientation-aware Scharr filters to sharpen crucial edge details often lost in low-contrast or blurred objects, and (b) Gaussian-prior-based feature refinement to suppress noise and regularize ambiguous feature responses, enhancing foreground saliency under challenging conditions. EGA module alleviates prevalent problems in reduced contrast, structural discontinuities, and ambiguous feature responses prevalent in degraded images, effectively improving model robustness while maintaining computational efficiency. Comprehensive evaluations across five benchmarks (DOTA-v1.0, v1.5, DIOR-R, FAIR1M-v1.0, and VisDrone2019) demonstrate that LEGNet achieves state-of-the-art performance, particularly in detecting low-quality objects.The code is available at https://github.com/AeroVILab-AHU/LEGNet.
>
---
#### [replaced 075] A robust and versatile deep learning model for prediction of the arterial input function in dynamic small animal $\left[^{18}\text{F}\right]$FDG PET imaging
- **分类: eess.IV; cs.CV; physics.med-ph; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2507.02367v2](http://arxiv.org/pdf/2507.02367v2)**

> **作者:** Christian Salomonsen; Luigi T Luppino; Fredrik Aspheim; Kristoffer K. Wickstrøm; Elisabeth Wetzer; Michael C. Kampffmeyer; Rodrigo Berzaghi; Rune Sundset; Robert Jenssen; Samuel Kuttner
>
> **备注:** 21 pages, 14 figures
>
> **摘要:** Dynamic positron emission tomography (PET) and kinetic modeling are pivotal in advancing tracer development research in small animal studies. Accurate kinetic modeling requires precise input function estimation, traditionally achieved via arterial blood sampling. However, arterial cannulation in small animals like mice, involves intricate, time-consuming, and terminal procedures, precluding longitudinal studies. This work proposes a non-invasive, fully convolutional deep learning-based approach (FC-DLIF) to predict input functions directly from PET imaging, potentially eliminating the need for blood sampling in dynamic small-animal PET. The proposed FC-DLIF model includes a spatial feature extractor acting on the volumetric time frames of the PET sequence, extracting spatial features. These are subsequently further processed in a temporal feature extractor that predicts the arterial input function. The proposed approach is trained and evaluated using images and arterial blood curves from [$^{18}$F]FDG data using cross validation. Further, the model applicability is evaluated on imaging data and arterial blood curves collected using two additional radiotracers ([$^{18}$F]FDOPA, and [$^{68}$Ga]PSMA). The model was further evaluated on data truncated and shifted in time, to simulate shorter, and shifted, PET scans. The proposed FC-DLIF model reliably predicts the arterial input function with respect to mean squared error and correlation. Furthermore, the FC-DLIF model is able to predict the arterial input function even from truncated and shifted samples. The model fails to predict the AIF from samples collected using different radiotracers, as these are not represented in the training data. Our deep learning-based input function offers a non-invasive and reliable alternative to arterial blood sampling, proving robust and flexible to temporal shifts and different scan durations.
>
---
#### [replaced 076] Register and [CLS] tokens yield a decoupling of local and global features in large ViTs
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05892v2](http://arxiv.org/pdf/2505.05892v2)**

> **作者:** Alexander Lappe; Martin A. Giese
>
> **摘要:** Recent work has shown that the attention maps of the widely popular DINOv2 model exhibit artifacts, which hurt both model interpretability and performance on dense image tasks. These artifacts emerge due to the model repurposing patch tokens with redundant local information for the storage of global image information. To address this problem, additional register tokens have been incorporated in which the model can store such information instead. We carefully examine the influence of these register tokens on the relationship between global and local image features, showing that while register tokens yield cleaner attention maps, these maps do not accurately reflect the integration of local image information in large models. Instead, global information is dominated by information extracted from register tokens, leading to a disconnect between local and global features. Inspired by these findings, we show that the [CLS] token itself leads to a very similar phenomenon in models without explicit register tokens. Our work shows that care must be taken when interpreting attention maps of large ViTs. Further, by clearly attributing the faulty behavior to register and [CLS] tokens, we show a path towards more interpretable vision models.
>
---
#### [replaced 077] RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.16500v2](http://arxiv.org/pdf/2509.16500v2)**

> **作者:** Tianyi Yan; Wencheng Han; Xia Zhou; Xueyang Zhang; Kun Zhan; Cheng-zhong Xu; Jianbing Shen
>
> **备注:** NeurIPS 2025
>
> **摘要:** Synthetic data is crucial for advancing autonomous driving (AD) systems, yet current state-of-the-art video generation models, despite their visual realism, suffer from subtle geometric distortions that limit their utility for downstream perception tasks. We identify and quantify this critical issue, demonstrating a significant performance gap in 3D object detection when using synthetic versus real data. To address this, we introduce Reinforcement Learning with Geometric Feedback (RLGF), RLGF uniquely refines video diffusion models by incorporating rewards from specialized latent-space AD perception models. Its core components include an efficient Latent-Space Windowing Optimization technique for targeted feedback during diffusion, and a Hierarchical Geometric Reward (HGR) system providing multi-level rewards for point-line-plane alignment, and scene occupancy coherence. To quantify these distortions, we propose GeoScores. Applied to models like DiVE on nuScenes, RLGF substantially reduces geometric errors (e.g., VP error by 21\%, Depth error by 57\%) and dramatically improves 3D object detection mAP by 12.7\%, narrowing the gap to real-data performance. RLGF offers a plug-and-play solution for generating geometrically sound and reliable synthetic videos for AD development.
>
---
#### [replaced 078] Mixture of Experts in Image Classification: What's the Sweet Spot?
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.18322v2](http://arxiv.org/pdf/2411.18322v2)**

> **作者:** Mathurin Videau; Alessandro Leite; Marc Schoenauer; Olivier Teytaud
>
> **备注:** Published in Transactions on Machine Learning Research
>
> **摘要:** Mixture-of-Experts (MoE) models have shown promising potential for parameter-efficient scaling across domains. However, their application to image classification remains limited, often requiring billion-scale datasets to be competitive. In this work, we explore the integration of MoE layers into image classification architectures using open datasets. We conduct a systematic analysis across different MoE configurations and model scales. We find that moderate parameter activation per sample provides the best trade-off between performance and efficiency. However, as the number of activated parameters increases, the benefits of MoE diminish. Our analysis yields several practical insights for vision MoE design. First, MoE layers most effectively strengthen tiny and mid-sized models, while gains taper off for large-capacity networks and do not redefine state-of-the-art ImageNet performance. Second, a Last-2 placement heuristic offers the most robust cross-architecture choice, with Every-2 slightly better for Vision Transform (ViT), and both remaining effective as data and model scale increase. Third, larger datasets (e.g., ImageNet-21k) allow more experts, up to 16, for ConvNeXt to be utilized effectively without changing placement, as increased data reduces overfitting and promotes broader expert specialization. Finally, a simple linear router performs best, suggesting that additional routing complexity yields no consistent benefit.
>
---
#### [replaced 079] Two Causally Related Needles in a Video Haystack
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19853v2](http://arxiv.org/pdf/2505.19853v2)**

> **作者:** Miaoyu Li; Qin Chao; Boyang Li
>
> **备注:** Accepted to NeurIPS 2025 D&B Track
>
> **摘要:** Properly evaluating the ability of Video-Language Models (VLMs) to understand long videos remains a challenge. We propose a long-context video understanding benchmark, Causal2Needles, that assesses two crucial abilities insufficiently addressed by existing benchmarks: (1) extracting information from two separate locations (two needles) in a long video and understanding them jointly, and (2) modeling the world in terms of cause and effect in human behaviors. Causal2Needles evaluates these abilities using noncausal one-needle, causal one-needle, and causal two-needle questions. The most complex question type, causal two-needle questions, require extracting information from both the cause and effect events from a long video and the associated narration text. To prevent textual bias, we introduce two complementary question formats: locating the video clip containing the answer, and verbal description of a visual detail from that video clip. Our experiments reveal that models excelling on existing benchmarks struggle with causal 2-needle questions, and the model performance is negatively correlated with the distance between the two needles. These findings highlight critical limitations in current VLMs. The dataset is available at: https://huggingface.co/datasets/causal2needles/Causal2Needles
>
---
#### [replaced 080] SegMASt3R: Geometry Grounded Segment Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.05051v2](http://arxiv.org/pdf/2510.05051v2)**

> **作者:** Rohit Jayanti; Swayam Agrawal; Vansh Garg; Siddharth Tourani; Muhammad Haris Khan; Sourav Garg; Madhava Krishna
>
> **备注:** Accepted to The 39th Annual Conference on Neural Information Processing Systems (NeurIPS 2025) as a Spotlight (top 3.5%)
>
> **摘要:** Segment matching is an important intermediate task in computer vision that establishes correspondences between semantically or geometrically coherent regions across images. Unlike keypoint matching, which focuses on localized features, segment matching captures structured regions, offering greater robustness to occlusions, lighting variations, and viewpoint changes. In this paper, we leverage the spatial understanding of 3D foundation models to tackle wide-baseline segment matching, a challenging setting involving extreme viewpoint shifts. We propose an architecture that uses the inductive bias of these 3D foundation models to match segments across image pairs with up to 180 degree view-point change rotation. Extensive experiments show that our approach outperforms state-of-the-art methods, including the SAM2 video propagator and local feature matching methods, by up to 30% on the AUPRC metric, on ScanNet++ and Replica datasets. We further demonstrate benefits of the proposed model on relevant downstream tasks, including 3D instance mapping and object-relative navigation. Project Page: https://segmast3r.github.io/
>
---
#### [replaced 081] Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2504.14882v2](http://arxiv.org/pdf/2504.14882v2)**

> **作者:** Mojtaba Kolahdouzi; Hatice Gunes; Ali Etemad
>
> **备注:** Accepted in NeurIPS 2025
>
> **摘要:** We study whether and how the choice of optimization algorithm can impact group fairness in deep neural networks. Through stochastic differential equation analysis of optimization dynamics in an analytically tractable setup, we demonstrate that the choice of optimization algorithm indeed influences fairness outcomes, particularly under severe imbalance. Furthermore, we show that when comparing two categories of optimizers, adaptive methods and stochastic methods, RMSProp (from the adaptive category) has a higher likelihood of converging to fairer minima than SGD (from the stochastic category). Building on this insight, we derive two new theoretical guarantees showing that, under appropriate conditions, RMSProp exhibits fairer parameter updates and improved fairness in a single optimization step compared to SGD. We then validate these findings through extensive experiments on three publicly available datasets, namely CelebA, FairFace, and MS-COCO, across different tasks as facial expression recognition, gender classification, and multi-label classification, using various backbones. Considering multiple fairness definitions including equalized odds, equal opportunity, and demographic parity, adaptive optimizers like RMSProp and Adam consistently outperform SGD in terms of group fairness, while maintaining comparable predictive accuracy. Our results highlight the role of adaptive updates as a crucial yet overlooked mechanism for promoting fair outcomes. We release the source code at: https://github.com/Mkolahdoozi/Some-Optimizers-Are-More-Equal.
>
---
#### [replaced 082] S$^2$NN: Sub-bit Spiking Neural Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24266v2](http://arxiv.org/pdf/2509.24266v2)**

> **作者:** Wenjie Wei; Malu Zhang; Jieyuan Zhang; Ammar Belatreche; Shuai Wang; Yimeng Shan; Hanwen Liu; Honglin Cao; Guoqing Wang; Yang Yang; Haizhou Li
>
> **备注:** 29 pages, 6 figures
>
> **摘要:** Spiking Neural Networks (SNNs) offer an energy-efficient paradigm for machine intelligence, but their continued scaling poses challenges for resource-limited deployment. Despite recent advances in binary SNNs, the storage and computational demands remain substantial for large-scale networks. To further explore the compression and acceleration potential of SNNs, we propose Sub-bit Spiking Neural Networks (S$^2$NNs) that represent weights with less than one bit. Specifically, we first establish an S$^2$NN baseline by leveraging the clustering patterns of kernels in well-trained binary SNNs. This baseline is highly efficient but suffers from \textit{outlier-induced codeword selection bias} during training. To mitigate this issue, we propose an \textit{outlier-aware sub-bit weight quantization} (OS-Quant) method, which optimizes codeword selection by identifying and adaptively scaling outliers. Furthermore, we propose a \textit{membrane potential-based feature distillation} (MPFD) method, improving the performance of highly compressed S$^2$NN via more precise guidance from a teacher model. Extensive results on vision tasks reveal that S$^2$NN outperforms existing quantized SNNs in both performance and efficiency, making it promising for edge computing applications.
>
---
#### [replaced 083] SharpZO: Hybrid Sharpness-Aware Vision Language Model Prompt Tuning via Forward-Only Passes
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20990v2](http://arxiv.org/pdf/2506.20990v2)**

> **作者:** Yifan Yang; Zhen Zhang; Rupak Vignesh Swaminathan; Jing Liu; Nathan Susanj; Zheng Zhang
>
> **摘要:** Fine-tuning vision language models (VLMs) has achieved remarkable performance across various downstream tasks; yet, it requires access to model gradients through backpropagation (BP), making them unsuitable for memory-constrained, inference-only edge devices. To address this limitation, previous work has explored various BP-free fine-tuning methods. However, these approaches often rely on high-variance evolutionary strategies (ES) or zeroth-order (ZO) optimization, and often fail to achieve satisfactory performance. In this paper, we propose a hybrid Sharpness-aware Zeroth-order optimization (SharpZO) approach, specifically designed to enhance the performance of ZO VLM fine-tuning via a sharpness-aware warm-up training. SharpZO features a two-stage optimization process: a sharpness-aware ES stage that globally explores and smooths the loss landscape to construct a strong initialization, followed by a fine-grained local search via sparse ZO optimization. The entire optimization relies solely on forward passes. Detailed theoretical analysis and extensive experiments on CLIP models demonstrate that SharpZO significantly improves accuracy and convergence speed, achieving up to 7% average gain over state-of-the-art forward-only methods.
>
---
#### [replaced 084] Enhancing Feature Fusion of U-like Networks with Dynamic Skip Connections
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.14610v3](http://arxiv.org/pdf/2509.14610v3)**

> **作者:** Yue Cao; Quansong He; Kaishen Wang; Jianlong Xiong; Tao He
>
> **摘要:** U-like networks have become fundamental frameworks in medical image segmentation through skip connections that bridge high-level semantics and low-level spatial details. Despite their success, conventional skip connections exhibit two key limitations: inter-feature constraints and intra-feature constraints. The inter-feature constraint refers to the static nature of feature fusion in traditional skip connections, where information is transmitted along fixed pathways regardless of feature content. The intra-feature constraint arises from the insufficient modeling of multi-scale feature interactions, thereby hindering the effective aggregation of global contextual information. To overcome these limitations, we propose a novel Dynamic Skip Connection (DSC) block that fundamentally enhances cross-layer connectivity through adaptive mechanisms. The DSC block integrates two complementary components. (1) Test-Time Training (TTT) module. This module addresses the inter-feature constraint by enabling dynamic adaptation of hidden representations during inference, facilitating content-aware feature refinement. (2) Dynamic Multi-Scale Kernel (DMSK) module. To mitigate the intra-feature constraint, this module adaptively selects kernel sizes based on global contextual cues, enhancing the network capacity for multi-scale feature integration. The DSC block is architecture-agnostic and can be seamlessly incorporated into existing U-like network structures. Extensive experiments demonstrate the plug-and-play effectiveness of the proposed DSC block across CNN-based, Transformer-based, hybrid CNN-Transformer, and Mamba-based U-like networks.
>
---
#### [replaced 085] MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20879v3](http://arxiv.org/pdf/2506.20879v3)**

> **作者:** Shubhankar Borse; Seokeon Choi; Sunghyun Park; Jeongho Kim; Shreya Kadambi; Risheek Garrepalli; Sungrack Yun; Munawar Hayat; Fatih Porikli
>
> **备注:** Accepted at the NeurIPS 2025 D&B Track
>
> **摘要:** Generation of images containing multiple humans, performing complex actions, while preserving their facial identities, is a significant challenge. A major factor contributing to this is the lack of a dedicated benchmark. To address this, we introduce MultiHuman-Testbench, a novel benchmark for rigorously evaluating generative models for multi-human generation. The benchmark comprises 1,800 samples, including carefully curated text prompts, describing a range of simple to complex human actions. These prompts are matched with a total of 5,550 unique human face images, sampled uniformly to ensure diversity across age, ethnic background, and gender. Alongside captions, we provide human-selected pose conditioning images which accurately match the prompt. We propose a multi-faceted evaluation suite employing four key metrics to quantify face count, ID similarity, prompt alignment, and action detection. We conduct a thorough evaluation of a diverse set of models, including zero-shot approaches and training-based methods, with and without regional priors. We also propose novel techniques to incorporate image and region isolation using human segmentation and Hungarian matching, significantly improving ID similarity. Our proposed benchmark and key findings provide valuable insights and a standardized tool for advancing research in multi-human image generation. The dataset and evaluation codes will be available at https://github.com/Qualcomm-AI-research/MultiHuman-Testbench.
>
---
#### [replaced 086] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems
- **分类: cs.CV; eess.SP; math.OC**

- **链接: [http://arxiv.org/pdf/2510.01608v2](http://arxiv.org/pdf/2510.01608v2)**

> **作者:** Roman Jacome; Romario Gualdrón-Hurtado; Leon Suarez; Henry Arguello
>
> **备注:** 25 pages, 12 tables, 10 figures. Accepted to NeurIPS 2025
>
> **摘要:** Imaging inverse problems aim to recover high-dimensional signals from undersampled, noisy measurements, a fundamentally ill-posed task with infinite solutions in the null-space of the sensing operator. To resolve this ambiguity, prior information is typically incorporated through handcrafted regularizers or learned models that constrain the solution space. However, these priors typically ignore the task-specific structure of that null-space. In this work, we propose Non-Linear Projections of the Null-Space (NPN), a novel class of regularization that, instead of enforcing structural constraints in the image domain, promotes solutions that lie in a low-dimensional projection of the sensing matrix's null-space with a neural network. Our approach has two key advantages: (1) Interpretability: by focusing on the structure of the null-space, we design sensing-matrix-specific priors that capture information orthogonal to the signal components that are fundamentally blind to the sensing process. (2) Flexibility: NPN is adaptable to various inverse problems, compatible with existing reconstruction frameworks, and complementary to conventional image-domain priors. We provide theoretical guarantees on convergence and reconstruction accuracy when used within plug-and-play methods. Empirical results across diverse sensing matrices demonstrate that NPN priors consistently enhance reconstruction fidelity in various imaging inverse problems, such as compressive sensing, deblurring, super-resolution, computed tomography, and magnetic resonance imaging, with plug-and-play methods, unrolling networks, deep image prior, and diffusion models.
>
---
#### [replaced 087] Lightweight Facial Landmark Detection in Thermal Images via Multi-Level Cross-Modal Knowledge Transfer
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.11128v2](http://arxiv.org/pdf/2510.11128v2)**

> **作者:** Qiyi Tong; Olivia Nocentini; Marta Lagomarsino; Kuanqi Cai; Marta Lorenzini; Arash Ajoudani
>
> **摘要:** Facial Landmark Detection (FLD) in thermal imagery is critical for applications in challenging lighting conditions, but it is hampered by the lack of rich visual cues. Conventional cross-modal solutions, like feature fusion or image translation from RGB data, are often computationally expensive or introduce structural artifacts, limiting their practical deployment. To address this, we propose Multi-Level Cross-Modal Knowledge Distillation (MLCM-KD), a novel framework that decouples high-fidelity RGB-to-thermal knowledge transfer from model compression to create both accurate and efficient thermal FLD models. A central challenge during knowledge transfer is the profound modality gap between RGB and thermal data, where traditional unidirectional distillation fails to enforce semantic consistency across disparate feature spaces. To overcome this, we introduce Dual-Injected Knowledge Distillation (DIKD), a bidirectional mechanism designed specifically for this task. DIKD establishes a connection between modalities: it not only guides the thermal student with rich RGB features but also validates the student's learned representations by feeding them back into the frozen teacher's prediction head. This closed-loop supervision forces the student to learn modality-invariant features that are semantically aligned with the teacher, ensuring a robust and profound knowledge transfer. Experiments show that our approach sets a new state-of-the-art on public thermal FLD benchmarks, notably outperforming previous methods while drastically reducing computational overhead.
>
---
#### [replaced 088] WMCopier: Forging Invisible Image Watermarks on Arbitrary Images
- **分类: cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22330v3](http://arxiv.org/pdf/2503.22330v3)**

> **作者:** Ziping Dong; Chao Shuai; Zhongjie Ba; Peng Cheng; Zhan Qin; Qinglong Wang; Kui Ren
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Invisible Image Watermarking is crucial for ensuring content provenance and accountability in generative AI. While Gen-AI providers are increasingly integrating invisible watermarking systems, the robustness of these schemes against forgery attacks remains poorly characterized. This is critical, as forging traceable watermarks onto illicit content leads to false attribution, potentially harming the reputation and legal standing of Gen-AI service providers who are not responsible for the content. In this work, we propose WMCopier, an effective watermark forgery attack that operates without requiring any prior knowledge of or access to the target watermarking algorithm. Our approach first models the target watermark distribution using an unconditional diffusion model, and then seamlessly embeds the target watermark into a non-watermarked image via a shallow inversion process. We also incorporate an iterative optimization procedure that refines the reconstructed image to further trade off the fidelity and forgery efficiency. Experimental results demonstrate that WMCopier effectively deceives both open-source and closed-source watermark systems (e.g., Amazon's system), achieving a significantly higher success rate than existing methods. Additionally, we evaluate the robustness of forged samples and discuss the potential defenses against our attack.
>
---
#### [replaced 089] Boosting Adversarial Transferability with Spatial Adversarial Alignment
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2501.01015v2](http://arxiv.org/pdf/2501.01015v2)**

> **作者:** Zhaoyu Chen; Haijing Guo; Kaixun Jiang; Jiyuan Fu; Xinyu Zhou; Dingkang Yang; Hao Tang; Bo Li; Wenqiang Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Deep neural networks are vulnerable to adversarial examples that exhibit transferability across various models. Numerous approaches are proposed to enhance the transferability of adversarial examples, including advanced optimization, data augmentation, and model modifications. However, these methods still show limited transferability, particularly in cross-architecture scenarios, such as from CNN to ViT. To achieve high transferability, we propose a technique termed Spatial Adversarial Alignment (SAA), which employs an alignment loss and leverages a witness model to fine-tune the surrogate model. Specifically, SAA consists of two key parts: spatial-aware alignment and adversarial-aware alignment. First, we minimize the divergences of features between the two models in both global and local regions, facilitating spatial alignment. Second, we introduce a self-adversarial strategy that leverages adversarial examples to impose further constraints, aligning features from an adversarial perspective. Through this alignment, the surrogate model is trained to concentrate on the common features extracted by the witness model. This facilitates adversarial attacks on these shared features, thereby yielding perturbations that exhibit enhanced transferability. Extensive experiments on various architectures on ImageNet show that aligned surrogate models based on SAA can provide higher transferable adversarial examples, especially in cross-architecture attacks.
>
---
#### [replaced 090] AGC-Drive: A Large-Scale Dataset for Real-World Aerial-Ground Collaboration in Driving Scenarios
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16371v2](http://arxiv.org/pdf/2506.16371v2)**

> **作者:** Yunhao Hou; Bochao Zou; Min Zhang; Ran Chen; Shangdong Yang; Yanmei Zhang; Junbao Zhuo; Siheng Chen; Jiansheng Chen; Huimin Ma
>
> **摘要:** By sharing information across multiple agents, collaborative perception helps autonomous vehicles mitigate occlusions and improve overall perception accuracy. While most previous work focus on vehicle-to-vehicle and vehicle-to-infrastructure collaboration, with limited attention to aerial perspectives provided by UAVs, which uniquely offer dynamic, top-down views to alleviate occlusions and monitor large-scale interactive environments. A major reason for this is the lack of high-quality datasets for aerial-ground collaborative scenarios. To bridge this gap, we present AGC-Drive, the first large-scale real-world dataset for Aerial-Ground Cooperative 3D perception. The data collection platform consists of two vehicles, each equipped with five cameras and one LiDAR sensor, and one UAV carrying a forward-facing camera and a LiDAR sensor, enabling comprehensive multi-view and multi-agent perception. Consisting of approximately 80K LiDAR frames and 360K images, the dataset covers 14 diverse real-world driving scenarios, including urban roundabouts, highway tunnels, and on/off ramps. Notably, 17% of the data comprises dynamic interaction events, including vehicle cut-ins, cut-outs, and frequent lane changes. AGC-Drive contains 350 scenes, each with approximately 100 frames and fully annotated 3D bounding boxes covering 13 object categories. We provide benchmarks for two 3D perception tasks: vehicle-to-vehicle collaborative perception and vehicle-to-UAV collaborative perception. Additionally, we release an open-source toolkit, including spatiotemporal alignment verification tools, multi-agent visualization systems, and collaborative annotation utilities. The dataset and code are available at https://github.com/PercepX/AGC-Drive.
>
---
#### [replaced 091] Distilled Decoding 1: One-step Sampling of Image Auto-regressive Models with Flow Matching
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.17153v3](http://arxiv.org/pdf/2412.17153v3)**

> **作者:** Enshu Liu; Xuefei Ning; Yu Wang; Zinan Lin
>
> **摘要:** Autoregressive (AR) models have achieved state-of-the-art performance in text and image generation but suffer from slow generation due to the token-by-token process. We ask an ambitious question: can a pre-trained AR model be adapted to generate outputs in just one or two steps? If successful, this would significantly advance the development and deployment of AR models. We notice that existing works that try to speed up AR generation by generating multiple tokens at once fundamentally cannot capture the output distribution due to the conditional dependencies between tokens, limiting their effectiveness for few-step generation. To address this, we propose Distilled Decoding (DD), which uses flow matching to create a deterministic mapping from Gaussian distribution to the output distribution of the pre-trained AR model. We then train a network to distill this mapping, enabling few-step generation. DD doesn't need the training data of the original AR model, making it more practical. We evaluate DD on state-of-the-art image AR models and present promising results on ImageNet-256. For VAR, which requires 10-step generation, DD enables one-step generation (6.3$\times$ speed-up), with an acceptable increase in FID from 4.19 to 9.96. For LlamaGen, DD reduces generation from 256 steps to 1, achieving an 217.8$\times$ speed-up with a comparable FID increase from 4.11 to 11.35. In both cases, baseline methods completely fail with FID>100. DD also excels on text-to-image generation, reducing the generation from 256 steps to 2 for LlamaGen with minimal FID increase from 25.70 to 28.95. As the first work to demonstrate the possibility of one-step generation for image AR models, DD challenges the prevailing notion that AR models are inherently slow, and opens up new opportunities for efficient AR generation. The project website is at https://imagination-research.github.io/distilled-decoding.
>
---
#### [replaced 092] A review of Recent Techniques for Person Re-Identification
- **分类: cs.CV; 68T45, 65D19**

- **链接: [http://arxiv.org/pdf/2509.22690v2](http://arxiv.org/pdf/2509.22690v2)**

> **作者:** Andrea Asperti; Salvatore Fiorilla; Simone Nardi; Lorenzo Orsini
>
> **摘要:** Person re-identification (ReId), a crucial task in surveillance, involves matching individuals across different camera views. The advent of Deep Learning, especially supervised techniques like Convolutional Neural Networks and Attention Mechanisms, has significantly enhanced person Re-ID. However, the success of supervised approaches hinges on vast amounts of annotated data, posing scalability challenges in data labeling and computational costs. To address these limitations, recent research has shifted towards unsupervised person re-identification. Leveraging abundant unlabeled data, unsupervised methods aim to overcome the need for pairwise labelled data. Although traditionally trailing behind supervised approaches, unsupervised techniques have shown promising developments in recent years, signalling a narrowing performance gap. Motivated by this evolving landscape, our survey pursues two primary objectives. First, we review and categorize significant publications in supervised person re-identification, providing an in-depth overview of the current state-of-the-art and emphasizing little room for further improvement in this domain. Second, we explore the latest advancements in unsupervised person re-identification over the past three years, offering insights into emerging trends and shedding light on the potential convergence of performance between supervised and unsupervised paradigms. This dual-focus survey aims to contribute to the evolving narrative of person re-identification, capturing both the mature landscape of supervised techniques and the promising outcomes in the realm of unsupervised learning.
>
---
#### [replaced 093] RT-DATR: Real-time Unsupervised Domain Adaptive Detection Transformer with Adversarial Feature Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09196v2](http://arxiv.org/pdf/2504.09196v2)**

> **作者:** Feng Lv; Guoqing Li; Jin Li; Chunlong Xia
>
> **摘要:** Despite domain-adaptive object detectors based on CNN and transformers have made significant progress in cross-domain detection tasks, it is regrettable that domain adaptation for real-time transformer-based detectors has not yet been explored. Directly applying existing domain adaptation algorithms has proven to be suboptimal. In this paper, we propose RT-DATR, a simple and efficient real-time domain adaptive detection transformer. Building on RT-DETR as our base detector, we first introduce a local object-level feature alignment module to significantly enhance the feature representation of domain invariance during object transfer. Additionally, we introduce a scene semantic feature alignment module designed to boost cross-domain detection performance by aligning scene semantic features. Finally, we introduced a domain query and decoupled it from the object query to further align the instance feature distribution within the decoder layer, reduce the domain gap, and maintain discriminative ability. Experimental results on various cross-domian benchmarks demonstrate that our method outperforms current state-of-the-art approaches. Code is available at https://github.com/Jeremy-lf/RT-DATR.
>
---
#### [replaced 094] Circle Representation for Medical Instance Object Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.11507v2](http://arxiv.org/pdf/2403.11507v2)**

> **作者:** Juming Xiong; Ethan H. Nguyen; Yilin Liu; Ruining Deng; Regina N Tyree; Hernan Correa; Girish Hiremath; Yaohong Wang; Haichun Yang; Agnes B. Fogo; Yuankai Huo
>
> **摘要:** Recently, circle representation has been introduced for medical imaging, designed specifically to enhance the detection of instance objects that are spherically shaped (e.g., cells, glomeruli, and nuclei). Given its outstanding effectiveness in instance detection, it is compelling to consider the application of circle representation for segmenting instance medical objects. In this study, we introduce CircleSnake, a simple end-to-end segmentation approach that utilizes circle contour deformation for segmenting ball-shaped medical objects at the instance level. The innovation of CircleSnake lies in these three areas: (1) It substitutes the complex bounding box-to-octagon contour transformation with a more consistent and rotation-invariant bounding circle-to-circle contour adaptation. This adaptation specifically targets ball-shaped medical objects. (2) The circle representation employed in CircleSnake significantly reduces the degrees of freedom to two, compared to eight in the octagon representation. This reduction enhances both the robustness of the segmentation performance and the rotational consistency of the method. (3) CircleSnake is the first end-to-end deep instance segmentation pipeline to incorporate circle representation, encompassing consistent circle detection, circle contour proposal, and circular convolution in a unified framework. This integration is achieved through the novel application of circular graph convolution within the context of circle detection and instance segmentation. In practical applications, such as the detection of glomeruli, nuclei, and eosinophils in pathological images, CircleSnake has demonstrated superior performance and greater rotation invariance when compared to benchmarks. The code has been made publicly available: https://github.com/hrlblab/CircleSnake.
>
---
#### [replaced 095] BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.20095v2](http://arxiv.org/pdf/2510.20095v2)**

> **作者:** Ziheng Zhang; Xinyue Ma; Arpita Chowdhury; Elizabeth G. Campolongo; Matthew J. Thompson; Net Zhang; Samuel Stevens; Hilmar Lapp; Tanya Berger-Wolf; Yu Su; Wei-Lun Chao; Jianyang Gu
>
> **备注:** Project page: https://imageomics.github.io/biocap/
>
> **摘要:** This work investigates descriptive captions as an additional source of supervision for biological multimodal foundation models. Images and captions can be viewed as complementary samples from the latent morphospace of a species, each capturing certain biological traits. Incorporating captions during training encourages alignment with this shared latent structure, emphasizing potentially diagnostic characters while suppressing spurious correlations. The main challenge, however, lies in obtaining faithful, instance-specific captions at scale. This requirement has limited the utilization of natural language supervision in organismal biology compared with many other scientific domains. We complement this gap by generating synthetic captions with multimodal large language models (MLLMs), guided by Wikipedia-derived visual information and taxon-tailored format examples. These domain-specific contexts help reduce hallucination and yield accurate, instance-based descriptive captions. Using these captions, we train BioCAP (i.e., BioCLIP with Captions), a biological foundation model that captures rich semantics and achieves strong performance in species classification and text-image retrieval. These results demonstrate the value of descriptive captions beyond labels in bridging biological images with multimodal foundation models.
>
---
#### [replaced 096] CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22854v2](http://arxiv.org/pdf/2505.22854v2)**

> **作者:** Kornel Howil; Joanna Waczyńska; Piotr Borycki; Tadeusz Dziarmaga; Marcin Mazur; Przemysław Spurek
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Gaussian Splatting (GS) has recently emerged as an efficient representation for rendering 3D scenes from 2D images and has been extended to images, videos, and dynamic 4D content. However, applying style transfer to GS-based representations, especially beyond simple color changes, remains challenging. In this work, we introduce CLIPGaussian, the first unified style transfer framework that supports text- and image-guided stylization across multiple modalities: 2D images, videos, 3D objects, and 4D scenes. Our method operates directly on Gaussian primitives and integrates into existing GS pipelines as a plug-in module, without requiring large generative models or retraining from scratch. The CLIPGaussian approach enables joint optimization of color and geometry in 3D and 4D settings, and achieves temporal coherence in videos, while preserving the model size. We demonstrate superior style fidelity and consistency across all tasks, validating CLIPGaussian as a universal and efficient solution for multimodal style transfer.
>
---
#### [replaced 097] One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.17611v2](http://arxiv.org/pdf/2510.17611v2)**

> **作者:** Jia Guo; Shuai Lu; Lei Fan; Zelin Li; Donglin Di; Yang Song; Weihang Zhang; Wenbing Zhu; Hong Yan; Fang Chen; Huiqi Li; Hongen Liao
>
> **备注:** Extended version of CVPR2025
>
> **摘要:** Unsupervised anomaly detection (UAD) has evolved from building specialized single-class models to unified multi-class models, yet existing multi-class models significantly underperform the most advanced one-for-one counterparts. Moreover, the field has fragmented into specialized methods tailored to specific scenarios (multi-class, 3D, few-shot, etc.), creating deployment barriers and highlighting the need for a unified solution. In this paper, we present Dinomaly2, the first unified framework for full-spectrum image UAD, which bridges the performance gap in multi-class models while seamlessly extending across diverse data modalities and task settings. Guided by the "less is more" philosophy, we demonstrate that the orchestration of five simple element achieves superior performance in a standard reconstruction-based framework. This methodological minimalism enables natural extension across diverse tasks without modification, establishing that simplicity is the foundation of true universality. Extensive experiments on 12 UAD benchmarks demonstrate Dinomaly2's full-spectrum superiority across multiple modalities (2D, multi-view, RGB-3D, RGB-IR), task settings (single-class, multi-class, inference-unified multi-class, few-shot) and application domains (industrial, biological, outdoor). For example, our multi-class model achieves unprecedented 99.9% and 99.3% image-level (I-) AUROC on MVTec-AD and VisA respectively. For multi-view and multi-modal inspection, Dinomaly2 demonstrates state-of-the-art performance with minimum adaptations. Moreover, using only 8 normal examples per class, our method surpasses previous full-shot models, achieving 98.7% and 97.4% I-AUROC on MVTec-AD and VisA. The combination of minimalistic design, computational scalability, and universal applicability positions Dinomaly2 as a unified solution for the full spectrum of real-world anomaly detection applications.
>
---
#### [replaced 098] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.07961v3](http://arxiv.org/pdf/2510.07961v3)**

> **作者:** Yidi Liu; Xueyang Fu; Jie Huang; Jie Xiao; Dong Li; Wenlong Zhang; Lei Bai; Zheng-Jun Zha
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Ultra-High Definition (UHD) image restoration faces a trade-off between computational efficiency and high-frequency detail retention. While Variational Autoencoders (VAEs) improve efficiency via latent-space processing, their Gaussian constraint often discards degradation-specific high-frequency information, hurting reconstruction fidelity. To overcome this, we propose Latent Harmony, a two-stage framework that redefines VAEs for UHD restoration by jointly regularizing the latent space and enforcing high-frequency-aware reconstruction.In Stage One, we introduce LH-VAE, which enhances semantic robustness through visual semantic constraints and progressive degradation perturbations, while latent equivariance strengthens high-frequency reconstruction.Stage Two jointly trains this refined VAE with a restoration model using High-Frequency Low-Rank Adaptation (HF-LoRA): an encoder LoRA guided by a fidelity-oriented high-frequency alignment loss to recover authentic details, and a decoder LoRA driven by a perception-oriented loss to synthesize realistic textures. Both LoRA modules are trained via alternating optimization with selective gradient propagation to preserve the pretrained latent structure.At inference, a tunable parameter {\alpha} enables flexible fidelity-perception trade-offs.Experiments show Latent Harmony achieves state-of-the-art performance across UHD and standard-resolution tasks, effectively balancing efficiency, perceptual quality, and reconstruction accuracy.
>
---
#### [replaced 099] DynamicPAE: Generating Scene-Aware Physical Adversarial Examples in Real-Time
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.08053v3](http://arxiv.org/pdf/2412.08053v3)**

> **作者:** Jin Hu; Xianglong Liu; Jiakai Wang; Junkai Zhang; Xianqi Yang; Haotong Qin; Yuqing Ma; Ke Xu
>
> **摘要:** Physical adversarial examples (PAEs) are regarded as whistle-blowers of real-world risks in deep-learning applications, thus worth further investigation. However, current PAE generation studies show limited adaptive attacking ability to diverse and varying scenes, revealing the urgent requirement of dynamic PAEs that are generated in real time and conditioned on the observation from the attacker. The key challenge in generating dynamic PAEs is learning the sparse relation between PAEs and the observation of attackers under the noisy feedback of attack training. To address the challenge, we present DynamicPAE, the first generative framework that enables scene-aware real-time physical attacks. Specifically, to address the noisy feedback problem that obfuscates the exploration of scene-related PAEs, we introduce the residual-guided adversarial pattern exploration technique. Residual-guided training, which relaxes the attack training with a reconstruction task, is proposed to enrich the feedback information, thereby achieving a more comprehensive exploration of PAEs. To address the alignment problem between the trained generator and the real-world scenario, we introduce the distribution-matched attack scenario alignment, consisting of the conditional-uncertainty-aligned data module and the skewness-aligned objective re-weighting module. The former aligns the training environment with the incomplete observation of the real-world attacker. The latter facilitates consistent stealth control across different attack targets with the skewness controller. Extensive digital and physical evaluations demonstrate the superior attack performance of DynamicPAE, attaining a 2.07 $\times$ boost (58.8% average AP drop under attack) on representative object detectors (e.g., DETR) over state-of-the-art static PAE generating methods. Overall, our work opens the door to end-to-end modeling of dynamic PAEs.
>
---
#### [replaced 100] Fréchet Power-Scenario Distance: A Metric for Evaluating Generative AI Models across Multiple Time-Scales in Smart Grids
- **分类: cs.LG; cs.AI; cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.08082v2](http://arxiv.org/pdf/2505.08082v2)**

> **作者:** Yuting Cai; Shaohuai Liu; Chao Tian; Le Xie
>
> **摘要:** Generative artificial intelligence (AI) models in smart grids have advanced significantly in recent years due to their ability to generate large amounts of synthetic data, which would otherwise be difficult to obtain in the real world due to confidentiality constraints. A key challenge in utilizing such synthetic data is how to assess the data quality produced from such generative models. Traditional Euclidean distance-based metrics only reflect pair-wise relations between two individual samples, and could fail in evaluating quality differences between groups of synthetic datasets. In this work, we propose a novel metric based on the Fr\'{e}chet Distance (FD) estimated between two datasets in a learned feature space. The proposed method evaluates the quality of generation from a distributional perspective. Empirical results demonstrate the superiority of the proposed metric across timescales and models, enhancing the reliability of data-driven decision-making in smart grid operations.
>
---
#### [replaced 101] TopoFR: A Closer Look at Topology Alignment on Face Recognition
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.10587v2](http://arxiv.org/pdf/2410.10587v2)**

> **作者:** Jun Dan; Yang Liu; Jiankang Deng; Haoyu Xie; Siyuan Li; Baigui Sun; Shan Luo
>
> **备注:** Accepted by NeurIPS 2024
>
> **摘要:** The field of face recognition (FR) has undergone significant advancements with the rise of deep learning. Recently, the success of unsupervised learning and graph neural networks has demonstrated the effectiveness of data structure information. Considering that the FR task can leverage large-scale training data, which intrinsically contains significant structure information, we aim to investigate how to encode such critical structure information into the latent space. As revealed from our observations, directly aligning the structure information between the input and latent spaces inevitably suffers from an overfitting problem, leading to a structure collapse phenomenon in the latent space. To address this problem, we propose TopoFR, a novel FR model that leverages a topological structure alignment strategy called PTSA and a hard sample mining strategy named SDE. Concretely, PTSA uses persistent homology to align the topological structures of the input and latent spaces, effectively preserving the structure information and improving the generalization performance of FR model. To mitigate the impact of hard samples on the latent space structure, SDE accurately identifies hard samples by automatically computing structure damage score (SDS) for each sample, and directs the model to prioritize optimizing these samples. Experimental results on popular face benchmarks demonstrate the superiority of our TopoFR over the state-of-the-art methods. Code and models are available at: https://github.com/modelscope/facechain/tree/main/face_module/TopoFR.
>
---
