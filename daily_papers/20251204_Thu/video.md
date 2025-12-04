# 计算机视觉 cs.CV

- **最新发布 130 篇**

- **更新 82 篇**

## 最新发布

#### [new 001] ReCamDriving: LiDAR-Free Camera-Controlled Novel Trajectory Video Generation
- **分类: cs.CV**

- **简介: 该论文提出ReCamDriving，一种无需LiDAR的纯视觉相机控制新轨迹视频生成框架。针对修复方法无法恢复复杂失真、LiDAR数据稀疏的问题，利用3DGS提供稠密几何引导，通过两阶段训练与跨轨迹数据构建，实现高精度相机可控生成，显著提升结构一致性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.03621v1](https://arxiv.org/pdf/2512.03621v1)**

> **作者:** Yaokun Li; Shuaixian Wang; Mantang Guo; Jiehui Huang; Taojun Ding; Mu Hu; Kaixuan Wang; Shaojie Shen; Guang Tan
>
> **备注:** Project page: https://recamdriving.github.io/
>
> **摘要:** We propose ReCamDriving, a purely vision-based, camera-controlled novel-trajectory video generation framework. While repair-based methods fail to restore complex artifacts and LiDAR-based approaches rely on sparse and incomplete cues, ReCamDriving leverages dense and scene-complete 3DGS renderings for explicit geometric guidance, achieving precise camera-controllable generation. To mitigate overfitting to restoration behaviors when conditioned on 3DGS renderings, ReCamDriving adopts a two-stage training paradigm: the first stage uses camera poses for coarse control, while the second stage incorporates 3DGS renderings for fine-grained viewpoint and geometric guidance. Furthermore, we present a 3DGS-based cross-trajectory data curation strategy to eliminate the train-test gap in camera transformation patterns, enabling scalable multi-trajectory supervision from monocular videos. Based on this strategy, we construct the ParaDrive dataset, containing over 110K parallel-trajectory video pairs. Extensive experiments demonstrate that ReCamDriving achieves state-of-the-art camera controllability and structural consistency.
>
---
#### [new 002] Diminishing Returns in Self-Supervised Learning
- **分类: cs.CV**

- **简介: 该论文研究小规模视觉变换器（ViT）在自监督学习中的性能提升问题。通过实验发现，预训练和微调虽有益但收益递减，而中间微调可能因任务差异导致性能下降。研究表明，小模型更需针对性预训练与数据筛选，避免盲目堆叠任务。**

- **链接: [https://arxiv.org/pdf/2512.03862v1](https://arxiv.org/pdf/2512.03862v1)**

> **作者:** Oli Bridge; Huey Sun; Botond Branyicskai-Nagy; Charles D'Ornano; Shomit Basu
>
> **摘要:** While transformer-based architectures have taken computer vision and NLP by storm, they often require a vast amount of parameters and training data to attain strong performance. In this work, we experiment with three distinct pre-training, intermediate fine-tuning, and downstream datasets and training objectives to explore their marginal benefits on a small 5M-parameter vision transformer. We find that while pre-training and fine-tuning always help our model but have diminishing returns, intermediate fine-tuning can actually show harmful impact on downstream performance, potentially due to dissimilarity in task mechanics. Taken together, our results suggest that small-scale ViTs benefit most from targeted pre-training and careful data selection, while indiscriminate stacking of intermediate tasks can waste compute and even degrade performance.
>
---
#### [new 003] Global-Local Aware Scene Text Editing
- **分类: cs.CV**

- **简介: 该论文针对场景文本编辑任务，解决现有方法在文本风格一致性与长度变化适应性上的不足。提出端到端的GLASTE框架，融合全局上下文与局部细节，通过联合损失和风格向量表示，实现风格保持与区域协调，有效处理不同长度文本的编辑。**

- **链接: [https://arxiv.org/pdf/2512.03574v1](https://arxiv.org/pdf/2512.03574v1)**

> **作者:** Fuxiang Yang; Tonghua Su; Donglin Di; Yin Chen; Xiangqian Wu; Zhongjie Wang; Lei Fan
>
> **摘要:** Scene Text Editing (STE) involves replacing text in a scene image with new target text while preserving both the original text style and background texture. Existing methods suffer from two major challenges: inconsistency and length-insensitivity. They often fail to maintain coherence between the edited local patch and the surrounding area, and they struggle to handle significant differences in text length before and after editing. To tackle these challenges, we propose an end-to-end framework called Global-Local Aware Scene Text Editing (GLASTE), which simultaneously incorporates high-level global contextual information along with delicate local features. Specifically, we design a global-local combination structure, joint global and local losses, and enhance text image features to ensure consistency in text style within local patches while maintaining harmony between local and global areas. Additionally, we express the text style as a vector independent of the image size, which can be transferred to target text images of various sizes. We use an affine fusion to fill target text images into the editing patch while maintaining their aspect ratio unchanged. Extensive experiments on real-world datasets validate that our GLASTE model outperforms previous methods in both quantitative metrics and qualitative results and effectively mitigates the two challenges.
>
---
#### [new 004] CartoMapQA: A Fundamental Benchmark Dataset Evaluating Vision-Language Models on Cartographic Map Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出CartoMapQA，一个用于评估视觉语言模型地图理解能力的基准数据集。针对LVLM在地图语义理解、地理空间推理和OCR误差方面的不足，构建包含2000+样本的问答数据集，涵盖符号识别、信息提取、尺度理解等任务，旨在推动模型在导航、城市规划等实际应用中的地图理解能力。**

- **链接: [https://arxiv.org/pdf/2512.03558v1](https://arxiv.org/pdf/2512.03558v1)**

> **作者:** Huy Quang Ung; Guillaume Habault; Yasutaka Nishimura; Hao Niu; Roberto Legaspi; Tomoki Oya; Ryoichi Kojima; Masato Taya; Chihiro Ono; Atsunori Minamikawa; Yan Liu
>
> **备注:** Accepted at SIGSPATIAL 2025 (Best paper candidates), 15 pages
>
> **摘要:** The rise of Visual-Language Models (LVLMs) has unlocked new possibilities for seamlessly integrating visual and textual information. However, their ability to interpret cartographic maps remains largely unexplored. In this paper, we introduce CartoMapQA, a benchmark specifically designed to evaluate LVLMs' understanding of cartographic maps through question-answering tasks. The dataset includes over 2000 samples, each composed of a cartographic map, a question (with open-ended or multiple-choice answers), and a ground-truth answer. These tasks span key low-, mid- and high-level map interpretation skills, including symbol recognition, embedded information extraction, scale interpretation, and route-based reasoning. Our evaluation of both open-source and proprietary LVLMs reveals persistent challenges: models frequently struggle with map-specific semantics, exhibit limited geospatial reasoning, and are prone to Optical Character Recognition (OCR)-related errors. By isolating these weaknesses, CartoMapQA offers a valuable tool for guiding future improvements in LVLM architectures. Ultimately, it supports the development of models better equipped for real-world applications that depend on robust and reliable map understanding, such as navigation, geographic search, and urban planning. Our source code and data are openly available to the research community at: https://github.com/ungquanghuy-kddi/CartoMapQA.git
>
---
#### [new 005] PosterCopilot: Toward Layout Reasoning and Controllable Editing for Professional Graphic Design
- **分类: cs.CV**

- **简介: 该论文针对专业图形设计中布局几何不准、缺乏可控编辑的问题，提出PosterCopilot框架。通过三阶段训练增强大模型的几何与美学理解，并结合生成模型实现分层、迭代的可控编辑，提升设计精度与一致性。**

- **链接: [https://arxiv.org/pdf/2512.04082v1](https://arxiv.org/pdf/2512.04082v1)**

> **作者:** Jiazhe Wei; Ken Li; Tianyu Lao; Haofan Wang; Liang Wang; Caifeng Shan; Chenyang Si
>
> **备注:** Project page: https://postercopilot.github.io/
>
> **摘要:** Graphic design forms the cornerstone of modern visual communication, serving as a vital medium for promoting cultural and commercial events. Recent advances have explored automating this process using Large Multimodal Models (LMMs), yet existing methods often produce geometrically inaccurate layouts and lack the iterative, layer-specific editing required in professional workflows. To address these limitations, we present PosterCopilot, a framework that advances layout reasoning and controllable editing for professional graphic design. Specifically, we introduce a progressive three-stage training strategy that equips LMMs with geometric understanding and aesthetic reasoning for layout design, consisting of Perturbed Supervised Fine-Tuning, Reinforcement Learning for Visual-Reality Alignment, and Reinforcement Learning from Aesthetic Feedback. Furthermore, we develop a complete workflow that couples the trained LMM-based design model with generative models, enabling layer-controllable, iterative editing for precise element refinement while maintaining global visual consistency. Extensive experiments demonstrate that PosterCopilot achieves geometrically accurate and aesthetically superior layouts, offering unprecedented controllability for professional iterative design.
>
---
#### [new 006] Structured Uncertainty Similarity Score (SUSS): Learning a Probabilistic, Interpretable, Perceptual Metric Between Images
- **分类: cs.CV**

- **简介: 该论文提出SUSS，一种可解释的图像感知相似性度量。针对现有方法在感知对齐与可解释性间的矛盾，通过生成式自监督学习建模图像为结构化多变量正态分布，利用人类难以察觉的增强数据训练，并结合人类感知数据学习权重。其线性残差变换使结果可透明分析，兼具高感知校准性与局部解释能力，适用于图像重建等下游任务。**

- **链接: [https://arxiv.org/pdf/2512.03701v1](https://arxiv.org/pdf/2512.03701v1)**

> **作者:** Paula Seidler; Neill D. F. Campbell; Ivor J A Simpson
>
> **摘要:** Perceptual similarity scores that align with human vision are critical for both training and evaluating computer vision models. Deep perceptual losses, such as LPIPS, achieve good alignment but rely on complex, highly non-linear discriminative features with unknown invariances, while hand-crafted measures like SSIM are interpretable but miss key perceptual properties. We introduce the Structured Uncertainty Similarity Score (SUSS); it models each image through a set of perceptual components, each represented by a structured multivariate Normal distribution. These are trained in a generative, self-supervised manner to assign high likelihood to human-imperceptible augmentations. The final score is a weighted sum of component log-probabilities with weights learned from human perceptual datasets. Unlike feature-based methods, SUSS learns image-specific linear transformations of residuals in pixel space, enabling transparent inspection through decorrelated residuals and sampling. SUSS aligns closely with human perceptual judgments, shows strong perceptual calibration across diverse distortion types, and provides localized, interpretable explanations of its similarity assessments. We further demonstrate stable optimization behavior and competitive performance when using SUSS as a perceptual loss for downstream imaging tasks.
>
---
#### [new 007] CookAnything: A Framework for Flexible and Consistent Multi-Step Recipe Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CookAnything框架，解决烹饪类多步指令图像生成中灵活性与一致性不足的问题。针对指令长度可变、步骤间视觉连贯性差及食材一致性难保持等挑战，引入区域控制、步态感知位置编码和跨步一致性控制，实现任意长度、语义一致的图像序列生成。**

- **链接: [https://arxiv.org/pdf/2512.03540v1](https://arxiv.org/pdf/2512.03540v1)**

> **作者:** Ruoxuan Zhang; Bin Wen; Hongxia Xie; Yi Yao; Songhan Zuo; Jian-Yu Jiang-Lin; Hong-Han Shuai; Wen-Huang Cheng
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Cooking is a sequential and visually grounded activity, where each step such as chopping, mixing, or frying carries both procedural logic and visual semantics. While recent diffusion models have shown strong capabilities in text-to-image generation, they struggle to handle structured multi-step scenarios like recipe illustration. Additionally, current recipe illustration methods are unable to adjust to the natural variability in recipe length, generating a fixed number of images regardless of the actual instructions structure. To address these limitations, we present CookAnything, a flexible and consistent diffusion-based framework that generates coherent, semantically distinct image sequences from textual cooking instructions of arbitrary length. The framework introduces three key components: (1) Step-wise Regional Control (SRC), which aligns textual steps with corresponding image regions within a single denoising process; (2) Flexible RoPE, a step-aware positional encoding mechanism that enhances both temporal coherence and spatial diversity; and (3) Cross-Step Consistency Control (CSCC), which maintains fine-grained ingredient consistency across steps. Experimental results on recipe illustration benchmarks show that CookAnything performs better than existing methods in training-based and training-free settings. The proposed framework supports scalable, high-quality visual synthesis of complex multi-step instructions and holds significant potential for broad applications in instructional media, and procedural content creation.
>
---
#### [new 008] CoDA: From Text-to-Image Diffusion Models to Training-Free Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文针对数据集蒸馏（DD）中生成模型依赖特定训练和分布不匹配的问题，提出CoDA框架。通过识别目标数据的“核心分布”并引导文本到图像模型生成对齐样本，实现无需目标数据训练的高效蒸馏，在ImageNet-1K上达到60.4%新高准确率。**

- **链接: [https://arxiv.org/pdf/2512.03844v1](https://arxiv.org/pdf/2512.03844v1)**

> **作者:** Letian Zhou; Songhua Liu; Xinchao Wang
>
> **备注:** 34 pages, 24 figures
>
> **摘要:** Prevailing Dataset Distillation (DD) methods leveraging generative models confront two fundamental limitations. First, despite pioneering the use of diffusion models in DD and delivering impressive performance, the vast majority of approaches paradoxically require a diffusion model pre-trained on the full target dataset, undermining the very purpose of DD and incurring prohibitive training costs. Second, although some methods turn to general text-to-image models without relying on such target-specific training, they suffer from a significant distributional mismatch, as the web-scale priors encapsulated in these foundation models fail to faithfully capture the target-specific semantics, leading to suboptimal performance. To tackle these challenges, we propose Core Distribution Alignment (CoDA), a framework that enables effective DD using only an off-the-shelf text-to-image model. Our key idea is to first identify the "intrinsic core distribution" of the target dataset using a robust density-based discovery mechanism. We then steer the generative process to align the generated samples with this core distribution. By doing so, CoDA effectively bridges the gap between general-purpose generative priors and target semantics, yielding highly representative distilled datasets. Extensive experiments suggest that, without relying on a generative model specifically trained on the target dataset, CoDA achieves performance on par with or even superior to previous methods with such reliance across all benchmarks, including ImageNet-1K and its subsets. Notably, it establishes a new state-of-the-art accuracy of 60.4% at the 50-images-per-class (IPC) setup on ImageNet-1K. Our code is available on the project webpage: https://github.com/zzzlt422/CoDA
>
---
#### [new 009] RELIC: Interactive Video World Model with Long-Horizon Memory
- **分类: cs.CV**

- **简介: 该论文提出RELIC，一种具备长时记忆的交互式视频世界模型，解决实时性、空间一致性与用户控制三者难以兼顾的问题。通过压缩的相机感知记忆结构与自强化蒸馏机制，实现16 FPS实时生成，显著提升长时序生成稳定性与记忆准确性。**

- **链接: [https://arxiv.org/pdf/2512.04040v1](https://arxiv.org/pdf/2512.04040v1)**

> **作者:** Yicong Hong; Yiqun Mei; Chongjian Ge; Yiran Xu; Yang Zhou; Sai Bi; Yannick Hold-Geoffroy; Mike Roberts; Matthew Fisher; Eli Shechtman; Kalyan Sunkavalli; Feng Liu; Zhengqi Li; Hao Tan
>
> **备注:** 22 pages
>
> **摘要:** A truly interactive world model requires three key ingredients: real-time long-horizon streaming, consistent spatial memory, and precise user control. However, most existing approaches address only one of these aspects in isolation, as achieving all three simultaneously is highly challenging-for example, long-term memory mechanisms often degrade real-time performance. In this work, we present RELIC, a unified framework that tackles these three challenges altogether. Given a single image and a text description, RELIC enables memory-aware, long-duration exploration of arbitrary scenes in real time. Built upon recent autoregressive video-diffusion distillation techniques, our model represents long-horizon memory using highly compressed historical latent tokens encoded with both relative actions and absolute camera poses within the KV cache. This compact, camera-aware memory structure supports implicit 3D-consistent content retrieval and enforces long-term coherence with minimal computational overhead. In parallel, we fine-tune a bidirectional teacher video model to generate sequences beyond its original 5-second training horizon, and transform it into a causal student generator using a new memory-efficient self-forcing paradigm that enables full-context distillation over long-duration teacher as well as long student self-rollouts. Implemented as a 14B-parameter model and trained on a curated Unreal Engine-rendered dataset, RELIC achieves real-time generation at 16 FPS while demonstrating more accurate action following, more stable long-horizon streaming, and more robust spatial-memory retrieval compared with prior work. These capabilities establish RELIC as a strong foundation for the next generation of interactive world modeling.
>
---
#### [new 010] PosA-VLA: Enhancing Action Generation via Pose-Conditioned Anchor Attention
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在具身任务中动作冗余、不精确的问题，提出PosA-VLA框架。通过姿态条件化的锚定注意力机制，引导模型聚焦任务相关区域，提升动作生成的精度与效率。无需额外感知模块，架构轻量，实现在复杂环境中的高效精准操作。**

- **链接: [https://arxiv.org/pdf/2512.03724v1](https://arxiv.org/pdf/2512.03724v1)**

> **作者:** Ziwen Li; Xin Wang; Hanlue Zhang; Runnan Chen; Runqi Lin; Xiao He; Han Huang; Yandong Guo; Fakhri Karray; Tongliang Liu; Mingming Gong
>
> **摘要:** The Vision-Language-Action (VLA) models have demonstrated remarkable performance on embodied tasks and shown promising potential for real-world applications. However, current VLAs still struggle to produce consistent and precise target-oriented actions, as they often generate redundant or unstable motions along trajectories, limiting their applicability in time-sensitive scenarios.In this work, we attribute these redundant actions to the spatially uniform perception field of existing VLAs, which causes them to be distracted by target-irrelevant objects, especially in complex environments.To address this issue, we propose an efficient PosA-VLA framework that anchors visual attention via pose-conditioned supervision, consistently guiding the model's perception toward task-relevant regions. The pose-conditioned anchor attention mechanism enables the model to better align instruction semantics with actionable visual cues, thereby improving action generation precision and efficiency. Moreover, our framework adopts a lightweight architecture and requires no auxiliary perception modules (e.g., segmentation or grounding networks), ensuring efficient inference. Extensive experiments verify that our method executes embodied tasks with precise and time-efficient behavior across diverse robotic manipulation benchmarks and shows robust generalization in a variety of challenging environments.
>
---
#### [new 011] DINO-RotateMatch: A Rotation-Aware Deep Framework for Robust Image Matching in Large-Scale 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对大规模3D重建中的图像匹配难题，提出DINO-RotateMatch框架。通过自适应配对与旋转感知的关键点提取，结合自监督全局描述子与增强局部匹配，提升在互联网无序图像中的匹配鲁棒性与精度，获Kaggle 2025挑战赛银奖。**

- **链接: [https://arxiv.org/pdf/2512.03715v1](https://arxiv.org/pdf/2512.03715v1)**

> **作者:** Kaichen Zhang; Tianxiang Sheng; Xuanming Shi
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** This paper presents DINO-RotateMatch, a deep-learning framework designed to address the chal lenges of image matching in large-scale 3D reconstruction from unstructured Internet images. The method integrates a dataset-adaptive image pairing strategy with rotation-aware keypoint extraction and matching. DINO is employed to retrieve semantically relevant image pairs in large collections, while rotation-based augmentation captures orientation-dependent local features using ALIKED and Light Glue. Experiments on the Kaggle Image Matching Challenge 2025 demonstrate consistent improve ments in mean Average Accuracy (mAA), achieving a Silver Award (47th of 943 teams). The results confirm that combining self-supervised global descriptors with rotation-enhanced local matching offers a robust and scalable solution for large-scale 3D reconstruction.
>
---
#### [new 012] Prostate biopsy whole slide image dataset from an underrepresented Middle Eastern population
- **分类: cs.CV**

- **简介: 该论文针对AI在数字病理中缺乏中东人群数据的问题，发布339例伊拉克患者前列腺活检全幻灯片图像数据集。包含多扫描仪、多病理医生评估的Gleason评分与分级，支持模型泛化性、颜色归一化及跨设备鲁棒性研究，推动全球多样性病理AI发展。**

- **链接: [https://arxiv.org/pdf/2512.03854v1](https://arxiv.org/pdf/2512.03854v1)**

> **作者:** Peshawa J. Muhammad Ali; Navin Vincent; Saman S. Abdulla; Han N. Mohammed Fadhl; Anders Blilie; Kelvin Szolnoky; Julia Anna Mielcarz; Xiaoyi Ji; Kimmo Kartasalo; Abdulbasit K. Al-Talabani; Nita Mulliqi
>
> **备注:** 13 pages, 2 figures and 1 table
>
> **摘要:** Artificial intelligence (AI) is increasingly used in digital pathology. Publicly available histopathology datasets remain scarce, and those that do exist predominantly represent Western populations. Consequently, the generalizability of AI models to populations from less digitized regions, such as the Middle East, is largely unknown. This motivates the public release of our dataset to support the development and validation of pathology AI models across globally diverse populations. We present 339 whole-slide images of prostate core needle biopsies from a consecutive series of 185 patients collected in Erbil, Iraq. The slides are associated with Gleason scores and International Society of Urological Pathology grades assigned independently by three pathologists. Scanning was performed using two high-throughput scanners (Leica and Hamamatsu) and one compact scanner (Grundium). All slides were de-identified and are provided in their native formats without further conversion. The dataset enables grading concordance analyses, color normalization, and cross-scanner robustness evaluations. Data will be deposited in the Bioimage Archive (BIA) under accession code: to be announced (TBA), and released under a CC BY 4.0 license.
>
---
#### [new 013] Multi-Scale Visual Prompting for Lightweight Small-Image Classification
- **分类: cs.CV**

- **简介: 该论文针对小图像分类任务，解决视觉提示在低分辨率数据上应用不足的问题。提出多尺度视觉提示（MSVP），通过融合全局、中尺度和局部提示图，以极低成本提升CNN与ViT模型性能，验证了多尺度提示在小图像上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.03663v1](https://arxiv.org/pdf/2512.03663v1)**

> **作者:** Salim Khazem
>
> **摘要:** Visual prompting has recently emerged as an efficient strategy to adapt vision models using lightweight, learnable parameters injected into the input space. However, prior work mainly targets large Vision Transformers and high-resolution datasets such as ImageNet. In contrast, small-image benchmarks like MNIST, Fashion-MNIST, and CIFAR-10 remain widely used in education, prototyping, and research, yet have received little attention in the context of prompting. In this paper, we introduce \textbf{Multi-Scale Visual Prompting (MSVP)}, a simple and generic module that learns a set of global, mid-scale, and local prompt maps fused with the input image via a lightweight $1 \times 1$ convolution. MSVP is backbone-agnostic, adds less than $0.02\%$ parameters, and significantly improves performance across CNN and Vision Transformer backbones. We provide a unified benchmark on MNIST, Fashion-MNIST, and CIFAR-10 using a simple CNN, ResNet-18, and a small Vision Transformer. Our method yields consistent improvements with negligible computational overhead. We further include ablations on prompt scales, fusion strategies, and backbone architectures, along with qualitative analyzes using prompt visualizations and Grad-CAM. Our results demonstrate that multi-scale prompting provides an effective inductive bias even on low-resolution images.
>
---
#### [new 014] Divide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对长视频理解中因上下文长度限制和计算成本高的问题，提出DIG框架。通过区分全局与局部查询类型，对前者采用高效均匀采样，后者则启用查询感知采样，实现无训练的自适应帧选择，显著提升大模型性能。**

- **链接: [https://arxiv.org/pdf/2512.04000v1](https://arxiv.org/pdf/2512.04000v1)**

> **作者:** Jialuo Li; Bin Li; Jiahao Li; Yan Lu
>
> **摘要:** The application of Large Multimodal Models (LMMs) to long-form video understanding is constrained by limited context lengths and the computationally prohibitive cost of processing dense video tokens. Consequently, recent research has focused on query-aware frame selection, methods that often incur significant computational overhead. This paper challenges the assumption that such complex search mechanisms are universally necessary. We first identify and validate a query typology distinguishing between global query and localized query. We demonstrate that while uniform sampling is both effective and efficient for global queries, localized queries indeed necessitate query-aware selection for optimal performance. Building on this insight, we propose DIG, a training-free frame selection framework that adapts its strategy based on the query type. Specifically,DIG employs efficient uniform sampling for global queries while activating a specialized pipeline to extract query-relevant frames for localized queries. Experiments on three long-form video understanding benchmarks demonstrate that DIG consistently outperforms existing baselines and robustly improves LMM performance, even when scaling the input frame count to 256.
>
---
#### [new 015] Stable Signer: Hierarchical Sign Language Generative Model
- **分类: cs.CV; cs.CL; cs.CY**

- **简介: 该论文针对手语生成中多阶段误差累积问题，提出端到端的Stable Signer模型。通过简化任务流程，引入SLUL与SLP-MoE模块，实现文本到手语视频的高效生成，显著提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2512.04048v1](https://arxiv.org/pdf/2512.04048v1)**

> **作者:** Sen Fang; Yalin Feng; Hongbin Zhong; Yanxin Zhang; Dimitris N. Metaxas
>
> **备注:** 12 pages, 7 figures. More Demo at https://stablesigner.github.io
>
> **摘要:** Sign Language Production (SLP) is the process of converting the complex input text into a real video. Most previous works focused on the Text2Gloss, Gloss2Pose, Pose2Vid stages, and some concentrated on Prompt2Gloss and Text2Avatar stages. However, this field has made slow progress due to the inaccuracy of text conversion, pose generation, and the rendering of poses into real human videos in these stages, resulting in gradually accumulating errors. Therefore, in this paper, we streamline the traditional redundant structure, simplify and optimize the task objective, and design a new sign language generative model called Stable Signer. It redefines the SLP task as a hierarchical generation end-to-end task that only includes text understanding (Prompt2Gloss, Text2Gloss) and Pose2Vid, and executes text understanding through our proposed new Sign Language Understanding Linker called SLUL, and generates hand gestures through the named SLP-MoE hand gesture rendering expert block to end-to-end generate high-quality and multi-style sign language videos. SLUL is trained using the newly developed Semantic-Aware Gloss Masking Loss (SAGM Loss). Its performance has improved by 48.6% compared to the current SOTA generation methods.
>
---
#### [new 016] AfroBeats Dance Movement Analysis Using Computer Vision: A Proof-of-Concept Framework Combining YOLO and Segment Anything Model
- **分类: cs.CV**

- **简介: 该论文属于动作分析任务，旨在无需标记或设备自动量化非洲节奏舞动作。通过融合YOLO与SAM模型，实现舞者检测、步数计数、空间覆盖与节奏一致性分析，验证了框架在单视频上的可行性，但存在样本单一等局限。**

- **链接: [https://arxiv.org/pdf/2512.03509v1](https://arxiv.org/pdf/2512.03509v1)**

> **作者:** Kwaku Opoku-Ware; Gideon Opoku
>
> **摘要:** This paper presents a preliminary investigation into automated dance movement analysis using contemporary computer vision techniques. We propose a proof-of-concept framework that integrates YOLOv8 and v11 for dancer detection with the Segment Anything Model (SAM) for precise segmentation, enabling the tracking and quantification of dancer movements in video recordings without specialized equipment or markers. Our approach identifies dancers within video frames, counts discrete dance steps, calculates spatial coverage patterns, and measures rhythm consistency across performance sequences. Testing this framework on a single 49-second recording of Ghanaian AfroBeats dance demonstrates technical feasibility, with the system achieving approximately 94% detection precision and 89% recall on manually inspected samples. The pixel-level segmentation provided by SAM, achieving approximately 83% intersection-over-union with visual inspection, enables motion quantification that captures body configuration changes beyond what bounding-box approaches can represent. Analysis of this preliminary case study indicates that the dancer classified as primary by our system executed 23% more steps with 37% higher motion intensity and utilized 42% more performance space compared to dancers classified as secondary. However, this work represents an early-stage investigation with substantial limitations including single-video validation, absence of systematic ground truth annotations, and lack of comparison with existing pose estimation methods. We present this framework to demonstrate technical feasibility, identify promising directions for quantitative dance metrics, and establish a foundation for future systematic validation studies.
>
---
#### [new 017] EEA: Exploration-Exploitation Agent for Long Video Understanding
- **分类: cs.CV**

- **简介: 该论文针对长视频理解任务，解决现有方法在计算效率与信息覆盖间的平衡问题。提出EEA框架，通过语义引导的分层树搜索，动态生成语义查询并定位关键帧，结合视觉语言模型奖励与不确定性建模，实现高效探索与精准评估。**

- **链接: [https://arxiv.org/pdf/2512.03500v1](https://arxiv.org/pdf/2512.03500v1)**

> **作者:** Te Yang; Xiangyu Zhu; Bo Wang; Quan Chen; Peng Jiang; Zhen Lei
>
> **摘要:** Long-form video understanding requires efficient navigation of extensive visual data to pinpoint sparse yet critical information. Current approaches to longform video understanding either suffer from severe computational overhead due to dense preprocessing, or fail to effectively balance exploration and exploitation, resulting in incomplete information coverage and inefficiency. In this work, we introduce EEA, a novel video agent framework that archives exploration-exploitation balance through semantic guidance with hierarchical tree search process. EEA autonomously discovers and dynamically updates task-relevant semantic queries, and collects video frames closely matched to these queries as semantic anchors. During the tree search process, instead of uniform expansion, EEA preferentially explores semantically relevant frames while ensuring sufficient coverage within unknown segments. Moreover, EEA adaptively combines intrinsic rewards from visionlanguage models (VLMs) with semantic priors by explicitly modeling uncertainty to achieve stable and precise evaluation of video segments. Experiments across various long-video benchmarks validate the superior performance and computational efficiency of our proposed method.
>
---
#### [new 018] SpatialReasoner: Active Perception for Large-Scale 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文针对大尺度3D场景理解难题，提出H²U3D数据集与SpatialReasoner主动感知框架。通过多楼层、细粒度标注和链式思维问答，构建了复杂家居场景的基准。SpatialReasoner以少样本（3-4图）高效探索，结合两阶段训练实现最优性能，显著优于GPT-4o等模型。**

- **链接: [https://arxiv.org/pdf/2512.03284v1](https://arxiv.org/pdf/2512.03284v1)**

> **作者:** Hongpei Zheng; Shijie Li; Yanran Li; Hujun Yin
>
> **摘要:** Spatial reasoning in large-scale 3D environments remains challenging for current vision-language models, which are typically constrained to room-scale scenarios. We introduce H$^2$U3D (Holistic House Understanding in 3D), a 3D visual question answering dataset designed for house-scale scene understanding. H$^2$U3D features multi-floor environments spanning up to three floors and 10-20 rooms, covering more than 300 m$^2$. Through an automated annotation pipeline, it constructs hierarchical coarse-to-fine visual representations and generates diverse question-answer pairs with chain-of-thought annotations. We further propose SpatialReasoner, an active perception framework that autonomously invokes spatial tools to explore 3D scenes based on textual queries. SpatialReasoner is trained through a two-stage strategy: a supervised cold start followed by reinforcement learning with an adaptive exploration reward that promotes efficient exploration while discouraging redundant operations. Extensive experiments demonstrate that SpatialReasoner achieves state-of-the-art performance on H$^2$U3D, outperforming strong baselines including GPT-4o and Gemini-2.5-Pro. Notably, our method attains superior results while using only 3-4 images in total on average, compared to baselines requiring 16+ images, highlighting the effectiveness of our coarse-to-fine active exploration paradigm.
>
---
#### [new 019] HBFormer: A Hybrid-Bridge Transformer for Microtumor and Miniature Organ Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割任务，聚焦微肿瘤和微型器官分割难题。针对视觉Transformer局部注意力难以融合全局上下文的问题，提出HBFormer：结合U型架构与Swin Transformer的混合编码器，设计多尺度特征融合解码器，通过空洞与深度卷积构建通道与空间注意力机制，有效整合多尺度特征与全局信息，提升边界精度与上下文理解能力。**

- **链接: [https://arxiv.org/pdf/2512.03597v1](https://arxiv.org/pdf/2512.03597v1)**

> **作者:** Fuchen Zheng; Xinyi Chen; Weixuan Li; Quanjun Li; Junhua Zhou; Xiaojiao Guo; Xuhang Chen; Chi-Man Pun; Shoujun Zhou
>
> **备注:** 6 pages, 4 figures, 3 tables
>
> **摘要:** Medical image segmentation is a cornerstone of modern clinical diagnostics. While Vision Transformers that leverage shifted window-based self-attention have established new benchmarks in this field, they are often hampered by a critical limitation: their localized attention mechanism struggles to effectively fuse local details with global context. This deficiency is particularly detrimental to challenging tasks such as the segmentation of microtumors and miniature organs, where both fine-grained boundary definition and broad contextual understanding are paramount. To address this gap, we propose HBFormer, a novel Hybrid-Bridge Transformer architecture. The 'Hybrid' design of HBFormer synergizes a classic U-shaped encoder-decoder framework with a powerful Swin Transformer backbone for robust hierarchical feature extraction. The core innovation lies in its 'Bridge' mechanism, a sophisticated nexus for multi-scale feature integration. This bridge is architecturally embodied by our novel Multi-Scale Feature Fusion (MFF) decoder. Departing from conventional symmetric designs, the MFF decoder is engineered to fuse multi-scale features from the encoder with global contextual information. It achieves this through a synergistic combination of channel and spatial attention modules, which are constructed from a series of dilated and depth-wise convolutions. These components work in concert to create a powerful feature bridge that explicitly captures long-range dependencies and refines object boundaries with exceptional precision. Comprehensive experiments on challenging medical image segmentation datasets, including multi-organ, liver tumor, and bladder tumor benchmarks, demonstrate that HBFormer achieves state-of-the-art results, showcasing its outstanding capabilities in microtumor and miniature organ segmentation. Code and models are available at: https://github.com/lzeeorno/HBFormer.
>
---
#### [new 020] Learning Group Actions In Disentangled Latent Image Representations
- **分类: cs.CV**

- **简介: 该论文提出一种端到端框架，自动学习图像隐空间中的群作用，解决传统方法需手动划分变换相关与不变子空间的问题。通过可学习二值掩码实现动态分割，联合优化隐表示解耦与群变换映射，提升可控图像变换能力。**

- **链接: [https://arxiv.org/pdf/2512.04015v1](https://arxiv.org/pdf/2512.04015v1)**

> **作者:** Farhana Hossain Swarnali; Miaomiao Zhang; Tonmoy Hossain
>
> **摘要:** Modeling group actions on latent representations enables controllable transformations of high-dimensional image data. Prior works applying group-theoretic priors or modeling transformations typically operate in the high-dimensional data space, where group actions apply uniformly across the entire input, making it difficult to disentangle the subspace that varies under transformations. While latent-space methods offer greater flexibility, they still require manual partitioning of latent variables into equivariant and invariant subspaces, limiting the ability to robustly learn and operate group actions within the representation space. To address this, we introduce a novel end-to-end framework that for the first time learns group actions on latent image manifolds, automatically discovering transformation-relevant structures without manual intervention. Our method uses learnable binary masks with straight-through estimation to dynamically partition latent representations into transformation-sensitive and invariant components. We formulate this within a unified optimization framework that jointly learns latent disentanglement and group transformation mappings. The framework can be seamlessly integrated with any standard encoder-decoder architecture. We validate our approach on five 2D/3D image datasets, demonstrating its ability to automatically learn disentangled latent factors for group actions in diverse data, while downstream classification tasks confirm the effectiveness of the learned representations. Our code is publicly available at https://github.com/farhanaswarnali/Learning-Group-Actions-In-Disentangled-Latent-Image-Representations .
>
---
#### [new 021] ViDiC: Video Difference Captioning
- **分类: cs.CV**

- **简介: 该论文提出视频差异描述（ViDiC）任务，旨在解决现有视觉语言模型难以捕捉动态场景中时空变化的问题。构建了包含1000对视频的ViDiC-1K数据集，涵盖七类差异，并设计双检查表评估框架，以精准衡量模型对视频相似性与差异性的理解能力。**

- **链接: [https://arxiv.org/pdf/2512.03405v1](https://arxiv.org/pdf/2512.03405v1)**

> **作者:** Jiangtao Wu; Shihao Li; Zhaozhou Bian; Yuanxing Zhang; Jialu Chen; Runzhe Wen; An Ping; Yiwen He; Jiakai Wang; Jiaheng Liu
>
> **摘要:** Understanding visual differences between dynamic scenes requires the comparative perception of compositional, spatial, and temporal changes--a capability that remains underexplored in existing vision-language systems. While prior work on Image Difference Captioning (IDC) has enabled models to describe semantic changes between static images, these approaches fail to capture motion continuity, event evolution, or editing consistency over time. We introduce the ViDiC (Video Difference Captioning) task and its corresponding ViDiC-1K dataset, designed to evaluate the ability of Multimodal Large Language Models (MLLMs) to provide fine-grained descriptions of similarities and differences between video pairs. ViDiC-1K comprises 1,000 curated video pairs annotated with over 4,000 comparative checklist items, covering seven categories: subject, style, background, cinematography, motion, location, and playback techniques. To ensure reliable evaluation, we propose a dual-checklist framework that measures the accuracy of similarity and difference separately, based on the LLM-as-a-Judge protocol. Experiments on nineteen representative multimodal models reveal a significant performance gap in their comparative description and difference perception abilities. We hope ViDiC-1K can be a challenging benchmark that lays a solid foundation for advancing video understanding, edit awareness, and comparative reasoning in multimodal intelligence.
>
---
#### [new 022] Dynamic Optical Test for Bot Identification (DOT-BI): A simple check to identify bots in surveys and online processes
- **分类: cs.CV; cs.CR**

- **简介: 该论文提出动态光学机器人识别方法（DOT-BI），通过人类对运动的感知差异，区分真人与自动化系统。针对在线调查中机器人冒充问题，设计一种隐藏数字仅在动态变化中可见的测试，利用人眼对运动的敏感性而算法无法捕捉。实验表明，主流AI模型无法识别，而人类参与者几乎全部成功且耗时短，验证了其有效性与实用性。**

- **链接: [https://arxiv.org/pdf/2512.03580v1](https://arxiv.org/pdf/2512.03580v1)**

> **作者:** Malte Bleeker; Mauro Gotsch
>
> **摘要:** We propose the Dynamic Optical Test for Bot Identification (DOT-BI): a quick and easy method that uses human perception of motion to differentiate between human respondents and automated systems in surveys and online processes. In DOT-BI, a 'hidden' number is displayed with the same random black-and-white pixel texture as its background. Only the difference in motion and scale between the number and the background makes the number perceptible to humans across frames, while frame-by-frame algorithmic processing yields no meaningful signal. We conducted two preliminary assessments. Firstly, state-of-the-art, video-capable, multimodal models (GPT-5-Thinking and Gemini 2.5 Pro) fail to extract the correct value, even when given explicit instructions about the mechanism. Secondly, in an online survey (n=182), 99.5% (181/182) of participants solved the task, with an average end-to-end completion time of 10.7 seconds; a supervised lab study (n=39) found no negative effects on perceived ease-of-use or completion time relative to a control. We release code to generate tests and 100+ pre-rendered variants to facilitate adoption in surveys and online processes.
>
---
#### [new 023] NavMapFusion: Diffusion-based Fusion of Navigation Maps for Online Vectorized HD Map Construction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出NavMapFusion，一种基于扩散模型的在线高精地图构建方法。针对传统导航地图分辨率低、更新滞后的问题，利用低精度先验地图与高精度传感器数据融合，通过迭代去噪实现地图更新。有效提升地图准确性与实时性，推动自动驾驶环境感知的可靠性。**

- **链接: [https://arxiv.org/pdf/2512.03317v1](https://arxiv.org/pdf/2512.03317v1)**

> **作者:** Thomas Monninger; Zihan Zhang; Steffen Staab; Sihao Ding
>
> **备注:** Accepted to 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2026)
>
> **摘要:** Accurate environmental representations are essential for autonomous driving, providing the foundation for safe and efficient navigation. Traditionally, high-definition (HD) maps are providing this representation of the static road infrastructure to the autonomous system a priori. However, because the real world is constantly changing, such maps must be constructed online from on-board sensor data. Navigation-grade standard-definition (SD) maps are widely available, but their resolution is insufficient for direct deployment. Instead, they can be used as coarse prior to guide the online map construction process. We propose NavMapFusion, a diffusion-based framework that performs iterative denoising conditioned on high-fidelity sensor data and on low-fidelity navigation maps. This paper strives to answer: (1) How can coarse, potentially outdated navigation maps guide online map construction? (2) What advantages do diffusion models offer for map fusion? We demonstrate that diffusion-based map construction provides a robust framework for map fusion. Our key insight is that discrepancies between the prior map and online perception naturally correspond to noise within the diffusion process; consistent regions reinforce the map construction, whereas outdated segments are suppressed. On the nuScenes benchmark, NavMapFusion conditioned on coarse road lines from OpenStreetMap data reaches a 21.4% relative improvement on 100 m, and even stronger improvements on larger perception ranges, while maintaining real-time capabilities. By fusing low-fidelity priors with high-fidelity sensor data, the proposed method generates accurate and up-to-date environment representations, guiding towards safer and more reliable autonomous driving. The code is available at https://github.com/tmonnin/navmapfusion
>
---
#### [new 024] Drainage: A Unifying Framework for Addressing Class Uncertainty
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对深度学习中的标签噪声与类别不确定性问题，提出基于“排水节点”的统一框架。通过在网络输出层添加可微的排水节点，将不确定样本的概率质量重新分配，增强模型对异常值、噪声和分布外样本的鲁棒性。实验表明，该方法在高噪声环境下显著提升分类准确率，并适用于半监督清洗与开放集识别等任务。**

- **链接: [https://arxiv.org/pdf/2512.03182v1](https://arxiv.org/pdf/2512.03182v1)**

> **作者:** Yasser Taha; Grégoire Montavon; Nils Körber
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Modern deep learning faces significant challenges with noisy labels, class ambiguity, as well as the need to robustly reject out-of-distribution or corrupted samples. In this work, we propose a unified framework based on the concept of a "drainage node'' which we add at the output of the network. The node serves to reallocate probability mass toward uncertainty, while preserving desirable properties such as end-to-end training and differentiability. This mechanism provides a natural escape route for highly ambiguous, anomalous, or noisy samples, particularly relevant for instance-dependent and asymmetric label noise. In systematic experiments involving the addition of varying proportions of instance-dependent noise or asymmetric noise to CIFAR-10/100 labels, our drainage formulation achieves an accuracy increase of up to 9\% over existing approaches in the high-noise regime. Our results on real-world datasets, such as mini-WebVision, mini-ImageNet and Clothing-1M, match or surpass existing state-of-the-art methods. Qualitative analysis reveals a denoising effect, where the drainage neuron consistently absorbs corrupt, mislabeled, or outlier data, leading to more stable decision boundaries. Furthermore, our drainage formulation enables applications well beyond classification, with immediate benefits for web-scale, semi-supervised dataset cleaning, and open-set applications.
>
---
#### [new 025] Generalization Evaluation of Deep Stereo Matching Methods for UAV-Based Forestry Applications
- **分类: cs.CV**

- **简介: 该论文针对无人机林业应用中的深度估计泛化问题，首次系统评估了八种先进立体匹配方法在植被密集环境下的零样本性能。通过新构建的林业数据集与多基准测试，发现基础模型在结构化场景表现优，而迭代方法更具跨域鲁棒性，DEFOM为最优基线。**

- **链接: [https://arxiv.org/pdf/2512.03427v1](https://arxiv.org/pdf/2512.03427v1)**

> **作者:** Yida Lin; Bing Xue; Mengjie Zhang; Sam Schofield; Richard Green
>
> **摘要:** Autonomous UAV forestry operations require robust depth estimation methods with strong cross-domain generalization. However, existing evaluations focus on urban and indoor scenarios, leaving a critical gap for specialized vegetation-dense environments. We present the first systematic zero-shot evaluation of eight state-of-the-art stereo methods--RAFT-Stereo, IGEV, IGEV++, BridgeDepth, StereoAnywhere, DEFOM (plus baseline methods ACVNet, PSMNet, TCstereo)--spanning iterative refinement, foundation model, and zero-shot adaptation paradigms. All methods are trained exclusively on Scene Flow and evaluated without fine-tuning on four standard benchmarks (ETH3D, KITTI 2012/2015, Middlebury) plus a novel 5,313-pair Canterbury forestry dataset captured with ZED Mini camera (1920x1080). Performance reveals scene-dependent patterns: foundation models excel on structured scenes (BridgeDepth: 0.23 px on ETH3D, 0.83-1.07 px on KITTI; DEFOM: 0.35-4.65 px across benchmarks), while iterative methods maintain cross-domain robustness (IGEV++: 0.36-6.77 px; IGEV: 0.33-21.91 px). Critical finding: RAFT-Stereo exhibits catastrophic ETH3D failure (26.23 px EPE, 98 percent error rate) due to negative disparity predictions, while performing normally on KITTI (0.90-1.11 px). Qualitative evaluation on Canterbury forestry dataset identifies DEFOM as the optimal gold-standard baseline for vegetation depth estimation, exhibiting superior depth smoothness, occlusion handling, and cross-domain consistency compared to IGEV++, despite IGEV++'s finer detail preservation.
>
---
#### [new 026] A Robust Camera-based Method for Breath Rate Measurement
- **分类: cs.CV**

- **简介: 该论文属于呼吸率检测任务，旨在解决现有视频呼吸率测量方法在非理想条件下精度不足的问题。提出一种基于数学变换的鲁棒方法，仅需普通摄像头，通过分析视频中人体运动变化，实现高精度呼吸率估计，平均绝对误差仅0.57次/分钟，且对运动干扰具有较强抗性。**

- **链接: [https://arxiv.org/pdf/2512.03827v1](https://arxiv.org/pdf/2512.03827v1)**

> **作者:** Alexey Protopopov
>
> **备注:** 9 pages, 4 figures, 2 tables
>
> **摘要:** Proliferation of cheap and accessible cameras makes it possible to measure a subject's breath rate from video footage alone. Recent works on this topic have proposed a variety of approaches for accurately measuring human breath rate, however they are either tested in near-ideal conditions, or produce results that are not sufficiently accurate. The present study proposes a more robust method to measure breath rate in humans with minimal hardware requirements using a combination of mathematical transforms with a relative deviation from the ground truth of less than 5%. The method was tested on videos taken from 14 volunteers with a total duration of over 2 hours 30 minutes. The obtained results were compared to reference data and the average mean absolute error was found to be at 0.57 respirations per minute, which is noticeably better than the results from previous works. The breath rate measurement method proposed in the present article is more resistant to distortions caused by subject movement and thus allows one to remotely measure the subject's breath rate without any significant limitations on the subject's behavior.
>
---
#### [new 027] GAOT: Generating Articulated Objects Through Text-Guided Diffusion Models
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出GAOT框架，解决文本引导下3D可动物体生成问题。通过三阶段流程：先用微调点云模型生成粗略物体，再用超图学习细化部件结构，最后基于扩散模型生成关节连接，实现从文本到可动物体的高质量生成。**

- **链接: [https://arxiv.org/pdf/2512.03566v1](https://arxiv.org/pdf/2512.03566v1)**

> **作者:** Hao Sun; Lei Fan; Donglin Di; Shaohui Liu
>
> **备注:** Accepted by ACM MM Asia2026
>
> **摘要:** Articulated object generation has seen increasing advancements, yet existing models often lack the ability to be conditioned on text prompts. To address the significant gap between textual descriptions and 3D articulated object representations, we propose GAOT, a three-phase framework that generates articulated objects from text prompts, leveraging diffusion models and hypergraph learning in a three-step process. First, we fine-tune a point cloud generation model to produce a coarse representation of objects from text prompts. Given the inherent connection between articulated objects and graph structures, we design a hypergraph-based learning method to refine these coarse representations, representing object parts as graph vertices. Finally, leveraging a diffusion model, the joints of articulated objects-represented as graph edges-are generated based on the object parts. Extensive qualitative and quantitative experiments on the PartNet-Mobility dataset demonstrate the effectiveness of our approach, achieving superior performance over previous methods.
>
---
#### [new 028] Difference Decomposition Networks for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文针对红外小目标检测（ISTD）中目标纹理缺失与背景杂乱问题，提出基于基分解的差异分解模块（BDM），构建SD²Net（单帧）和STD²Net（多帧）网络，通过空间与时间差异分解增强目标、抑制背景，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.03470v1](https://arxiv.org/pdf/2512.03470v1)**

> **作者:** Chen Hu; Mingyu Zhou; Shuai Yuan; Hongbo Hu; Xiangyu Qiu; Junhai Luo; Tian Pu; Xiyin Li
>
> **摘要:** Infrared small target detection (ISTD) faces two major challenges: a lack of discernible target texture and severe background clutter, which results in the background obscuring the target. To enhance targets and suppress backgrounds, we propose the Basis Decomposition Module (BDM) as an extensible and lightweight module based on basis decomposition, which decomposes a complex feature into several basis features and enhances certain information while eliminating redundancy. Extending BDM leads to a series of modules, including the Spatial Difference Decomposition Module (SD$^\mathrm{2}$M), Spatial Difference Decomposition Downsampling Module (SD$^\mathrm{3}$M), and Temporal Difference Decomposition Module (TD$^\mathrm{2}$M). Based on these modules, we develop the Spatial Difference Decomposition Network (SD$^\mathrm{2}$Net) for single-frame ISTD (SISTD) and the Spatiotemporal Difference Decomposition Network (STD$^\mathrm{2}$Net) for multi-frame ISTD (MISTD). SD$^\mathrm{2}$Net integrates SD$^\mathrm{2}$M and SD$^\mathrm{3}$M within an adapted U-shaped architecture. We employ TD$^\mathrm{2}$M to introduce motion information, which transforms SD$^\mathrm{2}$Net into STD$^\mathrm{2}$Net. Extensive experiments on SISTD and MISTD datasets demonstrate state-of-the-art (SOTA) performance. On the SISTD task, SD$^\mathrm{2}$Net performs well compared to most established networks. On the MISTD datasets, STD$^\mathrm{2}$Net achieves a mIoU of 87.68\%, outperforming SD$^\mathrm{2}$Net, which achieves a mIoU of 64.97\%. Our codes are available: https://github.com/greekinRoma/IRSTD_HC_Platform.
>
---
#### [new 029] Optical Context Compression Is Just (Bad) Autoencoding
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对视觉上下文压缩在语言模型中的应用，检验其有效性。通过对比视觉编码器与简单替代方法，发现后者在文本重建和语言建模任务中表现相当或更优，且视觉压缩无法超越截断策略。结论质疑了光学压缩的优越性，指出当前兴奋过度，缺乏实证支持。**

- **链接: [https://arxiv.org/pdf/2512.03643v1](https://arxiv.org/pdf/2512.03643v1)**

> **作者:** Ivan Yee Lee; Cheng Yang; Taylor Berg-Kirkpatrick
>
> **摘要:** DeepSeek-OCR demonstrates that rendered text can be reconstructed with high fidelity from a small number of vision tokens. This finding has sparked excitement about vision-based context compression for language models. But the evaluation stops at reconstruction; whether these representations help language modeling remains untested. We test two assumptions implicit in the optical-compression narrative: that vision-based compression provides unique advantages for text reconstruction from compressed representations, and that DeepSeek-OCR's reconstruction results are evidence that vision-based compression will be useful for language modeling. Comparing their vision encoder against simple alternatives--parameter-free mean pooling and a learned hierarchical encoder--we find that these simple approaches match or surpass vision for reconstruction at matched compression ratios, and outperform it for language modeling--where vision-based compression fails to beat truncation. The excitement around optical context compression outpaces the evidence. Code and checkpoints are available at https://github.com/ivnle/bad-autoencoding
>
---
#### [new 030] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对光学与合成孔径雷达（SAR）图像间的跨模态船舶重识别任务，旨在缓解二者巨大的模态差距。提出MOS框架，通过特征对齐与跨模态数据生成融合，实现模态一致的特征学习，显著提升识别性能。**

- **链接: [https://arxiv.org/pdf/2512.03404v1](https://arxiv.org/pdf/2512.03404v1)**

> **作者:** Yujian Zhao; Hankun Liu; Guanglin Niu
>
> **摘要:** Cross-modal ship re-identification (ReID) between optical and synthetic aperture radar (SAR) imagery has recently emerged as a critical yet underexplored task in maritime intelligence and surveillance. However, the substantial modality gap between optical and SAR images poses a major challenge for robust identification. To address this issue, we propose MOS, a novel framework designed to mitigate the optical-SAR modality gap and achieve modality-consistent feature learning for optical-SAR cross-modal ship ReID. MOS consists of two core components: (1) Modality-Consistent Representation Learning (MCRL) applies denoise SAR image procession and a class-wise modality alignment loss to align intra-identity feature distributions across modalities. (2) Cross-modal Data Generation and Feature fusion (CDGF) leverages a brownian bridge diffusion model to synthesize cross-modal samples, which are subsequently fused with original features during inference to enhance alignment and discriminability. Extensive experiments on the HOSS ReID dataset demonstrate that MOS significantly surpasses state-of-the-art methods across all evaluation protocols, achieving notable improvements of +3.0%, +6.2%, and +16.4% in R1 accuracy under the ALL to ALL, Optical to SAR, and SAR to Optical settings, respectively. The code and trained models will be released upon publication.
>
---
#### [new 031] UniComp: Rethinking Video Compression Through Informational Uniqueness
- **分类: cs.CV**

- **简介: 该论文针对视频压缩任务，提出基于信息独特性的UniComp框架。通过最小化条件熵优化压缩，引入信息独特性度量冗余，设计三模块实现语义分组、资源分配与空间压缩，有效提升有限算力下视觉信息保真度。**

- **链接: [https://arxiv.org/pdf/2512.03575v1](https://arxiv.org/pdf/2512.03575v1)**

> **作者:** Chao Yuan; Shimin Chen; Minliang Lin; Limeng Qiao; Guanglu Wan; Lin Ma
>
> **摘要:** Distinct from attention-based compression methods, this paper presents an information uniqueness driven video compression framework, termed UniComp, which aims to maximize the information fidelity of video representations under constrained computational budgets. Starting from the information-theoretic perspective, we formulate the vision compression as an optimization problem that minimizes conditional entropy (reconstruction error) between retained and full tokens. To achieve this, we introduce the notion of information uniqueness to measure intrinsic redundancy among tokens to link with reconstruction error. Based on uniqueness, we design three modules-Frame Group Fusion, Token Allocation, and Spatial Dynamic Compression-that progressively perform semantic frame grouping, adaptive resource allocation, and fine-grained spatial compression. Extensive experiments demonstrate that UniComp consistently outperforms existing compression methods in preserving essential visual tokens under limited computational budgets, highlighting the pivotal role of information uniqueness in token compression efficacy.
>
---
#### [new 032] CloseUpAvatar: High-Fidelity Animatable Full-Body Avatars with Mixture of Multi-Scale Textures
- **分类: cs.CV**

- **简介: 该论文提出CloseUpAvatar，用于高保真可动画全身虚拟人像的表示。针对一般相机运动下近景渲染质量下降的问题，通过多尺度纹理混合实现距离感知的细节控制，动态调整高频纹理使用，提升远近视角下的视觉真实感与渲染效率。**

- **链接: [https://arxiv.org/pdf/2512.03593v1](https://arxiv.org/pdf/2512.03593v1)**

> **作者:** David Svitov; Pietro Morerio; Lourdes Agapito; Alessio Del Bue
>
> **摘要:** We present a CloseUpAvatar - a novel approach for articulated human avatar representation dealing with more general camera motions, while preserving rendering quality for close-up views. CloseUpAvatar represents an avatar as a set of textured planes with two sets of learnable textures for low and high-frequency detail. The method automatically switches to high-frequency textures only for cameras positioned close to the avatar's surface and gradually reduces their impact as the camera moves farther away. Such parametrization of the avatar enables CloseUpAvatar to adjust rendering quality based on camera distance ensuring realistic rendering across a wider range of camera orientations than previous approaches. We provide experiments using the ActorsHQ dataset with high-resolution input images. CloseUpAvatar demonstrates both qualitative and quantitative improvements over existing methods in rendering from novel wide range camera positions, while maintaining high FPS by limiting the number of required primitives.
>
---
#### [new 033] Research on Brain Tumor Classification Method Based on Improved ResNet34 Network
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在解决脑肿瘤图像手动分类效率低、传统模型精度不足的问题。提出改进的ResNet34网络，引入多尺度输入、Inception v2下采样及通道注意力机制，提升分类准确率至98.8%，参数量仅为原模型的80%。**

- **链接: [https://arxiv.org/pdf/2512.03751v1](https://arxiv.org/pdf/2512.03751v1)**

> **作者:** Yufeng Li; Wenchao Zhao; Bo Dang; Weimin Wang
>
> **摘要:** Previously, image interpretation in radiology relied heavily on manual methods. However, manual classification of brain tumor medical images is time-consuming and labor-intensive. Even with shallow convolutional neural network models, the accuracy is not ideal. To improve the efficiency and accuracy of brain tumor image classification, this paper proposes a brain tumor classification model based on an improved ResNet34 network. This model uses the ResNet34 residual network as the backbone network and incorporates multi-scale feature extraction. It uses a multi-scale input module as the first layer of the ResNet34 network and an Inception v2 module as the residual downsampling layer. Furthermore, a channel attention mechanism module assigns different weights to different channels of the image from a channel domain perspective, obtaining more important feature information. The results after a five-fold crossover experiment show that the average classification accuracy of the improved network model is approximately 98.8%, which is not only 1% higher than ResNet34, but also only 80% of the number of parameters of the original model. Therefore, the improved network model not only improves accuracy but also reduces clutter, achieving a classification effect with fewer parameters and higher accuracy.
>
---
#### [new 034] UniMo: Unifying 2D Video and 3D Human Motion with an Autoregressive Framework
- **分类: cs.CV**

- **简介: 该论文提出UniMo，一个统一建模2D视频与3D人体运动的自回归框架。针对两者结构差异大、联合生成难的问题，将二者视为统一标记序列，设计专用嵌入与3D运动分层量化器，实现双向同步生成与精准动作捕捉，推动多模态人体建模发展。**

- **链接: [https://arxiv.org/pdf/2512.03918v1](https://arxiv.org/pdf/2512.03918v1)**

> **作者:** Youxin Pang; Yong Zhang; Ruizhi Shao; Xiang Deng; Feng Gao; Xu Xiaoming; Xiaoming Wei; Yebin Liu
>
> **备注:** https://carlyx.github.io/UniMo/
>
> **摘要:** We propose UniMo, an innovative autoregressive model for joint modeling of 2D human videos and 3D human motions within a unified framework, enabling simultaneous generation and understanding of these two modalities for the first time. Current methods predominantly focus on generating one modality given another as the condition or integrating either of them with other modalities such as text and audio. Unifying 2D videos and 3D motions for simultaneous optimization and generation remains largely unexplored, presenting significant challenges due to their substantial structural and distributional differences. Inspired by the LLM's ability to unify different modalities, our method models videos and 3D motions as a unified tokens sequence, utilizing separate embedding layers to mitigate distribution gaps. Additionally, we devise a sequence modeling strategy that integrates two distinct tasks within a single framework, proving the effectiveness of unified modeling. Moreover, to efficiently align with visual tokens and preserve 3D spatial information, we design a novel 3D motion tokenizer with a temporal expansion strategy, using a single VQ-VAE to produce quantized motion tokens. It features multiple expert decoders that handle body shapes, translation, global orientation, and body poses for reliable 3D motion reconstruction. Extensive experiments demonstrate that our method simultaneously generates corresponding videos and motions while performing accurate motion capture. This work taps into the capacity of LLMs to fuse diverse data types, paving the way for integrating human-centric information into existing models and potentially enabling multimodal, controllable joint modeling of humans, objects, and scenes.
>
---
#### [new 035] Towards Object-centric Understanding for Instructional Videos
- **分类: cs.CV**

- **简介: 该论文针对指令视频理解任务，解决现有方法因侧重动作而难以应对步骤顺序灵活的问题。提出对象中心范式，构建Object-IVQA基准，设计代理框架实现对象级推理与多跳证据检索，显著提升模型在状态演变、错判识别等维度的表现。**

- **链接: [https://arxiv.org/pdf/2512.03479v1](https://arxiv.org/pdf/2512.03479v1)**

> **作者:** Wenliang Guo; Yu Kong
>
> **摘要:** Understanding procedural activities is crucial for developing future assistive AI that can reason about complex real-world tasks. Existing action-centric methods struggle with the flexibility of real procedures, where step order varies depending on object states. In this work, we propose to shift the focus to an object-centric paradigm by regarding actions as mechanisms that drive state transitions. To advance this direction, we introduce Object-IVQA, a long-form instructional video benchmark with 107 videos and 514 open-ended question-answer pairs annotated with temporally grounded evidence. The benchmark evaluates four dimensions of object-centric reasoning, including state evolution, precondition verification, counterfactual reasoning and mistake recognition. We further propose an agent framework that orchestrates object-centric planning, perception, analysis and generation tools, enabling explicit evidence retrieval and multi-hop reasoning across disjoint segments. Experiments show that existing large vision-language models struggle in object-level recognition and reasoning, whereas our framework achieves substantially improvement.
>
---
#### [new 036] LSRS: Latent Scale Rejection Sampling for Visual Autoregressive Modeling
- **分类: cs.CV**

- **简介: 该论文针对视觉自回归模型（VAR）生成图像时因并行采样导致的结构错误问题，提出潜空间拒绝采样（LSRS）方法。通过轻量级评分模型在每尺度筛选高质量候选图块，逐步优化生成结果，有效提升图像质量并保持高效推理。**

- **链接: [https://arxiv.org/pdf/2512.03796v1](https://arxiv.org/pdf/2512.03796v1)**

> **作者:** Hong-Kai Zheng; Piji Li
>
> **摘要:** Visual Autoregressive (VAR) modeling approach for image generation proposes autoregressive processing across hierarchical scales, decoding multiple tokens per scale in parallel. This method achieves high-quality generation while accelerating synthesis. However, parallel token sampling within a scale may lead to structural errors, resulting in suboptimal generated images. To mitigate this, we propose Latent Scale Rejection Sampling (LSRS), a method that progressively refines token maps in the latent scale during inference to enhance VAR models. Our method uses a lightweight scoring model to evaluate multiple candidate token maps sampled at each scale, selecting the high-quality map to guide subsequent scale generation. By prioritizing early scales critical for structural coherence, LSRS effectively mitigates autoregressive error accumulation while maintaining computational efficiency. Experiments demonstrate that LSRS significantly improves VAR's generation quality with minimal additional computational overhead. For the VAR-d30 model, LSRS increases the inference time by merely 1% while reducing its FID score from 1.95 to 1.78. When the inference time is increased by 15%, the FID score can be further reduced to 1.66. LSRS offers an efficient test-time scaling solution for enhancing VAR-based generation.
>
---
#### [new 037] LAMP: Language-Assisted Motion Planning for Controllable Video Generation
- **分类: cs.CV**

- **简介: 该论文提出LAMP框架，解决视频生成中运动控制难的问题。通过大语言模型将自然语言描述转化为3D运动轨迹，实现对象与相机运动的可控生成。构建了大规模运动程序数据集，显著提升运动可控性与用户意图对齐度，是首个直接从自然语言生成物体与相机运动的框架。**

- **链接: [https://arxiv.org/pdf/2512.03619v1](https://arxiv.org/pdf/2512.03619v1)**

> **作者:** Muhammed Burak Kizil; Enes Sanli; Niloy J. Mitra; Erkut Erdem; Aykut Erdem; Duygu Ceylan
>
> **摘要:** Video generation has achieved remarkable progress in visual fidelity and controllability, enabling conditioning on text, layout, or motion. Among these, motion control - specifying object dynamics and camera trajectories - is essential for composing complex, cinematic scenes, yet existing interfaces remain limited. We introduce LAMP that leverages large language models (LLMs) as motion planners to translate natural language descriptions into explicit 3D trajectories for dynamic objects and (relatively defined) cameras. LAMP defines a motion domain-specific language (DSL), inspired by cinematography conventions. By harnessing program synthesis capabilities of LLMs, LAMP generates structured motion programs from natural language, which are deterministically mapped to 3D trajectories. We construct a large-scale procedural dataset pairing natural text descriptions with corresponding motion programs and 3D trajectories. Experiments demonstrate LAMP's improved performance in motion controllability and alignment with user intent compared to state-of-the-art alternatives establishing the first framework for generating both object and camera motions directly from natural language specifications.
>
---
#### [new 038] ProtoEFNet: Dynamic Prototype Learning for Inherently Interpretable Ejection Fraction Estimation in Echocardiography
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ProtoEFNet，用于超声心动图中射血分数（EF）的可解释性估计。针对传统方法耗时及深度学习模型黑箱问题，提出动态时空原型学习与原型角度分离损失，实现连续EF回归的同时提供临床可理解的推理过程，提升模型透明度与可信度。**

- **链接: [https://arxiv.org/pdf/2512.03339v1](https://arxiv.org/pdf/2512.03339v1)**

> **作者:** Yeganeh Ghamary; Victoria Wu; Hooman Vaseli; Christina Luong; Teresa Tsang; Siavash Bigdeli; Purang Abolmaesumi
>
> **备注:** 11 pages, Accepted in IMIMIC Workshop at MICCAI 2025
>
> **摘要:** Ejection fraction (EF) is a crucial metric for assessing cardiac function and diagnosing conditions such as heart failure. Traditionally, EF estimation requires manual tracing and domain expertise, making the process time-consuming and subject to interobserver variability. Most current deep learning methods for EF prediction are black-box models with limited transparency, which reduces clinical trust. Some post-hoc explainability methods have been proposed to interpret the decision-making process after the prediction is made. However, these explanations do not guide the model's internal reasoning and therefore offer limited reliability in clinical applications. To address this, we introduce ProtoEFNet, a novel video-based prototype learning model for continuous EF regression. The model learns dynamic spatiotemporal prototypes that capture clinically meaningful cardiac motion patterns. Additionally, the proposed Prototype Angular Separation (PAS) loss enforces discriminative representations across the continuous EF spectrum. Our experiments on the EchonetDynamic dataset show that ProtoEFNet can achieve accuracy on par with its non-interpretable counterpart while providing clinically relevant insight. The ablation study shows that the proposed loss boosts performance with a 2% increase in F1 score from 77.67$\pm$2.68 to 79.64$\pm$2.10. Our source code is available at: https://github.com/DeepRCL/ProtoEF
>
---
#### [new 039] Does Head Pose Correction Improve Biometric Facial Recognition?
- **分类: cs.CV**

- **简介: 该论文研究面部识别在真实场景下因姿态偏斜、图像质量差导致的准确率下降问题。针对此，评估三种AI修复方法（3D重建、2D正面化、特征增强）对识别性能的影响。结果表明，盲目使用会降低准确率，但选择性结合CFR-GAN与CodeFormer可有效提升识别效果。**

- **链接: [https://arxiv.org/pdf/2512.03199v1](https://arxiv.org/pdf/2512.03199v1)**

> **作者:** Justin Norman; Hany Farid
>
> **摘要:** Biometric facial recognition models often demonstrate significant decreases in accuracy when processing real-world images, often characterized by poor quality, non-frontal subject poses, and subject occlusions. We investigate whether targeted, AI-driven, head-pose correction and image restoration can improve recognition accuracy. Using a model-agnostic, large-scale, forensic-evaluation pipeline, we assess the impact of three restoration approaches: 3D reconstruction (NextFace), 2D frontalization (CFR-GAN), and feature enhancement (CodeFormer). We find that naive application of these techniques substantially degrades facial recognition accuracy. However, we also find that selective application of CFR-GAN combined with CodeFormer yields meaningful improvements.
>
---
#### [new 040] Hierarchical Process Reward Models are Symbolic Vision Learners
- **分类: cs.CV**

- **简介: 该论文提出一种基于符号化视觉的自监督学习框架，旨在解决传统像素模型难以实现可解释性理解的问题。通过构建层次化过程奖励机制，将图表解析为几何原语及其关系，在保持高精度重建的同时提升推理能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.03126v1](https://arxiv.org/pdf/2512.03126v1)**

> **作者:** Shan Zhang; Aotian Chen; Kai Zou; Jindong Gu; Yuan Xue; Anton van den Hengel
>
> **摘要:** Symbolic computer vision represents diagrams through explicit logical rules and structured representations, enabling interpretable understanding in machine vision. This requires fundamentally different learning paradigms from pixel-based visual models. Symbolic visual learners parse diagrams into geometric primitives-points, lines, and shapes-whereas pixel-based learners operate on textures and colors. We propose a novel self-supervised symbolic auto-encoder that encodes diagrams into structured primitives and their interrelationships within the latent space, and decodes them through our executable engine to reconstruct the input diagrams. Central to this architecture is Symbolic Hierarchical Process Reward Modeling, which applies hierarchical step-level parsing rewards to enforce point-on-line, line-on-shape, and shape-on-relation consistency. Since vanilla reinforcement learning exhibits poor exploration in the policy space during diagram reconstruction; we thus introduce stabilization mechanisms to balance exploration and exploitation. We fine-tune our symbolic encoder on downstream tasks, developing a neuro-symbolic system that integrates the reasoning capabilities of neural networks with the interpretability of symbolic models through reasoning-grounded visual rewards. Evaluations across reconstruction, perception, and reasoning tasks demonstrate the effectiveness of our approach: achieving a 98.2% reduction in MSE for geometric diagram reconstruction, surpassing GPT-4o by 0.6% with a 7B model on chart reconstruction, and improving by +13% on the MathGlance perception benchmark, and by +3% on MathVerse and GeoQA reasoning benchmarks.
>
---
#### [new 041] Think Before You Drive: World Model-Inspired Multimodal Grounding for Autonomous Vehicles
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对自动驾驶中自然语言指令的视觉定位任务，解决指令歧义与场景动态变化带来的定位难题。提出ThinkDeeper框架，基于世界模型预判未来空间状态，结合超图引导解码，实现对复杂场景的鲁棒定位。构建DrivePilot数据集，验证方法在多基准上的领先性能。**

- **链接: [https://arxiv.org/pdf/2512.03454v1](https://arxiv.org/pdf/2512.03454v1)**

> **作者:** Haicheng Liao; Huanming Shen; Bonan Wang; Yongkang Li; Yihong Tang; Chengyue Wang; Dingyi Zhuang; Kehua Chen; Hai Yang; Chengzhong Xu; Zhenning Li
>
> **摘要:** Interpreting natural-language commands to localize target objects is critical for autonomous driving (AD). Existing visual grounding (VG) methods for autonomous vehicles (AVs) typically struggle with ambiguous, context-dependent instructions, as they lack reasoning over 3D spatial relations and anticipated scene evolution. Grounded in the principles of world models, we propose ThinkDeeper, a framework that reasons about future spatial states before making grounding decisions. At its core is a Spatial-Aware World Model (SA-WM) that learns to reason ahead by distilling the current scene into a command-aware latent state and rolling out a sequence of future latent states, providing forward-looking cues for disambiguation. Complementing this, a hypergraph-guided decoder then hierarchically fuses these states with the multimodal input, capturing higher-order spatial dependencies for robust localization. In addition, we present DrivePilot, a multi-source VG dataset in AD, featuring semantic annotations generated by a Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT)-prompted LLM pipeline. Extensive evaluations on six benchmarks, ThinkDeeper ranks #1 on the Talk2Car leaderboard and surpasses state-of-the-art baselines on DrivePilot, MoCAD, and RefCOCO/+/g benchmarks. Notably, it shows strong robustness and efficiency in challenging scenes (long-text, multi-agent, ambiguity) and retains superior performance even when trained on 50% of the data.
>
---
#### [new 042] Multi-Aspect Knowledge-Enhanced Medical Vision-Language Pretraining with Multi-Agent Data Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医疗视觉-语言预训练中网页数据噪声多、长文本难处理的问题，提出MAGEN数据生成与O-MAKE知识增强框架。通过多智能体生成高质量图文对，并基于医学本体分解文本，实现细粒度跨模态对齐。在皮肤科任务上取得领先零样本性能。**

- **链接: [https://arxiv.org/pdf/2512.03445v1](https://arxiv.org/pdf/2512.03445v1)**

> **作者:** Xieji Li; Siyuan Yan; Yingsheng Liu; H. Peter Soyer; Monika Janda; Victoria Mar; Zongyuan Ge
>
> **备注:** 10 pages. Under Review
>
> **摘要:** Vision-language pretraining (VLP) has emerged as a powerful paradigm in medical image analysis, enabling representation learning from large-scale image-text pairs without relying on expensive manual annotations. However, existing methods often struggle with the noise inherent in web-collected data and the complexity of unstructured long medical texts. To address these challenges, we propose a novel VLP framework integrating a Multi-Agent data GENeration (MAGEN) system and Ontology-based Multi-Aspect Knowledge-Enhanced (O-MAKE) pretraining. First, MAGEN enhances data quality by synthesizing knowledge-enriched descriptions via a foundation model-assisted captioning and retrieval-based verification pipeline. Second, O-MAKE addresses the difficulty of learning from long, unstructured texts by decomposing them into distinct knowledge aspects. This facilitates fine-grained alignment at both global and patch levels, while explicitly modeling medical concept relationships through ontology-guided mechanisms. We validate our framework in the field of dermatology, where comprehensive experiments demonstrate the effectiveness of each component. Our approach achieves state-of-the-art zero-shot performance on disease classification and cross-modal retrieval tasks across eight datasets. Our code and the augmented dataset Derm1M-AgentAug, comprising over 400k skin-image-text pairs, will be released at https://github.com/SiyuanYan1/Derm1M.
>
---
#### [new 043] 2-Shots in the Dark: Low-Light Denoising with Minimal Data Acquisition
- **分类: cs.CV**

- **简介: 该论文针对低光图像去噪任务，解决真实配对数据难获取的问题。提出仅需单张噪声图和暗帧的噪声合成方法，结合泊松分布与傅里叶域谱采样，精准建模信号相关与无关噪声，实现高保真噪声生成，显著提升去噪性能。**

- **链接: [https://arxiv.org/pdf/2512.03245v1](https://arxiv.org/pdf/2512.03245v1)**

> **作者:** Liying Lu; Raphaël Achddou; Sabine Süsstrunk
>
> **摘要:** Raw images taken in low-light conditions are very noisy due to low photon count and sensor noise. Learning-based denoisers have the potential to reconstruct high-quality images. For training, however, these denoisers require large paired datasets of clean and noisy images, which are difficult to collect. Noise synthesis is an alternative to large-scale data acquisition: given a clean image, we can synthesize a realistic noisy counterpart. In this work, we propose a general and practical noise synthesis method that requires only one single noisy image and one single dark frame per ISO setting. We represent signal-dependent noise with a Poisson distribution and introduce a Fourier-domain spectral sampling algorithm to accurately model signal-independent noise. The latter generates diverse noise realizations that maintain the spatial and statistical properties of real sensor noise. As opposed to competing approaches, our method neither relies on simplified parametric models nor on large sets of clean-noisy image pairs. Our synthesis method is not only accurate and practical, it also leads to state-of-the-art performances on multiple low-light denoising benchmarks.
>
---
#### [new 044] PyroFocus: A Deep Learning Approach to Real-Time Wildfire Detection in Multispectral Remote Sensing Imagery
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对实时火灾检测任务，解决高维遥感数据下计算资源受限的挑战。提出PyroFocus两阶段框架，结合分类与辐射功率回归/分割，提升检测效率与精度，适用于星载/机载平台实时部署。**

- **链接: [https://arxiv.org/pdf/2512.03257v1](https://arxiv.org/pdf/2512.03257v1)**

> **作者:** Mark Moussa; Andre Williams; Seth Roffe; Douglas Morton
>
> **摘要:** Rapid and accurate wildfire detection is crucial for emergency response and environmental management. In airborne and spaceborne missions, real-time algorithms must distinguish between no fire, active fire, and post-fire conditions, and estimate fire intensity. Multispectral and hyperspectral thermal imagers provide rich spectral information, but high data dimensionality and limited onboard resources make real-time processing challenging. As wildfires increase in frequency and severity, the need for low-latency and computationally efficient onboard detection methods is critical. We present a systematic evaluation of multiple deep learning architectures, including custom Convolutional Neural Networks (CNNs) and Transformer-based models, for multi-class fire classification. We also introduce PyroFocus, a two-stage pipeline that performs fire classification followed by fire radiative power (FRP) regression or segmentation to reduce inference time and computational cost for onboard deployment. Using data from NASA's MODIS/ASTER Airborne Simulator (MASTER), which is similar to a next-generation fire detection sensor, we compare accuracy, inference latency, and resource efficiency. Experimental results show that the proposed two-stage pipeline achieves strong trade-offs between speed and accuracy, demonstrating significant potential for real-time edge deployment in future wildfire monitoring missions.
>
---
#### [new 045] MUT3R: Motion-aware Updating Transformer for Dynamic 3D Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对动态3D重建中运动引起的伪影问题，提出MUT3R框架。通过分析自注意力图发现预训练模型隐含运动线索，设计无训练的注意力级门控模块，在早期抑制动态区域影响，提升时序一致性与相机位姿鲁棒性，实现无需微调的运动感知重建。**

- **链接: [https://arxiv.org/pdf/2512.03939v1](https://arxiv.org/pdf/2512.03939v1)**

> **作者:** Guole Shen; Tianchen Deng; Xingrui Qin; Nailin Wang; Jianyu Wang; Yanbo Wang; Yongtao Chen; Hesheng Wang; Jingchuan Wang
>
> **摘要:** Recent stateful recurrent neural networks have achieved remarkable progress on static 3D reconstruction but remain vulnerable to motion-induced artifacts, where non-rigid regions corrupt attention propagation between the spatial memory and image feature. By analyzing the internal behaviors of the state and image token updating mechanism, we find that aggregating self-attention maps across layers reveals a consistent pattern: dynamic regions are naturally down-weighted, exposing an implicit motion cue that the pretrained transformer already encodes but never explicitly uses. Motivated by this observation, we introduce MUT3R, a training-free framework that applies the attention-derived motion cue to suppress dynamic content in the early layers of the transformer during inference. Our attention-level gating module suppresses the influence of dynamic regions before their artifacts propagate through the feature hierarchy. Notably, we do not retrain or fine-tune the model; we let the pretrained transformer diagnose its own motion cues and correct itself. This early regulation stabilizes geometric reasoning in streaming scenarios and leads to improvements in temporal consistency and camera pose robustness across multiple dynamic benchmarks, offering a simple and training-free pathway toward motion-aware streaming reconstruction.
>
---
#### [new 046] BlurDM: A Blur Diffusion Model for Image Deblurring
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像去模糊任务，提出BlurDM模型，将模糊形成过程融入扩散模型。通过双扩散前向过程隐式建模运动模糊，实现噪声与模糊的联合扩散；在反向生成中同步完成去噪与去模糊。基于潜在空间的灵活设计提升了现有方法性能，在多个基准数据集上表现优异。**

- **链接: [https://arxiv.org/pdf/2512.03979v1](https://arxiv.org/pdf/2512.03979v1)**

> **作者:** Jin-Ting He; Fu-Jen Tsai; Yan-Tsung Peng; Min-Hung Chen; Chia-Wen Lin; Yen-Yu Lin
>
> **备注:** NeurIPS 2025
>
> **摘要:** Diffusion models show promise for dynamic scene deblurring; however, existing studies often fail to leverage the intrinsic nature of the blurring process within diffusion models, limiting their full potential. To address it, we present a Blur Diffusion Model (BlurDM), which seamlessly integrates the blur formation process into diffusion for image deblurring. Observing that motion blur stems from continuous exposure, BlurDM implicitly models the blur formation process through a dual-diffusion forward scheme, diffusing both noise and blur onto a sharp image. During the reverse generation process, we derive a dual denoising and deblurring formulation, enabling BlurDM to recover the sharp image by simultaneously denoising and deblurring, given pure Gaussian noise conditioned on the blurred image as input. Additionally, to efficiently integrate BlurDM into deblurring networks, we perform BlurDM in the latent space, forming a flexible prior generation network for deblurring. Extensive experiments demonstrate that BlurDM significantly and consistently enhances existing deblurring methods on four benchmark datasets. The source code is available at https://github.com/Jin-Ting-He/BlurDM.
>
---
#### [new 047] C3G: Learning Compact 3D Representations with 2K Gaussians
- **分类: cs.CV**

- **简介: 该论文针对无姿态稀疏视图下的3D场景重建与理解任务，解决现有方法因冗余高斯分布导致的内存过高和特征聚合不佳问题。提出C3G框架，通过可学习令牌引导生成紧凑的3D高斯，实现高效特征提升，显著提升重建质量与内存效率。**

- **链接: [https://arxiv.org/pdf/2512.04021v1](https://arxiv.org/pdf/2512.04021v1)**

> **作者:** Honggyu An; Jaewoo Jung; Mungyeom Kim; Sunghwan Hong; Chaehyun Kim; Kazumi Fukuda; Minkyeong Jeon; Jisang Han; Takuya Narihira; Hyuna Ko; Junsu Kim; Yuki Mitsufuji; Seungryong Kim
>
> **备注:** Project Page : https://cvlab-kaist.github.io/C3G/
>
> **摘要:** Reconstructing and understanding 3D scenes from unposed sparse views in a feed-forward manner remains as a challenging task in 3D computer vision. Recent approaches use per-pixel 3D Gaussian Splatting for reconstruction, followed by a 2D-to-3D feature lifting stage for scene understanding. However, they generate excessive redundant Gaussians, causing high memory overhead and sub-optimal multi-view feature aggregation, leading to degraded novel view synthesis and scene understanding performance. We propose C3G, a novel feed-forward framework that estimates compact 3D Gaussians only at essential spatial locations, minimizing redundancy while enabling effective feature lifting. We introduce learnable tokens that aggregate multi-view features through self-attention to guide Gaussian generation, ensuring each Gaussian integrates relevant visual features across views. We then exploit the learned attention patterns for Gaussian decoding to efficiently lift features. Extensive experiments on pose-free novel view synthesis, 3D open-vocabulary segmentation, and view-invariant feature aggregation demonstrate our approach's effectiveness. Results show that a compact yet geometrically meaningful representation is sufficient for high-quality scene reconstruction and understanding, achieving superior memory efficiency and feature fidelity compared to existing methods.
>
---
#### [new 048] Active Visual Perception: Opportunities and Challenges
- **分类: cs.CV**

- **简介: 该论文探讨主动视觉感知在机器人、自动驾驶等领域的应用，旨在解决复杂环境中信息获取不足的问题。通过动态调节感知行为，提升系统对环境的理解能力，重点分析实时处理、动态决策与多模态融合等挑战，提出研究方向与解决方案。**

- **链接: [https://arxiv.org/pdf/2512.03687v1](https://arxiv.org/pdf/2512.03687v1)**

> **作者:** Yian Li; Xiaoyu Guo; Hao Zhang; Shuiwang Li; Xiaowei Dai
>
> **摘要:** Active visual perception refers to the ability of a system to dynamically engage with its environment through sensing and action, allowing it to modify its behavior in response to specific goals or uncertainties. Unlike passive systems that rely solely on visual data, active visual perception systems can direct attention, move sensors, or interact with objects to acquire more informative data. This approach is particularly powerful in complex environments where static sensing methods may not provide sufficient information. Active visual perception plays a critical role in numerous applications, including robotics, autonomous vehicles, human-computer interaction, and surveillance systems. However, despite its significant promise, there are several challenges that need to be addressed, including real-time processing of complex visual data, decision-making in dynamic environments, and integrating multimodal sensory inputs. This paper explores both the opportunities and challenges inherent in active visual perception, providing a comprehensive overview of its potential, current research, and the obstacles that must be overcome for broader adoption.
>
---
#### [new 049] FloodDiffusion: Tailored Diffusion Forcing for Streaming Motion Generation
- **分类: cs.CV**

- **简介: 该论文提出FloodDiffusion，用于文本驱动的流式人体运动生成。针对现有方法在实时性与连续性上的不足，提出改进的扩散强迫框架，通过双向注意力、下三角时间调度和连续文本条件引入，实现高保真、无缝的运动生成，在HumanML3D上达到0.057 FID，性能领先。**

- **链接: [https://arxiv.org/pdf/2512.03520v1](https://arxiv.org/pdf/2512.03520v1)**

> **作者:** Yiyi Cai; Yuhan Wu; Kunhang Li; You Zhou; Bo Zheng; Haiyang Liu
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** We present FloodDiffusion, a new framework for text-driven, streaming human motion generation. Given time-varying text prompts, FloodDiffusion generates text-aligned, seamless motion sequences with real-time latency. Unlike existing methods that rely on chunk-by-chunk or auto-regressive model with diffusion head, we adopt a diffusion forcing framework to model this time-series generation task under time-varying control events. We find that a straightforward implementation of vanilla diffusion forcing (as proposed for video models) fails to model real motion distributions. We demonstrate that to guarantee modeling the output distribution, the vanilla diffusion forcing must be tailored to: (i) train with a bi-directional attention instead of casual attention; (ii) implement a lower triangular time scheduler instead of a random one; (iii) utilize a continues time-varying way to introduce text conditioning. With these improvements, we demonstrate in the first time that the diffusion forcing-based framework achieves state-of-the-art performance on the streaming motion generation task, reaching an FID of 0.057 on the HumanML3D benchmark. Models, code, and weights are available. https://shandaai.github.io/FloodDiffusion/
>
---
#### [new 050] Beyond the Ground Truth: Enhanced Supervision for Image Restoration
- **分类: cs.CV**

- **简介: 该论文针对真实世界图像修复中因数据集真值图像质量受限导致模型性能瓶颈的问题，提出通过自适应频率掩码融合原图与超分结果，生成感知增强的真值图像，以提供更优监督信号。进而训练轻量级输出优化网络，显著提升修复质量。**

- **链接: [https://arxiv.org/pdf/2512.03932v1](https://arxiv.org/pdf/2512.03932v1)**

> **作者:** Donghun Ryou; Inju Ha; Sanghyeok Chu; Bohyung Han
>
> **摘要:** Deep learning-based image restoration has achieved significant success. However, when addressing real-world degradations, model performance is limited by the quality of ground-truth images in datasets due to practical constraints in data acquisition. To address this limitation, we propose a novel framework that enhances existing ground truth images to provide higher-quality supervision for real-world restoration. Our framework generates perceptually enhanced ground truth images using super-resolution by incorporating adaptive frequency masks, which are learned by a conditional frequency mask generator. These masks guide the optimal fusion of frequency components from the original ground truth and its super-resolved variants, yielding enhanced ground truth images. This frequency-domain mixup preserves the semantic consistency of the original content while selectively enriching perceptual details, preventing hallucinated artifacts that could compromise fidelity. The enhanced ground truth images are used to train a lightweight output refinement network that can be seamlessly integrated with existing restoration models. Extensive experiments demonstrate that our approach consistently improves the quality of restored images. We further validate the effectiveness of both supervision enhancement and output refinement through user studies. Code is available at https://github.com/dhryougit/Beyond-the-Ground-Truth.
>
---
#### [new 051] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对领域泛化语义分割中视觉与文本语义错位问题，提出DPMFormer框架。通过域感知提示学习对齐多模态语义，结合域感知对比学习与纹理扰动增强域多样性，并引入域鲁棒一致性学习提升模型抗环境变化能力，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.03508v1](https://arxiv.org/pdf/2512.03508v1)**

> **作者:** Seogkyu Jeon; Kibeom Hong; Hyeran Byun
>
> **备注:** ICCV 2025 (poster)
>
> **摘要:** Recent domain generalized semantic segmentation (DGSS) studies have achieved notable improvements by distilling semantic knowledge from Vision-Language Models (VLMs). However, they overlook the semantic misalignment between visual and textual contexts, which arises due to the rigidity of a fixed context prompt learned on a single source domain. To this end, we present a novel domain generalization framework for semantic segmentation, namely Domain-aware Prompt-driven Masked Transformer (DPMFormer). Firstly, we introduce domain-aware prompt learning to facilitate semantic alignment between visual and textual cues. To capture various domain-specific properties with a single source dataset, we propose domain-aware contrastive learning along with the texture perturbation that diversifies the observable domains. Lastly, to establish a framework resilient against diverse environmental changes, we have proposed the domain-robust consistency learning which guides the model to minimize discrepancies of prediction from original and the augmented images. Through experiments and analyses, we demonstrate the superiority of the proposed framework, which establishes a new state-of-the-art on various DGSS benchmarks. The code is available at https://github.com/jone1222/DPMFormer.
>
---
#### [new 052] HieroGlyphTranslator: Automatic Recognition and Translation of Egyptian Hieroglyphs to English
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于跨模态翻译任务，旨在解决古埃及象形文字图像到英文的自动识别与翻译问题。研究提出三阶段方法：基于Contour和Detectron2的分割、符号映射至Gardiner码、使用CNN模型翻译，利用两个数据集训练，取得42.2的BLEU得分，显著提升翻译准确性。**

- **链接: [https://arxiv.org/pdf/2512.03817v1](https://arxiv.org/pdf/2512.03817v1)**

> **作者:** Ahmed Nasser; Marwan Mohamed; Alaa Sherif; Basmala Mahmoud; Shereen Yehia; Asmaa Saad; Mariam S. El-Rahmany; Ensaf H. Mohamed
>
> **摘要:** Egyptian hieroglyphs, the ancient Egyptian writing system, are composed entirely of drawings. Translating these glyphs into English poses various challenges, including the fact that a single glyph can have multiple meanings. Deep learning translation applications are evolving rapidly, producing remarkable results that significantly impact our lives. In this research, we propose a method for the automatic recognition and translation of ancient Egyptian hieroglyphs from images to English. This study utilized two datasets for classification and translation: the Morris Franken dataset and the EgyptianTranslation dataset. Our approach is divided into three stages: segmentation (using Contour and Detectron2), mapping symbols to Gardiner codes, and translation (using the CNN model). The model achieved a BLEU score of 42.2, a significant result compared to previous research.
>
---
#### [new 053] ShelfGaussian: Shelf-Supervised Open-Vocabulary Gaussian-based 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文提出ShelfGaussian，一种基于多模态高斯的开放词汇3D场景理解框架。针对现有方法在闭集语义建模或仅依赖2D自监督导致几何退化与场景受限的问题，提出多模态高斯变换器与货架式监督范式，利用现成视觉基础模型联合优化2D与3D特征，实现高效、高保真的零样本语义占据预测与真实场景泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.03370v1](https://arxiv.org/pdf/2512.03370v1)**

> **作者:** Lingjun Zhao; Yandong Luo; James Hay; Lu Gan
>
> **摘要:** We introduce ShelfGaussian, an open-vocabulary multi-modal Gaussian-based 3D scene understanding framework supervised by off-the-shelf vision foundation models (VFMs). Gaussian-based methods have demonstrated superior performance and computational efficiency across a wide range of scene understanding tasks. However, existing methods either model objects as closed-set semantic Gaussians supervised by annotated 3D labels, neglecting their rendering ability, or learn open-set Gaussian representations via purely 2D self-supervision, leading to degraded geometry and limited to camera-only settings. To fully exploit the potential of Gaussians, we propose a Multi-Modal Gaussian Transformer that enables Gaussians to query features from diverse sensor modalities, and a Shelf-Supervised Learning Paradigm that efficiently optimizes Gaussians with VFM features jointly at 2D image and 3D scene levels. We evaluate ShelfGaussian on various perception and planning tasks. Experiments on Occ3D-nuScenes demonstrate its state-of-the-art zero-shot semantic occupancy prediction performance. ShelfGaussian is further evaluated on an unmanned ground vehicle (UGV) to assess its in the-wild performance across diverse urban scenarios. Project website: https://lunarlab-gatech.github.io/ShelfGaussian/.
>
---
#### [new 054] GaussianBlender: Instant Stylization of 3D Gaussians with Disentangled Latent Spaces
- **分类: cs.CV**

- **简介: 该论文针对3D stylization任务，解决现有方法依赖耗时优化、多视角不一致的问题。提出GaussianBlender框架，通过学习解耦的潜在空间与扩散模型，实现文本驱动的即时、高保真、多视图一致的3D风格化，无需逐资产优化，支持大规模应用。**

- **链接: [https://arxiv.org/pdf/2512.03683v1](https://arxiv.org/pdf/2512.03683v1)**

> **作者:** Melis Ocal; Xiaoyan Xing; Yue Li; Ngo Anh Vien; Sezer Karaoglu; Theo Gevers
>
> **摘要:** 3D stylization is central to game development, virtual reality, and digital arts, where the demand for diverse assets calls for scalable methods that support fast, high-fidelity manipulation. Existing text-to-3D stylization methods typically distill from 2D image editors, requiring time-intensive per-asset optimization and exhibiting multi-view inconsistency due to the limitations of current text-to-image models, which makes them impractical for large-scale production. In this paper, we introduce GaussianBlender, a pioneering feed-forward framework for text-driven 3D stylization that performs edits instantly at inference. Our method learns structured, disentangled latent spaces with controlled information sharing for geometry and appearance from spatially-grouped 3D Gaussians. A latent diffusion model then applies text-conditioned edits on these learned representations. Comprehensive evaluations show that GaussianBlender not only delivers instant, high-fidelity, geometry-preserving, multi-view consistent stylization, but also surpasses methods that require per-instance test-time optimization - unlocking practical, democratized 3D stylization at scale.
>
---
#### [new 055] Hierarchical Attention for Sparse Volumetric Anomaly Detection in Subclinical Keratoconus
- **分类: cs.CV**

- **简介: 该论文针对3D眼科OCT图像中亚临床圆锥角膜的稀疏异常检测任务，解决传统模型因局部性或全局注意力过强导致早期微弱信号丢失的问题。通过对比16种架构，提出层级注意力模型，实现更优的空间尺度匹配，显著提升检测灵敏度与特异性，为早期病灶识别提供有效方案。**

- **链接: [https://arxiv.org/pdf/2512.03346v1](https://arxiv.org/pdf/2512.03346v1)**

> **作者:** Lynn Kandakji; William Woof; Nikolas Pontikos
>
> **备注:** 16 pages, 7 figures, 6 tables
>
> **摘要:** The detection of weak, spatially distributed anomalies in volumetric medical imaging remains a major challenge. The subtle, non-adjacent nature of early disease signals is often lost due to suboptimal architectural inductive biases: 2D/3D CNNs impose strong locality, while ViTs diffuse unconstrained global attention. This conflict leaves the optimal inductive structure for robust, sparse volumetric pattern recognition unresolved. This study presents a controlled comparison of sixteen modern deep learning architectures spanning 2D/3D convolutional, hybrid, and volumetric transformer families for subclinical keratoconus (SKC) detection from 3D anterior segment OCT volumes. We demonstrate that hierarchical attention models offer a superior and more parameter-efficient inductive bias, surpassing the performance of both 2D and 3D CNNs and ViTs. Our results show 21-23% higher sensitivity and specificity in the sparse anomaly (subclinical) regime. Mechanistic analyses reveal that this advantage stems from precise spatial scale alignment: hierarchical windowing produces effective receptive fields matched to the intermediate, multi-slice extent of subclinical abnormalities. This avoids excessive CNN locality and diffuse global attention. Attention-distance measurements confirm a key insight into architectural adaptation: the required spatial integration length shifts significantly based on the signal strength, with subclinical cases necessitating longer integration compared to both healthy and manifest disease states. Representational similarity and auxiliary age/sex prediction tasks further support the generalizability of these inductive principles. The findings provide design guidance for future volumetric anomaly detection systems, establishing hierarchical attention as a principled and effective approach for early pathological change analysis in 3D medical imaging.
>
---
#### [new 056] Dual-level Modality Debiasing Learning for Unsupervised Visible-Infrared Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对无监督可见光-红外行人重识别任务，解决两阶段学习中模态偏差传播问题。提出双层次去偏学习框架DMDL，通过因果干预模块抑制模态特异性伪相关，结合协同无偏训练策略，实现跨模态特征不变性与模型泛化。**

- **链接: [https://arxiv.org/pdf/2512.03745v1](https://arxiv.org/pdf/2512.03745v1)**

> **作者:** Jiaze Li; Yan Lu; Bin Liu; Guojun Yin; Mang Ye
>
> **摘要:** Two-stage learning pipeline has achieved promising results in unsupervised visible-infrared person re-identification (USL-VI-ReID). It first performs single-modality learning and then operates cross-modality learning to tackle the modality discrepancy. Although promising, this pipeline inevitably introduces modality bias: modality-specific cues learned in the single-modality training naturally propagate into the following cross-modality learning, impairing identity discrimination and generalization. To address this issue, we propose a Dual-level Modality Debiasing Learning (DMDL) framework that implements debiasing at both the model and optimization levels. At the model level, we propose a Causality-inspired Adjustment Intervention (CAI) module that replaces likelihood-based modeling with causal modeling, preventing modality-induced spurious patterns from being introduced, leading to a low-biased model. At the optimization level, a Collaborative Bias-free Training (CBT) strategy is introduced to interrupt the propagation of modality bias across data, labels, and features by integrating modality-specific augmentation, label refinement, and feature alignment. Extensive experiments on benchmark datasets demonstrate that DMDL could enable modality-invariant feature learning and a more generalized model.
>
---
#### [new 057] Ultra-lightweight Neural Video Representation Compression
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对轻量级神经视频压缩任务，解决现有隐式神经表示（INR）编码慢、复杂度高的问题。提出NVRC-Lite，引入多尺度特征网格提升性能，并设计基于八叉树的上下文模型加速熵编码，显著提升压缩效率与速度。**

- **链接: [https://arxiv.org/pdf/2512.04019v1](https://arxiv.org/pdf/2512.04019v1)**

> **作者:** Ho Man Kwan; Tianhao Peng; Ge Gao; Fan Zhang; Mike Nilsson; Andrew Gower; David Bull
>
> **摘要:** Recent works have demonstrated the viability of utilizing over-fitted implicit neural representations (INRs) as alternatives to autoencoder-based models for neural video compression. Among these INR-based video codecs, Neural Video Representation Compression (NVRC) was the first to adopt a fully end-to-end compression framework that compresses INRs, achieving state-of-the-art performance. Moreover, some recently proposed lightweight INRs have shown comparable performance to their baseline codecs with computational complexity lower than 10kMACs/pixel. In this work, we extend NVRC toward lightweight representations, and propose NVRC-Lite, which incorporates two key changes. Firstly, we integrated multi-scale feature grids into our lightweight neural representation, and the use of higher resolution grids significantly improves the performance of INRs at low complexity. Secondly, we address the issue that existing INRs typically leverage autoregressive models for entropy coding: these are effective but impractical due to their slow coding speed. In this work, we propose an octree-based context model for entropy coding high-dimensional feature grids, which accelerates the entropy coding module of the model. Our experimental results demonstrate that NVRC-Lite outperforms C3, one of the best lightweight INR-based video codecs, with up to 21.03% and 23.06% BD-rate savings when measured in PSNR and MS-SSIM, respectively, while achieving 8.4x encoding and 2.5x decoding speedup. The implementation of NVRC-Lite will be made available.
>
---
#### [new 058] FireSentry: A Multi-Modal Spatio-temporal Benchmark Dataset for Fine-Grained Wildfire Spread Forecasting
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对细粒度野火蔓延预测任务，解决现有研究依赖低分辨率数据、难以捕捉局部动态的问题。提出FireSentry多模态时空数据集（亚米级空间、亚秒级时间），并构建基准测试。设计FiReDiff模型，融合红外视频生成与火区掩码精分割，显著提升预测精度与视觉质量。**

- **链接: [https://arxiv.org/pdf/2512.03369v1](https://arxiv.org/pdf/2512.03369v1)**

> **作者:** Nan Zhou; Huandong Wang; Jiahao Li; Han Li; Yali Song; Qiuhua Wang; Yong Li; Xinlei Chen
>
> **摘要:** Fine-grained wildfire spread prediction is crucial for enhancing emergency response efficacy and decision-making precision. However, existing research predominantly focuses on coarse spatiotemporal scales and relies on low-resolution satellite data, capturing only macroscopic fire states while fundamentally constraining high-precision localized fire dynamics modeling capabilities. To bridge this gap, we present FireSentry, a provincial-scale multi-modal wildfire dataset characterized by sub-meter spatial and sub-second temporal resolution. Collected using synchronized UAV platforms, FireSentry provides visible and infrared video streams, in-situ environmental measurements, and manually validated fire masks. Building on FireSentry, we establish a comprehensive benchmark encompassing physics-based, data-driven, and generative models, revealing the limitations of existing mask-only approaches. Our analysis proposes FiReDiff, a novel dual-modality paradigm that first predicts future video sequences in the infrared modality, and then precisely segments fire masks in the mask modality based on the generated dynamics. FiReDiff achieves state-of-the-art performance, with video quality gains of 39.2% in PSNR, 36.1% in SSIM, 50.0% in LPIPS, 29.4% in FVD, and mask accuracy gains of 3.3% in AUPRC, 59.1% in F1 score, 42.9% in IoU, and 62.5% in MSE when applied to generative models. The FireSentry benchmark dataset and FiReDiff paradigm collectively advance fine-grained wildfire forecasting and dynamic disaster simulation. The processed benchmark dataset is publicly available at: https://github.com/Munan222/FireSentry-Benchmark-Dataset.
>
---
#### [new 059] OpenTrack3D: Towards Accurate and Generalizable Open-Vocabulary 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文针对开放词汇3D实例分割（OV-3DIS）在无结构、无网格场景下的泛化难题，提出OpenTrack3D框架。解决现有方法依赖特定数据集提案和文本推理弱的问题，通过在线视觉-空间跟踪生成跨视图一致的3D提案，并结合多模态大语言模型增强语义理解，实现无需网格的高精度、强泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.03532v1](https://arxiv.org/pdf/2512.03532v1)**

> **作者:** Zhishan Zhou; Siyuan Wei; Zengran Wang; Chunjie Wang; Xiaosheng Yan; Xiao Liu
>
> **摘要:** Generalizing open-vocabulary 3D instance segmentation (OV-3DIS) to diverse, unstructured, and mesh-free environments is crucial for robotics and AR/VR, yet remains a significant challenge. We attribute this to two key limitations of existing methods: (1) proposal generation relies on dataset-specific proposal networks or mesh-based superpoints, rendering them inapplicable in mesh-free scenarios and limiting generalization to novel scenes; and (2) the weak textual reasoning of CLIP-based classifiers, which struggle to recognize compositional and functional user queries. To address these issues, we introduce OpenTrack3D, a generalizable and accurate framework. Unlike methods that rely on pre-generated proposals, OpenTrack3D employs a novel visual-spatial tracker to construct cross-view consistent object proposals online. Given an RGB-D stream, our pipeline first leverages a 2D open-vocabulary segmenter to generate masks, which are lifted to 3D point clouds using depth. Mask-guided instance features are then extracted using DINO feature maps, and our tracker fuses visual and spatial cues to maintain instance consistency. The core pipeline is entirely mesh-free, yet we also provide an optional superpoints refinement module to further enhance performance when scene mesh is available. Finally, we replace CLIP with a multi-modal large language model (MLLM), significantly enhancing compositional reasoning for complex user queries. Extensive experiments on diverse benchmarks, including ScanNet200, Replica, ScanNet++, and SceneFun3D, demonstrate state-of-the-art performance and strong generalization capabilities.
>
---
#### [new 060] Emergent Outlier View Rejection in Visual Geometry Grounded Transformers
- **分类: cs.CV**

- **简介: 该论文研究视觉几何增强的Transformer在3D重建中的隐式异常视图剔除问题。针对无监督前馈模型缺乏显式去噪机制导致的性能下降，发现模型特定层可自发抑制噪声图像。通过分析其内部表征，提出无需微调即可实现高效异常视图剔除的方法，在多种数据集上验证了其泛化性与有效性。**

- **链接: [https://arxiv.org/pdf/2512.04012v1](https://arxiv.org/pdf/2512.04012v1)**

> **作者:** Jisang Han; Sunghwan Hong; Jaewoo Jung; Wooseok Jang; Honggyu An; Qianqian Wang; Seungryong Kim; Chen Feng
>
> **备注:** Project page: https://cvlab-kaist.github.io/RobustVGGT/
>
> **摘要:** Reliable 3D reconstruction from in-the-wild image collections is often hindered by "noisy" images-irrelevant inputs with little or no view overlap with others. While traditional Structure-from-Motion pipelines handle such cases through geometric verification and outlier rejection, feed-forward 3D reconstruction models lack these explicit mechanisms, leading to degraded performance under in-the-wild conditions. In this paper, we discover that the existing feed-forward reconstruction model, e.g., VGGT, despite lacking explicit outlier-rejection mechanisms or noise-aware training, can inherently distinguish distractor images. Through an in-depth analysis under varying proportions of synthetic distractors, we identify a specific layer that naturally exhibits outlier-suppressing behavior. Further probing reveals that this layer encodes discriminative internal representations that enable an effective noise-filtering capability, which we simply leverage to perform outlier-view rejection in feed-forward 3D reconstruction without any additional fine-tuning or supervision. Extensive experiments on both controlled and in-the-wild datasets demonstrate that this implicit filtering mechanism is consistent and generalizes well across diverse scenarios.
>
---
#### [new 061] FeatureLens: A Highly Generalizable and Interpretable Framework for Detecting Adversarial Examples Based on Image Features
- **分类: cs.CV**

- **简介: 该论文针对深度神经网络在图像分类中易受对抗攻击的问题，提出轻量级可解释的检测框架FeatureLens。通过51维图像特征与浅层分类器结合，实现高精度（97.8%~99.75%）和强泛化能力（86.17%~99.6%），兼顾效率与透明性，有效提升对抗样本检测的可靠性。**

- **链接: [https://arxiv.org/pdf/2512.03625v1](https://arxiv.org/pdf/2512.03625v1)**

> **作者:** Zhigang Yang; Yuan Liu; Jiawei Zhang; Puning Zhang; Xinqiang Ma
>
> **摘要:** Although the remarkable performance of deep neural networks (DNNs) in image classification, their vulnerability to adversarial attacks remains a critical challenge. Most existing detection methods rely on complex and poorly interpretable architectures, which compromise interpretability and generalization. To address this, we propose FeatureLens, a lightweight framework that acts as a lens to scrutinize anomalies in image features. Comprising an Image Feature Extractor (IFE) and shallow classifiers (e.g., SVM, MLP, or XGBoost) with model sizes ranging from 1,000 to 30,000 parameters, FeatureLens achieves high detection accuracy ranging from 97.8% to 99.75% in closed-set evaluation and 86.17% to 99.6% in generalization evaluation across FGSM, PGD, CW, and DAmageNet attacks, using only 51 dimensional features. By combining strong detection performance with excellent generalization, interpretability, and computational efficiency, FeatureLens offers a practical pathway toward transparent and effective adversarial defense.
>
---
#### [new 062] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到视觉生成中的意图对齐问题，提出PRIS框架，通过在推理时动态重设计提示来提升生成质量。其核心是基于细粒度事实校正验证器，识别生成失败模式并迭代优化提示，实现提示与视觉的协同缩放，显著提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2512.03534v1](https://arxiv.org/pdf/2512.03534v1)**

> **作者:** Subin Kim; Sangwoo Mo; Mamshad Nayeem Rizve; Yiran Xu; Difan Liu; Jinwoo Shin; Tobias Hinz
>
> **备注:** Visualizations are available at the website: https://subin-kim-cv.github.io/PRIS
>
> **摘要:** Achieving precise alignment between user intent and generated visuals remains a central challenge in text-to-visual generation, as a single attempt often fails to produce the desired output. To handle this, prior approaches mainly scale the visual generation process (e.g., increasing sampling steps or seeds), but this quickly leads to a quality plateau. This limitation arises because the prompt, crucial for guiding generation, is kept fixed. To address this, we propose Prompt Redesign for Inference-time Scaling, coined PRIS, a framework that adaptively revises the prompt during inference in response to the scaled visual generations. The core idea of PRIS is to review the generated visuals, identify recurring failure patterns across visuals, and redesign the prompt accordingly before regenerating the visuals with the revised prompt. To provide precise alignment feedback for prompt revision, we introduce a new verifier, element-level factual correction, which evaluates the alignment between prompt attributes and generated visuals at a fine-grained level, achieving more accurate and interpretable assessments than holistic measures. Extensive experiments on both text-to-image and text-to-video benchmarks demonstrate the effectiveness of our approach, including a 15% gain on VBench 2.0. These results highlight that jointly scaling prompts and visuals is key to fully leveraging scaling laws at inference-time. Visualizations are available at the website: https://subin-kim-cv.github.io/PRIS.
>
---
#### [new 063] Training for Identity, Inference for Controllability: A Unified Approach to Tuning-Free Face Personalization
- **分类: cs.CV**

- **简介: 该论文针对面部个性化任务，解决现有方法在身份保真度与文本可控性难以兼顾的问题。提出UniID统一框架，融合文本嵌入与适配器方法，通过分阶段训练与归一化重缩放机制，实现高保真身份保留与灵活文本控制。**

- **链接: [https://arxiv.org/pdf/2512.03964v1](https://arxiv.org/pdf/2512.03964v1)**

> **作者:** Lianyu Pang; Ji Zhou; Qiping Wang; Baoquan Zhao; Zhenguo Yang; Qing Li; Xudong Mao
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** Tuning-free face personalization methods have developed along two distinct paradigms: text embedding approaches that map facial features into the text embedding space, and adapter-based methods that inject features through auxiliary cross-attention layers. While both paradigms have shown promise, existing methods struggle to simultaneously achieve high identity fidelity and flexible text controllability. We introduce UniID, a unified tuning-free framework that synergistically integrates both paradigms. Our key insight is that when merging these approaches, they should mutually reinforce only identity-relevant information while preserving the original diffusion prior for non-identity attributes. We realize this through a principled training-inference strategy: during training, we employ an identity-focused learning scheme that guides both branches to capture identity features exclusively; at inference, we introduce a normalized rescaling mechanism that recovers the text controllability of the base diffusion model while enabling complementary identity signals to enhance each other. This principled design enables UniID to achieve high-fidelity face personalization with flexible text controllability. Extensive experiments against six state-of-the-art methods demonstrate that UniID achieves superior performance in both identity preservation and text controllability. Code will be available at https://github.com/lyuPang/UniID
>
---
#### [new 064] Fast & Efficient Normalizing Flows and Applications of Image Generative Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文聚焦生成模型效率与应用，提出六项创新提升归一化流计算效率，构建高效图像生成框架；应用于农业质检、地质映射、自动驾驶隐私保护及艺术修复，解决数据不平衡、隐私泄露与多类型退化等问题。**

- **链接: [https://arxiv.org/pdf/2512.04039v1](https://arxiv.org/pdf/2512.04039v1)**

> **作者:** Sandeep Nagar
>
> **备注:** PhD Thesis
>
> **摘要:** This thesis presents novel contributions in two primary areas: advancing the efficiency of generative models, particularly normalizing flows, and applying generative models to solve real-world computer vision challenges. The first part introduce significant improvements to normalizing flow architectures through six key innovations: 1) Development of invertible 3x3 Convolution layers with mathematically proven necessary and sufficient conditions for invertibility, (2) introduction of a more efficient Quad-coupling layer, 3) Design of a fast and efficient parallel inversion algorithm for kxk convolutional layers, 4) Fast & efficient backpropagation algorithm for inverse of convolution, 5) Using inverse of convolution, in Inverse-Flow, for the forward pass and training it using proposed backpropagation algorithm, and 6) Affine-StableSR, a compact and efficient super-resolution model that leverages pre-trained weights and Normalizing Flow layers to reduce parameter count while maintaining performance. The second part: 1) An automated quality assessment system for agricultural produce using Conditional GANs to address class imbalance, data scarcity and annotation challenges, achieving good accuracy in seed purity testing; 2) An unsupervised geological mapping framework utilizing stacked autoencoders for dimensionality reduction, showing improved feature extraction compared to conventional methods; 3) We proposed a privacy preserving method for autonomous driving datasets using on face detection and image inpainting; 4) Utilizing Stable Diffusion based image inpainting for replacing the detected face and license plate to advancing privacy-preserving techniques and ethical considerations in the field.; and 5) An adapted diffusion model for art restoration that effectively handles multiple types of degradation through unified fine-tuning.
>
---
#### [new 065] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation
- **分类: cs.CV**

- **简介: 该论文提出SeeU，解决视觉生成中因仅依赖2D观测导致的性能局限。通过2D→4D→2D框架，从稀疏单目帧重建连续4D动态，结合低秩表示与物理约束，实现时空一致的未见内容生成，适用于时序、空间及视频编辑任务。**

- **链接: [https://arxiv.org/pdf/2512.03350v1](https://arxiv.org/pdf/2512.03350v1)**

> **作者:** Yu Yuan; Tharindu Wickremasinghe; Zeeshan Nadir; Xijun Wang; Yiheng Chi; Stanley H. Chan
>
> **备注:** Project Page: https://yuyuanspace.com/SeeU/
>
> **摘要:** Images and videos are discrete 2D projections of the 4D world (3D space + time). Most visual understanding, prediction, and generation operate directly on 2D observations, leading to suboptimal performance. We propose SeeU, a novel approach that learns the continuous 4D dynamics and generate the unseen visual contents. The principle behind SeeU is a new 2D$\to$4D$\to$2D learning framework. SeeU first reconstructs the 4D world from sparse and monocular 2D frames (2D$\to$4D). It then learns the continuous 4D dynamics on a low-rank representation and physical constraints (discrete 4D$\to$continuous 4D). Finally, SeeU rolls the world forward in time, re-projects it back to 2D at sampled times and viewpoints, and generates unseen regions based on spatial-temporal context awareness (4D$\to$2D). By modeling dynamics in 4D, SeeU achieves continuous and physically-consistent novel visual generation, demonstrating strong potentials in multiple tasks including unseen temporal generation, unseen spatial generation, and video editing.
>
---
#### [new 066] TempR1: Improving Temporal Understanding of MLLMs via Temporal-Aware Multi-Task Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型（MLLMs）在长视频分析中时间理解能力不足的问题，提出TempR1框架。通过构建多任务语料库并设计分类型定位奖励，基于组相对策略优化实现跨任务协同训练，显著提升模型在时间定位、动作检测等任务上的性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.03963v1](https://arxiv.org/pdf/2512.03963v1)**

> **作者:** Tao Wu; Li Yang; Gen Zhan; Yiting Liao; Junlin Li; Deliang Fu; Li Zhang; Limin Wang
>
> **摘要:** Enhancing the temporal understanding of Multimodal Large Language Models (MLLMs) is essential for advancing long-form video analysis, enabling tasks such as temporal localization, action detection, and time-sensitive question answering. While reinforcement learning (RL) has recently been explored for improving temporal reasoning, existing approaches are often confined to limited task types and data, restricting their generalization across diverse temporal understanding scenarios. To address this challenge, we present TempR1, a temporal-aware multi-task reinforcement learning framework that systematically strengthens MLLMs' temporal comprehension. We curate a multi-task corpus that exposes the model to diverse temporal structures and semantics, and build upon the Group Relative Policy Optimization (GRPO) algorithm to achieve stable and effective cross-task optimization. Specifically, we categorize temporal tasks into three correspondence types between predicted intervals and ground-truth instances, and design tailored localization rewards for each, enabling TempR1 to capture fine-grained temporal dependencies and adapt to different temporal patterns. Extensive experiments demonstrate that TempR1 attains state-of-the-art performance across multiple benchmarks. Moreover, its joint optimization over complementary tasks yields a strong synergistic effect, enhancing both generalization and single-task performance, establishing a scalable and principled paradigm for temporal reasoning in MLLMs.
>
---
#### [new 067] PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对视频理解与生成任务中注意力机制计算复杂度高的问题，提出金字塔稀疏注意力（PSA）模块。通过多层级池化键值表示和动态分配策略，实现细粒度稀疏，减少信息损失，提升效率与性能，支持高效硬件执行。**

- **链接: [https://arxiv.org/pdf/2512.04025v1](https://arxiv.org/pdf/2512.04025v1)**

> **作者:** Xiaolong Li; Youping Gu; Xi Lin; Weijie Wang; Bohan Zhuang
>
> **备注:** Tech report
>
> **摘要:** Attention mechanisms are the core of foundation models, but their quadratic complexity remains a critical bottleneck for scaling. This challenge has driven the development of efficient attention mechanisms, with sparsity emerging as the dominant paradigm. Current methods typically retain or discard entire key-value blocks with binary masks, resulting in substantial information loss under high sparsity. To mitigate this gap, we present Pyramid Sparse Attention (PSA), a versatile module applicable to both video understanding and generation tasks. Instead of binary masking, PSA introduces multi-level pooled KV representations, enabling finer mask granularity. Specifically, each query block dynamically allocates lower pooling levels to critical KV blocks and higher levels to less important ones, creating an informative interpolation between full retention and complete pruning. This design, analogous to fixed-point quantization and classical feature pyramid networks in computer vision, effectively mitigates information loss while preserving computational efficiency under a low compute budget. It works with a native, hardware-friendly kernel that leverages decoupled block-tile design to ensure efficient execution. Across video understanding and generation benchmarks, PSA preserves contextual information and visual fidelity, consistently outperforming or achieving comparable performance over existing sparse attention baselines with superior efficiency-quality trade-offs. Our code and model weights are publicly available at: http://ziplab.co/PSA
>
---
#### [new 068] Cross-Stain Contrastive Learning for Paired Immunohistochemistry and Histopathology Slide Representation Learning
- **分类: cs.CV**

- **简介: 该论文针对计算病理中多染色全切片图像（WSI）表示学习问题，解决因染色间配准偏差导致的特征不一致难题。提出跨染色对比学习（CSCL）框架，通过分阶段预训练与轻量适配器，实现H&E与IHC图像在像素级和滑片级的对齐融合，提升通用、可迁移的滑片表示性能。**

- **链接: [https://arxiv.org/pdf/2512.03577v1](https://arxiv.org/pdf/2512.03577v1)**

> **作者:** Yizhi Zhang; Lei Fan; Zhulin Tao; Donglin Di; Yang Song; Sidong Liu; Cong Cong
>
> **备注:** 6 pages, 2 figures. Camera-ready version accepted for IEEE BIBM 2025
>
> **摘要:** Universal, transferable whole-slide image (WSI) representations are central to computational pathology. Incorporating multiple markers (e.g., immunohistochemistry, IHC) alongside H&E enriches H&E-based features with diverse, biologically meaningful information. However, progress is limited by the scarcity of well-aligned multi-stain datasets. Inter-stain misalignment shifts corresponding tissue across slides, hindering consistent patch-level features and degrading slide-level embeddings. To address this, we curated a slide-level aligned, five-stain dataset (H&E, HER2, KI67, ER, PGR) to enable paired H&E-IHC learning and robust cross-stain representation. Leveraging this dataset, we propose Cross-Stain Contrastive Learning (CSCL), a two-stage pretraining framework with a lightweight adapter trained using patch-wise contrastive alignment to improve the compatibility of H&E features with corresponding IHC-derived contextual cues, and slide-level representation learning with Multiple Instance Learning (MIL), which uses a cross-stain attention fusion module to integrate stain-specific patch features and a cross-stain global alignment module to enforce consistency among slide-level embeddings across different stains. Experiments on cancer subtype classification, IHC biomarker status classification, and survival prediction show consistent gains, yielding high-quality, transferable H&E slide-level representations. The code and data are available at https://github.com/lily-zyz/CSCL.
>
---
#### [new 069] A Hybrid Deep Learning Framework with Explainable AI for Lung Cancer Classification with DenseNet169 and SVM
- **分类: cs.CV**

- **简介: 该论文针对肺癌早期诊断中人工阅片效率低、易出错的问题，提出一种结合DenseNet169与SVM的混合深度学习框架。通过注意力机制、焦点损失和多尺度特征融合提升分类精度，并引入Grad-CAM与SHAP增强模型可解释性，实现98%准确率，推动肺癌智能诊断的准确性与透明度。**

- **链接: [https://arxiv.org/pdf/2512.03359v1](https://arxiv.org/pdf/2512.03359v1)**

> **作者:** Md Rashidul Islam; Bakary Gibba; Altagi Abdallah Bakheit Abdelgadir
>
> **摘要:** Lung cancer is a very deadly disease worldwide, and its early diagnosis is crucial for increasing patient survival rates. Computed tomography (CT) scans are widely used for lung cancer diagnosis as they can give detailed lung structures. However, manual interpretation is time-consuming and prone to human error. To surmount this challenge, the study proposes a deep learning-based automatic lung cancer classification system to enhance detection accuracy and interpretability. The IQOTHNCCD lung cancer dataset is utilized, which is a public CT scan dataset consisting of cases categorized into Normal, Benign, and Malignant and used DenseNet169, which includes Squeezeand-Excitation blocks for attention-based feature extraction, Focal Loss for handling class imbalance, and a Feature Pyramid Network (FPN) for multi-scale feature fusion. In addition, an SVM model was developed using MobileNetV2 for feature extraction, improving its classification performance. For model interpretability enhancement, the study integrated Grad-CAM for the visualization of decision-making regions in CT scans and SHAP (Shapley Additive Explanations) for explanation of feature contributions within the SVM model. Intensive evaluation was performed, and it was found that both DenseNet169 and SVM models achieved 98% accuracy, suggesting their robustness for real-world medical practice. These results open up the potential for deep learning to improve the diagnosis of lung cancer by a higher level of accuracy, transparency, and robustness.
>
---
#### [new 070] Thinking with Programming Vision: Towards a Unified View for Thinking with Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型在图像推理中工具使用受限、鲁棒性差的问题，提出CodeVision框架，通过生成代码作为通用工具接口，实现灵活、可扩展的图像操作。采用两阶段训练策略，结合高质量数据与密集过程奖励，显著提升模型在复杂任务中的表现与错误恢复能力。**

- **链接: [https://arxiv.org/pdf/2512.03746v1](https://arxiv.org/pdf/2512.03746v1)**

> **作者:** Zirun Guo; Minjie Hong; Feng Zhang; Kai Jia; Tao Jin
>
> **摘要:** Multimodal large language models (MLLMs) that think with images can interactively use tools to reason about visual inputs, but current approaches often rely on a narrow set of tools with limited real-world necessity and scalability. In this work, we first reveal a critical and previously overlooked weakness: even state-of-the-art MLLMs are surprisingly brittle, showing significant performance degradation on images with simple orientation changes or natural corruptions, underscoring the need for more robust tool-based reasoning. To address this, we propose CodeVision, a flexible and scalable code-as-tool framework where the model generates code as a universal interface to invoke any image operation, moving beyond fixed tool registries. We train our model using a two-stage methodology, beginning with Supervised Fine-Tuning (SFT) on a high-quality dataset curated for complex, multi-turn tool composition and error recovery, followed by Reinforcement Learning (RL) with a novel and dense process reward function to encourage strategic and efficient tool use. To facilitate this research, we construct new SFT and RL datasets and introduce a challenging new benchmark suite designed to rigorously evaluate robustness to orientation changes and multi-tool reasoning. Experiments on Qwen2.5-VL and Qwen3-VL series show that our approach significantly improves model performance and fosters emergent capabilities such as flexible tool composition, efficient chained execution, and robust error recovery from runtime feedback. Code is available at https://github.com/ByteDance-BandAI/CodeVision.
>
---
#### [new 071] Traffic Image Restoration under Adverse Weather via Frequency-Aware Mamba
- **分类: cs.CV**

- **简介: 该论文针对恶劣天气下交通图像恢复任务，解决现有方法忽视频域先验的问题。提出FAMamba框架，通过双分支特征提取与波形高频残差学习，结合自适应频域扫描机制，实现频域引导的序列建模，有效提升图像细节恢复质量。**

- **链接: [https://arxiv.org/pdf/2512.03852v1](https://arxiv.org/pdf/2512.03852v1)**

> **作者:** Liwen Pan; Longguang Wang; Guangwei Gao; Jun Wang; Jun Shi; Juncheng Li
>
> **备注:** 12pages, 13 figures, 5tables
>
> **摘要:** Traffic image restoration under adverse weather conditions remains a critical challenge for intelligent transportation systems. Existing methods primarily focus on spatial-domain modeling but neglect frequency-domain priors. Although the emerging Mamba architecture excels at long-range dependency modeling through patch-wise correlation analysis, its potential for frequency-domain feature extraction remains unexplored. To address this, we propose Frequency-Aware Mamba (FAMamba), a novel framework that integrates frequency guidance with sequence modeling for efficient image restoration. Our architecture consists of two key components: (1) a Dual-Branch Feature Extraction Block (DFEB) that enhances local-global interaction via bidirectional 2D frequency-adaptive scanning, dynamically adjusting traversal paths based on sub-band texture distributions; and (2) a Prior-Guided Block (PGB) that refines texture details through wavelet-based high-frequency residual learning, enabling high-quality image reconstruction with precise details. Meanwhile, we design a novel Adaptive Frequency Scanning Mechanism (AFSM) for the Mamba architecture, which enables the Mamba to achieve frequency-domain scanning across distinct subgraphs, thereby fully leveraging the texture distribution characteristics inherent in subgraph structures. Extensive experiments demonstrate the efficiency and effectiveness of FAMamba.
>
---
#### [new 072] Zero-Shot Video Translation and Editing with Frame Spatial-Temporal Correspondence
- **分类: cs.CV**

- **简介: 该论文针对零样本视频生成任务，解决视频帧间时空不一致问题。提出FRESCO框架，通过融合帧内与帧间对应关系，构建更强的时空约束，显式优化特征以提升视频一致性与视觉连贯性，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.03905v1](https://arxiv.org/pdf/2512.03905v1)**

> **作者:** Shuai Yang; Junxin Lin; Yifan Zhou; Ziwei Liu; Chen Change Loy
>
> **备注:** Code: https://github.com/Sunnycookies/FRESCO-v2, Project: https://williamyang1991.github.io/projects/FRESCOv2/
>
> **摘要:** The remarkable success in text-to-image diffusion models has motivated extensive investigation of their potential for video applications. Zero-shot techniques aim to adapt image diffusion models for videos without requiring further model training. Recent methods largely emphasize integrating inter-frame correspondence into attention mechanisms. However, the soft constraint applied to identify the valid features to attend is insufficient, which could lead to temporal inconsistency. In this paper, we present FRESCO, which integrates intra-frame correspondence with inter-frame correspondence to formulate a more robust spatial-temporal constraint. This enhancement ensures a consistent transformation of semantically similar content between frames. Our method goes beyond attention guidance to explicitly optimize features, achieving high spatial-temporal consistency with the input video, significantly enhancing the visual coherence of manipulated videos. We verify FRESCO adaptations on two zero-shot tasks of video-to-video translation and text-guided video editing. Comprehensive experiments demonstrate the effectiveness of our framework in generating high-quality, coherent videos, highlighting a significant advance over current zero-shot methods.
>
---
#### [new 073] Beyond Boundary Frames: Audio-Visual Semantic Guidance for Context-Aware Video Interpolation
- **分类: cs.CV**

- **简介: 该论文针对视频帧插值任务，解决复杂运动下生成不清晰、不一致的问题。提出BBF框架，通过多模态条件输入与解耦融合机制，实现音视频语义引导的上下文感知插值，结合渐进式训练提升生成质量，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.03590v1](https://arxiv.org/pdf/2512.03590v1)**

> **作者:** Yuchen Deng; Xiuyang Wu; Hai-Tao Zheng; Jie Wang; Feidiao Yang; Yuxing Han
>
> **摘要:** Handling fast, complex, and highly non-linear motion patterns has long posed challenges for video frame interpolation. Although recent diffusion-based approaches improve upon traditional optical-flow-based methods, they still struggle to cover diverse application scenarios and often fail to produce sharp, temporally consistent frames in fine-grained motion tasks such as audio-visual synchronized interpolation. To address these limitations, we introduce BBF (Beyond Boundary Frames), a context-aware video frame interpolation framework, which could be guided by audio/visual semantics. First, we enhance the input design of the interpolation model so that it can flexibly handle multiple conditional modalities, including text, audio, images, and video. Second, we propose a decoupled multimodal fusion mechanism that sequentially injects different conditional signals into a DiT backbone. Finally, to maintain the generation abilities of the foundation model, we adopt a progressive multi-stage training paradigm, where the start-end frame difference embedding is used to dynamically adjust both the data sampling and the loss weighting. Extensive experimental results demonstrate that BBF outperforms specialized state-of-the-art methods on both generic interpolation and audio-visual synchronized interpolation tasks, establishing a unified framework for video frame interpolation under coordinated multi-channel conditioning.
>
---
#### [new 074] Text-Printed Image: Bridging the Image-Text Modality Gap for Text-centric Training of Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究文本主导的视觉语言模型训练，旨在解决真实图像数据收集成本高、难扩展的问题。提出Text-Printed Image（TPI）方法，将文本直接渲染为合成图像，弥合图文模态差距，实现低成本、可自动扩增的数据生成，显著提升文本中心训练效果。**

- **链接: [https://arxiv.org/pdf/2512.03463v1](https://arxiv.org/pdf/2512.03463v1)**

> **作者:** Shojiro Yamabe; Futa Waseda; Daiki Shiono; Tsubasa Takahashi
>
> **摘要:** Recent large vision-language models (LVLMs) have been applied to diverse VQA tasks. However, achieving practical performance typically requires task-specific fine-tuning with large numbers of image-text pairs, which are costly to collect. In this work, we study text-centric training, a setting where only textual descriptions are available and no real images are provided, as a paradigm for low-cost data scaling. Unlike images, whose collection is often restricted by privacy constraints and scarcity in niche domains, text is widely available. Moreover, text is easily editable, enabling automatic diversification and expansion with LLMs at minimal human effort. While this offers clear advantages over image collection in terms of scalability and cost, training on raw text without images still yields limited gains on VQA tasks because of the image-text modality gap. To address this issue, we propose a Text-Printed Image (TPI), which generates synthetic images by directly rendering the given textual description on a plain white canvas. This simple rendering projects text into the image modality and can be integrated into arbitrary existing LVLM training pipelines at low cost. Moreover, TPI preserves the semantics of the text, whereas text-to-image models often fail to do. Across four models and seven benchmarks, our systematic experiments show that TPI enables more effective text-centric training than synthetic images generated by a diffusion model. We further explore TPI as a low-cost data-augmentation strategy and demonstrate its practical utility. Overall, our findings highlight the significant potential of text-centric training and, more broadly, chart a path toward fully automated data generation for LVLMs.
>
---
#### [new 075] GeoVideo: Introducing Geometric Regularization into Video Generation Model
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决现有方法在2D空间中生成视频时出现的几何不一致、运动不合理等问题。通过引入深度图预测与多视角几何损失，将3D结构信息融入扩散模型，增强时空一致性与物理合理性，提升生成视频的几何稳定性。**

- **链接: [https://arxiv.org/pdf/2512.03453v1](https://arxiv.org/pdf/2512.03453v1)**

> **作者:** Yunpeng Bai; Shaoheng Fang; Chaohui Yu; Fan Wang; Qixing Huang
>
> **备注:** Project Page: https://geovideo.github.io/GeoVideo/
>
> **摘要:** Recent advances in video generation have enabled the synthesis of high-quality and visually realistic clips using diffusion transformer models. However, most existing approaches operate purely in the 2D pixel space and lack explicit mechanisms for modeling 3D structures, often resulting in temporally inconsistent geometries, implausible motions, and structural artifacts. In this work, we introduce geometric regularization losses into video generation by augmenting latent diffusion models with per-frame depth prediction. We adopted depth as the geometric representation because of the great progress in depth prediction and its compatibility with image-based latent encoders. Specifically, to enforce structural consistency over time, we propose a multi-view geometric loss that aligns the predicted depth maps across frames within a shared 3D coordinate system. Our method bridges the gap between appearance generation and 3D structure modeling, leading to improved spatio-temporal coherence, shape consistency, and physical plausibility. Experiments across multiple datasets show that our approach produces significantly more stable and geometrically consistent results than existing baselines.
>
---
#### [new 076] CSMapping: Scalable Crowdsourced Semantic Mapping and Topology Inference for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CSMapping系统，解决低质量众包数据下自动驾驶地图构建的精度与可扩展性问题。通过训练隐空间扩散模型建立地图结构先验，结合约束优化实现鲁棒语义映射与拓扑中心线生成，显著提升地图质量并随数据量增长而持续改进。**

- **链接: [https://arxiv.org/pdf/2512.03510v1](https://arxiv.org/pdf/2512.03510v1)**

> **作者:** Zhijian Qiao; Zehuan Yu; Tong Li; Chih-Chung Chou; Wenchao Ding; Shaojie Shen
>
> **摘要:** Crowdsourcing enables scalable autonomous driving map construction, but low-cost sensor noise hinders quality from improving with data volume. We propose CSMapping, a system that produces accurate semantic maps and topological road centerlines whose quality consistently increases with more crowdsourced data. For semantic mapping, we train a latent diffusion model on HD maps (optionally conditioned on SD maps) to learn a generative prior of real-world map structure, without requiring paired crowdsourced/HD-map supervision. This prior is incorporated via constrained MAP optimization in latent space, ensuring robustness to severe noise and plausible completion in unobserved areas. Initialization uses a robust vectorized mapping module followed by diffusion inversion; optimization employs efficient Gaussian-basis reparameterization, projected gradient descent zobracket multi-start, and latent-space factor-graph for global consistency. For topological mapping, we apply confidence-weighted k-medoids clustering and kinematic refinement to trajectories, yielding smooth, human-like centerlines robust to trajectory variation. Experiments on nuScenes, Argoverse 2, and a large proprietary dataset achieve state-of-the-art semantic and topological mapping performance, with thorough ablation and scalability studies.
>
---
#### [new 077] Object Counting with GPT-4o and GPT-5: A Comparative Study
- **分类: cs.CV**

- **简介: 该论文研究零样本物体计数任务，旨在无需标注数据或视觉样例，仅通过文本提示实现对新类别物体的计数。利用GPT-4o与GPT-5的多模态能力，在FSC-147和CARPK数据集上进行零样本计数，结果达到甚至超过现有先进方法水平。**

- **链接: [https://arxiv.org/pdf/2512.03233v1](https://arxiv.org/pdf/2512.03233v1)**

> **作者:** Richard Füzesséry; Kaziwa Saleh; Sándor Szénási; Zoltán Vámossy
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Zero-shot object counting attempts to estimate the number of object instances belonging to novel categories that the vision model performing the counting has never encountered during training. Existing methods typically require large amount of annotated data and often require visual exemplars to guide the counting process. However, large language models (LLMs) are powerful tools with remarkable reasoning and data understanding abilities, which suggest the possibility of utilizing them for counting tasks without any supervision. In this work we aim to leverage the visual capabilities of two multi-modal LLMs, GPT-4o and GPT-5, to perform object counting in a zero-shot manner using only textual prompts. We evaluate both models on the FSC-147 and CARPK datasets and provide a comparative analysis. Our findings show that the models achieve performance comparable to the state-of-the-art zero-shot approaches on FSC-147, in some cases, even surpass them.
>
---
#### [new 078] ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文针对扩散变压器模型部署中的高内存与高延迟问题，提出ConvRot方法，通过分组旋转与哈达玛变换抑制行列异常值，实现无需重训练的4位量化（W4A4）。设计Plug-and-play模块ConvLinear4bit，显著提升推理速度并减少内存占用，首次在扩散模型中实现高效旋转基4位量化。**

- **链接: [https://arxiv.org/pdf/2512.03673v1](https://arxiv.org/pdf/2512.03673v1)**

> **作者:** Feice Huang; Zuliang Han; Xing Zhou; Yihuang Chen; Lifei Zhu; Haoqian Wang
>
> **摘要:** Diffusion transformers have demonstrated strong capabilities in generating high-quality images. However, as model size increases, the growing memory footprint and inference latency pose significant challenges for practical deployment. Recent studies in large language models (LLMs) show that rotation-based techniques can smooth outliers and enable 4-bit quantization, but these approaches often incur substantial overhead and struggle with row-wise outliers in diffusion transformers. To address these challenges, we propose ConvRot, a group-wise rotation-based quantization method that leverages regular Hadamard transform (RHT) to suppress both row-wise and column-wise outliers while reducing complexity from quadratic to linear. Building on this, we design ConvLinear4bit, a plug-and-play module that integrates rotation, quantization, GEMM, and dequantization, enabling W4A4 inference without retraining and preserving visual quality. Experiments on FLUX.1-dev demonstrate a 2.26$\times$ speedup and 4.05$\times$ memory reduction while maintaining image fidelity. To our knowledge, this is the first application of rotation-based quantization for plug-and-play W4A4 inference in diffusion transformers.
>
---
#### [new 079] SimFlow: Simplified and End-to-End Training of Latent Normalizing Flows
- **分类: cs.CV**

- **简介: 该论文针对归一化流（NF）训练中数据增强复杂和VAE编码器固定导致生成质量不佳的问题，提出固定VAE编码器输出方差为常数的简化方法。该方法无需额外噪声与去噪步骤，实现端到端训练，显著提升生成质量，在ImageNet 256×256上取得gFID 1.91，达到当前归一化流最佳水平。**

- **链接: [https://arxiv.org/pdf/2512.04084v1](https://arxiv.org/pdf/2512.04084v1)**

> **作者:** Qinyu Zhao; Guangting Zheng; Tao Yang; Rui Zhu; Xingjian Leng; Stephen Gould; Liang Zheng
>
> **备注:** Project Page: https://qinyu-allen-zhao.github.io/SimFlow/
>
> **摘要:** Normalizing Flows (NFs) learn invertible mappings between the data and a Gaussian distribution. Prior works usually suffer from two limitations. First, they add random noise to training samples or VAE latents as data augmentation, introducing complex pipelines including extra noising and denoising steps. Second, they use a pretrained and frozen VAE encoder, resulting in suboptimal reconstruction and generation quality. In this paper, we find that the two issues can be solved in a very simple way: just fixing the variance (which would otherwise be predicted by the VAE encoder) to a constant (e.g., 0.5). On the one hand, this method allows the encoder to output a broader distribution of tokens and the decoder to learn to reconstruct clean images from the augmented token distribution, avoiding additional noise or denoising design. On the other hand, fixed variance simplifies the VAE evidence lower bound, making it stable to train an NF with a VAE jointly. On the ImageNet $256 \times 256$ generation task, our model SimFlow obtains a gFID score of 2.15, outperforming the state-of-the-art method STARFlow (gFID 2.40). Moreover, SimFlow can be seamlessly integrated with the end-to-end representation alignment (REPA-E) method and achieves an improved gFID of 1.91, setting a new state of the art among NFs.
>
---
#### [new 080] Highly Efficient Test-Time Scaling for T2I Diffusion Models with Text Embedding Perturbation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像扩散模型的测试时扩展（TTS）任务，解决现有方法忽视噪声随机性影响的问题。提出文本嵌入扰动新策略，结合频率分析，实现步进式扰动与自适应强度调整，提升生成图像的多样性和质量，可无缝集成且几乎无额外计算开销。**

- **链接: [https://arxiv.org/pdf/2512.03996v1](https://arxiv.org/pdf/2512.03996v1)**

> **作者:** Hang Xu; Linjiang Huang; Feng Zhao
>
> **摘要:** Test-time scaling (TTS) aims to achieve better results by increasing random sampling and evaluating samples based on rules and metrics. However, in text-to-image(T2I) diffusion models, most related works focus on search strategies and reward models, yet the impact of the stochastic characteristic of noise in T2I diffusion models on the method's performance remains unexplored. In this work, we analyze the effects of randomness in T2I diffusion models and explore a new format of randomness for TTS: text embedding perturbation, which couples with existing randomness like SDE-injected noise to enhance generative diversity and quality. We start with a frequency-domain analysis of these formats of randomness and their impact on generation, and find that these two randomness exhibit complementary behavior in the frequency domain: spatial noise favors low-frequency components (early steps), while text embedding perturbation enhances high-frequency details (later steps), thereby compensating for the potential limitations of spatial noise randomness in high-frequency manipulation. Concurrently, text embedding demonstrates varying levels of tolerance to perturbation across different dimensions of the generation process. Specifically, our method consists of two key designs: (1) Introducing step-based text embedding perturbation, combining frequency-guided noise schedules with spatial noise perturbation. (2) Adapting the perturbation intensity selectively based on their frequency-specific contributions to generation and tolerance to perturbation. Our approach can be seamlessly integrated into existing TTS methods and demonstrates significant improvements on multiple benchmarks with almost no additional computation. Code is available at \href{https://github.com/xuhang07/TEP-Diffusion}{https://github.com/xuhang07/TEP-Diffusion}.
>
---
#### [new 081] PixPerfect: Seamless Latent Diffusion Local Editing with Discriminative Pixel-Space Refinement
- **分类: cs.CV**

- **简介: 该论文针对潜在扩散模型（LDM）在图像局部编辑中因潜空间压缩导致的像素级不一致问题，提出PixPerfect框架。通过可微分判别像素空间、真实伪影模拟训练及直接像素空间优化，有效消除色差、纹理错位和边界痕迹，提升编辑质量与泛化能力，显著增强图像修复与局部编辑的感知保真度。**

- **链接: [https://arxiv.org/pdf/2512.03247v1](https://arxiv.org/pdf/2512.03247v1)**

> **作者:** Haitian Zheng; Yuan Yao; Yongsheng Yu; Yuqian Zhou; Jiebo Luo; Zhe Lin
>
> **备注:** Published in the Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Latent Diffusion Models (LDMs) have markedly advanced the quality of image inpainting and local editing. However, the inherent latent compression often introduces pixel-level inconsistencies, such as chromatic shifts, texture mismatches, and visible seams along editing boundaries. Existing remedies, including background-conditioned latent decoding and pixel-space harmonization, usually fail to fully eliminate these artifacts in practice and do not generalize well across different latent representations or tasks. We introduce PixPerfect, a pixel-level refinement framework that delivers seamless, high-fidelity local edits across diverse LDM architectures and tasks. PixPerfect leverages (i) a differentiable discriminative pixel space that amplifies and suppresses subtle color and texture discrepancies, (ii) a comprehensive artifact simulation pipeline that exposes the refiner to realistic local editing artifacts during training, and (iii) a direct pixel-space refinement scheme that ensures broad applicability across diverse latent representations and tasks. Extensive experiments on inpainting, object removal, and insertion benchmarks demonstrate that PixPerfect substantially enhances perceptual fidelity and downstream editing performance, establishing a new standard for robust and high-fidelity localized image editing.
>
---
#### [new 082] DirectDrag: High-Fidelity, Mask-Free, Prompt-Free Drag-based Image Editing via Readout-Guided Feature Alignment
- **分类: cs.CV**

- **简介: 该论文提出DirectDrag，一种无需掩码和文本提示的拖拽图像编辑方法。针对现有方法依赖人工标注导致效率低、易产生视觉伪影的问题，提出自动软掩码生成与读出引导特征对齐机制，实现高保真、精准的点驱动编辑，显著提升图像质量和交互性。**

- **链接: [https://arxiv.org/pdf/2512.03981v1](https://arxiv.org/pdf/2512.03981v1)**

> **作者:** Sheng-Hao Liao; Shang-Fu Chen; Tai-Ming Huang; Wen-Huang Cheng; Kai-Lung Hua
>
> **摘要:** Drag-based image editing using generative models provides intuitive control over image structures. However, existing methods rely heavily on manually provided masks and textual prompts to preserve semantic fidelity and motion precision. Removing these constraints creates a fundamental trade-off: visual artifacts without masks and poor spatial control without prompts. To address these limitations, we propose DirectDrag, a novel mask- and prompt-free editing framework. DirectDrag enables precise and efficient manipulation with minimal user input while maintaining high image fidelity and accurate point alignment. DirectDrag introduces two key innovations. First, we design an Auto Soft Mask Generation module that intelligently infers editable regions from point displacement, automatically localizing deformation along movement paths while preserving contextual integrity through the generative model's inherent capacity. Second, we develop a Readout-Guided Feature Alignment mechanism that leverages intermediate diffusion activations to maintain structural consistency during point-based edits, substantially improving visual fidelity. Despite operating without manual mask or prompt, DirectDrag achieves superior image quality compared to existing methods while maintaining competitive drag accuracy. Extensive experiments on DragBench and real-world scenarios demonstrate the effectiveness and practicality of DirectDrag for high-quality, interactive image manipulation. Project Page: https://frakw.github.io/DirectDrag/. Code is available at: https://github.com/frakw/DirectDrag.
>
---
#### [new 083] LLM-Guided Material Inference for 3D Point Clouds
- **分类: cs.CV; cs.GR**

- **简介: 该论文针对3D点云中材料属性缺失的问题，提出一种两阶段LLM引导的材料推断方法。通过解耦语义与材料推理，零样本地从几何分割中推断物体类别及合理材质，提升3D内容的真实性。**

- **链接: [https://arxiv.org/pdf/2512.03237v1](https://arxiv.org/pdf/2512.03237v1)**

> **作者:** Nafiseh Izadyar; Teseo Schneider
>
> **摘要:** Most existing 3D shape datasets and models focus solely on geometry, overlooking the material properties that determine how objects appear. We introduce a two-stage large language model (LLM) based method for inferring material composition directly from 3D point clouds with coarse segmentations. Our key insight is to decouple reasoning about what an object is from what it is made of. In the first stage, an LLM predicts the object's semantic; in the second stage, it assigns plausible materials to each geometric segment, conditioned on the inferred semantics. Both stages operate in a zero-shot manner, without task-specific training. Because existing datasets lack reliable material annotations, we evaluate our method using an LLM-as-a-Judge implemented in DeepEval. Across 1,000 shapes from Fusion/ABS and ShapeNet, our method achieves high semantic and material plausibility. These results demonstrate that language models can serve as general-purpose priors for bridging geometric reasoning and material understanding in 3D data.
>
---
#### [new 084] PULSE: A Unified Multi-Task Architecture for Cardiac Segmentation, Diagnosis, and Few-Shot Cross-Modality Clinical Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PULSE，一个统一的多任务心脏影像分析框架，解决分割、诊断与跨模态临床适配分离的问题。通过自监督表示与复合监督策略，实现像素级分割、疾病分类与临床报告生成一体化，具备跨数据集与模态的强泛化能力，支持少样本适应，推动心脏影像分析向可扩展的通用模型发展。**

- **链接: [https://arxiv.org/pdf/2512.03848v1](https://arxiv.org/pdf/2512.03848v1)**

> **作者:** Hania Ghouse; Maryam Alsharqi; Farhad R. Nezami; Muzammil Behzad
>
> **摘要:** Cardiac image analysis remains fragmented across tasks: anatomical segmentation, disease classification, and grounded clinical report generation are typically handled by separate networks trained under different data regimes. No existing framework unifies these objectives within a single architecture while retaining generalization across imaging modalities and datasets. We introduce PULSE, a multi-task vision-language framework built on self-supervised representations and optimized through a composite supervision strategy that balances region overlap learning, pixel wise classification fidelity, and boundary aware IoU refinement. A multi-scale token reconstruction decoder enables anatomical segmentation, while shared global representations support disease classification and clinically grounded text output allowing the model to transition from pixels to structures and finally clinical reasoning within one architecture. Unlike prior task-specific pipelines, PULSE learns task-invariant cardiac priors, generalizes robustly across datasets, and can be adapted to new imaging modalities with minimal supervision. This moves the field closer to a scalable, foundation style cardiac analysis framework.
>
---
#### [new 085] Heatmap Pooling Network for Action Recognition from RGB Videos
- **分类: cs.CV**

- **简介: 该论文针对RGB视频中动作识别任务，解决特征冗余、噪声敏感和存储成本高等问题。提出Heatmap Pooling Network（HP-Net），通过反馈池化模块提取紧凑鲁棒的体感特征，并设计空间-运动协同学习与文本精修模块，融合多模态信息，显著提升识别性能。**

- **链接: [https://arxiv.org/pdf/2512.03837v1](https://arxiv.org/pdf/2512.03837v1)**

> **作者:** Mengyuan Liu; Jinfu Liu; Yongkang Jiang; Bin He
>
> **备注:** Final Version of IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** Human action recognition (HAR) in videos has garnered widespread attention due to the rich information in RGB videos. Nevertheless, existing methods for extracting deep features from RGB videos face challenges such as information redundancy, susceptibility to noise and high storage costs. To address these issues and fully harness the useful information in videos, we propose a novel heatmap pooling network (HP-Net) for action recognition from videos, which extracts information-rich, robust and concise pooled features of the human body in videos through a feedback pooling module. The extracted pooled features demonstrate obvious performance advantages over the previously obtained pose data and heatmap features from videos. In addition, we design a spatial-motion co-learning module and a text refinement modulation module to integrate the extracted pooled features with other multimodal data, enabling more robust action recognition. Extensive experiments on several benchmarks namely NTU RGB+D 60, NTU RGB+D 120, Toyota-Smarthome and UAV-Human consistently verify the effectiveness of our HP-Net, which outperforms the existing human action recognition methods. Our code is publicly available at: https://github.com/liujf69/HPNet-Action.
>
---
#### [new 086] Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对直播内容审核中时效性、多模态与新形式违规难以应对的问题，提出融合监督分类与MLLM增强相似性匹配的混合框架。通过多模态输入处理，实现对已知与新型违规内容的高效检测，在生产环境中显著降低不良内容传播。**

- **链接: [https://arxiv.org/pdf/2512.03553v1](https://arxiv.org/pdf/2512.03553v1)**

> **作者:** Wei Chee Yew; Hailun Xu; Sanjay Saha; Xiaotian Fan; Hiok Hian Ong; David Yuchen Wang; Kanchan Sarkar; Zhenheng Yang; Danhui Guan
>
> **备注:** Accepted at KDD 2026
>
> **摘要:** Content moderation remains a critical yet challenging task for large-scale user-generated video platforms, especially in livestreaming environments where moderation must be timely, multimodal, and robust to evolving forms of unwanted content. We present a hybrid moderation framework deployed at production scale that combines supervised classification for known violations with reference-based similarity matching for novel or subtle cases. This hybrid design enables robust detection of both explicit violations and novel edge cases that evade traditional classifiers. Multimodal inputs (text, audio, visual) are processed through both pipelines, with a multimodal large language model (MLLM) distilling knowledge into each to boost accuracy while keeping inference lightweight. In production, the classification pipeline achieves 67% recall at 80% precision, and the similarity pipeline achieves 76% recall at 80% precision. Large-scale A/B tests show a 6-8% reduction in user views of unwanted livestreams}. These results demonstrate a scalable and adaptable approach to multimodal content governance, capable of addressing both explicit violations and emerging adversarial behaviors.
>
---
#### [new 087] GalaxyDiT: Efficient Video Generation with Guidance Alignment and Adaptive Proxy in Diffusion Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对视频生成中扩散模型计算效率低的问题，提出GalaxyDiT方法。通过引导对齐与自适应代理选择，实现训练-free加速。在保持高质量的前提下，显著提升生成速度，最大提速2.37倍，同时大幅降低计算开销。**

- **链接: [https://arxiv.org/pdf/2512.03451v1](https://arxiv.org/pdf/2512.03451v1)**

> **作者:** Zhiye Song; Steve Dai; Ben Keller; Brucek Khailany
>
> **摘要:** Diffusion models have revolutionized video generation, becoming essential tools in creative content generation and physical simulation. Transformer-based architectures (DiTs) and classifier-free guidance (CFG) are two cornerstones of this success, enabling strong prompt adherence and realistic video quality. Despite their versatility and superior performance, these models require intensive computation. Each video generation requires dozens of iterative steps, and CFG doubles the required compute. This inefficiency hinders broader adoption in downstream applications. We introduce GalaxyDiT, a training-free method to accelerate video generation with guidance alignment and systematic proxy selection for reuse metrics. Through rank-order correlation analysis, our technique identifies the optimal proxy for each video model, across model families and parameter scales, thereby ensuring optimal computational reuse. We achieve $1.87\times$ and $2.37\times$ speedup on Wan2.1-1.3B and Wan2.1-14B with only 0.97% and 0.72% drops on the VBench-2.0 benchmark. At high speedup rates, our approach maintains superior fidelity to the base model, exceeding prior state-of-the-art approaches by 5 to 10 dB in peak signal-to-noise ratio (PSNR).
>
---
#### [new 088] Flux4D: Flow-based Unsupervised 4D Reconstruction
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出Flux4D，一种无监督的4D动态场景重建方法。针对现有方法在大规模动态场景重建中依赖标注、可扩展性差的问题，Flux4D通过光度损失与“尽可能静态”正则化，直接从原始数据预测3D高斯及其运动，实现高效、可扩展、泛化性强的无监督重建。**

- **链接: [https://arxiv.org/pdf/2512.03210v1](https://arxiv.org/pdf/2512.03210v1)**

> **作者:** Jingkang Wang; Henry Che; Yun Chen; Ze Yang; Lily Goli; Sivabalan Manivasagam; Raquel Urtasun
>
> **备注:** NeurIPS 2025. Project page: https://waabi.ai/flux4d/
>
> **摘要:** Reconstructing large-scale dynamic scenes from visual observations is a fundamental challenge in computer vision, with critical implications for robotics and autonomous systems. While recent differentiable rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved impressive photorealistic reconstruction, they suffer from scalability limitations and require annotations to decouple actor motion. Existing self-supervised methods attempt to eliminate explicit annotations by leveraging motion cues and geometric priors, yet they remain constrained by per-scene optimization and sensitivity to hyperparameter tuning. In this paper, we introduce Flux4D, a simple and scalable framework for 4D reconstruction of large-scale dynamic scenes. Flux4D directly predicts 3D Gaussians and their motion dynamics to reconstruct sensor observations in a fully unsupervised manner. By adopting only photometric losses and enforcing an "as static as possible" regularization, Flux4D learns to decompose dynamic elements directly from raw data without requiring pre-trained supervised models or foundational priors simply by training across many scenes. Our approach enables efficient reconstruction of dynamic scenes within seconds, scales effectively to large datasets, and generalizes well to unseen environments, including rare and unknown objects. Experiments on outdoor driving datasets show Flux4D significantly outperforms existing methods in scalability, generalization, and reconstruction quality.
>
---
#### [new 089] Colon-X: Advancing Intelligent Colonoscopy from Multimodal Understanding to Clinical Reasoning
- **分类: cs.CV**

- **简介: 该论文提出Colon-X，面向结肠镜检查的多模态智能研究。针对多模态理解到临床推理的转化难题，构建了全球最大规模的结肠镜视觉问答数据集ColonVQA，并提出基于专家辩论的推理数据集ColonReason与专用模型ColonR1，显著提升在数据稀缺下的推理性能，推动结肠镜智能分析发展。**

- **链接: [https://arxiv.org/pdf/2512.03667v1](https://arxiv.org/pdf/2512.03667v1)**

> **作者:** Ge-Peng Ji; Jingyi Liu; Deng-Ping Fan; Nick Barnes
>
> **备注:** Technical report
>
> **摘要:** In this study, we present Colon-X, an open initiative aimed at advancing multimodal intelligence in colonoscopy. We begin by constructing ColonVQA, the most comprehensive multimodal dataset ever built for colonoscopy, featuring over 1.1M+ visual question answering entries across 76 clinical findings and 18 multimodal tasks. Beyond serving as a community-wide data foundation, we further investigate a critical yet underexplored transition in colonoscopy - evolving from multimodal understanding to clinical reasoning: (a) To capture the current landscape of multimodal understanding behaviors, we systematically assess the generalizability of 22 multimodal large language models and examine their reliability under human-induced perturbations. The results reveal that clinical outputs from leading MLLMs remain far from robust and trustworthy. (b) To narrow this gap, we further explore reasoning-centric intelligence tailored for colonoscopy. Specifically, we curate ColonReason, a clinically grounded reasoning dataset annotated through a multi-expert debating pipeline, and develop ColonR1, the first R1-styled model incorporating task-adaptive rewarding and gradient-stable optimization techniques. Under data-scarce conditions, our ColonR1 achieves 56.61% overall accuracy, outperforming supervised fine-tuning by 25.22%, and sets a new reasoning-enabled baseline for multimodal colonoscopy analysis. All data and model resources are publicly available at https://github.com/ai4colonoscopy/Colon-X.
>
---
#### [new 090] LM-CartSeg: Automated Segmentation of Lateral and Medial Cartilage and Subchondral Bone for Radiomics Analysis
- **分类: cs.CV**

- **简介: 该论文提出LM-CartSeg，一个全自动的膝关节软骨与骨下骨分割及内外侧分区方法，解决传统放射组学依赖人工标注、缺乏质量控制的问题。基于双数据集训练的nnU-Net模型，结合几何规则后处理与主成分分析实现稳定分区，生成可用于多中心研究的高质量放射组学特征。**

- **链接: [https://arxiv.org/pdf/2512.03449v1](https://arxiv.org/pdf/2512.03449v1)**

> **作者:** Tongxu Zhang
>
> **摘要:** Background and Objective: Radiomics of knee MRI requires robust, anatomically meaningful regions of interest (ROIs) that jointly capture cartilage and subchondral bone. Most existing work relies on manual ROIs and rarely reports quality control (QC). We present LM-CartSeg, a fully automatic pipeline for cartilage/bone segmentation, geometric lateral/medial (L/M) compartmentalisation and radiomics analysis. Methods: Two 3D nnU-Net models were trained on SKM-TEA (138 knees) and OAIZIB-CM (404 knees). At test time, zero-shot predictions were fused and refined by simple geometric rules: connected-component cleaning, construction of 10 mm subchondral bone bands in physical space, and a data-driven tibial L/M split based on PCA and k-means. Segmentation was evaluated on an OAIZIB-CM test set (103 knees) and on SKI-10 (100 knees). QC used volume and thickness signatures. From 10 ROIs we extracted 4 650 non-shape radiomic features to study inter-compartment similarity, dependence on ROI size, and OA vs. non-OA classification on OAIZIB-CM Results: Post-processing improved macro ASSD on OAIZIB-CM from 2.63 to 0.36 mm and HD95 from 25.2 to 3.35 mm, with DSC 0.91; zero-shot DSC on SKI-10 was 0.80. The geometric L/M rule produced stable compartments across datasets, whereas a direct L/M nnU-Net showed domain-dependent side swaps. Only 6 to 12 percent of features per ROI were strongly correlated with volume or thickness. Radiomics-based models models restricted to size-linked features. Conclusions: LM-CartSeg yields automatic, QCd ROIs and radiomic features that carry discriminative information beyond simple morphometry, providing a practical foundation for multi-centre knee OA radiomics studies.
>
---
#### [new 091] MKSNet: Advanced Small Object Detection in Remote Sensing Imagery with Multi-Kernel and Dual Attention Mechanisms
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对遥感图像中小目标检测难题，提出MKSNet模型。通过多核选择机制增强上下文感知，结合空间与通道双重注意力，有效提升小目标特征表达与定位精度。在DOTA-v1.0和HRSC2016数据集上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.03640v1](https://arxiv.org/pdf/2512.03640v1)**

> **作者:** Jiahao Zhang; Xiao Zhao; Guangyu Gao
>
> **摘要:** Deep convolutional neural networks (DCNNs) have substantially advanced object detection capabilities, particularly in remote sensing imagery. However, challenges persist, especially in detecting small objects where the high resolution of these images and the small size of target objects often result in a loss of critical information in the deeper layers of conventional CNNs. Additionally, the extensive spatial redundancy and intricate background details typical in remote-sensing images tend to obscure these small targets. To address these challenges, we introduce Multi-Kernel Selection Network (MKSNet), a novel network architecture featuring a novel Multi-Kernel Selection mechanism. The MKS mechanism utilizes large convolutional kernels to effectively capture an extensive range of contextual information. This innovative design allows for adaptive kernel size selection, significantly enhancing the network's ability to dynamically process and emphasize crucial spatial details for small object detection. Furthermore, MKSNet also incorporates a dual attention mechanism, merging spatial and channel attention modules. The spatial attention module adaptively fine-tunes the spatial weights of feature maps, focusing more intensively on relevant regions while mitigating background noise. Simultaneously, the channel attention module optimizes channel information selection, improving feature representation and detection accuracy. Empirical evaluations on the DOTA-v1.0 and HRSC2016 benchmark demonstrate that MKSNet substantially surpasses existing state-of-the-art models in detecting small objects in remote sensing images. These results highlight MKSNet's superior ability to manage the complexities associated with multi-scale and high-resolution image data, confirming its effectiveness and innovation in remote sensing object detection.
>
---
#### [new 092] Dual Cross-Attention Siamese Transformer for Rectal Tumor Regrowth Assessment in Watch-and-Wait Endoscopy
- **分类: cs.CV**

- **简介: 该论文针对直肠癌患者在“观察与等待”策略中早期检测肿瘤复发病变的难题，提出双交叉注意力孪生Swin Transformer模型（SSDCA），通过对比治疗后与随访内镜图像，实现对临床完全缓解与局部复发病变的准确区分，提升了诊断的敏感性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.03883v1](https://arxiv.org/pdf/2512.03883v1)**

> **作者:** Jorge Tapias Gomez; Despoina Kanata; Aneesh Rangnekar; Christina Lee; Julio Garcia-Aguilar; Joshua Jesse Smith; Harini Veeraraghavan
>
> **备注:** 6 pages, 5 figures, 1 table, submitted to ISBI conference
>
> **摘要:** Increasing evidence supports watch-and-wait (WW) surveillance for patients with rectal cancer who show clinical complete response (cCR) at restaging following total neoadjuvant treatment (TNT). However, objectively accurate methods to early detect local regrowth (LR) from follow-up endoscopy images during WW are essential to manage care and prevent distant metastases. Hence, we developed a Siamese Swin Transformer with Dual Cross-Attention (SSDCA) to combine longitudinal endoscopic images at restaging and follow-up and distinguish cCR from LR. SSDCA leverages pretrained Swin transformers to extract domain agnostic features and enhance robustness to imaging variations. Dual cross attention is implemented to emphasize features from the two scans without requiring any spatial alignment of images to predict response. SSDCA as well as Swin-based baselines were trained using image pairs from 135 patients and evaluated on a held-out set of image pairs from 62 patients. SSDCA produced the best balanced accuracy (81.76\% $\pm$ 0.04), sensitivity (90.07\% $\pm$ 0.08), and specificity (72.86\% $\pm$ 0.05). Robustness analysis showed stable performance irrespective of artifacts including blood, stool, telangiectasia, and poor image quality. UMAP clustering of extracted features showed maximal inter-cluster separation (1.45 $\pm$ 0.18) and minimal intra-cluster dispersion (1.07 $\pm$ 0.19) with SSDCA, confirming discriminative representation learning.
>
---
#### [new 093] YOLOA: Real-Time Affordance Detection via LLM Adapter
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出YOLOA，一种实时交互式物体与使用属性联合检测模型。针对现有方法忽视“是什么”“在哪里”或任务分离的问题，引入轻量级LLM适配器协同优化检测与使用属性预测，实现高精度（52.8/73.1 mAP）与高速率（最高89.77 FPS）的统一。**

- **链接: [https://arxiv.org/pdf/2512.03418v1](https://arxiv.org/pdf/2512.03418v1)**

> **作者:** Yuqi Ji; Junjie Ke; Lihuo He; Jun Liu; Kaifan Zhang; Yu-Kun Lai; Guiguang Ding; Xinbo Gao
>
> **备注:** 13 pages, 9 figures, conference
>
> **摘要:** Affordance detection aims to jointly address the fundamental "what-where-how" challenge in embodied AI by understanding "what" an object is, "where" the object is located, and "how" it can be used. However, most affordance learning methods focus solely on "how" objects can be used while neglecting the "what" and "where" aspects. Other affordance detection methods treat object detection and affordance learning as two independent tasks, lacking effective interaction and real-time capability. To overcome these limitations, we introduce YOLO Affordance (YOLOA), a real-time affordance detection model that jointly handles these two tasks via a large language model (LLM) adapter. Specifically, YOLOA employs a lightweight detector consisting of object detection and affordance learning branches refined through the LLM Adapter. During training, the LLM Adapter interacts with object and affordance preliminary predictions to refine both branches by generating more accurate class priors, box offsets, and affordance gates. Experiments on our relabeled ADG-Det and IIT-Heat benchmarks demonstrate that YOLOA achieves state-of-the-art accuracy (52.8 / 73.1 mAP on ADG-Det / IIT-Heat) while maintaining real-time performance (up to 89.77 FPS, and up to 846.24 FPS for the lightweight variant). This indicates that YOLOA achieves an excellent trade-off between accuracy and efficiency.
>
---
#### [new 094] Procedural Mistake Detection via Action Effect Modeling
- **分类: cs.CV**

- **简介: 该论文针对程序性任务中的错误检测问题，提出基于动作效应建模（AEM）的统一框架。传统方法仅关注动作执行，忽略其结果。本文通过建模动作产生的实际效果，结合视觉与符号化场景信息，生成鲁棒的效应感知表示，并设计提示驱动检测器，在两个基准上实现领先性能，证明了同时考虑动作与结果对提升错误检测可靠性的重要性。**

- **链接: [https://arxiv.org/pdf/2512.03474v1](https://arxiv.org/pdf/2512.03474v1)**

> **作者:** Wenliang Guo; Yujiang Pu; Yu Kong
>
> **摘要:** Mistake detection in procedural tasks is essential for building intelligent systems that support learning and task execution. Existing approaches primarily analyze how an action is performed, while overlooking what it produces, i.e., the \textbf{action effect}. Yet many errors manifest not in the execution itself but in the resulting outcome, such as an unintended object state or incorrect spatial arrangement. To address this gap, we propose Action Effect Modeling (AEM), a unified framework that jointly captures action execution and its outcomes through a probabilistic formulation. AEM first identifies the outcome of an action by selecting the most informative effect frame based on semantic relevance and visual quality. It then extracts complementary cues from visual grounding and symbolic scene graphs, aligning them in a shared latent space to form robust effect-aware representations. To detect mistakes, we further design a prompt-based detector that incorporates task-specific prompts and aligns each action segment with its intended execution semantics. Our approach achieves state-of-the-art performance on the EgoPER and CaptainCook4D benchmarks under the challenging one-class classification (OCC) setting. These results demonstrate that modeling both execution and outcome yields more reliable mistake detection, and highlight the potential of effect-aware representations to benefit a broader range of downstream applications.
>
---
#### [new 095] Out-of-the-box: Black-box Causal Attacks on Object Detectors
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对对象检测器的黑盒攻击问题，提出BlackCAtt算法，通过识别因果像素实现可解释、不可察觉且与架构无关的攻击。其工作是利用因果像素与检测框结合，精准干扰检测结果，显著优于现有方法，在去除、修改和伪造检测上分别提升2.7至5.75倍。**

- **链接: [https://arxiv.org/pdf/2512.03730v1](https://arxiv.org/pdf/2512.03730v1)**

> **作者:** Melane Navaratnarajah; David A. Kelly; Hana Chockler
>
> **摘要:** Adversarial perturbations are a useful way to expose vulnerabilities in object detectors. Existing perturbation methods are frequently white-box and architecture specific. More importantly, while they are often successful, it is rarely clear why they work. Insights into the mechanism of this success would allow developers to understand and analyze these attacks, as well as fine-tune the model to prevent them. This paper presents BlackCAtt, a black-box algorithm and a tool, which uses minimal, causally sufficient pixel sets to construct explainable, imperceptible, reproducible, architecture-agnostic attacks on object detectors. BlackCAtt combines causal pixels with bounding boxes produced by object detectors to create adversarial attacks that lead to the loss, modification or addition of a bounding box. BlackCAtt works across different object detectors of different sizes and architectures, treating the detector as a black box. We compare the performance of BlackCAtt with other black-box attack methods and show that identification of causal pixels leads to more precisely targeted and less perceptible attacks. On the COCO test dataset, our approach is 2.7 times better than the baseline in removing a detection, 3.86 times better in changing a detection, and 5.75 times better in triggering new, spurious, detections. The attacks generated by BlackCAtt are very close to the original image, and hence imperceptible, demonstrating the power of causal pixels.
>
---
#### [new 096] Fairness-Aware Fine-Tuning of Vision-Language Models for Medical Glaucoma Diagnosis
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对医学视图语言模型在青光眼诊断中存在的人群诊断准确率差异问题，提出公平性感知的低秩微调方法（FR-LoRA、GR-LoRA、Hybrid-LoRA），通过可微分的MaxAccGap损失与梯度加权机制，实现公平性优化。实验表明其显著降低差距，仅需0.24%可训练参数，适用于资源受限场景。**

- **链接: [https://arxiv.org/pdf/2512.03477v1](https://arxiv.org/pdf/2512.03477v1)**

> **作者:** Zijian Gu; Yuxi Liu; Zhenhao Zhang; Song Wang
>
> **备注:** 10 pages, 3 tables
>
> **摘要:** Vision-language models achieve expert-level performance on medical imaging tasks but exhibit significant diagnostic accuracy disparities across demographic groups. We introduce fairness-aware Low-Rank Adaptation for medical VLMs, combining parameter efficiency with explicit fairness optimization. Our key algorithmic contribution is a differentiable MaxAccGap loss that enables end-to-end optimization of accuracy parity across demographic groups. We propose three methods: FR-LoRA integrates MaxAccGap regularization into the training objective, GR-LoRA applies inverse frequency weighting to balance gradient contributions, and Hybrid-LoRA combines both mechanisms.Evaluated on 10,000 glaucoma fundus images, GR-LoRA reduces diagnostic accuracy disparities by 69% while maintaining 53.15% overall accuracy. Ablation studies reveal that strong regularization strength achieves optimal fairness with minimal accuracy trade-off, and race-specific optimization yields 60% disparity reduction. Our approach requires only 0.24% trainable parameters, enabling practical deployment of fair medical AI in resource-constrained healthcare settings.
>
---
#### [new 097] ToG-Bench: Task-Oriented Spatio-Temporal Grounding in Egocentric Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ToG-Bench，首个面向第一人称视频的任务导向时空定位基准。针对现有研究局限于对象中心描述、忽视任务推理的问题，构建包含显式/隐式、一对一多对象标注的基准数据集，设计任务级评估指标，系统评测多模态大模型在复杂交互场景下的感知与推理能力。**

- **链接: [https://arxiv.org/pdf/2512.03666v1](https://arxiv.org/pdf/2512.03666v1)**

> **作者:** Qi'ao Xu; Tianwen Qian; Yuqian Fu; Kailing Li; Yang Jiao; Jiacheng Zhang; Xiaoling Wang; Liang He
>
> **备注:** 26 pages
>
> **摘要:** A core capability towards general embodied intelligence lies in localizing task-relevant objects from an egocentric perspective, formulated as Spatio-Temporal Video Grounding (STVG). Despite recent progress, existing STVG studies remain largely confined to object-centric and descriptive instructions, neglecting the task-oriented reasoning that is crucial for embodied agents to accomplish goal-directed interactions. To bridge this gap, we introduce \textbf{ToG-Bench}, the first task-oriented spatio-temporal video grounding benchmark for egocentric videos. ToG-Bench is characterized by three key features: (1) \textbf{Task-oriented Grounding}, which requires identifying and localizing objects based on intended tasks rather than straightforward descriptions; (2) \textbf{Explicit-Implicit Dual Grounding}, where target objects can be either explicitly mentioned or implicitly inferred by contextual reasoning; (3) \textbf{One-to-Many Grounding}, where a single instruction may correspond to multiple objects involved in task execution. Built upon videos sourced from ScanNet, ToG-Bench comprises 100 annotated clips with 2,704 task-oriented grounding instructions, constructed via a semi-automated pipeline that combines foundation model annotation and human refinement. In addition, we introduce a set of task-level evaluation metrics tailored for multi-object and explicit-implicit object grounding, and systematically benchmark seven state-of-the-art MLLMs. Extensive experiments reveal the intrinsic challenges of task-oriented STVG and substantial performance gaps across explicit-implicit and multi-object grounding, highlighting the difficulty of bridging perception and interaction in embodied scenarios. Data and code will be released at: \href{https://github.com/qaxuDev/ToG-Bench}{https://github.com/qaxuDev/ToG-Bench}..
>
---
#### [new 098] Memory-Guided Point Cloud Completion for Dental Reconstruction
- **分类: cs.CV**

- **简介: 该论文针对牙科点云完成任务，解决因遮挡和扫描视角有限导致的大面积缺失问题。提出基于原型记忆的检索增强框架，在编码器-解码器结构中引入可学习的牙形原型记忆，通过融合相似原型特征提供结构先验，提升重建精度与细节恢复能力，实现端到端优化且兼容主流模型。**

- **链接: [https://arxiv.org/pdf/2512.03598v1](https://arxiv.org/pdf/2512.03598v1)**

> **作者:** Jianan Sun; Yukang Huang; Dongzhihan Wang; Mingyu Fan
>
> **摘要:** Partial dental point clouds often suffer from large missing regions caused by occlusion and limited scanning views, which bias encoder-only global features and force decoders to hallucinate structures. We propose a retrieval-augmented framework for tooth completion that integrates a prototype memory into standard encoder--decoder pipelines. After encoding a partial input into a global descriptor, the model retrieves the nearest manifold prototype from a learnable memory and fuses it with the query feature through confidence-gated weighting before decoding. The memory is optimized end-to-end and self-organizes into reusable tooth-shape prototypes without requiring tooth-position labels, thereby providing structural priors that stabilize missing-region inference and free decoder capacity for detail recovery. The module is plug-and-play and compatible with common completion backbones, while keeping the same training losses. Experiments on a self-processed Teeth3DS benchmark demonstrate consistent improvements in Chamfer Distance, with visualizations showing sharper cusps, ridges, and interproximal transitions. Our approach provides a simple yet effective way to exploit cross-sample regularities for more accurate and faithful dental point-cloud completion.
>
---
#### [new 099] DM3D: Deformable Mamba via Offset-Guided Gaussian Sequencing for Point Cloud Understanding
- **分类: cs.CV**

- **简介: 该论文针对点云理解中状态空间模型（SSM）依赖输入顺序与点云不规则性冲突的问题，提出DM3D架构。通过偏移引导的高斯序列化机制，实现结构自适应的局部重采样与全局重排序，并结合三路径频率融合模块，提升特征表达能力。实验表明，该方法在分类、少样本学习和部件分割任务上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2512.03424v1](https://arxiv.org/pdf/2512.03424v1)**

> **作者:** Bin Liu; Chunyang Wang; Xuelian Liu
>
> **摘要:** State Space Models (SSMs) demonstrate significant potential for long-sequence modeling, but their reliance on input order conflicts with the irregular nature of point clouds. Existing approaches often rely on predefined serialization strategies, which cannot adjust based on diverse geometric structures. To overcome this limitation, we propose \textbf{DM3D}, a deformable Mamba architecture for point cloud understanding. Specifically, DM3D introduces an offset-guided Gaussian sequencing mechanism that unifies local resampling and global reordering within a deformable scan. The Gaussian-based KNN Resampling (GKR) enhances structural awareness by adaptively reorganizing neighboring points, while the Gaussian-based Differentiable Reordering (GDR) enables end-to-end optimization of serialization order. Furthermore, a Tri-Path Frequency Fusion module enhances feature complementarity and reduces aliasing. Together, these components enable structure-adaptive serialization of point clouds. Extensive experiments on benchmark datasets show that DM3D achieves state-of-the-art performance in classification, few-shot learning, and part segmentation, demonstrating that adaptive serialization effectively unlocks the potential of SSMs for point cloud understanding.
>
---
#### [new 100] NAS-LoRA: Empowering Parameter-Efficient Fine-Tuning for Visual Foundation Models with Searchable Adaptation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视觉基础模型SAM在医疗、农业等特定领域适应性差的问题，提出NAS-LoRA方法。通过引入轻量级神经架构搜索模块，动态优化权重更新中的先验知识，结合分阶段优化策略，提升模型对高层语义信息的捕捉能力，显著改善适配性能并降低24.14%训练成本。**

- **链接: [https://arxiv.org/pdf/2512.03499v1](https://arxiv.org/pdf/2512.03499v1)**

> **作者:** Renqi Chen; Haoyang Su; Shixiang Tang
>
> **摘要:** The Segment Anything Model (SAM) has emerged as a powerful visual foundation model for image segmentation. However, adapting SAM to specific downstream tasks, such as medical and agricultural imaging, remains a significant challenge. To address this, Low-Rank Adaptation (LoRA) and its variants have been widely employed to enhancing SAM's adaptation performance on diverse domains. Despite advancements, a critical question arises: can we integrate inductive bias into the model? This is particularly relevant since the Transformer encoder in SAM inherently lacks spatial priors within image patches, potentially hindering the acquisition of high-level semantic information. In this paper, we propose NAS-LoRA, a new Parameter-Efficient Fine-Tuning (PEFT) method designed to bridge the semantic gap between pre-trained SAM and specialized domains. Specifically, NAS-LoRA incorporates a lightweight Neural Architecture Search (NAS) block between the encoder and decoder components of LoRA to dynamically optimize the prior knowledge integrated into weight updates. Furthermore, we propose a stage-wise optimization strategy to help the ViT encoder balance weight updates and architectural adjustments, facilitating the gradual learning of high-level semantic information. Various Experiments demonstrate our NAS-LoRA improves existing PEFT methods, while reducing training cost by 24.14% without increasing inference cost, highlighting the potential of NAS in enhancing PEFT for visual foundation models.
>
---
#### [new 101] SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉语言模型在空间推理中缺乏度量精度的问题，提出SpaceTools框架，通过双阶段交互式强化学习（DIRL）实现多工具协同。解决了传统方法依赖手工提示或固定工具链的局限，使模型能自主发现最优工具使用策略，在多个基准上取得领先性能，并成功应用于7-DOF机器人真实操作。**

- **链接: [https://arxiv.org/pdf/2512.04069v1](https://arxiv.org/pdf/2512.04069v1)**

> **作者:** Siyi Chen; Mikaela Angelina Uy; Chan Hee Song; Faisal Ladhak; Adithyavairavan Murali; Qing Qu; Stan Birchfield; Valts Blukis; Jonathan Tremblay
>
> **摘要:** Vision Language Models (VLMs) demonstrate strong qualitative visual understanding, but struggle with metrically precise spatial reasoning required for embodied applications. The agentic paradigm promises that VLMs can use a wide variety of tools that could augment these capabilities, such as depth estimators, segmentation models, and pose estimators. Yet it remains an open challenge how to realize this vision without solely relying on handcrafted prompting strategies or enforcing fixed, predefined tool pipelines that limit VLMs' ability to discover optimal tool-use patterns. Reinforcement Learning could overcome this gap, but has so far been limited to reasoning with a single visual tool due to the large search space in multi-tool reasoning. We introduce Double Interactive Reinforcement Learning (DIRL), a two-phase training framework where VLMs learn to coordinate multiple tools through interactive exploration and feedback. In the teaching phase, we combine demonstrations from a single tool specialist trained via interactive RL with traces from a frontier model using all tools. In the exploration phase, the model further refines multi-tool coordination through continued RL. Our model, SpaceTools, with tool-augmented spatial reasoning ability, achieves state-of-the-art performance on spatial understanding benchmarks (RoboSpatial-Home, BLINK, BOP-ASK) and demonstrates reliable real-world manipulation using a 7-DOF robot as a tool. DIRL provides substantial improvements over the vanilla SFT (+12% on RoboSpatial) and RL (+16% on RoboSpatial) baselines. Project page: https://spacetools.github.io/.
>
---
#### [new 102] V-ITI: Mitigating Hallucinations in Multimodal Large Language Models via Visual Inference-Time Intervention
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态大模型（MLLMs）中的幻觉问题，提出V-ITI框架。通过检测头级别激活模式识别视觉忽视，仅在必要时干预，缓解过度干预问题，有效减少视觉相关幻觉，提升可靠性。**

- **链接: [https://arxiv.org/pdf/2512.03542v1](https://arxiv.org/pdf/2512.03542v1)**

> **作者:** Nan Sun; Zhenyu Zhang; Xixun Lin; Kun Wang; Yanmin Shang; Naibin Gu; Shuohuan Wang; Yu Sun; Hua Wu; Haifeng Wang; Yanan Cao
>
> **摘要:** Multimodal Large Language Models (MLLMs) excel in numerous vision-language tasks yet suffer from hallucinations, producing content inconsistent with input visuals, that undermine reliability in precision-sensitive domains. This issue stems from a fundamental problem of visual neglect, where models fail to adequately prioritize input images. Existing methods typically alleviate hallucinations by intervening in the attention score or output logits, focusing on "how to intervene" but overlooking the prerequisite "when to intervene", which leads to the "over-intervention" problem and subsequently introduces new hallucinations and unnecessary computational overhead. To address this gap, we first investigate the mechanism of visual neglect and reveal it can be accurately detected via head-level activation patterns in MLLMs. We thus propose V-ITI, a lightweight visual inference-time intervention framework integrating a Visual Neglect Detector that identifies visual neglect via head-level discriminative probes and a Visual Recall Intervenor that modulates activations with prestored visual activation information only when the visual neglect is detected. Extensive experiments across eight benchmarks and different MLLM families demonstrate that V-ITI consistently mitigates vision-related hallucinations while preserving general task performance.
>
---
#### [new 103] An Automated Framework for Large-Scale Graph-Based Cerebrovascular Analysis
- **分类: cs.CV; cs.CY**

- **简介: 该论文提出CaravelMetrics框架，用于自动化大规模脑血管分析。针对脑血管形态量化困难的问题，通过骨架化构建图模型，提取15种多尺度特征，实现对血管结构的全局与区域分析，支持年龄、性别及教育水平相关研究，为血管健康与衰老的群体研究提供可扩展的自动化工具。**

- **链接: [https://arxiv.org/pdf/2512.03869v1](https://arxiv.org/pdf/2512.03869v1)**

> **作者:** Daniele Falcetta; Liane S. Canas; Lorenzo Suppa; Matteo Pentassuglia; Jon Cleary; Marc Modat; Sébastien Ourselin; Maria A. Zuluaga
>
> **备注:** Submitted to ISBI 2026. 6 pages, 6 figures
>
> **摘要:** We present CaravelMetrics, a computational framework for automated cerebrovascular analysis that models vessel morphology through skeletonization-derived graph representations. The framework integrates atlas-based regional parcellation, centerline extraction, and graph construction to compute fifteen morphometric, topological, fractal, and geometric features. The features can be estimated globally from the complete vascular network or regionally within arterial territories, enabling multiscale characterization of cerebrovascular organization. Applied to 570 3D TOF-MRA scans from the IXI dataset (ages 20-86), CaravelMetrics yields reproducible vessel graphs capturing age- and sex-related variations and education-associated increases in vascular complexity, consistent with findings reported in the literature. The framework provides a scalable and fully automated approach for quantitative cerebrovascular feature extraction, supporting normative modeling and population-level studies of vascular health and aging.
>
---
#### [new 104] Lean Unet: A Compact Model for Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割任务，解决传统Unet模型参数多、内存占用高、推理慢的问题。提出轻量级LUnet架构，采用固定通道数的扁平结构，通过减少瓶颈层通道数并优化跳接设计，在参数减少30倍以上的情况下，性能优于标准Unet和剪枝模型。**

- **链接: [https://arxiv.org/pdf/2512.03834v1](https://arxiv.org/pdf/2512.03834v1)**

> **作者:** Ture Hassler; Ida Åkerholm; Marcus Nordström; Gabriele Balletti; Orcun Goksel
>
> **摘要:** Unet and its variations have been standard in semantic image segmentation, especially for computer assisted radiology. Current Unet architectures iteratively downsample spatial resolution while increasing channel dimensions to preserve information content. Such a structure demands a large memory footprint, limiting training batch sizes and increasing inference latency. Channel pruning compresses Unet architecture without accuracy loss, but requires lengthy optimization and may not generalize across tasks and datasets. By investigating Unet pruning, we hypothesize that the final structure is the crucial factor, not the channel selection strategy of pruning. Based on our observations, we propose a lean Unet architecture (LUnet) with a compact, flat hierarchy where channels are not doubled as resolution is halved. We evaluate on a public MRI dataset allowing comparable reporting, as well as on two internal CT datasets. We show that a state-of-the-art pruning solution (STAMP) mainly prunes from the layers with the highest number of channels. Comparatively, simply eliminating a random channel at the pruning-identified layer or at the largest layer achieves similar or better performance. Our proposed LUnet with fixed architectures and over 30 times fewer parameters achieves performance comparable to both conventional Unet counterparts and data-adaptively pruned networks. The proposed lean Unet with constant channel count across layers requires far fewer parameters while achieving performance superior to standard Unet for the same total number of parameters. Skip connections allow Unet bottleneck channels to be largely reduced, unlike standard encoder-decoder architectures requiring increased bottleneck channels for information propagation.
>
---
#### [new 105] Label-Efficient Hyperspectral Image Classification via Spectral FiLM Modulation of Low-Level Pretrained Diffusion Features
- **分类: cs.CV**

- **简介: 该论文针对高光谱图像分类中标签稀缺与空间分辨率低的问题，提出一种标签高效框架。利用预训练扩散模型提取低层空间特征，结合谱-空融合的FiLM模块，实现稀疏标注下的鲁棒分类。实验表明，该方法在两个数据集上优于现有技术。**

- **链接: [https://arxiv.org/pdf/2512.03430v1](https://arxiv.org/pdf/2512.03430v1)**

> **作者:** Yuzhen Hu; Biplab Banerjee; Saurabh Prasad
>
> **备注:** Accepted to the ICML 2025 TerraBytes Workshop (June 9, 2025)
>
> **摘要:** Hyperspectral imaging (HSI) enables detailed land cover classification, yet low spatial resolution and sparse annotations pose significant challenges. We present a label-efficient framework that leverages spatial features from a frozen diffusion model pretrained on natural images. Our approach extracts low-level representations from high-resolution decoder layers at early denoising timesteps, which transfer effectively to the low-texture structure of HSI. To integrate spectral and spatial information, we introduce a lightweight FiLM-based fusion module that adaptively modulates frozen spatial features using spectral cues, enabling robust multimodal learning under sparse supervision. Experiments on two recent hyperspectral datasets demonstrate that our method outperforms state-of-the-art approaches using only the provided sparse training labels. Ablation studies further highlight the benefits of diffusion-derived features and spectral-aware fusion. Overall, our results indicate that pretrained diffusion models can support domain-agnostic, label-efficient representation learning for remote sensing and broader scientific imaging tasks.
>
---
#### [new 106] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对视觉问答任务中视觉语言模型计算开销大的问题，提出AdaptVision，通过自适应视觉信息获取机制，实现动态减少视觉标记数量。其核心是基于粗到精策略与强化学习的工具调用机制，有效平衡精度与效率，显著降低资源消耗并提升性能。**

- **链接: [https://arxiv.org/pdf/2512.03794v1](https://arxiv.org/pdf/2512.03794v1)**

> **作者:** Zichuan Lin; Yicheng Liu; Yang Yang; Lvfang Tao; Deheng Ye
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable success in visual question answering tasks, but their reliance on large numbers of visual tokens introduces significant computational overhead. While existing efficient VLM approaches reduce visual tokens through fixed-ratio compression, they operate passively and lack the ability to adapt to varying task requirements. This motivates a fundamental question: Can VLMs autonomously determine the minimum number of visual tokens required for each sample? Inspired by human active vision mechanisms, we introduce AdaptVision, an efficient VLM paradigm that enables adaptive visual token acquisition through a coarse-to-fine approach. Our model initially processes compressed visual tokens from low-resolution images and selectively acquires additional visual information by invoking a bounding box tool to crop key regions when necessary. We train AdaptVision using a reinforcement learning framework that carefully balances accuracy and efficiency. Central to our approach is Decoupled Turn Policy Optimization (DTPO), which decouples the learning objective into two components: (1) tool learning, which optimizes correct tool utilization, and (2) accuracy improvement, which refines the generated responses to improve answer correctness. Based on this formulation, we further decouple advantage estimation by computing separate advantages for tokens associated with each objective. This formulation enables more effective optimization for AdaptVision compared to vanilla GRPO. Comprehensive experiments across multiple VQA benchmarks demonstrate that AdaptVision achieves superior performance while consuming substantially fewer visual tokens than state-of-the-art efficient VLM methods.
>
---
#### [new 107] KeyPointDiffuser: Unsupervised 3D Keypoint Learning via Latent Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出KeyPointDiffuser，一个基于潜在扩散模型的无监督3D关键点学习框架。旨在解决现有方法不适用于无条件生成场景的问题。通过从点云数据中学习具有空间结构的紧凑关键点，实现对3D形状的高效重建与插值，显著提升关键点一致性。**

- **链接: [https://arxiv.org/pdf/2512.03450v1](https://arxiv.org/pdf/2512.03450v1)**

> **作者:** Rhys Newbury; Juyan Zhang; Tin Tran; Hanna Kurniawati; Dana Kulić
>
> **摘要:** Understanding and representing the structure of 3D objects in an unsupervised manner remains a core challenge in computer vision and graphics. Most existing unsupervised keypoint methods are not designed for unconditional generative settings, restricting their use in modern 3D generative pipelines; our formulation explicitly bridges this gap. We present an unsupervised framework for learning spatially structured 3D keypoints from point cloud data. These keypoints serve as a compact and interpretable representation that conditions an Elucidated Diffusion Model (EDM) to reconstruct the full shape. The learned keypoints exhibit repeatable spatial structure across object instances and support smooth interpolation in keypoint space, indicating that they capture geometric variation. Our method achieves strong performance across diverse object categories, yielding a 6 percentage-point improvement in keypoint consistency compared to prior approaches.
>
---
#### [new 108] Unique Lives, Shared World: Learning from Single-Life Videos
- **分类: cs.CV**

- **简介: 该论文提出“单人生”学习范式，利用单一人的第一视角视频自监督训练视觉模型。通过分析不同个体的视频数据，发现模型在几何理解上高度一致，且能泛化到未见环境，仅需30小时单人数据即可达到与多样化网络数据相当的性能，揭示了世界共享结构对视觉表征学习的强信号作用。**

- **链接: [https://arxiv.org/pdf/2512.04085v1](https://arxiv.org/pdf/2512.04085v1)**

> **作者:** Tengda Han; Sayna Ebrahimi; Dilara Gokay; Li Yang Ku; Maks Ovsjanikov; Iva Babukova; Daniel Zoran; Viorica Patraucean; Joao Carreira; Andrew Zisserman; Dima Damen
>
> **摘要:** We introduce the "single-life" learning paradigm, where we train a distinct vision model exclusively on egocentric videos captured by one individual. We leverage the multiple viewpoints naturally captured within a single life to learn a visual encoder in a self-supervised manner. Our experiments demonstrate three key findings. First, models trained independently on different lives develop a highly aligned geometric understanding. We demonstrate this by training visual encoders on distinct datasets each capturing a different life, both indoors and outdoors, as well as introducing a novel cross-attention-based metric to quantify the functional alignment of the internal representations developed by different models. Second, we show that single-life models learn generalizable geometric representations that effectively transfer to downstream tasks, such as depth estimation, in unseen environments. Third, we demonstrate that training on up to 30 hours from one week of the same person's life leads to comparable performance to training on 30 hours of diverse web data, highlighting the strength of single-life representation learning. Overall, our results establish that the shared structure of the world, both leads to consistency in models trained on individual lives, and provides a powerful signal for visual representation learning.
>
---
#### [new 109] Step-by-step Layered Design Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出“分步分层设计生成”新任务，旨在模拟设计师逐步改进设计的复杂过程。针对现有方法将设计合成视为单步生成的问题，提出SLEDGE模型，通过多模态大模型实现基于指令的逐层更新。构建了数据集与评估基准，实验证明其有效性，推动该领域发展。**

- **链接: [https://arxiv.org/pdf/2512.03335v1](https://arxiv.org/pdf/2512.03335v1)**

> **作者:** Faizan Farooq Khan; K J Joseph; Koustava Goswami; Mohamed Elhoseiny; Balaji Vasan Srinivasan
>
> **摘要:** Design generation, in its essence, is a step-by-step process where designers progressively refine and enhance their work through careful modifications. Despite this fundamental characteristic, existing approaches mainly treat design synthesis as a single-step generation problem, significantly underestimating the inherent complexity of the creative process. To bridge this gap, we propose a novel problem setting called Step-by-Step Layered Design Generation, which tasks a machine learning model with generating a design that adheres to a sequence of instructions from a designer. Leveraging recent advancements in multi-modal LLMs, we propose SLEDGE: Step-by-step LayEred Design GEnerator to model each update to a design as an atomic, layered change over its previous state, while being grounded in the instruction. To complement our new problem setting, we introduce a new evaluation suite, including a dataset and a benchmark. Our exhaustive experimental analysis and comparison with state-of-the-art approaches tailored to our new setup demonstrate the efficacy of our approach. We hope our work will attract attention to this pragmatic and under-explored research area.
>
---
#### [new 110] HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像修复中生成模型的幻觉问题，提出HalluGen框架，通过可控方式合成逼真幻觉图像。构建首个大规模标注幻觉数据集，用于评估与缓解幻觉，推动安全关键领域图像修复的可靠性。**

- **链接: [https://arxiv.org/pdf/2512.03345v1](https://arxiv.org/pdf/2512.03345v1)**

> **作者:** Seunghoi Kim; Henry F. J. Tregidgo; Chen Jin; Matteo Figini; Daniel C. Alexander
>
> **摘要:** Generative models are prone to hallucinations: plausible but incorrect structures absent in the ground truth. This issue is problematic in image restoration for safety-critical domains such as medical imaging, industrial inspection, and remote sensing, where such errors undermine reliability and trust. For example, in low-field MRI, widely used in resource-limited settings, restoration models are essential for enhancing low-quality scans, yet hallucinations can lead to serious diagnostic errors. Progress has been hindered by a circular dependency: evaluating hallucinations requires labeled data, yet such labels are costly and subjective. We introduce HalluGen, a diffusion-based framework that synthesizes realistic hallucinations with controllable type, location, and severity, producing perceptually realistic but semantically incorrect outputs (segmentation IoU drops from 0.86 to 0.36). Using HalluGen, we construct the first large-scale hallucination dataset comprising 4,350 annotated images derived from 1,450 brain MR images for low-field enhancement, enabling systematic evaluation of hallucination detection and mitigation. We demonstrate its utility in two applications: (1) benchmarking image quality metrics and developing Semantic Hallucination Assessment via Feature Evaluation (SHAFE), a feature-based metric with soft-attention pooling that improves hallucination sensitivity over traditional metrics; and (2) training reference-free hallucination detectors that generalize to real restoration failures. Together, HalluGen and its open dataset establish the first scalable foundation for evaluating hallucinations in safety-critical image restoration.
>
---
#### [new 111] DIQ-H: Evaluating Hallucination Persistence in VLMs Under Temporal Visual Degradation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型（VLMs）在动态视觉退化下的幻觉持续问题，提出DIQ-H基准。通过物理驱动的时序退化模拟，评估模型在连续视频流中的鲁棒性与错误恢复能力，引入不确定性引导的迭代精炼方法提升标注效率，揭示了主流VLMs在误差传播与时间一致性上的显著缺陷。**

- **链接: [https://arxiv.org/pdf/2512.03992v1](https://arxiv.org/pdf/2512.03992v1)**

> **作者:** Zexin Lin; Hawen Wan; Yebin Zhong; Xiaoqiang
>
> **摘要:** Vision-Language Models (VLMs) deployed in safety-critical applications such as autonomous driving must handle continuous visual streams under imperfect conditions. However, existing benchmarks focus on static, high-quality images and ignore temporal degradation and error propagation, which are critical failure modes where transient visual corruption induces hallucinations that persist across subsequent frames. We introduce DIQ-H, the first benchmark for evaluating VLM robustness under dynamic visual degradation in temporal sequences. DIQ-H applies physics-based corruptions including motion blur, sensor noise, and compression artifacts, and measures hallucination persistence, error recovery, and temporal consistency through multi-turn question-answering tasks. To enable scalable annotation, we propose Uncertainty-Guided Iterative Refinement (UIR), which generates reliable pseudo-ground-truth using lightweight VLMs with uncertainty filtering, achieving a 15.3 percent accuracy improvement. Experiments on 16 state-of-the-art VLMs reveal substantial robustness gaps: even advanced models such as GPT-4o achieve only a 78.5 percent recovery rate, while open-source models struggle with temporal consistency at less than 60 percent. DIQ-H provides a comprehensive platform for evaluating VLM reliability in real-world deployments.
>
---
#### [new 112] On the Temporality for Sketch Representation Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究草图表示学习中的时序性问题，探讨将草图视为序列的合理性及不同顺序的重要性。通过对比位置编码与解码器类型，发现绝对坐标优于相对坐标，非自回归解码更优，且时序重要性取决于顺序与任务。**

- **链接: [https://arxiv.org/pdf/2512.04007v1](https://arxiv.org/pdf/2512.04007v1)**

> **作者:** Marcelo Isaias de Moraes Junior; Moacir Antonelli Ponti
>
> **摘要:** Sketches are simple human hand-drawn abstractions of complex scenes and real-world objects. Although the field of sketch representation learning has advanced significantly, there is still a gap in understanding the true relevance of the temporal aspect to the quality of these representations. This work investigates whether it is indeed justifiable to treat sketches as sequences, as well as which internal orders play a more relevant role. The results indicate that, although the use of traditional positional encodings is valid for modeling sketches as sequences, absolute coordinates consistently outperform relative ones. Furthermore, non-autoregressive decoders outperform their autoregressive counterparts. Finally, the importance of temporality was shown to depend on both the order considered and the task evaluated.
>
---
#### [new 113] Fully Unsupervised Self-debiasing of Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对文本生成图像模型中的偏见问题，提出一种全无监督的测试时去偏方法SelfDebias。通过在图像编码器嵌入空间中识别语义簇，引导扩散过程最小化输出分布与均匀分布的KL散度，无需标注数据即可有效降低模型在人口统计和抽象概念上的偏见，同时保持图像质量。**

- **链接: [https://arxiv.org/pdf/2512.03749v1](https://arxiv.org/pdf/2512.03749v1)**

> **作者:** Korada Sri Vardhana; Shrikrishna Lolla; Soma Biswas
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Text-to-image (T2I) diffusion models have achieved widespread success due to their ability to generate high-resolution, photorealistic images. These models are trained on large-scale datasets, like LAION-5B, often scraped from the internet. However, since this data contains numerous biases, the models inherently learn and reproduce them, resulting in stereotypical outputs. We introduce SelfDebias, a fully unsupervised test-time debiasing method applicable to any diffusion model that uses a UNet as its noise predictor. SelfDebias identifies semantic clusters in an image encoder's embedding space and uses these clusters to guide the diffusion process during inference, minimizing the KL divergence between the output distribution and the uniform distribution. Unlike supervised approaches, SelfDebias does not require human-annotated datasets or external classifiers trained for each generated concept. Instead, it is designed to automatically identify semantic modes. Extensive experiments show that SelfDebias generalizes across prompts and diffusion model architectures, including both conditional and unconditional models. It not only effectively debiases images along key demographic dimensions while maintaining the visual fidelity of the generated images, but also more abstract concepts for which identifying biases is also challenging.
>
---
#### [new 114] Motion4D: Learning 3D-Consistent Motion and Semantics for 4D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文针对单目视频中动态场景的4D理解任务，解决2D基础模型缺乏3D一致性的问题。提出Motion4D框架，通过融合2D先验与4D高斯点云表示，结合局部与全局优化，实现运动与语义的3D一致建模，并引入自适应重采样与语义迭代优化，显著提升追踪、分割与新视角合成性能。**

- **链接: [https://arxiv.org/pdf/2512.03601v1](https://arxiv.org/pdf/2512.03601v1)**

> **作者:** Haoran Zhou; Gim Hee Lee
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Recent advancements in foundation models for 2D vision have substantially improved the analysis of dynamic scenes from monocular videos. However, despite their strong generalization capabilities, these models often lack 3D consistency, a fundamental requirement for understanding scene geometry and motion, thereby causing severe spatial misalignment and temporal flickering in complex 3D environments. In this paper, we present Motion4D, a novel framework that addresses these challenges by integrating 2D priors from foundation models into a unified 4D Gaussian Splatting representation. Our method features a two-part iterative optimization framework: 1) Sequential optimization, which updates motion and semantic fields in consecutive stages to maintain local consistency, and 2) Global optimization, which jointly refines all attributes for long-term coherence. To enhance motion accuracy, we introduce a 3D confidence map that dynamically adjusts the motion priors, and an adaptive resampling process that inserts new Gaussians into under-represented regions based on per-pixel RGB and semantic errors. Furthermore, we enhance semantic coherence through an iterative refinement process that resolves semantic inconsistencies by alternately optimizing the semantic fields and updating prompts of SAM2. Extensive evaluations demonstrate that our Motion4D significantly outperforms both 2D foundation models and existing 3D-based approaches across diverse scene understanding tasks, including point-based tracking, video object segmentation, and novel view synthesis. Our code is available at https://hrzhou2.github.io/motion4d-web/.
>
---
#### [new 115] Harnessing Hypergraphs in Geometric Deep Learning for 3D RNA Inverse Folding
- **分类: cs.CV**

- **简介: 该论文针对RNA逆折叠任务，旨在设计能形成特定二级结构的核酸序列。提出HyperRNA框架，利用超图建模高阶相互作用，结合编码器-解码器结构与注意力机制，在PDBBind和RNAsolo数据集上实现高效序列生成，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.03592v1](https://arxiv.org/pdf/2512.03592v1)**

> **作者:** Guang Yang; Lei Fan
>
> **摘要:** The RNA inverse folding problem, a key challenge in RNA design, involves identifying nucleotide sequences that can fold into desired secondary structures, which are critical for ensuring molecular stability and function. The inherent complexity of this task stems from the intricate relationship between sequence and structure, making it particularly challenging. In this paper, we propose a framework, named HyperRNA, a generative model with an encoder-decoder architecture that leverages hypergraphs to design RNA sequences. Specifically, our HyperRNA model consists of three main components: preprocessing, encoding and decoding. In the preprocessing stage, graph structures are constructed by extracting the atom coordinates of RNA backbone based on 3-bead coarse-grained representation. The encoding stage processes these graphs, capturing higher order dependencies and complex biomolecular interactions using an attention embedding module and a hypergraph-based encoder. Finally, the decoding stage generates the RNA sequence in an autoregressive manner. We conducted quantitative and qualitative experiments on the PDBBind and RNAsolo datasets to evaluate the inverse folding task for RNA sequence generation and RNA-protein complex sequence generation. The experimental results demonstrate that HyperRNA not only outperforms existing RNA design methods but also highlights the potential of leveraging hypergraphs in RNA engineering.
>
---
#### [new 116] MSG-Loc: Multi-Label Likelihood-based Semantic Graph Matching for Object-Level Global Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对未知物体类别与语义模糊下的全局定位问题，提出基于多标签似然的语义图匹配方法。通过多标签图表示捕捉语义上下文，利用上下文感知的似然传播增强对应关系，提升定位精度与鲁棒性，在真实与合成环境中验证了其有效性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.03522v1](https://arxiv.org/pdf/2512.03522v1)**

> **作者:** Gihyeon Lee; Jungwoo Lee; Juwon Kim; Young-Sik Shin; Younggun Cho
>
> **备注:** Accepted in IEEE Robotics and Automation Letters (2025)
>
> **摘要:** Robots are often required to localize in environments with unknown object classes and semantic ambiguity. However, when performing global localization using semantic objects, high semantic ambiguity intensifies object misclassification and increases the likelihood of incorrect associations, which in turn can cause significant errors in the estimated pose. Thus, in this letter, we propose a multi-label likelihood-based semantic graph matching framework for object-level global localization. The key idea is to exploit multi-label graph representations, rather than single-label alternatives, to capture and leverage the inherent semantic context of object observations. Based on these representations, our approach enhances semantic correspondence across graphs by combining the likelihood of each node with the maximum likelihood of its neighbors via context-aware likelihood propagation. For rigorous validation, data association and pose estimation performance are evaluated under both closed-set and open-set detection configurations. In addition, we demonstrate the scalability of our approach to large-vocabulary object categories in both real-world indoor scenes and synthetic environments.
>
---
#### [new 117] Cyclical Temporal Encoding and Hybrid Deep Ensembles for Multistep Energy Forecasting
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对短时多步电力负荷预测任务，提出融合周期性时间编码与混合LSTM-CNN集成模型的框架。通过正弦余弦编码保留日历特征周期性，构建LSTM、CNN与MLP元学习器的集成模型，有效捕捉长期季节性与短期局部模式，在真实数据上显著降低RMSE和MAE。**

- **链接: [https://arxiv.org/pdf/2512.03656v1](https://arxiv.org/pdf/2512.03656v1)**

> **作者:** Salim Khazem; Houssam Kanso
>
> **摘要:** Accurate electricity consumption forecasting is essential for demand management and smart grid operations. This paper introduces a unified deep learning framework that integrates cyclical temporal encoding with hybrid LSTM-CNN architectures to enhance multistep energy forecasting. We systematically transform calendar-based attributes using sine cosine encodings to preserve periodic structure and evaluate their predictive relevance through correlation analysis. To exploit both long-term seasonal effects and short-term local patterns, we employ an ensemble model composed of an LSTM, a CNN, and a meta-learner of MLP regressors specialized for each forecast horizon. Using a one year national consumption dataset, we conduct an extensive experimental study including ablation analyses with and without cyclical encodings and calendar features and comparisons with established baselines from the literature. Results demonstrate consistent improvements across all seven forecast horizons, with our hybrid model achieving lower RMSE and MAE than individual architectures and prior methods. These findings confirm the benefit of combining cyclical temporal representations with complementary deep learning structures. To our knowledge, this is the first work to jointly evaluate temporal encodings, calendar-based features, and hybrid ensemble architectures within a unified short-term energy forecasting framework.
>
---
#### [new 118] Energy-Efficient Federated Learning via Adaptive Encoder Freezing for MRI-to-CT Conversion: A Green AI-Guided Research
- **分类: cs.LG; cs.AI; cs.CV; cs.DC; physics.med-ph**

- **简介: 该论文针对联邦学习（FL）在医疗影像领域能耗高、资源需求大的问题，提出一种自适应编码器冻结策略，用于MRI-to-CT转换任务。通过动态冻结更新微小的编码器层，显著降低计算负荷与碳排放，同时保持模型性能，推动绿色AI在医疗公平中的应用。**

- **链接: [https://arxiv.org/pdf/2512.03054v1](https://arxiv.org/pdf/2512.03054v1)**

> **作者:** Ciro Benito Raggio; Lucia Migliorelli; Nils Skupien; Mathias Krohmer Zabaleta; Oliver Blanck; Francesco Cicone; Giuseppe Lucio Cascini; Paolo Zaffino; Maria Francesca Spadea
>
> **备注:** 22 pages, 13 figures
>
> **摘要:** Federated Learning (FL) holds the potential to advance equality in health by enabling diverse institutions to collaboratively train deep learning (DL) models, even with limited data. However, the significant resource requirements of FL often exclude centres with limited computational infrastructure, further widening existing healthcare disparities. To address this issue, we propose a Green AI-oriented adaptive layer-freezing strategy designed to reduce energy consumption and computational load while maintaining model performance. We tested our approach using different federated architectures for Magnetic Resonance Imaging (MRI)-to-Computed Tomography (CT) conversion. The proposed adaptive strategy optimises the federated training by selectively freezing the encoder weights based on the monitored relative difference of the encoder weights from round to round. A patience-based mechanism ensures that freezing only occurs when updates remain consistently minimal. The energy consumption and CO2eq emissions of the federation were tracked using the CodeCarbon library. Compared to equivalent non-frozen counterparts, our approach reduced training time, total energy consumption and CO2eq emissions by up to 23%. At the same time, the MRI-to-CT conversion performance was maintained, with only small variations in the Mean Absolute Error (MAE). Notably, for three out of the five evaluated architectures, no statistically significant differences were observed, while two architectures exhibited statistically significant improvements. Our work aligns with a research paradigm that promotes DL-based frameworks meeting clinical requirements while ensuring climatic, social, and economic sustainability. It lays the groundwork for novel FL evaluation frameworks, advancing privacy, equity and, more broadly, justice in AI-driven healthcare.
>
---
#### [new 119] Artificial Microsaccade Compensation: Stable Vision for an Ornithopter
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对尾翼式扑翼机因高频振动（12–20 Hz）导致视频抖动的问题，提出“人工微跳变补偿”方法。通过优化SO(3)空间的三维旋转，实时稳定视频，消除畸变，提升视觉质量。相比商用软件Adobe Premier Pro，本方法效果更优且可实时运行。**

- **链接: [https://arxiv.org/pdf/2512.03995v1](https://arxiv.org/pdf/2512.03995v1)**

> **作者:** Levi Burner; Guido de Croon; Yiannis Aloimonos
>
> **备注:** 29 pages, 5 figures, 2 tables, under review
>
> **摘要:** Animals with foveated vision, including humans, experience microsaccades, small, rapid eye movements that they are not aware of. Inspired by this phenomenon, we develop a method for "Artificial Microsaccade Compensation". It can stabilize video captured by a tailless ornithopter that has resisted attempts to use camera-based sensing because it shakes at 12-20 Hz. Our approach minimizes changes in image intensity by optimizing over 3D rotation represented in SO(3). This results in a stabilized video, computed in real time, suitable for human viewing, and free from distortion. When adapted to hold a fixed viewing orientation, up to occasional saccades, it can dramatically reduce inter-frame motion while also benefiting from an efficient recursive update. When compared to Adobe Premier Pro's warp stabilizer, which is widely regarded as the best commercial video stabilization software available, our method achieves higher quality results while also running in real time.
>
---
#### [new 120] Kaleidoscopic Scintillation Event Imaging
- **分类: physics.ins-det; cs.CV; eess.IV**

- **简介: 该论文针对高能粒子探测中光子数过少导致成像困难的问题，提出一种分形镜面结构的闪烁体（kaleidoscopic scintillator），通过镜像反射增强光收集并保留事件空间信息。结合单光子相机与机器视觉算法，实现对单个闪烁事件的高分辨率三维定位，提升辐射成像精度。**

- **链接: [https://arxiv.org/pdf/2512.03216v1](https://arxiv.org/pdf/2512.03216v1)**

> **作者:** Alex Bocchieri; John Mamish; David Appleyard; Andreas Velten
>
> **摘要:** Scintillators are transparent materials that interact with high-energy particles and emit visible light as a result. They are used in state of the art methods of measuring high-energy particles and radiation sources. Most existing methods use fast single-pixel detectors to detect and time scintillation events. Cameras provide spatial resolution but can only capture an average over many events, making it difficult to image the events associated with an individual particle. Emerging single-photon avalanche diode cameras combine speed and spatial resolution to enable capturing images of individual events. This allows us to use machine vision techniques to analyze events, enabling new types of detectors. The main challenge is the very low brightness of the events. Techniques have to work with a very limited number of photons. We propose a kaleidoscopic scintillator to increase light collection in a single-photon camera while preserving the event's spatial information. The kaleidoscopic geometry creates mirror reflections of the event in known locations for a given event location that are captured by the camera. We introduce theory for imaging an event in a kaleidoscopic scintillator and an algorithm to estimate the event's 3D position. We find that the kaleidoscopic scintillator design provides sufficient light collection to perform high-resolution event measurements for advanced radiation imaging techniques using a commercial CMOS single-photon camera. Code and data are available at https://github.com/bocchs/kaleidoscopic_scintillator.
>
---
#### [new 121] RoboScape-R: Unified Reward-Observation World Models for Generalizable Robotics Training via RL
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人泛化能力不足的问题，提出RoboScape-R框架，利用世界模型生成内生奖励信号，构建通用训练环境。通过统一的奖励-观测世界模型，克服传统RL依赖人工奖励的局限，显著提升策略在跨场景下的泛化性能，实验显示平均性能提升37.5%。**

- **链接: [https://arxiv.org/pdf/2512.03556v1](https://arxiv.org/pdf/2512.03556v1)**

> **作者:** Yinzhou Tang; Yu Shang; Yinuo Chen; Bingwen Wei; Xin Zhang; Shu'ang Yu; Liangzhi Shi; Chao Yu; Chen Gao; Wei Wu; Yong Li
>
> **摘要:** Achieving generalizable embodied policies remains a key challenge. Traditional policy learning paradigms, including both Imitation Learning (IL) and Reinforcement Learning (RL), struggle to cultivate generalizability across diverse scenarios. While IL policies often overfit to specific expert trajectories, RL suffers from the inherent lack of a unified and general reward signal necessary for effective multi-scene generalization. We posit that the world model is uniquely capable of serving as a universal environment proxy to address this limitation. However, current world models primarily focus on their ability to predict observations and still rely on task-specific, handcrafted reward functions, thereby failing to provide a truly general training environment. Toward this problem, we propose RoboScape-R, a framework leveraging the world model to serve as a versatile, general-purpose proxy for the embodied environment within the RL paradigm. We introduce a novel world model-based general reward mechanism that generates ''endogenous'' rewards derived from the model's intrinsic understanding of real-world state transition dynamics. Extensive experiments demonstrate that RoboScape-R effectively addresses the limitations of traditional RL methods by providing an efficient and general training environment that substantially enhances the generalization capability of embodied policies. Our approach offers critical insights into utilizing the world model as an online training strategy and achieves an average 37.5% performance improvement over baselines under out-of-domain scenarios.
>
---
#### [new 122] Radiance Meshes for Volumetric Reconstruction
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出辐射率网格（Radiance Meshes），用于体素重建。通过Delaunay四面体化构建常密度四面体单元，结合压缩式NeRF结构，实现高效、精确的体渲染。解决了传统方法在拓扑变化时的不连续问题，支持实时高质量视图合成，并可拓展至畸变校正、物理模拟等应用。**

- **链接: [https://arxiv.org/pdf/2512.04076v1](https://arxiv.org/pdf/2512.04076v1)**

> **作者:** Alexander Mai; Trevor Hedstrom; George Kopanas; Janne Kontkanen; Falko Kuester; Jonathan T. Barron
>
> **备注:** Website: half-potato.gitlab.io/rm
>
> **摘要:** We introduce radiance meshes, a technique for representing radiance fields with constant density tetrahedral cells produced with a Delaunay tetrahedralization. Unlike a Voronoi diagram, a Delaunay tetrahedralization yields simple triangles that are natively supported by existing hardware. As such, our model is able to perform exact and fast volume rendering using both rasterization and ray-tracing. We introduce a new rasterization method that achieves faster rendering speeds than all prior radiance field representations (assuming an equivalent number of primitives and resolution) across a variety of platforms. Optimizing the positions of Delaunay vertices introduces topological discontinuities (edge flips). To solve this, we use a Zip-NeRF-style backbone which allows us to express a smoothly varying field even when the topology changes. Our rendering method exactly evaluates the volume rendering equation and enables high quality, real-time view synthesis on standard consumer hardware. Our tetrahedral meshes also lend themselves to a variety of exciting applications including fisheye lens distortion, physics-based simulation, editing, and mesh extraction.
>
---
#### [new 123] LATTICE: Democratize High-Fidelity 3D Generation at Scale
- **分类: cs.GR; cs.CV**

- **简介: 该论文针对高保真3D生成中质量与可扩展性不足的问题，提出LATTICE框架。通过引入VoxSet半结构化表示，实现高效、位置感知的3D资产生成。采用两阶段流程，支持任意分辨率解码与灵活推理，显著提升生成质量与效率，推动大规模高质量3D内容生成。**

- **链接: [https://arxiv.org/pdf/2512.03052v1](https://arxiv.org/pdf/2512.03052v1)**

> **作者:** Zeqiang Lai; Yunfei Zhao; Zibo Zhao; Haolin Liu; Qingxiang Lin; Jingwei Huang; Chunchao Guo; Xiangyu Yue
>
> **备注:** Technical Report
>
> **摘要:** We present LATTICE, a new framework for high-fidelity 3D asset generation that bridges the quality and scalability gap between 3D and 2D generative models. While 2D image synthesis benefits from fixed spatial grids and well-established transformer architectures, 3D generation remains fundamentally more challenging due to the need to predict both spatial structure and detailed geometric surfaces from scratch. These challenges are exacerbated by the computational complexity of existing 3D representations and the lack of structured and scalable 3D asset encoding schemes. To address this, we propose VoxSet, a semi-structured representation that compresses 3D assets into a compact set of latent vectors anchored to a coarse voxel grid, enabling efficient and position-aware generation. VoxSet retains the simplicity and compression advantages of prior VecSet methods while introducing explicit structure into the latent space, allowing positional embeddings to guide generation and enabling strong token-level test-time scaling. Built upon this representation, LATTICE adopts a two-stage pipeline: first generating a sparse voxelized geometry anchor, then producing detailed geometry using a rectified flow transformer. Our method is simple at its core, but supports arbitrary resolution decoding, low-cost training, and flexible inference schemes, achieving state-of-the-art performance on various aspects, and offering a significant step toward scalable, high-quality 3D asset creation.
>
---
#### [new 124] Multi-Agent Reinforcement Learning and Real-Time Decision-Making in Robotic Soccer for Virtual Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究多智能体强化学习在虚拟机器人足球中的实时决策问题。针对任务复杂性与可扩展性挑战，提出分层强化学习结合平均场理论的框架，实现高效协作与策略优化，在4v4仿真中显著提升进球数、控球率与传球准确率。**

- **链接: [https://arxiv.org/pdf/2512.03166v1](https://arxiv.org/pdf/2512.03166v1)**

> **作者:** Aya Taourirte; Md Sohag Mia
>
> **摘要:** The deployment of multi-agent systems in dynamic, adversarial environments like robotic soccer necessitates real-time decision-making, sophisticated cooperation, and scalable algorithms to avoid the curse of dimensionality. While Reinforcement Learning (RL) offers a promising framework, existing methods often struggle with the multi-granularity of tasks (long-term strategy vs. instant actions) and the complexity of large-scale agent interactions. This paper presents a unified Multi-Agent Reinforcement Learning (MARL) framework that addresses these challenges. First, we establish a baseline using Proximal Policy Optimization (PPO) within a client-server architecture for real-time action scheduling, with PPO demonstrating superior performance (4.32 avg. goals, 82.9% ball control). Second, we introduce a Hierarchical RL (HRL) structure based on the options framework to decompose the problem into a high-level trajectory planning layer (modeled as a Semi-Markov Decision Process) and a low-level action execution layer, improving global strategy (avg. goals increased to 5.26). Finally, to ensure scalability, we integrate mean-field theory into the HRL framework, simplifying many-agent interactions into a single agent vs. the population average. Our mean-field actor-critic method achieves a significant performance boost (5.93 avg. goals, 89.1% ball control, 92.3% passing accuracy) and enhanced training stability. Extensive simulations of 4v4 matches in the Webots environment validate our approach, demonstrating its potential for robust, scalable, and cooperative behavior in complex multi-agent domains.
>
---
#### [new 125] Jina-VLM: Small Multilingual Vision Language Model
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Jina-VLM，一个2.4B参数的小型多语言视觉语言模型。针对开放领域2B级模型在多语言视觉问答任务中性能不足的问题，采用SigLIP2视觉编码器与Qwen3语言模型结合，通过注意力池化连接实现任意分辨率图像的高效处理，在多语言VQA任务上达到领先水平，同时保持优异的纯文本性能。**

- **链接: [https://arxiv.org/pdf/2512.04032v1](https://arxiv.org/pdf/2512.04032v1)**

> **作者:** Andreas Koukounas; Georgios Mastrapas; Florian Hönicke; Sedigheh Eslami; Guillaume Roncari; Scott Martens; Han Xiao
>
> **备注:** 18 pages, 1-7 main content
>
> **摘要:** We present Jina-VLM, a 2.4B parameter vision-language model that achieves state-of-the-art multilingual visual question answering among open 2B-scale VLMs. The model couples a SigLIP2 vision encoder with a Qwen3 language backbone through an attention-pooling connector that enables token-efficient processing of arbitrary-resolution images. Across standard VQA benchmarks and multilingual evaluations, Jina-VLM outperforms comparable models while preserving competitive text-only performance.
>
---
#### [new 126] What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦机器人3D场景表示任务，旨在回答“何种表示最佳”。系统综述点云、体素、NeRF、3DGS及基础模型等方法，分析其在感知、定位、导航等模块的优劣，探讨基础模型作为统一解决方案的潜力与挑战，为未来研究提供参考。**

- **链接: [https://arxiv.org/pdf/2512.03422v1](https://arxiv.org/pdf/2512.03422v1)**

> **作者:** Tianchen Deng; Yue Pan; Shenghai Yuan; Dong Li; Chen Wang; Mingrui Li; Long Chen; Lihua Xie; Danwei Wang; Jingchuan Wang; Javier Civera; Hesheng Wang; Weidong Chen
>
> **摘要:** In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models. While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance. Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence. In this paper, we categorized the core modules of robotics into five parts (Perception, Mapping, Localization, Navigation, Manipulation). We start by presenting the standard formulation of different scene representation methods and comparing the advantages and disadvantages of scene representation across different modules. This survey is centered around the question: What is the best 3D scene representation for robotics? We then discuss the future development trends of 3D scene representations, with a particular focus on how the 3D Foundation Model could replace current methods as the unified solution for future robotic applications. The remaining challenges in fully realizing this model are also explored. We aim to offer a valuable resource for both newcomers and experienced researchers to explore the future of 3D scene representations and their application in robotics. We have published an open-source project on GitHub and will continue to add new works and technologies to this project.
>
---
#### [new 127] M3DR: Towards Universal Multilingual Multimodal Document Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出M3DR框架，解决多语言多模态文档检索中英语主导的问题。通过合成多语言数据与对比学习，实现跨语言、跨模态的统一表征，支持多种模型架构与检索范式，在22种语言上实现显著性能提升，推动通用多语言多模态检索发展。**

- **链接: [https://arxiv.org/pdf/2512.03514v1](https://arxiv.org/pdf/2512.03514v1)**

> **作者:** Adithya S Kolavi; Vyoman Jain
>
> **摘要:** Multimodal document retrieval systems have shown strong progress in aligning visual and textual content for semantic search. However, most existing approaches remain heavily English-centric, limiting their effectiveness in multilingual contexts. In this work, we present M3DR (Multilingual Multimodal Document Retrieval), a framework designed to bridge this gap across languages, enabling applicability across diverse linguistic and cultural contexts. M3DR leverages synthetic multilingual document data and generalizes across different vision-language architectures and model sizes, enabling robust cross-lingual and cross-modal alignment. Using contrastive training, our models learn unified representations for text and document images that transfer effectively across languages. We validate this capability on 22 typologically diverse languages, demonstrating consistent performance and adaptability across linguistic and script variations. We further introduce a comprehensive benchmark that captures real-world multilingual scenarios, evaluating models under monolingual, multilingual, and mixed-language settings. M3DR generalizes across both single dense vector and ColBERT-style token-level multi-vector retrieval paradigms. Our models, NetraEmbed and ColNetraEmbed achieve state-of-the-art performance with ~150% relative improvements on cross-lingual retrieval.
>
---
#### [new 128] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer
- **分类: q-bio.GN; cs.AI; cs.CV**

- **简介: 该论文针对癌症单细胞数据建模中表示学习效率与评估标准缺失的问题，提出轻量级混合模型PanFoMa，融合Transformer与状态空间模型优势，实现高效精准的转录组建模。同时构建大规模基准PanFoMaBench，验证其在细胞类型注释、批次整合等任务上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.03111v1](https://arxiv.org/pdf/2512.03111v1)**

> **作者:** Xiaoshui Huang; Tianlin Zhu; Yifan Zuo; Xue Xia; Zonghan Wu; Jiebin Yan; Dingli Hua; Zongyi Xu; Yuming Fang; Jian Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Single-cell RNA sequencing (scRNA-seq) is essential for decoding tumor heterogeneity. However, pan-cancer research still faces two key challenges: learning discriminative and efficient single-cell representations, and establishing a comprehensive evaluation benchmark. In this paper, we introduce PanFoMa, a lightweight hybrid neural network that combines the strengths of Transformers and state-space models to achieve a balance between performance and efficiency. PanFoMa consists of a front-end local-context encoder with shared self-attention layers to capture complex, order-independent gene interactions; and a back-end global sequential feature decoder that efficiently integrates global context using a linear-time state-space model. This modular design preserves the expressive power of Transformers while leveraging the scalability of Mamba to enable transcriptome modeling, effectively capturing both local and global regulatory signals. To enable robust evaluation, we also construct a large-scale pan-cancer single-cell benchmark, PanFoMaBench, containing over 3.5 million high-quality cells across 33 cancer subtypes, curated through a rigorous preprocessing pipeline. Experimental results show that PanFoMa outperforms state-of-the-art models on our pan-cancer benchmark (+4.0\%) and across multiple public tasks, including cell type annotation (+7.4\%), batch integration (+4.0\%) and multi-omics integration (+3.1\%). The code is available at https://github.com/Xiaoshui-Huang/PanFoMa.
>
---
#### [new 129] Culture Affordance Atlas: Reconciling Object Diversity Through Functional Mapping
- **分类: cs.CY; cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言模型中文化与经济偏见问题，提出功能中心的“文化可及性图谱”框架。通过重构Dollar Street数据集，以46种功能重新标注288类物品，显著缩小高低收入群体间性能差距，提升模型对非西方、低收入语境的适应性，推动更公平的AI发展。**

- **链接: [https://arxiv.org/pdf/2512.03173v1](https://arxiv.org/pdf/2512.03173v1)**

> **作者:** Joan Nwatu; Longju Bai; Oana Ignat; Rada Mihalcea
>
> **摘要:** Culture shapes the objects people use and for what purposes, yet mainstream Vision-Language (VL) datasets frequently exhibit cultural biases, disproportionately favoring higher-income, Western contexts. This imbalance reduces model generalizability and perpetuates performance disparities, especially impacting lower-income and non-Western communities. To address these disparities, we propose a novel function-centric framework that categorizes objects by the functions they fulfill, across diverse cultural and economic contexts. We implement this framework by creating the Culture Affordance Atlas, a re-annotated and culturally grounded restructuring of the Dollar Street dataset spanning 46 functions and 288 objects publicly available at https://lit.eecs.umich.edu/CultureAffordance-Atlas/index.html. Through extensive empirical analyses using the CLIP model, we demonstrate that function-centric labels substantially reduce socioeconomic performance gaps between high- and low-income groups by a median of 6 pp (statistically significant), improving model effectiveness for lower-income contexts. Furthermore, our analyses reveals numerous culturally essential objects that are frequently overlooked in prominent VL datasets. Our contributions offer a scalable pathway toward building inclusive VL datasets and equitable AI systems.
>
---
#### [new 130] Tada-DIP: Input-adaptive Deep Image Prior for One-shot 3D Image Reconstruction
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文针对3D图像重建任务，解决深度图像先验（DIP）在3D场景中易过拟合、应用受限的问题。提出Tada-DIP方法，通过输入自适应与去噪正则化，实现高质量、无需训练数据的单次3D重建，在稀疏视角CT重建中表现优异，媲美监督学习模型。**

- **链接: [https://arxiv.org/pdf/2512.03962v1](https://arxiv.org/pdf/2512.03962v1)**

> **作者:** Evan Bell; Shijun Liang; Ismail Alkhouri; Saiprasad Ravishankar
>
> **备注:** 6 pages, 8 figures, 2025 Asilomar Conference on Signals, Systems, and Computers. Code is available at github.com/evanbell02/Tada-DIP/
>
> **摘要:** Deep Image Prior (DIP) has recently emerged as a promising one-shot neural-network based image reconstruction method. However, DIP has seen limited application to 3D image reconstruction problems. In this work, we introduce Tada-DIP, a highly effective and fully 3D DIP method for solving 3D inverse problems. By combining input-adaptation and denoising regularization, Tada-DIP produces high-quality 3D reconstructions while avoiding the overfitting phenomenon that is common in DIP. Experiments on sparse-view X-ray computed tomography reconstruction validate the effectiveness of the proposed method, demonstrating that Tada-DIP produces much better reconstructions than training-data-free baselines and achieves reconstruction performance on par with a supervised network trained using a large dataset with fully-sampled volumes.
>
---
## 更新

#### [replaced 001] SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08710v2](https://arxiv.org/pdf/2506.08710v2)**

> **作者:** Mengjiao Ma; Qi Ma; Yue Li; Jiahuan Cheng; Runyi Yang; Bin Ren; Nikola Popovic; Mingqiang Wei; Nicu Sebe; Luc Van Gool; Theo Gevers; Martin R. Oswald; Danda Pani Paudel
>
> **备注:** 15 pages, codes, data and benchmark are released
>
> **摘要:** 3D Gaussian Splatting (3DGS) serves as a highly performant and efficient encoding of scene geometry, appearance, and semantics. Moreover, grounding language in 3D scenes has proven to be an effective strategy for 3D scene understanding. Current Language Gaussian Splatting line of work fall into three main groups: (i) per-scene optimization-based, (ii) per-scene optimization-free, and (iii) generalizable approach. However, most of them are evaluated only on rendered 2D views of a handful of scenes and viewpoints close to the training views, limiting ability and insight into holistic 3D understanding. To address this gap, we propose the first large-scale benchmark that systematically assesses these three groups of methods directly in 3D space, evaluating on 1060 scenes across three indoor datasets and one outdoor dataset. Benchmark results demonstrate a clear advantage of the generalizable paradigm, particularly in relaxing the scene-specific limitation, enabling fast feed-forward inference on novel scenes, and achieving superior segmentation performance. We further introduce GaussianWorld-49K a carefully curated 3DGS dataset comprising around 49K diverse indoor and outdoor scenes obtained from multiple sources, with which we demonstrate the generalizable approach could harness strong data priors. Our codes, benchmark, and datasets are public to accelerate research in generalizable 3DGS scene understanding.
>
---
#### [replaced 002] LargeAD: Large-Scale Cross-Sensor Data Pretraining for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文针对自动驾驶中3D场景理解不足的问题，提出LargeAD框架，通过视觉基础模型生成语义超像素并与LiDAR点云对齐，实现跨模态预训练。工作包括超像素生成、对比学习、时序一致性保持及多源数据训练，显著提升分割与检测性能。**

- **链接: [https://arxiv.org/pdf/2501.04005v3](https://arxiv.org/pdf/2501.04005v3)**

> **作者:** Lingdong Kong; Xiang Xu; Youquan Liu; Jun Cen; Runnan Chen; Wenwei Zhang; Liang Pan; Kai Chen; Ziwei Liu
>
> **备注:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Recent advancements in vision foundation models (VFMs) have revolutionized visual perception in 2D, yet their potential for 3D scene understanding, particularly in autonomous driving applications, remains underexplored. In this paper, we introduce LargeAD, a versatile and scalable framework designed for large-scale 3D pretraining across diverse real-world driving datasets. Our framework leverages VFMs to extract semantically rich superpixels from 2D images, which are aligned with LiDAR point clouds to generate high-quality contrastive samples. This alignment facilitates cross-modal representation learning, enhancing the semantic consistency between 2D and 3D data. We introduce several key innovations: (i) VFM-driven superpixel generation for detailed semantic representation, (ii) a VFM-assisted contrastive learning strategy to align multimodal features, (iii) superpoint temporal consistency to maintain stable representations across time, and (iv) multi-source data pretraining to generalize across various LiDAR configurations. Our approach achieves substantial gains over state-of-the-art methods in linear probing and fine-tuning for LiDAR-based segmentation and object detection. Extensive experiments on 11 large-scale multi-sensor datasets highlight our superior performance, demonstrating adaptability, efficiency, and robustness in real-world autonomous driving scenarios.
>
---
#### [replaced 003] GS4: Generalizable Sparse Splatting Semantic SLAM
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.06517v3](https://arxiv.org/pdf/2506.06517v3)**

> **作者:** Mingqi Jiang; Chanho Kim; Chen Ziwen; Li Fuxin
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Traditional SLAM algorithms excel at camera tracking, but typically produce incomplete and low-resolution maps that are not tightly integrated with semantics prediction. Recent work integrates Gaussian Splatting (GS) into SLAM to enable dense, photorealistic 3D mapping, yet existing GS-based SLAM methods require per-scene optimization that is slow and consumes an excessive number of Gaussians. We present GS4, the first generalizable GS-based semantic SLAM system. Compared with prior approaches, GS4 runs 10x faster, uses 10x fewer Gaussians, and achieves state-of-the-art performance across color, depth, semantic mapping and camera tracking. From an RGB-D video stream, GS4 incrementally builds and updates a set of 3D Gaussians using a feed-forward network. First, the Gaussian Prediction Model estimates a sparse set of Gaussian parameters from input frame, which integrates both color and semantic prediction with the same backbone. Then, the Gaussian Refinement Network merges new Gaussians with the existing set while avoiding redundancy. Finally, when significant pose changes are detected, we perform only 1-5 iterations of joint Gaussian-pose optimization to correct drift, remove floaters, and further improve tracking accuracy. Experiments on the real-world ScanNet and ScanNet++ benchmarks demonstrate state-of-the-art semantic SLAM performance, with strong generalization capability shown through zero-shot transfer to the NYUv2 and TUM RGB-D datasets.
>
---
#### [replaced 004] MindGPT-4ov: An Enhanced MLLM via a Multi-Stage Post-Training Paradigm
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02895v2](https://arxiv.org/pdf/2512.02895v2)**

> **作者:** Wei Chen; Chaoqun Du; Feng Gu; Wei He; Qizhen Li; Zide Liu; Xuhao Pan; Chang Ren; Xudong Rao; Chenfeng Wang; Tao Wei; Chengjun Yu; Pengfei Yu; Yufei Zheng; Chunpeng Zhou; Pan Zhou; Xuhan Zhu
>
> **备注:** 33 pages, 14 figures
>
> **摘要:** We present MindGPT-4ov, a multimodal large language model (MLLM) that introduces a general post-training paradigm spanning data production, model training, and efficient deployment. It achieves state-of-the-art performance across multiple benchmarks at low cost, effectively enhancing the foundational capabilities of MLLMs and the generalization ability. Focusing on data construction, supervised fine-tuning strategies, and multimodal reinforcement learning methods, this work proposes three key innovations: (1) An information density-based data generation scheme, integrated with a dual-dimensional tree-structured label system, enabling automated generation of high-quality cross-domain data. (2) A collaborative curriculum supervised fine-tuning approach that balances the injection of domain-specific knowledge with the preservation of general capabilities. (3) A hybrid reinforcement learning paradigm that enhances reasoning ability while simultaneously addressing multi-objective optimization such as diversity exploration, maintenance of multimodal perception, and response conciseness. Moreover, we implement a series of infrastructure optimizations, such as 5D parallel training, operator optimization, and inference quantization to enhance training and inference efficiency while reducing the cost of domain adaptation. Experimental results demonstrate that the MindGPT-4ov model outperforms state-of-the-art models on benchmarks such as MMBench, MMStar, MathVision, and MathVista. In addition, MindGPT-4ov also demonstrates superior user experience in vertical domain tasks, enabling a seamless transition from academic research to industrial deployment. MindGPT-4ov provides a general post-training paradigm applicable to a wide range of MLLMs. The model weights, datasets, and code for the Qwen3-VL-based variants will be recently open-sourced to support the community's development of MLLMs.
>
---
#### [replaced 005] Pan-LUT: Efficient Pan-sharpening via Learnable Look-Up Tables
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.23793v2](https://arxiv.org/pdf/2503.23793v2)**

> **作者:** Zhongnan Cai; Yingying Wang; Hui Zheng; Panwang Pan; ZiXu Lin; Ge Meng; Chenxin Li; Chunming He; Jiaxin Xie; Yunlong Lin; Junbin Lu; Yue Huang; Xinghao Ding
>
> **摘要:** Recently, deep learning-based pan-sharpening algorithms have achieved notable advancements over traditional methods. However, deep learning-based methods incur substantial computational overhead during inference, especially with large images. This excessive computational demand limits the applicability of these methods in real-world scenarios, particularly in the absence of dedicated computing devices such as GPUs and TPUs. To address these challenges, we propose Pan-LUT, a novel learnable look-up table (LUT) framework for pan-sharpening that strikes a balance between performance and computational efficiency for large remote sensing images. Our method makes it possible to process 15K*15K remote sensing images on a 24GB GPU. To finely control the spectral transformation, we devise the PAN-guided look-up table (PGLUT) for channel-wise spectral mapping. To effectively capture fine-grained spatial details, we introduce the spatial details look-up table (SDLUT). Furthermore, to adaptively aggregate channel information for generating high-resolution multispectral images, we design an adaptive output look-up table (AOLUT). Our model contains fewer than 700K parameters and processes a 9K*9K image in under 1 ms using one RTX 2080 Ti GPU, demonstrating significantly faster performance compared to other methods. Experiments reveal that Pan-LUT efficiently processes large remote sensing images in a lightweight manner, bridging the gap to real-world applications. Furthermore, our model surpasses SOTA methods in full-resolution scenes under real-world conditions, highlighting its effectiveness and efficiency.
>
---
#### [replaced 006] Assessing the Alignment of Popular CNNs to the Brain for Valence Appraisal
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21384v2](https://arxiv.org/pdf/2509.21384v2)**

> **作者:** Laurent Mertens; Elahe' Yargholi; Laura Van Hove; Hans Op de Beeck; Jan Van den Stock; Joost Vennekens
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Convolutional Neural Networks (CNNs) are a popular type of computer model that have proven their worth in many computer vision tasks. Moreover, they form an interesting study object for the field of psychology, with shown correspondences between the workings of CNNs and the human brain. However, these correspondences have so far mostly been studied in the context of general visual perception. In contrast, this paper explores to what extent this correspondence also holds for a more complex brain process, namely social cognition. To this end, we assess the alignment between popular CNN architectures and both human behavioral and fMRI data for image valence appraisal through a correlation analysis. We show that for this task CNNs struggle to go beyond simple visual processing, and do not seem to reflect higher-order brain processing. Furthermore, we present Object2Brain, a novel framework that combines GradCAM and object detection at the CNN-filter level with the aforementioned correlation analysis to study the influence of different object classes on the CNN-to-human correlations. Despite similar correlation trends, different CNN architectures are shown to display different object class sensitivities.
>
---
#### [replaced 007] PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference
- **分类: cs.CV; cs.AI; cs.PF**

- **链接: [https://arxiv.org/pdf/2405.14430v4](https://arxiv.org/pdf/2405.14430v4)**

> **作者:** Jiarui Fang; Jinzhe Pan; Aoyu Li; Xibo Sun; Jiannan Wang
>
> **摘要:** This paper presents PipeFusion, an innovative parallel methodology to tackle the high latency issues associated with generating high-resolution images using diffusion transformers (DiTs) models. PipeFusion partitions images into patches and the model layers across multiple GPUs. It employs a patch-level pipeline parallel strategy to orchestrate communication and computation efficiently. By capitalizing on the high similarity between inputs from successive diffusion steps, PipeFusion reuses one-step stale feature maps to provide context for the current pipeline step. This approach notably reduces communication costs compared to existing DiTs inference parallelism, including tensor parallel, sequence parallel and DistriFusion. PipeFusion enhances memory efficiency through parameter distribution across devices, ideal for large DiTs like Flux.1. Experimental results demonstrate that PipeFusion achieves state-of-the-art performance on 8$\times$L40 PCIe GPUs for Pixart, Stable-Diffusion 3, and Flux.1 models. Our source code is available at https://github.com/xdit-project/xDiT.
>
---
#### [replaced 008] You Point, I Learn: Online Adaptation of Interactive Segmentation Models for Handling Distribution Shifts in Medical Imaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.06717v2](https://arxiv.org/pdf/2503.06717v2)**

> **作者:** Wentian Xu; Ziyun Liang; Harry Anthony; Yasin Ibrahim; Felix Cohen; Guang Yang; Konstantinos Kamnitsas
>
> **摘要:** Interactive segmentation uses real-time user inputs, such as mouse clicks, to iteratively refine model predictions. Although not originally designed to address distribution shifts, this paradigm naturally lends itself to such challenges. In medical imaging, where distribution shifts are common, interactive methods can use user inputs to guide models towards improved predictions. Moreover, once a model is deployed, user corrections can be used to adapt the network parameters to the new data distribution, mitigating distribution shift. Based on these insights, we aim to develop a practical, effective method for improving the adaptive capabilities of interactive segmentation models to new data distributions in medical imaging. Firstly, we found that strengthening the model's responsiveness to clicks is important for the initial training process. Moreover, we show that by treating the post-interaction user-refined model output as pseudo-ground-truth, we can design a lean, practical online adaptation method that enables a model to learn effectively across sequential test images. The framework includes two components: (i) a Post-Interaction adaptation process, updating the model after the user has completed interactive refinement of an image, and (ii) a Mid-Interaction adaptation process, updating incrementally after each click. Both processes include a Click-Centered Gaussian loss that strengthens the model's reaction to clicks and enhances focus on user-guided, clinically relevant regions. Experiments on 5 fundus and 4 brain-MRI databases show that our approach consistently outperforms existing methods under diverse distribution shifts, including unseen imaging modalities and pathologies. Code and pretrained models will be released upon publication.
>
---
#### [replaced 009] LoVoRA: Text-guided and Mask-free Video Object Removal and Addition with Learnable Object-aware Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02933v2](https://arxiv.org/pdf/2512.02933v2)**

> **作者:** Zhihan Xiao; Lin Liu; Yixin Gao; Xiaopeng Zhang; Haoxuan Che; Songping Mai; Qi Tian
>
> **摘要:** Text-guided video editing, particularly for object removal and addition, remains a challenging task due to the need for precise spatial and temporal consistency. Existing methods often rely on auxiliary masks or reference images for editing guidance, which limits their scalability and generalization. To address these issues, we propose LoVoRA, a novel framework for mask-free video object removal and addition using object-aware localization mechanism. Our approach utilizes a unique dataset construction pipeline that integrates image-to-video translation, optical flow-based mask propagation, and video inpainting, enabling temporally consistent edits. The core innovation of LoVoRA is its learnable object-aware localization mechanism, which provides dense spatio-temporal supervision for both object insertion and removal tasks. By leveraging a Diffusion Mask Predictor, LoVoRA achieves end-to-end video editing without requiring external control signals during inference. Extensive experiments and human evaluation demonstrate the effectiveness and high-quality performance of LoVoRA. https://cz-5f.github.io/LoVoRA.github.io
>
---
#### [replaced 010] Sat2Flow: A Structure-Aware Diffusion Framework for Human Flow Generation from Satellite Imagery
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.19499v2](https://arxiv.org/pdf/2508.19499v2)**

> **作者:** Xiangxu Wang; Tianhong Zhao; Wei Tu; Bowen Zhang; Guanzhou Chen; Jinzhou Cao
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Origin-Destination (OD) flow matrices are critical for urban mobility analysis, supporting traffic forecasting, infrastructure planning, and policy design. Existing methods face two key limitations: (1) reliance on costly auxiliary features (e.g., Points of Interest, socioeconomic statistics) with limited spatial coverage, and (2) fragility to spatial topology changes, where reordering urban regions disrupts the structural coherence of generated flows. We propose Sat2Flow, a structure-aware diffusion framework that generates structurally coherent OD flows using only satellite imagery. Our approach employs a multi-kernel encoder to capture diverse regional interactions and a permutation-aware diffusion process that maintains consistency across regional orderings. Through joint contrastive training linking satellite features with OD patterns and equivariant diffusion training enforcing structural invariance, Sat2Flow ensures topological robustness under arbitrary regional reindexing. Experiments on real-world datasets show that Sat2Flow outperforms physics-based and data-driven baselines in accuracy while preserving flow distributions and spatial structures under index permutations. Sat2Flow offers a globally scalable solution for OD flow generation in data-scarce environments, eliminating region-specific auxiliary data dependencies while maintaining structural robustness for reliable mobility modeling.
>
---
#### [replaced 011] STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13186v4](https://arxiv.org/pdf/2510.13186v4)**

> **作者:** Zhen Li; Xibin Jin; Guoliang Li; Shuai Wang; Miaowen Wen; Huseyin Arslan; Derrick Wing Kwan Ng; Chengzhong Xu
>
> **摘要:** Edge Gaussian splatting (EGS), which aggregates data from distributed clients (e.g., drones) and trains a global GS model at the edge (e.g., ground server), is an emerging paradigm for scene reconstruction in low-altitude economy. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead. Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments reveal that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. The GS-oriented objective can be accurately predicted with low sampling ratios (e.g., 10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
>
---
#### [replaced 012] Efficient Transferable Optimal Transport via Min-Sliced Transport Plans
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19741v2](https://arxiv.org/pdf/2511.19741v2)**

> **作者:** Xinran Liu; Elaheh Akbari; Rocio Diaz Martin; Navid NaderiAlizadeh; Soheil Kolouri
>
> **摘要:** Optimal Transport (OT) offers a powerful framework for finding correspondences between distributions and addressing matching and alignment problems in various areas of computer vision, including shape analysis, image generation, and multimodal tasks. The computation cost of OT, however, hinders its scalability. Slice-based transport plans have recently shown promise for reducing the computational cost by leveraging the closed-form solutions of 1D OT problems. These methods optimize a one-dimensional projection (slice) to obtain a conditional transport plan that minimizes the transport cost in the ambient space. While efficient, these methods leave open the question of whether learned optimal slicers can transfer to new distribution pairs under distributional shift. Understanding this transferability is crucial in settings with evolving data or repeated OT computations across closely related distributions. In this paper, we study the min-Sliced Transport Plan (min-STP) framework and investigate the transferability of optimized slicers: can a slicer trained on one distribution pair yield effective transport plans for new, unseen pairs? Theoretically, we show that optimized slicers remain close under slight perturbations of the data distributions, enabling efficient transfer across related tasks. To further improve scalability, we introduce a minibatch formulation of min-STP and provide statistical guarantees on its accuracy. Empirically, we demonstrate that the transferable min-STP achieves strong one-shot matching performance and facilitates amortized training for point cloud alignment and flow-based generative modeling.
>
---
#### [replaced 013] Score Distillation of Flow Matching Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.25127v2](https://arxiv.org/pdf/2509.25127v2)**

> **作者:** Mingyuan Zhou; Yi Gu; Huangjie Zheng; Liangchen Song; Guande He; Yizhe Zhang; Wenze Hu; Yinfei Yang
>
> **摘要:** Diffusion models achieve high-quality image generation but are limited by slow iterative sampling. Distillation methods alleviate this by enabling one- or few-step generation. Flow matching, originally introduced as a distinct framework, has since been shown to be theoretically equivalent to diffusion under Gaussian assumptions, raising the question of whether distillation techniques such as score distillation transfer directly. We provide a simple derivation -- based on Bayes' rule and conditional expectations -- that unifies Gaussian diffusion and flow matching without relying on ODE/SDE formulations. Building on this view, we extend Score identity Distillation (SiD) to pretrained text-to-image flow-matching models, including SANA, SD3-Medium, SD3.5-Medium/Large, and FLUX.1-dev, all with DiT backbones. Experiments show that, with only modest flow-matching- and DiT-specific adjustments, SiD works out of the box across these models, in both data-free and data-aided settings, without requiring teacher finetuning or architectural changes. This provides the first systematic evidence that score distillation applies broadly to text-to-image flow matching models, resolving prior concerns about stability and soundness and unifying acceleration techniques across diffusion- and flow-based generators. A project page is available at https://yigu1008.github.io/SiD-DiT.
>
---
#### [replaced 014] IW-Bench: Evaluating Large Multimodal Models for Converting Image-to-Web
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对大模型图像到网页转换任务，提出IW-Bench基准。解决现有评估方法忽略隐式元素与布局信息的问题，创新性引入元素准确率与布局准确率，并设计五跳多模态思维链提示，全面评估模型生成网页的完整性与布局准确性。**

- **链接: [https://arxiv.org/pdf/2409.18980v2](https://arxiv.org/pdf/2409.18980v2)**

> **作者:** Hongcheng Guo; Wei Zhang; Junhao Chen; Yaonan Gu; Jian Yang; Junjia Du; Shaosheng Cao; Binyuan Hui; Tianyu Liu; Jianxin Ma; Chang Zhou; Zhoujun Li
>
> **摘要:** Recently advancements in large multimodal models have led to significant strides in image comprehension capabilities. Despite these advancements, there is a lack of the robust benchmark specifically for assessing the Image-to-Web conversion proficiency of these large models. Primarily, it is essential to ensure the integrity of the web elements generated. These elements comprise visible and invisible categories. Previous evaluation methods (e.g.,BLEU) are notably susceptible to significant alterations due to the presence of invisible elements in Web. Furthermore, it is crucial to measure the layout information of web pages, referring to the positional relationships between elements, which is overlooked by previous work. To address challenges, we have curated and aligned a benchmark of images and corresponding web codes (IW-BENCH). Specifically, we propose the Element Accuracy, which tests the completeness of the elements by parsing the Document Object Model (DOM) tree. Layout Accuracy is also proposed to analyze the positional relationships of elements by converting DOM tree into a common subsequence. Besides, we design a five-hop multimodal Chain-of-Thought Prompting for better performance, which contains five hop: 1) SoM prompt injection. 2) Inferring Elements. 3) Inferring Layout. 4) Inferring Web code. 5) Reflection. Our benchmark comprises 1200 pairs of images and web codes with varying levels of difficulty. We have conducted extensive experiments on existing large multimodal models, offering insights into their performance and areas for improvement in image-to-web domain.
>
---
#### [replaced 015] TransUNet-GradCAM: A Hybrid Transformer-U-Net with Self-Attention and Explainable Visualizations for Foot Ulcer Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03758v3](https://arxiv.org/pdf/2508.03758v3)**

> **作者:** Akwasi Asare; Mary Sagoe; Justice Williams Asare; Stephen Edward Moore
>
> **摘要:** Automated segmentation of diabetic foot ulcers (DFUs) plays a critical role in clinical diagnosis, therapeutic planning, and longitudinal wound monitoring. However, this task remains challenging due to the heterogeneous appearance, irregular morphology, and complex backgrounds associated with ulcer regions in clinical photographs. Traditional convolutional neural networks (CNNs), such as U-Net, provide strong localization capabilities but struggle to model long-range spatial dependencies due to their inherently limited receptive fields. To address this, we employ the TransUNet architecture, a hybrid framework that integrates the global attention mechanism of Vision Transformers (ViTs) into the U-Net structure. This combination allows the model to extract global contextual features while maintaining fine-grained spatial resolution. We trained the model on the public Foot Ulcer Segmentation Challenge (FUSeg) dataset using a robust augmentation pipeline and a hybrid loss function to mitigate class imbalance. On the validation set, the model achieved a Dice Similarity Coefficient (F1-score) of 0.8799 using an optimized threshold of 0.4389. To ensure clinical transparency, we integrated Grad-CAM visualizations to highlight model focus areas. Furthermore, a clinical utility analysis demonstrated a strong correlation (Pearson r = 0.9631) between predicted and ground-truth wound areas. These outcomes demonstrate that our approach effectively integrates global and local feature extraction, offering a reliable, effective, and explainable solution for automated foot ulcer assessment.
>
---
#### [replaced 016] Vision-Based Mistake Analysis in Procedural Activities: A Review of Advances and Challenges
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.19292v2](https://arxiv.org/pdf/2510.19292v2)**

> **作者:** Konstantinos Bacharidis; Antonis A. Argyros
>
> **备注:** 23pages, 6 figures, 2 tables
>
> **摘要:** Mistake analysis in procedural activities is a critical area of research with applications spanning industrial automation, physical rehabilitation, education and human-robot collaboration. This paper reviews vision-based methods for detecting and predicting mistakes in structured tasks, focusing on procedural and executional errors. By leveraging advancements in computer vision, including action recognition, anticipation and activity understanding, vision-based systems can identify deviations in task execution, such as incorrect sequencing, use of improper techniques, or timing errors. We explore the challenges posed by intra-class variability, viewpoint differences and compositional activity structures, which complicate mistake detection. Additionally, we provide a comprehensive overview of existing datasets, evaluation metrics and state-of-the-art methods, categorizing approaches based on their use of procedural structure, supervision levels and learning strategies. Open challenges, such as distinguishing permissible variations from true mistakes and modeling error propagation are discussed alongside future directions, including neuro-symbolic reasoning and counterfactual state modeling. This work aims to establish a unified perspective on vision-based mistake analysis in procedural activities, highlighting its potential to enhance safety, efficiency and task performance across diverse domains.
>
---
#### [replaced 017] D$^{2}$-VPR: A Parameter-efficient Visual-foundation-model-based Visual Place Recognition Method via Knowledge Distillation and Deformable Aggregation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12528v2](https://arxiv.org/pdf/2511.12528v2)**

> **作者:** Zheyuan Zhang; Jiwei Zhang; Boyu Zhou; Linzhimeng Duan; Hong Chen
>
> **摘要:** Visual Place Recognition (VPR) aims to determine the geographic location of a query image by retrieving its most visually similar counterpart from a geo-tagged reference database. Recently, the emergence of the powerful visual foundation model, DINOv2, trained in a self-supervised manner on massive datasets, has significantly improved VPR performance. This improvement stems from DINOv2's exceptional feature generalization capabilities but is often accompanied by increased model complexity and computational overhead that impede deployment on resource-constrained devices. To address this challenge, we propose $D^{2}$-VPR, a $D$istillation- and $D$eformable-based framework that retains the strong feature extraction capabilities of visual foundation models while significantly reducing model parameters and achieving a more favorable performance-efficiency trade-off. Specifically, first, we employ a two-stage training strategy that integrates knowledge distillation and fine-tuning. Additionally, we introduce a Distillation Recovery Module (DRM) to better align the feature spaces between the teacher and student models, thereby minimizing knowledge transfer losses to the greatest extent possible. Second, we design a Top-Down-attention-based Deformable Aggregator (TDDA) that leverages global semantic features to dynamically and adaptively adjust the Regions of Interest (ROI) used for aggregation, thereby improving adaptability to irregular structures. Extensive experiments demonstrate that our method achieves competitive performance compared to state-of-the-art approaches. Meanwhile, it reduces the parameter count by approximately 64.2% and FLOPs by about 62.6% (compared to CricaVPR).Code is available at https://github.com/tony19980810/D2VPR.
>
---
#### [replaced 018] Beyond Top Activations: Efficient and Reliable Crowdsourced Evaluation of Automated Interpretability
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.07985v2](https://arxiv.org/pdf/2506.07985v2)**

> **作者:** Tuomas Oikarinen; Ge Yan; Akshay Kulkarni; Tsui-Wei Weng
>
> **摘要:** Interpreting individual neurons or directions in activation space is an important topic in mechanistic interpretability. Numerous automated interpretability methods have been proposed to generate such explanations, but it remains unclear how reliable these explanations are, and which methods produce the most accurate descriptions. While crowd-sourced evaluations are commonly used, existing pipelines are noisy, costly, and typically assess only the highest-activating inputs, leading to unreliable results. In this paper, we introduce two techniques to enable cost-effective and accurate crowdsourced evaluation of automated interpretability methods beyond top activating inputs. First, we propose Model-Guided Importance Sampling (MG-IS) to select the most informative inputs to show human raters. In our experiments, we show this reduces the number of inputs needed to reach the same evaluation accuracy by ~13x. Second, we address label noise in crowd-sourced ratings through Bayesian Rating Aggregation (BRAgg), which allows us to reduce the number of ratings per input required to overcome noise by ~3x. Together, these techniques reduce the evaluation cost by ~40x, making large-scale evaluation feasible. Finally, we use our methods to conduct a large scale crowd-sourced study comparing recent automated interpretability methods for vision networks.
>
---
#### [replaced 019] \textit{ViRectify}: A Challenging Benchmark for Video Reasoning Correction with Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01424v2](https://arxiv.org/pdf/2512.01424v2)**

> **作者:** Xusen Hei; Jiali Chen; Jinyu Yang; Mengchen Zhao; Yi Cai
>
> **备注:** 22 pages, 11 figures
>
> **摘要:** As multimodal large language models (MLLMs) frequently exhibit errors in complex video reasoning scenarios, correcting these errors is critical for uncovering their weaknesses and improving performance. However, existing benchmarks lack systematic evaluation of MLLMs' ability to identify and correct these video reasoning errors. To bridge this gap, we propose \textit{ViRectify}, a comprehensive benchmark to evaluate their fine-grained correction capability. Through an AI-assisted annotation pipeline with human verification, we construct a dataset of over 30\textit{K} instances spanning dynamic perception, scientific reasoning, and embodied decision-making domains. In \textit{ViRectify}, we challenge MLLMs to perform step-wise error identification and generate rationales with key video evidence grounding. In addition, we further propose the trajectory evidence-driven correction framework, comprising step-wise error trajectory and reward modeling on visual evidence-grounded correction. It encourages the model to explicitly concentrate on error propagation and key timestamps for correction. Extensive evaluation across 16 advanced MLLMs demonstrates that our \textit{ViRectify} serves as a challenging testbed, where GPT-5 achieves only 31.94\% correction accuracy. Our framework enables a Qwen2.5-VL-7B to consistently outperform the variants of 72B on \textit{ViRectify}, showing the effectiveness of our approach. Further analysis uncovers systematic asymmetries in error correction across models, and our dataset is also a valuable data resource to perform reflection learning. We believe \textit{ViRectify} provides a new direction for comprehensively evaluating the advanced MLLMs in video reasoning.
>
---
#### [replaced 020] OneThinker: All-in-one Reasoning Model for Image and Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03043v2](https://arxiv.org/pdf/2512.03043v2)**

> **作者:** Kaituo Feng; Manyuan Zhang; Hongyu Li; Kaixuan Fan; Shuang Chen; Yilei Jiang; Dian Zheng; Peiwen Sun; Yiyuan Zhang; Haoze Sun; Yan Feng; Peng Pei; Xunliang Cai; Xiangyu Yue
>
> **备注:** Project page: https://github.com/tulerfeng/OneThinker
>
> **摘要:** Reinforcement learning (RL) has recently achieved remarkable success in eliciting visual reasoning within Multimodal Large Language Models (MLLMs). However, existing approaches typically train separate models for different tasks and treat image and video reasoning as disjoint domains. This results in limited scalability toward a multimodal reasoning generalist, which restricts practical versatility and hinders potential knowledge sharing across tasks and modalities. To this end, we propose OneThinker, an all-in-one reasoning model that unifies image and video understanding across diverse fundamental visual tasks, including question answering, captioning, spatial and temporal grounding, tracking, and segmentation. To achieve this, we construct the OneThinker-600k training corpus covering all these tasks and employ commercial models for CoT annotation, resulting in OneThinker-SFT-340k for SFT cold start. Furthermore, we propose EMA-GRPO to handle reward heterogeneity in multi-task RL by tracking task-wise moving averages of reward standard deviations for balanced optimization. Extensive experiments on diverse visual benchmarks show that OneThinker delivers strong performance on 31 benchmarks, across 10 fundamental visual understanding tasks. Moreover, it exhibits effective knowledge transfer between certain tasks and preliminary zero-shot generalization ability, marking a step toward a unified multimodal reasoning generalist. All code, model, and data are released.
>
---
#### [replaced 021] DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DynamicCity，解决城市场景生成中动态性与大规模4D语义建模问题。通过可变自编码器与DiT扩散模型，构建高效HexPlane表示，实现高精度、大尺度动态4D场景生成，支持多种条件驱动应用，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2410.18084v3](https://arxiv.org/pdf/2410.18084v3)**

> **作者:** Hengwei Bian; Lingdong Kong; Haozhe Xie; Liang Pan; Yu Qiao; Ziwei Liu
>
> **备注:** ICLR 2025 Spotlight; 35 pages, 18 figures, 15 tables; Project Page at https://dynamic-city.github.io/
>
> **摘要:** Urban scene generation has been developing rapidly recently. However, existing methods primarily focus on generating static and single-frame scenes, overlooking the inherently dynamic nature of real-world driving environments. In this work, we introduce DynamicCity, a novel 4D occupancy generation framework capable of generating large-scale, high-quality dynamic 4D scenes with semantics. DynamicCity mainly consists of two key models. 1) A VAE model for learning HexPlane as the compact 4D representation. Instead of using naive averaging operations, DynamicCity employs a novel Projection Module to effectively compress 4D features into six 2D feature maps for HexPlane construction, which significantly enhances HexPlane fitting quality (up to 12.56 mIoU gain). Furthermore, we utilize an Expansion & Squeeze Strategy to reconstruct 3D feature volumes in parallel, which improves both network training efficiency and reconstruction accuracy than naively querying each 3D point (up to 7.05 mIoU gain, 2.06x training speedup, and 70.84% memory reduction). 2) A DiT-based diffusion model for HexPlane generation. To make HexPlane feasible for DiT generation, a Padded Rollout Operation is proposed to reorganize all six feature planes of the HexPlane as a squared 2D feature map. In particular, various conditions could be introduced in the diffusion or sampling process, supporting versatile 4D generation applications, such as trajectory- and command-driven generation, inpainting, and layout-conditioned generation. Extensive experiments on the CarlaSC and Waymo datasets demonstrate that DynamicCity significantly outperforms existing state-of-the-art 4D occupancy generation methods across multiple metrics. The code and models have been released to facilitate future research.
>
---
#### [replaced 022] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文针对多语言、多条件交织的语义检索任务，提出首个相关数据集MERIT与新框架Coral。针对现有模型忽视查询中具体条件的问题，通过嵌入重建与对比学习，提升细粒度条件保留与全局语义理解能力，显著提升检索性能。**

- **链接: [https://arxiv.org/pdf/2506.03144v3](https://arxiv.org/pdf/2506.03144v3)**

> **作者:** Wei Chow; Yuan Gao; Linfeng Li; Xian Wang; Qi Xu; Hang Song; Lingdong Kong; Ran Zhou; Yi Zeng; Yidong Cai; Botian Jiang; Shilin Xu; Jiajun Zhang; Minghui Qiu; Xiangtai Li; Tianshu Yang; Siliang Tang; Juncheng Li
>
> **备注:** NeurIPS 2025; Project Page, Code, and Dataset at: https://merit-2025.github.io/
>
> **摘要:** Semantic retrieval is crucial for modern applications yet remains underexplored in current research. Existing datasets are limited to single languages, single images, or singular retrieval conditions, often failing to fully exploit the expressive capacity of visual information as evidenced by maintained performance when images are replaced with captions. However, practical retrieval scenarios frequently involve interleaved multi-condition queries with multiple images. Hence, this paper introduces MERIT, the first multilingual dataset for interleaved multi-condition semantic retrieval, comprising 320,000 queries with 135,000 products in 5 languages, covering 7 distinct product categories. Extensive experiments on MERIT identify existing models's limitation: focusing solely on global semantic information while neglecting specific conditional elements in queries. Consequently, we propose Coral, a novel fine-tuning framework that adapts pre-trained MLLMs by integrating embedding reconstruction to preserve fine-grained conditional elements and contrastive learning to extract comprehensive global semantics. Experiments demonstrate that Coral achieves a 45.9% performance improvement over conventional approaches on MERIT, with strong generalization capabilities validated across 8 established retrieval benchmarks. Collectively, our contributions - a novel dataset, identification of critical limitations in existing approaches, and an innovative fine-tuning framework - establish a foundation for future research in interleaved multi-condition semantic retrieval.
>
---
#### [replaced 023] MACS: Measurement-Aware Consistency Sampling for Inverse Problems
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.02208v2](https://arxiv.org/pdf/2510.02208v2)**

> **作者:** Amirreza Tanevardi; Pooria Abbas Rad Moghadam; Seyed Mohammad Eshtehardian; Sajjad Amini; Babak Khalaj
>
> **备注:** 10 pages, 4 figures, This work has been submitted to the IEEE for possible publication
>
> **摘要:** Diffusion models have emerged as powerful generative priors for solving inverse imaging problems. However, their practical deployment is hindered by the substantial computational cost of slow, multi-step sampling. Although Consistency Models (CMs) address this limitation by enabling high-quality generation in only one or a few steps, their direct application to inverse problems has remained largely unexplored. This paper introduces a modified consistency sampling framework specifically designed for inverse problems. The proposed approach regulates the sampler's stochasticity through a measurement-consistency mechanism that leverages the degradation operator, thereby enforcing fidelity to the observed data while preserving the computational efficiency of consistency-based generation. Comprehensive experiments on the Fashion-MNIST and LSUN Bedroom datasets demonstrate consistent improvements across both perceptual and pixel-level metrics, including the Fréchet Inception Distance (FID), Kernel Inception Distance (KID), peak signal-to-noise ratio (PSNR), and structural similarity index measure (SSIM), compared with baseline consistency and diffusion-based sampling methods. The proposed method achieves competitive or superior reconstruction quality with only a small number of sampling steps.
>
---
#### [replaced 024] DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文针对自动驾驶中多传感器语义感知的鲁棒性问题，提出深度引导的融合方法DGFusion。通过引入激光雷达提供的深度信息，构建深度感知特征，动态调整跨模态融合策略，提升在复杂场景下的分割性能。**

- **链接: [https://arxiv.org/pdf/2509.09828v2](https://arxiv.org/pdf/2509.09828v2)**

> **作者:** Tim Broedermannn; Christos Sakaridis; Luigi Piccinelli; Wim Abbeloos; Luc Van Gool
>
> **备注:** Code and models will be available at https://github.com/timbroed/DGFusion
>
> **摘要:** Robust semantic perception for autonomous vehicles relies on effectively combining multiple sensors with complementary strengths and weaknesses. State-of-the-art sensor fusion approaches to semantic perception often treat sensor data uniformly across the spatial extent of the input, which hinders performance when faced with challenging conditions. By contrast, we propose a novel depth-guided multimodal fusion method that upgrades condition-aware fusion by integrating depth information. Our network, DGFusion, poses multimodal segmentation as a multi-task problem, utilizing the lidar measurements, which are typically available in outdoor sensor suites, both as one of the model's inputs and as ground truth for learning depth. Our corresponding auxiliary depth head helps to learn depth-aware features, which are encoded into spatially varying local depth tokens that condition our attentive cross-modal fusion. Together with a global condition token, these local depth tokens dynamically adapt sensor fusion to the spatially varying reliability of each sensor across the scene, which largely depends on depth. In addition, we propose a robust loss for our depth, which is essential for learning from lidar inputs that are typically sparse and noisy in adverse conditions. Our method achieves state-of-the-art panoptic and semantic segmentation performance on the challenging MUSES and DeLiVER datasets. Code and models will be available at https://github.com/timbroed/DGFusion
>
---
#### [replaced 025] Automatic Labelling for Low-Light Pedestrian Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.02513v2](https://arxiv.org/pdf/2507.02513v2)**

> **作者:** Dimitrios Bouzoulas; Eerik Alamikkotervo; Risto Ojala
>
> **摘要:** Pedestrian detection in RGB images is a key task in pedestrian safety, as the most common sensor in autonomous vehicles and advanced driver assistance systems is the RGB camera. A challenge in RGB pedestrian detection, that does not appear to have large public datasets, is low-light conditions. As a solution, in this research, we propose an automated infrared-RGB labeling pipeline. The proposed pipeline consists of 1) Infrared detection, where a fine-tuned model for infrared pedestrian detection is used 2) Label transfer process from the infrared detections to their RGB counterparts 3) Training object detection models using the generated labels for low-light RGB pedestrian detection. The research was performed using the KAIST dataset. For the evaluation, object detection models were trained on the generated autolabels and ground truth labels. When compared on a previously unseen image sequence, the results showed that the models trained on generated labels outperformed the ones trained on ground-truth labels in 6 out of 9 cases for the mAP@50 and mAP@50-95 metrics. The source code for this research is available at https://github.com/BouzoulasDimitrios/IR-RGB-Automated-LowLight-Pedestrian-Labeling
>
---
#### [replaced 026] SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control
- **分类: cs.GR; cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对物理角色控制中运动先验复用性差的问题，提出可重用的Score-Matching Motion Priors（SMP）。通过预训练运动扩散模型与分数蒸馏采样，构建任务无关的通用运动先验，可冻结复用于多种下游任务，实现风格迁移与组合，生成高质量自然动作。**

- **链接: [https://arxiv.org/pdf/2512.03028v2](https://arxiv.org/pdf/2512.03028v2)**

> **作者:** Yuxuan Mu; Ziyu Zhang; Yi Shi; Minami Matsumoto; Kotaro Imamura; Guy Tevet; Chuan Guo; Michael Taylor; Chang Shu; Pengcheng Xi; Xue Bin Peng
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Data-driven motion priors that can guide agents toward producing naturalistic behaviors play a pivotal role in creating life-like virtual characters. Adversarial imitation learning has been a highly effective method for learning motion priors from reference motion data. However, adversarial priors, with few exceptions, need to be retrained for each new controller, thereby limiting their reusability and necessitating the retention of the reference motion data when training on downstream tasks. In this work, we present Score-Matching Motion Priors (SMP), which leverages pre-trained motion diffusion models and score distillation sampling (SDS) to create reusable task-agnostic motion priors. SMPs can be pre-trained on a motion dataset, independent of any control policy or task. Once trained, SMPs can be kept frozen and reused as general-purpose reward functions to train policies to produce naturalistic behaviors for downstream tasks. We show that a general motion prior trained on large-scale datasets can be repurposed into a variety of style-specific priors. Furthermore SMP can compose different styles to synthesize new styles not present in the original dataset. Our method produces high-quality motion comparable to state-of-the-art adversarial imitation learning methods through reusable and modular motion priors. We demonstrate the effectiveness of SMP across a diverse suite of control tasks with physically simulated humanoid characters. Video demo available at https://youtu.be/ravlZJteS20
>
---
#### [replaced 027] Test-time Correction: An Online 3D Detection System via Visual Prompting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.07768v3](https://arxiv.org/pdf/2412.07768v3)**

> **作者:** Hanxue Zhang; Zetong Yang; Yanan Sun; Li Chen; Fei Xia; Fatma Güney; Hongyang Li
>
> **摘要:** This paper introduces Test-time Correction (TTC), an online 3D detection system designed to rectify test-time errors using various auxiliary feedback, aiming to enhance the safety of deployed autonomous driving systems. Unlike conventional offline 3D detectors that remain fixed during inference, TTC enables immediate online error correction without retraining, allowing autonomous vehicles to adapt to new scenarios and reduce deployment risks. To achieve this, we equip existing 3D detectors with an Online Adapter (OA) module -- a prompt-driven query generator for real-time correction. At the core of OA module are visual prompts: image-based descriptions of objects of interest derived from auxiliary feedback such as mismatches with 2D detections, road descriptions, or user clicks. These visual prompts, collected from risky objects during inference, are maintained in a visual prompt buffer to enable continuous correction in future frames. By leveraging this mechanism, TTC consistently detects risky objects, achieving reliable, adaptive, and versatile driving autonomy. Extensive experiments show that TTC significantly improves instant error rectification over frozen 3D detectors, even under limited labels, zero-shot settings, and adverse conditions. We hope this work inspires future research on post-deployment online rectification systems for autonomous driving.
>
---
#### [replaced 028] ActiveInitSplat: How Active Image Selection Helps Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.06859v2](https://arxiv.org/pdf/2503.06859v2)**

> **作者:** Konstantinos D. Polyzos; Athanasios Bacharis; Saketh Madhuvarasu; Nikos Papanikolopoulos; Tara Javidi
>
> **摘要:** Gaussian splatting (GS) along with its extensions and variants provides outstanding performance in real-time scene rendering while meeting reduced storage demands and computational efficiency. While the selection of 2D images capturing the scene of interest is crucial for the proper initialization and training of GS, hence markedly affecting the rendering performance, prior works rely on passively and typically densely selected 2D images. In contrast, this paper proposes `ActiveInitSplat', a novel framework for active selection of training images for proper initialization and training of GS. ActiveInitSplat relies on density and occupancy criteria of the resultant 3D scene representation from the selected 2D images, to ensure that the latter are captured from diverse viewpoints leading to better scene coverage and that the initialized Gaussian functions are well aligned with the actual 3D structure. Numerical tests on well-known simulated and real environments demonstrate the merits of ActiveInitSplat resulting in significant GS rendering performance improvement over passive GS baselines in both dense- and sparse-view settings, in the widely adopted LPIPS, SSIM, and PSNR metrics.
>
---
#### [replaced 029] Exploring the Potentials of Spiking Neural Networks for Image Deraining
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02258v2](https://arxiv.org/pdf/2512.02258v2)**

> **作者:** Shuang Chen; Tomas Krajnik; Farshad Arvin; Amir Atapour-Abarghouei
>
> **备注:** Accepted By AAAI2026
>
> **摘要:** Biologically plausible and energy-efficient frameworks such as Spiking Neural Networks (SNNs) have not been sufficiently explored in low-level vision tasks. Taking image deraining as an example, this study addresses the representation of the inherent high-pass characteristics of spiking neurons, specifically in image deraining and innovatively proposes the Visual LIF (VLIF) neuron, overcoming the obstacle of lacking spatial contextual understanding present in traditional spiking neurons. To tackle the limitation of frequency-domain saturation inherent in conventional spiking neurons, we leverage the proposed VLIF to introduce the Spiking Decomposition and Enhancement Module and the lightweight Spiking Multi-scale Unit for hierarchical multi-scale representation learning. Extensive experiments across five benchmark deraining datasets demonstrate that our approach significantly outperforms state-of-the-art SNN-based deraining methods, achieving this superior performance with only 13\% of their energy consumption. These findings establish a solid foundation for deploying SNNs in high-performance, energy-efficient low-level vision tasks.
>
---
#### [replaced 030] Differentiable, Bit-shifting, and Scalable Quantization without training neural network from scratch
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [https://arxiv.org/pdf/2510.16088v4](https://arxiv.org/pdf/2510.16088v4)**

> **作者:** Zia Badar
>
> **摘要:** Quantization of neural networks provides benefits of inference in less compute and memory requirements. Previous work in quantization lack two important aspects which this work provides. First almost all previous work in quantization used a non-differentiable approach and for learning; the derivative is usually set manually in backpropogation which make the learning ability of algorithm questionable, our approach is not just differentiable, we also provide proof of convergence of our approach to the optimal neural network. Second previous work in shift/logrithmic quantization either have avoided activation quantization along with weight quantization or achieved less accuracy. Learning logrithmic quantize values of form $2^n$ requires the quantization function can scale to more than 1 bit quantization which is another benifit of our quantization that it provides $n$ bits quantization as well. Our approach when tested with image classification task using imagenet dataset, resnet18 and weight quantization only achieves less than 1 percent accuracy compared to full precision accuracy while taking only 15 epochs to train using shift bit quantization and achieves comparable to SOTA approaches accuracy in both weight and activation quantization using shift bit quantization in 15 training epochs with slightly higher(only higher cpu instructions) inference cost compared to 1 bit quantization(without logrithmic quantization) and not requiring any higher precision multiplication.
>
---
#### [replaced 031] SpecGen: Neural Spectral BRDF Generation via Spectral-Spatial Tri-plane Aggregation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.17316v2](https://arxiv.org/pdf/2508.17316v2)**

> **作者:** Zhenyu Jin; Wenjie Li; Zhanyu Ma; Heng Guo
>
> **摘要:** Synthesizing spectral images across different wavelengths is essential for photorealistic rendering. Unlike conventional spectral uplifting methods that convert RGB images into spectral ones, we introduce SpecGen, a novel method that generates spectral bidirectional reflectance distribution functions (BRDFs) from a single RGB image of a sphere. This enables spectral image rendering under arbitrary illuminations and shapes covered by the corresponding material. A key challenge in spectral BRDF generation is the scarcity of measured spectral BRDF data. To address this, we propose the Spectral-Spatial Tri-plane Aggregation (SSTA) network, which models reflectance responses across wavelengths and incident-outgoing directions, allowing the training strategy to leverage abundant RGB BRDF data to enhance spectral BRDF generation. Experiments show that our method accurately reconstructs spectral BRDFs from limited spectral data and surpasses state-of-the-art methods in hyperspectral image reconstruction, achieving an improvement of 8 dB in PSNR. Codes and data will be released upon acceptance.
>
---
#### [replaced 032] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.15122v4](https://arxiv.org/pdf/2504.15122v4)**

> **作者:** Minh-Quan Viet Bui; Jongmin Park; Juan Luis Gonzalez Bello; Jaeho Moon; Jihyong Oh; Munchurl Kim
>
> **备注:** This paper has been accepted to AAAI 2026. The first two authors contributed equally to this work (equal contribution). The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/mobgs-site/
>
> **摘要:** We present MoBGS, a novel motion deblurring 3D Gaussian Splatting (3DGS) framework capable of reconstructing sharp and high-quality novel spatio-temporal views from blurry monocular videos in an end-to-end manner. Existing dynamic novel view synthesis (NVS) methods are highly sensitive to motion blur in casually captured videos, resulting in significant degradation of rendering quality. While recent approaches address motion-blurred inputs for NVS, they primarily focus on static scene reconstruction and lack dedicated motion modeling for dynamic objects. To overcome these limitations, our MoBGS introduces a novel Blur-adaptive Latent Camera Estimation (BLCE) method using a proposed Blur-adaptive Neural Ordinary Differential Equation (ODE) solver for effective latent camera trajectory estimation, improving global camera motion deblurring. In addition, we propose a Latent Camera-induced Exposure Estimation (LCEE) method to ensure consistent deblurring of both a global camera and local object motions. Extensive experiments on the Stereo Blur dataset and real-world blurry videos show that our MoBGS significantly outperforms the very recent methods, achieving state-of-the-art performance for dynamic NVS under motion blur.
>
---
#### [replaced 033] SDPose: Exploiting Diffusion Priors for Out-of-Domain and Robust Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24980v2](https://arxiv.org/pdf/2509.24980v2)**

> **作者:** Shuang Liang; Jing He; Chuanmeizhi Wang; Lejun Liao; Guo Zhang; Yingcong Chen; Yuan Yuan
>
> **备注:** 20 pages, 10 figures, 7 tables
>
> **摘要:** Pre-trained diffusion models provide rich multi-scale latent features and are emerging as powerful vision backbones. While recent works such as Marigold and Lotus adapt diffusion priors for dense prediction with strong cross-domain generalization, their potential for structured outputs remains underexplored. In this paper, we propose SDPose, a fine-tuning framework built upon Stable Diffusion to fully exploit pre-trained diffusion priors for human pose estimation. First, rather than modifying cross-attention modules or introducing learnable embeddings, we directly predict keypoint heatmaps in the SD U-Net's image latent space to preserve the original generative priors. Second, we map these latent features into keypoint heatmaps through a lightweight convolutional pose head, which avoids disrupting the pre-trained backbone. Finally, to prevent overfitting and enhance out-of-distribution robustness, we incorporate an auxiliary RGB reconstruction branch that preserves domain-transferable generative semantics. To evaluate robustness under domain shift, we further construct COCO-OOD, a style-transferred variant of COCO with preserved annotations. With just one-fifth of the training schedule used by Sapiens on COCO, SDPose attains parity with Sapiens-1B/2B on the COCO validation set and establishes a new state of the art on the cross-domain benchmarks HumanArt and COCO-OOD. Extensive ablations highlight the importance of diffusion priors, RGB reconstruction, and multi-scale SD U-Net features for cross-domain generalization, and t-SNE analyses further explain SD's domain-invariant latent structure. We also show that SDPose serves as an effective zero-shot pose annotator for controllable image and video generation.
>
---
#### [replaced 034] Rethinking the Learning Paradigm for Facial Expression Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2209.15402v4](https://arxiv.org/pdf/2209.15402v4)**

> **作者:** Weijie Wang; Bo Li; Nicu Sebe; Bruno Lepri
>
> **摘要:** Due to the subjective crowdsourcing annotations and the inherent inter-class similarity of facial expressions, the real-world Facial Expression Recognition (FER) datasets usually exhibit ambiguous annotation. To simplify the learning paradigm, most previous methods convert ambiguous annotation results into precise one-hot annotations and train FER models in an end-to-end supervised manner. In this paper, we rethink the existing training paradigm and propose that it is better to use weakly supervised strategies to train FER models with original ambiguous annotation.
>
---
#### [replaced 035] Flow to the Mode: Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.11056v3](https://arxiv.org/pdf/2503.11056v3)**

> **作者:** Kyle Sargent; Kyle Hsu; Justin Johnson; Li Fei-Fei; Jiajun Wu
>
> **备注:** ICCV 2025, 19 pages
>
> **摘要:** Since the advent of popular visual generation frameworks like VQGAN and latent diffusion models, state-of-the-art image generation systems have generally been two-stage systems that first tokenize or compress visual data into a lower-dimensional latent space before learning a generative model. Tokenizer training typically follows a standard recipe in which images are compressed and reconstructed subject to a combination of MSE, perceptual, and adversarial losses. Diffusion autoencoders have been proposed in prior work as a way to learn end-to-end perceptually-oriented image compression, but have not yet shown state-of-the-art performance on the competitive task of ImageNet-1K reconstruction. We propose FlowMo, a transformer-based diffusion autoencoder that achieves a new state-of-the-art for image tokenization at multiple compression rates without using convolutions, adversarial losses, spatially-aligned two-dimensional latent codes, or distilling from other tokenizers. Our key insight is that FlowMo training should be broken into a mode-matching pre-training stage and a mode-seeking post-training stage. In addition, we conduct extensive analyses and explore the training of generative models atop the FlowMo tokenizer. Our code and models will be available at http://kylesargent.github.io/flowmo .
>
---
#### [replaced 036] Generative Action Tell-Tales: Assessing Human Motion in Synthesized Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01803v2](https://arxiv.org/pdf/2512.01803v2)**

> **作者:** Xavier Thomas; Youngsun Lim; Ananya Srinivasan; Audrey Zheng; Deepti Ghadiyaram
>
> **摘要:** Despite rapid advances in video generative models, robust metrics for evaluating visual and temporal correctness of complex human actions remain elusive. Critically, existing pure-vision encoders and Multimodal Large Language Models (MLLMs) are strongly appearance-biased, lack temporal understanding, and thus struggle to discern intricate motion dynamics and anatomical implausibilities in generated videos. We tackle this gap by introducing a novel evaluation metric derived from a learned latent space of real-world human actions. Our method first captures the nuances, constraints, and temporal smoothness of real-world motion by fusing appearance-agnostic human skeletal geometry features with appearance-based features. We posit that this combined feature space provides a robust representation of action plausibility. Given a generated video, our metric quantifies its action quality by measuring the distance between its underlying representations and this learned real-world action distribution. For rigorous validation, we develop a new multi-faceted benchmark specifically designed to probe temporally challenging aspects of human action fidelity. Through extensive experiments, we show that our metric achieves substantial improvement of more than 68% compared to existing state-of-the-art methods on our benchmark, performs competitively on established external benchmarks, and has a stronger correlation with human perception. Our in-depth analysis reveals critical limitations in current video generative models and establishes a new standard for advanced research in video generation.
>
---
#### [replaced 037] Multilingual Training-Free Remote Sensing Image Captioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00887v2](https://arxiv.org/pdf/2512.00887v2)**

> **作者:** Carlos Rebelo; Gil Rocha; João Daniel Silva; Bruno Martins
>
> **摘要:** Remote sensing image captioning has advanced rapidly through encoder--decoder models, although the reliance on large annotated datasets and the focus on English restricts global applicability. To address these limitations, we propose the first training-free multilingual approach, based on retrieval-augmented prompting. For a given aerial image, we employ a domain-adapted SigLIP2 encoder to retrieve related captions and few-shot examples from a datastore, which are then provided to a language model. We explore two variants: an image-blind setup, where a multilingual Large Language Model (LLM) generates the caption from textual prompts alone, and an image-aware setup, where a Vision--Language Model (VLM) jointly processes the prompt and the input image. To improve the coherence of the retrieved content, we introduce a graph-based re-ranking strategy using PageRank on a graph of images and captions. Experiments on four benchmark datasets across ten languages demonstrate that our approach is competitive with fully supervised English-only systems and generalizes to other languages. Results also highlight the importance of re-ranking with PageRank, yielding up to 35% improvements in performance metrics. Additionally, it was observed that while VLMs tend to generate visually grounded but lexically diverse captions, LLMs can achieve stronger BLEU and CIDEr scores. Lastly, directly generating captions in the target language consistently outperforms other translation-based strategies. Overall, our work delivers one of the first systematic evaluations of multilingual, training-free captioning for remote sensing imagery, advancing toward more inclusive and scalable multimodal Earth observation systems.
>
---
#### [replaced 038] Accuracy-Robustness Trade Off via Spiking Neural Network Gradient Sparsity Trail
- **分类: cs.NE; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23762v3](https://arxiv.org/pdf/2509.23762v3)**

> **作者:** Luu Trong Nhan; Luu Trung Duong; Pham Ngoc Nam; Truong Cong Thang
>
> **备注:** Work under peer-review
>
> **摘要:** Spiking Neural Networks (SNNs) have attracted growing interest in both computational neuroscience and artificial intelligence, primarily due to their inherent energy efficiency and compact memory footprint. However, achieving adversarial robustness in SNNs, (particularly for vision-related tasks) remains a nascent and underexplored challenge. Recent studies have proposed leveraging sparse gradients as a form of regularization to enhance robustness against adversarial perturbations. In this work, we present a surprising finding: under specific architectural configurations, SNNs exhibit natural gradient sparsity and can achieve state-of-the-art adversarial defense performance without the need for any explicit regularization. Further analysis reveals a trade-off between robustness and generalization: while sparse gradients contribute to improved adversarial resilience, they can impair the model's ability to generalize; conversely, denser gradients support better generalization but increase vulnerability to attacks. Our findings offer new insights into the dual role of gradient sparsity in SNN training.
>
---
#### [replaced 039] BitMark: Watermarking Bitwise Autoregressive Image Generative Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.21209v2](https://arxiv.org/pdf/2506.21209v2)**

> **作者:** Louis Kerner; Michel Meintz; Bihe Zhao; Franziska Boenisch; Adam Dziedzic
>
> **备注:** Accepted as a Conference Paper at NeurIPS 2025
>
> **摘要:** State-of-the-art text-to-image models generate photorealistic images at an unprecedented speed. This work focuses on models that operate in a bitwise autoregressive manner over a discrete set of tokens that is practically infinite in size. However, their impressive generative power comes with a growing risk: as their outputs increasingly populate the Internet, they are likely to be scraped and reused as training data-potentially by the very same models. This phenomenon has been shown to lead to model collapse, where repeated training on generated content, especially from the models' own previous versions, causes a gradual degradation in performance. A promising mitigation strategy is watermarking, which embeds human-imperceptible yet detectable signals into generated images-enabling the identification of generated content. In this work, we introduce BitMark, a robust bitwise watermarking framework. Our method embeds a watermark directly at the bit level of the token stream during the image generation process. Our bitwise watermark subtly influences the bits to preserve visual fidelity and generation speed while remaining robust against a spectrum of removal techniques. Furthermore, it exhibits high radioactivity, i.e., when watermarked generated images are used to train another image generative model, this second model's outputs will also carry the watermark. The radioactive traces remain detectable even when only fine-tuning diffusion or image autoregressive models on images watermarked with our BitMark. Overall, our approach provides a principled step toward preventing model collapse in image generative models by enabling reliable detection of generated outputs. The code is available at https://github.com/sprintml/BitMark.
>
---
#### [replaced 040] GNSS-Inertial State Initialization Using Inter-Epoch Baseline Residuals
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.11534v2](https://arxiv.org/pdf/2506.11534v2)**

> **作者:** Samuel Cerezo; Javier Civera
>
> **备注:** 8 pages, 7 figures, accepted to RA-L
>
> **摘要:** Initializing the state of a sensorized platform can be challenging, as a limited set of measurements often provide low-informative constraints that are in addition highly non-linear. This may lead to poor initial estimates that may converge to local minima during subsequent non-linear optimization. We propose an adaptive GNSS-inertial initialization strategy that delays the incorporation of global GNSS constraints until they become sufficiently informative. In the initial stage, our method leverages inter-epoch baseline vector residuals between consecutive GNSS fixes to mitigate inertial drift. To determine when to activate global constraints, we introduce a general criterion based on the evolution of the Hessian matrix's singular values, effectively quantifying system observability. Experiments on EuRoC, GVINS and MARS-LVIG datasets show that our approach consistently outperforms the naive strategy of fusing all measurements from the outset, yielding more accurate and robust initializations.
>
---
#### [replaced 041] Language-Driven Object-Oriented Two-Stage Method for Scene Graph Anticipation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.05661v2](https://arxiv.org/pdf/2509.05661v2)**

> **作者:** Xiaomeng Zhu; Changwei Wang; Haozhe Wang; Xinyu Liu; Fangzhen Lin
>
> **摘要:** A scene graph is a structured representation of objects and their spatio-temporal relationships in dynamic scenes. Scene Graph Anticipation (SGA) involves predicting future scene graphs from video clips, enabling applications in intelligent surveillance and human-machine collaboration. While recent SGA approaches excel at leveraging visual evidence, long-horizon forecasting fundamentally depends on semantic priors and commonsense temporal regularities that are challenging to extract purely from visual features. To explicitly model these semantic dynamics, we propose Linguistic Scene Graph Anticipation (LSGA), a linguistic formulation of SGA that performs temporal relational reasoning over sequences of textualized scene graphs, with visual scene-graph detection handled by a modular front-end when operating on video. Building on this formulation, we introduce Object-Oriented Two-Stage Method (OOTSM), a language-based framework that anticipates object-set dynamics and forecasts object-centric relation trajectories with temporal consistency regularization, and we evaluate it on a dedicated benchmark constructed from Action Genome annotations. Extensive experiments show that compact fine-tuned language models with up to 3B parameters consistently outperform strong zero- and one-shot API baselines, including GPT-4o, GPT-4o-mini, and DeepSeek-V3, under matched textual inputs and context windows. When coupled with off-the-shelf visual scene-graph generators, the resulting multimodal system achieves substantial improvements on video-based SGA, boosting long-horizon mR@50 by up to 21.9\% over strong visual SGA baselines.
>
---
#### [replaced 042] Can VLMs Detect and Localize Fine-Grained AI-Edited Images?
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2505.15644v2](https://arxiv.org/pdf/2505.15644v2)**

> **作者:** Zhen Sun; Ziyi Zhang; Zeren Luo; Zhiyuan Zhong; Zeyang Sha; Tianshuo Cong; Zheng Li; Shiwen Cui; Weiqiang Wang; Jiaheng Wei; Xinlei He; Qi Li; Qian Wang
>
> **备注:** 14pages,19 figures
>
> **摘要:** Fine-grained detection and localization of localized image edits is crucial for assessing content authenticity, especially as modern diffusion models and image editors can produce highly realistic manipulations. However, this problem faces three key challenges: (1) most AIGC detectors produce only a global real-or-fake label without indicating where edits occur; (2) traditional computer vision methods for edit localization typically rely on costly pixel-level annotations; and (3) there is no large-scale, modern benchmark specifically targeting edited-image detection. To address these gaps, we develop an automated data-generation pipeline and construct FragFake, a large-scale benchmark of AI-edited images spanning multiple source datasets, diverse editing models, and several common edit types. Building on FragFake, we are the first to systematically study vision language models (VLMs) for edited-image classification and edited-region localization. Our experiments show that pretrained VLMs, including GPT4o, perform poorly on this task, whereas fine-tuned models such as Qwen2.5-VL achieve high accuracy and substantially higher object precision across all settings. We further explore GRPO-based RLVR training, which yields modest metric gains while improving the interpretability of model outputs. Ablation and transfer analyses reveal how data balancing, training size, LoRA rank, and training domain affect performance, and highlight both the potential and the limitations of cross-editor and cross-dataset generalization. We anticipate that this work will establish a solid foundation to facilitate and inspire subsequent research endeavors in the domain of multimodal content authenticity.
>
---
#### [replaced 043] SATORI-R1: Incentivizing Multimodal Reasoning through Explicit Visual Anchoring
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.19094v2](https://arxiv.org/pdf/2505.19094v2)**

> **作者:** Chuming Shen; Wei Wei; Xiaoye Qu; Yu Cheng
>
> **备注:** 21 pages, 8 figures
>
> **摘要:** DeepSeek-R1 has demonstrated powerful reasoning capabilities in the text domain through stable reinforcement learning (RL). Recently, in the multimodal domain, works have begun to directly apply RL to generate R1-like free-form reasoning for Visual Question Answering (VQA) tasks. However, multimodal tasks share an intrinsically different nature from textual tasks, which heavily rely on the understanding of the input image to solve the problem. Therefore, such free-form reasoning faces two critical limitations in the VQA task: (1) Extended reasoning chains diffuse visual focus away from task-critical regions, degrading answer accuracy. (2) Unverifiable intermediate steps amplify policy-gradient variance and computational costs overhead. To address these issues, in this paper, we introduce SATORI ($\textbf{S}patially$ $\textbf{A}nchored$ $\textbf{T}ask$ $\textbf{O}ptimization$ with $\textbf{R}e\textbf{I}nforcement$ Learning), which decomposes VQA into three verifiable stages, including global image captioning, region localization, and answer prediction, each supplying explicit reward signals. Furthermore, we also introduce VQA-Verify, a 12k dataset annotated with answer-aligned captions and bounding-boxes to facilitate training. Experiments demonstrate consistent performance improvements across seven VQA benchmarks, achieving up to $15.7\%$ improvement in accuracy in accuracy compared to the R1-like baseline. Our analysis of the attention map confirms enhanced focus on critical regions, which brings improvements in accuracy. Our code is available at https://github.com/justairr/SATORI-R1.
>
---
#### [replaced 044] Revisiting Data Challenges of Computational Pathology: A Pack-based Multiple Instance Learning Training Framework
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.20923v2](https://arxiv.org/pdf/2509.20923v2)**

> **作者:** Wenhao Tang; Heng Fang; Ge Wu; Xiang Li; Ming-Ming Cheng
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** Computational pathology (CPath) digitizes pathology slides into whole slide images (WSIs), enabling analysis for critical healthcare tasks such as cancer diagnosis and prognosis. However, WSIs possess extremely long sequence lengths (up to 200K), significant length variations (from 200 to 200K), and limited supervision. These extreme variations in sequence length lead to high data heterogeneity and redundancy. Conventional methods often compromise on training efficiency and optimization to preserve such heterogeneity under limited supervision. To comprehensively address these challenges, we propose a pack-based MIL framework. It packs multiple sampled, variable-length feature sequences into fixed-length ones, enabling batched training while preserving data heterogeneity. Moreover, we introduce a residual branch that composes discarded features from multiple slides into a hyperslide which is trained with tailored labels. It offers multi-slide supervision while mitigating feature loss from sampling. Meanwhile, an attention-driven downsampler is introduced to compress features in both branches to reduce redundancy. By alleviating these challenges, our approach achieves an accuracy improvement of up to 8% while using only 12% of the training time in the PANDA(UNI). Extensive experiments demonstrate that focusing data challenges in CPath holds significant potential in the era of foundation models. The code is https://github.com/FangHeng/PackMIL
>
---
#### [replaced 045] A Machine Learning-Driven Solution for Denoising Inertial Confinement Fusion Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.16717v2](https://arxiv.org/pdf/2511.16717v2)**

> **作者:** Asya Y. Akkus; Bradley T. Wolfe; Pinghan Chu; Chengkun Huang; Chris S. Campbell; Mariana Alvarado Alvarez; Petr Volegov; David Fittinghoff; Robert Reinovsky; Zhehui Wang
>
> **摘要:** Neutron imaging is essential for diagnosing and optimizing inertial confinement fusion implosions at the National Ignition Facility. Due to the required 10-micrometer resolution, however, neutron image require image reconstruction using iterative algorithms. For low-yield sources, the images may be degraded by various types of noise. Gaussian and Poisson noise often coexist within one image, obscuring fine details and blurring the edges where the source information is encoded. Traditional denoising techniques, such as filtering and thresholding, can inadvertently alter critical features or reshape the noise statistics, potentially impacting the ultimate fidelity of the iterative image reconstruction pipeline. However, recent advances in synthetic data production and machine learning have opened new opportunities to address these challenges. In this study, we present an unsupervised autoencoder with a Cohen-Daubechies- Feauveau (CDF 97) wavelet transform in the latent space, designed to suppress for mixed Gaussian-Poisson noise while preserving essential image features. The network successfully denoises neutron imaging data. Benchmarking against both simulated and experimental NIF datasets demonstrates that our approach achieves lower reconstruction error and superior edge preservation compared to conventional filtering methods such as Block-matching and 3D filtering (BM3D). By validating the effectiveness of unsupervised learning for denoising neutron images, this study establishes a critical first step towards fully AI-driven, end-to-end reconstruction frameworks for ICF diagnostics.
>
---
#### [replaced 046] From Pixels to Prose: Advancing Multi-Modal Language Models for Remote Sensing
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.05826v2](https://arxiv.org/pdf/2411.05826v2)**

> **作者:** Xintian Sun; Benji Peng; Charles Zhang; Fei Jin; Qian Niu; Junyu Liu; Keyu Chen; Ming Li; Pohsun Feng; Ziqian Bi; Ming Liu; Xinyuan Song; Yichao Zhang
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** Remote sensing has evolved from simple image acquisition to complex systems capable of integrating and processing visual and textual data. This review examines the development and application of multi-modal language models (MLLMs) in remote sensing, focusing on their ability to interpret and describe satellite imagery using natural language. We cover the technical underpinnings of MLLMs, including dual-encoder architectures, Transformer models, self-supervised and contrastive learning, and cross-modal integration. The unique challenges of remote sensing data--varying spatial resolutions, spectral richness, and temporal changes--are analyzed for their impact on MLLM performance. Key applications such as scene description, object detection, change detection, text-to-image retrieval, image-to-text generation, and visual question answering are discussed to demonstrate their relevance in environmental monitoring, urban planning, and disaster response. We review significant datasets and resources supporting the training and evaluation of these models. Challenges related to computational demands, scalability, data quality, and domain adaptation are highlighted. We conclude by proposing future research directions and technological advancements to further enhance MLLM utility in remote sensing.
>
---
#### [replaced 047] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12409v3](https://arxiv.org/pdf/2508.12409v3)**

> **作者:** Liang Lv; Di Wang; Jing Zhang; Lefei Zhang
>
> **备注:** AAAI 2026 Oral
>
> **摘要:** Semi-supervised semantic segmentation (S4) has advanced remote sensing (RS) analysis by leveraging unlabeled data through pseudo-labeling and consistency learning. However, existing S4 studies often rely on small-scale datasets and models, limiting their practical applicability. To address this, we propose S5, the first scalable framework for semi-supervised semantic segmentation in RS, which unlocks the potential of vast unlabeled Earth observation data typically underutilized due to costly pixel-level annotations. Built upon existing large-scale RS datasets, S5 introduces a data selection strategy that integrates entropy-based filtering and diversity expansion, resulting in the RS4P-1M dataset. Using this dataset, we systematically scale up S4 into a new pretraining paradigm, S4 pre-training (S4P), to pretrain RS foundation models (RSFMs) of varying sizes on this extensive corpus, significantly boosting their performance on land cover segmentation and object detection tasks. Furthermore, during fine-tuning, we incorporate a Mixture-of-Experts (MoE)-based multi-dataset fine-tuning approach, which enables efficient adaptation to multiple RS benchmarks with fewer parameters. This approach improves the generalization and versatility of RSFMs across diverse RS benchmarks. The resulting RSFMs achieve state-of-the-art performance across all benchmarks, underscoring the viability of scaling semi-supervised learning for RS applications. All datasets, code, and models will be released at https://github.com/MiliLab/S5
>
---
#### [replaced 048] All You Need for Object Detection: From Pixels, Points, and Prompts to Next-Gen Fusion and Multimodal LLMs/VLMs in Autonomous Vehicles
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.26641v2](https://arxiv.org/pdf/2510.26641v2)**

> **作者:** Sayed Pedram Haeri Boroujeni; Niloufar Mehrabi; Hazim Alzorgan; Mahlagha Fazeli; Abolfazl Razi
>
> **摘要:** Autonomous Vehicles (AVs) are transforming the future of transportation through advances in intelligent perception, decision-making, and control systems. However, their success is tied to one core capability, reliable object detection in complex and multimodal environments. While recent breakthroughs in Computer Vision (CV) and Artificial Intelligence (AI) have driven remarkable progress, the field still faces a critical challenge as knowledge remains fragmented across multimodal perception, contextual reasoning, and cooperative intelligence. This survey bridges that gap by delivering a forward-looking analysis of object detection in AVs, emphasizing emerging paradigms such as Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative AI rather than re-examining outdated techniques. We begin by systematically reviewing the fundamental spectrum of AV sensors (camera, ultrasonic, LiDAR, and Radar) and their fusion strategies, highlighting not only their capabilities and limitations in dynamic driving environments but also their potential to integrate with recent advances in LLM/VLM-driven perception frameworks. Next, we introduce a structured categorization of AV datasets that moves beyond simple collections, positioning ego-vehicle, infrastructure-based, and cooperative datasets (e.g., V2V, V2I, V2X, I2I), followed by a cross-analysis of data structures and characteristics. Ultimately, we analyze cutting-edge detection methodologies, ranging from 2D and 3D pipelines to hybrid sensor fusion, with particular attention to emerging transformer-driven approaches powered by Vision Transformers (ViTs), Large and Small Language Models (SLMs), and VLMs. By synthesizing these perspectives, our survey delivers a clear roadmap of current capabilities, open challenges, and future opportunities.
>
---
#### [replaced 049] Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19582v2](https://arxiv.org/pdf/2505.19582v2)**

> **作者:** Kaiqing Lin; Zhiyuan Yan; Ke-Yue Zhang; Li Hao; Yue Zhou; Yuzhen Lin; Weixiang Li; Taiping Yao; Shouhong Ding; Bin Li
>
> **摘要:** Securing personal identity against deepfake attacks is increasingly critical in the digital age, especially for celebrities and political figures whose faces are easily accessible and frequently targeted. Most existing deepfake detection methods focus on general-purpose scenarios and often ignore the valuable prior knowledge of known facial identities, e.g., "VIP individuals" whose authentic facial data are already available. In this paper, we propose \textbf{VIPGuard}, a unified multimodal framework designed to capture fine-grained and comprehensive facial representations of a given identity, compare them against potentially fake or similar-looking faces, and reason over these comparisons to make accurate and explainable predictions. Specifically, our framework consists of three main stages. First, fine-tune a multimodal large language model (MLLM) to learn detailed and structural facial attributes. Second, we perform identity-level discriminative learning to enable the model to distinguish subtle differences between highly similar faces, including real and fake variations. Finally, we introduce user-specific customization, where we model the unique characteristics of the target face identity and perform semantic reasoning via MLLM to enable personalized and explainable deepfake detection. Our framework shows clear advantages over previous detection works, where traditional detectors mainly rely on low-level visual cues and provide no human-understandable explanations, while other MLLM-based models often lack a detailed understanding of specific face identities. To facilitate the evaluation of our method, we built a comprehensive identity-aware benchmark called \textbf{VIPBench} for personalized deepfake detection, involving the latest 7 face-swapping and 7 entire face synthesis techniques for generation. The code is available at https://github.com/KQL11/VIPGuard .
>
---
#### [replaced 050] TabletopGen: Instance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01204v2](https://arxiv.org/pdf/2512.01204v2)**

> **作者:** Ziqian Wang; Yonghao He; Licheng Yang; Wei Zou; Hongxuan Ma; Liu Liu; Wei Sui; Yuxin Guo; Hu Su
>
> **备注:** Project page: https://d-robotics-ai-lab.github.io/TabletopGen.project/
>
> **摘要:** Generating high-fidelity, physically interactive 3D simulated tabletop scenes is essential for embodied AI--especially for robotic manipulation policy learning and data synthesis. However, current text- or image-driven 3D scene generation methods mainly focus on large-scale scenes, struggling to capture the high-density layouts and complex spatial relations that characterize tabletop scenes. To address these challenges, we propose TabletopGen, a training-free, fully automatic framework that generates diverse, instance-level interactive 3D tabletop scenes. TabletopGen accepts a reference image as input, which can be synthesized by a text-to-image model to enhance scene diversity. We then perform instance segmentation and completion on the reference to obtain per-instance images. Each instance is reconstructed into a 3D model followed by canonical coordinate alignment. The aligned 3D models then undergo pose and scale estimation before being assembled into a collision-free, simulation-ready tabletop scene. A key component of our framework is a novel pose and scale alignment approach that decouples the complex spatial reasoning into two stages: a Differentiable Rotation Optimizer for precise rotation recovery and a Top-view Spatial Alignment mechanism for robust translation and scale estimation, enabling accurate 3D reconstruction from 2D reference. Extensive experiments and user studies show that TabletopGen achieves state-of-the-art performance, markedly surpassing existing methods in visual fidelity, layout accuracy, and physical plausibility, capable of generating realistic tabletop scenes with rich stylistic and spatial diversity. Our code will be publicly available.
>
---
#### [replaced 051] MambaScope: Coarse-to-Fine Scoping for Efficient Vision Mamba
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.00647v2](https://arxiv.org/pdf/2512.00647v2)**

> **作者:** Shanhui Liu; Rui Xu; Yunke Wang
>
> **摘要:** Vision Mamba has emerged as a promising and efficient alternative to Vision Transformers, yet its efficiency remains fundamentally constrained by the number of input tokens. Existing token reduction approaches typically adopt token pruning or merging to reduce computation. However, they inherently lead to information loss as they discard or compress token representations. This problem is further exacerbated when the same fine-grained token processing is uniformly applied across all images regardless of visual complexity. We observe that not all inputs require fine-grained processing: simple images can be effectively handled at a coarse resolution, while only complex ones require refinement. Based on this insight, we propose MambaScope, an adaptive framework for efficient inference for Vision Mamba. MambaScope first performs coarse-grained inference by dividing the input image into large patches, significantly reducing token length and computation. When the model's prediction confidence is low, selected regions are re-processed at a finer resolution to recover essential visual details with minimal additional cost. This dynamic resolution assignment strategy allows MambaScope to allocate computation adaptively according to image complexity, achieving efficient processing without compromising accuracy. Experiments across various vision tasks demonstrate that MambaScope outperforms both the baseline Vision Mamba and state-of-the-art token reduction techniques in terms of accuracy and efficiency.
>
---
#### [replaced 052] A Novel Attention-Augmented Wavelet YOLO System for Real-time Brain Vessel Segmentation on Transcranial Color-coded Doppler
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13875v2](https://arxiv.org/pdf/2508.13875v2)**

> **作者:** Wenxuan Zhang; Shuai Li; Xinyi Wang; Yu Sun; Hongyu Kang; Pui Yuk Chryste Wan; Jing Qin; Yuanpeng Zhang; Yong-Ping Zheng; Sai-Kit Lam
>
> **摘要:** The Circle of Willis (CoW), vital for ensuring consistent blood flow to the brain, is closely linked to ischemic stroke. Accurate assessment of the CoW is important for identifying individuals at risk and guiding appropriate clinical management. Among existing imaging methods, Transcranial Color-coded Doppler (TCCD) offers unique advantages due to its radiation-free nature, affordability, and accessibility. However, reliable TCCD assessments depend heavily on operator expertise for identifying anatomical landmarks and performing accurate angle correction, which limits its widespread adoption. To address this challenge, we propose an AI-powered, real-time CoW auto-segmentation system capable of efficiently capturing cerebral arteries. No prior studies have explored AI-driven cerebrovascular segmentation using TCCD. In this work, we introduce a novel Attention-Augmented Wavelet YOLO (AAW-YOLO) network tailored for TCCD data, designed to provide real-time guidance for brain vessel segmentation in the CoW. We prospectively collected TCCD data comprising 738 annotated frames and 3,419 labeled artery instances to establish a high-quality dataset for model training and evaluation. The proposed AAW-YOLO demonstrated strong performance in segmenting both ipsilateral and contralateral CoW vessels, achieving an average Dice score of 0.901, IoU of 0.823, precision of 0.882, recall of 0.926, and mAP of 0.953, with a per-frame inference speed of 14.199 ms. This system offers a practical solution to reduce reliance on operator experience in TCCD-based cerebrovascular screening, with potential applications in routine clinical workflows and resource-constrained settings. Future research will explore bilateral modeling and larger-scale validation.
>
---
#### [replaced 053] InteractiveOmni: A Unified Omni-modal Model for Audio-Visual Multi-turn Dialogue
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13747v2](https://arxiv.org/pdf/2510.13747v2)**

> **作者:** Wenwen Tong; Hewei Guo; Dongchuan Ran; Jiangnan Chen; Jiefan Lu; Kaibin Wang; Keqiang Li; Xiaoxu Zhu; Jiakui Li; Kehan Li; Xueheng Li; Lumin Li; Chenxu Guo; Jiasheng Zhou; Jiandong Chen; Xianye Wu; Jiahao Wang; Silei Wu; Lei Chen; Hanming Deng; Yuxuan Song; Dinghao Zhou; Guiping Zhong; Ken Zheng; Shiyin Kang; Lewei Lu
>
> **摘要:** We introduce InteractiveOmni, a unified and open-source omni-modal large language model for audio-visual multi-turn interaction, ranging from 4B to 8B parameters, designed to lead the field of lightweight models by offering comprehensive omni-modal understanding and speech generation capabilities. To achieve this, we integrate the vision encoder, audio encoder, large language model, and speech decoder into a unified model for understanding and generation tasks. We design a multi-stage training strategy to ensure robust cross-modal capabilities, including pre-training for omni-modal understanding, followed by post-training with speech conversation and audio-visual interaction. To enable human-like long-term conversational ability, we meticulously curate a multi-turn training dataset that enhances the model's ability to handle complex and multi-turn interactions. To effectively evaluate the multi-turn memory and speech interaction capabilities, we construct the multi-modal multi-turn memory benchmark and the multi-turn speech interaction benchmark. Experiments demonstrate that InteractiveOmni significantly outperforms leading open-source models and provides a more intelligent multi-turn audio-visual experience, particularly in its long-term memory capabilities. Notably, InteractiveOmni-4B is comparable to the much larger model like Qwen2.5-Omni-7B on general benchmarks, and it can retain 97% of the performance of the InteractiveOmni-8B while utilizing only 50% of the model size. Achieving state-of-the-art results against similarly sized models across image, audio, video understanding, and speech generation tasks, InteractiveOmni is an accessible, open-source foundation for next-generation intelligent interactive systems.
>
---
#### [replaced 054] Two-Stage Vision Transformer for Image Restoration: Colorization Pretraining + Residual Upsampling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02512v2](https://arxiv.org/pdf/2512.02512v2)**

> **作者:** Aditya Chaudhary; Prachet Dev Singh; Ankit Jha
>
> **备注:** Accepted as a Tiny Paper at the 13th Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP 2025), IIT Mandi, India. 3 pages, 1 figure
>
> **摘要:** In computer vision, Single Image Super-Resolution (SISR) is still a difficult problem. We present ViT-SR, a new technique to improve the performance of a Vision Transformer (ViT) employing a two-stage training strategy. In our method, the model learns rich, generalizable visual representations from the data itself through a self-supervised pretraining phase on a colourization task. The pre-trained model is then adjusted for 4x super-resolution. By predicting the addition of a high-frequency residual image to an initial bicubic interpolation, this design simplifies residual learning. ViT-SR, trained and evaluated on the DIV2K benchmark dataset, achieves an impressive SSIM of 0.712 and PSNR of 22.90 dB. These results demonstrate the efficacy of our two-stage approach and highlight the potential of self-supervised pre-training for complex image restoration tasks. Further improvements may be possible with larger ViT architectures or alternative pretext tasks.
>
---
#### [replaced 055] Delving into Dynamic Scene Cue-Consistency for Robust 3D Multi-Object Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11323v2](https://arxiv.org/pdf/2508.11323v2)**

> **作者:** Haonan Zhang; Xinyao Wang; Boxi Wu; Tu Zheng; Wang Yunhua; Zheng Yang
>
> **摘要:** 3D multi-object tracking is a critical and challenging task in the field of autonomous driving. A common paradigm relies on modeling individual object motion, e.g., Kalman filters, to predict trajectories. While effective in simple scenarios, this approach often struggles in crowded environments or with inaccurate detections, as it overlooks the rich geometric relationships between objects. This highlights the need to leverage spatial cues. However, existing geometry-aware methods can be susceptible to interference from irrelevant objects, leading to ambiguous features and incorrect associations. To address this, we propose focusing on cue-consistency: identifying and matching stable spatial patterns over time. We introduce the Dynamic Scene Cue-Consistency Tracker (DSC-Track) to implement this principle. Firstly, we design a unified spatiotemporal encoder using Point Pair Features (PPF) to learn discriminative trajectory embeddings while suppressing interference. Secondly, our cue-consistency transformer module explicitly aligns consistent feature representations between historical tracks and current detections. Finally, a dynamic update mechanism preserves salient spatiotemporal information for stable online tracking. Extensive experiments on the nuScenes and Waymo Open Datasets validate the effectiveness and robustness of our approach. On the nuScenes benchmark, for instance, our method achieves state-of-the-art performance, reaching 73.2% and 70.3% AMOTA on the validation and test sets, respectively.
>
---
#### [replaced 056] Some Modalities are More Equal Than Others: Decoding and Architecting Multimodal Integration in MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22826v2](https://arxiv.org/pdf/2511.22826v2)**

> **作者:** Tianle Chen; Chaitanya Chakka; Arjun Reddy Akula; Xavier Thomas; Deepti Ghadiyaram
>
> **摘要:** Despite remarkable advancements in Multimodal Large Language Models (MLLMs), a fundamental question remains: are MLLMs robust to contradicting modalities? To rigorously study this, we introduce MMA-Bench comprising videos and tasks that probe a model's reliance on specific modalities. Using black-box and white-box interpretability techniques, we provide a critical analysis of the brittleness of both open- and closed-sourced MLLMs. We show that current MLLMs struggle under misaligned audio-visual pairs and simple misleading text, thereby lacking robust multi-modal reasoning. Building on these findings, we propose a modality alignment tuning strategy to teach the model when to prioritize, leverage, or ignore specific modality cues. Through extensive experiments and analysis, we show that our alignment tuning yields demonstrably stronger multimodal grounding. This work provides both interpretability tools and a clear path toward developing MLLMs with intrinsically reliable cross-modal reasoning. Code and dataset will be publicly available.
>
---
#### [replaced 057] Does Hearing Help Seeing? Investigating Audio-Video Joint Denoising for Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02457v2](https://arxiv.org/pdf/2512.02457v2)**

> **作者:** Jianzong Wu; Hao Lian; Dachao Hao; Ye Tian; Qingyu Shi; Biaolong Chen; Hao Jiang; Yunhai Tong
>
> **备注:** Project page at https://jianzongwu.github.io/projects/does-hearing-help-seeing/
>
> **摘要:** Recent audio-video generative systems suggest that coupling modalities benefits not only audio-video synchrony but also the video modality itself. We pose a fundamental question: Does audio-video joint denoising training improve video generation, even when we only care about video quality? To study this, we introduce a parameter-efficient Audio-Video Full DiT (AVFullDiT) architecture that leverages pre-trained text-to-video (T2V) and text-to-audio (T2A) modules for joint denoising. We train (i) a T2AV model with AVFullDiT and (ii) a T2V-only counterpart under identical settings. Our results provide the first systematic evidence that audio-video joint denoising can deliver more than synchrony. We observe consistent improvements on challenging subsets featuring large and object contact motions. We hypothesize that predicting audio acts as a privileged signal, encouraging the model to internalize causal relationships between visual events and their acoustic consequences (e.g., collision $\times$ impact sound), which in turn regularizes video dynamics. Our findings suggest that cross-modal co-training is a promising approach to developing stronger, more physically grounded world models. Code and dataset will be made publicly available.
>
---
#### [replaced 058] The Outline of Deception: Physical Adversarial Attacks on Traffic Signs Using Edge Patches
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00765v2](https://arxiv.org/pdf/2512.00765v2)**

> **作者:** Haojie Ji; Te Hu; Haowen Li; Long Jin; Chongshi Xin; Yuchi Yao; Jiarui Xiao
>
> **摘要:** Intelligent driving systems are vulnerable to physical adversarial attacks on traffic signs. These attacks can cause misclassification, leading to erroneous driving decisions that compromise road safety. Moreover, within V2X networks, such misinterpretations can propagate, inducing cascading failures that disrupt overall traffic flow and system stability. However, a key limitation of current physical attacks is their lack of stealth. Most methods apply perturbations to central regions of the sign, resulting in visually salient patterns that are easily detectable by human observers, thereby limiting their real-world practicality. This study proposes TESP-Attack, a novel stealth-aware adversarial patch method for traffic sign classification. Based on the observation that human visual attention primarily focuses on the central regions of traffic signs, we employ instance segmentation to generate edge-aligned masks that conform to the shape characteristics of the signs. A U-Net generator is utilized to craft adversarial patches, which are then optimized through color and texture constraints along with frequency domain analysis to achieve seamless integration with the background environment, resulting in highly effective visual concealment. The proposed method demonstrates outstanding attack success rates across traffic sign classification models with varied architectures, achieving over 90% under limited query budgets. It also exhibits strong cross-model transferability and maintains robust real-world performance that remains stable under varying angles and distances.
>
---
#### [replaced 059] Neural Radiance and Gaze Fields for Visual Attention Modeling in 3D Environments
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.07828v2](https://arxiv.org/pdf/2503.07828v2)**

> **作者:** Andrei Chubarau; Yinan Wang; James J. Clark
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** We introduce Neural Radiance and Gaze Fields (NeRGs), a novel approach for representing visual attention in complex environments. Much like how Neural Radiance Fields (NeRFs) perform novel view synthesis, NeRGs reconstruct gaze patterns from arbitrary viewpoints, implicitly mapping visual attention to 3D surfaces. We achieve this by augmenting a standard NeRF with an additional network that models local egocentric gaze probability density, conditioned on scene geometry and observer position. The output of a NeRG is a rendered view of the scene alongside a pixel-wise salience map representing the conditional probability that a given observer fixates on visible surfaces. Unlike prior methods, our system is lightweight and enables visualization of gaze fields at interactive framerates. Moreover, NeRGs allow the observer perspective to be decoupled from the rendering camera and correctly account for gaze occlusion due to intervening geometry. We demonstrate the effectiveness of NeRGs using head pose from skeleton tracking as a proxy for gaze, employing our proposed gaze probes to aggregate noisy rays into robust probability density targets for supervision.
>
---
#### [replaced 060] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism
- **分类: cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.01513v2](https://arxiv.org/pdf/2507.01513v2)**

> **作者:** Beitao Chen; Xinyu Lyu; Lianli Gao; Jingkuan Song; Heng Tao Shen
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.
>
---
#### [replaced 061] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中失败诊断与学习的难题，提出ViFailback框架，通过视觉符号提升故障诊断效率。构建了包含58,126个VQA对的真实世界失败数据集，并设立评估基准ViFailback-Bench。基于此，训练出ViFailback-8B模型，可生成可视化的纠正指导，实验证明其能有效帮助机器人从失败中恢复。**

- **链接: [https://arxiv.org/pdf/2512.02787v2](https://arxiv.org/pdf/2512.02787v2)**

> **作者:** Xianchao Zeng; Xinyu Zhou; Youcheng Li; Jiayou Shi; Tianle Li; Liangming Chen; Lei Ren; Yong-Lu Li
>
> **摘要:** Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic manipulation, yet they remain limited in failure diagnosis and learning from failures. Additionally, existing failure datasets are mostly generated programmatically in simulation, which limits their generalization to the real world. In light of these, we introduce ViFailback, a framework designed to diagnose robotic manipulation failures and provide both textual and visual correction guidance. Our framework utilizes explicit visual symbols to enhance annotation efficiency. We further release the ViFailback dataset, a large-scale collection of 58,126 Visual Question Answering (VQA) pairs along with their corresponding 5,202 real-world manipulation trajectories. Based on the dataset, we establish ViFailback-Bench, a benchmark of 11 fine-grained VQA tasks designed to assess the failure diagnosis and correction abilities of Vision-Language Models (VLMs), featuring ViFailback-Bench Lite for closed-ended and ViFailback-Bench Hard for open-ended evaluation. To demonstrate the effectiveness of our framework, we built the ViFailback-8B VLM, which not only achieves significant overall performance improvement on ViFailback-Bench but also generates visual symbols for corrective action guidance. Finally, by integrating ViFailback-8B with a VLA model, we conduct real-world robotic experiments demonstrating its ability to assist the VLA model in recovering from failures. Project Website: https://x1nyuzhou.github.io/vifailback.github.io/
>
---
#### [replaced 062] PixCell: A generative foundation model for digital histopathology images
- **分类: eess.IV; cs.CV; q-bio.QM**

- **链接: [https://arxiv.org/pdf/2506.05127v2](https://arxiv.org/pdf/2506.05127v2)**

> **作者:** Srikar Yellapragada; Alexandros Graikos; Zilinghan Li; Kostas Triaridis; Varun Belagali; Tarak Nath Nandi; Karen Bai; Beatrice S. Knudsen; Tahsin Kurc; Rajarsi R. Gupta; Prateek Prasanna; Ravi K Madduri; Joel Saltz; Dimitris Samaras
>
> **备注:** Project page - https://histodiffusion.github.io/docs/projects/pixcell
>
> **摘要:** The digitization of histology slides has revolutionized pathology, providing massive datasets for cancer diagnosis and research. Self-supervised and vision-language models have been shown to effectively mine large pathology datasets to learn discriminative representations. On the other hand, there are unique problems in pathology, such as annotated data scarcity, privacy regulations in data sharing, and inherently generative tasks like virtual staining. Generative models, capable of synthesizing realistic and diverse images, present a compelling solution to address these problems through image synthesis. We introduce PixCell, the first generative foundation model for histopathology images. PixCell is a diffusion model trained on PanCan-30M, a large, diverse dataset derived from 69,184 H&E-stained whole slide images of various cancer types. We employ a progressive training strategy and a self-supervision-based conditioning that allows us to scale up training without any human-annotated data. By conditioning on real slides, the synthetic images capture the properties of the real data and can be used as data augmentation for small-scale datasets to boost classification performance. We prove the foundational versatility of PixCell by applying it to two generative downstream tasks: privacy-preserving synthetic data generation and virtual IHC staining. PixCell's high-fidelity conditional generation enables institutions to use their private data to synthesize highly realistic, site-specific surrogate images that can be shared in place of raw patient data. Furthermore, using datasets of roughly paired H&E-IHC tiles, we learn to translate PixCell's conditioning from H&E to multiple IHC stains, allowing the generation of IHC images from H&E inputs. Our trained models are publicly released to accelerate research in computational pathology.
>
---
#### [replaced 063] On Efficient Variants of Segment Anything Model: A Survey
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.04960v4](https://arxiv.org/pdf/2410.04960v4)**

> **作者:** Xiaorui Sun; Jun Liu; Heng Tao Shen; Xiaofeng Zhu; Ping Hu
>
> **摘要:** The Segment Anything Model (SAM) is a foundational model for image segmentation tasks, known for its strong generalization across diverse applications. However, its impressive performance comes with significant computational and resource demands, making it challenging to deploy in resource-limited environments such as edge devices. To address this, a variety of SAM variants have been proposed to enhance efficiency while keeping accuracy. This survey provides the first comprehensive review of these efficient SAM variants. We begin by exploring the motivations driving this research. We then present core techniques used in SAM and model acceleration. This is followed by a detailed exploration of SAM acceleration strategies, categorized by approach, and a discussion of several future research directions. Finally, we offer a unified and extensive evaluation of these methods across various hardware, assessing their efficiency and accuracy on representative benchmarks, and providing a clear comparison of their overall performance.
>
---
#### [replaced 064] 3D and 4D World Modeling: A Survey
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦3D/4D世界建模任务，针对现有研究忽视原生3D/4D数据及缺乏统一定义与分类的问题，提出涵盖视频、占据网格和LiDAR的系统性分类，梳理数据集、评估指标与应用，总结挑战与方向，为该领域提供首个全面综述。**

- **链接: [https://arxiv.org/pdf/2509.07996v3](https://arxiv.org/pdf/2509.07996v3)**

> **作者:** Lingdong Kong; Wesley Yang; Jianbiao Mei; Youquan Liu; Ao Liang; Dekai Zhu; Dongyue Lu; Wei Yin; Xiaotao Hu; Mingkai Jia; Junyuan Deng; Kaiwen Zhang; Yang Wu; Tianyi Yan; Shenyuan Gao; Song Wang; Linfeng Li; Liang Pan; Yong Liu; Jianke Zhu; Wei Tsang Ooi; Steven C. H. Hoi; Ziwei Liu
>
> **备注:** Survey; 50 pages, 10 figures, 14 tables; GitHub Repo at https://github.com/worldbench/awesome-3d-4d-world-models
>
> **摘要:** World modeling has become a cornerstone in AI research, enabling agents to understand, represent, and predict the dynamic environments they inhabit. While prior work largely emphasizes generative methods for 2D image and video data, they overlook the rapidly growing body of work that leverages native 3D and 4D representations such as RGB-D imagery, occupancy grids, and LiDAR point clouds for large-scale scene modeling. At the same time, the absence of a standardized definition and taxonomy for ``world models'' has led to fragmented and sometimes inconsistent claims in the literature. This survey addresses these gaps by presenting the first comprehensive review explicitly dedicated to 3D and 4D world modeling and generation. We establish precise definitions, introduce a structured taxonomy spanning video-based (VideoGen), occupancy-based (OccGen), and LiDAR-based (LiDARGen) approaches, and systematically summarize datasets and evaluation metrics tailored to 3D/4D settings. We further discuss practical applications, identify open challenges, and highlight promising research directions, aiming to provide a coherent and foundational reference for advancing the field. A systematic summary of existing literature is available at https://github.com/worldbench/awesome-3d-4d-world-models
>
---
#### [replaced 065] GT23D-Bench: A Comprehensive General Text-to-3D Generation Benchmark
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.09997v2](https://arxiv.org/pdf/2412.09997v2)**

> **作者:** Xiao Cai; Sitong Su; Jingkuan Song; Pengpeng Zeng; Ji Zhang; Qinhong Du; Mengqi Li; Heng Tao Shen; Lianli Gao
>
> **摘要:** Text-to-3D (T23D) generation has emerged as a crucial visual generation task, aiming at synthesizing 3D content from textual descriptions. Studies of this task are currently shifting from per-scene T23D, which requires optimization of the model for every content generated, to General T23D (GT23D), which requires only one pre-trained model to generate different content without re-optimization, for more generalized and efficient 3D generation. Despite notable advancements, GT23D is severely bottlenecked by two interconnected challenges: the lack of high-quality, large-scale training data and the prevalence of evaluation metrics that overlook intrinsic 3D properties. Existing datasets often suffer from incomplete annotations, noisy organization, and inconsistent quality, while current evaluations rely heavily on 2D image-text similarity or scoring, failing to thoroughly assess 3D geometric integrity and semantic relevance. To address these fundamental gaps, we introduce GT23D-Bench, the first comprehensive benchmark specifically designed for GT23D training and evaluation. We first construct a high-quality dataset of 400K 3D assets, featuring diverse visual annotations (70M+ visual samples) and multi-granularity hierarchical captions (1M+ descriptions) to foster robust semantic learning. Second, we propose a comprehensive evaluation suite with 10 metrics assessing both text-3D alignment and 3D visual quality at multiple levels. Crucially, we demonstrate through rigorous experiments that our proposed metrics exhibit significantly higher correlation with human judgment compared to existing methods. Our in-depth analysis of eight leading GT23D models using this benchmark provides the community with critical insights into current model capabilities and their shared failure modes. GT23D-Bench will be publicly available to facilitate rigorous and reproducible research.
>
---
#### [replaced 066] DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03000v2](https://arxiv.org/pdf/2512.03000v2)**

> **作者:** Kairun Wen; Yuzhi Huang; Runyu Chen; Hui Zheng; Yunlong Lin; Panwang Pan; Chenxin Li; Wenyan Cong; Jian Zhang; Junbin Lu; Chenguo Lin; Dilin Wang; Zhicheng Yan; Hongyu Xu; Justin Theiss; Yue Huang; Xinghao Ding; Rakesh Ranjan; Zhiwen Fan
>
> **摘要:** Understanding the dynamic physical world, characterized by its evolving 3D structure, real-world motion, and semantic content with textual descriptions, is crucial for human-agent interaction and enables embodied agents to perceive and act within real environments with human-like capabilities. However, existing datasets are often derived from limited simulators or utilize traditional Structurefrom-Motion for up-to-scale annotation and offer limited descriptive captioning, which restricts the capacity of foundation models to accurately interpret real-world dynamics from monocular videos, commonly sourced from the internet. To bridge these gaps, we introduce DynamicVerse, a physical-scale, multimodal 4D world modeling framework for dynamic real-world video. We employ large vision, geometric, and multimodal models to interpret metric-scale static geometry, real-world dynamic motion, instance-level masks, and holistic descriptive captions. By integrating window-based Bundle Adjustment with global optimization, our method converts long real-world video sequences into a comprehensive 4D multimodal format. DynamicVerse delivers a large-scale dataset consisting of 100K+ videos with 800K+ annotated masks and 10M+ frames from internet videos. Experimental evaluations on three benchmark tasks, namely video depth estimation, camera pose estimation, and camera intrinsics estimation, demonstrate that our 4D modeling achieves superior performance in capturing physical-scale measurements with greater global accuracy than existing methods.
>
---
#### [replaced 067] LoRA Patching: Exposing the Fragility of Proactive Defenses against Deepfakes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.03747v2](https://arxiv.org/pdf/2510.03747v2)**

> **作者:** Zuomin Qu; Yimao Guo; Qianyue Hu; Wei Lu
>
> **摘要:** Deepfakes pose significant societal risks, motivating the development of proactive defenses that embed adversarial perturbations in facial images to prevent manipulation. However, in this paper, we show that these preemptive defenses often lack robustness and reliability. We propose a novel approach, Low-Rank Adaptation (LoRA) patching, which injects a plug-and-play LoRA patch into Deepfake generators to bypass state-of-the-art defenses. A learnable gating mechanism adaptively controls the effect of the LoRA patch and prevents gradient explosions during fine-tuning. We also introduce a Multi-Modal Feature Alignment (MMFA) loss, encouraging the features of adversarial outputs to align with those of the desired outputs at the semantic level. Beyond bypassing, we present defensive LoRA patching, embedding visible warnings in the outputs as a complementary solution to mitigate this newly identified security vulnerability. With only 1,000 facial examples and a single epoch of fine-tuning, LoRA patching successfully defeats multiple proactive defenses. These results reveal a critical weakness in current paradigms and underscore the need for more robust Deepfake defense strategies. Our code is available at https://github.com/ZOMIN28/LoRA-Patching.
>
---
#### [replaced 068] Universal Multi-Domain Translation via Diffusion Routers
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.03252v2](https://arxiv.org/pdf/2510.03252v2)**

> **作者:** Duc Kieu; Kien Do; Tuan Hoang; Thao Minh Le; Tung Kieu; Dang Nguyen; Thin Nguyen
>
> **摘要:** Multi-domain translation (MDT) aims to learn translations between multiple domains, yet existing approaches either require fully aligned tuples or can only handle domain pairs seen in training, limiting their practicality and excluding many cross-domain mappings. We introduce universal MDT (UMDT), a generalization of MDT that seeks to translate between any pair of $K$ domains using only $K-1$ paired datasets with a central domain. To tackle this problem, we propose Diffusion Router (DR), a unified diffusion-based framework that models all central$\leftrightarrow$non-central translations with a single noise predictor conditioned on the source and target domain labels. DR enables indirect non-central translations by routing through the central domain. We further introduce a novel scalable learning strategy with a variational-bound objective and an efficient Tweedie refinement procedure to support direct non-central mappings. Through evaluation on three large-scale UMDT benchmarks, DR achieves state-of-the-art results for both indirect and direct translations, while lowering sampling cost and unlocking novel tasks such as sketch$\leftrightarrow$segmentation. These results establish DR as a scalable and versatile framework for universal translation across multiple domains.
>
---
#### [replaced 069] MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [https://arxiv.org/pdf/2512.02906v2](https://arxiv.org/pdf/2512.02906v2)**

> **作者:** Fan Yang; Kaihao Zhang
>
> **摘要:** Understanding high-resolution images remains a significant challenge for multimodal large language models (MLLMs). Recent study address this issue by dividing the image into smaller crops and computing the semantic similarity between each crop and a query using a pretrained retrieval-augmented generation (RAG) model. The most relevant crops are then selected to localize the target object and suppress irrelevant information. However, such crop-based processing can fragment complete objects across multiple crops, thereby disrupting the computation of semantic similarity. In our experiments, we find that image crops of objects with different sizes are better handled at different resolutions. Based on this observation, we propose Multi-resolution Retrieval-Detection (MRD), a training-free framework for high-resolution image understanding. To address the issue of semantic similarity bias caused by objects being split across different image crops, we propose a multi-resolution semantic fusion method, which integrates semantic similarity maps obtained at different resolutions to produce more accurate semantic information and preserve the integrity of target objects. Furthermore, to achieve direct localization of target objects at a global scale, we introduce an open-vocalbulary object detection (OVD) model that identifies object regions using a sliding-window approach.Experiments on high-resolution image understanding benchmarks using different MLLMs demonstrate the effectiveness of our approach.
>
---
#### [replaced 070] MagicView: Multi-View Consistent Identity Customization via Priors-Guided In-Context Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00293v2](https://arxiv.org/pdf/2511.00293v2)**

> **作者:** Hengjia Li; Jianjin Xu; Keli Cheng; Lei Wang; Ning Bi; Boxi Wu; Fernando De la Torre; Deng Cai
>
> **摘要:** Recent advances in personalized generative models have demonstrated impressive capabilities in producing identity-consistent images of the same individual across diverse scenes. However, most existing methods lack explicit viewpoint control and fail to ensure multi-view consistency of generated identities. To address this limitation, we present MagicView, a lightweight adaptation framework that equips existing generative models with multi-view generation capability through 3D priors-guided in-context learning. While prior studies have shown that in-context learning preserves identity consistency across grid samples, its effectiveness in multi-view settings remains unexplored. Building upon this insight, we conduct an in-depth analysis of the multi-view in-context learning ability, and design a conditioning architecture that leverages 3D priors to activate this capability for multi-view consistent identity customization. On the other hand, acquiring robust multi-view capability typically requires large-scale multi-dimensional datasets, which makes incorporating multi-view contextual learning under limited data regimes prone to textual controllability degradation. To address this issue, we introduce a novel Semantic Correspondence Alignment loss, which effectively preserves semantic alignment while maintaining multi-view consistency. Extensive experiments demonstrate that MagicView substantially outperforms recent baselines in multi-view consistency, text alignment, identity similarity, and visual quality, achieving strong results with only 100 multi-view training samples.
>
---
#### [replaced 071] A$^2$LC: Active and Automated Label Correction for Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.11599v2](https://arxiv.org/pdf/2506.11599v2)**

> **作者:** Youjin Jeon; Kyusik Cho; Suhan Woo; Euntai Kim
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Active Label Correction (ALC) has emerged as a promising solution to the high cost and error-prone nature of manual pixel-wise annotation in semantic segmentation, by actively identifying and correcting mislabeled data. Although recent work has improved correction efficiency by generating pseudo-labels using foundation models, substantial inefficiencies still remain. In this paper, we introduce A$^2$LC, an Active and Automated Label Correction framework for semantic segmentation, where manual and automatic correction stages operate in a cascaded manner. Specifically, the automatic correction stage leverages human feedback to extend label corrections beyond the queried samples, thereby maximizing cost efficiency. In addition, we introduce an adaptively balanced acquisition function that emphasizes underrepresented tail classes, working in strong synergy with the automatic correction stage. Extensive experiments on Cityscapes and PASCAL VOC 2012 demonstrate that A$^2$LC significantly outperforms previous state-of-the-art methods. Notably, A$^2$LC exhibits high efficiency by outperforming previous methods with only 20% of their budget, and shows strong effectiveness by achieving a 27.23% performance gain under the same budget on Cityscapes.
>
---
#### [replaced 072] Defense That Attacks: How Robust Models Become Better Attackers
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.02830v2](https://arxiv.org/pdf/2512.02830v2)**

> **作者:** Mohamed Awad; Mahmoud Akrm; Walid Gomaa
>
> **摘要:** Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples.
>
---
#### [replaced 073] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.06424v2](https://arxiv.org/pdf/2406.06424v2)**

> **作者:** Jiwoo Hong; Sayak Paul; Noah Lee; Kashif Rasul; James Thorne; Jongheon Jeong
>
> **备注:** Accepted to AAAI 2026 Main Technical Track
>
> **摘要:** Modern preference alignment methods, such as DPO, rely on divergence regularization to a reference model for training stability-but this creates a fundamental problem we call "reference mismatch." In this paper, we investigate the negative impacts of reference mismatch in aligning text-to-image (T2I) diffusion models, showing that larger reference mismatch hinders effective adaptation given the same amount of data, e.g., as when learning new artistic styles, or personalizing to specific objects. We demonstrate this phenomenon across text-to-image (T2I) diffusion models and introduce margin-aware preference optimization (MaPO), a reference-agnostic approach that breaks free from this constraint. By directly optimizing the likelihood margin between preferred and dispreferred outputs under the Bradley-Terry model without anchoring to a reference, MaPO transforms diverse T2I tasks into unified pairwise preference optimization. We validate MaPO's versatility across five challenging domains: (1) safe generation, (2) style adaptation, (3) cultural representation, (4) personalization, and (5) general preference alignment. Our results reveal that MaPO's advantage grows dramatically with reference mismatch severity, outperforming both DPO and specialized methods like DreamBooth while reducing training time by 15%. MaPO thus emerges as a versatile and memory-efficient method for generic T2I adaptation tasks.
>
---
#### [replaced 074] Look, Recite, Then Answer: Enhancing VLM Performance via Self-Generated Knowledge Hints
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.00882v3](https://arxiv.org/pdf/2512.00882v3)**

> **作者:** Xisheng Feng
>
> **摘要:** Vision-Language Models (VLMs) exhibit significant performance plateaus in specialized domains like precision agriculture, primarily due to "Reasoning-Driven Hallucination" where linguistic priors override visual perception. A key bottleneck is the "Modality Gap": visual embeddings fail to reliably activate the fine-grained expert knowledge already encoded in model parameters. We propose "Look, Recite, Then Answer," a parameter-efficient framework that enhances VLMs via self-generated knowledge hints while keeping backbone models frozen. The framework decouples inference into three stages: (1) Look generates objective visual descriptions and candidate sets; (2) Recite employs a lightweight 1.7B router to transform visual cues into targeted queries that trigger candidate-specific parametric knowledge; (3) Answer performs parallel evidence alignment between descriptions and recited knowledge to select the most consistent label. On AgroBench, our method achieves state-of-the-art results, improving Weed Identification accuracy by 23.52% over Qwen2-VL-72B and surpassing GPT-4o without external search overhead. This modular design mitigates hallucinations by transforming passive perception into active, controllable knowledge retrieval
>
---
#### [replaced 075] A Tractable Two-Step Linear Mixing Model Solved with Second-Order Optimization for Spectral Unmixing under Variability
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2502.17212v3](https://arxiv.org/pdf/2502.17212v3)**

> **作者:** Xander Haijen; Bikram Koirala; Xuanwen Tao; Paul Scheunders
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In this paper, we propose a Two-Step Linear Mixing Model (2LMM) that bridges the gap between model complexity and computational tractability. The model achieves this by introducing two distinct scaling steps: an endmember scaling step across the image, and another for pixel-wise scaling. We show that this model leads to only a mildly non-convex optimization problem, which we solve with an optimization algorithm that incorporates second-order information. To the authors' knowledge, this work represents the first application of second-order optimization techniques to solve a spectral unmixing problem that models endmember variability. Our method is highly robust, as it requires virtually no hyperparameter tuning and can therefore be used easily and quickly in a wide range of unmixing tasks. We show through extensive experiments on both simulated and real data that the new model is competitive and in some cases superior to the state of the art in unmixing. The model also performs very well in challenging scenarios, such as blind unmixing.
>
---
#### [replaced 076] UniEdit-I: Training-free Image Editing for Unified VLM via Iterative Understanding, Editing and Verifying
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03142v2](https://arxiv.org/pdf/2508.03142v2)**

> **作者:** Chengyu Bai; Jintao Chen; Xiang Bai; Yilong Chen; Qi She; Ming Lu; Shanghang Zhang
>
> **摘要:** While Unified Vision-Language Models promise to synergistically combine the high-level semantic understanding of vision-language models with the generative fidelity of diffusion models, current editing methodologies remain fundamentally decoupled and open loop performing static, pre-defined transformations without dynamic feedback between semantic interpretation and visual generation. A central limitation stems from the representation gap: understanding typically leverages high-level, language aligned encoders, whereas generation relies on low level, pixel-space autoencoders, resulting in misaligned feature spaces. To bridge this gap, Recent advances such as Representation Autoencoders and BLIP3-o advocate performing diffusion-based modeling directly in high level features from pretrained semantic encoders. We find editing in the semantic latent space modifies conceptual representations rather than pixels, ensuring intermediates that are both semantically coherent and visually plausible. Building on this insight, We propose UniEdit-I, the first training-free, closed-loop image editing framework that operates entirely within the semantic latent space of a unified VLM by introducing an Understanding-Editing-Verifying (UEV) loop, By transforming the VLM from a posthoc evaluator into an in-process conductor, UniEdit-I establishes the first semantics-driven, self-correcting closed-loop image editing pipeline. Evaluated on GEdit-Bench, UniEdit-I achieves state of the art performance without any fine tuning or architectural modifications, and even surpasses several largescale pretrained editors.
>
---
#### [replaced 077] VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对多模态安全评估中忽视视觉与语言联合理解的问题，提出VLSU框架，通过细粒度分类与组合分析，构建包含8,187样本的基准数据集。研究发现当前模型在联合推理时性能大幅下降，存在严重组合推理缺失与安全边界误判问题，揭示了多模态安全模型的关键缺陷。**

- **链接: [https://arxiv.org/pdf/2510.18214v2](https://arxiv.org/pdf/2510.18214v2)**

> **作者:** Shruti Palaskar; Leon Gatys; Mona Abdelrahman; Mar Jacobo; Larry Lindsey; Rutika Moharir; Gunnar Lund; Yang Xu; Navid Shiee; Jeffrey Bigham; Charles Maalouf; Joseph Yitan Cheng
>
> **备注:** 10 pages, 5 figures, 4 tables, detailed appendix. Under review
>
> **摘要:** Safety evaluation of multimodal foundation models often treats vision and language inputs separately, missing risks from joint interpretation where benign content becomes harmful in combination. Existing approaches also fail to distinguish clearly unsafe content from borderline cases, leading to problematic over-blocking or under-refusal of genuinely harmful content. We present Vision Language Safety Understanding (VLSU), a comprehensive framework to systematically evaluate multimodal safety through fine-grained severity classification and combinatorial analysis across 17 distinct safety patterns. Using a multi-stage pipeline with real-world images and human annotation, we construct a large-scale benchmark of 8,187 samples spanning 15 harm categories. Our evaluation of eleven state-of-the-art models reveals systematic joint understanding failures: while models achieve 90%-plus accuracy on clear unimodal safety signals, performance degrades substantially to 20-55% when joint image-text reasoning is required to determine the safety label. Most critically, 34% of errors in joint image-text safety classification occur despite correct classification of the individual modalities, further demonstrating absent compositional reasoning capabilities. Additionally, we find that models struggle to balance refusing unsafe content while still responding to borderline cases that deserve engagement. For example, we find that instruction framing can reduce the over-blocking rate on borderline content from 62.4% to 10.4% in Gemini-1.5, but only at the cost of under-refusing on unsafe content with refusal rate dropping from 90.8% to 53.9%. Overall, our framework exposes weaknesses in joint image-text understanding and alignment gaps in current models, and provides a critical test bed to enable the next milestones in research on robust vision-language safety.
>
---
#### [replaced 078] FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.08136v2](https://arxiv.org/pdf/2508.08136v2)**

> **作者:** Yitong Yang; Yinglin Wang; Changshuo Wang; Huajie Wang; Shuting He
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** The success of 3DGS in generative and editing applications has sparked growing interest in 3DGS-based style transfer. However, current methods still face two major challenges: (1) multi-view inconsistency often leads to style conflicts, resulting in appearance smoothing and distortion; and (2) heavy reliance on VGG features, which struggle to disentangle style and content from style images, often causing content leakage and excessive stylization. To tackle these issues, we introduce \textbf{FantasyStyle}, a 3DGS-based style transfer framework, and the first to rely entirely on diffusion model distillation. It comprises two key components: (1) \textbf{Multi-View Frequency Consistency}. We enhance cross-view consistency by applying a 3D filter to multi-view noisy latent, selectively reducing low-frequency components to mitigate stylized prior conflicts. (2) \textbf{Controllable Stylized Distillation}. To suppress content leakage from style images, we introduce negative guidance to exclude undesired content. In addition, we identify the limitations of Score Distillation Sampling and Delta Denoising Score in 3D style transfer and remove the reconstruction term accordingly. Building on these insights, we propose a controllable stylized distillation that leverages negative guidance to more effectively optimize the 3D Gaussians. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art approaches, achieving higher stylization quality and visual realism across various scenes and styles. The code is available at https://github.com/yangyt46/FantasyStyle.
>
---
#### [replaced 079] AugMapNet: Improving Spatial Latent Structure via BEV Grid Augmentation for Enhanced Vectorized Online HD Map Construction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对自动驾驶中矢量高精地图的实时构建任务，提出AugMapNet模型。通过引入BEV特征网格增强，提升隐空间结构化程度，融合向量解码与密集空间监督，显著改善地图预测精度，尤其在大范围场景下表现优异。**

- **链接: [https://arxiv.org/pdf/2503.13430v2](https://arxiv.org/pdf/2503.13430v2)**

> **作者:** Thomas Monninger; Md Zafar Anwar; Stanislaw Antol; Steffen Staab; Sihao Ding
>
> **备注:** Accepted to 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2026)
>
> **摘要:** Autonomous driving requires understanding infrastructure elements, such as lanes and crosswalks. To navigate safely, this understanding must be derived from sensor data in real-time and needs to be represented in vectorized form. Learned Bird's-Eye View (BEV) encoders are commonly used to combine a set of camera images from multiple views into one joint latent BEV grid. Traditionally, from this latent space, an intermediate raster map is predicted, providing dense spatial supervision but requiring post-processing into the desired vectorized form. More recent models directly derive infrastructure elements as polylines using vectorized map decoders, providing instance-level information. Our approach, Augmentation Map Network (AugMapNet), proposes latent BEV feature grid augmentation, a novel technique that significantly enhances the latent BEV representation. AugMapNet combines vector decoding and dense spatial supervision more effectively than existing architectures while remaining easy to integrate compared to other hybrid approaches. It additionally benefits from extra processing on its latent BEV features. Experiments on nuScenes and Argoverse2 datasets demonstrate significant improvements on vectorized map prediction of up to 13.3% over the StreamMapNet baseline on 60 m range and greater improvements on larger ranges. We confirm transferability by applying our method to another baseline, SQD-MapNet, and find similar improvements. A detailed analysis of the latent BEV grid confirms a more structured latent space of AugMapNet and shows the value of our novel concept beyond pure performance improvement. The code can be found at https://github.com/tmonnin/augmapnet
>
---
#### [replaced 080] HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶仿真中视点变化大时视觉失真、几何不一致的问题，提出HybridWorldSim框架，融合神经重建与生成模型，实现高保真、可控的动态场景仿真。构建MIRROR数据集用于基准测试，实验表明其显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.22187v3](https://arxiv.org/pdf/2511.22187v3)**

> **作者:** Qiang Li; Yingwenqi Jiang; Tuoxi Li; Duyu Chen; Xiang Feng; Yucheng Ao; Shangyue Liu; Xingchen Yu; Youcheng Cai; Yumeng Liu; Yuexin Ma; Xin Hu; Li Liu; Yu Zhang; Linkun Xu; Bingtao Gao; Xueyuan Wang; Shuchang Zhou; Xianming Liu; Ligang Liu
>
> **备注:** Project page: https://hybridworldsim.github.io/
>
> **摘要:** Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.
>
---
#### [replaced 081] Context Cascade Compression: Exploring the Upper Limits of Text Compression
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对大模型长文本处理中的计算与内存瓶颈，提出上下文级联压缩C3方法。通过两级大模型协作，将长文本压缩至极短潜在标记（如32/64），在20倍压缩比下实现98%解码准确率，40倍时仍保持93%，显著优于现有光学压缩方案，验证了纯文本压缩的高效性与上限潜力。**

- **链接: [https://arxiv.org/pdf/2511.15244v2](https://arxiv.org/pdf/2511.15244v2)**

> **作者:** Fanfan Liu; Haibo Qiu
>
> **摘要:** Million-level token inputs in long-context tasks pose significant computational and memory challenges for Large Language Models (LLMs). Recently, DeepSeek-OCR conducted research into the feasibility of Contexts Optical Compression and achieved preliminary results. Inspired by this, we introduce Context Cascade Compression C3 to explore the upper limits of text compression. Our method cascades two LLMs of different sizes to handle the compression and decoding tasks. Specifically, a small LLM, acting as the first stage, performs text compression by condensing a long context into a set of latent tokens (e.g., 32 or 64 in length), achieving a high ratio of text tokens to latent tokens. A large LLM, as the second stage, then executes the decoding task on this compressed context. Experiments show that at a 20x compression ratio (where the number of text tokens is 20 times the number of latent tokens), our model achieves 98% decoding accuracy, compared to approximately 60% for DeepSeek-OCR. When we further increase the compression ratio to 40x, the accuracy is maintained at around 93%. This indicates that in the domain of context compression, C3 Compression demonstrates superior performance and feasibility over optical character compression. C3 uses a simpler, pure-text pipeline that ignores factors like layout, color, and information loss from a visual encoder. This also suggests a potential upper bound for compression ratios in future work on optical character compression, OCR, and related fields. Codes and model weights are publicly accessible at https://github.com/liufanfanlff/C3-Context-Cascade-Compression
>
---
#### [replaced 082] NVRC: Neural Video Representation Compression
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2409.07414v2](https://arxiv.org/pdf/2409.07414v2)**

> **作者:** Ho Man Kwan; Ge Gao; Fan Zhang; Andrew Gower; David Bull
>
> **摘要:** Recent advances in implicit neural representation (INR)-based video coding have demonstrated its potential to compete with both conventional and other learning-based approaches. With INR methods, a neural network is trained to overfit a video sequence, with its parameters compressed to obtain a compact representation of the video content. However, although promising results have been achieved, the best INR-based methods are still out-performed by the latest standard codecs, such as VVC VTM, partially due to the simple model compression techniques employed. In this paper, rather than focusing on representation architectures as in many existing works, we propose a novel INR-based video compression framework, Neural Video Representation Compression (NVRC), targeting compression of the representation. Based on the novel entropy coding and quantization models proposed, NVRC, for the first time, is able to optimize an INR-based video codec in a fully end-to-end manner. To further minimize the additional bitrate overhead introduced by the entropy models, we have also proposed a new model compression framework for coding all the network, quantization and entropy model parameters hierarchically. Our experiments show that NVRC outperforms many conventional and learning-based benchmark codecs, with a 24% average coding gain over VVC VTM (Random Access) on the UVG dataset, measured in PSNR. As far as we are aware, this is the first time an INR-based video codec achieving such performance. The implementation of NVRC will be released.
>
---
